"""
NVENC video encoding with per-frame QP delta maps.

Uses NVEncC (rigaya) when available for full QP delta map support.
Falls back to ffmpeg + NVENC (without QP maps) or ffmpeg + x265.
"""

import os
import subprocess
import tempfile
import shutil

import numpy as np
import torch
from PIL import Image


# Encoder discovery

def find_executable(name):
    """Find an executable in PATH or return None."""
    path = shutil.which(name)
    if path:
        return os.path.abspath(path)
    return None


def discover_encoder(preferred="nvenc"):
    """
    Return (encoder_backend, executable_path).

    Backends: "nvencc", "ffmpeg_nvenc", "ffmpeg_x265", "ffmpeg_x264"
    """
    if preferred == "nvenc":
        # Try NVEncC first (full QP delta map support)
        for name in ["NVEncC64.exe", "NVEncC"]:
            path = find_executable(name)
            if path:
                return "nvencc", path

        # Fallback to ffmpeg NVENC (no QP delta maps)
        ffmpeg = find_executable("ffmpeg")
        if ffmpeg:
            return "ffmpeg_nvenc", ffmpeg

    # Last resort: x265 / x264
    ffmpeg = find_executable("ffmpeg")
    if ffmpeg:
        return "ffmpeg_x265", ffmpeg

    return None, None


# Frame and QP map I/O

def write_frame_sequence(images, tmpdir, fps):
    """
    Write image tensor [F, H, W, C] as PNG sequence to tmpdir.

    Returns list of frame file paths.
    """
    num_frames = images.shape[0]
    paths = []
    for i in range(num_frames):
        frame = images[i]
        if frame.dtype == torch.float16 or frame.dtype == torch.float32:
            frame = (frame.cpu().numpy() * 255).astype(np.uint8)
        else:
            frame = frame.cpu().numpy()
        # Remove alpha channel if present (4 -> 3)
        if frame.shape[-1] == 4:
            frame = frame[:, :, :3]
        img = Image.fromarray(frame)
        path = os.path.join(tmpdir, f"frame_{i:05d}.png")
        img.save(path)
        paths.append(path)
    return paths


def write_qp_delta_maps(qp_maps, tmpdir):
    """
    Write QP delta maps [F, mb_H, mb_W] int8 as binary files.

    Returns list of (frame_index, qp_file_path) tuples.
    """
    num_frames = qp_maps.shape[0]
    paths = []
    for i in range(num_frames):
        qp_frame = qp_maps[i].cpu().numpy().astype(np.int8)
        path = os.path.join(tmpdir, f"qpmap_{i:05d}.bin")
        qp_frame.tofile(path)
        paths.append((i, path, qp_frame.shape))
    return paths


# NVEncC encoding

def encode_with_nvencc(frame_paths, qp_map_info, output_path, crf=23,
                       preset="medium", codec="hevc", fps=24, pix_fmt="yuv420p",
                       nvencc_path="NVEncC64.exe"):
    """
    Encode using NVEncC with QP delta maps.

    NVEncC supports -qpDeltaMapFile which accepts per-frame binary QP maps.
    """
    if not frame_paths:
        raise ValueError("No frame paths provided")

    first_frame = frame_paths[0]
    img = Image.open(first_frame)
    width, height = img.size

    # Build QP delta map base path (NVEncC expects %d placeholder)
    tmpdir = os.path.dirname(qp_map_info[0][1])
    qp_base = os.path.join(tmpdir, "qpmap_%05d.bin")

    # Determine preset mapping
    preset_map = {
        "p1": "p1", "p2": "p2", "p3": "p3", "p4": "p4",
        "p5": "p5", "p6": "p6", "p7": "p7",
        "slow": "p5", "medium": "p4", "fast": "p3",
        "faster": "p2", "veryfast": "p1",
    }
    nv_preset = preset_map.get(preset, "p4")

    codec_flag = "hevc" if codec == "hevc" else "h264"

    cmd = [
        nvencc_path,
        "--input", frame_paths[0],  # NVEncC auto-detects PNG sequence from first frame
        "--output", output_path,
        "--codec", codec_flag,
        "--vbr-quality", str(crf),
        "--preset", nv_preset,
        "--qpDeltaMapFile", qp_base,
        "--fps", str(fps),
        "--sw-scaled", "off",
    ]

    # Try to add pixel format
    if pix_fmt == "yuv420p":
        cmd.extend(["--y4m", "on"])

    # Run encoding
    print(f"[AGC] NVEncC command: {' '.join(cmd)}")
    result = subprocess.run(
        cmd, capture_output=True, text=True, timeout=600,
    )

    if result.returncode != 0:
        print(f"[AGC] NVEncC stderr: {result.stderr[:500]}")
        raise RuntimeError(f"NVEncC encoding failed (exit code {result.returncode})")

    return output_path


# FFmpeg encoding (fallback, no QP delta maps)

def encode_with_ffmpeg(frame_paths, output_path, crf=23, preset="medium",
                       codec="h264_nvenc", fps=24, pix_fmt="yuv420p",
                       ffmpeg_path="ffmpeg"):
    """
    Encode using ffmpeg.  QP delta maps are NOT supported in this fallback.
    """
    if not frame_paths:
        raise ValueError("No frame paths provided")

    first_frame = frame_paths[0]
    img = Image.open(first_frame)
    width, height = img.size

    # Input pattern for ffmpeg (PNG sequence)
    input_pattern = os.path.join(
        os.path.dirname(frame_paths[0]), "frame_%05d.png"
    )

    preset_map = {
        "p1": "hp", "p2": "hp", "p3": "medium", "p4": "medium",
        "p5": "slow", "p6": "slower", "p7": "veryslow",
        "slow": "slow", "medium": "medium", "fast": "fast",
        "faster": "faster", "veryfast": "veryfast",
    }
    ff_preset = preset_map.get(preset, "medium")

    cmd = [
        ffmpeg_path, "-y",
        "-framerate", str(fps),
        "-i", input_pattern,
        "-c:v", codec,
        "-crf", str(crf),
        "-preset", ff_preset,
        "-pix_fmt", pix_fmt,
    ]

    if codec == "hevc_nvenc":
        cmd.extend(["-b:v", "0"])

    cmd.append(output_path)

    print(f"[AGC] FFmpeg command: {' '.join(cmd)}")
    result = subprocess.run(
        cmd, capture_output=True, text=True, timeout=600,
    )

    if result.returncode != 0:
        print(f"[AGC] FFmpeg stderr: {result.stderr[:500]}")
        raise RuntimeError(f"FFmpeg encoding failed (exit code {result.returncode})")

    return output_path



# Main encode entry point



def encode_video(images, qp_maps_dict, output_dir, filename="output.mp4",
                 format="mp4", video_codec="hevc_nvenc", crf=23,
                 preset="medium", fps=24, pix_fmt="yuv420p",
                 nvencc_path=None, ffmpeg_path=None):
    """
    Encode video frames with optional QP delta maps.

    Args:
        images: tensor [F, H, W, C] float in [0, 1]
        qp_maps_dict: dict with "qp_maps" key -> tensor [F, mb_H, mb_W] int8
        output_dir: directory to write output file
        filename: output filename
        format: container format ("mp4", "webm", "mov")
        video_codec: codec name
        crf: constant rate factor
        preset: encoding preset
        fps: frames per second
        pix_fmt: pixel format
        nvencc_path: path to NVEncC executable (auto-detected if None)
        ffmpeg_path: path to ffmpeg (auto-detected if None)

    Returns:
        dict with "filename", "filepath", "encoder_backend"
    """
    output_path = os.path.join(output_dir, filename)
    tmpdir = tempfile.mkdtemp(prefix="agc_")

    try:
        # Write frames
        frame_paths = write_frame_sequence(images, tmpdir, fps)

        # Discover encoder
        if nvencc_path:
            backend, exe_path = "nvencc", nvencc_path
        elif ffmpeg_path:
            backend, exe_path = "ffmpeg_nvenc", ffmpeg_path
        else:
            backend, exe_path = discover_encoder("nvenc")

        if backend is None:
            raise RuntimeError("No suitable encoder found. Install ffmpeg or NVEncC.")

        # Write QP maps
        qp_map_info = []
        if qp_maps_dict and "qp_maps" in qp_maps_dict:
            qp_map_info = write_qp_delta_maps(qp_maps_dict["qp_maps"], tmpdir)

        # Encode
        if backend == "nvencc" and qp_map_info:
            encode_with_nvencc(
                frame_paths, qp_map_info, output_path,
                crf=crf, preset=preset,
                codec="hevc" if "hevc" in video_codec else "h264",
                fps=fps, pix_fmt=pix_fmt,
                nvencc_path=exe_path,
            )
        else:
            # Fallback: encode without QP maps
            actual_codec = video_codec
            if backend == "ffmpeg_x265":
                actual_codec = "libx265"
            elif backend == "ffmpeg_x264":
                actual_codec = "libx264"

            print(f"[AGC] Encoding with {backend} (QP delta maps {'applied' if backend == 'nvencc' and qp_map_info else 'NOT applied (fallback)'} )")

            encode_with_ffmpeg(
                frame_paths, output_path,
                crf=crf, preset=preset,
                codec=actual_codec,
                fps=fps, pix_fmt=pix_fmt,
                ffmpeg_path=exe_path,
            )

        return {
            "filename": filename,
            "filepath": output_path,
            "encoder_backend": backend,
            "qp_maps_applied": backend == "nvencc" and bool(qp_map_info),
        }

    finally:
        # Cleanup temp directory
        shutil.rmtree(tmpdir, ignore_errors=True)
