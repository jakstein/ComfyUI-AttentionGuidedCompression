"""
SVT-AV1 video encoding with per-frame ROI QP delta maps.

Uses ffmpeg + libsvtav1 with the --roi-map-file parameter for per-superblock
QP offset maps via AV1's alternate quantizer segment feature.
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


# Frame and ROI map I/O

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
        if frame.shape[-1] == 4:
            frame = frame[:, :, :3]
        img = Image.fromarray(frame)
        path = os.path.join(tmpdir, f"frame_{i:05d}.png")
        img.save(path)
        paths.append(path)
    return paths


def write_roi_map(qp_maps_tensor, tmpdir):
    """
    Write QP delta maps [F, mb_H, mb_W] int8 as a single SVT-AV1 ROI map text file.

    Format: one line per frame: "frame_num offset1 offset2 offset3 ..."
    Each offset = QP delta for one 64x64 superblock, row-by-row raster order.
    Values clamped to [-63, 63].

    Args:
        qp_maps_tensor: tensor [F, mb_H, mb_W] int8
        tmpdir: directory to write the file

    Returns:
        path to the ROI map text file
    """
    qp_maps = qp_maps_tensor.cpu().numpy().astype(np.int8)
    qp_maps = np.clip(qp_maps, -63, 63)

    roi_path = os.path.join(tmpdir, "roi_map.txt")
    with open(roi_path, "w") as f:
        for frame_idx in range(qp_maps.shape[0]):
            frame = qp_maps[frame_idx]
            values = frame.flatten().tolist()
            f.write(f"{frame_idx} {' '.join(str(v) for v in values)}\n")

    return roi_path


# SVT-AV1 encoding via ffmpeg

def encode_with_svtav1(frame_paths, output_path, roi_map_path, crf, preset,
                       fps, pix_fmt, ffmpeg_path):
    """
    Encode using ffmpeg with libsvtav1 codec.

    Args:
        frame_paths: list of frame file paths
        output_path: output video file path
        roi_map_path: path to ROI map text file (or None to encode without QP maps)
        crf: constant rate factor (0-63)
        preset: SVT-AV1 preset (-2 to 13)
        fps: frames per second
        pix_fmt: pixel format
        ffmpeg_path: path to ffmpeg executable
    """
    if not frame_paths:
        raise ValueError("No frame paths provided")

    input_pattern = os.path.join(
        os.path.dirname(frame_paths[0]), "frame_%05d.png"
    )

    cmd = [
        ffmpeg_path, "-y",
        "-framerate", str(fps),
        "-i", input_pattern,
        "-c:v", "libsvtav1",
        "-crf", str(crf),
        "-preset", str(preset),
        "-pix_fmt", pix_fmt,
    ]

    if roi_map_path:
        # Convert to forward slashes to avoid Windows path : conflicts
        # with the svtav1-params key:value delimiter
        roi_path_escaped = roi_map_path.replace("\\", "/")
        cmd.extend([
            "-svtav1-params", f"roi-map-file={roi_path_escaped}"
        ])

    cmd.append(output_path)

    print(f"[AGC] FFmpeg SVT-AV1 command: {' '.join(cmd)}")
    result = subprocess.run(
        cmd, capture_output=True, text=True, timeout=600,
    )

    if result.returncode != 0:
        print(f"[AGC] FFmpeg stderr: {result.stderr[:1000]}")
        raise RuntimeError(
            f"FFmpeg SVT-AV1 encoding failed (exit code {result.returncode})"
        )

    return output_path


# Main encode entry point

def encode_video(images, qp_maps_dict, output_dir, filename="output.mp4",
                 format="mp4", crf=28, preset="8", fps=24, pix_fmt="yuv420p",
                 ffmpeg_path=None):
    """
    Encode video frames with SVT-AV1 and optional ROI QP delta maps.

    Args:
        images: tensor [F, H, W, C] float in [0, 1]
        qp_maps_dict: dict with "qp_maps" key -> tensor [F, mb_H, mb_W] int8
        output_dir: directory to write output file
        filename: output filename
        format: container format ("mp4", "webm", "mkv")
        crf: constant rate factor (0-63)
        preset: SVT-AV1 preset string ("4", "6", "8", "10", "11", "13")
                or friendly name ("fast", "medium", "slow", etc.)
        fps: frames per second
        pix_fmt: pixel format
        ffmpeg_path: path to ffmpeg (auto-detected if None)

    Returns:
        dict with "filename", "filepath", "encoder_backend", "qp_maps_applied"
    """
    # Map friendly preset names to SVT-AV1 preset numbers
    preset_map = {
        "p4": "8", "medium": "8",
        "p3": "6", "fast": "6",
        "p5": "10", "slow": "10",
        "p2": "4", "faster": "4",
        "p6": "11", "slower": "11",
        "p7": "2", "veryfast": "2",
        "4": "4", "6": "6", "8": "8",
        "10": "10", "11": "11", "13": "13",
    }
    svt_preset = preset_map.get(preset, "8")

    output_path = os.path.join(output_dir, filename)
    tmpdir = tempfile.mkdtemp(prefix="agc_")

    try:
        # Resolve ffmpeg
        if not ffmpeg_path:
            ffmpeg_path = find_executable("ffmpeg")
        if not ffmpeg_path:
            raise RuntimeError(
                "ffmpeg not found in PATH. Install ffmpeg with libsvtav1 support."
            )

        # Write frames
        frame_paths = write_frame_sequence(images, tmpdir, fps)

        # Write ROI map
        roi_map_path = None
        qp_maps_applied = False
        if qp_maps_dict and "qp_maps" in qp_maps_dict:
            roi_map_path = write_roi_map(qp_maps_dict["qp_maps"], tmpdir)
            qp_maps_applied = True

        # Encode
        encode_with_svtav1(
            frame_paths, output_path, roi_map_path,
            crf=crf, preset=svt_preset,
            fps=fps, pix_fmt=pix_fmt,
            ffmpeg_path=ffmpeg_path,
        )

        print(
            f"[AGC] Encoding with libsvtav1 preset {svt_preset} "
            f"(QP maps {'applied' if qp_maps_applied else 'NOT applied'})"
        )

        return {
            "filename": filename,
            "filepath": output_path,
            "encoder_backend": "libsvtav1",
            "qp_maps_applied": qp_maps_applied,
        }

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
