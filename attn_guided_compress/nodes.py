"""
ComfyUI node definitions for Attention-Guided Compression.

Three nodes:
  1. AttnGuidedModelPatch   — patch model to capture cross-attention
  2. SaliencyPostProcess     — convert attention maps to QP delta maps
  3. AttnGuidedVideoCombine  — encode video with NVENC + QP maps
"""

import os
import torch

from .attention_extractor import CrossAttentionCapture
from .saliency_processor import process_attention_to_qp
from .nvenc_encoder import encode_video

import folder_paths

OUTPUT_DIR = folder_paths.get_output_directory()


# Node 1: Model Patch — captures cross-attention during sampling

class AttnGuidedModelPatch:
    """
    Patches the model's cross-attention (attn2) layers to record attention
    score matrices during KSampler execution.  The returned capture handle
    should be passed to SaliencyPostProcess after sampling completes.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "layer_start": ("INT", {
                    "default": 8, "min": 0, "max": 127, "step": 1,
                    "display": "number",
                }),
                "layer_end": ("INT", {
                    "default": 40, "min": 0, "max": 127, "step": 1,
                    "display": "number",
                }),
                "capture_last_step_only": ("BOOLEAN", {
                    "default": False,
                }),
            },
        }

    RETURN_TYPES = ("MODEL", "ATTN_CAPTURE",)
    RETURN_NAMES = ("model", "attn_capture",)
    FUNCTION = "execute"
    CATEGORY = "AttentionGuidedCompression"
    DESCRIPTION = (
        "Patches cross-attention layers to capture attention maps during "
        "diffusion sampling.  Connect the patched model to your KSampler, "
        "then pass the attn_capture handle to SaliencyPostProcess."
    )

    def execute(self, model, layer_start, layer_end, capture_last_step_only):
        if layer_start > layer_end:
            raise ValueError(
                f"layer_start ({layer_start}) must be <= layer_end ({layer_end})"
            )

        capture = CrossAttentionCapture(
            layer_start=layer_start,
            layer_end=layer_end,
            capture_last_step_only=capture_last_step_only,
        )

        m = model.clone()

        for layer_idx in range(layer_start, layer_end + 1):
            patch_fn = capture.create_attn2_patch(layer_idx)
            # For DiT models (LTX, etc.) blocks are named ("input", idx)
            m.set_model_attn2_replace(patch_fn, "input", layer_idx)

        print(
            f"[AGC] Patched attn2 in layers {layer_start}-{layer_end} "
            f"(capture_last_step_only={capture_last_step_only})"
        )

        return (m, capture,)


# Node 2: Saliency Post-Process — attention maps → QP deltas

class SaliencyPostProcess:
    """
    Converts captured cross-attention maps into per-frame QP delta maps
    and a visual saliency preview.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "attn_capture": ("ATTN_CAPTURE",),
                "images": ("IMAGE",),
                "spatial_blur_sigma": ("FLOAT", {
                    "default": 2.0, "min": 0.0, "max": 20.0, "step": 0.1,
                }),
                "temporal_smoothing": ([
                    "none", "gaussian", "median", "exponential",
                ], {
                    "default": "gaussian",
                }),
                "temporal_window": ("INT", {
                    "default": 5, "min": 1, "max": 63, "step": 1,
                }),
                "qp_min": ("FLOAT", {
                    "default": -8.0, "min": -23.0, "max": 0.0, "step": 0.5,
                }),
                "qp_max": ("FLOAT", {
                    "default": 10.0, "min": 0.0, "max": 30.0, "step": 0.5,
                }),
                "normalization": ([
                    "minmax", "percentile", "zscore",
                ], {
                    "default": "percentile",
                }),
                "percentile_low": ("FLOAT", {
                    "default": 5.0, "min": 0.0, "max": 50.0, "step": 0.5,
                }),
                "percentile_high": ("FLOAT", {
                    "default": 95.0, "min": 50.0, "max": 100.0, "step": 0.5,
                }),
                "saliency_metric": ([
                    "magnitude", "entropy",
                ], {
                    "default": "magnitude",
                }),
                "vae_scale_spatial": ("INT", {
                    "default": 32, "min": 1, "max": 128, "step": 1,
                }),
                "vae_scale_temporal": ("INT", {
                    "default": 8, "min": 1, "max": 64, "step": 1,
                }),
            },
        }

    RETURN_TYPES = ("QP_MAPS", "IMAGE",)
    RETURN_NAMES = ("qp_maps", "saliency_preview",)
    FUNCTION = "execute"
    CATEGORY = "AttentionGuidedCompression"
    DESCRIPTION = (
        "Processes captured attention maps into QP delta maps for NVENC "
        "encoding.  Also outputs a saliency preview image."
    )

    def execute(self, attn_capture, images, spatial_blur_sigma,
                temporal_smoothing, temporal_window, qp_min, qp_max,
                normalization, percentile_low, percentile_high,
                saliency_metric, vae_scale_spatial, vae_scale_temporal):

        if not attn_capture.has_data():
            raise RuntimeError(
                "No attention data captured.  "
                "Make sure you ran KSampler with the patched model."
            )

        result = process_attention_to_qp(
            attn_capture=attn_capture,
            images=images,
            spatial_blur_sigma=spatial_blur_sigma,
            temporal_smoothing=temporal_smoothing,
            temporal_window=temporal_window,
            qp_min=qp_min,
            qp_max=qp_max,
            normalization=normalization,
            percentile_low=percentile_low,
            percentile_high=percentile_high,
            saliency_metric=saliency_metric,
            vae_scale_spatial=vae_scale_spatial,
            vae_scale_temporal=vae_scale_temporal,
        )

        print(
            f"[AGC] Saliency processed: {result['qp_maps'].shape} QP map, "
            f"metric={saliency_metric}, range=[{qp_min}, {qp_max}]"
        )

        # Clear captured data to free memory
        attn_capture.clear()

        return (result, result["saliency_preview"],)


# Node 3: Video Combine — encode with NVENC + QP delta maps

class AttnGuidedVideoCombine:
    """
    Encodes image frames to video using NVENC.  When QP maps are provided
    and NVEncC is available, applies per-macroblock QP deltas for
    perceptual quality optimization.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "qp_maps": ("QP_MAPS",),
                "fps": ("INT", {
                    "default": 24, "min": 1, "max": 120, "step": 1,
                }),
                "crf": ("INT", {
                    "default": 23, "min": 0, "max": 51, "step": 1,
                }),
                "format": (["mp4", "webm", "mov", "mkv"], {
                    "default": "mp4",
                }),
                "video_codec": (
                    ["hevc_nvenc", "h264_nvenc", "libx265", "libx264"],
                    {"default": "hevc_nvenc"},
                ),
                "preset": (
                    ["p1", "p2", "p3", "p4", "p5", "p6", "p7",
                     "slow", "medium", "fast", "faster"],
                    {"default": "p4"},
                ),
                "pix_fmt": (["yuv420p", "yuv422p", "yuv444p"], {
                    "default": "yuv420p",
                }),
                "filename_prefix": ("STRING", {
                    "default": "agc_video",
                }),
            },
            "optional": {
                "audio": ("AUDIO",),
                "nvencc_path": ("STRING", {
                    "default": "",
                }),
                "ffmpeg_path": ("STRING", {
                    "default": "",
                }),
            },
        }

    RETURN_TYPES = ("VIDEO",)
    RETURN_NAMES = ("video_info",)
    FUNCTION = "execute"
    CATEGORY = "AttentionGuidedCompression"
    DESCRIPTION = (
        "Encodes frames to video with optional per-macroblock QP deltas. "
        "Requires NVEncC for QP map support; falls back to ffmpeg NVENC."
    )

    def execute(self, images, qp_maps, fps, crf, format, video_codec,
                preset, pix_fmt, filename_prefix, audio=None,
                nvencc_path="", ffmpeg_path=""):

        num_frames = images.shape[0]
        ext = f".{format}"
        filename = f"{filename_prefix}{ext}"

        # Resolve encoder paths
        nv_path = nvencc_path if nvencc_path else None
        ff_path = ffmpeg_path if ffmpeg_path else None

        result = encode_video(
            images=images,
            qp_maps_dict=qp_maps,
            output_dir=OUTPUT_DIR,
            filename=filename,
            format=format,
            video_codec=video_codec,
            crf=crf,
            preset=preset,
            fps=fps,
            pix_fmt=pix_fmt,
            nvencc_path=nv_path,
            ffmpeg_path=ff_path,
        )

        video_info = {
            "filename": result["filename"],
            "subfolder": "",
            "type": "output",
            "fps": fps,
            "frame_count": num_frames,
            "encoder": result["encoder_backend"],
            "qp_maps_applied": result["qp_maps_applied"],
        }

        print(
            f"[AGC] Video saved: {result['filepath']}  "
            f"({result['encoder_backend']}, "
            f"QP maps {'applied' if result['qp_maps_applied'] else 'NOT applied'})"
        )

        return (video_info,)


# Registration

NODE_CLASS_MAPPINGS = {
    "AttnGuidedModelPatch": AttnGuidedModelPatch,
    "SaliencyPostProcess": SaliencyPostProcess,
    "AttnGuidedVideoCombine": AttnGuidedVideoCombine,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AttnGuidedModelPatch": "Attn-Guided Model Patch",
    "SaliencyPostProcess": "Saliency Post-Process",
    "AttnGuidedVideoCombine": "Attn-Guided Video Combine",
}
