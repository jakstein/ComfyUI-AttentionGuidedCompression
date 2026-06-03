"""
Saliency post-processing pipeline.

Converts raw cross-attention maps captured during sampling into per-frame
QP delta maps suitable for NVENC's qpDeltaMap feature.
"""

import torch
import torch.nn.functional as F

from .utils import (
    gaussian_blur_2d,
    temporal_smooth,
    normalize_tensor,
    pool_to_macroblocks,
    compute_attention_entropy,
    linear_map,
)


def process_attention_to_qp(
    attn_capture,
    images,
    spatial_blur_sigma: float = 2.0,
    temporal_smoothing: str = "gaussian",
    temporal_window: int = 5,
    qp_min: float = -8.0,
    qp_max: float = 10.0,
    normalization: str = "percentile",
    percentile_low: float = 5.0,
    percentile_high: float = 95.0,
    saliency_metric: str = "magnitude",
    vae_scale_spatial: int = 32,
    vae_scale_temporal: int = 8,
):
    """
    Full pipeline: raw attention  ->  QP delta maps.

    Args:
        attn_capture: CrossAttentionCapture instance with populated maps
        images: tensor [F, H, W, C]  decoded frames (for resolution reference)
        spatial_blur_sigma: Gaussian blur sigma on spatial saliency maps
        temporal_smoothing: "none" | "gaussian" | "median" | "exponential"
        temporal_window: window size for temporal smoothing
        qp_min: most negative QP delta (best quality, high-saliency regions)
        qp_max: most positive QP delta (worst quality, low-saliency regions)
        normalization: "minmax" | "percentile" | "frame_percentile" | "zscore"
        percentile_low: low percentile clip
        percentile_high: high percentile clip
        saliency_metric: "magnitude" (max attn weight) | "entropy" (focused-ness)
        vae_scale_spatial: VAE spatial downscale factor (32 for LTX)
        vae_scale_temporal: VAE temporal downscale factor (8 for LTX)

    Returns:
        qp_maps: dict with
            - "qp_maps": tensor [F, mb_H, mb_W]  int8  (signed QP deltas)
            - "saliency_preview": tensor [F, H, W, 1]  float  (for visualization)
            - "mb_h", "mb_w": macroblock grid dimensions
    """
    num_frames, img_h, img_w, _ = images.shape

    # -- 1. Infer latent grid 
    latent_t, latent_h, latent_w = attn_capture.infer_latent_shape(
        img_h, img_w, num_frames, vae_scale_spatial, vae_scale_temporal,
    )
    total_tokens = latent_t * latent_h * latent_w

    # -- 2. Aggregate across layers 
    aggregated = attn_capture.get_aggregated()
    if not aggregated:
        raise RuntimeError(
            "No attention maps captured. "
            "Make sure you ran KSampler with the patched model."
        )

    # Convert each compatible attention tensor to saliency first. The fallback
    # capture path may see non-video cross-attention; keep only tensors whose
    # query sequence matches the inferred latent video grid.
    saliency_tensors = []
    skipped_shapes = []
    for layer_idx in sorted(aggregated.keys()):
        t = aggregated[layer_idx]
        if t.shape[1] != total_tokens:
            skipped_shapes.append((layer_idx, tuple(t.shape)))
            continue

        if saliency_metric == "magnitude":
            # Max attention weight across conditioning tokens per spatial position.
            saliency_tensors.append(t.max(dim=-1).values.mean(dim=0))
        elif saliency_metric == "entropy":
            # Negative normalized entropy: focused attention -> higher saliency.
            entropy = compute_attention_entropy(t)
            max_entropy = torch.log(
                torch.tensor(t.shape[-1], dtype=t.dtype, device=t.device)
            )
            saliency_tensors.append((1.0 - entropy / max_entropy).mean(dim=0))
        else:
            raise ValueError(f"Unknown saliency_metric: {saliency_metric}")

    if skipped_shapes:
        print(
            f"[AGC] Skipped {len(skipped_shapes)} captured attention tensor(s) "
            f"that did not match latent token count {total_tokens}: {skipped_shapes[:3]}"
        )

    if not saliency_tensors:
        captured_shapes = [(idx, tuple(t.shape)) for idx, t in sorted(aggregated.items())]
        raise RuntimeError(
            "No captured attention maps matched the inferred latent token count "
            f"{total_tokens}. Captured shapes: {captured_shapes}"
        )

    saliency = torch.stack(saliency_tensors, dim=0)

    # -- 4. Average across layers / fallback shape buckets
    # NOTE: This assumes batch_size == 1 (the ComfyUI video norm).
    #   With batch > 1, the saliency from different frames gets mixed together
    #   because the captured shape is [num_layers, batch*heads, total_tokens].
    #   Supporting batch > 1 would require splitting the batch*heads dimension
    #   and producing per-frame saliency maps before averaging.
    saliency = saliency.mean(dim=0)  # [total_tokens]

    # -- 5. Reshape to (latent_t, latent_h, latent_w) 
    saliency = saliency.reshape(latent_t, latent_h, latent_w)

    # -- 6. Upsample to pixel resolution
    # Spatial upsample
    saliency_up = F.interpolate(
        saliency.unsqueeze(0).unsqueeze(0).float(),
        size=(latent_t, img_h, img_w),
        mode="trilinear",
        align_corners=False,
    ).squeeze(0).squeeze(0)

    # Temporal upsample if latent_t < num_frames
    if saliency_up.shape[0] != num_frames:
        saliency_up = F.interpolate(
            saliency_up.unsqueeze(0).unsqueeze(0),
            size=(num_frames, img_h, img_w),
            mode="trilinear",
            align_corners=False,
        ).squeeze(0).squeeze(0)

    # -- 7. Spatial blur 
    if spatial_blur_sigma > 0:
        saliency_up = gaussian_blur_2d(saliency_up, spatial_blur_sigma)

    # -- 8. Temporal smoothing 
    saliency_up = temporal_smooth(saliency_up, temporal_smoothing, temporal_window)

    # -- 9. Normalize to [0, 1] 
    saliency_norm = normalize_tensor(
        saliency_up, normalization, percentile_low, percentile_high,
    )

    # -- 10. Map to QP delta range 
    # High saliency -> low QP (negative delta, better quality)
    # Low saliency  -> high QP (positive delta, worse quality)
    qp_deltas = linear_map(saliency_norm, 0.0, 1.0, qp_max, qp_min)

    # -- 11. Downsample to macroblock resolution 
    qp_mb = pool_to_macroblocks(qp_deltas, mb_size=16)

    # Clip to int8 range and cast
    qp_mb = torch.clamp(qp_mb, -128, 127).to(torch.int8)

    # -- 12. Build saliency preview as RGB for ComfyUI image saving
    preview = torch.clamp(saliency_norm, 0.0, 1.0).unsqueeze(-1).expand(-1, -1, -1, 3)

    mb_h, mb_w = qp_mb.shape[1], qp_mb.shape[2]

    return {
        "qp_maps": qp_mb,
        "saliency_preview": preview,
        "mb_h": mb_h,
        "mb_w": mb_w,
    }
