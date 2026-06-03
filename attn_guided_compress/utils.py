"""
Shared tensor utilities: Gaussian blur, temporal smoothing, normalization, macroblock pooling.
"""

import torch
import torch.nn.functional as F
from math import ceil


def gaussian_blur_2d(frames, sigma):
    """
    Apply separable Gaussian blur to spatial dimensions of each frame.

    Args:
        frames: tensor [F, H, W] or [F, C, H, W]
        sigma: standard deviation of Gaussian kernel

    Returns:
        blurred: tensor same shape as input
    """
    if sigma <= 0:
        return frames

    kernel_size = max(3, int(2 * ceil(3 * sigma)))
    if kernel_size % 2 == 0:
        kernel_size += 1

    x = torch.linspace(
        -(kernel_size // 2), kernel_size // 2, kernel_size,
        device=frames.device, dtype=frames.dtype,
    )
    kernel = torch.exp(-0.5 * (x / sigma) ** 2)
    kernel /= kernel.sum()

    if frames.ndim == 3:
        frames = frames.unsqueeze(1)

    # Horizontal pass
    kernel_h = kernel.view(1, 1, 1, kernel_size).expand(
        frames.shape[1], 1, 1, kernel_size
    )
    padded = F.pad(frames, (kernel_size // 2, kernel_size // 2, 0, 0), mode="reflect")
    blurred_h = F.conv2d(padded, kernel_h, groups=frames.shape[1])

    # Vertical pass
    kernel_v = kernel.view(1, 1, kernel_size, 1).expand(
        frames.shape[1], 1, kernel_size, 1
    )
    padded = F.pad(blurred_h, (0, 0, kernel_size // 2, kernel_size // 2), mode="reflect")
    blurred = F.conv2d(padded, kernel_v, groups=frames.shape[1])

    if frames.ndim == 4 and frames.shape[1] == 1 and blurred.shape[1] == 1:
        blurred = blurred.squeeze(1)

    return blurred


def temporal_smooth(frames, method="gaussian", window=5, alpha=0.5):
    """
    Apply temporal smoothing across the frame dimension.

    Args:
        frames: tensor [F, H, W] or [F, C, H, W]
        method: one of "none", "gaussian", "median", "exponential"
        window: window size for gaussian / median
        alpha: smoothing factor for exponential (0-1)

    Returns:
        smoothed: tensor same shape as input
    """
    if method == "none":
        return frames

    num_frames = frames.shape[0]
    if num_frames <= 1:
        return frames

    if method == "gaussian":
        window = min(window, num_frames)
        if window < 3:
            return frames

        sigma = max(1.0, window / 6.0)
        t = torch.linspace(0, window - 1, window, device=frames.device, dtype=frames.dtype)
        kernel_1d = torch.exp(-0.5 * ((t - (window - 1) / 2) / sigma) ** 2)
        kernel_1d /= kernel_1d.sum()

        # Convolve along temporal axis using 1D convolution
        # [F, ...] -> [1, F, ...] for conv1d on dim 1
        x = frames.unsqueeze(0)  # [1, F, ...]
        kernel = kernel_1d.view(1, 1, window)  # [1, 1, window]
        pad = window // 2
        x_padded = F.pad(x, (pad, pad), mode="reflect")
        out = F.conv1d(x_padded, kernel, groups=1)
        return out.squeeze(0)

    elif method == "median":
        window = min(window, num_frames)
        if window < 3:
            return frames

        result = torch.zeros_like(frames)
        half = window // 2
        for i in range(num_frames):
            start = max(0, i - half)
            end = min(num_frames, i + half + 1)
            result[i] = torch.median(frames[start:end], dim=0).values
        return result

    elif method == "exponential":
        result = torch.zeros_like(frames)
        result[0] = frames[0]
        for i in range(1, num_frames):
            result[i] = alpha * frames[i] + (1 - alpha) * result[i - 1]
        return result

    else:
        raise ValueError(f"Unknown temporal smoothing method: {method}")


def normalize_tensor(tensor, method="minmax", low=5.0, high=95.0, eps=1e-6):
    """
    Normalize tensor values to [0, 1] range.

    Args:
        tensor: input tensor (any shape)
        method: one of "minmax", "percentile", "zscore"
        low: low percentile for percentile normalization
        high: high percentile for percentile normalization
        eps: numerical stability epsilon

    Returns:
        normalized: tensor squashed to roughly [0, 1]
    """
    flat = tensor.flatten().float()

    if method == "minmax":
        mn = flat.min()
        mx = flat.max()
        return (tensor.float() - mn) / (mx - mn + eps)

    elif method == "percentile":
        lo = torch.quantile(flat, low / 100.0)
        hi = torch.quantile(flat, high / 100.0)
        clipped = torch.clamp(tensor.float(), lo, hi)
        return (clipped - lo) / (hi - lo + eps)

    elif method == "zscore":
        mean = flat.mean()
        std = flat.std()
        z = (tensor.float() - mean) / (std + eps)
        return torch.sigmoid(z)

    else:
        raise ValueError(f"Unknown normalization method: {method}")


def pool_to_macroblocks(frames, mb_size=16):
    """
    Average-pool frames down to macroblock resolution.

    Args:
        frames: tensor [F, H, W] or [F, C, H, W]
        mb_size: macroblock pixel size (default 16)

    Returns:
        mb_frames: tensor [F, mb_H, mb_W] (single-channel pooled)
    """
    if frames.ndim == 3:
        frames = frames.unsqueeze(1)

    h, w = frames.shape[2], frames.shape[3]
    mb_h = (h + mb_size - 1) // mb_size
    mb_w = (w + mb_size - 1) // mb_size

    target_h = mb_h * mb_size
    target_w = mb_w * mb_size

    if target_h != h or target_w != w:
        frames = F.pad(frames, (0, target_w - w, 0, target_h - h), mode="replicate")

    pooled = F.avg_pool2d(frames, kernel_size=mb_size, stride=mb_size)
    return pooled.squeeze(1)


def compute_attention_entropy(attn_weights, eps=1e-8):
    """
    Compute per-element entropy of attention distributions.

    For each (batch, spatial_pos), compute entropy over the text-token dimension.
    Lower entropy  ->  attention is focused on few tokens  ->  more salient.

    Args:
        attn_weights: softmax attention [*, text_tokens]
        eps: numerical stability

    Returns:
        entropy: [*,]  (all dims except last squeezed)
    """
    log_w = torch.log(attn_weights + eps)
    entropy = -(attn_weights * log_w).sum(dim=-1)
    return entropy


def linear_map(value, in_min, in_max, out_min, out_max):
    """
    Linearly map a value (or tensor) from [in_min, in_max] to [out_min, out_max].

    Clamped to output range.
    """
    normalized = (value - in_min) / (in_max - in_min + 1e-8)
    normalized = torch.clamp(normalized, 0.0, 1.0)
    return out_min + normalized * (out_max - out_min)
