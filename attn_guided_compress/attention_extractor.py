"""
Cross-attention capture for DiT models (LTX 2.3, etc.).

Patches attn2 (cross-attention) in selected transformer blocks to record
attention score matrices during sampling.  Captured maps are stored on CPU
to avoid VRAM pressure.
"""

import threading
import torch
from torch import einsum
from einops import rearrange, repeat

# Explicit attention that also returns the softmax attention matrix

def attention_with_scores(q, k, v, heads, mask=None, attn_precision=None):
    """
    Scaled dot-product attention that returns both output and attention scores.
    Bypasses flash-attention / xformers so we can inspect the score matrix.

    Args:
        q: [batch, q_seq, hidden]
        k: [batch, k_seq, hidden]
        v: [batch, k_seq, hidden]
        heads: int
        mask: optional boolean mask
        attn_precision: dtype for score computation

    Returns:
        out: [batch, q_seq, hidden]
        sim: [batch*heads, q_seq, k_seq]  softmax attention weights
    """
    b, _, dim_total = q.shape
    dim_head = dim_total // heads
    scale = dim_head ** -0.5

    q, k, v = map(
        lambda t: (
            t.unsqueeze(3)
            .reshape(b, -1, heads, dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b * heads, -1, dim_head)
            .contiguous()
        ),
        (q, k, v),
    )

    if attn_precision == torch.float32:
        sim = einsum("b i d, b j d -> b i j", q.float(), k.float()) * scale
    else:
        sim = einsum("b i d, b j d -> b i j", q, k) * scale

    del q, k

    if mask is not None:
        mask = rearrange(mask, "b ... -> b (...)")
        max_neg = -torch.finfo(sim.dtype).max
        mask = repeat(mask, "b j -> (b h) () j", h=heads)
        sim = sim.masked_fill(~mask, max_neg)

    sim = sim.softmax(dim=-1)
    out = einsum("b i j, b j d -> b i d", sim.to(v.dtype), v)

    out = (
        out.unsqueeze(0)
        .reshape(b, heads, -1, dim_head)
        .permute(0, 2, 1, 3)
        .reshape(b, -1, heads * dim_head)
    )

    return out, sim


# Capture buffer

class CrossAttentionCapture:
    """
    Holds cross-attention maps captured during diffusion sampling.

    Each entry is keyed by layer index and stores either:
      - a running sum + count  (when capturing all steps)
      - the last-step tensor   (when capture_last_step_only=True)
    """

    def __init__(self, layer_start: int, layer_end: int,
                 capture_last_step_only: bool = False):
        self.layer_start = layer_start
        self.layer_end = layer_end
        self.capture_last_step_only = capture_last_step_only
        # layer_idx -> tensor  [batch*heads, total_tokens, text_tokens]  (CPU float32)
        self._maps: dict[int, torch.Tensor] = {}
        # layer_idx -> int  (number of steps accumulated)
        self._counts: dict[int, int] = {}
        self._lock = threading.Lock()
        self._latent_shape_inferred = None  # (latent_t, latent_h, latent_w)

    # -- patch factory 

    def create_attn2_patch(self, layer_idx: int):
        """Return a function suitable for set_model_attn2_replace."""

        def patch_fn(q, k, v, extra_options):
            heads = extra_options["n_heads"]
            cond_or_uncond = extra_options["cond_or_uncond"]
            attn_precision = extra_options.get("attn_precision", None)

            out, sim = attention_with_scores(
                q, k, v, heads, attn_precision=attn_precision,
            )

            # Capture only the conditional pass (index 0 in cond_or_uncond)
            if 0 in cond_or_uncond:
                cond_idx = cond_or_uncond.index(0)
                b = q.shape[0] // len(cond_or_uncond)
                n_slices = heads * b
                captured = sim[n_slices * cond_idx: n_slices * (cond_idx + 1)].float().cpu()

                with self._lock:
                    if self.capture_last_step_only:
                        self._maps[layer_idx] = captured
                        self._counts[layer_idx] = 1
                    else:
                        if layer_idx not in self._maps:
                            self._maps[layer_idx] = captured.clone()
                            self._counts[layer_idx] = 1
                        else:
                            self._maps[layer_idx] += captured
                            self._counts[layer_idx] += 1

            return out

        return patch_fn

    # -- aggregation 

    def get_aggregated(self) -> dict[int, torch.Tensor]:
        """
        Return averaged attention maps per layer.

        Returns:
            {layer_idx: [batch*heads, total_tokens, text_tokens]}
        """
        result = {}
        for layer_idx, tensor in self._maps.items():
            cnt = self._counts.get(layer_idx, 1)
            result[layer_idx] = tensor / cnt
        return result

    def infer_latent_shape(self, image_h: int, image_w: int, num_frames: int,
                           vae_scale_spatial: int = 32,
                           vae_scale_temporal: int = 8):
        """
        Infer (latent_t, latent_h, latent_w) from image resolution and frame count.
        """
        latent_h = image_h // vae_scale_spatial
        latent_w = image_w // vae_scale_spatial
        latent_t = max(1, num_frames // vae_scale_temporal)
        self._latent_shape_inferred = (latent_t, latent_h, latent_w)
        return self._latent_shape_inferred

    @property
    def latent_shape(self):
        return self._latent_shape_inferred

    def has_data(self) -> bool:
        return len(self._maps) > 0

    def clear(self):
        self._maps.clear()
        self._counts.clear()
        self._latent_shape_inferred = None
