"""
Cross-attention capture for DiT models (LTX 2.3, LTXAV, etc.).

Uses optimized_attention_override to intercept attention at the CrossAttention
level, with dit block wrappers to inject block_index into transformer_options.
"""

import threading
import math
import torch
from torch import einsum
from einops import rearrange, repeat


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


class CrossAttentionCapture:
    """
    Holds cross-attention maps captured during diffusion sampling.

    Uses optimized_attention_override to intercept attention at the lowest
    level, detecting cross-attention by q/k sequence length mismatch.
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
        self._debug_printed = False
        self._override_called = False
        self._wrapper_called = False
        self._fallback_printed = False
        self._fallback_shape_keys: dict[tuple[int, int], int] = {}

    # -- override factory

    def create_attention_override(self):
        """
        Return a function suitable for transformer_options["optimized_attention_override"].

        Signature: override(orig_func, q, k, v, heads, mask=None, attn_precision=None, transformer_options={}, **kwargs)
        """
        capture = self

        def override(orig_func, q, k, v, heads, mask=None, attn_precision=None, transformer_options={}, **kwargs):
            if not capture._override_called:
                print(f"[AGC] DEBUG: optimized_attention_override CALLED! transformer_options keys: {list(transformer_options.keys()) if transformer_options else 'NONE'}")
                capture._override_called = True

            block_index = transformer_options.get("agc_block_index", None)

            # Only capture in target layer range when block wrappers are active.
            # Some model paths (for example LTX/FLUX via MultimodalGuider) call
            # optimized_attention_override but do not use patches_replace["dit"].
            # In that case there is no layer index, so fall back to shape buckets.
            if block_index is not None and (
                block_index < capture.layer_start or block_index > capture.layer_end
            ):
                return orig_func(q, k, v, heads, mask=mask, attn_precision=attn_precision, transformer_options=transformer_options, **kwargs)

            # Detect cross-attention: q seq len != k seq len
            is_cross_attn = q.shape[1] != k.shape[1]

            if not is_cross_attn:
                return orig_func(q, k, v, heads, mask=mask, attn_precision=attn_precision, transformer_options=transformer_options, **kwargs)

            if block_index is None and not capture._fallback_printed:
                print(
                    "[AGC] DEBUG: No block wrapper context; capturing cross-attention "
                    "by q/k sequence shape instead of layer range"
                )
                capture._fallback_printed = True

            # This is cross-attention — compute explicit attention to capture scores
            out, sim = attention_with_scores(
                q, k, v, heads, mask=mask, attn_precision=attn_precision,
            )

            # Debug: print once
            if not capture._debug_printed:
                layer_desc = block_index if block_index is not None else "shape-bucket"
                print(f"[AGC] Capturing cross-attn: layer={layer_desc}, q_seq={q.shape[1]}, k_seq={k.shape[1]}, heads={heads}")
                capture._debug_printed = True

            # Capture conditional pass only
            cond_or_uncond = transformer_options.get("cond_or_uncond", None)
            if cond_or_uncond is not None and 0 in cond_or_uncond:
                cond_idx = cond_or_uncond.index(0)
                b = q.shape[0] // len(cond_or_uncond)
                n_slices = heads * b
                captured = sim[n_slices * cond_idx: n_slices * (cond_idx + 1)].float().cpu()

                with capture._lock:
                    layer_key = block_index
                    if layer_key is None:
                        shape_key = (int(q.shape[1]), int(k.shape[1]))
                        layer_key = capture._fallback_shape_keys.get(shape_key)
                        if layer_key is None:
                            layer_key = -(len(capture._fallback_shape_keys) + 1)
                            capture._fallback_shape_keys[shape_key] = layer_key

                    if capture.capture_last_step_only:
                        capture._maps[layer_key] = captured
                        capture._counts[layer_key] = 1
                    else:
                        if layer_key not in capture._maps:
                            capture._maps[layer_key] = captured.clone()
                            capture._counts[layer_key] = 1
                        else:
                            capture._maps[layer_key] += captured
                            capture._counts[layer_key] += 1

            return out

        return override

    def create_block_wrapper(self, block_index: int):
        """
        Create a dit block wrapper that injects agc_block_index into transformer_options.

        This is used with patches_replace["dit"][("double_block", i)] to ensure
        the attention override knows which block it's running in.
        """
        capture = self

        def wrapper(args, kwargs):
            if not capture._wrapper_called:
                print(f"[AGC] DEBUG: Block wrapper FIRST CALL for layer {block_index}")
                capture._wrapper_called = True

            to = args.get("transformer_options", {})
            if to is None:
                to = {}
            to = to.copy()
            to["agc_block_index"] = block_index

            args_copy = args.copy()
            args_copy["transformer_options"] = to

            return kwargs["original_block"](args_copy)

        return wrapper

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
        latent_h = math.ceil(image_h / vae_scale_spatial)
        latent_w = math.ceil(image_w / vae_scale_spatial)
        latent_t = max(1, math.ceil(num_frames / vae_scale_temporal))
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
        self._debug_printed = False
        self._override_called = False
        self._wrapper_called = False
        self._fallback_printed = False
        self._fallback_shape_keys.clear()
