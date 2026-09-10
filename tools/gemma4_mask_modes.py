# SPDX-License-Identifier: Apache-2.0
"""mlx-vlm Gemma 4 mask modes for parity references.

mlx-vlm 0.6.8 applies the blockwise bidirectional overlay on every layer and
composes it as ``(causal AND window) OR block``; HF/vLLM apply it only on
sliding layers as ``(causal OR block) AND window``.  ``hf`` reproduces HF,
``causal`` disables the overlay (the causal engine semantics), ``asis``
restores mlx-vlm's own behaviour.
"""

from __future__ import annotations

import math
from typing import Any

MODES = ("asis", "causal", "hf")
_ORIGINALS: dict[str, Any] = {}


def _mask_owner(lang_module: Any) -> type:
    return next(
        obj
        for obj in vars(lang_module).values()
        if isinstance(obj, type) and hasattr(obj, "_make_masks")
    )


def _window_only_mask(mx: Any, n: int, offset: int, window: int) -> Any:
    q = mx.arange(offset, offset + n)[:, None]
    k = mx.arange(offset + n)[None]
    return q < k + window


def _make_masks_hf(self: Any, h: Any, cache: Any, mm_token_type_ids: Any = None):
    import mlx.core as mx
    from mlx_vlm.models.gemma4 import language as lang

    mask: dict[str, Any] = {}
    masks = []
    has_audio = (
        mm_token_type_ids is not None and int(mx.sum(mm_token_type_ids == 3).item()) > 0
    )
    has_visual = (
        mm_token_type_ids is not None
        and int(mx.sum((mm_token_type_ids == 1) | (mm_token_type_ids == 2)).item()) > 0
    )
    use_bidi = (
        getattr(self.config, "use_bidirectional_attention", None) == "vision"
        and mm_token_type_ids is not None
        and has_visual
        and not has_audio
        and h.shape[1] > 1
    )
    for layer, c in zip(self.layers, cache, strict=False):
        if layer.layer_type not in mask:
            if layer.layer_type == "full_attention":
                mask["full_attention"] = lang.create_attention_mask(
                    h, c, return_array=(getattr(c, "left_padding", None) is not None)
                )
            elif layer.layer_type == "sliding_attention":
                offset = int(mx.max(mx.array(c.offset)).item()) if c is not None else 0
                return_array = (h.shape[1] > 1 and offset > 0) or use_bidi
                mask["sliding_attention"] = lang.create_attention_mask(
                    h, c, window_size=self.window_size, return_array=return_array
                )
                if use_bidi:
                    base = mask["sliding_attention"]
                    if isinstance(base, str):
                        base = lang.create_causal_mask(
                            h.shape[1], window_size=self.window_size
                        )
                    overlaid = _ORIGINALS["overlay"](self, base, mm_token_type_ids)
                    window = _window_only_mask(mx, h.shape[1], offset, self.window_size)
                    mask["sliding_attention"] = overlaid & mx.expand_dims(window, 0)
        masks.append(mask[layer.layer_type])
    return masks


def patch_mlx_vlm_mask_mode(mode: str) -> None:
    """Install ``mode`` on mlx-vlm's Gemma 4 language model class (process-global)."""
    if mode not in MODES:
        raise ValueError(f"unknown mask mode {mode!r}; expected one of {MODES}")
    from mlx_vlm.models.gemma4 import language as lang

    owner = _mask_owner(lang)
    if not _ORIGINALS:
        _ORIGINALS["make"] = owner._make_masks
        _ORIGINALS["overlay"] = owner._apply_blockwise_bidirectional_overlay
    owner._make_masks = _ORIGINALS["make"]
    owner._apply_blockwise_bidirectional_overlay = _ORIGINALS["overlay"]
    if mode == "causal":
        owner._apply_blockwise_bidirectional_overlay = lambda self, base_mask, ids: (
            base_mask
        )
    elif mode == "hf":
        owner._make_masks = _make_masks_hf


def kl_over_support(p: dict[int, float], q: dict[int, float]) -> float | None:
    """Approximate KL(p || q) over the ids both dicts carry (log-probs)."""
    support = set(p) & set(q)
    if not support:
        return None
    return sum(math.exp(p[t]) * (p[t] - q[t]) for t in support)
