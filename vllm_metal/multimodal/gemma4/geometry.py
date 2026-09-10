# SPDX-License-Identifier: Apache-2.0
"""Gemma 4 image geometry: soft-token counts, resolved the way vLLM does.

A leaf module on purpose — the platform hook needs these facts at config time,
long before any vision weight exists, so nothing here may reach for mlx,
mlx_vlm, torch, the Gemma 4 adapter or the vision sidecar.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

# Image soft-token counts vLLM's Gemma 4 processor accepts (gemma4_mm.py).
SUPPORTED_SOFT_TOKENS = (70, 140, 280, 560, 1120)


def effective_image_soft_tokens(model_config: Any) -> int:
    """Soft tokens per image, as vLLM's ``_get_max_soft_tokens`` resolves them."""
    hf_config = getattr(model_config, "hf_config", None)
    vision_config = getattr(hf_config, "vision_config", None)
    default = int(getattr(vision_config, "default_output_length", 280))
    kwargs = getattr(model_config, "mm_processor_kwargs", None) or {}
    value = kwargs.get("max_soft_tokens")
    if value is None:
        images_kwargs = kwargs.get("images_kwargs")
        if isinstance(images_kwargs, Mapping):
            value = images_kwargs.get("max_soft_tokens")
    if isinstance(value, int) and value in SUPPORTED_SOFT_TOKENS:
        return value
    return default
