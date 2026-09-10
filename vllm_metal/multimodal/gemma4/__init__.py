# SPDX-License-Identifier: Apache-2.0
"""Gemma 4 multimodal support: vision sidecar on the mlx_lm text backbone."""

from __future__ import annotations

from vllm_metal.multimodal.gemma4.sidecar import (
    Gemma4VisionSidecar,
    has_vision_weights,
)

__all__ = [
    "Gemma4VisionSidecar",
    "has_vision_weights",
]
