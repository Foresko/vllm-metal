# SPDX-License-Identifier: Apache-2.0
"""Tests for Gemma 4 image geometry (no checkpoints, no mlx)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from vllm_metal.multimodal.gemma4.geometry import (
    SUPPORTED_SOFT_TOKENS,
    effective_image_soft_tokens,
)


def _model_config(mm_processor_kwargs=None, *, vision_config=True) -> SimpleNamespace:
    hf_config = SimpleNamespace(model_type="gemma4")
    if vision_config:
        hf_config.vision_config = SimpleNamespace(default_output_length=280)
    return SimpleNamespace(hf_config=hf_config, mm_processor_kwargs=mm_processor_kwargs)


def test_supported_soft_tokens_matches_the_processor() -> None:
    assert SUPPORTED_SOFT_TOKENS == (70, 140, 280, 560, 1120)


@pytest.mark.parametrize(
    "kwargs, expected",
    [
        (None, 280),  # unset: the vision config's default geometry
        ({"max_soft_tokens": 1120}, 1120),  # top-level override
        ({"images_kwargs": {"max_soft_tokens": 1120}}, 1120),  # nested override
        ({"max_soft_tokens": 999}, 280),  # unsupported value: ignored
    ],
)
def test_effective_image_soft_tokens_resolves_the_kwarg_shapes(
    kwargs, expected
) -> None:
    assert effective_image_soft_tokens(_model_config(kwargs)) == expected


def test_effective_image_soft_tokens_without_a_vision_config() -> None:
    """A text-only hf_config still resolves to the processor's own default."""
    assert effective_image_soft_tokens(_model_config(vision_config=False)) == 280
