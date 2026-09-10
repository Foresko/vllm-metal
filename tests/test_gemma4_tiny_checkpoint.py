# SPDX-License-Identifier: Apache-2.0
"""Builds the tiny synthetic Gemma 4 checkpoint and checks both loaders accept it."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from types import ModuleType

import pytest

pytestmark = [pytest.mark.slow, pytest.mark.network]


def _load_tool_module() -> ModuleType:
    """Load ``tools/gemma4_tiny_checkpoint.py`` directly; ``tools/`` isn't a package."""
    path = Path(__file__).resolve().parents[1] / "tools" / "gemma4_tiny_checkpoint.py"
    spec = importlib.util.spec_from_file_location("gemma4_tiny_checkpoint", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.skipif(os.environ.get("VLLM_METAL_E2E", "1") != "1", reason="e2e disabled")
def test_tiny_checkpoint_loads_everywhere(tmp_path: Path) -> None:
    module = _load_tool_module()

    out = module.build_tiny_checkpoint(tmp_path / "tiny")

    from mlx_lm import load as mlx_lm_load
    from mlx_vlm.utils import load_model as mlx_vlm_load_model
    from transformers import AutoProcessor

    from vllm_metal.multimodal.gemma4 import Gemma4VisionSidecar, has_vision_weights

    text_model, tokenizer = mlx_lm_load(str(out))
    assert text_model.language_model.model.embed_scale == 8.0
    composite = mlx_vlm_load_model(out, lazy=True)
    assert composite.vision_tower is not None
    assert has_vision_weights(out) is True
    sidecar = Gemma4VisionSidecar.load(out)
    assert sidecar.num_parameters > 0
    processor = AutoProcessor.from_pretrained(str(out))
    assert type(processor).__name__ == "Gemma4Processor"


@pytest.mark.skipif(os.environ.get("VLLM_METAL_E2E", "1") != "1", reason="e2e disabled")
def test_tiny_checkpoint_without_vision(tmp_path: Path) -> None:
    module = _load_tool_module()

    from vllm_metal.multimodal.gemma4 import has_vision_weights

    out = module.build_tiny_checkpoint(tmp_path / "tiny-text", with_vision=False)
    assert has_vision_weights(out) is False
