# SPDX-License-Identifier: Apache-2.0
"""Tests for the Gemma 4 vision sidecar loader."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import pytest

from vllm_metal.multimodal.gemma4 import Gemma4VisionSidecar, has_vision_weights


def _write_index(model_dir: Path, keys: list[str]) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "model.safetensors.index.json").write_text(
        json.dumps(
            {"weight_map": dict.fromkeys(keys, "model-00001-of-00001.safetensors")}
        )
    )


class TestHasVisionWeights:
    def test_index_with_vision_tower_key(self, tmp_path: Path) -> None:
        _write_index(
            tmp_path, ["language_model.model.embed_tokens.weight", "vision_tower.x"]
        )
        assert has_vision_weights(tmp_path) is True

    def test_index_with_hf_model_prefix(self, tmp_path: Path) -> None:
        _write_index(tmp_path, ["model.vision_tower.encoder.layers.0.w"])
        assert has_vision_weights(tmp_path) is True

    def test_index_without_vision(self, tmp_path: Path) -> None:
        _write_index(tmp_path, ["language_model.model.embed_tokens.weight"])
        assert has_vision_weights(tmp_path) is False

    def test_headers_fallback_without_index(self, tmp_path: Path) -> None:
        tmp_path.mkdir(exist_ok=True)
        mx.save_safetensors(
            str(tmp_path / "model.safetensors"),
            {"vision_tower.patch_embedder.input_proj.weight": mx.zeros((2, 2))},
        )
        assert has_vision_weights(tmp_path) is True

    def test_empty_dir(self, tmp_path: Path) -> None:
        assert has_vision_weights(tmp_path) is False


class _Tower(nn.Module):
    def __init__(self, proj: nn.Module) -> None:
        super().__init__()
        self.patch_embedder = SimpleNamespace(input_proj=proj)
        self.encoder_weight = mx.ones((4, 4), dtype=mx.bfloat16)
        self.proj = proj


class _Embed(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding_projection = nn.Linear(4, 8, bias=False)


def _composite(tower: nn.Module | None, embed: nn.Module | None) -> SimpleNamespace:
    return SimpleNamespace(
        vision_tower=tower, embed_vision=embed, language_model=object()
    )


class TestGemma4VisionSidecarLoad:
    def test_loads_tower_and_embedder_and_records_dtype(self, tmp_path: Path) -> None:
        tower = _Tower(nn.Linear(4, 4, bias=False))
        embed = _Embed()
        seen: list[Path] = []

        def _fake_load(path: Path) -> SimpleNamespace:
            seen.append(path)
            return _composite(tower, embed)

        sidecar = Gemma4VisionSidecar.load(tmp_path, load_composite=_fake_load)

        assert seen == [tmp_path]
        assert sidecar.vision_tower is tower
        assert sidecar.embed_vision is embed
        assert sidecar.pixel_dtype == mx.float32
        assert sidecar.num_parameters == 4 * 4 + 4 * 4 + 4 * 8
        assert sidecar.num_bytes > 0

    def test_quantized_input_projection_is_refused(self, tmp_path: Path) -> None:
        proj = nn.QuantizedLinear(64, 64, bias=False, group_size=32, bits=4)
        tower = _Tower(proj)

        with pytest.raises(RuntimeError, match="floating-point patch projection"):
            Gemma4VisionSidecar.load(
                tmp_path, load_composite=lambda _: _composite(tower, _Embed())
            )

    def test_missing_vision_modules_are_refused(self, tmp_path: Path) -> None:
        with pytest.raises(RuntimeError, match="vision_tower/embed_vision"):
            Gemma4VisionSidecar.load(
                tmp_path, load_composite=lambda _: _composite(None, _Embed())
            )

    def test_loader_exception_propagates(self, tmp_path: Path) -> None:
        def _boom(_: Path) -> SimpleNamespace:
            raise ValueError("missing parameters: vision_tower.x")

        with pytest.raises(ValueError, match="missing parameters"):
            Gemma4VisionSidecar.load(tmp_path, load_composite=_boom)
