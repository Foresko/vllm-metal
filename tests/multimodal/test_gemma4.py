# SPDX-License-Identifier: Apache-2.0
"""Tests for the Gemma 4 multimodal adapter on fakes (no checkpoints)."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import pytest
import torch
from vllm.multimodal.inputs import MultiModalFieldConfig, MultiModalKwargsItem

from vllm_metal.multimodal import MultiModalFeatureSpec, PlaceholderRange
from vllm_metal.multimodal.gemma4 import (
    Gemma4MultimodalAdapter,
    Gemma4VisionEncodeResult,
    Gemma4VisionSidecar,
)

HIDDEN = 8  # text hidden size of the fake backbone
TOWER_DIM = 4  # fake vision hidden size
POOL = 9  # pooling_kernel_size ** 2


class _Backbone(nn.Module):
    """Mirror of mlx_lm ``Gemma4TextModel``: ``embed_tokens`` + ``embed_scale``."""

    def __init__(self, dtype: mx.Dtype = mx.bfloat16) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(16, HIDDEN)
        self.embed_tokens.weight = self.embed_tokens.weight.astype(dtype)
        self.embed_scale = HIDDEN**0.5


class _TextModel:
    """Mirror of mlx_lm ``gemma4.Model``: records the ``input_embeddings`` call."""

    def __init__(self, backbone: _Backbone) -> None:
        self.language_model = SimpleNamespace(model=backbone)
        self.calls: list[dict[str, Any]] = []

    def __call__(
        self, inputs: mx.array, cache: Any = None, input_embeddings: Any = None
    ) -> Any:
        self.calls.append(
            {"inputs": inputs, "cache": cache, "input_embeddings": input_embeddings}
        )
        return SimpleNamespace(logits=mx.zeros((1, inputs.shape[1], 16)))


class _Tower:
    def __init__(self) -> None:
        self.calls: list[tuple[mx.Dtype, tuple[int, ...], tuple[int, ...]]] = []

    def __call__(
        self, pixel_values: mx.array, pixel_position_ids: mx.array
    ) -> mx.array:
        self.calls.append(
            (pixel_values.dtype, pixel_values.shape, pixel_position_ids.shape)
        )
        rows = pixel_values.shape[-2] // POOL
        return mx.ones((1, rows, TOWER_DIM), dtype=pixel_values.dtype)


class _EmbedVision:
    def __call__(self, features: mx.array) -> mx.array:
        return mx.concatenate([features, features], axis=-1) * 3.0  # (1, rows, HIDDEN)


def _sidecar(
    tower: _Tower | None = None, pixel_dtype: mx.Dtype = mx.bfloat16
) -> Gemma4VisionSidecar:
    return Gemma4VisionSidecar(
        vision_tower=tower or _Tower(),
        embed_vision=_EmbedVision(),
        pixel_dtype=pixel_dtype,
        num_parameters=1,
        num_bytes=2,
    )


def _adapter(tower: _Tower | None = None) -> tuple[Gemma4MultimodalAdapter, _TextModel]:
    text_model = _TextModel(_Backbone())
    return Gemma4MultimodalAdapter.from_loaded(text_model, _sidecar(tower)), text_model


def _image_item(num_patches: int) -> MultiModalKwargsItem:
    pixels = MultiModalFieldConfig.batched("image", keep_on_cpu=True).build_elems(
        "pixel_values", torch.zeros((1, num_patches, 768), dtype=torch.float32)
    )[0]
    grid = torch.zeros((1, num_patches, 2), dtype=torch.int64)
    positions = MultiModalFieldConfig.batched("image", keep_on_cpu=True).build_elems(
        "pixel_position_ids", grid
    )[0]
    return MultiModalKwargsItem(
        {"pixel_values": pixels, "pixel_position_ids": positions}
    )


def _feature(
    *,
    offset: int = 1,
    num_embeds: int = 2,
    num_patches: int | None = None,
    modality: str = "image",
    with_boi_eoi: bool = True,
) -> MultiModalFeatureSpec:
    patches = num_patches if num_patches is not None else num_embeds * POOL
    if with_boi_eoi:
        position = PlaceholderRange(
            offset=offset,
            length=num_embeds + 2,
            is_embed=torch.tensor([False] + [True] * num_embeds + [False]),
        )
    else:
        position = PlaceholderRange(offset=offset, length=num_embeds)
    return MultiModalFeatureSpec(
        data=_image_item(patches),
        modality=modality,
        identifier=f"{modality}-{offset}",
        mm_position=position,
    )


class TestFlags:
    def test_contract_flags(self) -> None:
        adapter, _ = _adapter()
        assert adapter.forward_ready is True
        assert adapter.requires_explicit_positions is False
        assert adapter.supplies_segment_positions is False
        assert adapter.text_path_selective_logits_ok is True

    def test_text_model_is_the_loaded_object(self) -> None:
        adapter, text_model = _adapter()
        assert adapter.text_model() is text_model


class TestFromLoaded:
    def test_missing_backbone_raises(self) -> None:
        with pytest.raises(RuntimeError, match="language_model.model"):
            Gemma4MultimodalAdapter.from_loaded(SimpleNamespace(), _sidecar())

    def test_missing_embed_scale_raises(self) -> None:
        backbone = SimpleNamespace(embed_tokens=nn.Embedding(16, HIDDEN))
        text_model = SimpleNamespace(language_model=SimpleNamespace(model=backbone))
        with pytest.raises(RuntimeError, match="embed_scale"):
            Gemma4MultimodalAdapter.from_loaded(text_model, _sidecar())

    def test_embed_scale_is_rounded_to_the_embedding_dtype(self) -> None:
        adapter, _ = _adapter()
        expected = float(mx.array(HIDDEN**0.5, dtype=mx.bfloat16).item())
        assert adapter.embed_scale_rounded == expected


class TestEmbedTokens:
    def test_returns_unscaled_embeddings(self) -> None:
        adapter, text_model = _adapter()
        ids = mx.array([[1, 2, 3]], dtype=mx.int32)
        raw = text_model.language_model.model.embed_tokens(ids)
        out = adapter.embed_tokens(ids)
        assert out.dtype == raw.dtype
        assert mx.array_equal(out, raw).item()


class TestEncodeMultimodal:
    def test_casts_pixels_and_divides_by_rounded_scale(self) -> None:
        tower = _Tower()
        adapter, _ = _adapter(tower)
        feature = _feature(num_embeds=2)

        [result] = adapter.encode_multimodal([feature])

        assert isinstance(result, Gemma4VisionEncodeResult)
        assert result.deepstack_visual_embeds is None
        assert tower.calls[0][0] == mx.bfloat16  # pixels cast to the tower dtype
        assert tower.calls[0][1] == (
            2 * POOL,
            768,
        )  # no batch axis added by the adapter
        assert tower.calls[0][2] == (2 * POOL, 2)
        assert result.hidden_states.shape == (2, HIDDEN)
        assert result.hidden_states.dtype == mx.bfloat16
        # embed_vision produced 3.0 everywhere; the model multiplies by the
        # rounded scale again, so this must restore 3.0 within 1 ulp bf16.
        restored = result.hidden_states.astype(mx.float32) * adapter.embed_scale_rounded
        assert mx.abs(restored - 3.0).max().item() <= 3.0 * 2**-7

    def test_row_count_mismatch_raises(self) -> None:
        adapter, _ = _adapter()
        feature = _feature(num_embeds=2, num_patches=3 * POOL)  # tower yields 3 rows

        with pytest.raises(ValueError, match="produced 3 rows for 2"):
            adapter.encode_multimodal([feature])

    def test_video_modality_rejected(self) -> None:
        adapter, _ = _adapter()
        with pytest.raises(ValueError, match="only supports image features"):
            adapter.encode_multimodal([_feature(modality="video")])

    def test_missing_data_rejected(self) -> None:
        adapter, _ = _adapter()
        feature = MultiModalFeatureSpec(
            data=None,
            modality="image",
            identifier="image-0",
            mm_position=PlaceholderRange(offset=0, length=2),
        )
        with pytest.raises(ValueError, match="feature.data is required"):
            adapter.encode_multimodal([feature])


class TestPositions:
    def test_empty_input(self) -> None:
        adapter, _ = _adapter()
        positions, delta = adapter.get_mrope_input_positions([], [])
        assert positions.shape == (3, 1, 0)
        assert delta == 0

    def test_sequential_positions_and_zero_delta(self) -> None:
        adapter, _ = _adapter()
        positions, delta = adapter.get_mrope_input_positions(
            [7, 5, 99, 99, 6, 8], [_feature()]
        )
        assert positions.shape == (3, 1, 6)
        assert positions[0, 0].tolist() == [0, 1, 2, 3, 4, 5]
        assert mx.array_equal(positions[1], positions[0]).item()
        assert delta == 0

    def test_video_feature_rejected(self) -> None:
        adapter, _ = _adapter()
        with pytest.raises(ValueError, match="only supports image features"):
            adapter.get_mrope_input_positions([1, 2], [_feature(modality="video")])


class TestCallLm:
    def test_passes_input_embeddings_and_cache(self) -> None:
        adapter, text_model = _adapter()
        ids = mx.array([[1, 2]], dtype=mx.int32)
        embeds = mx.zeros((1, 2, HIDDEN), dtype=mx.bfloat16)
        cache = [object(), object()]

        out = adapter.call_lm(
            ids,
            embeds,
            cache,
            mx.zeros((3, 1, 2), dtype=mx.int32),
            visual_pos_masks=mx.array([[False, True]]),
        )

        call = text_model.calls[0]
        assert call["inputs"] is ids
        assert call["input_embeddings"] is embeds
        assert call["cache"] is cache
        assert out.logits.shape == (1, 2, 16)

    def test_deepstack_is_refused(self) -> None:
        adapter, _ = _adapter()
        with pytest.raises(RuntimeError, match="deepstack"):
            adapter.call_lm(
                mx.array([[1]], dtype=mx.int32),
                mx.zeros((1, 1, HIDDEN)),
                [],
                mx.zeros((3, 1, 1), dtype=mx.int32),
                deepstack_visual_embeds=[mx.zeros((1, HIDDEN))],
            )


class TestProfileFeatures:
    def test_one_full_size_feature_with_real_grid(self) -> None:
        adapter, _ = _adapter()
        [feature] = adapter.profile_features()
        assert feature.modality == "image"
        assert feature.mm_position.get_num_embeds() == 280
        pixels = feature.data["pixel_values"].data
        positions = feature.data["pixel_position_ids"].data
        assert tuple(pixels.shape) == (2520, 768)
        assert tuple(positions.shape) == (2520, 2)
        assert int(positions.min()) == 0  # a real grid, no -1 padding
        assert int(positions[:, 0].max()) == 55 and int(positions[:, 1].max()) == 44

    def test_profile_feature_encodes_to_280_rows(self) -> None:
        adapter, _ = _adapter()
        [result] = adapter.encode_multimodal(adapter.profile_features())
        assert result.hidden_states.shape == (280, HIDDEN)
