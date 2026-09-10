# SPDX-License-Identifier: Apache-2.0
"""Image-block ranges reach the paged context; split blocks fall back to causal."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import mlx.core as mx
import pytest
import torch

import vllm_metal.v1.model_runner as model_runner_module
from tests.v1.mm.test_paged_forward_mm import (
    _mm_prefill,
    _MmAdapter,
    _put_encode,
    _runner,
    _scheduler_output,
)
from vllm_metal.attention.context import get_context
from vllm_metal.multimodal import MultiModalFeatureSpec, PlaceholderRange
from vllm_metal.v1.model_runner import PrefillRequest

PROMPT = [10, 7, 99, 99, 8, 11]  # boi at 1, image tokens 2..3, eoi at 4


class _BidiAdapter(_MmAdapter):
    bidirectional_layer_kinds = frozenset({"sliding"})


def _image_feature(
    identifier: str = "img-0", *, offset: int = 1
) -> MultiModalFeatureSpec:
    return MultiModalFeatureSpec(
        data=None,
        modality="image",
        identifier=identifier,
        mm_position=PlaceholderRange(
            offset=offset,
            length=4,
            is_embed=torch.tensor([False, True, True, False]),
        ),
    )


def _capture(adapter: _MmAdapter) -> dict[str, Any]:
    captured: dict[str, Any] = {}
    original = adapter.call_lm

    def call_lm(*args, **kwargs):
        ctx = get_context()
        captured["ranges"] = ctx.segment_bidi_ranges
        captured["kinds"] = ctx.bidi_layer_kinds
        return original(*args, **kwargs)

    adapter.call_lm = call_lm  # type: ignore[method-assign]
    return captured


def _forward(runner, prefill: PrefillRequest, decode_reqs=()) -> None:
    runner._spec_decode_controller.build_decode_segments = MagicMock(return_value=())
    runner._start_paged_forward(
        batch=MagicMock(),
        prefill_reqs=[prefill],
        decode_reqs=list(decode_reqs),
        scheduler_output=_scheduler_output(),
    )


def _warnings(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    seen: list[str] = []
    monkeypatch.setattr(
        model_runner_module.logger,
        "warning",
        lambda msg, *args: seen.append(msg % args if args else msg),
    )
    return seen


class TestRangesReachTheContext:
    def test_block_inside_the_chunk_is_passed_half_open(self) -> None:
        adapter = _BidiAdapter()
        runner = _runner(adapter)
        runner.encoder_cache.add_request("req", [_image_feature()])
        _put_encode(runner, "img-0", hidden_states=mx.ones((2, adapter.hidden_size)))
        captured = _capture(adapter)

        _forward(
            runner,
            _mm_prefill("req", token_ids=PROMPT, prompt_len=6, full_prompt=PROMPT),
        )

        assert captured["ranges"] == [[(2, 4)]]
        assert captured["kinds"] == frozenset({"sliding"})

    def test_two_images_give_two_ranges(self) -> None:
        adapter = _BidiAdapter()
        runner = _runner(adapter)
        prompt = [10, 7, 99, 99, 8, 7, 99, 99, 8, 11]
        runner.encoder_cache.add_request(
            "req",
            [_image_feature("img-0", offset=1), _image_feature("img-1", offset=5)],
        )
        for name in ("img-0", "img-1"):
            _put_encode(runner, name, hidden_states=mx.ones((2, adapter.hidden_size)))
        captured = _capture(adapter)

        _forward(
            runner,
            _mm_prefill("req", token_ids=prompt, prompt_len=10, full_prompt=prompt),
        )

        assert captured["ranges"] == [[(2, 4), (6, 8)]]

    def test_adapter_without_the_attribute_leaves_the_defaults(self) -> None:
        adapter = _MmAdapter()
        runner = _runner(adapter)
        runner.encoder_cache.add_request("req", [_image_feature()])
        _put_encode(runner, "img-0", hidden_states=mx.ones((2, adapter.hidden_size)))
        captured = _capture(adapter)

        _forward(
            runner,
            _mm_prefill("req", token_ids=PROMPT, prompt_len=6, full_prompt=PROMPT),
        )

        assert captured["ranges"] is None
        assert captured["kinds"] == frozenset()

    def test_text_prefill_next_to_an_image_prefill_gets_none(self) -> None:
        adapter = _BidiAdapter()
        runner = _runner(adapter)
        runner.encoder_cache.add_request("req", [_image_feature()])
        _put_encode(runner, "img-0", hidden_states=mx.ones((2, adapter.hidden_size)))
        captured = _capture(adapter)
        text = PrefillRequest(
            req_id="req-text",
            token_ids=[1, 2, 3],
            sampling_params=_mm_prefill(
                "x", token_ids=[1], prompt_len=1
            ).sampling_params,
            block_ids=[[1]],
            generator=None,
            prompt_len=3,
            start_pos=0,
            full_prompt_token_ids=[1, 2, 3],
        )
        runner._spec_decode_controller.build_decode_segments = MagicMock(
            return_value=()
        )
        runner._start_paged_forward(
            batch=MagicMock(),
            prefill_reqs=[
                _mm_prefill("req", token_ids=PROMPT, prompt_len=6, full_prompt=PROMPT),
                text,
            ],
            decode_reqs=[],
            scheduler_output=_scheduler_output(),
        )
        assert captured["ranges"] == [[(2, 4)], None]


class TestKeepDropRule:
    def test_block_split_across_chunks_falls_back_to_causal_for_the_request(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        adapter = _BidiAdapter()
        runner = _runner(adapter)
        runner.encoder_cache.add_request("req", [_image_feature()])
        _put_encode(runner, "img-0", hidden_states=mx.ones((2, adapter.hidden_size)))
        captured = _capture(adapter)
        warnings = _warnings(monkeypatch)

        # Chunk 1 covers positions 0..2: the block [2, 4) starts here but does not fit.
        _forward(
            runner,
            _mm_prefill(
                "req",
                token_ids=PROMPT[:3],
                prompt_len=None,
                start_pos=0,
                full_prompt=PROMPT,
            ),
        )
        assert captured["ranges"] is None
        assert len(warnings) == 1
        assert "falling back to causal attention" in warnings[0]
        assert runner._mm_bidi_states["req"].causal_only is True

        # Chunk 2: still causal, no second warning.
        _forward(
            runner,
            _mm_prefill(
                "req",
                token_ids=PROMPT[3:],
                prompt_len=6,
                start_pos=3,
                full_prompt=PROMPT,
            ),
        )
        assert captured["ranges"] is None
        assert len(warnings) == 1

    def test_prefix_hit_inside_the_block_keeps_the_tail(self) -> None:
        adapter = _BidiAdapter()
        runner = _runner(adapter)
        runner.encoder_cache.add_request("req", [_image_feature()])
        _put_encode(runner, "img-0", hidden_states=mx.ones((2, adapter.hidden_size)))
        captured = _capture(adapter)

        # First chunk of this request starts at 3 (prefix hit): block [2, 4) tail fits.
        _forward(
            runner,
            _mm_prefill(
                "req",
                token_ids=PROMPT[3:],
                prompt_len=6,
                start_pos=3,
                full_prompt=PROMPT,
            ),
        )

        assert captured["ranges"] == [[(2, 4)]]
        assert runner._mm_bidi_states["req"].first_prefill_start == 3

    def test_block_entirely_in_the_context_is_not_passed(self) -> None:
        adapter = _BidiAdapter()
        runner = _runner(adapter)
        runner.encoder_cache.add_request("req", [_image_feature()])
        _put_encode(runner, "img-0", hidden_states=mx.ones((2, adapter.hidden_size)))
        captured = _capture(adapter)

        _forward(
            runner,
            _mm_prefill(
                "req",
                token_ids=PROMPT[4:],
                prompt_len=6,
                start_pos=4,
                full_prompt=PROMPT,
            ),
        )

        assert captured["ranges"] is None

    def test_ranges_are_extracted_once_per_request(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        adapter = _BidiAdapter()
        runner = _runner(adapter)
        runner.encoder_cache.add_request("req", [_image_feature()])
        _put_encode(runner, "img-0", hidden_states=mx.ones((2, adapter.hidden_size)))
        _capture(adapter)
        calls: list[int] = []
        original = PlaceholderRange.extract_embeds_range

        def counting(self):
            calls.append(1)
            return original(self)

        monkeypatch.setattr(PlaceholderRange, "extract_embeds_range", counting)
        _forward(
            runner,
            _mm_prefill(
                "req",
                token_ids=PROMPT[:2],
                prompt_len=None,
                start_pos=0,
                full_prompt=PROMPT,
            ),
        )
        _forward(
            runner,
            _mm_prefill(
                "req",
                token_ids=PROMPT[2:],
                prompt_len=6,
                start_pos=2,
                full_prompt=PROMPT,
            ),
        )
        assert len(calls) == 1


class TestLifecycle:
    def test_state_is_dropped_on_eviction_and_resume(self) -> None:
        adapter = _BidiAdapter()
        runner = _runner(adapter)
        runner.encoder_cache.add_request("req", [_image_feature()])
        _put_encode(runner, "img-0", hidden_states=mx.ones((2, adapter.hidden_size)))
        _capture(adapter)
        _forward(
            runner,
            _mm_prefill("req", token_ids=PROMPT, prompt_len=6, full_prompt=PROMPT),
        )
        assert "req" in runner._mm_bidi_states

        runner._reconcile_request_lifecycle(set(), resumed_req_ids={"req"})
        assert "req" not in runner._mm_bidi_states

        _forward(
            runner,
            _mm_prefill("req", token_ids=PROMPT, prompt_len=6, full_prompt=PROMPT),
        )
        runner._reconcile_request_lifecycle({"req"})
        assert "req" not in runner._mm_bidi_states
