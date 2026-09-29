# SPDX-License-Identifier: Apache-2.0
"""Shared draft-model double for the draft proposer tests."""

from __future__ import annotations

import mlx.core as mx

from vllm_metal.attention.context import OffsetCache, get_context

VOCAB_SIZE = 64


class StubDraftModel:
    """mlx_lm-shaped draft model: logits per input token, recorded block tables."""

    def __init__(self) -> None:
        self.block_tables: list[list[list[int]]] = []
        self.input_lens: list[int] = []

    def __call__(self, input_ids: mx.array, *, cache: list[OffsetCache]) -> mx.array:
        ctx = get_context()
        assert ctx is not None
        self.block_tables.append([list(block_ids) for block_ids in ctx.block_tables])
        self.input_lens.append(int(input_ids.shape[1]))
        return mx.zeros((1, int(input_ids.shape[1]), VOCAB_SIZE), dtype=mx.float32)
