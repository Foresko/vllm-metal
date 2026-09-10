# SPDX-License-Identifier: Apache-2.0
"""``build_mm_prefix_rows`` against vLLM's ``fill_mm_prefix_query_ranges``."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from vllm.v1.attention.backends.utils import fill_mm_prefix_query_ranges

from vllm_metal.attention.impls.mm_prefix import (
    MM_PREFIX_PATHS,
    build_mm_prefix_rows,
    resolve_mm_prefix_path,
)


def _random_batch(
    rng: np.random.Generator,
) -> tuple[list[int], list[int], list[list[tuple[int, int]] | None]]:
    cu, context_lens, ranges = [0], [], []
    for _ in range(int(rng.integers(1, 5))):
        if rng.random() < 0.3:  # decode segment
            n, seq_len, seg_ranges = 1, int(rng.integers(1, 300)), None
        else:
            n = int(rng.integers(2, 200))
            seq_len = n + int(rng.integers(0, 200))
            seg_ranges = []
            cursor = max(0, seq_len - n - 100)  # blocks may start in the context
            for _ in range(int(rng.integers(0, 4))):
                start = cursor + int(rng.integers(0, 40))
                length = int(rng.integers(2, 80))  # never a one-token block
                seg_ranges.append((start, start + length))
                cursor = start + length + 1
        cu.append(cu[-1] + n)
        context_lens.append(seq_len)
        ranges.append(seg_ranges)
    return cu, context_lens, ranges


@pytest.mark.parametrize("seed", range(40))
def test_matches_vllm_fill_mm_prefix_query_ranges(seed: int) -> None:
    rng = np.random.default_rng(seed)
    cu, context_lens, ranges = _random_batch(rng)
    got = build_mm_prefix_rows(cu, context_lens, ranges)
    expected = np.full((cu[-1], 2), -1, dtype=np.int32)
    # vLLM's ranges are inclusive on both ends; ours are half-open.
    vllm_ranges = {
        i: [(r0, r1 - 1) for r0, r1 in segs] for i, segs in enumerate(ranges) if segs
    }
    written = fill_mm_prefix_query_ranges(
        expected,
        vllm_ranges,
        torch.tensor(cu, dtype=torch.int32),
        torch.tensor(context_lens, dtype=torch.int32),
    )
    if written == 0:
        assert got is None
    else:
        assert got is not None and got.dtype == np.int32
        np.testing.assert_array_equal(got, expected)


def test_none_when_no_row_lies_in_a_block() -> None:
    assert build_mm_prefix_rows([0, 4], [10], [[(0, 3)]]) is None  # block in context
    assert build_mm_prefix_rows([0, 4], [10], [[(12, 20)]]) is None  # beyond the chunk
    assert build_mm_prefix_rows([0, 1, 5], [7, 10], [None, None]) is None
    assert build_mm_prefix_rows([0, 1, 5], [7, 10], [None, []]) is None


def test_degenerate_ranges_are_ignored() -> None:
    assert build_mm_prefix_rows([0, 4], [4], [[(2, 2), (3, 1)]]) is None


def test_block_head_in_context_is_clipped_to_the_chunk_rows() -> None:
    rows = build_mm_prefix_rows([0, 4], [10], [[(4, 8)]])  # q_lo = 6: rows 6, 7
    assert rows is not None
    assert rows.tolist() == [[4, 7], [4, 7], [-1, -1], [-1, -1]]


def test_one_token_block_yields_an_inclusive_pair() -> None:
    """Our contract keeps a 1-token block; vLLM would drop the inclusive (p, p)."""
    rows = build_mm_prefix_rows([0, 3], [3], [[(1, 2)]])
    assert rows is not None
    assert rows.tolist() == [[-1, -1], [1, 1], [-1, -1]]


def test_decode_rows_and_text_segments_stay_minus_one() -> None:
    rows = build_mm_prefix_rows([0, 1, 4], [20, 6], [None, [(3, 6)]])  # q_lo = 3
    assert rows is not None
    assert rows.tolist() == [[-1, -1], [3, 5], [3, 5], [3, 5]]


@pytest.mark.parametrize(
    ("value", "supported", "expected"),
    [
        (None, True, "kernel"),
        (None, False, "recompute"),
        ("kernel", True, "kernel"),
        ("kernel", False, "recompute"),
        ("recompute", True, "recompute"),
        ("recompute", False, "recompute"),
    ],
)
def test_resolve_mm_prefix_path(value, supported, expected) -> None:
    assert resolve_mm_prefix_path(value, supported) == expected


def test_unknown_path_is_rejected() -> None:
    assert MM_PREFIX_PATHS == ("kernel", "recompute")
    with pytest.raises(ValueError, match="VLLM_METAL_MM_PREFIX_PATH"):
        resolve_mm_prefix_path("kernle", True)
