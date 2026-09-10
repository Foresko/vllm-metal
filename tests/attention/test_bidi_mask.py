# SPDX-License-Identifier: Apache-2.0
"""Bidirectional image-block mask helpers against a brute-force reference."""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from vllm_metal.attention.impls.bidi_prefill import (
    build_bidi_mask,
    gather_kv,
    intersecting_ranges,
    slot_indices,
)


def _reference(q_lo, n, k_lo, num_keys, block, window):
    out = np.zeros((n, num_keys), dtype=bool)
    for i in range(n):
        q = q_lo + i
        for j in range(num_keys):
            k = k_lo + j
            allowed = k <= q or block[0] <= k < block[1]
            if window is not None:
                allowed = allowed and (q - k) < window
            out[i, j] = allowed
    return out


@pytest.mark.parametrize("window", [None, 1024, 8])
@pytest.mark.parametrize(
    "q_lo, n, k_lo, block",
    [
        (0, 12, 0, (3, 9)),  # block inside the chunk
        (20, 10, 5, (24, 28)),  # context below q_lo, block inside
        (20, 10, 0, (15, 24)),  # block started before q_lo (prefix hit)
        (20, 10, 0, (2, 6)),  # block entirely in the context
        (100, 300, 0, (100, 400)),  # block longer than the window
    ],
)
def test_mask_matches_brute_force(q_lo, n, k_lo, block, window) -> None:
    num_keys = q_lo + n - k_lo
    got = build_bidi_mask(q_lo, n, k_lo, num_keys, block, window)
    assert got.dtype == bool and got.shape == (n, num_keys)
    np.testing.assert_array_equal(
        got, _reference(q_lo, n, k_lo, num_keys, block, window)
    )


def test_mask_without_block_is_causal_and_windowed() -> None:
    got = build_bidi_mask(4, 4, 0, 8, (0, 0), 3)
    assert got.tolist() == [
        [False, False, True, True, True, False, False, False],
        [False, False, False, True, True, True, False, False],
        [False, False, False, False, True, True, True, False],
        [False, False, False, False, False, True, True, True],
    ]


def test_intersecting_ranges_keeps_only_ranges_touching_the_queries() -> None:
    ranges = [(0, 4), (4, 10), (10, 12), (12, 20)]
    assert intersecting_ranges(5, 12, ranges) == [(4, 10), (10, 12)]
    assert intersecting_ranges(20, 25, ranges) == []
    assert intersecting_ranges(0, 1, [(0, 1)]) == [(0, 1)]


def test_slot_indices_follow_the_block_table() -> None:
    table = mx.array([7, 3, 9], dtype=mx.int32)
    slots = slot_indices(table, 4, 2, 11)
    assert slots.tolist() == [7 * 4 + 2, 7 * 4 + 3, 12, 13, 14, 15, 36, 37, 38]


def test_gather_kv_reads_rows_and_slices_head_dim() -> None:
    cache = mx.arange(3 * 4 * 2 * 6, dtype=mx.float32).reshape(3, 4, 2, 6)
    slots = mx.array([5, 0, 11], dtype=mx.int32)
    got = gather_kv(cache, slots, 4)
    flat = cache.reshape(-1, 2, 6)
    expected = mx.stack([flat[5], flat[0], flat[11]])[:, :, :4]
    assert got.shape == (3, 2, 4)
    assert mx.array_equal(got, expected)
