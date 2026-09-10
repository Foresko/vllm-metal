# SPDX-License-Identifier: Apache-2.0
"""Bidirectional attention inside image blocks for prefill segments.

Gemma 4 (HF ``create_masks_for_vision_model``) lets the soft tokens of one
image attend to each other on sliding-window layers: a query row inside an
image block ``[b0, b1)`` may see key ``k`` iff
``(k <= q or b0 <= k < b1) and (q - k < window)``.  The Metal paged kernel
only knows the causal + window mask, so the rows of image blocks are
recomputed here with ``mx.fast.scaled_dot_product_attention`` over K/V
gathered from the paged cache, and spliced into the kernel output.  Text
rows, decode rows and full-attention layers never enter this module.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
from vllm.logger import init_logger

logger = init_logger(__name__)


def intersecting_ranges(
    q_lo: int, q_hi: int, ranges: list[tuple[int, int]]
) -> list[tuple[int, int]]:
    """Ranges (half-open, absolute) containing at least one query in [q_lo, q_hi)."""
    return [(start, end) for start, end in ranges if start < q_hi and end > q_lo]


def build_bidi_mask(
    q_lo: int,
    n: int,
    k_lo: int,
    num_keys: int,
    block: tuple[int, int],
    window: int | None,
) -> np.ndarray:
    """``(n, num_keys)`` bool mask for query rows ``[q_lo, q_lo + n)`` of one block.

    Keys are absolute positions ``[k_lo, k_lo + num_keys)``.  ``(k <= q or
    b0 <= k < b1) and (q - k < window)`` — HF's ``(causal OR blockwise) AND
    sliding_window`` for rows that lie inside the block.
    """
    q = np.arange(q_lo, q_lo + n, dtype=np.int64)[:, None]
    k = np.arange(k_lo, k_lo + num_keys, dtype=np.int64)[None, :]
    b0, b1 = block
    allowed = (k <= q) | ((k >= b0) & (k < b1))
    if window is not None:
        allowed &= (q - k) < window
    return allowed


def slot_indices(
    block_table_row: mx.array, block_size: int, k_lo: int, k_hi: int
) -> mx.array:
    """Flat cache rows for absolute positions ``[k_lo, k_hi)``.

    ``bt[p // bs] * bs + p % bs`` with MLX ops on the device-side block table
    row (kernel format, kernel block size) — no host sync.
    """
    positions = mx.arange(k_lo, k_hi, dtype=mx.int32)
    blocks = block_table_row[positions // block_size]
    return blocks * block_size + positions % block_size


def gather_kv(cache: mx.array, slots: mx.array, head_dim: int) -> mx.array:
    """Rows ``slots`` of a ``(num_blocks, block_size, kv_heads, cache_hd)`` cache.

    Returns ``(len(slots), kv_heads, head_dim)``: the cache's zero-padded tail
    beyond the layer's real ``head_dim`` is sliced off.
    """
    flat = cache.reshape(-1, cache.shape[-2], cache.shape[-1])
    return mx.take(flat, slots, axis=0)[:, :, :head_dim]
