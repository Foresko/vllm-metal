# SPDX-License-Identifier: Apache-2.0
"""Tiled prefill kernel and the sliding window: skip KV tiles left of the window.

Gemma 4 runs 25 of 30 layers with ``sliding_window=1024``.  A kernel that
only *masks* keys outside the window still visits every KV tile of the
sequence, so those layers cost O(n^2) like full attention, and prefill of a
32K prompt took 108 s on an M3 Ultra (docs in foresko-inference,
``docs/reviews/2026-09-17-prefill-window-and-sliding-kernel.md``).  The NAX
kernel already starts its tile loop at the window; these tests pin the same
contract on the tiled kernel: results match the fp32 reference for every
chunking of the prefill, and the windowed kernel really does less work than
the full-attention one on a long sequence.
"""

from __future__ import annotations

import time

import mlx.core as mx
import numpy as np
import pytest

from vllm_metal.metal import get_ops

BLOCK = 16
HEADS, KV_HEADS = 4, 2
DTYPE = mx.float16
ATOL, RTOL = 1.5e-2, 1e-2


def _setup(seed: int, *, n: int, seq_len: int, hd: int = 64):
    mx.random.seed(seed)
    num_blocks = (seq_len + BLOCK - 1) // BLOCK + 2
    key_cache = mx.random.normal((num_blocks, BLOCK, KV_HEADS, hd)).astype(DTYPE)
    value_cache = mx.random.normal((num_blocks, BLOCK, KV_HEADS, hd)).astype(DTYPE)
    query = mx.random.normal((n, HEADS, hd)).astype(DTYPE)
    nblocks = (seq_len + BLOCK - 1) // BLOCK
    table = mx.array([list(range(1, nblocks + 1))], dtype=mx.int32)
    mx.eval(key_cache, value_cache, query, table)
    return key_cache, value_cache, query, table


def _kernel(query, key_cache, value_cache, table, *, kv_lens, cu_seqlens_q, window):
    hd = int(query.shape[-1])
    out = mx.array(0)
    get_ops().paged_attention_primitive(
        query,
        key_cache,
        value_cache,
        KV_HEADS,
        hd**-0.5,
        0.0,
        table,
        mx.array(kv_lens, dtype=mx.int32),
        mx.array(cu_seqlens_q, dtype=mx.int32),
        BLOCK,
        max(kv_lens),
        window if window is not None else -1,
        out,
    )
    mx.eval(out)
    return out


def _rows(cache: mx.array, table_row: list[int], lo: int, hi: int) -> np.ndarray:
    arr = np.array(cache.astype(mx.float32))
    return np.stack([arr[table_row[p // BLOCK], p % BLOCK] for p in range(lo, hi)])


def _reference(query, key_cache, value_cache, table_row, *, q_lo, seq_len, window):
    """fp32 attention of rows ``[q_lo, q_lo + n)`` over keys ``[0, seq_len)``
    with the causal mask and, when set, the sliding window ``q - k < window``."""
    q = np.array(query.astype(mx.float32))
    k = _rows(key_cache, table_row, 0, seq_len)
    v = _rows(value_cache, table_row, 0, seq_len)
    n = q.shape[0]
    hd = q.shape[-1]
    n_rep = q.shape[1] // k.shape[1]
    k = np.repeat(k, n_rep, axis=1)
    v = np.repeat(v, n_rep, axis=1)
    qi = np.arange(q_lo, q_lo + n)[:, None]
    ki = np.arange(seq_len)[None, :]
    allowed = ki <= qi
    if window is not None:
        allowed &= (qi - ki) < window
    scores = np.einsum("qhd,khd->hqk", q, k) * hd**-0.5
    scores = np.where(allowed[None], scores, -1e30)
    m = scores.max(axis=-1, keepdims=True)
    probs = np.exp(scores - m)
    probs /= probs.sum(axis=-1, keepdims=True)
    return np.einsum("hqk,khd->qhd", probs, v)


@pytest.mark.parametrize("window", [64, 100, 256])
@pytest.mark.parametrize(
    ("seq_len", "chunks"),
    [
        # Whole prompt in one pass: the first tiles of late Q blocks are fully
        # left of every row's window.
        (640, [640]),
        # Chunked prefill: context_len > 0, the window starts inside the
        # context for every row of the chunk.
        (640, [320, 320]),
        # Chunk boundaries that are not tile-aligned, plus a short last chunk.
        (700, [300, 250, 150]),
        # Window wider than the chunk but narrower than the sequence.
        (1200, [600, 600]),
    ],
)
def test_windowed_prefill_matches_reference_for_every_chunking(
    window, seq_len, chunks, force_tiled_prefill
) -> None:
    key_cache, value_cache, query, table = _setup(3, n=seq_len, seq_len=seq_len)
    table_row = table[0].tolist()
    done = 0
    for n in chunks:
        q_lo = done
        kv_len = done + n
        chunk_query = query[q_lo:kv_len]
        got = _kernel(
            chunk_query,
            key_cache,
            value_cache,
            table,
            kv_lens=[kv_len],
            cu_seqlens_q=[0, n],
            window=window,
        )
        ref = _reference(
            chunk_query,
            key_cache,
            value_cache,
            table_row,
            q_lo=q_lo,
            seq_len=kv_len,
            window=window,
        )
        np.testing.assert_allclose(np.array(got), ref, atol=ATOL, rtol=RTOL)
        done = kv_len


def test_windowed_prefill_matches_reference_across_varlen_sequences(
    force_tiled_prefill,
) -> None:
    """Two sequences in one call, different lengths and contexts; the second
    starts at a Q block that is not the first of the grid."""
    window = 96
    lens = [(400, 250), (900, 300)]  # (kv_len, query_len) per sequence
    caches = [_setup(10 + i, n=q, seq_len=kv) for i, (kv, q) in enumerate(lens)]
    # One shared cache: place both sequences' blocks in the first sequence's
    # tensors by using separate rows of one block table.
    key_cache = mx.concatenate([caches[0][0], caches[1][0]], axis=0)
    value_cache = mx.concatenate([caches[0][1], caches[1][1]], axis=0)
    offset = int(caches[0][0].shape[0])
    rows = []
    max_blocks = max((kv + BLOCK - 1) // BLOCK for kv, _ in lens)
    for i, (kv, _) in enumerate(lens):
        nb = (kv + BLOCK - 1) // BLOCK
        row = [(offset if i else 0) + b for b in range(1, nb + 1)]
        rows.append(row + [0] * (max_blocks - nb))
    table = mx.array(rows, dtype=mx.int32)
    query = mx.concatenate([caches[0][2], caches[1][2]], axis=0)
    mx.eval(key_cache, value_cache, table, query)
    got = np.array(
        _kernel(
            query,
            key_cache,
            value_cache,
            table,
            kv_lens=[kv for kv, _ in lens],
            cu_seqlens_q=[0, lens[0][1], lens[0][1] + lens[1][1]],
            window=window,
        )
    )
    start = 0
    for i, (kv, q) in enumerate(lens):
        ref = _reference(
            query[start : start + q],
            key_cache,
            value_cache,
            rows[i],
            q_lo=kv - q,
            seq_len=kv,
            window=window,
        )
        np.testing.assert_allclose(got[start : start + q], ref, atol=ATOL, rtol=RTOL)
        start += q


def _median_seconds(fn, *, repeats: int = 5) -> float:
    fn()  # warm the pipeline
    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    return sorted(samples)[len(samples) // 2]


@pytest.mark.slow
def test_windowed_prefill_does_less_work_than_full_attention(force_tiled_prefill) -> None:
    """On an 8K sequence a 1024 window touches at most ~1/4 of the KV tiles a
    full-attention pass touches (causal triangle vs. band), so the windowed
    kernel must be clearly faster.  A kernel that only masks runs the same
    tile loop and lands near 1.0x; the bound is loose so it holds on any
    Apple GPU."""
    seq_len, window = 8192, 1024
    key_cache, value_cache, query, table = _setup(5, n=seq_len, seq_len=seq_len)
    common = {"kv_lens": [seq_len], "cu_seqlens_q": [0, seq_len]}
    full = _median_seconds(
        lambda: _kernel(query, key_cache, value_cache, table, window=None, **common)
    )
    windowed = _median_seconds(
        lambda: _kernel(query, key_cache, value_cache, table, window=window, **common)
    )
    assert full / windowed >= 2.0, (
        f"windowed prefill {windowed * 1e3:.1f} ms vs full {full * 1e3:.1f} ms: "
        "the tile loop is not bounded by the window"
    )
