# SPDX-License-Identifier: Apache-2.0
"""Bidirectional image-block rows against the Metal kernel and an fp32 reference."""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from vllm_metal.attention.context import PagedAttentionContext
from vllm_metal.attention.impls.bidi_prefill import (
    apply_bidirectional_segments,
    build_bidi_mask,
    gather_kv,
    slot_indices,
)
from vllm_metal.attention.impls.sdpa import _build_block_tables
from vllm_metal.metal import get_ops

BLOCK = 16
HEADS, KV_HEADS, HD = 4, 2, 64
DTYPE = mx.float16


def _setup(seed: int, *, n: int, seq_len: int, num_blocks: int = 128):
    mx.random.seed(seed)
    key_cache = mx.random.normal((num_blocks, BLOCK, KV_HEADS, HD)).astype(DTYPE)
    value_cache = mx.random.normal((num_blocks, BLOCK, KV_HEADS, HD)).astype(DTYPE)
    query = mx.random.normal((n, HEADS, HD)).astype(DTYPE)
    nblocks = (seq_len + BLOCK - 1) // BLOCK
    table = mx.array([list(range(1, nblocks + 1))], dtype=mx.int32)
    mx.eval(key_cache, value_cache, query, table)
    return key_cache, value_cache, query, table


def _kernel(query, key_cache, value_cache, table, *, n, seq_len, window):
    out = mx.array(0)
    get_ops().paged_attention_primitive(
        query,
        key_cache,
        value_cache,
        KV_HEADS,
        HD**-0.5,
        0.0,
        table,
        mx.array([seq_len], dtype=mx.int32),
        mx.array([0, n], dtype=mx.int32),
        BLOCK,
        seq_len,
        window if window is not None else -1,
        out,
    )
    mx.eval(out)
    return out


def _rows(cache: mx.array, table_row: list[int], lo: int, hi: int) -> np.ndarray:
    arr = np.array(cache.astype(mx.float32))
    return np.stack([arr[table_row[p // BLOCK], p % BLOCK] for p in range(lo, hi)])


def _ref_attention(q: np.ndarray, k: np.ndarray, v: np.ndarray, mask: np.ndarray):
    n_rep = q.shape[1] // k.shape[1]
    k = np.repeat(k, n_rep, axis=1)
    v = np.repeat(v, n_rep, axis=1)
    scores = np.einsum("qhd,khd->hqk", q, k) * HD**-0.5
    scores = np.where(mask[None], scores, -1e30)
    scores -= scores.max(axis=-1, keepdims=True)
    probs = np.exp(scores)
    probs /= probs.sum(axis=-1, keepdims=True)
    return np.einsum("hqk,khd->qhd", probs, v)


def _ctx(n: int, seq_len: int, ranges) -> PagedAttentionContext:
    return PagedAttentionContext(
        slot_mapping=[],
        cu_seqlens=[0, n],
        context_lens=[seq_len],
        segment_bidi_ranges=[ranges],
        bidi_layer_kinds=frozenset({"sliding", "full"}),
    )


@pytest.mark.parametrize("window", [None, 128])
def test_assembled_causal_path_matches_kernel(window) -> None:
    """slot_indices + gather_kv + build_bidi_mask(no block) + SDPA == kernel rows."""
    n, seq_len = 96, 600
    key_cache, value_cache, query, table = _setup(0, n=n, seq_len=seq_len)
    out = _kernel(
        query, key_cache, value_cache, table, n=n, seq_len=seq_len, window=window
    )
    q_lo = seq_len - n
    k_lo = max(0, q_lo - window + 1) if window is not None else 0
    slots = slot_indices(table[0], BLOCK, k_lo, seq_len)
    keys = gather_kv(key_cache, slots, HD)
    values = gather_kv(value_cache, slots, HD)
    mask = mx.array(build_bidi_mask(q_lo, n, k_lo, seq_len - k_lo, (0, 0), window))
    got = mx.fast.scaled_dot_product_attention(
        query.transpose(1, 0, 2)[None],
        keys.transpose(1, 0, 2)[None],
        values.transpose(1, 0, 2)[None],
        scale=HD**-0.5,
        mask=mask[None, None],
    )[0].transpose(1, 0, 2)
    mx.eval(got)
    np.testing.assert_allclose(np.array(got), np.array(out), atol=1.5e-2, rtol=1e-2)


@pytest.mark.parametrize("window", [128, None])
def test_block_rows_are_recomputed_with_the_bidirectional_mask(window) -> None:
    """A 200-row block inside a 256-row chunk, longer than the 128 window."""
    n, seq_len = 256, 600
    b0, b1 = 400, 600
    key_cache, value_cache, query, table = _setup(1, n=n, seq_len=seq_len)
    out = _kernel(
        query, key_cache, value_cache, table, n=n, seq_len=seq_len, window=window
    )
    ctx = _ctx(n, seq_len, [(b0, b1)])
    got = apply_bidirectional_segments(
        out,
        query,
        key_cache,
        value_cache,
        block_tables=table,
        block_size=BLOCK,
        cu_seqlens=[0, n],
        context_lens=[seq_len],
        ctx=ctx,
        window=window,
        scale=HD**-0.5,
        head_dim=HD,
        softcap=0.0,
        sinks=None,
        turboquant=False,
    )
    mx.eval(got)
    q_lo = seq_len - n
    k_lo = max(0, b0 - window + 1) if window is not None else 0
    row_table = table[0].tolist()
    ref = _ref_attention(
        np.array(query.astype(mx.float32))[b0 - q_lo :],
        _rows(key_cache, row_table, k_lo, b1),
        _rows(value_cache, row_table, k_lo, b1),
        build_bidi_mask(b0, b1 - b0, k_lo, b1 - k_lo, (b0, b1), window),
    )
    np.testing.assert_allclose(np.array(got)[b0 - q_lo :], ref, atol=1.5e-2, rtol=1e-2)
    # Rows before the block keep the kernel result bit-for-bit.
    np.testing.assert_array_equal(
        np.array(got)[: b0 - q_lo], np.array(out)[: b0 - q_lo]
    )
    assert ctx.bidi_logged is True


def test_block_head_in_context_is_read_from_the_cache() -> None:
    """Prefix-hit shape: the block starts below q_lo, only its tail is recomputed."""
    n, seq_len, window = 64, 300, 128
    b0, b1 = 200, 300  # q_lo = 236: 36 block rows are already in the cache
    key_cache, value_cache, query, table = _setup(2, n=n, seq_len=seq_len)
    out = _kernel(
        query, key_cache, value_cache, table, n=n, seq_len=seq_len, window=window
    )
    got = apply_bidirectional_segments(
        out,
        query,
        key_cache,
        value_cache,
        block_tables=table,
        block_size=BLOCK,
        cu_seqlens=[0, n],
        context_lens=[seq_len],
        ctx=_ctx(n, seq_len, [(b0, b1)]),
        window=window,
        scale=HD**-0.5,
        head_dim=HD,
        softcap=0.0,
        sinks=None,
        turboquant=False,
    )
    mx.eval(got)
    q_lo = seq_len - n
    k_lo = max(0, q_lo - window + 1)
    row_table = table[0].tolist()
    ref = _ref_attention(
        np.array(query.astype(mx.float32)),
        _rows(key_cache, row_table, k_lo, b1),
        _rows(value_cache, row_table, k_lo, b1),
        build_bidi_mask(q_lo, n, k_lo, b1 - k_lo, (b0, b1), window),
    )
    np.testing.assert_allclose(np.array(got), ref, atol=1.5e-2, rtol=1e-2)


def test_ranges_outside_the_queries_leave_the_output_untouched() -> None:
    n, seq_len = 32, 200
    key_cache, value_cache, query, table = _setup(3, n=n, seq_len=seq_len)
    out = _kernel(
        query, key_cache, value_cache, table, n=n, seq_len=seq_len, window=128
    )
    ctx = _ctx(n, seq_len, [(10, 50)])
    got = apply_bidirectional_segments(
        out,
        query,
        key_cache,
        value_cache,
        block_tables=table,
        block_size=BLOCK,
        cu_seqlens=[0, n],
        context_lens=[seq_len],
        ctx=ctx,
        window=128,
        scale=HD**-0.5,
        head_dim=HD,
        softcap=0.0,
        sinks=None,
        turboquant=False,
    )
    assert got is out
    assert ctx.bidi_logged is False


def test_gather_follows_hybrid_block_size_translation() -> None:
    """vLLM block 64 → kernel block 32: slots must index the reshaped view."""
    cache64 = mx.random.normal((8, 64, KV_HEADS, HD)).astype(DTYPE)
    tables, kernel_bs = _build_block_tables([[3, 5]], 64)
    view = cache64.reshape(-1, kernel_bs, KV_HEADS, HD)
    got = gather_kv(view, slot_indices(tables[0], kernel_bs, 10, 100), HD)
    expected = mx.stack([cache64[[3, 5][p // 64], p % 64] for p in range(10, 100)])
    mx.eval(got, expected)
    assert mx.array_equal(got, expected)


def test_softcap_and_turboquant_are_refused() -> None:
    n, seq_len = 8, 8
    key_cache, value_cache, query, table = _setup(4, n=n, seq_len=seq_len)
    out = mx.zeros((n, HEADS, HD), dtype=DTYPE)
    common = {
        "block_tables": table,
        "block_size": BLOCK,
        "cu_seqlens": [0, n],
        "context_lens": [seq_len],
        "ctx": _ctx(n, seq_len, [(0, 8)]),
        "window": None,
        "scale": 1.0,
        "head_dim": HD,
        "sinks": None,
    }
    with pytest.raises(NotImplementedError, match="softcap"):
        apply_bidirectional_segments(
            out, query, key_cache, value_cache, softcap=30.0, turboquant=False, **common
        )
    with pytest.raises(RuntimeError, match="TurboQuant"):
        apply_bidirectional_segments(
            out, query, key_cache, value_cache, softcap=0.0, turboquant=True, **common
        )
