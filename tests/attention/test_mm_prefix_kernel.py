# SPDX-License-Identifier: Apache-2.0
"""Tiled prefill kernel with per-row image-block ranges (mm_prefix) on Metal."""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from vllm_metal.attention.context import PagedAttentionContext
from vllm_metal.attention.impls.bidi_prefill import apply_bidirectional_segments
from vllm_metal.metal import get_ops

BLOCK = 16
HEADS, KV_HEADS = 4, 2
DTYPE = mx.float16
ATOL, RTOL = 1.5e-2, 1e-2


def _setup(seed: int, *, n: int, seq_len: int, hd: int = 64, num_blocks: int = 128):
    mx.random.seed(seed)
    key_cache = mx.random.normal((num_blocks, BLOCK, KV_HEADS, hd)).astype(DTYPE)
    value_cache = mx.random.normal((num_blocks, BLOCK, KV_HEADS, hd)).astype(DTYPE)
    query = mx.random.normal((n, HEADS, hd)).astype(DTYPE)
    nblocks = (seq_len + BLOCK - 1) // BLOCK
    table = mx.array([list(range(1, nblocks + 1))], dtype=mx.int32)
    mx.eval(key_cache, value_cache, query, table)
    return key_cache, value_cache, query, table


def _kernel(
    query,
    key_cache,
    value_cache,
    table,
    *,
    kv_lens: list[int],
    cu_seqlens_q: list[int],
    window: int | None,
    ranges: mx.array | None = None,
    block_size: int = BLOCK,
    **kwargs,
):
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
        block_size,
        max(kv_lens),
        window if window is not None else -1,
        out,
        mm_prefix_ranges=ranges,
        **kwargs,
    )
    mx.eval(out)
    return out


def _range_rows(cu_seqlens, context_lens, blocks_per_segment) -> mx.array:
    """Inclusive ``(L, 2)`` int32 rows from half-open blocks, ``(-1, -1)`` elsewhere.

    Built inline so the kernel tests do not depend on the host module.
    """
    rows = np.full((cu_seqlens[-1], 2), -1, dtype=np.int32)
    for i, blocks in enumerate(blocks_per_segment):
        n = cu_seqlens[i + 1] - cu_seqlens[i]
        q_lo = context_lens[i] - n
        for b0, b1 in blocks or ():
            a, b = max(b0, q_lo), min(b1, q_lo + n)
            if b > a:
                rows[cu_seqlens[i] + a - q_lo : cu_seqlens[i] + b - q_lo] = (
                    b0,
                    b1 - 1,
                )
    return mx.array(rows)


def _hf_mask(q_lo: int, n: int, seq_len: int, blocks, window: int | None):
    """``(n, seq_len)`` bool: ``(causal OR same block) AND window`` over all keys."""
    q = np.arange(q_lo, q_lo + n)[:, None]
    k = np.arange(seq_len)[None, :]
    allowed = k <= q
    for b0, b1 in blocks:
        allowed |= ((q >= b0) & (q < b1)) & ((k >= b0) & (k < b1))
    if window is not None:
        allowed &= (q - k) < window
    return allowed


def _rows(cache: mx.array, table_row: list[int], lo: int, hi: int) -> np.ndarray:
    arr = np.array(cache.astype(mx.float32))
    return np.stack([arr[table_row[p // BLOCK], p % BLOCK] for p in range(lo, hi)])


def _ref_attention(q, k, v, mask):
    hd = q.shape[-1]
    n_rep = q.shape[1] // k.shape[1]
    k = np.repeat(k, n_rep, axis=1)
    v = np.repeat(v, n_rep, axis=1)
    scores = np.einsum("qhd,khd->hqk", q, k) * hd**-0.5
    scores = np.where(mask[None], scores, -1e30)
    scores -= scores.max(axis=-1, keepdims=True)
    probs = np.exp(scores)
    probs /= probs.sum(axis=-1, keepdims=True)
    return np.einsum("hqk,khd->qhd", probs, v)


def _reference(
    query, key_cache, value_cache, table_row, *, q_lo, seq_len, blocks, window
):
    """fp32 attention of ``query`` rows ``[q_lo, q_lo + n)`` over keys ``[0, seq_len)``."""
    n = int(query.shape[0])
    return _ref_attention(
        np.array(query.astype(mx.float32)),
        _rows(key_cache, table_row, 0, seq_len),
        _rows(value_cache, table_row, 0, seq_len),
        _hf_mask(q_lo, n, seq_len, blocks, window),
    )


def test_supports_mm_prefix() -> None:
    assert get_ops().supports_mm_prefix() is True


@pytest.mark.parametrize("window", [128, None])
def test_block_longer_than_window_matches_reference_and_recompute(
    window, force_tiled_prefill
) -> None:
    """A 200-row block in a 256-row chunk; its tail lies right of the causal frontier.

    Without the extended tile-loop exit the kernel never visits the block's
    last tiles (error 0.05–0.29 against atol 0.015).  ``force_tiled_prefill``
    keeps the ranges-free baseline on the tiled kernel too: on an M5 it would
    otherwise run on NAX, and the bit-for-bit comparison below would compare
    two different kernels.
    """
    n, seq_len = 256, 600
    block = (400, 600)
    key_cache, value_cache, query, table = _setup(1, n=n, seq_len=seq_len)
    common = {"kv_lens": [seq_len], "cu_seqlens_q": [0, n], "window": window}
    plain = _kernel(query, key_cache, value_cache, table, **common)
    ranges = _range_rows([0, n], [seq_len], [[block]])
    got = _kernel(query, key_cache, value_cache, table, ranges=ranges, **common)
    q_lo = seq_len - n
    ref = _reference(
        query,
        key_cache,
        value_cache,
        table[0].tolist(),
        q_lo=q_lo,
        seq_len=seq_len,
        blocks=[block],
        window=window,
    )
    got_np, plain_np = np.array(got), np.array(plain)
    # (б) fp32 reference with HF semantics.
    np.testing.assert_allclose(got_np, ref, atol=ATOL, rtol=RTOL)
    # (а) the phase-2 recompute over the same cache agrees.
    ctx = PagedAttentionContext(
        slot_mapping=[],
        cu_seqlens=[0, n],
        context_lens=[seq_len],
        segment_bidi_ranges=[[block]],
        bidi_layer_kinds=frozenset({"sliding", "full"}),
    )
    recomputed = apply_bidirectional_segments(
        plain,
        query,
        key_cache,
        value_cache,
        block_tables=table,
        block_size=BLOCK,
        cu_seqlens=[0, n],
        context_lens=[seq_len],
        ctx=ctx,
        window=window,
        scale=64**-0.5,
        head_dim=64,
        softcap=0.0,
        sinks=None,
        turboquant=False,
    )
    mx.eval(recomputed)
    np.testing.assert_allclose(got_np, np.array(recomputed), atol=ATOL, rtol=RTOL)
    # (г) rows before the block are bit-for-bit the plain kernel's.
    split = block[0] - q_lo
    np.testing.assert_array_equal(got_np[:split], plain_np[:split])
    # (е) the block rows really changed; a silently ignored buffer passes (г) alone.
    assert np.abs(got_np[split:] - plain_np[split:]).max() > 0.1


def test_ranges_are_validated_before_dispatch() -> None:
    n, seq_len = 32, 64
    key_cache, value_cache, query, table = _setup(2, n=n, seq_len=seq_len)
    common = {"kv_lens": [seq_len], "cu_seqlens_q": [0, n], "window": None}
    ranges = _range_rows([0, n], [seq_len], [[(40, 64)]])
    with pytest.raises(ValueError, match="mm_prefix_ranges must be int32"):
        _kernel(
            query,
            key_cache,
            value_cache,
            table,
            ranges=ranges.astype(mx.int64),
            **common,
        )
    with pytest.raises(ValueError, match=r"shape \(query rows, 2\)"):
        _kernel(
            query,
            key_cache,
            value_cache,
            table,
            ranges=mx.concatenate([ranges, mx.zeros((1, 2), dtype=mx.int32)]),
            **common,
        )
    with pytest.raises(ValueError, match="tiled prefill kernel"):
        _kernel(
            query.astype(mx.float32),
            key_cache,
            value_cache,
            table,
            ranges=ranges,
            **common,
        )
    with pytest.raises(ValueError, match="tiled prefill kernel"):
        _kernel(
            query,
            key_cache,
            value_cache,
            table,
            ranges=ranges,
            window_seqlen_q=n,
            **common,
        )
    # A pure-decode batch (one query row per segment) never reaches the tiled kernel.
    decode_ranges = _range_rows([0, 1], [seq_len], [[(seq_len - 1, seq_len)]])
    with pytest.raises(ValueError, match="tiled prefill kernel"):
        _kernel(
            query[:1],
            key_cache,
            value_cache,
            table,
            ranges=decode_ranges,
            kv_lens=[seq_len],
            cu_seqlens_q=[0, 1],
            window=None,
        )
