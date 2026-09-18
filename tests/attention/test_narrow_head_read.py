# SPDX-License-Identifier: Apache-2.0
"""Paged attention kernels reading a cache wider than the query's head_dim.

Gemma 4 allocates one cache at head_dim 512 for both its full-attention
layers (512) and its sliding layers (256); the sliding layers' K/V are
zero-padded to 512 on the scatter.  The kernels address K/V rows through the
cache's runtime strides and touch only the first ``HEAD_SIZE`` elements of a
row, so a 256-wide query against that cache must give the 256-dim attention
result -- on the tiled prefill kernel and on the decode kernel alike.  This is
what lets ``sdpa_forward`` hand the sliding layers to the far cheaper
HEAD_SIZE=256 instantiation.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from vllm_metal.metal import get_ops

BLOCK = 16
HEADS, KV_HEADS = 4, 2
DTYPE = mx.float16
ATOL, RTOL = 1.5e-2, 1e-2


def _setup(seed: int, *, n: int, seq_len: int, hd: int, cache_hd: int):
    """Query at ``hd``; cache rows ``cache_hd`` wide with real data in
    ``[0, hd)`` and zeros after, exactly as the padded scatter leaves them."""
    mx.random.seed(seed)
    num_blocks = (seq_len + BLOCK - 1) // BLOCK + 2
    k_real = mx.random.normal((num_blocks, BLOCK, KV_HEADS, hd)).astype(DTYPE)
    v_real = mx.random.normal((num_blocks, BLOCK, KV_HEADS, hd)).astype(DTYPE)
    pad = [(0, 0), (0, 0), (0, 0), (0, cache_hd - hd)]
    key_cache = mx.pad(k_real, pad)
    value_cache = mx.pad(v_real, pad)
    query = mx.random.normal((n, HEADS, hd)).astype(DTYPE)
    nblocks = (seq_len + BLOCK - 1) // BLOCK
    table = mx.array([list(range(1, nblocks + 1))], dtype=mx.int32)
    mx.eval(key_cache, value_cache, query, table)
    return key_cache, value_cache, query, table


def _kernel(query, key_cache, value_cache, table, *, kv_len, q_len, window):
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
        mx.array([kv_len], dtype=mx.int32),
        mx.array([0, q_len], dtype=mx.int32),
        BLOCK,
        kv_len,
        window if window is not None else -1,
        out,
    )
    mx.eval(out)
    return out


def _reference(query, key_cache, value_cache, table_row, *, q_lo, seq_len, window):
    hd = int(query.shape[-1])
    q = np.array(query.astype(mx.float32))
    kc = np.array(key_cache.astype(mx.float32))
    vc = np.array(value_cache.astype(mx.float32))
    k = np.stack([kc[table_row[p // BLOCK], p % BLOCK, :, :hd] for p in range(seq_len)])
    v = np.stack([vc[table_row[p // BLOCK], p % BLOCK, :, :hd] for p in range(seq_len)])
    n_rep = q.shape[1] // k.shape[1]
    k = np.repeat(k, n_rep, axis=1)
    v = np.repeat(v, n_rep, axis=1)
    n = q.shape[0]
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


@pytest.mark.parametrize("window", [None, 96])
@pytest.mark.parametrize(("hd", "cache_hd"), [(64, 128), (128, 256)])
def test_prefill_on_a_wider_cache_matches_the_narrow_reference(
    hd, cache_hd, window, force_tiled_prefill
) -> None:
    seq_len = 400
    key_cache, value_cache, query, table = _setup(
        1, n=seq_len, seq_len=seq_len, hd=hd, cache_hd=cache_hd
    )
    assert int(key_cache.shape[-1]) == cache_hd
    got = _kernel(
        query,
        key_cache,
        value_cache,
        table,
        kv_len=seq_len,
        q_len=seq_len,
        window=window,
    )
    assert int(got.shape[-1]) == hd
    ref = _reference(
        query,
        key_cache,
        value_cache,
        table[0].tolist(),
        q_lo=0,
        seq_len=seq_len,
        window=window,
    )
    np.testing.assert_allclose(np.array(got), ref, atol=ATOL, rtol=RTOL)


@pytest.mark.parametrize("window", [None, 96])
@pytest.mark.parametrize(("hd", "cache_hd"), [(64, 128), (128, 256)])
def test_decode_on_a_wider_cache_matches_the_narrow_reference(
    hd, cache_hd, window
) -> None:
    """One query token over a 400-token context: the decode kernel path."""
    seq_len = 400
    key_cache, value_cache, query, table = _setup(
        2, n=1, seq_len=seq_len, hd=hd, cache_hd=cache_hd
    )
    got = _kernel(
        query, key_cache, value_cache, table, kv_len=seq_len, q_len=1, window=window
    )
    assert int(got.shape[-1]) == hd
    ref = _reference(
        query,
        key_cache,
        value_cache,
        table[0].tolist(),
        q_lo=seq_len - 1,
        seq_len=seq_len,
        window=window,
    )
    np.testing.assert_allclose(np.array(got), ref, atol=ATOL, rtol=RTOL)
