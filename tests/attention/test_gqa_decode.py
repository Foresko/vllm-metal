# SPDX-License-Identifier: Apache-2.0
"""GQA-packed paged decode kernel (paged_attention_gqa_decode).

The per-token decode kernel runs one threadgroup per query head, so a KV
head shared by G query heads is streamed G times: 8x the KV traffic for
Gemma 4's full-attention layers (16 query / 2 KV heads, head_dim 512).  The
GQA kernel serves up to 8 query heads of one KV head per threadgroup and
reads each KV head once.  These tests pin it against an fp32 reference that
never repeats K/V, prove routing through the native dispatch counters, and
use adversarial inputs that random data cannot see.
"""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import numpy as np
import pytest

from vllm_metal.metal import get_ops

# Unit roundoff of the output type: tolerances scale with the data.
EPS = {mx.float16: 2.0**-11, mx.bfloat16: 2.0**-8}
RTOL = 1e-2
GEMMA_FULL = {"heads": 16, "kv_heads": 2, "hd": 512}


@dataclass
class Case:
    query: mx.array  # [num_seqs, num_heads, head_dim]
    key_cache: mx.array  # [num_blocks, block, kv_heads, head_dim]
    value_cache: mx.array
    tables: np.ndarray  # [num_seqs, max_blocks] int32
    kv_lens: list[int]
    kv_heads: int
    block: int


def _make_case(
    *,
    kv_lens: list[int],
    heads: int,
    kv_heads: int,
    hd: int,
    dtype,
    block: int = 16,
    seed: int = 0,
) -> Case:
    """Random decode batch with shuffled physical blocks; block 0 unused."""
    rng = np.random.default_rng(seed)
    blocks_per_seq = [(n + block - 1) // block for n in kv_lens]
    num_blocks = sum(blocks_per_seq) + 1
    perm = rng.permutation(np.arange(1, num_blocks))
    tables = np.zeros((len(kv_lens), max(blocks_per_seq)), dtype=np.int32)
    offset = 0
    for i, nb in enumerate(blocks_per_seq):
        tables[i, :nb] = perm[offset : offset + nb]
        offset += nb
    shape = (num_blocks, block, kv_heads, hd)
    key = rng.standard_normal(shape, dtype=np.float32)
    value = rng.standard_normal(shape, dtype=np.float32)
    query = rng.standard_normal((len(kv_lens), heads, hd), dtype=np.float32)
    return Case(
        mx.array(query).astype(dtype),
        mx.array(key).astype(dtype),
        mx.array(value).astype(dtype),
        tables,
        list(kv_lens),
        kv_heads,
        block,
    )


def _counts() -> tuple[int, int]:
    single, partitioned = get_ops().gqa_decode_dispatch_counts()
    return int(single), int(partitioned)


def _run(
    case: Case,
    *,
    scale: float,
    window: int | None = None,
    sinks: np.ndarray | None = None,
    softcap: float = 0.0,
    max_seq_len: int | None = None,
    expect: str | None = None,
) -> np.ndarray:
    """One decode step through paged_attention_primitive, as fp32 numpy.

    ``expect="gqa"`` asserts that exactly one GQA decode dispatch ran,
    ``expect="per_token"`` that none did.
    """
    before = sum(_counts())
    out = mx.array(0)
    kwargs = {}
    if sinks is not None:
        kwargs["sinks"] = mx.array(sinks, dtype=mx.float32)
    num_seqs = len(case.kv_lens)
    get_ops().paged_attention_primitive(
        case.query,
        case.key_cache,
        case.value_cache,
        case.kv_heads,
        scale,
        softcap,
        mx.array(case.tables),
        mx.array(case.kv_lens, dtype=mx.int32),
        mx.array(list(range(num_seqs + 1)), dtype=mx.int32),
        case.block,
        max_seq_len or max(case.kv_lens),
        -1 if window is None else window,
        out,
        **kwargs,
    )
    mx.eval(out)
    ran = sum(_counts()) - before
    if expect == "gqa":
        assert ran == 1, f"expected one GQA decode dispatch, got {ran}"
    elif expect == "per_token":
        assert ran == 0, f"expected the per-token kernel, got {ran} GQA dispatches"
    return np.array(out.astype(mx.float32))


def _reference(
    case: Case,
    *,
    scale: float,
    window: int | None = None,
    sinks: np.ndarray | None = None,
    softcap: float = 0.0,
) -> np.ndarray:
    """fp32 attention from the (already rounded) inputs, grouped by KV head."""
    q = np.array(case.query.astype(mx.float32))
    kc = np.array(case.key_cache.astype(mx.float32))
    vc = np.array(case.value_cache.astype(mx.float32))
    _, heads, hd = q.shape
    g = heads // case.kv_heads
    out = np.zeros_like(q)
    for i, n in enumerate(case.kv_lens):
        pos = np.arange(n)
        blocks = case.tables[i, pos // case.block]
        k = kc[blocks, pos % case.block]  # [n, kv_heads, hd]
        v = vc[blocks, pos % case.block]
        lo = 0 if window is None else max(0, n - window)
        k, v = k[lo:], v[lo:]
        qg = q[i].reshape(case.kv_heads, g, hd)
        s = np.einsum("kgd,nkd->kgn", qg, k) * scale
        if softcap > 0:
            s = softcap * np.tanh(s / softcap)
        m = s.max(axis=-1, keepdims=True)
        if sinks is not None:
            sk = np.asarray(sinks, np.float32).reshape(case.kv_heads, g, 1)
            m = np.maximum(m, sk)
        p = np.exp(s - m)
        denom = p.sum(axis=-1, keepdims=True)
        if sinks is not None:
            denom = denom + np.exp(sk - m)
        out[i] = (np.einsum("kgn,nkd->kgd", p, v) / denom).reshape(heads, hd)
    return out


def _assert_close(got: np.ndarray, ref: np.ndarray, dtype) -> None:
    atol = 8 * EPS[dtype] * float(np.abs(ref).max())
    np.testing.assert_allclose(got, ref, atol=atol, rtol=RTOL)


def _reset(ops) -> None:
    ops.set_gqa_decode_enabled(True)
    ops.set_gqa_decode_force(False)
    ops.set_gqa_decode_partition_size(-1)


@pytest.fixture(params=["per_token", "gqa"])
def decode_path(request):
    """Run a test once on the per-token kernel and once on the forced GQA kernel."""
    ops = get_ops()
    if request.param == "per_token":
        ops.set_gqa_decode_enabled(False)
    else:
        ops.set_gqa_decode_force(True)
    try:
        yield request.param
    finally:
        _reset(ops)


@pytest.fixture
def force_gqa():
    ops = get_ops()
    ops.set_gqa_decode_force(True)
    try:
        yield ops
    finally:
        _reset(ops)


# Window starts (seq_len - 1024) at 2047, 2048, 2049: residues 511, 0, 1 mod 512.
CONTEXTS = [100, 511, 512, 513, 700, 1500, 3071, 3072, 3073, 8192]


@pytest.mark.parametrize("window", [None, 1024])
@pytest.mark.parametrize("n", CONTEXTS)
def test_gemma_full_layer_matches_reference(decode_path, n, window) -> None:
    case = _make_case(kv_lens=[n], dtype=mx.bfloat16, seed=n, **GEMMA_FULL)
    scale = 512**-0.5
    got = _run(case, scale=scale, window=window, expect=decode_path)
    _assert_close(got, _reference(case, scale=scale, window=window), mx.bfloat16)


def test_routing_follows_the_table_and_the_switches() -> None:
    ops = get_ops()
    full = _make_case(kv_lens=[2048], dtype=mx.bfloat16, **GEMMA_FULL)
    sliding = _make_case(
        kv_lens=[2048], heads=16, kv_heads=8, hd=256, dtype=mx.bfloat16
    )
    mha = _make_case(kv_lens=[600], heads=16, kv_heads=16, hd=512, dtype=mx.bfloat16)
    hd64 = _make_case(kv_lens=[600], heads=16, kv_heads=2, hd=64, dtype=mx.bfloat16)
    fp32 = _make_case(kv_lens=[600], heads=16, kv_heads=2, hd=512, dtype=mx.float32)
    try:
        _run(full, scale=512**-0.5, expect="gqa")  # (512, 8): measured win
        _run(sliding, scale=256**-0.5, expect="per_token")  # (256, 2): not measured
        ops.set_gqa_decode_enabled(False)
        _run(full, scale=512**-0.5, expect="per_token")  # kill switch
        ops.set_gqa_decode_enabled(True)
        ops.set_gqa_decode_force(True)
        _run(sliding, scale=256**-0.5, expect="gqa")  # force: any supported shape
        _run(mha, scale=512**-0.5, expect="per_token")  # G = 1: unsupported
        _run(hd64, scale=64**-0.5, expect="per_token")  # head_dim 64: unsupported
        _run(fp32, scale=512**-0.5, expect="per_token")  # fp32: unsupported
    finally:
        _reset(ops)


def test_single_pass_and_partitions_agree(force_gqa) -> None:
    case = _make_case(kv_lens=[5000], dtype=mx.bfloat16, seed=23, **GEMMA_FULL)
    outs = {}
    for size in (0, 256, 512):
        force_gqa.set_gqa_decode_partition_size(size)
        outs[size] = _run(case, scale=512**-0.5, expect="gqa")
    _assert_close(outs[256], outs[0], mx.bfloat16)
    _assert_close(outs[512], outs[0], mx.bfloat16)


def test_automatic_split_follows_the_grid_threshold(force_gqa) -> None:
    # 16 query / 2 KV heads: two head groups per token.
    tokens_above = -(-force_gqa.gqa_decode_min_grid() // 2)
    big = _make_case(
        kv_lens=[3000] + [700] * (tokens_above - 1),
        dtype=mx.bfloat16,
        seed=29,
        **GEMMA_FULL,
    )
    single = Case(
        big.query[:1],
        big.key_cache,
        big.value_cache,
        big.tables[:1],
        big.kv_lens[:1],
        big.kv_heads,
        big.block,
    )
    s0, p0 = _counts()
    out_single = _run(single, scale=512**-0.5)
    s1, p1 = _counts()
    assert (s1 - s0, p1 - p0) == (0, 1), "one token is below the threshold: split"
    out_big = _run(big, scale=512**-0.5)
    s2, p2 = _counts()
    assert (s2 - s1, p2 - p1) == (1, 0), "a grid at the threshold runs single-pass"
    _assert_close(out_big[:1], out_single, mx.bfloat16)


def test_padded_max_seq_len(force_gqa) -> None:
    case = _make_case(kv_lens=[700, 1300], dtype=mx.bfloat16, seed=3, **GEMMA_FULL)
    got = _run(case, scale=512**-0.5, max_seq_len=4096, expect="gqa")
    _assert_close(got, _reference(case, scale=512**-0.5), mx.bfloat16)


def test_bitwise_deterministic_and_default_route_is_the_gqa_kernel() -> None:
    ops = get_ops()
    case = _make_case(kv_lens=[4096, 1000], dtype=mx.bfloat16, seed=31, **GEMMA_FULL)
    a = _run(case, scale=512**-0.5, expect="gqa")
    b = _run(case, scale=512**-0.5, expect="gqa")
    assert np.array_equal(a, b)
    ops.set_gqa_decode_force(True)
    try:
        c = _run(case, scale=512**-0.5, expect="gqa")
    finally:
        _reset(ops)
    assert np.array_equal(a, c)
