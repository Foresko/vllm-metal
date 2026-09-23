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

import time
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
    full = _make_case(kv_lens=[4096], dtype=mx.bfloat16, **GEMMA_FULL)
    four = _make_case(kv_lens=[1024] * 4, dtype=mx.bfloat16, seed=1, **GEMMA_FULL)
    short = _make_case(kv_lens=[2048], dtype=mx.bfloat16, seed=2, **GEMMA_FULL)
    three = _make_case(kv_lens=[1024] * 3, dtype=mx.bfloat16, seed=3, **GEMMA_FULL)
    sliding = _make_case(
        kv_lens=[4096], heads=16, kv_heads=8, hd=256, dtype=mx.bfloat16
    )
    mha = _make_case(kv_lens=[600], heads=16, kv_heads=16, hd=512, dtype=mx.bfloat16)
    hd64 = _make_case(kv_lens=[600], heads=16, kv_heads=2, hd=64, dtype=mx.bfloat16)
    fp32 = _make_case(kv_lens=[600], heads=16, kv_heads=2, hd=512, dtype=mx.float32)
    try:
        # (512, 8) is a measured win once tokens x context reaches 4096.
        _run(full, scale=512**-0.5, expect="gqa")  # 1 x 4096
        _run(four, scale=512**-0.5, expect="gqa")  # 4 x 1024
        _run(short, scale=512**-0.5, expect="per_token")  # 1 x 2048
        _run(three, scale=512**-0.5, expect="per_token")  # 3 x 1024
        _run(sliding, scale=256**-0.5, expect="per_token")  # (256, 2): not measured
        ops.set_gqa_decode_enabled(False)
        _run(full, scale=512**-0.5, expect="per_token")  # kill switch
        ops.set_gqa_decode_enabled(True)
        ops.set_gqa_decode_force(True)
        _run(short, scale=512**-0.5, expect="gqa")  # force ignores the bucket
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


def test_partition_rule_follows_the_measured_thresholds() -> None:
    ops = get_ops()
    rule = ops.gqa_decode_partition_size
    cores = ops.gqa_decode_min_grid() // 8
    edge = -(-3 * cores // 2)  # ceil(1.5 threadgroups per core)
    # Small split grid: 256-token partitions ...
    assert rule(1, 1, 512 * (edge - 1)) == 256
    # ... until grid x ceil(ctx / 512) reaches 1.5 per core: 512.
    assert rule(1, 1, 512 * edge) == 512
    # One partition covering the whole context: single pass.
    assert rule(1, 1, 256) == 0
    assert rule(1, edge, 512) == 0
    # One token over: that partition no longer covers the context, so split.
    assert rule(1, 1, 257) == 256
    assert rule(1, edge, 513) == 512
    # Full grid: single pass; one threadgroup below it: split.
    grid = ops.gqa_decode_min_grid()
    assert rule(grid, 1, 65536) == 0
    assert rule(1, grid - 1, 65536) == 512
    ops.set_gqa_decode_partition_size(256)
    try:
        assert rule(grid, 1, 65536) == 256  # the test override wins
    finally:
        _reset(ops)


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


def test_varlen_batch_keeps_short_rows_inert(decode_path) -> None:
    case = _make_case(
        kv_lens=[600, 40000, 1, 17, 513], dtype=mx.bfloat16, seed=7, **GEMMA_FULL
    )
    for window in (None, 1024):
        got = _run(case, scale=512**-0.5, window=window, expect=decode_path)
        _assert_close(
            got, _reference(case, scale=512**-0.5, window=window), mx.bfloat16
        )


@pytest.mark.parametrize("window", [None, 1024])
def test_peaked_attention_pins_single_keys(decode_path, window) -> None:
    """Each head puts logit 40 on one key; a key off by one flips the row."""
    n, hd, heads, kv_heads = 3073, 512, 16, 2
    case = _make_case(kv_lens=[n], dtype=mx.bfloat16, seed=11, **GEMMA_FULL)
    kc = np.array(case.key_cache.astype(mx.float32))
    table = case.tables[0]
    g = heads // kv_heads
    ws = 0 if window is None else n - window
    targets = [j for j in (ws - 1, ws, 511, 512, 1023, n - 1) if 0 <= j < n]
    q = np.zeros((1, heads, hd), np.float32)
    for h in range(heads):
        j = targets[h % len(targets)]
        key = kc[table[j // case.block], j % case.block, h // g]
        q[0, h] = key * (40.0 / (float(key @ key) * hd**-0.5))
    case.query = mx.array(q).astype(mx.bfloat16)
    ref = _reference(case, scale=hd**-0.5, window=window)
    got = _run(case, scale=hd**-0.5, window=window, expect=decode_path)
    tol = 2.0**-7 * np.abs(ref).max(axis=-1, keepdims=True)
    assert np.all(np.abs(got - ref) <= tol)


@pytest.mark.parametrize("window", [None, 1024])
def test_poisoned_cache_outside_the_attended_range_never_leaks(
    decode_path, window
) -> None:
    n = 3073
    clean = _make_case(kv_lens=[n], dtype=mx.bfloat16, seed=13, **GEMMA_FULL)
    ref = _reference(clean, scale=512**-0.5, window=window)
    kc = np.array(clean.key_cache.astype(mx.float32))
    vc = np.array(clean.value_cache.astype(mx.float32))
    poison_k = np.full_like(kc, 1e3)
    poison_v = np.full_like(vc, 1e3)
    lo = 0 if window is None else n - window
    for p in range(lo, n):
        b, o = clean.tables[0, p // clean.block], p % clean.block
        poison_k[b, o] = kc[b, o]
        poison_v[b, o] = vc[b, o]
    poisoned = Case(
        clean.query,
        mx.array(poison_k).astype(mx.bfloat16),
        mx.array(poison_v).astype(mx.bfloat16),
        clean.tables,
        clean.kv_lens,
        clean.kv_heads,
        clean.block,
    )
    got = _run(poisoned, scale=512**-0.5, window=window, expect=decode_path)
    _assert_close(got, ref, mx.bfloat16)


def test_realistic_logit_range(decode_path) -> None:
    """Gemma 4 RMS-normalizes q and k per head and uses scale 1.0."""
    case = _make_case(kv_lens=[4096], dtype=mx.bfloat16, seed=17, **GEMMA_FULL)

    def rms_normalize(x: mx.array) -> mx.array:
        a = np.array(x.astype(mx.float32))
        a = a / np.sqrt((a * a).mean(axis=-1, keepdims=True))
        return mx.array(a).astype(mx.bfloat16)

    case.query = rms_normalize(case.query)
    case.key_cache = rms_normalize(case.key_cache)
    got = _run(case, scale=1.0, expect=decode_path)
    _assert_close(got, _reference(case, scale=1.0), mx.bfloat16)


@pytest.mark.parametrize("partition", [0, 256, 512])
@pytest.mark.parametrize("g", [8, 12])
def test_sinks_with_and_without_partitions(force_gqa, g, partition) -> None:
    force_gqa.set_gqa_decode_partition_size(partition)
    case = _make_case(
        kv_lens=[1500, 700],
        heads=2 * g,
        kv_heads=2,
        hd=512,
        dtype=mx.bfloat16,
        seed=g,
    )
    sinks = np.linspace(-2.0, 3.0, 2 * g, dtype=np.float32)
    got = _run(case, scale=512**-0.5, sinks=sinks, expect="gqa")
    _assert_close(got, _reference(case, scale=512**-0.5, sinks=sinks), mx.bfloat16)


def test_softcap(force_gqa) -> None:
    case = _make_case(kv_lens=[2000], dtype=mx.bfloat16, seed=19, **GEMMA_FULL)
    got = _run(case, scale=0.125, softcap=30.0, expect="gqa")
    _assert_close(got, _reference(case, scale=0.125, softcap=30.0), mx.bfloat16)


def test_accuracy_not_worse_than_the_per_token_kernel() -> None:
    ops = get_ops()
    case = _make_case(kv_lens=[2048] * 16, dtype=mx.bfloat16, seed=37, **GEMMA_FULL)
    ref = _reference(case, scale=512**-0.5)  # 16 x 16 x 512 = 131072 outputs
    ops.set_gqa_decode_enabled(False)
    try:
        old = _run(case, scale=512**-0.5, expect="per_token")
    finally:
        _reset(ops)
    new = _run(case, scale=512**-0.5, expect="gqa")
    e_old, e_new = np.abs(old - ref), np.abs(new - ref)
    for stat in (np.mean, lambda e: np.percentile(e, 99), np.max):
        assert stat(e_new) <= 1.1 * stat(e_old) + 1e-7


SHAPES = [(hd, g) for hd in (128, 256, 512) for g in (2, 4, 8, 12, 16)]


@pytest.mark.parametrize(("hd", "g"), SHAPES)
def test_every_supported_shape_matches_reference_fp16(force_gqa, hd, g) -> None:
    case = _make_case(
        kv_lens=[1500],
        heads=2 * g,
        kv_heads=2,
        hd=hd,
        dtype=mx.float16,
        seed=hd + g,
    )
    got = _run(case, scale=hd**-0.5, expect="gqa")
    _assert_close(got, _reference(case, scale=hd**-0.5), mx.float16)


@pytest.mark.slow
@pytest.mark.parametrize("window", [None, 1024])
@pytest.mark.parametrize("block", [8, 16, 32])
@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
@pytest.mark.parametrize(("hd", "g"), SHAPES)
def test_shape_matrix(force_gqa, hd, g, dtype, block, window) -> None:
    for n in (700, 3071, 3073):
        case = _make_case(
            kv_lens=[n],
            heads=2 * g,
            kv_heads=2,
            hd=hd,
            dtype=dtype,
            block=block,
            seed=n + hd + g,
        )
        got = _run(case, scale=hd**-0.5, window=window, expect="gqa")
        _assert_close(got, _reference(case, scale=hd**-0.5, window=window), dtype)


@pytest.mark.slow
@pytest.mark.parametrize("hd", [128, 256, 512])
@pytest.mark.parametrize("block", [8, 16, 32])
@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
def test_every_pipeline_compiles_and_runs(force_gqa, dtype, block, hd) -> None:
    # Sinks off and on for each partition size in one process, so a pipeline
    # cache key that misses a function constant is caught here.
    case = _make_case(
        kv_lens=[600], heads=16, kv_heads=2, hd=hd, dtype=dtype, block=block
    )
    sinks = np.linspace(-1.0, 1.0, 16, dtype=np.float32)
    for size in (0, 256, 512):
        force_gqa.set_gqa_decode_partition_size(size)
        for s in (None, sinks):
            got = _run(case, scale=hd**-0.5, sinks=s, expect="gqa")
            _assert_close(got, _reference(case, scale=hd**-0.5, sinks=s), dtype)


@pytest.mark.slow
def test_gqa_decode_costs_at_most_half_at_32k() -> None:
    """Catches a pathologically slow kernel; counters cover silent fallback."""
    ops = get_ops()
    case = _make_case(kv_lens=[32768], dtype=mx.bfloat16, seed=41, **GEMMA_FULL)
    table = mx.array(case.tables)
    lens = mx.array(case.kv_lens, dtype=mx.int32)
    cu = mx.array([0, 1], dtype=mx.int32)
    flush = mx.random.normal((128 * 1024 * 1024,)).astype(mx.bfloat16)  # 256 MB
    mx.eval(flush, table, lens, cu)
    calls = 4

    def graph(with_attention: bool) -> None:
        outs = []
        for _ in range(calls):
            f = flush.sum()
            outs.append(f)
            if with_attention:
                q = case.query + (f * 0).astype(case.query.dtype)
                out = mx.array(0)
                ops.paged_attention_primitive(
                    q,
                    case.key_cache,
                    case.value_cache,
                    2,
                    512**-0.5,
                    0.0,
                    table,
                    lens,
                    cu,
                    16,
                    32768,
                    -1,
                    out,
                )
                outs.append(out)
        mx.eval(*outs)

    def median_ms(with_attention: bool) -> float:
        graph(with_attention)
        times = []
        for _ in range(5):
            t0 = time.perf_counter()
            graph(with_attention)
            times.append(time.perf_counter() - t0)
        return sorted(times)[2] / calls * 1e3

    flush_ms = median_ms(False)
    ops.set_gqa_decode_enabled(False)
    try:
        old = median_ms(True) - flush_ms
    finally:
        _reset(ops)
    new = median_ms(True) - flush_ms
    assert new <= 0.5 * old, f"per-token {old:.3f} ms, GQA {new:.3f} ms"
