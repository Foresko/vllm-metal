# SPDX-License-Identifier: Apache-2.0
"""Validate calibrated long-context admission and its scheduler reservation."""

from unittest.mock import patch

import mlx.core as mx
import numpy as np
import pytest

from tools.benchmark.tq_prefill_case import build_case
from vllm_metal.attention.caches.turboquant import QUANT_PARAMS, V_QUANT_PARAMS
from vllm_metal.attention.context import PagedAttentionContext
from vllm_metal.attention.impls import sdpa
from vllm_metal.attention.impls import turboquant_prefill as policy

CALIBRATED = [(8, 2, 128), (8, 8, 128), (32, 8, 128), (16, 2, 128)]


@pytest.mark.parametrize("key_format", [None, *QUANT_PARAMS])
@pytest.mark.parametrize("value_bits", [None, 2, 3, 4, 5, 8])
def test_only_measured_encoding_and_its_alias_use_lower_threshold(
    key_format, value_bits
):
    expected = 64 if key_format in ("q8_0", "int8") and value_bits == 3 else 128
    assert (
        policy.min_prefill_tokens(
            32,
            8,
            128,
            context_len=8192,
            key_quant_type=key_format,
            value_bits=value_bits,
        )
        == expected
    )


@pytest.mark.parametrize(
    "key_format,value_format,calibrated",
    [
        ("q8_0", "q3_0", True),
        ("int8", "q3_0", True),
        ("uint8", "q3_0", False),
        ("q4_0", "q3_0", False),
        ("q8_0", "q4_0", False),
    ],
)
@pytest.mark.parametrize("queries", [63, 64, 65, 128])
def test_format_gate_reaches_production_forward(
    key_format, value_format, calibrated, queries, force_tiled_prefill
):
    case = build_case(
        qlens=(queries,),
        context_lens=(8192,),
        k_quant=key_format,
        v_quant=value_format,
    )
    with patch.object(
        sdpa, "materialize_turboquant_pages", wraps=sdpa.materialize_turboquant_pages
    ) as materialize:
        assert_parity(case)
    assert materialize.call_count == int(queries >= (64 if calibrated else 128))


def test_plan_cache_separates_formats_only_when_admission_changes():
    case = build_case(qlens=(64,), context_lens=(8192,))
    calibrated = plan_for(case)
    assert calibrated is not None
    assert plan_for(case, key_quant_type="uint8") is None
    assert plan_for(case, value_bits=4) is None
    assert plan_for(case, key_quant_type="int8") is calibrated
    assert plan_for(case) is calibrated


@pytest.mark.parametrize(
    "key_format,value_format", [("q8_0", "q3_0"), ("q4_0", "q3_0"), ("q8_0", "q4_0")]
)
def test_format_aware_workspace_covers_selected_independent_histories(
    key_format, value_format
):
    # A 193-query step can admit three K8/V3 histories, but only one under
    # the old threshold. Verify the cap against actual plan arrays and routing.
    minimum = 64 if (key_format, value_format) == ("q8_0", "q3_0") else 128
    bits = V_QUANT_PARAMS[value_format]["bits"]
    allowance = policy.workspace_upper_bound(
        max_model_len=8192,
        max_num_seqs=4,
        max_num_batched_tokens=193,
        num_query_heads=8,
        num_kv_heads=2,
        head_dim=128,
        block_size=16,
        key_quant_type=key_format,
        value_bits=bits,
    )
    case = build_case(
        qlens=(minimum, 64, 1),
        context_lens=(8192, 8192, 8192),
        k_quant=key_format,
        v_quant=value_format,
    )
    case.ctx.tq_prefill_workspace_bytes = allowance
    plan = plan_for(case)
    assert plan is not None
    assert plan.prefill.seq_lens.size == (2 if minimum == 64 else 1)
    assert plan.workspace_bytes <= allowance


@pytest.fixture(autouse=True)
def enable_prefill(monkeypatch):
    monkeypatch.setenv("VLLM_METAL_TQ_PREFILL", "1")
    monkeypatch.setenv("VLLM_METAL_TQ_PREFILL_MAX_MIB", "256")


def assert_parity(case):
    output = case.forward()
    mx.eval(output)
    reference = case.reference()
    mx.eval(reference)
    assert mx.all(mx.isfinite(output)).item()
    assert mx.all(mx.isfinite(reference)).item()
    np.testing.assert_allclose(
        np.array(output.astype(mx.float32)),
        np.array(reference.astype(mx.float32)),
        atol=0.02,
        rtol=0.03,
    )


def plan_for(case, **format_override):
    meta = sdpa._kernel_metadata(
        case.ctx,
        None,
        case.ctx.slot_mapping,
        case.ctx.block_tables,
        case.cache.block_size,
    )
    return policy._turboquant_prefill_plan(
        case.ctx,
        meta,
        case.ctx.block_tables,
        case.cache.block_size,
        case.inner.n_heads,
        case.inner.n_kv_heads,
        case.inner.head_dim,
        **dict(
            {"key_quant_type": case.cache.k_quant, "value_bits": case.cache.v_bits},
            **format_override,
        ),
    )


@pytest.mark.parametrize("geometry", CALIBRATED)
@pytest.mark.parametrize("context_len", [None, 64, 2048, 8191, 8192, 32768, 131072])
def test_policy_changes_only_at_long_context_boundary(geometry, context_len):
    old = max(
        128, geometry[2] // 2, (256 * geometry[1] + geometry[0] - 1) // geometry[0]
    )
    expected = 64 if context_len is not None and context_len >= 8192 else old
    assert (
        policy.min_prefill_tokens(
            *geometry, context_len=context_len, key_quant_type="q8_0", value_bits=3
        )
        == expected
    )


@pytest.mark.parametrize(
    "geometry,expected",
    [
        ((8, 2, 64), 128),
        ((8, 8, 64), 256),
        ((24, 4, 256), 128),
        ((8, 1, 512), 256),
        ((16, 4, 128), 128),
        ((32, 32, 128), 256),
    ],
)
def test_unmeasured_geometry_keeps_original_policy(geometry, expected):
    for context_len in (None, 8191, 8192, 32768):
        assert (
            policy.min_prefill_tokens(
                *geometry, context_len=context_len, key_quant_type="q8_0", value_bits=3
            )
            == expected
        )


@pytest.mark.parametrize("geometry", CALIBRATED)
@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
@pytest.mark.parametrize("qlen", [63, 64, 65])
def test_long_context_dispatch_boundary(geometry, dtype, qlen, force_tiled_prefill):
    nq, nkv, hd = geometry
    case = build_case(
        qlens=(qlen,),
        context_lens=(8192,),
        n_heads=nq,
        n_kv_heads=nkv,
        head_dim=hd,
        dtype=dtype,
    )
    with patch.object(
        sdpa, "materialize_turboquant_pages", wraps=sdpa.materialize_turboquant_pages
    ) as materialize:
        assert_parity(case)
    assert materialize.call_count == int(qlen >= 64)
    plan = plan_for(case)
    assert (plan is not None) == (qlen >= 64)
    if plan is not None:
        assert plan.fallback is None
        assert plan.workspace_bytes <= case.ctx.tq_prefill_workspace_bytes


@pytest.mark.parametrize("geometry", CALIBRATED)
@pytest.mark.parametrize("context_len", [8191, 8192])
def test_context_boundary_changes_actual_dispatch(
    geometry, context_len, force_tiled_prefill
):
    nq, nkv, hd = geometry
    case = build_case(
        qlens=(64,),
        context_lens=(context_len,),
        n_heads=nq,
        n_kv_heads=nkv,
        head_dim=hd,
    )
    with patch.object(
        sdpa, "materialize_turboquant_pages", wraps=sdpa.materialize_turboquant_pages
    ) as materialize:
        assert_parity(case)
    assert materialize.call_count == int(context_len >= 8192)


@pytest.mark.parametrize("geometry", CALIBRATED)
@pytest.mark.parametrize("shared_prefix", [False, True])
def test_mixed_contexts_restore_query_order(
    geometry, shared_prefix, force_tiled_prefill
):
    nq, nkv, hd = geometry
    case = build_case(
        qlens=(1, 64, 64, 63, 65),
        context_lens=(8192, 8192, 8191, 8192, 8208),
        n_heads=nq,
        n_kv_heads=nkv,
        head_dim=hd,
        shared_prefix=shared_prefix,
        page_padding=512,
    )
    case.ctx.num_decode_requests = 1
    case.ctx.num_decode_tokens = 1
    case.ctx.max_decode_context_len = 8192
    with patch.object(
        sdpa, "materialize_turboquant_pages", wraps=sdpa.materialize_turboquant_pages
    ) as materialize:
        assert_parity(case)
    assert materialize.call_count == 1
    plan = plan_for(case)
    assert plan is not None and plan.fallback is not None
    assert plan.prefill.seq_lens.tolist() == [8192, 8208]
    assert plan.fallback.seq_lens.tolist() == [8192, 8191, 8192]
    selected = plan.prefill.query_indices.tolist()
    fallback = plan.fallback.query_indices.tolist()
    assert selected == list(range(1, 65)) + list(range(192, 257))
    order = selected + fallback
    assert [order[index] for index in plan.restore_indices.tolist()] == list(range(257))
    assert plan.workspace_bytes <= case.ctx.tq_prefill_workspace_bytes


@pytest.mark.parametrize("geometry", CALIBRATED)
@pytest.mark.parametrize("block_size", [16, 32, 544])
def test_workspace_reserves_three_new_candidates_and_decode(geometry, block_size):
    nq, nkv, hd = geometry
    length = 8193
    pages = (length + block_size - 1) // block_size
    tables = [list(range(i * pages, (i + 1) * pages)) for i in range(4)]
    allowance = policy.workspace_upper_bound(
        max_model_len=length,
        max_num_seqs=4,
        max_num_batched_tokens=3 * 64 + 1,
        num_query_heads=nq,
        num_kv_heads=nkv,
        head_dim=hd,
        block_size=block_size,
        key_quant_type="q8_0",
        value_bits=3,
    )
    ctx = PagedAttentionContext(
        slot_mapping=[],
        block_tables=tables,
        context_lens=[length] * 4,
        cu_seqlens=[0, 64, 65, 129, 193],
        tq_prefill_workspace_bytes=allowance,
    )
    meta = sdpa._kernel_metadata(ctx, 0, [], tables, block_size)
    plan = policy._turboquant_prefill_plan(
        ctx,
        meta,
        tables,
        block_size,
        nq,
        nkv,
        hd,
        key_quant_type="q8_0",
        value_bits=3,
    )
    assert plan is not None and plan.fallback is not None
    assert plan.prefill.seq_lens.tolist() == [length] * 3
    assert plan.fallback.seq_lens.tolist() == [length]
    metadata = [plan.pool_pages, plan.pool_offsets, plan.restore_indices]
    for batch in (plan.prefill, plan.fallback):
        metadata.extend(
            [
                batch.query_indices,
                batch.block_tables,
                batch.seq_lens,
                batch.cu_seqlens_q,
            ]
        )
    # Check retained array shapes and the three routing copies independently.
    actual_bytes = (
        4 * plan.pool_pages.size * nkv * hd
        + 3 * 193 * nq * hd * 2
        + sum(array.nbytes for array in metadata)
    )
    assert actual_bytes <= plan.workspace_bytes <= allowance
    # The old threshold reserves at most one independent history here.
    old_minimum = policy.min_prefill_tokens(*geometry)
    with patch.object(policy, "min_prefill_tokens", return_value=old_minimum):
        old_allowance = policy.workspace_upper_bound(
            max_model_len=length,
            max_num_seqs=4,
            max_num_batched_tokens=193,
            num_query_heads=nq,
            num_kv_heads=nkv,
            head_dim=hd,
            block_size=block_size,
        )
    assert old_allowance < plan.workspace_bytes


@pytest.mark.parametrize("geometry", CALIBRATED)
def test_exact_budget_rejects_before_materialization(geometry, force_tiled_prefill):
    nq, nkv, hd = geometry
    case = build_case(
        qlens=(64,), context_lens=(8192,), n_heads=nq, n_kv_heads=nkv, head_dim=hd
    )
    required = plan_for(case).workspace_bytes
    for allowance, selected in ((required - 1, False), (required, True)):
        case.ctx.kernel_metadata_cache.clear()
        case.ctx.tq_prefill_workspace_bytes = allowance
        with patch.object(
            sdpa,
            "materialize_turboquant_pages",
            wraps=sdpa.materialize_turboquant_pages,
        ) as materialize:
            assert_parity(case)
        assert materialize.call_count == int(selected)


def test_plan_key_tracks_per_sequence_policy():
    case = build_case(qlens=(64, 64), context_lens=(8192, 8191))
    current = plan_for(case)
    assert current is not None and current.fallback is not None
    with patch.object(policy, "min_prefill_tokens", return_value=128):
        assert plan_for(case) is None
    with patch.object(
        policy,
        "min_prefill_tokens",
        side_effect=lambda *_, context_len, **_kwargs: (
            128 if context_len >= 8192 else 64
        ),
    ):
        reversed_plan = plan_for(case)
    # Both policies have a minimum of 64, but select different request rows.
    assert reversed_plan is not current
    assert reversed_plan.prefill.seq_lens.tolist() == [8191]
    assert plan_for(case) is current


@pytest.mark.parametrize("geometry", CALIBRATED)
def test_workspace_bound_is_monotonic_at_context_transition(geometry):
    nq, nkv, hd = geometry
    bounds = [
        policy.workspace_upper_bound(
            max_model_len=length,
            max_num_seqs=4,
            max_num_batched_tokens=193,
            num_query_heads=nq,
            num_kv_heads=nkv,
            head_dim=hd,
            block_size=16,
            key_quant_type="q8_0",
            value_bits=3,
        )
        for length in (8191, 8192, 8193, 32768)
    ]
    assert bounds == sorted(bounds)
    assert bounds[1] > bounds[0]
