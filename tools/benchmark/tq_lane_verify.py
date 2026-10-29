# SPDX-License-Identifier: Apache-2.0
"""Production TurboQuant prefill correctness, interleaved timing and memory.

Both arms execute sdpa_forward, including projections, fused cache writes,
block translation and dispatch. The reference disables only the new planner.
The fixture binds upstream storage and uses nonidentity tables, optional page
padding, shared prefixes and extra unused pool capacity. Reported error is
against the compressed native attention path on the same quantized cache.

The crossover-hd128 suite overrides only the query-count threshold, retaining
shape and workspace checks and reporting the normal production policy separately.
Other suites retain normal routing, including fallbacks.

    PYTHONPATH=. python tools/benchmark/tq_lane_verify.py --suite crossover
    PYTHONPATH=. python tools/benchmark/tq_lane_verify.py --suite crossover-hd128
    PYTHONPATH=. python tools/benchmark/tq_lane_verify.py --suite crossover-hd128 --tiled
    # Override hd128 geometry and include fresh prompts with context 0:
    PYTHONPATH=. python tools/benchmark/tq_lane_verify.py --suite crossover-hd128 \\
        --head-pairs 32:8 16:2 --query-tokens 127 128 129 --context-tokens 0 256 --tiled
    PYTHONPATH=. python tools/benchmark/tq_lane_verify.py --suite long
    PYTHONPATH=. python tools/benchmark/tq_lane_verify.py --suite geometry --tiled
"""

import argparse
import gc
import json
import os
import statistics
import time
from contextlib import ExitStack
from unittest.mock import patch

import mlx.core as mx
import numpy as np

from tools.attention_bench_utils import package_versions
from tools.benchmark.tq_prefill_case import build_case
from vllm_metal.attention.caches.turboquant import prefill_workspace_bytes
from vllm_metal.attention.impls import sdpa
from vllm_metal.attention.impls import turboquant_prefill as tq_prefill
from vllm_metal.metal import get_ops


def run(case, reference, *, force_materialization=False):
    # A fresh forward's metadata; normal multi-layer serving reuses it after
    # the first layer. Both arms pay the same projection and cache-write work.
    case.ctx.kernel_metadata_cache.clear()
    with ExitStack() as stack:
        if reference:
            stack.enter_context(
                patch.object(sdpa, "_turboquant_prefill_plan", return_value=None)
            )
        elif force_materialization:
            # Measure the algorithm below the production threshold, without
            # bypassing shape, budget or memory-lifetime guards.
            stack.enter_context(
                patch.object(tq_prefill, "min_prefill_tokens", return_value=2)
            )
        output = case.forward()
        mx.eval(output)
    return output


def measure(label, kwargs, reps, warmup, *, force_materialization=False):
    case = build_case(**kwargs)
    reference = run(case, True)
    production = run(case, False)
    production_selected = any(
        plan is not None
        for meta in case.ctx.kernel_metadata_cache.values()
        for plan in meta.tq_prefill_plans.values()
    )
    lane = run(case, False, force_materialization=force_materialization)
    selected = any(
        plan is not None
        for meta in case.ctx.kernel_metadata_cache.values()
        for plan in meta.tq_prefill_plans.values()
    )
    if force_materialization and not selected:
        raise RuntimeError(f"{label}: materialization rejected; no crossover measured")
    if not all(
        mx.all(mx.isfinite(output)).item() for output in (reference, production, lane)
    ):
        raise RuntimeError(
            f"{label}: nonfinite attention output; no crossover measured"
        )
    np.testing.assert_allclose(
        np.array(production.astype(mx.float32)),
        np.array(reference.astype(mx.float32)),
        atol=0.02,
        rtol=0.03,
    )
    delta = mx.abs(lane.astype(mx.float32) - reference.astype(mx.float32)).max().item()
    np.testing.assert_allclose(
        np.array(lane.astype(mx.float32)),
        np.array(reference.astype(mx.float32)),
        atol=0.02,
        rtol=0.03,
    )
    del production, lane, reference
    samples = {"compressed": [], "prefill": []}
    for rep in range(warmup + reps):
        arms = [True, False] if rep % 2 == 0 else [False, True]
        for reference in arms:
            start = time.perf_counter()
            output = run(case, reference, force_materialization=force_materialization)
            elapsed = (time.perf_counter() - start) * 1000
            if rep >= warmup:
                samples["compressed" if reference else "prefill"].append(elapsed)
            del output
    memory = {}
    for reference in (True, False):
        gc.collect()
        mx.synchronize()
        mx.clear_cache()
        mx.reset_peak_memory()
        before = mx.get_active_memory()
        output = run(case, reference, force_materialization=force_materialization)
        mx.synchronize()
        memory["compressed" if reference else "prefill"] = (
            mx.get_peak_memory() - before
        ) / 2**20
        del output
    median = {k: statistics.median(v) for k, v in samples.items()}
    plans = [
        plan
        for meta in case.ctx.kernel_metadata_cache.values()
        for plan in meta.tq_prefill_plans.values()
        if plan is not None
    ]
    print(
        json.dumps(
            {
                "case": label,
                "config": kwargs,
                "max_abs_error": delta,
                "median_ms": median,
                "speedup": median["compressed"] / median["prefill"],
                "peak_extra_mib": memory,
                "samples_ms": samples,
                "lane_selected": bool(plans),
                "production_lane_selected": production_selected,
                "forced_materialization": force_materialization,
                "lane_requests": sum(plan.prefill.seq_lens.shape[0] for plan in plans),
                "workspace_estimate_bytes": sum(plan.workspace_bytes for plan in plans),
            },
            default=str,
        ),
        flush=True,
    )
    del case
    gc.collect()
    mx.synchronize()
    mx.clear_cache()


def head_pair(value):
    """Parse an attention geometry as Q:KV, rejecting unsupported ratios."""
    try:
        n_heads, n_kv_heads = (int(part) for part in value.split(":"))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "head pairs must use Q:KV, e.g. 32:8"
        ) from error
    if n_heads < 1 or n_kv_heads < 1 or n_heads % n_kv_heads:
        raise argparse.ArgumentTypeError(
            "head pairs require positive Q and KV with Q divisible by KV"
        )
    return n_heads, n_kv_heads


def cases(suite, *, head_pairs=None, query_tokens=None, context_tokens=None):
    if suite != "crossover-hd128" and (
        head_pairs is not None or query_tokens is not None or context_tokens is not None
    ):
        raise ValueError("matrix overrides require crossover-hd128")
    common = {"head_dim": 256, "n_heads": 24, "n_kv_heads": 4}
    if suite == "crossover":
        for qlen, context in [
            (2, 4096),
            (2, 32768),
            (8, 32768),
            (32, 32768),
            (128, 8192),
            (512, 8192),
            (128, 32768),
            (512, 32768),
            (1153, 1153),
        ]:
            yield (
                f"q{qlen}-kv{context}",
                dict(common, qlens=(qlen,), context_lens=(context,)),
            )
    elif suite == "crossover-hd128":
        geometries = head_pairs if head_pairs is not None else [(8, 2), (8, 8)]
        lengths = (
            query_tokens
            if query_tokens is not None
            else [16, 32, 64, 96, 128, 192, 256, 512]
        )
        contexts = context_tokens if context_tokens is not None else [8192, 32768]
        for n_heads, n_kv_heads in geometries:
            for dtype in [mx.float16, mx.bfloat16]:
                for context_tokens in contexts:
                    for qlen in lengths:
                        context = context_tokens or qlen
                        yield (
                            f"d128-q{n_heads}-kv{n_kv_heads}-{dtype}-q{qlen}-ctx{context}",
                            {
                                "head_dim": 128,
                                "n_heads": n_heads,
                                "n_kv_heads": n_kv_heads,
                                "dtype": dtype,
                                "qlens": (qlen,),
                                "context_lens": (context,),
                            },
                        )
    elif suite == "geometry":
        for hd, nq, nkv in [(64, 8, 8), (128, 8, 2), (256, 24, 4), (512, 8, 1)]:
            for dtype in [mx.float16, mx.bfloat16]:
                for qlen in [128, 256, 512]:
                    yield (
                        f"d{hd}-q{nq}-kv{nkv}-{dtype}-{qlen}",
                        {
                            "head_dim": hd,
                            "n_heads": nq,
                            "n_kv_heads": nkv,
                            "dtype": dtype,
                            "qlens": (qlen,),
                            "context_lens": (8192,),
                        },
                    )
    elif suite == "long":
        for context in [8192, 16384, 32768, 65536, 131072, 262144]:
            yield (
                f"long-kv{context}",
                dict(common, qlens=(128,), context_lens=(context,)),
            )
        yield (
            "independent4k-4",
            dict(common, qlens=(128,) * 4, context_lens=(4096,) * 4),
        )
        yield (
            "shared128k-4",
            dict(
                common, qlens=(128,) * 4, context_lens=(131072,) * 4, shared_prefix=True
            ),
        )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--suite",
        choices=["crossover", "crossover-hd128", "geometry", "long"],
        default="crossover",
    )
    ap.add_argument("--reps", type=int, default=7)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--tiled", action="store_true")
    ap.add_argument(
        "--head-pairs",
        type=head_pair,
        nargs="+",
        help="Q:KV head counts for crossover-hd128 (default: 8:2 8:8)",
    )
    ap.add_argument(
        "--query-tokens",
        type=int,
        nargs="+",
        help="query lengths for crossover-hd128 (at least 2 and at most each context)",
    )
    ap.add_argument(
        "--context-tokens",
        type=int,
        nargs="+",
        help="total context lengths for crossover-hd128; 0 means fresh prompts",
    )
    args = ap.parse_args()
    if args.reps < 1 or args.warmup < 0:
        ap.error("reps must be positive and warmup must be nonnegative")
    if args.suite != "crossover-hd128" and (
        args.head_pairs is not None
        or args.query_tokens is not None
        or args.context_tokens is not None
    ):
        ap.error("matrix overrides require --suite crossover-hd128")
    if args.query_tokens is not None and any(qlen < 2 for qlen in args.query_tokens):
        ap.error("--query-tokens must be at least 2")
    if args.context_tokens is not None and any(ctx < 0 for ctx in args.context_tokens):
        ap.error("--context-tokens must be nonnegative; 0 means fresh prompts")
    contexts = args.context_tokens if args.context_tokens is not None else [8192, 32768]
    max_query = max(args.query_tokens) if args.query_tokens is not None else 512
    if any(0 < context < max_query for context in contexts):
        ap.error("positive --context-tokens must be at least the largest query length")
    ops = get_ops()
    if args.tiled:
        os.environ["VLLM_METAL_TQ_PREFILL"] = "1"
    ops.set_nax_enabled(not args.tiled)
    print(
        json.dumps(
            {
                "device": mx.device_info()["device_name"],
                "nax_ready": ops.nax_ready(),
                "workspace_limit_bytes": prefill_workspace_bytes(),
                "versions": package_versions("vllm", "mlx", "mlx-lm"),
            }
        ),
        flush=True,
    )
    try:
        for label, kwargs in cases(
            args.suite,
            head_pairs=args.head_pairs,
            query_tokens=args.query_tokens,
            context_tokens=args.context_tokens,
        ):
            measure(
                label,
                kwargs,
                args.reps,
                args.warmup,
                force_materialization=args.suite == "crossover-hd128",
            )
    finally:
        ops.set_nax_enabled(True)


if __name__ == "__main__":
    main()
