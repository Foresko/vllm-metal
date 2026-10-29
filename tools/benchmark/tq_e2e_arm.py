# SPDX-License-Identifier: Apache-2.0
"""Warm paired model TTFT or teacher-forced perplexity through real vLLM.

Both TQ arms share a loaded model and quantized cache settings; the reference
disables only the prefill planner. Prefix caching is off except in the explicit
prefix-reuse probe. vLLM request metrics
provide TTFT separately from total generation time. This is an in-process
benchmark, excluding tokenization, HTTP and concurrent serving queues.

``--compare-policies`` instead compares the exact old admission threshold with
the current production helper, while retaining the production planner in both
arms. Both arms use an old-policy seed and the same startup reservation (the
larger old/new cap, subject to the ordinary workspace ceiling). This controls
KV capacity; it does not measure the deployment capacity change from reserving
different budgets. ``--mixed-prefix-probe`` submits all requested suffix lengths
in one real scheduler batch, with separate cached histories and a required mix
of admitted and fallback requests in the new arm.

    PYTHONPATH=. python tools/benchmark/tq_e2e_arm.py --model /path/to/model
    PYTHONPATH=. python tools/benchmark/tq_e2e_arm.py --model /path/to/model \
        --quality-text /path/to/wikitext-test.txt --output quality.json
    PYTHONPATH=. python tools/benchmark/tq_e2e_arm.py --model /path/to/model \
        --prefix-probe --prefix-tokens 8192 --query-tokens 32 64 96 128
    PYTHONPATH=. python tools/benchmark/tq_e2e_arm.py --model /path/to/model \
        --compare-policies --mixed-prefix-probe --prefix-tokens 8192 \
        --query-tokens 64 9 1 --max-tokens 8
"""

import argparse
import hashlib
import inspect
import json
import math
import os
import statistics
import subprocess
import sys
import time
import weakref
from contextlib import contextmanager
from pathlib import Path

PARA = (
    "The city library opened its doors at eight in the morning, and by nine "
    "the reading rooms were already half full. Students spread their notes "
    "across the long oak tables, older visitors settled into the armchairs "
    "by the tall windows, and the librarians moved quietly between the "
    "shelves, returning books to their places. "
)


def old_min_prefill_tokens(
    num_query_heads,
    num_kv_heads,
    head_dim,
    *,
    context_len=None,
    key_quant_type=None,
    value_bits=None,
):
    """The exact production policy before the PR #990 crossover calibration."""
    return max(
        128,
        head_dim // 2,
        (256 * num_kv_heads + num_query_heads - 1) // num_query_heads,
    )


@contextmanager
def prefill_policy(module, policy):
    """Temporarily change admission only; never bypass the production planner."""
    original = module.min_prefill_tokens
    module.min_prefill_tokens = policy
    try:
        yield
    finally:
        module.min_prefill_tokens = original


def shared_reservation_policy(new_policy):
    """Lower admission minimum gives the larger of the two workspace caps."""
    return lambda *geometry, **kwargs: min(
        old_min_prefill_tokens(*geometry, **kwargs), new_policy(*geometry, **kwargs)
    )


def validate_policy_dispatch(row, query_tokens, *, mixed=False):
    """Require one complete scheduler step and all eligible requests admitted."""
    dispatch = row["dispatch"]
    events = dispatch["layer_events"]
    # A lone one-token query uses the decode path without calling the planner.
    if not events and query_tokens == [1]:
        return
    if not events or dispatch["prefill_scheduler_steps"] != 1:
        raise RuntimeError("Policy probe requires one observed scheduler step")
    for event in events:
        if sorted(event["query_lengths"]) != sorted(query_tokens):
            raise RuntimeError("Policy probe requests were not co-scheduled in full")
        if event["selected_requests"] != event["threshold_eligible_requests"]:
            raise RuntimeError(
                "Policy probe did not admit every threshold-eligible request"
            )
        if event["fallback_requests"] + event["selected_requests"] != len(query_tokens):
            raise RuntimeError("Policy probe lost a request in dispatch")
        if (
            mixed
            and row["arm"] == "new-policy"
            and not (0 < event["selected_requests"] < len(query_tokens))
        ):
            raise RuntimeError(
                "Mixed probe requires selected and fallback requests together"
            )


def policy_summary(rows, arms):
    """Keep seeding and warmups out of every reported policy comparison."""
    measured = [
        row for row in rows if row["phase"] == "prefix-reuse" and row["trial"] >= 0
    ]
    summary = {
        "comparison": "old_vs_new_production_policy",
        "shared_workspace_budget": True,
        "median_ttft_s": {},
        "median_gen_wall_s": {},
        "lane_layer_calls": {},
        "query_thresholds": {},
    }
    for arm in arms:
        selected = [row for row in measured if row["arm"] == arm]
        if not selected:
            raise RuntimeError(f"No measured policy rows for {arm}")
        for key in ("ttft_s", "gen_wall_s"):
            summary[f"median_{key}"][arm] = statistics.median(
                row[key] for row in selected
            )
        summary["lane_layer_calls"][arm] = sorted(
            {row["dispatch"]["lane_layer_calls"] for row in selected}
        )
        summary["query_thresholds"][arm] = sorted(
            {value for row in selected for value in row["dispatch"]["query_thresholds"]}
        )
    trials = {row["trial"] for row in measured}
    summary["greedy_tokens_match"] = all(
        len({json.dumps(row["tokens"]) for row in measured if row["trial"] == trial})
        == 1
        and {row["arm"] for row in measured if row["trial"] == trial} == set(arms)
        for trial in trials
    )
    summary["paired_prompts_match"] = all(
        len(
            {
                tuple(request["prompt_sha256"] for request in row["requests"])
                for row in measured
                if row["trial"] == trial
            }
        )
        == 1
        for trial in trials
    )
    summary["old_over_new_ttft_speedup"] = (
        summary["median_ttft_s"]["old-policy"] / summary["median_ttft_s"]["new-policy"]
    )
    summary["measured_repetitions"] = len(trials)
    return summary


def prefix_probe_layout(prefix_tokens, query_tokens, block_size, max_prompt):
    """Keep an exact, block-aligned cached prefix and leave query rows uncached."""
    cached = 2 * block_size if prefix_tokens is None else prefix_tokens
    if cached <= 0 or cached % block_size:
        raise ValueError(f"--prefix-tokens must be a positive multiple of {block_size}")
    required = cached + max(query_tokens)
    if required > max_prompt:
        raise ValueError(f"Prefix probe needs --prompt-tokens >= {required}")
    return cached, required


def validate_prefix_reuse(seed, row, cached_tokens, query_tokens):
    """Reject a cache miss or a different query count instead of mislabelling it."""
    if seed["num_cached_tokens"] != 0 or row["num_cached_tokens"] != cached_tokens:
        raise RuntimeError("Prefix probe did not reuse the exact requested prefix")
    if row["prompt_tokens"] - row["num_cached_tokens"] != query_tokens:
        raise RuntimeError("Prefix probe executed a different query length")
    if row["arm"] in ("tq", "old-policy", "new-policy"):
        dispatch = row["dispatch"]
        eligible = dispatch["threshold_eligible_layer_calls"]
        selected = dispatch["lane_layer_calls"]
        disagrees = (
            selected != eligible
            if row["arm"] in ("old-policy", "new-policy")
            else selected > eligible or (eligible and not selected)
        )
        if disagrees:
            raise RuntimeError(
                "Prefix probe dispatch disagrees with the query threshold; "
                "check hardware opt-in and workspace budget"
            )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument(
        "--arm", choices=("paired", "bf16", "tq", "tq-reference"), default="paired"
    )
    ap.add_argument("--prompt-tokens", type=int, nargs="+", default=[1153, 8192, 16384])
    ap.add_argument("--max-tokens", type=int, default=1)
    ap.add_argument("--batch-tokens", type=int, default=2048)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.7)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--k-quant", default="q8_0")
    ap.add_argument("--v-quant", default="q3_0")
    ap.add_argument("--quality-text", type=Path)
    ap.add_argument("--prefix-probe", action="store_true")
    ap.add_argument(
        "--compare-policies",
        action="store_true",
        help="Compare the exact old threshold with current production admission; requires a prefix probe",
    )
    ap.add_argument(
        "--mixed-prefix-probe",
        action="store_true",
        help="Co-schedule --query-tokens as one batch with distinct seeded histories; implies --prefix-probe",
    )
    ap.add_argument(
        "--prefix-tokens",
        type=int,
        help="Exact cached prefix length, block-aligned; default: two cache blocks",
    )
    ap.add_argument(
        "--query-tokens",
        type=int,
        nargs="+",
        help="Uncached query lengths for the prefix probe; default: 1 9 257",
    )
    ap.add_argument("--quality-windows", type=int, default=16)
    ap.add_argument("--quality-window", type=int, default=1024)
    ap.add_argument("--progress-interval", type=float, default=0)
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()
    if args.mixed_prefix_probe:
        args.prefix_probe = True
    if args.compare_policies and not args.prefix_probe:
        ap.error("--compare-policies requires --prefix-probe or --mixed-prefix-probe")
    if args.mixed_prefix_probe and not args.compare_policies:
        ap.error("--mixed-prefix-probe requires --compare-policies")
    if not 0 < args.gpu_memory_utilization <= 1:
        ap.error("--gpu-memory-utilization must be in (0, 1]")
    if args.compare_policies and args.output and args.output.exists():
        ap.error("policy comparison --output already exists; use a new evidence path")
    if args.query_tokens is not None and not args.prefix_probe:
        ap.error("--query-tokens requires --prefix-probe")
    if args.query_tokens is None:
        args.query_tokens = [1, 9, 257]
    if (
        min(
            *args.prompt_tokens,
            args.max_tokens,
            args.batch_tokens,
            args.reps,
            args.quality_windows,
            args.quality_window,
            *args.query_tokens,
        )
        < 1
        or args.warmup < 0
        or args.progress_interval < 0
    ):
        ap.error("lengths/repetitions must be positive and warmup nonnegative")
    if args.quality_text and args.quality_window < 256:
        ap.error("quality scoring requires window >= 256")
    if args.prefix_probe and (args.arm != "paired" or args.quality_text):
        ap.error("prefix probe requires --arm paired without --quality-text")
    if args.prefix_tokens is not None and (
        not args.prefix_probe or args.prefix_tokens <= 0
    ):
        ap.error("--prefix-tokens requires --prefix-probe and a positive length")
    if args.prefix_probe and not args.compare_policies and args.max_tokens != 1:
        ap.error("prefix probe requires --max-tokens 1 to preserve the seeded prefix")
    if args.prefix_probe and max(args.query_tokens) > args.batch_tokens:
        ap.error("prefix probe query lengths must fit in one --batch-tokens step")
    if args.mixed_prefix_probe and len(args.query_tokens) < 2:
        ap.error("mixed prefix probe requires at least two --query-tokens lengths")
    if args.mixed_prefix_probe and sum(args.query_tokens) > args.batch_tokens:
        ap.error(
            "mixed prefix probe queries must together fit in one --batch-tokens step"
        )

    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

    from vllm_metal.attention.impls import sdpa

    original_planner = sdpa._turboquant_prefill_plan
    try:
        return _run(args)
    finally:
        # Include model/tokenizer setup and metadata collection in the same
        # lifetime as measurement; callers may reuse this Python process.
        sdpa._turboquant_prefill_plan = original_planner


def _run(args):
    import mlx.core as mx
    import numpy as np
    from vllm import LLM, SamplingParams

    from tools.attention_bench_utils import package_versions
    from vllm_metal.attention.caches.turboquant import prefill_workspace_bytes
    from vllm_metal.attention.impls import sdpa
    from vllm_metal.attention.impls import turboquant_prefill as tq_prefill
    from vllm_metal.metal import get_ops

    original_planner = sdpa._turboquant_prefill_plan
    new_policy = tq_prefill.min_prefill_tokens
    policy_parameters = inspect.signature(new_policy).parameters
    dispatch = {}
    step_contexts = []
    active_arm = args.arm
    progress_start = last_progress = time.perf_counter()

    def counted_planner(*planner_args, **kwargs):
        nonlocal last_progress
        dispatch["prefill_layer_calls"] += 1
        dispatch["workspace_limit_bytes"] = planner_args[1].tq_prefill_workspace_bytes
        geometry = planner_args[4:7]
        old_threshold = old_min_prefill_tokens(*geometry)
        new_thresholds = []
        for context_len in planner_args[0].context_lens:
            policy_kwargs = dict(kwargs, context_len=context_len)
            new_thresholds.append(
                new_policy(
                    *geometry,
                    **{
                        k: v for k, v in policy_kwargs.items() if k in policy_parameters
                    },
                )
            )
        policy = old_min_prefill_tokens if active_arm == "old-policy" else new_policy
        thresholds = (
            [old_threshold] * len(new_thresholds)
            if active_arm == "old-policy"
            else new_thresholds
        )
        cu = planner_args[0].cu_seqlens
        query_lengths = [b - a for a, b in zip(cu[:-1], cu[1:], strict=True)]
        eligible_requests = sum(
            length >= threshold
            for length, threshold in zip(query_lengths, thresholds, strict=True)
        )
        dispatch["threshold_eligible_layer_calls"] += int(eligible_requests > 0)
        for threshold in thresholds:
            if threshold not in dispatch["query_thresholds"]:
                dispatch["query_thresholds"].append(threshold)
        with prefill_policy(tq_prefill, policy):
            plan = (
                None
                if active_arm == "tq-reference"
                else original_planner(*planner_args, **kwargs)
            )
        if args.compare_policies:
            context = planner_args[0]
            if not any(context is seen() for seen in step_contexts):
                step_contexts.append(weakref.ref(context))
            selected_requests = 0 if plan is None else plan.prefill.seq_lens.shape[0]
            fallback_requests = (
                len(query_lengths)
                if plan is None
                else 0
                if plan.fallback is None
                else plan.fallback.seq_lens.shape[0]
            )
            dispatch["prefill_scheduler_steps"] = len(step_contexts)
            dispatch["layer_events"].append(
                {
                    "layer_call": dispatch["prefill_layer_calls"],
                    "prefill_scheduler_step": next(
                        i for i, seen in enumerate(step_contexts) if seen() is context
                    ),
                    "cu_seqlens": list(cu),
                    "query_lengths": query_lengths,
                    "context_lens": list(context.context_lens),
                    "num_query_heads": geometry[0],
                    "num_kv_heads": geometry[1],
                    "head_dim": geometry[2],
                    "old_thresholds": [old_threshold] * len(query_lengths),
                    "new_thresholds": new_thresholds,
                    "active_thresholds": thresholds,
                    "threshold_eligible_requests": eligible_requests,
                    "selected_requests": selected_requests,
                    "fallback_requests": fallback_requests,
                }
            )
        if plan is not None:
            dispatch["lane_layer_calls"] += 1
            dispatch["lane_segments"] += plan.prefill.seq_lens.shape[0]
            dispatch["max_gathered_tokens"] = max(
                dispatch["max_gathered_tokens"], plan.pool_pages.shape[0]
            )
            dispatch["max_workspace_bytes"] = max(
                dispatch["max_workspace_bytes"], plan.workspace_bytes
            )
        if args.progress_interval:
            now = time.perf_counter()
            if now - last_progress >= args.progress_interval:
                print(
                    json.dumps(
                        {
                            "progress": {
                                "arm": active_arm,
                                "elapsed_s": now - progress_start,
                                "context_tokens": max(planner_args[0].context_lens),
                                "dispatch": dict(dispatch),
                                "active_bytes": mx.get_active_memory(),
                                "peak_active_bytes": mx.get_peak_memory(),
                            }
                        }
                    ),
                    flush=True,
                )
                last_progress = now
        return plan

    sdpa._turboquant_prefill_plan = counted_planner

    def reset_dispatch():
        step_contexts.clear()
        dispatch.update(
            prefill_layer_calls=0,
            threshold_eligible_layer_calls=0,
            query_thresholds=[],
            lane_layer_calls=0,
            lane_segments=0,
            max_gathered_tokens=0,
            max_workspace_bytes=0,
            workspace_limit_bytes=None,  # Unknown until a planner call is observed.
            prefill_scheduler_steps=0,
            layer_events=[],
        )

    reset_dispatch()

    kwargs = {}
    if args.arm != "bf16":
        kwargs["additional_config"] = {
            "turboquant": True,
            "k_quant": args.k_quant,
            "v_quant": args.v_quant,
        }
    max_prompt = args.quality_window if args.quality_text else max(args.prompt_tokens)
    if args.prefix_probe and args.prefix_tokens is not None:
        max_prompt = args.prefix_tokens + max(args.query_tokens)
    t0 = time.perf_counter()
    # This is a controlled admission-policy comparison: reserve the larger
    # old/new cap once, before cache sizing, and share that exact KV budget.
    # Hardware opt-in and the user's workspace ceiling still apply.
    reservation_policy = (
        shared_reservation_policy(new_policy) if args.compare_policies else new_policy
    )
    with prefill_policy(tq_prefill, reservation_policy):
        llm = LLM(
            model=os.path.expanduser(args.model),
            max_model_len=max_prompt + args.max_tokens,
            max_num_batched_tokens=args.batch_tokens,
            max_num_seqs=len(args.query_tokens) if args.mixed_prefix_probe else 1,
            gpu_memory_utilization=args.gpu_memory_utilization,
            enable_prefix_caching=args.prefix_probe,
            disable_log_stats=False,
            **kwargs,
        )
    load_s = time.perf_counter() - t0
    tokenizer = llm.get_tokenizer()
    arms = ["tq-reference", "tq"] if args.arm == "paired" else [args.arm]
    if args.compare_policies:
        arms = ["old-policy", "new-policy"]
    metadata = {
        "model": args.model,
        "device": mx.device_info()["device_name"],
        "nax_ready": get_ops().nax_ready(),
        # The worker may cap this ceiling by the model and scheduler geometry.
        "workspace_ceiling_bytes": prefill_workspace_bytes(),
        "versions": package_versions(
            "vllm", "mlx", "mlx-lm", "mlx-vlm", "torch", "numpy"
        ),
        "mlx_enable_tf32": os.getenv("MLX_ENABLE_TF32"),
        "prefix_caching": args.prefix_probe,
        "arguments": vars(args),
        "load_s": load_s,
    }
    if args.compare_policies:
        model_config = llm.llm_engine.vllm_config.model_config
        config_json = json.dumps(
            model_config.hf_config.to_dict(), sort_keys=True, default=str
        )
        source_root = Path(__file__).resolve().parents[2]
        metadata.update(
            comparison="old_vs_new_production_policy",
            seed_policy="old-policy",
            prefill_step_definition="planner-observed prefill contexts; subsequent decode-only steps are excluded",
            old_threshold_formula="max(128, head_dim // 2, ceil(256 * num_kv_heads / num_query_heads))",
            new_threshold_policy="current production min_prefill_tokens",
            workspace_reservation="shared max(old_cap,new_cap), constrained by normal hardware opt-in and configured ceiling",
            activation_dtype=str(model_config.dtype),
            model_attention_geometry={
                key: getattr(model_config.hf_config, key, None)
                for key in (
                    "num_attention_heads",
                    "num_key_value_heads",
                    "head_dim",
                    "hidden_size",
                    "num_hidden_layers",
                )
            },
            model_config_sha256=hashlib.sha256(config_json.encode()).hexdigest(),
            source_sha=subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=source_root, text=True
            ).strip(),
            source_diff_sha256=hashlib.sha256(
                subprocess.check_output(["git", "diff", "HEAD", "--"], cwd=source_root)
            ).hexdigest(),
        )
    records = []
    summaries = []
    print(json.dumps({"metadata": metadata}, default=str), flush=True)

    def run(
        ids, arm, phase, trial, *, quality=False, allow_no_lane=False, batch_ids=None
    ):
        nonlocal active_arm, progress_start, last_progress
        active_arm = arm
        reset_dispatch()
        mx.synchronize()
        active_bytes = mx.get_active_memory()
        mx.reset_peak_memory()
        start = time.perf_counter()
        progress_start = last_progress = start
        prompts = [ids] if batch_ids is None else batch_ids
        output_tokens = 1 if quality or phase == "prefix-seed" else args.max_tokens
        results = llm.generate(
            [{"prompt_token_ids": prompt} for prompt in prompts],
            SamplingParams(
                temperature=0,
                max_tokens=output_tokens,
                ignore_eos=True,
                prompt_logprobs=1 if quality else None,
            ),
            use_tqdm=False,
        )
        elapsed = time.perf_counter() - start
        mx.synchronize()
        requests = []
        for index, (prompt, result) in enumerate(zip(prompts, results, strict=True)):
            ttft = getattr(result.metrics, "first_token_latency", None)
            if ttft is None or not math.isfinite(ttft) or ttft <= 0:
                raise RuntimeError("vLLM did not report a valid first_token_latency")
            requests.append(
                {
                    "request_index": index,
                    "request_id": result.request_id,
                    "prompt_tokens": len(prompt),
                    "prompt_sha256": hashlib.sha256(
                        json.dumps(prompt).encode()
                    ).hexdigest(),
                    "num_cached_tokens": result.num_cached_tokens,
                    "remaining_query_tokens": len(prompt) - result.num_cached_tokens,
                    "tokens": list(result.outputs[0].token_ids),
                    "output_tokens": len(result.outputs[0].token_ids),
                    "ttft_s": ttft,
                }
            )
        result = results[0]
        row = {
            "phase": phase,
            "trial": trial,
            "arm": arm,
            "prompt_tokens": sum(request["prompt_tokens"] for request in requests),
            "tokens": requests[0]["tokens"]
            if batch_ids is None
            else [request["tokens"] for request in requests],
            "ttft_s": max(request["ttft_s"] for request in requests),
            "gen_wall_s": elapsed,
            "num_cached_tokens": sum(
                request["num_cached_tokens"] for request in requests
            ),
            "requests": requests,
            "active_before_bytes": active_bytes,
            "peak_active_bytes": mx.get_peak_memory(),
            "peak_extra_bytes": max(0, mx.get_peak_memory() - active_bytes),
            "dispatch": dict(dispatch),
        }
        if arm == "tq" and not allow_no_lane and not dispatch["lane_layer_calls"]:
            raise RuntimeError(
                "No prefill lane calls: check hardware opt-in, lengths and workspace budget"
            )
        if quality:
            if result.prompt_logprobs is None or len(result.prompt_logprobs) != len(
                ids
            ):
                raise RuntimeError("Missing teacher-forced prompt logprobs")
            logprobs = []
            for target, probs in zip(ids[1:], result.prompt_logprobs[1:], strict=True):
                if not probs or target not in probs:
                    raise RuntimeError("Missing ground-truth prompt token logprob")
                logprobs.append(probs[target].logprob)
            if not all(math.isfinite(p) for p in logprobs):
                raise RuntimeError("Nonfinite prompt logprobs")
            row.update(
                scored_tokens=len(logprobs),
                nll_sum=-sum(logprobs),
                logprobs=logprobs,
            )
        records.append(row)
        print(
            json.dumps({k: v for k, v in row.items() if k != "logprobs"}),
            flush=True,
        )
        if args.compare_policies and any(
            request["output_tokens"] != output_tokens for request in requests
        ):
            raise RuntimeError(
                "Policy probe returned an unexpected number of output tokens"
            )
        return row

    try:
        if args.prefix_probe:
            block_size = llm.llm_engine.vllm_config.cache_config.block_size
            cached_tokens, required = prefix_probe_layout(
                args.prefix_tokens, args.query_tokens, block_size, max_prompt
            )
            all_ids = tokenizer.encode(
                PARA * (required // 32 + 2), add_special_tokens=False
            )[:required]
            if len(all_ids) != required:
                raise RuntimeError(
                    "Prefix probe did not construct enough prompt tokens"
                )
            query_batches = (
                [args.query_tokens]
                if args.mixed_prefix_probe
                else [[length] for length in args.query_tokens]
            )
            for query_batch in query_batches:
                query_tokens = query_batch[0]
                histories = [all_ids]
                if args.mixed_prefix_probe:
                    histories = [
                        tokenizer.encode(
                            f"Independent request {index}: "
                            + PARA * (required // 32 + 2),
                            add_special_tokens=False,
                        )[:required]
                        for index in range(len(query_batch))
                    ]
                    if any(len(history) != required for history in histories):
                        raise RuntimeError(
                            "Mixed probe did not construct enough prompt tokens"
                        )
                    if len(
                        {tuple(history[:block_size]) for history in histories}
                    ) != len(histories):
                        raise RuntimeError(
                            "Mixed probe histories must differ within the first cache block"
                        )
                measured = []
                for trial in range(-args.warmup, args.reps):
                    for arm in arms if trial % 2 == 0 else arms[::-1]:
                        if not llm.reset_prefix_cache():
                            raise RuntimeError("Unable to reset the prefix cache")
                        # The extra token lets vLLM cache every requested prefix
                        # block while retaining a query row to compute logits.
                        # Seed both arms with the same production path so their
                        # cached hidden states do not depend on the measured arm.
                        seed = run(
                            histories[0][: cached_tokens + 1],
                            "old-policy" if args.compare_policies else "tq",
                            "prefix-seed",
                            trial,
                            allow_no_lane=True,
                            batch_ids=[
                                history[: cached_tokens + 1] for history in histories
                            ]
                            if args.mixed_prefix_probe
                            else None,
                        )
                        row = run(
                            histories[0][: cached_tokens + query_tokens],
                            arm,
                            "prefix-warmup" if trial < 0 else "prefix-reuse",
                            trial,
                            allow_no_lane=True,
                            batch_ids=[
                                history[: cached_tokens + query]
                                for history, query in zip(
                                    histories, query_batch, strict=True
                                )
                            ]
                            if args.mixed_prefix_probe
                            else None,
                        )
                        for seed_request, request, query in zip(
                            seed["requests"], row["requests"], query_batch, strict=True
                        ):
                            validate_prefix_reuse(
                                seed_request,
                                {
                                    **request,
                                    "arm": row["arm"],
                                    "dispatch": row["dispatch"],
                                },
                                cached_tokens,
                                query,
                            )
                        if args.compare_policies:
                            validate_policy_dispatch(
                                row, query_batch, mixed=args.mixed_prefix_probe
                            )
                        if trial >= 0:
                            measured.append(row)
                summary = {
                    "kind": "prefix-reuse",
                    "comparison": "compressed_vs_production_policy",
                    "block_size": block_size,
                    "num_cached_tokens": cached_tokens,
                    "remaining_query_tokens": query_tokens,
                    "median_ttft_s": {
                        arm: statistics.median(
                            row["ttft_s"] for row in measured if row["arm"] == arm
                        )
                        for arm in arms
                    },
                    "lane_layer_calls": {
                        arm: sorted(
                            {
                                row["dispatch"]["lane_layer_calls"]
                                for row in measured
                                if row["arm"] == arm
                            }
                        )
                        for arm in arms
                    },
                    "greedy_tokens_match": all(
                        len(
                            {
                                json.dumps(row["tokens"])
                                for row in measured
                                if row["trial"] == trial
                            }
                        )
                        == 1
                        for trial in range(args.reps)
                    ),
                }
                if args.compare_policies:
                    summary.update(policy_summary(measured, arms))
                    summary["remaining_query_tokens"] = (
                        query_batch if args.mixed_prefix_probe else query_tokens
                    )
                    summary["kind"] = (
                        "mixed-prefix-reuse"
                        if args.mixed_prefix_probe
                        else "prefix-reuse"
                    )
                    summary["ttft_definition"] = (
                        "maximum per-request first_token_latency in each batch"
                    )
                    summary["num_cached_tokens_per_request"] = cached_tokens
                    summary["request_count"] = len(query_batch)
                summaries.append(summary)
                print(json.dumps(summary), flush=True)
                if args.compare_policies:
                    if not summary["paired_prompts_match"]:
                        raise RuntimeError(
                            "Old/new policy prompts differ; inspect saved records"
                        )
                    if not summary["greedy_tokens_match"]:
                        raise RuntimeError(
                            "Old/new policy greedy output tokens differ; inspect saved records"
                        )
        elif args.quality_text:
            text = args.quality_text.read_text()
            token_ids = tokenizer.encode(text, add_special_tokens=False)
            needed = args.quality_windows * args.quality_window
            if len(token_ids) < needed:
                raise ValueError(
                    f"Quality corpus has {len(token_ids)} tokens; need {needed}"
                )
            metadata["corpus_sha256"] = hashlib.sha256(text.encode()).hexdigest()
            metadata["scored_token_ids_sha256"] = hashlib.sha256(
                json.dumps(token_ids[:needed]).encode()
            ).hexdigest()
            nll = {arm: [] for arm in arms}
            for window in range(args.quality_windows):
                ids = token_ids[
                    window * args.quality_window : (window + 1) * args.quality_window
                ]
                for arm in arms if window % 2 == 0 else arms[::-1]:
                    row = run(ids, arm, "quality", window, quality=True)
                    nll[arm].append(row["nll_sum"])
            count = args.quality_windows * (args.quality_window - 1)
            ppl = {arm: math.exp(sum(values) / count) for arm, values in nll.items()}
            summary = {
                "kind": "quality",
                "scored_tokens": count,
                "perplexity": ppl,
            }
            if len(arms) == 2:
                # Resample paired windows, preserving token alignment and context.
                deltas = (np.array(nll["tq"]) - np.array(nll["tq-reference"])) / (
                    args.quality_window - 1
                )
                bootstrap = (
                    np.random.default_rng(0)
                    .choice(deltas, (10000, len(deltas)))
                    .mean(axis=1)
                )
                summary.update(
                    relative_ppl_change_percent=(ppl["tq"] / ppl["tq-reference"] - 1)
                    * 100,
                    mean_delta_nll=float(deltas.mean()),
                    delta_nll_bootstrap_95ci=np.quantile(
                        bootstrap, [0.025, 0.975]
                    ).tolist(),
                )
            summaries.append(summary)
            print(json.dumps(summary), flush=True)
        else:
            for length in args.prompt_tokens:
                ids = tokenizer.encode(
                    PARA * (length // 32 + 2), add_special_tokens=False
                )[:length]
                assert len(ids) == length
                measured = []
                for trial in range(-args.warmup, args.reps):
                    for arm in arms if trial % 2 == 0 else arms[::-1]:
                        row = run(
                            ids, arm, "warmup" if trial < 0 else "measured", trial
                        )
                        if trial >= 0:
                            measured.append(row)
                summary = {
                    "kind": "latency",
                    "prompt_tokens": length,
                    "output_tokens": args.max_tokens,
                    "median_ttft_s": {},
                    "median_gen_wall_s": {},
                }
                for arm in arms:
                    for field in ("ttft_s", "gen_wall_s"):
                        summary[f"median_{field}"][arm] = statistics.median(
                            row[field] for row in measured if row["arm"] == arm
                        )
                summaries.append(summary)
                print(json.dumps(summary), flush=True)
    except BaseException as exc:
        metadata["validation_error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        if args.output:
            args.output.write_text(
                json.dumps(
                    {"metadata": metadata, "summaries": summaries, "records": records},
                    indent=2,
                    default=str,
                )
                + "\n"
            )


if __name__ == "__main__":
    sys.exit(main())
