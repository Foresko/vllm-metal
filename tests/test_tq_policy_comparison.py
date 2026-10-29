# SPDX-License-Identifier: Apache-2.0
"""CPU-only safeguards for paired production-policy benchmark evidence."""

from types import SimpleNamespace

import pytest

from tools.benchmark import tq_e2e_arm as bench


@pytest.mark.parametrize(
    "geometry, expected",
    [((8, 2, 128), 128), ((8, 8, 128), 256), ((32, 8, 128), 128), ((8, 2, 512), 256)],
)
def test_old_policy_preserves_exact_formula(geometry, expected):
    assert bench.old_min_prefill_tokens(*geometry) == expected
    assert bench.old_min_prefill_tokens(*geometry, context_len=16384) == expected


def test_shared_reservation_covers_both_policies_and_forwards_context():
    def current_policy(qh, kvh, hd, *, context_len=None):
        return 64 if context_len and context_len >= 8192 else 512

    reserve = bench.shared_reservation_policy(current_policy)
    # Below the new cutoff the old policy needs the larger workspace; above
    # the cutoff the new policy does. The common cap must cover either case.
    assert reserve(8, 8, 128, context_len=1024) == 256
    assert reserve(8, 8, 128, context_len=8192) == 64


def test_policy_restores_after_planner_failure():
    def current(*args, **kwargs):
        return 64

    module = SimpleNamespace(min_prefill_tokens=current)
    with pytest.raises(RuntimeError, match="planner failure"):
        with bench.prefill_policy(module, bench.old_min_prefill_tokens):
            assert module.min_prefill_tokens(8, 2, 128) == 128
            raise RuntimeError("planner failure")
    assert module.min_prefill_tokens is current


def dispatch_row(*, queries=(64, 9), selected=1, eligible=1, steps=1):
    return {
        "arm": "new-policy",
        "dispatch": {
            "prefill_scheduler_steps": steps,
            "layer_events": [
                {
                    "query_lengths": list(queries),
                    "threshold_eligible_requests": eligible,
                    "selected_requests": selected,
                    "fallback_requests": len(queries) - selected,
                }
                for _ in range(2)
            ],
        },
    }


def test_mixed_dispatch_requires_observed_batch_and_fallback():
    bench.validate_policy_dispatch(dispatch_row(), [64, 9], mixed=True)
    old = dispatch_row(selected=0, eligible=0)
    old["arm"] = "old-policy"
    bench.validate_policy_dispatch(old, [64, 9], mixed=True)


@pytest.mark.parametrize(
    "kwargs, queries, error",
    [
        ({"steps": 2}, [64, 9], "one observed scheduler step"),
        ({"queries": (32, 9)}, [64, 9], "not co-scheduled in full"),
        ({"selected": 1, "eligible": 2}, [64, 9], "every threshold-eligible"),
        ({"selected": 0, "eligible": 0}, [64, 9], "selected and fallback"),
        ({"selected": 2, "eligible": 2}, [64, 9], "selected and fallback"),
    ],
)
def test_mixed_dispatch_rejects_unproven_or_budget_limited_evidence(
    kwargs, queries, error
):
    with pytest.raises(RuntimeError, match=error):
        bench.validate_policy_dispatch(dispatch_row(**kwargs), queries, mixed=True)


def test_prefix_reuse_rejects_partial_layer_admission():
    row = {
        "arm": "new-policy",
        "num_cached_tokens": 8192,
        "prompt_tokens": 8256,
        "dispatch": {"threshold_eligible_layer_calls": 2, "lane_layer_calls": 1},
    }
    with pytest.raises(RuntimeError, match="dispatch disagrees"):
        bench.validate_prefix_reuse({"num_cached_tokens": 0}, row, 8192, 64)


def summary_row(arm, trial, ttft, *, phase="prefix-reuse", tokens=None):
    return {
        "arm": arm,
        "phase": phase,
        "trial": trial,
        "ttft_s": ttft,
        "gen_wall_s": ttft + 1,
        "tokens": [[17], [29]] if tokens is None else tokens,
        "requests": [
            {"prompt_sha256": "history-one"},
            {"prompt_sha256": "history-two"},
        ],
        "dispatch": {
            "lane_layer_calls": 0 if arm == "old-policy" else 2,
            "query_thresholds": [128 if arm == "old-policy" else 64],
        },
    }


def test_summary_excludes_seeds_and_warmups_and_matches_each_request():
    rows = [
        summary_row("old-policy", 0, 100, phase="prefix-seed", tokens=[[999]]),
        summary_row("new-policy", -1, 100, phase="prefix-warmup", tokens=[[999]]),
        summary_row("old-policy", 0, 2),
        summary_row("new-policy", 0, 1),
        summary_row("old-policy", 1, 4),
        summary_row("new-policy", 1, 2),
    ]
    summary = bench.policy_summary(rows, ["old-policy", "new-policy"])
    assert summary["median_ttft_s"] == {"old-policy": 3, "new-policy": 1.5}
    assert summary["old_over_new_ttft_speedup"] == 2
    assert summary["measured_repetitions"] == 2
    assert summary["greedy_tokens_match"] is True
    assert summary["paired_prompts_match"] is True
    rows[-1]["tokens"] = [[17], [30]]
    assert not bench.policy_summary(rows, ["old-policy", "new-policy"])[
        "greedy_tokens_match"
    ]
    rows[-1]["requests"][0]["prompt_sha256"] = "wrong-prompt"
    assert not bench.policy_summary(rows, ["old-policy", "new-policy"])[
        "paired_prompts_match"
    ]


@pytest.mark.parametrize(
    "arguments, error",
    [
        (["--compare-policies"], "requires --prefix-probe"),
        (["--mixed-prefix-probe"], "requires --compare-policies"),
        (
            ["--mixed-prefix-probe", "--compare-policies", "--query-tokens", "64"],
            "at least two",
        ),
        (
            [
                "--mixed-prefix-probe",
                "--compare-policies",
                "--query-tokens",
                "64",
                "9",
                "--batch-tokens",
                "64",
            ],
            "together fit",
        ),
        (["--gpu-memory-utilization", "nan"], "must be in"),
        (["--gpu-memory-utilization", "1.1"], "must be in"),
    ],
)
def test_invalid_cli_fails_before_runtime_import(monkeypatch, capsys, arguments, error):
    monkeypatch.setattr("sys.argv", ["tq_e2e_arm.py", "--model", "unused", *arguments])
    # Accessing the runtime would fail with a different exception.
    monkeypatch.setitem(__import__("sys").modules, "mlx.core", None)
    with pytest.raises(SystemExit) as exc:
        bench.main()
    assert exc.value.code == 2
    assert error in capsys.readouterr().err


def test_comparison_never_overwrites_existing_evidence(tmp_path, monkeypatch, capsys):
    evidence = tmp_path / "failed.json"
    evidence.write_text('{"validation_error": "previous evidence"}')
    monkeypatch.setattr(
        "sys.argv",
        [
            "tq_e2e_arm.py",
            "--model",
            "unused",
            "--prefix-probe",
            "--compare-policies",
            "--output",
            str(evidence),
        ],
    )
    with pytest.raises(SystemExit) as exc:
        bench.main()
    assert exc.value.code == 2
    assert "already exists" in capsys.readouterr().err
    assert evidence.read_text() == '{"validation_error": "previous evidence"}'


@pytest.mark.parametrize("failure", ["model", "tokenizer", "git"])
def test_main_restores_planner_when_initialization_fails(monkeypatch, failure):
    import vllm

    from vllm_metal import metal
    from vllm_metal.attention.caches import turboquant
    from vllm_metal.attention.impls import sdpa

    original = sdpa._turboquant_prefill_plan

    class FakeLLM:
        def __init__(self, **kwargs):
            assert sdpa._turboquant_prefill_plan is not original
            if failure == "model":
                raise RuntimeError("injected model failure")
            self.llm_engine = SimpleNamespace(
                vllm_config=SimpleNamespace(
                    model_config=SimpleNamespace(
                        hf_config=SimpleNamespace(to_dict=lambda: {}), dtype="bfloat16"
                    )
                )
            )

        def get_tokenizer(self):
            if failure == "tokenizer":
                raise RuntimeError("injected tokenizer failure")
            return object()

    def fail_git(*args, **kwargs):
        raise RuntimeError("injected git failure")

    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    monkeypatch.setattr(vllm, "LLM", FakeLLM)
    monkeypatch.setattr(
        metal, "get_ops", lambda: SimpleNamespace(nax_ready=lambda: True)
    )
    monkeypatch.setattr(turboquant, "prefill_workspace_bytes", lambda: 1)
    monkeypatch.setattr(bench.subprocess, "check_output", fail_git)
    monkeypatch.setattr(
        "sys.argv",
        ["tq_e2e_arm.py", "--model", "unused", "--prefix-probe", "--compare-policies"],
    )
    with pytest.raises(RuntimeError, match=f"injected {failure} failure"):
        bench.main()
    assert sdpa._turboquant_prefill_plan is original
