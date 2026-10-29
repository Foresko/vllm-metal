# SPDX-License-Identifier: Apache-2.0

from copy import deepcopy

import pytest

from tools.hybrid_spec_parity import (
    SAMPLING_KWARGS,
    build_restore_prompt,
    compare_arms,
)


def _arms(block_size=160):
    checkpoint = 2 * block_size
    reference = {
        "prompts": {
            "edge_tail": [1] * (checkpoint - 2),
            "edge_restore": [1] * (checkpoint + 1),
        },
        "tokens": {"edge_tail": [1] * 48, "edge_restore": [1] * 48},
        "sampling": dict(SAMPLING_KWARGS),
        "block_size": block_size,
        "prefix_cache_retention_interval": None,
        "verify": [],
        "caps": [],
        "admissions": [
            {"case": "edge_tail", "computed": 0},
            {"case": "edge_restore", "computed": checkpoint},
        ],
    }
    spec = deepcopy(reference)
    spec["verify"] = [
        {"case": "edge_tail", "start": checkpoint - 2, "span": 2, "committed": 2},
        {"case": "edge_restore", "start": checkpoint + 1, "span": 4, "committed": 1},
    ]
    spec["caps"] = [{"start": checkpoint - 2, "raw": 3, "cap": 1, "drafts": 1}]
    return reference, spec


@pytest.mark.parametrize("block_size", [160, 640])
def test_exact_outputs_with_observed_boundary_restore_pass(block_size):
    reference, spec = _arms(block_size)
    assert compare_arms(reference, spec, prefix_caching=True)["ok"]


@pytest.mark.parametrize("change", ["token", "shorter", "longer", "missing", "extra"])
def test_nonidentical_outputs_fail(change):
    reference, spec = _arms()
    if change == "token":
        spec["tokens"]["edge_tail"][1] = 9
    elif change == "shorter":
        spec["tokens"]["edge_tail"].pop()
    elif change == "longer":
        spec["tokens"]["edge_tail"].append(1)
    elif change == "missing":
        del spec["tokens"]["edge_tail"]
    else:
        spec["tokens"]["unexpected"] = [1] * 48
    report = compare_arms(reference, spec, prefix_caching=True)
    assert not report["ok"]
    assert report["mismatches"]


def test_matching_truncated_outputs_still_fail():
    reference, spec = _arms()
    reference["tokens"]["edge_tail"].pop()
    spec["tokens"]["edge_tail"].pop()
    assert not compare_arms(reference, spec, prefix_caching=True)["ok"]


@pytest.mark.parametrize("change", ["prompt", "sampling", "block_size", "retention"])
def test_matching_outputs_require_same_inputs_and_configuration(change):
    reference, spec = _arms()
    if change == "prompt":
        spec["prompts"]["edge_restore"][0] = 2
    elif change == "sampling":
        reference["sampling"]["logprobs"] = 5
    elif change == "block_size":
        spec["block_size"] *= 4
    else:
        spec["prefix_cache_retention_interval"] = 0
    assert not compare_arms(reference, spec, prefix_caching=True)["ok"]


@pytest.mark.parametrize("arm", ["reference", "spec"])
def test_an_unrelated_prefix_hit_does_not_prove_boundary_restore(arm):
    reference, spec = _arms()
    target = reference if arm == "reference" else spec
    target["admissions"][1]["computed"] = 160
    target["admissions"].append({"case": "other", "computed": 320})
    assert not compare_arms(reference, spec, prefix_caching=True)["ok"]


def test_boundary_producer_must_start_without_a_cached_prefix():
    reference, spec = _arms()
    spec["admissions"][0]["computed"] = 160
    assert not compare_arms(reference, spec, prefix_caching=True)["ok"]


@pytest.mark.parametrize("change", ["not_trimmed", "not_honored", "not_verified"])
def test_configuring_a_cap_does_not_prove_it_was_exercised(change):
    reference, spec = _arms()
    if change == "not_trimmed":
        spec["caps"][0]["raw"] = 1
    elif change == "not_honored":
        spec["caps"][0]["drafts"] = 3
    else:
        spec["verify"][0]["start"] += 1
    assert not compare_arms(reference, spec, prefix_caching=True)["ok"]


def test_boundary_checkpoint_must_be_reached_by_the_capped_verify():
    reference, spec = _arms()
    spec["verify"][0]["committed"] = 1
    # Full acceptance elsewhere does not cover the protected checkpoint.
    spec["verify"].append(
        {"case": "edge_restore", "start": 325, "span": 4, "committed": 4}
    )
    report = compare_arms(reference, spec, prefix_caching=True)
    assert not report["ok"]
    assert (
        "boundary probe did not fully accept the capped two-row span"
        in report["errors"]
    )


@pytest.mark.parametrize("missing_path", ["full", "rejected"])
def test_both_acceptance_paths_must_execute(missing_path):
    reference, spec = _arms()
    spec["verify"].pop(0 if missing_path == "full" else 1)
    assert not compare_arms(reference, spec, prefix_caching=False)["ok"]


def test_cache_disabled_arm_does_not_require_prefix_hits_or_caps():
    reference, spec = _arms()
    for arm in (reference, spec):
        arm["admissions"] = []
        arm["caps"] = []
    assert compare_arms(reference, spec, prefix_caching=False)["ok"]


def test_restore_prompt_uses_generated_history_and_leaves_one_uncached_token():
    prompt = list(range(318))
    generated = [900, 901, 902, 903]
    assert build_restore_prompt(prompt, generated, 320) == prompt + generated[:3]


def test_restore_prompt_rejects_incomplete_producer():
    with pytest.raises(ValueError, match="enough tokens"):
        build_restore_prompt([1] * 318, [2, 3], 320)
