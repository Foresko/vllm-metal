# SPDX-License-Identifier: Apache-2.0
"""Kill switch, force switch and counters of the GQA-packed decode kernel."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import vllm_metal.envs as envs
from vllm_metal.metal import _configure_gqa_decode


def _ops():
    state = {"enabled": True}
    return SimpleNamespace(
        set_gqa_decode_enabled=Mock(
            side_effect=lambda enabled: state.__setitem__("enabled", enabled)
        ),
        gqa_decode_ready=Mock(side_effect=lambda: state["enabled"]),
    )


def test_disable_gqa_decode_env_is_a_negative_override(monkeypatch) -> None:
    monkeypatch.delenv("VLLM_METAL_DISABLE_GQA_DECODE", raising=False)
    assert envs.VLLM_METAL_DISABLE_GQA_DECODE is False

    monkeypatch.setenv("VLLM_METAL_DISABLE_GQA_DECODE", "1")
    assert envs.VLLM_METAL_DISABLE_GQA_DECODE is True


@pytest.mark.parametrize("disabled", [True, False])
def test_configure_applies_the_kill_switch(disabled: bool) -> None:
    ops = _ops()
    assert _configure_gqa_decode(ops, disabled=disabled) is (not disabled)  # type: ignore[arg-type]
    ops.set_gqa_decode_enabled.assert_called_once_with(not disabled)


def test_native_switches_round_trip() -> None:
    from vllm_metal.metal import get_ops

    ops = get_ops()
    ops.set_gqa_decode_enabled(False)
    try:
        assert ops.gqa_decode_ready() is False
    finally:
        ops.set_gqa_decode_enabled(True)
    assert ops.gqa_decode_ready() is True

    single, partitioned = ops.gqa_decode_dispatch_counts()
    assert single >= 0 and partitioned >= 0

    for size in (-1, 0, 256, 512):
        ops.set_gqa_decode_partition_size(size)
    ops.set_gqa_decode_partition_size(-1)
    with pytest.raises(ValueError):
        ops.set_gqa_decode_partition_size(128)

    ops.set_gqa_decode_force(True)
    ops.set_gqa_decode_force(False)
