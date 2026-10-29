# SPDX-License-Identifier: Apache-2.0
"""run_in_spawn_process reports a child's signal death by name."""

import signal

import pytest


def _die_sigterm() -> None:
    signal.raise_signal(signal.SIGTERM)


def test_spawn_failure_reports_the_signal_name(run_in_spawn_process):
    with pytest.raises(AssertionError, match=r"sigterm-child:.*signal 15 \(SIGTERM\)"):
        run_in_spawn_process(_die_sigterm, label="sigterm-child")
