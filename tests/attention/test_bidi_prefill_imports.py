# SPDX-License-Identifier: Apache-2.0
"""``bidi_prefill`` must not pull in sdpa, the runner or the worker."""

from __future__ import annotations

import subprocess
import sys
import textwrap


def test_bidi_prefill_import_boundary() -> None:
    code = textwrap.dedent(
        """
        import sys

        import vllm_metal.attention.impls.bidi_prefill

        for name in (
            "vllm_metal.attention.impls.sdpa",
            "vllm_metal.v1.model_runner",
            "vllm_metal.v1.worker",
        ):
            if name in sys.modules:
                raise SystemExit(f"{name} was imported")
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", code], check=False, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
