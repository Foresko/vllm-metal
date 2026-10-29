# SPDX-License-Identifier: Apache-2.0
"""The in-process registry fixture must not outlive the module requesting it.

``inspect_vllm_models_in_process`` replaces vLLM's ``_run_in_subprocess`` for
``test_gguf_vllm_integration.py`` only. This module does not request it, so the
original must be in place here whether it runs alone or after the GGUF module.
"""

from __future__ import annotations

from vllm.model_executor.models import registry


def test_registry_subprocess_is_the_original() -> None:
    fn = registry._run_in_subprocess
    assert fn.__name__ == "_run_in_subprocess"
    assert fn.__module__ == registry.__name__
