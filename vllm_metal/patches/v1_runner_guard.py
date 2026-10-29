# SPDX-License-Identifier: Apache-2.0
"""Let Metal-served features past vLLM's V1 GPU-runner support check."""

from collections.abc import Callable
from functools import wraps
from typing import Any


def allow_v1_runner_feature(feature: str, allowed: Callable[[Any], bool]) -> None:
    """Drop ``feature`` from ``VllmConfig._get_v1_model_runner_unsupported_features``.

    vLLM reports features its V1 GPU runner lacks, and ``MetalModelRunner``
    serves some of them itself. One wrapper holds every rule, so independent
    bridges (DSpark, diffusion) share it instead of stacking their own; the
    rules live on that wrapper, so restoring the method drops them too.
    ``allowed(vllm_config)`` decides per config; a feature without a matching
    rule still fails upstream's check. Install from platform config
    validation, after ``VllmConfig`` is fully imported: importing
    ``vllm.config`` at plugin registration is circular.
    """
    from vllm.config import VllmConfig

    original = getattr(VllmConfig, "_get_v1_model_runner_unsupported_features", None)
    if original is None:
        return
    # Feature label in vLLM's unsupported list -> whether Metal serves it.
    rules: dict[str, Callable[[Any], bool]] | None = getattr(
        original, "_metal_v1_runner_rules", None
    )
    if rules is not None:
        rules[feature] = allowed
        return
    rules = {feature: allowed}

    @wraps(original)
    def unsupported_features(self: Any) -> list[str]:
        return [
            item for item in original(self) if not (item in rules and rules[item](self))
        ]

    unsupported_features._metal_v1_runner_rules = rules  # type: ignore[attr-defined]
    VllmConfig._get_v1_model_runner_unsupported_features = unsupported_features
