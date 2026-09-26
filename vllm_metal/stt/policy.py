# SPDX-License-Identifier: Apache-2.0
"""STT-specific scheduler policy at the platform boundary."""

from __future__ import annotations

from typing import Protocol

# Nominal memory reported to vLLM scheduler for STT models.
# No KV cache is actually allocated; this just passes minimum-memory checks.
STT_SCHED_AVAILABLE_BYTES = 1 << 30  # 1 GiB

# Block size reported to vLLM for STT models (minimal, no real KV cache).
STT_SCHED_BLOCK_BYTES = 1

# Nominal head size for the placeholder KV spec used only to satisfy
# vLLM scheduler initialization for STT models.
STT_SCHED_NOMINAL_HEAD_SIZE = 64


class _ModelConfigLike(Protocol):
    model: str
    tokenizer: str | None


class _SchedulerConfigLike(Protocol):
    async_scheduling: bool


class _CacheConfigLike(Protocol):
    enable_prefix_caching: bool


def apply_stt_scheduler_policy(
    model_config: _ModelConfigLike,
    scheduler_config: _SchedulerConfigLike,
    cache_config: _CacheConfigLike,
) -> None:
    """Apply STT scheduler compatibility policy for Metal runtime.

    STT requests are processed as one-shot execute calls, so async scheduling
    (which expects decode-phase queuing) must be disabled.

    The one-shot runner keeps no KV cache, so a prefix-cache hit has nothing
    to reuse, and vLLM strips the audio features of any request whose audio
    placeholder the hit covers. Prefix caching must be disabled so every
    request reaches the runner with its audio.
    """
    if not model_config.tokenizer:
        model_config.tokenizer = model_config.model
    if scheduler_config.async_scheduling:
        scheduler_config.async_scheduling = False
    if cache_config.enable_prefix_caching:
        cache_config.enable_prefix_caching = False
