# SPDX-License-Identifier: Apache-2.0
"""Kimi Delta Attention with scheduler-managed recurrent state."""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn

from vllm_metal.attention.caches.state_cache import (
    PagedStateCache,
    StateSlotArraysCache,
)
from vllm_metal.attention.context import PagedAttentionContext, get_context


def is_kda_attention(module: nn.Module) -> bool:
    """Return whether a module exposes the KDA interface used by the wrapper."""
    return all(
        hasattr(module, name)
        for name in (
            "q_proj",
            "k_proj",
            "v_proj",
            "q_conv1d",
            "k_conv1d",
            "v_conv1d",
            "projection_size",
            "conv_kernel_size",
        )
    )


@dataclass(frozen=True, slots=True)
class _KDAStep:
    cu_seqlens: list[int]
    slot_ids: list[int]
    num_requests: int
    num_decode_requests: int


class KDAPagedAttentionWrapper(nn.Module):
    """Run packed KDA segments against scheduler-managed state slots."""

    def __init__(
        self,
        inner: nn.Module,
        layer_idx: int,
        cache_idx: int,
        state_cache: PagedStateCache,
    ) -> None:
        super().__init__()
        if not is_kda_attention(inner):
            raise TypeError(f"{type(inner).__name__} is not a KDA module")
        object.__setattr__(self, "_inner", inner)
        object.__setattr__(self, "_kda_layer_idx", layer_idx)
        self.rebind_state_cache(state_cache, cache_idx=cache_idx)

    def _validate_state_cache(
        self, state_cache: PagedStateCache, cache_idx: int
    ) -> None:
        inner = self._inner
        projection_size = int(inner.projection_size)
        expected_conv = (int(inner.conv_kernel_size) - 1, 3 * projection_size)
        actual_conv = state_cache.conv_states[cache_idx].shape[1:]
        if actual_conv != expected_conv:
            raise RuntimeError(
                f"KDA conv state shape {actual_conv} does not match {expected_conv}"
            )
        expected_recurrent = (
            int(inner.num_heads),
            int(inner.head_dim),
            int(inner.head_dim),
        )
        actual_recurrent = state_cache.recurrent_states[cache_idx].shape[1:]
        if actual_recurrent != expected_recurrent:
            raise RuntimeError(
                f"KDA recurrent state shape {actual_recurrent} does not match "
                f"{expected_recurrent}"
            )
        state_cache.require_mixer_dtype(
            inner.q_conv1d.weight.dtype, layer_idx=self._kda_layer_idx
        )

    def rebind_state_cache(
        self, state_cache: PagedStateCache, *, cache_idx: int
    ) -> None:
        """Refresh pooled state references for a cached model."""
        self._validate_state_cache(state_cache, cache_idx)
        object.__setattr__(self, "_kda_cache_idx", cache_idx)
        object.__setattr__(self, "_kda_state_cache", state_cache)

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | None = None,
        cache: nn.Module | None = None,
    ) -> mx.array:
        ctx = get_context()
        if ctx is None:
            return self._inner(x, mask=mask, cache=cache)

        step = self._prepare_step(ctx)
        self._kda_state_cache.apply_pending_conv_state(self._kda_cache_idx)
        self._kda_state_cache.apply_pending_recurrent_state(self._kda_cache_idx)
        if step.num_decode_requests == step.num_requests:
            return self._run_decode(x, step)
        return self._run_requests(x, step)

    def _prepare_step(self, ctx: PagedAttentionContext) -> _KDAStep:
        cu_seqlens = ctx.cu_seqlens
        if cu_seqlens is None or len(cu_seqlens) < 2:
            raise RuntimeError("KDA wrapper requires cu_seqlens")
        num_requests = len(cu_seqlens) - 1
        return _KDAStep(
            cu_seqlens=cu_seqlens,
            slot_ids=self._kda_state_cache.step_slot_ids(
                ctx, self._kda_cache_idx, num_requests
            ),
            num_requests=num_requests,
            num_decode_requests=ctx.num_decode_requests,
        )

    def _run_decode(self, x: mx.array, step: _KDAStep) -> mx.array:
        rows = x.reshape(step.num_requests, 1, x.shape[-1])
        cache = self._slot_cache(mx.array(step.slot_ids, dtype=mx.int32))
        output = self._inner(rows, mask=None, cache=cache)
        cache.flush()
        return output.reshape(1, step.num_requests, -1)

    def _run_requests(self, x: mx.array, step: _KDAStep) -> mx.array:
        outputs = []
        for req_idx, slot in enumerate(step.slot_ids):
            start = step.cu_seqlens[req_idx]
            end = step.cu_seqlens[req_idx + 1]
            cache = self._slot_cache(mx.array([slot], dtype=mx.int32))
            outputs.append(self._inner(x[:, start:end, :], mask=None, cache=cache))
            cache.flush()
        return mx.concatenate(outputs, axis=1)

    def _slot_cache(self, slot_ids: mx.array) -> StateSlotArraysCache:
        projection_size = int(self._inner.projection_size)
        return StateSlotArraysCache(
            self._kda_state_cache,
            self._kda_cache_idx,
            slot_ids,
            conv_widths=(projection_size,) * 3,
        )
