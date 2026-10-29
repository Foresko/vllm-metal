# SPDX-License-Identifier: Apache-2.0
"""Mamba-2 mixers on paged recurrent state.

The wrapper drives the unmodified mlx_lm mixer and hands it a per-step
``ArraysCache`` stand-in whose reads and writes address rows of the shared
state pools, so mlx_lm keeps owning the conv and SSM math.
"""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn

from vllm_metal.attention.caches.state_cache import (
    PagedStateCache,
    StateSlotArraysCache,
)
from vllm_metal.attention.context import PagedAttentionContext, get_context

# mlx_lm's single-token ssm_update kernel covers this many state columns per
# thread; a state width off that grid is left unwritten.
_SSM_KERNEL_STATE_COLUMNS = 32


def is_mamba2_mixer(module: nn.Module) -> bool:
    """Return True for an mlx_lm Mamba-2 mixer.

    Matches the surface the wrapper reads (``conv1d`` and ``ssm_state_size``),
    which GDN modules do not carry.
    """
    return hasattr(module, "conv1d") and hasattr(module, "ssm_state_size")


@dataclass(frozen=True, slots=True)
class _Mamba2Step:
    cu_seqlens: list[int]
    slot_ids: list[int]
    num_requests: int
    num_decode_requests: int


class Mamba2PagedStateWrapper(nn.Module):
    """Runs an mlx_lm Mamba-2 mixer against per-request rows of the state pools.

    Prefill-containing steps run one request at a time (the SSD path keeps
    one state per batch row and mlx_lm drops ``lengths`` before
    ``ssm_update``); decode-only steps run as one batch gathered by slot.
    Without a ``PagedAttentionContext`` the call delegates to the mixer as is.
    """

    def __init__(
        self,
        inner: nn.Module,
        layer_idx: int,
        cache_idx: int,
        state_cache: PagedStateCache,
    ) -> None:
        super().__init__()
        self._validate_ssm_state_size(inner, layer_idx)
        state_cache.require_mixer_dtype(inner.conv1d.weight.dtype, layer_idx=layer_idx)
        object.__setattr__(self, "_inner", inner)
        object.__setattr__(self, "_mamba2_layer_idx", layer_idx)
        object.__setattr__(self, "_mamba2_cache_idx", cache_idx)
        object.__setattr__(self, "_mamba2_state_cache", state_cache)

    def rebind_state_cache(
        self, state_cache: PagedStateCache, *, cache_idx: int
    ) -> None:
        """Refresh pooled state refs in place (cached model reuse)."""
        object.__setattr__(self, "_mamba2_cache_idx", cache_idx)
        object.__setattr__(self, "_mamba2_state_cache", state_cache)

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | str | None = None,
        cache: object | None = None,
    ) -> mx.array:
        ctx = get_context()
        if ctx is None:
            return self._inner(x, mask=mask, cache=cache)
        step = self._prepare_step(ctx)
        if step.num_decode_requests == step.num_requests:
            return self._run_decode(x, step)
        return self._run_requests(x, step)

    def _prepare_step(self, ctx: PagedAttentionContext) -> _Mamba2Step:
        cu_seqlens = ctx.cu_seqlens
        if cu_seqlens is None or len(cu_seqlens) < 2:
            raise RuntimeError("Mamba-2 wrapper requires cu_seqlens in context")
        num_requests = len(cu_seqlens) - 1
        slot_ids = self._mamba2_state_cache.step_slot_ids(
            ctx, self._mamba2_cache_idx, num_requests
        )
        return _Mamba2Step(
            cu_seqlens=cu_seqlens,
            slot_ids=slot_ids,
            num_requests=num_requests,
            num_decode_requests=ctx.num_decode_requests,
        )

    def _run_decode(self, x: mx.array, step: _Mamba2Step) -> mx.array:
        rows = x.reshape(step.num_requests, 1, x.shape[-1])
        slot_ids = mx.array(step.slot_ids, dtype=mx.int32)
        cache = self._slot_cache(slot_ids)
        y = self._inner(rows, mask=None, cache=cache)
        cache.flush()
        return y.reshape(1, step.num_requests, -1)

    def _run_requests(self, x: mx.array, step: _Mamba2Step) -> mx.array:
        # Each slice is one unpadded request, so the mixer needs no mask.
        outputs = []
        for request_idx, slot in enumerate(step.slot_ids):
            start = step.cu_seqlens[request_idx]
            end = step.cu_seqlens[request_idx + 1]
            slot_ids = mx.array([slot], dtype=mx.int32)
            cache = self._slot_cache(slot_ids)
            outputs.append(self._inner(x[:, start:end], mask=None, cache=cache))
            cache.flush()
        return mx.concatenate(outputs, axis=1)

    def _slot_cache(self, slot_ids: mx.array) -> StateSlotArraysCache:
        return StateSlotArraysCache(
            self._mamba2_state_cache, self._mamba2_cache_idx, slot_ids
        )

    @staticmethod
    def _validate_ssm_state_size(inner: nn.Module, layer_idx: int) -> None:
        if inner.ssm_state_size % _SSM_KERNEL_STATE_COLUMNS != 0:
            raise NotImplementedError(
                f"Mamba-2 paged decode at layer {layer_idx} requires ssm_state_size "
                f"to be a multiple of {_SSM_KERNEL_STATE_COLUMNS}, got "
                f"ssm_state_size={inner.ssm_state_size}."
            )
