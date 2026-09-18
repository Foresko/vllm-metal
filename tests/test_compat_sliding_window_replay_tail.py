# SPDX-License-Identifier: Apache-2.0
"""Prefix-cache hits for a prompt that extends a cached one by a few tokens.

vLLM 0.29 defaults ``prefix_cache_retention_interval`` to 0: a sliding-window
group keeps only the window-sized tail of blocks ending at the prompt's
replay boundary, aligned down to the hybrid alignment (32 tokens for Gemma 4:
full-attention blocks of 32, sliding blocks of 16).  A later request whose
tokens diverge inside that last aligned block -- the same user turn with a
few words appended -- gets a full-attention hit one aligned block shorter,
and the sliding lookup then finds only 62-63 contiguous cached blocks where
it needs 64: no match, zero hit, the whole prompt recomputed.  Measured on
production Gemma 4 26B: 4 100-token prompt, +3 tokens -> 0 cached, 2.2 s;
4 110 -> 4 096 cached, 0.13 s.  ``apply_compat_patches`` retains one extra
alignment block at every reachable boundary so the run below the shorter
hit is long enough.
"""

from __future__ import annotations

import hashlib
import random

import pytest
import torch
from vllm.sampling_params import SamplingParams
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheTensor,
    SlidingWindowSpec,
)
from vllm.v1.request import Request

from vllm_metal.compat import apply_compat_patches

HASH_BLOCK = 16


def _hash_fn(value):
    return hashlib.sha256(repr(value).encode()).digest()


def _manager(retention_interval: int | None) -> KVCacheManager:
    """Gemma 4 26B's production layout: 5 full-attention layers at block 32,
    25 sliding layers (window 1024) in five groups at block 16."""
    full = FullAttentionSpec(
        block_size=32, num_kv_heads=2, head_size=512, dtype=torch.bfloat16
    )
    slide = SlidingWindowSpec(
        block_size=16,
        num_kv_heads=8,
        head_size=256,
        dtype=torch.bfloat16,
        sliding_window=1024,
    )
    full_layers = [f"full{i}" for i in range(5)]
    slide_layers = [f"slide{i}" for i in range(25)]
    groups = [KVCacheGroupSpec(full_layers, full)] + [
        KVCacheGroupSpec(slide_layers[g * 5 : (g + 1) * 5], slide) for g in range(5)
    ]
    tensors = [
        KVCacheTensor(
            size=24882 * full.page_size_bytes,
            layers=[layer],
            layer_stride=0,
            block_stride=full.page_size_bytes,
        )
        for layer in full_layers + slide_layers
    ]
    config = KVCacheConfig(
        num_blocks=24882,
        kv_cache_tensors=tensors,
        kv_cache_groups=groups,
        prefix_cache_retention_interval=retention_interval,
    )
    return KVCacheManager(
        config,
        max_model_len=65536,
        scheduler_block_size=32,
        hash_block_size=HASH_BLOCK,
        enable_caching=True,
    )


def _hit_after_extension(
    manager: KVCacheManager, length: int, replaced_tail: int = 6
) -> int:
    """Cache a prompt of ``length`` tokens as a finished one-step request, then
    look up a prompt equal to it except for the last ``replaced_tail`` tokens
    plus three appended: how the same user turn looks with words appended."""
    hasher = get_request_block_hasher(HASH_BLOCK, _hash_fn)
    rng = random.Random(length)
    tokens = [rng.randrange(1000, 200000) for _ in range(length)]
    params = SamplingParams(max_tokens=1)
    cold = Request("cold", tokens, params, None, block_hasher=hasher)
    blocks, computed, _ = manager.get_computed_blocks(cold)
    assert manager.allocate_slots(cold, length - computed, computed, blocks) is not None
    cold.num_computed_tokens = length
    cold.append_output_token_ids(7)
    manager.free(cold)
    extended = tokens[: length - replaced_tail] + [
        rng.randrange(1000, 200000) for _ in range(replaced_tail + 3)
    ]
    tail = Request("tail", extended, params, None, block_hasher=hasher)
    return manager.get_computed_blocks(tail)[1]


@pytest.fixture(autouse=True)
def _none_hash():
    init_none_hash(_hash_fn)


@pytest.mark.parametrize(
    ("length", "expected"),
    [
        # Divergence inside the last aligned block: missed entirely before.
        (4100, 4064),
        (6980, 6944),
        (8194, 8160),
        # Divergence after the aligned boundary: hit before, still hits.
        (4110, 4096),
        (6990, 6976),
        (16383, 16352),
    ],
)
def test_extending_a_cached_prompt_hits_with_default_retention(
    length, expected
) -> None:
    apply_compat_patches()
    assert _hit_after_extension(_manager(retention_interval=0), length) == expected


def test_dense_retention_is_unchanged() -> None:
    """``None`` caches every block; the patch must not touch that path."""
    apply_compat_patches()
    assert _hit_after_extension(_manager(retention_interval=None), 4100) == 4064
