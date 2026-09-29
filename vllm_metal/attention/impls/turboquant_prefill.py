# SPDX-License-Identifier: Apache-2.0
"""Bounded TurboQuant prefill admission using scheduler-owned page metadata."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import mlx.core as mx
from vllm.logger import init_logger

from vllm_metal.attention.caches.turboquant import (
    FWHT_SUPPORTED_HEAD_DIMS,
    prefill_bytes_per_token,
)
from vllm_metal.attention.context import PagedAttentionContext

if TYPE_CHECKING:
    from vllm_metal.attention.impls.sdpa import _KernelMetadata

logger = init_logger(__name__)

# The paged attention Metal kernel is template-instantiated for these block
# sizes only.  Sorted descending so the kernel block size picker selects the
# largest valid divisor first, minimising the block-table expansion ratio.
KERNEL_BLOCK_SIZES = (32, 16, 8)

# Dequantizing cached context must be amortized by enough query rows. Keep
# short suffixes (including prefix hits) on the compressed kernel. MHA needs
# more rows because it cannot amortize dequant across grouped query heads.
# See tools/benchmark/tq_lane_verify.py's production-path crossover sweep.
_TQ_MIN_PREFILL_TOKENS = 128
_TQ_MIN_QUERIES_PER_KV_HEAD = 256


def min_prefill_tokens(num_query_heads: int, num_kv_heads: int, head_dim: int) -> int:
    """Amortize materialization across query rows and grouped query heads."""
    return max(
        _TQ_MIN_PREFILL_TOKENS,
        # Wide-head tiled prefill needs more rows to amortize materialization.
        head_dim // 2,
        (_TQ_MIN_QUERIES_PER_KV_HEAD * num_kv_heads + num_query_heads - 1)
        // num_query_heads,
    )


def unsupported_reason(
    *,
    dtype: mx.Dtype,
    head_dim: int,
    kernel_block_size: int,
    cache_block_size: int,
    stored_block_size: int,
) -> str | None:
    if dtype not in (mx.bfloat16, mx.float16):
        return f"activation dtype {dtype} requires FP16/BF16"
    if head_dim not in FWHT_SUPPORTED_HEAD_DIMS:
        return f"head_dim {head_dim} is unsupported"
    if (
        kernel_block_size not in KERNEL_BLOCK_SIZES
        or stored_block_size != cache_block_size
        or cache_block_size % kernel_block_size
    ):
        return (
            f"cache layout (stored={stored_block_size}, scheduler={cache_block_size}, "
            f"kernel={kernel_block_size}) is unsupported"
        )
    return None


def workspace_upper_bound(
    *,
    max_model_len: int,
    max_num_seqs: int,
    max_num_batched_tokens: int,
    num_query_heads: int,
    num_kv_heads: int,
    head_dim: int,
    block_size: int,
) -> int:
    """Conservative per-layer cap, including independent histories and split copies.

    Queries are limited by a scheduler step; the historical KV each can read
    is limited by max_model_len. Layers reuse one allowance, so do not sum
    this bound over model layers. The minimum kernel page is eight tokens;
    round histories to a full scheduler page to cover every translated view.
    """
    queries = min(max_num_batched_tokens, max_num_seqs * max_model_len)
    minimum = min_prefill_tokens(num_query_heads, num_kv_heads, head_dim)
    if max_model_len < minimum:
        return 0
    candidates = min(max_num_seqs, queries // minimum)
    if not candidates:
        return 0
    tokens = ((max_model_len + block_size - 1) // block_size) * block_size
    table_width = tokens // min(KERNEL_BLOCK_SIZES)
    materialized = candidates * (
        tokens * prefill_bytes_per_token(num_kv_heads, head_dim) + table_width * 4
    )
    if max_num_seqs == 1:
        return materialized
    routing = queries * (3 * num_query_heads * head_dim * 2 + 2 * 4)
    split_metadata = (2 * max_num_seqs + 2) * 4 + max_num_seqs * (table_width + 1) * 4
    return materialized + routing + split_metadata


@dataclass(frozen=True, eq=False)
class _AttentionBatch:
    query_indices: mx.array | None
    block_tables: mx.array
    seq_lens: mx.array
    cu_seqlens_q: mx.array
    max_seq_len: int


@dataclass(frozen=True, eq=False)
class _TurboQuantPrefillPlan:
    prefill: _AttentionBatch
    fallback: _AttentionBatch | None
    restore_indices: mx.array | None
    pool_pages: mx.array
    pool_offsets: mx.array
    workspace_bytes: int


def _turboquant_prefill_plan(
    ctx: PagedAttentionContext,
    meta: _KernelMetadata,
    raw_block_tables: list[list[int]],
    cache_block_size: int,
    num_query_heads: int,
    num_kv_heads: int,
    head_dim: int,
) -> _TurboQuantPrefillPlan | None:
    """Plan bounded materialization using the existing CPU scheduler metadata.

    Only referenced kernel blocks are gathered; shared prefixes are deduplicated.
    Admit independent histories as well as shared prefixes while the combined
    gather fits the absolute workspace reserved before KV allocation. An
    oversized candidate is skipped before building its gather indices.
    """
    workspace_limit = meta.tq_prefill_workspace_bytes
    if not workspace_limit:
        return None
    min_tokens = min_prefill_tokens(num_query_heads, num_kv_heads, head_dim)
    key = (
        cache_block_size,
        min_tokens,
        num_query_heads,
        num_kv_heads,
        head_dim,
        workspace_limit,
    )
    if key in meta.tq_prefill_plans:
        return meta.tq_prefill_plans[key]
    if ctx.cu_seqlens is None:
        raise ValueError("TurboQuant prefill requires cumulative query lengths")
    cu_seqlens = ctx.cu_seqlens
    lengths = [b - a for a, b in zip(cu_seqlens[:-1], cu_seqlens[1:], strict=True)]
    candidates = [i for i, length in enumerate(lengths) if length >= min_tokens]
    if not candidates:
        meta.tq_prefill_plans[key] = None
        return None

    kernel_bs = meta.block_size
    ratio = cache_block_size // kernel_bs
    bytes_per_block = kernel_bs * prefill_bytes_per_token(num_kv_heads, head_dim)
    # Splitting also gathers Q and concatenates/restores the outputs. Charge
    # those three FP16/BF16 copies plus query-gather and output-restore indices.
    query_row_bytes = num_query_heads * head_dim * 2  # FP16/BF16 element width.
    routing_bytes = cu_seqlens[-1] * (3 * query_row_bytes + 2 * 4)

    def split_metadata_bytes(fallback_count: int) -> int:
        # Split seq_lens have N int32s; the two cu_seqlens_q have N + 2.
        # Each fallback needs a padded table row and its int32 gather index.
        sequence_bytes = (2 * len(lengths) + 2) * 4
        table_bytes = fallback_count * (meta.block_tables.shape[1] + 1) * 4
        return sequence_bytes + table_bytes

    def select_rows(
        limit: int,
        *,
        split: bool,
    ) -> tuple[dict[int, int], list[int], list[list[int]], int]:
        block_limit = max(0, limit) // bytes_per_block
        source_blocks: dict[int, int] = {}
        prefill_ids: list[int] = []
        rows: list[list[int]] = []
        table_width = 0
        for i in candidates:
            num_blocks = (ctx.context_lens[i] + kernel_bs - 1) // kernel_bs
            if num_blocks > block_limit:
                continue
            sources = [
                raw_block_tables[i][j // ratio] * ratio + j % ratio
                for j in range(num_blocks)
            ]
            # Inspect only new blocks, without copying all admitted histories.
            additions = dict.fromkeys(
                source for source in sources if source not in source_blocks
            )
            next_width = max(table_width, num_blocks)
            next_bytes = (len(source_blocks) + len(additions)) * bytes_per_block
            next_bytes += (len(rows) + 1) * next_width * 4
            if split:
                next_bytes += split_metadata_bytes(len(lengths) - len(rows) - 1)
            if next_bytes > limit:
                continue
            row = []
            for source in sources:
                if source not in source_blocks:
                    source_blocks[source] = len(source_blocks)
                row.append(source_blocks[source])
            prefill_ids.append(i)
            rows.append(row)
            table_width = next_width
        return source_blocks, prefill_ids, rows, table_width

    # First allow the whole batch to fit without copies. If a split becomes
    # necessary, repeat CPU-only admission with its routing cost reserved.
    must_split = len(candidates) < len(lengths)
    source_blocks, prefill_ids, rows, table_width = select_rows(
        workspace_limit - (routing_bytes if must_split else 0), split=must_split
    )
    if not must_split and 0 < len(prefill_ids) < len(lengths):
        source_blocks, prefill_ids, rows, table_width = select_rows(
            workspace_limit - routing_bytes, split=True
        )
    if len(prefill_ids) < len(candidates):
        logger.info_once(
            "Metal: TurboQuant prefill workspace limit reached; "
            "overflow histories use compressed attention."
        )
    if not prefill_ids:
        meta.tq_prefill_plans[key] = None
        return None
    selected = set(prefill_ids)
    fallback_ids = [i for i in range(len(lengths)) if i not in selected]
    query_order: list[int] = []

    def batch(ids: list[int], tables: mx.array) -> _AttentionBatch:
        if not fallback_ids:
            # The whole batch keeps its original query order and metadata.
            return _AttentionBatch(
                None, tables, meta.seq_lens, meta.cu_seqlens_q, meta.max_seq_len
            )
        indices = [q for i in ids for q in range(cu_seqlens[i], cu_seqlens[i + 1])]
        query_order.extend(indices)
        cu = [0]
        for i in ids:
            cu.append(cu[-1] + lengths[i])
        return _AttentionBatch(
            mx.array(indices, dtype=mx.int32),
            tables,
            mx.array([ctx.context_lens[i] for i in ids], dtype=mx.int32),
            mx.array(cu, dtype=mx.int32),
            max(ctx.context_lens[i] for i in ids),
        )

    tables = mx.array([row + [0] * (table_width - len(row)) for row in rows], mx.int32)
    prefill = batch(prefill_ids, tables)
    fallback = (
        batch(fallback_ids, meta.block_tables[mx.array(fallback_ids, mx.int32)])
        if fallback_ids
        else None
    )
    restore = None
    if fallback_ids:
        inverse = [0] * len(query_order)
        for packed, original in enumerate(query_order):
            inverse[original] = packed
        restore = mx.array(inverse, dtype=mx.int32)
    blocks = mx.array(list(source_blocks), dtype=mx.int32)
    # Kernel block IDs, including translated IDs, must fit int32 as required
    # by the primitive. Multiplying those IDs by kernel_bs may still overflow;
    # gather instead with scheduler-page and within-page token coordinates.
    pages = mx.repeat(blocks // ratio, kernel_bs)
    offsets = (
        (blocks % ratio)[:, None] * kernel_bs + mx.arange(kernel_bs, dtype=mx.int32)
    ).reshape(-1)
    plan = _TurboQuantPrefillPlan(
        prefill,
        fallback,
        restore,
        pages,
        offsets,
        len(source_blocks) * bytes_per_block
        + len(rows) * table_width * 4
        + (
            routing_bytes + split_metadata_bytes(len(fallback_ids))
            if fallback_ids
            else 0
        ),
    )
    meta.tq_prefill_plans[key] = plan
    return plan
