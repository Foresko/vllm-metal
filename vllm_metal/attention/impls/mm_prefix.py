# SPDX-License-Identifier: Apache-2.0
"""Per-query image-block ranges for the tiled prefill kernel (Gemma 4 vision).

Mirrors vLLM's ``fill_mm_prefix_query_ranges`` (``vllm/v1/attention/backends/
utils.py``): every query row of the batch carries the inclusive absolute
``[start, end]`` of the image block it lies in, ``(-1, -1)`` otherwise.  The
kernel unmasks ``start <= key <= end`` for such rows on top of the causal rule
and ANDs the sliding window afterwards, which is HF's
``(causal OR same_block) AND window``.

Leaf module: numpy and ``vllm_metal.envs`` only, so the attention impl, the
model runner and tests can import it without MLX or torch.
"""

from __future__ import annotations

import functools

import numpy as np

import vllm_metal.envs as envs

MM_PREFIX_PATHS = envs.MM_PREFIX_PATHS


def build_mm_prefix_rows(
    cu_seqlens: list[int],
    context_lens: list[int],
    segment_bidi_ranges: list[list[tuple[int, int]] | None],
) -> np.ndarray | None:
    """``(L, 2)`` int32 rows of inclusive absolute block bounds, ``(-1, -1)`` elsewhere.

    Ranges are half-open ``[r0, r1)`` on input (the runner's convention) and
    stored inclusive as ``(r0, r1 - 1)``.  Degenerate ranges (``r1 <= r0``) and
    one-token blocks are skipped: a lone token's block row is its causal row,
    and vLLM likewise drops an inclusive ``(p, p)``.  Returns ``None`` when no
    query row lies inside a block, so the caller keeps the kernel's plain path;
    a batch of one-token blocks (a DiffusionGemma canvas of length 1, or one
    truncated at ``max_model_len``) must not reach the tiled kernel, which
    refuses ranges without a multi-token segment.
    """
    total = cu_seqlens[-1]
    rows = np.full((total, 2), -1, dtype=np.int32)
    found = False
    for i, ranges in enumerate(segment_bidi_ranges):
        if not ranges:
            continue
        q_start = cu_seqlens[i]
        n = cu_seqlens[i + 1] - q_start
        q_lo = context_lens[i] - n
        for r0, r1 in ranges:
            if r1 - r0 <= 1:
                continue
            a, b = max(r0, q_lo), min(r1, q_lo + n)
            if b <= a:
                continue
            rows[q_start + (a - q_lo) : q_start + (b - q_lo)] = (r0, r1 - 1)
            found = True
    return rows if found else None


def resolve_mm_prefix_path(value: str | None, supported: bool) -> str:
    """``"kernel"`` or ``"recompute"`` for ``VLLM_METAL_MM_PREFIX_PATH``.

    ``None`` (unset) means ``"kernel"``.  The kernel path needs the compiled
    ops to advertise ``supports_mm_prefix`` and becomes ``"recompute"``
    otherwise, with a warning the first time.  Any other value raises so an
    A/B typo cannot quietly measure the wrong path.
    """
    if value is None:
        value = "kernel"
    if value not in MM_PREFIX_PATHS:
        raise ValueError(
            f"VLLM_METAL_MM_PREFIX_PATH must be 'kernel' or 'recompute', got {value!r}"
        )
    if value == "kernel" and not supported:
        _warn_kernel_path_unavailable()
        return "recompute"
    return value


def mm_prefix_path(ops: object) -> str:
    """The configured image-block attention path for these compiled ops.

    The native ops always advertise ``supports_mm_prefix``, so the probe only
    fails for a build that predates it (or a test fake).  Such ops serve image
    blocks through the recompute: the attention impl passes
    ``mm_prefix_ranges=`` only on the kernel path, and
    ``resolve_mm_prefix_path`` warns once about the fallback.
    """
    supported = bool(getattr(ops, "supports_mm_prefix", lambda: False)())
    return resolve_mm_prefix_path(envs.VLLM_METAL_MM_PREFIX_PATH, supported)


def image_block_path(ops: object, *, float32_cache: bool) -> str:
    """The path image blocks take with these compiled ops and this KV cache.

    ``mm_prefix_path``, except that a float32 cache keeps the recompute: the
    tiled kernel has no float32 instantiation, so handing it the ranges would
    reach the primitive's eager ValueError mid-request.  The attention impl
    calls this per forward; the model runner calls it once at warm-up, so a
    bad value or an old build shows at startup and the log names the path.
    """
    path = mm_prefix_path(ops)
    return "recompute" if float32_cache else path


@functools.cache
def _warn_kernel_path_unavailable() -> None:
    """Warn once that image blocks fell back from the kernel to the recompute.

    The native ops always advertise ``supports_mm_prefix``, so this is a build
    that predates it, which would otherwise serve every image block through the
    slower recompute without a trace.
    """
    from vllm.logger import init_logger  # here, so this module stays numpy-only

    init_logger(__name__).warning(
        "Metal: the compiled ops predate mm_prefix support, so image blocks take "
        "the MLX recompute path; rebuild the native extension to use the tiled "
        "prefill kernel"
    )
