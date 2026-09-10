# SPDX-License-Identifier: Apache-2.0
"""Gemma 4 vision sidecar: ``vision_tower`` + ``embed_vision`` next to mlx_lm.

The text backbone stays the mlx_lm Gemma 4 model that vllm-metal already
serves; only the vision tower and the multimodal embedder are taken from the
mlx-vlm composite, loaded lazily so the language-model weights are never
materialised twice.
"""

from __future__ import annotations

import json
import struct
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import mlx.core as mx
from mlx.utils import tree_flatten
from vllm.logger import init_logger

logger = init_logger(__name__)

_VISION_PREFIX = "vision_tower."
_INDEX_FILE = "model.safetensors.index.json"


def _strip_hf_prefix(key: str) -> str:
    """HF checkpoints prefix every weight with ``model.``; MLX ones do not."""
    return key.removeprefix("model.")


def _safetensors_keys(path: Path) -> list[str]:
    with path.open("rb") as fh:
        header_len = struct.unpack("<Q", fh.read(8))[0]
        header = json.loads(fh.read(header_len))
    return [key for key in header if key != "__metadata__"]


def has_vision_weights(model_path: Path) -> bool:
    """Whether a local checkpoint carries ``vision_tower.*`` tensors.

    Reads ``model.safetensors.index.json`` when present (the OptiQ sidecar
    shard in ``optiq/`` is indexed there), otherwise the headers of the
    ``*.safetensors`` files in the checkpoint root.
    """
    index = model_path / _INDEX_FILE
    if index.is_file():
        weight_map = json.loads(index.read_text()).get("weight_map", {})
        return any(
            _strip_hf_prefix(key).startswith(_VISION_PREFIX) for key in weight_map
        )
    for shard in sorted(model_path.glob("*.safetensors")):
        if any(
            _strip_hf_prefix(key).startswith(_VISION_PREFIX)
            for key in _safetensors_keys(shard)
        ):
            return True
    return False


def _load_composite(model_path: Path) -> Any:
    """Build the mlx-vlm Gemma 4 composite with lazy (unmaterialised) weights."""
    from mlx_vlm.utils import load_model

    return load_model(model_path, lazy=True)


@dataclass(frozen=True)
class Gemma4VisionSidecar:
    """The vision tower and multimodal embedder of a Gemma 4 checkpoint."""

    vision_tower: Any
    embed_vision: Any
    pixel_dtype: mx.Dtype
    num_parameters: int
    num_bytes: int

    @classmethod
    def load(
        cls,
        model_path: Path,
        *,
        load_composite: Callable[[Path], Any] = _load_composite,
    ) -> Gemma4VisionSidecar:
        """Load only the vision modules of the checkpoint at ``model_path``.

        ``model_path`` must be the original checkpoint directory (not the
        mlx_lm shard-compatibility view).  Any failure raises: by the time
        this runs the frontend already accepts images, so a silent fallback
        would leave the engine to die on the first image request.
        """
        composite = load_composite(model_path)
        vision_tower = getattr(composite, "vision_tower", None)
        embed_vision = getattr(composite, "embed_vision", None)
        if vision_tower is None or embed_vision is None:
            raise RuntimeError(
                "mlx_vlm Gemma 4 composite exposes no vision_tower/embed_vision; "
                "mlx-vlm version drift detected."
            )
        # `tree_flatten` is overloaded `list[tuple[str, Any]] | dict[str, Any]`
        # depending on the `destination` kwarg; with `destination=None`
        # (default) it returns the list. Narrow at runtime so mypy can add
        # the two lists together below.
        params = cast(
            "list[tuple[str, Any]]", tree_flatten(vision_tower.parameters())
        ) + cast("list[tuple[str, Any]]", tree_flatten(embed_vision.parameters()))
        if not params:
            raise RuntimeError(f"no vision parameters were loaded from {model_path}")

        input_proj = getattr(
            getattr(vision_tower, "patch_embedder", None), "input_proj", None
        )
        weight = getattr(input_proj, "weight", None)
        if weight is None or not mx.issubdtype(weight.dtype, mx.floating):
            raise RuntimeError(
                "Gemma 4 vision sidecar needs a floating-point patch projection; "
                "mlx-vlm casts pixels to the packed weight dtype of a quantized "
                "projection, so quantized vision towers are unsupported."
            )

        mx.eval(*(value for _, value in params))
        del composite
        mx.clear_cache()

        num_parameters = sum(int(value.size) for _, value in params)
        num_bytes = sum(int(value.nbytes) for _, value in params)
        logger.info(
            "Gemma 4 vision sidecar loaded: %d parameters, %.1f MiB, pixel dtype %s",
            num_parameters,
            num_bytes / (1024 * 1024),
            weight.dtype,
        )
        return cls(
            vision_tower=vision_tower,
            embed_vision=embed_vision,
            pixel_dtype=weight.dtype,
            num_parameters=num_parameters,
            num_bytes=num_bytes,
        )
