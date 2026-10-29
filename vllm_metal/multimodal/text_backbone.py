# SPDX-License-Identifier: Apache-2.0
"""Shared text-backbone resolution for the multimodal adapters.

mlx-lm/mlx-vlm language wrappers keep the headless transformer at
``language_model.model`` (``text_model.language_model.model`` for the
Gemma 4 sidecar): it owns ``embed_tokens`` and produces the hidden
states the output head projects.  Resolving it while the adapter is
constructed turns an upstream rename or restructure into a load-time
error instead of an attribute error mid-forward.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


def resolve_text_backbone(language_model: Any, *, owner: str) -> Any:
    """Return ``language_model.model`` or raise a drift error."""
    backbone = getattr(language_model, "model", None)
    if backbone is None:
        raise RuntimeError(
            "language_model.model attribute missing; "
            f"{owner} version drift detected.  Expected the bottom-level "
            "LM module that exposes embed_tokens."
        )
    return backbone


def resolve_backbone_embed_tokens(backbone: Any, *, owner: str) -> Callable[[Any], Any]:
    """Return ``backbone.embed_tokens`` or raise a drift error."""
    embed_tokens = getattr(backbone, "embed_tokens", None)
    if embed_tokens is None or not callable(embed_tokens):
        raise RuntimeError(
            "language_model.model.embed_tokens missing or not callable; "
            f"{owner} version drift detected."
        )
    return embed_tokens
