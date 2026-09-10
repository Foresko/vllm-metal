# SPDX-License-Identifier: Apache-2.0
"""Runs the Gemma 4 vision smoke tool (tiny checkpoint, real vLLM engine)."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parent.parent / "tools"
TOOL = TOOLS / "gemma4_vision_smoke.py"

# `tools/` is not a package; the mask-mode helper is imported by module name.
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))


@pytest.mark.slow
@pytest.mark.network
@pytest.mark.skipif(os.environ.get("VLLM_METAL_E2E", "1") != "1", reason="e2e disabled")
def test_gemma4_vision_smoke(tmp_path: Path) -> None:
    result = subprocess.run(
        [sys.executable, str(TOOL), "--workdir", str(tmp_path)],
        capture_output=True,
        text=True,
        timeout=1800,
    )
    assert result.returncode == 0 and "SMOKE PASS" in result.stdout, (
        f"smoke failed (exit {result.returncode}):\n"
        f"{result.stdout[-4000:]}\n{result.stderr[-4000:]}"
    )


class TestMaskModes:
    def test_kl_over_support_is_zero_for_identical_distributions(self) -> None:
        from gemma4_mask_modes import kl_over_support

        p = {1: -0.5, 2: -1.2, 3: -2.0}
        assert kl_over_support(p, dict(p)) == pytest.approx(0.0)
        assert kl_over_support(p, {9: -0.1}) is None

    def test_unknown_mode_is_rejected(self) -> None:
        from gemma4_mask_modes import patch_mlx_vlm_mask_mode

        with pytest.raises(ValueError, match="mask mode"):
            patch_mlx_vlm_mask_mode("sideways")

    def test_modes_restore_the_originals(self) -> None:
        from gemma4_mask_modes import _mask_owner, patch_mlx_vlm_mask_mode
        from mlx_vlm.models.gemma4 import language as lang

        owner = _mask_owner(lang)
        original_make = owner._make_masks
        original_overlay = owner._apply_blockwise_bidirectional_overlay
        patch_mlx_vlm_mask_mode("causal")
        assert owner._apply_blockwise_bidirectional_overlay is not original_overlay
        patch_mlx_vlm_mask_mode("hf")
        assert owner._apply_blockwise_bidirectional_overlay is original_overlay
        assert owner._make_masks is not original_make
        patch_mlx_vlm_mask_mode("asis")
        assert owner._make_masks is original_make
