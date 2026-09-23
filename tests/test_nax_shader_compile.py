# SPDX-License-Identifier: Apache-2.0
"""Compile the NAX prefill shader the way each of its load paths does.

``python -m vllm_metal.metal.build`` compiles the NAX library at its 26.2
deployment floor, which defaults to Metal 4.0. Under
``VLLM_METAL_BUILD_FROM_SOURCE`` MLX compiles it in-process at the newest
version of the running OS instead: Metal 4.1 on macOS 27. A source only one of
them accepts goes unnoticed at runtime, where the loader logs a warning and
keeps the tiled kernel, so both are compiled here. Only the Metal toolchain is
needed, not an M5.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from vllm_metal.metal import _build_nax_source, build

_TARGET_FLAG = "-mmacosx-version-min="


def _sdk_version() -> str:
    return subprocess.run(
        ["xcrun", "-sdk", "macosx", "--show-sdk-version"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def _metal_toolchain_installed() -> bool:
    # Source mode compiles in-process and needs no offline toolchain, which
    # Xcode 26+ ships as a separate download.
    return (
        subprocess.run(
            ["xcrun", "-sdk", "macosx", "metal", "--version"], capture_output=True
        ).returncode
        == 0
    )


@pytest.mark.parametrize("target", ["deployment-floor", "sdk-version"])
def test_nax_source_compiles(tmp_path: Path, target: str) -> None:
    if not build._sdk_supports_nax():
        pytest.skip(f"macOS SDK < {build.NAX_MIN_MACOS_VERSION} cannot build NAX")
    if not _metal_toolchain_installed():
        pytest.skip("Metal toolchain not installed")

    flags = list(build._metallib_flags(build.NAX_METALLIB_NAME))
    if target == "sdk-version":
        # Targeting the SDK's own version selects the newest Metal version it
        # offers, the one MLX compiles at on an OS of that version.
        flags = [f for f in flags if not f.startswith(_TARGET_FLAG)]
        flags.append(f"{_TARGET_FLAG}{_sdk_version()}")

    src = tmp_path / "pagedattention_nax.metal"
    src.write_text(_build_nax_source())
    # Syntax-only still instantiates every kernel, which is where a
    # language-version break surfaces, and skips codegen.
    result = subprocess.run(
        [*flags, "-fsyntax-only", str(src)], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
