# SPDX-License-Identifier: Apache-2.0
"""Tests for shared Metal utilities."""

import importlib.metadata
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import mlx.core as mx
import pytest

from tools.attention_bench_utils import package_versions
from vllm_metal.utils import get_model_download_path, set_wired_limit


def test_benchmark_versions_allow_missing_distributions(monkeypatch):
    def version(name):
        if name == "optional":
            raise importlib.metadata.PackageNotFoundError(name)
        return "1.0"

    monkeypatch.setattr(importlib.metadata, "version", version)
    assert package_versions("installed", "optional", "another") == {
        "installed": "1.0",
        "optional": None,
        "another": "1.0",
    }


@pytest.mark.parametrize("revision", [None, "release-tag", "a" * 40])
def test_modelscope_download_preserves_revision(monkeypatch, revision):
    monkeypatch.setattr("vllm.envs.VLLM_USE_MODELSCOPE", True)
    monkeypatch.setenv("VLLM_METAL_MODELSCOPE_CACHE", "/model-cache")
    download = Mock(return_value="/model-cache/snapshot")
    monkeypatch.setitem(
        sys.modules,
        "modelscope.hub.snapshot_download",
        SimpleNamespace(snapshot_download=download),
    )

    assert (
        get_model_download_path("org/model", revision=revision)
        == "/model-cache/snapshot"
    )
    kwargs = {"revision": revision} if revision is not None else {}
    download.assert_called_once_with("org/model", cache_dir="/model-cache", **kwargs)


def test_local_model_path_is_preserved(tmp_path):
    assert get_model_download_path(str(tmp_path), revision="release-tag") == str(
        tmp_path
    )


def test_set_wired_limit_uses_pinned_mlx_api(monkeypatch) -> None:
    calls: list[int] = []

    monkeypatch.setattr(
        mx.metal,
        "device_info",
        lambda: {"max_recommended_working_set_size": 123},
    )
    monkeypatch.setattr(mx, "set_wired_limit", calls.append)

    set_wired_limit()

    assert calls == [123]
