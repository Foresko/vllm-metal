# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the ``whisper_hub_tokenizer_cache`` test fixture.

A fake loader stands in for ``WhisperTokenizer.from_pretrained`` so these run
without the hub, and a module-local cache keeps fakes out of the session cache
the real fallback tests share.
"""

from __future__ import annotations

import pytest
from transformers import WhisperTokenizer

HUB = "openai/whisper-tiny"


class _FakeTokenizer:
    def __init__(self, name: str, kwargs: dict) -> None:
        self.name = name
        self.kwargs = kwargs
        self.vocab = {"a": 0}


@pytest.fixture
def _whisper_hub_tokenizers() -> dict:
    # Shadows the session cache from conftest for this module only.
    return {}


@pytest.fixture
def fake_loader(monkeypatch: pytest.MonkeyPatch):
    calls: list[tuple[str, dict]] = []
    failures: list[str] = []

    def fake_from_pretrained(name_or_path, *args, **kwargs):
        calls.append((name_or_path, kwargs))
        if failures and not kwargs:
            raise OSError(failures.pop())
        return _FakeTokenizer(name_or_path, kwargs)

    # Patched before the cache fixture captures the "real" loader.
    monkeypatch.setattr(
        WhisperTokenizer, "from_pretrained", staticmethod(fake_from_pretrained)
    )
    return calls, failures


@pytest.fixture
def cached(fake_loader, request):
    request.getfixturevalue("whisper_hub_tokenizer_cache")
    return fake_loader


def test_repeated_hub_calls_load_once(cached) -> None:
    calls, _ = cached
    WhisperTokenizer.from_pretrained(HUB)
    WhisperTokenizer.from_pretrained(HUB)
    WhisperTokenizer.from_pretrained("openai/whisper-base")
    WhisperTokenizer.from_pretrained("openai/whisper-base")
    assert [name for name, _ in calls] == [HUB, "openai/whisper-base"]


def test_returned_tokenizers_are_independent(cached) -> None:
    first = WhisperTokenizer.from_pretrained(HUB)
    first.vocab["mutated"] = 1
    second = WhisperTokenizer.from_pretrained(HUB)
    assert second is not first
    assert second.vocab == {"a": 0}


@pytest.mark.parametrize(
    "kwargs",
    [{"revision": "main"}, {"local_files_only": True}],
    ids=["revision", "local_files_only"],
)
def test_calls_with_arguments_pass_through(cached, kwargs) -> None:
    calls, _ = cached
    WhisperTokenizer.from_pretrained(HUB)
    WhisperTokenizer.from_pretrained(HUB, **kwargs)
    WhisperTokenizer.from_pretrained(HUB, **kwargs)
    assert calls == [(HUB, {}), (HUB, kwargs), (HUB, kwargs)]


def test_local_paths_pass_through(cached, tmp_path) -> None:
    calls, _ = cached
    WhisperTokenizer.from_pretrained(str(tmp_path))
    WhisperTokenizer.from_pretrained(str(tmp_path))
    assert calls == [(str(tmp_path), {}), (str(tmp_path), {})]


def test_oserror_is_not_cached_and_retry_reaches_loader(cached) -> None:
    calls, failures = cached
    failures.append("hub unreachable")
    with pytest.raises(OSError, match="hub unreachable"):
        WhisperTokenizer.from_pretrained(HUB)
    # The transcriber's offline retry goes to the real loader, not the cache.
    retry = WhisperTokenizer.from_pretrained(HUB, local_files_only=True)
    assert retry.kwargs == {"local_files_only": True}
    # The failure left no entry behind: the next plain call loads again.
    WhisperTokenizer.from_pretrained(HUB)
    assert calls == [(HUB, {}), (HUB, {"local_files_only": True}), (HUB, {})]
