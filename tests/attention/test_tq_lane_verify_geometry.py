# SPDX-License-Identifier: Apache-2.0
"""Validate requested crossover matrices without executing attention kernels."""

import argparse
from itertools import product

import mlx.core as mx
import pytest

from tools.benchmark import tq_lane_verify as bench


def test_default_hd128_matrix_is_preserved():
    rows = list(bench.cases("crossover-hd128"))
    expected = list(
        product(
            [(8, 2), (8, 8)],
            [mx.float16, mx.bfloat16],
            [8192, 32768],
            [16, 32, 64, 96, 128, 192, 256, 512],
        )
    )
    assert len(rows) == len(expected)
    for (label, config), ((nq, nkv), dtype, context, query) in zip(
        rows, expected, strict=True
    ):
        assert label == f"d128-q{nq}-kv{nkv}-{dtype}-q{query}-ctx{context}"
        assert config == {
            "head_dim": 128,
            "n_heads": nq,
            "n_kv_heads": nkv,
            "dtype": dtype,
            "qlens": (query,),
            "context_lens": (context,),
        }


def test_model_geometries_and_threshold_boundaries_are_included():
    rows = list(
        bench.cases(
            "crossover-hd128",
            head_pairs=[(32, 8), (16, 2)],
            query_tokens=[127, 128, 129],
        )
    )
    assert len(rows) == 2 * 2 * 2 * 3
    assert len({label for label, _ in rows}) == len(rows)
    assert {
        (config["n_heads"], config["n_kv_heads"], config["qlens"][0])
        for _, config in rows
    } == set(product([32], [8], [127, 128, 129])) | set(
        product([16], [2], [127, 128, 129])
    )


def test_fresh_and_short_context_matrix():
    rows = list(
        bench.cases(
            "crossover-hd128",
            head_pairs=[(8, 2), (8, 8), (32, 8), (16, 2)],
            query_tokens=[64, 96, 128, 192],
            context_tokens=[0, 256, 2048],
        )
    )
    assert len(rows) == 4 * 2 * 4 * 3
    assert len({label for label, _ in rows}) == len(rows)
    for query in [64, 96, 128, 192]:
        assert {
            config["context_lens"][0]
            for _, config in rows
            if config["qlens"] == (query,)
        } == {query, 256, 2048}


@pytest.mark.parametrize("value", ["8:2", "8:8", "32:8", "16:2"])
def test_valid_head_pair(value):
    assert bench.head_pair(value) == tuple(map(int, value.split(":")))


@pytest.mark.parametrize(
    "value", ["8", "8:2:1", "x:2", "0:2", "8:0", "-8:2", "8:-2", "2:8", "8:3"]
)
def test_invalid_head_pair(value):
    with pytest.raises(argparse.ArgumentTypeError):
        bench.head_pair(value)


@pytest.mark.parametrize(
    "arguments,message",
    [
        (["--head-pairs", "32:8"], "require --suite crossover-hd128"),
        (["--query-tokens", "128"], "require --suite crossover-hd128"),
        (["--context-tokens", "0"], "require --suite crossover-hd128"),
        (
            ["--suite", "crossover-hd128", "--query-tokens", "1"],
            "must be at least 2",
        ),
        (
            ["--suite", "crossover-hd128", "--query-tokens", "8193"],
            "at least the largest query length",
        ),
        (
            ["--suite", "crossover-hd128", "--context-tokens", "-1"],
            "must be nonnegative",
        ),
        (
            ["--suite", "crossover-hd128", "--context-tokens", "0", "256"],
            "at least the largest query length",
        ),
    ],
)
def test_invalid_options_fail_before_loading_metal(
    monkeypatch, capsys, arguments, message
):
    monkeypatch.setattr("sys.argv", ["tq_lane_verify.py", *arguments])

    def unexpected_metal_load():
        pytest.fail("invalid arguments must fail before loading Metal")

    monkeypatch.setattr(bench, "get_ops", unexpected_metal_load)
    with pytest.raises(SystemExit) as error:
        bench.main()
    assert error.value.code == 2
    assert message in capsys.readouterr().err


@pytest.mark.parametrize("suite", ["crossover", "geometry", "long"])
def test_matrix_overrides_do_not_affect_other_suites(suite):
    with pytest.raises(ValueError, match="require crossover-hd128"):
        list(bench.cases(suite, head_pairs=[(32, 8)]))
    with pytest.raises(ValueError, match="require crossover-hd128"):
        list(bench.cases(suite, query_tokens=[127, 128, 129]))
    with pytest.raises(ValueError, match="require crossover-hd128"):
        list(bench.cases(suite, context_tokens=[0, 256]))
