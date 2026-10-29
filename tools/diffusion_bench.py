#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Measure DiffusionGemma serving latency and throughput against a running server.

vLLM's periodic "Avg prompt/generation throughput" log lines average over a
10 s window, so a single short request looks slow there. This script times
whole requests instead, after a warmup that absorbs MLX kernel compilation and
lazy weight loading.

Usage:
    PYTHONPATH=$PWD vllm serve mlx-community/diffusiongemma-26B-A4B-it-4bit \
        --diffusion-config '{"canvas_length": 32}'

    python tools/diffusion_bench.py
    python tools/diffusion_bench.py --concurrency 1 4 --repeats 3
    python tools/diffusion_bench.py --json results.json

The "Mean denoising steps per canvas" line in the server log, printed by the
DiffusionDecoding metrics, gives the per-canvas cost behind these numbers.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

PROMPTS = {
    "short": ("Reply with only YES or NO: Is Paris the capital of France?", 20),
    "long": ("Explain in three short paragraphs why the sky is blue.", 300),
}


@dataclass
class Sample:
    latency_s: float
    completion_tokens: int
    finish_reason: str


def chat(url: str, model: str, prompt: str, max_tokens: int) -> Sample:
    body = json.dumps(
        {
            "model": model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
    ).encode()
    request = urllib.request.Request(
        f"{url}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    start = time.perf_counter()
    with urllib.request.urlopen(request, timeout=600) as response:
        payload = json.load(response)
    latency = time.perf_counter() - start
    return Sample(
        latency_s=latency,
        completion_tokens=payload["usage"]["completion_tokens"],
        finish_reason=payload["choices"][0]["finish_reason"],
    )


def run_batch(
    url: str, model: str, prompt: str, max_tokens: int, concurrency: int
) -> tuple[list[Sample], float]:
    """Send `concurrency` identical requests at once; return samples and wall time."""
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [
            pool.submit(chat, url, model, prompt, max_tokens)
            for _ in range(concurrency)
        ]
        samples = [f.result() for f in futures]
    return samples, time.perf_counter() - start


def served_model(url: str) -> str:
    with urllib.request.urlopen(f"{url}/v1/models", timeout=30) as response:
        return json.load(response)["data"][0]["id"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default=None, help="default: first served model")
    parser.add_argument(
        "--prompts",
        nargs="+",
        choices=sorted(PROMPTS),
        default=sorted(PROMPTS),
    )
    parser.add_argument("--concurrency", nargs="+", type=int, default=[1])
    parser.add_argument("--repeats", type=int, default=3, help="timed runs per cell")
    parser.add_argument("--warmup", type=int, default=1, help="untimed runs per prompt")
    parser.add_argument("--json", default=None, help="write the results to this file")
    args = parser.parse_args()

    url = args.url.rstrip("/")
    model = args.model or served_model(url)
    print(f"server {url}  model {model}")

    results = []
    for name in args.prompts:
        prompt, max_tokens = PROMPTS[name]
        for _ in range(args.warmup):
            warm, _ = run_batch(url, model, prompt, max_tokens, 1)
            print(f"[{name}] warmup: {warm[0].latency_s:.2f}s (not counted)")

        for concurrency in args.concurrency:
            latencies: list[float] = []
            walls: list[float] = []
            tokens: list[int] = []
            finishes: set[str] = set()
            for _ in range(args.repeats):
                samples, wall = run_batch(url, model, prompt, max_tokens, concurrency)
                walls.append(wall)
                latencies += [s.latency_s for s in samples]
                tokens.append(sum(s.completion_tokens for s in samples))
                finishes |= {s.finish_reason for s in samples}
            row = {
                "prompt": name,
                "concurrency": concurrency,
                "repeats": args.repeats,
                "completion_tokens_per_run": statistics.mean(tokens),
                "latency_median_s": statistics.median(latencies),
                "latency_max_s": max(latencies),
                "aggregate_tokens_per_s": sum(tokens) / sum(walls),
                "finish_reasons": sorted(finishes),
            }
            results.append(row)
            print(
                f"[{name}] concurrency={concurrency}  "
                f"tokens/run={row['completion_tokens_per_run']:.0f}  "
                f"latency median={row['latency_median_s']:.2f}s "
                f"max={row['latency_max_s']:.2f}s  "
                f"aggregate={row['aggregate_tokens_per_s']:.2f} tok/s  "
                f"finish={row['finish_reasons']}"
            )

    if args.json:
        with open(args.json, "w") as f:
            json.dump({"model": model, "results": results}, f, indent=2)
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
