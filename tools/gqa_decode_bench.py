# SPDX-License-Identifier: Apache-2.0
"""Microbenchmark: GQA-packed vs per-token paged decode attention.

One layer.  For every (context, batch, path, partition size) it times a
strictly serial chain of paged_attention_primitive calls -- each call's query
depends on the previous call's output, as attention sits between dependent
ops inside a model step, so a low-occupancy kernel cannot hide behind the
next call -- and prints the median and p10 of --reps chains in milliseconds
per call, plus the effective bandwidth over the unique K+V bytes one call
must read.  --cold-mb rotates the calls over disjoint copies of the KV cache,
so at least that many MB of other KV pass through the system-level cache
before a copy is read again (inside a model step a layer's KV is cold); 0
keeps one warm copy.  --idle-url waits for `vllm:num_requests_running 0` on
that worker before every measurement (the GPU may be shared with serving).

Example (Gemma 4 full-attention layer):
  python tools/gqa_decode_bench.py --heads 16 --kv-heads 2 --head-dim 512 \
      --contexts 1024,8192,16384,32768,65536 --batches 1,4,16,64,128 \
      --paths per_token,gqa --partitions auto,0,256,512 --cold-mb 512 \
      --idle-url http://127.0.0.1:8101/metrics --json /tmp/bench.json
"""

from __future__ import annotations

import argparse
import json
import platform
import re
import statistics
import subprocess
import time
import urllib.request
from pathlib import Path

import mlx.core as mx

from vllm_metal.metal import get_ops

BLOCK = 16
DTYPES = {"bf16": mx.bfloat16, "fp16": mx.float16}
PATHS = ("per_token", "gqa")
PARTITIONS = ("auto", "0", "256", "512")


def _busy(url: str) -> int:
    try:
        text = urllib.request.urlopen(url, timeout=3).read().decode()
    except OSError:
        return -1
    match = re.search(r"^vllm:num_requests_running\{[^}]*\}\s+([0-9.]+)", text, re.M)
    return int(float(match.group(1))) if match else -1


def _wait_idle(url: str | None, limit: float = 300.0) -> bool:
    if not url:
        return True
    t0 = time.time()
    while time.time() - t0 < limit:
        if _busy(url) == 0:
            return True
        time.sleep(0.5)
    return False


def _inputs(
    nseq: int, ctx: int, heads: int, kv_heads: int, hd: int, dtype, cold_mb: int
):
    """Query, paged K/V and one block table per disjoint copy of the KV cache."""
    nb = (ctx + BLOCK - 1) // BLOCK
    kv_bytes = nseq * nb * BLOCK * kv_heads * hd * dtype.size * 2
    copies = max(1, -(-cold_mb * 2**20 // kv_bytes)) if cold_mb else 1
    mx.random.seed(0)
    shape = (copies * nseq * nb + 1, BLOCK, kv_heads, hd)
    k = mx.random.normal(shape).astype(dtype)
    v = mx.random.normal(shape).astype(dtype)
    q = mx.random.normal((nseq, heads, hd)).astype(dtype)
    tables = [
        mx.arange(1 + c * nseq * nb, 1 + (c + 1) * nseq * nb, dtype=mx.int32).reshape(
            nseq, nb
        )
        for c in range(copies)
    ]
    lens = mx.full((nseq,), ctx, dtype=mx.int32)
    cu = mx.arange(nseq + 1, dtype=mx.int32)
    mx.eval(k, v, q, lens, cu, *tables)
    return q, k, v, tables, lens, cu


def _set_path(path: str, partition: str) -> None:
    ops = get_ops()
    ops.set_gqa_decode_enabled(path == "gqa")
    ops.set_gqa_decode_force(path == "gqa")
    ops.set_gqa_decode_partition_size(-1 if partition == "auto" else int(partition))


def _time(inputs, *, ctx, kv_heads, hd, calls, reps) -> list[float]:
    """Per-call milliseconds of `reps` serial chains of `calls` calls each.

    Every call's query is q + 0 * (previous output), so the calls run one
    after another; each rep subtracts the same chain without attention (the
    dependency ops alone).  Calls rotate over the KV copies and the rotation
    continues across chains, so a copy is read again only after every other
    copy was.
    """
    q, k, v, tables, lens, cu = inputs
    ops = get_ops()
    rotation = 0

    def chain(with_attention: bool) -> float:
        nonlocal rotation
        dep = mx.zeros_like(q)
        for _ in range(calls):
            qq = q + dep
            out = qq
            if with_attention:
                out = mx.array(0)
                ops.paged_attention_primitive(
                    qq,
                    k,
                    v,
                    kv_heads,
                    hd**-0.5,
                    0.0,
                    tables[rotation % len(tables)],
                    lens,
                    cu,
                    BLOCK,
                    ctx,
                    -1,
                    out,
                )
                rotation += 1
            dep = out * 0
        t0 = time.perf_counter()
        mx.eval(dep)
        return (time.perf_counter() - t0) / calls * 1e3

    chain(True)  # warm-up: pipelines compiled, buffers allocated
    samples = []
    for _ in range(reps):
        base = chain(False)
        samples.append(chain(True) - base)
    return samples


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--heads", type=int, default=16)
    ap.add_argument("--kv-heads", type=int, default=2)
    ap.add_argument("--head-dim", type=int, default=512)
    ap.add_argument("--dtype", choices=sorted(DTYPES), default="bf16")
    ap.add_argument("--contexts", default="1024,8192,16384,32768,65536")
    ap.add_argument("--batches", default="1,4,16,64,128")
    ap.add_argument("--paths", default="per_token,gqa")
    ap.add_argument("--partitions", default="auto")
    ap.add_argument("--cold-mb", type=int, default=512)
    ap.add_argument("--calls", type=int, default=8)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--max-kv-gb", type=float, default=8.0)
    ap.add_argument("--idle-url", default=None)
    ap.add_argument("--variant", default="v1")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    paths = args.paths.split(",")
    partitions = args.partitions.split(",")
    for p in paths:
        if p not in PATHS:
            ap.error(f"--paths: unknown path {p!r} (choose from {', '.join(PATHS)})")
    for p in partitions:
        if p not in PARTITIONS:
            ap.error(
                f"--partitions: unknown size {p!r} (choose from {', '.join(PARTITIONS)})"
            )
    if args.idle_url and _busy(args.idle_url) < 0:
        ap.error(f"--idle-url {args.idle_url}: no vllm:num_requests_running there")

    ops = get_ops()
    dtype = DTYPES[args.dtype]
    commit = (
        subprocess.run(
            [
                "git",
                "-C",
                str(Path(__file__).resolve().parent),
                "rev-parse",
                "--short",
                "HEAD",
            ],
            capture_output=True,
            text=True,
        ).stdout.strip()
        or "unknown"
    )
    header = {
        "gpu": mx.device_info().get("device_name"),
        "gpu_cores": ops.min_decode_grid() // 8,
        "gqa_decode_min_grid": ops.gqa_decode_min_grid(),
        "macos": platform.mac_ver()[0],
        "commit": commit,
        "variant": args.variant,
        "heads": args.heads,
        "kv_heads": args.kv_heads,
        "head_dim": args.head_dim,
        "dtype": args.dtype,
        "cold_mb": args.cold_mb,
        "calls": args.calls,
    }
    print(json.dumps(header))

    rows = []

    def save() -> None:
        if args.json:
            with open(args.json, "w") as f:
                json.dump({"header": header, "rows": rows}, f, indent=2)

    print(
        f"{'path':<9} {'part':>4} {'batch':>5} {'ctx':>6} {'median':>8} {'p10':>8} "
        f"{'GB/s':>6} {'copies':>6} idle"
    )
    try:
        for batch in (int(x) for x in args.batches.split(",")):
            for ctx in (int(x) for x in args.contexts.split(",")):
                kv_bytes = batch * ctx * args.kv_heads * args.head_dim * dtype.size * 2
                if kv_bytes / 1e9 > args.max_kv_gb:
                    continue
                inputs = _inputs(
                    batch,
                    ctx,
                    args.heads,
                    args.kv_heads,
                    args.head_dim,
                    dtype,
                    args.cold_mb,
                )
                copies = len(inputs[3])
                for path in paths:
                    parts = ["auto"] if path == "per_token" else partitions
                    for part in parts:
                        _set_path(path, part)
                        idle = _wait_idle(args.idle_url)
                        s = _time(
                            inputs,
                            ctx=ctx,
                            kv_heads=args.kv_heads,
                            hd=args.head_dim,
                            calls=args.calls,
                            reps=args.reps,
                        )
                        med = statistics.median(s)
                        p10 = sorted(s)[max(0, len(s) // 10)]
                        row = {
                            "path": path,
                            "partition": part,
                            "batch": batch,
                            "ctx": ctx,
                            "median_ms": med,
                            "p10_ms": p10,
                            "gbps": kv_bytes / (med / 1e3) / 1e9 if med > 0 else None,
                            "copies": copies,
                            "idle": idle,
                            "samples_ms": s,
                        }
                        rows.append(row)
                        save()
                        print(
                            f"{path:<9} {part:>4} {batch:>5} {ctx:>6} {med:>8.3f} "
                            f"{p10:>8.3f} {row['gbps'] or 0:>6.0f} {copies:>6} "
                            f"{'y' if idle else 'n'}",
                            flush=True,
                        )
                del inputs
    finally:
        _set_path("gqa", "auto")
        ops.set_gqa_decode_force(False)
        save()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
