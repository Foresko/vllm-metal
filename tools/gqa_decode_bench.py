# SPDX-License-Identifier: Apache-2.0
"""Microbenchmark: GQA-packed vs per-token paged decode attention.

One layer.  For every (context, batch, path, partition size) it times the
paged_attention_primitive call and prints the median and p10 of --reps runs
in milliseconds and the effective bandwidth over the unique K+V bytes the
call must read.  --flush-mb touches a buffer before every call so KV is not
served from the system-level cache, as inside a model step.  --idle-url
waits for `vllm:num_requests_running 0` on that worker before every
measurement (the GPU may be shared with serving).

Example (Gemma 4 full-attention layer):
  python tools/gqa_decode_bench.py --heads 16 --kv-heads 2 --head-dim 512 \
      --contexts 1024,8192,16384,32768,65536 --batches 1,4,16,64,128 \
      --paths per_token,gqa --partitions auto,0,256,512 --flush-mb 256 \
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

import mlx.core as mx

from vllm_metal.metal import get_ops

BLOCK = 16
DTYPES = {"bf16": mx.bfloat16, "fp16": mx.float16}


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


def _inputs(nseq: int, ctx: int, heads: int, kv_heads: int, hd: int, dtype):
    nb = (ctx + BLOCK - 1) // BLOCK
    mx.random.seed(0)
    k = mx.random.normal((nb * nseq + 1, BLOCK, kv_heads, hd)).astype(dtype)
    v = mx.random.normal((nb * nseq + 1, BLOCK, kv_heads, hd)).astype(dtype)
    q = mx.random.normal((nseq, heads, hd)).astype(dtype)
    table = mx.arange(1, nb * nseq + 1, dtype=mx.int32).reshape(nseq, nb)
    lens = mx.full((nseq,), ctx, dtype=mx.int32)
    cu = mx.arange(nseq + 1, dtype=mx.int32)
    mx.eval(k, v, q, table, lens, cu)
    return q, k, v, table, lens, cu


def _set_path(path: str, partition: str) -> None:
    ops = get_ops()
    ops.set_gqa_decode_enabled(path == "gqa")
    ops.set_gqa_decode_force(path == "gqa")
    ops.set_gqa_decode_partition_size(-1 if partition == "auto" else int(partition))


def _time(inputs, *, ctx, kv_heads, hd, flush, calls, reps) -> list[float]:
    q, k, v, table, lens, cu = inputs
    ops = get_ops()

    def graph(with_attention: bool) -> None:
        outs = []
        for _ in range(calls):
            qq = q
            if flush is not None:
                f = flush.sum()
                outs.append(f)
                qq = q + (f * 0).astype(q.dtype)
            if with_attention:
                out = mx.array(0)
                ops.paged_attention_primitive(
                    qq,
                    k,
                    v,
                    kv_heads,
                    hd**-0.5,
                    0.0,
                    table,
                    lens,
                    cu,
                    BLOCK,
                    ctx,
                    -1,
                    out,
                )
                outs.append(out)
        mx.eval(*outs)

    def once(with_attention: bool) -> float:
        t0 = time.perf_counter()
        graph(with_attention)
        return (time.perf_counter() - t0) / calls * 1e3

    graph(True)
    samples = []
    for _ in range(reps):
        base = once(False) if flush is not None else 0.0
        samples.append(once(True) - base)
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
    ap.add_argument("--flush-mb", type=int, default=256)
    ap.add_argument("--calls", type=int, default=4)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--max-kv-gb", type=float, default=8.0)
    ap.add_argument("--idle-url", default=None)
    ap.add_argument("--variant", default="v1")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    ops = get_ops()
    dtype = DTYPES[args.dtype]
    commit = (
        subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True
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
        "flush_mb": args.flush_mb,
    }
    print(json.dumps(header))
    flush = None
    if args.flush_mb:
        flush = mx.random.normal((args.flush_mb * 1024 * 1024 // 2,)).astype(
            mx.bfloat16
        )
        mx.eval(flush)

    rows = []
    print(
        f"{'path':<9} {'part':>4} {'batch':>5} {'ctx':>6} {'median':>8} {'p10':>8} {'GB/s':>6} idle"
    )
    for batch in (int(x) for x in args.batches.split(",")):
        for ctx in (int(x) for x in args.contexts.split(",")):
            kv_bytes = batch * ctx * args.kv_heads * args.head_dim * 2 * 2
            if kv_bytes / 1e9 > args.max_kv_gb:
                continue
            inputs = _inputs(
                batch, ctx, args.heads, args.kv_heads, args.head_dim, dtype
            )
            for path in args.paths.split(","):
                parts = ["auto"] if path == "per_token" else args.partitions.split(",")
                for part in parts:
                    _set_path(path, part)
                    idle = _wait_idle(args.idle_url)
                    s = _time(
                        inputs,
                        ctx=ctx,
                        kv_heads=args.kv_heads,
                        hd=args.head_dim,
                        flush=flush,
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
                        "idle": idle,
                        "samples_ms": s,
                    }
                    rows.append(row)
                    print(
                        f"{path:<9} {part:>4} {batch:>5} {ctx:>6} {med:>8.3f} {p10:>8.3f} "
                        f"{row['gbps'] or 0:>6.0f} {'y' if idle else 'n'}",
                        flush=True,
                    )
            del inputs
    _set_path("gqa", "auto")
    ops.set_gqa_decode_force(False)
    if args.json:
        with open(args.json, "w") as f:
            json.dump({"header": header, "rows": rows}, f, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
