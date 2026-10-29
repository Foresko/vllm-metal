#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Live serve smoke for the local GGUF path (#415 stack).

Starts ``vllm serve <model.gguf> --tokenizer <config-dir>``, waits for /health,
verifies its model in /v1/models, then greedy-decodes through /v1/completions
and checks the continuation. This proves the user-facing GGUF serve workflow
end to end through the Metal paged runner (the path that surfaced the head_dim
bug); kept as a maintainer script rather than a skipped slow unit test.

Run one vLLM process at a time; set the memory fraction for this machine:

    python tools/gguf_serve_smoke.py /path/Qwen3-0.6B-Q8_0.gguf \\
        --tokenizer /path/Qwen3-0.6B --gpu-memory-utilization 0.5
"""

from __future__ import annotations

import argparse
import json
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid

_HEALTH_TIMEOUT_S = 300
_DEFAULT_PROMPT = "The capital of France is"
_DEFAULT_EXPECT = "Paris"


def _probe_port(requested_port: int) -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", requested_port))
        return sock.getsockname()[1]


def _wait_for_health(
    base_url: str, timeout_s: float, serve: subprocess.Popen, served_model: str
) -> bool:
    deadline = time.monotonic() + timeout_s
    while (remaining := deadline - time.monotonic()) > 0:
        if serve.poll() is not None:
            return False
        try:
            with urllib.request.urlopen(
                f"{base_url}/health", timeout=min(5, remaining)
            ) as resp:
                healthy = resp.status == 200
            if healthy:
                if serve.poll() is not None:
                    return False
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                with urllib.request.urlopen(
                    f"{base_url}/v1/models", timeout=min(5, remaining)
                ) as models_resp:
                    models = json.load(models_resp)
                model_data = models.get("data") if isinstance(models, dict) else None
                if isinstance(model_data, list) and any(
                    model.get("id") == served_model
                    for model in model_data
                    if isinstance(model, dict)
                ):
                    return serve.poll() is None
        except (urllib.error.URLError, ConnectionError, OSError, ValueError):
            pass
        if serve.poll() is not None:
            return False
        time.sleep(min(2, max(0, deadline - time.monotonic())))
    return False


def _greedy_completion(base_url: str, model: str, prompt: str, max_tokens: int) -> str:
    payload = json.dumps(
        {
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0,
        }
    ).encode()
    req = urllib.request.Request(
        f"{base_url}/v1/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        body = json.load(resp)
    return body["choices"][0]["text"]


def _stop_server(serve: subprocess.Popen) -> None:
    if serve.poll() is None:
        serve.terminate()
    try:
        serve.wait(timeout=30)
    except subprocess.TimeoutExpired:
        serve.kill()
        try:
            serve.wait(timeout=5)
        except subprocess.TimeoutExpired:
            print("WARN: server did not exit after kill", file=sys.stderr)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", help="path to a local .gguf file")
    parser.add_argument(
        "--tokenizer", required=True, help="companion config/tokenizer directory"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=0,
        help="Server port (default: choose an available port)",
    )
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.5)
    parser.add_argument("--max-tokens", type=int, default=24)
    parser.add_argument("--prompt", default=_DEFAULT_PROMPT)
    parser.add_argument("--expect", default=_DEFAULT_EXPECT)
    args = parser.parse_args()

    try:
        port = _probe_port(args.port)
    except OSError as exc:
        print(f"FAIL: cannot bind localhost port {args.port}: {exc}", file=sys.stderr)
        return 1
    base_url = f"http://127.0.0.1:{port}"
    served_model = f"gguf-smoke-{uuid.uuid4().hex}"
    serve = subprocess.Popen(
        [
            "vllm",
            "serve",
            args.model,
            "--tokenizer",
            args.tokenizer,
            "--served-model-name",
            served_model,
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--max-model-len",
            str(args.max_model_len),
            "--gpu-memory-utilization",
            str(args.gpu_memory_utilization),
        ]
    )
    try:
        print(f"Waiting for {base_url}/health ...", flush=True)
        if not _wait_for_health(base_url, _HEALTH_TIMEOUT_S, serve, served_model):
            exit_code = serve.poll()
            message = (
                f"server exited during startup (exit code {exit_code})"
                if exit_code is not None
                else "server did not become healthy"
            )
            print(f"FAIL: {message}", file=sys.stderr)
            return 1
        text = _greedy_completion(base_url, served_model, args.prompt, args.max_tokens)
        if (exit_code := serve.poll()) is not None:
            print(
                f"FAIL: server exited during completion (exit code {exit_code})",
                file=sys.stderr,
            )
            return 1
        ok = args.expect in text
        print(f"prompt:     {args.prompt!r}")
        print(f"completion: {text!r}")
        print(
            f"[{'PASS' if ok else 'FAIL'}] expected {args.expect!r} in the continuation"
        )
        return 0 if ok else 1
    finally:
        _stop_server(serve)


if __name__ == "__main__":
    sys.exit(main())
