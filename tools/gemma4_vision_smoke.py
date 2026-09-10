# SPDX-License-Identifier: Apache-2.0
"""Gemma 4 vision smoke on a tiny synthetic checkpoint through the real engine.

Builds the checkpoint (or uses ``--checkpoint``), starts ``vllm.LLM`` with
the Metal plugin, sends one chat request per image size in the matrix plus a
two-image request, then starts a second engine on the text-only variant and
checks it reports the text-only mode.  Prints ``SMOKE PASS`` on success.

The text-only half runs in a child process (``--text-only-check``): a
second ``vllm.LLM`` in the same process is not supported (duplicate plugin
registration / "already initialized" errors), so the main path shells out
to itself with that flag once the vision half is done.
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw

os.environ.setdefault("VLLM_METAL_USE_PAGED_ATTENTION", "1")
os.environ.setdefault("VLLM_METAL_MEMORY_FRACTION", "0.3")
os.environ.setdefault("VLLM_LOGGING_LEVEL", "INFO")
# Production runs the engine in-process this way; it also keeps EngineCore
# logs in this process so `_ModeCapture` sees the mode-selection log lines.
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

SIZES = [(1, 1), (900, 3), (3, 900), (224, 224), (300, 200), (4096, 4096)]


def _image(size: tuple[int, int], seed: int) -> Image.Image:
    width, height = size
    image = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    draw.rectangle([0, 0, width // 2, height // 2], fill=(200 + seed % 50, 0, 0))
    return image


class _ModeCapture(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())


def _start(checkpoint: Path, capture: _ModeCapture):
    from vllm import LLM

    logging.getLogger("vllm_metal").addHandler(capture)
    return LLM(
        model=str(checkpoint),
        max_model_len=2048,
        max_num_batched_tokens=1024,
        gpu_memory_utilization=0.3,
        limit_mm_per_prompt={"image": 2},
        enable_prefix_caching=True,
    )


def _chat(llm, images: list[Image.Image], text: str) -> str:
    from vllm import SamplingParams

    content: list[dict[str, object]] = [
        {"type": "image_pil", "image_pil": image} for image in images
    ]
    content.append({"type": "text", "text": text})
    outputs = llm.chat(
        [{"role": "user", "content": content}],
        sampling_params=SamplingParams(max_tokens=4, temperature=0),
    )
    return outputs[0].outputs[0].text


def _run_vision_half(checkpoint: Path) -> bool:
    """Start the vision-sidecar engine and run the size matrix. True on success."""
    capture = _ModeCapture()
    llm = _start(checkpoint, capture)
    if not any("vision sidecar" in message for message in capture.messages):
        print("FAIL: engine did not report the vision sidecar mode", file=sys.stderr)
        return False

    for index, size in enumerate(SIZES):
        text = _chat(llm, [_image(size, index)], "Describe the image.")
        print(f"size {size}: {len(text)} chars")
    text = _chat(
        llm,
        [_image((224, 224), 1), _image((300, 200), 2)],
        "Compare the images.",
    )
    print(f"two images: {len(text)} chars")
    text = _chat(llm, [], "Say hello.")
    print(f"text only: {len(text)} chars")
    del llm
    return True


def _run_text_only_check(checkpoint: Path) -> bool:
    """Start the text-only engine. True if it reports the text-only mode."""
    capture = _ModeCapture()
    llm = _start(checkpoint, capture)
    ok = any("forcing text-only backbone" in m for m in capture.messages)
    if not ok:
        print(
            "FAIL: text-only checkpoint did not report the text-only mode",
            file=sys.stderr,
        )
    del llm
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument(
        "--workdir", type=Path, default=Path("/tmp/gemma4-vision-smoke")
    )
    parser.add_argument(
        "--text-only-check",
        action="store_true",
        help="Internal: run only the text-only engine check and exit 0/1.",
    )
    args = parser.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from gemma4_tiny_checkpoint import build_tiny_checkpoint

    if args.text_only_check:
        checkpoint = args.checkpoint or build_tiny_checkpoint(
            args.workdir / "tiny-text", with_vision=False
        )
        return 0 if _run_text_only_check(checkpoint) else 1

    checkpoint = args.checkpoint or build_tiny_checkpoint(args.workdir / "tiny")
    if not _run_vision_half(checkpoint):
        return 1

    # A second `vllm.LLM` in this process hits duplicate plugin registration /
    # "already initialized" errors, so the text-only half runs as a subprocess.
    build_tiny_checkpoint(args.workdir / "tiny-text", with_vision=False)
    result = subprocess.run(
        [
            sys.executable,
            __file__,
            "--text-only-check",
            "--workdir",
            str(args.workdir),
        ],
    )
    if result.returncode != 0:
        return 1

    print("SMOKE PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
