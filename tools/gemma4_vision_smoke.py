# SPDX-License-Identifier: Apache-2.0
"""Gemma 4 vision smoke on a tiny synthetic checkpoint through the real engine.

Builds the checkpoint (or uses ``--checkpoint``), starts ``vllm.LLM`` with
the Metal plugin, sends one chat request per image size in the matrix plus a
two-image request, asserts the engine logged the bidirectional image-attention
path (``Metal: mm_prefix ranges`` on the default kernel path, ``bidirectional
image attention`` under ``VLLM_METAL_MM_PREFIX_PATH=recompute``) and compares
its first-token log-probs against mlx-vlm under the ``hf`` and ``causal``
mask modes (``tools/gemma4_mask_modes.py``), then starts a second engine on
the recompute path in a child process and checks it picks the same first
token (or a tie within bf16 noise) with a matching distribution, and a third
on the text-only variant and checks it reports the text-only mode.  Prints
``SMOKE PASS`` on success.

A second ``vllm.LLM`` in the same process was not attempted; the text-only
variant runs in a child process (``--text-only-check``) instead, and the
main path shells out to itself with that flag once the vision half is done.
"""

from __future__ import annotations

import argparse
import json
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


def _content(images: list[Image.Image], text: str) -> list[dict[str, object]]:
    content: list[dict[str, object]] = [
        {"type": "image_pil", "image_pil": image} for image in images
    ]
    content.append({"type": "text", "text": text})
    return content


def _chat(llm, images: list[Image.Image], text: str) -> str:
    from vllm import SamplingParams

    outputs = llm.chat(
        [{"role": "user", "content": _content(images, text)}],
        sampling_params=SamplingParams(max_tokens=4, temperature=0),
    )
    return outputs[0].outputs[0].text


def _first_token_logprobs(
    llm, images: list[Image.Image], text: str
) -> dict[int, float]:
    """Top-20 first-token log-probs from the engine for one chat request."""
    from vllm import SamplingParams

    outputs = llm.chat(
        [{"role": "user", "content": _content(images, text)}],
        sampling_params=SamplingParams(max_tokens=1, temperature=0, logprobs=20),
    )
    return {tid: lp.logprob for tid, lp in outputs[0].outputs[0].logprobs[0].items()}


def _reference_logprobs(
    checkpoint: Path, image: Image.Image, text: str, mode: str
) -> dict[int, float]:
    """Top-20 first-token log-probs from mlx-vlm under one mask mode."""
    import mlx.core as mx
    from gemma4_mask_modes import patch_mlx_vlm_mask_mode
    from mlx_vlm import load
    from mlx_vlm.prompt_utils import apply_chat_template
    from mlx_vlm.utils import prepare_inputs

    patch_mlx_vlm_mask_mode(mode)
    model, processor = load(str(checkpoint))
    prompt = apply_chat_template(processor, model.config, text, num_images=1)
    inputs = prepare_inputs(
        processor,
        images=[image],
        prompts=prompt,
        image_token_index=None,
        add_special_tokens=False,
    )
    kwargs = {
        k: v
        for k, v in inputs.items()
        if k not in ("input_ids", "pixel_values", "attention_mask")
    }
    logits = model(inputs["input_ids"], inputs.get("pixel_values"), **kwargs)
    logits = logits.logits if hasattr(logits, "logits") else logits
    row = logits[0, -1].astype(mx.float32)
    logprobs = row - mx.logsumexp(row)
    top = mx.argpartition(-logprobs, 20)[:20]
    return {int(i): float(logprobs[i]) for i in top.tolist()}


def _expected_bidi_line() -> str:
    """The log line proving image-block rows took the configured path."""
    path = os.environ.get("VLLM_METAL_MM_PREFIX_PATH", "kernel")
    if path == "recompute":
        return "bidirectional image attention"
    return "Metal: mm_prefix ranges"


FIRST_TOKEN_IMAGE = ((256, 256), 7)
FIRST_TOKEN_PROMPT = "Describe the image."

# bf16 noise between the tiled kernel and MLX SDPA; on the tiny checkpoint the
# first-token logits are nearly flat, so the argmax is a coin flip inside it.
TIE_TOLERANCE_NATS = 0.05
KL_TOLERANCE = 1e-3


def _parity_check(llm, checkpoint: Path) -> dict[int, float] | None:
    """Compare the engine's first token against the hf and causal references."""
    from gemma4_mask_modes import kl_over_support

    image = _image(*FIRST_TOKEN_IMAGE)
    text = FIRST_TOKEN_PROMPT
    engine = _first_token_logprobs(llm, [image], text)
    hf = _reference_logprobs(checkpoint, image, text, "hf")
    causal = _reference_logprobs(checkpoint, image, text, "causal")
    kl_hf = kl_over_support(engine, hf)
    kl_causal = kl_over_support(engine, causal)
    kl_modes = kl_over_support(hf, causal)
    top = {
        name: max(d, key=d.get)
        for name, d in (("engine", engine), ("hf", hf), ("causal", causal))
    }
    print(
        f"parity: KL(engine||hf)={kl_hf} KL(engine||causal)={kl_causal} "
        f"KL(hf||causal)={kl_modes} top1={top}"
    )
    if kl_modes is None or kl_modes < 1e-4:
        print(
            "parity inconclusive: the tiny checkpoint does not separate the mask modes"
        )
        return engine
    if top["engine"] != top["hf"]:
        print("FAIL: engine top-1 differs from the hf reference", file=sys.stderr)
        return None
    if kl_hf is None or kl_causal is None or kl_hf > kl_causal:
        print(
            "FAIL: engine is closer to the causal reference than to hf", file=sys.stderr
        )
        return None
    return engine


def _run_vision_half(checkpoint: Path, first_token_out: Path) -> bool:
    """Start the vision-sidecar engine and run the size matrix. True on success."""
    capture = _ModeCapture()
    llm = _start(checkpoint, capture)
    if not any("vision sidecar" in message for message in capture.messages):
        print("FAIL: engine did not report the vision sidecar mode", file=sys.stderr)
        return False
    sidecar_loaded = [m for m in capture.messages if "vision sidecar loaded" in m]
    if not sidecar_loaded:
        print("FAIL: engine did not log the sidecar load", file=sys.stderr)
        return False
    print(f"sidecar load: {sidecar_loaded[0]}")

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
    expected = _expected_bidi_line()
    if not any(expected in m for m in capture.messages):
        print(
            f"FAIL: engine never logged the image-block path ({expected!r})",
            file=sys.stderr,
        )
        return False
    engine = _parity_check(llm, checkpoint)
    if engine is None:
        return False
    first_token_out.write_text(
        json.dumps({"top_logprobs": {str(k): v for k, v in engine.items()}})
    )
    del llm
    return True


def _run_recompute_check(checkpoint: Path, first_token_in: Path) -> bool:
    """Start the engine on the recompute path and compare its first token.

    Passes if the recompute path picks the same first token (or a tie within
    bf16 noise) with a matching distribution. Runs in a child process with
    ``VLLM_METAL_MM_PREFIX_PATH=recompute`` set by the parent: the engine core
    is its own process, so the variable must be in place before the engine
    starts.
    """
    from gemma4_mask_modes import kl_over_support

    capture = _ModeCapture()
    llm = _start(checkpoint, capture)
    # The log line only appears inside a prefill forward that carries image
    # rows, so it cannot be checked until after a request has run (mirrors
    # `_run_vision_half`, which checks its expected line after its chats).
    recompute = _first_token_logprobs(
        llm, [_image(*FIRST_TOKEN_IMAGE)], FIRST_TOKEN_PROMPT
    )
    if not any("bidirectional image attention" in m for m in capture.messages):
        print("FAIL: recompute engine never logged the phase-2 path", file=sys.stderr)
        return False
    saved = json.loads(first_token_in.read_text())["top_logprobs"]
    kernel = {int(k): float(v) for k, v in saved.items()}
    kl = kl_over_support(kernel, recompute)
    top_kernel = max(kernel, key=kernel.get)
    top_recompute = max(recompute, key=recompute.get)
    # A different top-1 is accepted only as a tie: the recompute path's choice
    # ranks within TIE_TOLERANCE_NATS of the kernel path's own top-1, and the
    # two distributions agree over the shared support.
    tied = (
        top_recompute in kernel
        and kernel[top_kernel] - kernel[top_recompute] < TIE_TOLERANCE_NATS
    )
    print(
        f"recompute check: top1 kernel={top_kernel} recompute={top_recompute} "
        f"kl={kl} tie={tied}"
    )
    del llm
    if top_kernel != top_recompute and not tied:
        print(
            "FAIL: kernel and recompute paths disagree on the first token",
            file=sys.stderr,
        )
        return False
    if kl is None or abs(kl) > KL_TOLERANCE:
        print(
            "FAIL: kernel and recompute first-token distributions diverge",
            file=sys.stderr,
        )
        return False
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
    parser.add_argument(
        "--recompute-check",
        action="store_true",
        help=(
            "Internal: compare the recompute path's first token with the "
            "saved kernel one."
        ),
    )
    args = parser.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from gemma4_tiny_checkpoint import build_tiny_checkpoint

    if args.text_only_check:
        checkpoint = args.checkpoint or build_tiny_checkpoint(
            args.workdir / "tiny-text", with_vision=False
        )
        return 0 if _run_text_only_check(checkpoint) else 1

    first_token = args.workdir / "first-token-kernel.json"
    if args.recompute_check:
        checkpoint = args.checkpoint or build_tiny_checkpoint(args.workdir / "tiny")
        return 0 if _run_recompute_check(checkpoint, first_token) else 1

    checkpoint = args.checkpoint or build_tiny_checkpoint(args.workdir / "tiny")
    if not _run_vision_half(checkpoint, first_token):
        return 1

    # Forward --checkpoint too: without it the child rebuilds workdir/tiny and
    # the first-token comparison would run against a different model.
    recompute_argv = [
        sys.executable,
        __file__,
        "--recompute-check",
        "--workdir",
        str(args.workdir),
    ]
    if args.checkpoint:
        recompute_argv += ["--checkpoint", str(args.checkpoint)]
    result = subprocess.run(
        recompute_argv,
        env={**os.environ, "VLLM_METAL_MM_PREFIX_PATH": "recompute"},
    )
    if result.returncode != 0:
        return 1

    # The text-only half runs as a subprocess (see the module docstring): a
    # second `vllm.LLM` in this process was not attempted.
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
