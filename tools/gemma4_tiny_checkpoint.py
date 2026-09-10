# SPDX-License-Identifier: Apache-2.0
"""Build a tiny random-weight Gemma 4 checkpoint for wiring tests.

Real tokenizer, chat template and processor config come from a public
mlx-community repository; the model config keeps every field of the real
26B-A4B checkpoint except the sizes; weights are random and saved in the
mlx-vlm parameter layout, which both mlx_lm and mlx-vlm load without any
key rewriting (that is the layout real MLX checkpoints use).
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import mlx.core as mx
from huggingface_hub import hf_hub_download
from mlx.utils import tree_flatten

SOURCE_REPO = "mlx-community/gemma-4-26B-A4B-it-qat-4bit"
COPIED_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "generation_config.json",
    "processor_config.json",
)

TEXT_OVERRIDES = {
    "hidden_size": 64,
    "intermediate_size": 128,
    "num_hidden_layers": 2,
    "num_attention_heads": 2,
    "num_key_value_heads": 1,
    "num_global_key_value_heads": 1,
    "head_dim": 256,
    "global_head_dim": 512,
    "layer_types": ["sliding_attention", "full_attention"],
    "sliding_window": 1024,
    "num_kv_shared_layers": 0,
    "hidden_size_per_layer_input": 0,
    "enable_moe_block": False,
    "num_experts": None,
    "top_k_experts": None,
    "moe_intermediate_size": None,
    "dtype": "bfloat16",
}
VISION_OVERRIDES = {
    "hidden_size": 64,
    "intermediate_size": 128,
    "num_hidden_layers": 2,
    "num_attention_heads": 2,
    "num_key_value_heads": 2,
    "head_dim": 32,
    "global_head_dim": 32,
    "dtype": "bfloat16",
}


def _write_processor_config(path: Path) -> None:
    """transformers 5.14 needs a ``video_processor`` block that no repo ships."""
    config = json.loads(path.read_text())
    image = config["image_processor"]
    config.setdefault(
        "video_processor",
        {
            **{k: v for k, v in image.items() if k != "image_processor_type"},
            "video_processor_type": "Gemma4VideoProcessor",
        },
    )
    path.write_text(json.dumps(config, indent=2))


def _tiny_config(source_config: dict, *, with_vision: bool) -> dict:
    config = json.loads(json.dumps(source_config))
    config.pop("quantization", None)
    config.pop("quantization_config", None)
    config["text_config"].update(TEXT_OVERRIDES)
    config["dtype"] = "bfloat16"
    if with_vision:
        config["vision_config"].update(VISION_OVERRIDES)
    else:
        config.pop("vision_config", None)
    return config


def _random_weights(
    config: dict, *, seed: int, with_vision: bool
) -> dict[str, mx.array]:
    import mlx_vlm.models.gemma4 as gemma4_module
    from mlx_vlm.utils import update_module_configs

    model_config = gemma4_module.ModelConfig.from_dict(config)
    model_config = update_module_configs(
        model_config, gemma4_module, config, ["text", "vision", "audio"]
    )
    model = gemma4_module.Model(model_config)
    mx.random.seed(seed)
    weights: dict[str, mx.array] = {}
    for name, value in tree_flatten(model.parameters()):
        if not with_vision and (
            name.startswith("vision_tower.") or name.startswith("embed_vision.")
        ):
            continue
        if name.endswith(("norm.weight", "position_embedding_table")) or (
            "norm" in name.split(".")[-2:]
        ):
            weights[name] = mx.ones(value.shape, dtype=mx.bfloat16)
        else:
            weights[name] = (mx.random.normal(value.shape) * 0.02).astype(mx.bfloat16)
    return weights


def build_tiny_checkpoint(
    out_dir: Path,
    *,
    source_repo: str = SOURCE_REPO,
    seed: int = 0,
    with_vision: bool = True,
) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in COPIED_FILES:
        shutil.copy(hf_hub_download(source_repo, name), out_dir / name)
    _write_processor_config(out_dir / "processor_config.json")

    source_config = json.loads(
        Path(hf_hub_download(source_repo, "config.json")).read_text()
    )
    config = _tiny_config(source_config, with_vision=with_vision)
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))

    weights = _random_weights(config, seed=seed, with_vision=with_vision)
    shard = "model.safetensors"
    mx.save_safetensors(str(out_dir / shard), weights)
    (out_dir / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {
                    "total_size": sum(int(v.nbytes) for v in weights.values())
                },
                "weight_map": dict.fromkeys(weights, shard),
            },
            indent=2,
        )
    )
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--no-vision", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    build_tiny_checkpoint(args.out_dir, seed=args.seed, with_vision=not args.no_vision)
    print(args.out_dir)


if __name__ == "__main__":
    main()
