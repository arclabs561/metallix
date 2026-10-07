#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "torch==2.13.0",
#   "transformers==5.18.0",
# ]
# ///
"""Write tiny synthetic Gemma 4 references for the gemma adapter's always-on tests.

Each variant builds a small `Gemma4ForCausalLM` from a `Gemma4TextConfig`,
replaces every parameter with seeded random values (norm scales near 1,
layer scalars and per-expert scales away from 1, so dropping any of them
changes the output), and records the source's prefill and cached greedy
decode logits. The JSON holds the configuration, every text tensor under
the names a checkpoint stores, and the logits:

- `ple`: per-layer input embeddings, K/V shared by the last two layers,
  double-wide MLP in those layers, sliding window 3. Shared layers also get
  the unused `k_proj`, `v_proj` and `k_norm` that published E2B and E4B
  checkpoints store and transformers ignores, marked `ignored`.
- `moe`: the 26B-A4B layer shape: a dense MLP plus routed experts in every
  layer, K-as-V full layers.

This is a numerical oracle for the layer math, not a model of any checkpoint.
"""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path
from typing import Any

import torch
import transformers
from transformers.models.gemma4 import configuration_gemma4, modeling_gemma4

DECODE_STEPS = 6
PROMPTS = {
    "short": [2, 17, 33, 5, 61, 9, 40, 12, 3],
    "long": [2, 7, 7, 30, 52, 11, 63, 1, 45, 28, 19, 6, 50, 34, 22, 8, 57, 13, 41, 26],
}
COMMON = {
    "vocab_size": 64,
    "hidden_size": 16,
    "num_attention_heads": 4,
    "head_dim": 8,
    "global_head_dim": 16,
    "rms_norm_eps": 1e-6,
    "max_position_embeddings": 64,
    "final_logit_softcapping": 30.0,
    "tie_word_embeddings": True,
    "hidden_activation": "gelu_pytorch_tanh",
    "rope_parameters": {
        "full_attention": {
            "partial_rotary_factor": 0.25,
            "rope_theta": 10000.0,
            "rope_type": "proportional",
        },
        "sliding_attention": {"rope_theta": 100.0, "rope_type": "default"},
    },
}
VARIANTS: dict[str, dict[str, Any]] = {
    "ple": {
        **COMMON,
        "num_hidden_layers": 6,
        "layer_types": ["sliding_attention", "sliding_attention", "full_attention"] * 2,
        "intermediate_size": 8,
        "num_key_value_heads": 2,
        "sliding_window": 3,
        "attention_k_eq_v": False,
        "hidden_size_per_layer_input": 4,
        "vocab_size_per_layer_input": 64,
        "num_kv_shared_layers": 2,
        "use_double_wide_mlp": True,
        "enable_moe_block": False,
    },
    "moe": {
        **COMMON,
        "num_hidden_layers": 4,
        "layer_types": ["sliding_attention", "full_attention"] * 2,
        "intermediate_size": 12,
        "num_key_value_heads": 2,
        "num_global_key_value_heads": 1,
        "sliding_window": 3,
        "attention_k_eq_v": True,
        "hidden_size_per_layer_input": 0,
        "num_kv_shared_layers": 0,
        "enable_moe_block": True,
        "num_experts": 8,
        "top_k_experts": 3,
        "moe_intermediate_size": 4,
    },
}


def f32_list(values: torch.Tensor) -> list[float]:
    """Nine significant digits round-trip every float32 exactly."""
    return [float(f"{value:.9g}") for value in values.flatten().tolist()]


def randomize(model: torch.nn.Module, seed: int) -> None:
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            noise = torch.rand(parameter.shape, generator=generator) - 0.5
            if name.endswith(("layer_scalar", "per_expert_scale")):
                parameter.copy_(0.75 + 0.5 * noise)
            elif "norm" in name or name.endswith("router.scale"):
                parameter.copy_(1.0 + 0.4 * noise)
            else:
                parameter.copy_(noise)
        for name, buffer in model.named_buffers():
            if name.endswith("layer_scalar"):
                buffer.copy_(
                    0.75 + 0.5 * (torch.rand(buffer.shape, generator=generator) - 0.5)
                )


def capture(model: torch.nn.Module, input_ids: list[int]) -> list[dict[str, Any]]:
    cache = transformers.DynamicCache(config=model.config)
    steps = []
    with torch.inference_mode():
        output = model(
            input_ids=torch.tensor([input_ids]), past_key_values=cache, use_cache=True
        )
        logits = output.logits[0, -1]
        steps.append({"input_ids": input_ids, "logits": f32_list(logits)})
        for _ in range(DECODE_STEPS):
            token = int(torch.argmax(logits))
            output = model(
                input_ids=torch.tensor([[token]]),
                past_key_values=output.past_key_values,
                use_cache=True,
            )
            logits = output.logits[0, -1]
            steps.append({"input_ids": [token], "logits": f32_list(logits)})
    return steps


def variant(name: str, settings: dict[str, Any], seed: int) -> dict[str, Any]:
    config = configuration_gemma4.Gemma4TextConfig(**settings)
    config._attn_implementation = "eager"
    torch.manual_seed(seed)
    model = modeling_gemma4.Gemma4ForCausalLM(config).to(torch.float32).eval()
    randomize(model, seed)
    tensors = {}
    for tensor_name, value in model.state_dict().items():
        if tensor_name == "lm_head.weight":
            continue  # tied to the embedding
        canonical = tensor_name.removeprefix("model.")
        tensors[canonical] = {
            "shape": list(value.shape),
            "values": f32_list(value),
        }
    # Published E2B/E4B shards keep K/V weights for shared layers; the
    # source never reads them. Random values make a reader that does use
    # them disagree with the reference.
    first_shared = settings["num_hidden_layers"] - settings["num_kv_shared_layers"]
    generator = torch.Generator().manual_seed(seed + 1)
    for layer in range(first_shared, settings["num_hidden_layers"]):
        full = settings["layer_types"][layer] == "full_attention"
        head_dim = settings["global_head_dim"] if full else settings["head_dim"]
        width = settings["num_key_value_heads"] * head_dim
        for suffix, shape in [
            ("k_proj.weight", [width, settings["hidden_size"]]),
            ("v_proj.weight", [width, settings["hidden_size"]]),
            ("k_norm.weight", [head_dim]),
        ]:
            values = torch.rand(shape, generator=generator) - 0.5
            tensors[f"layers.{layer}.self_attn.{suffix}"] = {
                "shape": shape,
                "values": f32_list(values),
                "ignored": True,
            }
    return {
        "name": name,
        "seed": seed,
        "config": {"model_type": "gemma4_text", **settings},
        "tensors": tensors,
        "cases": [
            {"name": prompt, "steps": capture(model, ids)}
            for prompt, ids in PROMPTS.items()
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, required=True, help="directory for the JSON files"
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    for index, (name, settings) in enumerate(VARIANTS.items()):
        document = {
            "reference": {
                "framework": f"transformers {transformers.__version__}",
                "runtime": f"torch {torch.__version__} CPU float32 eager, one thread",
                "platform": platform.platform(),
                "script": "scripts/gemma4-tiny-reference.py",
            },
            **variant(name, settings, seed=1000 + index),
        }
        path = args.output / f"{name}-reference.json"
        path.write_text(json.dumps(document, sort_keys=True) + "\n")
        print(path, sum(len(case["steps"]) for case in document["cases"]), "steps")


if __name__ == "__main__":
    main()
