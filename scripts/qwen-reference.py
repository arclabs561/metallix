#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "torch==2.13.0",
#   "transformers==5.12.1",
# ]
# ///
"""Capture a deterministic CPU float32 Qwen3 numerical-forward reference.

This deliberately takes raw IDs rather than tokenizer text.  It is a narrow
checkpoint-parity oracle for Metallix, not an inference benchmark.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import struct
from pathlib import Path
from typing import Any

import torch
import transformers


def parse_input_ids(value: str) -> list[int]:
    try:
        input_ids = [int(item) for item in value.split(",")]
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "input IDs must be comma-separated integers"
        ) from error
    if not input_ids:
        raise argparse.ArgumentTypeError("input IDs must not be empty")
    if any(input_id < 0 for input_id in input_ids):
        raise argparse.ArgumentTypeError("input IDs must be non-negative")
    return input_ids


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sampled_vector(values: torch.Tensor, count: int = 8) -> list[float]:
    return [float(value) for value in values.detach().cpu().reshape(-1)[:count]]


def write_f32_le(path: Path, values: torch.Tensor) -> dict[str, int | str]:
    """Write a contiguous CPU tensor as raw little-endian IEEE-754 f32."""
    floats = [float(value) for value in values.detach().cpu().reshape(-1)]
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = struct.pack(f"<{len(floats)}f", *floats)
    with path.open("wb") as file:
        file.write(payload)
    return {
        "element_count": len(floats),
        "byte_count": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def capture(
    model_dir: Path,
    input_ids: list[int],
    logits_output: Path | None = None,
    model_id: str = "Qwen/Qwen3-0.6B",
) -> dict[str, Any]:
    config_path = model_dir / "config.json"
    weights_path = model_dir / "model.safetensors"
    for path in (config_path, weights_path):
        if not path.is_file():
            raise FileNotFoundError(f"required local model artifact is missing: {path}")

    # A single CPU thread and eager attention make this a reproducible
    # correctness oracle. They are intentionally not performance settings.
    torch.set_num_threads(1)
    torch.set_grad_enabled(False)
    config = transformers.AutoConfig.from_pretrained(
        model_dir, local_files_only=True, trust_remote_code=False
    )
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_dir,
        attn_implementation="eager",
        local_files_only=True,
        dtype=torch.float32,
        trust_remote_code=False,
    )
    model.to(device="cpu", dtype=torch.float32)
    model.eval()

    tokens = torch.tensor([input_ids], device="cpu", dtype=torch.long)
    with torch.inference_mode():
        embeddings = model.get_input_embeddings()(tokens)
        output = model(
            input_ids=tokens,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )

    first_layer = output.hidden_states[1]
    final_logits = output.logits[0, -1]
    logits_sidecar = (
        write_f32_le(logits_output, final_logits) if logits_output is not None else None
    )
    top_logits, top_ids = torch.topk(final_logits, k=8)
    return {
        "model_id": model_id,
        "input_ids": input_ids,
        "reference": {
            "framework": f"transformers {transformers.__version__}",
            "runtime": f"torch {torch.__version__} CPU float32 eager, one thread",
            "platform": platform.platform(),
            "config_sha256": sha256_file(config_path),
            "weights_sha256": sha256_file(weights_path),
            "model_type": config.model_type,
            "vocab_size": config.vocab_size,
            "hidden_size": config.hidden_size,
            "layer_count": config.num_hidden_layers,
        },
        **(
            {"last_token_logits_f32le": logits_sidecar}
            if logits_sidecar is not None
            else {}
        ),
        "embedding": {
            "shape": list(embeddings.shape),
            "token_0_first8": sampled_vector(embeddings[0, 0]),
            "last_token_first8": sampled_vector(embeddings[0, -1]),
        },
        "first_layer_hidden_state": {
            "shape": list(first_layer.shape),
            "token_0_first8": sampled_vector(first_layer[0, 0]),
            "last_token_first8": sampled_vector(first_layer[0, -1]),
        },
        "last_token_logits": {
            "first8": sampled_vector(final_logits),
            "top8": {
                "ids": [int(value) for value in top_ids.cpu().tolist()],
                "logits": [float(value) for value in top_logits.cpu().tolist()],
            },
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", type=Path, required=True, help="local Qwen3 model directory"
    )
    parser.add_argument(
        "--model-id",
        default="Qwen/Qwen3-0.6B",
        help="human-readable source ID recorded in the capture",
    )
    parser.add_argument(
        "--input-ids",
        type=parse_input_ids,
        default=[1, 2, 3],
        help="comma-separated raw token IDs (default: 1,2,3)",
    )
    parser.add_argument(
        "--logits-output",
        type=Path,
        help="optional runtime sidecar: full final-token logits as raw little-endian f32",
    )
    args = parser.parse_args()
    print(
        json.dumps(
            capture(args.model, args.input_ids, args.logits_output, args.model_id),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
