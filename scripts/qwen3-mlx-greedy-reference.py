#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "mlx==0.32.3",
#   "mlx-lm==0.32.0",
# ]
# ///
"""Capture mlx-lm's greedy decoding of an MLX affine-quantized Qwen3 checkpoint.

For each fixed chat prompt this runs mlx-lm's own model and prompt cache,
takes the argmax of each step's logits, and records every step's token and
the gap between its two largest logits. A native implementation is
teacher-forced on these tokens: it must pick the same token wherever the
recorded gap is not a near tie. The printed JSON is the fixture.

This is a decoding-agreement oracle, not an inference benchmark.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache

PROMPTS = [
    "Explain how a hash map handles collisions.",
    "Write a Python function that merges two sorted lists.",
    "What causes the seasons on Earth?",
    "Summarize the plot of Romeo and Juliet in three sentences.",
    "List five prime numbers greater than 100 and explain how you checked them.",
    "Translate 'The library opens at nine tomorrow morning' into French and German.",
    "Write a short Rust function that reverses the words in a string.",
    "Why is the sky blue? Answer for a ten-year-old.",
]


def chat_text(user: str) -> str:
    # Qwen3's template for one user turn with thinking disabled.
    return (
        f"<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n"
        "<think>\n\n</think>\n\n"
    )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def greedy(model, prompt: list[int], steps: int) -> tuple[list[int], list[float]]:
    cache = make_prompt_cache(model)
    logits = model(mx.array(prompt)[None], cache=cache)[0, -1].astype(mx.float32)
    tokens: list[int] = []
    gaps: list[float] = []
    for step in range(steps):
        top_two = mx.sort(logits)[-2:]
        token = mx.argmax(logits)
        mx.eval(top_two, token)
        tokens.append(int(token.item()))
        gaps.append(float(top_two[1].item() - top_two[0].item()))
        if step + 1 < steps:
            logits = model(token.reshape(1, 1), cache=cache)[0, -1].astype(mx.float32)
    return tokens, gaps


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", type=Path, required=True, help="local checkpoint directory"
    )
    parser.add_argument("--steps", type=int, default=128)
    args = parser.parse_args()

    model, tokenizer = load(str(args.model))
    cases = []
    for user in PROMPTS:
        prompt = tokenizer.encode(chat_text(user), add_special_tokens=False)
        tokens, gaps = greedy(model, prompt, args.steps)
        cases.append(
            {"user": user, "prompt_ids": prompt, "tokens": tokens, "top2_gaps": gaps}
        )
    weights = sorted(args.model.glob("*.safetensors"))
    print(
        json.dumps(
            {
                "oracle": "mlx-lm greedy argmax with its prompt cache",
                "mlx": importlib.metadata.version("mlx"),
                "mlx_lm": importlib.metadata.version("mlx-lm"),
                "config_sha256": file_sha256(args.model / "config.json"),
                "weights_sha256": {path.name: file_sha256(path) for path in weights},
                "cases": cases,
            },
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
