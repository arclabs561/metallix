#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy==2.4.3", "torch==2.13.0"]
# ///
"""Capture FP32 rotary-tail fixtures from pinned DeepSeek-V4.1 source."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path

import torch

REVISION = "dba1be0a40aa45a94ad051997016db3960a90277"
SOURCE_SHA256 = "4e9ae23620edc8028ccc5d5fef552ab7fdc7dcd6f79608754fe9f67644056f65"
SOURCE_URL = f"https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/{REVISION}/inference/model.py"

CASES = [
    ("rank_three_forward", 1, 3, 1, 2, False),
    ("rank_four_multi_batch_head", 2, 2, 3, 2, False),
    ("rank_four_inverse", 1, 2, 2, 3, True),
]


def values_for(size: int, seed: int) -> list[float]:
    return [((index * 17 + seed) % 41 - 20) / 9 for index in range(size)]


def frequencies_for(positions: int, pairs: int, seed: int) -> torch.Tensor:
    angles = torch.tensor(
        [((index * 13 + seed) % 29 - 14) / 7 for index in range(positions * pairs)],
        dtype=torch.float32,
    ).reshape(positions, pairs)
    return torch.polar(torch.ones_like(angles), angles)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path, required=True, help="pinned inference/model.py"
    )
    args = parser.parse_args()
    source = args.source.read_bytes()
    if hashlib.sha256(source).hexdigest() != SOURCE_SHA256:
        parser.error("source SHA256 differs from the pinned official implementation")
    tree = ast.parse(source)
    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "apply_rotary_emb"
    ]
    if len(functions) != 1:
        parser.error("expected exactly one apply_rotary_emb function")
    namespace = {"torch": torch}
    exec(  # noqa: S102 - only the inspected function from the hash-pinned source
        compile(ast.Module(body=functions, type_ignores=[]), str(args.source), "exec"),
        namespace,
    )
    rotate = namespace["apply_rotary_emb"]
    torch.set_num_threads(1)
    captured = []
    with torch.inference_mode():
        for case_index, (name, batches, positions, heads, pairs, inverse) in enumerate(
            CASES
        ):
            values = values_for(batches * positions * heads * pairs * 2, case_index + 3)
            frequencies = frequencies_for(positions, pairs, case_index + 5)
            shape = (
                (batches, positions, pairs * 2)
                if heads == 1 and case_index == 0
                else (
                    batches,
                    positions,
                    heads,
                    pairs * 2,
                )
            )
            input_values = torch.tensor(values, dtype=torch.float32).reshape(shape)
            actual = rotate(input_values.clone(), frequencies, inverse=inverse)
            captured.append(
                {
                    "name": name,
                    "batches": batches,
                    "positions": positions,
                    "heads": heads,
                    "pairs": pairs,
                    "direction": "inverse" if inverse else "forward",
                    "values": values,
                    "frequencies": [
                        [value.real.item(), value.imag.item()]
                        for value in frequencies.flatten()
                    ],
                    "expected_values": actual.flatten().tolist(),
                }
            )
    print(
        json.dumps(
            {
                "schema_version": 1,
                "source": {
                    "url": SOURCE_URL,
                    "revision": REVISION,
                    "sha256": SOURCE_SHA256,
                    "symbol": "apply_rotary_emb",
                    "license": "MIT",
                },
                "reference": {
                    "torch_version": torch.__version__,
                    "device": "cpu",
                    "dtype": "float32",
                },
                "scope": "Synthetic FP32 adjacent-complex-pair rotary tails only. Covers upstream rank-three and rank-four broadcasting plus inverse conjugation. No RoPE-frequency generation, BF16/FP4 rounding, projections, KV cache, sparse attention, model weights, or Metal execution.",
                "cases": captured,
            },
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
