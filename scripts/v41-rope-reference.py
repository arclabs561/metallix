#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy==2.4.3", "torch==2.13.0"]
# ///
"""Capture bounded RoPE-frequency slices from pinned DeepSeek-V4.1 source."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
from pathlib import Path

import torch

REVISION = "dba1be0a40aa45a94ad051997016db3960a90277"
SOURCE_SHA256 = "4e9ae23620edc8028ccc5d5fef552ab7fdc7dcd6f79608754fe9f67644056f65"
SOURCE_URL = f"https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/{REVISION}/inference/model.py"

# The final case asks the upstream function for a prefix ending at 65,537,
# then stores only its last four rows. This qualifies high absolute positions
# without committing a multi-megabyte golden fixture.
CASES = [
    ("local_no_yarn_prefix", 8, 0, 10_000.0, 16.0, 32.0, 1.0, 0, 4, False),
    (
        "compressed_yarn_mixed_ramp_prefix",
        64,
        65_536,
        160_000.0,
        16.0,
        32.0,
        1.0,
        0,
        4,
        False,
    ),
    (
        "compressed_yarn_mixed_ramp_high_positions",
        64,
        65_536,
        160_000.0,
        16.0,
        32.0,
        1.0,
        65_533,
        4,
        True,
    ),
]


def extract_functions(source: bytes, source_path: Path):
    tree = ast.parse(source)
    frequency_functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "precompute_freqs_cis"
    ]
    rotary_functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "apply_rotary_emb"
    ]
    if len(frequency_functions) != 1:
        raise ValueError("expected exactly one precompute_freqs_cis function")
    if len(rotary_functions) != 1:
        raise ValueError("expected exactly one apply_rotary_emb function")
    # Caching is not part of the numerical operator fixture. Removing this
    # decorator also avoids importing unrelated module globals from the model.
    frequency_functions[0].decorator_list = []
    namespace = {"math": math, "torch": torch}
    exec(  # noqa: S102 - only the inspected function from hash-pinned source
        compile(
            ast.fix_missing_locations(
                ast.Module(
                    body=[frequency_functions[0], rotary_functions[0]], type_ignores=[]
                )
            ),
            str(source_path),
            "exec",
        ),
        namespace,
    )
    return namespace["precompute_freqs_cis"], namespace["apply_rotary_emb"]


def values_for(size: int, seed: int) -> list[float]:
    return [((index * 17 + seed) % 41 - 20) / 9 for index in range(size)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path, required=True, help="pinned inference/model.py"
    )
    args = parser.parse_args()
    source = args.source.read_bytes()
    if hashlib.sha256(source).hexdigest() != SOURCE_SHA256:
        parser.error("source SHA256 differs from the pinned official implementation")
    generator, rotate = extract_functions(source, args.source)

    torch.set_num_threads(1)
    captured = []
    with torch.inference_mode():
        for (
            name,
            rotary_width,
            original_sequence_length,
            base,
            factor,
            beta_fast,
            beta_slow,
            start_position,
            positions,
            inverse,
        ) in CASES:
            source_seqlen = start_position + positions
            frequencies = generator(
                rotary_width,
                source_seqlen,
                original_sequence_length,
                base,
                factor,
                beta_fast,
                beta_slow,
            )[start_position:source_seqlen]
            values = values_for(positions * 2 * rotary_width, start_position + 7)
            input_values = torch.tensor(values, dtype=torch.float32).reshape(
                1, positions, 2, rotary_width
            )
            rotated = rotate(input_values.clone(), frequencies, inverse=inverse)
            captured.append(
                {
                    "name": name,
                    "rotary_width": rotary_width,
                    "original_sequence_length": original_sequence_length,
                    "base": base,
                    "factor": factor,
                    "beta_fast": beta_fast,
                    "beta_slow": beta_slow,
                    "start_position": start_position,
                    "positions": positions,
                    "batches": 1,
                    "heads": 2,
                    "direction": "inverse" if inverse else "forward",
                    "expected_frequencies": [
                        [value.real.item(), value.imag.item()]
                        for value in frequencies.flatten()
                    ],
                    "values": values,
                    "expected_rotated_values": rotated.flatten().tolist(),
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
                    "symbol": "precompute_freqs_cis",
                    "license": "MIT",
                },
                "reference": {
                    "torch_version": torch.__version__,
                    "device": "cpu",
                    "dtype": "float32",
                },
                "scope": "Synthetic FP32 RoPE-frequency slices and their application to rank-four adjacent-complex tails only. Includes the source's no-YaRN path, the released compressed-KV YaRN parameters, forward and inverse rotation, and high absolute positions. The capture removes only the source cache decorator. No BF16/FP4 rounding, projections, KV cache, sparse attention, model weights, or Metal execution.",
                "cases": captured,
            },
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
