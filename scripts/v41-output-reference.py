#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = ["torch==2.13.0"]
# ///
"""Capture the pinned V4.1 grouped BF16 output projection on CPU only.

This is an independent small-tensor capture of the grouped ``wo_a`` einsum.
It verifies the pinned upstream source for provenance, but neither imports nor
executes the model, CUDA, Triton, or checkpoint weights.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path

import torch

REVISION = "dba1be0a40aa45a94ad051997016db3960a90277"
SOURCE_SHA256 = "4e9ae23620edc8028ccc5d5fef552ab7fdc7dcd6f79608754fe9f67644056f65"
SOURCE_URL = (
    "https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/"
    f"{REVISION}/inference/model.py"
)
EXPECTED_TORCH_VERSION = "2.13.0"
EINSUM_EQUATION = "bsgd,grd->bsgr"


def signed_i16(bits: int) -> int:
    """Convert one serialized BF16 word to the signed representation Torch views."""
    return bits if bits < 0x8000 else bits - 0x1_0000


def bf16_tensor(bits: list[int], shape: list[int]) -> torch.Tensor:
    """Construct a CPU BF16 tensor from exact serialized BF16 words."""
    expected = 1
    for dimension in shape:
        expected *= dimension
    if len(bits) != expected:
        raise ValueError(f"BF16 input has {len(bits)} values, expected {expected}")
    return (
        torch.tensor(
            [signed_i16(value) for value in bits], dtype=torch.int16, device="cpu"
        )
        .view(torch.bfloat16)
        .reshape(shape)
    )


def bf16_bits(value: torch.Tensor) -> list[int]:
    """Serialize a CPU BF16 tensor without converting its values to JSON floats."""
    if value.device.type != "cpu" or value.dtype is not torch.bfloat16:
        raise ValueError("expected a CPU BF16 tensor")
    words = value.contiguous().view(torch.int16).reshape(-1).tolist()
    return [int(word) & 0xFFFF for word in words]


def case(
    name: str,
    input_shape: list[int],
    input_bf16_bits: list[int],
    weight_shape: list[int],
    weight_bf16_bits: list[int],
) -> dict[str, object]:
    """Execute the exact grouped projection expression on one bounded case."""
    input_value = bf16_tensor(input_bf16_bits, input_shape)
    weight = bf16_tensor(weight_bf16_bits, weight_shape)
    with torch.inference_mode():
        output = torch.einsum(EINSUM_EQUATION, input_value, weight)
    if output.dtype is not torch.bfloat16:
        raise ValueError("CPU grouped projection did not preserve BF16 output")
    return {
        "name": name,
        "input_shape": input_shape,
        "input_bf16_bits": input_bf16_bits,
        "weight_shape": weight_shape,
        "weight_bf16_bits": weight_bf16_bits,
        "expected_shape": list(output.shape),
        "expected_output_bf16_bits": bf16_bits(output),
    }


def assert_grouped_projection_in_source(tree: ast.Module) -> None:
    """Check the pinned Attention source still contains the captured operation."""
    attention = next(
        (
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "Attention"
        ),
        None,
    )
    if attention is None:
        raise ValueError("expected exactly one Attention class")
    forward = next(
        (
            node
            for node in attention.body
            if isinstance(node, ast.FunctionDef) and node.name == "forward"
        ),
        None,
    )
    if forward is None:
        raise ValueError("expected Attention.forward")
    calls = [
        node
        for node in ast.walk(forward)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "torch"
        and node.func.attr == "einsum"
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and node.args[0].value == EINSUM_EQUATION
    ]
    if len(calls) != 1:
        raise ValueError(f"expected one Attention.forward {EINSUM_EQUATION!r} einsum")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path, required=True, help="pinned inference/model.py"
    )
    args = parser.parse_args()
    source = args.source.read_bytes()
    if hashlib.sha256(source).hexdigest() != SOURCE_SHA256:
        parser.error("source SHA256 differs from the pinned official implementation")
    if torch.__version__.split("+", maxsplit=1)[0] != EXPECTED_TORCH_VERSION:
        parser.error(
            f"requires torch=={EXPECTED_TORCH_VERSION}, found {torch.__version__}"
        )
    assert_grouped_projection_in_source(ast.parse(source))
    torch.set_num_threads(1)

    cases = [
        case(
            "nontrivial_batch_sequence_group_rank",
            [2, 2, 2, 3],
            [
                0x3F80,
                0x3F00,
                0xBF80,
                0x4000,
                0xBF00,
                0x3E80,
                0xBF00,
                0x3F80,
                0x4000,
                0x3FC0,
                0xBF80,
                0x3F00,
                0x3E80,
                0x4000,
                0x3F00,
                0xBF80,
                0x3FC0,
                0xBF00,
                0x3F80,
                0xBF00,
                0x4000,
                0xBF80,
                0x3E80,
                0x3FC0,
            ],
            [2, 2, 3],
            [
                0x3F80,
                0xBF80,
                0x3F00,
                0x3E80,
                0x4000,
                0xBF00,
                0xBF80,
                0x3F00,
                0x3F80,
                0x4000,
                0x3E80,
                0xBF00,
            ],
        ),
        case(
            "group_isolation",
            [1, 1, 2, 3],
            [0x3F80, 0xC000, 0x4040, 0x0000, 0x8000, 0x0000],
            [2, 2, 3],
            [
                0x3F80,
                0x3F00,
                0xBF80,
                0xBF00,
                0x4000,
                0x3F80,
                0x42C8,
                0xC2C8,
                0x42C8,
                0xC2C8,
                0x42C8,
                0xC2C8,
            ],
        ),
        case(
            "output_rounding_after_multi_term_accumulation",
            [1, 1, 1, 4],
            [0x3F80, 0x3F80, 0x3F80, 0x3F80],
            [1, 1, 4],
            [0x3F80, 0x3C00, 0x3B80, 0x3B00],
        ),
    ]
    print(
        json.dumps(
            {
                "schema_version": 1,
                "source": {
                    "url": SOURCE_URL,
                    "revision": REVISION,
                    "sha256": SOURCE_SHA256,
                    "symbol": "Attention.forward:wo_a_grouped_einsum",
                    "license": "MIT",
                },
                "reference": {
                    "torch_version": torch.__version__,
                    "device": "cpu",
                    "input_dtype": "bfloat16",
                    "output_dtype": "bfloat16_bits",
                    "einsum": EINSUM_EQUATION,
                },
                "scope": (
                    "Independent CPU Torch BF16 grouped wo_a projection on bounded "
                    "synthetic inputs. The pinned source is hash-checked and AST-inspected "
                    "for the einsum, but this script does not import or execute the full "
                    "model, CUDA, Triton, checkpoint weights, attention, RoPE, routing, or wo_b."
                ),
                "cases": cases,
            },
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
