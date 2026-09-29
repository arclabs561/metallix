#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["torch==2.13.0", "transformers==5.0.0"]
# ///
"""Pinned-source eager attention control for the fixed cal_len7 calibration case."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
from types import ModuleType
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
CASE = {
    "name": "cal_len7",
    "input_ids": [2, 4, 6, 1, 3, 5, 7],
    "attention_mask": [True, True, True, True, True, True, True],
}
WIDTH, HEADS, HEAD_DIM = 384, 6, 64
CONTROL_TOLERANCE = 1e-5
TORCH_THREADS = 1


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_module() -> ModuleType:
    path = ROOT / "scripts/julia_full_prefill_reference.py"
    spec = importlib.util.spec_from_file_location("julia_full_prefill_reference", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load Julia full-prefill source control")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_case(case: dict[str, Any]) -> None:
    if case != CASE:
        raise ValueError(
            "source oracle only permits the fixed cal_len7 calibration case"
        )


def validate_arguments(check_case: bool, run_source: bool, output: Path | None) -> None:
    if check_case:
        if run_source or output is not None:
            raise ValueError("--check-case cannot execute or write a source receipt")
    elif not run_source or output is None:
        raise ValueError("source execution requires --run-source and --output")


def validate_tensor(
    torch: Any, value: object, shape: tuple[int, ...], label: str
) -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{label} is not a tensor")
    if tuple(value.shape) != shape:
        raise ValueError(f"{label} shape {tuple(value.shape)} does not equal {shape}")
    if not torch.isfinite(value).all():
        raise ValueError(f"{label} contains a non-finite value")


def validate_trace(torch: Any, trace: dict[str, Any], label: str) -> None:
    positions = len(CASE["input_ids"])
    validate_tensor(torch, trace.get("qkv"), (positions, 3 * WIDTH), f"{label} QKV")
    validate_tensor(
        torch, trace.get("logits"), (HEADS, positions, positions), f"{label} logits"
    )
    validate_tensor(
        torch,
        trace.get("probabilities"),
        (HEADS, positions, positions),
        f"{label} probabilities",
    )
    validate_tensor(
        torch, trace.get("attended"), (positions, WIDTH), f"{label} attended"
    )


def write_exclusive(path: Path, receipt: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as output:
        output.write(json.dumps(receipt, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-case", action="store_true")
    parser.add_argument("--run-source", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    validate_arguments(args.check_case, args.run_source, args.output)
    if args.check_case:
        validate_case(CASE)
        print(json.dumps({"case": CASE["name"], "held_out_accessed": False}))
        return

    import torch

    torch.set_num_threads(TORCH_THREADS)
    torch.set_num_interop_threads(TORCH_THREADS)
    full = source_module()
    eager = full.layer0_trace(CASE, "eager")
    sdpa = full.layer0_trace(CASE, "sdpa")
    validate_trace(torch, eager, "eager")
    validate_trace(torch, sdpa, "SDPA")
    qkv = eager["qkv"].reshape(len(CASE["input_ids"]), 3, HEADS, HEAD_DIM)
    values = qkv[:, 2]
    reconstructed = torch.einsum(
        "hqk,khd->qhd", eager["probabilities"], values
    ).reshape(len(CASE["input_ids"]), WIDTH)
    validate_tensor(
        torch,
        reconstructed,
        (len(CASE["input_ids"]), WIDTH),
        "explicit eager attended",
    )
    eager_gap = (eager["attended"] - reconstructed).abs().max().item()
    if not math.isfinite(eager_gap) or eager_gap > CONTROL_TOLERANCE:
        raise RuntimeError(
            f"explicit eager reconstruction {eager_gap} exceeds {CONTROL_TOLERANCE}"
        )
    receipt = {
        "schema_version": 1,
        "scope": "cal_len7 calibration source eager oracle control only",
        "held_out_accessed": False,
        "case": CASE,
        "input_sha256": hashlib.sha256(
            json.dumps(CASE, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "expected_shapes": {
            "qkv": [len(CASE["input_ids"]), 3 * WIDTH],
            "logits": [HEADS, len(CASE["input_ids"]), len(CASE["input_ids"])],
            "probabilities": [
                HEADS,
                len(CASE["input_ids"]),
                len(CASE["input_ids"]),
            ],
            "attended": [len(CASE["input_ids"]), WIDTH],
        },
        "source": {
            "modernbert_revision": full.ENCODER.REVISION,
            "modernbert_sha256": full.ENCODER.SOURCE_SHA256,
            "full_prefill_reference_sha256": sha256(
                ROOT / "scripts/julia_full_prefill_reference.py"
            ),
            "oracle_script_sha256": sha256(Path(__file__)),
        },
        "runtime": {
            "torch_version": torch.__version__,
            "torch_config": torch.__config__.show(),
            "torch_threads": torch.get_num_threads(),
            "torch_interop_threads": torch.get_num_interop_threads(),
            "attention_implementations": ["eager", "sdpa"],
            "model_config": {
                "hidden_size": WIDTH,
                "heads": HEADS,
                "head_dim": HEAD_DIM,
                "layers": 22,
                "rope_theta": 160000.0,
                "norm_eps": 1e-5,
            },
        },
        "control_tolerance": CONTROL_TOLERANCE,
        "eager_actual_vs_explicit_attended_max_abs": eager_gap,
        "sdpa_actual_vs_eager_actual_attended_max_abs": (
            sdpa["attended"] - eager["attended"]
        )
        .abs()
        .max()
        .item(),
        "sdpa_actual_vs_eager_actual_qkv_max_abs": (sdpa["qkv"] - eager["qkv"])
        .abs()
        .max()
        .item(),
    }
    write_exclusive(args.output, receipt)


if __name__ == "__main__":
    main()
