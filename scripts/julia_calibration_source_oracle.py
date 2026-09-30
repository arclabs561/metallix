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
NATIVE_TRACE_MAX_BYTES = 2 * 1024 * 1024
NATIVE_TRACE_SCHEMA_VERSION = 2


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


def validate_arguments(
    check_case: bool,
    run_source: bool,
    output: Path | None,
    native_trace: Path | None = None,
) -> None:
    if check_case:
        if run_source or output is not None or native_trace is not None:
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


def validate_base_trace(torch: Any, trace: dict[str, Any], label: str) -> None:
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


def validate_trace(torch: Any, trace: dict[str, Any], label: str) -> None:
    positions = len(CASE["input_ids"])
    validate_base_trace(torch, trace, label)
    for name in ("source_pre_rotary_query", "source_pre_rotary_key"):
        validate_tensor(
            torch,
            trace.get(name),
            (1, HEADS, positions, HEAD_DIM),
            f"{label} {name}",
        )
    for name in ("source_rotary_cos", "source_rotary_sin"):
        validate_tensor(
            torch,
            trace.get(name),
            (1, positions, HEAD_DIM),
            f"{label} {name}",
        )
    for name in ("source_rotary_query", "source_rotary_key"):
        validate_tensor(
            torch,
            trace.get(name),
            (1, HEADS, positions, HEAD_DIM),
            f"{label} {name}",
        )


def rotate_half(torch: Any, value: Any) -> Any:
    half = value.shape[-1] // 2
    return torch.cat((-value[..., half:], value[..., :half]), dim=-1)


def replay_rope(torch: Any, trace: dict[str, Any]) -> tuple[Any, Any]:
    cos = trace["source_rotary_cos"].unsqueeze(1)
    sin = trace["source_rotary_sin"].unsqueeze(1)
    query = trace["source_pre_rotary_query"]
    key = trace["source_pre_rotary_key"]
    return (
        (query * cos) + (rotate_half(torch, query) * sin),
        (key * cos) + (rotate_half(torch, key) * sin),
    )


def source_rotary_positions(trace: dict[str, Any], name: str) -> Any:
    return trace[name].squeeze(0).permute(1, 0, 2)


def max_abs(left: Any, right: Any) -> float:
    return (left - right).abs().max().item()


def load_native_trace(torch: Any, path: Path) -> tuple[dict[str, Any], str]:
    with path.open("rb") as source:
        payload = source.read(NATIVE_TRACE_MAX_BYTES + 1)
    if len(payload) > NATIVE_TRACE_MAX_BYTES:
        raise ValueError(f"native trace exceeds {NATIVE_TRACE_MAX_BYTES} bytes")
    try:
        trace = json.loads(payload)
    except json.JSONDecodeError as error:
        raise ValueError("native trace is not valid JSON") from error
    if not isinstance(trace, dict):
        raise TypeError("native trace must be a JSON object")
    if trace.get("schema_version") != NATIVE_TRACE_SCHEMA_VERSION:
        raise ValueError(
            f"native trace schema_version does not match {NATIVE_TRACE_SCHEMA_VERSION}"
        )
    if trace.get("case") != CASE["name"]:
        raise ValueError("native trace case is not the fixed cal_len7 calibration case")
    if trace.get("attention_layout") != "head_query_key":
        raise ValueError("native trace attention layout is not head_query_key")
    positions = len(CASE["input_ids"])
    if trace.get("attention_shape") != [HEADS, positions, positions]:
        raise ValueError("native trace attention shape is not exact")
    for name in ("rotated_query", "rotated_key", "qkv"):
        try:
            value = torch.tensor(trace[name], dtype=torch.float32)
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"native trace {name} is not a numeric tensor") from error
        shape = (
            (positions, 3 * WIDTH) if name == "qkv" else (positions, HEADS, HEAD_DIM)
        )
        validate_tensor(torch, value, shape, f"native {name}")
        trace[name] = value
    return trace, hashlib.sha256(payload).hexdigest()


def write_exclusive(path: Path, receipt: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as output:
        output.write(json.dumps(receipt, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-case", action="store_true")
    parser.add_argument("--run-source", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--native-trace", type=Path)
    args = parser.parse_args()
    validate_arguments(args.check_case, args.run_source, args.output, args.native_trace)
    if args.check_case:
        validate_case(CASE)
        print(json.dumps({"case": CASE["name"], "held_out_accessed": False}))
        return

    import torch

    torch.set_num_threads(TORCH_THREADS)
    torch.set_num_interop_threads(TORCH_THREADS)
    full = source_module()
    eager = full.layer0_trace(CASE, "eager", capture_rope=True)
    sdpa = full.layer0_trace(CASE, "sdpa", capture_rope=True)
    sdpa_unobserved = full.layer0_trace(CASE, "sdpa")
    validate_trace(torch, eager, "eager")
    validate_trace(torch, sdpa, "SDPA")
    validate_base_trace(torch, sdpa_unobserved, "unobserved SDPA")
    eager_replayed_query, eager_replayed_key = replay_rope(torch, eager)
    sdpa_replayed_query, sdpa_replayed_key = replay_rope(torch, sdpa)
    rope_replay_gaps = {
        "eager_query": (eager["source_rotary_query"] - eager_replayed_query)
        .abs()
        .max()
        .item(),
        "eager_key": (eager["source_rotary_key"] - eager_replayed_key)
        .abs()
        .max()
        .item(),
        "sdpa_query": (sdpa["source_rotary_query"] - sdpa_replayed_query)
        .abs()
        .max()
        .item(),
        "sdpa_key": (sdpa["source_rotary_key"] - sdpa_replayed_key).abs().max().item(),
    }
    if any(gap != 0.0 for gap in rope_replay_gaps.values()):
        raise RuntimeError(
            f"captured source RoPE replay is not exact: {rope_replay_gaps}"
        )
    observer_fields = (
        "embedding",
        "qkv",
        "attended",
        "wo",
        "post_wo_residual",
        "rotated_query",
        "rotated_key",
        "logits",
        "probabilities",
    )
    observer_gaps = {
        field: max_abs(sdpa[field], sdpa_unobserved[field]) for field in observer_fields
    }
    observer_exact_bits = all(
        torch.equal(
            sdpa[field].contiguous().view(torch.int32),
            sdpa_unobserved[field].contiguous().view(torch.int32),
        )
        for field in observer_fields
    )
    if not observer_exact_bits:
        raise RuntimeError(f"RoPE observer changed SDPA trace: {observer_gaps}")
    actual_vs_reconstructed = {
        "eager_query": max_abs(
            source_rotary_positions(eager, "source_rotary_query"),
            eager["rotated_query"],
        ),
        "eager_key": max_abs(
            source_rotary_positions(eager, "source_rotary_key"),
            eager["rotated_key"],
        ),
        "sdpa_query": max_abs(
            source_rotary_positions(sdpa, "source_rotary_query"),
            sdpa["rotated_query"],
        ),
        "sdpa_key": max_abs(
            source_rotary_positions(sdpa, "source_rotary_key"),
            sdpa["rotated_key"],
        ),
    }
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
    native_comparison = None
    if args.native_trace is not None:
        native, native_sha256 = load_native_trace(torch, args.native_trace)
        native_qkv = native["qkv"].reshape(len(CASE["input_ids"]), 3, HEADS, HEAD_DIM)
        native_source_rotary = dict(sdpa)
        native_source_rotary["source_pre_rotary_query"] = (
            native_qkv[:, 0].permute(1, 0, 2).unsqueeze(0)
        )
        native_source_rotary["source_pre_rotary_key"] = (
            native_qkv[:, 1].permute(1, 0, 2).unsqueeze(0)
        )
        native_replayed = replay_rope(torch, native_source_rotary)
        native_comparison = {
            "native_vs_source_qkv_max_abs": max_abs(native["qkv"], sdpa["qkv"]),
            "native_rotary_vs_source_formula_on_native_qkv_max_abs": {
                name: max_abs(
                    native["rotated_" + name], value.squeeze(0).permute(1, 0, 2)
                )
                for name, value in zip(("query", "key"), native_replayed, strict=True)
            },
            "path": str(args.native_trace),
            "sha256": native_sha256,
            "schema_version": NATIVE_TRACE_SCHEMA_VERSION,
            "native_vs_source_actual_rotary_max_abs": {
                "query": max_abs(
                    native["rotated_query"],
                    source_rotary_positions(sdpa, "source_rotary_query"),
                ),
                "key": max_abs(
                    native["rotated_key"],
                    source_rotary_positions(sdpa, "source_rotary_key"),
                ),
            },
        }
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
            "source_pre_rotary_query": [1, HEADS, len(CASE["input_ids"]), HEAD_DIM],
            "source_pre_rotary_key": [1, HEADS, len(CASE["input_ids"]), HEAD_DIM],
            "source_rotary_cos": [1, len(CASE["input_ids"]), HEAD_DIM],
            "source_rotary_sin": [1, len(CASE["input_ids"]), HEAD_DIM],
            "source_rotary_query": [1, HEADS, len(CASE["input_ids"]), HEAD_DIM],
            "source_rotary_key": [1, HEADS, len(CASE["input_ids"]), HEAD_DIM],
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
        "captured_source_rope_replay_max_abs": rope_replay_gaps,
        "observer_noninterference_max_abs": observer_gaps,
        "observer_noninterference_exact_bits": observer_exact_bits,
        "source_actual_vs_existing_reconstructed_rotary_max_abs": actual_vs_reconstructed,
        "native_trace_comparison": native_comparison,
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
