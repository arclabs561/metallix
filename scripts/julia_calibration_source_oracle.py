#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["torch==2.13.0", "transformers==5.0.0"]
# ///
"""Pinned-source controls for fixed cal_len7 and cal_len2 calibration probes."""

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
LEN2_CASE = {
    "name": "cal_len2",
    "input_ids": [6, 1],
    "attention_mask": [True, True],
}
WIDTH, HEADS, HEAD_DIM = 384, 6, 64
CONTROL_TOLERANCE = 1e-5
TORCH_THREADS = 1
NATIVE_TRACE_MAX_BYTES = 2 * 1024 * 1024
CALIBRATION_REPORT_MAX_BYTES = 128 * 1024 * 1024
NATIVE_TRACE_SCHEMA_VERSION = 2
NATIVE_EMBEDDING_TRACE_SCHEMA_VERSION = 1
LEN2_INTERVENTION_SCHEMA_VERSION = 1
MANIFEST_SHA256 = "e41b492e7ec8e0b0545515eda40fae63d4b1f87ea8fb54545acb277911f62b82"
WEIGHT_SHA256 = "db22ef523c79b55a019a8e62f8b096157035af0945d380a9bbe6bab5586cdf68"


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


def normalization_tree_module() -> ModuleType:
    path = ROOT / "scripts/julia_calibration_norm_tree.py"
    spec = importlib.util.spec_from_file_location("julia_calibration_norm_tree", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load Julia normalization-tree control")
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
    native_embedding_trace: Path | None = None,
) -> None:
    if check_case:
        if (
            run_source
            or output is not None
            or native_trace is not None
            or native_embedding_trace is not None
        ):
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
    validate_tensor(torch, trace.get("lookup"), (positions, WIDTH), f"{label} lookup")
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


def load_native_embedding_trace(torch: Any, path: Path) -> tuple[dict[str, Any], str]:
    with path.open("rb") as source:
        payload = source.read(NATIVE_TRACE_MAX_BYTES + 1)
    if len(payload) > NATIVE_TRACE_MAX_BYTES:
        raise ValueError(
            f"native embedding trace exceeds {NATIVE_TRACE_MAX_BYTES} bytes"
        )
    try:
        trace = json.loads(payload)
    except json.JSONDecodeError as error:
        raise ValueError("native embedding trace is not valid JSON") from error
    if not isinstance(trace, dict):
        raise TypeError("native embedding trace must be a JSON object")
    if trace.get("schema_version") != NATIVE_EMBEDDING_TRACE_SCHEMA_VERSION:
        raise ValueError("native embedding trace schema_version does not match 1")
    if trace.get("case") != CASE["name"]:
        raise ValueError("native embedding trace case is not cal_len7")
    if trace.get("input_ids") != CASE["input_ids"]:
        raise ValueError("native embedding trace input IDs do not match cal_len7")
    if trace.get("attention_mask") != CASE["attention_mask"]:
        raise ValueError("native embedding trace mask does not match cal_len7")
    positions = len(CASE["input_ids"])
    for name in ("lookup", "embedding"):
        try:
            value = torch.tensor(trace[name], dtype=torch.float32)
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"native embedding trace {name} is not a numeric tensor"
            ) from error
        validate_tensor(torch, value, (positions, WIDTH), f"native {name}")
        trace[name] = value
    return trace, hashlib.sha256(payload).hexdigest()


def validate_len2_intervention_arguments(
    run_len2_intervention: bool,
    output: Path | None,
    native_trace: Path | None,
    calibration_report: Path | None,
    check_case: bool,
    run_source: bool,
    legacy_native_trace: Path | None,
    native_embedding_trace: Path | None,
) -> None:
    if not run_len2_intervention:
        if native_trace is not None or calibration_report is not None:
            raise ValueError(
                "--len2-native-trace and --len2-calibration-report require --run-len2-intervention"
            )
        return
    if (
        output is None
        or native_trace is None
        or calibration_report is None
        or check_case
        or run_source
        or legacy_native_trace is not None
        or native_embedding_trace is not None
    ):
        raise ValueError(
            "--run-len2-intervention requires its trace, report, and output without legacy modes"
        )


def read_bounded(path: Path, limit: int, label: str) -> bytes:
    with path.open("rb") as source:
        payload = source.read(limit + 1)
    if len(payload) > limit:
        raise ValueError(f"{label} exceeds {limit} bytes")
    return payload


def as_f32_tensor(torch: Any, value: object, shape: tuple[int, ...], label: str) -> Any:
    try:
        result = torch.tensor(value, dtype=torch.float32)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} is not a numeric tensor") from error
    validate_tensor(torch, result, shape, label)
    return result


def require_exact_f32_bits(torch: Any, left: Any, right: Any, label: str) -> None:
    if tuple(left.shape) != tuple(right.shape) or not torch.equal(
        left.contiguous().view(torch.int32), right.contiguous().view(torch.int32)
    ):
        raise RuntimeError(f"{label} is not F32-bit-exact")


def comparison_metric(torch: Any, left: Any, right: Any) -> dict[str, float | int]:
    difference = (left.to(torch.float64) - right.to(torch.float64)).abs().flatten()
    maximum, index = difference.max(dim=0)
    return {
        "max_abs": maximum.item(),
        "flat_index": index.item(),
        "count": difference.numel(),
    }


def qkv_override_gate(
    comparisons: dict[str, dict[str, dict[str, float | int]]],
) -> dict[str, Any]:
    """Apply the fixed stagewise non-inferiority check for the QKV intervention."""
    stages = ("attended", "post_wo_residual", "layer_0_output")
    results = []
    for stage in stages:
        values = comparisons.get(stage)
        if not isinstance(values, dict):
            raise TypeError(f"QKV override comparison lacks {stage}")
        baseline = values.get("scalar_native_vs_same_input_source")
        candidate = values.get("scalar_native_vs_qkv_override_source")
        if not isinstance(baseline, dict) or not isinstance(candidate, dict):
            raise TypeError(f"QKV override comparison lacks {stage} metrics")
        baseline_max = baseline.get("max_abs")
        candidate_max = candidate.get("max_abs")
        if type(baseline_max) not in {float, int} or type(candidate_max) not in {
            float,
            int,
        }:
            raise TypeError(f"QKV override comparison {stage} has invalid maxima")
        results.append(
            {
                "stage": stage,
                "baseline_max_abs": baseline_max,
                "override_max_abs": candidate_max,
                "no_worse": candidate_max <= baseline_max,
                "strict_improvement": candidate_max < baseline_max,
            }
        )
    no_worse = all(result["no_worse"] for result in results)
    strict_improvement = any(result["strict_improvement"] for result in results)
    return {
        "stages": results,
        "no_worse_each_downstream_stage": no_worse,
        "strict_improvement_at_least_one_downstream_stage": strict_improvement,
        "passes": no_worse and strict_improvement,
        "limitation": "stage maxima can occur at different coordinates, so this necessary diagnostic gate does not establish QKV projection as the dominant error source",
    }


def load_len2_intervention_trace(torch: Any, path: Path) -> tuple[dict[str, Any], str]:
    payload = read_bounded(
        path, NATIVE_TRACE_MAX_BYTES, "native len2 intervention trace"
    )
    try:
        trace = json.loads(payload)
    except json.JSONDecodeError as error:
        raise ValueError("native len2 intervention trace is not valid JSON") from error
    if not isinstance(trace, dict):
        raise TypeError("native len2 intervention trace must be a JSON object")
    if (
        trace.get("schema_version") != LEN2_INTERVENTION_SCHEMA_VERSION
        or trace.get("case") != LEN2_CASE["name"]
        or trace.get("input_ids") != LEN2_CASE["input_ids"]
        or trace.get("attention_mask") != LEN2_CASE["attention_mask"]
        or trace.get("manifest_sha256") != MANIFEST_SHA256
        or trace.get("weight_f32_sha256") != WEIGHT_SHA256
    ):
        raise ValueError("native len2 intervention trace does not bind frozen cal_len2")
    positions = len(LEN2_CASE["input_ids"])
    shapes = {
        "qkv": (positions, 3 * WIDTH),
        "rotated_query": (positions, HEADS, HEAD_DIM),
        "rotated_key": (positions, HEADS, HEAD_DIM),
        "logits": (HEADS, positions, positions),
        "probabilities": (HEADS, positions, positions),
        "attended": (positions, WIDTH),
        "post_wo_residual": (positions, WIDTH),
        "output": (positions, WIDTH),
    }
    for branch in ("scalar", "balanced"):
        record = trace.get(branch)
        if not isinstance(record, dict) or not isinstance(record.get("trace"), dict):
            raise TypeError(f"native {branch} intervention record is malformed")
        record["embedding"] = as_f32_tensor(
            torch,
            record.get("embedding"),
            (positions, WIDTH),
            f"native {branch} embedding",
        )
        values = record["trace"]
        if values.get("attention_layout") != "head_query_key" or values.get(
            "attention_shape"
        ) != [HEADS, positions, positions]:
            raise ValueError(f"native {branch} attention layout is invalid")
        for name, shape in shapes.items():
            values[name] = as_f32_tensor(
                torch, values.get(name), shape, f"native {branch} {name}"
            )
    return trace, hashlib.sha256(payload).hexdigest()


def load_len2_cached_calibration_case(
    torch: Any, path: Path
) -> tuple[dict[str, Any], str]:
    payload = read_bounded(path, CALIBRATION_REPORT_MAX_BYTES, "calibration report")
    try:
        report = json.loads(payload)
    except json.JSONDecodeError as error:
        raise ValueError("calibration report is not valid JSON") from error
    if (
        not isinstance(report, dict)
        or report.get("manifest_sha256") != MANIFEST_SHA256
        or report.get("weight_f32_sha256") != WEIGHT_SHA256
    ):
        raise ValueError("calibration report does not bind frozen identities")
    matches = [
        case
        for case in report.get("cases", [])
        if isinstance(case, dict)
        and case.get("name") == LEN2_CASE["name"]
        and case.get("split") == "calibration"
    ]
    if len(matches) != 1:
        raise ValueError("calibration report requires one cal_len2 calibration case")
    case = matches[0]
    native = case.get("native_f32_vs_f64")
    source = case.get("source_f32_boundaries")
    if (
        not isinstance(native, dict)
        or not isinstance(source, dict)
        or not isinstance(native.get("boundaries"), list)
        or len(native["boundaries"]) != 24
    ):
        raise ValueError(
            "calibration report lacks exact cal_len2 source/native boundaries"
        )
    positions = len(LEN2_CASE["input_ids"])
    case["cached_native_embedding"] = as_f32_tensor(
        torch, native["boundaries"][0], (positions, WIDTH), "cached native embedding"
    )
    case["cached_native_layer0"] = as_f32_tensor(
        torch, native["boundaries"][1], (positions, WIDTH), "cached native layer_0"
    )
    for name in ("embedding", "layer_0"):
        value = source.get(name)
        if not isinstance(value, dict):
            raise TypeError(f"cached source {name} is invalid")
        case[f"cached_source_{name}"] = as_f32_tensor(
            torch,
            value.get("value"),
            (positions, WIDTH),
            f"cached source {name}",
        )
    return case, hashlib.sha256(payload).hexdigest()


def run_len2_intervention(
    torch: Any, native_path: Path, calibration_path: Path, output: Path
) -> None:
    native, native_sha256 = load_len2_intervention_trace(torch, native_path)
    cached, calibration_sha256 = load_len2_cached_calibration_case(
        torch, calibration_path
    )
    full = source_module()
    tree = normalization_tree_module()
    tree.torch = torch
    ordinary = full.layer0_trace(LEN2_CASE, "sdpa", capture_rope=True)
    scalar = full.layer0_trace(
        LEN2_CASE,
        "sdpa",
        capture_rope=True,
        embedding_override=native["scalar"]["embedding"],
    )
    scalar_qkv_override = full.layer0_trace(
        LEN2_CASE,
        "sdpa",
        capture_rope=True,
        embedding_override=native["scalar"]["embedding"],
        qkv_override=native["scalar"]["trace"]["qkv"],
    )
    balanced = full.layer0_trace(
        LEN2_CASE,
        "sdpa",
        capture_rope=True,
        embedding_override=native["balanced"]["embedding"],
    )
    positions = len(LEN2_CASE["input_ids"])
    for label, trace in (
        ("ordinary source", ordinary),
        ("scalar same-input source", scalar),
        ("scalar QKV-override source", scalar_qkv_override),
        ("balanced same-input source", balanced),
    ):
        for name, shape in {
            "embedding": (positions, WIDTH),
            "qkv": (positions, 3 * WIDTH),
            "attended": (positions, WIDTH),
            "post_wo_residual": (positions, WIDTH),
            "layer_0": (positions, WIDTH),
        }.items():
            validate_tensor(torch, trace.get(name), shape, f"{label} {name}")
    require_exact_f32_bits(
        torch,
        native["scalar"]["embedding"],
        cached["cached_native_embedding"],
        "scalar native embedding versus cached baseline",
    )
    require_exact_f32_bits(
        torch,
        native["scalar"]["trace"]["output"],
        cached["cached_native_layer0"],
        "scalar native layer_0 versus cached baseline",
    )
    require_exact_f32_bits(
        torch,
        ordinary["embedding"],
        cached["cached_source_embedding"],
        "ordinary source embedding versus cached baseline",
    )
    require_exact_f32_bits(
        torch,
        ordinary["layer_0"],
        cached["cached_source_layer_0"],
        "ordinary source layer_0 versus cached baseline",
    )
    require_exact_f32_bits(
        torch,
        scalar["embedding"],
        native["scalar"]["embedding"],
        "source scalar intervention embedding",
    )
    require_exact_f32_bits(
        torch,
        scalar_qkv_override["embedding"],
        native["scalar"]["embedding"],
        "source scalar QKV-override embedding",
    )
    require_exact_f32_bits(
        torch,
        scalar_qkv_override["qkv"],
        native["scalar"]["trace"]["qkv"],
        "source scalar QKV override",
    )
    require_exact_f32_bits(
        torch,
        balanced["embedding"],
        native["balanced"]["embedding"],
        "source balanced intervention embedding",
    )
    generated_lookup = full.ENCODER.values((full.VOCAB, WIDTH), 200)[
        torch.tensor(LEN2_CASE["input_ids"], dtype=torch.int64)
    ]
    require_exact_f32_bits(
        torch,
        ordinary["lookup"],
        generated_lookup,
        "ordinary source lookup versus generated rows",
    )
    expected_balanced_embedding = tree.balanced_two_pass(
        generated_lookup, full.near_one(201)
    )
    require_exact_f32_bits(
        torch,
        native["balanced"]["embedding"],
        expected_balanced_embedding,
        "native balanced embedding versus independent Python tree",
    )
    stages = {
        "qkv": "qkv",
        "attended": "attended",
        "post_wo_residual": "post_wo_residual",
        "layer_0_output": "output",
    }
    comparisons = {}
    for stage, native_name in stages.items():
        source_name = "layer_0" if stage == "layer_0_output" else native_name
        comparisons[stage] = {
            "scalar_native_vs_same_input_source": comparison_metric(
                torch, native["scalar"]["trace"][native_name], scalar[source_name]
            ),
            "scalar_native_vs_qkv_override_source": comparison_metric(
                torch,
                native["scalar"]["trace"][native_name],
                scalar_qkv_override[source_name],
            ),
            "balanced_native_vs_same_input_source": comparison_metric(
                torch,
                native["balanced"]["trace"][native_name],
                balanced[source_name],
            ),
            "source_balanced_vs_scalar_input_effect": comparison_metric(
                torch, balanced[source_name], scalar[source_name]
            ),
            "native_balanced_vs_scalar_total_effect": comparison_metric(
                torch,
                native["balanced"]["trace"][native_name],
                native["scalar"]["trace"][native_name],
            ),
            "ordinary_source_vs_scalar_input_effect": comparison_metric(
                torch, scalar[source_name], ordinary[source_name]
            ),
        }
    qkv_gate = qkv_override_gate(comparisons)
    write_exclusive(
        output,
        {
            "schema_version": 1,
            "scope": "cal_len2 layer-zero scalar/balanced embedding intervention, stagewise association only",
            "held_out_accessed": False,
            "case": LEN2_CASE,
            "identities": {
                "native_trace_sha256": native_sha256,
                "calibration_report_sha256": calibration_sha256,
                "manifest_sha256": MANIFEST_SHA256,
                "weight_f32_sha256": WEIGHT_SHA256,
                "full_prefill_reference_sha256": sha256(
                    ROOT / "scripts/julia_full_prefill_reference.py"
                ),
                "normalization_tree_control_sha256": sha256(
                    ROOT / "scripts/julia_calibration_norm_tree.py"
                ),
                "oracle_script_sha256": sha256(Path(__file__)),
            },
            "controls": {
                "scalar_native_matches_cached_baseline_bits": True,
                "ordinary_source_matches_cached_baseline_bits": True,
                "source_scalar_embedding_matches_native_bits": True,
                "source_scalar_qkv_override_matches_native_bits": True,
                "source_balanced_embedding_matches_native_bits": True,
                "native_balanced_embedding_matches_python_tree_bits": True,
            },
            "comparisons": comparisons,
            "qkv_override_gate": qkv_gate,
            "interpretation": "scalar and balanced source replays separate same-input native/source arithmetic gaps from the pure source response to the changed embedding; the QKV override is a necessary stagewise discriminator, not dominance proof or runtime authorization",
        },
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
    parser.add_argument("--native-trace", type=Path)
    parser.add_argument("--native-embedding-trace", type=Path)
    parser.add_argument("--run-len2-intervention", action="store_true")
    parser.add_argument("--len2-native-trace", type=Path)
    parser.add_argument("--len2-calibration-report", type=Path)
    args = parser.parse_args()
    validate_len2_intervention_arguments(
        args.run_len2_intervention,
        args.output,
        args.len2_native_trace,
        args.len2_calibration_report,
        args.check_case,
        args.run_source,
        args.native_trace,
        args.native_embedding_trace,
    )
    if args.run_len2_intervention:
        import torch

        torch.set_num_threads(TORCH_THREADS)
        torch.set_num_interop_threads(TORCH_THREADS)
        run_len2_intervention(
            torch,
            args.len2_native_trace,
            args.len2_calibration_report,
            args.output,
        )
        return
    validate_arguments(
        args.check_case,
        args.run_source,
        args.output,
        args.native_trace,
        args.native_embedding_trace,
    )
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
        "lookup",
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
    native_embedding_comparison = None
    if args.native_embedding_trace is not None:
        native_embedding, native_embedding_sha256 = load_native_embedding_trace(
            torch, args.native_embedding_trace
        )
        generated_lookup = full.ENCODER.values((full.VOCAB, WIDTH), 200)[
            torch.tensor(CASE["input_ids"], dtype=torch.int64)
        ]
        source_lookup_identity = max_abs(sdpa["lookup"], generated_lookup)
        if source_lookup_identity != 0.0:
            raise RuntimeError(
                "captured source lookup does not exactly reproduce generated F32 rows"
            )
        native_embedding_comparison = {
            "path": str(args.native_embedding_trace),
            "sha256": native_embedding_sha256,
            "schema_version": NATIVE_EMBEDDING_TRACE_SCHEMA_VERSION,
            "source_lookup_vs_generated_rows_max_abs": source_lookup_identity,
            "native_vs_source_actual_max_abs": {
                "lookup": max_abs(native_embedding["lookup"], sdpa["lookup"]),
                "embedding": max_abs(native_embedding["embedding"], sdpa["embedding"]),
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
        "native_embedding_trace_comparison": native_embedding_comparison,
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
