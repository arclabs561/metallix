#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "torch==2.13.0",
#   "numpy==2.5.3",
#   "sympy==1.14.0",
#   "tokenizers==0.23.2",
# ]
# ///
"""Project the source layer-two FFN tail that feeds the layer-three Engram."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any

MAX_FIXTURE_BYTES = 1_000_000
SOURCE_FIELDS = (
    "revision",
    "model_sha256",
    "engram_sha256",
    "kernel_source_sha256",
    "cpu_backend_sha256",
    "loader_sha256",
    "runner_sha256",
    "forward_observers_sha256",
)
LAYER = "layers.2"


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def serialized_capture(receipt: dict[str, object]) -> bytes:
    return (
        json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode()


def _object(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{label} must be an object")
    return value


def _tensor(
    value: object, label: str, *, dtype: str, shape: list[int]
) -> dict[str, Any]:
    record = _object(value, label)
    if record.get("dtype") != dtype or record.get("shape") != shape:
        raise RuntimeError(f"{label} has an unexpected dtype or shape")
    expected = 1
    for dimension in shape:
        expected *= dimension
    if record.get("numel") != expected or record.get("finite") is not True:
        raise RuntimeError(f"{label} has an invalid finite receipt")
    storage_hex = record.get("storage_hex")
    storage_sha256 = record.get("storage_sha256")
    if not isinstance(storage_hex, str) or not isinstance(storage_sha256, str):
        raise TypeError(f"{label} lacks exact storage")
    try:
        raw = bytes.fromhex(storage_hex)
    except ValueError as error:
        raise RuntimeError(f"{label} has invalid storage hex") from error
    byte_width = {
        "torch.bfloat16": 2,
        "torch.float32": 4,
        "torch.int64": 8,
        "torch.float8_e4m3fn": 1,
        "torch.float8_e8m0fnu": 1,
        # The source stores its packed FP4 tensor as one observable uint8 per
        # logical entry in this receipt, so preserve the recorded byte count.
        "torch.float4_e2m1fn_x2": 1,
    }[dtype]
    expected_bytes = expected * byte_width
    if len(raw) != expected_bytes or _sha256(raw) != storage_sha256:
        raise RuntimeError(f"{label} exact storage does not match its receipt")
    nonfinite = False
    if dtype == "torch.float8_e4m3fn":
        nonfinite = any(code in (0x7F, 0xFF) for code in raw)
    elif dtype == "torch.float8_e8m0fnu":
        nonfinite = 0xFF in raw
    elif dtype in ("torch.bfloat16", "torch.float32"):
        exponent_mask = 0x7F80 if dtype == "torch.bfloat16" else 0x7F800000
        width = 2 if dtype == "torch.bfloat16" else 4
        nonfinite = any(
            int.from_bytes(raw[offset : offset + width], "little") & exponent_mask
            == exponent_mask
            for offset in range(0, len(raw), width)
        )
    if nonfinite:
        raise RuntimeError(f"{label} contains nonfinite storage despite its receipt")
    return record


def _parameters(encoded: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    ffn = f"{LAYER}.ffn"
    routed = tuple(f"{ffn}.experts.{index}" for index in range(4))
    experts = (*routed, f"{ffn}.shared_experts")
    required = {
        f"{ffn}.gate.weight",
        f"{ffn}.gate.bias",
        *(
            f"{prefix}.{projection}.{field}"
            for prefix in experts
            for projection in ("w1", "w2", "w3")
            for field in ("weight", "scale")
        ),
    }
    selected = {name: encoded.get(name) for name in required}
    if any(value is None for value in selected.values()):
        raise RuntimeError("layer-two fixture lacks a routed or shared MoE parameter")
    layouts: dict[str, tuple[str, list[int]]] = {
        f"{ffn}.gate.weight": ("torch.bfloat16", [4, 128]),
        f"{ffn}.gate.bias": ("torch.float32", [4]),
    }
    for prefix in routed:
        for projection in ("w1", "w2", "w3"):
            layouts[f"{prefix}.{projection}.weight"] = (
                "torch.float4_e2m1fn_x2",
                [128, 64],
            )
            layouts[f"{prefix}.{projection}.scale"] = ("torch.float8_e8m0fnu", [128, 4])
    for projection in ("w1", "w2", "w3"):
        layouts[f"{ffn}.shared_experts.{projection}.weight"] = (
            "torch.float8_e4m3fn",
            [128, 128],
        )
        layouts[f"{ffn}.shared_experts.{projection}.scale"] = (
            "torch.float8_e8m0fnu",
            [4, 4],
        )
    for name, (dtype, shape) in layouts.items():
        _tensor(selected[name], name, dtype=dtype, shape=shape)

    block_names = (
        f"{LAYER}.hc_ffn_fn",
        f"{LAYER}.hc_ffn_base",
        f"{LAYER}.hc_ffn_scale",
        f"{LAYER}.ffn_norm.weight",
    )
    block = {name: encoded.get(name) for name in block_names}
    expected = {
        f"{LAYER}.hc_ffn_fn": ("torch.float32", [8, 256]),
        f"{LAYER}.hc_ffn_base": ("torch.float32", [8]),
        f"{LAYER}.hc_ffn_scale": ("torch.float32", [3]),
        f"{LAYER}.ffn_norm.weight": ("torch.bfloat16", [128]),
    }
    for name, (dtype, shape) in expected.items():
        _tensor(block[name], name, dtype=dtype, shape=shape)
    return selected, block


def _coefficients(value: object, label: str, sequence: int) -> dict[str, Any]:
    record = _object(value, label)
    if set(record) != {"pre", "post", "comb"}:
        raise RuntimeError(f"{label} lacks the complete HC coefficient set")
    _tensor(
        record["pre"], f"{label} pre", dtype="torch.float32", shape=[1, sequence, 2]
    )
    return record


def layer2_ffn_fixture(receipt: dict[str, object]) -> dict[str, object]:
    """Project the actual source tail from layer-two attention output to Engram3."""
    if (
        receipt.get("capture_status")
        != "completed synthetic source-forward capture; no parity claim"
    ):
        raise RuntimeError("layer-two FFN fixture requires a completed source capture")
    coverage = receipt.get("coverage_status")
    source = _object(receipt.get("source"), "source")
    runtime = _object(receipt.get("runtime"), "runtime")
    model = _object(receipt.get("model_args"), "model args")
    encoded = _object(receipt.get("encoded_parameters"), "encoded parameters")
    steps = receipt.get("steps")
    if (
        not isinstance(coverage, dict)
        or coverage.get("pending") != []
        or not isinstance(steps, list)
    ):
        raise RuntimeError("layer-two FFN fixture requires complete source coverage")
    if runtime.get("storage_byteorder") != "little":
        raise RuntimeError("layer-two FFN fixture requires little-endian storage")
    if any(not isinstance(source.get(field), str) for field in SOURCE_FIELDS):
        raise RuntimeError("layer-two FFN fixture has incomplete source provenance")
    if any(
        model.get(name) != value
        for name, value in {
            "dim": 128,
            "hc_mult": 2,
            "moe_inter_dim": 128,
            "n_routed_experts": 4,
            "n_activated_experts": 2,
            "n_shared_experts": 1,
        }.items()
    ):
        raise RuntimeError(
            "layer-two FFN fixture has unexpected reduced model geometry"
        )
    parameters, block_parameters = _parameters(encoded)

    cases: list[dict[str, object]] = []
    for step in steps:
        source_step = _object(step, "capture step")
        start = source_step.get("start_pos")
        sequence = 5 if start == 0 else 1
        if start not in (0, 5, 6):
            raise RuntimeError("layer-two FFN fixture requires the pinned trace")
        intermediates = _object(source_step.get("intermediates"), "step intermediates")
        calls = source_step.get("hyper_connection_mixes")
        if not isinstance(calls, list):
            raise TypeError("step lacks HC observations")
        by_kind = {
            call.get("sublayer"): call
            for call in calls
            if isinstance(call, dict) and call.get("layer_id") == 2
        }
        if set(by_kind) != {"attention", "ffn"}:
            raise RuntimeError("step lacks both layer-two HC observations")
        attention = _coefficients(
            by_kind["attention"].get("outputs"), "attention coefficients", sequence
        )
        ffn = _coefficients(by_kind["ffn"].get("outputs"), "FFN coefficients", sequence)
        # Decode rows are singleton, so only the position dimension varies.
        for record in (
            attention["post"],
            attention["comb"],
            ffn["pre"],
            ffn["post"],
            ffn["comb"],
        ):
            if (
                not isinstance(record, dict)
                or record.get("shape", [None, None])[1] != sequence
            ):
                raise RuntimeError("layer-two HC coefficient position shape changed")
        after_attention = _tensor(
            intermediates.get(f"{LAYER}.after_attention_residual"),
            "layer-two post-attention residual",
            dtype="torch.bfloat16",
            shape=[1, sequence, 2, 128],
        )
        collapsed = _tensor(
            intermediates.get(f"{LAYER}.ffn_collapsed"),
            "layer-two FFN collapsed input",
            dtype="torch.bfloat16",
            shape=[1, sequence, 128],
        )
        moe_input = _tensor(
            intermediates.get(f"{LAYER}.ffn_input"),
            "layer-two MoE input",
            dtype="torch.bfloat16",
            shape=[1, sequence, 128],
        )
        moe_output = _tensor(
            intermediates.get(f"{LAYER}.ffn"),
            "layer-two MoE output",
            dtype="torch.bfloat16",
            shape=[1, sequence, 128],
        )
        terminal = intermediates.get(LAYER)
        if not isinstance(terminal, list) or len(terminal) != 2:
            raise TypeError(
                "layer-two terminal state must be residual plus next pre-mix"
            )
        output = _tensor(
            terminal[0],
            "layer-two terminal residual",
            dtype="torch.bfloat16",
            shape=[1, sequence, 2, 128],
        )
        next_pre = _tensor(
            terminal[1],
            "layer-two returned pre-mix",
            dtype="torch.float32",
            shape=[1, sequence, 2],
        )
        engram = _object(
            intermediates.get("layers.3.engram_input"), "layer-three Engram input"
        )
        engram_stream = _tensor(
            engram.get("stream"),
            "layer-three Engram stream",
            dtype="torch.bfloat16",
            shape=[1, sequence, 2, 128],
        )
        layer_three = _object(
            intermediates.get("layers.3.block_input"), "layer-three block input"
        )
        layer_three_pre = _tensor(
            layer_three.get("incoming_pre"),
            "layer-three incoming pre-mix",
            dtype="torch.float32",
            shape=[1, sequence, 2],
        )
        if output["storage_sha256"] != engram_stream["storage_sha256"]:
            raise RuntimeError(
                "layer-two terminal residual does not feed the Engram stream"
            )
        if next_pre["storage_sha256"] != layer_three_pre["storage_sha256"]:
            raise RuntimeError("layer-two returned pre-mix does not feed layer three")
        raw = {
            kind: _object(by_kind[kind].get("inputs"), f"{kind} HC inputs").get("mixes")
            for kind in ("attention", "ffn")
        }
        for kind, record in raw.items():
            _tensor(
                record,
                f"{kind} raw HC mixes",
                dtype="torch.float32",
                shape=[1, sequence, 8],
            )
        cases.append(
            {
                "start_pos": start,
                "after_attention_residual": after_attention,
                "attention_pre": attention["pre"],
                "attention_coefficients": attention,
                "attention_hc_mixes": raw["attention"],
                "ffn_collapsed": collapsed,
                "moe_input": moe_input,
                "moe_output": moe_output,
                "ffn_coefficients": ffn,
                "ffn_hc_mixes": raw["ffn"],
                "output": output,
                "next_pre": next_pre,
                "engram_stream": engram_stream,
                "layer_three_incoming_pre": layer_three_pre,
            }
        )
    if [case["start_pos"] for case in cases] != [0, 5, 6]:
        raise RuntimeError(
            "layer-two FFN fixture must retain prefill and both decode calls"
        )
    return {
        "schema_version": 1,
        "scope": "layer-two source FFN tail from captured post-attention residual to the layer-three Engram stream and block incoming pre-mix; not layer-two attention, layer-one shared-compression production, full-model parity, or serving",
        "source": {
            **{field: source[field] for field in SOURCE_FIELDS},
            "complete_capture_sha256": _sha256(serialized_capture(receipt)),
            "storage_byteorder": runtime["storage_byteorder"],
        },
        "model": {
            name: model[name]
            for name in (
                "dim",
                "hc_mult",
                "moe_inter_dim",
                "n_routed_experts",
                "n_activated_experts",
                "n_shared_experts",
                "score_func",
                "gate_temp",
                "norm_topk_prob",
                "route_scale",
                "swiglu_limit",
                "expert_dtype",
            )
        },
        "block_config": {
            "copies": model["hc_mult"],
            "hc_sinkhorn_iters": model["hc_sinkhorn_iters"],
            "hc_eps": model["hc_eps"],
            "norm_eps": model["norm_eps"],
        },
        "encoded_parameters": copy.deepcopy(parameters),
        "block_parameters": copy.deepcopy(block_parameters),
        "cases": cases,
        "comparison_policy": {
            "ffn_collapsed_bf16": "exact storage bits",
            "moe_input_bf16": "exact storage bits",
            "moe_output_bf16": "exact storage bits",
            "output_engram_stream": "exact storage identity",
            "next_pre_layer_three_input": "exact storage identity",
            "hc_coefficients": "source records retained for fixed source-derived envelope checks",
        },
    }


def _runner():
    path = Path(__file__).with_name("v41-forward-reference.py")
    spec = importlib.util.spec_from_file_location("v41_layer2_ffn_runner", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("source runner import unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    fixture = layer2_ffn_fixture(_runner().run_capture())
    payload = (
        json.dumps(fixture, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode()
    if len(payload) > MAX_FIXTURE_BYTES:
        raise RuntimeError(f"layer-two FFN fixture exceeds {MAX_FIXTURE_BYTES} bytes")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(payload)
    print(
        json.dumps(
            {
                "artifact_sha256": _sha256(payload),
                "bytes": len(payload),
                "path": str(args.output),
                "status": "source_layer_two_ffn_fixture",
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
