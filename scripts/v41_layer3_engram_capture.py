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
"""Project a bounded source receipt for the layer-three Engram-to-block seam."""

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
PARAMETER_LAYOUTS = {
    "layers.3.engram.embed.weight": ("torch.float8_e4m3fn", [204, 32]),
    "layers.3.engram.embed.scale": ("torch.float8_e8m0fnu", [204, 1]),
    "layers.3.engram.wkv.weight": ("torch.float8_e4m3fn", [384, 192]),
    "layers.3.engram.wkv.scale": ("torch.float8_e8m0fnu", [12, 6]),
    "layers.3.engram.q_weight": ("torch.bfloat16", [2, 128]),
    "layers.3.engram.k_weight": ("torch.bfloat16", [2, 128]),
}


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _require_dict(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{label} must be an object")
    return value


def _require_tensor(
    value: object, label: str, *, dtype: str, shape: list[int]
) -> dict[str, Any]:
    record = _require_dict(value, label)
    if record.get("dtype") != dtype or record.get("shape") != shape:
        raise RuntimeError(f"{label} has an unexpected dtype or shape")
    numel = 1
    for width in shape:
        numel *= width
    if record.get("numel") != numel or record.get("finite") is not True:
        raise RuntimeError(f"{label} has an invalid finite tensor receipt")
    storage_hex = record.get("storage_hex")
    storage_sha256 = record.get("storage_sha256")
    if not isinstance(storage_hex, str) or not isinstance(storage_sha256, str):
        raise TypeError(f"{label} lacks exact storage")
    try:
        raw = bytes.fromhex(storage_hex)
    except ValueError as error:
        raise RuntimeError(f"{label} has invalid storage hex") from error
    widths = {
        "torch.bfloat16": 2,
        "torch.float32": 4,
        "torch.int64": 8,
        "torch.float8_e4m3fn": 1,
        "torch.float8_e8m0fnu": 1,
    }
    if len(raw) != numel * widths[dtype] or _sha256(raw) != storage_sha256:
        raise RuntimeError(f"{label} exact storage does not match its receipt")
    nonfinite = False
    if dtype == "torch.float8_e4m3fn":
        nonfinite = any(code in (0x7F, 0xFF) for code in raw)
    elif dtype == "torch.float8_e8m0fnu":
        nonfinite = 0xFF in raw
    elif dtype in ("torch.bfloat16", "torch.float32"):
        width = widths[dtype]
        exponent_mask = 0x7F80 if width == 2 else 0x7F800000
        nonfinite = any(
            int.from_bytes(raw[offset : offset + width], "little") & exponent_mask
            == exponent_mask
            for offset in range(0, len(raw), width)
        )
    if nonfinite:
        raise RuntimeError(f"{label} contains nonfinite storage despite its receipt")
    return record


def _split_wkv(
    record: dict[str, Any], sequence: int
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Split each source WKV row into contiguous BF16 key and value tensors."""
    raw = bytes.fromhex(record["storage_hex"])
    key_raw = bytearray()
    value_raw = bytearray()
    row_width = 384 * 2
    for offset in range(0, len(raw), row_width):
        row = raw[offset : offset + row_width]
        key_raw.extend(row[: 256 * 2])
        value_raw.extend(row[256 * 2 :])

    def derived(raw_value: bytes, shape: list[int]) -> dict[str, Any]:
        return {
            "dtype": "torch.bfloat16",
            "shape": shape,
            "numel": len(raw_value) // 2,
            "finite": True,
            "storage_hex": raw_value.hex(),
            "storage_sha256": _sha256(raw_value),
        }

    return (
        derived(bytes(key_raw), [1, sequence, 2, 128]),
        derived(bytes(value_raw), [1, sequence, 128]),
    )


def engram_fixture(receipt: dict[str, object]) -> dict[str, object]:
    """Project only source records needed for native layer-three Engram replay."""
    if (
        receipt.get("capture_status")
        != "completed synthetic source-forward capture; no parity claim"
    ):
        raise RuntimeError("Engram fixture export requires a completed source capture")
    if receipt.get("coverage_status", {}).get("pending") != []:
        raise RuntimeError("Engram fixture export requires complete capture coverage")
    source = _require_dict(receipt.get("source"), "source")
    engram = _require_dict(receipt.get("engram"), "engram")
    encoded = _require_dict(receipt.get("encoded_parameters"), "encoded parameters")
    runtime = _require_dict(receipt.get("runtime"), "runtime")
    model_args = _require_dict(receipt.get("model_args"), "model args")
    steps = receipt.get("steps")
    if not isinstance(steps, list):
        raise TypeError("steps must be an array")
    if runtime.get("storage_byteorder") != "little":
        raise RuntimeError("Engram fixture export requires little-endian storage")
    if any(not isinstance(source.get(field), str) for field in SOURCE_FIELDS):
        raise RuntimeError("Engram fixture has incomplete source provenance")

    layout = _require_dict(engram.get("layout"), "Engram layout")
    hash_state = _require_dict(engram.get("hash_state"), "Engram hash state")
    if layout != {
        "layer_ids": [1, 3],
        "max_ngram_size": 4,
        "n_heads": 2,
        "head_dim": 32,
        "num_embeddings": [72, 204],
        "implied_prime_bucket_rows": [72, 204],
    }:
        raise RuntimeError("Engram fixture has an unexpected reduced hash layout")
    for name, shape, dtype in (
        ("token_map", [8], "torch.int64"),
        ("primes", [2, 3, 2], "torch.int64"),
        ("offsets", [2, 6], "torch.int64"),
        ("multipliers", [2, 4], "torch.int64"),
    ):
        _require_tensor(
            hash_state.get(name), f"Engram hash state {name}", dtype=dtype, shape=shape
        )
    if hash_state.get("pad_id") != 1:
        raise RuntimeError("Engram fixture has an unexpected hash pad ID")
    for name, (dtype, shape) in PARAMETER_LAYOUTS.items():
        _require_tensor(encoded.get(name), name, dtype=dtype, shape=shape)

    cases: list[dict[str, object]] = []
    for step in steps:
        step_record = _require_dict(step, "capture step")
        start = step_record.get("start_pos")
        sequence = 5 if start == 0 else 1
        if start not in (0, 5, 6):
            raise RuntimeError(
                "Engram fixture requires the pinned prefill/decode trace"
            )
        input_ids = _require_tensor(
            step_record.get("input_ids"),
            "step input IDs",
            dtype="torch.int64",
            shape=[1, sequence],
        )
        intermediates = _require_dict(
            step_record.get("intermediates"), "step intermediates"
        )
        gate_input = _require_dict(
            intermediates.get("layers.3.engram_input"), "layer-three Engram input"
        )
        stream = _require_tensor(
            gate_input.get("stream"),
            "layer-three Engram stream",
            dtype="torch.bfloat16",
            shape=[1, sequence, 2, 128],
        )
        hash_ids = _require_tensor(
            gate_input.get("hash_ids"),
            "layer-three hash IDs",
            dtype="torch.int64",
            shape=[1, sequence, 6],
        )
        if gate_input.get("mask") is not None:
            raise RuntimeError("pinned text trace unexpectedly supplied an Engram mask")
        embedding = _require_tensor(
            intermediates.get("layers.3.engram.embed"),
            "layer-three Engram embedding",
            dtype="torch.bfloat16",
            shape=[1, sequence, 6, 32],
        )
        wkv = _require_tensor(
            intermediates.get("layers.3.engram.wkv"),
            "layer-three Engram WKV",
            dtype="torch.bfloat16",
            shape=[1, sequence, 384],
        )
        output = _require_tensor(
            intermediates.get("layers.3.engram"),
            "layer-three Engram output",
            dtype="torch.bfloat16",
            shape=[1, sequence, 2, 128],
        )
        block_input = _require_dict(
            intermediates.get("layers.3.block_input"), "layer-three block input"
        )
        block_residual = _require_tensor(
            block_input.get("residual"),
            "layer-three block residual",
            dtype="torch.bfloat16",
            shape=[1, sequence, 2, 128],
        )
        if output["storage_sha256"] != block_residual["storage_sha256"]:
            raise RuntimeError(
                "layer-three Engram output does not feed layer-three block input"
            )
        key, value = _split_wkv(wkv, sequence)
        cases.append(
            {
                "start_pos": start,
                "input_ids": input_ids,
                "stream": stream,
                "captured_hash_ids": hash_ids,
                "embedding": embedding,
                "wkv_output": wkv,
                "key": key,
                "value": value,
                "output": output,
                "block_entry": block_residual,
            }
        )
    if [case["start_pos"] for case in cases] != [0, 5, 6]:
        raise RuntimeError("Engram fixture must retain prefill plus both decode steps")
    return {
        "schema_version": 1,
        "scope": (
            "layer-three source Engram hash, embedding, FP8 WKV projection, and residual "
            "gate into the layer-three block entry; not earlier layer residual production, "
            "full-model parity, or production serving"
        ),
        "source": {
            **{field: source[field] for field in SOURCE_FIELDS},
            "complete_capture_sha256": _sha256(serialized_capture(receipt)),
            "storage_byteorder": runtime["storage_byteorder"],
        },
        "model": {
            "copies": model_args.get("hc_mult"),
            "dim": model_args.get("dim"),
            "norm_eps": model_args.get("norm_eps"),
            "gate_clamp": 1e-6,
            "hash_columns": 6,
            "embedding_dim": 32,
            "wkv_width": 384,
        },
        "engram": copy.deepcopy(engram),
        "encoded_parameters": {
            name: copy.deepcopy(encoded[name]) for name in PARAMETER_LAYOUTS
        },
        "cases": cases,
        "comparison_policy": {
            "hash_ids": "native hash-state result must match captured IDs exactly",
            "embedding_bf16": "exact storage bits",
            "wkv_output_bf16": "exact storage bits",
            "key_value": "per-token exact BF16 split of WKV output",
            "output_block_entry": "exact storage identity",
        },
    }


def serialized_capture(receipt: dict[str, object]) -> bytes:
    return (
        json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode()


def _runner():
    path = Path(__file__).with_name("v41-forward-reference.py")
    spec = importlib.util.spec_from_file_location("v41_engram_runner", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("source runner import unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    fixture = engram_fixture(_runner().run_capture())
    payload = (
        json.dumps(fixture, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode()
    if len(payload) > MAX_FIXTURE_BYTES:
        raise RuntimeError(f"Engram fixture exceeds {MAX_FIXTURE_BYTES} bytes")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(payload)
    print(
        json.dumps(
            {
                "artifact_sha256": _sha256(payload),
                "bytes": len(payload),
                "path": str(args.output),
                "status": "source_layer_three_engram_fixture",
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
