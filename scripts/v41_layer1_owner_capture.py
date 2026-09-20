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
"""Project the source-owned layer-one ratio-two KV/index publication."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
from pathlib import Path
from typing import Any

MAX_INPUT_BYTES = 16 * 1024 * 1024
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
STARTS = ((0, 5, 2, 5), (5, 1, 3, 6), (6, 1, 3, 6))
LAYER = "layers.1.attn"
PINNED_SOURCE = {
    "revision": "dba1be0a40aa45a94ad051997016db3960a90277",
    "model_sha256": "4e9ae23620edc8028ccc5d5fef552ab7fdc7dcd6f79608754fe9f67644056f65",
    "engram_sha256": "11f35ecbead8150c35aa002b3d180ef290b05a25afe883a11884f94d476d3897",
    "kernel_source_sha256": "1236c3507019ed176f5dba5e04bcea58867cf654818c6cf138ed4845398c2455",
}


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _object(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"layer-one owner fixture lacks object {label}")
    return value


def _tensor(
    value: object,
    label: str,
    *,
    dtype: str,
    shape: list[int],
    negative_infinity_only: bool = False,
) -> dict[str, Any]:
    record = _object(value, label)
    if record.get("dtype") != dtype or record.get("shape") != shape:
        raise RuntimeError(f"{label} has an unexpected dtype or shape")
    expected = math.prod(shape)
    if record.get("numel") != expected:
        raise RuntimeError(f"{label} has an invalid element count")
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
        "torch.int32": 4,
        "torch.complex64": 8,
        "torch.float8_e4m3fn": 1,
        "torch.float8_e8m0fnu": 1,
    }
    if len(raw) != expected * widths[dtype] or _sha256(raw) != storage_sha256:
        raise RuntimeError(f"{label} exact storage does not match its receipt")
    if dtype == "torch.bfloat16":
        words = [
            int.from_bytes(raw[offset : offset + 2], "little")
            for offset in range(0, len(raw), 2)
        ]
        nonfinite = [word for word in words if ((word >> 7) & 0xFF) == 0xFF]
        if negative_infinity_only:
            if (
                record.get("finite") is not False
                or not nonfinite
                or any(word != 0xFF80 for word in nonfinite)
            ):
                raise RuntimeError(f"{label} must contain only negative infinity")
        elif record.get("finite") is not True or nonfinite:
            raise RuntimeError(f"{label} contains nonfinite BF16 storage")
    elif dtype == "torch.float32":
        if record.get("finite") is not True or any(
            not math.isfinite(value) for value in struct.unpack(f"<{expected}f", raw)
        ):
            raise RuntimeError(f"{label} contains nonfinite FP32 storage")
    elif dtype == "torch.complex64":
        if record.get("finite") is not True or any(
            not math.isfinite(value)
            for value in struct.unpack(f"<{expected * 2}f", raw)
        ):
            raise RuntimeError(f"{label} contains nonfinite complex storage")
    elif dtype == "torch.float8_e4m3fn":
        if record.get("finite") is not True or any(
            byte in (0x7F, 0xFF) for byte in raw
        ):
            raise RuntimeError(f"{label} contains nonfinite E4M3 storage")
    elif dtype == "torch.float8_e8m0fnu":
        if record.get("finite") is not True or 0xFF in raw:
            raise RuntimeError(f"{label} contains nonfinite E8M0 storage")
    elif record.get("finite") is not True:
        raise RuntimeError(f"{label} is not finite")
    return record


def _parameters(encoded: dict[str, Any]) -> dict[str, Any]:
    layouts = {
        f"{LAYER}.compressor.wkv.weight": ("torch.float32", [64, 128]),
        f"{LAYER}.compressor.wgate.weight": ("torch.float32", [64, 128]),
        f"{LAYER}.compressor.norm.weight": ("torch.bfloat16", [64]),
        f"{LAYER}.indexer.wk.weight": ("torch.bfloat16", [64, 64]),
        f"{LAYER}.indexer.k_norm.weight": ("torch.bfloat16", [64]),
        f"{LAYER}.indexer.wq_b.weight": ("torch.float8_e4m3fn", [128, 32]),
        f"{LAYER}.indexer.wq_b.scale": ("torch.float8_e8m0fnu", [4, 1]),
        f"{LAYER}.indexer.weights_proj.weight": ("torch.bfloat16", [2, 128]),
    }
    return {
        name: _tensor(encoded.get(name), name, dtype=dtype, shape=shape)
        for name, (dtype, shape) in layouts.items()
    }


def _validate_source(source: dict[str, Any]) -> None:
    for field, expected in PINNED_SOURCE.items():
        if source.get(field) != expected:
            raise RuntimeError(f"layer-one owner fixture has unexpected source {field}")
    scripts = Path(__file__).parent
    for field, filename in (
        ("cpu_backend_sha256", "v41_cpu_kernels.py"),
        ("loader_sha256", "v41_source_loader.py"),
        ("runner_sha256", "v41-forward-reference.py"),
        ("forward_observers_sha256", "v41_forward_observers.py"),
    ):
        if source.get(field) != _sha256((scripts / filename).read_bytes()):
            raise RuntimeError(f"layer-one owner fixture has stale source {field}")


def _operations(
    value: object,
    sequence: int,
    compressed: int,
    *,
    published_groups: int,
) -> dict[str, Any]:
    operations = _object(value, "layer-one index operations")
    expected = {
        "q_after_rope_fp4": ("torch.bfloat16", [1, sequence, 2, 64], False),
        "scaled_weights": ("torch.bfloat16", [1, sequence, 2], False),
        "scores_einsum": ("torch.bfloat16", [1, sequence, 2, compressed], False),
        "scores_after_relu": ("torch.bfloat16", [1, sequence, 2, compressed], False),
        "scores_weighted_per_head": (
            "torch.bfloat16",
            [1, sequence, 2, compressed],
            False,
        ),
        "scores_after_head_sum": ("torch.bfloat16", [1, sequence, compressed], False),
        "weights_proj_output": ("torch.bfloat16", [1, sequence, 2], False),
    }
    if published_groups:
        expected["k_after_rope_fp4"] = (
            "torch.bfloat16",
            [1, published_groups, 64],
            False,
        )
    if sequence > 1:
        expected["scores_after_causal_mask"] = (
            "torch.bfloat16",
            [1, sequence, compressed],
            True,
        )
    if set(operations) != set(expected):
        raise RuntimeError(
            "layer-one index operation set does not match owner schedule"
        )
    return {
        name: _tensor(
            operations[name],
            f"layer-one index {name}",
            dtype=dtype,
            shape=shape,
            negative_infinity_only=negative_infinity_only,
        )
        for name, (dtype, shape, negative_infinity_only) in expected.items()
    }


def layer1_owner_fixture(receipt: dict[str, object]) -> dict[str, object]:
    """Export exact source records for the layer-one ratio-two publication."""
    if (
        receipt.get("capture_status")
        != "completed synthetic source-forward capture; no parity claim"
    ):
        raise RuntimeError(
            "layer-one owner fixture requires a completed source capture"
        )
    coverage = _object(receipt.get("coverage_status"), "capture coverage")
    source = _object(receipt.get("source"), "source")
    runtime = _object(receipt.get("runtime"), "runtime")
    model = _object(receipt.get("model_args"), "model args")
    encoded = _object(receipt.get("encoded_parameters"), "encoded parameters")
    steps = receipt.get("steps")
    if coverage.get("pending") != [] or not isinstance(steps, list):
        raise RuntimeError("layer-one owner fixture requires complete capture coverage")
    if runtime.get("storage_byteorder") != "little":
        raise RuntimeError("layer-one owner fixture requires little-endian storage")
    if any(not isinstance(source.get(field), str) for field in SOURCE_FIELDS):
        raise RuntimeError("layer-one owner fixture has incomplete source provenance")
    _validate_source(source)
    if (
        tuple(model.get("compress_ratios", ())) != (0, 2, 2, 1, 1)
        or tuple(model.get("kv_source_layers", ())) != (1, 3)
        or tuple(model.get("index_source_layers", ())) != (1, 3, 4)
        or model.get("candidate_source_layer") != 3
        or model.get("index_topk") != 1
        or model.get("head_dim") != 64
        or model.get("index_head_dim") != 64
        or model.get("rope_head_dim") != 32
    ):
        raise RuntimeError("layer-one owner fixture has unexpected reduced schedule")
    norm_eps = model.get("norm_eps")
    if (
        not isinstance(norm_eps, (int, float))
        or not math.isfinite(norm_eps)
        or norm_eps <= 0
    ):
        raise RuntimeError("layer-one owner fixture has invalid norm epsilon")
    parameters = _parameters(encoded)
    if len(steps) != len(STARTS):
        raise RuntimeError(
            "layer-one owner fixture requires the pinned three-call trace"
        )

    cases: list[dict[str, object]] = []
    frequency_table: dict[str, Any] | None = None
    for step, (start, sequence, compressed, offset) in zip(steps, STARTS, strict=True):
        item = _object(step, "capture step")
        if item.get("start_pos") != start:
            raise RuntimeError("layer-one owner fixture trace order changed")
        intermediates = _object(item.get("intermediates"), "step intermediates")
        indexer = _object(
            intermediates.get(f"{LAYER}.indexer_observation"), "layer-one indexer"
        )
        inputs = _object(indexer.get("inputs"), "layer-one index inputs")
        if set(inputs) != {
            "x",
            "qr",
            "latent",
            "start_pos",
            "offset",
            "frequency_table",
            "shared_index_k_prefix",
            "owner_key_prefix",
        }:
            raise RuntimeError("layer-one index input set implies candidate semantics")
        if inputs["start_pos"] != start or inputs["offset"] != offset:
            raise RuntimeError("layer-one index source call metadata changed")
        table = _tensor(
            inputs["frequency_table"],
            "layer-one source frequency table",
            dtype="torch.complex64",
            shape=[8, 16],
        )
        if frequency_table is None:
            frequency_table = table
        elif frequency_table["storage_sha256"] != table["storage_sha256"]:
            raise RuntimeError("layer-one source frequency table changed between calls")
        latent = inputs["latent"]
        published = start != 6
        if published:
            latent = _tensor(
                latent,
                "layer-one pre-RoPE latent",
                dtype="torch.bfloat16",
                shape=[1, 1 if start else 2, 64],
            )
            compressor_output = _tensor(
                intermediates.get(f"{LAYER}.compressor"),
                "layer-one compressor output",
                dtype="torch.bfloat16",
                shape=[1, 1 if start else 2, 64],
            )
            if latent["storage_sha256"] != compressor_output["storage_sha256"]:
                raise RuntimeError("layer-one compressor and Indexer latent disagree")
        elif latent is not None or intermediates.get(f"{LAYER}.compressor") is not None:
            raise RuntimeError(
                "partial layer-one ratio-two call unexpectedly published"
            )
        compressed_record = _object(
            intermediates.get(f"{LAYER}.compressed"), "layer-one compressed publication"
        )
        case = {
            "start_pos": start,
            "sequence": sequence,
            "compressed_prefix": compressed,
            "offset": offset,
            "group_frequency_positions": [0, 2]
            if start == 0
            else [4]
            if start == 5
            else [],
            "input": _tensor(
                intermediates.get("layers.1.attention_input"),
                "layer-one attention input",
                dtype="torch.bfloat16",
                shape=[1, sequence, 128],
            ),
            "wkv_projection": _tensor(
                intermediates.get(f"{LAYER}.compressor.wkv"),
                "layer-one compressor WKV projection",
                dtype="torch.float32",
                shape=[1, sequence, 64],
            ),
            "wgate_projection": _tensor(
                intermediates.get(f"{LAYER}.compressor.wgate"),
                "layer-one compressor gate projection",
                dtype="torch.float32",
                shape=[1, sequence, 64],
            ),
            "latent": latent,
            "index_input": _tensor(
                inputs["x"],
                "layer-one index input",
                dtype="torch.bfloat16",
                shape=[1, sequence, 128],
            ),
            "index_qr": _tensor(
                inputs["qr"],
                "layer-one index QR",
                dtype="torch.bfloat16",
                shape=[1, sequence, 32],
            ),
            "index_score_key_prefix": _tensor(
                inputs["shared_index_k_prefix"],
                "layer-one index score key prefix",
                dtype="torch.bfloat16",
                shape=[1, compressed, 64],
            ),
            "index_key_prefix": _tensor(
                inputs["owner_key_prefix"],
                "layer-one owned index key prefix",
                dtype="torch.bfloat16",
                shape=[1, compressed, 64],
            ),
            "index_operations": _operations(
                indexer.get("operations"),
                sequence,
                compressed,
                published_groups=1 if start == 5 else 2 if start == 0 else 0,
            ),
            "selected_indices": _tensor(
                indexer.get("output_indices"),
                "layer-one selected indices",
                dtype="torch.int32",
                shape=[1, sequence, 1],
            ),
            "compressed_kv_prefix": _tensor(
                compressed_record.get("borrowed_kv"),
                "layer-one compressed KV prefix",
                dtype="torch.bfloat16",
                shape=[1, compressed, 64],
            ),
            "compressed_indices": _tensor(
                compressed_record.get("indices"),
                "layer-one compressed indices",
                dtype="torch.int32",
                shape=[1, sequence, 1],
            ),
        }
        if (
            case["selected_indices"]["storage_sha256"]
            != case["compressed_indices"]["storage_sha256"]
        ):
            raise RuntimeError(
                "layer-one selected indices do not reach compressed attention"
            )
        cases.append(case)
    if frequency_table is None:
        raise RuntimeError("layer-one owner fixture has no frequency table")
    for field in ("index_key_prefix", "compressed_kv_prefix"):
        if cases[1][field]["storage_sha256"] != cases[2][field]["storage_sha256"]:
            raise RuntimeError(
                f"layer-one partial decode changed published {field.replace('_', ' ')}"
            )
    return {
        "schema_version": 1,
        "scope": "source layer-one ratio-two owner publication; partial decode retains owner key/KV prefixes while source scoring can read a later layer's global shared key prefix; excludes candidate masks, layer-two attention, and native execution",
        "source": {**source, "complete_capture_sha256": _sha256(_serialized(receipt))},
        "model": {
            "owner_layer": 1,
            "ratio": 2,
            "index_topk": 1,
            "window_size": model["window_size"],
            "norm_eps": norm_eps,
            "candidate_source_layer": model["candidate_source_layer"],
        },
        "frequency_table": frequency_table,
        "encoded_parameters": parameters,
        "cases": cases,
    }


def _serialized(receipt: dict[str, object]) -> bytes:
    return (
        json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.input.is_file() or args.input.stat().st_size > MAX_INPUT_BYTES:
        raise RuntimeError("layer-one owner input must be a bounded regular file")
    receipt = json.loads(
        args.input.read_text(),
        parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError(f"layer-one owner fixture rejects non-finite JSON {value}")
        ),
    )
    fixture = layer1_owner_fixture(receipt)
    encoded = (
        json.dumps(fixture, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()
    if len(encoded) > MAX_FIXTURE_BYTES:
        raise RuntimeError("layer-one owner fixture exceeds its bounded size")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(encoded)
    print(
        json.dumps(
            {
                "artifact_sha256": _sha256(encoded),
                "bytes": len(encoded),
                "path": str(args.output),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
