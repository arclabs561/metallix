#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Project alternate-partition startup through L0 to the observed L1 stream."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path
from typing import Any

from v41_layer0_to_layer1_capture import layer_zero_projection
from v41_layer1_engram_capture import engram_projection
from v41_partition_owner_capture import CaptureError, partition_owner_fixture

SCHEDULE = ((0, 4), (4, 1), (5, 1), (6, 1))
MAX_FIXTURE_BYTES = 800 * 1024


def join_startup_calls(
    alternate: dict[str, Any], token_ids: list[int]
) -> list[dict[str, Any]]:
    """Join observed token records to capture calls using both call coordinates."""
    observed = alternate["calls"]
    captured = alternate["alternate_capture"]["calls"]
    if len(observed) != len(SCHEDULE) or len(captured) != len(SCHEDULE):
        raise CaptureError("startup requires four observed and captured calls")
    joined = []
    for observation, capture, (start, count) in zip(
        observed, captured, SCHEDULE, strict=True
    ):
        for record in (observation, capture):
            if any(
                type(record.get(key)) is not int or record[key] != expected
                for key, expected in (("start_pos", start), ("token_count", count))
            ):
                raise CaptureError("startup observed/captured schedule differs")
        ids = observation["input_ids"]
        raw = bytes.fromhex(ids["storage_hex"])
        if (
            ids.get("dtype") != "torch.int64"
            or ids.get("shape") != [1, count]
            or ids.get("numel") != count
            or ids.get("finite") is not True
            or len(raw) != count * 8
            or hashlib.sha256(raw).hexdigest() != ids.get("storage_sha256")
        ):
            raise CaptureError("startup observed token storage is invalid")
        actual = [value for (value,) in struct.iter_unpack("<q", raw)]
        if actual != token_ids[start : start + count]:
            raise CaptureError("startup observed tokens differ from experiment input")
        joined.append({**capture, "input_ids": ids})
    return joined


def validate_engram_join(
    projection: dict[str, Any], layer_one_engram: dict[str, Any]
) -> None:
    """Bind alternate L0 output, Engram operands, and observed L1 boundaries."""
    for name in ("source", "source_receipt_sha256", "capture_identity"):
        if projection.get(name) is None or projection.get(name) != layer_one_engram.get(
            name
        ):
            raise CaptureError(f"startup and Engram {name} provenance differs")
    layer_zero_cases = projection.get("cases")
    layer_one_cases = layer_one_engram.get("cases")
    if not isinstance(layer_zero_cases, list) or not isinstance(layer_one_cases, list):
        raise CaptureError("startup and Engram cases must be arrays")
    if len(layer_zero_cases) != len(SCHEDULE) or len(layer_one_cases) != len(SCHEDULE):
        raise CaptureError("startup and Engram require four joined cases")
    for layer_zero, layer_one, (start, _) in zip(
        layer_zero_cases, layer_one_cases, SCHEDULE, strict=True
    ):
        if not isinstance(layer_zero, dict) or not isinstance(layer_one, dict):
            raise CaptureError("startup and Engram cases must be objects")
        startup = layer_zero.get("startup")
        downstream = layer_zero.get("downstream")
        if (
            type(layer_zero.get("start_pos")) is not int
            or type(layer_one.get("start_pos")) is not int
            or layer_zero.get("start_pos") != start
            or layer_one.get("start_pos") != start
            or not isinstance(startup, dict)
            or not isinstance(downstream, dict)
            or not isinstance(startup.get("input_ids"), dict)
            or startup.get("input_ids") != layer_one.get("input_ids")
        ):
            raise CaptureError("startup and Engram cases do not share observed IDs")
        if not isinstance(layer_one.get("stream"), dict) or layer_one.get(
            "stream"
        ) != layer_zero.get("block_output"):
            raise CaptureError("Engram stream does not equal layer-zero block output")
        if not isinstance(layer_one.get("output"), dict) or layer_one.get(
            "output"
        ) != downstream.get("layer_one_engram_output"):
            raise CaptureError(
                "Engram output does not equal layer-zero downstream output"
            )


def partition_startup_fixture(raw: bytes) -> dict[str, Any]:
    # Reuse the complete alternate receipt's provenance and noninterference gates.
    owner = partition_owner_fixture(raw)
    root = json.loads(raw)
    alternate = root["alternate"]
    capture = alternate["alternate_capture"]
    if root["execution"]["input_ids"] != list(range(7)):
        raise CaptureError("startup experiment token sequence differs")
    steps = join_startup_calls(alternate, root["execution"]["input_ids"])
    projection = layer_zero_projection(
        capture["actual_model_args"],
        capture["encoded_parameters"],
        steps,
        schedule=SCHEDULE,
    )
    layer_one_engram = engram_projection(
        capture["engram"],
        capture["actual_model_args"],
        capture["encoded_parameters"],
        steps,
        schedule=SCHEDULE,
    )
    layer_one_engram.update(
        {
            "source": owner["source"],
            "source_receipt_sha256": owner["source_receipt_sha256"],
            "capture_identity": owner["capture_identity"],
        }
    )
    fixture = {
        **projection,
        "contract": {
            **projection["contract"],
            "layer_one_consumer": "source-pinned native Engram and HC/RMSNorm into observed layer-one attention input",
        },
        "source": owner["source"],
        "source_receipt_sha256": owner["source_receipt_sha256"],
        "capture_identity": owner["capture_identity"],
        "layer_one_engram": layer_one_engram,
    }
    validate_engram_join(fixture, layer_one_engram)
    return fixture


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    fixture = partition_startup_fixture(args.input.read_bytes())
    output = (
        json.dumps(fixture, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode()
    if len(output) > MAX_FIXTURE_BYTES:
        raise CaptureError("startup projection exceeds 800 KiB cap")
    with args.output.open("xb") as stream:
        stream.write(output)
    print(
        json.dumps({"bytes": len(output), "sha256": hashlib.sha256(output).hexdigest()})
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
