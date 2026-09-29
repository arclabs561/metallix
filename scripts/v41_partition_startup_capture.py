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
from v41_partition_owner_capture import CaptureError, partition_owner_fixture

SCHEDULE = ((0, 4), (4, 1), (5, 1), (6, 1))


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
    return {
        **projection,
        "contract": {
            **projection["contract"],
            "layer_one_consumer": "observed layer-one Engram input; native alternate consumer remains unqualified",
        },
        "source": owner["source"],
        "source_receipt_sha256": owner["source_receipt_sha256"],
        "capture_identity": owner["capture_identity"],
    }


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
    if len(output) > 768 * 1024:
        raise CaptureError("startup projection exceeds 768 KiB cap")
    with args.output.open("xb") as stream:
        stream.write(output)
    print(
        json.dumps({"bytes": len(output), "sha256": hashlib.sha256(output).hexdigest()})
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
