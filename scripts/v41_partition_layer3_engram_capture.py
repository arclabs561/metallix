# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Project alternate-partition L3 Engram operands and source handoffs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from v41_layer3_engram_capture import engram_projection
from v41_partition_layer2_capture import partition_layer2_fixture
from v41_partition_owner_capture import CaptureError

SCHEDULE = ((0, 4), (4, 1), (5, 1), (6, 1))
MAX_FIXTURE_BYTES = 768 * 1024


def join_alternate_calls(alternate: dict[str, Any]) -> list[dict[str, Any]]:
    """Join observed input IDs to captured L3 steps by the four call coordinates."""
    observed = alternate.get("calls")
    capture = alternate.get("alternate_capture")
    captured = capture.get("calls") if isinstance(capture, dict) else None
    if not isinstance(observed, list) or not isinstance(captured, list):
        raise CaptureError(
            "alternate L3 Engram receipt lacks observed and captured calls"
        )
    if len(observed) != len(SCHEDULE) or len(captured) != len(SCHEDULE):
        raise CaptureError(
            "alternate L3 Engram requires four observed and captured calls"
        )
    joined = []
    for observed_call, captured_call, (start, count) in zip(
        observed, captured, SCHEDULE, strict=True
    ):
        if not isinstance(observed_call, dict) or not isinstance(captured_call, dict):
            raise CaptureError("alternate L3 Engram calls must be objects")
        for call in (observed_call, captured_call):
            if (
                type(call.get("start_pos")) is not int
                or type(call.get("token_count")) is not int
                or call["start_pos"] != start
                or call["token_count"] != count
            ):
                raise CaptureError(
                    "alternate L3 Engram observed/captured schedule differs"
                )
        input_ids = observed_call.get("input_ids")
        if not isinstance(input_ids, dict):
            raise CaptureError("alternate L3 Engram observed call lacks input IDs")
        joined.append({**captured_call, "input_ids": input_ids})
    return joined


def _require_provenance(projection: dict[str, Any], label: str) -> None:
    if not isinstance(projection.get("source"), dict) or not projection["source"]:
        raise CaptureError(f"{label} lacks source provenance")
    receipt = projection.get("source_receipt_sha256")
    if not isinstance(receipt, str) or len(receipt) != 64:
        raise CaptureError(f"{label} has invalid receipt digest")
    if not isinstance(projection.get("capture_identity"), dict):
        raise CaptureError(f"{label} lacks capture identity")


def validate_layer3_engram_join(
    layer_two: dict[str, Any], fixture: dict[str, Any]
) -> None:
    """Bind retained L2 FFN entries to alternate L3 Engram source records."""
    if not isinstance(layer_two, dict):
        raise CaptureError("alternate L3 Engram requires a layer-two projection")
    _require_provenance(layer_two, "alternate layer-two projection")
    _require_provenance(fixture, "alternate L3 Engram fixture")
    for name in ("source", "source_receipt_sha256", "capture_identity"):
        if fixture.get(name) != layer_two.get(name):
            raise CaptureError(f"alternate L2/L3 Engram {name} differs")
    ffn = layer_two.get("ffn")
    cases = fixture.get("cases")
    ffn_cases = ffn.get("cases") if isinstance(ffn, dict) else None
    if not isinstance(cases, list) or not isinstance(ffn_cases, list):
        raise CaptureError("alternate L3 Engram fixture lacks joined cases")
    if len(cases) != len(SCHEDULE) or len(ffn_cases) != len(SCHEDULE):
        raise CaptureError("alternate L2/L3 Engram requires four cases")
    for ffn_case, case, (start, sequence) in zip(
        ffn_cases, cases, SCHEDULE, strict=True
    ):
        if not isinstance(ffn_case, dict) or not isinstance(case, dict):
            raise CaptureError("alternate L2/L3 Engram cases must be objects")
        if ffn_case.get("start_pos") != start or case.get("start_pos") != start:
            raise CaptureError("alternate L2/L3 Engram schedules differ")
        if ffn_case.get("output") != case.get("stream"):
            raise CaptureError("layer-two FFN output does not feed layer-three Engram")
        if case.get("output") != case.get("block_entry"):
            raise CaptureError("layer-three Engram output does not feed block entry")
        if case.get("stream", {}).get("shape") != [1, sequence, 2, 128]:
            raise CaptureError("alternate layer-three Engram stream shape differs")


def partition_layer3_engram_fixture(raw: bytes) -> dict[str, Any]:
    """Project L3 Engram operands without mapping their receipt to canonical metadata."""
    layer_two = partition_layer2_fixture(raw)
    try:
        receipt = json.loads(raw)
    except json.JSONDecodeError as error:
        raise CaptureError("alternate L3 Engram receipt is not valid JSON") from error
    alternate = receipt.get("alternate")
    capture = (
        alternate.get("alternate_capture") if isinstance(alternate, dict) else None
    )
    if not isinstance(capture, dict):
        raise CaptureError("alternate L3 Engram receipt lacks capture operands")
    model = capture.get("actual_model_args")
    encoded = capture.get("encoded_parameters")
    engram = capture.get("engram")
    if (
        not isinstance(model, dict)
        or not isinstance(encoded, dict)
        or not isinstance(engram, dict)
    ):
        raise CaptureError(
            "alternate L3 Engram receipt has malformed numerical operands"
        )
    projection = engram_projection(
        engram, model, encoded, join_alternate_calls(alternate), schedule=SCHEDULE
    )
    projection.update(
        {
            "source": layer_two["source"],
            "source_receipt_sha256": layer_two["source_receipt_sha256"],
            "capture_identity": layer_two["capture_identity"],
        }
    )
    validate_layer3_engram_join(layer_two, projection)
    return projection


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    fixture = partition_layer3_engram_fixture(args.input.read_bytes())
    output = (
        json.dumps(fixture, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode()
    if len(output) > MAX_FIXTURE_BYTES:
        raise CaptureError("alternate L3 Engram projection exceeds 768 KiB")
    with args.output.open("xb") as stream:
        stream.write(output)
    print(
        json.dumps({"bytes": len(output), "sha256": hashlib.sha256(output).hexdigest()})
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
