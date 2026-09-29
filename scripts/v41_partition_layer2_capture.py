# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Project alternate-partition L2 HC, attention, and FFN-tail operands."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from v41_layer2_attention_capture import layer_two_attention_projection
from v41_layer2_ffn_capture import layer2_ffn_projection
from v41_layer2_hc_capture import layer2_hc_projection
from v41_partition_layer1_capture import partition_layer1_fixture
from v41_partition_owner_capture import CaptureError

SCHEDULE = ((0, 4), (4, 1), (5, 1), (6, 1))
MAX_FIXTURE_BYTES = 768 * 1024


def _same_provenance(root: dict[str, Any], child: dict[str, Any], label: str) -> None:
    for name in ("source", "source_receipt_sha256", "capture_identity"):
        if root.get(name) != child.get(name):
            raise CaptureError(f"alternate layer-two {label} {name} differs")


def _require_provenance(projection: dict[str, Any], label: str) -> None:
    if not isinstance(projection.get("source"), dict) or not projection["source"]:
        raise CaptureError(f"{label} lacks source provenance")
    receipt = projection.get("source_receipt_sha256")
    if not isinstance(receipt, str) or len(receipt) != 64:
        raise CaptureError(f"{label} has invalid receipt digest")
    if not isinstance(projection.get("capture_identity"), dict):
        raise CaptureError(f"{label} lacks capture identity")


def validate_layer2_join(layer_one: dict[str, Any], fixture: dict[str, Any]) -> None:
    """Validate alternate L1-to-L2 and L2-to-Engram boundary records."""
    if not isinstance(layer_one, dict):
        raise CaptureError("alternate layer-two requires a layer-one projection")
    _require_provenance(layer_one, "alternate layer-one projection")
    _require_provenance(fixture, "alternate layer-two fixture")
    for name in ("source", "source_receipt_sha256", "capture_identity"):
        if fixture.get(name) != layer_one.get(name):
            raise CaptureError(f"alternate layer-one and layer-two {name} differs")
    hc = fixture.get("hc")
    attention = fixture.get("attention")
    ffn = fixture.get("ffn")
    if not all(isinstance(value, dict) for value in (hc, attention, ffn)):
        raise CaptureError("alternate layer-two fixture lacks numerical projections")
    assert (
        isinstance(hc, dict) and isinstance(attention, dict) and isinstance(ffn, dict)
    )
    for label, child in (("HC", hc), ("attention", attention), ("FFN", ffn)):
        _same_provenance(fixture, child, label)
    if attention.get("runtime") != {"storage_byteorder": "little"}:
        raise CaptureError(
            "alternate layer-two attention requires little-endian storage"
        )

    l1_tail = layer_one.get("tail")
    l1_cases = layer_one.get("cases")
    hc_cases = hc.get("cases")
    attention_cases = attention.get("cases")
    ffn_cases = ffn.get("cases")
    if not all(
        isinstance(value, list)
        for value in (
            l1_cases,
            l1_tail.get("cases") if isinstance(l1_tail, dict) else None,
            hc_cases,
            attention_cases,
            ffn_cases,
        )
    ):
        raise CaptureError("alternate layer-two fixture lacks joined case arrays")
    assert isinstance(l1_tail, dict)
    tail_cases = l1_tail["cases"]
    assert isinstance(l1_cases, list) and isinstance(tail_cases, list)
    assert (
        isinstance(hc_cases, list)
        and isinstance(attention_cases, list)
        and isinstance(ffn_cases, list)
    )
    if any(
        len(cases) != len(SCHEDULE)
        for cases in (l1_cases, tail_cases, hc_cases, attention_cases, ffn_cases)
    ):
        raise CaptureError(
            "alternate layer-one/layer-two projections require four cases"
        )

    for owner, tail, hc_case, attention_case, ffn_case, (start, sequence) in zip(
        l1_cases,
        tail_cases,
        hc_cases,
        attention_cases,
        ffn_cases,
        SCHEDULE,
        strict=True,
    ):
        if not all(
            isinstance(case, dict)
            for case in (owner, tail, hc_case, attention_case, ffn_case)
        ):
            raise CaptureError("alternate layer-one/layer-two cases must be objects")
        if any(
            case.get("start_pos") != start
            for case in (owner, tail, hc_case, attention_case, ffn_case)
        ):
            raise CaptureError("alternate layer-one/layer-two schedules differ")
        if tail.get("output") != hc_case.get("residual") or tail.get(
            "next_pre"
        ) != hc_case.get("incoming_pre"):
            raise CaptureError("layer-one tail does not feed layer-two HC")
        if attention_case.get("input") != hc_case.get("attention_input"):
            raise CaptureError("layer-two HC does not feed attention input")
        if attention_case.get("output") != hc_case.get(
            "attention_output"
        ) or attention_case.get("after_attention_residual") != hc_case.get(
            "after_attention_residual"
        ):
            raise CaptureError("layer-two attention does not feed HC output")
        if attention_case.get("layer_one_published_kv") != owner.get(
            "compressed_kv_prefix"
        ) or attention_case.get("layer_one_published_indices") != owner.get(
            "selected_indices"
        ):
            raise CaptureError(
                "layer-two attention does not borrow live layer-one publication"
            )
        if attention_case.get("compressed_kv") != owner.get(
            "compressed_kv_prefix"
        ) or attention_case.get("compressed_indices") != owner.get("selected_indices"):
            raise CaptureError(
                "layer-two attention compressed operands differ from layer one"
            )
        if ffn_case.get("after_attention_residual") != hc_case.get(
            "after_attention_residual"
        ) or ffn_case.get("attention_pre") != hc_case.get("attention_pre"):
            raise CaptureError("layer-two HC does not feed FFN tail")
        if ffn_case.get("output") != ffn_case.get("engram_stream") or ffn_case.get(
            "next_pre"
        ) != ffn_case.get("layer_three_incoming_pre"):
            raise CaptureError("layer-two FFN does not feed observed layer-three entry")
        if ffn_case.get("output", {}).get("shape") != [1, sequence, 2, 128]:
            raise CaptureError("layer-two FFN output shape differs")


def partition_layer2_fixture(raw: bytes) -> dict[str, Any]:
    """Project L2 alternate operands without mapping their receipt to canonical metadata."""
    layer_one = partition_layer1_fixture(raw)
    try:
        receipt = json.loads(raw)
    except json.JSONDecodeError as error:
        raise CaptureError("alternate layer-two receipt is not valid JSON") from error
    alternate = receipt.get("alternate")
    capture = (
        alternate.get("alternate_capture") if isinstance(alternate, dict) else None
    )
    if not isinstance(capture, dict):
        raise CaptureError("alternate layer-two receipt lacks capture operands")
    model = capture.get("actual_model_args")
    encoded = capture.get("encoded_parameters")
    calls = capture.get("calls")
    if (
        not isinstance(model, dict)
        or not isinstance(encoded, dict)
        or not isinstance(calls, list)
    ):
        raise CaptureError(
            "alternate layer-two receipt has malformed numerical operands"
        )
    if len(calls) != len(SCHEDULE):
        raise CaptureError("alternate layer-two receipt has unexpected call count")
    hc = layer2_hc_projection(model, encoded, calls, schedule=SCHEDULE)
    attention = layer_two_attention_projection(model, encoded, calls, schedule=SCHEDULE)
    ffn = layer2_ffn_projection(model, encoded, calls, schedule=SCHEDULE)
    for child in (hc, attention, ffn):
        child.update(
            {
                "source": layer_one["source"],
                "source_receipt_sha256": layer_one["source_receipt_sha256"],
                "capture_identity": layer_one["capture_identity"],
            }
        )
    runtime = receipt.get("runtime")
    if not isinstance(runtime, dict) or runtime.get("storage_byteorder") != "little":
        raise CaptureError("alternate layer-two receipt requires little-endian storage")
    attention["runtime"] = {"storage_byteorder": runtime["storage_byteorder"]}
    fixture = {
        "schema_version": 1,
        "scope": (
            "source alternate layer-two HC, attention, and FFN-tail operands for the "
            "observed 4/1/1/1 partition; not canonical capture metadata or native execution"
        ),
        "source": layer_one["source"],
        "source_receipt_sha256": layer_one["source_receipt_sha256"],
        "capture_identity": layer_one["capture_identity"],
        "hc": hc,
        "attention": attention,
        "ffn": ffn,
    }
    validate_layer2_join(layer_one, fixture)
    return fixture


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    fixture = partition_layer2_fixture(args.input.read_bytes())
    output = (
        json.dumps(fixture, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode()
    if len(output) > MAX_FIXTURE_BYTES:
        raise CaptureError("alternate layer-two projection exceeds 768 KiB")
    with args.output.open("xb") as stream:
        stream.write(output)
    print(
        json.dumps({"bytes": len(output), "sha256": hashlib.sha256(output).hexdigest()})
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
