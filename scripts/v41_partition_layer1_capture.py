#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Project alternate-partition layer-one owner operands without canonical metadata."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from v41_layer1_attention_capture import layer_one_attention_projection
from v41_layer1_owner_capture import _tensor, layer1_owner_projection
from v41_partition_owner_capture import CaptureError, partition_owner_fixture

SCHEDULE = ((0, 4, 2, 4, 2), (4, 1, 2, 6, 0), (5, 1, 3, 6, 1), (6, 1, 3, 6, 0))
MAX_FIXTURE_BYTES = 1 << 20


def _query_parameters(encoded: dict[str, Any]) -> dict[str, Any]:
    specs = {
        "layers.1.attn.wq_a.weight": ("torch.float8_e4m3fn", [32, 128]),
        "layers.1.attn.wq_a.scale": ("torch.float8_e8m0fnu", [1, 4]),
        "layers.1.attn.q_norm.weight": ("torch.bfloat16", [32]),
    }
    return {
        name: _tensor(encoded.get(name), name, dtype=dtype, shape=shape)
        for name, (dtype, shape) in specs.items()
    }


def validate_layer1_join(fixture: dict[str, Any]) -> None:
    """Validate retained alternate L1 owner/attention joins, not source capture proof."""
    attention = fixture.get("attention")
    if not isinstance(attention, dict):
        raise CaptureError("alternate layer-one fixture lacks attention projection")
    source = fixture.get("source")
    receipt = fixture.get("source_receipt_sha256")
    identity = fixture.get("capture_identity")
    if not isinstance(source, dict) or not source:
        raise CaptureError("alternate layer-one fixture lacks source provenance")
    if not isinstance(receipt, str) or len(receipt) != 64:
        raise CaptureError("alternate layer-one fixture has invalid receipt digest")
    if not isinstance(identity, dict):
        raise CaptureError("alternate layer-one fixture lacks capture identity")
    for name in ("source", "source_receipt_sha256", "capture_identity"):
        if fixture.get(name) != attention.get(name):
            raise CaptureError(f"alternate layer-one attention {name} differs")
    if attention.get("runtime") != {"storage_byteorder": "little"}:
        raise CaptureError(
            "alternate layer-one attention requires little-endian storage"
        )
    if identity.get("schedule") != [4, 1, 1, 1]:
        raise CaptureError(
            "alternate layer-one fixture has unexpected partition schedule"
        )
    owner_cases = fixture.get("cases")
    attention_cases = attention.get("cases")
    if not isinstance(owner_cases, list) or not isinstance(attention_cases, list):
        raise CaptureError("alternate layer-one cases must be arrays")
    if len(owner_cases) != len(SCHEDULE) or len(attention_cases) != len(SCHEDULE):
        raise CaptureError("alternate layer-one fixture requires four cases")
    latest_published: dict[str, Any] | None = None
    for owner_case, attention_case, (
        start,
        sequence,
        compressed,
        offset,
        published_groups,
    ) in zip(owner_cases, attention_cases, SCHEDULE, strict=True):
        if not isinstance(owner_case, dict) or not isinstance(attention_case, dict):
            raise CaptureError("alternate layer-one cases must be objects")
        if (
            owner_case.get("start_pos") != start
            or owner_case.get("sequence") != sequence
            or owner_case.get("compressed_prefix") != compressed
            or owner_case.get("offset") != offset
            or attention_case.get("start_pos") != start
        ):
            raise CaptureError(
                "alternate layer-one owner and attention schedules differ"
            )
        indexer = attention_case.get("indexer")
        if not isinstance(indexer, dict):
            raise CaptureError("alternate layer-one attention lacks indexer output")
        for left, right, label in (
            (attention_case.get("input"), owner_case.get("input"), "input"),
            (
                attention_case.get("compressed_kv"),
                owner_case.get("compressed_kv_prefix"),
                "compressed KV prefix",
            ),
            (
                attention_case.get("compressed_indices"),
                owner_case.get("selected_indices"),
                "selected IDs",
            ),
            (
                indexer.get("output_indices"),
                owner_case.get("selected_indices"),
                "indexer selected IDs",
            ),
        ):
            if left != right:
                raise CaptureError(
                    f"alternate layer-one owner-to-attention {label} differs"
                )
        if published_groups:
            if owner_case.get("latent") is None:
                raise CaptureError("layer-one publication lacks its latent group")
            if owner_case.get("index_score_key_prefix") != owner_case.get(
                "index_key_prefix"
            ):
                raise CaptureError(
                    "published layer-one score prefix differs from own keys"
                )
            latest_published = owner_case
        else:
            if owner_case.get("latent") is not None or latest_published is None:
                raise CaptureError(
                    "layer-one partial call has invalid publication state"
                )
            for field in ("index_key_prefix", "compressed_kv_prefix"):
                if owner_case.get(field) != latest_published.get(field):
                    raise CaptureError(
                        f"layer-one partial call changed retained {field.replace('_', ' ')}"
                    )
            if owner_case.get("index_score_key_prefix") == owner_case.get(
                "index_key_prefix"
            ):
                raise CaptureError(
                    "partial layer-one score prefix must not alias own keys"
                )


def partition_layer1_fixture(raw: bytes) -> dict[str, Any]:
    """Project the observed 4/1/1/1 L1 owner without rewriting its identity."""
    owner = partition_owner_fixture(raw)
    try:
        root = json.loads(raw)
    except json.JSONDecodeError as error:
        raise CaptureError("alternate owner receipt is not valid JSON") from error
    alternate = root.get("alternate")
    if not isinstance(alternate, dict):
        raise CaptureError("alternate owner receipt lacks alternate capture")
    capture = alternate.get("alternate_capture")
    if not isinstance(capture, dict):
        raise CaptureError("alternate owner receipt lacks alternate capture operands")
    model = capture.get("actual_model_args")
    encoded = capture.get("encoded_parameters")
    calls = capture.get("calls")
    if (
        not isinstance(model, dict)
        or not isinstance(encoded, dict)
        or not isinstance(calls, list)
    ):
        raise CaptureError("alternate owner receipt has malformed numerical operands")
    runtime = root.get("runtime")
    if not isinstance(runtime, dict) or runtime.get("storage_byteorder") != "little":
        raise CaptureError("alternate owner receipt requires little-endian storage")
    projection = layer1_owner_projection(model, encoded, calls, schedule=SCHEDULE)
    for source, case, (_, token_count, _, _, _) in zip(
        calls, projection["cases"], SCHEDULE, strict=True
    ):
        intermediates = source.get("intermediates")
        if not isinstance(intermediates, dict):
            raise CaptureError("alternate owner call lacks intermediates")
        case["wq_a_output"] = _tensor(
            intermediates.get("layers.1.attn.wq_a"),
            "alternate layer-one WQ-A output",
            dtype="torch.bfloat16",
            shape=[1, token_count, 32],
        )
    attention = layer_one_attention_projection(
        model,
        encoded,
        calls,
        schedule=tuple((start, count) for start, count, _, _, _ in SCHEDULE),
    )
    attention.update(
        {
            "source": owner["source"],
            "source_receipt_sha256": owner["source_receipt_sha256"],
            "capture_identity": owner["capture_identity"],
            "runtime": {"storage_byteorder": runtime["storage_byteorder"]},
        }
    )
    fixture = {
        **projection,
        "scope": (
            "source alternate layer-one ratio-two owner and query operands for the "
            "observed 4/1/1/1 partition; partial calls retain the prior published "
            "key and KV prefixes; not canonical capture metadata or native execution"
        ),
        "source": owner["source"],
        "source_receipt_sha256": owner["source_receipt_sha256"],
        "capture_identity": owner["capture_identity"],
        "query_parameters": _query_parameters(encoded),
        "attention": attention,
    }
    validate_layer1_join(fixture)
    return fixture


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    fixture = partition_layer1_fixture(args.input.read_bytes())
    output = (
        json.dumps(fixture, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode()
    if len(output) > MAX_FIXTURE_BYTES:
        raise CaptureError("alternate layer-one owner projection exceeds one MiB")
    with args.output.open("xb") as stream:
        stream.write(output)
    print(
        json.dumps({"bytes": len(output), "sha256": hashlib.sha256(output).hexdigest()})
    )


if __name__ == "__main__":
    main()
