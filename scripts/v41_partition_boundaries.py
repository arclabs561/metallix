#!/usr/bin/env python3
"""Validate captured bridge operands for the pinned 4/1/1/1 partition.

This is deliberately a receipt validator, not a source runner or a numerical
oracle.  It checks that the full captured tensors crossing the named layer
boundaries are well formed and byte-identical where the source hands one
operand to the next consumer.
"""

from __future__ import annotations

import hashlib
import json
import math
import struct
from collections.abc import Mapping
from typing import Any

_SCHEDULE = ((0, 4), (4, 1), (5, 1), (6, 1))
_BYTE_WIDTHS = {
    "torch.bfloat16": 2,
    "torch.float32": 4,
    "torch.int32": 4,
    "torch.bool": 1,
}


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")  # noqa: TRY004 -- malformed receipt
    return value


def _field(record: Mapping[str, Any], path: tuple[str, ...], label: str) -> object:
    value: object = record
    for name in path:
        value = _mapping(value, label).get(name)
        if value is None:
            raise ValueError(f"{label} lacks {'.'.join(path)}")
    return value


def _tensor(value: object, label: str) -> tuple[Mapping[str, Any], bytes]:
    record = _mapping(value, label)
    dtype = record.get("dtype")
    shape = record.get("shape")
    numel = record.get("numel")
    storage_hex = record.get("storage_hex")
    digest = record.get("storage_sha256")
    if (
        dtype not in _BYTE_WIDTHS
        or not isinstance(shape, list)
        or not all(type(width) is int and width >= 0 for width in shape)
        or type(numel) is not int
        or not isinstance(storage_hex, str)
        or not isinstance(digest, str)
        or record.get("finite") is not True
    ):
        raise ValueError(f"{label} has malformed tensor metadata")
    if math.prod(shape) != numel:
        raise ValueError(f"{label} tensor shape does not match numel")
    try:
        raw = bytes.fromhex(storage_hex)
    except ValueError as error:
        raise ValueError(f"{label} has invalid storage hex") from error
    if len(raw) != numel * _BYTE_WIDTHS[dtype]:
        raise ValueError(f"{label} storage length does not match tensor metadata")
    if hashlib.sha256(raw).hexdigest() != digest:
        raise ValueError(f"{label} storage hash does not match storage hex")
    if dtype == "torch.bfloat16" and any(
        (int.from_bytes(raw[index : index + 2], "little") >> 7) & 0xFF == 0xFF
        for index in range(0, len(raw), 2)
    ):
        raise ValueError(f"{label} contains nonfinite bfloat16 storage")
    if dtype == "torch.float32" and not all(
        math.isfinite(number) for number in struct.unpack(f"<{numel}f", raw)
    ):
        raise ValueError(f"{label} contains nonfinite float32 storage")
    if dtype == "torch.bool" and any(byte not in (0, 1) for byte in raw):
        raise ValueError(f"{label} contains invalid bool storage")
    return record, raw


def _identity(record: Mapping[str, Any]) -> dict[str, object]:
    return {
        "dtype": record["dtype"],
        "shape": list(record["shape"]),
        "numel": record["numel"],
        "storage_sha256": record["storage_sha256"],
    }


def _same_tensor(left: object, right: object, label: str) -> dict[str, object]:
    left_record, left_raw = _tensor(left, f"{label} producer")
    right_record, right_raw = _tensor(right, f"{label} consumer")
    if (
        left_record["dtype"] != right_record["dtype"]
        or left_record["shape"] != right_record["shape"]
        or left_record["numel"] != right_record["numel"]
        or left_raw != right_raw
    ):
        raise ValueError(f"{label} source bridge is not byte-identical")
    return _identity(left_record)


def _call_intermediates(call: Mapping[str, Any], start_pos: int) -> Mapping[str, Any]:
    return _mapping(call.get("intermediates"), f"call {start_pos} intermediates")


def _bridge(
    intermediates: Mapping[str, Any], layer: int, name: str
) -> Mapping[str, Any]:
    return _mapping(
        intermediates.get(f"layers.{layer}.attn.{name}"),
        f"layers.{layer}.attn.{name}",
    )


def _validate_prior_layer_three_to_layer_one(
    predecessor: Mapping[str, Any], current: Mapping[str, Any]
) -> dict[str, object]:
    predecessor_cache = _mapping(
        predecessor.get("caches_after"), "predecessor caches_after"
    )
    cache_name = "layer_3.index_k"
    cache, cache_raw = _tensor(
        predecessor_cache.get(cache_name), f"predecessor {cache_name}"
    )
    if (
        cache["dtype"] != "torch.bfloat16"
        or cache["shape"][:1] != [1]
        or cache["shape"][2:] != [64]
        or cache["shape"][1] < 6
    ):
        raise ValueError("predecessor index-key cache cannot supply active six keys")

    current_intermediates = _call_intermediates(current, 6)
    owner_inputs = _mapping(
        _field(
            _bridge(current_intermediates, 1, "indexer_observation"),
            ("inputs",),
            "layer-one indexer observation",
        ),
        "layer-one indexer inputs",
    )
    consumed, consumed_raw = _tensor(
        owner_inputs.get("shared_index_k_prefix"),
        "layer-one consumed shared index-key prefix",
    )
    if consumed["dtype"] != "torch.bfloat16" or consumed["shape"] != [1, 3, 64]:
        raise ValueError("layer-one must consume exactly the leading three index keys")
    key_width = _BYTE_WIDTHS["torch.bfloat16"]
    active_raw = cache_raw[: 6 * 64 * key_width]
    consumed_raw_expected = active_raw[: 3 * 64 * key_width]
    if consumed_raw != consumed_raw_expected:
        raise ValueError(
            "layer-one consumed prefix does not match predecessor layer-three index keys"
        )
    return {
        "predecessor_start_pos": 5,
        "predecessor_end_pos": 6,
        "cache_field": cache_name,
        "producer_active_keys": 6,
        "producer_active_storage_sha256": hashlib.sha256(active_raw).hexdigest(),
        "consumed_prefix_keys": 3,
        "consumed": _identity(consumed),
    }


def validate_bridges(capture: object) -> dict[str, object]:
    """Validate only the source operands in the named 4/1/1/1 experiment."""
    root = _mapping(capture, "capture")
    calls = root.get("calls")
    if not isinstance(calls, list) or len(calls) != len(_SCHEDULE):
        raise ValueError("capture must retain exactly four named partition calls")

    normalized: list[Mapping[str, Any]] = []
    for call, (start_pos, token_count) in zip(calls, _SCHEDULE, strict=True):
        item = _mapping(call, f"call {start_pos}")
        observed_start = item.get("start_pos")
        observed_count = item.get("token_count")
        if (
            type(observed_start) is not int
            or type(observed_count) is not int
            or observed_start != start_pos
            or observed_count != token_count
        ):
            raise ValueError("capture calls must be the contiguous 0/4/5/6 schedule")
        normalized.append(item)

    layer_one_to_two: list[dict[str, object]] = []
    layer_three_to_four: list[dict[str, object]] = []
    for call, (start_pos, _) in zip(normalized, _SCHEDULE, strict=True):
        intermediates = _call_intermediates(call, start_pos)
        layer_one_compressed = _bridge(intermediates, 1, "compressed")
        layer_two_compressed = _bridge(intermediates, 2, "compressed")
        layer_one_to_two.append(
            {
                "start_pos": start_pos,
                "borrowed_kv": _same_tensor(
                    layer_one_compressed.get("borrowed_kv"),
                    layer_two_compressed.get("borrowed_kv"),
                    f"call {start_pos} layer-one to layer-two borrowed KV",
                ),
                "indices": _same_tensor(
                    layer_one_compressed.get("indices"),
                    layer_two_compressed.get("indices"),
                    f"call {start_pos} layer-one to layer-two indices",
                ),
            }
        )

        layer_three_compressed = _bridge(intermediates, 3, "compressed")
        layer_four_compressed = _bridge(intermediates, 4, "compressed")
        layer_three_indexer = _bridge(intermediates, 3, "indexer_observation")
        layer_four_indexer = _bridge(intermediates, 4, "indexer_observation")
        layer_four_inputs = _mapping(
            _field(layer_four_indexer, ("inputs",), "layer-four indexer observation"),
            "layer-four indexer inputs",
        )
        layer_three_inputs = _mapping(
            _field(layer_three_indexer, ("inputs",), "layer-three indexer observation"),
            "layer-three indexer inputs",
        )
        layer_three_to_four.append(
            {
                "start_pos": start_pos,
                "borrowed_kv": _same_tensor(
                    layer_three_compressed.get("borrowed_kv"),
                    layer_four_compressed.get("borrowed_kv"),
                    f"call {start_pos} layer-three to layer-four borrowed KV",
                ),
                "candidate_mask": _same_tensor(
                    layer_three_indexer.get("candidate_mask_after"),
                    layer_four_inputs.get("candidate_mask"),
                    f"call {start_pos} layer-three candidate mask to layer-four",
                ),
                "shared_index_k_prefix": _same_tensor(
                    layer_three_inputs.get("shared_index_k_prefix"),
                    layer_four_inputs.get("shared_index_k_prefix"),
                    f"call {start_pos} layer-three to layer-four shared index keys",
                ),
            }
        )

    return {
        "call_count": len(normalized),
        "layer1_to_layer2": layer_one_to_two,
        "layer3_to_layer4": layer_three_to_four,
        "prior_layer3_to_layer1": _validate_prior_layer_three_to_layer_one(
            normalized[2], normalized[3]
        ),
    }


def project_bridges(raw_receipt: bytes) -> dict[str, object]:
    """Select observed operands verbatim from a successfully controlled capture."""
    root = _mapping(json.loads(raw_receipt), "partition receipt")
    if root.get("status") != "completed_source_partition_experiment":
        raise ValueError("partition experiment did not complete")
    control = _mapping(root.get("observer_noninterference"), "observer control")
    checks = control.get("per_call")
    if (
        control.get("exact_noninterference") is not True
        or not isinstance(checks, list)
        or len(checks) != 4
        or any(
            _mapping(c, "observer call").get("exact_noninterference") is not True
            for c in checks
        )
    ):
        raise ValueError("capture requires four successful observer controls")
    alternate = _mapping(root.get("alternate"), "alternate run")
    capture = _mapping(alternate.get("alternate_capture"), "alternate capture")
    validate_bridges(capture)
    calls = []
    for call in capture["calls"]:
        observed = call["intermediates"]
        selected = {
            f"layers.{layer}.attn.compressed": observed[
                f"layers.{layer}.attn.compressed"
            ]
            for layer in (1, 2, 3, 4)
        }
        for layer in (1, 3, 4):
            name = f"layers.{layer}.attn.indexer_observation"
            indexer = observed[name]
            selected[name] = {
                "inputs": {
                    key: indexer["inputs"][key]
                    for key in ("shared_index_k_prefix", "candidate_mask")
                    if key in indexer["inputs"]
                },
                **(
                    {"candidate_mask_after": indexer["candidate_mask_after"]}
                    if "candidate_mask_after" in indexer
                    else {}
                ),
            }
        calls.append(
            {
                "start_pos": call["start_pos"],
                "token_count": call["token_count"],
                "intermediates": selected,
                "caches_after": {
                    "layer_3.index_k": call["caches_after"]["layer_3.index_k"]
                },
            }
        )
    result = {
        "schema_version": 1,
        "scope": "Source bridge observations for 4/1/1/1; not native qualification",
        "source_receipt_sha256": hashlib.sha256(raw_receipt).hexdigest(),
        "capture_identity": capture["capture_identity"],
        "source": root["source"],
        "baseline_oracle": root["baseline_oracle"],
        "calls": calls,
    }
    validate_bridges(result)
    return result


def main() -> None:
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw = args.input.read_bytes()
    projection = project_bridges(raw)
    with args.output.open("x") as output:
        json.dump(projection, output, sort_keys=True, separators=(",", ":"))
        output.write("\n")


if __name__ == "__main__":
    main()
