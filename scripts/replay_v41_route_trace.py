#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Replay a source-compatible V4.1 route trace against an explicit LRU budget.

This is offline accounting. It neither reads checkpoint payloads nor estimates
SSD latency, model throughput, or a resource budget that the operator has not
supplied.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import OrderedDict
from pathlib import Path
from typing import Any

MAX_TRACE_BYTES = 16 << 20
EXPECTED_GEOMETRY = {
    "hidden_width": 5120,
    "backbone_layers": 40,
    "routed_experts": 384,
    "selected_experts": 6,
}
PINNED_SOURCE = {
    "revision": "dba1be0a40aa45a94ad051997016db3960a90277",
    "config_sha256": "8be45ce0476004a3f529fd896115a4a2e800a129ad2d3ec05b16050f52e21879",
    "model_sha256": "4e9ae23620edc8028ccc5d5fef552ab7fdc7dcd6f79608754fe9f67644056f65",
}
VALID_PHASES = frozenset(("prefill", "decode"))


class TraceError(ValueError):
    """A route trace does not meet the offline replay contract."""


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be an integer") from error
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _nonnegative_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be an integer") from error
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return parsed


def _read_trace(path: Path) -> tuple[dict[str, Any], str]:
    if not path.is_file():
        raise TraceError("route trace must be an existing bounded regular file")
    try:
        with path.open("rb") as source:
            encoded = source.read(MAX_TRACE_BYTES + 1)
    except OSError as error:
        raise TraceError("route trace could not be read") from error
    if len(encoded) > MAX_TRACE_BYTES:
        raise TraceError("route trace exceeds the byte limit")
    try:
        document = json.loads(encoded.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise TraceError("route trace must be valid JSON") from error
    if not isinstance(document, dict):
        raise TraceError("route trace must be a JSON object")
    return document, hashlib.sha256(encoded).hexdigest()


def validate_trace(document: dict[str, Any]) -> list[dict[str, Any]]:
    """Validate the source-compatible route-only trace schema and return rows."""
    if (
        type(document.get("schema_version")) is not int
        or document["schema_version"] != 1
    ):
        raise TraceError("route trace schema_version must be 1")
    if "synthetic_geometry" in document:
        raise TraceError("synthetic route traces cannot establish V4.1 locality")
    source = document.get("source")
    geometry = document.get("geometry")
    rows = document.get("rows")
    if (
        not isinstance(source, dict)
        or not isinstance(geometry, dict)
        or not isinstance(rows, list)
    ):
        raise TraceError("route trace requires source, geometry, and rows")
    if any(source.get(name) != value for name, value in PINNED_SOURCE.items()):
        raise TraceError("route trace source is not the pinned V4.1 source")
    if any(
        type(geometry.get(name)) is not int or geometry[name] != value
        for name, value in EXPECTED_GEOMETRY.items()
    ):
        raise TraceError("route trace geometry is not source-compatible V4.1")
    if not rows:
        raise TraceError("route trace requires at least one token group")

    groups: dict[tuple[str, str, int], list[dict[str, Any]]] = {}
    group_order: list[tuple[str, str, int]] = []
    for row in rows:
        if not isinstance(row, dict):
            raise TraceError("route trace row must be an object")
        request_id = row.get("request_id")
        phase = row.get("phase")
        token_position = row.get("token_position")
        layer = row.get("layer")
        expert_ids = row.get("expert_ids")
        if (
            not isinstance(request_id, str)
            or not request_id
            or not isinstance(phase, str)
            or phase not in VALID_PHASES
            or type(token_position) is not int
            or token_position < 0
            or type(layer) is not int
            or not 0 <= layer < EXPECTED_GEOMETRY["backbone_layers"]
            or not isinstance(expert_ids, list)
            or len(expert_ids) != EXPECTED_GEOMETRY["selected_experts"]
            or not all(type(expert) is int for expert in expert_ids)
            or not all(
                0 <= expert < EXPECTED_GEOMETRY["routed_experts"]
                for expert in expert_ids
            )
            or len(set(expert_ids)) != len(expert_ids)
        ):
            raise TraceError("route trace row has invalid route fields")
        key = (request_id, phase, token_position)
        if key not in groups:
            groups[key] = []
            group_order.append(key)
        groups[key].append(row)

    expected_layers = list(range(EXPECTED_GEOMETRY["backbone_layers"]))
    for key in group_order:
        group = groups[key]
        if [row["layer"] for row in group] != expected_layers:
            raise TraceError(
                "each token group must contain layers 0 through 39 in order"
            )
    # `rows` is already chronological router-event order.  Do not regroup it:
    # prefill is commonly layer-major, and changing event order changes LRU hits.
    return rows


def _phase_report() -> dict[str, int]:
    return {"route_rows": 0, "selections": 0, "hits": 0, "misses": 0}


POLICIES = ("lru", "belady", "frequency")
POLICY_DESCRIPTIONS = {
    "lru": "exact layer/expert LRU; initially empty; no prefetch",
    "belady": (
        "offline optimal eviction (evict the resident expert reused furthest in "
        "the future); an upper bound no online policy can exceed; initially empty"
    ),
    "frequency": (
        "static residency of the most frequently routed layer/experts counted on "
        "calibration requests only; misses are read and not retained"
    ),
}


def _keys(rows: list[dict[str, Any]]) -> list[tuple[int, int]]:
    return [(row["layer"], expert) for row in rows for expert in row["expert_ids"]]


def _belady_hits(rows: list[dict[str, Any]], capacity: int) -> tuple[list[bool], int]:
    """Exact Belady/MIN replay; returns the per-selection hit flags and evictions."""
    keys = _keys(rows)
    upcoming: dict[tuple[int, int], list[int]] = {}
    for index in range(len(keys) - 1, -1, -1):
        upcoming.setdefault(keys[index], []).append(index)
    resident: set[tuple[int, int]] = set()
    flags: list[bool] = []
    evictions = 0
    never = len(keys)
    for index, key in enumerate(keys):
        upcoming[key].pop()
        if key in resident:
            flags.append(True)
            continue
        flags.append(False)
        if capacity == 0:
            continue
        if len(resident) == capacity:
            victim = max(
                resident,
                key=lambda item: upcoming[item][-1] if upcoming[item] else never,
            )
            resident.remove(victim)
            evictions += 1
        resident.add(key)
    return flags, evictions


def replay(
    rows: list[dict[str, Any]],
    *,
    expert_cache_bytes: int,
    expert_bytes: int,
    policy: str = "lru",
    calibration_rows: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Replay exact layer/expert selections under one residency policy."""
    if not isinstance(rows, list) or not rows:
        raise TraceError("replay requires nonempty validated route rows")
    if type(expert_cache_bytes) is not int or expert_cache_bytes < 0:
        raise TraceError("expert cache bytes must be a nonnegative integer")
    if type(expert_bytes) is not int or expert_bytes <= 0:
        raise TraceError("expert bytes must be a positive integer")
    if policy not in POLICIES:
        raise TraceError(f"policy must be one of {', '.join(POLICIES)}")
    if (policy == "frequency") != (calibration_rows is not None):
        raise TraceError("the frequency policy requires separate calibration rows")
    capacity_items = expert_cache_bytes // expert_bytes
    evictions = 0
    resident_at_end: int
    if policy == "lru":
        cache: OrderedDict[tuple[int, int], None] = OrderedDict()
        flags = []
        for key in _keys(rows):
            if key in cache:
                flags.append(True)
                cache.move_to_end(key)
                continue
            flags.append(False)
            if capacity_items == 0:
                continue
            if len(cache) == capacity_items:
                cache.popitem(last=False)
                evictions += 1
            cache[key] = None
        resident_at_end = len(cache)
    elif policy == "belady":
        flags, evictions = _belady_hits(rows, capacity_items)
        resident_at_end = min(capacity_items, len(set(_keys(rows))))
    else:
        assert calibration_rows is not None
        counts: dict[tuple[int, int], int] = {}
        for key in _keys(calibration_rows):
            counts[key] = counts.get(key, 0) + 1
        # Deterministic ties: higher count first, then layer, then expert id.
        ranked = sorted(counts, key=lambda key: (-counts[key], key))
        static = set(ranked[:capacity_items])
        flags = [key in static for key in _keys(rows)]
        resident_at_end = len(static)
    phases = {phase: _phase_report() for phase in sorted(VALID_PHASES)}
    position = 0
    for row in rows:
        phase_report = phases[row["phase"]]
        phase_report["route_rows"] += 1
        for _ in row["expert_ids"]:
            phase_report["selections"] += 1
            phase_report["hits" if flags[position] else "misses"] += 1
            position += 1
    hits = sum(flags)
    misses = len(flags) - hits
    selections = hits + misses
    return {
        "policy": POLICY_DESCRIPTIONS[policy],
        "supplied_parameters": {
            "expert_cache_bytes": expert_cache_bytes,
            "expert_bytes": expert_bytes,
            "capacity_experts": capacity_items,
            "unused_cache_bytes": expert_cache_bytes % expert_bytes,
        },
        "observed": {
            "route_rows": len(rows),
            "selections": selections,
            "unique_layer_experts": len(
                {(row["layer"], expert) for row in rows for expert in row["expert_ids"]}
            ),
            "hits": hits,
            "misses": misses,
            "hit_fraction": hits / selections,
            "miss_fraction": misses / selections,
            "miss_useful_bytes": misses * expert_bytes,
            "evictions": evictions,
            "resident_layer_experts_at_end": resident_at_end,
            "by_phase": phases,
        },
        "limitations": [
            "does not measure physical reads, range coalescing, SSD latency, compute, or token throughput",
            "does not include Engram, dense weights, mutable state, staging, allocator retention, or OS headroom",
            "a result is conditional on this trace and supplied cache/expert byte parameters",
            "rows are replayed in supplied chronological router-event order without regrouping",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--expert-cache-bytes", type=_nonnegative_int, required=True)
    parser.add_argument("--expert-bytes", type=_positive_int, required=True)
    parser.add_argument("--policy", choices=POLICIES, default="lru")
    parser.add_argument(
        "--calibration-request",
        action="append",
        default=[],
        help="request id used only to fit the frequency policy; excluded from replay",
    )
    parser.add_argument("--output", type=Path, required=True, help="new JSON report")
    args = parser.parse_args()
    try:
        output = args.output.resolve()
        if output == args.trace.resolve():
            raise TraceError("report output must not overwrite the route trace")
        document, trace_sha256 = _read_trace(args.trace)
        rows = validate_trace(document)
        calibration = set(args.calibration_request)
        calibration_rows = [row for row in rows if row["request_id"] in calibration]
        if calibration and len({row["request_id"] for row in calibration_rows}) != len(
            calibration
        ):
            raise TraceError("every calibration request must appear in the trace")
        held_out = [row for row in rows if row["request_id"] not in calibration]
        if not held_out:
            raise TraceError("calibration requests leave no held-out rows to replay")
        report = {
            "schema_version": 1,
            "scope": "offline source-compatible V4.1 route-trace LRU accounting; not a checkpoint read or serving measurement",
            "trace": {
                "sha256": trace_sha256,
                "provenance": (
                    "trace-declared source identity matches current pins; this does not "
                    "independently prove that a router emitted the rows"
                ),
            },
            "trace_source": document["source"],
            "trace_geometry": document["geometry"],
            "calibration_requests": sorted(calibration),
            **replay(
                held_out,
                expert_cache_bytes=args.expert_cache_bytes,
                expert_bytes=args.expert_bytes,
                policy=args.policy,
                calibration_rows=calibration_rows
                if args.policy == "frequency"
                else None,
            ),
        }
        encoded = json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n"
        output.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        with os.fdopen(descriptor, "w", encoding="utf-8") as destination:
            destination.write(encoded)
    except (OSError, TraceError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
