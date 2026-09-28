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
"""Bundle source projections for one reduced V4.1 native forward trace.

This exporter never runs model arithmetic.  It projects one already completed,
hash-gated source receipt through the existing narrow extractors, then proves
that every included projection names that exact receipt.  Consumers may use the
bundle to compose native joins without splicing separately observed traces.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any

import v41_attention_capture
import v41_candidate_capture
import v41_index_key_capture
import v41_layer0_to_layer1_capture
import v41_layer1_attention_capture
import v41_layer1_engram_capture
import v41_layer1_owner_capture
import v41_layer1_tail_capture
import v41_layer2_attention_capture
import v41_layer2_ffn_capture
import v41_layer2_hc_capture
import v41_layer3_attention_capture
import v41_layer3_engram_capture

SCRIPTS = Path(__file__).parent
MAX_INPUT_BYTES = 16 << 20
MAX_OUTPUT_BYTES = 12 << 20
_ROUTE_LAYERS = (3, 4)
_ROUTE_STARTS = (0, 5, 6)
_ROUTE_TOKEN_COUNTS = (5, 1, 1)
_SYNTHETIC_ROUTE_GEOMETRY = {
    "hidden_width": 128,
    "routed_experts": 4,
    "selected_experts": 2,
}
_SYNTHETIC_MOE_MODEL_GEOMETRY = {
    "dim": _SYNTHETIC_ROUTE_GEOMETRY["hidden_width"],
    "n_routed_experts": _SYNTHETIC_ROUTE_GEOMETRY["routed_experts"],
    "n_activated_experts": _SYNTHETIC_ROUTE_GEOMETRY["selected_experts"],
}


def _serialized(receipt: dict[str, object]) -> bytes:
    return (
        json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode()


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _runner_module() -> Any:
    path = SCRIPTS / "v41-forward-reference.py"
    spec = importlib.util.spec_from_file_location("v41_reduced_runner_source", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("reduced-runner exporter cannot load source fixture helpers")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _source_identity(fixture: object, name: str) -> str:
    if not isinstance(fixture, dict):
        raise TypeError(f"reduced-runner projection {name} is not an object")
    source = fixture.get("source")
    if not isinstance(source, dict):
        raise TypeError(f"reduced-runner projection {name} lacks source provenance")
    identity = source.get("complete_capture_sha256")
    if not isinstance(identity, str) or len(identity) != 64:
        raise RuntimeError(f"reduced-runner projection {name} lacks capture identity")
    return identity


def _synthetic_route_rows(
    fixture: object, *, layer: int
) -> tuple[dict[str, object], list[dict[str, object]]]:
    """Decode one captured synthetic source gate without evaluating a router."""
    if not isinstance(fixture, dict):
        raise TypeError(f"synthetic route fixture for layer {layer} is not an object")
    source = fixture.get("source")
    model = fixture.get("model")
    cases = fixture.get("cases")
    if (
        not isinstance(source, dict)
        or not isinstance(model, dict)
        or not isinstance(cases, list)
    ):
        raise TypeError(f"synthetic route fixture for layer {layer} is malformed")
    if any(
        model.get(name) != value
        for name, value in _SYNTHETIC_MOE_MODEL_GEOMETRY.items()
    ):
        raise RuntimeError(
            f"synthetic route fixture for layer {layer} changed geometry"
        )
    if source.get("storage_byteorder") != "little":
        raise RuntimeError(
            "synthetic route fixture must retain little-endian source storage"
        )
    starts = [
        case.get("start_pos") if isinstance(case, dict) else None for case in cases
    ]
    if starts != list(_ROUTE_STARTS) or any(type(start) is not int for start in starts):
        raise RuntimeError(
            "synthetic route fixture changed the pinned prefill/decode starts"
        )

    rows: list[dict[str, object]] = []
    for case_index, case in enumerate(cases):
        if not isinstance(case, dict):
            raise TypeError(f"synthetic route case {case_index} is not an object")
        indices = case.get("gate_indices")
        if not isinstance(indices, dict):
            raise TypeError(
                f"synthetic route case {case_index} lacks source gate indices"
            )
        shape = indices.get("shape")
        storage_hex = indices.get("storage_hex")
        storage_sha256 = indices.get("storage_sha256")
        if (
            indices.get("dtype") != "torch.int64"
            or not isinstance(shape, list)
            or len(shape) != 2
            or not all(type(value) is int for value in shape)
            or shape
            != [
                _ROUTE_TOKEN_COUNTS[case_index],
                _SYNTHETIC_ROUTE_GEOMETRY["selected_experts"],
            ]
            or not isinstance(storage_hex, str)
            or not isinstance(storage_sha256, str)
            or len(storage_sha256) != 64
        ):
            raise RuntimeError(
                f"synthetic route case {case_index} has invalid source gate layout"
            )
        try:
            storage = bytes.fromhex(storage_hex)
        except ValueError as error:
            raise RuntimeError(
                f"synthetic route case {case_index} has invalid source gate storage"
            ) from error
        if (
            len(storage) != shape[0] * shape[1] * 8
            or _sha256(storage) != storage_sha256
        ):
            raise RuntimeError(
                f"synthetic route case {case_index} source gate storage is not exact"
            )
        start_pos = case["start_pos"]
        phase = "prefill" if start_pos == 0 else "decode"
        for token_offset in range(shape[0]):
            offset = token_offset * shape[1] * 8
            expert_ids = [
                int.from_bytes(
                    storage[offset + expert * 8 : offset + (expert + 1) * 8],
                    "little",
                    signed=True,
                )
                for expert in range(shape[1])
            ]
            if any(
                expert < 0 or expert >= _SYNTHETIC_ROUTE_GEOMETRY["routed_experts"]
                for expert in expert_ids
            ) or len(set(expert_ids)) != len(expert_ids):
                raise RuntimeError(
                    f"synthetic route case {case_index} has invalid selected experts"
                )
            rows.append(
                {
                    "layer": layer,
                    "phase": phase,
                    "start_pos": start_pos,
                    "token_offset": token_offset,
                    "expert_ids": expert_ids,
                    "source_gate_indices_sha256": storage_sha256,
                }
            )
    return source, rows


def synthetic_route_trace_from_moe_projections(
    projections: dict[str, object],
) -> dict[str, object]:
    """Project exact source-selected IDs for synthetic collector qualification.

    This deliberately captures only the reduced source fixture's four-expert,
    two-selected routes.  It is not a pretrained V4.1 routing trace and cannot
    establish real-checkpoint cache locality or selected-byte traffic.
    """
    sources: list[dict[str, object]] = []
    rows: list[dict[str, object]] = []
    for layer in _ROUTE_LAYERS:
        source, layer_rows = _synthetic_route_rows(
            projections.get(f"layer{layer}_moe"), layer=layer
        )
        sources.append(source)
        rows.extend(layer_rows)
    identity = sources[0].get("complete_capture_sha256")
    if not isinstance(identity, str) or len(identity) != 64:
        raise RuntimeError(
            "synthetic route trace lacks a complete source capture identity"
        )
    identity_fields = (
        "revision",
        "model_sha256",
        "runner_sha256",
        "complete_capture_sha256",
        "storage_byteorder",
    )
    for field in identity_fields:
        value = sources[0].get(field)
        if not isinstance(value, str) or not value:
            raise RuntimeError(f"synthetic route trace lacks source {field}")
        if any(source.get(field) != value for source in sources[1:]):
            raise RuntimeError(f"synthetic route trace mixes source {field}")
    return {
        "schema_version": 1,
        "scope": (
            "exact source-selected routes from the bounded synthetic V4.1 fixture; "
            "collector qualification only, not pretrained-model locality, cache-hit, "
            "selected-byte traffic, or checkpoint-serving evidence"
        ),
        "source": {
            "revision": sources[0].get("revision"),
            "model_sha256": sources[0].get("model_sha256"),
            "complete_capture_sha256": identity,
            "runner_sha256": sources[0].get("runner_sha256"),
            "storage_byteorder": sources[0].get("storage_byteorder"),
            "route_projection_sha256": _sha256(Path(__file__).read_bytes()),
        },
        "synthetic_geometry": {
            **_SYNTHETIC_ROUTE_GEOMETRY,
            "layers": list(_ROUTE_LAYERS),
            "starts": list(_ROUTE_STARTS),
        },
        "rows": rows,
    }


def synthetic_route_trace(receipt: dict[str, object]) -> dict[str, object]:
    """Project synthetic source-route IDs from one completed source receipt."""
    runner = _runner_module()
    return synthetic_route_trace_from_moe_projections(
        {
            "layer3_moe": runner.moe_fixture(receipt, layer=3),
            "layer4_moe": runner.moe_fixture(receipt),
        }
    )


def reduced_runner_fixture(receipt: dict[str, object]) -> dict[str, object]:
    """Project every currently-native reduced-forward seam from one receipt."""
    if (
        receipt.get("capture_status")
        != "completed synthetic source-forward capture; no parity claim"
    ):
        raise RuntimeError("reduced-runner fixture requires a completed source capture")
    if receipt.get("coverage_status", {}).get("pending") != []:
        raise RuntimeError("reduced-runner fixture requires complete source coverage")
    serialized = _serialized(receipt)
    canonical = _sha256(serialized)
    # Project the same JSON representation for in-process and file-based callers.
    # The source runner may retain tuple-valued model arguments in memory.
    receipt = json.loads(serialized)
    runner = _runner_module()
    projections = {
        "layer0_to_layer1": v41_layer0_to_layer1_capture.layer0_to_layer1_fixture(
            receipt
        ),
        "layer1_engram": v41_layer1_engram_capture.engram_fixture(receipt),
        "layer1_owner": v41_layer1_owner_capture.layer1_owner_fixture(receipt),
        "layer1_attention": v41_layer1_attention_capture.attention_fixture(
            receipt, helper_path=SCRIPTS / "v41_layer1_attention_capture.py"
        ),
        "layer1_tail": v41_layer1_tail_capture.layer1_tail_fixture(receipt),
        "layer2_attention": v41_layer2_attention_capture.attention_fixture(
            receipt, helper_path=SCRIPTS / "v41_layer2_attention_capture.py"
        ),
        "layer2_hc": v41_layer2_hc_capture.layer2_hc_fixture(receipt),
        "layer2_ffn": v41_layer2_ffn_capture.layer2_ffn_fixture(receipt),
        "layer3_engram": v41_layer3_engram_capture.engram_fixture(receipt),
        "layer3_attention": v41_layer3_attention_capture.layer_three_attention_fixture(
            receipt, helper_path=SCRIPTS / "v41_layer3_attention_capture.py"
        ),
        "layer3_moe": runner.moe_fixture(receipt, layer=3),
        "layer3_candidate": v41_candidate_capture.candidate_fixture(receipt),
        "layer3_index_key": v41_index_key_capture.index_key_fixture(receipt),
        "layer3_compressor": v41_index_key_capture.compressor_fixture(receipt),
        "layer3_to_layer1": v41_index_key_capture.layer3_to_layer1_fixture(receipt),
        "layer4_attention": v41_attention_capture.attention_fixture(
            receipt, helper_path=SCRIPTS / "v41_attention_capture.py"
        ),
        "layer4_moe": runner.moe_fixture(receipt),
        "head": runner.head_fixture(receipt),
    }
    for name, fixture in projections.items():
        if _source_identity(fixture, name) != canonical:
            raise RuntimeError(
                f"reduced-runner projection {name} does not derive from the supplied capture"
            )
    source = receipt.get("source")
    if not isinstance(source, dict):
        raise TypeError("reduced-runner receipt lacks source provenance")
    return {
        "schema_version": 1,
        "scope": (
            "same-trace source projections for the reduced native V4.1 forward; "
            "not a whole-model implementation or pretrained-model parity claim"
        ),
        "source": {
            "revision": source.get("revision"),
            "model_sha256": source.get("model_sha256"),
            "engram_sha256": source.get("engram_sha256"),
            "complete_capture_sha256": canonical,
            "runner_sha256": source.get("runner_sha256"),
            "forward_observers_sha256": source.get("forward_observers_sha256"),
            "extractor_sha256": _sha256(Path(__file__).read_bytes()),
        },
        "trace": {"starts": [0, 5, 6], "input_ids": [[0, 1, 2, 3, 4, 5, 6]]},
        "projections": projections,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--synthetic-route-trace-output",
        type=Path,
        help=(
            "write the additive source-selected synthetic route trace here; "
            "it qualifies collector accounting, not pretrained locality"
        ),
    )
    args = parser.parse_args()
    route_output: Path | None = None
    if args.synthetic_route_trace_output is not None:
        route_output = args.synthetic_route_trace_output.resolve()
        if (
            route_output == args.input.resolve()
            or route_output == args.output.resolve()
        ):
            raise RuntimeError(
                "synthetic route trace output must not clobber input or fixture"
            )
    if not args.input.is_file() or args.input.stat().st_size > MAX_INPUT_BYTES:
        raise RuntimeError("reduced-runner input must be a bounded complete capture")
    receipt = json.loads(
        args.input.read_text(),
        parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError(f"reduced-runner fixture rejects non-finite JSON {value}")
        ),
    )
    fixture = reduced_runner_fixture(receipt)
    encoded = (
        json.dumps(fixture, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode()
    if len(encoded) > MAX_OUTPUT_BYTES:
        raise RuntimeError("reduced-runner fixture exceeds its bounded size")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(encoded)
    if route_output is not None:
        route_trace = synthetic_route_trace(receipt)
        route_encoded = (
            json.dumps(
                route_trace, sort_keys=True, separators=(",", ":"), allow_nan=False
            )
            + "\n"
        ).encode()
        if len(route_encoded) > MAX_OUTPUT_BYTES:
            raise RuntimeError(
                "synthetic route trace exceeds its bounded output budget"
            )
        route_output.parent.mkdir(parents=True, exist_ok=True)
        route_output.write_bytes(route_encoded)
    print(
        json.dumps(
            {
                "artifact_sha256": _sha256(encoded),
                "bytes": len(encoded),
                "path": str(args.output),
                "status": "source_reduced_runner_fixture",
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
