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


def reduced_runner_fixture(receipt: dict[str, object]) -> dict[str, object]:
    """Project every currently-native reduced-forward seam from one receipt."""
    if (
        receipt.get("capture_status")
        != "completed synthetic source-forward capture; no parity claim"
    ):
        raise RuntimeError("reduced-runner fixture requires a completed source capture")
    if receipt.get("coverage_status", {}).get("pending") != []:
        raise RuntimeError("reduced-runner fixture requires complete source coverage")
    canonical = _sha256(_serialized(receipt))
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
    args = parser.parse_args()
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
