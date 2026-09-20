"""Project layer-three source attention records into a distinct fixture.

The generic layer-four projector already validates all tensor storage and
attention layouts. This adapter permits only the layer-three records required
by the same native attention operator, maps them through that projector, then
restores their true source names. It performs no attention arithmetic.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

from v41_attention_capture import _sha256_bytes, attention_fixture

LAYER = 3
REQUIRED_PARAMETERS = (
    "attn_sink",
    "wq_a.weight",
    "wq_a.scale",
    "q_norm.weight",
    "wq_b.weight",
    "wq_b.scale",
    "wkv.weight",
    "wkv.scale",
    "kv_norm.weight",
    "wo_a.weight",
    "wo_b.weight",
    "wo_b.scale",
)
MAX_INPUT_BYTES = 16 * 1024 * 1024


def _require_dict(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"layer-three attention capture lacks {label}")
    return value


def _receipt_for_generic_projector(receipt: dict[str, object]) -> dict[str, object]:
    """Make only exact layer-three aliases accepted by the generic projector."""
    mapped = copy.deepcopy(receipt)
    encoded = _require_dict(mapped.get("encoded_parameters"), "encoded parameters")
    static = _require_dict(mapped.get("attention_static"), "attention static inputs")
    if "layer_3_freqs_cis" not in static:
        raise RuntimeError(
            "layer-three attention capture lacks layer-three frequencies"
        )
    static["layer_4_freqs_cis"] = static["layer_3_freqs_cis"]
    for suffix in REQUIRED_PARAMETERS:
        source = f"layers.{LAYER}.attn.{suffix}"
        target = f"layers.4.attn.{suffix}"
        if source not in encoded:
            raise RuntimeError(
                f"layer-three attention capture lacks parameter {source}"
            )
        encoded[target] = encoded[source]
    for step in mapped.get("steps", []):
        item = _require_dict(step, "source step")
        intermediate = _require_dict(item.get("intermediates"), "source intermediates")
        for suffix in (
            "attention_input",
            "attn.wq_a",
            "attn.q_norm",
            "attn.wq_b",
            "attn.window",
            "attn.compressed",
            "attn.indexer_observation",
            "attn.wo_b_input",
            "attn",
        ):
            source = f"layers.{LAYER}.{suffix}"
            target = f"layers.4.{suffix}"
            if source not in intermediate:
                raise RuntimeError(
                    f"layer-three attention capture lacks boundary {source}"
                )
            intermediate[target] = intermediate[source]
        calls = item.get("sparse_attention_calls")
        if not isinstance(calls, list):
            raise TypeError("layer-three attention capture lacks sparse calls")
        layer_calls = [
            call
            for call in calls
            if isinstance(call, dict) and call.get("layer_id") == LAYER
        ]
        if len(layer_calls) != 1:
            raise RuntimeError(
                "layer-three attention capture requires one sparse call per step"
            )
        sparse = copy.deepcopy(layer_calls[0])
        sparse["layer_id"] = 4
        item["sparse_attention_calls"] = [sparse]
    return mapped


def layer_three_attention_fixture(
    receipt: dict[str, object], *, helper_path: Path
) -> dict[str, object]:
    """Export exact layer-three source attention boundaries and provenance."""
    fixture = attention_fixture(
        _receipt_for_generic_projector(receipt),
        helper_path=Path(__file__).with_name("v41_attention_capture.py"),
    )
    parameters = _require_dict(
        fixture.get("encoded_parameters"), "projected parameters"
    )
    fixture["encoded_parameters"] = {
        f"layers.{LAYER}.attn.{suffix}": parameters[f"layers.4.attn.{suffix}"]
        for suffix in REQUIRED_PARAMETERS
    }
    fixture["scope"] = (
        "layer-three source attention boundaries fed by native owner publication; "
        "not native block continuation, runner transaction, or full-model parity"
    )
    fixture["frequency_scope"] = "full layer-three source freqs_cis"
    policy = _require_dict(fixture.get("comparison_policy"), "comparison policy")
    policy["compressed_kv_and_indices"] = (
        "source layer-three _compress_kv read and selected IDs; native owner must "
        "supply the same complete prefix before attention"
    )
    source = _require_dict(fixture.get("source"), "source provenance")
    source["complete_capture_sha256"] = _sha256_bytes(
        (json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    )
    source["layer_attention_helper_sha256"] = _sha256_bytes(helper_path.read_bytes())
    source["frequency_source"] = "attention_static.layer_3_freqs_cis"
    return fixture


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.input.is_file() or args.input.stat().st_size > MAX_INPUT_BYTES:
        raise RuntimeError("layer-three attention input must be a bounded regular file")
    receipt = json.loads(
        args.input.read_text(),
        parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError(
                f"layer-three attention rejects non-finite JSON constant {value}"
            )
        ),
    )
    fixture = layer_three_attention_fixture(receipt, helper_path=Path(__file__))
    data = (json.dumps(fixture, sort_keys=True, separators=(",", ":")) + "\n").encode()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(data)
    print(
        json.dumps(
            {
                "artifact_sha256": _sha256_bytes(data),
                "bytes": len(data),
                "path": str(args.output),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
