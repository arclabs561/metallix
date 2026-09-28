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
"""Project the source layer-zero producer into the layer-one Engram seam.

The reduced source trace is the authority for the layer-zero block result.  We
retain the complete layer-zero HC/attention/FFN boundary records and prove that
the terminal residual is the exact storage supplied to layer one's Engram.  The Rust integration replays the captured attention operands natively, then
uses the resulting block stream only after this extractor verifies the
provenance-preserving handoff.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
from pathlib import Path
from typing import Any

MAX_INPUT_BYTES = 16 * 1024 * 1024
MAX_FIXTURE_BYTES = 576 * 1024
EXPECTED_START_POSITIONS = (0, 5, 6)
SOURCE_FIELDS = (
    "revision",
    "model_sha256",
    "engram_sha256",
    "kernel_source_sha256",
    "cpu_backend_sha256",
    "loader_sha256",
    "runner_sha256",
    "forward_observers_sha256",
)
WIDTHS = {
    "torch.bfloat16": 2,
    "torch.float32": 4,
    "torch.int64": 8,
    "torch.int32": 4,
    "torch.complex64": 8,
    "torch.float8_e4m3fn": 1,
    "torch.float8_e8m0fnu": 1,
    # The source receipt exposes one packed FP4 byte per logical entry.
    "torch.float4_e2m1fn_x2": 1,
}


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def serialized_capture(receipt: dict[str, object]) -> bytes:
    return (
        json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode()


def _object(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"layer-zero bridge lacks object {label}")
    return value


def _tensor(
    value: object, label: str, *, dtype: str, shape: list[int]
) -> dict[str, Any]:
    record = _object(value, label)
    if record.get("dtype") != dtype or record.get("shape") != shape:
        raise RuntimeError(f"layer-zero bridge has unexpected {label} layout")
    count = math.prod(shape)
    if record.get("numel") != count or record.get("finite") is not True:
        raise RuntimeError(f"layer-zero bridge has invalid {label} receipt")
    raw_hex, digest = record.get("storage_hex"), record.get("storage_sha256")
    if not isinstance(raw_hex, str) or not isinstance(digest, str):
        raise TypeError(f"layer-zero bridge lacks exact {label} storage")
    try:
        raw = bytes.fromhex(raw_hex)
    except ValueError as error:
        raise RuntimeError(
            f"layer-zero bridge has malformed {label} storage"
        ) from error
    if len(raw) != count * WIDTHS[dtype] or _sha256(raw) != digest:
        raise RuntimeError(f"layer-zero bridge has invalid {label} storage")
    if dtype == "torch.bfloat16" and any(
        ((word >> 7) & 0xFF) == 0xFF for (word,) in struct.iter_unpack("<H", raw)
    ):
        raise RuntimeError(f"layer-zero bridge has nonfinite {label} storage")
    if dtype == "torch.float32" and any(
        not math.isfinite(number) for (number,) in struct.iter_unpack("<f", raw)
    ):
        raise RuntimeError(f"layer-zero bridge has nonfinite {label} storage")
    if dtype == "torch.complex64" and any(
        not math.isfinite(component)
        for component in struct.unpack(f"<{count * 2}f", raw)
    ):
        raise RuntimeError(f"layer-zero bridge has nonfinite {label} storage")
    if dtype == "torch.float8_e4m3fn" and any(code in (0x7F, 0xFF) for code in raw):
        raise RuntimeError(f"layer-zero bridge has nonfinite {label} storage")
    if dtype == "torch.float8_e8m0fnu" and 0xFF in raw:
        raise RuntimeError(f"layer-zero bridge has nonfinite {label} storage")
    return record


def _coefficients(value: object, label: str, sequence: int) -> dict[str, Any]:
    record = _object(value, label)
    if set(record) != {"pre", "post", "comb"}:
        raise RuntimeError(f"layer-zero bridge lacks complete {label} coefficients")
    _tensor(
        record["pre"], f"{label} pre", dtype="torch.float32", shape=[1, sequence, 2]
    )
    _tensor(
        record["post"], f"{label} post", dtype="torch.float32", shape=[1, sequence, 2]
    )
    _tensor(
        record["comb"],
        f"{label} comb",
        dtype="torch.float32",
        shape=[1, sequence, 2, 2],
    )
    return record


def _parameters(encoded: dict[str, Any]) -> dict[str, Any]:
    layouts = {
        "embed.weight": ("torch.bfloat16", [8, 128]),
        "layers.0.hc_attn_fn": ("torch.float32", [8, 256]),
        "layers.0.hc_attn_base": ("torch.float32", [8]),
        "layers.0.hc_attn_scale": ("torch.float32", [3]),
        "layers.0.attn_norm.weight": ("torch.bfloat16", [128]),
        "layers.0.hc_ffn_fn": ("torch.float32", [8, 256]),
        "layers.0.hc_ffn_base": ("torch.float32", [8]),
        "layers.0.hc_ffn_scale": ("torch.float32", [3]),
        "layers.0.ffn_norm.weight": ("torch.bfloat16", [128]),
        "layers.1.attn_norm.weight": ("torch.bfloat16", [128]),
    }
    ffn = "layers.0.ffn"
    routed = tuple(f"{ffn}.experts.{index}" for index in range(4))
    layouts.update(
        {
            f"{ffn}.gate.weight": ("torch.bfloat16", [4, 128]),
            f"{ffn}.gate.bias": ("torch.float32", [4]),
        }
    )
    for prefix in routed:
        for projection in ("w1", "w2", "w3"):
            layouts[f"{prefix}.{projection}.weight"] = (
                "torch.float4_e2m1fn_x2",
                [128, 64],
            )
            layouts[f"{prefix}.{projection}.scale"] = ("torch.float8_e8m0fnu", [128, 4])
    for projection in ("w1", "w2", "w3"):
        layouts[f"{ffn}.shared_experts.{projection}.weight"] = (
            "torch.float8_e4m3fn",
            [128, 128],
        )
        layouts[f"{ffn}.shared_experts.{projection}.scale"] = (
            "torch.float8_e8m0fnu",
            [4, 4],
        )
    attention = "layers.0.attn"
    layouts.update(
        {
            f"{attention}.wq_a.weight": ("torch.float8_e4m3fn", [32, 128]),
            f"{attention}.wq_a.scale": ("torch.float8_e8m0fnu", [1, 4]),
            f"{attention}.q_norm.weight": ("torch.bfloat16", [32]),
            f"{attention}.wq_b.weight": ("torch.float8_e4m3fn", [128, 32]),
            f"{attention}.wq_b.scale": ("torch.float8_e8m0fnu", [4, 1]),
            f"{attention}.wkv.weight": ("torch.float8_e4m3fn", [64, 128]),
            f"{attention}.wkv.scale": ("torch.float8_e8m0fnu", [2, 4]),
            f"{attention}.kv_norm.weight": ("torch.bfloat16", [64]),
            f"{attention}.attn_sink": ("torch.float32", [2]),
            f"{attention}.wo_a.weight": ("torch.bfloat16", [64, 64]),
            f"{attention}.wo_b.weight": ("torch.float8_e4m3fn", [128, 64]),
            f"{attention}.wo_b.scale": ("torch.float8_e8m0fnu", [4, 2]),
        }
    )
    result = {name: encoded.get(name) for name in layouts}
    if any(value is None for value in result.values()):
        raise RuntimeError("layer-zero bridge lacks required HC or norm parameter")
    for name, (dtype, shape) in layouts.items():
        _tensor(result[name], name, dtype=dtype, shape=shape)
    return result


def layer0_to_layer1_fixture(receipt: dict[str, object]) -> dict[str, object]:
    """Extract one same-trace layer-zero producer to native layer-one seam."""
    if (
        receipt.get("capture_status")
        != "completed synthetic source-forward capture; no parity claim"
    ):
        raise RuntimeError("layer-zero bridge requires a completed source capture")
    source = _object(receipt.get("source"), "source provenance")
    runtime = _object(receipt.get("runtime"), "runtime provenance")
    model = _object(receipt.get("model_args"), "model arguments")
    encoded = _object(receipt.get("encoded_parameters"), "encoded parameters")
    steps = receipt.get("steps")
    coverage = _object(receipt.get("coverage_status"), "coverage")
    if runtime.get("storage_byteorder") != "little" or coverage.get("pending") != []:
        raise RuntimeError(
            "layer-zero bridge requires complete little-endian capture coverage"
        )
    if any(not isinstance(source.get(field), str) for field in SOURCE_FIELDS):
        raise RuntimeError("layer-zero bridge has incomplete source provenance")
    if not isinstance(steps, list) or len(steps) != len(EXPECTED_START_POSITIONS):
        raise RuntimeError("layer-zero bridge requires the pinned prefill/decode trace")
    geometry = {"dim": 128, "hc_mult": 2, "n_layers": 5, "vocab_size": 8}
    geometry.update(
        {
            "moe_inter_dim": 128,
            "n_routed_experts": 4,
            "n_activated_experts": 2,
            "n_shared_experts": 1,
            "score_func": "sqrtsoftplus",
            "gate_temp": 1.0,
            "norm_topk_prob": True,
            "route_scale": 1.0,
            "swiglu_limit": 0.0,
            "n_heads": 2,
            "head_dim": 64,
            "rope_head_dim": 32,
            "q_lora_rank": 32,
            "o_lora_rank": 32,
            "o_groups": 2,
            "window_size": 6,
            "norm_eps": 1e-20,
            "hc_eps": 1e-6,
            "hc_sinkhorn_iters": 20,
        }
    )
    if any(model.get(name) != value for name, value in geometry.items()):
        raise RuntimeError("layer-zero bridge has unexpected reduced model geometry")

    cases: list[dict[str, object]] = []
    for step, start_pos in zip(steps, EXPECTED_START_POSITIONS, strict=True):
        source_step = _object(step, "capture step")
        if source_step.get("start_pos") != start_pos:
            raise RuntimeError("layer-zero bridge start positions drifted")
        sequence = 5 if start_pos == 0 else 1
        values = _object(source_step.get("intermediates"), "step intermediates")
        block_input = _object(
            values.get("layers.0.block_input"), "layer-zero block input"
        )
        block_residual = _tensor(
            block_input.get("residual"),
            "layer-zero block residual",
            dtype="torch.bfloat16",
            shape=[1, sequence, 2, 128],
        )
        block_pre = _tensor(
            block_input.get("incoming_pre"),
            "layer-zero block pre",
            dtype="torch.float32",
            shape=[1, sequence, 2],
        )
        input_ids = _tensor(
            source_step.get("input_ids"),
            "layer-zero input IDs",
            dtype="torch.int64",
            shape=[1, sequence],
        )
        embedding = _tensor(
            values.get("embed"),
            "layer-zero embedding",
            dtype="torch.bfloat16",
            shape=[1, sequence, 128],
        )
        attention_input = _tensor(
            values.get("layers.0.attention_input"),
            "layer-zero attention input",
            dtype="torch.bfloat16",
            shape=[1, sequence, 128],
        )
        attention_output = _tensor(
            values.get("layers.0.attn"),
            "layer-zero attention output",
            dtype="torch.bfloat16",
            shape=[1, sequence, 128],
        )
        after_attention = _tensor(
            values.get("layers.0.after_attention_residual"),
            "layer-zero post-attention residual",
            dtype="torch.bfloat16",
            shape=[1, sequence, 2, 128],
        )
        ffn_input = _tensor(
            values.get("layers.0.ffn_input"),
            "layer-zero FFN input",
            dtype="torch.bfloat16",
            shape=[1, sequence, 128],
        )
        ffn_collapsed = _tensor(
            values.get("layers.0.ffn_collapsed"),
            "layer-zero FFN collapsed",
            dtype="torch.bfloat16",
            shape=[1, sequence, 128],
        )
        ffn_output = _tensor(
            values.get("layers.0.ffn"),
            "layer-zero FFN output",
            dtype="torch.bfloat16",
            shape=[1, sequence, 128],
        )
        gate = values.get("layers.0.ffn.gate")
        if not isinstance(gate, list) or len(gate) != 2:
            raise TypeError("layer-zero bridge lacks FFN route weights and indices")
        gate_weights = _tensor(
            gate[0],
            "layer-zero route weights",
            dtype="torch.float32",
            shape=[sequence, 2],
        )
        gate_indices = _tensor(
            gate[1],
            "layer-zero route indices",
            dtype="torch.int64",
            shape=[sequence, 2],
        )
        frequency = _tensor(
            values.get("layers.0.attn.freqs_cis"),
            "layer-zero frequency table",
            dtype="torch.complex64",
            shape=[8, 16],
        )
        attention_stages = {
            "wq_a": _tensor(
                values.get("layers.0.attn.wq_a"),
                "layer-zero WQ-A",
                dtype="torch.bfloat16",
                shape=[1, sequence, 32],
            ),
            "q_norm": _tensor(
                values.get("layers.0.attn.q_norm"),
                "layer-zero Q norm",
                dtype="torch.bfloat16",
                shape=[1, sequence, 32],
            ),
            "wq_b": _tensor(
                values.get("layers.0.attn.wq_b"),
                "layer-zero WQ-B",
                dtype="torch.bfloat16",
                shape=[1, sequence, 128],
            ),
            "wo_b_input": _tensor(
                values.get("layers.0.attn.wo_b_input"),
                "layer-zero WO-B input",
                dtype="torch.bfloat16",
                shape=[1, sequence, 64],
            ),
        }
        window = _object(
            values.get("layers.0.attn.window"), "layer-zero attention window"
        )
        window_shape = [1, sequence if start_pos == 0 else 6, 64]
        attention_window = {
            "prepared": _tensor(
                window.get("prepared_window_kv"),
                "layer-zero prepared window",
                dtype="torch.bfloat16",
                shape=[1, sequence, 64],
            ),
            "read": _tensor(
                window.get("window_kv"),
                "layer-zero window read",
                dtype="torch.bfloat16",
                shape=window_shape,
            ),
            "indices": _tensor(
                window.get("indices"),
                "layer-zero window indices",
                dtype="torch.int32",
                shape=[1, sequence, sequence if start_pos == 0 else 6],
            ),
            "ring_after": _tensor(
                window.get("ring_after"),
                "layer-zero window ring",
                dtype="torch.bfloat16",
                shape=[1, 6, 64],
            ),
        }
        sparse_calls = [
            call
            for call in source_step.get("sparse_attention_calls", [])
            if isinstance(call, dict) and call.get("layer_id") == 0
        ]
        if len(sparse_calls) != 1:
            raise RuntimeError("layer-zero bridge requires one layer-zero sparse call")
        sparse_inputs = _object(
            sparse_calls[0].get("inputs"), "layer-zero sparse inputs"
        )
        attention_sparse = {
            "q_after_rope": _tensor(
                sparse_inputs.get("q_after_rope"),
                "layer-zero sparse query",
                dtype="torch.bfloat16",
                shape=[1, sequence, 2, 64],
            ),
            "kv": _tensor(
                sparse_inputs.get("kv"),
                "layer-zero sparse KV",
                dtype="torch.bfloat16",
                shape=window_shape,
            ),
            "sink": _tensor(
                sparse_inputs.get("sink"),
                "layer-zero sparse sink",
                dtype="torch.float32",
                shape=[2],
            ),
            "indices": _tensor(
                sparse_inputs.get("indices"),
                "layer-zero sparse indices",
                dtype="torch.int32",
                shape=[1, sequence, sequence if start_pos == 0 else 6],
            ),
            "output_pre_inverse_rope": _tensor(
                sparse_calls[0].get("output_pre_inverse_rope"),
                "layer-zero sparse output",
                dtype="torch.bfloat16",
                shape=[1, sequence, 2, 64],
            ),
        }
        terminal = values.get("layers.0")
        if not isinstance(terminal, list) or len(terminal) != 2:
            raise TypeError("layer-zero bridge lacks terminal residual plus pre-mix")
        output = _tensor(
            terminal[0],
            "layer-zero terminal residual",
            dtype="torch.bfloat16",
            shape=[1, sequence, 2, 128],
        )
        next_pre = _tensor(
            terminal[1],
            "layer-zero terminal pre",
            dtype="torch.float32",
            shape=[1, sequence, 2],
        )
        layer_one_engram_output = _tensor(
            values.get("layers.1.engram"),
            "layer-one Engram output",
            dtype="torch.bfloat16",
            shape=[1, sequence, 2, 128],
        )
        layer_one_attention_input = _tensor(
            values.get("layers.1.attention_input"),
            "layer-one attention input",
            dtype="torch.bfloat16",
            shape=[1, sequence, 128],
        )
        engram = _object(values.get("layers.1.engram_input"), "layer-one Engram input")
        stream = _tensor(
            engram.get("stream"),
            "layer-one Engram stream",
            dtype="torch.bfloat16",
            shape=[1, sequence, 2, 128],
        )
        if output["storage_sha256"] != stream["storage_sha256"]:
            raise RuntimeError(
                "layer-zero terminal residual does not feed layer-one Engram"
            )
        calls = source_step.get("hyper_connection_mixes")
        if not isinstance(calls, list):
            raise TypeError("layer-zero bridge lacks HC calls")
        layer_calls = {
            call.get("sublayer"): call
            for call in calls
            if isinstance(call, dict) and call.get("layer_id") == 0
        }
        if set(layer_calls) != {"attention", "ffn"}:
            raise RuntimeError("layer-zero bridge requires both HC calls")
        hc: dict[str, dict[str, Any]] = {}
        for kind, call in layer_calls.items():
            inputs = _object(call.get("inputs"), f"layer-zero {kind} HC inputs")
            mixes = _tensor(
                inputs.get("mixes"),
                f"layer-zero {kind} HC mixes",
                dtype="torch.float32",
                shape=[1, sequence, 8],
            )
            coefficients = _coefficients(
                call.get("outputs"), f"layer-zero {kind} HC", sequence
            )
            hc[kind] = {"mixes": mixes, "coefficients": coefficients}
        cases.append(
            {
                "start_pos": start_pos,
                "startup": {"input_ids": input_ids, "embedding": embedding},
                "block_input": {"residual": block_residual, "incoming_pre": block_pre},
                "attention_input": attention_input,
                "attention_output": attention_output,
                "after_attention_residual": after_attention,
                "ffn_input": ffn_input,
                "ffn_collapsed": ffn_collapsed,
                "ffn_output": ffn_output,
                "gate_weights": gate_weights,
                "gate_indices": gate_indices,
                "attention": {
                    "frequencies": frequency,
                    "stages": attention_stages,
                    "window": attention_window,
                    "sparse": attention_sparse,
                },
                "block_output": output,
                "block_next_pre": next_pre,
                "layer_one_engram_stream": stream,
                "downstream": {
                    "layer_one_engram_output": layer_one_engram_output,
                    "layer_one_attention_input": layer_one_attention_input,
                },
                "hc": hc,
            }
        )
    capture = serialized_capture(receipt)
    return {
        "schema_version": 1,
        "source": {
            **{field: source[field] for field in SOURCE_FIELDS},
            "complete_capture_sha256": _sha256(capture),
            "extractor_sha256": _sha256(Path(__file__).read_bytes()),
        },
        "model": geometry,
        "contract": {
            "layer_zero_producer": "source-pinned native startup, attention, FFN, and HC composition",
            "layer_one_consumer": "same-storage source oracle consumed by the native Engram entry",
            "remaining_producer_boundary": "no layer-zero runtime numerical operand remains source-captured; retained HC records are exact source oracles",
        },
        "parameters": _parameters(encoded),
        "cases": cases,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw = args.input.read_bytes()
    if len(raw) > MAX_INPUT_BYTES:
        raise RuntimeError("layer-zero bridge input exceeds receipt cap")
    fixture = layer0_to_layer1_fixture(json.loads(raw))
    output = (
        json.dumps(fixture, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode()
    if len(output) >= MAX_FIXTURE_BYTES:
        raise RuntimeError("layer-zero bridge fixture exceeds compact receipt cap")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(output)
    print(
        json.dumps(
            {
                "artifact_sha256": _sha256(output),
                "bytes": len(output),
                "complete_capture_sha256": fixture["source"]["complete_capture_sha256"],
                "path": str(args.output),
                "status": "source_layer_zero_to_layer_one_fixture",
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
