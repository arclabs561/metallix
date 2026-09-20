"""Project layer-two source attention and its borrowed layer-one publication.

The source layer has no Indexer: its compressed KV and indices must be the
current layer-one ratio-two owner publication.  This exporter performs no
attention arithmetic; every tensor is exact source storage.
"""

from __future__ import annotations

import hashlib
import json
import math
import struct
import sys
from pathlib import Path
from typing import Any

PINNED_SOURCE = {
    "revision": "dba1be0a40aa45a94ad051997016db3960a90277",
    "model_sha256": "4e9ae23620edc8028ccc5d5fef552ab7fdc7dcd6f79608754fe9f67644056f65",
    "engram_sha256": "11f35ecbead8150c35aa002b3d180ef290b05a25afe883a11884f94d476d3897",
    "kernel_source_sha256": "1236c3507019ed176f5dba5e04bcea58867cf654818c6cf138ed4845398c2455",
}


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _serialized_capture(receipt: dict[str, object]) -> bytes:
    return (
        json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _tensor(record: object, label: str, *, dtype: str | None = None) -> dict[str, Any]:
    if not isinstance(record, dict) or not isinstance(record.get("storage_hex"), str):
        raise TypeError(f"attention fixture lacks complete storage for {label}")
    shape = record.get("shape")
    recorded_dtype = record.get("dtype")
    widths = {
        "torch.bfloat16": 2,
        "torch.float32": 4,
        "torch.int32": 4,
        "torch.complex64": 8,
        "torch.float8_e4m3fn": 1,
        "torch.float8_e8m0fnu": 1,
    }
    if (
        not isinstance(shape, list)
        or not all(type(n) is int and n >= 0 for n in shape)
        or recorded_dtype not in widths
        or type(record.get("numel")) is not int
        or record.get("numel") != math.prod(shape)
    ):
        raise TypeError(f"attention fixture has invalid tensor metadata for {label}")
    if dtype is not None and recorded_dtype != dtype:
        raise RuntimeError(
            f"attention fixture expected {label} to be {dtype}, got {recorded_dtype!r}"
        )
    try:
        raw = bytes.fromhex(record["storage_hex"])
    except ValueError as error:
        raise RuntimeError(
            f"attention fixture has invalid storage hex for {label}"
        ) from error
    if len(raw) != record["numel"] * widths[recorded_dtype] or _sha256_bytes(
        raw
    ) != record.get("storage_sha256"):
        raise RuntimeError(
            f"attention fixture exact storage does not match its receipt for {label}"
        )
    if record.get("finite") is not True:
        raise RuntimeError(f"attention fixture has nonfinite receipt for {label}")
    nonfinite = False
    if recorded_dtype == "torch.bfloat16":
        nonfinite = any(
            (int.from_bytes(raw[i : i + 2], "little") >> 7) & 0xFF == 0xFF
            for i in range(0, len(raw), 2)
        )
    elif recorded_dtype == "torch.float32":
        nonfinite = any(
            not math.isfinite(value)
            for value in struct.unpack(f"<{record['numel']}f", raw)
        )
    elif recorded_dtype == "torch.complex64":
        nonfinite = any(
            not math.isfinite(value)
            for value in struct.unpack(f"<{record['numel'] * 2}f", raw)
        )
    elif recorded_dtype == "torch.float8_e4m3fn":
        nonfinite = any(code in (0x7F, 0xFF) for code in raw)
    elif recorded_dtype == "torch.float8_e8m0fnu":
        nonfinite = 0xFF in raw
    if nonfinite:
        raise RuntimeError(f"attention fixture contains nonfinite storage for {label}")
    return record


def _raw(record: dict[str, Any]) -> bytes:
    return bytes.fromhex(record["storage_hex"])


def _sparse_owner_operands(
    sparse_inputs: dict[str, object],
    window: dict[str, Any],
    compressed: dict[str, Any],
    label: str,
) -> None:
    kv = _tensor(sparse_inputs.get("kv"), f"{label} sparse KV", dtype="torch.bfloat16")
    indices = _tensor(
        sparse_inputs.get("indices"), f"{label} sparse indices", dtype="torch.int32"
    )
    window_kv = _tensor(
        window.get("window_kv"), f"{label} window KV", dtype="torch.bfloat16"
    )
    compressed_kv = _tensor(
        compressed.get("borrowed_kv"), f"{label} compressed KV", dtype="torch.bfloat16"
    )
    window_indices = _tensor(
        window.get("indices"), f"{label} window indices", dtype="torch.int32"
    )
    compressed_indices = _tensor(
        compressed.get("indices"), f"{label} compressed indices", dtype="torch.int32"
    )
    if kv["shape"] != [
        1,
        window_kv["shape"][1] + compressed_kv["shape"][1],
        64,
    ] or indices["shape"] != [
        1,
        window_indices["shape"][1],
        window_indices["shape"][2] + compressed_indices["shape"][2],
    ]:
        raise RuntimeError(
            f"{label} sparse operand shape disagrees with cache operands"
        )
    if _raw(kv) != _raw(window_kv) + _raw(compressed_kv):
        raise RuntimeError(
            f"{label} sparse KV does not include the borrowed owner publication"
        )
    row_window = window_indices["shape"][2] * 4
    row_compressed = compressed_indices["shape"][2] * 4
    expected_indices = b"".join(
        _raw(window_indices)[row * row_window : (row + 1) * row_window]
        + _raw(compressed_indices)[row * row_compressed : (row + 1) * row_compressed]
        for row in range(window_indices["shape"][1])
    )
    if _raw(indices) != expected_indices:
        raise RuntimeError(
            f"{label} sparse indices do not include the borrowed owner publication"
        )


def _complex_f32_pairs(record: object) -> dict[str, object]:
    """Preserve source complex64 frequencies as exact real/imaginary FP32 bits."""
    value = _tensor(record, "layer_2_freqs_cis", dtype="torch.complex64")
    raw = bytes.fromhex(value["storage_hex"])
    numel = value["numel"]
    if len(raw) != numel * 8:
        raise RuntimeError("layer_2_freqs_cis storage length does not match complex64")
    if sys.byteorder != "little":
        raise RuntimeError(
            "attention fixture export requires little-endian CPU storage"
        )
    return {
        "shape": value["shape"],
        "complex_dtype": "torch.complex64",
        "fp32_pairs": [
            [
                int.from_bytes(raw[offset : offset + 4], "little"),
                int.from_bytes(raw[offset + 4 : offset + 8], "little"),
            ]
            for offset in range(0, len(raw), 8)
        ],
    }


def _source(receipt: dict[str, object]) -> tuple[dict[str, object], dict[str, object]]:
    source = receipt.get("source")
    runtime = receipt.get("runtime")
    if not isinstance(source, dict) or not isinstance(runtime, dict):
        raise TypeError("complete capture lacks source provenance")
    if runtime.get("storage_byteorder") != "little":
        raise RuntimeError("attention fixture requires little-endian source storage")
    for field, expected in PINNED_SOURCE.items():
        if source.get(field) != expected:
            raise RuntimeError(
                f"layer-two attention fixture has unexpected source {field}"
            )
    scripts = Path(__file__).parent
    for field, filename in (
        ("cpu_backend_sha256", "v41_cpu_kernels.py"),
        ("loader_sha256", "v41_source_loader.py"),
        ("runner_sha256", "v41-forward-reference.py"),
        ("forward_observers_sha256", "v41_forward_observers.py"),
    ):
        if source.get(field) != _sha256_bytes((scripts / filename).read_bytes()):
            raise RuntimeError(f"layer-two attention fixture has stale source {field}")
    return source, runtime


def attention_fixture(
    receipt: dict[str, object], *, helper_path: Path
) -> dict[str, object]:
    """Export exact source boundaries for the fixed layer-two attention graph with borrowed layer-one publication."""
    if (
        receipt.get("capture_status")
        != "completed synthetic source-forward capture; no parity claim"
    ):
        raise RuntimeError(
            "attention fixture export requires a completed source capture"
        )
    coverage = receipt.get("coverage_status")
    if not isinstance(coverage, dict) or coverage.get("pending") != []:
        raise RuntimeError(
            "attention fixture export requires complete capture coverage"
        )
    source, runtime = _source(receipt)
    encoded = receipt.get("encoded_parameters")
    model_args = receipt.get("model_args")
    steps = receipt.get("steps")
    manifest_sha = receipt.get("manifest_canonical_sha256")
    if (
        not isinstance(encoded, dict)
        or not isinstance(model_args, dict)
        or not isinstance(steps, list)
        or not isinstance(manifest_sha, str)
    ):
        raise TypeError("complete capture has an invalid attention-fixture shape")

    parameters = {
        name: _tensor(record, name)
        for name, record in encoded.items()
        if isinstance(name, str) and name.startswith("layers.2.attn.")
    }
    if not parameters:
        raise RuntimeError(
            "complete capture lacks encoded layer-two attention parameters"
        )
    for required in (
        "layers.2.attn.attn_sink",
        "layers.2.attn.wq_a.weight",
        "layers.2.attn.wq_a.scale",
        "layers.2.attn.q_norm.weight",
        "layers.2.attn.wq_b.weight",
        "layers.2.attn.wq_b.scale",
        "layers.2.attn.wkv.weight",
        "layers.2.attn.wkv.scale",
        "layers.2.attn.kv_norm.weight",
        "layers.2.attn.wo_a.weight",
        "layers.2.attn.wo_b.weight",
        "layers.2.attn.wo_b.scale",
    ):
        if required not in parameters:
            raise RuntimeError(
                f"complete capture lacks required attention parameter {required}"
            )
    frequencies: dict[str, object] | None = None

    cases: list[dict[str, object]] = []
    for step in steps:
        if not isinstance(step, dict):
            raise TypeError("complete capture includes an invalid attention step")
        start_pos = step.get("start_pos")
        intermediate = step.get("intermediates")
        sparse_calls = step.get("sparse_attention_calls")
        hc_calls = step.get("hyper_connection_mixes")
        if not isinstance(start_pos, int) or not isinstance(intermediate, dict):
            raise TypeError("complete capture step lacks attention boundaries")
        if not isinstance(sparse_calls, list) or not isinstance(hc_calls, list):
            raise TypeError("complete capture step lacks sparse or HC observations")
        attention_hc = [
            call
            for call in hc_calls
            if isinstance(call, dict)
            and call.get("layer_id") == 2
            and call.get("sublayer") == "attention"
        ]
        if len(attention_hc) != 1 or not isinstance(
            attention_hc[0].get("outputs"), dict
        ):
            raise RuntimeError("complete capture lacks layer-two attention HC gate")
        attention_pre = attention_hc[0]["outputs"].get("pre")
        layer_two_calls = [
            call
            for call in sparse_calls
            if isinstance(call, dict) and call.get("layer_id") == 2
        ]
        if len(layer_two_calls) != 1:
            raise RuntimeError(
                "complete capture must retain exactly one layer-two sparse call"
            )
        sparse = layer_two_calls[0]
        sparse_inputs = sparse.get("inputs")
        if not isinstance(sparse_inputs, dict):
            raise TypeError("layer-two sparse observation lacks inputs")
        window = intermediate.get("layers.2.attn.window")
        compressed = intermediate.get("layers.2.attn.compressed")
        if not isinstance(window, dict) or not isinstance(compressed, dict):
            raise TypeError("complete capture lacks prepared layer-two source caches")
        _sparse_owner_operands(sparse_inputs, window, compressed, "layer-two")
        frequencies_record = _tensor(
            intermediate.get("layers.2.attn.freqs_cis"),
            "layer-two source frequency table",
            dtype="torch.complex64",
        )
        frequencies_current = _complex_f32_pairs(frequencies_record)
        if frequencies is None:
            frequencies = frequencies_current
        elif frequencies != frequencies_current:
            raise RuntimeError("layer-two source frequency table changed between calls")
        layer_one_compressed = intermediate.get("layers.1.attn.compressed")
        if not isinstance(layer_one_compressed, dict):
            raise TypeError("complete capture lacks layer-one KV/index publication")
        cases.append(
            {
                "start_pos": start_pos,
                "input": _tensor(
                    intermediate.get("layers.2.attention_input"),
                    "layers.2.attention_input",
                    dtype="torch.bfloat16",
                ),
                "after_attention_residual": _tensor(
                    intermediate.get("layers.2.after_attention_residual"),
                    "layers.2 post-attention residual",
                    dtype="torch.bfloat16",
                ),
                "attention_pre": _tensor(
                    attention_pre,
                    "layers.2 post-attention pre",
                    dtype="torch.float32",
                ),
                "wq_a_output": _tensor(
                    intermediate.get("layers.2.attn.wq_a"),
                    "layers.2.attn.wq_a",
                    dtype="torch.bfloat16",
                ),
                "q_norm_output": _tensor(
                    intermediate.get("layers.2.attn.q_norm"),
                    "layers.2.attn.q_norm",
                    dtype="torch.bfloat16",
                ),
                "wq_b_pre_rope": _tensor(
                    intermediate.get("layers.2.attn.wq_b"),
                    "layers.2.attn.wq_b",
                    dtype="torch.bfloat16",
                ),
                "q_after_rope": _tensor(
                    sparse_inputs.get("q_after_rope"),
                    "layers.2 sparse q_after_rope",
                    dtype="torch.bfloat16",
                ),
                "window_kv": _tensor(
                    window.get("window_kv"),
                    "layers.2 returned window KV",
                    dtype="torch.bfloat16",
                ),
                "prepared_window_kv": _tensor(
                    window.get("prepared_window_kv"),
                    "layers.2 prepared window KV",
                    dtype="torch.bfloat16",
                ),
                "window_indices": _tensor(
                    window.get("indices"),
                    "layers.2 window indices",
                    dtype="torch.int32",
                ),
                "window_ring_after": _tensor(
                    window.get("ring_after"),
                    "layers.2 window ring",
                    dtype="torch.bfloat16",
                ),
                "layer_one_published_kv": _tensor(
                    layer_one_compressed.get("borrowed_kv"),
                    "layers.1 published compressed KV",
                    dtype="torch.bfloat16",
                ),
                "layer_one_published_indices": _tensor(
                    layer_one_compressed.get("indices"),
                    "layers.1 published compressed indices",
                    dtype="torch.int32",
                ),
                "compressed_kv": _tensor(
                    compressed.get("borrowed_kv"),
                    "layers.2 borrowed compressed KV",
                    dtype="torch.bfloat16",
                ),
                "compressed_indices": _tensor(
                    compressed.get("indices"),
                    "layers.2 compressed indices",
                    dtype="torch.int32",
                ),
                "sparse_output_pre_inverse_rope": _tensor(
                    sparse.get("output_pre_inverse_rope"),
                    "layers.2 sparse output",
                    dtype="torch.bfloat16",
                ),
                "wo_b_input": _tensor(
                    intermediate.get("layers.2.attn.wo_b_input"),
                    "layers.2.attn.wo_b input",
                    dtype="torch.bfloat16",
                ),
                "output": _tensor(
                    intermediate.get("layers.2.attn"),
                    "layers.2.attn output",
                    dtype="torch.bfloat16",
                ),
            }
        )
    if [case["start_pos"] for case in cases] != [0, 5, 6]:
        raise RuntimeError("attention fixture requires the pinned prefill/decode trace")
    for case, sequence in zip(cases, (5, 1, 1), strict=True):
        if case["input"]["shape"] != [1, sequence, 128] or case["output"]["shape"] != [
            1,
            sequence,
            128,
        ]:
            raise RuntimeError("layer-two attention width changed")
        if case["after_attention_residual"]["shape"] != [1, sequence, 2, 128] or case[
            "attention_pre"
        ]["shape"] != [1, sequence, 2]:
            raise RuntimeError("layer-two FFN handoff shape changed")

    model_names = (
        "dim",
        "n_heads",
        "head_dim",
        "rope_head_dim",
        "q_lora_rank",
        "o_groups",
        "o_lora_rank",
        "index_n_heads",
        "index_head_dim",
        "index_topk",
        "window_size",
        "compress_ratios",
        "kv_source_layers",
        "index_source_layers",
        "candidate_source_layer",
        "candidate_topk_blocks",
        "candidate_block_size",
        "rope_theta",
        "compress_rope_theta",
        "original_seq_len",
        "rope_factor",
        "beta_fast",
        "beta_slow",
        "norm_eps",
    )
    missing_model = [name for name in model_names if name not in model_args]
    if missing_model:
        raise RuntimeError(
            f"complete capture lacks attention model args: {missing_model}"
        )
    if (
        tuple(model_args["index_source_layers"]) != (1, 3, 4)
        or 2 in model_args["index_source_layers"]
    ):
        raise RuntimeError(
            "layer-two attention fixture has an unexpected Indexer schedule"
        )
    if frequencies is None:
        raise RuntimeError("layer-two attention fixture has no frequency table")
    for case in cases:
        if (
            case["compressed_kv"]["dtype"] != case["layer_one_published_kv"]["dtype"]
            or case["compressed_kv"]["shape"] != case["layer_one_published_kv"]["shape"]
            or case["compressed_indices"]["dtype"]
            != case["layer_one_published_indices"]["dtype"]
            or case["compressed_indices"]["shape"]
            != case["layer_one_published_indices"]["shape"]
        ):
            raise RuntimeError(
                "layer-two borrowed publication geometry disagrees with layer one"
            )
        if (
            case["compressed_kv"]["storage_sha256"]
            != case["layer_one_published_kv"]["storage_sha256"]
        ):
            raise RuntimeError("layer-two compressed KV is not layer-one publication")
        if (
            case["compressed_indices"]["storage_sha256"]
            != case["layer_one_published_indices"]["storage_sha256"]
        ):
            raise RuntimeError(
                "layer-two compressed indices are not layer-one publication"
            )
    return {
        "schema_version": 1,
        "scope": (
            "layer-two source attention boundaries, including source-prepared local "
            "and layer-one-published compressed KV; not native attention, Rust "
            "acceptance, or full-model parity"
        ),
        "source": {
            "revision": source.get("revision"),
            "model_sha256": source.get("model_sha256"),
            "engram_sha256": source.get("engram_sha256"),
            "kernel_source_sha256": source.get("kernel_source_sha256"),
            "cpu_backend_sha256": source.get("cpu_backend_sha256"),
            "loader_sha256": source.get("loader_sha256"),
            "runner_sha256": source.get("runner_sha256"),
            "forward_observers_sha256": source.get("forward_observers_sha256"),
            "attention_helper_sha256": _sha256_bytes(helper_path.read_bytes()),
            "complete_capture_sha256": _sha256_bytes(_serialized_capture(receipt)),
            "manifest_canonical_sha256": manifest_sha,
            "storage_byteorder": runtime.get("storage_byteorder"),
        },
        "model": {name: model_args[name] for name in model_names},
        "frequency_scope": (
            "full layer-two source freqs_cis schedule captured from the module prehook; exact "
            "complex64 real/imaginary FP32 storage pairs, not recomputed per-case slices"
        ),
        "frequencies": frequencies,
        "encoded_parameters": parameters,
        "cases": cases,
        "comparison_policy": {
            "captured_tensors": "exact source storage bytes; exporter performs no attention arithmetic",
            "q_after_rope_bf16": "exact sparse_attn query operand after source RoPE",
            "sparse_output_pre_inverse_rope_bf16": "exact sparse_attn result cloned before source inverse RoPE",
            "window_kv_and_ring_bf16": "source _window_kv returned read and cache snapshot after its write",
            "prepared_window_kv_bf16": "source _window_kv newly published value; decode is the just-written ring slot",
            "layer_one_published_kv_and_indices": "exact current layer-one ratio-two owner publication used by source layer-two _compress_kv",
            "compressed_kv_and_indices": "source layer-two _compress_kv borrowed layer-one publication; layer two has no Indexer",
            "output_bf16": "exact Attention.forward output after source inverse RoPE and output projections",
            "fixed_before_candidate_execution": True,
        },
    }
