"""Project layer-four source-attention observations into a compact fixture.

This module consumes an already-completed capture from ``v41-forward-reference``.
It deliberately performs no attention arithmetic: every expected tensor is an
exact storage record captured from the pinned source graph.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _serialized_capture(receipt: dict[str, object]) -> bytes:
    return (
        json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _tensor(record: object, label: str, *, dtype: str | None = None) -> dict[str, Any]:
    if not isinstance(record, dict) or not isinstance(record.get("storage_hex"), str):
        raise TypeError(f"attention fixture lacks complete storage for {label}")
    if not isinstance(record.get("shape"), list) or not isinstance(
        record.get("numel"), int
    ):
        raise TypeError(f"attention fixture has invalid tensor metadata for {label}")
    if dtype is not None and record.get("dtype") != dtype:
        raise RuntimeError(
            f"attention fixture expected {label} to be {dtype}, got {record.get('dtype')!r}"
        )
    return record


def _complex_f32_pairs(record: object) -> dict[str, object]:
    """Preserve source complex64 frequencies as exact real/imaginary FP32 bits."""
    value = _tensor(record, "layer_4_freqs_cis", dtype="torch.complex64")
    raw = bytes.fromhex(value["storage_hex"])
    numel = value["numel"]
    if len(raw) != numel * 8:
        raise RuntimeError("layer_4_freqs_cis storage length does not match complex64")
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
    return source, runtime


def attention_fixture(
    receipt: dict[str, object], *, helper_path: Path
) -> dict[str, object]:
    """Export exact source boundaries for the fixed layer-four attention graph."""
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
    static = receipt.get("attention_static")
    manifest_sha = receipt.get("manifest_canonical_sha256")
    if (
        not isinstance(encoded, dict)
        or not isinstance(model_args, dict)
        or not isinstance(steps, list)
        or not isinstance(static, dict)
        or not isinstance(manifest_sha, str)
    ):
        raise TypeError("complete capture has an invalid attention-fixture shape")

    parameters = {
        name: _tensor(record, name)
        for name, record in encoded.items()
        if isinstance(name, str) and name.startswith("layers.4.attn.")
    }
    if not parameters:
        raise RuntimeError(
            "complete capture lacks encoded layer-four attention parameters"
        )
    for required in (
        "layers.4.attn.attn_sink",
        "layers.4.attn.wq_a.weight",
        "layers.4.attn.wq_a.scale",
        "layers.4.attn.q_norm.weight",
        "layers.4.attn.wq_b.weight",
        "layers.4.attn.wq_b.scale",
        "layers.4.attn.wkv.weight",
        "layers.4.attn.wkv.scale",
        "layers.4.attn.kv_norm.weight",
        "layers.4.attn.wo_a.weight",
        "layers.4.attn.wo_b.weight",
        "layers.4.attn.wo_b.scale",
    ):
        if required not in parameters:
            raise RuntimeError(
                f"complete capture lacks required attention parameter {required}"
            )
    frequencies = _complex_f32_pairs(static.get("layer_4_freqs_cis"))

    cases: list[dict[str, object]] = []
    for step in steps:
        if not isinstance(step, dict):
            raise TypeError("complete capture includes an invalid attention step")
        start_pos = step.get("start_pos")
        intermediate = step.get("intermediates")
        sparse_calls = step.get("sparse_attention_calls")
        if not isinstance(start_pos, int) or not isinstance(intermediate, dict):
            raise TypeError("complete capture step lacks attention boundaries")
        if not isinstance(sparse_calls, list):
            raise TypeError("complete capture step lacks sparse attention observations")
        layer_four_calls = [
            call
            for call in sparse_calls
            if isinstance(call, dict) and call.get("layer_id") == 4
        ]
        if len(layer_four_calls) != 1:
            raise RuntimeError(
                "complete capture must retain exactly one layer-four sparse call"
            )
        sparse = layer_four_calls[0]
        sparse_inputs = sparse.get("inputs")
        if not isinstance(sparse_inputs, dict):
            raise TypeError("layer-four sparse observation lacks inputs")
        window = intermediate.get("layers.4.attn.window")
        compressed = intermediate.get("layers.4.attn.compressed")
        if not isinstance(window, dict) or not isinstance(compressed, dict):
            raise TypeError("complete capture lacks prepared layer-four source caches")
        cases.append(
            {
                "start_pos": start_pos,
                "input": _tensor(
                    intermediate.get("layers.4.attention_input"),
                    "layers.4.attention_input",
                    dtype="torch.bfloat16",
                ),
                "wq_a_output": _tensor(
                    intermediate.get("layers.4.attn.wq_a"),
                    "layers.4.attn.wq_a",
                    dtype="torch.bfloat16",
                ),
                "q_norm_output": _tensor(
                    intermediate.get("layers.4.attn.q_norm"),
                    "layers.4.attn.q_norm",
                    dtype="torch.bfloat16",
                ),
                "wq_b_pre_rope": _tensor(
                    intermediate.get("layers.4.attn.wq_b"),
                    "layers.4.attn.wq_b",
                    dtype="torch.bfloat16",
                ),
                "q_after_rope": _tensor(
                    sparse_inputs.get("q_after_rope"),
                    "layers.4 sparse q_after_rope",
                    dtype="torch.bfloat16",
                ),
                "window_kv": _tensor(
                    window.get("window_kv"),
                    "layers.4 returned window KV",
                    dtype="torch.bfloat16",
                ),
                "prepared_window_kv": _tensor(
                    window.get("prepared_window_kv"),
                    "layers.4 prepared window KV",
                    dtype="torch.bfloat16",
                ),
                "window_indices": _tensor(
                    window.get("indices"),
                    "layers.4 window indices",
                    dtype="torch.int32",
                ),
                "window_ring_after": _tensor(
                    window.get("ring_after"),
                    "layers.4 window ring",
                    dtype="torch.bfloat16",
                ),
                "compressed_kv": _tensor(
                    compressed.get("borrowed_kv"),
                    "layers.4 borrowed compressed KV",
                    dtype="torch.bfloat16",
                ),
                "compressed_indices": _tensor(
                    compressed.get("indices"),
                    "layers.4 compressed indices",
                    dtype="torch.int32",
                ),
                "sparse_output_pre_inverse_rope": _tensor(
                    sparse.get("output_pre_inverse_rope"),
                    "layers.4 sparse output",
                    dtype="torch.bfloat16",
                ),
                "wo_b_input": _tensor(
                    intermediate.get("layers.4.attn.wo_b_input"),
                    "layers.4.attn.wo_b input",
                    dtype="torch.bfloat16",
                ),
                "output": _tensor(
                    intermediate.get("layers.4.attn"),
                    "layers.4.attn output",
                    dtype="torch.bfloat16",
                ),
            }
        )
    if [case["start_pos"] for case in cases] != [0, 5, 6]:
        raise RuntimeError("attention fixture requires the pinned prefill/decode trace")

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
    return {
        "schema_version": 1,
        "scope": (
            "layer-four source attention boundaries, including source-prepared local "
            "and layer-three-published compressed KV; not native attention, Rust "
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
            "attention_helper_sha256": _sha256_bytes(helper_path.read_bytes()),
            "complete_capture_sha256": _sha256_bytes(_serialized_capture(receipt)),
            "manifest_canonical_sha256": manifest_sha,
            "storage_byteorder": runtime.get("storage_byteorder"),
        },
        "model": {name: model_args[name] for name in model_names},
        "frequency_scope": (
            "full layer-four source freqs_cis schedule for max_seq_len; exact "
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
            "compressed_kv_and_indices": "source _compress_kv read from shared layer-three publication; indices retain source offset domain",
            "output_bf16": "exact Attention.forward output after source inverse RoPE and output projections",
            "fixed_before_candidate_execution": True,
        },
    }
