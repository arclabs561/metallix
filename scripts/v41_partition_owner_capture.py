#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Project the source-owned layer-three operands from the 4/1/1/1 receipt.

This is a compact, byte-preserving source receipt for a prospective native
owner/compressor/index-key implementation.  It is not a new qualified fixture:
the source execution and its observer noninterference gate remain authoritative.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import struct
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from v41_partition_boundaries import project_bridges

_SCHEDULE = ((0, 4), (4, 1), (5, 1), (6, 1))
_MAX_INPUT_BYTES = 16 << 20
_MAX_OUTPUT_BYTES = 1 << 20
_WIDTHS = {
    "torch.bfloat16": 2,
    "torch.float32": 4,
    "torch.int32": 4,
    "torch.int64": 8,
    "torch.bool": 1,
    "torch.complex64": 8,
    "torch.float8_e4m3fn": 1,
    "torch.float8_e8m0fnu": 1,
}
_PINNED_SOURCE = {
    "cpu_backend_sha256": "b1f1f3cfdb93b674a5f96a114cf45bf5be9ad3a555ae95ac24add567f9f5232e",
    "engram_sha256": "11f35ecbead8150c35aa002b3d180ef290b05a25afe883a11884f94d476d3897",
    "kernel_source_sha256": "1236c3507019ed176f5dba5e04bcea58867cf654818c6cf138ed4845398c2455",
    "loader_sha256": "359c4c961bdc8e200e2ccd13e7499974220a8d6942b5f6627316ab54210bef03",
    "model_sha256": "4e9ae23620edc8028ccc5d5fef552ab7fdc7dcd6f79608754fe9f67644056f65",
    "observer_sha256": "c1cf54629878b928e30bde288e3614873863104fc320f71a05c10cae07c3f292",
    "probe_sha256": "dd8b74bcddb7ef5458d197ff2bf4c3d73f559af4daa9c40c1166ad8cb1c14bc2",
    "revision": "dba1be0a40aa45a94ad051997016db3960a90277",
    "runner_sha256": "7f217a4c42039e6cb55d9914ae095b656ad199a240c585eef11a2d2eace750f0",
}

_ATTENTION_PARAMETERS = {
    "attn_sink": ("torch.float32", [2]),
    "wq_a.weight": ("torch.float8_e4m3fn", [32, 128]),
    "wq_a.scale": ("torch.float8_e8m0fnu", [1, 4]),
    "q_norm.weight": ("torch.bfloat16", [32]),
    "wq_b.weight": ("torch.float8_e4m3fn", [128, 32]),
    "wq_b.scale": ("torch.float8_e8m0fnu", [4, 1]),
    "wkv.weight": ("torch.float8_e4m3fn", [64, 128]),
    "wkv.scale": ("torch.float8_e8m0fnu", [2, 4]),
    "kv_norm.weight": ("torch.bfloat16", [64]),
    "wo_a.weight": ("torch.bfloat16", [64, 64]),
    "wo_b.weight": ("torch.float8_e4m3fn", [128, 64]),
    "wo_b.scale": ("torch.float8_e8m0fnu", [4, 2]),
}

_ATTENTION_MODEL_INTS = (
    "dim",
    "head_dim",
    "n_heads",
    "q_lora_rank",
    "rope_head_dim",
    "window_size",
    "o_groups",
    "o_lora_rank",
    "candidate_source_layer",
)


class CaptureError(ValueError):
    """The source receipt cannot safely produce an owner projection."""


def _object(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CaptureError(f"{label} must be an object")
    return value


def _tensor(
    value: object, label: str, *, negative_infinity_only: bool = False
) -> tuple[Mapping[str, Any], bytes]:
    record = _object(value, label)
    dtype = record.get("dtype")
    shape = record.get("shape")
    numel = record.get("numel")
    storage_hex = record.get("storage_hex")
    digest = record.get("storage_sha256")
    if (
        dtype not in _WIDTHS
        or not isinstance(shape, list)
        or not all(type(size) is int and size >= 0 for size in shape)
        or type(numel) is not int
        or numel < 0
        or not isinstance(storage_hex, str)
        or not isinstance(digest, str)
    ):
        raise CaptureError(f"{label} has malformed tensor metadata")
    if math.prod(shape) != numel:
        raise CaptureError(f"{label} shape does not match numel")
    try:
        raw = bytes.fromhex(storage_hex)
    except ValueError as error:
        raise CaptureError(f"{label} has invalid storage hex") from error
    if len(raw) != numel * _WIDTHS[dtype]:
        raise CaptureError(f"{label} storage length does not match metadata")
    if hashlib.sha256(raw).hexdigest() != digest:
        raise CaptureError(f"{label} storage hash does not match storage hex")
    if dtype == "torch.bfloat16":
        nonfinite = [
            int.from_bytes(raw[offset : offset + 2], "little")
            for offset in range(0, len(raw), 2)
            if (int.from_bytes(raw[offset : offset + 2], "little") >> 7) & 0xFF == 0xFF
        ]
        if negative_infinity_only:
            if (
                record.get("finite") is not False
                or not nonfinite
                or any(word != 0xFF80 for word in nonfinite)
            ):
                raise CaptureError(f"{label} must contain only negative infinities")
        elif record.get("finite") is not True or nonfinite:
            raise CaptureError(f"{label} contains nonfinite bfloat16 storage")
    if dtype == "torch.float32" and (
        record.get("finite") is not True
        or not all(math.isfinite(number) for number in struct.unpack(f"<{numel}f", raw))
    ):
        raise CaptureError(f"{label} contains nonfinite float32 storage")
    if dtype == "torch.complex64" and (
        record.get("finite") is not True
        or not all(
            math.isfinite(number) for number in struct.unpack(f"<{numel * 2}f", raw)
        )
    ):
        raise CaptureError(f"{label} contains nonfinite complex64 storage")
    if dtype == "torch.bool" and (
        record.get("finite") is not True or any(byte not in (0, 1) for byte in raw)
    ):
        raise CaptureError(f"{label} contains invalid bool storage")
    if dtype == "torch.float8_e4m3fn" and (
        record.get("finite") is not True or any(byte & 0x7F == 0x7F for byte in raw)
    ):
        raise CaptureError(f"{label} contains nonfinite E4M3FN storage")
    if dtype == "torch.float8_e8m0fnu" and (
        record.get("finite") is not True or 0xFF in raw
    ):
        raise CaptureError(f"{label} contains nonfinite E8M0FNU storage")
    if dtype in {"torch.int32", "torch.int64"} and record.get("finite") is not True:
        raise CaptureError(f"{label} must be finite")
    return record, raw


def _require_tensor(
    value: object, label: str, dtype: str, shape: list[int]
) -> Mapping[str, Any]:
    record, _ = _tensor(value, label)
    if record["dtype"] != dtype or record["shape"] != shape:
        raise CaptureError(f"{label} must have dtype {dtype} and shape {shape}")
    return record


def _same(left: object, right: object, label: str) -> None:
    left_record, left_bytes = _tensor(left, f"{label} left")
    right_record, right_bytes = _tensor(right, f"{label} right")
    if (
        left_record["dtype"] != right_record["dtype"]
        or left_record["shape"] != right_record["shape"]
        or left_bytes != right_bytes
    ):
        raise CaptureError(f"{label} is not byte-identical")


def _next_layer_one_prefix(calls: list[Mapping[str, Any]], index: int) -> object:
    if index + 1 == len(calls):
        return None
    next_call = calls[index + 1]
    next_inputs = _object(
        _object(
            _object(next_call.get("intermediates"), "next intermediates").get(
                "layers.1.attn.indexer_observation"
            ),
            "next layer-one observation",
        ).get("inputs"),
        "next layer-one inputs",
    )
    if next_inputs.get("latent") is not None:
        return None
    return next_inputs.get("shared_index_k_prefix")


def _validate_raw_source(raw: bytes) -> Mapping[str, Any]:
    if len(raw) > _MAX_INPUT_BYTES:
        raise CaptureError(f"source receipt exceeds {_MAX_INPUT_BYTES} byte cap")
    try:
        decoded = json.loads(
            raw,
            parse_constant=lambda value: (_ for _ in ()).throw(
                CaptureError(f"source receipt contains nonfinite JSON constant {value}")
            ),
        )
    except json.JSONDecodeError as error:
        raise CaptureError("source receipt is not valid JSON") from error
    return _object(decoded, "source receipt")


def partition_owner_fixture(raw: bytes) -> dict[str, object]:
    """Validate and project one complete, observer-controlled source receipt."""
    root = _validate_raw_source(raw)
    if (
        _object(root.get("runtime"), "source runtime").get("storage_byteorder")
        != "little"
    ):
        raise CaptureError("source tensors require little-endian storage")
    bridge = project_bridges(raw)
    source_sha256 = hashlib.sha256(raw).hexdigest()
    if bridge.get("source_receipt_sha256") != source_sha256:
        raise CaptureError("bridge projection does not identify the supplied receipt")

    alternate = _object(root.get("alternate"), "alternate run")
    capture = _object(alternate.get("alternate_capture"), "alternate capture")
    if bridge.get("capture_identity") != capture.get("capture_identity"):
        raise CaptureError(
            "bridge projection capture identity differs from raw capture"
        )
    calls_value = capture.get("calls")
    if not isinstance(calls_value, list) or len(calls_value) != len(_SCHEDULE):
        raise CaptureError("alternate capture must retain four calls")
    calls = [_object(call, "alternate call") for call in calls_value]

    parameters = _object(capture.get("encoded_parameters"), "encoded parameters")
    weight_paths = {
        "wkv": "layers.3.attn.compressor.wkv.weight",
        "compressor_norm": "layers.3.attn.compressor.norm.weight",
        "wk": "layers.3.attn.indexer.wk.weight",
        "key_norm": "layers.3.attn.indexer.k_norm.weight",
    }
    expected_weights = {
        "wkv": [64, 128],
        "compressor_norm": [64],
        "wk": [64, 64],
        "key_norm": [64],
    }
    weights = {
        name: _require_tensor(
            parameters.get(path),
            f"weight {name}",
            "torch.bfloat16",
            expected_weights[name],
        )
        for name, path in weight_paths.items()
    }
    selection_weight_specs = {
        "wq_a_codes": ("layers.3.attn.wq_a.weight", "torch.float8_e4m3fn", [32, 128]),
        "wq_a_scales": ("layers.3.attn.wq_a.scale", "torch.float8_e8m0fnu", [1, 4]),
        "q_norm": ("layers.3.attn.q_norm.weight", "torch.bfloat16", [32]),
        "wq_b_codes": (
            "layers.3.attn.indexer.wq_b.weight",
            "torch.float8_e4m3fn",
            [128, 32],
        ),
        "wq_b_scales": (
            "layers.3.attn.indexer.wq_b.scale",
            "torch.float8_e8m0fnu",
            [4, 1],
        ),
        "weights_proj": (
            "layers.3.attn.indexer.weights_proj.weight",
            "torch.bfloat16",
            [2, 128],
        ),
    }
    selection_weights = {
        name: _require_tensor(
            parameters.get(path), f"selection weight {name}", dtype, shape
        )
        for name, (path, dtype, shape) in selection_weight_specs.items()
    }
    actual_model = _object(capture.get("actual_model_args"), "actual model arguments")
    if (
        any(type(actual_model.get(name)) is not int for name in _ATTENTION_MODEL_INTS)
        or not isinstance(actual_model.get("compress_ratios"), list)
        or any(type(value) is not int for value in actual_model["compress_ratios"])
        or type(actual_model.get("norm_eps")) is not float
    ):
        raise CaptureError("source layer-three attention model has invalid types")
    attention_weights = {
        f"layers.3.attn.{suffix}": _require_tensor(
            parameters.get(f"layers.3.attn.{suffix}"),
            f"layer-three attention weight {suffix}",
            dtype,
            shape,
        )
        for suffix, (dtype, shape) in _ATTENTION_PARAMETERS.items()
    }
    expected_actual_model = {
        "max_batch_size": 1,
        "max_seq_len": 8,
        "dim": 128,
        "head_dim": 64,
        "q_lora_rank": 32,
        "index_n_heads": 2,
        "index_head_dim": 64,
        "candidate_block_size": 1,
        "candidate_topk_blocks": 2,
        "index_topk": 1,
    }
    if any(
        actual_model.get(name) != expected
        for name, expected in expected_actual_model.items()
    ):
        raise CaptureError("source selection model geometry differs")
    if (
        actual_model.get("rope_head_dim") != 32
        or actual_model.get("compress_ratios") != [0, 2, 2, 1, 1]
        or actual_model.get("candidate_source_layer") != 3
    ):
        raise CaptureError("source owner/rotary geometry differs")
    model = {
        "batches": actual_model.get("max_batch_size"),
        "input_dimension": actual_model.get("dim"),
        "latent_dimension": actual_model.get("head_dim"),
        "key_dimension": actual_model.get("index_head_dim"),
        "rope_pairs": 16,
        "cache_capacity": actual_model.get("max_seq_len"),
        "owner_layer": 3,
        "norm_epsilon": actual_model.get("norm_eps"),
    }
    selection_model = {
        "query_rank": actual_model.get("q_lora_rank"),
        "index_heads": actual_model.get("index_n_heads"),
        "candidate_block_size": actual_model.get("candidate_block_size"),
        "candidate_topk_blocks": actual_model.get("candidate_topk_blocks"),
        "index_topk": actual_model.get("index_topk"),
    }

    bridge_calls = bridge.get("calls")
    if not isinstance(bridge_calls, list) or len(bridge_calls) != len(_SCHEDULE):
        raise CaptureError("bridge projection lacks the four validated calls")
    cases: list[dict[str, object]] = []
    expected_frequency: Mapping[str, Any] | None = None
    for index, (call, bridge_call, (start_pos, token_count)) in enumerate(
        zip(calls, bridge_calls, _SCHEDULE, strict=True)
    ):
        if (
            call.get("start_pos") != start_pos
            or call.get("token_count") != token_count
            or _object(bridge_call, "bridge call").get("start_pos") != start_pos
            or _object(bridge_call, "bridge call").get("token_count") != token_count
        ):
            raise CaptureError("calls must use the 0/4/5/6 schedule")
        intermediates = _object(
            call.get("intermediates"), f"call {start_pos} intermediates"
        )
        input_record = _require_tensor(
            intermediates.get("layers.3.attention_input"),
            f"call {start_pos} input",
            "torch.bfloat16",
            [1, token_count, 128],
        )
        projected = _require_tensor(
            intermediates.get("layers.3.attn.compressor.wkv"),
            f"call {start_pos} projected",
            "torch.bfloat16",
            [1, token_count, 64],
        )
        latent = _require_tensor(
            intermediates.get("layers.3.attn.compressor"),
            f"call {start_pos} latent",
            "torch.bfloat16",
            [1, token_count, 64],
        )
        compressed = _object(
            intermediates.get("layers.3.attn.compressed"),
            f"call {start_pos} compressed",
        )
        borrowed = _require_tensor(
            compressed.get("borrowed_kv"),
            f"call {start_pos} borrowed KV prefix",
            "torch.bfloat16",
            [1, start_pos + token_count, 64],
        )
        indexer_inputs = _object(
            _object(
                intermediates.get("layers.3.attn.indexer_observation"),
                f"call {start_pos} indexer observation",
            ).get("inputs"),
            f"call {start_pos} indexer inputs",
        )
        if indexer_inputs.get("start_pos") != start_pos:
            raise CaptureError(f"call {start_pos} indexer start position differs")
        _same(input_record, indexer_inputs.get("x"), f"call {start_pos} indexer input")
        _same(latent, indexer_inputs.get("latent"), f"call {start_pos} indexer latent")
        frequency = _require_tensor(
            indexer_inputs.get("frequency_table"),
            f"call {start_pos} indexer frequency table",
            "torch.complex64",
            [8, 16],
        )
        if expected_frequency is None:
            expected_frequency = frequency
        else:
            _same(expected_frequency, frequency, "layer-three indexer frequency table")
        index_prefix = _require_tensor(
            indexer_inputs.get("shared_index_k_prefix"),
            f"call {start_pos} layer-three index key prefix",
            "torch.bfloat16",
            [1, start_pos + token_count, 64],
        )
        observation = _object(
            intermediates.get("layers.3.attn.indexer_observation"),
            f"call {start_pos} indexer observation",
        )
        operations = _object(
            observation.get("operations"), f"call {start_pos} indexer operations"
        )
        offset = indexer_inputs.get("offset")
        if offset != (4 if start_pos == 0 else 6):
            raise CaptureError(f"call {start_pos} has unexpected indexer offset")
        end_pos = start_pos + token_count
        score_shape = [1, token_count, 2, end_pos]
        reduced_shape = [1, token_count, end_pos]
        selection: dict[str, object] = {
            "offset": offset,
            "wq_a": _require_tensor(
                intermediates.get("layers.3.attn.wq_a"),
                f"call {start_pos} WQ-A output",
                "torch.bfloat16",
                [1, token_count, 32],
            ),
            "qr": _require_tensor(
                indexer_inputs.get("qr"),
                f"call {start_pos} normalized query",
                "torch.bfloat16",
                [1, token_count, 32],
            ),
            "q_after_rope_fp4": _require_tensor(
                operations.get("q_after_rope_fp4"),
                f"call {start_pos} rotary query",
                "torch.bfloat16",
                [1, token_count, 2, 64],
            ),
            "weights_proj_output": _require_tensor(
                operations.get("weights_proj_output"),
                f"call {start_pos} head weights",
                "torch.bfloat16",
                [1, token_count, 2],
            ),
            "scaled_weights": _require_tensor(
                operations.get("scaled_weights"),
                f"call {start_pos} scaled head weights",
                "torch.bfloat16",
                [1, token_count, 2],
            ),
            "dot_products": _require_tensor(
                operations.get("scores_einsum"),
                f"call {start_pos} score dot products",
                "torch.bfloat16",
                score_shape,
            ),
            "rectified": _require_tensor(
                operations.get("scores_after_relu"),
                f"call {start_pos} rectified scores",
                "torch.bfloat16",
                score_shape,
            ),
            "weighted": _require_tensor(
                operations.get("scores_weighted_per_head"),
                f"call {start_pos} weighted scores",
                "torch.bfloat16",
                score_shape,
            ),
            "scores": _require_tensor(
                operations.get("scores_after_head_sum"),
                f"call {start_pos} reduced scores",
                "torch.bfloat16",
                reduced_shape,
            ),
            "candidate_mask": _require_tensor(
                observation.get("candidate_mask_after"),
                f"call {start_pos} candidate mask",
                "torch.bool",
                reduced_shape,
            ),
            "indices": _require_tensor(
                observation.get("output_indices"),
                f"call {start_pos} selected indices",
                "torch.int32",
                [1, token_count, 1],
            ),
        }
        if start_pos == 0:
            causal, _ = _tensor(
                operations.get("scores_after_causal_mask"),
                "call 0 causal scores",
                negative_infinity_only=True,
            )
            if (
                causal.get("dtype") != "torch.bfloat16"
                or causal.get("shape") != reduced_shape
            ):
                raise CaptureError("call 0 causal scores have unexpected geometry")
            selection["causal_scores"] = causal
        elif "scores_after_causal_mask" in operations:
            raise CaptureError("decode calls must not retain causal scores")
        else:
            selection["causal_scores"] = None

        sparse_calls = call.get("sparse_attention_calls")
        if not isinstance(sparse_calls, list):
            raise CaptureError(f"call {start_pos} lacks sparse attention observations")
        layer_three_sparse = [
            sparse
            for sparse in sparse_calls
            if isinstance(sparse, Mapping) and sparse.get("layer_id") == 3
        ]
        if len(layer_three_sparse) != 1:
            raise CaptureError(
                f"call {start_pos} requires one layer-three sparse attention observation"
            )
        sparse = layer_three_sparse[0]
        sparse_inputs = _object(
            sparse.get("inputs"), f"call {start_pos} layer-three sparse inputs"
        )
        window = _object(
            intermediates.get("layers.3.attn.window"),
            f"call {start_pos} layer-three window",
        )
        # Prefill retains its active four rows; source decode reads the fixed
        # six-slot window, including causally masked future physical slots.
        window_positions = (
            token_count if start_pos == 0 else actual_model["window_size"]
        )
        attention = {
            "start_pos": start_pos,
            "input": input_record,
            "wq_a_output": _require_tensor(
                intermediates.get("layers.3.attn.wq_a"),
                f"call {start_pos} attention WQ-A output",
                "torch.bfloat16",
                [1, token_count, 32],
            ),
            "q_norm_output": _require_tensor(
                intermediates.get("layers.3.attn.q_norm"),
                f"call {start_pos} attention normalized query",
                "torch.bfloat16",
                [1, token_count, 32],
            ),
            "wq_b_pre_rope": _require_tensor(
                intermediates.get("layers.3.attn.wq_b"),
                f"call {start_pos} attention WQ-B output",
                "torch.bfloat16",
                [1, token_count, 128],
            ),
            "q_after_rope": _require_tensor(
                sparse_inputs.get("q_after_rope"),
                f"call {start_pos} sparse query after rotary",
                "torch.bfloat16",
                [1, token_count, 2, 64],
            ),
            "prepared_window_kv": _require_tensor(
                window.get("prepared_window_kv"),
                f"call {start_pos} prepared window KV",
                "torch.bfloat16",
                [1, token_count, 64],
            ),
            "window_kv": _require_tensor(
                window.get("window_kv"),
                f"call {start_pos} returned window KV",
                "torch.bfloat16",
                [1, window_positions, 64],
            ),
            "window_indices": _require_tensor(
                window.get("indices"),
                f"call {start_pos} window indices",
                "torch.int32",
                [1, token_count, window_positions],
            ),
            "window_ring_after": _require_tensor(
                window.get("ring_after"),
                f"call {start_pos} window ring",
                "torch.bfloat16",
                [1, actual_model["window_size"], 64],
            ),
            "compressed_kv": borrowed,
            "compressed_indices": _require_tensor(
                compressed.get("indices"),
                f"call {start_pos} compressed indices",
                "torch.int32",
                [1, token_count, 1],
            ),
            "sparse_output_pre_inverse_rope": _require_tensor(
                sparse.get("output_pre_inverse_rope"),
                f"call {start_pos} sparse output",
                "torch.bfloat16",
                [1, token_count, 2, 64],
            ),
            "wo_b_input": _require_tensor(
                intermediates.get("layers.3.attn.wo_b_input"),
                f"call {start_pos} attention WO-B input",
                "torch.bfloat16",
                [1, token_count, 64],
            ),
            "output": _require_tensor(
                intermediates.get("layers.3.attn"),
                f"call {start_pos} attention output",
                "torch.bfloat16",
                [1, token_count, 128],
            ),
        }
        _same(
            selection["qr"],
            intermediates.get("layers.3.attn.q_norm"),
            f"call {start_pos} normalized query boundary",
        )
        _same(
            selection["indices"],
            compressed.get("indices"),
            f"call {start_pos} selected candidate IDs",
        )
        _same(attention["input"], input_record, f"call {start_pos} attention input")
        _same(
            attention["q_norm_output"],
            selection["qr"],
            f"call {start_pos} attention query boundary",
        )
        _same(
            attention["compressed_kv"],
            borrowed,
            f"call {start_pos} attention compressed KV boundary",
        )
        _same(
            attention["compressed_indices"],
            selection["indices"],
            f"call {start_pos} attention selected-ID boundary",
        )
        bridge_intermediates = _object(
            _object(bridge_call, "bridge call").get("intermediates"),
            "bridge intermediates",
        )
        _same(
            index_prefix,
            _object(
                bridge_intermediates.get("layers.3.attn.indexer_observation"),
                "bridge layer-three indexer",
            )["inputs"]["shared_index_k_prefix"],
            f"call {start_pos} bridge layer-three key prefix",
        )
        _same(
            borrowed,
            _object(
                bridge_intermediates.get("layers.3.attn.compressed"),
                "bridge layer-three compressed",
            ).get("borrowed_kv"),
            f"call {start_pos} bridge borrowed KV prefix",
        )
        next_prefix = _next_layer_one_prefix(calls, index)
        if next_prefix is not None:
            _tensor(next_prefix, f"call {start_pos} next layer-one score prefix")
        cases.append(
            {
                "start_pos": start_pos,
                "token_count": token_count,
                "input": input_record,
                "projected": projected,
                "latent": latent,
                "index_key_prefix": index_prefix,
                "compressed_kv_prefix": borrowed,
                "next_layer1_score_prefix": next_prefix,
                "selection": selection,
                "attention": attention,
            }
        )

    attention_static = _object(
        capture.get("attention_static"), "layer-three attention static inputs"
    )
    _same(
        expected_frequency,
        attention_static.get("layer_3_freqs_cis"),
        "layer-three attention and indexer frequency table",
    )

    fixture: dict[str, object] = {
        "schema_version": 1,
        "scope": "Source owner operands for the observed 4/1/1/1 partition; not native qualification",
        "source_receipt_sha256": source_sha256,
        "source": root.get("source"),
        "capture_identity": capture.get("capture_identity"),
        "model": model,
        "weights": weights,
        "selection_model": selection_model,
        "selection_weights": selection_weights,
        "attention_model": actual_model,
        "attention_weights": attention_weights,
        "frequencies": expected_frequency,
        "cases": cases,
    }
    validate_fixture(fixture)
    return fixture


def validate_fixture(fixture: object) -> None:
    """Check fixture provenance, schedule, geometry, and all retained tensor bytes."""
    root = _object(fixture, "fixture")
    if type(root.get("schema_version")) is not int or root.get("schema_version") != 1:
        raise CaptureError("fixture schema version must be 1")
    if (
        not isinstance(root.get("source_receipt_sha256"), str)
        or len(root["source_receipt_sha256"]) != 64
        or any(char not in "0123456789abcdef" for char in root["source_receipt_sha256"])
    ):
        raise CaptureError("fixture lacks a source receipt SHA-256")
    source = _object(root.get("source"), "fixture source")
    for name, expected in _PINNED_SOURCE.items():
        if source.get(name) != expected:
            raise CaptureError(f"fixture source {name} differs from the pinned source")
    identity = _object(root.get("capture_identity"), "fixture capture identity")
    schedule = identity.get("schedule")
    if (
        not isinstance(schedule, list)
        or any(type(count) is not int for count in schedule)
        or schedule != [4, 1, 1, 1]
    ):
        raise CaptureError(
            "fixture capture identity does not pin the alternate schedule"
        )
    payload = {"probe_sha256": source["probe_sha256"], "schedule": schedule}
    expected_identity = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if (
        identity.get("probe_sha256") != source["probe_sha256"]
        or identity.get("sha256") != expected_identity
    ):
        raise CaptureError(
            "capture identity does not bind the pinned probe and schedule"
        )
    model = _object(root.get("model"), "fixture model")
    expected_model = {
        "batches": 1,
        "input_dimension": 128,
        "latent_dimension": 64,
        "key_dimension": 64,
        "rope_pairs": 16,
        "cache_capacity": 8,
        "owner_layer": 3,
        "norm_epsilon": 1e-20,
    }
    if model != expected_model or any(
        type(model.get(key)) is not int
        for key in expected_model
        if key != "norm_epsilon"
    ):
        raise CaptureError(
            "fixture model geometry differs from the pinned source model"
        )
    weights = _object(root.get("weights"), "fixture weights")
    for name, shape in (
        ("wkv", [64, 128]),
        ("compressor_norm", [64]),
        ("wk", [64, 64]),
        ("key_norm", [64]),
    ):
        _require_tensor(
            weights.get(name), f"fixture weight {name}", "torch.bfloat16", shape
        )
    selection_model = _object(root.get("selection_model"), "fixture selection model")
    if selection_model != {
        "query_rank": 32,
        "index_heads": 2,
        "candidate_block_size": 1,
        "candidate_topk_blocks": 2,
        "index_topk": 1,
    } or any(type(value) is not int for value in selection_model.values()):
        raise CaptureError("fixture selection model differs from the pinned source")
    selection_weights = _object(
        root.get("selection_weights"), "fixture selection weights"
    )
    for name, dtype, shape in (
        ("wq_a_codes", "torch.float8_e4m3fn", [32, 128]),
        ("wq_a_scales", "torch.float8_e8m0fnu", [1, 4]),
        ("q_norm", "torch.bfloat16", [32]),
        ("wq_b_codes", "torch.float8_e4m3fn", [128, 32]),
        ("wq_b_scales", "torch.float8_e8m0fnu", [4, 1]),
        ("weights_proj", "torch.bfloat16", [2, 128]),
    ):
        _require_tensor(
            selection_weights.get(name),
            f"fixture selection weight {name}",
            dtype,
            shape,
        )
    attention_model = _object(root.get("attention_model"), "fixture attention model")
    if (
        any(
            type(attention_model.get(name)) is not int for name in _ATTENTION_MODEL_INTS
        )
        or attention_model.get("dim") != 128
        or attention_model.get("head_dim") != 64
        or attention_model.get("n_heads") != 2
        or attention_model.get("q_lora_rank") != 32
        or attention_model.get("rope_head_dim") != 32
        or attention_model.get("window_size") != 6
        or attention_model.get("o_groups") != 2
        or attention_model.get("o_lora_rank") != 32
        or attention_model.get("candidate_source_layer") != 3
        or attention_model.get("compress_ratios") != [0, 2, 2, 1, 1]
        or any(type(value) is not int for value in attention_model["compress_ratios"])
        or type(attention_model.get("norm_eps")) is not float
        or attention_model["norm_eps"] != 1.0e-20
    ):
        raise CaptureError("fixture attention model differs from the pinned source")
    attention_weights = _object(
        root.get("attention_weights"), "fixture attention weights"
    )
    for suffix, (dtype, shape) in _ATTENTION_PARAMETERS.items():
        _require_tensor(
            attention_weights.get(f"layers.3.attn.{suffix}"),
            f"fixture attention weight {suffix}",
            dtype,
            shape,
        )
    _require_tensor(
        root.get("frequencies"), "fixture frequencies", "torch.complex64", [8, 16]
    )
    cases = root.get("cases")
    if not isinstance(cases, list) or len(cases) != len(_SCHEDULE):
        raise CaptureError("fixture must retain four partition cases")
    for case, (start_pos, token_count) in zip(cases, _SCHEDULE, strict=True):
        item = _object(case, f"fixture case {start_pos}")
        if (
            type(item.get("start_pos")) is not int
            or type(item.get("token_count")) is not int
            or item.get("start_pos") != start_pos
            or item.get("token_count") != token_count
        ):
            raise CaptureError("fixture cases do not use the pinned schedule")
        _require_tensor(
            item.get("input"),
            f"case {start_pos} input",
            "torch.bfloat16",
            [1, token_count, 128],
        )
        _require_tensor(
            item.get("projected"),
            f"case {start_pos} projected",
            "torch.bfloat16",
            [1, token_count, 64],
        )
        _require_tensor(
            item.get("latent"),
            f"case {start_pos} latent",
            "torch.bfloat16",
            [1, token_count, 64],
        )
        _require_tensor(
            item.get("index_key_prefix"),
            f"case {start_pos} index key prefix",
            "torch.bfloat16",
            [1, start_pos + token_count, 64],
        )
        selection = _object(item.get("selection"), f"case {start_pos} selection")
        if selection.get("offset") != (4 if start_pos == 0 else 6):
            raise CaptureError(f"case {start_pos} selection offset differs")
        score_shape = [1, token_count, 2, start_pos + token_count]
        reduced_shape = [1, token_count, start_pos + token_count]
        for name, dtype, shape in (
            ("wq_a", "torch.bfloat16", [1, token_count, 32]),
            ("qr", "torch.bfloat16", [1, token_count, 32]),
            ("q_after_rope_fp4", "torch.bfloat16", [1, token_count, 2, 64]),
            ("weights_proj_output", "torch.bfloat16", [1, token_count, 2]),
            ("scaled_weights", "torch.bfloat16", [1, token_count, 2]),
            ("dot_products", "torch.bfloat16", score_shape),
            ("rectified", "torch.bfloat16", score_shape),
            ("weighted", "torch.bfloat16", score_shape),
            ("scores", "torch.bfloat16", reduced_shape),
            ("candidate_mask", "torch.bool", reduced_shape),
            ("indices", "torch.int32", [1, token_count, 1]),
        ):
            _require_tensor(
                selection.get(name), f"case {start_pos} selection {name}", dtype, shape
            )
        causal = selection.get("causal_scores")
        if start_pos == 0:
            causal_record, _ = _tensor(
                causal, "case 0 selection causal scores", negative_infinity_only=True
            )
            if (
                causal_record.get("dtype") != "torch.bfloat16"
                or causal_record.get("shape") != reduced_shape
            ):
                raise CaptureError("case 0 causal score geometry differs")
        elif causal is not None:
            raise CaptureError(f"case {start_pos} must not retain causal scores")
        attention = _object(item.get("attention"), f"case {start_pos} attention")
        if attention.get("start_pos") != start_pos:
            raise CaptureError(f"case {start_pos} attention start differs")
        window_positions = token_count if start_pos == 0 else 6
        for name, dtype, shape in (
            ("input", "torch.bfloat16", [1, token_count, 128]),
            ("wq_a_output", "torch.bfloat16", [1, token_count, 32]),
            ("q_norm_output", "torch.bfloat16", [1, token_count, 32]),
            ("wq_b_pre_rope", "torch.bfloat16", [1, token_count, 128]),
            ("q_after_rope", "torch.bfloat16", [1, token_count, 2, 64]),
            ("prepared_window_kv", "torch.bfloat16", [1, token_count, 64]),
            ("window_kv", "torch.bfloat16", [1, window_positions, 64]),
            ("window_indices", "torch.int32", [1, token_count, window_positions]),
            ("window_ring_after", "torch.bfloat16", [1, 6, 64]),
            ("compressed_kv", "torch.bfloat16", [1, start_pos + token_count, 64]),
            ("compressed_indices", "torch.int32", [1, token_count, 1]),
            (
                "sparse_output_pre_inverse_rope",
                "torch.bfloat16",
                [1, token_count, 2, 64],
            ),
            ("wo_b_input", "torch.bfloat16", [1, token_count, 64]),
            ("output", "torch.bfloat16", [1, token_count, 128]),
        ):
            _require_tensor(
                attention.get(name), f"case {start_pos} attention {name}", dtype, shape
            )
        _same(
            attention.get("input"),
            item.get("input"),
            f"case {start_pos} attention input",
        )
        _same(
            attention.get("q_norm_output"),
            selection.get("qr"),
            f"case {start_pos} attention query boundary",
        )
        _same(
            attention.get("compressed_kv"),
            item.get("compressed_kv_prefix"),
            f"case {start_pos} attention compressed KV boundary",
        )
        _same(
            attention.get("compressed_indices"),
            selection.get("indices"),
            f"case {start_pos} attention selected-ID boundary",
        )
        _require_tensor(
            item.get("compressed_kv_prefix"),
            f"case {start_pos} borrowed KV prefix",
            "torch.bfloat16",
            [1, start_pos + token_count, 64],
        )
        handoff = item.get("next_layer1_score_prefix")
        if start_pos in (0, 5):
            _, index_raw = _tensor(
                item.get("index_key_prefix"), f"case {start_pos} index key prefix"
            )
            _, handoff_raw = _tensor(
                handoff, f"case {start_pos} next layer-one score prefix"
            )
            expected_shape = [1, 2 if start_pos == 0 else 3, 64]
            _require_tensor(
                handoff,
                f"case {start_pos} next layer-one score prefix",
                "torch.bfloat16",
                expected_shape,
            )
            if not index_raw.startswith(handoff_raw):
                raise CaptureError(
                    f"case {start_pos} layer-one score prefix is not the layer-three key prefix"
                )
        elif handoff is not None:
            raise CaptureError(
                f"case {start_pos} must not retain a layer-one score prefix"
            )


def _read_input(path: Path) -> bytes:
    if not path.is_file():
        raise CaptureError("input must be a regular file")
    size = path.stat().st_size
    if size > _MAX_INPUT_BYTES:
        raise CaptureError(f"source receipt exceeds {_MAX_INPUT_BYTES} byte cap")
    return path.read_bytes()


def _write_new(path: Path, fixture: Mapping[str, object]) -> tuple[int, str]:
    encoded = (
        json.dumps(fixture, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()
    if len(encoded) > _MAX_OUTPUT_BYTES:
        raise CaptureError(f"fixture exceeds {_MAX_OUTPUT_BYTES} byte cap")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError as error:
        raise CaptureError("output already exists; refusing to overwrite") from error
    with os.fdopen(descriptor, "wb") as output:
        output.write(encoded)
    return len(encoded), hashlib.sha256(encoded).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw = _read_input(args.input)
    fixture = partition_owner_fixture(raw)
    size, digest = _write_new(args.output, fixture)
    print(
        json.dumps(
            {"bytes": size, "schema_version": 1, "sha256": digest, "status": "created"},
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
