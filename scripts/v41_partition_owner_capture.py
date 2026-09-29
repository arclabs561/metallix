#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Project source operands for alternate layer-three native qualification.

The compact, byte-preserving fixture retains the L3 owner, selection, attention,
and post-attention block boundaries from the observed 4/1/1/1 receipt. It is a
source capture, not a claim that native execution or the source runner is canonical.
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
_MAX_OUTPUT_BYTES = 2 << 20
_WIDTHS = {
    "torch.bfloat16": 2,
    "torch.float32": 4,
    "torch.int32": 4,
    "torch.int64": 8,
    "torch.bool": 1,
    "torch.complex64": 8,
    "torch.float8_e4m3fn": 1,
    "torch.float8_e8m0fnu": 1,
    "torch.float4_e2m1fn_x2": 1,
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

_POST_ATTENTION_MODEL = {
    "dim": 128,
    "hc_mult": 2,
    "moe_inter_dim": 128,
    "n_routed_experts": 4,
    "n_activated_experts": 2,
    "n_shared_experts": 1,
    "score_func": "sqrtsoftplus",
    "gate_temp": 1.0,
    "norm_topk_prob": True,
    "route_scale": 1.0,
    "swiglu_limit": 0.0,
    "expert_dtype": "fp4",
}

_BLOCK_PARAMETER_SPECS = {
    "layers.3.hc_attn_fn": ("torch.float32", [8, 256]),
    "layers.3.hc_attn_base": ("torch.float32", [8]),
    "layers.3.hc_attn_scale": ("torch.float32", [3]),
    "layers.3.hc_ffn_fn": ("torch.float32", [8, 256]),
    "layers.3.hc_ffn_base": ("torch.float32", [8]),
    "layers.3.hc_ffn_scale": ("torch.float32", [3]),
    "layers.3.attn_norm.weight": ("torch.bfloat16", [128]),
    "layers.3.ffn_norm.weight": ("torch.bfloat16", [128]),
}

_LAYER_FOUR_BLOCK_PARAMETER_SPECS = {
    "layers.4.hc_attn_fn": ("torch.float32", [8, 256]),
    "layers.4.hc_attn_base": ("torch.float32", [8]),
    "layers.4.hc_attn_scale": ("torch.float32", [3]),
    "layers.4.hc_ffn_fn": ("torch.float32", [8, 256]),
    "layers.4.hc_ffn_base": ("torch.float32", [8]),
    "layers.4.hc_ffn_scale": ("torch.float32", [3]),
    "layers.4.attn_norm.weight": ("torch.bfloat16", [128]),
    "layers.4.ffn_norm.weight": ("torch.bfloat16", [128]),
}


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
    if dtype == "torch.float4_e2m1fn_x2" and record.get("finite") is not True:
        raise CaptureError(f"{label} must be finite")
    return record, raw


def _require_tensor(
    value: object, label: str, dtype: str, shape: list[int]
) -> Mapping[str, Any]:
    record, _ = _tensor(value, label)
    if record["dtype"] != dtype or record["shape"] != shape:
        raise CaptureError(f"{label} must have dtype {dtype} and shape {shape}")
    return record


def _require_negative_infinity_mask(
    value: object, label: str, shape: list[int]
) -> Mapping[str, Any]:
    record, _ = _tensor(value, label, negative_infinity_only=True)
    if record["dtype"] != "torch.bfloat16" or record["shape"] != shape:
        raise CaptureError(f"{label} must have bfloat16 shape {shape}")
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


def _post_attention_parameters(
    encoded: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Keep the complete L3 MoE and HC parameter subset consumed by the block tail."""
    ffn = "layers.3.ffn"
    routed = tuple(f"{ffn}.experts.{index}" for index in range(4))
    experts = (*routed, f"{ffn}.shared_experts")
    names = {
        f"{ffn}.gate.weight",
        f"{ffn}.gate.bias",
        *(
            f"{expert}.{projection}.{field}"
            for expert in experts
            for projection in ("w1", "w2", "w3")
            for field in ("weight", "scale")
        ),
    }
    parameters = {name: encoded.get(name) for name in names}
    if any(value is None for value in parameters.values()):
        raise CaptureError(
            "source layer-three post-attention path lacks MoE parameters"
        )
    specs: dict[str, tuple[str, list[int]]] = {
        f"{ffn}.gate.weight": ("torch.bfloat16", [4, 128]),
        f"{ffn}.gate.bias": ("torch.float32", [4]),
    }
    for expert in routed:
        for projection in ("w1", "w2", "w3"):
            specs[f"{expert}.{projection}.weight"] = (
                "torch.float4_e2m1fn_x2",
                [128, 64],
            )
            specs[f"{expert}.{projection}.scale"] = ("torch.float8_e8m0fnu", [128, 4])
    for projection in ("w1", "w2", "w3"):
        specs[f"{ffn}.shared_experts.{projection}.weight"] = (
            "torch.float8_e4m3fn",
            [128, 128],
        )
        specs[f"{ffn}.shared_experts.{projection}.scale"] = (
            "torch.float8_e8m0fnu",
            [4, 4],
        )
    for name, (dtype, shape) in specs.items():
        _require_tensor(
            parameters[name], f"layer-three MoE parameter {name}", dtype, shape
        )
    block = {name: encoded.get(name) for name in _BLOCK_PARAMETER_SPECS}
    for name, (dtype, shape) in _BLOCK_PARAMETER_SPECS.items():
        _require_tensor(
            block[name], f"layer-three block parameter {name}", dtype, shape
        )
    return parameters, block


def _layer_four_parameters(
    encoded: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Keep exactly the layer-four MoE and HC parameters consumed downstream."""
    ffn = "layers.4.ffn"
    routed = tuple(f"{ffn}.experts.{index}" for index in range(4))
    experts = (*routed, f"{ffn}.shared_experts")
    names = {
        f"{ffn}.gate.weight",
        f"{ffn}.gate.bias",
        *(
            f"{expert}.{projection}.{field}"
            for expert in experts
            for projection in ("w1", "w2", "w3")
            for field in ("weight", "scale")
        ),
    }
    parameters = {name: encoded.get(name) for name in names}
    if any(value is None for value in parameters.values()):
        raise CaptureError("source layer-four path lacks MoE parameters")
    specs: dict[str, tuple[str, list[int]]] = {
        f"{ffn}.gate.weight": ("torch.bfloat16", [4, 128]),
        f"{ffn}.gate.bias": ("torch.float32", [4]),
    }
    for expert in routed:
        for projection in ("w1", "w2", "w3"):
            specs[f"{expert}.{projection}.weight"] = (
                "torch.float4_e2m1fn_x2",
                [128, 64],
            )
            specs[f"{expert}.{projection}.scale"] = ("torch.float8_e8m0fnu", [128, 4])
    for projection in ("w1", "w2", "w3"):
        specs[f"{ffn}.shared_experts.{projection}.weight"] = (
            "torch.float8_e4m3fn",
            [128, 128],
        )
        specs[f"{ffn}.shared_experts.{projection}.scale"] = (
            "torch.float8_e8m0fnu",
            [4, 4],
        )
    for name, (dtype, shape) in specs.items():
        _require_tensor(
            parameters[name], f"layer-four MoE parameter {name}", dtype, shape
        )
    block = {name: encoded.get(name) for name in _LAYER_FOUR_BLOCK_PARAMETER_SPECS}
    if any(value is None for value in block.values()):
        raise CaptureError("source layer-four path lacks HC parameters")
    for name, (dtype, shape) in _LAYER_FOUR_BLOCK_PARAMETER_SPECS.items():
        _require_tensor(block[name], f"layer-four block parameter {name}", dtype, shape)
    return parameters, block


def _coefficients(value: object, label: str, positions: int) -> Mapping[str, Any]:
    coefficients = _object(value, label)
    if set(coefficients) != {"pre", "post", "comb"}:
        raise CaptureError(f"{label} lacks complete HC coefficients")
    for name, shape in (
        ("pre", [1, positions, 2]),
        ("post", [1, positions, 2]),
        ("comb", [1, positions, 2, 2]),
    ):
        _require_tensor(
            coefficients.get(name), f"{label} {name}", "torch.float32", shape
        )
    return coefficients


def _validate_post_attention(
    root: Mapping[str, Any], owner_cases: list[object]
) -> None:
    post = _object(root.get("post_attention"), "fixture post-attention")
    if post.get("source") != root.get("source"):
        raise CaptureError(
            "post-attention source provenance differs from fixture source"
        )
    if post.get("source_receipt_sha256") != root.get("source_receipt_sha256"):
        raise CaptureError(
            "post-attention receipt identity differs from fixture source"
        )
    if post.get("capture_identity") != root.get("capture_identity"):
        raise CaptureError(
            "post-attention capture identity differs from fixture source"
        )
    model = _object(post.get("model"), "fixture post-attention model")
    if model != _POST_ATTENTION_MODEL or any(
        type(model.get(name)) is not type(expected)
        for name, expected in _POST_ATTENTION_MODEL.items()
    ):
        raise CaptureError(
            "fixture post-attention model differs from the pinned source"
        )
    config = _object(post.get("block_config"), "fixture post-attention HC config")
    expected_config = {
        "copies": 2,
        "hc_sinkhorn_iters": 20,
        "hc_eps": 1.0e-6,
        "norm_eps": 1.0e-20,
    }
    if config != expected_config or any(
        type(config.get(name)) is not type(expected)
        for name, expected in expected_config.items()
    ):
        raise CaptureError("fixture post-attention HC configuration differs")
    encoded = _object(post.get("encoded_parameters"), "fixture post-attention MoE")
    block = _object(
        post.get("block_parameters"), "fixture post-attention block parameters"
    )
    if set(encoded) & set(block) or set(block) != set(_BLOCK_PARAMETER_SPECS):
        raise CaptureError(
            "post-attention parameter maps overlap or misplace block keys"
        )
    expected_encoded, _ = _post_attention_parameters({**encoded, **block})
    if set(encoded) != set(expected_encoded):
        raise CaptureError(
            "post-attention MoE parameter keys differ from the source contract"
        )
    policy = _object(post.get("comparison_policy"), "fixture post-attention policy")
    expected_policy = {
        "output_bf16": "exact storage bits",
        "route_weight_abs_error_max": 9.5367431640625e-07,
        "fixed_before_candidate_execution": True,
        "block_next_pre_abs_error_max": 9.5367431640625e-07,
    }
    if policy != expected_policy or any(
        type(policy.get(name)) is not type(expected)
        for name, expected in expected_policy.items()
    ):
        raise CaptureError("fixture post-attention comparison policy differs")
    cases = post.get("cases")
    if not isinstance(cases, list) or len(cases) != len(_SCHEDULE):
        raise CaptureError("fixture post-attention must retain four cases")
    for owner_case, case, (start_pos, token_count) in zip(
        owner_cases, cases, _SCHEDULE, strict=True
    ):
        owner = _object(owner_case, f"case {start_pos} owner")
        item = _object(case, f"case {start_pos} post-attention")
        if type(item.get("start_pos")) is not int or item.get("start_pos") != start_pos:
            raise CaptureError("post-attention cases do not use the pinned schedule")
        for name, dtype, shape in (
            ("input", "torch.bfloat16", [1, token_count, 128]),
            ("gate_weights", "torch.float32", [token_count, 2]),
            ("gate_indices", "torch.int64", [token_count, 2]),
            ("output", "torch.bfloat16", [1, token_count, 128]),
            ("block_input", "torch.bfloat16", [1, token_count, 2, 128]),
            ("block_incoming_pre", "torch.float32", [1, token_count, 2]),
            ("attention_input", "torch.bfloat16", [1, token_count, 128]),
            ("attention_output", "torch.bfloat16", [1, token_count, 128]),
            ("after_attention_residual", "torch.bfloat16", [1, token_count, 2, 128]),
            ("attention_hc_mixes", "torch.float32", [1, token_count, 8]),
            ("ffn_collapsed", "torch.bfloat16", [1, token_count, 128]),
            ("ffn_hc_mixes", "torch.float32", [1, token_count, 8]),
            ("block_output", "torch.bfloat16", [1, token_count, 2, 128]),
            ("block_next_pre", "torch.float32", [1, token_count, 2]),
        ):
            _require_tensor(
                item.get(name), f"case {start_pos} post-attention {name}", dtype, shape
            )
        _coefficients(
            item.get("attention_coefficients"),
            f"case {start_pos} attention HC",
            token_count,
        )
        _coefficients(
            item.get("ffn_coefficients"), f"case {start_pos} FFN HC", token_count
        )
        next_entry = _object(
            item.get("next_block_entry"), f"case {start_pos} layer-four entry"
        )
        _require_tensor(
            next_entry.get("residual"),
            f"case {start_pos} layer-four residual",
            "torch.bfloat16",
            [1, token_count, 2, 128],
        )
        _require_tensor(
            next_entry.get("incoming_pre"),
            f"case {start_pos} layer-four pre",
            "torch.float32",
            [1, token_count, 2],
        )
        attention = _object(owner.get("attention"), f"case {start_pos} attention")
        _same(
            item.get("attention_input"),
            attention.get("input"),
            f"case {start_pos} attention-to-block input",
        )
        _same(
            item.get("attention_output"),
            attention.get("output"),
            f"case {start_pos} attention-to-block output",
        )
        _same(
            item.get("block_output"),
            next_entry.get("residual"),
            f"case {start_pos} layer-three-to-four residual",
        )
        _same(
            item.get("block_next_pre"),
            next_entry.get("incoming_pre"),
            f"case {start_pos} layer-three-to-four pre",
        )


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


def _post_layer_three_projection(
    *,
    root_source: object,
    source_receipt_sha256: str,
    capture_identity: object,
    actual_model: Mapping[str, Any],
    parameters: Mapping[str, Any],
    attention_static: Mapping[str, Any],
    calls: list[Mapping[str, Any]],
    owner_cases: list[dict[str, object]],
    layer_three_tail_cases: list[dict[str, object]],
) -> dict[str, object]:
    """Project L4's observed consumer path without inventing canonical provenance."""
    selection_weight_specs = {
        "wq_a_codes": ("layers.4.attn.wq_a.weight", "torch.float8_e4m3fn", [32, 128]),
        "wq_a_scales": ("layers.4.attn.wq_a.scale", "torch.float8_e8m0fnu", [1, 4]),
        "q_norm": ("layers.4.attn.q_norm.weight", "torch.bfloat16", [32]),
        "wq_b_codes": (
            "layers.4.attn.indexer.wq_b.weight",
            "torch.float8_e4m3fn",
            [128, 32],
        ),
        "wq_b_scales": (
            "layers.4.attn.indexer.wq_b.scale",
            "torch.float8_e8m0fnu",
            [4, 1],
        ),
        "weights_proj": (
            "layers.4.attn.indexer.weights_proj.weight",
            "torch.bfloat16",
            [2, 128],
        ),
    }
    selection_weights = {
        name: _require_tensor(
            parameters.get(path), f"layer-four selection {name}", dtype, shape
        )
        for name, (path, dtype, shape) in selection_weight_specs.items()
    }
    attention_weights = {
        f"layers.4.attn.{suffix}": _require_tensor(
            parameters.get(f"layers.4.attn.{suffix}"),
            f"layer-four attention weight {suffix}",
            dtype,
            shape,
        )
        for suffix, (dtype, shape) in _ATTENTION_PARAMETERS.items()
    }
    encoded_parameters, block_parameters = _layer_four_parameters(parameters)
    block_config = {
        "copies": actual_model.get("hc_mult"),
        "hc_sinkhorn_iters": actual_model.get("hc_sinkhorn_iters"),
        "hc_eps": actual_model.get("hc_eps"),
        "norm_eps": actual_model.get("norm_eps"),
    }
    if (
        block_config
        != {"copies": 2, "hc_sinkhorn_iters": 20, "hc_eps": 1.0e-6, "norm_eps": 1.0e-20}
        or type(block_config["copies"]) is not int
        or type(block_config["hc_sinkhorn_iters"]) is not int
        or type(block_config["hc_eps"]) is not float
        or type(block_config["norm_eps"]) is not float
    ):
        raise CaptureError("source layer-four HC configuration differs")
    static = _object(actual_model, "layer-four source model")
    selection_model = {
        "query_rank": static.get("q_lora_rank"),
        "index_heads": static.get("index_n_heads"),
        "candidate_block_size": static.get("candidate_block_size"),
        "candidate_topk_blocks": static.get("candidate_topk_blocks"),
        "index_topk": static.get("index_topk"),
    }
    if selection_model != {
        "query_rank": 32,
        "index_heads": 2,
        "candidate_block_size": 1,
        "candidate_topk_blocks": 2,
        "index_topk": 1,
    } or any(type(value) is not int for value in selection_model.values()):
        raise CaptureError("source layer-four selection geometry differs")
    attention_cases: list[dict[str, object]] = []
    selection_cases: list[dict[str, object]] = []
    tail_cases: list[dict[str, object]] = []
    head_cases: list[dict[str, object]] = []
    expected_frequency: Mapping[str, Any] | None = None
    for call, owner_case, prior_tail, (start_pos, token_count) in zip(
        calls, owner_cases, layer_three_tail_cases, _SCHEDULE, strict=True
    ):
        intermediates = _object(
            call.get("intermediates"), f"call {start_pos} intermediates"
        )
        observation = _object(
            intermediates.get("layers.4.attn.indexer_observation"),
            f"call {start_pos} layer-four indexer observation",
        )
        inputs = _object(
            observation.get("inputs"), f"call {start_pos} layer-four indexer inputs"
        )
        operations = _object(
            observation.get("operations"),
            f"call {start_pos} layer-four indexer operations",
        )
        if inputs.get("start_pos") != start_pos or inputs.get("offset") != (
            4 if start_pos == 0 else 6
        ):
            raise CaptureError(
                f"call {start_pos} layer-four selection geometry differs"
            )
        frequency = _require_tensor(
            inputs.get("frequency_table"),
            f"call {start_pos} layer-four frequency table",
            "torch.complex64",
            [8, 16],
        )
        if expected_frequency is None:
            expected_frequency = frequency
        else:
            _same(expected_frequency, frequency, "layer-four selection frequency table")
        owner = _object(owner_case, f"call {start_pos} layer-three owner")
        owner_selection = _object(
            owner.get("selection"), f"call {start_pos} layer-three selection"
        )
        key_prefix = _require_tensor(
            inputs.get("shared_index_k_prefix"),
            f"call {start_pos} layer-four shared keys",
            "torch.bfloat16",
            [1, start_pos + token_count, 64],
        )
        candidate_mask = _require_tensor(
            inputs.get("candidate_mask"),
            f"call {start_pos} layer-four candidate mask",
            "torch.bool",
            [1, token_count, start_pos + token_count],
        )
        _same(
            key_prefix,
            owner.get("index_key_prefix"),
            f"call {start_pos} L3-to-L4 key prefix",
        )
        _same(
            candidate_mask,
            owner_selection.get("candidate_mask"),
            f"call {start_pos} L3 candidate set to L4",
        )
        score_shape = [1, token_count, 2, start_pos + token_count]
        reduced_shape = [1, token_count, start_pos + token_count]
        selection: dict[str, object] = {
            "start_pos": start_pos,
            "offset": inputs.get("offset"),
            "wq_a": _require_tensor(
                intermediates.get("layers.4.attn.wq_a"),
                f"call {start_pos} layer-four WQ-A",
                "torch.bfloat16",
                [1, token_count, 32],
            ),
            "qr": _require_tensor(
                inputs.get("qr"),
                f"call {start_pos} layer-four QR",
                "torch.bfloat16",
                [1, token_count, 32],
            ),
            "q_after_rope_fp4": _require_tensor(
                operations.get("q_after_rope_fp4"),
                f"call {start_pos} layer-four rotary query",
                "torch.bfloat16",
                [1, token_count, 2, 64],
            ),
            "weights_proj_output": _require_tensor(
                operations.get("weights_proj_output"),
                f"call {start_pos} layer-four head weights",
                "torch.bfloat16",
                [1, token_count, 2],
            ),
            "scaled_weights": _require_tensor(
                operations.get("scaled_weights"),
                f"call {start_pos} layer-four scaled head weights",
                "torch.bfloat16",
                [1, token_count, 2],
            ),
            "dot_products": _require_tensor(
                operations.get("scores_einsum"),
                f"call {start_pos} layer-four dot products",
                "torch.bfloat16",
                score_shape,
            ),
            "rectified": _require_tensor(
                operations.get("scores_after_relu"),
                f"call {start_pos} layer-four rectified scores",
                "torch.bfloat16",
                score_shape,
            ),
            "weighted": _require_tensor(
                operations.get("scores_weighted_per_head"),
                f"call {start_pos} layer-four weighted scores",
                "torch.bfloat16",
                score_shape,
            ),
            "scores": _require_tensor(
                operations.get("scores_after_head_sum"),
                f"call {start_pos} layer-four scores",
                "torch.bfloat16",
                reduced_shape,
            ),
            "scores_after_candidate_mask": _require_negative_infinity_mask(
                operations.get("scores_after_candidate_mask"),
                f"call {start_pos} layer-four masked scores",
                reduced_shape,
            ),
            "candidate_mask": candidate_mask,
            "indices": _require_tensor(
                observation.get("output_indices"),
                f"call {start_pos} layer-four selected IDs",
                "torch.int32",
                [1, token_count, 1],
            ),
        }
        causal = operations.get("scores_after_causal_mask")
        if start_pos == 0:
            selection["causal_scores"] = _require_negative_infinity_mask(
                causal, f"call {start_pos} layer-four causal scores", reduced_shape
            )
        elif causal is not None:
            raise CaptureError(
                f"call {start_pos} must not retain layer-four causal scores"
            )
        else:
            selection["causal_scores"] = None
        compressed = _object(
            intermediates.get("layers.4.attn.compressed"),
            f"call {start_pos} layer-four compressed",
        )
        window = _object(
            intermediates.get("layers.4.attn.window"),
            f"call {start_pos} layer-four window",
        )
        sparse_calls = call.get("sparse_attention_calls")
        if not isinstance(sparse_calls, list):
            raise CaptureError(f"call {start_pos} lacks layer-four sparse attention")
        sparse = [
            item
            for item in sparse_calls
            if isinstance(item, Mapping) and item.get("layer_id") == 4
        ]
        if len(sparse) != 1:
            raise CaptureError(
                f"call {start_pos} must retain one layer-four sparse call"
            )
        sparse_inputs = _object(
            sparse[0].get("inputs"), f"call {start_pos} layer-four sparse inputs"
        )
        window_positions = token_count if start_pos == 0 else 6
        attention = {
            "start_pos": start_pos,
            "input": _require_tensor(
                intermediates.get("layers.4.attention_input"),
                f"call {start_pos} layer-four attention input",
                "torch.bfloat16",
                [1, token_count, 128],
            ),
            "wq_a_output": selection["wq_a"],
            "q_norm_output": selection["qr"],
            "wq_b_pre_rope": _require_tensor(
                intermediates.get("layers.4.attn.wq_b"),
                f"call {start_pos} layer-four WQ-B",
                "torch.bfloat16",
                [1, token_count, 128],
            ),
            "q_after_rope": _require_tensor(
                sparse_inputs.get("q_after_rope"),
                f"call {start_pos} layer-four sparse query",
                "torch.bfloat16",
                [1, token_count, 2, 64],
            ),
            "prepared_window_kv": _require_tensor(
                window.get("prepared_window_kv"),
                f"call {start_pos} layer-four prepared window KV",
                "torch.bfloat16",
                [1, token_count, 64],
            ),
            "window_kv": _require_tensor(
                window.get("window_kv"),
                f"call {start_pos} layer-four window KV",
                "torch.bfloat16",
                [1, window_positions, 64],
            ),
            "window_indices": _require_tensor(
                window.get("indices"),
                f"call {start_pos} layer-four window indices",
                "torch.int32",
                [1, token_count, window_positions],
            ),
            "window_ring_after": _require_tensor(
                window.get("ring_after"),
                f"call {start_pos} layer-four window ring",
                "torch.bfloat16",
                [1, 6, 64],
            ),
            "compressed_kv": _require_tensor(
                compressed.get("borrowed_kv"),
                f"call {start_pos} layer-four borrowed KV",
                "torch.bfloat16",
                [1, start_pos + token_count, 64],
            ),
            "compressed_indices": _require_tensor(
                compressed.get("indices"),
                f"call {start_pos} layer-four compressed indices",
                "torch.int32",
                [1, token_count, 1],
            ),
            "sparse_output_pre_inverse_rope": _require_tensor(
                sparse[0].get("output_pre_inverse_rope"),
                f"call {start_pos} layer-four sparse output",
                "torch.bfloat16",
                [1, token_count, 2, 64],
            ),
            "wo_b_input": _require_tensor(
                intermediates.get("layers.4.attn.wo_b_input"),
                f"call {start_pos} layer-four WO-B input",
                "torch.bfloat16",
                [1, token_count, 64],
            ),
            "output": _require_tensor(
                intermediates.get("layers.4.attn"),
                f"call {start_pos} layer-four attention output",
                "torch.bfloat16",
                [1, token_count, 128],
            ),
        }
        _same(
            attention["input"],
            intermediates.get("layers.4.attention_input"),
            f"call {start_pos} L4 attention input",
        )
        _same(
            attention["q_norm_output"],
            selection["qr"],
            f"call {start_pos} L4 attention query",
        )
        _same(
            attention["compressed_kv"],
            owner.get("compressed_kv_prefix"),
            f"call {start_pos} L3-to-L4 KV prefix",
        )
        _same(
            attention["compressed_indices"],
            selection["indices"],
            f"call {start_pos} L4 selected IDs to attention",
        )
        hc_calls = call.get("hyper_connection_mixes")
        if not isinstance(hc_calls, list):
            raise CaptureError(f"call {start_pos} lacks layer-four HC observations")
        layer_hc = {
            item.get("sublayer"): item
            for item in hc_calls
            if isinstance(item, Mapping) and item.get("layer_id") == 4
        }
        if set(layer_hc) != {"attention", "ffn"}:
            raise CaptureError(f"call {start_pos} lacks layer-four HC sublayers")
        attention_hc = _object(
            layer_hc["attention"], f"call {start_pos} layer-four attention HC"
        )
        ffn_hc = _object(layer_hc["ffn"], f"call {start_pos} layer-four FFN HC")
        terminal = intermediates.get("layers.4")
        gate = intermediates.get("layers.4.ffn.gate")
        block_input = _object(
            intermediates.get("layers.4.block_input"),
            f"call {start_pos} layer-four block input",
        )
        if (
            not isinstance(terminal, list)
            or len(terminal) != 2
            or not isinstance(gate, list)
            or len(gate) != 2
        ):
            raise CaptureError(f"call {start_pos} lacks layer-four terminal boundaries")
        tail = {
            "start_pos": start_pos,
            "input": _require_tensor(
                intermediates.get("layers.4.ffn_input"),
                f"call {start_pos} layer-four MoE input",
                "torch.bfloat16",
                [1, token_count, 128],
            ),
            "gate_weights": _require_tensor(
                gate[0],
                f"call {start_pos} layer-four gate weights",
                "torch.float32",
                [token_count, 2],
            ),
            "gate_indices": _require_tensor(
                gate[1],
                f"call {start_pos} layer-four gate indices",
                "torch.int64",
                [token_count, 2],
            ),
            "output": _require_tensor(
                intermediates.get("layers.4.ffn"),
                f"call {start_pos} layer-four MoE output",
                "torch.bfloat16",
                [1, token_count, 128],
            ),
            "block_input": _require_tensor(
                block_input.get("residual"),
                f"call {start_pos} layer-four residual entry",
                "torch.bfloat16",
                [1, token_count, 2, 128],
            ),
            "block_incoming_pre": _require_tensor(
                block_input.get("incoming_pre"),
                f"call {start_pos} layer-four incoming pre",
                "torch.float32",
                [1, token_count, 2],
            ),
            "attention_input": attention["input"],
            "attention_output": attention["output"],
            "after_attention_residual": _require_tensor(
                intermediates.get("layers.4.after_attention_residual"),
                f"call {start_pos} layer-four post-attention residual",
                "torch.bfloat16",
                [1, token_count, 2, 128],
            ),
            "attention_hc_mixes": _require_tensor(
                _object(
                    attention_hc.get("inputs"),
                    f"call {start_pos} layer-four attention HC inputs",
                ).get("mixes"),
                f"call {start_pos} layer-four attention HC mixes",
                "torch.float32",
                [1, token_count, 8],
            ),
            "attention_coefficients": _coefficients(
                attention_hc.get("outputs"),
                f"call {start_pos} layer-four attention HC",
                token_count,
            ),
            "ffn_collapsed": _require_tensor(
                intermediates.get("layers.4.ffn_collapsed"),
                f"call {start_pos} layer-four collapsed FFN input",
                "torch.bfloat16",
                [1, token_count, 128],
            ),
            "ffn_hc_mixes": _require_tensor(
                _object(
                    ffn_hc.get("inputs"), f"call {start_pos} layer-four FFN HC inputs"
                ).get("mixes"),
                f"call {start_pos} layer-four FFN HC mixes",
                "torch.float32",
                [1, token_count, 8],
            ),
            "ffn_coefficients": _coefficients(
                ffn_hc.get("outputs"),
                f"call {start_pos} layer-four FFN HC",
                token_count,
            ),
            "block_output": _require_tensor(
                terminal[0],
                f"call {start_pos} layer-four terminal residual",
                "torch.bfloat16",
                [1, token_count, 2, 128],
            ),
            "block_next_pre": _require_tensor(
                terminal[1],
                f"call {start_pos} layer-four terminal pre",
                "torch.float32",
                [1, token_count, 2],
            ),
        }
        _same(
            tail["block_input"],
            prior_tail["next_block_entry"]["residual"],
            f"call {start_pos} L3-to-L4 residual",
        )
        _same(
            tail["block_incoming_pre"],
            prior_tail["next_block_entry"]["incoming_pre"],
            f"call {start_pos} L3-to-L4 pre",
        )
        _same(
            tail["attention_input"],
            attention["input"],
            f"call {start_pos} L4 attention-to-tail input",
        )
        _same(
            tail["attention_output"],
            attention["output"],
            f"call {start_pos} L4 attention-to-tail output",
        )
        norm_input = _require_tensor(
            intermediates.get("norm_input"),
            f"call {start_pos} final norm input",
            "torch.bfloat16",
            [1, token_count, 128],
        )
        head_cases.append(
            {
                "start_pos": start_pos,
                "norm_input": norm_input,
                "norm": _require_tensor(
                    intermediates.get("norm"),
                    f"call {start_pos} final norm output",
                    "torch.bfloat16",
                    [1, token_count, 128],
                ),
                "logits": _require_tensor(
                    intermediates.get("head"),
                    f"call {start_pos} head logits",
                    "torch.float32",
                    [1, 8],
                ),
            }
        )
        attention_cases.append(attention)
        selection_cases.append(selection)
        tail_cases.append(tail)
    if expected_frequency is None:
        raise CaptureError("source layer-four frequency table is absent")
    _same(
        expected_frequency,
        attention_static.get("layer_4_freqs_cis"),
        "layer-four attention and indexer frequency table",
    )
    return {
        "source": root_source,
        "source_receipt_sha256": source_receipt_sha256,
        "capture_identity": capture_identity,
        "attention": {
            "schema_version": 1,
            "source": root_source,
            "model": actual_model,
            "frequencies": expected_frequency,
            "encoded_parameters": attention_weights,
            "cases": attention_cases,
            "comparison_policy": {
                "fixed_before_candidate_execution": True,
                "output_bf16": "exact storage bits",
                "compressed_kv_and_indices": "live layer-three publication and layer-four computed IDs",
            },
        },
        "selection": {
            "model": selection_model,
            "weights": selection_weights,
            "cases": selection_cases,
        },
        "tail": {
            "model": {name: actual_model[name] for name in _POST_ATTENTION_MODEL},
            "block_config": block_config,
            "encoded_parameters": encoded_parameters,
            "block_parameters": block_parameters,
            "cases": tail_cases,
            "comparison_policy": {
                "output_bf16": "exact storage bits",
                "route_weight_abs_error_max": 9.5367431640625e-07,
                "fixed_before_candidate_execution": True,
                "block_next_pre_abs_error_max": 9.5367431640625e-07,
            },
        },
        "head": {
            "norm_weight": _require_tensor(
                parameters.get("norm.weight"),
                "final norm weight",
                "torch.bfloat16",
                [128],
            ),
            "head_weight": _require_tensor(
                parameters.get("head.weight"), "head weight", "torch.float32", [8, 128]
            ),
            "norm_epsilon": actual_model.get("norm_eps"),
            "cases": head_cases,
        },
    }


def _same_metadata(actual: object, expected: object, label: str) -> None:
    # JSON encodings distinguish booleans, integers and floating-point scalars.
    if json.dumps(actual, sort_keys=True, allow_nan=False) != json.dumps(
        expected, sort_keys=True, allow_nan=False
    ):
        raise CaptureError(f"{label} differs from the source contract")


def _tensor_geometry_like(
    actual: object, expected: Mapping[str, Any], label: str
) -> None:
    _require_tensor(actual, label, expected["dtype"], expected["shape"])


def _validate_suffix_case(
    owner: Mapping[str, Any],
    prior_tail: Mapping[str, Any],
    attention: Mapping[str, Any],
    selection: Mapping[str, Any],
    tail: Mapping[str, Any],
    head: Mapping[str, Any],
) -> None:
    start = owner["start_pos"]
    positions = owner["token_count"]
    for name, case in (
        ("attention", attention),
        ("selection", selection),
        ("tail", tail),
        ("head", head),
    ):
        _same_metadata(case.get("start_pos"), start, f"L4 {name} call position")
    _same_metadata(
        selection.get("offset"), owner["selection"]["offset"], "L4 selection offset"
    )
    for name, expected in owner["attention"].items():
        if name != "start_pos":
            _tensor_geometry_like(attention.get(name), expected, f"L4 attention {name}")
    for name, expected in owner["selection"].items():
        if name in ("offset", "causal_scores"):
            continue
        _tensor_geometry_like(selection.get(name), expected, f"L4 selection {name}")
    shape = [1, positions, start + positions]
    if start == 0:
        _require_negative_infinity_mask(
            selection.get("causal_scores"), "L4 causal scores", shape
        )
    elif selection.get("causal_scores") is not None:
        raise CaptureError("L4 decode cannot contain causal scores")
    _require_negative_infinity_mask(
        selection.get("scores_after_candidate_mask"), "L4 masked scores", shape
    )
    for name, expected in prior_tail.items():
        if name in ("start_pos", "next_block_entry"):
            continue
        if name in ("attention_coefficients", "ffn_coefficients"):
            _coefficients(tail.get(name), f"L4 {name}", positions)
        else:
            _tensor_geometry_like(tail.get(name), expected, f"L4 tail {name}")
    for name in ("norm_input", "norm"):
        _require_tensor(
            head.get(name), f"head {name}", "torch.bfloat16", [1, positions, 128]
        )
    _require_tensor(head.get("logits"), "head logits", "torch.float32", [1, 8])
    for actual, expected, label in (
        (
            selection.get("candidate_mask"),
            owner["selection"]["candidate_mask"],
            "L3-to-L4 candidates",
        ),
        (attention.get("compressed_kv"), owner["compressed_kv_prefix"], "L3-to-L4 KV"),
        (
            attention.get("compressed_indices"),
            selection.get("indices"),
            "L4 selected IDs",
        ),
        (attention.get("wq_a_output"), selection.get("wq_a"), "L4 WQ-A boundary"),
        (attention.get("q_norm_output"), selection.get("qr"), "L4 QR boundary"),
        (tail.get("block_input"), prior_tail["block_output"], "L3-to-L4 residual"),
        (tail.get("block_incoming_pre"), prior_tail["block_next_pre"], "L3-to-L4 pre"),
        (tail.get("attention_input"), attention.get("input"), "L4 attention input"),
        (tail.get("attention_output"), attention.get("output"), "L4 attention output"),
    ):
        _same(actual, expected, label)


def _validate_post_layer_three(root: Mapping[str, Any]) -> None:
    suffix = _object(root.get("post_layer_three"), "post-layer-three projection")
    for name in ("source", "source_receipt_sha256", "capture_identity"):
        _same_metadata(suffix.get(name), root[name], f"suffix {name}")
    attention = _object(suffix.get("attention"), "L4 attention")
    selection = _object(suffix.get("selection"), "L4 selection")
    tail = _object(suffix.get("tail"), "L4 tail")
    head = _object(suffix.get("head"), "head")
    _same_metadata(attention.get("schema_version"), 1, "L4 attention schema")
    _same_metadata(attention.get("source"), root["source"], "L4 attention source")
    _same_metadata(
        attention.get("model"), root["attention_model"], "L4 attention model"
    )
    _same_metadata(
        attention.get("comparison_policy"),
        {
            "fixed_before_candidate_execution": True,
            "output_bf16": "exact storage bits",
            "compressed_kv_and_indices": "live layer-three publication and layer-four computed IDs",
        },
        "L4 attention policy",
    )
    _same(
        attention.get("frequencies"),
        root["frequencies"],
        "observed L3 and L4 rotary tables",
    )
    _same_metadata(
        selection.get("model"), root["selection_model"], "L4 selection model"
    )
    weights = _object(selection.get("weights"), "L4 selection weights")
    if set(weights) != set(root["selection_weights"]):
        raise CaptureError("L4 selection weight keys differ")
    for name, expected in root["selection_weights"].items():
        _tensor_geometry_like(weights[name], expected, f"L4 selection weight {name}")
    attention_weights = _object(
        attention.get("encoded_parameters"), "L4 attention weights"
    )
    expected_weights = {
        name.replace("layers.3.", "layers.4."): value
        for name, value in root["attention_weights"].items()
    }
    if set(attention_weights) != set(expected_weights):
        raise CaptureError("L4 attention weight keys differ")
    for name, expected in expected_weights.items():
        _tensor_geometry_like(attention_weights[name], expected, name)
    for query_name, attention_name in (
        ("wq_a_codes", "wq_a.weight"),
        ("wq_a_scales", "wq_a.scale"),
        ("q_norm", "q_norm.weight"),
    ):
        _same(
            weights[query_name],
            attention_weights[f"layers.4.attn.{attention_name}"],
            "L4 query weights",
        )
    for name in ("model", "block_config", "comparison_policy"):
        _same_metadata(tail.get(name), root["post_attention"][name], f"L4 tail {name}")
    encoded = _object(tail.get("encoded_parameters"), "L4 MoE parameters")
    block = _object(tail.get("block_parameters"), "L4 block parameters")
    if set(encoded) & set(block) or set(block) != set(
        _LAYER_FOUR_BLOCK_PARAMETER_SPECS
    ):
        raise CaptureError("L4 parameter maps overlap or misplace block keys")
    expected_encoded, _ = _layer_four_parameters({**encoded, **block})
    if set(encoded) != set(expected_encoded):
        raise CaptureError("L4 MoE parameter keys differ")
    _require_tensor(
        head.get("norm_weight"), "head norm weight", "torch.bfloat16", [128]
    )
    _require_tensor(head.get("head_weight"), "head weight", "torch.float32", [8, 128])
    _same_metadata(
        head.get("norm_epsilon"), root["model"]["norm_epsilon"], "head epsilon"
    )
    for name, section in (
        ("attention", attention),
        ("selection", selection),
        ("tail", tail),
        ("head", head),
    ):
        if not isinstance(section.get("cases"), list) or len(section["cases"]) != len(
            _SCHEDULE
        ):
            raise CaptureError(f"L4 {name} must retain four calls")
    for records in zip(
        root["cases"],
        root["post_attention"]["cases"],
        attention["cases"],
        selection["cases"],
        tail["cases"],
        head["cases"],
        strict=True,
    ):
        _validate_suffix_case(*(_object(record, "L4 call") for record in records))


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
    if any(
        actual_model.get(name) != value for name, value in _POST_ATTENTION_MODEL.items()
    ):
        raise CaptureError("source layer-three post-attention model differs")
    post_attention_parameters, post_attention_block_parameters = (
        _post_attention_parameters(parameters)
    )
    post_attention_config = {
        "copies": actual_model.get("hc_mult"),
        "hc_sinkhorn_iters": actual_model.get("hc_sinkhorn_iters"),
        "hc_eps": actual_model.get("hc_eps"),
        "norm_eps": actual_model.get("norm_eps"),
    }
    if (
        post_attention_config["copies"] != 2
        or post_attention_config["hc_sinkhorn_iters"] != 20
        or type(post_attention_config["hc_eps"]) is not float
        or post_attention_config["hc_eps"] != 1.0e-6
        or type(post_attention_config["norm_eps"]) is not float
        or post_attention_config["norm_eps"] != 1.0e-20
    ):
        raise CaptureError("source layer-three post-attention HC configuration differs")
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
    post_attention_cases: list[dict[str, object]] = []
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
        hc_calls = call.get("hyper_connection_mixes")
        if not isinstance(hc_calls, list):
            raise CaptureError(f"call {start_pos} lacks HC observations")
        layer_three_hc = {
            item.get("sublayer"): item
            for item in hc_calls
            if isinstance(item, Mapping) and item.get("layer_id") == 3
        }
        if set(layer_three_hc) != {"attention", "ffn"}:
            raise CaptureError(f"call {start_pos} lacks layer-three HC sublayers")
        attention_hc = layer_three_hc["attention"]
        ffn_hc = layer_three_hc["ffn"]
        attention_coefficients = _coefficients(
            attention_hc.get("outputs"), f"call {start_pos} attention HC", token_count
        )
        ffn_coefficients = _coefficients(
            ffn_hc.get("outputs"), f"call {start_pos} FFN HC", token_count
        )
        attention_hc_mixes = _require_tensor(
            _object(
                attention_hc.get("inputs"), f"call {start_pos} attention HC inputs"
            ).get("mixes"),
            f"call {start_pos} attention HC mixes",
            "torch.float32",
            [1, token_count, 8],
        )
        ffn_hc_mixes = _require_tensor(
            _object(ffn_hc.get("inputs"), f"call {start_pos} FFN HC inputs").get(
                "mixes"
            ),
            f"call {start_pos} FFN HC mixes",
            "torch.float32",
            [1, token_count, 8],
        )
        block_input = _object(
            intermediates.get("layers.3.block_input"),
            f"call {start_pos} layer-three block input",
        )
        next_block_entry = _object(
            intermediates.get("layers.4.block_input"),
            f"call {start_pos} layer-four block entry",
        )
        terminal = intermediates.get("layers.3")
        if not isinstance(terminal, list) or len(terminal) != 2:
            raise CaptureError(f"call {start_pos} lacks layer-three terminal state")
        gate = intermediates.get("layers.3.ffn.gate")
        if not isinstance(gate, list) or len(gate) != 2:
            raise CaptureError(f"call {start_pos} lacks layer-three MoE gate")
        post_attention = {
            "start_pos": start_pos,
            "input": _require_tensor(
                intermediates.get("layers.3.ffn_input"),
                f"call {start_pos} layer-three MoE input",
                "torch.bfloat16",
                [1, token_count, 128],
            ),
            "gate_weights": _require_tensor(
                gate[0],
                f"call {start_pos} layer-three gate weights",
                "torch.float32",
                [token_count, 2],
            ),
            "gate_indices": _require_tensor(
                gate[1],
                f"call {start_pos} layer-three gate indices",
                "torch.int64",
                [token_count, 2],
            ),
            "output": _require_tensor(
                intermediates.get("layers.3.ffn"),
                f"call {start_pos} layer-three MoE output",
                "torch.bfloat16",
                [1, token_count, 128],
            ),
            "block_input": _require_tensor(
                block_input.get("residual"),
                f"call {start_pos} layer-three residual entry",
                "torch.bfloat16",
                [1, token_count, 2, 128],
            ),
            "block_incoming_pre": _require_tensor(
                block_input.get("incoming_pre"),
                f"call {start_pos} layer-three incoming pre",
                "torch.float32",
                [1, token_count, 2],
            ),
            "attention_input": input_record,
            "attention_output": attention["output"],
            "after_attention_residual": _require_tensor(
                intermediates.get("layers.3.after_attention_residual"),
                f"call {start_pos} post-attention residual",
                "torch.bfloat16",
                [1, token_count, 2, 128],
            ),
            "attention_hc_mixes": attention_hc_mixes,
            "attention_coefficients": attention_coefficients,
            "ffn_collapsed": _require_tensor(
                intermediates.get("layers.3.ffn_collapsed"),
                f"call {start_pos} FFN collapsed input",
                "torch.bfloat16",
                [1, token_count, 128],
            ),
            "ffn_hc_mixes": ffn_hc_mixes,
            "ffn_coefficients": ffn_coefficients,
            "block_output": _require_tensor(
                terminal[0],
                f"call {start_pos} layer-three terminal residual",
                "torch.bfloat16",
                [1, token_count, 2, 128],
            ),
            "block_next_pre": _require_tensor(
                terminal[1],
                f"call {start_pos} layer-three terminal pre",
                "torch.float32",
                [1, token_count, 2],
            ),
            "next_block_entry": {
                "residual": _require_tensor(
                    next_block_entry.get("residual"),
                    f"call {start_pos} layer-four residual entry",
                    "torch.bfloat16",
                    [1, token_count, 2, 128],
                ),
                "incoming_pre": _require_tensor(
                    next_block_entry.get("incoming_pre"),
                    f"call {start_pos} layer-four incoming pre",
                    "torch.float32",
                    [1, token_count, 2],
                ),
            },
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
        _same(
            post_attention["attention_input"],
            attention["input"],
            f"call {start_pos} attention-to-block input",
        )
        _same(
            post_attention["attention_output"],
            attention["output"],
            f"call {start_pos} attention-to-block output",
        )
        _same(
            post_attention["block_output"],
            post_attention["next_block_entry"]["residual"],
            f"call {start_pos} layer-three-to-four residual",
        )
        _same(
            post_attention["block_next_pre"],
            post_attention["next_block_entry"]["incoming_pre"],
            f"call {start_pos} layer-three-to-four pre",
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
        post_attention_cases.append(post_attention)

    attention_static = _object(
        capture.get("attention_static"), "layer-three attention static inputs"
    )
    _same(
        expected_frequency,
        attention_static.get("layer_3_freqs_cis"),
        "layer-three attention and indexer frequency table",
    )
    post_layer_three = _post_layer_three_projection(
        root_source=root.get("source"),
        source_receipt_sha256=source_sha256,
        capture_identity=capture.get("capture_identity"),
        actual_model=actual_model,
        parameters=parameters,
        attention_static=attention_static,
        calls=calls,
        owner_cases=cases,
        layer_three_tail_cases=post_attention_cases,
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
        "post_attention": {
            "source": root.get("source"),
            "source_receipt_sha256": source_sha256,
            "capture_identity": capture.get("capture_identity"),
            "model": {name: actual_model[name] for name in _POST_ATTENTION_MODEL},
            "block_config": post_attention_config,
            "encoded_parameters": post_attention_parameters,
            "block_parameters": post_attention_block_parameters,
            "cases": post_attention_cases,
            "comparison_policy": {
                "output_bf16": "exact storage bits",
                "route_weight_abs_error_max": 9.5367431640625e-07,
                "fixed_before_candidate_execution": True,
                "block_next_pre_abs_error_max": 9.5367431640625e-07,
            },
        },
        "post_layer_three": post_layer_three,
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

    _validate_post_attention(root, cases)
    _validate_post_layer_three(root)


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
