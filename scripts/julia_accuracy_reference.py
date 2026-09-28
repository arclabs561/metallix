#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["torch==2.13.0", "transformers==5.0.0"]
# ///
"""Opt-in F64 accuracy evidence for the bounded Julia full-prefill fixture.

This is a diagnostic only.  It keeps the frozen F32 tensor generator and inputs,
casts those exact values to F64, and spells out the encoder and decision head
without calling the pinned forward path.  It never changes the strict F32 gate.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
from types import ModuleType
from typing import Any

import torch

ROOT = Path(__file__).resolve().parent.parent
FIXTURE = ROOT / "fixtures/julia-1/full-prefill-reference.json"
MANIFEST = ROOT / "fixtures/julia-1/accuracy-cases.json"
MANIFEST_SHA256 = "e41b492e7ec8e0b0545515eda40fae63d4b1f87ea8fb54545acb277911f62b82"
WIDTH, HEADS, HEAD_DIM, FF, LAYERS = 384, 6, 64, 1152, 22
LEGACY = (
    "padded_base",
    "padded_changed_rows",
    "unmasked_control",
    "no_padding",
    "masked_marker",
)


def load(name: str, filename: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / filename)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {filename}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


FULL = load("julia_full_prefill_reference", "julia_full_prefill_reference.py")


def f64(value: torch.Tensor) -> torch.Tensor:
    """Cast a generated F32 value; never regenerate it directly in F64."""
    if value.dtype != torch.float32:
        raise TypeError("accuracy reference expects generated F32 tensors")
    return value.to(torch.float64)


def require_f64(value: torch.Tensor, operation: str) -> None:
    if value.dtype != torch.float64:
        raise TypeError(f"{operation} must remain F64")


def norm(value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    require_f64(value, "layer norm input")
    require_f64(weight, "layer norm weight")
    mean = value.mean(dim=-1, keepdim=True)
    variance = (value - mean).square().mean(dim=-1, keepdim=True)
    return (value - mean) * torch.rsqrt(variance + 1e-5) * weight


def rope(
    value: torch.Tensor, source_f32_coefficients: bool, wrong_theta: bool = False
) -> torch.Tensor:
    require_f64(value, "rotary input")
    coefficient_dtype = torch.float32 if source_f32_coefficients else torch.float64
    positions = torch.arange(value.shape[0], dtype=coefficient_dtype)[:, None]
    theta = 10000.0 if wrong_theta else 160000.0
    frequencies = 1.0 / theta ** (
        torch.arange(0, HEAD_DIM, 2, dtype=coefficient_dtype) / HEAD_DIM
    )
    phase = positions * frequencies[None, :]
    cos = torch.cat((phase, phase), dim=-1).cos().to(torch.float64)[:, None, :]
    sin = torch.cat((phase, phase), dim=-1).sin().to(torch.float64)[:, None, :]
    return (
        value * cos
        + torch.cat((-value[..., HEAD_DIM // 2 :], value[..., : HEAD_DIM // 2]), dim=-1)
        * sin
    )


def encoder_block(
    value: torch.Tensor,
    mask: torch.Tensor,
    layer_id: int,
    weights: dict[str, torch.Tensor],
    source_f32_rope: bool,
    wrong_rope_theta: bool,
) -> torch.Tensor:
    require_f64(value, "encoder block input")
    attention_input = value if layer_id == 0 else norm(value, f64(weights["attn_norm"]))
    qkv = (attention_input @ f64(weights["wqkv"]).transpose(0, 1)).reshape(
        value.shape[0], 3, HEADS, HEAD_DIM
    )
    query, key, val = qkv.unbind(dim=1)
    query, key = (
        rope(query, source_f32_rope, wrong_rope_theta),
        rope(key, source_f32_rope, wrong_rope_theta),
    )
    scores = torch.einsum("qhd,khd->hqk", query, key) / math.sqrt(HEAD_DIM)
    require_f64(scores, "encoder attention scores")
    allowed = mask[None, :].expand(value.shape[0], -1).clone()
    if layer_id % 3:
        positions = torch.arange(value.shape[0])
        allowed &= (positions[:, None] - positions[None, :]).abs() <= 64
    scores = scores.masked_fill(~allowed[None, :, :], torch.finfo(torch.float32).min)
    attended = torch.einsum("hqk,khd->qhd", scores.softmax(dim=-1), val).reshape_as(
        value
    )
    residual = value + attended @ f64(weights["attn_wo"]).transpose(0, 1)
    normalized = norm(residual, f64(weights["mlp_norm"]))
    first, gate = (normalized @ f64(weights["wi"]).transpose(0, 1)).chunk(2, dim=-1)
    return residual + (torch.nn.functional.gelu(first) * gate) @ f64(
        weights["mlp_wo"]
    ).transpose(0, 1)


def head_layer(
    value: torch.Tensor, layer: torch.nn.TransformerEncoderLayer, padding: torch.Tensor
) -> torch.Tensor:
    require_f64(value, "head layer input")
    attention = layer.self_attn
    normalized = norm(value, f64(layer.norm1.weight)) + f64(layer.norm1.bias)
    qkv = normalized @ f64(attention.in_proj_weight).transpose(0, 1) + f64(
        attention.in_proj_bias
    )
    query, key, val = qkv.chunk(3, dim=-1)
    tokens = value.shape[0]
    query = query.reshape(tokens, HEADS, HEAD_DIM).transpose(0, 1)
    key = key.reshape(tokens, HEADS, HEAD_DIM).transpose(0, 1)
    val = val.reshape(tokens, HEADS, HEAD_DIM).transpose(0, 1)
    scores = query @ key.transpose(-2, -1) / math.sqrt(HEAD_DIM)
    require_f64(scores, "head attention scores")
    scores = scores.masked_fill(padding[None, None, :], float("-inf"))
    attended = scores.softmax(dim=-1) @ val
    attended = attended.transpose(0, 1).reshape(tokens, WIDTH)
    residual = (
        value
        + attended @ f64(attention.out_proj.weight).transpose(0, 1)
        + f64(attention.out_proj.bias)
    )
    normalized = norm(residual, f64(layer.norm2.weight)) + f64(layer.norm2.bias)
    feed_forward = torch.relu(
        normalized @ f64(layer.linear1.weight).transpose(0, 1) + f64(layer.linear1.bias)
    )
    return (
        residual
        + feed_forward @ f64(layer.linear2.weight).transpose(0, 1)
        + f64(layer.linear2.bias)
    )


def softmax_probabilities(
    scores: torch.Tensor, marker_mask: torch.Tensor
) -> torch.Tensor:
    require_f64(scores, "probability scores")
    probabilities = torch.zeros_like(scores)
    probabilities[marker_mask] = scores[marker_mask].softmax(dim=-1)
    return probabilities


def f64_case(
    case: dict[str, Any],
    source_f32_rope: bool,
    *,
    invert_mask: bool = False,
    omit_final_norm: bool = False,
    wrong_layer_mapping: bool = False,
    wrong_rope_theta: bool = False,
) -> dict[str, torch.Tensor | list[torch.Tensor]]:
    input_ids = torch.tensor(case["input_ids"], dtype=torch.int64)
    mask = torch.tensor(case["attention_mask"], dtype=torch.bool)
    if invert_mask:
        mask = ~mask
    value = f64(FULL.ENCODER.values((FULL.VOCAB, WIDTH), 200)[input_ids])
    value = norm(value, f64(FULL.near_one(201)))
    stages = [value]
    for layer_id in range(LAYERS):
        weight_layer = (layer_id + 1) % LAYERS if wrong_layer_mapping else layer_id
        value = encoder_block(
            value,
            mask,
            layer_id,
            FULL.layer_weights(weight_layer),
            source_f32_rope,
            wrong_rope_theta,
        )
        stages.append(value)
    hidden = value if omit_final_norm else norm(value, f64(FULL.near_one(202)))
    stages.append(hidden)

    head, _ = FULL.HEAD.build_source_oracle()
    value = hidden + f64(head.type_emb.weight[case["qtype"]])[None, :]
    padding = ~mask
    for layer in head.head.layers:
        value = head_layer(value, layer, padding)
    marker_pos = torch.tensor(case["marker_pos"], dtype=torch.int64)
    marker_mask = torch.tensor(case["marker_mask"], dtype=torch.bool)
    markers = value[marker_pos]
    scorer = head.scorer
    score_value = norm(markers, f64(scorer[0].weight)) + f64(scorer[0].bias)
    score_value = score_value @ f64(scorer[1].weight).transpose(0, 1) + f64(
        scorer[1].bias
    )
    score_value = torch.nn.functional.gelu(score_value)
    scores = (
        score_value @ f64(scorer[3].weight).transpose(0, 1) + f64(scorer[3].bias)
    ).squeeze(-1)
    scores = scores.masked_fill(~marker_mask, -10000.0)
    return {
        "hidden": hidden,
        "scores": scores,
        "probabilities": softmax_probabilities(scores, marker_mask),
        "stages": stages,
    }


def f64_layer0_trace(case: dict[str, Any]) -> dict[str, torch.Tensor]:
    input_ids = torch.tensor(case["input_ids"], dtype=torch.int64)
    mask = torch.tensor(case["attention_mask"], dtype=torch.bool)
    embedding = norm(
        f64(FULL.ENCODER.values((FULL.VOCAB, WIDTH), 200)[input_ids]),
        f64(FULL.near_one(201)),
    )
    weights = FULL.layer_weights(0)
    qkv = (embedding @ f64(weights["wqkv"]).transpose(0, 1)).reshape(
        embedding.shape[0], 3, HEADS, HEAD_DIM
    )
    query, key, val = qkv.unbind(dim=1)
    query, key = rope(query, False), rope(key, False)
    scores = torch.einsum("qhd,khd->hqk", query, key) / math.sqrt(HEAD_DIM)
    scores = scores.masked_fill(~mask[None, None, :], torch.finfo(torch.float32).min)
    attended = torch.einsum("hqk,khd->qhd", scores.softmax(dim=-1), val).reshape_as(
        embedding
    )
    return {
        "qkv": qkv.reshape(embedding.shape[0], 3 * WIDTH),
        "rotated_query": query,
        "rotated_key": key,
        "logits": scores,
        "probabilities": scores.softmax(dim=-1),
        "attended": attended,
        "post_wo_residual": embedding
        + attended @ f64(weights["attn_wo"]).transpose(0, 1),
    }


def error(actual: torch.Tensor, reference: torch.Tensor) -> dict[str, Any]:
    difference = (actual.to(torch.float64) - reference).abs().flatten()
    maximum, index = difference.max(dim=0)
    return {
        "max_abs": maximum.item(),
        "flat_index": index.item(),
        "count": difference.numel(),
    }


def trace_tensor(
    value: object,
    shape: tuple[int, ...],
    label: str,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    tensor = torch.tensor(value, dtype=dtype)
    if tuple(tensor.shape) != shape:
        raise ValueError(f"{label} shape {tuple(tensor.shape)} does not equal {shape}")
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{label} contains a non-finite value")
    return tensor


def serial_f32_scores(query: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
    """Replay Rust's scalar F32 product/add order from captured rotated vectors."""
    if query.dtype != torch.float32 or key.dtype != torch.float32:
        raise TypeError("serial score replay requires F32 rotated vectors")
    positions = query.shape[0]
    scores = torch.empty((HEADS, positions, positions), dtype=torch.float32)
    scale = torch.tensor(HEAD_DIM**-0.5, dtype=torch.float32)
    for head in range(HEADS):
        for query_index in range(positions):
            for key_index in range(positions):
                total = torch.tensor(0.0, dtype=torch.float32)
                for dimension in range(HEAD_DIM):
                    total = (
                        total
                        + query[query_index, head, dimension]
                        * key[key_index, head, dimension]
                    )
                scores[head, query_index, key_index] = total * scale
    return scores


def balanced_f32_scores(query: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
    """Replay the trial's fixed pairwise F32 reduction from captured vectors."""
    if query.dtype != torch.float32 or key.dtype != torch.float32:
        raise TypeError("balanced score replay requires F32 rotated vectors")
    positions = query.shape[0]
    scores = torch.empty((HEADS, positions, positions), dtype=torch.float32)
    scale = torch.tensor(HEAD_DIM**-0.5, dtype=torch.float32)
    for head in range(HEADS):
        for query_index in range(positions):
            for key_index in range(positions):
                products = query[query_index, head] * key[key_index, head]
                while products.numel() > 1:
                    products = products[0::2] + products[1::2]
                scores[head, query_index, key_index] = products[0] * scale
    return scores


def calibration_case_for_projection(report: dict[str, Any]) -> dict[str, Any]:
    if report.get("manifest_sha256") != MANIFEST_SHA256:
        raise ValueError("projection calibration report manifest identity mismatch")
    if report.get("weight_f32_sha256") != weights_f32_sha256():
        raise ValueError("projection calibration report weight identity mismatch")
    cases = report.get("cases")
    if not isinstance(cases, list):
        raise TypeError("projection calibration report lacks cases")
    if not all(isinstance(case, dict) for case in cases):
        raise TypeError("projection calibration report has malformed cases")
    matches = [case for case in cases if case.get("name") == "cal_len7"]
    if len(matches) != 1 or matches[0].get("split") != "calibration":
        raise ValueError(
            "projection calibration report requires one cal_len7 calibration case"
        )
    return matches[0]


def projection_ablation(
    calibration_path: Path,
    native_qkv: torch.Tensor,
    source_qkv: torch.Tensor,
    ideal_qkv: torch.Tensor,
) -> dict[str, Any]:
    report = json.loads(calibration_path.read_text())
    case = calibration_case_for_projection(report)
    native_record = case.get("native_f32_vs_f64")
    source_boundaries = case.get("source_f32_boundaries")
    ideal_record = case.get("ideal_f64")
    if not isinstance(native_record, dict) or not isinstance(source_boundaries, dict):
        raise TypeError(
            "projection calibration report lacks captured native/source boundaries"
        )
    if not isinstance(ideal_record, dict):
        raise TypeError("projection calibration report lacks ideal boundaries")
    native_boundaries = native_record.get("boundaries")
    ideal_stages = ideal_record.get("stages")
    source_embedding_record = source_boundaries.get("embedding")
    if not isinstance(native_boundaries, list) or len(native_boundaries) != len(
        stage_names()
    ):
        raise ValueError("projection calibration native boundary count mismatch")
    if not isinstance(ideal_stages, list) or len(ideal_stages) != len(stage_names()):
        raise ValueError("projection calibration ideal boundary count mismatch")
    if not isinstance(source_embedding_record, dict):
        raise TypeError("projection calibration source embedding record is invalid")
    positions = native_qkv.shape[0]
    embedding_shape = (positions, WIDTH)
    qkv_shape = (positions, 3 * WIDTH)
    native_embedding = trace_tensor(
        native_boundaries[0], embedding_shape, "native calibration embedding"
    )
    source_embedding = trace_tensor(
        source_embedding_record.get("value"),
        embedding_shape,
        "source calibration embedding",
    )
    ideal_embedding = trace_tensor(
        ideal_stages[0], embedding_shape, "ideal calibration embedding", torch.float64
    )
    weight = f64(FULL.layer_weights(0)["wqkv"])

    def project(value: torch.Tensor) -> torch.Tensor:
        return value.to(torch.float64) @ weight.transpose(0, 1)

    native_projected = project(native_embedding)
    source_projected = project(source_embedding)
    ideal_projected = project(ideal_embedding)
    ideal_identity = error(ideal_projected, ideal_qkv)
    if ideal_identity["max_abs"] != 0.0:
        raise ValueError("F64 ideal embedding projection does not reproduce ideal QKV")
    return {
        "calibration_report_sha256": hashlib.sha256(
            calibration_path.read_bytes()
        ).hexdigest(),
        "manifest_sha256": report["manifest_sha256"],
        "weight_f32_sha256": report["weight_f32_sha256"],
        "case": "cal_len7",
        "shapes": {"embedding": list(embedding_shape), "qkv": list(qkv_shape)},
        "comparisons": {
            "native_projection_accumulation": error(native_qkv, native_projected),
            "source_projection_accumulation": error(source_qkv, source_projected),
            "upstream_native_vs_source_embedding_propagation": error(
                native_projected, source_projected
            ),
            "source_embedding_norm_propagation": error(
                source_projected, ideal_projected
            ),
            "native_embedding_vs_source": error(
                native_embedding, source_embedding.to(torch.float64)
            ),
            "native_embedding_vs_ideal": error(native_embedding, ideal_embedding),
            "ideal_projection_identity": ideal_identity,
        },
    }


def layer0_replay(
    native_path: Path,
    source_path: Path,
    score_reduction: str = "serial_f32",
    calibration_path: Path | None = None,
) -> dict[str, Any]:
    native = json.loads(native_path.read_text())
    source = json.loads(source_path.read_text())
    if (
        native.get("schema_version") != 2
        or native.get("attention_layout") != "head_query_key"
    ):
        raise ValueError("native replay trace must use schema 2 head/query/key layout")
    if native.get("case") != "cal_len7" or source.get("case") != "cal_len7":
        raise ValueError("layer-zero replay only accepts the frozen cal_len7 traces")
    positions = len(native.get("qkv", []))
    if positions != 7:
        raise ValueError(f"cal_len7 trace has {positions} positions")
    shape_qkv = (positions, 3 * WIDTH)
    shape_rotated = (positions, HEADS, HEAD_DIM)
    shape_scores = (HEADS, positions, positions)
    native_qkv = trace_tensor(native.get("qkv"), shape_qkv, "native qkv")
    native_query = trace_tensor(
        native.get("rotated_query"), shape_rotated, "native rotated query"
    )
    native_key = trace_tensor(
        native.get("rotated_key"), shape_rotated, "native rotated key"
    )
    native_logits = trace_tensor(native.get("logits"), shape_scores, "native logits")
    source_values = source.get("source_f32")
    if not isinstance(source_values, dict):
        raise TypeError("source replay trace lacks source_f32 tensors")
    ideal_values = source.get("ideal_f64")
    if not isinstance(ideal_values, dict):
        raise TypeError("source replay trace lacks ideal_f64 tensors")
    source_qkv = trace_tensor(source_values.get("qkv"), shape_qkv, "source qkv")
    source_query = trace_tensor(
        source_values.get("rotated_query"), shape_rotated, "source rotated query"
    )
    source_key = trace_tensor(
        source_values.get("rotated_key"), shape_rotated, "source rotated key"
    )
    source_logits = trace_tensor(
        source_values.get("logits"), shape_scores, "source logits"
    )
    ideal_qkv = trace_tensor(
        ideal_values.get("qkv"), shape_qkv, "ideal QKV", torch.float64
    )

    native_parts = native_qkv.reshape(positions, 3, HEADS, HEAD_DIM)
    source_operator_query = FULL.ENCODER.rope(native_parts[:, 0])
    source_operator_key = FULL.ENCODER.rope(native_parts[:, 1])
    source_operator_native_qkv = (
        torch.einsum("qhd,khd->hqk", source_operator_query, source_operator_key)
        / HEAD_DIM**0.5
    )
    source_reconstructed = (
        torch.einsum("qhd,khd->hqk", source_query, source_key) / HEAD_DIM**0.5
    )
    native_rotation_tensor = (
        torch.einsum("qhd,khd->hqk", native_query, native_key) / HEAD_DIM**0.5
    )
    source_rotation_serial = serial_f32_scores(
        source_operator_query, source_operator_key
    )
    native_reduction = {
        "serial_f32": serial_f32_scores,
        "balanced_f32": balanced_f32_scores,
    }.get(score_reduction)
    if native_reduction is None:
        raise ValueError(f"unsupported replay reduction {score_reduction}")
    native_rotation_replay = native_reduction(native_query, native_key)
    native_query_f64 = native_parts[:, 0].to(torch.float64)
    native_key_f64 = native_parts[:, 1].to(torch.float64)
    source_coefficients_f64 = torch.einsum(
        "qhd,khd->hqk",
        rope(native_query_f64, True),
        rope(native_key_f64, True),
    ) / math.sqrt(HEAD_DIM)
    ideal_coefficients_f64 = torch.einsum(
        "qhd,khd->hqk",
        rope(native_query_f64, False),
        rope(native_key_f64, False),
    ) / math.sqrt(HEAD_DIM)
    replay = {
        "case": "cal_len7",
        "native_trace_schema": 2,
        "native_score_reduction": score_reduction,
        "source_trace": "explicit_source_qkv_reconstruction",
        "input_sha256": {
            "native_trace": hashlib.sha256(native_path.read_bytes()).hexdigest(),
            "source_trace": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        },
        "comparisons": {
            "source_tensor_reconstruction_identity": error(
                source_logits, source_reconstructed.to(torch.float64)
            ),
            "raw_qkv_native_vs_source": error(native_qkv, source_qkv.to(torch.float64)),
            "source_rotated_native_vs_source_qkv": error(
                source_operator_native_qkv, source_logits.to(torch.float64)
            ),
            "native_rotation_vs_source_rotation_same_native_qkv": error(
                native_rotation_tensor, source_operator_native_qkv.to(torch.float64)
            ),
            "source_tensor_vs_scalar_reduction_same_source_rotation": error(
                source_operator_native_qkv, source_rotation_serial.to(torch.float64)
            ),
            "native_vs_source_rotation_same_native_qkv_serial": error(
                serial_f32_scores(native_query, native_key),
                source_rotation_serial.to(torch.float64),
            ),
            "native_reduction_vs_tensor_same_rotated": error(
                native_logits, native_rotation_tensor.to(torch.float64)
            ),
            "native_reduction_replay_exactness": error(
                native_logits, native_rotation_replay.to(torch.float64)
            ),
            "source_f32_vs_ideal_f64_rope_coefficients_same_native_qkv": error(
                source_coefficients_f64, ideal_coefficients_f64
            ),
            "native_rotated_query_vs_source": error(
                native_query, source_query.to(torch.float64)
            ),
            "native_rotated_key_vs_source": error(
                native_key, source_key.to(torch.float64)
            ),
        },
    }
    if calibration_path is not None:
        replay["projection_ablation"] = projection_ablation(
            calibration_path, native_qkv, source_qkv, ideal_qkv
        )
    return replay


def weights_f32_sha256() -> str:
    digest = hashlib.sha256()
    for layer_id in range(LAYERS):
        for name in ("wqkv", "attn_wo", "wi", "mlp_wo", "attn_norm", "mlp_norm"):
            value = FULL.layer_weights(layer_id)[name].contiguous()
            digest.update(
                f"encoder.{layer_id}.{name}:{tuple(value.shape)}:{value.dtype}\n".encode()
            )
            digest.update(value.numpy().tobytes())
    for name, value in (
        ("embedding_rows", FULL.ENCODER.values((FULL.VOCAB, WIDTH), 200)),
        ("embedding_norm", FULL.near_one(201)),
        ("final_norm", FULL.near_one(202)),
    ):
        value = value.contiguous()
        digest.update(f"{name}:{tuple(value.shape)}:{value.dtype}\n".encode())
        digest.update(value.numpy().tobytes())
    head, _ = FULL.HEAD.build_source_oracle()
    for name, parameter in head.named_parameters():
        value = parameter.detach().contiguous()
        digest.update(f"head.{name}:{tuple(value.shape)}:{value.dtype}\n".encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def local_reference_hashes() -> dict[str, str]:
    return {
        filename: sha256(ROOT / "scripts" / filename)
        for filename in (
            "julia_accuracy_reference.py",
            "julia_full_prefill_reference.py",
            "julia_encoder_reference.py",
            "julia_head_reference.py",
        )
    }


def validate_native_receipt(
    payload: dict[str, Any], expected_names: set[str]
) -> dict[str, Any]:
    if (
        payload.get("protocol_schema") != 1
        or payload.get("manifest_sha256") != MANIFEST_SHA256
        or payload.get("weight_f32_sha256") != weights_f32_sha256()
    ):
        raise ValueError(
            "native receipt does not bind the frozen protocol, manifest, and weights"
        )
    cases = payload.get("cases")
    if not isinstance(cases, list):
        raise TypeError("native receipt cases must be a list")
    names = [case.get("name") for case in cases if isinstance(case, dict)]
    if set(names) != expected_names or len(names) != len(expected_names):
        raise ValueError("native receipt must contain each expected case exactly once")
    return {case["name"]: case for case in cases}


def manifest_cases() -> list[dict[str, Any]]:
    payload = MANIFEST.read_bytes()
    actual = hashlib.sha256(payload).hexdigest()
    if actual != MANIFEST_SHA256:
        raise RuntimeError(f"accuracy manifest SHA {actual} does not match frozen pin")
    cases = json.loads(payload)["cases"]
    splits = {case["split"] for case in cases}
    if splits != {"calibration", "held_out"}:
        raise ValueError(
            "accuracy manifest must contain calibration and held_out splits"
        )
    identities: dict[tuple[tuple[int, ...], tuple[bool, ...]], str] = {}
    groups: dict[str, str] = {}
    for case in cases:
        identity = (tuple(case["input_ids"]), tuple(case["attention_mask"]))
        prior_split = identities.setdefault(identity, case["split"])
        if prior_split != case["split"]:
            raise ValueError("identical inputs cannot cross calibration and held_out")
        group = case["name"].removesuffix("_rows")
        prior_group = groups.setdefault(group, case["split"])
        if prior_group != case["split"]:
            raise ValueError("related padding groups cannot cross splits")
    if sum(case["split"] == "calibration" for case in cases) < 8:
        raise ValueError("accuracy manifest requires at least eight calibration cases")
    if sum(case["split"] == "held_out" for case in cases) < 8:
        raise ValueError("accuracy manifest requires at least eight held_out cases")
    return cases


def source_case(
    case: dict[str, Any], legacy: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    if legacy:
        fixture_case = next(
            item
            for item in json.loads(FIXTURE.read_text())["cases"]
            if item["name"] == case["name"]
        )
        return FULL.source_case(fixture_case)
    return FULL.source_case(case)


def tensor_record(value: torch.Tensor) -> list[Any]:
    return value.tolist()


def check_shape_finite(value: torch.Tensor, shape: tuple[int, ...], label: str) -> None:
    if tuple(value.shape) != shape:
        raise ValueError(f"{label} shape {tuple(value.shape)} != {shape}")
    if not torch.isfinite(value).all():
        raise ValueError(f"{label} has a nonfinite value")


def check_scores(scores: torch.Tensor, marker_mask: torch.Tensor, label: str) -> None:
    if tuple(scores.shape) != (marker_mask.numel(),):
        raise ValueError(f"{label} score shape does not match marker mask")
    if not torch.isfinite(scores).all():
        raise ValueError(f"{label} has a nonfinite score")
    if not torch.equal(
        scores[~marker_mask], torch.full_like(scores[~marker_mask], -10000.0)
    ):
        raise ValueError(f"{label} invalid markers are not exactly -10000")


def stage_names() -> list[str]:
    return ["embedding", *(f"layer_{index}" for index in range(LAYERS)), "final_norm"]


def boundary_metrics(
    source: torch.Tensor, reference: torch.Tensor, scale: float
) -> dict[str, Any]:
    require_f64(reference, "boundary reference")
    absolute = (source.to(torch.float64) - reference).abs().flatten()
    normalized = absolute / (scale + reference.abs().flatten())
    return {
        **error(source, reference),
        "p99_abs": torch.quantile(absolute, 0.99).item(),
        "normalized_rms": normalized.square().mean().sqrt().item(),
        "normalized_max": normalized.max().item(),
    }


def calibration_limits(
    boundaries: list[tuple[dict[str, torch.Tensor], list[torch.Tensor]]],
) -> dict[str, Any]:
    limits = {}
    unit_roundoff = 2.0**-24
    for index, name in enumerate(stage_names()):
        source_values = [source[name].to(torch.float64) for source, _ in boundaries]
        reference_values = [reference[index] for _, reference in boundaries]
        scale = max(
            1.0, *(value.square().mean().sqrt().item() for value in reference_values)
        )
        normalized = max(
            ((source - reference).abs() / (scale + reference.abs())).max().item()
            for source, reference in zip(source_values, reference_values)
        )
        limits[name] = {
            "reference_rms_scale": scale,
            "source_f32_normalized_max": normalized,
            "prospective_boundary": max(8 * unit_roundoff, 2 * normalized),
        }
    return limits


def defect_controls() -> dict[str, Any]:
    case = next(case for case in manifest_cases() if case["name"] == "cal_sparse8")
    reference = f64_case(case, source_f32_rope=False)
    defects = {
        "omit_final_norm": f64_case(case, source_f32_rope=False, omit_final_norm=True),
        "wrong_layer_mapping": f64_case(
            case, source_f32_rope=False, wrong_layer_mapping=True
        ),
        "inverted_attention_mask": f64_case(
            case, source_f32_rope=False, invert_mask=True
        ),
        "wrong_rope_theta": f64_case(
            case, source_f32_rope=False, wrong_rope_theta=True
        ),
    }
    result = {}
    for name, actual in defects.items():
        measured = error(actual["hidden"], reference["hidden"])
        if measured["max_abs"] <= 1e-5:
            raise RuntimeError(f"{name} did not breach the hidden defect control")
        result[name] = measured
    return result


def operator_properties() -> dict[str, float]:
    ones = torch.ones((2, WIDTH), dtype=torch.float64)
    normalized = norm(ones, ones)
    nonconstant = f64(FULL.ENCODER.values((2, WIDTH), 33))
    unit_normalized = norm(nonconstant, ones)
    rotary_input = f64(FULL.ENCODER.values((2, HEADS, HEAD_DIM), 9))
    rotary = rope(rotary_input, source_f32_coefficients=False)
    base = next(case for case in manifest_cases() if case["name"] == "cal_perm_pad")
    padded_rows = next(
        case for case in manifest_cases() if case["name"] == "cal_perm_pad_rows"
    )
    base_output = f64_case(base, source_f32_rope=False)
    padded_output = f64_case(padded_rows, source_f32_rope=False)
    visible = torch.tensor(base["attention_mask"], dtype=torch.bool)
    swapped = {**base, "marker_pos": list(reversed(base["marker_pos"]))}
    swapped_output = f64_case(swapped, source_f32_rope=False)
    results = {
        "norm_constant_max_abs": normalized.abs().max().item(),
        "norm_unit_weight_zero_mean_max_abs": unit_normalized.mean(dim=-1)
        .abs()
        .max()
        .item(),
        "rope_position_zero_max_abs": (rotary[0] - rotary_input[0]).abs().max().item(),
        "rope_norm_max_abs": (rotary.square().sum(-1) - rotary_input.square().sum(-1))
        .abs()
        .max()
        .item(),
        "padding_visible_hidden_max_abs": (
            base_output["hidden"][visible] - padded_output["hidden"][visible]
        )
        .abs()
        .max()
        .item(),
        "padding_valid_score_max_abs": (base_output["scores"] - padded_output["scores"])
        .abs()
        .max()
        .item(),
        "marker_permutation_max_abs": (
            swapped_output["scores"] - base_output["scores"].flip(0)
        )
        .abs()
        .max()
        .item(),
    }
    if any(value > 1e-12 for value in results.values()):
        raise RuntimeError(f"F64 operator property failed: {results}")
    return results


def report(
    native_path: Path | None,
    include_held_out: bool,
) -> dict[str, Any]:
    if include_held_out:
        raise ValueError(
            "held-out comparison is intentionally closed after calibration failed"
        )
    legacy_cases = [
        case
        for case in json.loads(FIXTURE.read_text())["cases"]
        if case["name"] in LEGACY
    ]
    cases = legacy_cases + [
        case
        for case in manifest_cases()
        if include_held_out or case["split"] == "calibration"
    ]
    native_cases = {}
    if native_path:
        native_payload = json.loads(native_path.read_text())
        expected_names = {
            case["name"]
            for case in manifest_cases()
            if case["split"] == ("held_out" if include_held_out else "calibration")
        }
        native_cases = validate_native_receipt(native_payload, expected_names)
    observed = []
    calibration_boundaries = []
    source_operators: set[str] = set()
    for case in cases:
        legacy = case["name"] in LEGACY
        ideal = f64_case(case, source_f32_rope=False)
        source_rope = f64_case(case, source_f32_rope=True)
        torch_hidden, torch_scores = source_case(case, legacy)
        marker_mask = torch.tensor(case["marker_mask"], dtype=torch.bool)
        positions = len(case["input_ids"])
        for index, stage in enumerate(ideal["stages"]):
            check_shape_finite(
                stage, (positions, WIDTH), f"{case['name']} ideal stage {index}"
            )
        check_shape_finite(
            torch_hidden.squeeze(0), (positions, WIDTH), f"{case['name']} source hidden"
        )
        check_scores(ideal["scores"], marker_mask, f"{case['name']} ideal")
        check_scores(torch_scores.squeeze(0), marker_mask, f"{case['name']} source")
        source_boundaries = None
        if not legacy and case["split"] == "calibration":
            source_boundaries, operators = FULL.boundaries(case)
            source_operators.update(operators)
            calibration_boundaries.append((source_boundaries, ideal["stages"]))
        record: dict[str, Any] = {
            "name": case["name"],
            "split": "legacy_diagnostic" if legacy else case["split"],
            "source_f32": {
                "hidden": tensor_record(torch_hidden.squeeze(0)),
                "scores": tensor_record(torch_scores.squeeze(0)),
                "probabilities": tensor_record(
                    softmax_probabilities(
                        torch_scores.squeeze(0).to(torch.float64), marker_mask
                    )
                ),
            },
            "ideal_f64": {
                "stages": [tensor_record(stage) for stage in ideal["stages"]],
                "scores": tensor_record(ideal["scores"]),
                "probabilities": tensor_record(ideal["probabilities"]),
            },
            "source_f32_rope_f64": {
                "hidden": error(source_rope["hidden"], ideal["hidden"]),
                "scores": error(source_rope["scores"], ideal["scores"]),
            },
            "torch_f32_vs_f64": {
                "hidden": error(torch_hidden.squeeze(0), ideal["hidden"]),
                "scores": error(torch_scores.squeeze(0), ideal["scores"]),
                "probabilities": error(
                    softmax_probabilities(
                        torch_scores.squeeze(0).to(torch.float64), marker_mask
                    ),
                    ideal["probabilities"],
                ),
            },
        }
        if source_boundaries:
            for name, boundary in source_boundaries.items():
                check_shape_finite(
                    boundary, (positions, WIDTH), f"{case['name']} source {name}"
                )
            record["source_f32_boundaries"] = {
                name: {
                    "value": tensor_record(source_boundaries[name]),
                }
                for name in stage_names()
            }
        if case["name"] in native_cases:
            native = native_cases[case["name"]]
            native_hidden = torch.tensor(native["hidden"], dtype=torch.float32)
            native_scores = torch.tensor(native["scores"], dtype=torch.float32)
            check_shape_finite(
                native_hidden, (positions, WIDTH), f"{case['name']} native hidden"
            )
            check_scores(native_scores, marker_mask, f"{case['name']} native")
            native_boundaries = native.get("boundaries")
            if not isinstance(native_boundaries, list) or len(native_boundaries) != len(
                stage_names()
            ):
                raise ValueError(f"{case['name']} native boundary count is not exact")
            for index, boundary in enumerate(native_boundaries):
                check_shape_finite(
                    torch.tensor(boundary, dtype=torch.float32),
                    (positions, WIDTH),
                    f"{case['name']} native {stage_names()[index]}",
                )
            record["native_f32_vs_f64"] = {
                "hidden": error(native_hidden, ideal["hidden"]),
                "scores": error(native_scores, ideal["scores"]),
                "probabilities": error(
                    softmax_probabilities(native_scores.to(torch.float64), marker_mask),
                    ideal["probabilities"],
                ),
                "boundaries": [
                    tensor_record(torch.tensor(boundary, dtype=torch.float32))
                    for boundary in native_boundaries
                ],
            }
        observed.append(record)
    limits = calibration_limits(calibration_boundaries)
    for record in observed:
        boundaries = record.get("source_f32_boundaries")
        if boundaries:
            for index, name in enumerate(stage_names()):
                boundaries[name]["vs_ideal_f64"] = boundary_metrics(
                    torch.tensor(boundaries[name]["value"], dtype=torch.float32),
                    torch.tensor(
                        record["ideal_f64"]["stages"][index], dtype=torch.float64
                    ),
                    limits[name]["reference_rms_scale"],
                )
        native = record.get("native_f32_vs_f64")
        if native:
            native["boundary_metrics"] = {
                name: boundary_metrics(
                    torch.tensor(native["boundaries"][index], dtype=torch.float32),
                    torch.tensor(
                        record["ideal_f64"]["stages"][index], dtype=torch.float64
                    ),
                    limits[name]["reference_rms_scale"],
                )
                for index, name in enumerate(stage_names())
            }
            failed_boundaries = []
            for name, measured in native["boundary_metrics"].items():
                budget = limits[name]["prospective_boundary"]
                measured["budget"] = budget
                measured["budget_ratio"] = measured["normalized_max"] / budget
                measured["passes_budget"] = measured["normalized_max"] <= budget
                if not measured["passes_budget"]:
                    failed_boundaries.append(name)
            native["hidden_contract"] = {
                "passes": not failed_boundaries,
                "failed_boundaries": failed_boundaries,
                "score_passes_1e_5": native["scores"]["max_abs"] <= 1e-5,
                "probability_passes_5e_6": native["probabilities"]["max_abs"] <= 5e-6,
            }
    calibration_native = [
        case["native_f32_vs_f64"]["hidden_contract"]
        for case in observed
        if case["split"] == "calibration" and "native_f32_vs_f64" in case
    ]
    return {
        "schema_version": 1,
        "manifest_sha256": MANIFEST_SHA256,
        "weight_policy": "generated F32 then cast to F64",
        "reference_mode": "explicit ideal F64 reductions/softmax/layer norm/GEGLU; source F32 RoPE coefficient sensitivity reported separately",
        "weight_f32_sha256": weights_f32_sha256(),
        "local_reference_hashes": local_reference_hashes(),
        "torch": {"version": torch.__version__, "config": torch.__config__.show()},
        "source_sdpa_cpu_operators": sorted(source_operators),
        "frozen_calibration_limits": limits,
        "calibration_diagnostic": {
            "cases_with_native": len(calibration_native),
            "hidden_passes": bool(calibration_native)
            and all(item["passes"] for item in calibration_native),
            "score_passes": bool(calibration_native)
            and all(item["score_passes_1e_5"] for item in calibration_native),
            "probability_passes": bool(calibration_native)
            and all(item["probability_passes_5e_6"] for item in calibration_native),
            "passes": bool(calibration_native)
            and all(
                item["passes"]
                and item["score_passes_1e_5"]
                and item["probability_passes_5e_6"]
                for item in calibration_native
            ),
            "held_out_opened": False,
        },
        "cases": observed,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--native-output", type=Path)
    parser.add_argument("--write-report", type=Path)
    parser.add_argument("--include-held-out", action="store_true")
    parser.add_argument("--check-defects", action="store_true")
    parser.add_argument("--check-properties", action="store_true")
    parser.add_argument("--trace-case")
    parser.add_argument("--trace-output", type=Path)
    parser.add_argument("--replay-native", type=Path)
    parser.add_argument("--replay-source", type=Path)
    parser.add_argument("--replay-output", type=Path)
    parser.add_argument("--replay-calibration-report", type=Path)
    parser.add_argument(
        "--replay-reduction",
        choices=("serial_f32", "balanced_f32"),
        default="serial_f32",
    )
    args = parser.parse_args()
    replay_paths = (args.replay_native, args.replay_source, args.replay_output)
    if any(replay_paths) and not all(replay_paths):
        raise ValueError("layer-zero replay requires native, source, and output paths")
    if args.replay_calibration_report is not None and not all(replay_paths):
        raise ValueError(
            "projection ablation requires complete layer-zero replay paths"
        )
    if all(replay_paths):
        replay = layer0_replay(
            args.replay_native,
            args.replay_source,
            args.replay_reduction,
            args.replay_calibration_report,
        )
        args.replay_output.parent.mkdir(parents=True, exist_ok=True)
        args.replay_output.write_text(json.dumps(replay, indent=2) + "\n")
        return
    if args.trace_case:
        if args.trace_case != "cal_len7" or args.trace_output is None:
            raise ValueError("only cal_len7 trace requires --trace-output")
        case = next(
            case for case in manifest_cases() if case["name"] == args.trace_case
        )
        source = FULL.layer0_trace(case)
        reference = f64_layer0_trace(case)
        trace = {
            "case": args.trace_case,
            "source_f32": {name: tensor_record(source[name]) for name in reference},
            "ideal_f64": {name: tensor_record(reference[name]) for name in reference},
        }
        args.trace_output.parent.mkdir(parents=True, exist_ok=True)
        args.trace_output.write_text(json.dumps(trace, indent=2) + "\n")
        return
    output = report(args.native_output, args.include_held_out)
    if args.check_defects:
        output["defect_controls"] = defect_controls()
    if args.check_properties:
        output["operator_properties"] = operator_properties()
    print(json.dumps(output, indent=2))
    if args.write_report:
        args.write_report.parent.mkdir(parents=True, exist_ok=True)
        args.write_report.write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    main()
