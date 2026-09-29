#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Export oracle-free numerical operands for the fixed scalar DeepSeek runner."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import struct
from pathlib import Path
from typing import Any

MAX_INPUT = 8 << 20
MAX_OUTPUT = 16 << 20
WIDTH = {
    "torch.float8_e4m3fn": 1,
    "torch.float8_e8m0fnu": 1,
    "torch.float4_e2m1fn_x2": 1,
    "torch.bfloat16": 2,
    "torch.float32": 4,
    "torch.int64": 8,
}
OUTPUT_DTYPE = {
    "torch.float8_e4m3fn": "u8",
    "torch.float8_e8m0fnu": "u8",
    "torch.float4_e2m1fn_x2": "u8",
    "torch.bfloat16": "bf16",
    "torch.float32": "f32",
    "torch.int64": "i64",
}


class ExportError(ValueError):
    pass


def obj(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ExportError(f"{label} must be an object")
    return value


def tensor(value: object, label: str) -> dict[str, Any]:
    item = obj(value, label)
    dtype, shape, numel, encoded, digest = (
        item.get(key)
        for key in ("dtype", "shape", "numel", "storage_hex", "storage_sha256")
    )
    if (
        dtype not in WIDTH
        or not isinstance(shape, list)
        or any(type(x) is not int or x < 0 for x in shape)
        or type(numel) is not int
        or not isinstance(encoded, str)
        or not isinstance(digest, str)
    ):
        raise ExportError(f"{label} has invalid tensor metadata")
    if math.prod(shape) != numel:
        raise ExportError(f"{label} shape disagrees with numel")
    try:
        raw = bytes.fromhex(encoded)
    except ValueError as error:
        raise ExportError(f"{label} storage is not hex") from error
    if (
        len(raw) != int(numel * WIDTH[dtype])
        or hashlib.sha256(raw).hexdigest() != digest
    ):
        raise ExportError(f"{label} storage integrity failed")
    if item.get("finite") is not True:
        raise ExportError(f"{label} must be finite")
    return {
        "dtype": OUTPUT_DTYPE[dtype],
        "shape": shape,
        "storage_hex": encoded,
        "storage_sha256": digest,
    }


def insert(
    result: dict[str, dict[str, Any]], name: str, record: dict[str, Any]
) -> None:
    old = result.get(name)
    if old is not None and old != record:
        raise ExportError(f"duplicate tensor {name} differs across source projections")
    result[name] = record


def insert_map(result: dict[str, dict[str, Any]], values: object, label: str) -> None:
    for name, value in obj(values, label).items():
        insert(result, name, tensor(value, f"{label}.{name}"))


def rotary(value: object, label: str) -> dict[str, Any]:
    source = obj(value, label)
    if (
        source.get("dtype") != "torch.complex64"
        or source.get("shape") != [8, 16]
        or source.get("numel") != 128
        or source.get("finite") is not True
    ):
        raise ExportError(
            f"{label} must retain the pinned finite complex [8,16] rotary table"
        )
    encoded, digest = source.get("storage_hex"), source.get("storage_sha256")
    if not isinstance(encoded, str) or not isinstance(digest, str):
        raise ExportError(f"{label} has invalid tensor metadata")
    try:
        raw = bytes.fromhex(encoded)
    except ValueError as error:
        raise ExportError(f"{label} storage is not hex") from error
    if len(raw) != 8 * 128 or hashlib.sha256(raw).hexdigest() != digest:
        raise ExportError(f"{label} storage integrity failed")
    if not all(math.isfinite(number) for number in struct.unpack("<256f", raw)):
        raise ExportError(f"{label} contains nonfinite f32")
    return {
        "dtype": "f32",
        "shape": [8, 16, 2],
        "storage_hex": raw.hex(),
        "storage_sha256": hashlib.sha256(raw).hexdigest(),
    }


def shared_rotary(value: object, label: str) -> dict[str, Any]:
    source = obj(value, label)
    shape, pairs = source.get("shape"), source.get("fp32_pairs")
    if (
        source.get("complex_dtype") != "torch.complex64"
        or shape != [8, 16]
        or not isinstance(pairs, list)
        or len(pairs) != 128
    ):
        raise ExportError(f"{label} must retain the pinned complex [8,16] rotary table")
    raw = bytearray()
    for pair in pairs:
        if (
            not isinstance(pair, list)
            or len(pair) != 2
            or any(
                type(bits) is not int or not 0 <= bits <= 0xFFFF_FFFF for bits in pair
            )
        ):
            raise ExportError(f"{label} has invalid f32 pair")
        for bits in pair:
            number = struct.unpack("<f", struct.pack("<I", bits))[0]
            if not math.isfinite(number):
                raise ExportError(f"{label} contains nonfinite rotary value")
            raw.extend(struct.pack("<I", bits))
    return {
        "dtype": "f32",
        "shape": [8, 16, 2],
        "storage_hex": raw.hex(),
        "storage_sha256": hashlib.sha256(raw).hexdigest(),
    }


def head_tensor(bits: object, shape: list[int], label: str) -> dict[str, Any]:
    if (
        not isinstance(bits, list)
        or len(bits) != math.prod(shape)
        or any(type(x) is not int or not 0 <= x <= 0xFFFF_FFFF for x in bits)
    ):
        raise ExportError(f"{label} has invalid f32 bits")
    raw = b"".join(struct.pack("<I", bit) for bit in bits)
    if not all(math.isfinite(value) for value in struct.unpack(f"<{len(bits)}f", raw)):
        raise ExportError(f"{label} contains nonfinite f32")
    return {
        "dtype": "f32",
        "shape": shape,
        "storage_hex": raw.hex(),
        "storage_sha256": hashlib.sha256(raw).hexdigest(),
    }


def same(left: object, right: object, label: str) -> None:
    if type(left) is not type(right) or left != right:
        raise ExportError(f"inconsistent source configuration for {label}")


def validate_models(
    model: dict[str, Any], attention: dict[str, Any], projections: dict[str, Any]
) -> None:
    common = (
        "dim",
        "head_dim",
        "n_heads",
        "norm_eps",
        "o_groups",
        "o_lora_rank",
        "q_lora_rank",
        "rope_head_dim",
        "window_size",
    )
    for layer in range(1, 5):
        candidate = obj(
            projections[f"layer{layer}_attention"].get("model"),
            f"layer {layer} attention model",
        )
        for key in common:
            same(candidate.get(key), model.get(key), f"layer {layer} {key}")
        for key in (
            "index_n_heads",
            "index_topk",
            "candidate_topk_blocks",
            "candidate_block_size",
        ):
            same(candidate.get(key), attention.get(key), f"layer {layer} {key}")
    for name in ("layer1_tail", "layer2_ffn", "layer3_moe", "layer4_moe"):
        tail = obj(projections[name].get("model"), f"{name} model")
        for key in (
            "dim",
            "gate_temp",
            "moe_inter_dim",
            "n_activated_experts",
            "n_routed_experts",
            "n_shared_experts",
            "norm_topk_prob",
            "route_scale",
            "swiglu_limit",
        ):
            same(tail.get(key), model.get(key), f"{name} {key}")
    hc = obj(projections["layer2_hc"].get("block_config"), "layer two HC configuration")
    for key in ("copies", "hc_eps", "hc_sinkhorn_iters", "norm_eps"):
        same(
            hc.get(key), model.get(key) if key != "copies" else 2, f"layer two HC {key}"
        )
    owner = obj(projections["layer1_owner"].get("model"), "layer one owner model")
    for key in ("index_topk", "norm_eps", "window_size"):
        same(
            owner.get(key),
            attention.get(key) if key == "index_topk" else model.get(key),
            f"layer one owner {key}",
        )
    first_engram = obj(
        projections["layer1_engram"].get("model"), "layer one engram model"
    )
    second_engram = obj(
        projections["layer3_engram"].get("model"), "layer three engram model"
    )
    for key in (
        "copies",
        "dim",
        "embedding_dim",
        "gate_clamp",
        "norm_eps",
        "wkv_width",
    ):
        same(second_engram.get(key), first_engram.get(key), f"layer three engram {key}")
    same(first_engram.get("copies"), 2, "layer one engram copies")
    same(first_engram.get("dim"), model.get("dim"), "layer one engram width")
    same(
        first_engram.get("norm_eps"),
        model.get("norm_eps"),
        "layer one engram norm epsilon",
    )
    head = obj(projections["head"], "head")
    bits = head.get("norm_epsilon_bits")
    if type(bits) is not int or not 0 <= bits <= 0xFFFF_FFFF:
        raise ExportError("head norm epsilon bits are invalid")
    source_epsilon = struct.unpack("<f", struct.pack("<I", bits))[0]
    configured_epsilon = struct.unpack("<f", struct.pack("<f", model.get("norm_eps")))[
        0
    ]
    if not math.isfinite(source_epsilon) or source_epsilon != configured_epsilon:
        raise ExportError("head norm epsilon differs from model norm epsilon")


def export(root: object) -> dict[str, object]:
    source = obj(root, "source bundle")
    if source.get("schema_version") != 1:
        raise ExportError("source bundle schema must be 1")
    projections = obj(source.get("projections"), "source projections")
    required = [
        "layer0_to_layer1",
        "layer1_attention",
        "layer1_engram",
        "layer1_owner",
        "layer1_tail",
        "layer2_attention",
        "layer2_ffn",
        "layer2_hc",
        "layer3_attention",
        "layer3_candidate",
        "layer3_compressor",
        "layer3_engram",
        "layer3_index_key",
        "layer3_moe",
        "layer4_attention",
        "layer4_moe",
        "head",
    ]
    if any(name not in projections for name in required):
        raise ExportError("source bundle lacks a required reduced projection")
    model = obj(projections["layer0_to_layer1"].get("model"), "startup model")
    attention = obj(projections["layer1_attention"].get("model"), "attention model")
    engram_model = obj(
        projections["layer1_engram"].get("model"), "layer one engram model"
    )
    validate_models(model, attention, projections)
    engram = obj(projections["layer1_engram"].get("engram"), "engram")
    layout = obj(engram.get("layout"), "engram layout")
    state = obj(engram.get("hash_state"), "engram hash state")
    config = {
        "width": model.get("dim"),
        "copies": engram_model.get("copies"),
        "vocabulary": model.get("vocab_size"),
        "max_tokens": 8,
        "heads": model.get("n_heads"),
        "head_dimension": model.get("head_dim"),
        "rope_pairs": model.get("rope_head_dim", 0) // 2,
        "query_rank": model.get("q_lora_rank"),
        "output_groups": model.get("o_groups"),
        "output_rank": model.get("o_lora_rank"),
        "window": model.get("window_size"),
        "intermediate": model.get("moe_inter_dim"),
        "routed_experts": model.get("n_routed_experts"),
        "active_experts": model.get("n_activated_experts"),
        "gate_temperature": model.get("gate_temp"),
        "normalize_topk": model.get("norm_topk_prob"),
        "route_scale": model.get("route_scale"),
        "swiglu_limit": model.get("swiglu_limit"),
        "norm_epsilon": model.get("norm_eps"),
        "hc_iterations": model.get("hc_sinkhorn_iters"),
        "hc_epsilon": model.get("hc_eps"),
        "index_heads": attention.get("index_n_heads"),
        "index_topk": attention.get("index_topk"),
        "candidate_topk_blocks": attention.get("candidate_topk_blocks"),
        "candidate_block_size": attention.get("candidate_block_size"),
        "engram_embedding_width": engram_model.get("embedding_dim"),
        "engram_rows": layout.get("num_embeddings"),
        "engram_ngram": layout.get("max_ngram_size"),
        "engram_heads": layout.get("n_heads"),
        "engram_pad_id": state.get("pad_id"),
        "engram_gate_clamp": engram_model.get("gate_clamp"),
    }
    if any(
        value is None or isinstance(value, bool) and key != "normalize_topk"
        for key, value in config.items()
    ):
        raise ExportError("source model has incomplete scalar configuration")
    if config["engram_rows"] != [72, 204] or config["max_tokens"] != 8:
        raise ExportError("source model differs from the bounded reduced geometry")
    tensors: dict[str, dict[str, Any]] = {}
    insert_map(
        tensors, projections["layer0_to_layer1"].get("parameters"), "startup parameters"
    )
    for layer in range(1, 5):
        insert_map(
            tensors,
            projections[f"layer{layer}_attention"].get("encoded_parameters"),
            f"layer {layer} attention",
        )
    for name in ("layer1_tail", "layer2_ffn", "layer2_hc", "layer3_moe", "layer4_moe"):
        projection = obj(projections[name], name)
        insert_map(
            tensors, projection.get("block_parameters", {}), f"{name} block parameters"
        )
        insert_map(
            tensors,
            projection.get("encoded_parameters", {}),
            f"{name} encoded parameters",
        )
    insert_map(
        tensors,
        projections["layer1_owner"].get("encoded_parameters"),
        "layer one owner",
    )
    insert_map(
        tensors,
        projections["layer3_candidate"].get("encoded_parameters"),
        "layer three candidate",
    )
    for layer in (1, 3):
        insert_map(
            tensors,
            projections[f"layer{layer}_engram"].get("encoded_parameters"),
            f"layer {layer} engram",
        )
    compressor = obj(
        projections["layer3_compressor"].get("weights"),
        "layer three compressor weights",
    )
    index = obj(
        projections["layer3_index_key"].get("weights"), "layer three index weights"
    )
    insert(
        tensors,
        "layers.3.attn.compressor.norm.weight",
        tensor(compressor.get("norm"), "L3 compressor norm"),
    )
    insert(
        tensors,
        "layers.3.attn.compressor.wkv.weight",
        tensor(compressor.get("wkv"), "L3 compressor WKV"),
    )
    insert(
        tensors,
        "layers.3.attn.indexer.k_norm.weight",
        tensor(index.get("norm"), "L3 key norm"),
    )
    insert(tensors, "layers.3.attn.indexer.wk.weight", tensor(index.get("wk"), "L3 WK"))
    for name in ("token_map", "primes", "offsets", "multipliers"):
        insert(tensors, f"engram.{name}", tensor(state.get(name), f"engram {name}"))
    head = obj(projections["head"], "head")
    norm_bits = head.get("norm_weight_bf16")
    if (
        not isinstance(norm_bits, list)
        or len(norm_bits) != 128
        or any(type(x) is not int or not 0 <= x <= 0xFFFF for x in norm_bits)
    ):
        raise ExportError("head norm must be 128 BF16 bits")
    raw_norm = b"".join(struct.pack("<H", bit) for bit in norm_bits)
    insert(
        tensors,
        "head.norm.weight",
        {
            "dtype": "bf16",
            "shape": [128],
            "storage_hex": raw_norm.hex(),
            "storage_sha256": hashlib.sha256(raw_norm).hexdigest(),
        },
    )
    shape = head.get("weight_shape")
    if not isinstance(shape, list) or shape != [8, 128]:
        raise ExportError("head weight geometry differs")
    insert(
        tensors,
        "head.weight",
        head_tensor(head.get("weight_fp32_bits"), shape, "head weight"),
    )
    cases = projections["layer0_to_layer1"].get("cases")
    if not isinstance(cases, list) or not cases:
        raise ExportError("startup projection lacks its first source case")
    startup_attention = obj(
        obj(cases[0], "startup case").get("attention"), "startup attention"
    )
    startup_table = rotary(
        startup_attention.get("frequencies"), "layer-zero rotary table"
    )
    shared_table = shared_rotary(
        projections["layer3_attention"].get("frequencies"), "layer-three rotary table"
    )
    for layer in (1, 2, 4):
        if (
            shared_rotary(
                projections[f"layer{layer}_attention"].get("frequencies"),
                f"layer-{layer} rotary table",
            )
            != shared_table
        ):
            raise ExportError(
                f"layer-{layer} rotary table differs from layer-three shared table"
            )
    insert(tensors, "rotary.startup", startup_table)
    insert(tensors, "rotary.shared", shared_table)
    return {
        "schema_version": 1,
        "format": "metallix.deepseek.reduced",
        "config": config,
        "tensors": dict(sorted(tensors.items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.source.is_file() or args.source.stat().st_size > MAX_INPUT:
        raise SystemExit("source must be a regular file no larger than 8 MiB")
    raw = args.source.read_bytes()
    try:
        root = json.loads(
            raw,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ExportError(f"nonfinite JSON {value}")
            ),
        )
        artifact = export(root)
    except (json.JSONDecodeError, ExportError) as error:
        raise SystemExit(f"reduced artifact export failed: {error}") from error
    encoded = (
        json.dumps(artifact, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()
    if len(encoded) > MAX_OUTPUT:
        raise SystemExit("reduced artifact exceeds 16 MiB")
    if args.output == Path("-"):
        print(encoded.decode(), end="")
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError as error:
        raise SystemExit("output exists; refusing overwrite") from error
    with os.fdopen(descriptor, "wb") as output:
        output.write(encoded)
    print(
        json.dumps(
            {
                "bytes": len(encoded),
                "sha256": hashlib.sha256(encoded).hexdigest(),
                "status": "created",
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
