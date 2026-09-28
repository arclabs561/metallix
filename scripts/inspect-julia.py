#!/usr/bin/env python3
"""Bounded metadata and safetensors-header inspection for pinned Julia-1.

This never executes remote code and never reads safetensors payload bytes.  It
is a prerequisite inspector, not a native Julia runtime or support claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

SOURCE_REPOSITORY = "SupersonicLabs/Julia-1"
SOURCE_REVISION = "a85b127321d580d65176c89ced8273f305745d85"
HEADER_CONTRACT = Path(__file__).parent.parent / "fixtures/julia-1/header-contract.json"
MAX_METADATA_BYTES = 16 * 1024 * 1024
MAX_HEADER_BYTES = 100 * 1024 * 1024
F32_BYTES = 4

TOP_LEVEL_CONFIG = {
    "format_version": 1,
    "architecture": "JuliaDecisionModel",
    "julia_config_file": "julia_config.json",
    "encoder_config_file": "encoder/config.json",
    "weights_file": "model.safetensors",
    "tokenizer_directory": "tokenizer",
}
JULIA_CONFIG = {
    "format_version": 1,
    "architecture": "JuliaDecisionModel",
    "weight_dtype": "float32",
    "head_layers": 2,
    "n_act": 2,
    "dropout": 0.1,
}
ENCODER_CONFIG = {
    "model_type": "modernbert",
    "hidden_size": 384,
    "num_hidden_layers": 22,
    "num_attention_heads": 6,
    "intermediate_size": 1152,
    "max_position_embeddings": 8192,
    "global_attn_every_n_layers": 3,
    "local_attention": 128,
    "position_embedding_type": "sans_pos",
    "pad_token_id": 0,
    "cls_token_id": 1,
    "mask_token_id": 4,
    "sep_token_id": 1,
}

# These keys follow directly from named JuliaDecisionModel-owned modules in the
# pinned source and give focused diagnostics.  Full tensor-name/type/shape
# strictness is separately verified by validate_published_header against the
# bounded, pinned published-header contract; this helper is not that full set.
KNOWN_JULIA_TENSORS = {
    "temperature": (3,),
    "type_emb.weight": (3, 384),
    "scorer.0.weight": (384,),
    "scorer.0.bias": (384,),
    "scorer.1.weight": (384, 384),
    "scorer.1.bias": (384,),
    "scorer.3.weight": (1, 384),
    "scorer.3.bias": (1,),
    "act_head.0.weight": (256, 388),
    "act_head.0.bias": (256,),
    "act_head.2.weight": (2, 256),
    "act_head.2.bias": (2,),
}


class InspectionError(ValueError):
    """A local artifact or its bounded header cannot meet this contract."""


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise InspectionError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _json_object(data: bytes, source: str) -> dict[str, Any]:
    try:
        value = json.loads(data.decode("utf-8"), object_pairs_hook=_unique_object)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise InspectionError(f"{source} is not valid UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise InspectionError(f"{source} must contain a JSON object")
    return value


def _product(shape: list[int], name: str) -> int:
    product = 1
    for dimension in shape:
        if type(dimension) is not int or dimension < 0:
            raise InspectionError(f"tensor {name!r} has an invalid shape")
        product *= dimension
    return product


def parse_safetensors_header(
    prefix_and_header: bytes, declared_file_bytes: int
) -> dict[str, dict[str, Any]]:
    """Parse exactly a safetensors prefix and header, never tensor payload.

    ``declared_file_bytes`` permits a header-only fixture to describe its full
    hypothetical file.  A normal artifact inspection uses the model file's
    actual byte length instead.
    """
    if type(declared_file_bytes) is not int or declared_file_bytes < 8:
        raise InspectionError(
            "declared safetensors file size must be at least eight bytes"
        )
    if len(prefix_and_header) < 8:
        raise InspectionError("safetensors prefix needs eight bytes")
    header_bytes = struct.unpack("<Q", prefix_and_header[:8])[0]
    if header_bytes > MAX_HEADER_BYTES:
        raise InspectionError("safetensors header exceeds the inspection limit")
    required = 8 + header_bytes
    if declared_file_bytes < required:
        raise InspectionError("declared safetensors file is shorter than its header")
    if len(prefix_and_header) != required:
        raise InspectionError(
            "supplied safetensors bytes must end exactly after the header"
        )
    header = _json_object(prefix_and_header[8:], "safetensors header")
    payload_bytes = declared_file_bytes - required
    tensors: dict[str, dict[str, Any]] = {}
    ranges: list[tuple[int, int, str]] = []
    for name, value in header.items():
        if name == "__metadata__":
            if not isinstance(value, dict) or not all(
                isinstance(key, str) and isinstance(item, str)
                for key, item in value.items()
            ):
                raise InspectionError(
                    "safetensors metadata must map strings to strings"
                )
            continue
        if not isinstance(name, str) or not name:
            raise InspectionError("safetensors tensor names must be nonempty strings")
        if not isinstance(value, dict):
            raise InspectionError(f"tensor {name!r} header must be an object")
        if set(value) != {"dtype", "shape", "data_offsets"}:
            raise InspectionError(f"tensor {name!r} header fields are unsupported")
        dtype, shape, offsets = value["dtype"], value["shape"], value["data_offsets"]
        if dtype != "F32":
            raise InspectionError(f"tensor {name!r} has dtype {dtype!r}, expected F32")
        if not isinstance(shape, list):
            raise InspectionError(f"tensor {name!r} shape must be an array")
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or any(type(offset) is not int or offset < 0 for offset in offsets)
            or offsets[0] > offsets[1]
        ):
            raise InspectionError(f"tensor {name!r} has invalid data offsets")
        start, end = offsets
        expected = _product(shape, name) * F32_BYTES
        if end - start != expected:
            raise InspectionError(
                f"tensor {name!r} byte range does not match F32 shape"
            )
        if end > payload_bytes:
            raise InspectionError(f"tensor {name!r} extends beyond declared payload")
        tensors[name] = {
            "dtype": dtype,
            "shape": tuple(shape),
            "offsets": (start, end),
        }
        ranges.append((start, end, name))
    if not tensors:
        raise InspectionError("safetensors header has no tensors")
    cursor = 0
    for start, end, name in sorted(ranges):
        if start != cursor:
            raise InspectionError(
                f"tensor {name!r} leaves a payload hole or overlaps another tensor"
            )
        cursor = end
    if cursor != payload_bytes:
        raise InspectionError("safetensors payload has trailing bytes")
    return tensors


def validate_known_julia_tensors(tensors: Mapping[str, Mapping[str, Any]]) -> None:
    """Check only named tensors whose names and shapes follow source directly."""
    for name, expected_shape in KNOWN_JULIA_TENSORS.items():
        tensor = tensors.get(name)
        if tensor is None:
            raise InspectionError(f"required Julia-owned tensor {name!r} is absent")
        if tensor["shape"] != expected_shape:
            raise InspectionError(
                f"Julia-owned tensor {name!r} has shape {tensor['shape']}, "
                f"expected {expected_shape}"
            )


def _tensor_contract_sha256(tensors: Mapping[str, Mapping[str, Any]]) -> str:
    contract = {
        name: {"dtype": tensor["dtype"], "shape": list(tensor["shape"])}
        for name, tensor in sorted(tensors.items())
    }
    encoded = json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _published_header_contract() -> dict[str, Any]:
    return _json_object(_read_regular(HEADER_CONTRACT), "Julia header contract")


def validate_published_header(
    prefix_and_header: bytes, file_bytes: int, tensors: Mapping[str, Mapping[str, Any]]
) -> None:
    """Verify the complete pinned tensor-name/type/shape set by its digest."""
    contract = _published_header_contract()
    if contract.get("source_revision") != SOURCE_REVISION:
        raise InspectionError("Julia header contract revision does not match inspector")
    if contract.get("declared_model_bytes") != file_bytes:
        raise InspectionError(
            "model.safetensors byte length differs from pinned header"
        )
    header = prefix_and_header[8:]
    if (
        contract.get("header_bytes") != len(header)
        or contract.get("header_sha256") != hashlib.sha256(header).hexdigest()
    ):
        raise InspectionError("model.safetensors header differs from the pinned header")
    if contract.get("tensor_count") != len(tensors) or contract.get(
        "tensor_contract_sha256"
    ) != _tensor_contract_sha256(tensors):
        raise InspectionError(
            "model.safetensors tensor names, dtypes, or shapes differ from the pinned contract"
        )


def _read_regular(path: Path, maximum: int = MAX_METADATA_BYTES) -> bytes:
    try:
        status = path.stat()
    except OSError as error:
        raise InspectionError(
            f"required artifact file is unavailable: {path.name}"
        ) from error
    if not path.is_file() or status.st_size > maximum:
        raise InspectionError(
            f"required artifact file is invalid or too large: {path.name}"
        )
    return path.read_bytes()


def _validate_config(path: Path, expected: Mapping[str, Any]) -> None:
    config = _json_object(_read_regular(path), path.name)
    for field, value in expected.items():
        if config.get(field) != value:
            raise InspectionError(
                f"{path.name} field {field!r} is {config.get(field)!r}, expected {value!r}"
            )


def _read_header_from_file(
    path: Path, declared_file_bytes: int | None
) -> tuple[bytes, int]:
    try:
        status = path.stat()
    except OSError as error:
        raise InspectionError("model.safetensors is unavailable") from error
    if not path.is_file():
        raise InspectionError("model.safetensors is not a regular file")
    file_bytes = status.st_size if declared_file_bytes is None else declared_file_bytes
    with path.open("rb") as stream:
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise InspectionError("model.safetensors prefix needs eight bytes")
        header_bytes = struct.unpack("<Q", prefix)[0]
        if header_bytes > MAX_HEADER_BYTES:
            raise InspectionError("safetensors header exceeds the inspection limit")
        header = stream.read(header_bytes)
    if len(header) != header_bytes:
        raise InspectionError("model.safetensors ended before its header")
    return prefix + header, file_bytes


def inspect_artifact(
    root: Path,
    *,
    declared_model_bytes: int | None = None,
    verify_published_header: bool = True,
) -> dict[str, Any]:
    """Inspect a local Julia artifact without loading model payload bytes.

    ``declared_model_bytes`` is reserved for a header-only fixture.  It must
    never be used as proof that a complete checkpoint is locally present.
    """
    if not root.is_dir():
        raise InspectionError("artifact root is not a directory")
    _validate_config(root / "config.json", TOP_LEVEL_CONFIG)
    _validate_config(root / "julia_config.json", JULIA_CONFIG)
    _validate_config(root / "encoder" / "config.json", ENCODER_CONFIG)
    tokenizer = root / "tokenizer"
    if not tokenizer.is_dir():
        raise InspectionError("tokenizer directory is unavailable")
    header, file_bytes = _read_header_from_file(
        root / "model.safetensors", declared_model_bytes
    )
    tensors = parse_safetensors_header(header, file_bytes)
    validate_known_julia_tensors(tensors)
    if verify_published_header:
        validate_published_header(header, file_bytes, tensors)
    return {
        "source_repository": SOURCE_REPOSITORY,
        "source_revision": SOURCE_REVISION,
        "tensor_count": len(tensors),
        "declared_model_bytes": file_bytes,
        "known_julia_tensors_checked": sorted(KNOWN_JULIA_TENSORS),
        "strict_tensor_keyset": (
            "verified against pinned complete tensor-name/type/shape contract"
            if verify_published_header
            else "unverified: published-header verification was explicitly disabled"
        ),
        "native_support": "unsupported",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path, help="local Julia artifact directory")
    parser.add_argument(
        "--declared-model-bytes",
        type=int,
        help="header-only fixture size; does not establish a complete checkpoint",
    )
    args = parser.parse_args(argv)
    try:
        report = inspect_artifact(
            args.artifact, declared_model_bytes=args.declared_model_bytes
        )
    except InspectionError as error:
        print(f"inspection failed: {error}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
