"""Checkpoint provenance for Qwen CPU reference captures.

This module intentionally uses only the standard library so its index and path
checks can run without importing Torch or Transformers.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

INDEX_FILENAME = "model.safetensors.index.json"
SINGLE_FILENAME = "model.safetensors"
MAX_INDEX_BYTES = 1024 * 1024
_SAFE_SHARD_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\.safetensors\Z")


class CheckpointIndexError(ValueError):
    """The local sharded checkpoint index has an unsafe or invalid shape."""


def is_safe_shard_filename(name: object) -> bool:
    """Return whether ``name`` is a flat safetensors file name."""
    return isinstance(name, str) and bool(_SAFE_SHARD_NAME.fullmatch(name))


def required_shards(index_path: Path) -> tuple[str, ...]:
    """Read an index and return its unique, validated shard file names."""
    try:
        size = index_path.stat().st_size
    except OSError as error:
        raise FileNotFoundError(f"missing checkpoint index: {index_path}") from error
    if size > MAX_INDEX_BYTES:
        raise CheckpointIndexError(
            f"checkpoint index exceeds {MAX_INDEX_BYTES} byte limit"
        )
    try:
        encoded = index_path.read_bytes()
    except (OSError, json.JSONDecodeError) as error:
        raise CheckpointIndexError(f"invalid checkpoint index: {error}") from error
    if len(encoded) > MAX_INDEX_BYTES:
        raise CheckpointIndexError("checkpoint index changed size while reading")
    try:
        payload = json.loads(encoded)
    except json.JSONDecodeError as error:
        raise CheckpointIndexError(f"invalid checkpoint index: {error}") from error
    if not isinstance(payload, dict):
        raise CheckpointIndexError("checkpoint index root must be an object")
    weight_map = payload.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise CheckpointIndexError("checkpoint index requires a nonempty weight_map")
    shards: set[str] = set()
    for tensor_name, filename in weight_map.items():
        if not isinstance(tensor_name, str) or not tensor_name:
            raise CheckpointIndexError("checkpoint index has an invalid tensor name")
        if not is_safe_shard_filename(filename):
            raise CheckpointIndexError("checkpoint index has an unsafe shard filename")
        shards.add(filename)
    return tuple(sorted(shards))


def checkpoint_weight_provenance(
    model_dir: Path, sha256_file: Callable[[Path], str]
) -> dict[str, Any]:
    """Return backward-compatible single-file or indexed-shard provenance.

    A present ``model.safetensors`` retains the established ``weights_sha256``
    receipt. Otherwise the fixed index and every shard it names are hashed.
    """
    single_path = model_dir / SINGLE_FILENAME
    if single_path.is_file():
        return {"weights_sha256": sha256_file(single_path)}

    index_path = model_dir / INDEX_FILENAME
    shards = required_shards(index_path)
    records = []
    for filename in shards:
        path = model_dir / filename
        if not path.is_file():
            raise FileNotFoundError(f"required checkpoint shard is missing: {path}")
        records.append({"filename": filename, "sha256": sha256_file(path)})
    return {
        "weights_index_sha256": sha256_file(index_path),
        "weight_shards": records,
    }
