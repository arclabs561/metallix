"""Project layer-three index-key publication inputs into a compact fixture.

This extractor consumes an already-complete synthetic source-forward capture.  It
performs no key-preparation or cache arithmetic: the selected tensors are exact
storage records.  In particular, the layer-three compressor observation is
recorded by the source observer before ``Attention`` can mutate its latent, and
the cache case is the completed prefix after that call's publication.

Layer four consumes layer-three's published keys in the pinned graph.  Its
``freqs_cis`` buffer is therefore exported only after explicitly checking the
source rule which makes this valid here: layers three and four both have
non-zero compression ratio one and use the common compression RoPE schedule.
This is deliberately not a generic frequency-reuse rule.

No checkpoint is downloaded and no tensor framework is imported.  The input is
fully local JSON, and output is deterministic JSON suitable for byte-level
fixture checks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import struct
from pathlib import Path
from typing import Any

OWNER_LAYER = 3
CONSUMER_LAYER = 4
EXPECTED_START_POSITIONS = (0, 5, 6)
BF16_BYTES = 2
COMPLEX64_BYTES = 8
MAX_INPUT_BYTES = 16 * 1024 * 1024
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
LOWER_HEX_RE = re.compile(r"^[0-9a-f]*$")
PINNED_REVISION = "dba1be0a40aa45a94ad051997016db3960a90277"
PINNED_MODEL_SHA256 = "4e9ae23620edc8028ccc5d5fef552ab7fdc7dcd6f79608754fe9f67644056f65"


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _serialized_capture(receipt: dict[str, object]) -> bytes:
    return (
        json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _product(shape: list[int]) -> int:
    result = 1
    for width in shape:
        result *= width
    return result


def _require_dict(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"index-key capture lacks object {label}")
    return value


def _require_int(value: object, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"index-key capture has invalid integer {label}")
    return value


def _tensor(
    record: object,
    label: str,
    *,
    dtype: str,
    shape: list[int],
    byte_width: int,
) -> dict[str, Any]:
    """Validate a complete, little-endian source tensor without decoding it."""
    value = _require_dict(record, label)
    if value.get("dtype") != dtype:
        raise RuntimeError(
            f"index-key capture expected {label} to be {dtype}, got {value.get('dtype')!r}"
        )
    actual_shape = value.get("shape")
    if (
        not isinstance(actual_shape, list)
        or any(
            not isinstance(width, int) or isinstance(width, bool) or width <= 0
            for width in actual_shape
        )
        or actual_shape != shape
    ):
        raise RuntimeError(
            f"index-key capture expected {label} shape {shape}, got {value.get('shape')!r}"
        )
    expected_numel = _product(shape)
    if value.get("numel") != expected_numel:
        raise RuntimeError(
            f"index-key capture expected {label} numel {expected_numel}, got {value.get('numel')!r}"
        )
    if value.get("finite") is not True:
        raise RuntimeError(f"index-key capture requires finite {label}")
    storage_hex = value.get("storage_hex")
    if not isinstance(storage_hex, str) or not LOWER_HEX_RE.fullmatch(storage_hex):
        raise TypeError(f"index-key capture lacks storage for {label}")
    expected_bytes = expected_numel * byte_width
    if len(storage_hex) != expected_bytes * 2:
        raise RuntimeError(
            f"index-key capture has invalid storage length for {label}: {len(storage_hex) // 2}"
        )
    raw = bytes.fromhex(storage_hex)
    if value.get("storage_sha256") != _sha256_bytes(raw):
        raise RuntimeError(f"index-key capture has invalid storage hash for {label}")
    if dtype == "torch.bfloat16":
        if any(
            ((word >> 7) & 0xFF) == 0xFF for (word,) in struct.iter_unpack("<H", raw)
        ):
            raise RuntimeError(
                f"index-key capture requires finite BF16 storage for {label}"
            )
    elif dtype == "torch.complex64" and any(
        not math.isfinite(part)
        for part in struct.unpack(f"<{expected_numel * 2}f", raw)
    ):
        raise RuntimeError(
            f"index-key capture requires finite complex64 storage for {label}"
        )
    return value


def _prefix_tensor(
    record: dict[str, Any], *, prefix_tokens: int, key_dimension: int
) -> dict[str, Any]:
    """Return the exact contiguous ``[1, prefix_tokens, key_dimension]`` prefix."""
    raw = bytes.fromhex(record["storage_hex"])
    prefix_bytes = prefix_tokens * key_dimension * BF16_BYTES
    prefix = raw[:prefix_bytes]
    result = dict(record)
    result["shape"] = [1, prefix_tokens, key_dimension]
    result["numel"] = prefix_tokens * key_dimension
    result["storage_hex"] = prefix.hex()
    result["storage_sha256"] = _sha256_bytes(prefix)
    return result


def _source(receipt: dict[str, object]) -> tuple[dict[str, Any], dict[str, Any]]:
    source = _require_dict(receipt.get("source"), "source provenance")
    runtime = _require_dict(receipt.get("runtime"), "runtime provenance")
    if runtime.get("storage_byteorder") != "little":
        raise RuntimeError("index-key fixture requires little-endian source storage")
    revision = source.get("revision")
    if revision != PINNED_REVISION:
        raise RuntimeError("index-key fixture requires the pinned source revision")
    if source.get("model_sha256") != PINNED_MODEL_SHA256:
        raise RuntimeError("index-key fixture requires the pinned model source hash")
    for name in (
        "model_sha256",
        "kernel_source_sha256",
        "runner_sha256",
        "forward_observers_sha256",
    ):
        if not isinstance(source.get(name), str) or not SHA256_RE.fullmatch(
            source[name]
        ):
            raise RuntimeError(f"index-key fixture has invalid source hash {name}")
    return source, runtime


def index_key_fixture(receipt: dict[str, object]) -> dict[str, object]:
    """Extract the pinned layer-three inputs and completed index-key prefixes."""
    if (
        receipt.get("capture_status")
        != "completed synthetic source-forward capture; no parity claim"
    ):
        raise RuntimeError("index-key fixture requires a completed source capture")
    coverage = _require_dict(receipt.get("coverage_status"), "capture coverage")
    if coverage.get("pending") != []:
        raise RuntimeError("index-key fixture requires complete capture coverage")
    source, runtime = _source(receipt)
    model_args = _require_dict(receipt.get("model_args"), "model args")
    static = _require_dict(receipt.get("attention_static"), "attention static inputs")
    parameters = _require_dict(receipt.get("encoded_parameters"), "encoded parameters")
    steps = receipt.get("steps")
    if not isinstance(steps, list):
        raise TypeError("index-key capture has invalid steps")

    # This compact fixture describes one exact source layout, rather than a
    # generalized scheduler.  The checks make its source-layer and RoPE scope
    # explicit before exporting a tensor named after a consumer-layer buffer.
    required_model = {
        "max_batch_size": 1,
        "max_seq_len": 8,
        "dim": 128,
        "head_dim": 64,
        "index_head_dim": 64,
        "rope_head_dim": 32,
        "candidate_source_layer": OWNER_LAYER,
    }
    for name, expected in required_model.items():
        if model_args.get(name) != expected:
            raise RuntimeError(
                f"index-key fixture expected model {name}={expected!r}, got {model_args.get(name)!r}"
            )
    ratios = model_args.get("compress_ratios")
    if not isinstance(ratios, list) or len(ratios) <= CONSUMER_LAYER:
        raise RuntimeError("index-key fixture lacks owner/consumer compression ratios")
    if ratios[OWNER_LAYER] != 1 or ratios[CONSUMER_LAYER] != 1:
        raise RuntimeError(
            "index-key fixture only permits the captured owner=3/consumer=4 ratio-one schedule"
        )
    if OWNER_LAYER not in model_args.get("index_source_layers", []):
        raise RuntimeError("index-key fixture lacks layer-three index publication")
    if OWNER_LAYER not in model_args.get("kv_source_layers", []):
        raise RuntimeError(
            "index-key fixture lacks layer-three shared-cache publication"
        )
    norm_epsilon = model_args.get("norm_eps")
    if (
        not isinstance(norm_epsilon, (int, float))
        or isinstance(norm_epsilon, bool)
        or not math.isfinite(norm_epsilon)
        or norm_epsilon <= 0.0
    ):
        raise TypeError("index-key fixture has invalid finite positive norm epsilon")

    key_dimension = model_args["index_head_dim"]
    latent_dimension = model_args["head_dim"]
    rope_pairs = model_args["rope_head_dim"] // 2
    wk = _tensor(
        parameters.get("layers.3.attn.indexer.wk.weight"),
        "layers.3.attn.indexer.wk.weight",
        dtype="torch.bfloat16",
        shape=[key_dimension, latent_dimension],
        byte_width=BF16_BYTES,
    )
    norm = _tensor(
        parameters.get("layers.3.attn.indexer.k_norm.weight"),
        "layers.3.attn.indexer.k_norm.weight",
        dtype="torch.bfloat16",
        shape=[key_dimension],
        byte_width=BF16_BYTES,
    )
    frequencies = _tensor(
        static.get("layer_4_freqs_cis"),
        "attention_static.layer_4_freqs_cis",
        dtype="torch.complex64",
        shape=[model_args["max_seq_len"], rope_pairs],
        byte_width=COMPLEX64_BYTES,
    )

    cases: list[dict[str, object]] = []
    for expected_start, step in zip(EXPECTED_START_POSITIONS, steps, strict=True):
        item = _require_dict(step, f"step {expected_start}")
        start_pos = _require_int(item.get("start_pos"), "step start_pos")
        if start_pos != expected_start:
            raise RuntimeError(
                f"index-key fixture expected start_pos {expected_start}, got {start_pos}"
            )
        intermediate = _require_dict(
            item.get("intermediates"), f"step {start_pos} intermediates"
        )
        caches_after = _require_dict(
            item.get("caches_after"), f"step {start_pos} caches_after"
        )
        latent = _tensor(
            intermediate.get("layers.3.attn.compressor"),
            f"step {start_pos} layer-three pre-mutation compressor latent",
            dtype="torch.bfloat16",
            shape=[1, 5 if start_pos == 0 else 1, latent_dimension],
            byte_width=BF16_BYTES,
        )
        cache = _tensor(
            caches_after.get("layer_3.index_k"),
            f"step {start_pos} layer-three index cache",
            dtype="torch.bfloat16",
            shape=[1, model_args["max_seq_len"], key_dimension],
            byte_width=BF16_BYTES,
        )
        prefix_tokens = start_pos + latent["shape"][1]
        if prefix_tokens > model_args["max_seq_len"]:
            raise RuntimeError("index-key fixture cache prefix exceeds source capacity")
        cases.append(
            {
                "start_pos": start_pos,
                "latent": latent,
                "index_cache_after": _prefix_tensor(
                    cache, prefix_tokens=prefix_tokens, key_dimension=key_dimension
                ),
            }
        )
    if len(cases) != len(EXPECTED_START_POSITIONS):
        raise RuntimeError(
            "index-key fixture requires exactly the pinned three-call trace"
        )

    return {
        "schema_version": 1,
        "scope": (
            "layer-three source index-key preparation inputs and completed cache prefixes; "
            "not cache ownership, scheduling, native key arithmetic, or full-model parity"
        ),
        "source": {
            "revision": source["revision"],
            "model_sha256": source["model_sha256"],
            "kernel_source_sha256": source["kernel_source_sha256"],
            "runner_sha256": source["runner_sha256"],
            "forward_observers_sha256": source["forward_observers_sha256"],
            "complete_capture_sha256": _sha256_bytes(_serialized_capture(receipt)),
            "manifest_canonical_sha256": receipt.get("manifest_canonical_sha256"),
            "storage_byteorder": runtime["storage_byteorder"],
            "frequency_source": "attention_static.layer_4_freqs_cis",
            "frequency_compatibility": (
                "captured source rule: layer three and layer four both use the "
                "common non-zero compression ratio-one RoPE schedule"
            ),
        },
        "model": {
            "batches": 1,
            "latent_dimension": latent_dimension,
            "key_dimension": key_dimension,
            "rope_pairs": rope_pairs,
            "norm_epsilon": norm_epsilon,
            "owner_layer": OWNER_LAYER,
            "consumer_layer": CONSUMER_LAYER,
            "cache_capacity": model_args["max_seq_len"],
            "expected_start_positions": list(EXPECTED_START_POSITIONS),
        },
        "weights": {"wk": wk, "norm": norm},
        "frequencies": frequencies,
        "cases": cases,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, required=True, help="complete source capture JSON"
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="fixture JSON to write"
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    input_stat = args.input.stat()
    if not args.input.is_file():
        raise RuntimeError("index-key capture input must be a regular file")
    if input_stat.st_size > MAX_INPUT_BYTES:
        raise RuntimeError(
            f"index-key capture input exceeds {MAX_INPUT_BYTES} byte limit: {input_stat.st_size}"
        )
    receipt = json.loads(
        args.input.read_text(),
        parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError(f"index-key capture rejects non-finite JSON constant {value}")
        ),
    )
    if not isinstance(receipt, dict):
        raise TypeError("index-key capture input must be a JSON object")
    fixture = index_key_fixture(receipt)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(fixture, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )


if __name__ == "__main__":
    main()
