"""Project fixed layer-three candidate-producer source boundaries into a fixture.

This is an offline, storage-preserving exporter.  It consumes a completed local
source capture and copies exact tensor records; it performs neither projections,
quantization, scoring, masking, nor selection arithmetic.  The fixed capture
only covers the layer-three producer's prefill plus two singleton decode calls.
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
EXPECTED_START_POSITIONS = (0, 5, 6)
EXPECTED_OFFSETS = (5, 6, 6)
MAX_INPUT_BYTES = 16 * 1024 * 1024
MAX_FIXTURE_BYTES = 128 * 1024
BF16_BYTES = 2
FP8_BYTES = 1
INT32_BYTES = 4
FP32_BYTES = 4
COMPLEX64_BYTES = 8
LOWER_HEX_RE = re.compile(r"^[0-9a-f]*$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
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
        raise TypeError(f"candidate capture lacks object {label}")
    return value


def _require_int(value: object, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"candidate capture has invalid integer {label}")
    return value


def _tensor(
    record: object,
    label: str,
    *,
    dtype: str,
    shape: list[int],
    byte_width: int,
    negative_infinity_only: bool = False,
) -> dict[str, Any]:
    """Validate exact source storage, including finite BF16/complex values."""
    value = _require_dict(record, label)
    if value.get("dtype") != dtype:
        raise RuntimeError(
            f"candidate capture expected {label} to be {dtype}, got {value.get('dtype')!r}"
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
            f"candidate capture expected {label} shape {shape}, got {actual_shape!r}"
        )
    expected_numel = _product(shape)
    if value.get("numel") != expected_numel:
        raise RuntimeError(
            f"candidate capture expected {label} numel {expected_numel}, got {value.get('numel')!r}"
        )
    storage_hex = value.get("storage_hex")
    if not isinstance(storage_hex, str) or not LOWER_HEX_RE.fullmatch(storage_hex):
        raise TypeError(
            f"candidate capture lacks lowercase hexadecimal storage for {label}"
        )
    expected_bytes = expected_numel * byte_width
    if len(storage_hex) != expected_bytes * 2:
        raise RuntimeError(f"candidate capture has invalid storage length for {label}")
    raw = bytes.fromhex(storage_hex)
    if value.get("storage_sha256") != _sha256_bytes(raw):
        raise RuntimeError(f"candidate capture has invalid storage hash for {label}")

    if dtype == "torch.bfloat16":
        words = [word for (word,) in struct.iter_unpack("<H", raw)]
        nonfinite = [word for word in words if ((word >> 7) & 0xFF) == 0xFF]
        if negative_infinity_only:
            if (
                value.get("finite") is not False
                or not nonfinite
                or any(word != 0xFF80 for word in nonfinite)
            ):
                raise RuntimeError(
                    f"candidate capture requires {label} non-finites to be negative infinity"
                )
        elif value.get("finite") is not True or nonfinite:
            raise RuntimeError(
                f"candidate capture requires finite BF16 storage for {label}"
            )
    elif dtype == "torch.complex64":
        if value.get("finite") is not True or any(
            not math.isfinite(part)
            for part in struct.unpack(f"<{expected_numel * 2}f", raw)
        ):
            raise RuntimeError(
                f"candidate capture requires finite complex64 storage for {label}"
            )
    elif dtype == "torch.bool":
        if value.get("finite") is not True or any(byte not in (0, 1) for byte in raw):
            raise RuntimeError(
                f"candidate capture requires boolean 0/1 storage for {label}"
            )
    elif dtype == "torch.float8_e4m3fn":
        if value.get("finite") is not True or any(byte & 0x7F == 0x7F for byte in raw):
            raise RuntimeError(
                f"candidate capture requires finite E4M3FN storage for {label}"
            )
    elif dtype == "torch.float8_e8m0fnu":
        if value.get("finite") is not True or 0xFF in raw:
            raise RuntimeError(
                f"candidate capture requires finite E8M0FNU storage for {label}"
            )
    elif value.get("finite") is not True:
        raise RuntimeError(f"candidate capture requires finite {label}")
    return value


def _same_storage(left: dict[str, Any], right: dict[str, Any], label: str) -> None:
    if left.get("storage_sha256") != right.get("storage_sha256"):
        raise RuntimeError(f"candidate capture source boundary disagrees for {label}")


def _source(receipt: dict[str, object]) -> tuple[dict[str, Any], dict[str, Any]]:
    source = _require_dict(receipt.get("source"), "source provenance")
    runtime = _require_dict(receipt.get("runtime"), "runtime provenance")
    if runtime.get("storage_byteorder") != "little":
        raise RuntimeError("candidate fixture requires little-endian source storage")
    if source.get("revision") != PINNED_REVISION:
        raise RuntimeError("candidate fixture requires the pinned source revision")
    if source.get("model_sha256") != PINNED_MODEL_SHA256:
        raise RuntimeError("candidate fixture requires the pinned model source hash")
    for name in (
        "model_sha256",
        "kernel_source_sha256",
        "runner_sha256",
        "forward_observers_sha256",
    ):
        if not isinstance(source.get(name), str) or not SHA256_RE.fullmatch(
            source[name]
        ):
            raise RuntimeError(f"candidate fixture has invalid source hash {name}")
    return source, runtime


def candidate_fixture(receipt: dict[str, object]) -> dict[str, object]:
    """Extract source-owned layer-three candidate-producer observations."""
    if (
        receipt.get("capture_status")
        != "completed synthetic source-forward capture; no parity claim"
    ):
        raise RuntimeError("candidate fixture requires a completed source capture")
    coverage = _require_dict(receipt.get("coverage_status"), "capture coverage")
    if coverage.get("pending") != []:
        raise RuntimeError("candidate fixture requires complete capture coverage")
    source, runtime = _source(receipt)
    model_args = _require_dict(receipt.get("model_args"), "model args")
    static = _require_dict(receipt.get("attention_static"), "attention static inputs")
    parameters = _require_dict(receipt.get("encoded_parameters"), "encoded parameters")
    steps = receipt.get("steps")
    if not isinstance(steps, list):
        raise TypeError("candidate capture has invalid steps")

    required_model = {
        "max_batch_size": 1,
        "max_seq_len": 8,
        "dim": 128,
        "head_dim": 64,
        "q_lora_rank": 32,
        "rope_head_dim": 32,
        "index_n_heads": 2,
        "index_head_dim": 64,
        "candidate_source_layer": OWNER_LAYER,
        "candidate_topk_blocks": 2,
        "candidate_block_size": 1,
        "index_topk": 1,
        "window_size": 6,
    }
    for name, expected in required_model.items():
        if model_args.get(name) != expected:
            raise RuntimeError(
                f"candidate fixture expected model {name}={expected!r}, got {model_args.get(name)!r}"
            )
    ratios = model_args.get("compress_ratios")
    if (
        not isinstance(ratios, list)
        or len(ratios) <= OWNER_LAYER + 1
        or ratios[OWNER_LAYER] != 1
        or ratios[OWNER_LAYER + 1] != 1
    ):
        raise RuntimeError(
            "candidate fixture requires fixed layer-three/layer-four ratio-one schedule"
        )
    if OWNER_LAYER not in model_args.get("index_source_layers", []):
        raise RuntimeError("candidate fixture lacks layer-three index publication")
    if OWNER_LAYER not in model_args.get("kv_source_layers", []):
        raise RuntimeError(
            "candidate fixture lacks layer-three shared-cache publication"
        )
    norm_eps = model_args.get("norm_eps")
    if (
        not isinstance(norm_eps, (int, float))
        or isinstance(norm_eps, bool)
        or not math.isfinite(norm_eps)
        or norm_eps <= 0.0
    ):
        raise TypeError("candidate fixture has invalid finite positive norm epsilon")

    input_dimension = model_args["dim"]
    latent_dimension = model_args["head_dim"]
    query_rank = model_args["q_lora_rank"]
    index_heads = model_args["index_n_heads"]
    index_head_dimension = model_args["index_head_dim"]
    rope_pairs = model_args["rope_head_dim"] // 2
    frequencies = _tensor(
        static.get("layer_4_freqs_cis"),
        "attention_static.layer_4_freqs_cis",
        dtype="torch.complex64",
        shape=[model_args["max_seq_len"], rope_pairs],
        byte_width=COMPLEX64_BYTES,
    )
    encoded_parameters = {
        "layers.3.attn_norm.weight": _tensor(
            parameters.get("layers.3.attn_norm.weight"),
            "layers.3.attn_norm.weight",
            dtype="torch.bfloat16",
            shape=[input_dimension],
            byte_width=BF16_BYTES,
        ),
        "layers.3.attn.wq_a.weight": _tensor(
            parameters.get("layers.3.attn.wq_a.weight"),
            "layers.3.attn.wq_a.weight",
            dtype="torch.float8_e4m3fn",
            shape=[query_rank, input_dimension],
            byte_width=FP8_BYTES,
        ),
        "layers.3.attn.wq_a.scale": _tensor(
            parameters.get("layers.3.attn.wq_a.scale"),
            "layers.3.attn.wq_a.scale",
            dtype="torch.float8_e8m0fnu",
            shape=[1, input_dimension // 32],
            byte_width=FP8_BYTES,
        ),
        "layers.3.attn.q_norm.weight": _tensor(
            parameters.get("layers.3.attn.q_norm.weight"),
            "layers.3.attn.q_norm.weight",
            dtype="torch.bfloat16",
            shape=[query_rank],
            byte_width=BF16_BYTES,
        ),
        "layers.3.attn.indexer.wq_b.weight": _tensor(
            parameters.get("layers.3.attn.indexer.wq_b.weight"),
            "layers.3.attn.indexer.wq_b.weight",
            dtype="torch.float8_e4m3fn",
            shape=[index_heads * index_head_dimension, query_rank],
            byte_width=FP8_BYTES,
        ),
        "layers.3.attn.indexer.wq_b.scale": _tensor(
            parameters.get("layers.3.attn.indexer.wq_b.scale"),
            "layers.3.attn.indexer.wq_b.scale",
            dtype="torch.float8_e8m0fnu",
            shape=[index_heads * index_head_dimension // 32, 1],
            byte_width=FP8_BYTES,
        ),
        "layers.3.attn.indexer.weights_proj.weight": _tensor(
            parameters.get("layers.3.attn.indexer.weights_proj.weight"),
            "layers.3.attn.indexer.weights_proj.weight",
            dtype="torch.bfloat16",
            shape=[index_heads, input_dimension],
            byte_width=BF16_BYTES,
        ),
    }

    cases: list[dict[str, object]] = []
    for expected_start, expected_offset, step in zip(
        EXPECTED_START_POSITIONS, EXPECTED_OFFSETS, steps, strict=True
    ):
        item = _require_dict(step, f"step {expected_start}")
        start_pos = _require_int(item.get("start_pos"), "step start_pos")
        if start_pos != expected_start:
            raise RuntimeError(
                f"candidate fixture expected start_pos {expected_start}, got {start_pos}"
            )
        sequence = 5 if start_pos == 0 else 1
        end_pos = start_pos + sequence
        intermediate = _require_dict(
            item.get("intermediates"), f"step {start_pos} intermediates"
        )
        block_input = _require_dict(
            intermediate.get("layers.3.block_input"),
            f"step {start_pos} layer-three block input",
        )
        observation = _require_dict(
            intermediate.get("layers.3.attn.indexer_observation"),
            f"step {start_pos} layer-three indexer observation",
        )
        inputs = _require_dict(
            observation.get("inputs"), f"step {start_pos} indexer inputs"
        )
        operations = _require_dict(
            observation.get("operations"), f"step {start_pos} indexer operations"
        )
        attention_input = _tensor(
            intermediate.get("layers.3.attention_input"),
            f"step {start_pos} layer-three attention input",
            dtype="torch.bfloat16",
            shape=[1, sequence, input_dimension],
            byte_width=BF16_BYTES,
        )
        block_residual = _tensor(
            block_input.get("residual"),
            f"step {start_pos} layer-three block residual",
            dtype="torch.bfloat16",
            shape=[1, sequence, 2, input_dimension],
            byte_width=BF16_BYTES,
        )
        block_incoming_pre = _tensor(
            block_input.get("incoming_pre"),
            f"step {start_pos} layer-three block incoming pre",
            dtype="torch.float32",
            shape=[1, sequence, 2],
            byte_width=FP32_BYTES,
        )
        wq_a_output = _tensor(
            intermediate.get("layers.3.attn.wq_a"),
            f"step {start_pos} layer-three wq_a output",
            dtype="torch.bfloat16",
            shape=[1, sequence, query_rank],
            byte_width=BF16_BYTES,
        )
        q_norm_output = _tensor(
            intermediate.get("layers.3.attn.q_norm"),
            f"step {start_pos} layer-three q_norm output",
            dtype="torch.bfloat16",
            shape=[1, sequence, query_rank],
            byte_width=BF16_BYTES,
        )
        input_x = _tensor(
            inputs.get("x"),
            f"step {start_pos} indexer input x",
            dtype="torch.bfloat16",
            shape=[1, sequence, input_dimension],
            byte_width=BF16_BYTES,
        )
        input_qr = _tensor(
            inputs.get("qr"),
            f"step {start_pos} indexer input qr",
            dtype="torch.bfloat16",
            shape=[1, sequence, query_rank],
            byte_width=BF16_BYTES,
        )
        input_latent = _tensor(
            inputs.get("latent"),
            f"step {start_pos} indexer input latent",
            dtype="torch.bfloat16",
            shape=[1, sequence, latent_dimension],
            byte_width=BF16_BYTES,
        )
        shared_index_keys = _tensor(
            inputs.get("shared_index_k_prefix"),
            f"step {start_pos} score-einsum shared index-key prefix",
            dtype="torch.bfloat16",
            shape=[1, end_pos, index_head_dimension],
            byte_width=BF16_BYTES,
        )
        if (
            inputs.get("start_pos") != start_pos
            or inputs.get("offset") != expected_offset
        ):
            raise RuntimeError(
                f"candidate fixture has invalid call offsets for start_pos {start_pos}"
            )
        _same_storage(attention_input, input_x, f"step {start_pos} attention input/x")
        _same_storage(q_norm_output, input_qr, f"step {start_pos} q_norm/qr")

        expected_score_shape = [1, sequence, index_heads, end_pos]
        expected_reduced_shape = [1, sequence, end_pos]
        captured_operations: dict[str, object] = {
            "q_after_rope_fp4": _tensor(
                operations.get("q_after_rope_fp4"),
                f"step {start_pos} q_after_rope_fp4",
                dtype="torch.bfloat16",
                shape=[1, sequence, index_heads, index_head_dimension],
                byte_width=BF16_BYTES,
            ),
            "k_after_rope_fp4": _tensor(
                operations.get("k_after_rope_fp4"),
                f"step {start_pos} k_after_rope_fp4",
                dtype="torch.bfloat16",
                shape=[1, sequence, index_head_dimension],
                byte_width=BF16_BYTES,
            ),
            "weights_proj_output": _tensor(
                operations.get("weights_proj_output"),
                f"step {start_pos} weights_proj_output",
                dtype="torch.bfloat16",
                shape=[1, sequence, index_heads],
                byte_width=BF16_BYTES,
            ),
            "scaled_weights": _tensor(
                operations.get("scaled_weights"),
                f"step {start_pos} scaled_weights",
                dtype="torch.bfloat16",
                shape=[1, sequence, index_heads],
                byte_width=BF16_BYTES,
            ),
            "scores_einsum": _tensor(
                operations.get("scores_einsum"),
                f"step {start_pos} scores_einsum",
                dtype="torch.bfloat16",
                shape=expected_score_shape,
                byte_width=BF16_BYTES,
            ),
            "scores_after_relu": _tensor(
                operations.get("scores_after_relu"),
                f"step {start_pos} scores_after_relu",
                dtype="torch.bfloat16",
                shape=expected_score_shape,
                byte_width=BF16_BYTES,
            ),
            "scores_weighted_per_head": _tensor(
                operations.get("scores_weighted_per_head"),
                f"step {start_pos} scores_weighted_per_head",
                dtype="torch.bfloat16",
                shape=expected_score_shape,
                byte_width=BF16_BYTES,
            ),
            "scores_after_head_sum": _tensor(
                operations.get("scores_after_head_sum"),
                f"step {start_pos} scores_after_head_sum",
                dtype="torch.bfloat16",
                shape=expected_reduced_shape,
                byte_width=BF16_BYTES,
            ),
        }
        if start_pos == 0:
            captured_operations["scores_after_causal_mask"] = _tensor(
                operations.get("scores_after_causal_mask"),
                "step 0 scores_after_causal_mask",
                dtype="torch.bfloat16",
                shape=expected_reduced_shape,
                byte_width=BF16_BYTES,
                negative_infinity_only=True,
            )
        elif "scores_after_causal_mask" in operations:
            raise RuntimeError(
                "candidate fixture decode observation must not contain a causal-mask stage"
            )

        candidate_mask = _tensor(
            observation.get("candidate_mask_after"),
            f"step {start_pos} produced candidate mask",
            dtype="torch.bool",
            shape=expected_reduced_shape,
            byte_width=1,
        )
        output_indices = _tensor(
            observation.get("output_indices"),
            f"step {start_pos} produced output indices",
            dtype="torch.int32",
            shape=[1, sequence, 1],
            byte_width=INT32_BYTES,
        )
        cases.append(
            {
                "start_pos": start_pos,
                "block_input": {
                    "residual": block_residual,
                    "incoming_pre": block_incoming_pre,
                },
                "attention_input": attention_input,
                "wq_a_output": wq_a_output,
                "q_norm_output": q_norm_output,
                "inputs": {
                    "x": input_x,
                    "qr": input_qr,
                    "latent": input_latent,
                    "start_pos": start_pos,
                    "offset": expected_offset,
                    "shared_index_k_prefix": shared_index_keys,
                },
                "operations": captured_operations,
                "candidate_mask": candidate_mask,
                "output_indices": output_indices,
            }
        )
    if len(cases) != len(EXPECTED_START_POSITIONS):
        raise RuntimeError(
            "candidate fixture requires exactly the pinned three-call trace"
        )

    return {
        "schema_version": 1,
        "scope": (
            "layer-three HC block operands, attention-normalization weight, source "
            "candidate-producer inputs, exact operation boundaries, "
            "candidate masks, and output indices; not native arithmetic, cache ownership, "
            "scheduler generalization, or full-model parity"
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
        },
        "model": {
            "batches": 1,
            "input_dimension": input_dimension,
            "latent_dimension": latent_dimension,
            "query_rank": query_rank,
            "index_heads": index_heads,
            "index_head_dimension": index_head_dimension,
            "rope_pairs": rope_pairs,
            "norm_epsilon": norm_eps,
            "owner_layer": OWNER_LAYER,
            "cache_capacity": model_args["max_seq_len"],
            "candidate_topk_blocks": model_args["candidate_topk_blocks"],
            "candidate_block_size": model_args["candidate_block_size"],
            "index_topk": model_args["index_topk"],
            "window_size": model_args["window_size"],
            "expected_start_positions": list(EXPECTED_START_POSITIONS),
            "expected_offsets": list(EXPECTED_OFFSETS),
        },
        "frequencies": frequencies,
        "encoded_parameters": encoded_parameters,
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
    if not args.input.is_file():
        raise RuntimeError("candidate capture input must be a regular file")
    if args.input.stat().st_size > MAX_INPUT_BYTES:
        raise RuntimeError(f"candidate capture input exceeds {MAX_INPUT_BYTES} bytes")
    receipt = json.loads(
        args.input.read_text(),
        parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError(f"candidate capture rejects non-finite JSON constant {value}")
        ),
    )
    if not isinstance(receipt, dict):
        raise TypeError("candidate capture input must be a JSON object")
    fixture = candidate_fixture(receipt)
    payload = (
        json.dumps(fixture, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode()
    if len(payload) > MAX_FIXTURE_BYTES:
        raise RuntimeError(f"candidate fixture exceeds {MAX_FIXTURE_BYTES} bytes")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(payload)


if __name__ == "__main__":
    main()
