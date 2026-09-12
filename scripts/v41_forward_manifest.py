"""Validate Metallix's intentionally small V4.1 CPU-reference graph manifest.

The manifest is a fixed synthetic contract for one reduced text-forward graph.
It is not a generic model-config parser, a checkpoint loader, or evidence that
the released model executes.  Its purpose is to make shortcut fixtures fail
before a reference runner allocates any tensors.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from typing import Any

GROUP_WIDTH = 32
EXPECTED_RATIOS = [0, 2, 2, 1, 1]
EXPECTED_KV_SOURCES = [1, 3]
EXPECTED_INDEX_SOURCES = [1, 3, 4]
EXPECTED_CANDIDATE_SOURCE = 3
EXPECTED_CANDIDATE_CONSUMERS = [4]


class ManifestError(ValueError):
    """The fixed reduced-forward contract was weakened or made incoherent."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ManifestError(message)


def _integer(value: object, name: str) -> int:
    _require(
        isinstance(value, int) and not isinstance(value, bool),
        f"{name} must be an integer",
    )
    return value


def _positive(value: object, name: str) -> int:
    result = _integer(value, name)
    _require(result > 0, f"{name} must be positive")
    return result


def _mapping(value: object, name: str) -> dict[str, Any]:
    _require(isinstance(value, dict), f"{name} must be an object")
    return value


def _list_of_integers(value: object, name: str) -> list[int]:
    _require(isinstance(value, list), f"{name} must be a list")
    return [_integer(item, f"{name}[{index}]") for index, item in enumerate(value)]


_MANIFEST: dict[str, Any] = {
    "schema_version": 1,
    "scope": "fixed synthetic CPU-reference-first reduced V4.1 text-forward graph",
    "not_a_claim": [
        "released-checkpoint execution",
        "upstream GPU-kernel parity",
        "production serving readiness",
    ],
    "oracle": {
        "tier": "independent_cpu_graph_with_explicit_kernel_substitutions",
        "substitutions_must_preserve": ["bf16 boundaries", "fp8/fp4 codes", "scales"],
        "forbidden": ["identity layers", "omitted Engram", "omitted routing"],
    },
    "model": {
        "n_layers": 5,
        "n_mtp_layers": 0,
        "vocab_size": 8,
        "dim": 128,
        "n_heads": 2,
        "head_dim": 64,
        "rope_head_dim": 32,
        "q_lora_rank": 32,
        "o_groups": 2,
        "o_lora_rank": 32,
        "window_size": 6,
        "compress_ratios": [0, 2, 2, 1, 1],
        "kv_source_layers": [1, 3],
        "index_source_layers": [1, 3, 4],
        "index_n_heads": 2,
        "index_head_dim": 64,
        "index_topk": 1,
        "candidate_source_layer": 3,
        "candidate_consumers": [4],
        "candidate_topk_blocks": 2,
        "candidate_block_size": 1,
        "hc_mult": 2,
        "n_routed_experts": 4,
        "n_activated_experts": 2,
        "n_shared_experts": 1,
        "moe_inter_dim": 128,
        "quantization": {
            "linear_fp8": {"group_width": 32, "scale_format": "e8m0"},
            "linear_fp4_weight": {"group_width": 32, "scale_format": "e8m0"},
            "window_kv_fp8": {"group_width": 32, "scale_format": "e8m0"},
            "index_fp4": {"group_width": 32, "scale_format": "e8m0"},
            "compressed_kv_fp4": {"group_width": 16, "scale_format": "e4m3"},
            "group_aligned_widths": [32, 64, 128],
        },
    },
    "engram": {
        "required_layer_ids": [1, 3],
        "max_ngram_size": 4,
        "n_heads": 2,
        "head_dim": 32,
        "synthetic_tokenizer": {
            "decoded_tokens": [
                " The",
                "the",
                "THE",
                " ",
                "caf\u00e9",
                "CAFE",
                "\ufffd",
                "",
            ],
            "raw_token_strings": [
                " The",
                "the",
                "THE",
                " ",
                "caf\u00e9",
                "CAFE",
                "<0xff>",
                "",
            ],
            "expected_compressed_token_map": [0, 0, 0, 1, 2, 2, 3, 4],
            "expected_compressed_vocab_size": 5,
            "raw_pad_id": 3,
            "expected_compressed_pad_id": 1,
        },
        "source_derived_layout": {
            "engram_vocab_size": 5,
            "num_embeddings": [72, 204],
            "prime_generation": "globally unique primes above engram_vocab_size minus one",
            "multiplier_generation": "NumPy default_rng(10007 * layer_id), captured before use",
        },
        "status": "ready",
        "capture_state": "not_yet_captured",
    },
    "trace": [
        {"kind": "prefill", "start_pos": 0, "token_count": 5},
        {"kind": "decode", "start_pos": 5, "token_count": 1},
        {"kind": "decode", "start_pos": 6, "token_count": 1},
    ],
}


def manifest() -> dict[str, Any]:
    """Return an isolated copy so tests and callers cannot mutate the baseline."""
    return copy.deepcopy(_MANIFEST)


def _validate_group_widths(model: dict[str, Any]) -> None:
    quantization = _mapping(model.get("quantization"), "model.quantization")
    required_paths = {
        "linear_fp8": (32, "e8m0"),
        "linear_fp4_weight": (32, "e8m0"),
        "window_kv_fp8": (32, "e8m0"),
        "index_fp4": (32, "e8m0"),
        "compressed_kv_fp4": (16, "e4m3"),
    }
    for key, (width, scale_format) in required_paths.items():
        path = _mapping(quantization.get(key), f"model.quantization.{key}")
        _require(
            _positive(path.get("group_width"), f"model.quantization.{key}.group_width")
            == width,
            f"model.quantization.{key}.group_width must remain {width}",
        )
        _require(
            path.get("scale_format") == scale_format,
            f"model.quantization.{key}.scale_format must remain {scale_format}",
        )
    widths = _list_of_integers(
        quantization.get("group_aligned_widths"),
        "model.quantization.group_aligned_widths",
    )
    _require(
        widths == [32, 64, 128], "group-aligned widths must be the fixed bounded set"
    )
    for name in (
        "dim",
        "head_dim",
        "rope_head_dim",
        "q_lora_rank",
        "o_lora_rank",
        "index_head_dim",
        "moe_inter_dim",
    ):
        width = _positive(model.get(name), f"model.{name}")
        _require(
            width % GROUP_WIDTH == 0,
            f"model.{name} must be an exact multiple of group width {GROUP_WIDTH}",
        )


def _validate_engram(engram: dict[str, Any]) -> bool:
    layer_ids = _list_of_integers(
        engram.get("required_layer_ids"), "engram.required_layer_ids"
    )
    _require(layer_ids == [1, 3], "Engram layer placement must remain [1, 3]")
    _require(
        _positive(engram.get("max_ngram_size"), "engram.max_ngram_size") == 4,
        "Engram max n-gram size must remain 4",
    )
    _require(
        _positive(engram.get("n_heads"), "engram.n_heads") == 2,
        "Engram head count must remain 2",
    )
    _require(
        _positive(engram.get("head_dim"), "engram.head_dim") == GROUP_WIDTH,
        "Engram head dimension must remain group aligned",
    )
    tokenizer = _mapping(
        engram.get("synthetic_tokenizer"), "engram.synthetic_tokenizer"
    )
    decoded = tokenizer.get("decoded_tokens")
    raw = tokenizer.get("raw_token_strings")
    expected_map = _list_of_integers(
        tokenizer.get("expected_compressed_token_map"),
        "engram.synthetic_tokenizer.expected_compressed_token_map",
    )
    _require(
        isinstance(decoded, list) and all(isinstance(item, str) for item in decoded),
        "Engram decoded tokens must be strings",
    )
    _require(
        isinstance(raw, list) and all(isinstance(item, str) for item in raw),
        "Engram raw token strings must be strings",
    )
    _require(
        len(decoded) == len(raw) == len(expected_map) == 8,
        "Engram synthetic tokenizer must keep eight fixed tokens",
    )
    _require(
        expected_map == [0, 0, 0, 1, 2, 2, 3, 4], "Engram compressed token map drifted"
    )
    _require(
        _positive(
            tokenizer.get("expected_compressed_vocab_size"),
            "engram.synthetic_tokenizer.expected_compressed_vocab_size",
        )
        == 5,
        "Engram compressed vocabulary size must remain five",
    )
    _require(
        _integer(tokenizer.get("raw_pad_id"), "engram.synthetic_tokenizer.raw_pad_id")
        == 3,
        "Engram raw pad ID must remain three",
    )
    _require(
        _integer(
            tokenizer.get("expected_compressed_pad_id"),
            "engram.synthetic_tokenizer.expected_compressed_pad_id",
        )
        == 1,
        "Engram compressed pad ID must remain one",
    )
    layout = _mapping(
        engram.get("source_derived_layout"), "engram.source_derived_layout"
    )
    _require(
        _positive(
            layout.get("engram_vocab_size"),
            "engram.source_derived_layout.engram_vocab_size",
        )
        == 5,
        "Engram prime start must remain five",
    )
    _require(
        _list_of_integers(
            layout.get("num_embeddings"), "engram.source_derived_layout.num_embeddings"
        )
        == [72, 204],
        "Engram table sizes must cover the source-derived buckets",
    )
    status = engram.get("status")
    _require(status == "ready", "Engram must remain source-runner ready")
    capture_state = engram.get("capture_state")
    _require(
        capture_state in {"not_yet_captured", "captured"},
        "Engram capture state must be explicit",
    )
    return capture_state == "captured"


def _validate_schedule(model: dict[str, Any]) -> None:
    n_layers = _positive(model.get("n_layers"), "model.n_layers")
    _require(n_layers == 5, "n_layers must remain the five-layer reduced schedule")
    _require(
        _integer(model.get("n_mtp_layers"), "model.n_mtp_layers") == 0,
        "MTP is out of scope",
    )
    ratios = _list_of_integers(model.get("compress_ratios"), "model.compress_ratios")
    _require(
        len(ratios) == n_layers and all(ratio >= 0 for ratio in ratios),
        "compression ratios must be nonnegative per backbone layer",
    )
    kv_sources = _list_of_integers(
        model.get("kv_source_layers"), "model.kv_source_layers"
    )
    index_sources = _list_of_integers(
        model.get("index_source_layers"), "model.index_source_layers"
    )
    _require(kv_sources == EXPECTED_KV_SOURCES, "KV sources must remain [1, 3]")
    _require(
        index_sources == EXPECTED_INDEX_SOURCES, "index sources must remain [1, 3, 4]"
    )
    _require(
        all(0 <= layer < n_layers for layer in kv_sources + index_sources),
        "source layer is outside the backbone",
    )
    _require(
        all(ratios[layer] > 0 for layer in kv_sources),
        "KV source must own a compressed domain",
    )
    candidate_source = _integer(
        model.get("candidate_source_layer"), "model.candidate_source_layer"
    )
    consumers = _list_of_integers(
        model.get("candidate_consumers"), "model.candidate_consumers"
    )
    _require(
        candidate_source == EXPECTED_CANDIDATE_SOURCE,
        "candidate source must remain layer 3",
    )
    _require(
        consumers == EXPECTED_CANDIDATE_CONSUMERS,
        "candidate consumer must remain layer 4",
    )
    _require(
        candidate_source in index_sources, "candidate source must publish index scores"
    )
    _require(
        all(layer in index_sources and layer > candidate_source for layer in consumers),
        "candidate consumer must be a later index source",
    )
    _require(
        all(ratios[layer] == ratios[candidate_source] for layer in consumers),
        "candidate producer and consumers must share a compressed-position domain",
    )
    _require(
        ratios == EXPECTED_RATIOS, "compression ratios must remain [0, 2, 2, 1, 1]"
    )
    _require(
        _positive(model.get("candidate_topk_blocks"), "model.candidate_topk_blocks")
        == 2,
        "candidate top-k blocks must remain two",
    )
    _require(
        _positive(model.get("candidate_block_size"), "model.candidate_block_size") == 1,
        "candidate block size must remain one",
    )


def _validate_model_shape(model: dict[str, Any]) -> None:
    dim = _positive(model.get("dim"), "model.dim")
    heads = _positive(model.get("n_heads"), "model.n_heads")
    head_dim = _positive(model.get("head_dim"), "model.head_dim")
    rope = _positive(model.get("rope_head_dim"), "model.rope_head_dim")
    _validate_group_widths(model)
    _require(heads * head_dim == dim, "n_heads * head_dim must equal dim")
    _require(
        rope <= head_dim and rope % 2 == 0,
        "rope_head_dim must be positive, even, and at most head_dim",
    )
    _require(
        _positive(model.get("o_groups"), "model.o_groups") == heads,
        "o_groups must preserve one group per attention head",
    )
    index_heads = _positive(model.get("index_n_heads"), "model.index_n_heads")
    _require(index_heads == heads, "index heads must preserve paired attention heads")
    _require(
        _positive(model.get("index_topk"), "model.index_topk") == 1,
        "index_topk must remain one",
    )
    _require(
        _positive(model.get("window_size"), "model.window_size") == 6,
        "window_size must remain six",
    )
    _require(
        _positive(model.get("hc_mult"), "model.hc_mult") >= 2,
        "hc_mult must retain multiple residual copies",
    )
    routed = _positive(model.get("n_routed_experts"), "model.n_routed_experts")
    activated = _positive(model.get("n_activated_experts"), "model.n_activated_experts")
    _require(
        routed >= 2 and 1 < activated <= routed,
        "routing must retain nontrivial top-k selection",
    )
    _require(
        _positive(model.get("n_shared_experts"), "model.n_shared_experts") == 1,
        "source MoE requires exactly one shared expert",
    )
    # This is one bounded synthetic graph, not a general shape validator. Group
    # divisibility alone would admit enormous allocations or all-expert routing.
    for name in (
        "dim",
        "n_heads",
        "head_dim",
        "rope_head_dim",
        "q_lora_rank",
        "o_lora_rank",
        "index_head_dim",
        "moe_inter_dim",
        "hc_mult",
        "n_routed_experts",
        "n_activated_experts",
    ):
        _require(
            model.get(name) == _MANIFEST["model"][name],
            f"model.{name} must retain the bounded fixture value",
        )


def _validate_trace(trace: object, *, window_size: int) -> dict[str, int]:
    _require(
        isinstance(trace, list) and len(trace) == 3,
        "trace must contain one prefill and two singleton decodes",
    )
    expected = (("prefill", 0, 5), ("decode", 5, 1), ("decode", 6, 1))
    end = 0
    for index, (kind, start, count) in enumerate(expected):
        call = _mapping(trace[index], f"trace[{index}]")
        _require(call.get("kind") == kind, f"trace[{index}] must be {kind}")
        _require(
            _integer(call.get("start_pos"), f"trace[{index}].start_pos") == end,
            "prefill/decode calls must be contiguous",
        )
        _require(
            _integer(call.get("start_pos"), f"trace[{index}].start_pos") == start,
            f"trace[{index}] has an invalid start_pos",
        )
        _require(
            _positive(call.get("token_count"), f"trace[{index}].token_count") == count,
            f"trace[{index}] has an invalid token count",
        )
        end = start + count
    _require(
        5 % 2 == 1 and 6 % 2 == 0, "trace must complete a ratio-two group during decode"
    )
    _require(
        6 % window_size == 0, "trace must cross the window-ring boundary during decode"
    )
    return {"prefill_tokens": 5, "decode_tokens": 2, "final_position": end}


def validate_manifest(
    candidate: dict[str, Any], *, require_execution_ready: bool = False
) -> dict[str, Any]:
    """Validate the fixed contract without allocating model or checkpoint tensors."""
    _require(
        _integer(candidate.get("schema_version"), "schema_version") == 1,
        "schema_version must be 1",
    )
    _require(
        candidate.get("scope") == _MANIFEST["scope"],
        "scope must retain the reduced synthetic graph boundary",
    )
    model = _mapping(candidate.get("model"), "model")
    _validate_schedule(model)
    _validate_model_shape(model)
    engram = _mapping(candidate.get("engram"), "engram")
    engram_captured = _validate_engram(engram)
    tokenizer = _mapping(engram["synthetic_tokenizer"], "engram.synthetic_tokenizer")
    decoded_tokens = tokenizer["decoded_tokens"]
    _require(
        _positive(model.get("vocab_size"), "model.vocab_size") == len(decoded_tokens),
        "model vocab_size must equal the synthetic tokenizer length for closed-loop decode",
    )
    trace = _validate_trace(
        trace=candidate.get("trace"), window_size=model["window_size"]
    )
    return {
        "validated": True,
        "allocated_tensors": 0,
        "ready_for_source_attempt": True,
        "reference_capture_complete": engram_captured,
        "coverage": {
            "window_attention": [0],
            "ratio_two_publication_and_reuse": [1, 2],
            "ratio_one_candidate_producer_and_consumer": [3, 4],
            "hyper_connections": True,
            "routed_and_shared_experts": True,
            "engram_required_before_execution": True,
            "final_fp32_logits_required": True,
        },
        "trace": trace,
    }


def model_args(candidate: dict[str, Any]) -> dict[str, Any]:
    """Return only source ``ModelArgs`` fields for the fixed text-only runner.

    Manifest-only validation metadata never reaches the pinned constructor.
    Lists become tuples because the source type contract uses immutable schedules.
    """
    validate_manifest(candidate, require_execution_ready=True)
    model = copy.deepcopy(_mapping(candidate["model"], "model"))
    for metadata in ("quantization", "candidate_consumers"):
        del model[metadata]
    for name in ("compress_ratios", "kv_source_layers", "index_source_layers"):
        model[name] = tuple(model[name])
    engram = _mapping(candidate["engram"], "engram")
    layout = _mapping(engram["source_derived_layout"], "engram.source_derived_layout")
    tokenizer = _mapping(engram["synthetic_tokenizer"], "engram.synthetic_tokenizer")
    model.update(
        {
            "max_batch_size": 1,
            "max_seq_len": 8,
            "temperature": 1.0,
            "dtype": "fp8",
            "expert_dtype": "fp4",
            "score_func": "sqrtsoftplus",
            "gate_temp": 1.0,
            "norm_topk_prob": True,
            "route_scale": 1.0,
            "swiglu_limit": 0.0,
            "norm_eps": 1e-20,
            "compress_rope_theta": 40000.0,
            "original_seq_len": 0,
            "rope_theta": 10000.0,
            "rope_factor": 40.0,
            "beta_fast": 32,
            "beta_slow": 1,
            "hc_sinkhorn_iters": 20,
            "hc_eps": 1e-6,
            "engram_layer_ids": tuple(engram["required_layer_ids"]),
            "engram_num_embeddings": tuple(layout["num_embeddings"]),
            "engram_max_ngram_size": engram["max_ngram_size"],
            "engram_vocab_size": layout["engram_vocab_size"],
            "engram_n_heads": engram["n_heads"],
            "engram_head_dim": engram["head_dim"],
            "engram_pad_id": tokenizer["raw_pad_id"],
            "engram_compressed_vocab_size": tokenizer["expected_compressed_vocab_size"],
            "vision_n_layers": 0,
            "dspark_block_size": 0,
            "dspark_noise_token_id": 0,
            "dspark_target_layer_ids": (),
            "dspark_n_routed_experts": 0,
            "dspark_n_activated_experts": 0,
        }
    )
    return model


def synthetic_tokenizer_spec(candidate: dict[str, Any]) -> dict[str, Any]:
    """Return the fixed raw-token projection the runner must expose to Engram."""
    validate_manifest(candidate, require_execution_ready=True)
    return copy.deepcopy(_mapping(candidate["engram"], "engram")["synthetic_tokenizer"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--require-execution-ready",
        action="store_true",
        help="assert that this fixed source-runner manifest is executable",
    )
    args = parser.parse_args()
    candidate = manifest()
    try:
        receipt = validate_manifest(
            candidate, require_execution_ready=args.require_execution_ready
        )
    except ManifestError as error:
        print(f"V4.1 reduced-forward manifest rejected: {error}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {"manifest": candidate, "validation_receipt": receipt},
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
