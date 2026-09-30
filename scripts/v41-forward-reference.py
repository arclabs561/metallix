#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "torch==2.13.0",
#   "numpy==2.5.3",
#   "sympy==1.14.0",
#   "tokenizers==0.23.2",
# ]
# ///
"""Run a bounded synthetic V4.1 text-forward capture through pinned source.

This is graph-composition evidence only.  It imports the hash-gated upstream
text graph and replaces its six TileLang boundaries with ``v41_cpu_kernels``.
The replacement preserves the source's BF16 / FP8 / packed-FP4 boundaries, but
does not execute CUDA or establish GPU numerical parity.  It never downloads a
checkpoint and never calls a Rust implementation for expected values.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import struct
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy
import sympy
import tokenizers
import torch

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
SOURCE_REVISION = "dba1be0a40aa45a94ad051997016db3960a90277"
KERNEL_SHA256 = "1236c3507019ed176f5dba5e04bcea58867cf654818c6cf138ed4845398c2455"
TRACE_INPUT_IDS = ((0, 1, 2, 3, 4, 5, 6),)
# Layer-zero window-only attention retains six additional exact source stages
# (frequency, Q prefix, window, and output-projection input).
MAX_HOOK_RECORDS = 104
SAMPLE_VALUES = 8
MAX_CAPTURE_BYTES = 16 << 20
# HC state and its source-produced collapse are both exact bit fixtures.  Keep
# the expanded oracle intentionally bounded, while allowing the three pinned
# prefill/decode cases to retain all required representations.
MAX_HEAD_FIXTURE_BYTES = 48 << 10
# The full layer-four MoE payload retains packed routed and FP8 shared expert
# storage.  It is intentionally bigger than the head fixture but bounded.
MAX_MOE_FIXTURE_BYTES = 512 << 10
# Layer-four attention retains all encoded projections plus three source cases.
# The reduced graph is deliberately small enough that complete tensor bytes fit
# under this limit; exceeding it is a schema/capture regression, not truncation.
MAX_ATTENTION_FIXTURE_BYTES = 512 << 10
# Layer zero retains both HC coefficient calls and the exact same-trace Engram
# handoff, but remains bounded below the complete receipt cap.
MAX_LAYER_ZERO_TO_LAYER_ONE_FIXTURE_BYTES = 576 << 10
# A generation receipt contains only two small vocab logits per default case;
# keep a separate cap so it cannot quietly become another full-forward capture.
MAX_GENERATION_ORACLE_BYTES = 64 << 10
GENERATION_PROMPTS = (
    ("prefill_five", (0, 1, 2, 3, 4)),
    ("prefill_four", (0, 1, 2, 3)),
)

sys.path.insert(0, str(SCRIPTS))
import v41_attention_capture
import v41_cpu_kernels as kernels
import v41_forward_manifest as forward_manifest
import v41_forward_observers
import v41_index_key_capture
import v41_layer0_to_layer1_capture
import v41_source_loader as source_loader


class SyntheticBackend:
    """Tokenizer backend with explicit, uniquely normalized text for every ID."""

    def __init__(
        self, decoded_texts: tuple[str, ...], raw_tokens: tuple[str, ...]
    ) -> None:
        self._decoded_texts = decoded_texts
        self._raw_tokens = raw_tokens

    def decode(self, ids: list[int], *, skip_special_tokens: bool = False) -> str:
        del skip_special_tokens
        return "".join(self._decoded_texts[token_id] for token_id in ids)

    def id_to_token(self, token_id: int) -> str:
        return self._raw_tokens[token_id]


class SyntheticTokenizer:
    """Small adapter exposing precisely the tokenizer methods Engram consumes."""

    def __init__(self, decoded_tokens: list[str], raw_token_strings: list[str]) -> None:
        if len(decoded_tokens) != len(raw_token_strings) or not decoded_tokens:
            raise ValueError(
                "synthetic tokenizer requires matching nonempty explicit token lists"
            )
        # The pinned Engram builder calls decode() first and only calls
        # id_to_token() for replacement-byte tokens.  Preserve both forms.
        self.backend_tokenizer = SyntheticBackend(
            tuple(decoded_tokens), tuple(raw_token_strings)
        )

    def __len__(self) -> int:
        return len(self.backend_tokenizer._decoded_texts)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def serialized_capture(receipt: dict[str, object]) -> bytes:
    """Return the canonical complete-capture artifact bytes for its receipt."""
    return (
        json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _json_preview_f32(value: torch.Tensor) -> float | str:
    """Keep JSON previews valid; exact nonfinite values remain in storage bytes."""
    number = float(value)
    if math.isfinite(number):
        return number
    if math.isnan(number):
        return "nan"
    return "inf" if number > 0 else "-inf"


def tensor_record(
    value: torch.Tensor, *, include_storage: bool = False
) -> dict[str, Any]:
    """Bounded tensor receipt retaining type, shape, storage and finite status."""
    tensor = value.detach().cpu().contiguous()
    # ``numpy`` cannot represent PyTorch FP8/FP4 values.  Storage bytes are
    # nevertheless exact and distinguish the source encodings from a widened
    # float summary.
    raw = tensor.view(torch.uint8).numpy().tobytes()
    if tensor.dtype == torch.float4_e2m1fn_x2:
        flat = kernels.unpack_fp4_e2m1x2(tensor.view(torch.uint8)).reshape(-1)
    elif tensor.is_complex():
        flat = torch.view_as_real(tensor).reshape(-1)
    else:
        flat = tensor.float().reshape(-1)
    # Masked source scores deliberately contain -inf.  JSON's nonfinite
    # literals are invalid under allow_nan=False, so only the diagnostic
    # preview uses explicit strings; storage_hex remains the exact oracle.
    sample = [_json_preview_f32(item) for item in flat[:SAMPLE_VALUES]]
    receipt = {
        "dtype": str(tensor.dtype),
        "shape": list(tensor.shape),
        "numel": tensor.numel(),
        "storage_sha256": _sha256_bytes(raw),
        "finite": bool(torch.isfinite(flat).all()),
        "sample_f32": sample,
    }
    if include_storage:
        receipt["storage_hex"] = raw.hex()
    return receipt


def object_record(value: object, *, include_storage: bool = False) -> object:
    """Summarize hook output without retaining an unbounded model trace."""
    if isinstance(value, torch.Tensor):
        return tensor_record(value, include_storage=include_storage)
    if value is None:
        return None
    if isinstance(value, tuple):
        return [object_record(item, include_storage=include_storage) for item in value]
    if isinstance(value, list):
        return [object_record(item, include_storage=include_storage) for item in value]
    raise TypeError(f"unrecordable hook output {type(value)!r}")


def deterministic_values(shape: tuple[int, ...], ordinal: int) -> torch.Tensor:
    """Stable nonzero values without a global RNG or ambient state."""
    count = 1
    for width in shape:
        count *= width
    positions = torch.arange(count, dtype=torch.float32)
    values = torch.sin(positions * 0.173 + ordinal * 0.619) * 0.18
    return values.reshape(shape)


def _fill_scale(parameter: torch.nn.Parameter) -> None:
    # Source E8M0 scales represent powers of two.  Keep all scales finite and
    # nonzero; E4M3 compressed-KV scales have the same controlled magnitude.
    parameter.copy_(torch.full_like(parameter, 0.125))


def initialize_parameters(model: torch.nn.Module) -> dict[str, object]:
    """Populate every source parameter, preserving logical encoded storage."""
    encoded: dict[str, object] = {}
    with torch.no_grad():
        for ordinal, (name, parameter) in enumerate(model.named_parameters()):
            if name.endswith(".scale") or name == "scale":
                _fill_scale(parameter)
                kind = "quant_scale"
            elif parameter.dtype == torch.float4_e2m1fn_x2:
                # CPU Torch cannot perform arithmetic on E2M1x2.  Fill its
                # physical storage as packed low-nibble-first bytes; the CPU
                # backend reads this exact view instead of widening it.
                storage = parameter.view(torch.uint8)
                lanes = torch.arange(storage.numel(), dtype=torch.uint8).reshape_as(
                    storage
                )
                low = ((lanes + ordinal) % 8) | (((lanes // 3 + ordinal) % 2) << 3)
                high = ((lanes * 5 + ordinal) % 8) | (
                    ((lanes // 5 + ordinal + 1) % 2) << 3
                )
                storage.copy_(low | (high << 4))
                kind = "packed_fp4_e2m1x2"
            elif (
                name.endswith("norm.weight")
                or ".q_norm.weight" in name
                or ".kv_norm.weight" in name
            ):
                parameter.copy_(torch.ones_like(parameter))
                kind = "norm"
            elif name.endswith(("q_weight", "k_weight")):
                parameter.copy_(torch.ones_like(parameter))
                kind = "engram_gate_weight"
            else:
                parameter.copy_(
                    deterministic_values(tuple(parameter.shape), ordinal).to(
                        parameter.dtype
                    )
                )
                kind = "deterministic_sine"
            encoded[name] = {
                "kind": kind,
                **tensor_record(parameter, include_storage=True),
            }
    return encoded


def executable_args(
    graph: ModuleType, candidate: dict[str, Any]
) -> tuple[object, SyntheticTokenizer]:
    """Make the validated manifest's real source args and Engram tokenizer."""
    spec = forward_manifest.synthetic_tokenizer_spec(candidate)
    tokenizer = SyntheticTokenizer(spec["decoded_tokens"], spec["raw_token_strings"])
    args = graph.ModelArgs(**forward_manifest.model_args(candidate))
    return args, tokenizer


def cache_snapshots(graph: ModuleType, model: torch.nn.Module) -> dict[str, object]:
    snapshots: dict[str, object] = {}
    for layer_id, layer in enumerate(model.layers):
        attention = layer.attn
        snapshots[f"layer_{layer_id}.window_kv"] = tensor_record(
            attention.window_kv_cache, include_storage=True
        )
        if hasattr(attention, "compress_kv_cache"):
            snapshots[f"layer_{layer_id}.compressed_kv"] = tensor_record(
                attention.compress_kv_cache, include_storage=True
            )
        if attention.indexer is not None and hasattr(attention.indexer, "k_cache"):
            snapshots[f"layer_{layer_id}.index_k"] = tensor_record(
                attention.indexer.k_cache, include_storage=True
            )
    if model.engram_hash is not None:
        snapshots["engram.hash_cache"] = tensor_record(
            model.engram_hash.cache, include_storage=True
        )
    for name in ("compress_kv", "index_k", "topk_idxs", "candidates"):
        value = getattr(graph.shared_attn, name)
        snapshots[f"shared.{name}"] = (
            None if value is None else tensor_record(value, include_storage=True)
        )
    return snapshots


def initialize_runtime_buffers(model: torch.nn.Module) -> list[str]:
    """Poison-free only cache/state buffers; never overwrite RoPE frequencies."""
    initialized: list[str] = []
    suffixes = (
        "kv_state",
        "score_state",
        "window_kv_cache",
        "compress_kv_cache",
        "k_cache",
        "cache",
    )
    with torch.no_grad():
        for name, buffer in model.named_buffers():
            if name.endswith(suffixes):
                if name.endswith("score_state"):
                    # Source uses -inf as the identity for an unwritten gate
                    # score, so a malformed partial decode cannot treat a zero
                    # as a real contribution.
                    buffer.fill_(-torch.inf)
                else:
                    buffer.zero_()
                initialized.append(name)
    return initialized


def candidate_receipt(graph: ModuleType, window_size: int) -> dict[str, object]:
    """Prove the final level-two pick is constrained by the level-one mask."""
    candidates = graph.shared_attn.candidates
    selected = graph.shared_attn.topk_idxs
    if candidates is None or selected is None:
        raise RuntimeError("candidate producer or consumer did not publish a result")
    if candidates.ndim != 3 or selected.ndim != 3:
        raise RuntimeError("candidate tensors have unexpected rank")
    # Layer four shifts compressed positions after its sliding-window slots.
    compressed = selected - window_size
    valid = compressed >= 0
    if not bool(valid.any()):
        raise RuntimeError("candidate consumer did not select a compressed position")
    chosen = compressed[valid].to(torch.int64)
    if not bool(candidates[0, -1, chosen].all()):
        raise RuntimeError("candidate consumer selected outside producer mask")
    visible = candidates[0, -1]
    if visible.numel() < 3 or bool(visible.all()):
        raise RuntimeError("candidate source did not exclude a visible position")
    return {
        "producer_mask": tensor_record(candidates, include_storage=True),
        "consumer_topk": tensor_record(selected, include_storage=True),
        "selected_compressed_positions": [int(item) for item in chosen.tolist()],
        "selected_all_in_producer_mask": True,
        "producer_excludes_at_least_one_visible_position": True,
    }


def trace_calls(
    manifest: dict[str, Any], input_ids: torch.Tensor
) -> tuple[tuple[int, torch.Tensor], ...]:
    """Partition explicit IDs from the manifest's validated trace only."""
    if input_ids.ndim != 2:
        raise RuntimeError("source capture input IDs must be [batch, tokens]")
    offset = 0
    calls: list[tuple[int, torch.Tensor]] = []
    for call in manifest["trace"]:
        start_pos = call["start_pos"]
        token_count = call["token_count"]
        if start_pos != offset:
            raise RuntimeError("manifest trace must be contiguous from position zero")
        stop = offset + token_count
        calls.append((start_pos, input_ids[:, offset:stop]))
        offset = stop
    if offset != input_ids.size(1) or any(chunk.size(1) == 0 for _, chunk in calls):
        raise RuntimeError(
            "explicit source-capture input does not exactly cover manifest trace"
        )
    return tuple(calls)


def hc_step_receipt(
    records: list[dict[str, object]], n_layers: int
) -> list[dict[str, object]]:
    """Label source-order HC observations and reject a weakened trace."""
    expected = 2 * n_layers
    if len(records) != expected:
        raise RuntimeError(
            f"expected {expected} HC kernel calls for {n_layers} blocks, got {len(records)}"
        )
    labeled: list[dict[str, object]] = []
    for index, record in enumerate(records):
        if not all(record["nontrivial"].values()):
            raise RuntimeError(
                f"HC kernel call {index} lacks nontrivial coefficient evidence"
            )
        labeled.append(
            {
                "layer_id": index // 2,
                "sublayer": "attention" if index % 2 == 0 else "ffn",
                **record,
            }
        )
    return labeled


def sparse_step_receipt(
    records: list[dict[str, object]], n_layers: int
) -> list[dict[str, object]]:
    """Label source-order sparse-attention observations without weakening scope."""
    if len(records) != n_layers:
        raise RuntimeError(
            f"expected {n_layers} sparse attention calls, got {len(records)}"
        )
    return [{"layer_id": index, **record} for index, record in enumerate(records)]


def engram_receipt(
    model: torch.nn.Module, manifest: dict[str, Any]
) -> dict[str, object]:
    """Bind the actual source-derived hash layout to its declared manifest rows."""
    state = model.engram_hash
    if state is None:
        raise RuntimeError("required Engram state is absent")
    declared = manifest["engram"]["source_derived_layout"]
    layout = state.layout
    declared_rows = declared["num_embeddings"]
    actual_rows = list(layout.num_embeddings)
    implied_rows = [sum(sum(heads) for heads in layer) for layer in layout.primes]
    if actual_rows != declared_rows or actual_rows != implied_rows:
        raise RuntimeError(
            "source Engram layout rows differ from declared prime-bucket layout"
        )
    if state.token_map.numel() != len(
        manifest["engram"]["synthetic_tokenizer"]["decoded_tokens"]
    ):
        raise RuntimeError("Engram token map does not cover the declared tokenizer")
    return {
        "layout": {
            "layer_ids": list(layout.layer_ids),
            "max_ngram_size": layout.max_ngram_size,
            "n_heads": layout.n_heads,
            "head_dim": layout.head_dim,
            "num_embeddings": actual_rows,
            "implied_prime_bucket_rows": implied_rows,
        },
        "hash_state": {
            "token_map": tensor_record(state.token_map, include_storage=True),
            "primes": tensor_record(state.primes, include_storage=True),
            "offsets": tensor_record(state.offsets, include_storage=True),
            "multipliers": tensor_record(state.multipliers, include_storage=True),
            "pad_id": state.pad_id,
        },
    }


def run_capture() -> dict[str, object]:
    manifest = forward_manifest.manifest()
    validation = forward_manifest.validate_manifest(
        manifest, require_execution_ready=True
    )
    kernel_bytes = (SCRIPTS / "v41_cpu_kernels.py").read_bytes()
    if (
        _sha256_bytes((ROOT / "artifacts" / "v41-kernel-pinned.py").read_bytes())
        != KERNEL_SHA256
    ):
        raise RuntimeError(
            "retained upstream kernel hash differs from the capture contract"
        )
    kernel_bundle, hc_records, sparse_records = (
        v41_forward_observers.tracing_kernel_bundle(
            kernels, source_loader.KERNEL_NAMES, tensor_record
        )
    )
    graph = source_loader.load_text_graph(kernel_bundle)
    graph.shared_attn = graph.SharedAttentionRuntime()
    # The source expects BF16 default execution results from its replacement
    # GEMMs.  This scope always restores the caller's default dtype.
    with graph.set_dtype(torch.bfloat16):
        args, tokenizer = executable_args(graph, manifest)
        model = graph.Transformer(args, tokenizer).eval()
        tokenizer_spec = forward_manifest.synthetic_tokenizer_spec(manifest)
        if model.engram_hash is None:
            raise RuntimeError(
                "source Transformer constructed without required Engram hash state"
            )
        actual_token_map = model.engram_hash.token_map.tolist()
        expected_token_map = tokenizer_spec["expected_compressed_token_map"]
        if actual_token_map != expected_token_map:
            raise RuntimeError(
                f"source tokenizer normalization map mismatch: {actual_token_map}"
            )
        if model.engram_hash.pad_id != tokenizer_spec["expected_compressed_pad_id"]:
            raise RuntimeError(
                "source tokenizer pad normalization differs from manifest"
            )
        engram = engram_receipt(model, manifest)
        encoded = initialize_parameters(model)
        initialized_buffers = initialize_runtime_buffers(model)
        input_ids = torch.tensor(TRACE_INPUT_IDS, dtype=torch.int64)
        if input_ids.max().item() >= args.vocab_size:
            raise RuntimeError("trace input id lies outside synthetic vocabulary")
        calls = trace_calls(manifest, input_ids)
        steps: list[dict[str, object]] = []
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(941)
            with v41_forward_observers.hooks_for(
                model, graph, tensor_record, object_record, MAX_HOOK_RECORDS
            ) as intermediate:
                for start_pos, chunk in calls:
                    intermediate.clear()
                    hc_records.clear()
                    sparse_records.clear()
                    output_ids, logits, main_hidden = model(chunk, start_pos=start_pos)
                    if not bool(torch.isfinite(logits).all()):
                        raise RuntimeError(f"nonfinite logits at start_pos={start_pos}")
                    steps.append(
                        {
                            "start_pos": start_pos,
                            "input_ids": tensor_record(chunk, include_storage=True),
                            "output_ids": tensor_record(
                                output_ids, include_storage=True
                            ),
                            "logits": tensor_record(logits, include_storage=True),
                            "main_hidden": None
                            if main_hidden is None
                            else tensor_record(main_hidden),
                            "intermediates": dict(intermediate),
                            "hyper_connection_mixes": v41_forward_observers.hc_step_receipt(
                                hc_records, len(model.layers)
                            ),
                            "sparse_attention_calls": v41_forward_observers.sparse_step_receipt(
                                sparse_records, len(model.layers)
                            ),
                            "caches_after": cache_snapshots(graph, model),
                        }
                    )
        candidates = candidate_receipt(graph, args.window_size)
        attention_static = {
            "layer_3_freqs_cis": tensor_record(
                model.layers[3].attn.freqs_cis, include_storage=True
            ),
            "layer_4_freqs_cis": tensor_record(
                model.layers[4].attn.freqs_cis, include_storage=True
            ),
        }
    return {
        "schema_version": 1,
        "capture_status": "completed synthetic source-forward capture; no parity claim",
        "coverage_status": {
            "scope": "recorded boundaries below; not every internal expression or Rust acceptance",
            "captured": [
                "embedding",
                "Engram hash and residual layers",
                "block outputs and pre-mix outputs",
                "attention, compressor, indexer, routing, and FFN module outputs",
                "all source HC kernel inputs and coefficients",
                "layer-four attention projections, prepared cache reads, and sparse kernel boundaries",
                "final RMSNorm and FP32 head logits",
                "cache and candidate-filter snapshots",
            ],
            "pending": [],
        },
        "source": {
            "revision": SOURCE_REVISION,
            "model_sha256": source_loader.MODEL_SHA256,
            "engram_sha256": source_loader.ENGRAM_SHA256,
            "loader": "v41_source_loader.load_text_graph",
            "loader_sha256": _sha256_bytes(
                (SCRIPTS / "v41_source_loader.py").read_bytes()
            ),
            "kernel_source_sha256": KERNEL_SHA256,
            "cpu_backend_sha256": _sha256_bytes(kernel_bytes),
            "runner_sha256": _sha256_bytes(Path(__file__).read_bytes()),
            "attention_helper_sha256": _sha256_bytes(
                (SCRIPTS / "v41_attention_capture.py").read_bytes()
            ),
            "forward_observers_sha256": _sha256_bytes(
                (SCRIPTS / "v41_forward_observers.py").read_bytes()
            ),
        },
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": numpy.__version__,
            "sympy": sympy.__version__,
            "tokenizers": tokenizers.__version__,
            "device": "cpu",
            "storage_byteorder": sys.byteorder,
            "torch_num_threads": torch.get_num_threads(),
            "torch_num_interop_threads": torch.get_num_interop_threads(),
        },
        "manifest_validation": validation,
        "manifest_canonical_sha256": _sha256_bytes(
            json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ),
        "manifest": manifest,
        "model_args": forward_manifest.model_args(manifest),
        "synthetic_contract": {
            "tokenizer": "explicit normalization-sensitive synthetic backend",
            "normalized_token_map": actual_token_map,
            "normalized_pad_id": model.engram_hash.pad_id,
            "parameter_initializer": "deterministic sine plus exact encoded FP8/packed-FP4 storage",
            "input_ids": list(TRACE_INPUT_IDS[0]),
            "trace": ["prefill:5", "decode:1", "decode:1"],
            "sample_rng_seed": 941,
            "initialized_cache_buffers": initialized_buffers,
        },
        "kernel_substitutions": {
            "module": "scripts/v41_cpu_kernels.py",
            "hc_observation": (
                "A receipt-only wrapper delegates directly to the same CPU "
                "hc_split_sinkhorn function with unchanged arguments and result; "
                "it is interception, not a numerical substitution."
            ),
            "representation_boundaries": {
                "window_and_linear": "G32 E8M0",
                "compressed_kv": "G16 E4M3",
                "index_query_and_key": "G32 E8M0",
                "expert_weights": "packed E2M1x2 viewed as uint8 only by the CPU backend",
            },
            "not_a_claim": [
                "upstream GPU-kernel numerical parity",
                "released-checkpoint execution",
                "Rust parity",
                "production serving readiness",
            ],
        },
        "encoded_parameters": encoded,
        "engram": engram,
        "attention_static": attention_static,
        "steps": steps,
        "candidate_filtering": candidates,
    }


def _generation_admission(
    prompt_ids: tuple[int, ...],
    max_new_tokens: int,
    *,
    vocab_size: int,
    max_seq_len: int,
    eos_token_id: int | None,
) -> None:
    """Reject an invalid request before it can partially mutate source caches.

    The final selected token is returned but never re-entered into the model, so
    a request of ``n`` new tokens consumes at most ``prompt + n - 1`` model
    positions.  This matches the reduced Rust generation preflight contract.
    """
    if len(prompt_ids) < 2:
        raise ValueError("generation prompt must contain at least two token IDs")
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    if len(prompt_ids) + max_new_tokens - 1 > max_seq_len:
        raise ValueError(
            "generation request exceeds max_seq_len before its final selection"
        )
    for index, token_id in enumerate(prompt_ids):
        if not 0 <= token_id < vocab_size:
            raise ValueError(f"prompt token {index} lies outside the source vocabulary")
    if eos_token_id is not None and not 0 <= eos_token_id < vocab_size:
        raise ValueError("eos_token_id lies outside the source vocabulary")


def _lowest_id_argmax(logits: torch.Tensor) -> int:
    """Choose a finite final-position logit, resolving ties by the lowest ID."""
    if logits.ndim != 2 or logits.size(0) != 1:
        raise RuntimeError(
            "source generation requires one batch of final-position logits"
        )
    if logits.dtype != torch.float32:
        raise RuntimeError("source generation requires FP32 final-position logits")
    row = logits[0]
    if not bool(torch.isfinite(row).all()):
        raise RuntimeError("source generation produced nonfinite final-position logits")
    maximum = row.max()
    tied = torch.nonzero(row == maximum, as_tuple=False).flatten()
    if tied.numel() == 0:
        raise RuntimeError("finite source logits have no argmax")
    return int(tied.min().item())


def _canonical_json_sha256(value: object) -> str:
    return _sha256_bytes(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    )


def _generation_case(
    model: torch.nn.Module,
    *,
    name: str,
    prompt_ids: tuple[int, ...],
    max_new_tokens: int,
    eos_token_id: int | None,
) -> dict[str, object]:
    """Run one cache-respecting greedy source decode without source sampling."""
    selections: list[dict[str, object]] = []
    generated_ids: list[int] = []
    next_input = torch.tensor([prompt_ids], dtype=torch.int64)
    start_pos = 0
    for selection_index in range(max_new_tokens):
        normalized_rows: list[torch.Tensor] = []

        def capture_normalized_row(
            _module: torch.nn.Module,
            inputs: tuple[torch.Tensor, ...],
            rows: list[torch.Tensor] = normalized_rows,
        ) -> None:
            if len(inputs) != 1:
                raise RuntimeError("source output head received an unexpected input")
            rows.append(inputs[0][:, -1].detach().cpu().contiguous())

        hook = model.head.register_forward_pre_hook(capture_normalized_row)
        try:
            _source_output_ids, logits, _main_hidden = model(
                next_input, start_pos=start_pos
            )
        finally:
            hook.remove()
        if len(normalized_rows) != 1:
            raise RuntimeError(
                "source generation did not expose one normalized head row"
            )
        normalized_row = normalized_rows[0]
        if normalized_row.dtype != torch.bfloat16 or list(normalized_row.shape) != [
            1,
            128,
        ]:
            raise RuntimeError(
                "source generation normalized head row has an invalid BF16 layout"
            )
        selected_id = _lowest_id_argmax(logits)
        selections.append(
            {
                "selection_index": selection_index,
                "start_pos": start_pos,
                "input_ids": [int(token_id) for token_id in next_input[0].tolist()],
                "normalized_bf16": tensor_record(normalized_row, include_storage=True),
                "logits": tensor_record(logits, include_storage=True),
                "selected_id": selected_id,
            }
        )
        generated_ids.append(selected_id)
        if eos_token_id is not None and selected_id == eos_token_id:
            return {
                "name": name,
                "prompt_ids": list(prompt_ids),
                "generated_ids": generated_ids,
                "selections": selections,
                "stop_reason": "eos",
            }
        if selection_index + 1 == max_new_tokens:
            break
        # Feed only selections which must produce a later output.  The final
        # selected ID remains output-only, which is also why admission reserves
        # one fewer cache position than generated IDs.
        next_input = torch.tensor([[selected_id]], dtype=torch.int64)
        start_pos = len(prompt_ids) + len(generated_ids) - 1
    return {
        "name": name,
        "prompt_ids": list(prompt_ids),
        "generated_ids": generated_ids,
        "selections": selections,
        "stop_reason": "max_new_tokens",
    }


def run_generation_oracle(
    *, max_new_tokens: int = 2, eos_token_id: int | None = None
) -> dict[str, object]:
    """Execute the bounded V4.1 source greedy loop for two fixed prompts.

    This is independent source execution, initialized from deterministic
    synthetic parameters.  It deliberately does not consume ``run_capture``
    output or any Rust values as expected results.
    """
    manifest = forward_manifest.manifest()
    validation = forward_manifest.validate_manifest(
        manifest, require_execution_ready=True
    )
    kernel_bytes = (SCRIPTS / "v41_cpu_kernels.py").read_bytes()
    if (
        _sha256_bytes((ROOT / "artifacts" / "v41-kernel-pinned.py").read_bytes())
        != KERNEL_SHA256
    ):
        raise RuntimeError(
            "retained upstream kernel hash differs from the capture contract"
        )
    graph = source_loader.load_text_graph(kernels)
    model_args = forward_manifest.model_args(manifest)
    encoded_parameters_sha256: str | None = None
    parameter_count: int | None = None
    initialized_buffers: list[str] | None = None
    source_head: dict[str, object] | None = None
    cases: list[dict[str, object]] = []
    with graph.set_dtype(torch.bfloat16):
        for _name, prompt_ids in GENERATION_PROMPTS:
            _generation_admission(
                prompt_ids,
                max_new_tokens,
                vocab_size=model_args["vocab_size"],
                max_seq_len=model_args["max_seq_len"],
                eos_token_id=eos_token_id,
            )
        # Each prompt owns a fresh source model and shared-attention runtime.
        # Resetting tensor caches alone would leave graph-global candidate state
        # from the first prompt observable to the second one.
        for name, prompt_ids in GENERATION_PROMPTS:
            graph.shared_attn = graph.SharedAttentionRuntime()
            args, tokenizer = executable_args(graph, manifest)
            model = graph.Transformer(args, tokenizer).eval()
            encoded_parameters = initialize_parameters(model)
            current_parameter_sha256 = _canonical_json_sha256(encoded_parameters)
            current_buffers = initialize_runtime_buffers(model)
            head_weight = encoded_parameters.get("head.weight")
            if not isinstance(head_weight, dict):
                raise TypeError("source generation lacks FP32 output-head weights")
            current_source_head = {
                "weight_shape": head_weight.get("shape"),
                "weight_fp32_bits": _storage_bits(
                    head_weight, width=4, dtype="torch.float32"
                ),
                "weight_storage_sha256": head_weight.get("storage_sha256"),
            }
            if encoded_parameters_sha256 is None:
                encoded_parameters_sha256 = current_parameter_sha256
                parameter_count = len(encoded_parameters)
                initialized_buffers = current_buffers
                source_head = current_source_head
            elif (
                current_parameter_sha256 != encoded_parameters_sha256
                or len(encoded_parameters) != parameter_count
                or current_buffers != initialized_buffers
                or current_source_head != source_head
            ):
                raise RuntimeError(
                    "source generation cases did not initialize identically"
                )
            cases.append(
                _generation_case(
                    model,
                    name=name,
                    prompt_ids=prompt_ids,
                    max_new_tokens=max_new_tokens,
                    eos_token_id=eos_token_id,
                )
            )
    if (
        encoded_parameters_sha256 is None
        or parameter_count is None
        or initialized_buffers is None
        or source_head is None
    ):
        raise RuntimeError("generation oracle did not initialize a source case")
    return {
        "schema_version": 1,
        "status": "completed synthetic source greedy generation; no Rust parity claim",
        "scope": (
            "two bounded cache-respecting greedy source decodes; no checkpoint, "
            "tokenizer-template, GPU, throughput, or serving claim"
        ),
        "selection": {
            "policy": "finite FP32 logits; maximum value, ties choose lowest token ID",
            "source_sample_is_not_used": True,
            "eos_token_id": eos_token_id,
            "max_new_tokens": max_new_tokens,
        },
        "context_bound": {
            "max_seq_len": args.max_seq_len,
            "admission": "prompt_len + max_new_tokens - 1 <= max_seq_len",
            "final_selected_id_is_not_fed_back": True,
        },
        "source": {
            "revision": SOURCE_REVISION,
            "model_sha256": source_loader.MODEL_SHA256,
            "engram_sha256": source_loader.ENGRAM_SHA256,
            "loader_sha256": _sha256_bytes(
                (SCRIPTS / "v41_source_loader.py").read_bytes()
            ),
            "runner_sha256": _sha256_bytes(Path(__file__).read_bytes()),
            "kernel_source_sha256": KERNEL_SHA256,
            "cpu_backend_sha256": _sha256_bytes(kernel_bytes),
        },
        "parameter_identity": {
            "initializer": "deterministic sine plus exact encoded FP8/packed-FP4 storage",
            "parameter_count": parameter_count,
            "encoded_parameters_canonical_sha256": encoded_parameters_sha256,
            "model_args_canonical_sha256": _canonical_json_sha256(model_args),
            "initialized_cache_buffers": initialized_buffers,
        },
        "source_head": source_head,
        "kernel_substitutions": {
            "module": "scripts/v41_cpu_kernels.py",
            "functions": list(source_loader.KERNEL_NAMES),
            "preserved_boundaries": "BF16, E8M0 FP8, E4M3 and packed E2M1x2 storage",
            "not_a_claim": [
                "upstream GPU-kernel parity",
                "released-checkpoint execution",
            ],
        },
        "manifest_canonical_sha256": _canonical_json_sha256(manifest),
        "manifest_validation": validation,
        "model": {"vocab_size": args.vocab_size, "max_seq_len": args.max_seq_len},
        "cases": cases,
    }


def _generation_fixture_logit(record: object, *, vocab_size: int) -> dict[str, object]:
    """Validate and retain the exact small FP32 vector needed for replay."""
    if not isinstance(record, dict):
        raise TypeError("generation fixture logits must be an object")
    if (
        record.get("dtype") != "torch.float32"
        or record.get("shape") != [1, vocab_size]
        or record.get("numel") != vocab_size
        or record.get("finite") is not True
    ):
        raise RuntimeError("generation fixture logits have an invalid FP32 layout")
    storage_hex = record.get("storage_hex")
    storage_sha256 = record.get("storage_sha256")
    if not isinstance(storage_hex, str) or not isinstance(storage_sha256, str):
        raise TypeError("generation fixture logits require exact storage identity")
    raw = bytes.fromhex(storage_hex)
    if len(raw) != vocab_size * 4 or _sha256_bytes(raw) != storage_sha256:
        raise RuntimeError("generation fixture logits have inconsistent storage")
    values = struct.unpack(f"<{vocab_size}f", raw)
    if not all(math.isfinite(value) for value in values):
        raise RuntimeError("generation fixture logits encode a nonfinite value")
    return {
        "dtype": "torch.float32",
        "shape": [1, vocab_size],
        "numel": vocab_size,
        "finite": True,
        "storage_hex": storage_hex,
        "storage_sha256": storage_sha256,
    }


def _generation_fixture_bf16_row(record: object) -> dict[str, object]:
    if not isinstance(record, dict):
        raise TypeError("generation fixture normalized row must be an object")
    if (
        record.get("dtype") != "torch.bfloat16"
        or record.get("shape") != [1, 128]
        or record.get("numel") != 128
        or record.get("finite") is not True
    ):
        raise RuntimeError(
            "generation fixture normalized row has an invalid BF16 layout"
        )
    storage_hex = record.get("storage_hex")
    storage_sha256 = record.get("storage_sha256")
    if not isinstance(storage_hex, str) or not isinstance(storage_sha256, str):
        raise TypeError("generation fixture normalized row requires exact storage")
    raw = bytes.fromhex(storage_hex)
    if len(raw) != 256 or _sha256_bytes(raw) != storage_sha256:
        raise RuntimeError("generation fixture normalized row has inconsistent storage")
    return {
        "dtype": "torch.bfloat16",
        "shape": [1, 128],
        "numel": 128,
        "finite": True,
        "storage_hex": storage_hex,
        "storage_sha256": storage_sha256,
    }


def generation_fixture(receipt: object) -> dict[str, object]:
    """Project a private source receipt into the durable replay oracle.

    The projection keeps only the four final-position vectors and the provenance
    that makes those values interpretable.  It intentionally drops runtime and
    manifest diagnostics, initialized-cache names, and all uncaptured source
    execution detail.
    """
    if not isinstance(receipt, dict) or receipt.get("schema_version") != 1:
        raise TypeError("generation fixture requires a schema-version-one receipt")
    model = receipt.get("model")
    source = receipt.get("source")
    parameters = receipt.get("parameter_identity")
    source_head = receipt.get("source_head")
    selection = receipt.get("selection")
    context_bound = receipt.get("context_bound")
    substitutions = receipt.get("kernel_substitutions")
    cases = receipt.get("cases")
    if not all(
        isinstance(value, dict)
        for value in (
            model,
            source,
            parameters,
            source_head,
            selection,
            context_bound,
            substitutions,
        )
    ) or not isinstance(cases, list):
        raise TypeError("generation receipt lacks a required compact-fixture field")
    vocab_size = model.get("vocab_size")
    max_seq_len = model.get("max_seq_len")
    if not isinstance(vocab_size, int) or vocab_size <= 0:
        raise RuntimeError("generation fixture requires a positive vocabulary size")
    if not isinstance(max_seq_len, int) or max_seq_len <= 0:
        raise RuntimeError("generation fixture requires a positive context limit")
    if (
        source_head.get("weight_shape") != [vocab_size, 128]
        or not isinstance(source_head.get("weight_fp32_bits"), list)
        or len(source_head["weight_fp32_bits"]) != vocab_size * 128
        or not isinstance(source_head.get("weight_storage_sha256"), str)
    ):
        raise RuntimeError("generation fixture lacks exact source output-head weights")
    if (
        selection.get("policy")
        != "finite FP32 logits; maximum value, ties choose lowest token ID"
    ):
        raise RuntimeError(
            "generation fixture requires the explicit lowest-ID argmax policy"
        )
    if selection.get("source_sample_is_not_used") is not True:
        raise RuntimeError("generation fixture must not use source sampling")
    max_new_tokens = selection.get("max_new_tokens")
    if not isinstance(max_new_tokens, int) or max_new_tokens <= 0:
        raise RuntimeError("generation fixture has an invalid token budget")
    if context_bound.get("final_selected_id_is_not_fed_back") is not True:
        raise RuntimeError("generation fixture must retain terminal feedback semantics")
    expected_prompts = {name: list(prompt) for name, prompt in GENERATION_PROMPTS}
    if len(cases) != len(expected_prompts):
        raise RuntimeError("generation fixture must retain both bounded prompts")
    compact_cases: list[dict[str, object]] = []
    for case in cases:
        if not isinstance(case, dict):
            raise TypeError("generation receipt contains a non-object case")
        name = case.get("name")
        prompt_ids = case.get("prompt_ids")
        generated_ids = case.get("generated_ids")
        selections = case.get("selections")
        stop_reason = case.get("stop_reason")
        if not isinstance(name, str) or prompt_ids != expected_prompts.pop(name, None):
            raise RuntimeError(
                "generation fixture case prompt is not the fixed source prompt"
            )
        if not isinstance(generated_ids, list) or not isinstance(selections, list):
            raise TypeError("generation fixture case has invalid selections")
        if (
            len(generated_ids) != len(selections)
            or not 1 <= len(selections) <= max_new_tokens
        ):
            raise RuntimeError("generation fixture case has an invalid selection count")
        if stop_reason not in ("eos", "max_new_tokens"):
            raise RuntimeError("generation fixture case has an invalid stop reason")
        expected_start = 0
        expected_input = prompt_ids
        compact_selections: list[dict[str, object]] = []
        for index, (selected, step) in enumerate(
            zip(generated_ids, selections, strict=True)
        ):
            if not isinstance(selected, int) or not 0 <= selected < vocab_size:
                raise RuntimeError(
                    "generation fixture selected ID lies outside the vocabulary"
                )
            if not isinstance(step, dict):
                raise TypeError("generation fixture step must be an object")
            if (
                step.get("selection_index") != index
                or step.get("start_pos") != expected_start
            ):
                raise RuntimeError(
                    "generation fixture has a noncontiguous decode schedule"
                )
            if (
                step.get("input_ids") != expected_input
                or step.get("selected_id") != selected
            ):
                raise RuntimeError(
                    "generation fixture feedback does not match its selection"
                )
            normalized = _generation_fixture_bf16_row(step.get("normalized_bf16"))
            logits = _generation_fixture_logit(
                step.get("logits"), vocab_size=vocab_size
            )
            values = struct.unpack(
                f"<{vocab_size}f", bytes.fromhex(logits["storage_hex"])
            )
            maximum = max(values)
            if selected != min(
                index for index, value in enumerate(values) if value == maximum
            ):
                raise RuntimeError(
                    "generation fixture selected ID is not the lowest-ID argmax"
                )
            compact_selections.append(
                {
                    "start_pos": expected_start,
                    "input_ids": expected_input,
                    "normalized_bf16": normalized,
                    "logits": logits,
                    "selected_id": selected,
                }
            )
            expected_start += len(expected_input)
            expected_input = [selected]
        if stop_reason == "max_new_tokens" and len(selections) != max_new_tokens:
            raise RuntimeError("generation fixture token-budget stop is premature")
        compact_cases.append(
            {
                "name": name,
                "prompt_ids": prompt_ids,
                "generated_ids": generated_ids,
                "selections": compact_selections,
                "stop_reason": stop_reason,
            }
        )
    if expected_prompts:
        raise RuntimeError("generation fixture omitted a fixed source prompt")
    required_source = (
        "revision",
        "model_sha256",
        "engram_sha256",
        "loader_sha256",
        "runner_sha256",
        "kernel_source_sha256",
        "cpu_backend_sha256",
    )
    if any(not isinstance(source.get(key), str) for key in required_source):
        raise TypeError("generation fixture source identity is incomplete")
    required_parameters = (
        "initializer",
        "parameter_count",
        "encoded_parameters_canonical_sha256",
        "model_args_canonical_sha256",
    )
    if any(key not in parameters for key in required_parameters):
        raise TypeError("generation fixture parameter identity is incomplete")
    return {
        "schema_version": 1,
        "scope": (
            "bounded synthetic V4.1 source greedy replay oracle; not checkpoint, GPU, "
            "throughput, tokenizer-template, or serving evidence"
        ),
        "selection": {
            "policy": selection["policy"],
            "eos_token_id": selection.get("eos_token_id"),
            "max_new_tokens": max_new_tokens,
            "source_sample_is_not_used": True,
        },
        "context_bound": {
            "max_seq_len": max_seq_len,
            "admission": context_bound.get("admission"),
            "final_selected_id_is_not_fed_back": True,
        },
        "source": {key: source[key] for key in required_source},
        "parameter_identity": {key: parameters[key] for key in required_parameters},
        "source_head": source_head,
        "comparison_policy": {
            "kind": "two_fp32_dot_error_bounds",
            "unit_roundoff_exponent": -24,
            "operation_count_per_dot": 256,
            "bound": "abs_error <= 2 * gamma(operation_count_per_dot) * sum_i(abs(x_i * w_i))",
            "reason": (
                "exact source normalized BF16 rows and FP32 head weights are retained; "
                "the envelope is fixed before any Rust candidate execution"
            ),
        },
        "kernel_substitutions": substitutions,
        "model": {"vocab_size": vocab_size, "max_seq_len": max_seq_len},
        "cases": compact_cases,
    }


def _storage_bits(record: dict[str, Any], *, width: int, dtype: str) -> list[int]:
    """Decode receipt storage to exact scalar bit patterns without float math."""
    if record.get("dtype") != dtype:
        raise RuntimeError(
            f"head fixture expected {dtype} storage, got {record.get('dtype')!r}"
        )
    storage_hex = record.get("storage_hex")
    if not isinstance(storage_hex, str):
        raise TypeError("head fixture requires complete tensor storage")
    raw = bytes.fromhex(storage_hex)
    numel = record.get("numel")
    if not isinstance(numel, int) or len(raw) != numel * width:
        raise RuntimeError("head fixture tensor storage length does not match numel")
    # tensor_record writes native CPU storage.  The resulting integers are
    # endianness-independent bit patterns, and the fixture pins this capture's
    # little-endian encoding so a Rust reader never interprets raw bytes.
    if sys.byteorder != "little":
        raise RuntimeError("head fixture export requires little-endian CPU storage")
    return [
        int.from_bytes(raw[offset : offset + width], byteorder="little")
        for offset in range(0, len(raw), width)
    ]


def head_fixture(receipt: dict[str, object]) -> dict[str, object]:
    """Select source-produced final-norm inputs and head logits for Rust tests.

    This function only decodes exact receipt storage.  It deliberately performs
    no head computation, rounding, or expected-value reconstruction.
    """
    if (
        receipt.get("capture_status")
        != "completed synthetic source-forward capture; no parity claim"
    ):
        raise RuntimeError("head fixture export requires a completed source capture")
    coverage = receipt.get("coverage_status")
    if not isinstance(coverage, dict) or coverage.get("pending") != []:
        raise RuntimeError("head fixture export requires a complete source capture")
    source = receipt.get("source")
    encoded = receipt.get("encoded_parameters")
    steps = receipt.get("steps")
    manifest_sha = receipt.get("manifest_canonical_sha256")
    if (
        not isinstance(source, dict)
        or not isinstance(encoded, dict)
        or not isinstance(steps, list)
        or not isinstance(manifest_sha, str)
    ):
        raise TypeError("complete capture has an invalid head-fixture shape")
    weight = encoded.get("head.weight")
    norm_weight = encoded.get("norm.weight")
    model_args = receipt.get("model_args")
    if not isinstance(weight, dict):
        raise TypeError("complete capture did not record head.weight")
    if not isinstance(norm_weight, dict):
        raise TypeError("complete capture did not record norm.weight")
    if not isinstance(model_args, dict):
        raise TypeError("complete capture did not record model args")
    weight_shape = weight.get("shape")
    if (
        not isinstance(weight_shape, list)
        or len(weight_shape) != 2
        or any(not isinstance(width, int) or width <= 0 for width in weight_shape)
    ):
        raise RuntimeError("head.weight must be a nonempty rank-two tensor")
    norm_eps = model_args.get("norm_eps")
    if not isinstance(norm_eps, (int, float)) or not float(norm_eps) > 0:
        raise RuntimeError("head fixture requires a positive source norm_eps")
    norm_epsilon_bits = struct.unpack("<I", struct.pack("<f", float(norm_eps)))[0]

    cases: list[dict[str, object]] = []
    for step in steps:
        if not isinstance(step, dict):
            raise TypeError("complete capture includes an invalid step")
        start_pos = step.get("start_pos")
        intermediates = step.get("intermediates")
        logits = step.get("logits")
        if not isinstance(start_pos, int) or not isinstance(intermediates, dict):
            raise TypeError("complete capture step lacks source boundaries")
        norm = intermediates.get("norm")
        norm_input = intermediates.get("norm_input")
        final_block = intermediates.get("layers.4")
        if not isinstance(norm, dict) or not isinstance(logits, dict):
            raise TypeError("complete capture step lacks final norm or logits")
        if not isinstance(norm_input, dict) or not isinstance(final_block, list):
            raise TypeError("complete capture step lacks final HC boundaries")
        if len(final_block) != 2 or not all(
            isinstance(value, dict) for value in final_block
        ):
            raise TypeError("final source block must return exactly two tensors")
        final_block_state, final_pre_mix = final_block
        input_shape = norm.get("shape")
        collapsed_shape = norm_input.get("shape")
        final_block_shape = final_block_state.get("shape")
        final_pre_shape = final_pre_mix.get("shape")
        logits_shape = logits.get("shape")
        if (
            not isinstance(input_shape, list)
            or len(input_shape) != 3
            or not isinstance(logits_shape, list)
            or len(logits_shape) != 2
        ):
            raise RuntimeError("head fixture source tensors have unexpected rank")
        if (
            collapsed_shape != input_shape
            or not isinstance(final_block_shape, list)
            or len(final_block_shape) != 4
            or not isinstance(final_pre_shape, list)
            or len(final_pre_shape) != 3
            or final_block_shape[:2] != input_shape[:2]
            or final_pre_shape[:2] != input_shape[:2]
            or final_pre_shape[-1] != final_block_shape[-2]
            or final_block_shape[-1] != input_shape[-1]
        ):
            raise RuntimeError("head fixture HC boundaries have unexpected shapes")
        if input_shape[-1] != weight_shape[1] or logits_shape[-1] != weight_shape[0]:
            raise RuntimeError("head fixture source tensors disagree with head.weight")
        cases.append(
            {
                "start_pos": start_pos,
                "input_shape": input_shape,
                "input_bf16": _storage_bits(norm, width=2, dtype="torch.bfloat16"),
                "final_block_shape": final_block_shape,
                "final_block_bf16": _storage_bits(
                    final_block_state, width=2, dtype="torch.bfloat16"
                ),
                "final_pre_shape": final_pre_shape,
                "final_pre_fp32_bits": _storage_bits(
                    final_pre_mix, width=4, dtype="torch.float32"
                ),
                "collapsed_bf16": _storage_bits(
                    norm_input, width=2, dtype="torch.bfloat16"
                ),
                "logits_shape": logits_shape,
                "logits_fp32_bits": _storage_bits(
                    logits, width=4, dtype="torch.float32"
                ),
            }
        )
    if [case["start_pos"] for case in cases] != [0, 5, 6]:
        raise RuntimeError("head fixture requires the pinned prefill/decode trace")
    if cases[0]["input_shape"][1] != 5:
        raise RuntimeError("head fixture must retain all five prefill norm rows")

    complete_bytes = serialized_capture(receipt)
    return {
        "schema_version": 1,
        "source": {
            "revision": source.get("revision"),
            "model_sha256": source.get("model_sha256"),
            "cpu_backend_sha256": source.get("cpu_backend_sha256"),
            "complete_capture_sha256": _sha256_bytes(complete_bytes),
            "manifest_canonical_sha256": manifest_sha,
        },
        "weight_shape": weight_shape,
        "weight_fp32_bits": _storage_bits(weight, width=4, dtype="torch.float32"),
        "norm_weight_bf16": _storage_bits(norm_weight, width=2, dtype="torch.bfloat16"),
        "norm_epsilon_bits": norm_epsilon_bits,
        "cases": cases,
        "comparison_policy": {
            "kind": "two_fp32_dot_error_bounds",
            "unit_roundoff_exponent": -24,
            "operation_count_per_dot": 2 * weight_shape[1],
            "bound": "abs_error <= 2 * gamma(operation_count_per_dot) * sum_i(abs(x_i * w_i))",
            "gamma": "gamma(n) = n * u / (1 - n * u)",
            "accumulator": "f64 bound evaluation; compared values are FP32",
            "assumptions": [
                "input, weight, and product values are finite normal values",
                "the FP32 dot product has no underflow or overflow",
                "input_bf16, weight_fp32_bits, and logits_fp32_bits are exact integer encodings from the source receipt",
            ],
        },
    }


def moe_fixture(receipt: dict[str, object], *, layer: int = 4) -> dict[str, object]:
    """Select actual layer-four MoE inputs, choices, outputs and encodings.

    This is a compact projection of a completed source capture.  It does not
    evaluate the gate or experts: expected values remain the hook records from
    the source's own selected-layer FFN execution.
    """
    if (
        receipt.get("capture_status")
        != "completed synthetic source-forward capture; no parity claim"
    ):
        raise RuntimeError("MoE fixture export requires a completed source capture")
    coverage = receipt.get("coverage_status")
    source = receipt.get("source")
    runtime = receipt.get("runtime")
    encoded = receipt.get("encoded_parameters")
    model_args = receipt.get("model_args")
    steps = receipt.get("steps")
    manifest_sha = receipt.get("manifest_canonical_sha256")
    if (
        not isinstance(coverage, dict)
        or coverage.get("pending") != []
        or not isinstance(source, dict)
        or not isinstance(runtime, dict)
        or not isinstance(encoded, dict)
        or not isinstance(model_args, dict)
        or not isinstance(steps, list)
        or not isinstance(manifest_sha, str)
    ):
        raise TypeError("complete capture has an invalid MoE-fixture shape")

    if layer not in (3, 4):
        raise ValueError("MoE fixture supports only the captured layer-three/four seam")
    layer_prefix = f"layers.{layer}"
    ffn_prefix = f"{layer_prefix}.ffn"
    parameters = {
        name: record
        for name, record in encoded.items()
        if isinstance(name, str) and name.startswith(f"{ffn_prefix}.")
    }
    routed = tuple(f"{ffn_prefix}.experts.{index}" for index in range(4))
    expert_prefixes = (*routed, f"{ffn_prefix}.shared_experts")
    required_names = {
        f"{ffn_prefix}.gate.weight",
        f"{ffn_prefix}.gate.bias",
        *(
            f"{prefix}.{projection}.{field}"
            for prefix in expert_prefixes
            for projection in ("w1", "w2", "w3")
            for field in ("weight", "scale")
        ),
    }
    if set(parameters) != required_names:
        missing = sorted(required_names - set(parameters))
        unexpected = sorted(set(parameters) - required_names)
        raise RuntimeError(
            f"complete capture layer-{layer} MoE parameter set differs: "
            f"missing={missing}, unexpected={unexpected}"
        )
    if not all(
        isinstance(record, dict) and "storage_hex" in record
        for record in parameters.values()
    ):
        raise TypeError("MoE fixture requires complete encoded parameter storage")

    expected_model = {
        "dim": 128,
        "moe_inter_dim": 128,
        "n_routed_experts": 4,
        "n_activated_experts": 2,
        "n_shared_experts": 1,
    }
    if any(model_args.get(name) != value for name, value in expected_model.items()):
        raise RuntimeError("MoE fixture requires the fixed reduced MoE dimensions")
    if runtime.get("storage_byteorder") != "little":
        raise RuntimeError("MoE fixture export requires little-endian tensor storage")

    expected_tensor_layouts = {
        f"{ffn_prefix}.gate.weight": ([4, 128], "torch.bfloat16"),
        f"{ffn_prefix}.gate.bias": ([4], "torch.float32"),
    }
    for prefix in routed:
        for projection in ("w1", "w2", "w3"):
            expected_tensor_layouts[f"{prefix}.{projection}.weight"] = (
                [128, 64],
                "torch.float4_e2m1fn_x2",
            )
            expected_tensor_layouts[f"{prefix}.{projection}.scale"] = (
                [128, 4],
                "torch.float8_e8m0fnu",
            )
    for projection in ("w1", "w2", "w3"):
        expected_tensor_layouts[f"{ffn_prefix}.shared_experts.{projection}.weight"] = (
            [128, 128],
            "torch.float8_e4m3fn",
        )
        expected_tensor_layouts[f"{ffn_prefix}.shared_experts.{projection}.scale"] = (
            [4, 4],
            "torch.float8_e8m0fnu",
        )
    for name, (shape, dtype) in expected_tensor_layouts.items():
        record = parameters[name]
        if record.get("shape") != shape or record.get("dtype") != dtype:
            raise RuntimeError(f"MoE parameter {name} has an unexpected shape or dtype")

    block_parameter_names = (
        f"{layer_prefix}.hc_attn_fn",
        f"{layer_prefix}.hc_attn_base",
        f"{layer_prefix}.hc_attn_scale",
        f"{layer_prefix}.hc_ffn_fn",
        f"{layer_prefix}.hc_ffn_base",
        f"{layer_prefix}.hc_ffn_scale",
        f"{layer_prefix}.attn_norm.weight",
        f"{layer_prefix}.ffn_norm.weight",
    )
    block_parameters = {name: encoded.get(name) for name in block_parameter_names}
    if set(block_parameters) != set(block_parameter_names) or not all(
        isinstance(record, dict) and "storage_hex" in record
        for record in block_parameters.values()
    ):
        raise RuntimeError("complete capture lacks exact layer-four block parameters")
    expected_block_layouts = {
        f"{layer_prefix}.hc_attn_fn": ([8, 256], "torch.float32"),
        f"{layer_prefix}.hc_attn_base": ([8], "torch.float32"),
        f"{layer_prefix}.hc_attn_scale": ([3], "torch.float32"),
        f"{layer_prefix}.hc_ffn_fn": ([8, 256], "torch.float32"),
        f"{layer_prefix}.hc_ffn_base": ([8], "torch.float32"),
        f"{layer_prefix}.hc_ffn_scale": ([3], "torch.float32"),
        f"{layer_prefix}.attn_norm.weight": ([128], "torch.bfloat16"),
        f"{layer_prefix}.ffn_norm.weight": ([128], "torch.bfloat16"),
    }
    for name, (shape, dtype) in expected_block_layouts.items():
        record = block_parameters[name]
        if record.get("shape") != shape or record.get("dtype") != dtype:
            raise RuntimeError(
                f"block parameter {name} has an unexpected shape or dtype"
            )

    cases: list[dict[str, object]] = []
    for step in steps:
        if not isinstance(step, dict):
            raise TypeError("complete capture includes an invalid step")
        start_pos = step.get("start_pos")
        intermediates = step.get("intermediates")
        if not isinstance(start_pos, int) or not isinstance(intermediates, dict):
            raise TypeError("complete capture step lacks MoE boundaries")
        ffn_input = intermediates.get(f"{ffn_prefix}_input")
        ffn_output = intermediates.get(ffn_prefix)
        gate = intermediates.get(f"{ffn_prefix}.gate")
        block_input = intermediates.get(f"{layer_prefix}.block_input")
        attention_input = intermediates.get(f"{layer_prefix}.attention_input")
        attention_output = intermediates.get(f"{layer_prefix}.attn")
        after_attention_residual = intermediates.get(
            f"{layer_prefix}.after_attention_residual"
        )
        ffn_collapsed = intermediates.get(f"{layer_prefix}.ffn_collapsed")
        block_result = intermediates.get(layer_prefix)
        hc_kernel_calls = step.get("hyper_connection_mixes")
        if (
            not isinstance(ffn_input, dict)
            or not isinstance(ffn_output, dict)
            or not isinstance(gate, list)
            or len(gate) != 2
            or not all(isinstance(value, dict) for value in gate)
            or not isinstance(block_input, dict)
            or not isinstance(attention_input, dict)
            or not isinstance(attention_output, dict)
            or not isinstance(after_attention_residual, dict)
            or not isinstance(ffn_collapsed, dict)
            or not isinstance(block_result, list)
            or len(block_result) != 2
            or not all(isinstance(value, dict) for value in block_result)
            or not isinstance(hc_kernel_calls, list)
        ):
            raise TypeError(
                f"complete capture step lacks layer-{layer} MoE hook records"
            )
        block_residual = block_input.get("residual")
        incoming_pre = block_input.get("incoming_pre")
        block_output, block_next_pre = block_result
        next_block_input = (
            intermediates.get("layers.4.block_input") if layer == 3 else None
        )
        if (
            ffn_input.get("shape") != ffn_output.get("shape")
            or ffn_input.get("dtype") != "torch.bfloat16"
            or ffn_output.get("dtype") != "torch.bfloat16"
            or gate[0].get("dtype") != "torch.float32"
            or gate[1].get("dtype") != "torch.int64"
            or not isinstance(block_residual, dict)
            or not isinstance(incoming_pre, dict)
            or block_residual.get("dtype") != "torch.bfloat16"
            or incoming_pre.get("dtype") != "torch.float32"
            or attention_input.get("dtype") != "torch.bfloat16"
            or attention_output.get("dtype") != "torch.bfloat16"
            or after_attention_residual.get("dtype") != "torch.bfloat16"
            or ffn_collapsed.get("dtype") != "torch.bfloat16"
            or block_output.get("dtype") != "torch.bfloat16"
            or block_next_pre.get("dtype") != "torch.float32"
        ):
            raise RuntimeError(
                f"layer-{layer} MoE hook storage has unexpected dtype or shape"
            )
        if layer == 3:
            if not isinstance(next_block_input, dict):
                raise TypeError(
                    "layer-three fixture lacks layer-four block-entry cross-gate"
                )
            next_residual = next_block_input.get("residual")
            next_pre = next_block_input.get("incoming_pre")
            if not isinstance(next_residual, dict) or not isinstance(next_pre, dict):
                raise TypeError(
                    "layer-three fixture has malformed layer-four block entry"
                )
            if block_output.get("storage_sha256") != next_residual.get(
                "storage_sha256"
            ) or block_next_pre.get("storage_sha256") != next_pre.get("storage_sha256"):
                raise RuntimeError(
                    "layer-three terminal state does not match layer-four block entry"
                )
        layer_hc_calls = [
            call
            for call in hc_kernel_calls
            if isinstance(call, dict) and call.get("layer_id") == layer
        ]
        if len(layer_hc_calls) != 2:
            raise RuntimeError(
                f"complete capture must retain two layer-{layer} HC calls"
            )
        coefficients = {
            call.get("sublayer"): call.get("outputs") for call in layer_hc_calls
        }
        raw_hc_mixes = {
            call.get("sublayer"): call.get("inputs", {}).get("mixes")
            for call in layer_hc_calls
        }
        if set(coefficients) != {"attention", "ffn"} or not all(
            isinstance(value, dict)
            and set(value) == {"pre", "post", "comb"}
            and all(
                isinstance(record, dict) and "storage_hex" in record
                for record in value.values()
            )
            for value in coefficients.values()
        ):
            raise RuntimeError(f"complete capture lacks layer-{layer} HC coefficients")
        if set(raw_hc_mixes) != {"attention", "ffn"} or not all(
            isinstance(value, dict)
            and value.get("dtype") == "torch.float32"
            and "storage_hex" in value
            for value in raw_hc_mixes.values()
        ):
            raise RuntimeError(f"complete capture lacks layer-{layer} raw HC mixes")
        case: dict[str, object] = {
            "start_pos": start_pos,
            "input": ffn_input,
            "gate_weights": gate[0],
            "gate_indices": gate[1],
            "output": ffn_output,
            "block_input": block_residual,
            "block_incoming_pre": incoming_pre,
            "attention_input": attention_input,
            "attention_output": attention_output,
            "after_attention_residual": after_attention_residual,
            "ffn_collapsed": ffn_collapsed,
            "block_output": block_output,
            "block_next_pre": block_next_pre,
            "attention_coefficients": coefficients["attention"],
            "ffn_coefficients": coefficients["ffn"],
            "attention_hc_mixes": raw_hc_mixes["attention"],
            "ffn_hc_mixes": raw_hc_mixes["ffn"],
        }
        if layer == 3:
            case["next_block_entry"] = next_block_input
        cases.append(case)
    if [case["start_pos"] for case in cases] != [0, 5, 6]:
        raise RuntimeError("MoE fixture requires the pinned prefill/decode trace")

    moe_args = {
        name: model_args[name]
        for name in (
            "dim",
            "moe_inter_dim",
            "n_routed_experts",
            "n_activated_experts",
            "n_shared_experts",
            "score_func",
            "gate_temp",
            "norm_topk_prob",
            "route_scale",
            "swiglu_limit",
            "expert_dtype",
        )
    }
    return {
        "schema_version": 1,
        "scope": (
            f"layer-{layer} source MoE plus native block composition with captured "
            "source attention output; not native attention, Rust acceptance, or "
            "full-model parity"
        ),
        "source": {
            "revision": source.get("revision"),
            "model_sha256": source.get("model_sha256"),
            "engram_sha256": source.get("engram_sha256"),
            "kernel_source_sha256": source.get("kernel_source_sha256"),
            "cpu_backend_sha256": source.get("cpu_backend_sha256"),
            "loader_sha256": source.get("loader_sha256"),
            "runner_sha256": source.get("runner_sha256"),
            "complete_capture_sha256": _sha256_bytes(serialized_capture(receipt)),
            "manifest_canonical_sha256": manifest_sha,
            "storage_byteorder": runtime.get("storage_byteorder"),
        },
        "model": moe_args,
        "encoded_parameters": parameters,
        "block_parameters": block_parameters,
        "block_config": {
            "copies": model_args["hc_mult"],
            "hc_sinkhorn_iters": model_args["hc_sinkhorn_iters"],
            "hc_eps": model_args["hc_eps"],
            "norm_eps": model_args["norm_eps"],
        },
        "cases": cases,
        "comparison_policy": {
            "output_bf16": "exact storage bits",
            "attention_input_bf16": "exact storage bits",
            "after_attention_residual_bf16": "exact storage bits",
            "ffn_collapsed_bf16": "exact storage bits",
            "block_output_bf16": "exact storage bits",
            "block_next_pre_abs_error_max": 2**-20,
            "selected_expert_ids": (
                "exact IDs; compare route weights by expert ID because source score "
                "order and a native sorted-ID traversal may differ"
            ),
            "route_weight_abs_error_max": 2**-20,
            "fixed_before_candidate_execution": True,
            "next_block_entry": (
                "exact source storage identity with layer-four block input"
                if layer == 3
                else "not applicable"
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, help="write the bounded capture JSON here"
    )
    parser.add_argument(
        "--head-fixture-output",
        type=Path,
        help=(
            "write the compact final-norm/FP32-head fixture derived from a "
            "complete source capture"
        ),
    )
    parser.add_argument(
        "--moe-fixture-output",
        type=Path,
        help=(
            "write the compact layer-four MoE fixture derived from a complete "
            "source capture"
        ),
    )
    parser.add_argument(
        "--layer3-moe-fixture-output",
        type=Path,
        help=(
            "write the compact layer-three MoE/block-tail fixture derived from "
            "the same complete source capture"
        ),
    )
    parser.add_argument(
        "--attention-fixture-output",
        type=Path,
        help=(
            "write the compact layer-four source-attention fixture derived from "
            "a complete source capture"
        ),
    )
    parser.add_argument(
        "--layer3-to-layer1-fixture-output",
        type=Path,
        help=(
            "write the compact same-trace layer-three key publication to "
            "partial layer-one score-prefix fixture"
        ),
    )
    parser.add_argument(
        "--layer0-to-layer1-fixture-output",
        type=Path,
        help=(
            "write the compact same-trace layer-zero producer to layer-one "
            "Engram input fixture"
        ),
    )
    parser.add_argument(
        "--generation-oracle-output",
        type=Path,
        help=(
            "write the compact independent source greedy-generation receipt for "
            "the fixed five-token and four-token prompts"
        ),
    )
    parser.add_argument(
        "--generation-max-new-tokens",
        type=int,
        default=2,
        help="number of greedy output IDs per generation-oracle case (default: 2)",
    )
    parser.add_argument(
        "--generation-eos-token-id",
        type=int,
        help="optional synthetic EOS ID for generation-oracle stopping",
    )
    parser.add_argument(
        "--generation-fixture-input",
        type=Path,
        help="read a private generation-oracle receipt to project into a compact fixture",
    )
    parser.add_argument(
        "--generation-fixture-output",
        type=Path,
        help="write the compact durable generation-replay fixture",
    )
    args = parser.parse_args()
    if (args.generation_fixture_input is None) != (
        args.generation_fixture_output is None
    ):
        parser.error(
            "--generation-fixture-input and --generation-fixture-output must be supplied together"
        )
    capture_requested = any(
        output is not None
        for output in (
            args.output,
            args.head_fixture_output,
            args.moe_fixture_output,
            args.layer3_moe_fixture_output,
            args.attention_fixture_output,
            args.layer3_to_layer1_fixture_output,
            args.layer0_to_layer1_fixture_output,
        )
    )
    receipt: dict[str, object] | None = None
    artifact_bytes: bytes | None = None
    if capture_requested or args.generation_oracle_output is None:
        receipt = run_capture()
        artifact_bytes = serialized_capture(receipt)
        if len(artifact_bytes) > MAX_CAPTURE_BYTES:
            raise RuntimeError(f"capture exceeds {MAX_CAPTURE_BYTES} byte receipt cap")
    if args.output is not None:
        assert receipt is not None and artifact_bytes is not None
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(artifact_bytes)
        print(
            json.dumps(
                {
                    "artifact_sha256": _sha256_bytes(artifact_bytes),
                    "bytes": len(artifact_bytes),
                    "coverage_pending": receipt["coverage_status"]["pending"],
                    "path": str(args.output),
                    "status": "source_forward_capture",
                    "steps": len(receipt["steps"]),
                },
                sort_keys=True,
            )
        )
    if args.head_fixture_output is not None:
        assert receipt is not None
        fixture = head_fixture(receipt)
        # The public fixture contains exact per-scalar encodings.  Compact JSON
        # keeps that audit surface below its deliberately small size cap.
        fixture_bytes = (
            json.dumps(fixture, sort_keys=True, separators=(",", ":"), allow_nan=False)
            + "\n"
        ).encode("utf-8")
        if len(fixture_bytes) >= MAX_HEAD_FIXTURE_BYTES:
            raise RuntimeError(
                f"head fixture is {len(fixture_bytes)} bytes; it must stay below "
                f"{MAX_HEAD_FIXTURE_BYTES} bytes"
            )
        args.head_fixture_output.parent.mkdir(parents=True, exist_ok=True)
        args.head_fixture_output.write_bytes(fixture_bytes)
        print(
            json.dumps(
                {
                    "artifact_sha256": _sha256_bytes(fixture_bytes),
                    "bytes": len(fixture_bytes),
                    "complete_capture_sha256": fixture["source"][
                        "complete_capture_sha256"
                    ],
                    "path": str(args.head_fixture_output),
                    "status": "source_forward_head_fixture",
                },
                sort_keys=True,
            )
        )
    if args.moe_fixture_output is not None:
        assert receipt is not None
        fixture = moe_fixture(receipt)
        fixture_bytes = (
            json.dumps(fixture, sort_keys=True, separators=(",", ":"), allow_nan=False)
            + "\n"
        ).encode("utf-8")
        if len(fixture_bytes) >= MAX_MOE_FIXTURE_BYTES:
            raise RuntimeError(
                f"MoE fixture is {len(fixture_bytes)} bytes; it must stay below "
                f"{MAX_MOE_FIXTURE_BYTES} bytes"
            )
        args.moe_fixture_output.parent.mkdir(parents=True, exist_ok=True)
        args.moe_fixture_output.write_bytes(fixture_bytes)
        print(
            json.dumps(
                {
                    "artifact_sha256": _sha256_bytes(fixture_bytes),
                    "bytes": len(fixture_bytes),
                    "complete_capture_sha256": fixture["source"][
                        "complete_capture_sha256"
                    ],
                    "path": str(args.moe_fixture_output),
                    "status": "source_forward_moe_fixture",
                },
                sort_keys=True,
            )
        )
    if args.layer3_moe_fixture_output is not None:
        assert receipt is not None
        fixture = moe_fixture(receipt, layer=3)
        fixture_bytes = (
            json.dumps(fixture, sort_keys=True, separators=(",", ":"), allow_nan=False)
            + "\n"
        ).encode("utf-8")
        if len(fixture_bytes) >= MAX_MOE_FIXTURE_BYTES:
            raise RuntimeError(
                f"layer-three MoE fixture is {len(fixture_bytes)} bytes; it must stay below "
                f"{MAX_MOE_FIXTURE_BYTES} bytes"
            )
        args.layer3_moe_fixture_output.parent.mkdir(parents=True, exist_ok=True)
        args.layer3_moe_fixture_output.write_bytes(fixture_bytes)
        print(
            json.dumps(
                {
                    "artifact_sha256": _sha256_bytes(fixture_bytes),
                    "bytes": len(fixture_bytes),
                    "complete_capture_sha256": fixture["source"][
                        "complete_capture_sha256"
                    ],
                    "path": str(args.layer3_moe_fixture_output),
                    "status": "source_forward_layer_three_moe_fixture",
                },
                sort_keys=True,
            )
        )
    if args.attention_fixture_output is not None:
        assert receipt is not None
        fixture = v41_attention_capture.attention_fixture(
            receipt, helper_path=SCRIPTS / "v41_attention_capture.py"
        )
        fixture_bytes = (
            json.dumps(fixture, sort_keys=True, separators=(",", ":"), allow_nan=False)
            + "\n"
        ).encode("utf-8")
        if len(fixture_bytes) >= MAX_ATTENTION_FIXTURE_BYTES:
            raise RuntimeError(
                f"attention fixture is {len(fixture_bytes)} bytes; it must stay below "
                f"{MAX_ATTENTION_FIXTURE_BYTES} bytes"
            )
        args.attention_fixture_output.parent.mkdir(parents=True, exist_ok=True)
        args.attention_fixture_output.write_bytes(fixture_bytes)
        print(
            json.dumps(
                {
                    "artifact_sha256": _sha256_bytes(fixture_bytes),
                    "bytes": len(fixture_bytes),
                    "complete_capture_sha256": fixture["source"][
                        "complete_capture_sha256"
                    ],
                    "path": str(args.attention_fixture_output),
                    "status": "source_forward_attention_fixture",
                },
                sort_keys=True,
            )
        )
    if args.layer3_to_layer1_fixture_output is not None:
        assert receipt is not None
        fixture = v41_index_key_capture.layer3_to_layer1_fixture(receipt)
        fixture_bytes = (
            json.dumps(fixture, sort_keys=True, separators=(",", ":"), allow_nan=False)
            + "\n"
        ).encode("utf-8")
        if len(fixture_bytes) >= MAX_MOE_FIXTURE_BYTES:
            raise RuntimeError(
                f"layer-three bridge fixture is {len(fixture_bytes)} bytes; it must stay below "
                f"{MAX_MOE_FIXTURE_BYTES} bytes"
            )
        args.layer3_to_layer1_fixture_output.parent.mkdir(parents=True, exist_ok=True)
        args.layer3_to_layer1_fixture_output.write_bytes(fixture_bytes)
        print(
            json.dumps(
                {
                    "artifact_sha256": _sha256_bytes(fixture_bytes),
                    "bytes": len(fixture_bytes),
                    "complete_capture_sha256": fixture["source"][
                        "complete_capture_sha256"
                    ],
                    "path": str(args.layer3_to_layer1_fixture_output),
                    "status": "source_forward_layer_three_to_layer_one_fixture",
                },
                sort_keys=True,
            )
        )
    if args.layer0_to_layer1_fixture_output is not None:
        assert receipt is not None
        fixture = v41_layer0_to_layer1_capture.layer0_to_layer1_fixture(receipt)
        fixture_bytes = (
            json.dumps(fixture, sort_keys=True, separators=(",", ":"), allow_nan=False)
            + "\n"
        ).encode("utf-8")
        if len(fixture_bytes) >= MAX_LAYER_ZERO_TO_LAYER_ONE_FIXTURE_BYTES:
            raise RuntimeError(
                f"layer-zero bridge fixture is {len(fixture_bytes)} bytes; it must stay below "
                f"{MAX_LAYER_ZERO_TO_LAYER_ONE_FIXTURE_BYTES} bytes"
            )
        args.layer0_to_layer1_fixture_output.parent.mkdir(parents=True, exist_ok=True)
        args.layer0_to_layer1_fixture_output.write_bytes(fixture_bytes)
        print(
            json.dumps(
                {
                    "artifact_sha256": _sha256_bytes(fixture_bytes),
                    "bytes": len(fixture_bytes),
                    "complete_capture_sha256": fixture["source"][
                        "complete_capture_sha256"
                    ],
                    "path": str(args.layer0_to_layer1_fixture_output),
                    "status": "source_layer_zero_to_layer_one_fixture",
                },
                sort_keys=True,
            )
        )
    if args.generation_oracle_output is not None:
        generation = run_generation_oracle(
            max_new_tokens=args.generation_max_new_tokens,
            eos_token_id=args.generation_eos_token_id,
        )
        generation_bytes = (
            json.dumps(
                generation, sort_keys=True, separators=(",", ":"), allow_nan=False
            )
            + "\n"
        ).encode("utf-8")
        if len(generation_bytes) >= MAX_GENERATION_ORACLE_BYTES:
            raise RuntimeError(
                f"generation oracle is {len(generation_bytes)} bytes; it must stay below "
                f"{MAX_GENERATION_ORACLE_BYTES} bytes"
            )
        args.generation_oracle_output.parent.mkdir(parents=True, exist_ok=True)
        args.generation_oracle_output.write_bytes(generation_bytes)
        print(
            json.dumps(
                {
                    "artifact_sha256": _sha256_bytes(generation_bytes),
                    "bytes": len(generation_bytes),
                    "cases": len(generation["cases"]),
                    "path": str(args.generation_oracle_output),
                    "status": "source_greedy_generation_oracle",
                },
                sort_keys=True,
            )
        )
    if args.generation_fixture_input is not None:
        assert args.generation_fixture_output is not None
        generation_receipt = json.loads(args.generation_fixture_input.read_text())
        fixture = generation_fixture(generation_receipt)
        fixture_bytes = (
            json.dumps(fixture, sort_keys=True, separators=(",", ":"), allow_nan=False)
            + "\n"
        ).encode("utf-8")
        if len(fixture_bytes) >= MAX_GENERATION_ORACLE_BYTES:
            raise RuntimeError(
                f"generation fixture is {len(fixture_bytes)} bytes; it must stay below "
                f"{MAX_GENERATION_ORACLE_BYTES} bytes"
            )
        args.generation_fixture_output.parent.mkdir(parents=True, exist_ok=True)
        args.generation_fixture_output.write_bytes(fixture_bytes)
        print(
            json.dumps(
                {
                    "artifact_sha256": _sha256_bytes(fixture_bytes),
                    "bytes": len(fixture_bytes),
                    "path": str(args.generation_fixture_output),
                    "status": "source_greedy_generation_fixture",
                },
                sort_keys=True,
            )
        )
    if (
        args.output is None
        and args.head_fixture_output is None
        and args.moe_fixture_output is None
        and args.layer3_moe_fixture_output is None
        and args.attention_fixture_output is None
        and args.layer3_to_layer1_fixture_output is None
        and args.layer0_to_layer1_fixture_output is None
        and args.generation_oracle_output is None
        and args.generation_fixture_output is None
    ):
        assert artifact_bytes is not None
        print(artifact_bytes.decode("utf-8"), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
