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
import platform
import struct
import sys
from collections.abc import Iterator
from contextlib import contextmanager
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
MAX_HOOK_RECORDS = 96
SAMPLE_VALUES = 8
MAX_CAPTURE_BYTES = 16 << 20
# HC state and its source-produced collapse are both exact bit fixtures.  Keep
# the expanded oracle intentionally bounded, while allowing the three pinned
# prefill/decode cases to retain all required representations.
MAX_HEAD_FIXTURE_BYTES = 48 << 10
# The full layer-four MoE payload retains packed routed and FP8 shared expert
# storage.  It is intentionally bigger than the head fixture but bounded.
MAX_MOE_FIXTURE_BYTES = 512 << 10

sys.path.insert(0, str(SCRIPTS))
import v41_cpu_kernels as kernels
import v41_forward_manifest as forward_manifest
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
    else:
        flat = tensor.float().reshape(-1)
    sample = [float(item) for item in flat[:SAMPLE_VALUES]]
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


def tracing_kernel_bundle() -> tuple[ModuleType, list[dict[str, object]]]:
    """Expose the CPU backend while observing, never altering, HC kernel calls."""
    bundle = ModuleType("_metallix_v41_tracing_kernels")
    records: list[dict[str, object]] = []
    for name in source_loader.KERNEL_NAMES:
        setattr(bundle, name, getattr(kernels, name))

    def traced_hc_split_sinkhorn(
        mixes: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
        hc_mult: int = 4,
        sinkhorn_iters: int = 20,
        eps: float = 1e-6,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Deliberately delegate before recording.  This is an observation
        # interceptor, not a numerical replacement or a source-body patch.
        output = kernels.hc_split_sinkhorn(
            mixes, hc_scale, hc_base, hc_mult, sinkhorn_iters, eps
        )
        pre, post, comb = output
        identity = torch.eye(hc_mult, dtype=comb.dtype, device=comb.device)
        records.append(
            {
                "inputs": {
                    "mixes": tensor_record(mixes, include_storage=True),
                    "hc_scale": tensor_record(hc_scale, include_storage=True),
                    "hc_base": tensor_record(hc_base, include_storage=True),
                    "hc_mult": hc_mult,
                    "sinkhorn_iters": sinkhorn_iters,
                    "eps": eps,
                },
                "outputs": {
                    "pre": tensor_record(pre, include_storage=True),
                    "post": tensor_record(post, include_storage=True),
                    "comb": tensor_record(comb, include_storage=True),
                },
                "nontrivial": {
                    "pre_varies": bool((pre.float().amax() - pre.float().amin()) > 0),
                    "post_varies": bool(
                        (post.float().amax() - post.float().amin()) > 0
                    ),
                    "comb_not_identity": not bool((comb == identity).all()),
                },
            }
        )
        return output

    bundle.hc_split_sinkhorn = traced_hc_split_sinkhorn
    return bundle, records


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


@contextmanager
def hooks_for(model: torch.nn.Module) -> Iterator[dict[str, object]]:
    """Record selected real source submodule outputs, capped at a fixed bound."""
    records: dict[str, object] = {}
    handles: list[torch.utils.hooks.RemovableHandle] = []

    def capture(name: str):
        def hook(
            _module: torch.nn.Module, _inputs: tuple[object, ...], output: object
        ) -> None:
            if len(records) >= MAX_HOOK_RECORDS:
                raise RuntimeError("hook receipt cap reached")
            records[name] = object_record(output, include_storage=True)

        return hook

    def capture_input(name: str, *, exactly_one: bool):
        """Capture a selected module's source input without recomputing it."""

        def hook(_module: torch.nn.Module, inputs: tuple[object, ...]) -> None:
            if len(records) >= MAX_HOOK_RECORDS:
                raise RuntimeError("hook receipt cap reached")
            if not inputs or (exactly_one and len(inputs) != 1):
                raise RuntimeError(
                    f"unexpected source inputs for {name}: got {len(inputs)}"
                )
            records[name] = object_record(inputs[0], include_storage=True)

        return hook

    for name, module in model.named_modules():
        pieces = name.split(".")
        selected = (
            name in {"embed", "engram_hash", "norm", "head"}
            or (len(pieces) == 2 and pieces[0] == "layers")
            or (
                len(pieces) == 3
                and pieces[0] == "layers"
                and pieces[2] in {"engram", "attn", "ffn"}
            )
            or (
                len(pieces) == 4
                and pieces[:3] == ["layers", pieces[1], "attn"]
                and pieces[3] in {"compressor", "indexer"}
            )
            or (
                len(pieces) == 4
                and pieces[:3] == ["layers", pieces[1], "ffn"]
                and pieces[3] == "gate"
            )
        )
        if selected:
            # Source block, Engram, attention, compressor/indexer and routing
            # boundaries are sufficient to reconstruct this bounded graph trace.
            handles.append(module.register_forward_hook(capture(name)))
        if name == "norm":
            # The final HC collapse is passed directly to RMSNorm.  Retain the
            # source-produced input rather than reimplementing either operation
            # while exporting a test fixture.
            handles.append(
                module.register_forward_pre_hook(
                    capture_input("norm_input", exactly_one=True)
                )
            )
        if name == "layers.4.ffn":
            # MoE.forward receives (x, image_mask); x is the actual source
            # hidden state and is the only tensor exported for this oracle.
            handles.append(
                module.register_forward_pre_hook(
                    capture_input("layers.4.ffn_input", exactly_one=False)
                )
            )
    try:
        yield records
    finally:
        for handle in handles:
            handle.remove()


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
    kernel_bundle, hc_records = tracing_kernel_bundle()
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
            with hooks_for(model) as intermediate:
                for start_pos, chunk in calls:
                    intermediate.clear()
                    hc_records.clear()
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
                            "hyper_connection_mixes": hc_step_receipt(
                                hc_records, len(model.layers)
                            ),
                            "caches_after": cache_snapshots(graph, model),
                        }
                    )
        candidates = candidate_receipt(graph, args.window_size)
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
        "steps": steps,
        "candidate_filtering": candidates,
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


def moe_fixture(receipt: dict[str, object]) -> dict[str, object]:
    """Select actual layer-four MoE inputs, choices, outputs and encodings.

    This is a compact projection of a completed source capture.  It does not
    evaluate the gate or experts: expected values remain the hook records from
    the source's own ``layers.4.ffn`` execution.
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

    parameters = {
        name: record
        for name, record in encoded.items()
        if isinstance(name, str) and name.startswith("layers.4.ffn.")
    }
    routed = tuple(f"layers.4.ffn.experts.{index}" for index in range(4))
    expert_prefixes = (*routed, "layers.4.ffn.shared_experts")
    required_names = {
        "layers.4.ffn.gate.weight",
        "layers.4.ffn.gate.bias",
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
            "complete capture layer-four MoE parameter set differs: "
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
        "layers.4.ffn.gate.weight": ([4, 128], "torch.bfloat16"),
        "layers.4.ffn.gate.bias": ([4], "torch.float32"),
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
        expected_tensor_layouts[f"layers.4.ffn.shared_experts.{projection}.weight"] = (
            [128, 128],
            "torch.float8_e4m3fn",
        )
        expected_tensor_layouts[f"layers.4.ffn.shared_experts.{projection}.scale"] = (
            [4, 4],
            "torch.float8_e8m0fnu",
        )
    for name, (shape, dtype) in expected_tensor_layouts.items():
        record = parameters[name]
        if record.get("shape") != shape or record.get("dtype") != dtype:
            raise RuntimeError(f"MoE parameter {name} has an unexpected shape or dtype")

    cases: list[dict[str, object]] = []
    for step in steps:
        if not isinstance(step, dict):
            raise TypeError("complete capture includes an invalid step")
        start_pos = step.get("start_pos")
        intermediates = step.get("intermediates")
        if not isinstance(start_pos, int) or not isinstance(intermediates, dict):
            raise TypeError("complete capture step lacks MoE boundaries")
        ffn_input = intermediates.get("layers.4.ffn_input")
        ffn_output = intermediates.get("layers.4.ffn")
        gate = intermediates.get("layers.4.ffn.gate")
        if (
            not isinstance(ffn_input, dict)
            or not isinstance(ffn_output, dict)
            or not isinstance(gate, list)
            or len(gate) != 2
            or not all(isinstance(value, dict) for value in gate)
        ):
            raise TypeError("complete capture step lacks layer-four MoE hook records")
        if (
            ffn_input.get("shape") != ffn_output.get("shape")
            or ffn_input.get("dtype") != "torch.bfloat16"
            or ffn_output.get("dtype") != "torch.bfloat16"
            or gate[0].get("dtype") != "torch.float32"
            or gate[1].get("dtype") != "torch.int64"
        ):
            raise RuntimeError(
                "layer-four MoE hook storage has unexpected dtype or shape"
            )
        cases.append(
            {
                "start_pos": start_pos,
                "input": ffn_input,
                "gate_weights": gate[0],
                "gate_indices": gate[1],
                "output": ffn_output,
            }
        )
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
        "scope": "layer-four complete source MoE only; not Rust acceptance or full-model parity",
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
        "cases": cases,
        "comparison_policy": {
            "output_bf16": "exact storage bits",
            "selected_expert_ids": (
                "exact IDs; compare route weights by expert ID because source score "
                "order and a native sorted-ID traversal may differ"
            ),
            "route_weight_abs_error_max": 2**-20,
            "fixed_before_candidate_execution": True,
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
    args = parser.parse_args()
    receipt = run_capture()
    artifact_bytes = serialized_capture(receipt)
    if len(artifact_bytes) > MAX_CAPTURE_BYTES:
        raise RuntimeError(f"capture exceeds {MAX_CAPTURE_BYTES} byte receipt cap")
    if args.output is not None:
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
    if (
        args.output is None
        and args.head_fixture_output is None
        and args.moe_fixture_output is None
    ):
        print(artifact_bytes.decode("utf-8"), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
