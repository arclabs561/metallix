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
"""Probe source-graph sensitivity to two bounded input partition schedules.

This is an opt-in source-only experiment.  It constructs fresh pinned source
graphs for the baseline and alternate schedules, then reports their terminal
logit and cache divergence.  It does not create a fixture, establish a
partition-invariance claim, download a checkpoint, or call Rust.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import struct
import sys
from collections.abc import Mapping
from contextlib import nullcontext
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
INPUT_IDS = (0, 1, 2, 3, 4, 5, 6)
BASELINE = (5, 1, 1)
ALTERNATE = (4, 1, 1, 1)
DEFAULT_PREFILL_TOKENS = 4
MAX_OUTPUT_BYTES = 16 << 20
SEED = 941


class ProbeError(ValueError):
    """The bounded probe input or captured run representation is invalid."""


def _canonical(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _file_sha256(path: Path) -> str:
    return _sha256(path.read_bytes())


def validate_schedule(counts: object) -> tuple[int, ...]:
    """Validate a named partition as positive chunks covering the seven IDs."""
    if not isinstance(counts, (tuple, list)) or not counts:
        raise ProbeError("schedule must be a nonempty sequence")
    normalized = tuple(counts)
    if any(type(count) is not int or count <= 0 for count in normalized):
        raise ProbeError("schedule counts must be positive integers")
    if sum(normalized) != len(INPUT_IDS):
        raise ProbeError("schedule must cover the seven pinned input IDs exactly")
    return normalized


def validate_prefill_tokens(value: object) -> int:
    """Admit only the bounded first chunk for the exploratory alternate sweep."""
    if type(value) is not int or not 2 <= value <= len(INPUT_IDS):
        raise ProbeError("prefill tokens must be an integer from 2 through 7")
    return value


def alternate_schedule(
    prefill_tokens: object = DEFAULT_PREFILL_TOKENS,
) -> tuple[int, ...]:
    """Build the alternate first chunk followed by one-token source calls."""
    prefill = validate_prefill_tokens(prefill_tokens)
    return validate_schedule((prefill,) + (1,) * (len(INPUT_IDS) - prefill))


def _tensor_words(record: object, label: str) -> tuple[list[int], list[int]]:
    if not isinstance(record, Mapping):
        raise ProbeError(f"{label} must be a tensor record")
    if record.get("dtype") != "torch.float32":
        raise ProbeError(f"{label} must retain torch.float32 logits")
    shape = record.get("shape")
    numel = record.get("numel")
    storage_hex = record.get("storage_hex")
    digest = record.get("storage_sha256")
    if (
        not isinstance(shape, list)
        or not all(type(width) is int and width > 0 for width in shape)
        or type(numel) is not int
        or not isinstance(storage_hex, str)
        or not isinstance(digest, str)
        or record.get("finite") is not True
    ):
        raise ProbeError(f"{label} is malformed or nonfinite")
    try:
        raw = bytes.fromhex(storage_hex)
    except ValueError as error:
        raise ProbeError(f"{label} has invalid storage hex") from error
    if math.prod(shape) != numel or len(raw) != numel * 4:
        raise ProbeError(f"{label} storage geometry is invalid")
    if _sha256(raw) != digest:
        raise ProbeError(f"{label} storage hash is invalid")
    words = list(struct.unpack(f"<{numel}I", raw))
    values = struct.unpack(f"<{numel}f", raw)
    if not all(math.isfinite(value) for value in values):
        raise ProbeError(f"{label} storage contains a nonfinite FP32 value")
    return shape, words


def _cache_identity(record: object, label: str) -> tuple[object, ...] | None:
    if record is None:
        return None
    if not isinstance(record, Mapping):
        raise ProbeError(f"{label} cache record must be an object or null")
    dtype = record.get("dtype")
    shape = record.get("shape")
    numel = record.get("numel")
    digest = record.get("storage_sha256")
    finite = record.get("finite")
    if (
        not isinstance(dtype, str)
        or not isinstance(shape, list)
        or not all(type(width) is int and width >= 0 for width in shape)
        or type(numel) is not int
        or not isinstance(digest, str)
        or type(finite) is not bool
    ):
        raise ProbeError(f"{label} cache record is malformed")
    if (
        math.prod(shape) != numel
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest.lower())
    ):
        raise ProbeError(f"{label} cache identity is invalid")
    return dtype, tuple(shape), numel, digest, finite


def compare_runs(baseline: object, alternate: object) -> dict[str, object]:
    """Compare completed run records without asserting partition invariance.

    Each run must contain a finite FP32 ``terminal_logits`` tensor record with
    exact ``storage_hex`` and a ``cache_after`` mapping of bounded tensor
    identity records.  Exact raw bits detect signed-zero changes separately
    from the floating maximum absolute difference.
    """
    if not isinstance(baseline, Mapping) or not isinstance(alternate, Mapping):
        raise ProbeError("runs must be objects")
    base_shape, base_words = _tensor_words(baseline.get("terminal_logits"), "baseline")
    alt_shape, alt_words = _tensor_words(alternate.get("terminal_logits"), "alternate")
    if base_shape != alt_shape:
        raise ProbeError("terminal logits have different shapes")
    if len(base_words) != len(alt_words):
        raise ProbeError("terminal logits have different scalar counts")
    base_values = struct.unpack(
        f"<{len(base_words)}f", struct.pack(f"<{len(base_words)}I", *base_words)
    )
    alt_values = struct.unpack(
        f"<{len(alt_words)}f", struct.pack(f"<{len(alt_words)}I", *alt_words)
    )
    max_abs = max(
        (abs(left - right) for left, right in zip(base_values, alt_values)), default=0.0
    )
    signed_zero_changes = sum(
        left != right and (left & 0x7FFFFFFF) == 0 and (right & 0x7FFFFFFF) == 0
        for left, right in zip(base_words, alt_words)
    )
    base_cache = baseline.get("cache_after")
    alt_cache = alternate.get("cache_after")
    if not isinstance(base_cache, Mapping) or not isinstance(alt_cache, Mapping):
        raise ProbeError("runs must retain cache_after mappings")
    base_names = set(base_cache)
    alt_names = set(alt_cache)
    matching: list[str] = []
    changed: list[dict[str, object]] = []
    for name in sorted(base_names & alt_names):
        base_identity = _cache_identity(base_cache[name], f"baseline cache {name}")
        alt_identity = _cache_identity(alt_cache[name], f"alternate cache {name}")
        if base_identity == alt_identity:
            matching.append(name)
        else:
            changed.append(
                {
                    "name": name,
                    "baseline": base_identity,
                    "alternate": alt_identity,
                }
            )
    return {
        "terminal": {
            "shape": base_shape,
            "exact_bits": base_words == alt_words,
            "max_abs_f32": max_abs,
            "signed_zero_bit_changes": signed_zero_changes,
        },
        "cache": {
            "matching_fields": matching,
            "changed_fields": [entry["name"] for entry in changed],
            "changed_identities": changed,
            "missing_from_baseline": sorted(alt_names - base_names),
            "missing_from_alternate": sorted(base_names - alt_names),
        },
    }


def compare_common_endpoints(
    baseline: Mapping, alternate: Mapping
) -> list[dict[str, object]]:
    """Compare states at equal token positions, not equal call ordinals."""
    base = {call["start_pos"] + call["token_count"]: call for call in baseline["calls"]}
    alt = {call["start_pos"] + call["token_count"]: call for call in alternate["calls"]}
    return [
        {
            "tokens_processed": end,
            **compare_runs(
                {
                    "terminal_logits": base[end]["logits"],
                    "cache_after": base[end]["cache_after"],
                },
                {
                    "terminal_logits": alt[end]["logits"],
                    "cache_after": alt[end]["cache_after"],
                },
            ),
        }
        for end in sorted(base.keys() & alt.keys())
    ]


def observer_noninterference(
    observed: Mapping[str, object], control: Mapping[str, object]
) -> dict[str, object]:
    """Require alternate observers to preserve each call's logits and caches."""
    observed_calls = observed.get("calls")
    control_calls = control.get("calls")
    if not isinstance(observed_calls, list) or not isinstance(control_calls, list):
        raise ProbeError("observer runs must retain ordered call records")
    if len(observed_calls) != len(control_calls):
        raise ProbeError("observer runs have different call counts")
    per_call: list[dict[str, object]] = []
    for index, (left, right) in enumerate(zip(observed_calls, control_calls)):
        if not isinstance(left, Mapping) or not isinstance(right, Mapping):
            raise ProbeError("observer call record is malformed")
        if left.get("start_pos") != right.get("start_pos") or left.get(
            "token_count"
        ) != right.get("token_count"):
            raise ProbeError("observer runs have different call geometry")
        comparison = compare_runs(
            {
                "terminal_logits": left.get("logits"),
                "cache_after": left.get("cache_after"),
            },
            {
                "terminal_logits": right.get("logits"),
                "cache_after": right.get("cache_after"),
            },
        )
        exact = (
            comparison["terminal"]["exact_bits"]
            and not comparison["cache"]["changed_fields"]
            and not comparison["cache"]["missing_from_baseline"]
            and not comparison["cache"]["missing_from_alternate"]
        )
        per_call.append(
            {
                "call_index": index,
                "start_pos": left["start_pos"],
                "token_count": left["token_count"],
                "exact_noninterference": exact,
                "comparison": comparison,
            }
        )
    return {
        "exact_noninterference": all(
            item["exact_noninterference"] for item in per_call
        ),
        "per_call": per_call,
    }


def _load_runner() -> Any:
    import importlib.util

    path = SCRIPTS / "v41-forward-reference.py"
    spec = importlib.util.spec_from_file_location("v41_partition_probe_runner", path)
    if spec is None or spec.loader is None:
        raise ProbeError("cannot load the pinned source runner helpers")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _cache_after(
    runner: Any, graph: Any, model: Any, tokens_processed: int
) -> dict[str, object]:
    """Record bounded cache identities without retaining cache storage twice."""
    snapshots: dict[str, object] = {}
    for layer_id, layer in enumerate(model.layers):
        attention = layer.attn
        snapshots[f"layer_{layer_id}.window_kv"] = runner.tensor_record(
            attention.window_kv_cache
        )
        if hasattr(attention, "compress_kv_cache"):
            snapshots[f"layer_{layer_id}.compressed_kv"] = runner.tensor_record(
                attention.compress_kv_cache
            )
            snapshots[f"layer_{layer_id}.compressed_kv_prefix"] = runner.tensor_record(
                attention.compress_kv_cache[
                    :, : tokens_processed // attention.compress_ratio
                ]
            )
        if attention.indexer is not None and hasattr(attention.indexer, "k_cache"):
            snapshots[f"layer_{layer_id}.index_k"] = runner.tensor_record(
                attention.indexer.k_cache
            )
    for name, buffer in model.named_buffers():
        if name.endswith(("kv_state", "score_state")):
            snapshots[f"buffer.{name}"] = runner.tensor_record(buffer)
    if model.engram_hash is not None:
        snapshots["engram.hash_cache"] = runner.tensor_record(model.engram_hash.cache)
    for name in ("compress_kv", "index_k", "topk_idxs", "candidates"):
        value = getattr(graph.shared_attn, name)
        snapshots[f"shared.{name}"] = (
            None if value is None else runner.tensor_record(value)
        )
    return snapshots


def _source_args(
    runner: Any, graph: Any, manifest: dict[str, Any]
) -> tuple[Any, Any, dict[str, Any], dict[str, Any]]:
    """Use canonical manifest model/token geometry; schedules stay probe-owned."""
    specification = runner.forward_manifest.synthetic_tokenizer_spec(manifest)
    tokenizer = runner.SyntheticTokenizer(
        specification["decoded_tokens"], specification["raw_token_strings"]
    )
    model_args = runner.forward_manifest.model_args(manifest)
    return graph.ModelArgs(**model_args), tokenizer, specification, model_args


def _run_schedule(
    runner: Any,
    manifest: dict[str, Any],
    counts: tuple[int, ...],
    *,
    capture: bool = False,
) -> dict[str, object]:
    """Run one schedule in a fresh source graph, runtime, and model instance."""
    import torch

    if capture:
        kernel_bundle, hc_records, sparse_records = (
            runner.v41_forward_observers.tracing_kernel_bundle(
                runner.kernels, runner.source_loader.KERNEL_NAMES, runner.tensor_record
            )
        )
    else:
        kernel_bundle, hc_records, sparse_records = runner.kernels, [], []
    graph = runner.source_loader.load_text_graph(kernel_bundle)
    graph.shared_attn = graph.SharedAttentionRuntime()
    with graph.set_dtype(torch.bfloat16):
        args, tokenizer, tokenizer_spec, model_args = _source_args(
            runner, graph, manifest
        )
        model = graph.Transformer(args, tokenizer).eval()
        encoded = runner.initialize_parameters(model)
        initialized_buffers = runner.initialize_runtime_buffers(model)
        input_ids = torch.tensor((INPUT_IDS,), dtype=torch.int64)
        if input_ids.max().item() >= args.vocab_size:
            raise ProbeError(
                "pinned input IDs exceed the canonical synthetic vocabulary"
            )
        calls: list[dict[str, object]] = []
        captured_calls: list[dict[str, object]] = []
        offset = 0

        def execute_call(count: int, intermediate: Any | None = None) -> None:
            nonlocal offset
            start = offset
            stop = start + count
            chunk = input_ids[:, start:stop]
            if intermediate is not None:
                intermediate.clear()
                hc_records.clear()
                sparse_records.clear()
            output_ids, logits, _ = model(chunk, start_pos=start)
            if not bool(torch.isfinite(logits).all()):
                raise ProbeError(
                    f"source model produced nonfinite logits at start {start}"
                )
            calls.append(
                {
                    "start_pos": start,
                    "token_count": count,
                    "input_ids": runner.tensor_record(chunk, include_storage=True),
                    "output_ids": runner.tensor_record(
                        output_ids, include_storage=True
                    ),
                    "logits": runner.tensor_record(logits, include_storage=True),
                    "cache_after": _cache_after(runner, graph, model, stop),
                }
            )
            if intermediate is not None:
                captured_calls.append(
                    {
                        "start_pos": start,
                        "token_count": count,
                        "intermediates": dict(intermediate),
                        "hyper_connection_mixes": runner.v41_forward_observers.hc_step_receipt(
                            hc_records, len(model.layers)
                        ),
                        "sparse_attention_calls": runner.v41_forward_observers.sparse_step_receipt(
                            sparse_records, len(model.layers)
                        ),
                        "caches_after": runner.cache_snapshots(graph, model),
                    }
                )
            offset = stop

        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(SEED)
            hook_scope = (
                runner.v41_forward_observers.hooks_for(
                    model,
                    graph,
                    runner.tensor_record,
                    runner.object_record,
                    runner.MAX_HOOK_RECORDS,
                )
                if capture
                else nullcontext(None)
            )
            with hook_scope as intermediate:
                for count in counts:
                    execute_call(count, intermediate)
        cache_after = _cache_after(runner, graph, model, offset)
        run: dict[str, object] = {
            "schedule": list(counts),
            "calls": calls,
            "terminal_logits": calls[-1]["logits"],
            "cache_after": cache_after,
            "source_cache_observers": {
                "layer_1_index_k": cache_after.get("layer_1.index_k"),
                "layer_3_index_k": cache_after.get("layer_3.index_k"),
            },
            "parameter_initializer_sha256": _sha256(_canonical(encoded)),
            "tokenizer_spec_sha256": _sha256(_canonical(tokenizer_spec)),
            "initialized_buffers": initialized_buffers,
        }
        if capture:
            probe_sha = _file_sha256(Path(__file__))
            run["alternate_capture"] = {
                "schema_version": 1,
                "scope": (
                    "alternate-partition source observations for later extraction; "
                    "not a canonical complete capture or qualified fixture"
                ),
                "capture_identity": {
                    "probe_sha256": probe_sha,
                    "schedule": list(counts),
                    "sha256": _sha256(
                        _canonical(
                            {"probe_sha256": probe_sha, "schedule": list(counts)}
                        )
                    ),
                },
                "encoded_parameters": encoded,
                "actual_model_args": model_args,
                "engram": runner.engram_receipt(model, manifest),
                "attention_static": {
                    "layer_3_freqs_cis": runner.tensor_record(
                        model.layers[3].attn.freqs_cis, include_storage=True
                    ),
                    "layer_4_freqs_cis": runner.tensor_record(
                        model.layers[4].attn.freqs_cis, include_storage=True
                    ),
                },
                "calls": captured_calls,
            }
        return run


def _baseline_oracle() -> dict[str, object]:
    bundle = json.loads(
        (
            ROOT / "fixtures" / "deepseek-v41" / "reduced-runner-reference.json"
        ).read_text()
    )
    head = bundle["projections"]["head"]["cases"][-1]
    shape = head["logits_shape"]
    bits = head["logits_fp32_bits"]
    if not (
        isinstance(shape, list)
        and all(type(width) is int and width > 0 for width in shape)
        and isinstance(bits, list)
        and len(bits) == math.prod(shape)
        and all(type(word) is int and 0 <= word <= 0xFFFFFFFF for word in bits)
    ):
        raise ProbeError("bundled source head oracle is malformed")
    compressed_kv = bundle["projections"]["layer3_attention"]["cases"][-1][
        "compressed_kv"
    ]
    return {
        "head": {"shape": shape, "fp32_bits": bits, "start_pos": head["start_pos"]},
        "publication": {
            "cache_field": "layer_3.compressed_kv_prefix",
            "storage_sha256": compressed_kv["storage_sha256"],
            "shape": compressed_kv["shape"],
            "dtype": compressed_kv["dtype"],
        },
    }


def _matches_baseline_oracle(
    run: Mapping[str, object], oracle: Mapping[str, object]
) -> dict[str, object]:
    head = oracle.get("head")
    publication = oracle.get("publication")
    if not isinstance(head, Mapping) or not isinstance(publication, Mapping):
        raise ProbeError("baseline oracle representation is malformed")
    shape, words = _tensor_words(run.get("terminal_logits"), "baseline terminal")
    expected_shape = head.get("shape")
    expected_words = head.get("fp32_bits")
    if not isinstance(expected_shape, list) or not isinstance(expected_words, list):
        raise ProbeError("head oracle representation is malformed")
    cache = run.get("cache_after")
    if not isinstance(cache, Mapping):
        raise ProbeError("baseline cache representation is malformed")
    field = publication.get("cache_field")
    if not isinstance(field, str):
        raise ProbeError("baseline publication oracle field is malformed")
    observed_publication = cache.get(field)
    if not isinstance(observed_publication, Mapping):
        raise ProbeError("baseline publication cache is absent")
    return {
        "head": {
            "shape_matches": shape == expected_shape,
            "exact_bits": words == expected_words,
            "oracle_start_pos": head.get("start_pos"),
        },
        "layer3_compressed_kv": {
            "field": field,
            "dtype_matches": observed_publication.get("dtype")
            == publication.get("dtype"),
            "shape_matches": observed_publication.get("shape")
            == publication.get("shape"),
            "storage_sha256_matches": observed_publication.get("storage_sha256")
            == publication.get("storage_sha256"),
        },
    }


def _source_metadata(runner: Any) -> dict[str, object]:
    return {
        "revision": runner.SOURCE_REVISION,
        "model_sha256": runner.source_loader.MODEL_SHA256,
        "engram_sha256": runner.source_loader.ENGRAM_SHA256,
        "runner_sha256": _file_sha256(SCRIPTS / "v41-forward-reference.py"),
        "loader_sha256": _file_sha256(SCRIPTS / "v41_source_loader.py"),
        "kernel_source_sha256": runner.KERNEL_SHA256,
        "cpu_backend_sha256": _file_sha256(SCRIPTS / "v41_cpu_kernels.py"),
        "observer_sha256": _file_sha256(SCRIPTS / "v41_forward_observers.py"),
        "probe_sha256": _file_sha256(Path(__file__)),
    }


def build_probe(
    capture_alternate: bool = False,
    prefill_tokens: object = DEFAULT_PREFILL_TOKENS,
) -> dict[str, object]:
    """Execute both named schedules and retain a divergence report or failure."""
    baseline = validate_schedule(BASELINE)
    alternate = alternate_schedule(prefill_tokens)
    runner = _load_runner()
    if (
        _file_sha256(ROOT / "artifacts" / "v41-kernel-pinned.py")
        != runner.KERNEL_SHA256
    ):
        raise ProbeError("retained source kernel hash differs from the probe contract")
    manifest = runner.forward_manifest.manifest()
    # ``model_args`` performs the canonical manifest validation before the
    # source constructor.  The probe deliberately owns only the chunk schedule.
    model_args = runner.forward_manifest.model_args(manifest)
    common = {
        "schema_version": 1,
        "scope": (
            "source-only bounded partition experiment; records divergence, not a "
            "partition-invariance, native-parity, checkpoint, or serving claim"
        ),
        "source": _source_metadata(runner),
        "execution": {
            "input_ids": list(INPUT_IDS),
            "baseline_schedule": list(baseline),
            "alternate_schedule": list(alternate),
            "seed": SEED,
            "fresh_graph_runtime_and_model_per_schedule": True,
            "capture_alternate": capture_alternate,
            "network": "not used",
            "checkpoint_download": "not used",
        },
        "canonical_model_geometry": {
            "model_args_sha256": _sha256(_canonical(model_args)),
            "manifest_sha256": _sha256(_canonical(manifest)),
            "manifest_trace_not_used_as_probe_schedule": True,
        },
        "runtime": {
            "python": sys.version.split()[0],
            "torch": runner.torch.__version__,
            "device": "cpu",
            "storage_byteorder": sys.byteorder,
            "torch_num_threads": runner.torch.get_num_threads(),
            "torch_num_interop_threads": runner.torch.get_num_interop_threads(),
        },
    }
    try:
        baseline_run = _run_schedule(runner, manifest, baseline)
    except (
        ArithmeticError,
        AssertionError,
        IndexError,
        KeyError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as error:
        return {
            **common,
            "status": "source_execution_failed",
            "failed_schedule": "baseline",
            "error": {"type": type(error).__name__, "message": str(error)},
        }
    baseline_oracle = _matches_baseline_oracle(baseline_run, _baseline_oracle())
    if not (
        baseline_oracle["head"]["shape_matches"]
        and baseline_oracle["head"]["exact_bits"]
        and baseline_oracle["layer3_compressed_kv"]["dtype_matches"]
        and baseline_oracle["layer3_compressed_kv"]["shape_matches"]
        and baseline_oracle["layer3_compressed_kv"]["storage_sha256_matches"]
    ):
        return {
            **common,
            "status": "baseline_oracle_mismatch",
            "baseline": baseline_run,
            "baseline_oracle": baseline_oracle,
        }
    try:
        alternate_run = _run_schedule(
            runner, manifest, alternate, capture=capture_alternate
        )
    except (
        ArithmeticError,
        AssertionError,
        IndexError,
        KeyError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as error:
        return {
            **common,
            "status": "source_execution_failed",
            "failed_schedule": "alternate",
            "baseline": baseline_run,
            "baseline_oracle": baseline_oracle,
            "error": {"type": type(error).__name__, "message": str(error)},
        }
    alternate_control: dict[str, object] | None = None
    observer_control: dict[str, object] | None = None
    if capture_alternate:
        try:
            alternate_control = _run_schedule(runner, manifest, alternate)
        except (
            ArithmeticError,
            AssertionError,
            IndexError,
            KeyError,
            RuntimeError,
            TypeError,
            ValueError,
        ) as error:
            return {
                **common,
                "status": "source_execution_failed",
                "failed_schedule": "alternate_unobserved_control",
                "baseline": baseline_run,
                "baseline_oracle": baseline_oracle,
                "alternate": alternate_run,
                "error": {"type": type(error).__name__, "message": str(error)},
            }
        observer_control = observer_noninterference(alternate_run, alternate_control)
        if not observer_control["exact_noninterference"]:
            return {
                **common,
                "status": "observer_interference_detected",
                "baseline": baseline_run,
                "baseline_oracle": baseline_oracle,
                "alternate": alternate_run,
                "alternate_unobserved_control": alternate_control,
                "observer_noninterference": observer_control,
            }
    if (
        baseline_run["parameter_initializer_sha256"]
        != alternate_run["parameter_initializer_sha256"]
        or baseline_run["tokenizer_spec_sha256"]
        != alternate_run["tokenizer_spec_sha256"]
    ):
        raise ProbeError(
            "fresh schedules did not retain identical initializer/tokenizer bytes"
        )
    return {
        **common,
        "status": "completed_source_partition_experiment",
        "baseline": baseline_run,
        "alternate": alternate_run,
        "baseline_oracle": baseline_oracle,
        "comparison": compare_runs(baseline_run, alternate_run),
        "common_endpoints": compare_common_endpoints(baseline_run, alternate_run),
        **(
            {
                "alternate_unobserved_control": alternate_control,
                "observer_noninterference": observer_control,
            }
            if capture_alternate
            else {}
        ),
    }


def _write_new(path: Path, receipt: Mapping[str, object]) -> None:
    encoded = (
        json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode()
    if len(encoded) > MAX_OUTPUT_BYTES:
        raise ProbeError("partition probe receipt exceeds the 16 MiB bound")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError as error:
        raise ProbeError(f"refusing to overwrite existing output: {path}") from error
    with os.fdopen(descriptor, "wb") as destination:
        destination.write(encoded)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run", action="store_true", help="execute the local source probe"
    )
    parser.add_argument("--output", type=Path, required=True, help="new receipt path")
    parser.add_argument(
        "--capture-alternate",
        action="store_true",
        help="retain full alternate-only source observers and verify they do not interfere",
    )
    parser.add_argument(
        "--prefill-tokens",
        type=int,
        default=DEFAULT_PREFILL_TOKENS,
        help="alternate first chunk size from 2 through 7; defaults to 4",
    )
    args = parser.parse_args(argv)
    if not args.run:
        parser.error("--run is required to execute the source graph")
    try:
        receipt = build_probe(
            capture_alternate=args.capture_alternate,
            prefill_tokens=args.prefill_tokens,
        )
        _write_new(args.output, receipt)
    except (OSError, ProbeError, ValueError, RuntimeError) as error:
        print(f"partition probe error: {error}", file=sys.stderr)
        return 1
    return 0 if receipt["status"] == "completed_source_partition_experiment" else 1


if __name__ == "__main__":
    raise SystemExit(main())
