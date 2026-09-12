#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["torch==2.13.0"]
# ///
"""Trace source compressed-KV publication order with deliberately non-math stubs.

Only checked `Attention._compress_kv` and `_compress_topk_idxs` methods execute
from the pinned source. Compressor, indexer, rotary, and quantization stubs
record observable handoff/order and visibly mutate data; they do not model
their native mathematics.
"""

from __future__ import annotations

import ast
import copy
import hashlib
import json
import struct
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

REVISION = "dba1be0a40aa45a94ad051997016db3960a90277"
SOURCE_SHA256 = "4e9ae23620edc8028ccc5d5fef552ab7fdc7dcd6f79608754fe9f67644056f65"
EXPECTED_TORCH_VERSION = "2.13.0"
ROOT = Path(__file__).resolve().parent.parent
SOURCE_PATH = ROOT / "artifacts" / "v41-reference-model.py"
EVENTS: list[dict[str, object]] = []


def bf16_bits(value: torch.Tensor) -> list[int]:
    """Return exact CPU BF16 storage words."""
    if value.dtype is not torch.bfloat16 or value.device.type != "cpu":
        raise TypeError("expected CPU BF16 tensor")
    return [
        int(word) & 0xFFFF
        for word in value.contiguous().view(torch.int16).reshape(-1).tolist()
    ]


def tensor_record(name: str, value: torch.Tensor) -> dict[str, object]:
    """Encode only BF16/FP32 tensors used as explicit trace payloads."""
    value = value.detach().cpu().contiguous()
    if value.dtype is torch.bfloat16:
        values = bf16_bits(value)
        raw = struct.pack(f"<{len(values)}H", *values)
        encoding = "bf16_bits_u16"
    elif value.dtype is torch.float32:
        values = [
            int(word) & 0xFFFF_FFFF
            for word in value.view(torch.int32).reshape(-1).tolist()
        ]
        raw = struct.pack(f"<{len(values)}I", *values)
        encoding = "f32_bits_u32"
    else:
        raise TypeError(f"unsupported tensor dtype {value.dtype}")
    return {
        "name": name,
        "dtype": str(value.dtype),
        "encoding": encoding,
        "shape": list(value.shape),
        "sha256_le_bytes": hashlib.sha256(raw).hexdigest(),
        "values": values,
    }


def source_methods() -> tuple[dict[str, Any], dict[str, Any]]:
    """SHA-check and extract exactly the two source publication methods."""
    source = SOURCE_PATH.read_bytes()
    actual = hashlib.sha256(source).hexdigest()
    if actual != SOURCE_SHA256:
        raise RuntimeError(f"refusing source SHA {actual}; expected {SOURCE_SHA256}")
    tree = ast.parse(source, filename=str(SOURCE_PATH))
    attention = next(
        (
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "Attention"
        ),
        None,
    )
    if attention is None:
        raise RuntimeError("pinned Attention class missing")
    wanted = {
        "_compress_topk_idxs": "compress_topk_idxs",
        "_compress_kv": "compress_kv",
    }
    methods: list[ast.FunctionDef] = []
    for source_name, extracted_name in wanted.items():
        method = next(
            (
                node
                for node in attention.body
                if isinstance(node, ast.FunctionDef) and node.name == source_name
            ),
            None,
        )
        if method is None:
            raise RuntimeError(f"pinned Attention.{source_name} missing")
        method = copy.deepcopy(method)
        method.name = extracted_name
        methods.append(method)
    namespace: dict[str, Any] = {
        "torch": torch,
        "shared_attn": SimpleNamespace(),
        "apply_rotary_emb": apply_rotary_stub,
        "fp4_act_quant": fp4_quant_stub,
    }
    exec(  # noqa: S102 -- two inspected methods from SHA-checked retained source only.
        compile(
            ast.fix_missing_locations(ast.Module(body=methods, type_ignores=[])),
            str(SOURCE_PATH),
            "exec",
        ),
        namespace,
    )
    return namespace, {name: namespace[name] for name in wanted.values()}


def apply_rotary_stub(tail: torch.Tensor, freqs: torch.Tensor) -> None:
    """Record frequency selection and visibly mutate the latent tail in place."""
    EVENTS.append(
        {
            "event": "rotary",
            "frequency_indices": [int(item) for item in freqs.reshape(-1).tolist()],
            "tail_before_bf16": bf16_bits(tail),
        }
    )
    tail.add_(10.0)


def fp4_quant_stub(
    value: torch.Tensor,
    block_size: int,
    inplace: bool,
    *,
    scale_dtype: torch.dtype,
) -> None:
    """Record pinned quant-call arguments and visibly mutate without quant math."""
    EVENTS.append(
        {
            "event": "quant",
            "block_size": block_size,
            "inplace": inplace,
            "scale_dtype": str(scale_dtype),
            "value_before_bf16": bf16_bits(value),
        }
    )
    value.add_(1.0)


class TraceCache:
    """Tensor-backed cache that records source writes and post-write prefix reads."""

    def __init__(self, storage: torch.Tensor) -> None:
        self.storage = storage

    def __setitem__(self, key: object, value: torch.Tensor) -> None:
        EVENTS.append(
            {"event": "cache_write", "key": repr(key), "value_bf16": bf16_bits(value)}
        )
        self.storage[key] = value

    def __getitem__(self, key: object) -> torch.Tensor:
        EVENTS.append({"event": "cache_read", "key": repr(key)})
        return self.storage[key]


class CompressorStub:
    """Explicit latent schedule; intentionally does not implement compression math."""

    def __init__(self, outputs: dict[tuple[int, int], torch.Tensor | None]) -> None:
        self.outputs = outputs

    def __call__(self, x: torch.Tensor, start_pos: int) -> torch.Tensor | None:
        key = (start_pos, x.size(1))
        value = self.outputs[key]
        EVENTS.append(
            {
                "event": "compressor",
                "start_pos": start_pos,
                "seqlen": x.size(1),
                "latent_bf16": None if value is None else bf16_bits(value),
            }
        )
        return None if value is None else value.clone()


class IndexerStub:
    """Records pre-RoPE latent visibility; intentionally does not score indices."""

    def __init__(self) -> None:
        self.freqs_cis: torch.Tensor | None = None
        self.calls = 0

    def __call__(
        self,
        x: torch.Tensor,
        _qr: torch.Tensor,
        latent: torch.Tensor | None,
        start_pos: int,
        offset: int,
    ) -> torch.Tensor:
        EVENTS.append(
            {
                "event": "indexer",
                "start_pos": start_pos,
                "offset": offset,
                "latent_before_mutation_bf16": None
                if latent is None
                else bf16_bits(latent),
            }
        )
        self.calls += 1
        return torch.full(
            (x.size(0), x.size(1), 1), self.calls, dtype=torch.int32, device=x.device
        )


class AttentionReceiver:
    """Only source fields accessed by the two extracted publication methods."""

    def __init__(
        self,
        methods: dict[str, Any],
        cache: TraceCache,
        compressor: CompressorStub | None,
        indexer: IndexerStub | None,
        *,
        kv_source: bool,
        index_source: bool,
    ) -> None:
        self.compress_ratio = 2
        self.is_kv_source = kv_source
        self.is_index_source = index_source
        self.compressor = compressor
        self.indexer = indexer
        self.compress_kv_cache = cache
        self.rope_head_dim = 2
        self.freqs_cis = torch.arange(8, dtype=torch.float32).reshape(8, 1)
        self._compress_topk_idxs = methods["compress_topk_idxs"].__get__(
            self, type(self)
        )
        self._compress_kv = methods["compress_kv"].__get__(self, type(self))


def case_record(
    name: str,
    receiver: AttentionReceiver,
    x: torch.Tensor,
    qr: torch.Tensor,
    start_pos: int,
    offset: int,
) -> dict[str, object]:
    """Execute one source call and preserve its event order and visible results."""
    EVENTS.clear()
    prefix, indices = receiver._compress_kv(x, qr, start_pos, offset)
    return {
        "name": name,
        "start_pos": start_pos,
        "seqlen": x.size(1),
        "offset": offset,
        "returned_prefix": tensor_record("prefix", prefix),
        "returned_indices_i32": [int(item) for item in indices.reshape(-1).tolist()],
        "events": list(EVENTS),
    }


def assert_event_order(case: dict[str, object], expected: list[str]) -> None:
    actual = [event["event"] for event in case["events"]]  # type: ignore[index]
    if actual != expected:
        raise AssertionError(
            f"{case['name']} event order {actual}; expected {expected}"
        )


def main() -> None:
    if torch.__version__.split("+", maxsplit=1)[0] != EXPECTED_TORCH_VERSION:
        raise RuntimeError(
            f"requires torch=={EXPECTED_TORCH_VERSION}, found {torch.__version__}"
        )
    torch.set_num_threads(1)
    namespace, methods = source_methods()
    cache = TraceCache(torch.zeros((1, 4, 2), dtype=torch.bfloat16))
    latent_prefill = torch.tensor([[[1.0, 2.0]]], dtype=torch.bfloat16)
    latent_decode = torch.tensor([[[3.0, 4.0]]], dtype=torch.bfloat16)
    compressor = CompressorStub(
        {
            (0, 3): latent_prefill,
            (3, 1): latent_decode,
            (4, 1): None,
            (0, 1): None,
        }
    )
    indexer = IndexerStub()
    source = AttentionReceiver(
        methods, cache, compressor, indexer, kv_source=True, index_source=True
    )
    namespace["shared_attn"] = SimpleNamespace()
    x3 = torch.zeros((1, 3, 2), dtype=torch.bfloat16)
    x1 = torch.zeros((1, 1, 2), dtype=torch.bfloat16)
    qr3 = torch.zeros((1, 3, 1), dtype=torch.bfloat16)
    qr1 = torch.zeros((1, 1, 1), dtype=torch.bfloat16)

    prefill = case_record("prefill_ratio2_s3", source, x3, qr3, 0, 7)
    completion = case_record("singleton_completion_start3", source, x1, qr1, 3, 7)
    nonboundary = case_record("nonboundary_start4", source, x1, qr1, 4, 7)
    assert_event_order(
        prefill,
        ["compressor", "indexer", "rotary", "quant", "cache_write", "cache_read"],
    )
    assert_event_order(
        completion,
        ["compressor", "indexer", "rotary", "quant", "cache_write", "cache_read"],
    )
    assert_event_order(nonboundary, ["compressor", "indexer", "cache_read"])
    if prefill["events"][1]["latent_before_mutation_bf16"] != bf16_bits(latent_prefill):  # type: ignore[index]
        raise AssertionError("indexer did not observe pre-RoPE pre-quant latent")
    if prefill["events"][4]["value_bf16"] == bf16_bits(latent_prefill):  # type: ignore[index]
        raise AssertionError("rotary/quant stubs did not visibly mutate cached latent")

    consumer = AttentionReceiver(
        methods,
        cache,
        compressor=None,
        indexer=None,
        kv_source=False,
        index_source=False,
    )
    consumer_case = case_record("consumer_reuses_shared_cache", consumer, x1, qr1, 4, 7)
    assert_event_order(consumer_case, ["cache_read"])
    if compressor.outputs[(3, 1)] is None or indexer.calls != 3:
        raise AssertionError(
            "consumer sharing unexpectedly recomputed a source publication"
        )

    if consumer_case["returned_indices_i32"] != nonboundary["returned_indices_i32"]:
        raise AssertionError("consumer did not reuse current source indices")
    zero_compress = case_record("initial_s1_zero_compress", source, x1, qr1, 0, 7)
    assert_event_order(zero_compress, ["compressor", "cache_read"])

    payload = {
        "schema_version": 1,
        "source": {
            "revision": REVISION,
            "sha256": SOURCE_SHA256,
            "path": "artifacts/v41-reference-model.py",
            "symbols": ["Attention._compress_kv", "Attention._compress_topk_idxs"],
            "url": f"https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/{REVISION}/inference/model.py#L722",
        },
        "receipt": {"device": "cpu", "torch_version": torch.__version__},
        "scope": {
            "included": "source publication control flow with explicit observable stubs",
            "excluded": [
                "compressor pooling and norm math",
                "index scoring/top-k math",
                "rotary math",
                "FP4 quantization math",
                "attention, cache allocation, and Metal execution",
            ],
        },
        "stub_contract": {
            "compressor": "scheduled explicit BF16 latent or None",
            "indexer": "records latent then returns sentinel int32 indices",
            "rotary": "adds 10 to the latent tail in place",
            "quant": "adds 1 in place after recording block-16 E4M3 arguments",
        },
        "assertions": [
            "indexer sees the latent before rotary/quant mutation",
            "source publishes index results before cache mutation",
            "returned prefix is read after source cache write",
            "prefill and singleton completion select group-start frequencies",
            "incomplete and zero-compress calls omit rotary/quant/cache write",
            "consumer reuse reads shared cache without compressor or indexer",
        ],
        "cases": [prefill, completion, nonboundary, zero_compress, consumer_case],
        "final_cache": tensor_record("compressed_cache", cache.storage),
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
