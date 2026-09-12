#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["torch==2.13.0"]
# ///
"""Capture a source-pinned reduced `Block.forward` sequencing oracle on CPU.

The capture invokes only checked AST extracts of `Block.forward`, its HC
helpers, and `RMSNorm.forward`. Explicit coefficient tensors and deterministic
feature-swap/identity sublayer stubs isolate the handoff order; this is neither
real attention/MoE nor full-model qualification.
"""

from __future__ import annotations

import ast
import copy
import hashlib
import json
import struct
from pathlib import Path
from typing import Any

import torch

REVISION = "dba1be0a40aa45a94ad051997016db3960a90277"
SOURCE_SHA256 = "4e9ae23620edc8028ccc5d5fef552ab7fdc7dcd6f79608754fe9f67644056f65"
EXPECTED_TORCH_VERSION = "2.13.0"
ROOT = Path(__file__).resolve().parent.parent
SOURCE_PATH = ROOT / "artifacts" / "v41-reference-model.py"


def extracted_methods() -> dict[str, Any]:
    """Hash-check and extract the four source methods this capture executes."""
    source = SOURCE_PATH.read_bytes()
    actual = hashlib.sha256(source).hexdigest()
    if actual != SOURCE_SHA256:
        raise RuntimeError(f"refusing source SHA {actual}; expected {SOURCE_SHA256}")
    tree = ast.parse(source, filename=str(SOURCE_PATH))
    requested = {
        ("Block", "hc_pre"): "block_hc_pre",
        ("Block", "hc_post"): "block_hc_post",
        ("Block", "forward"): "block_forward",
        ("RMSNorm", "forward"): "rms_norm_forward",
    }
    methods: list[ast.FunctionDef] = []
    for (class_name, method_name), extracted_name in requested.items():
        class_node = next(
            (
                node
                for node in tree.body
                if isinstance(node, ast.ClassDef) and node.name == class_name
            ),
            None,
        )
        if class_node is None:
            raise RuntimeError(f"pinned {class_name} class missing")
        method = next(
            (
                node
                for node in class_node.body
                if isinstance(node, ast.FunctionDef) and node.name == method_name
            ),
            None,
        )
        if method is None:
            raise RuntimeError(f"pinned {class_name}.{method_name} missing")
        method = copy.deepcopy(method)
        method.name = extracted_name
        methods.append(method)
    namespace: dict[str, Any] = {"torch": torch}
    exec(  # noqa: S102 -- methods are individually inspected from a SHA-checked source.
        compile(
            ast.fix_missing_locations(ast.Module(body=methods, type_ignores=[])),
            str(SOURCE_PATH),
            "exec",
        ),
        namespace,
    )
    return namespace


def bf16_bits(value: torch.Tensor) -> list[int]:
    """Encode a CPU BF16 tensor as its exact storage words."""
    if value.device.type != "cpu" or value.dtype is not torch.bfloat16:
        raise TypeError("expected CPU BF16 tensor")
    return [
        int(word) & 0xFFFF
        for word in value.contiguous().view(torch.int16).reshape(-1).tolist()
    ]


def tensor_record(name: str, value: torch.Tensor) -> dict[str, object]:
    """Serialize tensor shape, exact flat storage, and a little-endian digest."""
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
        raise TypeError(f"unsupported fixture dtype {value.dtype}")
    return {
        "name": name,
        "dtype": str(value.dtype),
        "encoding": encoding,
        "shape": list(value.shape),
        "sha256_le_bytes": hashlib.sha256(raw).hexdigest(),
        "values": values,
    }


class CapturedNorm:
    """Run source RMSNorm and retain its normalized input to each stub."""

    def __init__(self, forward: Any) -> None:
        self.forward = forward
        self.eps = 1.0e-6
        self.weight = torch.ones(2, dtype=torch.bfloat16)
        self.values: list[torch.Tensor] = []

    def __call__(self, value: torch.Tensor) -> torch.Tensor:
        result = self.forward(self, value)
        self.values.append(result.detach().clone())
        return result


class ReducedBlock:
    """Only the source fields/method calls consumed by extracted Block.forward."""

    def __init__(
        self,
        methods: dict[str, Any],
        attention_mix: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        ffn_mix: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> None:
        self.hc_attn_fn = object()
        self.hc_ffn_fn = object()
        self.hc_attn_scale = object()
        self.hc_ffn_scale = object()
        self.hc_attn_base = object()
        self.hc_ffn_base = object()
        self._attention_mix = attention_mix
        self._ffn_mix = ffn_mix
        self.hc_pre = methods["block_hc_pre"].__get__(self, type(self))
        self.hc_post = methods["block_hc_post"].__get__(self, type(self))
        self.attn_norm = CapturedNorm(methods["rms_norm_forward"])
        self.ffn_norm = CapturedNorm(methods["rms_norm_forward"])

    def hc_mixes(
        self,
        _x: torch.Tensor,
        fn: object,
        _scale: object,
        _base: object,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if fn is self.hc_attn_fn:
            return self._attention_mix
        if fn is self.hc_ffn_fn:
            return self._ffn_mix
        raise AssertionError("Block.forward requested an unknown HC projection")

    def attn(self, value: torch.Tensor, start_pos: int, *_args: object) -> torch.Tensor:
        if start_pos != 0:
            raise AssertionError("reduced source capture expects start_pos=0")
        return value.flip(-1)  # explicit deterministic attention stub

    def ffn(self, value: torch.Tensor, image_mask: object) -> torch.Tensor:
        if image_mask is not None:
            raise AssertionError("reduced source capture expects no image mask")
        return value  # explicit deterministic FFN stub


def mix(
    pre: list[float], post: list[float], comb: list[list[float]]
) -> tuple[torch.Tensor, ...]:
    return (
        torch.tensor([[[*pre]]], dtype=torch.float32),
        torch.tensor([[[*post]]], dtype=torch.float32),
        torch.tensor([[[*comb]]], dtype=torch.float32),
    )


def main() -> None:
    if torch.__version__.split("+", maxsplit=1)[0] != EXPECTED_TORCH_VERSION:
        raise RuntimeError(
            f"requires torch=={EXPECTED_TORCH_VERSION}, found {torch.__version__}"
        )
    torch.set_num_threads(1)
    methods = extracted_methods()
    block_forward = methods["block_forward"]

    # [batch=1, tokens=1, copies=2, width=2], copy-major values matching the
    # Rust reduced harness: [[2,-1], [6,3]].
    initial = torch.tensor([[[[2.0, -1.0], [6.0, 3.0]]]], dtype=torch.bfloat16)
    previous_ffn_pre = torch.tensor([[[0.25, 0.5]]], dtype=torch.float32)
    block0_attention = mix([0.75, 0.125], [1.0, 0.5], [[1.0, 0.0], [0.0, 1.0]])
    block0_ffn = mix([0.625, 0.375], [0.75, 1.25], [[0.0, 1.0], [1.0, 0.0]])
    block1_attention = mix([0.5, 0.5], [1.0, 1.0], [[1.0, 0.0], [0.0, 1.0]])
    block1_ffn = mix([0.4, 0.6], [1.0, 1.0], [[1.0, 0.0], [0.0, 1.0]])

    first = ReducedBlock(methods, block0_attention, block0_ffn)
    block0_output, block0_next_pre = block_forward(
        first, initial, 0, previous_ffn_pre, None
    )
    second = ReducedBlock(methods, block1_attention, block1_ffn)
    block1_output, block1_next_pre = block_forward(
        second, block0_output, 0, block0_next_pre, None
    )
    if not torch.equal(block0_next_pre, block0_ffn[0]):
        raise AssertionError("Block.forward did not return the current FFN pre mix")
    if not torch.equal(block1_next_pre, block1_ffn[0]):
        raise AssertionError("second Block.forward returned the wrong next pre mix")

    payload = {
        "schema_version": 1,
        "source": {
            "revision": REVISION,
            "sha256": SOURCE_SHA256,
            "path": "artifacts/v41-reference-model.py",
            "symbols": [
                "Block.forward",
                "Block.hc_pre",
                "Block.hc_post",
                "RMSNorm.forward",
            ],
            "url": f"https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/{REVISION}/inference/model.py#L907",
        },
        "receipt": {"device": "cpu", "torch_version": torch.__version__},
        "scope": {
            "included": "two reduced source Block.forward calls with source HC and RMSNorm methods",
            "excluded": [
                "real attention",
                "real MoE/FFN",
                "HC coefficient projection and Sinkhorn",
                "checkpoint loading",
                "multi-token/cache behavior",
                "Metal execution",
            ],
        },
        "assertions": [
            "previous FFN pre collapses block 0 attention input",
            "current attention pre collapses block 0 FFN input",
            "current FFN pre is returned for block 1 attention input",
            "attention stub swaps features and FFN stub preserves them",
        ],
        "tensors": [
            tensor_record("initial", initial),
            tensor_record("previous_ffn_pre", previous_ffn_pre),
            tensor_record("block0_attn_pre", block0_attention[0]),
            tensor_record("block0_attn_post", block0_attention[1]),
            tensor_record("block0_attn_comb", block0_attention[2]),
            tensor_record("block0_ffn_pre", block0_ffn[0]),
            tensor_record("block0_ffn_post", block0_ffn[1]),
            tensor_record("block0_ffn_comb", block0_ffn[2]),
            tensor_record("block0_attn_norm_input", first.attn_norm.values[0]),
            tensor_record("block0_ffn_norm_input", first.ffn_norm.values[0]),
            tensor_record("block0_output", block0_output),
            tensor_record("block0_next_pre", block0_next_pre),
            tensor_record("block1_attn_pre", block1_attention[0]),
            tensor_record("block1_attn_post", block1_attention[1]),
            tensor_record("block1_attn_comb", block1_attention[2]),
            tensor_record("block1_ffn_pre", block1_ffn[0]),
            tensor_record("block1_ffn_post", block1_ffn[1]),
            tensor_record("block1_ffn_comb", block1_ffn[2]),
            tensor_record("block1_attn_norm_input", second.attn_norm.values[0]),
            tensor_record("block1_ffn_norm_input", second.ffn_norm.values[0]),
            tensor_record("block1_output", block1_output),
            tensor_record("block1_next_pre", block1_next_pre),
        ],
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
