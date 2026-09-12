#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["torch==2.13.0"]
# ///
"""Capture a source-pinned synthetic V4.1 Engram residual-gate reference.

Only ``Engram.forward`` is extracted from the retained source. Fixed embedding
and projection stubs supply explicit preprojected key/value data, so this is a
gate/residual composition capture, not an Engram-table, FP8-row, tokenizer, or
checkpoint test.
"""

from __future__ import annotations

import ast
import hashlib
import json
import struct
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

REVISION = "dba1be0a40aa45a94ad051997016db3960a90277"
SOURCE_SHA256 = "4e9ae23620edc8028ccc5d5fef552ab7fdc7dcd6f79608754fe9f67644056f65"
SOURCE_PATH = (
    Path(__file__).resolve().parent.parent / "artifacts" / "v41-reference-model.py"
)
EXPECTED_TORCH_VERSION = "2.13.0"


def load_pinned_forward() -> Callable[..., torch.Tensor]:
    """Hash-check and AST-extract only ``Engram.forward`` from the source."""
    source = SOURCE_PATH.read_bytes()
    actual = hashlib.sha256(source).hexdigest()
    if actual != SOURCE_SHA256:
        raise RuntimeError(f"refusing source SHA {actual}; expected {SOURCE_SHA256}")
    module = ast.parse(source, filename=str(SOURCE_PATH))
    engram = next(
        (
            node
            for node in module.body
            if isinstance(node, ast.ClassDef) and node.name == "Engram"
        ),
        None,
    )
    if engram is None:
        raise RuntimeError("pinned Engram class missing")
    forward = next(
        (
            node
            for node in engram.body
            if isinstance(node, ast.FunctionDef) and node.name == "forward"
        ),
        None,
    )
    if forward is None:
        raise RuntimeError("pinned Engram.forward missing")
    extracted = ast.Module(body=[forward], type_ignores=[])
    ast.fix_missing_locations(extracted)
    namespace: dict[str, Any] = {"torch": torch}
    exec(compile(extracted, str(SOURCE_PATH), "exec"), namespace)  # noqa: S102 -- source hash is checked above.
    result = namespace.get("forward")
    if not callable(result):
        raise TypeError("AST extraction did not produce callable forward")
    return result


def bf16_bits(value: torch.Tensor) -> list[int]:
    """Serialize CPU BF16 values exactly as unsigned 16-bit words."""
    if value.device.type != "cpu" or value.dtype is not torch.bfloat16:
        raise TypeError("expected CPU BF16 tensor")
    return [
        int(word) & 0xFFFF
        for word in value.contiguous().view(torch.int16).reshape(-1).tolist()
    ]


def tensor_record(name: str, value: torch.Tensor) -> dict[str, object]:
    """Emit values and a deterministic little-endian byte digest."""
    value = value.detach().cpu().contiguous()
    if value.dtype is torch.bfloat16:
        bits = bf16_bits(value)
        raw = struct.pack(f"<{len(bits)}H", *bits)
        values: object = bits
        encoding = "bf16_bits_u16"
    elif value.dtype is torch.float32:
        bits = [
            int(word) & 0xFFFF_FFFF
            for word in value.view(torch.int32).reshape(-1).tolist()
        ]
        raw = struct.pack(f"<{len(bits)}I", *bits)
        values = bits
        encoding = "f32_bits_u32"
    elif value.dtype is torch.bool:
        flat = value.reshape(-1).tolist()
        raw = bytes(int(item) for item in flat)
        values = flat
        encoding = "bool"
    elif value.dtype is torch.int64:
        flat = value.reshape(-1).tolist()
        raw = struct.pack(f"<{len(flat)}q", *flat)
        values = flat
        encoding = "i64"
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


class FixedEmbed:
    """Stub returning explicit synthetic rows while checking requested IDs."""

    def __init__(self, expected_ids: torch.Tensor, rows: torch.Tensor) -> None:
        self.expected_ids = expected_ids
        self.rows = rows

    def __call__(self, hash_ids: torch.Tensor) -> torch.Tensor:
        if not torch.equal(hash_ids, self.expected_ids):
            raise AssertionError("Engram.forward changed hash-ID handoff to embedding")
        return self.rows


class FixedWkv:
    """Stub returning explicit key/value projection output after row flattening."""

    def __init__(self, expected_input: torch.Tensor, output: torch.Tensor) -> None:
        self.expected_input = expected_input
        self.output = output

    def __call__(self, rows: torch.Tensor) -> torch.Tensor:
        if not torch.equal(rows, self.expected_input):
            raise AssertionError(
                "Engram.forward changed embedding-row flattening before wkv"
            )
        return self.output


def recomputed_gate(
    x: torch.Tensor,
    key: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    eps: float,
    clamp_value: float,
    token_mask: torch.Tensor,
) -> torch.Tensor:
    """Recompute the source gate expression to check fixture consistency."""
    h = x.float()
    weight = q_weight.float() * k_weight.float()
    rstd = torch.rsqrt(h.square().mean(-1) + eps) * torch.rsqrt(
        key.square().mean(-1) + eps
    )
    dot = (h * weight * key).sum(-1) * rstd * x.shape[-1] ** -0.5
    gate = torch.sigmoid(torch.copysign(dot.abs().clamp_min(clamp_value).sqrt(), dot))
    return gate.masked_fill(~token_mask.unsqueeze(-1), 0)


def holder(
    hash_ids: torch.Tensor,
    rows: torch.Tensor,
    kv: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
) -> SimpleNamespace:
    """Build only the fields read by the extracted method."""
    return SimpleNamespace(
        embed=FixedEmbed(hash_ids, rows),
        wkv=FixedWkv(rows.flatten(-2), kv),
        hc_mult=2,
        dim=4,
        eps=1.0e-6,
        clamp_value=1.0e-6,
        q_weight=q_weight,
        k_weight=k_weight,
    )


def main() -> None:
    if torch.__version__.split("+", maxsplit=1)[0] != EXPECTED_TORCH_VERSION:
        raise RuntimeError(
            f"requires torch=={EXPECTED_TORCH_VERSION}, found {torch.__version__}"
        )
    torch.set_num_threads(1)
    forward = load_pinned_forward()

    # [B=1, L=3, hc=2, dim=4]. Position 0 has opposite-sign copy dots;
    # position 1 exercises unequal per-copy norms; position 2 is masked.
    x = torch.tensor(
        [
            [
                [[1.0, 2.0, -1.0, 0.5], [2.0, -1.0, 0.5, 1.0]],
                [[0.25, -0.5, 1.0, 2.0], [3.0, 0.5, -2.0, 1.0]],
                [[1.0, 1.0, 1.0, 1.0], [-1.0, -1.0, -1.0, -1.0]],
            ]
        ],
        dtype=torch.bfloat16,
    )
    hash_ids = torch.tensor([[[2, 5], [7, 11], [13, 17]]], dtype=torch.int64)
    rows = torch.tensor(
        [
            [
                [[0.5, -1.0], [1.5, 0.25]],
                [[-0.5, 2.0], [0.75, -1.5]],
                [[1.0, 1.0], [-1.0, -1.0]],
            ]
        ],
        dtype=torch.bfloat16,
    )
    key = torch.tensor(
        [
            [
                [[1.0, 0.5, -0.5, 2.0], [-1.0, -0.5, 0.5, -2.0]],
                [[-2.0, 1.0, 0.25, -0.5], [0.5, 2.0, -1.0, 0.25]],
                [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
            ]
        ],
        dtype=torch.bfloat16,
    )
    value = torch.tensor(
        [[[0.5, -1.0, 2.0, 0.25], [1.0, 0.5, -0.25, 2.0], [4.0, -4.0, 2.0, -2.0]]],
        dtype=torch.bfloat16,
    )
    kv = torch.cat((key.flatten(-2), value), dim=-1)
    q_weight = torch.tensor([[2.0, 0.5, 3.0, 1.0], [0.25, 4.0, 0.5, 2.0]])
    k_weight = torch.tensor([[0.5, 2.0, 0.5, 1.0], [4.0, 0.25, 2.0, 0.5]])
    token_mask = torch.tensor([[True, True, False]])

    receiver = holder(hash_ids, rows, kv, q_weight, k_weight)
    output = forward(receiver, x, hash_ids, token_mask)
    gate = recomputed_gate(
        x,
        key.float(),
        q_weight,
        k_weight,
        receiver.eps,
        receiver.clamp_value,
        token_mask,
    )
    expected = (x.float() + gate.unsqueeze(-1) * value.float().unsqueeze(-2)).to(
        x.dtype
    )
    if not torch.equal(output, expected):
        raise AssertionError(
            "pinned Engram.forward diverged from the stated gate composition"
        )
    if not (gate[0, 0, 0] > 0.5 and gate[0, 0, 1] < 0.5):
        raise AssertionError(
            "synthetic signed positive/negative gate case lost direction"
        )
    if not torch.equal(output[:, 2], x[:, 2]):
        raise AssertionError("masked Engram position did not preserve the input copies")

    # The zero-dot third position is masked above. Re-run it unmasked so the
    # clamp floor has observable source output rather than a hidden gate.
    unmasked_gate = recomputed_gate(
        x,
        key.float(),
        q_weight,
        k_weight,
        receiver.eps,
        receiver.clamp_value,
        torch.ones_like(token_mask),
    )
    unmasked_output = forward(receiver, x, hash_ids, None)
    expected_unmasked = (
        x.float() + unmasked_gate.unsqueeze(-1) * value.float().unsqueeze(-2)
    ).to(x.dtype)
    if not torch.equal(unmasked_output, expected_unmasked):
        raise AssertionError(
            "unmasked Engram.forward diverged from the recomputed gate"
        )
    clamp_gate = torch.sigmoid(torch.sqrt(torch.tensor(receiver.clamp_value)))
    if not (unmasked_gate[0, 2] == clamp_gate).all() or not (clamp_gate > 0.5):
        raise AssertionError(
            "unmasked zero dot did not exercise the positive clamp floor"
        )
    wrong_copy_specific = value.float().unsqueeze(-2).expand(-1, -1, 2, -1).clone()
    wrong_copy_specific[:, :, 1].zero_()
    wrong_output = (x.float() + unmasked_gate.unsqueeze(-1) * wrong_copy_specific).to(
        x.dtype
    )
    if torch.equal(unmasked_output, wrong_output):
        raise AssertionError(
            "fixture did not distinguish shared value broadcast across HC copies"
        )

    # Changing q/k factors but preserving their pointwise product is observationally identical.
    q_factorized = q_weight * 2.0
    k_factorized = k_weight * 0.5
    factorized = forward(
        holder(hash_ids, rows, kv, q_factorized, k_factorized), x, hash_ids, token_mask
    )
    if not torch.equal(output, factorized):
        raise AssertionError("Engram q/k weights were not used solely by their product")

    # A deliberately incorrect joint-HC normalization must differ from the published per-copy gate.
    h, k = x.float(), key.float()
    joint_rstd = torch.rsqrt(h.square().mean((-1, -2)) + receiver.eps) * torch.rsqrt(
        k.square().mean((-1, -2)) + receiver.eps
    )
    joint_dot = (
        (h * (q_weight * k_weight) * k).sum(-1)
        * joint_rstd.unsqueeze(-1)
        * x.shape[-1] ** -0.5
    )
    joint_gate = torch.sigmoid(
        torch.copysign(
            joint_dot.abs().clamp_min(receiver.clamp_value).sqrt(), joint_dot
        )
    )
    if torch.equal(gate[:, :2], joint_gate[:, :2]):
        raise AssertionError(
            "fixture did not distinguish per-copy from joint-HC normalization"
        )

    payload = {
        "schema_version": 1,
        "source": {
            "revision": REVISION,
            "sha256": SOURCE_SHA256,
            "path": "artifacts/v41-reference-model.py",
            "symbol": "Engram.forward",
            "url": f"https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/{REVISION}/inference/model.py#L328",
        },
        "receipt": {"device": "cpu", "torch_version": torch.__version__},
        "scope": {
            "included": "Engram.forward residual gate with explicit stubbed embedding and wkv output",
            "excluded": [
                "Engram hash generation",
                "FP8 table rows and scales",
                "embedding sharding and all_reduce",
                "wkv weight projection",
                "checkpoint loading",
                "CED and Metal execution",
            ],
        },
        "assertions": [
            "fixed hash IDs reach embedding and rows flatten before wkv",
            "per-HC normalization differs from joint-HC normalization",
            "positive and negative dots produce opposite signed-sqrt gate directions",
            "q_weight and k_weight affect output only through their product",
            "one shared value broadcasts across HC copies",
            "masked token preserves BF16 input copies",
            "unmasked zero dot applies the positive signed-sqrt clamp floor",
        ],
        "parameters": {
            "eps": receiver.eps,
            "clamp_value": receiver.clamp_value,
            "eps_f32_bits": struct.unpack("<I", struct.pack("<f", receiver.eps))[0],
            "clamp_value_f32_bits": struct.unpack(
                "<I", struct.pack("<f", receiver.clamp_value)
            )[0],
        },
        "tensors": [
            tensor_record("x", x),
            tensor_record("hash_ids", hash_ids),
            tensor_record("stubbed_embedding_rows", rows),
            tensor_record("stubbed_wkv_key", key),
            tensor_record("stubbed_wkv_value", value),
            tensor_record("q_weight", q_weight),
            tensor_record("k_weight", k_weight),
            tensor_record("token_mask", token_mask),
            tensor_record("expected_gate", gate),
            tensor_record("expected_output", output),
            tensor_record("unmasked_zero_dot_gate", unmasked_gate),
            tensor_record("unmasked_zero_dot_output", unmasked_output),
        ],
    }
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
