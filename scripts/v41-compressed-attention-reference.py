#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["torch==2.13.0"]
# ///
"""Independent CPU oracle for compressed KV rotate/quantize/sparse-attention.

This is arithmetic only, not a source-kernel capture: it pins source hashes as
provenance and uses explicit software RNE assumptions for E2M1/E4M3 staging.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
MODEL = ROOT / "artifacts/v41-reference-model.py"
KERNEL = ROOT / "artifacts/v41-kernel-pinned.py"
MODEL_SHA = "4e9ae23620edc8028ccc5d5fef552ab7fdc7dcd6f79608754fe9f67644056f65"
KERNEL_SHA = "1236c3507019ed176f5dba5e04bcea58867cf654818c6cf138ed4845398c2455"


def bits_bf16(x: torch.Tensor) -> list[int]:
    return [
        int(v) & 0xFFFF for v in x.contiguous().view(torch.int16).reshape(-1).tolist()
    ]


def e4_scale(x: torch.Tensor) -> float:
    # Use Torch's CPU E4M3 cast; normal-code parity is not derivable from a
    # simple value mantissa rule.
    raw = x.float().abs().max().clamp_min(6 * 2**-9) / 6
    return raw.to(torch.float8_e4m3fn).float().item()


def e2_code(v: float) -> float:
    codebook = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
    sign = -1.0 if math.copysign(1.0, v) < 0 else 1.0
    mag = abs(v)
    _, quantized = min(
        enumerate(codebook), key=lambda item: (abs(item[1] - mag), item[0] & 1)
    )
    return sign * quantized


def fp4_reconstruct(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for key in range(x.size(0)):
        scale = e4_scale(x[key])
        normalized = (x[key].float() / torch.tensor(scale, dtype=torch.float32)).clamp(
            -6.0, 6.0
        )
        values = [e2_code(float(v)) for v in normalized.tolist()]
        out[key] = (torch.tensor(values, dtype=torch.float32) * scale).to(
            torch.bfloat16
        )
    return out


def rotate(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    y = x.float().clone()
    tail = y[:, -4:].reshape(-1, 2, 2)
    real, imag = freqs[:, :, 0], freqs[:, :, 1]
    a, b = tail[..., 0].clone(), tail[..., 1].clone()
    tail[..., 0] = a * real - b * imag
    tail[..., 1] = a * imag + b * real
    return y.to(torch.bfloat16)


def sparse(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    sink: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    out = torch.zeros((2, 1, 16), dtype=torch.float32)
    for qi in range(2):
        terms: list[tuple[float, int]] = [(float(sink[0]), -2)]
        for idx in indices[qi].tolist():
            if idx >= 0:
                terms.append(
                    (float((q[qi, 0].double() * kv[idx].double()).sum() * scale), idx)
                )
        maximum = max(score for score, _ in terms)
        weights = [math.exp(score - maximum) for score, _ in terms]
        denom = sum(weights)
        accumulator = torch.zeros(16, dtype=torch.float64)
        for weight, (_, idx) in zip(weights, terms):
            if idx >= 0:
                accumulator += kv[idx].double() * (weight / denom)
        out[qi, 0] = accumulator.float()
    return out


def main() -> None:
    if torch.__version__.split("+")[0] != "2.13.0":
        raise RuntimeError("Torch 2.13.0 required")
    if (
        hashlib.sha256(MODEL.read_bytes()).hexdigest() != MODEL_SHA
        or hashlib.sha256(KERNEL.read_bytes()).hexdigest() != KERNEL_SHA
    ):
        raise RuntimeError("pinned source hash mismatch")
    latents = torch.tensor(
        [
            [
                1.0,
                -2.0,
                3.0,
                -4.0,
                5.0,
                -6.0,
                2.0,
                -1.0,
                4.0,
                -3.0,
                1.0,
                2.0,
                1.0,
                2.0,
                3.0,
                4.0,
            ],
            [
                -2.0,
                1.0,
                -4.0,
                3.0,
                -1.0,
                2.0,
                -3.0,
                4.0,
                5.0,
                1.0,
                -2.0,
                3.0,
                -4.0,
                2.0,
                1.0,
                -3.0,
            ],
        ],
        dtype=torch.bfloat16,
    )
    freqs = torch.tensor(
        [[[0.6, 0.8], [0.8, 0.6]], [[0.8, 0.6], [0.6, 0.8]]], dtype=torch.float32
    )
    query = torch.tensor(
        [
            [
                [
                    1.0,
                    -0.5,
                    2.0,
                    -1.0,
                    0.5,
                    1.0,
                    -2.0,
                    0.25,
                    1.0,
                    -1.0,
                    0.5,
                    2.0,
                    -1.0,
                    1.0,
                    2.0,
                    -2.0,
                ]
            ],
            [
                [
                    -0.5,
                    1.0,
                    -0.25,
                    2.0,
                    1.0,
                    -1.0,
                    0.5,
                    -2.0,
                    1.0,
                    0.5,
                    -1.0,
                    2.0,
                    1.0,
                    -2.0,
                    0.5,
                    1.0,
                ]
            ],
        ],
        dtype=torch.float32,
    )
    indices = torch.tensor([[1, 0, 1, -1], [-1, -1, -1, -1]], dtype=torch.int32)
    sink = torch.tensor([0.35], dtype=torch.float32)
    scale = 0.25
    rotated = rotate(latents, freqs)
    reconstructed = fp4_reconstruct(rotated)
    output = sparse(query, reconstructed, indices, sink, scale)
    wrong_keys = rotate(fp4_reconstruct(latents), freqs)
    wrong = sparse(query, wrong_keys, indices, sink, scale)
    unquantized = sparse(query, rotated.float(), indices, sink, scale)
    if (
        torch.equal(reconstructed, wrong_keys)
        or (output - wrong).abs().max() <= 1.0e-3
        or (output - unquantized).abs().max() <= 1.0e-3
    ):
        raise AssertionError("fixture does not distinguish required ordering")
    payload = {
        "schema_version": 1,
        "source": {
            "revision": "dba1be0a40aa45a94ad051997016db3960a90277",
            "model_sha256": MODEL_SHA,
            "kernel_sha256": KERNEL_SHA,
        },
        "receipt": {"device": "cpu", "torch_version": torch.__version__},
        "scope": {
            "included": "BF16 rotate/narrow, assumed software FP4 G16 reconstruct, FP64 sparse attention",
            "excluded": [
                "source kernel capture",
                "indexer/window/full model",
                "GPU",
                "BF16 attention kernel",
            ],
        },
        "assumptions": {
            "fp4": "software nearest-code ties are a reference assumption",
            "attention_tolerance_abs": 2e-6,
        },
        "latent_bf16": bits_bf16(latents),
        "frequencies_f32": freqs.reshape(-1).tolist(),
        "rotated_bf16": bits_bf16(rotated),
        "reconstructed_bf16": bits_bf16(reconstructed),
        "query_f32": query.reshape(-1).tolist(),
        "indices_i32": indices.reshape(-1).tolist(),
        "sink_f32": sink.tolist(),
        "softmax_scale": scale,
        "output_f32": output.reshape(-1).tolist(),
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
