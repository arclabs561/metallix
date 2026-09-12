#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["torch==2.13.0"]
# ///
"""CPU arithmetic oracle for FP4 activation reconstruction, NOT a kernel capture.

Torch supplies FP32 operations, E4M3 scale casts and BF16 narrowing. E2M1
round-to-nearest ties-to-even is an explicit software assumption implemented
with midpoint intervals. No TileLang or GPU code executes.
"""

import hashlib
import json
import math
from pathlib import Path

import torch

REVISION = "dba1be0a40aa45a94ad051997016db3960a90277"
KERNEL_SHA = "1236c3507019ed176f5dba5e04bcea58867cf654818c6cf138ed4845398c2455"
ROOT = Path(__file__).resolve().parent.parent
LEVELS = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
MIDPOINTS = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)


def e2m1(value: float) -> float:
    magnitude = abs(value)
    code = sum(magnitude > midpoint for midpoint in MIDPOINTS)
    if code < 7 and magnitude == MIDPOINTS[code] and code % 2:
        code += 1
    return math.copysign(LEVELS[code], value)


def words(tensor: torch.Tensor) -> list[int]:
    return [int(word) & 0xFFFF for word in tensor.view(torch.int16).tolist()]


def record(mode: str, name: str, values: list[float]) -> dict:
    group = 16 if mode == "compressed_kv" else 32
    x = torch.tensor((values * group)[:group], dtype=torch.bfloat16)
    promoted = x.float()
    amax = promoted.abs().max()
    if mode == "compressed_kv":
        scale = (amax.clamp_min(6 * 2**-9) / 6).to(torch.float8_e4m3fn).float()
    else:
        scaled = amax.clamp_min(6 * 2**-126) * (1.0 / 6.0)
        scale = torch.tensor(2.0 ** math.ceil(math.log2(scaled.item())))
    if not scale.isfinite() or scale <= 0:
        raise ValueError("oracle case must have a positive finite scale")
    normalized = (promoted / scale).clamp(-6, 6)
    quantized = torch.tensor([e2m1(v) for v in normalized.tolist()])
    result = (quantized * scale).to(torch.bfloat16)
    if not result.isfinite().all():
        raise ValueError("oracle case must reconstruct finite BF16")
    return {
        "name": f"{mode}_{name}",
        "mode": mode,
        "input_bf16": words(x),
        "scale_f32_bits": int(scale.view(torch.int32).item()) & 0xFFFF_FFFF,
        "output_bf16": words(result),
    }


def main() -> None:
    if torch.__version__ != "2.13.0":
        raise RuntimeError("requires exact Torch 2.13.0")
    source = (ROOT / "artifacts/v41-kernel-pinned.py").read_bytes()
    if hashlib.sha256(source).hexdigest() != KERNEL_SHA:
        raise RuntimeError("kernel source identity mismatch")
    torch.set_num_threads(1)
    cases = [
        record(mode, name, values)
        for mode in ("compressed_kv", "index")
        for name, values in (
            ("zero", [0.0, -0.0]),
            ("ties", [6.0, 0.25, -0.25, 0.75, 1.25, -1.75, 2.5, 3.5, 5.0]),
            ("nonpower_scale", [9.0, -9.0, 1.0, -1.0, 3.0, 5.0]),
            ("subnormal", [2**-133, -(2**-133), 0.0, -0.0]),
            ("scale_boundary", [6.375, -6.375, 0.5, 1.0, 2.0, 4.0]),
            ("upper_finite_scale", [2688.0, -2688.0, 224.0, 672.0, 1120.0]),
        )
    ]
    print(
        json.dumps(
            {
                "schema_version": 1,
                "source": {"revision": REVISION, "kernel_sha256": KERNEL_SHA},
                "receipt": {"torch_version": torch.__version__, "device": "cpu"},
                "scope": "independent CPU arithmetic oracle; no upstream kernel execution",
                "rounding": "software E2M1 RNE assumption; Torch CPU E4M3 and BF16 casts",
                "cases": cases,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
