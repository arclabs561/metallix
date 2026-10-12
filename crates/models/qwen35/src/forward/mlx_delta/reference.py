"""Regenerate recurrence fixtures with official mlx-lm 0.32.0 / MLX 0.32.3."""

import hashlib
import json
import os
from importlib.metadata import version
from pathlib import Path

import mlx.core as mx
from mlx_lm.models.gated_delta import gated_delta_kernel, gated_delta_ops

assert version("mlx-lm") == "0.32.0"
assert version("mlx") == "0.32.3"
assert os.environ.get("MLX_GDN_PACKED", "1") != "0"
mx.set_cache_limit(64 << 20)


def values(shape, salt):
    count = 1
    for dim in shape:
        count *= dim
    return mx.array([((i * 17 + salt) % 101 - 50) / 256 for i in range(count)]).reshape(
        shape
    )


def digest(array):
    mx.eval(array)
    return hashlib.sha256(memoryview(array).tobytes()).hexdigest()


cases = []
for dk, dv in [(128, 8), (64, 7), (16, 7), (33, 7)]:
    state = values((1, 2, dv, dk), 29)
    records = []
    for seq, salt in [(3, 0), (1, 37)]:
        q = values((1, seq, 1, dk), salt + 1).astype(mx.bfloat16)
        k = values((1, seq, 1, dk), salt + 3).astype(mx.bfloat16)
        v = values((1, seq, 2, dv), salt + 5).astype(mx.bfloat16)
        g = values((1, seq, 2), salt + 7) + 0.75
        beta = (values((1, seq, 2), salt + 11) + 0.5).astype(mx.bfloat16)
        fn = gated_delta_ops if dk < 32 or dk % 32 else gated_delta_kernel
        y, state = fn(q, k, v, g, beta, state)
        records.append({"output": digest(y.view(mx.uint16)), "state": digest(state)})
        assert mx.get_peak_memory() < 256 << 20
    cases.append({"dk": dk, "dv": dv, "steps": records})
Path(__file__).with_name("reference.json").write_text(
    json.dumps(
        {
            "mlx": version("mlx"),
            "mlx_lm": version("mlx-lm"),
            "packed_enabled": os.environ.get("MLX_GDN_PACKED", "1"),
            "source_sha256": hashlib.sha256(
                Path(
                    __import__("mlx_lm.models.gated_delta", fromlist=["x"]).__file__
                ).read_bytes()
            ).hexdigest(),
            "cases": cases,
        },
        indent=2,
    )
    + "\n"
)
