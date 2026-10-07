# /// script
# requires-python = ">=3.11"
# dependencies = ["gguf==0.19.0", "numpy"]
# ///
"""Writes Q8_0 reference vectors for blockfloat's GGUF decoder test.

Random blocks (fixed seed) plus hand-picked scales, each decoded by gguf-py's
`dequantize`, the Python port of ggml's `dequantize_row_q8_0`.
Usage: uv run scripts/gguf-q8_0-vectors.py > crates/blockfloat/tests/data/gguf-q8_0-vectors.json
"""

import json
import sys

import numpy as np
from gguf import GGMLQuantizationType
from gguf.quants import dequantize

rng = np.random.default_rng(20261006)
scales = [0x3800, 0xC000, 0x0000, 0x8000, 0x0001, 0x03FF, 0x0400, 0x5BFF]
scales += [int(s) for s in rng.integers(0, 0x7BFF, 56)]
blocks = []
for index, scale in enumerate(scales):
    codes = rng.integers(-128, 128, 32, dtype=np.int16).astype(np.int8)
    if index == 0:
        codes[:] = np.arange(-128, 128, 8, dtype=np.int16).astype(np.int8)
    block = np.frombuffer(np.uint16(scale).tobytes() + codes.tobytes(), dtype=np.uint8)
    blocks.append(block)
payload = np.concatenate(blocks)
values = dequantize(payload, GGMLQuantizationType.Q8_0).astype(np.float32)
json.dump(
    {
        "generator": "scripts/gguf-q8_0-vectors.py with gguf-py 0.19.0",
        "payload_hex": payload.tobytes().hex(),
        "values_f32_bits": [int(v) for v in values.view(np.uint32)],
    },
    sys.stdout,
)
sys.stdout.write("\n")
