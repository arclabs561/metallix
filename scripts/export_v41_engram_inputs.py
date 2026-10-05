# /// script
# requires-python = ">=3.12"
# dependencies = ["torch==2.13.0", "numpy==2.5.3", "sympy==1.14.0", "tokenizers==0.23.2"]
# ///
"""Export the real DeepSeek V4.1 Engram hash inputs from the unmodified pinned source.

The compressed token map, bucket primes, offsets and multipliers are read from the buffers
of the source's own `NgramHashState`, so nothing here re-derives the tokenizer normalization,
prime search or NumPy RNG stream. Rust loads the result with
`deepseek::engram::inputs::EngramHashInputs`.

Wire format (all integers little-endian):
    8 bytes   magic b"MXENGRAM"
    u32       header length in bytes
    header    UTF-8 JSON object (sorted keys)
    payload   token map, one u32 compressed id per raw token id

Regenerate:
    uv run scripts/export_v41_engram_inputs.py --output <path>
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import struct
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
TRACE = ROOT / ".agents/receipts/route-trace"
REVISION = "dba1be0a40aa45a94ad051997016db3960a90277"
ENGRAM_SOURCE_SHA256 = (
    "11f35ecbead8150c35aa002b3d180ef290b05a25afe883a11884f94d476d3897"
)
MAGIC = b"MXENGRAM"
FORMAT = "metallix-v41-engram-inputs"


class ExportError(ValueError):
    pass


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_source(path: Path):
    digest = sha256(path)
    if digest != ENGRAM_SOURCE_SHA256:
        raise ExportError(f"{path} sha256 {digest} is not the pinned Engram source")
    spec = importlib.util.spec_from_file_location("engram", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Tok:
    """The surface the pinned Engram code reads, matching route-trace/record.py."""

    def __init__(self, path: Path):
        from tokenizers import Tokenizer

        self.backend_tokenizer = Tokenizer.from_file(str(path))

    def __len__(self) -> int:
        return self.backend_tokenizer.get_vocab_size(with_added_tokens=True)


def export(source: Path, tokenizer: Path, config: Path) -> bytes:
    engram = load_source(source)
    cfg = json.loads(config.read_text())
    args = SimpleNamespace(**cfg, max_batch_size=1, max_seq_len=1)
    layout = engram.EngramLayout.from_args(args)
    if layout is None:
        raise ExportError("config has no Engram layers")
    state = engram.NgramHashState(args, layout, Tok(tokenizer))

    token_map = state.token_map.tolist()
    layers = len(layout.layer_ids)
    header = {
        "format": FORMAT,
        "schema_version": 1,
        "identity": {
            "revision": REVISION,
            "engram_source_sha256": sha256(source),
            "tokenizer_sha256": sha256(tokenizer),
            "inference_config_sha256": sha256(config),
        },
        "layer_ids": list(layout.layer_ids),
        "num_embeddings": list(layout.num_embeddings),
        "engram_vocab_size": cfg["engram_vocab_size"],
        "max_ngram_size": layout.max_ngram_size,
        "heads": layout.n_heads,
        "raw_pad_id": cfg["engram_pad_id"],
        "compressed_pad_id": int(state.pad_id),
        "compressed_vocab_size": cfg["engram_compressed_vocab_size"],
        "primes": state.primes.reshape(layers, -1).tolist(),
        "offsets": state.offsets.reshape(layers, -1).tolist(),
        "multipliers": state.multipliers.reshape(layers, -1).tolist(),
        "token_map": {"count": len(token_map), "dtype": "u32", "sha256": ""},
    }
    payload = struct.pack(f"<{len(token_map)}I", *token_map)
    header["token_map"]["sha256"] = hashlib.sha256(payload).hexdigest()
    encoded = json.dumps(header, sort_keys=True, separators=(",", ":")).encode()
    return MAGIC + struct.pack("<I", len(encoded)) + encoded + payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--source", type=Path, default=ROOT / "artifacts/v41-engram-pinned.py"
    )
    parser.add_argument(
        "--tokenizer", type=Path, default=TRACE / "tokenizer/tokenizer.json"
    )
    parser.add_argument("--config", type=Path, default=TRACE / "inference-config.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = export(args.source, args.tokenizer, args.config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(data)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        )
    )


if __name__ == "__main__":
    main()
