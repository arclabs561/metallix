# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "torch==2.13.0",
#   "transformers==5.12.1",
#   "sentence-transformers==6.1.0",
# ]
# ///
"""Capture a CPU float32 pplx-embed-context-v1-0.6b reference from the card's encode path.

Runs the pinned revision's own `PPLXQwen3ContextualModel.encode` (remote code, sha
checked) on a few multi-chunk documents: chunks joined with <|endoftext|> into one
sequence, bidirectional attention, mean pooling of each span between separators
after the final norm, then int8 tanh quantization (and binary). The float pooled
vectors before quantization are captured through the same model's own extraction
method. A second path runs stock transformers `Qwen3Model` with causality disabled
and checks it agrees, so the reference does not rest on the remote mask code alone.

The tolerance policy below is declared before any comparison is run; the Rust
opt-in test (crates/models/qwen/tests/pplx_context_checkpoint.rs) asserts it.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import platform
import shutil
import struct
import sys
import tempfile
from pathlib import Path

import torch
import transformers

ROOT = Path(__file__).resolve().parent.parent
MODEL_ID = "perplexity-ai/pplx-embed-context-v1-0.6b"
REVISION = "b42df969d4d78d1840769e45c68c4e4ce763768b"
# From the Hub API at REVISION (LFS sha256) and the files fetched at REVISION.
PINNED_SHA256 = {
    "model.safetensors": (
        "dba2f2e1818c61012c053468480a54d52bfae47641e8dfcbcf02715e763a4486"
    ),
    "tokenizer.json": (
        "c6fb5c5bbba5fa5f8332edfb6d8aa67bd7fb3d75365b1765f108201698eaebf5"
    ),
    "config.json": "f73a7efb2b3b914bd6c2bf5816408dd8de62fae0b01325e852c63e89057db6a9",
    "modeling.py": "a3bd384bf515fa0f806cee27f4676d4677597d5fa1877b9a9d2dc1f4116688d9",
    "configuration.py": (
        "e914aa73614c91703364c77917313da0821ea04a14d73a7836b5cb40abab1671"
    ),
    "st_quantize.py": (
        "11ac8a5e595d9da8dcd5cbbafbb18d0e756ce0d353be25fadb7e4c8471770ede"
    ),
}
SEP_ID = 151643
MAX_TOKENS = 512  # the native uncached path's limit
TOLERANCE_POLICY = {
    "token_ids": "exact equality with this reference",
    "pooled_cosine_min": 1 - 1e-5,
    "int8_max_abs_code_diff": 1,
    "binary": "equal except where |reference pooled value| <= 1e-4",
    "native_precision": "checkpoint float32 weights",
    "rationale": (
        "Declared before running. Native and this oracle are both float32 and "
        "differ only in reduction order, so pooled vectors must agree closely; "
        "int8 codes can still differ by one where tanh(x) * 127 sits on a "
        "rounding boundary, and binary signs only where a value is near zero."
    ),
}
DOCUMENTS = [
    (
        "card_curiosity",
        [
            "Curiosity begins in childhood with endless questions about the world.",
            "As we grow, curiosity drives us to explore new ideas.",
            "Scientific breakthroughs often start with a curious question.",
        ],
    ),
    (
        "card_mars",
        [
            "The curiosity rover explores Mars searching for ancient life.",
            "Each discovery on Mars sparks new questions about the universe.",
        ],
    ),
    (
        "pronoun_context",
        [
            "Marie Curie was born in Warsaw in 1867.",
            "She later moved to Paris to study physics and mathematics.",
            "Her work on radioactivity earned her two Nobel Prizes.",
            "It was the first time anyone had won in two sciences.",
        ],
    ),
    ("empty_last_chunk", ["The meeting starts at nine.", "Bring the report.", ""]),
    ("single_chunk", ["A single chunk document with no separator at all."]),
    (
        "multilingual",
        [
            "東京は日本の首都です。",
            "Paris est la capitale de la France.",
            "Berlin ist die Hauptstadt von Deutschland.",
        ],
    ),
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_pinned_package(model_dir: Path):
    """Import the revision's modeling code as a local package.

    The files are sha-checked above. Importing them directly avoids the
    dynamic-module loader, which resolves cache symlinks into the blob store and
    can load the config class twice.
    """
    root = Path(tempfile.mkdtemp(prefix="pplx-context-"))
    package = root / "pplx_pinned"
    package.mkdir()
    (package / "__init__.py").write_text("")
    for name in ("modeling.py", "configuration.py", "st_quantize.py"):
        shutil.copyfile(model_dir / name, package / name)
    sys.path.insert(0, str(root))
    return importlib.import_module("pplx_pinned.modeling")


def codes(values: torch.Tensor) -> list[list[int]]:
    return [[int(v) for v in row] for row in values.round().to(torch.int64)]


def f32le_hex(values: torch.Tensor) -> str:
    return struct.pack(f"<{values.numel()}f", *values.reshape(-1).tolist()).hex()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "fixtures/pplx-embed-context-v1-0.6b/context-reference.json",
    )
    args = parser.parse_args()
    model_dir = args.model_dir
    if model_dir is None:
        from huggingface_hub import snapshot_download

        model_dir = Path(
            snapshot_download(
                MODEL_ID,
                revision=REVISION,
                allow_patterns=["*.json", "*.py", "*.txt", "model.safetensors"],
            )
        )
    files_sha256 = {name: sha256_file(model_dir / name) for name in PINNED_SHA256}
    for name, expected in PINNED_SHA256.items():
        if files_sha256[name] != expected:
            raise SystemExit(f"{name} sha256 {files_sha256[name]} != {expected}")
    modeling = load_pinned_package(model_dir)

    torch.set_num_threads(1)
    torch.set_grad_enabled(False)
    model = modeling.PPLXQwen3ContextualModel.from_pretrained(
        model_dir,
        local_files_only=True,
        attn_implementation="eager",
        dtype=torch.float32,
    ).eval()
    tokenizer = model.tokenizer
    if tokenizer.sep_token_id != SEP_ID:
        raise SystemExit(f"unexpected SEP id {tokenizer.sep_token_id}")
    # Eager attention always builds a causal mask; SDPA with no mask honors
    # `is_causal = False`, which makes this an independent bidirectional path.
    stock = transformers.Qwen3Model.from_pretrained(
        model_dir,
        local_files_only=True,
        attn_implementation="sdpa",
        dtype=torch.float32,
    ).eval()
    for layer in stock.layers:
        layer.self_attn.is_causal = False

    records, stock_max_abs, code_offset = [], 0.0, 0.0
    for name, chunks in DOCUMENTS:
        int8 = torch.as_tensor(model.encode([chunks], quantization="int8")[0])
        binary = torch.as_tensor(model.encode([chunks], quantization="binary")[0])
        # The quantizers return hard + soft - soft.detach(): floats within
        # rounding error of the integer codes, so round rather than truncate.
        for output in (int8, binary):
            code_offset = max(code_offset, float((output - output.round()).abs().max()))
        inputs = tokenizer([tokenizer.sep_token.join(chunks)], return_tensors="pt")
        ids = inputs["input_ids"][0].tolist()
        if len(ids) > MAX_TOKENS:
            raise SystemExit(f"{name}: {len(ids)} tokens exceeds {MAX_TOKENS}")
        hidden = model(**inputs).last_hidden_state
        pooled = torch.stack(
            model._extract_chunks_from_concatenated(
                inputs["input_ids"], hidden, inputs["attention_mask"]
            )[0]
        )
        # Stock Qwen3Model, causality off, no mask: one unpadded sequence.
        stock_hidden = stock(
            input_ids=inputs["input_ids"],
            attention_mask=None,
        ).last_hidden_state
        stock_max_abs = max(stock_max_abs, float((stock_hidden - hidden).abs().max()))
        records.append(
            {
                "name": name,
                "chunks": chunks,
                "input_ids": ids,
                "pooled_f32le": [f32le_hex(row) for row in pooled],
                "int8": codes(int8),
                "binary": codes(binary),
            }
        )

    fixture = {
        "schema_version": 1,
        "model_id": MODEL_ID,
        "revision": REVISION,
        "files_sha256": files_sha256,
        "reference": {
            "framework": f"transformers {transformers.__version__}",
            "runtime": f"torch {torch.__version__} CPU float32 eager, one thread",
            "platform": platform.platform(),
            "path": "pinned modeling.py PPLXQwen3ContextualModel.encode",
            "separator_id": SEP_ID,
            "pooling": "mean over each span between separators, after final norm",
            "quantization": "int8: clamp(round(tanh(x) * 127), -128, 127); binary: sign",
        },
        "tolerance_policy": TOLERANCE_POLICY,
        "documents": records,
        "observed": {
            "stock_qwen3_noncausal_max_abs_hidden": stock_max_abs,
            "encode_output_max_offset_from_integer_codes": code_offset,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(fixture, indent=1) + "\n")
    print(json.dumps({"output": str(args.output), **fixture["observed"]}))


if __name__ == "__main__":
    main()
