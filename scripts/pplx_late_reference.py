# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "torch==2.13.0",
#   "transformers==5.12.1",
#   "sentence-transformers==6.1.0",
# ]
# ///
"""Capture a CPU float32 pplx-embed-v1-late-0.6b reference from the card's MultiVectorEncoder.

Runs the pinned revision through sentence-transformers' `MultiVectorEncoder` (the card's
first usage path) with its remote `modeling.py` (sha checked). For each query and document
it records the encoder input ids and attention mask, the scoring mask, and the per-token
128-dimension embeddings, plus MaxSim scores for query/document pairs and the card's
published scores. The recorded features settle three behaviors the native port must copy:
the query-expansion token and whether the encoder attends to it, whether the document
skiplist is applied on token ids, and whether each token vector is L2-normalized.

The tolerance policy below is declared before any native comparison; the Rust opt-in
test (crates/models/qwen/tests/pplx_late_checkpoint.rs) asserts it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import struct
import tempfile
from pathlib import Path

import sentence_transformers
import torch
import transformers
from sentence_transformers import MultiVectorEncoder

ROOT = Path(__file__).resolve().parent.parent
MODEL_ID = "perplexity-ai/pplx-embed-v1-late-0.6b"
REVISION = "4d28cf627d225552cfc29fb7df6cb0705ea0f1b3"
# From the Hub API at REVISION (LFS sha256) and the small files fetched at REVISION.
PINNED_SHA256 = {
    "model.safetensors": (
        "9bb17c37a5d377f1ca8bc30e784fe285d0886b942d5f1e4859be635469e6c6b6"
    ),
    "1_Dense/model.safetensors": (
        "9df08b0a94db0438ebea9a3547d33b6d0871a358de5e5a2071a0fbabd5c34988"
    ),
    "tokenizer.json": (
        "971941ca0ba0be4b0353fb079996a35cad610bea52dffab110551436e30b75e5"
    ),
}
# The card's printed MaxSim scores for CARD_QUERY against CARD_DOCUMENTS.
CARD_SCORES = [31.4841, 31.2462, 31.4041]
TOLERANCE_POLICY = {
    "token_ids": "query ids (with expansion) and document ids equal this reference",
    "scored_positions": "the set of scored token positions equals this reference",
    "token_cosine_min": 1 - 1e-5,
    "maxsim_abs_vs_reference": 1e-3,
    "maxsim_abs_vs_card": 1e-2,
    "native_precision": "checkpoint float32 weights",
    "rationale": (
        "Declared before the native run. Native and this oracle are both float32 "
        "and differ only in reduction order, so each normalized 128-dimension token "
        "vector must agree closely, and a MaxSim score (a sum of up to 32 cosines, "
        "about 30) within 1e-3. The card prints scores to four decimals with "
        "unstated hardware, so its anchor uses a looser bound."
    ),
}
CARD_QUERY = "What motivates scientific discovery?"
CARD_DOCUMENTS = [
    "Scientists explore the universe driven by curiosity.",
    "Children learn through curious exploration.",
    "Historical discoveries began with curious questions.",
]
LONG_DOCUMENT = (
    "Glaciers carve valleys into a U shape, while rivers cut narrower V-shaped "
    "valleys; both leave sediment that farmers later rely on. "
) * 40
QUERIES = [
    ("card_query", CARD_QUERY),
    ("punctuated_query", "Who wrote 'Hamlet', and when (roughly)?"),
    (
        "long_query",
        (
            "Explain in detail how photosynthesis converts light, water and carbon "
            "dioxide into sugar and oxygen, including the role of chlorophyll, the "
            "light-dependent reactions, the Calvin cycle and the stomata in leaves."
        ),
    ),
    ("multilingual_query", "東京は日本の首都ですか？"),
]
DOCUMENTS = [
    ("card_document_0", CARD_DOCUMENTS[0]),
    ("card_document_1", CARD_DOCUMENTS[1]),
    ("card_document_2", CARD_DOCUMENTS[2]),
    (
        "punctuated_document",
        "Hamlet (c. 1600) was written by William Shakespeare; it's his longest play!",
    ),
    ("code_document", "fn add(a: i32, b: i32) -> i32 { a + b } // returns a+b."),
    ("multilingual_document", "東京は日本の首都で、世界有数の大都市です。"),
    ("long_document", LONG_DOCUMENT),
]
PAIRS = [
    ("card_query", "card_document_0"),
    ("card_query", "card_document_1"),
    ("card_query", "card_document_2"),
    ("punctuated_query", "punctuated_document"),
    ("punctuated_query", "code_document"),
    ("long_query", "long_document"),
    ("multilingual_query", "multilingual_document"),
    ("multilingual_query", "card_document_0"),
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stage(model_dir: Path) -> Path:
    """Copy the small files into a real directory and link the weights.

    The dynamic-module loader resolves `modeling.py` through the cache symlink into
    the blob store, where its sibling `configuration.py` is not found.
    """
    staged = Path(tempfile.mkdtemp(prefix="pplx-late-"))
    for path in model_dir.rglob("*"):
        target = staged / path.relative_to(model_dir)
        if path.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        elif path.suffix == ".safetensors":
            target.parent.mkdir(parents=True, exist_ok=True)
            target.symlink_to(path.resolve())
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
    return staged


def f32le_hex(values: torch.Tensor) -> str:
    flat = values.reshape(-1).tolist()
    return struct.pack(f"<{len(flat)}f", *flat).hex()


def record(model: MultiVectorEncoder, name: str, text: str, task: str) -> dict:
    """Encoder inputs, scoring mask and scored token vectors for one text."""
    encode = model.encode_query if task == "query" else model.encode_document
    features = encode([text], output_value=None, convert_to_numpy=False)[0]
    prompt = model.prompts[task]
    encoder = model.preprocess([text], prompt=prompt, task=task)
    scored = features["attention_mask"].bool()
    embeddings = features["token_embeddings"][scored].float()
    norms = embeddings.norm(dim=-1)
    return {
        "name": name,
        "task": task,
        "text": text,
        "input_ids": encoder["input_ids"][0].tolist(),
        "encoder_attention_mask": encoder["attention_mask"][0].tolist(),
        "scored_positions": scored.nonzero().flatten().tolist(),
        "token_norm_range": [float(norms.min()), float(norms.max())],
        "embeddings_f32le": f32le_hex(embeddings),
        "embedding": embeddings,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "fixtures/pplx-embed-v1-late-0.6b/late-reference.json",
    )
    args = parser.parse_args()
    model_dir = args.model_dir
    if model_dir is None:
        from huggingface_hub import snapshot_download

        model_dir = Path(
            snapshot_download(
                MODEL_ID,
                revision=REVISION,
                allow_patterns=["*.json", "*.py", "*.txt", "*.safetensors"],
            )
        )
    files_sha256 = {
        name: sha256_file(model_dir / name)
        for name in (*PINNED_SHA256, "config.json", "modeling.py", "configuration.py")
    }
    for name, expected in PINNED_SHA256.items():
        if files_sha256[name] != expected:
            raise SystemExit(f"{name} sha256 {files_sha256[name]} != {expected}")
    model_dir = stage(model_dir)

    torch.set_num_threads(1)
    torch.set_grad_enabled(False)
    model = MultiVectorEncoder(
        str(model_dir),
        trust_remote_code=True,
        local_files_only=True,
        device="cpu",
        model_kwargs={"attn_implementation": "eager", "dtype": torch.float32},
    ).eval()
    modules = [type(module).__name__ for module in model]

    records = [record(model, name, text, "query") for name, text in QUERIES]
    records += [record(model, name, text, "document") for name, text in DOCUMENTS]
    by_name = {item["name"]: item for item in records}
    pair_scores = [
        {
            "query": query,
            "document": document,
            "score": float(
                model.similarity(
                    [by_name[query]["embedding"]], [by_name[document]["embedding"]]
                )[0][0]
            ),
        }
        for query, document in PAIRS
    ]
    card_query = model.encode_query(CARD_QUERY, convert_to_numpy=False)
    card_documents = model.encode_document(CARD_DOCUMENTS, convert_to_numpy=False)
    card_path_scores = model.similarity([card_query], card_documents)[0].tolist()
    for item in records:
        del item["embedding"]

    mask_token_id = model.tokenizer.mask_token_id
    skiplist_ids = sorted(
        {
            model.tokenizer.convert_tokens_to_ids(word)
            for word in model[-2].skiplist_words
        }
    )
    fixture = {
        "schema_version": 1,
        "model_id": MODEL_ID,
        "revision": REVISION,
        "files_sha256": files_sha256,
        "reference": {
            "framework": (
                f"sentence-transformers {sentence_transformers.__version__}, "
                f"transformers {transformers.__version__}"
            ),
            "runtime": f"torch {torch.__version__} CPU float32 eager, one thread",
            "platform": platform.platform(),
            "path": "MultiVectorEncoder.encode_query / encode_document",
            "modules": modules,
            "prompts": model.prompts,
            "mask_token_id": mask_token_id,
            "skiplist_token_ids": skiplist_ids,
        },
        "tolerance_policy": TOLERANCE_POLICY,
        "inputs": records,
        "pair_scores": pair_scores,
        "card_scores": CARD_SCORES,
        "observed": {
            "card_path_scores": card_path_scores,
            "card_scores_max_abs": max(
                abs(a - b) for a, b in zip(card_path_scores, CARD_SCORES)
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(fixture, indent=1) + "\n")
    print(
        json.dumps(
            {"output": str(args.output), "modules": modules, **fixture["observed"]}
        )
    )


if __name__ == "__main__":
    main()
