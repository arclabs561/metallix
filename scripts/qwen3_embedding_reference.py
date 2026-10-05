# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "torch==2.13.0",
#   "transformers==5.12.1",
# ]
# ///
"""Capture a CPU float32 Qwen3-Embedding-0.6B reference from the model card's recipe.

Each input is tokenized with the pinned tokenizer's special-token template (which
appends <|endoftext|>), run alone through `AutoModel` in float32 eager mode on one
CPU thread, pooled at its last position and L2-normalized, as in the card's
Transformers snippet. The card's four retrieval inputs are also run as the card
does (one left-padded batch) to check that batching does not change the result.

The tolerance policy below is declared before any comparison is run; the Rust
opt-in test (crates/models/qwen/tests/embedding_checkpoint.rs) asserts it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import torch
import torch.nn.functional as F
import transformers

ROOT = Path(__file__).resolve().parent.parent
MODEL_ID = "Qwen/Qwen3-Embedding-0.6B"
REVISION = "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"
PINNED_SHA256 = {
    "config.json": "b5bf1f51fc45be473a54718cef92448d90a1be001bf9b9a44b8c7f10a19feaa9",
    "tokenizer.json": "def76fb086971c7867b829c23a26261e38d9d74e02139253b38aeb9df8b4b50a",
    "model.safetensors": (
        "0437e45c94563b09e13cb7a64478fc406947a93cb34a7e05870fc8dcd48e23fd"
    ),
}
TASK = "Given a web search query, retrieve relevant passages that answer the query"
# The card's published Transformers scores: queries (rows) by documents (columns).
CARD_SCORES = [
    [0.7645568251609802, 0.14142508804798126],
    [0.13549736142158508, 0.5999549627304077],
]
TOLERANCE_POLICY = {
    "token_ids": "exact equality with this reference",
    "embedding_cosine_min": 1 - 1e-5,
    "score_abs_vs_reference": 1e-4,
    "score_abs_vs_card": 1e-3,
    "native_precision": "weights promoted to float32 (prepare_float32)",
    "rationale": (
        "Declared before running. Native f32 and this f32 oracle differ only in "
        "reduction order, so per-vector cosine and pairwise scores must agree "
        "closely. The card's precision and hardware are unstated (its own vLLM "
        "numbers differ from its Transformers numbers by about 2.5e-3), so the "
        "card anchor uses a looser bound and checks recipe, not arithmetic."
    ),
}
LONG_SENTENCE = (
    "The river carried silt from the mountains to the delta, where farmers "
    "planted rice in the rich soil every spring. "
)
MAX_TOKENS = 512  # the native uncached path's limit


def query(text: str) -> str:
    return f"Instruct: {TASK}\nQuery:{text}"


INPUTS = [
    ("card_query_0", query("What is the capital of China?")),
    ("card_query_1", query("Explain gravity")),
    ("card_document_0", "The capital of China is Beijing."),
    (
        "card_document_1",
        (
            "Gravity is a force that attracts two bodies towards each other. It gives "
            "weight to physical objects and is responsible for the movement of planets "
            "around the sun."
        ),
    ),
    ("single_word", "hello"),
    ("non_ascii", "北京是中国的首都。Café naïve résumé — 東京"),
    ("long", LONG_SENTENCE * 17),
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_model_dir(model_dir: Path | None) -> Path:
    if model_dir is None:
        from huggingface_hub import snapshot_download

        model_dir = Path(snapshot_download(MODEL_ID, revision=REVISION))
    for name, expected in PINNED_SHA256.items():
        actual = sha256_file(model_dir / name)
        if actual != expected:
            raise SystemExit(f"{name} sha256 {actual} is not the pinned {expected}")
    return model_dir


def last_token_pool(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """The card's pooling: the last position, or the last unpadded one."""
    if bool(mask[:, -1].sum() == mask.shape[0]):
        return hidden[:, -1]
    lengths = mask.sum(dim=1) - 1
    return hidden[torch.arange(hidden.shape[0]), lengths]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "fixtures/qwen3-embedding-0.6b/embedding-reference.json",
    )
    args = parser.parse_args()
    model_dir = resolve_model_dir(args.model_dir)

    torch.set_num_threads(1)
    torch.set_grad_enabled(False)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_dir, local_files_only=True, padding_side="left"
    )
    model = transformers.AutoModel.from_pretrained(
        model_dir,
        attn_implementation="eager",
        local_files_only=True,
        dtype=torch.float32,
        trust_remote_code=False,
    )
    model.to(device="cpu", dtype=torch.float32)
    model.eval()

    records, embeddings = [], {}
    for name, text in INPUTS:
        batch = tokenizer([text], return_tensors="pt")
        ids = batch["input_ids"][0].tolist()
        if ids[-1] != tokenizer.pad_token_id or len(ids) > MAX_TOKENS:
            raise SystemExit(f"{name}: unexpected template or length ({len(ids)})")
        with torch.inference_mode():
            hidden = model(**batch).last_hidden_state
        embedding = F.normalize(last_token_pool(hidden, batch["attention_mask"]), dim=1)
        embeddings[name] = embedding[0]
        records.append(
            {
                "name": name,
                "text": text,
                "input_ids": ids,
                "embedding": [float(v) for v in embedding[0].tolist()],
            }
        )

    queries = torch.stack([embeddings["card_query_0"], embeddings["card_query_1"]])
    documents = torch.stack(
        [embeddings["card_document_0"], embeddings["card_document_1"]]
    )
    scores = (queries @ documents.T).tolist()

    # The card's own batched, left-padded run of the same four inputs.
    card_texts = [text for name, text in INPUTS if name.startswith("card_")]
    batch = tokenizer(card_texts, padding=True, return_tensors="pt")
    with torch.inference_mode():
        hidden = model(**batch).last_hidden_state
    batched = F.normalize(last_token_pool(hidden, batch["attention_mask"]), dim=1)
    batched_scores = (batched[:2] @ batched[2:].T).tolist()

    def max_diff(left: list[list[float]], right: list[list[float]]) -> float:
        return max(abs(a - b) for la, lb in zip(left, right) for a, b in zip(la, lb))

    fixture = {
        "schema_version": 1,
        "model_id": MODEL_ID,
        "revision": REVISION,
        "files_sha256": PINNED_SHA256,
        "reference": {
            "framework": f"transformers {transformers.__version__}",
            "runtime": f"torch {torch.__version__} CPU float32 eager, one thread",
            "platform": platform.platform(),
            "pooling": "last token (appended <|endoftext|>), then L2 normalize",
            "task": TASK,
        },
        "tolerance_policy": TOLERANCE_POLICY,
        "inputs": records,
        "scores": scores,
        "card_scores": CARD_SCORES,
        "observed": {
            "scores_max_abs_vs_card": max_diff(scores, CARD_SCORES),
            "batched_scores_max_abs_vs_single": max_diff(batched_scores, scores),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(fixture, indent=1) + "\n")
    print(json.dumps({"output": str(args.output), **fixture["observed"]}))


if __name__ == "__main__":
    main()
