#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "torch==2.13.0",
#   "transformers==5.18.0",
# ]
# ///
"""Capture a deterministic CPU Qwen3.5-family hybrid reference with cached decode.

Dense (``qwen3_5``) and mixture-of-experts (``qwen3_5_moe``) checkpoints are
both accepted; the model class follows the configuration's ``model_type``.

Each case runs one uncached-to-cached prefill and then single-token decode
steps through the source cache, so the reference covers Transformers' chunked
gated-delta prefill and its recurrent decode update (plus the short-convolution
state carried between them). Decode inputs are the reference's own greedy
choices, recorded so a runtime can replay the identical token stream.

Like ``qwen-reference.py`` this takes raw IDs rather than tokenizer text and is
a numerical oracle, not a benchmark.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path
from typing import Any

import torch
import transformers
from qwen_reference_checkpoint import checkpoint_weight_provenance

# Qwen3.5 tokenizer IDs (revision 2fc0636 of Qwen/Qwen3.5-0.8B, no template).
PROSE = (
    760, 81787, 9197, 5728, 8570, 264, 8097, 6948, 1528, 5979, 440, 1754, 3817,
    11, 748, 4779, 25710, 6570, 430, 279, 2193, 26817, 13, 47955, 3983, 916,
    20852, 2250, 1680, 13094, 12, 51319, 13224, 440, 799, 2400, 12, 51319, 6000,
    421, 13205, 449, 18541, 1328, 55616, 6297, 13,
)  # fmt: skip
CODE = (
    8556, 1822, 363, 313, 198, 262, 1042, 2702, 25, 560, 21, 19, 283, 318, 16,
    486, 28, 16, 15, 553, 2117, 21729, 87, 91, 830, 348, 830, 553, 1200, 2061,
    198, 262, 13356, 85871, 4873, 92, 4876, 198, 92, 198,
)  # fmt: skip

CASES: tuple[tuple[str, tuple[int, ...]], ...] = (
    # Shorter than the 4-tap convolution window: decode must use zero history.
    ("length_1_last_vocab_row", (248_319,)),
    # "Hello, world"
    ("length_3_ordinary_text", (9_419, 11, 1_814)),
    # Crosses the 64-token chunk of the source prefill.
    ("length_87_prose_then_code", PROSE + CODE),
    # Three source chunks, the last one partial.
    ("length_141_repeated_prose", PROSE * 3),
)
DEFAULT_DECODE_STEPS = 4


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_f32_le(path: Path, values: torch.Tensor) -> dict[str, int | str]:
    """Write a CPU tensor as raw little-endian IEEE-754 f32."""
    floats = values.detach().to(torch.float32).cpu().reshape(-1).tolist()
    payload = struct.pack(f"<{len(floats)}f", *floats)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return {
        "file": path.name,
        "element_count": len(floats),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def step_record(
    logits: torch.Tensor, sidecar: Path | None, input_ids: list[int]
) -> dict[str, Any]:
    top_logits, top_ids = torch.topk(logits.to(torch.float32), k=8)
    record: dict[str, Any] = {
        "input_ids": input_ids,
        "argmax": int(top_ids[0]),
        "top8": {
            "ids": [int(value) for value in top_ids.tolist()],
            "logits": [float(value) for value in top_logits.tolist()],
        },
    }
    if sidecar is not None:
        record["logits_f32le"] = write_f32_le(sidecar, logits)
    return record


def capture(
    model_dir: Path,
    model_id: str,
    revision: str | None,
    dtype: torch.dtype,
    decode_steps: int,
    logits_dir: Path | None,
    threads: int = 1,
) -> dict[str, Any]:
    config_path = model_dir / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(
            f"required local model artifact is missing: {config_path}"
        )
    weight_provenance = checkpoint_weight_provenance(model_dir, sha256_file)

    # One CPU thread (the default) and eager attention make this reproducible;
    # more threads may reorder reductions. Transformers falls back to its torch
    # gated-delta and convolution paths when the optional fla/causal-conv1d
    # kernels are absent, as they are on CPU.
    torch.set_num_threads(threads)
    torch.set_grad_enabled(False)
    config = transformers.AutoConfig.from_pretrained(
        model_dir, local_files_only=True, trust_remote_code=False
    )
    model_class = {
        "qwen3_5": transformers.Qwen3_5ForConditionalGeneration,
        "qwen3_5_moe": transformers.Qwen3_5MoeForConditionalGeneration,
    }[config.model_type]
    model = model_class.from_pretrained(
        model_dir,
        attn_implementation="eager",
        local_files_only=True,
        dtype=dtype,
        trust_remote_code=False,
    )
    model.to(device="cpu", dtype=dtype)
    model.eval()

    text = config.text_config
    thread_count = "one thread" if threads == 1 else f"{threads} threads"
    cases = []
    for name, ids in CASES:
        steps = []
        with torch.inference_mode():
            output = model(
                input_ids=torch.tensor([list(ids)], dtype=torch.long),
                use_cache=True,
                return_dict=True,
            )
            sidecar = logits_dir / f"{name}.prefill.f32le" if logits_dir else None
            steps.append(step_record(output.logits[0, -1], sidecar, list(ids)))
            cache = output.past_key_values
            for step in range(decode_steps):
                token = steps[-1]["argmax"]
                output = model(
                    input_ids=torch.tensor([[token]], dtype=torch.long),
                    past_key_values=cache,
                    use_cache=True,
                    return_dict=True,
                )
                cache = output.past_key_values
                sidecar = (
                    logits_dir / f"{name}.decode{step}.f32le" if logits_dir else None
                )
                steps.append(step_record(output.logits[0, -1], sidecar, [token]))
        cases.append({"name": name, "steps": steps})

    return {
        "model_id": model_id,
        "revision": revision,
        "reference": {
            "framework": f"transformers {transformers.__version__}",
            "runtime": f"torch {torch.__version__} CPU {dtype} eager, {thread_count}",
            "script": "scripts/qwen35-hybrid-reference.py",
            "config_sha256": sha256_file(config_path),
            **weight_provenance,
            "model_type": config.model_type,
            "vocab_size": text.vocab_size,
            "hidden_size": text.hidden_size,
            "layer_types": list(text.layer_types),
        },
        "decode": "greedy argmax of the previous step, fed one token at a time "
        "through the source cache",
        "cases": cases,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="local checkpoint")
    parser.add_argument(
        "--model-id",
        default="Qwen/Qwen3.5-0.8B",
        help="human-readable source ID recorded in the capture",
    )
    parser.add_argument("--revision", help="immutable Hub revision of --model")
    parser.add_argument(
        "--dtype",
        choices=("float32", "bfloat16"),
        default="float32",
        help="reference precision; bfloat16 bounds memory for 27B checkpoints",
    )
    parser.add_argument("--decode-steps", type=int, default=DEFAULT_DECODE_STEPS)
    parser.add_argument(
        "--threads",
        type=int,
        default=1,
        help="CPU threads; one keeps 0.8B captures bit-reproducible",
    )
    parser.add_argument(
        "--logits-dir",
        type=Path,
        help="optional directory for full-vocabulary logits as raw little-endian f32",
    )
    args = parser.parse_args()
    if args.decode_steps < 0:
        parser.error("--decode-steps must be non-negative")
    if args.threads <= 0:
        parser.error("--threads must be positive")
    print(
        json.dumps(
            capture(
                args.model,
                args.model_id,
                args.revision,
                getattr(torch, args.dtype),
                args.decode_steps,
                args.logits_dir,
                args.threads,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
