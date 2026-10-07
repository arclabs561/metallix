#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "torch==2.13.0",
#   "transformers==5.12.1",
# ]
# ///
"""Capture a CPU float32 Qwen2.5 prefill and cached-decode logit reference.

Qwen2.5-0.5B-Instruct declares ``Qwen2ForCausalLM``: Q/K/V projections with a
bias, no Q/K norms, a tied output embedding. For each case this records
the final-position logits of a cached prefill, then of each greedy decode step
fed through the same ``past_key_values``. Full-vocabulary logits are written
as raw little-endian f32 files into ``--output-dir`` (too large to commit);
the printed JSON fixture binds each file by SHA-256 and records the IDs,
rendered prompt, argmax and top-8 a native implementation must reproduce.

This is a checkpoint-parity oracle, not an inference benchmark.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import struct
from pathlib import Path
from typing import Any

import torch
import transformers
from qwen_reference_checkpoint import checkpoint_weight_provenance

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"

READ_FILE_TOOL = {
    "type": "function",
    "function": {
        "name": "read_file",
        "description": "Read a UTF-8 file inside the workspace.",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    },
}

# (name, kind, payload, decode steps). Raw cases bypass the tokenizer; chat
# cases render the checkpoint template exactly as the server must.
CASES: tuple[tuple[str, str, Any, int], ...] = (
    # vocab_size is 151,936, so this exercises the last (tied) embedding row.
    ("raw_last_vocab_row", "raw", [151_935], 4),
    ("raw_ordinary_text", "text", "The quick brown fox jumps over the lazy dog.", 8),
    (
        "chat_user",
        "chat",
        {
            "messages": [
                {
                    "role": "user",
                    "content": "What is the capital of France? Answer in one word.",
                }
            ],
        },
        8,
    ),
    (
        "chat_tool_call",
        "chat",
        {
            "messages": [
                {"role": "system", "content": "You are a coding assistant."},
                {"role": "user", "content": "Show me the contents of README.md."},
            ],
            "tools": [READ_FILE_TOOL],
        },
        24,
    ),
)


def sorted_keys(value: Any) -> Any:
    """Rebuild dicts in sorted key order, as Metallix receives request JSON."""
    return json.loads(json.dumps(value, sort_keys=True))


SORTED_READ_FILE = sorted_keys(READ_FILE_TOOL)

# Render-only cases: the exact prompt a server must build from typed
# messages. Tool and argument dicts are key-sorted because Metallix parses
# request JSON without insertion order.
TEMPLATE_CASES: tuple[tuple[str, dict[str, Any]], ...] = (
    (
        "tools_sorted_with_system",
        {
            "messages": [
                {"role": "system", "content": "You are a coding assistant."},
                {"role": "user", "content": "Show me the contents of README.md."},
            ],
            "tools": [SORTED_READ_FILE],
        },
    ),
    (
        "tools_without_system",
        {
            "messages": [{"role": "user", "content": "List the files."}],
            "tools": [SORTED_READ_FILE],
        },
    ),
    (
        "tool_call_history",
        {
            "messages": [
                {"role": "user", "content": "Read README.md, line 3."},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "type": "function",
                            "function": {
                                "name": "read_file",
                                "arguments": sorted_keys(
                                    {"path": "README.md", "line": 3, "note": "a <b>\nc"}
                                ),
                            },
                        }
                    ],
                },
                {"role": "tool", "content": "# Title"},
                {"role": "user", "content": "thanks"},
            ],
            "tools": [SORTED_READ_FILE],
        },
    ),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_f32_le(path: Path, values: torch.Tensor) -> str:
    """Write a CPU tensor as raw little-endian IEEE-754 f32; return its SHA-256."""
    floats = values.detach().to(torch.float32).cpu().reshape(-1).tolist()
    payload = struct.pack(f"<{len(floats)}f", *floats)
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def step_record(logits: torch.Tensor, path: Path, fed: int | None) -> dict[str, Any]:
    top_logits, top_ids = torch.topk(logits, k=8)
    return {
        "fed_token": fed,
        "logits_file": path.name,
        "logits_sha256": write_f32_le(path, logits),
        "argmax": int(top_ids[0]),
        "top8_ids": [int(value) for value in top_ids.tolist()],
        "top8_logits": [float(value) for value in top_logits.tolist()],
    }


def capture(model_dir: Path, output_dir: Path) -> dict[str, Any]:
    config_path = model_dir / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(
            f"required local model artifact is missing: {config_path}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    weight_provenance = checkpoint_weight_provenance(model_dir, sha256_file)

    # One CPU thread and eager attention keep the oracle reproducible.
    torch.set_num_threads(1)
    torch.set_grad_enabled(False)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_dir, local_files_only=True, trust_remote_code=False
    )
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_dir,
        attn_implementation="eager",
        local_files_only=True,
        dtype=torch.float32,
        trust_remote_code=False,
    )
    model.to(device="cpu", dtype=torch.float32)
    model.eval()
    if type(model).__name__ != "Qwen2ForCausalLM":
        raise RuntimeError(f"expected Qwen2ForCausalLM, loaded {type(model).__name__}")

    cases = []
    for name, kind, payload, steps in CASES:
        rendered = None
        if kind == "raw":
            input_ids = list(payload)
        elif kind == "text":
            rendered = payload
            input_ids = tokenizer.encode(payload, add_special_tokens=False)
        else:
            rendered = tokenizer.apply_chat_template(
                payload["messages"],
                tools=payload.get("tools"),
                add_generation_prompt=True,
                enable_thinking=False,
                tokenize=False,
            )
            input_ids = tokenizer.encode(rendered, add_special_tokens=False)

        records = []
        with torch.inference_mode():
            output = model(
                input_ids=torch.tensor([input_ids], dtype=torch.long),
                use_cache=True,
                return_dict=True,
            )
            logits = output.logits[0, -1]
            records.append(
                step_record(logits, output_dir / f"{name}.prefill.f32le", None)
            )
            cache = output.past_key_values
            for step in range(steps):
                token = int(torch.argmax(logits))
                output = model(
                    input_ids=torch.tensor([[token]], dtype=torch.long),
                    past_key_values=cache,
                    use_cache=True,
                    return_dict=True,
                )
                cache = output.past_key_values
                logits = output.logits[0, -1]
                path = output_dir / f"{name}.decode{step:02}.f32le"
                records.append(step_record(logits, path, token))

        generated = [record["fed_token"] for record in records[1:]] + [
            records[-1]["argmax"]
        ]
        cases.append(
            {
                "name": name,
                "kind": kind,
                **({"rendered": rendered} if rendered is not None else {}),
                "input_ids": input_ids,
                "greedy_tokens": generated,
                "greedy_text": tokenizer.decode(generated, skip_special_tokens=False),
                "steps": records,
            }
        )

    rendered_templates = [
        {
            "name": name,
            **payload,
            "rendered": tokenizer.apply_chat_template(
                payload["messages"],
                tools=payload.get("tools"),
                add_generation_prompt=True,
                enable_thinking=payload.get("enable_thinking", False),
                tokenize=False,
            ),
        }
        for name, payload in TEMPLATE_CASES
    ]

    config = model.config
    return {
        "model_id": MODEL_ID,
        "revision": REVISION,
        "reference": {
            "framework": f"transformers {transformers.__version__}",
            "runtime": f"torch {torch.__version__} CPU float32 eager, one thread",
            "platform": platform.platform(),
            "config_sha256": sha256_file(config_path),
            "tokenizer_sha256": sha256_file(model_dir / "tokenizer.json"),
            "tokenizer_config_sha256": sha256_file(model_dir / "tokenizer_config.json"),
            **weight_provenance,
            "architecture": type(model).__name__,
            "vocab_size": config.vocab_size,
            "hidden_size": config.hidden_size,
            "layer_count": config.num_hidden_layers,
        },
        "cases": cases,
        "template_cases": rendered_templates,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", type=Path, required=True, help="local checkpoint directory"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="directory for the full-vocabulary f32 logit files",
    )
    args = parser.parse_args()
    print(json.dumps(capture(args.model, args.output_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
