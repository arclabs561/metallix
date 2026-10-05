#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "torch==2.13.0",
#   "transformers==5.18.0",
#   "jinja2==3.1.6",
# ]
# ///
"""Capture a deterministic CPU Gemma 4 text reference: prefill and cached decode.

Each case prefills a fixed prompt with the source KV cache, then greedily
decodes a few tokens one at a time, and writes every step's full-vocabulary
logits as raw little-endian f32. One case is longer than the 1024-token
sliding window, so the window mask and the windowed cache are exercised in
both prefill and decode. The template cases record the source chat-template
rendering and token IDs for tokenizer and template parity.

A bfloat16 capture with ``--decode-from`` feeds the earlier capture's decode
tokens and records, per step, its distance from that capture: the source's
own bfloat16 rounding noise against its float32 output. ``--fixture`` then
writes the float32 capture, with that per-step distance and without the logit
sidecars' contents, as the compact JSON fixture the Metallix tests read.

This is a numerical oracle for Metallix, not an inference benchmark.
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

DECODE_STEPS = 6
# A float32 copy doubles the stored bf16 bytes. Above this many stored bytes
# (12B is about 24 GB with its vision and audio towers) the float32 oracle no
# longer fits beside other work on a 128 GB Mac; use bfloat16 instead.
MAX_FLOAT32_STORED_BYTES = 32 * 1024**3
WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name."},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            },
            "required": ["city"],
        },
    },
}

# Ordinary prose that varies per sentence, so a long prompt is not one
# repeated n-gram that the model can predict trivially.
LONG_SENTENCE = (
    "Entry {n}: the survey team walked {d} kilometres north of camp {c}, "
    "logged the river depth at {r} centimetres, and noted that the {w} "
    "weather held until evening."
)
WEATHER = ("clear", "windy", "humid", "cold", "overcast", "dry", "foggy")

TEMPLATE_CASES: tuple[tuple[str, dict[str, Any]], ...] = (
    (
        "user_only",
        {"messages": [{"role": "user", "content": "Name three primary colors."}]},
    ),
    (
        "system_user_assistant_user",
        {
            "messages": [
                {"role": "system", "content": "You answer in one short sentence."},
                {"role": "user", "content": "What is 2 + 2?"},
                {"role": "assistant", "content": "2 + 2 is 4."},
                {"role": "user", "content": "And 3 + 3?"},
            ]
        },
    ),
    (
        "unicode_user",
        {"messages": [{"role": "user", "content": "Traduis « bonjour » en 日本語."}]},
    ),
    (
        "thinking_enabled",
        {
            "messages": [{"role": "user", "content": "Is 91 prime?"}],
            "enable_thinking": True,
        },
    ),
    (
        "tool_declaration",
        {
            "messages": [{"role": "user", "content": "Weather in Paris?"}],
            "tools": [WEATHER_TOOL],
        },
    ),
    (
        "tool_call_and_response",
        {
            "messages": [
                {"role": "user", "content": "Weather in Paris?"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": {"city": "Paris", "unit": "celsius"},
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_1",
                    "content": '{"temperature": 18, "sky": "clear"}',
                },
            ],
            "tools": [WEATHER_TOOL],
        },
    ),
)


def long_prompt_text(sentences: int) -> str:
    return (
        " ".join(
            LONG_SENTENCE.format(
                n=index + 1,
                d=(index * 7) % 23 + 2,
                c=chr(ord("A") + index % 26),
                r=(index * 13) % 90 + 10,
                w=WEATHER[index % len(WEATHER)],
            )
            for index in range(sentences)
        )
        + " Summarize the expedition in one sentence:"
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_f32_le(path: Path, values: torch.Tensor) -> dict[str, int | str]:
    """Write a CPU tensor as raw little-endian IEEE-754 f32."""
    floats = values.detach().to(device="cpu", dtype=torch.float32).reshape(-1)
    payload = struct.pack(f"<{floats.numel()}f", *floats.tolist())
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return {
        "file": path.name,
        "element_count": floats.numel(),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def top8(logits: torch.Tensor) -> dict[str, list[Any]]:
    values, ids = torch.topk(logits.to(torch.float32), k=8)
    return {"ids": [int(i) for i in ids], "logits": [float(v) for v in values]}


def read_f32_le(path: Path) -> torch.Tensor:
    payload = path.read_bytes()
    return torch.tensor(struct.unpack(f"<{len(payload) // 4}f", payload))


def weight_files(model_dir: Path) -> list[dict[str, str]]:
    return [
        {"file": path.name, "sha256": sha256_file(path)}
        for path in sorted(model_dir.glob("*.safetensors"))
    ]


def capture_case(
    model: torch.nn.Module,
    name: str,
    input_ids: list[int],
    output_dir: Path,
    forced: list[dict[str, Any]] | None = None,
    forced_dir: Path | None = None,
) -> dict[str, Any]:
    """``forced`` holds an earlier capture's steps for this case: their decode
    tokens are fed instead of greedy ones and each step is compared with them."""

    def record(input_ids: list[int], file: str, logits: torch.Tensor, index: int):
        step = {
            "input_ids": input_ids,
            "logits_f32le": write_f32_le(output_dir / file, logits),
            "argmax": int(torch.argmax(logits)),
            "top8": top8(logits),
        }
        if forced is not None and forced_dir is not None:
            earlier = read_f32_le(forced_dir / forced[index]["logits_f32le"]["file"])
            step["versus_decode_from"] = {
                "max_abs": float((logits.to(torch.float32) - earlier).abs().max()),
                "argmax_match": step["argmax"] == forced[index]["argmax"],
            }
        return step

    tokens = torch.tensor([input_ids], dtype=torch.long)
    cache = transformers.DynamicCache(config=model.config.get_text_config())
    steps = []
    with torch.inference_mode():
        output = model(input_ids=tokens, past_key_values=cache, use_cache=True)
        logits = output.logits[0, -1]
        steps.append(record(input_ids, f"{name}.prefill.f32", logits, 0))
        for step in range(DECODE_STEPS):
            # Greedy feedback keeps the decode inputs a pure function of the
            # checkpoint and prompt; forced tokens align a second capture with
            # an earlier one.
            token = (
                forced[step + 1]["input_ids"][0]
                if forced
                else int(torch.argmax(logits))
            )
            output = model(
                input_ids=torch.tensor([[token]], dtype=torch.long),
                past_key_values=output.past_key_values,
                use_cache=True,
            )
            logits = output.logits[0, -1]
            steps.append(
                record([token], f"{name}.decode{step + 1}.f32", logits, step + 1)
            )
    return {"name": name, "prompt_tokens": len(input_ids), "steps": steps}


def capture(
    model_dir: Path,
    model_id: str,
    revision: str,
    output_dir: Path,
    threads: int,
    dtype: str,
    decode_from: Path | None = None,
) -> dict[str, Any]:
    config_path = model_dir / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(
            f"required local model artifact is missing: {config_path}"
        )
    stored_bytes = sum(path.stat().st_size for path in model_dir.glob("*.safetensors"))
    if dtype == "float32" and stored_bytes > MAX_FLOAT32_STORED_BYTES:
        raise SystemExit(
            f"{stored_bytes / 1024**3:.0f} GiB of stored weights would need about "
            f"{2 * stored_bytes / 1024**3:.0f} GiB in float32; capture bfloat16 instead"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    # Eager attention and float32 make this a correctness oracle; a bfloat16
    # capture measures the source's own rounding noise. The thread count is
    # recorded because it can change reduction order.
    torch_dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16}[dtype]
    torch.set_num_threads(threads)
    torch.set_grad_enabled(False)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_dir, local_files_only=True, trust_remote_code=False
    )
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_dir,
        attn_implementation="eager",
        local_files_only=True,
        dtype=torch_dtype,
        trust_remote_code=False,
    )
    model.to(device="cpu", dtype=torch_dtype)
    model.eval()
    text_config = model.config.get_text_config()

    def chat(content: str) -> list[int]:
        return list(
            tokenizer.apply_chat_template(
                [{"role": "user", "content": content}],
                add_generation_prompt=True,
                tokenize=True,
                return_dict=False,
            )
        )

    chat_ids = chat("Write one sentence about the sea.")
    # A chat turn, so a coherent source continuation shows the oracle itself
    # handles a prompt longer than the window.
    long_ids = chat(long_prompt_text(40))
    if len(long_ids) <= text_config.sliding_window + DECODE_STEPS:
        raise ValueError(
            f"long prompt has {len(long_ids)} tokens; it must exceed the "
            f"{text_config.sliding_window}-token window"
        )
    cases = [
        (
            "raw_short",
            tokenizer("The capital of France is", add_special_tokens=True)["input_ids"],
        ),
        ("chat_turn", list(chat_ids)),
        ("long_window", long_ids),
    ]

    forced_steps = {}
    if decode_from is not None:
        earlier = json.loads(decode_from.read_text())
        forced_steps = {case["name"]: case["steps"] for case in earlier["cases"]}

    templates = []
    for name, case in TEMPLATE_CASES:
        rendered = tokenizer.apply_chat_template(
            case["messages"],
            tools=case.get("tools"),
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=case.get("enable_thinking", False),
        )
        templates.append(
            {
                "name": name,
                "messages": case["messages"],
                "tools": case.get("tools", []),
                "enable_thinking": case.get("enable_thinking", False),
                "add_generation_prompt": True,
                "rendered": rendered,
                # The rendered text already starts with <bos>.
                "input_ids": tokenizer(rendered, add_special_tokens=False)["input_ids"],
            }
        )

    return {
        "model_id": model_id,
        "revision": revision,
        "reference": {
            "framework": f"transformers {transformers.__version__}",
            "runtime": f"torch {torch.__version__} CPU {dtype} eager, {threads} threads",
            "model_class": type(model).__name__,
            "platform": platform.platform(),
            "config_sha256": sha256_file(config_path),
            "tokenizer_sha256": sha256_file(model_dir / "tokenizer.json"),
            "chat_template_sha256": sha256_file(model_dir / "chat_template.jinja"),
            "weights": weight_files(model_dir),
            "model_type": model.config.model_type,
            "vocab_size": text_config.vocab_size,
            "sliding_window": text_config.sliding_window,
            "layer_count": text_config.num_hidden_layers,
        },
        "decode_steps": DECODE_STEPS,
        "cases": [
            {
                **(
                    captured := capture_case(
                        model,
                        name,
                        ids,
                        output_dir,
                        forced,
                        decode_from.parent if decode_from else None,
                    )
                ),
                # For a human sanity check that the oracle itself is coherent.
                "greedy_text": tokenizer.decode(
                    [step["argmax"] for step in captured["steps"]]
                ),
            }
            for name, ids in cases
            for forced in [forced_steps.get(name)]
        ],
        "templates": templates,
    }


def fixture(float32: dict[str, Any], bfloat16: dict[str, Any]) -> dict[str, Any]:
    """The float32 capture plus, per step, the bfloat16 capture's distance."""
    noise = {case["name"]: case["steps"] for case in bfloat16["cases"]}
    for case in float32["cases"]:
        for step, other in zip(case["steps"], noise[case["name"]], strict=True):
            if step["input_ids"] != other["input_ids"]:
                raise ValueError(f"{case['name']}: the captures fed different tokens")
            step["source_bfloat16"] = {
                "argmax": other["argmax"],
                **other["versus_decode_from"],
            }
    float32["source_bfloat16_runtime"] = bfloat16["reference"]["runtime"]
    return float32


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", type=Path, required=True, help="local Gemma 4 model directory"
    )
    parser.add_argument("--model-id", default="google/gemma-4-12B-it")
    parser.add_argument(
        "--revision",
        default="707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7",
        help="Hub revision of the local files, recorded in the manifest",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="directory for manifest.json and logits",
    )
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="float32")
    parser.add_argument(
        "--decode-from",
        type=Path,
        help="manifest.json whose decode tokens to feed instead of greedy ones",
    )
    parser.add_argument(
        "--fixture",
        type=Path,
        help="with --decode-from: write the combined fixture JSON here",
    )
    args = parser.parse_args()
    if args.fixture and not args.decode_from:
        parser.error("--fixture needs --decode-from")
    manifest = capture(
        args.model,
        args.model_id,
        args.revision,
        args.output,
        args.threads,
        args.dtype,
        args.decode_from,
    )
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    )
    if args.fixture:
        combined = fixture(json.loads(args.decode_from.read_text()), manifest)
        args.fixture.write_text(
            json.dumps(combined, indent=1, sort_keys=True, ensure_ascii=False) + "\n"
        )
    print(
        json.dumps(
            {
                case["name"]: [s["argmax"] for s in case["steps"]]
                for case in manifest["cases"]
            }
        )
    )


if __name__ == "__main__":
    main()
