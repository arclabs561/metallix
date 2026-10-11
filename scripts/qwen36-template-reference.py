#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "jinja2==3.1.6",
#   "tokenizers==0.23.2",
#   "transformers==5.18.0",
# ]
# ///
"""Render Qwen3.6 chat-template references with transformers.

Writes the checkpoint's chat template, its source manifest and the
renderings of multi-turn agent conversations with ``preserve_thinking``
set, unset and false, in thinking and non-thinking mode. mx must render
each case byte for byte (crates/chat-format), which pins how caller
template variables reach the template. Needs only the tokenizer files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import transformers
from transformers import AutoTokenizer

MODEL_ID = "Qwen/Qwen3.6-35B-A3B"
REVISION = "995ad96eacd98c81ed38be0c5b274b04031597b0"

# Keys in sorted order: mx's request JSON keeps objects key-sorted, and
# transformers' `tojson` keeps insertion order, so only a sorted tool
# compares the kwargs handling rather than key order.
EXEC_TOOL = {
    "function": {
        "description": "Run a shell command.",
        "name": "exec_command",
        "parameters": {
            "properties": {"cmd": {"type": "string"}},
            "required": ["cmd"],
            "type": "object",
        },
    },
    "type": "function",
}


def call(cmd: str) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {"name": "exec_command", "arguments": {"cmd": cmd}},
    }


def conversations() -> list[tuple[str, list[dict[str, Any]], list[dict[str, Any]]]]:
    """Agent histories whose earlier assistant turns carry reasoning."""
    system = {"role": "system", "content": "You are a coding agent."}
    port = [
        system,
        {"role": "user", "content": "Where is the HTTP port set?"},
        {
            "role": "assistant",
            "reasoning_content": "The port is likely in a config file. Search for it.",
            "content": "",
            "tool_calls": [call("rg -n port src")],
        },
        {"role": "tool", "content": "src/config.rs:12:    pub port: u16,"},
        {
            "role": "assistant",
            "reasoning_content": "Found it in src/config.rs.",
            "content": "It is set in src/config.rs, line 12.",
        },
        {"role": "user", "content": "Change it to 8081."},
    ]
    mid_task = port[:4]
    plain = [
        {"role": "user", "content": "Say hi."},
        {"role": "assistant", "reasoning_content": "A greeting.", "content": "Hi!"},
        {"role": "user", "content": "Again, in French."},
    ]
    inline = [
        {"role": "user", "content": "2 + 2?"},
        {"role": "assistant", "content": "<think>\nAdd them.\n</think>\n\n4"},
        {"role": "user", "content": "Times 3?"},
    ]
    return [
        ("tool_history_new_query", port, [EXEC_TOOL]),
        ("tool_history_mid_task", mid_task, [EXEC_TOOL]),
        ("plain_history", plain, []),
        ("reasoning_inline_in_content", inline, []),
    ]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", type=Path, required=True, help="local Qwen3.6 snapshot directory"
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="fixture directory to write"
    )
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(str(args.model))
    template_path = args.model / "chat_template.jinja"
    config_path = args.model / "tokenizer_config.json"
    config = json.loads(config_path.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(template_path, args.output / "chat-template.jinja")
    manifest = {
        "schema_version": 1,
        "source": {
            "model_id": MODEL_ID,
            "revision": REVISION,
            "field": "chat_template.jinja",
            "tokenizer_config_sha256": sha256_file(config_path),
            "template_sha256": sha256_file(template_path),
            "bos_token": config.get("bos_token"),
            "eos_token": config.get("eos_token"),
        },
    }
    (args.output / "chat-template.manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )

    cases = []
    for name, messages, tools in conversations():
        for thinking in (True, False):
            for preserve in (None, True, False):
                kwargs = {} if preserve is None else {"preserve_thinking": preserve}
                rendered = tokenizer.apply_chat_template(
                    messages,
                    tools=tools or None,
                    add_generation_prompt=True,
                    tokenize=False,
                    enable_thinking=thinking,
                    **kwargs,
                )
                cases.append(
                    {
                        "name": name,
                        "messages": messages,
                        "tools": tools,
                        "enable_thinking": thinking,
                        "chat_template_kwargs": kwargs,
                        "rendered": rendered,
                    }
                )
    reference = {
        "model_id": MODEL_ID,
        "revision": REVISION,
        "framework": f"transformers {transformers.__version__}",
        "chat_template_sha256": manifest["source"]["template_sha256"],
        "cases": cases,
    }
    (args.output / "template-reference.json").write_text(
        json.dumps(reference, indent=1) + "\n"
    )


if __name__ == "__main__":
    main()
