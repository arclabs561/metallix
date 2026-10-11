#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "mlx==0.32.2",
#   "mlx-lm==0.32.0",
#   "numpy==2.5.3",
# ]
# ///
"""Reference outputs of mlx-lm for an MLX affine Qwen3.5-family checkpoint.

MLX is pinned to the release mlx-sys bundles for the Rust side, so both run
the same kernels and a native implementation can match these bit for bit.

``truncate`` copies the first ``--layers`` decoder layers of an mlx-lm
checkpoint (``language_model.model.*``) plus the embedding, final norm and
output head into one ``model.safetensors``, with a matching ``config.json``,
so a 35B mixture-of-experts checkpoint can be checked in a few GB.

``reference`` loads a checkpoint with mlx-lm and prints JSON with:

* ``components``: a SHA-256 of the raw BF16 output bytes of every quantized
  linear module, the embedding and the output head, and of every
  mixture-of-experts block, for synthetic inputs both sides can rebuild
  exactly. Input element ``i`` is ``(((i * 2654435761) mod 2^32) >> 24) -
  128) / 64``, exact in BF16. Linear modules and the head take ``[1, 1, in]``
  and ``[1, 5, in]``; ``MoE`` blocks take ``[1, 1, H]`` and ``[1, 4, H]`` (8
  and 32 selections, the unsorted gather), ``[1, 16, H]`` and
  ``[1, 128, H]`` (sorted by expert; at 128 rows the sorted kernels round
  differently from the unsorted ones). At 1024 or more selections in one
  call MLX switches to another kernel, so longer calls are not covered.
* ``cases``: for a few chat prompts (one over 200 tokens), the prompt IDs,
  then mlx-lm's greedy tokens from its prompt cache, with each step's top-8
  IDs and logits and the gap between the two largest logits.
* ``aa``: per case and step, how far mlx-lm moves from itself when its
  gated-delta recurrence runs as ops instead of its Metal kernel
  (``use_kernel=False``), replaying the same tokens: on that step's top-8
  IDs, the centered error ``max |d - mean(d)|``, the error of the top-1 minus
  top-2 margin, and ``max |d|``, with ``d`` the logit differences. A native
  implementation's distance from mlx-lm is judged against this floor.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import re
import shutil
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx import nn
from mlx_lm import load
from mlx_lm.models import qwen3_5
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.models.switch_layers import QuantizedSwitchLinear

PROMPTS = [
    "Explain how a hash map handles collisions.",
    "Write a Python function that merges two sorted lists.",
    "What causes the seasons on Earth?",
    (
        "Read the passage and answer the question after it.\n\n"
        "The lighthouse at the end of the harbor was built in 1871 from granite "
        "quarried on the island itself. For its first fifty years a keeper and "
        "his family lived in the cottage at its base, trimming the wick every "
        "four hours and winding the clockwork that turned the lens. In 1923 the "
        "light was converted to acetylene, which burned without attention for "
        "months at a time, and the keeper's family moved to the mainland. The "
        "cottage stood empty until 1958, when the harbor authority leased it to "
        "a marine biology station that counted seabirds on the outer rocks and "
        "measured the water temperature every morning at six. The station kept "
        "handwritten logs, and those logs, now digitized, are the longest "
        "continuous record of sea temperature on this stretch of coast. In 1991 "
        "the light was automated with an electric lamp and a solar panel, the "
        "biologists moved to a new building in town, and the cottage became a "
        "small museum that opens on summer weekends.\n\n"
        "Question: list, in order, the three ways the light was powered, and say "
        "what the biologists measured each morning."
    ),
]
COPY = (
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "generation_config.json",
)
PREFIX = "language_model.model."


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def keep(name: str, layers: int) -> bool:
    if name.startswith("language_model.lm_head."):
        return True
    if not name.startswith(PREFIX):
        return False
    rest = name[len(PREFIX) :]
    match = re.match(r"layers\.(\d+)\.", rest)
    if match:
        return int(match.group(1)) < layers
    return rest.startswith(("embed_tokens.", "norm."))


def truncate(source: Path, output: Path, layers: int) -> None:
    config = json.loads((source / "config.json").read_text())
    text = config.get("text_config", config)
    text["num_hidden_layers"] = layers
    text["layer_types"] = text["layer_types"][:layers]
    for key in ("quantization", "quantization_config"):
        if key in config:
            config[key] = {
                path: value
                for path, value in config[key].items()
                if not isinstance(value, (dict, bool)) or keep(path + ".", layers)
            }
    index = json.loads((source / "model.safetensors.index.json").read_text())
    shards = sorted(
        {shard for name, shard in index["weight_map"].items() if keep(name, layers)}
    )
    tensors = {}
    for shard in shards:
        stored = mx.load(str(source / shard))
        tensors.update(
            {name: value for name, value in stored.items() if keep(name, layers)}
        )
    output.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(
        str(output / "model.safetensors"), tensors, metadata={"format": "mlx"}
    )
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    for name in COPY:
        if (source / name).exists():
            shutil.copy(source / name, output / name)


def synthetic(shape: tuple[int, ...]) -> mx.array:
    count = int(np.prod(shape))
    index = np.arange(count, dtype=np.uint64)
    codes = ((index * np.uint64(2654435761)) % np.uint64(1 << 32)) >> np.uint64(24)
    values = (codes.astype(np.float32) - 128.0) / 64.0
    return mx.array(values.reshape(shape)).astype(mx.bfloat16)


def digest(array: mx.array) -> str:
    raw = np.array(mx.view(array.astype(mx.bfloat16), mx.uint16)).astype("<u2")
    return hashlib.sha256(raw.tobytes()).hexdigest()


def components(model) -> dict[str, str]:
    out: dict[str, str] = {}
    for path, module in model.named_modules():
        if isinstance(module, QuantizedSwitchLinear):
            continue
        if isinstance(module, nn.QuantizedEmbedding):
            ids = mx.array([[0, 1, 2, 151643, 248319]], dtype=mx.int32)
            out[f"{path}:embed"] = digest(module(ids))
            columns = module.scales.shape[-1] * module.group_size
            for rows in (1, 5):
                out[f"{path}:as_linear:{rows}"] = digest(
                    module.as_linear(synthetic((1, rows, columns)))
                )
        elif isinstance(module, nn.QuantizedLinear):
            columns = module.scales.shape[-1] * module.group_size
            for rows in (1, 5):
                out[f"{path}:{rows}"] = digest(module(synthetic((1, rows, columns))))
        elif path.endswith(".mlp") and hasattr(module, "switch_mlp"):
            hidden = module.gate.scales.shape[-1] * module.gate.group_size
            for rows in (1, 4, 16, 128):
                out[f"{path}:{rows}"] = digest(module(synthetic((1, rows, hidden))))
    return out


def chat_text(user: str) -> str:
    # One user turn with thinking disabled.
    return (
        f"<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n"
        "<think>\n\n</think>\n\n"
    )


def step_record(logits: mx.array) -> dict:
    logits = logits.astype(mx.float32)
    order = mx.argsort(-logits)[:8]
    values = logits[order]
    mx.eval(order, values)
    top = [float(v) for v in values.tolist()]
    return {
        "top8_ids": [int(i) for i in order.tolist()],
        "top8_logits": top,
        "top2_gap": top[0] - top[1],
    }


def greedy(model, prompt: list[int], steps: int) -> list[dict]:
    cache = make_prompt_cache(model)
    logits = model(mx.array(prompt)[None], cache=cache)[0, -1]
    records = []
    for step in range(steps):
        record = step_record(logits)
        records.append(record)
        if step + 1 < steps:
            token = mx.array([[record["top8_ids"][0]]])
            logits = model(token, cache=cache)[0, -1]
    return records


def replay(model, prompt: list[int], records: list[dict]) -> list[dict]:
    """``aa`` entries: this model's logits on ``records``' token stream,
    compared on each step's recorded top-8 IDs."""
    cache = make_prompt_cache(model)
    logits = model(mx.array(prompt)[None], cache=cache)[0, -1]
    out = []
    for step, record in enumerate(records):
        values = logits.astype(mx.float32)[mx.array(record["top8_ids"])]
        mx.eval(values)
        got = [float(v) for v in values.tolist()]
        want = record["top8_logits"]
        d = [g - w for g, w in zip(got, want, strict=True)]
        mean = sum(d) / len(d)
        out.append(
            {
                "centered": max(abs(x - mean) for x in d),
                "margin": abs((got[0] - got[1]) - (want[0] - want[1])),
                "max_abs": max(abs(x) for x in d),
            }
        )
        if step + 1 < len(records):
            token = mx.array([[record["top8_ids"][0]]])
            logits = model(token, cache=cache)[0, -1]
    return out


def reference(model_dir: Path, steps: int) -> None:
    model, tokenizer = load(str(model_dir))
    cases = []
    for user in PROMPTS:
        prompt = tokenizer.encode(chat_text(user), add_special_tokens=False)
        cases.append(
            {"user": user, "prompt_ids": prompt, "steps": greedy(model, prompt, steps)}
        )
    kernel = qwen3_5.gated_delta_update
    qwen3_5.gated_delta_update = lambda *args, **kwargs: kernel(
        *args, **{**kwargs, "use_kernel": False}
    )
    try:
        for case in cases:
            case["aa"] = replay(model, case["prompt_ids"], case["steps"])
    finally:
        qwen3_5.gated_delta_update = kernel
    weights = sorted(model_dir.glob("*.safetensors"))
    print(
        json.dumps(
            {
                "oracle": "mlx-lm forward with its prompt cache; component digests of BF16 outputs",
                "mlx": importlib.metadata.version("mlx"),
                "mlx_lm": importlib.metadata.version("mlx-lm"),
                "config_sha256": file_sha256(model_dir / "config.json"),
                "weights_sha256": {path.name: file_sha256(path) for path in weights},
                "components": components(model),
                "cases": cases,
            },
            indent=1,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    cut = commands.add_parser("truncate")
    cut.add_argument("--source", type=Path, required=True)
    cut.add_argument("--output", type=Path, required=True)
    cut.add_argument("--layers", type=int, default=4)
    ref = commands.add_parser("reference")
    ref.add_argument("--model", type=Path, required=True)
    ref.add_argument("--steps", type=int, default=32)
    args = parser.parse_args()
    if args.command == "truncate":
        truncate(args.source, args.output, args.layers)
    else:
        reference(args.model, args.steps)


if __name__ == "__main__":
    main()
