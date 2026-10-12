#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "mlx==0.32.3",
#   "mlx-lm==0.32.0",
#   "numpy==2.5.3",
# ]
# ///
"""Reference outputs of mlx-lm for an MLX affine Qwen3.5-family checkpoint.

MLX is pinned to the release mlx-sys bundles for the Rust side. This aligns
the backend version; matching graph shapes, dispatch and arithmetic still
requires independent comparison. Historical captures from other versions
must be regenerated for this qualification.

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
  IDs and logits and the gap between the two largest logits. ``--prompts
  heldout`` uses a second set of prompts, kept apart so a bound tuned on the
  first can be checked on cases it never saw.
* ``aa_arms``: per case, how far mlx-lm moves from itself in each of 23
  other configurations that compute the same function, replaying the same
  tokens: its gated-delta recurrence as its Metal kernel or as ops
  (``use_kernel=False``), times the prompt prefilled whole or in chunks of
  16, 12, 8, 1, 7, 9, 15, 17, 32, 64 or 128 tokens, minus the configuration that produced ``cases``.
  Per step, on that step's top-8 IDs: the centered error ``max |d -
  mean(d)|``, the error of the top-1 minus top-2 margin, and ``max |d|``,
  with ``d`` the logit differences. An arm whose logits equal an earlier
  arm's, or the reference's own, at every step is marked ``duplicate``.
  (On a dense model, chunks of 4 and 8 tokens round identically.)
* ``aa``: per step, the largest of each error over the arms that are not
  duplicates. One arm samples only one point of how far rounding alone moves
  greedy decoding, and expert routing amplifies that unevenly between steps,
  so a native implementation is judged against the largest of several.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
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
HELDOUT_PROMPTS = [
    "Describe how a bicycle's gears change the effort needed to climb a hill.",
    "Write a SQL query that returns the five customers with the most orders.",
    "Why does ice float on water?",
    (
        "Read the passage and answer the question after it.\n\n"
        "The bakery on the village square opened in 1902, when a miller's son "
        "built a brick oven behind his house and sold bread from the front room. "
        "For its first decades the oven burned wood cut in the hills above the "
        "village, and flour arrived twice a week by horse and cart from the mill "
        "on the river. In 1931 the baker's daughter began a ledger in which she "
        "wrote down, every evening, how many loaves had been baked and how many "
        "were left unsold; the ledger has been kept every day since. In 1948 the "
        "wood oven was rebuilt to burn gas, and a few years later a truck "
        "replaced the cart. The gas oven served until 1986, when the bakery "
        "installed an electric deck oven that bakes more evenly and needs no "
        "night-time firing. The square has changed around it, but the bakery "
        "still opens at six, and the ledger now fills its fortieth volume.\n\n"
        "Question: list the three ways the oven was heated, in order, and say "
        "what the ledger records."
    ),
]
# Retain the original arms; extra sizes straddle routing-sort and chunk/kernel
# thresholds. They sample reference arithmetic, without observing the candidate.
PREFILL_CHUNKS = (None, 16, 12, 8, 1, 7, 9, 15, 17, 32, 64, 128)
GPU_CAP_BYTES: int | None = None


def check_gpu_memory() -> dict[str, int]:
    """Check MLX active/peak allocation bytes after materialization.

    This cooperative bound detects transient peaks too; it is not an allocator
    hard limit or a measurement of all process/driver memory.
    """
    memory = {
        "active_bytes": mx.get_active_memory(),
        "peak_bytes": mx.get_peak_memory(),
        "cache_bytes": mx.get_cache_memory(),
    }
    if (
        GPU_CAP_BYTES is not None
        and max(memory["peak_bytes"], memory["active_bytes"] + memory["cache_bytes"])
        > GPU_CAP_BYTES
    ):
        raise RuntimeError(f"MLX memory cap exceeded: {memory}, cap={GPU_CAP_BYTES}")
    return memory


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
    check_gpu_memory()
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
    check_gpu_memory()
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


def prefill(model, prompt: list[int], cache, chunk: int | None) -> mx.array:
    """The last prompt position's logits, prefilling ``chunk`` tokens per
    call (``None``: the whole prompt in one call)."""
    step = chunk or len(prompt)
    for start in range(0, len(prompt), step):
        logits = model(mx.array(prompt[start : start + step])[None], cache=cache)
        mx.eval(logits)
        check_gpu_memory()
    return logits[0, -1]


def replay(
    model, prompt: list[int], records: list[dict], chunk: int | None
) -> list[dict]:
    """``aa_arms`` entries: this model's logits on ``records``' token stream,
    compared on each step's recorded top-8 IDs."""
    cache = make_prompt_cache(model)
    logits = prefill(model, prompt, cache, chunk)
    out = []
    for step, record in enumerate(records):
        values = logits.astype(mx.float32)[mx.array(record["top8_ids"])]
        mx.eval(values)
        check_gpu_memory()
        got = [float(v) for v in values.tolist()]
        want = record["top8_logits"]
        d = [g - w for g, w in zip(got, want, strict=True)]
        mean = sum(d) / len(d)
        out.append(
            {
                "centered": max(abs(x - mean) for x in d),
                "margin": abs((got[0] - got[1]) - (want[0] - want[1])),
                "max_abs": max(abs(x) for x in d),
                "logits": got,
            }
        )
        if step + 1 < len(records):
            token = mx.array([[record["top8_ids"][0]]])
            logits = model(token, cache=cache)[0, -1]
    return out


def arms(model, case: dict) -> list[dict]:
    kernel = qwen3_5.gated_delta_update
    out = []
    try:
        for use_kernel in (True, False):
            qwen3_5.gated_delta_update = lambda *args, use=use_kernel, **kwargs: kernel(
                *args, **{**kwargs, "use_kernel": use}
            )
            for chunk in PREFILL_CHUNKS:
                if use_kernel and chunk is None:
                    continue  # the configuration that produced ``cases``
                errors = replay(model, case["prompt_ids"], case["steps"], chunk)
                logits = [e.pop("logits") for e in errors]
                seen = [arm["_logits"] for arm in out]
                seen.append([step["top8_logits"] for step in case["steps"]])
                out.append(
                    {
                        "gdn": "kernel" if use_kernel else "ops",
                        "prefill_chunk": chunk,
                        "duplicate": logits in seen,
                        "steps": errors,
                        "_logits": logits,
                    }
                )
    finally:
        qwen3_5.gated_delta_update = kernel
    for arm in out:
        arm["top8_logits"] = arm.pop("_logits")
    return out


def reference(model_dir: Path, steps: int, prompts: list[str]) -> None:
    model, tokenizer = load(str(model_dir))
    check_gpu_memory()
    cases = []
    for user in prompts:
        prompt = tokenizer.encode(chat_text(user), add_special_tokens=False)
        cases.append(
            {"user": user, "prompt_ids": prompt, "steps": greedy(model, prompt, steps)}
        )
    for case in cases:
        case["aa_arms"] = arms(model, case)
        distinct = [arm["steps"] for arm in case["aa_arms"] if not arm["duplicate"]]
        keys = ("centered", "margin", "max_abs")
        case["aa"] = [
            {
                key: max((arm[step][key] for arm in distinct), default=0.0)
                for key in keys
            }
            for step in range(steps)
        ]
    weights = sorted(model_dir.glob("*.safetensors"))
    component_digests = components(model)
    memory = check_gpu_memory()
    print(
        json.dumps(
            {
                "oracle": "mlx-lm forward with its prompt cache; component digests of BF16 outputs",
                "mlx": importlib.metadata.version("mlx"),
                "mlx_lm": importlib.metadata.version("mlx-lm"),
                "config_sha256": file_sha256(model_dir / "config.json"),
                "weights_sha256": {path.name: file_sha256(path) for path in weights},
                "components": component_digests,
                "qualification": {
                    "bar": "v2",
                    "steps": steps,
                    "min_distinct_arms": 8,
                    "prefill_chunks": PREFILL_CHUNKS,
                    "prompts": "heldout" if prompts == HELDOUT_PROMPTS else "dev",
                    "abort_gpu_bytes": GPU_CAP_BYTES,
                    "memory": memory,
                },
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
    ref.add_argument("--prompts", choices=("dev", "heldout"), default="dev")
    ref.add_argument("--abort-gpu-gib", type=float, required=True)
    args = parser.parse_args()
    if args.command == "truncate":
        truncate(args.source, args.output, args.layers)
    else:
        if not math.isfinite(args.abort_gpu_gib) or args.abort_gpu_gib <= 0:
            parser.error("--abort-gpu-gib must be finite and positive")
        global GPU_CAP_BYTES
        GPU_CAP_BYTES = int(args.abort_gpu_gib * 1024**3)
        mx.reset_peak_memory()
        prompts = HELDOUT_PROMPTS if args.prompts == "heldout" else PROMPTS
        reference(args.model, args.steps, prompts)


if __name__ == "__main__":
    main()
