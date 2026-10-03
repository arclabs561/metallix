#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["torch==2.13.0", "transformers==5.0.0", "safetensors", "tokenizers"]
# ///
"""Float64 reference for the native Julia encoder on the published checkpoint.

Runs the pinned source model, imported from the checkpoint directory's own
``julia`` package, in FP32 SDPA and in float64 eager attention on fixed
requests. Writes the float64 hidden states and scores plus the FP32 source's
own error, which the native real-checkpoint test gates against.

Usage: uv run scripts/julia_real_f64_reference.py CHECKPOINT_DIR OUTPUT.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

REVISION = "a85b127321d580d65176c89ced8273f305745d85"
# Calibration cases were run first; held-out cases were added unchanged afterwards.
REQUESTS = (
    ("calib-a", {"state": "hp 3", "question": "next?", "options": ["heal", "attack"]}),
    (
        "calib-b",
        {"state": "low", "question": "go?", "options": ["yes", "no"], "type": "noul"},
    ),
    ("held-a", {"state": "rain", "question": "take?", "options": ["umbrella", "hat"]}),
    (
        "held-b",
        {
            "state": "x=2",
            "question": "rate",
            "options": ["good", "bad"],
            "type": "score",
        },
    ),
    ("held-c", {"state": "", "question": "pick", "options": ["a", "b"]}),
    (
        "held-long",
        {
            "state": {
                "inventory": ["sword", "potion", "map"],
                "enemies": 3,
                "hp": 12,
                "weather": "storm",
            },
            "question": "What should the party do next?",
            "options": ["retreat to camp", "fight", "drink potion", "read map"],
        },
    ),
)


def main() -> None:
    checkpoint, output = Path(sys.argv[1]).resolve(), Path(sys.argv[2])
    sys.path.insert(0, str(checkpoint))
    from julia.data import sequence
    from julia.model import JuliaDecisionModel
    from transformers import PreTrainedTokenizerFast

    config = json.loads((checkpoint / "tokenizer/tokenizer_config.json").read_text())
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=str(checkpoint / "tokenizer/tokenizer.json"),
        **{key: value for key, value in config.items() if key.endswith("_token")},
    )
    torch.set_num_threads(1)
    source = JuliaDecisionModel.from_pretrained(checkpoint).eval()
    ideal = JuliaDecisionModel.from_pretrained(checkpoint).double().eval()
    ideal.encoder.config._attn_implementation = "eager"
    cases = []
    for name, request in REQUESTS:
        encoded = sequence(tokenizer, request, max_length=128, head_length=32)
        ids, markers = encoded["ids"], encoded["markers"]
        arguments = (
            torch.tensor([ids]),
            torch.ones(1, len(ids), dtype=torch.bool),
            torch.tensor([markers]),
            torch.ones(1, len(markers), dtype=torch.bool),
            torch.tensor([encoded["qtype"]]),
        )
        with torch.inference_mode():
            hidden32 = source.encoder(
                input_ids=arguments[0], attention_mask=arguments[1]
            ).last_hidden_state[0]
            scores32 = source(*arguments)[0]
            hidden64 = ideal.encoder(
                input_ids=arguments[0], attention_mask=arguments[1]
            ).last_hidden_state[0]
            scores64 = ideal(*arguments)[0]
        cases.append(
            {
                "name": name,
                "input_ids": ids,
                "markers": markers,
                "qtype": encoded["qtype"],
                "f64_hidden": hidden64.tolist(),
                "f64_scores": scores64.tolist(),
                "source_f32_scores": scores32.tolist(),
                "source_f32_hidden_max_abs": (hidden32.double() - hidden64)
                .abs()
                .max()
                .item(),
            }
        )
        print(name, len(ids), cases[-1]["source_f32_hidden_max_abs"], flush=True)
    output.write_text(json.dumps({"revision": REVISION, "cases": cases}))


if __name__ == "__main__":
    main()
