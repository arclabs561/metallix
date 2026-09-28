#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["torch==2.13.0", "transformers==5.0.0"]
# ///
"""Source-backed two-layer Julia prefill plus decision-head fixture."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Any

import torch

ROOT = Path(__file__).resolve().parent.parent
FIXTURE = ROOT / "fixtures/julia-1/prefill-reference.json"


def load(name: str, filename: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / filename)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {filename}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ENCODER = load("julia_encoder_reference", "julia_encoder_reference.py")
HEAD = load("julia_head_reference", "julia_head_reference.py")
WIDTH = 384


def hidden(perturb_padding: bool = False) -> torch.Tensor:
    values = torch.arange(6 * WIDTH, dtype=torch.float32).reshape(6, WIDTH)
    output = ((values * 7).remainder(29) - 14) / 20
    if perturb_padding:
        perturbation = torch.arange(2 * WIDTH, dtype=torch.float32).reshape(2, WIDTH)
        output[4:] += ((perturbation * 11).remainder(31) - 15) / 3
    return output


def source_scores(case: dict[str, Any]) -> torch.Tensor:
    source_model, fake_encoder = HEAD.build_source_oracle()
    value = hidden(case.get("perturb_padding", False))
    mask = torch.tensor(case["attention_mask"], dtype=torch.bool)
    value = ENCODER.source_block(value, mask, 0, ENCODER.weights(0))
    value = ENCODER.source_block(value, mask, 1, ENCODER.weights(6))
    fake_encoder.hidden = value.unsqueeze(0)
    marker_pos = torch.tensor([case["marker_pos"]], dtype=torch.int64)
    marker_mask = torch.tensor([case["marker_mask"]], dtype=torch.bool)
    qtype = torch.tensor([case["qtype"]], dtype=torch.int64)
    input_ids = torch.arange(6, dtype=torch.int64).unsqueeze(0)
    with torch.inference_mode():
        return source_model(
            input_ids, mask.unsqueeze(0), marker_pos, marker_mask, qtype
        )


def records() -> dict[str, Any]:
    cases = (
        {
            "name": "base",
            "attention_mask": [True, True, True, True, False, False],
            "marker_pos": [1, 3],
            "marker_mask": [True, True],
            "qtype": 2,
        },
        {
            "name": "padding_perturbation",
            "attention_mask": [True, True, True, True, False, False],
            "marker_pos": [1, 3],
            "marker_mask": [True, True],
            "qtype": 2,
            "perturb_padding": True,
        },
        {
            "name": "option_permutation",
            "attention_mask": [True, True, True, True, False, False],
            "marker_pos": [3, 1],
            "marker_mask": [True, True],
            "qtype": 2,
        },
        {
            "name": "unmasked_padding_control",
            "attention_mask": [True, True, True, True, True, True],
            "marker_pos": [1, 3],
            "marker_mask": [True, True],
            "qtype": 2,
            "perturb_padding": True,
        },
        {
            "name": "masked_marker",
            "attention_mask": [True, True, True, True, False, False],
            "marker_pos": [1, 3, 2],
            "marker_mask": [True, True, False],
            "qtype": 2,
        },
    )
    observed = []
    for case in cases:
        scores = source_scores(case)
        observed.append({**case, "expected_scores": scores.squeeze(0).tolist()})
    return {
        "schema_version": 1,
        "sources": {
            "julia_model": {
                "revision": "a85b127321d580d65176c89ced8273f305745d85",
                "sha256": HEAD.SOURCE_SHA256,
            },
            "modernbert": {
                "revision": ENCODER.REVISION,
                "sha256": ENCODER.SOURCE_SHA256,
            },
        },
        "operator_config": {
            "encoder_layers": [0, 1],
            "head_layers": 2,
            "positions": 6,
            "width": WIDTH,
            "tolerance": 1e-5,
        },
        "input_generation": "arange(6*384), ((x*7)%29-14)/20; optional padded rows 4..5 add ((x*11)%31-15)/3",
        "weight_generation": "encoder ordinals layer0=0..5, layer1=6..11; Julia head uses pinned named-parameter ordinals from head-reference",
        "cases": observed,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    output = records()
    if args.write:
        FIXTURE.write_text(json.dumps(output, indent=2) + "\n")
    elif output != json.loads(FIXTURE.read_text()):
        raise SystemExit("prefill fixture does not match pinned source")


if __name__ == "__main__":
    main()
