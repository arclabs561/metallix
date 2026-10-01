#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["torch==2.13.0", "transformers==5.0.0"]
# ///
"""Compare native and FP32 source Julia encoders against the same source in float64.

Reads native hidden states (from the ignored Rust diagnostic test) at the path in
NATIVE and writes per-case maximum errors on valid positions to the path in OUT.
"""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent


def load_reference():
    spec = importlib.util.spec_from_file_location(
        "julia_full_prefill_reference", ROOT / "scripts/julia_full_prefill_reference.py"
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load julia_full_prefill_reference.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def max_error(
    value: torch.Tensor, reference: torch.Tensor, mask: torch.Tensor
) -> float:
    return float((value - reference)[mask].abs().max())


def main() -> None:
    reference = load_reference()
    torch.set_num_threads(1)
    fixture = json.loads(reference.FIXTURE.read_text())
    native = json.loads(Path(os.environ["NATIVE"]).read_text())
    f64 = reference.source_encoder("eager").double()
    f32_eager = reference.source_encoder("eager")
    results = []
    for case, observed in zip(fixture["cases"], native, strict=True):
        if case["name"] != observed["case"]:
            raise ValueError("native report case order differs from fixture")
        ids = torch.tensor([case["input_ids"]])
        mask = torch.tensor([case["attention_mask"]])
        with torch.inference_mode():
            exact = f64(input_ids=ids, attention_mask=mask).last_hidden_state[0]
            eager = f32_eager(input_ids=ids, attention_mask=mask).last_hidden_state[0]
        sdpa = torch.tensor(case["expected_hidden"], dtype=torch.float64)
        hidden = torch.tensor(observed["hidden"], dtype=torch.float64).reshape_as(sdpa)
        valid = mask[0].bool()
        results.append(
            {
                "case": case["name"],
                "native_vs_f64": max_error(hidden, exact, valid),
                "source_sdpa_f32_vs_f64": max_error(sdpa, exact, valid),
                "source_eager_f32_vs_f64": max_error(eager.double(), exact, valid),
                "native_vs_sdpa": max_error(hidden, sdpa, valid),
            }
        )
    Path(os.environ["OUT"]).write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
