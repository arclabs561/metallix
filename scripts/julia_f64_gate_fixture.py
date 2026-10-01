#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["torch==2.13.0", "transformers==5.0.0"]
# ///
"""Generate the float64 accuracy gate fixture for the native Julia encoder.

For every legacy, calibration and held-out case, records the explicit float64
reference hidden state and scores plus the FP32 SDPA source's maximum hidden
error against it. The native test accepts a case when its own error is at most
GATE_RATIO times the source's. The ratio was chosen on calibration cases only.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "fixtures/julia-1/f64-gate-reference.json"
GATE_RATIO = 4.0


def load_accuracy():
    spec = importlib.util.spec_from_file_location(
        "julia_accuracy_reference", ROOT / "scripts/julia_accuracy_reference.py"
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load julia_accuracy_reference.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    accuracy = load_accuracy()
    torch.set_num_threads(1)
    legacy = [
        (case, True)
        for case in json.loads(accuracy.FIXTURE.read_text())["cases"]
        if case["name"] in accuracy.LEGACY
    ]
    manifest = [(case, False) for case in accuracy.manifest_cases()]
    cases = []
    for case, is_legacy in legacy + manifest:
        ideal = accuracy.f64_case(case, source_f32_rope=False)
        hidden, _ = accuracy.source_case(case, is_legacy)
        valid = torch.tensor(case["attention_mask"], dtype=torch.bool)
        source_error = (hidden.squeeze(0).double() - ideal["hidden"])[valid]
        cases.append(
            {
                "name": case["name"],
                "split": "legacy" if is_legacy else case["split"],
                "input_ids": case["input_ids"],
                "attention_mask": case["attention_mask"],
                "marker_pos": case["marker_pos"],
                "marker_mask": case["marker_mask"],
                "qtype": case["qtype"],
                "f64_hidden": ideal["hidden"].tolist(),
                "f64_scores": ideal["scores"].tolist(),
                "source_f32_hidden_max_abs": float(source_error.abs().max()),
            }
        )
    record = {
        "schema_version": 1,
        "gate_ratio": GATE_RATIO,
        "score_max_abs": 1e-5,
        "reference": "explicit float64 encoder/head from julia_accuracy_reference.f64_case",
        "local_reference_hashes": accuracy.local_reference_hashes(),
        "manifest_sha256": accuracy.MANIFEST_SHA256,
        "cases": cases,
    }
    text = json.dumps(record, separators=(",", ":")) + "\n"
    args.output.write_text(text)
    print(hashlib.sha256(text.encode()).hexdigest(), len(cases))


if __name__ == "__main__":
    main()
