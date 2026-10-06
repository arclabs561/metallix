#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "diffusers==0.41.0",
#   "numpy>=2",
#   "torch==2.14.1",
# ]
# ///
"""Capture diffusers' flow-matching sigma schedules as exact float32 bits.

For each scheduler config and (steps, image sequence length) pair this runs the
pipeline's own recipe: starting sigmas `linspace(1, 1/steps, steps)`, the
pipeline's `mu` (FLUX.2's empirical fit or Qwen-Image's linear shift), then
`FlowMatchEulerDiscreteScheduler.set_timesteps`. The fixture stores each sigma
as its float32 bit pattern so the Rust schedule is compared bit for bit.
"""

from __future__ import annotations

import argparse
import json
import platform
import struct
from pathlib import Path

import diffusers
import numpy as np
import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.pipelines.flux2.pipeline_flux2_klein import compute_empirical_mu
from diffusers.pipelines.qwenimage21.pipeline_qwenimage21 import calculate_shift

STEPS = [1, 2, 3, 4, 8, 28, 40, 50]
SEQ_LENS = [256, 1024, 4096, 4301, 6400, 16384]


def f32_bits(value: float) -> str:
    return struct.pack(">f", value).hex()


def f64_bits(value: float) -> str:
    return struct.pack(">d", value).hex()


def mu_for(rule: str, config: dict, seq_len: int, steps: int) -> float:
    if rule == "flux2_empirical":
        return compute_empirical_mu(image_seq_len=seq_len, num_steps=steps)
    if rule == "linear":
        return calculate_shift(
            seq_len,
            config.get("base_image_seq_len", 256),
            config.get("max_image_seq_len", 4096),
            config.get("base_shift", 0.5),
            config.get("max_shift", 1.15),
        )
    raise ValueError(rule)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        action="append",
        nargs=3,
        metavar=("NAME", "MU_RULE", "PATH"),
        required=True,
        help="a scheduler_config.json, its mu rule (flux2_empirical or linear) and a label",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    schedules = []
    for name, rule, path in args.config:
        text = Path(path).read_text()
        config = json.loads(text)
        cases = []
        for steps in STEPS:
            for seq_len in SEQ_LENS:
                scheduler = FlowMatchEulerDiscreteScheduler.from_config(config)
                mu = mu_for(rule, config, seq_len, steps)
                sigmas = np.linspace(1.0, 1 / steps, steps)
                scheduler.set_timesteps(steps, sigmas=sigmas, mu=mu)
                cases.append(
                    {
                        "steps": steps,
                        "image_seq_len": seq_len,
                        "mu_f64_bits": f64_bits(mu),
                        "sigmas_f32_bits": [
                            f32_bits(v) for v in scheduler.sigmas.tolist()
                        ],
                        "timesteps_f32_bits": [
                            f32_bits(v) for v in scheduler.timesteps.tolist()
                        ],
                    }
                )
        schedules.append(
            {"name": name, "mu_rule": rule, "config": config, "cases": cases}
        )

    fixture = {
        "source": "diffusers FlowMatchEulerDiscreteScheduler.set_timesteps",
        "versions": {
            "diffusers": diffusers.__version__,
            "numpy": np.__version__,
            "torch": torch.__version__,
            "python": platform.python_version(),
        },
        "schedules": schedules,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(fixture, indent=1) + "\n")


if __name__ == "__main__":
    main()
