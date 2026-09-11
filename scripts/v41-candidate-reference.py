#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = ["torch==2.13.0"]
# ///
"""Capture weight-free candidate masks from the pinned official V4.1 helper."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path

import torch
import torch.nn.functional as F

REVISION = "dba1be0a40aa45a94ad051997016db3960a90277"
SOURCE_SHA256 = "4e9ae23620edc8028ccc5d5fef552ab7fdc7dcd6f79608754fe9f67644056f65"
SOURCE_URL = f"https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/{REVISION}/inference/model.py"

# None denotes an already causally masked score (-inf), not missing data.
CASES = [
    ("newest_block_is_pinned", [-10, -9, 20, 19, -40], 5, 1, 2),
    ("block_max_not_average", [-50, 9, 5, 6, -100], 5, 2, 2),
    ("prefill_partial_block", [4, 3, 1, None, None, None, None, None], 3, 4, 2),
    ("no_visible_positions", [None] * 6, 0, 2, 2),
    ("topk_exceeds_blocks", [1, 2, 3, 4, 5, 6, 7], 7, 99, 3),
    ("zero_topk", [1, 2, 3, 4], 4, 0, 2),
    ("single_position_blocks", [20, 10, -5], 3, 2, 1),
    ("masked_scores_but_latest_pinned", [None] * 4, 4, 3, 2),
    ("ties_entirely_selected", [5, 5, -2], 3, 3, 1),
    ("masked_tail_dropped", [1, 2, None, None, None, None], 2, 3, 2),
    ("configured_block_size", list(range(24)), 24, 2, 8),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path, required=True, help="pinned inference/model.py"
    )
    args = parser.parse_args()
    source = args.source.read_bytes()
    if hashlib.sha256(source).hexdigest() != SOURCE_SHA256:
        parser.error("source SHA256 differs from the pinned official implementation")
    tree = ast.parse(source)
    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "select_candidate_blocks"
    ]
    if len(functions) != 1:
        parser.error("expected exactly one select_candidate_blocks function")
    # Execute only the inspected, hash-pinned pure Torch helper. Do not import
    # model.py: its top-level imports require CUDA/Triton and unrelated modules.
    namespace = {"torch": torch, "F": F}
    exec(  # noqa: S102 - only the inspected function from the hash-pinned source
        compile(ast.Module(body=functions, type_ignores=[]), str(args.source), "exec"),
        namespace,
    )
    select = namespace["select_candidate_blocks"]
    torch.set_num_threads(1)
    cases = []
    with torch.inference_mode():
        for name, scores, compress_len, topk_blocks, block_size in CASES:
            logits = torch.tensor(
                [float("-inf") if value is None else value for value in scores],
                dtype=torch.float32,
                device="cpu",
            )
            mask = select(logits, compress_len, topk_blocks, block_size)
            cases.append(
                {
                    "name": name,
                    "logits": scores,
                    "compress_len": compress_len,
                    "topk_blocks": topk_blocks,
                    "block_size": block_size,
                    "expected_mask": mask.tolist(),
                }
            )
    print(
        json.dumps(
            {
                "schema_version": 1,
                "source": {
                    "url": SOURCE_URL,
                    "revision": REVISION,
                    "sha256": SOURCE_SHA256,
                    "symbol": "select_candidate_blocks",
                    "license": "MIT",
                },
                "reference": {
                    "torch_version": torch.__version__,
                    "device": "cpu",
                    "dtype": "float32",
                },
                "scope": "Synthetic one-query first-stage candidate masks only; no model weights, index scores, second-stage top-k, attention, or Metal execution. Ambiguous cutoff ties are excluded.",
                "cases": cases,
            },
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
