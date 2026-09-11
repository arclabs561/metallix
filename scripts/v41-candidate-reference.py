#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = ["torch==2.13.0"]
# ///
"""Capture weight-free masks, FP32 scores, or final selection from V4.1 source."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

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


def capture_index_scores(tree: ast.Module) -> dict[str, object]:
    indexer = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "Indexer"
    )
    forward = next(
        node
        for node in indexer.body
        if isinstance(node, ast.FunctionDef) and node.name == "forward"
    )
    statements = [
        node
        for node in forward.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "index_score"
            for target in node.targets
        )
    ]
    if len(statements) != 2:
        raise ValueError("expected the two official index-score assignments")
    score_code = compile(
        ast.Module(body=statements, type_ignores=[]), SOURCE_URL, "exec"
    )
    cases = [
        ("signed_head_weights", 2, [1, 0, -1, 1], [2, 0, 0, 2, -2, 1], [1, -0.5]),
        ("relu_before_weight_and_sum", 1, [1, -2], [1, -1], [-1, 1]),
        ("zero_weights", 2, [1, 2, -3, 4], [5, -6, 7, 8], [0, 0]),
        ("negative_dots_are_zero", 2, [1, 2], [-3, -4, -5, -6], [-2]),
        (
            "configured_heads_and_dimension",
            128,
            [((index * 7 + 3) % 19 - 9) / 7 for index in range(32 * 128)],
            [((index * 11 + 5) % 29 - 14) / 11 for index in range(17 * 128)],
            [(index % 7 - 3) / 13 for index in range(32)],
        ),
    ]
    captured = []
    with torch.inference_mode():
        for name, head_dim, query, keys, weights in cases:
            heads, positions = len(weights), len(keys) // head_dim
            namespace = {
                "torch": torch,
                "q": torch.tensor(query, dtype=torch.float32).reshape(
                    1, 1, heads, head_dim
                ),
                "index_k": torch.tensor(keys, dtype=torch.float32).reshape(
                    1, positions, head_dim
                ),
                "weights": torch.tensor(weights, dtype=torch.float32).reshape(
                    1, 1, heads
                ),
            }
            exec(score_code, namespace)  # noqa: S102 - inspected hash-pinned assignments only
            captured.append(
                {
                    "name": name,
                    "head_dim": head_dim,
                    "query": query,
                    "keys": keys,
                    "head_weights": weights,
                    "expected_scores": namespace["index_score"].flatten().tolist(),
                }
            )
    return {
        "schema_version": 1,
        "source": {
            "url": SOURCE_URL,
            "revision": REVISION,
            "sha256": SOURCE_SHA256,
            "symbol": "Indexer.forward:index_score",
            "license": "MIT",
        },
        "reference": {
            "torch_version": torch.__version__,
            "device": "cpu",
            "dtype": "float32",
        },
        "scope": "Synthetic one-query FP32 score arithmetic only. Inputs stand in for post-RoPE query/key vectors and already-scaled head weights. No FP4 rounding, BF16 execution, projections, distributed reduction, causal masking, Top-K, or model weights.",
        "cases": captured,
    }


def capture_selection(tree: ast.Module) -> dict[str, object]:
    indexer = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "Indexer"
    )
    forward = next(
        node
        for node in indexer.body
        if isinstance(node, ast.FunctionDef) and node.name == "forward"
    )
    # Preserve the upstream assignments and return expression verbatim in AST.
    statements = forward.body[-3:]
    if not (
        isinstance(statements[0], ast.Assign)
        and isinstance(statements[0].targets[0], ast.Name)
        and statements[0].targets[0].id == "topk"
        and isinstance(statements[1], ast.Assign)
        and isinstance(statements[1].targets[0], ast.Name)
        and statements[1].targets[0].id == "idxs"
        and isinstance(statements[2], ast.Return)
    ):
        raise ValueError("expected upstream Top-K, position sort, and return")
    wrapper = ast.parse("def select():\n    pass\n")
    wrapper.body[0].body = statements
    code = compile(ast.fix_missing_locations(wrapper), SOURCE_URL, "exec")
    cases = [
        ("position_order_not_score_order", [3, 9, 1, 7], 4, 2, 10),
        ("width_clamps_topk", [3, 1, 2], 3, 99, 4),
        ("zero_topk", [3, 2, 1], 3, 0, 4),
        ("zero_width", [], 0, 512, 4),
        ("future_positions_become_sentinels", [9, 2, None, None], 2, 4, 10),
        ("no_visible_positions", [None, None, None], 0, 3, 10),
        ("masked_reachable_positions_are_not_sentinels", [8, None, None], 3, 3, 10),
        ("ties_wholly_selected", [5, 5, 1], 3, 2, 0),
        ("signed_zero_wholly_selected", [0.0, -0.0, -1], 3, 2, 0),
        ("negative_scores", [-9, -2, -5, -1], 4, 2, 100),
    ]
    captured = []
    with torch.inference_mode():
        for name, logits, compress_len, topk, offset in cases:
            namespace = {
                "torch": torch,
                "self": SimpleNamespace(index_topk=topk),
                "end_pos": len(logits),
                "ratio": 1,
                "compress_lens": compress_len,
                "offset": offset,
                "index_score": torch.tensor(
                    [float("-inf") if value is None else value for value in logits],
                    dtype=torch.float32,
                ),
            }
            exec(code, namespace)  # noqa: S102 - hash-pinned inspected AST only
            captured.append(
                {
                    "name": name,
                    "logits": logits,
                    "compress_len": compress_len,
                    "index_topk": topk,
                    "offset": offset,
                    "expected_indices": namespace["select"]().tolist(),
                }
            )
    return {
        "schema_version": 1,
        "source": {
            "url": SOURCE_URL,
            "revision": REVISION,
            "sha256": SOURCE_SHA256,
            "symbol": "Indexer.forward:selection",
            "license": "MIT",
        },
        "reference": {
            "torch_version": torch.__version__,
            "device": "cpu",
            "dtype": "float32",
        },
        "scope": "Synthetic one-query final selection from already-masked scores. No cutoff ties, candidate mask computation, quantization, attention, model weights, or Metal execution. Negative infinity at a reachable position does not imply a -1 output.",
        "cases": captured,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path, required=True, help="pinned inference/model.py"
    )
    parser.add_argument(
        "--kind",
        choices=("candidates", "index-scores", "selection"),
        default="candidates",
    )
    args = parser.parse_args()
    source = args.source.read_bytes()
    if hashlib.sha256(source).hexdigest() != SOURCE_SHA256:
        parser.error("source SHA256 differs from the pinned official implementation")
    tree = ast.parse(source)
    torch.set_num_threads(1)
    if args.kind == "selection":
        print(json.dumps(capture_selection(tree), indent=2, allow_nan=False))
        return
    if args.kind == "index-scores":
        print(json.dumps(capture_index_scores(tree), indent=2, allow_nan=False))
        return
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
