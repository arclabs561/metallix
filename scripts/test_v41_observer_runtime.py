#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "torch==2.13.0",
#   "numpy==2.5.3",
#   "sympy==1.14.0",
#   "tokenizers==0.23.2",
# ]
# ///
"""Execute observer invariants against the retained, hash-gated source graph.

This opt-in test needs the pinned source files, but downloads no model weights.
The dependency-free policy tests remain part of the ordinary local gate.
"""

from __future__ import annotations

import importlib.util
import json
import unittest
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch

import torch


def load_runner():
    path = Path(__file__).with_name("v41-forward-reference.py")
    spec = importlib.util.spec_from_file_location("v41_observer_test_runner", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("source runner import unavailable")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    return runner


class ObserverRuntimeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.runner = load_runner()

    def test_observation_does_not_change_source_outputs_or_cache_bytes(self) -> None:
        observed = self.runner.run_capture()
        projected = self.runner.v41_attention_capture.attention_fixture(
            observed, helper_path=self.runner.SCRIPTS / "v41_attention_capture.py"
        )
        # Compare at the fixture's JSON boundary (source config tuples become lists).
        projected = json.loads(json.dumps(projected, allow_nan=False))
        historical = json.loads(
            (
                self.runner.ROOT
                / "fixtures/deepseek-v41/forward-attention-reference.json"
            ).read_text()
        )
        for field, expected in historical.items():
            if field != "source":
                self.assert_historical_values(projected[field], expected)
        with patch.object(
            self.runner.v41_forward_observers,
            "hooks_for",
            side_effect=lambda *_args, **_kwargs: nullcontext({}),
        ):
            unobserved = self.runner.run_capture()
        self.assertEqual(len(observed["steps"]), 3)
        self.assertEqual(len(unobserved["steps"]), 3)
        for actual, baseline in zip(
            observed["steps"], unobserved["steps"], strict=True
        ):
            self.assertEqual(
                {key: value for key, value in actual.items() if key != "intermediates"},
                {
                    key: value
                    for key, value in baseline.items()
                    if key != "intermediates"
                },
            )
            records = actual["intermediates"]
            producer = records["layers.3.attn.indexer_observation"]
            self.assertEqual(producer["inputs"]["qr"], records["layers.3.attn.q_norm"])
            self.assertEqual(
                producer["candidate_mask_after"],
                actual["caches_after"]["shared.candidates"],
            )
            self.assertEqual(
                producer["candidate_mask_after"],
                records["layers.4.attn.indexer_observation"]["inputs"][
                    "candidate_mask"
                ],
            )
            operations = producer["operations"]
            self.assertEqual(len(operations["q_after_rope_fp4"]["shape"]), 4)
            self.assertEqual(len(operations["k_after_rope_fp4"]["shape"]), 3)
            self.assertNotEqual(
                operations["q_after_rope_fp4"]["storage_sha256"],
                operations["k_after_rope_fp4"]["storage_sha256"],
            )
        self.assertEqual(
            observed["candidate_filtering"], unobserved["candidate_filtering"]
        )

    def assert_historical_values(self, actual: object, expected: object) -> None:
        """Allow additive observations, never changed historical numerical values."""
        if isinstance(expected, dict):
            self.assertIsInstance(actual, dict)
            for key, value in expected.items():
                self.assert_historical_values(actual[key], value)
        elif isinstance(expected, list):
            self.assertIsInstance(actual, list)
            self.assertEqual(len(actual), len(expected))
            for value, prior in zip(actual, expected, strict=True):
                self.assert_historical_values(value, prior)
        else:
            self.assertEqual(actual, expected)

    def test_source_exception_restores_hooks_and_instance_graph_bindings(self) -> None:
        runner = self.runner
        graph = runner.source_loader.load_text_graph(runner.kernels)
        graph.shared_attn = graph.SharedAttentionRuntime()
        with graph.set_dtype(torch.bfloat16):
            args, tokenizer = runner.executable_args(
                graph, runner.forward_manifest.manifest()
            )
            model = graph.Transformer(args, tokenizer).eval()
            runner.initialize_parameters(model)
            runner.initialize_runtime_buffers(model)
            modules = dict(model.named_modules())
            hook_counts = {
                name: (len(module._forward_hooks), len(module._forward_pre_hooks))
                for name, module in modules.items()
            }
            targets = (
                (model.layers[3], "hc_mixes"),
                (model.layers[3].attn, "_window_kv"),
                (model.layers[3].attn, "_compress_kv"),
                (model.layers[3].attn.indexer, "forward"),
                (model.layers[4].attn.indexer, "forward"),
                (model.layers[4], "hc_mixes"),
                (model.layers[4].attn, "_window_kv"),
                (model.layers[4].attn, "_compress_kv"),
            )
            originals = [
                (name in obj.__dict__, obj.__dict__.get(name)) for obj, name in targets
            ]
            graph_originals = {
                name: getattr(graph, name)
                for name in ("torch", "apply_rotary_emb", "fp4_act_quant")
            }
            with (
                patch.object(
                    model.layers[3].attn.indexer.wq_b,
                    "forward",
                    side_effect=RuntimeError("injected candidate-query failure"),
                ),
                self.assertRaisesRegex(
                    RuntimeError, "injected candidate-query failure"
                ),
                runner.v41_forward_observers.hooks_for(
                    model,
                    graph,
                    runner.tensor_record,
                    runner.object_record,
                    runner.MAX_HOOK_RECORDS,
                ),
            ):
                model(torch.tensor([runner.TRACE_INPUT_IDS[0][:5]]), start_pos=0)
            for (obj, name), (existed, original) in zip(
                targets, originals, strict=True
            ):
                self.assertEqual(name in obj.__dict__, existed)
                self.assertIs(obj.__dict__.get(name), original)
            for name, original in graph_originals.items():
                self.assertIs(getattr(graph, name), original)
            for name, module in modules.items():
                self.assertEqual(
                    (len(module._forward_hooks), len(module._forward_pre_hooks)),
                    hook_counts[name],
                )


if __name__ == "__main__":
    unittest.main()
