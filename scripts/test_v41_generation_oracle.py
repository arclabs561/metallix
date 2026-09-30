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
"""Focused policy tests for the bounded V4.1 source generation oracle."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import struct
import unittest
from pathlib import Path

import torch

FIXTURE = (
    Path(__file__).parent.parent / "fixtures/deepseek-v41/generation-reference.json"
)


def load_runner():
    path = Path(__file__).with_name("v41-forward-reference.py")
    spec = importlib.util.spec_from_file_location("v41_generation_oracle_runner", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("source runner import unavailable")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    return runner


class FakeSourceModel:
    """Records feeds while returning deliberately tied, then distinct FP32 logits."""

    def __init__(self) -> None:
        self.calls: list[tuple[int, list[int]]] = []
        self.head = torch.nn.Identity()
        self._logits = (
            torch.tensor([[0.0, 3.0, 3.0, -1.0]], dtype=torch.float32),
            torch.tensor([[0.0, -2.0, 1.0, 4.0]], dtype=torch.float32),
        )

    def __call__(self, input_ids: torch.Tensor, *, start_pos: int):
        self.calls.append((start_pos, [int(item) for item in input_ids[0].tolist()]))
        self.head(torch.ones((1, input_ids.size(1), 128), dtype=torch.bfloat16))
        logits = self._logits[len(self.calls) - 1]
        return torch.tensor([0]), logits, None


class GenerationOracleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.runner = load_runner()

    def test_lowest_id_breaks_a_finite_argmax_tie(self) -> None:
        logits = torch.tensor([[1.0, 4.0, 4.0, 0.0]], dtype=torch.float32)
        self.assertEqual(self.runner._lowest_id_argmax(logits), 1)

    def test_argmax_rejects_nonfinite_or_wrong_shape_logits(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "nonfinite"):
            self.runner._lowest_id_argmax(torch.tensor([[0.0, float("nan")]]))
        with self.assertRaisesRegex(RuntimeError, "one batch"):
            self.runner._lowest_id_argmax(torch.zeros((2, 4), dtype=torch.float32))

    def test_admission_reserves_only_tokens_that_are_fed_back(self) -> None:
        self.runner._generation_admission(
            (0, 1, 2, 3, 4),
            4,
            vocab_size=8,
            max_seq_len=8,
            eos_token_id=None,
        )
        with self.assertRaisesRegex(ValueError, "max_seq_len"):
            self.runner._generation_admission(
                (0, 1, 2, 3, 4),
                5,
                vocab_size=8,
                max_seq_len=8,
                eos_token_id=None,
            )

    def test_case_feeds_only_nonterminal_selection_back_to_source(self) -> None:
        model = FakeSourceModel()
        case = self.runner._generation_case(
            model,
            name="test",
            prompt_ids=(0, 1, 2, 3, 4),
            max_new_tokens=2,
            eos_token_id=None,
        )
        # First logits tie at IDs 1 and 2, so the next source call receives 1.
        self.assertEqual(model.calls, [(0, [0, 1, 2, 3, 4]), (5, [1])])
        self.assertEqual(case["generated_ids"], [1, 3])
        self.assertEqual(case["stop_reason"], "max_new_tokens")
        self.assertEqual(case["selections"][1]["start_pos"], 5)

    def test_eos_is_returned_and_does_not_trigger_another_source_call(self) -> None:
        model = FakeSourceModel()
        case = self.runner._generation_case(
            model,
            name="test",
            prompt_ids=(0, 1, 2, 3, 4),
            max_new_tokens=2,
            eos_token_id=1,
        )
        self.assertEqual(model.calls, [(0, [0, 1, 2, 3, 4])])
        self.assertEqual(case["generated_ids"], [1])
        self.assertEqual(case["stop_reason"], "eos")


class GenerationFixtureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.fixture = json.loads(FIXTURE.read_text())

    def test_compact_fixture_retains_only_replay_essential_data(self) -> None:
        fixture = self.fixture
        self.assertEqual(fixture["schema_version"], 1)
        self.assertEqual(fixture["model"], {"vocab_size": 8, "max_seq_len": 8})
        self.assertEqual(fixture["selection"]["max_new_tokens"], 2)
        self.assertTrue(fixture["selection"]["source_sample_is_not_used"])
        self.assertTrue(fixture["context_bound"]["final_selected_id_is_not_fed_back"])
        self.assertEqual(
            fixture["comparison_policy"],
            {
                "kind": "two_fp32_dot_error_bounds",
                "unit_roundoff_exponent": -24,
                "operation_count_per_dot": 256,
                "bound": "abs_error <= 2 * gamma(operation_count_per_dot) * sum_i(abs(x_i * w_i))",
                "reason": "exact source normalized BF16 rows and FP32 head weights are retained; the envelope is fixed before any Rust candidate execution",
            },
        )
        self.assertNotIn("runtime", fixture)
        self.assertNotIn("manifest", fixture)
        self.assertNotIn("initialized_cache_buffers", fixture["parameter_identity"])
        self.assertEqual(fixture["source_head"]["weight_shape"], [8, 128])
        bits = fixture["source_head"]["weight_fp32_bits"]
        self.assertEqual(len(bits), 8 * 128)
        head_raw = b"".join(struct.pack("<I", value) for value in bits)
        self.assertEqual(
            hashlib.sha256(head_raw).hexdigest(),
            fixture["source_head"]["weight_storage_sha256"],
        )

    def test_fixture_logits_and_feedback_are_exact_and_greedy(self) -> None:
        cases = self.fixture["cases"]
        self.assertEqual(
            [
                (case["name"], case["prompt_ids"], case["generated_ids"])
                for case in cases
            ],
            [
                ("prefill_five", [0, 1, 2, 3, 4], [6, 6]),
                ("prefill_four", [0, 1, 2, 3], [7, 7]),
            ],
        )
        for case in cases:
            self.assertEqual(case["stop_reason"], "max_new_tokens")
            expected_start = 0
            expected_input = case["prompt_ids"]
            for selected, step in zip(
                case["generated_ids"], case["selections"], strict=True
            ):
                self.assertEqual(step["start_pos"], expected_start)
                self.assertEqual(step["input_ids"], expected_input)
                self.assertEqual(step["selected_id"], selected)
                normalized = step["normalized_bf16"]
                self.assertEqual(normalized["dtype"], "torch.bfloat16")
                self.assertEqual(normalized["shape"], [1, 128])
                normalized_raw = bytes.fromhex(normalized["storage_hex"])
                self.assertEqual(len(normalized_raw), 256)
                self.assertEqual(
                    hashlib.sha256(normalized_raw).hexdigest(),
                    normalized["storage_sha256"],
                )
                logits = step["logits"]
                self.assertEqual(logits["dtype"], "torch.float32")
                self.assertEqual(logits["shape"], [1, 8])
                raw = bytes.fromhex(logits["storage_hex"])
                self.assertEqual(
                    hashlib.sha256(raw).hexdigest(), logits["storage_sha256"]
                )
                values = struct.unpack("<8f", raw)
                maximum = max(values)
                self.assertEqual(
                    selected,
                    min(
                        index for index, value in enumerate(values) if value == maximum
                    ),
                )
                expected_start += len(expected_input)
                expected_input = [selected]


if __name__ == "__main__":
    unittest.main()
