"""Dependency-free guards for source-observer operation identity policy."""

from __future__ import annotations

import ast
import unittest
from pathlib import Path

OBSERVERS = Path(__file__).with_name("v41_forward_observers.py")


class SourceObserverPolicyTest(unittest.TestCase):
    def test_indexer_dispatch_records_distinct_source_phases(self) -> None:
        source = OBSERVERS.read_text()
        ast.parse(source)
        self.assertIn('"scores_after_causal_mask"', source)
        self.assertIn('"scores_after_candidate_mask"', source)
        self.assertIn('"scores_weighted_per_head"', source)
        self.assertIn('"scores_after_head_sum"', source)
        self.assertIn("args[0] is self.state.weights_proj_output", source)
        self.assertIn("args[0] is self.state.einsum_output", source)
        self.assertNotIn("result.shape == projected.shape", source)

    def test_observer_restores_every_patched_source_binding(self) -> None:
        source = OBSERVERS.read_text()
        for name in (
            "graph.torch = original_torch",
            "graph.apply_rotary_emb = original_rotary",
            "graph.fp4_act_quant = original_fp4_quant",
            '"_window_kv"',
            '"_compress_kv"',
            '"forward"',
        ):
            self.assertIn(name, source)


if __name__ == "__main__":
    unittest.main()
