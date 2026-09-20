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
        self.assertIn("args[0] is state.weights_proj_output", source)
        self.assertIn("args[0] is state.einsum_output", source)
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

    def test_observer_records_connected_block_hc_operands(self) -> None:
        source = OBSERVERS.read_text()
        self.assertIn('record_name = f"layers.{layer_id}.block_input"', source)
        self.assertIn('name in {"layers.2", "layers.3", "layers.4"}', source)
        self.assertIn(
            '"incoming_pre": object_record(inputs[2], include_storage=True)', source
        )

    def test_observer_records_layer_three_attention_cache_boundaries(self) -> None:
        source = OBSERVERS.read_text()
        self.assertIn("def observed_window_kv_for(", source)
        self.assertIn("def observed_compress_kv_for(", source)
        self.assertIn(
            "layer_three_attention._window_kv = observed_window_kv_for(", source
        )
        self.assertIn(
            "layer_three_attention._compress_kv = observed_compress_kv_for(", source
        )
        self.assertIn('record_name = f"layers.{layer_id}.attn.compressed"', source)
        self.assertIn('name in {"layers.3.attn.wo_b", "layers.4.attn.wo_b"}', source)


if __name__ == "__main__":
    unittest.main()
