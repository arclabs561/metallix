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
"""Integrity gate for the unified reduced V4.1 runner source projection."""

from __future__ import annotations

import importlib.util
import json
import os
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).parent
FIXTURE = SCRIPTS.parent / "fixtures/deepseek-v41/reduced-runner-reference.json"
EXPECTED = {
    "layer0_to_layer1",
    "layer1_engram",
    "layer1_owner",
    "layer1_attention",
    "layer1_tail",
    "layer2_attention",
    "layer2_hc",
    "layer2_ffn",
    "layer3_engram",
    "layer3_attention",
    "layer3_moe",
    "layer3_to_layer1",
    "layer4_attention",
    "layer4_moe",
    "head",
}


def load(filename: str, name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / filename)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {filename}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ReducedRunnerFixtureTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.fixture = json.loads(FIXTURE.read_text())

    def test_all_native_seams_name_one_complete_capture(self) -> None:
        fixture = self.fixture
        self.assertEqual(fixture["schema_version"], 1)
        self.assertEqual(
            fixture["trace"],
            {"starts": [0, 5, 6], "input_ids": [[0, 1, 2, 3, 4, 5, 6]]},
        )
        self.assertEqual(set(fixture["projections"]), EXPECTED)
        identity = fixture["source"]["complete_capture_sha256"]
        self.assertEqual(len(identity), 64)
        for name, projection in fixture["projections"].items():
            self.assertEqual(
                projection["source"]["complete_capture_sha256"], identity, name
            )

    def test_cross_projection_handoffs_keep_exact_source_storage(self) -> None:
        projections = self.fixture["projections"]
        layer0 = projections["layer0_to_layer1"]
        layer1_engram = projections["layer1_engram"]
        layer1_owner = projections["layer1_owner"]
        layer2_attention = projections["layer2_attention"]
        head = projections["head"]
        self.assertEqual([case["start_pos"] for case in layer0["cases"]], [0, 5, 6])
        self.assertEqual(
            [case["start_pos"] for case in layer1_engram["cases"]], [0, 5, 6]
        )
        self.assertEqual(
            [case["start_pos"] for case in layer1_owner["cases"]], [0, 5, 6]
        )
        self.assertEqual(
            [case["start_pos"] for case in layer2_attention["cases"]], [0, 5, 6]
        )
        self.assertEqual([case["start_pos"] for case in head["cases"]], [0, 5, 6])
        for zero, engram in zip(layer0["cases"], layer1_engram["cases"]):
            self.assertEqual(
                zero["block_output"]["storage_sha256"],
                engram["stream"]["storage_sha256"],
            )
        for owner, attention in zip(layer1_owner["cases"], layer2_attention["cases"]):
            self.assertEqual(
                owner["compressed_kv_prefix"]["storage_sha256"],
                attention["layer_one_published_kv"]["storage_sha256"],
            )
            self.assertEqual(
                owner["selected_indices"]["storage_sha256"],
                attention["layer_one_published_indices"]["storage_sha256"],
            )

    def test_synthetic_source_route_projection_is_bounded_and_pinned(self) -> None:
        exporter = load("v41_reduced_runner_capture.py", "v41_reduced_runner_exporter")
        trace = exporter.synthetic_route_trace_from_moe_projections(
            self.fixture["projections"]
        )
        self.assertEqual(trace["schema_version"], 1)
        self.assertIn("collector qualification only", trace["scope"])
        self.assertEqual(
            trace["source"]["complete_capture_sha256"],
            self.fixture["source"]["complete_capture_sha256"],
        )
        self.assertEqual(
            trace["synthetic_geometry"],
            {
                "hidden_width": 128,
                "routed_experts": 4,
                "selected_experts": 2,
                "layers": [3, 4],
                "starts": [0, 5, 6],
            },
        )
        self.assertEqual(len(trace["rows"]), 14)
        self.assertEqual(trace["rows"][0]["expert_ids"], [2, 0])
        self.assertEqual(
            trace["rows"][0]["source_gate_indices_sha256"],
            "9e6537257f2121080ec1e88a7952d920d773cb43f74ab6674af1ea52ddd3f335",
        )
        for row in trace["rows"]:
            self.assertIn(row["layer"], (3, 4))
            self.assertIn(row["phase"], ("prefill", "decode"))
            self.assertEqual(len(row["expert_ids"]), 2)
            self.assertTrue(all(0 <= expert < 4 for expert in row["expert_ids"]))
            self.assertEqual(len(row["source_gate_indices_sha256"]), 64)

    def test_synthetic_source_route_projection_rejects_changed_gate_storage(
        self,
    ) -> None:
        exporter = load("v41_reduced_runner_capture.py", "v41_reduced_runner_exporter")
        altered = json.loads(json.dumps(self.fixture["projections"]))
        altered["layer3_moe"]["cases"][0]["gate_indices"]["storage_hex"] = "00"
        with self.assertRaisesRegex(RuntimeError, "source gate storage"):
            exporter.synthetic_route_trace_from_moe_projections(altered)

    def test_synthetic_source_route_projection_rejects_non_int64_gate_indices(
        self,
    ) -> None:
        exporter = load("v41_reduced_runner_capture.py", "v41_reduced_runner_exporter")
        altered = json.loads(json.dumps(self.fixture["projections"]))
        altered["layer4_moe"]["cases"][1]["gate_indices"]["dtype"] = "torch.int32"
        with self.assertRaisesRegex(RuntimeError, "invalid source gate layout"):
            exporter.synthetic_route_trace_from_moe_projections(altered)

    def test_synthetic_source_route_projection_rejects_unpinned_geometry(self) -> None:
        exporter = load("v41_reduced_runner_capture.py", "v41_reduced_runner_exporter")
        altered = json.loads(json.dumps(self.fixture["projections"]))
        altered["layer3_moe"]["cases"][0]["gate_indices"]["shape"] = [4, 2]
        with self.assertRaisesRegex(RuntimeError, "invalid source gate layout"):
            exporter.synthetic_route_trace_from_moe_projections(altered)

    def test_synthetic_source_route_projection_rejects_mixed_source_identity(
        self,
    ) -> None:
        exporter = load("v41_reduced_runner_capture.py", "v41_reduced_runner_exporter")
        altered = json.loads(json.dumps(self.fixture["projections"]))
        altered["layer4_moe"]["source"]["runner_sha256"] = "different-runner"
        with self.assertRaisesRegex(RuntimeError, "mixes source runner_sha256"):
            exporter.synthetic_route_trace_from_moe_projections(altered)


@unittest.skipUnless(
    os.environ.get("V41_REGENERATE_SOURCE") == "1",
    "set V41_REGENERATE_SOURCE=1 to run the pinned source regeneration gate",
)
class ReducedRunnerSourceRegenerationTest(unittest.TestCase):
    def test_committed_fixture_matches_current_complete_source_capture(self) -> None:
        runner = load("v41-forward-reference.py", "v41_reduced_runner_source")
        exporter = load("v41_reduced_runner_capture.py", "v41_reduced_runner_exporter")
        generated = json.loads(
            json.dumps(
                exporter.reduced_runner_fixture(runner.run_capture()),
                sort_keys=True,
                allow_nan=False,
            )
        )
        self.assertEqual(json.loads(FIXTURE.read_text()), generated)


if __name__ == "__main__":
    unittest.main()
