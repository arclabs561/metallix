"""Negative controls for the fixed reduced V4.1 forward manifest."""

from __future__ import annotations

import copy
import json
import subprocess
import sys
import unittest
from pathlib import Path

import v41_forward_manifest as forward_manifest

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "v41_forward_manifest.py"


class ForwardManifestTests(unittest.TestCase):
    def test_shape_bounds_reject_large_but_group_aligned_allocations(self) -> None:
        for name in ("q_lora_rank", "moe_inter_dim", "hc_mult", "n_routed_experts"):
            candidate = forward_manifest.manifest()
            candidate["model"][name] = 1 << 30
            with (
                self.subTest(name=name),
                self.assertRaisesRegex(
                    forward_manifest.ManifestError, "bounded fixture value"
                ),
            ):
                forward_manifest.validate_manifest(candidate)

    def test_baseline_is_source_runner_ready_but_not_a_completed_capture(self) -> None:
        receipt = forward_manifest.validate_manifest(forward_manifest.manifest())
        self.assertTrue(receipt["validated"])
        self.assertEqual(receipt["allocated_tensors"], 0)
        self.assertTrue(receipt["ready_for_source_attempt"])
        self.assertFalse(receipt["reference_capture_complete"])
        args = forward_manifest.model_args(forward_manifest.manifest())
        self.assertNotIn("quantization", args)
        self.assertNotIn("candidate_consumers", args)
        self.assertEqual(args["compress_ratios"], (0, 2, 2, 1, 1))
        self.assertEqual(args["engram_num_embeddings"], (72, 204))

    def test_cli_emits_manifest_and_validation_receipt(self) -> None:
        result = subprocess.run(
            [sys.executable, str(SCRIPT)],
            check=True,
            capture_output=True,
            cwd=ROOT,
            text=True,
        )
        payload = json.loads(result.stdout)
        self.assertEqual(
            payload["manifest"]["model"]["compress_ratios"], [0, 2, 2, 1, 1]
        )
        self.assertTrue(payload["validation_receipt"]["ready_for_source_attempt"])

    def test_disabled_required_mechanisms_are_rejected(self) -> None:
        candidate = forward_manifest.manifest()
        candidate["model"]["hc_mult"] = 1
        with self.assertRaisesRegex(forward_manifest.ManifestError, "hc_mult"):
            forward_manifest.validate_manifest(candidate)

        candidate = forward_manifest.manifest()
        candidate["model"]["n_routed_experts"] = 1
        with self.assertRaisesRegex(forward_manifest.ManifestError, "routing"):
            forward_manifest.validate_manifest(candidate)

        candidate = forward_manifest.manifest()
        candidate["model"]["candidate_source_layer"] = -1
        with self.assertRaisesRegex(forward_manifest.ManifestError, "candidate source"):
            forward_manifest.validate_manifest(candidate)

        candidate = forward_manifest.manifest()
        candidate["engram"]["status"] = "blocked"
        with self.assertRaisesRegex(forward_manifest.ManifestError, "Engram"):
            forward_manifest.validate_manifest(candidate)

    def test_quantization_width_mismatch_is_rejected(self) -> None:
        candidate = forward_manifest.manifest()
        candidate["model"]["head_dim"] = 48
        with self.assertRaisesRegex(
            forward_manifest.ManifestError, "head_dim must be an exact multiple"
        ):
            forward_manifest.validate_manifest(candidate)

        candidate = forward_manifest.manifest()
        candidate["model"]["quantization"]["compressed_kv_fp4"]["group_width"] = 32
        with self.assertRaisesRegex(
            forward_manifest.ManifestError, "compressed_kv_fp4"
        ):
            forward_manifest.validate_manifest(candidate)

    def test_candidate_domains_must_match(self) -> None:
        candidate = forward_manifest.manifest()
        candidate["model"]["compress_ratios"][4] = 2
        with self.assertRaisesRegex(
            forward_manifest.ManifestError, "compressed-position domain"
        ):
            forward_manifest.validate_manifest(candidate)

    def test_candidate_capacity_cannot_collapse_to_the_pinned_newest_block(
        self,
    ) -> None:
        candidate = forward_manifest.manifest()
        candidate["model"]["candidate_topk_blocks"] = 1
        with self.assertRaisesRegex(
            forward_manifest.ManifestError, "candidate top-k blocks"
        ):
            forward_manifest.validate_manifest(candidate)

    def test_prefill_and_decode_trace_cannot_skip_or_avoid_boundaries(self) -> None:
        candidate = forward_manifest.manifest()
        candidate["trace"][1]["start_pos"] = 6
        with self.assertRaisesRegex(forward_manifest.ManifestError, "contiguous"):
            forward_manifest.validate_manifest(candidate)

        candidate = forward_manifest.manifest()
        candidate["trace"][0]["token_count"] = 4
        with self.assertRaisesRegex(
            forward_manifest.ManifestError, "invalid token count"
        ):
            forward_manifest.validate_manifest(candidate)

    def test_mutation_does_not_change_future_baselines(self) -> None:
        candidate = forward_manifest.manifest()
        copy.deepcopy(candidate)["model"]["compress_ratios"][0] = 7
        self.assertEqual(
            forward_manifest.manifest()["model"]["compress_ratios"], [0, 2, 2, 1, 1]
        )


if __name__ == "__main__":
    unittest.main()
