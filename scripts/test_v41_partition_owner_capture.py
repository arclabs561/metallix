#!/usr/bin/env python3
"""Counterfactual controls for the observed four-call native-owner operands."""

from __future__ import annotations

import copy
import hashlib
import json
import unittest
from pathlib import Path

import v41_partition_owner_capture as capture


class PartitionOwnerCaptureTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.fixture = json.loads(
            (
                Path(__file__).resolve().parent.parent
                / "fixtures/deepseek-v41/partition-owner-reference.json"
            ).read_text()
        )

    def test_observed_owner_schedule_and_partial_consumers(self) -> None:
        capture.validate_fixture(self.fixture)
        self.assertEqual(
            [(c["start_pos"], c["token_count"]) for c in self.fixture["cases"]],
            [(0, 4), (4, 1), (5, 1), (6, 1)],
        )
        self.assertEqual(
            [c["index_key_prefix"]["shape"] for c in self.fixture["cases"]],
            [[1, n, 64] for n in (4, 5, 6, 7)],
        )
        for case, count in zip(self.fixture["cases"], (2, None, 3, None), strict=True):
            prefix = case["next_layer1_score_prefix"]
            if count is None:
                self.assertIsNone(prefix)
            else:
                self.assertEqual(prefix["shape"], [1, count, 64])
                self.assertTrue(
                    case["index_key_prefix"]["storage_hex"].startswith(
                        prefix["storage_hex"]
                    )
                )

    def test_rejects_malformed_geometry_and_storage(self) -> None:
        for defect in ("shape", "hash", "nonfinite", "schedule", "count", "source"):
            with self.subTest(defect=defect):
                fixture = copy.deepcopy(self.fixture)
                tensor = fixture["cases"][1]["input"]
                if defect == "shape":
                    tensor["shape"] = [128]
                elif defect == "hash":
                    tensor["storage_sha256"] = "0" * 64
                elif defect == "nonfinite":
                    raw = b"\x80\x7f" + bytes.fromhex(tensor["storage_hex"])[2:]
                    tensor["storage_hex"] = raw.hex()
                    tensor["storage_sha256"] = hashlib.sha256(raw).hexdigest()
                elif defect == "schedule":
                    fixture["cases"][1]["start_pos"] = 5
                elif defect == "count":
                    fixture["cases"][1]["token_count"] = True
                else:
                    fixture["source"]["model_sha256"] = "0" * 64
                with self.assertRaises((ValueError, RuntimeError, TypeError)):
                    capture.validate_fixture(fixture)

    def test_rejects_relabelled_capture_identity(self) -> None:
        mutations = (
            ("source", "probe_sha256", "0" * 64),
            ("source", "cpu_backend_sha256", "0" * 64),
            ("source", "observer_sha256", "0" * 64),
            ("capture_identity", "probe_sha256", "0" * 64),
            ("capture_identity", "sha256", "0" * 64),
            ("capture_identity", "schedule", [4, True, 1, 1]),
            ("model", "batches", True),
        )
        for section, key, value in mutations:
            with self.subTest(section=section, key=key):
                fixture = copy.deepcopy(self.fixture)
                fixture[section][key] = value
                with self.assertRaises(capture.CaptureError):
                    capture.validate_fixture(fixture)
        for key, value in (
            ("source_receipt_sha256", "g" * 64),
            ("schema_version", True),
        ):
            with self.subTest(key=key):
                fixture = copy.deepcopy(self.fixture)
                fixture[key] = value
                with self.assertRaises(capture.CaptureError):
                    capture.validate_fixture(fixture)

    def test_selection_geometry_and_causal_boundary(self) -> None:
        capture.validate_fixture(self.fixture)
        self.assertEqual(
            [case["selection"]["offset"] for case in self.fixture["cases"]],
            [4, 6, 6, 6],
        )
        for case in self.fixture["cases"]:
            selection = case["selection"]
            self.assertEqual(selection["candidate_mask"]["dtype"], "torch.bool")
            self.assertEqual(selection["indices"]["shape"], [1, case["token_count"], 1])
            if case["start_pos"] == 0:
                self.assertFalse(selection["causal_scores"]["finite"])
            else:
                self.assertIsNone(selection["causal_scores"])

    def test_rejects_malformed_selection_operands(self) -> None:
        for defect in (
            "offset",
            "mask",
            "fp8",
            "causal_nan",
            "query_shape",
            "model",
            "model_unit",
        ):
            with self.subTest(defect=defect):
                fixture = copy.deepcopy(self.fixture)
                selection = fixture["cases"][0]["selection"]
                if defect == "offset":
                    selection["offset"] = 5
                elif defect == "query_shape":
                    selection["qr"]["shape"] = [1, 5, 32]
                elif defect == "model":
                    fixture["selection_model"]["index_heads"] = True
                elif defect == "model_unit":
                    fixture["selection_model"]["candidate_block_size"] = True
                else:
                    if defect == "mask":
                        tensor, replacement = selection["candidate_mask"], b"\x02"
                    elif defect == "fp8":
                        tensor, replacement = (
                            fixture["selection_weights"]["wq_a_codes"],
                            b"\x7f",
                        )
                    else:
                        tensor, replacement = selection["causal_scores"], b"\xc0\x7f"
                    raw = (
                        replacement
                        + bytes.fromhex(tensor["storage_hex"])[len(replacement) :]
                    )
                    tensor["storage_hex"] = raw.hex()
                    tensor["storage_sha256"] = hashlib.sha256(raw).hexdigest()
                with self.assertRaises((ValueError, RuntimeError, TypeError)):
                    capture.validate_fixture(fixture)

    def test_attention_window_and_publication_boundaries(self) -> None:
        capture.validate_fixture(self.fixture)
        for case in self.fixture["cases"]:
            attention = case["attention"]
            self.assertEqual(attention["start_pos"], case["start_pos"])
            self.assertEqual(
                attention["output"]["shape"], [1, case["token_count"], 128]
            )
            self.assertEqual(attention["window_ring_after"]["shape"], [1, 6, 64])
            self.assertEqual(
                attention["compressed_indices"]["storage_sha256"],
                case["selection"]["indices"]["storage_sha256"],
            )
            self.assertEqual(
                attention["compressed_kv"]["storage_sha256"],
                case["compressed_kv_prefix"]["storage_sha256"],
            )

    def test_rejects_detached_attention_inputs_and_publications(self) -> None:
        for field in ("input", "compressed_kv", "compressed_indices"):
            with self.subTest(field=field):
                fixture = copy.deepcopy(self.fixture)
                tensor = fixture["cases"][1]["attention"][field]
                raw = bytearray.fromhex(tensor["storage_hex"])
                raw[0] ^= 1
                tensor["storage_hex"] = raw.hex()
                tensor["storage_sha256"] = hashlib.sha256(raw).hexdigest()
                with self.assertRaises((ValueError, RuntimeError, TypeError)):
                    capture.validate_fixture(fixture)

    def test_rejects_substituted_partial_consumer(self) -> None:
        fixture = copy.deepcopy(self.fixture)
        tensor = fixture["cases"][0]["next_layer1_score_prefix"]
        raw = bytearray.fromhex(tensor["storage_hex"])
        raw[0] ^= 1
        tensor["storage_hex"] = raw.hex()
        tensor["storage_sha256"] = hashlib.sha256(raw).hexdigest()
        with self.assertRaises((ValueError, RuntimeError)):
            capture.validate_fixture(fixture)

    def test_every_retained_tensor_requires_matching_storage_digest(self) -> None:
        def tensors(value, path=()):
            if isinstance(value, dict):
                if "storage_hex" in value:
                    yield path
                else:
                    for key, child in value.items():
                        yield from tensors(child, (*path, key))
            elif isinstance(value, list):
                for index, child in enumerate(value):
                    yield from tensors(child, (*path, index))

        paths = list(tensors(self.fixture))
        self.assertTrue(paths, "the source fixture must retain numerical operands")
        for path in paths:
            with self.subTest(path=path):
                fixture = copy.deepcopy(self.fixture)
                tensor = fixture
                for key in path:
                    tensor = tensor[key]
                tensor["storage_sha256"] = "0" * 64
                with self.assertRaises(capture.CaptureError):
                    capture.validate_fixture(fixture)

    def test_post_attention_preserves_call_and_layer_handoffs(self) -> None:
        capture.validate_fixture(self.fixture)
        cases = self.fixture["post_attention"]["cases"]
        self.assertEqual([case["start_pos"] for case in cases], [0, 4, 5, 6])
        for owner, case in zip(self.fixture["cases"], cases, strict=True):
            self.assertEqual(
                case["attention_output"]["storage_sha256"],
                owner["attention"]["output"]["storage_sha256"],
            )
            self.assertEqual(
                case["block_output"]["storage_sha256"],
                case["next_block_entry"]["residual"]["storage_sha256"],
            )
            self.assertEqual(
                case["block_next_pre"]["storage_sha256"],
                case["next_block_entry"]["incoming_pre"]["storage_sha256"],
            )

    def test_rejects_detached_post_attention_handoffs(self) -> None:
        for path in (
            ("attention_input",),
            ("attention_output",),
            ("next_block_entry", "residual"),
            ("next_block_entry", "incoming_pre"),
        ):
            with self.subTest(path=path):
                fixture = copy.deepcopy(self.fixture)
                tensor = fixture["post_attention"]["cases"][1]
                for key in path:
                    tensor = tensor[key]
                raw = bytearray.fromhex(tensor["storage_hex"])
                raw[0] ^= 1
                tensor["storage_hex"] = raw.hex()
                tensor["storage_sha256"] = hashlib.sha256(raw).hexdigest()
                with self.assertRaises(capture.CaptureError):
                    capture.validate_fixture(fixture)

    def test_rejects_relabelled_post_attention_contract(self) -> None:
        for path, replacement in (
            (("source", "runner_sha256"), "0" * 64),
            (("source_receipt_sha256",), "0" * 64),
            (("capture_identity", "sha256"), "0" * 64),
            (("model", "n_shared_experts"), True),
            (("model", "norm_topk_prob"), 1),
            (("model", "route_scale"), True),
            (("block_config", "copies"), 2.0),
            (("comparison_policy", "fixed_before_candidate_execution"), 1),
            (("cases", 0, "start_pos"), False),
        ):
            with self.subTest(path=path):
                fixture = copy.deepcopy(self.fixture)
                section = fixture["post_attention"]
                for key in path[:-1]:
                    section = section[key]
                section[path[-1]] = replacement
                with self.assertRaises(capture.CaptureError):
                    capture.validate_fixture(fixture)

    def test_rejects_shadowed_or_misplaced_post_attention_parameters(self) -> None:
        for defect in ("shadow", "move", "extra"):
            with self.subTest(defect=defect):
                fixture = copy.deepcopy(self.fixture)
                post = fixture["post_attention"]
                name = "layers.3.ffn.gate.weight"
                tensor = post["encoded_parameters"][name]
                if defect == "shadow":
                    post["block_parameters"][name] = copy.deepcopy(tensor)
                    tensor["storage_sha256"] = "0" * 64
                elif defect == "move":
                    post["block_parameters"][name] = post["encoded_parameters"].pop(
                        name
                    )
                else:
                    post["encoded_parameters"]["unused.weight"] = copy.deepcopy(tensor)
                with self.assertRaises(capture.CaptureError):
                    capture.validate_fixture(fixture)

    def test_layer_four_consumes_layer_three_publications_and_block_output(
        self,
    ) -> None:
        capture.validate_fixture(self.fixture)
        suffix = self.fixture["post_layer_three"]
        for section in ("attention", "selection", "tail", "head"):
            self.assertEqual(
                [case["start_pos"] for case in suffix[section]["cases"]],
                [0, 4, 5, 6],
            )
        for index, owner in enumerate(self.fixture["cases"]):
            attention = suffix["attention"]["cases"][index]
            selection = suffix["selection"]["cases"][index]
            tail = suffix["tail"]["cases"][index]
            producer_tail = self.fixture["post_attention"]["cases"][index]
            for consumer, producer in (
                (selection["candidate_mask"], owner["selection"]["candidate_mask"]),
                (attention["compressed_kv"], owner["compressed_kv_prefix"]),
                (attention["compressed_indices"], selection["indices"]),
                (tail["block_input"], producer_tail["block_output"]),
                (tail["block_incoming_pre"], producer_tail["block_next_pre"]),
                (tail["attention_output"], attention["output"]),
            ):
                self.assertEqual(consumer["storage_sha256"], producer["storage_sha256"])

    def test_rejects_detached_layer_four_handoffs(self) -> None:
        for section, field in (
            ("selection", "candidate_mask"),
            ("attention", "compressed_kv"),
            ("attention", "compressed_indices"),
            ("tail", "block_input"),
            ("tail", "block_incoming_pre"),
            ("tail", "attention_output"),
        ):
            with self.subTest(section=section, field=field):
                fixture = copy.deepcopy(self.fixture)
                tensor = fixture["post_layer_three"][section]["cases"][1][field]
                raw = bytearray.fromhex(tensor["storage_hex"])
                raw[0] ^= 1
                tensor["storage_hex"] = raw.hex()
                tensor["storage_sha256"] = hashlib.sha256(raw).hexdigest()
                with self.assertRaises(capture.CaptureError):
                    capture.validate_fixture(fixture)

    def test_rejects_relabelled_suffix_provenance(self) -> None:
        for path in (
            ("source", "runner_sha256"),
            ("source_receipt_sha256",),
            ("capture_identity", "sha256"),
        ):
            with self.subTest(path=path):
                fixture = copy.deepcopy(self.fixture)
                section = fixture["post_layer_three"]
                for key in path[:-1]:
                    section = section[key]
                section[path[-1]] = "0" * 64
                with self.assertRaises(capture.CaptureError):
                    capture.validate_fixture(fixture)

    def test_rejects_shadowed_layer_four_tail_parameters(self) -> None:
        fixture = copy.deepcopy(self.fixture)
        tail = fixture["post_layer_three"]["tail"]
        name = "layers.4.ffn.gate.weight"
        tensor = tail["encoded_parameters"][name]
        tail["block_parameters"][name] = copy.deepcopy(tensor)
        tensor["storage_sha256"] = "0" * 64
        with self.assertRaises(capture.CaptureError):
            capture.validate_fixture(fixture)

    def test_rejects_relabelled_suffix_geometry(self) -> None:
        for path, replacement in (
            (("selection", "model", "index_topk"), True),
            (("tail", "model", "n_shared_experts"), True),
            (("attention", "model", "o_groups"), True),
            (("selection", "cases", 0, "start_pos"), False),
            (("attention", "cases", 0, "start_pos"), False),
            (("tail", "cases", 0, "start_pos"), False),
            (("head", "cases", 0, "start_pos"), False),
        ):
            with self.subTest(path=path):
                fixture = copy.deepcopy(self.fixture)
                section = fixture["post_layer_three"]
                for key in path[:-1]:
                    section = section[key]
                section[path[-1]] = replacement
                with self.assertRaises(capture.CaptureError):
                    capture.validate_fixture(fixture)


if __name__ == "__main__":
    unittest.main()
