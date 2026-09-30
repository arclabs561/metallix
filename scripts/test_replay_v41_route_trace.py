#!/usr/bin/env python3
"""CPU-only tests for source-compatible V4.1 route trace replay."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).parent
SCRIPT = SCRIPTS / "replay_v41_route_trace.py"


def load():
    spec = importlib.util.spec_from_file_location(
        "replay_v41_route_trace", SCRIPTS / "replay_v41_route_trace.py"
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load route replay")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def trace(*, positions: int = 2, layer_major: bool = False) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    pairs = (
        ((position, layer) for layer in range(40) for position in range(positions))
        if layer_major
        else ((position, layer) for position in range(positions) for layer in range(40))
    )
    for position, layer in pairs:
        rows.append(
            {
                "request_id": "fixed",
                "phase": "prefill" if position == 0 else "decode",
                "token_position": position,
                "layer": layer,
                "expert_ids": [(layer * 6 + offset) % 384 for offset in range(6)],
            }
        )
    return {
        "schema_version": 1,
        "scope": "source-compatible V4.1 route-only trace",
        "source": {
            "revision": "dba1be0a40aa45a94ad051997016db3960a90277",
            "config_sha256": "8be45ce0476004a3f529fd896115a4a2e800a129ad2d3ec05b16050f52e21879",
            "model_sha256": "4e9ae23620edc8028ccc5d5fef552ab7fdc7dcd6f79608754fe9f67644056f65",
        },
        "geometry": {
            "hidden_width": 5120,
            "backbone_layers": 40,
            "routed_experts": 384,
            "selected_experts": 6,
        },
        "rows": rows,
    }


def run_cli(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *arguments],
        capture_output=True,
        text=True,
        check=False,
    )


class ReplayV41RouteTraceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = load()

    def test_replay_reports_exact_lru_hits_with_supplied_parameters(self) -> None:
        rows = self.module.validate_trace(trace())
        report = self.module.replay(rows, expert_cache_bytes=240, expert_bytes=1)
        observed = report["observed"]
        self.assertEqual(observed["route_rows"], 80)
        self.assertEqual(observed["selections"], 480)
        self.assertEqual(observed["misses"], 240)
        self.assertEqual(observed["hits"], 240)
        self.assertEqual(observed["miss_useful_bytes"], 240)
        self.assertEqual(report["supplied_parameters"]["capacity_experts"], 240)
        self.assertEqual(observed["by_phase"]["prefill"]["misses"], 240)
        self.assertEqual(observed["by_phase"]["decode"]["hits"], 240)

    def test_rejects_unpinned_provenance_and_boolean_geometry(self) -> None:
        document = trace()
        document["source"]["model_sha256"] = "0" * 64
        with self.assertRaisesRegex(self.module.TraceError, "pinned V4.1 source"):
            self.module.validate_trace(document)
        document = trace()
        document["geometry"]["selected_experts"] = True
        with self.assertRaisesRegex(self.module.TraceError, "source-compatible V4.1"):
            self.module.validate_trace(document)
        document = trace()
        document["schema_version"] = True
        with self.assertRaisesRegex(self.module.TraceError, "schema_version"):
            self.module.validate_trace(document)

    def test_replay_rejects_invalid_budget_or_empty_rows(self) -> None:
        rows = self.module.validate_trace(trace())
        for cache_bytes, expert_bytes in ((-1, 1), (True, 1), (0, 0), (0, True)):
            with (
                self.subTest(cache_bytes=cache_bytes, expert_bytes=expert_bytes),
                self.assertRaises(self.module.TraceError),
            ):
                self.module.replay(
                    rows,
                    expert_cache_bytes=cache_bytes,
                    expert_bytes=expert_bytes,
                )
        with self.assertRaisesRegex(self.module.TraceError, "nonempty"):
            self.module.replay([], expert_cache_bytes=0, expert_bytes=1)

    def test_zero_cache_is_an_explicit_all_miss_control(self) -> None:
        rows = self.module.validate_trace(trace(positions=1))
        report = self.module.replay(rows, expert_cache_bytes=0, expert_bytes=7)
        observed = report["observed"]
        self.assertEqual(observed["hits"], 0)
        self.assertEqual(observed["misses"], 240)
        self.assertEqual(observed["miss_useful_bytes"], 1680)

    def test_layer_major_event_order_is_preserved_and_changes_lru_hits(self) -> None:
        token_major = self.module.validate_trace(trace())
        layer_major_document = trace(layer_major=True)
        layer_major = self.module.validate_trace(layer_major_document)
        self.assertEqual(layer_major, layer_major_document["rows"])
        token_major_report = self.module.replay(
            token_major, expert_cache_bytes=6, expert_bytes=1
        )
        layer_major_report = self.module.replay(
            layer_major, expert_cache_bytes=6, expert_bytes=1
        )
        self.assertEqual(token_major_report["observed"]["hits"], 0)
        self.assertEqual(layer_major_report["observed"]["hits"], 240)

    def test_rejects_synthetic_trace(self) -> None:
        document = trace()
        document["synthetic_geometry"] = {"routed_experts": 4}
        with self.assertRaisesRegex(self.module.TraceError, "synthetic"):
            self.module.validate_trace(document)

    def test_rejects_malformed_phase_as_trace_error(self) -> None:
        document = trace()
        document["rows"][0]["phase"] = []
        with self.assertRaisesRegex(self.module.TraceError, "invalid route fields"):
            self.module.validate_trace(document)

    def test_rejects_incomplete_or_reordered_token_group(self) -> None:
        document = trace()
        document["rows"].pop()
        with self.assertRaisesRegex(self.module.TraceError, "layers 0 through 39"):
            self.module.validate_trace(document)
        document = trace()
        document["rows"][0], document["rows"][1] = (
            document["rows"][1],
            document["rows"][0],
        )
        with self.assertRaisesRegex(self.module.TraceError, "layers 0 through 39"):
            self.module.validate_trace(document)

    def test_rejects_duplicate_or_out_of_range_experts(self) -> None:
        for expert_ids in ([0, 0, 2, 3, 4, 5], [0, 1, 2, 3, 4, 384]):
            document = trace()
            document["rows"][0]["expert_ids"] = expert_ids
            with (
                self.subTest(expert_ids=expert_ids),
                self.assertRaisesRegex(self.module.TraceError, "invalid route fields"),
            ):
                self.module.validate_trace(document)

    def test_cli_writes_new_report_and_refuses_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "trace.json"
            input_path.write_text(json.dumps(trace()), encoding="utf-8")
            output_path = root / "report.json"
            arguments = (
                "--trace",
                str(input_path),
                "--expert-cache-bytes",
                "240",
                "--expert-bytes",
                "1",
                "--output",
                str(output_path),
            )
            first = run_cli(*arguments)
            self.assertEqual(first.returncode, 0, first.stderr)
            self.assertEqual(json.loads(output_path.read_text())["schema_version"], 1)
            repeated = run_cli(*arguments)
            self.assertNotEqual(repeated.returncode, 0)
            overwrite = run_cli(
                "--trace",
                str(input_path),
                "--expert-cache-bytes",
                "240",
                "--expert-bytes",
                "1",
                "--output",
                str(input_path),
            )
            self.assertNotEqual(overwrite.returncode, 0)


if __name__ == "__main__":
    unittest.main()
