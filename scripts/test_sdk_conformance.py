"""Offline tests for the SDK conformance suite's parsing and guard helpers."""

from __future__ import annotations

import importlib.util
import pathlib
import sys
import unittest
from unittest.mock import patch

SCRIPT = pathlib.Path(__file__).with_name("sdk-conformance.py")
SPEC = importlib.util.spec_from_file_location("sdk_conformance", SCRIPT)
assert SPEC and SPEC.loader
module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)


class FootprintTest(unittest.TestCase):
    def test_reads_each_unit(self):
        header = "mx [42]: 64-bit    Footprint: {} (16384 bytes per page)"
        self.assertEqual(module.footprint_bytes(header.format("1824 KB")), 1824 * 2**10)
        self.assertEqual(module.footprint_bytes(header.format("1.5 GB")), 3 * 2**29)
        self.assertEqual(module.footprint_bytes(header.format("12 MB")), 12 * 2**20)
        self.assertIsNone(module.footprint_bytes("footprint: no such process"))

    def test_guard_stops_the_group_once_above_the_cap(self):
        guard = module.FootprintGuard(pgid=4242, cap=2**30)
        with (
            patch.object(guard, "sample", side_effect=[2**29, 2**31, 2**31]),
            patch.object(module.os, "killpg") as killpg,
            patch.object(guard._stop, "wait", side_effect=[False, False, False, True]),
        ):
            guard._run()
        killpg.assert_called_once_with(4242, module.signal.SIGTERM)
        self.assertEqual(guard.peak, 2**31)
        self.assertIn("passed 1 GiB", guard.aborted)


class SseTest(unittest.TestCase):
    def test_splits_named_events_and_skips_comments(self):
        lines = [
            "event: message_start",
            'data: {"type":"message_start"}',
            "",
            ": keepalive",
            "",
            "data: [DONE]",
        ]
        self.assertEqual(
            module.sse_events(lines),
            [("message_start", '{"type":"message_start"}'), (None, "[DONE]")],
        )

    def test_joins_multiline_data(self):
        self.assertEqual(
            module.sse_events([b"data: a", b"data: b", b""]), [(None, "a\nb")]
        )


class ReportTest(unittest.TestCase):
    def test_notes_do_not_fail_but_are_listed(self):
        report = module.Report(case="c")
        report.present({"a": 1}, ("a",), "body")
        report.present({"a": 1}, ("b",), "body", level="note")
        report.check("broken", False, "why")
        self.assertEqual(
            [(status, name) for _, name, status, _ in report.failures()],
            [("note", "body has b"), ("fail", "broken")],
        )


if __name__ == "__main__":
    unittest.main()
