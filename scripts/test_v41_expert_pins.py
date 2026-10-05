#!/usr/bin/env python3
"""Tests for the V4.1 routed-expert pin file writer."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).parent / "v41_expert_pins.py"


def run_file(name: str, routes: list[tuple[int, list[list[int]]]]) -> dict[str, object]:
    return {
        "runs": [
            {
                "name": name,
                "routes": [{"layer": layer, "ids": ids} for layer, ids in routes],
            }
        ]
    }


class PinFileTest(unittest.TestCase):
    def test_pins_every_expert_any_token_routes_to_once_per_layer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            trace = Path(tmp)
            (trace / "one.json").write_text(
                json.dumps(run_file("one", [(0, [[3, 1], [1, 2]]), (1, [[3]])]))
            )
            (trace / "two.json").write_text(
                json.dumps(run_file("two", [(0, [[2, 7]])]))
            )
            subprocess.run(
                [sys.executable, str(SCRIPT), str(trace), "one.json", "two.json"],
                check=True,
                capture_output=True,
            )
            pins = json.loads((trace / "pinned-experts.json").read_text())
        self.assertEqual(pins["sources"], ["one.json", "two.json"])
        # Layer 0 and layer 1 expert 3 are distinct experts; repeats collapse.
        self.assertEqual(pins["experts"], [[0, 1], [0, 2], [0, 3], [0, 7], [1, 3]])


if __name__ == "__main__":
    unittest.main()
