# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""CPU-only tests for the `mx serve` benchmark's log parsing and request plan."""

from __future__ import annotations

import importlib.util
import pathlib
import sys
import unittest

SCRIPT = pathlib.Path(__file__).with_name("bench_serve.py")
SPEC = importlib.util.spec_from_file_location("bench_serve", SCRIPT)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load {SCRIPT}")
bench_serve = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = bench_serve
SPEC.loader.exec_module(bench_serve)


class LogParsing(unittest.TestCase):
    # Lines as `mx serve` prints them (front process, then a forwarded child line).
    FRONT = "mx listening on http://127.0.0.1:55534; models=1; running=1; one child process per model"
    CHILD = (
        "[embed] mx listening on http://127.0.0.1:55568; models=1; one request per model; "
        "2048 total tokens; kv_budget_bytes=536870912"
    )

    def test_front_and_child_addresses_are_told_apart(self) -> None:
        self.assertEqual(bench_serve.front_address(self.FRONT), "127.0.0.1:55534")
        self.assertIsNone(bench_serve.front_address(self.CHILD))
        self.assertEqual(
            bench_serve.child_address(self.CHILD), ("embed", "127.0.0.1:55568")
        )
        self.assertIsNone(bench_serve.child_address(self.FRONT))
        self.assertIsNone(
            bench_serve.child_address("[embed] mx loaded model=embed; load_ms=909.34")
        )


class RequestPlan(unittest.TestCase):
    def test_each_capability_gets_its_route(self) -> None:
        paths = lambda caps: [
            path for _, path, _ in bench_serve.requests_for("m", caps)
        ]
        self.assertEqual(
            paths(["generate", "decide"]), ["/v1/responses", "/v1/decisions"]
        )
        self.assertEqual(paths(["decide"]), ["/v1/decisions"])
        self.assertEqual(
            paths(["embed"]), ["/v1/embeddings"] * len(bench_serve.EMBEDDING_WORDS)
        )
        self.assertEqual(paths([]), [])
        for _, _, body in bench_serve.requests_for(
            "m", ["generate", "decide", "embed"]
        ):
            self.assertEqual(body["model"], "m")
        embedding = bench_serve.requests_for("m", ["embed"])[-1][2]["input"]
        self.assertEqual(len(embedding.split()), bench_serve.EMBEDDING_WORDS[-1])

    def test_summary_reports_median_and_range(self) -> None:
        self.assertEqual(
            bench_serve.summary([3.0, 1.0, 2.0]),
            {"median_ms": 2.0, "min_ms": 1.0, "max_ms": 3.0, "samples": 3},
        )


if __name__ == "__main__":
    unittest.main()
