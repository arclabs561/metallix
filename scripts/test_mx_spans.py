# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Tests for joining mx serve span timelines with client timings."""

from __future__ import annotations

import json
import pathlib
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import bench_load
import mx_spans


def span(ph, name, tid, ms, **args):
    event = {"ph": ph, "pid": 1, "tid": tid, "ts": ms * 1000, "name": name}
    if args:
        event["args"] = {
            k: v if isinstance(v, str) else json.dumps(v) for k, v in args.items()
        }
    return event


# Shaped like timelines recorded from `mx serve --trace-out` at 717f8b1: the
# request span is entered briefly on one thread and then for the whole
# generation on another; fields are strings, quoted when they are strings; the
# proxy can exit before writing its request span's end.
CHILD = [
    span("B", "http.request", 1, 100.0, request_id="r1", route="/v1/chat/completions"),
    span("E", "http.request", 1, 100.1, request_id="r1", route="/v1/chat/completions"),
    span("B", "http.request", 0, 100.2, request_id="r1", route="/v1/chat/completions"),
    span("B", "chat.prefill", 0, 101.0, prompt_tokens="15"),
    span("E", "chat.prefill", 0, 121.0, prompt_tokens="15"),
    span("B", "chat.decode_step", 0, 125.0, step="1"),
    span("E", "chat.decode_step", 0, 135.0, step="1"),
    span("B", "chat.decode_step", 0, 135.0, step="2"),
    span("E", "chat.decode_step", 0, 141.0, step="2"),
    span(
        "E",
        "http.request",
        0,
        150.0,
        request_id="r1",
        route="/v1/chat/completions",
        **{"gen_ai.usage.output_tokens": "3", "mlx.peak_bytes": "1214382208"},
    ),
    span("B", "http.request", 0, 200.0, request_id="m", route="/v1/models"),
    span("E", "http.request", 0, 201.0, request_id="m", route="/v1/models"),
]
PROXY = [
    span("B", "http.request", 1, 99.0, request_id="r1"),
    span("B", "proxy.queue", 1, 99.1),
    span(
        "E",
        "proxy.queue",
        1,
        99.2,
        **{"queue.wait_ms": "4.5", "queue.outcome": '"admitted"'},
    ),
]


class Timelines(unittest.TestCase):
    def test_requests_join_queue_prefill_and_decode(self) -> None:
        (row,) = mx_spans.request_timings(PROXY, CHILD)
        self.assertEqual(row["request_id"], "r1")
        self.assertEqual(row["queue_wait_ms"], 4.5)
        self.assertAlmostEqual(row["prefill_ms"], 20.0)
        # Request span start (100.2) to the end of prefill (121.0), which
        # yields the first token.
        self.assertAlmostEqual(row["first_token_ms"], 20.8)
        self.assertEqual(row["decode_steps"], 2)
        self.assertAlmostEqual(row["decode_step_ms"], 8.0)
        self.assertEqual((row["output_tokens"], row["mlx_peak_bytes"]), (3, 1214382208))

    def test_a_timeline_cut_off_at_exit_still_loads(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "t.json"
            path.write_text("[\n" + ",\n".join(json.dumps(e) for e in PROXY) + ",\n")
            self.assertEqual(mx_spans.load_trace(path), PROXY)
            self.assertEqual(
                mx_spans.child_path(path, "qwen3-0.6b").name, "t.qwen3-0.6b.json"
            )

    def test_client_and_server_disagreements_are_reported(self) -> None:
        server = mx_spans.summarize(mx_spans.request_timings(PROXY, CHILD))
        agree = {"ttft_ms": {"p50": 26.0}, "tpot_ms": {"p50": 9.0}}
        self.assertEqual(mx_spans.compare(server, agree), [])
        # The client saw TTFT four times the server's queue plus first token.
        slow = {"ttft_ms": {"p50": 101.2}, "tpot_ms": {"p50": 9.0}}
        self.assertEqual(
            mx_spans.compare(server, slow),
            ["TTFT p50: client 101.2 ms vs server 25.3 ms (4.00x)"],
        )


class Wiring(unittest.TestCase):
    def test_spans_are_opt_in_and_need_a_timeline_build(self) -> None:
        args = bench_load.build_parser().parse_args(
            ["--server", "metallix", "--model-path", "m", "--mx-spans"]
        )
        spec = bench_load.ServerSpec("metallix", "chat", ["mx", "serve"], {}, {})
        log = pathlib.Path("/logs/metallix-short-concurrency16.log")
        with mock.patch.object(
            bench_load, "mx_serve_flags", lambda mx: frozenset({"--trace-out"})
        ):
            traced, trace = bench_load.with_spans(spec, log, args)
        self.assertEqual(
            trace, pathlib.Path("/logs/metallix-short-concurrency16.trace.json")
        )
        self.assertEqual(traced.argv[-2:], ["--trace-out", str(trace)])
        self.assertEqual(traced.environment, {"METALLIX_LOG": "info"})
        with mock.patch.object(bench_load, "mx_serve_flags", lambda mx: frozenset()):
            self.assertEqual(bench_load.with_spans(spec, log, args), (spec, None))
        args.mx_spans = False
        self.assertEqual(bench_load.with_spans(spec, log, args), (spec, None))


if __name__ == "__main__":
    unittest.main()
