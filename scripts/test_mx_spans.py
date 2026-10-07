# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Tests for joining mx serve span timelines with client timings."""

from __future__ import annotations

import io
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

    def test_overlapping_requests_do_not_borrow_each_others_phases(self) -> None:
        child = [
            span(
                "B",
                "http.request",
                1,
                0,
                request_id="outer",
                route="/v1/chat/completions",
            ),
            span("B", "chat.prefill", 1, 1),
            span("E", "chat.prefill", 1, 4),
            span(
                "B",
                "http.request",
                2,
                5,
                request_id="other",
                route="/v1/chat/completions",
            ),
            span("B", "chat.prefill", 2, 6),
            span("E", "chat.prefill", 2, 8),
            span("B", "chat.decode_step", 2, 8),
            span("E", "chat.decode_step", 2, 9),
            span("E", "http.request", 2, 10),
            # Nested request on the same thread also must not leak phases.
            span(
                "B",
                "http.request",
                1,
                11,
                request_id="nested",
                route="/v1/chat/completions",
            ),
            span("B", "chat.prefill", 1, 12),
            span("E", "chat.prefill", 1, 13),
            span("E", "http.request", 1, 14),
            span("E", "http.request", 1, 20),
        ]
        rows = {row["request_id"]: row for row in mx_spans.request_timings([], child)}
        self.assertEqual(rows["outer"]["prefill_ms"], 3)
        self.assertEqual(rows["outer"]["decode_steps"], 0)
        self.assertEqual(rows["other"]["prefill_ms"], 2)
        self.assertEqual(rows["other"]["decode_steps"], 1)
        self.assertEqual(rows["nested"]["prefill_ms"], 1)

    def test_request_metadata_tracks_identity_and_omits_payloads(self) -> None:
        child = [
            span(
                "B", "http.request", 1, 0, request_id="a", route="/v1/chat/completions"
            ),
            span(
                "B",
                "chat.sampling",
                1,
                1,
                path="gpu_rule",
                mode="greedy",
                prompt="private",
                logits=[1, 2],
            ),
            span("E", "chat.sampling", 1, 2),
            span(
                "B", "http.request", 2, 1, request_id="b", route="/v1/chat/completions"
            ),
            span("B", "chat.sampling", 2, 2, path="full_row", mode="sampled"),
            span("E", "chat.sampling", 2, 3),
            span(
                "E",
                "http.request",
                2,
                4,
                **{"gen_ai.response.finish_reason": "cancelled"},
            ),
            span(
                "E",
                "http.request",
                1,
                5,
                **{
                    "gen_ai.request.max_tokens": 16,
                    "metallix.output_tokens.limit": 16,
                    "metallix.output_budget.source": "request",
                    "metallix.context_tokens.effective": 4608,
                    "gen_ai.response.finish_reason": "eos",
                },
            ),
            # Same-time diagnostic without a request ancestor cannot be joined.
            span("B", "chat.sampling", 3, 1, path="constrained_full_row"),
            span("E", "chat.sampling", 3, 2),
        ]
        proxy = [
            span("B", "http.request", 4, 0, request_id="b"),
            span("B", "proxy.forward", 4, 1),
            span(
                "E",
                "proxy.forward",
                4,
                6,
                **{
                    "metallix.context_tokens.requested": 4608,
                    "cancel.reason": "client_disconnected",
                    "cancel.cleanup": "ready",
                },
            ),
            # Missing outer E is permitted for historical recovery.
        ]
        rows = {
            r["request_id"]: r
            for r in mx_spans.request_timings(proxy, child, metadata=True)
        }
        self.assertEqual(rows["a"]["requested_max_tokens"], 16)
        self.assertEqual(rows["a"]["effective_max_tokens"], 16)
        self.assertEqual(rows["a"]["effective_context_tokens"], 4608)
        self.assertEqual(rows["a"]["budget_source"], "request")
        self.assertEqual(rows["a"]["finish_reason"], "eos")
        self.assertIsNone(rows["a"]["cancel_reason"])
        self.assertEqual(rows["b"]["cancel_reason"], "client_disconnected")
        self.assertEqual(rows["b"]["cancel_cleanup"], "ready")
        self.assertEqual(rows["b"]["requested_context_tokens"], 4608)
        self.assertIsNone(rows["b"]["requested_max_tokens"])
        self.assertEqual([s["path"] for s in rows["a"]["sampling"]], ["gpu_rule"])
        self.assertEqual([s["path"] for s in rows["b"]["sampling"]], ["full_row"])
        self.assertIsNone(rows["a"]["sampling"][0]["outcome"])
        self.assertIsNone(rows["a"]["sampling"][0]["rng_commit"])
        self.assertNotIn("private", json.dumps(rows))
        self.assertNotIn("logits", json.dumps(rows))
        # No prefill: default timing consumers still omit these diagnostic-only rows.
        self.assertEqual(mx_spans.request_timings(proxy, child), [])

    def test_sampling_outcome_does_not_imply_rng_commit(self) -> None:
        cases = [
            ("gpu_rule", "sampled", "peek", "not_committed"),
            ("gpu_candidates", "sampled", "fallback", "not_committed"),
            ("full_row", "sampled", "selected", "committed"),
            ("full_row", "sampled", "error", "not_committed"),
            ("full_row", "greedy", "selected", "not_committed"),
        ]
        events = [
            span(
                "B", "http.request", 1, 0, request_id="r", route="/v1/chat/completions"
            )
        ]
        for index, (path, mode, outcome, committed) in enumerate(cases):
            events.append(
                span(
                    "B",
                    "chat.sampling",
                    1,
                    index * 2 + 1,
                    path=path,
                    mode=mode,
                    commit_ordinal="unavailable",
                    legal_mass="not_collected",
                )
            )
            fields = {"outcome": outcome, "rng_commit": committed}
            if outcome == "fallback":
                fields["fallback_reason"] = "candidate_capacity"
            events.append(span("E", "chat.sampling", 1, index * 2 + 2, **fields))
        events.append(span("E", "http.request", 1, 12))
        (row,) = mx_spans.request_timings([], events, metadata=True)
        sampling = row["sampling"]
        self.assertEqual(
            [(s["path"], s["mode"], s["outcome"], s["rng_commit"]) for s in sampling],
            cases,
        )
        self.assertEqual([s["commit_ordinal"] for s in sampling], ["unavailable"] * 5)
        self.assertEqual([s["legal_mass"] for s in sampling], ["not_collected"] * 5)
        self.assertEqual(
            [s["fallback_reason"] for s in sampling],
            [None, "candidate_capacity", None, None, None],
        )
        self.assertTrue(all("compiled_reuse" not in s for s in sampling))

    def test_request_cli_reports_recovery_without_changing_default_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            front = pathlib.Path(directory) / "t.json"
            child = mx_spans.child_path(front, "test")
            front.write_text(json.dumps(PROXY))
            child.write_text(json.dumps(CHILD)[:-1])
            original = child.read_bytes()
            argv = ["mx_spans", str(front), "--model-id", "test"]
            output = io.StringIO()
            with mock.patch.object(sys, "argv", argv), mock.patch("sys.stdout", output):
                self.assertEqual(mx_spans.main(), 0)
            summary = mx_spans.summarize(mx_spans.request_timings(PROXY, CHILD))
            self.assertEqual(
                output.getvalue(),
                mx_spans.one_line(summary)
                + "\n"
                + json.dumps(summary, indent=1)
                + "\n",
            )
            output = io.StringIO()
            with (
                mock.patch.object(sys, "argv", [*argv, "--requests"]),
                mock.patch("sys.stdout", output),
            ):
                self.assertEqual(mx_spans.main(), 1)
            report = json.loads(output.getvalue())
            self.assertFalse(report["strict_artifacts_valid"])
            self.assertTrue(report["artifacts"][1]["recovered"])
            self.assertEqual(report["requests"][0]["request_id"], "r1")
            self.assertIsNone(report["requests"][0]["requested_max_tokens"])
            self.assertIsNone(report["requests"][0]["sampling"])
            self.assertEqual(child.read_bytes(), original)
            child.write_text(json.dumps(CHILD))
            self.assertTrue(
                mx_spans.request_report(front, "test")["strict_artifacts_valid"]
            )

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
