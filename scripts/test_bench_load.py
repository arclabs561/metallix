# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""CPU-only tests for the load benchmark's stream parsing and metric math."""

from __future__ import annotations

import importlib.util
import itertools
import json
import pathlib
import random
import sys
import unittest

SCRIPTS = pathlib.Path(__file__).parent
sys.path.insert(0, str(SCRIPTS))  # bench_load imports its sibling bench_serve.
SPEC = importlib.util.spec_from_file_location("bench_load", SCRIPTS / "bench_load.py")
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot load bench_load.py")
bench_load = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = bench_load
SPEC.loader.exec_module(bench_load)


def chat_chunk(content: str | None = None, usage: dict | None = None) -> str:
    event = {
        "choices": [{"delta": {"content": content}}] if content is not None else []
    }
    if usage:
        event["usage"] = usage
    return "data: " + json.dumps(event)


def responses_event(kind: str, **fields) -> str:
    return "data: " + json.dumps({"type": kind, **fields})


class ChatStream(unittest.TestCase):
    def test_token_times_and_usage_from_synthetic_sse(self) -> None:
        # Send at t=10.0; role chunk, then three content chunks 20 ms apart.
        state = bench_load.StreamState("chat")
        state.feed(": keep-alive", 10.05)
        state.feed('data: {"choices":[{"delta":{"role":"assistant"}}]}', 10.10)
        state.feed(chat_chunk("Hello"), 10.25)
        state.feed(chat_chunk(" there"), 10.27)
        state.feed(chat_chunk("!"), 10.29)
        state.feed(
            chat_chunk(usage={"prompt_tokens": 41, "completion_tokens": 3}), 10.30
        )
        state.feed("data: [DONE]", 10.30)
        self.assertEqual(state.chunk_times, [10.25, 10.27, 10.29])
        self.assertEqual((state.input_tokens, state.output_tokens), (41, 3))
        self.assertTrue(state.done)
        t = bench_load.timings(10.0, state.chunk_times, 10.31, state.output_tokens)
        self.assertAlmostEqual(t["ttft_ms"], 250.0)
        self.assertAlmostEqual(t["e2e_ms"], 310.0)
        # (310 - 250) ms over the 2 tokens after the first.
        self.assertAlmostEqual(t["tpot_ms"], 30.0)
        self.assertEqual([round(g, 6) for g in t["itl_ms"]], [20.0, 20.0])

    def test_reasoning_deltas_count_as_tokens(self) -> None:
        state = bench_load.StreamState("chat")
        state.feed('data: {"choices":[{"delta":{"reasoning_content":"Hm"}}]}', 1.0)
        self.assertEqual(state.chunk_times, [1.0])

    def test_error_event_is_recorded(self) -> None:
        state = bench_load.StreamState("chat")
        state.feed('data: {"error":{"message":"overloaded"}}', 1.0)
        self.assertIn("overloaded", state.error)


class ResponsesStream(unittest.TestCase):
    def test_metallix_style_events(self) -> None:
        state = bench_load.StreamState("responses")
        state.feed("event: response.created", 0.1)
        state.feed(responses_event("response.created", response={"id": "r"}), 0.1)
        state.feed(responses_event("response.output_text.delta", delta="A"), 0.5)
        state.feed(responses_event("response.output_text.delta", delta="B"), 0.6)
        state.feed(
            responses_event(
                "response.completed",
                response={"usage": {"input_tokens": 9, "output_tokens": 2}},
            ),
            0.6,
        )
        self.assertEqual(state.chunk_times, [0.5, 0.6])
        self.assertEqual((state.input_tokens, state.output_tokens), (9, 2))
        self.assertTrue(state.done)

    def test_failed_response_is_an_error(self) -> None:
        state = bench_load.StreamState("responses")
        state.feed(
            responses_event(
                "response.failed",
                response={"error": {"code": "generation_timeout"}},
            ),
            1.0,
        )
        self.assertIn("generation_timeout", state.error)
        self.assertFalse(state.done)


class Timings(unittest.TestCase):
    def test_chunk_count_stands_in_without_usage(self) -> None:
        # Four chunks, no usage: TPOT = (e2e - ttft) / 3.
        t = bench_load.timings(0.0, [0.1, 0.2, 0.3, 0.4], 0.4, None)
        self.assertAlmostEqual(t["tpot_ms"], 100.0)

    def test_multi_token_chunks_use_reported_tokens(self) -> None:
        # Two chunks carrying 9 tokens: TPOT spreads decode time over 8 tokens.
        t = bench_load.timings(0.0, [0.1, 0.9], 0.9, 9)
        self.assertAlmostEqual(t["tpot_ms"], 100.0)

    def test_single_token_has_no_tpot_and_no_tokens_no_ttft(self) -> None:
        self.assertIsNone(bench_load.timings(0.0, [0.2], 0.2, 1)["tpot_ms"])
        empty = bench_load.timings(0.0, [], 0.5, None)
        self.assertIsNone(empty["ttft_ms"])
        self.assertAlmostEqual(empty["e2e_ms"], 500.0)


def record(outcome="ok", ttft=100.0, tpot=20.0, out=10, inp=50, index=0):
    return bench_load.Record(
        index=index,
        label="x",
        outcome=outcome,
        send_s=0.0,
        ttft_ms=ttft if outcome == "ok" else None,
        tpot_ms=tpot if outcome == "ok" else None,
        e2e_ms=1000.0,
        output_tokens=out if outcome == "ok" else None,
        input_tokens=inp if outcome == "ok" else None,
    )


class Metrics(unittest.TestCase):
    def test_percentile_matches_linear_interpolation(self) -> None:
        values = [float(v) for v in range(1, 101)]  # 1..100
        self.assertAlmostEqual(bench_load.percentile(values, 50), 50.5)
        self.assertAlmostEqual(bench_load.percentile(values, 90), 90.1)
        self.assertAlmostEqual(bench_load.percentile(values, 99), 99.01)
        self.assertEqual(bench_load.percentile([7.0], 99), 7.0)
        self.assertIsNone(bench_load.percentile([], 50))

    def test_slo_needs_both_objectives_and_success(self) -> None:
        meets = lambda r: bench_load.meets_slo(r, ttft_slo_ms=200, tpot_slo_ms=50)
        self.assertTrue(meets(record(ttft=200, tpot=50)))
        self.assertFalse(meets(record(ttft=201, tpot=10)))
        self.assertFalse(meets(record(ttft=10, tpot=51)))
        self.assertFalse(meets(record(outcome="http_503")))
        self.assertTrue(meets(record(ttft=10, tpot=None)))

    def test_summary_counts_failures_against_attainment_and_not_throughput(
        self,
    ) -> None:
        records = [record(index=i) for i in range(6)]
        records += [record(ttft=5000.0, index=6), record(outcome="http_503", index=7)]
        s = bench_load.summarize(
            records, duration_s=2.0, ttft_slo_ms=200, tpot_slo_ms=50
        )
        self.assertEqual(s["outcomes"], {"ok": 7, "http_503": 1})
        self.assertAlmostEqual(s["slo_attainment"], 6 / 8)
        self.assertAlmostEqual(s["request_throughput"], 7 / 2)
        self.assertAlmostEqual(s["output_token_throughput"], 70 / 2)
        self.assertAlmostEqual(s["slo_request_rate"], 6 / 2)
        self.assertEqual(s["ttft_ms"]["n"], 7)


class Goodput(unittest.TestCase):
    @staticmethod
    def attainment(rate: float, capacity: float, rng: random.Random, n: int = 400):
        """Share of n requests meeting a 1 s TTFT SLO under a synthetic M/M/1-like
        queue: sojourn times are exponential with mean 1 / (capacity - rate)."""
        if rate >= capacity:
            return 0.0
        mean = 1 / (capacity - rate)
        return sum(rng.expovariate(1 / mean) <= 1.0 for _ in range(n)) / n

    def test_goodput_from_synthetic_latency_distributions(self) -> None:
        # P(sojourn <= 1 s) = 1 - exp(-(capacity - rate)); >= 0.9 needs
        # capacity - rate >= ln 10 = 2.30. With capacity 6, rates up to 3.5 pass.
        rng = random.Random(7)
        rates = [1.0, 2.0, 3.0, 3.5, 4.0, 5.0]
        sweep = {rate: self.attainment(rate, 6.0, rng, n=4000) for rate in rates}
        self.assertGreater(sweep[3.5], 0.9)
        self.assertLess(sweep[4.0], 0.9)
        self.assertEqual(bench_load.goodput(sweep), 3.5)

    def test_goodput_stops_at_first_miss_and_handles_none(self) -> None:
        self.assertEqual(bench_load.goodput({1: 0.95, 2: 0.80, 4: 0.92}), 1)
        self.assertIsNone(bench_load.goodput({1: 0.5, 2: 0.4}))
        self.assertEqual(bench_load.goodput({0.5: 0.9}), 0.5)  # The goal is inclusive.


class Arrivals(unittest.TestCase):
    def test_poisson_offsets_are_seeded_and_average_one_over_rate(self) -> None:
        a = bench_load.poisson_arrivals(4.0, 2000, random.Random(3))
        self.assertEqual(a, bench_load.poisson_arrivals(4.0, 2000, random.Random(3)))
        self.assertEqual(a[0], 0.0)
        self.assertTrue(all(x < y for x, y in itertools.pairwise(a)))
        mean_gap = a[-1] / (len(a) - 1)
        self.assertAlmostEqual(mean_gap, 0.25, delta=0.02)


class Prompts(unittest.TestCase):
    count = staticmethod(lambda text: len(text.split()))  # One token per word.

    def test_shared_prefix_shares_system_and_long_prompts_differ(self) -> None:
        shared = bench_load.build_prompts("shared-prefix", 4, self.count, seed=0)
        self.assertEqual(len({p.system for p in shared}), 1)
        self.assertGreaterEqual(self.count(shared[0].system), 2048)
        self.assertEqual(len({p.user for p in shared}), 4)
        long = bench_load.build_prompts("long", 4, self.count, seed=0)
        sizes = [self.count(p.user) - self.count(bench_load.CONTINUE) for p in long]
        self.assertEqual(sizes, [2048, 4096, 6144, 8000])
        self.assertEqual(len({p.user[:200] for p in long}), 4)

    def test_mixed_interleaves_kinds(self) -> None:
        mixed = bench_load.build_prompts("mixed", 5, self.count, seed=0)
        self.assertEqual(
            [p.label.split("-")[0] for p in mixed],
            ["short", "shared", "short", "long", "short"],
        )

    def test_request_bodies_per_api(self) -> None:
        prompt = bench_load.Prompt("x", "sys", "hi")
        path, body = bench_load.request_body(
            "chat", "m", prompt, 64, {"ignore_eos": True}
        )
        self.assertEqual(path, "/v1/chat/completions")
        self.assertEqual([m["role"] for m in body["messages"]], ["system", "user"])
        self.assertTrue(body["stream"] and body["ignore_eos"])
        path, body = bench_load.request_body("responses", "m", prompt, 64, {})
        self.assertEqual(path, "/v1/responses")
        self.assertEqual((body["instructions"], body["input"]), ("sys", "hi"))
        self.assertEqual(body["max_output_tokens"], 64)


if __name__ == "__main__":
    unittest.main()
