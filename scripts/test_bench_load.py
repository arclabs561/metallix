# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""CPU-only tests for the load benchmark's stream parsing and metric math."""

from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
import pathlib
import random
import sys
import tempfile
import time
import unittest
from unittest import mock

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


# Shaped like an entry of SiliconBench's prompts/agent_benchmark_prompts.json
# (github.com/WindChimeRan/SiliconBench at 616aa51c): pre-baked tool turns with
# empty assistant content, string arguments and a per-prompt output cap.
AGENT_ENTRY = {
    "name": "a000_vlong_bfcl_v3",
    "description": "BFCL multi-turn",
    "max_tokens": 256,
    "messages": [
        {"role": "system", "content": "You have tools."},
        {"role": "user", "content": "Touch notes.txt."},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_0_0",
                    "type": "function",
                    "function": {"name": "touch", "arguments": '{"file_name": "a"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_0_0", "content": '{"ok": true}'},
        {"role": "user", "content": "Now list the directory."},
    ],
}


class PromptsFile(unittest.TestCase):
    def write(self, text: str) -> pathlib.Path:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = pathlib.Path(directory.name) / "prompts.json"
        path.write_text(text)
        return path

    def test_json_array_and_json_lines_load_in_file_order(self) -> None:
        plain = {"name": "p1", "system": "s", "user": "u"}
        for text in (
            json.dumps([AGENT_ENTRY, plain]),
            json.dumps(AGENT_ENTRY) + "\n\n" + json.dumps(plain) + "\n",
        ):
            prompts = bench_load.load_prompts_file(self.write(text))
            self.assertEqual([p.label for p in prompts], ["a000_vlong_bfcl_v3", "p1"])
            self.assertEqual(prompts[0].max_tokens, 256)
            self.assertEqual(len(prompts[0].messages), 5)
            self.assertEqual((prompts[1].system, prompts[1].user), ("s", "u"))
            self.assertIsNone(prompts[1].messages)

    def test_bad_entries_are_rejected(self) -> None:
        for entry in ({"name": "x"}, {"user": "u", "max_tokens": 0}):
            with self.assertRaises(ValueError):
                bench_load.load_prompts_file(self.write(json.dumps([entry])))
        with self.assertRaises(ValueError):
            bench_load.load_prompts_file(self.write("[]"))

    def test_chat_sends_recorded_messages_with_the_prompt_cap(self) -> None:
        prompt = bench_load.load_prompts_file(self.write(json.dumps([AGENT_ENTRY])))[0]
        path, body = bench_load.request_body("chat", "m", prompt, 128, {})
        self.assertEqual(path, "/v1/chat/completions")
        self.assertEqual(body["messages"], AGENT_ENTRY["messages"])
        self.assertEqual(body["max_tokens"], 256)

    def test_responses_turns_tool_calls_into_items(self) -> None:
        prompt = bench_load.load_prompts_file(self.write(json.dumps([AGENT_ENTRY])))[0]
        path, body = bench_load.request_body("responses", "m", prompt, 128, {})
        self.assertEqual(path, "/v1/responses")
        self.assertNotIn("instructions", body)
        self.assertEqual(body["max_output_tokens"], 256)
        self.assertEqual(
            [item["type"] for item in body["input"]],
            ["message", "message", "function_call", "function_call_output", "message"],
        )
        call, output = body["input"][2], body["input"][3]
        self.assertEqual(
            (call["call_id"], call["name"], call["arguments"]),
            ("call_0_0", "touch", '{"file_name": "a"}'),
        )
        self.assertEqual(
            (output["call_id"], output["output"]), ("call_0_0", '{"ok": true}')
        )

    def test_cells_wrap_and_warmup_comes_from_the_tail(self) -> None:
        prompts = [bench_load.Prompt(f"p{i}", None, str(i)) for i in range(4)]
        cell = bench_load.set_prompts("file", 6, None, 0, prompts)
        self.assertEqual([p.label for p in cell], ["p0", "p1", "p2", "p3", "p0", "p1"])
        warm = bench_load.warmup_prompts("file", 2, None, 0, prompts)
        self.assertEqual([p.label for p in warm], ["p3", "p2"])


class Levels(unittest.TestCase):
    def args(self, **overrides):
        values = {
            "url": None,
            "api": "chat",
            "model_path": pathlib.Path("model"),
            "model_id": "m",
            "prefix_cache": "off",
            "seed": 0,
            "warmup": 3,
            "file_prompts": [],
            "max_tokens": 8,
            "request_timeout": 5,
            "start_timeout": 5,
            "slo_ttft_ms": 2000,
            "slo_tpot_ms": 50,
            "log_dir": pathlib.Path(tempfile.gettempdir()),
            "concurrency": [1, 4],
            "rates": [2.0],
            "min_requests": 2,
            "requests_per_slot": 1,
            "rate_requests": 2,
            "abort_load": None,
            "abort_gpu_gib": None,
        }
        return argparse.Namespace(**(values | overrides))

    def fakes(self, load: float) -> tuple[list, list]:
        """Patch servers, requests and machine probes; return (starts, warmups)."""
        started, warmups = [], []

        class FakeServer:
            def __init__(self, spec, address, log_path):
                started.append(spec.argv)
                self.ready_s = 0.1
                self.process = argparse.Namespace(pid=4242)

            def wait_ready(self, timeout):
                pass

            def stop(self):
                started.append("stopped")

        def fake_run_load(address, api, model, prompts, max_tokens, extra, **kw):
            if kw.get("rate") is None and len(prompts) == 3:
                warmups.append(kw["concurrency"])
            time.sleep(0.05)  # Long enough for the sampler's first sample.
            records = [record(index=i) for i in range(len(prompts))]
            return records, 1.0

        def fake_probe(pgid):
            self.assertEqual(pgid, 4242)
            return {"load_1m": load, "system_used_bytes": 3 * 2**30}

        sampler = bench_load.bench_system.Sampler

        spec = bench_load.ServerSpec("vllm-metal", "chat", ["vllm"], {}, {})
        patches = [
            mock.patch.object(bench_load, "ManagedServer", FakeServer),
            mock.patch.object(bench_load, "run_load", fake_run_load),
            mock.patch.object(bench_load, "server_spec", lambda *a: spec),
            mock.patch.object(bench_load.bench_serve, "free_address", lambda: "h:1"),
            mock.patch.object(
                bench_load.bench_system,
                "Sampler",
                lambda **kw: sampler(probe=fake_probe, **kw),
            ),
        ]
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)
        return started, warmups

    def test_every_level_starts_its_own_server_and_warms_at_its_concurrency(
        self,
    ) -> None:
        started, warmups = self.fakes(load=1.0)
        count = lambda text: len(text.split())
        out = bench_load.measure_set("vllm-metal", "short", self.args(), count)
        # c=1, c=4 and one rate: three servers, each stopped before the next.
        argv = ["vllm", "--no-enable-prefix-caching"]
        self.assertEqual(started, [argv, "stopped"] * 3)
        self.assertEqual(warmups, [1, 4, 1])
        self.assertTrue(all(run["restarted"] for run in out["concurrency"]))
        self.assertEqual(out["goodput_rps"], 2.0)
        memory = out["concurrency"][0]["memory"]
        self.assertEqual(memory["peak_system_used_bytes"], 3 * 2**30)
        self.assertIsNone(memory["aborted"])

    def test_load_above_the_abort_threshold_stops_the_level(self) -> None:
        started, _ = self.fakes(load=4.5)
        args = self.args(concurrency=[2], rates=[], abort_load=4.0)
        run = bench_load.measure_set("vllm-metal", "short", args, str.split)[
            "concurrency"
        ][0]
        self.assertIn("rose above 4", run["aborted"])
        self.assertNotIn("summary", run)  # Cut-off requests are not engine failures.
        self.assertEqual(started.count("stopped"), 2)  # The abort, then cleanup.

    def test_every_managed_engine_is_measured_on_chat_completions(self) -> None:
        args = bench_load.build_parser().parse_args(
            ["--server", "metallix", "--model-path", "model"]
        )
        for name in ("metallix", "vllm-metal", "mtplx", "mlx-lm"):
            with mock.patch.object(bench_load, "env_python_versions", lambda *a: {}):
                spec = bench_load.server_spec(name, pathlib.Path("m"), "m", "h:1", args)
            self.assertEqual(spec.api, "chat", name)

    def test_prefix_cache_arms_per_server(self) -> None:
        spec = lambda name: bench_load.ServerSpec(name, "chat", [name], {}, {})
        on = bench_load.with_prefix_cache(spec("vllm-metal"), "on")
        self.assertEqual(on.argv, ["vllm-metal", "--enable-prefix-caching"])
        off = bench_load.with_prefix_cache(spec("metallix"), "off")
        self.assertEqual(off.argv, ["metallix", "--prefix-cache-mib", "0"])
        off = bench_load.with_prefix_cache(spec("mlx-lm"), "off")
        self.assertEqual(off.argv, ["mlx-lm", "--prompt-cache-size", "0"])
        self.assertEqual(
            bench_load.with_prefix_cache(spec("mtplx"), "default").argv, ["mtplx"]
        )
        with self.assertRaises(ValueError):
            bench_load.with_prefix_cache(spec("mtplx"), "off")


class TokenCounts(unittest.TestCase):
    @staticmethod
    def records(counts: list[int | None]) -> list[dict]:
        return [
            {
                "index": i,
                "outcome": "ok" if n is not None else "http_503",
                "output_tokens": n,
            }
            for i, n in enumerate(counts)
        ]

    def test_differing_counts_on_shared_prompts_warn(self) -> None:
        cells = {
            ("short", "c", 4): {
                "metallix": self.records([128, 128, 128]),
                "mlx-lm": self.records([128, 90, None]),
            }
        }
        (warning,) = bench_load.token_count_warnings(cells)
        self.assertIn("short c=4", warning)
        # Prompt 2 failed on mlx-lm, so only prompts 0 and 1 compare.
        self.assertIn("differ on 1/2 prompts", warning)
        self.assertIn("metallix 256, mlx-lm 218", warning)

    def test_equal_counts_and_single_engines_are_quiet(self) -> None:
        same = self.records([128, 128])
        cells = {
            ("short", "c", 1): {"a": same, "b": same},
            ("long", "c", 1): {"a": self.records([5])},
        }
        self.assertEqual(bench_load.token_count_warnings(cells), [])

    def test_failures_list_prompt_labels_per_engine(self) -> None:
        records = self.records([128, None, 128, None])
        for r, label in zip(records, ["a000", "a003_vlong_hermes", "a004", "a007"]):
            r["label"] = label
        records[3]["outcome"] = "error"
        cells = {
            ("file", "c", 8): {"metallix": records, "vllm-metal": self.records([1])}
        }
        self.assertEqual(
            bench_load.failure_warnings(cells),
            [
                (
                    "metallix file c=8: 2/4 failed: "
                    "a003_vlong_hermes (http_503), a007 (error)"
                )
            ],
        )

    def test_cells_come_from_measured_levels_only(self) -> None:
        run = {"concurrency": 2, "summary": {}, "records": self.records([1])}
        servers = [
            {
                "name": "a",
                "sets": [
                    {
                        "set": "short",
                        "concurrency": [run, {"concurrency": 4, "error": "x"}],
                        "rates": [],
                    }
                ],
            },
            {"name": "b", "error": "not started"},
        ]
        self.assertEqual(
            bench_load.report_cells(servers), {("short", "c", 2): {"a": run["records"]}}
        )


class Metadata(unittest.TestCase):
    def test_version_pairs(self) -> None:
        self.assertEqual(
            bench_load.version_pair("llama.cpp=7049ff0"), ("llama.cpp", "7049ff0")
        )
        self.assertEqual(bench_load.version_pair("a=b=c"), ("a", "b=c"))
        for bad in ("nover", "=1"):
            with self.assertRaises(argparse.ArgumentTypeError):
                bench_load.version_pair(bad)


if __name__ == "__main__":
    unittest.main()
