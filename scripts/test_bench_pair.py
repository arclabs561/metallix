# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Tests for the paired A/B driver: order, bootstrap, and drift cancellation.

The drift tests check the property the paired mode exists for: on a machine
whose speed changes during the comparison, the paired ratio's interval covers
the true ratio while a naive all-A-then-all-B comparison reports the wrong
direction. One test runs real stub servers whose speed drifts.
"""

from __future__ import annotations

import math
import pathlib
import random
import statistics
import sys
import tempfile
import time
import unittest
from unittest import mock

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import bench_load
import bench_pair

TRUE_RATIO = 1.25  # B's throughput over A's at equal machine state.


def drifting_machine(drift_per_run: float, noise: float, seed: int):
    """A fake measure: throughput 100 (A) or 125 (B), divided by a slowdown
    that grows by `drift_per_run` with every run, times seeded noise."""
    rng = random.Random(seed)
    runs = []

    def measure(arm: str) -> dict:
        slowdown = 1 + drift_per_run * len(runs)
        base = 100.0 if arm == "A" else 100.0 * TRUE_RATIO
        value = base / slowdown * rng.uniform(1 - noise, 1 + noise)
        runs.append(arm)
        return {
            "summary": {
                "output_token_throughput": value,
                "tpot_ms": {"p50": 1000 / value},
                "ttft_ms": {"p50": 20.0 * slowdown},
            },
            "baseline": {"load_1m": 10 + 20 * drift_per_run * len(runs)},
        }

    return measure, runs


class Tails(unittest.TestCase):
    def test_tail_metrics_are_compared_and_flagged_when_unresolved(self) -> None:
        self.assertIn("ttft_p90_ms", bench_pair.METRICS)
        self.assertIn("itl_p99_ms", bench_pair.METRICS)

        def run(scale: float) -> dict:
            ttft = bench_load.distribution([scale * v for v in range(1, 33)])
            itl = bench_load.distribution([scale * v for v in range(1, 61)])
            return {
                "summary": {
                    "output_token_throughput": 100.0 / scale,
                    "tpot_ms": {"p50": scale},
                    "ttft_ms": ttft,
                    "itl_ms": itl,
                }
            }

        pairs = [
            {"pair": i, "order": "AB", "A": run(1.0), "B": run(1.1)} for i in range(4)
        ]
        summary = bench_pair.summarize_pairs(pairs, random.Random(0))
        p90 = summary["metrics"]["ttft_p90_ms"]
        self.assertAlmostEqual(p90["median_ratio"], 1.1)
        self.assertIsNone(p90["unresolved_n"])  # 32 requests resolve p90.
        itl = summary["metrics"]["itl_p99_ms"]
        self.assertEqual(itl["unresolved_n"], 60)  # 60 gaps do not resolve p99.
        self.assertIn("compare the max, not p99", bench_pair.render(summary, pairs))


class RunLogs(unittest.TestCase):
    def test_each_run_gets_its_own_log_directory(self) -> None:
        logs = tempfile.TemporaryDirectory()
        self.addCleanup(logs.cleanup)
        seen = []

        def fake_measure_level(name, set_name, kind, value, count, args, counter):
            seen.append(args.log_dir)
            return {
                "summary": {
                    "output_token_throughput": 1.0,
                    "ttft_ms": {},
                    "tpot_ms": {},
                }
            }

        argv = [
            "bench_pair.py", "--server", "stub", "--model-path", ".",
            "--sets", "short", "--concurrency", "1", "--pairs", "2",
            "--cooldown", "0", "--log-dir", logs.name,
        ]  # fmt: skip
        with (
            mock.patch.object(sys, "argv", argv),
            mock.patch.object(
                bench_pair.bench_load, "measure_level", fake_measure_level
            ),
            mock.patch.object(
                bench_pair.bench_load, "tokenizer_counter", lambda path: len
            ),
        ):
            bench_pair.main()
        root = pathlib.Path(logs.name)
        self.assertEqual(
            seen,
            [root / "run00-A", root / "run01-B", root / "run02-B", root / "run03-A"],
        )


class LoadGuard(unittest.TestCase):
    def pair(self, index: int, load_a: float, load_b: float) -> dict:
        def run(load: float) -> dict:
            return {
                "summary": {
                    "output_token_throughput": 100.0,
                    "tpot_ms": {"p50": 10.0},
                    "ttft_ms": {"p50": 20.0},
                },
                "baseline": {"load_1m": load},
            }

        return {"pair": index, "order": "AB", "A": run(load_a), "B": run(load_b)}

    def test_pairs_above_the_declared_delta_are_dropped_and_counted(self) -> None:
        pairs = [
            self.pair(i, 4.0, 4.0 + delta)
            for i, delta in enumerate([0.2, 1.6, 0.9, 0.0])
        ]
        self.assertEqual(bench_pair.drop_load_swings(pairs, 1.0), 1)
        self.assertEqual(pairs[1]["dropped"], "load delta 1.6 > 1")
        summary = bench_pair.summarize_pairs(pairs, random.Random(0))
        self.assertEqual(summary["kept"], 3)
        self.assertNotIn("refused", summary)
        self.assertIn("output_tok_s", summary["metrics"])
        # No threshold, no guard.
        fresh = [self.pair(0, 1.0, 9.0)]
        self.assertEqual(bench_pair.drop_load_swings(fresh, None), 0)

    def test_more_than_a_quarter_dropped_refuses_the_result(self) -> None:
        pairs = [
            self.pair(i, 4.0, 4.0 + delta)
            for i, delta in enumerate([2.0, 2.0, 0.1, 0.1])
        ]
        bench_pair.drop_load_swings(pairs, 1.0)
        summary = bench_pair.summarize_pairs(pairs, random.Random(0))
        self.assertEqual(summary["metrics"], {})
        self.assertIn("2 of 4 pairs dropped", summary["refused"])
        summary |= {"max_load_delta": 1.0, "load_dropped": 2}
        text = bench_pair.render(summary, pairs)
        self.assertIn("RESULT REFUSED", text)
        self.assertIn("2 pairs dropped for a load delta above 1", text)
        # Exactly a quarter is still reported.
        pairs = [
            self.pair(i, 4.0, 4.0 + delta)
            for i, delta in enumerate([2.0, 0.1, 0.1, 0.1])
        ]
        bench_pair.drop_load_swings(pairs, 1.0)
        self.assertNotIn("refused", bench_pair.summarize_pairs(pairs, random.Random(0)))


class NoiseFloor(unittest.TestCase):
    def test_log_ratio_interval_by_hand(self) -> None:
        # log ratios +0.1 and -0.1: mean 0, sd 0.1 * sqrt(2), n 2, t(1) 12.706.
        out = bench_pair.log_ratio_interval([math.exp(0.1), math.exp(-0.1)])
        self.assertAlmostEqual(out["geomean_ratio"], 1.0)
        self.assertAlmostEqual(out["sd_log"], 0.141421, places=5)
        self.assertAlmostEqual(out["ci95"][0], math.exp(-1.2706), places=4)
        self.assertAlmostEqual(out["ci95"][1], math.exp(1.2706), places=4)
        self.assertAlmostEqual(out["mde"], math.exp(0.28) - 1, places=6)
        self.assertIsNone(bench_pair.log_ratio_interval([1.1])["ci95"])

    def test_t_quantiles_past_the_table_match_published_values(self) -> None:
        for df, published in ((40, 2.021), (60, 2.000), (120, 1.980)):
            self.assertAlmostEqual(bench_pair.t_975(df), published, delta=0.001)

    def test_pairs_favoring_b_follow_each_metrics_direction(self) -> None:
        def run(tok_s: float, ttft: float) -> dict:
            return {
                "summary": {
                    "output_token_throughput": tok_s,
                    "tpot_ms": {"p50": 10.0},
                    "ttft_ms": {"p50": ttft},
                }
            }

        # B: faster throughput in 3 of 4 pairs; lower TTFT in 1 of 4.
        values = [
            ((100, 50), (110, 60)),
            ((100, 50), (105, 40)),
            ((100, 50), (102, 55)),
            ((100, 50), (95, 55)),
        ]
        pairs = [
            {"pair": i, "order": "AB", "A": run(*a), "B": run(*b)}
            for i, (a, b) in enumerate(values)
        ]
        summary = bench_pair.summarize_pairs(pairs, random.Random(0))
        self.assertEqual(summary["metrics"]["output_tok_s"]["favor_b"], [3, 4])
        self.assertEqual(summary["metrics"]["ttft_p50_ms"]["favor_b"], [1, 4])
        text = bench_pair.render(summary, pairs)
        self.assertIn("3 of 4 pairs favor B", text)
        self.assertIn("MDE", text)


class Order(unittest.TestCase):
    def test_pairs_alternate_which_arm_goes_first(self) -> None:
        self.assertEqual(
            bench_pair.pair_order(4), [("A", "B"), ("B", "A"), ("A", "B"), ("B", "A")]
        )

    def test_bootstrap_interval_brackets_the_median_and_is_seeded(self) -> None:
        values = [1.2, 1.25, 1.3, 1.22, 1.27, 1.24, 1.26, 1.23]
        ci = bench_pair.bootstrap_median_ci(values, random.Random(1))
        self.assertEqual(ci, bench_pair.bootstrap_median_ci(values, random.Random(1)))
        self.assertLessEqual(ci[0], statistics.median(values))
        self.assertGreaterEqual(ci[1], statistics.median(values))
        self.assertGreater(ci[0], 1.19)
        self.assertLess(ci[1], 1.31)


class Drift(unittest.TestCase):
    def test_paired_ratio_covers_the_truth_under_drift(self) -> None:
        # The machine slows 5% per run: 80% slower by the end of 16 runs.
        measure, runs = drifting_machine(0.05, noise=0.02, seed=3)
        sleeps = []
        pairs = bench_pair.run_pairs(8, measure, sleeps.append, cooldown=1)
        self.assertEqual(len(sleeps), 15)  # Between runs, not before the first.
        summary = bench_pair.summarize_pairs(pairs, random.Random(0))
        tput = summary["metrics"]["output_tok_s"]
        low, high = tput["ci95"]
        self.assertLess(low, TRUE_RATIO)
        self.assertGreater(high, TRUE_RATIO)
        self.assertEqual(tput["verdict"], "B better")
        self.assertEqual(summary["metrics"]["tpot_p50_ms"]["verdict"], "B better")
        self.assertEqual("".join(runs), "ABBA" * 4)
        # Load is recorded per run of each pair.
        self.assertEqual(len(bench_pair.pair_load(pairs[0])), 2)

    def test_an_unpaired_comparison_under_the_same_drift_misleads(self) -> None:
        measure, _ = drifting_machine(0.05, noise=0.02, seed=3)
        a = [measure("A")["summary"]["output_token_throughput"] for _ in range(8)]
        b = [measure("B")["summary"]["output_token_throughput"] for _ in range(8)]
        unpaired = statistics.median(b) / statistics.median(a)
        # B is 25% faster, but measured after A on a slowing machine it looks
        # slower: the wrong direction, not just a wider error.
        self.assertLess(unpaired, 1.0)

    def test_a_pair_with_an_aborted_run_is_dropped_with_its_reason(self) -> None:
        measure, _ = drifting_machine(0.0, noise=0.0, seed=0)
        calls = []

        def flaky(arm: str) -> dict:
            calls.append(arm)
            if len(calls) == 2:
                return {"aborted": "1-min load 9.00 rose above 4"}
            return measure(arm)

        # Four pairs, so the one dropped stays within the 25% refusal limit.
        pairs = bench_pair.run_pairs(4, flaky, lambda s: None, 0)
        summary = bench_pair.summarize_pairs(pairs, random.Random(0))
        self.assertEqual(pairs[0]["dropped"], "B: 1-min load 9.00 rose above 4")
        self.assertEqual((summary["pairs"], summary["kept"]), (4, 3))
        self.assertAlmostEqual(summary["metrics"]["output_tok_s"]["median_ratio"], 1.25)


class StubServers(unittest.TestCase):
    def test_drifting_stub_servers_end_to_end(self) -> None:
        # Arm A is the 2 ms/token stub and arm B the 10 ms/token one; both slow
        # down from the same moment, as on a machine growing busier.
        logs = tempfile.TemporaryDirectory()
        self.addCleanup(logs.cleanup)
        epoch = time.time()
        shared = [
            "--model-path", ".", "--sets", "short", "--concurrency", "2",
            "--max-tokens", "8", "--warmup", "0", "--log-dir", logs.name,
            "--start-timeout", "30", "--request-timeout", "30",
            "--server-arg=--drift-per-s", "--server-arg=0.3",
            f"--server-arg=--drift-epoch={epoch}",
        ]  # fmt: skip
        arms = {
            "A": bench_pair.arm_args(shared, "--server stub"),
            "B": bench_pair.arm_args(shared, "--server stub-slow"),
        }
        self.assertEqual(arms["A"].server_args[-1], f"--drift-epoch={epoch}")

        def measure(arm: str) -> dict:
            args = arms[arm]
            return bench_load.measure_level(
                args.server, "short", "concurrency", 2, 4, args, None
            )

        pairs = bench_pair.run_pairs(4, measure, lambda s: None, 0)
        summary = bench_pair.summarize_pairs(pairs, random.Random(0))
        self.assertEqual(summary["kept"], 4)
        # Every run left its own server log and a journal of its requests.
        journals = [pair[arm]["requests_log"] for pair in pairs for arm in "AB"]
        self.assertEqual(len(set(journals)), 8)
        for journal in journals:
            lines = pathlib.Path(journal).read_text().splitlines()
            self.assertEqual(len(lines), 4)
        tput = summary["metrics"]["output_tok_s"]
        # A fifth of the token rate, diluted by the fixed 20 ms first-token
        # delay. The wider gap tolerates more shared scheduler delay, though
        # sufficiently large overhead or asymmetric drift can still fail it.
        self.assertLess(tput["ci95"][1], 0.9)
        self.assertEqual(tput["verdict"], "A better")


if __name__ == "__main__":
    unittest.main()
