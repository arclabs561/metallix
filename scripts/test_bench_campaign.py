# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Tests for the benchmark campaign driver: plan, gate, claim rule, summaries.

One test starts the model-free stub server and measures it for real; the rest
use fakes and need no network.
"""

from __future__ import annotations

import pathlib
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import bench_campaign
import bench_load


class Claims(unittest.TestCase):
    # The rule as decided for public claims: "faster" means the subject's worst
    # pass beats the competitor's best; "significantly" also needs a median
    # ratio >= 1.5; parity means medians within 10%.

    def test_significantly_faster_needs_separation_and_ratio(self) -> None:
        c = bench_campaign.claim([300, 310, 320], [190, 200, 205], True)
        self.assertEqual(c["verdict"], "significantly faster")
        self.assertAlmostEqual(c["ratio"], 1.55)
        self.assertFalse(c["parity"])

    def test_separated_but_below_the_ratio_is_only_faster(self) -> None:
        c = bench_campaign.claim([210, 212, 215], [200, 201, 205], True)
        self.assertEqual(c["verdict"], "faster")
        self.assertTrue(c["parity"])

    def test_a_big_median_gap_with_overlapping_passes_is_no_claim(self) -> None:
        # Median ratio 2.0, but the subject's worst pass (150) is below the
        # competitor's best (200).
        c = bench_campaign.claim([150, 400, 420], [100, 200, 200], True)
        self.assertEqual(c["verdict"], "overlapping")
        self.assertAlmostEqual(c["ratio"], 2.0)

    def test_latency_is_lower_is_better(self) -> None:
        c = bench_campaign.claim([5.0, 5.2, 5.1], [9.0, 8.9, 9.4], False)
        self.assertEqual(c["verdict"], "significantly faster")
        self.assertAlmostEqual(c["ratio"], 9.0 / 5.1)
        c = bench_campaign.claim([9.0, 8.9, 9.4], [5.0, 5.2, 5.1], False)
        self.assertEqual(c["verdict"], "slower")

    def test_fewer_than_three_passes_is_insufficient(self) -> None:
        c = bench_campaign.claim([300, 310], [100, 100, 100], True)
        self.assertEqual(c["verdict"], "insufficient passes")
        self.assertIsNone(c["ratio"])

    def test_spread_reports_median_range_and_deviation(self) -> None:
        s = bench_campaign.spread([100.0, 104.0, 95.0])
        self.assertEqual(
            (s["median"], s["min"], s["max"], s["n"]), (100.0, 95.0, 104.0, 3)
        )
        self.assertAlmostEqual(s["max_deviation"], 0.05)


class Plan(unittest.TestCase):
    def test_orders_are_seeded_shuffled_per_pass_and_cover_every_arm(self) -> None:
        engines = ["metallix", "vllm-metal", "mlx-lm"]
        arms, orders, skipped = bench_campaign.plan(
            engines, ["short"], ["on", "off"], [1, 4], passes=3, seed=11
        )
        again = bench_campaign.plan(engines, ["short"], ["on", "off"], [1, 4], 3, 11)
        self.assertEqual((arms, orders), again[:2])
        self.assertEqual(len(arms), 3 * 3 * 2 * 2)
        self.assertEqual(skipped, [])
        for number, order in enumerate(orders, start=1):
            self.assertEqual(sorted(order), sorted(engines))
            ran = [a["engine"] for a in arms if a["pass"] == number]
            # Each engine's arms run together, in the pass's order.
            self.assertEqual(list(dict.fromkeys(ran)), order)
        self.assertGreater(len({tuple(o) for o in orders}), 1)

    def test_an_engine_without_a_cache_switch_runs_only_the_default_arm(self) -> None:
        arms, _, skipped = bench_campaign.plan(
            ["mtplx", "metallix"], ["short"], ["default", "off"], [1], 1, 0
        )
        self.assertEqual(
            sorted((a["engine"], a["cache"]) for a in arms),
            [("metallix", "default"), ("metallix", "off"), ("mtplx", "default")],
        )
        self.assertEqual(skipped, ["mtplx cache=off: no prefix-cache switch"])


class Driver(unittest.TestCase):
    def test_cooldown_between_arms_and_a_failed_gate_skips_the_arm(self) -> None:
        events = []
        gates = iter(
            [
                {"passed": True, "failures": []},
                {"passed": False, "failures": ["running: cargo"]},
                {"passed": True, "failures": []},
            ]
        )
        arms = [
            {"pass": 1, "engine": e, "set": "s", "cache": "default", "concurrency": 1}
            for e in ("a", "b", "c")
        ]

        def measure(arm):
            events.append(("measure", arm["engine"]))
            return {"summary": {}}

        bench_campaign.run_campaign(
            arms,
            measure,
            lambda: next(gates),
            lambda s: events.append(("sleep", s)),
            60,
        )
        self.assertEqual(
            events,
            [("measure", "a"), ("sleep", 60), ("sleep", 60), ("measure", "c")],
        )
        self.assertEqual(arms[1]["skipped"], "not idle: running: cargo")
        self.assertNotIn("result", arms[1])

    def test_no_gate_marks_arms_not_passed(self) -> None:
        arms = [{"pass": 1, "engine": "a", "set": "s", "cache": "on", "concurrency": 2}]
        bench_campaign.run_campaign(arms, lambda arm: {}, None, lambda s: None, 0)
        self.assertEqual(arms[0]["idle_gate"], {"passed": False, "skipped": True})

    def test_wait_for_idle_polls_until_idle_or_timeout(self) -> None:
        loads = iter([5.0, 3.0, 1.0])
        sleeps = []
        result = bench_campaign.wait_for_idle(
            lambda: {"load_1m": next(loads)},
            lambda r: [] if r["load_1m"] < 2 else ["busy"],
            sleeps.append,
            timeout=600,
            poll=30,
        )
        self.assertTrue(result["passed"])
        self.assertEqual(
            (result["attempts"], result["waited_s"], sleeps), (3, 60, [30, 30])
        )
        result = bench_campaign.wait_for_idle(
            lambda: {"load_1m": 9.0}, lambda r: ["busy"], lambda s: None, 60, 30
        )
        self.assertFalse(result["passed"])
        self.assertEqual((result["attempts"], result["failures"]), (3, ["busy"]))


def row(engine, value, number, tokens=128, grade=True):
    return {
        "engine": engine,
        "set": "shared-prefix",
        "cache": "on",
        "concurrency": 8,
        "pass": f"pass {number}",
        "summary": {
            "output_token_throughput": value,
            "ttft_ms": {"p50": 100.0},
            "tpot_ms": {"p50": 10.0},
        },
        "records": [{"index": 0, "outcome": "ok", "output_tokens": tokens}],
        "claim_grade": grade,
    }


class Summaries(unittest.TestCase):
    def test_claims_compare_the_subject_with_each_competitor(self) -> None:
        rows = [row("metallix", v, n) for n, v in enumerate([600, 610, 620], 1)]
        rows += [row("vllm-metal", v, n) for n, v in enumerate([390, 400, 405], 1)]
        summary = bench_campaign.summarize_rows(rows)
        (cell,) = summary["cells"]
        claims = cell["claims"]["vllm-metal"]
        self.assertTrue(claims["claim_grade"])
        self.assertEqual(claims["output_tok_s"]["verdict"], "significantly faster")
        self.assertTrue(claims["ttft_p50_ms"]["parity"])
        self.assertEqual(summary["warnings"], [])
        self.assertIn("metallix vs vllm-metal", bench_campaign.render(summary))

    def test_unequal_tokens_noisy_passes_and_ungated_rows_warn(self) -> None:
        rows = [row("metallix", v, n) for n, v in enumerate([600, 700, 610], 1)]
        rows += [row("mlx-lm", 400, n, tokens=90, grade=n != 2) for n in (1, 2, 3)]
        summary = bench_campaign.summarize_rows(rows)
        warnings = "\n".join(summary["warnings"])
        self.assertIn(
            "metallix shared-prefix/on c=8: throughput passes deviate", warnings
        )
        self.assertEqual(warnings.count("output tokens differ"), 3)  # Once per pass.
        self.assertIn(
            "1 engine cells include passes without a passed idle gate", warnings
        )
        self.assertFalse(summary["cells"][0]["claims"]["mlx-lm"]["claim_grade"])

    def test_load_reports_count_as_one_pass_graded_by_their_idle_record(self) -> None:
        report = {
            "idle_gate": {"readings": {}, "failures": []},
            "servers": [
                {
                    "name": "llama.cpp",
                    "prefix_cache": "default",
                    "sets": [
                        {
                            "set": "file",
                            "concurrency": [
                                {"concurrency": 8, "summary": {"x": 1}, "records": []},
                                {"concurrency": 16, "error": "refused"},
                            ],
                            "rates": [],
                        }
                    ],
                }
            ],
        }
        (only,) = bench_campaign.rows_from_load_report(report, "run1.json")
        self.assertEqual(
            (only["engine"], only["concurrency"], only["pass"]),
            ("llama.cpp", 8, "run1.json"),
        )
        self.assertTrue(only["claim_grade"])
        del report["idle_gate"]
        self.assertFalse(
            bench_campaign.rows_from_load_report(report, "r")[0]["claim_grade"]
        )


class StubEndToEnd(unittest.TestCase):
    def test_measure_a_fresh_stub_server(self) -> None:
        logs = tempfile.TemporaryDirectory()
        self.addCleanup(logs.cleanup)
        parser = bench_load.build_parser()
        args = parser.parse_args(
            [
                "--server", "stub", "--model-path", ".", "--sets", "short",
                "--max-tokens", "4", "--warmup", "1", "--start-timeout", "30",
                "--request-timeout", "30", "--log-dir", logs.name,
            ]
        )  # fmt: skip
        args.file_prompts = []
        measure = bench_campaign.measure_with(args, None)
        args.requests = 3
        arm = {
            "pass": 1,
            "engine": "stub",
            "set": "short",
            "cache": "off",
            "concurrency": 2,
        }
        result = measure(arm)
        self.assertTrue(result["restarted"])
        self.assertEqual(result["argv"][-2:], ["--prefix-cache", "off"])
        self.assertEqual(result["summary"]["outcomes"], {"ok": 3})
        self.assertEqual(result["summary"]["output_tokens"]["mean"], 4)
        self.assertGreaterEqual(result["memory"]["samples"], 1)


if __name__ == "__main__":
    unittest.main()
