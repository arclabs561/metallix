# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Sampler qualification refusals and fixed promotion bars, without a device."""

from __future__ import annotations

import json
import pathlib
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import qualify_sampler as qualification


def pairs(ratio=1.0):
    def run(value):
        return {
            "summary": {
                "requests": 32,
                "outcomes": {"ok": 32},
                "output_token_throughput": 100 / value,
                "tpot_ms": {"p50": value, "n": 32},
            }
        }

    return [
        {"order": "AB" if i % 2 == 0 else "BA", "A": run(1), "B": run(ratio)}
        for i in range(8)
    ]


class Qualification(unittest.TestCase):
    def test_independent_oracle_zero_skipped_and_failed_are_not_evidence(self):
        good = "test result: ok. 1 passed; 0 failed; 0 ignored; 300 filtered out"
        self.assertTrue(qualification.oracle_passed(good, 0))
        for text, code in [
            (good, 1),
            (good + " skipping: no model", 0),
            (good.replace("1 passed", "0 passed"), 0),
            (good.replace("0 ignored", "1 ignored"), 0),
        ]:
            self.assertFalse(qualification.oracle_passed(text, code))

    def test_oracle_skip_on_stderr_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root = pathlib.Path(d)
            binary = root / "binary"
            binary.write_bytes(b"test")
            result = qualification.subprocess.CompletedProcess(
                [],
                0,
                "test result: ok. 1 passed; 0 failed; 0 ignored;",
                "skipping: fixture unavailable",
            )
            with (
                mock.patch.object(qualification.subprocess, "run", return_value=result),
                self.assertRaises(ValueError),
            ):
                qualification.run_oracles(
                    {k: binary for k in qualification.ORACLES}, root
                )

    def test_declared_a_a_floor_rejects_bias_and_imprecision(self):
        for ci in ([1.01, 1.02], [0.97, 1.01], [0.99, 1.03]):
            self.assertEqual(
                qualification.verdict({"ci95": ci, "median_ratio": 1}, "aa", 0.02),
                "inconclusive",
            )
        self.assertEqual(
            qualification.verdict(
                {"ci95": [0.99, 1.01], "median_ratio": 1}, "aa", 0.02
            ),
            "pass",
        )

    def test_fixed_bars_keep_boundary_strictness(self):
        for median, ci, gate, expected in [
            (0.94, [0.93, 0.95], "constrained", "inconclusive"),
            (0.93, [0.92, 0.94], "constrained", "pass"),
            (1.02, [1.01, 1.04], "constrained", "refuted"),
            (1.02, [0.99, 1.05], "top_k", "pass"),
            (1.021, [0.99, 1.04], "top_k", "inconclusive"),
        ]:
            self.assertEqual(
                qualification.verdict({"ci95": ci, "median_ratio": median}, gate, 0.02),
                expected,
            )

    def test_ratios_recomputed_from_pairs_not_a_claimed_summary(self):
        metric = qualification.trial_metric(pairs(0.9))
        self.assertAlmostEqual(metric["median_ratio"], 0.9)
        self.assertEqual(qualification.verdict(metric, "constrained", 0.02), "pass")
        changed = pairs(0.9)
        changed[0]["dropped"] = "loaded"
        with self.assertRaises(ValueError):
            qualification.trial_metric(changed)

    def test_bad_or_missing_trial_does_not_enter_the_ratio(self):
        for change in (
            lambda p: p[0].update(order="BA"),
            lambda p: p[0]["A"].update(aborted="GPU cap"),
            lambda p: p[0]["A"]["summary"].update(outcomes={"ok": 31, "error": 1}),
            lambda p: p[0]["A"]["summary"]["tpot_ms"].update(p50=float("nan")),
            lambda p: p[0]["A"]["summary"]["tpot_ms"].update(n=31),
        ):
            value = pairs()
            change(value)
            with self.assertRaises(ValueError):
                qualification.trial_metric(value)
        with self.assertRaises(ValueError):
            qualification.trial_metric(pairs()[:7])

    def test_actual_span_metadata_requires_selected_path_and_sampler(self):
        report = {
            "strict_artifacts_valid": True,
            "requests": [
                {
                    "sampling": [
                        {
                            "sampler": "race-v2",
                            "mode": "sampled",
                            "outcome": "selected",
                            "path": "gpu_candidates",
                            "top_k": "Some(20)",
                        }
                    ]
                }
            ],
        }
        with mock.patch.object(
            qualification.mx_spans, "request_report", return_value=report
        ):
            self.assertEqual(
                qualification.selection_counts(
                    pathlib.Path("t"), "m", "race-v2", False
                ),
                {"gpu_candidates": 1},
            )
            report["requests"][0]["sampling"][0]["outcome"] = "peek"
            with self.assertRaises(ValueError):
                qualification.selection_counts(pathlib.Path("t"), "m", "race-v2", False)
            report["requests"][0]["sampling"] = None
            with self.assertRaises(ValueError):
                qualification.selection_counts(pathlib.Path("t"), "m", "race-v2", False)

    def test_biased_a_a_stops_before_any_a_b_measurement(self):
        with tempfile.TemporaryDirectory() as d:
            root = pathlib.Path(d)
            model = root / "model"
            model.mkdir()
            (model / "generation_config.json").write_text(
                '{"top_k":20,"temperature":0.8}'
            )
            binary = root / "binary"
            binary.write_bytes(b"build")
            build = root / "build.json"
            build.write_text(
                json.dumps(
                    {
                        "source_revision": "a" * 40,
                        "source_tree": "a" * 40,
                        "artifacts": {
                            k: qualification.digest(binary)
                            for k in ("mx", "engine", "qwen", "cli")
                        },
                    }
                )
            )
            output = root / "out"
            seen = []

            def measure(_name, _set, _kind, _width, _count, args, _counter):
                label = args.log_dir.name
                seen.append(label)
                row = pairs()[0]["A"]
                row["summary"]["tpot_ms"]["p50"] = 1.03 if label.endswith("-B") else 1
                return row

            argv = [
                "qualify_sampler.py",
                "--output",
                str(output),
                "--mx",
                str(binary),
                "--model-path",
                str(model),
                "--build-receipt",
                str(build),
                "--abort-gpu-gib",
                "8",
            ]
            for kind in qualification.ORACLES:
                argv += [f"--{kind}-tests", str(binary)]
            with (
                mock.patch.object(sys, "argv", argv),
                mock.patch.object(
                    qualification.subprocess,
                    "check_output",
                    side_effect=lambda command, **_: (
                        "" if "status" in command else "a" * 40
                    ),
                ),
                mock.patch.object(
                    qualification, "run_oracles", return_value=[{"passed": True}]
                ),
                mock.patch.object(
                    qualification,
                    "selection_counts",
                    return_value={"gpu_candidates": 1},
                ),
                mock.patch.object(
                    qualification.bench_load,
                    "tokenizer_counter",
                    return_value=lambda _text: 1,
                ),
                mock.patch.object(
                    qualification.bench_load, "measure_level", side_effect=measure
                ),
                mock.patch.object(
                    qualification.bench_system, "idle_readings", return_value={}
                ),
                mock.patch.object(
                    qualification.bench_system, "idle_failures", return_value=[]
                ),
                mock.patch.object(qualification.time, "sleep"),
            ):
                self.assertEqual(qualification.main(), 1)
            report = json.loads((output / "qualification.json").read_text())
            self.assertFalse(report["eligible_for_default_review"])
            self.assertEqual(set(report["trials"]), {"aa-constrained"})
            self.assertEqual(
                report["trials"]["aa-constrained"]["verdict"], "inconclusive"
            )
            self.assertEqual(len(seen), 20)

    def test_dirty_tracked_or_untracked_public_source_refuses_build_receipt(self):
        with tempfile.TemporaryDirectory() as d:
            path = pathlib.Path(d) / "binary"
            path.write_bytes(b"built")
            receipt = {
                "source_revision": "a" * 40,
                "source_tree": "a" * 40,
                "artifacts": {"mx": qualification.digest(path)},
            }
            for status in (
                " M crates/core/engine/src/sampling.rs",
                "?? crates/untracked.rs",
            ):
                with (
                    mock.patch.object(
                        qualification.subprocess,
                        "check_output",
                        side_effect=["a" * 40, "a" * 40, status],
                    ),
                    self.assertRaises(ValueError),
                ):
                    qualification.verify_build(receipt, {"mx": path})
            receipt["source_tree"] = "b" * 40
            with (
                mock.patch.object(
                    qualification.subprocess,
                    "check_output",
                    side_effect=["a" * 40, "a" * 40],
                ),
                self.assertRaises(ValueError),
            ):
                qualification.verify_build(receipt, {"mx": path})

    def test_build_identity_binds_all_executables_and_revision(self):
        with tempfile.TemporaryDirectory() as d:
            path = pathlib.Path(d) / "test"
            path.write_bytes(b"real build")
            receipt = {
                "source_revision": "a" * 40,
                "source_tree": "a" * 40,
                "artifacts": {"mx": qualification.digest(path)},
            }
            with mock.patch.object(
                qualification.subprocess,
                "check_output",
                side_effect=lambda command, **_: (
                    "" if "status" in command else "a" * 40
                ),
            ):
                qualification.verify_build(receipt, {"mx": path})
                path.write_bytes(b"different build")
                with self.assertRaises(ValueError):
                    qualification.verify_build(receipt, {"mx": path})


if __name__ == "__main__":
    unittest.main()
