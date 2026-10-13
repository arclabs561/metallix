#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = ["tokenizers>=0.20"]
# ///
"""Qualify race-v2 with existing oracles, path probes and idle paired trials.

The default stays icdf-v1. A passing receipt supports an explicit default review,
not an automatic flip. Build the three test executables and mx (Metal,
structured-output and timeline) first; this runner never builds or downloads.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import subprocess
import sys
import time
from pathlib import Path

import bench_load
import bench_pair
import bench_system
import mx_spans

ORACLES = {
    "engine": [
        "sampling::race::tests::philox_matches_the_random123_known_answers",
        "sampling::race::tests::noise_matches_an_independent_reference",
        "sampling::race::tests::draws_follow_the_legal_temperature_law",
    ],
    "qwen": [
        "forward::picks::tests::race_scores_stay_within_a_quarter_window_of_the_reference",
        "forward::picks::tests::certified_race_picks_are_the_reference_token",
        "forward::picks::tests::listed_race_rows_are_the_best_scores_and_bound_the_rest",
    ],
    "cli": [
        "qwen_forward::tests::icdf_v1_golden_tokens_pin_the_seeded_sampler",
        "qwen_forward::tests::constrained_race_from_listed_rows_matches_the_whole_row",
        "qwen_forward::tests::listed_rows_defer_when_an_unlisted_legal_row_can_win",
    ],
}
SCHEMA = {
    "type": "array",
    "minItems": 32,
    "maxItems": 32,
    "items": {"type": "integer", "minimum": 0, "maximum": 9},
}


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def oracle_passed(stdout: str, returncode: int) -> bool:
    return (
        returncode == 0
        and "skipping:" not in stdout
        and bool(re.search(r"test result: ok\. 1 passed; 0 failed; 0 ignored;", stdout))
    )


def verify_build(receipt: dict, binaries: dict[str, Path]) -> None:
    require(
        bool(re.fullmatch(r"[0-9a-f]{40}", receipt.get("source_revision", ""))),
        "source revision missing",
    )
    current = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    require(
        receipt["source_revision"] == current,
        "build receipt source revision differs from checkout",
    )
    tree = subprocess.check_output(
        ["git", "rev-parse", "HEAD^{tree}"], text=True
    ).strip()
    require(
        receipt.get("source_tree") == tree,
        "build receipt source tree differs from checkout",
    )
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=all"], text=True
    )
    require(
        not dirty.strip(),
        "qualification requires clean tracked and untracked public source",
    )
    for kind, path in binaries.items():
        require(
            receipt.get("artifacts", {}).get(kind) == digest(path),
            f"{kind} build identity mismatch",
        )


def run_oracles(binaries: dict[str, Path], output: Path) -> list[dict]:
    receipts = []
    for kind, names in ORACLES.items():
        for index, name in enumerate(names):
            command = [str(binaries[kind].resolve()), name, "--exact", "--nocapture"]
            result = subprocess.run(
                command, capture_output=True, text=True, timeout=300, check=False
            )
            log = output / f"oracle-{kind}-{index}.log"
            log.write_text(result.stdout + result.stderr)
            passed = oracle_passed(result.stdout + result.stderr, result.returncode)
            receipts.append(
                {
                    "test": name,
                    "binary_sha256": digest(binaries[kind]),
                    "log": str(log),
                    "passed": passed,
                }
            )
            require(passed, f"oracle did not execute and pass: {name}; see {log}")
    return receipts


def selection_counts(
    trace: Path, model_id: str, sampler: str, constrained: bool
) -> dict:
    report = mx_spans.request_report(trace, model_id)
    require(report["strict_artifacts_valid"], "invalid or incomplete sampling trace")
    spans = [
        s for request in report["requests"] for s in (request.get("sampling") or [])
    ]
    require(bool(spans), "DEBUG sampling spans absent")
    counts: dict[str, int] = {}
    for fields in spans:
        require(fields.get("mode") == "sampled", "sampled mode not observed")
        require(
            fields.get("sampler") == sampler,
            "observed sampler differs from requested arm",
        )
        if not constrained:
            require(
                str(fields.get("top_k")) in ("Some(20)", "20"),
                "top_k=20 path not observed",
            )
        if fields.get("outcome") == "selected":
            path = fields.get("path")
            counts[path] = counts.get(path, 0) + 1
    expected = (
        "constrained_full_row"
        if constrained and sampler == "icdf-v1"
        else "gpu_candidates"
    )
    require(counts.get(expected, 0) > 0, f"no selected {expected} path observed")
    return counts


def trial_metric(pairs: list[dict]) -> dict:
    require(len(pairs) >= 8, "at least eight declared pairs required")
    for index, pair in enumerate(pairs):
        require(
            pair.get("order") == ("AB" if index % 2 == 0 else "BA"),
            "paired order changed",
        )
        require("dropped" not in pair, "qualification refuses dropped pairs")
        for arm in "AB":
            run = pair[arm]
            require(not run.get("error") and not run.get("aborted"), "failed trial")
            summary = run["summary"]
            require(summary.get("requests", 0) >= 32, "too few measured requests")
            require(
                summary["tpot_ms"].get("n") == summary["requests"],
                "some requests lack TPOT",
            )
            require(
                summary.get("outcomes") == {"ok": summary.get("requests")},
                "failed or missing requests",
            )
            value = summary["tpot_ms"]["p50"]
            require(
                isinstance(value, (int, float)) and math.isfinite(value) and value > 0,
                "invalid TPOT",
            )
    return bench_pair.summarize_pairs(pairs, random.Random(0))["metrics"]["tpot_p50_ms"]


def verdict(metric: dict, gate: str, aa_tolerance: float) -> str:
    low, high = metric["ci95"]
    median = metric["median_ratio"]
    require(
        all(math.isfinite(v) and v > 0 for v in (low, high, median)),
        "invalid ratio interval",
    )
    if gate == "aa":
        return (
            "pass"
            if 1 - aa_tolerance <= low <= 1 <= high <= 1 + aa_tolerance
            else "inconclusive"
        )
    if gate == "constrained":
        return "pass" if high < 0.95 else "refuted" if low > 1 else "inconclusive"
    require(gate == "top_k", "unknown gate")
    return "pass" if median <= 1.02 and high <= 1.05 else "inconclusive"


def request_extra(sampler: str, constrained: bool) -> dict:
    extra = {"metallix_sampler": sampler, "seed": 0}
    if constrained:
        extra["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": "digits",
                "strict": True,
                "schema": SCHEMA,
            },
        }
    return extra


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-receipt", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--mx", required=True, type=Path)
    parser.add_argument("--model-path", required=True, type=Path)
    for kind in ORACLES:
        parser.add_argument(f"--{kind}-tests", required=True, type=Path)
    parser.add_argument("--pairs", type=int, default=8)
    parser.add_argument("--requests", type=int, default=32)
    parser.add_argument("--aa-tolerance", type=float, default=0.02)
    parser.add_argument("--abort-gpu-gib", type=float, required=True)
    args = parser.parse_args()
    require(
        args.pairs >= 8 and args.requests >= 32, "minimum eight pairs and32 requests"
    )
    require(0 < args.aa_tolerance <= 0.02, "A/A tolerance must be in (0,.02]")
    require(
        math.isfinite(args.abort_gpu_gib) and args.abort_gpu_gib > 0, "GPU cap required"
    )
    generation = args.model_path / "generation_config.json"
    require(
        json.loads(generation.read_text()).get("top_k") == 20,
        "control checkpoint must default top_k=20",
    )
    binaries = {k: getattr(args, f"{k}_tests") for k in ORACLES}
    build = json.loads(args.build_receipt.read_text())
    verify_build(build, binaries | {"mx": args.mx})
    require(
        json.loads(generation.read_text()).get("temperature", 0) > 0,
        "control checkpoint must default to sampled temperature",
    )
    args.output.mkdir(parents=True, exist_ok=False)
    receipt = {
        "status": "running",
        "default_sampler": "icdf-v1",
        "eligible_for_default_review": False,
        "build_receipt": build,
        "build_receipt_sha256": digest(args.build_receipt),
        "parameters": vars(args),
        "mx_sha256": digest(args.mx),
        "generation_config_sha256": digest(generation),
        "oracles": [],
        "probes": {},
        "trials": {},
    }

    def save():
        (args.output / "qualification.json").write_text(
            json.dumps(receipt, indent=2, default=str) + "\n"
        )

    save()
    original_log = os.environ.get("METALLIX_LOG")
    try:
        receipt["oracles"] = run_oracles(binaries, args.output)
        counter = bench_load.tokenizer_counter(args.model_path)
        shared = [
            "--server",
            "metallix",
            "--model-path",
            str(args.model_path),
            "--mx",
            str(args.mx),
            "--sets",
            "short",
            "--concurrency",
            "1",
            "--max-tokens",
            "128",
            "--abort-load",
            "2",
            "--abort-gpu-gib",
            str(args.abort_gpu_gib),
        ]

        def measure(label, sampler, constrained, trace=None):
            idle = bench_system.idle_readings()
            failures = bench_system.idle_failures(idle)
            require(not failures, f"idle gate failed: {failures}")
            trial = bench_pair.arm_args(shared, "")
            trial.request_extra = request_extra(sampler, constrained)
            trial.log_dir = args.output / label
            if trace:
                os.environ["METALLIX_LOG"] = "debug"
                trial.server_args += ["--trace-out", str(trace)]
            else:
                os.environ["METALLIX_LOG"] = "off"
            result = bench_load.measure_level(
                "metallix",
                "short",
                "concurrency",
                1,
                4 if trace else args.requests,
                trial,
                counter,
            )
            result["idle_gate"] = idle
            require(bench_pair.unusable(result) is None, f"unusable trial: {label}")
            return result

        # DEBUG path evidence is separate from measured runs.
        for constrained in (True, False):
            for sampler in ("icdf-v1", "race-v2"):
                label = f"probe-{constrained}-{sampler}"
                trace = (args.output / f"{label}.json").resolve()
                measure(label, sampler, constrained, trace)
                receipt["probes"][label] = selection_counts(
                    trace, "qwen3-0.6b", sampler, constrained
                )
                save()
        for gate in ("aa-constrained", "aa-top_k", "constrained", "top_k"):
            constrained = gate.endswith("constrained")
            runs = iter(range(args.pairs * 2))

            def measured(arm, gate=gate, runs=runs, constrained=constrained):
                sampler = (
                    "race-v2"
                    if arm == "B" and not gate.startswith("aa-")
                    else "icdf-v1"
                )
                return measure(f"{gate}-{next(runs):02d}-{arm}", sampler, constrained)

            pairs = bench_pair.run_pairs(args.pairs, measured, time.sleep, 5)
            receipt["trials"][gate] = {"pairs": pairs}
            save()
            metric = trial_metric(pairs)
            outcome = verdict(
                metric, "aa" if gate.startswith("aa-") else gate, args.aa_tolerance
            )
            receipt["trials"][gate].update(metric=metric, verdict=outcome)
            save()
            require(outcome == "pass", f"{gate}: {outcome}; later comparisons not run")
        verify_build(build, binaries | {"mx": args.mx})
        require(
            digest(generation) == receipt["generation_config_sha256"],
            "generation defaults changed during run",
        )
        receipt.update(status="passed", eligible_for_default_review=True)
        return 0
    except (ValueError, OSError, subprocess.TimeoutExpired) as error:
        receipt.update(status="incomplete", reason=str(error))
        return 1
    finally:
        if original_log is None:
            os.environ.pop("METALLIX_LOG", None)
        else:
            os.environ["METALLIX_LOG"] = original_log
        save()


if __name__ == "__main__":
    sys.exit(main())
