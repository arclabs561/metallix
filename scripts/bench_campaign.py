#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = ["tokenizers>=0.20"]
# ///
"""Run the same-Mac serving comparison: passes, idle gate, claim rule.

A campaign measures every arm (engine x prompt set x prefix-cache arm x
concurrency) once per pass, for several passes. Each pass visits the engines
in an order shuffled from a recorded seed, so slow drift on the machine does
not always land on the same engine. Each arm waits out a cooldown, must pass
an idle gate (load, compilers, AC power, thermal state, GPU idle) whose
readings are kept, and runs on a freshly started server. A load spike during
an arm aborts it.

Results are the median pass with the min-max range. The claim rule compares a
subject engine (metallix by default) with each competitor on one metric:
"faster" needs the subject's worst pass to beat the competitor's best pass;
"significantly faster" also needs a median ratio of at least 1.5.

Usage:
  uv run scripts/bench_campaign.py --server metallix,vllm-metal,mlx-lm \\
      --model-path DIR --sets shared-prefix --concurrency 1,4,8 \\
      --cache-arms on,off --passes 3 --json campaign.json
  uv run scripts/bench_campaign.py summarize campaign.json load-run.json ...

Load options (--max-tokens, --mx, --vllm-memory-fraction, ...) are those of
bench_load.py. `summarize` also accepts bench_load.py reports, each counted as
one pass, so engines started by hand (llama.cpp, Ollama) join the comparison.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import statistics
import sys
import time
from collections.abc import Callable
from pathlib import Path

import bench_load
import bench_system

SUBJECT = "metallix"
# (summary key, higher is better)
METRICS = {
    "output_tok_s": (("output_token_throughput",), True),
    "ttft_p50_ms": (("ttft_ms", "p50"), False),
    "tpot_p50_ms": (("tpot_ms", "p50"), False),
}
SIGNIFICANT_RATIO = 1.5
PARITY_BAND = 0.10
# The baseline gate: each engine's passes within this share of their median.
SPREAD_LIMIT = 0.05


def cell_size(concurrency: int) -> int:
    """Requests per measured cell: 64 up to c=8, 100 above (SiliconBench's n)."""
    return 64 if concurrency <= 8 else 100


# ---------------------------------------------------------------------------
# Plan and run


def plan(
    engines: list[str],
    sets: list[str],
    cache_arms: list[str],
    levels: list[int],
    passes: int,
    seed: int,
) -> tuple[list[dict], list[list[str]], list[str]]:
    """Arms in run order, the engine order of each pass, and skipped pairs.

    An engine without a prefix-cache switch runs only the default arm.
    """
    rng = random.Random(seed)
    arms, orders, skipped = [], [], []
    for engine in engines:
        for cache in cache_arms:
            if cache != "default" and engine not in bench_load.PREFIX_CACHE_FLAGS:
                skipped.append(f"{engine} cache={cache}: no prefix-cache switch")
    for number in range(1, passes + 1):
        order = list(engines)
        rng.shuffle(order)
        orders.append(order)
        for engine in order:
            for set_name in sets:
                for cache in cache_arms:
                    if (
                        cache != "default"
                        and engine not in bench_load.PREFIX_CACHE_FLAGS
                    ):
                        continue
                    for level in levels:
                        arms.append(
                            {
                                "pass": number,
                                "engine": engine,
                                "set": set_name,
                                "cache": cache,
                                "concurrency": level,
                            }
                        )
    return arms, orders, skipped


def wait_for_idle(
    readings: Callable[[], dict],
    failures: Callable[[dict], list[str]],
    sleep: Callable[[float], None],
    timeout: float,
    poll: float = 30,
) -> dict:
    """Poll the idle gate until it passes or `timeout` seconds have gone by."""
    waited, attempts = 0.0, 0
    while True:
        attempts += 1
        reading = readings()
        failed = failures(reading)
        if not failed or waited >= timeout:
            return {
                "passed": not failed,
                "failures": failed,
                "readings": reading,
                "attempts": attempts,
                "waited_s": waited,
            }
        sleep(poll)
        waited += poll


def run_campaign(
    arms: list[dict],
    measure: Callable[[dict], dict],
    gate: Callable[[], dict] | None,
    sleep: Callable[[float], None],
    cooldown: float,
) -> list[dict]:
    """Measure each arm in order: cooldown, idle gate, then a fresh server.

    `gate` returns a wait_for_idle result; None skips the gate (dry runs),
    which marks every arm as not claim-grade.
    """
    for index, arm in enumerate(arms):
        if index:
            sleep(cooldown)
        print(
            f"pass {arm['pass']} {arm['engine']} {arm['set']} "
            f"cache={arm['cache']} c={arm['concurrency']}",
            flush=True,
        )
        arm["idle_gate"] = gate() if gate else {"passed": False, "skipped": True}
        if gate and not arm["idle_gate"]["passed"]:
            arm["skipped"] = "not idle: " + "; ".join(arm["idle_gate"]["failures"])
            print(f"  skipped: {arm['skipped']}", flush=True)
            continue
        arm["result"] = measure(arm)
    return arms


def measure_with(args, count_tokens) -> Callable[[dict], dict]:
    def measure(arm: dict) -> dict:
        level_args = copy.copy(args)
        level_args.prefix_cache = arm["cache"]
        level = arm["concurrency"]
        count = args.requests or cell_size(level)
        return bench_load.measure_level(
            arm["engine"],
            arm["set"],
            "concurrency",
            level,
            count,
            level_args,
            count_tokens,
        )

    return measure


# ---------------------------------------------------------------------------
# Statistics and the claim rule


def spread(values: list[float]) -> dict:
    """Median, min-max and the largest relative deviation from the median."""
    median = statistics.median(values)
    deviation = max(abs(v - median) for v in values) / median if median else None
    return {
        "median": median,
        "min": min(values),
        "max": max(values),
        "n": len(values),
        "max_deviation": deviation,
    }


def claim(
    subject: list[float],
    competitor: list[float],
    higher_is_better: bool,
    min_passes: int = 3,
) -> dict:
    """The claim rule for one metric in one cell.

    `ratio` is > 1 when the subject is better (throughput divided, latency
    inverted). "faster" needs the subject's worst pass to beat the competitor's
    best; "significantly faster" also needs ratio >= 1.5. "slower" is the
    mirror image; anything else is "overlapping". `parity` is a separate test:
    medians within 10%.
    """
    if len(subject) < min_passes or len(competitor) < min_passes:
        return {
            "verdict": "insufficient passes",
            "ratio": None,
            "parity": None,
            "passes": [len(subject), len(competitor)],
        }
    s_med, c_med = statistics.median(subject), statistics.median(competitor)
    if higher_is_better:
        ratio = s_med / c_med if c_med else None
        faster = min(subject) > max(competitor)
        slower = max(subject) < min(competitor)
    else:
        ratio = c_med / s_med if s_med else None
        faster = max(subject) < min(competitor)
        slower = min(subject) > max(competitor)
    if faster and ratio is not None and ratio >= SIGNIFICANT_RATIO:
        verdict = "significantly faster"
    elif faster:
        verdict = "faster"
    elif slower:
        verdict = "slower"
    else:
        verdict = "overlapping"
    return {
        "verdict": verdict,
        "ratio": ratio,
        "parity": ratio is not None and abs(ratio - 1) <= PARITY_BAND,
        "passes": [len(subject), len(competitor)],
    }


def metric(summary: dict, path: tuple[str, ...]) -> float | None:
    value = summary
    for key in path:
        value = value.get(key) if isinstance(value, dict) else None
    return value


def rows_from_campaign(report: dict) -> list[dict]:
    """One row per measured arm: (engine, set, cache, c, pass) plus its summary,
    with whether the arm is claim-grade (idle gate passed, not aborted)."""
    rows = []
    for arm in report["arms"]:
        result = arm.get("result") or {}
        if "summary" not in result:
            continue
        rows.append(
            {
                "engine": arm["engine"],
                "set": arm["set"],
                "cache": arm["cache"],
                "concurrency": arm["concurrency"],
                "pass": f"pass {arm['pass']}",
                "summary": result["summary"],
                "records": result["records"],
                "claim_grade": arm["idle_gate"].get("passed", False),
            }
        )
    return rows


def rows_from_load_report(report: dict, name: str) -> list[dict]:
    """A bench_load.py report as one pass, claim-grade when the idle gate it
    recorded at the start passed (it cannot restart a --url server per level)."""
    grade = not (report.get("idle_gate") or {"failures": ["none recorded"]})["failures"]
    rows = []
    for server in report["servers"]:
        for result in server.get("sets", []):
            for run in result["concurrency"]:
                if "summary" not in run:
                    continue
                rows.append(
                    {
                        "engine": server["name"],
                        "set": result["set"],
                        "cache": server.get("prefix_cache", "default"),
                        "concurrency": run["concurrency"],
                        "pass": name,
                        "summary": run["summary"],
                        "records": run["records"],
                        "claim_grade": grade,
                    }
                )
    return rows


def summarize_rows(rows: list[dict], subject: str = SUBJECT) -> dict:
    """Per-cell statistics, claims against each competitor, and warnings."""
    cells: dict[tuple, dict[str, list[dict]]] = {}
    for row in rows:
        key = (row["set"], row["cache"], row["concurrency"])
        cells.setdefault(key, {}).setdefault(row["engine"], []).append(row)
    table, warnings, ungated = [], [], 0
    for (set_name, cache, level), by_engine in sorted(cells.items()):
        stats = {}
        for engine, engine_rows in by_engine.items():
            stats[engine] = {}
            for name, (path, _) in METRICS.items():
                values = [
                    v
                    for r in engine_rows
                    if (v := metric(r["summary"], path)) is not None
                ]
                stats[engine][name] = spread(values) if values else None
            deviation = (stats[engine]["output_tok_s"] or {}).get("max_deviation")
            if deviation is not None and deviation > SPREAD_LIMIT:
                warnings.append(
                    f"{engine} {set_name}/{cache} c={level}: throughput passes deviate "
                    f"{deviation:.1%} from their median (gate {SPREAD_LIMIT:.0%})"
                )
            if not all(r["claim_grade"] for r in engine_rows):
                ungated += 1
        claims = {}
        if subject in by_engine:
            for engine in by_engine:
                if engine == subject:
                    continue
                grade = all(
                    r["claim_grade"] for r in by_engine[subject] + by_engine[engine]
                )
                claims[engine] = {"claim_grade": grade} | {
                    name: claim(
                        [
                            v
                            for r in by_engine[subject]
                            if (v := metric(r["summary"], path)) is not None
                        ],
                        [
                            v
                            for r in by_engine[engine]
                            if (v := metric(r["summary"], path)) is not None
                        ],
                        higher,
                    )
                    for name, (path, higher) in METRICS.items()
                }
        passes: dict[str, dict[str, list[dict]]] = {}
        for engine, engine_rows in by_engine.items():
            for r in engine_rows:
                passes.setdefault(r["pass"], {})[engine] = r["records"]
        for pass_name, records in sorted(passes.items()):
            cell = {(f"{set_name}/{cache} {pass_name}", "c", level): records}
            warnings += bench_load.failure_warnings(cell)
            warnings += bench_load.token_count_warnings(cell)
        table.append(
            {
                "set": set_name,
                "cache": cache,
                "concurrency": level,
                "stats": stats,
                "claims": claims,
            }
        )
    if ungated:
        warnings.append(
            f"{ungated} engine cells include passes without a passed idle gate; "
            "they are not claim-grade"
        )
    return {"subject": subject, "cells": table, "warnings": warnings}


def render(summary: dict) -> str:
    fmt = bench_load.fmt
    lines = [
        (
            "set/cache            c  engine        out tok/s med [min-max]   "
            "TTFT p50 ms med [min-max]   TPOT p50 ms med [min-max]   n"
        )
    ]
    for cell in summary["cells"]:
        for engine, stats in cell["stats"].items():
            parts = []
            for name in METRICS:
                s = stats[name]
                digits = 1 if name == "tpot_p50_ms" else 0
                parts.append(
                    "-"
                    if s is None
                    else f"{fmt(s['median'], digits)} [{fmt(s['min'], digits)}-"
                    f"{fmt(s['max'], digits)}]"
                )
            n = (stats["output_tok_s"] or {}).get("n", 0)
            lines.append(
                f"{cell['set'] + '/' + cell['cache']:<20} {cell['concurrency']:>2}  "
                f"{engine:<12}  {parts[0]:<26} {parts[1]:<27} {parts[2]:<27} {n}"
            )
        for engine, claims in cell["claims"].items():
            verdicts = ", ".join(
                f"{name} {c['verdict']}"
                + (f" ({c['ratio']:.2f}x)" if c["ratio"] is not None else "")
                + (" parity" if c["parity"] else "")
                for name, c in claims.items()
                if name in METRICS
            ) + ("" if claims["claim_grade"] else "; NOT CLAIM-GRADE")
            lines.append(f"{'':<23} {summary['subject']} vs {engine}: {verdicts}")
    lines += [f"warning: {warning}" for warning in summary["warnings"]]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Command line


def summarize_files(paths: list[Path], subject: str) -> int:
    rows = []
    for path in paths:
        report = json.loads(path.read_text())
        if "arms" in report:
            rows += rows_from_campaign(report)
        else:
            rows += rows_from_load_report(report, path.name)
    print(render(summarize_rows(rows, subject)))
    return 0


def main() -> int:
    if sys.argv[1:2] == ["summarize"]:
        parser = argparse.ArgumentParser(description="Summarize saved reports.")
        parser.add_argument("reports", type=Path, nargs="+")
        parser.add_argument("--subject", default=SUBJECT)
        args = parser.parse_args(sys.argv[2:])
        return summarize_files(args.reports, args.subject)

    parser = bench_load.build_parser(__doc__.splitlines()[0])
    parser.add_argument("--passes", type=int, default=3)
    parser.add_argument(
        "--order-seed",
        type=int,
        help="seed for the engine order of each pass (default: drawn and recorded)",
    )
    parser.add_argument(
        "--cache-arms",
        default="default",
        help=f"comma-separated, from {', '.join(bench_load.CACHE_ARMS)}",
    )
    parser.add_argument("--cooldown", type=float, default=60, help="seconds")
    parser.add_argument(
        "--idle-timeout",
        type=float,
        default=1800,
        help="seconds to wait for the idle gate before skipping an arm",
    )
    parser.add_argument(
        "--skip-idle-gate",
        action="store_true",
        help="dry runs only: results are marked not claim-grade",
    )
    parser.add_argument(
        "--requests", type=int, help="requests per cell (default: 64, or 100 at c>8)"
    )
    parser.add_argument("--subject", default=SUBJECT)
    parser.set_defaults(abort_load=4.0)
    args = parser.parse_args()
    if args.url:
        parser.error(
            "a campaign restarts its servers; measure --url engines with "
            "bench_load.py and combine the reports with `summarize`"
        )
    if args.rates:
        parser.error("a campaign measures concurrency levels only")
    cache_arms = [arm for arm in args.cache_arms.split(",") if arm]
    for arm in cache_arms:
        if arm not in bench_load.CACHE_ARMS:
            parser.error(f"unknown cache arm {arm!r}")
    sets = bench_load.prompt_sets(parser, args)
    seed = args.order_seed
    if seed is None:
        seed = random.SystemRandom().randrange(2**32)
    engines = args.server.split(",")
    arms, orders, skipped = plan(
        engines, sets, cache_arms, args.concurrency, args.passes, seed
    )
    print(f"order seed {seed}; engine order per pass: {orders}", flush=True)
    for line in skipped:
        print(f"skipped: {line}", flush=True)
    gate = None
    if not args.skip_idle_gate:
        gate = lambda: wait_for_idle(
            bench_system.idle_readings,
            bench_system.idle_failures,
            time.sleep,
            args.idle_timeout,
        )
    count_tokens = bench_load.tokenizer_counter(args.model_path)
    report = {
        "argv": sys.argv,
        "revision": bench_load.bench_serve.repo_revision(),
        "system": bench_system.system_info(),
        "model_path": str(args.model_path),
        "model_id": args.model_id,
        "max_tokens": args.max_tokens,
        "prompts_file": bench_load.file_identity(args.prompts_file),
        "order_seed": seed,
        "orders": orders,
        "skipped": skipped,
        "idle_gate": not args.skip_idle_gate,
        "cooldown_s": args.cooldown,
        "abort_load": args.abort_load,
        "arms": arms,
    }
    try:
        run_campaign(
            arms, measure_with(args, count_tokens), gate, time.sleep, args.cooldown
        )
    finally:
        # Keep whatever was measured if the run is interrupted.
        report["summary"] = summarize_rows(rows_from_campaign(report), args.subject)
        if args.json:
            args.json.parent.mkdir(parents=True, exist_ok=True)
            args.json.write_text(json.dumps(report, indent=1) + "\n")
    print()
    print(render(report["summary"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
