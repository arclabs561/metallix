#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = ["tokenizers>=0.20"]
# ///
"""Paired A/B comparison of two server configurations on a loaded machine.

Runs arm A and arm B at one level in the order A,B, B,A, A,B, ... for N
pairs, each run on a fresh server, and reports per metric the median of the
per-pair ratios B/A with a bootstrap 95% confidence interval. Machine load
that drifts slowly affects both runs of a pair about equally, and alternating
which arm goes first cancels a steady trend, so a ratio interval that
excludes 1 is evidence of a difference even when the machine is busy.
Each metric also reports how many pairs favor B, the geometric mean ratio
with a t interval on log ratios, and the minimum detectable effect at that
spread, 2.8 sd(log ratio) / sqrt(pairs); an A/A run (one build in both arms)
measures that floor. --max-load-delta drops a pair whose two runs started at
1-min loads further apart than a threshold declared before the run; dropped
pairs are counted, and more than a quarter dropped refuses the result.
Absolute numbers from a loaded machine are still not results; publish those
only from the idle protocol (bench_campaign.py).

Arms share every bench_load.py option given on the command line; --a-args
and --b-args add or override options per arm (later options win).

Usage:
  uv run scripts/bench_pair.py --server metallix --model-path DIR \\
      --sets short --concurrency 1 --pairs 8 \\
      --a-args "--mx build-a/mx" --b-args "--mx build-b/mx"
  uv run scripts/bench_pair.py --server metallix --model-path DIR \\
      --sets short --concurrency 1 --b-args "--server-arg=--draft-model --server-arg=D"
"""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import math
import random
import shlex
import statistics
import sys
import time
from collections.abc import Callable
from pathlib import Path

import bench_load
import bench_system

# (summary key path, higher is better). Ratios are always B/A.
METRICS = {
    "output_tok_s": (("output_token_throughput",), True),
    "tpot_p50_ms": (("tpot_ms", "p50"), False),
    "ttft_p50_ms": (("ttft_ms", "p50"), False),
    "ttft_p90_ms": (("ttft_ms", "p90"), False),
    "itl_p99_ms": (("itl_ms", "p99"), False),
}


def pair_order(pairs: int) -> list[tuple[str, str]]:
    """A,B then B,A, alternating, so a steady trend favors neither arm."""
    return [("A", "B") if i % 2 == 0 else ("B", "A") for i in range(pairs)]


def bootstrap_median_ci(
    values: list[float], rng: random.Random, resamples: int = 2000, level: float = 0.95
) -> tuple[float, float]:
    """Percentile bootstrap interval for the median of `values`."""
    medians = sorted(
        statistics.median(rng.choices(values, k=len(values))) for _ in range(resamples)
    )
    tail = (1 - level) / 2 * 100
    return (
        bench_load.percentile(medians, tail),
        bench_load.percentile(medians, 100 - tail),
    )


# Two-sided 97.5% quantiles of Student's t for 1-30 degrees of freedom.
T_975 = (
    12.706,
    4.303,
    3.182,
    2.776,
    2.571,
    2.447,
    2.365,
    2.306,
    2.262,
    2.228,
    2.201,
    2.179,
    2.160,
    2.145,
    2.131,
    2.120,
    2.110,
    2.101,
    2.093,
    2.086,
    2.080,
    2.074,
    2.069,
    2.064,
    2.060,
    2.056,
    2.052,
    2.048,
    2.045,
    2.042,
)
# z(0.975) + z(0.80): the detectable effect at 80% power and 5% two-sided.
MDE_FACTOR = 2.8


def t_975(df: int) -> float:
    """Student's t 97.5% quantile; past the table, the Cornish-Fisher
    expansion around the normal quantile (error under 0.001 for df > 30)."""
    if df <= len(T_975):
        return T_975[df - 1]
    z = 1.959964
    return z + (z**3 + z) / (4 * df) + (5 * z**5 + 16 * z**3 + 3 * z) / (96 * df**2)


def log_ratio_interval(ratios: list[float]) -> dict:
    """Mean of log(B/A) with a t interval, as ratios, and the minimum
    detectable effect MDE = 2.8 sd / sqrt(n) at this spread and pair count.

    A log ratio is symmetric (B twice A and A twice B are equally far from
    0), and the mean over pairs suits a t interval where the median suits
    the bootstrap. None when fewer than two pairs.
    """
    n = len(ratios)
    if n < 2:
        return {"geomean_ratio": None, "ci95": None, "sd_log": None, "mde": None}
    logs = [math.log(ratio) for ratio in ratios]
    mean, sd = statistics.fmean(logs), statistics.stdev(logs)
    half = t_975(n - 1) * sd / math.sqrt(n)
    return {
        "geomean_ratio": math.exp(mean),
        "ci95": [math.exp(mean - half), math.exp(mean + half)],
        "sd_log": sd,
        "mde": math.exp(MDE_FACTOR * sd / math.sqrt(n)) - 1,
    }


def metric(summary: dict, path: tuple[str, ...]) -> float | None:
    for key in path:
        summary = summary.get(key) if isinstance(summary, dict) else None
    return summary


def run_pairs(
    pairs: int,
    measure: Callable[[str], dict],
    sleep: Callable[[float], None],
    cooldown: float,
) -> list[dict]:
    """Measure every pair; a pair where either run has no summary (aborted,
    failed to start) is kept with its reason and left out of the ratios."""
    out = []
    for index, order in enumerate(pair_order(pairs)):
        pair: dict = {"pair": index, "order": "".join(order)}
        for position, arm in enumerate(order):
            if index or position:
                sleep(cooldown)
            pair[arm] = measure(arm)
        reasons = {arm: unusable(pair[arm]) for arm in "AB"}
        if any(reasons.values()):
            pair["dropped"] = "; ".join(
                f"{arm}: {reason}" for arm, reason in reasons.items() if reason
            )
        out.append(pair)
    return out


def unusable(run: dict) -> str | None:
    """Why a run cannot enter a ratio, or None. A run whose requests all
    failed has a summary but no completed output to compare."""
    if "summary" not in run:
        return str(run.get("aborted") or run.get("error"))
    if not run["summary"].get("output_token_throughput"):
        return f"no completed requests ({run['summary'].get('outcomes')})"
    return None


def pair_load(pair: dict) -> list[float | None]:
    """The 1-min load when each run of the pair started, A first."""
    return [(pair[arm].get("baseline") or {}).get("load_1m") for arm in "AB"]


# A comparison whose dropped pairs exceed this share is not reported: the
# pairs left are the ones the machine happened to leave alone.
MAX_DROPPED_SHARE = 0.25


def drop_load_swings(pairs: list[dict], max_delta: float | None) -> int:
    """Drops each pair whose two runs started at 1-min loads more than
    `max_delta` apart, a threshold declared before the run. A load swing
    inside a pair is what spreads an A/A, so such a pair compares machine
    states, not arms. Returns how many it dropped."""
    if max_delta is None:
        return 0
    dropped = 0
    for pair in pairs:
        if "dropped" in pair:
            continue
        loads = pair_load(pair)
        if None in loads:
            continue
        delta = abs(loads[0] - loads[1])
        if delta > max_delta:
            pair["dropped"] = f"load delta {delta:.1f} > {max_delta:g}"
            dropped += 1
    return dropped


def summarize_pairs(pairs: list[dict], rng: random.Random) -> dict:
    kept = [pair for pair in pairs if "dropped" not in pair]
    out: dict = {"pairs": len(pairs), "kept": len(kept), "metrics": {}}
    if pairs and (len(pairs) - len(kept)) / len(pairs) > MAX_DROPPED_SHARE:
        out["refused"] = (
            f"{len(pairs) - len(kept)} of {len(pairs)} pairs dropped, more than "
            f"{MAX_DROPPED_SHARE:.0%}"
        )
        return out
    for name, (path, higher) in METRICS.items():
        ratios = []
        for pair in kept:
            a = metric(pair["A"]["summary"], path)
            b = metric(pair["B"]["summary"], path)
            if a and b is not None:
                ratios.append(b / a)
        if not ratios:
            continue
        low, high = bootstrap_median_ci(ratios, rng)
        # A tail percentile from too few samples is the run's max, so its ratio
        # compares maxima; say so rather than report it as a percentile.
        unresolved = []
        if path[-1] in ("p90", "p99"):
            q = int(path[-1][1:])
            unresolved = [
                n
                for pair in kept
                for arm in "AB"
                if (n := pair[arm]["summary"][path[0]].get("n")) is not None
                and not bench_load.resolves(q, n)
            ]
        out["metrics"][name] = {
            "unresolved_n": min(unresolved) if unresolved else None,
            "log_ratio": log_ratio_interval([r for r in ratios if r > 0]),
            # Pairs whose ratio favors B, out of the pairs compared.
            "favor_b": [sum((r > 1) == higher for r in ratios if r != 1), len(ratios)],
            "median_ratio": statistics.median(ratios),
            "ci95": [low, high],
            "ratios": ratios,
            "higher_is_better": higher,
            # Which arm the interval favors, if it excludes 1.
            "verdict": (
                "no difference shown"
                if low <= 1 <= high
                else ("B better" if (low > 1) == higher else "A better")
            ),
        }
    return out


def render(summary: dict, pairs: list[dict]) -> str:
    lines = []
    for pair in pairs:
        load = ", ".join(bench_load.fmt(value, 1) for value in pair_load(pair))
        if "dropped" in pair:
            lines.append(
                f"pair {pair['pair']} ({pair['order']}): dropped: {pair['dropped']}"
            )
            continue
        ratio = metric(pair["B"]["summary"], METRICS["output_tok_s"][0]) / metric(
            pair["A"]["summary"], METRICS["output_tok_s"][0]
        )
        lines.append(
            f"pair {pair['pair']} ({pair['order']}): out tok/s B/A {ratio:.3f}; "
            f"load at start A, B: {load}"
        )
    lines.append(f"{summary['kept']} of {summary['pairs']} pairs kept")
    if summary.get("max_load_delta") is not None:
        lines.append(
            f"{summary['load_dropped']} pairs dropped for a load delta above "
            f"{summary['max_load_delta']:g}"
        )
    if "refused" in summary:
        lines.append(f"RESULT REFUSED: {summary['refused']}")
    for name, m in summary["metrics"].items():
        direction = "higher" if m["higher_is_better"] else "lower"
        log = m["log_ratio"]
        lines.append(
            f"{name} B/A median {m['median_ratio']:.3f} "
            f"[95% CI {m['ci95'][0]:.3f}-{m['ci95'][1]:.3f}] ({direction} is better): "
            f"{m['verdict']}; {m['favor_b'][0]} of {m['favor_b'][1]} pairs favor B"
            + (
                f"; geomean {log['geomean_ratio']:.3f} "
                f"[t 95% CI {log['ci95'][0]:.3f}-{log['ci95'][1]:.3f}], "
                f"MDE {log['mde']:.1%}"
                if log["geomean_ratio"] is not None
                else ""
            )
            + (
                f" (runs with n={m['unresolved_n']} compare the max, not {METRICS[name][0][-1]})"
                if m.get("unresolved_n") is not None
                else ""
            )
        )
    return "\n".join(lines)


def arm_args(shared: list[str], extra: str) -> argparse.Namespace:
    """bench_load options for one arm: the shared ones, then the arm's own."""
    parser = bench_load.build_parser()
    args = parser.parse_args(shared + shlex.split(extra))
    bench_load.prompt_sets(parser, args)
    if args.rates or len(args.concurrency) != 1:
        parser.error("a paired comparison runs one concurrency level")
    if args.url is None and "," in args.server:
        parser.error("each arm runs one server")
    return args


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        epilog="Every other option is a bench_load.py option shared by both arms.",
    )
    parser.add_argument("--a-args", default="", help="bench_load options for arm A")
    parser.add_argument("--b-args", default="", help="bench_load options for arm B")
    parser.add_argument("--pairs", type=int, default=8)
    parser.add_argument(
        "--cooldown", type=float, default=5, help="seconds between runs"
    )
    parser.add_argument("--requests", type=int, default=32, help="requests per run")
    parser.add_argument("--boot-seed", type=int, default=0)
    parser.add_argument(
        "--max-load-delta",
        type=float,
        help="drop a pair whose runs started at 1-min loads further apart than "
        "this, for example 1.0; off unless given, and declared (on the A/B "
        "card) before the run, never fitted afterwards",
    )
    own, shared = parser.parse_known_args()
    arms = {"A": arm_args(shared, own.a_args), "B": arm_args(shared, own.b_args)}
    tokenizers: dict[Path, Callable] = {}

    runs = itertools.count()

    def measure(arm: str) -> dict:
        # Each run gets its own server log and request journal directory.
        args = copy.copy(arms[arm])
        args.log_dir = arms[arm].log_dir / f"run{next(runs):02d}-{arm}"
        if args.model_path not in tokenizers:
            tokenizers[args.model_path] = bench_load.tokenizer_counter(args.model_path)
        name = args.label if args.url else args.server
        set_name = bench_load.prompt_sets(argparse.ArgumentParser(), args)[0]
        print(f"arm {arm} ({name}):", flush=True)
        return bench_load.measure_level(
            name,
            set_name,
            "concurrency",
            args.concurrency[0],
            own.requests,
            args,
            tokenizers[args.model_path],
        )

    pairs = run_pairs(own.pairs, measure, time.sleep, own.cooldown)
    load_dropped = drop_load_swings(pairs, own.max_load_delta)
    summary = summarize_pairs(pairs, random.Random(own.boot_seed))
    summary["max_load_delta"] = own.max_load_delta
    summary["load_dropped"] = load_dropped
    print()
    print(render(summary, pairs))
    path = arms["A"].json  # bench_load's --json, shared by both arms.
    if path:
        report = {
            "argv": sys.argv,
            "system": bench_system.system_info(),
            "arms": {
                arm: {"args": vars(args) | {"file_prompts": None}}
                for arm, args in arms.items()
            },
            "summary": summary,
            "pairs": pairs,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=1, default=str) + "\n")
    return 1 if "refused" in summary else 0


if __name__ == "__main__":
    sys.exit(main())
