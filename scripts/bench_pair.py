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
import json
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
        missing = [arm for arm in "AB" if "summary" not in pair[arm]]
        if missing:
            pair["dropped"] = "; ".join(
                f"{arm}: {pair[arm].get('aborted') or pair[arm].get('error')}"
                for arm in missing
            )
        out.append(pair)
    return out


def pair_load(pair: dict) -> list[float | None]:
    """The 1-min load when each run of the pair started, A first."""
    return [(pair[arm].get("baseline") or {}).get("load_1m") for arm in "AB"]


def summarize_pairs(pairs: list[dict], rng: random.Random) -> dict:
    kept = [pair for pair in pairs if "dropped" not in pair]
    out: dict = {"pairs": len(pairs), "kept": len(kept), "metrics": {}}
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
        out["metrics"][name] = {
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
    for name, m in summary["metrics"].items():
        direction = "higher" if m["higher_is_better"] else "lower"
        lines.append(
            f"{name} B/A median {m['median_ratio']:.3f} "
            f"[95% CI {m['ci95'][0]:.3f}-{m['ci95'][1]:.3f}] ({direction} is better): "
            f"{m['verdict']}"
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
    own, shared = parser.parse_known_args()
    arms = {"A": arm_args(shared, own.a_args), "B": arm_args(shared, own.b_args)}
    tokenizers: dict[Path, Callable] = {}

    def measure(arm: str) -> dict:
        args = arms[arm]
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
    summary = summarize_pairs(pairs, random.Random(own.boot_seed))
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
    return 0


if __name__ == "__main__":
    sys.exit(main())
