#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Dry-run-first synthetic schedule evaluation for bounded candidate control.

The four tasks and two seeds below are frozen before any model execution. This
is a synthetic mechanism check, not a held-out benchmark. Candidate retries
receive more generation work than the schema-only baseline, so this reports
quality and cost separately and makes neither a speed nor calibration claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

MAX_ATTEMPTS = 4
SEEDS = (17, 43)
# Frozen before the first confirmation run; never tune these rows after scoring.
CONFIRMATION_SEEDS = (101, 211)
CONFIRMATION_CASES = (
    (
        "offset_pair",
        "Schedule two non-overlapping intervals with durations 1 and 4, entirely between 3 and 12.",
        2,
        (1, 4),
        (3, 12),
    ),
    (
        "tight_three",
        "Fit three non-overlapping intervals of lengths 2, 3, and 1 into the window from 4 to 10. Touching endpoints are allowed.",
        3,
        (2, 3, 1),
        (4, 10),
    ),
    (
        "uneven_four",
        "Place four intervals, of lengths 1, 3, 2, and 4, between 2 and 17 without overlap.",
        4,
        (1, 3, 2, 4),
        (2, 17),
    ),
    (
        "late_pair",
        "Return a schedule of two disjoint intervals lasting 4 and 2 units, both inside the time window 11 to 19.",
        2,
        (4, 2),
        (11, 19),
    ),
)
CASES = (
    (
        "two_short",
        "Return two non-overlapping appointments, each two units long, between 0 and 8.",
        2,
        (2, 2),
        (0, 8),
    ),
    (
        "mixed_three",
        "Return three non-overlapping appointments of lengths 1, 2, and 3 between 0 and 12.",
        3,
        (1, 2, 3),
        (0, 12),
    ),
    (
        "adjacent_pair",
        "Return two non-overlapping appointments of length 3 between 2 and 10. Adjacent appointments are allowed.",
        2,
        (3, 3),
        (2, 10),
    ),
    (
        "wide_four",
        "Return four non-overlapping appointments of lengths 1, 1, 2, and 2 between 0 and 16.",
        4,
        (1, 1, 2, 2),
        (0, 16),
    ),
)
SCHEDULE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["intervals"],
    "properties": {
        "intervals": {
            "type": "array",
            "minItems": 1,
            "maxItems": 4,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["start", "end"],
                "properties": {
                    "start": {"type": "integer", "minimum": 0, "maximum": 24},
                    "end": {"type": "integer", "minimum": 0, "maximum": 24},
                },
            },
        }
    },
}


@dataclass(frozen=True)
class Task:
    name: str
    prompt: str
    count: int
    durations: tuple[int, ...]
    window: tuple[int, int]


@dataclass(frozen=True)
class RunPlan:
    arm: str
    task: Task
    seed: int
    command: tuple[str, ...]
    requirements: dict[str, Any] | None = None
    requirements_path: Path | None = None


def frozen_tasks(suite: str = "development") -> tuple[Task, ...]:
    """Return the fixed task set; callers have no task or seed selection knob."""
    if suite == "development":
        cases = CASES
    elif suite == "confirmation-v1":
        cases = CONFIRMATION_CASES
    else:
        raise ValueError(f"unknown frozen suite {suite!r}")
    return tuple(Task(*case) for case in cases)


def suite_seeds(suite: str = "development") -> tuple[int, ...]:
    if suite == "development":
        return SEEDS
    if suite == "confirmation-v1":
        return CONFIRMATION_SEEDS
    raise ValueError(f"unknown frozen suite {suite!r}")


def task_hash(
    max_tokens: int, max_candidate_ms: int, suite: str = "development"
) -> str:
    frozen = {
        "cases": [asdict(task) for task in frozen_tasks(suite)],
        "max_attempts": MAX_ATTEMPTS,
        "max_candidate_ms": max_candidate_ms,
        "max_tokens": max_tokens,
        "schema": SCHEDULE_SCHEMA,
        "seeds": suite_seeds(suite),
        "temperature": 1,
    }
    encoded = json.dumps(frozen, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def requirement_payload(task: Task) -> dict[str, Any]:
    """Project frozen task fields into the verifier's structured contract."""
    return {
        "durations": list(task.durations),
        "window": {"start": task.window[0], "end": task.window[1]},
    }


def requirement_path(output: Path | None, task: Task, seed: int) -> Path:
    relative = Path("requirements") / f"candidate-{task.name}-seed{seed}.json"
    return output / relative if output is not None else relative


def command_for(
    binary: Path,
    model: Path,
    task: Task,
    seed: int,
    arm: str,
    max_tokens: int,
    max_candidate_ms: int,
    requirements_file: Path | None = None,
    chat_template: bool = False,
) -> tuple[str, ...]:
    command = [
        str(binary.resolve()),
        "gen",
        "--model",
        str(model.resolve()),
        "--prompt",
        task.prompt,
        "--sample",
        "--temperature",
        "1",
        "--seed",
        str(seed),
        "--max-tokens",
        str(max_tokens),
        "--json-schema-inline",
        json.dumps(SCHEDULE_SCHEMA, sort_keys=True, separators=(",", ":")),
    ]
    if chat_template:
        command.append("--chat-template")
    if arm == "candidate":
        command.extend(
            [
                "--verify-schedule",
                "--max-attempts",
                str(MAX_ATTEMPTS),
                "--max-candidate-ms",
                str(max_candidate_ms),
            ]
        )
        if requirements_file is not None:
            command.extend(["--schedule-requirements", str(requirements_file)])
    elif arm != "baseline":
        raise ValueError(f"unknown arm {arm!r}")
    return tuple(command)


def plan(args: argparse.Namespace) -> tuple[RunPlan, ...]:
    return tuple(
        RunPlan(
            arm=arm,
            task=task,
            seed=seed,
            command=command_for(
                args.binary,
                args.model,
                task,
                seed,
                arm,
                args.max_tokens,
                args.max_candidate_ms,
                requirement_path(args.output, task, seed)
                if args.task_aware and arm == "candidate"
                else None,
                chat_template=getattr(args, "chat_template", False),
            ),
            requirements=requirement_payload(task)
            if args.task_aware and arm == "candidate"
            else None,
            requirements_path=requirement_path(args.output, task, seed)
            if args.task_aware and arm == "candidate"
            else None,
        )
        for task in frozen_tasks(getattr(args, "suite", "development"))
        for seed in suite_seeds(getattr(args, "suite", "development"))
        for arm in ("baseline", "candidate")
    )


def write_requirements(plans: tuple[RunPlan, ...], output: Path) -> None:
    """Persist only the task-aware candidate verifier inputs under output."""
    for plan_item in plans:
        if plan_item.requirements is None:
            continue
        if plan_item.requirements_path is None:
            raise ValueError("task-aware plan has no requirement file path")
        path = plan_item.requirements_path
        try:
            path.relative_to(output)
        except ValueError as error:
            raise ValueError(
                "requirement path must be below the output directory"
            ) from error
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(plan_item.requirements, sort_keys=True, separators=(",", ":"))
            + "\n"
        )


def semantic_valid(value: object) -> bool:
    """Independent pairwise half-open overlap oracle, unlike runtime sorting."""
    if not isinstance(value, dict):
        return False
    intervals = value.get("intervals")
    if not isinstance(intervals, list) or not 1 <= len(intervals) <= 128:
        return False
    for interval in intervals:
        if not isinstance(interval, dict):
            return False
        start, end = interval.get("start"), interval.get("end")
        if type(start) not in (int, float) or type(end) not in (int, float):
            return False
        if not math.isfinite(start) or not math.isfinite(end) or start >= end:
            return False
    return all(
        left["end"] <= right["start"] or right["end"] <= left["start"]
        for index, left in enumerate(intervals)
        for right in intervals[index + 1 :]
    )


def task_adherent(value: object, task: Task) -> bool:
    if not semantic_valid(value):
        return False
    intervals = value["intervals"]
    if len(intervals) != task.count:
        return False
    window_start, window_end = task.window
    if any(
        interval["start"] < window_start or interval["end"] > window_end
        for interval in intervals
    ):
        return False
    return sorted(
        interval["end"] - interval["start"] for interval in intervals
    ) == sorted(task.durations)


def receipt_value(report: dict[str, Any]) -> object | None:
    constraint = report.get("constraint")
    if not isinstance(constraint, dict):
        return None
    generated = constraint.get("generated_text")
    if not isinstance(generated, str):
        return None
    try:
        return json.loads(generated)
    except json.JSONDecodeError:
        return None


def generated_work(report: dict[str, Any], arm: str) -> int:
    if arm == "candidate":
        verification = report.get("candidate_verification")
        if not isinstance(verification, dict):
            raise ValueError("candidate receipt lacks candidate_verification")
        work = verification.get("total_generated_tokens")
    else:
        generated = report.get("generated_ids")
        if not isinstance(generated, list):
            raise ValueError("baseline receipt lacks generated IDs")
        work = len(generated)
    if type(work) is not int or work < 0:
        raise ValueError("receipt has invalid generated work")
    return work


def receipt_elapsed_ms(report: dict[str, Any], arm: str) -> float | None:
    if arm == "candidate":
        verification = report.get("candidate_verification")
        costs = verification.get("costs_ms") if isinstance(verification, dict) else None
        required = {
            "load",
            "prefill",
            "constraint_setup",
            "fork_total",
            "decode_total",
            "verifier_total",
            "request_total",
        }
        if not isinstance(costs, dict) or not required.issubset(costs):
            return None
        value = costs["request_total"]
    else:
        decode = report.get("decode_ms")
        load = report.get("load_ms")
        prefill = report.get("prefill_ms")
        if (
            not isinstance(decode, list)
            or type(load) not in (int, float)
            or type(prefill) not in (int, float)
        ):
            return None
        if any(type(item) not in (int, float) for item in decode):
            return None
        value = sum(decode) + load + prefill
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        return None
    return float(value)


def score(
    report: dict[str, Any], plan_item: RunPlan, returncode: int, wall_elapsed_ms: float
) -> dict[str, Any]:
    value = receipt_value(report)
    accepted: bool | None = None
    exhausted: bool | None = None
    if plan_item.arm == "candidate":
        verification = report.get("candidate_verification")
        if not isinstance(verification, dict):
            raise ValueError("candidate receipt lacks verification status")
        status = verification.get("status")
        if status not in ("accepted", "exhausted"):
            raise ValueError("candidate receipt has unknown verification status")
        accepted = status == "accepted"
        exhausted = status == "exhausted"
        if accepted != (returncode == 0):
            raise ValueError("candidate exit code disagrees with acceptance")
        if exhausted and (
            returncode == 0
            or report.get("generated_ids")
            or receipt_value(report) is not None
        ):
            raise ValueError("exhausted candidate published output or succeeded")
        if accepted and not semantic_valid(value):
            raise ValueError(
                "accepted candidate is not independently semantically valid"
            )
    elapsed = receipt_elapsed_ms(report, plan_item.arm)
    if elapsed is None:
        raise ValueError("receipt has missing or invalid elapsed costs")
    return {
        "arm": plan_item.arm,
        "case": plan_item.task.name,
        "seed": plan_item.seed,
        "exit_code": returncode,
        "accepted": accepted,
        "exhausted": exhausted,
        "semantic_valid": semantic_valid(value),
        "task_adherent": task_adherent(value, plan_item.task),
        "generated_work_tokens": generated_work(report, plan_item.arm),
        "receipt_elapsed_ms": elapsed,
        "wall_elapsed_ms": wall_elapsed_ms,
        "error": None,
    }


def aggregate(rows: list[dict[str, Any]], arm: str) -> dict[str, Any]:
    selected = [row for row in rows if row["arm"] == arm]
    work = [
        row["generated_work_tokens"]
        for row in selected
        if row["generated_work_tokens"] is not None
    ]
    receipt_elapsed = [
        row["receipt_elapsed_ms"]
        for row in selected
        if row["receipt_elapsed_ms"] is not None
    ]
    wall_elapsed = [
        row["wall_elapsed_ms"] for row in selected if row["wall_elapsed_ms"] is not None
    ]
    return {
        "runs": len(selected),
        "errors": sum(row["error"] is not None for row in selected),
        "semantic_valid": sum(row["semantic_valid"] for row in selected),
        "task_adherent": sum(row["task_adherent"] for row in selected),
        "accepted": sum(row["accepted"] is True for row in selected),
        "exhausted": sum(row["exhausted"] is True for row in selected),
        "generated_work_tokens": sum(work),
        "generated_work_measured_runs": len(work),
        "receipt_elapsed_ms": sum(receipt_elapsed),
        "receipt_elapsed_measured_runs": len(receipt_elapsed),
        "wall_elapsed_ms": sum(wall_elapsed),
        "wall_elapsed_measured_runs": len(wall_elapsed),
    }


def run(
    plan_item: RunPlan, output: Path, timeout_seconds: int
) -> tuple[dict[str, Any], int, float]:
    started = time.monotonic()
    completed = subprocess.run(
        plan_item.command,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
        check=False,
    )
    elapsed = (time.monotonic() - started) * 1000.0
    stem = f"{plan_item.arm}-{plan_item.task.name}-seed{plan_item.seed}"
    (output / f"{stem}.stdout.json").write_text(completed.stdout)
    (output / f"{stem}.stderr.log").write_text(completed.stderr)
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise ValueError(f"{stem} did not emit one JSON receipt") from error
    if not isinstance(payload, dict):
        raise TypeError(f"{stem} receipt is not an object")
    return payload, completed.returncode, elapsed


def error_row(
    plan_item: RunPlan, error: Exception, wall_elapsed_ms: float
) -> dict[str, Any]:
    """Retain failed runs in the denominator without inventing work or timing."""
    return {
        "arm": plan_item.arm,
        "case": plan_item.task.name,
        "seed": plan_item.seed,
        "exit_code": None,
        "accepted": False,
        "exhausted": False,
        "semantic_valid": False,
        "task_adherent": False,
        "generated_work_tokens": None,
        "receipt_elapsed_ms": None,
        "wall_elapsed_ms": wall_elapsed_ms,
        "error": f"{type(error).__name__}: {error}",
    }


def positive_at_most_120(value: str) -> int:
    parsed = int(value)
    if not 1 <= parsed <= 120:
        raise argparse.ArgumentTypeError("must be between 1 and 120 seconds")
    return parsed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--suite",
        choices=("development", "confirmation-v1"),
        default="development",
        help="fixed task/seed set; confirmation-v1 is disjoint from development",
    )
    parser.add_argument(
        "--chat-template",
        action="store_true",
        help="render each frozen prompt as one user message using the checkpoint template",
    )
    parser.add_argument(
        "--task-aware",
        action="store_true",
        help="give only candidate verification the frozen task durations and window",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--max-candidate-ms", type=int, default=90_000)
    parser.add_argument("--timeout-seconds", type=positive_at_most_120, default=120)
    args = parser.parse_args(argv)
    if not 1 <= args.max_tokens <= 256:
        parser.error("--max-tokens must be between 1 and 256")
    if not 1 <= args.max_candidate_ms <= 110_000:
        parser.error("--max-candidate-ms must be between 1 and 110000")
    plans = plan(args)
    manifest = {
        "schema_version": 1,
        "task_hash": task_hash(args.max_tokens, args.max_candidate_ms, args.suite),
        "status": "execute" if args.execute else "dry_run",
        "prompt_format": "checkpoint_chat" if args.chat_template else "plain",
        "workload": {
            "suite": args.suite,
            "cases": len(frozen_tasks(args.suite)),
            "seeds": list(suite_seeds(args.suite)),
            "arms": ["baseline", "candidate"],
            "max_attempts": MAX_ATTEMPTS,
            "per_case_timeout_seconds": args.timeout_seconds,
        },
        "verifier": {
            "contract": "schedule_requirements_v1"
            if args.task_aware
            else "non_overlapping_schedule_v1",
            "task_aware": args.task_aware,
            "max_attempts": MAX_ATTEMPTS,
            "max_candidate_ms": args.max_candidate_ms,
            "requirements_source": "frozen task durations and window"
            if args.task_aware
            else None,
        },
        "scope": "fixed synthetic scheduling mechanism check; candidate retries have a higher generation-work budget than baseline; this is neither a speedup nor calibrated-quality claim and not a held-out benchmark",
        "plans": [
            {
                "arm": item.arm,
                "case": item.task.name,
                "seed": item.seed,
                "command": item.command,
                "schedule_requirements": (
                    {
                        "path": str(item.requirements_path),
                        "payload": item.requirements,
                    }
                    if item.requirements is not None
                    else None
                ),
            }
            for item in plans
        ],
    }
    if not args.execute:
        print(json.dumps(manifest, indent=2))
        return 0
    if args.output is None:
        parser.error("--execute requires --output for receipts and logs")
    args.output.mkdir(parents=True, exist_ok=True)
    write_requirements(plans, args.output)
    (args.output / "plan.json").write_text(json.dumps(manifest, indent=2) + "\n")
    rows = []
    for item in plans:
        started = time.monotonic()
        try:
            report, returncode, elapsed = run(item, args.output, args.timeout_seconds)
            rows.append(score(report, item, returncode, elapsed))
        except (OSError, subprocess.TimeoutExpired, TypeError, ValueError) as error:
            rows.append(error_row(item, error, (time.monotonic() - started) * 1000.0))
    manifest["results"] = rows
    manifest["summary"] = {
        arm: aggregate(rows, arm) for arm in ("baseline", "candidate")
    }
    (args.output / "summary.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    return int(any(row["error"] is not None for row in rows))


if __name__ == "__main__":
    raise SystemExit(main())
