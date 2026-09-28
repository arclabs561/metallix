#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Bounded real-CLI semantic candidate checks; dry-run unless --execute is set."""

import argparse
import json
import math
import subprocess
from pathlib import Path


def non_overlapping(value: object) -> bool:
    """Independent pairwise oracle, deliberately unlike the sorted runtime check."""
    if not isinstance(value, dict):
        return False
    rows = value.get("intervals")
    if not isinstance(rows, list) or not 1 <= len(rows) <= 128:
        return False
    for row in rows:
        if not isinstance(row, dict):
            return False
        for key in ("start", "end"):
            x = row.get(key)
            if type(x) not in (int, float) or not math.isfinite(x):
                return False
        if row["start"] >= row["end"]:
            return False
    return all(
        a["end"] <= b["start"] or b["end"] <= a["start"]
        for i, a in enumerate(rows)
        for b in rows[i + 1 :]
    )


def check_receipt(report: dict, exit_code: int, accepted: bool) -> None:
    verification = report["candidate_verification"]
    attempts = verification["attempts"]
    assert attempts and len(attempts) <= verification["max_attempts"]
    assert len({attempt["seed"] for attempt in attempts}) == len(attempts)
    assert verification["total_generated_tokens"] == sum(
        attempt["generated_tokens"] for attempt in attempts
    )
    assert (
        verification["total_generated_tokens"]
        <= verification["max_total_generated_tokens"]
    )
    if accepted:
        assert exit_code == 0 and verification["status"] == "accepted"
        assert attempts[-1]["status"] == "accepted"
        assert non_overlapping(json.loads(report["constraint"]["generated_text"]))
        assert report["generated_ids"]
        assert report["cached_tokens"] == len(report["input_ids"]) + len(
            report["generated_ids"]
        )
    else:
        assert exit_code != 0 and verification["status"] == "exhausted"
        assert all(a["status"] != "accepted" for a in attempts)
        assert not report.get("generated_ids")
        assert not report.get("generated_text")
        assert not report.get("constraint", {}).get("generated_text")
        assert report["cached_tokens"] == len(report["input_ids"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    cases = (
        ("touching", [(0, 1), (1, 2)], True),
        ("nested_overlap", [(0, 3), (1, 2)], False),
    )
    results = []
    for name, spans, accepted in cases:
        value = {"intervals": [{"start": a, "end": b} for a, b in spans]}
        command = [
            str(args.binary.resolve()),
            "gen",
            "--model",
            str(args.model.resolve()),
            "--prompt",
            "Return a schedule as JSON.",
            "--sample",
            "--seed",
            "41",
            "--temperature",
            "1",
            "--max-tokens",
            "128",
            "--verify-schedule",
            "--max-attempts",
            "2",
            "--json-schema-inline",
            json.dumps({"const": value}),
        ]
        if not args.execute:
            results.append({"case": name, "command": command})
            continue
        completed = subprocess.run(
            command, capture_output=True, text=True, timeout=120, check=False
        )
        report = json.loads(completed.stdout)
        check_receipt(report, completed.returncode, accepted)
        results.append(
            {"case": name, "exit_code": completed.returncode, "report": report}
        )
    for token_budget in (1, 2):
        command = [
            str(args.binary.resolve()),
            "gen",
            "--model",
            str(args.model.resolve()),
            "--max-tokens",
            str(token_budget),
            "--verify-cache",
            "--json-schema-inline",
            json.dumps({"const": {"receipt": "comparison coverage"}}),
        ]
        if not args.execute:
            results.append(
                {"case": f"branch_budget_{token_budget}", "command": command}
            )
            continue
        completed = subprocess.run(
            command, capture_output=True, text=True, timeout=120, check=False
        )
        report = json.loads(completed.stdout)
        expected = token_budget - 1
        assert report["branch_comparison_count"] == expected
        assert len(report["branch_comparisons"]) == expected
        assert report["branch_checked_token_positions"] == ([3] if expected else [])
        assert report["branch_verification"] == (
            "resident_fork_parent_match" if expected else "not_run_no_eligible_decode"
        )
        results.append({"case": f"branch_budget_{token_budget}", "report": report})
    print(json.dumps({"executed": args.execute, "cases": results}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
