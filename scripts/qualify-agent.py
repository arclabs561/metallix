#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Qualify native read tools on synthetic tasks; dry-run by default, no downloads."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import time
from pathlib import Path

CASES = (
    {
        "name": "literal_search",
        "prompt": "Use search_file to find beta in inventory.txt. What color is beta?",
        "answer": "indigo",
        "tools": {"search_file": 1},
    },
    {
        "name": "two_file_chain",
        "prompt": "Read pointer.txt and follow its instruction to find the launch code. Report the launch code.",
        "answer": "ORCHID-728",
        "tools": {"read_file": 2},
    },
    {
        "name": "long_read",
        "prompt": "Read long.txt. Report the final marker at the end of the file.",
        "answer": "LANTERN-493",
        "tools": {"read_file": 1},
    },
)
FILES = {
    "inventory.txt": "alpha: copper\nbeta: indigo\ngamma: amber\n",
    "pointer.txt": "The answer is in detail.txt. Read that file next.\n",
    "detail.txt": "The launch code is ORCHID-728.\n",
    "long.txt": "ordinary note " * 700 + "\nThe final marker is LANTERN-493.\n",
}


def positive(raw: str) -> int:
    value = int(raw)
    if value < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def assess(case: dict, status: int | None, stdout: str, stderr: str) -> dict:
    """Exit success is necessary, but task evidence and the answer must also match."""
    calls = [line[6:] for line in stderr.splitlines() if line.startswith("tool: ")]
    checks = {
        "process_success": status == 0,
        "answer_present": case["answer"].casefold() in stdout.casefold(),
        "required_tools_observed": all(
            calls.count(tool) >= count for tool, count in case["tools"].items()
        ),
    }
    return {"passed": all(checks.values()), "checks": checks, "tool_calls": calls}


def execute(command: list[str], timeout: int) -> tuple[int | None, str, str]:
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout)
        return process.returncode, stdout, stderr
    except (subprocess.TimeoutExpired, KeyboardInterrupt) as error:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        stdout, stderr = process.communicate()
        if isinstance(error, KeyboardInterrupt):
            raise
        return None, stdout, stderr


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--binary", type=Path, default=Path("target/release/mx"))
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--expected-model-revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=positive, default=3)
    parser.add_argument("--timeout-seconds", type=positive, default=60)
    args = parser.parse_args()
    plan = {
        "schema_version": 1,
        "scope": "Synthetic read-tool tasks, not general coding-agent qualification",
        "cases": CASES,
        "repeats": args.repeats,
        "max_tokens": 256,
        "max_turns": 6,
        "context_tokens": 2048,
        "model_revision": args.expected_model_revision,
        "workspace_sha256": hashlib.sha256(
            json.dumps(FILES, sort_keys=True).encode()
        ).hexdigest(),
    }
    if not args.run:
        print(json.dumps({"status": "dry_run", **plan}, indent=2))
        return 0
    if (
        args.model.resolve().parent.name != "snapshots"
        or args.model.resolve().name != args.expected_model_revision
    ):
        parser.error("model snapshot directory does not match expected revision")
    if not args.model.is_dir() or not args.binary.is_file():
        parser.error("model directory and binary must already exist")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("output must be a new or empty directory")
    args.output.mkdir(parents=True, exist_ok=True)
    runs = []
    receipt = {**plan, "runs": runs, "status": "running"}
    receipt_path = args.output / "receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    try:
        workspace = args.output / "workspace"
        workspace.mkdir()
        for name, content in FILES.items():
            (workspace / name).write_text(content)
        receipt["binary_sha256"] = hashlib.sha256(args.binary.read_bytes()).hexdigest()
        for case in CASES:
            for repeat in range(args.repeats):
                command = [
                    str(args.binary.resolve()),
                    "agent",
                    "--model",
                    str(args.model.resolve()),
                    "--workspace",
                    str(workspace.resolve()),
                    "--prompt",
                    case["prompt"],
                    "--max-tokens",
                    "256",
                    "--max-turns",
                    "6",
                    "--context-tokens",
                    "2048",
                ]
                started = time.monotonic()
                status, stdout, stderr = execute(command, args.timeout_seconds)
                row = {
                    "case": case["name"],
                    "repeat": repeat,
                    "exit_code": status,
                    "fresh_process_wall_ms": (time.monotonic() - started) * 1000,
                    "stdout": stdout,
                    "stderr": stderr,
                    **assess(case, status, stdout, stderr),
                }
                runs.append(row)
                (args.output / "receipt.json").write_text(
                    json.dumps(receipt, indent=2) + "\n"
                )
                print(
                    f"{case['name']} trial {repeat + 1}: {'pass' if row['passed'] else 'FAIL'}",
                    flush=True,
                )
        passed = all(row["passed"] for row in runs)
        receipt["status"] = "passed" if passed else "failed"
        (args.output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
        return 0 if passed else 1
    except (OSError, KeyboardInterrupt) as error:
        receipt["status"] = (
            "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        )
        receipt["error"] = str(error) or "interrupted by operator"
        receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
        return 130 if isinstance(error, KeyboardInterrupt) else 1


if __name__ == "__main__":
    raise SystemExit(main())
