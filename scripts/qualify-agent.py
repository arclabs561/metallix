#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Qualify native read tools from structured agent receipts; dry-run by default."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
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
        "expected_calls": (
            {
                "name": "search_file",
                "path": "inventory.txt",
                "arguments": {"path": "inventory.txt", "query": "beta"},
            },
        ),
    },
    {
        "name": "two_file_chain",
        "prompt": "Read pointer.txt and follow its instruction to find the launch code. Report the launch code.",
        "answer": "ORCHID-728",
        "expected_calls": (
            {
                "name": "read_file",
                "path": "pointer.txt",
                "arguments": {"path": "pointer.txt"},
            },
            {
                "name": "read_file",
                "path": "detail.txt",
                "arguments": {"path": "detail.txt"},
            },
        ),
    },
    {
        "name": "long_read",
        "prompt": "Read long.txt. Report the final marker at the end of the file.",
        "answer": "LANTERN-493",
        "expected_calls": (
            {
                "name": "read_file",
                "path": "long.txt",
                "arguments": {"path": "long.txt"},
            },
        ),
    },
    {
        "name": "held_out_factual_join",
        "prompt": "Read cedar.txt, then signal.txt. Combine their values as the hyphenated launch code.",
        "answer": "CEDAR-441",
        "expected_calls": (
            {
                "name": "read_file",
                "path": "cedar.txt",
                "arguments": {"path": "cedar.txt"},
            },
            {
                "name": "read_file",
                "path": "signal.txt",
                "arguments": {"path": "signal.txt"},
            },
        ),
    },
)
FILES = {
    "inventory.txt": "alpha: copper\nbeta: indigo\ngamma: amber\n",
    "pointer.txt": "The answer is in detail.txt. Read that file next.\n",
    "detail.txt": "The launch code is ORCHID-728.\n",
    "long.txt": "ordinary note " * 700 + "\nThe final marker is LANTERN-493.\n",
    "cedar.txt": "The first launch fragment is CEDAR.\n",
    "signal.txt": "The second launch fragment is 441.\n",
}


def positive(raw: str) -> int:
    value = int(raw)
    if value < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def canonical_json(value: object) -> str:
    """Serialize JSON in the same recursively canonical form as `mx agent --json`."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def arguments_sha256(arguments: object) -> str:
    return hashlib.sha256(canonical_json(arguments).encode()).hexdigest()


def expected_calls(case: dict) -> list[dict]:
    return [
        {
            "name": call["name"],
            "relative_path": call["path"],
            "arguments_sha256": arguments_sha256(call["arguments"]),
        }
        for call in case["expected_calls"]
    ]


def case_plan(case: dict) -> dict:
    return {
        "name": case["name"],
        "prompt": case["prompt"],
        "answer": case["answer"],
        "expected_calls": expected_calls(case),
    }


def parse_agent_receipt(stdout: str) -> dict | None:
    try:
        value = json.loads(stdout)
    except json.JSONDecodeError:
        return None
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        return None
    if value.get("status") not in {"completed", "failed"}:
        return None
    turns = value.get("turns")
    if not isinstance(turns, list):
        return None
    final_text = value.get("final_text")
    if final_text is not None and not isinstance(final_text, str):
        return None
    for index, turn in enumerate(turns):
        if not isinstance(turn, dict):
            return None
        if turn.get("turn_index") != index or isinstance(turn.get("turn_index"), bool):
            return None
        if turn.get("finish_reason") not in {"eos", "length"}:
            return None
        if not isinstance(turn.get("metrics"), dict):
            return None
        digest = turn.get("generated_text_sha256")
        if not isinstance(digest, str) or len(digest) != 64:
            return None
        try:
            int(digest, 16)
        except ValueError:
            return None
        byte_count = turn.get("generated_text_utf8_bytes")
        if (
            not isinstance(byte_count, int)
            or isinstance(byte_count, bool)
            or byte_count < 0
        ):
            return None
    return value


def receipt_calls(receipt: dict) -> list[dict] | None:
    calls = []
    for turn in receipt["turns"]:
        if not isinstance(turn, dict) or not isinstance(turn.get("calls"), list):
            return None
        for call in turn["calls"]:
            if not isinstance(call, dict):
                return None
            if not isinstance(call.get("name"), str):
                return None
            relative_path = call.get("relative_path")
            if relative_path is not None and not isinstance(relative_path, str):
                return None
            if not isinstance(call.get("arguments_sha256"), str):
                return None
            if call.get("outcome") not in {"ok", "error"}:
                return None
            calls.append(
                {
                    "name": call["name"],
                    "relative_path": relative_path,
                    "arguments_sha256": call["arguments_sha256"],
                    "outcome": call["outcome"],
                }
            )
    return calls


def phase_costs(receipt: dict | None) -> dict | None:
    """Summarize one agent receipt without double-counting session setup."""
    if receipt is None or not receipt["turns"]:
        return None
    metrics = [
        turn.get("metrics") for turn in receipt["turns"] if isinstance(turn, dict)
    ]
    if len(metrics) != len(receipt["turns"]) or not all(
        isinstance(metric, dict) for metric in metrics
    ):
        return None
    required = (
        "session_load_ms",
        "prefill_ms",
        "decode_total_ms",
        "prompt_tokens",
        "generated_tokens",
    )
    if not all(
        all(
            isinstance(metric.get(key), (int, float))
            and not isinstance(metric.get(key), bool)
            and math.isfinite(metric[key])
            for key in required
        )
        for metric in metrics
    ):
        return None
    return {
        "session_load_ms": metrics[0]["session_load_ms"],
        "prefill_ms": sum(metric["prefill_ms"] for metric in metrics),
        "decode_total_ms": sum(metric["decode_total_ms"] for metric in metrics),
        "prompt_tokens": sum(metric["prompt_tokens"] for metric in metrics),
        "generated_tokens": sum(metric["generated_tokens"] for metric in metrics),
    }


def contains_ordered_calls(calls: list[dict], expected: list[dict]) -> bool:
    """Require each planned execution while permitting successful extra exploration."""
    expected_index = 0
    for call in calls:
        observed = {
            key: call[key] for key in ("name", "relative_path", "arguments_sha256")
        }
        if expected_index < len(expected) and observed == expected[expected_index]:
            expected_index += 1
    return expected_index == len(expected)


def assess(case: dict, status: int | None, stdout: str, stderr: str) -> dict:
    """Keep process, completed execution, evidence, and answer as separate gates."""
    receipt = parse_agent_receipt(stdout)
    calls = receipt_calls(receipt) if receipt is not None else None
    expected = expected_calls(case)
    checks = {
        "process_success": status == 0,
        "structured_receipt_valid": calls is not None,
        "execution_completed": receipt is not None and receipt["status"] == "completed",
        "answer_present": receipt is not None
        and isinstance(receipt.get("final_text"), str)
        and case["answer"].casefold() in receipt["final_text"].casefold(),
        "required_executions_observed": calls is not None
        and contains_ordered_calls(calls, expected),
        "all_executions_succeeded": calls is not None
        and all(call["outcome"] == "ok" for call in calls),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "agent_receipt": receipt,
        "executed_calls": calls,
        "extra_execution_count": (
            len(calls) - len(expected)
            if checks["required_executions_observed"]
            else None
        ),
        "phase_costs": phase_costs(receipt),
        "stderr": stderr,
    }


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
        "schema_version": 2,
        "scope": "Synthetic read-tool tasks, not general coding-agent qualification",
        "cases": [case_plan(case) for case in CASES],
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
                    "--json",
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
