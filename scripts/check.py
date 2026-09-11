#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Run Metallix's bounded local quality gate without model downloads."""

import argparse
import os
import platform
import signal
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def run(command: list[str], timeout_seconds: int, *, docs: bool = False) -> int:
    print("+", " ".join(command), flush=True)
    environment = os.environ | ({"RUSTDOCFLAGS": "-Dwarnings"} if docs else {})
    try:
        process = subprocess.Popen(
            command, cwd=ROOT, env=environment, shell=False, start_new_session=True
        )
    except OSError as error:
        print(f"could not start {command[0]}: {error}", file=sys.stderr)
        return 1
    try:
        return process.wait(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        status = f"timed out after {timeout_seconds}s"
    except KeyboardInterrupt:
        status = "interrupted"
    print(f"{status}: {' '.join(command)}", file=sys.stderr)
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()
    if status == "interrupted":
        return 130
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metal", action="store_true", help="include Metal feature checks"
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=300,
        help="per-command timeout (default: 300)",
    )
    args = parser.parse_args()
    if args.timeout_seconds <= 0:
        parser.error("--timeout-seconds must be positive")
    if args.metal and (platform.system() != "Darwin" or platform.machine() != "arm64"):
        parser.error("--metal requires Darwin arm64")
    feature_args = ["--all-features"] if args.metal else []
    commands = [
        (["cargo", "fmt", "--check"], False),
        (["cargo", "test", "--workspace", *feature_args], False),
        (
            [
                "cargo",
                "clippy",
                "--workspace",
                "--all-targets",
                *feature_args,
                "--",
                "-D",
                "warnings",
            ],
            False,
        ),
        (["cargo", "doc", "--workspace", "--no-deps", *feature_args], True),
        ([sys.executable, "scripts/test_benchmark_qwen.py"], False),
        (["node", "--test", "scripts/benchmark-openai.test.mjs"], False),
        (["ruff", "check", "scripts"], False),
        (["ruff", "format", "--check", "scripts"], False),
    ]
    for command, docs in commands:
        if run(command, args.timeout_seconds, docs=docs) != 0:
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
