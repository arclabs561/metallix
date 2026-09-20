#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Dry-run-first resident chat qualification; never starts a server or downloads a model."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import signal
import statistics
import subprocess
import sys
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path

LONG_PROMPT = "Explain this machine:" + " bicycle mechanism" * 983
SHORT_PROMPT = "Write a detailed explanation of how a bicycle works."
BUDGET = re.compile(r"received\s+(\d+)\s+\+\s+(\d+)\s+=\s+(\d+)")
RSS = re.compile(r"([0-9]+)\s+maximum resident set size")
ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class CliRun:
    text_sha256: str
    ids: tuple[int, ...]
    finish_reason: str
    metrics: dict[str, object]
    peak_rss_bytes: int


def positive(value: str) -> int:
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def model_revision(model: Path) -> str | None:
    parts = model.resolve().parts
    try:
        return parts[parts.index("snapshots") + 1]
    except (ValueError, IndexError):
        return None


def parse_budget_error(payload: object, requested_output: int) -> int:
    if not isinstance(payload, dict) or not isinstance(payload.get("error"), dict):
        raise TypeError("preflight did not return an error envelope")
    message = payload["error"].get("message")
    if not isinstance(message, str) or (match := BUDGET.search(message)) is None:
        raise ValueError("preflight did not return the context budget contract")
    prompt, output, total = map(int, match.groups())
    if output != requested_output or total != prompt + output:
        raise ValueError("preflight budget arithmetic is inconsistent")
    return prompt


def parse_cli(
    stdout: str,
    stderr: str,
    max_tokens: int,
    context_tokens: int,
    prompt_tokens: int,
) -> CliRun:
    payload = json.loads(stdout)
    if not isinstance(payload, dict) or payload.get("finish_reason") != "length":
        raise ValueError("CLI did not end at the expected output cap")
    text = payload.get("text")
    ids = payload.get("generated_token_ids")
    metrics = payload.get("metrics")
    match = RSS.search(stderr)
    if (
        not isinstance(text, str)
        or not isinstance(ids, list)
        or not isinstance(metrics, dict)
        or match is None
    ):
        raise ValueError("CLI receipt is incomplete")
    if len(ids) != max_tokens or any(
        type(token) is not int or token < 0 for token in ids
    ):
        raise ValueError("CLI token count is not the requested cap")
    expected_metrics = {
        "generated_tokens": max_tokens,
        "context_tokens": context_tokens,
        "prompt_tokens": prompt_tokens,
    }
    for field, expected in expected_metrics.items():
        if type(metrics.get(field)) is not int or metrics[field] != expected:
            raise ValueError(f"CLI metrics {field} disagrees with the workload")
    return CliRun(
        hashlib.sha256(text.encode()).hexdigest(),
        tuple(ids),
        "length",
        metrics,
        int(match.group(1)),
    )


def parse_http(
    stdout: str, max_tokens: int, expected_samples: int
) -> dict[str, object]:
    payload = json.loads(stdout)
    if not isinstance(payload, dict):
        raise TypeError("HTTP benchmark receipt is not an object")
    samples = payload.get("samples")
    if (
        payload.get("status") != "incomplete"
        or not isinstance(samples, list)
        or len(samples) != expected_samples
    ):
        raise ValueError("HTTP workload did not end expected-incomplete")
    hashes = []
    for sample in samples:
        if (
            not isinstance(sample, dict)
            or sample.get("terminal_status") != "incomplete"
            or sample.get("received_completed") is not False
            or sample.get("completion_tokens") != max_tokens
        ):
            raise ValueError("HTTP sample has an invalid capped terminal state")
        value = sample.get("output_text_sha256")
        if not isinstance(value, str) or len(value) != 64:
            raise ValueError("HTTP sample lacks output hash")
        hashes.append(value)
    if len(set(hashes)) != 1:
        raise ValueError("HTTP output hashes differ across same-workload trials")
    return {"hash": hashes[0], "samples": len(samples), "receipt": payload}


def command_run(
    command: list[str], output: Path, label: str, timeout: int
) -> subprocess.CompletedProcess[str]:
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired as error:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            # The process exited in the small timeout-to-kill race; drain it below.
            pass
        stdout, stderr = process.communicate()
        (output / f"{label}.stdout.log").write_text(stdout)
        (output / f"{label}.stderr.log").write_text(stderr)
        raise RuntimeError(f"{label} timed out after {timeout}s") from error
    (output / f"{label}.stdout.log").write_text(stdout)
    (output / f"{label}.stderr.log").write_text(stderr)
    return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)


def http_command(
    args: argparse.Namespace, prompt: str, label: str, output: Path
) -> dict[str, object]:
    result = command_run(
        [
            "node",
            str(ROOT / "scripts/benchmark-openai.mjs"),
            "--url",
            args.url,
            "--api",
            "responses",
            "--model",
            args.model_id,
            "--cache-condition",
            "warm",
            "--prompt",
            prompt,
            "--max-tokens",
            str(args.max_tokens),
            "--warmup",
            "0",
            "--requests",
            str(args.runs),
            "--concurrency",
            "1",
        ],
        output,
        label,
        args.timeout_seconds,
    )
    if result.returncode != 1:
        raise RuntimeError(f"{label} unexpectedly exited {result.returncode}")
    parsed = parse_http(result.stdout, args.max_tokens, args.runs)
    (output / f"{label}.receipt.json").write_text(
        json.dumps(parsed["receipt"], indent=2) + "\n"
    )
    return parsed


def cli_trials(
    args: argparse.Namespace, output: Path, prompt_tokens: int
) -> list[CliRun]:
    runs = []
    for index in range(1, args.runs + 1):
        result = command_run(
            [
                "/usr/bin/time",
                "-l",
                args.binary,
                "chat",
                "--model",
                args.model,
                "--prompt",
                LONG_PROMPT,
                "--max-tokens",
                str(args.max_tokens),
                "--context-tokens",
                str(args.context_tokens),
                "--kv-budget-mib",
                str(args.kv_budget_mib),
                "--json",
            ],
            output,
            f"cli-long-{index}",
            args.timeout_seconds,
        )
        if result.returncode:
            raise RuntimeError(f"CLI trial {index} failed")
        runs.append(
            parse_cli(
                result.stdout,
                result.stderr,
                args.max_tokens,
                args.context_tokens,
                prompt_tokens,
            )
        )
    if (
        len({run.text_sha256 for run in runs}) != 1
        or len({run.ids for run in runs}) != 1
    ):
        raise RuntimeError("CLI output differs across same-workload trials")
    return runs


def start_sample(output: Path, pid: int) -> subprocess.Popen[str]:
    """Sample only the explicitly named owned server process for five seconds."""
    return subprocess.Popen(
        [
            "/usr/bin/sample",
            str(pid),
            "5",
            "1",
            "-mayDie",
            "-file",
            str(output / "server-cpu.sample.txt"),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def finish_sample(sample: subprocess.Popen[str], output: Path) -> dict[str, object]:
    try:
        stdout, stderr = sample.communicate(timeout=10)
    except subprocess.TimeoutExpired as error:
        sample.terminate()
        stdout, stderr = sample.communicate()
        raise RuntimeError("bounded CPU sample did not finish") from error
    (output / "server-cpu.sample.stdout.log").write_text(stdout)
    (output / "server-cpu.sample.stderr.log").write_text(stderr)
    if sample.returncode:
        raise RuntimeError(f"bounded CPU sample exited {sample.returncode}")
    return {
        "pid": sample.args[1],
        "duration_seconds": 5,
        "artifact": "server-cpu.sample.txt",
    }


def preflight(args: argparse.Namespace) -> int:
    body = json.dumps(
        {
            "model": args.model_id,
            "input": LONG_PROMPT,
            "stream": False,
            "max_output_tokens": 256,
            "temperature": 0,
        }
    ).encode()
    request = urllib.request.Request(
        f"{args.url.rstrip('/')}/v1/responses",
        body,
        {"content-type": "application/json"},
        method="POST",
    )
    try:
        urllib.request.urlopen(request, timeout=args.timeout_seconds)
    except urllib.error.HTTPError as error:
        prompt_tokens = parse_budget_error(json.loads(error.read()), 256)
    else:
        raise RuntimeError("preflight unexpectedly generated")
    if prompt_tokens + args.max_tokens > args.context_tokens:
        raise RuntimeError("long prompt would overflow the configured context")
    return prompt_tokens


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--binary")
    parser.add_argument("--model")
    parser.add_argument("--url", default="http://127.0.0.1:8321")
    parser.add_argument("--model-id", default="metallix-qwen3")
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/chat-qualification")
    )
    parser.add_argument("--runs", type=positive, default=3)
    parser.add_argument("--max-tokens", type=positive, default=64)
    parser.add_argument("--context-tokens", type=positive, default=2048)
    parser.add_argument("--kv-budget-mib", type=positive, default=512)
    parser.add_argument("--timeout-seconds", type=positive, default=120)
    parser.add_argument("--expected-model-revision")
    parser.add_argument("--sample-pid", type=positive)
    args = parser.parse_args(argv)
    if (
        args.runs < 3
        or args.max_tokens > 256
        or args.context_tokens > 2048
        or args.kv_budget_mib > 8192
    ):
        parser.error(
            "runs must be >=3; max tokens <=256; context tokens <=2048; "
            "K/V budget MiB <=8192"
        )
    dry = {
        "status": "dry_run",
        "runs": args.runs,
        "context_tokens": args.context_tokens,
        "kv_budget_mib": args.kv_budget_mib,
        "long_prompt_sha256": hashlib.sha256(LONG_PROMPT.encode()).hexdigest(),
        "long_prompt_utf8_bytes": len(LONG_PROMPT.encode()),
        "sample": None
        if args.sample_pid is None
        else {"pid": args.sample_pid, "duration_seconds": 5},
    }
    if not args.run:
        print(json.dumps(dry, indent=2, sort_keys=True))
        return 0
    if not args.binary or not args.model:
        parser.error("--run requires --binary and --model")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("--output must be new or empty")
    revision = model_revision(Path(args.model))
    if not args.expected_model_revision:
        parser.error("--run requires --expected-model-revision")
    if revision != args.expected_model_revision:
        parser.error("model path does not match --expected-model-revision")
    args.output.mkdir(parents=True, exist_ok=True)
    try:
        prompt_tokens = preflight(args)
        short_before = http_command(
            args, SHORT_PROMPT, "http-short-before", args.output
        )
        cli = cli_trials(args, args.output, prompt_tokens)
        sampler = (
            start_sample(args.output, args.sample_pid) if args.sample_pid else None
        )
        try:
            long = http_command(args, LONG_PROMPT, "http-long", args.output)
        finally:
            sample_receipt = finish_sample(sampler, args.output) if sampler else None
        short_after = http_command(args, SHORT_PROMPT, "http-short-after", args.output)
        if short_before["hash"] != short_after["hash"]:
            raise RuntimeError("HTTP short output changed after the long workload")
        if long["hash"] != cli[0].text_sha256:
            raise RuntimeError("HTTP and CLI long outputs differ")
        rss_values = [run.peak_rss_bytes for run in cli]
        receipt = {
            "status": "completed",
            "model_revision_from_path": revision,
            "preflight_prompt_tokens": prompt_tokens,
            "context_tokens": args.context_tokens,
            "kv_budget_mib": args.kv_budget_mib,
            "long_total_tokens": prompt_tokens + args.max_tokens,
            "cli": [asdict(run) for run in cli],
            "cli_peak_rss_bytes": {
                "samples": rss_values,
                "median": statistics.median(rss_values),
                "sample_stdev": statistics.stdev(rss_values)
                if len(rss_values) > 1
                else 0,
            },
            "http": {
                "short_before": short_before["hash"],
                "long": long["hash"],
                "short_after": short_after["hash"],
                "short_reset_proven": True,
            },
            "cpu_sample": sample_receipt,
            "memory_scope": "CLI fresh-process RSS only; no remote server RSS claim",
        }
    except (
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
        urllib.error.URLError,
    ) as error:
        receipt = {
            "status": "failed",
            "kv_budget_mib": args.kv_budget_mib,
            "error": f"{type(error).__name__}: {error}",
        }
        (args.output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
        print(json.dumps(receipt, indent=2), file=sys.stderr)
        return 1
    (args.output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
