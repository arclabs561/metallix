#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Measure repeated fixed-length Qwen Metal decode; not a correctness oracle."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TextIO


@dataclass(frozen=True)
class Run:
    generated_ids: tuple[int, ...]
    decode_ms: tuple[float, ...]
    load_ms: float
    prefill_ms: float
    backend: str
    cached_tokens: int
    logical_kv_bytes: int
    logical_weight_bytes: int


def milliseconds(value: object, *, positive: bool = False) -> float:
    if type(value) not in (int, float):
        raise ValueError("timings must be numbers, not booleans")
    try:
        result = float(value)
    except OverflowError as error:
        raise ValueError("timing is outside the finite float range") from error
    if not math.isfinite(result) or result < 0 or (positive and result == 0):
        raise ValueError(
            "timings must be finite and nonnegative; decode must be positive"
        )
    return result


def token_ids(value: object) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError("token IDs must be a nonempty array")
    if any(type(token) is not int or not 0 <= token <= 2**31 - 1 for token in value):
        raise ValueError("token IDs must be nonnegative int32 values")
    return tuple(value)


def parse_run(payload: object, input_ids: list[int], max_tokens: int) -> Run:
    if not isinstance(payload, dict):
        # Malformed external JSON is one parse-error category, including shape.
        raise ValueError("generation JSON must be an object")  # noqa: TRY004
    if type(payload.get("schema_version")) is not int or payload["schema_version"] != 1:
        raise ValueError("unsupported generation schema_version")
    if payload.get("operation") != "qwen3_greedy_cached_generation":
        raise ValueError("unexpected generation operation")
    if token_ids(payload.get("input_ids")) != tuple(input_ids):
        raise ValueError("generation input IDs do not match the requested workload")
    generated = token_ids(payload.get("generated_ids"))
    if len(generated) != max_tokens:
        raise ValueError(
            "generation stopped early; fixed-length benchmark is incomparable"
        )
    if payload.get("finish_reason") not in ("length", "eos"):
        raise ValueError("generation did not finish normally")
    if payload.get("cache_comparisons") != []:
        raise ValueError("cache verification must be disabled during timing")
    decode = payload.get("decode_ms")
    if not isinstance(decode, list) or len(decode) != len(generated) - 1:
        raise ValueError("decode sample count must equal generated tokens minus one")
    backend = payload.get("backend")
    if not isinstance(backend, str) or not backend.strip():
        raise ValueError("generation backend is missing")
    for key in ("cached_tokens", "logical_kv_bytes", "logical_weight_bytes"):
        if type(payload.get(key)) is not int or payload[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    if payload["cached_tokens"] != len(input_ids) + len(generated) - 1:
        raise ValueError("cached token count does not match the generation workload")
    return Run(
        generated_ids=generated,
        decode_ms=tuple(milliseconds(value, positive=True) for value in decode),
        load_ms=milliseconds(payload.get("load_ms")),
        prefill_ms=milliseconds(payload.get("prefill_ms")),
        backend=backend,
        cached_tokens=payload["cached_tokens"],
        logical_kv_bytes=payload["logical_kv_bytes"],
        logical_weight_bytes=payload["logical_weight_bytes"],
    )


def summarize_runs(runs: list[Run], discard_decode: int) -> dict[str, object]:
    if len(runs) < 3:
        raise ValueError("at least three runs are required")
    if type(discard_decode) is not int or discard_decode < 0:
        raise ValueError("discard_decode must be a nonnegative integer")
    if any(run.generated_ids != runs[0].generated_ids for run in runs):
        raise ValueError("generated IDs differ between runs")
    if any(run.backend != runs[0].backend for run in runs):
        raise ValueError("backend differs between runs")
    windows = [run.decode_ms[discard_decode:] for run in runs]
    if any(len(window) < 2 for window in windows):
        raise ValueError("each run must retain at least two decode observations")
    samples = [sample for window in windows for sample in window]
    return {
        "sample_count": len(samples),
        "median_ms": statistics.median(samples),
        "mean_ms": statistics.mean(samples),
        "sample_stdev_ms": statistics.stdev(samples),
        "per_run_median_ms": [statistics.median(window) for window in windows],
        "generated_ids": runs[0].generated_ids,
        "backend": runs[0].backend,
    }


def remaining_seconds(deadline: float) -> float:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("benchmark exceeded its total deadline")
    return remaining


def sha256_file(path: Path, deadline: float) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            remaining_seconds(deadline)
            digest.update(chunk)
    return digest.hexdigest()


def checkout_state(root: Path, deadline: float) -> dict[str, object]:
    # This observes the checkout, not the revision that built an arbitrary binary.
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=remaining_seconds(deadline),
        ).strip()
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=root,
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=remaining_seconds(deadline),
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return {"available": False}
    return {"available": True, "revision": revision, "dirty": bool(dirty)}


def write_record(output: TextIO, record: dict[str, object]) -> None:
    # The file is exclusively created by this process. Preserve a readable
    # running/failed receipt even if a model subprocess cannot produce a result.
    encoded = json.dumps(record, indent=2, allow_nan=False) + "\n"
    output.seek(0)
    output.write(encoded)
    output.truncate()
    output.flush()


def execute(
    args: argparse.Namespace, record: dict[str, object], output: TextIO
) -> None:
    started = time.monotonic()
    deadline = started + args.timeout_seconds
    binary = args.binary.resolve(strict=True)
    model = args.model.resolve(strict=True)
    paths = {
        "binary": binary,
        "config": model / "config.json",
        "weights": model / "model.safetensors",
    }
    hashes = {name: sha256_file(path, deadline) for name, path in paths.items()}
    record["sha256"] = hashes
    record["checkout"] = checkout_state(
        Path(__file__).resolve().parent.parent, deadline
    )
    if platform.system() == "Darwin":
        hardware = subprocess.check_output(
            ["/usr/sbin/sysctl", "-n", "machdep.cpu.brand_string", "hw.memsize"],
            text=True,
            timeout=remaining_seconds(deadline),
        ).splitlines()
        if len(hardware) != 2:
            raise ValueError("unexpected macOS hardware metadata")
        record["apple_hardware"] = {
            "cpu": hardware[0],
            "memory_bytes": int(hardware[1]),
        }
    write_record(output, record)
    runs: list[Run] = []
    for index in range(args.runs):
        print(
            f"Qwen benchmark run {index + 1}/{args.runs}", file=sys.stderr, flush=True
        )
        completed = subprocess.run(
            [
                str(binary),
                "generate-qwen-metal",
                "--model",
                str(model),
                "--input-ids",
                ",".join(map(str, args.input_ids)),
                "--max-tokens",
                str(args.max_tokens),
            ],
            capture_output=True,
            text=True,
            timeout=remaining_seconds(deadline),
            check=False,
        )
        if completed.returncode != 0:
            raise ValueError(
                f"generation run {index + 1} exited {completed.returncode}"
            )
        runs.append(
            parse_run(json.loads(completed.stdout), args.input_ids, args.max_tokens)
        )
        record["runs"] = [asdict(run) for run in runs]
        write_record(output, record)
    # Refuse results if the measured executable or checkpoint changed mid-run.
    if hashes != {name: sha256_file(path, deadline) for name, path in paths.items()}:
        raise ValueError("binary or checkpoint changed during measurement")
    record["warm_decode"] = summarize_runs(runs, args.discard_decode)
    record["elapsed_seconds"] = time.monotonic() - started
    record["status"] = "completed"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=Path("target/release/metallix"))
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path, required=True, help="new JSON receipt; never overwritten"
    )
    parser.add_argument("--input-ids", default="785,6722,315,9625,374")
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--discard-decode", type=int, default=1)
    parser.add_argument("--timeout-seconds", type=float, default=300)
    args = parser.parse_args()
    try:
        args.input_ids = list(
            token_ids([int(token) for token in args.input_ids.split(",")])
        )
    except ValueError as error:
        parser.error(str(error))
    if not 3 <= args.runs <= 100:
        parser.error("--runs must be between 3 and 100")
    if not 4 <= args.max_tokens <= 256 or len(args.input_ids) + args.max_tokens > 512:
        parser.error("--max-tokens must be 4..256 and prompt plus output at most 512")
    if not 0 <= args.discard_decode <= args.max_tokens - 3:
        parser.error("--discard-decode must leave at least two decode observations")
    if not math.isfinite(args.timeout_seconds) or args.timeout_seconds <= 0:
        parser.error("--timeout-seconds must be finite and positive")
    record: dict[str, object] = {
        "schema_version": 1,
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "host": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "workload": {
            "input_ids": args.input_ids,
            "max_tokens": args.max_tokens,
            "runs": args.runs,
            "discard_decode": args.discard_decode,
        },
        "scope": "single sequence; fresh process/KV per run; file hashes warm OS cache; excludes load/prefill and initial decode observations; not HTTP throughput or an independent correctness oracle; checkout revision is not binary build provenance",
        "runs": [],
    }
    try:
        with args.output.open("x", encoding="utf-8") as output:
            write_record(output, record)
            try:
                execute(args, record, output)
            except (OSError, ValueError, subprocess.SubprocessError) as error:
                record.update(
                    status="failed", error=type(error).__name__ + ": " + str(error)
                )
            except KeyboardInterrupt:
                record.update(status="interrupted", error="benchmark interrupted")
            write_record(output, record)
    except OSError as error:
        print(f"Cannot create benchmark receipt: {error}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "status": record["status"],
                "output": str(args.output),
                "warm_decode": record.get("warm_decode"),
            },
            allow_nan=False,
        )
    )
    return 0 if record["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
