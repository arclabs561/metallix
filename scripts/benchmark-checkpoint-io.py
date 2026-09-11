#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Measure bounded aligned random reads from one checkpoint artifact."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import platform
import random
import stat
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TextIO

READ_SIZES = (4 * 1024, 32 * 1024, 256 * 1024, 1024 * 1024)
DARWIN_F_NOCACHE = 48


@dataclass(frozen=True)
class FileMetadata:
    dev: int
    ino: int
    size: int
    mtime_ns: int


def metadata_record(metadata: FileMetadata) -> dict[str, int]:
    return {
        "dev": metadata.dev,
        "ino": metadata.ino,
        "size": metadata.size,
        "mtime_ns": metadata.mtime_ns,
    }


def file_metadata(path: Path) -> FileMetadata:
    stat = path.stat()
    return FileMetadata(stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)


def descriptor_metadata(fd: int) -> FileMetadata:
    descriptor_stat = os.fstat(fd)
    if not stat.S_ISREG(descriptor_stat.st_mode):
        raise ValueError("checkpoint artifact must be a regular file")
    return FileMetadata(
        descriptor_stat.st_dev,
        descriptor_stat.st_ino,
        descriptor_stat.st_size,
        descriptor_stat.st_mtime_ns,
    )


def remaining_seconds(deadline: float) -> float:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("checkpoint I/O probe exceeded its total deadline")
    return remaining


def percentile_95(values: list[float]) -> float:
    ordered = sorted(values)
    return ordered[math.ceil(len(ordered) * 0.95) - 1]


def enable_uncached(fd: int) -> None:
    if (
        sys.platform != "darwin"
        or getattr(fcntl, "F_NOCACHE", None) != DARWIN_F_NOCACHE
    ):
        raise ValueError("--uncached requires Darwin fcntl.F_NOCACHE verified as 48")
    fcntl.fcntl(fd, DARWIN_F_NOCACHE, 1)


def sha256_file(path: Path, deadline: float) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            remaining_seconds(deadline)
            digest.update(chunk)
    return digest.hexdigest()


def checkout_state(root: Path, deadline: float) -> dict[str, object]:
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=min(1, remaining_seconds(deadline)),
        ).strip()
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=root,
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=min(1, remaining_seconds(deadline)),
        )
    except (
        FileNotFoundError,
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
    ):
        return {"available": False}
    return {"available": True, "revision": revision, "dirty": bool(dirty)}


def read_offsets(
    file_size: int, read_size: int, reads: int, randomizer: random.Random
) -> list[int]:
    slots = (file_size - read_size) // read_size + 1
    if slots < 1:
        raise ValueError(
            f"checkpoint is smaller than required {read_size}-byte read size"
        )
    return [randomizer.randrange(slots) * read_size for _ in range(reads)]


def summarize_size(
    read_size: int, samples: list[dict[str, object]]
) -> dict[str, object]:
    latencies = [sample["latency_ms"] for sample in samples]
    byte_counts = [sample["bytes"] for sample in samples]
    if not all(isinstance(value, float) for value in latencies):
        raise ValueError("internal latency sample type error")
    if not all(type(value) is int for value in byte_counts):
        raise ValueError("internal byte sample type error")
    total_ms = sum(latencies)
    total_bytes = sum(byte_counts)
    return {
        "read_size_bytes": read_size,
        "sample_count": len(samples),
        "total_bytes": total_bytes,
        "total_ms": total_ms,
        "median_ms": statistics.median(latencies),
        "p95_ms": percentile_95(latencies),
        "throughput_bytes_per_second": total_bytes / (total_ms / 1000),
    }


def run_probe(
    path: Path, reads: int, seed: int, uncached: bool, deadline: float
) -> dict[str, object]:
    if not path.is_file():
        raise ValueError("checkpoint artifact must be an existing regular file")
    before = file_metadata(path)
    if before.size < max(READ_SIZES):
        raise ValueError(
            f"checkpoint must be at least {max(READ_SIZES)} bytes for the read-size curve"
        )
    randomizer = random.Random(seed)
    samples_by_size: dict[int, list[dict[str, object]]] = {
        size: [] for size in READ_SIZES
    }
    fd = os.open(path, os.O_RDONLY)
    opened: FileMetadata | None = None
    after: FileMetadata | None = None
    try:
        opened = descriptor_metadata(fd)
        if opened != before:
            raise ValueError("checkpoint path changed while opening")
        if uncached:
            enable_uncached(fd)
        for read_size in READ_SIZES:
            for offset in read_offsets(before.size, read_size, reads, randomizer):
                remaining_seconds(deadline)
                started = time.perf_counter_ns()
                data = os.pread(fd, read_size, offset)
                latency_ms = (time.perf_counter_ns() - started) / 1_000_000
                remaining_seconds(deadline)
                if len(data) != read_size:
                    raise ValueError(
                        f"short pread at offset {offset}: expected {read_size}, got {len(data)}"
                    )
                samples_by_size[read_size].append(
                    {
                        "offset": offset,
                        "read_size_bytes": read_size,
                        "bytes": len(data),
                        "latency_ms": latency_ms,
                    }
                )
    finally:
        os.close(fd)
        after = file_metadata(path)
        if opened is None or before != after or before != opened:
            raise ValueError("checkpoint metadata changed during the read-only probe")
    return {
        "file_metadata": {
            "path_before": metadata_record(before),
            "opened_descriptor": metadata_record(opened),
            "path_after": metadata_record(after),
            "path_metadata_unchanged": True,
        },
        "samples": [sample for size in READ_SIZES for sample in samples_by_size[size]],
        "curve": [summarize_size(size, samples_by_size[size]) for size in READ_SIZES],
    }


def write_record(output: TextIO, record: dict[str, object]) -> None:
    output.seek(0)
    output.write(json.dumps(record, indent=2, allow_nan=False) + "\n")
    output.truncate()
    output.flush()


def public_error(error: OSError | TimeoutError | ValueError) -> str:
    if isinstance(error, OSError):
        return f"{type(error).__name__}: checkpoint I/O operation failed"
    return f"{type(error).__name__}: {error}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--file", type=Path, required=True, help="existing safetensors artifact"
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="new JSON receipt; never overwritten"
    )
    parser.add_argument(
        "--reads", type=int, default=64, help="random reads per size (1..4096)"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--uncached",
        action="store_true",
        help="request Darwin F_NOCACHE (a hint, not SSD proof)",
    )
    parser.add_argument("--timeout-seconds", type=float, default=30)
    args = parser.parse_args()
    if not 1 <= args.reads <= 4096:
        parser.error("--reads must be between 1 and 4096")
    if not math.isfinite(args.timeout_seconds) or args.timeout_seconds <= 0:
        parser.error("--timeout-seconds must be finite and positive")
    record: dict[str, object] = {
        "schema_version": 1,
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "scope": "aligned random pread latency curve for one checkpoint artifact; not model inference or a weight load; deadline is cooperative around blocking pread calls",
        "environment": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "probe_sha256": sha256_file(Path(__file__).resolve(), time.monotonic() + 1),
        },
        "workload": {
            "artifact_basename": args.file.name,
            "reads_per_size": args.reads,
            "seed": args.seed,
            "read_sizes_bytes": list(READ_SIZES),
            "cache_mode": (
                "Darwin F_NOCACHE requested; this hint does not prove physical SSD traffic"
                if args.uncached
                else "cached mode; OS cache state is uncontrolled"
            ),
        },
    }
    try:
        with args.output.open("x", encoding="utf-8") as output:
            write_record(output, record)
            try:
                deadline = time.monotonic() + args.timeout_seconds
                record["checkout"] = checkout_state(
                    Path(__file__).resolve().parent.parent, deadline
                )
                record["result"] = run_probe(
                    args.file.resolve(strict=True),
                    args.reads,
                    args.seed,
                    args.uncached,
                    deadline,
                )
                record["status"] = "completed"
            except (OSError, TimeoutError, ValueError) as error:
                record.update(status="failed", error=public_error(error))
            except KeyboardInterrupt:
                record.update(status="interrupted", error="benchmark interrupted")
            write_record(output, record)
    except OSError as error:
        print(f"Cannot create checkpoint I/O receipt: {error}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {"status": record["status"], "output": str(args.output)}, allow_nan=False
        )
    )
    return 0 if record["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
