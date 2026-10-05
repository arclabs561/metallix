#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Measure `mx serve` latency for every model in a registry; not a correctness oracle.

Starts `mx serve --registry`, then for each registered model reports the time
to its first answer (which includes starting an on-demand child), the steady
latency of each route it serves, and the front process's overhead measured as
interleaved pairs of the same request sent through the front and directly to
the model's child. The machine's load average is recorded before and after,
because numbers from a shared machine are hard to read without it.

Usage: uv run scripts/bench_serve.py --registry models.json [--mx target/release/mx]
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import socket
import statistics
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

FRONT_LISTENING = re.compile(r"^mx listening on http://([0-9.]+:\d+);")
CHILD_LISTENING = re.compile(
    r"^\[(?P<id>[^\]]+)\] mx listening on http://(?P<address>[0-9.]+:\d+);"
)
DECISION = {
    "state": {"hp": 3},
    "questions": {
        "action": {
            "type": "choice",
            "instructions": "Pick the next action.",
            "criteria": {"heal": "drink a potion", "attack": "swing the sword"},
        }
    },
}
# Roughly 8, 64 and 256 tokens for common tokenizers.
EMBEDDING_WORDS = (6, 48, 192)
FILLER = ["the", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog"]


def child_address(line: str) -> tuple[str, str] | None:
    """The `(model id, host:port)` a child announces in a front-process log line."""
    match = CHILD_LISTENING.match(line)
    return (match["id"], match["address"]) if match else None


def front_address(line: str) -> str | None:
    match = FRONT_LISTENING.match(line)
    return match[1] if match else None


def summary(milliseconds: list[float]) -> dict[str, float]:
    ordered = sorted(milliseconds)
    return {
        "median_ms": statistics.median(ordered),
        "min_ms": ordered[0],
        "max_ms": ordered[-1],
        "samples": len(ordered),
    }


def words(count: int) -> str:
    return " ".join(FILLER[i % len(FILLER)] for i in range(count))


def requests_for(model: str, capabilities: list[str]) -> list[tuple[str, str, dict]]:
    """`(label, path, body)` requests covering every route the model serves."""
    out = []
    if "generate" in capabilities:
        out.append(
            (
                "generate 16 tokens",
                "/v1/responses",
                {"model": model, "input": "Say hello.", "max_output_tokens": 16},
            )
        )
    if "decide" in capabilities:
        out.append(("decide 1 question", "/v1/decisions", {"model": model, **DECISION}))
    if "embed" in capabilities:
        for count in EMBEDDING_WORDS:
            out.append(
                (
                    f"embed {count} words",
                    "/v1/embeddings",
                    {"model": model, "input": words(count)},
                )
            )
    return out


def post(address: str, path: str, body: dict) -> tuple[float, dict]:
    request = urllib.request.Request(
        f"http://{address}{path}", data=json.dumps(body).encode(), method="POST"
    )
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=300) as response:
        payload = json.loads(response.read())
    return (time.perf_counter() - started) * 1000, payload


def get(address: str, path: str) -> dict:
    with urllib.request.urlopen(f"http://{address}{path}", timeout=30) as response:
        return json.loads(response.read())


def free_address() -> str:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return f"127.0.0.1:{probe.getsockname()[1]}"


def repo_revision() -> str:
    try:
        return subprocess.run(
            [
                "git",
                "-C",
                str(Path(__file__).resolve().parent),
                "rev-parse",
                "--short",
                "HEAD",
            ],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


class Server:
    """`mx serve` with its stderr read on a thread for announced addresses."""

    def __init__(self, mx: Path, registry: Path, extra: list[str]):
        self.address = free_address()
        self.children: dict[str, str] = {}
        self.listening = threading.Event()
        self.started = time.perf_counter()
        self.ready_ms = None
        self.process = subprocess.Popen(
            [
                str(mx),
                "serve",
                "--registry",
                str(registry),
                "--listen",
                self.address,
                *extra,
            ],
            stderr=subprocess.PIPE,
            text=True,
        )
        threading.Thread(target=self._read_log, daemon=True).start()

    def _read_log(self) -> None:
        stderr = self.process.stderr
        assert stderr is not None, "started with stderr=PIPE"
        for line in stderr:
            if child := child_address(line):
                self.children[child[0]] = child[1]
            if front_address(line):
                self.ready_ms = (time.perf_counter() - self.started) * 1000
                self.listening.set()

    def wait(self, seconds: float) -> None:
        if not self.listening.wait(seconds):
            self.stop()
            raise SystemExit(f"mx serve did not listen within {seconds:.0f} s")

    def stop(self) -> None:
        self.process.terminate()
        self.process.wait()


def measure(server: Server, samples: int) -> list[dict]:
    results = []
    for model in get(server.address, "/v1/models")["data"]:
        cases = requests_for(model["id"], model["capabilities"])
        if not cases:
            continue
        _, path, body = cases[0]
        first_ms, _ = post(server.address, path, body)
        record = {
            "model": model["id"],
            "capabilities": model["capabilities"],
            "first_request_ms": first_ms,
            "routes": {},
        }
        for label, path, body in cases:
            record["routes"][label] = summary(
                [post(server.address, path, body)[0] for _ in range(samples)]
            )
        child = server.children.get(model["id"])
        if child:
            label, path, body = cases[0]
            # Interleave so drift on a shared machine affects both paths alike.
            pairs = [
                (post(server.address, path, body)[0], post(child, path, body)[0])
                for _ in range(samples)
            ]
            record["proxy"] = {
                "route": label,
                "front": summary([front for front, _ in pairs]),
                "direct": summary([direct for _, direct in pairs]),
                "median_paired_overhead_ms": statistics.median(f - d for f, d in pairs),
            }
        results.append(record)
    return results


def render(report: dict) -> str:
    lines = [
        f"mx {report['mx']}; repo at {report['revision']}; {report['machine']}",
        f"load average before {report['load_before']}, after {report['load_after']}",
        f"front listening after {report['ready_ms']:.0f} ms",
    ]
    for model in report["models"]:
        lines.append(
            f"{model['model']} ({', '.join(model['capabilities'])}): "
            f"first request {model['first_request_ms']:.0f} ms"
        )
        for label, stats in model["routes"].items():
            lines.append(
                f"  {label}: median {stats['median_ms']:.2f} ms "
                f"(min {stats['min_ms']:.2f}, max {stats['max_ms']:.2f}, n={stats['samples']})"
            )
        if proxy := model.get("proxy"):
            lines.append(
                f"  proxy on {proxy['route']}: front {proxy['front']['median_ms']:.2f} ms, "
                f"direct {proxy['direct']['median_ms']:.2f} ms, "
                f"median paired overhead {proxy['median_paired_overhead_ms']:.2f} ms"
            )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--mx", type=Path, default=Path("target/release/mx"))
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--start-timeout", type=float, default=300)
    parser.add_argument("--json", type=Path, help="also write the report as JSON")
    parser.add_argument(
        "serve_args", nargs="*", help="extra `mx serve` arguments, after --"
    )
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("--samples must be positive")

    load_before = [round(value, 2) for value in os.getloadavg()]
    server = Server(args.mx, args.registry, args.serve_args)
    try:
        server.wait(args.start_timeout)
        models = measure(server, args.samples)
    finally:
        server.stop()
    report = {
        "mx": str(args.mx),
        "revision": repo_revision(),
        "machine": f"{platform.machine()} {platform.platform()}",
        "load_before": load_before,
        "load_after": [round(value, 2) for value in os.getloadavg()],
        "ready_ms": server.ready_ms,
        "samples": args.samples,
        "models": models,
    }
    print(render(report))
    if args.json:
        args.json.write_text(json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
