#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = ["tokenizers>=0.20"]
# ///
"""Load-test OpenAI-compatible streaming servers; latency under load, not correctness.

Drives `/v1/chat/completions` or `/v1/responses` with streaming requests, either
at a fixed concurrency or with Poisson arrivals at a set of request rates, and
records per request the time to first token (TTFT), time per output token after
the first (TPOT), the gaps between streamed chunks, end-to-end latency, token
counts and the outcome (HTTP status or stream error). Each run reports
throughput, p50/p90/p99 and SLO attainment; a rate sweep also reports goodput
as defined by DistServe (Zhong et al., OSDI '24): the highest request rate at
which at least 90% of requests meet both the TTFT and the TPOT objective.

The harness can start and stop the server under test: `mx serve`, vllm-metal,
MTPLX or mlx-lm's server, the Python ones from the virtual environments in
--envs. Every level gets a fresh server, so no level inherits an earlier
level's prefix cache. Load averages are recorded around every run, because
numbers from a shared machine are hard to read without them.

Usage:
  uv run scripts/bench_load.py --server metallix --model-path DIR \\
      --sets shared-prefix,long --concurrency 1,2,4,8 --rates 0.5,1,2
  uv run scripts/bench_load.py --url http://127.0.0.1:8000 --api chat \\
      --model-id qwen3 --model-path DIR --concurrency 1
  uv run scripts/bench_load.py --server metallix --model-path DIR \\
      --sets file --prompts-file agent_benchmark_prompts.json --concurrency 1,8
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import http.client
import itertools
import json
import math
import os
import platform
import random
import re
import shlex
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import bench_serve
import bench_system
import mx_spans

ROOT = Path(__file__).resolve().parent.parent
ENVS = ROOT / ".agents" / "bench-envs"
PERCENTILES = (50, 90, 99)
# DistServe's attainment goal: goodput is the highest rate where at least this
# share of requests meets both latency objectives.
ATTAINMENT_GOAL = 0.9

# A fixed vocabulary keeps prompts deterministic for a seed while making every
# long prompt distinct, so a prefix cache cannot reuse one long prompt for another.
WORDS = (  # noqa: SIM905 (one string reads better than a long list)
    "river stone lantern harbor copper meadow signal orbit marble cedar "
    "falcon glacier ember quartz willow thunder canyon prism velvet anchor "
    "beacon compass delta fable garnet hollow island jasper kettle lagoon "
    "mosaic nectar oasis pepper quiver raven saddle timber umbra vessel "
    "wander yarrow zephyr amber basil cobalt dune echo fern grove heron "
    "ivory juniper kelp linen maple nimbus olive pebble quill reed sable "
    "thistle valley walnut builds measures carries follows gathers watches "
    "reaches settles turns under across beyond behind within toward before "
    "after during between"
).split()
TOOL_NAMES = (  # noqa: SIM905 (one string reads better than a long list)
    "read_file write_file list_directory search_code run_tests git_status "
    "git_diff open_issue fetch_url summarize_log query_metrics "
    "restart_service"
).split()


# ---------------------------------------------------------------------------
# Prompt sets


@dataclass(frozen=True)
class Prompt:
    """One request's text: an optional system part and the user turn.

    A prompt loaded from a file may instead carry a recorded conversation in
    OpenAI chat form (tool calls and results included), sent as is, and its own
    output cap; `system` and `user` are then unused.
    """

    label: str
    system: str | None
    user: str
    messages: tuple[dict, ...] | None = None
    max_tokens: int | None = None


def sentence(rng: random.Random, length: int) -> str:
    return " ".join(rng.choice(WORDS) for _ in range(length)).capitalize() + "."


def text_of_tokens(
    rng: random.Random, tokens: int, count_tokens, header: str = ""
) -> str:
    """Seeded filler text close to `tokens` tokens under `count_tokens`."""
    parts = [header] if header else []
    # Grow in sentence steps counted one at a time (a sum that slightly
    # overestimates the joined count), then trim words against the exact count.
    total = count_tokens(header) if header else 0
    while total < tokens + 16:
        part = sentence(rng, 12)
        parts.append(part)
        total += count_tokens(" " + part)
    text = " ".join(parts)
    words = text.split(" ")
    while len(words) > 1 and count_tokens(" ".join(words)) > tokens:
        words.pop()
    return " ".join(words)


def agent_preamble(tokens: int, count_tokens) -> str:
    """A long, fixed system prompt shaped like an agent's tool listing."""
    rng = random.Random(1234)
    lines = [
        "You are a coding agent working in a repository. Use the tools below.",
        "Answer briefly and cite the files you used.",
        "",
        "# Tools",
    ]
    index = 0
    while count_tokens("\n".join(lines)) < tokens:
        name = f"{TOOL_NAMES[index % len(TOOL_NAMES)]}_{index // len(TOOL_NAMES)}"
        schema = {
            "name": name,
            "description": sentence(rng, 18),
            "parameters": {
                "type": "object",
                "properties": {
                    word: {"type": "string", "description": sentence(rng, 8)}
                    for word in rng.sample(WORDS, 3)
                },
            },
        }
        lines.append(json.dumps(schema))
        index += 1
    return "\n".join(lines)


CONTINUE = "Continue the passage above as a long, detailed story."


def build_prompts(name: str, count: int, count_tokens, seed: int) -> list[Prompt]:
    """`count` prompts for one set; deterministic for a seed."""
    rng = random.Random(f"{name}:{seed}")
    if name == "short":
        return [
            Prompt(
                "short",
                None,
                f"{sentence(rng, rng.randint(12, 40))} Tell a long story about it.",
            )
            for _ in range(count)
        ]
    if name == "long":
        # 2k to 8k tokens, unique per request (the leading tag defeats prefix reuse).
        sizes = (2048, 4096, 6144, 8000)
        return [
            Prompt(
                f"long-{sizes[i % len(sizes)]}",
                None,
                text_of_tokens(
                    rng,
                    sizes[i % len(sizes)],
                    count_tokens,
                    header=f"Document {seed}-{i}.",
                )
                + " "
                + CONTINUE,
            )
            for i in range(count)
        ]
    if name == "shared-prefix":
        preamble = agent_preamble(2048, count_tokens)
        return [
            Prompt(
                "shared-prefix",
                preamble,
                f"Task {i}: {sentence(rng, rng.randint(20, 60))} "
                "Explain step by step what you would do.",
            )
            for i in range(count)
        ]
    if name == "mixed":
        # 60% short, 20% shared-prefix, 20% long, interleaved deterministically.
        pools = {
            kind: build_prompts(kind, count, count_tokens, seed)
            for kind in ("short", "shared-prefix", "long")
        }
        pattern = ("short", "shared-prefix", "short", "long", "short")
        return [pools[pattern[i % len(pattern)]][i] for i in range(count)]
    raise ValueError(f"unknown prompt set {name!r}")


# The set name for prompts read with --prompts-file.
FILE_SET = "file"
PROMPT_SETS = ("short", "long", "shared-prefix", "mixed", FILE_SET)


def load_prompts_file(path: Path) -> list[Prompt]:
    """Prompts from a JSON array or JSON Lines file, in file order.

    Each entry has either `messages` (OpenAI chat messages, as in SiliconBench's
    prompts/agent_benchmark_prompts.json) or `user` with an optional `system`,
    plus optional `name` and `max_tokens`.
    """
    text = path.read_text()
    stripped = text.lstrip()
    entries = (
        json.loads(text)
        if stripped.startswith("[")
        else [json.loads(line) for line in text.splitlines() if line.strip()]
    )
    prompts = []
    for index, entry in enumerate(entries):
        label = str(entry.get("name") or entry.get("label") or f"file-{index}")
        max_tokens = entry.get("max_tokens")
        if max_tokens is not None and (
            not isinstance(max_tokens, int) or max_tokens < 1
        ):
            raise ValueError(f"{path}: entry {index} has a bad max_tokens")
        if isinstance(entry.get("messages"), list) and entry["messages"]:
            prompts.append(
                Prompt(label, None, "", tuple(entry["messages"]), max_tokens)
            )
        elif isinstance(entry.get("user"), str):
            prompts.append(
                Prompt(label, entry.get("system"), entry["user"], None, max_tokens)
            )
        else:
            raise ValueError(f"{path}: entry {index} has neither messages nor user")
    if not prompts:
        raise ValueError(f"{path}: no prompts")
    return prompts


def file_identity(path: Path | None) -> dict | None:
    """Name and SHA-256 of a prompts file, so a report pins the exact prompts."""
    if path is None:
        return None
    return {"path": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def cycle(prompts: list[Prompt], count: int) -> list[Prompt]:
    """The first `count` prompts, wrapping around when the file is shorter."""
    return [prompts[i % len(prompts)] for i in range(count)]


def set_prompts(
    name: str, count: int, count_tokens, seed: int, file_prompts: list[Prompt]
) -> list[Prompt]:
    if name == FILE_SET:
        return cycle(file_prompts, count)
    return build_prompts(name, count, count_tokens, seed)


def warmup_prompts(
    name: str, count: int, count_tokens, seed: int, file_prompts: list[Prompt]
) -> list[Prompt]:
    """Discarded requests before a measurement, distinct from the measured ones
    where the set allows: a fresh seed, or the tail of a prompts file."""
    if name == FILE_SET:
        return cycle(file_prompts[::-1], count)
    return build_prompts(name, count, count_tokens, seed + 999)


def tokenizer_counter(model_path: Path):
    """Token counter for the model's tokenizer.json (no special tokens)."""
    from tokenizers import Tokenizer  # Imported here so the tests need no deps.

    tokenizer = Tokenizer.from_file(str(model_path / "tokenizer.json"))
    return lambda text: len(tokenizer.encode(text, add_special_tokens=False).ids)


# ---------------------------------------------------------------------------
# Requests and streams


def request_body(
    api: str, model: str, prompt: Prompt, max_tokens: int, extra: dict
) -> tuple[str, dict]:
    """`(path, body)` for one streaming request."""
    max_tokens = prompt.max_tokens or max_tokens
    if api == "chat":
        if prompt.messages is not None:
            messages = list(prompt.messages)
        else:
            messages = (
                [{"role": "system", "content": prompt.system}] if prompt.system else []
            )
            messages.append({"role": "user", "content": prompt.user})
        body = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        return "/v1/chat/completions", body | extra
    if api == "responses":
        body = {
            "model": model,
            "input": (
                prompt.user
                if prompt.messages is None
                else responses_input(prompt.messages)
            ),
            "max_output_tokens": max_tokens,
            "temperature": 0,
            "stream": True,
        }
        if prompt.system:
            body["instructions"] = prompt.system
        return "/v1/responses", body | extra
    raise ValueError(f"unknown api {api!r}")


def responses_input(messages: tuple[dict, ...]) -> list[dict]:
    """Chat messages as Responses input items: an assistant turn's tool calls
    become `function_call` items and tool results `function_call_output` items."""
    items: list[dict] = []
    for message in messages:
        role, content = message["role"], message.get("content") or ""
        if role == "tool":
            items.append(
                {
                    "type": "function_call_output",
                    "call_id": message["tool_call_id"],
                    "output": content,
                }
            )
            continue
        if content or not message.get("tool_calls"):
            items.append({"type": "message", "role": role, "content": content})
        for call in message.get("tool_calls") or []:
            items.append(
                {
                    "type": "function_call",
                    "call_id": call["id"],
                    "name": call["function"]["name"],
                    "arguments": arguments
                    if isinstance(arguments := call["function"]["arguments"], str)
                    else json.dumps(arguments),
                }
            )
    return items


@dataclass
class StreamState:
    """What one SSE stream has said so far, with the arrival time of each token chunk.

    `feed` takes one line of the event stream and the time it arrived. A chunk
    counts as a token arrival when it carries generated text (content or
    reasoning), for either API.
    """

    api: str
    chunk_times: list[float] = field(default_factory=list)
    input_tokens: int | None = None
    output_tokens: int | None = None
    done: bool = False
    error: str | None = None

    def feed(self, line: str, now: float) -> None:
        line = line.strip()
        if not line.startswith("data:"):
            return
        data = line[5:].strip()
        if data == "[DONE]":
            self.done = True
            return
        try:
            event = json.loads(data)
        except json.JSONDecodeError:
            self.error = f"undecodable event: {data[:80]}"
            return
        if self.api == "chat":
            self._chat(event, now)
        else:
            self._responses(event, now)

    def _usage(self, usage: dict | None, prompt_key: str, output_key: str) -> None:
        if usage:
            self.input_tokens = usage.get(prompt_key, self.input_tokens)
            self.output_tokens = usage.get(output_key, self.output_tokens)

    def _chat(self, event: dict, now: float) -> None:
        if "error" in event:
            self.error = json.dumps(event["error"])[:200]
            return
        for choice in event.get("choices") or []:
            delta = choice.get("delta") or {}
            if any(
                delta.get(key) for key in ("content", "reasoning_content", "reasoning")
            ):
                self.chunk_times.append(now)
                break
        self._usage(event.get("usage"), "prompt_tokens", "completion_tokens")

    def _responses(self, event: dict, now: float) -> None:
        kind = event.get("type", "")
        if kind in (
            "response.output_text.delta",
            "response.reasoning_text.delta",
        ) and event.get("delta"):
            self.chunk_times.append(now)
        elif kind in ("response.completed", "response.incomplete"):
            self.done = True
            self._usage(
                (event.get("response") or {}).get("usage"),
                "input_tokens",
                "output_tokens",
            )
        elif kind in ("response.failed", "error"):
            failure = (event.get("response") or {}).get("error") or event
            self.error = json.dumps(failure)[:200]


@dataclass
class Record:
    """One request's outcome and timings, in milliseconds from its send time."""

    index: int
    label: str
    outcome: str  # "ok", "http_<status>", or "error"
    send_s: float  # Seconds after the run started.
    ttft_ms: float | None = None
    tpot_ms: float | None = None
    e2e_ms: float | None = None
    itl_ms: list[float] = field(default_factory=list)
    chunks: int = 0
    input_tokens: int | None = None
    output_tokens: int | None = None
    detail: str | None = None


def timings(
    start: float, chunk_times: list[float], end: float, output_tokens: int | None
) -> dict:
    """TTFT, TPOT and inter-chunk gaps (ms) from send, chunk and end times (s).

    TPOT follows the usual serving-benchmark definition: decode time after the
    first token divided by the remaining tokens, (e2e - ttft) / (tokens - 1).
    When the server reports no token count, the chunk count stands in, which
    is exact only for servers that stream one token per chunk.
    """
    if not chunk_times:
        return {
            "ttft_ms": None,
            "tpot_ms": None,
            "e2e_ms": (end - start) * 1000,
            "itl_ms": [],
        }
    ttft = chunk_times[0] - start
    e2e = end - start
    tokens = output_tokens if output_tokens is not None else len(chunk_times)
    tpot = (e2e - ttft) / (tokens - 1) if tokens > 1 else None
    gaps = [(b - a) * 1000 for a, b in itertools.pairwise(chunk_times)]
    return {
        "ttft_ms": ttft * 1000,
        "tpot_ms": None if tpot is None else tpot * 1000,
        "e2e_ms": e2e * 1000,
        "itl_ms": gaps,
    }


def send(
    address: str,
    path: str,
    body: dict,
    api: str,
    index: int,
    label: str,
    run_start: float,
    timeout: float,
) -> Record:
    host, port = address.rsplit(":", 1)
    connection = http.client.HTTPConnection(host, int(port), timeout=timeout)
    start = time.perf_counter()
    record = Record(index, label, "error", start - run_start)
    state = StreamState(api)
    try:
        connection.request(
            "POST", path, json.dumps(body), {"Content-Type": "application/json"}
        )
        response = connection.getresponse()
        if response.status != 200:
            record.outcome = f"http_{response.status}"
            record.detail = response.read(400).decode(errors="replace")
            record.e2e_ms = (time.perf_counter() - start) * 1000
            return record
        while True:
            line = response.readline()
            if not line:
                break
            state.feed(line.decode(errors="replace"), time.perf_counter())
            if state.done or state.error:
                break
        end = time.perf_counter()
    except (OSError, http.client.HTTPException) as error:
        record.detail = f"{type(error).__name__}: {error}"
        record.e2e_ms = (time.perf_counter() - start) * 1000
        return record
    finally:
        connection.close()
    record.__dict__.update(timings(start, state.chunk_times, end, state.output_tokens))
    record.chunks = len(state.chunk_times)
    record.input_tokens = state.input_tokens
    record.output_tokens = (
        state.output_tokens if state.output_tokens is not None else record.chunks
    )
    if state.error:
        record.detail = state.error
    elif not state.chunk_times:
        record.detail = "stream ended without tokens"
    elif not state.done:
        record.detail = "stream ended without a completion event"
    else:
        record.outcome = "ok"
    return record


def poisson_arrivals(rate: float, count: int, rng: random.Random) -> list[float]:
    """Send offsets (s) for `count` requests with exponential gaps at `rate` per s."""
    offsets, now = [], 0.0
    for _ in range(count):
        offsets.append(now)
        now += rng.expovariate(rate)
    return offsets


def run_load(
    address: str,
    api: str,
    model: str,
    prompts: list[Prompt],
    max_tokens: int,
    extra: dict,
    *,
    concurrency: int | None = None,
    rate: float | None = None,
    seed: int = 0,
    timeout: float = 600,
) -> tuple[list[Record], float]:
    """Send every prompt once, at a fixed concurrency or with Poisson arrivals."""
    records: list[Record] = []
    lock = threading.Lock()
    run_start = time.perf_counter()

    def one(index: int) -> None:
        prompt = prompts[index]
        path, body = request_body(api, model, prompt, max_tokens, extra)
        record = send(address, path, body, api, index, prompt.label, run_start, timeout)
        with lock:
            records.append(record)

    if concurrency is not None:
        queue = iter(range(len(prompts)))

        def worker() -> None:
            while True:
                with lock:
                    index = next(queue, None)
                if index is None:
                    return
                one(index)

        threads = [threading.Thread(target=worker) for _ in range(concurrency)]
    else:
        assert rate is not None
        offsets = poisson_arrivals(rate, len(prompts), random.Random(seed))

        def delayed(index: int) -> None:
            delay = run_start + offsets[index] - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            one(index)

        threads = [
            threading.Thread(target=delayed, args=(i,)) for i in range(len(prompts))
        ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    duration = time.perf_counter() - run_start
    return sorted(records, key=lambda record: record.index), duration


# ---------------------------------------------------------------------------
# Metrics


def percentile(values: list[float], q: float) -> float | None:
    """Linear-interpolated percentile (numpy's default method)."""
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * q / 100
    low = math.floor(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def distribution(values: list[float]) -> dict:
    out = {f"p{q}": percentile(values, q) for q in PERCENTILES}
    out["mean"] = sum(values) / len(values) if values else None
    out["n"] = len(values)
    return out


def meets_slo(record: Record, ttft_slo_ms: float, tpot_slo_ms: float) -> bool:
    """A failed request misses; a one-token answer has no TPOT and meets that part."""
    if record.outcome != "ok" or record.ttft_ms is None:
        return False
    if record.ttft_ms > ttft_slo_ms:
        return False
    return record.tpot_ms is None or record.tpot_ms <= tpot_slo_ms


def summarize(
    records: list[Record], duration_s: float, ttft_slo_ms: float, tpot_slo_ms: float
) -> dict:
    ok = [record for record in records if record.outcome == "ok"]
    outcomes: dict[str, int] = {}
    for record in records:
        outcomes[record.outcome] = outcomes.get(record.outcome, 0) + 1
    meeting = sum(meets_slo(record, ttft_slo_ms, tpot_slo_ms) for record in records)
    output_tokens = sum(record.output_tokens or 0 for record in ok)
    input_tokens = sum(record.input_tokens or 0 for record in ok)
    return {
        "requests": len(records),
        "outcomes": outcomes,
        "duration_s": duration_s,
        "request_throughput": len(ok) / duration_s if duration_s else 0.0,
        "output_token_throughput": output_tokens / duration_s if duration_s else 0.0,
        "input_token_throughput": input_tokens / duration_s if duration_s else 0.0,
        "ttft_ms": distribution([r.ttft_ms for r in ok if r.ttft_ms is not None]),
        "tpot_ms": distribution([r.tpot_ms for r in ok if r.tpot_ms is not None]),
        "itl_ms": distribution([gap for r in ok for gap in r.itl_ms]),
        "e2e_ms": distribution([r.e2e_ms for r in ok if r.e2e_ms is not None]),
        "output_tokens": distribution([float(r.output_tokens or 0) for r in ok]),
        "input_tokens": distribution(
            [float(r.input_tokens) for r in ok if r.input_tokens is not None]
        ),
        "slo_attainment": meeting / len(records) if records else 0.0,
        # vLLM's "request goodput": SLO-meeting completions per second of this run.
        "slo_request_rate": meeting / duration_s if duration_s else 0.0,
    }


def goodput(attainment_by_rate: dict[float, float], goal: float = ATTAINMENT_GOAL):
    """DistServe goodput over a measured sweep: the highest rate meeting `goal`.

    Rates are scanned in increasing order and the scan stops at the first rate
    that misses, so a lucky pass above a failing rate (noise on a shared
    machine) does not count. None when even the lowest rate misses.
    """
    best = None
    for rate in sorted(attainment_by_rate):
        if attainment_by_rate[rate] < goal:
            break
        best = rate
    return best


# ---------------------------------------------------------------------------
# Servers


def env_python_versions(env: Path, distributions: list[str]) -> dict[str, str]:
    code = (
        "import importlib.metadata as m, json, sys\n"
        "out = {}\n"
        "for name in sys.argv[1:]:\n"
        "    try: out[name] = m.version(name)\n"
        "    except m.PackageNotFoundError: pass\n"
        "print(json.dumps(out))"
    )
    try:
        result = subprocess.run(
            [str(env / "bin" / "python"), "-c", code, *distributions],
            capture_output=True,
            text=True,
            check=True,
            timeout=60,
        )
        return json.loads(result.stdout)
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError) as error:
        return {"error": str(error)}


@dataclass
class ServerSpec:
    """How to start one server: its argv, API, request extras and version probe."""

    name: str
    api: str
    argv: list[str]
    extra: dict
    versions: dict
    environment: dict = field(default_factory=dict)


def server_spec(
    name: str, model_path: Path, model_id: str, address: str, args
) -> ServerSpec:
    host, port = address.rsplit(":", 1)
    context = str(args.context_tokens)
    if name == "metallix":
        mx = args.mx.resolve()
        return ServerSpec(
            name,
            # The path every other engine is measured on; thinking is off
            # unless a request sets reasoning_effort.
            "chat",
            [
                str(mx),
                "serve",
                "--model",
                str(model_path),
                "--model-id",
                model_id,
                "--listen",
                address,
                "--context-tokens",
                context,
                "--kv-budget-mib",
                str(args.mx_kv_budget_mib),
            ],
            {},
            {"mx": str(mx), "mx_revision": args.mx_revision},
        )
    if name == "vllm-metal":
        env = args.envs / "vllm-metal"
        return ServerSpec(
            name,
            "chat",
            [
                str(env / "bin" / "vllm"),
                "serve",
                str(model_path),
                "--served-model-name",
                model_id,
                "--host",
                host,
                "--port",
                port,
                "--max-model-len",
                context,
                # vllm-metal reserves this share of Metal memory for paged KV up
                # front (0.92 by default, about 97 GiB here), which starves other
                # work on a shared machine; 0.1 still holds several full contexts.
                "--gpu-memory-utilization",
                str(args.vllm_memory_fraction),
            ],
            # vLLM extensions: fixed-length output, and Qwen3 without thinking
            # to match metallix, which never enables it.
            {"ignore_eos": True, "chat_template_kwargs": {"enable_thinking": False}},
            env_python_versions(env, ["vllm", "vllm-metal", "mlx", "mlx-lm"]),
        )
    if name == "mtplx":
        env = args.envs / "mtplx"
        return ServerSpec(
            name,
            "chat",
            [
                str(env / "bin" / "mtplx"),
                "serve",
                "--model",
                str(model_path),
                "--model-id",
                model_id,
                "--host",
                host,
                "--port",
                port,
                "--no-auth",
                "--yes",
                "--reasoning",
                "off",
                "--scheduler-mode",
                args.mtplx_scheduler,
                # The default preset ("latency") admits one request at a time.
                "--batching-preset",
                args.mtplx_preset,
                "--context-window",
                context,
                "--no-stats-footer",
            ],
            {},
            env_python_versions(env, ["mtplx", "mlx", "mlx-lm"]),
        )
    if name == "mlx-lm":
        env = args.envs / "mlx-lm"
        return ServerSpec(
            name,
            "chat",
            [
                str(env / "bin" / "mlx_lm.server"),
                "--model",
                str(model_path),
                "--host",
                host,
                "--port",
                port,
                "--chat-template-args",
                json.dumps({"enable_thinking": False}),
            ],
            # mlx-lm treats any other model name as a Hub repository to load.
            {"model": "default_model"},
            env_python_versions(env, ["mlx-lm", "mlx"]),
        )
    if name in STUB_TPOT_MS:
        return ServerSpec(
            name,
            "chat",
            [
                sys.executable,
                str(ROOT / "scripts" / "bench_stub_server.py"),
                "--listen",
                address,
                "--tpot-ms",
                str(STUB_TPOT_MS[name]),
            ],
            {},
            {"stub": "dry run, no model"},
        )
    raise ValueError(f"unknown server {name!r}")


# Model-free servers for dry runs of the harness, at two fixed speeds.
STUB_TPOT_MS = {"stub": 2.0, "stub-slow": 4.0}
SERVERS = ("metallix", "vllm-metal", "mtplx", "mlx-lm", *STUB_TPOT_MS)

# Flags that force each server's prompt-prefix cache on or off, so shared-prefix
# results can be attributed to cache reuse or to the engine itself. MTPLX has
# no in-memory switch, so it runs only the "default" arm.
PREFIX_CACHE_FLAGS = {
    "metallix": {"on": [], "off": ["--prefix-cache-mib", "0"]},
    "vllm-metal": {
        "on": ["--enable-prefix-caching"],
        "off": ["--no-enable-prefix-caching"],
    },
    # mlx-lm evicts as soon as its LRU holds more than this many caches.
    "mlx-lm": {"on": [], "off": ["--prompt-cache-size", "0"]},
    **{
        stub: {"on": ["--prefix-cache", "on"], "off": ["--prefix-cache", "off"]}
        for stub in STUB_TPOT_MS
    },
}
CACHE_ARMS = ("default", "on", "off")


def with_prefix_cache(spec: ServerSpec, arm: str) -> ServerSpec:
    if arm == "default":
        return spec
    flags = PREFIX_CACHE_FLAGS.get(spec.name)
    if flags is None:
        raise ValueError(f"{spec.name} has no prefix-cache switch; use the default arm")
    return replace(spec, argv=spec.argv + flags[arm])


class ManagedServer:
    """A server process in its own process group, logging to a file."""

    def __init__(self, spec: ServerSpec, address: str, log_path: Path):
        self.spec = spec
        self.address = address
        self.log_path = log_path
        self.started = time.perf_counter()
        self.ready_s: float | None = None
        self.stop_lock = threading.Lock()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log = log_path.open("w")
        self.process = subprocess.Popen(
            spec.argv,
            stdout=self.log,
            stderr=subprocess.STDOUT,
            env=os.environ | spec.environment,
            start_new_session=True,
        )

    def wait_ready(self, timeout: float) -> None:
        deadline = time.perf_counter() + timeout
        while time.perf_counter() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError(
                    f"{self.spec.name} exited with {self.process.returncode}; see {self.log_path}"
                )
            try:
                with urllib.request.urlopen(
                    f"http://{self.address}/v1/models", timeout=5
                ) as response:
                    if response.status == 200:
                        self.ready_s = time.perf_counter() - self.started
                        return
            except (OSError, urllib.error.URLError):
                pass
            time.sleep(1)
        raise RuntimeError(f"{self.spec.name} not ready within {timeout:.0f} s")

    def stop(self) -> None:
        # The load sampler may stop the server from its thread to abort a level.
        with self.stop_lock:
            if self.log.closed:
                return
            self._stop()

    def _stop(self) -> None:
        # vLLM runs its engine in child processes, so signal the whole group.
        for sig, wait in ((signal.SIGTERM, 20), (signal.SIGKILL, 10)):
            try:
                os.killpg(self.process.pid, sig)
            except ProcessLookupError:
                break
            try:
                self.process.wait(timeout=wait)
                break
            except subprocess.TimeoutExpired:
                continue
        self.log.close()


# ---------------------------------------------------------------------------
# Runs


def load_average() -> list[float]:
    return [round(value, 2) for value in os.getloadavg()]


def level_spec(name: str, address: str, args) -> ServerSpec:
    """The spec for one engine; an empty argv for an already running --url server."""
    if args.url:
        return ServerSpec(name, args.api, [], {}, {})
    spec = with_prefix_cache(
        server_spec(name, args.model_path, args.model_id, address, args),
        args.prefix_cache,
    )
    return replace(spec, argv=spec.argv + args.server_args)


@functools.cache
def mx_serve_flags(mx: Path) -> frozenset[str]:
    """Long options this `mx serve` build lists in its help."""
    help_text = bench_system.command_output([str(mx), "serve", "--help"])
    return frozenset(re.findall(r"--[a-z][a-z0-9-]*", help_text))


def metallix_level_flags(supported: frozenset[str], in_flight: int) -> list[str]:
    """Admission flags sized to a level, limited to those the build accepts.

    `mx serve` refuses requests beyond its queue depth (default 8), so a level
    above that would measure rejections instead of serving; a batching build
    also caps its running sequences.
    """
    flags = []
    if "--queue-depth" in supported:
        flags += ["--queue-depth", str(max(in_flight, 8))]
    if "--max-num-seqs" in supported:
        flags += ["--max-num-seqs", str(in_flight)]
    return flags


def with_level(spec: ServerSpec, kind: str, value: float, args) -> ServerSpec:
    """A spec sized to one level: requests in flight at a concurrency, or every
    request of a rate run in the worst case."""
    if spec.name != "metallix" or args.url:
        return spec
    in_flight = int(value) if kind == "concurrency" else args.rate_requests
    flags = metallix_level_flags(mx_serve_flags(args.mx.resolve()), in_flight)
    return replace(spec, argv=spec.argv + flags)


def measure_level(
    name: str, set_name: str, kind: str, value: float, count: int, args, count_tokens
) -> dict:
    """One concurrency or rate level on a fresh server: warm up, then measure.

    A server that lived through earlier levels carries their prefix cache and
    allocator state into this one, so every managed level starts its own
    process. A --url server cannot be restarted; its levels share one lifetime.
    """
    result: dict = {kind: value, "restarted": not args.url}
    prompts = set_prompts(set_name, count, count_tokens, args.seed, args.file_prompts)
    warm = warmup_prompts(
        set_name, args.warmup, count_tokens, args.seed, args.file_prompts
    )
    server = None
    try:
        if args.url:
            address = args.url.removeprefix("http://").rstrip("/")
        else:
            address = bench_serve.free_address()
        spec = with_level(level_spec(name, address, args), kind, value, args)
        # The GPU-memory abort reads a machine-wide total, so record what was
        # already in use (other jobs) before this level's server started.
        baseline = bench_system.probe(None)
        result["baseline"] = {
            key: baseline.get(key)
            for key in ("gpu_in_use_bytes", "system_used_bytes", "load_1m")
        }
        if not args.url:
            log = args.log_dir / f"{name}-{set_name}-{kind}{value:g}.log"
            spec, trace = with_spans(spec, log, args)
            if trace:
                result["trace"] = str(trace)
            server = ManagedServer(spec, address, log)
            server.wait_ready(args.start_timeout)
            result |= {"ready_s": server.ready_s, "argv": spec.argv, "log": str(log)}
        level_concurrency = int(value) if kind == "concurrency" else 1
        # Warmup is sampled too: memory it allocates stays held for the level.
        sampler = bench_system.Sampler(
            pgid=server.process.pid if server else None,
            abort_load=args.abort_load,
            abort_gpu_bytes=(
                int(args.abort_gpu_gib * 2**30) if args.abort_gpu_gib else None
            ),
            on_abort=lambda reason: server.stop() if server else None,
        )
        with sampler:
            if warm:
                run_load(
                    address,
                    spec.api,
                    args.model_id,
                    warm,
                    args.max_tokens,
                    spec.extra,
                    concurrency=level_concurrency,
                    timeout=args.request_timeout,
                )
            before = load_average()
            records, duration = run_load(
                address,
                spec.api,
                args.model_id,
                prompts,
                args.max_tokens,
                spec.extra,
                concurrency=level_concurrency if kind == "concurrency" else None,
                rate=value if kind == "rate" else None,
                seed=args.seed,
                timeout=args.request_timeout,
            )
        result |= {
            "load_before": before,
            "load_after": load_average(),
            "memory": sampler.summary(),
            "summary": summarize(records, duration, args.slo_ttft_ms, args.slo_tpot_ms),
            "records": [asdict(record) for record in records],
        }
        if sampler.aborted:
            # Requests cut off by the abort would read as engine failures.
            before_gib = gib(result["baseline"]["gpu_in_use_bytes"])
            result["aborted"] = sampler.aborted + (
                f" ({before_gib:.1f} GiB in use before the server started)"
                if before_gib is not None
                else ""
            )
            del result["summary"]
            print(
                f"  {set_name} {kind}={value}: aborted: {result['aborted']}", flush=True
            )
        else:
            print(
                f"  {set_name} {kind}={value}: {one_line(result['summary'])}",
                flush=True,
            )
    except (RuntimeError, OSError) as error:
        result["error"] = str(error)
        print(f"  {set_name} {kind}={value}: not measured: {error}", flush=True)
    finally:
        if server:
            server.stop()
    # The timelines are complete only once the server has exited.
    if result.get("trace") and "summary" in result:
        add_server_spans(result, Path(result["trace"]), args.model_id, args.warmup)
    return result


def with_spans(spec: ServerSpec, log: Path, args) -> tuple[ServerSpec, Path | None]:
    """With --mx-spans, have metallix write its span timeline beside its log."""
    if spec.name != "metallix" or not getattr(args, "mx_spans", False):
        return spec, None
    if "--trace-out" not in mx_serve_flags(args.mx.resolve()):
        print("  warning: this mx build has no --trace-out (features: timeline)")
        return spec, None
    trace = log.with_suffix(".trace.json")
    return (
        replace(
            spec,
            argv=spec.argv + ["--trace-out", str(trace)],
            environment=spec.environment | {"METALLIX_LOG": "info"},
        ),
        trace,
    )


def add_server_spans(result: dict, trace: Path, model_id: str, warmup: int) -> None:
    """Server-side request timings next to the client's, and any disagreement."""
    try:
        server = mx_spans.summarize_files(trace, model_id, skip=warmup)
    except (OSError, ValueError) as error:
        result["server_spans"] = {"error": str(error)}
        return
    result["server_spans"] = server
    result["span_mismatches"] = mx_spans.compare(server, result["summary"])
    print(f"    {mx_spans.one_line(server)}", flush=True)
    for mismatch in result["span_mismatches"]:
        print(f"    mismatch: {mismatch}", flush=True)


def measure_set(name: str, set_name: str, args, count_tokens) -> dict:
    out: dict = {"set": set_name, "concurrency": [], "rates": []}
    for concurrency in args.concurrency:
        count = max(args.min_requests, args.requests_per_slot * concurrency)
        out["concurrency"].append(
            measure_level(
                name, set_name, "concurrency", concurrency, count, args, count_tokens
            )
        )
    for rate in args.rates:
        out["rates"].append(
            measure_level(
                name, set_name, "rate", rate, args.rate_requests, args, count_tokens
            )
        )
    if out["rates"]:
        # A rate that could not be measured counts as a miss.
        out["goodput_rps"] = goodput(
            {
                run["rate"]: run["summary"]["slo_attainment"] if "summary" in run else 0
                for run in out["rates"]
            }
        )
    return out


def report_cells(servers: list[dict]) -> dict[tuple, dict[str, list[dict]]]:
    """Records per (set, mode, level) and engine, from a report's server entries."""
    cells: dict[tuple, dict[str, list[dict]]] = {}
    for server in servers:
        for result in server.get("sets", []):
            for mode, kind, runs in (
                ("c", "concurrency", result["concurrency"]),
                ("r", "rate", result["rates"]),
            ):
                for run in runs:
                    if "summary" in run:
                        key = (result["set"], mode, run[kind])
                        cells.setdefault(key, {})[server["name"]] = run["records"]
    return cells


def failure_warnings(cells: dict[tuple, dict[str, list[dict]]]) -> list[str]:
    """Each engine's failed requests per cell, by prompt label and outcome.

    A rejected prompt (for example a recorded tool conversation an engine
    refuses) counts against that engine's throughput and SLO attainment; the
    labels show which prompts did it, so the prompt set stays whole.
    """
    warnings = []
    for (set_name, mode, value), by_engine in cells.items():
        for engine, records in by_engine.items():
            failed = [r for r in records if r["outcome"] != "ok"]
            if failed:
                # Synthetic sets give every prompt the set's label, so repeats
                # are counted rather than listed.
                groups: dict[tuple[str, str], int] = {}
                for r in sorted(failed, key=lambda r: r["index"]):
                    key = (r["label"], r["outcome"])
                    groups[key] = groups.get(key, 0) + 1
                listed = ", ".join(
                    f"{count} x {label} ({outcome})"
                    if count > 1
                    else f"{label} ({outcome})"
                    for (label, outcome), count in groups.items()
                )
                warnings.append(
                    f"{engine} {set_name} {mode}={value:g}: "
                    f"{len(failed)}/{len(records)} failed: {listed}"
                )
    return warnings


def token_count_warnings(cells: dict[tuple, dict[str, list[dict]]]) -> list[str]:
    """Cells where engines generated different output token counts for the same
    prompts. Throughput then compares unequal work: an engine that stops early
    at EOS, or ignores ignore_eos, looks faster than it is."""
    warnings = []
    for (set_name, mode, value), by_engine in cells.items():
        counts = {
            engine: {
                r["index"]: r["output_tokens"] for r in records if r["outcome"] == "ok"
            }
            for engine, records in by_engine.items()
        }
        if len(counts) < 2:
            continue
        shared = sorted(set.intersection(*(set(c) for c in counts.values())))
        differing = [i for i in shared if len({c[i] for c in counts.values()}) > 1]
        if differing:
            totals = ", ".join(
                f"{engine} {sum(c[i] for i in shared)}" for engine, c in counts.items()
            )
            warnings.append(
                f"{set_name} {mode}={value:g}: output tokens differ on "
                f"{len(differing)}/{len(shared)} prompts completed by every engine "
                f"(totals: {totals})"
            )
    return warnings


def fmt(value, digits: int = 0) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def one_line(summary: dict) -> str:
    failed = {k: v for k, v in summary["outcomes"].items() if k != "ok"}
    return (
        f"{summary['request_throughput']:.2f} req/s, "
        f"{summary['output_token_throughput']:.0f} out tok/s, "
        f"TTFT p50 {fmt(summary['ttft_ms']['p50'])} p99 {fmt(summary['ttft_ms']['p99'])} ms, "
        f"TPOT p50 {fmt(summary['tpot_ms']['p50'], 1)} p99 {fmt(summary['tpot_ms']['p99'], 1)} ms, "
        f"SLO {summary['slo_attainment']:.0%}"
        + (f", failed {failed}" if failed else "")
    )


def render(report: dict) -> str:
    lines = [
        f"{report['machine']}; repo at {report['revision']}; model {report['model_path']}",
        (
            f"SLO: TTFT <= {report['slo']['ttft_ms']} ms and TPOT <= {report['slo']['tpot_ms']} ms; "
            f"max output {report['max_tokens']} tokens"
        ),
        "",
        "server      set            load    mode     req/s  out tok/s  TTFT p50/p90/p99 ms   TPOT p50/p90/p99 ms  ok/n   SLO  mem GiB",
    ]
    for server in report["servers"]:
        if server.get("error"):
            lines.append(f"{server['name']:<11} not measured: {server['error']}")
            continue
        for result in server["sets"]:
            runs = [("c", r["concurrency"], r) for r in result["concurrency"]]
            runs += [("r", r["rate"], r) for r in result["rates"]]
            for kind, value, run in runs:
                if "summary" not in run:
                    lines.append(
                        f"{server['name']:<11} {result['set']:<14} {'':>5}  "
                        f"{kind}={value:<6g} not measured: "
                        f"{run.get('error') or run.get('aborted')}"
                    )
                    continue
                s = run["summary"]
                ttft, tpot = s["ttft_ms"], s["tpot_ms"]
                lines.append(
                    f"{server['name']:<11} {result['set']:<14} {run['load_before'][0]:>5.1f}  "
                    f"{kind}={value:<6g} {s['request_throughput']:>5.2f}  {s['output_token_throughput']:>9.0f}  "
                    f"{fmt(ttft['p50']):>6}/{fmt(ttft['p90']):>6}/{fmt(ttft['p99']):>6}   "
                    f"{fmt(tpot['p50'], 1):>5}/{fmt(tpot['p90'], 1):>5}/{fmt(tpot['p99'], 1):>5}    "
                    f"{s['outcomes'].get('ok', 0):>2}/{s['requests']:<2}  {s['slo_attainment']:>4.0%}  "
                    f"{fmt(gib(run['memory']['peak_system_used_bytes']), 1):>7}"
                )
            if "goodput_rps" in result:
                lines.append(
                    f"{'':<11} {result['set']:<14} goodput {fmt(result['goodput_rps'], 2)} req/s "
                    f"(highest swept rate with >= {ATTAINMENT_GOAL:.0%} SLO attainment)"
                )
    lines += [f"warning: {warning}" for warning in report.get("warnings", [])]
    return "\n".join(lines)


def idle_gate() -> dict:
    """The campaign's idle gate, recorded (not enforced) before a load run."""
    readings = bench_system.idle_readings()
    failures = bench_system.idle_failures(readings)
    for failure in failures:
        print(f"warning: machine not idle: {failure}", flush=True)
    return {"readings": readings, "failures": failures}


def gib(value: float | None) -> float | None:
    return None if value is None else value / 2**30


def version_pair(text: str) -> tuple[str, str]:
    key, sep, value = text.partition("=")
    if not sep or not key:
        raise argparse.ArgumentTypeError(f"expected NAME=VERSION, got {text!r}")
    return key, value


def floats(text: str) -> list[float]:
    return [float(part) for part in text.split(",") if part]


def build_parser(description: str = __doc__.splitlines()[0]) -> argparse.ArgumentParser:
    """Every option of a load run; the campaign driver adds its own to these."""
    parser = argparse.ArgumentParser(description=description)
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--server", help=f"comma-separated, from {', '.join(SERVERS)}")
    target.add_argument("--url", help="an already running server (no start or stop)")
    parser.add_argument(
        "--label", default="external", help="engine name for a --url server"
    )
    parser.add_argument(
        "--launch-argv",
        type=shlex.split,
        help="how the --url server was started, as one shell-quoted string",
    )
    parser.add_argument(
        "--version",
        type=version_pair,
        action="append",
        default=[],
        dest="versions",
        metavar="NAME=VERSION",
        help="an engine version or commit for a --url server; repeatable",
    )
    parser.add_argument(
        "--abort-load",
        type=float,
        help="abort a level when the 1-min load average rises above this",
    )
    parser.add_argument(
        "--abort-gpu-gib",
        type=float,
        default=64.0,
        help="abort a level when GPU-resident memory rises above this many GiB "
        "(a leaking server otherwise swaps the machine to a halt; 0 disables)",
    )
    parser.add_argument("--api", choices=("chat", "responses"), default="chat")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--model-id", default="qwen3-0.6b")
    parser.add_argument("--sets", default="shared-prefix,long")
    parser.add_argument(
        "--prompts-file",
        type=Path,
        help=f"JSON or JSON Lines prompts for the {FILE_SET!r} set "
        "(for example SiliconBench's agent split)",
    )
    parser.add_argument(
        "--concurrency",
        type=lambda t: [int(x) for x in floats(t)],
        default=[1, 2, 4, 8],
    )
    parser.add_argument("--rates", type=floats, default=[], help="requests per second")
    parser.add_argument("--requests-per-slot", type=int, default=2)
    parser.add_argument("--min-requests", type=int, default=6)
    parser.add_argument("--rate-requests", type=int, default=16)
    parser.add_argument(
        "--warmup",
        type=int,
        default=3,
        help="discarded requests before each level, sent at its concurrency",
    )
    parser.add_argument(
        "--prefix-cache",
        choices=CACHE_ARMS,
        default="default",
        help="force each managed server's prompt-prefix cache on or off; with "
        "--url only a label for how the server was started",
    )
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--slo-ttft-ms", type=float, default=2000)
    parser.add_argument("--slo-tpot-ms", type=float, default=50)
    parser.add_argument("--context-tokens", type=int, default=10240)
    parser.add_argument("--mx", type=Path, default=Path("target/release/mx"))
    parser.add_argument(
        "--mx-revision", default="unknown", help="source revision of --mx"
    )
    parser.add_argument("--mx-kv-budget-mib", type=int, default=4096)
    parser.add_argument(
        "--server-arg",
        action="append",
        default=[],
        dest="server_args",
        metavar="ARG",
        help="append ARG to each managed server's command line; repeatable "
        "(write --server-arg=--flag for values that start with a dash)",
    )
    parser.add_argument(
        "--mx-spans",
        action="store_true",
        help="have metallix write a span timeline per level (METALLIX_LOG=info, "
        "--trace-out) and compare its timings with the client's; tracing adds "
        "work to the server, so use it on a separate pass, not a measured one",
    )
    parser.add_argument("--mtplx-scheduler", default="ar_batch")
    parser.add_argument("--mtplx-preset", default="throughput")
    parser.add_argument("--vllm-memory-fraction", type=float, default=0.1)
    parser.add_argument("--start-timeout", type=float, default=600)
    parser.add_argument("--request-timeout", type=float, default=600)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json", type=Path, help="write the full report here")
    parser.add_argument(
        "--envs",
        type=Path,
        default=ENVS,
        help="directory holding one virtual environment per Python server, "
        "named vllm-metal, mtplx and mlx-lm",
    )
    parser.add_argument("--log-dir", type=Path, default=ENVS / "logs")
    return parser


def prompt_sets(parser: argparse.ArgumentParser, args) -> list[str]:
    """Check --sets against --prompts-file and load the file into args."""
    sets = [name for name in args.sets.split(",") if name]
    for name in sets:
        if name not in PROMPT_SETS:
            parser.error(f"unknown set {name!r}; choose from {', '.join(PROMPT_SETS)}")
    if (FILE_SET in sets) != (args.prompts_file is not None):
        parser.error(f"--prompts-file and the {FILE_SET!r} set go together")
    args.file_prompts = (
        load_prompts_file(args.prompts_file) if args.prompts_file else []
    )
    return sets


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    sets = prompt_sets(parser, args)
    count_tokens = tokenizer_counter(args.model_path)
    report = {
        "machine": f"{platform.machine()} {platform.platform()} "
        + subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip(),
        "revision": bench_serve.repo_revision(),
        "model_path": str(args.model_path),
        "model_id": args.model_id,
        "max_tokens": args.max_tokens,
        "prompts_file": file_identity(args.prompts_file),
        "slo": {
            "ttft_ms": args.slo_ttft_ms,
            "tpot_ms": args.slo_tpot_ms,
            "attainment": ATTAINMENT_GOAL,
        },
        "system": bench_system.system_info(),
        "idle_gate": idle_gate(),
        "load_before": load_average(),
        "servers": [],
    }
    names = args.server.split(",") if args.server else [args.label]
    for name in names:
        print(f"{name}:", flush=True)
        entry: dict = {"name": name, "prefix_cache": args.prefix_cache}
        report["servers"].append(entry)
        try:
            spec = level_spec(name, "127.0.0.1:0", args)
        except (ValueError, OSError) as error:
            entry["error"] = str(error)
            print(f"  not measured: {error}", flush=True)
            continue
        entry["api"] = spec.api
        entry["versions"] = spec.versions | dict(args.versions)
        if args.url:
            entry["url"] = args.url
            entry["launch_argv"] = args.launch_argv
        entry["request_extra"] = spec.extra
        entry["load_before"] = load_average()
        entry["sets"] = [
            measure_set(name, set_name, args, count_tokens) for set_name in sets
        ]
        entry["load_after"] = load_average()
    report["load_after"] = load_average()
    cells = report_cells(report["servers"])
    report["warnings"] = failure_warnings(cells) + token_count_warnings(cells)
    print()
    print(render(report))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
