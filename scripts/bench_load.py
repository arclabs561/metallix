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
MTPLX or mlx-lm's server, the Python ones from virtual environments under
.agents/bench-envs/. Load averages are recorded around every run, because
numbers from a shared machine are hard to read without them.

Usage:
  uv run scripts/bench_load.py --server metallix --model-path DIR \\
      --sets shared-prefix,long --concurrency 1,2,4,8 --rates 0.5,1,2
  uv run scripts/bench_load.py --url http://127.0.0.1:8000 --api chat \\
      --model-id qwen3 --model-path DIR --concurrency 1
"""

from __future__ import annotations

import argparse
import http.client
import itertools
import json
import math
import os
import platform
import random
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path

import bench_serve

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
    """One request's text: an optional system part and the user turn."""

    label: str
    system: str | None
    user: str


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


PROMPT_SETS = ("short", "long", "shared-prefix", "mixed")


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
    if api == "chat":
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
            "input": prompt.user,
            "max_output_tokens": max_tokens,
            "temperature": 0,
            "stream": True,
        }
        if prompt.system:
            body["instructions"] = prompt.system
        return "/v1/responses", body | extra
    raise ValueError(f"unknown api {api!r}")


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
            "responses",
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
        env = ENVS / "vllm-metal"
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
        env = ENVS / "mtplx"
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
        env = ENVS / "mlx-lm"
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
    raise ValueError(f"unknown server {name!r}")


SERVERS = ("metallix", "vllm-metal", "mtplx", "mlx-lm")


class ManagedServer:
    """A server process in its own process group, logging to a file."""

    def __init__(self, spec: ServerSpec, address: str, log_path: Path):
        self.spec = spec
        self.address = address
        self.log_path = log_path
        self.started = time.perf_counter()
        self.ready_s: float | None = None
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


def measure_set(
    address: str, spec: ServerSpec, model_id: str, set_name: str, args, count_tokens
) -> dict:
    out: dict = {"set": set_name, "concurrency": [], "rates": []}
    warm = build_prompts(set_name, args.warmup, count_tokens, seed=args.seed + 999)
    if warm:
        run_load(
            address,
            spec.api,
            model_id,
            warm,
            args.max_tokens,
            spec.extra,
            concurrency=1,
        )

    def one_run(kind: str, value: float, prompts: list[Prompt]) -> dict:
        before = load_average()
        records, duration = run_load(
            address,
            spec.api,
            model_id,
            prompts,
            args.max_tokens,
            spec.extra,
            concurrency=int(value) if kind == "concurrency" else None,
            rate=value if kind == "rate" else None,
            seed=args.seed,
            timeout=args.request_timeout,
        )
        result = {
            kind: value,
            "load_before": before,
            "load_after": load_average(),
            "summary": summarize(records, duration, args.slo_ttft_ms, args.slo_tpot_ms),
            "records": [asdict(record) for record in records],
        }
        print(f"  {set_name} {kind}={value}: {one_line(result['summary'])}", flush=True)
        return result

    for concurrency in args.concurrency:
        count = max(args.min_requests, args.requests_per_slot * concurrency)
        prompts = build_prompts(set_name, count, count_tokens, args.seed)
        out["concurrency"].append(one_run("concurrency", concurrency, prompts))
    for rate in args.rates:
        prompts = build_prompts(set_name, args.rate_requests, count_tokens, args.seed)
        out["rates"].append(one_run("rate", rate, prompts))
    if out["rates"]:
        out["goodput_rps"] = goodput(
            {run["rate"]: run["summary"]["slo_attainment"] for run in out["rates"]}
        )
    return out


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
        "server      set            load    mode     req/s  out tok/s  TTFT p50/p90/p99 ms   TPOT p50/p90/p99 ms  ok/n   SLO",
    ]
    for server in report["servers"]:
        if server.get("error"):
            lines.append(f"{server['name']:<11} not measured: {server['error']}")
            continue
        for result in server["sets"]:
            runs = [("c", r["concurrency"], r) for r in result["concurrency"]]
            runs += [("r", r["rate"], r) for r in result["rates"]]
            for kind, value, run in runs:
                s = run["summary"]
                ttft, tpot = s["ttft_ms"], s["tpot_ms"]
                lines.append(
                    f"{server['name']:<11} {result['set']:<14} {run['load_before'][0]:>5.1f}  "
                    f"{kind}={value:<6g} {s['request_throughput']:>5.2f}  {s['output_token_throughput']:>9.0f}  "
                    f"{fmt(ttft['p50']):>6}/{fmt(ttft['p90']):>6}/{fmt(ttft['p99']):>6}   "
                    f"{fmt(tpot['p50'], 1):>5}/{fmt(tpot['p90'], 1):>5}/{fmt(tpot['p99'], 1):>5}    "
                    f"{s['outcomes'].get('ok', 0):>2}/{s['requests']:<2}  {s['slo_attainment']:>4.0%}"
                )
            if "goodput_rps" in result:
                lines.append(
                    f"{'':<11} {result['set']:<14} goodput {fmt(result['goodput_rps'], 2)} req/s "
                    f"(highest swept rate with >= {ATTAINMENT_GOAL:.0%} SLO attainment)"
                )
    return "\n".join(lines)


def floats(text: str) -> list[float]:
    return [float(part) for part in text.split(",") if part]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--server", help=f"comma-separated, from {', '.join(SERVERS)}")
    target.add_argument("--url", help="an already running server (no start or stop)")
    parser.add_argument("--api", choices=("chat", "responses"), default="chat")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--model-id", default="qwen3-0.6b")
    parser.add_argument("--sets", default="shared-prefix,long")
    parser.add_argument(
        "--concurrency",
        type=lambda t: [int(x) for x in floats(t)],
        default=[1, 2, 4, 8],
    )
    parser.add_argument("--rates", type=floats, default=[], help="requests per second")
    parser.add_argument("--requests-per-slot", type=int, default=2)
    parser.add_argument("--min-requests", type=int, default=6)
    parser.add_argument("--rate-requests", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--slo-ttft-ms", type=float, default=2000)
    parser.add_argument("--slo-tpot-ms", type=float, default=50)
    parser.add_argument("--context-tokens", type=int, default=10240)
    parser.add_argument("--mx", type=Path, default=Path("target/release/mx"))
    parser.add_argument(
        "--mx-revision", default="unknown", help="source revision of --mx"
    )
    parser.add_argument("--mx-kv-budget-mib", type=int, default=4096)
    parser.add_argument("--mtplx-scheduler", default="ar_batch")
    parser.add_argument("--mtplx-preset", default="throughput")
    parser.add_argument("--vllm-memory-fraction", type=float, default=0.1)
    parser.add_argument("--start-timeout", type=float, default=600)
    parser.add_argument("--request-timeout", type=float, default=600)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json", type=Path, help="write the full report here")
    parser.add_argument("--log-dir", type=Path, default=ENVS / "logs")
    args = parser.parse_args()
    sets = [name for name in args.sets.split(",") if name]
    for name in sets:
        if name not in PROMPT_SETS:
            parser.error(f"unknown set {name!r}; choose from {', '.join(PROMPT_SETS)}")

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
        "slo": {
            "ttft_ms": args.slo_ttft_ms,
            "tpot_ms": args.slo_tpot_ms,
            "attainment": ATTAINMENT_GOAL,
        },
        "load_before": load_average(),
        "servers": [],
    }
    names = args.server.split(",") if args.server else ["external"]
    for name in names:
        print(f"{name}:", flush=True)
        entry: dict = {"name": name}
        report["servers"].append(entry)
        server = None
        try:
            if args.url:
                address = args.url.removeprefix("http://").rstrip("/")
                spec = ServerSpec(name, args.api, [], {}, {})
            else:
                address = bench_serve.free_address()
                spec = server_spec(name, args.model_path, args.model_id, address, args)
                server = ManagedServer(spec, address, args.log_dir / f"{name}.log")
                server.wait_ready(args.start_timeout)
                entry["ready_s"] = server.ready_s
                entry["argv"] = spec.argv
                entry["log"] = str(server.log_path)
            entry["api"] = spec.api
            entry["versions"] = spec.versions
            entry["request_extra"] = spec.extra
            entry["load_before"] = load_average()
            entry["sets"] = [
                measure_set(address, spec, args.model_id, set_name, args, count_tokens)
                for set_name in sets
            ]
        except (RuntimeError, OSError) as error:
            entry["error"] = str(error)
            print(f"  not measured: {error}", flush=True)
        finally:
            if server:
                server.stop()
            entry["load_after"] = load_average()
    report["load_after"] = load_average()
    print()
    print(render(report))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
