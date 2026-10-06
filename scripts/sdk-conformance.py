#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "openai==3.24.0",
#   "anthropic==1.11.0",
#   "httpx==0.28.1",
#   "jsonschema==4.25.1",
# ]
# ///
"""Opt-in client conformance: drive `mx serve` through the official OpenAI and
Anthropic Python SDKs and check every reply against the published contracts.

Starts `mx serve` on a local checkpoint, runs Chat Completions, Responses and
Messages cases (plain and streamed, tool round trips, output limits, stop
sequences, logprobs, `ignore_eos`, error shapes), then stops the server. The
server's process group is stopped if its memory footprint, which counts Metal
buffers, passes `--max-footprint-gib`.

Expected values come from the specs, never from what the server returns:

- OpenAI: the OpenAPI document at OPENAI_SPEC_URL (pinned by commit and
  SHA-256), validated with JSON Schema, plus the field descriptions it states.
- Anthropic: the API reference (https://platform.claude.com/docs/en/api/errors,
  https://platform.claude.com/docs/en/build-with-claude/streaming) and the
  response types of the official TypeScript SDK at ANTHROPIC_TYPES_URL, where
  a field without `?` is always present (possibly null).

A check is either `fail` (the contract is broken) or `note` (a field the
contract lists as always present but nullable is missing, which the SDKs
tolerate). The exit status is 1 when any check fails.

    uv run scripts/sdk-conformance.py --mx target/release/mx \\
        --model /path/to/Qwen3-0.6B
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Self

OPENAI_SPEC_COMMIT = "8f5077ae70efcd2755a24d4df3c705de26ce84d2"
OPENAI_SPEC_URL = (
    "https://raw.githubusercontent.com/openai/openai-openapi/"
    f"{OPENAI_SPEC_COMMIT}/openapi.json"
)
OPENAI_SPEC_SHA256 = "c1d96b52bc4d401b66fc6129c07b1807c5572a011939e12b6d75d75a178e7e95"
ANTHROPIC_TYPES_URL = (
    "https://github.com/anthropics/anthropic-sdk-typescript/blob/"
    "d49bdab458000bcdffe77bd84b03293f31824fb3/src/resources/messages/messages.ts"
)

MODEL_ID = "conformance-model"
WEATHER_ARGS = {
    "type": "object",
    "properties": {"city": {"type": "string"}},
    "required": ["city"],
}
WEATHER_QUESTION = "What is the weather in Paris right now? Use the tool."
COUNTING = "Count from one to twenty in words, separated by commas."

# Always present per the TypeScript SDK types (no `?`), nullable or not.
ANTHROPIC_MESSAGE = (
    "id", "type", "role", "content", "model", "stop_reason", "stop_sequence",
    "usage", "container", "diagnostics", "stop_details",
)  # fmt: skip
ANTHROPIC_MESSAGE_CORE = ANTHROPIC_MESSAGE[:8]
ANTHROPIC_USAGE = (
    "input_tokens", "output_tokens", "cache_creation", "cache_creation_input_tokens",
    "cache_read_input_tokens", "inference_geo", "output_tokens_details",
    "server_tool_use", "service_tier",
)  # fmt: skip
ANTHROPIC_USAGE_CORE = ANTHROPIC_USAGE[:2]
ANTHROPIC_STOP_REASONS = {
    "end_turn", "max_tokens", "stop_sequence", "tool_use", "pause_turn",
    "refusal", "model_context_window_exceeded",
}  # fmt: skip


# ---------------------------------------------------------------------------
# Results


@dataclass
class Report:
    checks: list[tuple[str, str, str, str]] = field(default_factory=list)
    case: str = ""

    def check(self, name: str, ok: bool, detail: object = "", level: str = "fail"):
        status = "pass" if ok else level
        self.checks.append((self.case, name, status, "" if ok else str(detail)))
        if not ok:
            print(f"  {status.upper()} {self.case}: {name}: {str(detail)[:300]}")
        return ok

    def present(self, obj: dict, keys, where: str, level: str = "fail") -> None:
        missing = [key for key in keys if key not in obj]
        self.check(
            f"{where} has {', '.join(keys)}", not missing, f"missing {missing}", level
        )

    def failures(self) -> list[tuple[str, str, str, str]]:
        return [row for row in self.checks if row[2] != "pass"]


# ---------------------------------------------------------------------------
# Server and footprint guard


def footprint_bytes(text: str) -> int | None:
    """The `Footprint:` total from `footprint <pid>` output, in bytes."""
    match = re.search(r"Footprint:\s*([\d.]+)\s*(B|KB|MB|GB)", text)
    if not match:
        return None
    scale = {"B": 1, "KB": 2**10, "MB": 2**20, "GB": 2**30}[match.group(2)]
    return int(float(match.group(1)) * scale)


class FootprintGuard:
    """Sums the footprint of every process in `pgid` each second and stops the
    group once it passes `cap` bytes. Footprint includes Metal buffers, which
    RSS does not."""

    def __init__(self, pgid: int, cap: int):
        self.pgid, self.cap = pgid, cap
        self.peak = 0
        self.aborted: str | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def sample(self) -> int:
        pids = subprocess.run(
            ["pgrep", "-g", str(self.pgid)], capture_output=True, text=True, check=False
        ).stdout.split()
        total = 0
        for pid in pids:
            out = subprocess.run(
                ["footprint", pid], capture_output=True, text=True, check=False
            )
            total += footprint_bytes(out.stdout) or 0
        return total

    def _run(self) -> None:
        while not self._stop.wait(1.0):
            total = self.sample()
            self.peak = max(self.peak, total)
            if total > self.cap and self.aborted is None:
                self.aborted = (
                    f"server footprint {total / 2**30:.2f} GiB passed "
                    f"{self.cap / 2**30:g} GiB"
                )
                print(f"ABORT: {self.aborted}", file=sys.stderr)
                os.killpg(self.pgid, signal.SIGTERM)

    def __enter__(self) -> Self:
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        self._thread.join()


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def wait_healthy(base: str, server: subprocess.Popen, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if server.poll() is not None:
            raise SystemExit(f"mx serve exited with {server.returncode} during startup")
        try:
            with urllib.request.urlopen(f"{base}/healthz", timeout=2) as reply:
                if reply.status == 200:
                    return
        except OSError:
            pass
        time.sleep(0.5)
    raise SystemExit(f"mx serve was not healthy after {timeout:g}s")


# ---------------------------------------------------------------------------
# Spec validation


def openai_spec(cache: Path) -> dict:
    path = cache / f"openai-openapi-{OPENAI_SPEC_COMMIT[:12]}.json"
    if not path.exists():
        cache.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(OPENAI_SPEC_URL, timeout=60) as reply:
            path.write_bytes(reply.read())
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != OPENAI_SPEC_SHA256:
        raise SystemExit(f"{path} has SHA-256 {digest}, expected {OPENAI_SPEC_SHA256}")
    return json.loads(raw)


class OpenAISchemas:
    def __init__(self, spec: dict):
        import jsonschema
        import referencing
        import referencing.jsonschema

        resource = referencing.Resource.from_contents(
            spec, default_specification=referencing.jsonschema.DRAFT202012
        )
        self.registry = referencing.Registry().with_resource("spec", resource)
        self.jsonschema = jsonschema
        self.schemas = spec["components"]["schemas"]
        # Stream event type -> schema name, from the ResponseStreamEvent union.
        self.response_events = {}
        for ref in self.schemas["ResponseStreamEvent"]["anyOf"]:
            name = ref["$ref"].rsplit("/", 1)[1]
            kind = self.schemas[name].get("properties", {}).get("type", {})
            for value in kind.get("enum", []):
                self.response_events[value] = name

    def errors(self, name: str, value: object) -> list[str]:
        validator = self.jsonschema.Draft202012Validator(
            {"$ref": f"spec#/components/schemas/{name}"}, registry=self.registry
        )
        return [
            f"{'/'.join(map(str, error.absolute_path)) or '<root>'}: {error.message}"
            for error in validator.iter_errors(value)
        ]


def sse_events(lines) -> list[tuple[str | None, str]]:
    """(event name, data) for each server-sent event in `lines`."""
    events, name, data = [], None, []
    for line in lines:
        line = line.decode() if isinstance(line, bytes) else line
        if line == "":
            if data:
                events.append((name, "\n".join(data)))
            name, data = None, []
        elif line.startswith(":"):
            continue
        elif line.startswith("event:"):
            name = line[6:].strip()
        elif line.startswith("data:"):
            data.append(line[5:].lstrip())
    if data:
        events.append((name, "\n".join(data)))
    return events


# ---------------------------------------------------------------------------
# Chat Completions


def chat_cases(client, schemas: OpenAISchemas, report: Report) -> None:
    report.case = "chat"
    import openai

    def valid(name: str, value: object, where: str) -> None:
        errors = schemas.errors(name, value)
        report.check(f"{where} matches {name}", not errors, errors[:5])

    base = {"model": MODEL_ID, "temperature": 0}
    hello = [{"role": "user", "content": "Say hello in one short sentence."}]

    report.case = "chat.plain"
    raw = client.chat.completions.with_raw_response.create(
        **base, messages=hello, max_completion_tokens=32
    )
    body = raw.http_response.json()
    valid("CreateChatCompletionResponse", body, "body")
    report.check(
        "object is chat.completion",
        body.get("object") == "chat.completion",
        body.get("object"),
    )
    choice = body["choices"][0]
    report.check(
        "finish_reason is stop",
        choice.get("finish_reason") == "stop",
        choice.get("finish_reason"),
    )
    usage = body.get("usage") or {}
    report.check(
        "total_tokens = prompt_tokens + completion_tokens",
        usage.get("total_tokens")
        == usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0),
        usage,
    )
    report.check(
        "SDK parses the reply",
        bool(raw.parse().choices[0].message.content),
        "empty content",
    )

    report.case = "chat.stream"
    with client.chat.completions.with_streaming_response.create(
        **base,
        messages=hello,
        max_completion_tokens=32,
        stream=True,
        stream_options={"include_usage": True},
    ) as reply:
        events = sse_events(reply.iter_lines())
    report.check(
        "stream ends with [DONE]",
        bool(events) and events[-1][1] == "[DONE]",
        events[-1:],
    )
    chunks = [json.loads(data) for _, data in events if data != "[DONE]"]
    for index, chunk in enumerate(chunks):
        valid("CreateChatCompletionStreamResponse", chunk, f"chunk {index}")
    report.check(
        "one id across chunks",
        len({chunk.get("id") for chunk in chunks}) == 1,
        {chunk.get("id") for chunk in chunks},
    )
    report.check(
        "every chunk is chat.completion.chunk",
        all(chunk.get("object") == "chat.completion.chunk" for chunk in chunks),
        [chunk.get("object") for chunk in chunks],
    )
    last = chunks[-1] if chunks else {}
    # ChatCompletionStreamOptions.include_usage: an extra final chunk with
    # usage and empty choices; every other chunk has usage null.
    report.check(
        "usage chunk has empty choices", last.get("choices") == [], last.get("choices")
    )
    report.check(
        "usage chunk carries usage",
        isinstance(last.get("usage"), dict),
        last.get("usage"),
    )
    report.check(
        "other chunks carry usage: null",
        all("usage" in chunk and chunk["usage"] is None for chunk in chunks[:-1]),
        [chunk.get("usage", "<absent>") for chunk in chunks[:-1]][:3],
    )
    finishes = [
        c["finish_reason"]
        for chunk in chunks
        for c in chunk.get("choices", [])
        if c.get("finish_reason")
    ]
    report.check("one finish_reason, stop", finishes == ["stop"], finishes)
    streamed = "".join(
        c.get("delta", {}).get("content") or ""
        for chunk in chunks
        for c in chunk.get("choices", [])
    )
    sdk_text = "".join(
        chunk.choices[0].delta.content or ""
        for chunk in client.chat.completions.create(
            **base, messages=hello, max_completion_tokens=32, stream=True
        )
        if chunk.choices
    )
    report.check(
        "SDK stream yields the same text",
        sdk_text == streamed and streamed,
        (sdk_text, streamed),
    )

    report.case = "chat.max_tokens"
    body = client.chat.completions.with_raw_response.create(
        **base,
        messages=[{"role": "user", "content": COUNTING}],
        max_completion_tokens=4,
    ).http_response.json()
    report.check(
        "finish_reason is length",
        body["choices"][0].get("finish_reason") == "length",
        body["choices"][0].get("finish_reason"),
    )
    report.check(
        "completion_tokens <= 4",
        (body.get("usage") or {}).get("completion_tokens", 99) <= 4,
        body.get("usage"),
    )

    report.case = "chat.stop"
    try:
        body = client.chat.completions.with_raw_response.create(
            **base,
            messages=[{"role": "user", "content": COUNTING}],
            max_completion_tokens=64,
            stop=[","],
        ).http_response.json()
        choice = body["choices"][0]
        # StopConfiguration: "The returned text will not contain the stop sequence."
        report.check(
            "finish_reason is stop",
            choice.get("finish_reason") == "stop",
            choice.get("finish_reason"),
        )
        report.check(
            "text excludes the stop sequence",
            "," not in (choice["message"].get("content") or ""),
            choice["message"].get("content"),
        )
    except openai.APIStatusError as error:
        report.check(
            "stop is accepted",
            False,
            f"{error.status_code} {error.response.text[:200]}",
        )

    report.case = "chat.logprobs"
    body = client.chat.completions.with_raw_response.create(
        **base,
        messages=[{"role": "user", "content": COUNTING}],
        max_completion_tokens=8,
        logprobs=True,
        top_logprobs=2,
    ).http_response.json()
    valid("CreateChatCompletionResponse", body, "body")
    content = ((body["choices"][0].get("logprobs") or {}).get("content")) or []
    report.check(
        "one logprob entry per generated token",
        len(content) == body["usage"]["completion_tokens"],
        (len(content), body["usage"]),
    )
    report.check(
        "two alternatives per token",
        all(len(entry.get("top_logprobs", [])) == 2 for entry in content),
        [len(e.get("top_logprobs", [])) for e in content],
    )
    report.check(
        "log probabilities are <= 0",
        all(entry.get("logprob", 1) <= 0 for entry in content),
        [e.get("logprob") for e in content],
    )

    report.case = "chat.ignore_eos"
    body = client.chat.completions.with_raw_response.create(
        **base,
        messages=hello,
        max_completion_tokens=24,
        extra_body={"ignore_eos": True},
    ).http_response.json()
    report.check(
        "runs to max_completion_tokens",
        body["usage"]["completion_tokens"] == 24,
        body["usage"],
    )
    report.check(
        "finish_reason is length",
        body["choices"][0].get("finish_reason") == "length",
        body["choices"][0].get("finish_reason"),
    )

    report.case = "chat.tools"
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Current weather for a city",
                "parameters": WEATHER_ARGS,
            },
        }
    ]
    question = [{"role": "user", "content": WEATHER_QUESTION}]
    body = client.chat.completions.with_raw_response.create(
        **base, messages=question, tools=tools, max_completion_tokens=128
    ).http_response.json()
    valid("CreateChatCompletionResponse", body, "body")
    choice = body["choices"][0]
    report.check(
        "finish_reason is tool_calls",
        choice.get("finish_reason") == "tool_calls",
        choice.get("finish_reason"),
    )
    calls = choice["message"].get("tool_calls") or []
    if report.check("one tool call", len(calls) == 1, calls):
        call = calls[0]
        args = json.loads(call["function"]["arguments"])
        report.check(
            "arguments are a JSON object with city",
            isinstance(args, dict) and isinstance(args.get("city"), str),
            args,
        )
        followup = client.chat.completions.create(
            **base,
            tools=tools,
            max_completion_tokens=96,
            messages=[
                *question,
                choice["message"],
                {
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "content": "18 C and sunny",
                },
            ],
        )
        report.check(
            "tool result round trip ends with stop",
            followup.choices[0].finish_reason == "stop",
            followup.choices[0].finish_reason,
        )
        report.check(
            "final answer is text",
            bool(followup.choices[0].message.content),
            followup.choices[0].message,
        )

    report.case = "chat.tools.stream"
    with client.chat.completions.with_streaming_response.create(
        **base, messages=question, tools=tools, max_completion_tokens=128, stream=True
    ) as reply:
        chunks = [
            json.loads(d) for _, d in sse_events(reply.iter_lines()) if d != "[DONE]"
        ]
    for index, chunk in enumerate(chunks):
        valid("CreateChatCompletionStreamResponse", chunk, f"chunk {index}")
    deltas = [
        d
        for chunk in chunks
        for c in chunk.get("choices", [])
        for d in (c.get("delta", {}).get("tool_calls") or [])
    ]
    first = deltas[0] if deltas else {}
    report.check(
        "first tool call delta has id, type and name",
        bool(first.get("id"))
        and first.get("type") == "function"
        and bool(first.get("function", {}).get("name")),
        first,
    )
    arguments = "".join(
        d.get("function", {}).get("arguments") or ""
        for d in deltas
        if d.get("index") == 0
    )
    try:
        report.check(
            "streamed arguments parse to an object",
            isinstance(json.loads(arguments), dict),
            arguments,
        )
    except json.JSONDecodeError as error:
        report.check(
            "streamed arguments parse to an object", False, f"{error}: {arguments!r}"
        )
    finishes = [
        c["finish_reason"]
        for chunk in chunks
        for c in chunk.get("choices", [])
        if c.get("finish_reason")
    ]
    report.check("finish_reason is tool_calls", finishes == ["tool_calls"], finishes)

    report.case = "chat.errors"
    for name, kwargs in [
        ("temperature 3", {"temperature": 3, "messages": hello}),
        ("no messages", {"messages": []}),
    ]:
        try:
            client.chat.completions.create(
                model=MODEL_ID, max_completion_tokens=8, **kwargs
            )
            report.check(f"{name} is rejected", False, "accepted")
        except openai.BadRequestError as error:
            valid("ErrorResponse", error.response.json(), f"{name} body")
        except openai.APIStatusError as error:
            report.check(f"{name} answers 400", False, error.status_code)
    try:
        client.chat.completions.create(
            model="no-such-model", messages=hello, max_completion_tokens=8
        )
        report.check("an unknown model is rejected", False, "accepted")
    except openai.NotFoundError as error:
        valid("ErrorResponse", error.response.json(), "unknown model body")
    except openai.APIStatusError as error:
        report.check(
            "an unknown model answers 404",
            False,
            f"{error.status_code} {error.response.text[:200]}",
        )


# ---------------------------------------------------------------------------
# Responses


def responses_cases(client, schemas: OpenAISchemas, report: Report) -> None:
    report.case = "responses"
    import openai

    def valid(name: str, value: object, where: str) -> None:
        errors = schemas.errors(name, value)
        report.check(f"{where} matches {name}", not errors, errors[:5])

    base = {"model": MODEL_ID, "temperature": 0}

    report.case = "responses.plain"
    raw = client.responses.with_raw_response.create(
        **base, input="Say hello in one short sentence.", max_output_tokens=32
    )
    body = raw.http_response.json()
    valid("Response", body, "body")
    report.check(
        "status is completed", body.get("status") == "completed", body.get("status")
    )
    report.check("SDK output_text is non-empty", bool(raw.parse().output_text), "empty")
    usage = body.get("usage") or {}
    report.check(
        "total_tokens = input_tokens + output_tokens",
        usage.get("total_tokens")
        == usage.get("input_tokens", 0) + usage.get("output_tokens", 0),
        usage,
    )

    report.case = "responses.stream"
    with client.responses.with_streaming_response.create(
        **base,
        input="Say hello in one short sentence.",
        max_output_tokens=32,
        stream=True,
    ) as reply:
        events = sse_events(reply.iter_lines())
    payloads = [json.loads(data) for _, data in events]
    for index, ((name, _), event) in enumerate(zip(events, payloads)):
        kind = event.get("type")
        schema = schemas.response_events.get(kind)
        if report.check(
            f"event {index} type {kind} is in the spec", schema is not None, kind
        ):
            valid(schema, event, f"event {index} ({kind})")
        if name is not None:
            report.check(
                f"event {index} name matches its type", name == kind, (name, kind)
            )
    numbers = [event.get("sequence_number") for event in payloads]
    report.check(
        "sequence_number counts up from 0",
        numbers == list(range(len(numbers))),
        numbers,
    )
    kinds = [event.get("type") for event in payloads]
    report.check(
        "first event is response.created", kinds[:1] == ["response.created"], kinds[:2]
    )
    report.check(
        "last event is response.completed",
        kinds[-1:] == ["response.completed"],
        kinds[-2:],
    )
    deltas = "".join(
        e.get("delta", "")
        for e in payloads
        if e.get("type") == "response.output_text.delta"
    )
    done = [
        e.get("text") for e in payloads if e.get("type") == "response.output_text.done"
    ]
    report.check("deltas join to output_text.done", done == [deltas], (done, deltas))
    sdk_text = "".join(
        event.delta
        for event in client.responses.create(
            **base,
            input="Say hello in one short sentence.",
            max_output_tokens=32,
            stream=True,
        )
        if event.type == "response.output_text.delta"
    )
    report.check(
        "SDK stream yields the same text",
        sdk_text == deltas and deltas,
        (sdk_text, deltas),
    )

    report.case = "responses.max_output_tokens"
    body = client.responses.with_raw_response.create(
        **base, input=COUNTING, max_output_tokens=4
    ).http_response.json()
    report.check(
        "status is incomplete", body.get("status") == "incomplete", body.get("status")
    )
    report.check(
        "incomplete_details.reason is max_output_tokens",
        (body.get("incomplete_details") or {}).get("reason") == "max_output_tokens",
        body.get("incomplete_details"),
    )

    report.case = "responses.logprobs"
    body = client.responses.with_raw_response.create(
        **base,
        input=COUNTING,
        max_output_tokens=8,
        top_logprobs=2,
        include=["message.output_text.logprobs"],
    ).http_response.json()
    valid("Response", body, "body")
    parts = [
        part
        for item in body.get("output", [])
        if item.get("type") == "message"
        for part in item.get("content", [])
    ]
    logprobs = [entry for part in parts for entry in (part.get("logprobs") or [])]
    report.check("logprobs are returned", bool(logprobs), parts)
    report.check(
        "two alternatives per token",
        all(len(e.get("top_logprobs", [])) == 2 for e in logprobs),
        [len(e.get("top_logprobs", [])) for e in logprobs],
    )

    report.case = "responses.ignore_eos"
    body = client.responses.with_raw_response.create(
        **base, input="Say hi.", max_output_tokens=24, extra_body={"ignore_eos": True}
    ).http_response.json()
    report.check(
        "runs to max_output_tokens",
        (body.get("usage") or {}).get("output_tokens") == 24,
        body.get("usage"),
    )

    report.case = "responses.tools"
    tools = [
        {
            "type": "function",
            "name": "get_weather",
            "description": "Current weather for a city",
            "parameters": WEATHER_ARGS,
        }
    ]
    body = client.responses.with_raw_response.create(
        **base, input=WEATHER_QUESTION, tools=tools, max_output_tokens=128
    ).http_response.json()
    valid("Response", body, "body")
    calls = [
        item for item in body.get("output", []) if item.get("type") == "function_call"
    ]
    if report.check("one function_call item", len(calls) == 1, body.get("output")):
        call = calls[0]
        report.check(
            "arguments parse to an object with city",
            isinstance(json.loads(call["arguments"]).get("city"), str),
            call["arguments"],
        )
        followup = client.responses.create(
            **base,
            tools=tools,
            max_output_tokens=96,
            input=[
                {"role": "user", "content": WEATHER_QUESTION},
                {
                    "type": "function_call",
                    "call_id": call["call_id"],
                    "name": call["name"],
                    "arguments": call["arguments"],
                },
                {
                    "type": "function_call_output",
                    "call_id": call["call_id"],
                    "output": "18 C and sunny",
                },
            ],
        )
        report.check(
            "tool result round trip completes with text",
            followup.status == "completed" and bool(followup.output_text),
            (followup.status, followup.output_text),
        )

    report.case = "responses.errors"
    try:
        client.responses.create(model=MODEL_ID, input="hi", temperature=3)
        report.check("temperature 3 is rejected", False, "accepted")
    except openai.BadRequestError as error:
        valid("ErrorResponse", error.response.json(), "temperature 3 body")
    except openai.APIStatusError as error:
        report.check("temperature 3 answers 400", False, error.status_code)


# ---------------------------------------------------------------------------
# Messages


def messages_cases(client, base_url: str, report: Report) -> None:
    report.case = "messages"
    import anthropic
    import httpx

    def message_shape(body: dict, where: str) -> None:
        report.present(body, ANTHROPIC_MESSAGE_CORE, where)
        report.present(body, ANTHROPIC_MESSAGE[8:], where, level="note")
        report.check(
            f"{where} type is message", body.get("type") == "message", body.get("type")
        )
        report.check(
            f"{where} role is assistant",
            body.get("role") == "assistant",
            body.get("role"),
        )
        usage = body.get("usage") or {}
        report.present(usage, ANTHROPIC_USAGE_CORE, f"{where} usage")
        report.present(usage, ANTHROPIC_USAGE[2:], f"{where} usage", level="note")
        for index, block in enumerate(body.get("content", [])):
            if block.get("type") == "text":
                report.present(block, ("type", "text"), f"{where} text block {index}")
                report.present(
                    block, ("citations",), f"{where} text block {index}", level="note"
                )
            elif block.get("type") == "tool_use":
                report.present(
                    block,
                    ("type", "id", "name", "input"),
                    f"{where} tool_use block {index}",
                )
                report.present(
                    block, ("caller",), f"{where} tool_use block {index}", level="note"
                )

    def error_shape(status: int, body: dict, where: str, kind: str) -> None:
        # https://platform.claude.com/docs/en/api/errors: "a top-level error
        # object that always includes a type and message".
        report.check(
            f"{where} status",
            status == {"invalid_request_error": 400, "not_found_error": 404}[kind],
            status,
        )
        report.check(f"{where} type is error", body.get("type") == "error", body)
        error = body.get("error") or {}
        report.check(f"{where} error.type is {kind}", error.get("type") == kind, error)
        report.check(
            f"{where} error.message is text",
            isinstance(error.get("message"), str),
            error,
        )
        report.present(body, ("request_id",), where, level="note")

    # anthropic 1.11.0 no longer sends temperature (the current API dropped
    # sampling parameters); greedy decoding here goes in the body directly.
    base = {"model": MODEL_ID, "extra_body": {"temperature": 0.0}}
    hello = [{"role": "user", "content": "Say hello in one short sentence."}]

    report.case = "messages.plain"
    raw = client.messages.with_raw_response.create(
        **base, messages=hello, max_tokens=32
    )
    body = raw.http_response.json()
    message_shape(body, "body")
    report.check(
        "stop_reason is end_turn",
        body.get("stop_reason") == "end_turn",
        body.get("stop_reason"),
    )
    report.check(
        "stop_sequence is null",
        body.get("stop_sequence", "absent") is None,
        body.get("stop_sequence", "absent"),
    )
    report.check(
        "SDK parses the reply",
        raw.parse().content[0].type == "text",
        raw.parse().content,
    )

    report.case = "messages.stream"
    with httpx.stream(
        "POST",
        f"{base_url}/v1/messages",
        timeout=120,
        headers={"anthropic-version": "2023-06-01", "x-api-key": "unused"},
        json={
            "model": MODEL_ID,
            "temperature": 0.0,
            "messages": hello,
            "max_tokens": 32,
            "stream": True,
        },
    ) as reply:
        events = sse_events(reply.iter_lines())
    payloads = [json.loads(data) for _, data in events]
    for index, ((name, _), event) in enumerate(zip(events, payloads)):
        report.check(
            f"event {index} name matches its type",
            name == event.get("type"),
            (name, event.get("type")),
        )
    kinds = [event.get("type") for event in payloads if event.get("type") != "ping"]
    # Streaming docs: message_start, blocks (start, deltas, stop), one or
    # more message_delta, message_stop.
    pattern = r"message_start(content_block_start(content_block_delta)+content_block_stop)*(message_delta)+message_stop"
    report.check(
        "events follow the documented order",
        re.fullmatch(pattern, "".join(kinds)) is not None,
        kinds,
    )
    start = payloads[0].get("message", {}) if payloads else {}
    message_shape(start, "message_start.message")
    report.check(
        "message_start content is empty",
        start.get("content") == [],
        start.get("content"),
    )
    report.check(
        "message_start usage reports input_tokens",
        (start.get("usage") or {}).get("input_tokens", 0) > 0,
        start.get("usage"),
    )
    deltas = [event for event in payloads if event.get("type") == "message_delta"]
    report.check(
        "message_delta carries usage.output_tokens",
        all(
            isinstance((d.get("usage") or {}).get("output_tokens"), int) for d in deltas
        ),
        deltas,
    )
    report.check(
        "message_delta stop_reason is end_turn",
        [d.get("delta", {}).get("stop_reason") for d in deltas][-1:] == ["end_turn"],
        deltas,
    )
    streamed = "".join(
        e.get("delta", {}).get("text", "")
        for e in payloads
        if e.get("type") == "content_block_delta"
    )
    with client.messages.stream(**base, messages=hello, max_tokens=32) as stream:
        final = stream.get_final_message()
    report.check(
        "SDK final message has the same text",
        final.content and final.content[0].text == streamed,
        (final.content, streamed),
    )

    report.case = "messages.max_tokens"
    body = client.messages.with_raw_response.create(
        **base, messages=[{"role": "user", "content": COUNTING}], max_tokens=4
    ).http_response.json()
    report.check(
        "stop_reason is max_tokens",
        body.get("stop_reason") == "max_tokens",
        body.get("stop_reason"),
    )
    report.check(
        "output_tokens <= 4",
        (body.get("usage") or {}).get("output_tokens", 99) <= 4,
        body.get("usage"),
    )

    report.case = "messages.stop_sequences"
    try:
        body = client.messages.with_raw_response.create(
            **base,
            messages=[{"role": "user", "content": COUNTING}],
            max_tokens=64,
            stop_sequences=[","],
        ).http_response.json()
        report.check(
            "stop_reason is stop_sequence",
            body.get("stop_reason") == "stop_sequence",
            body.get("stop_reason"),
        )
        report.check(
            "stop_sequence names the match",
            body.get("stop_sequence") == ",",
            body.get("stop_sequence"),
        )
        text = "".join(b.get("text", "") for b in body.get("content", []))
        report.check("text excludes the stop sequence", "," not in text, text)
    except anthropic.APIStatusError as error:
        report.check(
            "stop_sequences is accepted",
            False,
            f"{error.status_code} {error.response.text[:200]}",
        )

    report.case = "messages.ignore_eos"
    body = client.messages.with_raw_response.create(
        model=MODEL_ID,
        messages=hello,
        max_tokens=24,
        extra_body={"temperature": 0.0, "ignore_eos": True},
    ).http_response.json()
    report.check(
        "runs to max_tokens",
        (body.get("usage") or {}).get("output_tokens") == 24,
        body.get("usage"),
    )
    report.check(
        "stop_reason is max_tokens",
        body.get("stop_reason") == "max_tokens",
        body.get("stop_reason"),
    )

    report.case = "messages.tools"
    tools = [
        {
            "name": "get_weather",
            "description": "Current weather for a city",
            "input_schema": WEATHER_ARGS,
        }
    ]
    question = [{"role": "user", "content": WEATHER_QUESTION}]
    body = client.messages.with_raw_response.create(
        **base, messages=question, tools=tools, max_tokens=128
    ).http_response.json()
    message_shape(body, "body")
    report.check(
        "stop_reason is tool_use",
        body.get("stop_reason") == "tool_use",
        body.get("stop_reason"),
    )
    uses = [b for b in body.get("content", []) if b.get("type") == "tool_use"]
    if report.check("one tool_use block", len(uses) == 1, body.get("content")):
        use = uses[0]
        report.check(
            "input is an object with city",
            isinstance(use.get("input", {}).get("city"), str),
            use.get("input"),
        )
        followup = client.messages.create(
            **base,
            tools=tools,
            max_tokens=96,
            messages=[
                *question,
                {"role": "assistant", "content": body["content"]},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": use["id"],
                            "content": "18 C and sunny",
                        }
                    ],
                },
            ],
        )
        report.check(
            "tool result round trip ends with end_turn",
            followup.stop_reason == "end_turn",
            followup.stop_reason,
        )
        report.check(
            "final answer is text",
            any(b.type == "text" and b.text for b in followup.content),
            followup.content,
        )

    report.case = "messages.errors"
    try:
        client.messages.create(**base, messages=[], max_tokens=8)
        report.check("empty messages are rejected", False, "accepted")
    except anthropic.APIStatusError as error:
        error_shape(
            error.status_code,
            error.response.json(),
            "empty messages",
            "invalid_request_error",
        )
    # max_tokens is required; the SDK would refuse to send this, so post it raw.
    reply = httpx.post(
        f"{base_url}/v1/messages",
        timeout=60,
        headers={"anthropic-version": "2023-06-01", "x-api-key": "unused"},
        json={"model": MODEL_ID, "messages": hello},
    )
    error_shape(
        reply.status_code, reply.json(), "missing max_tokens", "invalid_request_error"
    )
    try:
        client.messages.create(model="no-such-model", messages=hello, max_tokens=8)
        report.check("an unknown model is rejected", False, "accepted")
    except anthropic.APIStatusError as error:
        error_shape(
            error.status_code, error.response.json(), "unknown model", "not_found_error"
        )


# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--mx", type=Path, required=True, help="the mx binary to serve with"
    )
    parser.add_argument(
        "--model", type=Path, default=os.environ.get("METALLIX_QWEN_MODEL")
    )
    parser.add_argument("--max-footprint-gib", type=float, default=8.0)
    parser.add_argument(
        "--spec-cache",
        type=Path,
        default=Path.home() / ".cache" / "metallix-conformance",
    )
    parser.add_argument("--log", type=Path, default=Path("sdk-conformance-server.log"))
    parser.add_argument(
        "--only", choices=["chat", "responses", "messages"], action="append"
    )
    args = parser.parse_args()
    if args.model is None:
        parser.error("--model or METALLIX_QWEN_MODEL is required")

    import anthropic
    import openai

    schemas = OpenAISchemas(openai_spec(args.spec_cache))
    port = free_port()
    base = f"http://127.0.0.1:{port}"
    argv = [
        str(args.mx), "serve", "--model", str(args.model), "--model-id", MODEL_ID,
        "--listen", f"127.0.0.1:{port}", "--context-tokens", "4096", "--kv-budget-mib", "1024",
    ]  # fmt: skip
    print("+", " ".join(argv))
    report = Report()
    with args.log.open("wb") as log:
        server = subprocess.Popen(
            argv, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        )
        guard = FootprintGuard(server.pid, int(args.max_footprint_gib * 2**30))
        try:
            with guard:
                wait_healthy(base, server, timeout=180)
                openai_client = openai.OpenAI(
                    base_url=f"{base}/v1", api_key="unused", max_retries=0, timeout=300
                )
                anthropic_client = anthropic.Anthropic(
                    base_url=base, api_key="unused", max_retries=0, timeout=300
                )
                suites = {
                    "chat": lambda: chat_cases(openai_client, schemas, report),
                    "responses": lambda: responses_cases(
                        openai_client, schemas, report
                    ),
                    "messages": lambda: messages_cases(anthropic_client, base, report),
                }
                for name in args.only or list(suites):
                    try:
                        suites[name]()
                    # A crashed suite is recorded as a failure; the rest still run.
                    except Exception as error:  # noqa: BLE001
                        report.check(
                            "case completes", False, f"{type(error).__name__}: {error}"
                        )
        finally:
            if server.poll() is None:
                os.killpg(server.pid, signal.SIGTERM)
                try:
                    server.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(server.pid, signal.SIGKILL)
                    server.wait()
    print(
        f"server footprint peak {guard.peak / 2**30:.2f} GiB (cap {args.max_footprint_gib:g} GiB)"
    )
    if guard.aborted:
        report.check("server footprint stays under the cap", False, guard.aborted)
    rows = report.failures()
    passed = sum(1 for row in report.checks if row[2] == "pass")
    print(
        f"{passed} passed, {sum(r[2] == 'fail' for r in rows)} failed, {sum(r[2] == 'note' for r in rows)} notes"
    )
    for case, name, status, detail in rows:
        print(f"{status}\t{case}\t{name}\t{detail[:400]}")
    return 1 if any(row[2] == "fail" for row in rows) else 0


if __name__ == "__main__":
    sys.exit(main())
