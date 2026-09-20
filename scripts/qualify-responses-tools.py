#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Dry-run-first Responses function-call replay qualification; never starts a server."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import http.client
import ipaddress
import json
import os
import signal
import time
from pathlib import Path
from urllib.parse import urlparse

MAX_REQUEST_BYTES = 1_048_576
MAX_RESPONSE_BYTES = 1_048_576
TOOL = {
    "type": "function",
    "name": "read_fact",
    "description": "Retrieve the current qualification value.",
    "parameters": {
        "type": "object",
        "properties": {"key": {"type": "string", "enum": ["qualification_value"]}},
        "required": ["key"],
        "additionalProperties": False,
    },
}
PROMPT = (
    "Call read_fact with key qualification_value. After receiving its result, reply "
    "exactly QUALIFIED:<the value returned> and no other text."
)


class ProtocolError(RuntimeError):
    pass


class ModelError(RuntimeError):
    pass


class TransportError(RuntimeError):
    pass


def positive(value: str) -> int:
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def loopback_base_url(value: str) -> tuple[str, str, int, str]:
    parsed = urlparse(value)
    if (
        parsed.scheme != "http"
        or not parsed.hostname
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
    ):
        raise argparse.ArgumentTypeError(
            "URL must be a credential-free loopback http URL"
        )
    try:
        is_loopback = ipaddress.ip_address(parsed.hostname).is_loopback
        port = parsed.port
    except ValueError as error:
        raise argparse.ArgumentTypeError("URL host or port is invalid") from error
    if not is_loopback or port is None:
        raise argparse.ArgumentTypeError(
            "URL must use a numeric loopback host and port"
        )
    path = parsed.path.rstrip("/") or "/v1"
    if path != "/v1":
        raise argparse.ArgumentTypeError("URL path must be empty or /v1")
    return value.rstrip("/"), parsed.hostname, port, path


def sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


@contextlib.contextmanager
def wall_deadline(seconds: int):
    """Interrupt all synchronous HTTP phases, including trickled response headers."""
    previous_handler = signal.getsignal(signal.SIGALRM)

    def expired(_signum: int, _frame: object) -> None:
        raise TransportError(f"request exceeded {seconds}s wall deadline")

    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)


def valid_usage(value: object) -> bool:
    if not isinstance(value, dict):
        return False
    input_tokens = value.get("input_tokens")
    output_tokens = value.get("output_tokens")
    total_tokens = value.get("total_tokens")
    return (
        type(input_tokens) is int
        and input_tokens > 0
        and type(output_tokens) is int
        and output_tokens > 0
        and type(total_tokens) is int
        and total_tokens == input_tokens + output_tokens
    )


def validate_response(response: object) -> dict:
    if not isinstance(response, dict):
        raise ProtocolError("response is not an object")
    if (
        not isinstance(response.get("id"), str)
        or not response["id"]
        or response.get("object") != "response"
    ):
        raise ProtocolError("response identity is invalid")
    if response.get("status") != "completed":
        raise ModelError("response did not complete")
    if not valid_usage(response.get("usage")):
        raise ProtocolError("response usage is invalid")
    if not isinstance(response.get("output"), list):
        raise ProtocolError("response output is not a list")
    return response


def parse_sse(raw: str) -> tuple[dict, list[dict]]:
    events: list[dict] = []
    event_name: str | None = None
    data_lines: list[str] = []

    def dispatch() -> None:
        nonlocal event_name
        if event_name is None and not data_lines:
            return
        if event_name is None or len(data_lines) != 1:
            raise ProtocolError("SSE frame lacks one event and one data field")
        try:
            event = json.loads(data_lines[0])
        except json.JSONDecodeError as error:
            raise ProtocolError("malformed SSE JSON") from error
        if not isinstance(event, dict) or event.get("type") != event_name:
            raise ProtocolError("SSE event field does not match JSON type")
        events.append(event)
        event_name = None
        data_lines.clear()

    for line in raw.splitlines():
        if line.startswith(":"):
            continue
        if not line:
            dispatch()
        elif line.startswith("event:"):
            if event_name is not None:
                raise ProtocolError("SSE frame has repeated event fields")
            event_name = line[6:].lstrip()
        elif line.startswith("data:"):
            data_lines.append(line[5:].lstrip())
        else:
            raise ProtocolError("unsupported SSE frame field")
    if event_name is not None or data_lines:
        raise ProtocolError("incomplete SSE event")
    if not events:
        raise ProtocolError("SSE stream has no events")
    if [event.get("sequence_number") for event in events] != list(range(len(events))):
        raise ProtocolError("SSE sequence numbers are not contiguous")
    terminal = [
        event
        for event in events
        if event.get("type")
        in {"response.completed", "response.incomplete", "response.failed"}
    ]
    if len(terminal) != 1 or events[-1] is not terminal[0]:
        raise ProtocolError("SSE stream lacks one terminal final event")
    if terminal[0].get("type") != "response.completed":
        raise ModelError("SSE response did not complete")
    response = terminal[0].get("response")
    response = validate_response(response)
    created = events[0]
    created_response = created.get("response")
    if (
        created.get("type") != "response.created"
        or not isinstance(created_response, dict)
        or created_response.get("id") != response["id"]
        or created_response.get("object") != "response"
        or created_response.get("status") != "in_progress"
    ):
        raise ProtocolError("SSE response.created identity does not match terminal")
    return response, events


def validate_stream_events(response: dict, events: list[dict]) -> None:
    output = response["output"]
    output_by_id = {
        item.get("id"): (index, item)
        for index, item in enumerate(output)
        if isinstance(item, dict)
    }
    for event in events:
        event_type = event.get("type")
        if event_type not in {
            "response.created",
            "response.output_item.added",
            "response.function_call_arguments.delta",
            "response.function_call_arguments.done",
            "response.output_text.delta",
            "response.output_text.done",
            "response.content_part.added",
            "response.content_part.done",
            "response.output_item.done",
            "response.completed",
        }:
            raise ProtocolError("unsupported SSE event type")
        if event_type in {
            "response.function_call_arguments.delta",
            "response.function_call_arguments.done",
            "response.output_text.delta",
            "response.output_text.done",
        }:
            item_id = event.get("item_id")
            output_index = event.get("output_index")
            matched = output_by_id.get(item_id)
            if (
                not isinstance(item_id, str)
                or not isinstance(output_index, int)
                or matched is None
                or matched[0] != output_index
            ):
                raise ProtocolError("SSE event refers to an unknown output item")
            item_type = matched[1].get("type")
            if (
                event_type.startswith("response.function_call")
                and item_type != "function_call"
            ) or (
                event_type.startswith("response.output_text") and item_type != "message"
            ):
                raise ProtocolError("SSE event type does not match output item")
    done = [
        event.get("item")
        for event in events
        if event.get("type") == "response.output_item.done"
    ]
    if done != output:
        raise ProtocolError("SSE output_item.done events do not match response output")
    for index, item in enumerate(output):
        if not isinstance(item, dict) or not isinstance(item.get("id"), str):
            raise ProtocolError("response output item lacks an ID")
        item_id = item["id"]

        def is_added(
            event: dict, expected_id: str = item_id, expected_index: int = index
        ) -> bool:
            candidate = event.get("item")
            event_item_id = event.get("item_id")
            if event_item_id is None and isinstance(candidate, dict):
                event_item_id = candidate.get("id")
            return (
                event.get("type") == "response.output_item.added"
                and event_item_id == expected_id
                and event.get("output_index") == expected_index
            )

        added = [event for event in events if is_added(event)]
        if len(added) != 1:
            raise ProtocolError("SSE output item lacks one matching added event")
        added_index = events.index(added[0])
        item_done_indexes = [
            event_index
            for event_index, event in enumerate(events)
            if event.get("type") == "response.output_item.done"
            and event.get("output_index") == index
            and event.get("item") == item
        ]
        if len(item_done_indexes) != 1:
            raise ProtocolError("SSE output item lacks one matching done event")
        item_done_index = item_done_indexes[0]
        if item.get("type") == "function_call":
            delta_events = [
                (event_index, event)
                for event_index, event in enumerate(events)
                if event.get("type") == "response.function_call_arguments.delta"
                and event.get("item_id") == item_id
                and event.get("output_index") == index
            ]
            deltas = [event.get("delta") for _, event in delta_events]
            completed_events = [
                (event_index, event)
                for event_index, event in enumerate(events)
                if event.get("type") == "response.function_call_arguments.done"
                and event.get("item_id") == item_id
                and event.get("output_index") == index
            ]
            completed = [event.get("arguments") for _, event in completed_events]
            if (
                not all(isinstance(delta, str) for delta in deltas)
                or "".join(deltas) != item.get("arguments")
                or completed != [item.get("arguments")]
                or not delta_events
                or not all(added_index < event_index for event_index, _ in delta_events)
                or not all(
                    event_index < completed_events[0][0]
                    for event_index, _ in delta_events
                )
                or not (completed_events[0][0] < item_done_index)
            ):
                raise ProtocolError("SSE function-call arguments do not assemble")
        elif item.get("type") == "message":
            content = item.get("content")
            if not isinstance(content, list) or not all(
                isinstance(part, dict) and isinstance(part.get("text"), str)
                for part in content
            ):
                raise ProtocolError("message output content is invalid")
            delta_events = [
                (event_index, event)
                for event_index, event in enumerate(events)
                if event.get("type") == "response.output_text.delta"
                and event.get("item_id") == item_id
                and event.get("output_index") == index
            ]
            deltas = [event.get("delta") for _, event in delta_events]
            text_done_events = [
                (event_index, event)
                for event_index, event in enumerate(events)
                if event.get("type") == "response.output_text.done"
                and event.get("item_id") == item_id
                and event.get("output_index") == index
            ]
            if (
                not all(isinstance(delta, str) for delta in deltas)
                or "".join(deltas) != "".join(part["text"] for part in content)
                or len(text_done_events) != 1
                or not all(
                    added_index < event_index < text_done_events[0][0]
                    for event_index, _ in delta_events
                )
                or not (text_done_events[0][0] < item_done_index)
            ):
                raise ProtocolError("SSE text deltas do not assemble")
        else:
            raise ProtocolError("unsupported response output item")


def call_from_response(response: dict) -> dict:
    output = response["output"]
    if len(output) != 1 or not isinstance(output[0], dict):
        raise ModelError("expected exactly one function call")
    call = output[0]
    if (
        call.get("type") != "function_call"
        or call.get("name") != TOOL["name"]
        or not isinstance(call.get("call_id"), str)
        or not call["call_id"]
        or not isinstance(call.get("arguments"), str)
        or call.get("status") != "completed"
    ):
        raise ModelError("model did not return the required function call")
    try:
        arguments = json.loads(call["arguments"])
    except json.JSONDecodeError as error:
        raise ModelError("model returned malformed function arguments") from error
    if arguments != {"key": "qualification_value"}:
        raise ModelError("model returned wrong function arguments")
    return call


def answer_text(response: dict) -> str:
    if not response["output"] or any(
        not isinstance(item, dict)
        or item.get("type") != "message"
        or item.get("status") != "completed"
        for item in response["output"]
    ):
        raise ModelError("answer response contains a function call or no message")
    try:
        return "".join(
            part["text"] for item in response["output"] for part in item["content"]
        )
    except (KeyError, TypeError) as error:
        raise ProtocolError("answer response message content is invalid") from error


def replay_input(call: dict, value: str) -> list[dict]:
    result = {
        "type": "function_call_output",
        "call_id": call["call_id"],
        "output": json.dumps({"qualification_value": value}),
    }
    return [
        {"role": "user", "content": PROMPT},
        call,
        result,
    ]


def validate_replay_input(items: object, call_id: str) -> None:
    if not isinstance(items, list) or len(items) != 3:
        raise ProtocolError("replay input shape is invalid")
    call, result = items[1:]
    if (
        not isinstance(call, dict)
        or not isinstance(result, dict)
        or call.get("type") != "function_call"
        or result.get("type") != "function_call_output"
        or call.get("call_id") != call_id
        or result.get("call_id") != call_id
    ):
        raise ProtocolError("replay output does not match the function call")


def request(
    host: str,
    port: int,
    path: str,
    payload: dict,
    timeout: int,
    output: Path,
    label: str,
) -> tuple[dict, list[dict] | None]:
    body = json.dumps(payload, separators=(",", ":")).encode()
    if len(body) > MAX_REQUEST_BYTES:
        raise ProtocolError("qualification request exceeds server body bound")
    (output / f"{label}.request.json").write_bytes(body + b"\n")
    connection = http.client.HTTPConnection(host, port, timeout=timeout)
    raw = bytearray()
    try:
        with wall_deadline(timeout):
            connection.request(
                "POST",
                f"{path}/responses",
                body,
                {"Content-Type": "application/json"},
            )
            response = connection.getresponse()
            if response.status != 200:
                raise ProtocolError(f"unexpected HTTP status {response.status}")
            while True:
                chunk = response.read1(min(65_536, MAX_RESPONSE_BYTES + 1 - len(raw)))
                if not chunk:
                    break
                raw.extend(chunk)
                if len(raw) > MAX_RESPONSE_BYTES:
                    raise ProtocolError("response exceeds qualification bound")
    except (TimeoutError, OSError, http.client.HTTPException) as error:
        raise TransportError(str(error)) from error
    finally:
        (output / f"{label}.response.raw").write_bytes(raw)
        connection.close()
    try:
        text = bytes(raw).decode("utf-8")
    except UnicodeDecodeError as error:
        raise ProtocolError("response is not UTF-8") from error
    if payload["stream"]:
        response, events = parse_sse(text)
        validate_stream_events(response, events)
        return response, events
    try:
        return validate_response(json.loads(text)), None
    except json.JSONDecodeError as error:
        raise ProtocolError("malformed JSON response") from error


def classify(error: ModelError | ProtocolError | TransportError) -> str:
    if isinstance(error, ModelError):
        return "model"
    if isinstance(error, ProtocolError):
        return "protocol"
    return "transport"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--url", type=loopback_base_url, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=positive, default=3)
    parser.add_argument("--timeout-seconds", type=positive, default=90)
    args = parser.parse_args()
    origin, host, port, base_path = args.url
    plan = {
        "schema_version": 1,
        "status": "dry_run",
        "scope": "Responses function-call and held-out result replay",
        "url": origin,
        "model_id": args.model_id,
        "repeats": args.repeats,
        "timeout_seconds": args.timeout_seconds,
        "script_sha256": sha256(Path(__file__).read_bytes()),
        "modes": ["json", "sse"],
        "server_action": "none",
    }
    if not args.run:
        print(json.dumps(plan, indent=2))
        return 0
    if args.output.exists() and (
        not args.output.is_dir() or any(args.output.iterdir())
    ):
        parser.error("--output must name a new or empty directory")
    args.output.mkdir(parents=True, exist_ok=True)
    receipt: dict = {**plan, "status": "running", "trials": []}
    receipt_path = args.output / "receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    try:
        for stream in (False, True):
            for repeat in range(1, args.repeats + 1):
                label = f"{'sse' if stream else 'json'}-{repeat}"
                started = time.monotonic()
                row: dict = {
                    "mode": "sse" if stream else "json",
                    "repeat": repeat,
                }
                try:
                    first_payload = {
                        "model": args.model_id,
                        "input": PROMPT,
                        "tools": [TOOL],
                        "stream": stream,
                        "max_output_tokens": 128,
                        "temperature": 0,
                        "store": False,
                    }
                    first, first_events = request(
                        host,
                        port,
                        base_path,
                        first_payload,
                        args.timeout_seconds,
                        args.output,
                        f"{label}-call",
                    )
                    call = call_from_response(first)
                    value = f"FACT-{os.urandom(16).hex()}"
                    replay = replay_input(call, value)
                    validate_replay_input(replay, call["call_id"])
                    second_payload = {**first_payload, "input": replay}
                    second, second_events = request(
                        host,
                        port,
                        base_path,
                        second_payload,
                        args.timeout_seconds,
                        args.output,
                        f"{label}-answer",
                    )
                    if answer_text(second) != f"QUALIFIED:{value}":
                        raise ModelError(
                            "model answer does not exactly reproduce held-out value"
                        )
                    row.update(
                        {
                            "passed": True,
                            "wall_ms": (time.monotonic() - started) * 1000,
                            "call_usage": first["usage"],
                            "answer_usage": second["usage"],
                            "call_event_count": None
                            if first_events is None
                            else len(first_events),
                            "answer_event_count": None
                            if second_events is None
                            else len(second_events),
                            "request_sha256": {
                                "call": sha256(
                                    (
                                        args.output / f"{label}-call.request.json"
                                    ).read_bytes()
                                ),
                                "answer": sha256(
                                    (
                                        args.output / f"{label}-answer.request.json"
                                    ).read_bytes()
                                ),
                            },
                        }
                    )
                except (ModelError, ProtocolError, TransportError) as error:
                    row.update(
                        {
                            "passed": False,
                            "wall_ms": (time.monotonic() - started) * 1000,
                            "failure": {
                                "class": classify(error),
                                "error": str(error),
                            },
                        }
                    )
                receipt["trials"].append(row)
                receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
                print(f"{label}: {'pass' if row['passed'] else 'FAIL'}", flush=True)
    except KeyboardInterrupt:
        receipt["status"] = "interrupted"
        receipt["error"] = "interrupted by operator"
        receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
        return 130
    receipt["status"] = (
        "passed" if all(row["passed"] for row in receipt["trials"]) else "failed"
    )
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    return 0 if receipt["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
