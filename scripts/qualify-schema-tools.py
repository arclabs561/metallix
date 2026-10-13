#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Paired schema-with-tools qualification against an existing loopback mx server.

Dry run by default. Sixteen paired samples hold prompts, seeds and sampling
fixed, changing only text.format. This measures output validity, call-choice
preservation and completion. It does not prove constraint-span coverage or
server/checkpoint identity; retain the server launch and tracing receipts too.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import hashlib
import http.client
import importlib.util
import ipaddress
import json
import signal
from pathlib import Path
from urllib.parse import urlparse

SPEC = importlib.util.spec_from_file_location(
    "responses_qualification", Path(__file__).with_name("qualify-responses-tools.py")
)
assert SPEC and SPEC.loader
responses = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(responses)
ProtocolError = responses.ProtocolError
TransportError = responses.TransportError
MAX_RESPONSE_BYTES = 1_048_576
SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "string"}},
    "required": ["answer"],
    "additionalProperties": False,
}
TOOL = {
    "type": "function",
    "name": "read_fact",
    "strict": False,
    "description": "Retrieve a current fact that is not present in the conversation.",
    "parameters": {
        "type": "object",
        "properties": {"key": {"type": "string"}},
        "required": ["key"],
        "additionalProperties": False,
    },
}
PROMPTS = [
    "What is the current deployment version? Fetch deployment_version with read_fact.",
    "Look up the current queue depth using read_fact with key queue_depth.",
    "I need the active release owner. Retrieve release_owner with read_fact.",
    "Read the latest build result using read_fact, key build_result.",
    "Find the current service state using read_fact with key service_state.",
    "Retrieve the current incident count using read_fact, key incident_count.",
    "The provided answer is blue. Return that answer without a tool call.",
    "The provided answer is seven. Return that answer without a tool call.",
]


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


def paired_requests(model: str, max_tokens: int) -> list[dict]:
    pairs = []
    for index, prompt in enumerate(PROMPTS):
        for seed in (17, 83):
            baseline = {
                "model": model,
                "input": [
                    {
                        "role": "user",
                        "content": prompt
                        + ' If you answer directly, use JSON {"answer": "your answer"}.',
                    }
                ],
                "tools": [copy.deepcopy(TOOL)],
                "tool_choice": "auto",
                "temperature": 0.6,
                "top_p": 0.95,
                "seed": seed,
                "max_output_tokens": max_tokens,
                "stream": False,
                "chat_template_kwargs": {"enable_thinking": False},
            }
            constrained = copy.deepcopy(baseline)
            constrained["text"] = {
                "format": {
                    "type": "json_schema",
                    "name": "answer",
                    "strict": True,
                    "schema": copy.deepcopy(SCHEMA),
                }
            }
            pairs.append(
                {
                    "id": f"prompt-{index}-seed-{seed}",
                    "baseline": baseline,
                    "constrained": constrained,
                }
            )
    return pairs


def fetch(host, port, path, payload, timeout, output, label):
    body = json.dumps(payload, separators=(",", ":")).encode()
    (output / f"{label}.request.json").write_bytes(body + b"\n")
    connection = http.client.HTTPConnection(host, port, timeout=timeout)
    raw = bytearray()
    try:
        with wall_deadline(timeout):
            connection.request(
                "POST", f"{path}/responses", body, {"Content-Type": "application/json"}
            )
            response = connection.getresponse()
            if response.status != 200:
                raise ProtocolError(f"unexpected HTTP status {response.status}")
            while True:
                chunk = response.read1(min(65536, MAX_RESPONSE_BYTES + 1 - len(raw)))
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
        return responses.strict_json(raw.decode("utf-8"))
    except (ValueError, UnicodeDecodeError) as error:
        raise ProtocolError("response is not strict UTF-8 JSON") from error


def classify_response(value, payload, constrained):
    if not isinstance(value, dict):
        raise ProtocolError("response must be an object")
    responses.validate_expected_model(value, payload["model"])
    if (
        value.get("object") != "response"
        or not isinstance(value.get("id"), str)
        or not value["id"]
        or not responses.valid_usage(value.get("usage"))
    ):
        raise ProtocolError("invalid response identity or usage")
    metadata = value.get("metallix")
    if not isinstance(metadata, dict) or not isinstance(metadata.get("sampling"), dict):
        raise ProtocolError("missing sampling receipt")
    sampling = metadata["sampling"]
    if any(
        sampling.get(key) != payload[key] for key in ("seed", "temperature", "top_p")
    ):
        raise ProtocolError("server sampling does not match paired request")
    if value.get("status") == "incomplete":
        details = value.get("incomplete_details")
        if (
            not isinstance(details, dict)
            or details.get("reason") != "max_output_tokens"
        ):
            raise ProtocolError("unexpected incomplete response reason")
        return "limit"
    if value.get("status") != "completed":
        raise ProtocolError("response failed or has unknown status")
    output = value.get("output")
    if not isinstance(output, list) or not output:
        raise ProtocolError("completed response has no output")
    calls = []
    messages = []
    for item in output:
        if not isinstance(item, dict):
            raise ProtocolError("invalid output item")
        if item.get("type") != "reasoning" and item.get("status") != "completed":
            raise ProtocolError("invalid or unfinished output item")
        kind = item.get("type")
        if kind == "function_call":
            if (
                item.get("name") != TOOL["name"]
                or not item.get("call_id")
                or not isinstance(item.get("arguments"), str)
            ):
                raise ProtocolError("invalid function call")
            try:
                arguments = responses.strict_json(item["arguments"])
            except ValueError as error:
                raise ProtocolError("invalid call arguments") from error
            if not responses.fixture_value_matches(arguments, TOOL["parameters"]):
                raise ProtocolError("call arguments violate fixture schema")
            calls.append(item)
        elif kind == "message":
            messages.append(responses.message_text(item))
        elif kind != "reasoning":
            raise ProtocolError("unexpected output item")
    if calls:
        return "call"
    if len(messages) != 1:
        raise ProtocolError("completed final requires exactly one message")
    if constrained:
        try:
            final = responses.strict_json(messages[0])
        except ValueError as error:
            raise ProtocolError("constrained final is not JSON") from error
        if not responses.fixture_value_matches(final, SCHEMA):
            raise ProtocolError("constrained final violates schema")
    return "final"


def verdict(rows):
    if len(rows) != 16 or len({row["id"] for row in rows}) != 16:
        return {"status": "fail", "reason": "requires sixteen distinct paired samples"}
    baseline_calls = sum(row.get("baseline") == "call" for row in rows)
    failures = [
        row["id"]
        for row in rows
        if row.get("error")
        or row.get("constrained") not in {"call", "final"}
        or (row.get("baseline") == "call" and row.get("constrained") != "call")
    ]
    if failures:
        return {
            "status": "fail",
            "failed_samples": failures,
            "baseline_calls": baseline_calls,
        }
    if baseline_calls < 4:
        return {
            "status": "inconclusive",
            "reason": "baseline call share below 25%",
            "baseline_calls": baseline_calls,
        }
    return {
        "status": "pass",
        "baseline_calls": baseline_calls,
        "scope": "paired validity, call preservation and completion only",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", type=loopback_base_url, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--max-output-tokens", type=positive, default=512)
    parser.add_argument("--timeout", type=positive, default=120)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    pairs = paired_requests(args.model, args.max_output_tokens)
    plan = {
        "pairs": pairs,
        "requests": 32,
        "deadline_per_request_s": args.timeout,
        "script_sha256": sha256(Path(__file__).read_bytes()),
        "helper_sha256": sha256(Path(responses.__file__).read_bytes()),
        "base_url": args.base_url[0],
    }
    if not args.execute:
        print(json.dumps({"status": "dry_run", **plan}, indent=2))
        return 0
    if args.output is None:
        parser.error("--execute requires a new --output directory")
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    _, host, port, path = args.base_url
    rows = []
    for pair in pairs:
        row = {"id": pair["id"]}
        for arm in ("baseline", "constrained"):
            try:
                response = fetch(
                    host,
                    port,
                    path,
                    pair[arm],
                    args.timeout,
                    args.output,
                    f"{pair['id']}-{arm}",
                )
                row[arm] = classify_response(response, pair[arm], arm == "constrained")
            except (ProtocolError, TransportError) as error:
                row["error"] = f"{arm}: {error}"
                break
        rows.append(row)
        result = {"verdict": verdict(rows), "samples": rows}
        (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        if row.get("error"):
            break
    print(json.dumps(result, indent=2))
    return 0 if result["verdict"]["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
