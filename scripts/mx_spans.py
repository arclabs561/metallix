#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Per-request server timings from `mx serve --trace-out` timelines.

`mx serve --trace-out T.json` writes a Chrome/Perfetto timeline for the proxy
(T.json) and one per model child (T.<model-id>.json). The proxy timeline holds
each request's admission-queue span (`proxy.queue`, with `queue.wait_ms`); the
child timeline holds the request span (`http.request`, with usage and MLX
memory) and, inside it, `chat.prefill` and one `chat.decode_step` per token.

This joins them by request id and summarizes a level next to the client's
TTFT and TPOT, so a gap between what the server measured and what the client
saw shows up (time spent outside the spans: queueing in the client, HTTP,
streaming, rendering the template).

Usage:
  scripts/mx_spans.py T.json --model-id qwen3-0.6b
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

# Client and server medians further apart than this ratio are reported.
MISMATCH_RATIO = 1.25


def load_trace(path: Path) -> list[dict]:
    """Events of a Chrome trace; a timeline cut off at exit may lack its `]`."""
    text = path.read_text().rstrip().rstrip(",")
    if not text.endswith("]"):
        text += "]"
    return json.loads(text)


def value(text):
    """A span field as Python: tracing records numbers and quoted strings as text."""
    if not isinstance(text, str):
        return text
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def intervals(events: list[dict]) -> list[dict]:
    """Closed spans from B/E pairs, matched per (pid, tid) as a stack, with the
    begin and end fields merged."""
    stacks: dict[tuple, list[dict]] = {}
    out = []
    for event in events:
        key = (event.get("pid"), event.get("tid"))
        if event.get("ph") == "B":
            stacks.setdefault(key, []).append(event)
        elif event.get("ph") == "E":
            stack = stacks.get(key) or []
            while stack and stack[-1]["name"] != event["name"]:
                stack.pop()
            if not stack:
                continue
            begin = stack.pop()
            fields = {
                k: value(v)
                for k, v in (
                    (begin.get("args") or {}) | (event.get("args") or {})
                ).items()
            }
            out.append(
                {
                    "name": event["name"],
                    "start": begin["ts"] / 1000,
                    "end": event["ts"] / 1000,
                    "fields": fields,
                }
            )
    return out


def within(inner: dict, outer: dict) -> bool:
    return outer["start"] <= inner["start"] and inner["end"] <= outer["end"]


def request_timings(proxy: list[dict], child: list[dict]) -> list[dict]:
    """One row per chat request served by the child, in start order (ms)."""
    spans = intervals(child)
    by_id: dict[str, dict] = {}
    for span in spans:
        rid = span["fields"].get("request_id")
        if span["name"] != "http.request" or not rid:
            continue
        if not str(span["fields"].get("route", "")).startswith("/v1/"):
            continue
        # An async span is entered once per poll, on whichever thread polls
        # it; the longest entry is the one that runs the generation.
        best = by_id.get(rid)
        if best is None or span["end"] - span["start"] > best["end"] - best["start"]:
            by_id[rid] = span
    # The proxy's queue span belongs to the request span open on its thread.
    # Matched by thread rather than by time: the proxy may exit before the
    # request span's end is written.
    queue_wait = {}
    open_request: dict[tuple, list[str]] = {}
    for event in proxy:
        key = (event.get("pid"), event.get("tid"))
        if event.get("name") == "http.request" and event.get("ph") == "B":
            open_request.setdefault(key, []).append(value(event["args"]["request_id"]))
        elif event.get("name") == "http.request" and event.get("ph") == "E":
            if open_request.get(key):
                open_request[key].pop()
        elif (
            event.get("name") == "proxy.queue"
            and event.get("ph") == "E"
            and open_request.get(key)
        ):
            wait = value((event.get("args") or {}).get("queue.wait_ms"))
            queue_wait[open_request[key][-1]] = wait
    rows = []
    for rid, request in sorted(by_id.items(), key=lambda item: item[1]["start"]):
        inside = [s for s in spans if s is not request and within(s, request)]
        prefill = [s for s in inside if s["name"] == "chat.prefill"]
        steps = [s for s in inside if s["name"] == "chat.decode_step"]
        if not prefill:
            continue  # Not a generation request (for example /v1/models).
        fields = request["fields"]
        step_ms = [s["end"] - s["start"] for s in steps]
        rows.append(
            {
                "request_id": rid,
                "queue_wait_ms": queue_wait.get(rid),
                "prefill_ms": sum(s["end"] - s["start"] for s in prefill),
                # The first token comes from the prefill's logits; decode
                # steps produce the rest.
                "first_token_ms": prefill[-1]["end"] - request["start"],
                "decode_steps": len(steps),
                "decode_step_ms": sum(step_ms) / len(step_ms) if step_ms else None,
                "server_ms": request["end"] - request["start"],
                "output_tokens": fields.get("gen_ai.usage.output_tokens"),
                "mlx_peak_bytes": fields.get("mlx.peak_bytes"),
            }
        )
    return rows


def median(values: list[float]) -> float | None:
    values = sorted(v for v in values if v is not None)
    if not values:
        return None
    middle = len(values) // 2
    return (
        values[middle] if len(values) % 2 else (values[middle - 1] + values[middle]) / 2
    )


def summarize(rows: list[dict]) -> dict:
    out = {"requests": len(rows)}
    for key in (
        "queue_wait_ms",
        "prefill_ms",
        "first_token_ms",
        "decode_step_ms",
        "server_ms",
    ):
        out[f"{key}_p50"] = median([row[key] for row in rows])
    peaks = [row["mlx_peak_bytes"] for row in rows if row["mlx_peak_bytes"] is not None]
    out["mlx_peak_bytes_max"] = max(peaks) if peaks else None
    return out


def child_path(proxy_path: Path, model_id: str) -> Path:
    """`mx serve` children write `<stem>.<model-id>.<extension>` beside it."""
    return proxy_path.with_name(f"{proxy_path.stem}.{model_id}{proxy_path.suffix}")


def summarize_files(proxy_path: Path, model_id: str, skip: int = 0) -> dict:
    """Summary of one level's timelines, leaving out its first `skip` requests
    (the warmup, which the client also leaves out)."""
    proxy = load_trace(proxy_path) if proxy_path.exists() else []
    child = load_trace(child_path(proxy_path, model_id))
    return summarize(request_timings(proxy, child)[skip:])


def compare(server: dict, client: dict) -> list[str]:
    """Where the client's medians and the server's disagree by more than
    MISMATCH_RATIO. Client TTFT is set against the server's queue wait plus
    time to the end of prefill; client TPOT against the mean decode step."""
    pairs = [
        (
            "TTFT",
            client["ttft_ms"]["p50"],
            (server["queue_wait_ms_p50"] or 0) + server["first_token_ms_p50"]
            if server["first_token_ms_p50"] is not None
            else None,
        ),
        ("TPOT", client["tpot_ms"]["p50"], server["decode_step_ms_p50"]),
    ]
    out = []
    for name, seen, served in pairs:
        if seen is None or served is None or served <= 0 or seen <= 0:
            continue
        if max(seen, served) / min(seen, served) > MISMATCH_RATIO:
            out.append(
                f"{name} p50: client {seen:.1f} ms vs server {served:.1f} ms "
                f"({seen / served:.2f}x)"
            )
    return out


def fmt(value: float | None, digits: int = 1) -> str:
    return (
        "-"
        if value is None or (isinstance(value, float) and math.isnan(value))
        else f"{value:.{digits}f}"
    )


def one_line(server: dict) -> str:
    return (
        f"server spans: {server['requests']} requests, queue p50 "
        f"{fmt(server['queue_wait_ms_p50'])} ms, prefill p50 {fmt(server['prefill_ms_p50'])} ms, "
        f"first token p50 {fmt(server['first_token_ms_p50'])} ms, decode step p50 "
        f"{fmt(server['decode_step_ms_p50'])} ms"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("trace", type=Path, help="the proxy timeline (--trace-out)")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--skip", type=int, default=0, help="warmup requests")
    args = parser.parse_args()
    summary = summarize_files(args.trace, args.model_id, args.skip)
    print(one_line(summary))
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
