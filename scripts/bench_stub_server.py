#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""A model-free OpenAI-compatible streaming server for benchmark dry runs.

Streams `max_tokens` one-word chunks at a fixed per-token delay on
`/v1/chat/completions` and `/v1/responses`, so the load harness and campaign
driver can be exercised end to end without a model or a GPU. Its numbers mean
nothing about any engine.

Usage:
  scripts/bench_stub_server.py --listen 127.0.0.1:8400 --tpot-ms 2
"""

from __future__ import annotations

import argparse
import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def handler(tpot_s: float, ttft_s: float):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args) -> None:
            pass

        def do_GET(self) -> None:
            if self.path != "/v1/models":
                self.send_error(404)
                return
            body = json.dumps({"data": [{"id": "stub"}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self) -> None:
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if self.path == "/v1/chat/completions":
                tokens = request.get("max_tokens") or 16
            elif self.path == "/v1/responses":
                tokens = request.get("max_output_tokens") or 16
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Connection", "close")
            self.end_headers()
            time.sleep(ttft_s)
            chat = self.path == "/v1/chat/completions"
            for _ in range(tokens):
                if chat:
                    event = {"choices": [{"delta": {"content": " tok"}}]}
                else:
                    event = {"type": "response.output_text.delta", "delta": " tok"}
                self.wfile.write(f"data: {json.dumps(event)}\n\n".encode())
                self.wfile.flush()
                time.sleep(tpot_s)
            if chat:
                usage = {"prompt_tokens": 1, "completion_tokens": tokens}
                events = [{"choices": [], "usage": usage}, "[DONE]"]
            else:
                usage = {"input_tokens": 1, "output_tokens": tokens}
                events = [{"type": "response.completed", "response": {"usage": usage}}]
            for event in events:
                data = event if isinstance(event, str) else json.dumps(event)
                self.wfile.write(f"data: {data}\n\n".encode())
            self.wfile.flush()

    return Handler


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--listen", default="127.0.0.1:8400")
    parser.add_argument("--tpot-ms", type=float, default=2.0)
    parser.add_argument("--ttft-ms", type=float, default=20.0)
    # Accepted so the campaign's cache arms can be dry-run; the stub has no cache.
    parser.add_argument("--prefix-cache", choices=("on", "off"))
    args = parser.parse_args()
    host, port = args.listen.rsplit(":", 1)
    server = ThreadingHTTPServer(
        (host, int(port)), handler(args.tpot_ms / 1000, args.ttft_ms / 1000)
    )
    server.daemon_threads = True
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
