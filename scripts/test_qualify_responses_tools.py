"""Unit contracts for the Responses tool-result replay qualification runner."""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
import tempfile
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from typing import ClassVar
from unittest.mock import patch

SCRIPT = Path(__file__).with_name("qualify-responses-tools.py")
SPEC = importlib.util.spec_from_file_location("qualify_responses_tools", SCRIPT)
assert SPEC and SPEC.loader
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def call_response() -> dict:
    return {
        "id": "resp_1",
        "object": "response",
        "status": "completed",
        "usage": {"input_tokens": 5, "output_tokens": 4, "total_tokens": 9},
        "output": [
            {
                "id": "fc_1",
                "type": "function_call",
                "call_id": "call_1",
                "name": "read_fact",
                "arguments": '{"key":"qualification_value"}',
                "status": "completed",
            }
        ],
    }


def sse(response: dict) -> str:
    call = response["output"][0]
    events = [
        {
            "sequence_number": 0,
            "type": "response.created",
            "response": {
                "id": response["id"],
                "object": "response",
                "status": "in_progress",
                "output": [],
            },
        },
        {
            "sequence_number": 1,
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {"id": call["id"]},
        },
        {
            "sequence_number": 2,
            "type": "response.function_call_arguments.delta",
            "item_id": call["id"],
            "output_index": 0,
            "delta": call["arguments"],
        },
        {
            "sequence_number": 3,
            "type": "response.function_call_arguments.done",
            "item_id": call["id"],
            "output_index": 0,
            "arguments": call["arguments"],
        },
        {
            "sequence_number": 4,
            "type": "response.output_item.done",
            "output_index": 0,
            "item": call,
        },
        {"sequence_number": 5, "type": "response.completed", "response": response},
    ]
    return "".join(
        f"event: {event['type']}\ndata: {json.dumps(event)}\n\n" for event in events
    )


class ResponsesToolsTests(unittest.TestCase):
    def test_call_and_replay_contract(self) -> None:
        response = module.validate_response(call_response())
        call = module.call_from_response(response)
        replay = module.replay_input(call, "FACT-hidden")
        module.validate_replay_input(replay, "call_1")
        self.assertEqual(replay[-1]["call_id"], "call_1")

    def test_sse_validates_sequence_ids_and_argument_assembly(self) -> None:
        response, events = module.parse_sse(sse(call_response()))
        module.validate_stream_events(response, events)

    def test_server_keepalive_comments_preserve_event_sequence(self) -> None:
        raw = sse(call_response()).replace("\n\n", "\n\n: generating\n\n")
        response, events = module.parse_sse(raw)
        module.validate_stream_events(response, events)
        self.assertEqual(response, call_response())

    def test_sse_rejects_mismatched_event_label_and_reordered_lifecycle(self) -> None:
        malformed = sse(call_response()).replace(
            "event: response.created", "event: response.failed", 1
        )
        with self.assertRaises(module.ProtocolError):
            module.parse_sse(malformed)
        frames = [frame for frame in sse(call_response()).split("\n\n") if frame]
        frames[1], frames[2] = frames[2], frames[1]
        for sequence, frame in enumerate(frames):
            lines = frame.splitlines()
            event = json.loads(lines[1][6:])
            event["sequence_number"] = sequence
            frames[sequence] = f"{lines[0]}\ndata: {json.dumps(event)}"
        reordered = "\n\n".join(frames) + "\n\n"
        response, parsed = module.parse_sse(reordered)
        with self.assertRaises(module.ProtocolError):
            module.validate_stream_events(response, parsed)

    def test_sse_rejects_created_id_and_unknown_item_delta(self) -> None:
        created_mismatch = sse(call_response()).replace(
            '"id": "resp_1"', '"id": "wrong"', 1
        )
        with self.assertRaises(module.ProtocolError):
            module.parse_sse(created_mismatch)
        injected = (
            sse(call_response())
            .replace(
                "event: response.completed",
                'event: response.output_text.delta\ndata: {"sequence_number": 5, "type": "response.output_text.delta", "item_id": "unknown", "output_index": 0, "delta": "x"}\n\nevent: response.completed',
                1,
            )
            .replace(
                '"sequence_number": 5, "type": "response.completed"',
                '"sequence_number": 6, "type": "response.completed"',
            )
        )
        response, parsed = module.parse_sse(injected)
        with self.assertRaises(module.ProtocolError):
            module.validate_stream_events(response, parsed)

    def test_malformed_or_mismatched_replay_evidence_is_protocol_failure(self) -> None:
        with self.assertRaises(module.ProtocolError):
            module.parse_sse("data: not-json\n\n")
        replay = module.replay_input(
            module.call_from_response(call_response()), "FACT-x"
        )
        replay[-1]["call_id"] = "other"
        with self.assertRaises(module.ProtocolError):
            module.validate_replay_input(replay, "call_1")

    def test_wrong_tool_call_and_answer_are_model_failures(self) -> None:
        response = call_response()
        response["output"][0]["name"] = "wrong"
        with self.assertRaises(module.ModelError):
            module.call_from_response(response)
        with self.assertRaises(module.ModelError):
            module.answer_text(module.validate_response(call_response()))

    def test_unsafe_urls_are_rejected(self) -> None:
        for value in (
            "https://127.0.0.1:1/v1",
            "http://example.com:1/v1",
            "http://127.0.0.1:1/no",
        ):
            with (
                self.subTest(value=value),
                self.assertRaises(argparse.ArgumentTypeError),
            ):
                module.loopback_base_url(value)
        self.assertEqual(module.loopback_base_url("http://127.0.0.1:8321/v1")[3], "/v1")
        self.assertEqual(module.loopback_base_url("http://127.0.0.1:8321")[3], "/v1")

    def test_request_normalizes_root_and_deadline_retains_partial_raw(self) -> None:
        class Handler(BaseHTTPRequestHandler):
            paths: ClassVar[list[str]] = []

            def log_message(self, *_: object) -> None:
                pass

            def do_POST(self) -> None:
                type(self).paths.append(self.path)
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", "100")
                self.end_headers()
                self.wfile.write(b"{")
                self.wfile.flush()
                time.sleep(1.2)

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as directory:
                _, host, port, path = module.loopback_base_url(
                    f"http://127.0.0.1:{server.server_port}"
                )
                with self.assertRaises(module.TransportError):
                    module.request(
                        host,
                        port,
                        path,
                        {"stream": False},
                        1,
                        Path(directory),
                        "deadline",
                    )
                self.assertEqual(Handler.paths, ["/v1/responses"])
                self.assertEqual(
                    (Path(directory) / "deadline.response.raw").read_bytes(), b"{"
                )
        finally:
            server.shutdown()
            server.server_close()

    def test_dry_run_never_connects_or_creates_output(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--url",
                "http://127.0.0.1:8321/v1",
                "--model-id",
                "test",
                "--output",
                "/definitely/not/created",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertEqual(json.loads(result.stdout)["status"], "dry_run")

    def test_dirty_output_is_rejected_before_network_work(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "output"
            output.mkdir()
            (output / "prior").write_text("prior")
            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--run",
                    "--url",
                    "http://127.0.0.1:8321/v1",
                    "--model-id",
                    "test",
                    "--output",
                    str(output),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual((output / "prior").read_text(), "prior")

    def test_interrupt_finalizes_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "output"
            with (
                patch.object(
                    sys,
                    "argv",
                    [
                        str(SCRIPT),
                        "--run",
                        "--url",
                        "http://127.0.0.1:8321/v1",
                        "--model-id",
                        "test",
                        "--output",
                        str(output),
                    ],
                ),
                patch.object(module, "request", side_effect=KeyboardInterrupt),
            ):
                self.assertEqual(module.main(), 130)
            self.assertEqual(
                json.loads((output / "receipt.json").read_text())["status"],
                "interrupted",
            )


if __name__ == "__main__":
    unittest.main()
