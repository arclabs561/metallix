"""Unit contracts for the Responses tool-result replay qualification runner."""

from __future__ import annotations

import argparse
import copy
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


def call_response(
    key: str = "first", call_id: str = "call_1", item_id: str = "fc_1"
) -> dict:
    return {
        "id": "resp_1",
        "object": "response",
        "model": "metallix-qwen3",
        "status": "completed",
        "usage": {"input_tokens": 5, "output_tokens": 4, "total_tokens": 9},
        "output": [
            {
                "id": item_id,
                "type": "function_call",
                "call_id": call_id,
                "name": "read_fact",
                "arguments": json.dumps({"key": key}, separators=(",", ":")),
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


def message_response() -> dict:
    return {
        "id": "resp_2",
        "object": "response",
        "model": "metallix-qwen3",
        "status": "completed",
        "usage": {"input_tokens": 5, "output_tokens": 4, "total_tokens": 9},
        "output": [
            {
                "id": "msg_2",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": "QUALIFIED:fact"}],
            }
        ],
    }


def message_sse(response: dict) -> str:
    item = response["output"][0]
    part = item["content"][0]
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
            "item": {"id": item["id"]},
        },
        {
            "sequence_number": 2,
            "type": "response.content_part.added",
            "item_id": item["id"],
            "output_index": 0,
            "content_index": 0,
            "part": {**part, "text": ""},
        },
        {
            "sequence_number": 3,
            "type": "response.output_text.delta",
            "item_id": item["id"],
            "output_index": 0,
            "content_index": 0,
            "delta": part["text"],
        },
        {
            "sequence_number": 4,
            "type": "response.output_text.done",
            "item_id": item["id"],
            "output_index": 0,
            "content_index": 0,
            "text": part["text"],
        },
        {
            "sequence_number": 5,
            "type": "response.content_part.done",
            "item_id": item["id"],
            "output_index": 0,
            "content_index": 0,
            "part": part,
        },
        {
            "sequence_number": 6,
            "type": "response.output_item.done",
            "output_index": 0,
            "item": item,
        },
        {"sequence_number": 7, "type": "response.completed", "response": response},
    ]
    return "".join(
        f"event: {event['type']}\ndata: {json.dumps(event)}\n\n" for event in events
    )


class ResponsesToolsTests(unittest.TestCase):
    def test_answer_requires_assistant_output_text_and_message_identity(self) -> None:
        response = call_response()
        item = {
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "status": "completed",
            "content": [{"type": "output_text", "text": "QUALIFIED:fact"}],
        }
        response["output"] = [item]
        self.assertEqual(module.answer_text(response), "QUALIFIED:fact")
        for field, value in (
            ("role", "user"),
            ("role", None),
            ("id", ""),
            ("id", None),
            ("content", []),
            ("content", [{"type": "input_text", "text": "QUALIFIED:fact"}]),
            ("content", [{"type": "refusal", "text": "QUALIFIED:fact"}]),
            ("content", [{"type": "output_text", "text": 123}]),
        ):
            malformed = copy.deepcopy(response)
            malformed["output"][0][field] = value
            with (
                self.subTest(field=field, value=value),
                self.assertRaises(module.ProtocolError),
            ):
                module.answer_text(malformed)

    def test_call_and_replay_contract(self) -> None:
        response = module.validate_response(call_response())
        first = module.call_from_response(response, "first")
        second = module.call_from_response(
            module.validate_response(call_response("second", "call_2", "fc_2")),
            "second",
        )
        replay = module.replay_input([(first, "FACT-first"), (second, "FACT-second")])
        module.validate_replay_input(
            replay, [(first, "FACT-first"), (second, "FACT-second")]
        )
        self.assertEqual(
            [
                item["call_id"]
                for item in replay
                if item.get("type") == "function_call_output"
            ],
            ["call_1", "call_2"],
        )
        self.assertEqual(
            [
                json.loads(item["output"])["qualification_value"]
                for item in replay
                if item.get("type") == "function_call_output"
            ],
            ["FACT-first", "FACT-second"],
        )

    def test_response_model_and_function_item_identity_are_required(self) -> None:
        response = call_response()
        module.validate_expected_model(response, "metallix-qwen3")
        for model in (None, "other-model"):
            malformed = copy.deepcopy(response)
            if model is None:
                del malformed["model"]
            else:
                malformed["model"] = model
            with self.assertRaises(module.ProtocolError):
                module.validate_expected_model(malformed, "metallix-qwen3")
        malformed = copy.deepcopy(response)
        del malformed["output"][0]["id"]
        with self.assertRaises(module.ProtocolError):
            module.call_from_response(malformed, "first")

    def test_sse_binds_message_content_parts_and_completed_text(self) -> None:
        response, events = module.parse_sse(message_sse(message_response()))
        module.validate_stream_events(response, events)
        malformed = [
            event for event in events if event["type"] != "response.content_part.added"
        ]
        with self.assertRaises(module.ProtocolError):
            module.validate_stream_events(response, malformed)
        malformed = copy.deepcopy(events)
        malformed[4]["text"] = "wrong"
        with self.assertRaises(module.ProtocolError):
            module.validate_stream_events(response, malformed)
        malformed = copy.deepcopy(events)
        malformed[5]["part"] = {"type": "output_text", "text": "wrong"}
        with self.assertRaises(module.ProtocolError):
            module.validate_stream_events(response, malformed)
        malformed = copy.deepcopy(events)
        malformed[2]["part"] = response["output"][0]["content"][0]
        with self.assertRaises(module.ProtocolError):
            module.validate_stream_events(response, malformed)
        malformed = copy.deepcopy(events)
        malformed[3]["content_index"] = 1
        with self.assertRaises(module.ProtocolError):
            module.validate_stream_events(response, malformed)

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
        first = module.call_from_response(call_response(), "first")
        replay = module.replay_input([(first, "FACT-x")])
        replay[-1]["call_id"] = "other"
        with self.assertRaises(module.ProtocolError):
            module.validate_replay_input(replay, [(first, "FACT-x")])
        replay = module.replay_input([(first, "FACT-x")])
        replay[-1]["output"] = json.dumps({"qualification_value": "swapped"})
        with self.assertRaises(module.ProtocolError):
            module.validate_replay_input(replay, [(first, "FACT-x")])

    def test_wrong_tool_call_and_answer_are_model_failures(self) -> None:
        response = call_response()
        response["output"][0]["name"] = "wrong"
        with self.assertRaises(module.ModelError):
            module.call_from_response(response, "first")
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


class FixtureCaseTests(unittest.TestCase):
    fixture = SCRIPT.parent.parent / "fixtures/tool-calls/cases.jsonl"

    def setUp(self):
        self.cases, self.fixture_sha = module.load_cases(self.fixture, "all")

    def test_split_identity_and_novel_runtime_facts_without_prompt_leak(self):
        self.assertEqual(
            [row["split"] for row in self.cases],
            ["calibration", "calibration", "heldout", "heldout"],
        )
        selected, digest = module.load_cases(self.fixture, "heldout")
        self.assertEqual(
            [row["id"] for row in selected], ["unicode-escaped", "name-prefix"]
        )
        self.assertEqual(digest, self.fixture_sha)
        first = module.instantiate_case(self.cases[0], "a" * 64, "trial")
        again = module.instantiate_case(self.cases[0], "a" * 64, "trial")
        novel = module.instantiate_case(self.cases[0], "b" * 64, "trial")
        self.assertEqual(first, again)
        self.assertNotEqual(first["steps"], novel["steps"])
        self.assertNotEqual(first["values"], novel["values"])
        for value in first["values"]:
            self.assertNotIn(value, first["prompt"])
        self.assertNotIn("$nonce", first["prompt"])

    def test_loader_rejects_duplicate_ids_wrong_types_and_schema_tools_mix(self):
        variants = []
        variants.append([self.cases[0], self.cases[0]])
        wrong = copy.deepcopy(self.cases[1])
        wrong["steps"][0]["arguments"]["query"]["limit"] = True
        variants.append([wrong])
        mixed = {**self.cases[0], "json_schema": {}}
        variants.append([mixed])
        for rows in variants:
            with self.subTest(rows=rows), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "cases.jsonl"
                path.write_text("\n".join(json.dumps(row) for row in rows))
                with self.assertRaises(ValueError):
                    module.load_cases(path, "all")

    def test_call_oracle_rejects_wrong_name_bool_integer_and_malformed_json(self):
        expected = {"name": "search_records", "arguments": {"query": {"limit": 1}}}
        response = call_response()
        call = response["output"][0]
        call.update(name="search_records", arguments='{"query":{"limit":1}}')
        self.assertEqual(module.case_call(response, expected), call)
        for arguments in (
            '{"query":{"limit":true}}',
            '{"query":{"limit":1.0}}',
            '{"query":{"limit":2}}',
        ):
            call["arguments"] = arguments
            with self.assertRaises(module.ModelError):
                module.case_call(response, expected)
        for arguments in ("{", '{"query":{},"query":{"limit":1}}', '{"query":NaN}'):
            call["arguments"] = arguments
            with self.assertRaises(module.ArgumentParseError):
                module.case_call(response, expected)
        call.update(name="invented_tool", arguments='{"query":{"limit":1}}')
        with self.assertRaises(module.ModelError):
            module.case_call(response, expected)

    def test_unicode_sse_offline_reconstructs_exact_arguments(self):
        instance = module.instantiate_case(self.cases[2], "a" * 64, "unicode")
        expected = instance["steps"][0]
        response = call_response()
        response["output"][0].update(
            name=expected["name"],
            arguments=json.dumps(expected["arguments"], ensure_ascii=False),
        )
        parsed, events = module.parse_sse(sse(response))
        module.validate_stream_events(parsed, events)
        self.assertEqual(
            module.case_call(parsed, expected)["arguments"],
            response["output"][0]["arguments"],
        )
        bad = copy.deepcopy(events)
        next(
            event
            for event in bad
            if event["type"] == "response.function_call_arguments.delta"
        )["delta"] = "fabricated"
        with self.assertRaises(module.ProtocolError):
            module.validate_stream_events(parsed, bad)

    def drive_case(self, *, corrupt_answer=False, duplicate_identity=False):
        case = self.cases[0]
        instance = module.instantiate_case(case, "a" * 64, "trial")
        responses = []
        for index, expected in enumerate(instance["steps"]):
            response = call_response(
                call_id=f"call_{0 if duplicate_identity else index}",
                item_id=f"item_{index}",
            )
            response["output"][0].update(
                name=expected["name"], arguments=json.dumps(expected["arguments"])
            )
            responses.append(response)
        answer = message_response()
        answer["output"][0]["content"][0]["text"] = (
            "QUALIFIED:invented|invented"
            if corrupt_answer
            else "QUALIFIED:" + "|".join(instance["values"])
        )
        responses.append(answer)
        payloads = []

        def fake_request(host, port, path, payload, timeout, output, label):
            payloads.append(copy.deepcopy(payload))
            return responses[len(payloads) - 1], None

        args = argparse.Namespace(
            model_id="test", timeout_seconds=30, output=Path("unused")
        )
        with patch.object(module, "request", side_effect=fake_request):
            result = module.run_case(
                case, instance, "127.0.0.1", 8321, "/v1", args, False, "trial"
            )
        self.assertEqual(len(result), 3)
        self.assertEqual(
            payloads[0]["input"], [{"role": "user", "content": instance["prompt"]}]
        )
        self.assertEqual(
            payloads[1]["input"][2],
            module.tool_result(responses[0]["output"][0], instance["values"][0]),
        )
        self.assertEqual(
            payloads[2]["input"][4],
            module.tool_result(responses[1]["output"][0], instance["values"][1]),
        )
        return result

    def test_ordered_replay_supplies_only_observed_call_results(self):
        self.drive_case()

    def test_hallucinated_answer_and_reused_call_id_fail(self):
        with self.assertRaises(module.ModelError):
            self.drive_case(corrupt_answer=True)
        with self.assertRaises(module.ProtocolError):
            self.drive_case(duplicate_identity=True)

    def test_case_run_receipt_records_identity_seed_and_protocol_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            args = argparse.Namespace(
                seed="c" * 64,
                output=Path(directory),
                repeats=1,
                model_id="test",
                timeout_seconds=30,
            )
            plan = {"fixture_sha256": self.fixture_sha, "status": "dry_run"}
            with patch.object(
                module, "request", side_effect=module.ProtocolError("bad SSE")
            ):
                code = module.run_cases(
                    args, plan, self.cases[:1], "127.0.0.1", 8321, "/v1"
                )
            receipt = json.loads((args.output / "receipt.json").read_text())
            self.assertEqual(code, 1)
            self.assertEqual(receipt["seed"], "c" * 64)
            self.assertEqual(receipt["fixture_sha256"], self.fixture_sha)
            self.assertEqual(receipt["status"], "failed")
            self.assertEqual(len(receipt["trials"]), 2)
            for row in receipt["trials"]:
                self.assertIsNone(row["task_correct"])
                self.assertEqual(row["failure"]["class"], "protocol")
                self.assertEqual(len(row["instance_sha256"]), 64)

    def test_non_object_call_output_finishes_case_receipt_as_failed(self):
        for item in (None, "text"):
            with self.subTest(item=item), tempfile.TemporaryDirectory() as directory:
                response = call_response()
                response["output"] = [item]
                response = module.validate_response(response)
                args = argparse.Namespace(
                    seed="d" * 64,
                    output=Path(directory),
                    repeats=1,
                    model_id="test",
                    timeout_seconds=30,
                )
                with patch.object(module, "request", return_value=(response, None)):
                    code = module.run_cases(
                        args, {}, self.cases[:1], "127.0.0.1", 8321, "/v1"
                    )
                receipt = json.loads((args.output / "receipt.json").read_text())
                self.assertEqual(code, 1)
                self.assertEqual(receipt["status"], "failed")
                self.assertEqual(len(receipt["trials"]), 2)
                for row in receipt["trials"]:
                    self.assertFalse(row["passed"])
                    self.assertEqual(row["failure"]["class"], "model")
                    self.assertEqual(
                        row["failure"]["error"],
                        "expected exactly one ordered function call",
                    )

    def test_cases_dry_run_never_contacts_server_or_creates_output(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "absent"
            args = [
                str(SCRIPT),
                "--url",
                "http://127.0.0.1:8321/v1",
                "--model-id",
                "test",
                "--output",
                str(output),
                "--cases",
                str(self.fixture),
                "--split",
                "heldout",
            ]
            with (
                patch.object(sys, "argv", args),
                patch.object(module, "request") as request,
            ):
                self.assertEqual(module.main(), 0)
                request.assert_not_called()
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
