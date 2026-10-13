"""Falsifiers for paired schema qualification; no model or network required."""

import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location(
    "schema_tools", Path(__file__).with_name("qualify-schema-tools.py")
)
assert SPEC and SPEC.loader
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def response(payload, text='{"answer":"blue"}'):
    # Shape follows crates/app/cli/src/responses.rs::response_value.
    return {
        "id": "resp_fixture",
        "object": "response",
        "model": payload["model"],
        "status": "completed",
        "usage": {"input_tokens": 10, "output_tokens": 8, "total_tokens": 18},
        "metallix": {
            "sampling": {
                "temperature": 0.0,
                "top_p": 1.0,
                "seed": None,
                "top_k": None,
                "sampler": None,
            }
        },
        "output": [
            {
                "id": "msg_fixture",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": text}],
            }
        ],
    }


class QualificationTests(unittest.TestCase):
    def setUp(self):
        self.pairs = module.paired_requests("metallix-qwen3", 512)
        self.payload = self.pairs[0]["constrained"]

    def test_transport_preserves_incomplete_body_and_caps_output(self):
        value = response(self.payload)
        value["status"] = "incomplete"
        value["incomplete_details"] = {"reason": "max_output_tokens"}
        raw = json.dumps(value).encode()

        class Reply:
            status = 200

            def __init__(self, body):
                self.body = body

            def read1(self, count):
                result, self.body = self.body[:count], self.body[count:]
                return result

        with tempfile.TemporaryDirectory() as root:
            output = Path(root)
            with patch.object(module.http.client, "HTTPConnection") as connection:
                connection.return_value.getresponse.return_value = Reply(raw)
                actual = module.fetch(
                    "127.0.0.1", 1, "/v1", self.payload, 1, output, "limited"
                )
                self.assertEqual(actual, value)
                self.assertEqual((output / "limited.response.raw").read_bytes(), raw)
                connection.return_value.close.assert_called_once()
            with patch.object(module.http.client, "HTTPConnection") as connection:
                connection.return_value.getresponse.return_value = Reply(b"x" * 65)
                with (
                    patch.object(module, "MAX_RESPONSE_BYTES", 64),
                    self.assertRaises(module.ProtocolError),
                ):
                    module.fetch(
                        "127.0.0.1", 1, "/v1", self.payload, 1, output, "oversize"
                    )
                self.assertEqual((output / "oversize.response.raw").stat().st_size, 65)
                connection.return_value.close.assert_called_once()

    def test_pair_changes_only_format_and_baseline_has_no_grammar(self):
        self.assertEqual(len(self.pairs), 16)
        self.assertEqual(
            len({p["baseline"]["input"][0]["content"] for p in self.pairs}), 16
        )
        for pair in self.pairs:
            constrained = copy.deepcopy(pair["constrained"])
            self.assertEqual(constrained.pop("text")["format"]["schema"], module.SCHEMA)
            self.assertEqual(pair["baseline"], constrained)
            self.assertEqual(pair["baseline"]["tool_choice"], "auto")
            self.assertIs(pair["baseline"]["tools"][0]["strict"], False)

    def test_final_schema_rejects_wrong_type_extra_keys_duplicate_and_prose(self):
        self.assertEqual(
            module.classify_response(response(self.payload), self.payload, True),
            "final",
        )
        for text in (
            '{"answer":3}',
            '{"answer":"x","extra":0}',
            '{"answer":"a","answer":"b"}',
            'Here is the answer: {"answer":"x"}',
        ):
            with self.subTest(text=text), self.assertRaises(module.ProtocolError):
                module.classify_response(
                    response(self.payload, text), self.payload, True
                )
        self.assertEqual(
            module.classify_response(
                response(self.payload, "blue"), self.payload, False
            ),
            "final",
        )

    def test_valid_call_can_have_commentary_and_reasoning(self):
        value = response(self.payload, "Let me check.")
        value["output"].insert(
            0, {"id": "rs_fixture", "type": "reasoning", "content": []}
        )
        call = {
            "id": "fc_fixture",
            "type": "function_call",
            "call_id": "call_fixture",
            "name": "read_fact",
            "arguments": json.dumps({"key": "deployment_version"}),
            "status": "completed",
        }
        value["output"].append(call)
        self.assertEqual(module.classify_response(value, self.payload, True), "call")
        call["arguments"] = '{"key":false}'
        with self.assertRaises(module.ProtocolError):
            module.classify_response(value, self.payload, True)

    def test_incomplete_is_not_final_and_wrong_sampling_is_not_comparable(self):
        value = response(self.payload)
        value["status"] = "incomplete"
        value["incomplete_details"] = {"reason": "max_output_tokens"}
        self.assertEqual(module.classify_response(value, self.payload, True), "limit")
        value["metallix"]["sampling"]["seed"] = 99
        with self.assertRaises(module.ProtocolError):
            module.classify_response(value, self.payload, True)

    def test_quarter_calls_required_and_every_call_preserved(self):
        rows = [
            {
                "id": str(i),
                "baseline": "call" if i < 4 else "final",
                "constrained": "call" if i < 4 else "final",
            }
            for i in range(16)
        ]
        self.assertEqual(module.verdict(rows)["status"], "pass")
        rows[0]["constrained"] = "final"
        self.assertEqual(module.verdict(rows)["status"], "fail")
        rows[0]["baseline"] = "final"
        self.assertEqual(module.verdict(rows)["status"], "inconclusive")

    def test_truncation_error_and_missing_samples_cannot_pass(self):
        rows = [
            {"id": str(i), "baseline": "call", "constrained": "call"} for i in range(16)
        ]
        for mutation in ({"constrained": "limit"}, {"error": "transport failed"}):
            bad = copy.deepcopy(rows)
            bad[9].update(mutation)
            self.assertEqual(module.verdict(bad)["status"], "fail")
        self.assertEqual(module.verdict(rows[:-1])["status"], "fail")
        rows[-1]["id"] = rows[0]["id"]
        self.assertEqual(module.verdict(rows)["status"], "fail")


if __name__ == "__main__":
    unittest.main()
