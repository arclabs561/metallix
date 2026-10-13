"""Offline contract checks for operator search examples; no provider requests."""

import contextlib
import importlib.util
import io
import json
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.parse import parse_qs, urlparse


def load(name):
    spec = importlib.util.spec_from_file_location(
        name, Path(__file__).parent / "web-search" / f"{name}.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Reply(io.BytesIO):
    status = 200


class AdapterTests(unittest.TestCase):
    def test_provider_output_shape_and_domain_filters(self):
        for name, provider in (
            (
                "searxng",
                {
                    "results": [
                        {
                            "url": "https://example.com/a",
                            "title": "Title",
                            "content": "Snippet",
                        }
                    ]
                },
            ),
            (
                "brave",
                {
                    "web": {
                        "results": [
                            {
                                "url": "https://example.com/a",
                                "title": "Title",
                                "description": "Snippet",
                            }
                        ]
                    }
                },
            ),
        ):
            with self.subTest(adapter=name):
                module = load(name)
                request = {
                    "query": "test query",
                    "max_results": 5,
                    "allowed_domains": ["example.com"],
                    "blocked_domains": ["blocked.example.com"],
                }
                output = io.StringIO()
                with (
                    patch.dict(
                        module.os.environ,
                        {
                            "SEARXNG_URL": "https://search.example",
                            "BRAVE_API_KEY": "test-fixture-not-a-key",
                        },
                    ),
                    patch.object(module.sys, "stdin", io.StringIO(json.dumps(request))),
                    contextlib.redirect_stdout(output),
                    patch.object(
                        module,
                        "urlopen",
                        return_value=Reply(json.dumps(provider).encode()),
                    ) as opened,
                ):
                    module.run()
                self.assertEqual(
                    json.loads(output.getvalue()),
                    [
                        {
                            "url": "https://example.com/a",
                            "title": "Title",
                            "snippet": "Snippet",
                        }
                    ],
                )
                sent = opened.call_args.args[0]
                query = parse_qs(
                    sent.data.decode()
                    if name == "searxng"
                    else urlparse(sent.full_url).query
                )["q"][0]
                self.assertIn("site:example.com", query)
                self.assertIn("-site:blocked.example.com", query)
                self.assertEqual(opened.call_args.kwargs["timeout"], 8)

    def test_health_never_reads_stdin_or_prints_results(self):
        for name in ("searxng", "brave"):
            module = load(name)
            output = io.StringIO()
            with (
                patch.dict(
                    module.os.environ,
                    {
                        "SEARXNG_URL": "https://search.example",
                        "BRAVE_API_KEY": "test-fixture-not-a-key",
                    },
                ),
                patch.object(module.sys, "stdin", io.StringIO("not json")),
                contextlib.redirect_stdout(output),
                patch.object(module, "urlopen", return_value=Reply(b"{}")),
            ):
                module.run(True)
            self.assertEqual(output.getvalue(), "")

    def test_provider_response_limit_is_enforced(self):
        module = load("searxng")
        with (
            patch.dict(module.os.environ, {"SEARXNG_URL": "https://search.example"}),
            patch.object(
                module.sys, "stdin", io.StringIO('{"query":"q","max_results":5}')
            ),
            patch.object(module, "urlopen", return_value=Reply(b"x" * 1_048_577)),
            self.assertRaises(ValueError),
        ):
            module.run()


if __name__ == "__main__":
    unittest.main()
