# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Offline tests for the trending Hub coverage script, on recorded listing data."""

from __future__ import annotations

import email.message
import io
import json
import pathlib
import sys
import tempfile
import unittest
import urllib.error

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import hub_trending

# Compacted objects from GET /api/models?sort=trendingScore&expand[]=... on
# 2026-10-06T00:25Z (file and tag lists trimmed).
QWEN3_BF16: dict = {
    "id": "allenai/AstaBrief_8B",
    "sha": "6a34f54dcd7887b790a570249b4a2a2fe81c6694",
    "trendingScore": 49,
    "pipeline_tag": "text-generation",
    "library_name": None,
    "tags": ["base_model:finetune:allenai/AstaBrief_8B_SFT"],
    "gated": False,
    "config": {"architectures": ["Qwen3ForCausalLM"], "model_type": "qwen3"},
    "safetensors": {},
    "gguf": {},
    "files": ["config.json", "pytorch_model-00001-of-00004.safetensors"],
}
GGUF_NEW_ARCH: dict = {
    "id": "Venastine-Research/Xing4.0-29B-A4B-GGUF",
    "sha": "845d645bfa2be61b7774f2a12a8bec9e2231ec5d",
    "trendingScore": 398,
    "pipeline_tag": "text-generation",
    "library_name": "transformers",
    "tags": ["gguf", "base_model:quantized:XingChen-AGI/Xing4.0-29B-A4B"],
    "gated": False,
    "config": {"architectures": ["Xing4_0ForCausalLM"], "model_type": "xing4_0"},
    "safetensors": {},
    "gguf": {"architecture": "xing4_0", "total": 31215031088},
    "files": ["Xing4.0-29B-A4B-IQ2_M.gguf", "Xing4.0-29B-A4B-Q4_K_M.gguf"],
}
MLX_4BIT: dict = {
    "id": "orcarouter/Qwen3.8-27B-Uncensored-MLX",
    "sha": "bc60491a36464626020ce82d5556b1b8f8cd90c7",
    "trendingScore": 26,
    "pipeline_tag": "image-text-to-text",
    "library_name": "mlx",
    "tags": ["mlx", "base_model:quantized:Qwen/Qwen3.8-27B"],
    "gated": False,
    "config": {
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "model_type": "qwen3_5",
        "quantization_config": {"bits": 4},
    },
    "safetensors": {"parameters": {"BF16": 463375600, "U32": 26893352960}},
    "gguf": {},
    "files": ["config.json", "model-00001-of-00002.safetensors"],
}
GATED_GGUF_NO_CONFIG: dict = {
    "id": "orcarouter/OrcaSAQ-2-Cyber-27B-Uncensored-GGUF",
    "sha": "a0ebe1b5ad5c009cd382908585c04b7e9e0cf0c0",
    "trendingScore": 222,
    "pipeline_tag": "text-generation",
    "library_name": "llama.cpp",
    "tags": ["gguf", "base_model:quantized:orcarouter/Qwen3.8-27B-Uncensored"],
    "gated": "auto",
    "config": {},
    "safetensors": {},
    "gguf": {"architecture": "qwen35", "total": 27320697856},
    "files": ["OrcaSAQ-2-27B-Uncensored.gguf"],
}
DIFFUSERS: dict = {
    "id": "Qwen/Qwen-Image-2.1",
    "sha": "d26bb61231c349cf6b7896fa83353113880e1ba3",
    "trendingScore": 286,
    "pipeline_tag": "text-to-image",
    "library_name": "diffusers",
    "tags": ["diffusers", "diffusers:QwenImage21Pipeline"],
    "gated": False,
    "config": {"diffusers": {"_class_name": "QwenImage21Pipeline"}},
    "safetensors": {"parameters": {"BF16": 7115124736}},
    "gguf": {},
    "files": ["model_index.json", "transformer/diffusion_pytorch_model.safetensors"],
}
CLASSIFIER: dict = {
    "id": "convaiinnovations/laya",
    "trendingScore": 688,
    "pipeline_tag": "text-classification",
    "library_name": "transformers",
    "tags": [],
    "config": {"architectures": ["LayaTypedDecisions"], "model_type": "laya"},
    "files": ["config.json", "model.safetensors"],
}
QWEN35_TEXT: dict = {"config": {"model_type": "qwen3_5_text"}}
LORA_NO_CONFIG: dict = {
    "library_name": "diffusers",
    "tags": ["base_model:adapter:Qwen/Qwen-Image-2.1"],
    "config": {},
}
# cdiamond/Qwen3.8-27B-iMatrix-NVFP4-MTP-GGUF: the listing's GGUF
# architecture is the vision projector's.
MMPROJ_GGUF: dict = {
    "tags": ["gguf", "base_model:Qwen/Qwen3.8-27B"],
    "config": {},
    "gguf": {"architecture": "clip"},
    "files": ["Qwen3.8-27B-iMatrix-NVFP4-MTP.gguf", "mmproj-Qwen3.8-27B-F16.gguf"],
}

# The live Link header shape (cursor shortened).
LINK = (
    "<https://huggingface.co/api/models?sort=trendingScore&limit=3"
    '&cursor=eyJ0cmVuZGluZ1Njb3JlIjo2ODV9%3D>; rel="next"'
)


def page(scores: list, next_url: str | None):
    body = json.dumps(
        [{"id": f"m{i}", "trendingScore": s} for i, s in enumerate(scores)]
    )
    headers = {"Link": f'<{next_url}>; rel="next"'} if next_url else {}
    return 200, headers, body.encode()


class FakeFetcher:
    def __init__(self, pages: list):
        self.pages, self.urls = list(pages), []

    def __call__(self, url):
        self.urls.append(url)
        return self.pages.pop(0) if self.pages else page([], None)


class LinkHeader(unittest.TestCase):
    def test_next_present(self):
        self.assertEqual(
            hub_trending.next_link(LINK),
            "https://huggingface.co/api/models?sort=trendingScore&limit=3"
            "&cursor=eyJ0cmVuZGluZ1Njb3JlIjo2ODV9%3D",
        )

    def test_next_absent_or_other_rel(self):
        self.assertIsNone(hub_trending.next_link(None))
        self.assertIsNone(hub_trending.next_link(""))
        self.assertIsNone(hub_trending.next_link('<https://x/?cursor=a>; rel="prev"'))

    def test_next_among_several_links_unquoted_rel(self):
        header = '<https://x/?cursor=p>; rel="prev", <https://x/?cursor=n>; rel=next'
        self.assertEqual(hub_trending.next_link(header), "https://x/?cursor=n")


class Paging(unittest.TestCase):
    def run_listing(self, pages, **kwargs):
        fetcher = FakeFetcher(pages)
        kwargs.setdefault("max_pages", 10)
        models, meta = hub_trending.list_trending(
            min_score=0, fetcher=fetcher, sleep=lambda _: None, **kwargs
        )
        return [m["trendingScore"] for m in models], meta, fetcher

    def test_stops_at_first_non_positive_score(self):
        scores, meta, fetcher = self.run_listing(
            [page([10, 5], "https://h/2"), page([3, 0, 0], "https://h/3")]
        )
        self.assertEqual(scores, [10, 5, 3])
        self.assertEqual(meta, {"pages": 2, "models": 3, "stop": "min_score"})
        self.assertEqual(fetcher.urls[1], "https://h/2")

    def test_null_score_ends_the_trending_set(self):
        scores, meta, _ = self.run_listing([page([4, None, 2], "https://h/2")])
        self.assertEqual((scores, meta["stop"]), ([4], "min_score"))

    def test_max_pages_caps_requests(self):
        scores, meta, fetcher = self.run_listing(
            [page([9], "https://h/2"), page([8], "https://h/3")], max_pages=1
        )
        self.assertEqual(
            (scores, meta["stop"], len(fetcher.urls)), ([9], "max_pages", 1)
        )

    def test_missing_next_link_ends_listing(self):
        scores, meta, _ = self.run_listing([page([2, 1], None)])
        self.assertEqual((scores, meta["stop"]), ([2, 1], "end_of_listing"))

    def test_http_error_is_a_stop_reason(self):
        scores, meta, _ = self.run_listing([page([2], "https://h/2"), (500, {}, b"")])
        self.assertEqual((scores, meta["stop"]), ([2], "http_500"))


class FakeResponse(io.BytesIO):
    status = 200
    headers = email.message.Message()


def http_error(code: int, retry_after: str | None = None):
    headers = email.message.Message()
    if retry_after is not None:
        headers["Retry-After"] = retry_after
    return urllib.error.HTTPError("https://h", code, "error", headers, io.BytesIO())


class Fetch(unittest.TestCase):
    def opener_for(self, outcomes):
        calls = []

        def opener(request, **_):
            calls.append(request)
            outcome = outcomes.pop(0)
            if isinstance(outcome, Exception):
                raise outcome
            return FakeResponse(outcome)

        return opener, calls

    def test_429_waits_retry_after_then_succeeds(self):
        opener, calls = self.opener_for([http_error(429, "7"), b"[]"])
        waits = []
        status, _, body = hub_trending.fetch(
            "https://h", opener=opener, sleep=waits.append
        )
        self.assertEqual((status, body, waits, len(calls)), (200, b"[]", [7.0], 2))

    def test_429_retries_are_bounded(self):
        opener, calls = self.opener_for([http_error(429, "1")] * 3)
        status, _, _ = hub_trending.fetch(
            "https://h", opener=opener, sleep=lambda _: None, retries=2
        )
        self.assertEqual((status, len(calls)), (429, 3))

    def test_gated_returns_without_retry(self):
        opener, calls = self.opener_for([http_error(401)])
        status, _, _ = hub_trending.fetch(
            "https://h", opener=opener, sleep=lambda _: None
        )
        self.assertEqual((status, len(calls)), (401, 1))

    def test_token_goes_only_in_authorization_header(self):
        opener, calls = self.opener_for([b"{}"])
        hub_trending.fetch("https://h/x", opener=opener, token="hf_secret")
        self.assertEqual(calls[0].get_header("Authorization"), "Bearer hf_secret")
        self.assertNotIn("hf_secret", calls[0].full_url)


class Classification(unittest.TestCase):
    def test_kind(self):
        self.assertEqual(hub_trending.kind(QWEN3_BF16), "text")
        self.assertEqual(hub_trending.kind(MLX_4BIT), "text")
        self.assertEqual(hub_trending.kind(DIFFUSERS), "diffusion")
        self.assertEqual(hub_trending.kind(CLASSIFIER), "other")

    def test_family_normalizes_gguf_and_text_variants(self):
        self.assertEqual(hub_trending.family(GATED_GGUF_NO_CONFIG), "qwen35")
        self.assertEqual(hub_trending.family(MLX_4BIT), "qwen35")
        self.assertEqual(hub_trending.family(QWEN35_TEXT), "qwen35")
        self.assertEqual(hub_trending.family(DIFFUSERS), "qwenimage21pipeline")
        self.assertEqual(
            hub_trending.family(LORA_NO_CONFIG), "base:Qwen/Qwen-Image-2.1"
        )
        self.assertEqual(hub_trending.family(MMPROJ_GGUF), "base:Qwen/Qwen3.8-27B")

    def test_weight_format_and_quantization(self):
        self.assertEqual(hub_trending.weight_format(QWEN3_BF16), "safetensors")
        self.assertEqual(hub_trending.weight_format(GGUF_NEW_ARCH), "gguf_only")
        self.assertIsNone(hub_trending.quantization(QWEN3_BF16))
        self.assertEqual(hub_trending.quantization(MLX_4BIT), "mlx_affine_q4")

    def test_config_file(self):
        self.assertEqual(hub_trending.config_file(QWEN3_BF16), "config.json")
        self.assertEqual(hub_trending.config_file(DIFFUSERS), "model_index.json")
        self.assertIsNone(hub_trending.config_file(GATED_GGUF_NO_CONFIG))

    def test_classify(self):
        ok = {"status": "ok"}
        yes = {"supported": True, "adapter": "qwen3"}
        no = {"supported": False, "adapter": None}
        refused = {"supported": False, "adapter": "qwen3"}
        self.assertEqual(hub_trending.classify(QWEN3_BF16, ok, yes), "supported")
        self.assertEqual(hub_trending.classify(QWEN3_BF16, ok, no), "unsupported_arch")
        self.assertEqual(
            hub_trending.classify(QWEN3_BF16, ok, refused), "adapter_refused"
        )
        self.assertEqual(hub_trending.classify(MLX_4BIT, ok, refused), "format_blocked")
        self.assertEqual(
            hub_trending.classify(MLX_4BIT, ok, refused, mx_checks_quant=False),
            "adapter_refused",
        )
        self.assertEqual(
            hub_trending.classify(GGUF_NEW_ARCH, ok, yes), "format_blocked"
        )
        self.assertEqual(
            hub_trending.classify(QWEN3_BF16, {"status": "gated"}, None), "gated"
        )
        self.assertEqual(
            hub_trending.classify(QWEN3_BF16, {"status": "missing"}, None), "no_config"
        )
        self.assertEqual(hub_trending.classify(QWEN3_BF16, ok, None), "unchecked")

    def test_quantized_config_needs_an_mx_that_checks_quant(self):
        ok, yes = {"status": "ok"}, {"supported": True, "adapter": "qwen3_5"}
        # mlx-community/Qwen3-0.6B-4bit reported supported=true before mx
        # checked quantization_config.
        self.assertEqual(
            hub_trending.classify(MLX_4BIT, ok, yes, mx_checks_quant=False),
            "quant_provisional",
        )
        self.assertEqual(
            hub_trending.classify(MLX_4BIT, ok, yes, mx_checks_quant=True), "supported"
        )
        probe = hub_trending.quant_probe_config(QWEN3_BF16["config"])
        self.assertEqual(hub_trending.quantization({"config": probe}), "mlx_affine_q4")
        self.assertNotIn("quantization_config", QWEN3_BF16["config"])

    def test_compact_drops_template_and_projects_quant(self):
        raw: dict = {**QWEN3_BF16, "siblings": [{"rfilename": "config.json"}]}
        raw["config"] = {
            **raw["config"],
            "chat_template_jinja": "{{ x }}",
            "quantization_config": {
                "quant_method": "fp8",
                "modules_to_not_convert": ["a"],
            },
        }
        out = hub_trending.compact(raw)
        self.assertNotIn("chat_template_jinja", out["config"])
        self.assertEqual(out["config"]["quantization_config"], {"quant_method": "fp8"})
        self.assertEqual(out["files"], ["config.json"])
        self.assertEqual(len(out["chat_template_sha256"]), 64)


class Configs(unittest.TestCase):
    def test_records_gated_and_missing_and_caches_ok(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = pathlib.Path(tmp)
            gated = hub_trending.fetch_config(
                QWEN3_BF16, cache, lambda _: (401, {}, b"")
            )
            self.assertEqual(gated["status"], "gated")
            missing = hub_trending.fetch_config(GATED_GGUF_NO_CONFIG, cache, None)
            self.assertEqual(missing["status"], "missing")
            urls = []
            ok = hub_trending.fetch_config(
                QWEN3_BF16, cache, lambda u: urls.append(u) or (200, {}, b"{}")
            )
            self.assertEqual(ok["status"], "ok")
            self.assertIn(f"/resolve/{QWEN3_BF16['sha']}/config.json", urls[0])
            again = hub_trending.fetch_config(QWEN3_BF16, cache, None)
            self.assertTrue(again["cached"])

    def test_parse_supports_skips_bad_lines(self):
        text = '{"path":"a","supported":true}\nnot json\n{"no_path":1}\n'
        self.assertEqual(list(hub_trending.parse_supports(text)), ["a"])


class Report(unittest.TestCase):
    def table(self):
        models = [
            QWEN3_BF16,
            GGUF_NEW_ARCH,
            MLX_4BIT,
            GATED_GGUF_NO_CONFIG,
            DIFFUSERS,
            CLASSIFIER,
        ]
        fetched = {
            QWEN3_BF16["id"]: {"status": "ok", "path": "p1"},
            GGUF_NEW_ARCH["id"]: {"status": "missing"},
            MLX_4BIT["id"]: {"status": "ok", "path": "p3"},
            GATED_GGUF_NO_CONFIG["id"]: {"status": "missing"},
            DIFFUSERS["id"]: {"status": "ok", "path": "p5"},
        }
        verdicts = {
            "p1": {"supported": True},
            "p3": {"supported": True},
            "p5": {"supported": False, "reason": "no adapter"},
        }
        return models, hub_trending.rows(models, fetched, verdicts)

    def test_gguf_without_config_inherits_supported_family(self):
        _, table = self.table()
        by_id = {r["id"]: r for r in table}
        self.assertEqual(
            by_id[GATED_GGUF_NO_CONFIG["id"]]["category"], "format_blocked"
        )
        self.assertTrue(by_id[GATED_GGUF_NO_CONFIG["id"]]["inferred"])
        self.assertEqual(by_id[GGUF_NEW_ARCH["id"]]["category"], "no_config")
        self.assertNotIn(CLASSIFIER["id"], by_id)

    def test_shares_carry_denominators(self):
        models, table = self.table()
        text = hub_trending.render_report(models, table, {"models": 6, "pages": 1})
        # Text repos: 4, summed score 49 + 398 + 26 + 222 = 695.
        self.assertIn(
            "Denominators: 4 repos, summed trendingScore 695 (head: whole listing).",
            text,
        )
        self.assertIn("| supported | 2 / 4 (50.0%) | 75 / 695 (10.8%) |", text)
        self.assertIn("| format_blocked | 1 / 4 (25.0%) | 222 / 695 (31.9%) |", text)
        self.assertIn("text 4 / 6 (66.7%)", text)
        self.assertIn("| xing40 | 1 | 1 | 398 / 695 (57.3%) |", text)
        self.assertNotIn("| qwen35 |", text)
        self.assertIn("| qwenimage21pipeline | 1 | 0 | 286 / 286 (100.0%) |", text)

    def test_without_verdicts_every_family_is_ranked_as_unknown(self):
        models = [QWEN3_BF16, GGUF_NEW_ARCH]
        fetched = {
            QWEN3_BF16["id"]: {"status": "ok", "path": "p1"},
            GGUF_NEW_ARCH["id"]: {"status": "missing"},
        }
        table = hub_trending.rows(models, fetched, {})
        text = hub_trending.render_report(models, table, {})
        self.assertIn("| unchecked | 1 / 2 (50.0%) | 49 / 447 (11.0%) |", text)
        self.assertIn("All families (check not run, support unknown)", text)
        self.assertNotIn("Unsupported families", text)
        self.assertIn("| qwen3 | 1 | 0 | 49 / 447 (11.0%) |", text)


# Score shape of the 2026-10-06 listing in miniature: a few heavy models,
# then a long tail at exactly 1. Total 665.
HEAD_LISTING = [
    {"id": f"m{i}", "trendingScore": s}
    for i, s in enumerate(
        [1, 400, 150, 60, 30, 4, 3, 3, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    )
]


class Head(unittest.TestCase):
    def test_score_cut_keeps_models_at_or_above_min_score(self):
        head, meta = hub_trending.select_head(HEAD_LISTING, min_score=3)
        self.assertEqual(
            [m["trendingScore"] for m in head], [400, 150, 60, 30, 4, 3, 3]
        )
        self.assertEqual(
            meta,
            {
                "rule": "trendingScore >= 3",
                "size": 7,
                "listing": 20,
                "weight": 650,
                "total_weight": 665,
            },
        )

    def test_coverage_cut_is_the_smallest_prefix_reaching_the_share(self):
        # 400 + 150 = 550 < 0.9 * 665 = 598.5; + 60 = 610 reaches it.
        head, meta = hub_trending.select_head(HEAD_LISTING, weight_coverage=0.9)
        self.assertEqual([m["trendingScore"] for m in head], [400, 150, 60])
        self.assertEqual(meta["weight"], 610)
        head, _ = hub_trending.select_head(HEAD_LISTING, weight_coverage=1.0)
        self.assertEqual(len(head), 20)
        # Reaching the share exactly stops the head there.
        exact = [{"id": c, "trendingScore": s} for c, s in zip("abc", (600, 300, 100))]
        head, _ = hub_trending.select_head(exact, weight_coverage=0.9)
        self.assertEqual([m["id"] for m in head], ["a", "b"])

    def test_all_and_report_note(self):
        head, meta = hub_trending.select_head(HEAD_LISTING, everything=True)
        self.assertEqual((len(head), meta["rule"]), (20, "whole listing"))
        _, meta = hub_trending.select_head(HEAD_LISTING, min_score=3)
        self.assertEqual(
            hub_trending.head_note(meta),
            "head: 7 of 20 models (trendingScore >= 3), 650 / 665 (97.7%) "
            "of listing trendingScore",
        )
        text = hub_trending.render_report([], [], {}, head=meta)
        self.assertIn("Checked head: 7 of 20 models", text)


if __name__ == "__main__":
    unittest.main()
