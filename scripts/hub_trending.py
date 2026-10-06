# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Which trending Hugging Face models can metallix load, and what to add next.

Phases, each reading and writing one run directory:

  list     page https://huggingface.co/api/models?sort=trendingScore until the
           trending score drops to 0 (or --max-pages), and write
           listing.jsonl plus listing_meta.json
  configs  pick the head of the listing (--min-score, --weight-coverage or
           --all; recorded in head.json), then download config.json
           (model_index.json for diffusers pipelines) of its text-generation,
           vision-language and diffusion repos into a cache keyed by repo and
           commit; record gated and missing files
  check    run `mx inspect supports --json` over the cached files
  report   write report.md: supported shares, blockers and the unsupported
           families ranked by summed trending score
  all      every phase in order (check only with --mx)

The parsers and classifiers take decoded data so tests feed recorded Hub
objects; HTTP and subprocess calls stay in thin wrappers around them. Never
downloads weights. An optional token comes from HF_TOKEN or the Hugging Face
CLI token file and is sent only as an Authorization header.
"""

from __future__ import annotations

import argparse
import collections
import concurrent.futures
import email.message
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterable
from contextlib import AbstractContextManager
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

ROOT = Path(__file__).resolve().parent.parent
HUB = "https://huggingface.co"
EXPAND = (
    "config",
    "pipeline_tag",
    "trendingScore",
    "gated",
    "safetensors",
    "gguf",
    "tags",
    "library_name",
    "createdAt",
    "downloads",
    "likes",
    "siblings",
    "sha",
)
TEXT_TASKS = {"text-generation", "image-text-to-text", "any-to-any"}
DIFFUSION_TASKS = {"text-to-image", "image-to-image", "text-to-video", "image-to-video"}
DIFFUSION_LIBRARIES = {"diffusers", "diffusion-single-file"}
# Projection of quantization_config kept in the snapshot; the full object can
# list thousands of module names.
QUANT_KEYS = ("quant_method", "bits", "group_size", "fmt", "weight_block_size")
CATEGORIES = (
    "supported",
    "quant_provisional",
    "format_blocked",
    "adapter_refused",
    "unsupported_arch",
    "gated",
    "no_config",
    "unchecked",
)


# ---------------------------------------------------------------------------
# Listing


def next_link(header: str | None) -> str | None:
    """The rel="next" target of an RFC 8288 Link header, or None."""
    for match in re.finditer(r"<([^>]*)>\s*;\s*([^,<]*)", header or ""):
        if re.search(r'rel\s*=\s*"?next"?', match.group(2)):
            return match.group(1)
    return None


def listing_url(limit: int) -> str:
    query = [("sort", "trendingScore"), ("limit", str(limit))]
    query += [("expand[]", field) for field in EXPAND]
    return f"{HUB}/api/models?{urllib.parse.urlencode(query)}"


def take_trending(models: list[dict], min_score: float) -> tuple[list[dict], bool]:
    """Models scoring above `min_score`, and whether the listing ended here.

    The listing is sorted by score, then continues through every model on the
    Hub with score 0 or null, so the first model at or below the threshold
    marks the end of the trending set.
    """
    kept = []
    for model in models:
        score = model.get("trendingScore")
        if score is None or score <= min_score:
            return kept, True
        kept.append(model)
    return kept, False


def compact(model: dict) -> dict:
    """The listing fields the later phases read, without chat templates."""
    config = dict(model.get("config") or {})
    template = config.pop("chat_template_jinja", None)
    config.pop("tokenizer_config", None)
    quant = config.get("quantization_config")
    if isinstance(quant, dict):
        config["quantization_config"] = {k: quant[k] for k in QUANT_KEYS if k in quant}
    safetensors = model.get("safetensors") or {}
    gguf = model.get("gguf") or {}
    return {
        "id": model["id"],
        "sha": model.get("sha"),
        "trendingScore": model.get("trendingScore"),
        "pipeline_tag": model.get("pipeline_tag"),
        "library_name": model.get("library_name"),
        "tags": model.get("tags") or [],
        "gated": model.get("gated"),
        "createdAt": model.get("createdAt"),
        "downloads": model.get("downloads"),
        "likes": model.get("likes"),
        "config": config,
        "chat_template_sha256": template
        and hashlib.sha256(template.encode()).hexdigest(),
        "safetensors": {
            k: safetensors[k] for k in ("parameters", "total") if k in safetensors
        },
        "gguf": {k: gguf[k] for k in ("architecture", "total") if k in gguf},
        "files": [s["rfilename"] for s in model.get("siblings") or []],
    }


# ---------------------------------------------------------------------------
# HTTP


class Response(Protocol):
    """The part of urlopen's response that fetch() reads."""

    @property
    def status(self) -> int: ...

    @property
    def headers(self) -> email.message.Message: ...

    def read(self) -> bytes: ...


# urlopen(request, timeout=...)
Opener = Callable[..., AbstractContextManager[Response]]


def hub_token() -> str | None:
    token = os.environ.get("HF_TOKEN", "").strip()
    if token:
        return token
    try:
        return (Path.home() / ".cache/huggingface/token").read_text().strip() or None
    except OSError:
        return None


def retry_after(headers, default: float) -> float:
    """Seconds to wait from a Retry-After header (delta-seconds form only)."""
    try:
        return max(0.0, float(headers.get("Retry-After")))
    except (TypeError, ValueError):
        return default


def fetch(
    url: str,
    *,
    opener: Opener = urllib.request.urlopen,
    sleep: Callable[[float], None] = time.sleep,
    token: str | None = None,
    retries: int = 4,
    backoff: float = 5.0,
    timeout: float = 15.0,
) -> tuple[int, dict, bytes]:
    """GET `url`; returns (status, headers, body) for any final status.

    The timeout also bounds each connect attempt: some IPv6 edge addresses
    for huggingface.co drop SYNs, and urllib tries addresses in turn without
    happy eyeballs, so a long timeout stalls on every such address.

    429 waits for Retry-After and 5xx or network errors back off, each up to
    `retries` times; other HTTP errors (401, 403, 404) return at once so the
    caller can record them.
    """
    headers = {"User-Agent": "metallix-hub-trending"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    for attempt in range(retries + 1):
        request = urllib.request.Request(url, headers=headers)
        try:
            with opener(request, timeout=timeout) as response:
                return response.status, dict(response.headers), response.read()
        except urllib.error.HTTPError as error:
            transient = error.code == 429 or error.code >= 500
            if not transient or attempt == retries:
                return error.code, dict(error.headers or {}), b""
            wait = backoff * 2**attempt
            if error.code == 429:
                wait = retry_after(error.headers or {}, wait)
            sleep(wait)
        except (urllib.error.URLError, TimeoutError):
            if attempt == retries:
                raise
            sleep(backoff * 2**attempt)
    raise AssertionError("unreachable")


def list_trending(
    *,
    min_score: float,
    max_pages: int,
    limit: int = 100,
    pace: float = 0.7,
    fetcher: Callable[[str], tuple[int, dict, bytes]],
    sleep: Callable[[float], None] = time.sleep,
) -> tuple[list[dict], dict]:
    """Every trending model, and how paging stopped."""
    url, models, pages = listing_url(limit), [], 0
    stop = "end_of_listing"
    while url:
        if pages == max_pages:
            stop = "max_pages"
            break
        if pages:
            sleep(pace)
        status, headers, body = fetcher(url)
        if status != 200:
            stop = f"http_{status}"
            break
        pages += 1
        kept, ended = take_trending(json.loads(body), min_score)
        models += kept
        if ended:
            stop = "min_score"
            break
        link = {k.lower(): v for k, v in headers.items()}.get("link")
        url = next_link(link)
    return models, {"pages": pages, "models": len(models), "stop": stop}


# ---------------------------------------------------------------------------
# Classification


def kind(model: dict) -> str:
    """Scope: "text", "diffusion" or "other", from pipeline tag, library and tags."""
    task = model.get("pipeline_tag")
    library = model.get("library_name")
    tags = set(model.get("tags") or [])
    if task in TEXT_TASKS:
        return "text"
    if library in DIFFUSION_LIBRARIES or "diffusers" in tags:
        return "diffusion"
    if task in DIFFUSION_TASKS and "model_index.json" in model.get("files", []):
        return "diffusion"
    if task is None:
        archs = (model.get("config") or {}).get("architectures") or []
        if any(a.endswith(("ForCausalLM", "ForConditionalGeneration")) for a in archs):
            return "text"
    return "other"


def family(model: dict) -> str:
    """One key per architecture family.

    model_type (or the GGUF architecture) with case, underscores, dashes and
    dots removed, so qwen3_5 and GGUF qwen35 match, and a trailing "text"
    folded into the multimodal parent (gemma4_text -> gemma4). Diffusers
    pipelines key on their pipeline class. Repos with none of these (LoRAs,
    single-file checkpoints) key on their declared base model, "base:<repo>".
    """
    config = model.get("config") or {}
    gguf_arch = (model.get("gguf") or {}).get("architecture")
    # The Hub can report "clip", the architecture of a vision projector
    # (mmproj) GGUF shipped beside the language model; all 34 such repos on
    # 2026-10-06 had one. It says nothing about the language model.
    name = (
        config.get("model_type")
        or (gguf_arch if gguf_arch != "clip" else None)
        or (config.get("diffusers") or {}).get("_class_name")
        or next(iter(config.get("architectures") or []), "")
    )
    key = re.sub(r"[_\-.]", "", str(name).lower())
    if key.endswith("text") and len(key) > 6:
        key = key[:-4]
    if key:
        return key
    for tag in model.get("tags") or []:
        # base_model:<repo> or base_model:<relation>:<repo>
        if tag.startswith("base_model:"):
            return "base:" + tag.split(":")[-1]
    return ""


def weight_format(model: dict) -> str:
    """The weight files a repo ships, from its file list and listing metadata.

    "safetensors" (any dtype, quantized or not), "gguf_only", "pytorch_only"
    (.bin/.pt/.pth), "onnx_only", or "none" when no weights are listed.
    """
    files = model.get("files") or []
    if any(f.endswith(".safetensors") for f in files) or model.get("safetensors"):
        return "safetensors"
    if any(f.endswith(".gguf") for f in files) or model.get("gguf"):
        return "gguf_only"
    if any(f.endswith((".bin", ".pt", ".pth")) for f in files):
        return "pytorch_only"
    if any(f.endswith(".onnx") for f in files):
        return "onnx_only"
    return "none"


def quantization(model: dict) -> str | None:
    """quant_method from quantization_config; MLX affine configs carry only bits."""
    quant = (model.get("config") or {}).get("quantization_config") or {}
    if quant.get("quant_method"):
        return str(quant["quant_method"]).lower()
    if "bits" in quant:
        return f"mlx_affine_q{quant['bits']}"
    return None


def config_file(model: dict) -> str | None:
    """Which file the check phase needs, or None when the repo has none."""
    files = model.get("files")
    wanted = "model_index.json" if kind(model) == "diffusion" else "config.json"
    if files is None or wanted in files:
        return wanted
    return None


def classify(
    model: dict,
    fetched: dict | None,
    verdict: dict | None,
    mx_checks_quant: bool = True,
) -> str:
    """One category from CATEGORIES for an in-scope repo.

    `fetched` is the configs-phase record and `verdict` the `mx inspect
    supports` line for its file, bucketed on (supported, adapter): no
    adapter is an unsupported architecture; an adapter that refuses is a
    format blocker when the config is quantized and mx judges quantization,
    and adapter_refused (a variant, or an adapter no serve kind loads)
    otherwise. Weight files come from
    the Hub file list, because config.json cannot say that the only weights
    are GGUF or pickled. When `mx_checks_quant` is false (the check phase
    probe found mx accepting a quantized config), a supported verdict on a
    quantized config is quant_provisional: the architecture loads, the
    quantization was not judged.
    """
    status = (fetched or {}).get("status")
    if status == "gated":
        return "gated"
    if status != "ok":
        return "no_config"
    if verdict is None or "supported" not in verdict:
        return "unchecked"
    if not verdict["supported"]:
        if not verdict.get("adapter"):
            return "unsupported_arch"
        # Before mx judged quantization, a refusal was never about it.
        quant_refused = quantization(model) and mx_checks_quant
        return "format_blocked" if quant_refused else "adapter_refused"
    if weight_format(model) != "safetensors":
        return "format_blocked"
    if quantization(model) and not mx_checks_quant:
        return "quant_provisional"
    return "supported"


# ---------------------------------------------------------------------------
# Configs


def cache_path(cache: Path, repo: str, sha: str | None, name: str) -> Path:
    return cache / repo.replace("/", "__") / (sha or "unknown") / name


def fetch_config(model: dict, cache: Path, fetcher) -> dict:
    """Download the repo's config file into the cache; never raises on HTTP."""
    record = {"id": model["id"], "sha": model.get("sha")}
    name = config_file(model)
    if name is None:
        return record | {"status": "missing", "file": None, "reason": "not in repo"}
    path = cache_path(cache, model["id"], model.get("sha"), name)
    record |= {"file": name, "path": str(path)}
    if path.exists():
        return record | {"status": "ok", "cached": True}
    revision = model.get("sha") or "main"
    status, _, body = fetcher(f"{HUB}/{model['id']}/resolve/{revision}/{name}")
    if status in (401, 403):
        return record | {"status": "gated", "http": status}
    if status == 404:
        return record | {"status": "missing", "http": status}
    if status != 200:
        return record | {"status": "error", "http": status}
    try:
        json.loads(body)
    except ValueError:
        return record | {"status": "error", "reason": "not JSON"}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(body)
    return record | {"status": "ok", "http": status}


# ---------------------------------------------------------------------------
# Check


def parse_supports(text: str) -> dict[str, dict]:
    """`mx inspect supports --json` lines keyed by path; bad lines are skipped."""
    out = {}
    for line in text.splitlines():
        try:
            record = json.loads(line)
        except ValueError:
            continue
        if isinstance(record, dict) and "path" in record:
            out[record["path"]] = record
    return out


def has_supports(mx: str) -> bool:
    result = subprocess.run(
        [mx, "inspect", "supports", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.returncode == 0


def run_supports(mx: str, paths: list[str], batch: int = 200) -> dict[str, dict]:
    out = {}
    for start in range(0, len(paths), batch):
        chunk = paths[start : start + batch]
        result = subprocess.run(
            [mx, "inspect", "supports", "--json", *chunk],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode not in (0, 2):
            sys.exit(f"mx inspect supports exited {result.returncode}: {result.stderr}")
        out |= parse_supports(result.stdout)
    return out


# ---------------------------------------------------------------------------
# Report


def share(part: float, whole: float) -> str:
    return (
        f"{part:g} / {whole:g} ({100 * part / whole:.1f}%)"
        if whole
        else f"{part:g} / 0"
    )


def rows(models, fetched, verdicts, mx_checks_quant: bool = True) -> list[dict]:
    """One flat row per in-scope model with its category.

    Repos without a config file (most GGUF-only quantizations) get no mx
    verdict; when their family has a supported repo in the same listing they
    count as format_blocked, marked inferred.
    """
    out = []
    for model in models:
        scope = kind(model)
        if scope == "other":
            continue
        record = fetched.get(model["id"])
        verdict = verdicts.get((record or {}).get("path"))
        out.append(
            {
                "id": model["id"],
                "kind": scope,
                "family": family(model),
                "score": model.get("trendingScore") or 0,
                "format": weight_format(model),
                "quant": quantization(model),
                "category": classify(model, record, verdict, mx_checks_quant),
                "adapter": (verdict or {}).get("adapter"),
                "reason": (verdict or {}).get("reason") or (record or {}).get("reason"),
                "inferred": False,
            }
        )
    supported = {
        r["family"]
        for r in out
        if r["category"] in ("supported", "quant_provisional", "format_blocked")
    }
    for r in out:
        blocked_format = r["format"] != "safetensors" and r["family"] in supported
        if r["category"] == "no_config" and r["family"] and blocked_format:
            r["category"], r["inferred"] = "format_blocked", True
    return out


def select_head(
    models: list[dict],
    *,
    min_score: float = 3,
    weight_coverage: float | None = None,
    everything: bool = False,
) -> tuple[list[dict], dict]:
    """The head of the listing that configs, check and components work on.

    Trending weight is concentrated: on 2026-10-06, 6,703 of 10,286 models
    had score exactly 1 and the top 1,000 held 68% of the summed score. The
    head is every model scoring at least `min_score`, or, with
    `weight_coverage`, the smallest score-ordered prefix whose summed score
    reaches that share of the total, or the whole listing with `everything`.
    """
    ranked = sorted(models, key=lambda m: -(m.get("trendingScore") or 0))
    scores = [m.get("trendingScore") or 0 for m in ranked]
    total = sum(scores)
    if everything:
        head, rule = ranked, "whole listing"
    elif weight_coverage is not None:
        size, reached = 0, 0.0
        while size < len(ranked) and reached < weight_coverage * total:
            reached += scores[size]
            size += 1
        head, rule = ranked[:size], f"smallest head with {weight_coverage:g} of weight"
    else:
        head = [m for m, s in zip(ranked, scores) if s >= min_score]
        rule = f"trendingScore >= {min_score:g}"
    weight = sum(m.get("trendingScore") or 0 for m in head)
    meta = {
        "rule": rule,
        "size": len(head),
        "listing": len(models),
        "weight": weight,
        "total_weight": total,
    }
    return head, meta


def head_note(head: dict) -> str:
    """The head's size and weight share, printed next to every percentage."""
    if not head:
        return "head: whole listing"
    return (
        f"head: {head['size']} of {head['listing']} models ({head['rule']}), "
        f"{share(head['weight'], head['total_weight'])} of listing trendingScore"
    )


def render_report(
    models: list[dict],
    table: list[dict],
    meta: dict,
    top: int = 25,
    head: dict | None = None,
) -> str:
    """Markdown summary; every count and share names its denominator.

    `models` is the head of the listing; `head` describes how it was cut."""
    note = head_note(head or {})
    lines = [
        "# Trending Hub models vs metallix",
        "",
        (
            f"Listing fetched {meta.get('fetched_at', '?')}: "
            f"{meta.get('models', len(models))} models with trendingScore > "
            f"{meta.get('min_score', 0)} over {meta.get('pages', '?')} pages "
            f"(stop: {meta.get('stop', '?')})."
        ),
        "",
        f"Checked {note}.",
        "",
    ]
    kinds = collections.Counter(kind(m) for m in models)
    lines.append(
        "In scope: "
        + ", ".join(
            f"{k} {share(kinds[k], len(models))}" for k in ("text", "diffusion")
        )
        + f"; other tasks {share(kinds['other'], len(models))} are out of scope."
    )
    if meta.get("mx_checks_quant") is False:
        lines += [
            "",
            (
                "mx accepted a quantized probe config, so its verdicts ignore "
                "quantization: supported verdicts on quantized configs are "
                "listed as quant_provisional."
            ),
        ]
    for scope in ("text", "diffusion"):
        subset = [r for r in table if r["kind"] == scope]
        if not subset:
            continue
        total_n, total_w = len(subset), sum(r["score"] for r in subset)
        lines += [
            "",
            f"## {scope.capitalize()} models",
            "",
            f"Denominators: {total_n} repos, summed trendingScore {total_w:g} ({note}).",
            "",
            "| Category | Repos | trendingScore |",
            "|---|---:|---:|",
        ]
        for category in CATEGORIES:
            chosen = [r for r in subset if r["category"] == category]
            lines.append(
                f"| {category} | {share(len(chosen), total_n)} | "
                f"{share(sum(r['score'] for r in chosen), total_w)} |"
            )
        blocked = [r for r in subset if r["category"] == "format_blocked"]
        if blocked:
            formats = collections.Counter(r["format"] for r in blocked)
            lines += [
                "",
                f"Format blockers ({len(blocked)} repos with a supported architecture, "
                f"{sum(r['inferred'] for r in blocked)} of them inferred from family "
                "because the repo has no config): "
                + ", ".join(f"{f} {n}" for f, n in formats.most_common())
                + ".",
            ]
        loads = [
            r for r in subset if r["category"] in ("supported", "quant_provisional")
        ]
        quantized = [r for r in loads if r["quant"]]
        if quantized:
            methods = collections.Counter(r["quant"] for r in quantized)
            lines.append(
                f"Quantized configs among repos whose architecture loads "
                f"({len(quantized)} of {len(loads)}): "
                + ", ".join(f"{q} {n}" for q, n in methods.most_common())
                + "."
            )
        lines += ["", *family_table(subset, top, note)]
    return "\n".join(lines) + "\n"


def family_table(subset: list[dict], top: int, note: str = "") -> list[str]:
    """Families with no supported or format-blocked repo, by summed score.

    Without any mx verdict (check not run) support is unknown, so the table
    ranks every family by demand and says so.
    """
    verdict_categories = (
        "supported",
        "quant_provisional",
        "format_blocked",
        "adapter_refused",
        "unsupported_arch",
    )
    checked = any(r["category"] in verdict_categories for r in subset)
    # A family with any adapter verdict is not a new architecture.
    covered = {r["family"] for r in subset if r["category"] in verdict_categories[:4]}
    open_rows = [
        r
        for r in subset
        if r["family"] not in covered and (r["category"] != "unchecked" or not checked)
    ]
    total_w = sum(r["score"] for r in subset)
    by = collections.defaultdict(list)
    for r in open_rows:
        by[r["family"] or "(unknown)"].append(r)
    ranked = sorted(by.items(), key=lambda kv: -sum(r["score"] for r in kv[1]))
    heading = (
        "Unsupported families"
        if checked
        else "All families (check not run, support unknown)"
    )
    out = [
        (
            f"{heading} ranked by summed trendingScore "
            f"(share of {total_w:g}; {len(ranked)} families, "
            f"top {min(top, len(ranked))} shown; {note or 'head: whole listing'}):"
        ),
        "",
        "| Family | Repos | Gated/no config | trendingScore | Top repo |",
        "|---|---:|---:|---:|---|",
    ]
    for name, members in ranked[:top]:
        weight = sum(r["score"] for r in members)
        blind = sum(r["category"] in ("gated", "no_config") for r in members)
        best = max(members, key=lambda r: r["score"])["id"]
        out.append(
            f"| {name} | {len(members)} | {blind} | {share(weight, total_w)} | "
            f"[{best}]({HUB}/{best}) |"
        )
    return out


# ---------------------------------------------------------------------------
# I/O


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def write_jsonl(path: Path, records: Iterable[dict]) -> None:
    path.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in records))


def resolve_run(out_root: Path, run: str | None) -> Path:
    if run and run != "latest":
        return Path(run)
    runs = sorted(p for p in out_root.glob("*") if (p / "listing.jsonl").exists())
    if not runs:
        sys.exit(f"no runs under {out_root}; run the list phase first")
    return runs[-1]


def make_fetcher(token: str | None, label: str):
    """fetch() with the token bound and a request count on stderr."""
    count = 0

    def fetcher(url: str):
        nonlocal count
        count += 1
        print(f"{label} request {count}", file=sys.stderr, flush=True)
        return fetch(url, token=token)

    return fetcher


def phase_list(args) -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run = Path(args.run) if args.run and args.run != "latest" else args.out_root / stamp
    run.mkdir(parents=True, exist_ok=True)
    models, meta = list_trending(
        min_score=0,
        max_pages=args.max_pages,
        pace=args.pace,
        fetcher=make_fetcher(hub_token(), "listing"),
    )
    meta |= {
        "fetched_at": stamp,
        "min_score": 0,
        "max_pages": args.max_pages,
    }
    write_jsonl(run / "listing.jsonl", (compact(m) for m in models))
    (run / "listing_meta.json").write_text(json.dumps(meta, indent=1) + "\n")
    print(f"{run}: {meta['models']} models, {meta['pages']} pages, stop {meta['stop']}")
    return run


def phase_configs(run: Path, args) -> None:
    """Config downloads count against the Hub's "resolvers" quota (3000 per
    5 minutes on 2026-10-06), separate from the API quota, so a few paced
    workers stay well under it."""
    fetcher = make_fetcher(hub_token(), "config")
    head, head_meta = select_head(
        read_jsonl(run / "listing.jsonl"),
        min_score=args.min_score,
        weight_coverage=args.weight_coverage,
        everything=args.all,
    )
    (run / "head.json").write_text(json.dumps(head_meta, indent=1) + "\n")
    print(head_note(head_meta))
    models = [m for m in head if kind(m) != "other"]

    def one(model: dict) -> dict:
        record = fetch_config(model, args.cache, fetcher)
        if "http" in record:
            time.sleep(args.pace)
        return record

    with concurrent.futures.ThreadPoolExecutor(args.config_workers) as pool:
        records = list(pool.map(one, models))
    write_jsonl(run / "configs.jsonl", records)
    counts = collections.Counter(r["status"] for r in records)
    fetched = sum("http" in r for r in records)
    print(f"configs: {dict(counts)} over {len(records)} repos ({fetched} requests)")


def phase_check(run: Path, args) -> bool:
    if not has_supports(args.mx):
        print(
            f"{args.mx} has no `inspect supports` subcommand; skipping check",
            file=sys.stderr,
        )
        return False
    paths = [
        r["path"] for r in read_jsonl(run / "configs.jsonl") if r["status"] == "ok"
    ]
    verdicts = run_supports(args.mx, paths)
    write_jsonl(run / "supports.jsonl", verdicts.values())
    checks_quant = probe_quant_check(args.mx, run, verdicts)
    (run / "supports_meta.json").write_text(
        json.dumps({"mx": args.mx, "mx_checks_quant": checks_quant}) + "\n"
    )
    print(
        f"check: {len(verdicts)} verdicts for {len(paths)} files; "
        f"mx checks quantization: {checks_quant}"
    )
    return True


def quant_probe_config(config: dict) -> dict:
    """A supported unquantized config with an MLX affine quantization added."""
    return config | {"quantization_config": {"bits": 4, "group_size": 64}}


def probe_quant_check(mx: str, run: Path, verdicts: dict[str, dict]) -> bool:
    """Whether mx refuses a quantized copy of a config it accepts.

    Decided by asking mx, not from a list: `inspect supports` gained the
    quantization check after it first shipped. False when no supported
    unquantized config is available to probe.
    """
    for path, verdict in verdicts.items():
        if not verdict.get("supported"):
            continue
        config = json.loads(Path(path).read_text())
        if "quantization_config" in config or "quantization" in config:
            continue
        probe = run / "quant_probe" / "config.json"
        probe.parent.mkdir(exist_ok=True)
        probe.write_text(json.dumps(quant_probe_config(config)))
        answer = run_supports(mx, [str(probe)]).get(str(probe), {})
        return answer.get("supported") is False
    return False


def load_table(run: Path):
    """Head of the listing, listing meta, the categorized rows, config
    records by repo, whether mx gave verdicts, and the head description.

    The head is the one the configs phase recorded in head.json, so the
    report covers exactly what was fetched and checked."""
    listing = read_jsonl(run / "listing.jsonl")
    meta = json.loads((run / "listing_meta.json").read_text())
    try:
        head = json.loads((run / "head.json").read_text())
    except OSError:
        head = None
    if head and head["size"] < len(listing):
        models, _ = select_head(listing, everything=True)
        models = models[: head["size"]]
    else:
        models = listing
    fetched = {r["id"]: r for r in read_jsonl(run / "configs.jsonl")}
    verdicts = {r["path"]: r for r in read_jsonl(run / "supports.jsonl")}
    try:
        supports_meta = json.loads((run / "supports_meta.json").read_text())
    except OSError:
        supports_meta = {}
    checks_quant = supports_meta.get("mx_checks_quant", False)
    if verdicts:
        meta["mx_checks_quant"] = checks_quant
    table = rows(models, fetched, verdicts, checks_quant)
    return models, meta, table, fetched, bool(verdicts), head


def phase_report(run: Path, args) -> None:
    models, meta, table, _, _, head = load_table(run)
    write_jsonl(run / "rows.jsonl", table)
    text = render_report(models, table, meta, args.top, head)
    (run / "report.md").write_text(text)
    print(text)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("phase", choices=("list", "configs", "check", "report", "all"))
    parser.add_argument(
        "--out-root", type=Path, default=ROOT / "artifacts/hub-trending"
    )
    parser.add_argument(
        "--run", help="run directory, or 'latest' (default for later phases)"
    )
    parser.add_argument(
        "--cache", type=Path, default=ROOT / "artifacts/hub-trending/cache"
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=3,
        help="head of the listing for configs/check/components: score >= this",
    )
    parser.add_argument(
        "--weight-coverage",
        type=float,
        help="instead of --min-score: smallest head with this share of total score",
    )
    parser.add_argument(
        "--all", action="store_true", help="work on the whole listing, tail included"
    )
    # The positive-score tail held ~10k models on 2026-10-06.
    parser.add_argument("--max-pages", type=int, default=300)
    parser.add_argument(
        "--pace", type=float, default=0.7, help="seconds between requests"
    )
    parser.add_argument(
        "--config-workers", type=int, default=4, help="parallel config downloads"
    )
    parser.add_argument("--mx", help="mx binary for the check phase")
    parser.add_argument(
        "--top", type=int, default=25, help="families in the ranked table"
    )
    args = parser.parse_args()
    if args.weight_coverage is not None and not 0 < args.weight_coverage <= 1:
        parser.error("--weight-coverage must be in (0, 1]")
    if args.all and args.weight_coverage is not None:
        parser.error("--all and --weight-coverage are exclusive")
    if args.phase in ("list", "all"):
        run = phase_list(args)
    else:
        run = resolve_run(args.out_root, args.run)
    if args.phase in ("configs", "all"):
        phase_configs(run, args)
    if args.phase == "check" or (args.phase == "all" and args.mx):
        if not args.mx:
            parser.error("check needs --mx")
        if not phase_check(run, args) and args.phase == "check":
            return 3
    if args.phase in ("report", "all"):
        phase_report(run, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
