#!/usr/bin/env python3
"""Operator search adapter. Set SEARXNG_URL; enable the instance's JSON format."""

import argparse
import json
import os
import sys
from urllib.parse import urlencode
from urllib.request import Request, urlopen


def run(health=False):
    base = os.environ["SEARXNG_URL"].rstrip("/")
    if health:
        with urlopen(base + "/healthz", timeout=8) as response:
            if response.status != 200:
                raise ValueError("health failed")
        return
    request = json.load(sys.stdin)
    query = request["query"]
    if request.get("allowed_domains"):
        query += (
            " (" + " OR ".join("site:" + d for d in request["allowed_domains"]) + ")"
        )
    query += "".join(" -site:" + d for d in request.get("blocked_domains", []))
    body = urlencode({"q": query, "format": "json"}).encode()
    with urlopen(Request(base + "/search", data=body), timeout=8) as response:
        raw = response.read(1_048_577)
    if len(raw) > 1_048_576:
        raise ValueError("provider response too large")
    result = json.loads(raw)
    json.dump(
        [
            {
                "url": item["url"],
                "title": item.get("title", ""),
                "snippet": item.get("content", ""),
            }
            for item in result["results"][: min(request["max_results"], 5)]
        ],
        sys.stdout,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--health", action="store_true")
    args = parser.parse_args()
    try:
        run(args.health)
    except (OSError, ValueError, KeyError, TypeError):
        print("search adapter failed", file=sys.stderr)
        raise SystemExit(1) from None
