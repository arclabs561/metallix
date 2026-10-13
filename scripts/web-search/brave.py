#!/usr/bin/env python3
"""Operator search adapter. BRAVE_API_KEY required; --health spends one query."""

import argparse
import json
import os
import sys
from urllib.parse import urlencode
from urllib.request import Request, urlopen


def run(health=False):
    key = os.environ["BRAVE_API_KEY"]
    if not key:
        raise ValueError("missing key")
    request = (
        {"query": "health check", "max_results": 1} if health else json.load(sys.stdin)
    )
    query = request["query"]
    if request.get("allowed_domains"):
        query += (
            " (" + " OR ".join("site:" + d for d in request["allowed_domains"]) + ")"
        )
    query += "".join(" -site:" + d for d in request.get("blocked_domains", []))
    url = "https://api.search.brave.com/res/v1/web/search?" + urlencode(
        {
            "q": query,
            "count": min(request["max_results"], 5),
        }
    )
    with urlopen(
        Request(
            url, headers={"X-Subscription-Token": key, "Accept": "application/json"}
        ),
        timeout=8,
    ) as response:
        raw = response.read(1_048_577)
    if len(raw) > 1_048_576:
        raise ValueError("provider response too large")
    results = json.loads(raw).get("web", {}).get("results", [])
    if not health:
        json.dump(
            [
                {
                    "url": item["url"],
                    "title": item.get("title", ""),
                    "snippet": item.get("description", ""),
                }
                for item in results[: min(request["max_results"], 5)]
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
