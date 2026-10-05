#!/usr/bin/env python3
"""Write the routed experts the DeepSeek V4.1 test captures use as a pin file.

The route-trace fetcher (crates/server/src/range_fetch.rs) never evicts an
expert listed in `<trace>/pinned-experts.json`, so the end-to-end capture
tests never re-download theirs. Each recorded run's `routes[].ids` holds, per
layer, one row of selected expert ids per token.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# The recorded runs behind capture-shell2 and capture-parity3.
DEFAULT_ROUTES = ("parity-shell.json", "parity-greedy3.json")


def pinned_experts(route_files: list[Path]) -> list[list[int]]:
    experts: set[tuple[int, int]] = set()
    for path in route_files:
        for run in json.loads(path.read_text())["runs"]:
            for entry in run["routes"]:
                for row in entry["ids"]:
                    experts.update((entry["layer"], expert) for expert in row)
    return [list(expert) for expert in sorted(experts)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace_dir", type=Path, help="route-trace directory")
    parser.add_argument(
        "routes",
        nargs="*",
        default=DEFAULT_ROUTES,
        help=f"route files in trace_dir (default: {' '.join(DEFAULT_ROUTES)})",
    )
    args = parser.parse_args()
    experts = pinned_experts([args.trace_dir / name for name in args.routes])
    pin_file = {"sources": list(args.routes), "experts": experts}
    out = args.trace_dir / "pinned-experts.json"
    out.write_text(json.dumps(pin_file, separators=(",", ":")) + "\n")
    print(f"{out}: {len(experts)} experts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
