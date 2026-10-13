#!/usr/bin/env python3
"""Deterministic adapter; WEB_SEARCH_FAKE_STATE names an operator-owned JSON file.

State fields: healthy (bool), delay_seconds (number), exit_code (int),
invalid_output (bool), output_bytes (int), results (array). No external I/O.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path


def run(health=False):
    state = json.loads(Path(os.environ["WEB_SEARCH_FAKE_STATE"]).read_text())
    if health:
        return 0 if state.get("healthy", True) else 1
    request = json.load(sys.stdin)
    assert isinstance(request["query"], str)
    time.sleep(state.get("delay_seconds", 0))
    if state.get("exit_code", 0):
        return state["exit_code"]
    if state.get("invalid_output"):
        print("invalid JSON")
    elif "output_bytes" in state:
        sys.stdout.write("x" * state["output_bytes"])
    else:
        json.dump(
            state.get(
                "results",
                [
                    {
                        "url": "https://example.com/fact",
                        "title": "Fixture",
                        "snippet": "The fixture value is blue.",
                    }
                ],
            ),
            sys.stdout,
        )
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--health", action="store_true")
    raise SystemExit(run(parser.parse_args().health))
