"""Validate fresh V2 references before admitting a native candidate.

Usage: check_qwen35_affine_reference.py DEV REPEAT_DEV HELDOUT REPEAT_HELDOUT
The repeats use the current pinned backend; older backend captures are not an
exact-output oracle. This checks reference validity, not candidate correctness.
"""

import ast
import json
import math
import sys
from pathlib import Path

CHUNKS = (None, 16, 12, 8, 1, 7, 9, 15, 17, 32, 64, 128)
KEYS = ("centered", "margin", "max_abs")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def declared_prompts(split):
    tree = ast.parse(
        Path(__file__).with_name("qwen35-mlx-affine-reference.py").read_text()
    )
    name = "HELDOUT_PROMPTS" if split == "heldout" else "PROMPTS"
    return next(
        ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == name for t in node.targets)
    )


def validate(ref, split):
    require(ref["mlx"] == "0.32.3" and ref["mlx_lm"] == "0.32.0", "backend pin")
    q = ref["qualification"]
    require(q["bar"] == "v2" and q["steps"] == 32, "V2 requires 32 steps")
    require(
        q["prompts"] == split and q["min_distinct_arms"] == 8, "qualification contract"
    )
    require(q["prefill_chunks"] == list(CHUNKS), "declared chunks")
    cap = q["abort_gpu_bytes"]
    require(isinstance(cap, int) and cap > 0, "memory cap")
    require(
        all(
            0 <= q["memory"][k] <= cap
            for k in ("active_bytes", "peak_bytes", "cache_bytes")
        ),
        "memory bound",
    )
    require(
        q["memory"]["active_bytes"] + q["memory"]["cache_bytes"] <= cap,
        "active plus cache bound",
    )
    require(
        [c["user"] for c in ref["cases"]] == declared_prompts(split),
        "fixed prompt text/order",
    )
    counts = []
    for ci, case in enumerate(ref["cases"]):
        steps = case["steps"]
        require(len(steps) == 32 and len(case["aa"]) == 32, f"case {ci}: steps")
        require(bool(case["prompt_ids"]), f"case {ci}: empty prompt")
        baseline = [s["top8_logits"] for s in steps]
        for step in steps:
            require(len(set(step["top8_ids"])) == 8, "eight distinct IDs")
            values = step["top8_logits"]
            require(
                len(values) == 8 and all(math.isfinite(v) for v in values),
                "finite logits",
            )
            require(values == sorted(values, reverse=True), "sorted logits")
            require(step["top2_gap"] == values[0] - values[1], "top2 gap")
        expected = [
            (gdn, chunk)
            for gdn in ("kernel", "ops")
            for chunk in CHUNKS
            if (gdn, chunk) != ("kernel", None)
        ]
        require(
            [(a["gdn"], a["prefill_chunk"]) for a in case["aa_arms"]] == expected,
            "arm configurations",
        )
        seen = [baseline]
        distinct = []
        for arm in case["aa_arms"]:
            logits = arm["top8_logits"]
            require(len(logits) == 32 and len(arm["steps"]) == 32, "arm steps")
            duplicate = logits in seen
            require(arm["duplicate"] is duplicate, "duplicate classification")
            seen.append(logits)
            for got, want, recorded in zip(logits, baseline, arm["steps"], strict=True):
                require(
                    len(got) == 8 and all(math.isfinite(v) for v in got),
                    "finite arm logits",
                )
                delta = [g - w for g, w in zip(got, want, strict=True)]
                mean = sum(delta) / 8
                errors = (
                    max(abs(d - mean) for d in delta),
                    abs((got[0] - got[1]) - (want[0] - want[1])),
                    max(abs(d) for d in delta),
                )
                require(
                    all(recorded[k] == v for k, v in zip(KEYS, errors, strict=True)),
                    "arm errors",
                )
            if not duplicate:
                distinct.append(arm)
        require(
            len(distinct) >= 8, f"case {ci}: {len(distinct)} distinct arms, require 8"
        )
        for si, floor in enumerate(case["aa"]):
            require(
                all(floor[k] == max(a["steps"][si][k] for a in distinct) for k in KEYS),
                "per-step floor",
            )
        counts.append(len(distinct))
    return counts


def check_references(dev, repeat_dev, heldout, repeat_heldout):
    for split, first, repeat in (
        ("dev", dev, repeat_dev),
        ("heldout", heldout, repeat_heldout),
    ):
        counts = validate(first, split)
        validate(repeat, split)
        for key in (
            "mlx",
            "mlx_lm",
            "config_sha256",
            "weights_sha256",
            "components",
            "cases",
        ):
            require(
                first[key] == repeat[key], f"{split}: fresh repeat differs at {key}"
            )
        print(
            f"PASS {split}: fresh repeat exact; distinct arms per case {counts} of 23"
        )
    for key in ("config_sha256", "weights_sha256"):
        require(dev[key] == heldout[key], f"checkpoint differs: {key}")
    dev_ids = {tuple(c["prompt_ids"]) for c in dev["cases"]}
    require(
        not any(tuple(c["prompt_ids"]) in dev_ids for c in heldout["cases"]),
        "heldout/dev overlap",
    )


def main():
    if len(sys.argv) != 5:
        raise SystemExit(__doc__)
    try:
        check_references(*(json.loads(Path(p).read_text()) for p in sys.argv[1:]))
    except (ValueError, KeyError, TypeError, IndexError) as error:
        raise SystemExit(f"INVALID reference: {error}") from error


if __name__ == "__main__":
    main()
