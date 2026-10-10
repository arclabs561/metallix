"""Check that every model crate has an architecture diagram and that each
committed SVG under docs/arch is what its Typst source renders.

A diagram is docs/arch/<kind>.typ with its rendering <kind>.svg and its
numbers <kind>.facts.json. A test in the model's crate writes the facts file
from the crate's own config parsing and names the crate in its `crate` field;
that field is how this check knows which crates are drawn.

Rendering needs typst 0.15.1. Without it the SVG comparison is skipped with a
notice, unless --require-typst is given (as in CI). --write re-renders every
SVG instead of comparing.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TYPST_VERSION = "0.15.1"
RENDER_TIMEOUT_SECONDS = 120

# Model crates not yet drawn. Remove a crate in the change that adds its
# diagram; never add one.
PENDING: frozenset[str] = frozenset(
    {"deepseek", "flux2", "gemma", "julia", "qwen3-asr", "qwen35"}
)


def diagrams(arch: Path) -> list[Path]:
    """Every diagram source in `arch`; lib.typ holds shared blocks only."""
    return sorted(path for path in arch.glob("*.typ") if path.name != "lib.typ")


def model_crates(root: Path) -> set[str]:
    """Directory names of the crates under crates/models."""
    return {path.parent.name for path in (root / "crates/models").glob("*/Cargo.toml")}


def problems(root: Path, pending: frozenset[str]) -> list[str]:
    """Describe every missing or unnamed diagram; empty when all hold."""
    arch = root / "docs/arch"
    found = []
    drawn = set()
    for source in diagrams(arch):
        for suffix in (".svg", ".facts.json"):
            if not source.with_suffix(suffix).exists():
                found.append(f"{source.relative_to(root)} has no {source.stem}{suffix}")
        facts = source.with_suffix(".facts.json")
        if facts.exists():
            crate = json.loads(facts.read_text(encoding="utf-8")).get("crate")
            if not isinstance(crate, str):
                found.append(f"{facts.relative_to(root)} names no crate")
            else:
                drawn.add(crate)
    crates = model_crates(root)
    found += [
        f"{name} names no model crate under crates/models"
        for name in sorted(drawn - crates)
    ]
    found += [
        f"PENDING lists {name}, which is not a model crate"
        for name in sorted(pending - crates)
    ]
    found += [
        f"PENDING lists {name}, which has a diagram: remove it"
        for name in sorted(pending & drawn)
    ]
    found += [
        f"crates/models/{name} has no diagram under docs/arch"
        for name in sorted(crates - drawn - pending)
    ]
    return found


def typst_binary() -> tuple[str | None, str]:
    """The typst on PATH if it is the pinned version, and what was found."""
    typst = shutil.which("typst")
    if typst is None:
        return None, "no typst on PATH"
    version = subprocess.run(
        [typst, "--version"],
        capture_output=True,
        text=True,
        check=True,
        timeout=RENDER_TIMEOUT_SECONDS,
    ).stdout.strip()
    if version.split()[:2] != ["typst", TYPST_VERSION]:
        return None, version
    return typst, version


def render(typst: str, source: Path, output: Path) -> None:
    """Render one diagram; bundled fonts only, so every machine agrees."""
    subprocess.run(
        [
            typst,
            "compile",
            "--root",
            str(source.parent),
            "--ignore-system-fonts",
            str(source),
            str(output),
        ],
        check=True,
        timeout=RENDER_TIMEOUT_SECONDS,
    )


def stale(root: Path, typst: str) -> list[str]:
    """Diagrams whose committed SVG differs from a fresh rendering."""
    found = []
    with tempfile.TemporaryDirectory() as scratch:
        for source in diagrams(root / "docs/arch"):
            fresh = Path(scratch) / f"{source.stem}.svg"
            render(typst, source, fresh)
            committed = source.with_suffix(".svg")
            if not committed.exists() or committed.read_bytes() != fresh.read_bytes():
                found.append(f"{committed.relative_to(root)} is stale")
    return found


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--write", action="store_true", help="re-render every SVG in place"
    )
    parser.add_argument(
        "--require-typst",
        action="store_true",
        help=f"fail instead of skipping when typst {TYPST_VERSION} is absent",
    )
    args = parser.parse_args()

    found = []
    typst, version = typst_binary()
    if typst is None:
        notice = f"needs typst {TYPST_VERSION}, found: {version}"
        if args.write or args.require_typst:
            found.append(notice)
        else:
            print(f"arch diagrams: SVG comparison skipped; {notice}", file=sys.stderr)
    elif args.write:
        for source in diagrams(ROOT / "docs/arch"):
            render(typst, source, source.with_suffix(".svg"))
    else:
        found += [
            f"{problem}; re-render with scripts/check_arch_diagrams.py --write"
            for problem in stale(ROOT, typst)
        ]
    # After --write, so a freshly rendered SVG counts as present.
    found = problems(ROOT, PENDING) + found
    for problem in found:
        print(f"arch diagrams: {problem}", file=sys.stderr)
    if found:
        return 1
    print(f"arch diagrams hold: {len(PENDING)} model crates pending")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
