"""Check that every workspace crate keeps the documentation lints it has earned.

A crate is either listed in `PENDING`, meaning its public items are not yet
documented to the standard in docs/code-documentation.md, or its library root
carries `#![deny(missing_docs)]` and, while the workspace allows
`clippy::missing_errors_doc`, `#![warn(clippy::missing_errors_doc)]`. The
Clippy and rustdoc steps of the main gate then enforce those lints; this check
only keeps the attributes from being dropped or never added.

`PENDING` only shrinks. A listed crate that already carries the attributes
fails here, so the list stays accurate, and a crate added to the workspace
must start documented rather than join the list.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import tomllib

ROOT = Path(__file__).resolve().parent.parent

# Workspace members not yet retrofitted. Remove a crate in the same change
# that adds its attributes; never add one.
PENDING = frozenset(
    {
        "crates/chat-format",
        "crates/models/deepseek",
        "crates/models/julia",
        "crates/models/qwen",
        "crates/server",
    }
)

# Inner attributes at the start of a line, so commented-out ones do not count.
INNER_ATTRIBUTE = re.compile(
    r"^#!\[(allow|warn|deny|forbid)\(([^)]*)\)\]", re.MULTILINE
)


def lint_levels(source: str) -> dict[str, str]:
    """Map each lint named in a crate root's inner attributes to its level."""
    levels = {}
    for match in INNER_ATTRIBUTE.finditer(source):
        for lint in match.group(2).split(","):
            levels[lint.strip()] = match.group(1)
    return levels


def workspace(root: Path) -> tuple[list[str], bool]:
    """Return the workspace members and whether it allows missing_errors_doc."""
    manifest = tomllib.loads((root / "Cargo.toml").read_text(encoding="utf-8"))
    members = manifest["workspace"]["members"]
    clippy = manifest["workspace"].get("lints", {}).get("clippy", {})
    level = clippy.get("missing_errors_doc")
    if isinstance(level, dict):
        level = level.get("level")
    return members, level == "allow"


def problems(root: Path, pending: frozenset[str]) -> list[str]:
    """Describe every way the workspace breaks the ratchet; empty when it holds."""
    members, errors_doc_allowed = workspace(root)
    found = [
        f"PENDING lists {name}, which is not a workspace member"
        for name in sorted(pending - set(members))
    ]
    for member in members:
        crate_root = root / member / "src/lib.rs"
        if not crate_root.exists():
            crate_root = root / member / "src/main.rs"
        levels = lint_levels(crate_root.read_text(encoding="utf-8"))
        missing = []
        if levels.get("missing_docs") not in {"deny", "forbid"}:
            missing.append("#![deny(missing_docs)]")
        if errors_doc_allowed and levels.get("clippy::missing_errors_doc") not in {
            "warn",
            "deny",
            "forbid",
        }:
            missing.append("#![warn(clippy::missing_errors_doc)]")
        relative = crate_root.relative_to(root)
        if member in pending and not missing:
            found.append(
                f"{relative} carries its documentation lints: remove {member} from PENDING"
            )
        elif member not in pending and missing:
            found.append(f"{relative} lacks {' and '.join(missing)}")
    return found


def main() -> int:
    found = problems(ROOT, PENDING)
    for problem in found:
        print(f"doc ratchet: {problem}", file=sys.stderr)
    if found:
        return 1
    print(f"doc ratchet holds: {len(PENDING)} crates pending")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
