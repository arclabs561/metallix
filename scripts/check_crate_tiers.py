"""Check that workspace crates depend only on their own tier or a lower one.

Every workspace member lives in one tier directory under crates/, lowest
first:

- core: no MLX; formats, contracts and host-side arithmetic;
- metal: MLX or Metal infrastructure shared by models;
- models: one crate per architecture;
- serve: the server, HTTP and the CLI.

A dependency may point to the same tier or a lower one: core may not depend
on metal, models or serve; metal may not depend on models or serve; models
may not depend on serve. Edges within a tier are allowed (flux2 and
qwen3-asr use qwen; Cargo already rejects cycles). Normal, dev, build and
target-specific dependencies all count, and so does a path dependency
inherited from [workspace.dependencies]. A member outside the tier
directories fails, so a new crate has to choose its tier.
"""

from __future__ import annotations

import sys
from pathlib import Path

import tomllib

ROOT = Path(__file__).resolve().parent.parent
TIERS = ("core", "metal", "models", "serve")
DEPENDENCY_TABLES = ("dependencies", "dev-dependencies", "build-dependencies")


def tier_of(member: str) -> str | None:
    """The tier directory of a member path such as crates/core/engine."""
    parts = Path(member).parts
    if len(parts) == 3 and parts[0] == "crates" and parts[1] in TIERS:
        return parts[1]
    return None


def path_dependencies(manifest: dict, workspace_paths: dict[str, str]):
    """Yield (table, name, path, inherited) for every path dependency.

    An inherited path comes from [workspace.dependencies] and is relative to
    the workspace root rather than the member.
    """
    tables = [(name, manifest.get(name, {})) for name in DEPENDENCY_TABLES]
    for target, sections in manifest.get("target", {}).items():
        tables += [
            (f"target.{target}.{name}", sections.get(name, {}))
            for name in DEPENDENCY_TABLES
        ]
    for table, dependencies in tables:
        for name, spec in dependencies.items():
            if not isinstance(spec, dict):
                continue
            if "path" in spec:
                yield table, name, spec["path"], False
            elif spec.get("workspace") and name in workspace_paths:
                yield table, name, workspace_paths[name], True


def problems(root: Path) -> list[str]:
    """Describe every tier violation in the workspace at `root`."""
    workspace = tomllib.loads((root / "Cargo.toml").read_text(encoding="utf-8"))
    members = workspace["workspace"]["members"]
    workspace_paths = {
        name: spec["path"]
        for name, spec in workspace["workspace"].get("dependencies", {}).items()
        if isinstance(spec, dict) and "path" in spec
    }
    tier_by_dir = {(root / member).resolve(): tier_of(member) for member in members}
    found = [
        f"{member} is not under crates/<tier>/ (tiers: {', '.join(TIERS)})"
        for member in members
        if tier_of(member) is None
    ]
    for member in members:
        tier = tier_of(member)
        if tier is None:
            continue
        manifest = tomllib.loads(
            (root / member / "Cargo.toml").read_text(encoding="utf-8")
        )
        for table, name, path, inherited in path_dependencies(
            manifest, workspace_paths
        ):
            base = root if inherited else root / member
            target_tier = tier_by_dir.get((base / path).resolve())
            if target_tier is not None and TIERS.index(target_tier) > TIERS.index(tier):
                found.append(
                    f"{member} ({tier}) depends on {name} ({target_tier}) in [{table}]: "
                    "a dependency may not point to a higher tier"
                )
    return found


def main() -> int:
    found = problems(ROOT)
    for problem in found:
        print(f"crate tiers: {problem}", file=sys.stderr)
    if found:
        return 1
    print(f"crate tiers hold: {' <- '.join(TIERS)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
