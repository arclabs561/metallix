"""Tests for the crate tier check, on the live tree and temporary workspaces."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import check_crate_tiers as tiers


def write_workspace(
    root: Path,
    crates: dict[str, str],
    *,
    workspace_dependencies: str = "",
) -> None:
    """A workspace whose members are `crates`' keys, each with the given
    dependency lines appended to its manifest."""
    members = ", ".join(f'"{member}"' for member in crates)
    (root / "Cargo.toml").write_text(
        f"[workspace]\nmembers = [{members}]\n{workspace_dependencies}",
        encoding="utf-8",
    )
    for member, dependencies in crates.items():
        (root / member).mkdir(parents=True)
        name = Path(member).name
        (root / member / "Cargo.toml").write_text(
            f'[package]\nname = "{name}"\n\n{dependencies}', encoding="utf-8"
        )


class CrateTierTests(unittest.TestCase):
    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)

    def test_live_workspace_holds(self) -> None:
        self.assertEqual(tiers.problems(tiers.ROOT), [])

    def test_downward_and_same_tier_edges_pass(self) -> None:
        write_workspace(
            self.root,
            {
                "crates/core/a": "",
                "crates/core/b": '[dependencies]\na = { path = "../a" }\n',
                "crates/metal/k": '[dependencies]\na = { path = "../../core/a" }\n',
                "crates/models/q": '[dependencies]\nk = { path = "../../metal/k" }\n',
                "crates/models/f": '[dependencies]\nq = { path = "../q" }\n',
                "crates/app/s": '[dependencies]\nf = { path = "../../models/f" }\n',
            },
        )
        self.assertEqual(tiers.problems(self.root), [])

    def test_planted_core_to_metal_edge_fails(self) -> None:
        write_workspace(
            self.root,
            {
                "crates/core/a": '[dependencies]\nk = { path = "../../metal/k" }\n',
                "crates/metal/k": "",
            },
        )
        self.assertEqual(
            tiers.problems(self.root),
            [
                (
                    "crates/core/a (core) depends on k (metal) in [dependencies]: "
                    "a dependency may not point to a higher tier"
                )
            ],
        )

    def test_dev_target_and_inherited_upward_edges_fail(self) -> None:
        write_workspace(
            self.root,
            {
                "crates/metal/k": '[dev-dependencies]\nq = { path = "../../models/q" }\n',
                "crates/models/q": (
                    "[target.'cfg(target_os = \"macos\")'.dependencies]\n"
                    "s = { workspace = true }\n"
                ),
                "crates/app/s": "",
            },
            workspace_dependencies='[workspace.dependencies]\ns = { path = "crates/app/s" }\n',
        )
        found = tiers.problems(self.root)
        self.assertEqual(len(found), 2, found)
        self.assertIn(
            "crates/metal/k (metal) depends on q (models) in [dev-dependencies]",
            found[0],
        )
        self.assertIn("crates/models/q (models) depends on s (app)", found[1])

    def test_member_outside_a_tier_fails(self) -> None:
        write_workspace(self.root, {"crates/engine": "", "crates/models/q": ""})
        self.assertEqual(
            tiers.problems(self.root),
            [
                "crates/engine is not under crates/<tier>/ (tiers: core, metal, models, app)"
            ],
        )


if __name__ == "__main__":
    unittest.main()
