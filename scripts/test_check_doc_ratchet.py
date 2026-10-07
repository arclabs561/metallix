"""Tests for the documentation-lint ratchet, on the live tree and temporary workspaces."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import check_doc_ratchet as ratchet

DOCUMENTED = (
    "//! A crate.\n\n#![deny(missing_docs)]\n#![warn(clippy::missing_errors_doc)]\n"
)
UNDOCUMENTED = "//! A crate.\n\npub fn f() {}\n"


def write_workspace(
    root: Path, crates: dict[str, str], *, allow_errors_doc: bool = True
) -> None:
    members = ", ".join(f'"{name}"' for name in crates)
    lints = '\n[workspace.lints.clippy]\nmissing_errors_doc = "allow"\n'
    (root / "Cargo.toml").write_text(
        f"[workspace]\nmembers = [{members}]\n" + (lints if allow_errors_doc else ""),
        encoding="utf-8",
    )
    for name, source in crates.items():
        (root / name / "src").mkdir(parents=True)
        (root / name / "src/lib.rs").write_text(source, encoding="utf-8")


class DocRatchetTests(unittest.TestCase):
    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)

    def test_live_workspace_holds(self) -> None:
        self.assertEqual(ratchet.problems(ratchet.ROOT, ratchet.PENDING), [])

    def test_undocumented_crate_outside_pending_fails(self) -> None:
        write_workspace(self.root, {"a": DOCUMENTED, "b": UNDOCUMENTED})
        found = ratchet.problems(self.root, frozenset())
        self.assertEqual(len(found), 1)
        self.assertIn("b/src/lib.rs lacks #![deny(missing_docs)]", found[0])

    def test_documented_crate_still_pending_fails(self) -> None:
        write_workspace(self.root, {"a": DOCUMENTED, "b": UNDOCUMENTED})
        found = ratchet.problems(self.root, frozenset({"a", "b"}))
        self.assertEqual(len(found), 1)
        self.assertIn("remove a from PENDING", found[0])

    def test_pending_must_name_a_member(self) -> None:
        write_workspace(self.root, {"a": DOCUMENTED})
        found = ratchet.problems(self.root, frozenset({"gone"}))
        self.assertEqual(found, ["PENDING lists gone, which is not a workspace member"])

    def test_errors_doc_attribute_required_only_while_workspace_allows_it(self) -> None:
        source = "#![deny(missing_docs)]\n"
        write_workspace(self.root, {"a": source})
        found = ratchet.problems(self.root, frozenset())
        self.assertEqual(
            found, ["a/src/lib.rs lacks #![warn(clippy::missing_errors_doc)]"]
        )

        other = self.root / "without-allow"
        other.mkdir()
        write_workspace(other, {"a": source}, allow_errors_doc=False)
        self.assertEqual(ratchet.problems(other, frozenset()), [])

    def test_attribute_lists_count_and_comments_do_not(self) -> None:
        listed = (
            "#![deny(missing_docs, rustdoc::broken_intra_doc_links)]\n"
            "#![deny(clippy::missing_errors_doc)]\n"
        )
        commented = "// #![deny(missing_docs)]\n#![warn(clippy::missing_errors_doc)]\n"
        write_workspace(self.root, {"a": listed, "b": commented})
        found = ratchet.problems(self.root, frozenset())
        self.assertEqual(found, ["b/src/lib.rs lacks #![deny(missing_docs)]"])

    def test_warn_missing_docs_is_not_enough(self) -> None:
        write_workspace(
            self.root,
            {"a": "#![warn(missing_docs)]\n#![warn(clippy::missing_errors_doc)]\n"},
        )
        found = ratchet.problems(self.root, frozenset())
        self.assertEqual(found, ["a/src/lib.rs lacks #![deny(missing_docs)]"])


if __name__ == "__main__":
    unittest.main()
