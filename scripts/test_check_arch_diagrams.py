"""Tests for the architecture-diagram check, on the live tree and temporary trees."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import check_arch_diagrams as check

# A diagram that needs no package, so rendering it works offline.
PLAIN_SOURCE = "#set page(width: auto, height: auto)\nA diagram.\n"


def write_tree(root: Path, crates: list[str], drawn: dict[str, str]) -> None:
    """Model crates under crates/models, and one diagram per `drawn` entry
    (diagram stem to crate name)."""
    for name in crates:
        (root / "crates/models" / name).mkdir(parents=True)
        (root / "crates/models" / name / "Cargo.toml").write_text("", encoding="utf-8")
    arch = root / "docs/arch"
    arch.mkdir(parents=True)
    (arch / "lib.typ").write_text("", encoding="utf-8")
    for stem, crate in drawn.items():
        (arch / f"{stem}.typ").write_text(PLAIN_SOURCE, encoding="utf-8")
        (arch / f"{stem}.svg").write_text("<svg/>", encoding="utf-8")
        (arch / f"{stem}.facts.json").write_text(
            json.dumps({"crate": crate}), encoding="utf-8"
        )


class ArchDiagramTests(unittest.TestCase):
    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)

    def test_live_tree_holds(self) -> None:
        self.assertEqual(check.problems(check.ROOT, check.PENDING), [])

    def test_undrawn_crate_outside_pending_fails(self) -> None:
        write_tree(self.root, ["a", "b"], {"a": "a"})
        self.assertEqual(
            check.problems(self.root, frozenset()),
            ["crates/models/b has no diagram under docs/arch"],
        )

    def test_one_crate_may_have_several_diagrams(self) -> None:
        write_tree(self.root, ["a"], {"a_dense": "a", "a_moe": "a"})
        self.assertEqual(check.problems(self.root, frozenset()), [])

    def test_drawn_crate_still_pending_fails(self) -> None:
        write_tree(self.root, ["a"], {"a": "a"})
        self.assertEqual(
            check.problems(self.root, frozenset({"a"})),
            ["PENDING lists a, which has a diagram: remove it"],
        )

    def test_pending_must_name_a_model_crate(self) -> None:
        write_tree(self.root, ["a"], {"a": "a"})
        self.assertEqual(
            check.problems(self.root, frozenset({"gone"})),
            ["PENDING lists gone, which is not a model crate"],
        )

    def test_diagram_needs_its_svg_and_facts(self) -> None:
        write_tree(self.root, ["a"], {"a": "a"})
        (self.root / "docs/arch/a.svg").unlink()
        (self.root / "docs/arch/a.facts.json").unlink()
        self.assertEqual(
            check.problems(self.root, frozenset()),
            [
                "docs/arch/a.typ has no a.svg",
                "docs/arch/a.typ has no a.facts.json",
                "crates/models/a has no diagram under docs/arch",
            ],
        )

    def test_facts_must_name_an_existing_crate(self) -> None:
        write_tree(self.root, ["a"], {"a": "a", "b": "renamed"})
        self.assertEqual(
            check.problems(self.root, frozenset()),
            ["renamed names no model crate under crates/models"],
        )

    def write_adapter_tests(self, source: str) -> None:
        path = self.root / check.ADAPTER_TESTS
        path.parent.mkdir(parents=True)
        path.write_text(source, encoding="utf-8")

    def test_live_undrawn_list_parses(self) -> None:
        self.assertIsNotNone(check.undrawn(check.ROOT))

    def test_undrawn_entries_parse_with_their_reasons(self) -> None:
        self.write_adapter_tests(
            "const UNDRAWN: [(Adapter, &str); 2] = [\n"
            '    (Adapter::EmbeddingGemma2, "drawn once its parser lands"),\n'
            '    (\n        Adapter::Julia,\n        "a \\"quoted\\" reason",\n    ),\n'
            "];\n"
        )
        self.assertEqual(
            check.undrawn(self.root),
            [
                ("EmbeddingGemma2", "drawn once its parser lands"),
                ("Julia", 'a \\"quoted\\" reason'),
            ],
        )

    def test_empty_undrawn_list_parses_as_empty(self) -> None:
        self.write_adapter_tests("const UNDRAWN: [(Adapter, &str); 0] = [];\n")
        self.assertEqual(check.undrawn(self.root), [])

    def test_missing_undrawn_list_is_reported(self) -> None:
        self.write_adapter_tests("fn diagram() {}\n")
        self.assertIsNone(check.undrawn(self.root))

    @unittest.skipIf(check.typst_binary()[0] is None, "needs the pinned typst")
    def test_edited_svg_is_stale_until_rendered(self) -> None:
        write_tree(self.root, ["a"], {"a": "a"})
        typst = check.typst_binary()[0]
        self.assertEqual(check.stale(self.root, typst), ["docs/arch/a.svg is stale"])
        check.render(
            typst, self.root / "docs/arch/a.typ", self.root / "docs/arch/a.svg"
        )
        self.assertEqual(check.stale(self.root, typst), [])


if __name__ == "__main__":
    unittest.main()
