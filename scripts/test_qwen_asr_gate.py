"""Regression checks for incomplete runs and the declared G3 boundaries."""

import unittest

from qwen_asr_gate import coverage, passes, transcripts


class GateTests(unittest.TestCase):
    def setUp(self):
        self.truth = {str(i): "hello" for i in range(100)}
        self.rows = [
            {"id": key, "text": text, "stopped": True}
            for key, text in self.truth.items()
        ]

    def test_complete_other_split(self):
        self.assertEqual(
            coverage("other", self.truth, self.truth, self.rows), self.truth
        )

    def test_intersection_cannot_qualify_a_stopped_run(self):
        with self.assertRaises(ValueError):
            coverage("other", self.truth, self.truth, self.rows[:-1])
        with self.assertRaises(ValueError):
            coverage("clean", self.truth, self.truth, self.rows)

    def test_duplicate_and_unfinished_rows_fail(self):
        with self.assertRaises(ValueError):
            coverage("other", self.truth, self.truth, self.rows + self.rows[:1])
        self.rows[0]["stopped"] = False
        with self.assertRaises(ValueError):
            coverage("other", self.truth, self.truth, self.rows)

    def test_bars_use_unrounded_values(self):
        self.assertTrue(passes(1.1, 1.0, 0.97))
        self.assertFalse(passes(1.20001, 1.0, 1.0))
        self.assertFalse(passes(1.1, 1.0, 0.96999))
        self.assertTrue(passes(1.4, 1.0, 0.9, resampled=True))
        self.assertFalse(passes(1.50001, 1.0, 1.0, resampled=True))
        self.assertFalse(passes(float("nan"), 1.0, 1.0))

    def test_reference_rows_cannot_hide_duplicates_or_malformed_text(self):
        for rows in [
            self.rows + self.rows[:1],
            [{"id": "x"}],
            [None],
            [{"id": "", "text": "hi"}],
        ]:
            with self.assertRaises(ValueError):
                transcripts(rows)


if __name__ == "__main__":
    unittest.main()
