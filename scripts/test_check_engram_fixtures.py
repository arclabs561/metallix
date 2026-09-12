"""Mutation tests for the dependency-free Engram fixture checker."""

from __future__ import annotations

import copy
import json
import unittest

import check_engram_fixtures as checker


def fixture(kind: str) -> dict[str, object]:
    path = checker.HASH_FIXTURE if kind == "hash" else checker.GATE_FIXTURE
    return json.loads(path.read_text(encoding="utf-8"))


class EngramFixtureCheckerTests(unittest.TestCase):
    def test_checked_in_fixtures_validate(self) -> None:
        checker.validate_fixture(fixture("hash"), fixture_kind="hash")
        checker.validate_fixture(fixture("gate"), fixture_kind="gate")

    def test_duplicate_tensor_name_is_rejected(self) -> None:
        payload = fixture("hash")
        tensors = payload["tensors"]
        assert isinstance(tensors, list)
        tensors.append(copy.deepcopy(tensors[0]))
        with self.assertRaisesRegex(checker.FixtureError, "duplicate tensor name"):
            checker.validate_fixture(payload, fixture_kind="hash")

    def test_hash_fixture_rejects_encoding_and_non_int64_dtype(self) -> None:
        payload = fixture("hash")
        tensors = payload["tensors"]
        assert isinstance(tensors, list) and isinstance(tensors[0], dict)
        tensors[0]["encoding"] = "i64"
        with self.assertRaisesRegex(checker.FixtureError, "must not declare encoding"):
            checker.validate_fixture(payload, fixture_kind="hash")

        payload = fixture("hash")
        tensors = payload["tensors"]
        assert isinstance(tensors, list) and isinstance(tensors[0], dict)
        tensors[0]["dtype"] = "torch.float32"
        with self.assertRaisesRegex(checker.FixtureError, "requires torch.int64"):
            checker.validate_fixture(payload, fixture_kind="hash")

    def test_gate_fixture_rejects_encoding_range_and_bool_drift(self) -> None:
        payload = fixture("gate")
        tensors = payload["tensors"]
        assert isinstance(tensors, list) and isinstance(tensors[0], dict)
        tensors[0]["encoding"] = "i64"
        with self.assertRaisesRegex(checker.FixtureError, "requires encoding"):
            checker.validate_fixture(payload, fixture_kind="gate")

        payload = fixture("gate")
        tensors = payload["tensors"]
        assert isinstance(tensors, list) and isinstance(tensors[0], dict)
        tensors[0]["values"][0] = 1 << 16
        with self.assertRaisesRegex(checker.FixtureError, "outside its bit range"):
            checker.validate_fixture(payload, fixture_kind="gate")

        payload = fixture("gate")
        tensors = payload["tensors"]
        assert isinstance(tensors, list) and isinstance(tensors[7], dict)
        tensors[7]["values"][0] = 1
        with self.assertRaisesRegex(checker.FixtureError, "JSON booleans"):
            checker.validate_fixture(payload, fixture_kind="gate")

    def test_shape_length_and_digest_corruption_are_rejected(self) -> None:
        payload = fixture("gate")
        tensors = payload["tensors"]
        assert isinstance(tensors, list) and isinstance(tensors[0], dict)
        tensors[0]["shape"] = [0]
        with self.assertRaisesRegex(checker.FixtureError, "positive integers"):
            checker.validate_fixture(payload, fixture_kind="gate")

        payload = fixture("gate")
        tensors = payload["tensors"]
        assert isinstance(tensors, list) and isinstance(tensors[0], dict)
        tensors[0]["values"].pop()
        with self.assertRaisesRegex(
            checker.FixtureError, "length does not match shape"
        ):
            checker.validate_fixture(payload, fixture_kind="gate")

        payload = fixture("gate")
        tensors = payload["tensors"]
        assert isinstance(tensors, list) and isinstance(tensors[0], dict)
        tensors[0]["values"][0] ^= 1
        with self.assertRaisesRegex(checker.FixtureError, "does not match values"):
            checker.validate_fixture(payload, fixture_kind="gate")

    def test_shape_layout_and_schema_version_cannot_drift_without_digest_change(
        self,
    ) -> None:
        payload = fixture("hash")
        tensors = payload["tensors"]
        assert isinstance(tensors, list) and isinstance(tensors[1], dict)
        values = tensors[1]["values"]
        assert isinstance(values, list)
        tensors[1]["values"] = checker.flatten(values, name="primes")
        with self.assertRaisesRegex(checker.FixtureError, "match shape nesting"):
            checker.validate_fixture(payload, fixture_kind="hash")

        payload = fixture("gate")
        tensors = payload["tensors"]
        assert isinstance(tensors, list) and isinstance(tensors[0], dict)
        values = tensors[0]["values"]
        assert isinstance(values, list)
        tensors[0]["values"] = [values]
        with self.assertRaisesRegex(checker.FixtureError, "flat scalar array"):
            checker.validate_fixture(payload, fixture_kind="gate")

        payload = fixture("hash")
        payload["schema_version"] = True
        with self.assertRaisesRegex(checker.FixtureError, "integer schema_version"):
            checker.validate_fixture(payload, fixture_kind="hash")


if __name__ == "__main__":
    unittest.main()
