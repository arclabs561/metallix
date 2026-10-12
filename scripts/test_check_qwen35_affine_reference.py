"""Reference admission tests; these synthetic receipts qualify the validator only."""

import ast
import copy
import unittest
from pathlib import Path
from types import SimpleNamespace

import check_qwen35_affine_reference as check


def receipt(split="heldout"):
    cases = []
    for i, prompt in enumerate(check.declared_prompts(split)):
        baseline = [8.0 - k for k in range(8)]
        arms = []
        for gdn in ("kernel", "ops"):
            for chunk in check.CHUNKS:
                if (gdn, chunk) == ("kernel", None):
                    continue
                d = (len(arms) + 1) / 32
                got = [baseline[0] + d, *baseline[1:]]
                arms.append(
                    {
                        "gdn": gdn,
                        "prefill_chunk": chunk,
                        "duplicate": False,
                        "top8_logits": [got] * 32,
                        "steps": [{"centered": d * 7 / 8, "margin": d, "max_abs": d}]
                        * 32,
                    }
                )
        cases.append(
            {
                "user": prompt,
                "prompt_ids": [i + (100 if split == "heldout" else 0)],
                "steps": [
                    {
                        "top8_ids": list(range(8)),
                        "top8_logits": baseline,
                        "top2_gap": 1.0,
                    }
                ]
                * 32,
                "aa_arms": arms,
                "aa": copy.deepcopy(arms[-1]["steps"]),
            }
        )
    return {
        "mlx": "0.32.3",
        "mlx_lm": "0.32.0",
        "config_sha256": "synthetic",
        "weights_sha256": {},
        "components": {},
        "cases": cases,
        "qualification": {
            "bar": "v2",
            "steps": 32,
            "prompts": split,
            "min_distinct_arms": 8,
            "prefill_chunks": list(check.CHUNKS),
            "abort_gpu_bytes": 1024,
            "memory": {"active_bytes": 1, "peak_bytes": 2, "cache_bytes": 0},
        },
    }


class ValidityTests(unittest.TestCase):
    def test_valid_and_repeat(self):
        dev, heldout = receipt("dev"), receipt()
        self.assertEqual(check.validate(heldout, "heldout"), [23] * 4)
        check.check_references(dev, copy.deepcopy(dev), heldout, copy.deepcopy(heldout))

    def test_duplicate_cannot_inflate_count(self):
        ref = receipt()
        arm = ref["cases"][0]["aa_arms"][1]
        arm["top8_logits"] = ref["cases"][0]["aa_arms"][0]["top8_logits"]
        with self.assertRaisesRegex(ValueError, "duplicate classification"):
            check.validate(ref, "heldout")

    def test_insufficient_distinct_is_invalid(self):
        ref = receipt()
        case = ref["cases"][0]
        for arm in case["aa_arms"][7:]:
            arm.update(
                copy.deepcopy(
                    {k: case["aa_arms"][0][k] for k in ("top8_logits", "steps")}
                )
            )
            arm["duplicate"] = True
        with self.assertRaisesRegex(ValueError, "7 distinct arms"):
            check.validate(ref, "heldout")

    def test_floor_must_be_step_local(self):
        ref = receipt()
        ref["cases"][0]["aa"][0] = {k: 9.0 for k in check.KEYS}
        with self.assertRaisesRegex(ValueError, "per-step floor"):
            check.validate(ref, "heldout")

    def test_nonfinite_and_short_capture_refused(self):
        for mutate in (
            lambda r: r["cases"][0]["steps"].pop(),
            lambda r: r["cases"][0]["steps"][0]["top8_logits"].__setitem__(
                0, float("nan")
            ),
        ):
            ref = receipt()
            mutate(ref)
            with self.assertRaises(ValueError):
                check.validate(ref, "heldout")

    def test_fresh_repeat_requires_same_components(self):
        dev, heldout = receipt("dev"), receipt()
        repeat = copy.deepcopy(heldout)
        repeat["components"]["changed"] = "digest"
        with self.assertRaisesRegex(ValueError, "fresh repeat differs"):
            check.check_references(dev, copy.deepcopy(dev), heldout, repeat)

    def test_memory_cap_checks_peak_and_cached_allocations(self):
        # Extract the actual pure boundary function without importing GPU modules.
        source = Path(__file__).with_name("qwen35-mlx-affine-reference.py").read_text()
        fn = next(
            n
            for n in ast.parse(source).body
            if isinstance(n, ast.FunctionDef) and n.name == "check_gpu_memory"
        )
        for active, peak, cache, fails in (
            (10, 10, 0, False),
            (1, 11, 0, True),
            (6, 6, 5, True),
        ):
            scope = {
                "GPU_CAP_BYTES": 10,
                "mx": SimpleNamespace(
                    get_active_memory=lambda v=active: v,
                    get_peak_memory=lambda v=peak: v,
                    get_cache_memory=lambda v=cache: v,
                ),
            }
            exec(  # noqa: S102 - compile only the checked-in memory guard
                compile(
                    ast.Module(body=[fn], type_ignores=[]), "memory-boundary", "exec"
                ),
                scope,
            )
            if fails:
                with self.assertRaisesRegex(RuntimeError, "cap exceeded"):
                    scope["check_gpu_memory"]()
            else:
                self.assertEqual(scope["check_gpu_memory"]()["active_bytes"], 10)


if __name__ == "__main__":
    unittest.main()
