# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Rust regression fixtures for passive telemetry policies."""

import unittest

from check_observability import violations


class ObservabilityTests(unittest.TestCase):
    def test_implicit_argument_debug_is_rejected(self):
        for attribute in ["tracing::instrument", "instrument", "observe"]:
            with self.subTest(attribute=attribute):
                source = f"use tracing::{{instrument, instrument as observe}}; #[{attribute}] fn f(tensor: Tensor) {{}}"
                self.assertEqual(len(violations(source)), 1)

    def test_nested_or_string_skip_all_does_not_opt_out(self):
        for args in [
            "fields(skip_all)",
            'name="skip_all"',
            "skip(tensor)",
            "fields(skip_all = true)",
        ]:
            with self.subTest(args=args):
                self.assertIn(
                    "top-level skip_all",
                    violations(f"#[tracing::instrument({args})] fn f() {{}}")[0][1],
                )

    def test_qualified_and_aliased_instrument(self):
        source = """use tracing as telemetry;
use telemetry::{instrument as observe};
#[observe(skip_all, fields(n = tensor.len()))] fn a() {}
#[telemetry::instrument(name="a", skip_all)] fn b() {}
"""
        self.assertEqual(violations(source), [])

    def test_readback_and_rng_in_macro_arguments_fail(self):
        for expression in [
            "tensor.eval()",
            "tensor.as_slice::<f32>()",
            "tensor.item::<f32>()",
            "tensor.item_exact::<f32>()",
            "rng.next_u64()",
            "policy.sample(&logits, false)",
            "engine::sample_categorical(&p, rng.next_f64())",
        ]:
            with self.subTest(expression=expression):
                self.assertTrue(violations(f"tracing::info!(value = ?{expression});"))

    def test_imported_macro_aliases_and_crate_aliases(self):
        for source in [
            "use tracing::info as log; log!(v = a.eval());",
            'use tracing::{info_span as timed}; timed!("test", v = a.item());',
            "use tracing as t; t::event!(Level::INFO, v = rng.next_u64());",
            "use tracing::*; warn!(v = a.as_slice::<f32>());",
        ]:
            with self.subTest(source=source):
                self.assertEqual(len(violations(source)), 1)

    def test_attribute_field_expressions_are_checked(self):
        source = "#[tracing::instrument(skip_all, fields(value = ?a.eval()))] fn f() {}"
        self.assertEqual(violations(source), [(1, "telemetry must not call eval")])

    def test_record_field_expressions_are_checked(self):
        self.assertTrue(violations('span.record("value", tensor.item_cast::<f32>());'))
        self.assertEqual(
            violations('span.record("elapsed", clock.elapsed().as_secs_f64());'), []
        )

    def test_comments_and_literals_do_not_invent_calls(self):
        source = r"""// #[tracing::instrument] a.eval()
/* nested /* tracing::info!(x = t.eval()) */ comment */
tracing::info!("a.eval() \\\"", text = r###"tensor.item::<f32>()"###,
    bytes = br#"rng.next_u64()"#, character = '\"', n = items.len());
let text = "#[tracing::instrument]";
"""
        self.assertEqual(violations(source), [])

    def test_allowed_metadata_and_timing(self):
        self.assertEqual(
            violations(
                "tracing::info!(count = items.len(), elapsed = start.elapsed().as_secs_f64(), value = %value);"
            ),
            [],
        )

    def test_real_call_after_raw_string_is_still_rejected(self):
        self.assertEqual(
            violations('tracing::info!(r#"a.eval()"#, value = a.eval());'),
            [(1, "telemetry must not call eval")],
        )

    def test_balanced_nested_fields_and_source_line(self):
        self.assertEqual(
            violations(
                '\ntracing::info_span!("s", value = { let n = [1, 2];\n tensor.eval() });'
            ),
            [(3, "telemetry must not call eval")],
        )

    def test_absolute_paths_and_grouped_self_alias(self):
        for source in [
            "#[::tracing::instrument] fn f(tensor: Tensor) {}",
            "use ::tracing::info; info!(x=tensor.eval());",
            "use tracing::{self as t}; t::info!(x=tensor.eval());",
            "use ::tracing::{self as t, instrument as observe}; #[observe] fn f() {}",
            "use tracing::{self as t}; use t::info as report; report!(x=tensor.eval());",
        ]:
            with self.subTest(source=source):
                self.assertEqual(len(violations(source)), 1)
        self.assertEqual(violations("#[::tracing::instrument(skip_all)] fn f() {}"), [])

    def test_record_turbofish_preserves_argument_check(self):
        self.assertEqual(
            violations('span.record::<f32>("value", tensor.item::<f32>());'),
            [(1, "telemetry must not call item")],
        )
        self.assertTrue(
            violations('span.record::<Option<Vec<f32>>>("value", tensor.eval());')
        )
        self.assertEqual(violations('span.record::<usize>("value", items.len());'), [])

    def test_unrelated_code_is_outside_scope(self):
        self.assertEqual(
            violations(
                "fn f() { tensor.eval(); rng.next_u64(); } other::info!(x = tensor.eval());"
            ),
            [],
        )


if __name__ == "__main__":
    unittest.main()
