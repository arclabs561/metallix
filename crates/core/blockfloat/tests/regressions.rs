//! Regression tests for BF16 narrowing edge cases.

use blockfloat::{bf16_to_f32, f32_to_bf16_rne};

/// Narrowing keeps a NaN a NaN, including payloads whose set bits all lie in
/// the 16 low mantissa bits that BF16 drops, and payloads whose rounding
/// increment would carry into the exponent.
#[test]
fn nan_stays_nan_through_bf16_rounding() {
    let mut wrong = Vec::new();
    for bits in [0x7f80_0001_u32, 0xff80_0001, 0xffff_ffff, 0x7fff_ffff] {
        let input = f32::from_bits(bits);
        assert!(input.is_nan());
        let out = f32_to_bf16_rne(input);
        if !bf16_to_f32(out).is_nan() {
            wrong.push(format!("{bits:#010x} -> {out:#06x} ({})", bf16_to_f32(out)));
        }
    }
    assert!(wrong.is_empty(), "NaN narrowed to non-NaN: {wrong:?}");
}
