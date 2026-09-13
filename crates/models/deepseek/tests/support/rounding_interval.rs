//! Test-only exact BF16 round-to-nearest-even cell checks.
//!
//! The interval is closed in [`f64::total_cmp`] order, which deliberately
//! distinguishes `-0.0` from `+0.0`. This lets a test ask whether a finite
//! higher-precision result could round to a particular *signed* BF16 zero.
//! Non-finite targets and bounds are rejected rather than given an implicit
//! overflow or NaN policy.

use std::cmp::Ordering;

use thiserror::Error;

/// An invalid request to test a finite BF16 rounding cell.
#[derive(Clone, Copy, Debug, Error, PartialEq)]
pub enum RoundingIntervalError {
    /// The BF16 storage bits encode infinity or NaN rather than a finite value.
    #[error("BF16 target {bits:#06x} is non-finite")]
    NonFiniteTarget {
        /// BF16 storage bits supplied by the caller.
        bits: u16,
    },
    /// A closed interval endpoint is NaN or infinity.
    #[error("rounding interval {bound} endpoint is non-finite")]
    NonFiniteBound {
        /// Endpoint role.
        bound: &'static str,
    },
    /// The endpoints are reversed in total floating-point order.
    #[error("rounding interval lower endpoint is above its upper endpoint")]
    ReversedBounds,
    /// A finite endpoint rounds to infinity instead of a finite BF16 value.
    #[error("rounding interval {bound} endpoint permits BF16 infinity overflow")]
    Overflow {
        /// Endpoint whose RNE result is infinite.
        bound: &'static str,
    },
}

/// Returns whether a closed finite `f64` interval intersects one BF16 RNE cell.
///
/// The target is supplied as BF16 storage bits. Midpoints belong to the even
/// adjacent BF16 value, including around subnormals. The cells for `+0` and
/// `-0` are distinct: the former contains `+0` and positive tiny values, while
/// the latter contains `-0` and negative tiny values. For largest finite
/// magnitudes, the outer bound is the finite midpoint at which RNE overflows
/// to infinity, so that bound is excluded.
///
/// # Errors
///
/// Returns [`RoundingIntervalError`] for NaN/infinite targets or bounds, or
/// reversed endpoints under [`f64::total_cmp`].
pub fn bf16_rounding_cell_intersects(
    bits: u16,
    lower: f64,
    upper: f64,
) -> Result<bool, RoundingIntervalError> {
    validate_interval(bits, lower, upper)?;
    let cell = rounding_cell(bits);
    Ok(intersects_closed_interval(lower, upper, cell))
}

/// Returns the finite BF16 RNE enclosure of a closed finite `f64` interval.
///
/// The returned values are the smallest and largest finite BF16 values
/// attainable by direct round-to-nearest-even of a value in `[lower, upper]`.
/// Endpoint selection operates in `f64` against exact BF16 midpoints; it does
/// not first narrow to FP32, so values immediately either side of a midpoint do
/// not double-round. Signed zero follows the total-order interval contract of
/// [`bf16_rounding_cell_intersects`].
///
/// # Errors
///
/// Returns [`RoundingIntervalError`] for non-finite or reversed bounds, or if
/// either endpoint maps to BF16 infinity. Monotonic RNE means an endpoint
/// overflow is exactly when the interval permits an infinite BF16 result.
pub fn bf16_rounded_enclosure(lower: f64, upper: f64) -> Result<[f64; 2], RoundingIntervalError> {
    validate_bounds(lower, upper)?;
    let minimum = round_f64_to_finite_bf16(lower)
        .ok_or(RoundingIntervalError::Overflow { bound: "lower" })?;
    let maximum = round_f64_to_finite_bf16(upper)
        .ok_or(RoundingIntervalError::Overflow { bound: "upper" })?;
    Ok([bf16_value(minimum), bf16_value(maximum)])
}

#[derive(Clone, Copy)]
struct Cell {
    lower: f64,
    lower_inclusive: bool,
    upper: f64,
    upper_inclusive: bool,
}

fn validate_interval(bits: u16, lower: f64, upper: f64) -> Result<(), RoundingIntervalError> {
    if !bf16_value(bits).is_finite() {
        return Err(RoundingIntervalError::NonFiniteTarget { bits });
    }
    validate_bounds(lower, upper)
}

fn validate_bounds(lower: f64, upper: f64) -> Result<(), RoundingIntervalError> {
    if !lower.is_finite() {
        return Err(RoundingIntervalError::NonFiniteBound { bound: "lower" });
    }
    if !upper.is_finite() {
        return Err(RoundingIntervalError::NonFiniteBound { bound: "upper" });
    }
    if lower.total_cmp(&upper).is_gt() {
        return Err(RoundingIntervalError::ReversedBounds);
    }
    Ok(())
}

fn round_f64_to_finite_bf16(value: f64) -> Option<u16> {
    let minimum = bf16_value(0xff7f);
    let maximum = bf16_value(0x7f7f);
    if value.total_cmp(&minimum).is_lt() {
        return point_in_cell(value, rounding_cell(0xff7f)).then_some(0xff7f);
    }
    if value.total_cmp(&maximum).is_gt() {
        return point_in_cell(value, rounding_cell(0x7f7f)).then_some(0x7f7f);
    }

    let floor = greatest_finite_code_at_most(value);
    let floor_value = bf16_value(floor);
    if floor_value.total_cmp(&value).is_eq() {
        return Some(floor);
    }
    let ceiling = ordered_code_to_bits(bits_to_ordered_code(floor) + 1);
    let midpoint = midpoint(floor_value, bf16_value(ceiling));
    match value.total_cmp(&midpoint) {
        Ordering::Less => Some(floor),
        Ordering::Greater => Some(ceiling),
        Ordering::Equal => Some(if floor & 1 == 0 { floor } else { ceiling }),
    }
}

fn greatest_finite_code_at_most(value: f64) -> u16 {
    let mut low = bits_to_ordered_code(0xff7f);
    let mut high = bits_to_ordered_code(0x7f7f);
    while low < high {
        let midpoint = low + (high - low).div_ceil(2);
        let candidate = ordered_code_to_bits(midpoint);
        if bf16_value(candidate).total_cmp(&value).is_le() {
            low = midpoint;
        } else {
            high = midpoint - 1;
        }
    }
    ordered_code_to_bits(low)
}

fn bits_to_ordered_code(bits: u16) -> u16 {
    if bits & 0x8000 == 0 {
        bits | 0x8000
    } else {
        !bits
    }
}

fn ordered_code_to_bits(code: u16) -> u16 {
    if code & 0x8000 == 0 {
        !code
    } else {
        code & 0x7fff
    }
}

fn rounding_cell(bits: u16) -> Cell {
    match bits {
        0x0000 => Cell {
            lower: 0.0,
            lower_inclusive: true,
            upper: midpoint(0.0, bf16_value(0x0001)),
            upper_inclusive: true,
        },
        0x8000 => Cell {
            lower: midpoint(bf16_value(0x8001), -0.0),
            lower_inclusive: true,
            upper: -0.0,
            upper_inclusive: true,
        },
        0x7f7f => {
            let value = bf16_value(bits);
            let previous = bf16_value(bits - 1);
            Cell {
                lower: midpoint(previous, value),
                lower_inclusive: false,
                upper: value + (value - previous) / 2.0,
                upper_inclusive: false,
            }
        }
        0xff7f => {
            let value = bf16_value(bits);
            let next = bf16_value(bits - 1);
            Cell {
                lower: value - (next - value) / 2.0,
                lower_inclusive: false,
                upper: midpoint(value, next),
                upper_inclusive: false,
            }
        }
        _ if bits & 0x8000 == 0 => {
            let value = bf16_value(bits);
            let previous = bf16_value(bits - 1);
            let next = bf16_value(bits + 1);
            let inclusive = bits & 1 == 0;
            Cell {
                lower: midpoint(previous, value),
                lower_inclusive: inclusive,
                upper: midpoint(value, next),
                upper_inclusive: inclusive,
            }
        }
        _ => {
            let value = bf16_value(bits);
            let previous = bf16_value(bits + 1);
            let next = bf16_value(bits - 1);
            let inclusive = bits & 1 == 0;
            Cell {
                lower: midpoint(previous, value),
                lower_inclusive: inclusive,
                upper: midpoint(value, next),
                upper_inclusive: inclusive,
            }
        }
    }
}

fn intersects_closed_interval(lower: f64, upper: f64, cell: Cell) -> bool {
    match upper.total_cmp(&cell.lower) {
        Ordering::Less => return false,
        Ordering::Equal if !cell.lower_inclusive => return false,
        Ordering::Equal | Ordering::Greater => {}
    }
    match lower.total_cmp(&cell.upper) {
        Ordering::Greater => false,
        Ordering::Equal if !cell.upper_inclusive => false,
        Ordering::Equal | Ordering::Less => true,
    }
}

fn point_in_cell(value: f64, cell: Cell) -> bool {
    intersects_closed_interval(value, value, cell)
}

fn bf16_value(bits: u16) -> f64 {
    f64::from(f32::from_bits(u32::from(bits) << 16))
}

fn midpoint(left: f64, right: f64) -> f64 {
    left + (right - left) / 2.0
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use super::*;

    #[test]
    fn adjacent_halfway_ties_choose_even_targets() {
        let even = 0x3f80;
        let odd = 0x3f81;
        let even_value = bf16_value(even);
        let odd_value = bf16_value(odd);
        let lower_even_midpoint = midpoint(bf16_value(even - 1), even_value);
        let upper_even_midpoint = midpoint(even_value, odd_value);
        let upper_odd_midpoint = midpoint(odd_value, bf16_value(odd + 1));

        assert!(
            bf16_rounding_cell_intersects(even, lower_even_midpoint, lower_even_midpoint)
                .expect("finite midpoint")
        );
        assert!(
            bf16_rounding_cell_intersects(even, upper_even_midpoint, upper_even_midpoint)
                .expect("finite midpoint")
        );
        assert!(
            !bf16_rounding_cell_intersects(odd, upper_even_midpoint, upper_even_midpoint)
                .expect("finite midpoint")
        );
        assert!(
            !bf16_rounding_cell_intersects(odd, upper_odd_midpoint, upper_odd_midpoint)
                .expect("finite midpoint")
        );
    }

    #[test]
    fn negative_halfway_ties_choose_even_targets() {
        let even = 0xbf80;
        let odd = 0xbf81;
        let even_value = bf16_value(even);
        let odd_value = bf16_value(odd);
        let midpoint_to_odd = midpoint(odd_value, even_value);
        let midpoint_above_even = midpoint(even_value, bf16_value(even - 1));

        assert!(
            bf16_rounding_cell_intersects(even, midpoint_to_odd, midpoint_to_odd)
                .expect("finite midpoint")
        );
        assert!(
            bf16_rounding_cell_intersects(even, midpoint_above_even, midpoint_above_even)
                .expect("finite midpoint")
        );
        assert!(
            !bf16_rounding_cell_intersects(odd, midpoint_to_odd, midpoint_to_odd)
                .expect("finite midpoint")
        );
    }

    #[test]
    fn signed_zero_and_subnormal_boundaries_are_distinct() {
        assert!(bf16_rounding_cell_intersects(0x0000, 0.0, 0.0).expect("positive zero"));
        assert!(!bf16_rounding_cell_intersects(0x0000, -0.0, -0.0).expect("negative zero"));
        assert!(bf16_rounding_cell_intersects(0x8000, -0.0, -0.0).expect("negative zero"));
        assert!(!bf16_rounding_cell_intersects(0x8000, 0.0, 0.0).expect("positive zero"));
        assert!(bf16_rounding_cell_intersects(0x0000, -0.0, 0.0).expect("both zeros"));
        assert!(bf16_rounding_cell_intersects(0x8000, -0.0, 0.0).expect("both zeros"));

        let positive_tie = midpoint(0.0, bf16_value(0x0001));
        assert!(
            bf16_rounding_cell_intersects(0x0000, positive_tie, positive_tie)
                .expect("zero has even storage")
        );
        assert!(
            !bf16_rounding_cell_intersects(0x0001, positive_tie, positive_tie)
                .expect("minimum subnormal is odd")
        );
    }

    #[test]
    fn finite_extreme_excludes_the_overflow_midpoint() {
        let max = bf16_value(0x7f7f);
        let previous = bf16_value(0x7f7e);
        let overflow_midpoint = max + (max - previous) / 2.0;
        let inside = f64::from_bits(overflow_midpoint.to_bits() - 1);
        let outside = f64::from_bits(overflow_midpoint.to_bits() + 1);

        assert!(bf16_rounding_cell_intersects(0x7f7f, inside, inside).expect("finite interior"));
        assert!(
            !bf16_rounding_cell_intersects(0x7f7f, overflow_midpoint, overflow_midpoint)
                .expect("finite overflow midpoint")
        );
        assert!(
            !bf16_rounding_cell_intersects(0x7f7f, outside, outside).expect("outside finite cell")
        );
    }

    #[test]
    fn enclosure_uses_exact_midpoint_ties_without_double_rounding() {
        let even = 0x3f80;
        let odd = 0x3f81;
        let midpoint_to_odd = midpoint(bf16_value(even), bf16_value(odd));
        let just_below = f64::from_bits(midpoint_to_odd.to_bits() - 1);
        let just_above = f64::from_bits(midpoint_to_odd.to_bits() + 1);

        assert_eq!(
            enclosure_bits(midpoint_to_odd, midpoint_to_odd).expect("even midpoint"),
            [bf16_value(even).to_bits(), bf16_value(even).to_bits()]
        );
        assert_eq!(
            enclosure_bits(just_below, just_below).expect("just below midpoint"),
            [bf16_value(even).to_bits(), bf16_value(even).to_bits()]
        );
        assert_eq!(
            enclosure_bits(just_above, just_above).expect("just above midpoint"),
            [bf16_value(odd).to_bits(), bf16_value(odd).to_bits()]
        );

        let odd = 0x3f81;
        let even = 0x3f82;
        let midpoint_to_even = midpoint(bf16_value(odd), bf16_value(even));
        let just_below = f64::from_bits(midpoint_to_even.to_bits() - 1);
        assert_eq!(
            enclosure_bits(midpoint_to_even, midpoint_to_even).expect("even midpoint"),
            [bf16_value(even).to_bits(), bf16_value(even).to_bits()]
        );
        assert_eq!(
            enclosure_bits(just_below, just_below).expect("just below midpoint"),
            [bf16_value(odd).to_bits(), bf16_value(odd).to_bits()]
        );
    }

    #[test]
    fn enclosure_preserves_signed_zero_and_rejects_infinite_rounding() {
        let positive = bf16_rounded_enclosure(0.0, 0.0).expect("positive zero");
        assert_eq!(positive[0].to_bits(), 0.0_f64.to_bits());
        assert_eq!(positive[1].to_bits(), 0.0_f64.to_bits());
        let negative = bf16_rounded_enclosure(-0.0, -0.0).expect("negative zero");
        assert_eq!(negative[0].to_bits(), (-0.0_f64).to_bits());
        assert_eq!(negative[1].to_bits(), (-0.0_f64).to_bits());
        let both = bf16_rounded_enclosure(-0.0, 0.0).expect("both signed zeros");
        assert_eq!(both[0].to_bits(), (-0.0_f64).to_bits());
        assert_eq!(both[1].to_bits(), 0.0_f64.to_bits());

        let maximum = bf16_value(0x7f7f);
        let previous = bf16_value(0x7f7e);
        let threshold = maximum + (maximum - previous) / 2.0;
        assert!(matches!(
            bf16_rounded_enclosure(maximum, threshold),
            Err(RoundingIntervalError::Overflow { bound: "upper" })
        ));
        assert!(matches!(
            bf16_rounded_enclosure(-threshold, -maximum),
            Err(RoundingIntervalError::Overflow { bound: "lower" })
        ));
    }

    fn enclosure_bits(lower: f64, upper: f64) -> Result<[u64; 2], RoundingIntervalError> {
        let enclosure = bf16_rounded_enclosure(lower, upper)?;
        Ok([enclosure[0].to_bits(), enclosure[1].to_bits()])
    }

    #[test]
    fn rejects_nonfinite_and_reversed_requests() {
        assert!(matches!(
            bf16_rounding_cell_intersects(0x7f80, 0.0, 1.0),
            Err(RoundingIntervalError::NonFiniteTarget { bits: 0x7f80 })
        ));
        assert!(matches!(
            bf16_rounding_cell_intersects(0x3f80, f64::NAN, 1.0),
            Err(RoundingIntervalError::NonFiniteBound { bound: "lower" })
        ));
        assert!(matches!(
            bf16_rounding_cell_intersects(0x3f80, 1.0, -1.0),
            Err(RoundingIntervalError::ReversedBounds)
        ));
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(64))]
        #[test]
        fn intersection_is_monotonic_when_a_finite_interval_expands(
            bits in 0_u16..=0x7f7f,
            center in -100.0_f64..100.0,
            radius in 0.0_f64..10.0,
            expansion in 0.0_f64..10.0,
        ) {
            let inner_lower = center - radius;
            let inner_upper = center + radius;
            let outer_lower = inner_lower - expansion;
            let outer_upper = inner_upper + expansion;
            let inner = bf16_rounding_cell_intersects(bits, inner_lower, inner_upper)?;
            let outer = bf16_rounding_cell_intersects(bits, outer_lower, outer_upper)?;
            prop_assert!(!inner || outer);
        }
    }
}
