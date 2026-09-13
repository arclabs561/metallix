//! Test-only interval envelope for the reduced two-copy Hyper-Connection split.
//!
//! This models a declared FP32 round-to-nearest target: each basic operation
//! has unit roundoff `u = 2^-24`, while direct `exp` has relative error at
//! most `2u`. Sigmoid's conservative composite budget is `gamma(8)`. It is an
//! output-checking model, not a guarantee about Rust,
//! Torch, or any backend's exponential implementation.

use std::fmt;

const U: f64 = 5.960_464_477_539_063e-8;
const MIXES: usize = 8;

/// A closed, outward-rounded finite FP64 interval.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Interval {
    pub(crate) lo: f64,
    pub(crate) hi: f64,
}

/// Interval results for the reduced two-copy coefficient split.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct CoefficientEnvelopes {
    pub(crate) pre: Vec<[f64; 2]>,
    pub(crate) post: Vec<[f64; 2]>,
    /// Row-major `[source_copy, destination_copy]` combination bounds.
    pub(crate) comb: Vec<[f64; 2]>,
}

/// A failed precondition for the test-only target arithmetic model.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum HcCoefficientBoundsError {
    MixLength { actual: usize },
    MixInterval { index: usize },
    Input { field: &'static str, index: usize },
    BaseLength { actual: usize },
    Iterations { iterations: usize },
    AffineDomain { index: usize },
    NonNormal { stage: &'static str, index: usize },
    NonFiniteEnvelope,
}

impl fmt::Display for HcCoefficientBoundsError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MixLength { actual } => write!(formatter, "mix length is {actual}, expected 8"),
            Self::MixInterval { index } => write!(formatter, "invalid raw mix interval {index}"),
            Self::Input { field, index } => write!(formatter, "invalid {field} input {index}"),
            Self::BaseLength { actual } => write!(formatter, "base length is {actual}, expected 8"),
            Self::Iterations { iterations } => {
                write!(formatter, "invalid iteration count {iterations}")
            }
            Self::AffineDomain { index } => {
                write!(formatter, "affine interval {index} exceeds [-1, 1]")
            }
            Self::NonNormal { stage, index } => {
                write!(
                    formatter,
                    "{stage} result {index} is outside normal FP32 domain"
                )
            }
            Self::NonFiniteEnvelope => formatter.write_str("non-finite interval envelope"),
        }
    }
}

impl std::error::Error for HcCoefficientBoundsError {}

/// Bounds the two-copy HC split from eight raw normalized mix intervals.
///
/// `mixes` are `[pre(2), post(2), comb(2, 2)]`; their endpoints are FP64
/// bounds around the scalar projection's raw FP32 values. `scale` and `base`
/// are exact FP32 controls. Affine values must remain within `[-1, 1]`, the
/// deliberately small domain supported by this first diagnostic helper.
///
/// The exponential endpoint uses a degree-24 positive Taylor polynomial. For
/// `x` in `[0, 2]`, its remainder is at most `8 * 2^25 / 25!`; negative
/// arguments use reciprocal monotonicity. Direct softmax exponentials inflate
/// this ideal result by `2u`; sigmoid uses `gamma(8)` for its exp, denominator,
/// and division form. This budget is derived solely from input intervals and
/// operation counts.
pub(crate) fn coefficient_envelopes(
    mixes: &[[f64; 2]],
    scale: &[f32; 3],
    base: &[f32],
    iterations: usize,
    epsilon: f32,
) -> Result<CoefficientEnvelopes, HcCoefficientBoundsError> {
    if mixes.len() != MIXES {
        return Err(HcCoefficientBoundsError::MixLength {
            actual: mixes.len(),
        });
    }
    if base.len() != MIXES {
        return Err(HcCoefficientBoundsError::BaseLength { actual: base.len() });
    }
    if iterations == 0 || iterations > 64 {
        return Err(HcCoefficientBoundsError::Iterations { iterations });
    }
    input(epsilon, "epsilon", 0)?;
    if epsilon <= 0.0 {
        return Err(HcCoefficientBoundsError::Input {
            field: "epsilon",
            index: 0,
        });
    }
    for (index, &value) in scale.iter().enumerate() {
        input(value, "scale", index)?;
    }
    for (index, &value) in base.iter().enumerate() {
        input(value, "base", index)?;
    }
    let mut raw = [Interval { lo: 0.0, hi: 0.0 }; MIXES];
    for (index, &endpoints) in mixes.iter().enumerate() {
        raw[index] = Interval::from_endpoints(endpoints, index)?;
    }

    let pre = [
        pre_value(affine(raw[0], scale[0], base[0], 0)?, epsilon, 0)?,
        pre_value(affine(raw[1], scale[0], base[1], 1)?, epsilon, 1)?,
    ];
    let post = [
        post_value(affine(raw[2], scale[1], base[2], 2)?, 0)?,
        post_value(affine(raw[3], scale[1], base[3], 3)?, 1)?,
    ];
    let logits = [
        affine(raw[4], scale[2], base[4], 4)?,
        affine(raw[5], scale[2], base[5], 5)?,
        affine(raw[6], scale[2], base[6], 6)?,
        affine(raw[7], scale[2], base[7], 7)?,
    ];
    let first_row = softmax_row(logits[0], logits[1], epsilon, 0)?;
    let second_row = softmax_row(logits[2], logits[3], epsilon, 1)?;
    let mut comb = [first_row[0], first_row[1], second_row[0], second_row[1]];
    normalize_columns(&mut comb, epsilon)?;
    for _ in 1..iterations {
        normalize_rows(&mut comb, epsilon)?;
        normalize_columns(&mut comb, epsilon)?;
    }
    Ok(CoefficientEnvelopes {
        pre: pre.map(Interval::endpoints).to_vec(),
        post: post.map(Interval::endpoints).to_vec(),
        comb: comb.map(Interval::endpoints).to_vec(),
    })
}

fn pre_value(
    value: Interval,
    epsilon: f32,
    index: usize,
) -> Result<Interval, HcCoefficientBoundsError> {
    normal_result(
        add(sigmoid(value)?, Interval::point(f64::from(epsilon))?)?,
        "pre",
        index,
    )
    .and_then(|value| relative_positive(value, U))
}

fn post_value(value: Interval, index: usize) -> Result<Interval, HcCoefficientBoundsError> {
    normal_result(scale_positive(sigmoid(value)?, 2.0)?, "post", index)
        .and_then(|value| relative_positive(value, U))
}

fn softmax_row(
    left: Interval,
    right: Interval,
    epsilon: f32,
    row: usize,
) -> Result<[Interval; 2], HcCoefficientBoundsError> {
    let row_max = max_interval(left, right)?;
    let left_exp = exp_target(subtract_target(left, row_max, row * 2)?)?;
    let right_exp = exp_target(subtract_target(right, row_max, row * 2 + 1)?)?;
    Ok([
        add_round(
            ratio(left_exp, right_exp, row * 2)?,
            epsilon,
            "softmax",
            row * 2,
        )?,
        add_round(
            ratio(right_exp, left_exp, row * 2 + 1)?,
            epsilon,
            "softmax",
            row * 2 + 1,
        )?,
    ])
}

fn affine(
    mix: Interval,
    scale: f32,
    base: f32,
    index: usize,
) -> Result<Interval, HcCoefficientBoundsError> {
    let product = multiply(mix, Interval::point(f64::from(scale))?)?;
    let value = add(product, Interval::point(f64::from(base))?)?;
    let error = up(gamma(2)? * up(max_abs(product) + f64::from(base).abs()));
    let value = finite(Interval {
        lo: down(value.lo - error),
        hi: up(value.hi + error),
    })?;
    if value.lo < -1.0 || value.hi > 1.0 {
        return Err(HcCoefficientBoundsError::AffineDomain { index });
    }
    normal_result(value, "affine", index)
}

fn sigmoid(value: Interval) -> Result<Interval, HcCoefficientBoundsError> {
    if value.lo < -1.0 || value.hi > 1.0 {
        return Err(HcCoefficientBoundsError::NonFiniteEnvelope);
    }
    let lo = sigmoid_endpoint(value.lo)?;
    let hi = sigmoid_endpoint(value.hi)?;
    // Eight operations conservatively cover an `exp` target with 2u relative
    // error, denominator add, division, and their composed roundings.
    relative_positive(
        finite(Interval {
            lo: down(lo.lo),
            hi: up(hi.hi),
        })?,
        gamma(8)?,
    )
}

fn sigmoid_endpoint(value: f64) -> Result<Interval, HcCoefficientBoundsError> {
    if value >= 0.0 {
        let exp = exp_nonnegative(value)?;
        finite(Interval {
            lo: down(exp.lo / up(1.0 + exp.hi)),
            hi: up(exp.hi / down(1.0 + exp.lo)),
        })
    } else {
        let exp = exp_nonnegative(-value)?;
        finite(Interval {
            lo: down(1.0 / up(1.0 + exp.hi)),
            hi: up(1.0 / down(1.0 + exp.lo)),
        })
    }
}

fn exp_nonnegative(value: f64) -> Result<Interval, HcCoefficientBoundsError> {
    if !(0.0..=2.0).contains(&value) {
        return Err(HcCoefficientBoundsError::NonFiniteEnvelope);
    }
    let mut lo_term = 1.0;
    let mut hi_term = 1.0;
    let mut lo_sum = 1.0;
    let mut hi_sum = 1.0;
    for degree in 1..=24 {
        let divisor = f64::from(degree);
        lo_term = down(down(lo_term * value) / divisor);
        hi_term = up(up(hi_term * value) / divisor);
        lo_sum = down(lo_sum + lo_term);
        hi_sum = up(hi_sum + hi_term);
    }
    let mut factorial = 1.0;
    for factor in 2..=25 {
        factorial = down(factorial * f64::from(factor));
    }
    finite(Interval {
        lo: down(lo_sum),
        hi: up(hi_sum + up(268_435_456.0 / factorial)),
    })
}

fn exp_target(value: Interval) -> Result<Interval, HcCoefficientBoundsError> {
    if value.lo < -2.0 || value.hi > 2.0 {
        return Err(HcCoefficientBoundsError::NonFiniteEnvelope);
    }
    let lo = exp_endpoint(value.lo)?;
    let hi = exp_endpoint(value.hi)?;
    relative_positive(
        finite(Interval {
            lo: down(lo.lo),
            hi: up(hi.hi),
        })?,
        up(2.0 * U),
    )
}

fn exp_endpoint(value: f64) -> Result<Interval, HcCoefficientBoundsError> {
    if value >= 0.0 {
        exp_nonnegative(value)
    } else {
        let positive = exp_nonnegative(-value)?;
        finite(Interval {
            lo: down(1.0 / positive.hi),
            hi: up(1.0 / positive.lo),
        })
    }
}

fn normalize_columns(
    values: &mut [Interval; 4],
    epsilon: f32,
) -> Result<(), HcCoefficientBoundsError> {
    normalize_pair(values, 0, 2, epsilon)?;
    normalize_pair(values, 1, 3, epsilon)
}

fn normalize_rows(
    values: &mut [Interval; 4],
    epsilon: f32,
) -> Result<(), HcCoefficientBoundsError> {
    normalize_pair(values, 0, 1, epsilon)?;
    normalize_pair(values, 2, 3, epsilon)
}

fn normalize_pair(
    values: &mut [Interval; 4],
    first: usize,
    second: usize,
    epsilon: f32,
) -> Result<(), HcCoefficientBoundsError> {
    let left = normalized(values[first], values[second], epsilon, first)?;
    let right = normalized(values[second], values[first], epsilon, second)?;
    values[first] = left;
    values[second] = right;
    Ok(())
}

fn ratio(
    own: Interval,
    other: Interval,
    index: usize,
) -> Result<Interval, HcCoefficientBoundsError> {
    let lo = down(own.lo / up(own.lo + other.hi));
    let hi = up(own.hi / down(own.hi + other.lo));
    normal_result(finite(Interval { lo, hi })?, "softmax_ratio", index)
        .and_then(|value| relative_positive(value, gamma(3)?))
}

fn normalized(
    own: Interval,
    other: Interval,
    epsilon: f32,
    index: usize,
) -> Result<Interval, HcCoefficientBoundsError> {
    let epsilon = f64::from(epsilon);
    let lo = down(own.lo / up(up(own.lo + other.hi) + epsilon));
    let hi = up(own.hi / down(down(own.hi + other.lo) + epsilon));
    normal_result(finite(Interval { lo, hi })?, "sinkhorn", index)
        .and_then(|value| relative_positive(value, gamma(3)?))
}

fn add_round(
    value: Interval,
    epsilon: f32,
    stage: &'static str,
    index: usize,
) -> Result<Interval, HcCoefficientBoundsError> {
    normal_result(
        add(value, Interval::point(f64::from(epsilon))?)?,
        stage,
        index,
    )
    .and_then(|value| relative_positive(value, U))
}

fn relative_positive(value: Interval, error: f64) -> Result<Interval, HcCoefficientBoundsError> {
    if value.lo <= 0.0 {
        return Err(HcCoefficientBoundsError::NonFiniteEnvelope);
    }
    finite(Interval {
        lo: down(value.lo * down(1.0 - error)),
        hi: up(value.hi * up(1.0 + error)),
    })
}

fn scale_positive(value: Interval, factor: f64) -> Result<Interval, HcCoefficientBoundsError> {
    finite(Interval {
        lo: down(value.lo * factor),
        hi: up(value.hi * factor),
    })
}

fn subtract_target(
    left: Interval,
    right: Interval,
    index: usize,
) -> Result<Interval, HcCoefficientBoundsError> {
    let error = up(U * up(max_abs(left) + max_abs(right)));
    normal_result(
        finite(Interval {
            lo: down(left.lo - right.hi),
            hi: up(left.hi - right.lo),
        })?,
        "softmax_subtract",
        index,
    )
    .and_then(|value| {
        finite(Interval {
            lo: down(value.lo - error),
            hi: up(value.hi + error),
        })
    })
}

fn max_interval(left: Interval, right: Interval) -> Result<Interval, HcCoefficientBoundsError> {
    finite(Interval {
        lo: down(left.lo.max(right.lo)),
        hi: up(left.hi.max(right.hi)),
    })
}

fn add(left: Interval, right: Interval) -> Result<Interval, HcCoefficientBoundsError> {
    finite(Interval {
        lo: down(left.lo + right.lo),
        hi: up(left.hi + right.hi),
    })
}

fn multiply(left: Interval, right: Interval) -> Result<Interval, HcCoefficientBoundsError> {
    let products = [
        left.lo * right.lo,
        left.lo * right.hi,
        left.hi * right.lo,
        left.hi * right.hi,
    ];
    finite(Interval {
        lo: down(products.iter().copied().fold(f64::INFINITY, f64::min)),
        hi: up(products.iter().copied().fold(f64::NEG_INFINITY, f64::max)),
    })
}

fn normal_result(
    value: Interval,
    stage: &'static str,
    index: usize,
) -> Result<Interval, HcCoefficientBoundsError> {
    let crosses_zero = value.lo <= 0.0 && value.hi >= 0.0;
    if max_abs(value) > f64::from(f32::MAX)
        || (!crosses_zero && max_abs(value) < f64::from(f32::MIN_POSITIVE))
    {
        return Err(HcCoefficientBoundsError::NonNormal { stage, index });
    }
    Ok(value)
}

fn input(value: f32, field: &'static str, index: usize) -> Result<(), HcCoefficientBoundsError> {
    if value.is_finite() && (value == 0.0 || value.is_normal()) {
        Ok(())
    } else {
        Err(HcCoefficientBoundsError::Input { field, index })
    }
}

impl Interval {
    fn endpoints(self) -> [f64; 2] {
        [self.lo, self.hi]
    }

    fn from_endpoints(endpoints: [f64; 2], index: usize) -> Result<Self, HcCoefficientBoundsError> {
        finite(Self {
            lo: endpoints[0],
            hi: endpoints[1],
        })
        .map_err(|_| HcCoefficientBoundsError::MixInterval { index })
    }

    fn point(value: f64) -> Result<Self, HcCoefficientBoundsError> {
        finite(Self {
            lo: down(value),
            hi: up(value),
        })
    }
}

fn finite(value: Interval) -> Result<Interval, HcCoefficientBoundsError> {
    if value.lo.is_finite() && value.hi.is_finite() && value.lo <= value.hi {
        Ok(value)
    } else {
        Err(HcCoefficientBoundsError::NonFiniteEnvelope)
    }
}

fn gamma(operations: usize) -> Result<f64, HcCoefficientBoundsError> {
    let operations =
        u32::try_from(operations).map_err(|_| HcCoefficientBoundsError::NonFiniteEnvelope)?;
    let value = up(f64::from(operations) * U);
    if value >= 1.0 {
        Err(HcCoefficientBoundsError::NonFiniteEnvelope)
    } else {
        Ok(up(value / down(1.0 - value)))
    }
}

fn max_abs(value: Interval) -> f64 {
    value.lo.abs().max(value.hi.abs())
}

fn down(value: f64) -> f64 {
    value.next_down()
}
fn up(value: f64) -> f64 {
    value.next_up()
}

#[cfg(test)]
mod tests {
    use super::{HcCoefficientBoundsError, Interval, coefficient_envelopes, softmax_row};

    fn zero_mixes() -> [[f64; 2]; 8] {
        [[0.0, 0.0]; 8]
    }

    #[test]
    fn zero_affines_contain_ideal_half_sigmoid_and_balanced_combination() {
        let bounds =
            coefficient_envelopes(&zero_mixes(), &[0.0; 3], &[0.0; 8], 1, f32::MIN_POSITIVE)
                .expect("bounded zero-affine split");
        assert!(bounds.pre[0][0] <= 0.5 && 0.5 <= bounds.pre[0][1]);
        assert!(bounds.post[0][0] <= 1.0 && 1.0 <= bounds.post[0][1]);
        assert!(bounds.comb[0][0] <= 0.5 && 0.5 <= bounds.comb[0][1]);
        assert!(bounds.comb[3][0] <= 0.5 && 0.5 <= bounds.comb[3][1]);
    }

    #[test]
    fn rejects_malformed_shapes_and_unsupported_affine_domain() {
        assert!(matches!(
            coefficient_envelopes(&zero_mixes(), &[0.0; 3], &[0.0; 7], 1, 1.0),
            Err(HcCoefficientBoundsError::BaseLength { .. })
        ));
        assert!(matches!(
            coefficient_envelopes(&[[2.0, 2.0]; 8], &[1.0; 3], &[0.0; 8], 1, 1.0),
            Err(HcCoefficientBoundsError::AffineDomain { .. })
        ));
        assert!(matches!(
            coefficient_envelopes(&zero_mixes(), &[0.0; 3], &[0.0; 8], 0, 1.0),
            Err(HcCoefficientBoundsError::Iterations { .. })
        ));
    }

    #[test]
    fn direct_rowmax_softmax_encloses_an_asymmetric_row() {
        let row = softmax_row(
            Interval::point(0.5).expect("finite left logit"),
            Interval::point(-0.5).expect("finite right logit"),
            f32::MIN_POSITIVE,
            0,
        )
        .expect("bounded direct softmax");
        let left = 1.0_f32 / (1.0 + (-1.0_f32).exp());
        let right = 1.0_f32 - left;
        assert!(row[0].lo <= f64::from(left) && f64::from(left) <= row[0].hi);
        assert!(row[1].lo <= f64::from(right) && f64::from(right) <= row[1].hi);
    }

    #[test]
    fn wider_raw_mix_intervals_widen_the_monotone_pre_envelope() {
        let point = coefficient_envelopes(&[[0.0, 0.0]; 8], &[1.0; 3], &[0.0; 8], 1, 1.0e-4)
            .expect("point envelope");
        let wide = coefficient_envelopes(&[[-0.25, 0.25]; 8], &[1.0; 3], &[0.0; 8], 1, 1.0e-4)
            .expect("wide envelope");
        assert!(wide.pre[0][0] <= point.pre[0][0]);
        assert!(wide.pre[0][1] >= point.pre[0][1]);
    }
}
