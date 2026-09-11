//! CPU and Metal qualification of the V4.1 rotary-tail layout.
//!
//! This mirrors `apply_rotary_emb` from the pinned upstream implementation
//! ([pinned source](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/dba1be0a40aa45a94ad051997016db3960a90277/inference/model.py#L392)).
//! The qualified representation is a contiguous FP32 tail with explicit
//! batch, sequence, head, and adjacent-complex-pair dimensions. It is not a
//! general tensor implementation, `RoPE`-frequency generator, Metal kernel, or
//! full V4.1 attention claim.

use std::num::NonZeroUsize;

#[cfg(feature = "metal")]
use mlx_rs::{Array, StreamOrDevice, ops::indexing::TryIndexOp};
use thiserror::Error;

/// One finite complex rotary frequency in real/imaginary order.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RotaryFrequency {
    real: f32,
    imaginary: f32,
}

impl RotaryFrequency {
    /// Creates one finite rotary frequency.
    ///
    /// # Errors
    ///
    /// Returns [`RotaryError::NonFiniteFrequency`] when either component is
    /// not finite. Unit magnitude is deliberately not required here: the
    /// pinned operation consumes an already prepared complex frequency tensor.
    pub fn new(real: f32, imaginary: f32) -> Result<Self, RotaryError> {
        if !real.is_finite() || !imaginary.is_finite() {
            return Err(RotaryError::NonFiniteFrequency);
        }
        Ok(Self { real, imaginary })
    }

    /// Returns the real component of this frequency.
    #[must_use]
    pub const fn real(self) -> f32 {
        self.real
    }

    /// Returns the imaginary component of this frequency.
    #[must_use]
    pub const fn imaginary(self) -> f32 {
        self.imaginary
    }
}

/// Validated parameters for V4.1's RoPE-frequency generator.
///
/// This represents the numerical arguments to the pinned
/// `precompute_freqs_cis` helper. It deliberately does not model the source
/// function's process-local cache; callers ask for only the position range
/// they need. Mapping a checkpoint configuration's nested `rope_scaling`
/// object into this type remains adapter wiring, not an inference claim.
/// Scalar parameters are FP32; arbitrary Python FP64 parameters must not be
/// silently narrowed to this API near a `YaRN` correction boundary. The pinned
/// V4.1 parameter values are exactly representable in FP32.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RotaryFrequencyParameters {
    rotary_width: NonZeroUsize,
    original_sequence_length: usize,
    base: f32,
    factor: f32,
    beta_fast: f32,
    beta_slow: f32,
}

impl RotaryFrequencyParameters {
    /// Validates parameters for source-compatible frequency generation.
    ///
    /// `original_sequence_length == 0` selects the source's non-YaRN path, so
    /// `factor` and the beta values are deliberately not interpreted there.
    ///
    /// # Errors
    ///
    /// Returns [`RotaryFrequencyError`] if the rotary width cannot be split
    /// into adjacent complex pairs or if an input required by the selected
    /// source path is not finite and usable.
    #[allow(
        clippy::float_cmp,
        reason = "base exactly one makes the YaRN logarithmic denominator zero"
    )]
    pub fn new(
        rotary_width: NonZeroUsize,
        original_sequence_length: usize,
        base: f32,
        factor: f32,
        beta_fast: f32,
        beta_slow: f32,
    ) -> Result<Self, RotaryFrequencyError> {
        if !rotary_width.get().is_multiple_of(2) {
            return Err(RotaryFrequencyError::OddRotaryWidth {
                width: rotary_width.get(),
            });
        }
        if !base.is_finite() || base <= 0.0 {
            return Err(RotaryFrequencyError::InvalidBase);
        }
        if original_sequence_length > 0
            && (base == 1.0
                || !factor.is_finite()
                || factor <= 0.0
                || !beta_fast.is_finite()
                || beta_fast <= 0.0
                || !beta_slow.is_finite()
                || beta_slow <= 0.0)
        {
            return Err(RotaryFrequencyError::InvalidYarnParameters);
        }
        Ok(Self {
            rotary_width,
            original_sequence_length,
            base,
            factor,
            beta_fast,
            beta_slow,
        })
    }

    /// Returns the source rotary-tail width in scalar values.
    #[must_use]
    pub const fn rotary_width(self) -> NonZeroUsize {
        self.rotary_width
    }

    /// Generates one contiguous, bounded position range of complex frequencies.
    ///
    /// This is numerically equivalent to slicing the rows
    /// `[start_position..start_position + positions]` from the source helper's
    /// output, while avoiding allocation for earlier positions. Results are in
    /// row-major `(position, adjacent-complex-pair)` order accepted by the
    /// CPU rotary-tail helper and, when enabled, its optional Metal path.
    ///
    /// Like the source's FP32 outer product, absolute positions are converted
    /// to `f32`; this does not promise exact integer-position distinction above
    /// the FP32 integer range.
    ///
    /// # Errors
    ///
    /// Returns [`RotaryFrequencyError`] if the requested position range or
    /// output size overflows, allocation fails, or source-equivalent FP32 math
    /// produces a non-finite frequency.
    #[allow(
        clippy::cast_precision_loss,
        reason = "match the pinned source's explicit FP32 index and position arithmetic"
    )]
    pub fn frequencies(
        self,
        start_position: usize,
        positions: NonZeroUsize,
    ) -> Result<Vec<RotaryFrequency>, RotaryFrequencyError> {
        let pairs = self.rotary_width.get() / 2;
        let end_position = start_position
            .checked_add(positions.get())
            .ok_or(RotaryFrequencyError::PositionRangeOverflow)?;
        let output_len = positions
            .get()
            .checked_mul(pairs)
            .ok_or(RotaryFrequencyError::OutputLengthOverflow)?;
        let mut output = Vec::new();
        output.try_reserve_exact(output_len).map_err(|_| {
            RotaryFrequencyError::AllocationFailed {
                elements: output_len,
            }
        })?;

        let yarn = self.yarn_bounds()?;
        let mut pair_frequencies = Vec::new();
        pair_frequencies
            .try_reserve_exact(pairs)
            .map_err(|_| RotaryFrequencyError::AllocationFailed { elements: pairs })?;
        for pair in 0..pairs {
            let exponent = (pair * 2) as f32 / self.rotary_width.get() as f32;
            let mut frequency = self.base.powf(exponent).recip();
            if let Some((low, high)) = yarn {
                let ramp = ((pair as f32 - low) / (high - low).max(1e-3)).clamp(0.0, 1.0);
                let smooth = 1.0 - ramp;
                frequency = frequency / self.factor * (1.0 - smooth) + frequency * smooth;
            }
            pair_frequencies.push(frequency);
        }
        for position in start_position..end_position {
            for (pair, frequency) in pair_frequencies.iter().copied().enumerate() {
                let angle = position as f32 * frequency;
                let (imaginary, real) = angle.sin_cos();
                if !real.is_finite() || !imaginary.is_finite() {
                    return Err(RotaryFrequencyError::NonFiniteOutput { position, pair });
                }
                output.push(RotaryFrequency { real, imaginary });
            }
        }
        Ok(output)
    }

    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        reason = "source computes scalar correction bounds in FP64 then applies them to an FP32 ramp"
    )]
    fn yarn_bounds(self) -> Result<Option<(f32, f32)>, RotaryFrequencyError> {
        if self.original_sequence_length == 0 {
            return Ok(None);
        }
        let width = self.rotary_width.get() as f64;
        let base = f64::from(self.base);
        let original = self.original_sequence_length as f64;
        let corrected_dimension = |rotations: f32| {
            width * (original / (f64::from(rotations) * 2.0 * std::f64::consts::PI)).ln()
                / (2.0 * base.ln())
        };
        let low = corrected_dimension(self.beta_fast).floor().max(0.0);
        let high = corrected_dimension(self.beta_slow).ceil().min(width - 1.0);
        if !low.is_finite() || !high.is_finite() || low > f64::from(f32::MAX) {
            return Err(RotaryFrequencyError::InvalidYarnParameters);
        }
        Ok(Some((low as f32, high as f32)))
    }
}

/// Errors from bounded V4.1 RoPE-frequency generation.
#[derive(Clone, Debug, Error, Eq, PartialEq)]
#[non_exhaustive]
pub enum RotaryFrequencyError {
    /// The source's adjacent-complex layout requires an even scalar width.
    #[error("rotary width {width} must be even")]
    OddRotaryWidth { width: usize },
    /// The source base must be finite and strictly positive.
    #[error("rotary base must be finite and strictly positive")]
    InvalidBase,
    /// The `YaRN` branch would be undefined or non-finite.
    #[error("YaRN parameters must be finite and strictly positive, with base other than one")]
    InvalidYarnParameters,
    /// The requested exclusive end position could not be represented.
    #[error("rotary frequency position range overflows usize")]
    PositionRangeOverflow,
    /// The requested result element count could not be represented.
    #[error("rotary frequency output element count overflows usize")]
    OutputLengthOverflow,
    /// The allocator could not reserve the bounded result buffer.
    #[error("could not allocate {elements} rotary frequencies")]
    AllocationFailed { elements: usize },
    /// Source-equivalent FP32 arithmetic produced a non-finite frequency.
    #[error("rotary frequency at position {position}, pair {pair} is not finite")]
    NonFiniteOutput { position: usize, pair: usize },
}

/// Validated shape of a contiguous V4.1 rotary tail.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RotaryTailLayout {
    batches: NonZeroUsize,
    positions: NonZeroUsize,
    heads: NonZeroUsize,
    pairs: NonZeroUsize,
}

impl RotaryTailLayout {
    /// Creates a layout whose final scalar dimension is `2 * pairs`.
    ///
    /// A rank-three upstream input uses `heads = 1`; rank-four attention
    /// inputs supply their actual head count.
    ///
    /// # Errors
    ///
    /// Returns [`RotaryError::LayoutOverflow`] when the scalar or frequency
    /// element count cannot be represented by `usize`.
    pub fn new(
        batches: NonZeroUsize,
        positions: NonZeroUsize,
        heads: NonZeroUsize,
        pairs: NonZeroUsize,
    ) -> Result<Self, RotaryError> {
        let layout = Self {
            batches,
            positions,
            heads,
            pairs,
        };
        let _ = layout.value_len()?;
        Ok(layout)
    }

    fn value_len(self) -> Result<usize, RotaryError> {
        self.batches
            .get()
            .checked_mul(self.positions.get())
            .and_then(|value| value.checked_mul(self.heads.get()))
            .and_then(|value| value.checked_mul(self.pairs.get()))
            .and_then(|value| value.checked_mul(2))
            .ok_or(RotaryError::LayoutOverflow)
    }

    fn frequency_len(self) -> Result<usize, RotaryError> {
        self.positions
            .get()
            .checked_mul(self.pairs.get())
            .ok_or(RotaryError::LayoutOverflow)
    }
}

/// Whether to apply the forward or conjugate (inverse) rotation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RotaryDirection {
    /// Multiply by the supplied complex frequencies.
    Forward,
    /// Multiply by their complex conjugates.
    Inverse,
}

/// Errors from V4.1 rotary-tail qualification.
#[derive(Clone, Debug, Error, Eq, PartialEq)]
#[non_exhaustive]
pub enum RotaryError {
    /// Shape arithmetic could not be represented by `usize`.
    #[error("rotary tail layout element count overflows usize")]
    LayoutOverflow,
    /// The input scalar buffer does not match the validated shape.
    #[error("rotary tail contains {actual} scalar values, expected {expected}")]
    ValueLengthMismatch { actual: usize, expected: usize },
    /// The frequency buffer does not contain one value per sequence/pair.
    #[error("rotary frequencies contain {actual} pairs, expected {expected}")]
    FrequencyLengthMismatch { actual: usize, expected: usize },
    /// One input scalar was not finite.
    #[error("rotary tail value at scalar index {index} is not finite")]
    NonFiniteValue { index: usize },
    /// A complex frequency component was not finite.
    #[error("rotary frequency components must be finite")]
    NonFiniteFrequency,
}

/// Errors from the Metal V4.1 rotary-tail qualification operation.
#[cfg(feature = "metal")]
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum RotaryMetalError {
    /// The typed CPU-side layout is not suitable for this operation.
    #[error(transparent)]
    Contract(#[from] RotaryError),
    /// A validated dimension cannot be represented by MLX's signed shape type.
    #[error("{field} does not fit MLX's shape representation")]
    DimensionOutOfRange { field: &'static str },
    /// MLX could not construct, evaluate, or read back the Metal graph.
    #[error("MLX Metal rotary-tail evaluation failed: {0}")]
    Mlx(#[from] mlx_rs::error::Exception),
}

/// Rotates a contiguous V4.1 `RoPE` tail in place.
///
/// The layout makes the upstream broadcasting explicit: each frequency at
/// `(position, pair)` is reused across every batch and head. Validation happens
/// before any write, so a rejected buffer remains unchanged.
///
/// This preserves the pinned FP32 operation's ordinary IEEE-754 arithmetic.
/// Finite inputs and frequencies can therefore produce a non-finite output by
/// overflow; this helper does not add a second output scan that the source does
/// not perform.
///
/// # Errors
///
/// Returns [`RotaryError`] for incompatible lengths, non-finite input, or
/// unrepresentable layout arithmetic.
pub fn rotate_tail(
    values: &mut [f32],
    layout: RotaryTailLayout,
    frequencies: &[RotaryFrequency],
    direction: RotaryDirection,
) -> Result<(), RotaryError> {
    validate_inputs(values, layout, frequencies)?;

    let positions = layout.positions.get();
    let heads = layout.heads.get();
    let pairs = layout.pairs.get();
    for batch in 0..layout.batches.get() {
        for position in 0..positions {
            for head in 0..heads {
                for pair in 0..pairs {
                    let value_index =
                        (((batch * positions + position) * heads + head) * pairs + pair) * 2;
                    let frequency = frequencies[position * pairs + pair];
                    let imaginary = match direction {
                        RotaryDirection::Forward => frequency.imaginary,
                        RotaryDirection::Inverse => -frequency.imaginary,
                    };
                    let real = values[value_index];
                    let input_imaginary = values[value_index + 1];
                    values[value_index] = real * frequency.real - input_imaginary * imaginary;
                    values[value_index + 1] = real * imaginary + input_imaginary * frequency.real;
                }
            }
        }
    }
    Ok(())
}

fn validate_inputs(
    values: &[f32],
    layout: RotaryTailLayout,
    frequencies: &[RotaryFrequency],
) -> Result<(), RotaryError> {
    let expected_values = layout.value_len()?;
    if values.len() != expected_values {
        return Err(RotaryError::ValueLengthMismatch {
            actual: values.len(),
            expected: expected_values,
        });
    }
    let expected_frequencies = layout.frequency_len()?;
    if frequencies.len() != expected_frequencies {
        return Err(RotaryError::FrequencyLengthMismatch {
            actual: frequencies.len(),
            expected: expected_frequencies,
        });
    }
    if let Some((index, _)) = values
        .iter()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        return Err(RotaryError::NonFiniteValue { index });
    }
    Ok(())
}

/// Rotates a contiguous V4.1 `RoPE` tail with an MLX Metal graph.
///
/// The caller supplies the same validated logical layout and external complex
/// frequencies as [`rotate_tail`]. The graph retains adjacent-pair semantics
/// while broadcasting one `(position, pair)` frequency across batch and head.
/// It is a diagnostic CPU-to-GPU round trip, not an in-place GPU cache update
/// or a fused attention kernel.
///
/// Like the pinned FP32 source, this deliberately has no non-finite-output
/// guarantee: finite inputs may overflow during arithmetic.
///
/// # Errors
///
/// Returns [`RotaryMetalError`] for an invalid typed contract, MLX shape range,
/// or MLX graph failure.
#[cfg(feature = "metal")]
pub fn rotate_tail_metal(
    values: &[f32],
    layout: RotaryTailLayout,
    frequencies: &[RotaryFrequency],
    direction: RotaryDirection,
) -> Result<Vec<f32>, RotaryMetalError> {
    validate_inputs(values, layout, frequencies)?;
    let batches = as_i32(layout.batches.get(), "batches")?;
    let positions = as_i32(layout.positions.get(), "positions")?;
    let heads = as_i32(layout.heads.get(), "heads")?;
    let pairs = as_i32(layout.pairs.get(), "pairs")?;
    let values_len = as_i32(layout.value_len()?, "scalar values")?;
    let frequency_real = frequencies
        .iter()
        .map(|frequency| frequency.real)
        .collect::<Vec<_>>();
    let frequency_imaginary = frequencies
        .iter()
        .map(|frequency| frequency.imaginary)
        .collect::<Vec<_>>();
    let stream = StreamOrDevice::gpu();
    let values = Array::from_slice(values, &[batches, positions, heads, pairs, 2]);
    let real = values.try_index_device((.., .., .., .., 0_i32), &stream)?;
    let imaginary = values.try_index_device((.., .., .., .., 1_i32), &stream)?;
    let frequency_real = Array::from_slice(&frequency_real, &[1, positions, 1, pairs]);
    let frequency_imaginary = Array::from_slice(&frequency_imaginary, &[1, positions, 1, pairs]);
    let frequency_imaginary = match direction {
        RotaryDirection::Forward => frequency_imaginary,
        RotaryDirection::Inverse => frequency_imaginary.negative_device(&stream)?,
    };
    let output_real = real
        .multiply_device(&frequency_real, &stream)?
        .subtract_device(
            &imaginary.multiply_device(&frequency_imaginary, &stream)?,
            &stream,
        )?;
    let output_imaginary = real
        .multiply_device(&frequency_imaginary, &stream)?
        .add_device(
            &imaginary.multiply_device(&frequency_real, &stream)?,
            &stream,
        )?;
    let output = mlx_rs::ops::stack_axis_device(&[&output_real, &output_imaginary], -1, &stream)?
        .reshape_device(&[values_len], &stream)?;
    output.eval()?;
    Ok(output.as_slice::<f32>().to_vec())
}

#[cfg(feature = "metal")]
fn as_i32(value: usize, field: &'static str) -> Result<i32, RotaryMetalError> {
    i32::try_from(value).map_err(|_| RotaryMetalError::DimensionOutOfRange { field })
}

#[cfg(test)]
mod tests {
    use super::{RotaryFrequencyError, RotaryFrequencyParameters};
    use std::num::NonZeroUsize;

    use serde::Deserialize;

    #[cfg(feature = "metal")]
    use super::rotate_tail_metal;
    use super::{RotaryDirection, RotaryError, RotaryFrequency, RotaryTailLayout, rotate_tail};

    #[derive(Debug, Deserialize)]
    struct Fixture {
        schema_version: u8,
        source: FixtureSource,
        cases: Vec<FixtureCase>,
    }

    #[derive(Debug, Deserialize)]
    struct FixtureSource {
        revision: String,
        sha256: String,
        symbol: String,
    }

    #[derive(Debug, Deserialize)]
    struct FixtureCase {
        name: String,
        batches: usize,
        positions: usize,
        heads: usize,
        pairs: usize,
        direction: String,
        values: Vec<f32>,
        frequencies: Vec<[f32; 2]>,
        expected_values: Vec<f32>,
    }

    #[derive(Debug, Deserialize)]
    struct FrequencyFixture {
        schema_version: u8,
        source: FixtureSource,
        cases: Vec<FrequencyFixtureCase>,
    }

    #[derive(Debug, Deserialize)]
    struct FrequencyFixtureCase {
        name: String,
        rotary_width: usize,
        original_sequence_length: usize,
        base: f32,
        factor: f32,
        beta_fast: f32,
        beta_slow: f32,
        start_position: usize,
        positions: usize,
        batches: usize,
        heads: usize,
        direction: String,
        expected_frequencies: Vec<[f32; 2]>,
        values: Vec<f32>,
        expected_rotated_values: Vec<f32>,
    }

    fn nonzero(value: usize) -> NonZeroUsize {
        NonZeroUsize::new(value).expect("fixture dimensions are nonzero")
    }

    fn layout(batches: usize, positions: usize, heads: usize, pairs: usize) -> RotaryTailLayout {
        RotaryTailLayout::new(
            nonzero(batches),
            nonzero(positions),
            nonzero(heads),
            nonzero(pairs),
        )
        .expect("small test layout")
    }

    fn assert_same_bits(actual: &[f32], expected: &[f32]) {
        assert_eq!(actual.len(), expected.len());
        for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
            assert_eq!(
                actual.to_bits(),
                expected.to_bits(),
                "scalar {index} changed from {expected:?} to {actual:?}",
            );
        }
    }

    fn pinned_frequency_fixture() -> FrequencyFixture {
        let fixture: FrequencyFixture = serde_json::from_str(include_str!(
            "../../../../fixtures/deepseek-v41/rope-frequency-reference.json"
        ))
        .expect("fixture JSON is valid");
        assert_eq!(fixture.schema_version, 1);
        assert_eq!(
            fixture.source.revision,
            "dba1be0a40aa45a94ad051997016db3960a90277"
        );
        assert_eq!(
            fixture.source.sha256,
            "4e9ae23620edc8028ccc5d5fef552ab7fdc7dcd6f79608754fe9f67644056f65"
        );
        assert_eq!(fixture.source.symbol, "precompute_freqs_cis");
        fixture
    }

    #[test]
    fn generates_and_applies_pinned_rope_frequency_fixture() {
        #[cfg(feature = "metal")]
        let _guard = crate::GPU_TEST_LOCK.lock().unwrap();
        let fixture = pinned_frequency_fixture();
        for case in fixture.cases {
            let parameters = RotaryFrequencyParameters::new(
                nonzero(case.rotary_width),
                case.original_sequence_length,
                case.base,
                case.factor,
                case.beta_fast,
                case.beta_slow,
            )
            .unwrap_or_else(|error| panic!("{}: {error}", case.name));
            let frequencies = parameters
                .frequencies(case.start_position, nonzero(case.positions))
                .unwrap_or_else(|error| panic!("{}: {error}", case.name));
            assert_eq!(
                frequencies.len(),
                case.expected_frequencies.len(),
                "{} frequency length",
                case.name
            );
            for (index, (actual, [expected_real, expected_imaginary])) in frequencies
                .iter()
                .zip(&case.expected_frequencies)
                .enumerate()
            {
                assert!(
                    (actual.real() - expected_real).abs() <= 0.000_001
                        && (actual.imaginary() - expected_imaginary).abs() <= 0.000_001,
                    "{} frequency {index}: actual ({}, {}), expected ({expected_real}, {expected_imaginary})",
                    case.name,
                    actual.real(),
                    actual.imaginary(),
                );
            }

            let direction = match case.direction.as_str() {
                "forward" => RotaryDirection::Forward,
                "inverse" => RotaryDirection::Inverse,
                _ => panic!("{}: invalid fixture direction", case.name),
            };
            #[cfg(feature = "metal")]
            let metal_values = rotate_tail_metal(
                &case.values,
                layout(
                    case.batches,
                    case.positions,
                    case.heads,
                    case.rotary_width / 2,
                ),
                &frequencies,
                direction,
            )
            .unwrap_or_else(|error| panic!("{} Metal: {error}", case.name));
            let mut values = case.values;
            rotate_tail(
                &mut values,
                layout(
                    case.batches,
                    case.positions,
                    case.heads,
                    case.rotary_width / 2,
                ),
                &frequencies,
                direction,
            )
            .unwrap_or_else(|error| panic!("{}: {error}", case.name));
            assert_eq!(
                values.len(),
                case.expected_rotated_values.len(),
                "{} rotation length",
                case.name
            );
            for (index, (actual, expected)) in
                values.iter().zip(&case.expected_rotated_values).enumerate()
            {
                assert!(
                    (actual - expected).abs() <= 0.000_001,
                    "{} rotated scalar {index}: actual {actual}, expected {expected}",
                    case.name
                );
                #[cfg(feature = "metal")]
                assert!(
                    (metal_values[index] - expected).abs() <= 0.000_001,
                    "{} Metal rotated scalar {index}: actual {}, expected {expected}",
                    case.name,
                    metal_values[index],
                );
            }
        }
    }

    #[test]
    fn frequency_parameters_reject_invalid_selected_path_inputs() {
        assert_eq!(
            RotaryFrequencyParameters::new(nonzero(3), 0, 10_000.0, 0.0, 0.0, 0.0),
            Err(RotaryFrequencyError::OddRotaryWidth { width: 3 })
        );
        assert_eq!(
            RotaryFrequencyParameters::new(nonzero(4), 0, 0.0, 0.0, 0.0, 0.0),
            Err(RotaryFrequencyError::InvalidBase)
        );
        assert_eq!(
            RotaryFrequencyParameters::new(nonzero(4), 1, 1.0, 16.0, 32.0, 1.0),
            Err(RotaryFrequencyError::InvalidYarnParameters)
        );
        let parameters = RotaryFrequencyParameters::new(nonzero(4), 0, 10_000.0, 0.0, 0.0, 0.0)
            .expect("non-YaRN ignores unused YaRN inputs like the source");
        assert!(matches!(
            parameters.frequencies(usize::MAX, nonzero(1)),
            Err(RotaryFrequencyError::PositionRangeOverflow)
        ));
    }

    #[test]
    fn frequency_parameters_validate_the_selected_yarn_branch_only() {
        for base in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 0.0, -1.0] {
            assert_eq!(
                RotaryFrequencyParameters::new(nonzero(4), 0, base, 1.0, 1.0, 1.0),
                Err(RotaryFrequencyError::InvalidBase),
                "base {base:?}"
            );
        }

        for (factor, beta_fast, beta_slow) in [
            (f32::NAN, 1.0, 1.0),
            (f32::INFINITY, 1.0, 1.0),
            (f32::NEG_INFINITY, 1.0, 1.0),
            (0.0, 1.0, 1.0),
            (-1.0, 1.0, 1.0),
            (1.0, f32::NAN, 1.0),
            (1.0, f32::INFINITY, 1.0),
            (1.0, f32::NEG_INFINITY, 1.0),
            (1.0, 0.0, 1.0),
            (1.0, -1.0, 1.0),
            (1.0, 1.0, f32::NAN),
            (1.0, 1.0, f32::INFINITY),
            (1.0, 1.0, f32::NEG_INFINITY),
            (1.0, 1.0, 0.0),
            (1.0, 1.0, -1.0),
        ] {
            assert_eq!(
                RotaryFrequencyParameters::new(
                    nonzero(4),
                    1,
                    10_000.0,
                    factor,
                    beta_fast,
                    beta_slow
                ),
                Err(RotaryFrequencyError::InvalidYarnParameters),
                "YaRN ({factor:?}, {beta_fast:?}, {beta_slow:?})"
            );
            RotaryFrequencyParameters::new(nonzero(4), 0, 10_000.0, factor, beta_fast, beta_slow)
                .expect("non-YaRN does not interpret YaRN-only parameters");
        }
    }

    #[test]
    fn frequency_requests_fail_before_large_allocation() {
        let small = RotaryFrequencyParameters::new(nonzero(2), 0, 10_000.0, 1.0, 1.0, 1.0)
            .expect("valid parameters");
        assert_eq!(
            small.frequencies(usize::MAX, nonzero(1)),
            Err(RotaryFrequencyError::PositionRangeOverflow)
        );

        let wide =
            RotaryFrequencyParameters::new(nonzero(usize::MAX - 1), 0, 10_000.0, 1.0, 1.0, 1.0)
                .expect("even width is syntactically valid");
        assert_eq!(
            wide.frequencies(0, nonzero(3)),
            Err(RotaryFrequencyError::OutputLengthOverflow)
        );

        assert_eq!(
            wide.frequencies(0, nonzero(1)),
            Err(RotaryFrequencyError::AllocationFailed {
                elements: (usize::MAX - 1) / 2,
            })
        );
    }

    #[test]
    fn frequency_ranges_are_slices_of_the_same_position_prefix() {
        let cases = [
            (4, 0, 10_000.0, 0.0, f32::NAN, f32::NEG_INFINITY),
            (6, 8, 10_000.0, 2.0, 16.0, 1.0),
            (8, 16, 1_000_000.0, 4.0, 32.0, 1.0),
        ];
        for (width, original, base, factor, beta_fast, beta_slow) in cases {
            let parameters = RotaryFrequencyParameters::new(
                nonzero(width),
                original,
                base,
                factor,
                beta_fast,
                beta_slow,
            )
            .expect("test parameters are valid for their selected path");
            let pairs = width / 2;
            for (start, count) in [(0, 1), (1, 2), (3, 3), (7, 2)] {
                let prefix = parameters
                    .frequencies(0, nonzero(start + count))
                    .expect("small prefix");
                let range = parameters
                    .frequencies(start, nonzero(count))
                    .expect("small range");
                assert_eq!(
                    range,
                    prefix[start * pairs..(start + count) * pairs],
                    "width {width}, original {original}, start {start}, count {count}"
                );
            }
        }
    }

    #[test]
    fn matches_pinned_official_cpu_reference_fixture() {
        let fixture: Fixture = serde_json::from_str(include_str!(
            "../../../../fixtures/deepseek-v41/rotary-reference.json"
        ))
        .expect("fixture JSON is valid");
        assert_eq!(fixture.schema_version, 1);
        assert_eq!(
            fixture.source.revision,
            "dba1be0a40aa45a94ad051997016db3960a90277"
        );
        assert_eq!(
            fixture.source.sha256,
            "4e9ae23620edc8028ccc5d5fef552ab7fdc7dcd6f79608754fe9f67644056f65"
        );
        assert_eq!(fixture.source.symbol, "apply_rotary_emb");

        for case in fixture.cases {
            let frequencies = case
                .frequencies
                .into_iter()
                .map(|[real, imaginary]| RotaryFrequency::new(real, imaginary))
                .collect::<Result<Vec<_>, _>>()
                .unwrap_or_else(|error| panic!("{}: {error}", case.name));
            let direction = match case.direction.as_str() {
                "forward" => RotaryDirection::Forward,
                "inverse" => RotaryDirection::Inverse,
                _ => panic!("{}: invalid fixture direction", case.name),
            };
            let mut values = case.values;
            rotate_tail(
                &mut values,
                layout(case.batches, case.positions, case.heads, case.pairs),
                &frequencies,
                direction,
            )
            .unwrap_or_else(|error| panic!("{}: {error}", case.name));
            assert_eq!(values.len(), case.expected_values.len(), "{}", case.name);
            for (index, (actual, expected)) in values.iter().zip(&case.expected_values).enumerate()
            {
                assert!(
                    (actual - expected).abs() <= 0.000_001,
                    "{} scalar {index}: actual {actual}, expected {expected}",
                    case.name
                );
            }
        }
    }

    #[cfg(feature = "metal")]
    #[test]
    fn metal_matches_pinned_official_cpu_reference_fixture() {
        let _guard = crate::GPU_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let fixture: Fixture = serde_json::from_str(include_str!(
            "../../../../fixtures/deepseek-v41/rotary-reference.json"
        ))
        .expect("fixture JSON is valid");
        assert_eq!(fixture.schema_version, 1);
        assert_eq!(
            fixture.source.revision,
            "dba1be0a40aa45a94ad051997016db3960a90277"
        );
        assert_eq!(
            fixture.source.sha256,
            "4e9ae23620edc8028ccc5d5fef552ab7fdc7dcd6f79608754fe9f67644056f65"
        );
        assert_eq!(fixture.source.symbol, "apply_rotary_emb");

        for case in fixture.cases {
            let frequencies = case
                .frequencies
                .into_iter()
                .map(|[real, imaginary]| RotaryFrequency::new(real, imaginary))
                .collect::<Result<Vec<_>, _>>()
                .unwrap_or_else(|error| panic!("{}: {error}", case.name));
            let direction = match case.direction.as_str() {
                "forward" => RotaryDirection::Forward,
                "inverse" => RotaryDirection::Inverse,
                _ => panic!("{}: invalid fixture direction", case.name),
            };
            let actual = rotate_tail_metal(
                &case.values,
                layout(case.batches, case.positions, case.heads, case.pairs),
                &frequencies,
                direction,
            )
            .unwrap_or_else(|error| panic!("{}: {error}", case.name));
            assert_eq!(actual.len(), case.expected_values.len(), "{}", case.name);
            for (index, (actual, expected)) in actual.iter().zip(&case.expected_values).enumerate()
            {
                assert!(
                    (actual - expected).abs() <= 0.000_001,
                    "{} scalar {index}: actual {actual}, expected {expected}",
                    case.name
                );
            }
        }
    }

    #[test]
    fn rejects_invalid_buffers_before_mutating_values() {
        let frequencies = [RotaryFrequency::new(1.0, 0.0).expect("finite")];
        let single_pair_layout = layout(1, 1, 1, 1);
        let mut non_finite = [f32::NAN, 2.0];
        let original = non_finite;
        assert_eq!(
            rotate_tail(
                &mut non_finite,
                single_pair_layout,
                &frequencies,
                RotaryDirection::Forward
            ),
            Err(RotaryError::NonFiniteValue { index: 0 })
        );
        assert_same_bits(&non_finite, &original);

        let mut wrong_length = [3.0];
        let original = wrong_length;
        assert!(matches!(
            rotate_tail(
                &mut wrong_length,
                single_pair_layout,
                &frequencies,
                RotaryDirection::Forward
            ),
            Err(RotaryError::ValueLengthMismatch { .. })
        ));
        assert_same_bits(&wrong_length, &original);

        let mut later_non_finite = [1.0, 2.0, f32::INFINITY, f32::NEG_INFINITY];
        let original = later_non_finite;
        assert_eq!(
            rotate_tail(
                &mut later_non_finite,
                layout(1, 1, 1, 2),
                &[
                    RotaryFrequency::new(1.0, 0.0).expect("finite"),
                    RotaryFrequency::new(1.0, 0.0).expect("finite"),
                ],
                RotaryDirection::Forward,
            ),
            Err(RotaryError::NonFiniteValue { index: 2 })
        );
        assert_same_bits(&later_non_finite, &original);

        let mut negative_infinity = [1.0, 2.0, 3.0, f32::NEG_INFINITY];
        let original = negative_infinity;
        assert_eq!(
            rotate_tail(
                &mut negative_infinity,
                layout(1, 1, 1, 2),
                &[
                    RotaryFrequency::new(1.0, 0.0).expect("finite"),
                    RotaryFrequency::new(1.0, 0.0).expect("finite"),
                ],
                RotaryDirection::Forward,
            ),
            Err(RotaryError::NonFiniteValue { index: 3 })
        );
        assert_same_bits(&negative_infinity, &original);
    }

    #[test]
    fn rejects_non_finite_frequencies_and_frequency_length_mismatch() {
        assert_eq!(
            RotaryFrequency::new(f32::INFINITY, 0.0),
            Err(RotaryError::NonFiniteFrequency)
        );
        let mut values = [1.0, 2.0, 3.0, 4.0];
        let original = values;
        assert!(matches!(
            rotate_tail(
                &mut values,
                layout(1, 2, 1, 1),
                &[RotaryFrequency::new(1.0, 0.0).expect("finite")],
                RotaryDirection::Forward
            ),
            Err(RotaryError::FrequencyLengthMismatch { .. })
        ));
        assert_same_bits(&values, &original);
    }

    #[test]
    fn preserves_upstream_finite_input_overflow_behavior() {
        let mut values = [f32::MAX, 0.0];
        rotate_tail(
            &mut values,
            layout(1, 1, 1, 1),
            &[RotaryFrequency::new(2.0, 0.0).expect("finite")],
            RotaryDirection::Forward,
        )
        .expect("finite inputs are accepted");
        assert!(values[0].is_infinite() && values[0].is_sign_positive());
        assert_eq!(values[1].to_bits(), 0.0_f32.to_bits());
    }

    #[test]
    fn rejects_unrepresentable_layouts() {
        assert_eq!(
            RotaryTailLayout::new(nonzero(usize::MAX), nonzero(2), nonzero(1), nonzero(1),),
            Err(RotaryError::LayoutOverflow)
        );
    }
}
