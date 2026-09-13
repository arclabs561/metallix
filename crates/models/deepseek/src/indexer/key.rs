//! Bounded preparation of V4.1 compressed-latent index keys.
//!
//! This owns the numerical path from a RoPE-free compressor latent through
//! `wk`, `k_norm`, rotary, and Index32/E8M0 FP4 reconstruction. It does not
//! decide when a compressor publishes a latent, retain a cache, score a query,
//! mask candidates, or select positions.

use std::num::NonZeroUsize;

use thiserror::Error;

use crate::{
    RotaryDirection, RotaryError, RotaryFrequency, RotaryTailLayout,
    norm::{MAX_RMS_NORM_WIDTH, RmsNormError},
    precision::{
        Bf16LinearError, Fp4ActivationError, Fp4ActivationMode, MAX_BF16_LINEAR_ELEMENTS,
        bf16_linear_reference, bf16_to_f32, f32_to_bf16_rne, requantize_bf16_activations_e2m1,
    },
    rms_norm_bf16_reference, rotate_tail,
};

const MAX_INDEX_KEY_ELEMENTS: usize = MAX_BF16_LINEAR_ELEMENTS;
const MAX_INDEX_KEY_WORK: usize = 1 << 24;

/// Explicit source geometry for a compressed-latent index-key preparation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct IndexKeyLayout {
    batches: NonZeroUsize,
    latent_dimension: NonZeroUsize,
    key_dimension: NonZeroUsize,
    rope_pairs: NonZeroUsize,
    norm_epsilon: f32,
}

impl IndexKeyLayout {
    /// Creates a bounded source-shaped index-key layout.
    pub fn new(
        batches: NonZeroUsize,
        latent_dimension: NonZeroUsize,
        key_dimension: NonZeroUsize,
        rope_pairs: NonZeroUsize,
        norm_epsilon: f32,
    ) -> Result<Self, IndexKeyLayoutError> {
        let rope_width =
            rope_pairs
                .get()
                .checked_mul(2)
                .ok_or(IndexKeyLayoutError::ShapeOverflow {
                    field: "rope width",
                })?;
        if rope_width > key_dimension.get() {
            return Err(IndexKeyLayoutError::RopeExceedsKey {
                rope_width,
                key_dimension: key_dimension.get(),
            });
        }
        if !key_dimension.get().is_multiple_of(32) {
            return Err(IndexKeyLayoutError::UngroupedKeyWidth {
                width: key_dimension.get(),
            });
        }
        if key_dimension.get() > MAX_RMS_NORM_WIDTH {
            return Err(IndexKeyLayoutError::KeyWidthTooLarge {
                width: key_dimension.get(),
                maximum: MAX_RMS_NORM_WIDTH,
            });
        }
        if !norm_epsilon.is_finite() || norm_epsilon <= 0.0 {
            return Err(IndexKeyLayoutError::InvalidEpsilon);
        }
        for (field, elements) in [
            (
                "one-position latent",
                layout_product(
                    &[batches.get(), latent_dimension.get()],
                    "one-position latent",
                )?,
            ),
            (
                "one-position key",
                layout_product(&[batches.get(), key_dimension.get()], "one-position key")?,
            ),
            (
                "projection weight",
                layout_product(
                    &[key_dimension.get(), latent_dimension.get()],
                    "projection weight",
                )?,
            ),
        ] {
            if elements > MAX_INDEX_KEY_ELEMENTS {
                return Err(IndexKeyLayoutError::ElementLimit { field, elements });
            }
        }
        Ok(Self {
            batches,
            latent_dimension,
            key_dimension,
            rope_pairs,
            norm_epsilon,
        })
    }

    /// Number of independent batch prefixes in this key layout.
    #[must_use]
    pub(crate) const fn batches(self) -> NonZeroUsize {
        self.batches
    }

    /// Width of one compressor latent consumed by `wk`.
    #[must_use]
    pub(crate) const fn latent_dimension(self) -> NonZeroUsize {
        self.latent_dimension
    }

    /// Width of one prepared key cached by the owner.
    #[must_use]
    pub(crate) const fn key_dimension(self) -> NonZeroUsize {
        self.key_dimension
    }

    /// Rotary complex-pair count shared with the owner compressed-KV path.
    #[must_use]
    pub(crate) const fn rope_pairs(self) -> NonZeroUsize {
        self.rope_pairs
    }
}

/// Borrowed BF16 weights used by [`prepare_index_keys`].
#[derive(Clone, Copy, Debug)]
pub struct IndexKeyWeights<'a> {
    wk: &'a [u16],
    norm: &'a [u16],
}

impl<'a> IndexKeyWeights<'a> {
    /// Borrows source-layout `wk` `[key_dimension, latent_dimension]` and
    /// `k_norm.weight` `[key_dimension]`; exact lengths are checked by the
    /// preparation call because they depend on its layout.
    #[must_use]
    pub const fn new(wk: &'a [u16], norm: &'a [u16]) -> Self {
        Self { wk, norm }
    }
}

/// Source-visible precision boundaries from index-key preparation.
#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub struct IndexKeyDiagnostic {
    /// BF16 `wk(latent)` before RMS normalization.
    pub projected: Vec<u16>,
    /// BF16 keys after `k_norm`, before rotary.
    pub normalized: Vec<u16>,
    /// BF16 keys after the rotary tail, before FP4 reconstruction.
    pub post_rope: Vec<u16>,
    /// BF16 keys after source-shaped G32/E8M0 FP4 reconstruction.
    pub post_fp4: Vec<u16>,
}

/// Invalid index-key geometry.
#[derive(Clone, Copy, Debug, Error, Eq, PartialEq)]
#[non_exhaustive]
pub enum IndexKeyLayoutError {
    #[error("index-key shape arithmetic overflowed for {field}")]
    ShapeOverflow { field: &'static str },
    #[error("rope width {rope_width} exceeds index-key dimension {key_dimension}")]
    RopeExceedsKey {
        rope_width: usize,
        key_dimension: usize,
    },
    #[error("index-key width {width} is not divisible by group 32")]
    UngroupedKeyWidth { width: usize },
    #[error("index-key width {width} exceeds RMSNorm maximum {maximum}")]
    KeyWidthTooLarge { width: usize, maximum: usize },
    #[error("index-key RMS epsilon must be finite and positive")]
    InvalidEpsilon,
    #[error("index-key {field} has {elements} elements, maximum is {MAX_INDEX_KEY_ELEMENTS}")]
    ElementLimit {
        field: &'static str,
        elements: usize,
    },
}

/// Rejected index-key preparation input.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum IndexKeyError {
    #[error(transparent)]
    Layout(#[from] IndexKeyLayoutError),
    #[error(transparent)]
    Linear(#[from] Bf16LinearError),
    #[error(transparent)]
    Norm(#[from] RmsNormError),
    #[error(transparent)]
    Rotary(#[from] RotaryError),
    #[error(transparent)]
    Fp4(#[from] Fp4ActivationError),
    #[error("index-key latent length is {actual}; expected a nonempty multiple of {stride}")]
    LatentLength { actual: usize, stride: usize },
    #[error("index-key {field} length is {actual}, expected {expected}")]
    WeightLength {
        field: &'static str,
        actual: usize,
        expected: usize,
    },
    #[error("index-key frequencies length is {actual}, expected {expected}")]
    FrequencyLength { actual: usize, expected: usize },
    #[error("index-key {field} has {elements} elements, maximum is {MAX_INDEX_KEY_ELEMENTS}")]
    ElementLimit {
        field: &'static str,
        elements: usize,
    },
    #[error("index-key scalar projection work {terms} exceeds {MAX_INDEX_KEY_WORK} terms")]
    WorkloadTooLarge { terms: usize },
    #[error("index-key shape arithmetic overflowed for {field}")]
    ShapeOverflow { field: &'static str },
    #[error("nonfinite BF16 index-key {field} at position {position}")]
    NonFiniteInput {
        field: &'static str,
        position: usize,
    },
    #[error("index-key rotary result could not narrow to finite BF16 at element {element}")]
    NonFiniteRotary {
        /// Flat element in `[batch, compressed_position, key_dimension]` storage.
        element: usize,
    },
    #[error("could not allocate {elements} index-key {field} elements")]
    AllocationFailed {
        field: &'static str,
        elements: usize,
    },
}

/// Prepares source-shaped FP4 index keys from immutable compressor latents.
///
/// `latent` is BF16 `[batch, compressed_position, latent_dimension]` and
/// `frequencies` is the call-local `[compressed_position, rope_pair]` slice
/// for the *first token* of each completed compressed group. `wk` projects
/// each latent, `k_norm` normalizes each projected key, then rotary and FP4
/// reconstruction happen in that order. The caller owns compressor scheduling,
/// position derivation, cache publication, scoring, and selection.
///
/// This is a CPU scalar precision-staging reference exercised by hand-staged
/// composition over source-captured latents and rotary frequencies. It is not
/// GPU execution, a captured owner-layer checkpoint-weight qualification,
/// generic tensor semantics, cache behavior, or arbitrary `PyTorch`
/// GEMM-reduction parity. All bounded shape, work, length, and finite-input
/// checks happen before staging output allocation; no partial diagnostic
/// escapes on error.
pub fn prepare_index_keys(
    latent: &[u16],
    frequencies: &[RotaryFrequency],
    weights: IndexKeyWeights<'_>,
    layout: IndexKeyLayout,
) -> Result<IndexKeyDiagnostic, IndexKeyError> {
    let shape = Shape::new(latent, frequencies, weights, layout)?;
    validate_finite(latent, "latent")?;
    validate_finite(weights.wk, "wk")?;
    validate_finite(weights.norm, "norm")?;

    let mut projected = reserve(shape.key_elements, "projected")?;
    bf16_linear_reference(
        latent,
        weights.wk,
        shape.rows,
        layout.latent_dimension.get(),
        layout.key_dimension.get(),
        &mut projected,
    )?;

    let mut normalized = reserve(shape.key_elements, "normalized")?;
    for (input, output) in projected
        .chunks_exact(layout.key_dimension.get())
        .zip(normalized.chunks_exact_mut(layout.key_dimension.get()))
    {
        rms_norm_bf16_reference(input, weights.norm, layout.norm_epsilon, output)?;
    }

    let mut post_rope = reserve(shape.key_elements, "post_rope")?;
    post_rope.copy_from_slice(&normalized);
    rotate_key_tail(&mut post_rope, frequencies, layout, shape.positions)?;

    let mut post_fp4 = reserve(shape.key_elements, "post_fp4")?;
    requantize_bf16_activations_e2m1(
        &post_rope,
        shape.rows,
        layout.key_dimension.get(),
        Fp4ActivationMode::Index32E8m0,
        &mut post_fp4,
    )?;
    Ok(IndexKeyDiagnostic {
        projected,
        normalized,
        post_rope,
        post_fp4,
    })
}

#[derive(Clone, Copy)]
struct Shape {
    positions: usize,
    rows: usize,
    key_elements: usize,
}

impl Shape {
    fn new(
        latent: &[u16],
        frequencies: &[RotaryFrequency],
        weights: IndexKeyWeights<'_>,
        layout: IndexKeyLayout,
    ) -> Result<Self, IndexKeyError> {
        let latent_stride = product(
            &[layout.batches.get(), layout.latent_dimension.get()],
            "latent stride",
        )?;
        if latent.is_empty() || !latent.len().is_multiple_of(latent_stride) {
            return Err(IndexKeyError::LatentLength {
                actual: latent.len(),
                stride: latent_stride,
            });
        }
        let positions = latent.len() / latent_stride;
        let rows = product(&[layout.batches.get(), positions], "key rows")?;
        let key_elements = product(&[rows, layout.key_dimension.get()], "key output")?;
        let projection_weights = product(
            &[layout.key_dimension.get(), layout.latent_dimension.get()],
            "projection weights",
        )?;
        for (field, actual, expected) in [
            ("wk", weights.wk.len(), projection_weights),
            ("norm", weights.norm.len(), layout.key_dimension.get()),
        ] {
            if actual != expected {
                return Err(IndexKeyError::WeightLength {
                    field,
                    actual,
                    expected,
                });
            }
        }
        let expected_frequencies = product(&[positions, layout.rope_pairs.get()], "frequencies")?;
        if frequencies.len() != expected_frequencies {
            return Err(IndexKeyError::FrequencyLength {
                actual: frequencies.len(),
                expected: expected_frequencies,
            });
        }
        for (field, elements) in [
            ("latent", latent.len()),
            ("wk", projection_weights),
            ("key output", key_elements),
        ] {
            if elements > MAX_INDEX_KEY_ELEMENTS {
                return Err(IndexKeyError::ElementLimit { field, elements });
            }
        }
        let terms = product(
            &[key_elements, layout.latent_dimension.get()],
            "projection work",
        )?;
        if terms > MAX_INDEX_KEY_WORK {
            return Err(IndexKeyError::WorkloadTooLarge { terms });
        }
        Ok(Self {
            positions,
            rows,
            key_elements,
        })
    }
}

fn rotate_key_tail(
    output: &mut [u16],
    frequencies: &[RotaryFrequency],
    layout: IndexKeyLayout,
    positions: usize,
) -> Result<(), IndexKeyError> {
    let prefix = layout.key_dimension.get() - layout.rope_pairs.get() * 2;
    let tail_elements = product(
        &[layout.batches.get(), positions, layout.rope_pairs.get(), 2],
        "rotary tail",
    )?;
    let mut tail = Vec::new();
    tail.try_reserve_exact(tail_elements)
        .map_err(|_| IndexKeyError::AllocationFailed {
            field: "rotary tail",
            elements: tail_elements,
        })?;
    for key in output.chunks_exact(layout.key_dimension.get()) {
        tail.extend(key[prefix..].iter().map(|&bits| bf16_to_f32(bits)));
    }
    let rotary_layout = RotaryTailLayout::new(
        layout.batches,
        NonZeroUsize::new(positions).expect("validated nonempty latent positions"),
        NonZeroUsize::new(1).expect("one key head"),
        layout.rope_pairs,
    )?;
    rotate_tail(
        &mut tail,
        rotary_layout,
        frequencies,
        RotaryDirection::Forward,
    )?;
    for (row, (key, rotated_tail)) in output
        .chunks_exact_mut(layout.key_dimension.get())
        .zip(tail.chunks_exact(layout.rope_pairs.get() * 2))
        .enumerate()
    {
        for (offset, &value) in rotated_tail.iter().enumerate() {
            let bits = f32_to_bf16_rne(value);
            if !bf16_to_f32(bits).is_finite() {
                return Err(IndexKeyError::NonFiniteRotary {
                    element: row * layout.key_dimension.get() + prefix + offset,
                });
            }
            key[prefix + offset] = bits;
        }
    }
    Ok(())
}

fn product(values: &[usize], field: &'static str) -> Result<usize, IndexKeyError> {
    values.iter().try_fold(1_usize, |total, &value| {
        total
            .checked_mul(value)
            .ok_or(IndexKeyError::ShapeOverflow { field })
    })
}

fn layout_product(values: &[usize], field: &'static str) -> Result<usize, IndexKeyLayoutError> {
    values.iter().try_fold(1_usize, |total, &value| {
        total
            .checked_mul(value)
            .ok_or(IndexKeyLayoutError::ShapeOverflow { field })
    })
}

fn validate_finite(values: &[u16], field: &'static str) -> Result<(), IndexKeyError> {
    values
        .iter()
        .position(|&bits| !bf16_to_f32(bits).is_finite())
        .map_or(Ok(()), |position| {
            Err(IndexKeyError::NonFiniteInput { field, position })
        })
}

fn reserve(elements: usize, field: &'static str) -> Result<Vec<u16>, IndexKeyError> {
    let mut output = Vec::new();
    output
        .try_reserve_exact(elements)
        .map_err(|_| IndexKeyError::AllocationFailed { field, elements })?;
    output.resize(elements, 0);
    Ok(output)
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroUsize;

    use super::{
        IndexKeyError, IndexKeyLayout, IndexKeyLayoutError, IndexKeyWeights, prepare_index_keys,
    };
    use crate::RotaryFrequency;

    fn nz(value: usize) -> NonZeroUsize {
        NonZeroUsize::new(value).expect("nonzero test dimension")
    }

    fn bf16(value: f32) -> u16 {
        let bits = value.to_bits();
        u16::try_from(bits.wrapping_add(0x7fff + ((bits >> 16) & 1)) >> 16).expect("BF16 high half")
    }

    fn layout() -> IndexKeyLayout {
        IndexKeyLayout::new(nz(1), nz(16), nz(32), nz(1), 1.0e-5).expect("test layout")
    }

    #[test]
    fn projects_normalizes_rotates_and_requantizes_in_source_order() {
        let mut wk = vec![0_u16; 32 * 16];
        wk[30 * 16] = bf16(1.0);
        wk[31 * 16 + 1] = bf16(1.0);
        let mut latent = vec![0_u16; 16];
        latent[0] = bf16(2.0);
        latent[1] = bf16(3.0);
        let diagnostic = prepare_index_keys(
            &latent,
            &[RotaryFrequency::new(0.0, 1.0).expect("finite frequency")],
            IndexKeyWeights::new(&wk, &[bf16(1.0); 32]),
            layout(),
        )
        .expect("finite staged key");
        assert_eq!(diagnostic.projected[30], bf16(2.0));
        assert_eq!(diagnostic.projected[31], bf16(3.0));
        let inverse_rms = (13.0_f32 / 32.0 + 1.0e-5).sqrt().recip();
        assert_eq!(diagnostic.normalized[30], bf16(2.0 * inverse_rms));
        assert_eq!(diagnostic.post_rope[30], bf16(-3.0 * inverse_rms));
        assert_eq!(diagnostic.post_rope[31], bf16(2.0 * inverse_rms));
        assert_eq!(diagnostic.post_fp4.len(), 32);
    }

    #[test]
    fn rejects_key_width_epsilon_and_input_boundaries_before_staging() {
        assert!(matches!(
            IndexKeyLayout::new(nz(1), nz(16), nz(31), nz(1), 1.0),
            Err(IndexKeyLayoutError::UngroupedKeyWidth { width: 31 })
        ));
        assert!(matches!(
            IndexKeyLayout::new(nz(1), nz(16), nz(32), nz(1), 0.0),
            Err(IndexKeyLayoutError::InvalidEpsilon)
        ));
        let error = prepare_index_keys(&[0; 15], &[], IndexKeyWeights::new(&[], &[]), layout())
            .expect_err("misaligned latent precedes borrowed weight checks");
        assert!(matches!(
            error,
            IndexKeyError::LatentLength {
                actual: 15,
                stride: 16
            }
        ));
    }

    #[test]
    fn rejects_nonfinite_weights_before_output_allocation() {
        let error = prepare_index_keys(
            &[0; 16],
            &[RotaryFrequency::new(1.0, 0.0).expect("finite frequency")],
            IndexKeyWeights::new(&[0; 32 * 16], &[0x7fc0; 32]),
            layout(),
        )
        .expect_err("NaN learned norm is rejected");
        assert!(matches!(
            error,
            IndexKeyError::NonFiniteInput {
                field: "norm",
                position: 0
            }
        ));
    }
}
