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

#[cfg(feature = "metal")]
use crate::{RotaryMetalError, rotate_tail_metal};

#[cfg(feature = "metal")]
use mlx_rs::{Array, Dtype, StreamOrDevice, ops::indexing::TryIndexOp, transforms};

const MAX_INDEX_KEY_ELEMENTS: usize = MAX_BF16_LINEAR_ELEMENTS;
const MAX_INDEX_KEY_WORK: usize = 1 << 24;

/// Rotary implementation selected by a model-local index-key caller.
///
/// Scalar FP32 rotation is the source-authoritative default. The optional
/// Metal variant is a bounded `DeepSeek` diagnostic and does not establish a
/// general execution-backend contract.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[non_exhaustive]
pub enum IndexKeyRotaryExecution {
    /// Rotate the BF16-expanded key tail with the scalar FP32 reference.
    #[default]
    Scalar,
    /// Rotate the same FP32 tail with the bounded MLX Metal diagnostic.
    #[cfg(feature = "metal")]
    MetalFp32,
}

/// Complete implementation selected for bounded index-key preparation.
///
/// [`Self::Scalar`] remains the source-authoritative default. The Metal
/// choices are explicit diagnostics. Qualification covers captured operands;
/// they do not promise arbitrary BF16 matrix-reduction parity.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[non_exhaustive]
pub enum IndexKeyPreparationExecution {
    /// Use scalar projection, normalization, rotary, and FP4 staging.
    #[default]
    Scalar,
    /// Retain scalar projection/normalization and use the prior Metal rotary diagnostic.
    #[cfg(feature = "metal")]
    MetalRotaryFp32,
    /// Keep projection, normalization, and rotary connected on Metal before scalar FP4 staging.
    #[cfg(feature = "metal")]
    MetalPreFp4,
}

impl From<IndexKeyRotaryExecution> for IndexKeyPreparationExecution {
    fn from(value: IndexKeyRotaryExecution) -> Self {
        match value {
            IndexKeyRotaryExecution::Scalar => Self::Scalar,
            #[cfg(feature = "metal")]
            IndexKeyRotaryExecution::MetalFp32 => Self::MetalRotaryFp32,
        }
    }
}

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
    #[cfg(feature = "metal")]
    #[error(transparent)]
    MetalRotary(#[from] RotaryMetalError),
    #[cfg(feature = "metal")]
    #[error(transparent)]
    MetalPreparation(#[from] MetalKeyPreparationError),
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

/// Errors from bounded connected Metal index-key preparation.
#[cfg(feature = "metal")]
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum MetalKeyPreparationError {
    /// MLX could not construct, evaluate, or read the bounded preparation graph.
    #[error("MLX Metal index-key preparation failed: {0}")]
    Mlx(#[from] mlx_rs::error::Exception),
    /// A staged BF16 readback did not retain its validated source shape.
    #[error("Metal index-key {stage} readback has {actual} elements, expected {expected}")]
    Readback {
        /// Source-stage boundary.
        stage: &'static str,
        /// Readback element count.
        actual: usize,
        /// Expected source-shape element count.
        expected: usize,
    },
    /// A finite validated input produced a nonfinite staged BF16 result.
    #[error("Metal index-key {stage} has nonfinite BF16 at element {element}")]
    NonFinite {
        /// Source-stage boundary.
        stage: &'static str,
        /// Flat staged element.
        element: usize,
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
    prepare_index_keys_with_execution(
        latent,
        frequencies,
        weights,
        layout,
        IndexKeyPreparationExecution::Scalar,
    )
}

/// Prepares source-shaped FP4 index keys using the selected bounded rotary path.
///
/// This keeps the scalar projection, normalization, BF16 narrowing, and FP4
/// reconstruction boundaries identical to [`prepare_index_keys`]. Metal
/// rotation is a diagnostic CPU-to-GPU round trip; it does not make key-cache
/// storage or the remaining stages device resident.
pub fn prepare_index_keys_with_rotary_execution(
    latent: &[u16],
    frequencies: &[RotaryFrequency],
    weights: IndexKeyWeights<'_>,
    layout: IndexKeyLayout,
    rotary_execution: IndexKeyRotaryExecution,
) -> Result<IndexKeyDiagnostic, IndexKeyError> {
    prepare_index_keys_with_execution(
        latent,
        frequencies,
        weights,
        layout,
        rotary_execution.into(),
    )
}

/// Prepares source-shaped FP4 index keys using the selected bounded preparation path.
///
/// All variants retain the source-observed BF16 stage boundaries and scalar FP4
/// reconstruction. `MetalPreFp4` is qualified
/// only for the captured V4.1 operands in this module's fixture, rather than
/// for arbitrary BF16 matrix reductions.
pub(crate) fn prepare_index_keys_with_execution(
    latent: &[u16],
    frequencies: &[RotaryFrequency],
    weights: IndexKeyWeights<'_>,
    layout: IndexKeyLayout,
    execution: IndexKeyPreparationExecution,
) -> Result<IndexKeyDiagnostic, IndexKeyError> {
    match execution {
        IndexKeyPreparationExecution::Scalar => prepare_index_keys_scalar(
            latent,
            frequencies,
            weights,
            layout,
            IndexKeyRotaryExecution::Scalar,
        ),
        #[cfg(feature = "metal")]
        IndexKeyPreparationExecution::MetalRotaryFp32 => prepare_index_keys_scalar(
            latent,
            frequencies,
            weights,
            layout,
            IndexKeyRotaryExecution::MetalFp32,
        ),
        #[cfg(feature = "metal")]
        IndexKeyPreparationExecution::MetalPreFp4 => {
            let shape = Shape::new(latent, frequencies, weights, layout)?;
            validate_finite(latent, "latent")?;
            validate_finite(weights.wk, "wk")?;
            validate_finite(weights.norm, "norm")?;
            let stages =
                prepare_index_key_stages_metal(latent, frequencies, weights, layout, shape)?;
            let mut post_fp4 = reserve(shape.key_elements, "post_fp4")?;
            requantize_bf16_activations_e2m1(
                &stages.post_rope,
                shape.rows,
                layout.key_dimension.get(),
                Fp4ActivationMode::Index32E8m0,
                &mut post_fp4,
            )?;
            Ok(IndexKeyDiagnostic {
                projected: stages.projected,
                normalized: stages.normalized,
                post_rope: stages.post_rope,
                post_fp4,
            })
        }
    }
}

fn prepare_index_keys_scalar(
    latent: &[u16],
    frequencies: &[RotaryFrequency],
    weights: IndexKeyWeights<'_>,
    layout: IndexKeyLayout,
    rotary_execution: IndexKeyRotaryExecution,
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
    rotate_key_tail(
        &mut post_rope,
        frequencies,
        layout,
        shape.positions,
        rotary_execution,
    )?;

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
    execution: IndexKeyRotaryExecution,
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
    match execution {
        IndexKeyRotaryExecution::Scalar => rotate_tail(
            &mut tail,
            rotary_layout,
            frequencies,
            RotaryDirection::Forward,
        )?,
        #[cfg(feature = "metal")]
        IndexKeyRotaryExecution::MetalFp32 => {
            tail = rotate_tail_metal(&tail, rotary_layout, frequencies, RotaryDirection::Forward)?;
        }
    }
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

#[cfg(feature = "metal")]
struct MetalKeyStages {
    projected: Vec<u16>,
    normalized: Vec<u16>,
    post_rope: Vec<u16>,
}

#[cfg(feature = "metal")]
fn prepare_index_key_stages_metal(
    latent: &[u16],
    frequencies: &[RotaryFrequency],
    weights: IndexKeyWeights<'_>,
    layout: IndexKeyLayout,
    shape: Shape,
) -> Result<MetalKeyStages, MetalKeyPreparationError> {
    let rows = i32::try_from(shape.rows).expect("bounded key rows fit MLX dimensions");
    let latent_dim = i32::try_from(layout.latent_dimension.get())
        .expect("bounded latent width fits MLX dimensions");
    let key_dim =
        i32::try_from(layout.key_dimension.get()).expect("bounded key width fits MLX dimensions");
    let batches = i32::try_from(layout.batches.get()).expect("bounded batches fit MLX dimensions");
    let positions = i32::try_from(shape.positions).expect("bounded positions fit MLX dimensions");
    let pairs =
        i32::try_from(layout.rope_pairs.get()).expect("bounded RoPE pairs fit MLX dimensions");
    let _device = crate::device_lock();
    let prefix = key_dim - pairs * 2;
    let stream = StreamOrDevice::gpu();
    let latent = Array::from_slice(latent, &[rows, latent_dim])
        .view_dtype_device(Dtype::Bfloat16, &stream)?;
    let wk = Array::from_slice(weights.wk, &[key_dim, latent_dim])
        .view_dtype_device(Dtype::Bfloat16, &stream)?;
    let projected = latent
        .matmul_device(wk.transpose_device(&stream)?, &stream)?
        .as_dtype_device(Dtype::Bfloat16, &stream)?;

    // Match the scalar stage boundary: normalization consumes the narrowed
    // projection, performs FP32 arithmetic, then narrows once to BF16.
    let projected_f32 = projected.as_dtype_device(Dtype::Float32, &stream)?;
    let width = Array::from_slice(
        &[f32::from(u16::try_from(layout.key_dimension.get()).expect(
            "validated RMSNorm width fits source FP32 conversion",
        ))],
        &[],
    );
    let epsilon = Array::from_slice(&[layout.norm_epsilon], &[]);
    let inverse_rms = projected_f32
        .square_device(&stream)?
        .sum_axis_device(1, true, &stream)?
        .divide_device(&width, &stream)?
        .add_device(&epsilon, &stream)?
        .rsqrt_device(&stream)?;
    let norm = Array::from_slice(weights.norm, &[key_dim])
        .view_dtype_device(Dtype::Bfloat16, &stream)?
        .as_dtype_device(Dtype::Float32, &stream)?;
    let normalized = projected_f32
        .multiply_device(&inverse_rms, &stream)?
        .multiply_device(&norm, &stream)?
        .as_dtype_device(Dtype::Bfloat16, &stream)?;

    let normalized_f32 = normalized.as_dtype_device(Dtype::Float32, &stream)?;
    let prefix_values = normalized_f32.try_index_device((.., 0..prefix), &stream)?;
    let tail = normalized_f32
        .try_index_device((.., prefix..key_dim), &stream)?
        .reshape_device(&[batches, positions, 1, pairs, 2], &stream)?;
    let real = tail.try_index_device((.., .., .., .., 0_i32), &stream)?;
    let imaginary = tail.try_index_device((.., .., .., .., 1_i32), &stream)?;
    let frequency_real = frequencies
        .iter()
        .map(|frequency| frequency.real())
        .collect::<Vec<_>>();
    let frequency_imaginary = frequencies
        .iter()
        .map(|frequency| frequency.imaginary())
        .collect::<Vec<_>>();
    let frequency_real = Array::from_slice(&frequency_real, &[1, positions, 1, pairs]);
    let frequency_imaginary = Array::from_slice(&frequency_imaginary, &[1, positions, 1, pairs]);
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
    let rotated_tail =
        mlx_rs::ops::stack_axis_device(&[&output_real, &output_imaginary], -1, &stream)?
            .reshape_device(&[rows, pairs * 2], &stream)?;
    let post_rope =
        mlx_rs::ops::concatenate_axis_device(&[&prefix_values, &rotated_tail], 1, &stream)?
            .as_dtype_device(Dtype::Bfloat16, &stream)?;

    let projected_words = projected.view_dtype_device(Dtype::Uint16, &stream)?;
    let normalized_words = normalized.view_dtype_device(Dtype::Uint16, &stream)?;
    let post_rope_words = post_rope.view_dtype_device(Dtype::Uint16, &stream)?;
    transforms::eval([&projected_words, &normalized_words, &post_rope_words])?;
    let projected = read_metal_key_stage(&projected_words, "projected", shape.key_elements)?;
    let normalized = read_metal_key_stage(&normalized_words, "normalized", shape.key_elements)?;
    let post_rope = read_metal_key_stage(&post_rope_words, "post_rope", shape.key_elements)?;
    Ok(MetalKeyStages {
        projected,
        normalized,
        post_rope,
    })
}

#[cfg(feature = "metal")]
fn read_metal_key_stage(
    words: &Array,
    stage: &'static str,
    expected: usize,
) -> Result<Vec<u16>, MetalKeyPreparationError> {
    let values = words.as_slice::<u16>();
    if values.len() != expected {
        return Err(MetalKeyPreparationError::Readback {
            stage,
            actual: values.len(),
            expected,
        });
    }
    if let Some(element) = values
        .iter()
        .position(|&bits| !bf16_to_f32(bits).is_finite())
    {
        return Err(MetalKeyPreparationError::NonFinite { stage, element });
    }
    Ok(values.to_vec())
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroUsize;

    use super::{
        IndexKeyError, IndexKeyLayout, IndexKeyLayoutError, IndexKeyWeights, prepare_index_keys,
    };
    #[cfg(feature = "metal")]
    use super::{
        IndexKeyPreparationExecution, IndexKeyRotaryExecution, MetalKeyPreparationError,
        prepare_index_keys_with_execution, prepare_index_keys_with_rotary_execution,
    };
    #[cfg(feature = "metal")]
    use crate::GPU_TEST_LOCK;
    use crate::RotaryFrequency;
    #[cfg(feature = "metal")]
    use serde::Deserialize;

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

    #[cfg(feature = "metal")]
    #[test]
    fn metal_rotary_preserves_simple_staged_key_boundaries() {
        let _gpu = GPU_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let mut wk = vec![0_u16; 32 * 16];
        wk[30 * 16] = bf16(2.0);
        wk[31 * 16 + 1] = bf16(3.0);
        let mut latent = vec![0_u16; 16];
        latent[0] = bf16(1.0);
        latent[1] = bf16(1.0);
        let frequencies = [RotaryFrequency::new(0.0, 1.0).expect("finite frequency")];
        let norm = [bf16(1.0); 32];
        let weights = IndexKeyWeights::new(&wk, &norm);
        let scalar = prepare_index_keys(&latent, &frequencies, weights, layout())
            .expect("finite scalar staged key");
        let metal = prepare_index_keys_with_rotary_execution(
            &latent,
            &frequencies,
            weights,
            layout(),
            IndexKeyRotaryExecution::MetalFp32,
        )
        .expect("finite Metal rotated staged key");
        assert_eq!(metal, scalar);
        let overflowing = [RotaryFrequency::new(f32::MAX, f32::MAX).unwrap()];
        assert!(matches!(
            prepare_index_keys_with_rotary_execution(
                &latent,
                &overflowing,
                weights,
                layout(),
                IndexKeyRotaryExecution::MetalFp32,
            ),
            Err(IndexKeyError::NonFiniteRotary { .. })
        ));
        assert_eq!(
            prepare_index_keys_with_rotary_execution(
                &latent,
                &frequencies,
                weights,
                layout(),
                IndexKeyRotaryExecution::MetalFp32,
            )
            .unwrap(),
            scalar,
            "finite call recovers after Metal rotary overflow",
        );
    }

    #[cfg(feature = "metal")]
    #[test]
    fn metal_key_candidate_preserves_simple_staged_key_boundaries() {
        let _gpu = GPU_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let mut wk = vec![0_u16; 32 * 16];
        wk[30 * 16] = bf16(2.0);
        wk[31 * 16 + 1] = bf16(3.0);
        let mut latent = vec![0_u16; 16];
        latent[0] = bf16(1.0);
        latent[1] = bf16(1.0);
        let frequencies = [RotaryFrequency::new(0.0, 1.0).expect("finite frequency")];
        let norm = [bf16(1.0); 32];
        let weights = IndexKeyWeights::new(&wk, &norm);
        let scalar = prepare_index_keys(&latent, &frequencies, weights, layout())
            .expect("finite scalar staged key");
        let candidate = prepare_index_keys_with_execution(
            &latent,
            &frequencies,
            weights,
            layout(),
            IndexKeyPreparationExecution::MetalPreFp4,
        )
        .expect("finite connected Metal key preparation");
        assert_eq!(
            candidate, scalar,
            "all BF16 and scalar FP4 stage boundaries"
        );
    }

    #[cfg(feature = "metal")]
    #[test]
    fn metal_pre_fp4_preserves_preflight_and_reports_staged_overflow() {
        let _gpu = GPU_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        assert!(matches!(
            prepare_index_keys_with_execution(
                &[0; 15],
                &[],
                IndexKeyWeights::new(&[], &[]),
                layout(),
                IndexKeyPreparationExecution::MetalPreFp4,
            ),
            Err(IndexKeyError::LatentLength {
                actual: 15,
                stride: 16,
            })
        ));
        let mut wk = vec![0_u16; 32 * 16];
        wk[0] = 0x7f7f; // Largest finite BF16; multiplying by two overflows the staged result.
        let mut latent = vec![0_u16; 16];
        latent[0] = bf16(2.0);
        assert!(matches!(
            prepare_index_keys_with_execution(
                &latent,
                &[RotaryFrequency::new(1.0, 0.0).expect("finite frequency")],
                IndexKeyWeights::new(&wk, &[bf16(1.0); 32]),
                layout(),
                IndexKeyPreparationExecution::MetalPreFp4,
            ),
            Err(IndexKeyError::MetalPreparation(
                MetalKeyPreparationError::NonFinite {
                    stage: "projected",
                    element: 0,
                }
            ))
        ));
    }

    #[cfg(feature = "metal")]
    #[derive(Deserialize)]
    struct SourceKeyFixture {
        model: SourceKeyModel,
        weights: SourceKeyWeights,
        frequencies: SourceKeyTensor,
        cases: Vec<SourceKeyCase>,
    }

    #[cfg(feature = "metal")]
    #[derive(Deserialize)]
    struct SourceKeyModel {
        batches: usize,
        latent_dimension: usize,
        key_dimension: usize,
        rope_pairs: usize,
        norm_epsilon: f32,
    }

    #[cfg(feature = "metal")]
    #[derive(Deserialize)]
    struct SourceKeyWeights {
        wk: SourceKeyTensor,
        norm: SourceKeyTensor,
    }

    #[cfg(feature = "metal")]
    #[derive(Deserialize)]
    struct SourceKeyCase {
        start_pos: usize,
        latent: SourceKeyTensor,
        index_cache_after: SourceKeyTensor,
    }

    #[cfg(feature = "metal")]
    #[derive(Deserialize)]
    struct SourceKeyTensor {
        dtype: String,
        shape: Vec<usize>,
        storage_hex: String,
    }

    #[cfg(feature = "metal")]
    impl SourceKeyTensor {
        fn bf16(&self) -> Vec<u16> {
            assert_eq!(self.dtype, "torch.bfloat16");
            assert_eq!(self.storage_hex.len() % 4, 0, "BF16 hex words");
            self.storage_hex
                .as_bytes()
                .chunks_exact(4)
                .map(|word| {
                    let low =
                        u8::from_str_radix(std::str::from_utf8(&word[..2]).expect("hex UTF-8"), 16)
                            .expect("hex byte");
                    let high =
                        u8::from_str_radix(std::str::from_utf8(&word[2..]).expect("hex UTF-8"), 16)
                            .expect("hex byte");
                    u16::from_le_bytes([low, high])
                })
                .collect()
        }

        fn frequencies(&self) -> Vec<RotaryFrequency> {
            assert_eq!(self.dtype, "torch.complex64");
            assert_eq!(self.storage_hex.len() % 16, 0, "complex FP32 hex words");
            self.storage_hex
                .as_bytes()
                .chunks_exact(16)
                .map(|word| {
                    let bytes = |start| {
                        std::array::from_fn(|offset| {
                            u8::from_str_radix(
                                std::str::from_utf8(
                                    &word[start + offset * 2..start + offset * 2 + 2],
                                )
                                .expect("hex UTF-8"),
                                16,
                            )
                            .expect("hex byte")
                        })
                    };
                    RotaryFrequency::new(f32::from_le_bytes(bytes(0)), f32::from_le_bytes(bytes(8)))
                        .expect("finite source frequency")
                })
                .collect()
        }
    }

    #[cfg(feature = "metal")]
    #[test]
    fn metal_key_candidate_matches_source_key_cache_and_scalar_stages() {
        let _gpu = GPU_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let fixture: SourceKeyFixture = serde_json::from_str(include_str!(
            "../../../../../fixtures/deepseek-v41/forward-index-key-reference.json"
        ))
        .expect("source key fixture");
        assert_eq!(
            fixture
                .cases
                .iter()
                .map(|case| case.start_pos)
                .collect::<Vec<_>>(),
            [0, 5, 6]
        );
        assert_eq!(fixture.weights.wk.shape, [64, 64]);
        assert_eq!(fixture.weights.norm.shape, [64]);
        assert_eq!(fixture.frequencies.shape, [8, 16]);
        let layout = IndexKeyLayout::new(
            nz(fixture.model.batches),
            nz(fixture.model.latent_dimension),
            nz(fixture.model.key_dimension),
            nz(fixture.model.rope_pairs),
            fixture.model.norm_epsilon,
        )
        .expect("source key layout");
        let wk = fixture.weights.wk.bf16();
        let norm = fixture.weights.norm.bf16();
        let frequencies = fixture.frequencies.frequencies();
        let weights = IndexKeyWeights::new(&wk, &norm);
        let mut wrong_frequency_detected = false;
        for case in fixture.cases {
            let latent = case.latent.bf16();
            let positions = latent.len() / fixture.model.latent_dimension;
            let source_frequencies = &frequencies[case.start_pos * fixture.model.rope_pairs
                ..(case.start_pos + positions) * fixture.model.rope_pairs];
            let scalar = prepare_index_keys(&latent, source_frequencies, weights, layout)
                .expect("source scalar key preparation");
            let candidate = prepare_index_keys_with_execution(
                &latent,
                source_frequencies,
                weights,
                layout,
                IndexKeyPreparationExecution::MetalPreFp4,
            )
            .expect("source connected Metal key preparation");
            assert_eq!(
                candidate, scalar,
                "all source BF16/FP4 stages at {}",
                case.start_pos
            );
            let cache = case.index_cache_after.bf16();
            let start = case.start_pos * fixture.model.key_dimension;
            let end = (case.start_pos + positions) * fixture.model.key_dimension;
            assert_eq!(
                candidate.post_fp4,
                cache[start..end],
                "source cache append at {}",
                case.start_pos
            );
            if case.start_pos != 0 {
                let wrong = prepare_index_keys_with_execution(
                    &latent,
                    &frequencies[..fixture.model.rope_pairs],
                    weights,
                    layout,
                    IndexKeyPreparationExecution::MetalPreFp4,
                )
                .expect("wrong but finite Metal preparation frequency");
                wrong_frequency_detected |= wrong.post_fp4 != candidate.post_fp4;
            }
        }
        assert!(
            wrong_frequency_detected,
            "source fixture detects replayed position-zero RoPE"
        );
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
