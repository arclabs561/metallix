//! Bounded scalar reference for V4.1's preprojected KV compressor.
//!
//! Learned `wkv` and `wgate` projections intentionally remain outside this
//! module.  It accepts their outputs, pools a token group with a per-feature
//! token-axis softmax, narrows the pool to BF16, and applies the existing
//! scalar [`crate::rms_norm_bf16_reference`].  This is not a checkpoint,
//! cache, scheduler, or hardware-parity implementation.

use crate::norm::{MAX_RMS_NORM_WIDTH, RmsNormError, rms_norm_bf16_reference};
use crate::precision::{bf16_to_f32, f32_to_bf16_rne};
use thiserror::Error;

/// Maximum elements in one compressor input, state, or result buffer.
pub const MAX_COMPRESSOR_ELEMENTS: usize = 1 << 20;

/// Preprojected inputs accepted by [`CompressorState::forward`].
///
/// All slices use row-major `[batch, position, width]` layout.  `ProjectedBf16`
/// is the ratio-one projected KV result.  `Gated` supplies the FP32 `wkv` and
/// `wgate` results used by a ratio greater than one.
#[derive(Clone, Copy, Debug)]
#[non_exhaustive]
pub enum CompressorInput<'a> {
    /// BF16 projected KV values for a ratio-one compressor.
    ProjectedBf16(&'a [u16]),
    /// FP32 projected KV values and gate scores for a pooling compressor.
    Gated {
        /// Preprojected KV values in `[batch, position, width]` order.
        kv: &'a [f32],
        /// Preprojected gate scores in `[batch, position, width]` order.
        scores: &'a [f32],
    },
}

/// Invalid compressor construction, stream geometry, or scalar result.
#[derive(Clone, Debug, Error, Eq, PartialEq)]
#[non_exhaustive]
pub enum CompressorError {
    /// Batch count, hidden width, and compression ratio must each be nonzero.
    #[error("compressor {field} must be nonzero")]
    EmptyDimension {
        /// The zero-valued dimension.
        field: &'static str,
    },
    /// A derived shape exceeded `usize`.
    #[error("compressor shape arithmetic overflowed for {field}")]
    ShapeOverflow {
        /// Derived buffer or position calculation.
        field: &'static str,
    },
    /// A state, input, or output buffer exceeds the scalar reference bound.
    #[error("compressor {field} has {elements} elements, maximum is {maximum}")]
    ElementLimit {
        /// Buffer role.
        field: &'static str,
        /// Requested element count.
        elements: usize,
        /// Maximum allowed element count.
        maximum: usize,
    },
    /// A bounded result or state buffer could not be allocated.
    #[error("could not allocate {elements} compressor {field} elements")]
    AllocationFailed {
        /// Buffer role.
        field: &'static str,
        /// Requested element count.
        elements: usize,
    },
    /// A caller-owned buffer has an unexpected exact length.
    #[error("compressor {field} length is {actual}, expected {expected}")]
    Length {
        /// Buffer role.
        field: &'static str,
        /// Actual element count.
        actual: usize,
        /// Required element count.
        expected: usize,
    },
    /// Ratio one requires projected BF16 input and larger ratios require gated FP32 input.
    #[error("compressor ratio {ratio} does not accept {input}")]
    InputMode {
        /// Configured compression ratio.
        ratio: usize,
        /// Supplied input variant.
        input: &'static str,
    },
    /// The RMS epsilon must be finite and positive.
    #[error("compressor epsilon must be finite and positive")]
    InvalidEpsilon,
    /// A BF16 input or learned normalization weight was NaN or infinity.
    #[error("compressor non-finite BF16 {field} at element {element}")]
    NonFiniteBf16 {
        /// Input or learned-weight role.
        field: &'static str,
        /// Flat element index.
        element: usize,
    },
    /// A gated FP32 input was NaN or infinity.
    #[error("compressor non-finite FP32 {field} at element {element}")]
    NonFiniteF32 {
        /// KV or score role.
        field: &'static str,
        /// Flat element index.
        element: usize,
    },
    /// The first successful stream call must explicitly begin at position zero.
    #[error("compressor first call must start at zero, got {actual}")]
    FirstPosition {
        /// Supplied starting position.
        actual: usize,
    },
    /// A ratio-greater-than-one continuation must contain exactly one token.
    #[error("compressor ratio {ratio} continuation has {positions} positions, expected one")]
    ChunkedContinuation {
        /// Configured compression ratio.
        ratio: usize,
        /// Supplied token count.
        positions: usize,
    },
    /// A continuation did not begin at the next sequential position.
    #[error("compressor continuation starts at {actual}, expected {expected}")]
    NonSequentialPosition {
        /// Expected next position.
        expected: usize,
        /// Supplied starting position.
        actual: usize,
    },
    /// A scalar pooling intermediate or BF16 narrowing became non-finite.
    #[error(
        "compressor scalar result overflowed at {stage}, batch {batch}, group {group}, feature {feature}"
    )]
    ValueOverflow {
        /// Named scalar stage.
        stage: &'static str,
        /// Batch index.
        batch: usize,
        /// Complete-group index.
        group: usize,
        /// Hidden feature index.
        feature: usize,
    },
    /// Existing `RMSNorm` validation or scalar evaluation failed.
    #[error("compressor RMSNorm failed: {0}")]
    RmsNorm(#[from] RmsNormError),
}

/// Streaming state for a preprojected V4.1 compressor.
///
/// `start == 0` is an explicit reset.  Otherwise calls must be sequential;
/// ratio-greater-than-one decoding accepts exactly one position at a time.
/// Every failing call, including a reset attempt, leaves this state unchanged.
#[derive(Clone, Debug)]
pub struct CompressorState {
    batches: usize,
    width: usize,
    ratio: usize,
    norm_weight: Vec<u16>,
    epsilon: f32,
    next_position: usize,
    kv_state: Vec<f32>,
    score_state: Vec<f32>,
}

impl CompressorState {
    /// Creates bounded empty stream state and validates its fixed normalization parameters.
    ///
    /// # Errors
    ///
    /// Returns [`CompressorError`] for invalid dimensions, normalization
    /// parameters, or bounded allocation failure.
    pub fn new(
        batches: usize,
        width: usize,
        ratio: usize,
        norm_weight: &[u16],
        epsilon: f32,
    ) -> Result<Self, CompressorError> {
        for (field, value) in [("batches", batches), ("width", width), ("ratio", ratio)] {
            if value == 0 {
                return Err(CompressorError::EmptyDimension { field });
            }
        }
        if width > MAX_RMS_NORM_WIDTH {
            return Err(CompressorError::ElementLimit {
                field: "width",
                elements: width,
                maximum: MAX_RMS_NORM_WIDTH,
            });
        }
        // Ratio one has no pooled state, but one `[batch, position, width]`
        // input/output buffer must still fit the same bound.
        checked_elements(batches, 1, width, "input")?;
        let state_elements = if ratio > 1 {
            checked_elements(batches, ratio, width, "state")?
        } else {
            0
        };
        if norm_weight.len() != width {
            return Err(CompressorError::Length {
                field: "norm_weight",
                actual: norm_weight.len(),
                expected: width,
            });
        }
        if !epsilon.is_finite() || epsilon <= 0.0 {
            return Err(CompressorError::InvalidEpsilon);
        }
        validate_bf16(norm_weight, "norm_weight")?;

        let mut weights = reserve_u16(width, "norm_weight")?;
        weights.extend_from_slice(norm_weight);
        let mut kv_state = reserve_f32(state_elements, "kv_state")?;
        kv_state.resize(state_elements, 0.0);
        let mut score_state = reserve_f32(state_elements, "score_state")?;
        score_state.resize(state_elements, f32::NEG_INFINITY);
        Ok(Self {
            batches,
            width,
            ratio,
            norm_weight: weights,
            epsilon,
            next_position: 0,
            kv_state,
            score_state,
        })
    }

    /// Pools one prefill or sequential continuation and returns completed normalized groups.
    ///
    /// The returned BF16 buffer is row-major `[batch, complete_groups, width]`.
    /// A partial group returns `None`.  Pooling follows the source's FP32
    /// per-feature multiply, divide, and sum sequence, then narrows once to
    /// BF16 before `RMSNorm`.
    ///
    /// ```
    /// use deepseek::compressor::{CompressorInput, CompressorState};
    ///
    /// let mut compressor = CompressorState::new(1, 1, 2, &[0x3f80], 1.0e-6)?;
    /// assert_eq!(
    ///     compressor.forward(CompressorInput::Gated { kv: &[1.0], scores: &[0.0] }, 1, 0)?,
    ///     None,
    /// );
    /// assert_eq!(
    ///     compressor.forward(CompressorInput::Gated { kv: &[1.0], scores: &[0.0] }, 1, 1)?,
    ///     Some(vec![0x3f80]),
    /// );
    /// # Ok::<(), deepseek::compressor::CompressorError>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`CompressorError`] for an invalid call.  No error changes the
    /// pending partial group or [`Self::next_position`].
    pub fn forward(
        &mut self,
        input: CompressorInput<'_>,
        positions: usize,
        start: usize,
    ) -> Result<Option<Vec<u16>>, CompressorError> {
        match input {
            CompressorInput::ProjectedBf16(values) if self.ratio == 1 => {
                self.forward_projected(values, positions, start)
            }
            CompressorInput::Gated { kv, scores } if self.ratio > 1 => {
                self.forward_gated(kv, scores, positions, start)
            }
            CompressorInput::ProjectedBf16(_) => Err(CompressorError::InputMode {
                ratio: self.ratio,
                input: "ProjectedBf16",
            }),
            CompressorInput::Gated { .. } => Err(CompressorError::InputMode {
                ratio: self.ratio,
                input: "Gated",
            }),
        }
    }

    fn forward_projected(
        &mut self,
        values: &[u16],
        positions: usize,
        start: usize,
    ) -> Result<Option<Vec<u16>>, CompressorError> {
        let elements = self.validate_call(positions, start)?;
        validate_length(values.len(), elements, "projected")?;
        validate_bf16(values, "projected")?;
        let output = normalize(values, self.width, &self.norm_weight, self.epsilon)?;
        self.next_position = next_position(start, positions)?;
        Ok(Some(output))
    }

    fn forward_gated(
        &mut self,
        kv: &[f32],
        scores: &[f32],
        positions: usize,
        start: usize,
    ) -> Result<Option<Vec<u16>>, CompressorError> {
        let elements = self.validate_call(positions, start)?;
        validate_length(kv.len(), elements, "kv")?;
        validate_length(scores.len(), elements, "scores")?;
        validate_f32(kv, "kv")?;
        validate_f32(scores, "scores")?;
        let next_position = next_position(start, positions)?;
        let mut next_kv = reserve_f32(self.kv_state.len(), "staged KV state")?;
        let mut next_scores = reserve_f32(self.score_state.len(), "staged score state")?;
        if start == 0 {
            next_kv.resize(self.kv_state.len(), 0.0);
            next_scores.resize(self.score_state.len(), f32::NEG_INFINITY);
        } else {
            next_kv.extend_from_slice(&self.kv_state);
            next_scores.extend_from_slice(&self.score_state);
        }
        let complete = positions / self.ratio;
        let pooled = if start == 0 {
            store_remainder(
                &mut next_kv,
                &mut next_scores,
                kv,
                scores,
                self.batches,
                positions,
                self.width,
                complete * self.ratio,
                positions % self.ratio,
            );
            pool_prefill(
                kv,
                scores,
                self.batches,
                positions,
                complete,
                self.ratio,
                self.width,
            )?
        } else {
            store_slot(
                &mut next_kv,
                &mut next_scores,
                kv,
                scores,
                self.batches,
                self.width,
                self.ratio,
                start % self.ratio,
            );
            if next_position % self.ratio == 0 {
                pool_state(&next_kv, &next_scores, self.batches, self.ratio, self.width)?
            } else {
                Vec::new()
            }
        };
        let output = (!pooled.is_empty())
            .then(|| normalize(&pooled, self.width, &self.norm_weight, self.epsilon))
            .transpose()?;
        self.kv_state = next_kv;
        self.score_state = next_scores;
        self.next_position = next_position;
        Ok(output)
    }

    fn validate_call(&self, positions: usize, start: usize) -> Result<usize, CompressorError> {
        if positions == 0 {
            return Err(CompressorError::EmptyDimension { field: "positions" });
        }
        let elements = checked_elements(self.batches, positions, self.width, "input")?;
        self.validate_stream(start, positions)?;
        Ok(elements)
    }

    /// Returns the position required by the next non-reset call.
    #[must_use]
    pub const fn next_position(&self) -> usize {
        self.next_position
    }

    fn validate_stream(&self, start: usize, positions: usize) -> Result<(), CompressorError> {
        if start == 0 {
            return Ok(());
        }
        if self.next_position == 0 {
            return Err(CompressorError::FirstPosition { actual: start });
        }
        if self.ratio > 1 && positions != 1 {
            return Err(CompressorError::ChunkedContinuation {
                ratio: self.ratio,
                positions,
            });
        }
        if start != self.next_position {
            return Err(CompressorError::NonSequentialPosition {
                expected: self.next_position,
                actual: start,
            });
        }
        Ok(())
    }
}

fn checked_elements(
    first: usize,
    second: usize,
    third: usize,
    field: &'static str,
) -> Result<usize, CompressorError> {
    let elements = first
        .checked_mul(second)
        .and_then(|value| value.checked_mul(third))
        .ok_or(CompressorError::ShapeOverflow { field })?;
    if elements > MAX_COMPRESSOR_ELEMENTS {
        return Err(CompressorError::ElementLimit {
            field,
            elements,
            maximum: MAX_COMPRESSOR_ELEMENTS,
        });
    }
    Ok(elements)
}

fn next_position(start: usize, positions: usize) -> Result<usize, CompressorError> {
    start
        .checked_add(positions)
        .ok_or(CompressorError::ShapeOverflow {
            field: "next_position",
        })
}

fn reserve_u16(elements: usize, field: &'static str) -> Result<Vec<u16>, CompressorError> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(elements)
        .map_err(|_| CompressorError::AllocationFailed { field, elements })?;
    Ok(values)
}

fn reserve_f32(elements: usize, field: &'static str) -> Result<Vec<f32>, CompressorError> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(elements)
        .map_err(|_| CompressorError::AllocationFailed { field, elements })?;
    Ok(values)
}

fn validate_length(
    actual: usize,
    expected: usize,
    field: &'static str,
) -> Result<(), CompressorError> {
    if actual == expected {
        Ok(())
    } else {
        Err(CompressorError::Length {
            field,
            actual,
            expected,
        })
    }
}

fn validate_bf16(values: &[u16], field: &'static str) -> Result<(), CompressorError> {
    for (element, &bits) in values.iter().enumerate() {
        if !bf16_to_f32(bits).is_finite() {
            return Err(CompressorError::NonFiniteBf16 { field, element });
        }
    }
    Ok(())
}

fn validate_f32(values: &[f32], field: &'static str) -> Result<(), CompressorError> {
    for (element, &value) in values.iter().enumerate() {
        if !value.is_finite() {
            return Err(CompressorError::NonFiniteF32 { field, element });
        }
    }
    Ok(())
}

// These arguments directly name the tensor slices and fixed `[B, S, D]`
// geometry; wrapping them would hide rather than reduce the scalar indexing.
#[allow(clippy::too_many_arguments)]
fn store_remainder(
    state_kv: &mut [f32],
    state_scores: &mut [f32],
    kv: &[f32],
    scores: &[f32],
    batches: usize,
    positions: usize,
    width: usize,
    cutoff: usize,
    remainder: usize,
) {
    for batch in 0..batches {
        for slot in 0..remainder {
            let source = (batch * positions + cutoff + slot) * width;
            // State stride is ratio rather than remainder.
            let state_stride = state_kv.len() / batches;
            let target = batch * state_stride + slot * width;
            state_kv[target..target + width].copy_from_slice(&kv[source..source + width]);
            state_scores[target..target + width].copy_from_slice(&scores[source..source + width]);
        }
    }
}

// This mirrors `store_remainder` for one sequential token and retains explicit
// tensor geometry at the scalar indexing boundary.
#[allow(clippy::too_many_arguments)]
fn store_slot(
    state_kv: &mut [f32],
    state_scores: &mut [f32],
    kv: &[f32],
    scores: &[f32],
    batches: usize,
    width: usize,
    ratio: usize,
    slot: usize,
) {
    for batch in 0..batches {
        let source = batch * width;
        let target = (batch * ratio + slot) * width;
        state_kv[target..target + width].copy_from_slice(&kv[source..source + width]);
        state_scores[target..target + width].copy_from_slice(&scores[source..source + width]);
    }
}

fn pool_prefill(
    kv: &[f32],
    scores: &[f32],
    batches: usize,
    positions: usize,
    complete: usize,
    ratio: usize,
    width: usize,
) -> Result<Vec<u16>, CompressorError> {
    let elements = checked_elements(batches, complete, width, "pooled output")?;
    let mut output = reserve_u16(elements, "pooled output")?;
    for batch in 0..batches {
        for group in 0..complete {
            let group_start = (batch * positions + group * ratio) * width;
            pool_group(
                kv,
                scores,
                group_start,
                ratio,
                width,
                batch,
                group,
                &mut output,
            )?;
        }
    }
    Ok(output)
}

fn pool_state(
    kv: &[f32],
    scores: &[f32],
    batches: usize,
    ratio: usize,
    width: usize,
) -> Result<Vec<u16>, CompressorError> {
    let elements = checked_elements(batches, 1, width, "pooled output")?;
    let mut output = reserve_u16(elements, "pooled output")?;
    for batch in 0..batches {
        pool_group(
            kv,
            scores,
            batch * ratio * width,
            ratio,
            width,
            batch,
            0,
            &mut output,
        )?;
    }
    Ok(output)
}

// Pooling keeps source and destination tensors plus `[B, group, D]` indices
// explicit so its multiply/divide/sum order remains auditable.
#[allow(clippy::too_many_arguments)]
fn pool_group(
    kv: &[f32],
    scores: &[f32],
    start: usize,
    ratio: usize,
    width: usize,
    batch: usize,
    group: usize,
    output: &mut Vec<u16>,
) -> Result<(), CompressorError> {
    for feature in 0..width {
        let mut maximum = f32::NEG_INFINITY;
        for slot in 0..ratio {
            maximum = maximum.max(scores[start + slot * width + feature]);
        }
        let mut denominator = 0.0_f32;
        for slot in 0..ratio {
            let difference = scores[start + slot * width + feature] - maximum;
            if !difference.is_finite() {
                return Err(overflow("score_difference", batch, group, feature));
            }
            let exponent = difference.exp();
            if !exponent.is_finite() {
                return Err(overflow("exp", batch, group, feature));
            }
            denominator += exponent;
            if !denominator.is_finite() {
                return Err(overflow("denominator", batch, group, feature));
            }
        }
        if denominator == 0.0 {
            return Err(overflow("denominator", batch, group, feature));
        }
        let mut sum = 0.0_f32;
        for slot in 0..ratio {
            let exponent = (scores[start + slot * width + feature] - maximum).exp();
            let probability = exponent / denominator;
            if !probability.is_finite() {
                return Err(overflow("probability", batch, group, feature));
            }
            let product = kv[start + slot * width + feature] * probability;
            if !product.is_finite() {
                return Err(overflow("product", batch, group, feature));
            }
            sum += product;
            if !sum.is_finite() {
                return Err(overflow("sum", batch, group, feature));
            }
        }
        let bits = f32_to_bf16_rne(sum);
        if !bf16_to_f32(bits).is_finite() {
            return Err(overflow("bf16", batch, group, feature));
        }
        output.push(bits);
    }
    Ok(())
}

fn normalize(
    input: &[u16],
    width: usize,
    weight: &[u16],
    epsilon: f32,
) -> Result<Vec<u16>, CompressorError> {
    let mut output = reserve_u16(input.len(), "normalized output")?;
    output.resize(input.len(), 0);
    for (source, target) in input
        .chunks_exact(width)
        .zip(output.chunks_exact_mut(width))
    {
        rms_norm_bf16_reference(source, weight, epsilon, target)
            .map_err(CompressorError::RmsNorm)?;
    }
    Ok(output)
}

const fn overflow(
    stage: &'static str,
    batch: usize,
    group: usize,
    feature: usize,
) -> CompressorError {
    CompressorError::ValueOverflow {
        stage,
        batch,
        group,
        feature,
    }
}
