//! Bounded Hyper-Connections collapse and RMS normalization for attention input.
//!
//! This component accepts one copy-major runtime residual and its incoming
//! coefficients. It does not load checkpoints, fixtures, or expected outputs.

use thiserror::Error;

use crate::{
    hc::mixing::{HcMixError, MAX_HC_COPIES, MAX_HC_MIX_WIDTH, hc_pre_bf16_reference},
    norm::{MAX_RMS_NORM_WIDTH, RmsNormError, rms_norm_bf16_reference},
    precision::bf16_to_f32,
};

/// Borrowed immutable normalization operands for one attention-input boundary.
#[derive(Clone, Copy, Debug)]
pub struct AttentionInput<'a> {
    norm_weight: &'a [u16],
    copies: usize,
    epsilon: f32,
    width: usize,
}

impl<'a> AttentionInput<'a> {
    /// Validates the normalization weight, HC copy count, and RMS epsilon.
    pub fn new(
        norm_weight: &'a [u16],
        copies: usize,
        epsilon: f32,
    ) -> Result<Self, AttentionInputError> {
        let width = norm_weight.len();
        if width == 0 {
            return Err(AttentionInputError::EmptyWidth);
        }
        if width > MAX_RMS_NORM_WIDTH || width > MAX_HC_MIX_WIDTH {
            return Err(AttentionInputError::WidthTooLarge {
                width,
                maximum: MAX_RMS_NORM_WIDTH.min(MAX_HC_MIX_WIDTH),
            });
        }
        if copies == 0 {
            return Err(AttentionInputError::EmptyCopies);
        }
        if copies > MAX_HC_COPIES {
            return Err(AttentionInputError::CopyCountTooLarge {
                copies,
                maximum: MAX_HC_COPIES,
            });
        }
        if !epsilon.is_finite() || epsilon <= 0.0 {
            return Err(AttentionInputError::InvalidEpsilon);
        }
        for (element, &bits) in norm_weight.iter().enumerate() {
            if !bf16_to_f32(bits).is_finite() {
                return Err(AttentionInputError::NonFiniteNormWeight { element });
            }
        }
        Ok(Self {
            norm_weight,
            copies,
            epsilon,
            width,
        })
    }

    /// Collapses and normalizes a single copy-major token residual.
    ///
    /// `residual` is `[copies, hidden_width]` BF16 storage and `pre` is one
    /// finite FP32 incoming coefficient per residual copy. Returned rows are
    /// owned, so failure cannot alter caller-owned inputs.
    pub fn forward(
        &self,
        residual: &[u16],
        pre: &[f32],
    ) -> Result<AttentionInputOutput, AttentionInputError> {
        let expected_residual = checked_product(self.copies, self.width, "residual")?;
        if residual.len() != expected_residual {
            return Err(AttentionInputError::Length {
                field: "residual",
                actual: residual.len(),
                expected: expected_residual,
            });
        }
        if pre.len() != self.copies {
            return Err(AttentionInputError::Length {
                field: "pre",
                actual: pre.len(),
                expected: self.copies,
            });
        }

        let mut collapsed_bf16 = allocate_u16("collapsed_bf16", self.width)?;
        hc_pre_bf16_reference(residual, pre, self.width, &mut collapsed_bf16)?;

        let mut normalized_bf16 = allocate_u16("normalized_bf16", self.width)?;
        rms_norm_bf16_reference(
            &collapsed_bf16,
            self.norm_weight,
            self.epsilon,
            &mut normalized_bf16,
        )?;

        Ok(AttentionInputOutput {
            collapsed_bf16,
            normalized_bf16,
        })
    }
}

/// Owned intermediate rows at the HC-to-attention boundary.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AttentionInputOutput {
    collapsed_bf16: Vec<u16>,
    normalized_bf16: Vec<u16>,
}

impl AttentionInputOutput {
    /// Returns the BF16 Hyper-Connections collapse before RMS normalization.
    #[must_use]
    pub fn collapsed_bf16(&self) -> &[u16] {
        &self.collapsed_bf16
    }

    /// Returns the BF16 normalized attention input.
    #[must_use]
    pub fn normalized_bf16(&self) -> &[u16] {
        &self.normalized_bf16
    }
}

/// An invalid attention-input construction or execution request.
#[derive(Clone, Debug, Error, Eq, PartialEq)]
#[non_exhaustive]
pub enum AttentionInputError {
    /// The learned normalization row determines the hidden width and is empty.
    #[error("attention input requires a nonempty hidden width")]
    EmptyWidth,
    /// The hidden width exceeds a scalar component bound.
    #[error("attention input width {width} exceeds maximum {maximum}")]
    WidthTooLarge { width: usize, maximum: usize },
    /// At least one Hyper-Connections residual copy is required.
    #[error("attention input requires at least one residual copy")]
    EmptyCopies,
    /// The residual copy count exceeds the scalar HC pre-mix bound.
    #[error("attention input copies {copies} exceeds maximum {maximum}")]
    CopyCountTooLarge { copies: usize, maximum: usize },
    /// The `RMSNorm` epsilon must be finite and positive.
    #[error("attention input epsilon must be finite and positive")]
    InvalidEpsilon,
    /// A derived runtime shape overflowed `usize`.
    #[error("attention input {field} shape overflowed")]
    ShapeOverflow { field: &'static str },
    /// A caller buffer has an unexpected exact length.
    #[error("attention input {field} length is {actual}, expected {expected}")]
    Length {
        field: &'static str,
        actual: usize,
        expected: usize,
    },
    /// A learned BF16 normalization weight is NaN or infinity.
    #[error("attention input normalization weight at element {element} is non-finite")]
    NonFiniteNormWeight { element: usize },
    /// A temporary output row could not be reserved.
    #[error("could not allocate {elements} attention-input elements for {field}")]
    AllocationFailed {
        field: &'static str,
        elements: usize,
    },
    /// Hyper-Connections pre-mix rejected a runtime input.
    #[error(transparent)]
    HcMix(#[from] HcMixError),
    /// `RMSNorm` rejected a runtime input.
    #[error(transparent)]
    RmsNorm(#[from] RmsNormError),
}

fn checked_product(
    left: usize,
    right: usize,
    field: &'static str,
) -> Result<usize, AttentionInputError> {
    left.checked_mul(right)
        .ok_or(AttentionInputError::ShapeOverflow { field })
}

fn allocate_u16(field: &'static str, elements: usize) -> Result<Vec<u16>, AttentionInputError> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(elements)
        .map_err(|_| AttentionInputError::AllocationFailed { field, elements })?;
    values.resize(elements, 0);
    Ok(values)
}
