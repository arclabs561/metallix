//! Bounded source-order composition for one V4.1 FFN sublayer.
//!
//! The pinned block derives this sublayer's Hyper-Connection coefficients from
//! its residual, but collapses that residual with the *incoming* attention
//! pre-mix. It then runs `RMSNorm`, the text `MoE`, and the FFN-specific
//! Hyper-Connection post-mix. This is a diagnostic scalar reference, not a
//! checkpoint loader or serving implementation.

use thiserror::Error;

use crate::{
    hc::{
        HcCoefficients,
        mixing::{HcMixError, hc_post_bf16_reference, hc_pre_bf16_reference},
        projection::{HcProjectionError, project_hc_coefficients},
    },
    moe::{MoEDiagnostic, MoEError, MoEReference},
    norm::{RmsNormError, rms_norm_bf16_reference},
};

/// Borrowed configuration for the FFN half of one Hyper-Connection block.
#[derive(Clone, Copy, Debug)]
pub struct FfnSublayerReference<'a> {
    moe: MoEReference<'a>,
    norm_weight: &'a [u16],
    hc_projection: &'a [f32],
    hc_scale: &'a [f32; 3],
    hc_base: &'a [f32],
    copies: usize,
    norm_epsilon: f32,
    sinkhorn_iterations: usize,
    hc_epsilon: f32,
}

impl<'a> FfnSublayerReference<'a> {
    /// Validates static buffer geometry for one FFN sublayer.
    ///
    /// The `norm_weight` length must equal the `MoE` hidden width. The HC
    /// projection and base are checked against that width and `copies` without
    /// executing artificial data. Forward-time leaf validation still checks
    /// finite encoded values and the bounded HC and norm numeric domains.
    #[allow(
        clippy::too_many_arguments,
        reason = "the source stores each FFN and Hyper-Connection role separately"
    )]
    pub fn new(
        moe: MoEReference<'a>,
        norm_weight: &'a [u16],
        hc_projection: &'a [f32],
        hc_scale: &'a [f32; 3],
        hc_base: &'a [f32],
        copies: usize,
        norm_epsilon: f32,
        sinkhorn_iterations: usize,
        hc_epsilon: f32,
    ) -> Result<Self, FfnError> {
        let width = moe.hidden_width();
        if norm_weight.len() != width {
            return Err(FfnError::Length {
                field: "norm_weight",
                actual: norm_weight.len(),
                expected: width,
            });
        }
        if copies == 0 {
            return Err(FfnError::InvalidCopies);
        }
        if !norm_epsilon.is_finite() || norm_epsilon <= 0.0 {
            return Err(FfnError::InvalidNormEpsilon);
        }
        if sinkhorn_iterations == 0 {
            return Err(FfnError::InvalidSinkhornIterations);
        }
        if !hc_epsilon.is_finite() || hc_epsilon <= 0.0 {
            return Err(FfnError::InvalidHcEpsilon);
        }
        if hc_scale.iter().any(|value| !value.is_finite()) {
            return Err(FfnError::InvalidHcScale);
        }

        let residual_elements = checked_product("residual", copies, width)?;
        let mix_rows = copies
            .checked_add(2)
            .and_then(|value| value.checked_mul(copies))
            .ok_or(FfnError::ShapeOverflow { field: "mix rows" })?;
        let projection_elements = checked_product("projection", mix_rows, residual_elements)?;
        if hc_projection.len() != projection_elements {
            return Err(FfnError::Length {
                field: "hc_projection",
                actual: hc_projection.len(),
                expected: projection_elements,
            });
        }
        if hc_base.len() != mix_rows {
            return Err(FfnError::Length {
                field: "hc_base",
                actual: hc_base.len(),
                expected: mix_rows,
            });
        }

        Ok(Self {
            moe,
            norm_weight,
            hc_projection,
            hc_scale,
            hc_base,
            copies,
            norm_epsilon,
            sinkhorn_iterations,
            hc_epsilon,
        })
    }

    /// Executes one source-ordered FFN sublayer token.
    ///
    /// `residual` is copy-major `[copies, hidden_width]`. `incoming_pre` is
    /// the attention sublayer's prior HC pre-mix, not the newly derived FFN
    /// coefficient. A successful diagnostic owns all intermediate buffers; no
    /// caller-owned output is modified on failure.
    ///
    /// # Errors
    ///
    /// Returns [`FfnError`] if projection, mixing, normalization, or `MoE`
    /// execution rejects an input or an internal allocation fails.
    pub fn forward_token(
        &self,
        residual: &[u16],
        incoming_pre: &[f32],
    ) -> Result<FfnDiagnostic, FfnError> {
        // The source derives the FFN coefficients before it consumes the
        // previous attention pre-mix. Keep this first: it bounds the residual
        // geometry before any FFN temporary allocation.
        let coefficients = project_hc_coefficients(
            residual,
            self.hc_projection,
            self.hc_scale,
            self.hc_base,
            self.copies,
            self.norm_epsilon,
            self.sinkhorn_iterations,
            self.hc_epsilon,
        )?;
        let width = self.moe.hidden_width();
        let residual_elements = checked_product("residual", coefficients.copies(), width)?;
        let mut collapsed_bf16 = allocate_u16("collapsed", width)?;
        hc_pre_bf16_reference(residual, incoming_pre, width, &mut collapsed_bf16)?;
        let mut normalized_bf16 = allocate_u16("normalized", width)?;
        rms_norm_bf16_reference(
            &collapsed_bf16,
            self.norm_weight,
            self.norm_epsilon,
            &mut normalized_bf16,
        )?;
        let moe = self.moe.forward_token(&normalized_bf16)?;
        let mut output_bf16 = allocate_u16("output", residual_elements)?;
        hc_post_bf16_reference(
            moe.output_bf16(),
            residual,
            coefficients.post(),
            coefficients.comb(),
            &mut output_bf16,
        )?;
        Ok(FfnDiagnostic {
            coefficients,
            collapsed_bf16,
            normalized_bf16,
            moe,
            output_bf16,
        })
    }
}

/// Source-order observations from one FFN sublayer token.
#[derive(Clone, Debug, PartialEq)]
pub struct FfnDiagnostic {
    coefficients: HcCoefficients,
    collapsed_bf16: Vec<u16>,
    normalized_bf16: Vec<u16>,
    moe: MoEDiagnostic,
    output_bf16: Vec<u16>,
}

impl FfnDiagnostic {
    /// Returns freshly derived FFN Hyper-Connection coefficients.
    #[must_use]
    pub const fn coefficients(&self) -> &HcCoefficients {
        &self.coefficients
    }

    /// Returns the BF16 residual collapsed with the incoming attention pre-mix.
    #[must_use]
    pub fn collapsed_bf16(&self) -> &[u16] {
        &self.collapsed_bf16
    }

    /// Returns the BF16 row passed to the `MoE` after `RMSNorm`.
    #[must_use]
    pub fn normalized_bf16(&self) -> &[u16] {
        &self.normalized_bf16
    }

    /// Returns routed and shared expert observations.
    #[must_use]
    pub const fn moe(&self) -> &MoEDiagnostic {
        &self.moe
    }

    /// Returns the expanded copy-major BF16 FFN output.
    #[must_use]
    pub fn output_bf16(&self) -> &[u16] {
        &self.output_bf16
    }
}

/// A malformed bounded FFN composition request or failed leaf execution.
#[derive(Clone, Debug, Error, PartialEq)]
#[non_exhaustive]
pub enum FfnError {
    /// The learned `RMSNorm` weight has an unexpected hidden width.
    #[error("FFN {field} length is {actual}, expected {expected}")]
    Length {
        /// Named supplied buffer.
        field: &'static str,
        /// Actual element count.
        actual: usize,
        /// Required element count.
        expected: usize,
    },
    /// At least one Hyper-Connection copy is required.
    #[error("FFN Hyper-Connection copies must be nonzero")]
    InvalidCopies,
    /// The `RMSNorm` epsilon must be finite and positive.
    #[error("FFN RMSNorm epsilon must be finite and positive")]
    InvalidNormEpsilon,
    /// At least one Sinkhorn normalization iteration is required.
    #[error("FFN Sinkhorn iterations must be nonzero")]
    InvalidSinkhornIterations,
    /// The Hyper-Connection stabilization epsilon must be finite and positive.
    #[error("FFN Hyper-Connection epsilon must be finite and positive")]
    InvalidHcEpsilon,
    /// Hyper-Connection affine scales must all be finite.
    #[error("FFN Hyper-Connection scales must be finite")]
    InvalidHcScale,
    /// Checked composition geometry overflowed.
    #[error("FFN {field} shape arithmetic overflowed")]
    ShapeOverflow {
        /// Derived shape role.
        field: &'static str,
    },
    /// A fallible temporary allocation failed.
    #[error("could not allocate FFN {field}")]
    Allocation {
        /// Temporary buffer role.
        field: &'static str,
    },
    /// Hyper-Connection projection rejected the residual or controls.
    #[error(transparent)]
    HcProjection(#[from] HcProjectionError),
    /// Hyper-Connection mixing rejected a shape or scalar result.
    #[error(transparent)]
    HcMix(#[from] HcMixError),
    /// `RMSNorm` rejected the collapsed row or learned weight.
    #[error(transparent)]
    Norm(#[from] RmsNormError),
    /// `MoE` rejected the normalized row or encoded expert weights.
    #[error(transparent)]
    MoE(#[from] MoEError),
}

fn checked_product(field: &'static str, left: usize, right: usize) -> Result<usize, FfnError> {
    left.checked_mul(right)
        .ok_or(FfnError::ShapeOverflow { field })
}

fn allocate_u16(field: &'static str, length: usize) -> Result<Vec<u16>, FfnError> {
    let mut output = Vec::new();
    output
        .try_reserve_exact(length)
        .map_err(|_| FfnError::Allocation { field })?;
    output.resize(length, 0);
    Ok(output)
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use super::*;
    use crate::moe::{Fp4ExpertWeights, Fp8ExpertWeights, MoEConfig};

    const WIDTH: usize = 32;
    const COPIES: usize = 2;

    struct ExpertBuffers {
        fp4_codes: Vec<u8>,
        fp4_scales: Vec<u8>,
        fp8_codes: Vec<u8>,
        fp8_scales: Vec<u8>,
        gate: Vec<u16>,
        bias: Vec<f32>,
        norm_weight: Vec<u16>,
        hc_projection: Vec<f32>,
        hc_base: Vec<f32>,
    }

    impl ExpertBuffers {
        fn new() -> Self {
            let matrix = WIDTH * WIDTH;
            let mix_rows = (COPIES + 2) * COPIES;
            Self {
                fp4_codes: vec![0x11; matrix / 2],
                fp4_scales: vec![127; WIDTH],
                fp8_codes: vec![0x30; matrix],
                fp8_scales: vec![127; 1],
                gate: vec![0; WIDTH],
                bias: vec![0.0],
                norm_weight: vec![0x3f80; WIDTH],
                hc_projection: vec![0.0; mix_rows * COPIES * WIDTH],
                hc_base: vec![0.0; mix_rows],
            }
        }

        fn forward(
            &self,
            residual: &[u16],
            incoming_pre: &[f32],
        ) -> Result<FfnDiagnostic, FfnError> {
            let routed = Fp4ExpertWeights::new(
                WIDTH,
                WIDTH,
                &self.fp4_codes,
                &self.fp4_scales,
                &self.fp4_codes,
                &self.fp4_scales,
                &self.fp4_codes,
                &self.fp4_scales,
            )
            .expect("fixed routed expert geometry");
            let shared = Fp8ExpertWeights::new(
                WIDTH,
                WIDTH,
                &self.fp8_codes,
                &self.fp8_scales,
                &self.fp8_codes,
                &self.fp8_scales,
                &self.fp8_codes,
                &self.fp8_scales,
            )
            .expect("fixed shared expert geometry");
            let moe = MoEReference::new(
                MoEConfig::new(WIDTH, WIDTH, 4.0, 1, 1.0, true, 0.3)
                    .expect("fixed MoE configuration"),
                &self.gate,
                &self.bias,
                std::slice::from_ref(&routed),
                shared,
            )
            .expect("fixed MoE buffers");
            let ffn = FfnSublayerReference::new(
                moe,
                &self.norm_weight,
                &self.hc_projection,
                &[1.0, 1.0, 1.0],
                &self.hc_base,
                COPIES,
                1.0e-6,
                1,
                1.0e-6,
            )
            .expect("fixed FFN buffers");
            ffn.forward_token(residual, incoming_pre)
        }
    }

    fn residual() -> Vec<u16> {
        let mut result = vec![0x3f80; WIDTH];
        result.extend(vec![0xbf80; WIDTH]);
        result
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(64))]
        /// FFN coefficients come from the residual, while the old attention
        /// pre-mix selects the normalized `MoE` input. Positive scales keep
        /// the two opposite-sign residual copies distinguishable after BF16
        /// rounding and normalization.
        #[test]
        fn incoming_pre_never_changes_fresh_coefficients(magnitude in 0.25_f32..4.0) {
            let buffers = ExpertBuffers::new();
            let residual = residual();
            let first = buffers.forward(&residual, &[magnitude, 0.0])?;
            let second = buffers.forward(&residual, &[0.0, magnitude])?;

            prop_assert_eq!(first.coefficients(), second.coefficients());
            prop_assert_ne!(first.collapsed_bf16(), second.collapsed_bf16());
            prop_assert_ne!(first.normalized_bf16(), second.normalized_bf16());
            prop_assert_ne!(
                first.moe().output_bf16(),
                second.moe().output_bf16(),
                "the changed incoming pre-mix reaches the MoE"
            );
            prop_assert_ne!(
                first.output_bf16(),
                second.output_bf16(),
                "the changed MoE row reaches the FFN post-mix"
            );
        }
    }

    #[test]
    fn constructor_rejects_norm_width_mismatch() {
        let buffers = ExpertBuffers::new();
        let routed = Fp4ExpertWeights::new(
            WIDTH,
            WIDTH,
            &buffers.fp4_codes,
            &buffers.fp4_scales,
            &buffers.fp4_codes,
            &buffers.fp4_scales,
            &buffers.fp4_codes,
            &buffers.fp4_scales,
        )
        .expect("fixed routed geometry");
        let shared = Fp8ExpertWeights::new(
            WIDTH,
            WIDTH,
            &buffers.fp8_codes,
            &buffers.fp8_scales,
            &buffers.fp8_codes,
            &buffers.fp8_scales,
            &buffers.fp8_codes,
            &buffers.fp8_scales,
        )
        .expect("fixed shared geometry");
        let moe = MoEReference::new(
            MoEConfig::new(WIDTH, WIDTH, 4.0, 1, 1.0, true, 0.3).expect("fixed MoE configuration"),
            &buffers.gate,
            &buffers.bias,
            std::slice::from_ref(&routed),
            shared,
        )
        .expect("fixed MoE buffers");

        let error = FfnSublayerReference::new(
            moe,
            &buffers.norm_weight[..WIDTH - 1],
            &buffers.hc_projection,
            &[1.0, 1.0, 1.0],
            &buffers.hc_base,
            COPIES,
            1.0e-6,
            1,
            1.0e-6,
        )
        .expect_err("mismatched norm width must fail");
        assert_eq!(
            error,
            FfnError::Length {
                field: "norm_weight",
                actual: WIDTH - 1,
                expected: WIDTH,
            }
        );
    }

    #[test]
    fn incoming_pre_shape_is_checked_after_fresh_coefficients() {
        let buffers = ExpertBuffers::new();
        let error = buffers
            .forward(&residual(), &[1.0])
            .expect_err("incoming attention pre-mix must match copies");
        assert_eq!(
            error,
            FfnError::HcMix(HcMixError::Length {
                field: "residual",
                actual: COPIES * WIDTH,
                expected: WIDTH,
            })
        );
    }
}
