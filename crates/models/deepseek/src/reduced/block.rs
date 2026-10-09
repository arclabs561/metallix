//! Bounded source-order attention-to-FFN block-tail composition.
//!
//! The caller supplies one token's copy-major residual and already computed
//! attention output. This module derives attention HC coefficients from the
//! residual, applies the HC post-mix, and passes that result to the existing
//! FFN reference. It does not load fixtures, compare source observations, or
//! retain request state.

use thiserror::Error;

use crate::{
    ffn::{FfnDiagnostic, FfnError, FfnSublayerReference},
    hc::{
        HcCoefficients, MAX_SINKHORN_ITERATIONS,
        mixing::{HcMixError, MAX_HC_COPIES, MAX_HC_MIX_WIDTH, hc_post_bf16_reference},
        projection::{HcProjectionError, project_hc_coefficients},
    },
    moe::RoutedExpertSource,
};

const MAX_BLOCK_TAIL_PROJECTION_ELEMENTS: usize = 1 << 20;

/// Borrowed immutable operands for the attention and FFN tail of one block.
#[derive(Clone, Copy, Debug)]
pub struct BlockTailReference<'a> {
    ffn: FfnSublayerReference<'a>,
    attn_projection: &'a [f32],
    attn_scale: &'a [f32; 3],
    attn_base: &'a [f32],
    copies: usize,
    width: usize,
    norm_eps: f32,
    sinkhorn_iters: usize,
    hc_eps: f32,
}

impl<'a> BlockTailReference<'a> {
    pub(crate) const fn geometry(self) -> (usize, usize) {
        (self.copies, self.width)
    }

    /// Validates static attention HC operands and their FFN geometry.
    ///
    /// The attention HC projection is row-major
    /// `[(copies + 2) * copies, copies * hidden_width]`. `ffn` remains the
    /// authority for the following FFN sublayer's independent operands.
    ///
    /// # Errors
    ///
    /// Returns [`BlockTailError::EmptyCopies`], [`BlockTailError::EmptyWidth`],
    /// [`BlockTailError::CopyCountTooLarge`],
    /// [`BlockTailError::WidthTooLarge`], [`BlockTailError::CopyCountMismatch`]
    /// for sizes outside their bounds, [`BlockTailError::InvalidNormEpsilon`],
    /// [`BlockTailError::InvalidHcEpsilon`],
    /// [`BlockTailError::InvalidSinkhornIterations`] for unusable controls,
    /// [`BlockTailError::Length`], [`BlockTailError::ProjectionTooLarge`],
    /// [`BlockTailError::ShapeOverflow`] when a buffer does not match, and
    /// [`BlockTailError::NonFiniteScale`],
    /// [`BlockTailError::NonFiniteProjection`],
    /// [`BlockTailError::NonFiniteBase`] for non-finite Hyper-Connection
    /// parameters.
    #[allow(
        clippy::too_many_arguments,
        reason = "the source stores each attention HC role separately"
    )]
    pub fn new(
        ffn: FfnSublayerReference<'a>,
        attn_projection: &'a [f32],
        attn_scale: &'a [f32; 3],
        attn_base: &'a [f32],
        copies: usize,
        norm_eps: f32,
        sinkhorn_iters: usize,
        hc_eps: f32,
    ) -> Result<Self, BlockTailError> {
        let ffn_copies = ffn.copies();
        if copies != ffn_copies {
            return Err(BlockTailError::CopyCountMismatch {
                requested: copies,
                ffn: ffn_copies,
            });
        }
        if copies == 0 {
            return Err(BlockTailError::EmptyCopies);
        }
        if copies > MAX_HC_COPIES {
            return Err(BlockTailError::CopyCountTooLarge {
                copies,
                maximum: MAX_HC_COPIES,
            });
        }
        let width = ffn.hidden_width();
        if width == 0 {
            return Err(BlockTailError::EmptyWidth);
        }
        if width > MAX_HC_MIX_WIDTH {
            return Err(BlockTailError::WidthTooLarge {
                width,
                maximum: MAX_HC_MIX_WIDTH,
            });
        }
        if !norm_eps.is_finite() || norm_eps <= 0.0 {
            return Err(BlockTailError::InvalidNormEpsilon);
        }
        if sinkhorn_iters == 0 || sinkhorn_iters > MAX_SINKHORN_ITERATIONS {
            return Err(BlockTailError::InvalidSinkhornIterations {
                iterations: sinkhorn_iters,
                maximum: MAX_SINKHORN_ITERATIONS,
            });
        }
        if !hc_eps.is_finite() || hc_eps <= 0.0 {
            return Err(BlockTailError::InvalidHcEpsilon);
        }
        if attn_scale.iter().any(|value| !value.is_finite()) {
            return Err(BlockTailError::NonFiniteScale);
        }

        let residual_elements = checked_product(copies, width, "residual")?;
        let mix_rows = copies
            .checked_add(2)
            .and_then(|value| value.checked_mul(copies))
            .ok_or(BlockTailError::ShapeOverflow { field: "mix rows" })?;
        let projection_elements = checked_product(mix_rows, residual_elements, "projection")?;
        if projection_elements > MAX_BLOCK_TAIL_PROJECTION_ELEMENTS {
            return Err(BlockTailError::ProjectionTooLarge {
                elements: projection_elements,
                maximum: MAX_BLOCK_TAIL_PROJECTION_ELEMENTS,
            });
        }
        if attn_projection.len() != projection_elements {
            return Err(BlockTailError::Length {
                field: "attn_projection",
                actual: attn_projection.len(),
                expected: projection_elements,
            });
        }
        if attn_base.len() != mix_rows {
            return Err(BlockTailError::Length {
                field: "attn_base",
                actual: attn_base.len(),
                expected: mix_rows,
            });
        }
        for (element, &value) in attn_projection.iter().enumerate() {
            if !value.is_finite() {
                return Err(BlockTailError::NonFiniteProjection { element });
            }
        }
        for (element, &value) in attn_base.iter().enumerate() {
            if !value.is_finite() {
                return Err(BlockTailError::NonFiniteBase { element });
            }
        }

        Ok(Self {
            ffn,
            attn_projection,
            attn_scale,
            attn_base,
            copies,
            width,
            norm_eps,
            sinkhorn_iters,
            hc_eps,
        })
    }

    /// Executes one source-ordered attention-to-FFN tail token.
    ///
    /// `residual` is copy-major `[copies, hidden_width]`; `attention` is the
    /// already computed BF16 attention row `[hidden_width]`. A successful
    /// result owns every intermediate and leaves the borrowed operands intact.
    ///
    /// # Errors
    ///
    /// Returns [`BlockTailError::Length`] when `residual` or `attention` does
    /// not match the copies and width, [`BlockTailError::HcProjection`],
    /// [`BlockTailError::HcMix`], [`BlockTailError::Ffn`] when a stage fails,
    /// and [`BlockTailError::AllocationFailed`] when a buffer cannot be
    /// reserved.
    pub fn forward_token(
        &self,
        residual: &[u16],
        attention: &[u16],
    ) -> Result<BlockTailDiagnostic, BlockTailError> {
        self.forward_token_inner(residual, attention, None)
    }

    /// Same as [`Self::forward_token`], with routed experts fetched from `source`
    /// after routing instead of from the FFN's construction-time table.
    ///
    /// # Errors
    ///
    /// Returns [`BlockTailError::Length`] when `residual` or `attention` does
    /// not match the copies and width, [`BlockTailError::HcProjection`],
    /// [`BlockTailError::HcMix`], [`BlockTailError::Ffn`] when a stage fails,
    /// and [`BlockTailError::AllocationFailed`] when a buffer cannot be
    /// reserved.
    pub fn forward_token_with(
        &self,
        residual: &[u16],
        attention: &[u16],
        source: &dyn RoutedExpertSource,
    ) -> Result<BlockTailDiagnostic, BlockTailError> {
        self.forward_token_inner(residual, attention, Some(source))
    }

    fn forward_token_inner(
        &self,
        residual: &[u16],
        attention: &[u16],
        source: Option<&dyn RoutedExpertSource>,
    ) -> Result<BlockTailDiagnostic, BlockTailError> {
        let residual_elements = checked_product(self.copies, self.width, "residual")?;
        if residual.len() != residual_elements {
            return Err(BlockTailError::Length {
                field: "residual",
                actual: residual.len(),
                expected: residual_elements,
            });
        }
        if attention.len() != self.width {
            return Err(BlockTailError::Length {
                field: "attention",
                actual: attention.len(),
                expected: self.width,
            });
        }

        let attention_coefficients = project_hc_coefficients(
            residual,
            self.attn_projection,
            self.attn_scale,
            self.attn_base,
            self.copies,
            self.norm_eps,
            self.sinkhorn_iters,
            self.hc_eps,
        )?;
        let mut after_attention_bf16 = allocate_u16("after_attention", residual_elements)?;
        hc_post_bf16_reference(
            attention,
            residual,
            attention_coefficients.post(),
            attention_coefficients.comb(),
            &mut after_attention_bf16,
        )?;
        let ffn = match source {
            Some(source) => self.ffn.forward_token_with(
                &after_attention_bf16,
                attention_coefficients.pre(),
                source,
            )?,
            None => self
                .ffn
                .forward_token(&after_attention_bf16, attention_coefficients.pre())?,
        };
        Ok(BlockTailDiagnostic {
            attention_coefficients,
            after_attention_bf16,
            ffn,
        })
    }
}

/// Owned observations from one attention-to-FFN block tail.
#[derive(Clone, Debug, PartialEq)]
pub struct BlockTailDiagnostic {
    attention_coefficients: HcCoefficients,
    after_attention_bf16: Vec<u16>,
    ffn: FfnDiagnostic,
}

impl BlockTailDiagnostic {
    /// Returns coefficients derived from the residual for the attention post-mix.
    #[must_use]
    pub const fn attention_coefficients(&self) -> &HcCoefficients {
        &self.attention_coefficients
    }

    /// Returns the copy-major BF16 residual after the attention HC post-mix.
    #[must_use]
    pub fn after_attention_bf16(&self) -> &[u16] {
        &self.after_attention_bf16
    }

    /// Returns the source-ordered FFN observations that consume this tail output.
    #[must_use]
    pub const fn ffn(&self) -> &FfnDiagnostic {
        &self.ffn
    }
}

/// An invalid block-tail construction or execution request.
#[derive(Clone, Debug, Error, PartialEq)]
#[non_exhaustive]
pub enum BlockTailError {
    /// Requested copies do not match the borrowed FFN's validated geometry.
    #[error("block tail copies {requested} do not match FFN copies {ffn}")]
    CopyCountMismatch {
        /// Requested attention HC copies.
        requested: usize,
        /// Borrowed FFN copy count.
        ffn: usize,
    },
    /// At least one residual copy is required.
    #[error("block tail requires at least one residual copy")]
    EmptyCopies,
    /// The copy count exceeds the HC scalar bound.
    #[error("block tail copies {copies} exceeds maximum {maximum}")]
    CopyCountTooLarge {
        /// Requested copy count.
        copies: usize,
        /// Maximum scalar copy count.
        maximum: usize,
    },
    /// The borrowed FFN has no hidden width.
    #[error("block tail requires a nonempty hidden width")]
    EmptyWidth,
    /// The hidden width exceeds the HC scalar bound.
    #[error("block tail width {width} exceeds maximum {maximum}")]
    WidthTooLarge {
        /// Requested hidden width.
        width: usize,
        /// Maximum scalar hidden width.
        maximum: usize,
    },
    /// The residual normalization epsilon must be finite and positive.
    #[error("block tail normalization epsilon must be finite and positive")]
    InvalidNormEpsilon,
    /// The Sinkhorn iteration count is outside the bounded scalar range.
    #[error("block tail Sinkhorn iterations {iterations} is outside 1 through {maximum}")]
    InvalidSinkhornIterations {
        /// Requested Sinkhorn iteration count.
        iterations: usize,
        /// Largest accepted scalar iteration count.
        maximum: usize,
    },
    /// The HC stabilization epsilon must be finite and positive.
    #[error("block tail HC epsilon must be finite and positive")]
    InvalidHcEpsilon,
    /// An attention HC scale is non-finite.
    #[error("block tail attention HC scales must be finite")]
    NonFiniteScale,
    /// A static attention HC projection weight is non-finite.
    #[error("block tail attention HC projection at element {element} is non-finite")]
    NonFiniteProjection {
        /// Projection element index.
        element: usize,
    },
    /// A static attention HC base control is non-finite.
    #[error("block tail attention HC base at element {element} is non-finite")]
    NonFiniteBase {
        /// Base element index.
        element: usize,
    },
    /// A checked static or runtime shape overflowed `usize`.
    #[error("block tail {field} shape overflowed")]
    ShapeOverflow {
        /// Derived shape role.
        field: &'static str,
    },
    /// The attention HC projection exceeds the scalar resource bound.
    #[error("block tail attention projection has {elements} elements, maximum is {maximum}")]
    ProjectionTooLarge {
        /// Requested projection elements.
        elements: usize,
        /// Maximum scalar projection elements.
        maximum: usize,
    },
    /// A supplied static or runtime buffer has an unexpected exact length.
    #[error("block tail {field} length is {actual}, expected {expected}")]
    Length {
        /// Buffer role.
        field: &'static str,
        /// Actual element count.
        actual: usize,
        /// Required element count.
        expected: usize,
    },
    /// A temporary BF16 tail row could not be reserved.
    #[error("could not allocate {elements} BF16 block-tail elements for {field}")]
    AllocationFailed {
        /// Temporary row role.
        field: &'static str,
        /// Required element count.
        elements: usize,
    },
    /// Attention HC projection rejected runtime operands.
    #[error(transparent)]
    HcProjection(#[from] HcProjectionError),
    /// Attention HC post-mix rejected runtime operands.
    #[error(transparent)]
    HcMix(#[from] HcMixError),
    /// The FFN tail rejected its input or static operands.
    #[error(transparent)]
    Ffn(#[from] FfnError),
}

fn checked_product(
    left: usize,
    right: usize,
    field: &'static str,
) -> Result<usize, BlockTailError> {
    left.checked_mul(right)
        .ok_or(BlockTailError::ShapeOverflow { field })
}

fn allocate_u16(field: &'static str, elements: usize) -> Result<Vec<u16>, BlockTailError> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(elements)
        .map_err(|_| BlockTailError::AllocationFailed { field, elements })?;
    values.resize(elements, 0);
    Ok(values)
}
