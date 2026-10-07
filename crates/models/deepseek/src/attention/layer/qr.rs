//! The stateless attention QR prefix shared with the indexer.

use std::num::NonZeroUsize;

use super::{
    LayerAttentionError, LayerAttentionLayoutError, MAX_LAYER_ATTENTION_ELEMENTS,
    ops::{checked_product, fp8_project_bf16, rms_norm_rows},
};

/// One borrowed FP8 checkpoint projection, stored in its runtime orientation.
#[derive(Clone, Copy, Debug)]
pub struct Fp8Projection<'a> {
    /// E4M3FN codes `[outputs, reduction]`.
    pub codes: &'a [u8],
    /// E8M0 scales `[ceil(outputs / 32), reduction / 32]`.
    pub scales: &'a [u8],
}

/// The stateless source-shaped dimensions shared by attention's QR prefix and the indexer.
///
/// This intentionally excludes attention-window, head, and publication settings:
/// QR is computed before either sparse attention or candidate selection consumes
/// it. Keeping that seam explicit lets a candidate source reuse the exact
/// `wq_a` then `RMSNorm` path without constructing an attention cache owner.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AttentionQrLayout {
    batches: NonZeroUsize,
    hidden_dimension: NonZeroUsize,
    q_rank: NonZeroUsize,
    norm_epsilon_bits: u32,
}

impl AttentionQrLayout {
    /// Validates the bounded, stateless QR prefix geometry.
    ///
    /// The bounds cover activation elements, not resident weight bytes or
    /// projection work. Callers must admit those resources independently.
    ///
    /// # Errors
    ///
    /// * [`LayerAttentionLayoutError::InvalidNormEpsilon`] for a non-finite or
    ///   non-positive epsilon.
    /// * [`LayerAttentionLayoutError::UngroupedFp8Reduction`] when the hidden
    ///   width or rank is not a multiple of 32.
    /// * [`LayerAttentionLayoutError::ShapeOverflow`] and
    ///   [`LayerAttentionLayoutError::ElementLimit`] when a staging buffer does
    ///   not fit its bound.
    pub fn new(
        batches: NonZeroUsize,
        hidden_dimension: NonZeroUsize,
        q_rank: NonZeroUsize,
        norm_epsilon: f32,
    ) -> Result<Self, LayerAttentionLayoutError> {
        if !norm_epsilon.is_finite() || norm_epsilon <= 0.0 {
            return Err(LayerAttentionLayoutError::InvalidNormEpsilon);
        }
        for (field, width) in [
            ("hidden dimension", hidden_dimension.get()),
            ("q rank", q_rank.get()),
        ] {
            if !width.is_multiple_of(32) {
                return Err(LayerAttentionLayoutError::UngroupedFp8Reduction { field, width });
            }
        }
        for (field, elements) in [
            (
                "one-position input",
                checked_product(
                    &[batches.get(), hidden_dimension.get()],
                    "one-position input",
                )?,
            ),
            (
                "one-position q rank",
                checked_product(&[batches.get(), q_rank.get()], "one-position q rank")?,
            ),
        ] {
            if elements > MAX_LAYER_ATTENTION_ELEMENTS {
                return Err(LayerAttentionLayoutError::ElementLimit { field, elements });
            }
        }
        Ok(Self {
            batches,
            hidden_dimension,
            q_rank,
            norm_epsilon_bits: norm_epsilon.to_bits(),
        })
    }

    fn norm_epsilon(self) -> f32 {
        f32::from_bits(self.norm_epsilon_bits)
    }
}

/// Borrowed parameters for attention's stateless QR prefix.
#[derive(Clone, Copy, Debug)]
pub struct AttentionQrWeights<'a> {
    pub(crate) wq_a: Fp8Projection<'a>,
    pub(crate) q_norm: &'a [u16],
}

impl<'a> AttentionQrWeights<'a> {
    /// Borrows the FP8 query projection and BF16 normalization row.
    ///
    /// [`prepare_attention_qr`] validates their exact lengths and numerical
    /// values against the supplied layout when executing the projection.
    #[must_use]
    pub const fn new(wq_a: Fp8Projection<'a>, q_norm: &'a [u16]) -> Self {
        Self { wq_a, q_norm }
    }
}

/// BF16 stages shared by attention query preparation and a candidate source.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AttentionQrDiagnostic {
    pub(crate) wq_a: Vec<u16>,
    pub(crate) qr: Vec<u16>,
}

impl AttentionQrDiagnostic {
    /// Returns the BF16 projection in `[batch, position, q_rank]` order.
    #[must_use]
    pub fn wq_a_bf16(&self) -> &[u16] {
        &self.wq_a
    }

    /// Returns the normalized BF16 query prefix in projection row order.
    #[must_use]
    pub fn qr_bf16(&self) -> &[u16] {
        &self.qr
    }
}

/// Computes the source attention QR prefix without owning a cache or state transition.
///
/// `input` is BF16 `[batch, position, hidden_dimension]`. The returned
/// diagnostics retain the BF16 `wq_a` projection and subsequent `RMSNorm`
/// result, which are also the source indexer's `qr` operand.
/// The caller supplies already collapsed, attention-normalized input. This
/// function does not apply Hyper-Connections, load weights, mutate a cache,
/// expand query heads, or execute sparse attention.
///
/// Activation storage is bounded; callers must separately budget checkpoint
/// weight bytes and projection work. Errors return no partial diagnostic.
///
/// # Errors
///
/// * [`LayerAttentionError::InputLength`] when `input` is not whole rows.
/// * [`LayerAttentionError::ShapeOverflow`] and
///   [`LayerAttentionError::ElementLimit`] when the rows do not fit the bound.
/// * [`LayerAttentionError::NonFiniteProjection`] and the wrapped FP8,
///   quantization and norm errors when a stage rejects its input or overflows.
pub fn prepare_attention_qr(
    input: &[u16],
    weights: AttentionQrWeights<'_>,
    layout: AttentionQrLayout,
) -> Result<AttentionQrDiagnostic, LayerAttentionError> {
    let input_stride = checked_product(
        &[layout.batches.get(), layout.hidden_dimension.get()],
        "attention QR input stride",
    )?;
    if input.is_empty() || !input.len().is_multiple_of(input_stride) {
        return Err(LayerAttentionError::InputLength {
            actual: input.len(),
            stride: input_stride,
        });
    }
    if input.len() > MAX_LAYER_ATTENTION_ELEMENTS {
        return Err(LayerAttentionError::ElementLimit {
            field: "attention QR input",
            elements: input.len(),
        });
    }
    let positions = input.len() / input_stride;
    let rows = checked_product(&[layout.batches.get(), positions], "attention QR rows")?;
    let output_elements = checked_product(
        &[rows, layout.q_rank.get()],
        "attention QR projection output",
    )?;
    if output_elements > MAX_LAYER_ATTENTION_ELEMENTS {
        return Err(LayerAttentionError::ElementLimit {
            field: "attention QR projection output",
            elements: output_elements,
        });
    }
    let wq_a = fp8_project_bf16(
        input,
        rows,
        layout.hidden_dimension.get(),
        layout.q_rank.get(),
        weights.wq_a,
    )?;
    let qr = rms_norm_rows(
        &wq_a,
        rows,
        layout.q_rank.get(),
        weights.q_norm,
        layout.norm_epsilon(),
    )?;
    Ok(AttentionQrDiagnostic { wq_a, qr })
}
