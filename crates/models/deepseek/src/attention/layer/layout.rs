//! Layer-attention geometry and its provenance contract.

use std::num::NonZeroUsize;

use super::{LayerAttentionLayoutError, MAX_LAYER_ATTENTION_ELEMENTS, ops::checked_product};

/// The model-local dimensions and provenance contract for one layer attention.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LayerAttentionLayout {
    pub(super) batches: NonZeroUsize,
    pub(super) hidden_dimension: NonZeroUsize,
    pub(super) heads: NonZeroUsize,
    pub(super) head_dimension: NonZeroUsize,
    pub(super) rope_pairs: NonZeroUsize,
    pub(super) q_rank: NonZeroUsize,
    pub(super) window: NonZeroUsize,
    pub(super) groups: NonZeroUsize,
    pub(super) output_rank: NonZeroUsize,
    pub(super) expected_source_layer: Option<u16>,
    pub(super) compressed_ratio: Option<NonZeroUsize>,
    norm_epsilon_bits: u32,
    softmax_scale_bits: u32,
}

impl LayerAttentionLayout {
    /// Crate-private geometry accessors let reduced request sessions verify
    /// that their producer publication matches this attention owner before
    /// allocating request state.
    #[must_use]
    pub(crate) const fn batches(self) -> NonZeroUsize {
        self.batches
    }

    #[must_use]
    pub(crate) const fn hidden_dimension(self) -> NonZeroUsize {
        self.hidden_dimension
    }

    #[must_use]
    pub(crate) const fn head_dimension(self) -> NonZeroUsize {
        self.head_dimension
    }

    #[must_use]
    pub(crate) const fn rope_pairs(self) -> NonZeroUsize {
        self.rope_pairs
    }

    #[must_use]
    pub(crate) const fn window(self) -> NonZeroUsize {
        self.window
    }

    #[must_use]
    pub(crate) fn compression(self) -> Option<(u16, NonZeroUsize)> {
        self.expected_source_layer.zip(self.compressed_ratio)
    }

    pub(crate) fn is_batch_one_window_only(self, width: usize) -> bool {
        self.batches.get() == 1
            && self.hidden_dimension.get() == width
            && self.expected_source_layer.is_none()
    }

    /// Creates a bounded source-shaped layout.
    ///
    /// `expected_source_layer` identifies the only producer whose borrowed
    /// compressed numerical values may be used. `compressed_ratio` controls
    /// the per-query causal limit during a prefill (a value of one is the
    /// pinned initial layer-4 arrangement).
    ///
    /// # Errors
    ///
    /// * [`LayerAttentionLayoutError::RopeExceedsHead`],
    ///   [`LayerAttentionLayoutError::HeadsNotGrouped`] and
    ///   [`LayerAttentionLayoutError::UngroupedFp8Reduction`] for dimensions
    ///   the source path cannot use.
    /// * [`LayerAttentionLayoutError::InvalidNormEpsilon`] and
    ///   [`LayerAttentionLayoutError::InvalidSoftmaxScale`] for a non-finite or
    ///   non-positive scalar.
    /// * [`LayerAttentionLayoutError::ShapeOverflow`] and
    ///   [`LayerAttentionLayoutError::ElementLimit`] when a staging buffer does
    ///   not fit its bound.
    #[allow(
        clippy::too_many_arguments,
        reason = "source dimensions stay explicit at the adapter boundary"
    )]
    pub fn new(
        batches: NonZeroUsize,
        hidden_dimension: NonZeroUsize,
        heads: NonZeroUsize,
        head_dimension: NonZeroUsize,
        rope_pairs: NonZeroUsize,
        q_rank: NonZeroUsize,
        window: NonZeroUsize,
        groups: NonZeroUsize,
        output_rank: NonZeroUsize,
        expected_source_layer: u16,
        compressed_ratio: NonZeroUsize,
        norm_epsilon: f32,
        softmax_scale: f32,
    ) -> Result<Self, LayerAttentionLayoutError> {
        Self::new_inner(
            batches,
            hidden_dimension,
            heads,
            head_dimension,
            rope_pairs,
            q_rank,
            window,
            groups,
            output_rank,
            Some((expected_source_layer, compressed_ratio)),
            norm_epsilon,
            softmax_scale,
        )
    }

    /// Creates the source-shaped window-only layout used by a layer without a
    /// compressed-KV publication.  It cannot be passed to
    /// [`LayerAttentionState::forward`](super::LayerAttentionState::forward); use
    /// [`LayerAttentionState::forward_window_only`](super::LayerAttentionState::forward_window_only)
    /// instead.
    ///
    /// # Errors
    ///
    /// * [`LayerAttentionLayoutError::RopeExceedsHead`],
    ///   [`LayerAttentionLayoutError::HeadsNotGrouped`] and
    ///   [`LayerAttentionLayoutError::UngroupedFp8Reduction`] for dimensions
    ///   the source path cannot use.
    /// * [`LayerAttentionLayoutError::InvalidNormEpsilon`] and
    ///   [`LayerAttentionLayoutError::InvalidSoftmaxScale`] for a non-finite or
    ///   non-positive scalar.
    /// * [`LayerAttentionLayoutError::ShapeOverflow`] and
    ///   [`LayerAttentionLayoutError::ElementLimit`] when a staging buffer does
    ///   not fit its bound.
    #[allow(
        clippy::too_many_arguments,
        reason = "the source-visible attention dimensions remain explicit"
    )]
    pub fn new_window_only(
        batches: NonZeroUsize,
        hidden_dimension: NonZeroUsize,
        heads: NonZeroUsize,
        head_dimension: NonZeroUsize,
        rope_pairs: NonZeroUsize,
        q_rank: NonZeroUsize,
        window: NonZeroUsize,
        groups: NonZeroUsize,
        output_rank: NonZeroUsize,
        norm_epsilon: f32,
        softmax_scale: f32,
    ) -> Result<Self, LayerAttentionLayoutError> {
        Self::new_inner(
            batches,
            hidden_dimension,
            heads,
            head_dimension,
            rope_pairs,
            q_rank,
            window,
            groups,
            output_rank,
            None,
            norm_epsilon,
            softmax_scale,
        )
    }

    #[allow(
        clippy::too_many_arguments,
        reason = "the shared constructor keeps compressed and window-only layouts identical"
    )]
    fn new_inner(
        batches: NonZeroUsize,
        hidden_dimension: NonZeroUsize,
        heads: NonZeroUsize,
        head_dimension: NonZeroUsize,
        rope_pairs: NonZeroUsize,
        q_rank: NonZeroUsize,
        window: NonZeroUsize,
        groups: NonZeroUsize,
        output_rank: NonZeroUsize,
        compression: Option<(u16, NonZeroUsize)>,
        norm_epsilon: f32,
        softmax_scale: f32,
    ) -> Result<Self, LayerAttentionLayoutError> {
        if !norm_epsilon.is_finite() || norm_epsilon <= 0.0 {
            return Err(LayerAttentionLayoutError::InvalidNormEpsilon);
        }
        if !softmax_scale.is_finite() || softmax_scale <= 0.0 {
            return Err(LayerAttentionLayoutError::InvalidSoftmaxScale);
        }
        let rope_width =
            rope_pairs
                .get()
                .checked_mul(2)
                .ok_or(LayerAttentionLayoutError::ShapeOverflow {
                    field: "rope width",
                })?;
        if rope_width > head_dimension.get() {
            return Err(LayerAttentionLayoutError::RopeExceedsHead {
                rope_width,
                head_dimension: head_dimension.get(),
            });
        }
        if !heads.get().is_multiple_of(groups.get()) {
            return Err(LayerAttentionLayoutError::HeadsNotGrouped {
                heads: heads.get(),
                groups: groups.get(),
            });
        }
        for (field, width) in [
            ("hidden dimension", hidden_dimension.get()),
            ("q rank", q_rank.get()),
            ("head dimension", head_dimension.get()),
            (
                "groups * output rank",
                checked_product(&[groups.get(), output_rank.get()], "flattened output rank")?,
            ),
        ] {
            if !width.is_multiple_of(32) {
                return Err(LayerAttentionLayoutError::UngroupedFp8Reduction { field, width });
            }
        }
        let ring_elements = checked_product(
            &[batches.get(), window.get(), head_dimension.get()],
            "window ring",
        )?;
        for (field, elements) in [
            ("window ring", ring_elements),
            (
                "one-position input",
                checked_product(
                    &[batches.get(), hidden_dimension.get()],
                    "one-position input",
                )?,
            ),
            (
                "one-position query",
                checked_product(
                    &[batches.get(), heads.get(), head_dimension.get()],
                    "one-position query",
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
            heads,
            head_dimension,
            rope_pairs,
            q_rank,
            window,
            groups,
            output_rank,
            expected_source_layer: compression.map(|(layer, _)| layer),
            compressed_ratio: compression.map(|(_, ratio)| ratio),
            norm_epsilon_bits: norm_epsilon.to_bits(),
            softmax_scale_bits: softmax_scale.to_bits(),
        })
    }

    /// The producer layer accepted for compressed numerical publications.
    #[must_use]
    pub const fn expected_source_layer(self) -> u16 {
        match self.expected_source_layer {
            Some(layer) => layer,
            None => 0,
        }
    }
}

impl LayerAttentionLayout {
    pub(super) fn norm_epsilon(self) -> f32 {
        f32::from_bits(self.norm_epsilon_bits)
    }
    pub(super) fn softmax_scale(self) -> f32 {
        f32::from_bits(self.softmax_scale_bits)
    }
}
