//! Request-local L3 owner, candidate, and compressed-attention composition.
//!
//! This narrow session joins the existing ratio-one owner transaction with a
//! runtime candidate projection and the L3 attention ring. It intentionally
//! does not retain a prior-L3 publication for a later L1 score policy: callers
//! receive a publication only after this session's attention succeeds.

use std::num::NonZeroUsize;

use thiserror::Error;

use crate::{
    RotaryFrequency,
    attention::layer::{
        CompressedAttentionPublication, LayerAttentionDiagnostic, LayerAttentionError,
        LayerAttentionLayout, LayerAttentionState, LayerAttentionWeights,
    },
    indexer::{
        cache::{IndexKeyPublicationId, IndexKeyStateError},
        key::{IndexKeyLayout, IndexKeyPreparationExecution, IndexKeyRotaryExecution},
        owner::{
            RatioOneCompressedOwner, RatioOneCompressedOwnerDiagnostic,
            RatioOneCompressedOwnerError, RatioOneOwnerCall, RatioOneOwnerWeights,
        },
        selection::{
            SelectionCall, SelectionDiagnostic, SelectionGeometry, SelectionGeometryError,
        },
    },
};

use super::{CandidateProjection, CandidateProjector, CandidateProjectorError};

/// Immutable state geometry for one batch-one L3 request session.
#[derive(Clone, Copy, Debug)]
pub struct LayerThreeConfig {
    owner_key_layout: IndexKeyLayout,
    owner_input_dimension: NonZeroUsize,
    owner_capacity: NonZeroUsize,
    attention_layout: LayerAttentionLayout,
    window_size: NonZeroUsize,
    index_topk: NonZeroUsize,
}

impl LayerThreeConfig {
    /// Returns the fixed L3 compressed-attention geometry for request composition.
    #[must_use]
    pub(crate) const fn attention_layout(self) -> LayerAttentionLayout {
        self.attention_layout
    }

    /// Validates the explicitly supplied L3 owner and candidate geometry.
    #[allow(
        clippy::too_many_arguments,
        reason = "owner and attention geometries remain explicit at the runtime boundary"
    )]
    pub fn new(
        owner_key_layout: IndexKeyLayout,
        owner_input_dimension: NonZeroUsize,
        owner_capacity: NonZeroUsize,
        attention_layout: LayerAttentionLayout,
        window_size: NonZeroUsize,
        index_topk: NonZeroUsize,
    ) -> Result<Self, LayerThreeSessionError> {
        if owner_key_layout.batches().get() != 1 {
            return Err(LayerThreeSessionError::OwnerBatchCount {
                actual: owner_key_layout.batches().get(),
            });
        }
        if attention_layout.batches().get() != 1 {
            return Err(LayerThreeSessionError::AttentionBatchCount {
                actual: attention_layout.batches().get(),
            });
        }
        if attention_layout.compression().is_none() {
            return Err(LayerThreeSessionError::AttentionSourceLayer);
        }
        if attention_layout.compression().map(|(_, ratio)| ratio.get()) != Some(1) {
            return Err(LayerThreeSessionError::AttentionCompressionRatio);
        }
        if attention_layout.window() != window_size {
            return Err(LayerThreeSessionError::WindowMismatch {
                configured: window_size.get(),
                attention: attention_layout.window().get(),
            });
        }
        if owner_input_dimension != attention_layout.hidden_dimension() {
            return Err(LayerThreeSessionError::InputDimensionMismatch {
                owner: owner_input_dimension.get(),
                attention: attention_layout.hidden_dimension().get(),
            });
        }
        if owner_key_layout.latent_dimension() != attention_layout.head_dimension() {
            return Err(LayerThreeSessionError::LatentDimensionMismatch {
                owner: owner_key_layout.latent_dimension().get(),
                attention: attention_layout.head_dimension().get(),
            });
        }
        if owner_key_layout.key_dimension() != attention_layout.head_dimension() {
            return Err(LayerThreeSessionError::KeyDimensionMismatch {
                owner: owner_key_layout.key_dimension().get(),
                attention: attention_layout.head_dimension().get(),
            });
        }
        if owner_key_layout.rope_pairs() != attention_layout.rope_pairs() {
            return Err(LayerThreeSessionError::RopePairMismatch {
                owner: owner_key_layout.rope_pairs().get(),
                attention: attention_layout.rope_pairs().get(),
            });
        }
        Ok(Self {
            owner_key_layout,
            owner_input_dimension,
            owner_capacity,
            attention_layout,
            window_size,
            index_topk,
        })
    }
}

/// Borrowed runtime operands for one L3 owner/selection/attention call.
#[derive(Clone, Copy, Debug)]
pub struct LayerThreeCall<'a> {
    input: &'a [u16],
    positions: NonZeroUsize,
    frequencies: &'a [RotaryFrequency],
    owner_weights: RatioOneOwnerWeights<'a>,
    candidate: CandidateProjector<'a>,
    attention_weights: LayerAttentionWeights<'a>,
}

impl<'a> LayerThreeCall<'a> {
    /// Groups one contiguous source partition and its caller-owned numerical operands.
    #[must_use]
    pub const fn new(
        input: &'a [u16],
        positions: NonZeroUsize,
        frequencies: &'a [RotaryFrequency],
        owner_weights: RatioOneOwnerWeights<'a>,
        candidate: CandidateProjector<'a>,
        attention_weights: LayerAttentionWeights<'a>,
    ) -> Self {
        Self {
            input,
            positions,
            frequencies,
            owner_weights,
            candidate,
            attention_weights,
        }
    }
}

/// One fully committed L3 publication and attention result.
#[derive(Clone, Debug)]
pub struct LayerThreeStepOutput {
    publication: IndexKeyPublicationId,
    owner: RatioOneCompressedOwnerDiagnostic,
    candidate: CandidateProjection,
    selection: SelectionDiagnostic,
    key_prefix: Vec<u16>,
    kv_prefix: Vec<u16>,
    attention: LayerAttentionDiagnostic,
}

impl LayerThreeStepOutput {
    /// Returns the publication identity shared by owner, candidate, and attention.
    #[must_use]
    pub const fn publication(&self) -> IndexKeyPublicationId {
        self.publication
    }

    /// Returns owner projection, compressor, and prepared-prefix diagnostics.
    #[must_use]
    pub const fn owner(&self) -> &RatioOneCompressedOwnerDiagnostic {
        &self.owner
    }

    /// Returns computed query scores and the candidate mask.
    #[must_use]
    pub const fn candidate(&self) -> &CandidateProjection {
        &self.candidate
    }

    /// Returns computed masked scores and selected compressed IDs.
    #[must_use]
    pub const fn selection(&self) -> &SelectionDiagnostic {
        &self.selection
    }

    /// Returns selected compressed IDs in attention's window-plus-prefix domain.
    #[must_use]
    pub fn selected_indices(&self) -> &[i32] {
        &self.selection.indices
    }

    /// Returns the completed L3 owner key prefix.
    #[must_use]
    pub fn key_prefix(&self) -> &[u16] {
        &self.key_prefix
    }

    /// Returns the completed L3 owner compressed-KV prefix.
    #[must_use]
    pub fn kv_prefix(&self) -> &[u16] {
        &self.kv_prefix
    }

    /// Returns all computed L3 attention stages.
    #[must_use]
    pub const fn attention(&self) -> &LayerAttentionDiagnostic {
        &self.attention
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Lifecycle {
    Healthy,
    Poisoned,
}

/// Live L3 state. A post-owner attention rejection poisons this session because
/// the owner publication is already committed and cannot be rolled back.
pub struct LayerThreeSession {
    config: LayerThreeConfig,
    owner: RatioOneCompressedOwner,
    attention: LayerAttentionState,
    expected_epoch: u64,
    lifecycle: Lifecycle,
}

impl LayerThreeSession {
    /// Allocates a synchronized empty ratio-one owner and L3 attention ring.
    pub fn new(
        config: LayerThreeConfig,
        source_layer: u16,
        compressor_norm: &[u16],
        compressor_epsilon: f32,
    ) -> Result<Self, LayerThreeSessionError> {
        // The owner and its own attention must agree on the publishing layer.
        let expected = config
            .attention_layout
            .compression()
            .map(|(source, _)| source);
        if expected != Some(source_layer) {
            return Err(LayerThreeSessionError::SourceLayer {
                actual: source_layer,
            });
        }
        let owner = RatioOneCompressedOwner::new(
            config.owner_key_layout,
            config.owner_input_dimension,
            config.owner_capacity,
            source_layer,
            compressor_norm,
            compressor_epsilon,
        )?;
        Ok(Self {
            config,
            attention: LayerAttentionState::new(config.attention_layout),
            expected_epoch: owner.epoch(),
            owner,
            lifecycle: Lifecycle::Healthy,
        })
    }

    /// Selects the complete index-key preparation implementation for subsequent calls.
    #[must_use]
    pub fn with_key_preparation_execution(
        mut self,
        key_preparation_execution: IndexKeyPreparationExecution,
    ) -> Self {
        self.owner = self
            .owner
            .with_key_preparation_execution(key_preparation_execution);
        self
    }

    /// Selects the legacy rotary-only implementation through its preparation mapping.
    #[must_use]
    pub fn with_key_rotary_execution(self, key_rotary_execution: IndexKeyRotaryExecution) -> Self {
        self.with_key_preparation_execution(key_rotary_execution.into())
    }

    /// Computes one L3 partition. Every admitted failure poisons the session.
    pub fn step(
        &mut self,
        call: LayerThreeCall<'_>,
    ) -> Result<LayerThreeStepOutput, LayerThreeSessionError> {
        if self.lifecycle == Lifecycle::Poisoned {
            return Err(LayerThreeSessionError::Poisoned);
        }
        self.lifecycle = Lifecycle::Poisoned;
        let result = self.step_admitted(&call);
        if result.is_ok() {
            self.lifecycle = Lifecycle::Healthy;
        }
        result
    }

    fn step_admitted(
        &mut self,
        call: &LayerThreeCall<'_>,
    ) -> Result<LayerThreeStepOutput, LayerThreeSessionError> {
        let start = self.owner.next_position();
        if self.owner.epoch() != self.expected_epoch {
            return Err(LayerThreeSessionError::OwnerEpochMismatch {
                actual: self.owner.epoch(),
                expected: self.expected_epoch,
            });
        }
        let publication = IndexKeyPublicationId::new(
            self.owner.source_layer(),
            self.expected_epoch,
            self.owner.next_call_id(),
        );
        let pending = self.owner.prepare(RatioOneOwnerCall::new(
            publication,
            start,
            call.positions,
            call.input,
            call.frequencies,
            call.owner_weights,
        ))?;
        let key_prefix = owned_prefix(pending.key_prefix(0)?, "staged key prefix")?;
        let kv_prefix = owned_prefix(pending.kv_prefix(0)?, "staged KV prefix")?;
        let key_count = NonZeroUsize::new(
            key_prefix.len() / self.config.owner_key_layout.key_dimension().get(),
        )
        .ok_or(LayerThreeSessionError::EmptyKeyPrefix)?;
        let window_offset = if start == 0 {
            call.positions.get()
        } else {
            self.config.window_size.get()
        };
        let selection_call = SelectionCall::new(
            pending.publication(),
            0,
            SelectionGeometry::new(
                start,
                call.positions,
                key_count,
                NonZeroUsize::new(1).expect("ratio-one selection geometry"),
                window_offset,
            )?,
        );
        let candidate =
            call.candidate
                .project(call.input, call.frequencies, &key_prefix, selection_call)?;
        let selection = candidate.select(self.config.index_topk.get())?;
        let owner = pending.commit()?;
        let attention = self.attention.forward(
            call.input,
            start,
            call.frequencies,
            call.attention_weights,
            CompressedAttentionPublication {
                source_layer: publication.source_layer(),
                epoch: publication.epoch(),
                call_id: publication.call_id(),
                numerical_bf16: &kv_prefix,
                indices: &selection.indices,
            },
        )?;
        Ok(LayerThreeStepOutput {
            publication,
            owner,
            candidate,
            selection,
            key_prefix,
            kv_prefix,
            attention,
        })
    }

    /// Clears owner prefixes and the attention ring together, then admits epoch+1.
    pub fn reset(&mut self) -> Result<(), LayerThreeSessionError> {
        self.lifecycle = Lifecycle::Poisoned;
        let next_epoch = self
            .expected_epoch
            .checked_add(1)
            .ok_or(LayerThreeSessionError::EpochOverflow)?;
        let mut staged_attention = self.attention.clone();
        staged_attention.reset()?;
        self.owner.reset()?;
        if self.owner.epoch() != next_epoch {
            return Err(LayerThreeSessionError::OwnerEpochMismatch {
                actual: self.owner.epoch(),
                expected: next_epoch,
            });
        }
        self.attention = staged_attention;
        self.expected_epoch = next_epoch;
        self.lifecycle = Lifecycle::Healthy;
        Ok(())
    }

    /// Returns the owner token position expected by the next call.
    #[must_use]
    pub const fn next_start(&self) -> usize {
        self.owner.next_position()
    }

    /// Returns whether a prior admitted failure requires reset.
    #[must_use]
    pub const fn is_poisoned(&self) -> bool {
        matches!(self.lifecycle, Lifecycle::Poisoned)
    }
}

/// Errors from L3 session construction, composition, lifecycle, or reset.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum LayerThreeSessionError {
    #[error("layer-three owner requires batch one, got {actual}")]
    OwnerBatchCount { actual: usize },
    #[error("layer-three attention requires batch one, got {actual}")]
    AttentionBatchCount { actual: usize },
    #[error("layer-three attention layout has no compressed source layer")]
    AttentionSourceLayer,
    #[error("layer-three attention layout does not use compression ratio one")]
    AttentionCompressionRatio,
    #[error("layer-three configured window {configured} differs from attention window {attention}")]
    WindowMismatch { configured: usize, attention: usize },
    #[error(
        "layer-three owner input dimension {owner} differs from attention hidden dimension {attention}"
    )]
    InputDimensionMismatch { owner: usize, attention: usize },
    #[error(
        "layer-three owner latent dimension {owner} differs from attention head dimension {attention}"
    )]
    LatentDimensionMismatch { owner: usize, attention: usize },
    #[error(
        "layer-three owner key dimension {owner} differs from attention head dimension {attention}"
    )]
    KeyDimensionMismatch { owner: usize, attention: usize },
    #[error("layer-three owner rope pairs {owner} differs from attention rope pairs {attention}")]
    RopePairMismatch { owner: usize, attention: usize },
    #[error("layer-three session source layer {actual} differs from its attention layout")]
    SourceLayer { actual: u16 },
    #[error("layer-three session is poisoned; reset is required")]
    Poisoned,
    #[error("layer-three owner epoch {actual} differs from expected {expected}")]
    OwnerEpochMismatch { actual: u64, expected: u64 },
    #[error("layer-three session epoch overflowed")]
    EpochOverflow,
    #[error("layer-three owner produced an empty key prefix")]
    EmptyKeyPrefix,
    #[error("could not allocate {elements} BF16 elements for {field}")]
    AllocationFailed {
        field: &'static str,
        elements: usize,
    },
    #[error(transparent)]
    Owner(#[from] RatioOneCompressedOwnerError),
    #[error(transparent)]
    Prefix(#[from] IndexKeyStateError),
    #[error(transparent)]
    SelectionGeometry(#[from] SelectionGeometryError),
    #[error(transparent)]
    Candidate(#[from] CandidateProjectorError),
    #[error(transparent)]
    Attention(#[from] LayerAttentionError),
}

fn owned_prefix(values: &[u16], field: &'static str) -> Result<Vec<u16>, LayerThreeSessionError> {
    let mut owned = Vec::new();
    owned.try_reserve_exact(values.len()).map_err(|_| {
        LayerThreeSessionError::AllocationFailed {
            field,
            elements: values.len(),
        }
    })?;
    owned.extend_from_slice(values);
    Ok(owned)
}
