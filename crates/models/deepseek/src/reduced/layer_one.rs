//! Request-local reduced L1 owner, direct index scoring, and attention.
//!
//! This session composes the source-shaped ratio-two owner with direct causal
//! index selection and the L1 compressed-attention ring. It is deliberately a
//! reduced, batch-one numerical component: it accepts live operands and never
//! loads checkpoints, capture cases, or source-oracle data.

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
        key::{IndexKeyPreparationExecution, IndexKeyRotaryExecution},
        query::{
            CandidateQueryLayout, CandidateQueryWeights, IndexKeyView, IndexScoreExecution,
            ScoredQueryDiagnostic, ScoredQueryError, prepare_scored_query_with_execution,
        },
    },
    precision::bf16_to_f32,
    selection::{SelectionError, select_indices},
};

use super::{
    RatioTwoCompressedOwner, RatioTwoOwnerCall, RatioTwoOwnerDiagnostic, RatioTwoOwnerError,
    RatioTwoOwnerLayout, RatioTwoOwnerWeights,
};

const SOURCE_RATIO: usize = 2;
const NEGATIVE_INFINITY_BF16: u16 = 0xff80;

/// Fixed reduced geometry for a batch-one L1 owner/query/attention request.
#[derive(Clone, Copy, Debug)]
pub struct LayerOneConfig {
    owner_layout: RatioTwoOwnerLayout,
    attention_layout: LayerAttentionLayout,
    index_topk: NonZeroUsize,
    source_layer: u16,
    previous_owner_layer: u16,
}

impl LayerOneConfig {
    /// Returns the fixed L1 compressed-attention geometry for request composition.
    #[must_use]
    pub(crate) const fn attention_layout(self) -> LayerAttentionLayout {
        self.attention_layout
    }

    /// Validates the L1 geometry shared by the ratio-two owner and attention ring.
    pub fn new(
        owner_layout: RatioTwoOwnerLayout,
        attention_layout: LayerAttentionLayout,
        index_topk: NonZeroUsize,
    ) -> Result<Self, LayerOneSessionError> {
        if owner_layout.batches().get() != 1 {
            return Err(LayerOneSessionError::OwnerBatchCount {
                actual: owner_layout.batches().get(),
            });
        }
        if attention_layout.batches().get() != 1 {
            return Err(LayerOneSessionError::AttentionBatchCount {
                actual: attention_layout.batches().get(),
            });
        }
        // The owner publishes under the source layer its attention layout expects to consume.
        let Some((source_layer, ratio)) = attention_layout.compression() else {
            return Err(LayerOneSessionError::AttentionSourceLayer);
        };
        if ratio.get() != SOURCE_RATIO {
            return Err(LayerOneSessionError::AttentionCompressionRatio);
        }
        if owner_layout.input_dimension() != attention_layout.hidden_dimension() {
            return Err(LayerOneSessionError::InputDimensionMismatch {
                owner: owner_layout.input_dimension().get(),
                attention: attention_layout.hidden_dimension().get(),
            });
        }
        if owner_layout.latent_dimension() != attention_layout.head_dimension() {
            return Err(LayerOneSessionError::LatentDimensionMismatch {
                owner: owner_layout.latent_dimension().get(),
                attention: attention_layout.head_dimension().get(),
            });
        }
        if owner_layout.rope_pairs() != attention_layout.rope_pairs() {
            return Err(LayerOneSessionError::RopePairMismatch {
                owner: owner_layout.rope_pairs().get(),
                attention: attention_layout.rope_pairs().get(),
            });
        }
        Ok(Self {
            owner_layout,
            attention_layout,
            index_topk,
            source_layer,
            previous_owner_layer: 3,
        })
    }

    /// Names the ratio-one owner whose previous-call keys score an incomplete group.
    ///
    /// [`Self::new`] accepts layer 3, the reduced schedule's owner.
    #[must_use]
    pub const fn with_previous_owner_layer(mut self, layer: u16) -> Self {
        self.previous_owner_layer = layer;
        self
    }
}

/// Borrowed preceding L3 key publication used only for an incomplete L1 group.
#[derive(Clone, Copy, Debug)]
pub struct PreviousLayerThreeKeys<'a> {
    publication: IndexKeyPublicationId,
    keys: &'a [u16],
}

impl<'a> PreviousLayerThreeKeys<'a> {
    /// Associates a live L3 publication identity with its full valid key prefix.
    #[must_use]
    pub const fn new(publication: IndexKeyPublicationId, keys: &'a [u16]) -> Self {
        Self { publication, keys }
    }
}

/// Borrowed live numerical operands for one contiguous L1 request partition.
#[derive(Clone, Copy, Debug)]
pub struct LayerOneCall<'a> {
    input: &'a [u16],
    positions: NonZeroUsize,
    frequencies: &'a [RotaryFrequency],
    owner_weights: RatioTwoOwnerWeights<'a>,
    query_weights: CandidateQueryWeights<'a>,
    query_layout: CandidateQueryLayout,
    attention_weights: LayerAttentionWeights<'a>,
    previous_layer_three: Option<PreviousLayerThreeKeys<'a>>,
}

impl<'a> LayerOneCall<'a> {
    /// Groups one live partition with its caller-owned weights and full `RoPE` table.
    #[must_use]
    #[allow(
        clippy::too_many_arguments,
        reason = "the source owner, query, and attention operands remain explicit"
    )]
    pub const fn new(
        input: &'a [u16],
        positions: NonZeroUsize,
        frequencies: &'a [RotaryFrequency],
        owner_weights: RatioTwoOwnerWeights<'a>,
        query_weights: CandidateQueryWeights<'a>,
        query_layout: CandidateQueryLayout,
        attention_weights: LayerAttentionWeights<'a>,
        previous_layer_three: Option<PreviousLayerThreeKeys<'a>>,
    ) -> Self {
        Self {
            input,
            positions,
            frequencies,
            owner_weights,
            query_weights,
            query_layout,
            attention_weights,
            previous_layer_three,
        }
    }
}

/// One committed L1 owner publication, direct score result, and attention result.
#[derive(Clone, Debug)]
pub struct LayerOneStepOutput {
    publication: IndexKeyPublicationId,
    owner: RatioTwoOwnerDiagnostic,
    scored: ScoredQueryDiagnostic,
    causal_scores: Vec<u16>,
    indices: Vec<i32>,
    key_prefix: Vec<u16>,
    kv_prefix: Vec<u16>,
    score_key_prefix: Vec<u16>,
    attention: LayerAttentionDiagnostic,
}

impl LayerOneStepOutput {
    /// Returns the L1 publication identity shared by owner and attention.
    #[must_use]
    pub const fn publication(&self) -> IndexKeyPublicationId {
        self.publication
    }

    /// Returns the owner projection, compressor, and prepared-prefix diagnostics.
    #[must_use]
    pub const fn owner(&self) -> &RatioTwoOwnerDiagnostic {
        &self.owner
    }

    /// Returns the direct query and score-preparation diagnostics.
    #[must_use]
    pub const fn scored(&self) -> &ScoredQueryDiagnostic {
        &self.scored
    }

    /// Returns BF16 scores after the ratio-two causal mask.
    #[must_use]
    pub fn causal_scores(&self) -> &[u16] {
        &self.causal_scores
    }

    /// Returns selected IDs in attention's window-plus-L1-prefix domain.
    #[must_use]
    pub fn selected_indices(&self) -> &[i32] {
        &self.indices
    }

    /// Returns the current owned L1 prepared key prefix.
    #[must_use]
    pub fn key_prefix(&self) -> &[u16] {
        &self.key_prefix
    }

    /// Returns the current owned L1 compressed-KV prefix.
    #[must_use]
    pub fn kv_prefix(&self) -> &[u16] {
        &self.kv_prefix
    }

    /// Returns the key prefix actually consumed by direct score preparation.
    #[must_use]
    pub fn score_key_prefix(&self) -> &[u16] {
        &self.score_key_prefix
    }

    /// Returns all computed L1 attention stages.
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

struct PreparedCall<'a> {
    start: usize,
    token_frequencies: &'a [RotaryFrequency],
    completed_frequencies: Vec<RotaryFrequency>,
    old_key_count: usize,
    publication: IndexKeyPublicationId,
}

/// Live reduced L1 state without checkpoint loading or fixture-owned inputs.
///
/// An admitted call commits the owner before direct query scoring and attention.
/// A later failure therefore poisons this session and requires [`Self::reset`];
/// this type does not claim cross-owner rollback.
pub struct LayerOneSession {
    config: LayerOneConfig,
    score_execution: IndexScoreExecution,
    owner: RatioTwoCompressedOwner,
    attention: LayerAttentionState,
    expected_epoch: u64,
    lifecycle: Lifecycle,
}

impl LayerOneSession {
    /// Allocates synchronized empty ratio-two owner and L1 attention state.
    pub fn new(
        config: LayerOneConfig,
        compressor_norm: &[u16],
    ) -> Result<Self, LayerOneSessionError> {
        let owner = RatioTwoCompressedOwner::new(
            config.owner_layout,
            config.source_layer,
            compressor_norm,
        )?;
        Ok(Self {
            config,
            score_execution: IndexScoreExecution::Scalar,
            attention: LayerAttentionState::new(config.attention_layout),
            expected_epoch: owner.epoch(),
            owner,
            lifecycle: Lifecycle::Healthy,
        })
    }

    /// Selects the score-stage implementation for subsequent calls.
    ///
    /// Construction remains scalar by default so direct users retain the
    /// source-authoritative path. This model-local choice has no effect on
    /// owner, attention, or reset state.
    #[must_use]
    pub const fn with_score_execution(mut self, score_execution: IndexScoreExecution) -> Self {
        self.score_execution = score_execution;
        self
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

    /// Computes one L1 partition. Every admitted error poisons this session.
    pub fn step(
        &mut self,
        call: LayerOneCall<'_>,
    ) -> Result<LayerOneStepOutput, LayerOneSessionError> {
        if self.lifecycle == Lifecycle::Poisoned {
            return Err(LayerOneSessionError::Poisoned);
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
        call: &LayerOneCall<'_>,
    ) -> Result<LayerOneStepOutput, LayerOneSessionError> {
        let prepared = self.prepare_call(call)?;
        let owner = self.owner.forward(RatioTwoOwnerCall::new(
            prepared.publication,
            prepared.start,
            call.positions,
            call.input,
            &prepared.completed_frequencies,
            call.owner_weights,
        ))?;
        let key_prefix = owned_prefix(self.owner.key_prefix(0)?, "L1 key prefix")?;
        let kv_prefix = owned_prefix(self.owner.kv_prefix(0)?, "L1 KV prefix")?;
        let score_key_prefix = self.score_prefix(
            &owner,
            prepared.start,
            prepared.old_key_count,
            call.previous_layer_three,
        )?;
        let keys = IndexKeyView::new(&score_key_prefix, self.config.owner_layout.key_dimension())?;
        let scored = prepare_scored_query_with_execution(
            call.input,
            prepared.token_frequencies,
            call.query_weights,
            call.query_layout,
            keys,
            self.score_execution,
        )?;
        let offset = if prepared.start == 0 {
            call.positions.get()
        } else {
            self.config.attention_layout.window().get()
        };
        let (causal_scores, indices) = causal_select(
            &scored.scores,
            prepared.start,
            call.positions.get(),
            score_key_prefix.len() / self.config.owner_layout.key_dimension().get(),
            self.config.index_topk.get(),
            offset,
        )?;
        let attention = self.attention.forward(
            call.input,
            prepared.start,
            prepared.token_frequencies,
            call.attention_weights,
            CompressedAttentionPublication {
                source_layer: prepared.publication.source_layer(),
                epoch: prepared.publication.epoch(),
                call_id: prepared.publication.call_id(),
                numerical_bf16: &kv_prefix,
                indices: &indices,
            },
        )?;
        Ok(LayerOneStepOutput {
            publication: prepared.publication,
            owner,
            scored,
            causal_scores,
            indices,
            key_prefix,
            kv_prefix,
            score_key_prefix,
            attention,
        })
    }

    fn prepare_call<'a>(
        &self,
        call: &LayerOneCall<'a>,
    ) -> Result<PreparedCall<'a>, LayerOneSessionError> {
        let start = self.owner.next_position();
        let end = start
            .checked_add(call.positions.get())
            .ok_or(LayerOneSessionError::PositionOverflow)?;
        if self.owner.epoch() != self.expected_epoch {
            return Err(LayerOneSessionError::OwnerEpochMismatch {
                actual: self.owner.epoch(),
                expected: self.expected_epoch,
            });
        }
        let (query_batches, query_hidden) = call.query_layout.input_geometry();
        if query_batches != 1 || query_hidden != self.config.owner_layout.input_dimension().get() {
            return Err(LayerOneSessionError::QueryInputGeometry {
                batches: query_batches,
                hidden: query_hidden,
                expected_hidden: self.config.owner_layout.input_dimension().get(),
            });
        }
        let expected_input = call
            .positions
            .get()
            .checked_mul(self.config.owner_layout.input_dimension().get())
            .ok_or(LayerOneSessionError::PositionOverflow)?;
        if call.input.len() != expected_input {
            return Err(LayerOneSessionError::InputLength {
                actual: call.input.len(),
                expected: expected_input,
            });
        }
        let completed_group_end = end / SOURCE_RATIO;
        if completed_group_end > self.config.owner_layout.capacity().get() {
            return Err(LayerOneSessionError::OwnerCapacity {
                completed: completed_group_end,
                capacity: self.config.owner_layout.capacity().get(),
            });
        }
        let token_frequencies = frequency_span(
            call.frequencies,
            start,
            call.positions.get(),
            self.config.owner_layout.rope_pairs().get(),
            "token frequencies",
        )?;
        let old_key_count =
            self.owner.key_prefix(0)?.len() / self.config.owner_layout.key_dimension().get();
        let expected_old_key_count = start / SOURCE_RATIO;
        if old_key_count != expected_old_key_count {
            return Err(LayerOneSessionError::OwnerKeyCount {
                actual: old_key_count,
                expected: expected_old_key_count,
            });
        }
        let completed_frequencies = completed_frequency_rows(
            call.frequencies,
            expected_old_key_count,
            completed_group_end,
            self.config.owner_layout.rope_pairs().get(),
        )?;
        Ok(PreparedCall {
            start,
            token_frequencies,
            completed_frequencies,
            old_key_count,
            publication: IndexKeyPublicationId::new(
                self.config.source_layer,
                self.expected_epoch,
                self.owner.next_call_id(),
            ),
        })
    }

    fn score_prefix(
        &self,
        owner: &RatioTwoOwnerDiagnostic,
        start: usize,
        owned_key_count: usize,
        previous_layer_three: Option<PreviousLayerThreeKeys<'_>>,
    ) -> Result<Vec<u16>, LayerOneSessionError> {
        if owner.latent().is_some() {
            if previous_layer_three.is_some() {
                return Err(LayerOneSessionError::UnexpectedPreviousLayerThreePrefix);
            }
            return owned_prefix(self.owner.key_prefix(0)?, "owned L1 score prefix");
        }
        let previous =
            previous_layer_three.ok_or(LayerOneSessionError::MissingPreviousLayerThreePrefix)?;
        let expected_call_id = self
            .owner
            .next_call_id()
            .checked_sub(2)
            .ok_or(LayerOneSessionError::NoPreviousLayerThreeCall)?;
        if previous.publication.source_layer() != self.config.previous_owner_layer
            || previous.publication.epoch() != self.expected_epoch
            || previous.publication.call_id() != expected_call_id
        {
            return Err(LayerOneSessionError::PreviousLayerThreeIdentity {
                source_layer: previous.publication.source_layer(),
                epoch: previous.publication.epoch(),
                call_id: previous.publication.call_id(),
                expected_source_layer: self.config.previous_owner_layer,
                expected_epoch: self.expected_epoch,
                expected_call_id,
            });
        }
        let key_width = self.config.owner_layout.key_dimension().get();
        let expected_full = start
            .checked_mul(key_width)
            .ok_or(LayerOneSessionError::PrefixLengthOverflow)?;
        if previous.keys.len() != expected_full {
            return Err(LayerOneSessionError::PreviousLayerThreeLength {
                actual: previous.keys.len(),
                expected: expected_full,
            });
        }
        let _ = IndexKeyView::new(previous.keys, self.config.owner_layout.key_dimension())?;
        let used = owned_key_count
            .checked_mul(key_width)
            .ok_or(LayerOneSessionError::PrefixLengthOverflow)?;
        owned_prefix(&previous.keys[..used], "previous L3 score prefix")
    }

    /// Clears owner prefixes and attention state together, then admits epoch+1.
    pub fn reset(&mut self) -> Result<(), LayerOneSessionError> {
        self.lifecycle = Lifecycle::Poisoned;
        let next_epoch = self
            .expected_epoch
            .checked_add(1)
            .ok_or(LayerOneSessionError::EpochOverflow)?;
        let mut staged_attention = self.attention.clone();
        staged_attention.reset()?;
        self.owner.reset()?;
        if self.owner.epoch() != next_epoch {
            return Err(LayerOneSessionError::OwnerEpochMismatch {
                actual: self.owner.epoch(),
                expected: next_epoch,
            });
        }
        self.attention = staged_attention;
        self.expected_epoch = next_epoch;
        self.lifecycle = Lifecycle::Healthy;
        Ok(())
    }

    /// Returns the token start expected by the next successful call.
    #[must_use]
    pub const fn next_start(&self) -> usize {
        self.owner.next_position()
    }

    /// Returns whether reset is required after an admitted failure.
    #[must_use]
    pub const fn is_poisoned(&self) -> bool {
        matches!(self.lifecycle, Lifecycle::Poisoned)
    }
}

/// Errors from L1 session construction, composition, lifecycle, or reset.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum LayerOneSessionError {
    #[error("layer-one owner requires batch one, got {actual}")]
    OwnerBatchCount { actual: usize },
    #[error("layer-one attention requires batch one, got {actual}")]
    AttentionBatchCount { actual: usize },
    #[error("layer-one attention layout has no compressed source layer")]
    AttentionSourceLayer,
    #[error("layer-one attention layout does not use compression ratio two")]
    AttentionCompressionRatio,
    #[error(
        "layer-one owner input dimension {owner} differs from attention hidden dimension {attention}"
    )]
    InputDimensionMismatch { owner: usize, attention: usize },
    #[error(
        "layer-one owner latent dimension {owner} differs from attention head dimension {attention}"
    )]
    LatentDimensionMismatch { owner: usize, attention: usize },
    #[error("layer-one owner rope pairs {owner} differs from attention rope pairs {attention}")]
    RopePairMismatch { owner: usize, attention: usize },
    #[error("layer-one session is poisoned; reset is required")]
    Poisoned,
    #[error("layer-one owner epoch {actual} differs from expected {expected}")]
    OwnerEpochMismatch { actual: u64, expected: u64 },
    #[error("layer-one session epoch overflowed")]
    EpochOverflow,
    #[error("layer-one position arithmetic overflowed")]
    PositionOverflow,
    #[error("layer-one prefix length arithmetic overflowed")]
    PrefixLengthOverflow,
    #[error(
        "layer-one query geometry is batch {batches}, hidden {hidden}; expected batch one, hidden {expected_hidden}"
    )]
    QueryInputGeometry {
        batches: usize,
        hidden: usize,
        expected_hidden: usize,
    },
    #[error("layer-one input length is {actual}, expected {expected}")]
    InputLength { actual: usize, expected: usize },
    #[error("layer-one completed position {completed} exceeds owner capacity {capacity}")]
    OwnerCapacity { completed: usize, capacity: usize },
    #[error(
        "layer-one {field} needs {required} rotary-frequency elements, but the supplied table has {available}"
    )]
    FrequencyTableTooShort {
        field: &'static str,
        required: usize,
        available: usize,
    },
    #[error("layer-one owner prefix has {actual} keys, expected {expected}")]
    OwnerKeyCount { actual: usize, expected: usize },
    #[error("layer-one complete owner group must not receive a previous layer-three prefix")]
    UnexpectedPreviousLayerThreePrefix,
    #[error("layer-one incomplete owner group requires a previous layer-three prefix")]
    MissingPreviousLayerThreePrefix,
    #[error("layer-one incomplete owner group has no preceding layer-three call")]
    NoPreviousLayerThreeCall,
    #[error(
        "previous layer-three publication ({source_layer}, {epoch}, {call_id}) does not match source {expected_source_layer} epoch {expected_epoch} call {expected_call_id}"
    )]
    PreviousLayerThreeIdentity {
        source_layer: u16,
        epoch: u64,
        call_id: u64,
        expected_source_layer: u16,
        expected_epoch: u64,
        expected_call_id: u64,
    },
    #[error("previous layer-three key prefix length is {actual}, expected {expected}")]
    PreviousLayerThreeLength { actual: usize, expected: usize },
    #[error("could not allocate {elements} elements for {field}")]
    AllocationFailed {
        field: &'static str,
        elements: usize,
    },
    #[error("scored query returned {actual} scores, expected {expected}")]
    ScoreLength { actual: usize, expected: usize },
    #[error(
        "layer-one causal position {position} reaches {reachable} keys, but only {key_count} were scored"
    )]
    ReachableKeyCount {
        position: usize,
        reachable: usize,
        key_count: usize,
    },
    #[error(transparent)]
    Owner(#[from] RatioTwoOwnerError),
    #[error(transparent)]
    Prefix(#[from] IndexKeyStateError),
    #[error(transparent)]
    Score(#[from] ScoredQueryError),
    #[error(transparent)]
    Selection(#[from] SelectionError),
    #[error(transparent)]
    Attention(#[from] LayerAttentionError),
}

fn frequency_span<'a>(
    frequencies: &'a [RotaryFrequency],
    start: usize,
    positions: usize,
    rope_pairs: usize,
    field: &'static str,
) -> Result<&'a [RotaryFrequency], LayerOneSessionError> {
    let first = start
        .checked_mul(rope_pairs)
        .ok_or(LayerOneSessionError::PositionOverflow)?;
    let end_position = start
        .checked_add(positions)
        .ok_or(LayerOneSessionError::PositionOverflow)?;
    let end = end_position
        .checked_mul(rope_pairs)
        .ok_or(LayerOneSessionError::PositionOverflow)?;
    frequencies
        .get(first..end)
        .ok_or(LayerOneSessionError::FrequencyTableTooShort {
            field,
            required: end,
            available: frequencies.len(),
        })
}

fn completed_frequency_rows(
    frequencies: &[RotaryFrequency],
    group_start: usize,
    group_end: usize,
    rope_pairs: usize,
) -> Result<Vec<RotaryFrequency>, LayerOneSessionError> {
    let groups = group_end
        .checked_sub(group_start)
        .ok_or(LayerOneSessionError::PositionOverflow)?;
    let elements = groups
        .checked_mul(rope_pairs)
        .ok_or(LayerOneSessionError::PositionOverflow)?;
    let mut completed = Vec::new();
    completed
        .try_reserve_exact(elements)
        .map_err(|_| LayerOneSessionError::AllocationFailed {
            field: "completed-group frequencies",
            elements,
        })?;
    for group in group_start..group_end {
        let position = group
            .checked_mul(SOURCE_RATIO)
            .ok_or(LayerOneSessionError::PositionOverflow)?;
        completed.extend_from_slice(frequency_span(
            frequencies,
            position,
            1,
            rope_pairs,
            "completed-group frequencies",
        )?);
    }
    Ok(completed)
}

fn causal_select(
    scores: &[u16],
    start: usize,
    positions: usize,
    key_count: usize,
    index_topk: usize,
    offset: usize,
) -> Result<(Vec<u16>, Vec<i32>), LayerOneSessionError> {
    let expected = positions
        .checked_mul(key_count)
        .ok_or(LayerOneSessionError::PositionOverflow)?;
    if scores.len() != expected {
        return Err(LayerOneSessionError::ScoreLength {
            actual: scores.len(),
            expected,
        });
    }
    if key_count == 0 {
        return Err(LayerOneSessionError::ScoreLength {
            actual: scores.len(),
            expected: positions,
        });
    }
    let mut causal_scores = Vec::new();
    causal_scores.try_reserve_exact(scores.len()).map_err(|_| {
        LayerOneSessionError::AllocationFailed {
            field: "causal scores",
            elements: scores.len(),
        }
    })?;
    causal_scores.extend_from_slice(scores);
    let index_elements = positions
        .checked_mul(index_topk.min(key_count))
        .ok_or(LayerOneSessionError::PositionOverflow)?;
    let mut indices = Vec::new();
    indices.try_reserve_exact(index_elements).map_err(|_| {
        LayerOneSessionError::AllocationFailed {
            field: "selected indices",
            elements: index_elements,
        }
    })?;
    for position in 0..positions {
        let reachable = start
            .checked_add(position)
            .and_then(|value| value.checked_add(1))
            .ok_or(LayerOneSessionError::PositionOverflow)?
            / SOURCE_RATIO;
        if reachable > key_count {
            return Err(LayerOneSessionError::ReachableKeyCount {
                position,
                reachable,
                key_count,
            });
        }
        for score in
            &mut causal_scores[position * key_count + reachable..(position + 1) * key_count]
        {
            *score = NEGATIVE_INFINITY_BF16;
        }
        let mut row = Vec::new();
        row.try_reserve_exact(key_count)
            .map_err(|_| LayerOneSessionError::AllocationFailed {
                field: "selection score row",
                elements: key_count,
            })?;
        row.extend(
            causal_scores[position * key_count..(position + 1) * key_count]
                .iter()
                .copied()
                .map(bf16_to_f32),
        );
        indices.extend(select_indices(&row, reachable, index_topk, offset)?);
    }
    Ok((causal_scores, indices))
}

fn owned_prefix(values: &[u16], field: &'static str) -> Result<Vec<u16>, LayerOneSessionError> {
    let mut owned = Vec::new();
    owned
        .try_reserve_exact(values.len())
        .map_err(|_| LayerOneSessionError::AllocationFailed {
            field,
            elements: values.len(),
        })?;
    owned.extend_from_slice(values);
    Ok(owned)
}
