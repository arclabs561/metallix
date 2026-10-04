//! Request-local L4 query, selection, and attention over committed L3 data.

use std::num::NonZeroUsize;

use thiserror::Error;

use crate::{
    RotaryFrequency,
    attention::layer::{
        CompressedAttentionPublication, LayerAttentionDiagnostic, LayerAttentionError,
        LayerAttentionLayout, LayerAttentionState, LayerAttentionWeights,
    },
    indexer::{
        cache::IndexKeyPublicationId,
        query::{
            CandidateQueryLayout, CandidateQueryWeights, IndexKeyView, IndexScoreExecution,
            ScoredQueryDiagnostic, ScoredQueryError, prepare_scored_query_with_execution,
        },
        selection::{
            CandidateSelection, SelectionAdapterError, SelectionDiagnostic, select_from_candidates,
        },
    },
};

/// Fixed L4 query and attention geometry.
#[derive(Clone, Copy, Debug)]
pub struct LayerFourConfig {
    query_layout: CandidateQueryLayout,
    attention_layout: LayerAttentionLayout,
    index_topk: NonZeroUsize,
}

impl LayerFourConfig {
    /// Returns the fixed L4 compressed-attention geometry for request composition.
    #[must_use]
    pub(crate) const fn attention_layout(self) -> LayerAttentionLayout {
        self.attention_layout
    }

    /// Validates the L4 consumer accepts only batch-one L3 publications.
    pub fn new(
        query_layout: CandidateQueryLayout,
        attention_layout: LayerAttentionLayout,
        index_topk: NonZeroUsize,
    ) -> Result<Self, LayerFourSessionError> {
        let (batches, hidden) = query_layout.input_geometry();
        if batches != 1 || attention_layout.batches().get() != 1 {
            return Err(LayerFourSessionError::BatchCount {
                query: batches,
                attention: attention_layout.batches().get(),
            });
        }
        if hidden != attention_layout.hidden_dimension().get() {
            return Err(LayerFourSessionError::InputDimension {
                query: hidden,
                attention: attention_layout.hidden_dimension().get(),
            });
        }
        if attention_layout.compression().is_none() {
            return Err(LayerFourSessionError::AttentionSourceLayer);
        }
        if attention_layout.compression().map(|(_, ratio)| ratio.get()) != Some(1) {
            return Err(LayerFourSessionError::AttentionCompressionRatio);
        }
        Ok(Self {
            query_layout,
            attention_layout,
            index_topk,
        })
    }
}

/// One fully committed L3 publication borrowed by L4.
#[derive(Clone, Copy, Debug)]
pub struct LayerThreePublication<'a> {
    publication: IndexKeyPublicationId,
    keys: &'a [u16],
    kv: &'a [u16],
    candidates: &'a CandidateSelection,
}

impl<'a> LayerThreePublication<'a> {
    /// Groups one L3 identity, numerical prefixes, and its actual candidate mask.
    #[must_use]
    pub const fn new(
        publication: IndexKeyPublicationId,
        keys: &'a [u16],
        kv: &'a [u16],
        candidates: &'a CandidateSelection,
    ) -> Self {
        Self {
            publication,
            keys,
            kv,
            candidates,
        }
    }
}

/// Borrowed live L4 operands for one contiguous request partition.
#[derive(Clone, Copy, Debug)]
pub struct LayerFourCall<'a> {
    input: &'a [u16],
    frequencies: &'a [RotaryFrequency],
    query_weights: CandidateQueryWeights<'a>,
    attention_weights: LayerAttentionWeights<'a>,
    publication: LayerThreePublication<'a>,
}

impl<'a> LayerFourCall<'a> {
    /// Groups L4 numerical operands with a committed L3 publication.
    #[must_use]
    pub const fn new(
        input: &'a [u16],
        frequencies: &'a [RotaryFrequency],
        query_weights: CandidateQueryWeights<'a>,
        attention_weights: LayerAttentionWeights<'a>,
        publication: LayerThreePublication<'a>,
    ) -> Self {
        Self {
            input,
            frequencies,
            query_weights,
            attention_weights,
            publication,
        }
    }
}

/// L4 score, selected IDs, and attention stages from one committed L3 publication.
#[derive(Clone, Debug)]
pub struct LayerFourStepOutput {
    scored: ScoredQueryDiagnostic,
    selection: SelectionDiagnostic,
    attention: LayerAttentionDiagnostic,
}

impl LayerFourStepOutput {
    /// Returns L4's query and score-preparation stages.
    #[must_use]
    pub const fn scored(&self) -> &ScoredQueryDiagnostic {
        &self.scored
    }
    /// Returns causal candidate filtering and selected sparse IDs.
    #[must_use]
    pub const fn selection(&self) -> &SelectionDiagnostic {
        &self.selection
    }
    /// Returns all L4 attention stages.
    #[must_use]
    pub const fn attention(&self) -> &LayerAttentionDiagnostic {
        &self.attention
    }
}

/// Persistent L4 attention state. An admitted failure poisons the request.
pub struct LayerFourSession {
    config: LayerFourConfig,
    score_execution: IndexScoreExecution,
    attention: LayerAttentionState,
    next_start: usize,
    poisoned: bool,
}

impl LayerFourSession {
    /// Allocates an empty L4 attention ring for one request.
    #[must_use]
    pub fn new(config: LayerFourConfig) -> Self {
        Self {
            attention: LayerAttentionState::new(config.attention_layout),
            config,
            score_execution: IndexScoreExecution::Scalar,
            next_start: 0,
            poisoned: false,
        }
    }

    /// Selects the score-stage implementation for subsequent calls.
    ///
    /// Construction remains scalar by default so direct users retain the
    /// source-authoritative path. This model-local choice has no effect on
    /// attention state or reset behavior.
    #[must_use]
    pub const fn with_score_execution(mut self, score_execution: IndexScoreExecution) -> Self {
        self.score_execution = score_execution;
        self
    }

    /// Computes one L4 partition from a committed L3 publication.
    pub fn step(
        &mut self,
        call: LayerFourCall<'_>,
    ) -> Result<LayerFourStepOutput, LayerFourSessionError> {
        if self.poisoned {
            return Err(LayerFourSessionError::Poisoned);
        }
        self.poisoned = true;
        let result = self.step_admitted(&call);
        if result.is_ok() {
            self.poisoned = false;
        }
        result
    }

    /// Accepts only publications from the layer this session's attention consumes.
    fn check_publication_source(&self, actual: u16) -> Result<(), LayerFourSessionError> {
        let expected = self
            .config
            .attention_layout
            .compression()
            .map(|(source, _)| source);
        if expected == Some(actual) {
            Ok(())
        } else {
            Err(LayerFourSessionError::PublicationSource { actual })
        }
    }

    fn step_admitted(
        &mut self,
        call: &LayerFourCall<'_>,
    ) -> Result<LayerFourStepOutput, LayerFourSessionError> {
        let selection_call = call.publication.candidates.call();
        self.check_publication_source(call.publication.publication.source_layer())?;
        if selection_call.publication() != call.publication.publication {
            return Err(LayerFourSessionError::CandidatePublicationMismatch);
        }
        if selection_call.batch_index() != 0 {
            return Err(LayerFourSessionError::CandidateBatch {
                actual: selection_call.batch_index(),
            });
        }
        if selection_call.compression_ratio().get() != 1 {
            return Err(LayerFourSessionError::CandidateCompressionRatio {
                actual: selection_call.compression_ratio().get(),
            });
        }
        if selection_call.token_start() != self.next_start {
            return Err(LayerFourSessionError::UnexpectedStart {
                actual: selection_call.token_start(),
                expected: self.next_start,
            });
        }
        let positions = selection_call.positions();
        let expected_input = positions
            .get()
            .checked_mul(self.config.attention_layout.hidden_dimension().get())
            .ok_or(LayerFourSessionError::InputLengthOverflow)?;
        if call.input.len() != expected_input {
            return Err(LayerFourSessionError::InputLength {
                actual: call.input.len(),
                expected: expected_input,
            });
        }
        let keys = IndexKeyView::new(
            call.publication.keys,
            self.config.attention_layout.head_dimension(),
        )?;
        let (_, candidate_keys) = selection_call.score_shape();
        let key_count =
            call.publication.keys.len() / self.config.attention_layout.head_dimension().get();
        if candidate_keys != key_count {
            return Err(LayerFourSessionError::CandidateKeyCount {
                actual: candidate_keys,
                expected: key_count,
            });
        }
        let expected_offset = if self.next_start == 0 {
            positions.get()
        } else {
            self.config.attention_layout.window().get()
        };
        if selection_call.offset() != expected_offset {
            return Err(LayerFourSessionError::CandidateOffset {
                actual: selection_call.offset(),
                expected: expected_offset,
            });
        }
        let scored = prepare_scored_query_with_execution(
            call.input,
            call.frequencies,
            call.query_weights,
            self.config.query_layout,
            keys,
            self.score_execution,
        )?;
        let selection = select_from_candidates(
            &scored.scores,
            selection_call,
            call.publication.candidates,
            self.config.index_topk.get(),
        )?;
        let attention = self.attention.forward(
            call.input,
            self.next_start,
            call.frequencies,
            call.attention_weights,
            CompressedAttentionPublication {
                source_layer: call.publication.publication.source_layer(),
                epoch: call.publication.publication.epoch(),
                call_id: call.publication.publication.call_id(),
                numerical_bf16: call.publication.kv,
                indices: &selection.indices,
            },
        )?;
        self.next_start = self
            .next_start
            .checked_add(positions.get())
            .ok_or(LayerFourSessionError::PositionOverflow)?;
        Ok(LayerFourStepOutput {
            scored,
            selection,
            attention,
        })
    }

    /// Clears the L4 cursor and attention ring after a failed or completed request.
    pub fn reset(&mut self) -> Result<(), LayerFourSessionError> {
        self.poisoned = true;
        self.attention.reset()?;
        self.next_start = 0;
        self.poisoned = false;
        Ok(())
    }

    #[must_use]
    /// Returns the token start required for the next successful call.
    pub const fn next_start(&self) -> usize {
        self.next_start
    }
    #[must_use]
    /// Returns whether a failed admitted call requires reset.
    pub const fn is_poisoned(&self) -> bool {
        self.poisoned
    }
}

#[derive(Debug, Error)]
#[non_exhaustive]
pub enum LayerFourSessionError {
    #[error("layer-four requires query/attention batch one, got {query}/{attention}")]
    BatchCount { query: usize, attention: usize },
    #[error("layer-four query width {query} differs from attention width {attention}")]
    InputDimension { query: usize, attention: usize },
    #[error("layer-four attention layout has no compressed source layer")]
    AttentionSourceLayer,
    #[error("layer-four attention compression ratio is not one")]
    AttentionCompressionRatio,
    #[error("layer-four session is poisoned; reset is required")]
    Poisoned,
    #[error("layer-four publication source {actual} differs from its attention layout")]
    PublicationSource { actual: u16 },
    #[error("layer-four candidate mask identity differs from its L3 publication")]
    CandidatePublicationMismatch,
    #[error("layer-four candidate batch {actual} is not batch zero")]
    CandidateBatch { actual: usize },
    #[error("layer-four candidate compression ratio {actual} is not one")]
    CandidateCompressionRatio { actual: usize },
    #[error(
        "layer-four candidate key count {actual} differs from supplied L3 key count {expected}"
    )]
    CandidateKeyCount { actual: usize, expected: usize },
    #[error("layer-four candidate offset {actual} differs from source window offset {expected}")]
    CandidateOffset { actual: usize, expected: usize },
    #[error("layer-four call start {actual} differs from expected {expected}")]
    UnexpectedStart { actual: usize, expected: usize },
    #[error("layer-four input length overflowed")]
    InputLengthOverflow,
    #[error("layer-four input length {actual} differs from expected {expected}")]
    InputLength { actual: usize, expected: usize },
    #[error("layer-four token position overflowed")]
    PositionOverflow,
    #[error(transparent)]
    Key(#[from] ScoredQueryError),
    #[error(transparent)]
    Selection(#[from] SelectionAdapterError),
    #[error(transparent)]
    Attention(#[from] LayerAttentionError),
}
