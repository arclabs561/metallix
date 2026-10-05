//! Immutable layer and block definitions composed into a request schedule.

use crate::{
    RotaryFrequency,
    attention::layer::{LayerAttentionLayout, LayerAttentionWeights},
    indexer::{
        key::IndexKeyPreparationExecution,
        query::{CandidateQueryLayout, CandidateQueryWeights, IndexScoreExecution},
    },
    reduced::{
        AttentionInput, BlockTailReference, EngramSession, EngramSessionConfig,
        EngramSessionWeights, LayerFourConfig, LayerFourSession, LayerOneConfig, LayerOneSession,
        LayerThreeConfig, LayerThreeSession, RatioTwoOwnerWeights, StartupSession,
    },
};

use super::RequestError;

/// Immutable runtime operands for the startup block.
#[derive(Clone, Copy, Debug)]
pub struct StartupDefinition<'a> {
    /// The dense table, or `None` for rows read per step from a source.
    table: Option<&'a [u16]>,
    vocabulary: usize,
    pub(super) norm: &'a [u16],
    epsilon: f32,
    pub(super) attention_layout: LayerAttentionLayout,
    attention_weights: LayerAttentionWeights<'a>,
    pub(super) tail: BlockTailReference<'a>,
    pub(super) frequencies: &'a [RotaryFrequency],
}

impl<'a> StartupDefinition<'a> {
    /// Groups startup lookup, normalization, attention, and tail operands.
    #[must_use]
    pub const fn new(
        table: &'a [u16],
        norm: &'a [u16],
        epsilon: f32,
        attention_layout: LayerAttentionLayout,
        attention_weights: LayerAttentionWeights<'a>,
        tail: BlockTailReference<'a>,
        frequencies: &'a [RotaryFrequency],
    ) -> Self {
        Self {
            table: Some(table),
            vocabulary: 0,
            norm,
            epsilon,
            attention_layout,
            attention_weights,
            tail,
            frequencies,
        }
    }

    /// Like [`Self::new`] over a `vocabulary`-row embedding table whose rows
    /// each step reads from [`super::StepSources::embedding_rows`].
    #[must_use]
    pub const fn with_row_source(
        vocabulary: usize,
        norm: &'a [u16],
        epsilon: f32,
        attention_layout: LayerAttentionLayout,
        attention_weights: LayerAttentionWeights<'a>,
        tail: BlockTailReference<'a>,
        frequencies: &'a [RotaryFrequency],
    ) -> Self {
        Self {
            table: None,
            vocabulary,
            norm,
            epsilon,
            attention_layout,
            attention_weights,
            tail,
            frequencies,
        }
    }

    /// This startup with its table replaced by per-step rows from a source.
    #[cfg(test)]
    pub(crate) const fn reading_rows(mut self, vocabulary: usize) -> Self {
        self.table = None;
        self.vocabulary = vocabulary;
        self
    }

    pub(super) fn session(self) -> Result<StartupSession<'a>, RequestError> {
        Ok(match self.table {
            Some(table) => StartupSession::new(
                table,
                self.norm,
                self.epsilon,
                self.attention_layout,
                self.attention_weights,
                self.tail,
            )?,
            None => StartupSession::with_row_source(
                self.vocabulary,
                self.norm,
                self.epsilon,
                self.attention_layout,
                self.attention_weights,
                self.tail,
            )?,
        })
    }
}

/// Immutable HC input and attention-to-FFN tail for one numbered block.
#[derive(Clone, Copy, Debug)]
pub struct BlockDefinition<'a> {
    pub(super) input: AttentionInput<'a>,
    pub(super) tail: BlockTailReference<'a>,
}

impl<'a> BlockDefinition<'a> {
    /// Groups one numbered block's input normalization and tail operands.
    #[must_use]
    pub const fn new(input: AttentionInput<'a>, tail: BlockTailReference<'a>) -> Self {
        Self { input, tail }
    }

    pub(super) fn geometry(self) -> (usize, usize) {
        self.tail.geometry()
    }
}

/// Owned immutable operands used to reconstruct one request-local Engram state.
#[derive(Clone, Debug)]
pub struct EngramDefinition {
    config: EngramSessionConfig,
    weights: EngramSessionWeights,
}

impl EngramDefinition {
    /// Takes owned Engram configuration and numerical weights for fresh sessions.
    #[must_use]
    pub const fn new(config: EngramSessionConfig, weights: EngramSessionWeights) -> Self {
        Self { config, weights }
    }

    pub(super) fn session(&self) -> Result<EngramSession, RequestError> {
        Ok(EngramSession::new(
            self.config.clone(),
            self.weights.clone(),
        )?)
    }
}

/// Immutable borrowed L1 owner/query/attention operands.
#[derive(Clone, Copy, Debug)]
pub struct LayerOneDefinition<'a> {
    pub(super) config: LayerOneConfig,
    compressor_norm: &'a [u16],
    pub(super) owner_weights: RatioTwoOwnerWeights<'a>,
    pub(super) query_weights: CandidateQueryWeights<'a>,
    pub(super) query_layout: CandidateQueryLayout,
    pub(super) attention_weights: LayerAttentionWeights<'a>,
}

impl<'a> LayerOneDefinition<'a> {
    /// Groups borrowed L1 owner, direct-query, and attention operands.
    #[must_use]
    pub const fn new(
        config: LayerOneConfig,
        compressor_norm: &'a [u16],
        owner_weights: RatioTwoOwnerWeights<'a>,
        query_weights: CandidateQueryWeights<'a>,
        query_layout: CandidateQueryLayout,
        attention_weights: LayerAttentionWeights<'a>,
    ) -> Self {
        Self {
            config,
            compressor_norm,
            owner_weights,
            query_weights,
            query_layout,
            attention_weights,
        }
    }

    pub(super) fn session(
        self,
        score_execution: IndexScoreExecution,
        key_preparation_execution: IndexKeyPreparationExecution,
    ) -> Result<LayerOneSession, RequestError> {
        Ok(LayerOneSession::new(self.config, self.compressor_norm)?
            .with_score_execution(score_execution)
            .with_key_preparation_execution(key_preparation_execution))
    }
}

/// Immutable borrowed L3 owner/candidate/attention operands.
#[derive(Clone, Copy, Debug)]
pub struct LayerThreeDefinition<'a> {
    pub(super) config: LayerThreeConfig,
    compressor_norm: &'a [u16],
    compressor_epsilon: f32,
    pub(super) owner_weights: crate::indexer::owner::RatioOneOwnerWeights<'a>,
    pub(super) candidate: crate::reduced::CandidateProjector<'a>,
    pub(super) attention_weights: LayerAttentionWeights<'a>,
}

impl<'a> LayerThreeDefinition<'a> {
    /// Groups borrowed L3 owner, candidate, and attention operands.
    #[must_use]
    pub const fn new(
        config: LayerThreeConfig,
        compressor_norm: &'a [u16],
        compressor_epsilon: f32,
        owner_weights: crate::indexer::owner::RatioOneOwnerWeights<'a>,
        candidate: crate::reduced::CandidateProjector<'a>,
        attention_weights: LayerAttentionWeights<'a>,
    ) -> Self {
        Self {
            config,
            compressor_norm,
            compressor_epsilon,
            owner_weights,
            candidate,
            attention_weights,
        }
    }

    pub(super) fn session(
        self,
        key_preparation_execution: IndexKeyPreparationExecution,
    ) -> Result<LayerThreeSession, RequestError> {
        let source_layer = self
            .config
            .attention_layout()
            .compression()
            .map_or(0, |(source, _)| source);
        Ok(LayerThreeSession::new(
            self.config,
            source_layer,
            self.compressor_norm,
            self.compressor_epsilon,
        )?
        .with_key_preparation_execution(key_preparation_execution))
    }
}

/// Immutable borrowed L4 consumer operands.
#[derive(Clone, Copy, Debug)]
pub struct LayerFourDefinition<'a> {
    config: LayerFourConfig,
    pub(super) query_weights: CandidateQueryWeights<'a>,
    pub(super) attention_weights: LayerAttentionWeights<'a>,
}

impl<'a> LayerFourDefinition<'a> {
    /// Groups borrowed L4 query-selection and attention operands.
    #[must_use]
    pub const fn new(
        config: LayerFourConfig,
        query_weights: CandidateQueryWeights<'a>,
        attention_weights: LayerAttentionWeights<'a>,
    ) -> Self {
        Self {
            config,
            query_weights,
            attention_weights,
        }
    }

    pub(super) fn session(self, score_execution: IndexScoreExecution) -> LayerFourSession {
        LayerFourSession::new(self.config).with_score_execution(score_execution)
    }
}

/// Immutable attention operands for a layer which owns no compressed state.
///
/// A compressed layout makes this a consumer of the latest publication from
/// its layout's source layer; a window-only layout makes it a window-only layer.
#[derive(Clone, Copy, Debug)]
pub struct ReusedAttentionDefinition<'a> {
    pub(super) layout: LayerAttentionLayout,
    pub(super) weights: LayerAttentionWeights<'a>,
}

impl<'a> ReusedAttentionDefinition<'a> {
    /// Groups one non-owning layer's attention layout and weights.
    #[must_use]
    pub const fn new(layout: LayerAttentionLayout, weights: LayerAttentionWeights<'a>) -> Self {
        Self { layout, weights }
    }
}

/// The attention role of one scheduled layer.
#[derive(Clone, Copy, Debug)]
#[non_exhaustive]
pub enum LayerKind<'a> {
    /// Sliding-window attention only (`compress_ratio` 0 after startup).
    WindowOnly(ReusedAttentionDefinition<'a>),
    /// Ratio-two compressed-KV owner with its own direct indexer.
    RatioTwoOwner(LayerOneDefinition<'a>),
    /// Ratio-two attention over the latest ratio-two owner's KV and indices.
    RatioTwoConsumer(ReusedAttentionDefinition<'a>),
    /// Ratio-one compressed-KV owner and candidate-block source.
    RatioOneOwner(LayerThreeDefinition<'a>),
    /// Ratio-one indexer scoring inside the owner's candidate blocks.
    RatioOneIndexer(LayerFourDefinition<'a>),
    /// Ratio-one attention over the owner's KV and the latest ratio-one indices.
    RatioOneConsumer(ReusedAttentionDefinition<'a>),
}

impl LayerKind<'_> {
    pub(super) fn attention_layout(&self) -> LayerAttentionLayout {
        match self {
            Self::WindowOnly(definition)
            | Self::RatioTwoConsumer(definition)
            | Self::RatioOneConsumer(definition) => definition.layout,
            Self::RatioTwoOwner(definition) => definition.config.attention_layout(),
            Self::RatioOneOwner(definition) => definition.config.attention_layout(),
            Self::RatioOneIndexer(definition) => definition.config.attention_layout(),
        }
    }

    pub(super) const fn ratio(&self) -> usize {
        match self {
            Self::WindowOnly(_) => 0,
            Self::RatioTwoOwner(_) | Self::RatioTwoConsumer(_) => 2,
            Self::RatioOneOwner(_) | Self::RatioOneIndexer(_) | Self::RatioOneConsumer(_) => 1,
        }
    }
}

/// One numbered layer after startup: its kind, block operands, and optional Engram.
#[derive(Clone, Copy, Debug)]
pub struct ScheduledLayer<'a> {
    pub(super) kind: LayerKind<'a>,
    pub(super) block: BlockDefinition<'a>,
    pub(super) engram: Option<usize>,
}

impl<'a> ScheduledLayer<'a> {
    /// Groups one layer's attention kind with its HC input and tail operands.
    #[must_use]
    pub const fn new(kind: LayerKind<'a>, block: BlockDefinition<'a>) -> Self {
        Self {
            kind,
            block,
            engram: None,
        }
    }

    /// Applies the model's `engram`-th Engram definition before this block.
    #[must_use]
    pub const fn with_engram(mut self, engram: usize) -> Self {
        self.engram = Some(engram);
        self
    }
}
