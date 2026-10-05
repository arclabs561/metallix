//! V4.1-shaped request composition over live numerical operands.
//!
//! A request is startup, a validated list of typed layers, and the final head.
//! Each layer is one of six fixed kinds (window-only, ratio-two owner or
//! consumer, ratio-one owner, candidate indexer or consumer) with an optional
//! Engram before its block. [`RequestSession::step`] walks that list in one
//! visible loop; consumers borrow the latest same-step publication of the
//! producer their attention layout names. The reduced five-block fixture and
//! the 40-layer checkpoint schedule are two lists of the same kinds. This
//! module has no checkpoint loader, fixture decoder, callback graph, or
//! general graph interpreter.

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
        key::{IndexKeyPreparationExecution, IndexKeyRotaryExecution},
        query::{CandidateQueryLayout, CandidateQueryWeights, IndexScoreExecution},
    },
};

use super::{
    AttentionInput, AttentionInputError, AttentionInputOutput, BlockTailDiagnostic, BlockTailError,
    BlockTailReference, EngramSession, EngramSessionConfig, EngramSessionError,
    EngramSessionWeights, EngramStepOutput, FinalHead, FinalHeadError, FinalHeadExecution,
    FinalHeadOutput, LayerFourCall, LayerFourConfig, LayerFourSession, LayerFourSessionError,
    LayerFourStepOutput, LayerOneCall, LayerOneConfig, LayerOneSession, LayerOneSessionError,
    LayerOneStepOutput, LayerThreeCall, LayerThreeConfig, LayerThreePublication, LayerThreeSession,
    LayerThreeSessionError, LayerThreeStepOutput, PreviousLayerThreeKeys, RatioTwoOwnerWeights,
    StartupSession, StartupSessionError, StartupStepOutput,
};

const MAX_REQUEST_ELEMENTS: usize = 1 << 20;

/// Immutable runtime operands for the startup block.
#[derive(Clone, Copy, Debug)]
pub struct StartupDefinition<'a> {
    table: &'a [u16],
    norm: &'a [u16],
    epsilon: f32,
    attention_layout: LayerAttentionLayout,
    attention_weights: LayerAttentionWeights<'a>,
    tail: BlockTailReference<'a>,
    frequencies: &'a [RotaryFrequency],
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
            table,
            norm,
            epsilon,
            attention_layout,
            attention_weights,
            tail,
            frequencies,
        }
    }

    fn session(self) -> Result<StartupSession<'a>, RequestError> {
        Ok(StartupSession::new(
            self.table,
            self.norm,
            self.epsilon,
            self.attention_layout,
            self.attention_weights,
            self.tail,
        )?)
    }
}

/// Immutable HC input and attention-to-FFN tail for one numbered block.
#[derive(Clone, Copy, Debug)]
pub struct BlockDefinition<'a> {
    input: AttentionInput<'a>,
    tail: BlockTailReference<'a>,
}

impl<'a> BlockDefinition<'a> {
    /// Groups one numbered block's input normalization and tail operands.
    #[must_use]
    pub const fn new(input: AttentionInput<'a>, tail: BlockTailReference<'a>) -> Self {
        Self { input, tail }
    }

    fn geometry(self) -> (usize, usize) {
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

    fn session(&self) -> Result<EngramSession, RequestError> {
        Ok(EngramSession::new(
            self.config.clone(),
            self.weights.clone(),
        )?)
    }
}

/// Immutable borrowed L1 owner/query/attention operands.
#[derive(Clone, Copy, Debug)]
pub struct LayerOneDefinition<'a> {
    config: LayerOneConfig,
    compressor_norm: &'a [u16],
    owner_weights: RatioTwoOwnerWeights<'a>,
    query_weights: CandidateQueryWeights<'a>,
    query_layout: CandidateQueryLayout,
    attention_weights: LayerAttentionWeights<'a>,
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

    fn session(
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
    config: LayerThreeConfig,
    compressor_norm: &'a [u16],
    compressor_epsilon: f32,
    owner_weights: crate::indexer::owner::RatioOneOwnerWeights<'a>,
    candidate: super::CandidateProjector<'a>,
    attention_weights: LayerAttentionWeights<'a>,
}

impl<'a> LayerThreeDefinition<'a> {
    /// Groups borrowed L3 owner, candidate, and attention operands.
    #[must_use]
    pub const fn new(
        config: LayerThreeConfig,
        compressor_norm: &'a [u16],
        compressor_epsilon: f32,
        owner_weights: crate::indexer::owner::RatioOneOwnerWeights<'a>,
        candidate: super::CandidateProjector<'a>,
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

    fn session(
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
    query_weights: CandidateQueryWeights<'a>,
    attention_weights: LayerAttentionWeights<'a>,
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

    fn session(self, score_execution: IndexScoreExecution) -> LayerFourSession {
        LayerFourSession::new(self.config).with_score_execution(score_execution)
    }
}

/// Immutable attention operands for a layer which owns no compressed state.
///
/// A compressed layout makes this a consumer of the latest publication from
/// its layout's source layer; a window-only layout makes it a window-only layer.
#[derive(Clone, Copy, Debug)]
pub struct ReusedAttentionDefinition<'a> {
    layout: LayerAttentionLayout,
    weights: LayerAttentionWeights<'a>,
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
    fn attention_layout(&self) -> LayerAttentionLayout {
        match self {
            Self::WindowOnly(definition)
            | Self::RatioTwoConsumer(definition)
            | Self::RatioOneConsumer(definition) => definition.layout,
            Self::RatioTwoOwner(definition) => definition.config.attention_layout(),
            Self::RatioOneOwner(definition) => definition.config.attention_layout(),
            Self::RatioOneIndexer(definition) => definition.config.attention_layout(),
        }
    }

    const fn ratio(&self) -> usize {
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
    kind: LayerKind<'a>,
    block: BlockDefinition<'a>,
    engram: Option<usize>,
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

/// One checkpoint-defined runtime model: startup, scheduled layers, and head.
#[derive(Clone, Debug)]
pub struct RequestModel<'a> {
    startup: StartupDefinition<'a>,
    layers: Vec<ScheduledLayer<'a>>,
    engrams: Vec<EngramDefinition>,
    head: FinalHead<'a>,
    frequencies: &'a [RotaryFrequency],
    max_tokens: NonZeroUsize,
    score_execution: IndexScoreExecution,
    key_preparation_execution: IndexKeyPreparationExecution,
}

impl<'a> RequestModel<'a> {
    /// Validates immutable operands for the reduced five-block schedule.
    ///
    /// This is [`Self::from_schedule`] over startup, Engram, L1 ratio-two
    /// owner, L2 ratio-two consumer, Engram, L3 ratio-one owner, and L4
    /// candidate indexer.
    #[allow(
        clippy::too_many_arguments,
        reason = "the fixed numbered block definitions are explicit"
    )]
    pub fn new(
        startup: StartupDefinition<'a>,
        blocks: [BlockDefinition<'a>; 4],
        engrams: [EngramDefinition; 2],
        layer_one: LayerOneDefinition<'a>,
        layer_two: ReusedAttentionDefinition<'a>,
        layer_three: LayerThreeDefinition<'a>,
        layer_four: LayerFourDefinition<'a>,
        head: FinalHead<'a>,
        frequencies: &'a [RotaryFrequency],
        max_tokens: NonZeroUsize,
    ) -> Result<Self, RequestError> {
        let compression = |layout: LayerAttentionLayout| {
            layout
                .compression()
                .map(|(source, ratio)| (source, ratio.get()))
        };
        if compression(layer_two.layout) != compression(layer_one.config.attention_layout()) {
            return Err(RequestError::LayerTwoGeometry);
        }
        let [one, two, three, four] = blocks;
        Self::from_schedule(
            startup,
            vec![
                ScheduledLayer::new(LayerKind::RatioTwoOwner(layer_one), one).with_engram(0),
                ScheduledLayer::new(LayerKind::RatioTwoConsumer(layer_two), two),
                ScheduledLayer::new(LayerKind::RatioOneOwner(layer_three), three).with_engram(1),
                ScheduledLayer::new(LayerKind::RatioOneIndexer(layer_four), four),
            ],
            engrams.into(),
            head,
            frequencies,
            max_tokens,
        )
    }

    /// Validates a layer schedule before any request state is allocated.
    ///
    /// `layers[i]` is model layer `i + 1`. An owner must publish under its own
    /// layer number; a consumer or indexer needs the latest preceding owner
    /// of its ratio to be the source its attention layout names; ratio-two
    /// layers precede ratio-one layers, and a ratio-two owner needs a
    /// ratio-one owner whose previous-step keys score incomplete groups.
    /// Window-only layers use the startup rotary table; every compressed
    /// layer uses `frequencies`.
    pub fn from_schedule(
        startup: StartupDefinition<'a>,
        layers: Vec<ScheduledLayer<'a>>,
        engrams: Vec<EngramDefinition>,
        head: FinalHead<'a>,
        frequencies: &'a [RotaryFrequency],
        max_tokens: NonZeroUsize,
    ) -> Result<Self, RequestError> {
        let mut layers = layers;
        // Incomplete ratio-two groups score the last ratio-one owner's keys.
        let previous_owner = layers.iter().rev().find_map(|layer| match layer.kind {
            LayerKind::RatioOneOwner(owner) => owner
                .config
                .attention_layout()
                .compression()
                .map(|(source, _)| source),
            _ => None,
        });
        if let Some(previous_owner) = previous_owner {
            for layer in &mut layers {
                if let LayerKind::RatioTwoOwner(owner) = &mut layer.kind {
                    owner.config = owner.config.with_previous_owner_layer(previous_owner);
                }
            }
        }
        let model = Self {
            startup,
            layers,
            engrams,
            head,
            frequencies,
            max_tokens,
            score_execution: IndexScoreExecution::Scalar,
            key_preparation_execution: IndexKeyPreparationExecution::Scalar,
        };
        model.validate()?;
        Ok(model)
    }

    /// Returns the scheduled layers after startup, in execution order.
    #[must_use]
    pub fn layers(&self) -> &[ScheduledLayer<'a>] {
        &self.layers
    }

    /// Selects the index-score implementation for every indexed request layer.
    ///
    /// [`Self::new`] preserves scalar BF16 staging. The alternate choice is a
    /// bounded `DeepSeek` diagnostic and is rebuilt unchanged on restart.
    #[must_use]
    pub const fn with_score_execution(mut self, score_execution: IndexScoreExecution) -> Self {
        self.score_execution = score_execution;
        self
    }

    /// Returns the model-local score-stage implementation.
    #[must_use]
    pub const fn score_execution(&self) -> IndexScoreExecution {
        self.score_execution
    }

    /// Selects the complete index-key preparation implementation for every owner.
    #[must_use]
    pub const fn with_key_preparation_execution(
        mut self,
        key_preparation_execution: IndexKeyPreparationExecution,
    ) -> Self {
        self.key_preparation_execution = key_preparation_execution;
        self
    }

    /// Selects the legacy rotary-only implementation through its preparation mapping.
    #[must_use]
    pub const fn with_key_rotary_execution(
        self,
        key_rotary_execution: IndexKeyRotaryExecution,
    ) -> Self {
        match key_rotary_execution {
            IndexKeyRotaryExecution::Scalar => {
                self.with_key_preparation_execution(IndexKeyPreparationExecution::Scalar)
            }
            #[cfg(feature = "metal")]
            IndexKeyRotaryExecution::MetalFp32 => {
                self.with_key_preparation_execution(IndexKeyPreparationExecution::MetalRotaryFp32)
            }
        }
    }

    /// Returns the model-local complete index-key preparation implementation.
    #[must_use]
    pub const fn key_preparation_execution(&self) -> IndexKeyPreparationExecution {
        self.key_preparation_execution
    }

    /// Returns the rotary component of the model-local key preparation implementation.
    ///
    /// [`Self::key_preparation_execution`] remains authoritative because it
    /// also distinguishes projection and normalization placement.
    #[must_use]
    pub const fn key_rotary_execution(&self) -> IndexKeyRotaryExecution {
        match self.key_preparation_execution {
            IndexKeyPreparationExecution::Scalar => IndexKeyRotaryExecution::Scalar,
            #[cfg(feature = "metal")]
            IndexKeyPreparationExecution::MetalRotaryFp32 => IndexKeyRotaryExecution::MetalFp32,
            #[cfg(feature = "metal")]
            IndexKeyPreparationExecution::MetalPreFp4 => IndexKeyRotaryExecution::MetalFp32,
        }
    }

    /// Selects the final vocabulary-projection implementation.
    ///
    /// [`Self::new`] preserves scalar FP32 projection. The selected immutable
    /// head is retained unchanged when a request restarts.
    #[must_use]
    pub const fn with_head_execution(mut self, execution: FinalHeadExecution) -> Self {
        self.head = self.head.with_execution(execution);
        self
    }

    /// Returns the model-local final vocabulary-projection implementation.
    #[must_use]
    pub const fn head_execution(&self) -> FinalHeadExecution {
        self.head.execution()
    }

    fn validate(&self) -> Result<(), RequestError> {
        let (copies, width) = self.startup.tail.geometry();
        if self.layers.iter().any(|layer| {
            layer.block.geometry() != (copies, width)
                || layer.block.input.geometry() != (copies, width)
        }) {
            return Err(RequestError::BlockGeometry);
        }
        if self.startup.norm.len() != width || self.head.geometry() != (copies, width) {
            return Err(RequestError::BlockGeometry);
        }
        let rope_pairs = self.startup.attention_layout.rope_pairs();
        if std::iter::once(self.startup.attention_layout)
            .chain(
                self.layers
                    .iter()
                    .map(|layer| layer.kind.attention_layout()),
            )
            .any(|layout| {
                layout.batches().get() != 1
                    || layout.hidden_dimension().get() != width
                    || layout.rope_pairs() != rope_pairs
            })
        {
            return Err(RequestError::AttentionGeometry);
        }
        self.validate_schedule()?;
        if self.max_tokens.get() > MAX_REQUEST_ELEMENTS {
            return Err(RequestError::ElementLimit {
                elements: self.max_tokens.get(),
            });
        }
        let required_frequencies = self
            .max_tokens
            .get()
            .checked_mul(rope_pairs.get())
            .ok_or(RequestError::PositionOverflow)?;
        for available in [self.frequencies.len(), self.startup.frequencies.len()] {
            if available < required_frequencies {
                return Err(RequestError::FrequencyTable {
                    required: required_frequencies,
                    available,
                });
            }
        }
        let _ = self.startup.session()?;
        for engram in &self.engrams {
            let _ = engram.session()?;
        }
        for layer in &self.layers {
            let _ = self.layer_state(layer)?;
        }
        Ok(())
    }

    fn validate_schedule(&self) -> Result<(), RequestError> {
        let mut latest_owner: Option<(u16, usize)> = None;
        let mut has_ratio_one_owner = false;
        let mut has_ratio_two_owner = false;
        for (index, layer) in self.layers.iter().enumerate() {
            let fail = |reason| RequestError::Schedule {
                layer: index + 1,
                reason,
            };
            let number = u16::try_from(index + 1).map_err(|_| fail(ScheduleError::LayerCount))?;
            if let Some(engram) = layer.engram.filter(|&engram| engram >= self.engrams.len()) {
                return Err(fail(ScheduleError::EngramIndex {
                    index: engram,
                    available: self.engrams.len(),
                }));
            }
            let compression = layer
                .kind
                .attention_layout()
                .compression()
                .map(|(source, ratio)| (source, ratio.get()));
            let ratio = layer.kind.ratio();
            if ratio == 0 {
                if compression.is_some() {
                    return Err(fail(ScheduleError::WindowLayoutCompressed));
                }
                continue;
            }
            if ratio == 2 && has_ratio_one_owner {
                return Err(fail(ScheduleError::RatioOrder));
            }
            let Some((source, layout_ratio)) = compression else {
                return Err(fail(ScheduleError::MissingCompression));
            };
            if layout_ratio != ratio {
                return Err(fail(ScheduleError::CompressionRatio {
                    expected: ratio,
                    actual: layout_ratio,
                }));
            }
            match layer.kind {
                LayerKind::RatioTwoOwner(_) | LayerKind::RatioOneOwner(_) => {
                    if source != number {
                        return Err(fail(ScheduleError::OwnerSource {
                            expected: number,
                            actual: source,
                        }));
                    }
                    latest_owner = Some((source, ratio));
                    has_ratio_two_owner |= ratio == 2;
                    has_ratio_one_owner |= ratio == 1;
                }
                _ => {
                    if latest_owner != Some((source, ratio)) {
                        return Err(fail(ScheduleError::MissingProducer {
                            source_layer: source,
                            ratio,
                        }));
                    }
                }
            }
        }
        if has_ratio_two_owner && !has_ratio_one_owner {
            return Err(RequestError::Schedule {
                layer: self.layers.len(),
                reason: ScheduleError::MissingRatioOneOwner,
            });
        }
        Ok(())
    }

    fn layer_state(&self, layer: &ScheduledLayer<'_>) -> Result<LayerState, RequestError> {
        Ok(match layer.kind {
            LayerKind::WindowOnly(definition) => {
                LayerState::WindowOnly(LayerAttentionState::new(definition.layout))
            }
            LayerKind::RatioTwoOwner(definition) => LayerState::RatioTwoOwner(
                definition.session(self.score_execution, self.key_preparation_execution)?,
            ),
            LayerKind::RatioTwoConsumer(definition) => {
                LayerState::RatioTwoConsumer(LayerAttentionState::new(definition.layout))
            }
            LayerKind::RatioOneOwner(definition) => {
                LayerState::RatioOneOwner(definition.session(self.key_preparation_execution)?)
            }
            LayerKind::RatioOneIndexer(definition) => {
                LayerState::RatioOneIndexer(definition.session(self.score_execution))
            }
            LayerKind::RatioOneConsumer(definition) => {
                LayerState::RatioOneConsumer(LayerAttentionState::new(definition.layout))
            }
        })
    }
}

/// Request-local state for one scheduled layer, matching its [`LayerKind`].
enum LayerState {
    WindowOnly(LayerAttentionState),
    RatioTwoOwner(LayerOneSession),
    RatioTwoConsumer(LayerAttentionState),
    RatioOneOwner(LayerThreeSession),
    RatioOneIndexer(LayerFourSession),
    RatioOneConsumer(LayerAttentionState),
}

/// The last ratio-one owner's keys from the previous successful step.
struct PriorRatioOneKeys {
    publication: IndexKeyPublicationId,
    keys: Vec<u16>,
}

/// Mutable request state for every scheduled layer, rebuilt as one unit on restart.
pub struct RequestSession<'a> {
    model: &'a RequestModel<'a>,
    startup: StartupSession<'a>,
    engrams: Vec<Option<EngramSession>>,
    layers: Vec<LayerState>,
    prior_ratio_one: Option<PriorRatioOneKeys>,
    next_start: usize,
    poisoned: bool,
}

impl<'a> RequestSession<'a> {
    /// Constructs all request-local owners from one immutable model definition.
    pub fn new(model: &'a RequestModel<'a>) -> Result<Self, RequestError> {
        Self::build(model)
    }

    fn build(model: &'a RequestModel<'a>) -> Result<Self, RequestError> {
        let startup = model.startup.session()?;
        let mut engrams = Vec::with_capacity(model.layers.len());
        let mut layers = Vec::with_capacity(model.layers.len());
        for layer in &model.layers {
            engrams.push(
                layer
                    .engram
                    .map(|index| model.engrams[index].session())
                    .transpose()?,
            );
            layers.push(model.layer_state(layer)?);
        }
        Ok(Self {
            model,
            startup,
            engrams,
            layers,
            prior_ratio_one: None,
            next_start: 0,
            poisoned: false,
        })
    }

    /// Returns the next absolute token position admitted by this request.
    #[must_use]
    pub const fn next_start(&self) -> usize {
        self.next_start
    }
    /// Reports whether a failed admitted stage requires [`Self::restart`].
    #[must_use]
    pub const fn is_poisoned(&self) -> bool {
        self.poisoned
    }

    /// Returns the score-stage implementation retained across request restart.
    #[must_use]
    pub const fn score_execution(&self) -> IndexScoreExecution {
        self.model.score_execution()
    }

    /// Returns the index-key rotary implementation retained across request restart.
    #[must_use]
    pub const fn key_rotary_execution(&self) -> IndexKeyRotaryExecution {
        self.model.key_rotary_execution()
    }

    /// Returns the complete index-key preparation implementation retained across restart.
    #[must_use]
    pub const fn key_preparation_execution(&self) -> IndexKeyPreparationExecution {
        self.model.key_preparation_execution()
    }

    /// Returns the final projection implementation retained across request restart.
    #[must_use]
    pub const fn head_execution(&self) -> FinalHeadExecution {
        self.model.head_execution()
    }

    /// Drops every request-local publication and reconstructs pristine inner state.
    pub fn restart(&mut self) -> Result<(), RequestError> {
        let rebuilt = Self::build(self.model)?;
        *self = rebuilt;
        Ok(())
    }

    /// Executes a prefill at start zero or one-token decode at the request cursor.
    pub fn step(&mut self, ids: &[i64]) -> Result<RequestStepOutput, RequestError> {
        if self.poisoned {
            return Err(RequestError::Poisoned);
        }
        if ids.is_empty() {
            return Err(RequestError::EmptyIds);
        }
        if self.next_start != 0 && ids.len() != 1 {
            return Err(RequestError::DecodeChunk { actual: ids.len() });
        }
        let end = self
            .next_start
            .checked_add(ids.len())
            .ok_or(RequestError::PositionOverflow)?;
        if end > self.model.max_tokens.get() {
            return Err(RequestError::TokenLimit {
                end,
                maximum: self.model.max_tokens.get(),
            });
        }
        self.poisoned = true;
        let result = self.step_admitted(ids, end);
        if result.is_ok() {
            self.poisoned = false;
        }
        result
    }

    #[allow(
        clippy::too_many_lines,
        reason = "the scheduled numerical order is intentionally visible in one request path"
    )]
    fn step_admitted(
        &mut self,
        ids: &[i64],
        end: usize,
    ) -> Result<RequestStepOutput, RequestError> {
        let start = self.next_start;
        let positions = NonZeroUsize::new(ids.len()).expect("nonempty ids");
        let rope_pairs = self.model.startup.attention_layout.rope_pairs().get();
        let frequencies = frequency_span(self.model.frequencies, start, ids.len(), rope_pairs)?;
        let startup_frequencies =
            frequency_span(self.model.startup.frequencies, start, ids.len(), rope_pairs)?;
        let startup_ids = ids_to_u64(ids)?;
        let startup = self
            .startup
            .step(start, &startup_ids, startup_frequencies)?;
        let mut residual = startup.residual().to_vec();
        let mut pre = startup.next_pre().to_vec();
        let partial_group = !completes_ratio_two_group(start, ids.len());
        let mut outputs: Vec<LayerStepOutput> = reserve(self.layers.len(), "layer outputs")?;
        // Indices into `outputs` of the latest owner and latest index publisher.
        let mut latest_owner = None;
        let mut latest_indices = None;
        for ((definition, state), engram) in self
            .model
            .layers
            .iter()
            .zip(&mut self.layers)
            .zip(&mut self.engrams)
        {
            let engram = engram
                .as_mut()
                .map(|engram| engram.step(start, ids, &residual))
                .transpose()?;
            let block_residual = engram.as_ref().map_or(&residual[..], |e| e.output());
            let attention_input = attention_inputs(&definition.block, block_residual, &pre)?;
            let input = normalized_rows(&attention_input)?;
            let attention = match (definition.kind, state) {
                (LayerKind::WindowOnly(layer), LayerState::WindowOnly(state)) => {
                    ScheduledAttentionOutput::WindowOnly(state.forward_window_only(
                        &input,
                        start,
                        startup_frequencies,
                        layer.weights,
                    )?)
                }
                (LayerKind::RatioTwoOwner(layer), LayerState::RatioTwoOwner(session)) => {
                    let prior = if partial_group {
                        let prior = self
                            .prior_ratio_one
                            .as_ref()
                            .ok_or(RequestError::MissingPriorLayerThree)?;
                        Some(PreviousLayerThreeKeys::new(prior.publication, &prior.keys))
                    } else {
                        None
                    };
                    ScheduledAttentionOutput::RatioTwoOwner(session.step(LayerOneCall::new(
                        &input,
                        positions,
                        self.model.frequencies,
                        layer.owner_weights,
                        layer.query_weights,
                        layer.query_layout,
                        layer.attention_weights,
                        prior,
                    ))?)
                }
                (LayerKind::RatioTwoConsumer(layer), LayerState::RatioTwoConsumer(state)) => {
                    let Some(ScheduledAttentionOutput::RatioTwoOwner(owner)) =
                        latest_owner.map(|index: usize| &outputs[index].attention)
                    else {
                        unreachable!("validated schedules precede consumers with their owner")
                    };
                    ScheduledAttentionOutput::RatioTwoConsumer(state.forward(
                        &input,
                        start,
                        frequencies,
                        layer.weights,
                        CompressedAttentionPublication {
                            source_layer: owner.publication().source_layer(),
                            epoch: owner.publication().epoch(),
                            call_id: owner.publication().call_id(),
                            numerical_bf16: owner.kv_prefix(),
                            indices: owner.selected_indices(),
                        },
                    )?)
                }
                (LayerKind::RatioOneOwner(layer), LayerState::RatioOneOwner(session)) => {
                    ScheduledAttentionOutput::RatioOneOwner(
                        session.step(LayerThreeCall::new(
                            &input,
                            positions,
                            frequencies,
                            layer.owner_weights,
                            layer
                                .candidate
                                .with_score_execution(self.model.score_execution),
                            layer.attention_weights,
                        ))?,
                    )
                }
                (LayerKind::RatioOneIndexer(layer), LayerState::RatioOneIndexer(session)) => {
                    let owner = ratio_one_owner(&outputs, latest_owner);
                    ScheduledAttentionOutput::RatioOneIndexer(session.step(LayerFourCall::new(
                        &input,
                        frequencies,
                        layer.query_weights,
                        layer.attention_weights,
                        LayerThreePublication::new(
                            owner.publication(),
                            owner.key_prefix(),
                            owner.kv_prefix(),
                            owner.candidate().candidates(),
                        ),
                    ))?)
                }
                (LayerKind::RatioOneConsumer(layer), LayerState::RatioOneConsumer(state)) => {
                    let owner = ratio_one_owner(&outputs, latest_owner);
                    let indices = match latest_indices.map(|index: usize| &outputs[index].attention)
                    {
                        Some(ScheduledAttentionOutput::RatioOneIndexer(indexer)) => {
                            indexer.selection().indices.as_slice()
                        }
                        _ => owner.selected_indices(),
                    };
                    ScheduledAttentionOutput::RatioOneConsumer(state.forward(
                        &input,
                        start,
                        frequencies,
                        layer.weights,
                        CompressedAttentionPublication {
                            source_layer: owner.publication().source_layer(),
                            epoch: owner.publication().epoch(),
                            call_id: owner.publication().call_id(),
                            numerical_bf16: owner.kv_prefix(),
                            indices,
                        },
                    )?)
                }
                _ => unreachable!("layer state is built from its own kind"),
            };
            let TailOutputs {
                diagnostics: tails,
                residual: next_residual,
                pre: next_pre,
            } = tails(
                &definition.block,
                block_residual,
                attention.final_output(),
                ids.len(),
            )?;
            match attention {
                ScheduledAttentionOutput::RatioTwoOwner(_)
                | ScheduledAttentionOutput::RatioOneOwner(_) => {
                    latest_owner = Some(outputs.len());
                    latest_indices = Some(outputs.len());
                }
                ScheduledAttentionOutput::RatioOneIndexer(_) => {
                    latest_indices = Some(outputs.len());
                }
                _ => {}
            }
            outputs.push(LayerStepOutput {
                engram,
                attention_input,
                attention,
                tails,
            });
            residual = next_residual;
            pre = next_pre;
        }
        let (copies, width) = self.model.startup.tail.geometry();
        let heads = final_heads(self.model.head, &residual, &pre, ids.len(), (copies, width))?;
        if let Some(owner) = outputs
            .iter()
            .rev()
            .find_map(|output| match &output.attention {
                ScheduledAttentionOutput::RatioOneOwner(owner) => Some(owner),
                _ => None,
            })
        {
            self.prior_ratio_one = Some(PriorRatioOneKeys {
                publication: owner.publication(),
                keys: owner.key_prefix().to_vec(),
            });
        }
        self.next_start = end;
        Ok(RequestStepOutput {
            startup,
            layers: outputs,
            residual,
            incoming_pre: pre,
            heads,
        })
    }
}

fn ratio_one_owner(outputs: &[LayerStepOutput], latest: Option<usize>) -> &LayerThreeStepOutput {
    match latest.map(|index| &outputs[index].attention) {
        Some(ScheduledAttentionOutput::RatioOneOwner(owner)) => owner,
        _ => unreachable!("validated schedules precede ratio-one layers with their owner"),
    }
}

/// The attention stages of one scheduled layer, by kind.
#[derive(Debug)]
#[non_exhaustive]
pub enum ScheduledAttentionOutput {
    /// Window-only attention stages.
    WindowOnly(LayerAttentionDiagnostic),
    /// Ratio-two owner, direct-score, and attention stages.
    RatioTwoOwner(LayerOneStepOutput),
    /// Attention over the latest ratio-two publication.
    RatioTwoConsumer(LayerAttentionDiagnostic),
    /// Ratio-one owner, candidate, selection, and attention stages.
    RatioOneOwner(LayerThreeStepOutput),
    /// Candidate-restricted scores, selection, and attention stages.
    RatioOneIndexer(LayerFourStepOutput),
    /// Attention over the ratio-one owner's KV and latest ratio-one indices.
    RatioOneConsumer(LayerAttentionDiagnostic),
}

impl ScheduledAttentionOutput {
    /// Returns this layer's attention output rows.
    #[must_use]
    pub fn final_output(&self) -> &[u16] {
        match self {
            Self::WindowOnly(attention)
            | Self::RatioTwoConsumer(attention)
            | Self::RatioOneConsumer(attention) => &attention.final_output,
            Self::RatioTwoOwner(layer) => &layer.attention().final_output,
            Self::RatioOneOwner(layer) => &layer.attention().final_output,
            Self::RatioOneIndexer(layer) => &layer.attention().final_output,
        }
    }
}

/// Owned numerical boundaries of one scheduled layer in one request chunk.
#[derive(Debug)]
pub struct LayerStepOutput {
    engram: Option<EngramStepOutput>,
    attention_input: Vec<AttentionInputOutput>,
    attention: ScheduledAttentionOutput,
    tails: Vec<BlockTailDiagnostic>,
}

impl LayerStepOutput {
    /// Returns the Engram operation applied before this block, if scheduled.
    #[must_use]
    pub const fn engram(&self) -> Option<&EngramStepOutput> {
        self.engram.as_ref()
    }
    /// Returns HC-collapse and normalization rows.
    #[must_use]
    pub fn attention_input(&self) -> &[AttentionInputOutput] {
        &self.attention_input
    }
    /// Returns this layer's attention stages.
    #[must_use]
    pub const fn attention(&self) -> &ScheduledAttentionOutput {
        &self.attention
    }
    /// Returns block tail diagnostics.
    #[must_use]
    pub fn tails(&self) -> &[BlockTailDiagnostic] {
        &self.tails
    }
}

/// Owned numerical boundaries from one scheduled request chunk.
///
/// The numbered accessors (`layer_one` through `tails_four`) name the
/// reduced schedule built by [`RequestModel::new`]; [`Self::layers`] covers
/// any schedule.
#[derive(Debug)]
pub struct RequestStepOutput {
    startup: StartupStepOutput,
    layers: Vec<LayerStepOutput>,
    residual: Vec<u16>,
    incoming_pre: Vec<f32>,
    heads: Vec<FinalHeadOutput>,
}

impl RequestStepOutput {
    fn numbered(&self, number: usize) -> &LayerStepOutput {
        self.layers
            .get(number - 1)
            .expect("numbered accessors require the reduced schedule")
    }

    fn engram_at(&self, number: usize) -> &EngramStepOutput {
        self.numbered(number)
            .engram()
            .expect("numbered Engram accessors require the reduced schedule")
    }

    /// Returns startup stages and its first block tail.
    #[must_use]
    pub const fn startup(&self) -> &StartupStepOutput {
        &self.startup
    }
    /// Returns every scheduled layer's boundaries in execution order.
    #[must_use]
    pub fn layers(&self) -> &[LayerStepOutput] {
        &self.layers
    }
    /// Returns the first live Engram operation.
    ///
    /// # Panics
    ///
    /// Panics unless layer one applied an Engram, as in the reduced schedule.
    #[must_use]
    pub fn engram_one(&self) -> &EngramStepOutput {
        self.engram_at(1)
    }
    /// Returns L1 HC-collapse and normalization rows.
    ///
    /// # Panics
    ///
    /// Panics if the schedule has no layer one.
    #[must_use]
    pub fn attention_one(&self) -> &[AttentionInputOutput] {
        self.numbered(1).attention_input()
    }
    /// Returns the L1 owner, direct-score, and attention stages.
    ///
    /// # Panics
    ///
    /// Panics unless layer one is a ratio-two owner, as in the reduced schedule.
    #[must_use]
    pub fn layer_one(&self) -> &LayerOneStepOutput {
        match self.numbered(1).attention() {
            ScheduledAttentionOutput::RatioTwoOwner(layer) => layer,
            _ => panic!("reduced layer one is a ratio-two owner"),
        }
    }
    /// Returns numbered block-one tail diagnostics.
    ///
    /// # Panics
    ///
    /// Panics if the schedule has no layer one.
    #[must_use]
    pub fn tails_one(&self) -> &[BlockTailDiagnostic] {
        self.numbered(1).tails()
    }
    /// Returns L2 HC-collapse and normalization rows.
    ///
    /// # Panics
    ///
    /// Panics if the schedule has no layer two.
    #[must_use]
    pub fn attention_two(&self) -> &[AttentionInputOutput] {
        self.numbered(2).attention_input()
    }
    /// Returns L2 attention over the live L1 publication.
    ///
    /// # Panics
    ///
    /// Panics unless layer two is a ratio-two consumer, as in the reduced schedule.
    #[must_use]
    pub fn layer_two(&self) -> &LayerAttentionDiagnostic {
        match self.numbered(2).attention() {
            ScheduledAttentionOutput::RatioTwoConsumer(layer) => layer,
            _ => panic!("reduced layer two is a ratio-two consumer"),
        }
    }
    /// Returns numbered block-two tail diagnostics.
    ///
    /// # Panics
    ///
    /// Panics if the schedule has no layer two.
    #[must_use]
    pub fn tails_two(&self) -> &[BlockTailDiagnostic] {
        self.numbered(2).tails()
    }
    /// Returns the second live Engram operation.
    ///
    /// # Panics
    ///
    /// Panics unless layer three applied an Engram, as in the reduced schedule.
    #[must_use]
    pub fn engram_three(&self) -> &EngramStepOutput {
        self.engram_at(3)
    }
    /// Returns L3 HC-collapse and normalization rows.
    ///
    /// # Panics
    ///
    /// Panics if the schedule has no layer three.
    #[must_use]
    pub fn attention_three(&self) -> &[AttentionInputOutput] {
        self.numbered(3).attention_input()
    }
    /// Returns L3 owner, candidate, selection, and attention stages.
    ///
    /// # Panics
    ///
    /// Panics unless layer three is a ratio-one owner, as in the reduced schedule.
    #[must_use]
    pub fn layer_three(&self) -> &LayerThreeStepOutput {
        match self.numbered(3).attention() {
            ScheduledAttentionOutput::RatioOneOwner(layer) => layer,
            _ => panic!("reduced layer three is a ratio-one owner"),
        }
    }
    /// Returns numbered block-three tail diagnostics.
    ///
    /// # Panics
    ///
    /// Panics if the schedule has no layer three.
    #[must_use]
    pub fn tails_three(&self) -> &[BlockTailDiagnostic] {
        self.numbered(3).tails()
    }
    /// Returns L4 HC-collapse and normalization rows.
    ///
    /// # Panics
    ///
    /// Panics if the schedule has no layer four.
    #[must_use]
    pub fn attention_four(&self) -> &[AttentionInputOutput] {
        self.numbered(4).attention_input()
    }
    /// Returns L4 scores, selection, and attention stages.
    ///
    /// # Panics
    ///
    /// Panics unless layer four is a ratio-one indexer, as in the reduced schedule.
    #[must_use]
    pub fn layer_four(&self) -> &LayerFourStepOutput {
        match self.numbered(4).attention() {
            ScheduledAttentionOutput::RatioOneIndexer(layer) => layer,
            _ => panic!("reduced layer four is a ratio-one indexer"),
        }
    }
    /// Returns numbered block-four tail diagnostics.
    ///
    /// # Panics
    ///
    /// Panics if the schedule has no layer four.
    #[must_use]
    pub fn tails_four(&self) -> &[BlockTailDiagnostic] {
        self.numbered(4).tails()
    }
    /// Returns the final copy-major residual rows.
    #[must_use]
    pub fn residual(&self) -> &[u16] {
        &self.residual
    }
    /// Returns final per-copy incoming coefficients.
    #[must_use]
    pub fn incoming_pre(&self) -> &[f32] {
        &self.incoming_pre
    }
    /// Returns one final-head result per token.
    #[must_use]
    pub fn heads(&self) -> &[FinalHeadOutput] {
        &self.heads
    }
}

/// Why a layer schedule was rejected before request allocation.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Error)]
#[non_exhaustive]
pub enum ScheduleError {
    /// More layers than a `u16` source-layer identity can name.
    #[error("schedule has more layers than source identities")]
    LayerCount,
    /// A layer names an Engram definition the model does not hold.
    #[error("Engram index {index} is outside {available} definitions")]
    EngramIndex { index: usize, available: usize },
    /// A window-only layer was given a compressed attention layout.
    #[error("window-only layer has a compressed attention layout")]
    WindowLayoutCompressed,
    /// A compressed kind was given a window-only attention layout.
    #[error("compressed layer has a window-only attention layout")]
    MissingCompression,
    /// The attention layout's ratio disagrees with the layer kind.
    #[error("layer kind needs compression ratio {expected}, layout has {actual}")]
    CompressionRatio { expected: usize, actual: usize },
    /// An owner does not publish under its own layer number.
    #[error("owner must publish as layer {expected}, layout names {actual}")]
    OwnerSource { expected: u16, actual: u16 },
    /// A consumer or indexer has no matching latest preceding owner.
    #[error("no latest preceding ratio-{ratio} owner at source layer {source_layer}")]
    MissingProducer { source_layer: u16, ratio: usize },
    /// A ratio-two layer follows a ratio-one owner.
    #[error("ratio-two layers must precede ratio-one layers")]
    RatioOrder,
    /// Ratio-two owners have no ratio-one keys for incomplete groups.
    #[error("ratio-two owners need a ratio-one owner for incomplete groups")]
    MissingRatioOneOwner,
}

#[derive(Debug, Error)]
#[non_exhaustive]
pub enum RequestError {
    /// Block tails, startup, or head disagree on copy or hidden width.
    #[error("request blocks do not share copy/hidden geometry")]
    BlockGeometry,
    /// L2 does not consume the fixed L1 ratio-two publication geometry.
    #[error("reused L2 attention must be batch-one source-one ratio-two with block hidden width")]
    LayerTwoGeometry,
    /// One attention layout disagrees on batch, hidden width, or `RoPE` pairs.
    #[error("request attention layouts do not share batch-one hidden/rotary geometry")]
    AttentionGeometry,
    /// The layer schedule is not a valid sequence of producers and consumers.
    #[error("schedule layer {layer}: {reason}")]
    Schedule { layer: usize, reason: ScheduleError },
    /// A requested static or dynamic request surface exceeds its bound.
    #[error("request has {elements} tokens beyond the bounded maximum")]
    ElementLimit { elements: usize },
    /// A prior admitted stage failed after possibly advancing inner state.
    #[error("request is poisoned; restart is required")]
    Poisoned,
    /// The caller supplied an empty token chunk.
    #[error("request needs at least one token ID")]
    EmptyIds,
    /// A post-prefill call included more than one decode token.
    #[error("request decode accepts one token, got {actual}")]
    DecodeChunk { actual: usize },
    /// Absolute token arithmetic overflowed before state mutation.
    #[error("request token position overflowed")]
    PositionOverflow,
    /// The requested call would exceed the immutable request token limit.
    #[error("request end {end} exceeds configured maximum {maximum}")]
    TokenLimit { end: usize, maximum: usize },
    /// The model's full rotary table is too short for a requested span.
    #[error("request needs {required} rotary-frequency elements, table has {available}")]
    FrequencyTable { required: usize, available: usize },
    /// Startup accepts unsigned IDs and rejected a negative input ID.
    #[error("request token ID {id} cannot convert to startup unsigned form")]
    NegativeToken { id: i64 },
    /// An incomplete ratio-two group had no previous-step ratio-one owner keys.
    #[error("a partial ratio-two owner call needs an earlier successful ratio-one publication")]
    MissingPriorLayerThree,
    /// A composed buffer did not match its exact stage geometry.
    #[error("request buffer {field} has {actual} elements, expected {expected}")]
    Length {
        field: &'static str,
        actual: usize,
        expected: usize,
    },
    /// A bounded owned stage buffer could not be allocated.
    #[error("request allocation failed for {field} with {elements} elements")]
    Allocation {
        field: &'static str,
        elements: usize,
    },
    /// Startup execution rejected its live input or operands.
    #[error(transparent)]
    Startup(#[from] StartupSessionError),
    /// An Engram state rejected its live stream.
    #[error(transparent)]
    Engram(#[from] EngramSessionError),
    /// An HC-collapse or RMS-normalization boundary failed.
    #[error(transparent)]
    Input(#[from] AttentionInputError),
    /// A ratio-two owner, its scoring, or its attention failed.
    #[error(transparent)]
    LayerOne(#[from] LayerOneSessionError),
    /// Window-only or reused-publication consumer attention failed.
    #[error(transparent)]
    LayerTwo(#[from] LayerAttentionError),
    /// A ratio-one owner, candidate, selection, or attention failed.
    #[error(transparent)]
    LayerThree(#[from] LayerThreeSessionError),
    /// A candidate indexer's query, selection, or attention failed.
    #[error(transparent)]
    LayerFour(#[from] LayerFourSessionError),
    /// A block tail failed.
    #[error(transparent)]
    Tail(#[from] BlockTailError),
    /// Final normalization or vocabulary projection failed.
    #[error(transparent)]
    Head(#[from] FinalHeadError),
}

fn completes_ratio_two_group(start: usize, positions: usize) -> bool {
    start.checked_add(positions).map(|end| end / 2) != Some(start / 2)
}

fn ids_to_u64(ids: &[i64]) -> Result<Vec<u64>, RequestError> {
    let mut values = reserve(ids.len(), "startup IDs")?;
    for &id in ids {
        values.push(u64::try_from(id).map_err(|_| RequestError::NegativeToken { id })?);
    }
    Ok(values)
}

fn frequency_span(
    all: &[RotaryFrequency],
    start: usize,
    positions: usize,
    rope_pairs: usize,
) -> Result<&[RotaryFrequency], RequestError> {
    let first = start
        .checked_mul(rope_pairs)
        .ok_or(RequestError::PositionOverflow)?;
    let end = start
        .checked_add(positions)
        .and_then(|value| value.checked_mul(rope_pairs))
        .ok_or(RequestError::PositionOverflow)?;
    all.get(first..end).ok_or(RequestError::FrequencyTable {
        required: end,
        available: all.len(),
    })
}

fn attention_inputs(
    block: &BlockDefinition<'_>,
    residual: &[u16],
    pre: &[f32],
) -> Result<Vec<AttentionInputOutput>, RequestError> {
    let (copies, width) = block.geometry();
    let stride = copies
        .checked_mul(width)
        .ok_or(RequestError::PositionOverflow)?;
    if !residual.len().is_multiple_of(stride) || pre.len() != residual.len() / width {
        return Err(RequestError::Length {
            field: "attention input rows",
            actual: residual.len(),
            expected: stride,
        });
    }
    let mut output = reserve(residual.len() / stride, "attention inputs")?;
    for (row, coefficients) in residual.chunks_exact(stride).zip(pre.chunks_exact(copies)) {
        output.push(block.input.forward(row, coefficients)?);
    }
    Ok(output)
}

fn normalized_rows(inputs: &[AttentionInputOutput]) -> Result<Vec<u16>, RequestError> {
    let first = inputs.first().ok_or(RequestError::Length {
        field: "attention inputs",
        actual: 0,
        expected: 1,
    })?;
    let width = first.normalized_bf16().len();
    let elements = inputs
        .len()
        .checked_mul(width)
        .ok_or(RequestError::PositionOverflow)?;
    let mut output = reserve(elements, "normalized attention inputs")?;
    for input in inputs {
        if input.normalized_bf16().len() != width {
            return Err(RequestError::Length {
                field: "normalized attention row",
                actual: input.normalized_bf16().len(),
                expected: width,
            });
        }
        output.extend_from_slice(input.normalized_bf16());
    }
    Ok(output)
}

struct TailOutputs {
    diagnostics: Vec<BlockTailDiagnostic>,
    residual: Vec<u16>,
    pre: Vec<f32>,
}

fn tails(
    block: &BlockDefinition<'_>,
    residual: &[u16],
    attention: &[u16],
    positions: usize,
) -> Result<TailOutputs, RequestError> {
    let (copies, width) = block.geometry();
    let residual_stride = copies
        .checked_mul(width)
        .ok_or(RequestError::PositionOverflow)?;
    let expected_residual = positions
        .checked_mul(residual_stride)
        .ok_or(RequestError::PositionOverflow)?;
    let expected_attention = positions
        .checked_mul(width)
        .ok_or(RequestError::PositionOverflow)?;
    if residual.len() != expected_residual {
        return Err(RequestError::Length {
            field: "tail residual",
            actual: residual.len(),
            expected: expected_residual,
        });
    }
    if attention.len() != expected_attention {
        return Err(RequestError::Length {
            field: "tail attention",
            actual: attention.len(),
            expected: expected_attention,
        });
    }
    let mut diagnostics = reserve(positions, "tail diagnostics")?;
    let mut next_residual = reserve(expected_residual, "tail residual output")?;
    let mut next_pre = reserve(
        positions
            .checked_mul(copies)
            .ok_or(RequestError::PositionOverflow)?,
        "tail incoming pre",
    )?;
    for (residual_row, attention_row) in residual
        .chunks_exact(residual_stride)
        .zip(attention.chunks_exact(width))
    {
        let tail = block.tail.forward_token(residual_row, attention_row)?;
        next_residual.extend_from_slice(tail.ffn().output_bf16());
        next_pre.extend_from_slice(tail.ffn().coefficients().pre());
        diagnostics.push(tail);
    }
    Ok(TailOutputs {
        diagnostics,
        residual: next_residual,
        pre: next_pre,
    })
}

fn final_heads(
    head: FinalHead<'_>,
    residual: &[u16],
    pre: &[f32],
    positions: usize,
    geometry: (usize, usize),
) -> Result<Vec<FinalHeadOutput>, RequestError> {
    let (copies, width) = geometry;
    let stride = copies
        .checked_mul(width)
        .ok_or(RequestError::PositionOverflow)?;
    if residual.len()
        != positions
            .checked_mul(stride)
            .ok_or(RequestError::PositionOverflow)?
        || pre.len()
            != positions
                .checked_mul(copies)
                .ok_or(RequestError::PositionOverflow)?
    {
        return Err(RequestError::Length {
            field: "final head",
            actual: residual.len(),
            expected: positions * stride,
        });
    }
    let mut outputs = reserve(positions, "head outputs")?;
    for (row, coefficients) in residual.chunks_exact(stride).zip(pre.chunks_exact(copies)) {
        outputs.push(head.forward(row, coefficients)?);
    }
    Ok(outputs)
}

fn reserve<T>(elements: usize, field: &'static str) -> Result<Vec<T>, RequestError> {
    if elements > MAX_REQUEST_ELEMENTS {
        return Err(RequestError::ElementLimit { elements });
    }
    let mut values = Vec::new();
    values
        .try_reserve_exact(elements)
        .map_err(|_| RequestError::Allocation { field, elements })?;
    Ok(values)
}
