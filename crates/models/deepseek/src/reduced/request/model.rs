//! The validated request model: startup, scheduled layers, and head.

use std::num::NonZeroUsize;

use crate::{
    RotaryFrequency,
    attention::layer::{LayerAttentionLayout, LayerAttentionState},
    indexer::{
        key::{IndexKeyPreparationExecution, IndexKeyRotaryExecution},
        query::IndexScoreExecution,
    },
    reduced::{FinalHead, FinalHeadExecution},
};

use super::{
    BlockDefinition, EngramDefinition, LayerFourDefinition, LayerKind, LayerOneDefinition,
    LayerThreeDefinition, MAX_REQUEST_ELEMENTS, RequestError, ReusedAttentionDefinition,
    ScheduleError, ScheduledLayer, StartupDefinition, session::LayerState,
};

/// One checkpoint-defined runtime model: startup, scheduled layers, and head.
#[derive(Clone, Debug)]
pub struct RequestModel<'a> {
    pub(super) startup: StartupDefinition<'a>,
    pub(super) layers: Vec<ScheduledLayer<'a>>,
    pub(super) engrams: Vec<EngramDefinition>,
    pub(super) head: FinalHead<'a>,
    pub(super) frequencies: &'a [RotaryFrequency],
    pub(super) max_tokens: NonZeroUsize,
    pub(super) score_execution: IndexScoreExecution,
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

    pub(super) fn layer_state(
        &self,
        layer: &ScheduledLayer<'_>,
    ) -> Result<LayerState, RequestError> {
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
