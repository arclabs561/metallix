//! Fixed five-block reduced request composition over live numerical operands.
//!
//! This module is deliberately a single V4.1-shaped path: startup, Engram,
//! L1, L2, Engram, L3, L4, and final head. It has no checkpoint loader,
//! fixture decoder, callback graph, or generic scheduler.

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
        key::IndexKeyRotaryExecution,
        query::{CandidateQueryLayout, CandidateQueryWeights, IndexScoreExecution},
    },
};

use super::{
    AttentionInput, AttentionInputError, AttentionInputOutput, BlockTailDiagnostic, BlockTailError,
    BlockTailReference, EngramSession, EngramSessionConfig, EngramSessionError,
    EngramSessionWeights, EngramStepOutput, FinalHead, FinalHeadError, FinalHeadOutput,
    LayerFourCall, LayerFourConfig, LayerFourSession, LayerFourSessionError, LayerFourStepOutput,
    LayerOneCall, LayerOneConfig, LayerOneSession, LayerOneSessionError, LayerOneStepOutput,
    LayerThreeCall, LayerThreeConfig, LayerThreePublication, LayerThreeSession,
    LayerThreeSessionError, LayerThreeStepOutput, PreviousLayerThreeKeys, RatioTwoOwnerWeights,
    StartupSession, StartupSessionError, StartupStepOutput,
};

const SOURCE_L1: u16 = 1;
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
        key_rotary_execution: IndexKeyRotaryExecution,
    ) -> Result<LayerOneSession, RequestError> {
        Ok(LayerOneSession::new(self.config, self.compressor_norm)?
            .with_score_execution(score_execution)
            .with_key_rotary_execution(key_rotary_execution))
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
        key_rotary_execution: IndexKeyRotaryExecution,
    ) -> Result<LayerThreeSession, RequestError> {
        Ok(LayerThreeSession::new(
            self.config,
            3,
            self.compressor_norm,
            self.compressor_epsilon,
        )?
        .with_key_rotary_execution(key_rotary_execution))
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

/// Immutable L2 attention operands which reuse the live L1 publication.
#[derive(Clone, Copy, Debug)]
pub struct ReusedAttentionDefinition<'a> {
    layout: LayerAttentionLayout,
    weights: LayerAttentionWeights<'a>,
}

impl<'a> ReusedAttentionDefinition<'a> {
    /// Groups L2 attention layout and weights for live L1 publication reuse.
    #[must_use]
    pub const fn new(layout: LayerAttentionLayout, weights: LayerAttentionWeights<'a>) -> Self {
        Self { layout, weights }
    }
}

/// One fixed five-block reduced runtime model definition.
#[derive(Clone, Debug)]
pub struct RequestModel<'a> {
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
    score_execution: IndexScoreExecution,
    key_rotary_execution: IndexKeyRotaryExecution,
}

impl<'a> RequestModel<'a> {
    /// Validates immutable operands for the fixed five-block request path.
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
        let model = Self {
            startup,
            blocks,
            engrams,
            layer_one,
            layer_two,
            layer_three,
            layer_four,
            head,
            frequencies,
            max_tokens,
            score_execution: IndexScoreExecution::Scalar,
            key_rotary_execution: IndexKeyRotaryExecution::Scalar,
        };
        model.validate()?;
        Ok(model)
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

    /// Selects the index-key rotary implementation for L1 and L3 owners.
    #[must_use]
    pub const fn with_key_rotary_execution(
        mut self,
        key_rotary_execution: IndexKeyRotaryExecution,
    ) -> Self {
        self.key_rotary_execution = key_rotary_execution;
        self
    }

    /// Returns the model-local index-key rotary implementation.
    #[must_use]
    pub const fn key_rotary_execution(&self) -> IndexKeyRotaryExecution {
        self.key_rotary_execution
    }

    fn validate(&self) -> Result<(), RequestError> {
        let (_, width) = self.blocks[0].geometry();
        let copies = self.blocks[0].geometry().0;
        if self.blocks.iter().any(|block| {
            block.geometry() != (copies, width) || block.input.geometry() != (copies, width)
        }) {
            return Err(RequestError::BlockGeometry);
        }
        if self.startup.tail.geometry() != (copies, width)
            || self.startup.norm.len() != width
            || self.head.geometry() != (copies, width)
        {
            return Err(RequestError::BlockGeometry);
        }
        let rope_pairs = self.layer_two.layout.rope_pairs();
        let attention_layouts = [
            self.startup.attention_layout,
            self.layer_one.config.attention_layout(),
            self.layer_two.layout,
            self.layer_three.config.attention_layout(),
            self.layer_four.config.attention_layout(),
        ];
        if attention_layouts.iter().any(|layout| {
            layout.batches().get() != 1
                || layout.hidden_dimension().get() != width
                || layout.rope_pairs() != rope_pairs
        }) {
            return Err(RequestError::AttentionGeometry);
        }
        if self.layer_two.layout.batches().get() != 1
            || self.layer_two.layout.hidden_dimension().get() != width
            || self
                .layer_two
                .layout
                .compression()
                .map(|(source, ratio)| (source, ratio.get()))
                != Some((SOURCE_L1, 2))
        {
            return Err(RequestError::LayerTwoGeometry);
        }
        if self.max_tokens.get() > MAX_REQUEST_ELEMENTS {
            return Err(RequestError::ElementLimit {
                elements: self.max_tokens.get(),
            });
        }
        let required_frequencies = self
            .max_tokens
            .get()
            .checked_mul(self.layer_two.layout.rope_pairs().get())
            .ok_or(RequestError::PositionOverflow)?;
        if self.frequencies.len() < required_frequencies {
            return Err(RequestError::FrequencyTable {
                required: required_frequencies,
                available: self.frequencies.len(),
            });
        }
        let required_startup_frequencies = self
            .max_tokens
            .get()
            .checked_mul(self.startup.attention_layout.rope_pairs().get())
            .ok_or(RequestError::PositionOverflow)?;
        if self.startup.frequencies.len() < required_startup_frequencies {
            return Err(RequestError::FrequencyTable {
                required: required_startup_frequencies,
                available: self.startup.frequencies.len(),
            });
        }
        let _ = self.startup.session()?;
        let _ = self.engrams[0].session()?;
        let _ = self.engrams[1].session()?;
        let _ = self
            .layer_one
            .session(self.score_execution, self.key_rotary_execution)?;
        let _ = self.layer_three.session(self.key_rotary_execution)?;
        let _ = self.layer_four.session(self.score_execution);
        Ok(())
    }
}

struct PriorLayerThreePublication {
    publication: IndexKeyPublicationId,
    keys: Vec<u16>,
}

/// Mutable five-block request state rebuilt as one unit on restart.
pub struct RequestSession<'a> {
    model: &'a RequestModel<'a>,
    startup: StartupSession<'a>,
    engram_one: EngramSession,
    layer_one: LayerOneSession,
    layer_two: LayerAttentionState,
    engram_three: EngramSession,
    layer_three: LayerThreeSession,
    layer_four: LayerFourSession,
    prior_layer_three: Option<PriorLayerThreePublication>,
    next_start: usize,
    poisoned: bool,
}

impl<'a> RequestSession<'a> {
    /// Constructs all request-local owners from one immutable model definition.
    pub fn new(model: &'a RequestModel<'a>) -> Result<Self, RequestError> {
        Self::build(model)
    }

    fn build(model: &'a RequestModel<'a>) -> Result<Self, RequestError> {
        Ok(Self {
            model,
            startup: model.startup.session()?,
            engram_one: model.engrams[0].session()?,
            layer_one: model
                .layer_one
                .session(model.score_execution, model.key_rotary_execution)?,
            layer_two: LayerAttentionState::new(model.layer_two.layout),
            engram_three: model.engrams[1].session()?,
            layer_three: model.layer_three.session(model.key_rotary_execution)?,
            layer_four: model.layer_four.session(model.score_execution),
            prior_layer_three: None,
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
        reason = "the fixed five-block numerical order is intentionally visible in one request path"
    )]
    fn step_admitted(
        &mut self,
        ids: &[i64],
        end: usize,
    ) -> Result<RequestStepOutput, RequestError> {
        let start = self.next_start;
        let frequencies = frequency_span(
            self.model.frequencies,
            start,
            ids.len(),
            self.model.layer_two.layout.rope_pairs().get(),
        )?;
        let startup_frequencies = frequency_span(
            self.model.startup.frequencies,
            start,
            ids.len(),
            self.model.startup.attention_layout.rope_pairs().get(),
        )?;
        let startup_ids = ids_to_u64(ids)?;
        let startup = self
            .startup
            .step(start, &startup_ids, startup_frequencies)?;
        let engram_one = self.engram_one.step(start, ids, startup.residual())?;
        let attention_one = attention_inputs(
            &self.model.blocks[0],
            engram_one.output(),
            startup.next_pre(),
        )?;
        let l1_is_partial = !completes_ratio_two_group(start, ids.len());
        let l1_prior = previous_l3_if_partial(self.prior_layer_three.as_ref(), start, ids.len());
        if l1_is_partial && l1_prior.is_none() {
            return Err(RequestError::MissingPriorLayerThree);
        }
        let l1_input = normalized_rows(&attention_one)?;
        let layer_one = self.layer_one.step(LayerOneCall::new(
            &l1_input,
            NonZeroUsize::new(ids.len()).expect("nonempty ids"),
            self.model.frequencies,
            self.model.layer_one.owner_weights,
            self.model.layer_one.query_weights,
            self.model.layer_one.query_layout,
            self.model.layer_one.attention_weights,
            l1_prior,
        ))?;
        let TailOutputs {
            diagnostics: tails_one,
            residual: residual_one,
            pre: pre_one,
        } = tails(
            &self.model.blocks[0],
            engram_one.output(),
            layer_one.attention().final_output.as_slice(),
            ids.len(),
        )?;
        let attention_two = attention_inputs(&self.model.blocks[1], &residual_one, &pre_one)?;
        let l2_input = normalized_rows(&attention_two)?;
        let layer_two = self.layer_two.forward(
            &l2_input,
            start,
            frequencies,
            self.model.layer_two.weights,
            CompressedAttentionPublication {
                source_layer: layer_one.publication().source_layer(),
                epoch: layer_one.publication().epoch(),
                call_id: layer_one.publication().call_id(),
                numerical_bf16: layer_one.kv_prefix(),
                indices: layer_one.selected_indices(),
            },
        )?;
        let TailOutputs {
            diagnostics: tails_two,
            residual: residual_two,
            pre: pre_two,
        } = tails(
            &self.model.blocks[1],
            &residual_one,
            &layer_two.final_output,
            ids.len(),
        )?;
        let engram_three = self.engram_three.step(start, ids, &residual_two)?;
        let attention_three =
            attention_inputs(&self.model.blocks[2], engram_three.output(), &pre_two)?;
        let l3_input = normalized_rows(&attention_three)?;
        let layer_three = self.layer_three.step(LayerThreeCall::new(
            &l3_input,
            NonZeroUsize::new(ids.len()).expect("nonempty ids"),
            frequencies,
            self.model.layer_three.owner_weights,
            self.model
                .layer_three
                .candidate
                .with_score_execution(self.model.score_execution),
            self.model.layer_three.attention_weights,
        ))?;
        let TailOutputs {
            diagnostics: tails_three,
            residual: residual_three,
            pre: pre_three,
        } = tails(
            &self.model.blocks[2],
            engram_three.output(),
            layer_three.attention().final_output.as_slice(),
            ids.len(),
        )?;
        let attention_four = attention_inputs(&self.model.blocks[3], &residual_three, &pre_three)?;
        let l4_input = normalized_rows(&attention_four)?;
        let layer_four = self.layer_four.step(LayerFourCall::new(
            &l4_input,
            frequencies,
            self.model.layer_four.query_weights,
            self.model.layer_four.attention_weights,
            LayerThreePublication::new(
                layer_three.publication(),
                layer_three.key_prefix(),
                layer_three.kv_prefix(),
                layer_three.candidate().candidates(),
            ),
        ))?;
        let TailOutputs {
            diagnostics: tails_four,
            residual: residual_four,
            pre: pre_four,
        } = tails(
            &self.model.blocks[3],
            &residual_three,
            &layer_four.attention().final_output,
            ids.len(),
        )?;
        let heads = final_heads(
            self.model.head,
            &residual_four,
            &pre_four,
            ids.len(),
            self.model.blocks[3].geometry(),
        )?;
        self.prior_layer_three = Some(PriorLayerThreePublication {
            publication: layer_three.publication(),
            keys: layer_three.key_prefix().to_vec(),
        });
        self.next_start = end;
        Ok(RequestStepOutput {
            startup,
            engram_one,
            attention_one,
            layer_one,
            tails_one,
            attention_two,
            layer_two,
            tails_two,
            engram_three,
            attention_three,
            layer_three,
            tails_three,
            attention_four,
            layer_four,
            tails_four,
            residual: residual_four,
            incoming_pre: pre_four,
            heads,
        })
    }
}

/// Owned numerical boundaries from one fixed request chunk.
#[derive(Debug)]
pub struct RequestStepOutput {
    startup: StartupStepOutput,
    engram_one: EngramStepOutput,
    attention_one: Vec<AttentionInputOutput>,
    layer_one: LayerOneStepOutput,
    tails_one: Vec<BlockTailDiagnostic>,
    attention_two: Vec<AttentionInputOutput>,
    layer_two: LayerAttentionDiagnostic,
    tails_two: Vec<BlockTailDiagnostic>,
    engram_three: EngramStepOutput,
    attention_three: Vec<AttentionInputOutput>,
    layer_three: LayerThreeStepOutput,
    tails_three: Vec<BlockTailDiagnostic>,
    attention_four: Vec<AttentionInputOutput>,
    layer_four: LayerFourStepOutput,
    tails_four: Vec<BlockTailDiagnostic>,
    residual: Vec<u16>,
    incoming_pre: Vec<f32>,
    heads: Vec<FinalHeadOutput>,
}

impl RequestStepOutput {
    /// Returns startup stages and its first block tail.
    #[must_use]
    pub const fn startup(&self) -> &StartupStepOutput {
        &self.startup
    }
    /// Returns the first live Engram operation.
    #[must_use]
    pub const fn engram_one(&self) -> &EngramStepOutput {
        &self.engram_one
    }
    /// Returns L1 HC-collapse and normalization rows.
    #[must_use]
    pub fn attention_one(&self) -> &[AttentionInputOutput] {
        &self.attention_one
    }
    /// Returns the L1 owner, direct-score, and attention stages.
    #[must_use]
    pub const fn layer_one(&self) -> &LayerOneStepOutput {
        &self.layer_one
    }
    /// Returns numbered block-one tail diagnostics.
    #[must_use]
    pub fn tails_one(&self) -> &[BlockTailDiagnostic] {
        &self.tails_one
    }
    /// Returns L2 HC-collapse and normalization rows.
    #[must_use]
    pub fn attention_two(&self) -> &[AttentionInputOutput] {
        &self.attention_two
    }
    /// Returns L2 attention over the live L1 publication.
    #[must_use]
    pub const fn layer_two(&self) -> &LayerAttentionDiagnostic {
        &self.layer_two
    }
    /// Returns numbered block-two tail diagnostics.
    #[must_use]
    pub fn tails_two(&self) -> &[BlockTailDiagnostic] {
        &self.tails_two
    }
    /// Returns the second live Engram operation.
    #[must_use]
    pub const fn engram_three(&self) -> &EngramStepOutput {
        &self.engram_three
    }
    /// Returns L3 HC-collapse and normalization rows.
    #[must_use]
    pub fn attention_three(&self) -> &[AttentionInputOutput] {
        &self.attention_three
    }
    /// Returns L3 owner, candidate, selection, and attention stages.
    #[must_use]
    pub const fn layer_three(&self) -> &LayerThreeStepOutput {
        &self.layer_three
    }
    /// Returns numbered block-three tail diagnostics.
    #[must_use]
    pub fn tails_three(&self) -> &[BlockTailDiagnostic] {
        &self.tails_three
    }
    /// Returns L4 HC-collapse and normalization rows.
    #[must_use]
    pub fn attention_four(&self) -> &[AttentionInputOutput] {
        &self.attention_four
    }
    /// Returns L4 scores, selection, and attention stages.
    #[must_use]
    pub const fn layer_four(&self) -> &LayerFourStepOutput {
        &self.layer_four
    }
    /// Returns numbered block-four tail diagnostics.
    #[must_use]
    pub fn tails_four(&self) -> &[BlockTailDiagnostic] {
        &self.tails_four
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

#[derive(Debug, Error)]
#[non_exhaustive]
pub enum RequestError {
    /// Numbered block tails disagree on copy or hidden width.
    #[error("fixed request blocks do not share copy/hidden geometry")]
    BlockGeometry,
    /// L2 does not consume the fixed L1 ratio-two publication geometry.
    #[error("reused L2 attention must be batch-one source-one ratio-two with block hidden width")]
    LayerTwoGeometry,
    /// One numbered attention layout disagrees on batch, hidden width, or `RoPE` pairs.
    #[error("fixed request attention layouts do not share batch-one hidden/rotary geometry")]
    AttentionGeometry,
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
    /// A ratio-two L1 partial group had no earlier successful L3 key prefix.
    #[error("a partial L1 call needs an earlier successful L3 publication")]
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
    /// Either Engram state rejected its live stream.
    #[error(transparent)]
    Engram(#[from] EngramSessionError),
    /// An HC-collapse or RMS-normalization boundary failed.
    #[error(transparent)]
    Input(#[from] AttentionInputError),
    /// L1 owner, scoring, or attention failed.
    #[error(transparent)]
    LayerOne(#[from] LayerOneSessionError),
    /// Reused-publication L2 attention failed.
    #[error(transparent)]
    LayerTwo(#[from] LayerAttentionError),
    /// L3 owner, candidate, selection, or attention failed.
    #[error(transparent)]
    LayerThree(#[from] LayerThreeSessionError),
    /// L4 query, candidate selection, or attention failed.
    #[error(transparent)]
    LayerFour(#[from] LayerFourSessionError),
    /// A numbered block tail failed.
    #[error(transparent)]
    Tail(#[from] BlockTailError),
    /// Final normalization or vocabulary projection failed.
    #[error(transparent)]
    Head(#[from] FinalHeadError),
}

fn previous_l3_if_partial(
    prior: Option<&PriorLayerThreePublication>,
    start: usize,
    positions: usize,
) -> Option<PreviousLayerThreeKeys<'_>> {
    let completes = completes_ratio_two_group(start, positions);
    if completes {
        None
    } else {
        prior
            .as_ref()
            .map(|value| PreviousLayerThreeKeys::new(value.publication, &value.keys))
    }
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
