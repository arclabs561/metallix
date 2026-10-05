//! Request-local state and the single scheduled stepping loop.

use std::num::NonZeroUsize;

use crate::{
    RotaryFrequency,
    attention::layer::{CompressedAttentionPublication, LayerAttentionState},
    engram::embedding::EngramRowSource,
    indexer::{
        cache::IndexKeyPublicationId,
        key::{IndexKeyPreparationExecution, IndexKeyRotaryExecution},
        query::IndexScoreExecution,
    },
    moe::RoutedExpertSource,
    reduced::{
        AttentionInputOutput, BlockTailDiagnostic, EngramSession, FinalHead, FinalHeadExecution,
        FinalHeadOutput, LayerFourCall, LayerFourSession, LayerOneCall, LayerOneSession,
        LayerThreeCall, LayerThreePublication, LayerThreeSession, LayerThreeStepOutput,
        PreviousLayerThreeKeys, StartupSession,
    },
};

use super::{
    BlockDefinition, LayerKind, LayerStepOutput, RequestError, RequestModel, RequestStepOutput,
    ScheduledAttentionOutput, reserve,
};

/// Request-local state for one scheduled layer, matching its [`LayerKind`].
pub(super) enum LayerState {
    WindowOnly(LayerAttentionState),
    RatioTwoOwner(LayerOneSession),
    RatioTwoConsumer(LayerAttentionState),
    RatioOneOwner(LayerThreeSession),
    RatioOneIndexer(LayerFourSession),
    RatioOneConsumer(LayerAttentionState),
}

/// Caller-owned numerical sources for one request step.
///
/// `experts[n]` serves model layer `n` (0 is startup, `n` is scheduled layer
/// `n`); `engram_rows[i]` serves the model's `i`-th Engram definition. An
/// empty slice or a `None` entry keeps the construction-time table.
#[derive(Clone, Copy, Default)]
pub struct StepSources<'a> {
    pub experts: &'a [Option<&'a dyn RoutedExpertSource>],
    pub engram_rows: &'a [Option<&'a dyn EngramRowSource>],
}

impl<'a> StepSources<'a> {
    fn expert(self, layer: usize) -> Option<&'a dyn RoutedExpertSource> {
        self.experts.get(layer).copied().flatten()
    }

    fn engram_rows(self, engram: usize) -> Option<&'a dyn EngramRowSource> {
        self.engram_rows.get(engram).copied().flatten()
    }
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
        self.step_with_sources(ids, StepSources::default())
    }

    /// Same as [`Self::step`], with routed experts and Engram embedding rows
    /// fetched from caller sources instead of construction-time tables.
    ///
    /// Each [`StepSources`] slice is either empty (no sources of that kind)
    /// or has exactly one entry per model layer or Engram definition; a
    /// mismatch is rejected before admission without poisoning the session.
    pub fn step_with_sources(
        &mut self,
        ids: &[i64],
        sources: StepSources<'_>,
    ) -> Result<RequestStepOutput, RequestError> {
        let layers = self.model.layers.len() + 1;
        if !sources.experts.is_empty() && sources.experts.len() != layers {
            return Err(RequestError::ExpertSourceCount {
                expected: layers,
                actual: sources.experts.len(),
            });
        }
        let engrams = self.model.engrams.len();
        if !sources.engram_rows.is_empty() && sources.engram_rows.len() != engrams {
            return Err(RequestError::EngramRowSourceCount {
                expected: engrams,
                actual: sources.engram_rows.len(),
            });
        }
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
        let result = self.step_admitted(ids, end, sources);
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
        sources: StepSources<'_>,
    ) -> Result<RequestStepOutput, RequestError> {
        let start = self.next_start;
        let positions = NonZeroUsize::new(ids.len()).expect("nonempty ids");
        let rope_pairs = self.model.startup.attention_layout.rope_pairs().get();
        let frequencies = frequency_span(self.model.frequencies, start, ids.len(), rope_pairs)?;
        let startup_frequencies =
            frequency_span(self.model.startup.frequencies, start, ids.len(), rope_pairs)?;
        let startup_ids = ids_to_u64(ids)?;
        let startup =
            self.startup
                .step_with(start, &startup_ids, startup_frequencies, sources.expert(0))?;
        let mut residual = startup.residual().to_vec();
        let mut pre = startup.next_pre().to_vec();
        let partial_group = !completes_ratio_two_group(start, ids.len());
        let mut outputs: Vec<LayerStepOutput> = reserve(self.layers.len(), "layer outputs")?;
        // Indices into `outputs` of the latest owner and latest index publisher.
        let mut latest_owner = None;
        let mut latest_indices = None;
        for (index, ((definition, state), engram)) in self
            .model
            .layers
            .iter()
            .zip(&mut self.layers)
            .zip(&mut self.engrams)
            .enumerate()
        {
            let engram = engram
                .as_mut()
                .map(|engram| {
                    match definition
                        .engram
                        .and_then(|index| sources.engram_rows(index))
                    {
                        Some(rows) => engram.step_with(start, ids, &residual, rows),
                        None => engram.step(start, ids, &residual),
                    }
                })
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
                sources.expert(index + 1),
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
    experts: Option<&dyn RoutedExpertSource>,
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
        let tail = match experts {
            Some(experts) => block
                .tail
                .forward_token_with(residual_row, attention_row, experts)?,
            None => block.tail.forward_token(residual_row, attention_row)?,
        };
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
