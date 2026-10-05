//! Fixed reduced-request numerical assembly from validated artifact tensors.

use std::{collections::BTreeSet, num::NonZeroUsize};

use crate::{
    RotaryFrequency,
    attention::layer::{Fp8Projection, LayerAttentionLayout, LayerAttentionWeights},
    engram::EngramHashLayout,
    ffn::FfnSublayerReference,
    indexer::{
        key::{IndexKeyLayout, IndexKeyPreparationExecution, IndexKeyWeights},
        owner::RatioOneOwnerWeights,
        query::{CandidateQueryLayout, CandidateQueryWeights, IndexQueryLayout, IndexQueryWeights},
    },
    moe::{Fp4ExpertWeights, Fp8ExpertWeights, MoEConfig, MoEReference},
    reduced::{
        AttentionInput, BlockDefinition, BlockTailReference, EngramDefinition, EngramSessionConfig,
        EngramSessionWeights, FinalHead, FinalHeadExecution, LayerFourConfig, LayerFourDefinition,
        LayerOneConfig, LayerOneDefinition, LayerThreeConfig, LayerThreeDefinition,
        RatioTwoOwnerLayout, RatioTwoOwnerWeights, RequestModel, RequestSession, RequestStepOutput,
        ReusedAttentionDefinition, StartupDefinition,
    },
};

use super::{ArtifactConfig, ArtifactError, TensorStore};

const STARTUP_LAYER: usize = 0;
const LAYER_ONE: usize = 1;
const LAYER_TWO: usize = 2;
const LAYER_THREE: usize = 3;
const LAYER_FOUR: usize = 4;

/// Returns the complete flat tensor inventory consumed by the fixed assembly.
///
/// Artifact admission uses this set before decoding payloads, so an artifact
/// cannot carry an ignored numerical input or omit an operand that would only
/// fail later during request construction.
pub(super) fn tensor_names(config: &ArtifactConfig) -> BTreeSet<String> {
    let mut names = BTreeSet::from([
        "embed.weight".to_owned(),
        "rotary.startup".to_owned(),
        "rotary.shared".to_owned(),
        "engram.token_map".to_owned(),
        "engram.primes".to_owned(),
        "engram.offsets".to_owned(),
        "engram.multipliers".to_owned(),
        "head.norm.weight".to_owned(),
        "head.weight".to_owned(),
    ]);
    for layer in STARTUP_LAYER..=LAYER_FOUR {
        let prefix = format!("layers.{layer}");
        names.extend([
            format!("{prefix}.attn.wq_a.weight"),
            format!("{prefix}.attn.wq_a.scale"),
            format!("{prefix}.attn.q_norm.weight"),
            format!("{prefix}.attn.wq_b.weight"),
            format!("{prefix}.attn.wq_b.scale"),
            format!("{prefix}.attn.wkv.weight"),
            format!("{prefix}.attn.wkv.scale"),
            format!("{prefix}.attn.kv_norm.weight"),
            format!("{prefix}.attn.attn_sink"),
            format!("{prefix}.attn.wo_a.weight"),
            format!("{prefix}.attn.wo_b.weight"),
            format!("{prefix}.attn.wo_b.scale"),
            format!("{prefix}.attn_norm.weight"),
            format!("{prefix}.ffn_norm.weight"),
            format!("{prefix}.hc_attn_fn"),
            format!("{prefix}.hc_attn_scale"),
            format!("{prefix}.hc_attn_base"),
            format!("{prefix}.hc_ffn_fn"),
            format!("{prefix}.hc_ffn_scale"),
            format!("{prefix}.hc_ffn_base"),
            format!("{prefix}.ffn.gate.weight"),
            format!("{prefix}.ffn.gate.bias"),
            format!("{prefix}.ffn.shared_experts.w1.weight"),
            format!("{prefix}.ffn.shared_experts.w1.scale"),
            format!("{prefix}.ffn.shared_experts.w2.weight"),
            format!("{prefix}.ffn.shared_experts.w2.scale"),
            format!("{prefix}.ffn.shared_experts.w3.weight"),
            format!("{prefix}.ffn.shared_experts.w3.scale"),
        ]);
        for expert in 0..config.routed_experts {
            let expert_prefix = format!("{prefix}.ffn.experts.{expert}");
            names.extend([
                format!("{expert_prefix}.w1.weight"),
                format!("{expert_prefix}.w1.scale"),
                format!("{expert_prefix}.w2.weight"),
                format!("{expert_prefix}.w2.scale"),
                format!("{expert_prefix}.w3.weight"),
                format!("{expert_prefix}.w3.scale"),
            ]);
        }
    }
    for layer in [LAYER_ONE, LAYER_THREE, LAYER_FOUR] {
        let prefix = format!("layers.{layer}.attn.indexer");
        names.extend([
            format!("{prefix}.wq_b.weight"),
            format!("{prefix}.wq_b.scale"),
            format!("{prefix}.weights_proj.weight"),
        ]);
    }
    for layer in [LAYER_ONE, LAYER_THREE] {
        let prefix = format!("layers.{layer}.attn");
        names.extend([
            format!("{prefix}.compressor.norm.weight"),
            format!("{prefix}.compressor.wkv.weight"),
            format!("{prefix}.indexer.wk.weight"),
            format!("{prefix}.indexer.k_norm.weight"),
        ]);
    }
    names.insert("layers.1.attn.compressor.wgate.weight".to_owned());
    for layer in [LAYER_ONE, LAYER_THREE] {
        let prefix = format!("layers.{layer}.engram");
        names.extend([
            format!("{prefix}.embed.weight"),
            format!("{prefix}.embed.scale"),
            format!("{prefix}.wkv.weight"),
            format!("{prefix}.wkv.scale"),
            format!("{prefix}.q_weight"),
            format!("{prefix}.k_weight"),
        ]);
    }
    names
}

/// Lends the fixed five-block synthetic request without fixture data.
///
/// Construction owns temporary decoded routing and frequency operands, so the
/// callback keeps every borrow valid while retaining one request session over
/// a prefill and later decode steps.
pub(super) fn with_request_model<T>(
    config: &ArtifactConfig,
    tensors: &TensorStore,
    execution: crate::indexer::query::IndexScoreExecution,
    key_preparation_execution: IndexKeyPreparationExecution,
    head_execution: FinalHeadExecution,
    body: impl FnOnce(&RequestModel<'_>) -> Result<T, ArtifactError>,
) -> Result<T, ArtifactError> {
    with_parts(config, tensors, |parts| {
        let model = RequestModel::new(
            parts.startup,
            parts.blocks,
            parts.engrams,
            parts.layer_one,
            parts.layer_two,
            parts.layer_three,
            parts.layer_four,
            parts.head,
            parts.frequencies,
            parts.max_tokens,
        )
        .map_err(ArtifactError::from)?
        .with_score_execution(execution)
        .with_key_preparation_execution(key_preparation_execution)
        .with_head_execution(head_execution);
        body(&model)
    })
}

/// Decoded immutable operands of the fixed five-block request.
pub(super) struct Parts<'a> {
    pub(super) startup: StartupDefinition<'a>,
    pub(super) blocks: [BlockDefinition<'a>; 4],
    pub(super) engrams: [EngramDefinition; 2],
    pub(super) layer_one: LayerOneDefinition<'a>,
    pub(super) layer_two: ReusedAttentionDefinition<'a>,
    pub(super) layer_three: LayerThreeDefinition<'a>,
    pub(super) layer_four: LayerFourDefinition<'a>,
    pub(super) head: FinalHead<'a>,
    pub(super) frequencies: &'a [RotaryFrequency],
    pub(super) max_tokens: NonZeroUsize,
}

/// Lends every decoded operand while the callback assembles a request model.
pub(super) fn with_parts<T>(
    config: &ArtifactConfig,
    tensors: &TensorStore,
    body: impl FnOnce(Parts<'_>) -> Result<T, ArtifactError>,
) -> Result<T, ArtifactError> {
    let startup_frequencies = frequencies(config, tensors, "rotary.startup")?;
    let shared_frequencies = frequencies(config, tensors, "rotary.shared")?;
    let routed_zero = routed(config, tensors, STARTUP_LAYER)?;
    let routed_one = routed(config, tensors, LAYER_ONE)?;
    let routed_two = routed(config, tensors, LAYER_TWO)?;
    let routed_three = routed(config, tensors, LAYER_THREE)?;
    let routed_four = routed(config, tensors, LAYER_FOUR)?;

    let tail_zero = tail(config, tensors, STARTUP_LAYER, &routed_zero)?;
    let tail_one = tail(config, tensors, LAYER_ONE, &routed_one)?;
    let tail_two = tail(config, tensors, LAYER_TWO, &routed_two)?;
    let tail_three = tail(config, tensors, LAYER_THREE, &routed_three)?;
    let tail_four = tail(config, tensors, LAYER_FOUR, &routed_four)?;

    let startup = StartupDefinition::new(
        tensors.u16("embed.weight", &[config.vocabulary, config.width])?,
        tensors.u16("layers.0.attn_norm.weight", &[config.width])?,
        config.norm_epsilon,
        attention_layout(config, None)?,
        attention_weights(config, tensors, STARTUP_LAYER)?,
        tail_zero,
        &startup_frequencies,
    );
    let blocks = [
        block(config, tensors, LAYER_ONE, &tail_one)?,
        block(config, tensors, LAYER_TWO, &tail_two)?,
        block(config, tensors, LAYER_THREE, &tail_three)?,
        block(config, tensors, LAYER_FOUR, &tail_four)?,
    ];
    let engrams = [
        EngramDefinition::new(
            engram_config(config, tensors, 0)?,
            engram_weights(config, tensors, LAYER_ONE)?,
        ),
        EngramDefinition::new(
            engram_config(config, tensors, 1)?,
            engram_weights(config, tensors, LAYER_THREE)?,
        ),
    ];
    let layer_one = layer_one(config, tensors)?;
    let layer_two = ReusedAttentionDefinition::new(
        attention_layout(config, Some((1, 2)))?,
        attention_weights(config, tensors, LAYER_TWO)?,
    );
    let layer_three = layer_three(config, tensors, 3)?;
    let layer_four = layer_four(config, tensors, 3)?;
    let head = FinalHead::new(
        tensors.u16("head.norm.weight", &[config.width])?,
        tensors.f32("head.weight", &[config.vocabulary, config.width])?,
        config.vocabulary,
        config.copies,
        config.norm_epsilon,
    )
    .map_err(invalid)?;
    body(Parts {
        startup,
        blocks,
        engrams,
        layer_one,
        layer_two,
        layer_three,
        layer_four,
        head,
        frequencies: &shared_frequencies,
        max_tokens: nonzero(config.max_tokens)?,
    })
}

/// Builds and runs the fixed five-block synthetic request over supplied IDs.
pub(super) fn run(
    config: &ArtifactConfig,
    tensors: &TensorStore,
    ids: &[i64],
    prefill: usize,
    execution: crate::indexer::query::IndexScoreExecution,
    key_preparation_execution: IndexKeyPreparationExecution,
    head_execution: FinalHeadExecution,
) -> Result<Vec<RequestStepOutput>, ArtifactError> {
    if ids.is_empty() || prefill == 0 || prefill > ids.len() {
        return Err(ArtifactError::Invalid(
            "prefill must be nonzero and no larger than the supplied IDs".into(),
        ));
    }
    with_request_model(
        config,
        tensors,
        execution,
        key_preparation_execution,
        head_execution,
        |model| {
            let mut request = RequestSession::new(model).map_err(ArtifactError::from)?;
            let mut outputs = Vec::with_capacity(ids.len() - prefill + 1);
            outputs.push(request.step(&ids[..prefill]).map_err(ArtifactError::from)?);
            for id in &ids[prefill..] {
                outputs.push(
                    request
                        .step(std::slice::from_ref(id))
                        .map_err(ArtifactError::from)?,
                );
            }
            Ok(outputs)
        },
    )
}

fn block<'a>(
    config: &ArtifactConfig,
    tensors: &'a TensorStore,
    layer: usize,
    tail: &BlockTailReference<'a>,
) -> Result<BlockDefinition<'a>, ArtifactError> {
    let norm = tensors.u16(&format!("layers.{layer}.attn_norm.weight"), &[config.width])?;
    let input = AttentionInput::new(norm, config.copies, config.norm_epsilon).map_err(invalid)?;
    Ok(BlockDefinition::new(input, *tail))
}

fn engram_config(
    config: &ArtifactConfig,
    tensors: &TensorStore,
    hash_layer: usize,
) -> Result<EngramSessionConfig, ArtifactError> {
    let hash_layout = EngramHashLayout::new(
        config.engram_ngram,
        config.engram_heads,
        2,
        config.engram_pad_id,
        tensors
            .i64(
                "engram.primes",
                &[2, config.engram_ngram - 1, config.engram_heads],
            )?
            .to_vec(),
        tensors
            .i64(
                "engram.offsets",
                &[2, (config.engram_ngram - 1) * config.engram_heads],
            )?
            .to_vec(),
        tensors
            .i64("engram.multipliers", &[2, config.engram_ngram])?
            .to_vec(),
    )
    .map_err(invalid)?;
    EngramSessionConfig::new(
        hash_layout,
        tensors
            .i64("engram.token_map", &[config.vocabulary])?
            .to_vec(),
        hash_layer,
        config.max_tokens,
        config.copies,
        config.width,
        config.engram_rows[hash_layer],
        config.engram_embedding_width,
        config.norm_epsilon,
        config.engram_gate_clamp,
    )
    .map_err(invalid)
}

fn engram_weights(
    config: &ArtifactConfig,
    tensors: &TensorStore,
    layer: usize,
) -> Result<EngramSessionWeights, ArtifactError> {
    let prefix = format!("layers.{layer}.engram");
    let columns = (config.engram_ngram - 1) * config.engram_heads;
    let reduction = columns * config.engram_embedding_width;
    let outputs = (config.copies + 1) * config.width;
    let rows = config.engram_rows[usize::from(layer == LAYER_THREE)];
    Ok(EngramSessionWeights::new(
        tensors
            .u8(
                &format!("{prefix}.embed.weight"),
                &[rows, config.engram_embedding_width],
            )?
            .to_vec(),
        tensors
            .u8(
                &format!("{prefix}.embed.scale"),
                &[rows, config.engram_embedding_width / 32],
            )?
            .to_vec(),
        tensors
            .u8(&format!("{prefix}.wkv.weight"), &[outputs, reduction])?
            .to_vec(),
        tensors
            .u8(
                &format!("{prefix}.wkv.scale"),
                &[outputs.div_ceil(32), reduction / 32],
            )?
            .to_vec(),
        tensors
            .u16(
                &format!("{prefix}.q_weight"),
                &[config.copies, config.width],
            )?
            .to_vec(),
        tensors
            .u16(
                &format!("{prefix}.k_weight"),
                &[config.copies, config.width],
            )?
            .to_vec(),
    ))
}

fn layer_one<'a>(
    config: &ArtifactConfig,
    tensors: &'a TensorStore,
) -> Result<LayerOneDefinition<'a>, ArtifactError> {
    let capacity = nonzero(config.max_tokens.div_ceil(2))?;
    let owner_layout = RatioTwoOwnerLayout::new(
        NonZeroUsize::MIN,
        nonzero(config.width)?,
        nonzero(config.head_dimension)?,
        nonzero(config.head_dimension)?,
        nonzero(config.rope_pairs)?,
        capacity,
        config.norm_epsilon,
    )
    .map_err(invalid)?;
    let owner_weights = RatioTwoOwnerWeights::new(
        tensors.f32(
            "layers.1.attn.compressor.wkv.weight",
            &[config.head_dimension, config.width],
        )?,
        tensors.f32(
            "layers.1.attn.compressor.wgate.weight",
            &[config.head_dimension, config.width],
        )?,
        IndexKeyWeights::new(
            tensors.u16(
                "layers.1.attn.indexer.wk.weight",
                &[config.head_dimension, config.head_dimension],
            )?,
            tensors.u16(
                "layers.1.attn.indexer.k_norm.weight",
                &[config.head_dimension],
            )?,
        ),
    );
    Ok(LayerOneDefinition::new(
        LayerOneConfig::new(
            owner_layout,
            attention_layout(config, Some((1, 2)))?,
            nonzero(config.index_topk)?,
        )
        .map_err(invalid)?,
        tensors.u16(
            "layers.1.attn.compressor.norm.weight",
            &[config.head_dimension],
        )?,
        owner_weights,
        query_weights(config, tensors, LAYER_ONE)?,
        query_layout(config)?,
        attention_weights(config, tensors, LAYER_ONE)?,
    ))
}

fn layer_three<'a>(
    config: &ArtifactConfig,
    tensors: &'a TensorStore,
    source_layer: u16,
) -> Result<LayerThreeDefinition<'a>, ArtifactError> {
    let owner_key_layout = IndexKeyLayout::new(
        NonZeroUsize::MIN,
        nonzero(config.head_dimension)?,
        nonzero(config.head_dimension)?,
        nonzero(config.rope_pairs)?,
        config.norm_epsilon,
    )
    .map_err(invalid)?;
    let owner_weights = RatioOneOwnerWeights::new(
        tensors.u16(
            "layers.3.attn.compressor.wkv.weight",
            &[config.head_dimension, config.width],
        )?,
        IndexKeyWeights::new(
            tensors.u16(
                "layers.3.attn.indexer.wk.weight",
                &[config.head_dimension, config.head_dimension],
            )?,
            tensors.u16(
                "layers.3.attn.indexer.k_norm.weight",
                &[config.head_dimension],
            )?,
        ),
    );
    Ok(LayerThreeDefinition::new(
        LayerThreeConfig::new(
            owner_key_layout,
            nonzero(config.width)?,
            nonzero(config.max_tokens)?,
            attention_layout(config, Some((source_layer, 1)))?,
            nonzero(config.window)?,
            nonzero(config.index_topk)?,
        )
        .map_err(invalid)?,
        tensors.u16(
            "layers.3.attn.compressor.norm.weight",
            &[config.head_dimension],
        )?,
        config.norm_epsilon,
        owner_weights,
        crate::reduced::CandidateProjector::new(
            query_weights(config, tensors, LAYER_THREE)?,
            query_layout(config)?,
            nonzero(config.head_dimension)?,
            nonzero(config.candidate_topk_blocks)?,
            nonzero(config.candidate_block_size)?,
        ),
        attention_weights(config, tensors, LAYER_THREE)?,
    ))
}

fn layer_four<'a>(
    config: &ArtifactConfig,
    tensors: &'a TensorStore,
    source_layer: u16,
) -> Result<LayerFourDefinition<'a>, ArtifactError> {
    Ok(LayerFourDefinition::new(
        LayerFourConfig::new(
            query_layout(config)?,
            attention_layout(config, Some((source_layer, 1)))?,
            nonzero(config.index_topk)?,
        )
        .map_err(invalid)?,
        query_weights(config, tensors, LAYER_FOUR)?,
        attention_weights(config, tensors, LAYER_FOUR)?,
    ))
}

fn tail<'a>(
    config: &ArtifactConfig,
    tensors: &'a TensorStore,
    layer: usize,
    routed: &'a [Fp4ExpertWeights<'a>],
) -> Result<BlockTailReference<'a>, ArtifactError> {
    let prefix = format!("layers.{layer}");
    let moe = MoEReference::new(
        MoEConfig::new(
            config.width,
            config.intermediate,
            config.swiglu_limit,
            config.active_experts,
            config.gate_temperature,
            config.normalize_topk,
            config.route_scale,
        )
        .map_err(invalid)?,
        tensors.u16(
            &format!("{prefix}.ffn.gate.weight"),
            &[config.routed_experts, config.width],
        )?,
        tensors.f32(&format!("{prefix}.ffn.gate.bias"), &[config.routed_experts])?,
        routed,
        shared(config, tensors, layer)?,
    )
    .map_err(invalid)?;
    let ffn = FfnSublayerReference::new(
        moe,
        tensors.u16(&format!("{prefix}.ffn_norm.weight"), &[config.width])?,
        tensors.f32(
            &format!("{prefix}.hc_ffn_fn"),
            &[hc_rows(config), config.copies * config.width],
        )?,
        scale3(tensors, &format!("{prefix}.hc_ffn_scale"))?,
        tensors.f32(&format!("{prefix}.hc_ffn_base"), &[hc_rows(config)])?,
        config.copies,
        config.norm_epsilon,
        config.hc_iterations,
        config.hc_epsilon,
    )
    .map_err(invalid)?;
    BlockTailReference::new(
        ffn,
        tensors.f32(
            &format!("{prefix}.hc_attn_fn"),
            &[hc_rows(config), config.copies * config.width],
        )?,
        scale3(tensors, &format!("{prefix}.hc_attn_scale"))?,
        tensors.f32(&format!("{prefix}.hc_attn_base"), &[hc_rows(config)])?,
        config.copies,
        config.norm_epsilon,
        config.hc_iterations,
        config.hc_epsilon,
    )
    .map_err(invalid)
}

fn routed<'a>(
    config: &ArtifactConfig,
    tensors: &'a TensorStore,
    layer: usize,
) -> Result<Vec<Fp4ExpertWeights<'a>>, ArtifactError> {
    (0..config.routed_experts)
        .map(|expert| {
            let prefix = format!("layers.{layer}.ffn.experts.{expert}");
            Fp4ExpertWeights::new(
                config.width,
                config.intermediate,
                tensors.u8(
                    &format!("{prefix}.w1.weight"),
                    &[config.intermediate, config.width / 2],
                )?,
                tensors.u8(
                    &format!("{prefix}.w1.scale"),
                    &[config.intermediate, config.width / 32],
                )?,
                tensors.u8(
                    &format!("{prefix}.w2.weight"),
                    &[config.width, config.intermediate / 2],
                )?,
                tensors.u8(
                    &format!("{prefix}.w2.scale"),
                    &[config.width, config.intermediate / 32],
                )?,
                tensors.u8(
                    &format!("{prefix}.w3.weight"),
                    &[config.intermediate, config.width / 2],
                )?,
                tensors.u8(
                    &format!("{prefix}.w3.scale"),
                    &[config.intermediate, config.width / 32],
                )?,
            )
            .map_err(invalid)
        })
        .collect()
}

fn shared<'a>(
    config: &ArtifactConfig,
    tensors: &'a TensorStore,
    layer: usize,
) -> Result<Fp8ExpertWeights<'a>, ArtifactError> {
    let prefix = format!("layers.{layer}.ffn.shared_experts");
    Fp8ExpertWeights::new(
        config.width,
        config.intermediate,
        tensors.u8(
            &format!("{prefix}.w1.weight"),
            &[config.intermediate, config.width],
        )?,
        tensors.u8(
            &format!("{prefix}.w1.scale"),
            &[config.intermediate.div_ceil(32), config.width / 32],
        )?,
        tensors.u8(
            &format!("{prefix}.w2.weight"),
            &[config.width, config.intermediate],
        )?,
        tensors.u8(
            &format!("{prefix}.w2.scale"),
            &[config.width.div_ceil(32), config.intermediate / 32],
        )?,
        tensors.u8(
            &format!("{prefix}.w3.weight"),
            &[config.intermediate, config.width],
        )?,
        tensors.u8(
            &format!("{prefix}.w3.scale"),
            &[config.intermediate.div_ceil(32), config.width / 32],
        )?,
    )
    .map_err(invalid)
}

fn attention_layout(
    config: &ArtifactConfig,
    compression: Option<(u16, usize)>,
) -> Result<LayerAttentionLayout, ArtifactError> {
    let scale = f32::from(u16::try_from(config.head_dimension).map_err(invalid)?)
        .sqrt()
        .recip();
    let common = (
        NonZeroUsize::MIN,
        nonzero(config.width)?,
        nonzero(config.heads)?,
        nonzero(config.head_dimension)?,
        nonzero(config.rope_pairs)?,
        nonzero(config.query_rank)?,
        nonzero(config.window)?,
        nonzero(config.output_groups)?,
        nonzero(config.output_rank)?,
    );
    match compression {
        Some((source, ratio)) => LayerAttentionLayout::new(
            common.0,
            common.1,
            common.2,
            common.3,
            common.4,
            common.5,
            common.6,
            common.7,
            common.8,
            source,
            nonzero(ratio)?,
            config.norm_epsilon,
            scale,
        ),
        None => LayerAttentionLayout::new_window_only(
            common.0,
            common.1,
            common.2,
            common.3,
            common.4,
            common.5,
            common.6,
            common.7,
            common.8,
            config.norm_epsilon,
            scale,
        ),
    }
    .map_err(invalid)
}

fn query_layout(config: &ArtifactConfig) -> Result<CandidateQueryLayout, ArtifactError> {
    CandidateQueryLayout::new(
        IndexQueryLayout::new(
            NonZeroUsize::MIN,
            nonzero(config.width)?,
            nonzero(config.query_rank)?,
            nonzero(config.index_heads)?,
            nonzero(config.head_dimension)?,
            nonzero(config.rope_pairs)?,
        )
        .map_err(invalid)?,
        config.norm_epsilon,
    )
    .map_err(invalid)
}

fn query_weights<'a>(
    config: &ArtifactConfig,
    tensors: &'a TensorStore,
    layer: usize,
) -> Result<CandidateQueryWeights<'a>, ArtifactError> {
    let prefix = format!("layers.{layer}.attn");
    Ok(CandidateQueryWeights {
        wq_a: fp8(
            tensors,
            &format!("{prefix}.wq_a"),
            config.query_rank,
            config.width,
        )?,
        q_norm: tensors.u16(&format!("{prefix}.q_norm.weight"), &[config.query_rank])?,
        index: IndexQueryWeights {
            wq_b_codes: tensors.u8(
                &format!("{prefix}.indexer.wq_b.weight"),
                &[
                    config.index_heads * config.head_dimension,
                    config.query_rank,
                ],
            )?,
            wq_b_scales: tensors.u8(
                &format!("{prefix}.indexer.wq_b.scale"),
                &[
                    config.index_heads * config.head_dimension / 32,
                    config.query_rank / 32,
                ],
            )?,
            weights_proj: tensors.u16(
                &format!("{prefix}.indexer.weights_proj.weight"),
                &[config.index_heads, config.width],
            )?,
        },
    })
}

fn attention_weights<'a>(
    config: &ArtifactConfig,
    tensors: &'a TensorStore,
    layer: usize,
) -> Result<LayerAttentionWeights<'a>, ArtifactError> {
    let prefix = format!("layers.{layer}.attn");
    Ok(LayerAttentionWeights {
        wq_a: fp8(
            tensors,
            &format!("{prefix}.wq_a"),
            config.query_rank,
            config.width,
        )?,
        q_norm: tensors.u16(&format!("{prefix}.q_norm.weight"), &[config.query_rank])?,
        wq_b: fp8(
            tensors,
            &format!("{prefix}.wq_b"),
            config.heads * config.head_dimension,
            config.query_rank,
        )?,
        wkv: fp8(
            tensors,
            &format!("{prefix}.wkv"),
            config.head_dimension,
            config.width,
        )?,
        kv_norm: tensors.u16(
            &format!("{prefix}.kv_norm.weight"),
            &[config.head_dimension],
        )?,
        attn_sink: tensors.f32(&format!("{prefix}.attn_sink"), &[config.heads])?,
        wo_a: tensors.u16(
            &format!("{prefix}.wo_a.weight"),
            &[
                config.output_groups * config.output_rank,
                config.heads / config.output_groups * config.head_dimension,
            ],
        )?,
        wo_b: fp8(
            tensors,
            &format!("{prefix}.wo_b"),
            config.width,
            config.output_groups * config.output_rank,
        )?,
    })
}

fn fp8<'a>(
    tensors: &'a TensorStore,
    prefix: &str,
    output: usize,
    input: usize,
) -> Result<Fp8Projection<'a>, ArtifactError> {
    Ok(Fp8Projection {
        codes: tensors.u8(&format!("{prefix}.weight"), &[output, input])?,
        scales: tensors.u8(
            &format!("{prefix}.scale"),
            &[output.div_ceil(32), input / 32],
        )?,
    })
}

fn frequencies(
    config: &ArtifactConfig,
    tensors: &TensorStore,
    name: &str,
) -> Result<Vec<RotaryFrequency>, ArtifactError> {
    tensors
        .f32(name, &[config.max_tokens, config.rope_pairs, 2])?
        .chunks_exact(2)
        .map(|pair| RotaryFrequency::new(pair[0], pair[1]).map_err(invalid))
        .collect()
}

fn scale3<'a>(tensors: &'a TensorStore, name: &str) -> Result<&'a [f32; 3], ArtifactError> {
    tensors
        .f32(name, &[3])?
        .try_into()
        .map_err(|_| ArtifactError::Invalid(format!("{name} must contain three values")))
}

fn hc_rows(config: &ArtifactConfig) -> usize {
    (config.copies + 2) * config.copies
}

fn nonzero(value: usize) -> Result<NonZeroUsize, ArtifactError> {
    NonZeroUsize::new(value)
        .ok_or_else(|| ArtifactError::Invalid("artifact geometry must be nonzero".into()))
}

fn invalid(error: impl std::fmt::Display) -> ArtifactError {
    ArtifactError::Invalid(error.to_string())
}

#[cfg(test)]
mod tests {
    use std::{path::Path, process::Command};

    use super::*;
    use crate::reduced::{
        HeadPositions, LayerKind, RequestError, ScheduleError, ScheduledAttentionOutput,
        ScheduledLayer, StepSources, artifact::ReducedArtifact,
    };

    fn artifact() -> ReducedArtifact {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../..");
        let output = Command::new("python3")
            .arg(root.join("scripts/export_v41_reduced_artifact.py"))
            .arg("--source")
            .arg(root.join("fixtures/deepseek-v41/reduced-runner-reference.json"))
            .args(["--output", "-"])
            .output()
            .expect("repository Python exporter launches");
        assert!(
            output.status.success(),
            "exporter failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        ReducedArtifact::parse(&output.stdout).expect("exported artifact parses")
    }

    /// A prefill of three, then decodes through an incomplete ratio-two group.
    fn ids(config: &ArtifactConfig) -> Vec<i64> {
        assert!(
            config.max_tokens >= 5,
            "decode must reach an incomplete group"
        );
        (0..config.max_tokens.min(7))
            .map(|index| i64::try_from((index * 7 + 1) % config.vocabulary).unwrap())
            .collect()
    }

    fn run(model: &RequestModel<'_>, ids: &[i64]) -> Vec<RequestStepOutput> {
        let mut request = RequestSession::new(model).unwrap();
        let mut outputs = vec![request.step(&ids[..3]).unwrap()];
        for id in &ids[3..] {
            outputs.push(request.step(std::slice::from_ref(id)).unwrap());
        }
        request.restart().unwrap();
        let mut rerun = vec![request.step(&ids[..3]).unwrap()];
        for id in &ids[3..] {
            rerun.push(request.step(std::slice::from_ref(id)).unwrap());
        }
        assert_eq!(
            format!("{outputs:?}"),
            format!("{rerun:?}"),
            "restart is pristine"
        );
        outputs
    }

    fn reduced_schedule<'a>(parts: &Parts<'a>) -> Vec<ScheduledLayer<'a>> {
        let [one, two, three, four] = parts.blocks;
        vec![
            ScheduledLayer::new(LayerKind::RatioTwoOwner(parts.layer_one), one).with_engram(0),
            ScheduledLayer::new(LayerKind::RatioTwoConsumer(parts.layer_two), two),
            ScheduledLayer::new(LayerKind::RatioOneOwner(parts.layer_three), three).with_engram(1),
            ScheduledLayer::new(LayerKind::RatioOneIndexer(parts.layer_four), four),
        ]
    }

    fn from_schedule<'a>(
        parts: &Parts<'a>,
        layers: Vec<ScheduledLayer<'a>>,
    ) -> Result<RequestModel<'a>, RequestError> {
        RequestModel::from_schedule(
            parts.startup,
            layers,
            parts.engrams.to_vec(),
            parts.head,
            parts.frequencies,
            parts.max_tokens,
        )
    }

    #[test]
    fn reduced_schedule_constructor_matches_fixed_constructor_bitwise() {
        let artifact = artifact();
        let ids = ids(&artifact.config);
        with_parts(&artifact.config, &artifact.tensors, |parts| {
            let fixed = RequestModel::new(
                parts.startup,
                parts.blocks,
                parts.engrams.clone(),
                parts.layer_one,
                parts.layer_two,
                parts.layer_three,
                parts.layer_four,
                parts.head,
                parts.frequencies,
                parts.max_tokens,
            )
            .unwrap();
            let scheduled = from_schedule(&parts, reduced_schedule(&parts)).unwrap();
            let (fixed, scheduled) = (run(&fixed, &ids), run(&scheduled, &ids));
            assert_eq!(fixed.len(), ids.len() - 2);
            for (fixed, scheduled) in fixed.iter().zip(&scheduled) {
                assert_eq!(format!("{fixed:?}"), format!("{scheduled:?}"));
                for (left, right) in fixed.heads().iter().zip(scheduled.heads()) {
                    assert!(
                        left.logits()
                            .iter()
                            .map(|value| value.to_bits())
                            .eq(right.logits().iter().map(|value| value.to_bits()))
                    );
                }
            }
            Ok(())
        })
        .unwrap();
    }

    /// Serves rows of the artifact's dense embedding table and records requests.
    struct TableRows<'a> {
        table: &'a [u16],
        width: usize,
        requests: std::cell::RefCell<Vec<Vec<usize>>>,
    }

    impl crate::reduced::EmbeddingRowSource for TableRows<'_> {
        fn read_rows(
            &self,
            rows: &[usize],
            output: &mut [u16],
        ) -> Result<(), crate::reduced::StartupSessionError> {
            self.requests.borrow_mut().push(rows.to_vec());
            for (row, out) in rows.iter().zip(output.chunks_exact_mut(self.width)) {
                out.copy_from_slice(&self.table[row * self.width..(row + 1) * self.width]);
            }
            Ok(())
        }
    }

    #[test]
    fn row_source_startup_reproduces_the_dense_table_startup() {
        let artifact = artifact();
        let config = &artifact.config;
        let ids = ids(config);
        with_parts(config, &artifact.tensors, |parts| {
            let dense = from_schedule(&parts, reduced_schedule(&parts)).unwrap();
            let rows = RequestModel::from_schedule(
                parts.startup.reading_rows(config.vocabulary),
                reduced_schedule(&parts),
                parts.engrams.to_vec(),
                parts.head,
                parts.frequencies,
                parts.max_tokens,
            )
            .unwrap();
            let source = TableRows {
                table: artifact
                    .tensors
                    .u16("embed.weight", &[config.vocabulary, config.width])
                    .unwrap(),
                width: config.width,
                requests: std::cell::RefCell::default(),
            };
            let sources = StepSources {
                embedding_rows: Some(&source),
                ..StepSources::default()
            };
            let mut dense_session = RequestSession::new(&dense).unwrap();
            let mut rows_session = RequestSession::new(&rows).unwrap();
            let chunks: Vec<&[i64]> = std::iter::once(&ids[..3])
                .chain(ids[3..].chunks(1))
                .collect();
            for &chunk in &chunks {
                let expected = dense_session.step(chunk).unwrap();
                let actual = rows_session.step_with_sources(chunk, sources).unwrap();
                assert_eq!(format!("{expected:?}"), format!("{actual:?}"));
            }
            // Each step reads exactly its distinct token rows, ascending.
            let expected: Vec<Vec<usize>> = chunks
                .iter()
                .map(|chunk| {
                    let mut rows: Vec<usize> = chunk
                        .iter()
                        .map(|&id| usize::try_from(id).unwrap())
                        .collect();
                    rows.sort_unstable();
                    rows.dedup();
                    rows
                })
                .collect();
            assert_eq!(*source.requests.borrow(), expected);

            // Startup alone (no scheduled layers) with unsorted, repeated IDs.
            let vocabulary = i64::try_from(config.vocabulary).unwrap();
            let repeated: Vec<i64> = [7, 2, 7, 2].iter().map(|id| id % vocabulary).collect();
            let startup_only = |startup| {
                RequestModel::from_schedule(
                    startup,
                    Vec::new(),
                    Vec::new(),
                    parts.head,
                    parts.frequencies,
                    parts.max_tokens,
                )
                .unwrap()
            };
            let (dense_only, rows_only) = (
                startup_only(parts.startup),
                startup_only(parts.startup.reading_rows(config.vocabulary)),
            );
            let (mut dense_only, mut rows_only) = (
                RequestSession::new(&dense_only).unwrap(),
                RequestSession::new(&rows_only).unwrap(),
            );
            source.requests.borrow_mut().clear();
            for chunk in [&repeated[..3], &repeated[3..]] {
                assert_eq!(
                    format!("{:?}", dense_only.step(chunk).unwrap()),
                    format!("{:?}", rows_only.step_with_sources(chunk, sources).unwrap())
                );
            }
            let row = |id: i64| usize::try_from(id).unwrap();
            let mut prefill = vec![row(repeated[0]), row(repeated[1])];
            prefill.sort_unstable();
            prefill.dedup();
            assert_eq!(*source.requests.borrow(), [prefill, vec![row(repeated[3])]]);

            let mut missing = RequestSession::new(&rows).unwrap();
            assert!(matches!(
                missing.step(&ids[..3]),
                Err(RequestError::Startup(
                    crate::reduced::StartupSessionError::MissingEmbeddingRows
                ))
            ));
            Ok(())
        })
        .unwrap();
    }

    #[test]
    fn chunked_prefill_equals_a_single_prefill_bit_for_bit() {
        let artifact = artifact();
        let ids = ids(&artifact.config);
        with_parts(&artifact.config, &artifact.tensors, |parts| {
            let single = from_schedule(&parts, reduced_schedule(&parts)).unwrap();
            let expected = RequestSession::new(&single).unwrap().step(&ids).unwrap();
            let rows = expected.residual().len() / ids.len();
            let last_row = &expected.residual()[expected.residual().len() - rows..];
            let bits = |output: &RequestStepOutput| -> Vec<u32> {
                let head = output.heads().last().unwrap();
                head.logits().iter().map(|value| value.to_bits()).collect()
            };
            for step in [1, 2, 3, ids.len() - 1, ids.len()] {
                let chunked = from_schedule(&parts, reduced_schedule(&parts))
                    .unwrap()
                    .with_max_step_tokens(NonZeroUsize::new(step).unwrap());
                let mut session = RequestSession::new(&chunked).unwrap();
                let last = session
                    .prefill_with_sources(&ids, StepSources::default())
                    .unwrap();
                assert_eq!(session.next_start(), ids.len());
                let tail = &last.residual()[last.residual().len() - rows..];
                assert_eq!(tail, last_row, "max step {step}");
                assert_eq!(bits(&last), bits(&expected), "max step {step}");
            }
            Ok(())
        })
        .unwrap();
    }

    #[test]
    fn last_position_heads_equal_the_last_of_all_position_heads() {
        let artifact = artifact();
        let ids = ids(&artifact.config);
        with_parts(&artifact.config, &artifact.tensors, |parts| {
            let all = from_schedule(&parts, reduced_schedule(&parts)).unwrap();
            assert_eq!(all.head_positions(), HeadPositions::All);
            let last = from_schedule(&parts, reduced_schedule(&parts))
                .unwrap()
                .with_head_positions(HeadPositions::Last);
            let (all, last) = (run(&all, &ids), run(&last, &ids));
            assert_eq!(
                all[0].heads().len(),
                3,
                "prefill heads cover every position"
            );
            for (all, last) in all.iter().zip(&last) {
                assert_eq!(last.heads().len(), 1);
                assert_eq!(
                    format!("{:?}", all.heads().last()),
                    format!("{:?}", last.heads().first())
                );
                assert_eq!(
                    format!("{:?}", all.layers()),
                    format!("{:?}", last.layers())
                );
            }
            Ok(())
        })
        .unwrap();
    }

    #[test]
    fn window_only_and_ratio_one_consumer_layers_extend_the_reduced_schedule() {
        let artifact = artifact();
        let (config, tensors) = (&artifact.config, &artifact.tensors);
        let ids = ids(config);
        with_parts(config, tensors, |parts| {
            let mut layers = reduced_schedule(&parts);
            // Layer 5 is window-only between the ratio-one owner and its
            // consumer at layer 6, which must still borrow L3's KV and L4's indices.
            layers.push(ScheduledLayer::new(
                LayerKind::WindowOnly(ReusedAttentionDefinition::new(
                    attention_layout(config, None)?,
                    attention_weights(config, tensors, LAYER_ONE)?,
                )),
                parts.blocks[0],
            ));
            layers.push(ScheduledLayer::new(
                LayerKind::RatioOneConsumer(ReusedAttentionDefinition::new(
                    attention_layout(config, Some((3, 1)))?,
                    attention_weights(config, tensors, LAYER_TWO)?,
                )),
                parts.blocks[1],
            ));
            let extended = run(&from_schedule(&parts, layers).unwrap(), &ids);
            let reduced = run(
                &from_schedule(&parts, reduced_schedule(&parts)).unwrap(),
                &ids,
            );
            for (extended, reduced) in extended.iter().zip(&reduced) {
                assert_eq!(extended.layers().len(), 6);
                assert_eq!(
                    format!("{:?}", &extended.layers()[..4]),
                    format!("{:?}", reduced.layers()),
                    "appended layers leave the producers they read untouched"
                );
                assert!(matches!(
                    extended.layers()[4].attention(),
                    ScheduledAttentionOutput::WindowOnly(_)
                ));
                assert!(matches!(
                    extended.layers()[5].attention(),
                    ScheduledAttentionOutput::RatioOneConsumer(_)
                ));
                assert_ne!(extended.residual(), reduced.residual());
                assert!(
                    extended
                        .heads()
                        .iter()
                        .all(|head| head.logits().iter().all(|value| value.is_finite()))
                );
            }
            Ok(())
        })
        .unwrap();
    }

    #[test]
    fn ratio_two_owners_score_the_scheduled_ratio_one_owner_keys() {
        let artifact = artifact();
        let (config, tensors) = (&artifact.config, &artifact.tensors);
        let ids = ids(config);
        with_parts(config, tensors, |parts| {
            let [one, two, three, _] = parts.blocks;
            let reused = |compression, layer| -> Result<_, ArtifactError> {
                Ok(ReusedAttentionDefinition::new(
                    attention_layout(config, compression)?,
                    attention_weights(config, tensors, layer)?,
                ))
            };
            // V4.1-shaped: the ratio-one owner sits at layer 4, after a
            // window-only layer, so incomplete ratio-two groups must accept
            // layer-4 keys rather than the reduced schedule's layer 3.
            let layers = vec![
                ScheduledLayer::new(LayerKind::RatioTwoOwner(parts.layer_one), one).with_engram(0),
                ScheduledLayer::new(LayerKind::RatioTwoConsumer(parts.layer_two), two),
                ScheduledLayer::new(LayerKind::WindowOnly(reused(None, LAYER_TWO)?), two),
                ScheduledLayer::new(
                    LayerKind::RatioOneOwner(layer_three(config, tensors, 4)?),
                    three,
                )
                .with_engram(1),
                ScheduledLayer::new(
                    LayerKind::RatioOneConsumer(reused(Some((4, 1)), LAYER_TWO)?),
                    two,
                ),
            ];
            let model = from_schedule(&parts, layers).unwrap();
            let mut request = RequestSession::new(&model).unwrap();
            request.step(&ids[..3]).unwrap();
            let complete = request.step(&ids[3..4]).unwrap();
            let ScheduledAttentionOutput::RatioOneOwner(owner) = complete.layers()[3].attention()
            else {
                panic!("layer 4 is the ratio-one owner");
            };
            assert_eq!(owner.publication().source_layer(), 4);
            // Position 4 opens a ratio-two group, so layer 1 scores with the
            // first two of layer 4's previous-call keys.
            let partial = request.step(&ids[4..5]).unwrap();
            assert!(partial.layer_one().owner().latent().is_none());
            let used = partial.layer_one().score_key_prefix().len();
            assert_eq!(used, 2 * config.head_dimension);
            assert_eq!(
                partial.layer_one().score_key_prefix(),
                &owner.key_prefix()[..used]
            );
            Ok(())
        })
        .unwrap();
    }

    /// Lends one layer's routed experts from the artifact table, counting calls.
    struct TableSource<'a> {
        experts: Vec<Fp4ExpertWeights<'a>>,
        calls: std::cell::Cell<usize>,
        available: bool,
    }

    impl<'a> TableSource<'a> {
        fn new(experts: Vec<Fp4ExpertWeights<'a>>, available: bool) -> Self {
            Self {
                experts,
                calls: std::cell::Cell::new(0),
                available,
            }
        }
    }

    impl crate::moe::RoutedExpertSource for TableSource<'_> {
        fn with_expert(
            &self,
            index: usize,
            run: &mut dyn FnMut(Fp4ExpertWeights<'_>) -> Result<Vec<u16>, crate::moe::MoEError>,
        ) -> Result<Vec<u16>, crate::moe::MoEError> {
            self.calls.set(self.calls.get() + 1);
            match self.experts.get(index) {
                Some(expert) if self.available => run(*expert),
                _ => Err(crate::moe::MoEError::ExpertUnavailable {
                    index,
                    reason: String::from("withheld by test source"),
                }),
            }
        }
    }

    /// Lends one Engram definition's embedding rows from the artifact table.
    struct RowSource<'a> {
        codes: &'a [u8],
        scales: &'a [u8],
        width: usize,
        calls: std::cell::Cell<usize>,
        available: bool,
    }

    impl<'a> RowSource<'a> {
        fn new(config: &ArtifactConfig, tensors: &'a TensorStore, layer: usize) -> Self {
            let prefix = format!("layers.{layer}.engram");
            let rows = config.engram_rows[usize::from(layer == LAYER_THREE)];
            let width = config.engram_embedding_width;
            Self {
                codes: tensors
                    .u8(&format!("{prefix}.embed.weight"), &[rows, width])
                    .unwrap(),
                scales: tensors
                    .u8(&format!("{prefix}.embed.scale"), &[rows, width / 32])
                    .unwrap(),
                width,
                calls: std::cell::Cell::new(0),
                available: true,
            }
        }
    }

    impl crate::engram::embedding::EngramRowSource for RowSource<'_> {
        fn read_rows(
            &self,
            rows: &[usize],
            codes: &mut [u8],
            scales: &mut [u8],
        ) -> Result<(), crate::engram::embedding::EngramEmbeddingError> {
            self.calls.set(self.calls.get() + 1);
            if !self.available {
                return Err(
                    crate::engram::embedding::EngramEmbeddingError::RowsUnavailable {
                        reason: String::from("withheld by test source"),
                    },
                );
            }
            let scale_width = self.width / 32;
            for (index, &row) in rows.iter().enumerate() {
                codes[index * self.width..(index + 1) * self.width]
                    .copy_from_slice(&self.codes[row * self.width..(row + 1) * self.width]);
                scales[index * scale_width..(index + 1) * scale_width]
                    .copy_from_slice(&self.scales[row * scale_width..(row + 1) * scale_width]);
            }
            Ok(())
        }
    }

    #[test]
    #[allow(
        clippy::too_many_lines,
        reason = "the equivalence run and each fail-closed source share one model"
    )]
    fn step_sources_reproduce_the_owned_expert_and_engram_tables() {
        use crate::{
            engram::embedding::EngramRowSource, moe::RoutedExpertSource, reduced::StepSources,
        };

        let artifact = artifact();
        let (config, tensors) = (&artifact.config, &artifact.tensors);
        let ids = ids(config);
        let experts = [STARTUP_LAYER, LAYER_ONE, LAYER_TWO, LAYER_THREE, LAYER_FOUR]
            .map(|layer| TableSource::new(routed(config, tensors, layer).unwrap(), true));
        let rows = [LAYER_ONE, LAYER_THREE].map(|layer| RowSource::new(config, tensors, layer));
        with_parts(config, tensors, |parts| {
            let model = from_schedule(&parts, reduced_schedule(&parts)).unwrap();
            // Every model layer, startup included, and both Engrams fetch.
            let expert_refs: Vec<Option<&dyn RoutedExpertSource>> = experts
                .iter()
                .map(|source| Some(source as &dyn RoutedExpertSource))
                .collect();
            let row_refs: Vec<Option<&dyn EngramRowSource>> = rows
                .iter()
                .map(|source| Some(source as &dyn EngramRowSource))
                .collect();
            let sources = StepSources {
                experts: &expert_refs,
                engram_rows: &row_refs,
                embedding_rows: None,
            };
            let mut owned = RequestSession::new(&model).unwrap();
            let mut fetched = RequestSession::new(&model).unwrap();

            // Malformed slices are rejected before admission.
            let short_experts = StepSources {
                experts: &expert_refs[1..],
                ..sources
            };
            assert!(matches!(
                fetched.step_with_sources(&ids[..3], short_experts),
                Err(RequestError::ExpertSourceCount {
                    expected: 5,
                    actual: 4
                })
            ));
            let short_rows = StepSources {
                engram_rows: &row_refs[1..],
                ..sources
            };
            assert!(matches!(
                fetched.step_with_sources(&ids[..3], short_rows),
                Err(RequestError::EngramRowSourceCount {
                    expected: 2,
                    actual: 1
                })
            ));
            assert!(!fetched.is_poisoned());

            let mut chunks = vec![&ids[..3]];
            chunks.extend(ids[3..].chunks(1));
            for chunk in chunks {
                let calls: Vec<usize> = experts
                    .iter()
                    .map(|source| source.calls.get())
                    .chain(rows.iter().map(|source| source.calls.get()))
                    .collect();
                let left = owned.step(chunk).unwrap();
                let right = fetched.step_with_sources(chunk, sources).unwrap();
                assert_eq!(format!("{left:?}"), format!("{right:?}"));
                let after = experts
                    .iter()
                    .map(|source| source.calls.get())
                    .chain(rows.iter().map(|source| source.calls.get()));
                for (after, before) in after.zip(calls) {
                    assert!(after > before, "every source was consulted");
                }
            }

            // A withheld source fails its stage and poisons the session.
            let fails = |sources| {
                let mut request = RequestSession::new(&model).unwrap();
                let error = request.step_with_sources(&ids[..3], sources).unwrap_err();
                assert!(request.is_poisoned());
                error
            };
            let withheld = TableSource::new(routed(config, tensors, LAYER_TWO)?, false);
            let mut failing = expert_refs.clone();
            failing[2] = Some(&withheld);
            let experts_failing = StepSources {
                experts: &failing,
                ..sources
            };
            assert!(matches!(fails(experts_failing), RequestError::Tail(_)));
            let withheld = TableSource::new(routed(config, tensors, STARTUP_LAYER)?, false);
            let mut startup_experts = expert_refs.clone();
            startup_experts[0] = Some(&withheld);
            let startup_failing = StepSources {
                experts: &startup_experts,
                ..sources
            };
            assert!(matches!(fails(startup_failing), RequestError::Startup(_)));
            let withheld = RowSource {
                available: false,
                ..RowSource::new(config, tensors, LAYER_THREE)
            };
            let rows_failing = [row_refs[0], Some(&withheld as &dyn EngramRowSource)];
            let rows_failing = StepSources {
                engram_rows: &rows_failing,
                ..sources
            };
            assert!(matches!(fails(rows_failing), RequestError::Engram(_)));
            assert!(withheld.calls.get() > 0);
            Ok(())
        })
        .unwrap();
    }

    #[test]
    #[allow(
        clippy::too_many_lines,
        reason = "one rejection case per schedule error stays in one table"
    )]
    fn invalid_schedules_are_rejected_before_allocation() {
        let artifact = artifact();
        let (config, tensors) = (&artifact.config, &artifact.tensors);
        with_parts(config, tensors, |parts| {
            let [one, two, three, four] = parts.blocks;
            let window = ReusedAttentionDefinition::new(
                attention_layout(config, None)?,
                attention_weights(config, tensors, LAYER_TWO)?,
            );
            let reject = |layers: Vec<ScheduledLayer<'_>>| match from_schedule(&parts, layers) {
                Err(RequestError::Schedule { layer, reason }) => (layer, reason),
                other => panic!("expected a schedule rejection, got {other:?}"),
            };
            let owner_two = ScheduledLayer::new(LayerKind::RatioTwoOwner(parts.layer_one), one);
            let consumer_two =
                ScheduledLayer::new(LayerKind::RatioTwoConsumer(parts.layer_two), two);
            let owner_one = ScheduledLayer::new(LayerKind::RatioOneOwner(parts.layer_three), three);
            let indexer = ScheduledLayer::new(LayerKind::RatioOneIndexer(parts.layer_four), four);

            assert_eq!(
                reject(vec![consumer_two]),
                (
                    1,
                    ScheduleError::MissingProducer {
                        source_layer: 1,
                        ratio: 2
                    }
                )
            );
            assert_eq!(
                reject(vec![owner_two, consumer_two, indexer]),
                (
                    3,
                    ScheduleError::MissingProducer {
                        source_layer: 3,
                        ratio: 1
                    }
                )
            );
            assert_eq!(
                reject(vec![owner_two, owner_one]),
                (
                    2,
                    ScheduleError::OwnerSource {
                        expected: 2,
                        actual: 3
                    }
                )
            );
            assert_eq!(
                reject(vec![
                    owner_two,
                    consumer_two,
                    owner_one,
                    indexer,
                    consumer_two
                ]),
                (5, ScheduleError::RatioOrder)
            );
            assert_eq!(
                reject(vec![owner_two, consumer_two]),
                (2, ScheduleError::MissingRatioOneOwner)
            );
            assert_eq!(
                reject(vec![owner_two.with_engram(2)]),
                (
                    1,
                    ScheduleError::EngramIndex {
                        index: 2,
                        available: 2
                    }
                )
            );
            assert_eq!(
                reject(vec![ScheduledLayer::new(
                    LayerKind::WindowOnly(parts.layer_two),
                    one
                )]),
                (1, ScheduleError::WindowLayoutCompressed)
            );
            assert_eq!(
                reject(vec![
                    owner_two,
                    ScheduledLayer::new(LayerKind::RatioTwoConsumer(window), two)
                ]),
                (2, ScheduleError::MissingCompression)
            );
            assert_eq!(
                reject(vec![
                    owner_two,
                    consumer_two,
                    owner_one,
                    ScheduledLayer::new(LayerKind::RatioOneConsumer(parts.layer_two), four),
                ]),
                (
                    4,
                    ScheduleError::CompressionRatio {
                        expected: 1,
                        actual: 2
                    }
                )
            );
            Ok(())
        })
        .unwrap();
    }
}
