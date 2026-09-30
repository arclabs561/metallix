//! Fixed reduced-request numerical assembly from validated artifact tensors.

use std::{collections::BTreeSet, num::NonZeroUsize};

use crate::{
    RotaryFrequency,
    attention::layer::{Fp8Projection, LayerAttentionLayout, LayerAttentionWeights},
    engram::EngramHashLayout,
    ffn::FfnSublayerReference,
    indexer::{
        key::{IndexKeyLayout, IndexKeyRotaryExecution, IndexKeyWeights},
        owner::RatioOneOwnerWeights,
        query::{CandidateQueryLayout, CandidateQueryWeights, IndexQueryLayout, IndexQueryWeights},
    },
    moe::{Fp4ExpertWeights, Fp8ExpertWeights, MoEConfig, MoEReference},
    reduced::{
        AttentionInput, BlockDefinition, BlockTailReference, EngramDefinition, EngramSessionConfig,
        EngramSessionWeights, FinalHead, LayerFourConfig, LayerFourDefinition, LayerOneConfig,
        LayerOneDefinition, LayerThreeConfig, LayerThreeDefinition, RatioTwoOwnerLayout,
        RatioTwoOwnerWeights, RequestModel, RequestSession, RequestStepOutput,
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

/// Builds and runs the fixed five-block synthetic request without fixture data.
pub(super) fn run(
    config: &ArtifactConfig,
    tensors: &TensorStore,
    ids: &[i64],
    prefill: usize,
    execution: crate::indexer::query::IndexScoreExecution,
    key_rotary_execution: IndexKeyRotaryExecution,
) -> Result<Vec<RequestStepOutput>, ArtifactError> {
    if ids.is_empty() || prefill == 0 || prefill > ids.len() {
        return Err(ArtifactError::Invalid(
            "prefill must be nonzero and no larger than the supplied IDs".into(),
        ));
    }

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
    let layer_three = layer_three(config, tensors)?;
    let layer_four = layer_four(config, tensors)?;
    let head = FinalHead::new(
        tensors.u16("head.norm.weight", &[config.width])?,
        tensors.f32("head.weight", &[config.vocabulary, config.width])?,
        config.vocabulary,
        config.copies,
        config.norm_epsilon,
    )
    .map_err(invalid)?;
    let model = RequestModel::new(
        startup,
        blocks,
        engrams,
        layer_one,
        layer_two,
        layer_three,
        layer_four,
        head,
        &shared_frequencies,
        nonzero(config.max_tokens)?,
    )
    .map_err(ArtifactError::from)?
    .with_score_execution(execution)
    .with_key_rotary_execution(key_rotary_execution);
    let mut request = RequestSession::new(&model).map_err(ArtifactError::from)?;
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
            attention_layout(config, Some((3, 1)))?,
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
) -> Result<LayerFourDefinition<'a>, ArtifactError> {
    Ok(LayerFourDefinition::new(
        LayerFourConfig::new(
            query_layout(config)?,
            attention_layout(config, Some((3, 1)))?,
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
