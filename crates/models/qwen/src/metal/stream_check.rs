//! Synchronous, layer-at-a-time Qwen3 qualification from bounded checkpoint reads.

use std::{path::Path, time::Instant};

use mlx_rs::{Array, StreamOrDevice};

use super::{
    Qwen3MetalLoadError, Qwen3MlxWeights, decode_bf16,
    embedding_check::{EmbeddingPlan, load_embedding},
    layer_check::{LAYER_SUFFIXES, load_layer, plan_weight_bytes},
    projection_check::{ProjectionPlan, TiledProjection, compare_projection_logits, project_tiled},
};
use crate::{
    checkpoint::{Qwen3CheckpointError, Qwen3CheckpointInspection},
    forward::{
        Qwen3ForwardConfig, Qwen3LayerKv, final_rms_norm, forward_cached_layer, forward_layer,
    },
};

const FINAL_NORM: &str = "model.norm.weight";
const ABSOLUTE_TOLERANCE: f32 = 5e-4;
const RELATIVE_TOLERANCE: f32 = 1e-4;

/// Per-layer candidate evidence in a synchronous streamed qualification.
#[derive(Debug, serde::Serialize)]
pub struct Qwen3StreamLayer {
    layer: usize,
    raw_payload_bytes: u64,
    planned_weight_and_staging_bytes: u64,
    load_ms: f64,
    execute_and_readback_ms: f64,
}

/// Evidence from a complete synchronous Qwen3 stream and resident control.
#[derive(Debug, serde::Serialize)]
pub struct Qwen3StreamCheck {
    schema_version: u32,
    operation: &'static str,
    input_ids: Vec<i32>,
    input_recipe: &'static str,
    hidden_shape: [usize; 3],
    embedding_raw_payload_bytes: u64,
    embedding_planned_weight_and_staging_bytes: u64,
    embedding_load_and_readback_ms: f64,
    layers: Vec<Qwen3StreamLayer>,
    final_norm_raw_payload_bytes: u64,
    final_norm_planned_weight_and_staging_bytes: u64,
    final_norm_execute_and_readback_ms: f64,
    projection_raw_payload_bytes: u64,
    projection_max_tile_raw_payload_bytes: u64,
    projection_planned_weight_and_staging_bytes: u64,
    projection_tile_count: usize,
    projection_load_ms: f64,
    projection_execute_and_readback_ms: f64,
    total_raw_payload_bytes: u64,
    planned_peak_weight_and_staging_bytes: u64,
    max_weight_bytes: u64,
    compared_logits: usize,
    absolute_tolerance: f32,
    relative_tolerance: f32,
    maximum_absolute_error: f32,
    mismatches: usize,
    candidate_total_ms: f64,
    resident_reference_ms: f64,
    scope: &'static str,
}

/// Candidate-only evidence from a complete synchronous Qwen3 stream.
///
/// Unlike [`Qwen3StreamCheck`], this report deliberately does not construct a
/// resident checkpoint oracle. Its logits are observable diagnostic output,
/// not a parity result.
#[derive(Debug, serde::Serialize)]
pub struct Qwen3StreamCandidateReport {
    schema_version: u32,
    operation: &'static str,
    input_ids: Vec<i32>,
    input_recipe: &'static str,
    hidden_shape: [usize; 3],
    embedding_raw_payload_bytes: u64,
    embedding_planned_weight_and_staging_bytes: u64,
    embedding_load_and_readback_ms: f64,
    layers: Vec<Qwen3StreamLayer>,
    final_norm_raw_payload_bytes: u64,
    final_norm_planned_weight_and_staging_bytes: u64,
    final_norm_execute_and_readback_ms: f64,
    projection_raw_payload_bytes: u64,
    projection_max_tile_raw_payload_bytes: u64,
    projection_planned_weight_and_staging_bytes: u64,
    projection_tile_count: usize,
    projection_load_ms: f64,
    projection_execute_and_readback_ms: f64,
    total_raw_payload_bytes: u64,
    planned_peak_weight_and_staging_bytes: u64,
    max_weight_bytes: u64,
    /// Actual candidate final-token logits in vocabulary order.
    candidate_logits: Vec<f32>,
    verification: Qwen3StreamVerification,
    candidate_total_ms: f64,
    scope: &'static str,
}

/// Whether a streamed report performed a resident-reference verification.
///
/// Candidate-only reports cannot represent comparison counts, tolerances, or
/// mismatch values because no comparison occurred.
#[derive(Debug, serde::Serialize)]
#[serde(rename_all = "snake_case")]
enum Qwen3StreamVerification {
    /// The candidate ran without a resident checkpoint oracle.
    CandidateOnly,
}

/// One teacher-forced cached append, including two independent resident controls.
#[derive(Debug, serde::Serialize)]
pub struct Qwen3StreamCachedStep {
    /// Total tokens in the cache after this append.
    total_tokens: usize,
    /// The known input IDs consumed at this step.
    appended_ids: Vec<i32>,
    /// Candidate execution only; neither resident oracle is included.
    candidate_ms: f64,
    /// Logical bytes in detached candidate K/V after this step.
    retained_kv_bytes: u64,
    /// Full vocabulary logits checked against both resident controls.
    compared_logits: usize,
    /// Largest absolute candidate-to-resident-cached error.
    cached_reference_maximum_absolute_error: f32,
    /// Largest absolute candidate-to-resident-full-prefix error.
    full_reference_maximum_absolute_error: f32,
}

/// Evidence from bounded, teacher-forced cached layer streaming.
///
/// This is a qualification scaffold for eventual generation, not a sampler:
/// append IDs are supplied by the caller so every cache transition has a
/// deterministic resident cached and full-forward control.
#[derive(Debug, serde::Serialize)]
pub struct Qwen3StreamCachedCheck {
    schema_version: u32,
    operation: &'static str,
    prompt_ids: Vec<i32>,
    decode_ids: Vec<i32>,
    maximum_total_tokens: usize,
    max_weight_bytes: u64,
    planned_peak_weight_and_staging_bytes: u64,
    max_kv_bytes: u64,
    planned_final_kv_bytes: u64,
    steps: Vec<Qwen3StreamCachedStep>,
    absolute_tolerance: f32,
    relative_tolerance: f32,
    scope: &'static str,
}

/// Qualifies bounded Qwen3 cached layer streaming using teacher-forced appends.
///
/// Every candidate layer loads its BF16 weights, executes one cached prefill or
/// decode append, then evaluates and rebuilds both its residual output and K/V
/// arrays before those weights leave scope. Each step compares every final
/// vocabulary logit to both a resident cached executor and a resident uncached
/// full-prefix forward. `max_weight_bytes` and `max_kv_bytes` are intentionally
/// separate contracts.
pub fn qualify_streamed_cached_forward(
    model: &Path,
    prompt_ids: &[i32],
    decode_ids: &[i32],
    max_weight_bytes: u64,
    max_kv_bytes: u64,
    tile_rows: usize,
) -> Result<Qwen3StreamCachedCheck, Qwen3MetalLoadError> {
    let all_ids = checked_cached_ids(prompt_ids, decode_ids)?;
    let plan = StreamForwardPlan::new(model, &all_ids, max_weight_bytes, tile_rows)?;
    let planned_final_kv_bytes = plan.config.cached_kv_bytes(all_ids.len())?;
    if planned_final_kv_bytes > max_kv_bytes {
        return Err(Qwen3MetalLoadError::CachedStateBudget {
            required: planned_final_kv_bytes,
            maximum: max_kv_bytes,
        });
    }

    let candidate = run_cached_candidate(&plan, prompt_ids, decode_ids)?;

    // Construct controls after candidate completion. They never participate in
    // the bounded candidate's planned residency or timings.
    let mut resident = Qwen3MlxWeights::load(model)?;
    resident.prepare_float32()?;
    let mut cached = resident.executor();
    let mut prefix = Vec::with_capacity(all_ids.len());
    let mut steps = Vec::with_capacity(candidate.len());
    for (index, candidate_step) in candidate.into_iter().enumerate() {
        let appended = if index == 0 {
            prompt_ids
        } else {
            &decode_ids[index - 1..index]
        };
        prefix.extend_from_slice(appended);
        let cached_logits = if index == 0 {
            cached.prefill_last_logits(appended)?
        } else {
            cached.decode_last_logits(appended[0])?
        };
        let full_logits = resident.forward_last_logits(&prefix)?;
        let cached_error = compare_cached_step_logits(
            &candidate_step.logits,
            &cached_logits,
            ABSOLUTE_TOLERANCE,
            RELATIVE_TOLERANCE,
            index,
            "resident cached",
        )?;
        let full_error = compare_cached_step_logits(
            &candidate_step.logits,
            &full_logits,
            ABSOLUTE_TOLERANCE,
            RELATIVE_TOLERANCE,
            index,
            "resident full-prefix",
        )?;
        steps.push(Qwen3StreamCachedStep {
            total_tokens: prefix.len(),
            appended_ids: appended.to_vec(),
            candidate_ms: candidate_step.candidate_ms,
            retained_kv_bytes: candidate_step.retained_kv_bytes,
            compared_logits: candidate_step.logits.len(),
            cached_reference_maximum_absolute_error: cached_error,
            full_reference_maximum_absolute_error: full_error,
        });
    }
    Ok(Qwen3StreamCachedCheck {
        schema_version: 1,
        operation: "qwen3_synchronous_streamed_cached_forward_check",
        prompt_ids: prompt_ids.to_vec(),
        decode_ids: decode_ids.to_vec(),
        maximum_total_tokens: 32,
        max_weight_bytes,
        planned_peak_weight_and_staging_bytes: plan.memory.peak,
        max_kv_bytes,
        planned_final_kv_bytes,
        steps,
        absolute_tolerance: ABSOLUTE_TOLERANCE,
        relative_tolerance: RELATIVE_TOLERANCE,
        scope: "bounded teacher-forced cached qualification only: each candidate layer evaluates and rebuilds its residual output plus K/V before its layer weights drop; every step compares all logits with independent resident cached and full-prefix controls; weight/staging and retained-KV budgets are separate and both exclude activations, operator scratch, allocator retention, headers, projection output, and resident controls; no sampling, server, process-peak, or beyond-RAM claim",
    })
}

struct CachedCandidateStep {
    logits: Vec<f32>,
    candidate_ms: f64,
    retained_kv_bytes: u64,
}

fn checked_cached_ids(
    prompt_ids: &[i32],
    decode_ids: &[i32],
) -> Result<Vec<i32>, Qwen3MetalLoadError> {
    validate_stream_length(prompt_ids)?;
    let total = prompt_ids.len().checked_add(decode_ids.len()).ok_or(
        Qwen3MetalLoadError::DimensionOutOfRange("cached token count"),
    )?;
    if total > 32 {
        return Err(crate::forward::Qwen3ForwardError::PromptTooLong {
            actual: total,
            maximum: 32,
        }
        .into());
    }
    let mut all = Vec::with_capacity(total);
    all.extend_from_slice(prompt_ids);
    all.extend_from_slice(decode_ids);
    Ok(all)
}

fn run_cached_candidate(
    plan: &StreamForwardPlan,
    prompt_ids: &[i32],
    decode_ids: &[i32],
) -> Result<Vec<CachedCandidateStep>, Qwen3MetalLoadError> {
    let mut cache = (0..plan.config.hidden_layers())
        .map(|_| None)
        .collect::<Vec<Option<Qwen3LayerKv>>>();
    let mut cached_tokens = 0_usize;
    let mut result = Vec::with_capacity(decode_ids.len() + 1);
    for appended in std::iter::once(prompt_ids).chain(decode_ids.iter().map(std::slice::from_ref)) {
        let started = Instant::now();
        let logits = run_cached_append(plan, appended, cached_tokens, &mut cache)?;
        cached_tokens = cached_tokens.checked_add(appended.len()).ok_or(
            Qwen3MetalLoadError::DimensionOutOfRange("cached token count"),
        )?;
        let retained_kv_bytes = cache.iter().flatten().try_fold(0_u64, |total, kv| {
            let bytes = u64::try_from(kv.keys.nbytes())
                .ok()
                .and_then(|keys| {
                    u64::try_from(kv.values.nbytes())
                        .ok()
                        .and_then(|values| keys.checked_add(values))
                })
                .ok_or(Qwen3MetalLoadError::DimensionOutOfRange("cached KV bytes"))?;
            total
                .checked_add(bytes)
                .ok_or(Qwen3MetalLoadError::DimensionOutOfRange("cached KV bytes"))
        })?;
        if retained_kv_bytes != plan.config.cached_kv_bytes(cached_tokens)? {
            return Err(crate::forward::Qwen3ForwardError::CacheInconsistent.into());
        }
        validate_candidate_logits(&logits)?;
        result.push(CachedCandidateStep {
            logits,
            candidate_ms: started.elapsed().as_secs_f64() * 1000.0,
            retained_kv_bytes,
        });
    }
    Ok(result)
}

fn run_cached_append(
    plan: &StreamForwardPlan,
    input_ids: &[i32],
    cached_tokens: usize,
    cache: &mut [Option<Qwen3LayerKv>],
) -> Result<Vec<f32>, Qwen3MetalLoadError> {
    let embedding_plan = EmbeddingPlan::new(
        input_ids,
        plan.inspection.contract().vocab_size(),
        plan.inspection.contract().hidden_size(),
        u64::MAX,
    )?;
    let embedding = load_embedding(&plan.inspection, input_ids, &embedding_plan)?;
    let shape = [
        1,
        i32::try_from(input_ids.len())
            .map_err(|_| Qwen3MetalLoadError::DimensionOutOfRange("tokens"))?,
        i32::try_from(plan.hidden)
            .map_err(|_| Qwen3MetalLoadError::DimensionOutOfRange("hidden"))?,
    ];
    let mut hidden = rebuild_hidden(&embedding, &shape)?;
    let sequence = shape[1];
    let rope_offset = i32::try_from(cached_tokens)
        .map_err(|_| Qwen3MetalLoadError::DimensionOutOfRange("cached tokens"))?;
    for layer in &plan.layers {
        let weights = load_layer(&plan.inspection, &layer.names, &layer.lengths)?;
        let output = forward_cached_layer(
            &plan.config,
            &weights,
            layer.layer,
            cache
                .get_mut(layer.layer)
                .ok_or(crate::forward::Qwen3ForwardError::CacheInconsistent)?,
            &hidden,
            sequence,
            rope_offset,
        )?;
        hidden = rebuild_hidden(&output, &shape)?;
        detach_kv(
            cache
                .get_mut(layer.layer)
                .ok_or(crate::forward::Qwen3ForwardError::CacheInconsistent)?,
        )?;
        drop(output);
        drop(weights);
    }
    let final_norm = load_bf16_tensor(&plan.inspection, FINAL_NORM, plan.final_norm_raw)?;
    let stream = StreamOrDevice::gpu();
    let last = hidden.take_axis_device(Array::from_slice(&[sequence - 1], &[1]), 1, &stream)?;
    let normalized = final_rms_norm(&plan.config, &last, &final_norm)?;
    let projection_input = rebuild_projection_input(&normalized, plan.hidden)?;
    Ok(project_tiled(
        &plan.inspection,
        &projection_input,
        &plan.projection,
        plan.projection.max_tile_raw_bytes,
    )?
    .logits)
}

fn detach_kv(cache: &mut Option<Qwen3LayerKv>) -> Result<(), Qwen3MetalLoadError> {
    let previous = cache
        .take()
        .ok_or(crate::forward::Qwen3ForwardError::CacheInconsistent)?;
    let keys = rebuild_array(&previous.keys)?;
    let values = rebuild_array(&previous.values)?;
    *cache = Some(Qwen3LayerKv { keys, values });
    Ok(())
}

fn rebuild_array(input: &Array) -> Result<Array, Qwen3MetalLoadError> {
    // `as_slice` in mlx-rs exposes the backing pointer without applying
    // strides. K/V is transposed to [batch, heads, sequence, dimension], so
    // flatten through MLX first: it materializes logical row-major order for
    // a non-contiguous view before host readback and reconstruction.
    let packed = input.reshape_device(&[-1], StreamOrDevice::gpu())?;
    packed.eval()?;
    let shape = input.shape().to_vec();
    let values = packed.as_slice::<f32>().to_vec();
    let rebuilt = Array::from_slice(&values, &shape);
    rebuilt.eval()?;
    Ok(rebuilt)
}

fn compare_cached_step_logits(
    candidate: &[f32],
    reference: &[f32],
    absolute_tolerance: f32,
    relative_tolerance: f32,
    step: usize,
    reference_kind: &'static str,
) -> Result<f32, Qwen3MetalLoadError> {
    compare_projection_logits(candidate, reference, absolute_tolerance, relative_tolerance).map_err(
        |error| match error {
            Qwen3MetalLoadError::RangeCheckMismatch { index } => {
                Qwen3MetalLoadError::CachedStreamParity {
                    step,
                    reference: reference_kind,
                    index,
                }
            }
            other => other,
        },
    )
}

/// Qualifies a complete, synchronous layer-streamed Qwen3 forward pass.
///
/// Each decoder layer is read, evaluated, copied to host, and rebuilt into a
/// fresh MLX hidden-state array before its weights are released. This prevents
/// lazy graph dependencies from retaining a prior layer's candidate weights.
/// The resident control starts only after the candidate has completed.
pub fn qualify_streamed_forward(
    model: &Path,
    input_ids: &[i32],
    max_weight_bytes: u64,
    tile_rows: usize,
) -> Result<Qwen3StreamCheck, Qwen3MetalLoadError> {
    let plan = StreamForwardPlan::new(model, input_ids, max_weight_bytes, tile_rows)?;

    let candidate = run_candidate(
        &plan.inspection,
        &plan.config,
        input_ids,
        &plan.embedding,
        &plan.layers,
        plan.final_norm_raw,
        &plan.projection,
    )?;

    let reference_started = Instant::now();
    let mut reference = Qwen3MlxWeights::load(model)?;
    reference.prepare_float32()?;
    let reference_logits = reference.forward_last_logits(input_ids)?;
    let resident_reference_ms = reference_started.elapsed().as_secs_f64() * 1000.0;
    let maximum_absolute_error = compare_projection_logits(
        &candidate.projection.logits,
        &reference_logits,
        ABSOLUTE_TOLERANCE,
        RELATIVE_TOLERANCE,
    )?;

    Ok(Qwen3StreamCheck {
        schema_version: 1,
        operation: "qwen3_synchronous_streamed_forward_check",
        input_ids: input_ids.to_vec(),
        input_recipe: "checkpoint BF16 token-embedding rows in caller order",
        hidden_shape: [1, input_ids.len(), plan.hidden],
        embedding_raw_payload_bytes: plan.embedding.raw_bytes,
        embedding_planned_weight_and_staging_bytes: plan.memory.embedding_peak,
        embedding_load_and_readback_ms: candidate.embedding_ms,
        layers: candidate.layers,
        final_norm_raw_payload_bytes: plan.final_norm_raw,
        final_norm_planned_weight_and_staging_bytes: plan.memory.final_norm_peak,
        final_norm_execute_and_readback_ms: candidate.norm_ms,
        projection_raw_payload_bytes: plan.projection.raw_payload_bytes,
        projection_max_tile_raw_payload_bytes: plan.projection.max_tile_raw_bytes,
        projection_planned_weight_and_staging_bytes: plan.memory.projection_peak,
        projection_tile_count: plan.projection.ranges.len(),
        projection_load_ms: candidate.projection.load_ms,
        projection_execute_and_readback_ms: candidate.projection.execute_ms,
        total_raw_payload_bytes: plan.memory.total_raw_payload_bytes,
        planned_peak_weight_and_staging_bytes: plan.memory.peak,
        max_weight_bytes,
        compared_logits: candidate.projection.logits.len(),
        absolute_tolerance: ABSOLUTE_TOLERANCE,
        relative_tolerance: RELATIVE_TOLERANCE,
        maximum_absolute_error,
        mismatches: 0,
        candidate_total_ms: candidate.total_ms,
        resident_reference_ms,
        scope: "synchronous candidate qualification only: each layer output is evaluated, copied to host, and reconstructed before its layer weights drop; planned budget covers candidate weights plus explicit loading/conversion staging, but excludes activations, attention/MLP scratch, allocator retention, headers, and the separate resident oracle; cumulative raw payload includes selected input embedding rows, every layer, final norm, and the full tiled output embedding; this is correctness evidence, not a model-performance or process-peak claim",
    })
}

/// Runs a complete synchronous Qwen3 stream without a resident reference.
///
/// This isolates the candidate process footprint. The report includes all
/// candidate logits for an external comparison, but that output serialization
/// itself is part of the measured process and the result is not parity
/// qualified.
pub fn run_streamed_forward_candidate(
    model: &Path,
    input_ids: &[i32],
    max_weight_bytes: u64,
    tile_rows: usize,
) -> Result<Qwen3StreamCandidateReport, Qwen3MetalLoadError> {
    let plan = StreamForwardPlan::new(model, input_ids, max_weight_bytes, tile_rows)?;
    let candidate = run_candidate(
        &plan.inspection,
        &plan.config,
        input_ids,
        &plan.embedding,
        &plan.layers,
        plan.final_norm_raw,
        &plan.projection,
    )?;
    validate_candidate_logits(&candidate.projection.logits)?;
    Ok(Qwen3StreamCandidateReport {
        schema_version: 1,
        operation: "qwen3_synchronous_streamed_forward_candidate",
        input_ids: input_ids.to_vec(),
        input_recipe: "checkpoint BF16 token-embedding rows in caller order",
        hidden_shape: [1, input_ids.len(), plan.hidden],
        embedding_raw_payload_bytes: plan.embedding.raw_bytes,
        embedding_planned_weight_and_staging_bytes: plan.memory.embedding_peak,
        embedding_load_and_readback_ms: candidate.embedding_ms,
        layers: candidate.layers,
        final_norm_raw_payload_bytes: plan.final_norm_raw,
        final_norm_planned_weight_and_staging_bytes: plan.memory.final_norm_peak,
        final_norm_execute_and_readback_ms: candidate.norm_ms,
        projection_raw_payload_bytes: plan.projection.raw_payload_bytes,
        projection_max_tile_raw_payload_bytes: plan.projection.max_tile_raw_bytes,
        projection_planned_weight_and_staging_bytes: plan.memory.projection_peak,
        projection_tile_count: plan.projection.ranges.len(),
        projection_load_ms: candidate.projection.load_ms,
        projection_execute_and_readback_ms: candidate.projection.execute_ms,
        total_raw_payload_bytes: plan.memory.total_raw_payload_bytes,
        planned_peak_weight_and_staging_bytes: plan.memory.peak,
        max_weight_bytes,
        candidate_logits: candidate.projection.logits,
        verification: Qwen3StreamVerification::CandidateOnly,
        candidate_total_ms: candidate.total_ms,
        scope: "candidate-only synchronous stream: no resident reference checkpoint is loaded and no parity comparison is performed; candidate logits are emitted for external comparison; output serialization is part of this process measurement; planned budget covers candidate weights plus explicit loading/conversion staging, but excludes activations, attention/MLP scratch, allocator retention, and headers; this is neither a parity qualification nor a model-performance or process-peak claim",
    })
}

fn validate_candidate_logits(logits: &[f32]) -> Result<(), Qwen3MetalLoadError> {
    logits
        .iter()
        .position(|logit| !logit.is_finite())
        .map_or(Ok(()), |index| {
            Err(Qwen3MetalLoadError::CandidateNonFiniteLogit { index })
        })
}

struct StreamForwardPlan {
    config: Qwen3ForwardConfig,
    inspection: Qwen3CheckpointInspection,
    hidden: usize,
    embedding: EmbeddingPlan,
    layers: Vec<LayerPlan>,
    final_norm_raw: u64,
    projection: ProjectionPlan,
    memory: StreamMemoryPlan,
}

impl StreamForwardPlan {
    fn new(
        model: &Path,
        input_ids: &[i32],
        max_weight_bytes: u64,
        tile_rows: usize,
    ) -> Result<Self, Qwen3MetalLoadError> {
        let config_path = model.join("config.json");
        let config_json = std::fs::read_to_string(&config_path).map_err(|source| {
            Qwen3CheckpointError::ReadConfig {
                path: config_path,
                source,
            }
        })?;
        let config = Qwen3ForwardConfig::parse(&config_json)?;
        validate_stream_length(input_ids)?;
        let inspection = Qwen3CheckpointInspection::inspect(model)?;
        let hidden = config.hidden_size();
        input_ids
            .len()
            .checked_mul(hidden)
            .filter(|&count| count <= 1_048_576)
            .ok_or(Qwen3MetalLoadError::DimensionOutOfRange(
                "stream hidden states",
            ))?;
        let embedding = EmbeddingPlan::new(
            input_ids,
            inspection.contract().vocab_size(),
            inspection.contract().hidden_size(),
            u64::MAX,
        )?;
        let projection = ProjectionPlan::new(
            hidden,
            usize::try_from(inspection.contract().vocab_size())
                .map_err(|_| Qwen3MetalLoadError::DimensionOutOfRange("vocab_size"))?,
            tile_rows,
            u64::MAX,
        )?;
        let layers = layer_plans(&inspection, &config)?;
        let final_norm_raw = inspection.bf16_tensor_bytes(FINAL_NORM)?;
        let memory = StreamMemoryPlan::new(&embedding, &layers, final_norm_raw, &projection)?;
        if memory.peak > max_weight_bytes {
            return Err(Qwen3MetalLoadError::LayerWeightBudget {
                required: memory.peak,
                maximum: max_weight_bytes,
            });
        }
        Ok(Self {
            config,
            inspection,
            hidden,
            embedding,
            layers,
            final_norm_raw,
            projection,
            memory,
        })
    }
}

fn validate_stream_length(input_ids: &[i32]) -> Result<(), Qwen3MetalLoadError> {
    use crate::forward::Qwen3ForwardError;
    if input_ids.is_empty() {
        return Err(Qwen3ForwardError::EmptyInput.into());
    }
    if input_ids.len() > 32 {
        return Err(Qwen3ForwardError::PromptTooLong {
            actual: input_ids.len(),
            maximum: 32,
        }
        .into());
    }
    Ok(())
}

struct StreamCandidate {
    embedding_ms: f64,
    layers: Vec<Qwen3StreamLayer>,
    norm_ms: f64,
    projection: TiledProjection,
    total_ms: f64,
}

fn run_candidate(
    inspection: &Qwen3CheckpointInspection,
    config: &Qwen3ForwardConfig,
    input_ids: &[i32],
    embedding_plan: &EmbeddingPlan,
    layers: &[LayerPlan],
    final_norm_raw: u64,
    projection_plan: &ProjectionPlan,
) -> Result<StreamCandidate, Qwen3MetalLoadError> {
    let hidden = config.hidden_size();
    let candidate_started = Instant::now();
    let embedding_started = Instant::now();
    let embedding = load_embedding(inspection, input_ids, embedding_plan)?;
    let shape = [
        1,
        i32::try_from(input_ids.len())
            .map_err(|_| Qwen3MetalLoadError::DimensionOutOfRange("tokens"))?,
        i32::try_from(hidden).map_err(|_| Qwen3MetalLoadError::DimensionOutOfRange("hidden"))?,
    ];
    let mut hidden_states = rebuild_hidden(&embedding, &shape)?;
    let embedding_load_and_readback_ms = embedding_started.elapsed().as_secs_f64() * 1000.0;

    let mut layer_results = Vec::with_capacity(layers.len());
    for plan in layers {
        let load_started = Instant::now();
        let weights = load_layer(inspection, &plan.names, &plan.lengths)?;
        let load_ms = load_started.elapsed().as_secs_f64() * 1000.0;
        let execute_started = Instant::now();
        let output = forward_layer(config, &weights, plan.layer, &hidden_states)?;
        hidden_states = rebuild_hidden(&output, &shape)?;
        let execute_and_readback_ms = execute_started.elapsed().as_secs_f64() * 1000.0;
        drop(output);
        drop(weights);
        layer_results.push(Qwen3StreamLayer {
            layer: plan.layer,
            raw_payload_bytes: plan.raw_payload_bytes,
            planned_weight_and_staging_bytes: plan.peak_weight_and_staging_bytes,
            load_ms,
            execute_and_readback_ms,
        });
    }

    let final_norm_started = Instant::now();
    let final_norm = load_bf16_tensor(inspection, FINAL_NORM, final_norm_raw)?;
    let stream = StreamOrDevice::gpu();
    let last = hidden_states.take_axis_device(
        Array::from_slice(
            &[i32::try_from(input_ids.len() - 1)
                .map_err(|_| Qwen3MetalLoadError::DimensionOutOfRange("last token"))?],
            &[1],
        ),
        1,
        &stream,
    )?;
    let normalized = final_rms_norm(config, &last, &final_norm)?;
    let projection_input = rebuild_projection_input(&normalized, hidden)?;
    let final_norm_execute_and_readback_ms = final_norm_started.elapsed().as_secs_f64() * 1000.0;
    drop(normalized);
    drop(last);
    drop(final_norm);

    let projection = project_tiled(
        inspection,
        &projection_input,
        projection_plan,
        projection_plan.max_tile_raw_bytes,
    )?;
    let candidate_total_ms = candidate_started.elapsed().as_secs_f64() * 1000.0;

    Ok(StreamCandidate {
        embedding_ms: embedding_load_and_readback_ms,
        layers: layer_results,
        norm_ms: final_norm_execute_and_readback_ms,
        projection,
        total_ms: candidate_total_ms,
    })
}

struct LayerPlan {
    layer: usize,
    names: Vec<String>,
    lengths: Vec<u64>,
    raw_payload_bytes: u64,
    peak_weight_and_staging_bytes: u64,
}

fn layer_plans(
    inspection: &Qwen3CheckpointInspection,
    config: &Qwen3ForwardConfig,
) -> Result<Vec<LayerPlan>, Qwen3MetalLoadError> {
    (0..config.hidden_layers())
        .map(|layer| {
            let names: Vec<_> = LAYER_SUFFIXES
                .iter()
                .map(|suffix| format!("model.layers.{layer}.{suffix}"))
                .collect();
            let lengths = names
                .iter()
                .map(|name| inspection.bf16_tensor_bytes(name))
                .collect::<Result<Vec<_>, _>>()?;
            let (raw_payload_bytes, _, peak_weight_and_staging_bytes) =
                plan_weight_bytes(&lengths)?;
            Ok(LayerPlan {
                layer,
                names,
                lengths,
                raw_payload_bytes,
                peak_weight_and_staging_bytes,
            })
        })
        .collect()
}

struct StreamMemoryPlan {
    embedding_peak: u64,
    final_norm_peak: u64,
    projection_peak: u64,
    peak: u64,
    total_raw_payload_bytes: u64,
}

impl StreamMemoryPlan {
    fn new(
        embedding: &EmbeddingPlan,
        layers: &[LayerPlan],
        final_norm_raw: u64,
        projection: &ProjectionPlan,
    ) -> Result<Self, Qwen3MetalLoadError> {
        let embedding_peak = embedding
            .raw_bytes
            .checked_mul(2)
            .and_then(|retained_rows| {
                embedding
                    .row_bytes
                    .checked_mul(3)
                    .and_then(|staging| retained_rows.checked_add(staging))
            })
            .ok_or(Qwen3MetalLoadError::DimensionOutOfRange(
                "embedding staging",
            ))?;
        let final_norm_peak =
            final_norm_raw
                .checked_mul(5)
                .ok_or(Qwen3MetalLoadError::DimensionOutOfRange(
                    "final norm staging",
                ))?;
        let projection_peak = projection.max_tile_raw_bytes.checked_mul(5).ok_or(
            Qwen3MetalLoadError::DimensionOutOfRange("projection staging"),
        )?;
        let layer_peak = layers
            .iter()
            .map(|plan| plan.peak_weight_and_staging_bytes)
            .max()
            .ok_or(Qwen3MetalLoadError::DimensionOutOfRange("layer plans"))?;
        let peak = embedding_peak
            .max(final_norm_peak)
            .max(projection_peak)
            .max(layer_peak);
        let layer_raw = layers.iter().try_fold(0_u64, |total, plan| {
            total.checked_add(plan.raw_payload_bytes).ok_or(
                Qwen3MetalLoadError::DimensionOutOfRange("layer raw payload"),
            )
        })?;
        let total_raw_payload_bytes = embedding
            .raw_bytes
            .checked_add(layer_raw)
            .and_then(|total| total.checked_add(final_norm_raw))
            .and_then(|total| total.checked_add(projection.raw_payload_bytes))
            .ok_or(Qwen3MetalLoadError::DimensionOutOfRange(
                "stream raw payload",
            ))?;
        Ok(Self {
            embedding_peak,
            final_norm_peak,
            projection_peak,
            peak,
            total_raw_payload_bytes,
        })
    }
}

fn rebuild_hidden(input: &Array, shape: &[i32; 3]) -> Result<Array, Qwen3MetalLoadError> {
    let reshaped = input.reshape_device(shape, StreamOrDevice::gpu())?;
    rebuild_array(&reshaped)
}

fn rebuild_projection_input(input: &Array, hidden: usize) -> Result<Array, Qwen3MetalLoadError> {
    let shape = [
        1,
        i32::try_from(hidden).map_err(|_| Qwen3MetalLoadError::DimensionOutOfRange("hidden"))?,
    ];
    let reshaped = input.reshape_device(&shape, StreamOrDevice::gpu())?;
    rebuild_array(&reshaped)
}

fn load_bf16_tensor(
    inspection: &Qwen3CheckpointInspection,
    name: &str,
    max_bytes: u64,
) -> Result<Array, Qwen3MetalLoadError> {
    let payload = inspection.read_tensor(name, max_bytes)?;
    let values = decode_bf16(payload.bytes())?;
    let shape = payload
        .shape()
        .iter()
        .map(|&dimension| {
            i32::try_from(dimension)
                .map_err(|_| Qwen3MetalLoadError::DimensionOutOfRange("final norm shape"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let tensor = Array::from_slice(&values, &shape);
    tensor.eval()?;
    Ok(tensor)
}

#[cfg(test)]
mod tests {
    use mlx_rs::{Array, StreamOrDevice};

    use crate::GPU_TEST_LOCK;

    use super::{
        EmbeddingPlan, LayerPlan, ProjectionPlan, Qwen3StreamCandidateReport, Qwen3StreamLayer,
        Qwen3StreamVerification, StreamMemoryPlan,
    };

    #[test]
    fn input_length_errors_describe_the_actual_limit() {
        assert!(super::validate_stream_length(&[1; 32]).is_ok());
        assert_eq!(
            super::validate_stream_length(&[]).unwrap_err().to_string(),
            "Qwen3 forward requires at least one token"
        );
        assert_eq!(
            super::validate_stream_length(&[1; 33])
                .unwrap_err()
                .to_string(),
            "Qwen3 reference prompt has 33 tokens, maximum is 32"
        );
    }

    #[test]
    fn cached_input_plan_keeps_prompt_and_teacher_forced_appends_bounded() {
        assert_eq!(
            super::checked_cached_ids(&[1, 2], &[3, 4]).unwrap(),
            [1, 2, 3, 4]
        );
        assert!(matches!(
            super::checked_cached_ids(&[1; 31], &[2, 3]),
            Err(super::Qwen3MetalLoadError::ForwardConfig(
                crate::forward::Qwen3ForwardError::PromptTooLong {
                    actual: 33,
                    maximum: 32
                }
            ))
        ));
    }

    #[test]
    fn stream_budget_is_the_largest_sequential_stage_not_the_cumulative_payload() {
        let embedding = EmbeddingPlan::new(&[0, 1], 8, 4, u64::MAX).unwrap();
        let projection = ProjectionPlan::new(4, 8, 3, u64::MAX).unwrap();
        let layers = [LayerPlan {
            layer: 0,
            names: Vec::new(),
            lengths: Vec::new(),
            raw_payload_bytes: 100,
            peak_weight_and_staging_bytes: 500,
        }];
        let plan = StreamMemoryPlan::new(&embedding, &layers, 8, &projection).unwrap();
        assert_eq!(plan.peak, 500);
        assert_eq!(plan.total_raw_payload_bytes, 16 + 100 + 8 + 64);
    }

    #[test]
    fn candidate_only_report_serializes_logits_without_parity_fields() {
        let report = Qwen3StreamCandidateReport {
            schema_version: 1,
            operation: "qwen3_synchronous_streamed_forward_candidate",
            input_ids: vec![1],
            input_recipe: "fixture",
            hidden_shape: [1, 1, 2],
            embedding_raw_payload_bytes: 4,
            embedding_planned_weight_and_staging_bytes: 8,
            embedding_load_and_readback_ms: 1.0,
            layers: vec![Qwen3StreamLayer {
                layer: 0,
                raw_payload_bytes: 2,
                planned_weight_and_staging_bytes: 10,
                load_ms: 2.0,
                execute_and_readback_ms: 3.0,
            }],
            final_norm_raw_payload_bytes: 2,
            final_norm_planned_weight_and_staging_bytes: 10,
            final_norm_execute_and_readback_ms: 4.0,
            projection_raw_payload_bytes: 8,
            projection_max_tile_raw_payload_bytes: 4,
            projection_planned_weight_and_staging_bytes: 20,
            projection_tile_count: 2,
            projection_load_ms: 5.0,
            projection_execute_and_readback_ms: 6.0,
            total_raw_payload_bytes: 16,
            planned_peak_weight_and_staging_bytes: 20,
            max_weight_bytes: 32,
            candidate_logits: vec![0.25, -0.5],
            verification: Qwen3StreamVerification::CandidateOnly,
            candidate_total_ms: 7.0,
            scope: "fixture",
        };
        let json = serde_json::to_value(report).unwrap();
        assert_eq!(json["verification"], "candidate_only");
        assert_eq!(json["candidate_logits"], serde_json::json!([0.25, -0.5]));
        for absent in [
            "compared_logits",
            "mismatches",
            "maximum_absolute_error",
            "resident_reference_ms",
        ] {
            assert!(json.get(absent).is_none(), "unexpected {absent}");
        }
    }

    #[test]
    fn candidate_logit_boundary_rejects_every_nonfinite_kind_and_keeps_finite_values() {
        assert!(super::validate_candidate_logits(&[-0.0, 1.0, f32::MIN, f32::MAX]).is_ok());
        for (logits, expected_index) in [
            (&[1.0, f32::NAN, 3.0][..], 1),
            (&[1.0, 2.0, f32::INFINITY][..], 2),
            (&[1.0, 2.0, 3.0, f32::NEG_INFINITY][..], 3),
        ] {
            assert!(matches!(
                super::validate_candidate_logits(logits),
                Err(super::Qwen3MetalLoadError::CandidateNonFiniteLogit { index }) if index == expected_index
            ));
        }
    }

    #[test]
    fn detached_transposed_kv_preserves_logical_sequence_order() {
        let _gpu = GPU_TEST_LOCK.lock().expect("GPU test lock");
        let original_values = (0_u16..24).map(f32::from).collect::<Vec<_>>();
        let original = Array::from_slice(&original_values, &[1, 3, 2, 4]);
        let stream = StreamOrDevice::gpu();
        let transposed = original
            .transpose_axes_device(&[0, 2, 1, 3], &stream)
            .expect("non-contiguous KV layout");
        let detached = super::rebuild_array(&transposed).expect("detach transposed KV");
        assert_eq!(detached.shape(), [1, 2, 3, 4]);
        assert_eq!(
            detached
                .as_slice::<f32>()
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            [
                0_u16, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 4, 5, 6, 7, 12, 13, 14, 15, 20, 21,
                22, 23,
            ]
            .iter()
            .map(|&value| f32::from(value).to_bits())
            .collect::<Vec<_>>()
        );
    }
}
