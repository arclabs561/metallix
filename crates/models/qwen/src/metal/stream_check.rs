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
    forward::{Qwen3ForwardConfig, final_rms_norm, forward_layer},
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
    let embedding_plan = EmbeddingPlan::new(
        input_ids,
        inspection.contract().vocab_size(),
        inspection.contract().hidden_size(),
        u64::MAX,
    )?;
    let projection_plan = ProjectionPlan::new(
        hidden,
        usize::try_from(inspection.contract().vocab_size())
            .map_err(|_| Qwen3MetalLoadError::DimensionOutOfRange("vocab_size"))?,
        tile_rows,
        u64::MAX,
    )?;
    let layers = layer_plans(&inspection, &config)?;
    let final_norm_raw = inspection.bf16_tensor_bytes(FINAL_NORM)?;
    let memory = StreamMemoryPlan::new(&embedding_plan, &layers, final_norm_raw, &projection_plan)?;
    if memory.peak > max_weight_bytes {
        return Err(Qwen3MetalLoadError::LayerWeightBudget {
            required: memory.peak,
            maximum: max_weight_bytes,
        });
    }

    let candidate = run_candidate(
        &inspection,
        &config,
        input_ids,
        &embedding_plan,
        &layers,
        final_norm_raw,
        &projection_plan,
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
        hidden_shape: [1, input_ids.len(), hidden],
        embedding_raw_payload_bytes: embedding_plan.raw_bytes,
        embedding_planned_weight_and_staging_bytes: memory.embedding_peak,
        embedding_load_and_readback_ms: candidate.embedding_ms,
        layers: candidate.layers,
        final_norm_raw_payload_bytes: final_norm_raw,
        final_norm_planned_weight_and_staging_bytes: memory.final_norm_peak,
        final_norm_execute_and_readback_ms: candidate.norm_ms,
        projection_raw_payload_bytes: projection_plan.raw_payload_bytes,
        projection_max_tile_raw_payload_bytes: projection_plan.max_tile_raw_bytes,
        projection_planned_weight_and_staging_bytes: memory.projection_peak,
        projection_tile_count: projection_plan.ranges.len(),
        projection_load_ms: candidate.projection.load_ms,
        projection_execute_and_readback_ms: candidate.projection.execute_ms,
        total_raw_payload_bytes: memory.total_raw_payload_bytes,
        planned_peak_weight_and_staging_bytes: memory.peak,
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
    input.eval()?;
    let values = input.as_slice::<f32>().to_vec();
    let rebuilt = Array::from_slice(&values, shape);
    rebuilt.eval()?;
    Ok(rebuilt)
}

fn rebuild_projection_input(input: &Array, hidden: usize) -> Result<Array, Qwen3MetalLoadError> {
    input.eval()?;
    let values = input.as_slice::<f32>().to_vec();
    let shape = [
        1,
        i32::try_from(hidden).map_err(|_| Qwen3MetalLoadError::DimensionOutOfRange("hidden"))?,
    ];
    let rebuilt = Array::from_slice(&values, &shape);
    rebuilt.eval()?;
    Ok(rebuilt)
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
    use super::{EmbeddingPlan, LayerPlan, ProjectionPlan, StreamMemoryPlan};

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
}
