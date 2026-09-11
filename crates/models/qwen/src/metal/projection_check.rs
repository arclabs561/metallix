//! Tiled tied-output projection qualification from bounded embedding rows.

use std::{ops::Range, path::Path, time::Instant};

use mlx_rs::{Array, StreamOrDevice};

use super::{Qwen3MetalLoadError, Qwen3MlxWeights, decode_bf16};
use crate::{
    checkpoint::{Qwen3CheckpointError, Qwen3CheckpointInspection},
    forward::Qwen3ForwardConfig,
};

const EMBEDDING: &str = "model.embed_tokens.weight";
const MAX_VOCABULARY: usize = 1_048_576;
const MAX_HIDDEN_SIZE: usize = 16_384;
const ABSOLUTE_TOLERANCE: f32 = 5e-5;
const RELATIVE_TOLERANCE: f32 = 1e-4;

/// A tiled tied-output projection compared with an independent resident MLX control.
#[derive(Debug, serde::Serialize)]
pub struct Qwen3ProjectionCheck {
    schema_version: u32,
    operation: &'static str,
    input_shape: [usize; 2],
    input_recipe: &'static str,
    tile_rows: usize,
    tile_count: usize,
    raw_payload_bytes: u64,
    full_embedding_bytes: u64,
    max_tile_raw_payload_bytes: u64,
    max_decoded_tile_bytes: u64,
    max_raw_payload_bytes_per_tile: u64,
    tile_load_ms: f64,
    tile_execution_ms: f64,
    reference_ms: f64,
    compared_logits: usize,
    absolute_tolerance: f32,
    relative_tolerance: f32,
    maximum_absolute_error: f32,
    mismatches: usize,
    within_tolerance: bool,
    scope: &'static str,
}

/// Computes tied output logits from BF16 embedding tiles and checks MLX's resident path.
///
/// The Qwen forward configuration is parsed first, so this diagnostic only uses
/// `model.embed_tokens.weight` as an output table when tied embeddings were
/// explicitly accepted. `max_bytes` applies independently to each raw tile;
/// it neither bounds the cumulative payload nor the process memory footprint.
/// Checkpoint files must stay immutable for the complete qualification.
pub fn qualify_projection(
    model: &Path,
    tile_rows: usize,
    max_bytes: u64,
) -> Result<Qwen3ProjectionCheck, Qwen3MetalLoadError> {
    let config_path = model.join("config.json");
    let config_json = std::fs::read_to_string(&config_path).map_err(|source| {
        Qwen3CheckpointError::ReadConfig {
            path: config_path,
            source,
        }
    })?;
    let config = Qwen3ForwardConfig::parse(&config_json)?;
    let inspection = Qwen3CheckpointInspection::inspect(model)?;
    let plan = ProjectionPlan::new(
        config.hidden_size(),
        usize::try_from(inspection.contract().vocab_size())
            .map_err(|_| Qwen3MetalLoadError::DimensionOutOfRange("vocab_size"))?,
        tile_rows,
        max_bytes,
    )?;
    let full_embedding_bytes = inspection.bf16_tensor_bytes(EMBEDDING)?;
    if full_embedding_bytes != plan.raw_payload_bytes {
        return Err(Qwen3MetalLoadError::RangeCheckShape);
    }

    let input_values = synthetic_hidden_row(plan.hidden);
    let input = Array::from_slice(&input_values, &[1, plan.hidden_i32]);
    input.eval()?;

    let candidate = project_tiled(&inspection, &input, &plan, max_bytes)?;

    let reference_started = Instant::now();
    let reference_weights = Qwen3MlxWeights::load(model)?;
    let reference_embedding = reference_weights
        .tensors
        .get(EMBEDDING)
        .ok_or(Qwen3MetalLoadError::MissingEmbedding)?
        .as_type_device::<f32>(StreamOrDevice::gpu())?;
    let stream = StreamOrDevice::gpu();
    let reference_logits =
        input.matmul_device(&reference_embedding.transpose_device(&stream)?, &stream)?;
    reference_logits.eval()?;
    let reference_values = reference_logits.as_slice::<f32>();
    let reference_ms = reference_started.elapsed().as_secs_f64() * 1000.0;
    let maximum_absolute_error = compare_projection_logits(
        &candidate.logits,
        reference_values,
        ABSOLUTE_TOLERANCE,
        RELATIVE_TOLERANCE,
    )?;

    Ok(Qwen3ProjectionCheck {
        schema_version: 1,
        operation: "qwen3_tiled_tied_output_projection_check",
        input_shape: [1, plan.hidden],
        input_recipe: "f32((flat_index % 101) - 50) * 0.001",
        tile_rows,
        tile_count: plan.ranges.len(),
        raw_payload_bytes: plan.raw_payload_bytes,
        full_embedding_bytes,
        max_tile_raw_payload_bytes: plan.max_tile_raw_bytes,
        max_decoded_tile_bytes: plan.max_decoded_tile_bytes,
        max_raw_payload_bytes_per_tile: max_bytes,
        tile_load_ms: candidate.load_ms,
        tile_execution_ms: candidate.execute_ms,
        reference_ms,
        compared_logits: candidate.logits.len(),
        absolute_tolerance: ABSOLUTE_TOLERANCE,
        relative_tolerance: RELATIVE_TOLERANCE,
        maximum_absolute_error,
        mismatches: 0,
        within_tolerance: true,
        scope: "tied output projection only; each BF16 embedding slab is synchronously read, widened, evaluated and read back before the next slab; all embedding bytes are still read cumulatively, while candidate logits remain retained for final comparison; per-tile raw budget excludes FP32 buffers, GPU scratch, allocator retention and resident reference, so it is not a global memory or process-peak bound",
    })
}

pub(super) struct ProjectionPlan {
    pub(super) vocabulary: usize,
    pub(super) hidden: usize,
    hidden_i32: i32,
    bytes_per_row: u64,
    pub(super) raw_payload_bytes: u64,
    pub(super) max_tile_raw_bytes: u64,
    pub(super) max_decoded_tile_bytes: u64,
    pub(super) ranges: Vec<Range<usize>>,
}

impl ProjectionPlan {
    pub(super) fn new(
        hidden: usize,
        vocabulary: usize,
        tile_rows: usize,
        max_bytes: u64,
    ) -> Result<Self, Qwen3MetalLoadError> {
        if tile_rows == 0
            || vocabulary == 0
            || vocabulary > MAX_VOCABULARY
            || hidden == 0
            || hidden > MAX_HIDDEN_SIZE
        {
            return Err(Qwen3MetalLoadError::DimensionOutOfRange(
                "projection shape or tile_rows",
            ));
        }
        let hidden_i32 = i32::try_from(hidden)
            .map_err(|_| Qwen3MetalLoadError::DimensionOutOfRange("hidden"))?;
        let bytes_per_row = u64::try_from(hidden)
            .ok()
            .and_then(|width| width.checked_mul(2))
            .ok_or(Qwen3MetalLoadError::DimensionOutOfRange(
                "embedding row bytes",
            ))?;
        let mut ranges = Vec::new();
        let mut start = 0;
        let mut max_tile_raw_bytes = 0;
        while start < vocabulary {
            let end = start.saturating_add(tile_rows).min(vocabulary);
            let rows = start..end;
            let tile_raw_bytes = raw_bytes_for_range(&rows, bytes_per_row)?;
            if tile_raw_bytes > max_bytes {
                return Err(Qwen3CheckpointError::TensorExceedsReadBudget {
                    tensor: EMBEDDING.to_owned(),
                    tensor_bytes: tile_raw_bytes,
                    max_bytes,
                }
                .into());
            }
            max_tile_raw_bytes = max_tile_raw_bytes.max(tile_raw_bytes);
            ranges.push(rows);
            start = end;
        }
        let raw_payload_bytes = u64::try_from(vocabulary)
            .ok()
            .and_then(|rows| rows.checked_mul(bytes_per_row))
            .ok_or(Qwen3MetalLoadError::DimensionOutOfRange("embedding bytes"))?;
        let max_decoded_tile_bytes =
            max_tile_raw_bytes
                .checked_mul(2)
                .ok_or(Qwen3MetalLoadError::DimensionOutOfRange(
                    "decoded tile bytes",
                ))?;
        Ok(Self {
            vocabulary,
            hidden,
            hidden_i32,
            bytes_per_row,
            raw_payload_bytes,
            max_tile_raw_bytes,
            max_decoded_tile_bytes,
            ranges,
        })
    }

    fn raw_bytes_for(&self, rows: &Range<usize>) -> Result<u64, Qwen3MetalLoadError> {
        raw_bytes_for_range(rows, self.bytes_per_row)
    }
}

fn raw_bytes_for_range(
    rows: &Range<usize>,
    bytes_per_row: u64,
) -> Result<u64, Qwen3MetalLoadError> {
    u64::try_from(rows.len())
        .ok()
        .and_then(|count| count.checked_mul(bytes_per_row))
        .ok_or(Qwen3MetalLoadError::DimensionOutOfRange("tile raw bytes"))
}

fn synthetic_hidden_row(hidden: usize) -> Vec<f32> {
    (0_i16..101)
        .cycle()
        .take(hidden)
        .map(|bounded| f32::from(bounded - 50) * 0.001)
        .collect()
}

pub(super) fn compare_projection_logits(
    candidate: &[f32],
    reference: &[f32],
    absolute_tolerance: f32,
    relative_tolerance: f32,
) -> Result<f32, Qwen3MetalLoadError> {
    if candidate.len() != reference.len() {
        return Err(Qwen3MetalLoadError::RangeCheckShape);
    }
    let mut maximum_absolute_error = 0.0_f32;
    for (index, (&candidate, &reference)) in candidate.iter().zip(reference).enumerate() {
        if !candidate.is_finite() || !reference.is_finite() {
            return Err(Qwen3MetalLoadError::RangeCheckMismatch { index });
        }
        let absolute_error = (candidate - reference).abs();
        maximum_absolute_error = maximum_absolute_error.max(absolute_error);
        let tolerance = absolute_tolerance + relative_tolerance * reference.abs();
        if absolute_error > tolerance {
            return Err(Qwen3MetalLoadError::RangeCheckMismatch { index });
        }
    }
    Ok(maximum_absolute_error)
}

pub(super) struct TiledProjection {
    pub(super) logits: Vec<f32>,
    pub(super) load_ms: f64,
    pub(super) execute_ms: f64,
}

pub(super) fn project_tiled(
    inspection: &Qwen3CheckpointInspection,
    input: &Array,
    plan: &ProjectionPlan,
    max_bytes: u64,
) -> Result<TiledProjection, Qwen3MetalLoadError> {
    let mut logits = Vec::with_capacity(plan.vocabulary);
    let mut load_ms = 0.0;
    let mut execute_ms = 0.0;
    for rows in &plan.ranges {
        let tile_raw_bytes = plan.raw_bytes_for(rows)?;
        if tile_raw_bytes > max_bytes {
            return Err(Qwen3CheckpointError::TensorExceedsReadBudget {
                tensor: EMBEDDING.to_owned(),
                tensor_bytes: tile_raw_bytes,
                max_bytes,
            }
            .into());
        }
        let load_started = Instant::now();
        let payload = inspection.read_bf16_rows(EMBEDDING, rows.clone(), max_bytes)?;
        let values = decode_bf16(payload.bytes())?;
        let shape = [
            i32::try_from(rows.len())
                .map_err(|_| Qwen3MetalLoadError::DimensionOutOfRange("tile rows"))?,
            plan.hidden_i32,
        ];
        let tile = Array::from_slice(&values, &shape);
        load_ms += load_started.elapsed().as_secs_f64() * 1000.0;
        let execute_started = Instant::now();
        let stream = StreamOrDevice::gpu();
        let output = input.matmul_device(&tile.transpose_device(&stream)?, &stream)?;
        output.eval()?;
        logits.extend_from_slice(output.as_slice::<f32>());
        execute_ms += execute_started.elapsed().as_secs_f64() * 1000.0;
    }
    Ok(TiledProjection {
        logits,
        load_ms,
        execute_ms,
    })
}

#[cfg(test)]
mod tests {
    use super::{ProjectionPlan, compare_projection_logits};

    #[test]
    fn plan_covers_an_irregular_tail_without_exceeding_its_per_tile_bound() {
        let plan = ProjectionPlan::new(4, 10, 4, 32).expect("three bounded tiles");
        assert_eq!(plan.ranges.as_slice(), &[0..4, 4..8, 8..10]);
        assert_eq!(plan.raw_payload_bytes, 80);
        assert_eq!(plan.max_tile_raw_bytes, 32);
        assert_eq!(plan.max_decoded_tile_bytes, 64);
    }

    #[test]
    fn plan_rejects_zero_tiles_and_a_tile_over_the_raw_bound() {
        assert!(ProjectionPlan::new(4, 10, 0, 32).is_err());
        assert!(ProjectionPlan::new(4, 10, 5, 32).is_err());
    }

    #[test]
    fn comparator_catches_nonleading_mismatch_and_nonfinite_values() {
        assert!(matches!(
            compare_projection_logits(&[1.0, 2.0, 3.1], &[1.0, 2.0, 3.0], 5e-5, 1e-4),
            Err(super::Qwen3MetalLoadError::RangeCheckMismatch { index: 2 })
        ));
        assert!(compare_projection_logits(&[1.0, f32::NAN], &[1.0, 2.0], 5e-5, 1e-4).is_err());
    }
}
