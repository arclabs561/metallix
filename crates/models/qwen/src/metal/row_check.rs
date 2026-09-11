//! Row-slab qualification for embedding lookup and tiled output projection.

use std::{ops::Range, path::Path, time::Instant};

use mlx_rs::{Array, StreamOrDevice, ops::indexing::TryIndexOp};

use super::{Qwen3MetalLoadError, Qwen3MlxWeights, compare_tensor_values, decode_bf16};
use crate::checkpoint::Qwen3CheckpointInspection;

/// A bounded row read compared with the independent resident MLX loader.
#[derive(Debug, serde::Serialize)]
pub struct Qwen3TensorRowsCheck {
    schema_version: u32,
    operation: &'static str,
    tensor: String,
    rows: [usize; 2],
    shape: Vec<u64>,
    raw_payload_bytes: usize,
    max_payload_bytes: u64,
    decoded_host_bytes: usize,
    candidate_array_bytes: usize,
    read_ms: f64,
    compared_values: usize,
    bit_exact: bool,
    scope: &'static str,
}

/// Compares a contiguous, half-open row range of a BF16 matrix with MLX.
///
/// Only selected raw bytes are budgeted. FP32 buffers, headers and the resident
/// reference are excluded. Keep checkpoint files immutable for the complete
/// comparison. This reads rows; it does not implement tiled model execution.
pub fn qualify_tensor_rows(
    model_dir: impl AsRef<Path>,
    tensor: &str,
    rows: Range<usize>,
    max_bytes: u64,
) -> Result<Qwen3TensorRowsCheck, Qwen3MetalLoadError> {
    let start = i32::try_from(rows.start)
        .map_err(|_| Qwen3MetalLoadError::DimensionOutOfRange("start row"))?;
    let end =
        i32::try_from(rows.end).map_err(|_| Qwen3MetalLoadError::DimensionOutOfRange("end row"))?;
    let inspection = Qwen3CheckpointInspection::inspect(model_dir.as_ref())?;
    let started = Instant::now();
    let payload = inspection.read_bf16_rows(tensor, rows.clone(), max_bytes)?;
    let read_ms = started.elapsed().as_secs_f64() * 1000.0;
    let values = decode_bf16(payload.bytes())?;
    let shape = payload
        .shape()
        .iter()
        .map(|&dimension| {
            i32::try_from(dimension)
                .map_err(|_| Qwen3MetalLoadError::DimensionOutOfRange("row shape"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let candidate = Array::from_slice(&values, &shape);
    candidate.eval()?;

    // Slice the independently loaded BF16 array before casting. The full
    // resident loader is deliberately outside the selected-read budget.
    let reference_weights = Qwen3MlxWeights::load(model_dir.as_ref())?;
    let reference = reference_weights
        .tensors
        .get(tensor)
        .ok_or_else(|| Qwen3MetalLoadError::RangeCheckMissingTensor(tensor.to_owned()))?
        .try_index_device((start..end, ..), StreamOrDevice::gpu())?
        .as_type_device::<f32>(StreamOrDevice::gpu())?;
    reference.eval()?;
    if reference.shape() != candidate.shape() {
        return Err(Qwen3MetalLoadError::RangeCheckShape);
    }
    compare_tensor_values(candidate.as_slice::<f32>(), reference.as_slice::<f32>())?;
    Ok(Qwen3TensorRowsCheck {
        schema_version: 1,
        operation: "qwen3_bf16_tensor_rows_check",
        tensor: tensor.to_owned(),
        rows: [rows.start, rows.end],
        shape: payload.shape().to_vec(),
        raw_payload_bytes: payload.bytes().len(),
        max_payload_bytes: max_bytes,
        decoded_host_bytes: values.len() * size_of::<f32>(),
        candidate_array_bytes: candidate.nbytes(),
        read_ms,
        compared_values: values.len(),
        bit_exact: true,
        scope: "contiguous BF16 matrix rows; raw payload budget excludes FP32 buffers and resident reference; immutable input; no tiled inference or physical SSD measurement",
    })
}
