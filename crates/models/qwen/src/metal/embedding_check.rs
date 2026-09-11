//! Token-ordered embedding lookup from bounded checkpoint rows.

use std::{
    collections::{BTreeMap, BTreeSet},
    path::Path,
    time::Instant,
};

use mlx_rs::{Array, StreamOrDevice};

use super::{
    Qwen3MetalLoadError, Qwen3MlxWeights, compare_tensor_values, decode_bf16, validate_token_ids,
};
use crate::checkpoint::{Qwen3CheckpointError, Qwen3CheckpointInspection};

const EMBEDDING: &str = "model.embed_tokens.weight";

/// Evidence for selected embedding reads, not a streamed decoder.
#[derive(Debug, serde::Serialize)]
pub struct Qwen3EmbeddingCheck {
    schema_version: u32,
    operation: &'static str,
    input_ids: Vec<i32>,
    shape: Vec<i32>,
    unique_rows: usize,
    raw_payload_bytes: u64,
    full_embedding_bytes: u64,
    candidate_array_bytes: usize,
    load_ms: f64,
    compared_values: usize,
    bit_exact: bool,
    scope: &'static str,
}

/// Looks up raw token IDs from selected BF16 rows and checks a resident oracle.
///
/// Repeated IDs are read once and restored in input order. The aggregate raw
/// payload budget excludes headers, FP32 buffers and the resident reference.
/// At most 512 tokens and 1,048,576 output elements are accepted. Checkpoints
/// must remain immutable throughout the call. This does not run the decoder.
pub fn qualify_embedding(
    model: &Path,
    input_ids: &[i32],
    max_bytes: u64,
) -> Result<Qwen3EmbeddingCheck, Qwen3MetalLoadError> {
    let inspection = Qwen3CheckpointInspection::inspect(model)?;
    let plan = EmbeddingPlan::new(
        input_ids,
        inspection.contract().vocab_size(),
        inspection.contract().hidden_size(),
        max_bytes,
    )?;
    let full_embedding_bytes = inspection.bf16_tensor_bytes(EMBEDDING)?;
    let started = Instant::now();
    let candidate = load_embedding(&inspection, input_ids, &plan)?;
    candidate.eval()?;
    let load_ms = started.elapsed().as_secs_f64() * 1000.0;

    let reference_weights = Qwen3MlxWeights::load(model)?;
    let ids = Array::from_slice(input_ids, &[plan.tokens]);
    let reference = reference_weights
        .tensors
        .get(EMBEDDING)
        .ok_or(Qwen3MetalLoadError::MissingEmbedding)?
        .take_axis_device(&ids, 0, StreamOrDevice::gpu())?
        .as_type_device::<f32>(StreamOrDevice::gpu())?;
    reference.eval()?;
    if candidate.shape() != reference.shape() {
        return Err(Qwen3MetalLoadError::RangeCheckShape);
    }
    compare_tensor_values(candidate.as_slice::<f32>(), reference.as_slice::<f32>())?;
    Ok(Qwen3EmbeddingCheck {
        schema_version: 1,
        operation: "qwen3_selected_embedding_check",
        input_ids: input_ids.to_vec(),
        shape: candidate.shape().to_vec(),
        unique_rows: plan.rows.len(),
        raw_payload_bytes: plan.raw_bytes,
        full_embedding_bytes,
        candidate_array_bytes: candidate.nbytes(),
        load_ms,
        compared_values: candidate.size(),
        bit_exact: true,
        scope: "token-ordered embedding lookup; repeated IDs read once; raw budget excludes FP32 buffers, headers and resident reference; no streamed decoder or physical SSD measurement",
    })
}

struct EmbeddingPlan {
    rows: BTreeSet<i32>,
    tokens: i32,
    hidden: i32,
    elements: usize,
    row_bytes: u64,
    raw_bytes: u64,
}

impl EmbeddingPlan {
    fn new(
        ids: &[i32],
        vocab: u32,
        hidden: u32,
        max_bytes: u64,
    ) -> Result<Self, Qwen3MetalLoadError> {
        validate_token_ids(ids, vocab)?;
        if ids.len() > 512 || hidden == 0 {
            return Err(Qwen3MetalLoadError::DimensionOutOfRange("embedding shape"));
        }
        let elements = ids
            .len()
            .checked_mul(hidden as usize)
            .filter(|&n| n <= 1_048_576)
            .ok_or(Qwen3MetalLoadError::DimensionOutOfRange(
                "embedding elements",
            ))?;
        let rows: BTreeSet<_> = ids.iter().copied().collect();
        let row_bytes = u64::from(hidden) * 2;
        let selected_bytes = row_bytes * rows.len() as u64;
        if selected_bytes > max_bytes {
            return Err(Qwen3CheckpointError::TensorExceedsReadBudget {
                tensor: EMBEDDING.to_owned(),
                tensor_bytes: selected_bytes,
                max_bytes,
            }
            .into());
        }
        Ok(Self {
            rows,
            elements,
            row_bytes,
            raw_bytes: selected_bytes,
            tokens: i32::try_from(ids.len())
                .map_err(|_| Qwen3MetalLoadError::DimensionOutOfRange("tokens"))?,
            hidden: i32::try_from(hidden)
                .map_err(|_| Qwen3MetalLoadError::DimensionOutOfRange("hidden"))?,
        })
    }
}

fn load_embedding(
    inspection: &Qwen3CheckpointInspection,
    input_ids: &[i32],
    plan: &EmbeddingPlan,
) -> Result<Array, Qwen3MetalLoadError> {
    let mut rows = BTreeMap::new();
    for &token in &plan.rows {
        let row = usize::try_from(token)
            .map_err(|_| Qwen3MetalLoadError::DimensionOutOfRange("token"))?;
        let payload = inspection.read_bf16_rows(EMBEDDING, row..row + 1, plan.row_bytes)?;
        rows.insert(token, decode_bf16(payload.bytes())?);
    }
    let mut values = Vec::with_capacity(plan.elements);
    for token in input_ids {
        values.extend_from_slice(
            rows.get(token)
                .ok_or(Qwen3MetalLoadError::RangeCheckShape)?,
        );
    }
    Ok(Array::from_slice(&values, &[plan.tokens, plan.hidden]))
}

#[cfg(test)]
mod tests {
    use super::EmbeddingPlan;

    #[test]
    fn budget_counts_unique_rows_but_output_preserves_token_count() {
        let plan = EmbeddingPlan::new(&[7, 0, 7, 3], 8, 4, 24).unwrap();
        assert_eq!(plan.rows.iter().copied().collect::<Vec<_>>(), [0, 3, 7]);
        assert_eq!((plan.raw_bytes, plan.elements, plan.tokens), (24, 16, 4));
        assert!(EmbeddingPlan::new(&[7, 0, 7, 3], 8, 4, 23).is_err());
    }

    #[test]
    fn rejects_invalid_ids_and_unbounded_outputs() {
        for ids in [vec![], vec![-1], vec![8], vec![0; 513]] {
            assert!(EmbeddingPlan::new(&ids, 8, 4, u64::MAX).is_err());
        }
        assert!(EmbeddingPlan::new(&[0; 512], 8, 4096, u64::MAX).is_err());
    }
}
