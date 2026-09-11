//! MLX checkpoint ownership and Metal execution entry points for Qwen3.
//! The loaded checkpoint retains its decoder configuration and lends tensors
//! to independent sequence executors.

use std::{collections::HashMap, fs, path::Path};

use mlx_rs::{Array, StreamOrDevice};
use thiserror::Error;

use crate::checkpoint::{Qwen3CheckpointError, Qwen3CheckpointInspection};

mod layer_check;
pub use layer_check::{LayerCheckMode, Qwen3LayerCheck, qualify_layer};
mod row_check;
pub use row_check::{Qwen3TensorRowsCheck, qualify_tensor_rows};
mod embedding_check;
pub use embedding_check::{Qwen3EmbeddingCheck, qualify_embedding};
mod projection_check;
pub use projection_check::{Qwen3ProjectionCheck, qualify_projection};
mod stream_check;
pub use stream_check::{
    Qwen3StreamCachedCandidateReport, Qwen3StreamCachedCheck, Qwen3StreamCandidateReport,
    Qwen3StreamCheck, qualify_streamed_cached_forward, qualify_streamed_forward,
    run_streamed_cached_candidate, run_streamed_forward_candidate,
};

/// A selected-tensor loader comparison, not a bounded-residency inference result.
#[derive(Debug, serde::Serialize)]
pub struct Qwen3TensorRangeCheck {
    schema_version: u32,
    operation: &'static str,
    tensor: String,
    shape: Vec<u64>,
    raw_payload_bytes: u64,
    max_payload_bytes: u64,
    decoded_host_bytes: usize,
    candidate_array_bytes: usize,
    read_ms: f64,
    compared_values: usize,
    bit_exact: bool,
    scope: &'static str,
}

/// Compares one exact BF16 payload read with MLX's resident safetensors loader.
///
/// `max_bytes` bounds only the selected raw payload allocation, not headers,
/// FP32 conversion, GPU allocations or the whole-checkpoint reference load.
/// Checkpoint files must remain immutable for the complete comparison.
/// The read timing includes file/metadata checks, excludes inspection and is
/// neither a repeated benchmark nor evidence of physical SSD traffic.
pub fn qualify_tensor_range(
    model_dir: impl AsRef<Path>,
    tensor: &str,
    max_bytes: u64,
) -> Result<Qwen3TensorRangeCheck, Qwen3MetalLoadError> {
    let inspection = Qwen3CheckpointInspection::inspect(model_dir.as_ref())?;
    let started = std::time::Instant::now();
    let payload = inspection.read_tensor(tensor, max_bytes)?;
    let read_ms = started.elapsed().as_secs_f64() * 1000.0;
    if payload.dtype() != "BF16" {
        return Err(Qwen3MetalLoadError::RangeCheckDtype(
            payload.dtype().to_owned(),
        ));
    }
    let values = decode_bf16(payload.bytes())?;
    let shape: Vec<i32> = payload
        .shape()
        .iter()
        .map(|&dimension| {
            i32::try_from(dimension)
                .map_err(|_| Qwen3MetalLoadError::DimensionOutOfRange("tensor shape"))
        })
        .collect::<Result<_, _>>()?;
    let candidate = Array::from_slice(&values, &shape);
    candidate.eval()?;

    // Keep the established reader independent. It intentionally loads the
    // resident checkpoint and must never be counted as bounded-reader memory.
    let reference_weights = Qwen3MlxWeights::load(model_dir.as_ref())?;
    let reference = reference_weights
        .tensors
        .get(tensor)
        .ok_or_else(|| Qwen3MetalLoadError::RangeCheckMissingTensor(tensor.to_owned()))?
        .as_type_device::<f32>(StreamOrDevice::gpu())?;
    reference.eval()?;
    if reference.shape() != candidate.shape() {
        return Err(Qwen3MetalLoadError::RangeCheckShape);
    }
    compare_tensor_values(candidate.as_slice::<f32>(), reference.as_slice::<f32>())?;
    Ok(Qwen3TensorRangeCheck {
        schema_version: 1,
        operation: "qwen3_bf16_tensor_range_check",
        tensor: tensor.to_owned(),
        shape: payload.shape().to_vec(),
        raw_payload_bytes: payload.bytes().len() as u64,
        max_payload_bytes: max_bytes,
        decoded_host_bytes: values.len() * size_of::<f32>(),
        candidate_array_bytes: candidate.nbytes(),
        read_ms,
        compared_values: values.len(),
        bit_exact: true,
        scope: "selected raw payload bounded; FP32 buffers and resident MLX reference excluded from that budget; no decoder execution or physical SSD measurement",
    })
}

fn decode_bf16(bytes: &[u8]) -> Result<Vec<f32>, Qwen3MetalLoadError> {
    if !bytes.len().is_multiple_of(2) {
        return Err(Qwen3MetalLoadError::RangeCheckOddBytes);
    }
    let mut values = vec![0.0_f32; bytes.len() / 2];
    let mut all_finite = true;
    for (destination, pair) in values.iter_mut().zip(bytes.chunks_exact(2)) {
        let bits = u32::from(u16::from_le_bytes([pair[0], pair[1]])) << 16;
        let value = f32::from_bits(bits);
        all_finite &= value.is_finite();
        *destination = value;
    }
    if !all_finite {
        return Err(Qwen3MetalLoadError::RangeCheckNonFinite);
    }
    Ok(values)
}

fn compare_tensor_values(left: &[f32], right: &[f32]) -> Result<(), Qwen3MetalLoadError> {
    if left.len() != right.len() {
        return Err(Qwen3MetalLoadError::RangeCheckShape);
    }
    for (index, (&left, &right)) in left.iter().zip(right).enumerate() {
        if !left.is_finite() || !right.is_finite() || left.to_bits() != right.to_bits() {
            return Err(Qwen3MetalLoadError::RangeCheckMismatch { index });
        }
    }
    Ok(())
}

/// Evidence that MLX evaluated a small graph on Metal.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Qwen3MetalSmoke {
    product: f32,
}

impl Qwen3MetalSmoke {
    /// Returns the evaluated matrix product.
    #[must_use]
    pub const fn product(self) -> f32 {
        self.product
    }
}

/// Runs a minimal GPU evaluation through the pinned MLX Rust binding.
///
/// # Errors
///
/// Returns [`Qwen3MetalSmokeError`] when MLX cannot construct or evaluate the
/// Metal graph, or if the readback differs from the known result.
pub fn run_metal_smoke() -> Result<Qwen3MetalSmoke, Qwen3MetalSmokeError> {
    let left = Array::from_slice(&[2.0_f32], &[1, 1]);
    let right = Array::from_slice(&[3.0_f32], &[1, 1]);
    let product = left.matmul_device(&right, StreamOrDevice::gpu())?;
    product.eval()?;
    let product = product.item::<f32>();
    if !product.is_finite() || (product - 6.0).abs() > f32::EPSILON {
        return Err(Qwen3MetalSmokeError::UnexpectedProduct(product));
    }
    Ok(Qwen3MetalSmoke { product })
}

/// A Qwen3 checkpoint represented as MLX arrays for Metal execution.
///
/// This owns the loaded tensor map and validated decoder configuration. Loading
/// proves that MLX accepts the checkpoint's actual tensor payloads; it does
/// not by itself execute the decoder.
pub struct Qwen3MlxWeights {
    inspection: Qwen3CheckpointInspection,
    forward_config: crate::forward::Qwen3ForwardConfig,
    tensors: HashMap<String, Array>,
    embedding_shape: Vec<i32>,
}

impl Qwen3MlxWeights {
    /// Starts an independent sequence executor borrowing this checkpoint.
    /// Its KV state cannot be transferred to a different checkpoint.
    #[must_use]
    pub fn executor(
        &self,
    ) -> crate::forward::Qwen3ForwardExecutor<'_, std::collections::hash_map::RandomState> {
        crate::forward::Qwen3ForwardExecutor::new(&self.forward_config, &self.tensors)
    }

    /// Materializes float32 weights once for comparison with a CPU float32 oracle.
    /// This increases resident weight memory relative to the BF16 checkpoint.
    pub fn prepare_float32(&mut self) -> Result<(), Qwen3MetalLoadError> {
        for weight in self.tensors.values_mut() {
            let converted = weight.as_type_device::<f32>(StreamOrDevice::gpu())?;
            converted.eval()?;
            *weight = converted;
        }
        Ok(())
    }

    /// Executes the dense qualification decoder using these checkpoint tensors.
    pub fn forward_last_logits(
        &self,
        input_ids: &[i32],
    ) -> Result<Vec<f32>, crate::forward::Qwen3ForwardError> {
        crate::forward::forward_last_logits(&self.tensors, &self.forward_config, input_ids)
    }

    /// Loads all validated safetensors shards as MLX arrays for Metal execution.
    ///
    /// The checkpoint headers are validated before payload loading. The token
    /// embedding is evaluated to force a real model tensor through MLX while
    /// leaving decoder execution to the subsequent parity gate.
    ///
    /// # Errors
    ///
    /// Returns [`Qwen3MetalLoadError`] if checkpoint inspection fails, MLX
    /// cannot load the payloads, or the loaded embedding is inconsistent with
    /// the checkpoint contract.
    pub fn load(model_dir: impl AsRef<Path>) -> Result<Self, Qwen3MetalLoadError> {
        let config_path = model_dir.as_ref().join("config.json");
        let config_json = fs::read_to_string(&config_path).map_err(|source| {
            Qwen3CheckpointError::ReadConfig {
                path: config_path,
                source,
            }
        })?;
        let forward_config = crate::forward::Qwen3ForwardConfig::parse(&config_json)?;
        let inspection = Qwen3CheckpointInspection::inspect(model_dir)?;
        let mut tensors = HashMap::with_capacity(inspection.tensor_count());
        for shard in inspection.shards() {
            // MLX safetensors I/O is defined on the CPU stream. Subsequent
            // graph operations choose the GPU stream explicitly.
            let shard_tensors = Array::load_safetensors_device(shard, StreamOrDevice::cpu())?;
            tensors.extend(shard_tensors);
        }

        if tensors.len() != inspection.tensor_count() {
            return Err(Qwen3MetalLoadError::UnexpectedTensorCount {
                expected: inspection.tensor_count(),
                actual: tensors.len(),
            });
        }

        let expected_embedding_shape = [
            i32::try_from(inspection.contract().vocab_size())
                .map_err(|_| Qwen3MetalLoadError::DimensionOutOfRange("vocab_size"))?,
            i32::try_from(inspection.contract().hidden_size())
                .map_err(|_| Qwen3MetalLoadError::DimensionOutOfRange("hidden_size"))?,
        ];
        let embedding = tensors
            .get("model.embed_tokens.weight")
            .ok_or(Qwen3MetalLoadError::MissingEmbedding)?;
        if embedding.shape() != expected_embedding_shape {
            return Err(Qwen3MetalLoadError::UnexpectedEmbeddingShape {
                expected: expected_embedding_shape.to_vec(),
                actual: embedding.shape().to_vec(),
            });
        }
        // A reduction avoids allocating an embedding-sized output while
        // forcing the actual checkpoint tensor through a GPU graph.
        let embedding_sum = embedding.sum_device(None, StreamOrDevice::gpu())?;
        embedding_sum.eval()?;
        let embedding_shape = embedding.shape().to_vec();

        Ok(Self {
            inspection,
            forward_config,
            tensors,
            embedding_shape,
        })
    }

    /// Returns the number of loaded tensors.
    #[must_use]
    pub fn tensor_count(&self) -> usize {
        self.tensors.len()
    }

    /// Logical bytes in the current tensor dtypes, excluding allocator overhead.
    #[must_use]
    pub fn logical_weight_bytes(&self) -> usize {
        self.tensors.values().map(Array::nbytes).sum()
    }

    /// Returns the evaluated token-embedding shape.
    #[must_use]
    pub fn embedding_shape(&self) -> &[i32] {
        &self.embedding_shape
    }

    /// Returns the header-derived checkpoint facts that preceded payload load.
    #[must_use]
    pub const fn inspection(&self) -> &Qwen3CheckpointInspection {
        &self.inspection
    }

    /// Evaluates an embedding lookup for raw token IDs on the GPU stream.
    ///
    /// This is the first numerical-forward building block. It intentionally
    /// stops before normalization, attention, or tied-logit projection.
    ///
    /// # Errors
    ///
    /// Returns [`Qwen3MetalLoadError`] when no IDs are supplied or MLX cannot
    /// construct or evaluate the lookup graph.
    pub fn embed_token_ids(&self, input_ids: &[i32]) -> Result<Vec<i32>, Qwen3MetalLoadError> {
        validate_token_ids(input_ids, self.inspection.contract().vocab_size())?;
        let token_ids = Array::from_slice(
            input_ids,
            &[i32::try_from(input_ids.len())
                .map_err(|_| Qwen3MetalLoadError::DimensionOutOfRange("input_ids"))?],
        );
        let embedding = self
            .tensors
            .get("model.embed_tokens.weight")
            .ok_or(Qwen3MetalLoadError::MissingEmbedding)?;
        let vectors = embedding.take_axis_device(&token_ids, 0, StreamOrDevice::gpu())?;
        vectors.eval()?;
        Ok(vectors.shape().to_vec())
    }
}

fn validate_token_ids(input_ids: &[i32], vocab_size: u32) -> Result<(), Qwen3MetalLoadError> {
    if input_ids.is_empty() {
        return Err(Qwen3MetalLoadError::EmptyInputIds);
    }
    for &token in input_ids {
        if u32::try_from(token).map_or(true, |id| id >= vocab_size) {
            return Err(Qwen3MetalLoadError::InvalidTokenId { token, vocab_size });
        }
    }
    Ok(())
}

/// A failed Metal substrate qualification.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum Qwen3MetalSmokeError {
    /// MLX failed to construct or evaluate the graph.
    #[error("MLX Metal evaluation failed: {0}")]
    Mlx(#[from] mlx_rs::error::Exception),
    /// GPU readback did not preserve the known matrix product.
    #[error("MLX Metal smoke read back {0}, expected 6")]
    UnexpectedProduct(f32),
}

/// A failure while loading a validated Qwen3 checkpoint onto Metal.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum Qwen3MetalLoadError {
    /// A cached streamed step differed from a named resident control.
    #[error("cached stream step {step} differs from {reference} at logit {index}")]
    CachedStreamParity {
        /// Zero-based prefill/decode step.
        step: usize,
        /// Independent control that rejected the candidate.
        reference: &'static str,
        /// Vocabulary-logit position.
        index: usize,
    },
    /// Retained detached KV arrays exceed the independent caller-selected cap.
    #[error("cached KV plan requires {required} bytes, above budget {maximum}")]
    CachedStateBudget {
        /// Logical f32 bytes for all retained layer K/V arrays.
        required: u64,
        /// Caller-selected ceiling; excludes activations and allocator overhead.
        maximum: u64,
    },
    /// The candidate layer's live weight/staging plan exceeds its budget.
    #[error("layer weight/staging plan requires {required} bytes, above budget {maximum}")]
    LayerWeightBudget {
        /// Peak planned logical bytes during weight loading.
        required: u64,
        /// Caller-selected ceiling; excludes operator scratch and the reference.
        maximum: u64,
    },
    /// The diagnostic currently qualifies only BF16 checkpoint payloads.
    #[error("tensor-range comparison requires BF16, got {0}")]
    RangeCheckDtype(String),
    /// The independently loaded tensor was missing.
    #[error("resident reference did not contain tensor {0:?}")]
    RangeCheckMissingTensor(String),
    /// Candidate and independently loaded tensor dimensions differed.
    #[error("tensor-range comparison shape mismatch")]
    RangeCheckShape,
    /// A BF16 payload did not contain complete elements.
    #[error("BF16 payload has an odd byte count")]
    RangeCheckOddBytes,
    /// The diagnostic does not qualify non-finite weight values.
    #[error("tensor-range comparison encountered a non-finite BF16 value")]
    RangeCheckNonFinite,
    /// Exact widening disagreed with the independent resident reader.
    #[error("tensor-range comparison differs at value {index}")]
    RangeCheckMismatch {
        /// Flattened value position, not a byte offset.
        index: usize,
    },
    /// A candidate-only streamed forward produced a value JSON cannot represent faithfully.
    #[error("candidate-only streamed forward produced a non-finite logit at index {index}")]
    CandidateNonFiniteLogit {
        /// Flattened vocabulary-logit position.
        index: usize,
    },
    /// The checkpoint requests unsupported decoder semantics.
    #[error(transparent)]
    ForwardConfig(#[from] crate::forward::Qwen3ForwardError),
    /// Checkpoint structure did not satisfy the Qwen3 loader contract.
    #[error(transparent)]
    Checkpoint(#[from] Qwen3CheckpointError),
    /// MLX could not load a safetensors payload on the Metal device.
    #[error("MLX Metal safetensors load failed: {0}")]
    Mlx(#[from] mlx_rs::error::IoError),
    /// MLX could not evaluate the embedding tensor on Metal.
    #[error("MLX Metal embedding evaluation failed: {0}")]
    Evaluation(#[from] mlx_rs::error::Exception),
    /// The checkpoint's configured dimension cannot be represented by MLX.
    #[error("Qwen3 {0} does not fit MLX's shape representation")]
    DimensionOutOfRange(&'static str),
    /// Loading did not preserve the checkpoint's validated tensor count.
    #[error("MLX loaded {actual} tensors, expected {expected}")]
    UnexpectedTensorCount { expected: usize, actual: usize },
    /// The required token embedding was absent after loading.
    #[error("MLX did not load model.embed_tokens.weight")]
    MissingEmbedding,
    /// MLX reported an embedding shape different from the model contract.
    #[error("MLX embedding shape {actual:?}, expected {expected:?}")]
    UnexpectedEmbeddingShape {
        expected: Vec<i32>,
        actual: Vec<i32>,
    },
    /// A forward lookup needs at least one token ID.
    #[error("Qwen3 embedding lookup requires at least one token ID")]
    EmptyInputIds,
    /// Token IDs must name rows in the configured embedding vocabulary.
    #[error("token ID {token} is outside vocabulary 0..{vocab_size}")]
    InvalidTokenId { token: i32, vocab_size: u32 },
}

#[cfg(test)]
mod tests {
    use super::{Qwen3MetalLoadError, run_metal_smoke, validate_token_ids};

    #[test]
    fn widens_known_bf16_bits_without_changing_signed_zero() {
        let values =
            super::decode_bf16(&[0, 0, 0, 128, 128, 63, 32, 192]).expect("finite BF16 values");
        let bits: Vec<_> = values.iter().map(|value| value.to_bits()).collect();
        assert_eq!(bits, [0, 0x8000_0000, 0x3f80_0000, 0xc020_0000]);
        assert!(super::decode_bf16(&[0]).is_err());
        assert!(super::decode_bf16(&[128, 127]).is_err());
        assert!(super::decode_bf16(&[192, 127]).is_err());
    }

    #[test]
    fn bf16_widening_covers_every_bit_pattern() {
        let mut finite_words = Vec::new();
        for word in 0..=u16::MAX {
            let decoded = super::decode_bf16(&word.to_le_bytes());
            if word & 0x7f80 == 0x7f80 {
                assert!(matches!(
                    decoded,
                    Err(Qwen3MetalLoadError::RangeCheckNonFinite)
                ));
            } else {
                assert_eq!(decoded.unwrap()[0].to_bits(), u32::from(word) << 16);
                finite_words.push(word);
            }
        }
        let bytes: Vec<_> = finite_words
            .iter()
            .flat_map(|word| word.to_le_bytes())
            .collect();
        let decoded = super::decode_bf16(&bytes).unwrap();
        assert_eq!(decoded.len(), finite_words.len());
        for (value, word) in decoded.iter().zip(finite_words) {
            assert_eq!(value.to_bits(), u32::from(word) << 16);
        }
    }

    #[test]
    fn bf16_widening_rejects_nonfinite_at_any_batch_position() {
        for position in [0, 512, 1023] {
            for word in [0x7f80_u16, 0xff80, 0x7f81, 0xffff] {
                let mut bytes = vec![0_u8; 2048];
                bytes[position * 2..position * 2 + 2].copy_from_slice(&word.to_le_bytes());
                assert!(matches!(
                    super::decode_bf16(&bytes),
                    Err(Qwen3MetalLoadError::RangeCheckNonFinite)
                ));
            }
        }
    }

    #[test]
    fn comparison_rejects_single_value_mutation_and_shape_drift() {
        assert!(super::compare_tensor_values(&[1.0, -2.5], &[1.0, -2.5]).is_ok());
        assert!(matches!(
            super::compare_tensor_values(&[1.0, -2.5], &[1.0, -2.0]),
            Err(Qwen3MetalLoadError::RangeCheckMismatch { index: 1 })
        ));
        assert!(super::compare_tensor_values(&[0.0], &[-0.0]).is_err());
        assert!(super::compare_tensor_values(&[1.0], &[]).is_err());
        assert!(super::compare_tensor_values(&[f32::NAN], &[f32::NAN]).is_err());
    }

    #[test]
    fn rejects_invalid_token_ids_before_gpu_indexing() {
        assert!(validate_token_ids(&[0, 151_935], 151_936).is_ok());
        for token in [-1, 151_936, i32::MAX] {
            assert!(matches!(
                validate_token_ids(&[token], 151_936),
                Err(Qwen3MetalLoadError::InvalidTokenId { .. })
            ));
        }
        assert!(matches!(
            validate_token_ids(&[], 151_936),
            Err(Qwen3MetalLoadError::EmptyInputIds)
        ));
    }

    #[test]
    fn evaluates_a_known_gpu_matrix_product() {
        let _guard = crate::GPU_TEST_LOCK.lock().expect("GPU test lock");
        let smoke = run_metal_smoke().expect("MLX Metal must evaluate the smoke graph");
        assert!((smoke.product() - 6.0).abs() <= f32::EPSILON);
    }
}
