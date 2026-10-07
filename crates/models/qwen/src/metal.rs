//! MLX checkpoint ownership and Metal execution entry points for Qwen3.
//! The loaded checkpoint retains its decoder configuration and lends tensors
//! to independent sequence executors.

#![allow(
    deprecated,
    reason = "mlx-rs 0.32 deprecates the *_device ops; the with_stream migration is a separate change"
)]

use std::{collections::HashMap, fs, path::Path};

use mlx_rs::{Array, Dtype, StreamOrDevice};
use thiserror::Error;

use crate::checkpoint::{Qwen3CheckpointError, Qwen3CheckpointInspection};
pub use crate::forward::{Qwen3FloatPrecision, Qwen3WeightPrecision};

mod layer_check;
pub use layer_check::{LayerCheckMode, Qwen3LayerCheck, qualify_layer};
mod row_check;
pub use row_check::{Qwen3TensorRowsCheck, qualify_tensor_rows};
mod embedding_check;
pub use embedding_check::{Qwen3EmbeddingCheck, qualify_embedding};
mod projection_check;
pub use projection_check::{Qwen3ProjectionCheck, qualify_projection};
#[cfg(test)]
mod affine_checkpoint_tests;
mod stream_check;
pub use stream_check::{
    Qwen3StreamCachedCandidateReport, Qwen3StreamCachedCheck, Qwen3StreamCandidateReport,
    Qwen3StreamCheck, Qwen3StreamExecutor, Qwen3StreamProfile, qualify_streamed_cached_forward,
    qualify_streamed_forward, run_streamed_cached_candidate, run_streamed_forward_candidate,
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
///
/// # Errors
///
/// * [`RangeCheckMissingTensor`](crate::metal::Qwen3MetalLoadError::RangeCheckMissingTensor),
///   [`RangeCheckDtype`](crate::metal::Qwen3MetalLoadError::RangeCheckDtype),
///   [`RangeCheckShape`](crate::metal::Qwen3MetalLoadError::RangeCheckShape)
///   and
///   [`RangeCheckOddBytes`](crate::metal::Qwen3MetalLoadError::RangeCheckOddBytes)
///   when the tensor is absent or not the layout the check reads.
/// * [`DimensionOutOfRange`](crate::metal::Qwen3MetalLoadError::DimensionOutOfRange)
///   when a dimension does not fit MLX.
/// * [`RangeCheckNonFinite`](crate::metal::Qwen3MetalLoadError::RangeCheckNonFinite)
///   and
///   [`RangeCheckMismatch`](crate::metal::Qwen3MetalLoadError::RangeCheckMismatch)
///   when the decoded values are not finite or differ from the device's.
/// * [`Checkpoint`](crate::metal::Qwen3MetalLoadError::Checkpoint) and
///   [`ForwardConfig`](crate::metal::Qwen3MetalLoadError::ForwardConfig) when
///   the checkpoint headers or configuration fail validation.
/// * [`Mlx`](crate::metal::Qwen3MetalLoadError::Mlx) and
///   [`Evaluation`](crate::metal::Qwen3MetalLoadError::Evaluation) when MLX
///   cannot load or evaluate a tensor.
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
        // Copy the chunk as a unit so LLVM can widen halfword loads, rather
        // than reconstructing each word from separate byte lanes. No alignment
        // assumption is made about the source slice.
        let word = u16::from_le_bytes(pair.try_into().expect("exact two-byte chunk"));
        let bits = u32::from(word) << 16;
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
    let product = product.item_exact::<f32>();
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
    /// Process-unique identity of these tensor values, carried by detached
    /// K/V snapshots so they cannot be restored onto other weights.
    binding: u64,
}

fn next_weights_binding() -> u64 {
    static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
    NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
}

/// An owned handle on loaded Qwen3 weights for a paged session.
///
/// The tensors are MLX array handles: cloning them shares the device
/// buffers, so this copies no weight data. It lets a model thread hold a
/// [`crate::forward::PagedQwen3Session`] for its whole life while the
/// [`Qwen3MlxWeights`] it came from stays free for other borrows.
pub struct Qwen3PagedWeights {
    config: crate::forward::Qwen3ForwardConfig,
    tensors: HashMap<String, Array>,
}

impl Qwen3PagedWeights {
    /// The decoder configuration.
    #[must_use]
    pub const fn config(&self) -> &crate::forward::Qwen3ForwardConfig {
        &self.config
    }

    /// Sizes a pool from a K/V byte budget at these weights' K/V precision.
    ///
    /// # Errors
    ///
    /// Returns
    /// [`KvPoolConfig`](crate::forward::Qwen3ForwardError::KvPoolConfig) when
    /// the budget cannot hold a valid pool, and
    /// [`ShapeOverflow`](crate::forward::Qwen3ForwardError::ShapeOverflow) when
    /// the per-token size overflows.
    pub fn pool_for_budget(
        &self,
        budget_bytes: u64,
        block_tokens: engine::blocks::BlockTokens,
    ) -> Result<engine::blocks::PoolConfig, crate::forward::Qwen3ForwardError> {
        crate::forward::PagedQwen3Session::pool_for_budget(
            &self.config,
            &self.tensors,
            budget_bytes,
            block_tokens,
        )
    }

    /// Allocates a paged K/V pool over these weights.
    ///
    /// # Errors
    ///
    /// Returns
    /// [`CachedBidirectional`](crate::forward::Qwen3ForwardError::CachedBidirectional),
    /// [`KvPoolDtype`](crate::forward::Qwen3ForwardError::KvPoolDtype) or
    /// [`Mlx`](crate::forward::Qwen3ForwardError::Mlx) as
    /// `PagedQwen3Session::new` describes.
    pub fn session(
        &self,
        pool: engine::blocks::PoolConfig,
    ) -> Result<
        crate::forward::PagedQwen3Session<'_, std::collections::hash_map::RandomState>,
        crate::forward::Qwen3ForwardError,
    > {
        crate::forward::PagedQwen3Session::new(&self.config, &self.tensors, pool)
    }
}

impl Qwen3MlxWeights {
    /// An owned handle on these weights, in their current precision, for a
    /// paged session.
    #[must_use]
    pub fn paged_weights(&self) -> Qwen3PagedWeights {
        Qwen3PagedWeights {
            config: self.forward_config.clone(),
            tensors: self.tensors.clone(),
        }
    }

    /// Starts an independent sequence executor borrowing this checkpoint.
    /// Its KV state cannot be transferred to a different checkpoint.
    #[must_use]
    pub fn executor(
        &self,
    ) -> crate::forward::Qwen3ForwardExecutor<'_, std::collections::hash_map::RandomState> {
        crate::forward::Qwen3ForwardExecutor::new(&self.forward_config, &self.tensors)
    }

    /// Starts a resident-chat executor after validating its context and logical
    /// K/V estimate against this checkpoint's decoder configuration.
    ///
    /// # Errors
    ///
    /// Returns the errors of `Qwen3ForwardExecutor::resident`: a non-causal
    /// model, a context or KV budget that does not fit, or an MLX allocation
    /// failure.
    pub fn resident_chat_executor(
        &self,
        maximum_context_tokens: usize,
        maximum_kv_bytes: u64,
    ) -> Result<
        crate::forward::Qwen3ForwardExecutor<'_, std::collections::hash_map::RandomState>,
        crate::forward::Qwen3ForwardError,
    > {
        let plan = self.forward_config.resident_chat_plan(
            maximum_context_tokens,
            maximum_kv_bytes,
            crate::forward::kv_precision(&self.forward_config, &self.tensors)?,
        )?;
        Ok(crate::forward::Qwen3ForwardExecutor::new_for_resident_chat(
            &self.forward_config,
            &self.tensors,
            plan,
        ))
    }

    /// Detaches the first `tokens` cached positions of a resident executor
    /// borrowed from these weights, for a later
    /// [`Self::resident_chat_executor_from`] on the same weights.
    ///
    /// # Errors
    ///
    /// Returns [`KvSnapshotMismatch`](crate::forward::Qwen3ForwardError::KvSnapshotMismatch) when `executor`
    /// borrows other weights, and [`KvSnapshotUnsupported`](crate::forward::Qwen3ForwardError::KvSnapshotUnsupported)
    /// or [`CacheInconsistent`](crate::forward::Qwen3ForwardError::CacheInconsistent) when its cache cannot be
    /// snapshotted at `tokens`.
    pub fn snapshot_resident_prefix(
        &self,
        executor: &crate::forward::Qwen3ForwardExecutor<
            '_,
            std::collections::hash_map::RandomState,
        >,
        tokens: usize,
    ) -> Result<crate::forward::Qwen3KvSnapshot, crate::forward::Qwen3ForwardError> {
        if !executor.borrows(&self.tensors) {
            return Err(crate::forward::Qwen3ForwardError::KvSnapshotMismatch);
        }
        executor.snapshot_prefix(tokens, self.binding)
    }

    /// Starts a resident-chat executor whose cache already holds `snapshot`'s
    /// prefix. The snapshot must come from this load (after any precision
    /// change) and the same context and K/V limits.
    ///
    /// # Errors
    ///
    /// Returns
    /// [`KvSnapshotMismatch`](crate::forward::Qwen3ForwardError::KvSnapshotMismatch)
    /// when `snapshot` came from other weights, and the errors of
    /// `resident_chat_executor` otherwise.
    pub fn resident_chat_executor_from(
        &self,
        snapshot: &crate::forward::Qwen3KvSnapshot,
        maximum_context_tokens: usize,
        maximum_kv_bytes: u64,
    ) -> Result<
        crate::forward::Qwen3ForwardExecutor<'_, std::collections::hash_map::RandomState>,
        crate::forward::Qwen3ForwardError,
    > {
        let plan = self.forward_config.resident_chat_plan(
            maximum_context_tokens,
            maximum_kv_bytes,
            crate::forward::kv_precision(&self.forward_config, &self.tensors)?,
        )?;
        if snapshot.binding() != self.binding || snapshot.plan() != plan {
            return Err(crate::forward::Qwen3ForwardError::KvSnapshotMismatch);
        }
        Ok(crate::forward::Qwen3ForwardExecutor::from_snapshot(
            &self.forward_config,
            &self.tensors,
            snapshot,
        ))
    }

    /// Materializes float32 weights once for comparison with a CPU float32 oracle.
    /// This increases resident weight memory relative to the BF16 checkpoint;
    /// quantized weights are dequantized first.
    ///
    /// # Errors
    ///
    /// Returns
    /// [`ForwardConfig`](crate::metal::Qwen3MetalLoadError::ForwardConfig) with
    /// [`MissingWeight`](crate::forward::Qwen3ForwardError::MissingWeight) when
    /// a dense weight is absent, and
    /// [`Evaluation`](crate::metal::Qwen3MetalLoadError::Evaluation) when MLX
    /// cannot convert it.
    pub fn prepare_float32(&mut self) -> Result<(), Qwen3MetalLoadError> {
        self.prepare_precision(Qwen3WeightPrecision::Dense(Qwen3FloatPrecision::Float32))
    }

    /// How the loaded tensors are stored now.
    ///
    /// # Errors
    ///
    /// Returns
    /// [`ForwardConfig`](crate::metal::Qwen3MetalLoadError::ForwardConfig) when
    /// the loaded weights do not share one supported precision.
    pub fn precision(&self) -> Result<Qwen3WeightPrecision, Qwen3MetalLoadError> {
        let activations = crate::forward::kv_precision(&self.forward_config, &self.tensors)?;
        Ok(match self.forward_config.quantization() {
            None => Qwen3WeightPrecision::Dense(activations),
            Some(quantization) => Qwen3WeightPrecision::Affine {
                quantization,
                activations,
            },
        })
    }

    /// Converts the weights to `precision` once: dequantizing packed
    /// projections when it is dense or packed differently, casting every
    /// unpacked tensor to its float dtype, and quantizing projections and the
    /// token embedding with MLX's affine `quantize` when it is affine. K/V
    /// caches take the resulting activation dtype.
    ///
    /// # Errors
    ///
    /// Returns
    /// [`ForwardConfig`](crate::metal::Qwen3MetalLoadError::ForwardConfig) with
    /// [`MissingWeight`](crate::forward::Qwen3ForwardError::MissingWeight) when
    /// a dense weight is absent, and
    /// [`Evaluation`](crate::metal::Qwen3MetalLoadError::Evaluation) when MLX
    /// cannot convert it.
    #[tracing::instrument(
        name = "qwen.weights.prepare_precision",
        level = "info",
        skip_all,
        fields(precision = tracing::field::Empty)
    )]
    pub fn prepare_precision(
        &mut self,
        precision: impl Into<Qwen3WeightPrecision>,
    ) -> Result<(), Qwen3MetalLoadError> {
        let precision = precision.into();
        tracing::Span::current().record("precision", tracing::field::debug(precision));
        let stream = StreamOrDevice::gpu();
        // A tied checkpoint may still ship an `lm_head` the forward never
        // reads; it is left as it is rather than packed.
        let tied = self.forward_config.tied_output_embedding();
        let stems: Vec<String> = self
            .tensors
            .keys()
            .filter(|name| crate::forward::is_quantizable(name))
            .filter(|name| !(tied && name.as_str() == "lm_head.weight"))
            .map(|name| name.trim_end_matches(".weight").to_owned())
            .collect();
        if self
            .forward_config
            .quantization()
            .is_some_and(|stored| precision.quantization() != Some(stored))
        {
            for stem in &stems {
                let dense =
                    crate::forward::dense_weight(&self.forward_config, &self.tensors, stem)?;
                dense.eval()?;
                self.tensors.insert(format!("{stem}.weight"), dense);
                self.tensors.remove(&format!("{stem}.scales"));
                self.tensors.remove(&format!("{stem}.biases"));
            }
            self.forward_config.set_quantization(None);
        }
        let dtype = precision.activations().dtype();
        for weight in self.tensors.values_mut() {
            if weight.dtype() == dtype || weight.dtype() == Dtype::Uint32 {
                continue;
            }
            let converted = weight.as_dtype_device(dtype, &stream)?;
            converted.eval()?;
            *weight = converted;
        }
        if let Some(quantization) = precision.quantization()
            && self.forward_config.quantization().is_none()
        {
            for stem in &stems {
                let name = format!("{stem}.weight");
                let dense = self.tensors.get(&name).ok_or_else(|| {
                    crate::forward::Qwen3ForwardError::MissingWeight(name.clone())
                })?;
                let (packed, scales, biases) = mlx_rs::ops::quantize_device(
                    dense,
                    quantization.group_size(),
                    quantization.bits(),
                    &stream,
                )?;
                mlx_rs::transforms::eval([&packed, &scales, &biases])?;
                self.tensors.insert(name, packed);
                self.tensors.insert(format!("{stem}.scales"), scales);
                self.tensors.insert(format!("{stem}.biases"), biases);
            }
            self.forward_config.set_quantization(Some(quantization));
        }
        // K/V computed from the earlier precision must not be restored here.
        self.binding = next_weights_binding();
        Ok(())
    }

    /// Executes the dense qualification decoder using these checkpoint tensors.
    ///
    /// # Errors
    ///
    /// * [`EmptyInput`](crate::forward::Qwen3ForwardError::EmptyInput),
    ///   [`InvalidTokenId`](crate::forward::Qwen3ForwardError::InvalidTokenId)
    ///   and
    ///   [`PromptTooLong`](crate::forward::Qwen3ForwardError::PromptTooLong)
    ///   when the input is empty, names a token outside the vocabulary, or
    ///   passes the context limit.
    /// * [`MissingWeight`](crate::forward::Qwen3ForwardError::MissingWeight)
    ///   and [`Mlx`](crate::forward::Qwen3ForwardError::Mlx) when a weight is
    ///   absent or MLX cannot build or evaluate the graph.
    pub fn forward_last_logits(
        &self,
        input_ids: &[i32],
    ) -> Result<Vec<f32>, crate::forward::Qwen3ForwardError> {
        crate::forward::forward_last_logits(&self.tensors, &self.forward_config, input_ids)
    }

    /// Embeds one sequence encoded with the tokenizer's special-token template:
    /// the final-norm hidden state at its last position (the appended
    /// `<|endoftext|>`), truncated to `dimensions` if given, then L2-normalized.
    ///
    /// # Errors
    ///
    /// Returns [`crate::embedding::Qwen3EmbeddingError`] when the IDs lack the
    /// appended `<|endoftext|>`, the dimension is unsupported, or the forward
    /// pass fails.
    pub fn embed(
        &self,
        input_ids: &[i32],
        dimensions: Option<usize>,
    ) -> Result<Vec<f32>, crate::embedding::Qwen3EmbeddingError> {
        self.require_attention(crate::Qwen3Attention::Causal)?;
        crate::embedding::check_embedding_input(input_ids)?;
        let hidden =
            crate::forward::forward_last_hidden(&self.tensors, &self.forward_config, input_ids)?;
        crate::embedding::normalize_embedding(&hidden, dimensions)
    }

    /// [`Self::embed`] for several sequences, returning one vector per
    /// sequence in input order. Sequences of similar length share one
    /// right-padded forward pass; see
    /// [`crate::embedding::EMBEDDING_BATCH_PADDING_RATIO`].
    ///
    /// # Errors
    ///
    /// Returns [`crate::embedding::Qwen3EmbeddingError`] as [`Self::embed`]
    /// does, for the first sequence that fails.
    pub fn embed_batch(
        &self,
        sequences: &[&[i32]],
        dimensions: Option<usize>,
    ) -> Result<Vec<Vec<f32>>, crate::embedding::Qwen3EmbeddingError> {
        self.require_attention(crate::Qwen3Attention::Causal)?;
        for input_ids in sequences {
            crate::embedding::check_embedding_input(input_ids)?;
        }
        let lengths: Vec<usize> = sequences.iter().map(|ids| ids.len()).collect();
        let mut vectors = vec![Vec::new(); sequences.len()];
        for group in crate::embedding::padding_groups(
            &lengths,
            crate::embedding::EMBEDDING_BATCH_PADDING_RATIO,
        ) {
            let members: Vec<&[i32]> = group.iter().map(|&index| sequences[index]).collect();
            let hidden = crate::forward::forward_last_hidden_batch(
                &self.tensors,
                &self.forward_config,
                &members,
            )?;
            for (index, row) in group.into_iter().zip(hidden) {
                vectors[index] = crate::embedding::normalize_embedding(&row, dimensions)?;
            }
        }
        Ok(vectors)
    }

    /// Embeds each chunk of one pplx-embed-context document, encoded from
    /// [`crate::embedding::join_context_chunks`]: the mean of its final-norm
    /// hidden states under bidirectional attention, one float vector per chunk.
    ///
    /// Apply [`crate::embedding::quantize_int8_tanh`] (the model's default) or
    /// [`crate::embedding::quantize_binary`] to match its published outputs.
    ///
    /// # Errors
    ///
    /// Returns [`crate::embedding::Qwen3EmbeddingError`] when the checkpoint is
    /// not bidirectional or the forward pass fails.
    pub fn embed_context_chunks(
        &self,
        input_ids: &[i32],
    ) -> Result<Vec<Vec<f32>>, crate::embedding::Qwen3EmbeddingError> {
        self.require_attention(crate::Qwen3Attention::Bidirectional)?;
        let hidden =
            crate::forward::forward_hidden_states(&self.tensors, &self.forward_config, input_ids)?;
        crate::embedding::mean_pool_context_chunks(
            &hidden,
            self.forward_config.hidden_size(),
            input_ids,
        )
    }

    /// Returns whether this checkpoint's decoder attends causally or
    /// bidirectionally, which decides the pooling it was trained for.
    #[must_use]
    pub fn attention(&self) -> crate::Qwen3Attention {
        self.forward_config.attention()
    }

    /// Every position's final-norm hidden state for one sequence, flattened
    /// `[positions, hidden_size]`, under this checkpoint's attention.
    ///
    /// # Errors
    ///
    /// Returns [`crate::forward::Qwen3ForwardError`] when the forward pass fails.
    pub fn hidden_states(
        &self,
        input_ids: &[i32],
    ) -> Result<Vec<f32>, crate::forward::Qwen3ForwardError> {
        crate::forward::forward_hidden_states(&self.tensors, &self.forward_config, input_ids)
    }

    /// Pre-norm residual streams after the selected (1-based) decoder layers
    /// of one right-padded sequence; see
    /// [`crate::forward::forward_layer_states`].
    ///
    /// # Errors
    ///
    /// Returns [`crate::forward::Qwen3ForwardError`] for a bidirectional
    /// checkpoint, an invalid layer selection or real length, or a failed
    /// forward pass.
    pub fn forward_layer_states(
        &self,
        input_ids: &[i32],
        real_len: usize,
        layers: &[usize],
    ) -> Result<Vec<Array>, crate::forward::Qwen3ForwardError> {
        crate::forward::forward_layer_states(
            &self.tensors,
            &self.forward_config,
            input_ids,
            real_len,
            layers,
        )
    }

    fn require_attention(
        &self,
        required: crate::Qwen3Attention,
    ) -> Result<(), crate::embedding::Qwen3EmbeddingError> {
        if self.forward_config.attention() == required {
            Ok(())
        } else {
            Err(crate::embedding::Qwen3EmbeddingError::AttentionMismatch { required })
        }
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
    #[tracing::instrument(
        name = "qwen.weights.load",
        level = "info",
        skip_all,
        fields(model_dir = %model_dir.as_ref().display())
    )]
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
        let mut tied_duplicates = 0_usize;
        for shard in inspection.shards() {
            // MLX safetensors I/O is defined on the CPU stream. Subsequent
            // graph operations choose the GPU stream explicitly.
            let shard_tensors = Array::load_safetensors_device(shard, StreamOrDevice::cpu())?;
            // Inspection keyed tensors by canonical name; the loaded map must
            // agree so forward code sees one naming for every checkpoint.
            let naming = inspection.tensor_naming();
            for (name, tensor) in shard_tensors {
                // Tied checkpoints may still store `lm_head.weight`. Logits
                // project through the embedding, so the copy is never read,
                // and keeping it would count its bytes in the resident total.
                if name == "lm_head.weight" && forward_config.tied_output_embedding() {
                    tied_duplicates += 1;
                    continue;
                }
                tensors.insert(naming.canonical_name(&name), tensor);
            }
        }

        if tensors.len() + tied_duplicates != inspection.tensor_count() {
            return Err(Qwen3MetalLoadError::UnexpectedTensorCount {
                expected: inspection.tensor_count() - tied_duplicates,
                actual: tensors.len(),
            });
        }

        let vocab = inspection.contract().vocab_size();
        let hidden = inspection.contract().hidden_size();
        let embedding_shape = vec![
            i32::try_from(vocab)
                .map_err(|_| Qwen3MetalLoadError::DimensionOutOfRange("vocab_size"))?,
            i32::try_from(hidden)
                .map_err(|_| Qwen3MetalLoadError::DimensionOutOfRange("hidden_size"))?,
        ];
        // Packed rows hold several elements per word.
        let stored_columns = forward_config
            .quantization()
            .map_or(u64::from(hidden), |quantization| {
                quantization.packed_columns(u64::from(hidden))
            });
        let expected_embedding_shape = [
            embedding_shape[0],
            i32::try_from(stored_columns)
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

        Ok(Self {
            inspection,
            forward_config,
            tensors,
            embedding_shape,
            binding: next_weights_binding(),
        })
    }

    /// Returns the number of loaded tensors, excluding a tied checkpoint's
    /// stored copy of `lm_head.weight`.
    #[must_use]
    pub fn tensor_count(&self) -> usize {
        self.tensors.len()
    }

    /// Logical bytes in the current tensor dtypes, excluding allocator overhead.
    #[must_use]
    pub fn logical_weight_bytes(&self) -> usize {
        self.tensors.values().map(Array::nbytes).sum()
    }

    /// Returns the token embedding's logical `[vocab, hidden]` shape,
    /// whether its rows are stored dense or packed.
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
        let vectors = crate::forward::embed_rows(&self.forward_config, &self.tensors, &token_ids)?;
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
    /// A streamed executor promised fewer total tokens than its initial prompt.
    #[error("streamed executor maximum {maximum} is smaller than prompt length {prompt_tokens}")]
    StreamMaximumBelowPrompt {
        /// Total context requested by the caller.
        maximum: usize,
        /// Tokens that must be consumed during prefill.
        prompt_tokens: usize,
    },
    /// A streamed executor requested more context than this qualified path supports.
    #[error("streamed executor requested {requested} total tokens, maximum is {maximum}")]
    StreamContextLimit {
        /// Total context requested by the caller.
        requested: usize,
        /// Effective model and qualification limit.
        maximum: usize,
    },
    /// Decode was called before the executor established its KV cache.
    #[error("streamed executor decode requires a successful prefill")]
    StreamDecodeWithoutPrefill,
    /// Prefill is intentionally single-use for this sequence executor.
    #[error("streamed executor prefill may only run once")]
    StreamPrefillAlreadyDone,
    /// The executor observed an execution failure after cache mutation and cannot be reused.
    #[error("streamed executor is poisoned after a failed cache-mutating operation")]
    StreamPoisoned,
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
    UnexpectedTensorCount {
        /// Tensors the validated headers declare.
        expected: usize,
        /// Tensors MLX loaded.
        actual: usize,
    },
    /// The required token embedding was absent after loading.
    #[error("MLX did not load model.embed_tokens.weight")]
    MissingEmbedding,
    /// MLX reported an embedding shape different from the model contract.
    #[error("MLX embedding shape {actual:?}, expected {expected:?}")]
    UnexpectedEmbeddingShape {
        /// `[vocab_size, hidden_size]`.
        expected: Vec<i32>,
        /// The loaded shape.
        actual: Vec<i32>,
    },
    /// A forward lookup needs at least one token ID.
    #[error("Qwen3 embedding lookup requires at least one token ID")]
    EmptyInputIds,
    /// Token IDs must name rows in the configured embedding vocabulary.
    #[error("token ID {token} is outside vocabulary 0..{vocab_size}")]
    InvalidTokenId {
        /// The token ID.
        token: i32,
        /// The vocabulary size.
        vocab_size: u32,
    },
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
    fn bf16_widening_handles_unaligned_slices_and_vector_tails() {
        for offset in 0..16 {
            for count in 0..=65 {
                let words: Vec<u16> = (0..count)
                    .map(|index| [0, 0x8000, 1, 0x807f, 0x3f80, 0xff7f][index % 6])
                    .collect();
                let mut bytes = vec![0xff; offset];
                bytes.extend(words.iter().flat_map(|word| word.to_le_bytes()));
                let values = super::decode_bf16(&bytes[offset..]).expect("finite unaligned words");
                assert_eq!(values.len(), count);
                for (value, word) in values.iter().zip(words) {
                    assert_eq!(value.to_bits(), u32::from(word) << 16);
                }
            }
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

    /// Writes a one-layer BF16 checkpoint whose names carry `prefix`. With
    /// `stored_head`, the tied checkpoint also stores an `lm_head.weight`
    /// whose values differ from the embedding.
    fn write_tiny_checkpoint(dir: &std::path::Path, prefix: &str, stored_head: bool) {
        std::fs::create_dir_all(dir).expect("checkpoint dir");
        std::fs::write(
            dir.join("config.json"),
            r#"{"model_type":"qwen3","num_hidden_layers":1,"hidden_size":4,"intermediate_size":8,"vocab_size":8,"num_attention_heads":2,"num_key_value_heads":1,"head_dim":4,"max_position_embeddings":16,"rms_norm_eps":0.000001,"rope_theta":1000000,"hidden_act":"silu","tie_word_embeddings":true,"attention_bias":false,"mlp_bias":false}"#,
        )
        .expect("config");
        let layer = "layers.0";
        let mut tensors: Vec<(String, Vec<i64>)> = [
            ("embed_tokens.weight".to_owned(), vec![8, 4]),
            ("norm.weight".to_owned(), vec![4]),
            (format!("{layer}.input_layernorm.weight"), vec![4]),
            (format!("{layer}.post_attention_layernorm.weight"), vec![4]),
            (format!("{layer}.self_attn.q_norm.weight"), vec![4]),
            (format!("{layer}.self_attn.k_norm.weight"), vec![4]),
            (format!("{layer}.self_attn.q_proj.weight"), vec![8, 4]),
            (format!("{layer}.self_attn.k_proj.weight"), vec![4, 4]),
            (format!("{layer}.self_attn.v_proj.weight"), vec![4, 4]),
            (format!("{layer}.self_attn.o_proj.weight"), vec![4, 8]),
            (format!("{layer}.mlp.gate_proj.weight"), vec![8, 4]),
            (format!("{layer}.mlp.up_proj.weight"), vec![8, 4]),
            (format!("{layer}.mlp.down_proj.weight"), vec![4, 8]),
        ]
        .into();
        if stored_head {
            tensors.push(("lm_head.weight".to_owned(), vec![8, 4]));
        }
        let mut header = serde_json::Map::new();
        let mut payload = Vec::new();
        for (name, shape) in tensors {
            let count = usize::try_from(shape.iter().product::<i64>()).expect("small");
            let start = payload.len();
            // A different period keeps the stored head unlike the embedding.
            let period = if name == "lm_head.weight" { 7 } else { 11 };
            for index in 0..count {
                // Multiples of 1/16 are exact in BF16.
                let value = f32::from(u8::try_from(index % period).expect("small") + 1) / 16.0;
                payload.extend_from_slice(
                    &u16::try_from(value.to_bits() >> 16)
                        .expect("bf16")
                        .to_le_bytes(),
                );
            }
            // Hugging Face stores the head outside the `model.` namespace.
            let key = if name == "lm_head.weight" {
                name
            } else {
                format!("{prefix}{name}")
            };
            header.insert(
                key,
                serde_json::json!({"dtype":"BF16","shape":shape,"data_offsets":[start, payload.len()]}),
            );
        }
        let header = serde_json::to_vec(&header).expect("header");
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend(header);
        bytes.extend(payload);
        std::fs::write(dir.join("model.safetensors"), bytes).expect("shard");
    }

    #[test]
    fn bare_and_prefixed_checkpoints_load_to_the_same_decoder() {
        let _guard = crate::GPU_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let root =
            std::env::temp_dir().join(format!("metallix-qwen-naming-{}", std::process::id()));
        write_tiny_checkpoint(&root.join("prefixed"), "model.", false);
        write_tiny_checkpoint(&root.join("bare"), "", false);
        let prefixed = super::Qwen3MlxWeights::load(root.join("prefixed")).expect("prefixed load");
        let bare = super::Qwen3MlxWeights::load(root.join("bare")).expect("bare load");
        std::fs::remove_dir_all(&root).expect("remove fixture");

        let ids = [1_i32, 5, 3];
        let expected = prefixed
            .forward_last_logits(&ids)
            .expect("prefixed forward");
        let actual = bare.forward_last_logits(&ids).expect("bare forward");
        assert_eq!(
            actual.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
        // Embedding input without the appended end-of-text is refused up front.
        assert!(matches!(
            bare.embed(&ids, None),
            Err(crate::embedding::Qwen3EmbeddingError::MissingEndOfText { last: Some(3) })
        ));
    }

    #[test]
    fn tied_checkpoint_drops_its_stored_head_from_resident_weights() {
        let _guard = crate::GPU_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let root = std::env::temp_dir().join(format!("metallix-qwen-tied-{}", std::process::id()));
        write_tiny_checkpoint(&root.join("stored"), "model.", true);
        write_tiny_checkpoint(&root.join("absent"), "model.", false);
        let stored = super::Qwen3MlxWeights::load(root.join("stored")).expect("stored-head load");
        let absent = super::Qwen3MlxWeights::load(root.join("absent")).expect("no-head load");
        std::fs::remove_dir_all(&root).expect("remove fixture");

        // The header declares the head (8 x 4 BF16 = 64 bytes); residency omits it.
        let inspection = stored.inspection();
        assert_eq!(stored.tensor_count() + 1, inspection.tensor_count());
        assert_eq!(stored.tensor_count(), absent.tensor_count());
        assert_eq!(
            u64::try_from(stored.logical_weight_bytes()).expect("small") + 64,
            inspection.tensor_bytes()
        );
        assert_eq!(stored.logical_weight_bytes(), absent.logical_weight_bytes());
        // The stored head differs from the embedding, so equal logits show
        // the projection still reads the embedding.
        let ids = [1_i32, 5, 3];
        let bits = |weights: &super::Qwen3MlxWeights| {
            weights
                .forward_last_logits(&ids)
                .expect("forward")
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>()
        };
        assert_eq!(bits(&stored), bits(&absent));
    }

    #[test]
    fn evaluates_a_known_gpu_matrix_product() {
        let _guard = crate::GPU_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let smoke = run_metal_smoke().expect("MLX Metal must evaluate the smoke graph");
        assert!((smoke.product() - 6.0).abs() <= f32::EPSILON);
    }
}
