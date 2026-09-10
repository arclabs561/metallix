//! Narrow MLX-on-Metal qualification for the Qwen adapter.
//!
//! This is deliberately a substrate smoke test, not a Qwen forward path. It
//! proves the pinned Rust binding can construct, evaluate, and read back a GPU
//! graph on the supported host before model tensors are introduced.

use std::{collections::HashMap, path::Path};

use mlx_rs::{Array, StreamOrDevice};
use thiserror::Error;

use crate::checkpoint::{Qwen3CheckpointError, Qwen3CheckpointInspection};

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
    if (product - 6.0).abs() > f32::EPSILON {
        return Err(Qwen3MetalSmokeError::UnexpectedProduct(product));
    }
    Ok(Qwen3MetalSmoke { product })
}

/// A Qwen3 checkpoint represented as MLX arrays for Metal execution.
///
/// This owns the loaded tensor map for a later forward implementation. Loading
/// proves that MLX accepts the checkpoint's actual tensor payloads; it does
/// not by itself execute the decoder.
pub struct Qwen3MlxWeights {
    inspection: Qwen3CheckpointInspection,
    tensors: HashMap<String, Array>,
    embedding_shape: Vec<i32>,
}

impl Qwen3MlxWeights {
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
            tensors,
            embedding_shape,
        })
    }

    /// Returns the number of loaded tensors.
    #[must_use]
    pub fn tensor_count(&self) -> usize {
        self.tensors.len()
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
}

#[cfg(test)]
mod tests {
    use super::run_metal_smoke;

    #[test]
    fn evaluates_a_known_gpu_matrix_product() {
        let smoke = run_metal_smoke().expect("MLX Metal must evaluate the smoke graph");
        assert!((smoke.product() - 6.0).abs() <= f32::EPSILON);
    }
}
