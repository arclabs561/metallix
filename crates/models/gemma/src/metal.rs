//! Gemma 4 text weights as MLX arrays.

#![allow(
    deprecated,
    reason = "mlx-rs 0.32 deprecates the *_device ops; the with_stream migration is a separate change"
)]

use std::{collections::HashMap, path::Path};

use mlx_rs::{Array, Dtype, StreamOrDevice};
use thiserror::Error;
use tracing::instrument;

use crate::{
    Gemma4LayerKind, Gemma4TextConfig,
    checkpoint::{Gemma4CheckpointError, Gemma4CheckpointInspection, expected_text_tensors},
    forward::{Gemma4Executor, Gemma4ForwardError},
};

/// Compute precision for weights, activations and K/V.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Gemma4Precision {
    /// The checkpoint's bf16 values; required for 31B on a 128 GB Mac.
    BFloat16,
    /// Float32 copies, for comparison with a float32 source oracle. This
    /// doubles resident weight memory relative to bf16.
    Float32,
}

impl Gemma4Precision {
    const fn dtype(self) -> Dtype {
        match self {
            Self::BFloat16 => Dtype::Bfloat16,
            Self::Float32 => Dtype::Float32,
        }
    }
}

/// Text weights and the per-precision constants the forward pass needs.
pub struct Gemma4MlxWeights {
    config: Gemma4TextConfig,
    pub(crate) tensors: HashMap<String, Array>,
    precision: Gemma4Precision,
    /// `sqrt(hidden_size)` rounded to the compute dtype, as the source does.
    pub(crate) embed_scale: Array,
    /// Unit weights for the V norm, which has no stored scale.
    pub(crate) sliding_value_norm: Array,
    pub(crate) full_value_norm: Array,
    pub(crate) sliding_wavelengths: Option<Array>,
    pub(crate) full_wavelengths: Option<Array>,
}

impl Gemma4MlxWeights {
    /// Validates every header, then loads the text tensors at `precision`.
    /// Vision and audio tensors are dropped after loading.
    ///
    /// # Errors
    ///
    /// Returns [`Gemma4LoadError`] when inspection fails or MLX cannot load
    /// or convert a tensor.
    #[instrument(name = "gemma.weights.load", level = "info", skip_all,
        fields(model_dir = %model_dir.as_ref().display()))]
    pub fn load(
        model_dir: impl AsRef<Path>,
        precision: Gemma4Precision,
    ) -> Result<Self, Gemma4LoadError> {
        let inspection = Gemma4CheckpointInspection::inspect(model_dir)?;
        let by_stored_name: HashMap<&str, &str> = inspection
            .tensors()
            .iter()
            .map(|(canonical, header)| (header.stored_name.as_str(), canonical.as_str()))
            .collect();
        let mut tensors = HashMap::with_capacity(by_stored_name.len());
        for shard in inspection.shards() {
            // MLX safetensors I/O runs on the CPU stream; conversions and the
            // forward graph use the GPU stream.
            for (name, tensor) in Array::load_safetensors_device(shard, StreamOrDevice::cpu())? {
                if let Some(canonical) = by_stored_name.get(name.as_str()) {
                    tensors.insert((*canonical).to_owned(), convert(tensor, precision)?);
                }
            }
        }
        Self::new(inspection.config().clone(), tensors, precision)
    }

    /// Wraps already-loaded text tensors keyed by their unprefixed names.
    ///
    /// # Errors
    ///
    /// Returns [`Gemma4LoadError`] when a tensor is missing, extra or has the
    /// wrong shape or dtype.
    pub fn from_tensors(
        config: Gemma4TextConfig,
        tensors: HashMap<String, Array>,
        precision: Gemma4Precision,
    ) -> Result<Self, Gemma4LoadError> {
        let tensors = tensors
            .into_iter()
            .map(|(name, tensor)| Ok((name, convert(tensor, precision)?)))
            .collect::<Result<_, Gemma4LoadError>>()?;
        Self::new(config, tensors, precision)
    }

    fn new(
        config: Gemma4TextConfig,
        tensors: HashMap<String, Array>,
        precision: Gemma4Precision,
    ) -> Result<Self, Gemma4LoadError> {
        let expected = expected_text_tensors(&config);
        if tensors.len() != expected.len() {
            return Err(Gemma4LoadError::TensorCount {
                expected: expected.len(),
                actual: tensors.len(),
            });
        }
        for (name, shape) in &expected {
            let tensor = tensors
                .get(name)
                .ok_or_else(|| Gemma4LoadError::MissingTensor(name.clone()))?;
            let actual: Vec<usize> = tensor
                .shape()
                .iter()
                .map(|&dimension| usize::try_from(dimension).unwrap_or(usize::MAX))
                .collect();
            if &actual != shape || tensor.dtype() != precision.dtype() {
                return Err(Gemma4LoadError::TensorLayout(name.clone()));
            }
        }
        let gpu = StreamOrDevice::gpu();
        let dtype = precision.dtype();
        let embed_scale = Array::from_f32(config.embed_scale()).as_dtype_device(dtype, &gpu)?;
        let unit = |kind| -> Result<Array, Gemma4LoadError> {
            let head_dim = i32::try_from(config.attention(kind).head_dim)
                .map_err(|_| Gemma4LoadError::TensorLayout("head_dim".into()))?;
            Ok(Array::ones_device::<f32>(&[head_dim], &gpu)?.as_dtype_device(dtype, &gpu)?)
        };
        let wavelengths = |kind| {
            config.attention(kind).rope_wavelengths().map(|values| {
                Array::from_slice(&values, &[i32::try_from(values.len()).unwrap_or(i32::MAX)])
            })
        };
        let weights = Self {
            sliding_value_norm: unit(Gemma4LayerKind::Sliding)?,
            full_value_norm: unit(Gemma4LayerKind::Full)?,
            sliding_wavelengths: wavelengths(Gemma4LayerKind::Sliding),
            full_wavelengths: wavelengths(Gemma4LayerKind::Full),
            config,
            tensors,
            precision,
            embed_scale,
        };
        // Force the embedding through a GPU graph so a bad payload fails at
        // load rather than on the first request.
        weights
            .tensor("embed_tokens.weight")?
            .sum_device(None, &gpu)?
            .eval()?;
        Ok(weights)
    }

    /// The validated configuration.
    #[must_use]
    pub const fn config(&self) -> &Gemma4TextConfig {
        &self.config
    }

    /// The compute precision of these weights.
    #[must_use]
    pub const fn precision(&self) -> Gemma4Precision {
        self.precision
    }

    /// Logical bytes of the text tensors, excluding allocator overhead.
    #[must_use]
    pub fn logical_weight_bytes(&self) -> usize {
        self.tensors.values().map(Array::nbytes).sum()
    }

    /// Starts a sequence executor admitting at most `maximum_context_tokens`.
    ///
    /// # Errors
    ///
    /// Returns [`Gemma4ForwardError::ContextLimit`] for a zero limit or one
    /// beyond the checkpoint's declared positions.
    pub fn executor(
        &self,
        maximum_context_tokens: usize,
    ) -> Result<Gemma4Executor<'_>, Gemma4ForwardError> {
        Gemma4Executor::new(self, maximum_context_tokens)
    }

    pub(crate) fn tensor(&self, name: &str) -> Result<&Array, Gemma4ForwardError> {
        self.tensors
            .get(name)
            .ok_or_else(|| Gemma4ForwardError::MissingWeight(name.to_owned()))
    }
}

fn convert(tensor: Array, precision: Gemma4Precision) -> Result<Array, Gemma4LoadError> {
    let converted = if tensor.dtype() == precision.dtype() {
        tensor
    } else {
        tensor.as_dtype_device(precision.dtype(), StreamOrDevice::gpu())?
    };
    // Evaluate one tensor at a time so only one source copy is alive. An
    // unevaluated safetensors array would instead be read during the first
    // request (1.7 s of a 20-token 12B prefill) and decode measured 144 ms
    // per token against 110 ms with every tensor evaluated at load.
    converted.eval()?;
    Ok(converted)
}

/// A failure while loading Gemma 4 weights onto Metal.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum Gemma4LoadError {
    /// Header validation failed before any payload was loaded.
    #[error(transparent)]
    Checkpoint(#[from] Gemma4CheckpointError),
    /// MLX failed to read a safetensors shard.
    #[error("MLX could not load a shard: {0}")]
    Io(#[from] mlx_rs::error::IoError),
    /// MLX failed to convert or evaluate a tensor.
    #[error("MLX failed: {0}")]
    Mlx(#[from] mlx_rs::error::Exception),
    /// The tensor set differs in size from the configuration's layout.
    #[error("expected {expected} text tensors, found {actual}")]
    TensorCount {
        /// Tensors the configuration requires.
        expected: usize,
        /// Tensors supplied.
        actual: usize,
    },
    /// A required tensor is absent.
    #[error("missing text tensor {0}")]
    MissingTensor(String),
    /// A tensor's shape or dtype differs from the configuration.
    #[error("text tensor {0} has an unexpected shape or dtype")]
    TensorLayout(String),
    /// Executor construction failed.
    #[error(transparent)]
    Forward(#[from] Gemma4ForwardError),
}
