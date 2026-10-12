//! Metal execution of the hybrid decoder: weight loading, prefill and cached
//! decode with `GatedDeltaNet` recurrent state alongside full-attention K/V.
//!
//! Every operator runs on the MLX GPU stream; only final logits are read back.
//! The `GatedDeltaNet` recurrence is evaluated in 64-token chunks for
//! multi-token calls and token by token for decode (see the crate
//! documentation). Prefill is split into bounded chunks whose graphs are
//! evaluated before the next chunk starts.

#![allow(
    deprecated,
    reason = "mlx-rs 0.32 deprecates the *_device ops; the with_stream migration is a separate change"
)]

use std::{
    cell::RefCell,
    collections::HashMap,
    fs,
    path::{Path, PathBuf},
    sync::Arc,
};

use blockfloat::gguf::{AffineParameters, GgufDecodeError, GgufEncoding, decode, repack_affine};
use checkpoint::{CheckpointError, TensorEncoding, TensorInfo, TensorSource, gguf::GgufFile};
use mlx_rs::{
    Array, Dtype, StreamOrDevice, fast,
    ops::{
        self,
        indexing::{Ellipsis, IndexMutOp, IndexOp},
    },
};
use serde::Deserialize;
use thiserror::Error;

use crate::{
    Qwen35Config, Qwen35ConfigError, Qwen35LayerKind, Qwen35Mlp, RopeScalingRequest,
    gguf::{ValueHeadAxis, canonical_name, permute_spans, tiled_positions, value_head_axis},
};

mod delta;
#[cfg(test)]
mod mlx_affine_tests;
mod paged;
#[cfg(test)]
mod paged_tests;

pub use paged::{Qwen35PagerStats, Qwen35Paging};

/// Tokens per prefill graph. This bounds graph size even though the delta
/// rule runs in 64-token chunks; longer prompts use successive graphs.
pub const PREFILL_CHUNK_TOKENS: usize = 128;

/// Bytes a verify may keep to roll back (see [`Qwen35Executor::verify_logits`]).
/// On Qwen3.5-9B at BF16 each verified token stashes about 400 KB across the
/// linear layers, so this admits over 128 tokens there.
const VERIFY_STASH_BYTES: usize = 64 << 20;

const TEXT_PREFIX: &str = "model.language_model.";
/// Multi-token-prediction head tensors keep this prefix in the tensor map.
const MTP_PREFIX: &str = "mtp.";
/// The text decoder's prefix in mlx-lm's layout, and its untied output head.
const MLX_TEXT_PREFIX: &str = "language_model.model.";
const MLX_HEAD_PREFIX: &str = "language_model.lm_head.";
const L2_NORM_EPS: f32 = 1e-6;

/// Weight and activation precision.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Qwen35Precision {
    /// Keep the checkpoint's stored dtypes (BF16 for published weights).
    /// Recurrent state and decay math stay f32, as in the source.
    Checkpoint,
    /// Materialize every weight as f32, for comparison with an f32 oracle.
    Float32,
}

/// A loaded `qwen3_5` text decoder.
pub struct Qwen35Weights {
    identity: Arc<()>,
    config: Qwen35Config,
    /// Dense tensors by name without the text-model prefix. Each linear
    /// layer's decay rate `exp(A_log)` is stored as `decay_rate`, in f32.
    tensors: HashMap<String, Array>,
    /// Projection and embedding matrices held in MLX affine quantization.
    quantized: HashMap<String, AffineWeight>,
    precision: Qwen35Precision,
    /// The activation dtype.
    compute: Dtype,
    /// `YaRN` rotary arrays when the configuration's scaling is in force.
    yarn: Option<YarnArrays>,
    numerics: Numerics,
    /// Routed experts read on demand instead of held in `quantized`.
    paged: Option<paged::PagedExperts>,
}

/// Precomputed `YaRN` inputs to MLX's rope: the per-pair periods it rotates
/// by (`1 / inverse frequency`), and a per-feature scale over the head that
/// multiplies the rotary features by the attention factor and leaves the
/// rest unchanged, as Transformers' scaled cos and sin do.
struct YarnArrays {
    periods: Array,
    feature_scale: Array,
}

fn yarn_arrays(config: &Qwen35Config, compute: Dtype) -> Result<Option<YarnArrays>, Qwen35Error> {
    let Some(scaling) = config.rope_scaling() else {
        return Ok(None);
    };
    let gpu = StreamOrDevice::gpu();
    let periods: Vec<f32> = scaling
        .inverse_frequencies(config.rotary_dim, config.rope_theta)
        .into_iter()
        .map(|inverse| 1.0 / inverse)
        .collect();
    #[allow(
        clippy::cast_possible_truncation,
        reason = "the attention factor is near 1 and multiplies in the activation dtype"
    )]
    let factor = scaling.attention_factor() as f32;
    let scale: Vec<f32> = (0..config.head_dim)
        .map(|feature| {
            if feature < config.rotary_dim {
                factor
            } else {
                1.0
            }
        })
        .collect();
    let periods = Array::from_slice(&periods, &[dim(periods.len())?]);
    let feature_scale =
        Array::from_slice(&scale, &[dim(scale.len())?]).as_dtype_device(compute, &gpu)?;
    mlx_rs::transforms::eval([&periods, &feature_scale])?;
    Ok(Some(YarnArrays {
        periods,
        feature_scale,
    }))
}

/// Which reference's tensor layout and op sequence the forward pass follows
/// where the two round differently. The two paths deliberately follow
/// different references: Hugging Face and GGUF checkpoints follow
/// Transformers; checkpoints in mlx-lm's layout (the MLX affine ones
/// mlx-community publishes) follow mlx-lm, which produced them and defines
/// what their values compute.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Numerics {
    Transformers,
    MlxLm,
}

/// Whether [`Qwen35Weights::load_with`] keeps the checkpoint's
/// multi-token-prediction head.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum MtpHead {
    /// Drop the head's tensors unread, as [`Qwen35Weights::load`] does.
    #[default]
    Skip,
    /// Load the one-layer dense head, sharing the decoder's embedding and LM
    /// head, for drafting.
    Load,
}

/// Optional parts of a checkpoint beyond the text decoder, for
/// [`Qwen35Weights::load_with`]. The default loads the text decoder only.
/// Build it with `LoadParts::default()` and set the fields you need.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[non_exhaustive]
pub struct LoadParts {
    /// The multi-token-prediction head.
    pub mtp_head: MtpHead,
}

/// A `[rows, columns]` matrix in MLX affine quantization: `bits`-wide codes
/// packed in `u32` words and one scale and bias per `group_size` columns.
struct AffineWeight {
    codes: Array,
    scales: Array,
    biases: Array,
    bits: i32,
    group_size: i32,
}

impl AffineWeight {
    /// The dense values in f32: `scale * code + bias` exact, with f32
    /// parameters.
    fn dequantize_f32(&self) -> Result<Array, Qwen35Error> {
        let gpu = StreamOrDevice::gpu();
        Ok(ops::dequantize_device(
            &self.codes,
            self.scales.as_type_device::<f32>(&gpu)?,
            &self.biases.as_type_device::<f32>(&gpu)?,
            self.group_size,
            self.bits,
            &gpu,
        )?)
    }
}

/// Largest single GGUF tensor this loader reads (a 248k x 8192 F32 matrix
/// is 8 GiB).
const MAX_GGUF_TENSOR_BYTES: u64 = 8 << 30;

impl Qwen35Weights {
    /// Loads and validates the text-decoder tensors of a local checkpoint,
    /// in the Hugging Face layout or in mlx-lm's (`language_model.model.*`,
    /// experts split into `switch_mlp`, convolution kernels `[C, K, 1]`, norms
    /// already `1 + weight`).
    ///
    /// Vision-tower and multi-token-prediction tensors are dropped unread.
    /// Hugging Face zero-centered RMS norm weights become `1 + weight` in f32
    /// before any precision conversion, and convolution kernels are stored
    /// `[K, C]`. A module stored with `scales` and `biases` is MLX affine
    /// quantized with the configuration's `quantization` for it; at
    /// [`Qwen35Precision::Checkpoint`] it stays quantized and activations
    /// take its scales' dtype, at [`Qwen35Precision::Float32`] it is
    /// dequantized to f32.
    ///
    /// # Errors
    ///
    /// Returns [`Qwen35Error`] for an unsupported configuration, an unsafe or
    /// missing shard, a missing tensor, a tensor of the wrong shape, or a
    /// quantized module the configuration does not describe.
    pub fn load(
        model_dir: impl AsRef<Path>,
        precision: Qwen35Precision,
    ) -> Result<Self, Qwen35Error> {
        Self::load_with(model_dir, precision, &LoadParts::default())
    }

    /// [`Self::load`], also keeping the parts `parts` asks for: the
    /// multi-token-prediction head when `parts.mtp_head` is [`MtpHead::Load`]. Its norms get the same `1 + weight` shift
    /// as the decoder's, including `pre_fc_norm_embedding` and
    /// `pre_fc_norm_hidden`, whose names do not end in a layer-norm suffix.
    ///
    /// # Errors
    ///
    /// As [`Self::load`], plus [`Qwen35Error::MtpUnsupported`] when
    /// `parts.mtp_head` is [`MtpHead::Load`] and the checkpoint has no one-layer dense head that
    /// shares the decoder's embedding.
    pub fn load_with(
        model_dir: impl AsRef<Path>,
        precision: Qwen35Precision,
        parts: &LoadParts,
    ) -> Result<Self, Qwen35Error> {
        let model_dir = model_dir.as_ref();
        let config = Qwen35Config::parse(&read_file(&model_dir.join("config.json"))?)?;
        let load_mtp = parts.mtp_head == MtpHead::Load;
        if load_mtp {
            check_mtp(&config)?;
        }
        let (mut tensors, numerics) = read_text_tensors(model_dir, &config, load_mtp)?;
        let mut quantized = take_quantized(&config, &mut tensors)?;
        let mut expected = expected_shapes(&config, numerics)?;
        if load_mtp {
            expected.extend(mtp_shapes(&config)?);
        }
        check_shapes(&expected, &tensors, &quantized)?;

        let gpu = StreamOrDevice::gpu();
        let compute = match quantized.get("embed_tokens.weight") {
            Some(embedding) => embedding.scales.dtype(),
            None => tensors["embed_tokens.weight"].dtype(),
        };
        if precision == Qwen35Precision::Float32 {
            check_dequantized_bytes(&quantized)?;
            for (name, weight) in quantized.drain() {
                let dense = weight.dequantize_f32()?;
                dense.eval()?;
                tensors.insert(name, dense);
            }
        }
        for (name, tensor) in &mut tensors {
            let converted = if numerics == Numerics::Transformers && is_zero_centered_norm(name) {
                let shifted = tensor
                    .as_type_device::<f32>(&gpu)?
                    .add_device(Array::from_f32(1.0), &gpu)?;
                match precision {
                    Qwen35Precision::Float32 => shifted,
                    Qwen35Precision::Checkpoint => shifted.as_dtype_device(compute, &gpu)?,
                }
            } else if name.ends_with("conv1d.weight") {
                // [C, 1, K] (Hugging Face) or [C, K, 1] (mlx-lm) to [K, C].
                let transposed = tensor
                    .reshape_device(&[tensor.shape()[0], dim(config.conv_kernel)?], &gpu)?
                    .transpose_device(&gpu)?;
                match precision {
                    Qwen35Precision::Float32 => transposed.as_type_device::<f32>(&gpu)?,
                    Qwen35Precision::Checkpoint => transposed,
                }
            } else {
                match precision {
                    Qwen35Precision::Float32 => tensor.as_type_device::<f32>(&gpu)?,
                    Qwen35Precision::Checkpoint => continue,
                }
            };
            converted.eval()?;
            *tensor = converted;
        }
        let rates: Vec<String> = tensors
            .keys()
            .filter(|name| name.ends_with(".A_log"))
            .cloned()
            .collect();
        for name in rates {
            let Some(log_rate) = tensors.remove(&name) else {
                continue;
            };
            let rate = log_rate.as_type_device::<f32>(&gpu)?.exp_device(&gpu)?;
            rate.eval()?;
            tensors.insert(name.replace(".A_log", ".decay_rate"), rate);
        }
        let compute = match precision {
            Qwen35Precision::Checkpoint => compute,
            Qwen35Precision::Float32 => Dtype::Float32,
        };
        Ok(Self {
            identity: Arc::new(()),
            yarn: yarn_arrays(&config, compute)?,
            config,
            tensors,
            quantized,
            precision,
            compute,
            numerics,
            paged: None,
        })
    }

    /// Loads like [`Self::load`] at [`Qwen35Precision::Checkpoint`], except
    /// that the routed experts of an MLX affine `qwen3_5_moe` checkpoint are
    /// never loaded whole: each `MoE` call reads the experts it routes to
    /// that are not resident, keeping at most `paging.budget_bytes` of them
    /// (least recently used evicted first). This bounds logical cache
    /// membership, not process or GPU footprint. Bit-identical outputs to
    /// [`Self::load`] are the qualification target, not yet established for
    /// every prefill shape.
    ///
    /// # Errors
    ///
    /// As [`Self::load`], plus [`Qwen35Error::Unsupported`] for a dense or
    /// non-affine checkpoint and [`Qwen35Error::PagerBudget`] for a budget
    /// smaller than one layer's experts.
    pub fn load_paged(
        model_dir: impl AsRef<Path>,
        paging: Qwen35Paging,
    ) -> Result<Self, Qwen35Error> {
        let model_dir = model_dir.as_ref();
        let mut weights = Self::load(model_dir, Qwen35Precision::Checkpoint)?;
        let Qwen35Mlp::Experts { experts, .. } = weights.config.mlp else {
            return Err(Qwen35Error::Unsupported("paging a dense checkpoint"));
        };
        if weights.numerics != Numerics::MlxLm {
            return Err(Qwen35Error::Unsupported(
                "paging experts outside mlx-lm's affine layout",
            ));
        }
        let layers = weights.config.layers().len();
        // The loaded expert arrays are still lazy: dropping them here means
        // their payloads are never read.
        weights
            .quantized
            .retain(|name, _| !name.contains(".mlp.switch_mlp."));
        weights.paged = Some(paged::PagedExperts::open(
            model_dir, layers, experts, paging,
        )?);
        Ok(weights)
    }

    /// The expert pager's counters, for a paged load.
    #[must_use]
    pub fn pager_stats(&self) -> Option<Qwen35PagerStats> {
        self.paged.as_ref().map(paged::PagedExperts::stats)
    }

    /// Loads the text decoder of a `qwen35` GGUF file whose tensors are
    /// F32, F16, BF16 or `Q8_0`.
    ///
    /// At [`Qwen35Precision::Checkpoint`] every `Q8_0` matrix becomes MLX
    /// affine 8-bit quantization (group 32) with the file's own scales, so
    /// its values are exactly the file's; activations are BF16. F32 vectors
    /// (norms, convolution kernels, decay rates) stay F32 except the
    /// zero-centered norms, which follow the activations as they do for a
    /// Hugging Face checkpoint. At [`Qwen35Precision::Float32`] every tensor
    /// is expanded to f32.
    ///
    /// The converter's changes are undone or adopted (see [`crate::gguf`]):
    /// norms are already `1 + weight`, the decay rate is `-ssm_a`, the
    /// convolution kernel is transposed to `[K, channels]`, and value heads
    /// are put back into grouped order.
    ///
    /// # Errors
    ///
    /// Returns [`Qwen35Error`] for an unreadable or invalid file, another
    /// architecture, a missing, unexpected or misshapen tensor, or an
    /// encoding this loader does not decode.
    pub fn load_gguf(
        path: impl AsRef<Path>,
        precision: Qwen35Precision,
    ) -> Result<Self, Qwen35Error> {
        let file = GgufFile::open(path)?;
        let config = Qwen35Config::from_gguf(file.metadata(), &file)?;
        let expected = expected_shapes(&config, Numerics::Transformers)?;
        let mut names = HashMap::new();
        let mut unexpected = Vec::new();
        for name in file.names() {
            match canonical_name(name).filter(|canonical| expected.contains_key(canonical)) {
                Some(canonical) => {
                    names.insert(canonical, name.to_owned());
                }
                None => unexpected.push(name.to_owned()),
            }
        }
        if !unexpected.is_empty() {
            unexpected.sort();
            return Err(Qwen35Error::UnexpectedTensors(unexpected));
        }
        let mut order: Vec<&String> = expected.keys().collect();
        order.sort();
        let compute = match precision {
            Qwen35Precision::Checkpoint => Dtype::Bfloat16,
            Qwen35Precision::Float32 => Dtype::Float32,
        };
        let heads = (config.linear_value_heads != config.linear_key_heads)
            .then(|| tiled_positions(config.linear_key_heads, config.linear_value_heads));
        let mut weights = Self {
            identity: Arc::new(()),
            config,
            tensors: HashMap::new(),
            quantized: HashMap::new(),
            precision,
            compute,
            // GGUF metadata declares no `RoPE` scaling; an operator override
            // goes through `with_rope_scaling`.
            yarn: None,
            numerics: Numerics::Transformers,
            paged: None,
        };
        for canonical in order {
            let stored = names
                .get(canonical)
                .ok_or_else(|| Qwen35Error::MissingTensor(canonical.clone()))?;
            let info = file
                .tensor(stored)
                .ok_or_else(|| Qwen35Error::MissingTensor(stored.clone()))?;
            let mut shape = expected[canonical].clone();
            if canonical.ends_with("conv1d.weight") {
                // [C, 1, K] in the Hugging Face checkpoint, squeezed in GGUF.
                shape.remove(1);
            }
            if info
                .shape()
                .iter()
                .map(|&dim| i32::try_from(dim).ok())
                .ne(shape.iter().map(|&dim| Some(dim)))
            {
                return Err(Qwen35Error::TensorShape {
                    name: canonical.clone(),
                    expected: shape,
                    actual: info
                        .shape()
                        .iter()
                        .map(|&dim| i32::try_from(dim).unwrap_or(i32::MAX))
                        .collect(),
                });
            }
            let mut bytes = file.read(stored, MAX_GGUF_TENSOR_BYTES)?;
            if let (Some(order), Some(axis)) = (&heads, value_head_axis(canonical, &weights.config))
            {
                reorder_value_heads(&mut bytes, info, axis, order, canonical)?;
            }
            weights.insert_gguf(canonical, info, &bytes, &shape)?;
        }
        Ok(weights)
    }

    /// Converts one GGUF tensor and stores it under `name`.
    fn insert_gguf(
        &mut self,
        name: &str,
        info: &TensorInfo,
        bytes: &[u8],
        shape: &[i32],
    ) -> Result<(), Qwen35Error> {
        let gpu = StreamOrDevice::gpu();
        let encoding = gguf_encoding(name, info)?;
        if encoding == GgufEncoding::Q8_0
            && shape.len() == 2
            && self.precision == Qwen35Precision::Checkpoint
        {
            let repack = repack_affine(encoding, bytes)?;
            let groups = shape[1]
                / i32::try_from(repack.layout.group_size)
                    .map_err(|_| Qwen35Error::ShapeOverflow)?;
            let parameters = match repack.layout.parameters {
                AffineParameters::F16 => Dtype::Float16,
                AffineParameters::F32 => Dtype::Float32,
            };
            let words = shape[1] * i32::from(repack.layout.bits) / 32;
            let weight = AffineWeight {
                codes: Array::from_slice(&repack.codes, &[shape[0], words]),
                scales: Array::from_slice(&repack.scales, &[shape[0], groups])
                    .as_dtype_device(parameters, &gpu)?,
                biases: Array::from_slice(&repack.biases, &[shape[0], groups])
                    .as_dtype_device(parameters, &gpu)?,
                bits: i32::from(repack.layout.bits),
                group_size: i32::try_from(repack.layout.group_size)
                    .map_err(|_| Qwen35Error::ShapeOverflow)?,
            };
            mlx_rs::transforms::eval([&weight.codes, &weight.scales, &weight.biases])?;
            self.quantized.insert(name.to_owned(), weight);
            return Ok(());
        }
        let mut tensor = match encoding {
            GgufEncoding::F16 | GgufEncoding::BF16
                if self.precision == Qwen35Precision::Checkpoint =>
            {
                let halves: Vec<u16> = bytes
                    .chunks_exact(2)
                    .map(|half| u16::from_le_bytes([half[0], half[1]]))
                    .collect();
                let dtype = if encoding == GgufEncoding::F16 {
                    Dtype::Float16
                } else {
                    Dtype::Bfloat16
                };
                Array::from_slice(&halves, shape).view_dtype_device(dtype, &gpu)?
            }
            GgufEncoding::F32 | GgufEncoding::F16 | GgufEncoding::BF16 | GgufEncoding::Q8_0 => {
                Array::from_slice(&decode(encoding, bytes)?, shape)
            }
            other => {
                return Err(Qwen35Error::Encoding {
                    tensor: name.to_owned(),
                    encoding: other.into(),
                });
            }
        };
        if name.ends_with("conv1d.weight") {
            tensor = tensor.transpose_device(&gpu)?;
        }
        let (name, tensor) = if let Some(layer) = name.strip_suffix(".A_log") {
            // The file stores -exp(A_log).
            (
                format!("{layer}.decay_rate"),
                tensor.as_type_device::<f32>(&gpu)?.negative_device(&gpu)?,
            )
        } else if is_zero_centered_norm(name) || self.precision == Qwen35Precision::Float32 {
            (name.to_owned(), tensor.as_dtype_device(self.compute, &gpu)?)
        } else {
            (name.to_owned(), tensor)
        };
        tensor.eval()?;
        self.tensors.insert(name, tensor);
        Ok(())
    }

    /// The validated decoder configuration.
    #[must_use]
    pub const fn config(&self) -> &Qwen35Config {
        &self.config
    }

    /// Applies an operator's `RoPE` scaling request (`--rope-scaling yarn:4`),
    /// which then counts as declared: [`Qwen35Config::context_limit`] becomes
    /// the qualified pair's context. Snapshots taken before the change are
    /// refused afterwards, since their keys were rotated differently.
    ///
    /// # Errors
    ///
    /// Returns [`Qwen35Error::Config`] for an unqualified pair or one that
    /// conflicts with the declared scaling, or [`Qwen35Error::Mlx`] if the
    /// rotary arrays cannot be built.
    pub fn with_rope_scaling(mut self, request: RopeScalingRequest) -> Result<Self, Qwen35Error> {
        self.config = self.config.with_rope_scaling(request)?;
        self.yarn = yarn_arrays(&self.config, self.compute)?;
        self.identity = Arc::new(());
        Ok(self)
    }

    /// The precision these weights were loaded at.
    #[must_use]
    pub const fn precision(&self) -> Qwen35Precision {
        self.precision
    }

    /// Logical bytes of the loaded decoder tensors.
    #[must_use]
    pub fn logical_weight_bytes(&self) -> usize {
        self.tensors.values().map(Array::nbytes).sum::<usize>()
            + self
                .quantized
                .values()
                .map(|weight| {
                    weight.codes.nbytes() + weight.scales.nbytes() + weight.biases.nbytes()
                })
                .sum::<usize>()
    }

    /// Starts an empty sequence borrowing these weights.
    #[must_use]
    pub fn executor(&self) -> Qwen35Executor<'_> {
        Qwen35Executor {
            weights: self,
            layers: self.config.layers.iter().map(|_| None).collect(),
            tokens: 0,
            pending: None,
        }
    }

    /// Starts a sequence from `snapshot`, as if its tokens had just been
    /// consumed. The snapshot shares its arrays; neither side copies data.
    ///
    /// # Errors
    ///
    /// Returns [`Qwen35Error::SnapshotMismatch`] for another weight load, or
    /// [`Qwen35Error::StateKind`] when its layers do not match the configuration.
    pub fn executor_from(
        &self,
        snapshot: &Qwen35Snapshot,
    ) -> Result<Qwen35Executor<'_>, Qwen35Error> {
        if !Arc::ptr_eq(&self.identity, &snapshot.identity) {
            return Err(Qwen35Error::SnapshotMismatch);
        }
        let fits = snapshot.layers.len() == self.config.layers.len()
            && snapshot
                .layers
                .iter()
                .zip(&self.config.layers)
                .all(|(state, kind)| {
                    matches!(
                        (state, kind),
                        (None, _)
                            | (
                                Some(LayerState::Linear { .. }),
                                Qwen35LayerKind::LinearAttention
                            )
                            | (
                                Some(LayerState::Full { .. }),
                                Qwen35LayerKind::FullAttention
                            )
                    )
                });
        if !fits {
            return Err(Qwen35Error::StateKind);
        }
        Ok(Qwen35Executor {
            weights: self,
            layers: snapshot.layers.clone(),
            tokens: snapshot.tokens,
            pending: None,
        })
    }

    fn tensor(&self, name: &str) -> Result<&Array, Qwen35Error> {
        self.tensors
            .get(name)
            .ok_or_else(|| Qwen35Error::MissingTensor(name.to_owned()))
    }

    /// The matrix (or stack of matrices) `name`, dense or quantized.
    fn matrix(&self, name: &str) -> Result<Matrix<'_>, Qwen35Error> {
        match self.quantized.get(name) {
            Some(weight) => Ok(Matrix::Affine(weight)),
            None => Ok(Matrix::Dense(self.tensor(name)?)),
        }
    }

    /// `input` times the transpose of the matrix `name`, in `input`'s dtype.
    fn project(&self, input: &Array, name: &str) -> Result<Array, Qwen35Error> {
        let gpu = StreamOrDevice::gpu();
        let output = self.matrix(name)?.apply(input)?;
        Ok(if output.dtype() == input.dtype() {
            output
        } else {
            output.as_dtype_device(input.dtype(), &gpu)?
        })
    }

    /// The embedding rows of `ids` as `[1, seq, hidden]` in the activation
    /// dtype.
    fn embed(&self, ids: &[i32]) -> Result<Array, Qwen35Error> {
        let gpu = StreamOrDevice::gpu();
        let seq = dim(ids.len())?;
        let rows = Array::from_slice(ids, &[seq]);
        let name = "embed_tokens.weight";
        let embedded = match self.quantized.get(name) {
            Some(weight) => {
                let take = |array: &Array| array.take_axis_device(&rows, 0, &gpu);
                // f32 parameters keep `scale * code + bias` exact before the
                // cast to the activation dtype.
                ops::dequantize_device(
                    take(&weight.codes)?,
                    take(&weight.scales)?.as_type_device::<f32>(&gpu)?,
                    &take(&weight.biases)?.as_type_device::<f32>(&gpu)?,
                    weight.group_size,
                    weight.bits,
                    &gpu,
                )?
            }
            None => self.tensor(name)?.take_axis_device(&rows, 0, &gpu)?,
        };
        Ok(embedded
            .as_dtype_device(self.compute, &gpu)?
            .reshape_device(&[1, seq, dim(self.config.hidden_size)?], &gpu)?)
    }
}

/// A GGUF file's tensor encoding; the GGUF loader decodes nothing else.
fn gguf_encoding(name: &str, info: &TensorInfo) -> Result<GgufEncoding, Qwen35Error> {
    info.encoding().gguf().ok_or_else(|| Qwen35Error::Encoding {
        tensor: name.to_owned(),
        encoding: info.encoding(),
    })
}

/// Puts a GGUF tensor's value heads back in grouped order, in place on its
/// stored bytes. Rows and whole column blocks move intact, so this is exact
/// for every encoding.
fn reorder_value_heads(
    bytes: &mut [u8],
    info: &TensorInfo,
    axis: ValueHeadAxis,
    order: &[usize],
    name: &str,
) -> Result<(), Qwen35Error> {
    let encoding = gguf_encoding(name, info)?;
    let shape = info.shape();
    let columns = if shape.len() == 2 { shape[1] } else { 1 };
    let size = |elements: u64| -> Result<usize, Qwen35Error> {
        encoding
            .byte_len(elements)
            .and_then(|bytes| usize::try_from(bytes).ok())
            .ok_or_else(|| Qwen35Error::Encoding {
                tensor: name.to_owned(),
                encoding: encoding.into(),
            })
    };
    let row_bytes = size(columns)?;
    let to_u64 = |value: usize| u64::try_from(value).map_err(|_| Qwen35Error::ShapeOverflow);
    match axis {
        ValueHeadAxis::Rows { skip, width } => {
            let total = bytes.len();
            permute_spans(bytes, 1, total, skip * row_bytes, width * row_bytes, order);
        }
        ValueHeadAxis::Columns { width } => {
            permute_spans(
                bytes,
                bytes.len() / row_bytes,
                row_bytes,
                0,
                size(to_u64(width)?)?,
                order,
            );
        }
    }
    Ok(())
}

/// Per-layer sequence state. Snapshot K/V is copied to its valid rows;
/// recurrent and convolution arrays are shared immutable values.
#[derive(Clone)]
enum LayerState {
    /// The last `K - 1` convolution inputs `[1, K - 1, conv_dim]` and the f32
    /// recurrent matrices `[Hv, Dk, Dv]`.
    Linear { conv: Array, recurrent: Array },
    /// Rotated keys and values `[1, kv_heads, capacity, head_dim]`, valid
    /// for the executor's first `tokens` positions. Capacity grows in
    /// [`Qwen35Config::kv_storage_tokens`] tiers so a decode step writes one
    /// row in place
    /// instead of copying the cache.
    Full { keys: Array, values: Array },
}

/// A sequence's state after its first [`Self::tokens`] tokens: every linear
/// layer's recurrent and convolution state and every full-attention layer's
/// K/V. Recurrent state cannot be cut back to an earlier position, so a prefix
/// can be reused only from a snapshot taken at exactly that position.
#[derive(Clone)]
pub struct Qwen35Snapshot {
    identity: Arc<()>,
    layers: Vec<Option<LayerState>>,
    tokens: usize,
}

impl Qwen35Snapshot {
    /// Tokens the snapshot holds.
    #[must_use]
    pub const fn tokens(&self) -> usize {
        self.tokens
    }

    /// Bytes of the arrays the snapshot keeps alive.
    #[must_use]
    pub fn state_bytes(&self) -> usize {
        self.layers
            .iter()
            .flatten()
            .map(|layer| match layer {
                LayerState::Linear { conv, recurrent } => conv.nbytes() + recurrent.nbytes(),
                LayerState::Full { keys, values } => keys.nbytes() + values.nbytes(),
            })
            .sum()
    }
}

/// One sequence's decoder state over borrowed weights.
pub struct Qwen35Executor<'a> {
    weights: &'a Qwen35Weights,
    layers: Vec<Option<LayerState>>,
    tokens: usize,
    /// What the last verify kept to roll back, until it is truncated or
    /// another extend commits it.
    pending: Option<VerifyStash>,
}

/// What a verify keeps so [`Qwen35Executor::truncate_to`] can roll back to
/// any of its prefixes. Recurrent state cannot drop positions, so each linear
/// layer keeps its state from before the verify (shared arrays, no copy) and
/// the verified rows' convolution, decay and beta inputs, which a rollback
/// replays token by token over the kept prefix. Full-attention layers need
/// nothing: their valid length is the token count.
struct VerifyStash {
    start: usize,
    verified: usize,
    layers: Vec<Option<LinearStash>>,
}

/// Hidden rows a forward hands back next to its tokens, for drafters that
/// read the target's activations. Each choice yields `[rows, width]` in the
/// compute dtype, one row per consumed position.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Qwen35Taps<'a> {
    /// No rows.
    None,
    /// The hidden state after the final norm, the input of the output head
    /// and of the multi-token-prediction head; `width` is the hidden size.
    FinalNormed,
    /// The residual after each listed layer (before the final norm), in list
    /// order and concatenated; `width` is the hidden size times the count.
    Layers(&'a [usize]),
}

/// A verify's per-row result with the taps it was asked for.
#[derive(Debug)]
pub struct Qwen35Verified<T> {
    /// One entry per verified token, as from the method that returned this.
    pub rows: T,
    /// `[tokens, width]`, evaluated; `None` for [`Qwen35Taps::None`].
    pub taps: Option<Array>,
}

/// One linear layer's part of a [`VerifyStash`].
struct LinearStash {
    conv: Array,
    recurrent: Array,
    /// `in_proj_qkv`, `in_proj_a` and `in_proj_b` outputs, `[1, seq, _]`.
    qkv: Array,
    a: Array,
    b: Array,
}

impl Qwen35Executor<'_> {
    /// Tokens consumed so far.
    #[must_use]
    pub const fn tokens(&self) -> usize {
        self.tokens
    }

    /// The state after every token consumed so far, evaluated so it holds
    /// no pending graph. Full-attention K/V is copied to the valid rows;
    /// recurrent and convolution state retain their exact boundary values.
    ///
    /// # Errors
    ///
    /// Returns [`Qwen35Error::Mlx`] if evaluating the state fails.
    pub fn snapshot(&self) -> Result<Qwen35Snapshot, Qwen35Error> {
        mlx_rs::transforms::eval(self.state_arrays())?;
        let gpu = StreamOrDevice::gpu();
        let rows = dim(self.tokens)?;
        let positions = Array::arange_device::<i32, i32>(0, rows, None, &gpu)?;
        let layers = self
            .layers
            .iter()
            .map(|layer| {
                match layer {
                    Some(LayerState::Full { keys, values }) => {
                        // A gather materializes only valid rows, so a snapshot cannot
                        // pin unused capacity after stepped K/V growth is integrated.
                        let keys = keys.take_axis_device(&positions, 2, &gpu)?;
                        let values = values.take_axis_device(&positions, 2, &gpu)?;
                        mlx_rs::transforms::eval([&keys, &values])?;
                        Ok(Some(LayerState::Full { keys, values }))
                    }
                    state => Ok(state.clone()),
                }
            })
            .collect::<Result<Vec<_>, Qwen35Error>>()?;
        Ok(Qwen35Snapshot {
            identity: Arc::clone(&self.weights.identity),
            layers,
            tokens: self.tokens,
        })
    }

    fn state_arrays(&self) -> Vec<&Array> {
        self.layers
            .iter()
            .flatten()
            .flat_map(|layer| match layer {
                LayerState::Linear { conv, recurrent } => [conv, recurrent],
                LayerState::Full { keys, values } => [keys, values],
            })
            .collect()
    }

    /// Clears all sequence state.
    pub fn reset(&mut self) {
        self.layers.iter_mut().for_each(|layer| *layer = None);
        self.tokens = 0;
        self.pending = None;
    }

    /// The most tokens one [`Self::verify_logits`] or [`Self::verify_greedy`]
    /// accepts: what the rollback stash budget holds, at most one prefill
    /// graph.
    #[must_use]
    pub fn max_verify_tokens(&self) -> usize {
        let config = &self.weights.config;
        let linear = config
            .layers
            .iter()
            .filter(|kind| **kind == Qwen35LayerKind::LinearAttention)
            .count();
        let element = if self.weights.compute == Dtype::Float32 {
            4
        } else {
            2
        };
        let per_token = linear * (config.conv_dim() + 2 * config.linear_value_heads) * element;
        (VERIFY_STASH_BYTES / per_token.max(1)).min(PREFILL_CHUNK_TOKENS)
    }

    /// Appends `tokens` and returns the f32 logits after every appended
    /// position, row-major `[tokens.len(), vocab]`: row `i` is the
    /// distribution after `tokens[..=i]`. Keeps what [`Self::truncate_to`]
    /// needs to roll back to any prefix of `tokens`; an earlier verify that was
    /// not truncated is committed. `taps` selects hidden rows to return
    /// beside the logits, one per token.
    ///
    /// # Errors
    ///
    /// Returns [`Qwen35Error::VerifyLength`] for more than
    /// [`Self::max_verify_tokens`] tokens, [`Qwen35Error::TapLayer`] for a
    /// bad tap list, or any error of [`Self::extend_last_logits`].
    pub fn verify_logits(
        &mut self,
        tokens: &[i32],
        taps: Qwen35Taps<'_>,
    ) -> Result<Qwen35Verified<Vec<f32>>, Qwen35Error> {
        let (hidden, taps) = self.verify_hidden(tokens, taps)?;
        let logits = self.row_logits(&hidden)?;
        mlx_rs::transforms::eval(std::iter::once(&logits).chain(&taps))?;
        Ok(Qwen35Verified {
            rows: logits.as_slice::<f32>().to_vec(),
            taps,
        })
    }

    /// [`Self::verify_logits`], returning the greedy token after every
    /// appended position (lowest ID on ties), picked on the GPU so only the
    /// IDs cross to the host.
    ///
    /// # Errors
    ///
    /// As [`Self::verify_logits`], plus [`Qwen35Error::NonFiniteLogits`] when
    /// any logit is NaN or infinite.
    pub fn verify_greedy(
        &mut self,
        tokens: &[i32],
        taps: Qwen35Taps<'_>,
    ) -> Result<Qwen35Verified<Vec<i32>>, Qwen35Error> {
        let (hidden, taps) = self.verify_hidden(tokens, taps)?;
        let logits = self.row_logits(&hidden)?;
        let gpu = StreamOrDevice::gpu();
        let finite = logits.is_finite_device(&gpu)?.all_device(false, &gpu)?;
        let picks = ops::indexing::argmax_axis_device(&logits, -1, false, &gpu)?;
        mlx_rs::transforms::eval([&picks, &finite].into_iter().chain(&taps))?;
        if !finite.item::<bool>() {
            return Err(Qwen35Error::NonFiniteLogits);
        }
        // MLX returns argmax indices as uint32.
        let rows = picks
            .as_slice::<u32>()
            .iter()
            .map(|&id| i32::try_from(id).map_err(|_| Qwen35Error::ShapeOverflow))
            .collect::<Result<_, _>>()?;
        Ok(Qwen35Verified { rows, taps })
    }

    /// Keeps the first `tokens` positions of the sequence, which must lie
    /// between the length before the last verify and its end, and drops the
    /// rest of that verify. Linear layers restore their state from before the
    /// verify and replay the kept rows token by token (the decode path's
    /// arithmetic); full-attention layers keep their first `tokens` rows.
    ///
    /// # Errors
    ///
    /// Returns [`Qwen35Error::NoPendingVerify`] when no verify is pending,
    /// [`Qwen35Error::TruncateRange`] for a length outside it, or
    /// [`Qwen35Error::Mlx`] if the replay fails.
    pub fn truncate_to(&mut self, tokens: usize) -> Result<(), Qwen35Error> {
        let pending = self.pending.take().ok_or(Qwen35Error::NoPendingVerify)?;
        let maximum = pending.start + pending.verified;
        if !(pending.start..=maximum).contains(&tokens) {
            let minimum = pending.start;
            self.pending = Some(pending);
            return Err(Qwen35Error::TruncateRange {
                requested: tokens,
                minimum,
                maximum,
            });
        }
        if tokens == maximum {
            return Ok(());
        }
        let keep = dim(tokens - pending.start)?;
        let weights = self.weights;
        for (index, stash) in pending.layers.into_iter().enumerate() {
            let Some(stash) = stash else { continue };
            let state = if keep == 0 {
                LayerState::Linear {
                    conv: stash.conv,
                    recurrent: stash.recurrent,
                }
            } else {
                let rows = |array: &Array| array.index((.., 0..keep, ..));
                let (_, conv, recurrent) = linear_recurrence(
                    weights,
                    &format!("layers.{index}.linear_attn"),
                    &stash.conv,
                    stash.recurrent,
                    &LinearInputs {
                        qkv: rows(&stash.qkv),
                        a: rows(&stash.a),
                        b: rows(&stash.b),
                    },
                    true,
                )?;
                LayerState::Linear { conv, recurrent }
            };
            self.layers[index] = Some(state);
        }
        self.tokens = tokens;
        mlx_rs::transforms::eval(self.state_arrays())?;
        Ok(())
    }

    fn verify_hidden(
        &mut self,
        tokens: &[i32],
        taps: Qwen35Taps<'_>,
    ) -> Result<(Array, Option<Array>), Qwen35Error> {
        self.check_input(tokens)?;
        self.check_taps(taps)?;
        let maximum = self.max_verify_tokens();
        if tokens.len() > maximum {
            return Err(Qwen35Error::VerifyLength {
                requested: tokens.len(),
                maximum,
            });
        }
        self.pending = Some(VerifyStash {
            start: self.tokens,
            verified: tokens.len(),
            layers: self.layers.iter().map(|_| None).collect(),
        });
        self.forward_chunk_tapped(tokens, taps)
    }

    /// Consumes a prompt into an empty sequence and returns the last
    /// position's logits.
    ///
    /// # Errors
    ///
    /// Returns [`Qwen35Error::NotEmpty`] if the sequence already holds tokens,
    /// or any error of [`Self::extend_last_logits`].
    pub fn prefill_last_logits(&mut self, input_ids: &[i32]) -> Result<Vec<f32>, Qwen35Error> {
        if self.tokens != 0 {
            return Err(Qwen35Error::NotEmpty(self.tokens));
        }
        self.extend_last_logits(input_ids)
    }

    /// Consumes one token after the cached sequence and returns its logits.
    ///
    /// # Errors
    ///
    /// As [`Self::extend_last_logits`].
    pub fn decode_last_logits(&mut self, input_id: i32) -> Result<Vec<f32>, Qwen35Error> {
        self.extend_last_logits(&[input_id])
    }

    /// Appends tokens to the sequence and returns the last position's logits.
    /// Inputs longer than [`PREFILL_CHUNK_TOKENS`] are evaluated in chunks.
    ///
    /// On error the sequence state is unspecified; call [`Self::reset`].
    ///
    /// # Errors
    ///
    /// Returns [`Qwen35Error`] for empty input, an out-of-vocabulary ID, a
    /// context over the limit, or an MLX failure.
    pub fn extend_last_logits(&mut self, input_ids: &[i32]) -> Result<Vec<f32>, Qwen35Error> {
        self.extend_last_logits_tapped(input_ids, Qwen35Taps::None, |_, _| Ok(()))
    }

    /// [`Self::extend_last_logits`], handing each evaluation chunk's `taps`
    /// rows to `sink` with the chunk's first position, in order. The rows are
    /// evaluated and dropped after `sink` returns, so a sink that keeps only
    /// what it needs holds at most one chunk of them.
    ///
    /// # Errors
    ///
    /// As [`Self::extend_last_logits`], plus [`Qwen35Error::TapLayer`] for a
    /// bad tap list and any error `sink` returns.
    pub fn extend_last_logits_tapped(
        &mut self,
        input_ids: &[i32],
        taps: Qwen35Taps<'_>,
        mut sink: impl FnMut(&Array, usize) -> Result<(), Qwen35Error>,
    ) -> Result<Vec<f32>, Qwen35Error> {
        let hidden = self.extend_hidden(input_ids, taps, &mut sink)?;
        let logits = self.last_logits(&hidden)?;
        logits.eval()?;
        Ok(logits.as_slice::<f32>().to_vec())
    }

    /// Consumes one token and returns the greedy next token, picked on the
    /// GPU so only one ID crosses to the host instead of the logits row.
    /// Ties go to the lowest ID, as in a host argmax that keeps the first
    /// maximum.
    ///
    /// # Errors
    ///
    /// As [`Self::extend_last_logits`], plus [`Qwen35Error::NonFiniteLogits`]
    /// when any logit is NaN or infinite.
    pub fn decode_greedy(&mut self, input_id: i32) -> Result<i32, Qwen35Error> {
        self.extend_greedy(&[input_id])
    }

    /// [`Self::extend_last_logits`] followed by a GPU greedy pick of the last
    /// position, as in [`Self::decode_greedy`].
    ///
    /// # Errors
    ///
    /// As [`Self::decode_greedy`].
    pub fn extend_greedy(&mut self, input_ids: &[i32]) -> Result<i32, Qwen35Error> {
        let hidden = self.extend_hidden(input_ids, Qwen35Taps::None, &mut |_, _| Ok(()))?;
        let logits = self.last_logits(&hidden)?;
        let gpu = StreamOrDevice::gpu();
        let finite = logits.is_finite_device(&gpu)?.all_device(false, &gpu)?;
        let token = ops::indexing::argmax_axis_device(&logits, -1, false, &gpu)?;
        mlx_rs::transforms::eval([&token, &finite])?;
        if !finite.item::<bool>() {
            return Err(Qwen35Error::NonFiniteLogits);
        }
        // MLX returns argmax indices as uint32.
        i32::try_from(token.item::<u32>()).map_err(|_| Qwen35Error::ShapeOverflow)
    }

    /// Runs `input_ids` through the decoder, updating the sequence state, and
    /// returns the final chunk's hidden states. Each chunk's `taps` rows are
    /// evaluated and passed to `sink` with the chunk's first position.
    fn extend_hidden(
        &mut self,
        input_ids: &[i32],
        taps: Qwen35Taps<'_>,
        sink: &mut dyn FnMut(&Array, usize) -> Result<(), Qwen35Error>,
    ) -> Result<Array, Qwen35Error> {
        self.check_input(input_ids)?;
        self.check_taps(taps)?;
        // An ordinary extend commits any verify; it can no longer roll back.
        self.pending = None;
        let mut chunks = input_ids.chunks(PREFILL_CHUNK_TOKENS).peekable();
        while let Some(chunk) = chunks.next() {
            let first = self.tokens;
            let (hidden, rows) = self.forward_chunk_tapped(chunk, taps)?;
            let last = chunks.peek().is_none();
            if let Some(rows) = rows {
                // Evaluating the state with the rows keeps the rows from
                // holding the chunk's graph after the sink returns.
                let mut arrays = self.state_arrays();
                arrays.push(&rows);
                mlx_rs::transforms::eval(arrays)?;
                sink(&rows, first)?;
            } else if !last {
                // Evaluate the carried state so the next chunk's graph starts
                // from materialized arrays instead of growing without bound.
                mlx_rs::transforms::eval(self.state_arrays())?;
            }
            if last {
                return Ok(hidden);
            }
        }
        unreachable!("input_ids is nonempty")
    }

    /// Refuses a [`Qwen35Taps::Layers`] list that is empty or names a layer
    /// the model lacks.
    fn check_taps(&self, taps: Qwen35Taps<'_>) -> Result<(), Qwen35Error> {
        let layers = self.weights.config.layers.len();
        if let Qwen35Taps::Layers(list) = taps {
            if list.is_empty() {
                return Err(Qwen35Error::TapLayer {
                    layer: None,
                    layers,
                });
            }
            if let Some(&layer) = list.iter().find(|&&layer| layer >= layers) {
                return Err(Qwen35Error::TapLayer {
                    layer: Some(layer),
                    layers,
                });
            }
        }
        Ok(())
    }

    /// [`Self::forward_chunk`] plus the chunk's `taps` rows `[seq, width]`,
    /// not yet evaluated. The tap list must have passed [`Self::check_taps`].
    fn forward_chunk_tapped(
        &mut self,
        input_ids: &[i32],
        taps: Qwen35Taps<'_>,
    ) -> Result<(Array, Option<Array>), Qwen35Error> {
        let gpu = StreamOrDevice::gpu();
        let rows = |array: &Array| -> Result<Array, Qwen35Error> {
            let shape = array.shape();
            Ok(array.reshape_device(&[shape[1], shape[2]], &gpu)?)
        };
        match taps {
            Qwen35Taps::None => Ok((self.forward_chunk(input_ids)?, None)),
            Qwen35Taps::FinalNormed => {
                let hidden = self.forward_chunk(input_ids)?;
                let normed = rows(&self.final_norm(&hidden)?)?;
                Ok((hidden, Some(normed)))
            }
            Qwen35Taps::Layers(list) => {
                let names: Vec<String> = list
                    .iter()
                    .map(|layer| format!("layers.{layer}.post_ffn"))
                    .collect();
                let mut found: Vec<Option<Array>> = vec![None; list.len()];
                let hidden = self.forward_chunk_observed(input_ids, &mut |stage, tensor| {
                    for (slot, name) in found.iter_mut().zip(&names) {
                        if stage == name {
                            *slot = Some(tensor.clone());
                        }
                    }
                    Ok(())
                })?;
                let found: Vec<Array> = found
                    .into_iter()
                    .map(|tap| tap.expect("check_taps admits only layers the forward visits"))
                    .collect();
                let joined = ops::concatenate_axis_device(&found, 2, &gpu)?;
                Ok((hidden, Some(rows(&joined)?)))
            }
        }
    }

    /// Refuses empty input, a context over the limit, and an ID outside the
    /// vocabulary.
    fn check_input(&self, input_ids: &[i32]) -> Result<(), Qwen35Error> {
        let config = &self.weights.config;
        if input_ids.is_empty() {
            return Err(Qwen35Error::EmptyInput);
        }
        let maximum = config.context_limit();
        let requested = self
            .tokens
            .checked_add(input_ids.len())
            .ok_or(Qwen35Error::ShapeOverflow)?;
        if requested > maximum {
            return Err(Qwen35Error::ContextLimit { requested, maximum });
        }
        for &token in input_ids {
            if usize::try_from(token).map_or(true, |id| id >= config.vocab_size) {
                return Err(Qwen35Error::InvalidTokenId {
                    token,
                    vocab_size: config.vocab_size,
                });
            }
        }
        Ok(())
    }

    fn forward_chunk(&mut self, input_ids: &[i32]) -> Result<Array, Qwen35Error> {
        self.forward_chunk_observed(input_ids, &mut |_, _| Ok(()))
    }

    /// Internal diagnostic seam; the normal forward supplies a no-op observer.
    fn forward_chunk_observed(
        &mut self,
        input_ids: &[i32],
        observe: &mut impl FnMut(&str, &Array) -> Result<(), Qwen35Error>,
    ) -> Result<Array, Qwen35Error> {
        let weights = self.weights;
        let config = &weights.config;
        let gpu = StreamOrDevice::gpu();
        let offset = dim(self.tokens)?;
        let mut hidden = weights.embed(input_ids)?;
        observe("embedding", &hidden)?;
        for (index, kind) in config.layers.iter().enumerate() {
            let base = format!("layers.{index}");
            let normed = rms_norm(
                &hidden,
                weights.tensor(&format!("{base}.input_layernorm.weight"))?,
                config.rms_norm_eps,
            )?;
            let state = &mut self.layers[index];
            let record = self
                .pending
                .as_mut()
                .map(|pending| &mut pending.layers[index]);
            let mixed = match kind {
                Qwen35LayerKind::LinearAttention => gated_delta_net(
                    weights,
                    &format!("{base}.linear_attn"),
                    &normed,
                    state,
                    record,
                )?,
                Qwen35LayerKind::FullAttention => gated_attention(
                    weights,
                    &format!("{base}.self_attn"),
                    &normed,
                    offset,
                    state,
                )?,
            };
            hidden = hidden.add_device(&mixed, &gpu)?;
            observe(&format!("{base}.post_attention"), &hidden)?;
            let normed = rms_norm(
                &hidden,
                weights.tensor(&format!("{base}.post_attention_layernorm.weight"))?,
                config.rms_norm_eps,
            )?;
            hidden = hidden.add_device(mlp(weights, &base, &normed)?, &gpu)?;
            observe(&format!("{base}.post_ffn"), &hidden)?;
        }
        self.tokens += input_ids.len();
        Ok(hidden)
    }

    /// `hidden` after the final norm, not yet evaluated.
    fn final_norm(&self, hidden: &Array) -> Result<Array, Qwen35Error> {
        let weights = self.weights;
        rms_norm(
            hidden,
            weights.tensor("norm.weight")?,
            weights.config.rms_norm_eps,
        )
    }

    /// Every position's f32 logits `[seq, vocab]`, not yet evaluated.
    fn row_logits(&self, hidden: &Array) -> Result<Array, Qwen35Error> {
        let weights = self.weights;
        let config = &weights.config;
        let gpu = StreamOrDevice::gpu();
        let seq = hidden.shape()[1];
        let normed = self.final_norm(hidden)?;
        let head = if config.tie_word_embeddings {
            "embed_tokens.weight"
        } else {
            "lm_head.weight"
        };
        Ok(weights
            .project(&normed, head)?
            .reshape_device(&[seq, dim(config.vocab_size)?], &gpu)?
            .as_type_device::<f32>(&gpu)?)
    }

    /// The last position's f32 logits `[vocab]`, not yet evaluated.
    fn last_logits(&self, hidden: &Array) -> Result<Array, Qwen35Error> {
        let weights = self.weights;
        let config = &weights.config;
        let gpu = StreamOrDevice::gpu();
        let last = hidden.shape()[1] - 1;
        let last = hidden.index((.., last..last + 1, ..));
        let normed = rms_norm(&last, weights.tensor("norm.weight")?, config.rms_norm_eps)?;
        let head = if config.tie_word_embeddings {
            "embed_tokens.weight"
        } else {
            "lm_head.weight"
        };
        Ok(weights
            .project(&normed, head)?
            .reshape_device(&[dim(config.vocab_size)?], &gpu)?
            .as_type_device::<f32>(&gpu)?)
    }
}

fn gated_delta_net(
    weights: &Qwen35Weights,
    base: &str,
    input: &Array,
    state: &mut Option<LayerState>,
    record: Option<&mut Option<LinearStash>>,
) -> Result<Array, Qwen35Error> {
    let config = &weights.config;
    let gpu = StreamOrDevice::gpu();
    let seq = input.shape()[1];
    let value_heads = dim(config.linear_value_heads)?;
    let value_dim = dim(config.linear_value_head_dim)?;
    let compute = input.dtype();
    let project = |name: &str| weights.project(input, &format!("{base}.{name}.weight"));

    let (conv, recurrent) = match state.take() {
        Some(LayerState::Linear { conv, recurrent }) => (conv, recurrent),
        Some(LayerState::Full { .. }) => return Err(Qwen35Error::StateKind),
        None => (
            ops::zeros_dtype_device(
                &[1, dim(config.conv_kernel - 1)?, dim(config.conv_dim())?],
                compute,
                &gpu,
            )?,
            ops::zeros_dtype_device(
                &[value_heads, dim(config.linear_key_head_dim)?, value_dim],
                Dtype::Float32,
                &gpu,
            )?,
        ),
    };
    let inputs = LinearInputs {
        qkv: project("in_proj_qkv")?,
        a: project("in_proj_a")?,
        b: project("in_proj_b")?,
    };
    if let Some(slot) = record {
        // Clones share the arrays; the stash keeps them alive for a rollback.
        *slot = Some(LinearStash {
            conv: conv.clone(),
            recurrent: recurrent.clone(),
            qkv: inputs.qkv.clone(),
            a: inputs.a.clone(),
            b: inputs.b.clone(),
        });
    }
    let (output, next_conv, recurrent) =
        linear_recurrence(weights, base, &conv, recurrent, &inputs, false)?;
    let normed = fast::rms_norm_device(
        &output,
        weights
            .tensor(&format!("{base}.norm.weight"))?
            .as_type_device::<f32>(&gpu)?,
        config.rms_norm_eps,
        &gpu,
    )?;
    let output_gate = project("in_proj_z")?
        .reshape_device(&[seq, value_heads, value_dim], &gpu)?
        .as_type_device::<f32>(&gpu)?;
    let gated = normed
        .multiply_device(silu(&output_gate)?, &gpu)?
        .as_dtype_device(compute, &gpu)?
        .reshape_device(&[1, seq, value_heads * value_dim], &gpu)?;
    *state = Some(LayerState::Linear {
        conv: next_conv,
        recurrent,
    });
    weights.project(&gated, &format!("{base}.out_proj.weight"))
}

/// A linear layer's input projections `[1, seq, _]`: everything its state
/// update reads, so a rollback can replay the update without the layer input.
struct LinearInputs {
    qkv: Array,
    a: Array,
    b: Array,
}

/// The convolution and delta rule of one linear layer from its projections:
/// returns the per-head output `[seq, Hv, Dv]` before the gated norm, the
/// next convolution window, and the next recurrent state. `by_token` forces
/// the decode path's token-by-token recurrence for any length; otherwise a
/// multi-token input uses the chunked form.
fn linear_recurrence(
    weights: &Qwen35Weights,
    base: &str,
    window: &Array,
    recurrent: Array,
    inputs: &LinearInputs,
    by_token: bool,
) -> Result<(Array, Array, Array), Qwen35Error> {
    let config = &weights.config;
    let gpu = StreamOrDevice::gpu();
    let seq = inputs.qkv.shape()[1];
    let key_heads = dim(config.linear_key_heads)?;
    let value_heads = dim(config.linear_value_heads)?;
    let key_dim = dim(config.linear_key_head_dim)?;
    let value_dim = dim(config.linear_value_head_dim)?;

    let (convolved, next_conv) = short_convolution(
        window,
        &inputs.qkv,
        weights.tensor(&format!("{base}.conv1d.weight"))?,
        weights.numerics,
    )?;

    let parts = ops::split_sections_device(
        &convolved,
        &[key_heads * key_dim, 2 * key_heads * key_dim],
        2,
        &gpu,
    )?;
    let heads = |part: &Array, count: i32, width: i32| -> Result<Array, Qwen35Error> {
        Ok(part
            .reshape_device(&[seq, count, width], &gpu)?
            .as_type_device::<f32>(&gpu)?)
    };
    let readout_scale =
        f32::from(u16::try_from(key_dim).map_err(|_| Qwen35Error::ShapeOverflow)?).powf(-0.5);
    let query = l2_normalize(&heads(&parts[0], key_heads, key_dim)?)?
        .multiply_device(Array::from_f32(readout_scale), &gpu)?;
    let key = l2_normalize(&heads(&parts[1], key_heads, key_dim)?)?;
    let value = heads(&parts[2], value_heads, value_dim)?;
    // Key head h serves value heads h * repeat .. (h + 1) * repeat.
    let repeat = value_heads / key_heads;
    let (query, key) = if repeat > 1 {
        (
            Array::repeat_axis_device::<f32>(query, repeat, 1, &gpu)?,
            Array::repeat_axis_device::<f32>(key, repeat, 1, &gpu)?,
        )
    } else {
        (query, key)
    };

    let beta = ops::sigmoid_device(
        inputs
            .b
            .reshape_device(&[seq, value_heads], &gpu)?
            .as_type_device::<f32>(&gpu)?,
        &gpu,
    )?;
    let decay_rate = weights.tensor(&format!("{base}.decay_rate"))?;
    let shifted = inputs
        .a
        .reshape_device(&[seq, value_heads], &gpu)?
        .as_type_device::<f32>(&gpu)?
        .add_device(
            weights
                .tensor(&format!("{base}.dt_bias"))?
                .as_type_device::<f32>(&gpu)?,
            &gpu,
        )?;
    let softplus = ops::logaddexp_device(&shifted, Array::from_f32(0.0), &gpu)?;
    let log_decay = softplus
        .multiply_device(decay_rate, &gpu)?
        .negative_device(&gpu)?;

    let (output, recurrent) = if by_token || seq == 1 {
        delta_rule_by_token(&query, &key, &value, &log_decay, &beta, recurrent)?
    } else {
        delta::chunked(&query, &key, &value, &log_decay, &beta, recurrent)?
    };
    Ok((output, next_conv, recurrent))
}

/// Depthwise causal convolution plus `silu` over `[1, seq, channels]`, with
/// `window` (`[1, K - 1, channels]`) holding the preceding inputs and `kernel`
/// stored `[K, channels]`. Output `t` sums tap `j` times input `t + j` of the
/// window-prefixed sequence. Returns the output and the next window.
fn short_convolution(
    window: &Array,
    input: &Array,
    kernel: &Array,
    numerics: Numerics,
) -> Result<(Array, Array), Qwen35Error> {
    let gpu = StreamOrDevice::gpu();
    let seq = input.shape()[1];
    let history = window.shape()[1];
    let padded = ops::concatenate_axis_device(&[window, input], 1, &gpu)?;
    if numerics == Numerics::MlxLm {
        // mlx-lm Conv1d stores [channels, K, 1] and accumulates each depthwise
        // convolution before BF16 rounding. Explicit BF16 tap additions round
        // at each tap and do not reproduce that checkpoint's reference.
        let channels = input.shape()[2];
        let filter = kernel
            .transpose_device(&gpu)?
            .reshape_device(&[channels, history + 1, 1], &gpu)?;
        let convolved = ops::conv1d_device(&padded, &filter, 1, 0, 1, channels, &gpu)?;
        let total = history + seq;
        return Ok((
            mlx_silu(&convolved)?,
            padded.index((.., total - history..total, ..)),
        ));
    }
    let mut convolved = padded
        .index((.., 0..seq, ..))
        .multiply_device(kernel.index(0..1), &gpu)?;
    for tap in 1..=history {
        convolved = convolved.add_device(
            padded
                .index((.., tap..tap + seq, ..))
                .multiply_device(kernel.index(tap..tap + 1), &gpu)?,
            &gpu,
        )?;
    }
    let total = history + seq;
    Ok((
        silu(&convolved)?,
        padded.index((.., total - history..total, ..)),
    ))
}

/// The gated delta rule, one token at a time, used for decode; multi-token
/// calls use `delta::chunked`. `query`, `key` are `[seq, Hv, Dk]`, `value` is
/// `[seq, Hv, Dv]`, `log_decay` (`g`) and `beta` are `[seq, Hv]`, all f32, and
/// `state` is `[Hv, Dk, Dv]`. Returns the outputs `[seq, Hv, Dv]` and the
/// final state.
fn delta_rule_by_token(
    query: &Array,
    key: &Array,
    value: &Array,
    log_decay: &Array,
    beta: &Array,
    mut state: Array,
) -> Result<(Array, Array), Qwen35Error> {
    let gpu = StreamOrDevice::gpu();
    let decay = log_decay.exp_device(&gpu)?;
    let seq = query.shape()[0];
    let heads = query.shape()[1];
    // Rows of token t as [Hv, 1, D] (or [Hv, 1, 1] for per-head scalars).
    let row = |array: &Array, t: i32| -> Result<Array, Qwen35Error> {
        let token = array.index(t..t + 1);
        Ok(if token.ndim() == 3 {
            token.transpose_axes_device(&[1, 0, 2], &gpu)?
        } else {
            token.reshape_device(&[heads, 1, 1], &gpu)?
        })
    };
    let mut outputs =
        Vec::with_capacity(usize::try_from(seq).map_err(|_| Qwen35Error::ShapeOverflow)?);
    for t in 0..seq {
        let key_t = row(key, t)?;
        state = state.multiply_device(row(&decay, t)?, &gpu)?;
        let predicted = key_t.matmul_device(&state, &gpu)?;
        let delta = row(value, t)?
            .subtract_device(&predicted, &gpu)?
            .multiply_device(row(beta, t)?, &gpu)?;
        state = state.add_device(
            key_t
                .transpose_axes_device(&[0, 2, 1], &gpu)?
                .matmul_device(&delta, &gpu)?,
            &gpu,
        )?;
        outputs.push(row(query, t)?.matmul_device(&state, &gpu)?);
    }
    // [Hv, seq, Dv] -> [seq, Hv, Dv]
    let output =
        ops::concatenate_axis_device(&outputs, 1, &gpu)?.transpose_axes_device(&[1, 0, 2], &gpu)?;
    Ok((output, state))
}

/// Rotary embedding of `[1, heads, seq, head_dim]` at positions starting at
/// `offset`: the configuration's plain rope, or `YaRN` with its periods and
/// rotary-feature scale.
fn rope(
    x: &Array,
    config: &Qwen35Config,
    yarn: Option<&YarnArrays>,
    offset: i32,
) -> Result<Array, Qwen35Error> {
    let gpu = StreamOrDevice::gpu();
    Ok(match yarn {
        None => fast::rope_device(
            x,
            dim(config.rotary_dim)?,
            false,
            Some(config.rope_theta),
            1.0,
            offset,
            Option::<&Array>::None,
            &gpu,
        )?,
        Some(yarn) => fast::rope_device(
            &x.multiply_device(&yarn.feature_scale, &gpu)?,
            dim(config.rotary_dim)?,
            false,
            None,
            1.0,
            offset,
            Some(&yarn.periods),
            &gpu,
        )?,
    })
}

fn gated_attention(
    weights: &Qwen35Weights,
    base: &str,
    input: &Array,
    offset: i32,
    state: &mut Option<LayerState>,
) -> Result<Array, Qwen35Error> {
    let config = &weights.config;
    let gpu = StreamOrDevice::gpu();
    let seq = input.shape()[1];
    let heads = dim(config.attention_heads)?;
    let kv_heads = dim(config.key_value_heads)?;
    let head_dim = dim(config.head_dim)?;

    let project = |name: &str| weights.project(input, &format!("{base}.{name}.weight"));
    let projected = project("q_proj")?.reshape_device(&[1, seq, heads, 2 * head_dim], &gpu)?;
    let halves = ops::split_sections_device(&projected, &[head_dim], 3, &gpu)?;
    let gate = halves[1].reshape_device(&[1, seq, heads * head_dim], &gpu)?;
    let key = project("k_proj")?.reshape_device(&[1, seq, kv_heads, head_dim], &gpu)?;
    let value = project("v_proj")?
        .reshape_device(&[1, seq, kv_heads, head_dim], &gpu)?
        .transpose_axes_device(&[0, 2, 1, 3], &gpu)?;
    let rotate = |x: &Array, norm: &str| -> Result<Array, Qwen35Error> {
        let normed = rms_norm(
            x,
            weights.tensor(&format!("{base}.{norm}.weight"))?,
            config.rms_norm_eps,
        )?
        .transpose_axes_device(&[0, 2, 1, 3], &gpu)?;
        rope(&normed, config, weights.yarn.as_ref(), offset)
    };
    let query = rotate(&halves[0], "q_norm")?;
    let key = rotate(&key, "k_norm")?;

    let next = offset.checked_add(seq).ok_or(Qwen35Error::ShapeOverflow)?;
    let capacity =
        dim(config
            .kv_storage_tokens(usize::try_from(next).map_err(|_| Qwen35Error::ShapeOverflow)?))?;
    let storage = [1, kv_heads, capacity, head_dim];
    let (mut keys, mut values) = match state.take() {
        Some(LayerState::Full { keys, values }) if keys.shape()[2] >= next => (keys, values),
        // Grow to the next tier: one copy of the valid prefix per tier.
        Some(LayerState::Full { keys, values }) => {
            let mut grown_keys = ops::zeros_dtype_device(&storage, key.dtype(), &gpu)?;
            let mut grown_values = ops::zeros_dtype_device(&storage, value.dtype(), &gpu)?;
            grown_keys.index_mut_device(
                (.., .., 0..offset, ..),
                &keys.index_device((.., .., 0..offset, ..), &gpu),
                &gpu,
            );
            grown_values.index_mut_device(
                (.., .., 0..offset, ..),
                &values.index_device((.., .., 0..offset, ..), &gpu),
                &gpu,
            );
            (grown_keys, grown_values)
        }
        Some(LayerState::Linear { .. }) => return Err(Qwen35Error::StateKind),
        None => (
            ops::zeros_dtype_device(&storage, key.dtype(), &gpu)?,
            ops::zeros_dtype_device(&storage, value.dtype(), &gpu)?,
        ),
    };
    keys.index_mut_device((.., .., offset..next, ..), &key, &gpu);
    values.index_mut_device((.., .., offset..next, ..), &value, &gpu);
    let attention_keys = keys.index_device((.., .., 0..next, ..), &gpu);
    let attention_values = values.index_device((.., .., 0..next, ..), &gpu);
    // MLX 0.25's fused causal mask is misaligned when a multi-token chunk
    // follows cached positions, so such chunks pass an explicit mask. A first
    // chunk uses the built-in causal mask; one decode token needs none.
    let explicit = if offset > 0 && seq > 1 {
        let total = offset + seq;
        let rows = Array::arange_device::<i32, i32>(offset, total, None, &gpu)?
            .reshape_device(&[seq, 1], &gpu)?;
        let columns = Array::arange_device::<i32, i32>(0, total, None, &gpu)?
            .reshape_device(&[1, total], &gpu)?;
        Some(rows.ge_device(&columns, &gpu)?)
    } else {
        None
    };
    let mask = match (&explicit, offset, seq) {
        (Some(mask), _, _) => Some(fast::ScaledDotProductAttentionMask::Array(mask)),
        (None, 0, 2..) => Some(fast::ScaledDotProductAttentionMask::Causal),
        (None, _, _) => None,
    };
    let scale =
        f32::from(u16::try_from(head_dim).map_err(|_| Qwen35Error::ShapeOverflow)?).powf(-0.5);
    let attended = fast::scaled_dot_product_attention_device(
        &query,
        &attention_keys,
        &attention_values,
        scale,
        mask,
        Option::<&Array>::None,
        &gpu,
    )?
    .transpose_axes_device(&[0, 2, 1, 3], &gpu)?
    .reshape_device(&[1, seq, heads * head_dim], &gpu)?;
    *state = Some(LayerState::Full { keys, values });
    let gated = attended.multiply_device(ops::sigmoid_device(&gate, &gpu)?, &gpu)?;
    weights.project(&gated, &format!("{base}.o_proj.weight"))
}

fn mlp(weights: &Qwen35Weights, base: &str, input: &Array) -> Result<Array, Qwen35Error> {
    let gpu = StreamOrDevice::gpu();
    let matrix = |name: &str| weights.matrix(&format!("{base}.mlp.{name}"));
    match weights.config.mlp {
        Qwen35Mlp::Dense { .. } => {
            let project = |name: &str, x: &Array| weights.project(x, &format!("{base}.mlp.{name}"));
            let hidden = gated(
                &project("gate_proj.weight", input)?,
                &project("up_proj.weight", input)?,
                weights.numerics,
            )?;
            project("down_proj.weight", &hidden)
        }
        Qwen35Mlp::Experts { top_k, .. } => {
            let experts = match (weights.numerics, &weights.paged) {
                (Numerics::Transformers, _) => Experts::Fused {
                    gate_up: weights.tensor(&format!("{base}.mlp.experts.gate_up_proj"))?,
                    down: weights.tensor(&format!("{base}.mlp.experts.down_proj"))?,
                },
                (Numerics::MlxLm, Some(paged)) => Experts::Paged {
                    paged,
                    layer: base
                        .strip_prefix("layers.")
                        .and_then(|layer| layer.parse().ok())
                        .ok_or(Qwen35Error::ShapeOverflow)?,
                },
                (Numerics::MlxLm, None) => Experts::Split {
                    gate: matrix("switch_mlp.gate_proj.weight")?,
                    up: matrix("switch_mlp.up_proj.weight")?,
                    down: matrix("switch_mlp.down_proj.weight")?,
                },
            };
            let block = MoeTensors {
                router: matrix("gate.weight")?,
                experts,
                shared_gate: matrix("shared_expert.gate_proj.weight")?,
                shared_up: matrix("shared_expert.up_proj.weight")?,
                shared_down: matrix("shared_expert.down_proj.weight")?,
                shared_scale: matrix("shared_expert_gate.weight")?,
            };
            match weights.numerics {
                // Transformers flattens the batch and sequence axes.
                Numerics::Transformers => {
                    let shape = input.shape().to_vec();
                    let tokens = input.reshape_device(&[-1, shape[shape.len() - 1]], &gpu)?;
                    Ok(moe_block(&tokens, &block, top_k)?.reshape_device(&shape, &gpu)?)
                }
                Numerics::MlxLm => moe_block(input, &block, top_k),
            }
        }
    }
}

/// A stored weight matrix, or a stack of them along leading axes.
#[derive(Clone, Copy)]
enum Matrix<'a> {
    Dense(&'a Array),
    Affine(&'a AffineWeight),
}

impl Matrix<'_> {
    /// `input` times the transpose of the matrix.
    fn apply(self, input: &Array) -> Result<Array, Qwen35Error> {
        match self {
            Self::Dense(weight) => linear(input, weight),
            Self::Affine(weight) => Ok(ops::quantized_matmul_device(
                input,
                &weight.codes,
                &weight.scales,
                &weight.biases,
                true,
                weight.group_size,
                weight.bits,
                StreamOrDevice::gpu(),
            )?),
        }
    }

    /// `input` times the transpose of the stacked matrix each of `indices`
    /// selects (`gather_mm` or `gather_qmm`).
    fn gather(self, input: &Array, indices: &Array, sorted: bool) -> Result<Array, Qwen35Error> {
        let gpu = StreamOrDevice::gpu();
        match self {
            Self::Dense(weight) => Ok(ops::gather_mm_device(
                input,
                weight.swap_axes_device(-1, -2, &gpu)?,
                None,
                indices,
                sorted,
                &gpu,
            )?),
            Self::Affine(weight) => Ok(ops::gather_qmm_device(
                input,
                &weight.codes,
                &weight.scales,
                &weight.biases,
                None,
                indices,
                true,
                weight.group_size,
                weight.bits,
                sorted,
                &gpu,
            )?),
        }
    }
}

/// One layer's routed experts.
enum Experts<'a> {
    /// Hugging Face's dense layout: `gate_up` `[E, 2I, H]` (gate rows, then up
    /// rows) and `down` `[E, H, I]`.
    Fused { gate_up: &'a Array, down: &'a Array },
    /// mlx-lm's layout: `gate` and `up` `[E, I, H]`, `down` `[E, H, I]`.
    Split {
        gate: Matrix<'a>,
        up: Matrix<'a>,
        down: Matrix<'a>,
    },
    /// mlx-lm's layout with the experts read on demand.
    Paged {
        paged: &'a paged::PagedExperts,
        layer: usize,
    },
}

/// One layer's `MoE` weights, as stored.
struct MoeTensors<'a> {
    /// `[E, H]`.
    router: Matrix<'a>,
    experts: Experts<'a>,
    shared_gate: Matrix<'a>,
    shared_up: Matrix<'a>,
    shared_down: Matrix<'a>,
    /// `[1, H]`; `sigmoid` of its product scales the shared expert.
    shared_scale: Matrix<'a>,
}

/// The experts each token is sent to and their mixing weights, both
/// `[.., top_k]`.
struct Routing {
    chosen: Array,
    weights: Array,
}

/// The routed experts plus the gated shared expert over `tokens` `[.., H]`.
/// Fused experts take Transformers' routing and `[n, H]` tokens; split
/// experts take mlx-lm's.
fn moe_block(tokens: &Array, block: &MoeTensors<'_>, top_k: usize) -> Result<Array, Qwen35Error> {
    let gpu = StreamOrDevice::gpu();
    let numerics = match block.experts {
        Experts::Fused { .. } => Numerics::Transformers,
        Experts::Split { .. } | Experts::Paged { .. } => Numerics::MlxLm,
    };
    let routed = match block.experts {
        Experts::Fused { gate_up, down } => {
            let routing = route_f32(tokens, block.router, top_k)?;
            fused_experts(tokens, gate_up, down, &routing)?
        }
        Experts::Split { gate, up, down } => {
            let routing = route_mlx(tokens, block.router, top_k)?;
            split_experts(tokens, [gate, up, down], &routing)?
        }
        Experts::Paged { paged, layer } => {
            let routing = route_mlx(tokens, block.router, top_k)?;
            let (stack, routing) = paged.stack(layer, &routing)?;
            split_experts(tokens, stack.matrices(), &routing)?
        }
    };
    let shared = swiglu(
        tokens,
        [block.shared_gate, block.shared_up, block.shared_down],
        numerics,
    )?
    .multiply_device(
        ops::sigmoid_device(block.shared_scale.apply(tokens)?, &gpu)?,
        &gpu,
    )?;
    Ok(routed.add_device(&shared, &gpu)?)
}

fn swiglu(
    input: &Array,
    [gate, up, down]: [Matrix<'_>; 3],
    numerics: Numerics,
) -> Result<Array, Qwen35Error> {
    down.apply(&gated(&gate.apply(input)?, &up.apply(input)?, numerics)?)
}

type CompiledGraph =
    Box<dyn for<'a> FnMut(&'a [Array]) -> Result<Vec<Array>, mlx_rs::error::Exception>>;

thread_local! {
    /// mlx-lm's `swiglu`, compiled once per thread (MLX keeps compiled
    /// functions per thread).
    static FUSED_SWIGLU: RefCell<Option<CompiledGraph>> = const { RefCell::new(None) };
    static FUSED_SILU: RefCell<Option<CompiledGraph>> = const { RefCell::new(None) };
}

/// mlx.nn.silu uses one compiled x * sigmoid(x) graph, preserving its
/// checkpoint-dtype rounding rather than materializing an intermediate sigmoid.
fn mlx_silu(input: &Array) -> Result<Array, Qwen35Error> {
    FUSED_SILU.with(|cell| {
        let mut cell = cell.borrow_mut();
        let compiled = cell.get_or_insert_with(|| {
            Box::new(mlx_rs::transforms::compile::compile(
                |inputs: &[Array]| {
                    let x = &inputs[0];
                    Ok(vec![x.multiply(ops::sigmoid(x)?)?])
                },
                true,
            ))
        });
        let mut outputs = compiled(std::slice::from_ref(input))?;
        outputs.pop().ok_or(Qwen35Error::ShapeOverflow)
    })
}

/// `silu(gate) * up`. mlx-lm computes it as one compiled, fused kernel
/// (`mx.compile(shapeless=True)`), which rounds differently from the same
/// ops run one by one: on 16.7M random BF16 inputs, 5,110 outputs differ.
/// mlx-lm-layout checkpoints therefore use the compiled form, the others
/// the separate ops Transformers' rounding follows.
fn gated(gate: &Array, up: &Array, numerics: Numerics) -> Result<Array, Qwen35Error> {
    match numerics {
        Numerics::Transformers => Ok(silu(gate)?.multiply_device(up, StreamOrDevice::gpu())?),
        Numerics::MlxLm => FUSED_SWIGLU.with(|cell| {
            let mut cell = cell.borrow_mut();
            let compiled = cell.get_or_insert_with(|| {
                Box::new(mlx_rs::transforms::compile::compile(
                    |inputs: &[Array]| {
                        let (gate, up) = (&inputs[0], &inputs[1]);
                        // nn.silu is `x * sigmoid(x)`.
                        Ok(vec![gate.multiply(ops::sigmoid(gate)?)?.multiply(up)?])
                    },
                    true,
                ))
            });
            let mut outputs = compiled(&[gate.clone(), up.clone()])?;
            outputs.pop().ok_or(Qwen35Error::ShapeOverflow)
        }),
    }
}

/// Transformers' routing: softmax over the router logits in f32, the
/// `top_k` largest renormalized to sum to 1 in f32, then cast to the
/// activation dtype.
fn route_f32(tokens: &Array, router: Matrix<'_>, top_k: usize) -> Result<Routing, Qwen35Error> {
    let gpu = StreamOrDevice::gpu();
    let probabilities = ops::softmax_axis_device(
        router.apply(tokens)?.as_type_device::<f32>(&gpu)?,
        -1,
        true,
        &gpu,
    )?;
    let (chosen, weights) = top(&probabilities, top_k)?;
    let weights = weights
        .divide_device(weights.sum_axis_device(-1, true, &gpu)?, &gpu)?
        .as_dtype_device(tokens.dtype(), &gpu)?;
    Ok(Routing { chosen, weights })
}

/// mlx-lm's routing: the same steps in the activation dtype (the softmax
/// accumulates in f32 but returns that dtype).
fn route_mlx(tokens: &Array, router: Matrix<'_>, top_k: usize) -> Result<Routing, Qwen35Error> {
    let gpu = StreamOrDevice::gpu();
    let probabilities = ops::softmax_axis_device(router.apply(tokens)?, -1, true, &gpu)?;
    let (chosen, weights) = top(&probabilities, top_k)?;
    let weights = weights.divide_device(weights.sum_axis_device(-1, true, &gpu)?, &gpu)?;
    Ok(Routing { chosen, weights })
}

/// The indices and values of the `top_k` largest `probabilities` on the last
/// axis, in no particular order.
fn top(probabilities: &Array, top_k: usize) -> Result<(Array, Array), Qwen35Error> {
    let gpu = StreamOrDevice::gpu();
    let experts = probabilities.shape()[probabilities.ndim() - 1];
    // After partitioning at `experts - top_k`, the last `top_k` positions
    // hold the largest probabilities.
    let kth = experts - dim(top_k)?;
    let chosen =
        ops::argpartition_axis_device(probabilities, kth, -1, &gpu)?.index((Ellipsis, kth..));
    let weights = probabilities.take_along_axis_device(&chosen, -1, &gpu)?;
    Ok((chosen, weights))
}

/// Fused experts over `tokens` `[n, H]`: each chosen expert applies `SwiGLU`
/// with its slice of `gate_up` and `down`; returns the weighted sum, `[n, H]`.
fn fused_experts(
    tokens: &Array,
    gate_up: &Array,
    down: &Array,
    routing: &Routing,
) -> Result<Array, Qwen35Error> {
    let gpu = StreamOrDevice::gpu();
    let width = down.shape()[2];
    // [n, 1, 1, H] against [E, H, 2I] gathered by `chosen` [n, k]: [n, k, 1, 2I].
    let rows = tokens.expand_dims_axes_device(&[-2, -3], &gpu)?;
    let projected = Matrix::Dense(gate_up).gather(&rows, &routing.chosen, false)?;
    let halves = ops::split_sections_device(&projected, &[width], -1, &gpu)?;
    let hidden = silu(&halves[0])?.multiply_device(&halves[1], &gpu)?;
    let outputs = Matrix::Dense(down)
        .gather(&hidden, &routing.chosen, false)?
        .squeeze_axes_device(&[-2], &gpu)?;
    Ok(outputs
        .multiply_device(routing.weights.expand_dims_device(-1, &gpu)?, &gpu)?
        .sum_axis_device(1, false, &gpu)?)
}

/// Split experts over `tokens` `[.., H]` as mlx-lm's `SwitchGLU` runs them:
/// with 64 or more selections the rows are sorted by expert first, so each
/// expert's weights are read once. Returns the weighted sum, `[.., H]`.
fn split_experts(
    tokens: &Array,
    [gate, up, down]: [Matrix<'_>; 3],
    routing: &Routing,
) -> Result<Array, Qwen35Error> {
    let gpu = StreamOrDevice::gpu();
    let rows = tokens.expand_dims_axes_device(&[-2, -3], &gpu)?;
    let top_k = routing.chosen.shape()[routing.chosen.ndim() - 1];
    let sorted = routing.chosen.size() >= 64;
    let (rows, chosen, restore) = if sorted {
        // Row `i` of the flattened selections belongs to token `i / top_k`.
        let flat = routing.chosen.reshape_device(&[-1], &gpu)?;
        let order = ops::argsort_axis_device(&flat, 0, &gpu)?;
        let restore = ops::argsort_axis_device(&order, 0, &gpu)?;
        let shape = rows.shape();
        let rows = rows
            .reshape_device(&[-1, shape[shape.len() - 2], shape[shape.len() - 1]], &gpu)?
            .take_axis_device(
                order.floor_divide_device(Array::from_int(top_k), &gpu)?,
                0,
                &gpu,
            )?;
        (rows, flat.take_axis_device(&order, 0, &gpu)?, Some(restore))
    } else {
        (rows, routing.chosen.clone(), None)
    };
    let projected_up = up.gather(&rows, &chosen, sorted)?;
    let projected_gate = gate.gather(&rows, &chosen, sorted)?;
    let hidden = gated(&projected_gate, &projected_up, Numerics::MlxLm)?;
    let mut outputs = down.gather(&hidden, &chosen, sorted)?;
    if let Some(restore) = restore {
        let shape = [routing.chosen.shape(), &outputs.shape()[1..]].concat();
        outputs = outputs
            .take_axis_device(&restore, 0, &gpu)?
            .reshape_device(&shape, &gpu)?;
    }
    Ok(outputs
        .squeeze_axes_device(&[-2], &gpu)?
        .multiply_device(routing.weights.expand_dims_device(-1, &gpu)?, &gpu)?
        .sum_axis_device(-2, false, &gpu)?)
}

/// `x / sqrt(sum(x^2) + eps)` over the last axis, matching the source's FLA
/// convention of adding epsilon to the sum rather than the mean.
fn l2_normalize(x: &Array) -> Result<Array, Qwen35Error> {
    let gpu = StreamOrDevice::gpu();
    let inverse = x
        .square_device(&gpu)?
        .sum_axis_device(-1, true, &gpu)?
        .add_device(Array::from_f32(L2_NORM_EPS), &gpu)?
        .rsqrt_device(&gpu)?;
    Ok(x.multiply_device(&inverse, &gpu)?)
}

fn silu(x: &Array) -> Result<Array, Qwen35Error> {
    let gpu = StreamOrDevice::gpu();
    Ok(ops::sigmoid_device(x, &gpu)?.multiply_device(x, &gpu)?)
}

fn rms_norm(input: &Array, scale: &Array, eps: f32) -> Result<Array, Qwen35Error> {
    Ok(fast::rms_norm_device(
        input,
        scale,
        eps,
        StreamOrDevice::gpu(),
    )?)
}

fn linear(input: &Array, weight: &Array) -> Result<Array, Qwen35Error> {
    let gpu = StreamOrDevice::gpu();
    Ok(input.matmul_device(weight.transpose_device(&gpu)?, &gpu)?)
}

fn dim(value: usize) -> Result<i32, Qwen35Error> {
    i32::try_from(value).map_err(|_| Qwen35Error::ShapeOverflow)
}

/// Norm weights stored as `weight - 1`. The multi-token-prediction head's
/// final and pre-projection norms are listed by name: their names end in
/// neither `norm.weight` alone nor a layer-norm suffix, and missing them
/// leaves the head's inputs scaled by roughly zero.
fn is_zero_centered_norm(name: &str) -> bool {
    matches!(
        name,
        "norm.weight"
            | "mtp.norm.weight"
            | "mtp.pre_fc_norm_embedding.weight"
            | "mtp.pre_fc_norm_hidden.weight"
    ) || [
        ".input_layernorm.weight",
        ".post_attention_layernorm.weight",
        ".q_norm.weight",
        ".k_norm.weight",
    ]
    .iter()
    .any(|suffix| name.ends_with(suffix))
}

/// The stored shapes of one layer's `MoE` tensors under `mlp`, from the
/// expert count and the routed and shared expert widths. Hugging Face fuses
/// the routed gate and up projections; mlx-lm splits them.
fn expert_shapes(
    put: &mut impl FnMut(String, &[i32]),
    mlp: &str,
    hidden: i32,
    [experts, expert, shared]: [usize; 3],
    numerics: Numerics,
) -> Result<(), Qwen35Error> {
    let (experts, expert, shared) = (dim(experts)?, dim(expert)?, dim(shared)?);
    put(format!("{mlp}.gate.weight"), &[experts, hidden]);
    match numerics {
        Numerics::Transformers => {
            put(
                format!("{mlp}.experts.gate_up_proj"),
                &[experts, 2 * expert, hidden],
            );
            put(
                format!("{mlp}.experts.down_proj"),
                &[experts, hidden, expert],
            );
        }
        Numerics::MlxLm => {
            let routed = format!("{mlp}.switch_mlp");
            put(
                format!("{routed}.gate_proj.weight"),
                &[experts, expert, hidden],
            );
            put(
                format!("{routed}.up_proj.weight"),
                &[experts, expert, hidden],
            );
            put(
                format!("{routed}.down_proj.weight"),
                &[experts, hidden, expert],
            );
        }
    }
    put(
        format!("{mlp}.shared_expert.gate_proj.weight"),
        &[shared, hidden],
    );
    put(
        format!("{mlp}.shared_expert.up_proj.weight"),
        &[shared, hidden],
    );
    put(
        format!("{mlp}.shared_expert.down_proj.weight"),
        &[hidden, shared],
    );
    put(format!("{mlp}.shared_expert_gate.weight"), &[1, hidden]);
    Ok(())
}

/// The stored shapes of one gated full-attention block under `attn`.
fn full_attention_shapes(
    put: &mut impl FnMut(String, &[i32]),
    attn: &str,
    config: &Qwen35Config,
) -> Result<(), Qwen35Error> {
    let hidden = dim(config.hidden_size)?;
    let heads = dim(config.attention_heads)?;
    let kv_heads = dim(config.key_value_heads)?;
    let head_dim = dim(config.head_dim)?;
    put(
        format!("{attn}.q_proj.weight"),
        &[2 * heads * head_dim, hidden],
    );
    put(
        format!("{attn}.k_proj.weight"),
        &[kv_heads * head_dim, hidden],
    );
    put(
        format!("{attn}.v_proj.weight"),
        &[kv_heads * head_dim, hidden],
    );
    put(format!("{attn}.o_proj.weight"), &[hidden, heads * head_dim]);
    put(format!("{attn}.q_norm.weight"), &[head_dim]);
    put(format!("{attn}.k_norm.weight"), &[head_dim]);
    Ok(())
}

/// The stored shapes of one dense `SwiGLU` block under `mlp`.
fn dense_mlp_shapes(
    put: &mut impl FnMut(String, &[i32]),
    mlp: &str,
    hidden: i32,
    intermediate_size: usize,
) -> Result<(), Qwen35Error> {
    let intermediate = dim(intermediate_size)?;
    put(format!("{mlp}.gate_proj.weight"), &[intermediate, hidden]);
    put(format!("{mlp}.up_proj.weight"), &[intermediate, hidden]);
    put(format!("{mlp}.down_proj.weight"), &[hidden, intermediate]);
    Ok(())
}

/// Refuses a multi-token-prediction head this loader does not implement.
fn check_mtp(config: &Qwen35Config) -> Result<(), Qwen35Error> {
    if config.mtp_layers() != 1 {
        return Err(Qwen35Error::MtpUnsupported(
            "a checkpoint without exactly one multi-token-prediction layer",
        ));
    }
    if config.mtp_dedicated_embeddings() {
        return Err(Qwen35Error::MtpUnsupported(
            "a head with its own embedding and LM head",
        ));
    }
    if matches!(config.mlp, Qwen35Mlp::Experts { .. }) {
        return Err(Qwen35Error::MtpUnsupported("a mixture-of-experts head"));
    }
    Ok(())
}

/// The stored shapes of the one-layer dense multi-token-prediction head:
/// pre-projection norms of the token embedding and the decoder's hidden
/// state, `fc` over their concatenation, one gated full-attention decoder
/// layer, and a final norm. The embedding and LM head are the decoder's.
fn mtp_shapes(config: &Qwen35Config) -> Result<HashMap<String, Vec<i32>>, Qwen35Error> {
    let Qwen35Mlp::Dense { intermediate_size } = config.mlp else {
        return Err(Qwen35Error::MtpUnsupported("a mixture-of-experts head"));
    };
    let hidden = dim(config.hidden_size)?;
    let mut shapes = HashMap::new();
    let mut put = |name: String, shape: &[i32]| {
        shapes.insert(name, shape.to_vec());
    };
    put("mtp.pre_fc_norm_embedding.weight".into(), &[hidden]);
    put("mtp.pre_fc_norm_hidden.weight".into(), &[hidden]);
    put("mtp.fc.weight".into(), &[hidden, 2 * hidden]);
    put("mtp.norm.weight".into(), &[hidden]);
    put("mtp.layers.0.input_layernorm.weight".into(), &[hidden]);
    put(
        "mtp.layers.0.post_attention_layernorm.weight".into(),
        &[hidden],
    );
    full_attention_shapes(&mut put, "mtp.layers.0.self_attn", config)?;
    dense_mlp_shapes(&mut put, "mtp.layers.0.mlp", hidden, intermediate_size)?;
    Ok(shapes)
}

/// Every decoder tensor this implementation reads, with its stored (for a
/// quantized module, its dense) shape in the layout `numerics` implies.
fn expected_shapes(
    config: &Qwen35Config,
    numerics: Numerics,
) -> Result<HashMap<String, Vec<i32>>, Qwen35Error> {
    let hidden = dim(config.hidden_size)?;
    let vocab = dim(config.vocab_size)?;
    let value_heads = dim(config.linear_value_heads)?;
    let value_width = dim(config.linear_value_heads * config.linear_value_head_dim)?;
    let conv = dim(config.conv_dim())?;
    let kernel = dim(config.conv_kernel)?;

    let mut shapes = HashMap::new();
    let mut put = |name: String, shape: &[i32]| {
        shapes.insert(name, shape.to_vec());
    };
    put("embed_tokens.weight".into(), &[vocab, hidden]);
    put("norm.weight".into(), &[hidden]);
    if !config.tie_word_embeddings {
        put("lm_head.weight".into(), &[vocab, hidden]);
    }
    for (index, kind) in config.layers.iter().enumerate() {
        let base = format!("layers.{index}");
        put(format!("{base}.input_layernorm.weight"), &[hidden]);
        put(format!("{base}.post_attention_layernorm.weight"), &[hidden]);
        let mlp = format!("{base}.mlp");
        match config.mlp {
            Qwen35Mlp::Dense { intermediate_size } => {
                dense_mlp_shapes(&mut put, &mlp, hidden, intermediate_size)?;
            }
            Qwen35Mlp::Experts {
                experts,
                expert_intermediate_size,
                shared_intermediate_size,
                ..
            } => expert_shapes(
                &mut put,
                &mlp,
                hidden,
                [experts, expert_intermediate_size, shared_intermediate_size],
                numerics,
            )?,
        }
        match kind {
            Qwen35LayerKind::LinearAttention => {
                let attn = format!("{base}.linear_attn");
                put(format!("{attn}.in_proj_qkv.weight"), &[conv, hidden]);
                put(format!("{attn}.in_proj_z.weight"), &[value_width, hidden]);
                put(format!("{attn}.in_proj_b.weight"), &[value_heads, hidden]);
                put(format!("{attn}.in_proj_a.weight"), &[value_heads, hidden]);
                put(
                    format!("{attn}.conv1d.weight"),
                    &match numerics {
                        Numerics::Transformers => [conv, 1, kernel],
                        Numerics::MlxLm => [conv, kernel, 1],
                    },
                );
                put(format!("{attn}.A_log"), &[value_heads]);
                put(format!("{attn}.dt_bias"), &[value_heads]);
                put(
                    format!("{attn}.norm.weight"),
                    &[dim(config.linear_value_head_dim)?],
                );
                put(format!("{attn}.out_proj.weight"), &[hidden, value_width]);
            }
            Qwen35LayerKind::FullAttention => {
                full_attention_shapes(&mut put, &format!("{base}.self_attn"), config)?;
            }
        }
    }
    Ok(shapes)
}

/// The text-decoder tensors of a safetensors checkpoint by name without its
/// prefix, and the layout they were stored in. An untied output head is
/// `lm_head.*` in both layouts. With `mtp`, the multi-token-prediction head's
/// `mtp.*` tensors are kept too; they exist only in the Transformers layout.
fn read_text_tensors(
    model_dir: &Path,
    config: &Qwen35Config,
    mtp: bool,
) -> Result<(HashMap<String, Array>, Numerics), Qwen35Error> {
    let mut tensors = HashMap::new();
    let mut layouts = (false, false);
    for shard in shard_paths(model_dir)? {
        for (name, tensor) in Array::load_safetensors_device(&shard, StreamOrDevice::cpu())? {
            let (name, mlx) = if let Some(name) = name.strip_prefix(TEXT_PREFIX) {
                (name.to_owned(), false)
            } else if let Some(name) = name.strip_prefix(MLX_TEXT_PREFIX) {
                (name.to_owned(), true)
            } else if let Some(part) = name.strip_prefix(MLX_HEAD_PREFIX) {
                (format!("lm_head.{part}"), true)
            } else if name.starts_with("lm_head.") || (mtp && name.starts_with(MTP_PREFIX)) {
                (name, false)
            } else {
                continue;
            };
            if name.starts_with("lm_head.") && config.tie_word_embeddings() {
                continue;
            }
            if mlx {
                layouts.1 = true;
            } else {
                layouts.0 = true;
            }
            tensors.insert(name, tensor);
        }
    }
    match layouts {
        (true, true) => Err(Qwen35Error::MixedLayouts),
        (false, true) => Ok((tensors, Numerics::MlxLm)),
        _ => Ok((tensors, Numerics::Transformers)),
    }
}

/// Moves every module stored with `scales` out of `tensors` as an
/// [`AffineWeight`] keyed `stem.weight`, with the configuration's bits and
/// group size for it.
fn take_quantized(
    config: &Qwen35Config,
    tensors: &mut HashMap<String, Array>,
) -> Result<HashMap<String, AffineWeight>, Qwen35Error> {
    let stems: Vec<String> = tensors
        .keys()
        .filter_map(|name| name.strip_suffix(".scales"))
        .map(str::to_owned)
        .collect();
    let mut quantized = HashMap::new();
    for stem in stems {
        let Some(affine) = config
            .quantization()
            .and_then(|quantization| quantization.module(&stem))
        else {
            return Err(Qwen35Error::UndeclaredQuantization(stem));
        };
        let mut take = |part: &str| {
            let name = format!("{stem}.{part}");
            tensors
                .remove(&name)
                .ok_or(Qwen35Error::MissingTensor(name))
        };
        let weight = AffineWeight {
            codes: take("weight")?,
            scales: take("scales")?,
            biases: take("biases")?,
            bits: i32::from(affine.bits()),
            group_size: i32::from(affine.group_size()),
        };
        quantized.insert(format!("{stem}.weight"), weight);
    }
    Ok(quantized)
}

/// Checks every expected tensor is stored once with its shape: dense as
/// listed, quantized as `u32` codes `[.., rows, columns * bits / 32]` with
/// scales and biases `[.., rows, columns / group_size]`.
fn check_shapes(
    expected: &HashMap<String, Vec<i32>>,
    tensors: &HashMap<String, Array>,
    quantized: &HashMap<String, AffineWeight>,
) -> Result<(), Qwen35Error> {
    let mismatch = |name: &str, expected: &[i32], actual: &[i32]| {
        Err(Qwen35Error::TensorShape {
            name: name.to_owned(),
            expected: expected.to_vec(),
            actual: actual.to_vec(),
        })
    };
    for (name, shape) in expected {
        if let Some(weight) = quantized.get(name) {
            let (columns, rows) = shape.split_last().ok_or(Qwen35Error::ShapeOverflow)?;
            let packed = |width: i32| [rows, &[width]].concat();
            let codes = packed(columns * weight.bits / 32);
            let groups = packed(columns / weight.group_size);
            if weight.codes.dtype() != Dtype::Uint32
                || columns % weight.group_size != 0
                || (columns * weight.bits) % 32 != 0
            {
                return mismatch(name, &codes, weight.codes.shape());
            }
            for (part, stored) in [
                (name.replace(".weight", ".scales"), &weight.scales),
                (name.replace(".weight", ".biases"), &weight.biases),
            ] {
                if !matches!(
                    stored.dtype(),
                    Dtype::Float16 | Dtype::Bfloat16 | Dtype::Float32
                ) {
                    return Err(Qwen35Error::QuantizationParameterDtype {
                        name: part,
                        dtype: format!("{:?}", stored.dtype()),
                    });
                }
            }
            for (part, stored, want) in [
                (name.clone(), &weight.codes, &codes),
                (name.replace(".weight", ".scales"), &weight.scales, &groups),
                (name.replace(".weight", ".biases"), &weight.biases, &groups),
            ] {
                if stored.shape() != want.as_slice() {
                    return mismatch(&part, want, stored.shape());
                }
            }
            continue;
        }
        let tensor = tensors
            .get(name)
            .ok_or_else(|| Qwen35Error::MissingTensor(name.clone()))?;
        if tensor.shape() != shape.as_slice() {
            return mismatch(name, shape, tensor.shape());
        }
    }
    if tensors.len() + quantized.len() != expected.len() {
        let mut unexpected: Vec<_> = tensors
            .keys()
            .chain(quantized.keys())
            .filter(|name| !expected.contains_key(*name))
            .cloned()
            .collect();
        unexpected.sort();
        return Err(Qwen35Error::UnexpectedTensors(unexpected));
    }
    Ok(())
}

/// Largest f32 footprint [`Qwen35Precision::Float32`] dequantizes a
/// quantized checkpoint into. f32 exists to compare small checkpoints with an
/// f32 reference; a 35B checkpoint would need about 140 GB.
const MAX_DEQUANTIZED_F32_BYTES: u64 = 16 << 30;

/// Refuses an f32 load whose dequantized matrices would exceed
/// [`MAX_DEQUANTIZED_F32_BYTES`].
fn check_dequantized_bytes(quantized: &HashMap<String, AffineWeight>) -> Result<(), Qwen35Error> {
    let bytes = quantized
        .values()
        .map(|weight| {
            // Each scale covers `group_size` values.
            u64::try_from(weight.scales.size())
                .ok()
                .and_then(|groups| groups.checked_mul(u64::try_from(weight.group_size).ok()?))
                .and_then(|values| values.checked_mul(4))
                .ok_or(Qwen35Error::ShapeOverflow)
        })
        .try_fold(0_u64, |total, bytes| {
            total.checked_add(bytes?).ok_or(Qwen35Error::ShapeOverflow)
        })?;
    if bytes > MAX_DEQUANTIZED_F32_BYTES {
        return Err(Qwen35Error::DequantizedTooLarge {
            bytes,
            limit: MAX_DEQUANTIZED_F32_BYTES,
        });
    }
    Ok(())
}

#[derive(Deserialize)]
struct ShardIndex {
    weight_map: HashMap<String, String>,
}

/// The single-file checkpoint, or every shard named by the index. Shard names
/// must be plain file names inside the model directory.
fn shard_paths(model_dir: &Path) -> Result<Vec<PathBuf>, Qwen35Error> {
    let single = model_dir.join("model.safetensors");
    if single.is_file() {
        return Ok(vec![single]);
    }
    let index: ShardIndex =
        serde_json::from_str(&read_file(&model_dir.join("model.safetensors.index.json"))?)
            .map_err(Qwen35Error::Index)?;
    let mut names: Vec<&String> = index.weight_map.values().collect();
    names.sort();
    names.dedup();
    names
        .into_iter()
        .map(|name| {
            let flat = Path::new(name)
                .file_name()
                .is_some_and(|file| file == name.as_str())
                && name.ends_with(".safetensors");
            if flat {
                Ok(model_dir.join(name))
            } else {
                Err(Qwen35Error::UnsafeShard(name.clone()))
            }
        })
        .collect()
}

fn read_file(path: &Path) -> Result<String, Qwen35Error> {
    fs::read_to_string(path).map_err(|source| Qwen35Error::Read {
        path: path.to_owned(),
        source,
    })
}

/// A failure while loading or executing the hybrid decoder.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum Qwen35Error {
    /// The configuration is invalid or unsupported.
    #[error(transparent)]
    Config(#[from] Qwen35ConfigError),
    /// A checkpoint file could not be read.
    #[error("could not read {path}: {source}")]
    Read {
        /// The file that failed.
        path: PathBuf,
        /// The underlying I/O error.
        source: std::io::Error,
    },
    /// The shard index is not valid JSON.
    #[error("invalid model.safetensors.index.json: {0}")]
    Index(serde_json::Error),
    /// The shard index names a path outside the model directory.
    #[error("unsafe shard file name {0:?}")]
    UnsafeShard(String),
    /// MLX could not load a shard.
    #[error("could not load safetensors: {0}")]
    Load(#[from] mlx_rs::error::IoError),
    /// A required decoder tensor is absent.
    #[error("checkpoint has no tensor {0}")]
    MissingTensor(String),
    /// The checkpoint holds decoder tensors this layout does not use.
    #[error("checkpoint has unexpected decoder tensors: {0:?}")]
    UnexpectedTensors(Vec<String>),
    /// A checkpoint stores tensors in both the Hugging Face and the mlx-lm
    /// layout.
    #[error("checkpoint mixes Hugging Face and mlx-lm tensor names")]
    MixedLayouts,
    /// A module is stored quantized but the configuration declares no
    /// quantization for it.
    #[error("{0} is stored quantized, but config.json declares no quantization for it")]
    UndeclaredQuantization(String),
    /// A quantized module's scales or biases are not floating point.
    #[error("{name} is stored as {dtype}, not a floating-point type")]
    QuantizationParameterDtype {
        /// The scales or biases tensor.
        name: String,
        /// Its stored dtype.
        dtype: String,
    },
    /// An f32 load would dequantize more than the f32 limit.
    #[error("dequantizing to f32 needs {bytes} bytes, above the {limit}-byte limit for f32 loads")]
    DequantizedTooLarge {
        /// Bytes the dequantized f32 matrices would take.
        bytes: u64,
        /// The limit.
        limit: u64,
    },
    /// A tensor's shape disagrees with the configuration.
    #[error("tensor {name} has shape {actual:?}, expected {expected:?}")]
    TensorShape {
        /// Tensor name without the text-model prefix.
        name: String,
        /// Shape implied by the configuration.
        expected: Vec<i32>,
        /// Stored shape.
        actual: Vec<i32>,
    },
    /// No input tokens were supplied.
    #[error("input is empty")]
    EmptyInput,
    /// Prefill was requested on a sequence that already holds tokens.
    #[error("prefill requires an empty sequence, which holds {0} tokens")]
    NotEmpty(usize),
    /// A token ID is outside the vocabulary.
    #[error("token {token} is outside the vocabulary of {vocab_size}")]
    InvalidTokenId {
        /// The rejected ID.
        token: i32,
        /// Vocabulary size.
        vocab_size: usize,
    },
    /// The sequence would exceed the context ceiling.
    #[error("sequence of {requested} tokens exceeds the maximum of {maximum}")]
    ContextLimit {
        /// Total tokens after this call.
        requested: usize,
        /// Effective ceiling.
        maximum: usize,
    },
    /// A logit was NaN or infinite, so no greedy token is defined.
    #[error("logits contain NaN or infinite values")]
    NonFiniteLogits,
    /// A verify asked for more tokens than its rollback stash holds.
    #[error("verify of {requested} tokens exceeds the maximum of {maximum}")]
    VerifyLength {
        /// Tokens in the verify.
        requested: usize,
        /// [`Qwen35Executor::max_verify_tokens`].
        maximum: usize,
    },
    /// A [`Qwen35Taps::Layers`] list was empty (`layer` is `None`) or named
    /// a layer past the last.
    #[error("tap layer {layer:?} is not one of the model's {layers} layers")]
    TapLayer {
        /// The offending index; `None` for an empty list.
        layer: Option<usize>,
        /// Decoder layers in the model.
        layers: usize,
    },
    /// A truncation was requested with no verify to roll back.
    #[error("no verify is pending, so the sequence cannot be truncated")]
    NoPendingVerify,
    /// A truncation length lies outside the pending verify.
    #[error(
        "cannot truncate to {requested} tokens; the pending verify allows {minimum} to {maximum}"
    )]
    TruncateRange {
        /// Requested sequence length.
        requested: usize,
        /// Length before the verify.
        minimum: usize,
        /// Length after the verify.
        maximum: usize,
    },
    /// The checkpoint's multi-token-prediction head is a layout this loader
    /// does not implement.
    #[error("unsupported multi-token-prediction head: {0}")]
    MtpUnsupported(&'static str),
    /// A layer's stored state does not match its kind.
    #[error("layer state does not match its layer kind")]
    StateKind,
    /// A snapshot belongs to a different loaded checkpoint instance.
    #[error("snapshot belongs to another weight load")]
    SnapshotMismatch,
    /// A size does not fit MLX's 32-bit shape arithmetic.
    #[error("shape overflows")]
    ShapeOverflow,
    /// The request needs a capability this loader does not implement.
    #[error("unsupported: {0}")]
    Unsupported(&'static str),
    /// The expert budget cannot hold one layer's experts, which one prefill
    /// call may route to.
    #[error("expert budget of {budget} bytes is below one layer's experts, {needed} bytes")]
    PagerBudget {
        /// The requested budget.
        budget: u64,
        /// One layer's routed experts.
        needed: u64,
    },
    /// The expert cache refused to hold an expert a call needs.
    #[error("expert pager: {0}")]
    Pager(String),
    /// MLX failed to construct or evaluate a graph.
    #[error("MLX evaluation failed: {0}")]
    Mlx(#[from] mlx_rs::error::Exception),
    /// A GGUF file could not be read or is invalid.
    #[error(transparent)]
    Checkpoint(#[from] CheckpointError),
    /// A GGUF tensor uses an encoding this loader does not decode.
    #[error("tensor {tensor} is stored as {encoding:?}, which this loader does not decode")]
    Encoding {
        /// The stored tensor's name.
        tensor: String,
        /// Its encoding.
        encoding: TensorEncoding,
    },
    /// A GGUF tensor's payload could not be decoded.
    #[error(transparent)]
    Decode(#[from] GgufDecodeError),
}

#[cfg(test)]
mod tests {
    use mlx_rs::Array;

    use super::{Experts, Matrix, MoeTensors, is_zero_centered_norm, moe_block, mtp_shapes};
    use crate::Qwen35Config;

    /// MLX's rope with the `YaRN` periods and feature scale rotates a head the
    /// way a float64 host rotation by Transformers' recipe does: rotary
    /// features scaled by the attention factor and rotated by
    /// `position * inverse_frequency`, the remaining features untouched.
    /// Tolerances follow the f32 angle error measured for this rotary shape
    /// (2.3e-2 at 1M, 7.5e-3 at 262K against float64).
    #[test]
    fn yarn_rope_matches_a_host_rotation() {
        let config = crate::Qwen35Config::parse(include_str!(
            "../../../../fixtures/qwen3.5-0.8b/config.json"
        ))
        .expect("fixture config")
        .with_rope_scaling(crate::RopeScalingRequest::Yarn { factor: 4 })
        .expect("qualified");
        let yarn = super::yarn_arrays(&config, mlx_rs::Dtype::Float32)
            .expect("arrays")
            .expect("scaling in force");
        let scaling = config.rope_scaling().expect("scaling");
        let inverse = scaling.inverse_frequencies(config.rotary_dim(), config.rope_theta);
        let factor = scaling.attention_factor();
        let head = config.head_dim;
        let half = config.rotary_dim() / 2;
        let input: Vec<f32> = (0..head)
            .map(|feature| {
                let mixed = (feature * 2_654_435_761) % 2_001;
                f32::from(u16::try_from(mixed).expect("bounded")) / 1_000.0 - 1.0
            })
            .collect();
        for (position, tolerance) in [(0_i32, 1e-6), (262_143, 1e-2), (900_000, 3e-2)] {
            let x = Array::from_slice(&input, &[1, 1, 1, i32::try_from(head).expect("dim")]);
            let rotated = super::rope(&x, &config, Some(&yarn), position).expect("rope");
            rotated.eval().expect("evaluates");
            let rotated = rotated.as_slice::<f32>();
            for feature in 0..head {
                let expected = if feature < 2 * half {
                    let pair = feature % half;
                    let angle = f64::from(position) * f64::from(inverse[pair]);
                    let (sin, cos) = angle.sin_cos();
                    let (first, second) = (f64::from(input[pair]), f64::from(input[pair + half]));
                    factor
                        * if feature < half {
                            first * cos - second * sin
                        } else {
                            second * cos + first * sin
                        }
                } else {
                    f64::from(input[feature])
                };
                let error = (f64::from(rotated[feature]) - expected).abs();
                assert!(
                    error <= tolerance,
                    "position {position}, feature {feature}: {error} over {tolerance}"
                );
            }
        }
    }

    #[test]
    fn snapshots_are_bound_to_one_weight_load() {
        let weights = || super::Qwen35Weights {
            identity: std::sync::Arc::new(()),
            config: crate::Qwen35Config::parse(include_str!(
                "../../../../fixtures/qwen3.5-0.8b/config.json"
            ))
            .expect("fixture config"),
            tensors: std::collections::HashMap::new(),
            quantized: std::collections::HashMap::new(),
            precision: super::Qwen35Precision::Float32,
            compute: mlx_rs::Dtype::Float32,
            yarn: None,
            numerics: super::Numerics::Transformers,
            paged: None,
        };
        let source = weights();
        let other = weights();
        let snapshot = super::Qwen35Snapshot {
            identity: std::sync::Arc::clone(&source.identity),
            layers: source.config.layers.iter().map(|_| None).collect(),
            tokens: 0,
        };
        assert!(source.executor_from(&snapshot).is_ok());
        assert!(matches!(
            other.executor_from(&snapshot),
            Err(super::Qwen35Error::SnapshotMismatch)
        ));
    }

    /// Deterministic values in [-1, 1) (xorshift64), so failures reproduce.
    pub(super) fn values(seed: u64, count: usize) -> Vec<f32> {
        let mut state = seed.max(1);
        (0..count)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                #[allow(
                    clippy::cast_precision_loss,
                    reason = "24 high bits fit an f32 mantissa exactly"
                )]
                let unit = (state >> 40) as f32 / (1_u64 << 24) as f32;
                2.0 * unit - 1.0
            })
            .collect()
    }

    struct Block {
        tokens: usize,
        experts: usize,
        hidden: usize,
        width: usize,
        x: Vec<f32>,
        router: Vec<f32>,
        gate_up: Vec<f32>,
        down: Vec<f32>,
        shared_gate: Vec<f32>,
        shared_up: Vec<f32>,
        shared_down: Vec<f32>,
        shared_scale: Vec<f32>,
    }

    impl Block {
        /// Inputs in [-1, 1); each weight matrix scaled by `1 / sqrt(fan_in)`,
        /// as real initializations are, so activations and outputs stay O(1).
        fn random(tokens: usize, experts: usize, hidden: usize, width: usize) -> Self {
            #[allow(clippy::cast_precision_loss, reason = "small test widths")]
            let scaled = |seed, count, fan_in: usize| -> Vec<f32> {
                let scale = (fan_in as f32).sqrt().recip();
                values(seed, count).iter().map(|v| v * scale).collect()
            };
            Self {
                tokens,
                experts,
                hidden,
                width,
                x: values(1, tokens * hidden),
                router: scaled(2, experts * hidden, hidden),
                gate_up: scaled(3, experts * 2 * width * hidden, hidden),
                down: scaled(4, experts * hidden * width, width),
                shared_gate: scaled(5, width * hidden, hidden),
                shared_up: scaled(6, width * hidden, hidden),
                shared_down: scaled(7, hidden * width, width),
                shared_scale: scaled(8, hidden, hidden),
            }
        }

        /// Makes `copy` an exact duplicate of `expert`, router row included,
        /// so the two tie and either choice gives the same output.
        fn duplicate(mut self, expert: usize, copy: usize) -> Self {
            let (hidden, width) = (self.hidden, self.width);
            let router = hidden;
            self.router
                .copy_within(expert * router..(expert + 1) * router, copy * router);
            let gate_up = 2 * width * hidden;
            self.gate_up
                .copy_within(expert * gate_up..(expert + 1) * gate_up, copy * gate_up);
            let down = hidden * width;
            self.down
                .copy_within(expert * down..(expert + 1) * down, copy * down);
            self
        }

        fn native(&self, top_k: usize) -> Vec<f32> {
            let shape = |dims: &[usize]| -> Vec<i32> {
                dims.iter()
                    .map(|&dim| i32::try_from(dim).expect("small"))
                    .collect()
            };
            let array = |values: &[f32], dims: &[usize]| Array::from_slice(values, &shape(dims));
            let (tokens, experts, hidden, width) =
                (self.tokens, self.experts, self.hidden, self.width);
            let (router, gate_up, down) = (
                array(&self.router, &[experts, hidden]),
                array(&self.gate_up, &[experts, 2 * width, hidden]),
                array(&self.down, &[experts, hidden, width]),
            );
            let (shared_gate, shared_up, shared_down, shared_scale) = (
                array(&self.shared_gate, &[width, hidden]),
                array(&self.shared_up, &[width, hidden]),
                array(&self.shared_down, &[hidden, width]),
                array(&self.shared_scale, &[1, hidden]),
            );
            let block = MoeTensors {
                router: Matrix::Dense(&router),
                experts: Experts::Fused {
                    gate_up: &gate_up,
                    down: &down,
                },
                shared_gate: Matrix::Dense(&shared_gate),
                shared_up: Matrix::Dense(&shared_up),
                shared_down: Matrix::Dense(&shared_down),
                shared_scale: Matrix::Dense(&shared_scale),
            };
            let out =
                moe_block(&array(&self.x, &[tokens, hidden]), &block, top_k).expect("moe block");
            out.eval().expect("eval");
            assert_eq!(out.shape(), shape(&[tokens, hidden]).as_slice());
            out.as_slice::<f32>().to_vec()
        }

        /// Softmax over every expert, the `top_k` largest (lower index first
        /// on ties) renormalized, each expert's `SwiGLU`, plus the shared
        /// `SwiGLU` scaled by `sigmoid(shared_scale . input)`, in f64.
        fn host(&self, top_k: usize) -> Vec<f32> {
            let (hidden, width) = (self.hidden, self.width);
            let mut out = Vec::with_capacity(self.tokens * hidden);
            for token in 0..self.tokens {
                let input: Vec<f64> = self.x[token * hidden..(token + 1) * hidden]
                    .iter()
                    .map(|value| f64::from(*value))
                    .collect();
                let dot = |row: &[f32], with: &[f64]| -> f64 {
                    row.iter().zip(with).map(|(a, b)| f64::from(*a) * b).sum()
                };
                // `SwiGLU` of `input` through `width` gate rows and `width` up rows.
                let swiglu = |gate: &[f32], up: &[f32]| -> Vec<f64> {
                    (0..width)
                        .map(|row| {
                            let rows = row * hidden..(row + 1) * hidden;
                            let g = dot(&gate[rows.clone()], &input);
                            g / (1.0 + (-g).exp()) * dot(&up[rows], &input)
                        })
                        .collect()
                };
                let project = |down: &[f32], activation: &[f64], scale: f64, into: &mut [f64]| {
                    for (row, value) in into.iter_mut().enumerate() {
                        *value += scale * dot(&down[row * width..(row + 1) * width], activation);
                    }
                };
                let exps: Vec<f64> = {
                    let logits: Vec<f64> = (0..self.experts)
                        .map(|expert| {
                            dot(&self.router[expert * hidden..(expert + 1) * hidden], &input)
                        })
                        .collect();
                    let max = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                    logits.iter().map(|logit| (logit - max).exp()).collect()
                };
                let mut order: Vec<usize> = (0..self.experts).collect();
                order.sort_by(|a, b| exps[*b].total_cmp(&exps[*a]).then(a.cmp(b)));
                let chosen = &order[..top_k];
                let mass: f64 = chosen.iter().map(|expert| exps[*expert]).sum();
                let mut output = vec![0.0_f64; hidden];
                for &expert in chosen {
                    let block = 2 * width * hidden;
                    let rows = &self.gate_up[expert * block..(expert + 1) * block];
                    let (gate, up) = rows.split_at(width * hidden);
                    let activation = swiglu(gate, up);
                    let down = &self.down[expert * hidden * width..(expert + 1) * hidden * width];
                    project(down, &activation, exps[expert] / mass, &mut output);
                }
                let shared = swiglu(&self.shared_gate, &self.shared_up);
                let scale = 1.0 / (1.0 + (-dot(&self.shared_scale, &input)).exp());
                project(&self.shared_down, &shared, scale, &mut output);
                #[allow(clippy::cast_possible_truncation, reason = "f32 comparison")]
                out.extend(output.iter().map(|value| *value as f32));
            }
            out
        }
    }

    fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len());
        a.iter()
            .zip(b)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f32::max)
    }

    /// f32 against the f64 host loop; outputs are O(1) (checked) and sums
    /// run over at most 64 terms, so 1e-5 leaves room only for f32 rounding.
    const TOLERANCE: f32 = 1e-5;

    #[test]
    fn moe_block_matches_a_host_loop() {
        let cases = [
            ("top 4 of 16", Block::random(5, 16, 64, 32), 4),
            // Experts 2 and 3 tie wherever either is chosen.
            (
                "tied duplicate",
                Block::random(5, 16, 64, 32).duplicate(2, 3),
                4,
            ),
            ("every expert", Block::random(3, 6, 64, 32), 6),
            ("one expert", Block::random(3, 6, 64, 32), 1),
        ];
        for (name, block, top_k) in cases {
            let host = block.host(top_k);
            let diff = max_abs_diff(&block.native(top_k), &host);
            let largest = host.iter().fold(0.0_f32, |max, value| max.max(value.abs()));
            eprintln!("{name}: max |diff| {diff:.3e}, largest |output| {largest:.3}");
            assert!(largest < 8.0, "{name}: outputs are not O(1): {largest}");
            assert!(diff <= TOLERANCE, "{name}: max |diff| {diff}");
        }
    }

    /// The text fields of Qwen/Qwen3.5-9B@c2022362 (32 layers, every fourth
    /// full attention), whose checkpoint carries a one-layer dense head.
    fn qwen35_9b() -> Qwen35Config {
        let layer_types = (0..32)
            .map(|layer| {
                if layer % 4 == 3 {
                    "\"full_attention\""
                } else {
                    "\"linear_attention\""
                }
            })
            .collect::<Vec<_>>()
            .join(",");
        Qwen35Config::parse(&format!(
            r#"{{"model_type": "qwen3_5", "tie_word_embeddings": false, "text_config": {{
                "model_type": "qwen3_5_text", "attention_bias": false, "attn_output_gate": true,
                "head_dim": 256, "hidden_act": "silu", "hidden_size": 4096,
                "intermediate_size": 12288, "layer_types": [{layer_types}],
                "linear_conv_kernel_dim": 4, "linear_key_head_dim": 128,
                "linear_num_key_heads": 16, "linear_num_value_heads": 32,
                "linear_value_head_dim": 128, "mamba_ssm_dtype": "float32",
                "max_position_embeddings": 262144, "mlp_only_layers": [],
                "mtp_num_hidden_layers": 1, "mtp_use_dedicated_embeddings": false,
                "num_attention_heads": 16, "num_hidden_layers": 32,
                "num_key_value_heads": 4, "rms_norm_eps": 1e-06, "vocab_size": 248320,
                "rope_parameters": {{"mrope_interleaved": true, "mrope_section": [11, 11, 10],
                    "rope_type": "default", "rope_theta": 10000000,
                    "partial_rotary_factor": 0.25}}}}}}"#
        ))
        .expect("Qwen3.5-9B config")
    }

    /// Every `mtp.*` tensor in Qwen/Qwen3.5-9B@c2022362's
    /// model.safetensors.index.json, with its stored shape.
    const QWEN35_9B_MTP: [(&str, &[i32]); 15] = [
        ("mtp.fc.weight", &[4096, 8192]),
        ("mtp.layers.0.input_layernorm.weight", &[4096]),
        ("mtp.layers.0.mlp.down_proj.weight", &[4096, 12288]),
        ("mtp.layers.0.mlp.gate_proj.weight", &[12288, 4096]),
        ("mtp.layers.0.mlp.up_proj.weight", &[12288, 4096]),
        ("mtp.layers.0.post_attention_layernorm.weight", &[4096]),
        ("mtp.layers.0.self_attn.k_norm.weight", &[256]),
        ("mtp.layers.0.self_attn.k_proj.weight", &[1024, 4096]),
        ("mtp.layers.0.self_attn.o_proj.weight", &[4096, 4096]),
        ("mtp.layers.0.self_attn.q_norm.weight", &[256]),
        ("mtp.layers.0.self_attn.q_proj.weight", &[8192, 4096]),
        ("mtp.layers.0.self_attn.v_proj.weight", &[1024, 4096]),
        ("mtp.norm.weight", &[4096]),
        ("mtp.pre_fc_norm_embedding.weight", &[4096]),
        ("mtp.pre_fc_norm_hidden.weight", &[4096]),
    ];

    /// The head's expected tensors are exactly the published ones, and every
    /// norm among them (and only a norm) gets the `1 + weight` shift. Missing
    /// the shift on `pre_fc_norm_*` drives draft acceptance to zero.
    #[test]
    fn mtp_head_matches_the_published_tensors_and_shifts_every_norm() {
        let expected = mtp_shapes(&qwen35_9b()).expect("dense head");
        assert_eq!(expected.len(), QWEN35_9B_MTP.len());
        for (name, shape) in QWEN35_9B_MTP {
            assert_eq!(expected.get(name).map(Vec::as_slice), Some(shape), "{name}");
            assert_eq!(
                is_zero_centered_norm(name),
                name.contains("norm"),
                "{name}: shifted iff a norm"
            );
        }
    }
}

/// This reports arithmetic localization, not a relaxed model qualification gate.
#[cfg(test)]
mod gguf_probe {
    use super::*;

    struct Snapshot {
        stage: String,
        shape: Vec<i32>,
        dtype: String,
        values: Vec<f32>,
    }

    fn snapshot(stage: &str, array: &Array) -> Result<Snapshot, Qwen35Error> {
        let shape = array.shape().to_vec();
        let last = shape[1] - 1;
        let values = array.index((.., last..last + 1, ..)).as_type::<f32>()?;
        values.eval()?;
        Ok(Snapshot {
            stage: stage.to_owned(),
            shape,
            dtype: format!("{:?}", array.dtype()),
            values: values.as_slice::<f32>().to_vec(),
        })
    }

    /// The oracle dialogue's token IDs, or `None` when
    /// `METALLIX_QWEN35_GGUF_ORACLE` is unset.
    fn dialogue_ids() -> Option<Vec<i32>> {
        let oracle = std::env::var_os("METALLIX_QWEN35_GGUF_ORACLE")?;
        let file =
            GgufFile::open(PathBuf::from(oracle).join("oracle-dialogue.gguf")).expect("oracle");
        let ids: Vec<i32> = file
            .read("tokens", 1 << 20)
            .expect("token payload")
            .chunks_exact(4)
            .map(|b| i32::from_le_bytes(b.try_into().expect("token")))
            .collect();
        // The failing receipt names zero-based dialogue position 129.
        assert!(ids.len() >= 130, "dialogue must include position 129");
        Some(ids)
    }

    #[test]
    #[ignore = "requires the pinned local GGUF, dialogue oracle and an isolated device slot"]
    fn gguf_dialogue_layer_probe() {
        let (Some(model), Some(ids)) = (std::env::var_os("METALLIX_QWEN35_GGUF"), dialogue_ids())
        else {
            eprintln!("skipping: METALLIX_QWEN35_GGUF or METALLIX_QWEN35_GGUF_ORACLE is not set");
            return;
        };
        let precision = match std::env::var("METALLIX_GGUF_PROBE_PRECISION").as_deref() {
            Ok("f32") => Qwen35Precision::Float32,
            Ok("checkpoint") | Err(_) => Qwen35Precision::Checkpoint,
            _ => panic!("precision must be checkpoint or f32"),
        };
        let cap: usize = std::env::var("METALLIX_GGUF_GATE_GPU_GIB")
            .unwrap_or_else(|_| "24".into())
            .parse()
            .expect("integer cap");
        assert!((1..=64).contains(&cap));
        let check_cap = || {
            let bytes = mlx_rs::memory::active_memory().expect("active")
                + mlx_rs::memory::cache_memory().expect("cache");
            assert!(bytes <= cap * (1 << 30), "probe exceeds MLX memory cap");
        };
        let weights = Qwen35Weights::load_gguf(PathBuf::from(model), precision).expect("weights");
        check_cap();
        for position in [129, ids.len() - 1] {
            let ids = &ids[..=position];
            let mut arms = Vec::new();
            for sequential in [true, false] {
                let mut executor = weights.executor();
                let mut snapshots = Vec::new();
                let chunks: Vec<&[i32]> = if sequential {
                    ids.chunks(1).collect()
                } else {
                    ids.chunks(PREFILL_CHUNK_TOKENS).collect()
                };
                let count = chunks.len();
                for (index, chunk) in chunks.into_iter().enumerate() {
                    let final_chunk = index + 1 == count;
                    let hidden = executor
                        .forward_chunk_observed(chunk, &mut |stage, tensor| {
                            if final_chunk {
                                snapshots.push(snapshot(stage, tensor)?);
                            }
                            Ok(())
                        })
                        .expect("forward");
                    // Materialize carried state just as extend_last_logits does.
                    let state: Vec<&Array> = executor
                        .layers
                        .iter()
                        .flatten()
                        .flat_map(|layer| match layer {
                            LayerState::Linear { conv, recurrent } => [conv, recurrent],
                            LayerState::Full { keys, values } => [keys, values],
                        })
                        .collect();
                    mlx_rs::transforms::eval(state).expect("state");
                    if final_chunk {
                        let last = hidden.shape()[1] - 1;
                        let normalized = rms_norm(
                            &hidden.index((.., last..last + 1, ..)),
                            weights.tensor("norm.weight").expect("norm"),
                            weights.config.rms_norm_eps,
                        )
                        .expect("output norm");
                        snapshots.push(snapshot("output_norm", &normalized).expect("snapshot"));
                        let logits = executor.last_logits(&hidden).expect("logits");
                        logits.eval().expect("logits evaluation");
                        let logits = logits.as_slice::<f32>().to_vec();
                        snapshots.push(Snapshot {
                            stage: "logits".into(),
                            shape: vec![1, 1, dim(logits.len()).expect("vocab")],
                            dtype: "Float32".into(),
                            values: logits,
                        });
                    }
                    check_cap();
                }
                arms.push(snapshots);
            }
            assert_eq!(arms[0].len(), arms[1].len());
            for (sequential, prefill) in arms[0].iter().zip(&arms[1]) {
                assert_eq!(sequential.stage, prefill.stage);
                assert_eq!(sequential.values.len(), prefill.values.len());
                let mut max_abs = 0.0_f64;
                let mut squared = 0.0_f64;
                for (&a, &b) in sequential.values.iter().zip(&prefill.values) {
                    assert!(
                        a.is_finite() && b.is_finite(),
                        "non-finite diagnostic state"
                    );
                    let delta = f64::from(a) - f64::from(b);
                    max_abs = max_abs.max(delta.abs());
                    squared += delta * delta;
                }
                #[allow(clippy::cast_precision_loss, reason = "bounded tensor element count")]
                let rms = (squared / sequential.values.len() as f64).sqrt();
                eprintln!(
                    "GGUF-LAYER {}",
                    serde_json::json!({"stage": sequential.stage, "position": position, "precision": format!("{precision:?}"), "sequential_shape": sequential.shape, "prefill_shape": prefill.shape, "sequential_dtype": sequential.dtype, "prefill_dtype": prefill.dtype, "elements": sequential.values.len(), "max_abs": max_abs, "rms": rms})
                );
            }
        }
    }
}
