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
    collections::HashMap,
    fs,
    path::{Path, PathBuf},
    sync::Arc,
};

use blockfloat::gguf::{AffineParameters, GgufDecodeError, GgufEncoding, decode, repack_affine};
use checkpoint::{CheckpointError, TensorInfo, TensorSource, gguf::GgufFile};
use mlx_rs::{
    Array, Dtype, StreamOrDevice, fast,
    ops::{
        self,
        indexing::{IndexMutOp, IndexOp},
    },
};
use serde::Deserialize;
use thiserror::Error;

use crate::{
    Qwen35Config, Qwen35ConfigError, Qwen35LayerKind, Qwen35Mlp,
    gguf::{ValueHeadAxis, canonical_name, permute_spans, tiled_positions, value_head_axis},
};

mod delta;

/// Tokens per prefill graph. This bounds graph size even though the delta
/// rule runs in 64-token chunks; longer prompts use successive graphs.
pub const PREFILL_CHUNK_TOKENS: usize = 128;

const TEXT_PREFIX: &str = "model.language_model.";
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

/// Largest single GGUF tensor this loader reads (a 248k x 8192 F32 matrix
/// is 8 GiB).
const MAX_GGUF_TENSOR_BYTES: u64 = 8 << 30;

impl Qwen35Weights {
    /// Loads and validates the text-decoder tensors of a local checkpoint.
    ///
    /// Vision-tower and multi-token-prediction tensors are dropped unread.
    /// Zero-centered RMS norm weights become `1 + weight` in f32 before any
    /// precision conversion, and convolution kernels are stored `[K, C]`.
    ///
    /// # Errors
    ///
    /// Returns [`Qwen35Error`] for an unsupported configuration, an unsafe or
    /// missing shard, a missing tensor, or a tensor of the wrong shape.
    pub fn load(
        model_dir: impl AsRef<Path>,
        precision: Qwen35Precision,
    ) -> Result<Self, Qwen35Error> {
        let model_dir = model_dir.as_ref();
        let config = Qwen35Config::parse(&read_file(&model_dir.join("config.json"))?)?;
        let mut tensors = HashMap::new();
        for shard in shard_paths(model_dir)? {
            for (name, tensor) in Array::load_safetensors_device(&shard, StreamOrDevice::cpu())? {
                if let Some(name) = name.strip_prefix(TEXT_PREFIX) {
                    tensors.insert(name.to_owned(), tensor);
                } else if name == "lm_head.weight" && !config.tie_word_embeddings() {
                    tensors.insert(name, tensor);
                }
            }
        }
        let expected = expected_shapes(&config)?;
        for (name, shape) in &expected {
            let tensor = tensors
                .get(name)
                .ok_or_else(|| Qwen35Error::MissingTensor(name.clone()))?;
            if tensor.shape() != shape.as_slice() {
                return Err(Qwen35Error::TensorShape {
                    name: name.clone(),
                    expected: shape.clone(),
                    actual: tensor.shape().to_vec(),
                });
            }
        }
        if tensors.len() != expected.len() {
            let mut unexpected: Vec<_> = tensors
                .keys()
                .filter(|name| !expected.contains_key(*name))
                .cloned()
                .collect();
            unexpected.sort();
            return Err(Qwen35Error::UnexpectedTensors(unexpected));
        }

        let gpu = StreamOrDevice::gpu();
        let compute = tensors["embed_tokens.weight"].dtype();
        for (name, tensor) in &mut tensors {
            let converted = if is_zero_centered_norm(name) {
                let shifted = tensor
                    .as_type_device::<f32>(&gpu)?
                    .add_device(Array::from_f32(1.0), &gpu)?;
                match precision {
                    Qwen35Precision::Float32 => shifted,
                    Qwen35Precision::Checkpoint => shifted.as_dtype_device(compute, &gpu)?,
                }
            } else if name.ends_with("conv1d.weight") {
                let shape = tensor.shape().to_vec();
                let transposed = tensor
                    .reshape_device(&[shape[0], shape[2]], &gpu)?
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
        Ok(Self {
            identity: Arc::new(()),
            config,
            tensors,
            quantized: HashMap::new(),
            precision,
            compute: match precision {
                Qwen35Precision::Checkpoint => compute,
                Qwen35Precision::Float32 => Dtype::Float32,
            },
        })
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
        let expected = expected_shapes(&config)?;
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
        let encoding = info.encoding();
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
                    encoding: other,
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
        })
    }

    fn tensor(&self, name: &str) -> Result<&Array, Qwen35Error> {
        self.tensors
            .get(name)
            .ok_or_else(|| Qwen35Error::MissingTensor(name.to_owned()))
    }

    /// `input` times the transpose of the matrix `name`, in `input`'s dtype.
    fn project(&self, input: &Array, name: &str) -> Result<Array, Qwen35Error> {
        let gpu = StreamOrDevice::gpu();
        let output = match self.quantized.get(name) {
            Some(weight) => ops::quantized_matmul_device(
                input,
                &weight.codes,
                &weight.scales,
                &weight.biases,
                true,
                weight.group_size,
                weight.bits,
                &gpu,
            )?,
            None => linear(input, self.tensor(name)?)?,
        };
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
    let encoding = info.encoding();
    let shape = info.shape();
    let columns = if shape.len() == 2 { shape[1] } else { 1 };
    let size = |elements: u64| -> Result<usize, Qwen35Error> {
        encoding
            .byte_len(elements)
            .and_then(|bytes| usize::try_from(bytes).ok())
            .ok_or_else(|| Qwen35Error::Encoding {
                tensor: name.to_owned(),
                encoding,
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
        let hidden = self.extend_hidden(input_ids)?;
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
        let hidden = self.extend_hidden(input_ids)?;
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
    /// returns the final chunk's hidden states.
    fn extend_hidden(&mut self, input_ids: &[i32]) -> Result<Array, Qwen35Error> {
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
        let mut chunks = input_ids.chunks(PREFILL_CHUNK_TOKENS).peekable();
        while let Some(chunk) = chunks.next() {
            let hidden = self.forward_chunk(chunk)?;
            if chunks.peek().is_none() {
                return Ok(hidden);
            }
            // Evaluate the carried state so the next chunk's graph starts from
            // materialized arrays instead of growing without bound.
            mlx_rs::transforms::eval(self.state_arrays())?;
        }
        unreachable!("input_ids is nonempty")
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
            let mixed = match kind {
                Qwen35LayerKind::LinearAttention => {
                    gated_delta_net(weights, &format!("{base}.linear_attn"), &normed, state)?
                }
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
) -> Result<Array, Qwen35Error> {
    let config = &weights.config;
    let gpu = StreamOrDevice::gpu();
    let seq = input.shape()[1];
    let key_heads = dim(config.linear_key_heads)?;
    let value_heads = dim(config.linear_value_heads)?;
    let key_dim = dim(config.linear_key_head_dim)?;
    let value_dim = dim(config.linear_value_head_dim)?;
    let compute = input.dtype();
    let project = |name: &str| weights.project(input, &format!("{base}.{name}.weight"));

    let (conv_state, recurrent) = match state.take() {
        Some(LayerState::Linear { conv, recurrent }) => (conv, recurrent),
        Some(LayerState::Full { .. }) => return Err(Qwen35Error::StateKind),
        None => (
            ops::zeros_dtype_device(
                &[1, dim(config.conv_kernel - 1)?, dim(config.conv_dim())?],
                compute,
                &gpu,
            )?,
            ops::zeros_dtype_device(&[value_heads, key_dim, value_dim], Dtype::Float32, &gpu)?,
        ),
    };
    let (convolved, next_conv) = short_convolution(
        &conv_state,
        &project("in_proj_qkv")?,
        weights.tensor(&format!("{base}.conv1d.weight"))?,
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
        project("in_proj_b")?
            .reshape_device(&[seq, value_heads], &gpu)?
            .as_type_device::<f32>(&gpu)?,
        &gpu,
    )?;
    let decay_rate = weights.tensor(&format!("{base}.decay_rate"))?;
    let shifted = project("in_proj_a")?
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

    let (output, recurrent) = if seq == 1 {
        delta_rule_by_token(&query, &key, &value, &log_decay, &beta, recurrent)?
    } else {
        delta::chunked(&query, &key, &value, &log_decay, &beta, recurrent)?
    };
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

/// Depthwise causal convolution plus `silu` over `[1, seq, channels]`, with
/// `window` (`[1, K - 1, channels]`) holding the preceding inputs and `kernel`
/// stored `[K, channels]`. Output `t` sums tap `j` times input `t + j` of the
/// window-prefixed sequence. Returns the output and the next window.
fn short_convolution(
    window: &Array,
    input: &Array,
    kernel: &Array,
) -> Result<(Array, Array), Qwen35Error> {
    let gpu = StreamOrDevice::gpu();
    let seq = input.shape()[1];
    let history = window.shape()[1];
    let padded = ops::concatenate_axis_device(&[window, input], 1, &gpu)?;
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
        Ok(fast::rope_device(
            &normed,
            dim(config.rotary_dim)?,
            false,
            Some(config.rope_theta),
            1.0,
            offset,
            Option::<&Array>::None,
            &gpu,
        )?)
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
    let tensor = |name: &str| weights.tensor(&format!("{base}.mlp.{name}"));
    match weights.config.mlp {
        Qwen35Mlp::Dense { .. } => {
            let gpu = StreamOrDevice::gpu();
            let project = |name: &str, x: &Array| weights.project(x, &format!("{base}.mlp.{name}"));
            let hidden = silu(&project("gate_proj.weight", input)?)?
                .multiply_device(project("up_proj.weight", input)?, &gpu)?;
            project("down_proj.weight", &hidden)
        }
        Qwen35Mlp::Experts { top_k, .. } => {
            let gpu = StreamOrDevice::gpu();
            let shape = input.shape().to_vec();
            let tokens = input.reshape_device(&[-1, shape[shape.len() - 1]], &gpu)?;
            let block = MoeTensors {
                router: tensor("gate.weight")?,
                gate_up: tensor("experts.gate_up_proj")?,
                down: tensor("experts.down_proj")?,
                shared_gate: tensor("shared_expert.gate_proj.weight")?,
                shared_up: tensor("shared_expert.up_proj.weight")?,
                shared_down: tensor("shared_expert.down_proj.weight")?,
                shared_scale: tensor("shared_expert_gate.weight")?,
            };
            Ok(moe_block(&tokens, &block, top_k)?.reshape_device(&shape, &gpu)?)
        }
    }
}

/// One layer's `MoE` weights, as stored.
struct MoeTensors<'a> {
    /// `[E, H]`.
    router: &'a Array,
    /// `[E, 2I, H]`, gate rows then up rows.
    gate_up: &'a Array,
    /// `[E, H, I]`.
    down: &'a Array,
    shared_gate: &'a Array,
    shared_up: &'a Array,
    shared_down: &'a Array,
    /// `[1, H]`; `sigmoid` of its product scales the shared expert.
    shared_scale: &'a Array,
}

/// The routed experts plus the gated shared expert over `tokens` `[n, H]`.
fn moe_block(tokens: &Array, block: &MoeTensors<'_>, top_k: usize) -> Result<Array, Qwen35Error> {
    let gpu = StreamOrDevice::gpu();
    let routed = routed_experts(tokens, block.router, block.gate_up, block.down, top_k)?;
    let shared = swiglu(
        tokens,
        block.shared_gate,
        block.shared_up,
        block.shared_down,
    )?
    .multiply_device(
        ops::sigmoid_device(linear(tokens, block.shared_scale)?, &gpu)?,
        &gpu,
    )?;
    Ok(routed.add_device(&shared, &gpu)?)
}

fn swiglu(input: &Array, gate: &Array, up: &Array, down: &Array) -> Result<Array, Qwen35Error> {
    let gpu = StreamOrDevice::gpu();
    linear(
        &silu(&linear(input, gate)?)?.multiply_device(linear(input, up)?, &gpu)?,
        down,
    )
}

/// The routed half of an `MoE` block over `tokens` `[n, H]`: the router
/// `[E, H]` picks `top_k` experts per token by softmax probability (in f32),
/// renormalized to sum to 1; each expert applies `SwiGLU` with its slice of
/// `gate_up` `[E, 2I, H]` (gate rows, then up rows) and `down` `[E, H, I]`.
/// Returns the probability-weighted sum, `[n, H]`.
fn routed_experts(
    tokens: &Array,
    router: &Array,
    gate_up: &Array,
    down: &Array,
    top_k: usize,
) -> Result<Array, Qwen35Error> {
    let gpu = StreamOrDevice::gpu();
    let compute = tokens.dtype();
    let experts = router.shape()[0];
    let top_k = dim(top_k)?;
    let width = down.shape()[2];
    let probabilities = ops::softmax_axis_device(
        linear(tokens, router)?.as_type_device::<f32>(&gpu)?,
        -1,
        true,
        &gpu,
    )?;
    // After partitioning at `experts - top_k`, the last `top_k` positions
    // hold the largest probabilities, in no particular order.
    let kth = experts - top_k;
    let chosen = ops::argpartition_axis_device(&probabilities, kth, -1, &gpu)?.index((.., kth..));
    let weights = probabilities.take_along_axis_device(&chosen, -1, &gpu)?;
    let weights = weights
        .divide_device(weights.sum_axis_device(-1, true, &gpu)?, &gpu)?
        .as_dtype_device(compute, &gpu)?;

    // [n, 1, 1, H] against [E, H, 2I] gathered by `chosen` [n, k]: [n, k, 1, 2I].
    let rows = tokens.expand_dims_axes_device(&[-2, -3], &gpu)?;
    let projected = ops::gather_mm_device(
        &rows,
        gate_up.swap_axes_device(-1, -2, &gpu)?,
        None,
        &chosen,
        None,
        &gpu,
    )?;
    let halves = ops::split_sections_device(&projected, &[width], -1, &gpu)?;
    let hidden = silu(&halves[0])?.multiply_device(&halves[1], &gpu)?;
    let outputs = ops::gather_mm_device(
        &hidden,
        down.swap_axes_device(-1, -2, &gpu)?,
        None,
        &chosen,
        None,
        &gpu,
    )?
    .squeeze_axes_device(&[-2], &gpu)?;
    Ok(outputs
        .multiply_device(weights.expand_dims_device(-1, &gpu)?, &gpu)?
        .sum_axis_device(1, false, &gpu)?)
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

fn is_zero_centered_norm(name: &str) -> bool {
    name == "norm.weight"
        || [
            ".input_layernorm.weight",
            ".post_attention_layernorm.weight",
            ".q_norm.weight",
            ".k_norm.weight",
        ]
        .iter()
        .any(|suffix| name.ends_with(suffix))
}

/// The stored shapes of one layer's `MoE` tensors under `mlp`, from the
/// expert count and the routed and shared expert widths.
fn expert_shapes(
    put: &mut impl FnMut(String, &[i32]),
    mlp: &str,
    hidden: i32,
    [experts, expert, shared]: [usize; 3],
) -> Result<(), Qwen35Error> {
    let (experts, expert, shared) = (dim(experts)?, dim(expert)?, dim(shared)?);
    put(format!("{mlp}.gate.weight"), &[experts, hidden]);
    put(
        format!("{mlp}.experts.gate_up_proj"),
        &[experts, 2 * expert, hidden],
    );
    put(
        format!("{mlp}.experts.down_proj"),
        &[experts, hidden, expert],
    );
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

/// Every decoder tensor this implementation reads, with its stored shape.
fn expected_shapes(config: &Qwen35Config) -> Result<HashMap<String, Vec<i32>>, Qwen35Error> {
    let hidden = dim(config.hidden_size)?;
    let vocab = dim(config.vocab_size)?;
    let heads = dim(config.attention_heads)?;
    let kv_heads = dim(config.key_value_heads)?;
    let head_dim = dim(config.head_dim)?;
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
                let intermediate = dim(intermediate_size)?;
                put(format!("{mlp}.gate_proj.weight"), &[intermediate, hidden]);
                put(format!("{mlp}.up_proj.weight"), &[intermediate, hidden]);
                put(format!("{mlp}.down_proj.weight"), &[hidden, intermediate]);
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
            )?,
        }
        match kind {
            Qwen35LayerKind::LinearAttention => {
                let attn = format!("{base}.linear_attn");
                put(format!("{attn}.in_proj_qkv.weight"), &[conv, hidden]);
                put(format!("{attn}.in_proj_z.weight"), &[value_width, hidden]);
                put(format!("{attn}.in_proj_b.weight"), &[value_heads, hidden]);
                put(format!("{attn}.in_proj_a.weight"), &[value_heads, hidden]);
                put(format!("{attn}.conv1d.weight"), &[conv, 1, kernel]);
                put(format!("{attn}.A_log"), &[value_heads]);
                put(format!("{attn}.dt_bias"), &[value_heads]);
                put(
                    format!("{attn}.norm.weight"),
                    &[dim(config.linear_value_head_dim)?],
                );
                put(format!("{attn}.out_proj.weight"), &[hidden, value_width]);
            }
            Qwen35LayerKind::FullAttention => {
                let attn = format!("{base}.self_attn");
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
            }
        }
    }
    Ok(shapes)
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
    /// A layer's stored state does not match its kind.
    #[error("layer state does not match its layer kind")]
    StateKind,
    /// A snapshot belongs to a different loaded checkpoint instance.
    #[error("snapshot belongs to another weight load")]
    SnapshotMismatch,
    /// A size does not fit MLX's 32-bit shape arithmetic.
    #[error("shape overflows")]
    ShapeOverflow,
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
        encoding: GgufEncoding,
    },
    /// A GGUF tensor's payload could not be decoded.
    #[error(transparent)]
    Decode(#[from] GgufDecodeError),
}

#[cfg(test)]
mod tests {
    use mlx_rs::Array;

    use super::{MoeTensors, moe_block};

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
    fn values(seed: u64, count: usize) -> Vec<f32> {
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
                router: &router,
                gate_up: &gate_up,
                down: &down,
                shared_gate: &shared_gate,
                shared_up: &shared_up,
                shared_down: &shared_down,
                shared_scale: &shared_scale,
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

    fn dialogue_ids() -> Vec<i32> {
        let oracle =
            std::env::var_os("METALLIX_QWEN35_GGUF_ORACLE").expect("oracle directory required");
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
        ids
    }

    #[test]
    #[ignore = "requires the pinned local GGUF, dialogue oracle and an isolated device slot"]
    fn gguf_dialogue_layer_probe() {
        let model = std::env::var_os("METALLIX_QWEN35_GGUF").expect("GGUF path required");
        let ids = dialogue_ids();
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
