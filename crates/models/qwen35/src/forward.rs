//! Metal execution of the hybrid decoder: weight loading, prefill and cached
//! decode with `GatedDeltaNet` recurrent state alongside full-attention K/V.
//!
//! Every operator runs on the MLX GPU stream; only final logits are read back.
//! The `GatedDeltaNet` recurrence is evaluated token by token (see the crate
//! documentation), so prefill is split into bounded chunks whose graphs are
//! evaluated before the next chunk starts.

#![allow(
    deprecated,
    reason = "mlx-rs 0.32 deprecates the *_device ops; the with_stream migration is a separate change"
)]

use std::{
    collections::HashMap,
    fs,
    path::{Path, PathBuf},
};

use mlx_rs::{
    Array, Dtype, StreamOrDevice, fast,
    ops::{self, indexing::IndexOp},
};
use serde::Deserialize;
use thiserror::Error;

use crate::{Qwen35Config, Qwen35ConfigError, Qwen35LayerKind};

/// Tokens per prefill graph. The recurrence unrolls one step per token, so
/// this bounds graph size; longer prompts are evaluated chunk by chunk.
pub const PREFILL_CHUNK_TOKENS: usize = 128;

/// Context ceiling of this qualification path, before the checkpoint's own
/// `max_position_embeddings`.
pub const MAX_CONTEXT_TOKENS: usize = 16_384;

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
    config: Qwen35Config,
    tensors: HashMap<String, Array>,
    precision: Qwen35Precision,
}

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
        Ok(Self {
            config,
            tensors,
            precision,
        })
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
        self.tensors.values().map(Array::nbytes).sum()
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

    fn tensor(&self, name: &str) -> Result<&Array, Qwen35Error> {
        self.tensors
            .get(name)
            .ok_or_else(|| Qwen35Error::MissingTensor(name.to_owned()))
    }
}

/// Per-layer sequence state.
enum LayerState {
    /// The last `K - 1` convolution inputs `[1, K - 1, conv_dim]` and the f32
    /// recurrent matrices `[Hv, Dk, Dv]`.
    Linear { conv: Array, recurrent: Array },
    /// Rotated keys and values `[1, kv_heads, tokens, head_dim]`.
    Full { keys: Array, values: Array },
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
        let config = &self.weights.config;
        if input_ids.is_empty() {
            return Err(Qwen35Error::EmptyInput);
        }
        let maximum = MAX_CONTEXT_TOKENS.min(config.max_position_embeddings);
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
                return self.read_logits(&hidden);
            }
            // Evaluate the carried state so the next chunk's graph starts from
            // materialized arrays instead of growing without bound.
            let state: Vec<&Array> = self
                .layers
                .iter()
                .flatten()
                .flat_map(|layer| match layer {
                    LayerState::Linear { conv, recurrent } => [conv, recurrent],
                    LayerState::Full { keys, values } => [keys, values],
                })
                .collect();
            mlx_rs::transforms::eval(state)?;
        }
        unreachable!("input_ids is nonempty")
    }

    fn forward_chunk(&mut self, input_ids: &[i32]) -> Result<Array, Qwen35Error> {
        let weights = self.weights;
        let config = &weights.config;
        let gpu = StreamOrDevice::gpu();
        let seq = dim(input_ids.len())?;
        let offset = dim(self.tokens)?;
        let mut hidden = weights
            .tensor("embed_tokens.weight")?
            .take_axis_device(Array::from_slice(input_ids, &[seq]), 0, &gpu)?
            .reshape_device(&[1, seq, dim(config.hidden_size)?], &gpu)?;
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
            let normed = rms_norm(
                &hidden,
                weights.tensor(&format!("{base}.post_attention_layernorm.weight"))?,
                config.rms_norm_eps,
            )?;
            hidden = hidden.add_device(mlp(weights, &base, &normed)?, &gpu)?;
        }
        self.tokens += input_ids.len();
        Ok(hidden)
    }

    fn read_logits(&self, hidden: &Array) -> Result<Vec<f32>, Qwen35Error> {
        let weights = self.weights;
        let config = &weights.config;
        let gpu = StreamOrDevice::gpu();
        let last = hidden.shape()[1] - 1;
        let last = hidden.index((.., last..last + 1, ..));
        let normed = rms_norm(&last, weights.tensor("norm.weight")?, config.rms_norm_eps)?;
        let head = if config.tie_word_embeddings {
            weights.tensor("embed_tokens.weight")?
        } else {
            weights.tensor("lm_head.weight")?
        };
        let logits = linear(&normed, head)?
            .reshape_device(&[dim(config.vocab_size)?], &gpu)?
            .as_type_device::<f32>(&gpu)?;
        logits.eval()?;
        Ok(logits.as_slice::<f32>().to_vec())
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
    let project = |name: &str| linear(input, weights.tensor(&format!("{base}.{name}.weight"))?);

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
    let decay_rate = weights
        .tensor(&format!("{base}.A_log"))?
        .as_type_device::<f32>(&gpu)?
        .exp_device(&gpu)?;
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
    let decay = softplus
        .multiply_device(&decay_rate, &gpu)?
        .negative_device(&gpu)?
        .exp_device(&gpu)?;

    let (output, recurrent) = delta_rule(&query, &key, &value, &decay, &beta, recurrent)?;
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
    linear(&gated, weights.tensor(&format!("{base}.out_proj.weight"))?)
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

/// The gated delta rule, one token at a time. `query`, `key` are
/// `[seq, Hv, Dk]`, `value` is `[seq, Hv, Dv]`, `decay` (already `exp(g)`) and
/// `beta` are `[seq, Hv]`, all f32, and `state` is `[Hv, Dk, Dv]`. Returns the
/// outputs `[seq, Hv, Dv]` and the final state.
fn delta_rule(
    query: &Array,
    key: &Array,
    value: &Array,
    decay: &Array,
    beta: &Array,
    mut state: Array,
) -> Result<(Array, Array), Qwen35Error> {
    let gpu = StreamOrDevice::gpu();
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
        state = state.multiply_device(row(decay, t)?, &gpu)?;
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

    let projected = linear(input, weights.tensor(&format!("{base}.q_proj.weight"))?)?
        .reshape_device(&[1, seq, heads, 2 * head_dim], &gpu)?;
    let halves = ops::split_sections_device(&projected, &[head_dim], 3, &gpu)?;
    let gate = halves[1].reshape_device(&[1, seq, heads * head_dim], &gpu)?;
    let key = linear(input, weights.tensor(&format!("{base}.k_proj.weight"))?)?
        .reshape_device(&[1, seq, kv_heads, head_dim], &gpu)?;
    let value = linear(input, weights.tensor(&format!("{base}.v_proj.weight"))?)?
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

    let (keys, values) = match state.take() {
        Some(LayerState::Full { keys, values }) => (
            ops::concatenate_axis_device(&[&keys, &key], 2, &gpu)?,
            ops::concatenate_axis_device(&[&values, &value], 2, &gpu)?,
        ),
        Some(LayerState::Linear { .. }) => return Err(Qwen35Error::StateKind),
        None => (key, value),
    };
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
        &keys,
        &values,
        scale,
        mask,
        Option::<&Array>::None,
        &gpu,
    )?
    .transpose_axes_device(&[0, 2, 1, 3], &gpu)?
    .reshape_device(&[1, seq, heads * head_dim], &gpu)?;
    *state = Some(LayerState::Full { keys, values });
    let gated = attended.multiply_device(ops::sigmoid_device(&gate, &gpu)?, &gpu)?;
    linear(&gated, weights.tensor(&format!("{base}.o_proj.weight"))?)
}

fn mlp(weights: &Qwen35Weights, base: &str, input: &Array) -> Result<Array, Qwen35Error> {
    let gpu = StreamOrDevice::gpu();
    let gate = linear(
        input,
        weights.tensor(&format!("{base}.mlp.gate_proj.weight"))?,
    )?;
    let up = linear(
        input,
        weights.tensor(&format!("{base}.mlp.up_proj.weight"))?,
    )?;
    linear(
        &silu(&gate)?.multiply_device(&up, &gpu)?,
        weights.tensor(&format!("{base}.mlp.down_proj.weight"))?,
    )
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

/// Every decoder tensor this implementation reads, with its stored shape.
fn expected_shapes(config: &Qwen35Config) -> Result<HashMap<String, Vec<i32>>, Qwen35Error> {
    let hidden = dim(config.hidden_size)?;
    let vocab = dim(config.vocab_size)?;
    let intermediate = dim(config.intermediate_size)?;
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
        put(
            format!("{base}.mlp.gate_proj.weight"),
            &[intermediate, hidden],
        );
        put(
            format!("{base}.mlp.up_proj.weight"),
            &[intermediate, hidden],
        );
        put(
            format!("{base}.mlp.down_proj.weight"),
            &[hidden, intermediate],
        );
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
    /// A layer's stored state does not match its kind.
    #[error("layer state does not match its layer kind")]
    StateKind,
    /// A size does not fit MLX's 32-bit shape arithmetic.
    #[error("shape overflows")]
    ShapeOverflow,
    /// MLX failed to construct or evaluate a graph.
    #[error("MLX evaluation failed: {0}")]
    Mlx(#[from] mlx_rs::error::Exception),
}
