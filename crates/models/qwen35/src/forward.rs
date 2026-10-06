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

use crate::{Qwen35Config, Qwen35ConfigError, Qwen35LayerKind, Qwen35Mlp};

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
    let tensor = |name: &str| weights.tensor(&format!("{base}.mlp.{name}"));
    match weights.config.mlp {
        Qwen35Mlp::Dense { .. } => swiglu(
            input,
            tensor("gate_proj.weight")?,
            tensor("up_proj.weight")?,
            tensor("down_proj.weight")?,
        ),
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

#[cfg(test)]
mod tests {
    use mlx_rs::Array;

    use super::{MoeTensors, moe_block};

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
