//! A deliberately small, dense Qwen3 forward path for numerical qualification.
//!
//! This is not the serving path: it has no KV cache and bounds the prompt
//! length so that a full causal-attention graph is safe to use as a parity
//! oracle. It keeps Qwen3's Q/K normalization and GQA tensors separate; any
//! kernel fusion belongs behind an equivalent numerical test.

use std::{collections::HashMap, hash::BuildHasher};

use mlx_rs::{Array, StreamOrDevice, fast, ops};
use serde::Deserialize;
use thiserror::Error;

/// The largest prompt accepted by the uncached qualification forward path.
pub const MAX_DENSE_DEBUG_TOKENS: usize = 512;

/// Dense Qwen3 configuration needed to build a reference forward graph.
#[derive(Clone, Debug, PartialEq)]
pub struct Qwen3ForwardConfig {
    hidden_layers: usize,
    hidden_size: usize,
    intermediate_size: usize,
    vocab_size: usize,
    attention_heads: usize,
    key_value_heads: usize,
    head_dim: usize,
    max_position_embeddings: usize,
    rms_norm_eps: f32,
    rope_theta: f32,
}

impl Qwen3ForwardConfig {
    /// Parses only the dense, bias-free Qwen3 layout this qualification path
    /// implements. Sliding-window and scaled `RoPE` variants require their own
    /// reference vectors, so they are refused rather than silently ignored.
    pub fn parse(json: &str) -> Result<Self, Qwen3ForwardError> {
        let raw: RawForwardConfig = serde_json::from_str(json)?;
        if raw.model_type != "qwen3" {
            return Err(Qwen3ForwardError::UnsupportedModelType(raw.model_type));
        }
        if raw.attention_bias || raw.mlp_bias {
            return Err(Qwen3ForwardError::UnsupportedBiasLayout);
        }
        if raw.hidden_act != "silu" {
            return Err(Qwen3ForwardError::UnsupportedActivation(raw.hidden_act));
        }
        if !raw.tie_word_embeddings {
            return Err(Qwen3ForwardError::UntiedOutputEmbedding);
        }
        if raw.rope_scaling.is_some() {
            return Err(Qwen3ForwardError::UnsupportedRopeScaling);
        }
        if raw.use_sliding_window || raw.sliding_window.is_some() {
            return Err(Qwen3ForwardError::UnsupportedSlidingWindow);
        }

        let fields = [
            ("num_hidden_layers", raw.num_hidden_layers),
            ("hidden_size", raw.hidden_size),
            ("intermediate_size", raw.intermediate_size),
            ("vocab_size", raw.vocab_size),
            ("num_attention_heads", raw.num_attention_heads),
            ("num_key_value_heads", raw.num_key_value_heads),
            ("head_dim", raw.head_dim),
            ("max_position_embeddings", raw.max_position_embeddings),
        ];
        for (name, value) in fields {
            if value == 0 {
                return Err(Qwen3ForwardError::MissingDimension(name));
            }
        }
        if raw.head_dim % 2 != 0 {
            return Err(Qwen3ForwardError::OddHeadDimension(raw.head_dim));
        }
        if raw.head_dim > usize::from(u16::MAX) {
            return Err(Qwen3ForwardError::HeadDimensionTooLarge(raw.head_dim));
        }
        if raw.num_attention_heads % raw.num_key_value_heads != 0 {
            return Err(Qwen3ForwardError::InvalidGroupedQueryLayout {
                attention_heads: raw.num_attention_heads,
                key_value_heads: raw.num_key_value_heads,
            });
        }
        if !raw.rms_norm_eps.is_finite() || raw.rms_norm_eps <= 0.0 {
            return Err(Qwen3ForwardError::InvalidRmsNormEpsilon(raw.rms_norm_eps));
        }
        if !raw.rope_theta.is_finite() || raw.rope_theta <= 0.0 {
            return Err(Qwen3ForwardError::InvalidRopeTheta(raw.rope_theta));
        }

        Ok(Self {
            hidden_layers: raw.num_hidden_layers,
            hidden_size: raw.hidden_size,
            intermediate_size: raw.intermediate_size,
            vocab_size: raw.vocab_size,
            attention_heads: raw.num_attention_heads,
            key_value_heads: raw.num_key_value_heads,
            head_dim: raw.head_dim,
            max_position_embeddings: raw.max_position_embeddings,
            rms_norm_eps: raw.rms_norm_eps,
            rope_theta: raw.rope_theta,
        })
    }

    /// Returns the residual-stream width used by every decoder layer.
    #[must_use]
    pub(crate) const fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    /// Returns the number of decoder layers in the qualified dense layout.
    #[must_use]
    pub(crate) const fn hidden_layers(&self) -> usize {
        self.hidden_layers
    }
}

/// Runs a complete uncached Qwen3 forward pass and reads back the last-token
/// logits as f32 values.
///
/// `weights` is the direct safetensors name-to-array map. All operators are
/// constructed on the Metal GPU stream; only the final logit vector crosses
/// back to the host. The caller must keep the map resident for the duration of
/// the invocation.
pub fn forward_last_logits<S: BuildHasher>(
    weights: &HashMap<String, Array, S>,
    config: &Qwen3ForwardConfig,
    input_ids: &[i32],
) -> Result<Vec<f32>, Qwen3ForwardError> {
    validate_input_ids(config, input_ids, 0)?;

    let stream = StreamOrDevice::gpu();
    let seq_len = i32::try_from(input_ids.len()).map_err(|_| Qwen3ForwardError::ShapeOverflow)?;
    let hidden = as_i32(config.hidden_size)?;

    let ids = Array::from_slice(input_ids, &[seq_len]);
    let embedding = weight(weights, "model.embed_tokens.weight")?;
    let mut hidden_states = embedding
        .take_axis_device(&ids, 0, &stream)?
        .reshape_device(&[1, seq_len, hidden], &stream)?;

    for layer in 0..config.hidden_layers {
        hidden_states = forward_layer(config, weights, layer, &hidden_states)?;
    }

    // Only the final position is requested; normalization and the tied output
    // projection are position-independent after the decoder layers.
    let last_hidden =
        hidden_states.take_axis_device(Array::from_slice(&[seq_len - 1], &[1]), 1, &stream)?;
    let normalized = rms_norm(
        &last_hidden,
        weight(weights, "model.norm.weight")?,
        config.rms_norm_eps,
    )?;
    let logits = linear(&normalized, embedding)?;
    read_last_logits(&logits, 1, config.vocab_size)
}

/// Executes one uncached dense Qwen3 decoder layer.
///
/// This is intentionally crate-private: selected-layer diagnostics borrow it
/// while retaining the same bounded, uncached attention contract as
/// [`forward_last_logits`]. Callers must provide a residual stream with shape
/// `[1, sequence, hidden_size]`; it is not a general batched-layer API.
pub(crate) fn forward_layer<S: BuildHasher>(
    config: &Qwen3ForwardConfig,
    weights: &HashMap<String, Array, S>,
    layer: usize,
    hidden_states: &Array,
) -> Result<Array, Qwen3ForwardError> {
    let seq_len = validate_layer_input(config, layer, hidden_states)?;
    let stream = StreamOrDevice::gpu();
    let hidden = as_i32(config.hidden_size)?;
    let intermediate = as_i32(config.intermediate_size)?;
    let base = format!("model.layers.{layer}");
    let attention_input = rms_norm(
        hidden_states,
        weight(weights, &format!("{base}.input_layernorm.weight"))?,
        config.rms_norm_eps,
    )?;
    let attention = attention(config, weights, &base, &attention_input, seq_len)?;
    let residual = hidden_states.add_device(&attention, &stream)?;

    let mlp_input = rms_norm(
        &residual,
        weight(weights, &format!("{base}.post_attention_layernorm.weight"))?,
        config.rms_norm_eps,
    )?;
    let gate = linear(
        &mlp_input,
        weight(weights, &format!("{base}.mlp.gate_proj.weight"))?,
    )?
    .reshape_device(&[1, seq_len, intermediate], &stream)?;
    let up = linear(
        &mlp_input,
        weight(weights, &format!("{base}.mlp.up_proj.weight"))?,
    )?
    .reshape_device(&[1, seq_len, intermediate], &stream)?;
    let activated = ops::sigmoid_device(&gate, &stream)?.multiply_device(&gate, &stream)?;
    let mlp = linear(
        &activated.multiply_device(&up, &stream)?,
        weight(weights, &format!("{base}.mlp.down_proj.weight"))?,
    )?
    .reshape_device(&[1, seq_len, hidden], &stream)?;
    residual.add_device(&mlp, &stream).map_err(Into::into)
}

/// Applies Qwen3's final RMS normalization to a selected hidden state.
///
/// The streamed qualification owns only the final norm vector at this point,
/// so it uses this narrow helper instead of retaining a decoder weight map.
pub(crate) fn final_rms_norm(
    config: &Qwen3ForwardConfig,
    hidden_state: &Array,
    scale: &Array,
) -> Result<Array, Qwen3ForwardError> {
    rms_norm(hidden_state, scale, config.rms_norm_eps)
}

/// A dense Qwen3 forward executor with KV state bound to one weights map.
///
/// The executor borrows its model tensors, while retaining its own per-layer
/// cache. Consequently there is no public cache object that could be applied
/// to an unrelated checkpoint.
pub struct Qwen3ForwardExecutor<'a, S: BuildHasher> {
    config: &'a Qwen3ForwardConfig,
    weights: &'a HashMap<String, Array, S>,
    cache: Vec<Option<LayerKv>>,
    cached_tokens: usize,
}

struct LayerKv {
    keys: Array,
    values: Array,
}

impl<'a, S: BuildHasher> Qwen3ForwardExecutor<'a, S> {
    /// Creates an executor with an empty cache for this exact weights map.
    #[must_use]
    pub fn new(config: &'a Qwen3ForwardConfig, weights: &'a HashMap<String, Array, S>) -> Self {
        Self {
            config,
            weights,
            cache: (0..config.hidden_layers).map(|_| None).collect(),
            cached_tokens: 0,
        }
    }

    /// Clears all resident KV arrays before a new sequence.
    pub fn reset(&mut self) {
        self.cache.iter_mut().for_each(|entry| *entry = None);
        self.cached_tokens = 0;
    }

    /// Returns the number of tokens represented in every layer's cache.
    #[must_use]
    pub const fn cached_tokens(&self) -> usize {
        self.cached_tokens
    }

    /// Returns the byte total of resident K and V arrays, excluding allocator
    /// overhead. MLX reports this from the arrays' actual dtype and shape.
    #[must_use]
    pub fn kv_bytes(&self) -> usize {
        self.cache
            .iter()
            .flatten()
            .map(|entry| entry.keys.nbytes() + entry.values.nbytes())
            .sum()
    }

    /// Starts a sequence, fills its KV cache, and returns its final logits.
    pub fn prefill_last_logits(
        &mut self,
        input_ids: &[i32],
    ) -> Result<Vec<f32>, Qwen3ForwardError> {
        self.reset();
        self.append(input_ids)
    }

    /// Appends exactly one token to the current sequence and returns its logits.
    pub fn decode_last_logits(&mut self, input_id: i32) -> Result<Vec<f32>, Qwen3ForwardError> {
        if self.cached_tokens == 0 {
            return Err(Qwen3ForwardError::DecodeWithoutPrefill);
        }
        self.append(&[input_id])
    }

    fn append(&mut self, input_ids: &[i32]) -> Result<Vec<f32>, Qwen3ForwardError> {
        let result = self.append_inner(input_ids);
        if result.is_err() {
            // A failed graph may have appended only some layers. Retaining
            // that partial state would make a later decode numerically wrong.
            self.reset();
        }
        result
    }

    fn append_inner(&mut self, input_ids: &[i32]) -> Result<Vec<f32>, Qwen3ForwardError> {
        validate_input_ids(self.config, input_ids, self.cached_tokens)?;
        if self.cached_tokens != 0 && input_ids.len() != 1 {
            return Err(Qwen3ForwardError::CachedAppendRequiresOneToken);
        }

        let stream = StreamOrDevice::gpu();
        let seq_len =
            i32::try_from(input_ids.len()).map_err(|_| Qwen3ForwardError::ShapeOverflow)?;
        let hidden = as_i32(self.config.hidden_size)?;
        let intermediate = as_i32(self.config.intermediate_size)?;
        let ids = Array::from_slice(input_ids, &[seq_len]);
        let embedding = weight(self.weights, "model.embed_tokens.weight")?;
        let mut hidden_states = embedding
            .take_axis_device(&ids, 0, &stream)?
            .reshape_device(&[1, seq_len, hidden], &stream)?;

        let rope_offset =
            i32::try_from(self.cached_tokens).map_err(|_| Qwen3ForwardError::ShapeOverflow)?;
        for layer in 0..self.config.hidden_layers {
            let base = format!("model.layers.{layer}");
            let attention_input = rms_norm(
                &hidden_states,
                weight(self.weights, &format!("{base}.input_layernorm.weight"))?,
                self.config.rms_norm_eps,
            )?;
            let attention = cached_attention(
                self.config,
                self.weights,
                self.cache
                    .get_mut(layer)
                    .ok_or(Qwen3ForwardError::CacheInconsistent)?,
                &base,
                &attention_input,
                seq_len,
                rope_offset,
            )?;
            let residual = hidden_states.add_device(&attention, &stream)?;
            let mlp_input = rms_norm(
                &residual,
                weight(
                    self.weights,
                    &format!("{base}.post_attention_layernorm.weight"),
                )?,
                self.config.rms_norm_eps,
            )?;
            let gate = linear(
                &mlp_input,
                weight(self.weights, &format!("{base}.mlp.gate_proj.weight"))?,
            )?
            .reshape_device(&[1, seq_len, intermediate], &stream)?;
            let up = linear(
                &mlp_input,
                weight(self.weights, &format!("{base}.mlp.up_proj.weight"))?,
            )?
            .reshape_device(&[1, seq_len, intermediate], &stream)?;
            let activated = ops::sigmoid_device(&gate, &stream)?.multiply_device(&gate, &stream)?;
            let mlp = linear(
                &activated.multiply_device(&up, &stream)?,
                weight(self.weights, &format!("{base}.mlp.down_proj.weight"))?,
            )?
            .reshape_device(&[1, seq_len, hidden], &stream)?;
            hidden_states = residual.add_device(&mlp, &stream)?;
        }

        self.cached_tokens += input_ids.len();
        let last_hidden =
            hidden_states.take_axis_device(Array::from_slice(&[seq_len - 1], &[1]), 1, &stream)?;
        let normalized = rms_norm(
            &last_hidden,
            weight(self.weights, "model.norm.weight")?,
            self.config.rms_norm_eps,
        )?;
        let logits = linear(&normalized, embedding)?;
        read_last_logits(&logits, 1, self.config.vocab_size)
    }
}

fn cached_attention<S: BuildHasher>(
    config: &Qwen3ForwardConfig,
    weights: &HashMap<String, Array, S>,
    cache: &mut Option<LayerKv>,
    base: &str,
    input: &Array,
    seq_len: i32,
    rope_offset: i32,
) -> Result<Array, Qwen3ForwardError> {
    let stream = StreamOrDevice::gpu();
    let heads = as_i32(config.attention_heads)?;
    let kv_heads = as_i32(config.key_value_heads)?;
    let head_dim = as_i32(config.head_dim)?;
    let attn = format!("{base}.self_attn");
    let query = linear(input, weight(weights, &format!("{attn}.q_proj.weight"))?)?
        .reshape_device(&[1, seq_len, heads, head_dim], &stream)?;
    let key = linear(input, weight(weights, &format!("{attn}.k_proj.weight"))?)?
        .reshape_device(&[1, seq_len, kv_heads, head_dim], &stream)?;
    let value = linear(input, weight(weights, &format!("{attn}.v_proj.weight"))?)?
        .reshape_device(&[1, seq_len, kv_heads, head_dim], &stream)?;
    let query = fast::rope_device(
        &rms_norm(
            &query,
            weight(weights, &format!("{attn}.q_norm.weight"))?,
            config.rms_norm_eps,
        )?
        .transpose_axes_device(&[0, 2, 1, 3], &stream)?,
        head_dim,
        false,
        Some(config.rope_theta),
        1.0,
        rope_offset,
        Option::<&Array>::None,
        &stream,
    )?;
    let key = fast::rope_device(
        &rms_norm(
            &key,
            weight(weights, &format!("{attn}.k_norm.weight"))?,
            config.rms_norm_eps,
        )?
        .transpose_axes_device(&[0, 2, 1, 3], &stream)?,
        head_dim,
        false,
        Some(config.rope_theta),
        1.0,
        rope_offset,
        Option::<&Array>::None,
        &stream,
    )?;
    let value = value.transpose_axes_device(&[0, 2, 1, 3], &stream)?;
    let (keys, values, causal) = if let Some(previous) = cache.take() {
        (
            ops::concatenate_axis_device(&[&previous.keys, &key], 2, &stream)?,
            ops::concatenate_axis_device(&[&previous.values, &value], 2, &stream)?,
            false,
        )
    } else {
        (key, value, true)
    };
    // Keep KV as dependencies of attention. The final-logits readback evaluates
    // the complete graph, including these retained arrays, in one submission
    // instead of blocking twice per layer. Reset drops all request-owned KV.
    let output = if causal {
        fast::scaled_dot_product_attention_device(
            &query,
            &keys,
            &values,
            attention_scale(config)?,
            Some(fast::ScaledDotProductAttentionMask::Causal),
            &stream,
        )?
    } else {
        fast::scaled_dot_product_attention_device(
            &query,
            &keys,
            &values,
            attention_scale(config)?,
            None::<fast::ScaledDotProductAttentionMask<'_>>,
            &stream,
        )?
    };
    *cache = Some(LayerKv { keys, values });
    let output = output
        .transpose_axes_device(&[0, 2, 1, 3], &stream)?
        .reshape_device(
            &[
                1,
                seq_len,
                heads
                    .checked_mul(head_dim)
                    .ok_or(Qwen3ForwardError::ShapeOverflow)?,
            ],
            &stream,
        )?;
    linear(&output, weight(weights, &format!("{attn}.o_proj.weight"))?)
}

fn validate_input_ids(
    config: &Qwen3ForwardConfig,
    input_ids: &[i32],
    existing_tokens: usize,
) -> Result<(), Qwen3ForwardError> {
    if input_ids.is_empty() {
        return Err(Qwen3ForwardError::EmptyInput);
    }
    let total = existing_tokens
        .checked_add(input_ids.len())
        .ok_or(Qwen3ForwardError::ShapeOverflow)?;
    let maximum = MAX_DENSE_DEBUG_TOKENS.min(config.max_position_embeddings);
    if total > maximum {
        return Err(Qwen3ForwardError::PromptTooLong {
            actual: total,
            maximum,
        });
    }
    for &id in input_ids {
        if id < 0
            || usize::try_from(id)
                .ok()
                .is_none_or(|id| id >= config.vocab_size)
        {
            return Err(Qwen3ForwardError::InvalidTokenId {
                token_id: id,
                vocab_size: config.vocab_size,
            });
        }
    }
    Ok(())
}

fn validate_layer_input(
    config: &Qwen3ForwardConfig,
    layer: usize,
    hidden_states: &Array,
) -> Result<i32, Qwen3ForwardError> {
    if layer >= config.hidden_layers {
        return Err(Qwen3ForwardError::LayerOutOfRange {
            layer,
            hidden_layers: config.hidden_layers,
        });
    }

    let shape = hidden_states.shape();
    let expected_hidden = as_i32(config.hidden_size)?;
    let Some(&seq_len) = shape.get(1) else {
        return Err(Qwen3ForwardError::InvalidLayerInputShape {
            actual: shape.to_vec(),
            hidden_size: config.hidden_size,
        });
    };
    if shape.len() != 3 || shape[0] != 1 || shape[2] != expected_hidden || seq_len <= 0 {
        return Err(Qwen3ForwardError::InvalidLayerInputShape {
            actual: shape.to_vec(),
            hidden_size: config.hidden_size,
        });
    }
    let sequence = usize::try_from(seq_len).map_err(|_| Qwen3ForwardError::ShapeOverflow)?;
    let maximum = MAX_DENSE_DEBUG_TOKENS.min(config.max_position_embeddings);
    if sequence > maximum {
        return Err(Qwen3ForwardError::PromptTooLong {
            actual: sequence,
            maximum,
        });
    }
    Ok(seq_len)
}

fn attention_scale(config: &Qwen3ForwardConfig) -> Result<f32, Qwen3ForwardError> {
    Ok(
        f32::from(u16::try_from(config.head_dim).map_err(|_| Qwen3ForwardError::ShapeOverflow)?)
            .powf(-0.5),
    )
}

fn read_last_logits(
    logits: &Array,
    seq_len: i32,
    vocab_size: usize,
) -> Result<Vec<f32>, Qwen3ForwardError> {
    let stream = StreamOrDevice::gpu();
    let last = logits
        .take_axis_device(Array::from_slice(&[seq_len - 1], &[1]), 1, &stream)?
        .reshape_device(&[as_i32(vocab_size)?], &stream)?
        .as_type_device::<f32>(&stream)?;
    last.eval()?;
    Ok(last.as_slice::<f32>().to_vec())
}

fn attention<S: BuildHasher>(
    config: &Qwen3ForwardConfig,
    weights: &HashMap<String, Array, S>,
    base: &str,
    input: &Array,
    seq_len: i32,
) -> Result<Array, Qwen3ForwardError> {
    let stream = StreamOrDevice::gpu();
    let heads = as_i32(config.attention_heads)?;
    let kv_heads = as_i32(config.key_value_heads)?;
    let head_dim = as_i32(config.head_dim)?;
    let attn = format!("{base}.self_attn");
    let query = linear(input, weight(weights, &format!("{attn}.q_proj.weight"))?)?
        .reshape_device(&[1, seq_len, heads, head_dim], &stream)?;
    let key = linear(input, weight(weights, &format!("{attn}.k_proj.weight"))?)?
        .reshape_device(&[1, seq_len, kv_heads, head_dim], &stream)?;
    let value = linear(input, weight(weights, &format!("{attn}.v_proj.weight"))?)?
        .reshape_device(&[1, seq_len, kv_heads, head_dim], &stream)?;

    // Qwen3 normalizes Q and K per head, then applies nontraditional RoPE.
    let query = rms_norm(
        &query,
        weight(weights, &format!("{attn}.q_norm.weight"))?,
        config.rms_norm_eps,
    )?
    .transpose_axes_device(&[0, 2, 1, 3], &stream)?;
    let key = rms_norm(
        &key,
        weight(weights, &format!("{attn}.k_norm.weight"))?,
        config.rms_norm_eps,
    )?
    .transpose_axes_device(&[0, 2, 1, 3], &stream)?;
    let value = value.transpose_axes_device(&[0, 2, 1, 3], &stream)?;
    let query = fast::rope_device(
        &query,
        head_dim,
        false,
        Some(config.rope_theta),
        1.0,
        0,
        Option::<&Array>::None,
        &stream,
    )?;
    let key = fast::rope_device(
        &key,
        head_dim,
        false,
        Some(config.rope_theta),
        1.0,
        0,
        Option::<&Array>::None,
        &stream,
    )?;
    let output = fast::scaled_dot_product_attention_device(
        &query,
        &key,
        &value,
        attention_scale(config)?,
        Some(fast::ScaledDotProductAttentionMask::Causal),
        &stream,
    )?
    .transpose_axes_device(&[0, 2, 1, 3], &stream)?
    .reshape_device(
        &[
            1,
            seq_len,
            heads
                .checked_mul(head_dim)
                .ok_or(Qwen3ForwardError::ShapeOverflow)?,
        ],
        &stream,
    )?;
    linear(&output, weight(weights, &format!("{attn}.o_proj.weight"))?)
}

fn rms_norm(input: &Array, scale: &Array, eps: f32) -> Result<Array, Qwen3ForwardError> {
    Ok(fast::rms_norm_device(
        input,
        scale,
        eps,
        StreamOrDevice::gpu(),
    )?)
}

fn linear(input: &Array, weight: &Array) -> Result<Array, Qwen3ForwardError> {
    let stream = StreamOrDevice::gpu();
    Ok(input.matmul_device(&weight.transpose_device(&stream)?, &stream)?)
}

fn weight<'a, S: BuildHasher>(
    weights: &'a HashMap<String, Array, S>,
    name: &str,
) -> Result<&'a Array, Qwen3ForwardError> {
    weights
        .get(name)
        .ok_or_else(|| Qwen3ForwardError::MissingWeight(name.to_owned()))
}

fn as_i32(value: usize) -> Result<i32, Qwen3ForwardError> {
    i32::try_from(value).map_err(|_| Qwen3ForwardError::ShapeOverflow)
}

#[allow(
    clippy::struct_excessive_bools,
    reason = "the fields directly mirror upstream Qwen JSON gates"
)]
#[derive(Debug, Deserialize)]
struct RawForwardConfig {
    #[serde(default)]
    model_type: String,
    #[serde(default)]
    num_hidden_layers: usize,
    #[serde(default)]
    hidden_size: usize,
    #[serde(default)]
    intermediate_size: usize,
    #[serde(default)]
    vocab_size: usize,
    #[serde(default)]
    num_attention_heads: usize,
    #[serde(default)]
    num_key_value_heads: usize,
    #[serde(default)]
    head_dim: usize,
    #[serde(default)]
    max_position_embeddings: usize,
    #[serde(default = "default_eps")]
    rms_norm_eps: f32,
    #[serde(default = "default_rope_theta")]
    rope_theta: f32,
    #[serde(default)]
    attention_bias: bool,
    #[serde(default)]
    mlp_bias: bool,
    #[serde(default)]
    hidden_act: String,
    #[serde(default)]
    tie_word_embeddings: bool,
    rope_scaling: Option<serde_json::Value>,
    sliding_window: Option<usize>,
    #[serde(default)]
    use_sliding_window: bool,
}

const fn default_eps() -> f32 {
    1e-6
}

const fn default_rope_theta() -> f32 {
    1_000_000.0
}

/// Errors from Qwen3 configuration qualification or its Metal forward graph.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum Qwen3ForwardError {
    /// The configuration was not JSON.
    #[error("invalid Qwen3 forward configuration: {0}")]
    Json(#[from] serde_json::Error),
    /// The configuration selected a non-Qwen3 architecture.
    #[error("expected model_type qwen3, got {0:?}")]
    UnsupportedModelType(String),
    /// The forward path needs a positive model dimension.
    #[error("Qwen3 configuration has no usable {0}")]
    MissingDimension(&'static str),
    /// Qwen3's rotary head width must be even.
    #[error("Qwen3 head_dim must be even, got {0}")]
    OddHeadDimension(usize),
    /// The qualification path keeps its scale calculation exact in f32.
    #[error("Qwen3 head_dim {0} exceeds the qualification forward limit")]
    HeadDimensionTooLarge(usize),
    /// The query heads cannot be divided among key/value heads.
    #[error("Qwen3 has {attention_heads} query heads and {key_value_heads} key/value heads")]
    InvalidGroupedQueryLayout {
        /// Query heads.
        attention_heads: usize,
        /// Key/value heads.
        key_value_heads: usize,
    },
    /// The normalization epsilon cannot define a stable norm.
    #[error("invalid rms_norm_eps {0}")]
    InvalidRmsNormEpsilon(f32),
    /// The `RoPE` base cannot define a rotary frequency table.
    #[error("invalid rope_theta {0}")]
    InvalidRopeTheta(f32),
    /// The adapter has no reference vectors for projection bias.
    #[error("Qwen3 attention or MLP bias is unsupported by the qualification forward path")]
    UnsupportedBiasLayout,
    /// The forward path implements Qwen3's SiLU-gated MLP only.
    #[error("Qwen3 hidden_act {0:?} is unsupported; expected silu")]
    UnsupportedActivation(String),
    /// The qualification path projects logits through the embedding table.
    #[error("Qwen3 untied output embeddings require a dedicated lm_head path")]
    UntiedOutputEmbedding,
    /// The adapter has no reference vectors for a RoPE-scaling variant.
    #[error("Qwen3 rope_scaling requires a dedicated qualification path")]
    UnsupportedRopeScaling,
    /// The adapter has no reference vectors for sliding-window attention.
    #[error("Qwen3 sliding_window requires a dedicated qualification path")]
    UnsupportedSlidingWindow,
    /// The raw input must contain at least one token.
    #[error("Qwen3 forward requires at least one token")]
    EmptyInput,
    /// The uncached reference attention graph is deliberately bounded.
    #[error("Qwen3 reference prompt has {actual} tokens, maximum is {maximum}")]
    PromptTooLong {
        /// Provided token count.
        actual: usize,
        /// Supported token count.
        maximum: usize,
    },
    /// A selected decoder layer is outside this model's configured range.
    #[error("Qwen3 layer {layer} is outside {hidden_layers} configured layers")]
    LayerOutOfRange {
        /// Requested zero-based layer index.
        layer: usize,
        /// Number of configured decoder layers.
        hidden_layers: usize,
    },
    /// A selected-layer diagnostic did not receive one residual stream.
    #[error(
        "Qwen3 layer input shape {actual:?} must be [1, sequence, {hidden_size}] with positive sequence"
    )]
    InvalidLayerInputShape {
        /// Actual MLX array shape.
        actual: Vec<i32>,
        /// Required residual-stream width.
        hidden_size: usize,
    },
    /// Decode needs a preceding prefill on this executor.
    #[error("Qwen3 decode requires a populated KV cache")]
    DecodeWithoutPrefill,
    /// Only decode-sized appends preserve the uncached reference contract.
    #[error("Qwen3 cached append requires exactly one token")]
    CachedAppendRequiresOneToken,
    /// The executor's layer cache no longer matches its model contract.
    #[error("Qwen3 layer KV cache is inconsistent with its configuration")]
    CacheInconsistent,
    /// A token does not fit the checkpoint vocabulary.
    #[error("token ID {token_id} is outside vocabulary size {vocab_size}")]
    InvalidTokenId {
        /// Raw token ID.
        token_id: i32,
        /// Checkpoint vocabulary size.
        vocab_size: usize,
    },
    /// A required checkpoint tensor was absent.
    #[error("Qwen3 checkpoint is missing {0}")]
    MissingWeight(String),
    /// A dimension cannot be represented by MLX's i32 shape API.
    #[error("Qwen3 shape exceeds MLX's i32 dimension limit")]
    ShapeOverflow,
    /// MLX could not construct, evaluate, or copy the Metal graph.
    #[error("MLX Qwen3 forward failed: {0}")]
    Mlx(#[from] mlx_rs::error::Exception),
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use mlx_rs::Array;

    use crate::GPU_TEST_LOCK;

    use super::{
        Qwen3ForwardConfig, Qwen3ForwardError, Qwen3ForwardExecutor, forward_last_logits,
        forward_layer, linear, read_last_logits, rms_norm, weight,
    };

    const QWEN3_06B: &str = r#"{
      "model_type":"qwen3",
      "num_hidden_layers":28,
      "hidden_size":1024,
      "intermediate_size":3072,
      "vocab_size":151936,
      "num_attention_heads":16,
      "num_key_value_heads":8,
      "head_dim":128,
      "max_position_embeddings":40960,
      "rms_norm_eps":0.000001,
      "rope_theta":1000000,
      "hidden_act":"silu",
      "tie_word_embeddings":true,
      "attention_bias":false,
      "mlp_bias":false,
      "sliding_window":null,
      "use_sliding_window":false
    }"#;

    #[test]
    fn accepts_qwen3_06b_expanded_query_width() {
        // Qwen3-0.6B deliberately has 16 * 128 = 2048 query features while
        // its residual stream is only 1024 wide; `o_proj` maps it back.
        Qwen3ForwardConfig::parse(QWEN3_06B).expect("official Qwen3-0.6B layout");
    }

    #[test]
    fn refuses_layouts_the_tied_dense_path_cannot_implement() {
        let untied = QWEN3_06B.replace(
            "\"tie_word_embeddings\":true",
            "\"tie_word_embeddings\":false",
        );
        assert!(matches!(
            Qwen3ForwardConfig::parse(&untied),
            Err(Qwen3ForwardError::UntiedOutputEmbedding)
        ));
        let activation = QWEN3_06B.replace("\"hidden_act\":\"silu\"", "\"hidden_act\":\"gelu\"");
        assert!(matches!(
            Qwen3ForwardConfig::parse(&activation),
            Err(Qwen3ForwardError::UnsupportedActivation(_))
        ));
        let window = QWEN3_06B.replace(
            "\"use_sliding_window\":false",
            "\"use_sliding_window\":true",
        );
        assert!(matches!(
            Qwen3ForwardConfig::parse(&window),
            Err(Qwen3ForwardError::UnsupportedSlidingWindow)
        ));
    }

    #[test]
    fn cached_decode_matches_full_causal_forward_for_nonzero_weights() {
        let _gpu = GPU_TEST_LOCK.lock().expect("GPU test lock");
        let config = Qwen3ForwardConfig::parse(
            r#"{
              "model_type":"qwen3",
              "num_hidden_layers":1,
              "hidden_size":4,
              "intermediate_size":8,
              "vocab_size":8,
              "num_attention_heads":2,
              "num_key_value_heads":1,
              "head_dim":4,
              "max_position_embeddings":16,
              "rms_norm_eps":0.000001,
              "rope_theta":1000000,
              "hidden_act":"silu",
              "tie_word_embeddings":true,
              "attention_bias":false,
              "mlp_bias":false
            }"#,
        )
        .expect("small dense Qwen3 config");
        let weights = deterministic_weights();
        let mut executor = Qwen3ForwardExecutor::new(&config, &weights);
        let _ = executor.prefill_last_logits(&[1, 2]).expect("prefill");
        assert_eq!(executor.cached_tokens(), 2);
        assert!(executor.kv_bytes() > 0);
        for (next, prefix) in [
            (3, &[1, 2, 3][..]),
            (4, &[1, 2, 3, 4][..]),
            (5, &[1, 2, 3, 4, 5][..]),
        ] {
            let cached = executor.decode_last_logits(next).expect("cached decode");
            assert_eq!(executor.cached_tokens(), prefix.len());
            assert_logits_match(
                forward_last_logits(&weights, &config, prefix).expect("full forward"),
                cached,
            );
        }

        // A new prefill must discard all prior request-owned KV, including
        // arrays retained through a previous lazy cached-decode graph.
        let _ = executor
            .prefill_last_logits(&[5, 6])
            .expect("second prefill");
        assert_eq!(executor.cached_tokens(), 2);
        let cached = executor
            .decode_last_logits(7)
            .expect("second cached decode");
        assert_eq!(executor.cached_tokens(), 3);
        assert_logits_match(
            forward_last_logits(&weights, &config, &[5, 6, 7]).expect("second full forward"),
            cached,
        );

        // An invalid append fails closed rather than leaving the first layers'
        // KV available for a later request.
        assert!(matches!(
            executor.decode_last_logits(8),
            Err(Qwen3ForwardError::InvalidTokenId { .. })
        ));
        assert_eq!(executor.cached_tokens(), 0);
        assert_eq!(executor.kv_bytes(), 0);
        assert!(matches!(
            executor.decode_last_logits(1),
            Err(Qwen3ForwardError::DecodeWithoutPrefill)
        ));

        executor.reset();
        assert_eq!(executor.cached_tokens(), 0);
        assert_eq!(executor.kv_bytes(), 0);
    }

    #[test]
    fn composed_nonzero_layer_matches_independent_cached_prefill() {
        let _gpu = GPU_TEST_LOCK.lock().expect("GPU test lock");
        let config = Qwen3ForwardConfig::parse(
            r#"{
              "model_type":"qwen3",
              "num_hidden_layers":2,
              "hidden_size":4,
              "intermediate_size":8,
              "vocab_size":8,
              "num_attention_heads":2,
              "num_key_value_heads":1,
              "head_dim":4,
              "max_position_embeddings":16,
              "rms_norm_eps":0.000001,
              "rope_theta":1000000,
              "hidden_act":"silu",
              "tie_word_embeddings":true,
              "attention_bias":false,
              "mlp_bias":false
            }"#,
        )
        .expect("two-layer dense Qwen3 config");
        let weights = deterministic_weights_for_layers(2);
        let input_ids = [1_i32, 2, 3];
        let stream = mlx_rs::StreamOrDevice::gpu();
        let embedding = weight(&weights, "model.embed_tokens.weight").expect("embedding");
        let ids = Array::from_slice(&input_ids, &[3]);
        let hidden = embedding
            .take_axis_device(&ids, 0, &stream)
            .expect("embedding lookup")
            .reshape_device(&[1, 3, 4], &stream)
            .expect("residual shape");

        // The cached executor has its own attention and MLP implementation.
        // Its prefill result therefore checks the extracted nonzero layer's
        // contribution instead of comparing this helper to itself.
        let after_layer_zero = forward_layer(&config, &weights, 0, &hidden).expect("layer zero");
        let after_layer_one =
            forward_layer(&config, &weights, 1, &after_layer_zero).expect("nonzero layer");
        let last = after_layer_one
            .take_axis_device(Array::from_slice(&[2_i32], &[1]), 1, &stream)
            .expect("last hidden state");
        let normalized = rms_norm(
            &last,
            weight(&weights, "model.norm.weight").expect("final norm"),
            config.rms_norm_eps,
        )
        .expect("final norm graph");
        let composed = read_last_logits(
            &linear(&normalized, embedding).expect("tied output projection"),
            1,
            8,
        )
        .expect("composed logits");

        let cached = Qwen3ForwardExecutor::new(&config, &weights)
            .prefill_last_logits(&input_ids)
            .expect("independent cached prefill");
        assert_logits_match(composed, cached);
    }

    #[test]
    fn selected_layer_rejects_out_of_range_and_invalid_residual_shape() {
        let config = Qwen3ForwardConfig::parse(
            r#"{
              "model_type":"qwen3", "num_hidden_layers":1, "hidden_size":4,
              "intermediate_size":8, "vocab_size":8, "num_attention_heads":2,
              "num_key_value_heads":1, "head_dim":4, "max_position_embeddings":16,
              "hidden_act":"silu", "tie_word_embeddings":true
            }"#,
        )
        .expect("small config");
        let weights = deterministic_weights();
        let bad_shape = Array::from_slice(&[1.0_f32; 8], &[2, 4]);
        assert!(matches!(
            forward_layer(&config, &weights, 1, &bad_shape),
            Err(Qwen3ForwardError::LayerOutOfRange { .. })
        ));
        assert!(matches!(
            forward_layer(&config, &weights, 0, &bad_shape),
            Err(Qwen3ForwardError::InvalidLayerInputShape { .. })
        ));
        let overlong = Array::from_slice(&[1.0_f32; 68], &[1, 17, 4]);
        assert!(matches!(
            forward_layer(&config, &weights, 0, &overlong),
            Err(Qwen3ForwardError::PromptTooLong {
                actual: 17,
                maximum: 16
            })
        ));
    }

    fn assert_logits_match(full: Vec<f32>, cached: Vec<f32>) {
        assert_eq!(full.len(), cached.len());
        for (full, cached) in full.into_iter().zip(cached) {
            assert!(
                (full - cached).abs() <= 5e-5,
                "cached logit {cached} differs from full logit {full}"
            );
        }
    }

    fn deterministic_weights() -> HashMap<String, Array> {
        deterministic_weights_for_layers(1)
    }

    fn deterministic_weights_for_layers(layers: usize) -> HashMap<String, Array> {
        let mut weights = HashMap::new();
        insert_matrix(&mut weights, "model.embed_tokens.weight", 8, 4);
        insert_vector(&mut weights, "model.norm.weight", 4);
        for layer in 0..layers {
            let base = format!("model.layers.{layer}");
            insert_vector(&mut weights, &format!("{base}.input_layernorm.weight"), 4);
            insert_vector(
                &mut weights,
                &format!("{base}.post_attention_layernorm.weight"),
                4,
            );
            let attn = format!("{base}.self_attn");
            insert_matrix(&mut weights, &format!("{attn}.q_proj.weight"), 8, 4);
            insert_matrix(&mut weights, &format!("{attn}.k_proj.weight"), 4, 4);
            insert_matrix(&mut weights, &format!("{attn}.v_proj.weight"), 4, 4);
            insert_matrix(&mut weights, &format!("{attn}.o_proj.weight"), 4, 8);
            insert_vector(&mut weights, &format!("{attn}.q_norm.weight"), 4);
            insert_vector(&mut weights, &format!("{attn}.k_norm.weight"), 4);
            let mlp = format!("{base}.mlp");
            insert_matrix(&mut weights, &format!("{mlp}.gate_proj.weight"), 8, 4);
            insert_matrix(&mut weights, &format!("{mlp}.up_proj.weight"), 8, 4);
            insert_matrix(&mut weights, &format!("{mlp}.down_proj.weight"), 4, 8);
        }
        weights
    }

    fn insert_matrix(weights: &mut HashMap<String, Array>, name: &str, rows: i32, columns: i32) {
        let count = usize::try_from(rows * columns).expect("small test shape");
        weights.insert(
            name.to_owned(),
            Array::from_slice(&nonzero_values(count), &[rows, columns]),
        );
    }

    fn insert_vector(weights: &mut HashMap<String, Array>, name: &str, length: i32) {
        weights.insert(
            name.to_owned(),
            Array::from_slice(
                &nonzero_values(usize::try_from(length).expect("small test shape")),
                &[length],
            ),
        );
    }

    fn nonzero_values(length: usize) -> Vec<f32> {
        (0..length)
            .map(|index| {
                (f32::from(u8::try_from(index % 11).expect("bounded test value")) + 1.0) * 0.017
            })
            .collect()
    }
}
