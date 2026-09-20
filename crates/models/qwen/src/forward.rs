//! Qwen3 forward paths for an uncached numerical oracle and cached resident chat.
//!
//! The uncached dense oracle remains bounded to 512 tokens so its full
//! causal-attention graph is safe for parity checks. The adapter-local cached
//! decoder has separate resident-context admission and retains Qwen3's Q/K
//! normalization and GQA tensors; any kernel fusion belongs behind an
//! equivalent numerical test.

use std::{collections::HashMap, hash::BuildHasher};

#[cfg(test)]
use std::{
    cell::RefCell,
    time::{Duration, Instant},
};

use mlx_rs::{
    Array, StreamOrDevice, fast, ops,
    ops::indexing::{IndexMutOp, IndexOp},
};
use serde::Deserialize;
use thiserror::Error;

/// The largest prompt accepted by the uncached qualification forward path.
pub const MAX_DENSE_DEBUG_TOKENS: usize = 512;

/// Experimental resident-chat admission ceiling. This does not expand the
/// 512-token uncached diagnostic forward path or qualify longer contexts.
pub const MAX_RESIDENT_CHAT_TOKENS: usize = 16_384;

/// Default logical K/V budget for the resident-chat control path.
pub const DEFAULT_RESIDENT_CHAT_KV_BUDGET_BYTES: u64 = 512 * 1024 * 1024;

/// A checked resident-chat context and its final logical K/V estimate.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Qwen3ResidentChatPlan {
    maximum_context_tokens: usize,
    planned_kv_bytes: u64,
}

impl Qwen3ResidentChatPlan {
    /// Maximum prompt-plus-generated tokens admitted by this executor.
    #[must_use]
    pub const fn maximum_context_tokens(self) -> usize {
        self.maximum_context_tokens
    }

    /// Logical f32 K/V bytes estimated for that maximum context.
    #[must_use]
    pub const fn planned_kv_bytes(self) -> u64 {
        self.planned_kv_bytes
    }
}

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
        if !raw.head_dim.is_multiple_of(2) {
            return Err(Qwen3ForwardError::OddHeadDimension(raw.head_dim));
        }
        if raw.head_dim > usize::from(u16::MAX) {
            return Err(Qwen3ForwardError::HeadDimensionTooLarge(raw.head_dim));
        }
        if !raw
            .num_attention_heads
            .is_multiple_of(raw.num_key_value_heads)
        {
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

    /// Logical f32 bytes for all layer K/V arrays at `tokens` positions.
    ///
    /// The streamed checker stores detached f32 arrays, so this is deliberately
    /// not a checkpoint-byte estimate or a generic cache-layout abstraction.
    pub(crate) fn cached_kv_bytes(&self, tokens: usize) -> Result<u64, Qwen3ForwardError> {
        let values = self
            .hidden_layers
            .checked_mul(2)
            .and_then(|value| value.checked_mul(self.key_value_heads))
            .and_then(|value| value.checked_mul(tokens))
            .and_then(|value| value.checked_mul(self.head_dim))
            .ok_or(Qwen3ForwardError::ShapeOverflow)?;
        u64::try_from(values)
            .ok()
            .and_then(|value| value.checked_mul(u64::try_from(size_of::<f32>()).ok()?))
            .ok_or(Qwen3ForwardError::ShapeOverflow)
    }

    /// Effective context limit of the bounded dense qualification path.
    #[must_use]
    pub(crate) fn maximum_cached_tokens(&self) -> usize {
        MAX_DENSE_DEBUG_TOKENS.min(self.max_position_embeddings)
    }

    /// Validates a resident-chat context before checkpoint payloads are loaded.
    ///
    /// The K/V budget covers only the estimated retained f32 cache arrays. It
    /// excludes model weights, activations, operator scratch, and allocator
    /// headroom; it does not preallocate or reserve MLX memory.
    pub fn resident_chat_plan(
        &self,
        maximum_context_tokens: usize,
        maximum_kv_bytes: u64,
    ) -> Result<Qwen3ResidentChatPlan, Qwen3ForwardError> {
        let maximum = MAX_RESIDENT_CHAT_TOKENS.min(self.max_position_embeddings);
        if maximum_context_tokens == 0 || maximum_context_tokens > maximum {
            return Err(Qwen3ForwardError::ResidentChatContextLimit {
                requested: maximum_context_tokens,
                maximum,
            });
        }
        let planned_kv_bytes = self.cached_kv_bytes(maximum_context_tokens)?;
        if planned_kv_bytes > maximum_kv_bytes {
            return Err(Qwen3ForwardError::ResidentChatKvBudget {
                required: planned_kv_bytes,
                maximum: maximum_kv_bytes,
            });
        }
        Ok(Qwen3ResidentChatPlan {
            maximum_context_tokens,
            planned_kv_bytes,
        })
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
    validate_input_ids(config, input_ids, 0, config.maximum_cached_tokens())?;

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
    cache: Vec<Option<Qwen3LayerKv>>,
    cached_tokens: usize,
    maximum_context_tokens: usize,
    resident_chat_plan: Option<Qwen3ResidentChatPlan>,
    resident_cache_capacity: Option<usize>,
}

/// Adapter-local Qwen3 KV state.
///
/// This stays crate-visible: the bounded streamed checker needs to rebuild it
/// after each layer so no cache graph retains that layer's transient weights.
pub(crate) struct Qwen3LayerKv {
    pub(crate) keys: Array,
    pub(crate) values: Array,
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
            maximum_context_tokens: config.maximum_cached_tokens(),
            resident_chat_plan: None,
            resident_cache_capacity: None,
        }
    }

    /// Creates a separately bounded resident-chat executor from a preflighted
    /// plan. The normal constructor remains on the 512-token diagnostic cap.
    #[must_use]
    pub(crate) fn new_for_resident_chat(
        config: &'a Qwen3ForwardConfig,
        weights: &'a HashMap<String, Array, S>,
        plan: Qwen3ResidentChatPlan,
    ) -> Self {
        Self {
            config,
            weights,
            cache: (0..config.hidden_layers).map(|_| None).collect(),
            cached_tokens: 0,
            maximum_context_tokens: plan.maximum_context_tokens,
            resident_chat_plan: Some(plan),
            resident_cache_capacity: Some(plan.maximum_context_tokens),
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

    /// Maximum prompt-plus-generated tokens accepted by this executor.
    #[must_use]
    pub const fn maximum_context_tokens(&self) -> usize {
        self.maximum_context_tokens
    }

    /// The explicit resident-chat plan, when this was created for chat.
    #[must_use]
    pub const fn resident_chat_plan(&self) -> Option<Qwen3ResidentChatPlan> {
        self.resident_chat_plan
    }

    /// Estimated final logical f32 K/V bytes for a resident-chat executor.
    ///
    /// This is absent on the 512-token diagnostic executor and does not
    /// represent an MLX allocation or reservation.
    #[must_use]
    pub const fn planned_kv_bytes(&self) -> Option<u64> {
        match self.resident_chat_plan {
            Some(plan) => Some(plan.planned_kv_bytes),
            None => None,
        }
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

    /// Creates an independent decoder branch from this fully materialized KV
    /// snapshot for the Qwen cache-replay qualification test. This is neither
    /// a serving operation nor a cross-model cache API.
    ///
    /// The retained arrays are evaluated before their handles are cloned.
    /// Later decode appends build replacement K/V arrays through concatenation;
    /// they do not mutate the snapshot arrays owned by this executor.
    #[cfg(test)]
    pub(crate) fn fork_prefilled(&self) -> Result<Self, Qwen3ForwardError> {
        if self.cached_tokens == 0 {
            return Err(Qwen3ForwardError::DecodeWithoutPrefill);
        }
        if self.cache.len() != self.config.hidden_layers || self.cache.iter().any(Option::is_none) {
            return Err(Qwen3ForwardError::CacheInconsistent);
        }
        let stored_tokens = self.resident_cache_capacity.unwrap_or(self.cached_tokens);
        let expected_shape = [
            1,
            as_i32(self.config.key_value_heads)?,
            as_i32(stored_tokens)?,
            as_i32(self.config.head_dim)?,
        ];
        for layer in self.cache.iter().flatten() {
            if layer.keys.shape() != expected_shape || layer.values.shape() != expected_shape {
                return Err(Qwen3ForwardError::CacheInconsistent);
            }
            layer.keys.eval()?;
            layer.values.eval()?;
        }
        Ok(Self {
            config: self.config,
            weights: self.weights,
            cache: self
                .cache
                .iter()
                .map(|entry| {
                    entry.as_ref().map(|layer| Qwen3LayerKv {
                        keys: layer.keys.clone(),
                        values: layer.values.clone(),
                    })
                })
                .collect(),
            cached_tokens: self.cached_tokens,
            maximum_context_tokens: self.maximum_context_tokens,
            resident_chat_plan: self.resident_chat_plan,
            resident_cache_capacity: self.resident_cache_capacity,
        })
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
        validate_input_ids(
            self.config,
            input_ids,
            self.cached_tokens,
            self.maximum_context_tokens,
        )?;
        if self.cached_tokens != 0 && input_ids.len() != 1 {
            return Err(Qwen3ForwardError::CachedAppendRequiresOneToken);
        }

        let stream = StreamOrDevice::gpu();
        let seq_len =
            i32::try_from(input_ids.len()).map_err(|_| Qwen3ForwardError::ShapeOverflow)?;
        let hidden = as_i32(self.config.hidden_size)?;
        let ids = Array::from_slice(input_ids, &[seq_len]);
        let embedding = weight(self.weights, "model.embed_tokens.weight")?;
        let mut hidden_states = embedding
            .take_axis_device(&ids, 0, &stream)?
            .reshape_device(&[1, seq_len, hidden], &stream)?;

        let rope_offset =
            i32::try_from(self.cached_tokens).map_err(|_| Qwen3ForwardError::ShapeOverflow)?;
        for layer in 0..self.config.hidden_layers {
            hidden_states = forward_cached_layer_with_capacity(
                self.config,
                self.weights,
                layer,
                self.cache
                    .get_mut(layer)
                    .ok_or(Qwen3ForwardError::CacheInconsistent)?,
                &hidden_states,
                seq_len,
                rope_offset,
                self.resident_cache_capacity,
            )?;
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

/// Executes one cached Qwen3 decoder layer and updates its adapter-local KV.
///
/// The cache is intentionally not a cross-model engine interface. It carries
/// Qwen3's GQA layout and `RoPE` contract, while allowing the streamed checker
/// to detach the cache immediately after the layer evaluates.
pub(crate) fn forward_cached_layer<S: BuildHasher>(
    config: &Qwen3ForwardConfig,
    weights: &HashMap<String, Array, S>,
    layer: usize,
    cache: &mut Option<Qwen3LayerKv>,
    hidden_states: &Array,
    seq_len: i32,
    rope_offset: i32,
) -> Result<Array, Qwen3ForwardError> {
    forward_cached_layer_with_capacity(
        config,
        weights,
        layer,
        cache,
        hidden_states,
        seq_len,
        rope_offset,
        None,
    )
}

#[allow(clippy::too_many_arguments)]
fn forward_cached_layer_with_capacity<S: BuildHasher>(
    config: &Qwen3ForwardConfig,
    weights: &HashMap<String, Array, S>,
    layer: usize,
    cache: &mut Option<Qwen3LayerKv>,
    hidden_states: &Array,
    seq_len: i32,
    rope_offset: i32,
    resident_cache_capacity: Option<usize>,
) -> Result<Array, Qwen3ForwardError> {
    let stream = StreamOrDevice::gpu();
    let hidden = as_i32(config.hidden_size)?;
    let intermediate = as_i32(config.intermediate_size)?;
    let base = format!("model.layers.{layer}");
    let attention_input = rms_norm(
        hidden_states,
        weight(weights, &format!("{base}.input_layernorm.weight"))?,
        config.rms_norm_eps,
    )?;
    let attention = cached_attention(
        config,
        weights,
        cache,
        &base,
        &attention_input,
        seq_len,
        rope_offset,
        resident_cache_capacity,
    )?;
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

#[allow(clippy::too_many_arguments)]
fn cached_attention<S: BuildHasher>(
    config: &Qwen3ForwardConfig,
    weights: &HashMap<String, Array, S>,
    cache: &mut Option<Qwen3LayerKv>,
    base: &str,
    input: &Array,
    seq_len: i32,
    rope_offset: i32,
    resident_cache_capacity: Option<usize>,
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
    let (keys, values, attention_keys, attention_values, causal) = match resident_cache_capacity {
        Some(maximum_capacity) => {
            stepped_cached_kv(cache, &key, &value, rope_offset, maximum_capacity, &stream)?
        }
        None => {
            if let Some(previous) = cache.take() {
                (
                    ops::concatenate_axis_device(&[&previous.keys, &key], 2, &stream)?,
                    ops::concatenate_axis_device(&[&previous.values, &value], 2, &stream)?,
                    None,
                    None,
                    false,
                )
            } else {
                (key, value, None, None, true)
            }
        }
    };
    let attention_keys = attention_keys.as_ref().unwrap_or(&keys);
    let attention_values = attention_values.as_ref().unwrap_or(&values);
    // Keep KV as dependencies of attention. The final-logits readback evaluates
    // the complete graph, including these retained arrays, in one submission
    // instead of blocking twice per layer. Reset drops all request-owned KV.
    let output = if causal {
        fast::scaled_dot_product_attention_device(
            &query,
            attention_keys,
            attention_values,
            attention_scale(config)?,
            Some(fast::ScaledDotProductAttentionMask::Causal),
            &stream,
        )?
    } else {
        fast::scaled_dot_product_attention_device(
            &query,
            attention_keys,
            attention_values,
            attention_scale(config)?,
            None::<fast::ScaledDotProductAttentionMask<'_>>,
            &stream,
        )?
    };
    *cache = Some(Qwen3LayerKv { keys, values });
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

type SteppedKv = (Array, Array, Option<Array>, Option<Array>, bool);

fn stepped_cached_kv(
    cache: &mut Option<Qwen3LayerKv>,
    key: &Array,
    value: &Array,
    rope_offset: i32,
    maximum_capacity: usize,
    stream: &StreamOrDevice,
) -> Result<SteppedKv, Qwen3ForwardError> {
    let appended = *key
        .shape()
        .get(2)
        .ok_or(Qwen3ForwardError::CacheInconsistent)?;
    let next = rope_offset
        .checked_add(appended)
        .ok_or(Qwen3ForwardError::ShapeOverflow)?;
    let next_usize = usize::try_from(next).map_err(|_| Qwen3ForwardError::ShapeOverflow)?;
    let capacity = stepped_capacity(next_usize, maximum_capacity)?;
    let capacity_i32 = as_i32(capacity)?;
    let key_shape = key.shape();
    let value_shape = value.shape();
    if key_shape.len() != 4 || value_shape != key_shape {
        return Err(Qwen3ForwardError::CacheInconsistent);
    }
    let expected_storage_shape = [key_shape[0], key_shape[1], capacity_i32, key_shape[3]];
    let (mut keys, mut values, causal) = match cache.take() {
        Some(previous)
            if previous.keys.shape() == expected_storage_shape
                && previous.values.shape() == expected_storage_shape =>
        {
            (previous.keys, previous.values, false)
        }
        Some(previous) => {
            let prior = previous
                .keys
                .shape()
                .get(2)
                .copied()
                .ok_or(Qwen3ForwardError::CacheInconsistent)?;
            if previous.keys.shape().len() != 4
                || previous.values.shape() != previous.keys.shape()
                || previous.keys.shape()[0] != key_shape[0]
                || previous.keys.shape()[1] != key_shape[1]
                || previous.keys.shape()[3] != key_shape[3]
                || prior < rope_offset
            {
                return Err(Qwen3ForwardError::CacheInconsistent);
            }
            let key_prefix = previous
                .keys
                .index_device((.., .., 0..rope_offset, ..), stream);
            let value_prefix = previous
                .values
                .index_device((.., .., 0..rope_offset, ..), stream);
            let mut keys = ops::zeros_dtype_device(&expected_storage_shape, key.dtype(), stream)?;
            let mut values =
                ops::zeros_dtype_device(&expected_storage_shape, value.dtype(), stream)?;
            keys.index_mut_device((.., .., 0..rope_offset, ..), &key_prefix, stream);
            values.index_mut_device((.., .., 0..rope_offset, ..), &value_prefix, stream);
            (keys, values, false)
        }
        None => (
            ops::zeros_dtype_device(&expected_storage_shape, key.dtype(), stream)?,
            ops::zeros_dtype_device(&expected_storage_shape, value.dtype(), stream)?,
            true,
        ),
    };
    keys.index_mut_device((.., .., rope_offset..next, ..), &key, stream);
    values.index_mut_device((.., .., rope_offset..next, ..), &value, stream);
    let attention_keys = keys.index_device((.., .., 0..next, ..), stream);
    let attention_values = values.index_device((.., .., 0..next, ..), stream);
    Ok((
        keys,
        values,
        Some(attention_keys),
        Some(attention_values),
        causal,
    ))
}

fn stepped_capacity(
    next_tokens: usize,
    maximum_capacity: usize,
) -> Result<usize, Qwen3ForwardError> {
    if next_tokens > maximum_capacity {
        return Err(Qwen3ForwardError::PromptTooLong {
            actual: next_tokens,
            maximum: maximum_capacity,
        });
    }
    Ok([128, 512]
        .into_iter()
        .find(|&boundary| next_tokens <= boundary && boundary <= maximum_capacity)
        .unwrap_or(maximum_capacity))
}

fn validate_input_ids(
    config: &Qwen3ForwardConfig,
    input_ids: &[i32],
    existing_tokens: usize,
    maximum: usize,
) -> Result<(), Qwen3ForwardError> {
    if input_ids.is_empty() {
        return Err(Qwen3ForwardError::EmptyInput);
    }
    let total = existing_tokens
        .checked_add(input_ids.len())
        .ok_or(Qwen3ForwardError::ShapeOverflow)?;
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
    #[cfg(test)]
    let evaluation_started = Instant::now();
    last.eval()?;
    #[cfg(test)]
    record_decode_profile_evaluation(evaluation_started.elapsed());

    #[cfg(test)]
    let readback_started = Instant::now();
    let output = last.as_slice::<f32>().to_vec();
    #[cfg(test)]
    record_decode_profile_readback(readback_started.elapsed());
    Ok(output)
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
    #[cfg(test)]
    let transpose_started = Instant::now();
    let transposed = weight.transpose_device(&stream)?;
    #[cfg(test)]
    record_decode_profile_transpose_node(transpose_started.elapsed());
    Ok(input.matmul_device(&transposed, &stream)?)
}

/// Per-thread, opt-in timings for the local checkpoint decode-profile test.
///
/// These durations are host-side intervals around MLX API calls. In
/// particular, `evaluation` includes waiting for MLX work and is not GPU time.
#[cfg(test)]
#[derive(Clone, Debug, Default)]
pub(super) struct DecodeProfile {
    pub(super) transpose_node: Duration,
    pub(super) transpose_node_count: usize,
    pub(super) evaluation: Duration,
    pub(super) evaluation_count: usize,
    pub(super) readback: Duration,
    pub(super) readback_count: usize,
}

#[cfg(test)]
thread_local! {
    static DECODE_PROFILE: RefCell<Option<DecodeProfile>> = const { RefCell::new(None) };
}

#[cfg(test)]
fn record_decode_profile_transpose_node(elapsed: Duration) {
    DECODE_PROFILE.with(|profile| {
        if let Some(profile) = profile.borrow_mut().as_mut() {
            profile.transpose_node += elapsed;
            profile.transpose_node_count += 1;
        }
    });
}

#[cfg(test)]
fn record_decode_profile_evaluation(elapsed: Duration) {
    DECODE_PROFILE.with(|profile| {
        if let Some(profile) = profile.borrow_mut().as_mut() {
            profile.evaluation += elapsed;
            profile.evaluation_count += 1;
        }
    });
}

#[cfg(test)]
fn record_decode_profile_readback(elapsed: Duration) {
    DECODE_PROFILE.with(|profile| {
        if let Some(profile) = profile.borrow_mut().as_mut() {
            profile.readback += elapsed;
            profile.readback_count += 1;
        }
    });
}

#[cfg(test)]
pub(super) struct DecodeProfileScope {
    active: bool,
}

#[cfg(test)]
impl DecodeProfileScope {
    pub(super) fn start() -> Self {
        DECODE_PROFILE.with(|profile| {
            assert!(
                profile.borrow().is_none(),
                "decode profile is already active"
            );
            *profile.borrow_mut() = Some(DecodeProfile::default());
        });
        Self { active: true }
    }

    pub(super) fn finish(mut self) -> DecodeProfile {
        let profile = DECODE_PROFILE.with(|profile| {
            profile
                .borrow_mut()
                .take()
                .expect("decode profile scope remains active")
        });
        self.active = false;
        profile
    }
}

#[cfg(test)]
impl Drop for DecodeProfileScope {
    fn drop(&mut self) {
        if self.active {
            DECODE_PROFILE.with(|profile| {
                profile.borrow_mut().take();
            });
        }
    }
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
    /// A resident-chat request exceeds its configured or qualified limit.
    #[error("Qwen3 resident chat context {requested} exceeds maximum {maximum}")]
    ResidentChatContextLimit {
        /// Requested prompt-plus-generated token capacity.
        requested: usize,
        /// Smaller of the model and resident-chat caps.
        maximum: usize,
    },
    /// A resident-chat K/V estimate exceeds the caller's logical budget.
    #[error("Qwen3 resident chat K/V requires {required} bytes, above budget {maximum}")]
    ResidentChatKvBudget {
        /// Logical final K/V bytes required.
        required: u64,
        /// Caller-provided logical K/V ceiling.
        maximum: u64,
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
    use proptest::prelude::*;

    use crate::GPU_TEST_LOCK;

    use super::{
        Qwen3ForwardConfig, Qwen3ForwardError, Qwen3ForwardExecutor, forward_last_logits,
        forward_layer, linear, read_last_logits, rms_norm, stepped_capacity, weight,
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

    fn qwen3_4b_kv_layout() -> Qwen3ForwardConfig {
        let layout = QWEN3_06B
            .replace("\"num_hidden_layers\":28", "\"num_hidden_layers\":36")
            .replace("\"hidden_size\":1024", "\"hidden_size\":2560")
            .replace("\"intermediate_size\":3072", "\"intermediate_size\":9728")
            .replace("\"num_attention_heads\":16", "\"num_attention_heads\":32");
        Qwen3ForwardConfig::parse(&layout).expect("Qwen3-4B K/V layout")
    }

    fn independent_kv_bytes(config: &Qwen3ForwardConfig, context_tokens: usize) -> u64 {
        u64::try_from(config.hidden_layers).expect("test layer count fits u64")
            * 2
            * u64::try_from(config.key_value_heads).expect("test K/V head count fits u64")
            * u64::try_from(context_tokens).expect("test context fits u64")
            * u64::try_from(config.head_dim).expect("test head dimension fits u64")
            * u64::try_from(size_of::<f32>()).expect("f32 byte width fits u64")
    }

    fn resident_production_layouts() -> [Qwen3ForwardConfig; 2] {
        [
            Qwen3ForwardConfig::parse(QWEN3_06B).expect("Qwen3-0.6B layout"),
            qwen3_4b_kv_layout(),
        ]
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(128))]

        /// Stepped resident storage always admits the requested prefix without
        /// exceeding the configured ceiling and only uses known growth points.
        #[test]
        fn stepped_capacity_preserves_prefix_bounds(
            maximum in 1_usize..=4096,
            next in 1_usize..=4096,
        ) {
            let result = stepped_capacity(next, maximum);
            if next > maximum {
                let rejected = matches!(
                    result,
                    Err(Qwen3ForwardError::PromptTooLong { actual, maximum: limit })
                        if actual == next && limit == maximum
                );
                prop_assert!(rejected);
            } else {
                let capacity = result.expect("admitted prefix has a capacity");
                prop_assert!(capacity >= next);
                prop_assert!(capacity <= maximum);
                prop_assert!(capacity == maximum || capacity == 128 || capacity == 512);
            }
        }

        /// Every admitted production-layout context uses exactly the logical
        /// f32 K/V formula, including the exact one-byte budget boundary.
        #[test]
        fn resident_kv_admission_matches_production_layouts(
            context_tokens in 1_usize..=super::MAX_RESIDENT_CHAT_TOKENS,
        ) {
            for config in resident_production_layouts() {
                let required = independent_kv_bytes(&config, context_tokens);
                let exact = config
                    .resident_chat_plan(context_tokens, required)
                    .expect("exact K/V budget admits a valid context");
                prop_assert_eq!(exact.maximum_context_tokens(), context_tokens);
                prop_assert_eq!(exact.planned_kv_bytes(), required);
                prop_assert!(matches!(
                    config.resident_chat_plan(context_tokens, required - 1),
                    Err(Qwen3ForwardError::ResidentChatKvBudget {
                        required: actual_required,
                        maximum,
                    }) if actual_required == required && maximum == required - 1
                ), "one byte below the exact K/V requirement must fail admission");
            }
        }

        /// One additional valid token must increase the retained K/V estimate
        /// for both the 0.6B and 36-layer 4B K/V layouts.
        #[test]
        fn resident_kv_estimate_is_strictly_monotone_for_production_layouts(
            context_tokens in 1_usize..super::MAX_RESIDENT_CHAT_TOKENS,
        ) {
            for config in resident_production_layouts() {
                let current = config
                    .resident_chat_plan(context_tokens, u64::MAX)
                    .expect("valid production-layout context");
                let next = config
                    .resident_chat_plan(context_tokens + 1, u64::MAX)
                    .expect("next valid production-layout context");
                prop_assert_eq!(
                    next.planned_kv_bytes() - current.planned_kv_bytes(),
                    independent_kv_bytes(&config, 1),
                );
                prop_assert!(next.planned_kv_bytes() > current.planned_kv_bytes());
            }
        }
    }

    #[test]
    fn resident_production_layouts_reject_the_token_after_the_admission_ceiling() {
        for config in resident_production_layouts() {
            assert!(matches!(
                config.resident_chat_plan(super::MAX_RESIDENT_CHAT_TOKENS + 1, u64::MAX),
                Err(Qwen3ForwardError::ResidentChatContextLimit {
                    requested,
                    maximum,
                }) if requested == super::MAX_RESIDENT_CHAT_TOKENS + 1
                    && maximum == super::MAX_RESIDENT_CHAT_TOKENS
            ));
        }
    }

    #[test]
    fn accepts_qwen3_06b_expanded_query_width() {
        // Qwen3-0.6B deliberately has 16 * 128 = 2048 query features while
        // its residual stream is only 1024 wide; `o_proj` maps it back.
        Qwen3ForwardConfig::parse(QWEN3_06B).expect("official Qwen3-0.6B layout");
    }

    #[test]
    fn detached_kv_plan_uses_gqa_not_query_head_width() {
        let config = Qwen3ForwardConfig::parse(QWEN3_06B).expect("official Qwen3-0.6B layout");
        // 28 layers * (K + V) * 8 KV heads * 3 positions * 128 values * f32.
        assert_eq!(config.cached_kv_bytes(3).unwrap(), 688_128);
    }

    #[test]
    fn resident_chat_plan_is_separate_from_the_512_token_diagnostic_limit() {
        let config = Qwen3ForwardConfig::parse(QWEN3_06B).expect("official Qwen3-0.6B layout");
        let plan = config
            .resident_chat_plan(2_048, super::DEFAULT_RESIDENT_CHAT_KV_BUDGET_BYTES)
            .expect("512 MiB admits Qwen3-0.6B K/V at 2048 tokens");
        assert_eq!(plan.maximum_context_tokens(), 2_048);
        assert_eq!(plan.planned_kv_bytes(), 469_762_048);
        assert!(matches!(
            config.resident_chat_plan(16_385, u64::MAX),
            Err(Qwen3ForwardError::ResidentChatContextLimit {
                requested: 16_385,
                maximum: 16_384,
            })
        ));
        assert!(matches!(
            config.resident_chat_plan(2_048, plan.planned_kv_bytes() - 1),
            Err(Qwen3ForwardError::ResidentChatKvBudget {
                required: 469_762_048,
                maximum: 469_762_047,
            })
        ));
        let smaller_model = QWEN3_06B.replace(
            "\"max_position_embeddings\":40960",
            "\"max_position_embeddings\":1024",
        );
        let smaller_model = Qwen3ForwardConfig::parse(&smaller_model).expect("smaller context");
        assert!(matches!(
            smaller_model.resident_chat_plan(2_048, u64::MAX),
            Err(Qwen3ForwardError::ResidentChatContextLimit {
                requested: 2_048,
                maximum: 1_024,
            })
        ));
    }

    #[test]
    fn default_executor_still_rejects_513_tokens_before_gpu_work() {
        let config = Qwen3ForwardConfig::parse(QWEN3_06B).expect("official Qwen3-0.6B layout");
        let weights = HashMap::new();
        let mut executor = Qwen3ForwardExecutor::new(&config, &weights);
        assert!(matches!(
            executor.prefill_last_logits(&vec![0; 513]),
            Err(Qwen3ForwardError::PromptTooLong {
                actual: 513,
                maximum: 512,
            })
        ));
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

    mod cache_component_profile;
    mod capacity_cache_profile;
    mod decode_profile;
    mod particle_replay;

    #[test]
    fn resident_chat_cached_decode_matches_fresh_prefill_beyond_512_tokens() {
        let _gpu = GPU_TEST_LOCK.lock().expect("GPU test lock");
        let config = long_small_config();
        let weights = deterministic_weights();
        let plan = config
            .resident_chat_plan(600, u64::MAX)
            .expect("long tiny-model resident plan");
        let mut cached = Qwen3ForwardExecutor::new_for_resident_chat(&config, &weights, plan);
        let mut prefix = vec![1; 513];
        let _ = cached.prefill_last_logits(&prefix).expect("long prefill");
        assert_eq!(cached.maximum_context_tokens(), 600);
        assert_eq!(cached.resident_chat_plan(), Some(plan));
        assert_eq!(cached.planned_kv_bytes(), Some(plan.planned_kv_bytes()));

        for token in [2, 3, 4] {
            let cached_logits = cached.decode_last_logits(token).expect("cached decode");
            prefix.push(token);
            let mut fresh = Qwen3ForwardExecutor::new_for_resident_chat(&config, &weights, plan);
            let fresh_logits = fresh
                .prefill_last_logits(&prefix)
                .expect("fresh long prefill");
            assert_logits_match(fresh_logits, cached_logits);
        }
    }

    #[test]
    fn resident_chat_context_error_resets_cached_kv() {
        let _gpu = GPU_TEST_LOCK.lock().expect("GPU test lock");
        let config = long_small_config();
        let weights = deterministic_weights();
        let plan = config
            .resident_chat_plan(513, u64::MAX)
            .expect("exact long tiny-model resident plan");
        let mut executor = Qwen3ForwardExecutor::new_for_resident_chat(&config, &weights, plan);
        executor
            .prefill_last_logits(&vec![1; 513])
            .expect("exact-limit prefill");
        assert!(matches!(
            executor.decode_last_logits(2),
            Err(Qwen3ForwardError::PromptTooLong {
                actual: 514,
                maximum: 513,
            })
        ));
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

    fn long_small_config() -> Qwen3ForwardConfig {
        Qwen3ForwardConfig::parse(
            r#"{
              "model_type":"qwen3",
              "num_hidden_layers":1,
              "hidden_size":4,
              "intermediate_size":8,
              "vocab_size":8,
              "num_attention_heads":2,
              "num_key_value_heads":1,
              "head_dim":4,
              "max_position_embeddings":1024,
              "rms_norm_eps":0.000001,
              "rope_theta":1000000,
              "hidden_act":"silu",
              "tie_word_embeddings":true,
              "attention_bias":false,
              "mlp_bias":false
            }"#,
        )
        .expect("long tiny-model config")
    }

    fn small_dense_config() -> Qwen3ForwardConfig {
        Qwen3ForwardConfig::parse(
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
        .expect("small dense Qwen3 config")
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
