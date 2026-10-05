//! Cached Gemma 4 text forward on Metal.
//!
//! One executor owns the K/V of one sequence. Sliding layers keep only the
//! `sliding_window - 1` most recent positions after each append, which is all
//! the next query can see; full layers keep every position. The numerics
//! follow the list in the crate documentation.

#![allow(
    deprecated,
    reason = "mlx-rs 0.32 deprecates the *_device ops; the with_stream migration is a separate change"
)]

use mlx_rs::{Array, StreamOrDevice, fast, ops, ops::indexing::IndexOp};
use thiserror::Error;

use crate::{Gemma4LayerKind, metal::Gemma4MlxWeights};

/// Longest chunk one prefill graph processes. Longer prompts are appended in
/// chunks, which bounds the `[heads, chunk, keys]` attention scores that the
/// unfused 256- and 512-wide attention materializes.
pub const PREFILL_CHUNK_TOKENS: usize = 2048;

/// K/V retained by one layer and the absolute position of its first row.
struct LayerKv {
    keys: Array,
    values: Array,
    start: usize,
}

/// A Gemma 4 decoder sequence bound to one set of loaded weights.
pub struct Gemma4Executor<'a> {
    weights: &'a Gemma4MlxWeights,
    cache: Vec<Option<LayerKv>>,
    cached_tokens: usize,
    maximum_context_tokens: usize,
    prefill_chunk_tokens: usize,
}

impl<'a> Gemma4Executor<'a> {
    pub(crate) fn new(
        weights: &'a Gemma4MlxWeights,
        maximum_context_tokens: usize,
    ) -> Result<Self, Gemma4ForwardError> {
        let maximum = weights.config().max_position_embeddings();
        if maximum_context_tokens == 0 || maximum_context_tokens > maximum {
            return Err(Gemma4ForwardError::ContextLimit {
                actual: maximum_context_tokens,
                maximum,
            });
        }
        Ok(Self {
            weights,
            cache: (0..weights.config().hidden_layers())
                .map(|_| None)
                .collect(),
            cached_tokens: 0,
            maximum_context_tokens,
            prefill_chunk_tokens: PREFILL_CHUNK_TOKENS,
        })
    }

    /// Clears the sequence.
    pub fn reset(&mut self) {
        self.cache.iter_mut().for_each(|entry| *entry = None);
        self.cached_tokens = 0;
    }

    /// Positions consumed so far.
    #[must_use]
    pub const fn cached_tokens(&self) -> usize {
        self.cached_tokens
    }

    /// Logical bytes of the retained K/V arrays, excluding allocator overhead.
    ///
    /// A trimmed sliding layer is a view of the buffer its last append built,
    /// so until the next append that buffer can hold up to
    /// `sliding_window - 1 + PREFILL_CHUNK_TOKENS` positions.
    #[must_use]
    pub fn kv_bytes(&self) -> usize {
        self.cache
            .iter()
            .flatten()
            .map(|layer| layer.keys.nbytes() + layer.values.nbytes())
            .sum()
    }

    /// Starts a new sequence and returns the logits after its last token.
    ///
    /// # Errors
    ///
    /// Returns [`Gemma4ForwardError`] for empty or out-of-range input, a
    /// context overrun, or an MLX failure; the sequence is then cleared.
    pub fn prefill_last_logits(
        &mut self,
        input_ids: &[i32],
    ) -> Result<Vec<f32>, Gemma4ForwardError> {
        self.reset();
        self.extend(input_ids)
    }

    /// Appends one token and returns its logits.
    ///
    /// # Errors
    ///
    /// As [`Self::prefill_last_logits`], and when nothing was prefilled.
    pub fn decode_last_logits(&mut self, input_id: i32) -> Result<Vec<f32>, Gemma4ForwardError> {
        self.extend_last_logits(&[input_id])
    }

    /// Appends several tokens to a prefilled sequence and returns the logits
    /// after the last one.
    ///
    /// # Errors
    ///
    /// As [`Self::decode_last_logits`].
    pub fn extend_last_logits(
        &mut self,
        input_ids: &[i32],
    ) -> Result<Vec<f32>, Gemma4ForwardError> {
        if self.cached_tokens == 0 {
            return Err(Gemma4ForwardError::DecodeWithoutPrefill);
        }
        self.extend(input_ids)
    }

    #[cfg(test)]
    pub(crate) fn set_prefill_chunk_tokens(&mut self, tokens: usize) {
        self.prefill_chunk_tokens = tokens.max(1);
    }

    fn extend(&mut self, input_ids: &[i32]) -> Result<Vec<f32>, Gemma4ForwardError> {
        let result = self.validate(input_ids).and_then(|()| {
            let mut logits = Vec::new();
            for chunk in input_ids.chunks(self.prefill_chunk_tokens) {
                logits = self.append(chunk)?;
            }
            Ok(logits)
        });
        if result.is_err() {
            // A failed graph may have updated only some layers.
            self.reset();
        }
        result
    }

    fn validate(&self, input_ids: &[i32]) -> Result<(), Gemma4ForwardError> {
        if input_ids.is_empty() {
            return Err(Gemma4ForwardError::EmptyInput);
        }
        let total = self
            .cached_tokens
            .checked_add(input_ids.len())
            .ok_or(Gemma4ForwardError::ShapeOverflow)?;
        if total > self.maximum_context_tokens {
            return Err(Gemma4ForwardError::ContextLimit {
                actual: total,
                maximum: self.maximum_context_tokens,
            });
        }
        let vocab_size = self.weights.config().vocab_size();
        if let Some(&token_id) = input_ids
            .iter()
            .find(|&&id| usize::try_from(id).map_or(true, |id| id >= vocab_size))
        {
            return Err(Gemma4ForwardError::InvalidTokenId {
                token_id,
                vocab_size,
            });
        }
        Ok(())
    }

    fn append(&mut self, input_ids: &[i32]) -> Result<Vec<f32>, Gemma4ForwardError> {
        let stream = StreamOrDevice::gpu();
        let weights = self.weights;
        let config = weights.config();
        let length = as_i32(input_ids.len())?;
        let hidden = as_i32(config.hidden_size())?;
        let embedding = weights.tensor("embed_tokens.weight")?;
        let mut hidden_states = embedding
            .take_axis_device(Array::from_slice(input_ids, &[length]), 0, &stream)?
            .reshape_device(&[1, length, hidden], &stream)?
            .multiply_device(&weights.embed_scale, &stream)?;
        for (layer, &kind) in config.layers().iter().enumerate() {
            let cache = self
                .cache
                .get_mut(layer)
                .ok_or(Gemma4ForwardError::CacheInconsistent)?;
            hidden_states = decoder_layer(
                weights,
                layer,
                kind,
                cache,
                &hidden_states,
                self.cached_tokens,
            )?;
        }
        self.cached_tokens += input_ids.len();

        let last = hidden_states.index_device((.., length - 1.., ..), &stream);
        let normalized = rms_norm(&last, weights.tensor("norm.weight")?, config.rms_norm_eps())?;
        let mut logits = linear(&normalized, embedding)?
            .reshape_device(&[as_i32(config.vocab_size())?], &stream)?
            .as_type_device::<f32>(&stream)?;
        if let Some(cap) = config.final_logit_softcapping() {
            let cap = Array::from_f32(cap);
            logits = ops::tanh_device(logits.divide_device(&cap, &stream)?, &stream)?
                .multiply_device(&cap, &stream)?;
        }
        logits.eval()?;
        Ok(logits.as_slice::<f32>().to_vec())
    }
}

fn decoder_layer(
    weights: &Gemma4MlxWeights,
    layer: usize,
    kind: Gemma4LayerKind,
    cache: &mut Option<LayerKv>,
    hidden_states: &Array,
    position: usize,
) -> Result<Array, Gemma4ForwardError> {
    let stream = StreamOrDevice::gpu();
    let eps = weights.config().rms_norm_eps();
    let base = format!("layers.{layer}");
    let norm = |name: &str, input: &Array| {
        rms_norm(
            input,
            weights.tensor(&format!("{base}.{name}.weight"))?,
            eps,
        )
    };

    let attention_input = norm("input_layernorm", hidden_states)?;
    let attention = attention(weights, &base, kind, cache, &attention_input, position)?;
    let residual =
        hidden_states.add_device(norm("post_attention_layernorm", &attention)?, &stream)?;

    let mlp_input = norm("pre_feedforward_layernorm", &residual)?;
    let gate = linear(
        &mlp_input,
        weights.tensor(&format!("{base}.mlp.gate_proj.weight"))?,
    )?;
    let up = linear(
        &mlp_input,
        weights.tensor(&format!("{base}.mlp.up_proj.weight"))?,
    )?;
    let mlp = linear(
        &gelu_tanh(&gate)?.multiply_device(&up, &stream)?,
        weights.tensor(&format!("{base}.mlp.down_proj.weight"))?,
    )?;
    let output = residual.add_device(norm("post_feedforward_layernorm", &mlp)?, &stream)?;
    // The source scales the whole layer output, residual included.
    output
        .multiply_device(weights.tensor(&format!("{base}.layer_scalar"))?, &stream)
        .map_err(Into::into)
}

fn attention(
    weights: &Gemma4MlxWeights,
    base: &str,
    kind: Gemma4LayerKind,
    cache: &mut Option<LayerKv>,
    input: &Array,
    position: usize,
) -> Result<Array, Gemma4ForwardError> {
    let stream = StreamOrDevice::gpu();
    let config = weights.config();
    let geometry = config.attention(kind);
    let eps = config.rms_norm_eps();
    let length = input.shape()[1];
    let heads = as_i32(geometry.heads)?;
    let kv_heads = as_i32(geometry.kv_heads)?;
    let head_dim = as_i32(geometry.head_dim)?;
    let offset = as_i32(position)?;
    let attn = format!("{base}.self_attn");
    let (wavelengths, value_norm) = match kind {
        Gemma4LayerKind::Sliding => (&weights.sliding_wavelengths, &weights.sliding_value_norm),
        Gemma4LayerKind::Full => (&weights.full_wavelengths, &weights.full_value_norm),
    };
    let rope = |input: &Array| -> Result<Array, Gemma4ForwardError> {
        let base = match geometry.rope {
            crate::Gemma4Rope::Default { theta } => Some(theta),
            crate::Gemma4Rope::Proportional { .. } => None,
        };
        Ok(fast::rope_device(
            input,
            head_dim,
            false,
            base,
            1.0,
            offset,
            wavelengths.as_ref(),
            &stream,
        )?)
    };

    let query = linear(input, weights.tensor(&format!("{attn}.q_proj.weight"))?)?
        .reshape_device(&[1, length, heads, head_dim], &stream)?;
    let query = rope(
        &rms_norm(
            &query,
            weights.tensor(&format!("{attn}.q_norm.weight"))?,
            eps,
        )?
        .transpose_axes_device(&[0, 2, 1, 3], &stream)?,
    )?;
    let raw_key = linear(input, weights.tensor(&format!("{attn}.k_proj.weight"))?)?
        .reshape_device(&[1, length, kv_heads, head_dim], &stream)?;
    let raw_value = if geometry.value_from_key {
        raw_key.clone()
    } else {
        linear(input, weights.tensor(&format!("{attn}.v_proj.weight"))?)?
            .reshape_device(&[1, length, kv_heads, head_dim], &stream)?
    };
    let key = rope(
        &rms_norm(
            &raw_key,
            weights.tensor(&format!("{attn}.k_norm.weight"))?,
            eps,
        )?
        .transpose_axes_device(&[0, 2, 1, 3], &stream)?,
    )?;
    let value =
        rms_norm(&raw_value, value_norm, eps)?.transpose_axes_device(&[0, 2, 1, 3], &stream)?;

    let (keys, values, start) = append(cache.take(), key, value, position, &stream)?;
    let window = match kind {
        Gemma4LayerKind::Sliding => Some(config.sliding_window()),
        Gemma4LayerKind::Full => None,
    };
    let mask = attention_mask(start, position, length, window, &stream)?;
    let output = fast::scaled_dot_product_attention_device(
        &query,
        &keys,
        &values,
        1.0,
        mask.as_ref()
            .map(fast::ScaledDotProductAttentionMask::Array),
        Option::<&Array>::None,
        &stream,
    )?
    .transpose_axes_device(&[0, 2, 1, 3], &stream)?
    .reshape_device(
        &[
            1,
            length,
            heads
                .checked_mul(head_dim)
                .ok_or(Gemma4ForwardError::ShapeOverflow)?,
        ],
        &stream,
    )?;

    *cache = Some(retain(
        keys,
        values,
        start,
        position + usize::try_from(length).map_err(|_| Gemma4ForwardError::ShapeOverflow)?,
        window,
        &stream,
    )?);
    linear(&output, weights.tensor(&format!("{attn}.o_proj.weight"))?)
}

/// The layer's K/V with this chunk's rows appended, and the absolute
/// position of the first row.
fn append(
    previous: Option<LayerKv>,
    key: Array,
    value: Array,
    position: usize,
    stream: &StreamOrDevice,
) -> Result<(Array, Array, usize), Gemma4ForwardError> {
    Ok(match previous {
        Some(previous) => (
            ops::concatenate_axis_device(&[&previous.keys, &key], 2, stream)?,
            ops::concatenate_axis_device(&[&previous.values, &value], 2, stream)?,
            previous.start,
        ),
        None => (key, value, position),
    })
}

/// Keeps what the next query can see: for a sliding layer, the last
/// `window - 1` positions before `end`.
fn retain(
    keys: Array,
    values: Array,
    start: usize,
    end: usize,
    window: Option<usize>,
    stream: &StreamOrDevice,
) -> Result<LayerKv, Gemma4ForwardError> {
    let Some(window) = window else {
        return Ok(LayerKv {
            keys,
            values,
            start,
        });
    };
    let keep_from = end.saturating_sub(window - 1).max(start);
    if keep_from == start {
        return Ok(LayerKv {
            keys,
            values,
            start,
        });
    }
    let drop = as_i32(keep_from - start)?;
    Ok(LayerKv {
        keys: keys.index_device((.., .., drop.., ..), stream),
        values: values.index_device((.., .., drop.., ..), stream),
        start: keep_from,
    })
}

/// `[queries, keys]` visibility for queries at absolute positions
/// `position..position + queries` and keys at `start..position + queries`:
/// key `k` is visible to query `q` when `k <= q` and, in a sliding layer,
/// `q - k < window`. `None` when every key is visible to the only query.
fn attention_mask(
    start: usize,
    position: usize,
    queries: i32,
    window: Option<usize>,
    stream: &StreamOrDevice,
) -> Result<Option<Array>, Gemma4ForwardError> {
    let first_query = as_i32(position)?;
    let end = first_query
        .checked_add(queries)
        .ok_or(Gemma4ForwardError::ShapeOverflow)?;
    let first_key = as_i32(start)?;
    let all_keys_in_window = window
        .is_none_or(|window| usize::try_from(end - first_key).is_ok_and(|keys| keys <= window));
    if queries == 1 && all_keys_in_window {
        return Ok(None);
    }
    let query_positions = Array::arange_device::<i32, i32>(first_query, end, None, stream)?
        .reshape_device(&[queries, 1], stream)?;
    let key_positions = Array::arange_device::<i32, i32>(first_key, end, None, stream)?
        .reshape_device(&[1, end - first_key], stream)?;
    let causal = query_positions.ge_device(&key_positions, stream)?;
    let Some(window) = window else {
        return Ok(Some(causal));
    };
    let distance = query_positions.subtract_device(&key_positions, stream)?;
    let in_window = distance.lt_device(Array::from_int(as_i32(window)?), stream)?;
    Ok(Some(causal.logical_and_device(&in_window, stream)?))
}

/// `0.5 x (1 + tanh(sqrt(2 / pi) (x + 0.044715 x^3)))` in the input dtype.
fn gelu_tanh(x: &Array) -> Result<Array, Gemma4ForwardError> {
    let stream = StreamOrDevice::gpu();
    let dtype = x.dtype();
    let scalar = |value: f32| Array::from_f32(value).as_dtype_device(dtype, &stream);
    let cubic = x
        .multiply_device(x, &stream)?
        .multiply_device(x, &stream)?
        .multiply_device(scalar(0.044_715)?, &stream)?;
    let inner = x
        .add_device(&cubic, &stream)?
        .multiply_device(scalar((2.0 / std::f32::consts::PI).sqrt())?, &stream)?;
    let gate = ops::tanh_device(&inner, &stream)?.add_device(scalar(1.0)?, &stream)?;
    Ok(x.multiply_device(scalar(0.5)?, &stream)?
        .multiply_device(&gate, &stream)?)
}

fn rms_norm(input: &Array, weight: &Array, eps: f32) -> Result<Array, Gemma4ForwardError> {
    Ok(fast::rms_norm_device(
        input,
        weight,
        eps,
        StreamOrDevice::gpu(),
    )?)
}

fn linear(input: &Array, weight: &Array) -> Result<Array, Gemma4ForwardError> {
    let stream = StreamOrDevice::gpu();
    Ok(input.matmul_device(weight.transpose_device(&stream)?, &stream)?)
}

fn as_i32(value: usize) -> Result<i32, Gemma4ForwardError> {
    i32::try_from(value).map_err(|_| Gemma4ForwardError::ShapeOverflow)
}

/// A failed Gemma 4 forward step.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum Gemma4ForwardError {
    /// MLX failed to build or evaluate the graph.
    #[error("MLX failed: {0}")]
    Mlx(#[from] mlx_rs::error::Exception),
    /// A tensor the forward pass needs is absent.
    #[error("missing weight {0}")]
    MissingWeight(String),
    /// No tokens were supplied.
    #[error("input must contain at least one token")]
    EmptyInput,
    /// A token ID is outside the vocabulary.
    #[error("token ID {token_id} is outside the {vocab_size}-token vocabulary")]
    InvalidTokenId {
        /// The rejected ID.
        token_id: i32,
        /// Vocabulary size.
        vocab_size: usize,
    },
    /// The sequence would exceed the executor's context limit.
    #[error("sequence of {actual} tokens exceeds the {maximum}-token context")]
    ContextLimit {
        /// Requested length.
        actual: usize,
        /// Admitted length.
        maximum: usize,
    },
    /// Decode or extend before any prefill.
    #[error("decode requires a prefilled sequence")]
    DecodeWithoutPrefill,
    /// The cache does not match the layer list.
    #[error("Gemma 4 K/V cache is inconsistent with the layer list")]
    CacheInconsistent,
    /// A dimension does not fit MLX's 32-bit shapes.
    #[error("shape exceeds MLX's 32-bit dimensions")]
    ShapeOverflow,
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use mlx_rs::{Array, StreamOrDevice};

    use super::{Gemma4ForwardError, attention_mask};
    use crate::{
        Gemma4TextConfig,
        checkpoint::{expected_text_tensors, tests::TINY_CONFIG},
        metal::{Gemma4MlxWeights, Gemma4Precision},
    };

    const WINDOW: usize = 3;

    /// Deterministic weights for the two-layer test layout. Norm scales sit
    /// near 1 and `layer_scalar` below 1, so a missing scale changes logits.
    fn weights(silence_full_attention: bool) -> Gemma4MlxWeights {
        let config = Gemma4TextConfig::parse(TINY_CONFIG).expect("tiny config");
        assert_eq!(config.sliding_window(), WINDOW);
        let mut state = 0x2545_f491_u32;
        let mut next = move || {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            f32::from(u16::try_from(state >> 16).expect("16 bits")) / 65_536.0 - 0.5
        };
        let tensors: HashMap<String, Array> = expected_text_tensors(&config)
            .into_iter()
            .map(|(name, shape)| {
                let count = shape.iter().product::<usize>();
                let values: Vec<f32> = if name.ends_with("layer_scalar") {
                    vec![0.75]
                } else if name.contains("norm") {
                    (0..count).map(|_| 1.0 + 0.2 * next()).collect()
                } else if silence_full_attention && name == "layers.1.self_attn.o_proj.weight" {
                    vec![0.0; count]
                } else {
                    (0..count).map(|_| next()).collect()
                };
                let shape: Vec<i32> = shape
                    .iter()
                    .map(|&d| i32::try_from(d).expect("small"))
                    .collect();
                (name, Array::from_slice(&values, &shape))
            })
            .collect();
        Gemma4MlxWeights::from_tensors(config, tensors, Gemma4Precision::Float32).expect("weights")
    }

    fn assert_close(left: &[f32], right: &[f32]) {
        assert_eq!(left.len(), right.len());
        let worst = left
            .iter()
            .zip(right)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        assert!(worst <= 1e-4, "max abs logit difference {worst}");
    }

    const IDS: [i32; 9] = [3, 14, 1, 5, 9, 2, 6, 5, 3];

    #[test]
    fn cached_decode_and_chunked_prefill_match_one_prefill() {
        let _gpu = crate::GPU_TEST_LOCK.lock().expect("GPU lock");
        let weights = weights(false);
        let mut whole = weights.executor(32).expect("executor");
        let expected = whole.prefill_last_logits(&IDS).expect("prefill");

        // Decode crosses the window: positions 4..8 each drop an older key.
        let mut stepped = weights.executor(32).expect("executor");
        stepped.prefill_last_logits(&IDS[..4]).expect("prefix");
        let mut decoded = Vec::new();
        for &id in &IDS[4..] {
            decoded = stepped.decode_last_logits(id).expect("decode");
        }
        assert_close(&decoded, &expected);

        let mut chunked = weights.executor(32).expect("executor");
        chunked.set_prefill_chunk_tokens(2);
        assert_close(
            &chunked.prefill_last_logits(&IDS).expect("chunks"),
            &expected,
        );
    }

    #[test]
    fn sliding_layers_retain_only_what_the_next_query_sees() {
        let _gpu = crate::GPU_TEST_LOCK.lock().expect("GPU lock");
        let weights = weights(false);
        let mut executor = weights.executor(32).expect("executor");
        executor.prefill_last_logits(&IDS).expect("prefill");
        let retained = weights.config().retained_kv_elements(IDS.len());
        assert_eq!(executor.kv_bytes() as u128, retained * 4);
        executor.decode_last_logits(7).expect("decode");
        assert_eq!(
            executor.kv_bytes() as u128,
            weights.config().retained_kv_elements(IDS.len() + 1) * 4
        );
    }

    /// With the full layer's attention output zeroed, the last position sees
    /// only the sliding window: `q - window < k <= q` in the source mask.
    #[test]
    fn a_token_leaves_the_receptive_field_exactly_window_positions_later() {
        let _gpu = crate::GPU_TEST_LOCK.lock().expect("GPU lock");
        let weights = weights(true);
        let mut executor = weights.executor(32).expect("executor");
        let mut logits = |ids: &[i32]| executor.prefill_last_logits(ids).expect("prefill");
        let changed_first = |ids: &[i32]| {
            let mut changed = ids.to_vec();
            changed[0] = 11;
            changed
        };

        let outside = &IDS[..=WINDOW];
        assert_close(&logits(outside), &logits(&changed_first(outside)));

        let inside = &IDS[..WINDOW];
        let difference = logits(inside)
            .iter()
            .zip(logits(&changed_first(inside)))
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        assert!(
            difference > 1e-3,
            "token 0 must still be visible: {difference}"
        );
    }

    #[test]
    fn masks_follow_the_source_window_predicate() {
        let _gpu = crate::GPU_TEST_LOCK.lock().expect("GPU lock");
        let stream = StreamOrDevice::gpu();
        // Queries 6..9 over a cache that starts at 4.
        let (start, position, queries) = (4_i32, 6_i32, 3_i32);
        for window in [Some(WINDOW), None] {
            let mask = attention_mask(4, 6, queries, window, &stream)
                .expect("mask")
                .expect("multi-query masks are explicit");
            mask.eval().expect("eval");
            let visible = mask.as_slice::<bool>();
            let keys = position + queries - start;
            for q in position..position + queries {
                for k in start..position + queries {
                    let index =
                        usize::try_from((q - position) * keys + (k - start)).expect("index");
                    let expected =
                        k <= q && window.is_none_or(|w| k > q - i32::try_from(w).expect("small"));
                    assert_eq!(visible[index], expected, "q={q} k={k} window={window:?}");
                }
            }
        }
        // One query whose keys all fit the window needs no mask.
        assert!(
            attention_mask(5, 7, 1, Some(WINDOW), &stream)
                .expect("mask")
                .is_none()
        );
        assert!(
            attention_mask(4, 7, 1, Some(WINDOW), &stream)
                .expect("mask")
                .is_some()
        );
    }

    #[test]
    fn rejects_bad_input_and_clears_the_sequence() {
        let _gpu = crate::GPU_TEST_LOCK.lock().expect("GPU lock");
        let weights = weights(false);
        let mut executor = weights.executor(8).expect("executor");
        assert!(matches!(
            executor.decode_last_logits(1),
            Err(Gemma4ForwardError::DecodeWithoutPrefill)
        ));
        executor.prefill_last_logits(&IDS[..4]).expect("prefill");
        assert!(matches!(
            executor.decode_last_logits(16),
            Err(Gemma4ForwardError::InvalidTokenId { token_id: 16, .. })
        ));
        assert_eq!(executor.cached_tokens(), 0);
        assert!(matches!(
            executor.prefill_last_logits(&IDS),
            Err(Gemma4ForwardError::ContextLimit {
                actual: 9,
                maximum: 8
            })
        ));
        assert!(weights.executor(65).is_err());
    }
}
