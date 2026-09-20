//! Test-only fixed-capacity K/V feasibility probe for Qwen3-0.6B.
//!
//! This is a numerical and timing comparison with exact-length concatenation,
//! not a serving-cache replacement or a claim that MLX donates buffers.

use std::{
    collections::HashMap,
    env,
    path::PathBuf,
    time::{Duration, Instant},
};

use mlx_rs::{
    Array, StreamOrDevice, fast, ops,
    ops::indexing::{IndexMutOp, IndexOp},
};
use proptest::prelude::*;

use crate::{GPU_TEST_LOCK, metal::Qwen3MlxWeights};

use super::super::{
    Qwen3ForwardConfig, Qwen3ForwardError, Qwen3ForwardExecutor, as_i32, attention_scale,
    forward_last_logits, linear, read_last_logits, rms_norm, weight,
};

const MAXIMUM_CONTEXT_TOKENS: usize = 2_048;
const MAXIMUM_KV_BYTES: u64 = 512 * 1024 * 1024;
const DECODE_STEPS: usize = 64;
const MEASURED_ROWS: usize = 3;
const QWEN3_06B_LAYERS: usize = 28;
const QWEN3_06B_HIDDEN_SIZE: usize = 1_024;
const QWEN3_06B_INTERMEDIATE_SIZE: usize = 3_072;
const QWEN3_06B_VOCAB: usize = 151_936;
const QWEN3_06B_ATTENTION_HEADS: usize = 16;
const QWEN3_06B_KEY_VALUE_HEADS: usize = 8;
const QWEN3_06B_HEAD_DIM: usize = 128;
const QWEN3_06B_MAX_POSITIONS: usize = 40_960;

type CapacitySnapshot = Vec<(Vec<i32>, Vec<f32>, Vec<i32>, Vec<f32>)>;

struct CapacityLayerKv {
    keys: Array,
    values: Array,
}

struct CapacityExecutor<'a> {
    config: &'a Qwen3ForwardConfig,
    weights: &'a HashMap<String, Array>,
    cache: Vec<Option<CapacityLayerKv>>,
    cached_tokens: usize,
    capacity: usize,
}

impl<'a> CapacityExecutor<'a> {
    fn new(
        config: &'a Qwen3ForwardConfig,
        weights: &'a HashMap<String, Array>,
        capacity: usize,
    ) -> Self {
        Self {
            config,
            weights,
            cache: (0..config.hidden_layers).map(|_| None).collect(),
            cached_tokens: 0,
            capacity,
        }
    }

    fn prefill_last_logits(&mut self, input_ids: &[i32]) -> Result<Vec<f32>, Qwen3ForwardError> {
        self.cache.iter_mut().for_each(|entry| *entry = None);
        self.cached_tokens = 0;
        self.append(input_ids)
    }

    fn decode_last_logits(&mut self, input_id: i32) -> Result<Vec<f32>, Qwen3ForwardError> {
        if self.cached_tokens == 0 {
            return Err(Qwen3ForwardError::DecodeWithoutPrefill);
        }
        self.append(&[input_id])
    }

    fn append(&mut self, input_ids: &[i32]) -> Result<Vec<f32>, Qwen3ForwardError> {
        if input_ids.is_empty() {
            return Err(Qwen3ForwardError::EmptyInput);
        }
        let next_tokens = self
            .cached_tokens
            .checked_add(input_ids.len())
            .ok_or(Qwen3ForwardError::ShapeOverflow)?;
        if next_tokens > self.capacity {
            return Err(Qwen3ForwardError::PromptTooLong {
                actual: next_tokens,
                maximum: self.capacity,
            });
        }
        for &id in input_ids {
            if id < 0
                || usize::try_from(id)
                    .ok()
                    .is_none_or(|id| id >= self.config.vocab_size)
            {
                return Err(Qwen3ForwardError::InvalidTokenId {
                    token_id: id,
                    vocab_size: self.config.vocab_size,
                });
            }
        }

        let stream = StreamOrDevice::gpu();
        let seq_len = as_i32(input_ids.len())?;
        let hidden = as_i32(self.config.hidden_size)?;
        let ids = Array::from_slice(input_ids, &[seq_len]);
        let embedding = weight(self.weights, "model.embed_tokens.weight")?;
        let mut hidden_states = embedding
            .take_axis_device(&ids, 0, &stream)?
            .reshape_device(&[1, seq_len, hidden], &stream)?;
        let offset = as_i32(self.cached_tokens)?;
        for layer in 0..self.config.hidden_layers {
            hidden_states = capacity_layer(
                self.config,
                self.weights,
                layer,
                self.cache
                    .get_mut(layer)
                    .ok_or(Qwen3ForwardError::CacheInconsistent)?,
                &hidden_states,
                seq_len,
                offset,
                as_i32(self.capacity)?,
            )?;
        }
        self.cached_tokens = next_tokens;
        let last_hidden =
            hidden_states.take_axis_device(Array::from_slice(&[seq_len - 1], &[1]), 1, &stream)?;
        let normalized = rms_norm(
            &last_hidden,
            weight(self.weights, "model.norm.weight")?,
            self.config.rms_norm_eps,
        )?;
        read_last_logits(&linear(&normalized, embedding)?, 1, self.config.vocab_size)
    }

    fn capacity_kv_bytes(&self) -> usize {
        self.cache
            .iter()
            .flatten()
            .map(|layer| layer.keys.nbytes() + layer.values.nbytes())
            .sum()
    }

    fn fork_prefilled(&self) -> Result<Self, Qwen3ForwardError> {
        if self.cached_tokens == 0 {
            return Err(Qwen3ForwardError::DecodeWithoutPrefill);
        }
        let mut fork = Self::new(self.config, self.weights, self.capacity);
        fork.cached_tokens = self.cached_tokens;
        let stream = StreamOrDevice::gpu();
        let valid = as_i32(self.cached_tokens)?;
        let capacity = as_i32(self.capacity)?;
        for (source, destination) in self.cache.iter().zip(fork.cache.iter_mut()) {
            let source = source
                .as_ref()
                .ok_or(Qwen3ForwardError::CacheInconsistent)?;
            let key_prefix = source.keys.index_device((.., .., 0..valid, ..), &stream);
            let value_prefix = source.values.index_device((.., .., 0..valid, ..), &stream);
            let shape = source.keys.shape();
            if shape[2] != capacity || source.values.shape() != shape {
                return Err(Qwen3ForwardError::CacheInconsistent);
            }
            let key_storage = Array::zeros_device::<f32>(shape, &stream)?;
            let value_storage = Array::zeros_device::<f32>(shape, &stream)?;
            let mut keys = key_storage;
            let mut values = value_storage;
            keys.index_mut_device((.., .., 0..valid, ..), &key_prefix, &stream);
            values.index_mut_device((.., .., 0..valid, ..), &value_prefix, &stream);
            keys.eval()?;
            values.eval()?;
            *destination = Some(CapacityLayerKv { keys, values });
        }
        Ok(fork)
    }
}

fn capacity_snapshot(executor: &CapacityExecutor<'_>) -> CapacitySnapshot {
    executor
        .cache
        .iter()
        .map(|entry| {
            let layer = entry.as_ref().expect("complete capacity cache");
            layer.keys.eval().expect("materialized capacity key cache");
            layer
                .values
                .eval()
                .expect("materialized capacity value cache");
            (
                layer.keys.shape().to_vec(),
                layer.keys.as_slice::<f32>().to_vec(),
                layer.values.shape().to_vec(),
                layer.values.as_slice::<f32>().to_vec(),
            )
        })
        .collect()
}

#[allow(
    clippy::too_many_arguments,
    reason = "the test-only helper mirrors the production cached-layer boundary"
)]
fn capacity_layer(
    config: &Qwen3ForwardConfig,
    weights: &HashMap<String, Array>,
    layer: usize,
    cache: &mut Option<CapacityLayerKv>,
    hidden_states: &Array,
    seq_len: i32,
    offset: i32,
    capacity: i32,
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
    let attention = capacity_attention(
        config,
        weights,
        cache,
        &base,
        &attention_input,
        seq_len,
        offset,
        capacity,
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

#[allow(
    clippy::too_many_arguments,
    reason = "the test-only helper mirrors the production cached-layer boundary"
)]
fn capacity_attention(
    config: &Qwen3ForwardConfig,
    weights: &HashMap<String, Array>,
    cache: &mut Option<CapacityLayerKv>,
    base: &str,
    input: &Array,
    seq_len: i32,
    offset: i32,
    capacity: i32,
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
        offset,
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
        offset,
        Option::<&Array>::None,
        &stream,
    )?;
    let value = value.transpose_axes_device(&[0, 2, 1, 3], &stream)?;
    let next_tokens = offset
        .checked_add(seq_len)
        .ok_or(Qwen3ForwardError::ShapeOverflow)?;
    if next_tokens > capacity {
        return Err(Qwen3ForwardError::PromptTooLong {
            actual: usize::try_from(next_tokens).map_err(|_| Qwen3ForwardError::ShapeOverflow)?,
            maximum: usize::try_from(capacity).map_err(|_| Qwen3ForwardError::ShapeOverflow)?,
        });
    }
    let (mut keys, mut values, causal) = match cache.take() {
        Some(previous) => (previous.keys, previous.values, false),
        None => (
            Array::zeros_device::<f32>(&[1, kv_heads, capacity, head_dim], &stream)?,
            Array::zeros_device::<f32>(&[1, kv_heads, capacity, head_dim], &stream)?,
            true,
        ),
    };
    keys.index_mut_device((.., .., offset..next_tokens, ..), &key, &stream);
    values.index_mut_device((.., .., offset..next_tokens, ..), &value, &stream);
    let key_prefix = keys.index_device((.., .., 0..next_tokens, ..), &stream);
    let value_prefix = values.index_device((.., .., 0..next_tokens, ..), &stream);
    let output = if causal {
        fast::scaled_dot_product_attention_device(
            &query,
            &key_prefix,
            &value_prefix,
            attention_scale(config)?,
            Some(fast::ScaledDotProductAttentionMask::Causal),
            &stream,
        )?
    } else {
        fast::scaled_dot_product_attention_device(
            &query,
            &key_prefix,
            &value_prefix,
            attention_scale(config)?,
            None::<fast::ScaledDotProductAttentionMask<'_>>,
            &stream,
        )?
    };
    *cache = Some(CapacityLayerKv { keys, values });
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

#[derive(Debug)]
struct Row {
    concat_decode: Duration,
    capacity_decode: Duration,
    concat_logits_bits: Vec<u32>,
    capacity_logits_bits: Vec<u32>,
    concat_trace_fnv1a64: u64,
    capacity_trace_fnv1a64: u64,
    concat_kv_bytes: usize,
    capacity_kv_bytes: usize,
}

fn fixed_tokens(count: usize, offset: usize) -> Vec<i32> {
    (0..count)
        .map(|index| {
            let value = ((index * 7_919) + offset) % 151_000 + 1;
            i32::try_from(value).expect("bounded Qwen3 token ID")
        })
        .collect()
}

fn assert_qwen3_06b_tied_embedding_layout(
    config: &Qwen3ForwardConfig,
    weights: &HashMap<String, Array>,
) {
    assert_eq!(config.hidden_layers, QWEN3_06B_LAYERS);
    assert_eq!(config.hidden_size, QWEN3_06B_HIDDEN_SIZE);
    assert_eq!(config.intermediate_size, QWEN3_06B_INTERMEDIATE_SIZE);
    assert_eq!(config.vocab_size, QWEN3_06B_VOCAB);
    assert_eq!(config.attention_heads, QWEN3_06B_ATTENTION_HEADS);
    assert_eq!(config.key_value_heads, QWEN3_06B_KEY_VALUE_HEADS);
    assert_eq!(config.head_dim, QWEN3_06B_HEAD_DIM);
    assert_eq!(config.max_position_embeddings, QWEN3_06B_MAX_POSITIONS);
    // `Qwen3ForwardConfig::parse` rejects untied output embeddings. Both
    // branches deliberately use this same embedding tensor as the output head.
    assert_eq!(
        weight(weights, "model.embed_tokens.weight")
            .expect("tied embedding tensor")
            .shape(),
        [
            i32::try_from(QWEN3_06B_VOCAB).expect("bounded Qwen3 vocabulary"),
            i32::try_from(QWEN3_06B_HIDDEN_SIZE).expect("bounded Qwen3 hidden size"),
        ]
    );
}

fn assert_logits_match(concat: &[f32], capacity: &[f32]) {
    assert_eq!(concat.len(), capacity.len());
    for (&concat, &capacity) in concat.iter().zip(capacity) {
        assert!((concat - capacity).abs() <= 5e-5);
    }
}

fn measurement_rows() -> usize {
    let Some(value) = env::var_os("METALLIX_CAPACITY_ROWS") else {
        return MEASURED_ROWS;
    };
    let value = value
        .into_string()
        .expect("METALLIX_CAPACITY_ROWS must be valid Unicode");
    let rows = value
        .parse::<usize>()
        .expect("METALLIX_CAPACITY_ROWS must be an integer from 3 through 30");
    assert!(
        (MEASURED_ROWS..=30).contains(&rows),
        "METALLIX_CAPACITY_ROWS must be from {MEASURED_ROWS} through 30"
    );
    rows
}

fn greedy_token(logits: &[f32]) -> usize {
    logits
        .iter()
        .enumerate()
        .max_by(|(_, left), (_, right)| left.total_cmp(right))
        .expect("nonempty logits")
        .0
}

fn extend_fnv1a64(mut hash: u64, logits: &[f32]) -> u64 {
    for value in logits {
        for byte in value.to_bits().to_le_bytes() {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    hash
}

fn run_row(weights: &Qwen3MlxWeights, prompt: &[i32], capacity_first: bool) -> Row {
    let mut concat = weights
        .resident_chat_executor(MAXIMUM_CONTEXT_TOKENS, MAXIMUM_KV_BYTES)
        .expect("concat resident executor");
    let mut capacity = CapacityExecutor::new(concat.config, concat.weights, MAXIMUM_CONTEXT_TOKENS);
    let concat_prefill = concat.prefill_last_logits(prompt).expect("concat prefill");
    let capacity_prefill = capacity
        .prefill_last_logits(prompt)
        .expect("capacity prefill");
    assert_logits_match(&concat_prefill, &capacity_prefill);
    assert_eq!(
        greedy_token(&concat_prefill),
        greedy_token(&capacity_prefill)
    );

    let mut concat_decode = Duration::ZERO;
    let mut capacity_decode = Duration::ZERO;
    let mut concat_logits_bits = Vec::new();
    let mut capacity_logits_bits = Vec::new();
    let mut concat_trace_fnv1a64 = extend_fnv1a64(0xcbf2_9ce4_8422_2325, &concat_prefill);
    let mut capacity_trace_fnv1a64 = extend_fnv1a64(0xcbf2_9ce4_8422_2325, &capacity_prefill);
    for (step, token) in fixed_tokens(DECODE_STEPS, 31_337).into_iter().enumerate() {
        // Rows choose opposite initial order and each decode reverses it, so
        // neither branch systematically inherits first- or second-evaluation
        // placement within a row.
        let capacity_first = capacity_first ^ step.is_multiple_of(2);
        let (concat_logits, capacity_logits) = if capacity_first {
            let started = Instant::now();
            let capacity_logits = capacity.decode_last_logits(token).expect("capacity decode");
            capacity_decode += started.elapsed();
            let started = Instant::now();
            let concat_logits = concat.decode_last_logits(token).expect("concat decode");
            concat_decode += started.elapsed();
            (concat_logits, capacity_logits)
        } else {
            let started = Instant::now();
            let concat_logits = concat.decode_last_logits(token).expect("concat decode");
            concat_decode += started.elapsed();
            let started = Instant::now();
            let capacity_logits = capacity.decode_last_logits(token).expect("capacity decode");
            capacity_decode += started.elapsed();
            (concat_logits, capacity_logits)
        };
        assert_logits_match(&concat_logits, &capacity_logits);
        assert_eq!(greedy_token(&concat_logits), greedy_token(&capacity_logits));
        concat_trace_fnv1a64 = extend_fnv1a64(concat_trace_fnv1a64, &concat_logits);
        capacity_trace_fnv1a64 = extend_fnv1a64(capacity_trace_fnv1a64, &capacity_logits);
        concat_logits_bits = concat_logits.into_iter().map(f32::to_bits).collect();
        capacity_logits_bits = capacity_logits.into_iter().map(f32::to_bits).collect();
    }
    assert_eq!(concat_logits_bits.len(), QWEN3_06B_VOCAB);
    assert_eq!(capacity_logits_bits.len(), QWEN3_06B_VOCAB);
    let expected_concat_kv_bytes = usize::try_from(
        concat
            .config
            .cached_kv_bytes(prompt.len() + DECODE_STEPS)
            .expect("concat K/V byte estimate"),
    )
    .expect("platform K/V byte count");
    let expected_capacity_kv_bytes = usize::try_from(
        capacity
            .config
            .cached_kv_bytes(capacity.capacity)
            .expect("capacity K/V byte estimate"),
    )
    .expect("platform K/V byte count");
    assert_eq!(concat.kv_bytes(), expected_concat_kv_bytes);
    assert_eq!(capacity.capacity_kv_bytes(), expected_capacity_kv_bytes);
    Row {
        concat_decode,
        capacity_decode,
        concat_logits_bits,
        capacity_logits_bits,
        concat_trace_fnv1a64,
        capacity_trace_fnv1a64,
        concat_kv_bytes: concat.kv_bytes(),
        capacity_kv_bytes: capacity.capacity_kv_bytes(),
    }
}

#[test]
fn capacity_executor_matches_concat_rejects_overflow_and_isolates_forks() {
    let _gpu = GPU_TEST_LOCK.lock().expect("GPU test lock");
    let config = super::small_dense_config();
    let weights = super::deterministic_weights();
    let mut capacity = CapacityExecutor::new(&config, &weights, 4);
    assert!(matches!(
        capacity.decode_last_logits(1),
        Err(Qwen3ForwardError::DecodeWithoutPrefill)
    ));

    let mut concat = Qwen3ForwardExecutor::new(&config, &weights);
    let concat_prefill = concat.prefill_last_logits(&[1, 2]).expect("concat prefill");
    let capacity_prefill = capacity
        .prefill_last_logits(&[1, 2])
        .expect("capacity prefill");
    assert_logits_match(&concat_prefill, &capacity_prefill);
    assert_eq!(capacity.cached_tokens, 2);
    assert_eq!(
        capacity.capacity_kv_bytes(),
        usize::try_from(config.cached_kv_bytes(4).expect("small capacity bytes"))
            .expect("platform byte count")
    );

    let before_invalid = capacity_snapshot(&capacity);
    assert!(matches!(
        capacity.decode_last_logits(8),
        Err(Qwen3ForwardError::InvalidTokenId { .. })
    ));
    assert_eq!(capacity.cached_tokens, 2);
    assert_eq!(capacity_snapshot(&capacity), before_invalid);

    let mut child = capacity
        .fork_prefilled()
        .expect("independent capacity fork");
    let expected_parent = concat.decode_last_logits(3).expect("concat parent decode");
    let actual_parent = capacity
        .decode_last_logits(3)
        .expect("capacity parent decode");
    assert_logits_match(&expected_parent, &actual_parent);
    let expected_child = Qwen3ForwardExecutor::new(&config, &weights)
        .prefill_last_logits(&[1, 2, 4])
        .expect("fresh child replay");
    let actual_child = child.decode_last_logits(4).expect("capacity child decode");
    assert_logits_match(&expected_child, &actual_child);
    assert_eq!(capacity.cached_tokens, 3, "child must not mutate parent");
    assert_eq!(child.cached_tokens, 3);

    let mut full = CapacityExecutor::new(&config, &weights, 4);
    full.prefill_last_logits(&[1, 2, 3, 4])
        .expect("capacity-filling prefill");
    let before_overflow = capacity_snapshot(&full);
    assert!(matches!(
        full.decode_last_logits(5),
        Err(Qwen3ForwardError::PromptTooLong {
            actual: 5,
            maximum: 4
        })
    ));
    assert_eq!(full.cached_tokens, 4, "overflow must not alter cache state");
    assert_eq!(capacity_snapshot(&full), before_overflow);
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(16))]

    #[test]
    fn capacity_forks_match_fresh_full_forward_and_leave_eos_snapshot_frozen(
        prompt_a in 0_i32..8,
        prompt_b in 0_i32..8,
        tail in 0_i32..8,
        branch_token in 0_i32..8,
    ) {
        let _gpu = GPU_TEST_LOCK.lock().expect("GPU test lock");
        let config = super::small_dense_config();
        let weights = super::deterministic_weights();
        let prompt = [prompt_a, prompt_b];
        let mut parent = CapacityExecutor::new(&config, &weights, 4);
        assert_logits_match(
            &forward_last_logits(&weights, &config, &prompt).expect("fresh prompt forward"),
            &parent.prefill_last_logits(&prompt).expect("capacity prompt prefill"),
        );
        let mut branch = parent.fork_prefilled().expect("branch fork");
        let eos = parent.fork_prefilled().expect("frozen EOS fork");
        let eos_snapshot = capacity_snapshot(&eos);

        let parent_logits = parent.decode_last_logits(tail).expect("parent tail decode");
        assert_logits_match(
            &forward_last_logits(&weights, &config, &[prompt_a, prompt_b, tail])
                .expect("fresh parent forward"),
            &parent_logits,
        );
        let branch_logits = branch
            .decode_last_logits(branch_token)
            .expect("branch token decode");
        assert_logits_match(
            &forward_last_logits(&weights, &config, &[prompt_a, prompt_b, branch_token])
                .expect("fresh branch forward"),
            &branch_logits,
        );
        prop_assert_eq!(parent.cached_tokens, 3);
        prop_assert_eq!(branch.cached_tokens, 3);
        prop_assert_eq!(eos.cached_tokens, 2);
        prop_assert_eq!(capacity_snapshot(&eos), eos_snapshot);
    }
}

fn assert_repeat_identity(reference: &Row, row: &Row) {
    assert_eq!(row.concat_logits_bits, reference.concat_logits_bits);
    assert_eq!(row.capacity_logits_bits, reference.capacity_logits_bits);
    assert_eq!(row.concat_trace_fnv1a64, reference.concat_trace_fnv1a64);
    assert_eq!(row.capacity_trace_fnv1a64, reference.capacity_trace_fnv1a64);
}

fn milliseconds(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1_000.0
}

fn memory_prompt_tokens() -> usize {
    let value = match env::var("METALLIX_CAPACITY_MEMORY_PROMPT_TOKENS") {
        Ok(value) => value,
        Err(env::VarError::NotPresent) => return 1_983,
        Err(env::VarError::NotUnicode(_)) => {
            panic!("METALLIX_CAPACITY_MEMORY_PROMPT_TOKENS must be valid Unicode")
        }
    };
    match value.as_str() {
        "1983" => 1_983,
        "128" => 128,
        _ => panic!("METALLIX_CAPACITY_MEMORY_PROMPT_TOKENS must be 128 or 1983, got {value:?}"),
    }
}

fn memory_mode() -> String {
    let mode = env::var("METALLIX_CAPACITY_MEMORY_MODE")
        .expect("METALLIX_CAPACITY_MEMORY_MODE must be concat or capacity");
    assert!(
        matches!(mode.as_str(), "concat" | "capacity"),
        "METALLIX_CAPACITY_MEMORY_MODE must be concat or capacity"
    );
    mode
}

#[test]
#[ignore = "requires METALLIX_QWEN_MODEL and METALLIX_CAPACITY_MEMORY_MODE=concat|capacity"]
fn fixed_capacity_memory_qualification_uses_one_cache_without_forks() {
    let model = env::var_os("METALLIX_QWEN_MODEL")
        .map(PathBuf::from)
        .expect("METALLIX_QWEN_MODEL is required for this ignored memory qualification");
    let mode = memory_mode();
    let prompt_tokens = memory_prompt_tokens();
    let prompt = fixed_tokens(prompt_tokens, 97);
    let _gpu = GPU_TEST_LOCK.lock().expect("GPU test lock");
    let mut weights = Qwen3MlxWeights::load(model).expect("checkpoint load");
    weights.prepare_float32().expect("resident float32 weights");
    let mut concat = weights
        .resident_chat_executor(MAXIMUM_CONTEXT_TOKENS, MAXIMUM_KV_BYTES)
        .expect("Qwen3-0.6B resident plan");
    assert_qwen3_06b_tied_embedding_layout(concat.config, concat.weights);

    let (elapsed, logical_kv_bytes, trace_fnv1a64, final_logits_bits) = if mode == "concat" {
        let started = Instant::now();
        let prefill = concat.prefill_last_logits(&prompt).expect("concat prefill");
        let mut trace = extend_fnv1a64(0xcbf2_9ce4_8422_2325, &prefill);
        let mut final_logits_bits = Vec::new();
        for token in fixed_tokens(DECODE_STEPS, 31_337) {
            let logits = concat.decode_last_logits(token).expect("concat decode");
            trace = extend_fnv1a64(trace, &logits);
            final_logits_bits = logits.into_iter().map(f32::to_bits).collect();
        }
        (
            started.elapsed(),
            concat.kv_bytes(),
            trace,
            final_logits_bits,
        )
    } else {
        let config = concat.config;
        let tensors = concat.weights;
        drop(concat);
        let mut capacity = CapacityExecutor::new(config, tensors, MAXIMUM_CONTEXT_TOKENS);
        let started = Instant::now();
        let prefill = capacity
            .prefill_last_logits(&prompt)
            .expect("capacity prefill");
        let mut trace = extend_fnv1a64(0xcbf2_9ce4_8422_2325, &prefill);
        let mut final_logits_bits = Vec::new();
        for token in fixed_tokens(DECODE_STEPS, 31_337) {
            let logits = capacity.decode_last_logits(token).expect("capacity decode");
            trace = extend_fnv1a64(trace, &logits);
            final_logits_bits = logits.into_iter().map(f32::to_bits).collect();
        }
        (
            started.elapsed(),
            capacity.capacity_kv_bytes(),
            trace,
            final_logits_bits,
        )
    };
    assert_eq!(final_logits_bits.len(), QWEN3_06B_VOCAB);
    println!(
        "capacity_cache_memory mode={mode} prompt_tokens={prompt_tokens} decode_steps={DECODE_STEPS} \
         elapsed_ms={:.3} logical_kv_bytes={logical_kv_bytes} \
         whole_trace_fnv1a64={trace_fnv1a64:016x} no_forks=true",
        milliseconds(elapsed),
    );
}

#[test]
#[ignore = "requires METALLIX_QWEN_MODEL pointing to Qwen3-0.6B on Apple-Silicon Metal"]
fn fixed_capacity_slice_update_matches_concat_and_preserves_fork_ancestry() {
    let model = env::var_os("METALLIX_QWEN_MODEL")
        .map(PathBuf::from)
        .expect("METALLIX_QWEN_MODEL is required for this ignored feasibility test");
    let _gpu = GPU_TEST_LOCK.lock().expect("GPU test lock");
    let mut weights = Qwen3MlxWeights::load(model).expect("checkpoint load");
    weights.prepare_float32().expect("resident float32 weights");
    let layout = weights
        .resident_chat_executor(MAXIMUM_CONTEXT_TOKENS, MAXIMUM_KV_BYTES)
        .expect("Qwen3-0.6B resident plan");
    assert_qwen3_06b_tied_embedding_layout(layout.config, layout.weights);
    let rows = measurement_rows();
    println!(
        "capacity_cache_profile scope=test_only_fixed_capacity no_donation_claim \
         alternating_paired_host_intervals context_tokens={MAXIMUM_CONTEXT_TOKENS} \
         kv_budget_bytes={MAXIMUM_KV_BYTES} rows={rows}",
    );
    for prompt_tokens in [128_usize, 512, 1_983] {
        let prompt = fixed_tokens(prompt_tokens, 97);
        let warmup = run_row(&weights, &prompt, false);
        for row in 1..=rows {
            let capacity_first = row.is_multiple_of(2);
            let measurement = run_row(&weights, &prompt, capacity_first);
            assert_repeat_identity(&warmup, &measurement);
            println!(
                "capacity_cache_profile prompt_tokens={prompt_tokens} row={row} \
                 concat_decode_ms={:.3} capacity_decode_ms={:.3} \
                 concat_kv_bytes={} capacity_kv_bytes={} capacity_first={} \
                 concat_trace_fnv1a64={:016x} capacity_trace_fnv1a64={:016x}",
                milliseconds(measurement.concat_decode),
                milliseconds(measurement.capacity_decode),
                measurement.concat_kv_bytes,
                measurement.capacity_kv_bytes,
                capacity_first,
                measurement.concat_trace_fnv1a64,
                measurement.capacity_trace_fnv1a64,
            );
        }
    }

    let prompt = fixed_tokens(128, 97);
    let base = weights
        .resident_chat_executor(MAXIMUM_CONTEXT_TOKENS, MAXIMUM_KV_BYTES)
        .expect("parent executor plan");
    let mut parent = CapacityExecutor::new(base.config, base.weights, MAXIMUM_CONTEXT_TOKENS);
    let mut concat_parent = weights
        .resident_chat_executor(MAXIMUM_CONTEXT_TOKENS, MAXIMUM_KV_BYTES)
        .expect("concat parent executor plan");
    parent.prefill_last_logits(&prompt).expect("parent prefill");
    concat_parent
        .prefill_last_logits(&prompt)
        .expect("concat parent prefill");
    let mut left = parent.fork_prefilled().expect("left capacity fork");
    let mut right = parent.fork_prefilled().expect("right capacity fork");
    let expected_parent = concat_parent
        .decode_last_logits(13)
        .expect("concat parent decode");
    let parent_logits = parent.decode_last_logits(13).expect("parent decode");
    let left_logits = left.decode_last_logits(13).expect("left decode");
    let right_logits = right.decode_last_logits(14).expect("right decode");
    assert_logits_match(&expected_parent, &parent_logits);
    assert_logits_match(&expected_parent, &left_logits);
    let mut concat_right = weights
        .resident_chat_executor(MAXIMUM_CONTEXT_TOKENS, MAXIMUM_KV_BYTES)
        .expect("concat right executor plan");
    concat_right
        .prefill_last_logits(&prompt)
        .expect("concat right prefill");
    let expected_right = concat_right
        .decode_last_logits(14)
        .expect("concat right decode");
    assert_logits_match(&expected_right, &right_logits);
}
