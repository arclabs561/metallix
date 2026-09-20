//! Qwen3-0.6B local-checkpoint component graphs for cache-growth diagnosis.
//!
//! These are standalone MLX graphs, not alternative Qwen forwards or serving
//! measurements. Each component has one terminal evaluation and no staged
//! synchronization, so its host interval cannot attribute an individual GPU
//! kernel or be subtracted from normal decode wall time.

use std::{
    env,
    path::PathBuf,
    time::{Duration, Instant},
};

use mlx_rs::{Array, Dtype, StreamOrDevice, fast, ops, transforms};

use crate::{GPU_TEST_LOCK, metal::Qwen3MlxWeights};

use super::super::{Qwen3ForwardExecutor, attention_scale};

const MAXIMUM_CONTEXT_TOKENS: usize = 2_048;
const MAXIMUM_KV_BYTES: u64 = 512 * 1024 * 1024;
const MEASURED_ROWS: usize = 3;
const QWEN3_06B_LAYERS: usize = 28;
const QWEN3_06B_VOCAB: usize = 151_936;

#[derive(Clone, Debug, Eq, PartialEq)]
struct CacheMetadata(Vec<LayerCacheMetadata>);

#[derive(Clone, Debug, Eq, PartialEq)]
struct ArrayMetadata {
    shape: Vec<i32>,
    dtype: Dtype,
    bytes: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct LayerCacheMetadata {
    keys: ArrayMetadata,
    values: ArrayMetadata,
}

fn array_metadata(array: &Array) -> ArrayMetadata {
    ArrayMetadata {
        shape: array.shape().to_vec(),
        dtype: array.dtype(),
        bytes: array.nbytes(),
    }
}

#[derive(Debug)]
struct ComponentRow {
    concat: Duration,
    attention: Duration,
    combined: Duration,
    attention_fnv1a64: u64,
    combined_fnv1a64: u64,
    decode_logits_bits: Vec<u32>,
}

fn fixed_tokens(count: usize, offset: usize) -> Vec<i32> {
    (0..count)
        .map(|index| {
            let value = ((index * 7_919) + offset) % 151_000 + 1;
            i32::try_from(value).expect("bounded Qwen3 token ID")
        })
        .collect()
}

fn cache_metadata(
    executor: &Qwen3ForwardExecutor<'_, std::collections::hash_map::RandomState>,
) -> CacheMetadata {
    CacheMetadata(
        executor
            .cache
            .iter()
            .map(|entry| {
                let layer = entry.as_ref().expect("complete resident cache");
                LayerCacheMetadata {
                    keys: array_metadata(&layer.keys),
                    values: array_metadata(&layer.values),
                }
            })
            .collect(),
    )
}

fn fnv1a64(outputs: &[Array]) -> u64 {
    outputs
        .iter()
        .fold(0xcbf2_9ce4_8422_2325, |mut hash, output| {
            for value in output.as_slice::<f32>() {
                for byte in value.to_bits().to_le_bytes() {
                    hash ^= u64::from(byte);
                    hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
                }
            }
            hash
        })
}

fn evaluate(outputs: &[Array]) -> Duration {
    let started = Instant::now();
    transforms::eval(outputs.iter()).expect("single terminal component evaluation");
    started.elapsed()
}

fn component_shapes(
    executor: &Qwen3ForwardExecutor<'_, std::collections::hash_map::RandomState>,
) -> ([i32; 4], [i32; 4]) {
    let layer = executor
        .cache
        .first()
        .and_then(Option::as_ref)
        .expect("first cache layer");
    assert_eq!(executor.cache.len(), QWEN3_06B_LAYERS);
    assert_eq!(layer.keys.dtype(), Dtype::Float32);
    assert_eq!(layer.values.dtype(), Dtype::Float32);
    let key_shape: [i32; 4] = layer.keys.shape().try_into().expect("rank-four K/V");
    let value_shape: [i32; 4] = layer.values.shape().try_into().expect("rank-four K/V");
    assert_eq!(key_shape, value_shape);
    let query_shape = [
        1,
        i32::try_from(executor.config.attention_heads).expect("head count"),
        1,
        key_shape[3],
    ];
    let update_shape = [key_shape[0], key_shape[1], 1, key_shape[3]];
    (query_shape, update_shape)
}

fn concat_outputs(
    executor: &Qwen3ForwardExecutor<'_, std::collections::hash_map::RandomState>,
) -> Vec<Array> {
    let (_, update_shape) = component_shapes(executor);
    let stream = StreamOrDevice::gpu();
    let mut outputs = Vec::with_capacity(executor.cache.len() * 2);
    for entry in &executor.cache {
        let layer = entry.as_ref().expect("complete resident cache");
        let key_update =
            Array::zeros_device::<f32>(&update_shape, &stream).expect("zero key update");
        let value_update =
            Array::zeros_device::<f32>(&update_shape, &stream).expect("zero value update");
        let keys = ops::concatenate_axis_device(&[&layer.keys, &key_update], 2, &stream)
            .expect("key concat graph");
        let values = ops::concatenate_axis_device(&[&layer.values, &value_update], 2, &stream)
            .expect("value concat graph");
        let mut expected_shape = layer.keys.shape().to_vec();
        expected_shape[2] += 1;
        assert_eq!(keys.shape(), expected_shape);
        assert_eq!(values.shape(), expected_shape);
        assert_eq!(keys.dtype(), Dtype::Float32);
        assert_eq!(values.dtype(), Dtype::Float32);
        assert_eq!(keys.nbytes(), layer.keys.nbytes() + key_update.nbytes());
        assert_eq!(
            values.nbytes(),
            layer.values.nbytes() + value_update.nbytes()
        );
        outputs.extend([keys, values]);
    }
    assert_eq!(outputs.len(), QWEN3_06B_LAYERS * 2);
    outputs
}

fn build_attention_outputs(
    executor: &Qwen3ForwardExecutor<'_, std::collections::hash_map::RandomState>,
    include_concat: bool,
) -> Vec<Array> {
    let (query_shape, update_shape) = component_shapes(executor);
    let stream = StreamOrDevice::gpu();
    let query = Array::zeros_device::<f32>(&query_shape, &stream).expect("zero attention query");
    let scale = attention_scale(executor.config).expect("attention scale");
    let mut outputs = Vec::with_capacity(executor.cache.len());
    for entry in &executor.cache {
        let layer = entry.as_ref().expect("complete resident cache");
        let (keys, values) = if include_concat {
            let key_update =
                Array::zeros_device::<f32>(&update_shape, &stream).expect("zero key update");
            let value_update =
                Array::zeros_device::<f32>(&update_shape, &stream).expect("zero value update");
            (
                ops::concatenate_axis_device(&[&layer.keys, &key_update], 2, &stream)
                    .expect("combined key concat graph"),
                ops::concatenate_axis_device(&[&layer.values, &value_update], 2, &stream)
                    .expect("combined value concat graph"),
            )
        } else {
            (layer.keys.clone(), layer.values.clone())
        };
        let output = fast::scaled_dot_product_attention_device(
            &query,
            &keys,
            &values,
            scale,
            None::<fast::ScaledDotProductAttentionMask<'_>>,
            &stream,
        )
        .expect("attention component graph");
        assert_eq!(output.shape(), query_shape);
        assert_eq!(output.dtype(), Dtype::Float32);
        outputs.push(output);
    }
    assert_eq!(outputs.len(), QWEN3_06B_LAYERS);
    outputs
}

fn run_row(weights: &Qwen3MlxWeights, prompt: &[i32]) -> ComponentRow {
    let mut live = weights
        .resident_chat_executor(MAXIMUM_CONTEXT_TOKENS, MAXIMUM_KV_BYTES)
        .expect("resident executor plan");
    live.prefill_last_logits(prompt).expect("prefill cache");
    let live_before = cache_metadata(&live);
    let snapshot = live.fork_prefilled().expect("materialized cache fork");
    assert_eq!(cache_metadata(&live), live_before);
    let snapshot_before = cache_metadata(&snapshot);
    let expected_decode_logits_bits = snapshot
        .fork_prefilled()
        .expect("pre-component cache fork")
        .decode_last_logits(fixed_tokens(1, 31_337)[0])
        .expect("pre-component ordinary cached decode")
        .into_iter()
        .map(f32::to_bits)
        .collect::<Vec<_>>();

    let concat_outputs = concat_outputs(&snapshot);
    let concat = evaluate(&concat_outputs);
    drop(concat_outputs);

    let attention_outputs = build_attention_outputs(&snapshot, false);
    let attention = evaluate(&attention_outputs);
    let attention_fnv1a64 = fnv1a64(&attention_outputs);
    drop(attention_outputs);

    let combined_outputs = build_attention_outputs(&snapshot, true);
    let combined = evaluate(&combined_outputs);
    let combined_fnv1a64 = fnv1a64(&combined_outputs);
    drop(combined_outputs);

    assert_eq!(cache_metadata(&snapshot), snapshot_before);
    let decode_logits_bits = snapshot
        .fork_prefilled()
        .expect("post-component cache fork")
        .decode_last_logits(fixed_tokens(1, 31_337)[0])
        .expect("post-component ordinary cached decode")
        .into_iter()
        .map(f32::to_bits)
        .collect::<Vec<_>>();
    assert_eq!(decode_logits_bits, expected_decode_logits_bits);
    assert_eq!(decode_logits_bits.len(), QWEN3_06B_VOCAB);
    ComponentRow {
        concat,
        attention,
        combined,
        attention_fnv1a64,
        combined_fnv1a64,
        decode_logits_bits,
    }
}

fn assert_repeat_identity(reference: &ComponentRow, row: &ComponentRow) {
    assert_eq!(row.attention_fnv1a64, reference.attention_fnv1a64);
    assert_eq!(row.combined_fnv1a64, reference.combined_fnv1a64);
    assert_eq!(row.decode_logits_bits, reference.decode_logits_bits);
}

fn milliseconds(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1_000.0
}

fn report_row(prompt_tokens: usize, row: usize, measurement: &ComponentRow) {
    println!(
        "cache_component_profile prompt_tokens={prompt_tokens} row={row} \
         concat_eval_ms={:.3} attention_eval_ms={:.3} combined_eval_ms={:.3} \
         concat_outputs=56 attention_outputs=28 combined_outputs=28 \
         attention_fnv1a64={:016x} combined_fnv1a64={:016x}",
        milliseconds(measurement.concat),
        milliseconds(measurement.attention),
        milliseconds(measurement.combined),
        measurement.attention_fnv1a64,
        measurement.combined_fnv1a64,
    );
}

#[test]
#[ignore = "requires METALLIX_QWEN_MODEL pointing to Qwen3-0.6B on Apple-Silicon Metal"]
fn cache_component_graphs_are_repeatable_without_mutating_resident_kv() {
    let model = env::var_os("METALLIX_QWEN_MODEL")
        .map(PathBuf::from)
        .expect("METALLIX_QWEN_MODEL is required for this ignored checkpoint qualification");
    let _gpu = GPU_TEST_LOCK.lock().expect("GPU test lock");
    let mut weights = Qwen3MlxWeights::load(model).expect("checkpoint load");
    weights.prepare_float32().expect("resident float32 weights");
    println!(
        "cache_component_profile scope=standalone_component_graphs \
         single_terminal_eval_per_component host_interval_not_gpu_kernel_time \
         no_component_subtraction_or_serving_comparison \
         context_tokens={MAXIMUM_CONTEXT_TOKENS} kv_budget_bytes={MAXIMUM_KV_BYTES}"
    );
    for prompt_tokens in [128_usize, 512, 1_983] {
        let prompt = fixed_tokens(prompt_tokens, 97);
        let warmup = run_row(&weights, &prompt);
        for row in 1..=MEASURED_ROWS {
            let measurement = run_row(&weights, &prompt);
            assert_repeat_identity(&warmup, &measurement);
            report_row(prompt_tokens, row, &measurement);
        }
    }
}
