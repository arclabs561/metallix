//! Qwen3-0.6B local-checkpoint decode phase probe. It is intentionally ignored: the
//! printed host timings diagnose a candidate; they are not a serving benchmark.

use std::{
    env,
    path::PathBuf,
    time::{Duration, Instant},
};

use crate::{GPU_TEST_LOCK, metal::Qwen3MlxWeights};

use super::super::{DecodeProfile, DecodeProfileScope};

const MAXIMUM_CONTEXT_TOKENS: usize = 2_048;
const MAXIMUM_KV_BYTES: u64 = 512 * 1024 * 1024;
const DECODE_STEPS: usize = 64;
const MEASURED_ROWS: usize = 5;
// Qwen3-0.6B has 28 layers. Each cached layer builds Q/K/V/O plus three MLP
// projections; the tied output projection adds one more transpose node.
const TRANSPOSE_NODES_PER_DECODE: usize = (28 * 7) + 1;

#[derive(Debug)]
struct DecodeProfileRow {
    setup: Duration,
    prefill: Duration,
    total_decode: Duration,
    profile: DecodeProfile,
    trace_fnv1a64: u64,
    final_logits_bits: Vec<u32>,
}

fn fixed_tokens(count: usize, offset: usize) -> Vec<i32> {
    (0..count)
        .map(|index| {
            let value = ((index * 7_919) + offset) % 151_000 + 1;
            i32::try_from(value).expect("bounded Qwen3 token ID")
        })
        .collect()
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

fn run_row(weights: &Qwen3MlxWeights, prompt: &[i32]) -> DecodeProfileRow {
    let setup_started = Instant::now();
    let mut executor = weights
        .resident_chat_executor(MAXIMUM_CONTEXT_TOKENS, MAXIMUM_KV_BYTES)
        .expect("resident executor plan");
    let setup = setup_started.elapsed();

    let prefill_started = Instant::now();
    let prefill_logits = executor
        .prefill_last_logits(prompt)
        .expect("teacher-forced prefill");
    let prefill = prefill_started.elapsed();

    let mut trace_fnv1a64 = extend_fnv1a64(0xcbf2_9ce4_8422_2325, &prefill_logits);
    let mut final_logits_bits = Vec::new();
    // The phase collector deliberately starts after prefill, matching
    // `total_decode` exactly. Its intervals are decode-only host timings.
    let scope = DecodeProfileScope::start();
    let decode_started = Instant::now();
    for token in fixed_tokens(DECODE_STEPS, 31_337) {
        let logits = executor
            .decode_last_logits(token)
            .expect("teacher-forced cached decode");
        trace_fnv1a64 = extend_fnv1a64(trace_fnv1a64, &logits);
        final_logits_bits = logits.into_iter().map(f32::to_bits).collect();
    }
    let total_decode = decode_started.elapsed();
    let profile = scope.finish();

    assert_eq!(executor.cached_tokens(), prompt.len() + DECODE_STEPS);
    assert_eq!(final_logits_bits.len(), 151_936);
    assert_eq!(profile.evaluation_count, DECODE_STEPS);
    assert_eq!(profile.readback_count, DECODE_STEPS);
    assert_eq!(
        profile.transpose_node_count,
        DECODE_STEPS * TRANSPOSE_NODES_PER_DECODE
    );
    DecodeProfileRow {
        setup,
        prefill,
        total_decode,
        profile,
        trace_fnv1a64,
        final_logits_bits,
    }
}

fn assert_repeat_identity(reference: &DecodeProfileRow, row: &DecodeProfileRow) {
    assert_eq!(row.trace_fnv1a64, reference.trace_fnv1a64);
    assert_eq!(row.final_logits_bits, reference.final_logits_bits);
}

fn milliseconds(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1_000.0
}

fn report_row(prompt_tokens: usize, row: usize, measurement: &DecodeProfileRow) {
    println!(
        "decode_profile prompt_tokens={prompt_tokens} row={row} \
         setup_ms={:.3} prefill_ms={:.3} total_decode_ms={:.3} \
         transpose_node_ms={:.3} transpose_node_count={} \
         eval_ms={:.3} eval_count={} readback_ms={:.3} readback_count={} \
         trace_fnv1a64={:016x}",
        milliseconds(measurement.setup),
        milliseconds(measurement.prefill),
        milliseconds(measurement.total_decode),
        milliseconds(measurement.profile.transpose_node),
        measurement.profile.transpose_node_count,
        milliseconds(measurement.profile.evaluation),
        measurement.profile.evaluation_count,
        milliseconds(measurement.profile.readback),
        measurement.profile.readback_count,
        measurement.trace_fnv1a64,
    );
}

#[test]
#[ignore = "requires METALLIX_QWEN_MODEL pointing to Qwen3-0.6B on Apple-Silicon Metal"]
fn resident_decode_phase_profile_is_repeatable_for_fixed_teacher_forcing() {
    let model = env::var_os("METALLIX_QWEN_MODEL")
        .map(PathBuf::from)
        .expect("METALLIX_QWEN_MODEL is required for this ignored checkpoint qualification");
    let _gpu = GPU_TEST_LOCK.lock().expect("GPU test lock");
    let mut weights = Qwen3MlxWeights::load(model).expect("checkpoint load");
    weights.prepare_float32().expect("resident float32 weights");
    println!(
        "decode_profile scope=host_intervals_only model_load_excluded \
         output_hashing_included_in_total_decode decode_only_counters=true \
         eval_is_not_gpu_time \
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
