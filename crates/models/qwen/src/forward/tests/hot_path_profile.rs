//! Qwen3-0.6B greedy-decode hot-path probe on a local checkpoint. Ignored: it
//! prints host wall time per generated token for each decode variant, which
//! diagnoses a candidate on the current machine; it is not a serving benchmark.

use std::{
    env,
    path::PathBuf,
    time::{Duration, Instant},
};

use crate::{
    GPU_TEST_LOCK,
    metal::{Qwen3FloatPrecision, Qwen3MlxWeights},
};

const MAXIMUM_CONTEXT_TOKENS: usize = 4_096;
const MAXIMUM_KV_BYTES: u64 = 2 * 1024 * 1024 * 1024;
const GENERATED_TOKENS: usize = 128;
const MEASURED_ROWS: usize = 3;

#[derive(Clone, Copy, Debug)]
enum Variant {
    /// Full logit row read back each step, argmax on the host.
    HostArgmax,
    /// GPU argmax, waited on before the next step is built.
    GpuArgmax,
    /// GPU argmax with step `t + 1` queued before step `t` is read back.
    Pipelined,
}

struct Row {
    tokens: Vec<i32>,
    step_times: Vec<Duration>,
}

fn fixed_tokens(count: usize, offset: usize) -> Vec<i32> {
    (0..count)
        .map(|index| {
            let value = ((index * 7_919) + offset) % 151_000 + 1;
            i32::try_from(value).expect("bounded Qwen3 token ID")
        })
        .collect()
}

fn host_argmax(logits: &[f32]) -> i32 {
    let mut best = 0;
    for (index, &value) in logits.iter().enumerate() {
        assert!(value.is_finite(), "non-finite logit");
        if value > logits[best] {
            best = index;
        }
    }
    i32::try_from(best).expect("vocabulary fits i32")
}

fn generate(weights: &Qwen3MlxWeights, prompt: &[i32], variant: Variant) -> Row {
    let mut executor = weights
        .resident_chat_executor(MAXIMUM_CONTEXT_TOKENS, MAXIMUM_KV_BYTES)
        .expect("resident executor plan");
    let first = host_argmax(&executor.prefill_last_logits(prompt).expect("prefill"));
    let mut tokens = vec![first];
    let mut step_times = Vec::with_capacity(GENERATED_TOKENS);
    let mut last = Instant::now();
    match variant {
        Variant::HostArgmax => {
            while tokens.len() < GENERATED_TOKENS {
                let previous = *tokens.last().expect("first token");
                let logits = executor.decode_last_logits(previous).expect("decode");
                tokens.push(host_argmax(&logits));
                let now = Instant::now();
                step_times.push(now - last);
                last = now;
            }
        }
        Variant::GpuArgmax => {
            while tokens.len() < GENERATED_TOKENS {
                let previous = *tokens.last().expect("first token");
                let pending = executor.decode_greedy(previous).expect("decode");
                tokens.push(pending.wait_one().expect("finite logits"));
                let now = Instant::now();
                step_times.push(now - last);
                last = now;
            }
        }
        Variant::Pipelined => {
            let mut pending = executor.decode_greedy(first).expect("decode");
            while tokens.len() < GENERATED_TOKENS {
                let next = (tokens.len() + 1 < GENERATED_TOKENS)
                    .then(|| executor.decode_greedy_after(&pending).expect("decode"));
                tokens.push(pending.wait_one().expect("finite logits"));
                let now = Instant::now();
                step_times.push(now - last);
                last = now;
                match next {
                    Some(next) => pending = next,
                    None => break,
                }
            }
        }
    }
    Row { tokens, step_times }
}

fn milliseconds(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1_000.0
}

fn summarize(step_times: &[Duration]) -> (f64, f64, f64) {
    let mut sorted: Vec<f64> = step_times.iter().copied().map(milliseconds).collect();
    sorted.sort_by(f64::total_cmp);
    let mean = sorted.iter().sum::<f64>()
        / f64::from(u32::try_from(sorted.len()).expect("step count fits u32"));
    (
        sorted[sorted.len() / 2],
        mean,
        sorted[sorted.len() * 9 / 10],
    )
}

fn first_divergence(left: &[i32], right: &[i32]) -> Option<usize> {
    left.iter()
        .zip(right)
        .position(|(left, right)| left != right)
}

fn load(model: &PathBuf, precision: Qwen3FloatPrecision) -> Qwen3MlxWeights {
    let mut weights = Qwen3MlxWeights::load(model).expect("checkpoint load");
    weights.prepare_precision(precision).expect("precision");
    weights
}

#[test]
#[ignore = "requires METALLIX_QWEN_MODEL pointing to Qwen3-0.6B on Apple-Silicon Metal"]
fn greedy_decode_hot_path_variants() {
    let model = env::var_os("METALLIX_QWEN_MODEL")
        .map(PathBuf::from)
        .expect("METALLIX_QWEN_MODEL is required for this ignored checkpoint probe");
    let _gpu = GPU_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    println!(
        "hot_path scope=host_wall_per_generated_token generated={GENERATED_TOKENS} \
         context_tokens={MAXIMUM_CONTEXT_TOKENS}"
    );
    let cases = [
        (Qwen3FloatPrecision::Float32, Variant::HostArgmax),
        (Qwen3FloatPrecision::BFloat16, Variant::HostArgmax),
        (Qwen3FloatPrecision::BFloat16, Variant::GpuArgmax),
        (Qwen3FloatPrecision::BFloat16, Variant::Pipelined),
    ];
    for prompt_tokens in [2_300_usize, 128] {
        let prompt = fixed_tokens(prompt_tokens, 97);
        let mut reference: Option<Vec<i32>> = None;
        let mut bf16_reference: Option<Vec<i32>> = None;
        for (precision, variant) in cases {
            let weights = load(&model, precision);
            let warmup = generate(&weights, &prompt, variant);
            for row in 1..=MEASURED_ROWS {
                let measured = generate(&weights, &prompt, variant);
                assert_eq!(measured.tokens, warmup.tokens, "repeat run changed tokens");
                let (p50, mean, p90) = summarize(&measured.step_times);
                println!(
                    "hot_path prompt_tokens={prompt_tokens} precision={precision:?} \
                     variant={variant:?} row={row} step_ms_p50={p50:.3} \
                     step_ms_mean={mean:.3} step_ms_p90={p90:.3}"
                );
            }
            match precision {
                Qwen3FloatPrecision::Float32 => reference = Some(warmup.tokens.clone()),
                Qwen3FloatPrecision::BFloat16 | Qwen3FloatPrecision::Float16 => {
                    // Same weights and logits: every BF16 variant must pick
                    // the same tokens, ties included.
                    match &bf16_reference {
                        Some(expected) => assert_eq!(&warmup.tokens, expected),
                        None => bf16_reference = Some(warmup.tokens.clone()),
                    }
                }
            }
        }
        if let (Some(float32), Some(bfloat16)) = (&reference, &bf16_reference) {
            println!(
                "hot_path prompt_tokens={prompt_tokens} f32_vs_bf16_first_divergence={:?}",
                first_divergence(float32, bfloat16)
            );
        }
    }
}

/// Per-token pipelined greedy decode of an affine-quantized checkpoint as
/// loaded, for comparison with the BF16 rows above.
#[test]
#[ignore = "requires METALLIX_QWEN_AFFINE_MODEL pointing to an MLX affine-quantized Qwen3 on Apple-Silicon Metal"]
fn quantized_greedy_decode_hot_path() {
    let model = env::var_os("METALLIX_QWEN_AFFINE_MODEL")
        .map(PathBuf::from)
        .expect("METALLIX_QWEN_AFFINE_MODEL is required for this ignored checkpoint probe");
    let _gpu = GPU_TEST_LOCK.lock().expect("GPU test lock");
    let weights = Qwen3MlxWeights::load(&model).expect("checkpoint load");
    let precision = weights.precision().expect("precision");
    for prompt_tokens in [2_300_usize, 128] {
        let prompt = fixed_tokens(prompt_tokens, 97);
        let warmup = generate(&weights, &prompt, Variant::Pipelined);
        for row in 1..=MEASURED_ROWS {
            let measured = generate(&weights, &prompt, Variant::Pipelined);
            assert_eq!(measured.tokens, warmup.tokens, "repeat run changed tokens");
            let (p50, mean, p90) = summarize(&measured.step_times);
            println!(
                "hot_path prompt_tokens={prompt_tokens} precision={precision:?} \
                 variant=Pipelined row={row} step_ms_p50={p50:.3} \
                 step_ms_mean={mean:.3} step_ms_p90={p90:.3}"
            );
        }
    }
}
