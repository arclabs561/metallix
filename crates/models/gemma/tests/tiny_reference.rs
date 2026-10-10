//! Always-on parity against tiny synthetic Gemma 4 models run through
//! transformers' own `Gemma4ForCausalLM`, from
//! `scripts/gemma4-tiny-reference.py`.
//!
//! `fixtures/gemma-4-tiny/ple-reference.json` has per-layer input embeddings,
//! K/V shared by the last two layers (one sliding, one full), a double-wide
//! MLP in those layers and a 3-token window, so every E2B/E4B feature runs
//! on a few thousand random weights.
#![cfg(feature = "metal")]

use std::{collections::HashMap, sync::Mutex};

use gemma::{
    Gemma4TextConfig,
    metal::{Gemma4MlxWeights, Gemma4Precision},
};
use mlx_rs::Array;
use serde::Deserialize;

/// Bound on |logit - float64 logit|. The fixture records the source's own
/// float64 logits beside its float32 ones: on these random weights (softmax
/// scale 1 over unit-norm Q and K, so attention is sharp) the source's float32
/// lands up to 4.1e-5 from float64 on the PLE model. Any float32
/// implementation carries that noise; 1e-4 allows about 2.5 times it, while
/// each wiring mistake the reference script lists moves logits by far more.
const MAX_ABS_LOGIT: f64 = 1e-4;

/// MLX's native test operations share process-global device initialization.
static GPU: Mutex<()> = Mutex::new(());

#[derive(Deserialize)]
struct Reference {
    config: serde_json::Value,
    tensors: HashMap<String, Tensor>,
    cases: Vec<Case>,
}

#[derive(Deserialize)]
struct Tensor {
    shape: Vec<i32>,
    values: Vec<f32>,
    #[serde(default)]
    ignored: bool,
}

#[derive(Deserialize)]
struct Case {
    name: String,
    steps: Vec<Step>,
}

#[derive(Deserialize)]
struct Step {
    input_ids: Vec<i32>,
    logits: Vec<f32>,
    logits_float64: Vec<f64>,
}

fn load(json: &str) -> (Reference, Gemma4MlxWeights) {
    let reference: Reference = serde_json::from_str(json).expect("reference JSON");
    let config = Gemma4TextConfig::parse(&reference.config.to_string()).expect("config");
    // A loader drops the K/V weights shared layers store but never read.
    let tensors = reference
        .tensors
        .iter()
        .filter(|(_, tensor)| !tensor.ignored)
        .map(|(name, tensor)| {
            (
                name.clone(),
                Array::from_slice(&tensor.values, &tensor.shape),
            )
        })
        .collect();
    let weights =
        Gemma4MlxWeights::from_tensors(config, tensors, Gemma4Precision::Float32).expect("weights");
    (reference, weights)
}

fn worst(actual: &[f32], expected: &[f64]) -> f64 {
    assert_eq!(actual.len(), expected.len());
    actual
        .iter()
        .zip(expected)
        .map(|(&a, b)| (f64::from(a) - b).abs())
        .fold(0.0, f64::max)
}

fn argmax(values: &[f32]) -> usize {
    values
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .expect("logits")
        .0
}

#[test]
fn per_layer_inputs_and_shared_kv_match_transformers() {
    let _gpu = GPU
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let (reference, weights) = load(include_str!(
        "../../../../fixtures/gemma-4-tiny/ple-reference.json"
    ));
    let config = weights.config();
    assert_eq!(config.per_layer_input_width(), Some(4));
    assert_eq!(
        (config.kv_source(4), config.kv_source(5)),
        (Some(3), Some(2))
    );
    for case in &reference.cases {
        let mut executor = weights.executor(64).expect("executor");
        let mut overall = 0.0_f64;
        for (index, step) in case.steps.iter().enumerate() {
            let logits = if index == 0 {
                executor.prefill_last_logits(&step.input_ids)
            } else {
                executor.decode_last_logits(step.input_ids[0])
            }
            .expect("forward");
            let difference = worst(&logits, &step.logits_float64);
            overall = overall.max(difference);
            assert!(
                difference <= MAX_ABS_LOGIT,
                "{} step {index}: max |dlogit| {difference}",
                case.name
            );
            assert_eq!(
                argmax(&logits),
                argmax(&step.logits),
                "{} step {index}",
                case.name
            );
        }
        eprintln!("{}: worst |logit - float64 logit| {overall:e}", case.name);
        // Shared layers keep no K/V of their own.
        assert_eq!(
            executor.kv_bytes() as u128,
            config.retained_kv_elements(executor.cached_tokens()) * 4,
            "{}",
            case.name
        );

        // A prompt split across an extend crosses the window inside the cache.
        let prompt = &case.steps[0].input_ids;
        for split in 1..prompt.len() {
            let mut executor = weights.executor(64).expect("executor");
            executor
                .prefill_last_logits(&prompt[..split])
                .expect("prefix");
            let logits = executor
                .extend_last_logits(&prompt[split..])
                .expect("extend");
            let difference = worst(&logits, &case.steps[0].logits_float64);
            assert!(
                difference <= MAX_ABS_LOGIT,
                "{} split at {split}: max |dlogit| {difference}",
                case.name
            );
        }
    }
}
