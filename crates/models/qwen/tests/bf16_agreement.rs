//! Opt-in: greedy agreement between BF16 serving weights and the float32
//! reference on a Qwen3 checkpoint, both on Metal.
//!
//! Set `METALLIX_QWEN_MODEL` to a Qwen3 checkpoint directory (Qwen3-0.6B was
//! the measured one). For each prompt, float32 weights generate a greedy
//! continuation and record each step's top-1/top-2 logit gap. BF16 weights are
//! then teacher-forced on that continuation, picking each token with the GPU
//! argmax the server uses, so every step is compared, not only the steps
//! before a first divergence.
//!
//! Rule, declared before the first run: BF16 must pick the float32 token at
//! every step whose float32 top-1/top-2 gap is at least `NEAR_TIE_GAP`. Steps
//! below it may differ; their count and the divergence rate are printed.
#![cfg(feature = "metal")]

use std::{env, path::PathBuf};

use qwen::metal::{Qwen3MlxWeights, Qwen3WeightPrecision};
use tokenizers::Tokenizer;

/// Logit units, which equal natural-log probability differences.
const NEAR_TIE_GAP: f32 = 0.5;
const GENERATED_TOKENS: usize = 128;
const CONTEXT_TOKENS: usize = 2_048;
const KV_BYTES: u64 = 1024 * 1024 * 1024;

const PROMPTS: [&str; 8] = [
    "Explain how a hash map handles collisions.",
    "Write a Python function that merges two sorted lists.",
    "What causes the seasons on Earth?",
    "Summarize the plot of Romeo and Juliet in three sentences.",
    "List five prime numbers greater than 100 and explain how you checked them.",
    "Translate 'The library opens at nine tomorrow morning' into French and German.",
    "Write a short Rust function that reverses the words in a string.",
    "Why is the sky blue? Answer for a ten-year-old.",
];

struct Reference {
    tokens: Vec<i32>,
    gaps: Vec<f32>,
}

fn top_two(logits: &[f32]) -> (i32, f32) {
    let (mut first, mut second) = (0, None::<usize>);
    for index in 1..logits.len() {
        assert!(logits[index].is_finite(), "non-finite reference logit");
        if logits[index] > logits[first] {
            second = Some(first);
            first = index;
        } else if second.is_none_or(|second| logits[index] > logits[second]) {
            second = Some(index);
        }
    }
    let second = second.expect("vocabulary has two rows");
    (
        i32::try_from(first).expect("vocabulary fits i32"),
        logits[first] - logits[second],
    )
}

fn chat_prompt(tokenizer: &Tokenizer, user: &str) -> Vec<i32> {
    let text = format!(
        "<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    );
    tokenizer
        .encode(text, false)
        .expect("prompt encodes")
        .get_ids()
        .iter()
        .map(|&id| i32::try_from(id).expect("ID fits i32"))
        .collect()
}

fn weights(model: &PathBuf, precision: Qwen3WeightPrecision) -> Qwen3MlxWeights {
    let mut weights = Qwen3MlxWeights::load(model).expect("checkpoint load");
    weights.prepare_precision(precision).expect("precision");
    weights
}

fn float32_reference(weights: &Qwen3MlxWeights, prompt: &[i32]) -> Reference {
    let mut executor = weights
        .resident_chat_executor(CONTEXT_TOKENS, KV_BYTES)
        .expect("executor");
    let mut logits = executor.prefill_last_logits(prompt).expect("prefill");
    let mut reference = Reference {
        tokens: Vec::new(),
        gaps: Vec::new(),
    };
    for step in 0..GENERATED_TOKENS {
        let (token, gap) = top_two(&logits);
        reference.tokens.push(token);
        reference.gaps.push(gap);
        if step + 1 < GENERATED_TOKENS {
            logits = executor.decode_last_logits(token).expect("decode");
        }
    }
    reference
}

/// BF16's own pick at each step of the float32 continuation.
fn bfloat16_teacher_forced(weights: &Qwen3MlxWeights, prompt: &[i32], forced: &[i32]) -> Vec<i32> {
    let mut executor = weights
        .resident_chat_executor(CONTEXT_TOKENS, KV_BYTES)
        .expect("executor");
    let (first, _) = top_two(&executor.prefill_last_logits(prompt).expect("prefill"));
    let mut picks = vec![first];
    for &token in &forced[..forced.len() - 1] {
        let pending = executor.decode_greedy(token).expect("decode");
        picks.push(pending.wait_one().expect("finite logits"));
    }
    picks
}

#[test]
#[ignore = "requires METALLIX_QWEN_MODEL pointing to a Qwen3 checkpoint on Apple-Silicon Metal"]
fn bfloat16_greedy_agrees_with_float32_outside_near_ties() {
    let model = env::var_os("METALLIX_QWEN_MODEL")
        .map(PathBuf::from)
        .expect("METALLIX_QWEN_MODEL is required for this ignored checkpoint test");
    let tokenizer = Tokenizer::from_file(model.join("tokenizer.json")).expect("tokenizer");
    let prompts: Vec<Vec<i32>> = PROMPTS
        .iter()
        .map(|prompt| chat_prompt(&tokenizer, prompt))
        .collect();

    let references: Vec<Reference> = {
        let float32 = weights(&model, Qwen3WeightPrecision::Float32);
        prompts
            .iter()
            .map(|prompt| float32_reference(&float32, prompt))
            .collect()
    };
    let bfloat16 = weights(&model, Qwen3WeightPrecision::BFloat16);

    let mut steps = 0_u32;
    let mut divergences = Vec::new();
    let mut below_gap = 0;
    for (index, (prompt, reference)) in prompts.iter().zip(&references).enumerate() {
        let picks = bfloat16_teacher_forced(&bfloat16, prompt, &reference.tokens);
        for (step, ((&pick, &expected), &gap)) in picks
            .iter()
            .zip(&reference.tokens)
            .zip(&reference.gaps)
            .enumerate()
        {
            steps += 1;
            if gap < NEAR_TIE_GAP {
                below_gap += 1;
            }
            if pick != expected {
                println!(
                    "bf16_agreement divergence prompt={index} step={step} \
                     float32={expected} bfloat16={pick} float32_gap={gap:.4}"
                );
                divergences.push(gap);
            }
        }
    }
    println!(
        "bf16_agreement steps={steps} divergences={} rate={:.4} \
         steps_below_gap={below_gap} near_tie_gap={NEAR_TIE_GAP}",
        divergences.len(),
        f64::from(u32::try_from(divergences.len()).expect("count fits u32")) / f64::from(steps),
    );
    assert!(
        divergences.iter().all(|&gap| gap < NEAR_TIE_GAP),
        "BF16 diverged from float32 at a step with a float32 gap of at least {NEAR_TIE_GAP}: {divergences:?}"
    );
}
