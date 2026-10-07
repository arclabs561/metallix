//! Qwen3-0.6B prompt-lookup speculation probe on a local checkpoint.
//!
//! Ignored: it needs `METALLIX_QWEN_MODEL`, and its timings are host-side
//! single-stream indications, not a serving benchmark. The baseline is the
//! default serving path: BF16 weights and pipelined GPU-greedy decode. It
//! prints the first token where speculation, verified from host logit rows
//! or from GPU picks, differs from that baseline: in BF16 a chunked verify
//! and a one-token decode round differently, so a near tie can flip. Exact
//! greedy identity is asserted in f32 by `speculative_verify`.

use std::{
    env,
    path::PathBuf,
    time::{Duration, Instant},
};

use engine::speculative::{
    DraftLength, GreedySpeculativeTarget, Pick, PositionLogits, PromptLookup, SpeculationStats,
    SpeculativeTarget, VerifyCost, greedy_speculative_step, speculative_step,
};

use crate::{GPU_TEST_LOCK, metal::Qwen3MlxWeights};

use super::super::{Qwen3FloatPrecision, Qwen3ForwardError, Qwen3ForwardExecutor};

const CONTEXT_TOKENS: usize = 4_096;
const KV_BYTES: u64 = 1024 * 1024 * 1024;
const OUTPUT_TOKENS: usize = 256;
const REPEATS: usize = 3;
const IM_END: i32 = 151_645;

const VERBATIM_EDIT: &str = r#"<|im_start|>user
Rename the function `parse_header` to `read_header` everywhere in this file and return the complete file, unchanged otherwise.

```python
import struct


def parse_header(data: bytes) -> dict:
    magic, version, count = struct.unpack_from("<4sHH", data, 0)
    if magic != b"MTLX":
        raise ValueError("bad magic")
    return {"version": version, "count": count}


def parse_records(data: bytes) -> list:
    header = parse_header(data)
    records = []
    offset = 8
    for _ in range(header["count"]):
        key, value = struct.unpack_from("<II", data, offset)
        records.append((key, value))
        offset += 8
    return records


def summarize(data: bytes) -> str:
    header = parse_header(data)
    records = parse_records(data)
    total = sum(value for _, value in records)
    return f"v{header['version']}: {len(records)} records, total {total}"
```<|im_end|>
<|im_start|>assistant
<think>

</think>

"#;

const PARAPHRASE: &str = r"<|im_start|>user
Paraphrase this paragraph in plain words, keeping every fact:

Speculative decoding lets a large language model emit several tokens per forward pass. A cheap drafter proposes a short continuation, the large model scores all proposed positions at once, and the longest prefix that agrees with the large model's own choices is kept. Because every kept token is one the large model would have produced anyway, the output distribution does not change; only the number of sequential model calls falls.<|im_end|>
<|im_start|>assistant
<think>

</think>

";

struct Target<'e, 'a>(&'e mut Qwen3ForwardExecutor<'a, std::collections::hash_map::RandomState>);

impl SpeculativeTarget for Target<'_, '_> {
    type Error = Qwen3ForwardError;

    fn cached_tokens(&self) -> usize {
        self.0.cached_tokens()
    }

    fn verify(&mut self, tokens: &[i32]) -> Result<PositionLogits, Qwen3ForwardError> {
        let rows = self.0.extend_all_logits(tokens)?;
        let vocab = rows.vocab_size();
        Ok(PositionLogits::new(rows.into_values(), vocab).expect("whole rows"))
    }

    fn truncate(&mut self, tokens: usize) -> Result<(), Qwen3ForwardError> {
        self.0.truncate_cached_tokens(tokens)
    }
}

impl GreedySpeculativeTarget for Target<'_, '_> {
    fn verify_greedy(&mut self, tokens: &[i32]) -> Result<Vec<i32>, Qwen3ForwardError> {
        self.0.extend_greedy(tokens)?.wait()
    }
}

/// How a speculative step verifies its draft.
#[derive(Clone, Copy, Debug)]
enum Verify {
    /// Read every logit row back and pick on the host, as sampled or
    /// constrained turns must.
    HostRows,
    /// Pick on the GPU and read back only token IDs.
    GpuGreedy,
}

fn argmax(logits: &[f32]) -> i32 {
    let mut best = 0;
    for (index, &value) in logits.iter().enumerate() {
        if value > logits[best] {
            best = index;
        }
    }
    i32::try_from(best).expect("vocabulary fits i32")
}

struct Run {
    tokens: Vec<i32>,
    decode: Duration,
    stats: SpeculationStats,
    target_calls: usize,
}

/// The default serving loop: GPU greedy picks with step `t + 1` queued
/// before step `t` is read back.
fn plain(weights: &Qwen3MlxWeights, prompt: &[i32]) -> Run {
    let mut executor = weights
        .resident_chat_executor(CONTEXT_TOKENS, KV_BYTES)
        .expect("executor");
    let logits = executor.prefill_last_logits(prompt).expect("prefill");
    let started = Instant::now();
    let mut tokens = vec![argmax(&logits)];
    let mut target_calls = 0;
    let mut pending = Some(executor.decode_greedy(tokens[0]).expect("decode"));
    while let Some(current) = pending.take() {
        target_calls += 1;
        if tokens.len() + 1 < OUTPUT_TOKENS {
            pending = Some(
                executor
                    .decode_greedy_after(&current)
                    .expect("queued decode"),
            );
        }
        let token = current.wait_one().expect("pick");
        tokens.push(token);
        if token == IM_END {
            break;
        }
    }
    Run {
        tokens,
        decode: started.elapsed(),
        stats: SpeculationStats::default(),
        target_calls,
    }
}

fn speculative(weights: &Qwen3MlxWeights, prompt: &[i32], cost: VerifyCost, verify: Verify) -> Run {
    let mut executor = weights
        .resident_chat_executor(CONTEXT_TOKENS, KV_BYTES)
        .expect("executor");
    let first = executor.prefill_last_logits(prompt).expect("prefill");
    let started = Instant::now();
    let lookup = PromptLookup::default();
    let mut length = DraftLength::new(8, cost).expect("draft range");
    let mut history = prompt.to_vec();
    let mut tokens = vec![argmax(&first)];
    history.push(tokens[0]);
    let mut stats = SpeculationStats::default();
    let mut target_calls = 0;
    while tokens.last() != Some(&IM_END) && tokens.len() < OUTPUT_TOKENS {
        let limit = length.next().min(OUTPUT_TOKENS - tokens.len() - 1);
        let draft = lookup.propose(&history, limit).to_vec();
        if draft.is_empty() {
            length.idle();
        }
        let last = *tokens.last().expect("first token");
        let mut target = Target(&mut executor);
        let outcome = match verify {
            Verify::HostRows => {
                let mut pick = |row: &[f32]| {
                    let token = argmax(row);
                    Ok::<_, std::convert::Infallible>(Pick {
                        token,
                        stop: token == IM_END,
                    })
                };
                speculative_step(&mut target, last, &draft, &mut pick).expect("verify")
            }
            Verify::GpuGreedy => {
                greedy_speculative_step(&mut target, last, &draft, &mut |token| token == IM_END)
                    .expect("verify")
            }
        };
        target_calls += 1;
        length.observe(outcome.drafted, outcome.accepted);
        stats.record(&outcome);
        history.extend(&outcome.emitted);
        tokens.extend(&outcome.emitted);
    }
    Run {
        tokens,
        decode: started.elapsed(),
        stats,
        target_calls,
    }
}

/// Median wall time of one verify of `rows` tokens after the prompt.
fn verify_time(weights: &Qwen3MlxWeights, prompt: &[i32], rows: usize, verify: Verify) -> f64 {
    let mut samples = Vec::new();
    for _ in 0..7 {
        let mut executor = weights
            .resident_chat_executor(CONTEXT_TOKENS, KV_BYTES)
            .expect("executor");
        executor.prefill_last_logits(prompt).expect("prefill");
        let chunk = vec![1_000; rows];
        let started = Instant::now();
        match verify {
            Verify::HostRows => {
                let _ = executor.extend_all_logits(&chunk).expect("verify");
            }
            Verify::GpuGreedy => {
                let _ = executor
                    .extend_greedy(&chunk)
                    .expect("verify")
                    .wait()
                    .expect("picks");
            }
        }
        samples.push(started.elapsed().as_secs_f64());
    }
    samples.sort_by(f64::total_cmp);
    samples[samples.len() / 2]
}

/// Fits `relative(k) = 1 + fixed + per_token * k` in units of one
/// pipelined decode token, through k = 1 and k = 8.
fn fit_cost(weights: &Qwen3MlxWeights, prompt: &[i32], step: f64, verify: Verify) -> VerifyCost {
    let mut relative = Vec::new();
    for rows in [1_usize, 2, 3, 5, 9] {
        let seconds = verify_time(weights, prompt, rows, verify);
        println!(
            "speculation_probe verify={verify:?} rows={rows} median_ms={:.3} relative_to_pipelined_token={:.3}",
            seconds * 1e3,
            seconds / step,
        );
        relative.push(seconds / step);
    }
    let per_token = ((relative[4] - relative[1]) / 7.0).max(0.0);
    let fixed = (relative[1] - 1.0 - per_token).max(0.0);
    println!("speculation_probe verify={verify:?} fixed={fixed:.3} per_token={per_token:.3}");
    VerifyCost::new(fixed, per_token).expect("fitted cost")
}

fn tokens_per_second(run: &Run) -> f64 {
    #[allow(clippy::cast_precision_loss, reason = "small token counts")]
    let emitted = (run.tokens.len() - 1) as f64;
    emitted / run.decode.as_secs_f64()
}

fn load_average() -> String {
    std::process::Command::new("sysctl")
        .args(["-n", "vm.loadavg"])
        .output()
        .map_or_else(
            |error| format!("unavailable: {error}"),
            |output| String::from_utf8_lossy(&output.stdout).trim().to_owned(),
        )
}

/// Top-1 and top-2 token IDs and the logit gap between them.
fn top_two(logits: &[f32]) -> (i32, i32, f32) {
    let mut order: Vec<usize> = (0..logits.len()).collect();
    order.sort_by(|&left, &right| {
        logits[right]
            .total_cmp(&logits[left])
            .then(left.cmp(&right))
    });
    let id = |index: usize| i32::try_from(order[index]).expect("vocabulary fits i32");
    (id(0), id(1), logits[order[0]] - logits[order[1]])
}

/// Reports the smallest top-1/top-2 logit gaps along the baseline output,
/// scored one token at a time as plain decode does, and the gap at
/// `divergence` both that way and as one chunk, as a verify scores it. A
/// divergence between plain and speculative decoding is numerics only at a
/// small gap.
fn report_gaps(
    weights: &Qwen3MlxWeights,
    prompt: &[i32],
    emitted: &[i32],
    divergence: Option<usize>,
    name: &str,
) {
    let mut stepped = weights
        .resident_chat_executor(CONTEXT_TOKENS, KV_BYTES)
        .expect("executor");
    let mut logits = stepped.prefill_last_logits(prompt).expect("prefill");
    // Entry `i` scores output token `i`, which follows `emitted[..i]`.
    let mut gaps = vec![top_two(&logits)];
    for &token in &emitted[..emitted.len() - 1] {
        logits = stepped.decode_last_logits(token).expect("decode");
        gaps.push(top_two(&logits));
    }
    let mut smallest: Vec<(usize, f32)> = gaps
        .iter()
        .enumerate()
        .map(|(index, gap)| (index, gap.2))
        .collect();
    smallest.sort_by(|left, right| left.1.total_cmp(&right.1));
    let listed: Vec<String> = smallest
        .iter()
        .take(3)
        .map(|(index, gap)| format!("{index}:{gap:.4}"))
        .collect();
    println!(
        "speculation_probe prompt={name} smallest_one_token_gaps={}",
        listed.join(",")
    );
    if let Some(position) = divergence.filter(|&position| position > 0) {
        let mut chunked = weights
            .resident_chat_executor(CONTEXT_TOKENS, KV_BYTES)
            .expect("executor");
        chunked.prefill_last_logits(prompt).expect("prefill");
        let rows = chunked
            .extend_all_logits(&emitted[..position])
            .expect("chunk");
        let (chunk_top, chunk_second, chunk_gap) = top_two(rows.row(position - 1).expect("row"));
        let (step_top, step_second, step_gap) = gaps[position];
        println!(
            "speculation_probe prompt={name} divergence_position={position} \
             one_token_top={step_top} second={step_second} gap={step_gap:.4} \
             chunk_top={chunk_top} second={chunk_second} gap={chunk_gap:.4}"
        );
    }
}

fn median(mut values: Vec<f64>) -> f64 {
    values.sort_by(f64::total_cmp);
    values[values.len() / 2]
}

#[test]
#[ignore = "requires METALLIX_QWEN_MODEL pointing to Qwen3-0.6B on Apple-Silicon Metal"]
fn prompt_lookup_speculation_matches_greedy_and_reports_speed() {
    let model = env::var_os("METALLIX_QWEN_MODEL")
        .map(PathBuf::from)
        .expect("METALLIX_QWEN_MODEL is required for this ignored checkpoint probe");
    let _gpu = GPU_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let tokenizer =
        tokenizers::Tokenizer::from_file(model.join("tokenizer.json")).expect("tokenizer");
    let mut weights = Qwen3MlxWeights::load(&model).expect("checkpoint load");
    weights
        .prepare_precision(Qwen3FloatPrecision::BFloat16)
        .expect("serving precision");
    println!(
        "speculation_probe weights=bf16 load_average_start={}",
        load_average()
    );

    let encode = |text: &str| -> Vec<i32> {
        tokenizer
            .encode(text, false)
            .expect("encode")
            .get_ids()
            .iter()
            .map(|&id| i32::try_from(id).expect("token id"))
            .collect()
    };

    let cost_prompt = encode(VERBATIM_EDIT);
    let _ = plain(&weights, &cost_prompt);
    let warm = plain(&weights, &cost_prompt);
    #[allow(clippy::cast_precision_loss, reason = "small token counts")]
    let step = warm.decode.as_secs_f64() / warm.target_calls as f64;
    println!("speculation_probe pipelined_token_ms={:.3}", step * 1e3);
    let costs = [
        (
            Verify::HostRows,
            fit_cost(&weights, &cost_prompt, step, Verify::HostRows),
        ),
        (
            Verify::GpuGreedy,
            fit_cost(&weights, &cost_prompt, step, Verify::GpuGreedy),
        ),
    ];

    for (name, text) in [("verbatim_edit", VERBATIM_EDIT), ("paraphrase", PARAPHRASE)] {
        let prompt = encode(text);
        let _ = plain(&weights, &prompt);
        for &(verify, cost) in &costs {
            let _ = speculative(&weights, &prompt, cost, verify);
        }
        let mut plain_rates = Vec::new();
        let mut spec_rates = [Vec::new(), Vec::new()];
        for repeat in 0..REPEATS {
            let reference = plain(&weights, &prompt);
            plain_rates.push(tokens_per_second(&reference));
            for (slot, &(verify, cost)) in costs.iter().enumerate() {
                let run = speculative(&weights, &prompt, cost, verify);
                let divergence = reference
                    .tokens
                    .iter()
                    .zip(&run.tokens)
                    .position(|(left, right)| left != right);
                println!(
                    "speculation_probe prompt={name} repeat={repeat} verify={verify:?} prompt_tokens={} \
                     plain_tokens={} plain_calls={} plain_tok_s={:.1} \
                     spec_tokens={} spec_calls={} spec_tok_s={:.1} \
                     verify_steps={} drafted={} accepted={} acceptance={:.3} first_divergence={divergence:?}",
                    prompt.len(),
                    reference.tokens.len(),
                    reference.target_calls,
                    tokens_per_second(&reference),
                    run.tokens.len(),
                    run.target_calls,
                    tokens_per_second(&run),
                    run.stats.verify_steps,
                    run.stats.drafted_tokens,
                    run.stats.accepted_tokens,
                    run.stats.acceptance_rate().unwrap_or(0.0),
                );
                spec_rates[slot].push(tokens_per_second(&run));
            }
        }
        let reference = plain(&weights, &prompt);
        let run = speculative(&weights, &prompt, costs[1].1, Verify::GpuGreedy);
        let divergence = reference
            .tokens
            .iter()
            .zip(&run.tokens)
            .position(|(left, right)| left != right);
        report_gaps(&weights, &prompt, &reference.tokens, divergence, name);
        let plain_median = median(plain_rates);
        for (slot, &(verify, _)) in costs.iter().enumerate() {
            let spec_median = median(spec_rates[slot].clone());
            println!(
                "speculation_probe prompt={name} verify={verify:?} median_plain_pipelined_tok_s={plain_median:.1} \
                 median_spec_tok_s={spec_median:.1} ratio={:.2}",
                spec_median / plain_median,
            );
        }
    }
    println!("speculation_probe load_average_end={}", load_average());
}
