//! Opt-in agreement of a `qwen35` GGUF file with llama.cpp running the same
//! file, as written by `scripts/gguf-oracle.py`.
//!
//! Set `METALLIX_QWEN35_GGUF` to the `.gguf` file, `METALLIX_QWEN35_GGUF_ORACLE`
//! to the script's output directory and `METALLIX_QWEN35_GGUF_BASE` to the
//! Hugging Face checkpoint the file was converted from (its tokenizer is
//! used). For each prompt the test:
//!
//! - encodes the prompt with the chat format built from the GGUF file and
//!   requires llama.cpp's token IDs;
//! - feeds llama.cpp's tokens one at a time and compares every position's
//!   logits with llama.cpp's (`llama-results`): top-1 agreement, KL
//!   divergence and the llama.cpp margin at each disagreement;
//! - checks that chunked prefill of the whole prompt agrees with that last
//!   decode position;
//! - decodes greedily and compares with llama.cpp's greedy tokens.
//!
//! It also checks resident and peak memory against the stored tensors, and
//! stops if MLX memory passes `METALLIX_GGUF_GATE_GPU_GIB` (default 24).
//! Bounds were declared before the first run; every encoding loaded here
//! holds the file's values exactly, so only arithmetic differs.

#![cfg(feature = "metal")]

use std::{env, fs, path::PathBuf};

use chat_format::ChatFormat;
use checkpoint::{GgufEncoding, TensorSource, gguf::GgufFile};
use qwen35::forward::{Qwen35Executor, Qwen35Precision, Qwen35Weights};
use serde_json::Value;

const MIN_TOP1_AGREEMENT: f64 = 0.98;
const MAX_MEAN_KL: f64 = 1.0e-3;
const MAX_KL: f64 = 1.0e-2;
/// Largest llama.cpp top-1/top-2 margin at which top-1 may differ.
const MAX_DISAGREEMENT_MARGIN: f64 = 0.10;
const MAX_PREFILL_KL: f64 = 1.0e-3;
/// Greedy sequences may diverge only where metallix's own margin is below
/// this.
const MAX_GREEDY_DIVERGENCE_MARGIN: f64 = 0.25;
const RESIDENT_TOLERANCE: f64 = 0.05;
const PEAK_SLACK_BYTES: u64 = 256 << 20;

struct Paths {
    model: PathBuf,
    oracle: PathBuf,
    base: PathBuf,
}

fn paths() -> Option<Paths> {
    let var = |name: &str| env::var_os(name).map(PathBuf::from);
    let (Some(model), Some(oracle), Some(base)) = (
        var("METALLIX_QWEN35_GGUF"),
        var("METALLIX_QWEN35_GGUF_ORACLE"),
        var("METALLIX_QWEN35_GGUF_BASE"),
    ) else {
        eprintln!(
            "skipping: set METALLIX_QWEN35_GGUF, METALLIX_QWEN35_GGUF_ORACLE and METALLIX_QWEN35_GGUF_BASE"
        );
        return None;
    };
    Some(Paths {
        model,
        oracle,
        base,
    })
}

fn gpu_cap_bytes() -> usize {
    let gib: usize = env::var("METALLIX_GGUF_GATE_GPU_GIB")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(24);
    gib << 30
}

/// Stops the process when MLX holds more than the cap, so a leak cannot
/// fill memory.
fn check_gpu(cap: usize, at: &str) {
    let held = mlx_rs::memory::active_memory().expect("active memory")
        + mlx_rs::memory::cache_memory().expect("cache memory");
    assert!(
        held <= cap,
        "MLX holds {held} bytes at {at}, over the {cap}-byte cap"
    );
}

/// Natural-log softmax in f64.
fn log_softmax(logits: &[f32]) -> Vec<f64> {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let shifted: Vec<f64> = logits.iter().map(|&v| f64::from(v - max)).collect();
    let total = shifted.iter().map(|v| v.exp()).sum::<f64>().ln();
    shifted.iter().map(|v| v - total).collect()
}

/// KL(p || q) for log-probabilities.
fn kl(p: &[f64], q: &[f64]) -> f64 {
    p.iter().zip(q).map(|(a, b)| a.exp() * (a - b)).sum()
}

/// Index and value of the two largest logits.
fn top2(logits: &[f32]) -> (usize, f64) {
    let (mut best, mut second) = ((0, f32::NEG_INFINITY), f32::NEG_INFINITY);
    for (index, &value) in logits.iter().enumerate() {
        if value > best.1 {
            second = best.1;
            best = (index, value);
        } else if value > second {
            second = value;
        }
    }
    (best.0, f64::from(best.1) - f64::from(second))
}

struct Oracle {
    tokens: Vec<i32>,
    vocab: usize,
    logits: Vec<f32>,
}

fn read_oracle(path: &PathBuf) -> Oracle {
    let file = GgufFile::open(path).expect("oracle GGUF");
    let tokens: Vec<i32> = file
        .read("tokens", 1 << 20)
        .expect("tokens")
        .chunks_exact(4)
        .map(|b| i32::from_le_bytes(b.try_into().expect("4")))
        .collect();
    let info = file.tensor("logits").expect("logits tensor");
    assert_eq!(info.encoding(), GgufEncoding::F32);
    // llama-results declares the tensor with the token count innermost but
    // writes each position's vocabulary contiguously, so take the bytes as
    // `[tokens, vocab]` and derive the width from the element count.
    let elements = usize::try_from(info.elements()).expect("elements");
    assert_eq!(elements % tokens.len(), 0, "logits are tokens x vocab");
    let vocab = elements / tokens.len();
    let logits = file
        .read("logits", 4 << 30)
        .expect("logits")
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().expect("4")))
        .collect();
    Oracle {
        tokens,
        vocab,
        logits,
    }
}

/// Expected resident bytes: `Q8_0` matrices at 9 bits per weight (codes plus
/// binary16 scale and bias per 32), zero-centered norms in BF16, everything
/// else as stored.
fn expected_resident(file: &GgufFile) -> (u64, u64) {
    let (mut resident, mut largest) = (0_u64, 0_u64);
    for name in file.names() {
        let info = file.tensor(name).expect("listed");
        let stored = info.range().end - info.range().start;
        largest = largest.max(stored);
        let elements = info.elements();
        let zero_centered = [
            "attn_norm",
            "post_attention_norm",
            "attn_q_norm",
            "attn_k_norm",
            "output_norm",
        ]
        .iter()
        .any(|norm| name.ends_with(&format!("{norm}.weight")));
        resident += match info.encoding() {
            GgufEncoding::Q8_0 if info.shape().len() == 2 => elements * 9 / 8,
            _ if zero_centered => elements * 2,
            _ => stored,
        };
    }
    (resident, largest)
}

#[derive(Default)]
struct Totals {
    positions: usize,
    agree: usize,
    kl_sum: f64,
    kl_max: f64,
    worst_margin: f64,
    failures: Vec<String>,
}

fn teacher_forced(
    executor: &mut Qwen35Executor<'_>,
    oracle: &Oracle,
    name: &str,
    totals: &mut Totals,
) -> Vec<f32> {
    let mut last = Vec::new();
    for (position, &token) in oracle.tokens.iter().enumerate() {
        let logits = if position == 0 {
            executor.prefill_last_logits(&[token])
        } else {
            executor.decode_last_logits(token)
        }
        .expect("forward");
        let reference = &oracle.logits[position * oracle.vocab..(position + 1) * oracle.vocab];
        assert_eq!(logits.len(), reference.len(), "vocabulary width");
        let divergence = kl(&log_softmax(reference), &log_softmax(&logits));
        let (native, _) = top2(&logits);
        let (expected, margin) = top2(reference);
        totals.positions += 1;
        totals.kl_sum += divergence;
        totals.kl_max = totals.kl_max.max(divergence);
        if native == expected {
            totals.agree += 1;
        } else {
            totals.worst_margin = totals.worst_margin.max(margin);
            eprintln!(
                "{name} position {position}: top-1 {native} vs llama.cpp {expected} (llama margin {margin:.4}, KL {divergence:.2e})"
            );
            if margin > MAX_DISAGREEMENT_MARGIN {
                totals.failures.push(format!(
                    "{name} position {position}: top-1 differs at llama margin {margin:.4}"
                ));
            }
        }
        last = logits;
    }
    last
}

struct Run<'a> {
    weights: &'a Qwen35Weights,
    format: &'a ChatFormat,
    oracle_dir: &'a PathBuf,
    cap: usize,
}

/// Tokenization, teacher-forced logits, prefill agreement and greedy decode
/// for one prompt.
fn check_prompt(run: &Run<'_>, prompt: &Value, totals: &mut Totals, failures: &mut Vec<String>) {
    let name = prompt["name"].as_str().expect("name");
    let oracle = read_oracle(&run.oracle_dir.join(format!("oracle-{name}.gguf")));
    assert_eq!(oracle.vocab, run.weights.config().vocab_size());
    let encoded = run
        .format
        .encode(prompt["text"].as_str().expect("text"))
        .expect("encode");
    if encoded != oracle.tokens {
        failures.push(format!("{name}: tokenization differs from llama.cpp"));
    }

    let mut executor = run.weights.executor();
    let last = teacher_forced(&mut executor, &oracle, name, totals);
    check_gpu(run.cap, name);

    let mut prefill = run.weights.executor();
    let whole = prefill
        .prefill_last_logits(&oracle.tokens)
        .expect("prefill");
    let prefill_kl = kl(&log_softmax(&last), &log_softmax(&whole));
    let same_top = top2(&last).0 == top2(&whole).0;
    eprintln!(
        "{name}: prefill vs decode at the last position: KL {prefill_kl:.2e}, same top-1 {same_top}"
    );
    if !same_top || prefill_kl > MAX_PREFILL_KL {
        failures.push(format!(
            "{name}: prefill disagrees with decode (KL {prefill_kl:.2e})"
        ));
    }

    let greedy: Vec<usize> = prompt["llama_greedy"]
        .as_array()
        .expect("llama_greedy")
        .iter()
        .map(|id| usize::try_from(id.as_u64().expect("id")).expect("usize"))
        .collect();
    let mut logits = whole;
    for (step, &expected) in greedy.iter().enumerate() {
        let (native, margin) = top2(&logits);
        if native != expected {
            eprintln!(
                "{name} greedy step {step}: {native} vs llama.cpp {expected}, metallix margin {margin:.4}"
            );
            if margin >= MAX_GREEDY_DIVERGENCE_MARGIN {
                failures.push(format!(
                    "{name}: greedy diverges at step {step} with metallix margin {margin:.4}"
                ));
            }
            break;
        }
        logits = prefill
            .decode_last_logits(i32::try_from(native).expect("token id"))
            .expect("decode");
    }
    check_gpu(run.cap, name);
}

#[test]
fn gguf_logits_agree_with_llama_cpp() {
    let Some(paths) = paths() else { return };
    let cap = gpu_cap_bytes();
    let prompts: Value = serde_json::from_str(
        &fs::read_to_string(paths.oracle.join("prompts.json")).expect("prompts.json"),
    )
    .expect("prompts JSON");

    let file = GgufFile::open(&paths.model).expect("GGUF");
    // The peak is process-wide: run this test alone (`--exact`) for the
    // memory bound to mean anything.
    mlx_rs::memory::reset_peak_memory().expect("reset peak");
    let weights =
        Qwen35Weights::load_gguf(&paths.model, Qwen35Precision::Checkpoint).expect("load");
    let peak = u64::try_from(mlx_rs::memory::peak_memory().expect("peak")).expect("u64");
    check_gpu(cap, "load");
    let resident = u64::try_from(weights.logical_weight_bytes()).expect("u64");
    let (expected, largest) = expected_resident(&file);
    #[allow(clippy::cast_precision_loss, reason = "a ratio of byte counts")]
    let resident_ratio = resident as f64 / expected as f64;
    let peak_bound = resident + largest * 3 / 2 + PEAK_SLACK_BYTES;
    eprintln!(
        "memory: resident {resident} expected {expected} (ratio {resident_ratio:.4}), load peak {peak} bound {peak_bound}, largest stored tensor {largest}"
    );

    let format = ChatFormat::from_gguf(
        &paths.base,
        &file.metadata().tokenizer().expect("GGUF tokenizer"),
        weights.config().vocab_size(),
    )
    .expect("chat format from GGUF");

    let mut totals = Totals::default();
    let mut failures = Vec::new();
    for prompt in prompts["prompts"].as_array().expect("prompt list") {
        check_prompt(
            &Run {
                weights: &weights,
                format: &format,
                oracle_dir: &paths.oracle,
                cap,
            },
            prompt,
            &mut totals,
            &mut failures,
        );
    }

    #[allow(clippy::cast_precision_loss, reason = "position counts")]
    let (agreement, mean_kl) = (
        totals.agree as f64 / totals.positions as f64,
        totals.kl_sum / totals.positions as f64,
    );
    eprintln!(
        "GATE positions {} top1 {:.4} mean_kl {mean_kl:.3e} max_kl {:.3e} worst_disagreement_margin {:.4} resident_ratio {resident_ratio:.4} peak {peak} peak_bound {peak_bound}",
        totals.positions, agreement, totals.kl_max, totals.worst_margin
    );
    failures.extend(totals.failures);
    if agreement < MIN_TOP1_AGREEMENT {
        failures.push(format!(
            "top-1 agreement {agreement:.4} < {MIN_TOP1_AGREEMENT}"
        ));
    }
    if mean_kl > MAX_MEAN_KL {
        failures.push(format!("mean KL {mean_kl:.3e} > {MAX_MEAN_KL}"));
    }
    if totals.kl_max > MAX_KL {
        failures.push(format!("max KL {:.3e} > {MAX_KL}", totals.kl_max));
    }
    if (resident_ratio - 1.0).abs() > RESIDENT_TOLERANCE {
        failures.push(format!("resident/expected {resident_ratio:.4}"));
    }
    if peak > peak_bound {
        failures.push(format!("load peak {peak} > {peak_bound}"));
    }
    assert!(
        failures.is_empty(),
        "gate failures:\n{}",
        failures.join("\n")
    );
}

/// Every `Q8_0` matrix of the file, repacked and dequantized by MLX with
/// f32 parameters, equals the reference decoder bit for bit.
#[test]
fn gguf_q8_0_tensors_dequantize_exactly() {
    use blockfloat::gguf::{decode, repack_affine};
    use mlx_rs::{Array, Dtype, ops};

    let Some(paths) = paths() else { return };
    let cap = gpu_cap_bytes();
    let file = GgufFile::open(&paths.model).expect("GGUF");
    let (mut tensors, mut mismatches, mut values) = (0, 0_usize, 0_usize);
    for name in file.names() {
        let info = file.tensor(name).expect("listed");
        if info.encoding() != GgufEncoding::Q8_0 || info.shape().len() != 2 {
            continue;
        }
        let bytes = file.read(name, 8 << 30).expect("read");
        let reference = decode(GgufEncoding::Q8_0, &bytes).expect("decode");
        let repack = repack_affine(GgufEncoding::Q8_0, &bytes).expect("repack");
        let [rows, columns] = [info.shape()[0], info.shape()[1]]
            .map(|dim| i32::try_from(dim).expect("i32 dimension"));
        let codes = Array::from_slice(&repack.codes, &[rows, columns / 4]);
        let scales = Array::from_slice(&repack.scales, &[rows, columns / 32]);
        let biases = Array::from_slice(&repack.biases, &[rows, columns / 32]);
        let dequantized = ops::dequantize(&codes, &scales, &biases, 32, 8).expect("dequantize");
        assert_eq!(dequantized.dtype(), Dtype::Float32);
        dequantized.eval().expect("eval");
        check_gpu(cap, name);
        let wrong = dequantized
            .as_slice::<f32>()
            .iter()
            .zip(&reference)
            .filter(|(got, want)| got.to_bits() != want.to_bits())
            .count();
        if wrong > 0 {
            eprintln!("{name}: {wrong} of {} values differ", reference.len());
        }
        tensors += 1;
        mismatches += wrong;
        values += reference.len();
    }
    eprintln!("GATE-G1 {tensors} Q8_0 tensors, {values} values, {mismatches} bitwise mismatches");
    assert!(tensors > 0, "no Q8_0 matrices");
    assert_eq!(mismatches, 0);
}
