//! CLI driver for an uncached numerical-forward measurement.

use std::{fs, path::Path, process::ExitCode, time::Instant};

use qwen::metal::Qwen3MlxWeights;
use serde_json::json;

use crate::parity::{compare_logits, read_reference};

#[derive(serde::Deserialize)]
struct GenerationConfig {
    eos_token_id: i32,
}

pub(crate) fn generate(
    model: &Path,
    input_ids: &[i32],
    max_tokens: u32,
    verify_cache: bool,
    verbose: bool,
) -> ExitCode {
    match generate_inner(model, input_ids, max_tokens, verify_cache, verbose) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("Qwen generation failed: {error}");
            ExitCode::FAILURE
        }
    }
}

fn generate_inner(
    model: &Path,
    input_ids: &[i32],
    max_tokens: u32,
    verify_cache: bool,
    verbose: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    if input_ids.len().saturating_add(max_tokens as usize) > qwen::forward::MAX_DENSE_DEBUG_TOKENS {
        return Err("prompt plus generation budget exceeds diagnostic context limit".into());
    }
    if verbose {
        let diagnostic = verbose_preflight(input_ids.len(), max_tokens, verify_cache);
        eprintln!("{diagnostic}");
    }
    let raw = fs::read_to_string(model.join("config.json"))?;
    let generation: GenerationConfig = serde_json::from_str(&raw)?;
    let started = Instant::now();
    let mut weights = Qwen3MlxWeights::load(model)?;
    weights.prepare_float32()?;
    let load_ms = started.elapsed().as_secs_f64() * 1000.0;
    if verbose {
        let logical_weight_bytes = weights.logical_weight_bytes();
        eprintln!(
            "qwen generation diagnostic: phase=load load_ms={load_ms:.3} logical_weight_bytes={logical_weight_bytes}",
        );
    }
    let mut executor = weights.executor();
    let started = Instant::now();
    let mut logits = executor.prefill_last_logits(input_ids)?;
    let prefill_ms = started.elapsed().as_secs_f64() * 1000.0;
    if verbose {
        let cached_tokens = executor.cached_tokens();
        let logical_kv_bytes = executor.kv_bytes();
        eprintln!(
            "qwen generation diagnostic: phase=prefill prefill_ms={prefill_ms:.3} cached_tokens={cached_tokens} logical_kv_bytes={logical_kv_bytes}",
        );
    }
    let mut prefix = input_ids.to_vec();
    let mut generated = Vec::new();
    let mut decode_ms = Vec::new();
    let mut comparisons = Vec::new();
    let mut finish_reason = "length";
    for step in 0..max_tokens {
        if verify_cache {
            let full = weights.forward_last_logits(&prefix)?;
            let result = compare_logits(&logits, &full)?;
            if !result.passed() {
                return Err(format!(
                    "cached/full logits differ at generation step {step}: {result:?}"
                )
                .into());
            }
            comparisons.push(result);
        }
        let token = greedy_token(&logits)?;
        generated.push(token);
        if token == generation.eos_token_id {
            finish_reason = "eos";
            break;
        }
        if step + 1 < max_tokens {
            prefix.push(token);
            let started = Instant::now();
            logits = executor.decode_last_logits(token)?;
            decode_ms.push(started.elapsed().as_secs_f64() * 1000.0);
        }
    }
    if verbose {
        let decode_total_ms = decode_ms.iter().sum::<f64>();
        let decode_steps = decode_ms.len();
        let cached_tokens = executor.cached_tokens();
        let logical_kv_bytes = executor.kv_bytes();
        eprintln!(
            "qwen generation diagnostic: phase=decode decode_steps={decode_steps} decode_total_ms={decode_total_ms:.3} cached_tokens={cached_tokens} logical_kv_bytes={logical_kv_bytes}",
        );
    }
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "schema_version": 1,
            "operation": "qwen3_greedy_cached_generation",
            "backend": "mlx-rs 0.25.3 Metal float32",
            "input_ids": input_ids,
            "generated_ids": generated,
            "finish_reason": finish_reason,
            "load_ms": load_ms,
            "prefill_ms": prefill_ms,
            "decode_ms": decode_ms,
            "cached_tokens": executor.cached_tokens(),
            "logical_kv_bytes": executor.kv_bytes(),
            "logical_weight_bytes": weights.logical_weight_bytes(),
            "cache_comparisons": comparisons,
            "scope": "single sequence; contiguous KV; first prefill not warmed; verification excluded from timed regions but may warm execution"
        }))?
    );
    Ok(())
}

fn verbose_preflight(prompt_tokens: usize, max_tokens: u32, verify_cache: bool) -> String {
    let context_limit = qwen::forward::MAX_DENSE_DEBUG_TOKENS;
    format!(
        "qwen generation diagnostic: phase=preflight prompt_tokens={prompt_tokens} max_tokens={max_tokens} context_limit={context_limit} verify_cache={verify_cache}",
    )
}

fn greedy_token(logits: &[f32]) -> Result<i32, String> {
    if logits.is_empty() || logits.iter().any(|value| !value.is_finite()) {
        return Err("greedy sampling requires finite nonempty logits".into());
    }
    let mut best = 0;
    for (index, &value) in logits.iter().enumerate().skip(1) {
        if value > logits[best] {
            best = index;
        }
    }
    i32::try_from(best).map_err(|error| error.to_string())
}

pub(crate) fn run(
    model: &Path,
    input_ids: &[i32],
    reference: Option<(&Path, &Path)>,
    repeats: u32,
) -> ExitCode {
    match measure(model, input_ids, reference, repeats) {
        Ok(passed) => {
            if passed {
                ExitCode::SUCCESS
            } else {
                ExitCode::FAILURE
            }
        }
        Err(error) => {
            eprintln!("Qwen forward failed: {error}");
            ExitCode::FAILURE
        }
    }
}

fn measure(
    model: &Path,
    input_ids: &[i32],
    reference: Option<(&Path, &Path)>,
    repeats: u32,
) -> Result<bool, Box<dyn std::error::Error>> {
    let started = Instant::now();
    let mut weights = Qwen3MlxWeights::load(model)?;
    weights.prepare_float32()?;
    let load_ms = started.elapsed().as_secs_f64() * 1000.0;
    let expected = reference
        .map(|(path, manifest)| {
            read_reference(
                path,
                manifest,
                model,
                input_ids,
                weights.inspection().contract().vocab_size() as usize,
            )
        })
        .transpose()?;
    let warmup_started = Instant::now();
    let mut logits = weights.forward_last_logits(input_ids)?;
    let warmup_ms = warmup_started.elapsed().as_secs_f64() * 1000.0;
    let mut samples = Vec::new();
    for _ in 0..repeats {
        let started = Instant::now();
        logits = weights.forward_last_logits(input_ids)?;
        samples.push(started.elapsed().as_secs_f64() * 1000.0);
    }
    if logits.iter().any(|value| !value.is_finite()) {
        return Err("non-finite forward logits".into());
    }
    let comparison = expected
        .as_ref()
        .map(|expected| compare_logits(&logits, expected))
        .transpose()?;
    let passed = comparison
        .as_ref()
        .is_none_or(crate::parity::LogitComparison::passed);
    let mut ranked: Vec<_> = logits.iter().copied().enumerate().collect();
    ranked.sort_by(|left, right| right.1.total_cmp(&left.1));
    ranked.truncate(8);
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "schema_version": 1,
            "operation": "uncached_qwen3_last_token_logits",
            "backend": "mlx-rs 0.25.3 Metal float32",
            "input_ids": input_ids,
            "vocabulary_logits": logits.len(),
        "checkpoint_payload_bytes": weights.inspection().tensor_bytes(),
        "logical_weight_bytes": weights.logical_weight_bytes(),
            "load_ms": load_ms,
            "excluded_warmup_ms": warmup_ms,
            "forward_ms": samples,
            "mean_forward_ms": samples.iter().sum::<f64>() / f64::from(repeats),
            "top8": ranked.iter().map(|(id, logit)| json!({"id": id, "logit": logit})).collect::<Vec<_>>(),
            "parity": comparison,
            "scope": "single sequence, no KV cache; timings include graph construction and final readback"
        }))?
    );
    Ok(passed)
}

#[cfg(test)]
mod tests {
    use super::verbose_preflight;

    #[test]
    fn verbose_preflight_is_deterministic_and_excludes_sensitive_arguments() {
        let diagnostic = verbose_preflight(3, 32, false);

        assert_eq!(
            diagnostic,
            "qwen generation diagnostic: phase=preflight prompt_tokens=3 max_tokens=32 context_limit=512 verify_cache=false"
        );
        assert!(!diagnostic.contains("model"));
        assert!(!diagnostic.contains("input_ids"));
        assert!(!diagnostic.contains("generated_ids"));
    }
}
