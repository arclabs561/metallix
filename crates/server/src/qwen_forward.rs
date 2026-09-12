//! CLI driver for an uncached numerical-forward measurement.

use std::{fs, path::Path, process::ExitCode, time::Instant};

use qwen::metal::Qwen3MlxWeights;
use serde_json::json;

use crate::parity::{compare_logits, read_reference};

#[derive(serde::Deserialize)]
struct GenerationConfig {
    eos_token_id: i32,
    vocab_size: usize,
    max_position_embeddings: usize,
}

impl GenerationConfig {
    fn validate(&self, input_ids: &[i32], max_tokens: u32) -> Result<(), String> {
        let maximum = self
            .max_position_embeddings
            .min(qwen::forward::MAX_DENSE_DEBUG_TOKENS);
        if input_ids.is_empty() || max_tokens == 0 {
            return Err(String::from(
                "prompt and generation budget must be nonempty",
            ));
        }
        let total_tokens = input_ids.len().saturating_add(max_tokens as usize);
        if total_tokens > maximum {
            return Err(format!(
                "model diagnostic requires prompt_tokens + max_tokens <= {maximum}; received {} + {max_tokens} = {total_tokens}",
                input_ids.len(),
            ));
        }
        if input_ids
            .iter()
            .chain(std::iter::once(&self.eos_token_id))
            .any(|&id| {
                usize::try_from(id)
                    .ok()
                    .is_none_or(|id| id >= self.vocab_size)
            })
        {
            return Err(String::from(
                "prompt or EOS token ID is outside model vocabulary",
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Copy)]
pub(crate) struct GenerationDiagnostics {
    pub(crate) verbose: bool,
    pub(crate) logprobs: bool,
    pub(crate) preview: bool,
}

/// Qwen-specific residency selection for the diagnostic generator.
///
/// This is deliberately not an engine-level execution policy: its streamed
/// form depends on this adapter's safetensors names, BF16 reader, and detached
/// Qwen KV layout.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum GenerationMemoryMode {
    Resident,
    Streamed,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct GenerationMemoryConfig {
    pub(crate) mode: GenerationMemoryMode,
    pub(crate) max_weight_bytes: Option<u64>,
    pub(crate) max_kv_bytes: Option<u64>,
    pub(crate) tile_rows: Option<usize>,
}

const DEFAULT_STREAMED_WEIGHT_BYTES: u64 = 81_798_144;
// The full qualified 32-token Qwen control allocation. This is a logical
// cap, not a claim that the executor allocates all of it up front.
const DEFAULT_STREAMED_KV_BYTES: u64 = 7_340_032;
const DEFAULT_STREAMED_TILE_ROWS: usize = 1_024;
const STREAMED_MAX_TOKENS: usize = 32;

#[derive(Clone, Copy, Debug)]
struct StreamedGenerationPlan {
    max_weight_bytes: u64,
    max_kv_bytes: u64,
    tile_rows: usize,
    maximum_total_tokens: usize,
}

impl GenerationMemoryConfig {
    fn streamed_plan(
        self,
        prompt_tokens: usize,
        max_tokens: u32,
    ) -> Result<Option<StreamedGenerationPlan>, String> {
        match self.mode {
            GenerationMemoryMode::Resident => {
                if self.max_weight_bytes.is_some()
                    || self.max_kv_bytes.is_some()
                    || self.tile_rows.is_some()
                {
                    return Err(String::from(
                        "streamed tuning flags require --memory-mode streamed",
                    ));
                }
                Ok(None)
            }
            GenerationMemoryMode::Streamed => {
                let maximum_total_tokens = prompt_tokens
                    .checked_add(max_tokens as usize)
                    .ok_or_else(|| String::from("prompt plus generation budget overflows"))?;
                if maximum_total_tokens > STREAMED_MAX_TOKENS {
                    return Err(format!(
                        "streamed generation requires prompt_tokens + max_tokens <= {STREAMED_MAX_TOKENS}; received {prompt_tokens} + {max_tokens} = {maximum_total_tokens}",
                    ));
                }
                Ok(Some(StreamedGenerationPlan {
                    max_weight_bytes: self
                        .max_weight_bytes
                        .unwrap_or(DEFAULT_STREAMED_WEIGHT_BYTES),
                    max_kv_bytes: self.max_kv_bytes.unwrap_or(DEFAULT_STREAMED_KV_BYTES),
                    tile_rows: self.tile_rows.unwrap_or(DEFAULT_STREAMED_TILE_ROWS),
                    maximum_total_tokens,
                }))
            }
        }
    }
}

/// A private adapter-local dispatch over the two Qwen cache implementations.
///
/// It is intentionally not a generic engine trait: a caller cannot move this
/// cache to a different checkpoint or model architecture.
enum GenerationExecutor<'a> {
    Resident(qwen::forward::Qwen3ForwardExecutor<'a, std::collections::hash_map::RandomState>),
    Streamed(Box<qwen::metal::Qwen3StreamExecutor>),
}

impl GenerationExecutor<'_> {
    fn enable_profiling(&mut self, enabled: bool) {
        if let Self::Streamed(executor) = self {
            executor.enable_profiling(enabled);
        }
    }

    fn take_last_profile(&mut self) -> Result<Option<serde_json::Value>, serde_json::Error> {
        match self {
            Self::Resident(_) => Ok(None),
            Self::Streamed(executor) => executor
                .take_last_profile()
                .map(serde_json::to_value)
                .transpose(),
        }
    }

    fn prefill_last_logits(
        &mut self,
        input_ids: &[i32],
    ) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        match self {
            Self::Resident(executor) => Ok(executor.prefill_last_logits(input_ids)?),
            Self::Streamed(executor) => Ok(executor.prefill_last_logits()?),
        }
    }

    fn decode_last_logits(
        &mut self,
        input_id: i32,
    ) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        match self {
            Self::Resident(executor) => Ok(executor.decode_last_logits(input_id)?),
            Self::Streamed(executor) => Ok(executor.decode_last_logits(input_id)?),
        }
    }

    fn cached_tokens(&self) -> usize {
        match self {
            Self::Resident(executor) => executor.cached_tokens(),
            Self::Streamed(executor) => executor.cached_tokens(),
        }
    }

    fn kv_bytes(&self) -> u64 {
        match self {
            Self::Resident(executor) => executor.kv_bytes() as u64,
            Self::Streamed(executor) => executor.kv_bytes(),
        }
    }

    fn planned_weight_bytes(&self) -> Option<u64> {
        match self {
            Self::Resident(_) => None,
            Self::Streamed(executor) => Some(executor.planned_weight_bytes()),
        }
    }
}

pub(crate) fn generate(
    model: &Path,
    input_ids: &[i32],
    max_tokens: u32,
    verify_cache: bool,
    memory: GenerationMemoryConfig,
    diagnostics: GenerationDiagnostics,
    #[cfg(feature = "structured-output")] json_schema: Option<&Path>,
) -> ExitCode {
    match generate_inner(
        model,
        input_ids,
        max_tokens,
        verify_cache,
        memory,
        diagnostics,
        #[cfg(feature = "structured-output")]
        json_schema,
    ) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("Qwen generation failed: {error}");
            ExitCode::FAILURE
        }
    }
}

#[allow(
    clippy::too_many_lines,
    reason = "linear diagnostic driver keeps model, verification and optional sampling timing boundaries explicit"
)]
fn generate_inner(
    model: &Path,
    input_ids: &[i32],
    max_tokens: u32,
    verify_cache: bool,
    memory: GenerationMemoryConfig,
    diagnostics: GenerationDiagnostics,
    #[cfg(feature = "structured-output")] json_schema: Option<&Path>,
) -> Result<(), Box<dyn std::error::Error>> {
    let GenerationDiagnostics {
        verbose,
        logprobs,
        preview,
    } = diagnostics;
    if input_ids.len().saturating_add(max_tokens as usize) > qwen::forward::MAX_DENSE_DEBUG_TOKENS {
        return Err("prompt plus generation budget exceeds diagnostic context limit".into());
    }
    let streamed = memory.streamed_plan(input_ids.len(), max_tokens)?;
    if verbose {
        let diagnostic = verbose_preflight(input_ids.len(), max_tokens, verify_cache, streamed);
        eprintln!("{diagnostic}");
    }
    let raw = fs::read_to_string(model.join("config.json"))?;
    let generation: GenerationConfig = serde_json::from_str(&raw)?;
    generation.validate(input_ids, max_tokens)?;
    #[cfg(feature = "structured-output")]
    let mut constraint = json_schema
        .map(|path| crate::qwen_constraints::ConstraintRun::load(model, path))
        .transpose()?;
    let started = Instant::now();
    let resident_weights = if streamed.is_none() {
        let mut weights = Qwen3MlxWeights::load(model)?;
        weights.prepare_float32()?;
        Some(weights)
    } else {
        None
    };
    let mut executor = match streamed {
        Some(plan) => {
            GenerationExecutor::Streamed(Box::new(qwen::metal::Qwen3StreamExecutor::new(
                model,
                input_ids,
                plan.maximum_total_tokens,
                plan.max_weight_bytes,
                plan.max_kv_bytes,
                plan.tile_rows,
            )?))
        }
        None => GenerationExecutor::Resident(
            resident_weights
                .as_ref()
                .ok_or("resident generation weights are unavailable")?
                .executor(),
        ),
    };
    let load_ms = started.elapsed().as_secs_f64() * 1000.0;
    // Stream profiles are opt-in diagnostics. Enable before prefill so each
    // cache-mutating candidate phase records exactly one consumed sample.
    executor.enable_profiling(verbose);
    if verbose {
        if let Some(plan) = streamed {
            eprintln!(
                "qwen generation diagnostic: phase=stream_setup setup_ms={load_ms:.3} max_weight_bytes={} max_kv_bytes={} tile_rows={} promised_total_tokens={}",
                plan.max_weight_bytes, plan.max_kv_bytes, plan.tile_rows, plan.maximum_total_tokens,
            );
        } else {
            let logical_weight_bytes = resident_weights
                .as_ref()
                .ok_or("resident generation weights are unavailable")?
                .logical_weight_bytes();
            eprintln!(
                "qwen generation diagnostic: phase=load load_ms={load_ms:.3} logical_weight_bytes={logical_weight_bytes}",
            );
        }
    }
    let started = Instant::now();
    let mut logits = executor.prefill_last_logits(input_ids)?;
    let prefill_ms = started.elapsed().as_secs_f64() * 1000.0;
    let mut phase_profiles = Vec::new();
    if verbose {
        record_stream_profile(&mut executor, &mut phase_profiles, "prefill", 0)?;
    }
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
    let mut token_scores = Vec::new();
    let mut finish_reason = "length";
    let mut streamed_oracle: Option<Qwen3MlxWeights> = None;
    let mut oracle_load_ms = None;
    for step in 0..max_tokens {
        if verify_cache {
            let full = if let Some(weights) = resident_weights.as_ref() {
                weights.forward_last_logits(&prefix)?
            } else {
                if streamed_oracle.is_none() {
                    let started = Instant::now();
                    let mut oracle = Qwen3MlxWeights::load(model)?;
                    oracle.prepare_float32()?;
                    oracle_load_ms = Some(started.elapsed().as_secs_f64() * 1000.0);
                    streamed_oracle = Some(oracle);
                }
                streamed_oracle
                    .as_ref()
                    .ok_or("streamed verification oracle is unavailable")?
                    .forward_last_logits(&prefix)?
            };
            let result = compare_logits(&logits, &full)?;
            if !result.passed() {
                return Err(format!(
                    "cached/full logits differ at generation step {step}: {result:?}"
                )
                .into());
            }
            comparisons.push(result);
        }
        #[cfg(feature = "structured-output")]
        let (token, scores) = match constraint.as_mut() {
            Some(constraint) => constraint.sample(&logits, logprobs)?,
            None => greedy_sample(&logits, logprobs)?,
        };
        #[cfg(not(feature = "structured-output"))]
        let (token, scores) = greedy_sample(&logits, logprobs)?;
        if let Some(mut scores) = scores {
            scores["token_id"] = json!(token);
            token_scores.push(scores);
        }
        generated.push(token);
        #[cfg(feature = "structured-output")]
        if constraint
            .as_ref()
            .is_some_and(crate::qwen_constraints::ConstraintRun::is_complete)
        {
            finish_reason = "grammar_complete";
            break;
        }
        if token == generation.eos_token_id {
            finish_reason = "eos";
            break;
        }
        if step + 1 < max_tokens {
            prefix.push(token);
            let started = Instant::now();
            logits = executor.decode_last_logits(token)?;
            decode_ms.push(started.elapsed().as_secs_f64() * 1000.0);
            if verbose {
                record_stream_profile(&mut executor, &mut phase_profiles, "decode", step + 1)?;
            }
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
        emit_stream_profile_summary(&phase_profiles);
    }
    let mut report = json!({
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
    "cache_comparisons": comparisons,
    "scope": "single sequence; contiguous KV; first prefill not warmed; verification excluded from timed regions but may warm execution"
    });
    if let Some(plan) = streamed {
        report["operation"] = json!("qwen3_greedy_streamed_generation");
        report["streamed"] = json!({
            "maximum_total_tokens": plan.maximum_total_tokens,
            "max_weight_bytes": plan.max_weight_bytes,
            "max_kv_bytes": plan.max_kv_bytes,
            "tile_rows": plan.tile_rows,
            "planned_weight_and_staging_bytes": executor.planned_weight_bytes(),
            "verification": if verify_cache { "resident_full_prefix_oracle" } else { "not_requested" },
            "oracle_memory_excluded": verify_cache,
            "oracle_load_ms_excluded": oracle_load_ms,
            "scope": "layer-streamed Qwen adapter path; promised context and separate planned weight/KV budgets are checked before candidate payload reads; planned budgets exclude activations, operator scratch, allocator retention, headers, projection output, and any opt-in resident oracle"
        });
        attach_verbose_phase_profiles(&mut report["streamed"], verbose, &phase_profiles);
        report["scope"] = json!(
            "single sequence; streamed Qwen layers with detached contiguous KV; first prefill not warmed; candidate timing excludes opt-in resident full-prefix verification and its memory"
        );
    } else {
        let logical_weight_bytes = resident_weights
            .as_ref()
            .ok_or("resident generation weights are unavailable")?
            .logical_weight_bytes();
        report["logical_weight_bytes"] = json!(logical_weight_bytes);
    }
    if logprobs {
        report["logprobs"] = json!({
            "log_base": "e",
            "tokens": token_scores,
            "scope": "selected-token probabilities under temperature-one model logits; constrained scores renormalize the current allowed set; not the probability of a complete valid sequence or of the deterministic greedy policy"
        });
    }
    #[cfg(feature = "structured-output")]
    let report = {
        let mut report = report;
        if let Some(constraint) = constraint.as_ref() {
            report["operation"] = json!(if streamed.is_some() {
                "qwen3_constrained_streamed_generation"
            } else {
                "qwen3_constrained_cached_generation"
            });
            report["constraint"] = constraint.report(verbose)?;
        }
        report
    };
    if preview {
        crate::generation_preview::emit(&report);
    }
    println!("{}", serde_json::to_string_pretty(&report)?);
    #[cfg(feature = "structured-output")]
    if constraint
        .as_ref()
        .is_some_and(|constraint| !constraint.is_complete())
    {
        return Err(
            "generation budget ended before grammar completion; output is incomplete".into(),
        );
    }
    Ok(())
}

fn verbose_preflight(
    prompt_tokens: usize,
    max_tokens: u32,
    verify_cache: bool,
    streamed: Option<StreamedGenerationPlan>,
) -> String {
    let context_limit = streamed.map_or(qwen::forward::MAX_DENSE_DEBUG_TOKENS, |_| {
        STREAMED_MAX_TOKENS
    });
    format!(
        "qwen generation diagnostic: phase=preflight prompt_tokens={prompt_tokens} max_tokens={max_tokens} context_limit={context_limit} verify_cache={verify_cache}",
    )
}

/// Moves the one retained streamed profile into the verbose report immediately
/// after its cache-mutating operation. Resident execution intentionally has no
/// profile surface.
fn record_stream_profile(
    executor: &mut GenerationExecutor<'_>,
    phase_profiles: &mut Vec<serde_json::Value>,
    phase: &'static str,
    generation_step: u32,
) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(profile) = executor.take_last_profile()? {
        phase_profiles.push(json!({
            "phase": phase,
            "generation_step": generation_step,
            "cached_tokens": executor.cached_tokens(),
            "profile": profile,
        }));
    }
    Ok(())
}

/// Prints only aggregate profile metadata; individual samples remain in the
/// verbose JSON report. These host wall-clock values are not GPU-kernel timing.
fn emit_stream_profile_summary(phase_profiles: &[serde_json::Value]) {
    if phase_profiles.is_empty() {
        return;
    }
    let total_ms = phase_profiles
        .iter()
        .filter_map(|entry| entry["profile"]["total_ms"].as_f64())
        .sum::<f64>();
    let layer_execute_readback_ms = phase_profiles
        .iter()
        .filter_map(|entry| entry["profile"]["layer_execute_readback_ms"].as_f64())
        .sum::<f64>();
    let layer_load_conversion_ms = phase_profiles
        .iter()
        .filter_map(|entry| entry["profile"]["layer_load_conversion_ms"].as_f64())
        .sum::<f64>();
    let tiled_projection_load_ms = phase_profiles
        .iter()
        .filter_map(|entry| entry["profile"]["tiled_projection_load_ms"].as_f64())
        .sum::<f64>();
    let tiled_projection_execute_ms = phase_profiles
        .iter()
        .filter_map(|entry| entry["profile"]["tiled_projection_execute_ms"].as_f64())
        .sum::<f64>();
    eprintln!(
        "qwen generation diagnostic: phase=stream_profile samples={} total_ms={total_ms:.3} layer_load_conversion_ms={layer_load_conversion_ms:.3} layer_execute_readback_ms={layer_execute_readback_ms:.3} tiled_projection_load_ms={tiled_projection_load_ms:.3} tiled_projection_execute_ms={tiled_projection_execute_ms:.3} scope=host_wall_clock_not_gpu_kernel_timing",
        phase_profiles.len(),
    );
}

/// The profile list is opt-in and stream-only: keeping this mutation in one
/// place prevents verbose diagnostics from quietly changing normal JSON.
fn attach_verbose_phase_profiles(
    streamed: &mut serde_json::Value,
    verbose: bool,
    phase_profiles: &[serde_json::Value],
) {
    if verbose {
        streamed["phase_profiles"] = json!(phase_profiles);
    }
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

fn greedy_sample(
    logits: &[f32],
    logprobs: bool,
) -> Result<(i32, Option<serde_json::Value>), String> {
    let token = greedy_token(logits)?;
    let scores = if logprobs {
        let selected =
            f64::from(logits[usize::try_from(token).map_err(|error| error.to_string())?]);
        let shifted_sum = logits
            .iter()
            .map(|&value| (f64::from(value) - selected).exp())
            .sum::<f64>();
        Some(json!({ "model_logprob": -shifted_sum.ln() }))
    } else {
        None
    };
    Ok((token, scores))
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
    use super::{GenerationMemoryConfig, GenerationMemoryMode, verbose_preflight};

    #[test]
    fn logprobs_are_opt_in_stable_and_do_not_change_greedy_ties() {
        let (token, scores) = super::greedy_sample(&[3.0, 3.0], true).expect("finite logits");
        assert_eq!(token, 0);
        let score = scores.expect("requested scores")["model_logprob"]
            .as_f64()
            .expect("numeric");
        assert!((score + 2.0_f64.ln()).abs() < 1e-12);
        assert_eq!(
            super::greedy_sample(&[3.0, 3.0], false).expect("finite logits"),
            (0, None)
        );
        let (_, extreme) =
            super::greedy_sample(&[-f32::MAX, f32::MAX], true).expect("finite extremes");
        assert!(
            extreme.expect("scores")["model_logprob"]
                .as_f64()
                .expect("finite number")
                .abs()
                < f64::EPSILON
        );
    }

    #[test]
    fn generation_preflight_enforces_the_model_limit_before_loading_weights() {
        let config = super::GenerationConfig {
            eos_token_id: 7,
            vocab_size: 8,
            max_position_embeddings: 16,
        };
        assert!(config.validate(&[1, 2], 14).is_ok());
        assert_eq!(
            config.validate(&[1, 2], 15),
            Err(String::from(
                "model diagnostic requires prompt_tokens + max_tokens <= 16; received 2 + 15 = 17"
            ))
        );
        assert!(config.validate(&[], 1).is_err());
        assert!(config.validate(&[-1], 1).is_err());
        assert!(config.validate(&[8], 1).is_err());
        assert!(config.validate(&[1], 0).is_err());
    }

    #[test]
    fn verbose_preflight_is_deterministic_and_excludes_sensitive_arguments() {
        let diagnostic = verbose_preflight(3, 32, false, None);

        assert_eq!(
            diagnostic,
            "qwen generation diagnostic: phase=preflight prompt_tokens=3 max_tokens=32 context_limit=512 verify_cache=false"
        );
        assert!(!diagnostic.contains("model"));
        assert!(!diagnostic.contains("input_ids"));
        assert!(!diagnostic.contains("generated_ids"));
    }

    #[test]
    fn streamed_generation_budgets_the_promised_context_and_rejects_resident_tuning() {
        let streamed = GenerationMemoryConfig {
            mode: GenerationMemoryMode::Streamed,
            max_weight_bytes: Some(81_798_144),
            max_kv_bytes: Some(1_376_256),
            tile_rows: Some(1_024),
        }
        .streamed_plan(3, 29)
        .expect("qualified streamed bound")
        .expect("streamed plan");
        assert_eq!(streamed.maximum_total_tokens, 32);
        assert_eq!(streamed.max_weight_bytes, 81_798_144);
        let defaults = GenerationMemoryConfig {
            mode: GenerationMemoryMode::Streamed,
            max_weight_bytes: None,
            max_kv_bytes: None,
            tile_rows: None,
        }
        .streamed_plan(3, 4)
        .expect("default streamed plan")
        .expect("streamed plan");
        assert_eq!(defaults.max_kv_bytes, 7_340_032);
        let over_limit = GenerationMemoryConfig {
            mode: GenerationMemoryMode::Streamed,
            max_weight_bytes: None,
            max_kv_bytes: None,
            tile_rows: None,
        }
        .streamed_plan(3, 30)
        .expect_err("over-limit streamed prompt must fail");
        assert_eq!(
            over_limit,
            "streamed generation requires prompt_tokens + max_tokens <= 32; received 3 + 30 = 33"
        );
        assert!(
            GenerationMemoryConfig {
                mode: GenerationMemoryMode::Resident,
                max_weight_bytes: Some(1),
                max_kv_bytes: None,
                tile_rows: None,
            }
            .streamed_plan(1, 1)
            .is_err()
        );
    }

    #[test]
    fn phase_profiles_are_present_only_for_verbose_streamed_reports() {
        let profiles = vec![serde_json::json!({
            "phase": "prefill",
            "generation_step": 0,
            "cached_tokens": 3,
            "profile": { "total_ms": 1.0 },
        })];
        let mut quiet = serde_json::json!({ "max_kv_bytes": 7_340_032 });
        super::attach_verbose_phase_profiles(&mut quiet, false, &profiles);
        assert!(quiet.get("phase_profiles").is_none());

        let mut verbose = serde_json::json!({ "max_kv_bytes": 7_340_032 });
        super::attach_verbose_phase_profiles(&mut verbose, true, &profiles);
        assert_eq!(verbose["phase_profiles"], serde_json::json!(profiles));
    }
}
