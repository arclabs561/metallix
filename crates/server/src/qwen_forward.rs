//! CLI driver for an uncached numerical-forward measurement.

use std::{fs, path::Path, process::ExitCode, time::Instant};

#[cfg(feature = "structured-output")]
use crate::qwen_constraints::{ConstraintRun, SchemaSource};
use engine::sampling::sample_categorical;
use qwen::metal::Qwen3MlxWeights;
use rand_chacha::ChaCha8Rng;
use rand_core::{RngCore, SeedableRng};
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

pub(crate) struct GenerationDiagnostics<'a> {
    pub(crate) verbose: bool,
    pub(crate) logprobs: bool,
    pub(crate) preview: bool,
    pub(crate) sampling: Option<SamplingConfiguration>,
    /// Optional decoder for text in the output receipt; execution still uses IDs.
    pub(crate) tokenizer: Option<&'a crate::qwen_tokenizer::QwenTokenizer>,
    /// Input preparation recorded in every ordinary and candidate receipt.
    pub(crate) input_format: GenerationInputFormat,
}

/// The exact prompt preparation performed before Qwen payload loading.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct GenerationInputFormat {
    pub(crate) kind: &'static str,
    pub(crate) chat_template_sha256: Option<String>,
}

impl GenerationInputFormat {
    fn report(&self) -> serde_json::Value {
        json!({
            "kind": self.kind,
            "chat_template_sha256": self.chat_template_sha256.as_deref(),
        })
    }
}

/// Bounded whole-candidate schedule verification. This is intentionally a
/// fixed local contract, not a general verifier/plugin execution surface.
#[cfg(feature = "structured-output")]
#[derive(Debug)]
pub(crate) struct ScheduleVerificationConfig {
    pub(crate) max_attempts: u32,
    pub(crate) max_elapsed_ms: u64,
    pub(crate) requirements: Option<crate::schedule_requirements::ScheduleRequirements>,
}

/// The explicit request settings for a reproducible categorical policy.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct SamplingConfiguration {
    pub(crate) seed: u64,
    pub(crate) temperature: f64,
}

impl SamplingConfiguration {
    fn report(self) -> serde_json::Value {
        json!({
            "algorithm": "rand_chacha::ChaCha8Rng",
            "crate_version": "0.9.0",
            "seed": self.seed,
            "temperature": self.temperature,
            "uniform": "next_u64 top 53 bits divided by 2^53",
            "truncation": "none",
        })
    }
}

pub(crate) struct SamplingPolicy {
    configuration: SamplingConfiguration,
    rng: ChaCha8Rng,
    legal_mask: Vec<bool>,
}

impl SamplingPolicy {
    pub(crate) fn new(configuration: SamplingConfiguration, vocabulary_size: usize) -> Self {
        Self {
            configuration,
            rng: ChaCha8Rng::seed_from_u64(configuration.seed),
            legal_mask: vec![true; vocabulary_size],
        }
    }

    pub(crate) fn sample(
        &mut self,
        logits: &[f32],
        logprobs: bool,
    ) -> Result<(i32, Option<serde_json::Value>), String> {
        // Advance a cloned stream first, committing it only after the sampler
        // accepts the logits. Invalid model output must not make a replay drift.
        let mut candidate_rng = self.rng.clone();
        let uniform = unit_uniform(candidate_rng.next_u64());
        let sample = sample_categorical(
            logits,
            &self.legal_mask,
            self.configuration.temperature,
            uniform,
        )
        .map_err(|error| error.to_string())?;
        let token = i32::try_from(sample.token_id).map_err(|error| error.to_string())?;
        let scores = if logprobs {
            Some(json!({
                "model_logprob": selected_model_logprob(logits, sample.token_id)?,
                "sampling_logprob": sample.sampling_logprob,
            }))
        } else {
            None
        };
        self.rng = candidate_rng;
        Ok((token, scores))
    }

    /// Samples from the `top_k` most likely tokens, then from the smallest
    /// most-likely set whose temperature-transformed mass, renormalized over
    /// those `top_k`, reaches `top_p` (the vLLM and Hugging Face order). A
    /// `top_p` of one and no `top_k` keep every token, matching
    /// [`Self::sample`].
    pub(crate) fn sample_nucleus(
        &mut self,
        logits: &[f32],
        top_p: f64,
        top_k: Option<usize>,
    ) -> Result<i32, String> {
        nucleus_mask(
            logits,
            self.configuration.temperature,
            top_p,
            top_k,
            &mut self.legal_mask,
        )?;
        self.sample(logits, false).map(|(token, _)| token)
    }

    #[cfg(feature = "structured-output")]
    pub(crate) fn sample_constrained(
        &mut self,
        constraint: &mut ConstraintRun,
        logits: &[f32],
        logprobs: bool,
    ) -> Result<(i32, Option<serde_json::Value>), Box<dyn std::error::Error>> {
        // `ConstraintRun` owns grammar-state commit. Keep the corresponding
        // entropy transition private until it reports the same successful draw.
        let checkpoint = constraint.checkpoint();
        let mut candidate_rng = self.rng.clone();
        let uniform = unit_uniform(candidate_rng.next_u64());
        let sampled = constraint.sample_categorical(
            logits,
            self.configuration.temperature,
            uniform,
            logprobs,
        );
        match sampled {
            Ok(sampled) => {
                self.rng = candidate_rng;
                Ok(sampled)
            }
            Err(error) => {
                constraint.restore(checkpoint).map_err(|restore_error| {
                    format!("{error}; rollback failed: {restore_error}")
                })?;
                Err(error)
            }
        }
    }
}

/// Marks the nucleus in `mask`: tokens in descending logit order (index order
/// on ties), limited to the first `top_k`, until their temperature-transformed
/// mass reaches `top_p` of the kept mass.
fn nucleus_mask(
    logits: &[f32],
    temperature: f64,
    top_p: f64,
    top_k: Option<usize>,
    mask: &mut [bool],
) -> Result<(), String> {
    if logits.len() != mask.len() || logits.is_empty() {
        return Err("nucleus mask requires one entry per nonempty logit row".into());
    }
    let valid = top_p > 0.0 && top_p <= 1.0 && temperature > 0.0 && temperature.is_finite();
    if !valid {
        return Err("nucleus sampling requires top_p in (0, 1] and a positive temperature".into());
    }
    let top_k = top_k.filter(|&top_k| top_k > 0 && top_k < logits.len());
    let keep_all = top_p >= 1.0 && top_k.is_none();
    mask.fill(keep_all);
    if keep_all {
        return Ok(());
    }
    if logits.iter().any(|logit| !logit.is_finite()) {
        return Err("nucleus sampling requires finite logits".into());
    }
    let maximum = f64::from(logits.iter().copied().fold(f32::NEG_INFINITY, f32::max));
    let weight = |logit: f32| ((f64::from(logit) - maximum) / temperature).exp();
    let total: f64 = logits.iter().map(|&logit| weight(logit)).sum();
    let target = top_p * total;
    let descending = |left: &usize, right: &usize| {
        logits[*right]
            .total_cmp(&logits[*left])
            .then(left.cmp(right))
    };
    let order = |candidates: &mut Vec<usize>| candidates.sort_unstable_by(descending);
    if let Some(top_k) = top_k {
        let mut head: Vec<usize> = (0..logits.len()).collect();
        head.select_nth_unstable_by(top_k - 1, descending);
        head.truncate(top_k);
        order(&mut head);
        let kept: f64 = head.iter().map(|&index| weight(logits[index])).sum();
        let mut covered = 0.0;
        for index in head {
            mask[index] = true;
            covered += weight(logits[index]);
            if covered >= top_p * kept {
                break;
            }
        }
        return Ok(());
    }
    // Sorting the whole vocabulary every step is the slow path; the head
    // above a tiny relative weight almost always holds the nucleus.
    let mut candidates: Vec<usize> = (0..logits.len())
        .filter(|&index| weight(logits[index]) >= total * 1e-9)
        .collect();
    order(&mut candidates);
    let mut covered = 0.0;
    for &index in &candidates {
        mask[index] = true;
        covered += weight(logits[index]);
        if covered >= target {
            return Ok(());
        }
    }
    let mut all: Vec<usize> = (0..logits.len()).collect();
    order(&mut all);
    mask.fill(false);
    covered = 0.0;
    for index in all {
        mask[index] = true;
        covered += weight(logits[index]);
        if covered >= target {
            break;
        }
    }
    Ok(())
}

fn unit_uniform(word: u64) -> f64 {
    const TWO_TO_21: f64 = 2_097_152.0;
    const TWO_TO_53: f64 = 9_007_199_254_740_992.0;
    let top_53 = word >> 11;
    let high = u32::try_from(top_53 >> 21).expect("top 53 bits split into 32 and 21 bits");
    let low =
        u32::try_from(top_53 & ((1 << 21) - 1)).expect("top 53 bits split into 32 and 21 bits");
    (f64::from(high) * TWO_TO_21 + f64::from(low)) / TWO_TO_53
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

    /// Forks a resident KV snapshot for opt-in branch verification. Streamed
    /// weights do not expose a branchable executor and report `None`.
    fn fork_prefilled(&self) -> Result<Option<Self>, Box<dyn std::error::Error>> {
        match self {
            Self::Resident(executor) => Ok(Some(Self::Resident(executor.fork_prefilled()?))),
            Self::Streamed(_) => Ok(None),
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

#[allow(
    clippy::too_many_arguments,
    reason = "CLI boundary keeps optional schema and candidate policy explicit"
)]
pub(crate) fn generate(
    model: &Path,
    input_ids: &[i32],
    max_tokens: u32,
    verify_cache: bool,
    memory: GenerationMemoryConfig,
    #[cfg(feature = "structured-output")] schedule_verification: Option<
        &ScheduleVerificationConfig,
    >,
    diagnostics: GenerationDiagnostics<'_>,
    #[cfg(feature = "structured-output")] schema_source: Option<SchemaSource<'_>>,
) -> ExitCode {
    match generate_inner(
        model,
        input_ids,
        max_tokens,
        verify_cache,
        memory,
        #[cfg(feature = "structured-output")]
        schedule_verification,
        diagnostics,
        #[cfg(feature = "structured-output")]
        schema_source,
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
    clippy::too_many_arguments,
    reason = "linear diagnostic driver keeps model, verification and optional sampling timing boundaries explicit"
)]
fn generate_inner(
    model: &Path,
    input_ids: &[i32],
    max_tokens: u32,
    verify_cache: bool,
    memory: GenerationMemoryConfig,
    #[cfg(feature = "structured-output")] schedule_verification: Option<
        &ScheduleVerificationConfig,
    >,
    diagnostics: GenerationDiagnostics<'_>,
    #[cfg(feature = "structured-output")] schema_source: Option<SchemaSource<'_>>,
) -> Result<(), Box<dyn std::error::Error>> {
    let GenerationDiagnostics {
        verbose,
        logprobs,
        preview,
        sampling: sampling_configuration,
        tokenizer,
        input_format,
    } = diagnostics;
    let metadata = fs::metadata(model).map_err(|error| {
        format!(
            "--model must name a readable Qwen3 checkpoint directory ({}): {error}",
            model.display()
        )
    })?;
    if !metadata.is_dir() {
        return Err(format!(
            "--model must name a Qwen3 checkpoint directory, not a file: {}",
            model.display()
        )
        .into());
    }
    if input_ids.len().saturating_add(max_tokens as usize) > qwen::forward::MAX_DENSE_DEBUG_TOKENS {
        return Err("prompt plus generation budget exceeds diagnostic context limit".into());
    }
    let streamed = memory.streamed_plan(input_ids.len(), max_tokens)?;
    #[cfg(feature = "structured-output")]
    if schedule_verification.is_some() {
        if verify_cache {
            return Err("--verify-schedule cannot combine with --verify-cache; candidate branch parity is not yet part of this receipt".into());
        }
        if streamed.is_some() {
            return Err("--verify-schedule requires --memory-mode resident because candidate promotion needs resident KV forks".into());
        }
        if schema_source.is_none() {
            return Err("--verify-schedule requires --json-schema or --json-schema-inline".into());
        }
        if sampling_configuration.is_none() {
            return Err("--verify-schedule requires --sample with --temperature and --seed so retry streams are distinct".into());
        }
    }
    if verbose {
        let diagnostic = verbose_preflight(input_ids.len(), max_tokens, verify_cache, streamed);
        eprintln!("{diagnostic}");
    }
    let raw = fs::read_to_string(model.join("config.json"))?;
    let generation: GenerationConfig = serde_json::from_str(&raw)?;
    generation.validate(input_ids, max_tokens)?;
    if let Some(tokenizer) = tokenizer {
        tokenizer.check_model_vocabulary(generation.vocab_size, generation.eos_token_id)?;
    }
    let request_started = Instant::now();
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
    let load_ms = request_started.elapsed().as_secs_f64() * 1000.0;
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
    let mut sampling_policy = sampling_configuration
        .map(|configuration| SamplingPolicy::new(configuration, logits.len()));
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
    #[cfg(feature = "structured-output")]
    if let Some(schedule_config) = schedule_verification {
        return generate_schedule_candidates(
            model,
            &raw,
            input_ids,
            max_tokens,
            &mut executor,
            &logits,
            load_ms,
            prefill_ms,
            request_started,
            schedule_config,
            sampling_configuration.ok_or("schedule verification sampling policy is unavailable")?,
            schema_source.ok_or("schedule verification schema is unavailable")?,
            &input_format,
            verbose,
            logprobs,
            preview,
        );
    }
    #[cfg(feature = "structured-output")]
    let mut constraint = schema_source
        .map(|source| crate::qwen_constraints::ConstraintRun::load(model, source))
        .transpose()?;
    let mut prefix = input_ids.to_vec();
    let mut generated = Vec::new();
    let mut decode_ms = Vec::new();
    let mut comparisons = Vec::new();
    let mut branch_comparisons = Vec::new();
    let mut branch_checked_positions = Vec::new();
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
        let (token, scores) = match (constraint.as_mut(), sampling_policy.as_mut()) {
            (Some(constraint), None) => constraint.sample(&logits, logprobs)?,
            (None, Some(policy)) => policy.sample(&logits, logprobs)?,
            (None, None) => greedy_sample(&logits, logprobs)?,
            (Some(constraint), Some(policy)) => {
                policy.sample_constrained(constraint, &logits, logprobs)?
            }
        };
        #[cfg(not(feature = "structured-output"))]
        let (token, scores) = sampling_policy.as_mut().map_or_else(
            || greedy_sample(&logits, logprobs),
            |policy| policy.sample(&logits, logprobs),
        )?;
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
            let mut branch = if verify_cache {
                executor.fork_prefilled()?
            } else {
                None
            };
            let branch_logits = if let Some(branch_executor) = branch.as_mut() {
                Some(branch_executor.decode_last_logits(token)?)
            } else {
                None
            };
            let started = Instant::now();
            logits = executor.decode_last_logits(token)?;
            decode_ms.push(started.elapsed().as_secs_f64() * 1000.0);
            if let Some(branch_logits) = branch_logits {
                let result = compare_logits(&logits, &branch_logits)?;
                if !result.passed() {
                    return Err(format!(
                        "parent/branch logits differ after generation step {step}: {result:?}"
                    )
                    .into());
                }
                branch_comparisons.push(result);
                branch_checked_positions.push(prefix.len() - 1);
            }
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
    let sampled = sampling_configuration.is_some();
    #[cfg(feature = "structured-output")]
    let sampled_constrained = constraint.is_some();
    #[cfg(not(feature = "structured-output"))]
    let sampled_constrained = false;
    let mut report = json!({
    "schema_version": 1,
    "operation": if sampled { "qwen3_sampled_cached_generation" } else { "qwen3_greedy_cached_generation" },
    "backend": "mlx-rs 0.25.3 Metal float32",
    "input_format": input_format.report(),
    "input_ids": input_ids,
    "generated_ids": generated,
    "finish_reason": finish_reason,
    "load_ms": load_ms,
    "prefill_ms": prefill_ms,
    "decode_ms": decode_ms,
    "cached_tokens": executor.cached_tokens(),
    "logical_kv_bytes": executor.kv_bytes(),
    "cache_comparisons": comparisons,
    "branch_comparison_count": branch_comparisons.len(),
    "branch_checked_token_positions": branch_checked_positions,
    "branch_comparisons": branch_comparisons,
    "branch_verification": if !verify_cache {
        "not_requested"
    } else if streamed.is_some() {
        "not_available_streamed"
    } else if branch_comparisons.is_empty() {
        "not_run_no_eligible_decode"
    } else {
        "resident_fork_parent_match"
    },
    "scope": "single sequence; contiguous KV; first prefill not warmed; verification excluded from timed regions but may warm execution"
    });
    if let Some(tokenizer) = tokenizer {
        report["generated_text"] = json!(tokenizer.decode_generated(&generated)?);
    }
    if let Some(plan) = streamed {
        report["operation"] = json!(if sampled {
            "qwen3_sampled_streamed_generation"
        } else {
            "qwen3_greedy_streamed_generation"
        });
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
    if let Some(configuration) = sampling_configuration {
        report["sampling_policy"] = configuration.report();
    }
    if logprobs {
        report["logprobs"] = json!({
            "log_base": "e",
            "tokens": token_scores,
            "scope": if sampled {
                if sampled_constrained {
                    "model_logprob is raw temperature-one model p; constrained_logprob conditions p on the grammar mask; allowed_log_mass is grammar-allowed p mass; sampling_logprob is deployed temperature-conditioned grammar-masked q with no truncation; none is a complete-sequence probability"
                } else {
                    "model_logprob is raw temperature-one model p; sampling_logprob is deployed temperature-conditioned q with no truncation; neither is a complete-sequence probability"
                }
            } else {
                "selected-token probabilities under temperature-one model logits; constrained scores renormalize the current allowed set; not the probability of a complete valid sequence or of the deterministic greedy policy"
            }
        });
    }
    #[cfg(feature = "structured-output")]
    let report = {
        let mut report = report;
        if let Some(constraint) = constraint.as_ref() {
            report["operation"] = json!(if sampled && streamed.is_some() {
                "qwen3_sampled_constrained_streamed_generation"
            } else if sampled {
                "qwen3_sampled_constrained_cached_generation"
            } else if streamed.is_some() {
                "qwen3_constrained_streamed_generation"
            } else {
                "qwen3_constrained_cached_generation"
            });
            report["constraint"] = constraint.report(verbose, sampled)?;
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

#[cfg(feature = "structured-output")]
#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
fn generate_schedule_candidates(
    model: &Path,
    model_config_json: &str,
    input_ids: &[i32],
    max_tokens: u32,
    parent: &mut GenerationExecutor<'_>,
    prompt_logits: &[f32],
    load_ms: f64,
    prefill_ms: f64,
    request_started: Instant,
    config: &ScheduleVerificationConfig,
    sampling: SamplingConfiguration,
    schema_source: SchemaSource<'_>,
    input_format: &GenerationInputFormat,
    verbose: bool,
    logprobs: bool,
    preview: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    use sha2::Digest;
    let mut attempts = Vec::new();
    let mut constraint_identity = serde_json::Value::Null;
    let mut constraint_owner = None;
    let mut constraint_setup_ms = 0.0;
    let mut total_generated_tokens = 0_u64;
    let mut total_decode_ms = 0.0;
    let mut total_fork_ms = 0.0;
    let mut total_verifier_ms = 0.0;
    let max_total_generated_tokens = u64::from(config.max_attempts) * u64::from(max_tokens);
    let mut accepted = None;
    let mut exhaustion_reason = "attempt_limit";

    for index in 0..config.max_attempts {
        if request_started.elapsed() >= std::time::Duration::from_millis(config.max_elapsed_ms) {
            exhaustion_reason = "elapsed_limit";
            break;
        }
        let attempt_seed = schedule_attempt_seed(sampling.seed, index);
        let fork_started = Instant::now();
        let mut child = parent
            .fork_prefilled()?
            .ok_or("schedule verification requires a resident forkable executor")?;
        let fork_ms = elapsed_ms(fork_started);
        total_fork_ms += fork_ms;
        if constraint_owner.is_none() {
            let setup_started = Instant::now();
            let constraint = ConstraintRun::load(model, schema_source)?;
            constraint_identity = constraint.identity();
            constraint_owner = Some(constraint);
            constraint_setup_ms = elapsed_ms(setup_started);
        }
        let constraint = constraint_owner
            .as_mut()
            .ok_or("candidate constraint state is unavailable")?;
        let checkpoint = constraint.checkpoint();
        let mut policy = SamplingPolicy::new(
            SamplingConfiguration {
                seed: attempt_seed,
                temperature: sampling.temperature,
            },
            prompt_logits.len(),
        );
        let mut logits = prompt_logits.to_vec();
        let mut generated = Vec::new();
        let mut token_scores = Vec::new();
        let mut decode_ms = Vec::new();
        let mut timeout = false;

        for step in 0..max_tokens {
            if request_started.elapsed() >= std::time::Duration::from_millis(config.max_elapsed_ms)
            {
                timeout = true;
                break;
            }
            let (token, scores) = policy.sample_constrained(constraint, &logits, logprobs)?;
            if let Some(mut scores) = scores {
                scores["token_id"] = json!(token);
                token_scores.push(scores);
            }
            generated.push(token);
            if constraint.is_complete() {
                break;
            }
            if step + 1 < max_tokens {
                let decode_started = Instant::now();
                logits = child.decode_last_logits(token)?;
                decode_ms.push(elapsed_ms(decode_started));
            }
        }
        let generated_tokens = u64::try_from(generated.len())?;
        total_generated_tokens += generated_tokens;
        let attempt_decode_ms = decode_ms.iter().sum::<f64>();
        total_decode_ms += attempt_decode_ms;

        if timeout {
            attempts.push(json!({
                "index": index,
                "seed": attempt_seed,
                "status": "incomplete",
                "reason": "elapsed_limit",
                "generated_tokens": generated_tokens,
                "fork_ms": fork_ms,
                "decode_ms": attempt_decode_ms,
            }));
            exhaustion_reason = "elapsed_limit";
            break;
        }
        if !constraint.is_complete() {
            attempts.push(json!({
                "index": index,
                "seed": attempt_seed,
                "status": "incomplete",
                "reason": "generation_budget",
                "generated_tokens": generated_tokens,
                "fork_ms": fork_ms,
                "decode_ms": attempt_decode_ms,
            }));
            constraint.restore(checkpoint)?;
            continue;
        }

        if request_started.elapsed() >= std::time::Duration::from_millis(config.max_elapsed_ms) {
            attempts.push(json!({
                "index": index,
                "seed": attempt_seed,
                "status": "incomplete",
                "reason": "elapsed_limit",
                "generated_tokens": generated_tokens,
                "fork_ms": fork_ms,
                "decode_ms": attempt_decode_ms,
            }));
            exhaustion_reason = "elapsed_limit";
            break;
        }
        let verification_started = Instant::now();
        let output = constraint.validated_output()?;
        let mut semantic = crate::qwen_constraints::verify_non_overlapping_schedule(&output);
        if semantic.accepted() {
            semantic.rejection = config
                .requirements
                .as_ref()
                .and_then(|requirements| requirements.rejection(&output));
        }
        let verifier_ms = elapsed_ms(verification_started);
        total_verifier_ms += verifier_ms;
        if semantic.accepted() {
            if request_started.elapsed() >= std::time::Duration::from_millis(config.max_elapsed_ms)
            {
                attempts.push(json!({
                    "index": index,
                    "seed": attempt_seed,
                    "status": "incomplete",
                    "reason": "elapsed_limit",
                    "generated_tokens": generated_tokens,
                    "fork_ms": fork_ms,
                    "decode_ms": attempt_decode_ms,
                    "verifier_ms": verifier_ms,
                }));
                exhaustion_reason = "elapsed_limit";
                break;
            }
            // Materialize the terminal grammar token before promotion. The
            // resulting resident child therefore contains the entire accepted
            // candidate, even though its next logits are intentionally unused.
            let materialize_started = Instant::now();
            let terminal = *generated
                .last()
                .ok_or("complete grammar candidate has no terminal token")?;
            let _ = child.decode_last_logits(terminal)?;
            let terminal_decode_ms = elapsed_ms(materialize_started);
            decode_ms.push(terminal_decode_ms);
            let attempt_decode_ms = decode_ms.iter().sum::<f64>();
            total_decode_ms += terminal_decode_ms;
            if request_started.elapsed() >= std::time::Duration::from_millis(config.max_elapsed_ms)
            {
                attempts.push(json!({
                    "index": index,
                    "seed": attempt_seed,
                    "status": "incomplete",
                    "reason": "elapsed_limit",
                    "generated_tokens": generated_tokens,
                    "fork_ms": fork_ms,
                    "decode_ms": attempt_decode_ms,
                    "verifier_ms": verifier_ms,
                }));
                exhaustion_reason = "elapsed_limit";
                break;
            }
            attempts.push(json!({
                "index": index,
                "seed": attempt_seed,
                "status": "accepted",
                "generated_tokens": generated_tokens,
                "interval_count": semantic.interval_count,
                "fork_ms": fork_ms,
                "decode_ms": attempt_decode_ms,
                "terminal_token_materialized": true,
                "verifier_ms": verifier_ms,
            }));
            // This is the commit point: every rejected child has been dropped,
            // while the accepted child becomes the request's resident state.
            *parent = child;
            let committed_constraint = constraint_owner
                .take()
                .ok_or("accepted constraint state is unavailable")?;
            accepted = Some((
                index,
                generated,
                token_scores,
                decode_ms,
                committed_constraint,
            ));
            break;
        }
        attempts.push(json!({
            "index": index,
            "seed": attempt_seed,
            "status": "rejected",
            "reason": semantic.rejection,
            "generated_tokens": generated_tokens,
            "interval_count": semantic.interval_count,
            "fork_ms": fork_ms,
            "decode_ms": attempt_decode_ms,
            "verifier_ms": verifier_ms,
        }));
        constraint.restore(checkpoint)?;
    }

    let accepted_status = accepted.is_some();
    let (generated, token_scores, decode_ms, constraint, finish_reason, accepted_attempt) =
        if let Some((index, generated, token_scores, decode_ms, constraint)) = accepted {
            (
                generated,
                token_scores,
                decode_ms,
                Some(constraint),
                "grammar_complete",
                Some(index),
            )
        } else {
            (
                Vec::new(),
                Vec::new(),
                Vec::new(),
                None,
                "verification_exhausted",
                None,
            )
        };
    let mut report = json!({
        "schema_version": 1,
        "operation": "qwen3_verified_schedule_candidate",
        "model_directory": model,
        "config_json_sha256": format!("{:x}", sha2::Sha256::digest(model_config_json.as_bytes())),
        "model_identity_scope": "local model directory and config bytes; checkpoint weights are not fingerprinted",
        "sampling": sampling.report(),
        "constraint_identity": constraint_identity,
        "backend": "mlx-rs 0.25.3 Metal float32",
        "input_format": input_format.report(),
        "input_ids": input_ids,
        "generated_ids": generated,
        "finish_reason": finish_reason,
        "load_ms": load_ms,
        "prefill_ms": prefill_ms,
        "decode_ms": decode_ms,
        "cached_tokens": parent.cached_tokens(),
        "logical_kv_bytes": parent.kv_bytes(),
        "branch_verification": "candidate_children_forked_and_discarded_or_promoted",
        "sampling_policy": {
            "base": sampling.report(),
            "attempt_seed_derivation": "splitmix64(base_seed + (attempt_index + 1) * 0x9E3779B97F4A7C15, wrapping arithmetic)",
        },
        "candidate_verification": {
            "kind": if config.requirements.is_some() { "deterministic_schedule_requirements" } else { "deterministic_schedule_non_overlap" },
            "requirements": config.requirements.as_ref().map(crate::schedule_requirements::ScheduleRequirements::report),
            "status": if accepted_status { "accepted" } else { "exhausted" },
            "accepted_attempt": accepted_attempt,
            "exhaustion_reason": if accepted_status { serde_json::Value::Null } else { json!(exhaustion_reason) },
            "max_attempts": config.max_attempts,
            "max_elapsed_ms": config.max_elapsed_ms,
            "max_total_generated_tokens": max_total_generated_tokens,
            "total_generated_tokens": total_generated_tokens,
            "attempts": attempts,
            "costs_ms": {
                "load": load_ms,
                "prefill": prefill_ms,
                "constraint_setup": constraint_setup_ms,
                "fork_total": total_fork_ms,
                "decode_total": total_decode_ms,
                "verifier_total": total_verifier_ms,
                "request_total": elapsed_ms(request_started),
            },
            "scope": "fixed local verifier for nonempty half-open intervals at output.intervals; adjacency is permitted; supplied requirements additionally enforce exact integer-tick count, durations and window; no requirements are inferred from prompt text; rejected candidate output is omitted",
        },
        "scope": "resident Qwen prompt KV forks; accepted child KV includes its terminal grammar token; complete grammar-constrained candidates are accepted only after deterministic schedule verification; candidate attempts are bounded by count, token budget, and cooperative elapsed time",
    });
    if let Some(constraint) = constraint.as_ref() {
        report["constraint"] = constraint.report(verbose, true)?;
    }
    if logprobs && accepted_status {
        report["logprobs"] = json!({
            "log_base": "e",
            "tokens": token_scores,
            "scope": "selected-token scores for the accepted candidate only; rejected candidate output and token scores are omitted",
        });
    }
    if preview {
        crate::generation_preview::emit(&report);
    }
    println!("{}", serde_json::to_string_pretty(&report)?);
    if accepted_status {
        Ok(())
    } else {
        Err(format!("schedule verification exhausted: {exhaustion_reason}").into())
    }
}

#[cfg(feature = "structured-output")]
fn elapsed_ms(started: Instant) -> f64 {
    started.elapsed().as_secs_f64() * 1000.0
}

/// `SplitMix64` gives each bounded candidate an independently derived reproducible
/// entropy stream without exposing a retry as the same rejected draw.
#[cfg(feature = "structured-output")]
fn schedule_attempt_seed(request_seed: u64, attempt_index: u32) -> u64 {
    let mut state = request_seed.wrapping_add(
        u64::from(attempt_index)
            .wrapping_add(1)
            .wrapping_mul(0x9E37_79B9_7F4A_7C15),
    );
    state = (state ^ (state >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    state = (state ^ (state >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    state ^ (state >> 31)
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
        Some(json!({
            "model_logprob": selected_model_logprob(
                logits,
                u32::try_from(token).map_err(|error| error.to_string())?,
            )?,
        }))
    } else {
        None
    };
    Ok((token, scores))
}

fn selected_model_logprob(logits: &[f32], token_id: u32) -> Result<f64, String> {
    if logits.is_empty() || logits.iter().any(|value| !value.is_finite()) {
        return Err("model log probability requires finite nonempty logits".into());
    }
    let token = usize::try_from(token_id).map_err(|error| error.to_string())?;
    let selected = *logits
        .get(token)
        .ok_or_else(|| String::from("selected token is outside model vocabulary"))?;
    let maximum = logits
        .iter()
        .map(|&value| f64::from(value))
        .fold(f64::NEG_INFINITY, f64::max);
    let shifted_sum = logits
        .iter()
        .map(|&value| (f64::from(value) - maximum).exp())
        .sum::<f64>();
    Ok(f64::from(selected) - maximum - shifted_sum.ln())
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
    use super::{
        GenerationInputFormat, GenerationMemoryConfig, GenerationMemoryMode, SamplingConfiguration,
        SamplingPolicy, selected_model_logprob, unit_uniform, verbose_preflight,
    };
    #[cfg(feature = "structured-output")]
    use engine::constraint::{ConstraintLimits, JsonConstraintSession};
    use rand_chacha::ChaCha8Rng;
    use rand_core::{RngCore, SeedableRng};
    use serde_json::json;

    #[cfg(feature = "structured-output")]
    const CONSTRAINT_TEST_VOCABULARY: usize = 18;

    #[cfg(feature = "structured-output")]
    fn constrained_run(max_output_bytes: usize) -> crate::qwen_constraints::ConstraintRun {
        let tokenizer = json!({
            "decoder": {"type": "ByteLevel"},
            "added_tokens": [{"id": 14, "content": "<eos>", "special": true}],
            "model": {"vocab": {
                "{": 0, "}": 1, "\"": 2, "o": 3, "k": 4, ":": 5,
                "t": 6, "r": 7, "u": 8, "e": 9, "f": 10, "a": 11,
                "l": 12, "s": 13
            }}
        });
        let session = JsonConstraintSession::new(
            &tokenizer,
            14,
            CONSTRAINT_TEST_VOCABULARY,
            json!({"type": "boolean"}),
            ConstraintLimits {
                max_schema_bytes: 4_096,
                max_tokenizer_bytes: 4_096,
                max_output_bytes,
            },
        )
        .expect("bounded categorical constraint session");
        crate::qwen_constraints::ConstraintRun::from_session(session)
    }

    #[cfg(feature = "structured-output")]
    fn forced_constraint_logits(token_id: usize) -> Vec<f32> {
        let mut logits = vec![-100.0; CONSTRAINT_TEST_VOCABULARY];
        logits[token_id] = 1.0;
        logits
    }

    fn independent_uniform(word: u64) -> f64 {
        const TWO_TO_21: f64 = 2_097_152.0;
        const TWO_TO_53: f64 = 9_007_199_254_740_992.0;
        let high = u32::try_from(word >> 32).expect("upper source bits fit u32");
        let low = u32::try_from((word >> 11) & ((1 << 21) - 1))
            .expect("lower retained source bits fit u32");
        (f64::from(high) * TWO_TO_21 + f64::from(low)) / TWO_TO_53
    }

    fn independently_enumerated_draw(logits: &[f32], temperature: f64, word: u64) -> (i32, f64) {
        let weights = logits
            .iter()
            .map(|&logit| (f64::from(logit) / temperature).exp())
            .collect::<Vec<_>>();
        let normalizer = weights.iter().sum::<f64>();
        let uniform = independent_uniform(word);
        let target = uniform * normalizer;
        let mut cumulative = 0.0;
        for (index, weight) in weights.iter().enumerate() {
            cumulative += weight;
            if target < cumulative {
                return (
                    i32::try_from(index).expect("small test vocabulary"),
                    (weight / normalizer).ln(),
                );
            }
        }
        let index = weights
            .iter()
            .rposition(|weight| *weight > 0.0)
            .expect("finite test logits have positive support");
        (
            i32::try_from(index).expect("small test vocabulary"),
            (weights[index] / normalizer).ln(),
        )
    }

    fn assert_close(actual: f64, expected: f64) {
        assert!(
            (actual - expected).abs() < 1e-12,
            "expected {expected}, got {actual}"
        );
    }

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
    fn generation_input_format_receipt_retains_template_identity() {
        let plain = GenerationInputFormat {
            kind: "plain_text_prompt",
            chat_template_sha256: None,
        };
        assert_eq!(plain.report()["kind"], "plain_text_prompt");
        assert!(plain.report()["chat_template_sha256"].is_null());

        let templated = GenerationInputFormat {
            kind: "qwen_chat_template_user_message",
            chat_template_sha256: Some(String::from("aabb")),
        };
        assert_eq!(
            templated.report(),
            json!({
                "kind": "qwen_chat_template_user_message",
                "chat_template_sha256": "aabb",
            })
        );
    }

    #[test]
    fn seeded_policy_replays_and_matches_analytic_raw_and_deployed_logprobs() {
        let configuration = SamplingConfiguration {
            seed: 7,
            temperature: 0.5,
        };
        let logits = [0.0, 1.0];
        let mut first = SamplingPolicy::new(configuration, logits.len());
        let mut replay = SamplingPolicy::new(configuration, logits.len());

        let first_tokens = (0..4)
            .map(|_| first.sample(&logits, true).expect("valid sampled token"))
            .collect::<Vec<_>>();
        let replay_tokens = (0..4)
            .map(|_| replay.sample(&logits, true).expect("valid replay token"))
            .collect::<Vec<_>>();
        assert_eq!(first_tokens, replay_tokens);

        let (_, scores) = &first_tokens[0];
        let scores = scores.as_ref().expect("requested sampled scores");
        let token = u32::try_from(first_tokens[0].0).expect("sampled token ID");
        let raw = scores["model_logprob"].as_f64().expect("raw score");
        let deployed = scores["sampling_logprob"].as_f64().expect("deployed score");
        let selected = f64::from(logits[usize::try_from(token).expect("two token IDs")]);
        let expected_raw = selected - (1.0_f64 + 1.0_f64.exp()).ln();
        let expected_deployed = 2.0 * selected - (1.0_f64 + 2.0_f64.exp()).ln();
        assert!((raw - expected_raw).abs() < 1e-12);
        assert!((deployed - expected_deployed).abs() < 1e-12);
        assert!(
            (raw - selected_model_logprob(&logits, token).expect("raw reference")).abs() < 1e-12
        );
    }

    #[test]
    fn uniform_conversion_is_half_open_and_uses_all_top_53_bits() {
        assert_eq!(unit_uniform(0).to_bits(), 0.0_f64.to_bits());
        assert_eq!(unit_uniform(u64::MAX).to_bits(), 0x3fef_ffff_ffff_ffff);
        assert_eq!(unit_uniform(1 << 11).to_bits(), 0x3ca0_0000_0000_0000);
        assert_eq!(
            unit_uniform((1 << 11) | ((1 << 11) - 1)).to_bits(),
            0x3ca0_0000_0000_0000
        );
    }

    #[test]
    fn uniform_conversion_maps_every_source_bit_and_extreme_exactly() {
        for bit in 0_u32..64 {
            let word = 1_u64 << bit;
            let expected = if bit < 11 {
                0.0
            } else {
                2.0_f64.powi(i32::try_from(bit).expect("bit fits i32") - 64)
            };
            assert_eq!(
                unit_uniform(word).to_bits(),
                expected.to_bits(),
                "bit {bit}"
            );
        }
        assert_eq!(unit_uniform(0).to_bits(), 0.0_f64.to_bits());
        assert_eq!(unit_uniform(u64::MAX).to_bits(), 0x3fef_ffff_ffff_ffff);
    }

    #[cfg(feature = "structured-output")]
    #[test]
    fn schedule_attempt_streams_are_reproducible_and_distinct() {
        let first = (0..4)
            .map(|index| super::schedule_attempt_seed(91, index))
            .collect::<Vec<_>>();
        let replay = (0..4)
            .map(|index| super::schedule_attempt_seed(91, index))
            .collect::<Vec<_>>();
        assert_eq!(first, replay);
        assert_eq!(
            first.len(),
            first
                .iter()
                .copied()
                .collect::<std::collections::BTreeSet<_>>()
                .len()
        );
        assert_ne!(first[0], 91);
        assert_ne!(
            super::schedule_attempt_seed(u64::MAX, 15),
            super::schedule_attempt_seed(u64::MAX, 14)
        );
    }

    #[test]
    fn failed_sampling_does_not_advance_the_seeded_rng() {
        let configuration = SamplingConfiguration {
            seed: 13,
            temperature: 1.0,
        };
        let mut after_failure = SamplingPolicy::new(configuration, 2);
        let mut fresh = SamplingPolicy::new(configuration, 2);
        assert!(after_failure.sample(&[f32::NAN, 0.0], true).is_err());
        assert_eq!(
            after_failure
                .sample(&[0.0, 1.0], true)
                .expect("valid sampled token"),
            fresh
                .sample(&[0.0, 1.0], true)
                .expect("same first sampled token")
        );
    }

    #[test]
    fn requesting_logprobs_does_not_change_the_seeded_token_sequence() {
        let configuration = SamplingConfiguration {
            seed: 29,
            temperature: 1.3,
        };
        let logits = [0.0, 0.5, 1.0];
        let mut with_scores = SamplingPolicy::new(configuration, logits.len());
        let mut without_scores = SamplingPolicy::new(configuration, logits.len());

        for _ in 0..8 {
            let scored = with_scores
                .sample(&logits, true)
                .expect("valid scored draw");
            let unscored = without_scores
                .sample(&logits, false)
                .expect("valid unscored draw");
            assert_eq!(scored.0, unscored.0);
            assert!(scored.1.is_some());
            assert!(unscored.1.is_none());
        }
    }

    #[test]
    fn seeded_policy_matches_independent_stream_and_enumerated_q_with_or_without_scores() {
        // This intentionally reuses ChaCha8Rng: it checks this policy's
        // documented seed consumption and uniform transform, not the RNG
        // algorithm's implementation.
        let configuration = SamplingConfiguration {
            seed: 41,
            temperature: 0.7,
        };
        let logits = [0.0, 1.0, -2.0];
        let mut expected_rng = ChaCha8Rng::seed_from_u64(configuration.seed);
        let mut scored = SamplingPolicy::new(configuration, logits.len());
        let mut unscored = SamplingPolicy::new(configuration, logits.len());

        for _ in 0..8 {
            let expected = independently_enumerated_draw(
                &logits,
                configuration.temperature,
                expected_rng.next_u64(),
            );
            let scored_draw = scored.sample(&logits, true).expect("scored policy draw");
            let unscored_draw = unscored
                .sample(&logits, false)
                .expect("unscored policy draw");
            assert_eq!(scored_draw.0, expected.0);
            assert_eq!(unscored_draw.0, expected.0);
            let score = scored_draw.1.expect("requested score");
            assert_close(
                score["sampling_logprob"].as_f64().expect("deployed q"),
                expected.1,
            );
            assert!(unscored_draw.1.is_none());
        }
    }

    #[test]
    fn rejected_policy_draw_preserves_the_next_documented_entropy_word() {
        let configuration = SamplingConfiguration {
            seed: 97,
            temperature: 1.0,
        };
        let mut policy = SamplingPolicy::new(configuration, 3);
        let mut expected_rng = ChaCha8Rng::seed_from_u64(configuration.seed);
        assert!(policy.sample(&[f32::NAN, 0.0, 1.0], true).is_err());

        let logits = [0.0, 0.0, -20.0];
        let expected_word = expected_rng.next_u64();
        assert_eq!(policy.rng.clone().next_u64(), expected_word);
        let expected =
            independently_enumerated_draw(&logits, configuration.temperature, expected_word);
        let actual = policy
            .sample(&logits, true)
            .expect("first valid policy draw");
        assert_eq!(actual.0, expected.0);
        assert_close(
            actual.1.expect("requested score")["sampling_logprob"]
                .as_f64()
                .expect("deployed q"),
            expected.1,
        );
    }

    #[cfg(feature = "structured-output")]
    #[test]
    fn sampled_constraints_preserve_p_conditionals_and_deployed_q() {
        let configuration = SamplingConfiguration {
            seed: 7,
            temperature: 0.5,
        };
        let mut scored_policy = SamplingPolicy::new(configuration, CONSTRAINT_TEST_VOCABULARY);
        let mut unscored_policy = SamplingPolicy::new(configuration, CONSTRAINT_TEST_VOCABULARY);
        let mut scored_constraint = constrained_run(1_024);
        let mut unscored_constraint = constrained_run(1_024);
        let mut logits = vec![-100.0; CONSTRAINT_TEST_VOCABULARY];
        logits[6] = 0.0;
        logits[10] = 1.0;
        logits[14] = 3.0;
        logits[15] = 4.0;

        let scored = scored_policy
            .sample_constrained(&mut scored_constraint, &logits, true)
            .expect("sampled constrained receipt");
        let unscored = unscored_policy
            .sample_constrained(&mut unscored_constraint, &logits, false)
            .expect("same sampled constrained draw without scores");
        assert_eq!(scored.0, unscored.0);
        assert!(unscored.1.is_none());

        let receipt = scored.1.expect("requested constrained scores");
        let selected = f64::from(logits[usize::try_from(scored.0).expect("token ID")]);
        let allowed_log_sum = (1.0_f64 + (-1.0_f64).exp()).ln();
        let model_log_sum = (1.0_f64
            + (-1.0_f64).exp()
            + (-3.0_f64).exp()
            + (-4.0_f64).exp()
            + 14.0 * (-104.0_f64).exp())
        .ln();
        let model_logprob = receipt["model_logprob"].as_f64().expect("raw p");
        let constrained_logprob = receipt["constrained_logprob"]
            .as_f64()
            .expect("grammar conditional p");
        let allowed_log_mass = receipt["allowed_log_mass"]
            .as_f64()
            .expect("allowed p mass");
        let sampling_logprob = receipt["sampling_logprob"].as_f64().expect("deployed q");
        assert!((model_logprob - (selected - 4.0 - model_log_sum)).abs() < 1e-12);
        assert!((constrained_logprob - (selected - 1.0 - allowed_log_sum)).abs() < 1e-12);
        assert!((allowed_log_mass - (-3.0 + allowed_log_sum - model_log_sum)).abs() < 1e-12);
        assert!(
            (sampling_logprob - (2.0 * selected - (1.0_f64 + 2.0_f64.exp()).ln())).abs() < 1e-12
        );
    }

    #[cfg(feature = "structured-output")]
    #[test]
    fn reused_candidate_constraint_restores_partial_and_terminal_attempts() {
        let mut constraint = constrained_run(1_024);
        for (tokens, complete, expected) in [
            (vec![10, 11], false, "fa"),
            (vec![10, 11, 12, 13, 9, 14], true, "false"),
        ] {
            let checkpoint = constraint.checkpoint();
            for token in tokens {
                constraint
                    .sample(&forced_constraint_logits(token), false)
                    .unwrap();
                if constraint.is_complete() {
                    break;
                }
            }
            assert_eq!(constraint.is_complete(), complete);
            assert_eq!(constraint.decoded_bytes(), expected.as_bytes());
            constraint.restore(checkpoint).unwrap();
            assert!(!constraint.is_complete());
            assert!(constraint.decoded_bytes().is_empty());
            assert_eq!(
                constraint.report(false, false).unwrap()["sampling_ms"],
                json!([])
            );
        }
        for token in [6, 7, 8, 9, 14] {
            constraint
                .sample(&forced_constraint_logits(token), false)
                .unwrap();
            if constraint.is_complete() {
                break;
            }
        }
        assert_eq!(constraint.validated_output().unwrap(), json!(true));
    }

    #[cfg(feature = "structured-output")]
    #[test]
    fn failed_sampled_constraint_preserves_rng_and_grammar_state() {
        let configuration = SamplingConfiguration {
            seed: 13,
            temperature: 1.0,
        };
        let mut policy = SamplingPolicy::new(configuration, CONSTRAINT_TEST_VOCABULARY);
        let mut constraint = constrained_run(3);
        for (token_id, expected) in [(6, "t"), (7, "tr"), (8, "tru")] {
            let token = policy
                .sample_constrained(&mut constraint, &forced_constraint_logits(token_id), false)
                .expect("bounded categorical prefix");
            assert_eq!(token.0, i32::try_from(token_id).expect("small test token"));
            assert_eq!(constraint.decoded_bytes(), expected.as_bytes());
        }
        let mut expected_rng = policy.rng.clone();
        assert!(
            policy
                .sample_constrained(&mut constraint, &forced_constraint_logits(9), false)
                .is_err()
        );
        assert_eq!(constraint.decoded_bytes(), b"tru");
        assert!(!constraint.is_complete());
        assert_eq!(policy.rng.next_u64(), expected_rng.next_u64());
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

    #[test]
    fn nucleus_keeps_the_smallest_head_reaching_top_p() {
        // Softmax of [3, 2, 1, 0] is about [0.644, 0.237, 0.087, 0.032].
        let logits = [1.0_f32, 3.0, 0.0, 2.0];
        let kept_k = |top_p: f64, temperature: f64, top_k: Option<usize>| {
            let mut mask = [false; 4];
            super::nucleus_mask(&logits, temperature, top_p, top_k, &mut mask).unwrap();
            mask
        };
        let kept = |top_p: f64, temperature: f64| kept_k(top_p, temperature, None);
        assert_eq!(kept(0.5, 1.0), [false, true, false, false]);
        assert_eq!(kept(0.8, 1.0), [false, true, false, true]);
        assert_eq!(kept(0.95, 1.0), [true, true, false, true]);
        assert_eq!(kept(1.0, 1.0), [true; 4]);
        // A hotter distribution needs more tokens for the same mass.
        assert_eq!(kept(0.7, 10.0), [true, true, false, true]);
        // top_k cuts first; top_p then applies to the mass the cut kept:
        // the top three hold about 0.968, and 0.7 of that needs two tokens.
        assert_eq!(kept_k(1.0, 1.0, Some(2)), [false, true, false, true]);
        assert_eq!(kept_k(0.7, 1.0, Some(3)), [false, true, false, true]);
        assert_eq!(kept_k(0.5, 1.0, Some(3)), [false, true, false, false]);
        assert_eq!(kept_k(1.0, 1.0, Some(1)), [false, true, false, false]);
        // A top_k of zero or of the whole vocabulary keeps everything.
        assert_eq!(kept_k(1.0, 1.0, Some(0)), [true; 4]);
        assert_eq!(kept_k(1.0, 1.0, Some(4)), [true; 4]);
        let mut mask = [false; 4];
        assert!(super::nucleus_mask(&logits, 1.0, 0.0, None, &mut mask).is_err());
    }

    #[test]
    fn seeded_nucleus_sampling_replays_and_stays_in_the_nucleus() {
        let configuration = super::SamplingConfiguration {
            seed: 11,
            temperature: 0.8,
        };
        let logits: Vec<f32> = (0_u16..64)
            .map(|index| f32::from(index * 37 % 17) / 4.0)
            .collect();
        let draw = |seed| {
            let mut policy = super::SamplingPolicy::new(
                super::SamplingConfiguration {
                    seed,
                    ..configuration
                },
                logits.len(),
            );
            (0..200)
                .map(|_| policy.sample_nucleus(&logits, 0.6, None).unwrap())
                .collect::<Vec<_>>()
        };
        let first = draw(11);
        assert_eq!(first, draw(11));
        assert_ne!(first, draw(12));
        let mut nucleus = vec![false; logits.len()];
        super::nucleus_mask(&logits, 0.8, 0.6, None, &mut nucleus).unwrap();
        assert!(
            first
                .iter()
                .all(|&token| nucleus[usize::try_from(token).unwrap()])
        );
        assert!(nucleus.iter().filter(|&&kept| kept).count() > 1);
    }
}
