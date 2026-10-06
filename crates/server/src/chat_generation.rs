//! Resident Qwen chat generation over the checkpoint's own Jinja template.
//!
//! This is deliberately a single-session, single-sequence core.  It retains
//! model weights between calls but creates fresh KV state for each rendered
//! conversation. Resident chat has separate context and logical-KV admission;
//! the diagnostic forward cap remains unchanged.

use std::{
    fmt, fs,
    path::{Path, PathBuf},
    time::{Duration, Instant},
};

use engine::speculative::{SpeculationRequest, SpeculationStats};
use qwen::{
    forward::{Qwen3PickRule, Qwen3RowCandidates, Qwen3Selection, Qwen3TokenPicks},
    metal::{Qwen3MlxWeights, Qwen3WeightPrecision},
};
use serde::Serialize;
use serde_json::Value;
use sha2::{Digest, Sha256};
use tracing::field::Empty;

use chat_format::{
    AssistantTurn, ChatFormat, Conversation, MAX_GENERATION_CONFIG_BYTES, Prompt, TurnDelta,
};
pub(crate) use chat_format::{ChatMessage, ChatRole, ChatToolCall, ChatToolResult};

use crate::qwen_forward::{SamplingConfiguration, SamplingPolicy};

#[path = "chat_decoder.rs"]
pub(crate) mod decoder;
#[path = "gemma_chat.rs"]
mod gemma;
#[path = "qwen_prefix_cache.rs"]
mod prefix_cache;
#[path = "qwen35_chat.rs"]
mod qwen35_decoder;
#[path = "qwen_speculation.rs"]
mod speculation;
#[path = "chat_turn.rs"]
mod turn;

pub(crate) use decoder::{ChatDecoderSession, FullRowDecoder};
pub(crate) use gemma::GemmaDecoder;
pub(crate) use qwen35_decoder::Qwen35Decoder;
pub(crate) use turn::{TurnModel, TurnStart, TurnStep};

#[cfg(test)]
#[path = "chat_decode_checkpoint_tests.rs"]
mod decode_checkpoint;

#[cfg(test)]
#[path = "chat_speculation_checkpoint_tests.rs"]
mod speculation_checkpoint;

#[cfg(test)]
#[path = "chat_memory_checkpoint_tests.rs"]
mod memory_checkpoint;

const MAX_CHAT_INPUT_BYTES: usize = 1024 * 1024;
const MAX_CHAT_MESSAGES: usize = 256;
const MAX_CHAT_TOOLS: usize = 64;
const MIB_BYTES: u64 = 1024 * 1024;

/// Resident weight and K/V precision. Decode reads every weight once per
/// token, so BF16 halves those bytes against float32, which stays the qwen
/// crate's reference precision. Admission plans K/V at this same precision.
const SERVING_PRECISION: Qwen3WeightPrecision = Qwen3WeightPrecision::BFloat16;

/// Default prompt-prefix cache budget: room for several multi-thousand-token
/// agent preambles of a small Qwen3 at BF16 K/V.
pub(crate) const DEFAULT_PREFIX_CACHE_MIB: u32 = 2048;

/// One resident-chat admission contract shared by session loading and turns.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct ResidentChatLimits {
    context_tokens: usize,
    kv_budget_bytes: u64,
    prefix_cache_bytes: u64,
}

impl ResidentChatLimits {
    #[must_use]
    pub(crate) const fn from_mib(context_tokens: usize, kv_budget_mib: u32) -> Self {
        Self {
            context_tokens,
            kv_budget_bytes: (kv_budget_mib as u64) * MIB_BYTES,
            prefix_cache_bytes: (DEFAULT_PREFIX_CACHE_MIB as u64) * MIB_BYTES,
        }
    }

    /// Sets the prompt-prefix cache budget, held apart from the resident K/V
    /// budget; zero disables reuse across turns.
    #[must_use]
    pub(crate) const fn with_prefix_cache_mib(self, prefix_cache_mib: u32) -> Self {
        Self {
            prefix_cache_bytes: (prefix_cache_mib as u64) * MIB_BYTES,
            ..self
        }
    }

    #[must_use]
    pub(crate) const fn prefix_cache_bytes(self) -> u64 {
        self.prefix_cache_bytes
    }

    #[must_use]
    pub(crate) const fn context_tokens(self) -> usize {
        self.context_tokens
    }

    #[must_use]
    pub(crate) const fn kv_budget_bytes(self) -> u64 {
        self.kv_budget_bytes
    }
}

/// A bounded one-user-message prompt rendered from the checkpoint's own template.
///
/// This only prepares generation input. It neither loads checkpoint weights
/// nor creates a resident chat session.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct GenerationTemplatePrompt {
    pub(crate) rendered: String,
    pub(crate) template_sha256: String,
}

/// Renders one non-thinking user message using the local checkpoint template.
///
/// It applies the resident-chat source, input, rendered-byte, and fuel limits
/// before a caller can begin a checkpoint payload load.
pub(crate) fn render_generation_prompt(
    model: &Path,
    prompt: &str,
) -> Result<GenerationTemplatePrompt, String> {
    let messages = [ChatMessage::text(ChatRole::User, prompt)];
    let request = ChatRequest::new(&messages, 1);
    validate_request(request)?;
    let (config, ..) = read_config(model)?;
    let format = ChatFormat::load(model, config.vocab_size)?;
    Ok(GenerationTemplatePrompt {
        rendered: format.template().render(request.conversation(), true)?,
        template_sha256: format.template().sha256().to_owned(),
    })
}

/// One chat turn. Tool schemas remain JSON because their externally defined
/// schema is a caller boundary, while message history stays typed.
#[derive(Clone, Copy, Debug)]
pub(crate) struct ChatRequest<'a> {
    pub(crate) messages: &'a [ChatMessage],
    pub(crate) tools: &'a [Value],
    /// `None` allows output up to the remaining context after the prompt.
    pub(crate) max_tokens: Option<u32>,
    pub(crate) enable_thinking: bool,
    /// Optional model-specific reasoning effort passed through to templates.
    pub(crate) reasoning_effort: Option<&'a str>,
    pub(crate) sampling: SamplingRequest,
    /// `Some(k)` records each output token's log probability and its `k`
    /// most likely alternatives.
    pub(crate) top_logprobs: Option<u8>,
    /// A JSON Schema the output must satisfy, enforced by a grammar mask.
    pub(crate) json_schema: Option<&'a Value>,
    /// Prefix-cache namespace, set only from the router-owned
    /// `x-metallix-cache-salt` header and never from a request body. Requests
    /// with different salts never reuse each other's cached K/V; `None`
    /// shares the default namespace.
    pub(crate) cache_salt: Option<&'a str>,
    /// Treats end-of-turn as an ordinary token, so the turn runs to
    /// `max_tokens` (vLLM's `ignore_eos`, for equal-length benchmarks).
    pub(crate) ignore_eos: bool,
    /// Whether decode may verify prompt-lookup drafts several tokens at a
    /// time. Greedy output is the plain greedy output up to floating-point
    /// near-ties; sampled output keeps the sampling distribution.
    pub(crate) speculation: SpeculationRequest,
}

impl<'a> ChatRequest<'a> {
    /// Creates a normal greedy non-thinking turn with no tools.
    #[must_use]
    pub(crate) const fn new(messages: &'a [ChatMessage], max_tokens: u32) -> Self {
        Self {
            messages,
            tools: &[],
            max_tokens: Some(max_tokens),
            enable_thinking: false,
            reasoning_effort: None,
            sampling: SamplingRequest::GREEDY,
            top_logprobs: None,
            json_schema: None,
            cache_salt: None,
            ignore_eos: false,
            speculation: SpeculationRequest::Automatic,
        }
    }

    /// The parts of this turn the chat template renders.
    #[must_use]
    pub(crate) const fn conversation(&self) -> Conversation<'a> {
        Conversation {
            messages: self.messages,
            tools: self.tools,
            enable_thinking: self.enable_thinking,
            reasoning_effort: self.reasoning_effort,
        }
    }
}

/// The caller's sampling fields. Omitted values take the checkpoint's
/// `generation_config.json` defaults, as vLLM does; a temperature of zero,
/// requested or defaulted, is greedy.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct SamplingRequest {
    pub(crate) temperature: Option<f64>,
    pub(crate) top_p: Option<f64>,
    /// Only the Messages protocol has a `top_k` field.
    pub(crate) top_k: Option<u32>,
    pub(crate) seed: Option<u64>,
}

/// Sampling defaults a checkpoint ships in `generation_config.json`.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct SamplingDefaults {
    pub(crate) temperature: Option<f64>,
    pub(crate) top_p: Option<f64>,
    pub(crate) top_k: Option<u32>,
}

impl SamplingDefaults {
    /// Reads the optional file, keeping only values generation can honor:
    /// a finite nonnegative temperature, `top_p` in (0, 1], a positive
    /// `top_k`. Other fields (`do_sample`, penalties) are not applied.
    fn load(model: &Path) -> Result<Self, String> {
        let path = model.join("generation_config.json");
        if !path.exists() {
            return Ok(Self::default());
        }
        let bytes = chat_format::read_regular_file(
            &path,
            MAX_GENERATION_CONFIG_BYTES,
            "generation_config.json",
        )?;
        let config: Value = serde_json::from_slice(&bytes)
            .map_err(|_| String::from("local generation_config.json could not be parsed"))?;
        Ok(Self {
            temperature: config["temperature"]
                .as_f64()
                .filter(|value| value.is_finite() && *value >= 0.0),
            top_p: config["top_p"]
                .as_f64()
                .filter(|value| *value > 0.0 && *value <= 1.0),
            top_k: config["top_k"]
                .as_u64()
                .and_then(|value| u32::try_from(value).ok())
                .filter(|value| *value > 0),
        })
    }
}

/// The sampling policy one turn actually used.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub(crate) struct AppliedSampling {
    /// Zero means greedy; `top_p`, `top_k` and `seed` then do not apply.
    pub(crate) temperature: f64,
    pub(crate) top_p: f64,
    pub(crate) top_k: Option<u32>,
    /// The seed drawn or requested, so an unseeded turn can be replayed.
    pub(crate) seed: Option<u64>,
    /// Fields taken from the checkpoint's `generation_config.json`.
    pub(crate) defaults_applied: Vec<&'static str>,
}

impl SamplingRequest {
    /// Explicitly greedy, for callers that predate model defaults.
    pub(crate) const GREEDY: Self = Self {
        temperature: Some(0.0),
        top_p: None,
        top_k: None,
        seed: None,
    };

    /// Fills omitted fields from `defaults`. Under a grammar mask the default
    /// `top_p` and `top_k` are not applied: the grammar owns the legal-token
    /// mask, so only temperature carries over.
    pub(crate) fn resolve(self, defaults: SamplingDefaults, constrained: bool) -> AppliedSampling {
        let mut defaults_applied = Vec::new();
        let temperature = self.temperature.unwrap_or_else(|| {
            defaults.temperature.map_or(0.0, |value| {
                defaults_applied.push("temperature");
                value
            })
        });
        if temperature <= 0.0 {
            return AppliedSampling {
                temperature: 0.0,
                top_p: 1.0,
                top_k: None,
                seed: None,
                defaults_applied,
            };
        }
        let top_p = match (self.top_p, defaults.top_p) {
            (Some(value), _) => value,
            (None, Some(value)) if !constrained => {
                defaults_applied.push("top_p");
                value
            }
            (None, _) => 1.0,
        };
        let top_k = match (self.top_k, defaults.top_k) {
            (Some(value), _) => Some(value),
            (None, Some(value)) if !constrained => {
                defaults_applied.push("top_k");
                Some(value)
            }
            (None, _) => None,
        };
        AppliedSampling {
            temperature,
            top_p,
            top_k,
            seed: Some(self.seed.unwrap_or_else(fresh_seed)),
            defaults_applied,
        }
    }
}

pub(crate) const MAX_TOP_LOGPROBS: u8 = 20;

/// Protocol-neutral generation controls. Each HTTP adapter parses its own
/// field names into this, validates it, and borrows it as a [`ChatRequest`].
#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct GenerationControls {
    /// `None` allows output up to the remaining context after the prompt.
    pub(crate) max_tokens: Option<u32>,
    pub(crate) sampling: SamplingRequest,
    pub(crate) top_logprobs: Option<u8>,
    pub(crate) enable_thinking: bool,
    pub(crate) reasoning_effort: Option<String>,
    pub(crate) json_schema: Option<Value>,
    /// See [`ChatRequest::ignore_eos`].
    pub(crate) ignore_eos: bool,
    /// See [`ChatRequest::speculation`].
    pub(crate) speculation: SpeculationRequest,
}

/// The `speculation` request field, a metallix extension shared by every
/// HTTP adapter. Unknown strings fail deserialization with the allowed values.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub(crate) enum SpeculationField {
    /// Speculate while this is the only request the model is decoding.
    #[default]
    Auto,
    /// Speculate even when other requests share the model.
    On,
    /// Never speculate.
    Off,
}

impl From<SpeculationField> for SpeculationRequest {
    fn from(field: SpeculationField) -> Self {
        match field {
            SpeculationField::Auto => Self::Automatic,
            SpeculationField::On => Self::Enabled,
            SpeculationField::Off => Self::Disabled,
        }
    }
}

impl GenerationControls {
    /// Rejects values and combinations generation cannot honor exactly.
    ///
    /// The grammar mask cannot also admit a tool envelope or a thinking block,
    /// and it owns the legal-token mask that nucleus truncation would need.
    /// Log probabilities are reported only for plain text turns, where every
    /// generated token belongs to the visible answer.
    pub(crate) fn validate(&self, has_tools: bool) -> Result<(), String> {
        if self.max_tokens == Some(0) {
            return Err("the output token limit must be positive".into());
        }
        if let Some(temperature) = self.sampling.temperature {
            if !(0.0..=2.0).contains(&temperature) {
                return Err("temperature must be in [0, 2]".into());
            }
        }
        if let Some(top_p) = self.sampling.top_p {
            if !(top_p > 0.0 && top_p <= 1.0) {
                return Err("top_p must be in (0, 1]".into());
            }
        }
        if self.sampling.top_k == Some(0) {
            return Err("top_k must be positive".into());
        }
        if let Some(top) = self.top_logprobs {
            if top > MAX_TOP_LOGPROBS {
                return Err(format!("top_logprobs must be 0..={MAX_TOP_LOGPROBS}"));
            }
            if has_tools || self.enable_thinking {
                return Err("logprobs cannot be combined with tools or reasoning".into());
            }
        }
        if self.json_schema.is_some() {
            if !cfg!(feature = "structured-output") {
                return Err(
                    "JSON schema output requires a server built with the structured-output feature"
                        .into(),
                );
            }
            if has_tools || self.enable_thinking {
                return Err("JSON schema output cannot be combined with tools or reasoning".into());
            }
            // A completed grammar admits only end-of-turn; there is nothing
            // valid to generate past it.
            if self.ignore_eos {
                return Err("JSON schema output cannot be combined with ignore_eos".into());
            }
            // Drafts would have to pass the grammar mask before verification;
            // automatic speculation simply skips schema turns.
            if self.speculation == SpeculationRequest::Enabled {
                return Err("JSON schema output cannot be combined with speculation \"on\"".into());
            }
            let greedy = self.sampling.temperature == Some(0.0);
            if !greedy && self.sampling.top_p.is_some_and(|top_p| top_p < 1.0) {
                return Err("JSON schema output cannot be combined with top_p below 1".into());
            }
            if !greedy && self.sampling.top_k.is_some() {
                return Err("JSON schema output cannot be combined with top_k".into());
            }
        }
        Ok(())
    }

    /// Borrows these controls as one chat turn.
    #[must_use]
    pub(crate) fn request<'a>(
        &'a self,
        messages: &'a [ChatMessage],
        tools: &'a [Value],
    ) -> ChatRequest<'a> {
        ChatRequest {
            messages,
            tools,
            max_tokens: self.max_tokens,
            enable_thinking: self.enable_thinking,
            reasoning_effort: self.reasoning_effort.as_deref(),
            sampling: self.sampling,
            top_logprobs: self.top_logprobs,
            json_schema: self.json_schema.as_ref(),
            cache_salt: None,
            ignore_eos: self.ignore_eos,
            speculation: self.speculation,
        }
    }
}

/// One output token's log probability under the raw (temperature-one) model
/// distribution, with its most likely alternatives. Probabilities are not
/// conditioned on sampling truncation or a grammar mask.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub(crate) struct TokenLogprob {
    /// The token's bytes as lossy UTF-8; a piece of a multi-byte character
    /// appears as U+FFFD here and exactly in `bytes`.
    pub(crate) token: String,
    pub(crate) bytes: Vec<u8>,
    pub(crate) logprob: f64,
    pub(crate) top_logprobs: Vec<TopLogprob>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub(crate) struct TopLogprob {
    pub(crate) token: String,
    pub(crate) bytes: Vec<u8>,
    pub(crate) logprob: f64,
}

/// Why a chat turn stopped.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum ChatFinishReason {
    Eos,
    Length,
}

/// Timings for one turn. `session_load_ms` measures the one-time session load;
/// it is repeated here only to make individual receipts self-describing.
#[derive(Clone, Debug, Serialize)]
pub(crate) struct ChatGenerationMetrics {
    pub(crate) context_tokens: usize,
    /// Planned logical KV at the context limit, not allocated or process memory.
    pub(crate) planned_kv_bytes: u64,
    pub(crate) session_load_ms: f64,
    pub(crate) render_ms: f64,
    /// Prefill through full vocabulary-logit readback; host greedy selection
    /// is outside this interval.
    pub(crate) prefill_ms: f64,
    pub(crate) time_to_first_token_ms: Option<f64>,
    /// Cached decode through full vocabulary-logit readback; host greedy
    /// selection is outside each interval.
    pub(crate) decode_ms: Vec<f64>,
    pub(crate) decode_total_ms: f64,
    pub(crate) prompt_tokens: usize,
    /// Leading prompt tokens restored from the prefix cache, not prefilled.
    pub(crate) cached_prompt_tokens: usize,
    /// Leading prompt tokens this turn copied into the prefix cache for later
    /// requests: the longest prefix it stored, or 0.
    pub(crate) cache_write_tokens: usize,
    pub(crate) generated_tokens: usize,
    /// Draft verification counts, present only when a verify step ran.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) speculation: Option<SpeculationReceipt>,
}

/// How speculative decoding went in one turn.
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub(crate) struct SpeculationReceipt {
    pub(crate) verify_steps: usize,
    pub(crate) drafted_tokens: usize,
    pub(crate) accepted_tokens: usize,
    pub(crate) acceptance_rate: f64,
}

impl SpeculationReceipt {
    fn from_stats(stats: SpeculationStats) -> Option<Self> {
        Some(Self {
            verify_steps: stats.verify_steps,
            drafted_tokens: stats.drafted_tokens,
            accepted_tokens: stats.accepted_tokens,
            acceptance_rate: stats.acceptance_rate()?,
        })
    }
}

/// One finished turn, parsed, with its model-level generation details.
#[derive(Clone, Debug, Serialize)]
pub(crate) struct ChatGeneration {
    /// The decoded output, markup included. Private to this module, so
    /// protocols read the turn only as `turn`.
    text: String,
    /// The turn parsed in the checkpoint's dialects, or why its text did
    /// not parse. Calls are not yet checked against their declarations.
    #[serde(skip)]
    pub(crate) turn: Result<AssistantTurn, String>,
    pub(crate) generated_token_ids: Vec<i32>,
    pub(crate) finish_reason: ChatFinishReason,
    pub(crate) metrics: ChatGenerationMetrics,
    /// One entry per generated token except a final EOS, when requested.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub(crate) logprobs: Vec<TokenLogprob>,
    /// The sampling policy the turn used; `None` from backends that do not
    /// report one.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) sampling: Option<AppliedSampling>,
}

impl ChatGeneration {
    /// SHA-256 and byte length of the decoded output, for receipts that
    /// identify a turn without carrying its text.
    #[must_use]
    pub(crate) fn text_digest(&self) -> (String, usize) {
        (
            format!("{:x}", Sha256::digest(self.text.as_bytes())),
            self.text.len(),
        )
    }

    /// A finished turn for scripted backends in tests: `text` written in
    /// [`QWEN3_TURN`], parsed as a session would parse it.
    #[cfg(test)]
    pub(crate) fn scripted(
        text: impl Into<String>,
        enable_thinking: bool,
        finish_reason: ChatFinishReason,
        metrics: ChatGenerationMetrics,
    ) -> Self {
        let text = text.into();
        Self {
            turn: chat_format::parse_turn_unchecked(
                QWEN3_TURN,
                &text,
                &[],
                enable_thinking,
                finish_reason == ChatFinishReason::Eos,
            ),
            text,
            generated_token_ids: Vec::new(),
            finish_reason,
            metrics,
            logprobs: Vec::new(),
            sampling: None,
        }
    }
}

/// Qwen3's dialects, which scripted backends in tests write.
#[cfg(test)]
pub(crate) const QWEN3_TURN: chat_format::TurnFormat = chat_format::TurnFormat {
    tools: chat_format::ToolDialect::JsonInTags,
    reasoning: chat_format::ReasoningDialect::ThinkTags,
};

/// A cooperative wall-clock budget for one complete chat turn.
///
/// This can only stop work at explicit host-side checkpoints. It deliberately
/// cannot interrupt an in-flight Metal evaluation.
#[derive(Clone, Copy, Debug)]
pub(crate) struct GenerationDeadline {
    deadline: Option<Instant>,
}

impl GenerationDeadline {
    #[must_use]
    pub(crate) const fn unlimited() -> Self {
        Self { deadline: None }
    }

    #[must_use]
    pub(crate) fn after(timeout: Duration) -> Self {
        Self {
            // An unrepresentable deadline must fail closed, never become unlimited.
            deadline: Instant::now()
                .checked_add(timeout)
                .or_else(|| Some(Instant::now())),
        }
    }

    fn check(self) -> Result<(), ChatGenerationError> {
        self.check_at(Instant::now())
    }

    fn check_at(self, now: Instant) -> Result<(), ChatGenerationError> {
        if self.expired_at(now) {
            Err(ChatGenerationError::DeadlineExceeded)
        } else {
            Ok(())
        }
    }

    #[must_use]
    fn expired_at(self, now: Instant) -> bool {
        self.deadline.is_some_and(|deadline| now >= deadline)
    }
}

/// A typed generation failure so protocol adapters can retain error semantics.
#[derive(Debug)]
pub(crate) enum ChatGenerationError {
    DeadlineExceeded,
    Message(String),
}

impl ChatGenerationError {
    fn message(message: impl Into<String>) -> Self {
        Self::Message(message.into())
    }
}

impl From<String> for ChatGenerationError {
    fn from(message: String) -> Self {
        Self::Message(message)
    }
}

impl fmt::Display for ChatGenerationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DeadlineExceeded => formatter.write_str("generation time budget exceeded"),
            Self::Message(message) => formatter.write_str(message),
        }
    }
}

/// A resident model and tokenizer for serial Qwen chat turns.
pub(crate) struct ChatSession {
    weights: Qwen3MlxWeights,
    format: ChatFormat,
    vocabulary_size: usize,
    context_limit: usize,
    kv_budget_bytes: u64,
    planned_kv_bytes: u64,
    load_ms: f64,
    config_sha256: String,
    tokenizer_sha256: String,
    /// Checkpoint directory, read again only to compile a JSON-schema grammar.
    model: PathBuf,
    sampling_defaults: SamplingDefaults,
    /// Prompt-prefix K/V kept across turns, bounded by `--prefix-cache-mib`.
    prefix_cache: prefix_cache::PrefixCache<qwen::forward::Qwen3KvSnapshot>,
}

/// The non-generative result of a single independently-prefilled chat prompt.
///
/// The logits are deliberately exposed only inside the server crate: callers
/// must select a small, validated answer-token set rather than serialize a
/// complete vocabulary distribution.
pub(crate) struct ChatPrefill {
    pub(crate) logits: Vec<f32>,
    pub(crate) prompt_tokens: usize,
    pub(crate) render_ms: f64,
    pub(crate) prefill_ms: f64,
}

/// Backend-neutral generation seam consumed by the Responses/Codex protocol.
///
/// Qwen remains the only implementation today. `DeepSeek` can implement this
/// contract once checkpoint-backed token generation is available, without
/// duplicating transport, streaming, timeout, or tool-call lifecycle logic.
pub(crate) trait ChatBackend {
    /// Returns the one-time backend load duration used in response metadata.
    fn load_ms(&self) -> f64;

    /// Generates one complete turn under the caller's cooperative deadline.
    fn generate_with_timeout(
        &mut self,
        request: ChatRequest<'_>,
        timeout: Duration,
        on_token: &mut dyn FnMut(TurnDelta) -> Result<(), String>,
    ) -> Result<ChatGeneration, ChatGenerationError>;
}

/// The model config fields the chat path reads; stop tokens come from
/// [`ChatFormat`].
#[derive(serde::Deserialize)]
struct ChatConfig {
    vocab_size: usize,
}

impl ChatSession {
    /// Loads and prepares a checkpoint once, including its template and tokenizer.
    pub(crate) fn load(model: &Path, limits: ResidentChatLimits) -> Result<Self, String> {
        let started = Instant::now();
        let (config, plan, config_sha256) = load_config(model, limits)?;
        let format = ChatFormat::load(model, config.vocab_size)?;
        let tokenizer_sha256 = format.tokenizer().source_sha256().to_owned();
        let template_sha256 = format.template().sha256();

        let mut weights = Qwen3MlxWeights::load(model).map_err(|error| error.to_string())?;
        weights
            .prepare_precision(SERVING_PRECISION)
            .map_err(|error| error.to_string())?;
        // The template, tokenizer and config (which carries the RoPE
        // settings) fix how tokens become K/V; the chat path loads no adapter.
        let prefix_identity = format!(
            "model={}\0config={config_sha256}\0tokenizer={tokenizer_sha256}\0template={template_sha256}\0adapter=none",
            model.display()
        );
        let prefix_budget = usize::try_from(limits.prefix_cache_bytes()).unwrap_or(usize::MAX);
        Ok(Self {
            weights,
            format,
            vocabulary_size: config.vocab_size,
            context_limit: limits.context_tokens(),
            kv_budget_bytes: limits.kv_budget_bytes(),
            planned_kv_bytes: plan.planned_kv_bytes(),
            load_ms: elapsed_ms(started.elapsed()),
            config_sha256,
            tokenizer_sha256,
            model: model.to_path_buf(),
            sampling_defaults: SamplingDefaults::load(model)?,
            prefix_cache: prefix_cache::PrefixCache::new(prefix_identity, prefix_budget),
        })
    }

    /// The one-time checkpoint, tokenizer, and template preparation duration.
    #[must_use]
    pub(crate) const fn load_ms(&self) -> f64 {
        self.load_ms
    }

    /// Hash of the exact checkpoint template source selected at session load.
    #[must_use]
    pub(crate) fn template_sha256(&self) -> &str {
        self.format.template().sha256()
    }

    /// Hashes of the exact local metadata parsed during session loading.
    #[must_use]
    pub(crate) fn config_sha256(&self) -> &str {
        &self.config_sha256
    }

    #[must_use]
    pub(crate) fn tokenizer_sha256(&self) -> &str {
        &self.tokenizer_sha256
    }

    /// Encodes a decision answer label as exactly one ordinary tokenizer token.
    pub(crate) fn answer_label_token(&self, label: &str) -> Result<i32, String> {
        let ids = self.format.tokenizer().encode_prompt(label)?;
        match ids.as_slice() {
            [token] => Ok(*token),
            _ => Err(format!(
                "decision answer label {label:?} must encode as exactly one token"
            )),
        }
    }

    /// Prefills a short fixed prompt and decodes a few pipelined greedy steps
    /// on a throwaway executor, so the first request does not pay Metal
    /// pipeline creation and allocator growth for the prefill and decode
    /// kernels. The prefix cache is not touched.
    pub(crate) fn warm(&mut self) -> Result<(), String> {
        const DECODE_STEPS: usize = 4;
        let ids = self
            .format
            .tokenizer()
            .encode_prompt("Warm up the prefill and decode kernels.")?;
        let Some(&last) = ids.last() else {
            return Ok(());
        };
        if ids.len() + DECODE_STEPS + 1 > self.context_limit {
            return Ok(());
        }
        let error = |error: qwen::forward::Qwen3ForwardError| error.to_string();
        let mut executor = self
            .weights
            .resident_chat_executor(self.context_limit, self.kv_budget_bytes)
            .map_err(error)?;
        executor.prefill_last_logits(&ids).map_err(error)?;
        let mut pending = executor.decode_greedy(last).map_err(error)?;
        for _ in 0..DECODE_STEPS {
            let next = executor.decode_greedy_after(&pending).map_err(error)?;
            pending.wait_one().map_err(error)?;
            pending = next;
        }
        pending.wait_one().map_err(error)?;
        Ok(())
    }

    /// Renders and pre-fills one fresh chat prompt without selecting or decoding
    /// any output token. Each call creates a separate resident executor, so no
    /// question can contribute K/V state to another question.
    pub(crate) fn prefill_chat(&mut self, messages: &[ChatMessage]) -> Result<ChatPrefill, String> {
        let request = ChatRequest::new(messages, 1);
        validate_request(request)?;
        let render = tracing::info_span!("chat.render", render_ms = Empty);
        let (prompt, render_ms) = timed(&render, "render_ms", || self.render(request))?;
        let input_ids = prompt.ids;
        if input_ids.len() > self.context_limit {
            return Err(format!(
                "decision requires prompt_tokens <= {}; received {}",
                self.context_limit,
                input_ids.len()
            ));
        }
        let mut executor = self
            .weights
            .resident_chat_executor(self.context_limit, self.kv_budget_bytes)
            .map_err(|error| error.to_string())?;
        // Ends with the full-vocabulary logit readback, so it times the GPU work.
        let prefill = tracing::info_span!(
            "chat.prefill",
            prompt_tokens = input_ids.len(),
            prefill_ms = Empty
        );
        let (logits, prefill_ms) = timed(&prefill, "prefill_ms", || {
            executor
                .prefill_last_logits(&input_ids)
                .map_err(|error| error.to_string())
        })?;
        Ok(ChatPrefill {
            logits,
            prompt_tokens: input_ids.len(),
            render_ms,
            prefill_ms,
        })
    }

    /// Generates one complete chat turn and streams only cumulative-decoder text deltas.
    pub(crate) fn generate(
        &mut self,
        request: ChatRequest<'_>,
        on_token: &mut dyn FnMut(TurnDelta) -> Result<(), String>,
    ) -> Result<ChatGeneration, String> {
        self.generate_with_deadline(request, GenerationDeadline::unlimited(), on_token)
            .map_err(|error| error.to_string())
    }

    /// Generates with a budget that is checked between host-visible phases.
    pub(crate) fn generate_with_timeout(
        &mut self,
        request: ChatRequest<'_>,
        timeout: Duration,
        on_token: &mut dyn FnMut(TurnDelta) -> Result<(), String>,
    ) -> Result<ChatGeneration, ChatGenerationError> {
        // Start before validation and rendering so the caller budgets the full turn.
        self.generate_with_deadline(request, GenerationDeadline::after(timeout), on_token)
    }

    #[allow(
        clippy::too_many_lines,
        reason = "one ordered turn: prefill, decode and the prefix-cache store, each in its own span"
    )]
    fn generate_with_deadline(
        &mut self,
        request: ChatRequest<'_>,
        deadline: GenerationDeadline,
        on_token: &mut dyn FnMut(TurnDelta) -> Result<(), String>,
    ) -> Result<ChatGeneration, ChatGenerationError> {
        let start = TurnStart::prepare(self.turn_model(), request, deadline)?;
        let (render_ms, max_tokens) = (start.render_ms, start.max_tokens);
        let (input_ids, mut turn, mut text) = start.into_parts();

        deadline.check()?;
        // Ends with the full-vocabulary logit readback, so it times the GPU work.
        let prefill = tracing::info_span!(
            "chat.prefill",
            prompt_tokens = input_ids.len(),
            cached_tokens = Empty,
            prefill_ms = Empty
        );
        let ((mut executor, mut logits, cached_prompt_tokens), prefill_ms) =
            timed(&prefill, "prefill_ms", || {
                prefix_cache::prefill(
                    &self.weights,
                    &mut self.prefix_cache,
                    self.context_limit,
                    self.kv_budget_bytes,
                    request,
                    &input_ids,
                )
                .map_err(ChatGenerationError::message)
            })?;
        prefill.record("cached_tokens", cached_prompt_tokens);
        deadline.check()?;
        let mut decode_ms = Vec::new();
        let mut finish_reason = ChatFinishReason::Length;
        // Greedy turns, and sampled turns with top_k, pick each token on the
        // GPU and queue step t + 1 before reading step t back; the host then
        // settles the token and any logprobs exactly from a few top
        // candidates. Other turns read the full logit row each step.
        let pipelined = turn.gpu_rule(&self.format, 0).is_some();
        let mut pending: Option<Qwen3TokenPicks> = None;
        let mut last_token_at = Instant::now();
        // A model worker decodes one request at a time today, so no other
        // sequence competes for the compute a verify uses, and requests
        // waiting in the front queue finish sooner when this one does. Once
        // decode batches requests, pass the batch here.
        let speculating = request.speculation.allows(0, 0) && request.json_schema.is_none();
        let gpu_verify = turn.verifies_with_gpu_greedy(&self.format);
        let lookup = engine::speculative::PromptLookup::default();
        let mut draft_length = speculation::draft_length(SERVING_PRECISION);
        let mut speculation_stats = SpeculationStats::default();

        'decode: for step in 0..max_tokens {
            deadline.check()?;
            let accepted = match pending.take() {
                Some(current) => {
                    let span = tracing::info_span!("chat.decode_step", step, decode_ms = Empty);
                    let (accepted, gpu_token) = span.in_scope(|| {
                        if turn.generated().len() + 1 < max_tokens as usize {
                            // This step's draw is not committed yet, so the
                            // next step's variate is one draw ahead.
                            let rule = turn.gpu_rule(&self.format, 1).ok_or_else(|| {
                                String::from("GPU pick rule changed within a turn")
                            })?;
                            pending = Some(
                                executor
                                    .decode_picks_after(&current, &rule)
                                    .map_err(|error| error.to_string())?,
                            );
                        }
                        turn.settle(&self.format, &current)
                    })?;
                    // Time between token readbacks: with a step always queued,
                    // that is the per-token cost, not one step's latency.
                    let step_ms = elapsed_ms(last_token_at.elapsed());
                    span.record("decode_ms", step_ms);
                    decode_ms.push(step_ms);
                    if accepted.token != gpu_token && pending.take().is_some() {
                        // The f32 GPU draw landed across a cumulative-mass
                        // boundary from the exact draw; the queued step was
                        // built on the wrong token.
                        executor
                            .truncate_cached_tokens(executor.cached_tokens() - 1)
                            .map_err(|error| ChatGenerationError::message(error.to_string()))?;
                    }
                    accepted
                }
                None => turn.pick(&self.format, &mut logits)?,
            };
            last_token_at = Instant::now();
            if accepted.visible {
                text.push(&self.format, accepted.token, on_token)?;
            }
            if let TurnStep::Stop(reason) = accepted.step {
                if !accepted.visible && pending.is_some() {
                    // The queued step appended this stop token.
                    executor
                        .truncate_cached_tokens(executor.cached_tokens() - 1)
                        .map_err(|error| ChatGenerationError::message(error.to_string()))?;
                }
                finish_reason = reason;
                break;
            }

            let mut last_token = accepted.token;
            // Verify drafts back to back while the history keeps proposing
            // them; decode one token when it does not.
            while speculating && turn.generated().len() < max_tokens as usize {
                // Discarding a step already queued ahead costs about one
                // decode step on top of the verify.
                let overhead = if pending.is_some() { 1.0 } else { 0.0 };
                let remaining = max_tokens as usize - turn.generated().len();
                let limit = draft_length
                    .next_with_overhead(overhead)
                    .min(remaining.saturating_sub(1));
                let mut history = input_ids.clone();
                history.extend_from_slice(turn.generated());
                let draft = lookup.propose(&history, limit).to_vec();
                if draft.is_empty() {
                    draft_length.idle();
                    break;
                }
                deadline.check()?;
                if pending.take().is_some() {
                    // The queued step appended `last_token`; the verify
                    // scores it together with the draft instead.
                    executor
                        .truncate_cached_tokens(executor.cached_tokens() - 1)
                        .map_err(|error| ChatGenerationError::message(error.to_string()))?;
                }
                // The serial worker verifies one sequence at a time.
                let span = tracing::info_span!(
                    "chat.verify_step",
                    step,
                    batch = 1,
                    drafted = draft.len(),
                    accepted = Empty,
                    decode_ms = Empty,
                    mlx.active_bytes = Empty,
                    mlx.peak_bytes = Empty
                );
                let ((outcome, verified), step_ms) = timed(&span, "decode_ms", || {
                    speculation::verify(
                        &self.format,
                        &mut executor,
                        &mut turn,
                        last_token,
                        &draft,
                        gpu_verify,
                    )
                    .map_err(ChatGenerationError::message)
                })?;
                span.record("accepted", outcome.accepted);
                crate::gpu::Memory::record_on(&span);
                decode_ms.push(step_ms);
                draft_length.observe(outcome.drafted, outcome.accepted);
                speculation_stats.record(&outcome);
                last_token_at = Instant::now();
                for verified_token in &verified {
                    if verified_token.visible {
                        text.push(&self.format, verified_token.token, on_token)?;
                    }
                    if let TurnStep::Stop(reason) = verified_token.step {
                        finish_reason = reason;
                        break 'decode;
                    }
                }
                last_token = verified
                    .last()
                    .ok_or_else(|| ChatGenerationError::message("a verify emits a token"))?
                    .token;
            }

            if pipelined && pending.is_none() {
                // After the prefill token or a rejected GPU draw; later steps
                // were queued above. Every draw so far is committed.
                deadline.check()?;
                let rule = turn.gpu_rule(&self.format, 0).ok_or_else(|| {
                    ChatGenerationError::message("GPU pick rule changed within a turn")
                })?;
                pending = Some(
                    executor
                        .decode_picks(last_token, &rule)
                        .map_err(|error| ChatGenerationError::message(error.to_string()))?,
                );
            } else if !pipelined {
                deadline.check()?;
                // One span per generated token, ending with its logit readback.
                let span = tracing::info_span!("chat.decode_step", step, decode_ms = Empty);
                let (next, step_ms) = timed(&span, "decode_ms", || {
                    executor
                        .decode_last_logits(last_token)
                        .map_err(|error| ChatGenerationError::message(error.to_string()))
                })?;
                logits = next;
                decode_ms.push(step_ms);
                deadline.check()?;
            }
        }
        // Copies the reusable prompt prefixes out of this turn's K/V; after the
        // last token, so it never delays the first one.
        let store = tracing::info_span!("chat.prefix_cache.store", store_ms = Empty);
        let (cache_write_tokens, _) = timed(&store, "store_ms", || {
            prefix_cache::remember(
                &self.weights,
                &mut self.prefix_cache,
                &self.format,
                &executor,
                request,
                &input_ids,
            )
        })?;

        deadline.check()?;
        let end = text.finish(
            &self.format,
            turn.generated(),
            request.tools,
            finish_reason == ChatFinishReason::Eos,
            on_token,
        )?;
        let output = turn.finish()?;
        let decode_total_ms = decode_ms.iter().sum();
        let generated_tokens = output.generated.len();
        deadline.check()?;
        Ok(ChatGeneration {
            text: end.text,
            turn: end.turn,
            generated_token_ids: output.generated,
            finish_reason,
            logprobs: output.logprobs,
            sampling: Some(output.sampling),
            metrics: ChatGenerationMetrics {
                context_tokens: self.context_limit,
                planned_kv_bytes: self.planned_kv_bytes,
                session_load_ms: self.load_ms,
                render_ms,
                prefill_ms,
                time_to_first_token_ms: end.time_to_first_token_ms,
                decode_ms,
                decode_total_ms,
                prompt_tokens: input_ids.len(),
                cached_prompt_tokens,
                cache_write_tokens,
                generated_tokens,
                speculation: SpeculationReceipt::from_stats(speculation_stats),
            },
        })
    }

    /// The checkpoint facts every turn is prepared from.
    pub(crate) fn turn_model(&self) -> TurnModel<'_> {
        TurnModel {
            format: &self.format,
            sampling_defaults: self.sampling_defaults,
            model: &self.model,
            vocabulary_size: self.vocabulary_size,
            context_limit: self.context_limit,
        }
    }

    /// Renders and encodes one turn; the render span covers both.
    fn render(&self, request: ChatRequest<'_>) -> Result<Prompt, String> {
        self.format.prompt(request.conversation(), true)
    }
}

impl ChatBackend for ChatSession {
    fn load_ms(&self) -> f64 {
        self.load_ms()
    }

    fn generate_with_timeout(
        &mut self,
        request: ChatRequest<'_>,
        timeout: Duration,
        on_token: &mut dyn FnMut(TurnDelta) -> Result<(), String>,
    ) -> Result<ChatGeneration, ChatGenerationError> {
        self.generate_with_timeout(request, timeout, on_token)
    }
}

fn read_config(model: &Path) -> Result<(ChatConfig, String), String> {
    let path = model.join("config.json");
    let raw = fs::read_to_string(&path)
        .map_err(|_| String::from("local model config.json could not be read"))?;
    let config = serde_json::from_str(&raw)
        .map_err(|_| String::from("local model config.json could not be parsed"))?;
    Ok((config, raw))
}

fn load_config(
    model: &Path,
    limits: ResidentChatLimits,
) -> Result<(ChatConfig, qwen::forward::Qwen3ResidentChatPlan, String), String> {
    let (config, raw) = read_config(model)?;
    // Reject context/KV admission before checkpoint payload loading.
    let plan = qwen::forward::Qwen3ForwardConfig::parse(&raw)
        .and_then(|config| {
            config.resident_chat_plan(
                limits.context_tokens(),
                limits.kv_budget_bytes(),
                SERVING_PRECISION,
            )
        })
        .map_err(|error| error.to_string())?;
    Ok((
        config,
        plan,
        format!("{:x}", Sha256::digest(raw.as_bytes())),
    ))
}

fn validate_request(request: ChatRequest<'_>) -> Result<(), String> {
    if request.messages.is_empty() || request.messages.len() > MAX_CHAT_MESSAGES {
        return Err(format!(
            "chat requires 1 through {MAX_CHAT_MESSAGES} messages; received {}",
            request.messages.len()
        ));
    }
    if request.tools.len() > MAX_CHAT_TOOLS {
        return Err(format!(
            "chat supports at most {MAX_CHAT_TOOLS} tools; received {}",
            request.tools.len()
        ));
    }
    if request.max_tokens == Some(0) {
        return Err(String::from("chat generation budget must be nonempty"));
    }
    let message_bytes = request
        .messages
        .iter()
        .try_fold(0_usize, |total, message| {
            let tool_call_bytes = message.tool_calls.iter().try_fold(0_usize, |total, call| {
                let arguments = serde_json::to_vec(&call.arguments).map_err(|_| {
                    String::from("chat tool-call arguments could not be serialized")
                })?;
                total
                    .checked_add(call.name.len())
                    .and_then(|total| total.checked_add(arguments.len()))
                    .ok_or_else(|| String::from("chat input byte count overflows"))
            })?;
            [
                message.content.len(),
                message.reasoning_content.as_ref().map_or(0, String::len),
                message.tool_call_id.as_ref().map_or(0, String::len),
                message.name.as_ref().map_or(0, String::len),
                tool_call_bytes,
            ]
            .into_iter()
            .try_fold(total, |total, bytes| {
                total
                    .checked_add(bytes)
                    .ok_or_else(|| String::from("chat input byte count overflows"))
            })
        })?;
    let input_bytes = request
        .tools
        .iter()
        .try_fold(message_bytes, |total, tool| {
            let encoded = serde_json::to_vec(tool)
                .map_err(|_| String::from("chat tool schema could not be serialized"))?;
            total
                .checked_add(encoded.len())
                .ok_or_else(|| String::from("chat input byte count overflows"))
        })?;
    if input_bytes > MAX_CHAT_INPUT_BYTES {
        return Err(format!(
            "chat messages and tools exceed the {MAX_CHAT_INPUT_BYTES}-byte limit"
        ));
    }
    Ok(())
}

/// Chooses each output token: greedy, seeded nucleus sampling, or either
/// under a JSON-schema grammar mask.
struct TokenPicker {
    /// The sampler with its `top_p` and `top_k`; `None` is greedy.
    policy: Option<(SamplingPolicy, f64, Option<usize>)>,
    #[cfg(feature = "structured-output")]
    constraint: Option<crate::qwen_constraints::ConstraintRun>,
    applied: AppliedSampling,
}

impl TokenPicker {
    fn new(
        request: ChatRequest<'_>,
        defaults: SamplingDefaults,
        model: &Path,
        vocabulary_size: usize,
    ) -> Result<Self, String> {
        let applied = request
            .sampling
            .resolve(defaults, request.json_schema.is_some());
        let policy = applied.seed.map(|seed| {
            let configuration = SamplingConfiguration {
                seed,
                temperature: applied.temperature,
            };
            (
                SamplingPolicy::new(configuration, vocabulary_size),
                applied.top_p,
                applied.top_k.and_then(|top_k| usize::try_from(top_k).ok()),
            )
        });
        #[cfg(feature = "structured-output")]
        let constraint = request
            .json_schema
            .map(|schema| {
                crate::qwen_constraints::ConstraintRun::load(
                    model,
                    crate::qwen_constraints::SchemaSource::Inline(&schema.to_string()),
                )
                .map_err(|error| format!("JSON schema could not be compiled: {error}"))
            })
            .transpose()?;
        #[cfg(not(feature = "structured-output"))]
        if request.json_schema.is_some() {
            let _ = model;
            return Err(
                "JSON schema output requires a server built with the structured-output feature"
                    .into(),
            );
        }
        Ok(Self {
            policy,
            #[cfg(feature = "structured-output")]
            constraint,
            applied,
        })
    }

    /// How the GPU picks this turn's tokens, with the variate `ahead` draws
    /// past the next uncommitted one, or `None` when every step must read
    /// the whole logit row: under a grammar mask, or sampling without
    /// `top_k`, whose nucleus normalizer spans the whole row. Candidates
    /// cover the receipts `top_logprobs` asks for and the `top_k` draw.
    fn gpu_rule(
        &self,
        ahead: usize,
        top_logprobs: Option<u8>,
        vocabulary_size: usize,
    ) -> Option<Qwen3PickRule> {
        #[cfg(feature = "structured-output")]
        if self.constraint.is_some() {
            return None;
        }
        let receipts = top_logprobs.map_or(0, |top| usize::from(top) + 9);
        let rule = match &self.policy {
            None => Qwen3PickRule {
                selection: Qwen3Selection::Greedy,
                candidates: receipts,
            },
            Some((policy, top_p, top_k)) => {
                let top_k = (*top_k)?;
                Qwen3PickRule {
                    selection: Qwen3Selection::TopKNucleus {
                        temperature: self.applied.temperature,
                        top_p: *top_p,
                        top_k,
                        uniforms: vec![policy.uniform_ahead(ahead)],
                    },
                    candidates: receipts.max(top_k + 1),
                }
            }
        };
        (rule.candidates < vocabulary_size).then_some(rule)
    }

    /// The exact token for a GPU-picked step, from its candidates, or `None`
    /// when the caller must pick from the whole row. Greedy keeps the GPU
    /// argmax; a `top_k` draw is re-derived on the host and consumes the
    /// step's variate.
    fn pick_from_candidates(
        &mut self,
        gpu_token: i32,
        candidates: Option<&Qwen3RowCandidates>,
    ) -> Result<Option<i32>, String> {
        match (self.policy.as_mut(), candidates) {
            (None, _) => Ok(Some(gpu_token)),
            (Some((policy, top_p, Some(top_k))), Some(candidates)) => {
                policy.sample_top_k_candidates(candidates, *top_p, *top_k)
            }
            (Some(_), _) => Ok(None),
        }
    }

    /// Returns the next token and whether it completed the schema grammar.
    fn pick(&mut self, logits: &[f32]) -> Result<(i32, bool), String> {
        #[cfg(feature = "structured-output")]
        if let Some(constraint) = self.constraint.as_mut() {
            let (token, _) = match self.policy.as_mut() {
                None => constraint.sample(logits, false),
                // Validation rejects a constrained top_p below one.
                Some((policy, ..)) => policy.sample_constrained(constraint, logits, false),
            }
            .map_err(|error| error.to_string())?;
            return Ok((token, constraint.is_complete()));
        }
        let token = match self.policy.as_mut() {
            None => greedy_token(logits)?,
            Some((policy, top_p, top_k)) => policy.sample_nucleus(logits, *top_p, *top_k)?,
        };
        Ok((token, false))
    }

    /// Validates a completed schema output independently of the grammar.
    #[cfg_attr(
        not(feature = "structured-output"),
        allow(
            clippy::unused_self,
            clippy::unnecessary_wraps,
            reason = "only schema output has anything to validate"
        )
    )]
    fn check_complete_output(&self) -> Result<(), String> {
        #[cfg(feature = "structured-output")]
        if let Some(constraint) = self.constraint.as_ref().filter(|run| run.is_complete()) {
            constraint
                .validated_output()
                .map_err(|error| format!("schema output failed validation: {error}"))?;
        }
        Ok(())
    }
}

/// A per-turn seed for unseeded sampling, from the standard library's
/// randomly keyed hasher.
fn fresh_seed() -> u64 {
    use std::hash::BuildHasher as _;
    std::hash::RandomState::new().hash_one(std::time::SystemTime::now())
}

fn greedy_token(logits: &[f32]) -> Result<i32, String> {
    let Some((&first, rest)) = logits.split_first() else {
        return Err(String::from("model produced no vocabulary logits"));
    };
    if !first.is_finite() {
        return Err(String::from("model produced non-finite vocabulary logits"));
    }
    let mut best = 0;
    for (index, &value) in rest.iter().enumerate() {
        if !value.is_finite() {
            return Err(String::from("model produced non-finite vocabulary logits"));
        }
        if value > logits[best] {
            best = index + 1;
        }
    }
    i32::try_from(best)
        .map_err(|_| String::from("model vocabulary token ID does not fit server token IDs"))
}

/// The whole logit row of a single-row GPU pick, for steps its candidates
/// cannot settle.
fn full_row(picks: &Qwen3TokenPicks) -> Result<Vec<f32>, String> {
    picks
        .full_rows()
        .map_err(|error| error.to_string())?
        .pop()
        .ok_or_else(|| String::from("a decode step has one logit row"))
}

/// Runs `work` inside `span` and records its wall time in milliseconds as
/// `field`, returning the same figure for the JSON metrics. For MLX work that
/// ends in a host readback, the span covers the GPU evaluation.
fn timed<T, E>(
    span: &tracing::Span,
    field: &'static str,
    work: impl FnOnce() -> Result<T, E>,
) -> Result<(T, f64), E> {
    let started = Instant::now();
    let value = span.in_scope(work)?;
    let milliseconds = elapsed_ms(started.elapsed());
    span.record(field, milliseconds);
    Ok((value, milliseconds))
}

fn elapsed_ms(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1000.0
}

#[cfg(test)]
mod tests {
    use std::{
        env,
        path::{Path, PathBuf},
        time::{Duration, Instant},
    };

    use proptest::prelude::*;
    use serde_json::{Value, json};
    use sha2::{Digest, Sha256};

    use super::{
        ChatFinishReason, ChatGenerationError, ChatMessage, ChatRequest, ChatRole, ChatSession,
        ChatToolCall, ChatToolResult, GenerationDeadline, MAX_CHAT_INPUT_BYTES, ResidentChatLimits,
        SamplingDefaults, TurnModel, TurnStart, TurnStep, render_generation_prompt,
    };
    use chat_format::{
        ChatFormat, ChatTemplate, SpecialTokens,
        test_model::{ModelDir, VOCABULARY_SIZE},
    };

    #[cfg(feature = "structured-output")]
    #[test]
    fn speculation_on_refuses_json_schema_output_and_auto_allows_it() {
        use engine::speculative::SpeculationRequest;
        let mut controls = super::GenerationControls {
            json_schema: Some(json!({"type":"object"})),
            speculation: SpeculationRequest::Enabled,
            ..super::GenerationControls::default()
        };
        let error = controls.validate(false).expect_err("on with a schema");
        assert!(error.contains("speculation"), "{error}");
        controls.speculation = SpeculationRequest::Automatic;
        controls
            .validate(false)
            .expect("automatic speculation skips schema turns");
    }

    const TEMPLATE: &str = include_str!("../../../fixtures/qwen3-0.6b/chat-template.jinja");
    const MANIFEST: &str = include_str!("../../../fixtures/qwen3-0.6b/chat-template.manifest.json");
    const TEMPLATE_SHA256: &str =
        "e132ae041e1217b5e1114eb9dc292484a7f478df945d72fa49ba01b88d8ec01a";

    fn render(messages: &[ChatMessage], tools: &[Value], enable_thinking: bool) -> String {
        let template = ChatTemplate::parse(TEMPLATE.to_owned(), SpecialTokens::default())
            .expect("fixture template parses");
        let mut request = ChatRequest::new(messages, 1);
        request.tools = tools;
        request.enable_thinking = enable_thinking;
        template
            .render(request.conversation(), true)
            .expect("fixture template renders")
    }

    #[test]
    fn qwen_fixture_has_pinned_checkpoint_provenance() {
        let manifest: Value = serde_json::from_str(MANIFEST).expect("fixture manifest JSON");
        assert_eq!(manifest["schema_version"], 1);
        assert_eq!(manifest["source"]["model_id"], "Qwen/Qwen3-0.6B");
        assert_eq!(
            manifest["source"]["revision"],
            "c1899de289a04d12100db370d81485cdf75e47ca"
        );
        assert_eq!(
            manifest["source"]["tokenizer_config_sha256"],
            "d5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101"
        );
        let digest = format!("{:x}", Sha256::digest(TEMPLATE.as_bytes()));
        assert_eq!(digest, TEMPLATE_SHA256);
        assert_eq!(manifest["source"]["template_sha256"], TEMPLATE_SHA256);
    }

    /// A model that always picks end-of-turn: without `ignore_eos` the first
    /// pick ends the turn; with it every pick is visible text and the turn
    /// ends at its output limit, with the final text agreeing with the
    /// streamed text.
    #[test]
    fn ignore_eos_runs_the_turn_loop_to_its_output_limit() {
        const EOS: usize = 3;
        let model = ModelDir::new(
            &json!({"chat_template": "{{ messages[0].content }}"}),
            &json!({"eos_token_id": EOS, "vocab_size": VOCABULARY_SIZE}),
            None,
        );
        let format = ChatFormat::load(model.path(), VOCABULARY_SIZE).expect("test format");
        let messages = [ChatMessage::text(ChatRole::User, "hi")];
        for ignore_eos in [false, true] {
            let mut request = ChatRequest::new(&messages, 4);
            request.ignore_eos = ignore_eos;
            let turn_model = TurnModel {
                format: &format,
                sampling_defaults: SamplingDefaults::default(),
                model: model.path(),
                vocabulary_size: VOCABULARY_SIZE,
                context_limit: 64,
            };
            let (_, mut turn, mut text) =
                TurnStart::prepare(turn_model, request, GenerationDeadline::unlimited())
                    .expect("prepare")
                    .into_parts();
            let mut steps = Vec::new();
            let mut streamed = String::new();
            loop {
                let mut logits = vec![0.0_f32; VOCABULARY_SIZE];
                logits[EOS] = 1.0;
                let accepted = turn.pick(&format, &mut logits).expect("pick");
                if accepted.visible {
                    text.push(&format, accepted.token, &mut |delta| {
                        if let chat_format::TurnDelta::Text(delta) = delta {
                            streamed.push_str(&delta);
                        }
                        Ok(())
                    })
                    .expect("push");
                }
                steps.push((accepted.visible, accepted.step));
                if accepted.step != TurnStep::Continue {
                    break;
                }
            }
            let final_text = text
                .finish(&format, turn.generated(), &[], true, &mut |delta| {
                    if let chat_format::TurnDelta::Text(delta) = delta {
                        streamed.push_str(&delta);
                    }
                    Ok(())
                })
                .expect("final text agrees with the streamed text")
                .text;
            assert_eq!(final_text, streamed);
            if ignore_eos {
                let continued = (true, TurnStep::Continue);
                let last = (true, TurnStep::Stop(ChatFinishReason::Length));
                assert_eq!(steps, [continued, continued, continued, last]);
                assert_eq!(final_text.matches("<|im_end|>").count(), 4);
            } else {
                assert_eq!(steps, [(false, TurnStep::Stop(ChatFinishReason::Eos))]);
                assert_eq!(final_text, "");
            }
        }
    }

    #[test]
    fn resident_limits_preserve_context_and_mib_budget() {
        let limits = ResidentChatLimits::from_mib(16_384, 8_192);
        assert_eq!(limits.context_tokens(), 16_384);
        assert_eq!(limits.kv_budget_bytes(), 8_192 * 1024 * 1024);
        assert_eq!(limits.prefix_cache_bytes(), 2_048 * 1024 * 1024);
        let sized = limits.with_prefix_cache_mib(3);
        assert_eq!(sized.prefix_cache_bytes(), 3 * 1024 * 1024);
        assert_eq!(sized.kv_budget_bytes(), limits.kv_budget_bytes());
        assert_eq!(limits.with_prefix_cache_mib(0).prefix_cache_bytes(), 0);
    }

    #[test]
    fn generation_template_renders_one_nonthinking_user_message_before_weights() {
        let model = ModelDir::new(
            &json!({"chat_template": TEMPLATE, "bos_token": null, "eos_token": "<|im_end|>"}),
            &json!({"eos_token_id": 3, "vocab_size": VOCABULARY_SIZE}),
            None,
        );
        let rendered = render_generation_prompt(model.path(), "hello")
            .expect("bounded generation template render");
        assert_eq!(rendered.template_sha256, TEMPLATE_SHA256);
        assert_eq!(
            rendered.rendered,
            "<|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        );
    }

    /// The prompt `mx generate` renders carries the BOS a template prints.
    #[test]
    fn generation_template_renders_the_tokenizer_bos() {
        let model = ModelDir::new(
            &json!({"chat_template": "{{ bos_token }}{{ messages[0].content }}", "bos_token": "<s>"}),
            &json!({"eos_token_id": [1, 2], "vocab_size": VOCABULARY_SIZE}),
            None,
        );
        let rendered = render_generation_prompt(model.path(), "hi").map(|prompt| prompt.rendered);
        assert_eq!(rendered, Ok(String::from("<s>hi")));
    }

    #[test]
    fn generation_template_rejects_oversize_user_input_before_template_reads() {
        let prompt = "x".repeat(MAX_CHAT_INPUT_BYTES + 1);
        let error = render_generation_prompt(Path::new("missing-model"), &prompt)
            .expect_err("input boundary must run before template reads");
        assert!(error.contains("byte limit"));
    }

    #[test]
    fn generation_deadline_is_absolute_and_expires_at_its_boundary() {
        let now = Instant::now();
        let unlimited = GenerationDeadline::unlimited();
        assert!(!unlimited.expired_at(now));
        assert!(unlimited.check().is_ok());

        let expired = GenerationDeadline {
            deadline: Some(now),
        };
        assert!(expired.expired_at(now));
        assert!(matches!(
            expired.check(),
            Err(ChatGenerationError::DeadlineExceeded)
        ));

        let stage_deadline = GenerationDeadline {
            deadline: now.checked_add(Duration::from_millis(10)),
        };
        assert!(stage_deadline.check_at(now).is_ok());
        // Checking again does not extend the absolute deadline.
        assert!(matches!(
            stage_deadline.check_at(now + Duration::from_millis(10)),
            Err(ChatGenerationError::DeadlineExceeded)
        ));
    }

    #[test]
    #[ignore = "requires METALLIX_QWEN_MODEL and a local Apple-Silicon Metal checkpoint"]
    fn checkpoint_callback_cancellation_allows_session_reuse() {
        let model = env::var_os("METALLIX_QWEN_MODEL")
            .map(PathBuf::from)
            .expect("explicit checkpoint test requires METALLIX_QWEN_MODEL");
        let limits = ResidentChatLimits::from_mib(2_048, 1_024);
        let mut session = ChatSession::load(&model, limits).expect("checkpoint session load");
        let load_ms = session.load_ms();
        let recovery_messages = [ChatMessage::text(
            ChatRole::User,
            "Reply with a short recovery acknowledgement.",
        )];
        let baseline = session
            .generate_with_timeout(
                ChatRequest::new(&recovery_messages, 8),
                Duration::from_secs(60),
                &mut |_| Ok(()),
            )
            .expect("clean baseline generation");
        let cancelled_messages = [ChatMessage::text(
            ChatRole::User,
            "Reply with a short acknowledgement.",
        )];
        let mut cancelled_deltas = Vec::new();
        let cancelled = session.generate_with_timeout(
            ChatRequest::new(&cancelled_messages, 32),
            Duration::from_secs(60),
            &mut |delta| {
                cancelled_deltas.push(delta);
                Err(String::from(
                    "test cancellation after first generated delta",
                ))
            },
        );
        assert!(matches!(
            cancelled,
            Err(ChatGenerationError::Message(message))
                if message == "test cancellation after first generated delta"
        ));
        assert_eq!(cancelled_deltas.len(), 1);
        assert!(
            matches!(&cancelled_deltas[0], chat_format::TurnDelta::Text(text) if !text.is_empty())
        );

        let recovered = session
            .generate_with_timeout(
                ChatRequest::new(&recovery_messages, 8),
                Duration::from_secs(60),
                &mut |_| Ok(()),
            )
            .expect("fresh generation after callback cancellation");
        assert_eq!(recovered.metrics.context_tokens, limits.context_tokens());
        assert_eq!(
            recovered.metrics.session_load_ms.to_bits(),
            load_ms.to_bits()
        );
        assert!(recovered.metrics.prompt_tokens > 0);
        assert_eq!(
            recovered.metrics.generated_tokens,
            recovered.generated_token_ids.len()
        );
        assert!(recovered.metrics.generated_tokens > 0);
        assert_eq!(recovered.generated_token_ids, baseline.generated_token_ids);
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(128))]

        #[test]
        fn generation_deadline_expiry_is_monotone(
            earlier_ms in 0_u64..10_000,
            extension_ms in 0_u64..10_000,
            probe_ms in 0_u64..20_000,
        ) {
            let origin = Instant::now();
            let earlier = GenerationDeadline {
                deadline: origin.checked_add(Duration::from_millis(earlier_ms)),
            };
            let later = GenerationDeadline {
                deadline: origin.checked_add(Duration::from_millis(earlier_ms + extension_ms)),
            };
            let probe = origin + Duration::from_millis(probe_ms);
            prop_assert!(!later.expired_at(probe) || earlier.expired_at(probe));
        }
    }

    #[test]
    fn qwen_fixture_renders_system_and_user_messages() {
        let rendered = render(
            &[
                ChatMessage::text(ChatRole::System, "You are concise."),
                ChatMessage::text(ChatRole::User, "hello"),
            ],
            &[],
            false,
        );
        assert_eq!(
            rendered,
            concat!(
                "<|im_start|>system\nYou are concise.<|im_end|>\n",
                "<|im_start|>user\nhello<|im_end|>\n",
                "<|im_start|>assistant\n<think>\n\n</think>\n\n",
            )
        );
    }

    #[test]
    fn qwen_fixture_control_disables_thinking_prefill() {
        let rendered = render(&[ChatMessage::text(ChatRole::User, "hello")], &[], true);
        assert_eq!(
            rendered,
            "<|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\n"
        );
        assert!(!rendered.contains("<think>"));
    }

    #[test]
    fn qwen_fixture_renders_tool_call_and_result_history() {
        let tool_result = ChatToolResult {
            tool_call_id: String::from("call_1"),
            name: Some(String::from("read_file")),
            content: String::from("title"),
        };
        let rendered = render(
            &[
                ChatMessage::text(ChatRole::System, "Use tools."),
                ChatMessage::text(ChatRole::User, "read README"),
                ChatMessage {
                    role: ChatRole::Assistant,
                    content: String::new(),
                    reasoning_content: None,
                    tool_calls: vec![ChatToolCall {
                        name: String::from("read_file"),
                        arguments: json!({"path": "README.md"}),
                    }],
                    tool_call_id: None,
                    name: None,
                },
                tool_result.into_message(),
                ChatMessage::text(ChatRole::User, "thanks"),
            ],
            &[json!({"name": "ping"})],
            false,
        );
        assert_eq!(
            rendered,
            concat!(
                "<|im_start|>system\nUse tools.\n\n",
                "# Tools\n\n",
                "You may call one or more functions to assist with the user query.\n\n",
                "You are provided with function signatures within <tools></tools> XML tags:\n",
                // transformers' `tojson` (Python `json.dumps` separators).
                "<tools>\n{\"name\": \"ping\"}\n</tools>\n\n",
                "For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n",
                "<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call><|im_end|>\n",
                "<|im_start|>user\nread README<|im_end|>\n",
                "<|im_start|>assistant\n<tool_call>\n{\"name\": \"read_file\", \"arguments\": {\"path\": \"README.md\"}}\n</tool_call><|im_end|>\n",
                "<|im_start|>user\n<tool_response>\ntitle\n</tool_response><|im_end|>\n",
                "<|im_start|>user\nthanks<|im_end|>\n",
                "<|im_start|>assistant\n<think>\n\n</think>\n\n",
            )
        );
    }

    #[test]
    fn tool_calls_expose_both_flat_and_function_views_for_templates() {
        let call = ChatToolCall {
            name: String::from("read_file"),
            arguments: serde_json::json!({"path": "README.md"}),
        };
        let value = serde_json::to_value(call).expect("tool call JSON");
        assert_eq!(value["name"], "read_file");
        assert_eq!(value["function"]["name"], "read_file");
        assert_eq!(value["function"]["arguments"]["path"], "README.md");
    }

    #[test]
    fn deepseek_template_shape_renders_mapping_and_nested_tool_calls() {
        let template = ChatTemplate::parse(
            concat!(
                "{% for message in messages %}{{ message.get('role') }}:{{ message.get('content') }};",
                "{% for call in message.get('tool_calls') or [] %}{{ call.function.name }}={{ call.function.arguments.path }};{% endfor %}",
                "{% endfor %}"
            )
            .to_owned(),
            SpecialTokens::default(),
        )
        .expect("DeepSeek-shaped template parses");
        let messages = [ChatMessage {
            role: ChatRole::Assistant,
            content: String::new(),
            reasoning_content: None,
            tool_calls: vec![ChatToolCall {
                name: String::from("read_file"),
                arguments: serde_json::json!({"path": "README.md"}),
            }],
            tool_call_id: None,
            name: None,
        }];
        let rendered = template
            .render(ChatRequest::new(&messages, 1).conversation(), false)
            .expect("DeepSeek-shaped context renders");
        assert_eq!(rendered, "assistant:;read_file=README.md;");
    }
}
