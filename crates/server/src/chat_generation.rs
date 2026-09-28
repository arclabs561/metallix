//! Resident Qwen chat generation over the checkpoint's own Jinja template.
//!
//! This is deliberately a single-session, single-sequence core.  It retains
//! model weights between calls but creates fresh KV state for each rendered
//! conversation. Resident chat has separate context and logical-KV admission;
//! the diagnostic forward cap remains unchanged.

use std::{
    fmt, fs,
    path::Path,
    time::{Duration, Instant},
};

use minijinja::{Environment, context};
use minijinja_contrib::pycompat::unknown_method_callback;
use qwen::metal::Qwen3MlxWeights;
use serde::{Deserialize, Serialize, ser::SerializeStruct};
use serde_json::Value;
use sha2::{Digest, Sha256};

use crate::qwen_tokenizer::QwenTokenizer;

const MAX_CHAT_TEMPLATE_BYTES: usize = 1024 * 1024;
const MAX_CHAT_RENDERED_BYTES: usize = 1024 * 1024;
const MAX_CHAT_INPUT_BYTES: usize = 1024 * 1024;
const MAX_CHAT_MESSAGES: usize = 256;
const MAX_CHAT_TOOLS: usize = 64;
const TEMPLATE_FUEL: u64 = 100_000;
const MIB_BYTES: u64 = 1024 * 1024;

/// One resident-chat admission contract shared by session loading and turns.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct ResidentChatLimits {
    context_tokens: usize,
    kv_budget_bytes: u64,
}

impl ResidentChatLimits {
    #[must_use]
    pub(crate) const fn from_mib(context_tokens: usize, kv_budget_mib: u32) -> Self {
        Self {
            context_tokens,
            kv_budget_bytes: (kv_budget_mib as u64) * MIB_BYTES,
        }
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

/// A checkpoint-template role with the spellings expected by Qwen's Jinja.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "lowercase")]
pub(crate) enum ChatRole {
    System,
    User,
    Assistant,
    Tool,
}

/// One completed assistant tool call retained in conversation history.
#[derive(Clone, Debug, Deserialize)]
pub(crate) struct ChatToolCall {
    pub name: String,
    pub arguments: Value,
}

impl Serialize for ChatToolCall {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let mut state = serializer.serialize_struct("ChatToolCall", 3)?;
        state.serialize_field("name", &self.name)?;
        state.serialize_field("arguments", &self.arguments)?;
        state.serialize_field(
            "function",
            &serde_json::json!({"name": self.name, "arguments": self.arguments}),
        )?;
        state.end()
    }
}

/// One tool result which can be converted into a template-ready tool message.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub(crate) struct ChatToolResult {
    pub tool_call_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    pub content: String,
}

impl ChatToolResult {
    /// Returns the tool-role message consumed by the checkpoint chat template.
    #[must_use]
    pub(crate) fn into_message(self) -> ChatMessage {
        ChatMessage {
            role: ChatRole::Tool,
            content: self.content,
            reasoning_content: None,
            tool_calls: Vec::new(),
            tool_call_id: Some(self.tool_call_id),
            name: self.name,
        }
    }
}

/// One concrete message supplied to the checkpoint's chat template.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub(crate) struct ChatMessage {
    pub role: ChatRole,
    pub content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<ChatToolCall>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

impl ChatMessage {
    /// Makes a text-only system, user, or assistant message.
    #[must_use]
    pub(crate) fn text(role: ChatRole, content: impl Into<String>) -> Self {
        Self {
            role,
            content: content.into(),
            reasoning_content: None,
            tool_calls: Vec::new(),
            tool_call_id: None,
            name: None,
        }
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
    let template_source = load_template(model)?;
    let template_sha256 = format!("{:x}", Sha256::digest(template_source.as_bytes()));
    let template = parse_template(template_source)?;
    let rendered = render_template(&template, request)?;
    Ok(GenerationTemplatePrompt {
        rendered,
        template_sha256,
    })
}

/// One chat turn. Tool schemas remain JSON because their externally defined
/// schema is a caller boundary, while message history stays typed.
#[derive(Clone, Copy, Debug)]
pub(crate) struct ChatRequest<'a> {
    pub(crate) messages: &'a [ChatMessage],
    pub(crate) tools: &'a [Value],
    pub(crate) max_tokens: u32,
    pub(crate) enable_thinking: bool,
    /// Optional model-specific reasoning effort passed through to templates.
    pub(crate) reasoning_effort: Option<&'a str>,
}

impl<'a> ChatRequest<'a> {
    /// Creates a normal non-thinking turn with no tools.
    #[must_use]
    pub(crate) const fn new(messages: &'a [ChatMessage], max_tokens: u32) -> Self {
        Self {
            messages,
            tools: &[],
            max_tokens,
            enable_thinking: false,
            reasoning_effort: None,
        }
    }
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
    pub(crate) generated_tokens: usize,
}

/// The completed visible text and model-level generation details for one turn.
#[derive(Clone, Debug, Serialize)]
pub(crate) struct ChatGeneration {
    pub(crate) text: String,
    pub(crate) generated_token_ids: Vec<i32>,
    pub(crate) finish_reason: ChatFinishReason,
    pub(crate) metrics: ChatGenerationMetrics,
}

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
    tokenizer: QwenTokenizer,
    template: Environment<'static>,
    eos_token_id: i32,
    vocabulary_size: usize,
    context_limit: usize,
    kv_budget_bytes: u64,
    planned_kv_bytes: u64,
    load_ms: f64,
    config_sha256: String,
    tokenizer_sha256: String,
    template_sha256: String,
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
        on_token: &mut dyn FnMut(&str) -> Result<(), String>,
    ) -> Result<ChatGeneration, ChatGenerationError>;
}

#[derive(serde::Deserialize)]
struct ChatConfig {
    eos_token_id: i32,
    vocab_size: usize,
}

impl ChatSession {
    /// Loads and prepares a checkpoint once, including its template and tokenizer.
    pub(crate) fn load(model: &Path, limits: ResidentChatLimits) -> Result<Self, String> {
        let started = Instant::now();
        let (config, plan, config_sha256) = load_config(model, limits)?;
        let tokenizer = QwenTokenizer::load(model)?;
        let tokenizer_sha256 = tokenizer.source_sha256().to_owned();
        tokenizer.check_model_vocabulary(config.vocab_size, config.eos_token_id)?;
        let template_source = load_template(model)?;
        let template_sha256 = format!("{:x}", Sha256::digest(template_source.as_bytes()));
        let template = parse_template(template_source)?;

        let mut weights = Qwen3MlxWeights::load(model).map_err(|error| error.to_string())?;
        weights
            .prepare_float32()
            .map_err(|error| error.to_string())?;
        Ok(Self {
            weights,
            tokenizer,
            template,
            eos_token_id: config.eos_token_id,
            vocabulary_size: config.vocab_size,
            context_limit: limits.context_tokens(),
            kv_budget_bytes: limits.kv_budget_bytes(),
            planned_kv_bytes: plan.planned_kv_bytes(),
            load_ms: elapsed_ms(started.elapsed()),
            config_sha256,
            tokenizer_sha256,
            template_sha256,
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
        &self.template_sha256
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
        let ids = self.tokenizer.encode_prompt(label)?;
        match ids.as_slice() {
            [token] => Ok(*token),
            _ => Err(format!(
                "decision answer label {label:?} must encode as exactly one token"
            )),
        }
    }

    /// Renders and pre-fills one fresh chat prompt without selecting or decoding
    /// any output token. Each call creates a separate resident executor, so no
    /// question can contribute K/V state to another question.
    pub(crate) fn prefill_chat(&mut self, messages: &[ChatMessage]) -> Result<ChatPrefill, String> {
        let request = ChatRequest::new(messages, 1);
        validate_request(request)?;
        let render_started = Instant::now();
        let prompt = self.render(request)?;
        let render_ms = elapsed_ms(render_started.elapsed());
        let input_ids = self.tokenizer.encode_prompt(&prompt)?;
        validate_ids(&input_ids, self.eos_token_id, self.vocabulary_size)?;
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
        let prefill_started = Instant::now();
        let logits = executor
            .prefill_last_logits(&input_ids)
            .map_err(|error| error.to_string())?;
        Ok(ChatPrefill {
            logits,
            prompt_tokens: input_ids.len(),
            render_ms,
            prefill_ms: elapsed_ms(prefill_started.elapsed()),
        })
    }

    /// Generates one complete chat turn and streams only cumulative-decoder text deltas.
    pub(crate) fn generate(
        &mut self,
        request: ChatRequest<'_>,
        on_token: &mut dyn FnMut(&str) -> Result<(), String>,
    ) -> Result<ChatGeneration, String> {
        self.generate_with_deadline(request, GenerationDeadline::unlimited(), on_token)
            .map_err(|error| error.to_string())
    }

    /// Generates with a budget that is checked between host-visible phases.
    pub(crate) fn generate_with_timeout(
        &mut self,
        request: ChatRequest<'_>,
        timeout: Duration,
        on_token: &mut dyn FnMut(&str) -> Result<(), String>,
    ) -> Result<ChatGeneration, ChatGenerationError> {
        // Start before validation and rendering so the caller budgets the full turn.
        self.generate_with_deadline(request, GenerationDeadline::after(timeout), on_token)
    }

    fn generate_with_deadline(
        &mut self,
        request: ChatRequest<'_>,
        deadline: GenerationDeadline,
        on_token: &mut dyn FnMut(&str) -> Result<(), String>,
    ) -> Result<ChatGeneration, ChatGenerationError> {
        deadline.check()?;
        validate_request(request)?;
        let turn_started = Instant::now();
        let render_started = Instant::now();
        let prompt = self.render(request)?;
        let render_ms = elapsed_ms(render_started.elapsed());
        deadline.check()?;
        let input_ids = self.tokenizer.encode_prompt(&prompt)?;
        deadline.check()?;
        validate_ids(&input_ids, self.eos_token_id, self.vocabulary_size)?;
        let total_tokens = input_ids
            .len()
            .checked_add(request.max_tokens as usize)
            .ok_or_else(|| {
                ChatGenerationError::message("chat prompt plus generation budget overflows")
            })?;
        if total_tokens > self.context_limit {
            return Err(ChatGenerationError::message(format!(
                "chat requires prompt_tokens + max_tokens <= {}; received {} + {} = {total_tokens}",
                self.context_limit,
                input_ids.len(),
                request.max_tokens,
            )));
        }

        deadline.check()?;
        let mut executor = self
            .weights
            .resident_chat_executor(self.context_limit, self.kv_budget_bytes)
            .map_err(|error| ChatGenerationError::message(error.to_string()))?;
        deadline.check()?;
        let prefill_started = Instant::now();
        let mut logits = executor
            .prefill_last_logits(&input_ids)
            .map_err(|error| ChatGenerationError::message(error.to_string()))?;
        let prefill_ms = elapsed_ms(prefill_started.elapsed());
        deadline.check()?;
        let mut generated = Vec::with_capacity(request.max_tokens as usize);
        let mut emitted = String::new();
        let mut text_decoder = QwenTokenizer::generated_decoder();
        let mut decode_ms = Vec::new();
        let mut ttft = None;
        let mut finish_reason = ChatFinishReason::Length;

        for step in 0..request.max_tokens {
            deadline.check()?;
            let token = greedy_token(&logits)?;
            generated.push(token);
            if token == self.eos_token_id {
                finish_reason = ChatFinishReason::Eos;
                break;
            }
            if let Some(delta) = self
                .tokenizer
                .decode_generated_token(&mut text_decoder, token)?
            {
                emit_delta(&delta, &mut emitted, on_token, &mut ttft, turn_started)?;
            }

            if step + 1 < request.max_tokens {
                deadline.check()?;
                let decode_started = Instant::now();
                logits = executor
                    .decode_last_logits(token)
                    .map_err(|error| ChatGenerationError::message(error.to_string()))?;
                decode_ms.push(elapsed_ms(decode_started.elapsed()));
                deadline.check()?;
            }
        }

        let visible_generated = generated
            .strip_suffix(&[self.eos_token_id])
            .unwrap_or(&generated);
        deadline.check()?;
        let text = self.tokenizer.decode_generated(visible_generated)?;
        let remaining = text.strip_prefix(&emitted).ok_or_else(|| {
            ChatGenerationError::message(
                "incremental tokenizer decoder diverged from complete generated text",
            )
        })?;
        emit_delta(remaining, &mut emitted, on_token, &mut ttft, turn_started)?;
        let decode_total_ms = decode_ms.iter().sum();
        let generated_tokens = generated.len();
        deadline.check()?;
        Ok(ChatGeneration {
            text,
            generated_token_ids: generated,
            finish_reason,
            metrics: ChatGenerationMetrics {
                context_tokens: self.context_limit,
                planned_kv_bytes: self.planned_kv_bytes,
                session_load_ms: self.load_ms,
                render_ms,
                prefill_ms,
                time_to_first_token_ms: ttft,
                decode_ms,
                decode_total_ms,
                prompt_tokens: input_ids.len(),
                generated_tokens,
            },
        })
    }

    fn render(&self, request: ChatRequest<'_>) -> Result<String, String> {
        render_template(&self.template, request)
    }
}

fn render_template(
    template: &Environment<'static>,
    request: ChatRequest<'_>,
) -> Result<String, String> {
    let messages = serde_json::to_value(request.messages)
        .map_err(|_| String::from("chat messages could not be serialized"))?;
    let tools = serde_json::to_value(request.tools)
        .map_err(|_| String::from("chat tools could not be serialized"))?;
    let rendered = template
        .get_template("qwen_chat")
        .map_err(|error| format!("local chat template is unavailable: {error}"))?
        .render(context! {
            messages => messages,
            tools => tools,
            add_generation_prompt => true,
            enable_thinking => request.enable_thinking,
            thinking_mode => if request.enable_thinking { "thinking" } else { "non-thinking" },
            reasoning_effort => request.reasoning_effort.unwrap_or("low"),
            drop_thinking => !request.enable_thinking,
        })
        .map_err(|error| format!("local chat template could not be rendered: {error}"))?;
    if rendered.len() > MAX_CHAT_RENDERED_BYTES {
        return Err(format!(
            "rendered chat prompt exceeds the {MAX_CHAT_RENDERED_BYTES}-byte limit"
        ));
    }
    Ok(rendered)
}

impl ChatBackend for ChatSession {
    fn load_ms(&self) -> f64 {
        self.load_ms()
    }

    fn generate_with_timeout(
        &mut self,
        request: ChatRequest<'_>,
        timeout: Duration,
        on_token: &mut dyn FnMut(&str) -> Result<(), String>,
    ) -> Result<ChatGeneration, ChatGenerationError> {
        self.generate_with_timeout(request, timeout, on_token)
    }
}

fn load_config(
    model: &Path,
    limits: ResidentChatLimits,
) -> Result<(ChatConfig, qwen::forward::Qwen3ResidentChatPlan, String), String> {
    let path = model.join("config.json");
    let raw = fs::read_to_string(&path)
        .map_err(|_| String::from("local model config.json could not be read"))?;
    let config = serde_json::from_str(&raw)
        .map_err(|_| String::from("local model config.json could not be parsed"))?;
    // Reject context/KV admission before checkpoint payload loading.
    let plan = qwen::forward::Qwen3ForwardConfig::parse(&raw)
        .and_then(|config| {
            config.resident_chat_plan(limits.context_tokens(), limits.kv_budget_bytes())
        })
        .map_err(|error| error.to_string())?;
    Ok((
        config,
        plan,
        format!("{:x}", Sha256::digest(raw.as_bytes())),
    ))
}

fn load_template(model: &Path) -> Result<String, String> {
    let path = model.join("tokenizer_config.json");
    let raw = String::from_utf8(crate::qwen_tokenizer::read_regular_file(
        &path,
        MAX_CHAT_TEMPLATE_BYTES,
        "tokenizer_config.json",
    )?)
    .map_err(|_| String::from("local tokenizer_config.json could not be read"))?;
    let config: Value = serde_json::from_str(&raw)
        .map_err(|_| String::from("local tokenizer_config.json could not be parsed"))?;
    if let Some(template) = config
        .get("chat_template")
        .and_then(Value::as_str)
        .filter(|template| !template.is_empty())
    {
        return Ok(template.to_owned());
    }

    let external = model.join("chat_template.jinja");
    if !external.exists() {
        return Err(String::from(
            "local tokenizer_config.json has no chat_template and chat_template.jinja is absent",
        ));
    }
    let template = String::from_utf8(crate::qwen_tokenizer::read_regular_file(
        &external,
        MAX_CHAT_TEMPLATE_BYTES,
        "chat_template.jinja",
    )?)
    .map_err(|_| String::from("local chat_template.jinja could not be read"))?;
    if template.trim().is_empty() {
        return Err(String::from("local chat_template.jinja must not be empty"));
    }
    Ok(template)
}

fn parse_template(template_source: String) -> Result<Environment<'static>, String> {
    let mut template = Environment::new();
    template.set_unknown_method_callback(unknown_method_callback);
    template.set_fuel(Some(TEMPLATE_FUEL));
    template
        .add_template_owned("qwen_chat".to_owned(), template_source)
        .map_err(|error| format!("local chat template could not be parsed: {error}"))?;
    Ok(template)
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
    if request.max_tokens == 0 {
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

fn validate_ids(ids: &[i32], eos_token_id: i32, vocabulary_size: usize) -> Result<(), String> {
    if ids.is_empty()
        || ids.iter().chain(std::iter::once(&eos_token_id)).any(|&id| {
            usize::try_from(id)
                .ok()
                .is_none_or(|id| id >= vocabulary_size)
        })
    {
        return Err(String::from(
            "chat template token IDs or EOS are outside model vocabulary",
        ));
    }
    Ok(())
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

fn emit_delta(
    delta: &str,
    emitted: &mut String,
    on_token: &mut dyn FnMut(&str) -> Result<(), String>,
    ttft: &mut Option<f64>,
    turn_started: Instant,
) -> Result<(), String> {
    if !delta.is_empty() {
        on_token(delta)?;
        emitted.push_str(delta);
        if ttft.is_none() {
            *ttft = Some(elapsed_ms(turn_started.elapsed()));
        }
    }
    Ok(())
}

fn elapsed_ms(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1000.0
}

#[cfg(test)]
mod tests {
    use std::{
        fs,
        path::Path,
        time::{Duration, Instant},
    };

    use minijinja::context;
    use proptest::prelude::*;
    use serde_json::{Value, json};
    use sha2::{Digest, Sha256};

    use super::{
        ChatGenerationError, ChatMessage, ChatRole, ChatToolCall, ChatToolResult,
        GenerationDeadline, MAX_CHAT_INPUT_BYTES, MAX_CHAT_TEMPLATE_BYTES, ResidentChatLimits,
        load_template, parse_template, render_generation_prompt,
    };

    const TEMPLATE: &str = include_str!("../../../fixtures/qwen3-0.6b/chat-template.jinja");
    const MANIFEST: &str = include_str!("../../../fixtures/qwen3-0.6b/chat-template.manifest.json");
    const TEMPLATE_SHA256: &str =
        "e132ae041e1217b5e1114eb9dc292484a7f478df945d72fa49ba01b88d8ec01a";

    fn render(messages: &[ChatMessage], tools: &[Value], enable_thinking: bool) -> String {
        let template = parse_template(TEMPLATE.to_owned()).expect("fixture template parses");
        template
            .get_template("qwen_chat")
            .expect("fixture template exists")
            .render(context! {
                messages => messages,
                tools => tools,
                add_generation_prompt => true,
                enable_thinking => enable_thinking,
            })
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

    #[test]
    fn resident_limits_preserve_context_and_mib_budget() {
        let limits = ResidentChatLimits::from_mib(16_384, 8_192);
        assert_eq!(limits.context_tokens(), 16_384);
        assert_eq!(limits.kv_budget_bytes(), 8_192 * 1024 * 1024);
    }

    #[test]
    fn external_chat_template_is_accepted_when_config_has_none() {
        let root =
            std::env::temp_dir().join(format!("metallix-chat-template-{}", std::process::id()));
        fs::create_dir_all(&root).expect("template test directory");
        fs::write(root.join("tokenizer_config.json"), b"{}").expect("tokenizer config");
        fs::write(root.join("chat_template.jinja"), b"{{ messages }}").expect("external template");

        assert_eq!(
            load_template(&root).expect("external template"),
            "{{ messages }}"
        );
        fs::remove_dir_all(root).expect("remove template test directory");
    }

    #[test]
    fn template_sources_reject_oversized_invalid_utf8_and_nonregular_files() {
        let root = std::env::temp_dir().join(format!(
            "metallix-template-source-limits-{}",
            std::process::id()
        ));
        fs::create_dir_all(&root).expect("template test directory");
        let config = root.join("tokenizer_config.json");
        let external = root.join("chat_template.jinja");
        let oversized = vec![b'x'; MAX_CHAT_TEMPLATE_BYTES + 1];

        fs::write(&config, &oversized).expect("oversized tokenizer config");
        assert!(
            load_template(&root)
                .expect_err("oversized tokenizer config must fail")
                .contains("byte limit")
        );
        fs::write(&config, [0xff]).expect("invalid UTF-8 tokenizer config");
        assert!(
            load_template(&root)
                .expect_err("invalid UTF-8 tokenizer config must fail")
                .contains("could not be read")
        );
        fs::remove_file(&config).expect("remove tokenizer config file");
        fs::create_dir(&config).expect("nonregular tokenizer config");
        assert!(
            load_template(&root)
                .expect_err("nonregular tokenizer config must fail")
                .contains("readable regular file")
        );
        fs::remove_dir(&config).expect("remove tokenizer config directory");

        fs::write(&config, b"{}").expect("fallback tokenizer config");
        fs::write(&external, &oversized).expect("oversized external template");
        assert!(
            load_template(&root)
                .expect_err("oversized external template must fail")
                .contains("byte limit")
        );
        fs::write(&external, [0xff]).expect("invalid UTF-8 external template");
        assert!(
            load_template(&root)
                .expect_err("invalid UTF-8 external template must fail")
                .contains("could not be read")
        );
        fs::remove_file(&external).expect("remove external template file");
        fs::create_dir(&external).expect("nonregular external template");
        assert!(
            load_template(&root)
                .expect_err("nonregular external template must fail")
                .contains("readable regular file")
        );

        fs::remove_dir_all(root).expect("remove template test directory");
    }

    #[test]
    fn generation_template_renders_one_nonthinking_user_message_before_weights() {
        let root = std::env::temp_dir().join(format!(
            "metallix-generation-template-{}",
            std::process::id()
        ));
        fs::create_dir_all(&root).expect("template test directory");
        fs::write(
            root.join("tokenizer_config.json"),
            serde_json::json!({"chat_template": TEMPLATE}).to_string(),
        )
        .expect("tokenizer config");
        let rendered =
            render_generation_prompt(&root, "hello").expect("bounded generation template render");
        assert_eq!(rendered.template_sha256, TEMPLATE_SHA256);
        assert_eq!(
            rendered.rendered,
            "<|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        );
        fs::remove_dir_all(root).expect("remove template test directory");
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
                "<tools>\n{\"name\":\"ping\"}\n</tools>\n\n",
                "For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n",
                "<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call><|im_end|>\n",
                "<|im_start|>user\nread README<|im_end|>\n",
                "<|im_start|>assistant\n<tool_call>\n{\"name\": \"read_file\", \"arguments\": {\"path\":\"README.md\"}}\n</tool_call><|im_end|>\n",
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
        let template = parse_template(
            concat!(
                "{% for message in messages %}{{ message.get('role') }}:{{ message.get('content') }};",
                "{% for call in message.get('tool_calls') or [] %}{{ call.function.name }}={{ call.function.arguments.path }};{% endfor %}",
                "{% endfor %}"
            )
            .to_owned(),
        )
        .expect("DeepSeek-shaped template parses");
        let messages = serde_json::to_value([ChatMessage {
            role: ChatRole::Assistant,
            content: String::new(),
            reasoning_content: None,
            tool_calls: vec![ChatToolCall {
                name: String::from("read_file"),
                arguments: serde_json::json!({"path": "README.md"}),
            }],
            tool_call_id: None,
            name: None,
        }])
        .expect("DeepSeek-shaped messages");
        let rendered = template
            .get_template("qwen_chat")
            .expect("template name")
            .render(context! { messages => messages, tools => Vec::<Value>::new() })
            .expect("DeepSeek-shaped context renders");
        assert_eq!(rendered, "assistant:;read_file=README.md;");
    }
}
