//! Resident Qwen chat generation over the checkpoint's own Jinja template.
//!
//! This is deliberately a single-session, single-sequence core.  It retains
//! model weights between calls but creates fresh KV state for each rendered
//! conversation.  The qualified dense forward cap remains explicit.

use std::{
    fs,
    path::Path,
    time::{Duration, Instant},
};

use minijinja::{Environment, context};
use minijinja_contrib::pycompat::unknown_method_callback;
use qwen::metal::Qwen3MlxWeights;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::qwen_tokenizer::QwenTokenizer;

const MAX_CHAT_TEMPLATE_BYTES: usize = 1024 * 1024;
const MAX_CHAT_RENDERED_BYTES: usize = 1024 * 1024;
const MAX_CHAT_INPUT_BYTES: usize = 1024 * 1024;
const MAX_CHAT_MESSAGES: usize = 256;
const MAX_CHAT_TOOLS: usize = 64;
const TEMPLATE_FUEL: u64 = 100_000;

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
#[derive(Clone, Debug, Deserialize, Serialize)]
pub(crate) struct ChatToolCall {
    pub name: String,
    pub arguments: Value,
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

/// One chat turn. Tool schemas remain JSON because their externally defined
/// schema is a caller boundary, while message history stays typed.
#[derive(Clone, Copy, Debug)]
pub(crate) struct ChatRequest<'a> {
    pub(crate) messages: &'a [ChatMessage],
    pub(crate) tools: &'a [Value],
    pub(crate) max_tokens: u32,
    pub(crate) enable_thinking: bool,
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

/// A resident model and tokenizer for serial Qwen chat turns.
pub(crate) struct ChatSession {
    weights: Qwen3MlxWeights,
    tokenizer: QwenTokenizer,
    template: Environment<'static>,
    eos_token_id: i32,
    vocabulary_size: usize,
    context_limit: usize,
    load_ms: f64,
}

#[derive(serde::Deserialize)]
struct ChatConfig {
    eos_token_id: i32,
    vocab_size: usize,
    max_position_embeddings: usize,
}

impl ChatSession {
    /// Loads and prepares a checkpoint once, including its template and tokenizer.
    pub(crate) fn load(model: &Path) -> Result<Self, String> {
        let started = Instant::now();
        let config = load_config(model)?;
        let tokenizer = QwenTokenizer::load(model)?;
        tokenizer.check_model_vocabulary(config.vocab_size, config.eos_token_id)?;
        let template_source = load_template(model)?;
        let template = parse_template(template_source)?;

        let mut weights = Qwen3MlxWeights::load(model).map_err(|error| error.to_string())?;
        weights
            .prepare_float32()
            .map_err(|error| error.to_string())?;
        let context_limit = config
            .max_position_embeddings
            .min(qwen::forward::MAX_DENSE_DEBUG_TOKENS);
        if context_limit == 0 {
            return Err(String::from("model chat context limit is zero"));
        }
        Ok(Self {
            weights,
            tokenizer,
            template,
            eos_token_id: config.eos_token_id,
            vocabulary_size: config.vocab_size,
            context_limit,
            load_ms: elapsed_ms(started.elapsed()),
        })
    }

    /// The one-time checkpoint, tokenizer, and template preparation duration.
    #[must_use]
    pub(crate) const fn load_ms(&self) -> f64 {
        self.load_ms
    }

    /// Generates one complete chat turn and streams only cumulative-decoder text deltas.
    pub(crate) fn generate(
        &mut self,
        request: ChatRequest<'_>,
        on_token: &mut dyn FnMut(&str) -> Result<(), String>,
    ) -> Result<ChatGeneration, String> {
        validate_request(request)?;
        let turn_started = Instant::now();
        let render_started = Instant::now();
        let prompt = self.render(request)?;
        let render_ms = elapsed_ms(render_started.elapsed());
        let input_ids = self.tokenizer.encode_prompt(&prompt)?;
        validate_ids(&input_ids, self.eos_token_id, self.vocabulary_size)?;
        let total_tokens = input_ids
            .len()
            .checked_add(request.max_tokens as usize)
            .ok_or_else(|| String::from("chat prompt plus generation budget overflows"))?;
        if total_tokens > self.context_limit {
            return Err(format!(
                "chat requires prompt_tokens + max_tokens <= {}; received {} + {} = {total_tokens}",
                self.context_limit,
                input_ids.len(),
                request.max_tokens,
            ));
        }

        let mut executor = self.weights.executor();
        let prefill_started = Instant::now();
        let mut logits = executor
            .prefill_last_logits(&input_ids)
            .map_err(|error| error.to_string())?;
        let prefill_ms = elapsed_ms(prefill_started.elapsed());
        let mut generated = Vec::with_capacity(request.max_tokens as usize);
        let mut emitted = String::new();
        let mut text_decoder = QwenTokenizer::generated_decoder();
        let mut decode_ms = Vec::new();
        let mut ttft = None;
        let mut finish_reason = ChatFinishReason::Length;

        for step in 0..request.max_tokens {
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
                let decode_started = Instant::now();
                logits = executor
                    .decode_last_logits(token)
                    .map_err(|error| error.to_string())?;
                decode_ms.push(elapsed_ms(decode_started.elapsed()));
            }
        }

        let visible_generated = generated
            .strip_suffix(&[self.eos_token_id])
            .unwrap_or(&generated);
        let text = self.tokenizer.decode_generated(visible_generated)?;
        let remaining = text.strip_prefix(&emitted).ok_or_else(|| {
            String::from("incremental tokenizer decoder diverged from complete generated text")
        })?;
        emit_delta(remaining, &mut emitted, on_token, &mut ttft, turn_started)?;
        let decode_total_ms = decode_ms.iter().sum();
        let generated_tokens = generated.len();
        Ok(ChatGeneration {
            text,
            generated_token_ids: generated,
            finish_reason,
            metrics: ChatGenerationMetrics {
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
        let rendered = self
            .template
            .get_template("qwen_chat")
            .map_err(|error| format!("local chat template is unavailable: {error}"))?
            .render(context! {
                messages => request.messages,
                tools => request.tools,
                add_generation_prompt => true,
                enable_thinking => request.enable_thinking,
            })
            .map_err(|error| format!("local chat template could not be rendered: {error}"))?;
        if rendered.len() > MAX_CHAT_RENDERED_BYTES {
            return Err(format!(
                "rendered chat prompt exceeds the {MAX_CHAT_RENDERED_BYTES}-byte limit"
            ));
        }
        Ok(rendered)
    }
}

fn load_config(model: &Path) -> Result<ChatConfig, String> {
    let path = model.join("config.json");
    let raw = fs::read_to_string(&path)
        .map_err(|_| String::from("local model config.json could not be read"))?;
    serde_json::from_str(&raw)
        .map_err(|_| String::from("local model config.json could not be parsed"))
}

fn load_template(model: &Path) -> Result<String, String> {
    let path = model.join("tokenizer_config.json");
    let metadata = path
        .metadata()
        .map_err(|_| String::from("local tokenizer_config.json must be a readable regular file"))?;
    if !metadata.is_file() || metadata.len() > MAX_CHAT_TEMPLATE_BYTES as u64 {
        return Err(format!(
            "local tokenizer_config.json exceeds the {MAX_CHAT_TEMPLATE_BYTES}-byte limit or is not regular"
        ));
    }
    let raw = fs::read_to_string(&path)
        .map_err(|_| String::from("local tokenizer_config.json could not be read"))?;
    if raw.len() > MAX_CHAT_TEMPLATE_BYTES {
        return Err(format!(
            "local tokenizer_config.json exceeds the {MAX_CHAT_TEMPLATE_BYTES}-byte limit"
        ));
    }
    let config: Value = serde_json::from_str(&raw)
        .map_err(|_| String::from("local tokenizer_config.json could not be parsed"))?;
    config
        .get("chat_template")
        .and_then(Value::as_str)
        .filter(|template| !template.is_empty())
        .map(str::to_owned)
        .ok_or_else(|| String::from("local tokenizer_config.json has no chat_template"))
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
    use minijinja::context;
    use serde_json::{Value, json};
    use sha2::{Digest, Sha256};

    use super::{ChatMessage, ChatRole, ChatToolCall, ChatToolResult, parse_template};

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
}
