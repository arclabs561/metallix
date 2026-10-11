//! Text-only Responses protocol: request validation, message and tool
//! reconstruction, and the streamed (SSE) or JSON response for one generation.

use std::{
    collections::{HashMap, HashSet},
    io::{BufWriter, Write},
    time::Duration,
};

use chat_format::{TemplateKwargs, ToolRequirement, TurnDelta};
use serde::{Deserialize, de::IgnoredAny};
use serde_json::{Value, json};

use crate::{
    chat_cli::message,
    chat_generation::{
        ChatBackend, ChatGenerationError, ChatMessage, ChatRequest, ChatRole, ChatToolCall,
        GenerationControls, SamplingRequest, TokenLogprob,
    },
    generation_routes,
    http_transport::Connection,
};

/// One `POST /v1/responses` body. Unknown fields are rejected so a control
/// this server cannot honor never silently changes meaning. Fields that
/// common clients (the `OpenAI` SDKs, Codex) send by default but that carry no
/// generation semantics here are accepted and ignored by name.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Request {
    pub(crate) model: String,
    /// Metallix extension: "auto" (default), "on" or "off" for
    /// prompt-lookup speculative decoding.
    #[serde(default)]
    speculation: crate::chat_generation::SpeculationField,
    input: Value,
    #[serde(default)]
    instructions: Option<String>,
    #[serde(default)]
    tools: Vec<Value>,
    #[serde(default)]
    stream: bool,
    #[serde(default)]
    max_output_tokens: Option<u32>,
    #[serde(default)]
    previous_response_id: Option<String>,
    #[serde(default)]
    tool_choice: Option<Value>,
    #[serde(default)]
    temperature: Option<f64>,
    #[serde(default)]
    top_p: Option<f64>,
    /// Not part of the `OpenAI` Responses schema; vLLM and llama.cpp accept it.
    #[serde(default)]
    seed: Option<u64>,
    /// Metallix extension, as in vLLM: generate past end-of-turn up to the
    /// output limit, for equal-length benchmark runs.
    #[serde(default)]
    ignore_eos: bool,
    #[serde(default)]
    store: Option<bool>,
    #[serde(default)]
    top_logprobs: Option<u8>,
    #[serde(default)]
    include: Option<Vec<String>>,
    #[serde(default)]
    reasoning: Option<Reasoning>,
    /// Metallix extension, as vLLM accepts it on Responses: extra
    /// chat-template variables; see [`thinking_and_template_kwargs`].
    #[serde(default)]
    chat_template_kwargs: Option<Value>,
    #[serde(default)]
    text: Option<TextConfig>,
    #[serde(default)]
    truncation: Option<String>,
    #[serde(default)]
    background: Option<bool>,
    #[serde(default)]
    max_tool_calls: Option<u32>,
    // Accepted and ignored: attribution, routing, caching and transport hints.
    // `parallel_tool_calls` is ignored because a turn may already carry
    // several calls and one call per turn is the model's choice, not ours.
    #[serde(default, rename = "parallel_tool_calls")]
    _parallel_tool_calls: Option<IgnoredAny>,
    #[serde(default, rename = "user")]
    _user: Option<IgnoredAny>,
    /// Echoed on the response, as the Responses API does.
    #[serde(default)]
    metadata: Option<Value>,
    #[serde(default, rename = "client_metadata")]
    _client_metadata: Option<IgnoredAny>,
    #[serde(default, rename = "safety_identifier")]
    _safety_identifier: Option<IgnoredAny>,
    #[serde(default, rename = "service_tier")]
    _service_tier: Option<IgnoredAny>,
    #[serde(default, rename = "prompt_cache_key")]
    _prompt_cache_key: Option<IgnoredAny>,
    #[serde(default, rename = "prompt_cache_retention")]
    _prompt_cache_retention: Option<IgnoredAny>,
    #[serde(default, rename = "stream_options")]
    _stream_options: Option<IgnoredAny>,
    #[serde(default, rename = "access_programs")]
    _access_programs: Option<IgnoredAny>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Reasoning {
    #[serde(default)]
    effort: Option<String>,
    // Qwen produces raw reasoning text, never a summary or a carried context.
    #[serde(default, rename = "summary")]
    _summary: Option<IgnoredAny>,
    #[serde(default, rename = "generate_summary")]
    _generate_summary: Option<IgnoredAny>,
    #[serde(default, rename = "context")]
    _context: Option<IgnoredAny>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct TextConfig {
    #[serde(default)]
    format: Option<TextFormat>,
    /// Ignored: the model has no verbosity control.
    #[serde(default, rename = "verbosity")]
    _verbosity: Option<IgnoredAny>,
}

#[derive(Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
enum TextFormat {
    Text,
    JsonObject,
    JsonSchema {
        schema: Value,
        #[serde(default, rename = "name")]
        _name: Option<IgnoredAny>,
        #[serde(default, rename = "description")]
        _description: Option<IgnoredAny>,
        /// Ignored: the grammar mask always enforces the schema, so
        /// `strict: false` receives strict output.
        #[serde(default, rename = "strict")]
        _strict: Option<IgnoredAny>,
    },
}

const LOGPROBS_INCLUDE: &str = "message.output_text.logprobs";
/// Every `include` value the `OpenAI` schema defines. Only the logprobs value
/// changes output; the rest name tools or encrypted state this server never
/// produces, so they are accepted with nothing to include.
const KNOWN_INCLUDES: [&str; 8] = [
    "file_search_call.results",
    "web_search_call.results",
    "web_search_call.action.sources",
    "message.input_image.image_url",
    "computer_call_output.output.image_url",
    "code_interpreter_call.outputs",
    "reasoning.encrypted_content",
    LOGPROBS_INCLUDE,
];

/// Maps an `OpenAI` reasoning effort onto Qwen's binary `enable_thinking`:
/// absent, `none` and `minimal` keep thinking off; `low` through `max` turn it
/// on and are passed to the template as `reasoning_effort`, which the stock
/// Qwen3 template ignores. An unknown effort is returned as the error.
pub(crate) fn thinking_for_effort(effort: Option<&str>) -> Result<(bool, Option<String>), &str> {
    match effort {
        None | Some("none" | "minimal") => Ok((false, None)),
        Some(effort @ ("low" | "medium" | "high" | "xhigh" | "max")) => {
            Ok((true, Some(effort.to_owned())))
        }
        Some(other) => Err(other),
    }
}

/// The thinking switch, reasoning effort and template variables from a
/// reasoning-effort field (named `effort_field` in errors) and a
/// `chat_template_kwargs` object. vLLM and `SGLang` clients set
/// `enable_thinking`, and `SGLang` ones `reasoning_effort`, inside the
/// kwargs; both map to the same switch as the effort field, and values that
/// disagree are refused rather than one winning silently.
pub(crate) fn thinking_and_template_kwargs(
    effort_field: &str,
    effort: Option<&str>,
    kwargs: Option<&Value>,
) -> Result<(bool, Option<String>, TemplateKwargs), String> {
    let Some(kwargs) = kwargs else {
        let (thinking, effort) = thinking_for_effort(effort)
            .map_err(|effort| format!("unsupported {effort_field} {effort:?}"))?;
        return Ok((thinking, effort, TemplateKwargs::default()));
    };
    let mut kwargs = kwargs.clone();
    let object = kwargs
        .as_object_mut()
        .ok_or("chat_template_kwargs must be an object")?;
    if object.len() > chat_format::MAX_TEMPLATE_KWARGS {
        return Err(format!(
            "chat_template_kwargs has more than {} entries",
            chat_format::MAX_TEMPLATE_KWARGS
        ));
    }
    let kwarg_effort = match object.remove("reasoning_effort") {
        None => None,
        Some(Value::String(effort)) => Some(effort),
        Some(_) => return Err("chat_template_kwargs.reasoning_effort must be a string".into()),
    };
    let kwarg_thinking = match object.remove("enable_thinking") {
        None => None,
        Some(Value::Bool(thinking)) => Some(thinking),
        Some(_) => return Err("chat_template_kwargs.enable_thinking must be a boolean".into()),
    };
    let effort = match (effort, kwarg_effort.as_deref()) {
        (Some(field), Some(kwarg)) if field != kwarg => {
            return Err(format!(
                "{effort_field} and chat_template_kwargs.reasoning_effort disagree"
            ));
        }
        (field, kwarg) => field.or(kwarg),
    };
    let (mut thinking, effort_out) = thinking_for_effort(effort)
        .map_err(|effort| format!("unsupported {effort_field} {effort:?}"))?;
    if let Some(kwarg) = kwarg_thinking {
        if effort.is_some() && kwarg != thinking {
            return Err(format!(
                "chat_template_kwargs.enable_thinking disagrees with {effort_field}"
            ));
        }
        thinking = kwarg;
    }
    Ok((thinking, effort_out, TemplateKwargs::parse(&kwargs)?))
}

/// Names `controls` sets that this model's chat template does not read; they
/// reach no prompt, so the reply lists them as
/// `metallix.ignored_template_kwargs`. Nothing is added when every name is
/// read or the backend cannot say.
pub(crate) fn note_ignored_template_kwargs(
    value: &mut Value,
    controls: &GenerationControls,
    session: &dyn ChatBackend,
) {
    let ignored: Vec<&str> = controls
        .template_kwargs
        .iter()
        .map(|(name, _)| name)
        .filter(|name| session.template_reads(name) == Some(false))
        .collect();
    if ignored.is_empty() {
        return;
    }
    let Some(object) = value.as_object_mut() else {
        return;
    };
    let block = object.entry("metallix").or_insert_with(|| json!({}));
    if let Some(block) = block.as_object_mut() {
        block.insert("ignored_template_kwargs".into(), json!(ignored));
    }
}

/// Maps the request's generation fields onto the protocol-neutral controls.
///
/// `reasoning.effort` maps through [`thinking_for_effort`]. As in vLLM, an
/// omitted `temperature` or `top_p` takes the checkpoint's
/// `generation_config.json` default (and its `top_k`, which has no Responses
/// field), greedy if it has none; an explicit temperature of zero is greedy.
/// The response's `metallix.sampling` records the policy used and which
/// defaults applied.
/// `tool_choice`: `auto` leaves the calls to the model, `required` demands
/// one, and `{"type": "function", "name": ..}` names the tool. `none` is
/// refused because these tools would still reach the prompt.
fn tool_requirement(choice: Option<&Value>) -> Result<ToolRequirement, String> {
    match choice {
        None => Ok(ToolRequirement::Auto),
        Some(choice) if choice == "auto" => Ok(ToolRequirement::Auto),
        Some(choice) if choice == "required" => Ok(ToolRequirement::Required),
        Some(choice) => match (choice["type"].as_str(), choice["name"].as_str()) {
            (Some("function"), Some(name)) => Ok(ToolRequirement::Named(name.to_owned())),
            _ => Err("tool_choice must be auto, required or a named function".into()),
        },
    }
}

pub(crate) fn controls(request: &Request) -> Result<GenerationControls, String> {
    if request.background == Some(true) {
        return Err("background responses are unsupported".into());
    }
    if request.max_tool_calls.is_some() {
        return Err("max_tool_calls is unsupported".into());
    }
    if request
        .truncation
        .as_deref()
        .is_some_and(|value| value != "disabled")
    {
        return Err("only truncation=disabled is supported; send input that fits".into());
    }
    let mut logprobs = false;
    for value in request.include.iter().flatten() {
        if !KNOWN_INCLUDES.contains(&value.as_str()) {
            return Err(format!("unsupported include value {value:?}"));
        }
        logprobs |= value == LOGPROBS_INCLUDE;
    }
    let (enable_thinking, reasoning_effort, template_kwargs) = thinking_and_template_kwargs(
        "reasoning.effort",
        request
            .reasoning
            .as_ref()
            .and_then(|reasoning| reasoning.effort.as_deref()),
        request.chat_template_kwargs.as_ref(),
    )?;
    let json_schema = match request.text.as_ref().and_then(|text| text.format.as_ref()) {
        None | Some(TextFormat::Text) => None,
        Some(TextFormat::JsonObject) => Some(json!({"type":"object"})),
        Some(TextFormat::JsonSchema { schema, .. }) => {
            if !schema.is_object() {
                return Err("text.format.schema must be a JSON Schema object".into());
            }
            Some(schema.clone())
        }
    };
    let controls = GenerationControls {
        max_tokens: request.max_output_tokens,
        sampling: SamplingRequest {
            temperature: request.temperature,
            top_p: request.top_p,
            top_k: None,
            seed: request.seed,
        },
        top_logprobs: (logprobs || request.top_logprobs.is_some())
            .then(|| request.top_logprobs.unwrap_or(0)),
        enable_thinking,
        reasoning_effort,
        json_schema,
        ignore_eos: request.ignore_eos,
        speculation: request.speculation.into(),
        tool_requirement: tool_requirement(request.tool_choice.as_ref())?,
        strict_tools: strict_tools(has_function_tools(request), &request.tools, |tool| {
            tool["strict"] == Value::Bool(true)
        }),
        template_kwargs,
    };
    controls.validate(has_function_tools(request))?;
    Ok(controls)
}

fn input_text(content: &Value) -> Result<String, String> {
    if let Some(text) = content.as_str() {
        return Ok(text.into());
    }
    let parts = content
        .as_array()
        .ok_or("message content must be text or an array of text parts")?;
    let mut text = String::new();
    for part in parts {
        if !matches!(part["type"].as_str(), Some("input_text" | "output_text")) {
            return Err("only text input parts are supported".into());
        }
        text.push_str(part["text"].as_str().ok_or("text part requires text")?);
    }
    Ok(text)
}

/// Validates the whole request, including its generation controls, and
/// rebuilds the typed message history.
pub(crate) fn messages(request: &Request) -> Result<Vec<ChatMessage>, String> {
    controls(request)?;
    if request.store == Some(true) {
        return Err("response storage is unsupported; use store=false".into());
    }
    if request.previous_response_id.is_some() {
        return Err("send complete input history; previous_response_id is not supported".into());
    }
    let mut messages = Vec::new();
    let mut pending_calls = HashMap::new();
    let mut seen_call_ids = HashSet::new();
    if let Some(instructions) = &request.instructions {
        messages.push(message(ChatRole::System, instructions.clone()));
    }
    if let Some(text) = request.input.as_str() {
        messages.push(message(ChatRole::User, text.into()));
        return Ok(messages);
    }
    for item in request
        .input
        .as_array()
        .ok_or("input must be text or an item array")?
    {
        match item["type"].as_str().unwrap_or("message") {
            "message" => {
                let role = match item["role"].as_str() {
                    Some("system" | "developer") => ChatRole::System,
                    Some("user") => ChatRole::User,
                    Some("assistant") => ChatRole::Assistant,
                    _ => return Err("unsupported message role".into()),
                };
                messages.push(message(role, input_text(&item["content"])?));
            }
            "function_call" => {
                let mut call = message(ChatRole::Assistant, String::new());
                let arguments = item["arguments"]
                    .as_str()
                    .ok_or("function arguments must be a JSON string")?;
                let arguments: Value =
                    serde_json::from_str(arguments).map_err(|e| e.to_string())?;
                if !arguments.is_object() {
                    return Err("function arguments must encode an object".into());
                }
                let name = item["name"].as_str().ok_or("function requires name")?;
                let call_id = item["call_id"]
                    .as_str()
                    .ok_or("function requires call_id")?;
                if call_id.is_empty() || !seen_call_ids.insert(call_id) {
                    return Err("function call IDs must be nonempty and unique".into());
                }
                pending_calls.insert(call_id, name);
                call.tool_calls.push(ChatToolCall {
                    name: name.into(),
                    arguments,
                });
                call.tool_call_id = Some(call_id.into());
                messages.push(call);
            }
            "function_call_output" => {
                let mut output = message(ChatRole::Tool, input_text(&item["output"])?);
                let call_id = item["call_id"]
                    .as_str()
                    .ok_or("tool output requires call_id")?;
                output.name = Some(
                    pending_calls
                        .remove(call_id)
                        .ok_or("tool output has no matching pending call")?
                        .into(),
                );
                output.tool_call_id = Some(call_id.into());
                messages.push(output);
            }
            // Earlier reasoning is not replayed: the Qwen template drops
            // thinking from history, and Codex echoes these items back.
            "reasoning" => {}
            _ => return Err("unsupported Responses input item".into()),
        }
    }
    if messages.is_empty() {
        return Err("input must contain at least one message".into());
    }
    Ok(messages)
}

/// `OpenAI` hosted tools: `OpenAI` runs them server-side, and mx has nothing to
/// run them with, so they are left out of the prompt and reported.
const HOSTED_TOOLS: [&str; 8] = [
    "web_search",
    "web_search_2025_08_26",
    "web_search_preview",
    "web_search_preview_2025_03_11",
    "file_search",
    "code_interpreter",
    "image_generation",
    "mcp",
];

fn is_hosted_tool(tool: &Value) -> bool {
    tool["type"]
        .as_str()
        .is_some_and(|kind| HOSTED_TOOLS.contains(&kind))
}

/// Whether any declared tool is one the model is offered (not hosted).
fn has_function_tools(request: &Request) -> bool {
    request.tools.iter().any(|tool| !is_hosted_tool(tool))
}

/// The hosted tools [`tools`] leaves out, by name or else by type.
pub(crate) fn dropped_tools(request: &Request) -> Vec<String> {
    request
        .tools
        .iter()
        .filter(|tool| is_hosted_tool(tool))
        .map(|tool| {
            tool["name"]
                .as_str()
                .or_else(|| tool["type"].as_str())
                .unwrap_or_default()
                .to_owned()
        })
        .collect()
}

/// The function tools in the template's shape; hosted tools are left out
/// (see [`dropped_tools`]) and any other tool type is refused.
pub(crate) fn tools(request: &Request) -> Result<Vec<Value>, String> {
    let mut names = HashSet::new();
    request.tools.iter().filter(|tool| !is_hosted_tool(tool)).map(|tool| {
        if tool["type"] != "function" || !tool["name"].is_string() || !tool["parameters"].is_object() {
            return Err("only function tools with name and parameters are supported".into());
        }
        let name = tool["name"].as_str().expect("checked string");
        if name.is_empty() || !names.insert(name) {
            return Err("function names must be nonempty and unique".into());
        }
        chat_format::validator(&tool["parameters"])?;
        Ok(json!({"type":"function","function":{"name":tool["name"],"description":tool.get("description").cloned().unwrap_or(json!("")),"parameters":tool["parameters"]}}))
    }).collect()
}

fn event(writer: &mut dyn Write, sequence: &mut u64, mut value: Value) -> Result<(), String> {
    value["sequence_number"] = json!(*sequence);
    *sequence += 1;
    writeln!(
        writer,
        "event: {}\ndata: {}\n",
        value["type"].as_str().ok_or("event type required")?,
        value
    )
    .map_err(|e| e.to_string())?;
    writer.flush().map_err(|e| e.to_string())
}

/// Adds the request id to the response's `metallix` block, creating the
/// block on routes that have none.
pub(crate) fn echo_request_id(value: &mut Value, request_id: Option<&str>) {
    let Some(request_id) = request_id else { return };
    let Some(object) = value.as_object_mut() else {
        return;
    };
    let block = object.entry("metallix").or_insert_with(|| json!({}));
    if let Some(block) = block.as_object_mut() {
        block.insert("request_id".into(), json!(request_id));
    }
}

/// Records the `GenAI` usage attributes on the current request span.
pub(crate) fn record_usage(generated: &crate::chat_generation::ChatGeneration) {
    let span = tracing::Span::current();
    let finish_reason = match generated.finish_reason {
        crate::chat_generation::ChatFinishReason::Eos => "eos",
        crate::chat_generation::ChatFinishReason::Length => "length",
    };
    span.record("gen_ai.response.finish_reason", finish_reason);
    span.record("gen_ai.usage.input_tokens", generated.metrics.prompt_tokens);
    span.record(
        "gen_ai.usage.output_tokens",
        generated.generated_token_ids.len(),
    );
    tracing::info!(finish_reason, "generation finished");
}

pub(crate) fn json_response(mut connection: Connection, status: u16, value: &Value) {
    connection.begin_response();
    let request_id = connection.request_id_header();
    let mut writer = BufWriter::new(connection);
    let body = value.to_string();
    let reason = match status {
        200 => "OK",
        400 => "Bad Request",
        501 => "Not Implemented",
        404 => "Not Found",
        408 => "Request Timeout",
        413 => "Content Too Large",
        417 => "Expectation Failed",
        431 => "Request Header Fields Too Large",
        503 => "Service Unavailable",
        _ => "Error",
    };
    let _ = write!(
        writer,
        "HTTP/1.1 {status} {reason}\r\nContent-Type: application/json\r\nContent-Length: {}\r\n{request_id}Connection: close\r\n\r\n{body}",
        body.len()
    );
    let _ = writer.flush();
}

/// The SSE half of one response. The HTTP head and `response.created` are
/// written with the first output, so a failure before any token (context
/// overflow, schema compilation, an expired budget) still answers with an
/// ordinary HTTP error status instead of a `200` stream.
struct EventStream<'a> {
    pending: Option<Connection>,
    writer: Option<BufWriter<Connection>>,
    sequence: u64,
    id: &'a str,
    /// The in-progress `Response` that `response.created` and a failure carry.
    envelope: Value,
    /// Whether the reasoning item streamed live.
    live_reasoning: bool,
    /// The message item's output index, once its text began to stream.
    live_message: Option<usize>,
    keepalive: crate::sse::Keepalive,
}

impl EventStream<'_> {
    fn open(&mut self) -> Result<&mut BufWriter<Connection>, String> {
        if self.writer.is_none() {
            let mut connection = self
                .pending
                .take()
                .ok_or("response connection is unavailable")?;
            connection.begin_response();
            let request_id = connection.request_id_header();
            let mut writer = BufWriter::new(connection);
            write!(writer,"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\n{request_id}Connection: close\r\n\r\n").map_err(|e| e.to_string())?;
            event(
                &mut writer,
                &mut self.sequence,
                json!({"type":"response.created","response":self.envelope}),
            )?;
            self.writer = Some(writer);
        }
        Ok(self.writer.as_mut().expect("opened stream writer"))
    }

    fn emit(&mut self, value: Value) -> Result<(), String> {
        self.open()?;
        let writer = self.writer.as_mut().expect("opened stream writer");
        event(writer, &mut self.sequence, value)
    }

    /// Writes an SSE comment while output is held back, so a disconnect is
    /// still noticed.
    fn keepalive(&mut self) -> Result<(), String> {
        self.keepalive.held(&mut self.pending, &mut self.writer)
    }

    /// Streams one delta, opening the reasoning item (always output 0) or
    /// the message item (after any reasoning) on its first delta.
    fn delta(&mut self, delta: TurnDelta) -> Result<(), String> {
        match delta {
            TurnDelta::Held => self.keepalive(),
            TurnDelta::Reasoning(reasoning) => {
                let item_id = format!("rs_{}", self.id);
                if !self.live_reasoning {
                    self.live_reasoning = true;
                    self.emit(json!({"type":"response.output_item.added","output_index":0,"item":{"id":item_id,"type":"reasoning","summary":[],"content":[]}}))?;
                }
                self.emit(json!({"type":"response.reasoning_text.delta","item_id":item_id,"output_index":0,"content_index":0,"delta":reasoning}))
            }
            TurnDelta::Text(text) => {
                let item_id = format!("msg_{}", self.id);
                let index = if let Some(index) = self.live_message {
                    index
                } else {
                    let index = usize::from(self.live_reasoning);
                    self.live_message = Some(index);
                    self.emit(json!({"type":"response.output_item.added","output_index":index,"item":{"id":item_id,"type":"message","role":"assistant","status":"in_progress","content":[]}}))?;
                    self.emit(json!({"type":"response.content_part.added","item_id":item_id,"output_index":index,"content_index":0,"part":{"type":"output_text","text":"","annotations":[],"logprobs":[]}}))?;
                    index
                };
                // Per-token logprobs arrive on `response.output_text.done`.
                self.emit(json!({"type":"response.output_text.delta","item_id":item_id,"output_index":index,"content_index":0,"delta":text,"logprobs":[]}))
            }
        }
    }
}

fn error_response(connection: Connection, error: &ChatGenerationError) {
    match error {
        ChatGenerationError::DeadlineExceeded => generation_routes::error_response(
            connection,
            408,
            Some("generation_timeout"),
            "generation time budget exceeded",
        ),
        ChatGenerationError::Message(message) => {
            generation_routes::error_response(connection, 400, None, message);
        }
    }
}

#[allow(
    clippy::too_many_lines,
    reason = "one ordered response event lifecycle"
)]
pub(crate) fn respond(
    request: Connection,
    parsed: &Request,
    messages: &[ChatMessage],
    tools: &[Value],
    session: &mut dyn ChatBackend,
    id: &str,
    generation_timeout: Duration,
) -> Result<(), String> {
    let controls = match controls(parsed) {
        Ok(controls) => controls,
        Err(error) => {
            generation_routes::error_response(request, 400, None, &error);
            return Ok(());
        }
    };
    if !parsed.stream {
        respond_json(
            request,
            parsed,
            &controls,
            messages,
            tools,
            session,
            id,
            generation_timeout,
        );
        return Ok(());
    }
    let request_id = request.request_id().map(str::to_owned);
    let cache_salt = request.cache_salt().map(str::to_owned);
    // Reasoning and text stream as they settle; the session holds back
    // markup, and function calls arrive whole once the turn ends.
    let mut stream = EventStream {
        pending: Some(request),
        writer: None,
        sequence: 0,
        id,
        envelope: {
            let mut envelope = envelope(parsed, id, unix_now());
            note_ignored_template_kwargs(&mut envelope, &controls, &*session);
            envelope
        },
        live_reasoning: false,
        live_message: None,
        keepalive: crate::sse::Keepalive::new(),
    };
    let generated = session.generate_with_timeout(
        ChatRequest {
            cache_salt: cache_salt.as_deref(),
            ..controls.request(messages, tools)
        },
        generation_timeout,
        &mut |delta| stream.delta(delta),
    );
    let generated = match generated {
        Ok(generated) => generated,
        Err(error) => {
            if let Some(connection) = stream.pending.take() {
                error_response(connection, &error);
                return Ok(());
            }
            let (code, message) = match error {
                ChatGenerationError::DeadlineExceeded => (
                    "generation_timeout",
                    "generation time budget exceeded".into(),
                ),
                ChatGenerationError::Message(message) => ("generation_failed", message),
            };
            let failure = failed(&stream.envelope, code, &message);
            return stream.emit(failure);
        }
    };
    record_usage(&generated);
    let response = match response_value(parsed, &controls, &generated, id) {
        Ok(mut response) => {
            // One creation time across the stream's events.
            response["created_at"] = stream.envelope["created_at"].clone();
            echo_request_id(&mut response, request_id.as_deref());
            note_ignored_template_kwargs(&mut response, &controls, &*session);
            response
        }
        Err(error) => {
            let failure = failed(&stream.envelope, "invalid_model_output", &error);
            return stream.emit(failure);
        }
    };
    for (index, item) in response["output"]
        .as_array()
        .ok_or("output array")?
        .iter()
        .enumerate()
    {
        if item["type"] == "function_call" {
            let mut started = item.clone();
            started["arguments"] = json!("");
            started["status"] = json!("in_progress");
            stream.emit(
                json!({"type":"response.output_item.added","output_index":index,"item":started}),
            )?;
            stream.emit(
                json!({"type":"response.function_call_arguments.delta","item_id":item["id"],"output_index":index,"delta":item["arguments"]}),
            )?;
            stream.emit(
                json!({"type":"response.function_call_arguments.done","item_id":item["id"],"output_index":index,"arguments":item["arguments"]}),
            )?;
        } else if item["type"] == "reasoning" {
            if stream.live_reasoning {
                stream.emit(
                    json!({"type":"response.reasoning_text.done","item_id":item["id"],"output_index":index,"content_index":0,"text":item["content"][0]["text"]}),
                )?;
            } else {
                stream.emit(
                    json!({"type":"response.output_item.added","output_index":index,"item":{"id":item["id"],"type":"reasoning","summary":[],"content":[]}}),
                )?;
            }
        } else {
            let part = &item["content"][0];
            let logprobs = part.get("logprobs").cloned().unwrap_or_else(|| json!([]));
            if stream.live_message.is_none() {
                stream.emit(
                    json!({"type":"response.output_item.added","output_index":index,"item":{"id":item["id"],"type":"message","role":"assistant","status":"in_progress","content":[]}}),
                )?;
                stream.emit(
                    json!({"type":"response.content_part.added","item_id":item["id"],"output_index":index,"content_index":0,"part":{"type":"output_text","text":"","annotations":[],"logprobs":[]}}),
                )?;
                stream.emit(
                    json!({"type":"response.output_text.delta","item_id":item["id"],"output_index":index,"content_index":0,"delta":part["text"],"logprobs":[]}),
                )?;
            }
            stream.emit(
                json!({"type":"response.output_text.done","item_id":item["id"],"output_index":index,"content_index":0,"text":part["text"],"logprobs":logprobs}),
            )?;
            stream.emit(
                json!({"type":"response.content_part.done","item_id":item["id"],"output_index":index,"content_index":0,"part":part}),
            )?;
        }
        stream
            .emit(json!({"type":"response.output_item.done","output_index":index,"item":item}))?;
    }
    let event_type = if response["status"] == "incomplete" {
        "response.incomplete"
    } else {
        "response.completed"
    };
    stream.emit(json!({"type":event_type,"response":response}))
}

#[allow(
    clippy::too_many_arguments,
    reason = "the connection, parsed request and its derived parts are one call"
)]
fn respond_json(
    mut request: Connection,
    parsed: &Request,
    controls: &GenerationControls,
    messages: &[ChatMessage],
    tools: &[Value],
    session: &mut dyn ChatBackend,
    id: &str,
    generation_timeout: Duration,
) {
    let request_id = request.request_id().map(str::to_owned);
    let cache_salt = request.cache_salt().map(str::to_owned);
    let result = session
        .generate_with_timeout(
            ChatRequest {
                cache_salt: cache_salt.as_deref(),
                ..controls.request(messages, tools)
            },
            generation_timeout,
            &mut request.stop_when_gone(),
        )
        .and_then(|generated| {
            record_usage(&generated);
            let mut response = response_value(parsed, controls, &generated, id)
                .map_err(ChatGenerationError::Message)?;
            echo_request_id(&mut response, request_id.as_deref());
            note_ignored_template_kwargs(&mut response, controls, &*session);
            Ok(response)
        });
    match result {
        Ok(response) => json_response(request, 200, &response),
        Err(error) => error_response(request, &error),
    }
}

pub(crate) fn logprobs_value(logprobs: &[TokenLogprob]) -> Value {
    serde_json::to_value(logprobs).unwrap_or_else(|_| json!([]))
}

pub(crate) use chat_format::AssistantTurn;

/// The parsed turn, checking schemas only for protocol-selected tool names.
/// This terminal validation does not constrain generation to the schema.
/// With a response format, a turn that ends without calls must also end in
/// a message matching `final_schema`; text before calls is commentary the
/// schema does not bind.
pub(crate) fn assistant_turn(
    tools: &[Value],
    strict: &[String],
    final_schema: Option<&Value>,
    generated: &crate::chat_generation::ChatGeneration,
) -> Result<AssistantTurn, String> {
    let turn = generated.turn.clone()?;
    for call in turn.calls.iter().filter(|call| strict.contains(&call.name)) {
        chat_format::check_call(tools, call)?;
    }
    if let Some(schema) = final_schema.filter(|_| turn.calls.is_empty() && turn.complete) {
        let message: Value = serde_json::from_str(turn.text.trim())
            .map_err(|_| String::from("the final message is not JSON"))?;
        if !chat_format::validator(schema)?.is_valid(&message) {
            return Err("the final message does not match the response format's schema".into());
        }
    }
    Ok(turn)
}

/// Names of the declared tools `select` picks, or none when tools are not
/// offered this turn (`tool_choice: none`). Each protocol passes its own
/// policy: the tools it constrains by grammar (an explicit `strict: true`),
/// or the tools whose calls it validates once generated.
pub(crate) fn strict_tools(
    offered: bool,
    declared: &[Value],
    select: impl Fn(&Value) -> bool,
) -> Vec<String> {
    if !offered {
        return Vec::new();
    }
    declared
        .iter()
        .filter(|tool| select(tool))
        .filter_map(|tool| tool.get("name").or_else(|| tool["function"].get("name")))
        .filter_map(Value::as_str)
        .map(str::to_owned)
        .collect()
}

/// The `Response` fields a reply carries before any output, filled from the
/// request. The `OpenAPI` document (<https://github.com/openai/openai-openapi>
/// at 8f5077ae70efcd2755a24d4df3c705de26ce84d2, schema `Response`) requires
/// `id`, `object`, `created_at`, `error`, `incomplete_details`,
/// `instructions`, `model`, `tools`, `output`, `parallel_tool_calls`,
/// `metadata`, `tool_choice`, `temperature`, `top_p` and `access_programs`;
/// the nullable ones are null here only when the request set nothing.
/// `parallel_tool_calls` is true because a turn may carry several calls
/// whatever the request asked, and `access_programs` is always null. Echoed
/// function tools get `strict: null` when the request left it out, since
/// `FunctionTool` requires the key.
fn envelope(request: &Request, id: &str, created_at: u64) -> Value {
    let tools: Vec<Value> = request
        .tools
        .iter()
        .map(|tool| {
            let mut tool = tool.clone();
            if let Some(fields) = tool.as_object_mut() {
                fields.entry("strict").or_insert(Value::Null);
            }
            tool
        })
        .collect();
    let mut envelope = json!({
        "id": id,
        "object": "response",
        "created_at": created_at,
        "status": "in_progress",
        "model": request.model,
        "output": [],
        "error": null,
        "incomplete_details": null,
        "instructions": request.instructions,
        "tools": tools,
        "tool_choice": request.tool_choice.clone().unwrap_or_else(|| json!("auto")),
        "parallel_tool_calls": true,
        "metadata": request.metadata,
        "temperature": request.temperature,
        "top_p": request.top_p,
        "access_programs": null,
        "usage": null,
    });
    let dropped = dropped_tools(request);
    if !dropped.is_empty() {
        envelope["metallix"] = json!({"dropped_tools": dropped});
    }
    envelope
}

/// `envelope` for a failed reply. `ResponseError.code` must be one of the
/// spec's `ResponseErrorCode` values (openai-openapi 8f5077ae); every
/// failure here happens on the server mid-generation (an expired budget,
/// a failed decode, output the turn parser refused), so it is
/// `server_error`. The finer metallix reason goes in `metallix_code`.
fn failed(envelope: &Value, metallix_code: &str, message: &str) -> Value {
    let mut response = envelope.clone();
    response["status"] = json!("failed");
    response["error"] =
        json!({"code": "server_error", "message": message, "metallix_code": metallix_code});
    json!({"type": "response.failed", "response": response})
}

fn unix_now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |elapsed| elapsed.as_secs())
}

fn response_value(
    request: &Request,
    controls: &GenerationControls,
    generated: &crate::chat_generation::ChatGeneration,
    id: &str,
) -> Result<Value, String> {
    let AssistantTurn {
        reasoning,
        text,
        calls,
        complete,
    } = {
        let offered = tools(request)?;
        // Responses validates every tool not marked `strict: false`; only an
        // explicit `strict: true` is also constrained by the grammar.
        let strict = strict_tools(!offered.is_empty(), &request.tools, |tool| {
            tool["strict"] != Value::Bool(false)
        });
        assistant_turn(&offered, &strict, controls.json_schema.as_ref(), generated)?
    };
    let reasoned = !reasoning.is_empty();
    let mut output = Vec::new();
    if !reasoning.is_empty() {
        output.push(json!({"id":format!("rs_{id}"),"type":"reasoning","summary":[],"content":[{"type":"reasoning_text","text":reasoning}]}));
    }
    if calls.is_empty() || !text.trim().is_empty() {
        // `OutputTextContent` requires `logprobs`; it is empty unless asked for.
        let logprobs = if controls.top_logprobs.is_some() {
            logprobs_value(&generated.logprobs)
        } else {
            json!([])
        };
        let part = json!({"type":"output_text","text":text,"annotations":[],"logprobs":logprobs});
        output.push(json!({"id":format!("msg_{id}"),"type":"message","role":"assistant","status":if complete {"completed"} else {"incomplete"},"content":[part]}));
    }
    for (index, call) in calls.into_iter().enumerate() {
        output.push(json!({"type":"function_call","id":format!("fc_{id}_{index}"),"call_id":format!("call_{id}_{index}"),"name":call.name,"arguments":call.arguments.to_string(),"status":"completed"}));
    }
    let mut response = envelope(request, id, unix_now());
    response["status"] = json!(if complete { "completed" } else { "incomplete" });
    response["output"] = json!(output);
    if !complete {
        response["incomplete_details"] = json!({"reason":"max_output_tokens"});
    }
    response["usage"] = json!({"input_tokens":generated.metrics.prompt_tokens,"output_tokens":generated.generated_token_ids.len(),"total_tokens":generated.metrics.prompt_tokens+generated.generated_token_ids.len()});
    response["usage"]["input_tokens_details"]["cached_tokens"] =
        generated.metrics.cached_prompt_tokens.into();
    response["usage"]["input_tokens_details"]["cache_write_tokens"] =
        generated.metrics.cache_write_tokens.into();
    // `ResponseUsage` requires output_tokens_details.reasoning_tokens. A turn
    // without reasoning has none; with reasoning the count is not tracked
    // per token yet, so the detail is left out rather than guessed.
    if !reasoned {
        response["usage"]["output_tokens_details"] = json!({"reasoning_tokens": 0});
    }
    response["metrics"] = json!(generated.metrics);
    if let Some(sampling) = &generated.sampling {
        // The values the turn used, which may come from generation_config.json.
        response["temperature"] = json!(sampling.temperature);
        response["top_p"] = json!(sampling.top_p);
        response["metallix"]["sampling"] = json!(sampling);
    }
    Ok(response)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chat_generation::ChatFinishReason;
    use proptest::prelude::*;

    #[test]
    fn chat_template_kwargs_limit_counts_thinking_aliases() {
        for count in [16, 17, 18] {
            let mut kwargs = serde_json::Map::new();
            kwargs.insert("enable_thinking".into(), json!(true));
            kwargs.insert("reasoning_effort".into(), json!("high"));
            for index in 2..count {
                kwargs.insert(format!("variable_{index}"), json!(true));
            }
            assert_eq!(kwargs.len(), count);
            let result = thinking_and_template_kwargs(
                "reasoning.effort",
                None,
                Some(&Value::Object(kwargs)),
            );
            if count == 16 {
                let (thinking, effort, variables) = result.expect("16 names are permitted");
                assert!(thinking);
                assert_eq!(effort.as_deref(), Some("high"));
                assert_eq!(variables.iter().count(), 14);
            } else {
                assert_eq!(
                    result.expect_err("thinking aliases count toward the request limit"),
                    "chat_template_kwargs has more than 16 entries"
                );
            }
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(128))]

        #[test]
        fn function_history_rejects_reused_completed_call_ids(
            id in "[a-zA-Z0-9_]{1,24}",
            output in any::<String>(),
        ) {
            let call = json!({"type":"function_call","name":"read_file","call_id":id,"arguments":"{}"});
            let result = json!({"type":"function_call_output","call_id":id,"output":output});
            let request: Request = serde_json::from_value(json!({"model":"control","input":[call.clone(),result.clone(),call,result]})).unwrap();
            prop_assert!(messages(&request).is_err());
        }

        #[test]
        fn function_results_follow_call_identity_not_completion_order(
            outputs in prop::collection::vec(any::<String>(), 1..=8),
        ) {
            let mut items: Vec<_> = (0..outputs.len()).map(|index| {
                json!({"type":"function_call","name":format!("tool_{index}"),"call_id":format!("call_{index}"),"arguments":"{}"})
            }).collect();
            for (index, output) in outputs.iter().enumerate().rev() {
                items.push(json!({"type":"function_call_output","call_id":format!("call_{index}"),"output":output}));
            }
            let request: Request = serde_json::from_value(json!({"model":"control","input":items})).unwrap();
            let history = messages(&request).map_err(TestCaseError::fail)?;
            prop_assert_eq!(history.len(), 2 * outputs.len());
            for (offset, output) in history[outputs.len()..].iter().enumerate() {
                let index = outputs.len() - 1 - offset;
                let expected_name = format!("tool_{index}");
                let expected_id = format!("call_{index}");
                prop_assert_eq!(output.name.as_deref(), Some(expected_name.as_str()));
                prop_assert_eq!(&output.content, &outputs[index]);
                prop_assert_eq!(output.tool_call_id.as_deref(), Some(expected_id.as_str()));
            }
        }

        #[test]
        fn sse_payloads_cannot_inject_events_or_change_sequence(
            deltas in prop::collection::vec(any::<String>(), 0..24),
        ) {
            let mut wire = Vec::new();
            let mut sequence = 0;
            for delta in &deltas {
                event(&mut wire, &mut sequence, json!({"type":"response.output_text.delta","delta":delta})).unwrap();
            }
            event(&mut wire, &mut sequence, json!({"type":"response.completed"})).unwrap();
            let wire = String::from_utf8(wire).unwrap();
            let frames: Vec<_> = wire.split("\n\n").filter(|frame| !frame.is_empty()).collect();
            prop_assert_eq!(frames.len(), deltas.len() + 1);
            for (index, frame) in frames.iter().enumerate() {
                let lines: Vec<_> = frame.lines().collect();
                prop_assert_eq!(lines.len(), 2);
                let value: Value = serde_json::from_str(lines[1].strip_prefix("data: ").unwrap()).unwrap();
                prop_assert_eq!(value["sequence_number"].as_u64(), Some(index as u64));
                if index < deltas.len() {
                    prop_assert_eq!(value["delta"].as_str(), Some(deltas[index].as_str()));
                } else {
                    prop_assert_eq!(value["type"].as_str(), Some("response.completed"));
                }
            }
        }
    }

    #[test]
    fn reconstructs_function_round_trip_and_rejects_nontext() {
        let request: Request = serde_json::from_value(json!({"model":"control","input":[{"role":"user","content":[{"type":"input_text","text":"read"}]},{"type":"function_call","name":"read_file","call_id":"call_1","arguments":"{\"path\":\"README.md\"}"},{"type":"function_call_output","call_id":"call_1","output":"hello"}]})).unwrap();
        let messages = messages(&request).unwrap();
        assert_eq!(messages.len(), 3);
        assert_eq!(messages[1].tool_calls[0].name, "read_file");
        assert_eq!(messages[2].tool_call_id.as_deref(), Some("call_1"));
        assert!(input_text(&json!([{"type":"input_image","image_url":"x"}])).is_err());
    }

    #[test]
    fn unanswered_function_calls_in_history_render_as_given() {
        let request: Request = serde_json::from_value(json!({"model":"control","input":[{"role":"user","content":"read"},{"type":"function_call","name":"read_file","call_id":"call_1","arguments":"{}"},{"role":"user","content":"never mind"}]})).unwrap();
        let messages = messages(&request).unwrap();
        assert_eq!(messages.len(), 3);
        assert_eq!(messages[1].tool_calls[0].name, "read_file");
        assert_eq!(messages[2].role, ChatRole::User);
    }

    fn request_with(extra: &Value) -> Result<Request, serde_json::Error> {
        let mut request = json!({"model":"control","input":"hello"});
        request
            .as_object_mut()
            .unwrap()
            .extend(extra.as_object().unwrap().clone());
        serde_json::from_value(request)
    }

    const TOOL: &str = r#"{"type":"function","name":"read_file","parameters":{"type":"object"}}"#;

    #[test]
    fn rejects_orphan_results_and_unsupported_request_semantics() {
        let tool: Value = serde_json::from_str(TOOL).unwrap();
        for extra in [
            json!({"input":[{"type":"function_call_output","call_id":"unknown","output":"x"}]}),
            json!({"store":true}),
            json!({"max_output_tokens":0}),
            json!({"temperature":2.5}),
            json!({"temperature":-0.5}),
            json!({"temperature":0.7,"top_p":0.0}),
            json!({"top_p":1.5}),
            json!({"top_logprobs":21}),
            json!({"include":["message.output_text.bogus"]}),
            json!({"reasoning":{"effort":"maximum"}}),
            json!({"background":true}),
            json!({"truncation":"auto"}),
            json!({"max_tool_calls":1}),
            json!({"text":{"format":{"type":"json_schema","name":"x","schema":true}}}),
            json!({"text":{"format":{"type":"json_schema","schema":{}}},"reasoning":{"effort":"low"}}),
            json!({"text":{"format":{"type":"json_schema","schema":{}}},"temperature":0.7,"top_p":0.9}),
            json!({"top_logprobs":2,"tools":[tool.clone()]}),
            json!({"top_logprobs":2,"reasoning":{"effort":"high"}}),
        ] {
            let request = request_with(&extra).unwrap();
            assert!(messages(&request).is_err(), "accepted {extra}");
        }
    }

    #[test]
    fn rejects_unknown_fields_but_accepts_client_defaults() {
        for extra in [
            json!({"frobnicate":true}),
            json!({"reasoning":{"effort":"low","budget":4}}),
            json!({"text":{"format":{"type":"text"},"style":"terse"}}),
            json!({"text":{"format":{"type":"grammar"}}}),
        ] {
            let error = request_with(&extra).err().expect("unknown field rejected");
            assert!(error.to_string().contains("unknown"), "{extra} -> {error}");
        }
        // Codex's default body plus the SDKs' attribution fields.
        let request = request_with(&json!({
            "stream": false,
            "service_tier": "auto",
            "tool_choice": "auto",
            "parallel_tool_calls": false,
            "reasoning": {"effort": "minimal", "summary": "auto"},
            "store": false,
            "stream_options": {"include_obfuscation": false},
            "include": ["reasoning.encrypted_content"],
            "prompt_cache_key": "session-1",
            "text": {"verbosity": "low"},
            "client_metadata": {"origin": "test"},
            "access_programs": null,
            "user": "u",
            "metadata": {"k": "v"},
            "safety_identifier": "s",
            "prompt_cache_retention": "24h",
            "truncation": "disabled",
            "background": false,
            "input": [
                {"type":"reasoning","id":"rs_1","summary":[]},
                {"role":"user","content":"hello"}
            ],
        }))
        .unwrap();
        assert_eq!(messages(&request).unwrap().len(), 1);
        assert_eq!(controls(&request).unwrap(), GenerationControls::default());
    }

    /// `required` and a named function map onto the turn's tool requirement;
    /// `none` is refused because the tools would still reach the prompt.
    #[test]
    fn tool_choice_and_strict_reach_the_controls() {
        let mut tool: Value = serde_json::from_str(TOOL).unwrap();
        let mapped = |extra: Value| controls(&request_with(&extra).unwrap());
        let required = mapped(json!({"tool_choice":"required","tools":[tool.clone()]})).unwrap();
        assert_eq!(required.tool_requirement, ToolRequirement::Required);
        let named = mapped(
            json!({"tool_choice":{"type":"function","name":"read_file"},"tools":[tool.clone()]}),
        )
        .unwrap();
        assert_eq!(
            named.tool_requirement,
            ToolRequirement::Named("read_file".into())
        );
        assert!(mapped(json!({"tool_choice":"none","tools":[tool.clone()]})).is_err());
        assert!(
            mapped(json!({"tool_choice":"required"})).is_err(),
            "required needs tools"
        );
        tool["strict"] = json!(true);
        assert_eq!(
            mapped(json!({"tools":[tool]})).unwrap().strict_tools,
            ["read_file"]
        );
    }

    #[test]
    fn maps_generation_fields_onto_controls() {
        let sampled = controls(
            &request_with(
                &json!({"temperature":0.7,"top_p":0.9,"seed":7,"max_output_tokens":4096}),
            )
            .unwrap(),
        )
        .unwrap();
        assert_eq!(
            sampled.sampling,
            SamplingRequest {
                temperature: Some(0.7),
                top_p: Some(0.9),
                top_k: None,
                seed: Some(7)
            }
        );
        assert_eq!(sampled.max_tokens, Some(4096));
        assert_eq!(
            sampled.speculation,
            engine::speculative::SpeculationRequest::Automatic
        );
        for (field, expected) in [
            ("auto", engine::speculative::SpeculationRequest::Automatic),
            ("on", engine::speculative::SpeculationRequest::Enabled),
            ("off", engine::speculative::SpeculationRequest::Disabled),
        ] {
            let parsed = controls(&request_with(&json!({"speculation":field})).unwrap()).unwrap();
            assert_eq!(parsed.speculation, expected, "{field}");
        }
        let Err(unknown) = request_with(&json!({"speculation":"maybe"})) else {
            panic!("unknown speculation mode must not parse");
        };
        let unknown = unknown.to_string();
        assert!(unknown.contains("`auto`, `on`, `off`"), "{unknown}");
        assert!(!sampled.ignore_eos);
        let ignoring = controls(&request_with(&json!({"ignore_eos":true})).unwrap()).unwrap();
        assert!(ignoring.ignore_eos);
        // Omitted fields stay omitted, for the model's defaults to fill.
        assert_eq!(
            controls(&request_with(&json!({})).unwrap())
                .unwrap()
                .sampling,
            SamplingRequest::default()
        );

        let logprobs = |extra: Value| {
            controls(&request_with(&extra).unwrap())
                .unwrap()
                .top_logprobs
        };
        assert_eq!(logprobs(json!({})), None);
        assert_eq!(logprobs(json!({"include":[LOGPROBS_INCLUDE]})), Some(0));
        assert_eq!(
            logprobs(json!({"include":[LOGPROBS_INCLUDE],"top_logprobs":5})),
            Some(5)
        );
        assert_eq!(logprobs(json!({"top_logprobs":3})), Some(3));

        for (effort, thinking) in [
            ("none", false),
            ("minimal", false),
            ("low", true),
            ("medium", true),
            ("high", true),
            ("xhigh", true),
            ("max", true),
        ] {
            let mapped =
                controls(&request_with(&json!({"reasoning":{"effort":effort}})).unwrap()).unwrap();
            assert_eq!(mapped.enable_thinking, thinking, "{effort}");
            assert_eq!(
                mapped.reasoning_effort.as_deref(),
                thinking.then_some(effort)
            );
        }

        let schema =
            json!({"type":"object","properties":{"ok":{"type":"boolean"}},"required":["ok"]});
        let format = |format: Value| {
            controls(&request_with(&json!({"text":{"format":format}})).unwrap())
                .map(|controls| controls.json_schema)
        };
        assert_eq!(format(json!({"type":"text"})), Ok(None));
        if cfg!(feature = "structured-output") {
            assert_eq!(
                format(json!({"type":"json_schema","name":"r","strict":true,"schema":schema})),
                Ok(Some(schema))
            );
            assert_eq!(
                format(json!({"type":"json_object"})),
                Ok(Some(json!({"type":"object"})))
            );
        } else {
            // A build without the grammar must refuse, never return free text.
            assert!(format(json!({"type":"json_object"})).is_err());
        }
    }

    /// The fields the `OpenAPI` `Response`, `ResponseUsage` and
    /// `OutputTextContent` schemas require (openai-openapi 8f5077ae), filled
    /// from the request where it set them.
    #[test]
    fn responses_carry_every_field_the_published_schema_requires() {
        let tool: Value = serde_json::from_str(TOOL).unwrap();
        let request = request_with(&json!({
            "instructions": "be brief",
            "metadata": {"run": "7"},
            "tools": [tool.clone()],
            "temperature": 0.0,
        }))
        .unwrap();
        let mut generated = generation("hi");
        generated.metrics.cache_write_tokens = 7;
        let response =
            response_value(&request, &controls(&request).unwrap(), &generated, "t").unwrap();
        for field in [
            "access_programs",
            "id",
            "object",
            "created_at",
            "error",
            "incomplete_details",
            "instructions",
            "model",
            "tools",
            "output",
            "parallel_tool_calls",
            "metadata",
            "tool_choice",
            "temperature",
            "top_p",
        ] {
            assert!(response.get(field).is_some(), "Response.{field} is missing");
        }
        assert_eq!(response["instructions"], "be brief");
        assert_eq!(response["metadata"], json!({"run": "7"}));
        let mut echoed = tool.clone();
        echoed["strict"] = Value::Null;
        assert_eq!(
            response["tools"],
            json!([echoed]),
            "FunctionTool requires strict"
        );
        assert_eq!(response["tool_choice"], "auto");
        assert!(response["created_at"].as_u64().unwrap() > 1_700_000_000);
        assert!(response["error"].is_null());
        let usage = &response["usage"];
        for field in [
            "input_tokens",
            "input_tokens_details",
            "output_tokens",
            "output_tokens_details",
            "total_tokens",
        ] {
            assert!(
                usage.get(field).is_some(),
                "ResponseUsage.{field} is missing"
            );
        }
        assert_eq!(usage["output_tokens_details"]["reasoning_tokens"], 0);
        for field in ["cached_tokens", "cache_write_tokens"] {
            assert!(
                usage["input_tokens_details"].get(field).is_some(),
                "input_tokens_details.{field} is missing"
            );
        }
        assert_eq!(usage["input_tokens_details"]["cache_write_tokens"], 7);
        let part = &response["output"][0]["content"][0];
        assert_eq!(part["logprobs"], json!([]), "OutputTextContent.logprobs");

        let bare = request_with(&json!({})).unwrap();
        let envelope = envelope(&bare, "t", 1);
        for field in [
            "instructions",
            "metadata",
            "temperature",
            "top_p",
            "access_programs",
        ] {
            assert!(
                envelope[field].is_null(),
                "{field} is null when the request set nothing"
            );
        }
        assert_eq!(envelope["tools"], json!([]));
        let failure = failed(&envelope, "generation_timeout", "boom");
        assert_eq!(failure["response"]["status"], "failed");
        assert_eq!(
            failure["response"]["error"],
            json!({"code": "server_error", "message": "boom", "metallix_code": "generation_timeout"})
        );
    }

    #[test]
    fn completion_trace_uses_actual_finish_reason() {
        use std::sync::{Arc, Mutex};
        use tracing_subscriber::{Layer, layer::SubscriberExt};

        struct Reasons(Arc<Mutex<Vec<String>>>);
        impl tracing::field::Visit for Reasons {
            fn record_debug(&mut self, _: &tracing::field::Field, _: &dyn std::fmt::Debug) {}
            fn record_str(&mut self, field: &tracing::field::Field, value: &str) {
                if field.name() == "finish_reason" {
                    self.0.lock().unwrap().push(value.to_owned());
                }
            }
        }
        struct Capture(Arc<Mutex<Vec<String>>>);
        impl<S: tracing::Subscriber> Layer<S> for Capture {
            fn on_event(
                &self,
                event: &tracing::Event<'_>,
                _: tracing_subscriber::layer::Context<'_, S>,
            ) {
                event.record(&mut Reasons(Arc::clone(&self.0)));
            }
        }
        crate::hold_trace_capture_anchor();
        let reasons = Arc::new(Mutex::new(Vec::new()));
        let subscriber = tracing_subscriber::registry().with(Capture(Arc::clone(&reasons)));
        tracing::subscriber::with_default(subscriber, || {
            // Identical generated output cannot determine why it stopped.
            for reason in [ChatFinishReason::Eos, ChatFinishReason::Length] {
                record_usage(&generation_ending("same output", reason));
            }
        });
        assert_eq!(*reasons.lock().unwrap(), ["eos", "length"]);
    }

    fn generation(text: &str) -> crate::chat_generation::ChatGeneration {
        generation_ending(text, ChatFinishReason::Eos)
    }

    fn generation_ending(
        text: &str,
        finish_reason: ChatFinishReason,
    ) -> crate::chat_generation::ChatGeneration {
        use crate::chat_generation::ChatGenerationMetrics;
        let mut generated = crate::chat_generation::ChatGeneration::scripted(
            text,
            true,
            finish_reason,
            ChatGenerationMetrics {
                context_tokens: 2048,
                planned_kv_bytes: 0,
                session_load_ms: 0.0,
                render_ms: 0.0,
                prefill_ms: 0.0,
                time_to_first_token_ms: None,
                decode_ms: vec![],
                decode_total_ms: 0.0,
                prompt_tokens: 1,
                cached_prompt_tokens: 0,
                cache_write_tokens: 0,
                generated_tokens: 2,
                speculation: None,
            },
        );
        generated.generated_token_ids = vec![1, 2];
        generated
    }

    #[test]
    fn omitted_sampling_fields_take_the_model_defaults() {
        use crate::chat_generation::SamplingDefaults;
        // Qwen3-0.6B's generation_config.json.
        let qwen = SamplingDefaults {
            temperature: Some(0.6),
            top_p: Some(0.95),
            top_k: Some(20),
        };
        let resolve = |extra: Value, defaults, constrained| {
            controls(&request_with(&extra).unwrap())
                .unwrap()
                .sampling
                .resolve(defaults, constrained)
        };
        let all = resolve(json!({"seed":4}), qwen, false);
        assert_eq!(
            (all.temperature, all.top_p, all.top_k, all.seed),
            (0.6, 0.95, Some(20), Some(4))
        );
        assert_eq!(all.defaults_applied, ["temperature", "top_p", "top_k"]);
        assert_eq!(all.sampler, Some(crate::qwen_forward::SamplerId::IcdfV1));
        assert_eq!(serde_json::to_value(&all).unwrap()["sampler"], "icdf-v1");
        // Explicit fields win; top_k has no request field, so it still applies.
        let some = resolve(json!({"temperature":1.0,"top_p":0.5}), qwen, false);
        assert_eq!(
            (some.temperature, some.top_p, some.top_k),
            (1.0, 0.5, Some(20))
        );
        assert_eq!(some.defaults_applied, ["top_k"]);
        assert!(
            some.seed.is_some(),
            "an unseeded turn reports its drawn seed"
        );
        // An explicit zero temperature is greedy and takes no defaults.
        let greedy = resolve(json!({"temperature":0.0,"top_p":0.5,"seed":1}), qwen, false);
        assert_eq!(
            (greedy.temperature, greedy.top_k, greedy.seed),
            (0.0, None, None)
        );
        assert!(greedy.defaults_applied.is_empty());
        assert_eq!(greedy.sampler, None, "greedy turns name no seeded sampler");
        // A model without defaults stays greedy.
        let bare = resolve(json!({}), SamplingDefaults::default(), false);
        assert_eq!((bare.temperature, bare.seed), (0.0, None));
        assert!(bare.defaults_applied.is_empty());
        // Under a grammar only the default temperature carries over.
        let constrained = resolve(json!({}), qwen, true);
        assert_eq!(
            (
                constrained.temperature,
                constrained.top_p,
                constrained.top_k
            ),
            (0.6, 1.0, None)
        );
        assert_eq!(constrained.defaults_applied, ["temperature"]);
    }

    #[test]
    fn response_carries_reasoning_logprobs_and_sampling() {
        let request = request_with(&json!({"reasoning":{"effort":"low"}})).unwrap();
        let thinking = controls(&request).unwrap();
        let response = response_value(
            &request,
            &thinking,
            &generation("<think>\nadd</think>\n\n4"),
            "t",
        )
        .unwrap();
        assert_eq!(response["output"][0]["type"], "reasoning");
        assert_eq!(response["output"][0]["content"][0]["text"], "add");
        assert_eq!(response["output"][1]["content"][0]["text"], "4");
        // `OutputTextContent` requires the key; it is empty unless asked for.
        assert_eq!(response["output"][1]["content"][0]["logprobs"], json!([]));

        let request = request_with(&json!({"top_logprobs":1,"temperature":1.0})).unwrap();
        let mut generated = generation("hi");
        generated.metrics.cached_prompt_tokens = 1;
        generated.sampling = Some(
            SamplingRequest {
                temperature: Some(1.0),
                seed: Some(9),
                ..SamplingRequest::default()
            }
            .resolve(crate::chat_generation::SamplingDefaults::default(), false),
        );
        generated.logprobs = vec![TokenLogprob {
            token: "hi".into(),
            bytes: b"hi".to_vec(),
            logprob: -0.25,
            top_logprobs: vec![],
        }];
        let response =
            response_value(&request, &controls(&request).unwrap(), &generated, "t").unwrap();
        let logprobs = &response["output"][0]["content"][0]["logprobs"];
        assert_eq!(logprobs[0]["token"], "hi");
        assert_eq!(logprobs[0]["bytes"], json!([104, 105]));
        assert_eq!(logprobs[0]["logprob"], -0.25);
        assert_eq!(response["metallix"]["sampling"]["seed"], 9);
        assert_eq!(
            response["usage"]["input_tokens_details"]["cached_tokens"],
            1
        );
        assert_eq!(response["metallix"]["sampling"]["temperature"], 1.0);
        assert_eq!(
            response["metallix"]["sampling"]["defaults_applied"],
            json!([])
        );
    }

    #[test]
    fn rejects_ambiguous_function_definitions() {
        let tool = json!({"type":"function","name":"read_file","parameters":{"type":"object"}});
        let request: Request = serde_json::from_value(
            json!({"model":"control","input":"hello","tools":[tool.clone(),tool]}),
        )
        .unwrap();
        assert!(tools(&request).unwrap_err().contains("unique"));
    }

    #[test]
    /// Explicitly non-strict arguments pass through; malformed or truncated
    /// calls still fail, and undeclared names remain client-validated.
    fn explicitly_non_strict_calls_preserve_parsing_boundaries() {
        let read_file = json!({"type":"function","name":"read_file","strict":false,"parameters":{"type":"object","properties":{"path":{"type":"string"}},"required":["path"],"additionalProperties":false}});
        let request: Request = serde_json::from_value(
            json!({"model":"control","input":"hello","tools":[read_file.clone()]}),
        )
        .unwrap();
        // The strict list comes from the request's controls.
        let response_value = |request: &Request, generated: &_, id| {
            response_value(request, &controls(request).unwrap(), generated, id)
        };
        assert!(response_value(&request, &generation("<tool_call>{"), "test").is_err());
        let undeclared = r#"<tool_call>{"name":"shell","arguments":{}}</tool_call>"#;
        let off_schema = r#"<tool_call>{"name":"read_file","arguments":{"path":5}}</tool_call>"#;
        let response = response_value(&request, &generation(undeclared), "test").unwrap();
        assert_eq!(response["output"][0]["name"], "shell");
        let response = response_value(&request, &generation(off_schema), "test").unwrap();
        assert_eq!(response["output"][0]["arguments"], r#"{"path":5}"#);

        let mut strict_tool = read_file;
        strict_tool["strict"] = json!(true);
        let strict: Request = serde_json::from_value(
            json!({"model":"control","input":"hello","tools":[strict_tool]}),
        )
        .unwrap();
        assert!(response_value(&strict, &generation(off_schema), "test").is_err());
        assert!(response_value(&strict, &generation(undeclared), "test").is_ok());
        let call =
            r#"<tool_call>{"name":"read_file","arguments":{"path":"README.md"}}</tool_call>"#;
        let response = response_value(&request, &generation(call), "test").unwrap();
        assert_eq!(response["output"][0]["type"], "function_call");
        assert_eq!(
            response["output"][0]["arguments"],
            r#"{"path":"README.md"}"#
        );
        let mixed = response_value(
            &request,
            &generation(&format!("I'll read it. {call}")),
            "test",
        )
        .unwrap();
        assert_eq!(mixed["output"][0]["type"], "message");
        assert_eq!(mixed["output"][0]["content"][0]["text"], "I'll read it. ");
        assert_eq!(mixed["output"][1]["type"], "function_call");
        let truncated = generation_ending(call, ChatFinishReason::Length);
        assert!(response_value(&request, &truncated, "test").is_err());
    }

    #[test]
    fn responses_declared_call_validation_respects_protocol_defaults() {
        let controls = GenerationControls::default();
        for strict in [
            None,
            Some(Value::Null),
            Some(json!(true)),
            Some(json!(false)),
        ] {
            for valid in [false, true] {
                let mut tool = json!({"type":"function","name":"read_file","parameters":{"type":"object","properties":{"path":{"type":"string"}},"required":["path"],"additionalProperties":false}});
                if let Some(flag) = &strict {
                    tool["strict"] = flag.clone();
                }
                let request: Request = serde_json::from_value(
                    json!({"model":"control","input":"hello","tools":[tool]}),
                )
                .unwrap();
                let arguments = if valid {
                    json!({"path":"README.md"})
                } else {
                    json!({"path":5})
                };
                let text = format!(
                    "<tool_call>{{\"name\":\"read_file\",\"arguments\":{arguments}}}</tool_call>"
                );
                let result = response_value(&request, &controls, &generation(&text), "test");
                let reject = strict != Some(json!(false)) && !valid;
                assert_eq!(result.is_err(), reject, "strict={strict:?}, valid={valid}");
                if let Ok(response) = result {
                    assert_eq!(response["output"][0]["name"], "read_file");
                    let returned: Value =
                        serde_json::from_str(response["output"][0]["arguments"].as_str().unwrap())
                            .unwrap();
                    assert_eq!(returned, arguments);
                }
            }
        }
    }

    mod wire {
        use std::{
            io::Read as _,
            net::{Shutdown, TcpListener, TcpStream},
            thread,
        };

        use super::*;
        use crate::{chat_generation::ChatRequest, http_transport::TransportLimits};

        /// Sends one body through the real transport and `respond`, returning
        /// the raw HTTP response.
        pub(super) fn exchange(body: &str, backend: &mut dyn ChatBackend) -> String {
            let listener = TcpListener::bind("127.0.0.1:0").unwrap();
            let address = listener.local_addr().unwrap();
            let body = body.to_owned();
            let client = thread::spawn(move || {
                let mut stream = TcpStream::connect(address).unwrap();
                write!(
                    stream,
                    "POST /v1/responses HTTP/1.1\r\nHost: localhost\r\nContent-Length: {}\r\n\r\n{body}",
                    body.len()
                )
                .unwrap();
                stream.shutdown(Shutdown::Write).unwrap();
                let mut response = Vec::new();
                stream.read_to_end(&mut response).unwrap();
                String::from_utf8(response).unwrap()
            });
            let (stream, _) = listener.accept().unwrap();
            let mut connection = Connection::accept(stream, TransportLimits::default());
            let wire = connection.read_request().unwrap();
            let request: Request = serde_json::from_slice(&wire.body).unwrap();
            let messages = messages(&request).unwrap();
            let tools = tools(&request).unwrap();
            respond(
                connection,
                &request,
                &messages,
                &tools,
                backend,
                "t",
                Duration::from_secs(600),
            )
            .unwrap();
            client.join().unwrap()
        }

        /// The JSON body of a non-streamed response, with its status line.
        pub(super) fn json_body(wire: &str) -> (String, Value) {
            crate::sse::test_support::json_body(wire)
        }

        /// A reply written only at the end still notices a client that
        /// left: generation stops within two client checks.
        #[test]
        fn a_client_that_leaves_stops_a_non_streamed_generation() {
            let mut backend = crate::sse::test_support::Scripted::new(&"word ".repeat(400));
            backend.piece_delay = Duration::from_millis(1);
            let body = r#"{"model":"control","input":"hello"}"#;
            crate::sse::test_support::abandon("/v1/responses", body, |connection, body| {
                let request: Request = serde_json::from_slice(body).unwrap();
                let messages = messages(&request).unwrap();
                let tools = tools(&request).unwrap();
                respond(
                    connection,
                    &request,
                    &messages,
                    &tools,
                    &mut backend,
                    "t",
                    Duration::from_secs(60),
                )
                .unwrap();
            });
            assert!(
                backend.deltas <= 2 * crate::http_transport::CLIENT_POLL_DELTAS,
                "{} pieces generated after the client left",
                backend.deltas
            );
        }

        /// The `data` payloads of a streamed response.
        pub(super) fn events(wire: &str) -> Vec<Value> {
            let (head, payload) = wire.split_once("\r\n\r\n").unwrap();
            assert!(head.starts_with("HTTP/1.1 200 OK\r\n"), "{head}");
            payload
                .split("\n\n")
                .filter_map(|frame| frame.lines().find_map(|line| line.strip_prefix("data: ")))
                .map(|data| serde_json::from_str(data).unwrap())
                .collect()
        }

        /// Max tokens, sampling, top logprobs, thinking, and schema presence.
        type Seen = (Option<u32>, SamplingRequest, Option<u8>, bool, bool);

        /// Records what reached the backend; optionally fails before any token.
        #[derive(Default)]
        struct Recording {
            seen: Option<Seen>,
            fail_before_output: bool,
        }

        impl ChatBackend for Recording {
            fn load_ms(&self) -> f64 {
                0.0
            }

            fn template_reads(&self, name: &str) -> Option<bool> {
                Some(name == "preserve_thinking")
            }

            fn generate_with_timeout(
                &mut self,
                request: ChatRequest<'_>,
                _timeout: Duration,
                on_token: &mut dyn FnMut(chat_format::TurnDelta) -> Result<(), String>,
            ) -> Result<crate::chat_generation::ChatGeneration, ChatGenerationError> {
                self.seen = Some((
                    request.max_tokens,
                    request.sampling,
                    request.top_logprobs,
                    request.enable_thinking,
                    request.json_schema.is_some(),
                ));
                if self.fail_before_output {
                    return Err(ChatGenerationError::Message(
                        "chat requires prompt_tokens + max_tokens <= 16384; received 9 + 20000 = 20009".into(),
                    ));
                }
                on_token(chat_format::TurnDelta::Text("{}".into()))
                    .map_err(ChatGenerationError::Message)?;
                Ok(generation("{}"))
            }
        }

        #[test]
        fn controls_reach_the_backend_unchanged() {
            let mut backend = Recording::default();
            let body = r#"{"model":"control","input":"hi","max_output_tokens":9000,"temperature":0.5,"top_p":0.8,"seed":3,"top_logprobs":2}"#;
            let (status, _) = json_body(&exchange(body, &mut backend));
            assert_eq!(status, "HTTP/1.1 200 OK");
            assert_eq!(
                backend.seen,
                Some((
                    Some(9000),
                    SamplingRequest {
                        temperature: Some(0.5),
                        top_p: Some(0.8),
                        top_k: None,
                        seed: Some(3)
                    },
                    Some(2),
                    false,
                    false
                ))
            );
            let schema = cfg!(feature = "structured-output");
            let body = format!(
                r#"{{"model":"control","input":"hi","stream":true,"reasoning":{{"effort":"none"}},"text":{{"format":{{"type":"{}"}}}}}}"#,
                if schema { "json_object" } else { "text" }
            );
            let events = events(&exchange(&body, &mut backend));
            assert_eq!(
                backend.seen,
                Some((None, SamplingRequest::default(), None, false, schema))
            );
            assert_eq!(events.last().unwrap()["type"], "response.completed");
            let done = events
                .iter()
                .find(|event| event["type"] == "response.output_text.done")
                .unwrap();
            assert_eq!(done["text"], "{}");
            assert_eq!(done["logprobs"], json!([]));
        }

        #[test]
        fn responses_streaming_validation_respects_protocol_defaults() {
            for strict in [
                None,
                Some(Value::Null),
                Some(json!(true)),
                Some(json!(false)),
            ] {
                for valid in [false, true] {
                    let mut tool: Value = serde_json::from_str(TOOL).unwrap();
                    if let Some(flag) = &strict {
                        tool["strict"] = flag.clone();
                    }
                    tool["parameters"] = json!({"type":"object","properties":{"path":{"type":"string"}},"required":["path"],"additionalProperties":false});
                    let arguments = if valid {
                        json!({"path":"README.md"})
                    } else {
                        json!({"path":5})
                    };
                    let output = format!(
                        "Checking.<tool_call>{{\"name\":\"read_file\",\"arguments\":{arguments}}}</tool_call>"
                    );
                    let mut backend = crate::sse::test_support::Scripted::new(&output);
                    let body =
                        json!({"model":"control","input":"hello","tools":[tool],"stream":true})
                            .to_string();
                    let frames = events(&exchange(&body, &mut backend));
                    let last = frames.last().expect("terminal response event");
                    if strict != Some(json!(false)) && !valid {
                        assert_eq!(last["type"], "response.failed");
                        assert_eq!(last["response"]["error"]["code"], "server_error");
                        assert_eq!(
                            last["response"]["error"]["metallix_code"],
                            "invalid_model_output"
                        );
                    } else {
                        assert_eq!(last["type"], "response.completed");
                        assert!(
                            last["response"]["output"]
                                .as_array()
                                .unwrap()
                                .iter()
                                .any(|item| item["type"] == "function_call"
                                    && item["name"] == "read_file")
                        );
                    }
                }
            }
        }

        /// A hosted tool next to a function tool is left out of the prompt
        /// and named in `metallix.dropped_tools`, streamed (from
        /// `response.created` on) and not; the function tool stays offered.
        #[test]
        fn hosted_tools_are_dropped_and_reported() {
            let tools = r#"[{"type":"function","name":"read_file","parameters":{"type":"object"}},{"type":"web_search"}]"#;
            let request: Request = serde_json::from_str(&format!(
                r#"{{"model":"control","input":"hi","tools":{tools}}}"#
            ))
            .unwrap();
            let offered = super::tools(&request).unwrap();
            assert_eq!(offered.len(), 1);
            assert_eq!(offered[0]["function"]["name"], "read_file");
            assert_eq!(super::dropped_tools(&request), ["web_search"]);
            let mut backend = Recording::default();
            let body = format!(r#"{{"model":"control","input":"hi","tools":{tools}}}"#);
            let (status, response) = json_body(&exchange(&body, &mut backend));
            assert_eq!(status, "HTTP/1.1 200 OK", "{response}");
            assert_eq!(response["metallix"]["dropped_tools"], json!(["web_search"]));
            let body =
                format!(r#"{{"model":"control","input":"hi","stream":true,"tools":{tools}}}"#);
            let events = events(&exchange(&body, &mut backend));
            for kind in ["response.created", "response.completed"] {
                let event = events.iter().find(|event| event["type"] == kind).unwrap();
                assert_eq!(
                    event["response"]["metallix"]["dropped_tools"],
                    json!(["web_search"]),
                    "{kind}"
                );
            }
        }

        /// `chat_template_kwargs` (a metallix extension here, as in vLLM):
        /// a name the model's template does not read is reported in
        /// `metallix.ignored_template_kwargs`, streamed (from
        /// `response.created` on) and not; `enable_thinking` there maps to
        /// the effort switch and must agree with `reasoning.effort`.
        #[test]
        fn chat_template_kwargs_report_unread_names() {
            let kwargs = r#""chat_template_kwargs":{"preserve_thinking":true,"other":"x"}"#;
            let mut backend = Recording::default();
            let body = format!(r#"{{"model":"control","input":"hi",{kwargs}}}"#);
            let (status, response) = json_body(&exchange(&body, &mut backend));
            assert_eq!(status, "HTTP/1.1 200 OK", "{response}");
            assert_eq!(
                response["metallix"]["ignored_template_kwargs"],
                json!(["other"])
            );
            let body = format!(r#"{{"model":"control","input":"hi","stream":true,{kwargs}}}"#);
            let events = events(&exchange(&body, &mut backend));
            for kind in ["response.created", "response.completed"] {
                let event = events.iter().find(|event| event["type"] == kind).unwrap();
                assert_eq!(
                    event["response"]["metallix"]["ignored_template_kwargs"],
                    json!(["other"]),
                    "{kind}"
                );
            }
            let request: Request = serde_json::from_str(
                r#"{"model":"control","input":"hi","chat_template_kwargs":{"enable_thinking":true}}"#,
            )
            .unwrap();
            assert!(super::controls(&request).unwrap().enable_thinking);
            let request: Request = serde_json::from_str(
                r#"{"model":"control","input":"hi","reasoning":{"effort":"none"},"chat_template_kwargs":{"enable_thinking":true}}"#,
            )
            .unwrap();
            assert!(
                super::controls(&request)
                    .unwrap_err()
                    .contains("disagrees with reasoning.effort")
            );
        }

        #[test]
        fn failure_before_any_output_is_an_http_error_even_when_streaming() {
            for stream in [false, true] {
                let mut backend = Recording {
                    fail_before_output: true,
                    ..Recording::default()
                };
                let body = format!(
                    r#"{{"model":"control","input":"hi","stream":{stream},"max_output_tokens":20000}}"#
                );
                let (status, body) = json_body(&exchange(&body, &mut backend));
                assert_eq!(status, "HTTP/1.1 400 Bad Request", "stream={stream}");
                assert!(
                    body["error"]["message"]
                        .as_str()
                        .unwrap()
                        .contains("prompt_tokens + max_tokens <= 16384")
                );
            }
        }
    }

    /// Opt-in checks against a real local Qwen3 checkpoint, for example
    /// `METALLIX_QWEN_MODEL=~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/<rev>`.
    mod real_qwen {
        use std::path::Path;

        use super::wire::{exchange, json_body};
        use super::*;
        use crate::chat_generation::{ChatSession, ResidentChatLimits};

        fn session() -> Option<ChatSession> {
            let model = std::env::var_os("METALLIX_QWEN_MODEL")?;
            Some(
                ChatSession::load(
                    Path::new(&model),
                    ResidentChatLimits::from_mib(16_384, 8_192),
                )
                .expect("load local Qwen checkpoint"),
            )
        }

        fn text(response: &Value) -> &str {
            response["output"]
                .as_array()
                .unwrap()
                .iter()
                .find(|item| item["type"] == "message")
                .unwrap()["content"][0]["text"]
                .as_str()
                .unwrap()
        }

        #[test]
        #[ignore = "requires METALLIX_QWEN_MODEL and a local Apple-Silicon Metal checkpoint"]
        fn seeded_sampling_logprobs_and_context_cap() {
            let Some(mut session) = session() else {
                eprintln!("skipping: METALLIX_QWEN_MODEL is not set");
                return;
            };
            let sampled = |seed: u64, session: &mut ChatSession| {
                let body = format!(
                    r#"{{"model":"q","input":"Write one sentence about the sea.","temperature":1.0,"top_p":0.95,"seed":{seed},"max_output_tokens":24}}"#
                );
                let (status, response) = json_body(&exchange(&body, session));
                assert_eq!(status, "HTTP/1.1 200 OK", "{response}");
                assert_eq!(response["metallix"]["sampling"]["seed"], seed);
                text(&response).to_owned()
            };
            let first = sampled(1234, &mut session);
            assert_eq!(first, sampled(1234, &mut session));
            assert_ne!(first, sampled(99, &mut session));

            // Omitted sampling fields take the checkpoint's generation_config.json.
            let body = r#"{"model":"q","input":"Say hello.","max_output_tokens":4}"#;
            let (_, response) = json_body(&exchange(body, &mut session));
            let sampling = &response["metallix"]["sampling"];
            // Qwen3 checkpoints ship temperature, top_p and top_k.
            assert_eq!(
                sampling["defaults_applied"],
                json!(["temperature", "top_p", "top_k"]),
                "{response}"
            );
            // session() already returned, so the variable is set.
            let model = std::env::var_os("METALLIX_QWEN_MODEL").unwrap();
            let config: Value = serde_json::from_slice(
                &std::fs::read(Path::new(&model).join("generation_config.json")).unwrap(),
            )
            .unwrap();
            assert_eq!(sampling["temperature"], config["temperature"]);
            assert_eq!(sampling["top_p"], config["top_p"]);
            assert_eq!(sampling["top_k"], config["top_k"]);

            let body = r#"{"model":"q","input":"Say hello.","temperature":0,"include":["message.output_text.logprobs"],"top_logprobs":3,"max_output_tokens":8}"#;
            let (_, response) = json_body(&exchange(body, &mut session));
            let logprobs = response["output"][0]["content"][0]["logprobs"]
                .as_array()
                .unwrap();
            let output_tokens = response["usage"]["output_tokens"].as_u64().unwrap();
            let eos = u64::from(response["status"] == "completed");
            assert_eq!(logprobs.len() as u64, output_tokens - eos);
            let mut joined = Vec::new();
            for entry in logprobs {
                let top = entry["top_logprobs"].as_array().unwrap();
                assert_eq!(top.len(), 3);
                // Greedy picks the most likely token, so it heads its own list.
                assert_eq!(top[0]["token"], entry["token"]);
                assert!(
                    (top[0]["logprob"].as_f64().unwrap() - entry["logprob"].as_f64().unwrap())
                        .abs()
                        < 1e-9
                );
                assert!(
                    top.windows(2)
                        .all(|pair| pair[0]["logprob"].as_f64() >= pair[1]["logprob"].as_f64())
                );
                assert!(entry["logprob"].as_f64().unwrap() <= 0.0);
                joined.extend(
                    entry["bytes"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(|byte| u8::try_from(byte.as_u64().unwrap()).unwrap()),
                );
            }
            assert_eq!(String::from_utf8(joined).unwrap(), text(&response));

            let body = r#"{"model":"q","input":"hi","stream":true,"max_output_tokens":16384}"#;
            let (status, response) = json_body(&exchange(body, &mut session));
            assert_eq!(status, "HTTP/1.1 400 Bad Request");
            assert!(
                response["error"]["message"]
                    .as_str()
                    .unwrap()
                    .contains("16384")
            );

            let body = r#"{"model":"q","input":"What is 2+3? Answer briefly.","reasoning":{"effort":"low"},"max_output_tokens":384}"#;
            let (_, response) = json_body(&exchange(body, &mut session));
            assert_eq!(response["output"][0]["type"], "reasoning", "{response}");
            assert!(
                !response["output"][0]["content"][0]["text"]
                    .as_str()
                    .unwrap()
                    .contains("<think>")
            );
        }

        #[cfg(feature = "structured-output")]
        #[test]
        #[ignore = "requires METALLIX_QWEN_MODEL and a local Apple-Silicon Metal checkpoint"]
        fn json_schema_output_validates_streamed_and_not() {
            use super::wire::events;

            let Some(mut session) = session() else {
                eprintln!("skipping: METALLIX_QWEN_MODEL is not set");
                return;
            };
            let schema = json!({
                "type": "object",
                "properties": {
                    "city": {"type": "string", "maxLength": 40},
                    "population": {"type": "integer", "minimum": 0}
                },
                "required": ["city", "population"],
                "additionalProperties": false
            });
            let validator = chat_format::validator(&schema).unwrap();
            for stream in [false, true] {
                let body = json!({
                    "model": "q",
                    "input": "Give the largest city in Japan and its population as JSON.",
                    "stream": stream,
                    "max_output_tokens": 96,
                    "temperature": 0.7,
                    "seed": 5,
                    "text": {"format": {"type": "json_schema", "name": "city", "strict": true, "schema": schema}}
                })
                .to_string();
                let wire = exchange(&body, &mut session);
                let response = if stream {
                    events(&wire).pop().unwrap()["response"].clone()
                } else {
                    json_body(&wire).1
                };
                assert_eq!(response["status"], "completed", "{response}");
                let value: Value = serde_json::from_str(text(&response)).unwrap();
                assert!(validator.is_valid(&value), "{value}");
            }
        }
    }
    // Recorded Codex 0.160.0 requests, trimmed (instructions, input and
    // descriptions replaced). A grammar loads only when the controls carry a
    // JSON schema (chat_generation builds its ConstraintRun from it), so this
    // decides f, the share of decode tokens on the constrained sampling path.
    #[test]
    fn recorded_codex_requests_load_a_grammar_only_with_an_output_schema() {
        let parse = |text: &str| serde_json::from_str::<Request>(text).unwrap();
        // Default session: tool_choice "auto", every tool strict false. The
        // whole request resolves, and its controls carry no schema.
        let edit = parse(include_str!(
            "fixtures/client_requests/codex-0.160.0-edit.json"
        ));
        assert_eq!(tools(&edit).map(|tools| tools.len()), Ok(4));
        assert!(messages(&edit).is_ok());
        assert_eq!(controls(&edit).map(|c| c.json_schema.is_some()), Ok(false));
        // `codex exec --output-schema` sends text.format json_schema (strict)
        // together with its tools and reasoning. The schema then binds the
        // turn's final message through the tool grammar, so this mode
        // reaches the constrained path.
        let recorded = include_str!("fixtures/client_requests/codex-0.160.0-output-schema.json");
        let schema = parse(recorded);
        let resolved = controls(&schema);
        if cfg!(feature = "structured-output") {
            let controls = resolved.unwrap();
            assert!(controls.json_schema.is_some());
            let messages = Vec::new();
            let tools = Vec::from([json!({"type":"function","function":{"name":"exec_command"}})]);
            assert!(controls.request(&messages, &tools).constrains_tool_calls());
        } else {
            assert!(resolved.is_err());
        }
        // Its hosted `web_search` tool is left out of the prompt and reported;
        // the seven function tools stay offered.
        assert_eq!(tools(&schema).map(|tools| tools.len()), Ok(7));
        assert_eq!(dropped_tools(&schema), ["web_search"]);
        // The same schema without tools and reasoning is what selects it.
        let mut bare: Value = serde_json::from_str(recorded).unwrap();
        let object = bare.as_object_mut().unwrap();
        object.remove("tools");
        object.remove("reasoning");
        let resolved =
            controls(&serde_json::from_value(bare).unwrap()).map(|c| c.json_schema.is_some());
        if cfg!(feature = "structured-output") {
            assert_eq!(resolved, Ok(true));
        } else {
            assert!(resolved.is_err());
        }
    }
}
