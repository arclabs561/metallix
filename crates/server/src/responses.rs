//! Text-only Responses protocol: request validation, message and tool
//! reconstruction, and the streamed (SSE) or JSON response for one generation.

use std::{
    collections::{HashMap, HashSet},
    io::{BufWriter, Write},
    time::Duration,
};

use serde::{Deserialize, de::IgnoredAny};
use serde_json::{Value, json};

use crate::{
    chat_cli::message,
    chat_generation::{
        ChatBackend, ChatFinishReason, ChatGenerationError, ChatMessage, ChatRequest, ChatRole,
        ChatToolCall, GenerationControls, SamplingRequest, TokenLogprob,
    },
    chat_tools,
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
    #[serde(default)]
    store: Option<bool>,
    #[serde(default)]
    top_logprobs: Option<u8>,
    #[serde(default)]
    include: Option<Vec<String>>,
    #[serde(default)]
    reasoning: Option<Reasoning>,
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
    #[serde(default, rename = "metadata")]
    _metadata: Option<IgnoredAny>,
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

/// Maps the request's generation fields onto the protocol-neutral controls.
///
/// `reasoning.effort` maps through [`thinking_for_effort`]. As in vLLM, an
/// omitted `temperature` or `top_p` takes the checkpoint's
/// `generation_config.json` default (and its `top_k`, which has no Responses
/// field), greedy if it has none; an explicit temperature of zero is greedy.
/// The response's `metallix.sampling` records the policy used and which
/// defaults applied.
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
    let (enable_thinking, reasoning_effort) = thinking_for_effort(
        request
            .reasoning
            .as_ref()
            .and_then(|reasoning| reasoning.effort.as_deref()),
    )
    .map_err(|effort| format!("unsupported reasoning.effort {effort:?}"))?;
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
    };
    controls.validate(!request.tools.is_empty())?;
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
    if request.tool_choice.as_ref().is_some_and(|v| v != "auto") {
        return Err("only automatic tool choice is supported".into());
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

pub(crate) fn tools(request: &Request) -> Result<Vec<Value>, String> {
    let mut names = HashSet::new();
    request.tools.iter().map(|tool| {
        if tool["type"] != "function" || !tool["name"].is_string() || !tool["parameters"].is_object() {
            return Err("only function tools with name and parameters are supported".into());
        }
        let name = tool["name"].as_str().expect("checked string");
        if name.is_empty() || !names.insert(name) {
            return Err("function names must be nonempty and unique".into());
        }
        chat_tools::validator(&tool["parameters"])?;
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
    span.record("gen_ai.usage.input_tokens", generated.metrics.prompt_tokens);
    span.record(
        "gen_ai.usage.output_tokens",
        generated.generated_token_ids.len(),
    );
}

pub(crate) fn json_response(mut connection: Connection, status: u16, value: &Value) {
    connection.begin_response();
    let request_id = connection.request_id_header();
    let mut writer = BufWriter::new(connection);
    let body = value.to_string();
    let reason = match status {
        200 => "OK",
        400 => "Bad Request",
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
    /// The message item opened with the stream when text streams live.
    live_message: Option<&'a str>,
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
                json!({"type":"response.created","response":{"id":self.id,"object":"response","status":"in_progress","output":[]}}),
            )?;
            if let Some(message_id) = self.live_message {
                event(
                    &mut writer,
                    &mut self.sequence,
                    json!({"type":"response.output_item.added","output_index":0,"item":{"id":message_id,"type":"message","role":"assistant","status":"in_progress","content":[]}}),
                )?;
                event(
                    &mut writer,
                    &mut self.sequence,
                    json!({"type":"response.content_part.added","item_id":message_id,"output_index":0,"content_index":0,"part":{"type":"output_text","text":"","annotations":[]}}),
                )?;
            }
            self.writer = Some(writer);
        }
        Ok(self.writer.as_mut().expect("opened stream writer"))
    }

    fn emit(&mut self, value: Value) -> Result<(), String> {
        self.open()?;
        let writer = self.writer.as_mut().expect("opened stream writer");
        event(writer, &mut self.sequence, value)
    }

    /// Withholds incomplete tool envelopes and reasoning, but still detects
    /// disconnects once output has begun.
    fn keepalive(&mut self) -> Result<(), String> {
        let writer = self.open()?;
        writer
            .write_all(b": generating\n\n")
            .map_err(|error| error.to_string())?;
        writer.flush().map_err(|error| error.to_string())
    }
}

fn error_response(connection: Connection, error: &ChatGenerationError) {
    match error {
        ChatGenerationError::DeadlineExceeded => json_response(
            connection,
            408,
            &json!({"error":{"code":"generation_timeout","message":"generation time budget exceeded"}}),
        ),
        ChatGenerationError::Message(message) => {
            json_response(connection, 400, &json!({"error":{"message":message}}));
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
            json_response(request, 400, &json!({"error":{"message":error}}));
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
    let message_id = format!("msg_{id}");
    // Text streams live only when no tool envelope or reasoning block has to
    // be parsed out of it first.
    let stream_text = tools.is_empty() && !controls.enable_thinking;
    let mut stream = EventStream {
        pending: Some(request),
        writer: None,
        sequence: 0,
        id,
        live_message: stream_text.then_some(message_id.as_str()),
    };
    let generated = session.generate_with_timeout(
        ChatRequest {
            cache_salt: cache_salt.as_deref(),
            ..controls.request(messages, tools)
        },
        generation_timeout,
        &mut |delta| {
            if stream_text {
                // Per-token logprobs arrive on `response.output_text.done`.
                stream.emit(json!({"type":"response.output_text.delta","item_id":message_id,"output_index":0,"content_index":0,"delta":delta,"logprobs":[]}))
            } else {
                stream.keepalive()
            }
        },
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
            return stream.emit(
                json!({"type":"response.failed","response":{"id":id,"status":"failed","error":{"code":code,"message":message}}}),
            );
        }
    };
    record_usage(&generated);
    let response = match response_value(parsed, &controls, &generated, id) {
        Ok(mut response) => {
            echo_request_id(&mut response, request_id.as_deref());
            response
        }
        Err(error) => {
            return stream.emit(
                json!({"type":"response.failed","response":{"id":id,"status":"failed","error":{"code":"invalid_model_output","message":error}}}),
            );
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
            stream.emit(
                json!({"type":"response.output_item.added","output_index":index,"item":{"id":item["id"],"type":"reasoning","summary":[],"content":[]}}),
            )?;
        } else {
            let part = &item["content"][0];
            let logprobs = part.get("logprobs").cloned().unwrap_or_else(|| json!([]));
            if !stream_text {
                stream.emit(
                    json!({"type":"response.output_item.added","output_index":index,"item":{"id":item["id"],"type":"message","role":"assistant","status":"in_progress","content":[]}}),
                )?;
                stream.emit(
                    json!({"type":"response.content_part.added","item_id":item["id"],"output_index":index,"content_index":0,"part":{"type":"output_text","text":"","annotations":[]}}),
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
    request: Connection,
    parsed: &Request,
    controls: &GenerationControls,
    messages: &[ChatMessage],
    tools: &[Value],
    session: &mut dyn ChatBackend,
    id: &str,
    generation_timeout: Duration,
) {
    let request_id = request.request_id().map(str::to_owned);
    let result = session
        .generate_with_timeout(
            ChatRequest {
                cache_salt: request.cache_salt(),
                ..controls.request(messages, tools)
            },
            generation_timeout,
            &mut |_| Ok(()),
        )
        .and_then(|generated| {
            record_usage(&generated);
            let mut response = response_value(parsed, controls, &generated, id)
                .map_err(ChatGenerationError::Message)?;
            echo_request_id(&mut response, request_id.as_deref());
            Ok(response)
        });
    match result {
        Ok(response) => json_response(request, 200, &response),
        Err(error) => error_response(request, &error),
    }
}

/// Splits a thinking turn into its reasoning and its answer. Qwen3 opens the
/// block itself; templates that pre-fill `<think>` leave only the close tag.
fn split_reasoning(text: &str) -> (&str, &str) {
    let (opened, body) = match text.trim_start().strip_prefix("<think>") {
        Some(body) => (true, body),
        None => (false, text),
    };
    match body.split_once("</think>") {
        Some((reasoning, answer)) => (reasoning.trim(), answer.trim_start()),
        None if opened => (body.trim(), ""),
        None => ("", text),
    }
}

pub(crate) fn logprobs_value(logprobs: &[TokenLogprob]) -> Value {
    serde_json::to_value(logprobs).unwrap_or_else(|_| json!([]))
}

/// A finished assistant turn split into reasoning, visible text and tool calls
/// checked against their declared schemas. Each protocol only reshapes it.
pub(crate) struct AssistantTurn {
    pub(crate) reasoning: String,
    pub(crate) text: String,
    pub(crate) calls: Vec<ChatToolCall>,
    /// The model ended its turn; otherwise it hit the output limit.
    pub(crate) complete: bool,
}

/// Parses one generation. `tools` are template-shaped definitions, as
/// [`tools`] returns them. A truncated tool turn is an error, never a partial
/// call.
pub(crate) fn assistant_turn(
    tools: &[Value],
    enable_thinking: bool,
    generated: &crate::chat_generation::ChatGeneration,
) -> Result<AssistantTurn, String> {
    let (reasoning, answer) = if enable_thinking {
        split_reasoning(&generated.text)
    } else {
        ("", generated.text.as_str())
    };
    let turn = chat_tools::parse_turn(answer)?;
    let complete = generated.finish_reason == ChatFinishReason::Eos;
    if !turn.calls.is_empty() && !complete {
        return Err("truncated tool turn; no function calls returned".into());
    }
    let mut calls = Vec::new();
    for call in turn.calls {
        let definition = tools
            .iter()
            .find(|tool| tool["function"]["name"] == call.name)
            .ok_or("model requested an undeclared tool")?;
        if !chat_tools::validator(&definition["function"]["parameters"])?.is_valid(&call.arguments)
        {
            return Err("model tool arguments do not match the declared schema".into());
        }
        calls.push(ChatToolCall {
            name: call.name,
            arguments: call.arguments,
        });
    }
    Ok(AssistantTurn {
        reasoning: reasoning.to_owned(),
        text: turn.text,
        calls,
        complete,
    })
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
    } = assistant_turn(&tools(request)?, controls.enable_thinking, generated)?;
    let mut output = Vec::new();
    if !reasoning.is_empty() {
        output.push(json!({"id":format!("rs_{id}"),"type":"reasoning","summary":[],"content":[{"type":"reasoning_text","text":reasoning}]}));
    }
    if calls.is_empty() || !text.trim().is_empty() {
        let mut part = json!({"type":"output_text","text":text,"annotations":[]});
        if controls.top_logprobs.is_some() {
            part["logprobs"] = logprobs_value(&generated.logprobs);
        }
        output.push(json!({"id":format!("msg_{id}"),"type":"message","role":"assistant","status":if complete {"completed"} else {"incomplete"},"content":[part]}));
    }
    for (index, call) in calls.into_iter().enumerate() {
        output.push(json!({"type":"function_call","id":format!("fc_{id}_{index}"),"call_id":format!("call_{id}_{index}"),"name":call.name,"arguments":call.arguments.to_string(),"status":"completed"}));
    }
    let mut response = json!({"id":id,"object":"response","model":request.model,"status":if complete {"completed"} else {"incomplete"},"output":output,"incomplete_details":if complete {Value::Null} else {json!({"reason":"max_output_tokens"})},"usage":{"input_tokens":generated.metrics.prompt_tokens,"output_tokens":generated.generated_token_ids.len(),"total_tokens":generated.metrics.prompt_tokens+generated.generated_token_ids.len()},"metrics":generated.metrics});
    response["usage"]["input_tokens_details"]["cached_tokens"] =
        generated.metrics.cached_prompt_tokens.into();
    if let Some(sampling) = &generated.sampling {
        response["metallix"] = json!({ "sampling": sampling });
    }
    Ok(response)
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

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
            json!({"text":{"format":{"type":"json_schema","schema":{}}},"tools":[tool.clone()]}),
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

    #[test]
    fn splits_reasoning_from_the_answer() {
        assert_eq!(
            split_reasoning("<think>\nadd them\n</think>\n\n4"),
            ("add them", "4")
        );
        assert_eq!(
            split_reasoning("pre-filled\n</think>\n\n4"),
            ("pre-filled", "4")
        );
        assert_eq!(split_reasoning("<think>\nunfinished"), ("unfinished", ""));
        assert_eq!(split_reasoning("plain answer"), ("", "plain answer"));
    }

    fn generation(text: &str) -> crate::chat_generation::ChatGeneration {
        use crate::chat_generation::ChatGenerationMetrics;
        crate::chat_generation::ChatGeneration {
            text: text.into(),
            generated_token_ids: vec![1, 2],
            finish_reason: ChatFinishReason::Eos,
            metrics: ChatGenerationMetrics {
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
                generated_tokens: 2,
            },
            logprobs: Vec::new(),
            sampling: None,
        }
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
        assert!(
            response["output"][1]["content"][0]
                .get("logprobs")
                .is_none()
        );

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
    fn schema_validation_rejects_undeclared_and_invalid_calls() {
        let request: Request = serde_json::from_value(json!({"model":"control","input":"hello","tools":[{"type":"function","name":"read_file","parameters":{"type":"object","properties":{"path":{"type":"string"}},"required":["path"],"additionalProperties":false}}]})).unwrap();
        let controls = GenerationControls::default();
        let response_value = |request: &Request, generated: &_, id| {
            response_value(request, &controls, generated, id)
        };
        let mut generated = generation("");
        for text in [
            "<tool_call>{",
            r#"<tool_call>{"name":"shell","arguments":{}}</tool_call>"#,
            r#"<tool_call>{"name":"read_file","arguments":{"path":5}}</tool_call>"#,
        ] {
            generated.text = text.into();
            assert!(response_value(&request, &generated, "test").is_err());
        }
        generated.text =
            r#"<tool_call>{"name":"read_file","arguments":{"path":"README.md"}}</tool_call>"#
                .into();
        let response = response_value(&request, &generated, "test").unwrap();
        assert_eq!(response["output"][0]["type"], "function_call");
        assert_eq!(
            response["output"][0]["arguments"],
            r#"{"path":"README.md"}"#
        );
        generated.text = format!("I'll read it. {}", generated.text);
        let mixed = response_value(&request, &generated, "test").unwrap();
        assert_eq!(mixed["output"][0]["type"], "message");
        assert_eq!(mixed["output"][0]["content"][0]["text"], "I'll read it. ");
        assert_eq!(mixed["output"][1]["type"], "function_call");
        generated.finish_reason = ChatFinishReason::Length;
        assert!(response_value(&request, &generated, "test").is_err());
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
            let (head, body) = wire.split_once("\r\n\r\n").unwrap();
            (
                head.lines().next().unwrap().to_owned(),
                serde_json::from_str(body).unwrap(),
            )
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

            fn generate_with_timeout(
                &mut self,
                request: ChatRequest<'_>,
                _timeout: Duration,
                on_token: &mut dyn FnMut(&str) -> Result<(), String>,
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
                on_token("{}").map_err(ChatGenerationError::Message)?;
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

        fn session() -> ChatSession {
            let model = std::env::var_os("METALLIX_QWEN_MODEL")
                .expect("real-checkpoint test requires METALLIX_QWEN_MODEL");
            ChatSession::load(
                Path::new(&model),
                ResidentChatLimits::from_mib(16_384, 8_192),
            )
            .expect("load local Qwen checkpoint")
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
            let mut session = session();
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

            let mut session = session();
            let schema = json!({
                "type": "object",
                "properties": {
                    "city": {"type": "string", "maxLength": 40},
                    "population": {"type": "integer", "minimum": 0}
                },
                "required": ["city", "population"],
                "additionalProperties": false
            });
            let validator = chat_tools::validator(&schema).unwrap();
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
}
