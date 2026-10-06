//! Anthropic Messages over the shared generation controls: request
//! validation, message and tool reconstruction, and the JSON or SSE response.

use std::{
    collections::{HashMap, HashSet},
    time::Duration,
};

use chat_format::TurnDelta;
use serde::{Deserialize, de::IgnoredAny};
use serde_json::{Value, json};

use crate::{
    chat_cli::message,
    chat_generation::{
        ChatBackend, ChatGeneration, ChatGenerationError, ChatMessage, ChatRequest, ChatRole,
        ChatToolCall, GenerationControls, SamplingRequest,
    },
    http_transport::Connection,
    responses::{AssistantTurn, assistant_turn, echo_request_id, json_response, record_usage},
    sse::LazySse,
};

/// One `POST /v1/messages` body. Unknown fields are rejected; fields the
/// Anthropic SDKs and Claude Code send with no generation meaning here are
/// accepted and ignored by name.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Request {
    pub(crate) model: String,
    /// Metallix extension: "auto" (default), "on" or "off" for
    /// prompt-lookup speculative decoding.
    #[serde(default)]
    speculation: crate::chat_generation::SpeculationField,
    max_tokens: u32,
    messages: Vec<Value>,
    #[serde(default)]
    system: Option<Value>,
    #[serde(default)]
    tools: Vec<Value>,
    #[serde(default)]
    tool_choice: Option<ToolChoice>,
    #[serde(default)]
    stream: bool,
    #[serde(default)]
    temperature: Option<f64>,
    #[serde(default)]
    top_p: Option<f64>,
    #[serde(default)]
    top_k: Option<u32>,
    /// Metallix extension, as in vLLM: generate past end-of-turn up to the
    /// output limit, for equal-length benchmark runs.
    #[serde(default)]
    ignore_eos: bool,
    #[serde(default)]
    stop_sequences: Vec<String>,
    #[serde(default)]
    thinking: Option<Thinking>,
    #[serde(default)]
    output_config: Option<OutputConfig>,
    #[serde(default)]
    container: Option<Value>,
    // Accepted and ignored: attribution, routing, prompt-cache markers and
    // context editing, none of which changes what this model generates.
    #[serde(default, rename = "metadata")]
    _metadata: Option<IgnoredAny>,
    #[serde(default, rename = "service_tier")]
    _service_tier: Option<IgnoredAny>,
    #[serde(default, rename = "cache_control")]
    _cache_control: Option<IgnoredAny>,
    #[serde(default, rename = "inference_geo")]
    _inference_geo: Option<IgnoredAny>,
    #[serde(default, rename = "diagnostics")]
    _diagnostics: Option<IgnoredAny>,
    #[serde(default, rename = "context_management")]
    _context_management: Option<IgnoredAny>,
    #[serde(default, rename = "user_profile_id")]
    _user_profile_id: Option<IgnoredAny>,
    #[serde(default, rename = "workspace_id")]
    _workspace_id: Option<IgnoredAny>,
}

#[derive(Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
enum ToolChoice {
    Auto {
        #[serde(default, rename = "disable_parallel_tool_use")]
        _disable_parallel_tool_use: Option<IgnoredAny>,
    },
    None,
    Any {
        #[serde(default, rename = "disable_parallel_tool_use")]
        _disable_parallel_tool_use: Option<IgnoredAny>,
    },
    Tool {
        #[serde(rename = "name")]
        _name: IgnoredAny,
        #[serde(default, rename = "disable_parallel_tool_use")]
        _disable_parallel_tool_use: Option<IgnoredAny>,
    },
}

/// Qwen's thinking is on or off; `budget_tokens` and `display` cannot be
/// honored separately and are ignored.
#[derive(Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
enum Thinking {
    Enabled {
        #[serde(rename = "budget_tokens")]
        _budget_tokens: IgnoredAny,
        #[serde(default, rename = "display")]
        _display: Option<IgnoredAny>,
    },
    Adaptive {
        #[serde(default, rename = "display")]
        _display: Option<IgnoredAny>,
    },
    Disabled,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct OutputConfig {
    #[serde(default)]
    effort: Option<String>,
    #[serde(default)]
    format: Option<OutputFormat>,
}

#[derive(Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
enum OutputFormat {
    JsonSchema { schema: Value },
}

/// A validated request with its rebuilt history, ready for a worker.
pub(crate) struct Prepared {
    request: Request,
    messages: Vec<ChatMessage>,
    tools: Vec<Value>,
    controls: GenerationControls,
}

/// An Anthropic-shaped error body.
pub(crate) fn error_body(kind: &str, message: &str) -> Value {
    json!({"type":"error","error":{"type":kind,"message":message}})
}

/// Parses and validates one body; the error is a ready `400` body.
pub(crate) fn prepare(body: &[u8]) -> Result<(String, Prepared), Value> {
    let invalid = |message: &str| error_body("invalid_request_error", message);
    let request: Request =
        serde_json::from_slice(body).map_err(|error| invalid(&error.to_string()))?;
    let prepared = (|| {
        let tools = tools(&request)?;
        let controls = controls(&request, !tools.is_empty())?;
        let messages = messages(&request)?;
        Ok::<_, String>((messages, tools, controls))
    })();
    let (messages, tools, controls) = prepared.map_err(|error| invalid(&error))?;
    Ok((
        request.model.clone(),
        Prepared {
            request,
            messages,
            tools,
            controls,
        },
    ))
}

/// Maps the request's fields onto the protocol-neutral controls.
///
/// `thinking` of `enabled` or `adaptive` turns Qwen's thinking on, and
/// `output_config.effort` then reaches the template as `reasoning_effort`.
/// Sampling follows the Responses mapping (omitted values take the model's
/// `generation_config.json` defaults), with `top_k` honored as sent.
fn controls(request: &Request, has_tools: bool) -> Result<GenerationControls, String> {
    if !request.stop_sequences.is_empty() {
        return Err("stop_sequences are unsupported".into());
    }
    if request
        .container
        .as_ref()
        .is_some_and(|value| !value.is_null())
    {
        return Err("containers are unsupported".into());
    }
    let enable_thinking = matches!(
        request.thinking,
        Some(Thinking::Enabled { .. } | Thinking::Adaptive { .. })
    );
    let config = request.output_config.as_ref();
    let reasoning_effort = match config.and_then(|config| config.effort.as_deref()) {
        None => None,
        Some(effort @ ("low" | "medium" | "high" | "xhigh" | "max")) => {
            enable_thinking.then(|| effort.to_owned())
        }
        Some(other) => return Err(format!("unsupported output_config.effort {other:?}")),
    };
    let json_schema = match config.and_then(|config| config.format.as_ref()) {
        None => None,
        Some(OutputFormat::JsonSchema { schema }) => {
            if !schema.is_object() {
                return Err("output_config.format.schema must be an object".into());
            }
            Some(schema.clone())
        }
    };
    let controls = GenerationControls {
        max_tokens: Some(request.max_tokens),
        sampling: SamplingRequest {
            temperature: request.temperature,
            top_p: request.top_p,
            top_k: request.top_k,
            seed: None,
        },
        top_logprobs: None,
        enable_thinking,
        reasoning_effort,
        json_schema,
        ignore_eos: request.ignore_eos,
        speculation: request.speculation.into(),
    };
    controls.validate(has_tools)?;
    Ok(controls)
}

/// Converts client tools to the template's shape. `tool_choice: none`
/// withholds them from the prompt; forcing a call is unsupported because
/// nothing here can make the model call. Server tools (web search, code
/// execution, editors) have no executor here.
fn tools(request: &Request) -> Result<Vec<Value>, String> {
    match request.tool_choice {
        None | Some(ToolChoice::Auto { .. }) => {}
        Some(ToolChoice::None) => return Ok(Vec::new()),
        Some(ToolChoice::Any { .. } | ToolChoice::Tool { .. }) => {
            return Err("only tool_choice auto or none is supported".into());
        }
    }
    let mut names = HashSet::new();
    request
        .tools
        .iter()
        .map(|tool| {
            if !matches!(tool.get("type").and_then(Value::as_str), None | Some("custom")) {
                return Err("only client tools with an input_schema are supported".to_owned());
            }
            let (Some(name), true) = (tool["name"].as_str(), tool["input_schema"].is_object())
            else {
                return Err("tools require a name and an input_schema object".to_owned());
            };
            if name.is_empty() || !names.insert(name) {
                return Err("tool names must be nonempty and unique".into());
            }
            chat_format::validator(&tool["input_schema"])?;
            Ok(json!({"type":"function","function":{"name":name,"description":tool.get("description").cloned().unwrap_or(json!("")),"parameters":tool["input_schema"]}}))
        })
        .collect()
}

/// Text from a string or text blocks; prompt-cache markers and citations on
/// a block are ignored.
fn text_of(content: &Value) -> Result<String, String> {
    match content {
        Value::String(text) => Ok(text.clone()),
        Value::Array(blocks) => blocks
            .iter()
            .map(
                |block| match (block["type"].as_str(), block["text"].as_str()) {
                    (Some("text"), Some(text)) => Ok(text.to_owned()),
                    _ => Err("only text blocks are supported here".to_owned()),
                },
            )
            .collect(),
        _ => Err("content must be text or an array of text blocks".into()),
    }
}

/// One `tool_result` block as a tool message, matched to its pending call.
fn tool_result(block: &Value, pending: &mut HashMap<&str, &str>) -> Result<ChatMessage, String> {
    let id = block["tool_use_id"]
        .as_str()
        .ok_or("tool_result requires tool_use_id")?;
    let name = pending
        .remove(id)
        .ok_or("tool_result has no matching pending tool_use")?;
    let mut output = match &block["content"] {
        Value::Null => String::new(),
        result => text_of(result)?,
    };
    if block["is_error"] == true {
        // The template has no error flag; keep the signal in the text.
        output.insert_str(0, "Error: ");
    }
    let mut result = message(ChatRole::Tool, output);
    result.name = Some(name.into());
    result.tool_call_id = Some(id.into());
    Ok(result)
}

/// Rebuilds history. A user turn's `tool_result` blocks become tool
/// messages, followed by its text; an assistant turn keeps its text and
/// `tool_use` calls and drops thinking, which the template drops anyway.
fn messages(request: &Request) -> Result<Vec<ChatMessage>, String> {
    let mut messages = Vec::new();
    if let Some(system) = &request.system {
        messages.push(message(ChatRole::System, text_of(system)?));
    }
    let mut pending: HashMap<&str, &str> = HashMap::new();
    let mut seen = HashSet::new();
    for item in &request.messages {
        let role = item["role"].as_str().unwrap_or_default();
        let content = &item["content"];
        match role {
            "system" => messages.push(message(ChatRole::System, text_of(content)?)),
            "user" => {
                let Some(blocks) = content.as_array() else {
                    messages.push(message(ChatRole::User, text_of(content)?));
                    continue;
                };
                let mut text = String::new();
                for block in blocks {
                    match block["type"].as_str() {
                        Some("text") => {
                            text.push_str(
                                block["text"].as_str().ok_or("text block requires text")?,
                            );
                        }
                        Some("tool_result") => messages.push(tool_result(block, &mut pending)?),
                        _ => {
                            return Err(
                                "user content supports only text and tool_result blocks".into()
                            );
                        }
                    }
                }
                if !text.is_empty() {
                    messages.push(message(ChatRole::User, text));
                }
            }
            "assistant" => {
                let Some(blocks) = content.as_array() else {
                    messages.push(message(ChatRole::Assistant, text_of(content)?));
                    continue;
                };
                let mut turn = message(ChatRole::Assistant, String::new());
                for block in blocks {
                    match block["type"].as_str() {
                        Some("text") => turn
                            .content
                            .push_str(block["text"].as_str().ok_or("text block requires text")?),
                        Some("thinking" | "redacted_thinking") => {}
                        Some("tool_use") => {
                            let (Some(id), Some(name), true) = (
                                block["id"].as_str().filter(|id| !id.is_empty()),
                                block["name"].as_str(),
                                block["input"].is_object(),
                            ) else {
                                return Err(
                                    "tool_use requires an id, a name and an input object".into()
                                );
                            };
                            if pending.insert(id, name).is_some() || !seen.insert(id) {
                                return Err("tool_use IDs must be unique".into());
                            }
                            turn.tool_calls.push(ChatToolCall {
                                name: name.into(),
                                arguments: block["input"].clone(),
                            });
                        }
                        _ => return Err(
                            "assistant content supports only text, thinking and tool_use blocks"
                                .into(),
                        ),
                    }
                }
                messages.push(turn);
            }
            _ => return Err(format!("unsupported message role {role:?}")),
        }
    }
    match messages.last().map(|message| message.role) {
        Some(ChatRole::User | ChatRole::Tool) => {}
        Some(ChatRole::Assistant) => {
            return Err("a final assistant message (prefill) is unsupported".into());
        }
        _ => return Err("messages must end with a user message".into()),
    }
    Ok(messages)
}

fn stop_reason(turn: &AssistantTurn) -> &'static str {
    if !turn.calls.is_empty() {
        "tool_use"
    } else if turn.complete {
        "end_turn"
    } else {
        "max_tokens"
    }
}

/// Anthropic partitions the prompt into uncached, cache-read and newly
/// cache-created tokens, without counting copied prefix positions twice.
fn usage(generated: &ChatGeneration) -> Value {
    let prompt = generated.metrics.prompt_tokens;
    let cached = generated.metrics.cached_prompt_tokens.min(prompt);
    // Producers report newly created positions. Clamp inconsistent metrics
    // to the remaining prompt positions without subtracting cache hits again.
    let created = generated.metrics.cache_write_tokens.min(prompt - cached);
    let input = prompt - cached - created;
    json!({"input_tokens":input,"output_tokens":generated.generated_token_ids.len(),"cache_read_input_tokens":cached,"cache_creation_input_tokens":created})
}

/// The content blocks of a finished turn, in order.
fn blocks(turn: &AssistantTurn, id: &str) -> Vec<Value> {
    let mut blocks = Vec::new();
    if !turn.reasoning.is_empty() {
        // Qwen's reasoning is not signed; the signature is empty.
        blocks.push(json!({"type":"thinking","thinking":turn.reasoning,"signature":""}));
    }
    if !turn.text.is_empty() || turn.calls.is_empty() {
        blocks.push(json!({"type":"text","text":turn.text}));
    }
    for (index, call) in turn.calls.iter().enumerate() {
        blocks.push(json!({"type":"tool_use","id":format!("toolu_{id}_{index}"),"name":call.name,"input":call.arguments}));
    }
    blocks
}

fn message_value(
    prepared: &Prepared,
    generated: &ChatGeneration,
    id: &str,
) -> Result<Value, String> {
    let turn = assistant_turn(&prepared.tools, generated)?;
    Ok(
        json!({"id":format!("msg_{id}"),"type":"message","role":"assistant","model":prepared.request.model,"content":blocks(&turn, id),"stop_reason":stop_reason(&turn),"stop_sequence":null,"usage":usage(generated),"metallix":{"sampling":generated.sampling,"metrics":generated.metrics}}),
    )
}

fn generation_error(connection: Connection, error: &ChatGenerationError) {
    match error {
        ChatGenerationError::DeadlineExceeded => json_response(
            connection,
            408,
            &error_body("timeout_error", "generation time budget exceeded"),
        ),
        ChatGenerationError::Message(message) => {
            json_response(
                connection,
                400,
                &error_body("invalid_request_error", message),
            );
        }
    }
}

/// Answers one prepared request, as JSON or as an event stream.
pub(crate) fn respond(
    mut connection: Connection,
    prepared: &Prepared,
    session: &mut dyn ChatBackend,
    id: &str,
    generation_timeout: Duration,
) -> Result<(), String> {
    // The router-owned tenant salt keeps prefix-cache reuse within a tenant.
    let cache_salt = connection.cache_salt().map(str::to_owned);
    let request = ChatRequest {
        cache_salt: cache_salt.as_deref(),
        ..prepared
            .controls
            .request(&prepared.messages, &prepared.tools)
    };
    if !prepared.request.stream {
        let request_id = connection.request_id().map(str::to_owned);
        let result = session
            .generate_with_timeout(
                request,
                generation_timeout,
                &mut connection.stop_when_gone(),
            )
            .and_then(|generated| {
                record_usage(&generated);
                let mut value = message_value(prepared, &generated, id)
                    .map_err(ChatGenerationError::Message)?;
                echo_request_id(&mut value, request_id.as_deref());
                Ok(value)
            });
        match result {
            Ok(value) => json_response(connection, 200, &value),
            Err(error) => generation_error(connection, &error),
        }
        return Ok(());
    }
    stream(
        connection,
        prepared,
        request,
        session,
        id,
        generation_timeout,
    )
}

fn event(sse: &mut LazySse, data: &Value) -> Result<(), String> {
    sse.event(data["type"].as_str(), data)
}

/// The thinking and text blocks a stream has opened, in order.
#[derive(Default)]
struct LiveBlocks {
    /// The open block's index and whether it holds thinking.
    open: Option<(usize, bool)>,
    count: usize,
}

impl LiveBlocks {
    /// Streams one delta, closing the open block when the kind changes.
    fn push(&mut self, sse: &mut LazySse, delta: &TurnDelta) -> Result<(), String> {
        let (thinking, body) = match delta {
            TurnDelta::Reasoning(text) => (true, json!({"type":"thinking_delta","thinking":text})),
            TurnDelta::Text(text) => (false, json!({"type":"text_delta","text":text})),
            TurnDelta::Held => return sse.keepalive(),
        };
        let index = match self.open {
            Some((index, open_thinking)) if open_thinking == thinking => index,
            _ => {
                self.close(sse)?;
                let index = self.count;
                self.count += 1;
                let block = if thinking {
                    json!({"type":"thinking","thinking":"","signature":""})
                } else {
                    json!({"type":"text","text":""})
                };
                event(
                    sse,
                    &json!({"type":"content_block_start","index":index,"content_block":block}),
                )?;
                self.open = Some((index, thinking));
                index
            }
        };
        event(
            sse,
            &json!({"type":"content_block_delta","index":index,"delta":body}),
        )
    }

    /// Closes the open block and returns how many blocks were streamed.
    fn close(&mut self, sse: &mut LazySse) -> Result<usize, String> {
        if let Some((index, _)) = self.open.take() {
            event(sse, &json!({"type":"content_block_stop","index":index}))?;
        }
        Ok(self.count)
    }
}

#[allow(clippy::too_many_lines, reason = "one ordered event lifecycle")]
fn stream(
    connection: Connection,
    prepared: &Prepared,
    request: ChatRequest<'_>,
    session: &mut dyn ChatBackend,
    id: &str,
    generation_timeout: Duration,
) -> Result<(), String> {
    // The prompt is counted only once generation ends, so the opening usage
    // is zero and `message_delta` carries the real counts.
    let start = json!({"type":"message_start","message":{"id":format!("msg_{id}"),"type":"message","role":"assistant","model":prepared.request.model,"content":[],"stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":0,"output_tokens":0}}});
    // Thinking and text stream as blocks while they settle; the session
    // holds back markup, and tool calls arrive whole once the turn ends.
    let mut sse = LazySse::new(connection);
    let mut started = false;
    let mut blocks = LiveBlocks::default();
    let generated = session.generate_with_timeout(request, generation_timeout, &mut |delta| {
        if delta == TurnDelta::Held {
            return sse.keepalive();
        }
        if !started {
            started = true;
            event(&mut sse, &start)?;
        }
        blocks.push(&mut sse, &delta)
    });
    let failure =
        |sse: &mut LazySse, kind: &str, message: &str| event(sse, &error_body(kind, message));
    let generated = match generated {
        Ok(generated) => generated,
        Err(error) => {
            if let Some(connection) = sse.take_unopened() {
                generation_error(connection, &error);
                return Ok(());
            }
            return match error {
                ChatGenerationError::DeadlineExceeded => {
                    failure(&mut sse, "timeout_error", "generation time budget exceeded")
                }
                ChatGenerationError::Message(message) => failure(&mut sse, "api_error", &message),
            };
        }
    };
    record_usage(&generated);
    let turn = match assistant_turn(&prepared.tools, &generated) {
        Ok(turn) => turn,
        Err(error) => return failure(&mut sse, "api_error", &error),
    };
    if !started {
        event(&mut sse, &start)?;
    }
    // The blocks streamed so far are exactly the leading thinking and text
    // blocks of the final message; the rest follow whole.
    let final_blocks = self::blocks(&turn, id);
    let streamed = blocks.close(&mut sse)?;
    for (index, block) in final_blocks.into_iter().enumerate().skip(streamed) {
        let (opened, delta) = match block["type"].as_str() {
            Some("thinking") => (
                json!({"type":"thinking","thinking":"","signature":""}),
                json!({"type":"thinking_delta","thinking":block["thinking"]}),
            ),
            Some("tool_use") => (
                json!({"type":"tool_use","id":block["id"],"name":block["name"],"input":{}}),
                json!({"type":"input_json_delta","partial_json":block["input"].to_string()}),
            ),
            _ => (
                json!({"type":"text","text":""}),
                json!({"type":"text_delta","text":block["text"]}),
            ),
        };
        event(
            &mut sse,
            &json!({"type":"content_block_start","index":index,"content_block":opened}),
        )?;
        event(
            &mut sse,
            &json!({"type":"content_block_delta","index":index,"delta":delta}),
        )?;
        event(
            &mut sse,
            &json!({"type":"content_block_stop","index":index}),
        )?;
    }
    event(
        &mut sse,
        &json!({"type":"message_delta","delta":{"stop_reason":stop_reason(&turn),"stop_sequence":null},"usage":usage(&generated)}),
    )?;
    event(&mut sse, &json!({"type":"message_stop"}))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        http_transport::CLIENT_POLL_DELTAS,
        sse::test_support::{
            SCRIPTED_EOS, Scripted, abandon, events, exchange, exchange_with, json_body,
        },
    };

    fn prepare_body(body: &Value) -> Result<Prepared, Value> {
        prepare(body.to_string().as_bytes()).map(|(_, prepared)| prepared)
    }

    fn with(extra: &Value) -> Value {
        let mut body =
            json!({"model":"m","max_tokens":64,"messages":[{"role":"user","content":"hi"}]});
        body.as_object_mut()
            .unwrap()
            .extend(extra.as_object().unwrap().clone());
        body
    }

    const TOOL: &str = r#"{"name":"read_file","description":"Read a file","input_schema":{"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}}"#;

    #[test]
    fn rejects_unknown_fields_and_unsupported_semantics() {
        let tool: Value = serde_json::from_str(TOOL).unwrap();
        for extra in [
            json!({"frobnicate":1}),
            json!({"thinking":{"type":"enabled","budget_tokens":2048,"style":"x"}}),
            json!({"output_config":{"effort":"maximum"}}),
            json!({"output_config":{"format":{"type":"json_schema","schema":true}}}),
            json!({"stop_sequences":["\n\nHuman:"]}),
            json!({"container":"c1"}),
            json!({"top_k":0}),
            json!({"temperature":2.5}),
            json!({"tool_choice":{"type":"any"},"tools":[tool.clone()]}),
            json!({"tool_choice":{"type":"tool","name":"read_file"},"tools":[tool.clone()]}),
            json!({"tools":[{"type":"web_search_20250305","name":"web_search"}]}),
            json!({"tools":[{"name":"x"}]}),
            json!({"messages":[{"role":"user","content":[{"type":"image","source":{"type":"url","url":"x"}}]}]}),
            json!({"messages":[{"role":"user","content":"hi"},{"role":"assistant","content":"Sure,"}]}),
            json!({"messages":[{"role":"user","content":[{"type":"tool_result","tool_use_id":"nope","content":"x"}]}]}),
            json!({"messages":[{"role":"user","content":"x"},{"role":"assistant","content":[{"type":"tool_use","id":"toolu_1","name":"read_file","input":{}}]}]}),
        ] {
            assert!(prepare_body(&with(&extra)).is_err(), "accepted {extra}");
        }
        assert!(
            prepare_body(&json!({"model":"m","messages":[{"role":"user","content":"hi"}]}))
                .is_err(),
            "max_tokens is required"
        );
        let error = prepare_body(&with(&json!({"frobnicate":1}))).err().unwrap();
        assert_eq!(error["type"], "error");
        assert_eq!(error["error"]["type"], "invalid_request_error");
    }

    #[test]
    fn unanswered_tool_use_in_history_renders_as_given() {
        let prepared = prepare_body(&with(&json!({"messages":[
            {"role":"user","content":"read"},
            {"role":"assistant","content":[
                {"type":"tool_use","id":"toolu_1","name":"read_file","input":{"path":"a"}},
                {"type":"tool_use","id":"toolu_2","name":"read_file","input":{"path":"b"}}
            ]},
            {"role":"user","content":[{"type":"tool_result","tool_use_id":"toolu_1","content":"A"}]}
        ]})))
        .unwrap();
        let roles: Vec<_> = prepared
            .messages
            .iter()
            .map(|message| message.role)
            .collect();
        assert_eq!(roles, [ChatRole::User, ChatRole::Assistant, ChatRole::Tool]);
        assert_eq!(prepared.messages[1].tool_calls.len(), 2);
        assert_eq!(
            prepared.messages[2].tool_call_id.as_deref(),
            Some("toolu_1")
        );
    }

    #[test]
    fn accepts_claude_code_shapes_and_maps_controls() {
        let mut tool: Value = serde_json::from_str(TOOL).unwrap();
        tool["cache_control"] = json!({"type":"ephemeral"});
        let prepared = prepare_body(&with(&json!({
            "system": [{"type":"text","text":"You are terse.","cache_control":{"type":"ephemeral","ttl":"1h"}}],
            "messages": [
                {"role":"user","content":[{"type":"text","text":"read a and b"}]},
                {"role":"assistant","content":[
                    {"type":"thinking","thinking":"plan","signature":"sig"},
                    {"type":"text","text":"Reading."},
                    {"type":"tool_use","id":"toolu_1","name":"read_file","input":{"path":"a"}},
                    {"type":"tool_use","id":"toolu_2","name":"read_file","input":{"path":"b"}}
                ]},
                {"role":"user","content":[
                    {"type":"tool_result","tool_use_id":"toolu_2","content":"B"},
                    {"type":"tool_result","tool_use_id":"toolu_1","content":[{"type":"text","text":"A"}],"is_error":true,"cache_control":{"type":"ephemeral"}},
                    {"type":"text","text":"now summarize"}
                ]}
            ],
            "tools": [tool],
            "tool_choice": {"type":"auto","disable_parallel_tool_use":false},
            "metadata": {"user_id":"u"},
            "stream": true,
            "thinking": {"type":"enabled","budget_tokens":4096},
            "output_config": {"effort":"high"},
            "temperature": 0.7, "top_p": 0.9, "top_k": 40,
            "service_tier": "auto",
            "context_management": {"edits":[]},
            "cache_control": {"type":"ephemeral"}
        })))
        .unwrap();
        let roles: Vec<_> = prepared
            .messages
            .iter()
            .map(|message| message.role)
            .collect();
        assert_eq!(
            roles,
            [
                ChatRole::System,
                ChatRole::User,
                ChatRole::Assistant,
                ChatRole::Tool,
                ChatRole::Tool,
                ChatRole::User
            ]
        );
        assert_eq!(prepared.messages[2].content, "Reading.");
        assert_eq!(prepared.messages[2].tool_calls.len(), 2);
        assert_eq!(prepared.messages[4].content, "Error: A");
        assert_eq!(
            prepared.messages[4].tool_call_id.as_deref(),
            Some("toolu_1")
        );
        assert_eq!(
            prepared.tools[0]["function"]["parameters"]["required"],
            json!(["path"])
        );
        let controls = &prepared.controls;
        assert_eq!(controls.max_tokens, Some(64));
        assert!(controls.enable_thinking);
        assert_eq!(controls.reasoning_effort.as_deref(), Some("high"));
        assert_eq!(
            controls.sampling,
            SamplingRequest {
                temperature: Some(0.7),
                top_p: Some(0.9),
                top_k: Some(40),
                seed: None
            }
        );

        let mapped = |extra: Value| prepare_body(&with(&extra)).unwrap().controls;
        assert!(!mapped(json!({"thinking":{"type":"disabled"}})).enable_thinking);
        assert!(mapped(json!({"thinking":{"type":"adaptive"}})).enable_thinking);
        // Effort reaches the template only with thinking on.
        assert_eq!(
            mapped(json!({"output_config":{"effort":"low"}})).reasoning_effort,
            None
        );
        assert!(prepare_body(&with(&json!({"tool_choice":{"type":"none"},"tools":[serde_json::from_str::<Value>(TOOL).unwrap()]}))).unwrap().tools.is_empty());
        if cfg!(feature = "structured-output") {
            let schema = json!({"type":"object","properties":{"ok":{"type":"boolean"}}});
            assert_eq!(
                mapped(json!({"output_config":{"format":{"type":"json_schema","schema":schema}}}))
                    .json_schema,
                Some(schema)
            );
        }
    }

    fn run(body: &Value, backend: &mut Scripted) -> String {
        exchange("/v1/messages", &body.to_string(), |connection, body| {
            let (_, prepared) = prepare(body).unwrap();
            respond(connection, &prepared, backend, "t", Duration::from_secs(60)).unwrap();
        })
    }

    #[test]
    fn answers_text_thinking_and_tool_use_in_both_shapes() {
        let mut backend = Scripted::new("Hello there");
        let (status, body) = json_body(&run(&with(&json!({})), &mut backend));
        assert_eq!(status, "HTTP/1.1 200 OK");
        assert_eq!(body["type"], "message");
        assert_eq!(body["id"], "msg_t");
        assert_eq!(
            body["content"],
            json!([{"type":"text","text":"Hello there"}])
        );
        assert_eq!(body["stop_reason"], "end_turn");
        assert_eq!(body["usage"]["input_tokens"], 1);
        assert_eq!(body["usage"]["output_tokens"], 2);

        let tool: Value = serde_json::from_str(TOOL).unwrap();
        let text = r#"<think>
need the file
</think>

Let me look.<tool_call>{"name":"read_file","arguments":{"path":"README.md"}}</tool_call>"#;
        let mut backend = Scripted::new(text);
        let body = with(&json!({"tools":[tool.clone()],"thinking":{"type":"adaptive"}}));
        let (_, message) = json_body(&run(&body, &mut backend));
        assert_eq!(message["stop_reason"], "tool_use");
        assert_eq!(
            message["content"][0],
            json!({"type":"thinking","thinking":"need the file","signature":""})
        );
        assert_eq!(
            message["content"][1],
            json!({"type":"text","text":"Let me look."})
        );
        assert_eq!(message["content"][2]["type"], "tool_use");
        assert_eq!(message["content"][2]["id"], "toolu_t_0");
        assert_eq!(message["content"][2]["input"], json!({"path":"README.md"}));

        let mut streamed = body;
        streamed["stream"] = json!(true);
        let mut backend = Scripted::new(text);
        let frames = events(&run(&streamed, &mut backend));
        // Thinking and text stream as many deltas; the tool call as one.
        let names: Vec<_> = frames
            .iter()
            .map(|(name, _)| name.clone().unwrap())
            .collect();
        let mut kinds = names.clone();
        kinds.dedup();
        assert_eq!(
            kinds,
            [
                "message_start",
                "content_block_start",
                "content_block_delta",
                "content_block_stop",
                "content_block_start",
                "content_block_delta",
                "content_block_stop",
                "content_block_start",
                "content_block_delta",
                "content_block_stop",
                "message_delta",
                "message_stop",
            ]
        );
        let data: Vec<Value> = frames
            .iter()
            .map(|(_, data)| serde_json::from_str(data).unwrap())
            .collect();
        for (name, value) in names.iter().zip(&data) {
            assert_eq!(value["type"], name.as_str(), "event name matches its data");
        }
        let starts: Vec<&Value> = data
            .iter()
            .filter(|value| value["type"] == "content_block_start")
            .collect();
        assert_eq!(starts[0]["content_block"]["type"], "thinking");
        assert_eq!(starts[1]["content_block"]["type"], "text");
        assert_eq!(
            starts[2]["content_block"],
            json!({"type":"tool_use","id":"toolu_t_0","name":"read_file","input":{}})
        );
        let streamed = |index: u64, field: &str| -> String {
            data.iter()
                .filter(|value| value["type"] == "content_block_delta" && value["index"] == index)
                .map(|value| value["delta"][field].as_str().unwrap().to_owned())
                .collect()
        };
        assert_eq!(streamed(0, "thinking"), "need the file");
        assert_eq!(streamed(1, "text"), "Let me look.");
        let partial: Value = serde_json::from_str(&streamed(2, "partial_json")).unwrap();
        assert_eq!(partial, json!({"path":"README.md"}));
        let end = data
            .iter()
            .find(|value| value["type"] == "message_delta")
            .unwrap();
        assert_eq!(end["delta"]["stop_reason"], "tool_use");
        assert_eq!(end["usage"]["input_tokens"], 1);
    }

    #[test]
    fn ignore_eos_runs_to_max_tokens_past_an_early_eos() {
        let early = [7, SCRIPTED_EOS];
        let mut backend = Scripted::emitting(&early);
        let (_, body) = json_body(&run(&with(&json!({"max_tokens":8})), &mut backend));
        assert_eq!(body["stop_reason"], "end_turn");
        assert_eq!(body["usage"]["output_tokens"], 2);

        let mut backend = Scripted::emitting(&early);
        let ignoring = with(&json!({"max_tokens":8,"ignore_eos":true}));
        let (_, body) = json_body(&run(&ignoring, &mut backend));
        assert_eq!(body["stop_reason"], "max_tokens");
        assert_eq!(body["content"][0]["text"], "a<eos>".repeat(4));
        assert_eq!(body["usage"]["output_tokens"], 8);
    }

    #[test]
    fn cached_prompt_tokens_are_reported_apart_from_input_tokens() {
        for (cached, written, input, created) in [
            (4, 5, 1, 5),
            (0, 9, 1, 9),
            (4, 0, 6, 0),
            (4, 2, 4, 2),
            (4, 99, 0, 6),
        ] {
            for stream in [false, true] {
                let mut backend = Scripted::new("hi");
                backend.prompt_tokens = 10;
                backend.cached_prompt_tokens = cached;
                backend.cache_write_tokens = written;
                let wire = run(&with(&json!({"stream":stream})), &mut backend);
                let usage = if stream {
                    let (_, data) = events(&wire)
                        .into_iter()
                        .find(|(name, _)| name.as_deref() == Some("message_delta"))
                        .unwrap();
                    serde_json::from_str::<Value>(&data).unwrap()["usage"].clone()
                } else {
                    json_body(&wire).1["usage"].clone()
                };
                assert_eq!(usage["input_tokens"], input, "stream={stream}");
                assert_eq!(usage["cache_read_input_tokens"], cached, "stream={stream}");
                assert_eq!(
                    usage["cache_creation_input_tokens"], created,
                    "stream={stream}"
                );
                assert_eq!(
                    usage["input_tokens"].as_u64().unwrap()
                        + usage["cache_read_input_tokens"].as_u64().unwrap()
                        + usage["cache_creation_input_tokens"].as_u64().unwrap(),
                    10,
                    "usage partitions the prompt: stream={stream}"
                );
            }
        }
        let mut backend = Scripted::new("hi");
        backend.prompt_tokens = 3;
        backend.cached_prompt_tokens = 5;
        let (status, body) = json_body(&run(&with(&json!({})), &mut backend));
        assert_eq!(
            status, "HTTP/1.1 200 OK",
            "inconsistent metrics do not panic"
        );
        assert_eq!(body["usage"]["input_tokens"], 0);
        assert_eq!(body["usage"]["cache_read_input_tokens"], 3);
        assert_eq!(body["usage"]["cache_creation_input_tokens"], 0);
    }

    /// A reply written only at the end still notices a client that left:
    /// generation stops within two client checks instead of running out.
    #[test]
    fn a_client_that_leaves_stops_a_non_streamed_generation() {
        let mut backend = Scripted::new(&"word ".repeat(400));
        backend.piece_delay = Duration::from_millis(1);
        abandon(
            "/v1/messages",
            &with(&json!({})).to_string(),
            |connection, body| {
                let (_, prepared) = prepare(body).unwrap();
                respond(
                    connection,
                    &prepared,
                    &mut backend,
                    "t",
                    Duration::from_secs(60),
                )
                .unwrap();
            },
        );
        assert!(
            backend.deltas <= 2 * CLIENT_POLL_DELTAS,
            "{} pieces generated after the client left",
            backend.deltas
        );
    }

    #[test]
    fn the_router_cache_salt_reaches_generation() {
        for stream in [false, true] {
            let mut backend = Scripted::new("hi");
            let body = with(&json!({"stream":stream})).to_string();
            exchange_with(
                "/v1/messages",
                "X-Metallix-Cache-Salt: tenant-1\r\n",
                &body,
                |connection, body| {
                    let (_, prepared) = prepare(body).unwrap();
                    respond(
                        connection,
                        &prepared,
                        &mut backend,
                        "t",
                        Duration::from_secs(60),
                    )
                    .unwrap();
                },
            );
            assert_eq!(backend.salt.as_deref(), Some("tenant-1"), "stream={stream}");
        }
        let mut backend = Scripted::new("hi");
        run(&with(&json!({})), &mut backend);
        assert_eq!(backend.salt, None, "no salt is invented");
    }

    #[test]
    fn streams_text_live_and_fails_before_output_with_http_status() {
        let mut backend = Scripted::new("abc");
        let frames = events(&run(&with(&json!({"stream":true})), &mut backend));
        let text: String = frames
            .iter()
            .filter_map(|(_, data)| serde_json::from_str::<Value>(data).ok())
            .filter(|event| event["type"] == "content_block_delta")
            .map(|event| event["delta"]["text"].as_str().unwrap().to_owned())
            .collect();
        assert_eq!(text, "abc");
        assert!(backend.deltas > 1);
        assert_eq!(frames.first().unwrap().0.as_deref(), Some("message_start"));
        assert_eq!(frames.last().unwrap().0.as_deref(), Some("message_stop"));

        for stream in [false, true] {
            let mut backend =
                Scripted::failing("chat requires prompt_tokens + max_tokens <= 16384");
            let (status, body) = json_body(&run(&with(&json!({"stream":stream})), &mut backend));
            assert_eq!(status, "HTTP/1.1 400 Bad Request", "stream={stream}");
            assert_eq!(body["error"]["type"], "invalid_request_error");
        }
    }

    /// Opt-in against a real local Qwen3 checkpoint named by
    /// `METALLIX_QWEN_MODEL`.
    #[test]
    #[ignore = "requires METALLIX_QWEN_MODEL and a local Apple-Silicon Metal checkpoint"]
    fn real_qwen_answers_streams_and_uses_tools() {
        use crate::chat_generation::{ChatSession, ResidentChatLimits};
        let model = std::env::var_os("METALLIX_QWEN_MODEL").expect("set METALLIX_QWEN_MODEL");
        let mut session = ChatSession::load(
            std::path::Path::new(&model),
            ResidentChatLimits::from_mib(16_384, 8_192),
        )
        .expect("load local Qwen checkpoint");
        let run = |body: &Value, session: &mut ChatSession| {
            exchange("/v1/messages", &body.to_string(), |connection, body| {
                let (_, prepared) = prepare(body).unwrap();
                respond(
                    connection,
                    &prepared,
                    session,
                    "q",
                    Duration::from_secs(600),
                )
                .unwrap();
            })
        };
        let body = json!({"model":"q","max_tokens":16,"temperature":0,"messages":[{"role":"user","content":"Say hello."}]});
        let (status, plain) = json_body(&run(&body, &mut session));
        assert_eq!(status, "HTTP/1.1 200 OK", "{plain}");
        let text = plain["content"][0]["text"].as_str().unwrap().to_owned();
        assert!(!text.is_empty());

        let mut streamed = body.clone();
        streamed["stream"] = json!(true);
        let joined: String = events(&run(&streamed, &mut session))
            .iter()
            .filter_map(|(_, data)| serde_json::from_str::<Value>(data).ok())
            .filter(|event| event["type"] == "content_block_delta")
            .map(|event| event["delta"]["text"].as_str().unwrap().to_owned())
            .collect();
        assert_eq!(joined, text);

        let tools = json!([{"name":"get_weather","description":"Current weather for a city","input_schema":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}]);
        let body = json!({"model":"q","max_tokens":96,"temperature":0,"tools":tools,"messages":[{"role":"user","content":"What is the weather in Paris? Use the tool."}]});
        let (_, called) = json_body(&run(&body, &mut session));
        assert_eq!(called["stop_reason"], "tool_use", "{called}");
        let call = called["content"]
            .as_array()
            .unwrap()
            .iter()
            .find(|block| block["type"] == "tool_use")
            .unwrap();
        assert_eq!(call["name"], "get_weather");
        assert!(call["input"]["city"].is_string());

        // Round trip: the call and its result go back; the model answers.
        let follow = json!({"model":"q","max_tokens":64,"temperature":0,"tools":tools,"messages":[
            {"role":"user","content":"What is the weather in Paris? Use the tool."},
            {"role":"assistant","content":called["content"]},
            {"role":"user","content":[{"type":"tool_result","tool_use_id":call["id"],"content":"18 C and sunny"}]}
        ]});
        let (status, answered) = json_body(&run(&follow, &mut session));
        assert_eq!(status, "HTTP/1.1 200 OK", "{answered}");
    }
}
