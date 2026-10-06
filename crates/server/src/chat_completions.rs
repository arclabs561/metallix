//! `OpenAI` Chat Completions over the shared generation controls: request
//! validation, message and tool reconstruction, and the JSON or SSE response.

use std::{
    collections::{HashMap, HashSet},
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use chat_format::TurnDelta;
use serde::{Deserialize, de::IgnoredAny};
use serde_json::{Map, Value, json};

use crate::{
    chat_cli::message,
    chat_generation::{
        ChatBackend, ChatGeneration, ChatGenerationError, ChatMessage, ChatRequest, ChatRole,
        ChatToolCall, GenerationControls, SamplingRequest,
    },
    http_transport::Connection,
    responses::{
        AssistantTurn, assistant_turn, echo_request_id, json_response, logprobs_value,
        record_usage, thinking_for_effort,
    },
    sse::LazySse,
};

/// One `POST /v1/chat/completions` body. Unknown fields are rejected; fields
/// the `OpenAI` SDKs and agent clients send with no generation meaning here
/// are accepted and ignored by name.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Request {
    pub(crate) model: String,
    /// Metallix extension: "auto" (default), "on" or "off" for
    /// prompt-lookup speculative decoding.
    #[serde(default)]
    speculation: crate::chat_generation::SpeculationField,
    messages: Vec<Value>,
    #[serde(default)]
    tools: Vec<Value>,
    #[serde(default)]
    tool_choice: Option<Value>,
    #[serde(default)]
    stream: bool,
    #[serde(default)]
    stream_options: Option<StreamOptions>,
    #[serde(default)]
    max_tokens: Option<u32>,
    #[serde(default)]
    max_completion_tokens: Option<u32>,
    #[serde(default)]
    temperature: Option<f64>,
    #[serde(default)]
    top_p: Option<f64>,
    #[serde(default)]
    seed: Option<u64>,
    /// Metallix extension, as in vLLM: generate past end-of-turn up to the
    /// output limit, for equal-length benchmark runs.
    #[serde(default)]
    ignore_eos: bool,
    #[serde(default)]
    logprobs: Option<bool>,
    #[serde(default)]
    top_logprobs: Option<u8>,
    #[serde(default)]
    reasoning_effort: Option<String>,
    #[serde(default)]
    response_format: Option<ResponseFormat>,
    #[serde(default)]
    n: Option<u32>,
    #[serde(default)]
    stop: Option<Value>,
    #[serde(default)]
    frequency_penalty: Option<f64>,
    #[serde(default)]
    presence_penalty: Option<f64>,
    #[serde(default)]
    logit_bias: Option<Map<String, Value>>,
    #[serde(default)]
    store: Option<bool>,
    #[serde(default)]
    modalities: Option<Vec<String>>,
    // Accepted and ignored: attribution, routing and caching hints. A turn
    // may already carry several calls, so `parallel_tool_calls` changes
    // nothing, and the model has no verbosity control.
    #[serde(default, rename = "parallel_tool_calls")]
    _parallel_tool_calls: Option<IgnoredAny>,
    #[serde(default, rename = "user")]
    _user: Option<IgnoredAny>,
    #[serde(default, rename = "metadata")]
    _metadata: Option<IgnoredAny>,
    #[serde(default, rename = "safety_identifier")]
    _safety_identifier: Option<IgnoredAny>,
    #[serde(default, rename = "service_tier")]
    _service_tier: Option<IgnoredAny>,
    #[serde(default, rename = "prompt_cache_key")]
    _prompt_cache_key: Option<IgnoredAny>,
    #[serde(default, rename = "prompt_cache_retention")]
    _prompt_cache_retention: Option<IgnoredAny>,
    #[serde(default, rename = "prompt_cache_options")]
    _prompt_cache_options: Option<IgnoredAny>,
    #[serde(default, rename = "verbosity")]
    _verbosity: Option<IgnoredAny>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct StreamOptions {
    #[serde(default)]
    include_usage: bool,
    /// Ignored: no stream here is padded for obfuscation.
    #[serde(default, rename = "include_obfuscation")]
    _include_obfuscation: Option<IgnoredAny>,
}

#[derive(Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
enum ResponseFormat {
    Text,
    JsonObject,
    JsonSchema { json_schema: JsonSchemaFormat },
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct JsonSchemaFormat {
    #[serde(rename = "name")]
    _name: IgnoredAny,
    #[serde(default)]
    schema: Option<Value>,
    #[serde(default, rename = "description")]
    _description: Option<IgnoredAny>,
    /// Ignored: the grammar mask always enforces the schema.
    #[serde(default, rename = "strict")]
    _strict: Option<IgnoredAny>,
}

/// A validated request with its rebuilt history, ready for a worker.
pub(crate) struct Prepared {
    request: Request,
    messages: Vec<ChatMessage>,
    tools: Vec<Value>,
    controls: GenerationControls,
}

/// An `OpenAI`-shaped error body.
pub(crate) fn error_body(message: &str) -> Value {
    json!({"error":{"message":message,"type":"invalid_request_error","param":null,"code":null}})
}

/// Parses and validates one body; the error is a ready `400` body.
pub(crate) fn prepare(body: &[u8]) -> Result<(String, Prepared), Value> {
    let request: Request =
        serde_json::from_slice(body).map_err(|error| error_body(&error.to_string()))?;
    let prepared = (|| {
        let tools = tools(&request)?;
        let controls = controls(&request, !tools.is_empty())?;
        let messages = messages(&request)?;
        Ok::<_, String>((messages, tools, controls))
    })();
    let (messages, tools, controls) = prepared.map_err(|error| error_body(&error))?;
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

/// Maps the request's fields onto the protocol-neutral controls. Sampling,
/// effort and format follow the Responses mapping; see
/// [`crate::responses::controls`].
fn controls(request: &Request, has_tools: bool) -> Result<GenerationControls, String> {
    if request.n.is_some_and(|n| n != 1) {
        return Err("only n=1 is supported".into());
    }
    let stops = match &request.stop {
        None | Some(Value::Null) => 0,
        Some(Value::String(_)) => 1,
        Some(Value::Array(stops)) => stops.len(),
        Some(_) => return Err("stop must be a string or an array of strings".into()),
    };
    if stops > 0 {
        return Err("stop sequences are unsupported".into());
    }
    if [request.frequency_penalty, request.presence_penalty]
        .into_iter()
        .flatten()
        .any(|penalty| penalty != 0.0)
    {
        return Err("frequency and presence penalties are unsupported".into());
    }
    if request
        .logit_bias
        .as_ref()
        .is_some_and(|bias| !bias.is_empty())
    {
        return Err("logit_bias is unsupported".into());
    }
    if request.store == Some(true) {
        return Err("completion storage is unsupported; use store=false".into());
    }
    if request
        .modalities
        .as_ref()
        .is_some_and(|modalities| modalities.iter().any(|modality| modality != "text"))
    {
        return Err("only text output is supported".into());
    }
    let max_tokens = match (request.max_completion_tokens, request.max_tokens) {
        (Some(left), Some(right)) if left != right => {
            return Err("max_completion_tokens and max_tokens disagree".into());
        }
        (Some(limit), _) | (None, Some(limit)) => Some(limit),
        (None, None) => None,
    };
    let top_logprobs = match (request.logprobs, request.top_logprobs) {
        (Some(true), top) => Some(top.unwrap_or(0)),
        (_, None) => None,
        (_, Some(_)) => return Err("top_logprobs requires logprobs: true".into()),
    };
    let (enable_thinking, reasoning_effort) =
        thinking_for_effort(request.reasoning_effort.as_deref())
            .map_err(|effort| format!("unsupported reasoning_effort {effort:?}"))?;
    let json_schema = match &request.response_format {
        None | Some(ResponseFormat::Text) => None,
        Some(ResponseFormat::JsonObject) => Some(json!({"type":"object"})),
        Some(ResponseFormat::JsonSchema { json_schema }) => {
            let schema = json_schema.schema.clone().unwrap_or_else(|| json!({}));
            if !schema.is_object() {
                return Err("response_format.json_schema.schema must be an object".into());
            }
            Some(schema)
        }
    };
    let controls = GenerationControls {
        max_tokens,
        sampling: SamplingRequest {
            temperature: request.temperature,
            top_p: request.top_p,
            top_k: None,
            seed: request.seed,
        },
        top_logprobs,
        enable_thinking,
        reasoning_effort,
        json_schema,
        ignore_eos: request.ignore_eos,
        speculation: request.speculation.into(),
    };
    controls.validate(has_tools)?;
    Ok(controls)
}

/// Converts function tools to the template's shape. `tool_choice: "none"`
/// withholds them from the prompt; forcing a call is unsupported because
/// nothing here can make the model call.
fn tools(request: &Request) -> Result<Vec<Value>, String> {
    match &request.tool_choice {
        None => {}
        Some(choice) if choice == "auto" => {}
        Some(choice) if choice == "none" => return Ok(Vec::new()),
        Some(_) => return Err("only tool_choice auto or none is supported".into()),
    }
    let mut names = HashSet::new();
    request
        .tools
        .iter()
        .map(|tool| {
            let function = &tool["function"];
            let Some(name) = function["name"].as_str().filter(|_| tool["type"] == "function")
            else {
                return Err("only function tools with a name are supported".to_owned());
            };
            if name.is_empty() || !names.insert(name) {
                return Err("function names must be nonempty and unique".into());
            }
            let parameters = function
                .get("parameters")
                .cloned()
                .unwrap_or_else(|| json!({"type":"object","properties":{}}));
            if !parameters.is_object() {
                return Err("function parameters must be a JSON Schema object".into());
            }
            chat_format::validator(&parameters)?;
            Ok(json!({"type":"function","function":{"name":name,"description":function.get("description").cloned().unwrap_or(json!("")),"parameters":parameters}}))
        })
        .collect()
}

/// Text from a string or an array of text parts; `null` only where allowed.
fn content_text(content: &Value, allow_null: bool) -> Result<String, String> {
    match content {
        Value::String(text) => Ok(text.clone()),
        Value::Null if allow_null => Ok(String::new()),
        Value::Array(parts) => parts
            .iter()
            .map(|part| match part["type"].as_str() {
                Some("text") => part["text"]
                    .as_str()
                    .map(str::to_owned)
                    .ok_or_else(|| "text part requires text".to_owned()),
                // An assistant refusal part is history text like any other.
                Some("refusal") => part["refusal"]
                    .as_str()
                    .map(str::to_owned)
                    .ok_or_else(|| "refusal part requires refusal".to_owned()),
                _ => Err("only text content parts are supported".to_owned()),
            })
            .collect(),
        _ => Err("message content must be text or an array of text parts".into()),
    }
}

fn messages(request: &Request) -> Result<Vec<ChatMessage>, String> {
    let mut messages = Vec::new();
    let mut pending: HashMap<&str, &str> = HashMap::new();
    let mut seen = HashSet::new();
    for item in &request.messages {
        let role = item["role"].as_str().unwrap_or_default();
        match role {
            "system" | "developer" => {
                messages.push(message(
                    ChatRole::System,
                    content_text(&item["content"], false)?,
                ));
            }
            "user" => messages.push(message(
                ChatRole::User,
                content_text(&item["content"], false)?,
            )),
            "assistant" => {
                let mut turn = message(ChatRole::Assistant, content_text(&item["content"], true)?);
                for call in item["tool_calls"].as_array().into_iter().flatten() {
                    let id = call["id"].as_str().filter(|id| !id.is_empty());
                    let name = call["function"]["name"].as_str();
                    let (Some(id), Some(name), true) = (id, name, call["type"] == "function")
                    else {
                        return Err("tool calls require an id, type function and a name".into());
                    };
                    if !seen.insert(id) {
                        return Err("tool call IDs must be unique".into());
                    }
                    let arguments: Value = serde_json::from_str(
                        call["function"]["arguments"]
                            .as_str()
                            .ok_or("function arguments must be a JSON string")?,
                    )
                    .map_err(|error| error.to_string())?;
                    if !arguments.is_object() {
                        return Err("function arguments must encode an object".into());
                    }
                    pending.insert(id, name);
                    turn.tool_calls.push(ChatToolCall {
                        name: name.into(),
                        arguments,
                    });
                }
                messages.push(turn);
            }
            "tool" => {
                let id = item["tool_call_id"]
                    .as_str()
                    .ok_or("tool message requires tool_call_id")?;
                let name = pending
                    .remove(id)
                    .ok_or("tool message has no matching pending call")?;
                let mut output = message(ChatRole::Tool, content_text(&item["content"], false)?);
                output.name = Some(name.into());
                output.tool_call_id = Some(id.into());
                messages.push(output);
            }
            _ => return Err(format!("unsupported message role {role:?}")),
        }
    }
    if !messages
        .iter()
        .any(|message| message.role != ChatRole::System)
    {
        return Err("messages must contain at least one non-system message".into());
    }
    Ok(messages)
}

fn finish_reason(turn: &AssistantTurn) -> &'static str {
    if !turn.calls.is_empty() {
        "tool_calls"
    } else if turn.complete {
        "stop"
    } else {
        "length"
    }
}

fn tool_calls(turn: &AssistantTurn, id: &str) -> Vec<Value> {
    turn.calls
        .iter()
        .enumerate()
        .map(|(index, call)| {
            json!({"index":index,"id":format!("call_{id}_{index}"),"type":"function","function":{"name":call.name,"arguments":call.arguments.to_string()}})
        })
        .collect()
}

fn usage(generated: &ChatGeneration) -> Value {
    let prompt = generated.metrics.prompt_tokens;
    let completion = generated.generated_token_ids.len();
    let cached = generated.metrics.cached_prompt_tokens;
    json!({"prompt_tokens":prompt,"completion_tokens":completion,"total_tokens":prompt+completion,"prompt_tokens_details":{"cached_tokens":cached}})
}

/// The `metallix` block: the sampling policy used and the turn's timings.
fn metallix(generated: &ChatGeneration) -> Value {
    json!({"sampling":generated.sampling,"metrics":generated.metrics})
}

fn created() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |elapsed| elapsed.as_secs())
}

fn completion_value(
    prepared: &Prepared,
    generated: &ChatGeneration,
    id: &str,
) -> Result<Value, String> {
    let turn = assistant_turn(&prepared.tools, generated)?;
    let mut message = json!({"role":"assistant","content":if turn.calls.is_empty() || !turn.text.trim().is_empty() {json!(turn.text)} else {Value::Null},"refusal":null});
    if !turn.calls.is_empty() {
        let calls: Vec<Value> = tool_calls(&turn, id)
            .into_iter()
            .map(|mut call| {
                call.as_object_mut().map(|call| call.remove("index"));
                call
            })
            .collect();
        message["tool_calls"] = json!(calls);
    }
    if !turn.reasoning.is_empty() {
        // vLLM's and DeepSeek's field; Chat Completions has no standard one.
        message["reasoning_content"] = json!(turn.reasoning);
    }
    let logprobs = if prepared.controls.top_logprobs.is_some() {
        json!({"content":logprobs_value(&generated.logprobs),"refusal":null})
    } else {
        Value::Null
    };
    Ok(
        json!({"id":format!("chatcmpl-{id}"),"object":"chat.completion","created":created(),"model":prepared.request.model,"choices":[{"index":0,"message":message,"logprobs":logprobs,"finish_reason":finish_reason(&turn)}],"usage":usage(generated),"metallix":metallix(generated)}),
    )
}

fn generation_error(connection: Connection, error: &ChatGenerationError) {
    match error {
        ChatGenerationError::DeadlineExceeded => json_response(
            connection,
            408,
            &json!({"error":{"message":"generation time budget exceeded","type":"timeout","param":null,"code":"generation_timeout"}}),
        ),
        ChatGenerationError::Message(message) => {
            json_response(connection, 400, &error_body(message));
        }
    }
}

/// Answers one prepared request, as JSON or as a chunk stream.
pub(crate) fn respond(
    connection: Connection,
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
    let request_id = connection.request_id().map(str::to_owned);
    if !prepared.request.stream {
        let result = session
            .generate_with_timeout(request, generation_timeout, &mut |_| Ok(()))
            .and_then(|generated| {
                record_usage(&generated);
                let mut value = completion_value(prepared, &generated, id)
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

#[allow(clippy::too_many_lines, reason = "one ordered chunk lifecycle")]
fn stream(
    connection: Connection,
    prepared: &Prepared,
    request: ChatRequest<'_>,
    session: &mut dyn ChatBackend,
    id: &str,
    generation_timeout: Duration,
) -> Result<(), String> {
    let include_usage = prepared
        .request
        .stream_options
        .as_ref()
        .is_some_and(|options| options.include_usage);
    let created = created();
    let chunk = |delta: Value, finish_reason: Value, logprobs: Value| {
        let mut chunk = json!({"id":format!("chatcmpl-{id}"),"object":"chat.completion.chunk","created":created,"model":prepared.request.model,"choices":[{"index":0,"delta":delta,"logprobs":logprobs,"finish_reason":finish_reason}]});
        if include_usage {
            chunk["usage"] = Value::Null;
        }
        chunk
    };
    // Text streams live only when no tool envelope or reasoning block has to
    // be parsed out of it first.
    let live = prepared.tools.is_empty() && !prepared.controls.enable_thinking;
    let mut sse = LazySse::new(connection);
    let mut role_sent = false;
    let generated = session.generate_with_timeout(request, generation_timeout, &mut |delta| {
        let TurnDelta::Text(delta) = delta else {
            return sse.keepalive();
        };
        if !live {
            return sse.keepalive();
        }
        if !role_sent {
            role_sent = true;
            sse.event(
                None,
                &chunk(
                    json!({"role":"assistant","content":""}),
                    Value::Null,
                    Value::Null,
                ),
            )?;
        }
        sse.event(
            None,
            &chunk(json!({"content":delta}), Value::Null, Value::Null),
        )
    });
    let failure = |sse: &mut LazySse, code: &str, message: &str| {
        sse.event(
            None,
            &json!({"error":{"message":message,"type":"server_error","param":null,"code":code}}),
        )?;
        sse.raw(None, "[DONE]")
    };
    let generated = match generated {
        Ok(generated) => generated,
        Err(error) => {
            if let Some(connection) = sse.take_unopened() {
                generation_error(connection, &error);
                return Ok(());
            }
            return match error {
                ChatGenerationError::DeadlineExceeded => failure(
                    &mut sse,
                    "generation_timeout",
                    "generation time budget exceeded",
                ),
                ChatGenerationError::Message(message) => {
                    failure(&mut sse, "generation_failed", &message)
                }
            };
        }
    };
    record_usage(&generated);
    let turn = match assistant_turn(&prepared.tools, &generated) {
        Ok(turn) => turn,
        Err(error) => return failure(&mut sse, "invalid_model_output", &error),
    };
    if !role_sent {
        sse.event(
            None,
            &chunk(
                json!({"role":"assistant","content":""}),
                Value::Null,
                Value::Null,
            ),
        )?;
    }
    if !live {
        if !turn.reasoning.is_empty() {
            sse.event(
                None,
                &chunk(
                    json!({"reasoning_content":turn.reasoning}),
                    Value::Null,
                    Value::Null,
                ),
            )?;
        }
        if !turn.text.is_empty() {
            sse.event(
                None,
                &chunk(json!({"content":turn.text}), Value::Null, Value::Null),
            )?;
        }
        for call in tool_calls(&turn, id) {
            sse.event(
                None,
                &chunk(json!({"tool_calls":[call]}), Value::Null, Value::Null),
            )?;
        }
    }
    let logprobs = if prepared.controls.top_logprobs.is_some() {
        json!({"content":logprobs_value(&generated.logprobs),"refusal":null})
    } else {
        Value::Null
    };
    sse.event(
        None,
        &chunk(json!({}), json!(finish_reason(&turn)), logprobs),
    )?;
    if include_usage {
        let last = json!({"id":format!("chatcmpl-{id}"),"object":"chat.completion.chunk","created":created,"model":prepared.request.model,"choices":[],"usage":usage(&generated),"metallix":metallix(&generated)});
        sse.event(None, &last)?;
    }
    sse.raw(None, "[DONE]")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sse::test_support::{
        SCRIPTED_EOS, Scripted, events, exchange, exchange_with, json_body,
    };

    fn prepare_body(body: &Value) -> Result<Prepared, Value> {
        prepare(body.to_string().as_bytes()).map(|(_, prepared)| prepared)
    }

    fn with(extra: &Value) -> Value {
        let mut body = json!({"model":"m","messages":[{"role":"user","content":"hi"}]});
        body.as_object_mut()
            .unwrap()
            .extend(extra.as_object().unwrap().clone());
        body
    }

    const TOOL: &str = r#"{"type":"function","function":{"name":"read_file","parameters":{"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}}}"#;

    #[test]
    fn rejects_unknown_fields_and_unsupported_semantics() {
        let tool: Value = serde_json::from_str(TOOL).unwrap();
        for extra in [
            json!({"frobnicate":1}),
            json!({"stream_options":{"include_usage":true,"chunk_size":4}}),
            json!({"response_format":{"type":"json_schema","json_schema":{"name":"x","schema":{},"extra":1}}}),
            json!({"ignore_eos":true,"response_format":{"type":"json_schema","json_schema":{"name":"x","schema":{"type":"object"}}}}),
            json!({"n":2}),
            json!({"stop":["\n"]}),
            json!({"stop":"END"}),
            json!({"frequency_penalty":0.5}),
            json!({"logit_bias":{"42":5}}),
            json!({"store":true}),
            json!({"modalities":["text","audio"]}),
            json!({"top_logprobs":2}),
            json!({"logprobs":true,"top_logprobs":21}),
            json!({"max_tokens":10,"max_completion_tokens":20}),
            json!({"temperature":3.0}),
            json!({"reasoning_effort":"maximum"}),
            json!({"tool_choice":"required","tools":[tool.clone()]}),
            json!({"tools":[{"type":"custom","custom":{"name":"x"}}]}),
            json!({"messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"x"}}]}]}),
            json!({"messages":[{"role":"function","name":"f","content":"x"}]}),
            json!({"messages":[{"role":"tool","tool_call_id":"nope","content":"x"}]}),
            json!({"messages":[{"role":"system","content":"only system"}]}),
        ] {
            assert!(prepare_body(&with(&extra)).is_err(), "accepted {extra}");
        }
        let error = prepare_body(&with(&json!({"frobnicate":1}))).err().unwrap();
        assert_eq!(error["error"]["type"], "invalid_request_error");
        assert!(
            error["error"]["message"]
                .as_str()
                .unwrap()
                .contains("frobnicate")
        );
    }

    /// Recorded agent traces (`SiliconBench` a003) carry parallel calls whose
    /// results were never logged. vLLM and mlx-lm render them as given.
    #[test]
    fn unanswered_tool_calls_in_history_render_as_given() {
        let call = |id: &str, path: &str| json!({"id":id,"type":"function","function":{"name":"read_file","arguments":json!({"path":path}).to_string()}});
        let prepared = prepare_body(&with(&json!({"messages":[
            {"role":"user","content":"find the readme"},
            {"role":"assistant","content":"","tool_calls":[call("c2","a.md"),call("c3","b.md")]},
            {"role":"tool","tool_call_id":"c2","content":"A"},
            {"role":"assistant","content":"","tool_calls":[call("c4","c.md")]},
            {"role":"tool","tool_call_id":"c4","content":"C"}
        ]})))
        .unwrap();
        let history: Vec<_> = prepared
            .messages
            .iter()
            .map(|message| {
                (
                    message.role,
                    message.tool_calls.len(),
                    message.tool_call_id.as_deref(),
                )
            })
            .collect();
        assert_eq!(
            history,
            [
                (ChatRole::User, 0, None),
                (ChatRole::Assistant, 2, None),
                (ChatRole::Tool, 0, Some("c2")),
                (ChatRole::Assistant, 1, None),
                (ChatRole::Tool, 0, Some("c4")),
            ]
        );
        assert_eq!(
            prepared.messages[1].tool_calls[1].arguments,
            json!({"path":"b.md"})
        );
        let late_result = json!({"messages":[
            {"role":"user","content":"x"},
            {"role":"assistant","content":"","tool_calls":[call("c1","a.md")]},
            {"role":"user","content":"never mind"},
            {"role":"tool","tool_call_id":"c1","content":"A"}
        ]});
        assert!(
            prepare_body(&with(&late_result)).is_ok(),
            "a later result still matches its call"
        );
        let unknown = json!({"messages":[
            {"role":"user","content":"x"},
            {"role":"assistant","content":"","tool_calls":[call("c1","a.md")]},
            {"role":"tool","tool_call_id":"c9","content":"A"}
        ]});
        assert!(
            prepare_body(&with(&unknown)).is_err(),
            "a result for an unknown call has no name to render"
        );
    }

    #[test]
    fn accepts_sdk_defaults_and_maps_controls() {
        let tool: Value = serde_json::from_str(TOOL).unwrap();
        let prepared = prepare_body(&with(&json!({
            "messages": [
                {"role":"developer","content":[{"type":"text","text":"be brief"}]},
                {"role":"user","content":"read it","name":"ann"},
                {"role":"assistant","content":null,"tool_calls":[
                    {"id":"c1","type":"function","function":{"name":"read_file","arguments":"{\"path\":\"a\"}"}},
                    {"id":"c2","type":"function","function":{"name":"read_file","arguments":"{\"path\":\"b\"}"}}
                ]},
                {"role":"tool","tool_call_id":"c2","content":"B"},
                {"role":"tool","tool_call_id":"c1","content":[{"type":"text","text":"A"}]}
            ],
            "tools": [tool],
            "tool_choice": "auto",
            "parallel_tool_calls": true,
            "stream": true,
            "stream_options": {"include_usage": true, "include_obfuscation": false},
            "user": "u", "metadata": {"k":"v"}, "store": false, "service_tier": "auto",
            "prompt_cache_key": "k", "safety_identifier": "s", "verbosity": "low",
            "n": 1, "stop": null, "frequency_penalty": 0, "presence_penalty": 0,
            "modalities": ["text"], "logit_bias": {},
            "max_completion_tokens": 64, "temperature": 0.7, "top_p": 0.9, "seed": 3,
            "logprobs": false,
            "reasoning_effort": "none"
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
                ChatRole::Tool
            ]
        );
        assert_eq!(prepared.messages[2].tool_calls.len(), 2);
        assert_eq!(prepared.messages[3].name.as_deref(), Some("read_file"));
        assert_eq!(prepared.messages[4].content, "A");
        assert_eq!(prepared.tools[0]["function"]["name"], "read_file");
        let controls = &prepared.controls;
        assert_eq!(controls.max_tokens, Some(64));
        assert_eq!(
            controls.sampling,
            SamplingRequest {
                temperature: Some(0.7),
                top_p: Some(0.9),
                top_k: None,
                seed: Some(3)
            }
        );
        assert_eq!(controls.top_logprobs, None);
        assert!(!controls.enable_thinking);

        let mapped = |extra: Value| prepare_body(&with(&extra)).unwrap().controls;
        assert_eq!(
            mapped(json!({"logprobs":true,"top_logprobs":4})).top_logprobs,
            Some(4)
        );
        assert_eq!(mapped(json!({"max_tokens":9})).max_tokens, Some(9));
        assert!(mapped(json!({"reasoning_effort":"max"})).enable_thinking);
        assert!(prepare_body(&with(&json!({"tool_choice":"none","tools":[serde_json::from_str::<Value>(TOOL).unwrap()]}))).unwrap().tools.is_empty());
        if cfg!(feature = "structured-output") {
            let schema = json!({"type":"object","properties":{"ok":{"type":"boolean"}}});
            assert_eq!(
                mapped(json!({"response_format":{"type":"json_schema","json_schema":{"name":"r","strict":true,"schema":schema}}})).json_schema,
                Some(schema)
            );
            assert_eq!(
                mapped(json!({"response_format":{"type":"json_object"}})).json_schema,
                Some(json!({"type":"object"}))
            );
        }
    }

    fn run(body: &Value, backend: &mut Scripted) -> String {
        exchange(
            "/v1/chat/completions",
            &body.to_string(),
            |connection, body| {
                let (_, prepared) = prepare(body).unwrap();
                respond(connection, &prepared, backend, "t", Duration::from_secs(60)).unwrap();
            },
        )
    }

    #[test]
    fn answers_text_and_tool_calls_in_both_shapes() {
        let mut backend = Scripted::new("Hello there");
        let (status, body) = json_body(&run(&with(&json!({"logprobs":true})), &mut backend));
        assert_eq!(status, "HTTP/1.1 200 OK");
        assert_eq!(body["object"], "chat.completion");
        assert_eq!(body["id"], "chatcmpl-t");
        assert_eq!(body["choices"][0]["message"]["content"], "Hello there");
        assert_eq!(body["choices"][0]["finish_reason"], "stop");
        assert_eq!(body["choices"][0]["logprobs"]["content"], json!([]));
        assert_eq!(body["usage"]["completion_tokens"], 2);
        assert_eq!(body["usage"]["total_tokens"], 3);

        let tool: Value = serde_json::from_str(TOOL).unwrap();
        let call = r#"Let me look.<tool_call>{"name":"read_file","arguments":{"path":"README.md"}}</tool_call>"#;
        let mut backend = Scripted::new(call);
        let (_, body) = json_body(&run(&with(&json!({"tools":[tool.clone()]})), &mut backend));
        let message = &body["choices"][0]["message"];
        assert_eq!(body["choices"][0]["finish_reason"], "tool_calls");
        assert_eq!(message["content"], "Let me look.");
        assert_eq!(message["tool_calls"][0]["id"], "call_t_0");
        assert_eq!(
            message["tool_calls"][0]["function"]["arguments"],
            r#"{"path":"README.md"}"#
        );
        assert!(message["tool_calls"][0].get("index").is_none());

        let mut backend = Scripted::new(call);
        let chunks = events(&run(
            &with(&json!({"tools":[tool],"stream":true,"stream_options":{"include_usage":true}})),
            &mut backend,
        ));
        assert_eq!(chunks.last().unwrap().1, "[DONE]");
        let data: Vec<Value> = chunks[..chunks.len() - 1]
            .iter()
            .map(|(name, data)| {
                assert!(name.is_none());
                serde_json::from_str(data).unwrap()
            })
            .collect();
        assert!(
            data.iter()
                .all(|chunk| chunk["object"] == "chat.completion.chunk")
        );
        assert_eq!(data[0]["choices"][0]["delta"]["role"], "assistant");
        let call = data
            .iter()
            .find_map(|chunk| chunk["choices"][0]["delta"]["tool_calls"].get(0))
            .unwrap();
        assert_eq!(call["index"], 0);
        assert_eq!(call["function"]["name"], "read_file");
        let finish = &data[data.len() - 2];
        assert_eq!(finish["choices"][0]["finish_reason"], "tool_calls");
        assert!(finish["usage"].is_null());
        let usage = data.last().unwrap();
        assert_eq!(usage["choices"], json!([]));
        assert_eq!(usage["usage"]["prompt_tokens"], 1);
    }

    #[test]
    fn ignore_eos_runs_to_max_tokens_past_an_early_eos() {
        let early = [7, SCRIPTED_EOS];
        let mut backend = Scripted::emitting(&early);
        let (_, body) = json_body(&run(&with(&json!({"max_tokens":8})), &mut backend));
        assert_eq!(body["choices"][0]["finish_reason"], "stop");
        assert_eq!(body["choices"][0]["message"]["content"], "a");
        assert_eq!(body["usage"]["completion_tokens"], 2);

        let mut backend = Scripted::emitting(&early);
        let ignoring = with(&json!({"max_tokens":8,"ignore_eos":true}));
        let (_, body) = json_body(&run(&ignoring, &mut backend));
        assert_eq!(body["choices"][0]["finish_reason"], "length");
        assert_eq!(body["choices"][0]["message"]["content"], "a<eos>".repeat(4));
        assert_eq!(body["usage"]["completion_tokens"], 8);

        let mut streamed = ignoring;
        streamed["stream"] = json!(true);
        let mut backend = Scripted::emitting(&early);
        let chunks: Vec<Value> = events(&run(&streamed, &mut backend))
            .into_iter()
            .filter_map(|(_, data)| serde_json::from_str(&data).ok())
            .collect();
        let text: String = chunks
            .iter()
            .filter_map(|chunk| chunk["choices"][0]["delta"]["content"].as_str())
            .collect();
        assert_eq!(text, "a<eos>".repeat(4));
        assert_eq!(
            chunks.last().unwrap()["choices"][0]["finish_reason"],
            "length"
        );
    }

    #[test]
    fn cached_prompt_tokens_are_a_subset_of_prompt_tokens() {
        for stream in [false, true] {
            let mut backend = Scripted::new("hi");
            backend.prompt_tokens = 10;
            backend.cached_prompt_tokens = 4;
            let body = if stream {
                with(&json!({"stream":true,"stream_options":{"include_usage":true}}))
            } else {
                with(&json!({}))
            };
            let wire = run(&body, &mut backend);
            let usage = if stream {
                let (_, data) = events(&wire)
                    .into_iter()
                    .rev()
                    .find(|(_, data)| data != "[DONE]")
                    .unwrap();
                serde_json::from_str::<Value>(&data).unwrap()["usage"].clone()
            } else {
                json_body(&wire).1["usage"].clone()
            };
            assert_eq!(usage["prompt_tokens"], 10, "stream={stream}");
            assert_eq!(
                usage["prompt_tokens_details"]["cached_tokens"], 4,
                "stream={stream}"
            );
            assert_eq!(usage["total_tokens"], 12, "stream={stream}");
        }
    }

    #[test]
    fn the_router_cache_salt_reaches_generation() {
        for stream in [false, true] {
            let mut backend = Scripted::new("hi");
            let body = with(&json!({"stream":stream})).to_string();
            exchange_with(
                "/v1/chat/completions",
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
        let chunks = events(&run(&with(&json!({"stream":true})), &mut backend));
        let deltas: String = chunks
            .iter()
            .filter_map(|(_, data)| serde_json::from_str::<Value>(data).ok())
            .filter_map(|chunk| {
                chunk["choices"][0]["delta"]["content"]
                    .as_str()
                    .map(str::to_owned)
            })
            .collect();
        assert_eq!(deltas, "abc");
        assert!(backend.deltas > 1, "text arrives as several deltas");

        for stream in [false, true] {
            let mut backend =
                Scripted::failing("chat requires prompt_tokens + max_tokens <= 16384");
            let (status, body) = json_body(&run(&with(&json!({"stream":stream})), &mut backend));
            assert_eq!(status, "HTTP/1.1 400 Bad Request", "stream={stream}");
            assert_eq!(body["error"]["type"], "invalid_request_error");
        }
    }

    fn real_session() -> crate::chat_generation::ChatSession {
        use crate::chat_generation::{ChatSession, ResidentChatLimits};
        let model = std::env::var_os("METALLIX_QWEN_MODEL").expect("set METALLIX_QWEN_MODEL");
        ChatSession::load(
            std::path::Path::new(&model),
            ResidentChatLimits::from_mib(16_384, 8_192),
        )
        .expect("load local Qwen checkpoint")
    }

    fn real_run(body: &Value, session: &mut crate::chat_generation::ChatSession) -> String {
        exchange(
            "/v1/chat/completions",
            &body.to_string(),
            |connection, body| {
                let (_, prepared) = prepare(body).unwrap();
                respond(
                    connection,
                    &prepared,
                    session,
                    "q",
                    Duration::from_secs(600),
                )
                .unwrap();
            },
        )
    }

    /// Opt-in: `ignore_eos` on every decode path of a real local Qwen3
    /// checkpoint named by `METALLIX_QWEN_MODEL`.
    #[test]
    #[ignore = "requires METALLIX_QWEN_MODEL and a local Apple-Silicon Metal checkpoint"]
    fn real_qwen_ignore_eos_reaches_max_tokens_on_every_decode_path() {
        let mut session = real_session();
        let run = real_run;
        let body = json!({"model":"q","messages":[{"role":"user","content":"Say hello."}],"temperature":0,"max_completion_tokens":16});
        let (_, plain) = json_body(&run(&body, &mut session));
        let text = plain["choices"][0]["message"]["content"]
            .as_str()
            .unwrap()
            .to_owned();
        // ignore_eos runs the short greedy answer out to the full output
        // limit, on the pipelined GPU-pick path and, with logprobs, on the
        // host path; both continue past the same end-of-turn identically.
        assert_eq!(plain["choices"][0]["finish_reason"], "stop", "{plain}");
        assert!(plain["usage"]["completion_tokens"].as_u64().unwrap() < 16);
        let mut ignoring = body.clone();
        ignoring["ignore_eos"] = json!(true);
        let (_, pipelined) = json_body(&run(&ignoring, &mut session));
        ignoring["logprobs"] = json!(true);
        let (_, host) = json_body(&run(&ignoring, &mut session));
        for long in [&pipelined, &host] {
            assert_eq!(long["choices"][0]["finish_reason"], "length", "{long}");
            assert_eq!(long["usage"]["completion_tokens"], 16, "{long}");
            let content = long["choices"][0]["message"]["content"].as_str().unwrap();
            assert!(content.starts_with(&text), "{content:?} extends {text:?}");
            assert!(content.contains("<|im_end|>"), "{content:?}");
        }
        assert_eq!(
            host["choices"][0]["logprobs"]["content"]
                .as_array()
                .unwrap()
                .len(),
            16
        );
        assert_eq!(
            pipelined["choices"][0]["message"]["content"],
            host["choices"][0]["message"]["content"]
        );
        // Seeded sampling with the checkpoint's top_k also picks on the GPU.
        let sampled = json!({"model":"q","messages":[{"role":"user","content":"Say hello."}],"seed":7,"max_completion_tokens":16,"ignore_eos":true});
        let (_, sampled) = json_body(&run(&sampled, &mut session));
        assert_eq!(
            sampled["choices"][0]["finish_reason"], "length",
            "{sampled}"
        );
        assert_eq!(sampled["usage"]["completion_tokens"], 16, "{sampled}");
    }

    /// Opt-in against a real local Qwen3 checkpoint named by
    /// `METALLIX_QWEN_MODEL`.
    #[test]
    #[ignore = "requires METALLIX_QWEN_MODEL and a local Apple-Silicon Metal checkpoint"]
    fn real_qwen_answers_streams_and_calls_tools() {
        let mut session = real_session();
        let run = real_run;
        let body = json!({"model":"q","messages":[{"role":"user","content":"Say hello."}],"temperature":0,"max_completion_tokens":16});
        let (status, plain) = json_body(&run(&body, &mut session));
        assert_eq!(status, "HTTP/1.1 200 OK", "{plain}");
        let text = plain["choices"][0]["message"]["content"]
            .as_str()
            .unwrap()
            .to_owned();
        assert!(!text.is_empty());

        // Greedy streaming reproduces the same text.
        let mut streamed = body.clone();
        streamed["stream"] = json!(true);
        let chunks = events(&run(&streamed, &mut session));
        let joined: String = chunks
            .iter()
            .filter_map(|(_, data)| serde_json::from_str::<Value>(data).ok())
            .filter_map(|chunk| {
                chunk["choices"][0]["delta"]["content"]
                    .as_str()
                    .map(str::to_owned)
            })
            .collect();
        assert_eq!(joined, text);

        let tools = json!([{"type":"function","function":{"name":"get_weather","description":"Current weather for a city","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}]);
        let body = json!({"model":"q","messages":[{"role":"user","content":"What is the weather in Paris? Use the tool."}],"tools":tools,"temperature":0,"max_completion_tokens":96});
        let (_, called) = json_body(&run(&body, &mut session));
        assert_eq!(
            called["choices"][0]["finish_reason"], "tool_calls",
            "{called}"
        );
        let call = &called["choices"][0]["message"]["tool_calls"][0];
        assert_eq!(call["function"]["name"], "get_weather");
        let arguments: Value =
            serde_json::from_str(call["function"]["arguments"].as_str().unwrap()).unwrap();
        assert!(arguments["city"].is_string());

        if cfg!(feature = "structured-output") {
            let schema = json!({"type":"object","properties":{"capital":{"type":"string","maxLength":40}},"required":["capital"],"additionalProperties":false});
            let body = json!({"model":"q","messages":[{"role":"user","content":"Capital of France, as JSON."}],"max_completion_tokens":48,"response_format":{"type":"json_schema","json_schema":{"name":"c","strict":true,"schema":schema}}});
            let (_, shaped) = json_body(&run(&body, &mut session));
            let value: Value =
                serde_json::from_str(shaped["choices"][0]["message"]["content"].as_str().unwrap())
                    .unwrap();
            assert!(
                chat_format::validator(&schema).unwrap().is_valid(&value),
                "{value}"
            );
        }
    }
}
