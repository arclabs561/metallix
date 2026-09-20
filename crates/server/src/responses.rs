//! Text-only Responses protocol boundary for the native Qwen control model.

use std::{
    collections::{HashMap, HashSet},
    io::{Read, Write},
    net::SocketAddr,
    path::Path,
    process::ExitCode,
};

use serde::Deserialize;
use serde_json::{Value, json};

use crate::{
    chat_cli::message,
    chat_generation::{
        ChatFinishReason, ChatMessage, ChatRequest, ChatRole, ChatSession, ChatToolCall,
    },
    chat_tools,
};

const MAX_BODY: u64 = 1024 * 1024;

#[derive(Deserialize)]
struct Request {
    model: String,
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
    #[serde(default)]
    seed: Option<u64>,
    #[serde(default)]
    store: Option<bool>,
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

#[allow(
    clippy::float_cmp,
    reason = "reject unsupported sampling values exactly"
)]
fn messages(request: &Request) -> Result<Vec<ChatMessage>, String> {
    if request.temperature.is_some_and(|value| value != 0.0)
        || request.top_p.is_some_and(|value| value != 1.0)
        || request.seed.is_some()
    {
        return Err(
            "native chat currently supports greedy sampling only (temperature=0, no seed)".into(),
        );
    }
    if request.store == Some(true) {
        return Err("response storage is unsupported; use store=false".into());
    }
    if request
        .max_output_tokens
        .is_some_and(|value| value == 0 || value > 256)
    {
        return Err("max_output_tokens must be 1..=256".into());
    }
    if request.previous_response_id.is_some() {
        return Err("send complete input history; previous_response_id is not supported".into());
    }
    if request.tool_choice.as_ref().is_some_and(|v| v != "auto") {
        return Err("only automatic tool choice is supported".into());
    }
    let mut messages = Vec::new();
    let mut pending_calls = HashMap::new();
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
                if call_id.is_empty() || pending_calls.insert(call_id, name).is_some() {
                    return Err("function call IDs must be nonempty and unique".into());
                }
                call.tool_calls.push(ChatToolCall {
                    name: name.into(),
                    arguments,
                });
                call.tool_call_id = Some(
                    item["call_id"]
                        .as_str()
                        .ok_or("function requires call_id")?
                        .into(),
                );
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
            _ => return Err("unsupported Responses input item".into()),
        }
    }
    if messages.is_empty() {
        return Err("input must contain at least one message".into());
    }
    if !pending_calls.is_empty() {
        return Err("tool calls require results before generation resumes".into());
    }
    Ok(messages)
}

fn tools(request: &Request) -> Result<Vec<Value>, String> {
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

fn json_response(request: tiny_http::Request, status: u16, value: &Value) {
    let response = tiny_http::Response::from_string(value.to_string())
        .with_status_code(status)
        .with_header(
            tiny_http::Header::from_bytes("Content-Type", "application/json")
                .expect("static header"),
        );
    let _ = request.respond(response);
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

pub(crate) fn serve(model: &Path, model_id: &str, address: SocketAddr) -> ExitCode {
    match serve_inner(model, model_id, address) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("mx serve: {error}");
            ExitCode::FAILURE
        }
    }
}

fn serve_inner(model: &Path, model_id: &str, address: SocketAddr) -> Result<(), String> {
    if !address.ip().is_loopback() {
        return Err("this experimental server binds only to loopback".into());
    }
    let mut session = ChatSession::load(model)?;
    let server = tiny_http::Server::http(address).map_err(|e| e.to_string())?;
    eprintln!(
        "mx listening on http://{address}; model={model_id}; single request; 512 total tokens; load_ms={:.2}",
        session.load_ms()
    );
    for (index, mut request) in server.incoming_requests().enumerate() {
        match (request.method().as_str(), request.url()) {
            ("GET", "/healthz") => {
                json_response(request, 200, &json!({"status":"ready"}));
                continue;
            }
            ("GET", "/v1/models") => {
                json_response(
                    request,
                    200,
                    &json!({"object":"list","data":[{"id":model_id,"object":"model","owned_by":"local"}]}),
                );
                continue;
            }
            ("POST", "/v1/responses") => {}
            _ => {
                json_response(
                    request,
                    404,
                    &json!({"error":{"message":"unknown endpoint"}}),
                );
                continue;
            }
        }
        if request.body_length().is_none_or(|n| n as u64 > MAX_BODY) {
            json_response(
                request,
                413,
                &json!({"error":{"message":"Content-Length required, maximum 1 MiB"}}),
            );
            continue;
        }
        let mut body = Vec::new();
        let parsed = request
            .as_reader()
            .take(MAX_BODY + 1)
            .read_to_end(&mut body)
            .map_err(|e| e.to_string())
            .and_then(|_| serde_json::from_slice::<Request>(&body).map_err(|e| e.to_string()));
        let parsed = match parsed {
            Ok(parsed) => parsed,
            Err(error) => {
                json_response(request, 400, &json!({"error":{"message":error}}));
                continue;
            }
        };
        if parsed.model != model_id {
            json_response(
                request,
                404,
                &json!({"error":{"message":"model is not loaded"}}),
            );
            continue;
        }
        let prepared =
            messages(&parsed).and_then(|messages| tools(&parsed).map(|tools| (messages, tools)));
        let (messages, tools) = match prepared {
            Ok(prepared) => prepared,
            Err(error) => {
                json_response(request, 400, &json!({"error":{"message":error}}));
                continue;
            }
        };
        let id = format!("resp_{}_{}", std::process::id(), index);
        if let Err(error) = respond(request, &parsed, &messages, &tools, &mut session, &id) {
            eprintln!("response failed: {error}");
        }
    }
    Ok(())
}

#[allow(
    clippy::too_many_lines,
    reason = "one ordered response event lifecycle"
)]
fn respond(
    request: tiny_http::Request,
    parsed: &Request,
    messages: &[ChatMessage],
    tools: &[Value],
    session: &mut ChatSession,
    id: &str,
) -> Result<(), String> {
    let mut sequence = 0;
    let mut writer = if parsed.stream {
        let mut writer = request.into_writer();
        write!(writer,"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: close\r\n\r\n").map_err(|e| e.to_string())?;
        event(
            &mut writer,
            &mut sequence,
            json!({"type":"response.created","response":{"id":id,"object":"response","status":"in_progress","output":[]}}),
        )?;
        Some(writer)
    } else {
        respond_json(request, parsed, messages, tools, session, id);
        return Ok(());
    };
    let message_id = format!("msg_{id}");
    let stream_text = tools.is_empty();
    if stream_text {
        let writer = writer.as_mut().expect("stream writer");
        event(
            writer,
            &mut sequence,
            json!({"type":"response.output_item.added","output_index":0,"item":{"id":message_id,"type":"message","role":"assistant","status":"in_progress","content":[]}}),
        )?;
        event(
            writer,
            &mut sequence,
            json!({"type":"response.content_part.added","item_id":message_id,"output_index":0,"content_index":0,"part":{"type":"output_text","text":"","annotations":[]}}),
        )?;
    }
    let generated = session.generate(ChatRequest {messages,tools,max_tokens:parsed.max_output_tokens.unwrap_or(128),enable_thinking:false}, &mut |delta| {
        if stream_text {
            event(writer.as_mut().expect("stream writer"),&mut sequence,json!({"type":"response.output_text.delta","item_id":message_id,"output_index":0,"content_index":0,"delta":delta}))?;
        } else {
            // Withhold incomplete tool envelopes, but still detect disconnects.
            let writer = writer.as_mut().expect("stream writer");
            writer.write_all(b": generating\n\n").map_err(|error| error.to_string())?;
            writer.flush().map_err(|error| error.to_string())?;
        }
        Ok(())
    });
    let generated = match generated {
        Ok(generated) => generated,
        Err(error) => {
            event(
                writer.as_mut().expect("stream writer"),
                &mut sequence,
                json!({"type":"response.failed","response":{"id":id,"status":"failed","error":{"code":"generation_failed","message":error}}}),
            )?;
            return Ok(());
        }
    };
    let response = match response_value(parsed, &generated, id) {
        Ok(response) => response,
        Err(error) => {
            event(
                writer.as_mut().expect("stream writer"),
                &mut sequence,
                json!({"type":"response.failed","response":{"id":id,"status":"failed","error":{"code":"invalid_model_output","message":error}}}),
            )?;
            return Ok(());
        }
    };
    let writer = writer.as_mut().expect("stream writer");
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
            event(
                writer,
                &mut sequence,
                json!({"type":"response.output_item.added","output_index":index,"item":started}),
            )?;
            event(
                writer,
                &mut sequence,
                json!({"type":"response.function_call_arguments.delta","item_id":item["id"],"output_index":index,"delta":item["arguments"]}),
            )?;
            event(
                writer,
                &mut sequence,
                json!({"type":"response.function_call_arguments.done","item_id":item["id"],"output_index":index,"arguments":item["arguments"]}),
            )?;
        } else {
            if !stream_text {
                event(
                    writer,
                    &mut sequence,
                    json!({"type":"response.output_item.added","output_index":index,"item":{"id":item["id"],"type":"message","role":"assistant","status":"in_progress","content":[]}}),
                )?;
                event(
                    writer,
                    &mut sequence,
                    json!({"type":"response.content_part.added","item_id":item["id"],"output_index":index,"content_index":0,"part":{"type":"output_text","text":"","annotations":[]}}),
                )?;
                event(
                    writer,
                    &mut sequence,
                    json!({"type":"response.output_text.delta","item_id":item["id"],"output_index":index,"content_index":0,"delta":item["content"][0]["text"]}),
                )?;
            }
            event(
                writer,
                &mut sequence,
                json!({"type":"response.output_text.done","item_id":item["id"],"output_index":index,"content_index":0,"text":item["content"][0]["text"]}),
            )?;
            event(
                writer,
                &mut sequence,
                json!({"type":"response.content_part.done","item_id":item["id"],"output_index":index,"content_index":0,"part":item["content"][0]}),
            )?;
        }
        event(
            writer,
            &mut sequence,
            json!({"type":"response.output_item.done","output_index":index,"item":item}),
        )?;
    }
    let event_type = if response["status"] == "incomplete" {
        "response.incomplete"
    } else {
        "response.completed"
    };
    event(
        writer,
        &mut sequence,
        json!({"type":event_type,"response":response}),
    )
}

fn respond_json(
    request: tiny_http::Request,
    parsed: &Request,
    messages: &[ChatMessage],
    tools: &[Value],
    session: &mut ChatSession,
    id: &str,
) {
    let result = session
        .generate(
            ChatRequest {
                messages,
                tools,
                max_tokens: parsed.max_output_tokens.unwrap_or(128),
                enable_thinking: false,
            },
            &mut |_| Ok(()),
        )
        .and_then(|generated| response_value(parsed, &generated, id));
    match result {
        Ok(response) => json_response(request, 200, &response),
        Err(error) => json_response(request, 400, &json!({"error":{"message":error}})),
    }
}

fn response_value(
    request: &Request,
    generated: &crate::chat_generation::ChatGeneration,
    id: &str,
) -> Result<Value, String> {
    let turn = chat_tools::parse_turn(&generated.text)?;
    let calls = turn.calls;
    let complete = generated.finish_reason == ChatFinishReason::Eos;
    if !calls.is_empty() && !complete {
        return Err("truncated tool turn; no function calls returned".into());
    }
    let mut output = Vec::new();
    if calls.is_empty() || !turn.text.trim().is_empty() {
        output.push(json!({"id":format!("msg_{id}"),"type":"message","role":"assistant","status":if complete {"completed"} else {"incomplete"},"content":[{"type":"output_text","text":turn.text,"annotations":[]}]}));
    }
    for (index, call) in calls.into_iter().enumerate() {
        let definition = request
            .tools
            .iter()
            .find(|tool| tool["name"] == call.name)
            .ok_or("model requested an undeclared tool")?;
        if !chat_tools::validator(&definition["parameters"])?.is_valid(&call.arguments) {
            return Err("model tool arguments do not match the declared schema".into());
        }
        output.push(json!({"type":"function_call","id":format!("fc_{id}_{index}"),"call_id":format!("call_{id}_{index}"),"name":call.name,"arguments":call.arguments.to_string(),"status":"completed"}));
    }
    Ok(
        json!({"id":id,"object":"response","model":request.model,"status":if complete {"completed"} else {"incomplete"},"output":output,"incomplete_details":if complete {Value::Null} else {json!({"reason":"max_output_tokens"})},"usage":{"input_tokens":generated.metrics.prompt_tokens,"output_tokens":generated.generated_token_ids.len(),"total_tokens":generated.metrics.prompt_tokens+generated.generated_token_ids.len()},"metrics":generated.metrics}),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn rejects_orphan_results_and_unsupported_request_semantics() {
        for extra in [
            json!({"input":[{"type":"function_call_output","call_id":"unknown","output":"x"}]}),
            json!({"temperature":0.8}),
            json!({"seed":42}),
            json!({"store":true}),
            json!({"max_output_tokens":0}),
        ] {
            let mut request = json!({"model":"control","input":"hello"});
            request
                .as_object_mut()
                .unwrap()
                .extend(extra.as_object().unwrap().clone());
            assert!(messages(&serde_json::from_value::<Request>(request).unwrap()).is_err());
        }
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
        use crate::chat_generation::ChatGenerationMetrics;
        let request: Request = serde_json::from_value(json!({"model":"control","input":"hello","tools":[{"type":"function","name":"read_file","parameters":{"type":"object","properties":{"path":{"type":"string"}},"required":["path"],"additionalProperties":false}}]})).unwrap();
        let mut generated = crate::chat_generation::ChatGeneration {
            text: String::new(),
            generated_token_ids: vec![1],
            finish_reason: ChatFinishReason::Eos,
            metrics: ChatGenerationMetrics {
                session_load_ms: 0.0,
                render_ms: 0.0,
                prefill_ms: 0.0,
                time_to_first_token_ms: None,
                decode_ms: vec![],
                decode_total_ms: 0.0,
                prompt_tokens: 1,
                generated_tokens: 1,
            },
        };
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
}
