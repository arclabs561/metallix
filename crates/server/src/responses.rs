//! Text-only Responses protocol boundary shared by local model backends.

use std::{
    collections::{HashMap, HashSet},
    io::{BufWriter, Write},
    net::{SocketAddr, TcpListener},
    path::Path,
    process::ExitCode,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
        mpsc::{Receiver, SyncSender, sync_channel},
    },
    thread,
    time::Duration,
};

use serde::Deserialize;
use serde_json::{Value, json};

use crate::{
    chat_cli::message,
    chat_generation::{
        ChatBackend, ChatFinishReason, ChatGenerationError, ChatMessage, ChatRequest, ChatRole,
        ChatSession, ChatToolCall, ResidentChatLimits,
    },
    chat_tools,
    http_transport::{Connection, TransportLimits},
};

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

fn json_response(mut connection: Connection, status: u16, value: &Value) {
    connection.begin_response();
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
        "HTTP/1.1 {status} {reason}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    );
    let _ = writer.flush();
}

fn busy_response(connection: Connection) {
    json_response(
        connection,
        503,
        &json!({"error":{"code":"server_busy","message":"one generation is already active"}}),
    );
}

fn unavailable_response(connection: Connection) {
    json_response(
        connection,
        503,
        &json!({"error":{"code":"model_worker_unavailable","message":"the model worker is unavailable"}}),
    );
}

struct Admission {
    occupied: Arc<AtomicBool>,
}

struct WorkerLiveness {
    alive: Arc<AtomicBool>,
}

impl Drop for WorkerLiveness {
    fn drop(&mut self) {
        self.alive.store(false, Ordering::Release);
    }
}

impl Admission {
    fn try_acquire(occupied: &Arc<AtomicBool>) -> Option<Self> {
        occupied
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .ok()
            .map(|_previous| Self {
                occupied: Arc::clone(occupied),
            })
    }
}

impl Drop for Admission {
    fn drop(&mut self) {
        self.occupied.store(false, Ordering::Release);
    }
}

struct GenerationJob {
    connection: Connection,
    request: Request,
    messages: Vec<ChatMessage>,
    tools: Vec<Value>,
    id: String,
    generation_timeout: Duration,
    _admission: Admission,
}

fn worker_loop(session: &mut dyn ChatBackend, jobs: Receiver<GenerationJob>) {
    for job in jobs {
        let GenerationJob {
            connection,
            request,
            messages,
            tools,
            id,
            generation_timeout,
            _admission,
        } = job;
        if let Err(error) = respond(
            connection,
            &request,
            &messages,
            &tools,
            session,
            &id,
            generation_timeout,
        ) {
            eprintln!("response failed: {error}");
        }
    }
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

pub(crate) fn serve(
    model: &Path,
    model_id: &str,
    address: SocketAddr,
    limits: ResidentChatLimits,
    generation_timeout: Duration,
) -> ExitCode {
    match serve_inner(model, model_id, address, limits, generation_timeout) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("mx serve: {error}");
            ExitCode::FAILURE
        }
    }
}

fn serve_inner(
    model: &Path,
    model_id: &str,
    address: SocketAddr,
    limits: ResidentChatLimits,
    generation_timeout: Duration,
) -> Result<(), String> {
    if !address.ip().is_loopback() {
        return Err("this experimental server binds only to loopback".into());
    }
    let (job_sender, job_receiver) = sync_channel(0);
    let (startup_sender, startup_receiver) = sync_channel(1);
    let worker_model = model.to_owned();
    let worker_alive = Arc::new(AtomicBool::new(false));
    let worker_liveness = Arc::clone(&worker_alive);
    let worker = thread::spawn(move || {
        let mut session = match ChatSession::load(&worker_model, limits) {
            Ok(session) => session,
            Err(error) => {
                let _ = startup_sender.send(Err(error));
                return;
            }
        };
        let liveness = WorkerLiveness {
            alive: worker_liveness,
        };
        liveness.alive.store(true, Ordering::Release);
        let load_ms = (&session as &dyn ChatBackend).load_ms();
        if startup_sender.send(Ok(load_ms)).is_ok() {
            worker_loop(&mut session, job_receiver);
        }
    });
    let session_load_ms = match startup_receiver.recv() {
        Ok(Ok(load_ms)) => load_ms,
        Ok(Err(error)) => {
            let _ = worker.join();
            return Err(error);
        }
        Err(_) => {
            let _ = worker.join();
            return Err(String::from("model worker ended before startup"));
        }
    };
    let server = match TcpListener::bind(address) {
        Ok(server) => server,
        Err(error) => {
            drop(job_sender);
            let _ = worker.join();
            return Err(error.to_string());
        }
    };
    eprintln!(
        "mx listening on http://{address}; model={model_id}; single request; {} total tokens; kv_budget_bytes={}; load_ms={:.2}",
        limits.context_tokens(),
        limits.kv_budget_bytes(),
        session_load_ms
    );
    let occupied = Arc::new(AtomicBool::new(false));
    let outcome = serve_listener(
        &server,
        model_id,
        &job_sender,
        &occupied,
        &worker_alive,
        generation_timeout,
        None,
    );
    drop(job_sender);
    worker
        .join()
        .map_err(|_| String::from("model worker panicked"))?;
    outcome
}

fn serve_listener(
    server: &TcpListener,
    model_id: &str,
    job_sender: &SyncSender<GenerationJob>,
    occupied: &Arc<AtomicBool>,
    worker_alive: &Arc<AtomicBool>,
    generation_timeout: Duration,
    request_limit: Option<usize>,
) -> Result<(), String> {
    serve_listener_with_limits(
        server,
        model_id,
        job_sender,
        occupied,
        worker_alive,
        generation_timeout,
        TransportLimits::default(),
        request_limit,
    )
}

#[allow(
    clippy::too_many_arguments,
    reason = "private transport-limit seam shares the production acceptor with bounded socket tests"
)]
fn serve_listener_with_limits(
    server: &TcpListener,
    model_id: &str,
    job_sender: &SyncSender<GenerationJob>,
    occupied: &Arc<AtomicBool>,
    worker_alive: &Arc<AtomicBool>,
    generation_timeout: Duration,
    transport_limits: TransportLimits,
    request_limit: Option<usize>,
) -> Result<(), String> {
    for (index, socket) in server
        .incoming()
        .take(request_limit.unwrap_or(usize::MAX))
        .enumerate()
    {
        let socket = socket.map_err(|error| error.to_string())?;
        let connection = Connection::accept(socket, transport_limits);
        if !worker_alive.load(Ordering::Acquire) {
            unavailable_response(connection);
            return Err(String::from("model worker is unavailable"));
        }
        let Some(admission) = Admission::try_acquire(occupied) else {
            busy_response(connection);
            continue;
        };
        let mut connection = connection;
        let request = match connection.read_request() {
            Ok(request) => request,
            Err(error) => {
                json_response(
                    connection,
                    error.status,
                    &json!({"error":{"message":error.message}}),
                );
                continue;
            }
        };
        match (request.method.as_str(), request.path.as_str()) {
            ("GET", "/healthz") => {
                json_response(connection, 200, &json!({"status":"ready"}));
                continue;
            }
            ("GET", "/v1/models") => {
                json_response(
                    connection,
                    200,
                    &json!({"object":"list","data":[{"id":model_id,"object":"model","owned_by":"local"}]}),
                );
                continue;
            }
            ("POST", "/v1/responses") => {}
            _ => {
                json_response(
                    connection,
                    404,
                    &json!({"error":{"message":"unknown endpoint"}}),
                );
                continue;
            }
        }
        let parsed =
            serde_json::from_slice::<Request>(&request.body).map_err(|error| error.to_string());
        let parsed = match parsed {
            Ok(parsed) => parsed,
            Err(error) => {
                json_response(connection, 400, &json!({"error":{"message":error}}));
                continue;
            }
        };
        if parsed.model != model_id {
            json_response(
                connection,
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
                json_response(connection, 400, &json!({"error":{"message":error}}));
                continue;
            }
        };
        let id = format!("resp_{}_{}", std::process::id(), index);
        let job = GenerationJob {
            connection,
            request: parsed,
            messages,
            tools,
            id,
            generation_timeout,
            _admission: admission,
        };
        if let Err(error) = job_sender.send(job) {
            unavailable_response(error.0.connection);
            return Err(String::from("model worker is unavailable"));
        }
    }
    Ok(())
}

#[allow(
    clippy::too_many_lines,
    reason = "one ordered response event lifecycle"
)]
fn respond(
    request: Connection,
    parsed: &Request,
    messages: &[ChatMessage],
    tools: &[Value],
    session: &mut dyn ChatBackend,
    id: &str,
    generation_timeout: Duration,
) -> Result<(), String> {
    let mut sequence = 0;
    let mut writer = if parsed.stream {
        let mut writer = BufWriter::new(request);
        writer.get_mut().begin_response();
        write!(writer,"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: close\r\n\r\n").map_err(|e| e.to_string())?;
        event(
            &mut writer,
            &mut sequence,
            json!({"type":"response.created","response":{"id":id,"object":"response","status":"in_progress","output":[]}}),
        )?;
        Some(writer)
    } else {
        respond_json(
            request,
            parsed,
            messages,
            tools,
            session,
            id,
            generation_timeout,
        );
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
    let generated = session.generate_with_timeout(ChatRequest {messages,tools,max_tokens:parsed.max_output_tokens.unwrap_or(128),enable_thinking:false,reasoning_effort:None}, generation_timeout, &mut |delta| {
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
        Err(ChatGenerationError::DeadlineExceeded) => {
            event(
                writer.as_mut().expect("stream writer"),
                &mut sequence,
                json!({"type":"response.failed","response":{"id":id,"status":"failed","error":{"code":"generation_timeout","message":"generation time budget exceeded"}}}),
            )?;
            return Ok(());
        }
        Err(error) => {
            event(
                writer.as_mut().expect("stream writer"),
                &mut sequence,
                json!({"type":"response.failed","response":{"id":id,"status":"failed","error":{"code":"generation_failed","message":error.to_string()}}}),
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
    request: Connection,
    parsed: &Request,
    messages: &[ChatMessage],
    tools: &[Value],
    session: &mut dyn ChatBackend,
    id: &str,
    generation_timeout: Duration,
) {
    let result = session
        .generate_with_timeout(
            ChatRequest {
                messages,
                tools,
                max_tokens: parsed.max_output_tokens.unwrap_or(128),
                enable_thinking: false,
                reasoning_effort: None,
            },
            generation_timeout,
            &mut |_| Ok(()),
        )
        .and_then(|generated| {
            response_value(parsed, &generated, id).map_err(ChatGenerationError::Message)
        });
    match result {
        Ok(response) => json_response(request, 200, &response),
        Err(ChatGenerationError::DeadlineExceeded) => json_response(
            request,
            408,
            &json!({"error":{"code":"generation_timeout","message":"generation time budget exceeded"}}),
        ),
        Err(error) => json_response(
            request,
            400,
            &json!({"error":{"message":error.to_string()}}),
        ),
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
    use std::{
        env,
        io::Read as _,
        net::{Shutdown, TcpListener, TcpStream},
        process::{Child, Command, Stdio},
        sync::{
            Arc,
            atomic::{AtomicBool, Ordering},
            mpsc::{Receiver, SyncSender, sync_channel},
        },
        thread,
        time::{Duration, Instant},
    };

    use super::*;
    use proptest::prelude::*;

    #[derive(Default)]
    struct DeadlineBackend {
        calls: usize,
    }

    impl ChatBackend for DeadlineBackend {
        fn load_ms(&self) -> f64 {
            0.0
        }

        fn generate_with_timeout(
            &mut self,
            _request: ChatRequest<'_>,
            _timeout: Duration,
            on_token: &mut dyn FnMut(&str) -> Result<(), String>,
        ) -> Result<crate::chat_generation::ChatGeneration, ChatGenerationError> {
            self.calls += 1;
            on_token("partial").map_err(ChatGenerationError::Message)?;
            Err(ChatGenerationError::DeadlineExceeded)
        }
    }

    struct BlockingBackend {
        turns: usize,
        entered: SyncSender<()>,
        release: Receiver<()>,
    }

    struct BackpressureBackend {
        turns: usize,
        started: SyncSender<()>,
        write_failure: SyncSender<(Duration, String)>,
    }

    impl ChatBackend for BackpressureBackend {
        fn load_ms(&self) -> f64 {
            0.0
        }

        fn generate_with_timeout(
            &mut self,
            _request: ChatRequest<'_>,
            _timeout: Duration,
            on_token: &mut dyn FnMut(&str) -> Result<(), String>,
        ) -> Result<crate::chat_generation::ChatGeneration, ChatGenerationError> {
            self.turns += 1;
            if self.turns == 1 {
                self.started
                    .send(())
                    .map_err(|_| ChatGenerationError::Message(String::from("test start closed")))?;
                let payload = "x".repeat(256 * 1024);
                for _ in 0..32 {
                    let began = Instant::now();
                    if let Err(error) = on_token(&payload) {
                        self.write_failure
                            .send((began.elapsed(), error.clone()))
                            .map_err(|_| {
                                ChatGenerationError::Message(String::from("test result closed"))
                            })?;
                        return Err(ChatGenerationError::Message(error));
                    }
                }
                return Err(ChatGenerationError::Message(String::from(
                    "bounded backpressure payload was fully accepted",
                )));
            }
            on_token("recovered").map_err(ChatGenerationError::Message)?;
            Ok(crate::chat_generation::ChatGeneration {
                text: String::from("recovered"),
                generated_token_ids: vec![1],
                finish_reason: ChatFinishReason::Eos,
                metrics: crate::chat_generation::ChatGenerationMetrics {
                    context_tokens: 2_048,
                    planned_kv_bytes: 0,
                    session_load_ms: 0.0,
                    render_ms: 0.0,
                    prefill_ms: 0.0,
                    time_to_first_token_ms: Some(0.0),
                    decode_ms: vec![],
                    decode_total_ms: 0.0,
                    prompt_tokens: 1,
                    generated_tokens: 1,
                },
            })
        }
    }

    struct StalledReader {
        child: Option<Child>,
    }

    impl StalledReader {
        fn start(address: std::net::SocketAddr) -> Self {
            const STALLED_READER: &str = r#"
import socket
import sys

host, port = sys.argv[1], int(sys.argv[2])
body = b'{"model":"control","input":"hold","stream":true}'
request = (
    b"POST /v1/responses HTTP/1.1\r\nHost: localhost\r\nContent-Length: "
    + str(len(body)).encode("ascii")
    + b"\r\n\r\n"
    + body
)
stream = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
stream.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1024)
stream.settimeout(2)
stream.connect((host, port))
stream.sendall(request)
stream.shutdown(socket.SHUT_WR)
if sys.stdin.buffer.read(1) != b"r":
    raise RuntimeError("slow-reader release was not received")
stream.close()
"#;
            let host = address.ip().to_string();
            let port = address.port().to_string();
            let child = Command::new("python3")
                .args(["-c", STALLED_READER, &host, &port])
                .stdin(Stdio::piped())
                .stdout(Stdio::null())
                .stderr(Stdio::piped())
                .spawn()
                .expect("start bounded slow reader");
            Self { child: Some(child) }
        }

        fn release(&mut self) {
            let mut child = self.child.take().expect("slow reader remains running");
            std::io::Write::write_all(child.stdin.as_mut().expect("slow reader stdin"), b"r")
                .expect("release slow reader");
            let output = child.wait_with_output().expect("wait for slow reader");
            assert!(
                output.status.success(),
                "slow reader failed: {}",
                String::from_utf8_lossy(&output.stderr)
            );
        }
    }

    impl Drop for StalledReader {
        fn drop(&mut self) {
            if let Some(mut child) = self.child.take() {
                let _ = child.kill();
                let _ = child.wait();
            }
        }
    }

    impl ChatBackend for BlockingBackend {
        fn load_ms(&self) -> f64 {
            0.0
        }

        fn generate_with_timeout(
            &mut self,
            _request: ChatRequest<'_>,
            _timeout: Duration,
            on_token: &mut dyn FnMut(&str) -> Result<(), String>,
        ) -> Result<crate::chat_generation::ChatGeneration, ChatGenerationError> {
            self.turns += 1;
            let text = if self.turns == 1 {
                "holding"
            } else {
                "recovered"
            };
            on_token(text).map_err(ChatGenerationError::Message)?;
            if self.turns == 1 {
                self.entered.send(()).map_err(|_| {
                    ChatGenerationError::Message(String::from("test barrier closed"))
                })?;
                self.release
                    .recv_timeout(Duration::from_secs(2))
                    .map_err(|_| {
                        ChatGenerationError::Message(String::from("test barrier timed out"))
                    })?;
            }
            Ok(crate::chat_generation::ChatGeneration {
                text: text.into(),
                generated_token_ids: vec![1],
                finish_reason: ChatFinishReason::Eos,
                metrics: crate::chat_generation::ChatGenerationMetrics {
                    context_tokens: 2_048,
                    planned_kv_bytes: 0,
                    session_load_ms: 0.0,
                    render_ms: 0.0,
                    prefill_ms: 0.0,
                    time_to_first_token_ms: Some(0.0),
                    decode_ms: vec![],
                    decode_total_ms: 0.0,
                    prompt_tokens: 1,
                    generated_tokens: 1,
                },
            })
        }
    }

    /// Reads one bounded, fixed-length HTTP response without relying on EOF.
    ///
    /// Busy admission deliberately responds before consuming a peer's request
    /// body. A peer that half-closes after writing can therefore observe a
    /// reset once the complete response has arrived; EOF is not the framing
    /// boundary for this JSON response.
    fn fixed_http_response(stream: &mut TcpStream) -> (u16, Vec<u8>) {
        const MAX_HEADER_BYTES: usize = 16 * 1024;
        const MAX_BODY_BYTES: usize = 1024 * 1024;

        let mut wire = Vec::new();
        let mut chunk = [0_u8; 1024];
        let header_end = loop {
            assert!(
                wire.len() <= MAX_HEADER_BYTES,
                "fixed response headers exceed {MAX_HEADER_BYTES} bytes"
            );
            if let Some(index) = wire.windows(4).position(|window| window == b"\r\n\r\n") {
                break index + 4;
            }
            let read = stream
                .read(&mut chunk)
                .expect("read fixed response headers");
            assert!(read > 0, "fixed response ended before headers");
            wire.extend_from_slice(&chunk[..read]);
        };
        assert!(
            header_end <= MAX_HEADER_BYTES,
            "fixed response headers exceed {MAX_HEADER_BYTES} bytes"
        );

        let headers = std::str::from_utf8(&wire[..header_end]).expect("response headers UTF-8");
        let mut lines = headers.split("\r\n");
        let status_line = lines.next().expect("response status line");
        let mut status_parts = status_line.split_whitespace();
        assert_eq!(
            status_parts.next(),
            Some("HTTP/1.1"),
            "response HTTP version"
        );
        let status = status_parts
            .next()
            .expect("response status code")
            .parse::<u16>()
            .expect("numeric response status");
        assert!(status_parts.next().is_some(), "response reason phrase");

        let mut content_length = None;
        for line in lines.take_while(|line| !line.is_empty()) {
            let (name, value) = line.split_once(':').expect("well-formed response header");
            if name.eq_ignore_ascii_case("content-length") {
                let length = value
                    .trim()
                    .parse::<usize>()
                    .expect("numeric content length");
                assert!(
                    content_length.replace(length).is_none(),
                    "response has one content length"
                );
            }
        }
        let content_length = content_length.expect("fixed response content length");
        assert!(
            content_length <= MAX_BODY_BYTES,
            "fixed response body exceeds {MAX_BODY_BYTES} bytes"
        );
        let response_end = header_end
            .checked_add(content_length)
            .expect("response length overflow");
        assert!(
            wire.len() <= response_end,
            "response exceeds declared content length"
        );
        while wire.len() < response_end {
            let remaining = response_end - wire.len();
            let read_len = remaining.min(chunk.len());
            let read = stream
                .read(&mut chunk[..read_len])
                .expect("read fixed response body");
            assert!(read > 0, "fixed response ended before declared body");
            wire.extend_from_slice(&chunk[..read]);
        }
        (status, wire[header_end..].to_vec())
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
    fn streaming_generation_deadline_emits_one_terminal_failure() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let client = thread::spawn(move || {
            let body = br#"{"model":"control","input":"hello","stream":true}"#;
            let mut stream = TcpStream::connect(address).unwrap();
            stream
                .set_read_timeout(Some(Duration::from_secs(2)))
                .unwrap();
            write!(
                stream,
                "POST /v1/responses HTTP/1.1\r\nHost: localhost\r\nContent-Length: {}\r\n\r\n",
                body.len()
            )
            .unwrap();
            stream.write_all(body).unwrap();
            stream.shutdown(Shutdown::Write).unwrap();
            let mut response = Vec::new();
            stream.read_to_end(&mut response).unwrap();
            response
        });
        let (stream, _) = listener.accept().unwrap();
        let mut connection = Connection::accept(stream, TransportLimits::default());
        let wire_request = connection.read_request().unwrap();
        let request: Request = serde_json::from_slice(&wire_request.body).unwrap();
        let messages = messages(&request).unwrap();
        let tools = tools(&request).unwrap();
        let mut backend = DeadlineBackend::default();

        respond(
            connection,
            &request,
            &messages,
            &tools,
            &mut backend,
            "deadline",
            Duration::from_secs(1),
        )
        .unwrap();
        assert_eq!(backend.calls, 1);

        let wire = String::from_utf8(client.join().unwrap()).unwrap();
        let (headers, payload) = wire.split_once("\r\n\r\n").unwrap();
        assert!(headers.starts_with("HTTP/1.1 200 OK\r\n"));
        let events: Vec<Value> = payload
            .split("\n\n")
            .filter(|frame| !frame.is_empty())
            .map(|frame| {
                let (_, data) = frame.split_once('\n').unwrap();
                serde_json::from_str(data.strip_prefix("data: ").unwrap()).unwrap()
            })
            .collect();
        assert_eq!(
            events
                .iter()
                .map(|event| event["type"].as_str())
                .collect::<Vec<_>>(),
            vec![
                Some("response.created"),
                Some("response.output_item.added"),
                Some("response.content_part.added"),
                Some("response.output_text.delta"),
                Some("response.failed"),
            ]
        );
        assert_eq!(
            events
                .iter()
                .map(|event| event["sequence_number"].as_u64())
                .collect::<Vec<_>>(),
            vec![Some(0), Some(1), Some(2), Some(3), Some(4)]
        );
        assert_eq!(events[3]["delta"], "partial");
        assert_eq!(events[4]["response"]["status"], "failed");
        assert_eq!(events[4]["response"]["error"]["code"], "generation_timeout");
    }

    #[test]
    #[allow(
        clippy::too_many_lines,
        reason = "the ordered socket lifecycle is the assertion under test"
    )]
    fn concurrent_arrival_is_rejected_while_worker_holds_the_model_then_recovers() {
        fn request_body() -> Vec<u8> {
            serde_json::to_vec(&json!({
                "model": "control",
                "input": "hello",
                "stream": true,
                "max_output_tokens": 8,
            }))
            .expect("request JSON")
        }

        fn request(stream: &mut TcpStream, body: &[u8]) {
            write!(
                stream,
                "POST /v1/responses HTTP/1.1\r\nHost: localhost\r\nContent-Length: {}\r\n\r\n",
                body.len()
            )
            .expect("request header");
            stream.write_all(body).expect("request body");
            stream
                .shutdown(Shutdown::Write)
                .expect("request half-close");
        }

        let listener = TcpListener::bind("127.0.0.1:0").expect("loopback listener");
        let address = listener.local_addr().expect("listener address");
        let (job_sender, job_receiver) = sync_channel(0);
        let (entered_sender, entered_receiver) = sync_channel(1);
        let (release_sender, release_receiver) = sync_channel(1);
        let worker = thread::spawn(move || {
            let mut backend = BlockingBackend {
                turns: 0,
                entered: entered_sender,
                release: release_receiver,
            };
            worker_loop(&mut backend, job_receiver);
        });
        let occupied = Arc::new(AtomicBool::new(false));
        let server_occupied = Arc::clone(&occupied);
        let worker_alive = Arc::new(AtomicBool::new(true));
        let server_worker_alive = Arc::clone(&worker_alive);
        let server_sender = job_sender.clone();
        let server = thread::spawn(move || {
            serve_listener(
                &listener,
                "control",
                &server_sender,
                &server_occupied,
                &server_worker_alive,
                Duration::from_secs(2),
                Some(4),
            )
        });

        let mut malformed = TcpStream::connect(address).expect("connect malformed request");
        malformed
            .set_read_timeout(Some(Duration::from_secs(2)))
            .expect("bound malformed reads");
        malformed
            .write_all(
                b"POST /v1/responses HTTP/1.1\r\nHost: localhost\r\nContent-Length: 1\r\n\r\n{",
            )
            .expect("write malformed request");
        malformed
            .shutdown(Shutdown::Write)
            .expect("malformed half-close");
        let mut malformed_wire = Vec::new();
        malformed
            .read_to_end(&mut malformed_wire)
            .expect("read malformed response");
        assert!(
            String::from_utf8(malformed_wire)
                .expect("malformed response UTF-8")
                .starts_with("HTTP/1.1 400 Bad Request\r\n")
        );

        let body = request_body();
        let mut primary = TcpStream::connect(address).expect("connect primary request");
        primary
            .set_read_timeout(Some(Duration::from_secs(2)))
            .expect("bound primary reads");
        request(&mut primary, &body);
        let mut primary_prefix = Vec::new();
        let mut chunk = [0_u8; 1024];
        while !primary_prefix
            .windows(b"response.output_text.delta".len())
            .any(|window| window == b"response.output_text.delta")
        {
            let read = primary.read(&mut chunk).expect("read primary delta");
            assert!(read > 0, "primary ended before its delta");
            primary_prefix.extend_from_slice(&chunk[..read]);
        }
        entered_receiver
            .recv_timeout(Duration::from_secs(2))
            .expect("worker holds after the real callback delta");

        let mut busy = TcpStream::connect(address).expect("connect concurrent request");
        busy.set_read_timeout(Some(Duration::from_secs(2)))
            .expect("bound busy reads");
        request(&mut busy, &body);
        let (busy_status, busy_body) = fixed_http_response(&mut busy);
        assert_eq!(busy_status, 503);
        assert_eq!(
            serde_json::from_slice::<Value>(&busy_body).expect("busy response JSON"),
            json!({"error":{"code":"server_busy","message":"one generation is already active"}})
        );

        release_sender.send(()).expect("release worker");
        primary
            .read_to_end(&mut primary_prefix)
            .expect("read primary completion");
        assert!(
            String::from_utf8(primary_prefix)
                .expect("primary UTF-8")
                .contains(r#""type":"response.completed""#)
        );

        let mut recovery = TcpStream::connect(address).expect("connect recovery request");
        recovery
            .set_read_timeout(Some(Duration::from_secs(2)))
            .expect("bound recovery reads");
        request(&mut recovery, &body);
        let mut recovery_wire = Vec::new();
        recovery
            .read_to_end(&mut recovery_wire)
            .expect("read recovery response");
        let recovery_wire = String::from_utf8(recovery_wire).expect("recovery UTF-8");
        assert!(recovery_wire.starts_with("HTTP/1.1 200 OK\r\n"));
        assert!(recovery_wire.contains(r#""type":"response.completed""#));
        assert!(recovery_wire.contains("recovered"));

        server
            .join()
            .expect("join bounded acceptor")
            .expect("acceptor result");
        drop(job_sender);
        worker.join().expect("join bounded worker");
        assert!(!occupied.load(Ordering::Acquire));
        assert!(worker_alive.load(Ordering::Acquire));
    }

    #[test]
    #[allow(
        clippy::too_many_lines,
        reason = "the ordered slow-reader and recovery lifecycle is the assertion under test"
    )]
    fn slow_reader_write_deadline_releases_admission_and_worker_recovers() {
        const WRITE_IDLE: Duration = Duration::from_millis(250);
        const RESPONSE_DEADLINE: Duration = Duration::from_secs(2);

        fn wait_for_admission_release(occupied: &AtomicBool) {
            let deadline = Instant::now() + Duration::from_secs(2);
            while occupied.load(Ordering::Acquire) {
                assert!(
                    Instant::now() < deadline,
                    "write failure did not release admission"
                );
                thread::yield_now();
            }
        }

        let listener = TcpListener::bind("127.0.0.1:0").expect("loopback listener");
        let address = listener.local_addr().expect("listener address");
        let (job_sender, job_receiver) = sync_channel(0);
        let (started_sender, started_receiver) = sync_channel(1);
        let (failure_sender, failure_receiver) = sync_channel(1);
        let worker = thread::spawn(move || {
            let mut backend = BackpressureBackend {
                turns: 0,
                started: started_sender,
                write_failure: failure_sender,
            };
            worker_loop(&mut backend, job_receiver);
            backend.turns
        });
        let occupied = Arc::new(AtomicBool::new(false));
        let server_occupied = Arc::clone(&occupied);
        let worker_alive = Arc::new(AtomicBool::new(true));
        let server_worker_alive = Arc::clone(&worker_alive);
        let server_sender = job_sender.clone();
        let server = thread::spawn(move || {
            serve_listener_with_limits(
                &listener,
                "control",
                &server_sender,
                &server_occupied,
                &server_worker_alive,
                Duration::from_secs(2),
                TransportLimits {
                    write_idle: WRITE_IDLE,
                    response_deadline: RESPONSE_DEADLINE,
                    ..TransportLimits::default()
                },
                Some(2),
            )
        });

        let mut stalled_reader = StalledReader::start(address);
        started_receiver
            .recv_timeout(Duration::from_secs(2))
            .expect("worker begins the stalled response");
        let (elapsed, error) = failure_receiver
            .recv_timeout(RESPONSE_DEADLINE + Duration::from_secs(1))
            .expect("stalled response reports a callback write failure");
        let lower_error = error.to_ascii_lowercase();
        let pressure_error = lower_error.contains("timed out")
            || lower_error.contains("would block")
            || lower_error.contains("temporarily unavailable");
        assert!(
            elapsed >= WRITE_IDLE || pressure_error,
            "callback failed before the write idle bound without a pressure error: {error} after {elapsed:?}"
        );
        stalled_reader.release();
        wait_for_admission_release(&occupied);

        let body = br#"{"model":"control","input":"recover","stream":true}"#;
        let mut recovery = TcpStream::connect(address).expect("connect recovery request");
        recovery
            .set_read_timeout(Some(Duration::from_secs(2)))
            .expect("bound recovery reads");
        write!(
            recovery,
            "POST /v1/responses HTTP/1.1\r\nHost: localhost\r\nContent-Length: {}\r\n\r\n",
            body.len()
        )
        .expect("recovery request header");
        recovery.write_all(body).expect("recovery request body");
        recovery
            .shutdown(Shutdown::Write)
            .expect("recovery half-close");
        let mut recovery_wire = Vec::new();
        recovery
            .read_to_end(&mut recovery_wire)
            .expect("read recovery response");
        let recovery_wire = String::from_utf8(recovery_wire).expect("recovery response UTF-8");
        assert!(recovery_wire.starts_with("HTTP/1.1 200 OK\r\n"));
        assert!(recovery_wire.contains(r#""type":"response.completed""#));
        assert!(recovery_wire.contains("recovered"));

        server
            .join()
            .expect("join bounded acceptor")
            .expect("acceptor result");
        drop(job_sender);
        assert_eq!(worker.join().expect("join worker"), 2);
        assert!(!occupied.load(Ordering::Acquire));
    }

    #[test]
    fn generation_error_releases_admission_for_the_next_request() {
        fn request(address: std::net::SocketAddr) -> String {
            let body = br#"{"model":"control","input":"hello","stream":true}"#;
            let mut stream = TcpStream::connect(address).expect("connect request");
            stream
                .set_read_timeout(Some(Duration::from_secs(2)))
                .expect("bound request reads");
            write!(
                stream,
                "POST /v1/responses HTTP/1.1\r\nHost: localhost\r\nContent-Length: {}\r\n\r\n",
                body.len()
            )
            .expect("request header");
            stream.write_all(body).expect("request body");
            stream
                .shutdown(Shutdown::Write)
                .expect("request half-close");
            let mut wire = Vec::new();
            stream.read_to_end(&mut wire).expect("read response");
            String::from_utf8(wire).expect("response UTF-8")
        }

        let listener = TcpListener::bind("127.0.0.1:0").expect("loopback listener");
        let address = listener.local_addr().expect("listener address");
        let (sender, receiver) = sync_channel(0);
        let worker = thread::spawn(move || {
            let mut backend = DeadlineBackend::default();
            worker_loop(&mut backend, receiver);
            backend.calls
        });
        let occupied = Arc::new(AtomicBool::new(false));
        let alive = Arc::new(AtomicBool::new(true));
        let server_occupied = Arc::clone(&occupied);
        let server_alive = Arc::clone(&alive);
        let server_sender = sender.clone();
        let server = thread::spawn(move || {
            serve_listener(
                &listener,
                "control",
                &server_sender,
                &server_occupied,
                &server_alive,
                Duration::from_secs(2),
                Some(2),
            )
        });

        for _ in 0..2 {
            let wire = request(address);
            assert!(wire.starts_with("HTTP/1.1 200 OK\r\n"));
            assert!(wire.contains(r#""type":"response.failed""#));
            assert!(wire.contains(r#""code":"generation_timeout""#));
        }
        server
            .join()
            .expect("join error acceptor")
            .expect("error acceptor result");
        drop(sender);
        assert_eq!(worker.join().expect("join error worker"), 2);
        assert!(!occupied.load(Ordering::Acquire));
    }

    #[test]
    fn unavailable_worker_rejects_without_reading_request_headers() {
        let listener = TcpListener::bind("127.0.0.1:0").expect("loopback listener");
        let address = listener.local_addr().expect("listener address");
        let (sender, receiver) = sync_channel::<GenerationJob>(0);
        drop(receiver);
        let occupied = Arc::new(AtomicBool::new(false));
        let unavailable = Arc::new(AtomicBool::new(false));
        let server_occupied = Arc::clone(&occupied);
        let server_unavailable = Arc::clone(&unavailable);
        let server = thread::spawn(move || {
            serve_listener(
                &listener,
                "control",
                &sender,
                &server_occupied,
                &server_unavailable,
                Duration::from_secs(2),
                Some(1),
            )
        });

        let mut client = TcpStream::connect(address).expect("connect without request headers");
        client
            .set_read_timeout(Some(Duration::from_secs(2)))
            .expect("bound unavailable read");
        let mut wire = Vec::new();
        client
            .read_to_end(&mut wire)
            .expect("read unavailable response");
        let wire = String::from_utf8(wire).expect("unavailable UTF-8");
        assert!(wire.starts_with("HTTP/1.1 503 Service Unavailable\r\n"));
        assert!(wire.contains(r#""code":"model_worker_unavailable""#));
        assert!(server.join().expect("join unavailable acceptor").is_err());
        assert!(!occupied.load(Ordering::Acquire));
    }

    #[test]
    fn dropped_worker_receiver_returns_unavailable_after_valid_intake() {
        let listener = TcpListener::bind("127.0.0.1:0").expect("loopback listener");
        let address = listener.local_addr().expect("listener address");
        let (sender, receiver) = sync_channel::<GenerationJob>(0);
        drop(receiver);
        let occupied = Arc::new(AtomicBool::new(false));
        let alive = Arc::new(AtomicBool::new(true));
        let server_occupied = Arc::clone(&occupied);
        let server_alive = Arc::clone(&alive);
        let server = thread::spawn(move || {
            serve_listener(
                &listener,
                "control",
                &sender,
                &server_occupied,
                &server_alive,
                Duration::from_secs(2),
                Some(1),
            )
        });

        let body = br#"{"model":"control","input":"hello","stream":true}"#;
        let mut client = TcpStream::connect(address).expect("connect valid request");
        client
            .set_read_timeout(Some(Duration::from_secs(2)))
            .expect("bound unavailable read");
        write!(
            client,
            "POST /v1/responses HTTP/1.1\r\nHost: localhost\r\nContent-Length: {}\r\n\r\n",
            body.len()
        )
        .expect("valid request header");
        client.write_all(body).expect("valid request body");
        client.shutdown(Shutdown::Write).expect("valid half-close");
        let mut wire = Vec::new();
        client
            .read_to_end(&mut wire)
            .expect("read unavailable response");
        let wire = String::from_utf8(wire).expect("unavailable UTF-8");
        assert!(wire.starts_with("HTTP/1.1 503 Service Unavailable\r\n"));
        assert!(wire.contains(r#""code":"model_worker_unavailable""#));
        assert!(server.join().expect("join unavailable acceptor").is_err());
        assert!(!occupied.load(Ordering::Acquire));
    }

    mod checkpoint_reset {
        use super::*;
        struct ObservedSession<'a> {
            session: &'a mut ChatSession,
            callback_failed: bool,
            generation_failed: bool,
        }
        impl ChatBackend for ObservedSession<'_> {
            fn load_ms(&self) -> f64 {
                self.session.load_ms()
            }
            fn generate_with_timeout(
                &mut self,
                request: ChatRequest<'_>,
                timeout: Duration,
                on_token: &mut dyn FnMut(&str) -> Result<(), String>,
            ) -> Result<crate::chat_generation::ChatGeneration, ChatGenerationError> {
                let mut failed = false;
                let result = self
                    .session
                    .generate_with_timeout(request, timeout, &mut |delta| {
                        let written = on_token(delta);
                        failed |= written.is_err();
                        written
                    });
                self.callback_failed = failed;
                self.generation_failed = matches!(&result, Err(ChatGenerationError::Message(_)));
                result
            }
        }

        struct BarrierSession {
            session: ChatSession,
            entered: SyncSender<()>,
            release: Receiver<()>,
        }

        impl ChatBackend for BarrierSession {
            fn load_ms(&self) -> f64 {
                self.session.load_ms()
            }

            fn generate_with_timeout(
                &mut self,
                request: ChatRequest<'_>,
                timeout: Duration,
                on_token: &mut dyn FnMut(&str) -> Result<(), String>,
            ) -> Result<crate::chat_generation::ChatGeneration, ChatGenerationError> {
                let hold_after_delta = request.max_tokens == 64;
                let mut held = false;
                self.session
                    .generate_with_timeout(request, timeout, &mut |delta| {
                        on_token(delta)?;
                        if hold_after_delta && !held {
                            held = true;
                            self.entered
                                .send(())
                                .map_err(|_| String::from("busy test barrier closed"))?;
                            self.release
                                .recv_timeout(Duration::from_secs(60))
                                .map_err(|_| String::from("busy test barrier timed out"))?;
                        }
                        Ok(())
                    })
            }
        }

        struct ReleaseOnDrop(Option<SyncSender<()>>);

        impl Drop for ReleaseOnDrop {
            fn drop(&mut self) {
                if let Some(sender) = self.0.take() {
                    let _ = sender.send(());
                }
            }
        }

        fn body(max_output_tokens: u32) -> Vec<u8> {
            serde_json::to_vec(&json!({
                "model": "metallix-qwen3",
                "input": if max_output_tokens == 64 {
                    "Count from one to one hundred, writing every number in words."
                } else { "Reply with a short recovery acknowledgement." },
                "stream": true,
                "max_output_tokens": max_output_tokens,
                "temperature": 0,
                "store": false,
            }))
            .expect("test request JSON")
        }

        fn request_stream(address: std::net::SocketAddr, body: &[u8]) -> TcpStream {
            let mut stream = TcpStream::connect(address).expect("connect loopback server");
            stream
                .set_read_timeout(Some(Duration::from_secs(60)))
                .expect("bound client reads");
            write!(
                stream,
                "POST /v1/responses HTTP/1.1\r\nHost: localhost\r\nContent-Length: {}\r\n\r\n",
                body.len()
            )
            .expect("write request header");
            stream.write_all(body).expect("write request body");
            stream
                .shutdown(Shutdown::Write)
                .expect("half-close request");
            stream
        }

        fn client(address: std::net::SocketAddr, body: Vec<u8>) -> thread::JoinHandle<Vec<u8>> {
            thread::spawn(move || {
                let mut stream = request_stream(address, &body);
                let mut wire = Vec::new();
                stream
                    .read_to_end(&mut wire)
                    .expect("read complete response");
                wire
            })
        }

        fn fixed_response_client(
            address: std::net::SocketAddr,
            body: Vec<u8>,
        ) -> thread::JoinHandle<(u16, Vec<u8>)> {
            thread::spawn(move || {
                let mut stream = request_stream(address, &body);
                fixed_http_response(&mut stream)
            })
        }

        fn reset_after_delta(
            address: std::net::SocketAddr,
            body: Vec<u8>,
        ) -> thread::JoinHandle<Vec<u8>> {
            thread::spawn(move || {
                const RESET_AFTER_DELTA: &str = r#"
import json
import socket
import struct
import sys

host, port = sys.argv[1], int(sys.argv[2])
body = sys.stdin.buffer.read()
if len(body) > 1_048_576:
    raise RuntimeError("request body is too large")
request = (
    b"POST /v1/responses HTTP/1.1\r\nHost: localhost\r\nContent-Length: "
    + str(len(body)).encode("ascii")
    + b"\r\n\r\n"
    + body
)
stream = socket.create_connection((host, port), timeout=60)
stream.settimeout(60)
stream.sendall(request)
stream.shutdown(socket.SHUT_WR)
raw = bytearray()
while len(raw) <= 65_536:
    chunk = stream.recv(min(1_024, 65_537 - len(raw)))
    if not chunk:
        raise RuntimeError("stream ended before a generated text delta")
    raw.extend(chunk)
    payload = bytes(raw).split(b"\r\n\r\n", 1)
    if len(payload) != 2:
        continue
    for frame in payload[1].split(b"\n\n")[:-1]:
        data = next((line[6:] for line in frame.splitlines() if line.startswith(b"data: ")), None)
        if data is None:
            continue
        event = json.loads(data)
        if event.get("type") == "response.output_text.delta" and isinstance(event.get("delta"), str) and event["delta"]:
            stream.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
            stream.close()
            sys.stdout.buffer.write(raw)
            raise SystemExit(0)
raise RuntimeError("stream exceeded 65536 bytes before a generated text delta")
"#;
                let host = address.ip().to_string();
                let port = address.port().to_string();
                let mut child = Command::new("python3")
                    .args(["-c", RESET_AFTER_DELTA, &host, &port])
                    .stdin(Stdio::piped())
                    .stdout(Stdio::piped())
                    .stderr(Stdio::piped())
                    .spawn()
                    .expect("start bounded reset client");
                std::io::Write::write_all(child.stdin.as_mut().expect("reset client stdin"), &body)
                    .expect("write reset request body");
                let output = child.wait_with_output().expect("wait for reset client");
                assert!(
                    output.status.success(),
                    "reset client failed: {}",
                    String::from_utf8_lossy(&output.stderr)
                );
                output.stdout
            })
        }

        fn accepted_request(listener: &TcpListener) -> (Connection, Request, Vec<ChatMessage>) {
            let (stream, _) = listener.accept().expect("accept loopback request");
            let mut connection = Connection::accept(stream, TransportLimits::default());
            let wire = connection.read_request().expect("read request");
            let request: Request = serde_json::from_slice(&wire.body).expect("parse request");
            let messages = messages(&request).expect("prepare request messages");
            (connection, request, messages)
        }

        fn terminal_text_and_usage(wire: Vec<u8>) -> (String, String, Value) {
            let wire = String::from_utf8(wire).expect("SSE response is UTF-8");
            let (headers, payload) = wire.split_once("\r\n\r\n").expect("SSE headers");
            assert!(headers.starts_with("HTTP/1.1 200 OK\r\n"));
            let events: Vec<Value> = payload
                .split("\n\n")
                .filter(|frame| !frame.is_empty())
                .map(|frame| {
                    let (_, data) = frame.split_once('\n').expect("SSE event data");
                    serde_json::from_str(data.strip_prefix("data: ").expect("SSE data prefix"))
                        .expect("SSE event JSON")
                })
                .collect();
            assert!(
                events.len() >= 6,
                "complete stream has its lifecycle events"
            );
            assert_eq!(
                events
                    .iter()
                    .map(|event| event["sequence_number"].as_u64())
                    .collect::<Vec<_>>(),
                (0..events.len())
                    .map(|index| Some(index as u64))
                    .collect::<Vec<_>>()
            );
            assert_eq!(events[0]["type"], "response.created");
            assert_eq!(events[1]["type"], "response.output_item.added");
            assert_eq!(events[2]["type"], "response.content_part.added");
            let terminal = events.last().expect("terminal event");
            assert!(
                matches!(
                    terminal["type"].as_str(),
                    Some("response.completed" | "response.incomplete")
                ),
                "stream must end completed or capped incomplete"
            );
            let response = terminal["response"].clone();
            let text = response["output"][0]["content"][0]["text"]
                .as_str()
                .expect("terminal assistant text")
                .to_owned();
            let deltas = events
                .iter()
                .filter(|event| event["type"] == "response.output_text.delta")
                .map(|event| event["delta"].as_str().expect("text delta"))
                .collect::<String>();
            assert_eq!(deltas, text, "stream deltas reconstruct terminal text");
            let usage = response["usage"].clone();
            assert!(
                usage["input_tokens"]
                    .as_u64()
                    .is_some_and(|value| value > 0)
            );
            assert!(
                usage["output_tokens"]
                    .as_u64()
                    .is_some_and(|value| value > 0)
            );
            assert_eq!(
                usage["total_tokens"].as_u64(),
                Some(
                    usage["input_tokens"].as_u64().expect("input tokens")
                        + usage["output_tokens"].as_u64().expect("output tokens")
                )
            );
            (
                terminal["type"].as_str().expect("terminal type").to_owned(),
                text,
                usage,
            )
        }

        #[test]
        #[ignore = "requires METALLIX_QWEN_MODEL and a local Apple-Silicon Metal checkpoint"]
        #[allow(
            clippy::too_many_lines,
            reason = "the ordered real-checkpoint lifecycle is the assertion under test"
        )]
        fn checkpoint_busy_rejection_after_actual_delta_then_session_recovers() {
            let model = env::var_os("METALLIX_QWEN_MODEL")
                .expect("explicit checkpoint test requires METALLIX_QWEN_MODEL");
            let listener = TcpListener::bind("127.0.0.1:0").expect("loopback listener");
            let address = listener.local_addr().expect("listener address");
            let (job_sender, job_receiver) = sync_channel(0);
            let (startup_sender, startup_receiver) = sync_channel(1);
            let (entered_sender, entered_receiver) = sync_channel(1);
            let (release_sender, release_receiver) = sync_channel(1);
            let worker = thread::spawn(move || {
                let session = ChatSession::load(
                    std::path::Path::new(&model),
                    ResidentChatLimits::from_mib(2_048, 1_024),
                );
                let session = match session {
                    Ok(session) => session,
                    Err(error) => {
                        let _ = startup_sender.send(Err(error));
                        return;
                    }
                };
                if startup_sender.send(Ok(())).is_ok() {
                    let mut backend = BarrierSession {
                        session,
                        entered: entered_sender,
                        release: release_receiver,
                    };
                    worker_loop(&mut backend, job_receiver);
                }
            });
            match startup_receiver.recv_timeout(Duration::from_secs(60)) {
                Ok(Ok(())) => {}
                Ok(Err(error)) => panic!("checkpoint session load: {error}"),
                Err(error) => panic!("checkpoint worker startup: {error}"),
            }
            let occupied = Arc::new(AtomicBool::new(false));
            let server_occupied = Arc::clone(&occupied);
            let worker_alive = Arc::new(AtomicBool::new(true));
            let server_worker_alive = Arc::clone(&worker_alive);
            let server_sender = job_sender.clone();
            let server = thread::spawn(move || {
                serve_listener(
                    &listener,
                    "metallix-qwen3",
                    &server_sender,
                    &server_occupied,
                    &server_worker_alive,
                    Duration::from_secs(60),
                    Some(4),
                )
            });

            let baseline_client = client(address, body(32));
            let baseline =
                terminal_text_and_usage(baseline_client.join().expect("baseline client"));

            let mut release = ReleaseOnDrop(Some(release_sender));
            let primary_body = body(64);
            let mut primary = TcpStream::connect(address).expect("connect primary request");
            primary
                .set_read_timeout(Some(Duration::from_secs(60)))
                .expect("bound primary reads");
            write!(
                primary,
                "POST /v1/responses HTTP/1.1\r\nHost: localhost\r\nContent-Length: {}\r\n\r\n",
                primary_body.len()
            )
            .expect("primary request header");
            primary
                .write_all(&primary_body)
                .expect("primary request body");
            primary
                .shutdown(Shutdown::Write)
                .expect("primary half-close");
            let mut primary_prefix = Vec::new();
            let mut chunk = [0_u8; 1024];
            while !primary_prefix
                .windows(b"response.output_text.delta".len())
                .any(|window| window == b"response.output_text.delta")
            {
                let read = primary.read(&mut chunk).expect("read primary delta");
                assert!(read > 0, "primary ended before an actual delta");
                primary_prefix.extend_from_slice(&chunk[..read]);
            }
            entered_receiver
                .recv_timeout(Duration::from_secs(60))
                .expect("worker holds after actual callback delta");

            let busy_client = fixed_response_client(address, body(32));
            let (busy_status, busy_body) = busy_client.join().expect("busy client");
            assert_eq!(busy_status, 503);
            assert_eq!(
                serde_json::from_slice::<Value>(&busy_body).expect("busy response JSON"),
                json!({"error":{"code":"server_busy","message":"one generation is already active"}})
            );

            release
                .0
                .take()
                .expect("release sender")
                .send(())
                .expect("release worker");
            primary
                .read_to_end(&mut primary_prefix)
                .expect("read primary completion");
            let (primary_status, _, primary_usage) = terminal_text_and_usage(primary_prefix);
            let generated = primary_usage["output_tokens"]
                .as_u64()
                .expect("output usage");
            assert!(generated <= 64, "the primary request retains its token cap");
            if primary_status == "response.incomplete" {
                assert_eq!(generated, 64, "incomplete must exhaust the declared cap");
            }

            let recovery_client = client(address, body(32));
            let recovery =
                terminal_text_and_usage(recovery_client.join().expect("recovery client"));
            assert_eq!(
                recovery, baseline,
                "recovery matches the uninterrupted baseline"
            );

            server
                .join()
                .expect("join bounded acceptor")
                .expect("acceptor result");
            drop(job_sender);
            worker.join().expect("join bounded worker");
            assert!(!occupied.load(Ordering::Acquire));
        }

        #[test]
        #[ignore = "requires METALLIX_QWEN_MODEL and a local Apple-Silicon Metal checkpoint"]
        fn checkpoint_stream_reset_after_delta_allows_fresh_response() {
            let model = env::var_os("METALLIX_QWEN_MODEL")
                .expect("explicit checkpoint test requires METALLIX_QWEN_MODEL");
            let limits = ResidentChatLimits::from_mib(2_048, 1_024);
            let mut session = ChatSession::load(std::path::Path::new(&model), limits)
                .expect("checkpoint session load");
            let listener = TcpListener::bind("127.0.0.1:0").expect("loopback listener");
            let address = listener.local_addr().expect("listener address");

            let baseline_client = client(address, body(32));
            let (connection, request, messages) = accepted_request(&listener);
            respond(
                connection,
                &request,
                &messages,
                &[],
                &mut session,
                "baseline",
                Duration::from_secs(60),
            )
            .expect("uninterrupted baseline response");
            let baseline =
                terminal_text_and_usage(baseline_client.join().expect("baseline client"));

            let reset_client = reset_after_delta(address, body(64));
            let (connection, request, messages) = accepted_request(&listener);
            let mut observed = ObservedSession {
                session: &mut session,
                callback_failed: false,
                generation_failed: false,
            };
            let reset_error = respond(
                connection,
                &request,
                &messages,
                &[],
                &mut observed,
                "reset",
                Duration::from_secs(60),
            )
            .expect_err("post-delta TCP reset must fail a later stream write");
            assert!(
                !reset_error.is_empty(),
                "reset failure carries transport context"
            );
            assert!(
                observed.callback_failed,
                "reset must reach the generation callback"
            );
            assert!(
                observed.generation_failed,
                "generation must propagate the callback failure"
            );
            let reset_prefix = reset_client.join().expect("reset client");
            assert!(
                reset_prefix
                    .windows(b"response.output_text.delta".len())
                    .any(|window| window == b"response.output_text.delta"),
                "client observed a real generated delta before resetting"
            );

            let recovery_client = client(address, body(32));
            let (connection, request, messages) = accepted_request(&listener);
            respond(
                connection,
                &request,
                &messages,
                &[],
                &mut session,
                "recovery",
                Duration::from_secs(60),
            )
            .expect("fresh response after reset");
            let recovery =
                terminal_text_and_usage(recovery_client.join().expect("recovery client"));
            assert_eq!(
                recovery, baseline,
                "fresh response matches uninterrupted baseline"
            );
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
                context_tokens: 2048,
                planned_kv_bytes: 0,
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
