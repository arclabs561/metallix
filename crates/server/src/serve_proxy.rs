//! `mx serve` front process: one child `mx serve` per registered model, each on
//! its own loopback port, with requests forwarded byte for byte.
//!
//! A child owns its model's memory, so stopping it returns that memory to the
//! system. A child that fails to start or dies makes only its model
//! unavailable; the front process keeps serving the others.

use std::{
    io::{self, BufRead, BufReader, Write},
    net::{Shutdown, SocketAddr, TcpListener, TcpStream},
    process::{Child, Command, ExitCode, Stdio},
    sync::{
        Arc, Mutex,
        atomic::{AtomicUsize, Ordering},
        mpsc::{Sender, channel},
    },
    thread,
    time::Duration,
};

use serde::Deserialize;
use serde_json::{Value, json};

use crate::{
    http_transport::{Connection, Request, TransportLimits},
    responses::json_response,
    serve_registry::ServedEntry,
};

const START_TIMEOUT: Duration = Duration::from_secs(300);
const CONNECT_TIMEOUT: Duration = Duration::from_secs(2);
// Grace beyond the client response deadline for the child's own response.
const CHILD_READ_GRACE: Duration = Duration::from_secs(5);
// ponytail: fixed cap on concurrent forwards; children reject their own overlap.
const MAX_IN_FLIGHT: usize = 64;
const LISTENING: &str = "mx listening on http://";

/// Command-line settings passed through to every child.
#[derive(Clone, Copy, Debug)]
pub(crate) struct ChildSettings {
    pub(crate) context_tokens: u32,
    pub(crate) kv_budget_mib: u32,
    pub(crate) generation_timeout_ms: u32,
}

/// One registered model and the child process serving it.
struct ChildModel {
    entry: ServedEntry,
    address: Option<SocketAddr>,
    process: Mutex<Option<Child>>,
    unavailable: Mutex<Option<String>>,
}

impl ChildModel {
    fn unavailable_reason(&self) -> Option<String> {
        let recorded = self.unavailable.lock().expect("unavailable lock").clone();
        recorded.or_else(|| {
            self.address
                .is_none()
                .then(|| String::from("model did not start"))
        })
    }

    /// Records why the child cannot serve, including its exit status if it exited.
    fn mark_unavailable(&self, cause: &str) -> String {
        let exited = self
            .process
            .lock()
            .expect("process lock")
            .as_mut()
            .and_then(|child| child.try_wait().ok().flatten())
            .map(|status| format!("; child exited with {status}"))
            .unwrap_or_default();
        let reason = format!("{cause}{exited}");
        eprintln!("mx serve: model {} is unavailable: {reason}", self.entry.id);
        *self.unavailable.lock().expect("unavailable lock") = Some(reason.clone());
        reason
    }
}

impl Drop for ChildModel {
    fn drop(&mut self) {
        if let Some(child) = self.process.get_mut().expect("process lock").as_mut() {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

pub(crate) fn serve(
    entries: &[ServedEntry],
    address: SocketAddr,
    settings: ChildSettings,
) -> ExitCode {
    match serve_inner(entries, address, settings) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("mx serve: {error}");
            ExitCode::FAILURE
        }
    }
}

fn serve_inner(
    entries: &[ServedEntry],
    address: SocketAddr,
    settings: ChildSettings,
) -> Result<(), String> {
    if !address.ip().is_loopback() {
        return Err("this experimental server binds only to loopback".into());
    }
    let executable = std::env::current_exe().map_err(|error| error.to_string())?;
    let started: Vec<_> = entries
        .iter()
        .map(|entry| (entry, start_child(&executable, entry, settings)))
        .collect();
    let models: Vec<Arc<ChildModel>> = started
        .into_iter()
        .map(|(entry, start)| Arc::new(wait_for_child(entry, start)))
        .collect();
    let server = TcpListener::bind(address).map_err(|error| error.to_string())?;
    let local = server.local_addr().map_err(|error| error.to_string())?;
    let available = models
        .iter()
        .filter(|model| model.unavailable_reason().is_none())
        .count();
    eprintln!(
        "mx listening on http://{local}; models={}; available={available}; one child process per model",
        models.len()
    );
    proxy_models(&server, &models, TransportLimits::default(), None)
}

type Started = Result<(Child, std::sync::mpsc::Receiver<SocketAddr>), String>;

/// Starts a child on an ephemeral loopback port. Its stdin stays open so the
/// child can exit when this process does.
fn start_child(
    executable: &std::path::Path,
    entry: &ServedEntry,
    settings: ChildSettings,
) -> Started {
    let entry_json = serde_json::to_string(entry).map_err(|error| error.to_string())?;
    let mut child = Command::new(executable)
        .args([
            "serve",
            "--worker-entry",
            &entry_json,
            "--listen",
            "127.0.0.1:0",
        ])
        .args(["--context-tokens", &settings.context_tokens.to_string()])
        .args(["--kv-budget-mib", &settings.kv_budget_mib.to_string()])
        .args([
            "--generation-timeout-ms",
            &settings.generation_timeout_ms.to_string(),
        ])
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|error| format!("could not start child: {error}"))?;
    let stderr = child.stderr.take().ok_or("child stderr is unavailable")?;
    let (sender, receiver) = channel();
    let id = entry.id.clone();
    thread::spawn(move || forward_child_log(&id, stderr, &sender));
    Ok((child, receiver))
}

/// Copies child log lines to this process's stderr, reporting the address the
/// child announces once it is listening.
fn forward_child_log(id: &str, stderr: impl io::Read, address: &Sender<SocketAddr>) {
    for line in BufReader::new(stderr).lines() {
        let Ok(line) = line else { return };
        if let Some(rest) = line.strip_prefix(LISTENING) {
            if let Some(parsed) = rest.split(';').next().and_then(|a| a.parse().ok()) {
                let _ = address.send(parsed);
            }
        }
        eprintln!("[{id}] {line}");
    }
}

fn wait_for_child(entry: &ServedEntry, started: Started) -> ChildModel {
    let model = |address, process, reason: Option<String>| {
        if let Some(reason) = &reason {
            eprintln!("mx serve: model {} is unavailable: {reason}", entry.id);
        }
        ChildModel {
            entry: entry.clone(),
            address,
            process: Mutex::new(process),
            unavailable: Mutex::new(reason),
        }
    };
    let (mut child, receiver) = match started {
        Ok(started) => started,
        Err(error) => return model(None, None, Some(error)),
    };
    // The log reader drops its sender when the child exits, so this returns early.
    if let Ok(address) = receiver.recv_timeout(START_TIMEOUT) {
        return model(Some(address), Some(child), None);
    }
    let reason = if let Ok(Some(status)) = child.try_wait() {
        format!("child exited during startup with {status}")
    } else {
        let _ = child.kill();
        format!("child did not listen within {} s", START_TIMEOUT.as_secs())
    };
    let _ = child.wait();
    model(None, None, Some(reason))
}

#[derive(Deserialize)]
struct Target {
    model: String,
}

/// Routes by the request's `model` and forwards it unchanged to that child.
fn proxy_models(
    server: &TcpListener,
    models: &[Arc<ChildModel>],
    limits: TransportLimits,
    request_limit: Option<usize>,
) -> Result<(), String> {
    let in_flight = Arc::new(AtomicUsize::new(0));
    let mut forwards = Vec::new();
    for socket in server.incoming().take(request_limit.unwrap_or(usize::MAX)) {
        let socket = socket.map_err(|error| error.to_string())?;
        let mut connection = Connection::accept(socket, limits);
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
                let data: Vec<Value> = models
                    .iter()
                    .map(|model| {
                        json!({"id":model.entry.id,"object":"model","owned_by":"local","capabilities":model.entry.kind.capabilities(),"loaded":model.unavailable_reason().is_none()})
                    })
                    .collect();
                json_response(connection, 200, &json!({"object":"list","data":data}));
                continue;
            }
            ("POST", "/v1/responses" | "/v1/decisions") => {}
            _ => {
                json_response(
                    connection,
                    404,
                    &json!({"error":{"message":"unknown endpoint"}}),
                );
                continue;
            }
        }
        let target = match serde_json::from_slice::<Target>(&request.body) {
            Ok(target) => target.model,
            Err(error) => {
                json_response(
                    connection,
                    400,
                    &json!({"error":{"message":error.to_string()}}),
                );
                continue;
            }
        };
        let Some(model) = models.iter().find(|model| model.entry.id == target) else {
            json_response(
                connection,
                404,
                &json!({"error":{"message":"model is not loaded"}}),
            );
            continue;
        };
        if in_flight.fetch_add(1, Ordering::AcqRel) >= MAX_IN_FLIGHT {
            in_flight.fetch_sub(1, Ordering::AcqRel);
            json_response(
                connection,
                503,
                &json!({"error":{"code":"server_busy","message":"too many requests in flight"}}),
            );
            continue;
        }
        let model = Arc::clone(model);
        let in_flight = Arc::clone(&in_flight);
        forwards.retain(|forward: &thread::JoinHandle<()>| !forward.is_finished());
        forwards.push(thread::spawn(move || {
            forward(&model, connection, &request, limits);
            in_flight.fetch_sub(1, Ordering::AcqRel);
        }));
    }
    for forward in forwards {
        let _ = forward.join();
    }
    Ok(())
}

fn forward(
    model: &ChildModel,
    mut connection: Connection,
    request: &Request,
    limits: TransportLimits,
) {
    let unavailable = |connection, reason: String| {
        json_response(
            connection,
            503,
            &json!({"error":{"code":"model_worker_unavailable","message":format!("model {} is unavailable: {reason}", model.entry.id)}}),
        );
    };
    if let Some(reason) = model.unavailable_reason() {
        return unavailable(connection, reason);
    }
    let address = model.address.expect("available models have an address");
    let mut child = match TcpStream::connect_timeout(&address, CONNECT_TIMEOUT) {
        Ok(child) => child,
        Err(error) => {
            let reason = model.mark_unavailable(&format!("child connection failed: {error}"));
            return unavailable(connection, reason);
        }
    };
    let sent = child
        .set_read_timeout(Some(limits.response_deadline + CHILD_READ_GRACE))
        .and_then(|()| {
            write!(
                child,
                "{} {} HTTP/1.1\r\nHost: {address}\r\nContent-Length: {}\r\n\r\n",
                request.method,
                request.path,
                request.body.len()
            )
        })
        .and_then(|()| child.write_all(&request.body))
        .and_then(|()| child.shutdown(Shutdown::Write));
    if let Err(error) = sent {
        let reason = model.mark_unavailable(&format!("child request failed: {error}"));
        return unavailable(connection, reason);
    }
    // The child writes one complete HTTP response and closes; pass it through.
    connection.begin_response();
    if let Err(error) = io::copy(&mut child, &mut connection).and_then(|_| connection.flush()) {
        eprintln!(
            "mx serve: forwarding a {} response failed: {error}",
            model.entry.id
        );
    }
}

#[cfg(test)]
mod tests {
    use std::io::Read as _;

    use super::*;
    use crate::serve_registry::ModelKind;

    fn exchange(address: SocketAddr, request: &str, body: &[u8]) -> (u16, Vec<u8>) {
        let mut stream = TcpStream::connect(address).expect("connect");
        stream
            .set_read_timeout(Some(Duration::from_secs(5)))
            .expect("bound reads");
        write!(
            stream,
            "{request} HTTP/1.1\r\nHost: localhost\r\nContent-Length: {}\r\n\r\n",
            body.len()
        )
        .expect("request header");
        stream.write_all(body).expect("request body");
        stream.shutdown(Shutdown::Write).expect("half-close");
        let mut wire = Vec::new();
        stream.read_to_end(&mut wire).expect("response");
        let status = std::str::from_utf8(&wire[9..12]).unwrap().parse().unwrap();
        let split = wire.windows(4).position(|w| w == b"\r\n\r\n").unwrap() + 4;
        (status, wire[split..].to_vec())
    }

    fn json(body: &[u8]) -> Value {
        serde_json::from_slice(body).expect("JSON body")
    }

    /// A stand-in child: answers every request with an event stream echoing
    /// the request line and body, so forwarding can be checked byte for byte.
    fn echo_child(requests: usize) -> (SocketAddr, thread::JoinHandle<()>) {
        let listener = TcpListener::bind("127.0.0.1:0").expect("child listener");
        let address = listener.local_addr().expect("child address");
        let child = thread::spawn(move || {
            for socket in listener.incoming().take(requests) {
                let mut connection =
                    Connection::accept(socket.unwrap(), TransportLimits::default());
                let request = connection.read_request().expect("forwarded request");
                connection.begin_response();
                let body = format!(
                    "event: echo\ndata: {} {} {}\n\n",
                    request.method,
                    request.path,
                    String::from_utf8(request.body).unwrap()
                );
                write!(
                    connection,
                    "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n{body}"
                )
                .unwrap();
            }
        });
        (address, child)
    }

    fn model(id: &str, kind: ModelKind, address: Option<SocketAddr>) -> Arc<ChildModel> {
        Arc::new(ChildModel {
            entry: ServedEntry {
                id: id.into(),
                kind,
                path: "/unused".into(),
            },
            address,
            process: Mutex::new(None),
            unavailable: Mutex::new(None),
        })
    }

    #[test]
    fn forwards_by_model_and_isolates_an_unavailable_child() {
        let (echo, echo_child) = echo_child(2);
        // A port with nothing listening stands in for a child that died.
        let dead = TcpListener::bind("127.0.0.1:0")
            .unwrap()
            .local_addr()
            .unwrap();
        let models = vec![
            model("qwen", ModelKind::Qwen, Some(echo)),
            model("julia", ModelKind::Julia, Some(dead)),
            model("broken", ModelKind::Julia, None),
        ];
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let server = thread::spawn(move || {
            proxy_models(&listener, &models, TransportLimits::default(), Some(9))
        });

        let listed = json(&exchange(address, "GET /v1/models", b"").1);
        let loaded = |listed: &Value| -> Vec<(String, bool)> {
            listed["data"]
                .as_array()
                .unwrap()
                .iter()
                .map(|m| {
                    (
                        m["id"].as_str().unwrap().to_owned(),
                        m["loaded"].as_bool().unwrap(),
                    )
                })
                .collect()
        };
        assert_eq!(
            loaded(&listed),
            [
                ("qwen".into(), true),
                ("julia".into(), true),
                ("broken".into(), false)
            ]
        );
        assert_eq!(
            listed["data"][0]["capabilities"],
            json!(["generate", "decide"])
        );

        // Bytes reach the child unchanged and its response comes back unchanged.
        let body = br#"{"model":"qwen","input":"hi","stream":true}"#;
        let (status, wire) = exchange(address, "POST /v1/responses", body);
        assert_eq!(status, 200);
        assert_eq!(
            wire,
            format!(
                "event: echo\ndata: POST /v1/responses {}\n\n",
                String::from_utf8_lossy(body)
            )
            .into_bytes()
        );

        // A failed child connection makes only that model unavailable.
        let (status, error) = exchange(address, "POST /v1/decisions", br#"{"model":"julia"}"#);
        assert_eq!(status, 503);
        assert_eq!(json(&error)["error"]["code"], "model_worker_unavailable");
        let (status, error) = exchange(address, "POST /v1/decisions", br#"{"model":"broken"}"#);
        assert_eq!(
            (status, json(&error)["error"]["code"].clone()),
            (503, json!("model_worker_unavailable"))
        );
        assert_eq!(
            loaded(&json(&exchange(address, "GET /v1/models", b"").1)),
            [
                ("qwen".into(), true),
                ("julia".into(), false),
                ("broken".into(), false)
            ]
        );
        let (status, _) = exchange(
            address,
            "POST /v1/decisions",
            br#"{"model":"qwen","state":"s"}"#,
        );
        assert_eq!(status, 200, "the healthy child still serves");

        let (status, _) = exchange(address, "POST /v1/decisions", br#"{"model":"missing"}"#);
        assert_eq!(status, 404);
        let (status, _) = exchange(address, "POST /v1/decisions", b"{");
        assert_eq!(status, 400);
        let (status, _) = exchange(address, "POST /v1/other", b"{}");
        assert_eq!(status, 404);

        server.join().unwrap().unwrap();
        echo_child.join().unwrap();
    }
}
