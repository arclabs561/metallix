//! `mx serve` front process: one child `mx serve` per registered model, each on
//! its own loopback port, with requests forwarded byte for byte.
//!
//! A child owns its model's memory, so stopping it returns that memory to the
//! system. Resident models start with the server; on-demand models start on
//! their first request, and idle on-demand children are stopped, least
//! recently used first, to keep declared `memory_mib` within the budget. A
//! child that fails to start or dies makes only its model unavailable until
//! a later request starts it again; the front process keeps serving.
//!
//! A model serves one request at a time; overlapping requests wait for it in
//! that model's FIFO queue (see [`crate::admission_queue`]).

use std::{
    io::{self, BufRead, BufReader, Read, Write},
    net::{Shutdown, SocketAddr, TcpListener, TcpStream},
    path::PathBuf,
    process::{Child, Command, ExitCode, Stdio},
    sync::{
        Arc, Mutex,
        atomic::{AtomicUsize, Ordering},
        mpsc::{Sender, channel},
    },
    thread,
    time::{Duration, Instant},
};

use serde::Deserialize;
use serde_json::{Value, json};

use crate::{
    admission_queue::{ModelQueue, QueueSettings, Refusal, Refused},
    http_transport::{Connection, Request, TransportLimits},
    responses::json_response,
    serve_registry::{Residency, ServedEntry},
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

enum ChildState {
    Stopped,
    /// `child` is `None` only for test stand-ins that have no process.
    Running {
        child: Option<Child>,
        address: SocketAddr,
    },
}

/// One registered model and the child process serving it.
struct ChildModel {
    entry: ServedEntry,
    state: Mutex<ChildState>,
    last_error: Mutex<Option<String>>,
    /// Requests that have chosen this model and not yet finished; such a
    /// child is never stopped to make room.
    in_flight: AtomicUsize,
    last_used: Mutex<Instant>,
    /// Requests waiting for this model; counted in `in_flight` while queued.
    queue: ModelQueue,
}

impl ChildModel {
    fn new(entry: ServedEntry, state: ChildState, queue: QueueSettings) -> Self {
        Self {
            queue: ModelQueue::new(entry.queue(queue)),
            entry,
            state: Mutex::new(state),
            last_error: Mutex::new(None),
            in_flight: AtomicUsize::new(0),
            last_used: Mutex::new(Instant::now()),
        }
    }

    fn address(&self) -> Option<SocketAddr> {
        match &*self.state.lock().expect("state lock") {
            ChildState::Running { address, .. } => Some(*address),
            ChildState::Stopped => None,
        }
    }

    fn record_error(&self, reason: String) -> String {
        eprintln!("mx serve: model {} is unavailable: {reason}", self.entry.id);
        *self.last_error.lock().expect("error lock") = Some(reason.clone());
        reason
    }

    /// Stops a child that failed a request, recording its exit status if it exited.
    fn fail(&self, cause: &str) -> String {
        let mut state = self.state.lock().expect("state lock");
        let exited = match &mut *state {
            ChildState::Running {
                child: Some(child), ..
            } => {
                let status = child.try_wait().ok().flatten();
                let _ = child.kill();
                let _ = child.wait();
                status.map(|status| format!("; child exited with {status}"))
            }
            _ => None,
        };
        *state = ChildState::Stopped;
        drop(state);
        self.record_error(format!("{cause}{}", exited.unwrap_or_default()))
    }

    /// Stops this child to free memory if no request is using it.
    fn stop_if_idle(&self) -> bool {
        let mut state = self.state.lock().expect("state lock");
        if self.in_flight.load(Ordering::Acquire) != 0 {
            return false;
        }
        if let ChildState::Running {
            child: Some(child), ..
        } = &mut *state
        {
            let _ = child.kill();
            let _ = child.wait();
        }
        *state = ChildState::Stopped;
        true
    }
}

impl Drop for ChildModel {
    fn drop(&mut self) {
        if let ChildState::Running {
            child: Some(child), ..
        } = self.state.get_mut().expect("state lock")
        {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

/// How the front process starts children. Tests run without one.
struct Launcher {
    executable: PathBuf,
    settings: ChildSettings,
}

struct Pool {
    models: Vec<Arc<ChildModel>>,
    launcher: Option<Launcher>,
    budget_mib: Option<u64>,
    /// Serializes starting children and choosing which to stop.
    residency: Mutex<()>,
}

/// What victim selection needs to know about one model.
#[derive(Clone, Copy, Debug)]
struct Slot {
    running: bool,
    evictable: bool,
    memory_mib: u64,
    last_used: Instant,
}

/// Least recently used idle on-demand children to stop so `target` fits.
fn victims(slots: &[Slot], target: usize, budget_mib: u64) -> Result<Vec<usize>, String> {
    let mut used: u64 = slots
        .iter()
        .enumerate()
        .filter(|(index, slot)| *index != target && slot.running)
        .map(|(_, slot)| slot.memory_mib)
        .sum();
    let need = slots[target].memory_mib;
    let mut candidates: Vec<usize> = (0..slots.len())
        .filter(|&index| index != target && slots[index].running && slots[index].evictable)
        .collect();
    candidates.sort_by_key(|&index| slots[index].last_used);
    let mut chosen = Vec::new();
    let mut candidates = candidates.into_iter();
    while used + need > budget_mib {
        let Some(index) = candidates.next() else {
            return Err(format!(
                "the {budget_mib} MiB memory budget is held by resident or busy models ({used} MiB in use, {need} MiB needed)"
            ));
        };
        used -= slots[index].memory_mib;
        chosen.push(index);
    }
    Ok(chosen)
}

impl Pool {
    /// The address of `model`'s running child, starting it first if needed.
    /// The caller has already counted the request in `model.in_flight`.
    fn acquire(&self, model: &ChildModel) -> Result<SocketAddr, String> {
        *model.last_used.lock().expect("last-used lock") = Instant::now();
        if let Some(address) = model.address() {
            return Ok(address);
        }
        let _residency = self.residency.lock().expect("residency lock");
        if let Some(address) = model.address() {
            return Ok(address);
        }
        self.make_room(model)?;
        let launcher = self
            .launcher
            .as_ref()
            .ok_or("this server cannot start children")?;
        let (child, address) =
            start_and_wait(launcher, &model.entry).map_err(|reason| model.record_error(reason))?;
        *model.state.lock().expect("state lock") = ChildState::Running {
            child: Some(child),
            address,
        };
        *model.last_error.lock().expect("error lock") = None;
        Ok(address)
    }

    fn make_room(&self, target: &ChildModel) -> Result<(), String> {
        let Some(budget) = self.budget_mib else {
            return Ok(());
        };
        let slots: Vec<Slot> = self
            .models
            .iter()
            .map(|model| Slot {
                running: model.address().is_some(),
                evictable: model.entry.residency == Residency::OnDemand
                    && model.in_flight.load(Ordering::Acquire) == 0,
                memory_mib: model.entry.memory_mib.unwrap_or(0),
                last_used: *model.last_used.lock().expect("last-used lock"),
            })
            .collect();
        let target_index = self
            .models
            .iter()
            .position(|model| std::ptr::eq(model.as_ref(), target))
            .expect("target is registered");
        for index in victims(&slots, target_index, budget)? {
            let victim = &self.models[index];
            if !victim.stop_if_idle() {
                return Err(format!("{} became busy while making room", victim.entry.id));
            }
            eprintln!(
                "mx serve: stopped idle model {} to free {} MiB for {}",
                victim.entry.id,
                victim.entry.memory_mib.unwrap_or(0),
                target.entry.id
            );
        }
        Ok(())
    }
}

pub(crate) fn serve(
    entries: &[ServedEntry],
    address: SocketAddr,
    settings: ChildSettings,
    budget_mib: Option<u64>,
    queue: QueueSettings,
) -> ExitCode {
    match serve_inner(entries, address, settings, budget_mib, queue) {
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
    budget_mib: Option<u64>,
    queue: QueueSettings,
) -> Result<(), String> {
    if !address.ip().is_loopback() {
        return Err("this experimental server binds only to loopback".into());
    }
    crate::serve_registry::check_budget(entries, budget_mib)?;
    let launcher = Launcher {
        executable: std::env::current_exe().map_err(|error| error.to_string())?,
        settings,
    };
    // Resident children start together; on-demand ones wait for a request.
    let started: Vec<_> = entries
        .iter()
        .map(|entry| {
            (entry.residency == Residency::Resident).then(|| start_child(&launcher, entry))
        })
        .collect();
    let models: Vec<Arc<ChildModel>> = entries
        .iter()
        .zip(started)
        .map(|(entry, started)| {
            let model = ChildModel::new(entry.clone(), ChildState::Stopped, queue);
            if let Some(started) = started {
                match wait_for_child(started) {
                    Ok((child, address)) => {
                        *model.state.lock().expect("state lock") = ChildState::Running {
                            child: Some(child),
                            address,
                        };
                    }
                    Err(reason) => {
                        model.record_error(reason);
                    }
                }
            }
            Arc::new(model)
        })
        .collect();
    let server = TcpListener::bind(address).map_err(|error| error.to_string())?;
    let local = server.local_addr().map_err(|error| error.to_string())?;
    let running = models
        .iter()
        .filter(|model| model.address().is_some())
        .count();
    eprintln!(
        "mx listening on http://{local}; models={}; running={running}; one child process per model",
        models.len()
    );
    let pool = Arc::new(Pool {
        models,
        launcher: Some(launcher),
        budget_mib,
        residency: Mutex::new(()),
    });
    proxy_models(&server, &pool, TransportLimits::default(), None)
}

type Started = Result<(Child, std::sync::mpsc::Receiver<SocketAddr>), String>;

/// Starts a child on an ephemeral loopback port. Its stdin stays open so the
/// child can exit when this process does.
fn start_child(launcher: &Launcher, entry: &ServedEntry) -> Started {
    let settings = launcher.settings;
    let entry_json = serde_json::to_string(entry).map_err(|error| error.to_string())?;
    let mut child = Command::new(&launcher.executable)
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

fn wait_for_child(started: Started) -> Result<(Child, SocketAddr), String> {
    let (mut child, receiver) = started?;
    // The log reader drops its sender when the child exits, so this returns early.
    if let Ok(address) = receiver.recv_timeout(START_TIMEOUT) {
        return Ok((child, address));
    }
    let reason = if let Ok(Some(status)) = child.try_wait() {
        format!("child exited during startup with {status}")
    } else {
        let _ = child.kill();
        format!("child did not listen within {} s", START_TIMEOUT.as_secs())
    };
    let _ = child.wait();
    Err(reason)
}

fn start_and_wait(launcher: &Launcher, entry: &ServedEntry) -> Result<(Child, SocketAddr), String> {
    eprintln!("mx serve: starting model {}", entry.id);
    wait_for_child(start_child(launcher, entry))
}

#[derive(Deserialize)]
struct Target {
    model: String,
}

/// Routes by the request's `model` and forwards it unchanged to that child.
fn proxy_models(
    server: &TcpListener,
    pool: &Arc<Pool>,
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
                let data: Vec<Value> = pool
                    .models
                    .iter()
                    .map(|model| {
                        json!({"id":model.entry.id,"object":"model","owned_by":"local","capabilities":model.entry.kind.capabilities(),"residency":model.entry.residency,"loaded":model.address().is_some(),"queued":model.queue.waiting()})
                    })
                    .collect();
                json_response(connection, 200, &json!({"object":"list","data":data}));
                continue;
            }
            ("POST", "/v1/responses" | "/v1/decisions" | "/v1/embeddings" | "/v1/rerank") => {}
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
        let Some(model) = pool.models.iter().find(|model| model.entry.id == target) else {
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
        // Counted before the forward thread runs, so the child cannot be stopped under it.
        model.in_flight.fetch_add(1, Ordering::AcqRel);
        let model = Arc::clone(model);
        let pool = Arc::clone(pool);
        let in_flight = Arc::clone(&in_flight);
        forwards.retain(|forward: &thread::JoinHandle<()>| !forward.is_finished());
        forwards.push(thread::spawn(move || {
            forward(&pool, &model, connection, &request, limits);
            model.in_flight.fetch_sub(1, Ordering::AcqRel);
            in_flight.fetch_sub(1, Ordering::AcqRel);
        }));
    }
    for forward in forwards {
        let _ = forward.join();
    }
    Ok(())
}

fn forward(
    pool: &Pool,
    model: &ChildModel,
    mut connection: Connection,
    request: &Request,
    limits: TransportLimits,
) {
    let admitted = match model.queue.admit(|| connection.client_gone()) {
        Ok(admitted) => admitted,
        Err(refused) => return refuse(connection, model, refused),
    };
    // Every forwarded response says how long it queued.
    let queue_headers = format!(
        "X-Metallix-Queue-Depth: {}\r\nX-Metallix-Queue-Wait-Ms: {}\r\n",
        admitted.depth,
        admitted.waited.as_millis()
    );
    let unavailable = |connection, reason: String| {
        json_response(
            connection,
            503,
            &json!({"error":{"code":"model_worker_unavailable","message":format!("model {} is unavailable: {reason}", model.entry.id)}}),
        );
    };
    let address = match pool.acquire(model) {
        Ok(address) => address,
        Err(reason) => return unavailable(connection, reason),
    };
    let mut child = match TcpStream::connect_timeout(&address, CONNECT_TIMEOUT) {
        Ok(child) => child,
        Err(error) => {
            let reason = model.fail(&format!("child connection failed: {error}"));
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
        let reason = model.fail(&format!("child request failed: {error}"));
        return unavailable(connection, reason);
    }
    // The child writes one complete HTTP response and closes; pass it through.
    connection.begin_response();
    let passed =
        pass_response(&mut child, &mut connection, &queue_headers).and_then(|_| connection.flush());
    // The model is free once its child has finished this response.
    drop(admitted);
    if let Err(error) = passed {
        eprintln!(
            "mx serve: forwarding a {} response failed: {error}",
            model.entry.id
        );
    }
}

/// Copies the child's response to the client, adding `headers` (complete
/// header lines) to its head. A head that does not end within 16 KiB passes
/// through unchanged.
fn pass_response(child: &mut impl Read, client: &mut impl Write, headers: &str) -> io::Result<u64> {
    let mut head = Vec::new();
    let mut chunk = [0_u8; 4096];
    let end = loop {
        if let Some(end) = head.windows(4).position(|window| window == b"\r\n\r\n") {
            break Some(end + 2);
        }
        let read = child.read(&mut chunk)?;
        if read == 0 || head.len() > 16 * 1024 {
            head.extend_from_slice(&chunk[..read]);
            break None;
        }
        head.extend_from_slice(&chunk[..read]);
    };
    let mut written = head.len() as u64;
    if let Some(end) = end {
        client.write_all(&head[..end])?;
        client.write_all(headers.as_bytes())?;
        client.write_all(&head[end..])?;
        written += headers.len() as u64;
    } else {
        client.write_all(&head)?;
    }
    Ok(written + io::copy(child, client)?)
}

/// Answers a request the model's queue did not admit. 503 rather than 429:
/// the limit is this server's capacity, not a per-client rate, and the `openai`
/// SDKs retry both alike, honoring `Retry-After`.
fn refuse(mut connection: Connection, model: &ChildModel, refused: Refused) {
    let id = &model.entry.id;
    let (code, message) = match refused.refusal {
        Refusal::Full => (
            "server_busy",
            format!("model {id} is busy and its request queue is full"),
        ),
        Refusal::Expired => (
            "queue_timeout",
            format!(
                "model {id} stayed busy for the {} ms queue wait",
                refused.waited.as_millis()
            ),
        ),
        // Nobody is left to answer.
        Refusal::Gone => return,
    };
    let body = json!({"error":{"code":code,"message":message}}).to_string();
    connection.begin_response();
    let _ = write!(
        connection,
        "HTTP/1.1 503 Service Unavailable\r\nContent-Type: application/json\r\nContent-Length: {}\r\nRetry-After: {}\r\nConnection: close\r\n\r\n{body}",
        body.len(),
        refused.retry_after_secs,
    )
    .and_then(|()| connection.flush());
}

#[cfg(test)]
mod tests {
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
        queued_model(id, kind, address, QueueSettings::default())
    }

    fn queued_model(
        id: &str,
        kind: ModelKind,
        address: Option<SocketAddr>,
        queue: QueueSettings,
    ) -> Arc<ChildModel> {
        let entry = ServedEntry {
            id: id.into(),
            kind,
            path: "/unused".into(),
            residency: Residency::Resident,
            memory_mib: None,
            queue_depth: None,
            queue_wait_ms: None,
        };
        let state = address.map_or(ChildState::Stopped, |address| ChildState::Running {
            child: None,
            address,
        });
        Arc::new(ChildModel::new(entry, state, queue))
    }

    /// A stand-in child that serves one request at a time and answers each
    /// only when the test releases it. It reports each request's `input` as it
    /// arrives, and whether the front process opened another connection to it
    /// while one was in progress.
    struct GatedChild {
        address: SocketAddr,
        seen: std::sync::mpsc::Receiver<String>,
        release: Sender<()>,
        overlapped: thread::JoinHandle<bool>,
    }

    fn gated_child(requests: usize) -> GatedChild {
        let listener = TcpListener::bind("127.0.0.1:0").expect("child listener");
        let address = listener.local_addr().expect("child address");
        let (seen_sender, seen) = channel();
        let (release, released) = channel::<()>();
        let overlapped = thread::spawn(move || {
            let mut overlapped = false;
            for _ in 0..requests {
                listener.set_nonblocking(false).unwrap();
                let (socket, _) = listener.accept().unwrap();
                socket.set_nonblocking(false).unwrap();
                let mut connection = Connection::accept(socket, TransportLimits::default());
                let request = connection.read_request().expect("forwarded request");
                let input = json(&request.body)["input"].as_str().unwrap().to_owned();
                seen_sender.send(input.clone()).unwrap();
                released.recv().unwrap();
                listener.set_nonblocking(true).unwrap();
                overlapped |= listener.accept().is_ok();
                connection.begin_response();
                let body = json!({"input": input}).to_string();
                write!(
                    connection,
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                    body.len()
                )
                .unwrap();
            }
            overlapped
        });
        GatedChild {
            address,
            seen,
            release,
            overlapped,
        }
    }

    /// Sends a request and leaves the connection open, as most clients do.
    fn send(address: SocketAddr, input: &str) -> TcpStream {
        let mut stream = TcpStream::connect(address).expect("connect");
        stream
            .set_read_timeout(Some(Duration::from_secs(10)))
            .expect("bound reads");
        let body = json!({"model": "m", "input": input}).to_string();
        write!(
            stream,
            "POST /v1/responses HTTP/1.1\r\nHost: localhost\r\nContent-Length: {}\r\n\r\n{body}",
            body.len()
        )
        .expect("request");
        stream
    }

    /// The final response's status, head and body, after any 1xx responses.
    fn finish(mut stream: TcpStream) -> (u16, String, Value) {
        let mut wire = Vec::new();
        stream.read_to_end(&mut wire).expect("response");
        let mut wire = String::from_utf8(wire).unwrap();
        while wire.starts_with("HTTP/1.1 1") {
            let end = wire.find("\r\n\r\n").unwrap() + 4;
            wire.drain(..end);
        }
        let split = wire.find("\r\n\r\n").unwrap();
        (
            wire[9..12].parse().unwrap(),
            wire[..split].to_owned(),
            json(&wire.as_bytes()[split + 4..]),
        )
    }

    fn header<'a>(head: &'a str, name: &str) -> Option<&'a str> {
        head.lines().find_map(|line| {
            let (key, value) = line.split_once(": ")?;
            key.eq_ignore_ascii_case(name).then_some(value)
        })
    }

    fn until_waiting(model: &ChildModel, count: usize) {
        let deadline = Instant::now() + Duration::from_secs(5);
        while model.queue.waiting() != count {
            assert!(Instant::now() < deadline, "queue never reached {count}");
            thread::sleep(Duration::from_millis(2));
        }
    }

    fn queued_pool(child: SocketAddr, queue: QueueSettings) -> Arc<Pool> {
        Arc::new(Pool {
            models: vec![queued_model("m", ModelKind::Qwen, Some(child), queue)],
            launcher: None,
            budget_mib: None,
            residency: Mutex::new(()),
        })
    }

    fn proxy(pool: &Arc<Pool>, requests: usize) -> (SocketAddr, thread::JoinHandle<()>) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let pool = Arc::clone(pool);
        let server = thread::spawn(move || {
            proxy_models(&listener, &pool, TransportLimits::default(), Some(requests)).unwrap();
        });
        (address, server)
    }

    #[test]
    fn overlapping_requests_wait_and_reach_the_model_one_at_a_time_in_order() {
        let child = gated_child(3);
        let pool = queued_pool(child.address, QueueSettings::default());
        let (address, server) = proxy(&pool, 3);
        let model = &pool.models[0];

        let first = send(address, "a");
        assert_eq!(child.seen.recv().unwrap(), "a");
        let second = send(address, "b");
        until_waiting(model, 1);
        // A client that half-closes after sending still waits its turn.
        let third = send(address, "c");
        third.shutdown(Shutdown::Write).expect("half-close");
        until_waiting(model, 2);
        // Waiters count as activity, so the idle stopper leaves the model running.
        assert!(!model.stop_if_idle());
        assert!(model.address().is_some());

        child.release.send(()).unwrap();
        assert_eq!(child.seen.recv().unwrap(), "b");
        child.release.send(()).unwrap();
        assert_eq!(child.seen.recv().unwrap(), "c");
        child.release.send(()).unwrap();

        for (stream, input, depth) in [(first, "a", "0"), (second, "b", "1"), (third, "c", "2")] {
            let (status, head, body) = finish(stream);
            assert_eq!((status, &body["input"]), (200, &json!(input)));
            assert_eq!(
                header(&head, "X-Metallix-Queue-Depth"),
                Some(depth),
                "{input}"
            );
            let waited: u64 = header(&head, "X-Metallix-Queue-Wait-Ms")
                .unwrap()
                .parse()
                .unwrap();
            assert_eq!(waited == 0, depth == "0", "{input} waited {waited} ms");
        }
        server.join().unwrap();
        assert!(
            !child.overlapped.join().unwrap(),
            "the child saw overlapping requests"
        );
        assert_eq!(model.in_flight.load(Ordering::Acquire), 0);
        assert!(model.stop_if_idle(), "an idle model can be stopped again");
    }

    #[test]
    fn a_full_queue_and_an_expired_wait_answer_503_with_retry_after() {
        let child = gated_child(1);
        let pool = queued_pool(
            child.address,
            QueueSettings {
                depth: 1,
                wait: Duration::from_millis(400),
            },
        );
        let (address, server) = proxy(&pool, 3);
        let model = &pool.models[0];

        let first = send(address, "a");
        assert_eq!(child.seen.recv().unwrap(), "a");
        let queued = send(address, "b");
        until_waiting(model, 1);
        let started = Instant::now();
        let (status, head, body) = finish(send(address, "c"));
        assert!(
            started.elapsed() < Duration::from_millis(400),
            "full refuses at once"
        );
        assert_eq!(
            (status, &body["error"]["code"]),
            (503, &json!("server_busy"))
        );
        assert_eq!(header(&head, "Retry-After"), Some("1"));

        let (status, head, body) = finish(queued);
        assert!(started.elapsed() >= Duration::from_millis(300));
        assert_eq!(
            (status, &body["error"]["code"]),
            (503, &json!("queue_timeout"))
        );
        assert_eq!(header(&head, "Retry-After"), Some("1"));
        assert_eq!(model.queue.waiting(), 0);

        child.release.send(()).unwrap();
        assert_eq!(finish(first).0, 200);
        server.join().unwrap();
        assert!(!child.overlapped.join().unwrap());
    }

    #[test]
    fn a_client_that_leaves_while_queued_is_never_forwarded() {
        let child = gated_child(2);
        let pool = queued_pool(child.address, QueueSettings::default());
        let (address, server) = proxy(&pool, 3);
        let model = &pool.models[0];

        let first = send(address, "a");
        assert_eq!(child.seen.recv().unwrap(), "a");
        let leaving = send(address, "b");
        until_waiting(model, 1);
        let staying = send(address, "c");
        until_waiting(model, 2);
        drop(leaving);
        until_waiting(model, 1);

        child.release.send(()).unwrap();
        assert_eq!(
            child.seen.recv().unwrap(),
            "c",
            "the departed request was skipped"
        );
        child.release.send(()).unwrap();
        assert_eq!(finish(first).0, 200);
        assert_eq!(finish(staying).0, 200);
        server.join().unwrap();
        assert!(!child.overlapped.join().unwrap());
    }

    #[test]
    fn queue_headers_join_the_response_head_and_the_body_is_unchanged() {
        let response = b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nhi";
        let mut client = Vec::new();
        let bytes = pass_response(&mut &response[..], &mut client, "X-A: 1\r\n").unwrap();
        assert_eq!(
            client,
            b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\nX-A: 1\r\n\r\nhi"
        );
        assert_eq!(bytes, client.len() as u64);
        // Without a complete head the bytes pass through untouched.
        let mut client = Vec::new();
        pass_response(&mut &b"garbage"[..], &mut client, "X-A: 1\r\n").unwrap();
        assert_eq!(client, b"garbage");
    }

    #[test]
    fn stops_least_recently_used_idle_on_demand_children_to_fit() {
        let now = Instant::now();
        let slot = |running, evictable, memory_mib, age_ms| Slot {
            running,
            evictable,
            memory_mib,
            last_used: now.checked_sub(Duration::from_millis(age_ms)).unwrap(),
        };
        // 0: resident 2000; 1: idle on-demand 1000 used 5 ms ago; 2: idle
        // on-demand 1500 used 9 ms ago; 3: busy on-demand 500; 4: target 1500.
        let slots = [
            slot(true, false, 2000, 1),
            slot(true, true, 1000, 5),
            slot(true, true, 1500, 9),
            slot(true, false, 500, 1),
            slot(false, true, 1500, 0),
        ];
        // 5000 running + 1500 needed against 6000: stopping the oldest (2) suffices.
        assert_eq!(victims(&slots, 4, 6000), Ok(vec![2]));
        // Against 4500 both idle on-demand children go, oldest first.
        assert_eq!(victims(&slots, 4, 4500), Ok(vec![2, 1]));
        // Resident and busy children are never chosen, so 3999 cannot fit.
        assert!(victims(&slots, 4, 3999).is_err());
        // A target that already fits stops nothing.
        assert_eq!(victims(&slots, 4, 6500), Ok(vec![]));
    }

    #[test]
    fn forwards_by_model_and_isolates_an_unavailable_child() {
        let (echo, echo_child) = echo_child(2);
        // A port with nothing listening stands in for a child that died.
        let dead = TcpListener::bind("127.0.0.1:0")
            .unwrap()
            .local_addr()
            .unwrap();
        let pool = Arc::new(Pool {
            models: vec![
                model("qwen", ModelKind::Qwen, Some(echo)),
                model("julia", ModelKind::Julia, Some(dead)),
                model("broken", ModelKind::Julia, None),
            ],
            launcher: None,
            budget_mib: None,
            residency: Mutex::new(()),
        });
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let server = thread::spawn(move || {
            proxy_models(&listener, &pool, TransportLimits::default(), Some(9))
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
