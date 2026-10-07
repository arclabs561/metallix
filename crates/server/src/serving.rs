//! `mx serve`: registered model workers, per-model admission and HTTP routing.

use std::{
    net::{SocketAddr, TcpListener},
    path::{Path, PathBuf},
    process::ExitCode,
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicUsize, Ordering},
        mpsc::{Receiver, SyncSender, sync_channel},
    },
    thread,
    time::{Duration, Instant},
};

use engine::blocks::BlockTokens;
use serde::Deserialize;
use serde_json::{Value, json};
use tracing::field::Empty;

use crate::{
    chat_generation::{EngineSeed, ResidentChatLimits},
    engine_loop::{self, EngineClient, EngineLimits, EngineMessage, EngineModel},
    generation_routes::{Generation, error_body, error_response},
    gpu,
    http_transport::{Connection, TransportLimits},
    responses::{echo_request_id, json_response},
    serve_registry::{self, ModelWorker, ServedEntry},
    trace_context::RequestContext,
};

#[cfg(test)]
use crate::chat_generation::{ChatBackend, ChatSession};

fn busy_response(connection: Connection) {
    error_response(
        connection,
        503,
        Some("server_busy"),
        "one generation is already active",
    );
}

fn unavailable_response(connection: Connection) {
    error_response(
        connection,
        503,
        Some("model_worker_unavailable"),
        "the model worker is unavailable",
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

/// One admitted request for a model worker: a generation or a decision.
struct GenerationJob {
    connection: Connection,
    work: Work,
    _admission: Admission,
    /// The request's `http.request` span, entered again on the worker thread.
    span: tracing::Span,
}

enum Work {
    Transcribe {
        request: crate::transcriptions::Request,
        generation_timeout: Duration,
    },
    Respond {
        generation: Box<Generation>,
        /// The server-unique suffix each protocol prefixes with its own style.
        id: String,
        generation_timeout: Duration,
    },
    Decide {
        body: Vec<u8>,
        model: String,
    },
    Embed {
        body: Vec<u8>,
        model: String,
    },
    Rerank {
        body: Vec<u8>,
        model: String,
    },
}

/// Adapts a generation-only test backend to the worker loop.
#[cfg(test)]
struct GenerationOnly<'a>(&'a mut dyn ChatBackend);

#[cfg(test)]
impl ModelWorker for GenerationOnly<'_> {
    fn chat(&mut self) -> Option<&mut dyn ChatBackend> {
        Some(&mut *self.0)
    }

    fn decide(&mut self, _body: &[u8], _model: &str) -> Option<Result<Value, String>> {
        None
    }
}

#[cfg(test)]
fn worker_loop(session: &mut dyn ChatBackend, jobs: Receiver<GenerationJob>) {
    model_worker_loop(&mut GenerationOnly(session), jobs);
}

#[cfg(test)]
fn model_worker_loop(worker: &mut dyn ModelWorker, jobs: Receiver<GenerationJob>) {
    serve_jobs(worker, jobs, None);
}

/// The startup report from a model thread: a batching engine's client, or
/// `None` for a model served one job at a time.
type Startup = Result<Option<EngineClient<GenerationJob>>, String>;

/// The batching engine's parts when `limits` ask for batching and the model
/// can page its state; otherwise the model is served one job at a time.
fn engine_seed(
    worker: &mut dyn ModelWorker,
    limits: EngineLimits,
) -> Option<Result<EngineSeed, String>> {
    if limits.batches() {
        worker.engine_seed()
    } else {
        None
    }
}

/// Builds a paged pool from `seed` and runs the batching engine on this
/// thread until the acceptor drops its client. Non-generation jobs run
/// between engine steps.
fn serve_engine(
    worker: &mut dyn ModelWorker,
    seed: Result<EngineSeed, String>,
    limits: EngineLimits,
    startup: &SyncSender<Startup>,
) {
    let built = seed.and_then(|seed| {
        let pool = seed
            .weights
            .pool_for_budget(seed.kv_budget_bytes, BlockTokens::DEFAULT)
            .map_err(|error| format!("KV pool: {error}"))?;
        Ok((seed, pool))
    });
    let (seed, pool) = match built {
        Ok(built) => built,
        Err(error) => {
            let _ = startup.send(Err(error));
            return;
        }
    };
    let session = match seed.weights.session(pool) {
        Ok(session) => session,
        Err(error) => {
            let _ = startup.send(Err(format!("KV pool: {error}")));
            return;
        }
    };
    tracing::info!(
        pool_bytes = session.pool_bytes(),
        blocks = session.blocks().total_blocks(),
        max_num_seqs = limits.max_num_seqs,
        cache_limit_bytes = gpu::cache_limit_bytes(),
        "batching engine ready"
    );
    let model = Arc::new(EngineModel {
        format: Arc::clone(&seed.format),
        sampling_defaults: seed.sampling_defaults,
        model: seed.model.clone(),
        vocabulary_size: seed.vocabulary_size,
        context_limit: seed.context_limit,
        pool_bytes: session.pool_bytes() as u64,
        load_ms: seed.load_ms,
    });
    let (messages, receiver) = sync_channel(limits.capacity());
    if startup
        .send(Ok(Some(EngineClient::new(model, messages))))
        .is_err()
    {
        return;
    }
    engine_loop::run(session, &seed.format, limits, &receiver, |job| {
        serve_job(worker, job, None);
    });
}

/// Runs admitted jobs in order, keeping wired model memory resident between
/// them (see [`gpu::keep_resident`]). `capture` wraps the first job in a
/// Metal capture.
fn serve_jobs(
    worker: &mut dyn ModelWorker,
    jobs: Receiver<GenerationJob>,
    mut capture: Option<PathBuf>,
) {
    for job in gpu::keep_resident(jobs) {
        serve_job(worker, job, capture.take());
    }
}

/// Runs one admitted job inside its request span, flushing after it exits.
fn serve_job(worker: &mut dyn ModelWorker, job: GenerationJob, capture: Option<PathBuf>) {
    let GenerationJob {
        connection,
        work,
        _admission: admission,
        span,
    } = job;
    span.in_scope(|| {
        let started = Instant::now();
        let capturing = capture.and_then(|path| {
            gpu::Capture::start(&path)
                .inspect_err(|error| tracing::error!("{error}"))
                .ok()
        });
        gpu::reset_peak_memory();
        run_job(worker, connection, work, admission, &span);
        gpu::Memory::record_on(&span);
        drop(capturing);
        tracing::info!(
            elapsed_ms = started.elapsed().as_secs_f64() * 1000.0,
            "request finished"
        );
    });
    crate::telemetry::flush();
}

fn run_job(
    worker: &mut dyn ModelWorker,
    mut connection: Connection,
    work: Work,
    admission: Admission,
    span: &tracing::Span,
) {
    let request_id = connection.request_id().map(str::to_owned);
    match work {
        Work::Transcribe {
            request,
            generation_timeout,
        } => {
            let _socket = connection.hold_open();
            let stage = tracing::info_span!("audio.transcribe");
            let outcome = {
                let mut cancelled = || connection.client_gone();
                let mut control =
                    crate::transcriptions::Control::new(generation_timeout, &mut cancelled);
                stage.in_scope(|| worker.transcribe(&request, &mut control))
            };
            drop(admission);
            crate::transcriptions::respond(connection, outcome);
        }
        Work::Respond {
            generation,
            id,
            generation_timeout,
        } => {
            // Dropped in reverse order: the model is free before the socket
            // closes, so a front process that admits its next request when
            // this response ends never finds the model still busy.
            let _socket = connection.hold_open();
            let _admission = admission;
            // Routing admits only models that declare generation.
            let Some(session) = worker.chat() else {
                unsupported_response(connection, "generate");
                return;
            };
            if let Err(error) = generation.respond(connection, session, &id, generation_timeout) {
                span.record("error.type", "response_failed");
                tracing::warn!("response failed: {error}");
            }
        }
        Work::Decide { body, model } => {
            let stage = tracing::info_span!("decide.score");
            let outcome = staged(&stage, None, span, request_id.as_deref(), || {
                worker.decide(&body, &model)
            });
            // Free the model before answering, so a client that sends its
            // next request as soon as this one completes is not refused.
            drop(admission);
            body_response(connection, outcome, "decide");
        }
        Work::Embed { body, model } => {
            let stage = tracing::info_span!("embed.batch", embed_ms = Empty);
            let outcome = staged(
                &stage,
                Some("embed_ms"),
                span,
                request_id.as_deref(),
                || worker.embed(&body, &model),
            );
            drop(admission);
            body_response(connection, outcome, "embed");
        }
        Work::Rerank { body, model } => {
            let stage = tracing::info_span!("rerank.score", rerank_ms = Empty);
            let outcome = staged(
                &stage,
                Some("rerank_ms"),
                span,
                request_id.as_deref(),
                || worker.rerank(&body, &model),
            );
            drop(admission);
            body_response(connection, outcome, "rerank");
        }
    }
}

/// Runs model work inside `stage`, which closes once the model has returned
/// host values (and so after its MLX evaluation). The response's own
/// `metallix` timing field, when it has one, is recorded on `stage`, so the
/// span and the JSON agree; usage and failures go on the request span.
fn staged(
    stage: &tracing::Span,
    timing: Option<&'static str>,
    request: &tracing::Span,
    request_id: Option<&str>,
    run: impl FnOnce() -> Option<Result<Value, String>>,
) -> Option<Result<Value, String>> {
    let mut outcome = stage.in_scope(run);
    match &mut outcome {
        Some(Ok(value)) => {
            if let Some((timing, ms)) =
                timing.and_then(|timing| Some((timing, value["metallix"][timing].as_f64()?)))
            {
                stage.record(timing, ms);
            }
            let usage = &value["usage"];
            if let Some(tokens) = usage["prompt_tokens"]
                .as_u64()
                .or_else(|| usage["input_tokens"].as_u64())
            {
                request.record("gen_ai.usage.input_tokens", tokens);
            }
            echo_request_id(value, request_id);
        }
        Some(Err(_)) => {
            request.record("error.type", "invalid_request");
        }
        None => {
            request.record("error.type", "unsupported_capability");
        }
    }
    outcome
}

fn body_response(connection: Connection, outcome: Option<Result<Value, String>>, capability: &str) {
    match outcome {
        Some(Ok(value)) => json_response(connection, 200, &value),
        Some(Err(error)) => error_response(connection, 400, None, &error),
        None => unsupported_response(connection, capability),
    }
}

fn unsupported_response(connection: Connection, capability: &str) {
    error_response(
        connection,
        400,
        Some("unsupported_capability"),
        &format!("model does not support {capability}"),
    );
}

/// Serves `entry` in this process as a child of the `mx serve` front process,
/// exiting when the front process closes this process's stdin. `capture`
/// records the first request as a Metal capture.
pub(crate) fn serve_child(
    entry: ServedEntry,
    address: SocketAddr,
    limits: ResidentChatLimits,
    engine_limits: EngineLimits,
    generation_timeout: Duration,
    capture: Option<&Path>,
) -> ExitCode {
    if let Err(error) = crate::telemetry::finish_on_signal() {
        tracing::error!("mx serve: {error}");
        return ExitCode::FAILURE;
    }
    if let Some(path) = capture {
        if let Err(error) = gpu::check_capture_path(path) {
            tracing::error!("mx serve: {error}");
            return ExitCode::FAILURE;
        }
    }
    thread::spawn(|| {
        let _ = std::io::copy(&mut std::io::stdin(), &mut std::io::sink());
        crate::telemetry::finish();
        std::process::exit(0);
    });
    match serve_inner(
        &[entry],
        address,
        limits,
        engine_limits,
        generation_timeout,
        capture,
    ) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            tracing::error!("mx serve: {error}");
            ExitCode::FAILURE
        }
    }
}

/// A registered model as the acceptor sees it: identity, capability and its
/// own worker channel, admission flag and liveness.
struct ServedModel {
    id: String,
    generates: bool,
    capabilities: &'static [&'static str],
    jobs: SyncSender<GenerationJob>,
    /// Admits one job at a time: a serial model's every request, or an
    /// engine model's non-generation work.
    occupied: Arc<AtomicBool>,
    alive: Arc<AtomicBool>,
    /// Set when the model decodes in batches: generation goes through the
    /// engine, up to its capacity, and other work runs between its steps.
    engine: Option<EngineRoute>,
}

/// How the acceptor reaches a batching engine.
struct EngineRoute {
    client: EngineClient<GenerationJob>,
    /// Admitted generations: running in the engine or waiting in it.
    admitted: Arc<AtomicUsize>,
    capacity: usize,
}

/// One admitted engine generation, released when its writer finishes.
struct EngineAdmission {
    admitted: Arc<AtomicUsize>,
}

impl EngineAdmission {
    fn try_acquire(route: &EngineRoute) -> Option<Self> {
        Self::reserve(&route.admitted, route.capacity)
    }

    /// Takes one of `capacity` slots counted by `admitted`, if one is free.
    fn reserve(admitted: &Arc<AtomicUsize>, capacity: usize) -> Option<Self> {
        let mut current = admitted.load(Ordering::Acquire);
        loop {
            if current >= capacity {
                return None;
            }
            match admitted.compare_exchange_weak(
                current,
                current + 1,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => break,
                Err(now) => current = now,
            }
        }
        Some(Self {
            admitted: Arc::clone(admitted),
        })
    }
}

#[cfg(test)]
pub(crate) fn test_engine_admission(admitted: &Arc<AtomicUsize>) -> impl Send + use<> {
    EngineAdmission::reserve(admitted, 1).expect("free test admission")
}

impl Drop for EngineAdmission {
    fn drop(&mut self) {
        self.admitted.fetch_sub(1, Ordering::AcqRel);
    }
}

#[allow(
    clippy::too_many_lines,
    reason = "one ordered load, announce and serve sequence"
)]
fn serve_inner(
    entries: &[ServedEntry],
    address: SocketAddr,
    limits: ResidentChatLimits,
    engine_limits: EngineLimits,
    generation_timeout: Duration,
    capture: Option<&Path>,
) -> Result<(), String> {
    if !address.ip().is_loopback() {
        return Err("this experimental server binds only to loopback".into());
    }
    let mut models = Vec::with_capacity(entries.len());
    let mut workers = Vec::with_capacity(entries.len());
    for entry in entries {
        let (job_sender, job_receiver) = sync_channel(0);
        let (startup_sender, startup_receiver) = sync_channel(1);
        let alive = Arc::new(AtomicBool::new(false));
        let worker_liveness = Arc::clone(&alive);
        let worker_entry = entry.clone();
        // One model per process today, so the capture goes to its first request.
        let worker_capture = capture.map(Path::to_path_buf);
        // This thread owns the model's MLX arrays; its name tags every log line.
        let worker = thread::Builder::new()
            .name(format!("model-{}", entry.id))
            .spawn(move || {
                let load = tracing::info_span!(
                    "model.load",
                    gen_ai.request.model = %worker_entry.id,
                    kind = ?worker_entry.kind,
                    load_ms = Empty,
                    mlx.active_bytes = Empty,
                    mlx.cache_bytes = Empty,
                    mlx.peak_bytes = Empty,
                    "error.type" = Empty,
                );
                let entered = load.enter();
                let started = Instant::now();
                let mut worker = match serve_registry::load(&worker_entry, limits) {
                    Ok(worker) => worker,
                    Err(error) => {
                        load.record("error.type", "load_failed");
                        let _ = startup_sender.send(Err(error));
                        return;
                    }
                };
                let liveness = WorkerLiveness {
                    alive: worker_liveness,
                };
                liveness.alive.store(true, Ordering::Release);
                // A generation backend reports its own load time; others use wall time.
                let load_ms = worker
                    .chat()
                    .map_or(started.elapsed().as_secs_f64() * 1000.0, |chat| {
                        chat.load_ms()
                    });
                load.record("load_ms", load_ms);
                gpu::Memory::record_on(&load);
                tracing::info!("model loaded");
                // Before the child announces itself, so the first request
                // does not wait on first-use kernel builds.
                if let Err(error) = worker.warm() {
                    load.record("error.type", "warmup_failed");
                    let _ = startup_sender.send(Err(format!("warmup failed: {error}")));
                    return;
                }
                // Before any request, since MLX must not change the wired
                // limit while an asynchronous evaluation runs.
                if worker.chat().is_some() {
                    let wired_bytes = gpu::wire_resident(
                        limits
                            .kv_budget_bytes()
                            .saturating_add(limits.prefix_cache_bytes()),
                    );
                    tracing::info!(wired_bytes = ?wired_bytes, "model memory wired");
                }
                drop(entered);
                drop(load);
                match engine_seed(worker.as_mut(), engine_limits) {
                    Some(seed) => {
                        serve_engine(worker.as_mut(), seed, engine_limits, &startup_sender);
                    }
                    None => {
                        if startup_sender.send(Ok(None)).is_ok() {
                            serve_jobs(worker.as_mut(), job_receiver, worker_capture);
                        }
                    }
                }
            })
            .map_err(|error| format!("{}: could not start the model thread: {error}", entry.id))?;
        workers.push(worker);
        let engine = match startup_receiver.recv() {
            Ok(Ok(client)) => client.map(|client| EngineRoute {
                client,
                admitted: Arc::new(AtomicUsize::new(0)),
                capacity: engine_limits.capacity(),
            }),
            failed => {
                drop(models);
                for worker in workers {
                    let _ = worker.join();
                }
                return Err(match failed {
                    Ok(Err(error)) => format!("{}: {error}", entry.id),
                    _ => format!("{}: model worker ended before startup", entry.id),
                });
            }
        };
        models.push(ServedModel {
            id: entry.id.clone(),
            generates: entry.kind.generates(),
            capabilities: entry.kind.capabilities(),
            jobs: job_sender,
            occupied: Arc::new(AtomicBool::new(false)),
            alive,
            engine,
        });
    }
    let outcome = match TcpListener::bind(address) {
        Ok(server) => {
            // The front process reads this line to find a child's ephemeral
            // port, so it is written raw rather than as a filtered log event.
            let address = server.local_addr().unwrap_or(address);
            let max_running = models
                .iter()
                .map(|model| {
                    model
                        .engine
                        .as_ref()
                        .map_or(1, |_| engine_limits.max_num_seqs)
                })
                .max()
                .unwrap_or(1);
            eprintln!(
                "mx listening on http://{address}; models={}; max_running={max_running}; {} total tokens; kv_budget_bytes={}",
                entries.len(),
                limits.context_tokens(),
                limits.kv_budget_bytes(),
            );
            serve_models(
                &server,
                &models,
                generation_timeout,
                TransportLimits::default(),
                None,
            )
        }
        Err(error) => Err(error.to_string()),
    };
    drop(models);
    for worker in workers {
        worker
            .join()
            .map_err(|_| String::from("model worker panicked"))?;
    }
    outcome
}

#[cfg(test)]
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
#[cfg(test)]
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
    let models = [ServedModel {
        id: model_id.to_owned(),
        generates: true,
        capabilities: &["generate"],
        jobs: job_sender.clone(),
        occupied: Arc::clone(occupied),
        engine: None,
        alive: Arc::clone(worker_alive),
    }];
    serve_models(
        server,
        &models,
        generation_timeout,
        transport_limits,
        request_limit,
    )
}

#[derive(Deserialize)]
struct DecisionTarget {
    model: String,
}

/// Routes each request to the model it names. Admission is per model and is
/// taken after the request is read, since the body names the model.
#[allow(
    clippy::too_many_lines,
    reason = "one ordered route, model, capability and admission sequence"
)]
fn serve_models(
    server: &TcpListener,
    models: &[ServedModel],
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
        // ponytail: one dead worker stops the server, as with a single model;
        // per-model unavailability belongs with on-demand loading.
        if models
            .iter()
            .any(|model| !model.alive.load(Ordering::Acquire))
        {
            unavailable_response(connection);
            return Err(String::from("model worker is unavailable"));
        }
        let mut connection = connection;
        let request = match connection
            .read_request_with(&crate::transcriptions::UploadLimits(transport_limits))
        {
            Ok(request) => request,
            Err(error) => {
                error_response(connection, error.status, None, &error.describe());
                continue;
            }
        };
        let context = RequestContext::from_headers(
            request.trace.traceparent.as_deref(),
            request.trace.request_id.as_deref(),
        );
        connection.set_request_id(&context.request_id);
        connection.set_cache_salt(request.trace.cache_salt.clone());
        let span = tracing::info_span!(
            "http.request",
            request_id = %context.request_id,
            trace_id = %context.trace.trace_id(),
            route = %request.path,
            http.request.method = %request.method,
            gen_ai.operation.name = Empty,
            gen_ai.provider.name = "metallix",
            gen_ai.request.model = Empty,
            gen_ai.request.max_tokens = Empty,
            metallix.output_tokens.limit = Empty,
            metallix.output_budget.source = Empty,
            metallix.context_tokens.effective = Empty,
            gen_ai.response.finish_reason = Empty,
            gen_ai.usage.input_tokens = Empty,
            gen_ai.usage.output_tokens = Empty,
            mlx.active_bytes = Empty,
            mlx.cache_bytes = Empty,
            mlx.peak_bytes = Empty,
            "error.type" = Empty,
        );
        let _entered = span.enter();
        let capability = match (request.method.as_str(), request.path.as_str()) {
            ("GET", "/healthz") => {
                json_response(connection, 200, &json!({"status":"ready"}));
                continue;
            }
            ("GET", "/v1/models") => {
                let data: Vec<Value> = models
                    .iter()
                    .map(|model| json!({"id":model.id,"object":"model","owned_by":"local","capabilities":model.capabilities}))
                    .collect();
                json_response(connection, 200, &json!({"object":"list","data":data}));
                continue;
            }
            ("POST", path) if Generation::serves(path) => "generate",
            ("POST", "/v1/decisions") => "decide",
            ("POST", "/v1/embeddings") => "embed",
            ("POST", "/v1/rerank") => "rerank",
            ("POST", crate::transcriptions::PATH) => "transcribe",
            _ => {
                error_response(connection, 404, None, "unknown endpoint");
                continue;
            }
        };
        span.record(
            "gen_ai.operation.name",
            match capability {
                "generate" => "chat",
                "embed" => "embeddings",
                other => other,
            },
        );
        let generation = capability == "generate";
        let parsed = if generation {
            Generation::parse(&request.path, &request.body)
                .map(|(model, generation)| (model, Some(generation)))
                .map_err(|error| (400, error))
        } else if capability == "transcribe" {
            crate::transcriptions::parse(&request.body, request.content_type.as_deref())
                .map(|form| (form.model.to_owned(), None))
                .map_err(|error| {
                    (
                        error.status,
                        error_body(
                            Some(&request.path),
                            error.status,
                            None,
                            &error.message,
                            None,
                        ),
                    )
                })
        } else {
            serde_json::from_slice::<DecisionTarget>(&request.body)
                .map(|target| (target.model, None))
                .map_err(|error| {
                    (
                        400,
                        error_body(
                            Some(request.path.as_str()),
                            400,
                            None,
                            &error.to_string(),
                            None,
                        ),
                    )
                })
        };
        let (model_id, parsed) = match parsed {
            Ok(parsed) => parsed,
            Err((status, error)) => {
                json_response(connection, status, &error);
                continue;
            }
        };
        span.record("gen_ai.request.model", model_id.as_str());
        if let Some(parsed) = &parsed {
            span.record("gen_ai.operation.name", parsed.operation());
        }
        let Some(model) = models.iter().find(|model| model.id == model_id) else {
            error_response(connection, 404, None, "model is not loaded");
            continue;
        };
        let work = if let Some(parsed) = parsed {
            if !model.generates {
                unsupported_response(connection, "generate");
                continue;
            }
            Work::Respond {
                generation: Box::new(parsed),
                id: format!("{}_{}", std::process::id(), index),
                generation_timeout,
            }
        } else {
            if !model.capabilities.contains(&capability) {
                unsupported_response(connection, capability);
                continue;
            }
            let (body, model) = (request.body, model_id);
            match capability {
                "decide" => Work::Decide { body, model },
                "rerank" => Work::Rerank { body, model },
                "transcribe" => match crate::transcriptions::Request::from_body(
                    body,
                    request.content_type.as_deref(),
                ) {
                    Ok(request) => Work::Transcribe {
                        request,
                        generation_timeout,
                    },
                    Err(error) => {
                        error_response(connection, error.status, None, &error.message);
                        continue;
                    }
                },
                _ => Work::Embed { body, model },
            }
        };
        if let Some(route) = &model.engine {
            route_to_engine(route, &model.occupied, connection, work, &span)?;
            continue;
        }
        let Some(admission) = Admission::try_acquire(&model.occupied) else {
            span.record("error.type", "server_busy");
            tracing::info!("rejected: model is busy");
            if generation {
                busy_response(connection);
            } else {
                error_response(
                    connection,
                    503,
                    Some("server_busy"),
                    "the model is busy with another request",
                );
            }
            continue;
        };
        let job = GenerationJob {
            connection,
            work,
            _admission: admission,
            span: span.clone(),
        };
        if let Err(error) = model.jobs.send(job) {
            unavailable_response(error.0.connection);
            return Err(String::from("model worker is unavailable"));
        }
    }
    Ok(())
}

/// Admits `work` for an engine model. A generation gets its own writer
/// thread, which prepares the turn and streams the engine's tokens; other work
/// waits for the engine to run it between steps. Errs only when the engine is
/// gone.
fn route_to_engine(
    route: &EngineRoute,
    occupied: &Arc<AtomicBool>,
    connection: Connection,
    work: Work,
    span: &tracing::Span,
) -> Result<(), String> {
    match work {
        Work::Respond {
            generation,
            id,
            generation_timeout,
        } => {
            let Some(admission) = EngineAdmission::try_acquire(route) else {
                span.record("error.type", "server_busy");
                tracing::info!("rejected: model is at its batching capacity");
                busy_response(connection);
                return Ok(());
            };
            let mut client = route.client.clone();
            let span = span.clone();
            let spawned = thread::Builder::new()
                .name(String::from("writer"))
                .spawn(move || {
                    span.in_scope(|| {
                        // Dropped in reverse order: capacity is released before
                        // the socket closes, as for serial jobs.
                        let _socket = connection.hold_open();
                        let _admission = admission;
                        if let Err(error) =
                            generation.respond(connection, &mut client, &id, generation_timeout)
                        {
                            span.record("error.type", "response_failed");
                            tracing::warn!("response failed: {error}");
                        }
                        gpu::Memory::record_on(&span);
                    });
                    // The writer span has exited, so this flush includes its end.
                    crate::telemetry::flush();
                });
            if let Err(error) = spawned {
                tracing::error!("could not start a response writer: {error}");
            }
            Ok(())
        }
        work => {
            let Some(admission) = Admission::try_acquire(occupied) else {
                span.record("error.type", "server_busy");
                tracing::info!("rejected: model is busy");
                error_response(
                    connection,
                    503,
                    Some("server_busy"),
                    "the model is busy with another request",
                );
                return Ok(());
            };
            let job = GenerationJob {
                connection,
                work,
                _admission: admission,
                span: span.clone(),
            };
            route
                .client
                .messages()
                .send(EngineMessage::Exclusive(job))
                .map_err(|error| {
                    if let EngineMessage::Exclusive(job) = error.0 {
                        unavailable_response(job.connection);
                    }
                    String::from("model worker is unavailable")
                })
        }
    }
}

#[cfg(test)]
mod tests {
    /// How long a test waits for a response, event or release before it
    /// calls the server hung. Generous because a loaded machine (other
    /// builds, MLX work) stretches these waits to seconds; what each test
    /// asserts does not depend on it.
    const HUNG: Duration = Duration::from_secs(30);

    use std::{
        env,
        io::{Read as _, Write as _},
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
    use crate::{
        chat_generation::{ChatFinishReason, ChatGenerationError, ChatMessage, ChatRequest},
        responses::{Request, messages, respond, tools},
    };

    fn wait_for_admission_release(occupied: &AtomicBool) {
        let deadline = Instant::now() + HUNG;
        while occupied.load(Ordering::Acquire) {
            assert!(
                Instant::now() < deadline,
                "worker completion did not release admission"
            );
            thread::yield_now();
        }
    }

    #[test]
    fn engine_admission_counts_up_to_capacity_and_releases_on_drop() {
        let admitted = Arc::new(AtomicUsize::new(0));
        let held = (0..3)
            .map(|_| EngineAdmission::reserve(&admitted, 3).expect("a free slot"))
            .collect::<Vec<_>>();
        assert!(EngineAdmission::reserve(&admitted, 3).is_none());
        drop(held);
        assert_eq!(admitted.load(Ordering::Acquire), 0);
        assert!(EngineAdmission::reserve(&admitted, 3).is_some());
    }

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
            on_token: &mut dyn FnMut(chat_format::TurnDelta) -> Result<(), String>,
        ) -> Result<crate::chat_generation::ChatGeneration, ChatGenerationError> {
            self.calls += 1;
            on_token(chat_format::TurnDelta::Text("partial".into()))
                .map_err(ChatGenerationError::Message)?;
            Err(ChatGenerationError::DeadlineExceeded)
        }
    }

    #[test]
    fn unsupported_scoring_has_a_typed_http_error_for_both_routes() {
        for (path, body) in [
            (
                "/v1/score",
                r#"{"model":"control","prompt":"a","continuation":"b"}"#,
            ),
            (
                "/v1/completions",
                r#"{"model":"control","prompt":"a","max_tokens":0,"echo":true}"#,
            ),
        ] {
            let mut backend = DeadlineBackend::default();
            let wire = crate::sse::test_support::exchange(path, body, |connection, body| {
                let (_, generation) = crate::generation_routes::Generation::parse(path, body)
                    .expect("valid scoring request");
                generation
                    .respond(connection, &mut backend, "score", HUNG)
                    .unwrap();
            });
            assert!(
                wire.starts_with("HTTP/1.1 501 Not Implemented\r\n"),
                "{wire}"
            );
            let (_, value) = crate::sse::test_support::json_body(&wire);
            assert_eq!(value["error"]["code"], "scoring_unsupported");
            assert_eq!(backend.calls, 0, "unsupported scoring must not generate");
        }
    }

    #[test]
    fn cancelled_scoring_holds_socket_and_admission_until_prefill_returns() {
        use crate::chat_generation::{ScoreError, ScoreRequest, ScoreResult};

        struct BlockingScore {
            entered: SyncSender<()>,
            release: Receiver<()>,
            calls: std::cell::Cell<usize>,
        }
        impl ChatBackend for BlockingScore {
            fn load_ms(&self) -> f64 {
                0.0
            }
            fn generate_with_timeout(
                &mut self,
                _: ChatRequest<'_>,
                _: Duration,
                _: &mut dyn FnMut(chat_format::TurnDelta) -> Result<(), String>,
            ) -> Result<crate::chat_generation::ChatGeneration, ChatGenerationError> {
                panic!("score must not enter generation")
            }
            fn score(&self, _: ScoreRequest<'_>) -> Result<ScoreResult, ScoreError> {
                self.calls.set(self.calls.get() + 1);
                if self.calls.get() == 1 {
                    self.entered.send(()).unwrap();
                    self.release
                        .recv_timeout(HUNG)
                        .expect("release blocked prefill");
                }
                Err(ScoreError::Execution(ChatGenerationError::Message(
                    "finished prefill".into(),
                )))
            }
        }
        fn connect(address: std::net::SocketAddr) -> TcpStream {
            let mut socket = TcpStream::connect(address).unwrap();
            socket.set_read_timeout(Some(HUNG)).unwrap();
            let body = r#"{"model":"control","prompt":"a","continuation":"b"}"#;
            write!(socket, "POST /v1/score HTTP/1.1\r\nHost: localhost\r\nx-metallix-cancel-on-eof: 1\r\nContent-Length: {}\r\n\r\n{body}", body.len()).unwrap();
            socket
        }
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let (jobs, receiver) = sync_channel(0);
        let (entered, started) = sync_channel(1);
        let (release, resume) = sync_channel(1);
        let worker = thread::spawn(move || {
            let mut backend = BlockingScore {
                entered,
                release: resume,
                calls: std::cell::Cell::new(0),
            };
            worker_loop(&mut backend, receiver);
            backend.calls.get()
        });
        let occupied = Arc::new(AtomicBool::new(false));
        let server_occupied = Arc::clone(&occupied);
        let sender = jobs.clone();
        let server = thread::spawn(move || {
            serve_listener(
                &listener,
                "control",
                &sender,
                &server_occupied,
                &Arc::new(AtomicBool::new(true)),
                Duration::from_millis(1),
                Some(3),
            )
        });
        let mut first = connect(address);
        started.recv_timeout(HUNG).unwrap();
        first.shutdown(Shutdown::Write).unwrap();
        first
            .set_read_timeout(Some(Duration::from_millis(50)))
            .unwrap();
        let held_read = first.read(&mut [0_u8; 1]);
        let held_admission = occupied.load(Ordering::Acquire);
        let mut busy = String::new();
        connect(address).read_to_string(&mut busy).unwrap();
        // Release before assertions so a failed invariant cannot strand the worker.
        release.send(()).unwrap();
        first.set_read_timeout(Some(HUNG)).unwrap();
        let mut finished = String::new();
        first.read_to_string(&mut finished).unwrap();
        wait_for_admission_release(&occupied);
        let mut next = String::new();
        connect(address).read_to_string(&mut next).unwrap();
        server.join().unwrap().unwrap();
        drop(jobs);
        let calls = worker.join().unwrap();
        assert!(held_admission, "cancelled prefill still owns admission");
        assert!(
            held_read.is_err_and(|error| matches!(
                error.kind(),
                std::io::ErrorKind::WouldBlock | std::io::ErrorKind::TimedOut
            )),
            "no response or EOF before prefill returns"
        );
        assert!(busy.starts_with("HTTP/1.1 503"), "{busy}");
        assert!(next.starts_with("HTTP/1.1 400"), "{next}");
        assert_eq!(calls, 2, "busy request must not execute");
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
            on_token: &mut dyn FnMut(chat_format::TurnDelta) -> Result<(), String>,
        ) -> Result<crate::chat_generation::ChatGeneration, ChatGenerationError> {
            self.turns += 1;
            if self.turns == 1 {
                self.started
                    .send(())
                    .map_err(|_| ChatGenerationError::Message(String::from("test start closed")))?;
                let payload = "x".repeat(256 * 1024);
                for _ in 0..32 {
                    let began = Instant::now();
                    if let Err(error) = on_token(chat_format::TurnDelta::Text(payload.clone())) {
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
            on_token(chat_format::TurnDelta::Text("recovered".into()))
                .map_err(ChatGenerationError::Message)?;
            Ok({
                let mut generated = crate::chat_generation::ChatGeneration::scripted(
                    String::from("recovered"),
                    true,
                    ChatFinishReason::Eos,
                    crate::chat_generation::ChatGenerationMetrics {
                        context_tokens: 2_048,
                        planned_kv_bytes: 0,
                        session_load_ms: 0.0,
                        render_ms: 0.0,
                        prefill_ms: 0.0,
                        time_to_first_token_ms: Some(0.0),
                        decode_ms: vec![],
                        decode_total_ms: 0.0,
                        prompt_tokens: 1,
                        cached_prompt_tokens: 0,
                        cache_write_tokens: 0,
                        generated_tokens: 1,
                        speculation: None,
                    },
                );
                generated.generated_token_ids = vec![1];
                generated
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
            on_token: &mut dyn FnMut(chat_format::TurnDelta) -> Result<(), String>,
        ) -> Result<crate::chat_generation::ChatGeneration, ChatGenerationError> {
            self.turns += 1;
            let text = if self.turns == 1 {
                "holding"
            } else {
                "recovered"
            };
            on_token(chat_format::TurnDelta::Text(text.into()))
                .map_err(ChatGenerationError::Message)?;
            if self.turns == 1 {
                self.entered.send(()).map_err(|_| {
                    ChatGenerationError::Message(String::from("test barrier closed"))
                })?;
                self.release.recv_timeout(HUNG).map_err(|_| {
                    ChatGenerationError::Message(String::from("test barrier timed out"))
                })?;
            }
            Ok({
                let mut generated = crate::chat_generation::ChatGeneration::scripted(
                    text,
                    true,
                    ChatFinishReason::Eos,
                    crate::chat_generation::ChatGenerationMetrics {
                        context_tokens: 2_048,
                        planned_kv_bytes: 0,
                        session_load_ms: 0.0,
                        render_ms: 0.0,
                        prefill_ms: 0.0,
                        time_to_first_token_ms: Some(0.0),
                        decode_ms: vec![],
                        decode_total_ms: 0.0,
                        prompt_tokens: 1,
                        cached_prompt_tokens: 0,
                        cache_write_tokens: 0,
                        generated_tokens: 1,
                        speculation: None,
                    },
                );
                generated.generated_token_ids = vec![1];
                generated
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

    #[test]
    fn streaming_generation_deadline_emits_one_terminal_failure() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let client = thread::spawn(move || {
            let body = br#"{"model":"control","input":"hello","stream":true}"#;
            let mut stream = TcpStream::connect(address).unwrap();
            stream.set_read_timeout(Some(HUNG)).unwrap();
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
        assert_eq!(events[4]["response"]["error"]["code"], "server_error");
        assert_eq!(
            events[4]["response"]["error"]["metallix_code"],
            "generation_timeout"
        );
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
            .set_read_timeout(Some(HUNG))
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
            .set_read_timeout(Some(HUNG))
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
            .recv_timeout(HUNG)
            .expect("worker holds after the real callback delta");

        let mut busy = TcpStream::connect(address).expect("connect concurrent request");
        busy.set_read_timeout(Some(HUNG)).expect("bound busy reads");
        request(&mut busy, &body);
        let (busy_status, busy_body) = fixed_http_response(&mut busy);
        assert_eq!(busy_status, 503);
        assert_eq!(
            serde_json::from_slice::<Value>(&busy_body).expect("busy response JSON"),
            json!({"error":{"message":"one generation is already active","type":"server_error","param":null,"code":"server_busy"}})
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
            .set_read_timeout(Some(HUNG))
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
            .recv_timeout(HUNG)
            .expect("worker begins the stalled response");
        let (elapsed, error) = failure_receiver
            .recv_timeout(RESPONSE_DEADLINE + HUNG)
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
            .set_read_timeout(Some(HUNG))
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
                .set_read_timeout(Some(HUNG))
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
            assert!(wire.contains(r#""code":"server_error""#));
            assert!(wire.contains(r#""metallix_code":"generation_timeout""#));
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
            .set_read_timeout(Some(HUNG))
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
            .set_read_timeout(Some(HUNG))
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

    /// A generating model that can batch, and records whether the engine
    /// asked for its parts.
    struct BatchCapable {
        asked: bool,
    }

    impl ModelWorker for BatchCapable {
        fn chat(&mut self) -> Option<&mut dyn ChatBackend> {
            None
        }

        fn engine_seed(&mut self) -> Option<Result<EngineSeed, String>> {
            self.asked = true;
            Some(Err(String::from("not built in this test")))
        }

        fn decide(&mut self, _body: &[u8], _model: &str) -> Option<Result<Value, String>> {
            None
        }
    }

    #[test]
    fn one_sequence_at_a_time_keeps_the_serial_path() {
        let mut worker = BatchCapable { asked: false };
        let serial = EngineLimits { max_num_seqs: 1 };
        assert!(super::engine_seed(&mut worker, serial).is_none());
        assert!(!worker.asked, "the serial default never builds an engine");
        let batching = EngineLimits { max_num_seqs: 2 };
        assert!(super::engine_seed(&mut worker, batching).is_some());
        assert!(worker.asked);
    }

    /// Answers decisions with a fixed receipt that names the requested model.
    struct FixedDecider;

    impl ModelWorker for FixedDecider {
        fn chat(&mut self) -> Option<&mut dyn ChatBackend> {
            None
        }

        fn decide(&mut self, body: &[u8], model: &str) -> Option<Result<Value, String>> {
            let request: Value = serde_json::from_slice(body).expect("decision JSON");
            Some(Ok(json!({"model": model, "state": request["state"]})))
        }
    }

    /// Answers embeddings with a fixed response naming the requested model.
    struct FixedEmbedder;

    impl ModelWorker for FixedEmbedder {
        fn chat(&mut self) -> Option<&mut dyn ChatBackend> {
            None
        }

        fn decide(&mut self, _body: &[u8], _model: &str) -> Option<Result<Value, String>> {
            None
        }

        fn embed(&mut self, body: &[u8], model: &str) -> Option<Result<Value, String>> {
            let request: Value = serde_json::from_slice(body).expect("embedding JSON");
            Some(Ok(json!({"model": model, "input": request["input"]})))
        }

        fn rerank(&mut self, body: &[u8], model: &str) -> Option<Result<Value, String>> {
            let request: Value = serde_json::from_slice(body).expect("rerank JSON");
            Some(Ok(json!({"model": model, "query": request["query"]})))
        }
    }

    /// Answers every body route with a response larger than the socket
    /// buffers, so writing it blocks until the client reads.
    struct LargeResponder;

    impl LargeResponder {
        fn large() -> Value {
            json!({"padding": "x".repeat(16 * 1024 * 1024)})
        }
    }

    impl ModelWorker for LargeResponder {
        fn chat(&mut self) -> Option<&mut dyn ChatBackend> {
            None
        }

        fn decide(&mut self, _body: &[u8], _model: &str) -> Option<Result<Value, String>> {
            Some(Ok(Self::large()))
        }

        fn embed(&mut self, _body: &[u8], _model: &str) -> Option<Result<Value, String>> {
            Some(Ok(Self::large()))
        }

        fn rerank(&mut self, _body: &[u8], _model: &str) -> Option<Result<Value, String>> {
            Some(Ok(Self::large()))
        }
    }

    /// Every worker route releases the model before writing its response.
    /// A new route whose worker-loop arm forgets the early release fails here.
    #[test]
    fn a_model_is_free_before_any_body_response_begins() {
        for (path, capability, body) in [
            (
                "/v1/decisions",
                "decide",
                r#"{"model":"m","state":"s","questions":{}}"#,
            ),
            ("/v1/embeddings", "embed", r#"{"model":"m","input":"x"}"#),
            (
                "/v1/rerank",
                "rerank",
                r#"{"model":"m","query":"q","documents":["d"]}"#,
            ),
        ] {
            let listener = TcpListener::bind("127.0.0.1:0").expect("loopback listener");
            let address = listener.local_addr().expect("listener address");
            let (jobs, receiver) = sync_channel(0);
            let worker = thread::spawn(move || model_worker_loop(&mut LargeResponder, receiver));
            let occupied = Arc::new(AtomicBool::new(false));
            let models = [ServedModel {
                id: "m".into(),
                generates: false,
                capabilities: &["decide", "embed", "rerank"],
                jobs,
                occupied: Arc::clone(&occupied),
                engine: None,
                alive: Arc::new(AtomicBool::new(true)),
            }];
            let server = thread::spawn(move || {
                serve_models(
                    &listener,
                    &models,
                    Duration::from_secs(2),
                    TransportLimits::default(),
                    Some(1),
                )
            });
            let mut stream = TcpStream::connect(address).expect("connect");
            write!(
                stream,
                "POST {path} HTTP/1.1\r\nHost: localhost\r\nContent-Length: {}\r\n\r\n{body}",
                body.len()
            )
            .expect("request");
            stream.shutdown(Shutdown::Write).expect("half-close");
            // The worker is now blocked writing the rest. The next request must
            // already be admissible; otherwise a client that sends it as soon as
            // this response completes races the release and is refused as busy.
            let mut first = [0_u8; 1];
            stream.read_exact(&mut first).expect("first response byte");
            assert!(
                !occupied.load(Ordering::Acquire),
                "{capability}: admission held while responding"
            );
            let mut rest = Vec::new();
            stream.read_to_end(&mut rest).expect("response");
            // The worker answered (a refusal before the worker would leave the
            // model free trivially and prove nothing).
            assert!(
                [&first[..], &rest[..]]
                    .concat()
                    .starts_with(b"HTTP/1.1 200 "),
                "{capability}: expected the worker's 200 response"
            );
            server
                .join()
                .expect("join acceptor")
                .expect("acceptor result");
            worker.join().expect("join worker");
        }
    }

    #[test]
    #[allow(
        clippy::too_many_lines,
        reason = "each route, its capability check and its echoed request id"
    )]
    fn embeddings_route_only_to_models_that_embed() {
        const TRACEPARENT: &str = "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01";
        fn post_with(
            address: std::net::SocketAddr,
            path: &str,
            correlation: &str,
            body: &str,
        ) -> (u16, Value) {
            let mut stream = TcpStream::connect(address).expect("connect");
            write!(
                stream,
                "POST {path} HTTP/1.1\r\nHost: localhost\r\nContent-Length: {}\r\n{correlation}\r\n{body}",
                body.len()
            )
            .expect("request");
            stream.shutdown(Shutdown::Write).expect("half-close");
            let (status, body) = fixed_http_response(&mut stream);
            (
                status,
                serde_json::from_slice(&body).expect("response JSON"),
            )
        }
        // Every request names its id; the model's response echoes it.
        fn post(address: std::net::SocketAddr, path: &str, body: &str) -> (u16, Value) {
            post_with(address, path, "x-request-id: req-7\r\n", body)
        }

        let listener = TcpListener::bind("127.0.0.1:0").expect("loopback listener");
        let address = listener.local_addr().expect("listener address");
        let (embed_jobs, embed_receiver) = sync_channel(0);
        let (decide_jobs, decide_receiver) = sync_channel(0);
        let embedder = thread::spawn(move || model_worker_loop(&mut FixedEmbedder, embed_receiver));
        let decider = thread::spawn(move || model_worker_loop(&mut FixedDecider, decide_receiver));
        let model = |id: &str, capabilities, jobs| ServedModel {
            id: id.into(),
            generates: false,
            capabilities,
            jobs,
            occupied: Arc::new(AtomicBool::new(false)),
            alive: Arc::new(AtomicBool::new(true)),
            engine: None,
        };
        let models = [
            model("embedder", &["embed", "rerank"], embed_jobs),
            model("julia", &["decide"], decide_jobs),
        ];
        let server = thread::spawn(move || {
            serve_models(
                &listener,
                &models,
                Duration::from_secs(2),
                TransportLimits::default(),
                Some(6),
            )
        });

        let (status, response) = post(
            address,
            "/v1/embeddings",
            r#"{"model":"embedder","input":"x"}"#,
        );
        assert_eq!(
            (status, response),
            (
                200,
                json!({"model":"embedder","input":"x","metallix":{"request_id":"req-7"}})
            )
        );
        // Without `x-request-id`, the caller's trace id is the request id.
        let (status, response) = post_with(
            address,
            "/v1/decisions",
            &format!("traceparent: {TRACEPARENT}\r\n"),
            r#"{"model":"julia","state":"s","questions":{}}"#,
        );
        assert_eq!(
            (status, &response["metallix"]["request_id"]),
            (200, &json!("4bf92f3577b34da6a3ce929d0e0e4736"))
        );
        let (status, error) = post(
            address,
            "/v1/embeddings",
            r#"{"model":"julia","input":"x"}"#,
        );
        assert_eq!(
            (status, &error["error"]["code"]),
            (400, &json!("unsupported_capability"))
        );
        let (status, error) = post(
            address,
            "/v1/decisions",
            r#"{"model":"embedder","state":"s","questions":{}}"#,
        );
        assert_eq!(
            (status, &error["error"]["code"]),
            (400, &json!("unsupported_capability"))
        );
        let (status, response) = post(
            address,
            "/v1/rerank",
            r#"{"model":"embedder","query":"q","documents":["d"]}"#,
        );
        assert_eq!(
            (status, response),
            (
                200,
                json!({"model":"embedder","query":"q","metallix":{"request_id":"req-7"}})
            )
        );
        let (status, error) = post(
            address,
            "/v1/rerank",
            r#"{"model":"julia","query":"q","documents":["d"]}"#,
        );
        assert_eq!(
            (status, &error["error"]["code"]),
            (400, &json!("unsupported_capability"))
        );

        server
            .join()
            .expect("join acceptor")
            .expect("acceptor result");
        embedder.join().expect("join embedder");
        decider.join().expect("join decider");
    }

    #[test]
    #[allow(
        clippy::too_many_lines,
        reason = "the busy-generation and concurrent-decision lifecycle is the assertion under test"
    )]
    fn decisions_route_by_model_with_per_model_admission() {
        fn exchange(address: std::net::SocketAddr, request: &str, body: &[u8]) -> (u16, Value) {
            let mut stream = TcpStream::connect(address).expect("connect");
            stream.set_read_timeout(Some(HUNG)).expect("bound reads");
            write!(
                stream,
                "{request} HTTP/1.1\r\nHost: localhost\r\nContent-Length: {}\r\n\r\n",
                body.len()
            )
            .expect("request header");
            stream.write_all(body).expect("request body");
            stream.shutdown(Shutdown::Write).expect("half-close");
            let (status, body) = fixed_http_response(&mut stream);
            (
                status,
                serde_json::from_slice(&body).expect("response JSON"),
            )
        }

        let listener = TcpListener::bind("127.0.0.1:0").expect("loopback listener");
        let address = listener.local_addr().expect("listener address");
        let (chat_jobs, chat_receiver) = sync_channel(0);
        let (decide_jobs, decide_receiver) = sync_channel(0);
        let (entered_sender, entered_receiver) = sync_channel(1);
        let (release_sender, release_receiver) = sync_channel(1);
        let chat_worker = thread::spawn(move || {
            let mut backend = BlockingBackend {
                turns: 0,
                entered: entered_sender,
                release: release_receiver,
            };
            worker_loop(&mut backend, chat_receiver);
        });
        let decide_worker = thread::spawn(move || {
            model_worker_loop(&mut FixedDecider, decide_receiver);
        });
        let model = |id: &str, generates, capabilities, jobs| ServedModel {
            id: id.into(),
            generates,
            capabilities,
            jobs,
            occupied: Arc::new(AtomicBool::new(false)),
            alive: Arc::new(AtomicBool::new(true)),
            engine: None,
        };
        let models = [
            model("chat", true, &["generate"], chat_jobs),
            model("julia", false, &["decide"], decide_jobs),
        ];
        let generation_occupied = Arc::clone(&models[0].occupied);
        let server = thread::spawn(move || {
            serve_models(
                &listener,
                &models,
                Duration::from_secs(2),
                TransportLimits::default(),
                Some(7),
            )
        });

        let (status, listed) = exchange(address, "GET /v1/models", b"");
        assert_eq!(status, 200);
        assert_eq!(
            listed["data"],
            json!([
                {"id":"chat","object":"model","owned_by":"local","capabilities":["generate"]},
                {"id":"julia","object":"model","owned_by":"local","capabilities":["decide"]},
            ])
        );

        // Hold the generation model mid-response.
        let generation = thread::spawn(move || {
            let body = br#"{"model":"chat","input":"hello"}"#;
            exchange(address, "POST /v1/responses", body)
        });
        entered_receiver
            .recv_timeout(HUNG)
            .expect("generation holds its worker");
        assert!(generation_occupied.load(Ordering::Acquire));

        let decision = br#"{"model":"julia","state":"s","questions":{}}"#;
        let (status, mut receipt) = exchange(address, "POST /v1/decisions", decision);
        // The request id is minted per request; the rest is the model's answer.
        let metallix = receipt.as_object_mut().and_then(|r| r.remove("metallix"));
        assert!(metallix.is_some_and(|block| block["request_id"].is_string()));
        assert_eq!(
            (status, receipt),
            (200, json!({"model":"julia","state":"s"}))
        );
        let (status, busy) = exchange(
            address,
            "POST /v1/responses",
            br#"{"model":"chat","input":"again"}"#,
        );
        assert_eq!(
            (status, &busy["error"]["code"]),
            (503, &json!("server_busy"))
        );
        let (status, error) = exchange(
            address,
            "POST /v1/decisions",
            br#"{"model":"chat","state":"s","questions":{}}"#,
        );
        assert_eq!(
            (status, &error["error"]["code"]),
            (400, &json!("unsupported_capability"))
        );
        let (status, error) = exchange(
            address,
            "POST /v1/responses",
            br#"{"model":"julia","input":"hello"}"#,
        );
        assert_eq!(
            (status, &error["error"]["code"]),
            (400, &json!("unsupported_capability"))
        );
        let (status, _) = exchange(
            address,
            "POST /v1/decisions",
            br#"{"model":"missing","state":"s","questions":{}}"#,
        );
        assert_eq!(status, 404);

        release_sender.send(()).expect("release generation");
        let (status, completed) = generation.join().expect("generation client");
        assert_eq!((status, &completed["status"]), (200, &json!("completed")));
        server
            .join()
            .expect("join bounded acceptor")
            .expect("acceptor result");
        chat_worker.join().expect("join chat worker");
        decide_worker.join().expect("join decision worker");
    }

    #[test]
    fn generation_routes_answer_in_their_own_protocol() {
        use crate::sse::test_support::Scripted;
        fn post(address: std::net::SocketAddr, path: &str, body: &str) -> (u16, Value) {
            let mut stream = TcpStream::connect(address).expect("connect");
            stream.set_read_timeout(Some(HUNG)).expect("bound reads");
            write!(
                stream,
                "POST {path} HTTP/1.1\r\nHost: localhost\r\nContent-Length: {}\r\n\r\n{body}",
                body.len()
            )
            .expect("request");
            stream.shutdown(Shutdown::Write).expect("half-close");
            let (status, body) = fixed_http_response(&mut stream);
            (
                status,
                serde_json::from_slice(&body).expect("response JSON"),
            )
        }

        let listener = TcpListener::bind("127.0.0.1:0").expect("loopback listener");
        let address = listener.local_addr().expect("listener address");
        let (sender, receiver) = sync_channel(0);
        let worker = thread::spawn(move || {
            let mut backend = Scripted::new("hi");
            worker_loop(&mut backend, receiver);
        });
        let occupied = Arc::new(AtomicBool::new(false));
        let alive = Arc::new(AtomicBool::new(true));
        let server = thread::spawn(move || {
            serve_listener(
                &listener,
                "control",
                &sender,
                &occupied,
                &alive,
                Duration::from_secs(2),
                Some(7),
            )
        });
        let (status, completion) = post(
            address,
            "/v1/chat/completions",
            r#"{"model":"control","messages":[{"role":"user","content":"hello"}]}"#,
        );
        assert_eq!(status, 200, "{completion}");
        assert_eq!(completion["object"], "chat.completion");
        assert_eq!(completion["choices"][0]["message"]["content"], "hi");
        let (status, rejected) = post(
            address,
            "/v1/chat/completions",
            r#"{"model":"control","messages":[],"frobnicate":1}"#,
        );
        assert_eq!(status, 400);
        assert_eq!(rejected["error"]["type"], "invalid_request_error");
        let (status, response) = post(
            address,
            "/v1/responses",
            r#"{"model":"control","input":"hello"}"#,
        );
        assert_eq!((status, &response["object"]), (200, &json!("response")));
        let (status, message) = post(
            address,
            "/v1/messages",
            r#"{"model":"control","max_tokens":8,"messages":[{"role":"user","content":"hello"}]}"#,
        );
        assert_eq!(status, 200, "{message}");
        assert_eq!(message["content"], json!([{"type":"text","text":"hi"}]));
        let (status, rejected) = post(
            address,
            "/v1/messages",
            r#"{"model":"control","messages":[{"role":"user","content":"hello"}]}"#,
        );
        assert_eq!(status, 400);
        assert_eq!(rejected["error"]["type"], "invalid_request_error");
        // The scoring routes reach the backend, which this one refuses.
        for (path, body) in [
            (
                "/v1/completions",
                r#"{"model":"control","prompt":"hello","max_tokens":0,"echo":true}"#,
            ),
            (
                "/v1/score",
                r#"{"model":"control","prompt":"hello","continuation":" there"}"#,
            ),
        ] {
            let (status, rejected) = post(address, path, body);
            assert_eq!(status, 501, "{path}: {rejected}");
            assert_eq!(
                rejected["error"]["message"], "this backend does not support scoring",
                "{path}"
            );
            assert_eq!(rejected["error"]["code"], "scoring_unsupported", "{path}");
        }
        server
            .join()
            .expect("join acceptor")
            .expect("acceptor result");
        worker.join().expect("join worker");
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
                on_token: &mut dyn FnMut(chat_format::TurnDelta) -> Result<(), String>,
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
                on_token: &mut dyn FnMut(chat_format::TurnDelta) -> Result<(), String>,
            ) -> Result<crate::chat_generation::ChatGeneration, ChatGenerationError> {
                let hold_after_delta = request.max_tokens == Some(64);
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
                json!({"error":{"message":"one generation is already active","type":"server_error","param":null,"code":"server_busy"}})
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
}

#[cfg(test)]
mod transcription_tests;
