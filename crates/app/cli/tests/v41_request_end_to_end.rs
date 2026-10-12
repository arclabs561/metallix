//! Real DeepSeek-V4.1 prompts, end to end, against the source captures.
//!
//! Opt-in: `cargo test -p metallix-cli --release --test v41_request_end_to_end -- --ignored --nocapture`.
//! Builds the 40-layer `RequestModel` from checkpoint tensors and prefills a
//! captured prompt (3 tokens, and the 17-token shell prompt) through
//! `RequestSession::prefill_with_sources`. Routed
//! experts, Engram rows and token-embedding rows come from caller sources over
//! a `FetchingSource` (missing ranges fetched from the pinned Hub revision,
//! inside the default envelope). Each layer chains from the native previous
//! layer, so differences compound; the result is reported per layer and the
//! final next-token argmax must match the source.

use std::{
    cell::{Cell, RefCell},
    io,
    num::NonZeroUsize,
    ops::Range,
    path::Path,
    sync::{
        Mutex,
        atomic::{AtomicU64, Ordering},
    },
};

use deepseek::{
    attention::layer::LayerAttentionDiagnostic,
    checkpoint::{
        embedding_rows::V41CachedEmbeddingRows,
        engram_rows::V41CachedEngramRows,
        range_cache::{V41CachedRoutedExperts, V41RangeCache},
    },
    engram::{
        embedding::EngramRowSource,
        inputs::{EngramHashInputs, V41_ENGRAM_INPUTS_IDENTITY},
    },
    moe::{Fp4ExpertWeights, MoEError, RoutedExpertSource},
    reduced::{
        BlockTailDiagnostic, RequestModel, RequestSession, RequestStepOutput,
        ScheduledAttentionOutput, StepSources,
        checkpoint_model::{V41Backend, V41CheckpointWeights, V41Engrams, V41InferenceConfig},
    },
};
use metallix_cli::range_fetch::{CurlHost, Envelope, FetchingSource, RangeHost};

const ENGRAM_WIDTH: usize = 256;

fn le_u16(bytes: &[u8]) -> Vec<u16> {
    bytes
        .chunks_exact(2)
        .map(|pair| u16::from_le_bytes([pair[0], pair[1]]))
        .collect()
}

/// (mismatching elements, max |difference|, cosine) over BF16 values.
fn agreement(native: &[u16], source: &[u16]) -> (usize, f64, f64) {
    assert_eq!(native.len(), source.len());
    let value = |bits: u16| f64::from(f32::from_bits(u32::from(bits) << 16));
    let (mut mismatch, mut max_abs, mut dot, mut nn, mut ss) = (0, 0_f64, 0_f64, 0_f64, 0_f64);
    for (&a, &b) in native.iter().zip(source) {
        let (x, y) = (value(a), value(b));
        mismatch += usize::from(a != b);
        max_abs = max_abs.max((x - y).abs());
        dot += x * y;
        nn += x * x;
        ss += y * y;
    }
    (mismatch, max_abs, dot / (nn.sqrt() * ss.sqrt()))
}

fn argmax(values: &[f32]) -> usize {
    values
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .map(|(index, _)| index)
        .expect("nonempty logits")
}

#[test]
#[ignore = "fetches missing experts and Engram rows from huggingface.co; needs .agents/receipts data and ~10 GB RAM"]
fn one_real_request_step_matches_the_source_capture() {
    run_capture("parity-greedy3.json", "capture-parity3", Check::Capture);
}

/// The 17-token capture-shell2 prompt, prefilled in steps of at most the
/// model's per-step token bound (one step here, since 17 fits).
#[test]
#[ignore = "fetches missing experts and Engram rows from huggingface.co; needs .agents/receipts data and ~10 GB RAM"]
fn real_shell_prompt_matches_the_source_greedy_token() {
    run_capture("parity-shell.json", "capture-shell2", Check::Capture);
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Check {
    Capture,
    StateEquivalence { tokens: usize, prefix: usize },
    PrefixDiagnostic,
}

/// Localizes schedule divergence; it does not qualify the full request.
#[test]
#[ignore = "needs an owned METALLIX_V41_STATE_RECEIPTS projection and external footprint/time watchdog; no network"]
fn cached_three_token_prefix_localizes_schedule_divergence() {
    if std::env::var_os("METALLIX_V41_STATE_RECEIPTS").is_none() {
        eprintln!("skipping: METALLIX_V41_STATE_RECEIPTS is not set");
        return;
    }
    assert!(matches!(backend(), V41Backend::Scalar), "scalar diagnostic");
    run_capture(
        "parity-shell.json",
        "capture-shell2",
        Check::PrefixDiagnostic,
    );
}

#[test]
#[ignore = "needs METALLIX_V41_STATE_RECEIPTS with an owned cache copy and an external footprint/time watchdog; no network"]
fn cached_shell_prefill_and_decode_have_identical_final_state() {
    if std::env::var_os("METALLIX_V41_STATE_RECEIPTS").is_none() {
        eprintln!("skipping: METALLIX_V41_STATE_RECEIPTS is not set");
        return;
    }
    assert!(
        matches!(backend(), V41Backend::Scalar),
        "state gate is scalar only"
    );
    run_capture(
        "parity-shell.json",
        "capture-shell2",
        Check::StateEquivalence {
            tokens: 17,
            prefix: 3,
        },
    );
}

#[test]
#[ignore = "needs METALLIX_V41_STATE_RECEIPTS with an owned cache copy and an external footprint/time watchdog; no network"]
fn cached_greedy_prefill_and_decode_have_identical_final_state() {
    if std::env::var_os("METALLIX_V41_STATE_RECEIPTS").is_none() {
        eprintln!("skipping: METALLIX_V41_STATE_RECEIPTS is not set");
        return;
    }
    assert!(
        matches!(backend(), V41Backend::Scalar),
        "state gate is scalar only"
    );
    run_capture(
        "parity-greedy3.json",
        "capture-parity3",
        Check::StateEquivalence {
            tokens: 3,
            prefix: 2,
        },
    );
}

/// `METALLIX_V41_BACKEND=metal` selects the Metal Engram WKV and final head
/// (with the `metal` feature); anything else, or unset, is the scalar path.
fn backend() -> V41Backend {
    match std::env::var("METALLIX_V41_BACKEND").as_deref() {
        #[cfg(feature = "metal")]
        Ok("metal") => V41Backend::Metal,
        Ok("scalar") | Err(_) => V41Backend::Scalar,
        Ok(other) => panic!("METALLIX_V41_BACKEND={other} is not available in this build"),
    }
}

/// Largest fetch [`Host`] allows with `METALLIX_V41_NO_EXPERT_FETCH=1`.
/// Engram and embedding row runs are at most a few KB; a routed expert
/// tensor is megabytes.
const ROW_FETCH_LIMIT: u64 = 1 << 20;

/// [`CurlHost`] that can refuse fetches by size, counting what it fetches.
/// `METALLIX_V41_NO_EXPERT_FETCH=1` refuses any fetch larger than a row run,
/// so a run that routes to an expert not stored yet (as a different
/// backend's numerics might) fails instead of downloading it.
/// `METALLIX_V41_NO_ROW_FETCH=1` refuses row runs, which a repeat run reads
/// from the row packs. With both set, the run makes no network request.
struct Host {
    no_experts: bool,
    no_rows: bool,
    fetches: AtomicU64,
    bytes: AtomicU64,
}

impl Host {
    fn from_env() -> Self {
        let set = |name| std::env::var(name).as_deref() == Ok("1");
        Self {
            no_experts: set("METALLIX_V41_NO_EXPERT_FETCH"),
            no_rows: set("METALLIX_V41_NO_ROW_FETCH"),
            fetches: AtomicU64::new(0),
            bytes: AtomicU64::new(0),
        }
    }
}

impl RangeHost for &Host {
    fn get_range(&self, url: &str, range: Range<u64>) -> io::Result<(Vec<u8>, String)> {
        let length = range.end.saturating_sub(range.start);
        let refused = if length > ROW_FETCH_LIMIT {
            self.no_experts.then_some("METALLIX_V41_NO_EXPERT_FETCH")
        } else {
            self.no_rows.then_some("METALLIX_V41_NO_ROW_FETCH")
        };
        if let Some(flag) = refused {
            return Err(io::Error::other(format!(
                "{flag}: refusing {url} bytes {range:?}"
            )));
        }
        self.fetches.fetch_add(1, Ordering::Relaxed);
        self.bytes.fetch_add(length, Ordering::Relaxed);
        CurlHost.get_range(url, range)
    }

    fn free_bytes(&self, path: &Path) -> Option<u64> {
        CurlHost.free_bytes(path)
    }
}

/// Prefills the recorded run's prompt through `prefill_with_sources`, reports
/// per-layer agreement with `capture`, and requires the source's greedy token.
#[allow(
    clippy::too_many_lines,
    reason = "one end-to-end request keeps every compared stage visible"
)]
fn run_capture(run_file: &str, capture: &str, check: Check) {
    let root = if check == Check::Capture {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../.agents/receipts")
    } else {
        let root = std::path::PathBuf::from(
            std::env::var_os("METALLIX_V41_STATE_RECEIPTS")
                .expect("explicit owned receipts projection is required"),
        );
        let repository = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../..")
            .canonicalize()
            .expect("worktree repository");
        let projection_base = repository.join(".agents/projections/deepseek-state");
        assert_no_symlink_components(&projection_base);
        assert_no_symlink_components(&root);
        let root = root.canonicalize().expect("existing receipts projection");
        assert!(
            root.starts_with(projection_base.canonicalize().unwrap()),
            "receipts must remain under this worktree's .agents/projections/deepseek-state"
        );
        // This marker records intent only. Private-copy provenance and a tested
        // external resource watchdog remain separate execution prerequisites.
        let marker = root.join("state-equivalence-owned-cache");
        assert_no_symlink_components(&marker);
        assert!(marker.is_file(), "owned projection marker required");
        let weights = root.join("route-trace/weights");
        assert_no_symlink_components(&weights);
        assert!(
            !weights.join(".rows").exists(),
            "legacy migration is outside this gate"
        );
        assert_unlinked_files(&weights);
        root
    };
    let trace = root.join("route-trace");
    let run: serde_json::Value =
        serde_json::from_slice(&std::fs::read(trace.join(run_file)).expect("recorded run"))
            .expect("recorded run JSON");
    let ids: Vec<i64> = run["runs"][0]["prompt_ids"]
        .as_array()
        .expect("prompt ids")
        .iter()
        .map(|id| id.as_i64().expect("id"))
        .collect();
    let greedy = run["runs"][0]["generated_ids"][0]
        .as_u64()
        .expect("recorded greedy token");

    if let Check::StateEquivalence { tokens, prefix } = check {
        assert_eq!(ids.len(), tokens, "fixed cached state workload");
        assert!(prefix >= 2 && prefix < tokens, "nonempty decode tail");
    }

    let registry: serde_json::Value =
        serde_json::from_str(include_str!("../../../../config/artifacts/deepseek.json"))
            .expect("embedded registry");
    let (repo, revision) = (
        registry["model_repo"].as_str().expect("repo").to_owned(),
        registry["model_revision"]
            .as_str()
            .expect("revision")
            .to_owned(),
    );
    let config = V41InferenceConfig::parse(
        &std::fs::read_to_string(trace.join("inference-config.json")).expect("config"),
    )
    .expect("pinned inference config");
    let mut host = Host::from_env();
    if check != Check::Capture {
        host.no_experts = true;
        host.no_rows = true;
    }
    let source = FetchingSource::new(
        trace.join("weights"),
        &trace,
        repo,
        revision.clone(),
        &host,
        Envelope::default(),
    )
    .expect("fetching source");
    let mut cache = V41RangeCache::load(
        source,
        &root.join("control/receipts/candidate-control/real-expert/model.safetensors.index.json"),
        &trace,
        &revision,
        2 << 30,
    )
    .expect("pinned cache");
    let mut weights = V41CheckpointWeights::load(
        &mut cache,
        &config,
        0..config.layers(),
        NonZeroUsize::new(128).expect("nonzero"),
    )
    .expect("non-expert layers");
    weights.load_head(&mut cache).expect("BF16 head");
    let backend = backend();
    weights.set_backend(backend).expect("backend");
    let inputs = EngramHashInputs::parse(
        &std::fs::read(root.join("engram-hash/v41-engram-inputs.bin")).expect("Engram inputs"),
        &V41_ENGRAM_INPUTS_IDENTITY,
    )
    .expect("pinned Engram inputs");
    let model = weights
        .request_model(V41Engrams::Definitions(
            weights
                .engram_definitions(&inputs)
                .expect("Engram definitions"),
        ))
        .expect("40-layer request model");

    let cache = Mutex::new(cache);
    let experts: Vec<_> = (0..config.layers())
        .map(|layer| {
            V41CachedRoutedExperts::new(&cache, layer, config.width(), config.intermediate_width())
        })
        .collect();
    let expert_refs: Vec<Option<&dyn RoutedExpertSource>> = experts
        .iter()
        .map(|source| Some(source as &dyn RoutedExpertSource))
        .collect();
    let engram_rows = [1, 14].map(|layer| V41CachedEngramRows::new(&cache, layer, ENGRAM_WIDTH));
    let row_refs: Vec<Option<&dyn EngramRowSource>> = engram_rows
        .iter()
        .map(|source| Some(source as &dyn EngramRowSource))
        .collect();
    let embedding = V41CachedEmbeddingRows::new(&cache, config.width());
    if check == Check::PrefixDiagnostic {
        assert_eq!(ids.len(), 17, "fixed shell fixture");
        prefix_diagnostic(
            &model,
            &ids[..3],
            StepSources {
                experts: &expert_refs,
                engram_rows: &row_refs,
                embedding_rows: Some(&embedding),
            },
        );
        assert_eq!(host.fetches.load(Ordering::Relaxed), 0, "no HTTP requests");
        assert_eq!(host.bytes.load(Ordering::Relaxed), 0, "no fetched bytes");
        return;
    }
    let mut session = RequestSession::new(&model).expect("request session");
    let started = std::time::Instant::now();
    let output = session
        .prefill_with_sources(
            &ids,
            StepSources {
                experts: &expert_refs,
                engram_rows: &row_refs,
                embedding_rows: Some(&embedding),
            },
        )
        .expect("real request prefill");
    eprintln!(
        "{backend:?} prefill of {} tokens: {:.1} s",
        ids.len(),
        started.elapsed().as_secs_f64()
    );
    assert_eq!(session.next_start(), ids.len());
    eprintln!(
        "network: {} fetches, {} bytes",
        host.fetches.load(Ordering::Relaxed),
        host.bytes.load(Ordering::Relaxed)
    );
    #[cfg(feature = "metal")]
    if let Some(counts) = weights.device_counts() {
        eprintln!("device {counts:?}");
        assert!(
            counts.resident_fp8 > 0 && counts.fp4_experts > 0 && counts.scalar_fallbacks == 0,
            "the Metal backend ran its linears on the device: {counts:?}"
        );
    }

    let capture = trace.join(capture);
    let read = |name: String| std::fs::read(capture.join(name)).expect("captured tensor");
    let layer_out = |layer: usize| le_u16(&read(format!("layer{layer:02}.out.torch.bfloat16.bin")));
    eprintln!(
        "layer 00 out {:?}",
        agreement(output.startup().residual(), &layer_out(0))
    );
    for (index, layer) in output.layers().iter().enumerate() {
        let number = index + 1;
        if let Some(engram) = layer.engram() {
            eprintln!(
                "layer {number:02} engram {:?}",
                agreement(
                    engram.output(),
                    &le_u16(&read(format!("layer{number:02}.in.torch.bfloat16.bin")))
                )
            );
        }
        let out: Vec<u16> = layer
            .tails()
            .iter()
            .flat_map(|tail| tail.ffn().output_bf16().iter().copied())
            .collect();
        eprintln!(
            "layer {number:02} out {:?}",
            agreement(&out, &layer_out(number))
        );
    }

    let heads = output.heads();
    assert_eq!(
        heads.len(),
        1,
        "the real model computes the last position only"
    );
    let logits = heads[0].logits();
    let source: Vec<f32> = read("logits.torch.float32.bin".to_owned())
        .chunks_exact(4)
        .map(|word| f32::from_le_bytes([word[0], word[1], word[2], word[3]]))
        .collect();
    assert_eq!(logits.len(), source.len());
    let max_abs = logits
        .iter()
        .zip(&source)
        .map(|(a, b)| (a - b).abs())
        .fold(0_f32, f32::max);
    let (native_top, source_top) = (argmax(logits), argmax(&source));
    eprintln!(
        "logits max_abs {max_abs} native argmax {native_top} ({}) source argmax {source_top} ({})",
        logits[native_top], source[source_top]
    );
    assert_eq!(
        source_top as u64, greedy,
        "the capture's argmax is the recorded token"
    );
    assert_eq!(native_top, source_top, "next-token argmax");
    if let Check::StateEquivalence { tokens, prefix } = check {
        let expected_cursor = session.next_start();
        let expected_logits: Vec<u32> = logits.iter().map(|value| value.to_bits()).collect();
        let width = output.residual().len() / ids.len();
        let expected_residual = output.residual()[output.residual().len() - width..].to_vec();
        drop(output);
        drop(session);
        // Reuse the one loaded weight set. Only the first prefix is batched;
        // every subsequent append is one token, as the current API requires.
        let sources = StepSources {
            experts: &expert_refs,
            engram_rows: &row_refs,
            embedding_rows: Some(&embedding),
        };
        let mut split = RequestSession::new(&model).expect("split request session");
        let mut actual = split
            .step_with_sources(&ids[..prefix], sources)
            .expect("initial prefix");
        for id in &ids[prefix..] {
            actual = split
                .step_with_sources(std::slice::from_ref(id), sources)
                .expect("one-token append");
        }
        assert_eq!(split.next_start(), expected_cursor, "request cursor");
        assert_eq!(actual.residual(), expected_residual, "last residual");
        let actual_logits: Vec<u32> = actual
            .heads()
            .last()
            .unwrap()
            .logits()
            .iter()
            .map(|value| value.to_bits())
            .collect();
        assert_eq!(actual_logits, expected_logits, "final logit bits");
        assert_eq!(host.fetches.load(Ordering::Relaxed), 0, "no HTTP requests");
        assert_eq!(host.bytes.load(Ordering::Relaxed), 0, "no fetched bytes");
        eprintln!(
            "scalar state equivalence: {tokens} tokens, prefix {prefix} then {} single-token appends, {:.1} s including baseline",
            tokens - prefix,
            started.elapsed().as_secs_f64()
        );
    }
}

struct ObservedExperts<'a> {
    source: &'a dyn RoutedExpertSource,
    phase: &'static str,
    layer: usize,
    expected: Option<Vec<(usize, usize)>>,
    expected_outputs: Option<Vec<Vec<u16>>>,
    calls: Cell<usize>,
    observed: RefCell<Vec<usize>>,
}

impl RoutedExpertSource for ObservedExperts<'_> {
    fn with_expert(
        &self,
        index: usize,
        run: &mut dyn FnMut(Fp4ExpertWeights<'_>) -> Result<Vec<u16>, MoEError>,
    ) -> Result<Vec<u16>, MoEError> {
        let call = self.calls.get();
        self.calls.set(call + 1);
        self.observed.borrow_mut().push(index);
        let expected = self.expected.as_ref().and_then(|routes| routes.get(call));
        eprintln!(
            "route phase={} layer={} call={call} expected_token_expert={expected:?} actual_expert={index}",
            self.phase, self.layer
        );
        if self.expected.is_some() && expected.is_none_or(|&(_, expert)| expert != index) {
            return Err(MoEError::ExpertUnavailable {
                index,
                reason: format!(
                    "diagnostic route divergence before delegation: phase={} layer={} call={call} expected_token_expert={expected:?} actual={index}",
                    self.phase, self.layer
                ),
            });
        }
        let mut invoked = false;
        self.source.with_expert(index, &mut |weights| {
            assert!(!invoked, "source callback must run exactly once");
            invoked = true;
            let output = run(weights)?;
            if let Some(outputs) = &self.expected_outputs {
                let baseline = &outputs[call];
                let mismatch = baseline.iter().zip(&output).position(|(a, b)| a != b);
                if baseline.len() != output.len() || mismatch.is_some() {
                    return Err(MoEError::ExpertUnavailable {
                        index,
                        reason: format!(
                            "FIRST_DIFFERENCE stage=weighted_expert_output phase={} layer={} expected_token_expert={expected:?} first_element={mismatch:?} expected_len={} actual_len={}",
                            self.phase, self.layer, baseline.len(), output.len()
                        ),
                    });
                }
            }
            Ok(output)
        })
    }
}

fn layer_tails(output: &RequestStepOutput, layer: usize) -> &[BlockTailDiagnostic] {
    if layer == 0 {
        output.startup().tails()
    } else {
        output.layers()[layer - 1].tails()
    }
}

fn observed_step(
    session: &mut RequestSession<'_>,
    ids: &[i64],
    sources: StepSources<'_>,
    phase: &'static str,
    baseline: Option<&RequestStepOutput>,
) -> RequestStepOutput {
    let start = session.next_start();
    let observers: Vec<_> = sources
        .experts
        .iter()
        .enumerate()
        .map(|(layer, source)| ObservedExperts {
            source: source.expect("real expert source for every layer"),
            phase,
            layer,
            expected: baseline.map(|output| {
                layer_tails(output, layer)
                    .iter()
                    .enumerate()
                    .skip(start)
                    .take(ids.len())
                    .flat_map(|(token, tail)| {
                        tail.ffn()
                            .moe()
                            .routes()
                            .iter()
                            .map(move |route| (token, route.expert_index()))
                    })
                    .collect()
            }),
            // run_routes retains the exact returned expert vector in route order.
            expected_outputs: baseline.map(|output| {
                layer_tails(output, layer)
                    .iter()
                    .skip(start)
                    .take(ids.len())
                    .flat_map(|tail| tail.ffn().moe().selected_outputs_bf16().iter().cloned())
                    .collect()
            }),
            calls: Cell::new(0),
            observed: RefCell::new(Vec::new()),
        })
        .collect();
    let refs: Vec<Option<&dyn RoutedExpertSource>> = observers
        .iter()
        .map(|observer| Some(observer as &dyn RoutedExpertSource))
        .collect();
    let output = session
        .step_with_sources(
            ids,
            StepSources {
                experts: &refs,
                ..sources
            },
        )
        .unwrap_or_else(|error| panic!("diagnostic phase={phase} token_start={start}: {error:?}"));
    for observer in &observers {
        let routes = layer_tails(&output, observer.layer);
        let count: usize = routes
            .iter()
            .map(|tail| tail.ffn().moe().routes().len())
            .sum();
        assert_eq!(observer.calls.get(), count, "one delegated call per route");
        let returned_routes: Vec<_> = routes
            .iter()
            .flat_map(|tail| {
                tail.ffn()
                    .moe()
                    .routes()
                    .iter()
                    .map(|route| route.expert_index())
            })
            .collect();
        assert_eq!(
            *observer.observed.borrow(),
            returned_routes,
            "source call order matches returned route order"
        );
        if let Some(expected) = &observer.expected {
            assert_eq!(count, expected.len(), "complete expected route sequence");
        }
        for (position, tail) in routes.iter().enumerate() {
            let token = start + position;
            for route in tail.ffn().moe().routes() {
                eprintln!(
                    "resolved route phase={phase} layer={} token={token} expert={} weight_bits={}",
                    observer.layer,
                    route.expert_index(),
                    route.weight().to_bits()
                );
            }
        }
    }
    output
}

fn prefix_diagnostic(model: &RequestModel<'_>, ids: &[i64], sources: StepSources<'_>) {
    let mut baseline_session = RequestSession::new(model).expect("baseline session");
    let baseline = observed_step(&mut baseline_session, ids, sources, "prefill3", None);
    drop(baseline_session);
    let mut split = RequestSession::new(model).expect("split session");
    let initial = observed_step(&mut split, &ids[..2], sources, "prefill2", Some(&baseline));
    compare_prefix(&baseline, &initial, 0, 2);
    drop(initial);
    let appended = observed_step(&mut split, &ids[2..], sources, "append2", Some(&baseline));
    compare_prefix(&baseline, &appended, 2, 1);
    assert_eq!(split.next_start(), 3, "diagnostic cursor");
    let expected: Vec<_> = baseline
        .heads()
        .last()
        .unwrap()
        .logits()
        .iter()
        .map(|v| v.to_bits())
        .collect();
    let actual: Vec<_> = appended
        .heads()
        .last()
        .unwrap()
        .logits()
        .iter()
        .map(|v| v.to_bits())
        .collect();
    assert_eq!(expected, actual, "three-token final logit bits");
    eprintln!("three-token boundaries match; this does not qualify the 17-token request");
}

struct Boundary {
    layer: usize,
    token: usize,
    actual_position: usize,
    actual_rows: usize,
}

impl Boundary {
    fn compare<T: PartialEq + std::fmt::Debug>(&self, stage: &str, expected: &[T], actual: &[T]) {
        assert_eq!(
            expected.len(),
            actual.len(),
            "layer={} token={} stage={stage} width",
            self.layer,
            self.token
        );
        if let Some(index) = expected.iter().zip(actual).position(|(a, b)| a != b) {
            panic!(
                "first numerical boundary: layer={} token={} stage={stage} element={index} expected={:?} actual={:?}",
                self.layer, self.token, expected[index], actual[index]
            );
        }
    }

    fn rows(&self, stage: &str, expected: &[u16], actual: &[u16]) {
        self.compare(
            stage,
            row(expected, 3, self.token),
            row(actual, self.actual_rows, self.actual_position),
        );
    }
}

fn row<T>(values: &[T], rows: usize, position: usize) -> &[T] {
    assert_eq!(values.len() % rows, 0, "token row geometry");
    let width = values.len() / rows;
    &values[position * width..(position + 1) * width]
}

fn attention_diagnostic(output: &ScheduledAttentionOutput) -> &LayerAttentionDiagnostic {
    match output {
        ScheduledAttentionOutput::WindowOnly(attention)
        | ScheduledAttentionOutput::RatioTwoConsumer(attention)
        | ScheduledAttentionOutput::RatioOneConsumer(attention) => attention,
        ScheduledAttentionOutput::RatioTwoOwner(layer) => layer.attention(),
        ScheduledAttentionOutput::RatioOneOwner(layer) => layer.attention(),
        ScheduledAttentionOutput::RatioOneIndexer(layer) => layer.attention(),
        _ => panic!("unsupported diagnostic attention kind"),
    }
}

fn compare_attention(
    boundary: &Boundary,
    expected: &LayerAttentionDiagnostic,
    actual: &LayerAttentionDiagnostic,
) {
    for (name, a, b) in [
        ("wq_a", &expected.wq_a, &actual.wq_a),
        ("qr", &expected.qr, &actual.qr),
        ("q_after_rope", &expected.q_after_rope, &actual.q_after_rope),
        (
            "prepared_window",
            &expected.prepared_window,
            &actual.prepared_window,
        ),
        (
            "sparse_attention",
            &expected.sparse_output,
            &actual.sparse_output,
        ),
        (
            "attention_final",
            &expected.final_output,
            &actual.final_output,
        ),
    ] {
        boundary.rows(name, a, b);
    }
    // At token positions 0..3 no window ring wraps. Ignore unreachable padding;
    // compare logical window token positions, never the future-filled ring.
    let a: Vec<_> = row(&expected.window_indices, 3, boundary.token)
        .iter()
        .copied()
        .filter(|&i| i >= 0)
        .collect();
    let b: Vec<_> = row(
        &actual.window_indices,
        boundary.actual_rows,
        boundary.actual_position,
    )
    .iter()
    .copied()
    .filter(|&i| i >= 0)
    .collect();
    boundary.compare("window_token_indices", &a, &b);
}

fn selected_indices(output: &ScheduledAttentionOutput) -> Option<&[i32]> {
    match output {
        ScheduledAttentionOutput::RatioTwoOwner(layer) => Some(layer.selected_indices()),
        ScheduledAttentionOutput::RatioOneOwner(layer) => Some(layer.selected_indices()),
        ScheduledAttentionOutput::RatioOneIndexer(layer) => Some(&layer.selection().indices),
        _ => None,
    }
}

fn compare_compressed_indices(
    boundary: &Boundary,
    expected: &ScheduledAttentionOutput,
    actual: &ScheduledAttentionOutput,
) {
    if let (Some(a), Some(b)) = (selected_indices(expected), selected_indices(actual)) {
        let expected_attention = attention_diagnostic(expected);
        let actual_attention = attention_diagnostic(actual);
        let key_width = expected_attention.prepared_window.len() / 3;
        let expected_offset =
            i32::try_from(expected_attention.window_read.len() / key_width).unwrap();
        let actual_offset = i32::try_from(actual_attention.window_read.len() / key_width).unwrap();
        let a: Vec<_> = row(a, 3, boundary.token)
            .iter()
            .copied()
            .filter(|&i| i >= 0)
            .map(|i| i - expected_offset)
            .collect();
        let b: Vec<_> = row(b, boundary.actual_rows, boundary.actual_position)
            .iter()
            .copied()
            .filter(|&i| i >= 0)
            .map(|i| i - actual_offset)
            .collect();
        boundary.compare("compressed_logical_indices", &a, &b);
    }
}

fn compare_prefix(
    expected: &RequestStepOutput,
    actual: &RequestStepOutput,
    start: usize,
    positions: usize,
) {
    assert_eq!(expected.layers().len(), actual.layers().len());
    for layer in 0..=expected.layers().len() {
        for position in 0..positions {
            let boundary = Boundary {
                layer,
                token: start + position,
                actual_position: position,
                actual_rows: positions,
            };
            if layer == 0 {
                boundary.rows(
                    "normalized_attention_input",
                    expected.startup().attention_input(),
                    actual.startup().attention_input(),
                );
                compare_attention(
                    &boundary,
                    expected.startup().attention(),
                    actual.startup().attention(),
                );
            } else {
                let a = &expected.layers()[layer - 1];
                let b = &actual.layers()[layer - 1];
                if let (Some(a), Some(b)) = (a.engram(), b.engram()) {
                    boundary.rows("engram_output", a.output(), b.output());
                }
                boundary.compare(
                    "collapsed_attention_input",
                    a.attention_input()[boundary.token].collapsed_bf16(),
                    b.attention_input()[position].collapsed_bf16(),
                );
                boundary.compare(
                    "normalized_attention_input",
                    a.attention_input()[boundary.token].normalized_bf16(),
                    b.attention_input()[position].normalized_bf16(),
                );
                compare_attention(
                    &boundary,
                    attention_diagnostic(a.attention()),
                    attention_diagnostic(b.attention()),
                );
                compare_compressed_indices(&boundary, a.attention(), b.attention());
            }
            let a = &layer_tails(expected, layer)[boundary.token];
            let b = &layer_tails(actual, layer)[position];
            boundary.compare(
                "after_attention",
                a.after_attention_bf16(),
                b.after_attention_bf16(),
            );
            boundary.compare(
                "ffn_collapsed",
                a.ffn().collapsed_bf16(),
                b.ffn().collapsed_bf16(),
            );
            boundary.compare(
                "ffn_normalized",
                a.ffn().normalized_bf16(),
                b.ffn().normalized_bf16(),
            );
            let routes = |tail: &BlockTailDiagnostic| {
                tail.ffn()
                    .moe()
                    .routes()
                    .iter()
                    .map(|r| (r.expert_index(), r.weight().to_bits()))
                    .collect::<Vec<_>>()
            };
            boundary.compare("routes", &routes(a), &routes(b));
            boundary.compare("ffn_output", a.ffn().output_bf16(), b.ffn().output_bf16());
        }
    }
    let boundary = Boundary {
        layer: expected.layers().len(),
        token: start,
        actual_position: 0,
        actual_rows: positions,
    };
    for position in 0..positions {
        Boundary {
            token: start + position,
            actual_position: position,
            ..boundary
        }
        .rows("final_residual", expected.residual(), actual.residual());
    }
}

// Reject lexical traversal and symlinks before canonicalization can hide them.
fn assert_no_symlink_components(path: &Path) {
    use std::path::Component;
    assert!(path.is_absolute(), "projection paths must be absolute");
    let mut prefix = std::path::PathBuf::new();
    for component in path.components() {
        assert!(
            matches!(component, Component::RootDir | Component::Normal(_)),
            "projection paths must not contain traversal components"
        );
        prefix.push(component.as_os_str());
        assert!(
            !std::fs::symlink_metadata(&prefix)
                .expect("existing projection path component")
                .file_type()
                .is_symlink(),
            "projection path contains a symlink"
        );
    }
}

// Cache recovery may rewrite row packs. These checks reject shared links;
// they do not establish private-copy provenance or prevent concurrent mutation.
#[cfg(unix)]
fn assert_unlinked_files(root: &Path) {
    use std::os::unix::fs::MetadataExt;
    for entry in std::fs::read_dir(root).unwrap() {
        let path = entry.unwrap().path();
        let metadata = std::fs::symlink_metadata(&path).unwrap();
        assert!(
            !metadata.file_type().is_symlink(),
            "cache projection contains a symlink"
        );
        if metadata.is_dir() {
            assert_unlinked_files(&path);
        } else {
            assert!(metadata.is_file());
            assert_eq!(metadata.nlink(), 1, "cache projection contains a hard link");
        }
    }
}

#[cfg(not(unix))]
fn assert_unlinked_files(_root: &Path) {
    panic!("state gate requires Unix file link-count checks");
}
