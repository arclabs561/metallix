//! Real DeepSeek-V4.1 prompts, end to end, against the source captures.
//!
//! Opt-in: `cargo test -p server --release --test v41_request_end_to_end -- --ignored --nocapture`.
//! Builds the 40-layer `RequestModel` from checkpoint tensors and prefills a
//! captured prompt (3 tokens, and the 17-token shell prompt) through
//! `RequestSession::prefill_with_sources`. Routed
//! experts, Engram rows and token-embedding rows come from caller sources over
//! a `FetchingSource` (missing ranges fetched from the pinned Hub revision,
//! inside the default envelope). Each layer chains from the native previous
//! layer, so differences compound; the result is reported per layer and the
//! final next-token argmax must match the source.

use std::{num::NonZeroUsize, path::Path, sync::Mutex};

use deepseek::{
    checkpoint::{
        embedding_rows::V41CachedEmbeddingRows,
        engram_rows::V41CachedEngramRows,
        range_cache::{V41CachedRoutedExperts, V41RangeCache},
    },
    engram::{
        embedding::EngramRowSource,
        inputs::{EngramHashInputs, V41_ENGRAM_INPUTS_IDENTITY},
    },
    moe::RoutedExpertSource,
    reduced::{
        RequestSession, StepSources,
        checkpoint_model::{V41CheckpointWeights, V41Engrams, V41InferenceConfig},
    },
};
use server::range_fetch::{CurlHost, Envelope, FetchingSource};

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
    run_capture("parity-greedy3.json", "capture-parity3");
}

/// The 17-token capture-shell2 prompt, prefilled in steps of at most the
/// model's per-step token bound (one step here, since 17 fits).
#[test]
#[ignore = "fetches missing experts and Engram rows from huggingface.co; needs .agents/receipts data and ~10 GB RAM"]
fn real_shell_prompt_matches_the_source_greedy_token() {
    run_capture("parity-shell.json", "capture-shell2");
}

/// Prefills the recorded run's prompt through `prefill_with_sources`, reports
/// per-layer agreement with `capture`, and requires the source's greedy token.
#[allow(
    clippy::too_many_lines,
    reason = "one end-to-end request keeps every compared stage visible"
)]
fn run_capture(run_file: &str, capture: &str) {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../.agents/receipts");
    let trace = root.join("route-trace");
    let registry: serde_json::Value =
        serde_json::from_str(include_str!("../../../config/artifacts/deepseek.json"))
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
    let source = FetchingSource::new(
        trace.join("weights"),
        &trace,
        repo,
        revision.clone(),
        CurlHost,
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
    let mut session = RequestSession::new(&model).expect("request session");
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
    assert_eq!(session.next_start(), ids.len());

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
}
