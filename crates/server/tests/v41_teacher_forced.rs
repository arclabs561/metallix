//! Teacher-forced real DeepSeek-V4.1 layers with missing experts fetched over the network.
//!
//! Opt-in: `cargo test -p server --release --test v41_teacher_forced -- --ignored --nocapture`.
//! Needs the local `.agents/receipts` route-trace metadata and capture-parity3. Routed
//! experts absent from the local weights store are fetched from the pinned Hub revision
//! into that store, inside the default envelope (64 GiB stored, 150 GiB free).

use std::{num::NonZeroUsize, path::Path, sync::Mutex};

use deepseek::{
    checkpoint::range_cache::{V41CachedRoutedExperts, V41RangeCache},
    reduced::checkpoint_model::{V41CheckpointWeights, V41InferenceConfig, V41TeacherState},
};
use server::range_fetch::{CurlHost, Envelope, FetchingSource};

const TOKENS: usize = 3;

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

#[test]
#[ignore = "fetches missing routed experts from huggingface.co; needs .agents/receipts route-trace data"]
fn every_real_layer_runs_teacher_forced_with_fetched_experts() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../.agents/receipts");
    let trace = root.join("route-trace");
    let registry = server_registry();
    let config = V41InferenceConfig::parse(
        &std::fs::read_to_string(trace.join("inference-config.json")).expect("config"),
    )
    .expect("pinned inference config");
    let layers = std::env::var("METALLIX_V41_TEACHER_LAYERS")
        .map_or(config.layers(), |value| value.parse().expect("layer count"));
    let source = FetchingSource::new(
        trace.join("weights"),
        &trace,
        registry.0,
        registry.1.clone(),
        CurlHost,
        Envelope::default(),
    )
    .expect("fetching source");
    let mut cache = V41RangeCache::load(
        source,
        &root.join("control/receipts/candidate-control/real-expert/model.safetensors.index.json"),
        &trace,
        &registry.1,
        256 << 20,
    )
    .expect("pinned cache");
    let weights = V41CheckpointWeights::load(
        &mut cache,
        &config,
        0..layers,
        NonZeroUsize::new(128).expect("nonzero"),
    )
    .expect("non-expert layers");
    let cache = Mutex::new(cache);
    let capture = trace.join("capture-parity3");
    let read = |layer: usize, kind: &str, ty: &str| {
        std::fs::read(capture.join(format!("layer{layer:02}.{kind}.torch.{ty}.bin")))
            .expect("captured tensor")
    };
    let mut state = V41TeacherState::default();
    for layer in 0..layers {
        let residual = le_u16(&read(layer, "in", "bfloat16"));
        let pre: Vec<f32> = read(layer, "premix_in", "float32")
            .chunks_exact(4)
            .map(|word| f32::from_le_bytes([word[0], word[1], word[2], word[3]]))
            .collect();
        assert_eq!(residual.len() % (TOKENS * config.width()), 0);
        let experts =
            V41CachedRoutedExperts::new(&cache, layer, config.width(), config.intermediate_width());
        let layer_trace = weights
            .teacher_forced_layer(layer, &residual, &pre, &mut state, &experts)
            .unwrap_or_else(|error| panic!("layer {layer}: {error}"));
        let attn_in = agreement(
            &layer_trace.attention_input,
            &le_u16(&read(layer, "attn_in", "bfloat16")),
        );
        let attn_out = agreement(
            &layer_trace.attention_output,
            &le_u16(&read(layer, "attn_out", "bfloat16")),
        );
        assert_eq!(
            attn_in.0, 0,
            "layer {layer}: attention input must match exactly"
        );
        let tail = layer_trace
            .tail
            .unwrap_or_else(|reason| panic!("layer {layer}: tail did not run: {reason}"));
        eprintln!(
            "layer {layer:02} attn_out {attn_out:?} ffn_in {:?} ffn_out {:?} out {:?}",
            agreement(&tail.ffn_input, &le_u16(&read(layer, "ffn_in", "bfloat16"))),
            agreement(
                &tail.ffn_output,
                &le_u16(&read(layer, "ffn_out", "bfloat16"))
            ),
            agreement(&tail.output, &le_u16(&read(layer, "out", "bfloat16"))),
        );
    }
}

/// The pinned model repository and revision from the embedded registry.
fn server_registry() -> (String, String) {
    let registry: serde_json::Value =
        serde_json::from_str(include_str!("../../../config/artifacts/deepseek.json"))
            .expect("embedded registry");
    (
        registry["model_repo"].as_str().expect("repo").to_owned(),
        registry["model_revision"]
            .as_str()
            .expect("revision")
            .to_owned(),
    )
}
