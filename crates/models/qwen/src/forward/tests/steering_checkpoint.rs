//! Ignored local-checkpoint execution qualification for Qwen residual steering.
//!
//! This checks numerical and cache-state mechanics only. It does not measure a
//! calibrated artifact or any held-out behavioral benefit.

use std::{env, path::PathBuf};

use mlx_rs::{Dtype, StreamOrDevice};

use super::*;

type CacheSnapshot = Vec<(Vec<i32>, Dtype, Vec<f32>, Vec<i32>, Dtype, Vec<f32>)>;

fn checkpoint_logits_match(label: &str, actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    assert!(actual.iter().chain(expected).all(|value| value.is_finite()));
    let maximum = actual
        .iter()
        .zip(expected)
        .map(|(left, right)| (left - right).abs())
        .fold(0.0_f32, f32::max);
    eprintln!("{label}: max absolute logit difference {maximum}");
    // Same pre-existing bound as the checkpoint cache/fork replay gate:
    // full prefill and one-token decode use different matrix shapes.
    assert!(maximum <= 5e-5, "{label}: logit difference {maximum}");
}

fn cache_snapshot(
    executor: &Qwen3ForwardExecutor<'_, std::collections::hash_map::RandomState>,
) -> CacheSnapshot {
    executor
        .cache
        .iter()
        .map(|entry| {
            let layer = entry.as_ref().expect("complete KV cache");
            let keys = layer
                .keys
                .as_type_device::<f32>(StreamOrDevice::gpu())
                .expect("lossless key widening")
                .contiguous()
                .expect("row-major key cache");
            let values = layer
                .values
                .as_type_device::<f32>(StreamOrDevice::gpu())
                .expect("lossless value widening")
                .contiguous()
                .expect("row-major value cache");
            keys.eval().expect("materialized key cache");
            values.eval().expect("materialized value cache");
            (
                layer.keys.shape().to_vec(),
                layer.keys.dtype(),
                keys.as_slice::<f32>().to_vec(),
                layer.values.shape().to_vec(),
                layer.values.dtype(),
                values.as_slice::<f32>().to_vec(),
            )
        })
        .collect()
}

fn synthetic_steering(config: &Qwen3ForwardConfig, coefficient: f32) -> Qwen3ResidualSteering {
    let residual = (0..config.hidden_size)
        .map(|index| {
            let phase = u16::try_from(index % 17).expect("bounded synthetic phase");
            (f32::from(phase) - 8.0) / 8.0
        })
        .collect();
    let artifact = Qwen3ResidualSteeringArtifact::new(
        "Qwen/Qwen3-0.6B@c1899de289a04d12100db370d81485cdf75e47ca",
        config.hidden_layers,
        config.hidden_size,
        0,
        residual,
        coefficient,
        Qwen3SteeringPositionRange::new(1, 4).expect("explicit nonempty position range"),
    )
    .expect("finite synthetic steering artifact");
    Qwen3ResidualSteering::bind(artifact, config).expect("artifact binds to local checkpoint")
}

#[test]
#[ignore = "requires METALLIX_QWEN_MODEL pointing to Qwen3-0.6B on Apple-Silicon Metal"]
fn checkpoint_residual_steering_preserves_decode_and_fork_mechanics() {
    let Some(model) = env::var_os("METALLIX_QWEN_MODEL").map(PathBuf::from) else {
        eprintln!("skipping: METALLIX_QWEN_MODEL is not set");
        return;
    };
    let _gpu = GPU_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let mut weights = crate::metal::Qwen3MlxWeights::load(model).expect("checkpoint load");
    weights.prepare_float32().expect("resident float32 weights");

    let prompt = [9_707_i32, 11, 1_879];
    let mut baseline = weights.executor();
    let baseline_logits = baseline
        .prefill_last_logits(&prompt)
        .expect("baseline full prefill");

    let mut baseline_chunked = weights.executor();
    let _ = baseline_chunked
        .prefill_last_logits(&prompt[..2])
        .expect("unsteered chunk prefill");
    let baseline_chunked_logits = baseline_chunked
        .decode_last_logits(prompt[2])
        .expect("unsteered chunk decode");
    checkpoint_logits_match(
        "unsteered full/chunk control",
        &baseline_chunked_logits,
        &baseline_logits,
    );

    let mut zero = weights.executor();
    let zero_steering = synthetic_steering(zero.config, 0.0);
    zero.set_residual_steering(Some(zero_steering))
        .expect("install zero steering");
    let zero_logits = zero
        .prefill_last_logits(&prompt)
        .expect("zero-steering full prefill");
    assert_eq!(
        zero_logits, baseline_logits,
        "zero coefficient must preserve the checkpoint logits exactly"
    );

    let mut full = weights.executor();
    let steering = synthetic_steering(full.config, 4.0);
    full.set_residual_steering(Some(steering.clone()))
        .expect("install synthetic steering");
    let full_logits = full
        .prefill_last_logits(&prompt)
        .expect("steered full prefill");
    assert!(
        full_logits.iter().all(|value| value.is_finite()),
        "steering must retain finite checkpoint logits"
    );
    assert!(
        full_logits
            .iter()
            .zip(&baseline_logits)
            .any(|(steered, plain)| steered.to_bits() != plain.to_bits()),
        "nonzero synthetic residual must have an observable numerical effect"
    );

    let mut chunked = weights.executor();
    chunked
        .set_residual_steering(Some(steering.clone()))
        .expect("install chunked steering");
    let _ = chunked
        .prefill_last_logits(&prompt[..2])
        .expect("steered chunk prefill");
    let chunked_logits = chunked
        .decode_last_logits(prompt[2])
        .expect("steered chunk decode");
    checkpoint_logits_match("steered full/chunk", &chunked_logits, &full_logits);

    let mut parent = weights.executor();
    parent
        .set_residual_steering(Some(steering))
        .expect("install parent steering");
    let _ = parent
        .prefill_last_logits(&prompt[..2])
        .expect("steered parent prefill");
    let parent_snapshot = cache_snapshot(&parent);
    let mut child = parent.fork_prefilled().expect("fork steered parent");
    let child_logits = child
        .decode_last_logits(prompt[2])
        .expect("steered child decode");
    checkpoint_logits_match("steered fork/full", &child_logits, &full_logits);
    assert!(
        child_logits == chunked_logits,
        "same-shape fork decode must retain steering state exactly"
    );
    assert_eq!(
        parent.cached_tokens(),
        2,
        "child decode must not advance parent"
    );
    assert_eq!(
        cache_snapshot(&parent),
        parent_snapshot,
        "child decode must not mutate parent KV"
    );
}
