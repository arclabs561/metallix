//! Test-only Qwen KV fork/replay differential.

use std::{env, path::PathBuf};

use mlx_rs::{Array, Dtype, StreamOrDevice};

use super::*;

type CacheSnapshot = Vec<(Vec<i32>, Dtype, Vec<f32>, Vec<i32>, Dtype, Vec<f32>)>;

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
                .expect("lossless BF16 key widening");
            let values = layer
                .values
                .as_type_device::<f32>(StreamOrDevice::gpu())
                .expect("lossless BF16 value widening");
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

#[test]
fn forked_particle_ancestry_replays_live_prefixes_and_retains_eos_snapshot() {
    let _gpu = GPU_TEST_LOCK.lock().expect("GPU test lock");
    let config = small_dense_config();
    let weights = deterministic_weights();
    let prompt = [1_i32, 2];
    // The test-only logical ancestry map duplicates B, discards A, and
    // preserves the EOS particle without a further decode.
    let parent_tails = [3_i32, 4, 5];
    let parents = parent_tails.map(|tail| {
        let mut executor = Qwen3ForwardExecutor::new(&config, &weights);
        executor
            .prefill_last_logits(&prompt)
            .expect("parent prefill");
        let logits = executor.decode_last_logits(tail).expect("parent tail");
        let cache = cache_snapshot(&executor);
        (executor, logits, cache)
    });
    let ancestry = [1_usize, 1, 2];
    let mut children = ancestry.map(|parent| {
        parents[parent]
            .0
            .fork_prefilled()
            .expect("prefilled snapshot fork")
    });

    for (child, &parent) in children.iter().zip(&ancestry) {
        assert_eq!(child.cached_tokens(), 3);
        assert_eq!(child.kv_bytes(), parents[parent].0.kv_bytes());
        assert_eq!(cache_snapshot(child), parents[parent].2);
        assert_logits_match(
            Qwen3ForwardExecutor::new(&config, &weights)
                .prefill_last_logits(&[1, 2, parent_tails[parent]])
                .expect("snapshot next-logit replay"),
            parents[parent].1.clone(),
        );
    }
    assert_eq!(children[0].cached_tokens(), children[1].cached_tokens());
    assert_eq!(children[2].cached_tokens(), 3, "EOS remains absorbed");
    let eos_cache = cache_snapshot(&children[2]);

    let first = children[0]
        .decode_last_logits(6)
        .expect("first B child decode");
    assert_logits_match(
        Qwen3ForwardExecutor::new(&config, &weights)
            .prefill_last_logits(&[1, 2, 4, 6])
            .expect("first independent replay"),
        first,
    );
    let second = children[1]
        .decode_last_logits(7)
        .expect("second B child decode");
    assert_logits_match(
        Qwen3ForwardExecutor::new(&config, &weights)
            .prefill_last_logits(&[1, 2, 4, 7])
            .expect("second independent replay"),
        second,
    );
    assert_eq!(children[0].cached_tokens(), 4);
    assert_eq!(children[1].cached_tokens(), 4);
    assert_eq!(children[2].cached_tokens(), 3, "EOS must not decode");
    assert_eq!(
        cache_snapshot(&children[2]),
        eos_cache,
        "EOS cache is unchanged"
    );
    assert_logits_match(
        Qwen3ForwardExecutor::new(&config, &weights)
            .prefill_last_logits(&[1, 2, 5])
            .expect("EOS independent replay"),
        parents[2].1.clone(),
    );

    // Each parent stayed at its original snapshot despite divergent child
    // appends. Replaying B from a fresh fork checks its actual KV state.
    for (parent, _, cache) in &parents {
        assert_eq!(parent.cached_tokens(), 3);
        assert_eq!(cache_snapshot(parent), *cache);
    }
    let mut parent_b = parents[1].0.fork_prefilled().expect("unchanged B parent");
    assert_logits_match(
        Qwen3ForwardExecutor::new(&config, &weights)
            .prefill_last_logits(&[1, 2, 4, 0])
            .expect("B parent continuation replay"),
        parent_b
            .decode_last_logits(0)
            .expect("B parent continuation"),
    );
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(16))]

    #[test]
    fn forked_live_branches_match_fresh_replay_for_bounded_ancestry(
        prompt_a in 0_i32..8,
        prompt_b in 0_i32..8,
        tail_a in 0_i32..8,
        tail_b in 0_i32..8,
        first_token in 0_i32..8,
        second_token in 0_i32..8,
        first_parent in 0_usize..2,
        second_parent in 0_usize..2,
    ) {
        let _gpu = GPU_TEST_LOCK.lock().expect("GPU test lock");
        let config = small_dense_config();
        let weights = deterministic_weights();
        let prompt = [prompt_a, prompt_b];
        let tails = [tail_a, tail_b, 5_i32];
        let parents = tails.map(|tail| {
            let mut executor = Qwen3ForwardExecutor::new(&config, &weights);
            executor.prefill_last_logits(&prompt).expect("parent prefill");
            let logits = executor.decode_last_logits(tail).expect("parent tail");
            let cache = cache_snapshot(&executor);
            (executor, logits, cache)
        });
        let ancestry = [first_parent, second_parent, 2_usize];
        let mut children = ancestry.map(|parent| {
            parents[parent]
                .0
                .fork_prefilled()
                .expect("prefilled snapshot fork")
        });
        let eos_cache = cache_snapshot(&children[2]);
        for (child, &parent) in children.iter().zip(&ancestry) {
            prop_assert_eq!(&cache_snapshot(child), &parents[parent].2);
            prop_assert_eq!(child.cached_tokens(), 3);
            prop_assert_eq!(parents[parent].0.cached_tokens(), 3);
        }
        for (child, (token, parent)) in children.iter_mut().take(2).zip([
            (first_token, first_parent),
            (second_token, second_parent),
        ]) {
            let logits = child.decode_last_logits(token).expect("live child decode");
            let expected = Qwen3ForwardExecutor::new(&config, &weights)
                .prefill_last_logits(&[prompt_a, prompt_b, tails[parent], token])
                .expect("independent replay");
            prop_assert_eq!(logits.len(), expected.len());
            for (actual, expected) in logits.into_iter().zip(expected) {
                prop_assert!((actual - expected).abs() <= 5e-5);
            }
        }
        prop_assert_eq!(children[2].cached_tokens(), 3);
        prop_assert_eq!(cache_snapshot(&children[2]), eos_cache);
        for (parent, _, cache) in &parents {
            prop_assert_eq!(parent.cached_tokens(), 3);
            prop_assert_eq!(&cache_snapshot(parent), cache);
        }
    }
}

#[test]
#[ignore = "requires METALLIX_QWEN_MODEL and a local Apple-Silicon Metal checkpoint"]
fn checkpoint_particle_ancestry_fork_replays_next_logits() {
    let Some(model) = env::var_os("METALLIX_QWEN_MODEL").map(PathBuf::from) else {
        eprintln!("skipping: METALLIX_QWEN_MODEL is not set");
        return;
    };
    let _gpu = GPU_TEST_LOCK.lock().expect("GPU test lock");
    let mut weights = crate::metal::Qwen3MlxWeights::load(model).expect("checkpoint load");
    // Match the qualified resident-chat precision path. BF16 full-prefill and
    // incremental reductions are a separate numerical qualification.
    weights.prepare_float32().expect("resident float32 weights");
    let prompt = [9_707_i32, 11];
    let parent_tails = [1_879_i32, 13, 151_645];
    let parents = parent_tails.map(|tail| {
        let mut executor = weights.executor();
        executor
            .prefill_last_logits(&prompt)
            .expect("parent prefill");
        let logits = executor.decode_last_logits(tail).expect("parent tail");
        let cache = cache_snapshot(&executor);
        (executor, logits, cache)
    });
    let ancestry = [1_usize, 1, 2];
    let mut children = ancestry.map(|parent| {
        parents[parent]
            .0
            .fork_prefilled()
            .expect("prefilled snapshot fork")
    });
    for (child, &parent) in children.iter().zip(&ancestry) {
        assert_eq!(child.cached_tokens(), 3);
        assert_eq!(child.kv_bytes(), parents[parent].0.kv_bytes());
        assert_eq!(cache_snapshot(child), parents[parent].2);
        assert_logits_match(
            weights
                .executor()
                .prefill_last_logits(&[9_707, 11, parent_tails[parent]])
                .expect("snapshot next-logit replay"),
            parents[parent].1.clone(),
        );
    }
    let eos_cache = cache_snapshot(&children[2]);
    for (child, (token, prefix)) in children.iter_mut().take(2).zip([
        (13_i32, &[9_707, 11, 13, 13][..]),
        (151_645, &[9_707, 11, 13, 151_645][..]),
    ]) {
        let logits = child.decode_last_logits(token).expect("live child decode");
        assert_logits_match(
            weights
                .executor()
                .prefill_last_logits(prefix)
                .expect("independent child replay"),
            logits,
        );
    }
    assert_eq!(children[2].cached_tokens(), 3, "EOS remains absorbed");
    assert_eq!(
        cache_snapshot(&children[2]),
        eos_cache,
        "EOS cache is unchanged"
    );
    assert_logits_match(
        weights
            .executor()
            .prefill_last_logits(&[9_707, 11, 151_645])
            .expect("EOS independent replay"),
        parents[2].1.clone(),
    );
    for (parent, _, cache) in &parents {
        assert_eq!(parent.cached_tokens(), 3);
        assert_eq!(cache_snapshot(parent), *cache);
    }
}

#[test]
fn fork_rejects_empty_missing_and_wrongly_shaped_kv() {
    let config = small_dense_config();
    let weights = deterministic_weights();
    let empty = Qwen3ForwardExecutor::new(&config, &weights);
    assert!(matches!(
        empty.fork_prefilled(),
        Err(Qwen3ForwardError::DecodeWithoutPrefill)
    ));

    let _gpu = GPU_TEST_LOCK.lock().expect("GPU test lock");
    let mut missing = Qwen3ForwardExecutor::new(&config, &weights);
    missing.prefill_last_logits(&[1, 2]).expect("prefill");
    missing.cache[0] = None;
    assert!(matches!(
        missing.fork_prefilled(),
        Err(Qwen3ForwardError::CacheInconsistent)
    ));

    let mut wrong_shape = Qwen3ForwardExecutor::new(&config, &weights);
    wrong_shape.prefill_last_logits(&[1, 2]).expect("prefill");
    wrong_shape.cache[0].as_mut().expect("layer cache").keys =
        Array::from_slice(&[0.0_f32; 4], &[1, 1, 1, 4]);
    assert!(matches!(
        wrong_shape.fork_prefilled(),
        Err(Qwen3ForwardError::CacheInconsistent)
    ));
}
