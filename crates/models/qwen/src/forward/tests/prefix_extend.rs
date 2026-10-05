//! Shared-prefix prefill followed by chunked suffix extension.
//!
//! The oracle is one fresh prefill of the concatenated tokens. Two decoder
//! layers make the chunk's causal mask observable: the second layer's K/V for
//! early chunk positions depend on what the first layer let them attend to.
//! The shared deterministic weights are near-uniform and small enough that an
//! unmasked chunk moves logits by only 1e-8, so these tests use signed,
//! unit-scale weights instead.

use super::*;

fn sensitive_weights() -> HashMap<String, Array> {
    let mut named = deterministic_weights_for_layers(2)
        .into_iter()
        .collect::<Vec<_>>();
    // Sorted so each tensor's salt, and therefore its values, is stable.
    named.sort_by(|left, right| left.0.cmp(&right.0));
    named
        .into_iter()
        .enumerate()
        .map(|(salt, (name, array))| {
            let count = array.size();
            let values = (0..count)
                .map(|index| {
                    let mixed = (index * 7919 + salt * 104_729) % 23;
                    (f32::from(u8::try_from(mixed).expect("bounded")) - 11.0) * 0.15
                })
                .collect::<Vec<_>>();
            let shaped = Array::from_slice(&values, array.shape());
            shaped.eval().expect("weights materialize");
            (name, shaped)
        })
        .collect()
}

fn two_layer_long_config() -> Qwen3ForwardConfig {
    let mut config = long_small_config();
    config.hidden_layers = 2;
    config
}

const PREFIX: [i32; 6] = [1, 5, 2, 7, 3, 6];
const SUFFIXES: [&[i32]; 3] = [&[4, 2, 7], &[6], &[3, 3, 1, 5, 0, 2]];

fn concatenated(suffix: &[i32]) -> Vec<i32> {
    PREFIX.iter().chain(suffix).copied().collect()
}

#[test]
fn forked_prefix_extension_matches_fresh_prefill_for_each_suffix() {
    let _gpu = GPU_TEST_LOCK.lock().expect("GPU test lock");
    let config = two_layer_long_config();
    let weights = sensitive_weights();
    let plan = config
        .resident_chat_plan(600, u64::MAX)
        .expect("tiny resident plan");
    let mut shared = Qwen3ForwardExecutor::new_for_resident_chat(&config, &weights, plan);
    shared.prefill_last_logits(&PREFIX).expect("shared prefill");
    for suffix in SUFFIXES {
        let mut branch = shared.fork_prefilled().expect("branch fork");
        let branched = branch.extend_last_logits(suffix).expect("suffix chunk");
        assert_eq!(branch.cached_tokens(), PREFIX.len() + suffix.len());
        let mut fresh = Qwen3ForwardExecutor::new_for_resident_chat(&config, &weights, plan);
        let expected = fresh
            .prefill_last_logits(&concatenated(suffix))
            .expect("fresh prefill");
        assert_logits_match(expected, branched);
        // A later decode sees the chunk's K/V in both layers.
        assert_logits_match(
            fresh.decode_last_logits(4).expect("fresh decode"),
            branch.decode_last_logits(4).expect("branch decode"),
        );
    }
    // Branches never wrote into the shared snapshot.
    assert_eq!(shared.cached_tokens(), PREFIX.len());
}

#[test]
fn chunk_crossing_a_capacity_step_matches_fresh_prefill() {
    let _gpu = GPU_TEST_LOCK.lock().expect("GPU test lock");
    let config = two_layer_long_config();
    let weights = sensitive_weights();
    let plan = config
        .resident_chat_plan(600, u64::MAX)
        .expect("tiny resident plan");
    // 120 cached tokens sit in the 128-row tier; the 20-token chunk grows the
    // storage to the 512-row tier inside the same append.
    let prompt = (0..140).map(|index| index % 8).collect::<Vec<i32>>();
    let mut branched = Qwen3ForwardExecutor::new_for_resident_chat(&config, &weights, plan);
    branched
        .prefill_last_logits(&prompt[..120])
        .expect("prefix prefill");
    let logits = branched
        .extend_last_logits(&prompt[120..])
        .expect("crossing chunk");
    let mut fresh = Qwen3ForwardExecutor::new_for_resident_chat(&config, &weights, plan);
    assert_logits_match(
        fresh.prefill_last_logits(&prompt).expect("fresh prefill"),
        logits,
    );
}

#[test]
fn diagnostic_executor_extension_matches_fresh_prefill() {
    let _gpu = GPU_TEST_LOCK.lock().expect("GPU test lock");
    let config = two_layer_long_config();
    let weights = sensitive_weights();
    let suffix = SUFFIXES[2];
    let mut branched = Qwen3ForwardExecutor::new(&config, &weights);
    branched.prefill_last_logits(&PREFIX).expect("prefix");
    let logits = branched.extend_last_logits(suffix).expect("chunk");
    let mut fresh = Qwen3ForwardExecutor::new(&config, &weights);
    assert_logits_match(
        fresh
            .prefill_last_logits(&concatenated(suffix))
            .expect("fresh"),
        logits,
    );
}

#[test]
fn extension_requires_a_prefilled_sequence() {
    let _gpu = GPU_TEST_LOCK.lock().expect("GPU test lock");
    let config = two_layer_long_config();
    let weights = sensitive_weights();
    let mut executor = Qwen3ForwardExecutor::new(&config, &weights);
    assert!(matches!(
        executor.extend_last_logits(&[1, 2]),
        Err(Qwen3ForwardError::DecodeWithoutPrefill)
    ));
}

/// Head width 64 and chunks longer than eight tokens select MLX's fused
/// full-attention Metal kernel instead of the generic fallback the 4-wide
/// fixtures use.
fn fused_kernel_config() -> Qwen3ForwardConfig {
    Qwen3ForwardConfig::parse(
        r#"{
          "model_type":"qwen3",
          "num_hidden_layers":2,
          "hidden_size":16,
          "intermediate_size":16,
          "vocab_size":16,
          "num_attention_heads":2,
          "num_key_value_heads":1,
          "head_dim":64,
          "max_position_embeddings":1024,
          "rms_norm_eps":0.000001,
          "rope_theta":1000000,
          "hidden_act":"silu",
          "tie_word_embeddings":true,
          "attention_bias":false,
          "mlp_bias":false
        }"#,
    )
    .expect("fused-kernel tiny config")
}

fn fused_kernel_weights() -> HashMap<String, Array> {
    let (hidden, query, kv, intermediate, vocab) = (16, 128, 64, 16, 16);
    let mut shapes = vec![
        ("model.embed_tokens.weight".to_owned(), vec![vocab, hidden]),
        ("model.norm.weight".to_owned(), vec![hidden]),
    ];
    for layer in 0..2 {
        let base = format!("model.layers.{layer}");
        let attn = format!("{base}.self_attn");
        let mlp = format!("{base}.mlp");
        shapes.extend([
            (format!("{base}.input_layernorm.weight"), vec![hidden]),
            (
                format!("{base}.post_attention_layernorm.weight"),
                vec![hidden],
            ),
            (format!("{attn}.q_proj.weight"), vec![query, hidden]),
            (format!("{attn}.k_proj.weight"), vec![kv, hidden]),
            (format!("{attn}.v_proj.weight"), vec![kv, hidden]),
            (format!("{attn}.o_proj.weight"), vec![hidden, query]),
            (format!("{attn}.q_norm.weight"), vec![64]),
            (format!("{attn}.k_norm.weight"), vec![64]),
            (
                format!("{mlp}.gate_proj.weight"),
                vec![intermediate, hidden],
            ),
            (format!("{mlp}.up_proj.weight"), vec![intermediate, hidden]),
            (
                format!("{mlp}.down_proj.weight"),
                vec![hidden, intermediate],
            ),
        ]);
    }
    shapes
        .into_iter()
        .enumerate()
        .map(|(salt, (name, shape))| {
            let count = shape.iter().product::<i32>();
            let values = (0..usize::try_from(count).expect("small test shape"))
                .map(|index| {
                    let mixed = (index * 7919 + salt * 104_729) % 23;
                    (f32::from(u8::try_from(mixed).expect("bounded")) - 11.0) * 0.15
                })
                .collect::<Vec<_>>();
            (name, Array::from_slice(&values, &shape))
        })
        .collect()
}

#[test]
fn fused_kernel_chunk_with_unaligned_offset_matches_fresh_prefill() {
    let _gpu = GPU_TEST_LOCK.lock().expect("GPU test lock");
    let config = fused_kernel_config();
    let weights = fused_kernel_weights();
    let plan = config
        .resident_chat_plan(600, u64::MAX)
        .expect("tiny resident plan");
    // 64 keys fill whole key blocks while the 40-token chunk starts at 24, so
    // the first query tile's diagonal begins inside an earlier key block.
    let prompt = (0..64)
        .map(|index| (index * 5 + 3) % 16)
        .collect::<Vec<i32>>();
    for split in [24, 30, 50] {
        let mut branched = Qwen3ForwardExecutor::new_for_resident_chat(&config, &weights, plan);
        branched
            .prefill_last_logits(&prompt[..split])
            .expect("prefix prefill");
        let logits = branched
            .extend_last_logits(&prompt[split..])
            .expect("chunk");
        let mut fresh = Qwen3ForwardExecutor::new_for_resident_chat(&config, &weights, plan);
        assert_logits_match(
            fresh.prefill_last_logits(&prompt).expect("fresh prefill"),
            logits,
        );
    }
}
