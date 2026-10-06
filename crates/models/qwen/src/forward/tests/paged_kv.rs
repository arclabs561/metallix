//! Paged K/V against the contiguous executor.
//!
//! For one sequence the paged session runs the same projections, `RoPE`,
//! masks and fused SDPA as the contiguous executor; only where K/V live
//! differs. Its logits must therefore be bit-identical, which is stronger
//! than the greedy-token gate and catches a wrong slot, block or position
//! immediately.
//!
//! The fixture is a two-layer model with signed weights, a 64-token
//! vocabulary and an untied output projection, so greedy decoding moves
//! through several tokens instead of settling on one, and the second layer's
//! K/V depend on the first layer's attention pattern.

use std::time::{Duration, Instant};

use mlx_rs::ops::indexing::IndexOp;

use engine::blocks::{BlockTokens, PoolConfig, SequenceId};

use super::*;
use crate::forward::PagedQwen3Session;

pub(super) const VOCAB: usize = 64;

pub(super) fn paged_config() -> Qwen3ForwardConfig {
    Qwen3ForwardConfig::parse(
        r#"{
          "model_type":"qwen3",
          "num_hidden_layers":2,
          "hidden_size":32,
          "intermediate_size":64,
          "vocab_size":64,
          "num_attention_heads":4,
          "num_key_value_heads":2,
          "head_dim":8,
          "max_position_embeddings":4096,
          "rms_norm_eps":0.000001,
          "rope_theta":10000,
          "hidden_act":"silu",
          "tie_word_embeddings":false,
          "attention_bias":false,
          "mlp_bias":false
        }"#,
    )
    .expect("paged fixture config")
}

/// Signed pseudo-random weights from a fixed linear congruential stream.
pub(super) fn paged_weights(config: &Qwen3ForwardConfig) -> HashMap<String, Array> {
    let mut state = 0x2545_f491_u32;
    let mut values = |count: usize, scale: f32| -> Vec<f32> {
        (0..count)
            .map(|_| {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                let unit = f32::from(u16::try_from(state >> 16).expect("16 bits")) / 65_535.0;
                (unit - 0.5) * scale
            })
            .collect()
    };
    let hidden = config.hidden_size;
    let shapes = |name: &str| -> Option<(usize, usize, f32)> {
        let q = config.attention_heads * config.head_dim;
        let kv = config.key_value_heads * config.head_dim;
        let inter = config.intermediate_size;
        Some(match name {
            "q_proj" => (q, hidden, 0.6),
            "k_proj" | "v_proj" => (kv, hidden, 0.6),
            "o_proj" => (hidden, q, 0.6),
            "gate_proj" | "up_proj" => (inter, hidden, 0.6),
            "down_proj" => (hidden, inter, 0.6),
            _ => return None,
        })
    };
    let mut weights = HashMap::new();
    let mut insert = |name: String, shape: &[usize], scale: f32, offset: f32| {
        let count = shape.iter().product();
        let data = values(count, scale)
            .into_iter()
            .map(|value| value + offset)
            .collect::<Vec<_>>();
        let dims = shape
            .iter()
            .map(|&dim| i32::try_from(dim).expect("small"))
            .collect::<Vec<_>>();
        let array = Array::from_slice(&data, &dims);
        array.eval().expect("weights materialize");
        weights.insert(name, array);
    };
    insert(
        "model.embed_tokens.weight".into(),
        &[VOCAB, hidden],
        2.0,
        0.0,
    );
    // Untied: with a tied output projection the input token's own embedding
    // dominates its logits, and greedy decoding repeats one token.
    insert("lm_head.weight".into(), &[VOCAB, hidden], 2.0, 0.0);
    insert("model.norm.weight".into(), &[hidden], 0.4, 1.0);
    for layer in 0..config.hidden_layers {
        let base = format!("model.layers.{layer}");
        for norm in ["input_layernorm", "post_attention_layernorm"] {
            insert(format!("{base}.{norm}.weight"), &[hidden], 0.4, 1.0);
        }
        for norm in ["q_norm", "k_norm"] {
            insert(
                format!("{base}.self_attn.{norm}.weight"),
                &[config.head_dim],
                0.4,
                1.0,
            );
        }
        for (group, names) in [
            ("self_attn", &["q_proj", "k_proj", "v_proj", "o_proj"][..]),
            ("mlp", &["gate_proj", "up_proj", "down_proj"][..]),
        ] {
            for name in names {
                let (rows, columns, scale) = shapes(name).expect("known projection");
                insert(
                    format!("{base}.{group}.{name}.weight"),
                    &[rows, columns],
                    scale,
                    0.0,
                );
            }
        }
    }
    weights
}

pub(super) fn prompt(len: usize, seed: usize) -> Vec<i32> {
    (0..len)
        .map(|index| i32::try_from((index * 37 + seed * 11 + 5) % VOCAB).expect("small"))
        .collect()
}

pub(super) fn argmax(logits: &[f32]) -> i32 {
    let (index, _) = logits
        .iter()
        .enumerate()
        .max_by(|left, right| left.1.total_cmp(right.1))
        .expect("non-empty logits");
    i32::try_from(index).expect("vocabulary index fits i32")
}

fn assert_bits_equal(paged: &[f32], contiguous: &[f32], context: &str) {
    assert_eq!(paged.len(), contiguous.len(), "{context}: logit count");
    for (index, (left, right)) in paged.iter().zip(contiguous).enumerate() {
        assert_eq!(
            left.to_bits(),
            right.to_bits(),
            "{context}: logit {index} paged {left} contiguous {right}"
        );
    }
}

pub(super) fn pool(slabs: u32) -> PoolConfig {
    PoolConfig::new(BlockTokens::DEFAULT, slabs)
        .expect("pool")
        .with_prefix_caching(false)
}

/// Greedy-decodes `steps` tokens on both paths, asserting identical logits at
/// every step, and returns the tokens.
fn decode_both(
    paged: &mut PagedQwen3Session<'_, std::collections::hash_map::RandomState>,
    seq: SequenceId,
    contiguous: &mut Qwen3ForwardExecutor<'_, std::collections::hash_map::RandomState>,
    mut logits: Vec<f32>,
    steps: usize,
    context: &str,
) -> Vec<i32> {
    let mut tokens = Vec::with_capacity(steps);
    for step in 0..steps {
        let token = argmax(&logits);
        tokens.push(token);
        let paged_logits = paged.decode_last_logits(seq, token).expect("paged decode");
        let contiguous_logits = contiguous
            .decode_last_logits(token)
            .expect("contiguous decode");
        assert_bits_equal(
            &paged_logits,
            &contiguous_logits,
            &format!("{context} decode step {step}"),
        );
        logits = paged_logits;
    }
    tokens
}

#[test]
fn paged_greedy_decode_matches_the_contiguous_executor() {
    let _gpu = GPU_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let config = paged_config();
    let weights = paged_weights(&config);
    let mut paged = PagedQwen3Session::new(&config, &weights, pool(1)).expect("paged session");
    let mut contiguous = Qwen3ForwardExecutor::new(&config, &weights);
    let seq = SequenceId(1);
    // 21 tokens: one full block and a partial one.
    let prompt = prompt(21, 1);

    let paged_logits = paged
        .prefill_last_logits(seq, &prompt)
        .expect("paged prefill");
    let contiguous_logits = contiguous
        .prefill_last_logits(&prompt)
        .expect("contiguous prefill");
    assert_bits_equal(&paged_logits, &contiguous_logits, "prefill");

    let tokens = decode_both(&mut paged, seq, &mut contiguous, paged_logits, 32, "greedy");
    let distinct = tokens
        .iter()
        .collect::<std::collections::BTreeSet<_>>()
        .len();
    assert!(
        distinct >= 4,
        "the fixture must not collapse onto one token: {tokens:?}"
    );
    assert_eq!(paged.cached_tokens(seq).expect("live"), 21 + 32);
    assert_eq!(
        paged.blocks().block_table(seq).expect("live").len(),
        (21 + 32_usize).div_ceil(16)
    );

    paged.free(seq).expect("live");
    assert_eq!(paged.blocks().free_blocks(), paged.blocks().total_blocks());
}

#[test]
fn paged_chunked_prefill_across_blocks_and_slabs_matches_contiguous_chunks() {
    let _gpu = GPU_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let config = paged_config();
    let weights = paged_weights(&config);
    let mut paged = PagedQwen3Session::new(&config, &weights, pool(2)).expect("paged session");
    // A resident filler takes 31 of slab 0's 32 blocks, so the sequence under
    // test starts in slab 0 and continues in slab 1.
    let filler = SequenceId(99);
    paged
        .prefill_last_logits(filler, &prompt(31 * 16, 7))
        .expect("filler prefill");
    let seq = SequenceId(1);
    let mut contiguous = Qwen3ForwardExecutor::new(&config, &weights);
    let prompt = prompt(45, 2);
    // Chunks end at 5, 21, 34, 45: each later chunk crosses a block boundary.
    let chunks = [&prompt[..5], &prompt[5..21], &prompt[21..34], &prompt[34..]];
    let mut paged_logits = paged.prefill_last_logits(seq, chunks[0]).expect("prefill");
    let mut contiguous_logits = contiguous.prefill_last_logits(chunks[0]).expect("prefill");
    assert_bits_equal(&paged_logits, &contiguous_logits, "chunk 0");
    for (index, chunk) in chunks.iter().enumerate().skip(1) {
        paged_logits = paged.extend_last_logits(seq, chunk).expect("paged chunk");
        contiguous_logits = contiguous.extend_last_logits(chunk).expect("chunk");
        assert_bits_equal(&paged_logits, &contiguous_logits, &format!("chunk {index}"));
    }
    let table = paged.blocks().block_table(seq).expect("live").to_vec();
    assert!(
        table.iter().any(|block| block.slab() == 0) && table.iter().any(|block| block.slab() == 1),
        "the sequence must span both slabs: {table:?}"
    );
    decode_both(
        &mut paged,
        seq,
        &mut contiguous,
        paged_logits,
        8,
        "after chunks",
    );
}

#[test]
fn paged_fork_copies_the_shared_tail_and_both_branches_match() {
    let _gpu = GPU_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let config = paged_config();
    let weights = paged_weights(&config);
    let mut paged = PagedQwen3Session::new(&config, &weights, pool(1)).expect("paged session");
    let (parent, child) = (SequenceId(1), SequenceId(2));
    let prompt = prompt(21, 3);
    let mut contiguous_parent = Qwen3ForwardExecutor::new(&config, &weights);
    let prefill = paged.prefill_last_logits(parent, &prompt).expect("prefill");
    let expected = contiguous_parent
        .prefill_last_logits(&prompt)
        .expect("contiguous prefill");
    assert_bits_equal(&prefill, &expected, "prefill");
    let mut contiguous_child = contiguous_parent.fork_prefilled().expect("contiguous fork");
    paged.fork(parent, child).expect("fork");

    let shared = paged.blocks().block_table(parent).expect("live").to_vec();
    assert_eq!(paged.blocks().block_table(child).expect("live"), shared);
    assert_eq!(paged.blocks().ref_count(shared[1]), Some(2));

    // Different next tokens, interleaved: each branch's first append into the
    // shared partial block must leave the other branch's rows intact.
    let mut parent_token = argmax(&prefill);
    let mut child_token = (parent_token + 1) % 64;
    let before = (0..config.hidden_layers)
        .map(|layer| buffer_address(paged.layer_arrays(layer).0))
        .collect::<Vec<_>>();
    for step in 0..6 {
        let parent_logits = paged
            .decode_last_logits(parent, parent_token)
            .expect("parent decode");
        let expected = contiguous_parent
            .decode_last_logits(parent_token)
            .expect("contiguous parent decode");
        assert_bits_equal(&parent_logits, &expected, &format!("parent step {step}"));
        let child_logits = paged
            .decode_last_logits(child, child_token)
            .expect("child decode");
        let expected = contiguous_child
            .decode_last_logits(child_token)
            .expect("contiguous child decode");
        assert_bits_equal(&child_logits, &expected, &format!("child step {step}"));
        parent_token = argmax(&parent_logits);
        child_token = argmax(&child_logits);
        if step == 0 {
            let parent_table = paged.blocks().block_table(parent).expect("live");
            let child_table = paged.blocks().block_table(child).expect("live");
            assert_eq!(parent_table[0], child_table[0], "full block stays shared");
            assert_eq!(paged.blocks().ref_count(shared[0]), Some(2));
            assert_ne!(parent_table[1], child_table[1], "tail was copied on write");
            // The copy wrote into the pool in place, not into a new array.
            for (layer, address) in before.iter().enumerate() {
                assert_eq!(
                    buffer_address(paged.layer_arrays(layer).0),
                    *address,
                    "layer {layer}: copy-on-write copied the pool array"
                );
            }
        }
    }
}

#[test]
fn paged_out_of_blocks_leaves_the_sequence_usable() {
    let _gpu = GPU_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let config = paged_config();
    let weights = paged_weights(&config);
    let mut paged = PagedQwen3Session::new(&config, &weights, pool(1)).expect("paged session");
    let mut contiguous = Qwen3ForwardExecutor::new_for_resident_chat(
        &config,
        &weights,
        config
            .resident_chat_plan(
                1024,
                u64::MAX,
                crate::forward::Qwen3WeightPrecision::Float32,
            )
            .expect("plan"),
    );
    let seq = SequenceId(1);
    // 31 full blocks and 15 tokens of the last: room for one more token.
    let prompt = prompt(32 * 16 - 1, 4);
    let logits = paged.prefill_last_logits(seq, &prompt).expect("prefill");
    let expected = contiguous.prefill_last_logits(&prompt).expect("prefill");
    assert_bits_equal(&logits, &expected, "prefill");
    let refused = paged.extend_last_logits(seq, &[1, 2]);
    assert!(
        matches!(refused, Err(Qwen3ForwardError::KvBlocks(_))),
        "{refused:?}"
    );
    assert_eq!(paged.cached_tokens(seq).expect("still live"), prompt.len());
    let token = argmax(&logits);
    let logits = paged.decode_last_logits(seq, token).expect("last slot");
    let expected = contiguous.decode_last_logits(token).expect("decode");
    assert_bits_equal(&logits, &expected, "decode into the last slot");
}

/// The address of a pool array's evaluated buffer.
fn buffer_address(array: &Array) -> usize {
    array.eval().expect("pool array evaluates");
    array.as_slice::<f32>().as_ptr() as usize
}

#[test]
fn paged_writes_reuse_the_pool_buffer() {
    let _gpu = GPU_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let config = paged_config();
    let weights = paged_weights(&config);
    let mut paged = PagedQwen3Session::new(&config, &weights, pool(2)).expect("paged session");
    let seq = SequenceId(1);
    let logits = paged
        .prefill_last_logits(seq, &prompt(21, 5))
        .expect("prefill");
    let tail = *paged
        .blocks()
        .block_table(seq)
        .expect("live")
        .last()
        .expect("non-empty");
    let before = (0..config.hidden_layers)
        .map(|layer| {
            let (keys, values) = paged.layer_arrays(layer);
            (buffer_address(keys), buffer_address(values))
        })
        .collect::<Vec<_>>();
    paged
        .decode_last_logits(seq, argmax(&logits))
        .expect("decode");
    for (layer, (keys_before, values_before)) in before.into_iter().enumerate() {
        let (keys, values) = paged.layer_arrays(layer);
        // The decode wrote position 21, offset 5 of the tail block.
        let row = keys.index((i32::try_from(tail.index() * 16 + 5).expect("small"), .., ..));
        let row = row.contiguous().expect("contiguous row");
        row.eval().expect("row");
        let written = row.as_slice::<f32>().iter().any(|value| *value != 0.0);
        assert!(written, "layer {layer}: the decode wrote its key row");
        assert_eq!(
            buffer_address(keys),
            keys_before,
            "layer {layer} keys copied"
        );
        assert_eq!(
            buffer_address(values),
            values_before,
            "layer {layer} values copied"
        );
    }
}

pub(super) fn fixed_tokens(len: usize, seed: u64) -> Vec<i32> {
    let mut state = seed;
    (0..len)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            // Ordinary-text range of the Qwen3 vocabulary.
            i32::try_from((state >> 33) % 100_000).expect("bounded") + 100
        })
        .collect()
}

pub(super) const MIB: u64 = 1024 * 1024;

type Weights = HashMap<String, Array>;

/// Qwen3-0.6B at F32: identical greedy tokens and logits against the
/// resident-chat executor, then the donation check at two pool sizes.
#[test]
#[ignore = "requires METALLIX_QWEN_MODEL pointing to Qwen3-0.6B on Apple-Silicon Metal"]
fn paged_qwen3_06b_matches_resident_chat_and_writes_in_place() {
    use crate::metal::Qwen3MlxWeights;

    let model = std::env::var_os("METALLIX_QWEN_MODEL")
        .map(std::path::PathBuf::from)
        .expect("METALLIX_QWEN_MODEL is required for this ignored test");
    let _gpu = GPU_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let mut checkpoint = Qwen3MlxWeights::load(model).expect("checkpoint load");
    checkpoint.prepare_float32().expect("float32 weights");
    let mut resident = checkpoint
        .resident_chat_executor(4096, u64::MAX)
        .expect("resident plan");
    let (config, weights) = (resident.config, resident.weights);
    let prompt = fixed_tokens(200, 17);
    greedy_identity(config, weights, &mut resident, &prompt);
    contiguous_reference_timing(&mut resident, &prompt);
    for (round, budget) in [256 * MIB, 2048 * MIB, 256 * MIB, 2048 * MIB]
        .into_iter()
        .enumerate()
    {
        donation_round(config, weights, &prompt, round, budget);
    }
}

fn greedy_identity(
    config: &Qwen3ForwardConfig,
    weights: &Weights,
    resident: &mut Qwen3ForwardExecutor<'_, std::collections::hash_map::RandomState>,
    prompt: &[i32],
) {
    let pool = PagedQwen3Session::pool_for_budget(config, weights, 512 * MIB, BlockTokens::DEFAULT)
        .expect("pool")
        .with_prefix_caching(false);
    let mut paged = PagedQwen3Session::new(config, weights, pool).expect("paged session");
    let seq = SequenceId(1);
    let mut paged_logits = paged
        .prefill_last_logits(seq, prompt)
        .expect("paged prefill");
    let mut resident_logits = resident
        .prefill_last_logits(prompt)
        .expect("resident prefill");
    let mut identical_steps = usize::from(paged_logits == resident_logits);
    let mut paged_tokens = Vec::new();
    let mut resident_tokens = Vec::new();
    for _ in 0..32 {
        let (paged_token, resident_token) = (argmax(&paged_logits), argmax(&resident_logits));
        paged_tokens.push(paged_token);
        resident_tokens.push(resident_token);
        paged_logits = paged.decode_last_logits(seq, paged_token).expect("paged");
        resident_logits = resident
            .decode_last_logits(resident_token)
            .expect("resident");
        identical_steps += usize::from(paged_logits == resident_logits);
    }
    println!(
        "paged_qwen3_06b greedy prompt_tokens={} decode_steps=32 tokens_identical={} \
         bit_identical_logit_steps={identical_steps}/33 tokens={paged_tokens:?}",
        prompt.len(),
        paged_tokens == resident_tokens,
    );
    assert_eq!(paged_tokens, resident_tokens);
    assert_eq!(identical_steps, 33, "logits must be bit-identical at B=1");
}

/// The contiguous executor's decode step at the same context, for the
/// gather's cost; not a gate.
fn contiguous_reference_timing(
    resident: &mut Qwen3ForwardExecutor<'_, std::collections::hash_map::RandomState>,
    prompt: &[i32],
) {
    let mut logits = resident.prefill_last_logits(prompt).expect("prefill");
    for _ in 0..4 {
        logits = resident.decode_last_logits(argmax(&logits)).expect("warm");
    }
    let mut steps = Vec::new();
    for _ in 0..48 {
        let started = Instant::now();
        logits = resident
            .decode_last_logits(argmax(&logits))
            .expect("decode");
        steps.push(started.elapsed());
    }
    steps.sort();
    println!(
        "paged_qwen3_06b contiguous_reference decode_steps=48 median_step_ms={:.3} p90_step_ms={:.3}",
        millis(steps[steps.len() / 2]),
        millis(steps[steps.len() * 9 / 10]),
    );
}

/// Decode step time and peak bytes must not depend on the pool size, and a
/// decode must leave every layer's pool arrays at the same buffers.
fn donation_round(
    config: &Qwen3ForwardConfig,
    weights: &Weights,
    prompt: &[i32],
    round: usize,
    budget: u64,
) {
    let seq = SequenceId(1);
    let pool = PagedQwen3Session::pool_for_budget(config, weights, budget, BlockTokens::DEFAULT)
        .expect("pool")
        .with_prefix_caching(false);
    let mut paged = PagedQwen3Session::new(config, weights, pool).expect("paged session");
    let mut logits = paged.prefill_last_logits(seq, prompt).expect("prefill");
    for _ in 0..4 {
        logits = paged
            .decode_last_logits(seq, argmax(&logits))
            .expect("warm");
    }
    let active_before = mlx_rs::memory::active_memory().expect("active");
    mlx_rs::memory::reset_peak_memory().expect("reset peak");
    let mut steps = Vec::new();
    let mut moved = 0;
    for _ in 0..48 {
        let before = layer_addresses(&paged, config.hidden_layers);
        let started = Instant::now();
        logits = paged
            .decode_last_logits(seq, argmax(&logits))
            .expect("decode");
        steps.push(started.elapsed());
        let after = layer_addresses(&paged, config.hidden_layers);
        moved += (0..config.hidden_layers)
            .filter(|&layer| before[layer] != after[layer])
            .count();
    }
    let peak = mlx_rs::memory::peak_memory().expect("peak");
    steps.sort();
    println!(
        "paged_qwen3_06b donation round={round} pool_mib={} pool_bytes={} slabs={} \
         decode_steps=48 median_step_ms={:.3} p90_step_ms={:.3} \
         peak_minus_active_before_bytes={} layer_buffers_moved={moved}/{}",
        budget / MIB,
        paged.pool_bytes(),
        pool.slabs(),
        millis(steps[steps.len() / 2]),
        millis(steps[steps.len() * 9 / 10]),
        peak.saturating_sub(active_before),
        48 * config.hidden_layers,
    );
    assert_eq!(moved, 0, "a decode write copied its layer's pool array");
}

/// Key and value buffer addresses per layer.
fn layer_addresses(
    paged: &PagedQwen3Session<'_, std::collections::hash_map::RandomState>,
    layers: usize,
) -> Vec<(usize, usize)> {
    (0..layers)
        .map(|layer| {
            let (keys, values) = paged.layer_arrays(layer);
            (buffer_address(keys), buffer_address(values))
        })
        .collect()
}

pub(super) fn millis(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1e3
}
