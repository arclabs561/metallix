//! Paged experts against fully resident ones on a tiny MLX affine
//! `qwen3_5_moe` checkpoint: logits must be bit-identical at every budget.

use std::fs;

use checkpoint::{ReadCache, safetensors::SafetensorsCheckpoint};
use serde_json::{Value, json};

use super::{
    Qwen35Error, Qwen35Paging, Qwen35Precision, Qwen35Weights,
    mlx_affine_tests::{tiny_config, tiny_tensors, write},
};
use crate::experts::Qwen35ExpertLocator;

const EXPERTS: usize = 8;

/// The tiny dense configuration with every layer's MLP replaced by 8
/// experts, 2 per token, plus a shared expert.
fn tiny_moe_config() -> Value {
    let mut config = tiny_config(&json!({"group_size": 32, "bits": 4, "mode": "affine"}));
    config["model_type"] = json!("qwen3_5_moe");
    let text = config["text_config"].as_object_mut().expect("text config");
    text.insert("model_type".into(), json!("qwen3_5_moe_text"));
    text.remove("intermediate_size");
    text.insert("num_experts".into(), json!(EXPERTS));
    text.insert("num_experts_per_tok".into(), json!(2));
    text.insert("moe_intermediate_size".into(), json!(64));
    text.insert("shared_expert_intermediate_size".into(), json!(64));
    config
}

/// Prefill logits, then greedy decode logits for `steps` tokens.
fn run(weights: &Qwen35Weights, prompt: &[i32], steps: usize) -> Vec<Vec<f32>> {
    let mut executor = weights.executor();
    let mut out = vec![executor.prefill_last_logits(prompt).expect("prefill")];
    for _ in 0..steps {
        let last = out.last().expect("logits");
        let token = last
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(index, _)| i32::try_from(index).expect("token"))
            .expect("nonempty");
        out.push(executor.decode_last_logits(token).expect("decode"));
    }
    out
}

#[test]
fn paged_experts_match_resident_ones_bit_for_bit() {
    let config = tiny_moe_config();
    let dir = write("paged", &config, &tiny_tensors(&config));
    let resident = Qwen35Weights::load(&dir, Qwen35Precision::Checkpoint).expect("resident");
    let source = SafetensorsCheckpoint::open(&dir, ReadCache::Use).expect("table");
    let expert = Qwen35ExpertLocator::mlx_affine(&source, 2, EXPERTS)
        .expect("locator")
        .expert_bytes(0);
    let layer = expert * EXPERTS as u64;

    // 3 tokens route 6 selections (unsorted path); 40 route 80 (sorted).
    let prompts: [Vec<i32>; 2] = [vec![1, 2, 3], (0..40).map(|i| (i * 7 + 3) % 128).collect()];
    let expected: Vec<_> = prompts
        .iter()
        .map(|prompt| run(&resident, prompt, 4))
        .collect();

    for budget in [layer, 2 * layer] {
        let paged = Qwen35Weights::load_paged(
            &dir,
            Qwen35Paging {
                budget_bytes: budget,
                readers: 4,
                cache: ReadCache::Bypass,
            },
        )
        .expect("paged");
        assert!(
            !paged
                .quantized
                .keys()
                .any(|name| name.contains("switch_mlp")),
            "routed experts are not held whole"
        );
        for (prompt, expected) in prompts.iter().zip(&expected) {
            assert_eq!(&run(&paged, prompt, 4), expected, "budget {budget}");
        }
        let stats = paged.pager_stats().expect("stats");
        assert!(stats.misses > 0 && stats.hits > 0, "{stats:?}");
        assert!(stats.resident_bytes <= budget, "{stats:?}");
        assert_eq!(stats.bytes_read, stats.misses * expert, "{stats:?}");
        if budget == 2 * layer {
            // Everything fits, so each expert is read at most once.
            assert!(stats.misses <= 2 * EXPERTS as u64, "{stats:?}");
        }
    }

    assert!(matches!(
        Qwen35Weights::load_paged(
            &dir,
            Qwen35Paging {
                budget_bytes: layer - 1,
                readers: 1,
                cache: ReadCache::Use,
            },
        ),
        Err(Qwen35Error::PagerBudget { needed, .. }) if needed == layer
    ));
    fs::remove_dir_all(&dir).expect("cleanup");
}
