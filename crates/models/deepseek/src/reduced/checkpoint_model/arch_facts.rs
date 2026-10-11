//! Writes the numbers `docs/arch/deepseek_v41.typ` draws, from
//! [`V41InferenceConfig`]'s parsing of the pinned `inference/config.json`,
//! and fails when the committed `docs/arch/deepseek_v41.facts.json`
//! disagrees.
//!
//! It is a child of the configuration's module because the widths it
//! reports are private fields. After a change here or in the config
//! parsing, rerun with `UPDATE_ARCH_FACTS=1` and re-render with
//! `scripts/check_arch_diagrams.py --write`.

use std::{env, fs, path::PathBuf};

use serde::Serialize;

use super::{V41InferenceConfig, V41LayerRole};

/// `inference/config.json` of `deepseek-ai/DeepSeek-V4.1-Flash`.
const REPO: &str = "deepseek-ai/DeepSeek-V4.1-Flash";
const REVISION: &str = "dba1be0a40aa45a94ad051997016db3960a90277";
const CONFIG: &str = "configs/DeepSeek-V4.1-Flash/inference-config.json";

// Field order is the file's key order; structs keep it fixed whatever
// serde_json features the workspace enables.
#[derive(Serialize)]
struct Facts {
    #[serde(rename = "crate")]
    krate: &'static str,
    reference: Reference,
    vocab_size: usize,
    width: usize,
    /// Each layer's attention role, in order.
    layers: Vec<&'static str>,
    attention: Attention,
    hyper_connection_copies: usize,
    engram_layers: Vec<usize>,
    engram_max_ngram_size: usize,
    moe: Moe,
}

#[derive(Serialize)]
struct Reference {
    repo: &'static str,
    revision: &'static str,
    config: &'static str,
}

#[derive(Serialize)]
struct Attention {
    heads: usize,
    head_dim: usize,
    rope_head_dim: usize,
    q_lora_rank: usize,
    o_groups: usize,
    o_lora_rank: usize,
    window_size: usize,
    index_heads: usize,
    index_head_dim: usize,
    index_topk: usize,
    candidate_topk_blocks: usize,
    candidate_block_size: usize,
}

#[derive(Serialize)]
struct Moe {
    routed_experts: usize,
    activated_experts: usize,
    shared_experts: usize,
    expert_width: usize,
    score_func: String,
}

fn arch_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../docs/arch")
}

fn role_name(role: V41LayerRole) -> &'static str {
    match role {
        V41LayerRole::Startup => "startup",
        V41LayerRole::WindowOnly => "window",
        V41LayerRole::RatioTwoOwner => "ratio2_owner",
        V41LayerRole::RatioTwoConsumer { .. } => "ratio2_consumer",
        V41LayerRole::RatioOneOwner => "ratio1_owner",
        V41LayerRole::RatioOneIndexer { .. } => "ratio1_indexer",
        V41LayerRole::RatioOneConsumer { .. } => "ratio1_consumer",
    }
}

#[test]
fn diagram_facts_match_the_config_parsing() {
    let json = fs::read_to_string(arch_dir().join(CONFIG)).expect("reference config");
    let c = V41InferenceConfig::parse(&json).expect("reference config parses");
    let facts = Facts {
        krate: "deepseek",
        reference: Reference {
            repo: REPO,
            revision: REVISION,
            config: CONFIG,
        },
        vocab_size: c.vocab_size,
        width: c.dim,
        layers: (0..c.layers())
            .map(|layer| role_name(c.role(layer).expect("role")))
            .collect(),
        attention: Attention {
            heads: c.n_heads,
            head_dim: c.head_dim,
            rope_head_dim: c.rope_head_dim,
            q_lora_rank: c.q_lora_rank,
            o_groups: c.o_groups,
            o_lora_rank: c.o_lora_rank,
            window_size: c.window_size,
            index_heads: c.index_n_heads,
            index_head_dim: c.index_head_dim,
            index_topk: c.index_topk,
            candidate_topk_blocks: c.candidate_topk_blocks,
            candidate_block_size: c.candidate_block_size,
        },
        hyper_connection_copies: c.hc_mult,
        engram_layers: c.engram_layer_ids.clone(),
        engram_max_ngram_size: c.engram_max_ngram_size,
        moe: Moe {
            routed_experts: c.n_routed_experts,
            activated_experts: c.n_activated_experts,
            shared_experts: c.n_shared_experts,
            expert_width: c.moe_inter_dim,
            score_func: c.score_func.clone(),
        },
    };
    let path = arch_dir().join("deepseek_v41.facts.json");
    let expected = serde_json::to_string_pretty(&facts).expect("facts serialize") + "\n";
    if env::var_os("UPDATE_ARCH_FACTS").is_some() {
        fs::write(&path, expected).expect("write facts");
        return;
    }
    let committed = fs::read_to_string(&path).unwrap_or_default();
    assert!(
        committed == expected,
        "{} is stale; rerun this test with UPDATE_ARCH_FACTS=1, then \
         scripts/check_arch_diagrams.py --write",
        path.display()
    );
}
