//! Writes the numbers `docs/arch/qwen35.typ` draws, from [`Qwen35Config`]'s
//! parsing of one dense and one mixture-of-experts reference configuration,
//! and fails when the committed `docs/arch/qwen35.facts.json` disagrees.
//!
//! It sits inside the crate because the head counts it reports are private
//! fields. After a change here or in the config parsing, rerun with
//! `UPDATE_ARCH_FACTS=1` and re-render with
//! `scripts/check_arch_diagrams.py --write`.

use std::{env, fs, path::PathBuf};

use serde::Serialize;

use crate::{Qwen35Config, Qwen35LayerKind, Qwen35Mlp};

/// Hugging Face `config.json` files, by repository and pinned revision.
const REFERENCES: [(&str, &str, &str); 2] = [
    (
        "Qwen/Qwen3.5-0.8B",
        "2fc06364715b967f1860aea9cf38778875588b17",
        "configs/Qwen3.5-0.8B.json",
    ),
    (
        "Qwen/Qwen3.6-35B-A3B",
        "995ad96eacd98c81ed38be0c5b274b04031597b0",
        "configs/Qwen3.6-35B-A3B.json",
    ),
];

// Field order is the file's key order; structs keep it fixed whatever
// serde_json features the workspace enables.
#[derive(Serialize)]
struct Facts {
    #[serde(rename = "crate")]
    krate: &'static str,
    models: Vec<Model>,
}

#[derive(Serialize)]
struct Model {
    repo: &'static str,
    revision: &'static str,
    config: &'static str,
    /// `"linear"` or `"full"` per decoder layer, in order.
    layers: Vec<&'static str>,
    hidden_size: usize,
    vocab_size: usize,
    attention_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    rotary_dim: usize,
    linear_key_heads: usize,
    linear_value_heads: usize,
    linear_key_head_dim: usize,
    linear_value_head_dim: usize,
    conv_kernel: usize,
    mlp: Mlp,
    tied_output_embedding: bool,
    mtp_layers: usize,
    max_position_embeddings: usize,
}

#[derive(Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum Mlp {
    Dense {
        intermediate_size: usize,
    },
    Experts {
        experts: usize,
        top_k: usize,
        expert_intermediate_size: usize,
        shared_intermediate_size: usize,
    },
}

fn arch_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../docs/arch")
}

fn model((repo, revision, config): (&'static str, &'static str, &'static str)) -> Model {
    let json = fs::read_to_string(arch_dir().join(config)).expect("reference config");
    let parsed = Qwen35Config::parse(&json).expect("reference config parses");
    let layers = parsed
        .layers()
        .iter()
        .map(|kind| match kind {
            Qwen35LayerKind::LinearAttention => "linear",
            Qwen35LayerKind::FullAttention => "full",
        })
        .collect();
    let mlp = match parsed.mlp() {
        Qwen35Mlp::Dense { intermediate_size } => Mlp::Dense { intermediate_size },
        Qwen35Mlp::Experts {
            experts,
            top_k,
            expert_intermediate_size,
            shared_intermediate_size,
        } => Mlp::Experts {
            experts,
            top_k,
            expert_intermediate_size,
            shared_intermediate_size,
        },
    };
    Model {
        repo,
        revision,
        config,
        layers,
        hidden_size: parsed.hidden_size,
        vocab_size: parsed.vocab_size,
        attention_heads: parsed.attention_heads,
        kv_heads: parsed.key_value_heads,
        head_dim: parsed.head_dim,
        rotary_dim: parsed.rotary_dim,
        linear_key_heads: parsed.linear_key_heads,
        linear_value_heads: parsed.linear_value_heads,
        linear_key_head_dim: parsed.linear_key_head_dim,
        linear_value_head_dim: parsed.linear_value_head_dim,
        conv_kernel: parsed.conv_kernel,
        mlp,
        tied_output_embedding: parsed.tie_word_embeddings,
        mtp_layers: parsed.mtp_layers,
        max_position_embeddings: parsed.max_position_embeddings,
    }
}

#[test]
fn diagram_facts_match_the_config_parsing() {
    let facts = Facts {
        krate: "qwen35",
        models: REFERENCES.into_iter().map(model).collect(),
    };
    let path = arch_dir().join("qwen35.facts.json");
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
