//! Writes the numbers `docs/arch/qwen3.typ` draws, from this crate's own
//! parsing of the reference configuration, and fails when the committed
//! `docs/arch/qwen3.facts.json` disagrees.
//!
//! After a change here or in the config parsing, rerun with
//! `UPDATE_ARCH_FACTS=1` and re-render with
//! `scripts/check_arch_diagrams.py --write`.

use std::{env, fs, path::PathBuf};

use qwen::{DecoderFamily, Qwen3Attention, Qwen3TextContract, preflight::Qwen3ExecutionPreflight};
use serde::Serialize;

/// Hugging Face `config.json` of `Qwen/Qwen3-0.6B`.
const REPO: &str = "Qwen/Qwen3-0.6B";
const REVISION: &str = "c1899de289a04d12100db370d81485cdf75e47ca";
const CONFIG: &str = "configs/Qwen3-0.6B.json";

/// Model types this decoder runs, each with the bidirectional flag its
/// published configurations carry.
const VARIANTS: [(&str, bool); 4] = [
    ("qwen3", false),
    ("llama", false),
    ("qwen2", false),
    ("bidirectional_pplx_qwen3", true),
];

// Field order is the file's key order; a struct keeps it fixed whatever
// serde_json features the workspace enables.
#[derive(Serialize)]
struct Facts {
    #[serde(rename = "crate")]
    krate: &'static str,
    reference: Reference,
    model_type: &'static str,
    layers: u32,
    hidden_size: u32,
    vocab_size: u32,
    attention_heads: u32,
    kv_heads: u32,
    head_dim: u32,
    max_position_embeddings: u32,
    kv_bytes_per_token_bf16: u64,
    variants: Vec<Variant>,
}

#[derive(Serialize)]
struct Reference {
    repo: &'static str,
    revision: &'static str,
    config: &'static str,
}

#[derive(Serialize)]
struct Variant {
    model_type: &'static str,
    qk_norm: bool,
    qkv_bias: bool,
    attention: &'static str,
}

fn arch_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../docs/arch")
}

fn facts() -> Facts {
    let json = fs::read_to_string(arch_dir().join(CONFIG)).expect("reference config");
    let contract = Qwen3TextContract::parse(&json).expect("reference config parses");
    let preflight = Qwen3ExecutionPreflight::from_contract(&contract).expect("preflight");
    let variants = VARIANTS
        .iter()
        .map(|&(model_type, bidirectional)| {
            let family = DecoderFamily::from_model_type(model_type).expect("known family");
            let attention = match Qwen3Attention::from_config(model_type, bidirectional) {
                Some(Qwen3Attention::Causal) => "causal",
                Some(Qwen3Attention::Bidirectional) => "bidirectional",
                None => panic!("{model_type} has no attention layout"),
            };
            Variant {
                model_type,
                qk_norm: family.has_qk_norm(),
                qkv_bias: family.has_qkv_bias(),
                attention,
            }
        })
        .collect();
    assert_eq!(contract.family(), DecoderFamily::Qwen3);
    Facts {
        krate: "qwen",
        reference: Reference {
            repo: REPO,
            revision: REVISION,
            config: CONFIG,
        },
        model_type: "qwen3",
        layers: contract.total_layers(),
        hidden_size: contract.hidden_size(),
        vocab_size: contract.vocab_size(),
        attention_heads: contract.attention_heads(),
        kv_heads: preflight.key_value_heads(),
        head_dim: preflight.head_dim(),
        max_position_embeddings: preflight.position_capacity(),
        kv_bytes_per_token_bf16: preflight.kv_bytes_per_token_bf16(),
        variants,
    }
}

#[test]
fn diagram_facts_match_the_config_parsing() {
    let path = arch_dir().join("qwen3.facts.json");
    let expected = serde_json::to_string_pretty(&facts()).expect("facts serialize") + "\n";
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
