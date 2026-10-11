//! Writes the numbers `docs/arch/gemma4.typ` draws, from this crate's own
//! parsing of the reference configurations, and fails when the committed
//! `docs/arch/gemma4.facts.json` disagrees.
//!
//! After a change here or in the config parsing, rerun with
//! `UPDATE_ARCH_FACTS=1` and re-render with
//! `scripts/check_arch_diagrams.py --write`.

use std::{env, fs, path::PathBuf};

use gemma::{Gemma4Attention, Gemma4LayerKind, Gemma4Rope, Gemma4TextConfig};
use serde::Serialize;

/// Hugging Face `config.json` files, by repository and pinned revision: a
/// plain dense model, and the smallest with per-layer inputs and shared K/V.
const REFERENCES: [(&str, &str, &str); 2] = [
    (
        "google/gemma-4-12B-it",
        "707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7",
        "configs/gemma-4-12B-it.json",
    ),
    (
        "google/gemma-4-E2B-it",
        "3e22461f65e89153144f8adb70e3b8c2cc9845a7",
        "configs/gemma-4-E2B-it.json",
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
    /// `"sliding"` or `"full"` per decoder layer, with `"_shared"` appended
    /// when the layer attends over an earlier layer's K/V.
    layers: Vec<String>,
    hidden_size: usize,
    vocab_size: usize,
    intermediate_size: usize,
    /// MLP width of the K/V-sharing layers, when it differs.
    shared_intermediate_size: Option<usize>,
    sliding_window: usize,
    final_logit_softcapping: Option<f32>,
    sliding: Attention,
    full: Attention,
    per_layer_input_width: Option<usize>,
}

#[derive(Serialize)]
struct Attention {
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    value_from_key: bool,
    rope: &'static str,
    rotated_pairs: usize,
}

fn attention(geometry: &Gemma4Attention) -> Attention {
    let (rope, rotated_pairs) = match geometry.rope {
        Gemma4Rope::Default { .. } => ("default", geometry.head_dim / 2),
        Gemma4Rope::Proportional { rotated_pairs, .. } => ("proportional", rotated_pairs),
    };
    Attention {
        heads: geometry.heads,
        kv_heads: geometry.kv_heads,
        head_dim: geometry.head_dim,
        value_from_key: geometry.value_from_key,
        rope,
        rotated_pairs,
    }
}

fn arch_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../docs/arch")
}

fn model((repo, revision, config): (&'static str, &'static str, &'static str)) -> Model {
    let json = fs::read_to_string(arch_dir().join(config)).expect("reference config");
    let parsed = Gemma4TextConfig::parse(&json).expect("reference config parses");
    let layers = parsed
        .layers()
        .iter()
        .enumerate()
        .map(|(index, kind)| {
            let kind = match kind {
                Gemma4LayerKind::Sliding => "sliding",
                Gemma4LayerKind::Full => "full",
            };
            match parsed.kv_source(index) {
                Some(_) => format!("{kind}_shared"),
                None => kind.to_owned(),
            }
        })
        .collect();
    let shared_intermediate_size = (0..parsed.hidden_layers())
        .find(|&layer| parsed.kv_source(layer).is_some())
        .map(|layer| parsed.mlp_width(layer))
        .filter(|&width| width != parsed.intermediate_size());
    Model {
        repo,
        revision,
        config,
        layers,
        hidden_size: parsed.hidden_size(),
        vocab_size: parsed.vocab_size(),
        intermediate_size: parsed.intermediate_size(),
        shared_intermediate_size,
        sliding_window: parsed.sliding_window(),
        final_logit_softcapping: parsed.final_logit_softcapping(),
        sliding: attention(parsed.attention(Gemma4LayerKind::Sliding)),
        full: attention(parsed.attention(Gemma4LayerKind::Full)),
        per_layer_input_width: parsed.per_layer_input_width(),
    }
}

#[test]
fn diagram_facts_match_the_config_parsing() {
    let facts = Facts {
        krate: "gemma",
        models: REFERENCES.into_iter().map(model).collect(),
    };
    let path = arch_dir().join("gemma4.facts.json");
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
