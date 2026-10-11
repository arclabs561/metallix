//! Writes the numbers `docs/arch/flux2.typ` draws, from this crate's parsing
//! of the reference transformer configuration and the qwen crate's parsing
//! of its text encoder, and fails when the committed
//! `docs/arch/flux2.facts.json` disagrees.
//!
//! After a change here or in the config parsing, rerun with
//! `UPDATE_ARCH_FACTS=1` and re-render with
//! `scripts/check_arch_diagrams.py --write`.

use std::{env, fs, path::PathBuf};

use flux2::{Flux2TransformerConfig, PIXELS_PER_TOKEN};
use qwen::Qwen3TextContract;
use serde::Serialize;

/// The diffusers checkpoint `black-forest-labs/FLUX.2-klein-4B`.
const REPO: &str = "black-forest-labs/FLUX.2-klein-4B";
const REVISION: &str = "e7b7dc27f91deacad38e78976d1f2b499d76a294";
const TRANSFORMER: &str = "configs/FLUX.2-klein-4B/transformer.json";
const TEXT_ENCODER: &str = "configs/FLUX.2-klein-4B/text_encoder.json";

// Field order is the file's key order; structs keep it fixed whatever
// serde_json features the workspace enables.
#[derive(Serialize)]
struct Facts {
    #[serde(rename = "crate")]
    krate: &'static str,
    reference: Reference,
    pixels_per_token: u32,
    text_encoder: TextEncoder,
    transformer: Transformer,
}

#[derive(Serialize)]
struct Reference {
    repo: &'static str,
    revision: &'static str,
    configs: [&'static str; 2],
}

#[derive(Serialize)]
struct TextEncoder {
    layers: u32,
    hidden_size: u32,
}

#[derive(Serialize)]
struct Transformer {
    double_blocks: usize,
    single_blocks: usize,
    heads: usize,
    head_dim: usize,
    width: usize,
    mlp_hidden: usize,
    latent_channels: usize,
    text_channels: usize,
    rope_axes: [usize; 4],
    timestep_channels: usize,
}

fn arch_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../docs/arch")
}

fn read(config: &str) -> String {
    fs::read_to_string(arch_dir().join(config)).expect("reference config")
}

fn facts() -> Facts {
    let transformer =
        Flux2TransformerConfig::from_json(&read(TRANSFORMER)).expect("transformer config parses");
    let text = Qwen3TextContract::parse(&read(TEXT_ENCODER)).expect("text encoder config parses");
    Facts {
        krate: "flux2",
        reference: Reference {
            repo: REPO,
            revision: REVISION,
            configs: [TRANSFORMER, TEXT_ENCODER],
        },
        pixels_per_token: PIXELS_PER_TOKEN,
        text_encoder: TextEncoder {
            layers: text.total_layers(),
            hidden_size: text.hidden_size(),
        },
        transformer: Transformer {
            double_blocks: transformer.double_blocks(),
            single_blocks: transformer.single_blocks(),
            heads: transformer.heads(),
            head_dim: transformer.head_dim(),
            width: transformer.inner_dim(),
            mlp_hidden: transformer.mlp_hidden(),
            latent_channels: transformer.in_channels(),
            text_channels: transformer.joint_attention_dim(),
            rope_axes: transformer.axes_dims_rope(),
            timestep_channels: transformer.timestep_channels(),
        },
    }
}

#[test]
fn diagram_facts_match_the_config_parsing() {
    let path = arch_dir().join("flux2.facts.json");
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
