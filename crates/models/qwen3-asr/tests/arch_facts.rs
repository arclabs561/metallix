//! Writes the numbers `docs/arch/qwen3_asr.typ` draws, from this crate's
//! parsing of the reference configuration and the qwen crate's parsing of
//! the decoder it hands on, and fails when the committed
//! `docs/arch/qwen3_asr.facts.json` disagrees.
//!
//! After a change here or in the config parsing, rerun with
//! `UPDATE_ARCH_FACTS=1` and re-render with
//! `scripts/check_arch_diagrams.py --write`.

use std::{env, fs, path::PathBuf};

use qwen::Qwen3TextContract;
use qwen3_asr::{
    Qwen3AsrConfig,
    config::{CHUNK_FRAMES, FRAMES_PER_CHUNK, MEL_BINS},
};
use serde::Serialize;

/// Hugging Face `config.json` of `Qwen/Qwen3-ASR-0.6B`.
const REPO: &str = "Qwen/Qwen3-ASR-0.6B";
const REVISION: &str = "5eb144179a02acc5e5ba31e748d22b0cf3e303b0";
const CONFIG: &str = "configs/Qwen3-ASR-0.6B.json";

// Field order is the file's key order; structs keep it fixed whatever
// serde_json features the workspace enables.
#[derive(Serialize)]
struct Facts {
    #[serde(rename = "crate")]
    krate: &'static str,
    reference: Reference,
    encoder: Encoder,
    decoder: Decoder,
}

#[derive(Serialize)]
struct Reference {
    repo: &'static str,
    revision: &'static str,
    config: &'static str,
}

#[derive(Serialize)]
struct Encoder {
    mel_bins: usize,
    chunk_frames: usize,
    frames_per_chunk: usize,
    conv_channels: usize,
    width: usize,
    layers: usize,
    heads: usize,
    ffn_width: usize,
    window_frames: usize,
    output_width: usize,
}

#[derive(Serialize)]
struct Decoder {
    layers: u32,
    hidden_size: u32,
    vocab_size: u32,
    attention_heads: u32,
    kv_heads: u32,
    head_dim: u32,
    intermediate_size: u32,
    tied_output_embedding: bool,
}

fn arch_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../docs/arch")
}

fn facts() -> Facts {
    let json = fs::read_to_string(arch_dir().join(CONFIG)).expect("reference config");
    let config = Qwen3AsrConfig::parse(&json).expect("reference config parses");
    let encoder = config.encoder();
    let decoder =
        Qwen3TextContract::parse(&config.text_config_json()).expect("decoder config parses");
    Facts {
        krate: "qwen3-asr",
        reference: Reference {
            repo: REPO,
            revision: REVISION,
            config: CONFIG,
        },
        encoder: Encoder {
            mel_bins: MEL_BINS,
            chunk_frames: CHUNK_FRAMES,
            frames_per_chunk: FRAMES_PER_CHUNK,
            conv_channels: encoder.conv_channels,
            width: encoder.d_model,
            layers: encoder.layers,
            heads: encoder.heads,
            ffn_width: encoder.ffn_dim,
            window_frames: encoder.window_frames(),
            output_width: encoder.output_dim,
        },
        decoder: Decoder {
            layers: decoder.total_layers(),
            hidden_size: decoder.hidden_size(),
            vocab_size: decoder.vocab_size(),
            attention_heads: decoder.attention_heads(),
            kv_heads: decoder.key_value_heads(),
            head_dim: decoder.head_dim(),
            intermediate_size: decoder
                .intermediate_size()
                .expect("reference decoder states intermediate_size")
                .get(),
            tied_output_embedding: decoder.tied_output_embedding(),
        },
    }
}

#[test]
fn diagram_facts_match_the_config_parsing() {
    let path = arch_dir().join("qwen3_asr.facts.json");
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
