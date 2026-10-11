//! Writes the numbers `docs/arch/julia.typ` draws, from the shape constants
//! the encoder and head execute with, and fails when the committed
//! `docs/arch/julia.facts.json` disagrees.
//!
//! Julia-1's shape is pinned in code rather than read from a configuration,
//! so these are the constants themselves. After changing one, rerun with
//! `UPDATE_ARCH_FACTS=1` and re-render with
//! `scripts/check_arch_diagrams.py --write`.

use std::{env, fs, path::PathBuf};

use serde::Serialize;

use crate::{
    ATTENTION_HEADS, ENCODER_FF_WIDTH, FEED_FORWARD_WIDTH, HEAD_LAYERS, WIDTH,
    encoder::{
        FULL_ENCODER_LAYERS, LOCAL_WINDOW, MAX_POSITIONS, PUBLISHED_VOCAB_SIZE, attends_globally,
    },
    head::HEAD_WIDTH,
};

// Field order is the file's key order; structs keep it fixed whatever
// serde_json features the workspace enables.
#[derive(Serialize)]
struct Facts {
    #[serde(rename = "crate")]
    krate: &'static str,
    reference: Reference,
    vocab_size: u64,
    width: usize,
    attention_heads: usize,
    head_width: usize,
    max_positions: usize,
    /// `"global"` or `"local"` per encoder layer, in order.
    encoder_layers: Vec<&'static str>,
    local_window: usize,
    encoder_ffn_width: usize,
    head_layers: usize,
    head_ffn_width: usize,
}

#[derive(Serialize)]
struct Reference {
    repo: &'static str,
    revision: &'static str,
}

fn arch_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../docs/arch")
}

#[test]
fn diagram_facts_match_the_pinned_shape() {
    let facts = Facts {
        krate: "julia",
        reference: Reference {
            repo: "SupersonicLabs/Julia-1",
            revision: "a85b127321d580d65176c89ced8273f305745d85",
        },
        vocab_size: PUBLISHED_VOCAB_SIZE,
        width: WIDTH,
        attention_heads: ATTENTION_HEADS,
        head_width: HEAD_WIDTH,
        max_positions: MAX_POSITIONS,
        encoder_layers: (0..FULL_ENCODER_LAYERS)
            .map(|layer| {
                if attends_globally(layer) {
                    "global"
                } else {
                    "local"
                }
            })
            .collect(),
        local_window: LOCAL_WINDOW,
        encoder_ffn_width: ENCODER_FF_WIDTH,
        head_layers: HEAD_LAYERS,
        head_ffn_width: FEED_FORWARD_WIDTH,
    };
    let path = arch_dir().join("julia.facts.json");
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
