//! `EngramHashState::write_and_hash` per token.
//!
//! Workload `real` parses `.agents/receipts/engram-hash/v41-engram-inputs.bin`
//! (both V4.1 Engram layers) and hashes the 17-token `parity-shell` prompt;
//! otherwise `fixture` uses the checked-in hash fixture's explicit layout with
//! the same token IDs reduced into its token map. The workload in use is
//! printed to stderr.
//!
//! - `decode_one`: one token at position 16 after a 16-token prefill, at a
//!   history capacity of `capacity` positions (the call copies the history).
//! - `prefill`: the whole 17-token chunk at position 0.

use std::{hint::black_box, path::Path, sync::LazyLock};

use deepseek::engram::{
    CompressedToken, EngramHashLayout, EngramHashState,
    inputs::{EngramHashInputs, V41_ENGRAM_INPUTS_IDENTITY},
};
use serde_json::Value;

const TOKENS: usize = 17;

struct Workload {
    name: &'static str,
    layout: EngramHashLayout,
    tokens: Vec<CompressedToken>,
}

fn i64s(value: &Value) -> Vec<i64> {
    match value {
        Value::Array(items) => items.iter().flat_map(i64s).collect(),
        other => vec![other.as_i64().expect("integer")],
    }
}

fn prompt_ids(repo: &Path) -> Vec<i64> {
    let shell = repo.join(".agents/receipts/route-trace/parity-shell.json");
    std::fs::read(shell)
        .ok()
        .and_then(|bytes| serde_json::from_slice::<Value>(&bytes).ok())
        .map(|run| i64s(&run["runs"][0]["prompt_ids"]))
        .filter(|ids| ids.len() == TOKENS)
        .unwrap_or_else(|| {
            (0..TOKENS)
                .map(|i| i64::try_from(i * 7_919 + 3).expect("small"))
                .collect()
        })
}

fn real(repo: &Path, ids: &[i64]) -> Option<Workload> {
    let bytes =
        std::fs::read(repo.join(".agents/receipts/engram-hash/v41-engram-inputs.bin")).ok()?;
    let inputs = EngramHashInputs::parse(&bytes, &V41_ENGRAM_INPUTS_IDENTITY).ok()?;
    Some(Workload {
        name: "real V4.1 inputs artifact, layers [1, 14], parity-shell prompt",
        layout: inputs.hash_layout().expect("real layout"),
        tokens: inputs.compress(ids).expect("prompt IDs"),
    })
}

fn fixture(ids: &[i64]) -> Workload {
    let fixture: Value = serde_json::from_str(include_str!(
        "../../../../fixtures/deepseek-v41/engram-hash-reference.json"
    ))
    .expect("checked-in Engram fixture");
    let tensor = |name: &str| {
        let entry = fixture["tensors"]
            .as_array()
            .expect("tensors")
            .iter()
            .find(|tensor| tensor["name"] == name)
            .unwrap_or_else(|| panic!("fixture tensor {name}"));
        (i64s(&entry["values"]), entry["shape"].clone())
    };
    let (primes, shape) = tensor("primes");
    let [layers, ngrams, heads] =
        [0, 1, 2].map(|axis| usize::try_from(shape[axis].as_u64().expect("shape")).expect("small"));
    let token_map = tensor("token_map").0;
    let layout = EngramHashLayout::new(
        ngrams + 1,
        heads,
        layers,
        fixture["explicit_state"]["compressed_pad_id"]
            .as_i64()
            .expect("pad"),
        primes,
        tensor("offsets").0,
        tensor("multipliers").0,
    )
    .expect("fixture layout");
    let tokens = ids
        .iter()
        .map(|&id| {
            CompressedToken::Live(
                token_map[usize::try_from(id.unsigned_abs()).expect("small") % token_map.len()],
            )
        })
        .collect();
    Workload {
        name: "checked-in fixture layout (artifact absent)",
        layout,
        tokens,
    }
}

static WORKLOAD: LazyLock<Workload> = LazyLock::new(|| {
    let repo = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../..");
    let ids = prompt_ids(&repo);
    let workload = real(&repo, &ids).unwrap_or_else(|| fixture(&ids));
    eprintln!("engram_hash workload: {}", workload.name);
    workload
});

#[divan::bench(args = [32, 4096])]
fn decode_one(bencher: divan::Bencher, capacity: usize) {
    let w = &*WORKLOAD;
    let mut state = EngramHashState::new(w.layout.clone(), 1, capacity).expect("state");
    state
        .write_and_hash(&w.tokens[..TOKENS - 1], TOKENS - 1, 0)
        .expect("prefill");
    let last = &w.tokens[TOKENS - 1..];
    bencher.bench_local(|| {
        black_box(
            state
                .write_and_hash(black_box(last), 1, TOKENS - 1)
                .expect("decode"),
        )
    });
}

#[divan::bench]
fn prefill(bencher: divan::Bencher) {
    let w = &*WORKLOAD;
    let mut state = EngramHashState::new(w.layout.clone(), 1, TOKENS).expect("state");
    bencher
        .counter(divan::counter::ItemsCount::new(TOKENS))
        .bench_local(|| {
            black_box(
                state
                    .write_and_hash(black_box(&w.tokens), TOKENS, 0)
                    .expect("prefill"),
            )
        });
}

fn main() {
    LazyLock::force(&WORKLOAD);
    divan::main();
}
