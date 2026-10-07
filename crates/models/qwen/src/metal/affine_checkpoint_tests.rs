//! Opt-in MLX affine-quantization checks against local checkpoints.
//!
//! `METALLIX_QWEN_MODEL` names the BF16 `Qwen/Qwen3-0.6B`;
//! `METALLIX_QWEN_AFFINE_MODEL` an mlx-community conversion of a Qwen3, and
//! `METALLIX_QWEN_AFFINE_FIXTURE` its `mlx-lm-greedy.json` (default
//! `fixtures/qwen3-0.6b-4bit/`). Under `fixtures/qwen3-0.6b-{4,6,8}bit/`,
//! `mlx-quantize.json` fingerprints `mx.quantize` of the BF16 checkpoint
//! (`scripts/qwen3-mlx-quantize-reference.py`) and `mlx-lm-greedy.json`
//! holds mlx-lm's greedy decoding of the published conversion
//! (`scripts/qwen3-mlx-greedy-reference.py`); `fixtures/qwen3-8b-4bit/`
//! holds the latter for `mlx-community/Qwen3-8B-4bit`.
//!
//! Gates:
//! - quantizing the BF16 checkpoint at load reproduces MLX's own `quantize`
//!   bit for bit, by fingerprint, for 4, 6 and 8 bits. The fingerprints were
//!   taken with the MLX release mlx-sys bundles, and the test refuses to
//!   compare against any other. The published conversions are not the
//!   oracle: they came from an older MLX whose scale rounding differs;
//! - teacher-forced on mlx-lm's tokens, the GPU argmax picks mlx-lm's token
//!   at every step whose recorded top-1/top-2 gap is at least
//!   `NEAR_TIE_GAP`, and at no fewer than `MINIMUM_AGREEMENT` of all steps.

use std::{collections::BTreeMap, env, fs, path::PathBuf};

use mlx_rs::{Array, Dtype};
use serde::Deserialize;

use super::{Qwen3MlxWeights, Qwen3WeightPrecision};
use crate::{
    GPU_TEST_LOCK,
    forward::{Qwen3AffineQuantization, Qwen3FloatPrecision},
};

/// Logit units, as in the BF16-against-float32 agreement rule.
const NEAR_TIE_GAP: f32 = 0.5;
const MINIMUM_AGREEMENT: f64 = 0.99;
const CONTEXT_TOKENS: usize = 1_024;
const KV_BYTES: u64 = 1024 * 1024 * 1024;

fn model(variable: &str) -> PathBuf {
    env::var_os(variable).map_or_else(
        || panic!("{variable} is required for this ignored checkpoint test"),
        PathBuf::from,
    )
}

fn fixture_path(relative: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../../fixtures")
        .join(relative)
}

#[derive(Debug, Deserialize, PartialEq)]
struct Fingerprint {
    count: u64,
    sum: u64,
    weighted: u64,
}

/// Wrapping sums of a tensor's raw bit patterns, plain and weighted by
/// 1-based position, as the reference script computes them.
fn fingerprint(array: &Array) -> Fingerprint {
    let words: Vec<u64> = if array.dtype() == Dtype::Uint32 {
        array.eval().expect("eval");
        array
            .as_slice::<u32>()
            .iter()
            .map(|&word| u64::from(word))
            .collect()
    } else {
        let bits = array.view::<u16>().expect("bit view");
        bits.eval().expect("eval");
        bits.as_slice::<u16>()
            .iter()
            .map(|&half| u64::from(half))
            .collect()
    };
    let (mut sum, mut weighted) = (0_u64, 0_u64);
    for (position, &value) in (1_u64..).zip(&words) {
        sum = sum.wrapping_add(value);
        weighted = weighted.wrapping_add(value.wrapping_mul(position));
    }
    Fingerprint {
        count: u64::try_from(words.len()).expect("count fits u64"),
        sum,
        weighted,
    }
}

#[derive(Deserialize)]
struct QuantizeFixture {
    mlx: String,
    tensors: BTreeMap<String, Fingerprint>,
}

/// The MLX release each `mlx-sys` version builds (the `GIT_TAG` of its
/// `mlx-c` build). A new `mlx-sys` needs new fingerprints, not a silent
/// compare.
const BUNDLED_MLX: [(&str, &str); 1] = [("0.6.0", "0.32.2")];

/// The MLX release this build links, from the workspace lockfile.
fn bundled_mlx() -> &'static str {
    let lock =
        fs::read_to_string(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../Cargo.lock"))
            .expect("workspace Cargo.lock");
    let mut blocks = lock.split("[[package]]");
    let mlx_sys = blocks
        .find(|block| block.contains("name = \"mlx-sys\""))
        .expect("mlx-sys in Cargo.lock");
    let version = mlx_sys
        .lines()
        .find_map(|line| line.strip_prefix("version = \""))
        .and_then(|rest| rest.strip_suffix('"'))
        .expect("mlx-sys version");
    BUNDLED_MLX
        .iter()
        .find(|(sys, _)| *sys == version)
        .map_or_else(
            || panic!("no MLX release recorded for mlx-sys {version}; refingerprint"),
            |(_, mlx)| *mlx,
        )
}

#[test]
#[ignore = "requires METALLIX_QWEN_MODEL on Apple-Silicon Metal"]
fn quantizing_bf16_at_load_reproduces_mlx_quantize() {
    let mlx = bundled_mlx();
    let _gpu = GPU_TEST_LOCK.lock().expect("GPU test lock");
    for bits in [4_u64, 6, 8] {
        let fixture: QuantizeFixture = serde_json::from_str(
            &fs::read_to_string(fixture_path(&format!(
                "qwen3-0.6b-{bits}bit/mlx-quantize.json"
            )))
            .expect("fixture"),
        )
        .expect("fixture JSON");
        assert_eq!(
            fixture.mlx, mlx,
            "fingerprints from MLX {} cannot check the MLX {mlx} this build links",
            fixture.mlx
        );
        let mut weights = Qwen3MlxWeights::load(model("METALLIX_QWEN_MODEL")).expect("BF16 load");
        weights
            .prepare_precision(Qwen3WeightPrecision::Affine {
                quantization: Qwen3AffineQuantization::from_parameters(bits, 64)
                    .expect("supported layout"),
                activations: Qwen3FloatPrecision::BFloat16,
            })
            .expect("quantize at load");
        let differing: Vec<&String> = fixture
            .tensors
            .iter()
            .filter(|(name, expected)| {
                let actual = weights
                    .tensors
                    .get(*name)
                    .expect("every fingerprinted tensor");
                fingerprint(actual) != **expected
            })
            .map(|(name, _)| name)
            .collect();
        let packed = weights
            .tensors
            .keys()
            .filter(|name| name.ends_with(".scales"))
            .count();
        println!(
            "affine_checkpoint quantize_at_load bits={bits} mlx={} tensors={} differing={}",
            fixture.mlx,
            fixture.tensors.len(),
            differing.len()
        );
        // Every packed projection is fingerprinted: three tensors per scale.
        assert_eq!(packed * 3, fixture.tensors.len());
        assert!(
            differing.is_empty(),
            "{bits}-bit tensors that differ: {differing:?}"
        );
    }
}

#[derive(Deserialize)]
struct Fixture {
    mlx: String,
    mlx_lm: String,
    cases: Vec<Case>,
}

#[derive(Deserialize)]
struct Case {
    prompt_ids: Vec<i32>,
    tokens: Vec<i32>,
    top2_gaps: Vec<f32>,
}

#[test]
#[ignore = "requires METALLIX_QWEN_AFFINE_MODEL on Apple-Silicon Metal"]
fn teacher_forced_greedy_agrees_with_mlx_lm_outside_near_ties() {
    // `METALLIX_QWEN_AFFINE_FIXTURE` selects another model's fixture, such as
    // `fixtures/qwen3-8b-4bit/mlx-lm-greedy.json`.
    let path = env::var_os("METALLIX_QWEN_AFFINE_FIXTURE").map_or_else(
        || fixture_path("qwen3-0.6b-4bit/mlx-lm-greedy.json"),
        PathBuf::from,
    );
    let fixture: Fixture =
        serde_json::from_str(&fs::read_to_string(&path).expect("fixture")).expect("fixture JSON");
    let _gpu = GPU_TEST_LOCK.lock().expect("GPU test lock");
    let weights =
        Qwen3MlxWeights::load(model("METALLIX_QWEN_AFFINE_MODEL")).expect("quantized load");
    assert!(matches!(
        weights.precision().expect("precision"),
        Qwen3WeightPrecision::Affine { .. }
    ));
    let (mut steps, mut agreed) = (0_u32, 0_u32);
    let mut decisive_divergences = Vec::new();
    for (index, case) in fixture.cases.iter().enumerate() {
        let mut executor = weights
            .resident_chat_executor(CONTEXT_TOKENS, KV_BYTES)
            .expect("executor");
        let logits = executor
            .prefill_last_logits(&case.prompt_ids)
            .expect("prefill");
        let mut picks = vec![argmax(&logits)];
        for &token in &case.tokens[..case.tokens.len() - 1] {
            picks.push(
                executor
                    .decode_greedy(token)
                    .expect("decode")
                    .wait_one()
                    .expect("finite logits"),
            );
        }
        for (step, ((&pick, &expected), &gap)) in picks
            .iter()
            .zip(&case.tokens)
            .zip(&case.top2_gaps)
            .enumerate()
        {
            steps += 1;
            if pick == expected {
                agreed += 1;
            } else {
                println!(
                    "affine_checkpoint divergence case={index} step={step} \
                     mlx_lm={expected} metallix={pick} mlx_lm_gap={gap:.4}"
                );
                if gap >= NEAR_TIE_GAP {
                    decisive_divergences.push((index, step, gap));
                }
            }
        }
    }
    let agreement = f64::from(agreed) / f64::from(steps);
    println!(
        "affine_checkpoint mlx={} mlx_lm={} steps={steps} agreed={agreed} \
         agreement={agreement:.4} near_tie_gap={NEAR_TIE_GAP}",
        fixture.mlx, fixture.mlx_lm
    );
    assert!(
        decisive_divergences.is_empty(),
        "diverged outside near ties: {decisive_divergences:?}"
    );
    assert!(agreement >= MINIMUM_AGREEMENT);
}

fn argmax(logits: &[f32]) -> i32 {
    let mut best = 0;
    for (index, &value) in logits.iter().enumerate() {
        if value > logits[best] {
            best = index;
        }
    }
    i32::try_from(best).expect("vocabulary fits i32")
}
