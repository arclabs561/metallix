//! Bounded fixture qualification for the V4.1 FP32 Metal score reduction.

use std::{fs::File, io::Read, num::NonZeroUsize, path::Path, process::ExitCode, time::Instant};

use serde::Deserialize;
use serde_json::json;

use crate::parity::compare_index_scores;

const MAX_FIXTURE_BYTES: u64 = 4 * 1024 * 1024;
const REVISION: &str = "dba1be0a40aa45a94ad051997016db3960a90277";
const SOURCE_SHA256: &str = "4e9ae23620edc8028ccc5d5fef552ab7fdc7dcd6f79608754fe9f67644056f65";

#[derive(Deserialize)]
struct Fixture {
    schema_version: u8,
    source: Source,
    reference: Reference,
    cases: Vec<Case>,
}

#[derive(Deserialize)]
struct Source {
    revision: String,
    sha256: String,
    symbol: String,
}

#[derive(Deserialize)]
struct Reference {
    device: String,
    dtype: String,
}

#[derive(Deserialize)]
struct Case {
    name: String,
    head_dim: NonZeroUsize,
    query: Vec<f32>,
    keys: Vec<f32>,
    head_weights: Vec<f32>,
    expected_scores: Vec<f32>,
}

fn parse_fixture(bytes: &[u8]) -> Result<Fixture, Box<dyn std::error::Error>> {
    let fixture: Fixture = serde_json::from_slice(bytes)?;
    if fixture.schema_version != 1
        || fixture.source.revision != REVISION
        || fixture.source.sha256 != SOURCE_SHA256
        || fixture.source.symbol != "Indexer.forward:index_score"
        || fixture.reference.device != "cpu"
        || fixture.reference.dtype != "float32"
    {
        return Err("fixture must identify the pinned CPU float32 index-score reference".into());
    }
    if fixture.cases.is_empty() || fixture.cases.len() > 100 {
        return Err("fixture must contain 1..100 cases".into());
    }
    Ok(fixture)
}

pub(crate) fn run(path: &Path, repeats: u32) -> ExitCode {
    match check(path, repeats) {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => ExitCode::FAILURE,
        Err(error) => {
            eprintln!("V4.1 index-score qualification failed: {error}");
            ExitCode::FAILURE
        }
    }
}

fn check(path: &Path, repeats: u32) -> Result<bool, Box<dyn std::error::Error>> {
    let mut bytes = Vec::new();
    File::open(path)?
        .take(MAX_FIXTURE_BYTES + 1)
        .read_to_end(&mut bytes)?;
    if u64::try_from(bytes.len())? > MAX_FIXTURE_BYTES {
        return Err("index-score fixture exceeds 4 MiB".into());
    }
    let fixture = parse_fixture(&bytes)?;
    let mut results = Vec::new();
    let mut passed = true;
    for case in fixture.cases {
        let evaluate = || {
            deepseek::index_scores_f32(&case.query, &case.keys, &case.head_weights, case.head_dim)
        };
        let warmup_started = Instant::now();
        let warmup = evaluate()?;
        let warmup_ms = warmup_started.elapsed().as_secs_f64() * 1000.0;
        let warmup_comparison = compare_index_scores(&warmup, &case.expected_scores)?;
        passed &= warmup_comparison.passed();
        let mut timings = Vec::new();
        let mut comparisons = Vec::new();
        for _ in 0..repeats {
            let started = Instant::now();
            let scores = evaluate()?;
            timings.push(started.elapsed().as_secs_f64() * 1000.0);
            let comparison = compare_index_scores(&scores, &case.expected_scores)?;
            passed &= comparison.passed();
            comparisons.push(comparison);
        }
        results.push(json!({
            "name": case.name,
            "heads": case.head_weights.len(),
            "head_dim": case.head_dim.get(),
            "positions": case.expected_scores.len(),
            "excluded_warmup_ms": warmup_ms,
            "warmup_comparison": warmup_comparison,
            "evaluation_ms": timings,
            "comparisons": comparisons,
        }));
    }
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "schema_version": 1,
            "operation": "v41_fp32_index_score_qualification",
            "passed": passed,
            "source_revision": REVISION,
            "source_sha256": SOURCE_SHA256,
            "cases": results,
            "scope": "One-query synthetic FP32 inputs; validation, array creation, GPU graph and readback included in timings. No FP4/BF16 qualification, causal mask, Top-K, model weights, or serving throughput. Fixture provenance fields are checked, not authenticated."
        }))?
    );
    Ok(passed)
}

#[cfg(test)]
mod tests {
    use super::parse_fixture;

    #[test]
    fn refuses_wrong_reference_or_empty_cases_before_gpu_work() {
        let bytes = include_bytes!("../../../fixtures/deepseek-v41/index-score-reference.json");
        let fixture = parse_fixture(bytes).expect("official fixture");
        assert!(!fixture.cases.is_empty());
        let mut value: serde_json::Value = serde_json::from_slice(bytes).unwrap();
        value["source"]["sha256"] = serde_json::json!("wrong");
        assert!(parse_fixture(&serde_json::to_vec(&value).unwrap()).is_err());
        value = serde_json::from_slice(bytes).unwrap();
        value["cases"] = serde_json::json!([]);
        assert!(parse_fixture(&serde_json::to_vec(&value).unwrap()).is_err());
    }
}
