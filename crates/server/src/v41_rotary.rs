//! Bounded fixture qualification for the V4.1 FP32 Metal rotary-tail operation.

use std::{
    fs::{self, File},
    io::Read,
    num::NonZeroUsize,
    path::Path,
    process::ExitCode,
    time::Instant,
};

use serde::Deserialize;
use serde_json::json;

const MAX_FIXTURE_BYTES: u64 = 1024 * 1024;
const MAX_CASES: usize = 100;
const MAX_REPEATS: u32 = 100;
const ABSOLUTE_TOLERANCE: f32 = 0.000_001;
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
    batches: usize,
    positions: usize,
    heads: usize,
    pairs: usize,
    direction: String,
    values: Vec<f32>,
    frequencies: Vec<[f32; 2]>,
    expected_values: Vec<f32>,
}

#[derive(serde::Serialize)]
struct Comparison {
    passed: bool,
    compared_values: usize,
    max_abs_error: f32,
}

fn parse_fixture(bytes: &[u8]) -> Result<Fixture, Box<dyn std::error::Error>> {
    let fixture: Fixture = serde_json::from_slice(bytes)?;
    if fixture.schema_version != 1
        || fixture.source.revision != REVISION
        || fixture.source.sha256 != SOURCE_SHA256
        || fixture.source.symbol != "apply_rotary_emb"
        || fixture.reference.device != "cpu"
        || fixture.reference.dtype != "float32"
    {
        return Err("fixture must identify the pinned CPU float32 rotary reference".into());
    }
    if fixture.cases.is_empty() || fixture.cases.len() > MAX_CASES {
        return Err("rotary fixture must contain 1..100 cases".into());
    }
    for case in &fixture.cases {
        if case.name.trim().is_empty()
            || case.batches == 0
            || case.positions == 0
            || case.heads == 0
            || case.pairs == 0
            || !matches!(case.direction.as_str(), "forward" | "inverse")
            || case.values.len() != case.expected_values.len()
            || expected_value_count(case) != Some(case.values.len())
            || expected_frequency_count(case) != Some(case.frequencies.len())
            || case.values.iter().any(|value| !value.is_finite())
            || case.expected_values.iter().any(|value| !value.is_finite())
            || case
                .frequencies
                .iter()
                .flatten()
                .any(|value| !value.is_finite())
        {
            return Err("rotary fixture contains an invalid case layout".into());
        }
    }
    Ok(fixture)
}

fn expected_value_count(case: &Case) -> Option<usize> {
    case.batches
        .checked_mul(case.positions)?
        .checked_mul(case.heads)?
        .checked_mul(case.pairs)?
        .checked_mul(2)
}

fn expected_frequency_count(case: &Case) -> Option<usize> {
    case.positions.checked_mul(case.pairs)
}

fn load_fixture(path: &Path) -> Result<Fixture, Box<dyn std::error::Error>> {
    if !fs::metadata(path)?.file_type().is_file() {
        return Err("rotary fixture must be a regular file".into());
    }
    let mut bytes = Vec::new();
    File::open(path)?
        .take(MAX_FIXTURE_BYTES + 1)
        .read_to_end(&mut bytes)?;
    if u64::try_from(bytes.len())? > MAX_FIXTURE_BYTES {
        return Err("rotary fixture exceeds 1 MiB".into());
    }
    parse_fixture(&bytes)
}

/// Runs the bounded V4.1 rotary Metal qualification CLI handler.
pub(crate) fn run(path: &Path, repeats: u32) -> ExitCode {
    match check(path, repeats) {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => ExitCode::FAILURE,
        Err(error) => {
            eprintln!("V4.1 rotary qualification failed: {error}");
            ExitCode::FAILURE
        }
    }
}

fn check(path: &Path, repeats: u32) -> Result<bool, Box<dyn std::error::Error>> {
    if !(1..=MAX_REPEATS).contains(&repeats) {
        return Err("rotary repeats must be in 1..=100".into());
    }
    let fixture = load_fixture(path)?;
    let mut results = Vec::new();
    let mut passed = true;
    for case in fixture.cases {
        let layout = deepseek::RotaryTailLayout::new(
            nonzero(case.batches)?,
            nonzero(case.positions)?,
            nonzero(case.heads)?,
            nonzero(case.pairs)?,
        )?;
        let direction = match case.direction.as_str() {
            "forward" => deepseek::RotaryDirection::Forward,
            "inverse" => deepseek::RotaryDirection::Inverse,
            _ => return Err("fixture direction was not validated".into()),
        };
        let frequencies = case
            .frequencies
            .into_iter()
            .map(|[real, imaginary]| deepseek::RotaryFrequency::new(real, imaginary))
            .collect::<Result<Vec<_>, _>>()?;
        let evaluate =
            || deepseek::rotate_tail_metal(&case.values, layout, &frequencies, direction);
        let warmup_started = Instant::now();
        let warmup = evaluate()?;
        let warmup_ms = warmup_started.elapsed().as_secs_f64() * 1000.0;
        let warmup_comparison = compare(&warmup, &case.expected_values)?;
        passed &= warmup_comparison.passed;
        let mut evaluation_ms = Vec::new();
        let mut comparisons = Vec::new();
        for _ in 0..repeats {
            let started = Instant::now();
            let actual = evaluate()?;
            evaluation_ms.push(started.elapsed().as_secs_f64() * 1000.0);
            let comparison = compare(&actual, &case.expected_values)?;
            passed &= comparison.passed;
            comparisons.push(comparison);
        }
        results.push(json!({
            "name": case.name,
            "counts": {
                "batches": case.batches,
                "positions": case.positions,
                "heads": case.heads,
                "complex_pairs": case.pairs,
                "input_scalars": case.values.len(),
                "frequencies": frequencies.len(),
                "repeats": repeats,
            },
            "direction": case.direction,
            "excluded_warmup_ms": warmup_ms,
            "warmup_comparison": warmup_comparison,
            "evaluation_ms": evaluation_ms,
            "comparisons": comparisons,
        }));
    }
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "schema_version": 1,
            "operation": "v41_fp32_rotary_metal_qualification",
            "passed": passed,
            "source_revision": REVISION,
            "source_sha256": SOURCE_SHA256,
            "absolute_tolerance": ABSOLUTE_TOLERANCE,
            "cases": results,
            "scope": "Synthetic FP32 rotary tails from a pinned CPU fixture; validation, host array creation, GPU graph construction, execution, and readback are included in timings after one excluded warmup. This is a diagnostic round trip, not in-place cache update, fused attention, model execution, or serving throughput. Fixture provenance fields are checked, not authenticated."
        }))?
    );
    Ok(passed)
}

fn nonzero(value: usize) -> Result<NonZeroUsize, Box<dyn std::error::Error>> {
    NonZeroUsize::new(value).ok_or_else(|| "rotary fixture dimension must be nonzero".into())
}

fn compare(actual: &[f32], expected: &[f32]) -> Result<Comparison, Box<dyn std::error::Error>> {
    if actual.len() != expected.len() {
        return Err("rotary output length does not match fixture output length".into());
    }
    let mut passed = true;
    let mut max_abs_error = 0.0_f32;
    for (&actual, &expected) in actual.iter().zip(expected) {
        if !actual.is_finite() || !expected.is_finite() {
            return Err("rotary comparison requires finite expected and actual values".into());
        }
        let error = (actual - expected).abs();
        max_abs_error = max_abs_error.max(error);
        passed &= error <= ABSOLUTE_TOLERANCE;
    }
    Ok(Comparison {
        passed,
        compared_values: actual.len(),
        max_abs_error,
    })
}

#[cfg(test)]
mod tests {
    use super::parse_fixture;

    #[test]
    fn refuses_wrong_reference_or_invalid_case_before_gpu_work() {
        let bytes = include_bytes!("../../../fixtures/deepseek-v41/rotary-reference.json");
        let fixture = parse_fixture(bytes).expect("official fixture");
        assert!(!fixture.cases.is_empty());
        let mut value: serde_json::Value = serde_json::from_slice(bytes).expect("valid JSON");
        value["source"]["symbol"] = serde_json::json!("wrong");
        assert!(parse_fixture(&serde_json::to_vec(&value).expect("serializes")).is_err());
        value = serde_json::from_slice(bytes).expect("valid JSON");
        value["cases"][0]["direction"] = serde_json::json!("sideways");
        assert!(parse_fixture(&serde_json::to_vec(&value).expect("serializes")).is_err());
        value = serde_json::from_slice(bytes).expect("valid JSON");
        value["cases"][0]["expected_values"][0] = serde_json::json!(null);
        assert!(parse_fixture(&serde_json::to_vec(&value).expect("serializes")).is_err());
    }
}
