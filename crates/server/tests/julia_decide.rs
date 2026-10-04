//! `mx decide-julia` on the published Julia-1 checkpoint against the source reference.

#![cfg(feature = "metal")]

use std::process::Command;

use serde_json::{Value, json};

/// The reference generator's requests, in the typed `decide` shape, as JSON text.
fn typed_requests() -> Vec<(&'static str, String)> {
    let choice = |state: Value, question: &str, options: &[&str]| {
        let criteria: serde_json::Map<_, _> = options
            .iter()
            .enumerate()
            .map(|(i, &o)| (format!("o{i}"), json!(o)))
            .collect();
        json!({"state": state, "questions": {"q": {"type": "choice", "instructions": question, "criteria": criteria}}})
            .to_string()
    };
    vec![
        ("calib-a", choice(json!("hp 3"), "next?", &["heal", "attack"])),
        (
            "calib-b",
            json!({"state": "low", "questions": {"q": {"type": "noul", "instructions": "go?", "criteria": {"false": "yes", "true": "no"}}}})
                .to_string(),
        ),
        ("held-a", choice(json!("rain"), "take?", &["umbrella", "hat"])),
        (
            "held-b",
            json!({"state": "x=2", "questions": {"q": {"type": "score", "instructions": "rate", "criteria": ["good", "bad"]}}})
                .to_string(),
        ),
        ("held-c", choice(json!(""), "pick", &["a", "b"])),
        (
            "held-long",
            // Raw text keeps the source's state member order; `Value` would sort it.
            String::from(
                r#"{"state": {"inventory": ["sword", "potion", "map"], "enemies": 3, "hp": 12, "weather": "storm"},
                    "questions": {"q": {"type": "choice", "instructions": "What should the party do next?",
                    "criteria": {"o0": "retreat to camp", "o1": "fight", "o2": "drink potion", "o3": "read map"}}}}"#,
            ),
        ),
    ]
}

/// Opt-in: `JULIA_CHECKPOINT_DIR` is the pinned checkpoint directory and
/// `JULIA_REAL_REFERENCE` the JSON from `scripts/julia_real_f64_reference.py`.
/// CLI token IDs must equal the source `sequence` IDs, and CLI scores must
/// stay within the published-checkpoint gate: `gate_ratio` times the FP32
/// source's own error against float64.
#[test]
#[allow(clippy::cast_possible_truncation, reason = "printed scores are F32")]
#[ignore = "needs the real Julia-1 checkpoint via JULIA_CHECKPOINT_DIR and JULIA_REAL_REFERENCE"]
fn decide_julia_matches_source_reference_on_the_published_checkpoint() {
    let checkpoint = std::env::var("JULIA_CHECKPOINT_DIR").unwrap();
    let reference: Value = serde_json::from_str(
        &std::fs::read_to_string(std::env::var("JULIA_REAL_REFERENCE").unwrap()).unwrap(),
    )
    .unwrap();
    let gate: Value = serde_json::from_str(include_str!(
        "../../../fixtures/julia-1/f64-gate-reference.json"
    ))
    .unwrap();
    let ratio = gate["gate_ratio"].as_f64().unwrap();
    let scratch = std::env::temp_dir().join(format!("julia-decide-{}", std::process::id()));
    std::fs::create_dir_all(&scratch).unwrap();
    for (name, request) in typed_requests() {
        let case = reference["cases"]
            .as_array()
            .unwrap()
            .iter()
            .find(|case| case["name"] == name)
            .unwrap();
        let path = scratch.join(format!("{name}.json"));
        std::fs::write(&path, request).unwrap();
        let output = Command::new(env!("CARGO_BIN_EXE_mx"))
            .args(["decide-julia", "--model", &checkpoint, "--request"])
            .arg(&path)
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{name}: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        let receipt: Value = serde_json::from_slice(&output.stdout).unwrap();
        let answer = &receipt["answers"]["q"];
        assert_eq!(answer["input_ids"], case["input_ids"], "{name} token IDs");
        assert_eq!(answer["markers"], case["markers"], "{name} markers");
        let error = |scores: &Value| {
            scores
                .as_array()
                .unwrap()
                .iter()
                .zip(case["f64_scores"].as_array().unwrap())
                // Scores are printed F32 values; recover them exactly before widening.
                .map(|(a, b)| (f64::from(a.as_f64().unwrap() as f32) - b.as_f64().unwrap()).abs())
                .fold(0.0, f64::max)
        };
        let (native, source) = (error(&answer["scores"]), error(&case["source_f32_scores"]));
        println!("{name}: native score error {native:e}, source FP32 {source:e}");
        assert!(
            native <= ratio * source,
            "{name}: native score error {native:e} exceeds {ratio} x source {source:e}"
        );
    }
    let _ = std::fs::remove_dir_all(&scratch);
}
