//! Test-private source parity for a two-layer encoded prefix and Julia head.

use crate::encoder::Layer0Trace;
use crate::{
    ATTENTION_HEADS, DecisionHead, ENCODER_FF_WIDTH, EncoderBlock, EncoderBlockInput,
    EncoderBlockWeights, EncoderInput, FEED_FORWARD_WIDTH, FullEncoderWeights, HeadInput,
    HeadLayerWeights, HeadWeights, JuliaEncoder, ScorerWeights, WIDTH,
};
use serde_json::{Value, json};
use std::{fs::OpenOptions, io::Write};

fn values(length: usize, ordinal: usize) -> Vec<f32> {
    (0..length)
        .map(|index| {
            (f32::from(u8::try_from((index + ordinal * 17) % 97).unwrap()) - 48.0) / 1000.0
        })
        .collect()
}

fn encoder_weights(base: usize) -> EncoderBlockWeights {
    EncoderBlockWeights {
        wqkv_weight: values(3 * WIDTH * WIDTH, base),
        wo_weight: values(WIDTH * WIDTH, base + 1),
        wi_weight: values(2 * ENCODER_FF_WIDTH * WIDTH, base + 2),
        wo_mlp_weight: values(WIDTH * ENCODER_FF_WIDTH, base + 3),
        attn_norm_weight: values(WIDTH, base + 4),
        mlp_norm_weight: values(WIDTH, base + 5),
    }
}

fn near_one(ordinal: usize) -> Vec<f32> {
    values(WIDTH, ordinal)
        .into_iter()
        .map(|value| 1.0 + value / 10.0)
        .collect()
}

fn full_weights() -> FullEncoderWeights {
    let layers = (0..22)
        .map(|layer| {
            let mut weights = encoder_weights(layer * 6);
            weights.attn_norm_weight = near_one(300 + layer * 2);
            weights.mlp_norm_weight = near_one(301 + layer * 2);
            weights
        })
        .collect();
    FullEncoderWeights {
        token_ids: (0..8).collect(),
        token_rows: values(8 * WIDTH, 200),
        embedding_norm_weight: near_one(201),
        layers,
        final_norm_weight: near_one(202),
    }
}

fn full_encoder() -> JuliaEncoder {
    JuliaEncoder::new(full_weights()).unwrap()
}

fn pairwise_sum(mut values: Vec<f32>) -> f32 {
    while values.len() > 1 {
        let pairs = values.len() / 2;
        let mut next = Vec::with_capacity(pairs + values.len() % 2);
        for pair in values[..pairs * 2].chunks_exact(2) {
            next.push(pair[0] + pair[1]);
        }
        if values.len() % 2 == 1 {
            next.push(*values.last().unwrap());
        }
        values = next;
    }
    values[0]
}

fn balanced_embedding_norm(rows: &[f32], weight: &[f32]) -> Vec<f32> {
    rows.chunks_exact(WIDTH)
        .flat_map(|row| {
            let mean = pairwise_sum(row.to_vec()) / 384.0;
            let deviations: Vec<f32> = row.iter().map(|value| *value - mean).collect();
            let variance =
                pairwise_sum(deviations.iter().map(|value| value * value).collect()) / 384.0;
            let inverse = (variance + 1e-5).sqrt().recip();
            deviations
                .into_iter()
                .zip(weight.iter())
                .map(move |(value, scale)| value * inverse * scale)
        })
        .collect()
}

fn write_new_diagnostic(path: &std::path::Path, value: &Value) {
    let mut output = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .expect("JULIA_DIAGNOSTIC_OUTPUT must name a fresh owner-local path");
    output
        .write_all(value.to_string().as_bytes())
        .and_then(|()| output.write_all(b"\n"))
        .expect("write Julia diagnostic output");
}

fn trace_rows(value: &[f32], width: usize) -> Vec<&[f32]> {
    value.chunks_exact(width).collect()
}

fn trace_heads_queries_keys(value: &[f32], positions: usize) -> Vec<Vec<Vec<f32>>> {
    (0..ATTENTION_HEADS)
        .map(|head| {
            (0..positions)
                .map(|query| {
                    (0..positions)
                        .map(|key| value[(query * ATTENTION_HEADS + head) * positions + key])
                        .collect()
                })
                .collect()
        })
        .collect()
}

fn trace_positions_heads_dimensions(value: &[f32]) -> Vec<Vec<&[f32]>> {
    value
        .chunks_exact(WIDTH)
        .map(|position| position.chunks_exact(WIDTH / ATTENTION_HEADS).collect())
        .collect()
}

fn layer0_trace_value(trace: &Layer0Trace, positions: usize) -> Value {
    json!({
        "qkv": trace_rows(&trace.qkv, 3 * WIDTH),
        "rotated_query": trace_positions_heads_dimensions(&trace.rotated_query),
        "rotated_key": trace_positions_heads_dimensions(&trace.rotated_key),
        "logits": trace_heads_queries_keys(&trace.logits, positions),
        "probabilities": trace_heads_queries_keys(&trace.probabilities, positions),
        "attention_layout": "head_query_key",
        "attention_shape": [ATTENTION_HEADS, positions, positions],
        "attended": trace_rows(&trace.attended, WIDTH),
        "post_wo_residual": trace_rows(&trace.post_wo_residual, WIDTH),
        "output": trace_rows(&trace.output, WIDTH),
    })
}

fn head_layer(base: usize) -> HeadLayerWeights {
    HeadLayerWeights {
        in_proj_weight: values(3 * WIDTH * WIDTH, base),
        in_proj_bias: values(3 * WIDTH, base + 1),
        out_proj_weight: values(WIDTH * WIDTH, base + 2),
        out_proj_bias: values(WIDTH, base + 3),
        linear1_weight: values(FEED_FORWARD_WIDTH * WIDTH, base + 4),
        linear1_bias: values(FEED_FORWARD_WIDTH, base + 5),
        linear2_weight: values(WIDTH * FEED_FORWARD_WIDTH, base + 6),
        linear2_bias: values(WIDTH, base + 7),
        norm1_weight: values(WIDTH, base + 8),
        norm1_bias: values(WIDTH, base + 9),
        norm2_weight: values(WIDTH, base + 10),
        norm2_bias: values(WIDTH, base + 11),
    }
}

fn head() -> DecisionHead {
    DecisionHead::new(HeadWeights {
        layers: [head_layer(0), head_layer(12)],
        type_embedding: values(3 * WIDTH, 24),
        scorer: ScorerWeights {
            norm_weight: values(WIDTH, 25),
            norm_bias: values(WIDTH, 26),
            linear1_weight: values(WIDTH * WIDTH, 27),
            linear1_bias: values(WIDTH, 28),
            linear2_weight: values(WIDTH, 29),
            linear2_bias: values(1, 30),
        },
    })
    .unwrap()
}

fn fixture() -> Value {
    serde_json::from_str(include_str!(
        "../../../../fixtures/julia-1/prefill-reference.json"
    ))
    .unwrap()
}

#[test]
fn two_block_prefix_then_head_matches_frozen_source_scores() {
    let blocks = [
        EncoderBlock::new(encoder_weights(0)).unwrap(),
        EncoderBlock::new(encoder_weights(6)).unwrap(),
    ];
    let head = head();
    for case in fixture()["cases"].as_array().unwrap() {
        let positions = 6;
        let mut hidden: Vec<f32> = (0..positions * WIDTH)
            .map(|index| (f32::from(u8::try_from((index * 7) % 29).unwrap()) - 14.0) / 20.0)
            .collect();
        if case.get("perturb_padding").and_then(Value::as_bool) == Some(true) {
            for index in 0..2 * WIDTH {
                hidden[4 * WIDTH + index] +=
                    (f32::from(u8::try_from((index * 11) % 31).unwrap()) - 15.0) / 3.0;
            }
        }
        let attention_mask: Vec<bool> = case["attention_mask"]
            .as_array()
            .unwrap()
            .iter()
            .map(|item| item.as_bool().unwrap())
            .collect();
        for (layer, block) in blocks.iter().enumerate() {
            hidden = block
                .forward(&EncoderBlockInput {
                    hidden,
                    positions,
                    attention_mask: attention_mask.clone(),
                    layer,
                })
                .unwrap();
        }
        let actual = head
            .scores(&HeadInput {
                hidden,
                positions,
                attention_mask,
                marker_pos: case["marker_pos"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|item| usize::try_from(item.as_u64().unwrap()).unwrap())
                    .collect(),
                marker_mask: case["marker_mask"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|item| item.as_bool().unwrap())
                    .collect(),
                qtype: usize::try_from(case["qtype"].as_u64().unwrap()).unwrap(),
            })
            .unwrap();
        let expected = case["expected_scores"].as_array().unwrap();
        assert_eq!(actual.len(), expected.len());
        for (actual, expected) in actual.iter().zip(expected) {
            let expected = serde_json::from_value::<f32>(expected.clone()).unwrap();
            assert!((actual - expected).abs() <= 1e-5, "{actual} != {expected}");
        }
    }
}

#[test]
#[ignore = "diagnostic: writes full-encoder error distribution to JULIA_ERROR_OUTPUT"]
fn write_full_encoder_error_distribution() {
    let path = std::env::var("JULIA_ERROR_OUTPUT").expect("JULIA_ERROR_OUTPUT");
    let fixture: Value = serde_json::from_str(include_str!(
        "../../../../fixtures/julia-1/full-prefill-reference.json"
    ))
    .unwrap();
    let encoder = full_encoder();
    let mut report = Vec::new();
    for case in fixture["cases"].as_array().unwrap() {
        let input = EncoderInput {
            input_ids: case["input_ids"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap())
                .collect(),
            attention_mask: case["attention_mask"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_bool().unwrap())
                .collect(),
        };
        let hidden = encoder.forward(&input).unwrap();
        let expected: Vec<f32> = case["expected_hidden"]
            .as_array()
            .unwrap()
            .iter()
            .flat_map(|row| {
                row.as_array()
                    .unwrap()
                    .iter()
                    .map(|v| serde_json::from_value::<f32>(v.clone()).unwrap())
            })
            .collect();
        let errors: Vec<f32> = hidden
            .iter()
            .zip(&expected)
            .map(|(a, b)| (a - b).abs())
            .collect();
        let over = errors.iter().filter(|e| **e > 1e-5).count();
        let max = errors.iter().copied().fold(0.0_f32, f32::max);
        let rows: Vec<f32> = errors
            .chunks(WIDTH)
            .map(|r| r.iter().copied().fold(0.0_f32, f32::max))
            .collect();
        report.push(serde_json::json!({"case": case["name"], "elements": errors.len(), "over_1e-5": over, "max_abs": max, "row_max_abs": rows, "hidden": hidden,
            "mean_abs": errors.iter().map(|e| f64::from(*e)).sum::<f64>() / f64::from(u32::try_from(errors.len()).unwrap())}));
    }
    std::fs::write(path, serde_json::to_string_pretty(&report).unwrap()).unwrap();
}

/// Native encoder and head accuracy against an explicit float64 reference.
///
/// A single FP32 source run is not an accuracy oracle: two FP32 attention
/// implementations of the same pinned model differ by more than `1e-5` on
/// `unmasked_control`. Each case instead accepts native hidden error up to
/// `gate_ratio` times the FP32 SDPA source's own error against float64. The
/// ratio was fixed on calibration cases before held-out cases were evaluated.
#[test]
fn full_encoder_then_head_matches_float64_reference() {
    let fixture: Value = serde_json::from_str(include_str!(
        "../../../../fixtures/julia-1/f64-gate-reference.json"
    ))
    .unwrap();
    let ratio = fixture["gate_ratio"].as_f64().unwrap();
    let score_limit = fixture["score_max_abs"].as_f64().unwrap();
    let encoder = full_encoder();
    let head = head();
    let cases = fixture["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 21, "legacy, calibration and held-out cases");
    for case in cases {
        let name = case["name"].as_str().unwrap();
        let ids = |key: &str| -> Vec<u64> {
            case[key]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap())
                .collect()
        };
        let flags = |key: &str| -> Vec<bool> {
            case[key]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_bool().unwrap())
                .collect()
        };
        let input = EncoderInput {
            input_ids: ids("input_ids"),
            attention_mask: flags("attention_mask"),
        };
        let hidden = encoder.forward(&input).unwrap();
        let reference: Vec<f64> = case["f64_hidden"]
            .as_array()
            .unwrap()
            .iter()
            .flat_map(|row| row.as_array().unwrap().iter().map(|v| v.as_f64().unwrap()))
            .collect();
        assert_eq!(hidden.len(), reference.len(), "{name} hidden shape");
        let mut native_error = 0.0_f64;
        for (row, valid) in input.attention_mask.iter().enumerate() {
            if *valid {
                for column in 0..WIDTH {
                    let index = row * WIDTH + column;
                    native_error =
                        native_error.max((f64::from(hidden[index]) - reference[index]).abs());
                }
            }
        }
        let source_error = case["source_f32_hidden_max_abs"].as_f64().unwrap();
        assert!(
            native_error <= ratio * source_error,
            "{name}: native hidden error {native_error:e} exceeds {ratio} x source FP32 error {source_error:e}"
        );
        let marker_mask = flags("marker_mask");
        let scores = head
            .scores(&HeadInput {
                hidden,
                positions: input.input_ids.len(),
                attention_mask: input.attention_mask,
                marker_pos: ids("marker_pos")
                    .into_iter()
                    .map(|v| usize::try_from(v).unwrap())
                    .collect(),
                marker_mask: marker_mask.clone(),
                qtype: usize::try_from(case["qtype"].as_u64().unwrap()).unwrap(),
            })
            .unwrap();
        for ((actual, expected), valid) in scores
            .iter()
            .zip(case["f64_scores"].as_array().unwrap())
            .zip(&marker_mask)
        {
            if *valid {
                let error = (f64::from(*actual) - expected.as_f64().unwrap()).abs();
                assert!(error <= score_limit, "{name}: score error {error:e}");
            }
        }
    }
}

#[test]
fn injected_embedding_uses_the_normal_prefill_loop_and_admission_checks() {
    let encoder = full_encoder();
    let input = EncoderInput {
        input_ids: vec![2, 4],
        attention_mask: vec![true, true],
    };
    let scalar = encoder.forward_boundaries(&input).unwrap();
    assert_eq!(
        encoder
            .forward_boundaries_from_embedding(&input, scalar[0].clone())
            .unwrap(),
        scalar
    );
    assert!(matches!(
        encoder.forward_boundaries_from_embedding(&input, vec![0.0]),
        Err(crate::JuliaEncoderError::Length {
            field: "prefill embedding",
            ..
        })
    ));
    let mut nonfinite = vec![0.0; 2 * WIDTH];
    nonfinite[0] = f32::NAN;
    assert!(matches!(
        encoder.forward_boundaries_from_embedding(&input, nonfinite),
        Err(crate::JuliaEncoderError::NonFinite {
            field: "prefill embedding",
            ..
        })
    ));
    assert_eq!(
        encoder.forward_boundaries_from_embedding(
            &EncoderInput {
                input_ids: vec![99],
                attention_mask: vec![true],
            },
            vec![0.0; WIDTH],
        ),
        Err(crate::JuliaEncoderError::TokenId(99))
    );
}

#[test]
#[ignore = "writes opt-in layer diagnostics to JULIA_DIAGNOSTIC_OUTPUT"]
fn write_full_encoder_boundaries_for_source_diagnosis() {
    let fixture: Value = serde_json::from_str(include_str!(
        "../../../../fixtures/julia-1/full-prefill-reference.json"
    ))
    .unwrap();
    let case_name =
        std::env::var("JULIA_DIAGNOSTIC_CASE").unwrap_or_else(|_| "padded_base".to_owned());
    let case = fixture["cases"]
        .as_array()
        .unwrap()
        .iter()
        .find(|case| case["name"].as_str() == Some(&case_name))
        .expect("JULIA_DIAGNOSTIC_CASE must name a frozen full fixture case");
    let input = EncoderInput {
        input_ids: case["input_ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|item| item.as_u64().unwrap())
            .collect(),
        attention_mask: case["attention_mask"]
            .as_array()
            .unwrap()
            .iter()
            .map(|item| item.as_bool().unwrap())
            .collect(),
    };
    let boundaries = full_encoder().forward_boundaries(&input).unwrap();
    let stages = std::iter::once("embedding".to_owned())
        .chain((0..22).map(|layer| format!("layer_{layer}")))
        .chain(std::iter::once("final_norm".to_owned()));
    let mut output = serde_json::Map::new();
    for (stage, boundary) in stages.zip(boundaries) {
        output.insert(
            stage,
            json!(boundary.chunks_exact(WIDTH).collect::<Vec<_>>()),
        );
    }
    let path = std::env::var("JULIA_DIAGNOSTIC_OUTPUT")
        .expect("set JULIA_DIAGNOSTIC_OUTPUT to an owner-local diagnostic path");
    std::fs::write(
        path,
        json!({ "case": case_name, "boundaries": output }).to_string() + "\n",
    )
    .unwrap();
}

#[test]
#[ignore = "writes opt-in all-case outputs to JULIA_DIAGNOSTIC_OUTPUT"]
fn write_full_encoder_outputs_for_source_diagnosis() {
    let fixture: Value = serde_json::from_str(include_str!(
        "../../../../fixtures/julia-1/full-prefill-reference.json"
    ))
    .unwrap();
    let encoder = full_encoder();
    let head = head();
    let cases: Vec<Value> = fixture["cases"]
        .as_array()
        .unwrap()
        .iter()
        .map(|case| {
            let input = EncoderInput {
                input_ids: case["input_ids"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|item| item.as_u64().unwrap())
                    .collect(),
                attention_mask: case["attention_mask"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|item| item.as_bool().unwrap())
                    .collect(),
            };
            let boundaries = encoder.forward_boundaries(&input).unwrap();
            let hidden = boundaries.last().unwrap().clone();
            let scores = head
                .scores(&HeadInput {
                    hidden: hidden.clone(),
                    positions: input.input_ids.len(),
                    attention_mask: input.attention_mask,
                    marker_pos: case["marker_pos"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(|item| usize::try_from(item.as_u64().unwrap()).unwrap())
                        .collect(),
                    marker_mask: case["marker_mask"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(|item| item.as_bool().unwrap())
                        .collect(),
                    qtype: usize::try_from(case["qtype"].as_u64().unwrap()).unwrap(),
                })
                .unwrap();
            json!({
                "name": case["name"],
                "hidden": hidden.chunks_exact(WIDTH).collect::<Vec<_>>(),
                "boundaries": boundaries
                    .iter()
                    .map(|boundary| boundary.chunks_exact(WIDTH).collect::<Vec<_>>())
                    .collect::<Vec<_>>(),
                "scores": scores,
            })
        })
        .collect();
    let path = std::env::var("JULIA_DIAGNOSTIC_OUTPUT")
        .expect("set JULIA_DIAGNOSTIC_OUTPUT to an owner-local diagnostic path");
    std::fs::write(
        path,
        json!({
            "schema_version": 1,
            "protocol_schema": 1,
            "manifest_sha256": "e41b492e7ec8e0b0545515eda40fae63d4b1f87ea8fb54545acb277911f62b82",
            "weight_f32_sha256": "db22ef523c79b55a019a8e62f8b096157035af0945d380a9bbe6bab5586cdf68",
            "cases": cases,
        })
        .to_string()
            + "\n",
    )
    .unwrap();
}

#[test]
#[ignore = "writes opt-in calibration-only accuracy outputs to JULIA_DIAGNOSTIC_OUTPUT"]
fn write_accuracy_calibration_outputs() {
    let manifest: Value = serde_json::from_str(include_str!(
        "../../../../fixtures/julia-1/accuracy-cases.json"
    ))
    .unwrap();
    let encoder = full_encoder();
    let head = head();
    let cases: Vec<Value> = manifest["cases"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|case| {
            case["split"].as_str()
                == Some(
                    std::env::var("JULIA_ACCURACY_SPLIT")
                        .as_deref()
                        .unwrap_or("calibration"),
                )
        })
        .map(|case| {
            let input = EncoderInput {
                input_ids: case["input_ids"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|item| item.as_u64().unwrap())
                    .collect(),
                attention_mask: case["attention_mask"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|item| item.as_bool().unwrap())
                    .collect(),
            };
            let boundaries = encoder.forward_boundaries(&input).unwrap();
            let hidden = boundaries.last().unwrap().clone();
            let scores = head
                .scores(&HeadInput {
                    hidden: hidden.clone(),
                    positions: input.input_ids.len(),
                    attention_mask: input.attention_mask,
                    marker_pos: case["marker_pos"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(|item| usize::try_from(item.as_u64().unwrap()).unwrap())
                        .collect(),
                    marker_mask: case["marker_mask"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(|item| item.as_bool().unwrap())
                        .collect(),
                    qtype: usize::try_from(case["qtype"].as_u64().unwrap()).unwrap(),
                })
                .unwrap();
            json!({
                "name": case["name"],
                "hidden": hidden.chunks_exact(WIDTH).collect::<Vec<_>>(),
                "boundaries": boundaries
                    .iter()
                    .map(|boundary| boundary.chunks_exact(WIDTH).collect::<Vec<_>>())
                    .collect::<Vec<_>>(),
                "scores": scores,
            })
        })
        .collect();
    let path = std::env::var("JULIA_DIAGNOSTIC_OUTPUT")
        .expect("set JULIA_DIAGNOSTIC_OUTPUT to an owner-local diagnostic path");
    std::fs::write(
        path,
        json!({
            "schema_version": 1,
            "protocol_schema": 1,
            "manifest_sha256": "e41b492e7ec8e0b0545515eda40fae63d4b1f87ea8fb54545acb277911f62b82",
            "weight_f32_sha256": "db22ef523c79b55a019a8e62f8b096157035af0945d380a9bbe6bab5586cdf68",
            "cases": cases,
        })
        .to_string()
            + "\n",
    )
    .unwrap();
}

#[test]
#[ignore = "writes opt-in paired scalar/balanced embedding propagation outputs to JULIA_DIAGNOSTIC_OUTPUT"]
fn write_accuracy_calibration_embedding_tree_outputs() {
    fn rows(value: &[f32]) -> Vec<&[f32]> {
        value.chunks_exact(WIDTH).collect()
    }

    let manifest: Value = serde_json::from_str(include_str!(
        "../../../../fixtures/julia-1/accuracy-cases.json"
    ))
    .unwrap();
    let weights = full_weights();
    let encoder = JuliaEncoder::new(weights.clone()).unwrap();
    let cases: Vec<Value> = manifest["cases"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|case| case["split"].as_str() == Some("calibration"))
        .map(|case| {
            let input = EncoderInput {
                input_ids: case["input_ids"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|item| item.as_u64().unwrap())
                    .collect(),
                attention_mask: case["attention_mask"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|item| item.as_bool().unwrap())
                    .collect(),
            };
            let lookup = encoder.lookup_rows_for_trace(&input).unwrap();
            let scalar_boundaries = encoder.forward_boundaries(&input).unwrap();
            let balanced_embedding =
                balanced_embedding_norm(&lookup, &weights.embedding_norm_weight);
            let balanced_boundaries = encoder
                .forward_boundaries_from_embedding(&input, balanced_embedding.clone())
                .unwrap();
            json!({
                "name": case["name"],
                "input_ids": input.input_ids,
                "attention_mask": input.attention_mask,
                "scalar_boundaries": scalar_boundaries
                    .iter()
                    .map(|boundary| rows(boundary))
                    .collect::<Vec<_>>(),
                "balanced_embedding": rows(&balanced_embedding),
                "balanced_boundaries": balanced_boundaries
                    .iter()
                    .map(|boundary| rows(boundary))
                    .collect::<Vec<_>>(),
            })
        })
        .collect();
    let path = std::env::var("JULIA_DIAGNOSTIC_OUTPUT")
        .expect("set JULIA_DIAGNOSTIC_OUTPUT to a fresh owner-local diagnostic path");
    write_new_diagnostic(
        std::path::Path::new(&path),
        &json!({
            "schema_version": 1,
            "protocol_schema": 1,
            "manifest_sha256": "e41b492e7ec8e0b0545515eda40fae63d4b1f87ea8fb54545acb277911f62b82",
            "weight_f32_sha256": "db22ef523c79b55a019a8e62f8b096157035af0945d380a9bbe6bab5586cdf68",
            "cases": cases,
        }),
    );
}

#[test]
#[ignore = "writes opt-in cal_len7 layer-zero trace to JULIA_DIAGNOSTIC_OUTPUT"]
fn write_cal_len7_layer0_trace() {
    fn rows(value: &[f32], width: usize) -> Vec<&[f32]> {
        value.chunks_exact(width).collect()
    }

    fn heads_queries_keys(value: &[f32], positions: usize) -> Vec<Vec<Vec<f32>>> {
        (0..ATTENTION_HEADS)
            .map(|head| {
                (0..positions)
                    .map(|query| {
                        (0..positions)
                            .map(|key| value[(query * ATTENTION_HEADS + head) * positions + key])
                            .collect()
                    })
                    .collect()
            })
            .collect()
    }

    fn positions_heads_dimensions(value: &[f32]) -> Vec<Vec<&[f32]>> {
        value
            .chunks_exact(WIDTH)
            .map(|position| position.chunks_exact(WIDTH / ATTENTION_HEADS).collect())
            .collect()
    }

    let manifest: Value = serde_json::from_str(include_str!(
        "../../../../fixtures/julia-1/accuracy-cases.json"
    ))
    .unwrap();
    let case = manifest["cases"]
        .as_array()
        .unwrap()
        .iter()
        .find(|case| case["name"].as_str() == Some("cal_len7"))
        .unwrap();
    let input = EncoderInput {
        input_ids: case["input_ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|item| item.as_u64().unwrap())
            .collect(),
        attention_mask: case["attention_mask"]
            .as_array()
            .unwrap()
            .iter()
            .map(|item| item.as_bool().unwrap())
            .collect(),
    };
    let mut boundaries = full_encoder().forward_boundaries(&input).unwrap();
    let embedding = boundaries.remove(0);
    let trace = EncoderBlock::new(full_weights().layers.into_iter().next().unwrap())
        .unwrap()
        .forward_layer0_trace(&EncoderBlockInput {
            hidden: embedding,
            positions: input.input_ids.len(),
            attention_mask: input.attention_mask,
            layer: 0,
        })
        .unwrap();
    let path = std::env::var("JULIA_DIAGNOSTIC_OUTPUT")
        .expect("set JULIA_DIAGNOSTIC_OUTPUT to an owner-local diagnostic path");
    std::fs::write(
        path,
        json!({
            "schema_version": 2,
            "case": "cal_len7",
            "attention_layout": "head_query_key",
            "qkv": rows(&trace.qkv, 3 * WIDTH),
            "rotated_query": positions_heads_dimensions(&trace.rotated_query),
            "rotated_key": positions_heads_dimensions(&trace.rotated_key),
            "logits": heads_queries_keys(&trace.logits, input.input_ids.len()),
            "probabilities": heads_queries_keys(&trace.probabilities, input.input_ids.len()),
            "attention_shape": [ATTENTION_HEADS, input.input_ids.len(), input.input_ids.len()],
            "attended": rows(&trace.attended, WIDTH),
            "post_wo_residual": rows(&trace.post_wo_residual, WIDTH),
        })
        .to_string()
            + "\n",
    )
    .unwrap();
}

#[test]
#[ignore = "writes opt-in cal_len2 scalar/balanced layer-zero intervention trace to JULIA_DIAGNOSTIC_OUTPUT"]
fn write_cal_len2_layer0_embedding_intervention_trace() {
    let manifest: Value = serde_json::from_str(include_str!(
        "../../../../fixtures/julia-1/accuracy-cases.json"
    ))
    .unwrap();
    let case = manifest["cases"]
        .as_array()
        .unwrap()
        .iter()
        .find(|case| case["name"].as_str() == Some("cal_len2"))
        .unwrap();
    let input = EncoderInput {
        input_ids: case["input_ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|item| item.as_u64().unwrap())
            .collect(),
        attention_mask: case["attention_mask"]
            .as_array()
            .unwrap()
            .iter()
            .map(|item| item.as_bool().unwrap())
            .collect(),
    };
    let weights = full_weights();
    let encoder = JuliaEncoder::new(weights.clone()).unwrap();
    let mut scalar_boundaries = encoder.forward_boundaries(&input).unwrap();
    let scalar_embedding = scalar_boundaries.remove(0);
    let balanced_embedding = balanced_embedding_norm(
        &encoder.lookup_rows_for_trace(&input).unwrap(),
        &weights.embedding_norm_weight,
    );
    let block = EncoderBlock::new(weights.layers[0].clone()).unwrap();
    let scalar = block
        .forward_layer0_trace(&EncoderBlockInput {
            hidden: scalar_embedding.clone(),
            positions: input.input_ids.len(),
            attention_mask: input.attention_mask.clone(),
            layer: 0,
        })
        .unwrap();
    let balanced = block
        .forward_layer0_trace(&EncoderBlockInput {
            hidden: balanced_embedding.clone(),
            positions: input.input_ids.len(),
            attention_mask: input.attention_mask.clone(),
            layer: 0,
        })
        .unwrap();
    let path = std::env::var("JULIA_DIAGNOSTIC_OUTPUT")
        .expect("set JULIA_DIAGNOSTIC_OUTPUT to a fresh owner-local diagnostic path");
    write_new_diagnostic(
        std::path::Path::new(&path),
        &json!({
            "schema_version": 1,
            "case": "cal_len2",
            "input_ids": input.input_ids,
            "attention_mask": input.attention_mask,
            "manifest_sha256": "e41b492e7ec8e0b0545515eda40fae63d4b1f87ea8fb54545acb277911f62b82",
            "weight_f32_sha256": "db22ef523c79b55a019a8e62f8b096157035af0945d380a9bbe6bab5586cdf68",
            "scalar": {
                "embedding": trace_rows(&scalar_embedding, WIDTH),
                "trace": layer0_trace_value(&scalar, 2),
            },
            "balanced": {
                "embedding": trace_rows(&balanced_embedding, WIDTH),
                "trace": layer0_trace_value(&balanced, 2),
            },
        }),
    );
}

#[test]
#[ignore = "writes opt-in cal_len7 embedding lookup trace to JULIA_DIAGNOSTIC_OUTPUT"]
fn write_cal_len7_embedding_trace() {
    fn rows(value: &[f32]) -> Vec<&[f32]> {
        value.chunks_exact(WIDTH).collect()
    }

    let manifest: Value = serde_json::from_str(include_str!(
        "../../../../fixtures/julia-1/accuracy-cases.json"
    ))
    .unwrap();
    let case = manifest["cases"]
        .as_array()
        .unwrap()
        .iter()
        .find(|case| case["name"].as_str() == Some("cal_len7"))
        .unwrap();
    let input = EncoderInput {
        input_ids: case["input_ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|item| item.as_u64().unwrap())
            .collect(),
        attention_mask: case["attention_mask"]
            .as_array()
            .unwrap()
            .iter()
            .map(|item| item.as_bool().unwrap())
            .collect(),
    };
    let encoder = full_encoder();
    let lookup = encoder.lookup_rows_for_trace(&input).unwrap();
    let mut boundaries = encoder.forward_boundaries(&input).unwrap();
    let embedding = boundaries.remove(0);
    let path = std::env::var("JULIA_DIAGNOSTIC_OUTPUT")
        .expect("set JULIA_DIAGNOSTIC_OUTPUT to an owner-local diagnostic path");
    std::fs::write(
        path,
        json!({
            "schema_version": 1,
            "case": "cal_len7",
            "input_ids": input.input_ids,
            "attention_mask": input.attention_mask,
            "lookup": rows(&lookup),
            "embedding": rows(&embedding),
        })
        .to_string()
            + "\n",
    )
    .unwrap();
}

#[test]
fn full_encoder_rejects_malformed_rows_layers_ids_and_masks() {
    let mut weights = full_weights();
    weights.token_rows.pop();
    assert!(matches!(
        JuliaEncoder::new(weights),
        Err(crate::JuliaEncoderError::Length {
            field: "selected token rows",
            ..
        })
    ));
    let mut weights = full_weights();
    weights.layers.pop();
    assert!(matches!(
        JuliaEncoder::new(weights),
        Err(crate::JuliaEncoderError::FullLayers(21))
    ));
    let mut weights = full_weights();
    weights.token_ids[7] = 256_000;
    assert!(matches!(
        JuliaEncoder::new(weights),
        Err(crate::JuliaEncoderError::VocabularyId(256_000))
    ));
    let mut weights = full_weights();
    weights.token_rows[0] = f32::NAN;
    assert!(matches!(
        JuliaEncoder::new(weights),
        Err(crate::JuliaEncoderError::NonFinite {
            field: "selected token rows",
            ..
        })
    ));
    let encoder = full_encoder();
    assert_eq!(
        encoder.forward(&EncoderInput {
            input_ids: vec![9],
            attention_mask: vec![true],
        }),
        Err(crate::JuliaEncoderError::TokenId(9))
    );
    assert!(matches!(
        encoder.forward(&EncoderInput {
            input_ids: vec![1],
            attention_mask: vec![],
        }),
        Err(crate::JuliaEncoderError::Length {
            field: "prefill attention mask",
            ..
        })
    ));
    let mut weights = full_weights();
    weights.final_norm_weight.fill(f32::MAX);
    let overflow = JuliaEncoder::new(weights).unwrap();
    assert!(matches!(
        overflow.forward(&EncoderInput {
            input_ids: vec![1],
            attention_mask: vec![true],
        }),
        Err(crate::JuliaEncoderError::NonFinite {
            field: "final norm",
            ..
        })
    ));
}

/// The native encoder and head on the published Julia-1 checkpoint, gated on float64.
///
/// Opt-in: `JULIA_CHECKPOINT_DIR` names the pinned revision's directory holding
/// `model.safetensors`; `JULIA_REAL_REFERENCE` names the JSON written by the
/// float64 real-checkpoint reference generator. The gate is the synthetic-weight
/// one: native hidden error at most `gate_ratio` times the FP32 source's own error.
#[test]
#[ignore = "needs the real Julia-1 checkpoint via JULIA_CHECKPOINT_DIR and JULIA_REAL_REFERENCE"]
fn real_checkpoint_encoder_then_head_matches_float64_reference() {
    let dir = std::path::PathBuf::from(std::env::var("JULIA_CHECKPOINT_DIR").unwrap());
    let reference: Value = serde_json::from_str(
        &std::fs::read_to_string(std::env::var("JULIA_REAL_REFERENCE").unwrap()).unwrap(),
    )
    .unwrap();
    let gate: Value = serde_json::from_str(include_str!(
        "../../../../fixtures/julia-1/f64-gate-reference.json"
    ))
    .unwrap();
    let ratio = gate["gate_ratio"].as_f64().unwrap();
    let cases = reference["cases"].as_array().unwrap();
    let checkpoint = crate::JuliaCheckpoint::load(&dir).unwrap();
    let head = checkpoint.head();
    let mut report = Vec::new();
    for case in cases {
        let name = case["name"].as_str().unwrap();
        let input_ids: Vec<u64> = case["input_ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap())
            .collect();
        let positions = input_ids.len();
        let encoder = checkpoint.encoder(&input_ids).unwrap();
        let input = EncoderInput {
            input_ids,
            attention_mask: vec![true; positions],
        };
        let hidden = encoder.forward(&input).unwrap();
        let reference: Vec<f64> = case["f64_hidden"]
            .as_array()
            .unwrap()
            .iter()
            .flat_map(|row| row.as_array().unwrap().iter().map(|v| v.as_f64().unwrap()))
            .collect();
        let native_error = hidden
            .iter()
            .zip(&reference)
            .map(|(a, b)| (f64::from(*a) - b).abs())
            .fold(0.0, f64::max);
        let source_error = case["source_f32_hidden_max_abs"].as_f64().unwrap();
        let markers: Vec<usize> = case["markers"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| usize::try_from(v.as_u64().unwrap()).unwrap())
            .collect();
        let scores = head
            .scores(&HeadInput {
                hidden,
                positions,
                attention_mask: vec![true; positions],
                marker_mask: vec![true; markers.len()],
                marker_pos: markers,
                qtype: usize::try_from(case["qtype"].as_u64().unwrap()).unwrap(),
            })
            .unwrap();
        let score_error = scores
            .iter()
            .zip(case["f64_scores"].as_array().unwrap())
            .map(|(a, b)| (f64::from(*a) - b.as_f64().unwrap()).abs())
            .fold(0.0, f64::max);
        let source_score_error = case["source_f32_scores"]
            .as_array()
            .unwrap()
            .iter()
            .zip(case["f64_scores"].as_array().unwrap())
            .map(|(a, b)| (a.as_f64().unwrap() - b.as_f64().unwrap()).abs())
            .fold(0.0, f64::max);
        println!(
            "{{\"case\":\"{name}\",\"positions\":{positions},\"native_hidden\":{native_error:e},\"source_hidden\":{source_error:e},\"native_score\":{score_error:e},\"source_score\":{source_score_error:e}}}"
        );
        report.push((
            name.to_owned(),
            native_error,
            source_error,
            score_error,
            source_score_error,
        ));
    }
    for (name, native, source, native_score, source_score) in report {
        assert!(
            native <= ratio * source,
            "{name}: native hidden error {native:e} exceeds {ratio} x source FP32 error {source:e}"
        );
        assert!(
            native_score <= ratio * source_score,
            "{name}: native score error {native_score:e} exceeds {ratio} x source FP32 error {source_score:e}"
        );
    }
}
