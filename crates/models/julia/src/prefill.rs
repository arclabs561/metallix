//! Test-private source parity for a two-layer encoded prefix and Julia head.

use crate::{
    ATTENTION_HEADS, DecisionHead, ENCODER_FF_WIDTH, EncoderBlock, EncoderBlockInput,
    EncoderBlockWeights, EncoderInput, FEED_FORWARD_WIDTH, FullEncoderWeights, HeadInput,
    HeadLayerWeights, HeadWeights, JuliaEncoder, ScorerWeights, WIDTH,
};
use serde_json::{Value, json};

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
fn full_encoder_then_head_matches_frozen_sdpa_source() {
    let fixture: Value = serde_json::from_str(include_str!(
        "../../../../fixtures/julia-1/full-prefill-reference.json"
    ))
    .unwrap();
    let encoder = full_encoder();
    let head = head();
    for case in fixture["cases"].as_array().unwrap() {
        let case_name = case["name"].as_str().unwrap();
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
        let hidden = encoder.forward(&input).unwrap();
        let expected_hidden = case["expected_hidden"].as_array().unwrap();
        assert_eq!(hidden.len(), expected_hidden.len() * WIDTH);
        for (index, (actual, expected)) in hidden
            .iter()
            .zip(
                expected_hidden
                    .iter()
                    .flat_map(|row| row.as_array().unwrap()),
            )
            .enumerate()
        {
            let expected = serde_json::from_value::<f32>(expected.clone()).unwrap();
            assert!(
                (actual - expected).abs() <= 1e-5,
                "{case_name} hidden[{index}]: {actual} != {expected}"
            );
        }
        let scores = head
            .scores(&HeadInput {
                hidden,
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
        for (actual, expected) in scores
            .iter()
            .zip(case["expected_scores"].as_array().unwrap())
        {
            let expected = serde_json::from_value::<f32>(expected.clone()).unwrap();
            assert!((actual - expected).abs() <= 1e-5, "{actual} != {expected}");
        }
    }
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
