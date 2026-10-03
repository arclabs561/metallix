//! Fixed-shape F32 Julia decision-head reference, without an encoder or checkpoint reader.

use thiserror::Error;

pub const WIDTH: usize = 384;
pub const ATTENTION_HEADS: usize = 6;
pub const HEAD_WIDTH: usize = WIDTH / ATTENTION_HEADS;
pub const FEED_FORWARD_WIDTH: usize = 1536;
pub const HEAD_LAYERS: usize = 2;
pub const INVALID_MARKER_SCORE: f32 = -10_000.0;
const EPSILON: f32 = 1e-5;
const WIDTH_F32: f32 = 384.0;
const HEAD_WIDTH_F32: f32 = 64.0;
// Matches the encoder's sequence bound so every encoded prefill can be scored.
const MAX_POSITIONS: usize = 126;
const MAX_MARKERS: usize = 20;
// Two layers at 126 positions plus 20 scored markers: 476.2M MACs.
const MAX_WORK: usize = 480_000_000;

/// Unvalidated transfer container; [`DecisionHead::new`] validates every field.
#[derive(Clone, Debug)]
pub struct HeadLayerWeights {
    pub in_proj_weight: Vec<f32>,
    pub in_proj_bias: Vec<f32>,
    pub out_proj_weight: Vec<f32>,
    pub out_proj_bias: Vec<f32>,
    pub linear1_weight: Vec<f32>,
    pub linear1_bias: Vec<f32>,
    pub linear2_weight: Vec<f32>,
    pub linear2_bias: Vec<f32>,
    pub norm1_weight: Vec<f32>,
    pub norm1_bias: Vec<f32>,
    pub norm2_weight: Vec<f32>,
    pub norm2_bias: Vec<f32>,
}
/// Unvalidated transfer container; [`DecisionHead::new`] validates every field.
#[derive(Clone, Debug)]
pub struct ScorerWeights {
    pub norm_weight: Vec<f32>,
    pub norm_bias: Vec<f32>,
    pub linear1_weight: Vec<f32>,
    pub linear1_bias: Vec<f32>,
    pub linear2_weight: Vec<f32>,
    pub linear2_bias: Vec<f32>,
}
/// Unvalidated transfer container; [`DecisionHead::new`] validates every field.
#[derive(Clone, Debug)]
pub struct HeadWeights {
    pub layers: [HeadLayerWeights; HEAD_LAYERS],
    pub type_embedding: Vec<f32>,
    pub scorer: ScorerWeights,
}
/// Unvalidated per-call input; [`DecisionHead::scores`] validates every field.
#[derive(Clone, Debug)]
pub struct HeadInput {
    pub hidden: Vec<f32>,
    pub positions: usize,
    pub attention_mask: Vec<bool>,
    pub marker_pos: Vec<usize>,
    pub marker_mask: Vec<bool>,
    pub qtype: usize,
}
#[derive(Clone, Debug, Error, Eq, PartialEq)]
#[non_exhaustive]
pub enum JuliaHeadError {
    #[error("Julia head positions must be 1..={MAX_POSITIONS}, got {0}")]
    Positions(usize),
    #[error("Julia head markers must be 1..={MAX_MARKERS}, got {0}")]
    Markers(usize),
    #[error("Julia head qtype must be 0..3, got {0}")]
    Qtype(usize),
    #[error("Julia head {field} length is {actual}, expected {expected}")]
    Length {
        field: &'static str,
        actual: usize,
        expected: usize,
    },
    #[error("Julia head {field} contains non-finite value at {index}")]
    NonFinite { field: &'static str, index: usize },
    #[error("Julia head marker {marker} is outside {positions} positions: {position}")]
    MarkerPosition {
        marker: usize,
        position: usize,
        positions: usize,
    },
    #[error("Julia head has no unmasked key positions")]
    NoKeys,
    #[error("Julia head scalar work exceeds {MAX_WORK}")]
    Work,
}

pub struct DecisionHead {
    weights: HeadWeights,
}
impl DecisionHead {
    pub fn new(weights: HeadWeights) -> Result<Self, JuliaHeadError> {
        validate_weights(&weights)?;
        Ok(Self { weights })
    }
    pub fn scores(&self, input: &HeadInput) -> Result<Vec<f32>, JuliaHeadError> {
        validate_input(input)?;
        let keys = input.attention_mask.iter().filter(|&&x| x).count();
        if keys == 0 {
            return Err(JuliaHeadError::NoKeys);
        }
        let affine_per_position = WIDTH
            .checked_mul(3 * WIDTH + WIDTH + 2 * FEED_FORWARD_WIDTH)
            .ok_or(JuliaHeadError::Work)?;
        // MAC budget includes each attention pair's Q·K and probability·V reductions.
        let attention_per_layer = input
            .positions
            .checked_mul(input.positions)
            .and_then(|x| x.checked_mul(2 * WIDTH))
            .ok_or(JuliaHeadError::Work)?;
        let layer_work = input
            .positions
            .checked_mul(affine_per_position)
            .and_then(|x| x.checked_add(attention_per_layer))
            .ok_or(JuliaHeadError::Work)?;
        let scorer_work = input
            .marker_pos
            .len()
            .checked_mul(WIDTH)
            .and_then(|x| x.checked_mul(2 * WIDTH + 2))
            .ok_or(JuliaHeadError::Work)?;
        let work = layer_work
            .checked_mul(HEAD_LAYERS)
            .and_then(|x| x.checked_add(scorer_work))
            .ok_or(JuliaHeadError::Work)?;
        if work > MAX_WORK {
            return Err(JuliaHeadError::Work);
        }
        let mut value = input.hidden.clone();
        let emb = &self.weights.type_embedding[input.qtype * WIDTH..(input.qtype + 1) * WIDTH];
        for row in value.chunks_exact_mut(WIDTH) {
            for (x, e) in row.iter_mut().zip(emb) {
                *x += e;
            }
        }
        for layer in &self.weights.layers {
            value = transformer_layer(value, input.positions, &input.attention_mask, layer)?;
        }
        let mut output = Vec::with_capacity(input.marker_pos.len());
        for (marker, (&position, &present)) in
            input.marker_pos.iter().zip(&input.marker_mask).enumerate()
        {
            if position >= input.positions {
                return Err(JuliaHeadError::MarkerPosition {
                    marker,
                    position,
                    positions: input.positions,
                });
            }
            if !present {
                output.push(INVALID_MARKER_SCORE);
                continue;
            }
            let row = &value[position * WIDTH..(position + 1) * WIDTH];
            output.push(score(row, &self.weights.scorer)?);
        }
        Ok(output)
    }
}
fn transformer_layer(
    value: Vec<f32>,
    positions: usize,
    mask: &[bool],
    w: &HeadLayerWeights,
) -> Result<Vec<f32>, JuliaHeadError> {
    let normalized = norm_rows(&value, positions, &w.norm1_weight, &w.norm1_bias)?;
    let qkv = linear(
        &normalized,
        positions,
        WIDTH,
        3 * WIDTH,
        &w.in_proj_weight,
        &w.in_proj_bias,
    )?;
    let mut attended = vec![0.0; positions * WIDTH];
    let scale = HEAD_WIDTH_F32.sqrt().recip();
    for query in 0..positions {
        for head in 0..ATTENTION_HEADS {
            let mut logits = Vec::with_capacity(positions);
            let mut maximum = f32::NEG_INFINITY;
            for key in 0..positions {
                if !mask[key] {
                    logits.push(f32::NEG_INFINITY);
                    continue;
                }
                let mut dot = 0.0;
                for dim in 0..HEAD_WIDTH {
                    dot += qkv[query * 3 * WIDTH + head * HEAD_WIDTH + dim]
                        * qkv[key * 3 * WIDTH + WIDTH + head * HEAD_WIDTH + dim];
                }
                let logit = dot * scale;
                maximum = maximum.max(logit);
                logits.push(logit);
            }
            let denominator: f32 = logits.iter().map(|x| (*x - maximum).exp()).sum();
            for dim in 0..HEAD_WIDTH {
                let mut total = 0.0;
                for key in 0..positions {
                    if mask[key] {
                        total += ((logits[key] - maximum).exp() / denominator)
                            * qkv[key * 3 * WIDTH + 2 * WIDTH + head * HEAD_WIDTH + dim];
                    }
                }
                attended[query * WIDTH + head * HEAD_WIDTH + dim] = total;
            }
        }
    }
    let projected = linear(
        &attended,
        positions,
        WIDTH,
        WIDTH,
        &w.out_proj_weight,
        &w.out_proj_bias,
    )?;
    let mut residual = value;
    for (a, b) in residual.iter_mut().zip(projected) {
        *a += b;
    }
    let normalized = norm_rows(&residual, positions, &w.norm2_weight, &w.norm2_bias)?;
    let mut feed = linear(
        &normalized,
        positions,
        WIDTH,
        FEED_FORWARD_WIDTH,
        &w.linear1_weight,
        &w.linear1_bias,
    )?;
    for x in &mut feed {
        *x = x.max(0.0);
    }
    let feed = linear(
        &feed,
        positions,
        FEED_FORWARD_WIDTH,
        WIDTH,
        &w.linear2_weight,
        &w.linear2_bias,
    )?;
    for (a, b) in residual.iter_mut().zip(feed) {
        *a += b;
    }
    Ok(residual)
}
fn score(row: &[f32], w: &ScorerWeights) -> Result<f32, JuliaHeadError> {
    let normalized = norm_rows(row, 1, &w.norm_weight, &w.norm_bias)?;
    let mut hidden = linear(
        &normalized,
        1,
        WIDTH,
        WIDTH,
        &w.linear1_weight,
        &w.linear1_bias,
    )?;
    for x in &mut hidden {
        *x = 0.5 * *x * (1.0 + libm::erff(*x / 2.0_f32.sqrt()));
    }
    Ok(linear(&hidden, 1, WIDTH, 1, &w.linear2_weight, &w.linear2_bias)?[0])
}
fn norm_rows(
    value: &[f32],
    rows: usize,
    weight: &[f32],
    bias: &[f32],
) -> Result<Vec<f32>, JuliaHeadError> {
    check("norm value", value)?;
    check_len("norm weight", weight, WIDTH)?;
    check_len("norm bias", bias, WIDTH)?;
    let mut out = Vec::with_capacity(value.len());
    for (row_index, row) in value.chunks_exact(WIDTH).take(rows).enumerate() {
        let mean = row.iter().sum::<f32>() / WIDTH_F32;
        if !mean.is_finite() {
            return Err(JuliaHeadError::NonFinite {
                field: "norm mean",
                index: row_index,
            });
        }
        let variance = row
            .iter()
            .map(|x| {
                let d = *x - mean;
                d * d
            })
            .sum::<f32>()
            / WIDTH_F32;
        if !variance.is_finite() {
            return Err(JuliaHeadError::NonFinite {
                field: "norm variance",
                index: row_index,
            });
        }
        let inv = (variance + EPSILON).sqrt().recip();
        if !inv.is_finite() {
            return Err(JuliaHeadError::NonFinite {
                field: "norm inverse standard deviation",
                index: row_index,
            });
        }
        out.extend(
            row.iter()
                .zip(weight)
                .zip(bias)
                .map(|((x, w), b)| (*x - mean) * inv * *w + *b),
        );
    }
    Ok(out)
}
fn linear(
    input: &[f32],
    rows: usize,
    reduction: usize,
    outputs: usize,
    weight: &[f32],
    bias: &[f32],
) -> Result<Vec<f32>, JuliaHeadError> {
    check("linear input", input)?;
    check_len("linear weight", weight, outputs * reduction)?;
    check_len("linear bias", bias, outputs)?;
    let mut out = vec![0.0; rows * outputs];
    for row in 0..rows {
        for output in 0..outputs {
            let mut sum = bias[output];
            for col in 0..reduction {
                sum += input[row * reduction + col] * weight[output * reduction + col];
            }
            if !sum.is_finite() {
                return Err(JuliaHeadError::NonFinite {
                    field: "linear output",
                    index: row * outputs + output,
                });
            }
            out[row * outputs + output] = sum;
        }
    }
    Ok(out)
}
fn check(field: &'static str, values: &[f32]) -> Result<(), JuliaHeadError> {
    for (i, x) in values.iter().enumerate() {
        if !x.is_finite() {
            return Err(JuliaHeadError::NonFinite { field, index: i });
        }
    }
    Ok(())
}
fn check_len(field: &'static str, values: &[f32], expected: usize) -> Result<(), JuliaHeadError> {
    if values.len() != expected {
        return Err(JuliaHeadError::Length {
            field,
            actual: values.len(),
            expected,
        });
    }
    check(field, values)
}
fn validate_input(i: &HeadInput) -> Result<(), JuliaHeadError> {
    if i.positions == 0 || i.positions > MAX_POSITIONS {
        return Err(JuliaHeadError::Positions(i.positions));
    }
    if i.marker_pos.is_empty() || i.marker_pos.len() > MAX_MARKERS {
        return Err(JuliaHeadError::Markers(i.marker_pos.len()));
    }
    if i.qtype >= 3 {
        return Err(JuliaHeadError::Qtype(i.qtype));
    }
    check_len("hidden", &i.hidden, i.positions * WIDTH)?;
    if i.attention_mask.len() != i.positions {
        return Err(JuliaHeadError::Length {
            field: "attention mask",
            actual: i.attention_mask.len(),
            expected: i.positions,
        });
    }
    if i.marker_mask.len() != i.marker_pos.len() {
        return Err(JuliaHeadError::Length {
            field: "marker mask",
            actual: i.marker_mask.len(),
            expected: i.marker_pos.len(),
        });
    }
    Ok(())
}
fn validate_weights(w: &HeadWeights) -> Result<(), JuliaHeadError> {
    check_len("type embedding", &w.type_embedding, 3 * WIDTH)?;
    for layer in &w.layers {
        check_len(
            "in projection weight",
            &layer.in_proj_weight,
            3 * WIDTH * WIDTH,
        )?;
        check_len("in projection bias", &layer.in_proj_bias, 3 * WIDTH)?;
        check_len(
            "out projection weight",
            &layer.out_proj_weight,
            WIDTH * WIDTH,
        )?;
        check_len("out projection bias", &layer.out_proj_bias, WIDTH)?;
        check_len(
            "linear1 weight",
            &layer.linear1_weight,
            FEED_FORWARD_WIDTH * WIDTH,
        )?;
        check_len("linear1 bias", &layer.linear1_bias, FEED_FORWARD_WIDTH)?;
        check_len(
            "linear2 weight",
            &layer.linear2_weight,
            WIDTH * FEED_FORWARD_WIDTH,
        )?;
        check_len("linear2 bias", &layer.linear2_bias, WIDTH)?;
        check_len("norm1 weight", &layer.norm1_weight, WIDTH)?;
        check_len("norm1 bias", &layer.norm1_bias, WIDTH)?;
        check_len("norm2 weight", &layer.norm2_weight, WIDTH)?;
        check_len("norm2 bias", &layer.norm2_bias, WIDTH)?;
    }
    let s = &w.scorer;
    check_len("scorer norm weight", &s.norm_weight, WIDTH)?;
    check_len("scorer norm bias", &s.norm_bias, WIDTH)?;
    check_len("scorer linear1 weight", &s.linear1_weight, WIDTH * WIDTH)?;
    check_len("scorer linear1 bias", &s.linear1_bias, WIDTH)?;
    check_len("scorer linear2 weight", &s.linear2_weight, WIDTH)?;
    check_len("scorer linear2 bias", &s.linear2_bias, 1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;
    fn fixture() -> Value {
        serde_json::from_str(include_str!(
            "../../../../fixtures/julia-1/head-reference.json"
        ))
        .unwrap()
    }
    fn generated(fixture: &Value, name: &str, shape: &[usize]) -> Vec<f32> {
        let entry = fixture["parameter_generation"]["named_parameters"]
            .as_array()
            .unwrap()
            .iter()
            .find(|item| item["name"] == name)
            .unwrap_or_else(|| panic!("missing {name}"));
        assert_eq!(
            entry["shape"]
                .as_array()
                .unwrap()
                .iter()
                .map(|x| usize::try_from(x.as_u64().unwrap()).unwrap())
                .collect::<Vec<_>>(),
            shape,
            "{name} shape"
        );
        let ordinal = usize::try_from(entry["ordinal"].as_u64().unwrap()).unwrap();
        values(shape.iter().product(), ordinal)
    }
    fn fixture_layer(fixture: &Value, layer: usize) -> HeadLayerWeights {
        let p = format!("head.layers.{layer}");
        HeadLayerWeights {
            in_proj_weight: generated(
                fixture,
                &format!("{p}.self_attn.in_proj_weight"),
                &[3 * WIDTH, WIDTH],
            ),
            in_proj_bias: generated(
                fixture,
                &format!("{p}.self_attn.in_proj_bias"),
                &[3 * WIDTH],
            ),
            out_proj_weight: generated(
                fixture,
                &format!("{p}.self_attn.out_proj.weight"),
                &[WIDTH, WIDTH],
            ),
            out_proj_bias: generated(fixture, &format!("{p}.self_attn.out_proj.bias"), &[WIDTH]),
            linear1_weight: generated(
                fixture,
                &format!("{p}.linear1.weight"),
                &[FEED_FORWARD_WIDTH, WIDTH],
            ),
            linear1_bias: generated(fixture, &format!("{p}.linear1.bias"), &[FEED_FORWARD_WIDTH]),
            linear2_weight: generated(
                fixture,
                &format!("{p}.linear2.weight"),
                &[WIDTH, FEED_FORWARD_WIDTH],
            ),
            linear2_bias: generated(fixture, &format!("{p}.linear2.bias"), &[WIDTH]),
            norm1_weight: generated(fixture, &format!("{p}.norm1.weight"), &[WIDTH]),
            norm1_bias: generated(fixture, &format!("{p}.norm1.bias"), &[WIDTH]),
            norm2_weight: generated(fixture, &format!("{p}.norm2.weight"), &[WIDTH]),
            norm2_bias: generated(fixture, &format!("{p}.norm2.bias"), &[WIDTH]),
        }
    }
    fn fixture_head(fixture: &Value) -> DecisionHead {
        DecisionHead::new(HeadWeights {
            layers: [fixture_layer(fixture, 0), fixture_layer(fixture, 1)],
            type_embedding: generated(fixture, "type_emb.weight", &[3, WIDTH]),
            scorer: ScorerWeights {
                norm_weight: generated(fixture, "scorer.0.weight", &[WIDTH]),
                norm_bias: generated(fixture, "scorer.0.bias", &[WIDTH]),
                linear1_weight: generated(fixture, "scorer.1.weight", &[WIDTH, WIDTH]),
                linear1_bias: generated(fixture, "scorer.1.bias", &[WIDTH]),
                linear2_weight: generated(fixture, "scorer.3.weight", &[1, WIDTH]),
                linear2_bias: generated(fixture, "scorer.3.bias", &[1]),
            },
        })
        .unwrap()
    }
    fn values(length: usize, ordinal: usize) -> Vec<f32> {
        (0..length)
            .map(|index| {
                (f32::from(u8::try_from((index + ordinal * 17) % 97).unwrap()) - 48.0) / 1000.0
            })
            .collect()
    }
    fn layer(base: usize) -> HeadLayerWeights {
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
            layers: [layer(0), layer(12)],
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
    fn input() -> HeadInput {
        HeadInput {
            hidden: (0..6 * WIDTH)
                .map(|x| (f32::from(u8::try_from((x * 7) % 29).unwrap()) - 14.0) / 20.0)
                .collect(),
            positions: 6,
            attention_mask: vec![true, true, true, true, false, false],
            marker_pos: vec![1, 3],
            marker_mask: vec![true, true],
            qtype: 2,
        }
    }
    #[test]
    fn consumes_all_frozen_source_cases_and_parameter_mapping() {
        let fixture = fixture();
        assert_eq!(fixture["tolerances"]["rtol"].as_f64(), Some(1e-5));
        assert_eq!(fixture["tolerances"]["atol"].as_f64(), Some(1e-5));
        let head = fixture_head(&fixture);
        let cases = fixture["cases"].as_array().unwrap();
        assert_eq!(cases.len(), 5);
        for case in cases {
            let mut input = input();
            input.attention_mask = case["attention_mask"]
                .as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_bool().unwrap())
                .collect();
            input.marker_pos = case["marker_pos"]
                .as_array()
                .unwrap()
                .iter()
                .map(|x| usize::try_from(x.as_u64().unwrap()).unwrap())
                .collect();
            input.marker_mask = case["marker_mask"]
                .as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_bool().unwrap())
                .collect();
            input.qtype = usize::try_from(case["qtype"].as_u64().unwrap()).unwrap();
            if case.get("perturb_padding").and_then(Value::as_bool) == Some(true) {
                for index in 0..2 * WIDTH {
                    input.hidden[4 * WIDTH + index] +=
                        (f32::from(u8::try_from((index * 11) % 31).unwrap()) - 15.0) / 3.0;
                }
            }
            let actual = head.scores(&input).unwrap();
            let expected: Vec<f32> = case["expected_scores"]
                .as_array()
                .unwrap()
                .iter()
                .map(|x| serde_json::from_value::<f32>(x.clone()).unwrap())
                .collect();
            assert_eq!(actual.len(), expected.len());
            for (actual, expected) in actual.iter().zip(expected) {
                assert!((actual - expected).abs() <= 1e-5, "{actual} != {expected}");
            }
        }
    }
    #[test]
    fn matches_frozen_source_scores_and_marker_permutation() {
        let h = head();
        let base = h.scores(&input()).unwrap();
        assert!((base[0] - 0.009_646_272).abs() < 1e-5);
        assert!((base[1] - 0.008_379_431).abs() < 1e-5);
        let mut swapped = input();
        swapped.marker_pos.swap(0, 1);
        assert_eq!(h.scores(&swapped).unwrap(), vec![base[1], base[0]]);
    }
    #[test]
    fn masked_marker_is_invalid_but_out_of_range_still_rejects() {
        let h = head();
        let mut valid = input();
        valid.marker_pos.push(2);
        valid.marker_mask.push(false);
        assert_eq!(
            h.scores(&valid).unwrap()[2].to_bits(),
            INVALID_MARKER_SCORE.to_bits()
        );
        valid.marker_pos[2] = 6;
        assert!(matches!(
            h.scores(&valid),
            Err(JuliaHeadError::MarkerPosition { .. })
        ));
    }
    #[test]
    fn rejects_nonfinite_hidden_and_missing_keys() {
        let h = head();
        let mut bad = input();
        bad.hidden[0] = f32::NAN;
        assert!(matches!(
            h.scores(&bad),
            Err(JuliaHeadError::NonFinite {
                field: "hidden",
                ..
            })
        ));
        bad = input();
        bad.attention_mask.fill(false);
        assert_eq!(h.scores(&bad), Err(JuliaHeadError::NoKeys));
    }
    #[test]
    fn rejects_finite_input_with_overflowing_layer_norm_variance() {
        let h = head();
        let mut bad = input();
        for (index, value) in bad.hidden.iter_mut().enumerate() {
            *value = if index.is_multiple_of(2) { 1e20 } else { -1e20 };
        }
        assert!(matches!(
            h.scores(&bad),
            Err(JuliaHeadError::NonFinite {
                field: "norm variance",
                ..
            })
        ));
    }
    #[test]
    fn rejects_work_that_exceeds_cap_and_malformed_weights() {
        let h = head();
        let widened = |positions: usize| {
            let mut wide = input();
            wide.positions = positions;
            wide.hidden.resize(positions * WIDTH, 0.0);
            wide.attention_mask.resize(positions, true);
            wide.marker_pos = vec![1];
            wide.marker_mask = vec![true];
            wide
        };
        // The work cap admits the largest admitted sequence; one more position is rejected.
        assert!(h.scores(&widened(126)).is_ok());
        assert_eq!(h.scores(&widened(127)), Err(JuliaHeadError::Positions(127)));
        let mut weights = HeadWeights {
            layers: [layer(0), layer(12)],
            type_embedding: values(3 * WIDTH, 24),
            scorer: ScorerWeights {
                norm_weight: values(WIDTH, 25),
                norm_bias: values(WIDTH, 26),
                linear1_weight: values(WIDTH * WIDTH, 27),
                linear1_bias: values(WIDTH, 28),
                linear2_weight: values(WIDTH, 29),
                linear2_bias: values(1, 30),
            },
        };
        weights.layers[0].norm1_bias.pop();
        assert!(matches!(
            DecisionHead::new(weights),
            Err(JuliaHeadError::Length {
                field: "norm1 bias",
                ..
            })
        ));
    }
}
