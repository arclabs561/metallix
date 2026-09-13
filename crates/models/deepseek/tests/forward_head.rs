//! Consume the final-head boundary from a complete synthetic source forward.
//!
//! This executes native final HC collapse, `RMSNorm` and FP32 linear, not a
//! full-model graph. The comparison policy is fixed before candidate execution:
//! two FP32 dot-product
//! error bounds, each gamma(2K) * sum(abs(x*w)), with u = 2^-24. It permits
//! reduction-order differences, not arbitrary relative error near cancellation.

use deepseek::precision::fp32_linear_reference;
use deepseek::{hc::mixing::hc_pre_bf16_reference, rms_norm_bf16_reference};
use serde::Deserialize;

const WIDTH: usize = 128;
const VOCAB: usize = 8;

#[derive(Deserialize)]
struct Fixture {
    schema_version: u32,
    source: Source,
    weight_shape: [usize; 2],
    weight_fp32_bits: Vec<u32>,
    cases: Vec<Case>,
    comparison_policy: Policy,
    norm_weight_bf16: Vec<u16>,
    norm_epsilon_bits: u32,
}

#[derive(Deserialize)]
struct Source {
    revision: String,
    model_sha256: String,
    cpu_backend_sha256: String,
    complete_capture_sha256: String,
    manifest_canonical_sha256: String,
}

#[derive(Deserialize)]
struct Case {
    start_pos: usize,
    input_shape: [usize; 3],
    input_bf16: Vec<u16>,
    logits_shape: [usize; 2],
    logits_fp32_bits: Vec<u32>,
    final_block_shape: [usize; 4],
    final_block_bf16: Vec<u16>,
    final_pre_shape: [usize; 3],
    final_pre_fp32_bits: Vec<u32>,
    collapsed_bf16: Vec<u16>,
}

#[derive(Deserialize)]
struct Policy {
    kind: String,
    unit_roundoff_exponent: i32,
    operation_count_per_dot: u32,
}

fn fixture() -> Fixture {
    let fixture: Fixture = serde_json::from_str(include_str!(
        "../../../../fixtures/deepseek-v41/forward-head-reference.json"
    ))
    .expect("source-forward head fixture");
    assert_eq!(fixture.schema_version, 1);
    assert_eq!(fixture.weight_shape, [VOCAB, WIDTH]);
    assert_eq!(fixture.weight_fp32_bits.len(), VOCAB * WIDTH);
    assert_eq!(fixture.cases.len(), 3);
    assert_eq!(fixture.norm_weight_bf16.len(), WIDTH);
    assert_eq!(fixture.norm_epsilon_bits, 1e-20_f32.to_bits());
    assert_eq!(
        fixture.source.revision,
        "dba1be0a40aa45a94ad051997016db3960a90277"
    );
    assert_eq!(
        fixture.source.model_sha256,
        "4e9ae23620edc8028ccc5d5fef552ab7fdc7dcd6f79608754fe9f67644056f65"
    );
    assert_eq!(
        fixture.source.cpu_backend_sha256,
        "b1f1f3cfdb93b674a5f96a114cf45bf5be9ad3a555ae95ac24add567f9f5232e"
    );
    for digest in [
        &fixture.source.complete_capture_sha256,
        &fixture.source.manifest_canonical_sha256,
    ] {
        assert_eq!(digest.len(), 64, "source capture digest");
        assert!(digest.bytes().all(|byte| byte.is_ascii_hexdigit()));
    }
    assert_eq!(fixture.comparison_policy.kind, "two_fp32_dot_error_bounds");
    assert_eq!(fixture.comparison_policy.operation_count_per_dot, 256);
    assert_eq!(fixture.comparison_policy.unit_roundoff_exponent, -24);
    for (case, (start, positions)) in fixture.cases.iter().zip([(0, 5), (5, 1), (6, 1)]) {
        assert_eq!(case.start_pos, start);
        assert_eq!(case.input_shape, [1, positions, WIDTH]);
        assert_eq!(case.input_bf16.len(), positions * WIDTH);
        assert_eq!(case.logits_shape, [1, VOCAB]);
        assert_eq!(case.logits_fp32_bits.len(), VOCAB);
        assert_eq!(case.final_block_shape, [1, positions, 2, WIDTH]);
        assert_eq!(case.final_block_bf16.len(), positions * 2 * WIDTH);
        assert_eq!(case.final_pre_shape, [1, positions, 2]);
        assert_eq!(case.final_pre_fp32_bits.len(), positions * 2);
        assert_eq!(case.collapsed_bf16.len(), positions * WIDTH);
    }
    fixture
}

fn inputs(case: &Case, position: usize) -> Vec<f32> {
    case.input_bf16[position * WIDTH..(position + 1) * WIDTH]
        .iter()
        .map(|&bits| f32::from_bits(u32::from(bits) << 16))
        .collect()
}

fn logits(input: &[f32], weights: &[f32]) -> Vec<f32> {
    let mut output = vec![0.0; VOCAB];
    fp32_linear_reference(input, weights, 1, WIDTH, VOCAB, &mut output)
        .expect("native finite FP32 output head");
    output
}

fn bounds(input: &[f32], weights: &[f32], policy: &Policy) -> Vec<f64> {
    let unit_roundoff = 2.0_f64.powi(policy.unit_roundoff_exponent);
    let nu = f64::from(policy.operation_count_per_dot) * unit_roundoff;
    let gamma = nu / (1.0 - nu);
    weights
        .chunks_exact(WIDTH)
        .map(|row| {
            let magnitude: f64 = input
                .iter()
                .zip(row)
                .map(|(&x, &w)| {
                    assert!(x.is_normal() || x == 0.0);
                    assert!(w.is_normal() || w == 0.0);
                    let product = x * w;
                    assert!(
                        product.is_normal() || x == 0.0 || w == 0.0,
                        "nonzero operands must not underflow to zero"
                    );
                    (f64::from(x) * f64::from(w)).abs()
                })
                .sum();
            // Inflate the FP64 magnitude sum for its own reduction roundoff.
            let f64_nu = f64::from(policy.operation_count_per_dot) * f64::EPSILON;
            2.0 * gamma * magnitude / (1.0 - f64_nu)
        })
        .collect()
}

fn agrees(actual: &[f32], expected: &[u32], limits: &[f64]) -> bool {
    assert_eq!(actual.len(), VOCAB);
    assert_eq!(expected.len(), VOCAB);
    assert_eq!(limits.len(), VOCAB);
    actual
        .iter()
        .zip(expected)
        .zip(limits)
        .all(|((&actual, &expected), &limit)| {
            let expected = f32::from_bits(expected);
            actual.is_finite()
                && expected.is_finite()
                && (f64::from(actual) - f64::from(expected)).abs() <= limit
        })
}

#[test]
fn native_head_matches_source_forward_logits_after_prefill_and_decode() {
    let fixture = fixture();
    let weights: Vec<f32> = fixture
        .weight_fp32_bits
        .iter()
        .copied()
        .map(f32::from_bits)
        .collect();
    for case in &fixture.cases {
        let input = inputs(case, case.input_shape[1] - 1);
        let limits = bounds(&input, &weights, &fixture.comparison_policy);
        let actual = logits(&input, &weights);
        assert!(
            agrees(&actual, &case.logits_fp32_bits, &limits),
            "source head disagreement at start {}: actual={actual:?}, bounds={limits:?}",
            case.start_pos
        );
    }
}

#[test]
fn source_fixture_distinguishes_first_token_from_last_and_vocabulary_order() {
    let fixture = fixture();
    let weights: Vec<f32> = fixture
        .weight_fp32_bits
        .iter()
        .copied()
        .map(f32::from_bits)
        .collect();
    let case = &fixture.cases[0];
    let last = inputs(case, case.input_shape[1] - 1);
    let limits = bounds(&last, &weights, &fixture.comparison_policy);
    assert!(
        !agrees(
            &logits(&inputs(case, 0), &weights),
            &case.logits_fp32_bits,
            &limits
        ),
        "first prefill position must not qualify as the generation head input"
    );
    let reversed: Vec<f32> = weights
        .chunks_exact(WIDTH)
        .rev()
        .flatten()
        .copied()
        .collect();
    assert!(
        !agrees(&logits(&last, &reversed), &case.logits_fp32_bits, &limits),
        "reversing vocabulary rows must not qualify"
    );
}

#[test]
fn native_final_hc_collapse_norm_and_head_match_source_forward() {
    let fixture = fixture();
    let weights: Vec<f32> = fixture
        .weight_fp32_bits
        .iter()
        .copied()
        .map(f32::from_bits)
        .collect();
    for case in &fixture.cases {
        let mut normalized = vec![0_u16; WIDTH];
        for position in 0..case.input_shape[1] {
            let residual = &case.final_block_bf16[position * 2 * WIDTH..(position + 1) * 2 * WIDTH];
            let pre: Vec<f32> = case.final_pre_fp32_bits[position * 2..(position + 1) * 2]
                .iter()
                .copied()
                .map(f32::from_bits)
                .collect();
            let mut collapsed = vec![0_u16; WIDTH];
            hc_pre_bf16_reference(residual, &pre, WIDTH, &mut collapsed)
                .expect("native final HC collapse");
            assert_eq!(
                collapsed,
                case.collapsed_bf16[position * WIDTH..(position + 1) * WIDTH],
                "exact BF16 collapse at start {}, position {position}",
                case.start_pos
            );
            rms_norm_bf16_reference(
                &collapsed,
                &fixture.norm_weight_bf16,
                f32::from_bits(fixture.norm_epsilon_bits),
                &mut normalized,
            )
            .expect("native final normalization");
            assert_eq!(
                normalized,
                case.input_bf16[position * WIDTH..(position + 1) * WIDTH],
                "exact BF16 normalization at start {}, position {position}",
                case.start_pos
            );
        }
        let native_input: Vec<f32> = normalized
            .iter()
            .map(|&bits| f32::from_bits(u32::from(bits) << 16))
            .collect();
        let limits = bounds(&native_input, &weights, &fixture.comparison_policy);
        assert!(
            agrees(
                &logits(&native_input, &weights),
                &case.logits_fp32_bits,
                &limits
            ),
            "native tail logits at start {}",
            case.start_pos
        );
    }
}

#[test]
fn tail_fixture_rejects_bypassing_hc_or_normalization() {
    let fixture = fixture();
    let case = &fixture.cases[0];
    let mut bypassed_hc = vec![0_u16; WIDTH];
    hc_pre_bf16_reference(
        &case.final_block_bf16[..2 * WIDTH],
        &[1.0, 0.0],
        WIDTH,
        &mut bypassed_hc,
    )
    .expect("one-copy negative control");
    assert_ne!(
        bypassed_hc,
        case.collapsed_bf16[..WIDTH],
        "passing one residual copy through must not match HC collapse"
    );
    assert_ne!(
        case.collapsed_bf16, case.input_bf16,
        "omitting final normalization must change the observed boundary"
    );
}
