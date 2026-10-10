//! G1-block and G1-quant (bars declared in the shared-MoE plan before any
//! run): the routed-expert block against transformers' `Gemma4TextRouter` and
//! `Gemma4TextExperts` on CPU (`fixtures/moe/gemma4-block.json`, written by
//! `scripts/moe-block-reference.py`), and affine experts against their
//! dequantized twin.

use mlx_rs::{Array, ops};
use serde::Deserialize;

use super::{AffineMatrix, Experts, Matrix, Routing, route, routed_experts, run_experts};
use crate::{
    GPU_TEST_LOCK,
    moe::{Activation, MoeError, Router, SORT_THRESHOLD},
};

/// Routing weights, against the source's float32.
const WEIGHT_BOUND: f32 = 1e-6;
/// Routed output, against the source's float32.
const OUTPUT_BOUND: f32 = 1e-5;
/// Sorted against unsorted dispatch on the same input.
const SORTED_BOUND: f32 = 1e-5;
/// Affine experts against the dense experts they dequantize to.
const AFFINE_BOUND: f32 = 1e-3;

#[derive(Deserialize)]
struct Fixture {
    hidden: usize,
    experts: usize,
    intermediate: usize,
    activation: String,
    tie_rule: String,
    cases: Vec<Case>,
}

#[derive(Deserialize)]
struct Case {
    name: String,
    top_k: usize,
    tokens: usize,
    expert_input: Vec<Vec<f32>>,
    per_expert_scale: Vec<f32>,
    gate_up: Vec<Vec<Vec<f32>>>,
    down: Vec<Vec<Vec<f32>>>,
    logits: Vec<Vec<f32>>,
    indices: Vec<Vec<u32>>,
    weights: Vec<Vec<f32>>,
    output: Vec<Vec<f32>>,
    #[serde(default)]
    tied_experts: Vec<u32>,
}

fn fixture() -> Fixture {
    serde_json::from_str(include_str!(
        "../../../../../../fixtures/moe/gemma4-block.json"
    ))
    .expect("fixture JSON")
}

fn array<const N: usize>(rows: &[Vec<f32>], shape: [usize; N]) -> Array {
    let flat: Vec<f32> = rows.iter().flatten().copied().collect();
    let shape: Vec<i32> = shape
        .iter()
        .map(|&d| i32::try_from(d).expect("small dim"))
        .collect();
    Array::from_slice(&flat, &shape)
}

fn stack(matrices: &[Vec<Vec<f32>>]) -> Array {
    let rows: Vec<Vec<f32>> = matrices.iter().flatten().cloned().collect();
    let shape = [matrices.len(), matrices[0].len(), matrices[0][0].len()];
    array(&rows, shape)
}

fn values_f32(array: &Array) -> Vec<f32> {
    // A top-k slice of the argsort is strided; read a row-contiguous copy.
    let array = array.contiguous().expect("contiguous");
    array.eval().expect("eval");
    array.as_slice::<f32>().to_vec()
}

fn values_u32(array: &Array) -> Vec<u32> {
    // A top-k slice of the argsort is strided; read a row-contiguous copy.
    let array = array.contiguous().expect("contiguous");
    array.eval().expect("eval");
    array.as_slice::<u32>().to_vec()
}

fn max_difference(actual: &[f32], expected: &[f32]) -> f32 {
    assert_eq!(actual.len(), expected.len());
    actual
        .iter()
        .zip(expected)
        .map(|(a, e)| {
            assert!(a.is_finite(), "non-finite output {a}");
            (a - e).abs()
        })
        .fold(0.0, f32::max)
}

struct Run {
    routing: Routing,
    gate_up: Array,
    down: Array,
    input: Array,
}

fn run(fixture: &Fixture, case: &Case) -> Run {
    let router = Router::softmax(case.top_k, true, true).expect("router");
    let logits = array(&case.logits, [case.tokens, fixture.experts]);
    let scale = Array::from_slice(&case.per_expert_scale, &[8]);
    Run {
        routing: route(&logits, &router, Some(&scale)).expect("route"),
        gate_up: stack(&case.gate_up),
        down: stack(&case.down),
        input: array(&case.expert_input, [case.tokens, fixture.hidden]),
    }
}

#[test]
fn routing_and_experts_match_the_transformers_block() {
    let _gpu = GPU_TEST_LOCK.lock().expect("GPU test lock");
    let fixture = fixture();
    assert_eq!(
        (fixture.hidden, fixture.experts, fixture.intermediate),
        (16, 8, 4)
    );
    assert_eq!(fixture.activation, "gelu_pytorch_tanh");
    // The rule the crate implements; the fixture's probe must agree.
    assert_eq!(fixture.tie_rule, "lowest_index");
    let mut sorted_case_seen = false;
    for case in &fixture.cases {
        let run = run(&fixture, case);
        let chosen = values_u32(&run.routing.chosen);
        let weights = values_f32(&run.routing.weights);
        for token in 0..case.tokens {
            let span = token * case.top_k..(token + 1) * case.top_k;
            let mut ours: Vec<(u32, f32)> = chosen[span.clone()]
                .iter()
                .copied()
                .zip(weights[span].iter().copied())
                .collect();
            let mut theirs: Vec<(u32, f32)> = case.indices[token]
                .iter()
                .copied()
                .zip(case.weights[token].iter().copied())
                .collect();
            ours.sort_by_key(|&(expert, _)| expert);
            theirs.sort_by_key(|&(expert, _)| expert);
            let ours_ids: Vec<u32> = ours.iter().map(|&(e, _)| e).collect();
            let theirs_ids: Vec<u32> = theirs.iter().map(|&(e, _)| e).collect();
            assert_eq!(ours_ids, theirs_ids, "{} token {token} experts", case.name);
            for ((expert, ours), (_, theirs)) in ours.iter().zip(&theirs) {
                assert!(
                    (ours - theirs).abs() <= WEIGHT_BOUND,
                    "{} token {token} expert {expert}: weight {ours} vs {theirs}",
                    case.name
                );
            }
        }
        if let [low, high] = case.tied_experts[..] {
            // The tied pair straddles the boundary: exactly the lower index
            // is kept, by the reference and by us.
            assert!(low < high);
            assert!(
                chosen.contains(&low) && !chosen.contains(&high),
                "{}",
                case.name
            );
            assert!(case.indices[0].contains(&low) && !case.indices[0].contains(&high));
        }
        let experts = Experts::Fused {
            gate_up: Matrix::Dense(&run.gate_up),
            down: Matrix::Dense(&run.down),
        };
        let output = routed_experts(&run.input, &experts, &run.routing, Activation::GeluTanh)
            .expect("routed experts");
        let expected: Vec<f32> = case.output.iter().flatten().copied().collect();
        let difference = max_difference(&values_f32(&output), &expected);
        assert!(
            difference <= OUTPUT_BOUND,
            "{}: max |d| {difference:e} over {OUTPUT_BOUND:e}",
            case.name
        );
        if run.routing.chosen.size() >= SORT_THRESHOLD {
            sorted_case_seen = true;
            let unsorted = run_experts(
                &run.input,
                &experts,
                &run.routing,
                Activation::GeluTanh,
                false,
            )
            .expect("unsorted");
            let difference = max_difference(&values_f32(&output), &values_f32(&unsorted));
            assert!(
                difference <= SORTED_BOUND,
                "{}: sorted vs unsorted {difference:e}",
                case.name
            );
        }
    }
    assert!(sorted_case_seen, "no case reached the sorted path");
}

#[test]
fn affine_experts_match_their_dequantized_twin() {
    let _gpu = GPU_TEST_LOCK.lock().expect("GPU test lock");
    // 64-wide inputs on both projections, so 64-element groups divide them.
    let (experts, hidden, intermediate, top_k) = (8, 64, 64, 3);
    let values = |count: i32, salt: i32| -> Vec<f32> {
        (0..count)
            .map(|i| {
                let bucket = (i * 7_919 + salt * 104_729).rem_euclid(2_001);
                f32::from(i16::try_from(bucket).expect("bucket")) / 4_000.0 - 0.25
            })
            .collect()
    };
    let gate_up = Array::from_slice(
        &values(experts * 2 * intermediate * hidden, 1),
        &[experts, 2 * intermediate, hidden],
    );
    let down = Array::from_slice(
        &values(experts * hidden * intermediate, 2),
        &[experts, hidden, intermediate],
    );
    let pack = |dense: &Array| {
        let (codes, scales, biases) = ops::quantize(dense, 64, 4).expect("quantize");
        let twin = ops::dequantize(&codes, &scales, &biases, 64, 4).expect("dequantize");
        (
            AffineMatrix {
                codes,
                scales,
                biases,
                group_size: 64,
                bits: 4,
            },
            twin,
        )
    };
    let (gate_up_q, gate_up_twin) = pack(&gate_up);
    let (down_q, down_twin) = pack(&down);
    let affine = Experts::Fused {
        gate_up: Matrix::Affine(&gate_up_q),
        down: Matrix::Affine(&down_q),
    };
    let dense = Experts::Fused {
        gate_up: Matrix::Dense(&gate_up_twin),
        down: Matrix::Dense(&down_twin),
    };
    let router = Router::softmax(top_k, true, false).expect("router");
    for tokens in [1, 30] {
        let input = Array::from_slice(&values(tokens * hidden, 3), &[tokens, hidden]);
        let logits = Array::from_slice(&values(tokens * experts, 4), &[tokens, experts]);
        let routing = route(&logits, &router, None).expect("route");
        let quantized =
            routed_experts(&input, &affine, &routing, Activation::Silu).expect("affine");
        let reference = routed_experts(&input, &dense, &routing, Activation::Silu).expect("dense");
        let difference = max_difference(&values_f32(&quantized), &values_f32(&reference));
        assert!(
            difference <= AFFINE_BOUND,
            "{tokens} tokens: affine vs dequantized {difference:e}"
        );
    }
}

#[test]
fn refuses_routing_and_stacks_that_do_not_compose() {
    let _gpu = GPU_TEST_LOCK.lock().expect("GPU test lock");
    let logits = Array::from_slice(&[0.0_f32; 8], &[2, 4]);
    let too_many = Router::softmax(5, true, false).expect("router");
    assert!(matches!(
        route(&logits, &too_many, None),
        Err(MoeError::TopK {
            top_k: 5,
            experts: Some(4)
        })
    ));
    let scaled = Router::softmax(2, true, true).expect("router");
    assert!(matches!(
        route(&logits, &scaled, None),
        Err(MoeError::MissingPerExpertScale)
    ));
    let wrong_scale = Array::from_slice(&[1.0_f32; 3], &[3]);
    assert!(matches!(
        route(&logits, &scaled, Some(&wrong_scale)),
        Err(MoeError::Shape(_))
    ));
    let routing = route(
        &logits,
        &Router::softmax(2, true, false).expect("router"),
        None,
    )
    .expect("route");
    let gate_up = Array::from_slice(&[0.0_f32; 4 * 6 * 3], &[4, 6, 3]);
    let wrong_down = Array::from_slice(&[0.0_f32; 4 * 3 * 2], &[4, 3, 2]);
    let experts = Experts::Fused {
        gate_up: Matrix::Dense(&gate_up),
        down: Matrix::Dense(&wrong_down),
    };
    let input = Array::from_slice(&[0.0_f32; 6], &[2, 3]);
    assert!(matches!(
        routed_experts(&input, &experts, &routing, Activation::Silu),
        Err(MoeError::Shape(_))
    ));
}
