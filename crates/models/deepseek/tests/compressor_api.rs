//! Public compressor API qualification against the pinned V4.1 CPU capture.
//!
//! Fixture outputs come from `Compressor.forward` plus `RMSNorm.forward` at
//! the checked-in source revision.  The adapter below supplies its documented
//! identity-KV and reversed/scaled gate stubs to the real Rust API; it does
//! not reproduce pooling or normalization arithmetic.

use deepseek::compressor::{CompressorError, CompressorInput, CompressorState};
use serde::Deserialize;

const BATCHES: usize = 2;
const WIDTH: usize = 4;

#[derive(Deserialize)]
struct Fixture {
    schema_version: u32,
    source: Source,
    batches: usize,
    width: usize,
    epsilon: f32,
    norm_weight_bf16: Vec<u16>,
    cases: Vec<Case>,
}

#[derive(Deserialize)]
struct Source {
    revision: String,
    sha256: String,
    symbols: Vec<String>,
}

#[derive(Deserialize)]
struct Case {
    ratio: usize,
    prefix: usize,
    calls: Vec<Call>,
}

#[derive(Deserialize)]
struct Call {
    start: usize,
    positions: usize,
    input_bf16: Vec<u16>,
    output_bf16: Option<Vec<u16>>,
}

fn fixture() -> Fixture {
    serde_json::from_str(include_str!(
        "../../../../fixtures/deepseek-v41/compressor-reference.json"
    ))
    .expect("checked-in Compressor capture")
}

fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}

/// Builds the explicit projection stubs used to create the fixture.
///
/// Ratio one directly accepts the captured BF16 projection. Larger ratios use
/// identity KV projection and a gate that reverses each width-four row then
/// scales it by 0.75. The pinned capture is the pooling oracle; the library
/// computes the actual result.
fn gated_fixture_inputs(input_bf16: &[u16]) -> (Vec<f32>, Vec<f32>) {
    let kv: Vec<f32> = input_bf16.iter().copied().map(bf16_to_f32).collect();
    let mut scores = Vec::with_capacity(kv.len());
    for row in kv.chunks_exact(WIDTH) {
        scores.extend(row.iter().rev().map(|value| value * 0.75));
    }
    (kv, scores)
}

fn forward_fixture_call(
    state: &mut CompressorState,
    ratio: usize,
    call: &Call,
) -> Result<Option<Vec<u16>>, CompressorError> {
    if ratio == 1 {
        state.forward(
            CompressorInput::ProjectedBf16(&call.input_bf16),
            call.positions,
            call.start,
        )
    } else {
        let (kv, scores) = gated_fixture_inputs(&call.input_bf16);
        state.forward(
            CompressorInput::Gated {
                kv: &kv,
                scores: &scores,
            },
            call.positions,
            call.start,
        )
    }
}

fn assert_error(result: Result<Option<Vec<u16>>, CompressorError>) {
    result.expect_err("request must be rejected");
}

#[test]
fn public_api_matches_pinned_compressor_capture_and_advances_after_each_call() {
    let fixture = fixture();
    assert_eq!(fixture.schema_version, 1, "fixture schema provenance");
    assert_eq!(
        fixture.source.revision,
        "dba1be0a40aa45a94ad051997016db3960a90277"
    );
    assert_eq!(
        fixture.source.sha256,
        "4e9ae23620edc8028ccc5d5fef552ab7fdc7dcd6f79608754fe9f67644056f65"
    );
    assert_eq!(
        fixture.source.symbols,
        vec![
            String::from("Compressor.forward"),
            String::from("RMSNorm.forward"),
        ]
    );
    assert_eq!((fixture.batches, fixture.width), (BATCHES, WIDTH));
    assert!(!fixture.cases.is_empty(), "fixture contains captured cases");

    for case in &fixture.cases {
        let mut state = CompressorState::new(
            BATCHES,
            WIDTH,
            case.ratio,
            &fixture.norm_weight_bf16,
            fixture.epsilon,
        )
        .expect("captured geometry is valid");
        for call in &case.calls {
            let actual = forward_fixture_call(&mut state, case.ratio, call)
                .expect("captured request is valid");
            assert_eq!(
                actual.as_ref(),
                call.output_bf16.as_ref(),
                "ratio {} prefix {} start {} exact BF16 words",
                case.ratio,
                case.prefix,
                call.start,
            );
            assert_eq!(
                state.next_position(),
                call.start + call.positions,
                "ratio {} prefix {} start {} position",
                case.ratio,
                case.prefix,
                call.start,
            );
        }
    }
}

#[test]
fn ratio_four_uniform_gates_pool_before_rms_norm() {
    // Four identical rows have uniform softmax weights, so their pooled row is
    // exactly `[1, -1]`; unit RMSNorm weights preserve those BF16 words.
    let mut state =
        CompressorState::new(1, 2, 4, &[0x3f80, 0x3f80], 1e-6).expect("small valid compressor");
    let kv = [1.0, -1.0].repeat(4);
    let scores = vec![0.0; 8];
    let output = state
        .forward(
            CompressorInput::Gated {
                kv: &kv,
                scores: &scores,
            },
            4,
            0,
        )
        .expect("uniform gated prefill")
        .expect("one complete group");
    assert_eq!(output, vec![0x3f80, 0xbf80]);
    assert_eq!(state.next_position(), 4);
}

#[test]
fn gated_scores_select_different_tokens_for_each_feature() {
    // The first feature assigns 3:1 weight to token one; the second assigns
    // 3:1 weight to token zero. Thus the analytically pooled pre-norm row is
    // `[2.5, 3.5]`, not a token-wise weighted row. These exact words include
    // the required BF16 narrowing before unit-weight RMSNorm.
    let mut state =
        CompressorState::new(1, 2, 2, &[0x3f80, 0x3f80], 1e-6).expect("small valid compressor");
    let kv = [1.0, 4.0, 3.0, 2.0];
    let log_three = 3.0_f32.ln();
    let scores = [0.0, log_three, log_three, 0.0];
    let output = state
        .forward(
            CompressorInput::Gated {
                kv: &kv,
                scores: &scores,
            },
            2,
            0,
        )
        .expect("finite per-feature gates")
        .expect("completed group");
    assert_eq!(output, vec![0x3f52, 0x3f93]);
    assert_eq!(state.next_position(), 2);
}

#[test]
fn softmax_underflow_in_an_earlier_token_still_emits_the_later_token() {
    // A stable softmax may underflow an earlier exp(score - max) to zero. The
    // nonzero later term remains a valid denominator and selects KV value one.
    let mut state = CompressorState::new(1, 1, 2, &[0x3f80], 1e-6).expect("small valid compressor");
    let kv = [100.0, 1.0];
    let scores = [-1000.0, 0.0];
    let output = state
        .forward(
            CompressorInput::Gated {
                kv: &kv,
                scores: &scores,
            },
            2,
            0,
        )
        .expect("stable finite softmax")
        .expect("completed group");
    assert_eq!(output, vec![0x3f80]);
    assert_eq!(state.next_position(), 2);
}

#[test]
fn constructor_rejects_invalid_scalars_and_oversized_geometry_without_buffers() {
    let unit = [0x3f80];
    assert!(CompressorState::new(0, 1, 1, &unit, 1e-6).is_err());
    assert!(CompressorState::new(1, 0, 1, &[], 1e-6).is_err());
    assert!(CompressorState::new(1, 1, 0, &unit, 1e-6).is_err());
    assert!(CompressorState::new(usize::MAX, 1, 1, &unit, 1e-6).is_err());
    assert!(CompressorState::new(1, 1, 1, &[0x7fc0], 1e-6).is_err());
    assert!(CompressorState::new(1, 1, 1, &unit, f32::NAN).is_err());
    assert!(CompressorState::new(1, 1, 1, &unit, 0.0).is_err());

    let wider_than_rms_norm = vec![0x3f80; deepseek::MAX_RMS_NORM_WIDTH + 1];
    assert!(
        CompressorState::new(1, wider_than_rms_norm.len(), 1, &wider_than_rms_norm, 1e-6,).is_err()
    );
}

#[test]
fn input_mode_and_nonfinite_requests_are_rejected_without_advancing() {
    let mut gated =
        CompressorState::new(1, 2, 2, &[0x3f80, 0x3f80], 1e-6).expect("valid gated compressor");
    assert_error(gated.forward(CompressorInput::ProjectedBf16(&[0x3f80, 0x3f80]), 1, 0));
    assert_eq!(gated.next_position(), 0);

    let kv = [1.0, 1.0];
    let scores = [f32::NAN, 0.0];
    assert_error(gated.forward(
        CompressorInput::Gated {
            kv: &kv,
            scores: &scores,
        },
        1,
        0,
    ));
    assert_eq!(gated.next_position(), 0);

    let mut projected =
        CompressorState::new(1, 2, 1, &[0x3f80, 0x3f80], 1e-6).expect("valid projected compressor");
    let kv = [1.0, 1.0];
    let scores = [0.0, 0.0];
    assert_error(projected.forward(
        CompressorInput::Gated {
            kv: &kv,
            scores: &scores,
        },
        1,
        0,
    ));
    assert_eq!(projected.next_position(), 0);
    assert_error(projected.forward(CompressorInput::ProjectedBf16(&[0x7fc0, 0x3f80]), 1, 0));
    assert_eq!(projected.next_position(), 0);

    let kv = [f32::NAN, 1.0];
    let scores = [0.0, 0.0];
    assert_error(gated.forward(
        CompressorInput::Gated {
            kv: &kv,
            scores: &scores,
        },
        1,
        0,
    ));
    assert_eq!(gated.next_position(), 0);
}

#[test]
fn continuation_requires_prior_sequential_prefill_without_mutating_state() {
    let mut state =
        CompressorState::new(1, 2, 3, &[0x3f80, 0x3f80], 1e-6).expect("valid compressor");
    let kv = [1.0, -1.0];
    let scores = [0.0, 0.0];
    assert_error(state.forward(
        CompressorInput::Gated {
            kv: &kv,
            scores: &scores,
        },
        1,
        1,
    ));
    assert_eq!(state.next_position(), 0);

    assert_error(state.forward(
        CompressorInput::Gated {
            kv: &kv,
            scores: &scores,
        },
        1,
        2,
    ));
    assert_eq!(state.next_position(), 0);
}

#[test]
fn failed_reset_or_late_norm_failure_preserves_prior_continuation() {
    // This prefill leaves two of three slots pending. A reset-sized complete
    // group with finite huge inputs reaches RMSNorm and overflows its square;
    // neither the proposed reset nor its failed completion may replace slots.
    let weight = [0x3f80, 0x3f80];
    let mut baseline = CompressorState::new(1, 2, 3, &weight, 1e-6).expect("baseline");
    let mut candidate = CompressorState::new(1, 2, 3, &weight, 1e-6).expect("candidate");
    let partial_kv = [1.0, -1.0, 2.0, -2.0];
    let partial_scores = [0.0; 4];
    let huge_kv = [1e30; 6];
    let huge_scores = [0.0; 6];
    let completion_kv = [3.0, -3.0];
    let completion_scores = [0.0; 2];

    baseline
        .forward(
            CompressorInput::Gated {
                kv: &partial_kv,
                scores: &partial_scores,
            },
            2,
            0,
        )
        .expect("baseline prefill");
    candidate
        .forward(
            CompressorInput::Gated {
                kv: &partial_kv,
                scores: &partial_scores,
            },
            2,
            0,
        )
        .expect("candidate prefill");
    assert!(matches!(
        candidate.forward(
            CompressorInput::Gated {
                kv: &huge_kv[..2],
                scores: &huge_scores[..2],
            },
            1,
            2,
        ),
        Err(CompressorError::RmsNorm(
            deepseek::norm::RmsNormError::ValueOverflow { .. }
        ))
    ));
    assert_eq!(
        candidate.next_position(),
        2,
        "failed singleton retry is atomic"
    );
    assert!(matches!(
        candidate.forward(
            CompressorInput::Gated {
                kv: &huge_kv,
                scores: &huge_scores,
            },
            3,
            0,
        ),
        Err(CompressorError::RmsNorm(
            deepseek::norm::RmsNormError::ValueOverflow { .. }
        ))
    ));
    assert_eq!(candidate.next_position(), 2);

    let expected = baseline
        .forward(
            CompressorInput::Gated {
                kv: &completion_kv,
                scores: &completion_scores,
            },
            1,
            2,
        )
        .expect("baseline continuation");
    let actual = candidate
        .forward(
            CompressorInput::Gated {
                kv: &completion_kv,
                scores: &completion_scores,
            },
            1,
            2,
        )
        .expect("continuation after failed reset");
    assert_eq!(actual, expected);
    assert_eq!(candidate.next_position(), baseline.next_position());
}
