use super::{
    AttentionQrLayout, AttentionQrWeights, CompressedAttentionPublication, Fp8Projection,
    LayerAttentionError, LayerAttentionLayout, LayerAttentionLayoutError, LayerAttentionState,
    ops::rotate_bf16_tail, prepare_attention_qr, state::TailShape,
};
use crate::{attention::window::WindowStep, precision::f32_to_bf16_rne};
use std::num::NonZeroUsize;

fn nonzero(value: usize) -> NonZeroUsize {
    NonZeroUsize::new(value).expect("test dimensions are nonzero")
}

fn layout() -> LayerAttentionLayout {
    LayerAttentionLayout::new(
        nonzero(1),
        nonzero(32),
        nonzero(2),
        nonzero(32),
        nonzero(1),
        nonzero(32),
        nonzero(4),
        nonzero(1),
        nonzero(32),
        7,
        nonzero(1),
        1e-5,
        0.25,
    )
    .expect("small grouped layout")
}

fn qr_layout() -> AttentionQrLayout {
    AttentionQrLayout::new(nonzero(1), nonzero(32), nonzero(32), 1e-5).expect("small QR layout")
}

fn qr_weights() -> AttentionQrWeights<'static> {
    AttentionQrWeights {
        wq_a: Fp8Projection {
            codes: &[0x38; 32 * 32],
            scales: &[127],
        },
        q_norm: &[0x3f80; 32],
    }
}

#[test]
fn stateless_qr_keeps_projection_and_norm_stages() {
    let result = prepare_attention_qr(&[0x3f80; 32], qr_weights(), qr_layout())
        .expect("finite source-shaped QR");
    assert_eq!(result.wq_a.len(), 32);
    assert_eq!(result.qr.len(), 32);
    assert!(result.wq_a.iter().all(|&bits| bits == 0x4200));
    assert!(result.qr.iter().all(|&bits| bits == 0x3f80));
}

#[test]
fn stateless_qr_consumes_connected_attention_input_in_token_order() {
    let norm = [0x3f80; 32];
    let input =
        crate::reduced::AttentionInput::new(&norm, 4, 1e-5).expect("four-copy attention input");
    let mut normalized = Vec::new();
    for value in [0x3f80, 0xbf80, 0x3f80] {
        let prepared = input
            .forward(&[value; 4 * 32], &[0.25; 4])
            .expect("connected collapse and attention norm");
        assert_eq!(prepared.collapsed_bf16(), &[value; 32]);
        normalized.extend_from_slice(prepared.normalized_bf16());
    }
    let weights = AttentionQrWeights::new(qr_weights().wq_a, &norm);
    let result =
        prepare_attention_qr(&normalized, weights, qr_layout()).expect("multi-token stateless QR");
    let expected_projection: Vec<_> = [0x4200, 0xc200, 0x4200]
        .into_iter()
        .flat_map(|value| [value; 32])
        .collect();
    let expected_qr: Vec<_> = [0x3f80, 0xbf80, 0x3f80]
        .into_iter()
        .flat_map(|value| [value; 32])
        .collect();
    assert_eq!(result.wq_a_bf16(), expected_projection);
    assert_eq!(result.qr_bf16(), expected_qr);
}

#[test]
#[allow(
    clippy::cast_precision_loss,
    reason = "small deterministic fixture indices"
)]
fn query_rotary_matches_source_tail_slice_reference() {
    // DeepSeek splits each projected query head into a non-rotary prefix
    // and a trailing qk_rope_head_dim slice. The source applies RoPE only
    // to that trailing slice after wq_b, broadcasting frequencies by
    // position and pair over heads.
    let head_dimension = 8;
    let rope_pairs = 2;
    let batches = 1;
    let positions = 2;
    let heads = 2;
    let values = (0..batches * positions * heads * head_dimension)
        .map(|index| f32_to_bf16_rne(index as f32 + 1.0))
        .collect::<Vec<_>>();
    let frequencies = [
        crate::RotaryFrequency::new(0.8, 0.6).expect("finite source frequency"),
        crate::RotaryFrequency::new(-0.5, 0.25).expect("finite source frequency"),
        crate::RotaryFrequency::new(0.25, -0.75).expect("finite source frequency"),
        crate::RotaryFrequency::new(0.6, 0.2).expect("finite source frequency"),
    ];
    let actual = rotate_bf16_tail(
        &values,
        TailShape {
            batches: nonzero(batches),
            positions,
            heads: nonzero(heads),
            head_dimension: nonzero(head_dimension),
            rope_pairs: nonzero(rope_pairs),
        },
        &frequencies,
        crate::RotaryDirection::Forward,
    )
    .expect("source-shaped query tail rotation");

    let prefix = head_dimension - rope_pairs * 2;
    let mut expected = values.clone();
    for position in 0..positions {
        for head in 0..heads {
            let row = (position * heads + head) * head_dimension;
            for pair in 0..rope_pairs {
                let input = [
                    f32::from_bits(u32::from(values[row + prefix + pair * 2]) << 16),
                    f32::from_bits(u32::from(values[row + prefix + pair * 2 + 1]) << 16),
                ];
                let frequency = frequencies[position * rope_pairs + pair];
                let rotated = [
                    input[0] * frequency.real() - input[1] * frequency.imaginary(),
                    input[0] * frequency.imaginary() + input[1] * frequency.real(),
                ];
                expected[row + prefix + pair * 2] = f32_to_bf16_rne(rotated[0]);
                expected[row + prefix + pair * 2 + 1] = f32_to_bf16_rne(rotated[1]);
            }
        }
    }
    assert_eq!(actual, expected);
    for position in 0..positions {
        for head in 0..heads {
            let row = (position * heads + head) * head_dimension;
            assert_eq!(
                &actual[row..row + prefix],
                &values[row..row + prefix],
                "RoPE must preserve the non-rotary query prefix"
            );
        }
    }
}

#[test]
fn stateless_qr_rejects_invalid_shape_and_layout_invariants() {
    assert!(matches!(
        AttentionQrLayout::new(nonzero(1), nonzero(32), nonzero(32), 0.0),
        Err(LayerAttentionLayoutError::InvalidNormEpsilon)
    ));
    assert!(matches!(
        AttentionQrLayout::new(nonzero(1), nonzero(31), nonzero(32), 1e-5),
        Err(LayerAttentionLayoutError::UngroupedFp8Reduction {
            field: "hidden dimension",
            width: 31
        })
    ));
    assert!(matches!(
        prepare_attention_qr(&[0x3f80; 31], qr_weights(), qr_layout()),
        Err(LayerAttentionError::InputLength {
            actual: 31,
            stride: 32
        })
    ));
}

#[test]
fn stateless_qr_rejects_an_oversized_one_position_input_before_allocation() {
    assert!(matches!(
        AttentionQrLayout::new(nonzero(262_145), nonzero(32), nonzero(32), 1e-5),
        Err(LayerAttentionLayoutError::ElementLimit {
            field: "one-position input",
            elements: 8_388_640
        })
    ));
}

#[test]
fn layout_retains_explicit_compressed_producer() {
    assert_eq!(layout().expected_source_layer(), 7);
}

#[test]
fn fp8_reduction_boundary_is_validated_up_front() {
    let error = LayerAttentionLayout::new(
        nonzero(1),
        nonzero(31),
        nonzero(1),
        nonzero(32),
        nonzero(1),
        nonzero(32),
        nonzero(4),
        nonzero(1),
        nonzero(32),
        0,
        nonzero(1),
        1e-5,
        0.25,
    )
    .expect_err("ungrouped input cannot enter FP8 projection");
    assert!(matches!(
        error,
        LayerAttentionLayoutError::UngroupedFp8Reduction {
            field: "hidden dimension",
            width: 31
        }
    ));
}

#[test]
fn reset_is_an_explicit_epoch_transition() {
    let mut state = LayerAttentionState::new(layout());
    state.reset().expect("first epoch advance fits");
    assert_eq!(state.epoch, 1);
    assert_eq!(state.next_position, None);
    assert_eq!(state.next_call_id, 0);
}

#[test]
fn zero_compressed_prefix_is_valid_and_mismatched_prefix_is_atomic() {
    let layout = LayerAttentionLayout::new(
        nonzero(1),
        nonzero(32),
        nonzero(2),
        nonzero(32),
        nonzero(1),
        nonzero(32),
        nonzero(4),
        nonzero(1),
        nonzero(32),
        7,
        nonzero(2),
        1e-5,
        0.25,
    )
    .expect("small ratio-two layout");
    let state = LayerAttentionState::new(layout);
    let step = WindowStep::Prefill {
        tokens: NonZeroUsize::MIN,
    };
    state
        .validate_publication(
            CompressedAttentionPublication {
                source_layer: 7,
                epoch: 0,
                call_id: 0,
                numerical_bf16: &[],
                indices: &[],
            },
            0,
            0,
            0,
            1,
            step,
        )
        .expect("ratio-two first token has a valid window-only prefix");
    let before = (
        state.ring.clone(),
        state.next_position,
        state.epoch,
        state.next_call_id,
    );
    let phantom_slots = state
        .validate_publication(
            CompressedAttentionPublication {
                source_layer: 7,
                epoch: 0,
                call_id: 0,
                numerical_bf16: &[],
                indices: &[-1],
            },
            0,
            0,
            0,
            1,
            step,
        )
        .expect_err("zero compressed keys cannot carry placeholder slots");
    assert!(matches!(
        phantom_slots,
        LayerAttentionError::CompressedSlotsWithoutKeys { slots: 1 }
    ));
    let error = state
        .validate_publication(
            CompressedAttentionPublication {
                source_layer: 7,
                epoch: 0,
                call_id: 0,
                numerical_bf16: &[0; 32],
                indices: &[],
            },
            0,
            0,
            0,
            1,
            step,
        )
        .expect_err("one compressed key is too long for the first ratio-two token");
    assert!(matches!(
        error,
        LayerAttentionError::CompressedKeyCount {
            actual: 1,
            expected: 0
        }
    ));
    assert_eq!(
        (
            state.ring,
            state.next_position,
            state.epoch,
            state.next_call_id
        ),
        before
    );
}

#[test]
fn duplicate_compressed_indices_are_rejected_per_query_row() {
    let layout = LayerAttentionLayout::new(
        nonzero(1),
        nonzero(32),
        nonzero(2),
        nonzero(32),
        nonzero(1),
        nonzero(32),
        nonzero(4),
        nonzero(1),
        nonzero(32),
        7,
        nonzero(2),
        1e-5,
        0.25,
    )
    .expect("small ratio-two layout");
    let state = LayerAttentionState::new(layout);
    let error = state
        .validate_publication(
            CompressedAttentionPublication {
                source_layer: 7,
                epoch: 0,
                call_id: 0,
                numerical_bf16: &[0; 32],
                indices: &[4, 4],
            },
            0,
            0,
            1,
            1,
            WindowStep::Decode {
                position: NonZeroUsize::MIN,
            },
        )
        .expect_err("a source top-k row cannot select one compressed key twice");
    assert!(matches!(
        error,
        LayerAttentionError::DuplicateCompressedIndex { row: 0, index: 4 }
    ));
}
