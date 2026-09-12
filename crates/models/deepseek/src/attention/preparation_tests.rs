//! Test-only scalar composition for the V4.1 attention preparation stages.
//!
//! These synthetic cases cover the pinned 32-element FP8 grouping through
//! BF16 projection, RMS normalization, activation requantization, and `RoPE`
//! staging. They deliberately stop before sparse attention, cache handling,
//! and any GPU or full-model parity claim.

use std::num::NonZeroUsize;

use crate::precision::{
    ActivationGroup, fp8_linear_runtime_f32, quantize_bf16_activations_e4m3fn,
    requantize_bf16_activations_e4m3fn,
};
use crate::{
    RotaryDirection, RotaryFrequency, RotaryTailLayout, rms_norm_bf16_reference, rotate_tail,
};

const WIDTH: usize = 32;
const BF16_ONE: u16 = 0x3f80;
const BF16_POINT_SIX_TWO_FIVE: u16 = 0x3f20;
const BF16_NEGATIVE_ONE_POINT_TWO_FIVE: u16 = 0xbfa0;
const BF16_ONE_POINT_TWO_FIVE: u16 = 0x3fa0;
const BF16_POINT_SIX_THREE_TWO_EIGHT_ONE_TWO_FIVE: u16 = 0x3f22;
const BF16_ONE_POINT_TWO_SIX_FIVE_SIX_TWO_FIVE: u16 = 0x3fa2;
const BF16_FIFTEEN: u16 = 0x4170;
const BF16_NEGATIVE_FIFTEEN: u16 = 0xc170;
const BF16_THIRTY: u16 = 0x41f0;
const BF16_NEGATIVE_THIRTY: u16 = 0xc1f0;

fn fp8_project_bf16(
    input: &[u16; WIDTH],
    weight_codes: &[u8],
    weight_scales: &[u8],
    outputs: usize,
) -> Vec<u16> {
    let mut activation_codes = [0_u8; WIDTH];
    let mut activation_scales = [0_u8; 1];
    quantize_bf16_activations_e4m3fn(
        input,
        1,
        WIDTH,
        ActivationGroup::Elements32,
        &mut activation_codes,
        &mut activation_scales,
    )
    .expect("synthetic BF16 activations fit pinned G32 preparation");

    let mut output = vec![0.0_f32; outputs];
    fp8_linear_runtime_f32(
        &activation_codes,
        &activation_scales,
        weight_codes,
        weight_scales,
        1,
        WIDTH,
        outputs,
        ActivationGroup::Elements32,
        &mut output,
    )
    .expect("synthetic G32 FP8 projection");
    output.into_iter().map(f32_to_bf16_rne).collect()
}

fn f32_to_bf16_rne(value: f32) -> u16 {
    assert!(value.is_finite(), "test projections must stay finite");
    let bits = value.to_bits();
    let rounded = bits.wrapping_add(0x7fff + ((bits >> 16) & 1));
    u16::try_from(rounded >> 16).expect("a finite FP32 value has a BF16 high half")
}

fn rotate_final_pair_per_head(values: &mut [u16], heads: usize) {
    assert_eq!(values.len(), heads * WIDTH);
    let mut tail = values
        .chunks_exact(WIDTH)
        .flat_map(|head| {
            head[WIDTH - 2..]
                .iter()
                .map(|&bits| f32::from_bits(u32::from(bits) << 16))
        })
        .collect::<Vec<_>>();
    let layout = RotaryTailLayout::new(
        NonZeroUsize::new(1).expect("one batch"),
        NonZeroUsize::new(1).expect("one position"),
        NonZeroUsize::new(heads).expect("at least one head"),
        NonZeroUsize::new(1).expect("one complex pair"),
    )
    .expect("small synthetic tail layout");
    let frequency = [RotaryFrequency::new(0.0, 1.0).expect("finite quarter-turn")];
    rotate_tail(&mut tail, layout, &frequency, RotaryDirection::Forward)
        .expect("small finite rotary tail");
    for (head, rotated) in tail.chunks_exact(2).enumerate() {
        values[head * WIDTH + WIDTH - 2] = f32_to_bf16_rne(rotated[0]);
        values[head * WIDTH + WIDTH - 1] = f32_to_bf16_rne(rotated[1]);
    }
}

fn half_then_one_weights(outputs: usize) -> Vec<u8> {
    assert_eq!(outputs, WIDTH);
    let mut weights = vec![0_u8; outputs * WIDTH];
    for (output, row) in weights.chunks_exact_mut(WIDTH).enumerate() {
        row.fill(if output < WIDTH / 2 { 0x30 } else { 0x38 });
    }
    weights
}

#[test]
fn query_preparation_uses_pinned_g32_before_head_rope() {
    let input = [BF16_ONE; WIDTH];
    let wq_a = half_then_one_weights(WIDTH);
    let pre_norm = fp8_project_bf16(&input, &wq_a, &[127], WIDTH);
    assert!(pre_norm[..WIDTH / 2].iter().all(|&bits| bits == 0x4180));
    assert!(pre_norm[WIDTH / 2..].iter().all(|&bits| bits == 0x4200));

    let mut normalized = [0_u16; WIDTH];
    rms_norm_bf16_reference(&pre_norm, &[BF16_ONE; WIDTH], 1.0e-20, &mut normalized)
        .expect("finite synthetic query norm");
    assert!(
        normalized[..WIDTH / 2]
            .iter()
            .all(|&bits| bits == BF16_POINT_SIX_THREE_TWO_EIGHT_ONE_TWO_FIVE)
    );
    assert!(
        normalized[WIDTH / 2..]
            .iter()
            .all(|&bits| bits == BF16_ONE_POINT_TWO_SIX_FIVE_SIX_TWO_FIVE)
    );

    let mut normalized_requantized = [0_u16; WIDTH];
    requantize_bf16_activations_e4m3fn(
        &normalized,
        1,
        WIDTH,
        ActivationGroup::Elements32,
        &mut normalized_requantized,
    )
    .expect("finite synthetic query requantization");
    assert!(
        normalized_requantized[..WIDTH / 2]
            .iter()
            .all(|&bits| bits == BF16_POINT_SIX_TWO_FIVE)
    );
    assert!(
        normalized_requantized[WIDTH / 2..]
            .iter()
            .all(|&bits| bits == BF16_ONE_POINT_TWO_FIVE)
    );

    let mut wq_b = vec![0_u8; 2 * WIDTH * WIDTH];
    for (output, row) in wq_b.chunks_exact_mut(WIDTH).enumerate() {
        row.fill(if output < WIDTH { 0x30 } else { 0xb8 });
    }
    let mut query = fp8_project_bf16(&normalized, &wq_b, &[127, 127], 2 * WIDTH);
    assert!(query[..WIDTH].iter().all(|&bits| bits == BF16_FIFTEEN));
    assert!(
        query[WIDTH..]
            .iter()
            .all(|&bits| bits == BF16_NEGATIVE_THIRTY)
    );

    rotate_final_pair_per_head(&mut query, 2);
    assert!(query[..WIDTH - 2].iter().all(|&bits| bits == BF16_FIFTEEN));
    assert_eq!(
        &query[WIDTH - 2..WIDTH],
        &[BF16_NEGATIVE_FIFTEEN, BF16_FIFTEEN]
    );
    assert!(
        query[WIDTH..2 * WIDTH - 2]
            .iter()
            .all(|&bits| bits == BF16_NEGATIVE_THIRTY)
    );
    assert_eq!(
        &query[2 * WIDTH - 2..],
        &[BF16_THIRTY, BF16_NEGATIVE_THIRTY]
    );
}

#[test]
fn window_kv_rotates_before_its_pinned_g32_requantization() {
    let input = [BF16_ONE; WIDTH];
    let wkv = half_then_one_weights(WIDTH);
    let pre_norm = fp8_project_bf16(&input, &wkv, &[127], WIDTH);
    let mut normalized = [0_u16; WIDTH];
    rms_norm_bf16_reference(&pre_norm, &[BF16_ONE; WIDTH], 1.0e-20, &mut normalized)
        .expect("finite synthetic KV norm");
    assert!(
        normalized[..WIDTH / 2]
            .iter()
            .all(|&bits| bits == BF16_POINT_SIX_THREE_TWO_EIGHT_ONE_TWO_FIVE)
    );
    assert!(
        normalized[WIDTH / 2..]
            .iter()
            .all(|&bits| bits == BF16_ONE_POINT_TWO_SIX_FIVE_SIX_TWO_FIVE)
    );

    rotate_final_pair_per_head(&mut normalized, 1);
    assert!(
        normalized[..WIDTH - 2]
            .iter()
            .enumerate()
            .all(|(index, &bits)| {
                if index < WIDTH / 2 {
                    bits == BF16_POINT_SIX_THREE_TWO_EIGHT_ONE_TWO_FIVE
                } else {
                    bits == BF16_ONE_POINT_TWO_SIX_FIVE_SIX_TWO_FIVE
                }
            })
    );
    assert_eq!(
        &normalized[WIDTH - 2..],
        &[
            f32_to_bf16_rne(-1.265_625),
            BF16_ONE_POINT_TWO_SIX_FIVE_SIX_TWO_FIVE,
        ]
    );

    let mut prepared_kv = [0_u16; WIDTH];
    requantize_bf16_activations_e4m3fn(
        &normalized,
        1,
        WIDTH,
        ActivationGroup::Elements32,
        &mut prepared_kv,
    )
    .expect("finite synthetic KV requantization");
    assert!(
        prepared_kv[..WIDTH / 2]
            .iter()
            .all(|&bits| bits == BF16_POINT_SIX_TWO_FIVE)
    );
    assert!(
        prepared_kv[WIDTH / 2..WIDTH - 2]
            .iter()
            .all(|&bits| bits == BF16_ONE_POINT_TWO_FIVE)
    );
    assert_eq!(
        &prepared_kv[WIDTH - 2..],
        &[BF16_NEGATIVE_ONE_POINT_TWO_FIVE, BF16_ONE_POINT_TWO_FIVE]
    );
}
