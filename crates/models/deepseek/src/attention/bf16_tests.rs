//! Independent closed-form precision probes, not a device-kernel oracle.

use super::bf16::{SparseAttentionBf16Error, sparse_attention_bf16_reference};
use super::{SparseAttentionLayout, sparse_attention_reference};
use std::num::NonZeroUsize;

fn layout(dimensions: usize, keys: usize, slots: usize) -> SparseAttentionLayout {
    let nz = |value| NonZeroUsize::new(value).unwrap();
    SparseAttentionLayout::new(nz(1), nz(1), nz(1), nz(dimensions), nz(keys), nz(slots)).unwrap()
}

fn bf16(value: f32) -> u16 {
    let bits = value.to_bits();
    let [_, _, lo, hi] = bits.wrapping_add(0x7fff + ((bits >> 16) & 1)).to_le_bytes();
    u16::from_le_bytes([lo, hi])
}

#[test]
fn empty_blocks_before_and_after_a_live_key_do_not_change_its_result() {
    for slots in [1, 63, 64, 65, 127, 128, 129] {
        for live_slot in [0, slots / 2, slots - 1] {
            let mut indices = vec![-1; slots];
            indices[live_slot] = 0;
            let output = sparse_attention_bf16_reference(
                &[0x3f80],
                &[0x3f80],
                &[0.0],
                &indices,
                1.0,
                layout(1, 1, slots),
            )
            .unwrap();
            // A single score-one/value-one key with sink zero: 1/(1+exp(-1)).
            assert_eq!(output, [0x3f3b], "slots={slots}, live_slot={live_slot}");
        }
        assert_eq!(
            sparse_attention_bf16_reference(
                &[0x3f80],
                &[0x3f80],
                &[0.0],
                &vec![-1; slots],
                1.0,
                layout(1, 1, slots),
            )
            .unwrap(),
            [0],
            "all-masked slots={slots}"
        );
    }
}

#[test]
fn numerator_cast_and_fp32_denominator_have_distinct_rounding_boundaries() {
    // q=1, KV=[0,k], sink=0: BF16(k*BF16(exp(k))/(2+exp(k))).
    // k=-1/16 catches a missing probability cast (0xbca4 instead of 0xbca3).
    // k=-11/8 catches casting denominator probabilities (0xbe1e, not 0xbe1d).
    for (key, expected) in [(0xbd80, 0xbca3), (0xbfb0, 0xbe1d)] {
        let output = sparse_attention_bf16_reference(
            &[0x3f80],
            &[0, key],
            &[0.0],
            &[0, 1],
            1.0,
            layout(1, 2, 2),
        )
        .unwrap();
        assert_eq!(output, [expected]);
    }
    let semantic = sparse_attention_reference(
        &[1.0],
        &[0.0, -0.0625],
        &[0.0],
        &[0, 1],
        1.0,
        layout(1, 2, 2),
    )
    .unwrap();
    assert_eq!(bf16(semantic[0]), 0xbca4);
}

#[test]
fn block_boundary_rescales_previous_numerator_without_bf16_recasting() {
    // q=[1,0], KV=[[0,1],[1,0]]. Score difference is exactly one.
    let query = [0x3f80, 0];
    let kv = [0, 0x3f80, 0x3f80, 0];
    let together =
        sparse_attention_bf16_reference(&query, &kv, &[-100.0], &[0, 1], 1.0, layout(2, 2, 2))
            .unwrap();
    // Numerator second component is BF16(e^-1)=0.3671875.
    assert_eq!(together, [0x3f3b, 0x3e89]);
    let mut split = [-1; 65];
    split[0] = 0;
    split[64] = 1;
    let across =
        sparse_attention_bf16_reference(&query, &kv, &[-100.0], &split, 1.0, layout(2, 2, 65))
            .unwrap();
    // Rescaling old FP32 accumulation uses e^-1 without a BF16 cast.
    assert_eq!(across, [0x3f3b, 0x3e8a]);
    assert_ne!(together, across);
}

#[test]
fn sink_is_added_after_key_blocks_without_changing_the_running_maximum() {
    let output =
        sparse_attention_bf16_reference(&[0, 0], &[0, 0x3f80], &[1.0], &[0], 1.0, layout(2, 1, 1))
            .unwrap();
    // 1/(1+e); including sink in max would quantize e^-1 in the numerator.
    assert_eq!(output, [0, 0x3e8a]);
}

#[test]
fn rejects_expensive_geometry_and_finite_input_arithmetic_overflow() {
    assert!(matches!(
        sparse_attention_bf16_reference(&[], &[], &[], &[], 1.0, layout(1024, 1, 8193)),
        Err(SparseAttentionBf16Error::WorkloadTooLarge { .. })
    ));
    assert!(matches!(
        sparse_attention_bf16_reference(&[0x7f7f], &[0x7f7f], &[0.0], &[0], 1.0, layout(1, 1, 1),),
        Err(SparseAttentionBf16Error::NonFiniteArithmetic {
            stage: "dot product",
            ..
        })
    ));
    assert!(matches!(
        sparse_attention_bf16_reference(&[0], &[0x3f80], &[1000.0], &[0], 1.0, layout(1, 1, 1),),
        Err(SparseAttentionBf16Error::NonFiniteArithmetic {
            stage: "sink exponential",
            ..
        })
    ));
}

#[test]
fn batch_query_and_head_offsets_preserve_distinct_values_and_sinks() {
    let nz = |value| NonZeroUsize::new(value).unwrap();
    let layout = SparseAttentionLayout::new(nz(2), nz(2), nz(2), nz(1), nz(2), nz(1)).unwrap();
    // Each query selects one key. Head zero halves it through sink=0;
    // head one has negligible sink mass. Batch one reverses key order.
    let output = sparse_attention_bf16_reference(
        &[0; 8],
        &[0x4000, 0x4080, 0x40c0, 0x4100],
        &[0.0, -100.0],
        &[0, 1, 1, 0],
        1.0,
        layout,
    )
    .unwrap();
    assert_eq!(
        output,
        [
            0x3f80, 0x4000, 0x4000, 0x4080, 0x4080, 0x4100, 0x4040, 0x40c0
        ]
    );
}
