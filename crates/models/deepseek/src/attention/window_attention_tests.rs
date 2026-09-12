//! Raw-window scheduling composed with the FP32 semantic attention reference.
//! These cases do not claim BF16 attention-kernel or full-model parity.

use std::num::NonZeroUsize;

use super::window::{WindowStep, window_topk_indices, write_window_kv_bf16};
use super::{SparseAttentionLayout, sparse_attention_reference};

fn nz(value: usize) -> NonZeroUsize {
    NonZeroUsize::new(value).expect("nonzero test geometry")
}

fn attend(values: &[u16], step: WindowStep, queries: usize, slots: usize) -> Vec<f32> {
    let kv = values
        .iter()
        .map(|&bits| f32::from_bits(u32::from(bits) << 16))
        .collect::<Vec<_>>();
    let indices = window_topk_indices(step, nz(4), nz(1)).expect("bounded window indices");
    let layout = SparseAttentionLayout::new(
        nz(1),
        nz(queries),
        nz(1),
        nz(1),
        nz(values.len()),
        nz(slots),
    )
    .expect("bounded attention layout");
    sparse_attention_reference(&vec![0.0; queries], &kv, &[0.0], &indices, 1.0, layout)
        .expect("finite window attention")
}

#[test]
fn prefill_attends_chunk_then_decode_attends_physical_ring() {
    // Exact BF16 encodings of 0, 1, 2, 3, 4, 5.
    let chunk = [0x0000, 0x3f80, 0x4000, 0x4040, 0x4080, 0x40a0];
    let prefill = WindowStep::Prefill { tokens: nz(6) };
    let mut ring = [0x42c6; 4]; // Stale finite 99s must not contribute.
    write_window_kv_bf16(prefill, &chunk, nz(1), nz(4), nz(1), &mut ring)
        .expect("seed wrapped ring");
    assert_eq!(ring, [0x4080, 0x40a0, 0x4000, 0x4040]);
    // Zero scores give each live KV and the denominator-only sink equal mass.
    let output = attend(&chunk, prefill, 6, 4);
    for (actual, expected) in output
        .into_iter()
        .zip([0.0, 1.0 / 3.0, 0.75, 1.2, 2.0, 2.8])
    {
        assert!((actual - expected).abs() < 1e-6);
    }

    let decode = WindowStep::Decode { position: nz(6) };
    write_window_kv_bf16(decode, &[0x40c0], nz(1), nz(4), nz(1), &mut ring)
        .expect("append BF16 six");
    assert_eq!(ring, [0x4080, 0x40a0, 0x40c0, 0x4040]);
    assert!((attend(&ring, decode, 1, 4)[0] - 3.6).abs() < 1e-6);
}

#[test]
fn partial_decode_masks_stale_physical_slot() {
    let mut ring = [0x42c6; 4];
    write_window_kv_bf16(
        WindowStep::Prefill { tokens: nz(2) },
        &[0, 0x3f80],
        nz(1),
        nz(4),
        nz(1),
        &mut ring,
    )
    .expect("short prefill");
    let decode = WindowStep::Decode { position: nz(2) };
    write_window_kv_bf16(decode, &[0x4000], nz(1), nz(4), nz(1), &mut ring)
        .expect("append BF16 two");
    assert_eq!(ring, [0, 0x3f80, 0x4000, 0x42c6]);
    assert!((attend(&ring, decode, 1, 4)[0] - 0.75).abs() < 1e-6);
}
