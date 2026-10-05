//! Final head at real V4.1 size: one position, BF16 weights `[129280, 5120]`.
//!
//! `bf16_last_position` is `FinalHead::forward` (HC collapse, `RMSNorm` and the
//! scalar FP32 projection over exactly widened BF16 weights) for the one
//! position a `HeadPositions::Last` step computes. The 1.3 GB weights are
//! synthetic finite BF16 values; the arithmetic does not depend on them.

use std::hint::black_box;

use deepseek::reduced::{FinalHead, HeadWeights};

const VOCABULARY: usize = 129_280;
const WIDTH: usize = 5120;
const COPIES: usize = 4;

#[divan::bench(sample_count = 5, sample_size = 1)]
fn bf16_last_position(bencher: divan::Bencher) {
    let weights: Vec<u16> = (0..VOCABULARY * WIDTH)
        .map(|index| 0x3c00 | u16::try_from(index % 0x3ff).expect("small"))
        .collect();
    let norm = vec![0x3f80; WIDTH];
    let head = FinalHead::with_weights(
        &norm,
        HeadWeights::Bf16(&weights),
        VOCABULARY,
        COPIES,
        1e-20,
    )
    .expect("real-size BF16 head");
    let residual: Vec<u16> = (0..COPIES * WIDTH)
        .map(|index| 0x3f00 | u16::try_from(index % 0x7f).expect("small"))
        .collect();
    let pre = [0.4, 0.3, 0.2, 0.1];
    bencher
        .counter(divan::counter::ItemsCount::new(VOCABULARY * WIDTH))
        .bench_local(|| black_box(head.forward(black_box(&residual), &pre).expect("logits")));
}

fn main() {
    divan::main();
}
