//! Final head at real V4.1 size: one position, BF16 weights `[129280, 5120]`.
//!
//! `bf16_last_position` is `FinalHead::forward` (HC collapse, `RMSNorm` and the
//! scalar FP32 projection over exactly widened BF16 weights) for the one
//! position a `HeadPositions::Last` step computes. The 1.3 GB weights are
//! synthetic finite BF16 values; the arithmetic does not depend on them.
//!
//! With `--features metal`, `bf16_last_position_mlx` is
//! `FinalHead::forward_metal` over a `MetalBf16Head` uploaded once before
//! timing (after one untimed warm-up step), and `bf16_head_upload_mlx` times
//! that one-off upload alone.

use std::hint::black_box;

use deepseek::reduced::{FinalHead, HeadWeights};

const VOCABULARY: usize = 129_280;
const WIDTH: usize = 5120;
const COPIES: usize = 4;

#[divan::bench(sample_count = 5, sample_size = 1)]
fn bf16_last_position(bencher: divan::Bencher) {
    let (weights, norm, residual) = bf16_head_fixture();
    let head = FinalHead::with_weights(
        &norm,
        HeadWeights::Bf16(&weights),
        VOCABULARY,
        COPIES,
        1e-20,
    )
    .expect("real-size BF16 head");
    let pre = [0.4, 0.3, 0.2, 0.1];
    bencher
        .counter(divan::counter::ItemsCount::new(VOCABULARY * WIDTH))
        .bench_local(|| black_box(head.forward(black_box(&residual), &pre).expect("logits")));
}

fn bf16_head_fixture() -> (Vec<u16>, Vec<u16>, Vec<u16>) {
    let weights: Vec<u16> = (0..VOCABULARY * WIDTH)
        .map(|index| 0x3c00 | u16::try_from(index % 0x3ff).expect("small"))
        .collect();
    let norm = vec![0x3f80; WIDTH];
    let residual: Vec<u16> = (0..COPIES * WIDTH)
        .map(|index| 0x3f00 | u16::try_from(index % 0x7f).expect("small"))
        .collect();
    (weights, norm, residual)
}

#[cfg(feature = "metal")]
#[divan::bench(sample_count = 20, sample_size = 1)]
fn bf16_last_position_mlx(bencher: divan::Bencher) {
    let (weights, norm, residual) = bf16_head_fixture();
    let head = FinalHead::with_weights(
        &norm,
        HeadWeights::Bf16(&weights),
        VOCABULARY,
        COPIES,
        1e-20,
    )
    .expect("real-size BF16 head");
    let resident =
        deepseek::reduced::MetalBf16Head::new(&weights, VOCABULARY, WIDTH).expect("upload");
    let pre = [0.4, 0.3, 0.2, 0.1];
    head.forward_metal(&residual, &pre, &resident)
        .expect("warm-up logits");
    bencher
        .counter(divan::counter::ItemsCount::new(VOCABULARY * WIDTH))
        .bench_local(|| {
            black_box(
                head.forward_metal(black_box(&residual), &pre, &resident)
                    .expect("logits"),
            )
        });
}

#[cfg(feature = "metal")]
#[divan::bench(sample_count = 5, sample_size = 1)]
fn bf16_head_upload_mlx(bencher: divan::Bencher) {
    let (weights, _, _) = bf16_head_fixture();
    bencher.bench_local(|| {
        black_box(
            deepseek::reduced::MetalBf16Head::new(black_box(&weights), VOCABULARY, WIDTH)
                .expect("upload"),
        )
    });
}

fn main() {
    divan::main();
}
