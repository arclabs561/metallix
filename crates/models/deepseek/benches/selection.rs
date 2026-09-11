use std::hint::black_box;

#[divan::bench(args = [512, 4096, 16384])]
fn selects_512_positions(bencher: divan::Bencher, width: usize) {
    // An odd multiplier permutes these power-of-two widths: no cutoff ties.
    // Generate once, outside timing. Validation, allocations and output drop
    // remain timed because callers pay for them on every selection.
    let scores: Vec<f32> = (0..width)
        .map(|position| {
            f32::from(u16::try_from((position * 7919) % width).expect("benchmark width fits u16"))
        })
        .collect();
    bencher.bench_local(|| {
        black_box(
            deepseek::select_indices(black_box(&scores), width, 512, 128)
                .expect("distinct finite scores and bounded offset"),
        );
    });
}

fn main() {
    divan::main();
}
