//! One reduced Engram session step at real V4.1 geometry.
//!
//! `decode_1` and `prefill_3` time `EngramSession::step_with` for 1 and 3
//! positions: hashing, a row read from an in-memory source, the FP8 WKV
//! projection (`25600 x 6144` synthetic zero codes) and the residual gate. The
//! projection dominates; its cost grows linearly with positions up to the
//! 128-position step bound.

use std::hint::black_box;

use deepseek::{
    engram::{
        EngramHashLayout,
        embedding::{EngramEmbeddingError, EngramRowSource},
    },
    reduced::{EngramSession, EngramSessionConfig, EngramSessionWeights},
};

const WIDTH: usize = 5120;

/// Every row is small finite codes with unit scales.
struct Rows;

impl EngramRowSource for Rows {
    fn read_rows(
        &self,
        _: &[usize],
        codes: &mut [u8],
        scales: &mut [u8],
    ) -> Result<(), EngramEmbeddingError> {
        codes.fill(0x38);
        scales.fill(127);
        Ok(())
    }
}

fn session() -> EngramSession {
    let layout = EngramHashLayout::new(4, 8, 1, 2, vec![3; 24], vec![0; 24], vec![1; 4])
        .expect("hash layout");
    let config = EngramSessionConfig::new(
        layout,
        vec![0, 1],
        0,
        128,
        4,
        WIDTH,
        384_006_168,
        256,
        1e-20,
        1e-6,
    )
    .expect("V4.1 Engram config");
    let weights = EngramSessionWeights::without_embedding_table(
        vec![0x38; 25_600 * 6_144],
        vec![120; 800 * 192],
        vec![0x3f80; 4 * WIDTH],
        vec![0x3f80; 4 * WIDTH],
    );
    EngramSession::new(config, weights).expect("real-size session")
}

fn bench_step(bencher: divan::Bencher, positions: usize) {
    let ids = vec![1; positions];
    let residual = vec![0x3f80; positions * 4 * WIDTH];
    bencher
        .with_inputs(session)
        .bench_local_values(|mut session| {
            black_box(
                session
                    .step_with(0, &ids, &residual, &Rows)
                    .expect("Engram step"),
            )
        });
}

#[divan::bench(sample_count = 5, sample_size = 1)]
fn decode_1(bencher: divan::Bencher) {
    bench_step(bencher, 1);
}

#[divan::bench(sample_count = 3, sample_size = 1)]
fn prefill_3(bencher: divan::Bencher) {
    bench_step(bencher, 3);
}

fn main() {
    divan::main();
}
