//! One reduced Engram session step at real V4.1 geometry.
//!
//! `decode_1` and `prefill_3` time `EngramSession::step_with` for 1 and 3
//! positions: hashing, a row read from an in-memory source, the FP8 WKV
//! projection (`25600 x 6144` synthetic zero codes) and the residual gate. The
//! projection dominates; its cost grows linearly with positions up to the
//! 128-position step bound.
//!
//! With `--features metal`, `metal_decode_1` and `metal_prefill_3` time the
//! same steps with the WKV projection on the fused FP8 Metal kernel and the
//! weight already resident; `metal_first_call` times a step on fresh operands,
//! including the one-time upload.

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
fn config() -> EngramSessionConfig {
    let layout = EngramHashLayout::new(4, 8, 1, 2, vec![3; 24], vec![0; 24], vec![1; 4])
        .expect("hash layout");
    EngramSessionConfig::new(
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
    .expect("V4.1 Engram config")
}

fn weights() -> EngramSessionWeights {
    EngramSessionWeights::without_embedding_table(
        vec![0x38; 25_600 * 6_144],
        vec![120; 800 * 192],
        vec![0x3f80; 4 * WIDTH],
        vec![0x3f80; 4 * WIDTH],
    )
}

fn session() -> EngramSession {
    EngramSession::new(config(), weights()).expect("real-size session")
}

fn bench_step(bencher: divan::Bencher, positions: usize, session: fn() -> EngramSession) {
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
    bench_step(bencher, 1, session);
}

#[divan::bench(sample_count = 3, sample_size = 1)]
fn prefill_3(bencher: divan::Bencher) {
    bench_step(bencher, 3, session);
}

/// Operands whose WKV weight is already resident on the device, as for every
/// session after a definition's first Metal step.
#[cfg(feature = "metal")]
static RESIDENT: std::sync::LazyLock<EngramSessionWeights> = std::sync::LazyLock::new(|| {
    let weights = weights();
    let mut warm = EngramSession::new(config(), weights.clone())
        .expect("real-size session")
        .with_metal_wkv(true);
    warm.step_with(0, &[1], &vec![0x3f80; 4 * WIDTH], &Rows)
        .expect("warm-up step");
    weights
});

#[cfg(feature = "metal")]
fn metal_session() -> EngramSession {
    EngramSession::new(config(), RESIDENT.clone())
        .expect("real-size session")
        .with_metal_wkv(true)
}

#[cfg(feature = "metal")]
#[divan::bench(sample_count = 5, sample_size = 1)]
fn metal_decode_1(bencher: divan::Bencher) {
    bench_step(bencher, 1, metal_session);
}

#[cfg(feature = "metal")]
#[divan::bench(sample_count = 3, sample_size = 1)]
fn metal_prefill_3(bencher: divan::Bencher) {
    bench_step(bencher, 3, metal_session);
}

/// A decode step on fresh operands: the finiteness scan, the 157 MB upload
/// and validation, then the step. Kernel compilation is cached by MLX after
/// the first sample in a process.
#[cfg(feature = "metal")]
#[divan::bench(sample_count = 3, sample_size = 1)]
fn metal_first_call(bencher: divan::Bencher) {
    let fresh = || (config(), weights());
    let residual = vec![0x3f80; 4 * WIDTH];
    bencher
        .with_inputs(fresh)
        .bench_local_values(|(config, weights)| {
            let mut session = EngramSession::new(config, weights)
                .expect("real-size session")
                .with_metal_wkv(true);
            black_box(
                session
                    .step_with(0, &[1], &residual, &Rows)
                    .expect("Engram step"),
            )
        });
}

fn main() {
    divan::main();
}
