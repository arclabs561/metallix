//! Log-mel parity against transformers' `WhisperFeatureExtractor` as
//! Qwen3-ASR configures it. The fixture comes from
//! `scripts/qwen3-asr-reference.py mel-fixture`; the signals are regenerated
//! here with the script's formulas.

use audio::{LogMelExtractor, LogMelSpec, MonoAudio};
use serde::Deserialize;

/// Largest accepted difference of any normalized log-mel value.
const MAX_ABS: f32 = 2e-4;
/// Largest accepted mean absolute difference over the sampled values.
const MEAN_ABS: f64 = 1e-5;

#[derive(Deserialize)]
struct Fixture {
    signals: Vec<Signal>,
}

#[derive(Deserialize)]
struct Signal {
    name: String,
    samples: usize,
    sample_sum: f64,
    bins: usize,
    frames: usize,
    stride: usize,
    values: Vec<f32>,
    max: f32,
    mean: f64,
}

const SAMPLE_RATE: f64 = 16_000.0;

#[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
fn chirp(count: usize) -> Vec<f32> {
    (0..count)
        .map(|index| {
            let t = index as f64 / SAMPLE_RATE;
            (0.5 * (std::f64::consts::TAU * (100.0 * t + 1500.0 * t * t)).sin()) as f32
        })
        .collect()
}

#[allow(clippy::cast_possible_truncation)]
fn lcg_noise(count: usize, seed: u32) -> Vec<f32> {
    let mut state = seed;
    (0..count)
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            ((f64::from(state >> 8) / 16_777_216.0 * 2.0 - 1.0) * 0.1) as f32
        })
        .collect()
}

#[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
fn quiet_tone(count: usize, silent: usize) -> Vec<f32> {
    (0..count)
        .map(|index| {
            if index < silent {
                0.0
            } else {
                let t = index as f64 / SAMPLE_RATE;
                (1e-3 * (std::f64::consts::TAU * 440.0 * t).sin()) as f32
            }
        })
        .collect()
}

fn signal(name: &str) -> Vec<f32> {
    match name {
        "chirp" => chirp(37_920),
        "noise" => lcg_noise(12_345, 0x2545_F491),
        "quiet_tone" => quiet_tone(8_000, 3_000),
        "minimum_length" => lcg_noise(201, 7),
        other => panic!("fixture names an unknown signal {other:?}"),
    }
}

#[test]
fn log_mel_matches_the_transformers_reference() {
    let fixture: Fixture =
        serde_json::from_str(include_str!("fixtures/qwen3_asr_mel.json")).unwrap();
    assert_eq!(fixture.signals.len(), 4);
    let extractor = LogMelExtractor::new(LogMelSpec::QWEN3_ASR);
    for reference in fixture.signals {
        let samples = signal(&reference.name);
        assert_eq!(samples.len(), reference.samples, "{}", reference.name);
        let sum: f64 = samples.iter().map(|&sample| f64::from(sample)).sum();
        assert!(
            (sum - reference.sample_sum).abs() < 1e-6,
            "{}: the regenerated signal differs from the script's ({sum} vs {})",
            reference.name,
            reference.sample_sum
        );

        let mel = extractor
            .extract(&MonoAudio::new(16_000, samples).unwrap())
            .unwrap();
        assert_eq!(
            (mel.bins(), mel.frames()),
            (reference.bins, reference.frames),
            "{}",
            reference.name
        );
        let ours: Vec<f32> = mel
            .values()
            .iter()
            .copied()
            .step_by(reference.stride)
            .collect();
        assert_eq!(ours.len(), reference.values.len(), "{}", reference.name);
        let mut worst = 0.0_f32;
        let mut total = 0.0_f64;
        for (&a, &b) in ours.iter().zip(&reference.values) {
            let difference = (a - b).abs();
            worst = worst.max(difference);
            total += f64::from(difference);
        }
        #[allow(clippy::cast_precision_loss)]
        let mean_difference = total / ours.len() as f64;
        assert!(
            worst <= MAX_ABS && mean_difference <= MEAN_ABS,
            "{}: max abs {worst}, mean abs {mean_difference}",
            reference.name
        );

        let max = mel
            .values()
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);
        #[allow(clippy::cast_precision_loss)]
        let mean = mel
            .values()
            .iter()
            .map(|&value| f64::from(value))
            .sum::<f64>()
            / mel.values().len() as f64;
        assert!((max - reference.max).abs() <= MAX_ABS, "{}", reference.name);
        assert!(
            (mean - reference.mean).abs() <= MEAN_ABS,
            "{}",
            reference.name
        );
    }
}
