//! Gates G0, G1, G2 and the encoder negative controls (G4 b, c) against a
//! capture from `scripts/qwen3-asr-reference.py capture`.
//!
//! Run with `--features parity-controls -- --ignored --nocapture`, setting
//! `METALLIX_QWEN3_ASR_MODEL` to the checkpoint and `METALLIX_QWEN3_ASR_REF` to
//! the capture directory. The capture's dtype selects the bars; the pass bars
//! were declared before the first run.
#![cfg(feature = "parity-controls")]

use std::path::{Path, PathBuf};

use audio::{LogMel, decode_wav};
use mlx_rs::{Array, Dtype};
use qwen3_asr::encoder::EncoderControl;
use qwen3_asr::{Qwen3Asr, TranscribeOptions};
use serde::Deserialize;

/// G0: normalized log-mel, from the WAV, against the reference's features.
const MEL_MAX_ABS: f32 = 2e-4;
const MEL_MEAN_ABS: f64 = 1e-5;
/// G1 for f32 and bf16 captures: relative L2 error and the smallest
/// per-frame cosine similarity of the encoder output.
const F32_REL_L2: f64 = 1e-4;
const F32_MIN_COSINE: f64 = 0.999_99;
const BF16_REL_L2: f64 = 2e-2;
const BF16_MIN_COSINE: f64 = 0.999;
/// G2 (f32 only): first-step logits.
const LOGIT_MAX_ABS: f32 = 1e-2;

#[derive(Deserialize)]
struct Capture {
    dtype: String,
    language: Option<String>,
    context: String,
    max_new_tokens: usize,
    clips: Vec<Clip>,
}

#[derive(Deserialize)]
struct Clip {
    id: String,
    wav: PathBuf,
    mel_shape: Vec<usize>,
    encoder_shape: Vec<usize>,
    prompt_ids: Vec<i32>,
    generated_ids: Vec<i32>,
    text: String,
}

fn read_f32(path: &Path, count: usize) -> Vec<f32> {
    let bytes = std::fs::read(path).unwrap_or_else(|error| panic!("{}: {error}", path.display()));
    assert_eq!(bytes.len(), count * 4, "{}", path.display());
    bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
}

fn host(array: &Array) -> Vec<f32> {
    let wide = array.as_dtype(Dtype::Float32).expect("f32 copy");
    wide.eval().expect("evaluate");
    wide.as_slice::<f32>().to_vec()
}

/// Relative L2 error and the smallest per-row cosine similarity.
#[allow(clippy::cast_precision_loss)]
fn compare_rows(ours: &[f32], reference: &[f32], width: usize) -> (f64, f64) {
    let (mut error, mut norm) = (0.0_f64, 0.0_f64);
    let mut min_cosine = f64::INFINITY;
    for (a_row, b_row) in ours.chunks_exact(width).zip(reference.chunks_exact(width)) {
        let (mut dot, mut a_norm, mut b_norm) = (0.0_f64, 0.0_f64, 0.0_f64);
        for (&a, &b) in a_row.iter().zip(b_row) {
            let (a, b) = (f64::from(a), f64::from(b));
            error += (a - b) * (a - b);
            norm += b * b;
            dot += a * b;
            a_norm += a * a;
            b_norm += b * b;
        }
        min_cosine = min_cosine.min(dot / (a_norm.sqrt() * b_norm.sqrt()));
    }
    ((error / norm).sqrt(), min_cosine)
}

fn top(logits: &[f32], count: usize) -> Vec<usize> {
    let mut order: Vec<usize> = (0..logits.len()).collect();
    order.sort_by(|&a, &b| logits[b].total_cmp(&logits[a]).then(a.cmp(&b)));
    order.truncate(count);
    order
}

#[test]
#[ignore = "needs METALLIX_QWEN3_ASR_MODEL and a reference capture in METALLIX_QWEN3_ASR_REF"]
#[allow(clippy::too_many_lines, clippy::cast_precision_loss)]
fn matches_the_reference_capture() {
    let model_dir = PathBuf::from(std::env::var_os("METALLIX_QWEN3_ASR_MODEL").expect("model"));
    let reference = PathBuf::from(std::env::var_os("METALLIX_QWEN3_ASR_REF").expect("capture"));
    let capture: Capture = serde_json::from_str(
        &std::fs::read_to_string(reference.join("manifest.json")).expect("manifest"),
    )
    .expect("capture manifest");
    let float32 = match capture.dtype.as_str() {
        "float32" => true,
        "bfloat16" => false,
        other => panic!("capture dtype {other}"),
    };
    let (rel_l2_bar, cosine_bar) = if float32 {
        (F32_REL_L2, F32_MIN_COSINE)
    } else {
        (BF16_REL_L2, BF16_MIN_COSINE)
    };

    let mut model = Qwen3Asr::load(&model_dir).expect("load checkpoint");
    if float32 {
        model.convert(Dtype::Float32).expect("f32 weights");
    }
    let width = model.config().encoder().output_dim;
    let options = TranscribeOptions {
        context: capture.context.clone(),
        language: capture.language.clone(),
        max_new_tokens: capture.max_new_tokens,
        ..TranscribeOptions::default()
    };
    let mut failures = Vec::new();
    for clip in &capture.clips {
        let stem = reference.join(&clip.id);
        let (bins, frames) = (clip.mel_shape[0], clip.mel_shape[1]);

        // G0.
        let audio = decode_wav(&std::fs::read(&clip.wav).expect("wav")).expect("decode wav");
        let ours = model.features(&audio).expect("features");
        let theirs = read_f32(&stem.with_extension("mel.f32"), bins * frames);
        assert_eq!((ours.bins(), ours.frames()), (bins, frames), "{}", clip.id);
        let mut mel_max = 0.0_f32;
        let mut mel_total = 0.0_f64;
        for (&a, &b) in ours.values().iter().zip(&theirs) {
            mel_max = mel_max.max((a - b).abs());
            mel_total += f64::from((a - b).abs());
        }
        let mel_mean = mel_total / theirs.len() as f64;
        println!(
            "{} G0 mel max_abs {mel_max:.3e} mean_abs {mel_mean:.3e}",
            clip.id
        );
        if mel_max > MEL_MAX_ABS || mel_mean > MEL_MEAN_ABS {
            failures.push(format!("{} G0", clip.id));
        }

        // G1, fed the reference features.
        let reference_mel = LogMel::from_values(bins, frames, theirs).expect("reference mel");
        let expected = read_f32(
            &stem.with_extension("encoder.f32"),
            clip.encoder_shape[0] * clip.encoder_shape[1],
        );
        let encoded = model.encoder().encode(&reference_mel).expect("encode");
        assert_eq!(
            encoded.shape(),
            [
                i32::try_from(clip.encoder_shape[0]).unwrap(),
                i32::try_from(clip.encoder_shape[1]).unwrap()
            ],
            "{}",
            clip.id
        );
        let (rel_l2, min_cosine) = compare_rows(&host(&encoded), &expected, width);
        println!(
            "{} G1 encoder rel_l2 {rel_l2:.3e} min_cosine {min_cosine:.7}",
            clip.id
        );
        if rel_l2 > rel_l2_bar || min_cosine < cosine_bar {
            failures.push(format!("{} G1", clip.id));
        }

        // G4 b and c: each deviation must fail G1 where it applies.
        let continuous = model
            .encoder()
            .encode_control(&reference_mel, EncoderControl::ContinuousPositions)
            .expect("continuous positions");
        let (rel_l2, min_cosine) = compare_rows(&host(&continuous), &expected, width);
        let applies = frames > 100;
        println!(
            "{} G4b continuous positions rel_l2 {rel_l2:.3e} min_cosine {min_cosine:.7}",
            clip.id
        );
        if applies && rel_l2 <= rel_l2_bar && min_cosine >= cosine_bar {
            failures.push(format!("{} G4b passed G1", clip.id));
        }
        let full = model
            .encoder()
            .encode_control(&reference_mel, EncoderControl::FullAttention)
            .expect("full attention");
        let (rel_l2, min_cosine) = compare_rows(&host(&full), &expected, width);
        let applies = clip.encoder_shape[0] > model.config().encoder().window_frames();
        println!(
            "{} G4c full attention rel_l2 {rel_l2:.3e} min_cosine {min_cosine:.7}",
            clip.id
        );
        if applies && rel_l2 <= rel_l2_bar && min_cosine >= cosine_bar {
            failures.push(format!("{} G4c passed G1", clip.id));
        }

        // G2: first-step logits and the greedy transcript.
        let logits = model.first_logits(&ours, &options).expect("first logits");
        let reference_logits = read_f32(&stem.with_extension("logits0.f32"), logits.len());
        let logit_max = logits
            .iter()
            .zip(&reference_logits)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        let (ours_top, theirs_top) = (top(&logits, 5), top(&reference_logits, 5));
        let mut ours_set = ours_top.clone();
        let mut theirs_set = theirs_top.clone();
        ours_set.sort_unstable();
        theirs_set.sort_unstable();
        let result = model.transcribe(&audio, &options).expect("transcribe");
        println!(
            "{} G2 logits max_abs {logit_max:.3e} argmax {} vs {} top5 equal {} tokens equal {} ({} vs {})",
            clip.id,
            ours_top[0],
            theirs_top[0],
            ours_set == theirs_set,
            result.generated == clip.generated_ids,
            result.generated.len(),
            clip.generated_ids.len()
        );
        println!(
            "{}   ours:   {}\n{}   theirs: {}",
            clip.id, result.transcript.text, clip.id, clip.text
        );
        assert_eq!(result.prompt_tokens, clip.prompt_ids.len(), "{}", clip.id);
        if float32
            && (ours_top[0] != theirs_top[0]
                || ours_set != theirs_set
                || logit_max > LOGIT_MAX_ABS
                || result.generated != clip.generated_ids)
        {
            failures.push(format!("{} G2", clip.id));
        }
    }
    assert!(failures.is_empty(), "failed: {failures:?}");
}
