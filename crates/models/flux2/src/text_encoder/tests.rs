//! Text-encoder parity (plan gate A1) and its control (gate C1) against
//! `scripts/flux2-reference.py text-encoder` outputs.
//!
//! ```text
//! METALLIX_FLUX2_MODEL=<FLUX.2-klein-4B dir> \
//! METALLIX_FLUX2_TEXT_REF=<te-f32-pN.safetensors> \
//!   cargo test -p flux2 --features metal text_encoder -- --ignored --nocapture
//! ```
//!
//! bf16 adds `METALLIX_FLUX2_PRECISION=bf16` and
//! `METALLIX_FLUX2_TEXT_REF_BF16=<te-bf16-pN.safetensors>` (the source's bf16
//! run); the bar is then twice the source's own bf16 distance per layer.
//! A missing or empty input fails the gate; it is never skipped.

use std::collections::HashMap;
use std::path::PathBuf;

use mlx_rs::Dtype;

use super::*;
use crate::gate_inputs::{self, InputError, Lookup, process_env, require};

/// Declared before the first run (plan gate A1): every element within
/// 1e-3 of its row's RMS, real and padding rows alike.
const F32_MAX_ABS_OVER_ROW_RMS: f64 = 1e-3;

fn load(path: &Path) -> HashMap<String, Array> {
    Array::load_safetensors(path).unwrap_or_else(|error| panic!("{}: {error}", path.display()))
}

fn rows(array: &Array, width: usize) -> Vec<Vec<f64>> {
    let flat = array
        .as_dtype(Dtype::Float32)
        .unwrap()
        .reshape(&[-1])
        .unwrap();
    flat.eval().unwrap();
    flat.as_slice::<f32>()
        .chunks(width)
        .map(|row| row.iter().map(|&v| f64::from(v)).collect())
        .collect()
}

/// Worst `max |a - b| / rms(b)` over rows, and the relative L2 of the whole.
fn distances(ours: &[Vec<f64>], reference: &[Vec<f64>]) -> (f64, f64) {
    let (worst, l2, _) = distances_with_worst(ours, reference);
    (worst, l2)
}

/// [`distances`] plus where the worst element sits, for diagnosis:
/// `(row, channel, ours, reference, row rms)`.
fn distances_with_worst(
    ours: &[Vec<f64>],
    reference: &[Vec<f64>],
) -> (f64, f64, (usize, usize, f64, f64, f64)) {
    let mut worst = 0.0_f64;
    let mut at = (0, 0, 0.0, 0.0, 0.0);
    let (mut diff, mut norm) = (0.0, 0.0);
    for (row, (a, b)) in ours.iter().zip(reference).enumerate() {
        #[allow(clippy::cast_precision_loss)]
        let rms = (b.iter().map(|v| v * v).sum::<f64>() / b.len() as f64).sqrt();
        for (channel, (x, y)) in a.iter().zip(b).enumerate() {
            let score = (x - y).abs() / rms;
            if score > worst {
                worst = score;
                at = (row, channel, *x, *y, rms);
            }
        }
        diff += a.iter().zip(b).map(|(x, y)| (x - y).powi(2)).sum::<f64>();
        norm += b.iter().map(|v| v * v).sum::<f64>();
    }
    (worst, (diff / norm).sqrt(), at)
}

struct Reference {
    model: PathBuf,
    tensors: HashMap<String, Array>,
    real_ids: Vec<i32>,
}

/// Resolves the checkpoint and reference paths and the real token count.
fn paths(lookup: Lookup<'_>) -> Result<(PathBuf, PathBuf, usize), InputError> {
    let model = gate_inputs::dir(lookup, "METALLIX_FLUX2_MODEL")?;
    let path = gate_inputs::file(lookup, "METALLIX_FLUX2_TEXT_REF")?;
    let real = gate_inputs::field(&gate_inputs::sidecar(&path)?, "real_tokens")?;
    let real = usize::try_from(real).map_err(|_| InputError::Field("real_tokens".into()))?;
    Ok((model, path, real))
}

fn reference(lookup: Lookup<'_>) -> Reference {
    let (model, path, real) = require(paths(lookup));
    let tensors = load(&path);
    let ids = tensors["input_ids"].reshape(&[-1]).unwrap();
    ids.eval().unwrap();
    let real_ids = ids.as_slice::<i32>()[..real].to_vec();
    Reference {
        model,
        tensors,
        real_ids,
    }
}

#[test]
#[ignore = "needs a FLUX.2 checkpoint and diffusers references; see the module docs"]
fn layer_states_match_diffusers() {
    let precision = require(gate_inputs::precision(&process_env));
    // Resolved up front so a missing bf16 reference fails before the run.
    let source_bf16_path = (precision == Flux2Precision::Bf16).then(|| {
        require(gate_inputs::cpu_reference(
            &process_env,
            "METALLIX_FLUX2_TEXT_REF_BF16",
        ))
    });
    let reference = reference(&process_env);
    let (padded, real_len) = KleinTextEncoder::pad(&reference.real_ids).unwrap();
    let expected_ids = reference.tensors["input_ids"].reshape(&[-1]).unwrap();
    expected_ids.eval().unwrap();
    assert_eq!(
        padded.as_slice(),
        expected_ids.as_slice::<i32>(),
        "padded ids"
    );

    let encoder = KleinTextEncoder::load(reference.model.join("text_encoder"), precision).unwrap();
    let states = encoder.layer_states(&reference.real_ids).unwrap();
    let hidden = usize::try_from(states[0].shape()[2]).unwrap();
    let source_bf16 = source_bf16_path.as_deref().map(load);
    // Every layer is measured and reported before any bar is enforced.
    let mut failures = Vec::new();
    for (state, layer) in states.iter().zip(TEXT_LAYERS) {
        let key = format!("hidden_states.{layer}");
        let ours = rows(state, hidden);
        let expected = rows(&reference.tensors[&key], hidden);
        let (real_worst, real_l2) = distances(&ours[..real_len], &expected[..real_len]);
        let (pad_worst, pad_l2, (row, channel, x, y, rms)) =
            distances_with_worst(&ours[real_len..], &expected[real_len..]);
        eprintln!(
            "{precision:?} layer {layer}: real max/rms {real_worst:.2e} rel L2 {real_l2:.2e}; \
             pad max/rms {pad_worst:.2e} rel L2 {pad_l2:.2e} \
             (worst at pad row {row}, channel {channel}: ours {x:.6e}, source {y:.6e}, row rms {rms:.3e})"
        );
        match &source_bf16 {
            None => {
                if real_worst > F32_MAX_ABS_OVER_ROW_RMS {
                    failures.push(format!("layer {layer} real rows"));
                }
                if pad_worst > F32_MAX_ABS_OVER_ROW_RMS {
                    failures.push(format!("layer {layer} padding rows"));
                }
            }
            Some(source) => {
                let source = rows(&source[&key], hidden);
                let (_, source_real) = distances(&source[..real_len], &expected[..real_len]);
                let (_, source_pad) = distances(&source[real_len..], &expected[real_len..]);
                eprintln!(
                    "  source bf16: real rel L2 {source_real:.2e}, pad rel L2 {source_pad:.2e}"
                );
                if real_l2 > 2.0 * source_real {
                    failures.push(format!("layer {layer} real rows"));
                }
                if pad_l2 > 2.0 * source_pad {
                    failures.push(format!("layer {layer} padding rows"));
                }
            }
        }
    }
    assert!(
        failures.is_empty(),
        "{precision:?} bar missed: {failures:?}"
    );
}

/// Gate C1: layer 8 compared with the reference's layer 9 must fail the f32
/// bar, so the bar can tell neighbouring layers apart.
#[test]
#[ignore = "needs a FLUX.2 checkpoint and diffusers references; see the module docs"]
fn a_neighbouring_layer_fails_the_text_gate() {
    let reference = reference(&process_env);
    let encoder =
        KleinTextEncoder::load(reference.model.join("text_encoder"), Flux2Precision::F32).unwrap();
    let (padded, real_len) = KleinTextEncoder::pad(&reference.real_ids).unwrap();
    let state = encoder
        .weights
        .forward_layer_states(&padded, real_len, &[8])
        .unwrap()
        .remove(0);
    let hidden = usize::try_from(state.shape()[2]).unwrap();
    let (worst, l2) = distances(
        &rows(&state, hidden),
        &rows(&reference.tensors["hidden_states.9"], hidden),
    );
    eprintln!("layer 8 vs reference layer 9: max/rms {worst:.2e}, rel L2 {l2:.2e}");
    assert!(worst > F32_MAX_ABS_OVER_ROW_RMS);
}

#[test]
fn the_rendered_prompt_matches_the_reference_template() {
    // From te-f32-p1.json ("rendered"), transformers 5.18 apply_chat_template.
    assert_eq!(
        klein_prompt_text("A red fox sitting in fresh snow at dawn, photograph"),
        "<|im_start|>user\nA red fox sitting in fresh snow at dawn, photograph<|im_end|>\n\
         <|im_start|>assistant\n<think>\n\n</think>\n\n"
    );
}

/// The gates above fail, not skip, without their inputs.
#[test]
fn missing_text_inputs_fail_the_gate() {
    let unset: Lookup<'_> = &|_| None;
    assert!(matches!(paths(unset), Err(InputError::Unset(_))));
}
