//! Image-space gate (plan gate B, with the source's prompt embedding) and
//! its negative control (gate C2).
//!
//! ```text
//! METALLIX_FLUX2_MODEL=<FLUX.2-klein-4B dir> \
//! METALLIX_FLUX2_PIPE_REF=<pipe-f32-pN.safetensors> \
//! METALLIX_FLUX2_TEXT_REF=<te-f32-pN.safetensors> \
//!   cargo test -p flux2 --features metal pipeline -- --ignored --nocapture
//! ```
//!
//! bf16 adds `METALLIX_FLUX2_PRECISION=bf16`, the bf16 text reference, and
//! `METALLIX_FLUX2_PIPE_REF_BF16=<pipe-bf16-pN.safetensors>`, the source's own
//! bf16 image whose distance from f32 sets the bar. The control takes
//! `METALLIX_FLUX2_OTHER_TEXT_REF`, another prompt's embedding.
//! `METALLIX_FLUX2_OUT=<file.ppm>` saves the generated image. A missing or
//! empty input fails the gate; it is never skipped.

use std::collections::HashMap;
use std::path::PathBuf;

use diffusion::metrics::{psnr_rgb8, ssim_rgb8};

use super::*;
use crate::gate_inputs::{self, InputError, Lookup, process_env, require};

/// Declared before the first run (plan gate B).
const F32_MIN_PSNR_DB: f64 = 40.0;
const F32_MIN_SSIM: f64 = 0.99;
const BF16_PSNR_MARGIN_DB: f64 = 3.0;
const BF16_MIN_SSIM: f64 = 0.95;

fn load(path: &Path) -> HashMap<String, Array> {
    Array::load_safetensors(path).unwrap_or_else(|error| panic!("{}: {error}", path.display()))
}

/// The reference's float image `[1, 3, H, W]` in `[0, 1]`, quantized the way
/// diffusers' `numpy_to_pil` does.
fn reference_rgb8(tensors: &HashMap<String, Array>) -> Vec<u8> {
    let image = tensors["image"]
        .as_dtype(Dtype::Float32)
        .unwrap()
        .transpose_axes(&[0, 2, 3, 1])
        .unwrap()
        .reshape(&[-1])
        .unwrap();
    image.eval().unwrap();
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    image
        .as_slice::<f32>()
        .iter()
        .map(|&v| (v * 255.0).round_ties_even() as u8)
        .collect()
}

struct Setup {
    pipeline: Flux2Pipeline,
    reference: HashMap<String, Array>,
    size: ImageSize,
    steps: usize,
}

/// Where a pipeline gate's checkpoint and reference live, resolved before
/// anything is loaded.
struct Paths {
    model: PathBuf,
    reference: PathBuf,
    width: u32,
    height: u32,
    steps: usize,
}

fn paths(lookup: Lookup<'_>) -> Result<Paths, InputError> {
    let model = gate_inputs::dir(lookup, "METALLIX_FLUX2_MODEL")?;
    let reference = gate_inputs::file(lookup, "METALLIX_FLUX2_PIPE_REF")?;
    let meta = gate_inputs::sidecar(&reference)?;
    let side = |key: &str| -> Result<u32, InputError> {
        u32::try_from(gate_inputs::field(&meta, key)?).map_err(|_| InputError::Field(key.into()))
    };
    Ok(Paths {
        width: side("width")?,
        height: side("height")?,
        steps: usize::try_from(gate_inputs::field(&meta, "steps")?)
            .map_err(|_| InputError::Field("steps".into()))?,
        model,
        reference,
    })
}

fn setup(paths: &Paths, precision: Flux2Precision) -> Setup {
    Setup {
        pipeline: Flux2Pipeline::load(&paths.model, precision).expect("load"),
        reference: load(&paths.reference),
        size: ImageSize::new(paths.width, paths.height).expect("size"),
        steps: paths.steps,
    }
}

fn generate(setup: &Setup, text_reference: &Path) -> Vec<u8> {
    let text = load(text_reference)
        .remove("prompt_embeds")
        .expect("prompt_embeds");
    let initial = setup
        .pipeline
        .initial_latents(
            InitialNoise::Latents(setup.reference["initial_latents"].clone()),
            setup.size,
        )
        .expect("latents");
    let packed = setup
        .pipeline
        .denoise(
            &text,
            &initial,
            setup.size,
            setup.steps,
            |index, latents| {
                if let Some(expected) = setup.reference.get(&format!("latents.step{index}")) {
                    let ours = latents.as_dtype(Dtype::Float32).unwrap();
                    let diff = ours
                        .subtract(expected)
                        .unwrap()
                        .square()
                        .unwrap()
                        .sum(None)
                        .unwrap();
                    let norm = expected.square().unwrap().sum(None).unwrap();
                    let relative = (diff.item_cast::<f32>() / norm.item_cast::<f32>()).sqrt();
                    eprintln!("step {index}: latent relative L2 {relative:.3e}");
                }
            },
        )
        .expect("denoise");
    let gib = |bytes: usize| {
        #[allow(clippy::cast_precision_loss)]
        let gib = bytes as f64 / f64::from(1u32 << 30);
        gib
    };
    let report = |phase: &str| {
        eprintln!(
            "{phase}: MLX active {:.2} GiB, cache {:.2} GiB, peak {:.2} GiB",
            gib(mlx_rs::memory::active_memory().unwrap()),
            gib(mlx_rs::memory::cache_memory().unwrap()),
            gib(mlx_rs::memory::peak_memory().unwrap()),
        );
    };
    report("after denoise");
    mlx_rs::memory::reset_peak_memory().unwrap();
    let pixels = setup
        .pipeline
        .decode_rgb8(&packed, setup.size)
        .expect("decode");
    report("after decode");
    if let Some(out) = process_env("METALLIX_FLUX2_OUT").map(PathBuf::from) {
        let mut ppm =
            format!("P6\n{} {}\n255\n", setup.size.width(), setup.size.height()).into_bytes();
        ppm.extend(&pixels);
        fs::write(out, ppm).expect("write image");
    }
    pixels
}

#[test]
#[ignore = "needs a FLUX.2 checkpoint and diffusers references; see the module docs"]
fn generated_image_matches_diffusers() {
    let precision = require(gate_inputs::precision(&process_env));
    let paths = require(paths(&process_env));
    let text = require(gate_inputs::file(&process_env, "METALLIX_FLUX2_TEXT_REF"));
    // Resolved up front so a missing bf16 reference fails before the run.
    let source_bf16 = (precision == Flux2Precision::Bf16).then(|| {
        require(gate_inputs::cpu_reference(
            &process_env,
            "METALLIX_FLUX2_PIPE_REF_BF16",
        ))
    });
    if let Some(source) = source_bf16.as_deref() {
        require(gate_inputs::matching_pipeline(&paths.reference, source));
    }
    let setup = setup(&paths, precision);
    let ours = generate(&setup, &text);
    let reference = reference_rgb8(&setup.reference);
    let (width, height) = (setup.size.width() as usize, setup.size.height() as usize);
    let psnr = psnr_rgb8(&ours, &reference);
    let ssim = ssim_rgb8(&ours, &reference, width, height);
    eprintln!("{precision:?} vs source f32: PSNR {psnr:.2} dB, SSIM {ssim:.4}");
    match precision {
        Flux2Precision::F32 => {
            assert!(
                psnr >= F32_MIN_PSNR_DB,
                "PSNR {psnr:.2} < {F32_MIN_PSNR_DB}"
            );
            assert!(ssim >= F32_MIN_SSIM, "SSIM {ssim:.4} < {F32_MIN_SSIM}");
        }
        Flux2Precision::Bf16 => {
            let source = reference_rgb8(&load(source_bf16.as_deref().expect("resolved above")));
            let source_psnr = psnr_rgb8(&source, &reference);
            eprintln!("source bf16 vs source f32: PSNR {source_psnr:.2} dB");
            assert!(
                psnr >= source_psnr - BF16_PSNR_MARGIN_DB,
                "PSNR {psnr:.2} < {source_psnr:.2} - {BF16_PSNR_MARGIN_DB}"
            );
            assert!(ssim >= BF16_MIN_SSIM, "SSIM {ssim:.4} < {BF16_MIN_SSIM}");
        }
    }
}

/// Gate C2: the same noise under another prompt must fail the f32 bar, or the
/// bar does not discriminate.
#[test]
#[ignore = "needs a FLUX.2 checkpoint and diffusers references; see the module docs"]
fn another_prompt_fails_the_image_gate() {
    let paths = require(paths(&process_env));
    let other = require(gate_inputs::file(
        &process_env,
        "METALLIX_FLUX2_OTHER_TEXT_REF",
    ));
    let setup = setup(&paths, Flux2Precision::F32);
    let ours = generate(&setup, &other);
    let reference = reference_rgb8(&setup.reference);
    let (width, height) = (setup.size.width() as usize, setup.size.height() as usize);
    let psnr = psnr_rgb8(&ours, &reference);
    let ssim = ssim_rgb8(&ours, &reference, width, height);
    eprintln!("other prompt vs reference: PSNR {psnr:.2} dB, SSIM {ssim:.4}");
    assert!(psnr < F32_MIN_PSNR_DB && ssim < F32_MIN_SSIM);
}

/// The gates above fail, not skip, without their inputs: an unset
/// environment resolves to an error. This runs without a checkpoint.
#[test]
fn missing_pipeline_inputs_fail_the_gate() {
    let unset: Lookup<'_> = &|_| None;
    assert!(matches!(paths(unset), Err(InputError::Unset(_))));
}
