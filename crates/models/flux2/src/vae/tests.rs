//! VAE parity against a `scripts/flux2-reference.py pipeline` reference.
//!
//! ```text
//! METALLIX_FLUX2_MODEL=<FLUX.2-klein-4B dir> \
//! METALLIX_FLUX2_PIPE_REF=<pipe-f32-*.safetensors> \
//!   cargo test -p flux2 --features metal vae -- --ignored --nocapture
//! ```

use std::path::PathBuf;

use super::*;

/// Declared before the first run (plan gate A3).
const F32_DECODE_MIN_PSNR_DB: f64 = 60.0;
/// Unpack, `BatchNorm` and unpatchify are exact rearrangements plus one affine.
const F32_LATENT_RELATIVE_L2: f64 = 1e-6;

struct Reference {
    model: PathBuf,
    tensors: HashMap<String, Array>,
    steps: u64,
    height: i32,
    width: i32,
}

fn reference() -> Option<Reference> {
    let model = PathBuf::from(std::env::var_os("METALLIX_FLUX2_MODEL")?);
    let path = PathBuf::from(std::env::var_os("METALLIX_FLUX2_PIPE_REF")?);
    let meta: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(path.with_extension("json")).expect("sidecar"))
            .expect("JSON");
    let dimension = |key: &str| i32::try_from(meta[key].as_u64().expect(key)).unwrap();
    Some(Reference {
        model,
        tensors: Array::load_safetensors(&path).expect("reference"),
        steps: meta["steps"].as_u64().expect("steps"),
        height: dimension("height"),
        width: dimension("width"),
    })
}

fn values(array: &Array) -> Vec<f64> {
    // Flatten first: as_slice reads the buffer, which a transposed view does not order.
    let wide = array
        .as_dtype(Dtype::Float32)
        .expect("cast")
        .reshape(&[-1])
        .expect("flatten");
    wide.eval().expect("eval");
    wide.as_slice::<f32>()
        .iter()
        .map(|&v| f64::from(v))
        .collect()
}

fn nchw_to_nhwc(array: &Array) -> Array {
    array.transpose_axes(&[0, 2, 3, 1]).expect("transpose")
}

#[test]
#[ignore = "needs a FLUX.2 checkpoint and a diffusers pipeline reference"]
fn latent_post_processing_matches_diffusers() {
    let Some(reference) = reference() else {
        eprintln!("skipping: METALLIX_FLUX2_MODEL and METALLIX_FLUX2_PIPE_REF are required");
        return;
    };
    let vae =
        Flux2VaeDecoder::load(reference.model.join("vae"), Flux2Precision::F32).expect("load");
    let last = &reference.tensors[&format!("latents.step{}", reference.steps - 1)];
    let ours = vae
        .latents_to_decoder_input(last, reference.height / 16, reference.width / 16)
        .expect("post-process");
    let expected = nchw_to_nhwc(&reference.tensors["vae_input"]);
    assert_eq!(ours.shape(), expected.shape());
    let (a, b) = (values(&ours), values(&expected));
    let diff: f64 = a.iter().zip(&b).map(|(x, y)| (x - y).powi(2)).sum();
    let norm: f64 = b.iter().map(|y| y * y).sum();
    let relative = (diff / norm).sqrt();
    eprintln!("latent post-processing relative L2 {relative:.3e}");
    assert!(relative <= F32_LATENT_RELATIVE_L2);
}

#[test]
#[ignore = "needs a FLUX.2 checkpoint and a diffusers pipeline reference"]
fn decode_matches_diffusers() {
    let Some(reference) = reference() else {
        eprintln!("skipping: METALLIX_FLUX2_MODEL and METALLIX_FLUX2_PIPE_REF are required");
        return;
    };
    let vae =
        Flux2VaeDecoder::load(reference.model.join("vae"), Flux2Precision::F32).expect("load");
    let decoded = vae
        .decode(&nchw_to_nhwc(&reference.tensors["vae_input"]))
        .expect("decode");
    // The reference stores the pipeline's float image, (x / 2 + 0.5) clamped.
    let ours = ops::minimum(
        ops::maximum(
            decoded
                .divide(Array::from_f32(2.0))
                .unwrap()
                .add(Array::from_f32(0.5))
                .unwrap(),
            Array::from_f32(0.0),
        )
        .unwrap(),
        Array::from_f32(1.0),
    )
    .unwrap();
    let expected = nchw_to_nhwc(&reference.tensors["image"]);
    assert_eq!(ours.shape(), expected.shape());
    let (a, b) = (values(&ours), values(&expected));
    #[allow(clippy::cast_precision_loss)]
    let mse = a.iter().zip(&b).map(|(x, y)| (x - y).powi(2)).sum::<f64>() / a.len() as f64;
    let psnr = 10.0 * (1.0 / mse).log10();
    eprintln!("f32 decode PSNR {psnr:.2} dB (mse {mse:.3e})");
    assert!(
        psnr >= F32_DECODE_MIN_PSNR_DB,
        "{psnr:.2} dB < {F32_DECODE_MIN_PSNR_DB} dB"
    );
}
