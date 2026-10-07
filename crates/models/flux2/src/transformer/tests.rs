//! Component parity against diffusers references from
//! `scripts/flux2-reference.py transformer`.
//!
//! ```text
//! METALLIX_FLUX2_MODEL=<FLUX.2-klein-4B dir> \
//! METALLIX_FLUX2_DIT_REF=<dit-f32-*.safetensors> \
//! METALLIX_FLUX2_TEXT_REF=<te-f32-*.safetensors> \
//!   cargo test -p flux2 --features metal -- --ignored --nocapture
//! ```
//!
//! With `METALLIX_FLUX2_PRECISION=bf16`, point the text reference at the
//! bf16 text-encoder output and set `METALLIX_FLUX2_DIT_REF_BF16` to the
//! source's bf16 transformer reference on those same inputs. The bf16 bar is
//! then relative: at most twice the source's own bf16 distance from f32.

use std::path::PathBuf;

use super::*;
use crate::rope::{image_position_ids, text_position_ids};

/// Declared before the first run (plan gate A2): f32 relative L2 per traced
/// output against the f32 source.
const F32_RELATIVE_L2: f64 = 1e-4;

fn env_path(name: &str) -> Option<PathBuf> {
    std::env::var_os(name).map(PathBuf::from)
}

fn load(path: &Path) -> HashMap<String, Array> {
    Array::load_safetensors(path).unwrap_or_else(|error| panic!("{}: {error}", path.display()))
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

fn relative_l2(ours: &Array, reference: &Array) -> f64 {
    assert_eq!(ours.shape(), reference.shape());
    let (a, b) = (values(ours), values(reference));
    let diff: f64 = a.iter().zip(&b).map(|(x, y)| (x - y).powi(2)).sum();
    let norm: f64 = b.iter().map(|y| y * y).sum();
    (diff / norm).sqrt()
}

struct Case {
    model: PathBuf,
    reference: HashMap<String, Array>,
    text: Array,
    timestep: f32,
    rope: RopeTables,
}

fn case(reference_var: &str) -> Option<Case> {
    let model = env_path("METALLIX_FLUX2_MODEL")?;
    let reference_path = env_path(reference_var)?;
    let text_path = env_path("METALLIX_FLUX2_TEXT_REF")?;
    let meta: serde_json::Value = serde_json::from_str(
        &fs::read_to_string(reference_path.with_extension("json")).expect("reference sidecar"),
    )
    .expect("sidecar JSON");
    #[allow(clippy::cast_possible_truncation)]
    let timestep = meta["scheduler_timestep"].as_f64().expect("timestep") as f32;
    let height = usize::try_from(meta["height"].as_u64().expect("height")).unwrap() / 16;
    let width = usize::try_from(meta["width"].as_u64().expect("width")).unwrap() / 16;
    let text = load(&text_path)
        .remove("prompt_embeds")
        .expect("prompt_embeds");
    let text_len = usize::try_from(text.shape()[1]).unwrap();
    let mut ids = text_position_ids(text_len);
    ids.extend(image_position_ids(height, width));
    Some(Case {
        model,
        reference: load(&reference_path),
        text,
        timestep,
        rope: RopeTables::new(&ids, [32, 32, 32, 32], 2000.0).unwrap(),
    })
}

fn run(case: &Case, precision: Flux2Precision) -> Flux2Trace {
    let model = Flux2Transformer::load(case.model.join("transformer"), precision).expect("load");
    let trace = model
        .forward_traced(
            &case.reference["packed_latents"],
            &case.text,
            case.timestep,
            &case.rope,
        )
        .expect("forward");
    mlx_rs::transforms::eval([
        &trace.double0_text,
        &trace.double0_image,
        &trace.single0,
        &trace.velocity,
    ])
    .expect("eval");
    trace
}

fn named(trace: &Flux2Trace) -> [(&'static str, &Array); 4] {
    [
        ("double_block_0.0", &trace.double0_text),
        ("double_block_0.1", &trace.double0_image),
        ("single_block_0", &trace.single0),
        ("velocity", &trace.velocity),
    ]
}

#[test]
#[ignore = "needs a FLUX.2 checkpoint and diffusers references; see the module docs"]
fn transformer_matches_diffusers_reference() {
    let precision = match std::env::var("METALLIX_FLUX2_PRECISION").as_deref() {
        Ok("bf16") => Flux2Precision::Bf16,
        _ => Flux2Precision::F32,
    };
    let Some(case) = case("METALLIX_FLUX2_DIT_REF") else {
        eprintln!("skipping: METALLIX_FLUX2_MODEL, _DIT_REF and _TEXT_REF are required");
        return;
    };
    let trace = run(&case, precision);

    match precision {
        Flux2Precision::F32 => {
            for (name, ours) in named(&trace) {
                let distance = relative_l2(ours, &case.reference[name]);
                eprintln!("f32 {name}: relative L2 {distance:.3e}");
                assert!(
                    distance <= F32_RELATIVE_L2,
                    "{name}: {distance:.3e} > {F32_RELATIVE_L2:e}"
                );
            }
        }
        Flux2Precision::Bf16 => {
            let source = load(
                &env_path("METALLIX_FLUX2_DIT_REF_BF16").expect("METALLIX_FLUX2_DIT_REF_BF16"),
            );
            for (name, ours) in named(&trace) {
                let ours_distance = relative_l2(ours, &case.reference[name]);
                let source_distance = relative_l2(&source[name], &case.reference[name]);
                eprintln!(
                    "bf16 {name}: ours {ours_distance:.3e}, source bf16 {source_distance:.3e}"
                );
                assert!(
                    ours_distance <= 2.0 * source_distance,
                    "{name}: {ours_distance:.3e} > 2 x {source_distance:.3e}"
                );
            }
        }
    }
}
