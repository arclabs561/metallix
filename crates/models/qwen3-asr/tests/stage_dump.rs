//! Writes the encoder's per-stage outputs in the layout of
//! `scripts/qwen3-asr-native-reference.py layers`, for locating where bf16
//! drifts from the f32 reference. A diagnostic, not a gate.
//!
//! `METALLIX_QWEN3_ASR_MODEL`: checkpoint. `METALLIX_QWEN3_ASR_REF`: an f32
//! `capture` directory whose log-mel features are fed in.
//! `METALLIX_QWEN3_ASR_LAYERS`: a `layers` directory naming the clips.
//! `METALLIX_QWEN3_ASR_STAGES_OUT`: where to write. `METALLIX_QWEN3_ASR_F32=1`
//! runs the encoder in f32 instead of the checkpoint's bf16.
#![cfg(feature = "parity-controls")]

use std::path::PathBuf;

use audio::LogMel;
use mlx_rs::Dtype;
use qwen3_asr::Qwen3Asr;
use serde_json::{Value, json};

fn path(name: &str) -> PathBuf {
    PathBuf::from(std::env::var_os(name).unwrap_or_else(|| panic!("set {name}")))
}

#[test]
#[ignore = "diagnostic: needs a checkpoint and reference captures"]
fn dump_encoder_stages() {
    let mut model = Qwen3Asr::load(&path("METALLIX_QWEN3_ASR_MODEL")).expect("load");
    if std::env::var_os("METALLIX_QWEN3_ASR_F32").is_some() {
        model.convert(Dtype::Float32).expect("f32");
    }
    let reference = path("METALLIX_QWEN3_ASR_REF");
    let capture: Value =
        serde_json::from_str(&std::fs::read_to_string(reference.join("manifest.json")).unwrap())
            .unwrap();
    let layers: Value = serde_json::from_str(
        &std::fs::read_to_string(path("METALLIX_QWEN3_ASR_LAYERS").join("manifest.json")).unwrap(),
    )
    .unwrap();
    let out = path("METALLIX_QWEN3_ASR_STAGES_OUT");
    std::fs::create_dir_all(&out).unwrap();
    let mut records = Vec::new();
    for clip in layers["clips"].as_array().unwrap() {
        let id = clip["id"].as_str().unwrap();
        let captured = capture["clips"]
            .as_array()
            .unwrap()
            .iter()
            .find(|entry| entry["id"] == id)
            .expect("clip in the f32 capture");
        let shape: Vec<usize> = serde_json::from_value(captured["mel_shape"].clone()).unwrap();
        let bytes = std::fs::read(reference.join(id).with_extension("mel.f32")).unwrap();
        let values = bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();
        let mel = LogMel::from_values(shape[0], shape[1], values).unwrap();
        let mut shapes = Vec::new();
        for (index, stage) in model
            .encoder()
            .encode_stages(&mel)
            .unwrap()
            .iter()
            .enumerate()
        {
            let wide = stage.as_dtype(Dtype::Float32).unwrap();
            wide.eval().unwrap();
            let bytes: Vec<u8> = wide
                .as_slice::<f32>()
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect();
            std::fs::write(out.join(format!("{id}.stage{index}.f32")), bytes).unwrap();
            shapes.push(wide.shape().to_vec());
        }
        records.push(json!({"id": id, "stages": shapes}));
    }
    std::fs::write(
        out.join("manifest.json"),
        serde_json::to_string_pretty(&json!({"clips": records})).unwrap(),
    )
    .unwrap();
}
