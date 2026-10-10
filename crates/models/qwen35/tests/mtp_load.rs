//! Opt-in: loading a checkpoint's multi-token-prediction head.
//!
//! Set `METALLIX_QWEN35_MODEL` to a dense `qwen3_5` checkpoint directory
//! (Qwen3.5-0.8B or 9B). Loading with [`MtpHead::Load`] must add exactly the
//! checkpoint's `mtp.*` tensors at their stored size (checkpoint precision
//! keeps BF16, and the shifted norms return to BF16), and the default
//! [`LoadParts`] must match [`Qwen35Weights::load`].

#![cfg(feature = "metal")]

use std::{env, fs, io::Read, path::Path, path::PathBuf};

use qwen35::forward::{LoadParts, MtpHead, Qwen35Precision, Qwen35Weights};
use serde_json::Value;

/// Bytes of every `mtp.*` tensor, read from the safetensors headers.
fn stored_mtp_bytes(model: &Path) -> usize {
    let mut total = 0;
    for entry in fs::read_dir(model).expect("model directory") {
        let path = entry.expect("entry").path();
        if path
            .extension()
            .is_none_or(|extension| extension != "safetensors")
        {
            continue;
        }
        let mut file = fs::File::open(&path).expect("shard");
        let mut length = [0_u8; 8];
        file.read_exact(&mut length).expect("header length");
        let mut header = vec![0_u8; usize::try_from(u64::from_le_bytes(length)).expect("usize")];
        file.read_exact(&mut header).expect("header");
        let header: Value = serde_json::from_slice(&header).expect("header JSON");
        for (name, tensor) in header.as_object().expect("header object") {
            if name.starts_with("mtp.") {
                let offsets = &tensor["data_offsets"];
                let span = offsets[1].as_u64().expect("end") - offsets[0].as_u64().expect("start");
                total += usize::try_from(span).expect("usize");
            }
        }
    }
    total
}

#[test]
fn loading_the_head_adds_exactly_its_stored_tensors() {
    let Some(model) = env::var_os("METALLIX_QWEN35_MODEL").map(PathBuf::from) else {
        eprintln!("skipping: METALLIX_QWEN35_MODEL is not set");
        return;
    };
    let expected = stored_mtp_bytes(&model);
    assert!(expected > 0, "checkpoint has no mtp.* tensors");
    // One load alive at a time: a 9B load holds ~18 GB of weights.
    let base = Qwen35Weights::load(&model, Qwen35Precision::Checkpoint)
        .expect("load")
        .logical_weight_bytes();
    let skipped =
        Qwen35Weights::load_with(&model, Qwen35Precision::Checkpoint, &LoadParts::default())
            .expect("load without the head")
            .logical_weight_bytes();
    assert_eq!(skipped, base);
    let mut parts = LoadParts::default();
    parts.mtp_head = MtpHead::Load;
    let with_head = Qwen35Weights::load_with(&model, Qwen35Precision::Checkpoint, &parts)
        .expect("load with the head");
    eprintln!(
        "decoder {base} bytes, head {} bytes (stored {expected})",
        with_head.logical_weight_bytes() - base
    );
    assert_eq!(with_head.logical_weight_bytes() - base, expected);
}
