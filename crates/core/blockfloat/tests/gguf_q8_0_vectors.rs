//! The `Q8_0` decoder and its affine repack against vectors decoded by
//! gguf-py's `dequantize` (written by `scripts/gguf-q8_0-vectors.py`).

use blockfloat::gguf::{GgufEncoding, affine_value, decode, repack_affine};
use serde_json::Value;

#[test]
fn q8_0_matches_gguf_py() {
    let vectors: Value =
        serde_json::from_str(include_str!("data/gguf-q8_0-vectors.json")).expect("vector JSON");
    let hex = vectors["payload_hex"].as_str().expect("payload");
    let payload: Vec<u8> = (0..hex.len())
        .step_by(2)
        .map(|at| u8::from_str_radix(&hex[at..at + 2], 16).expect("hex byte"))
        .collect();
    let expected: Vec<u32> = vectors["values_f32_bits"]
        .as_array()
        .expect("values")
        .iter()
        .map(|bits| u32::try_from(bits.as_u64().expect("bits")).expect("u32"))
        .collect();

    let decoded = decode(GgufEncoding::Q8_0, &payload).expect("decode");
    assert_eq!(decoded.len(), expected.len());
    let decoder_mismatches = decoded
        .iter()
        .zip(&expected)
        .filter(|(value, bits)| value.to_bits() != **bits)
        .count();
    assert_eq!(decoder_mismatches, 0, "decoder vs gguf-py");

    let repack = repack_affine(GgufEncoding::Q8_0, &payload).expect("repack");
    let repack_mismatches = (0..decoded.len())
        .filter(|&index| {
            let value = affine_value(&repack, index);
            let want = f32::from_bits(expected[index]);
            value.to_bits() != want.to_bits() && !(value == 0.0 && want == 0.0)
        })
        .count();
    assert_eq!(repack_mismatches, 0, "affine repack vs gguf-py");
}
