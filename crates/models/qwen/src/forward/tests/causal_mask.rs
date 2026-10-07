//! The fused `Causal` mask against the explicit chunk mask it replaced, at
//! cached lengths off the kernel's key-block multiples, where MLX 0.25's
//! fused kernel under-masked. Bit-identical outputs, or the swap is wrong.

use mlx_rs::{Array, Dtype, StreamOrDevice, fast};

use super::super::chunk_causal_mask;
use crate::GPU_TEST_LOCK;

/// Deterministic bf16 values in [-1, 1) from a counter hash, so a failure
/// reproduces without a random generator.
fn filled(shape: &[i32], salt: usize) -> Array {
    let count: usize = shape
        .iter()
        .map(|&d| usize::try_from(d).expect("dim"))
        .product();
    let values = (0..count)
        .map(|index| {
            let mixed = (index.wrapping_mul(2_654_435_761) ^ salt.wrapping_mul(40_503)) % 2_001;
            f32::from(u16::try_from(mixed).expect("bounded")) / 1_000.0 - 1.0
        })
        .collect::<Vec<_>>();
    Array::from_slice(&values, shape)
        .as_dtype_device(Dtype::Bfloat16, StreamOrDevice::gpu())
        .expect("bf16")
}

#[test]
fn fused_causal_mask_matches_the_explicit_chunk_mask() {
    let _gpu = GPU_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let stream = StreamOrDevice::gpu();
    // Qwen3-0.6B attention: 16 query heads over 8 K/V heads of 128.
    for (cached, chunk) in [
        (1, 2),
        (37, 5),
        (37, 64),
        (127, 3),
        (129, 200),
        (300, 33),
        (1_000, 128),
        (2_047, 513),
    ] {
        let total = cached + chunk;
        let query = filled(&[1, 16, chunk, 128], 1);
        let keys = filled(&[1, 8, total, 128], 2);
        let values = filled(&[1, 8, total, 128], 3);
        let scale = 1.0 / 128_f32.sqrt();
        let fused = fast::scaled_dot_product_attention_device(
            &query,
            &keys,
            &values,
            scale,
            Some(fast::ScaledDotProductAttentionMask::Causal),
            Option::<&Array>::None,
            &stream,
        )
        .expect("fused causal");
        let mask = chunk_causal_mask(cached, chunk, &stream).expect("explicit mask");
        let explicit = fast::scaled_dot_product_attention_device(
            &query,
            &keys,
            &values,
            scale,
            Some(fast::ScaledDotProductAttentionMask::Array(&mask)),
            Option::<&Array>::None,
            &stream,
        )
        .expect("explicit mask");
        let fused = fused.as_dtype_device(Dtype::Float32, &stream).expect("f32");
        let explicit = explicit
            .as_dtype_device(Dtype::Float32, &stream)
            .expect("f32");
        // SDPA may return a strided array; host slices require contiguous
        // storage even after dtype conversion. Preserve logical element order.
        let fused = fused.contiguous().expect("contiguous fused readback");
        let explicit = explicit.contiguous().expect("contiguous explicit readback");
        fused.eval().expect("fused evaluates");
        explicit.eval().expect("explicit evaluates");
        let (fused, explicit) = (fused.as_slice::<f32>(), explicit.as_slice::<f32>());
        let differing = fused
            .iter()
            .zip(explicit)
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        assert_eq!(
            differing,
            0,
            "cached {cached}, chunk {chunk}: {differing} of {} outputs differ",
            fused.len()
        );
    }
}
