//! Regression test for the MLX 0.25 short causal batch attention bug.
//!
//! With MLX 0.25 (mlx-sys 0.2.0), causal scaled-dot-product attention over a
//! batch of two or more sequences returned wrong rows when the query length
//! was 2 through 8 and the head width was 64 or 128, so `src/forward.rs`
//! padded batches to 9 positions. MLX 0.32 (mlx-sys 0.6.0) computes those rows
//! exactly; this test keeps it that way so the padding can stay deleted.
#![cfg(feature = "metal")]
#![allow(
    deprecated,
    reason = "mlx-rs 0.32 deprecates the *_device ops; the with_stream migration is a separate change"
)]

use mlx_rs::{Array, StreamOrDevice, fast, random};

/// Largest absolute difference between each batch row of causal attention and
/// the same row computed alone, on deterministic random inputs.
fn batched_minus_alone(batch: i32, length: i32, heads: i32, kv_heads: i32, width: i32) -> f32 {
    let stream = StreamOrDevice::gpu();
    let normal = |shape: &[i32], seed: u64| {
        random::normal::<f32>(shape, None, None, &random::key(seed).unwrap()).unwrap()
    };
    let query = normal(&[batch, heads, length, width], 7);
    let key = normal(&[batch, kv_heads, length, width], 8);
    let value = normal(&[batch, kv_heads, length, width], 9);
    #[allow(clippy::cast_precision_loss, reason = "small head widths")]
    let scale = (width as f32).powf(-0.5);
    let causal = || Some(fast::ScaledDotProductAttentionMask::Causal);
    let batched = fast::scaled_dot_product_attention_device(
        &query,
        &key,
        &value,
        scale,
        causal(),
        Option::<&Array>::None,
        &stream,
    )
    .unwrap();
    (0..batch)
        .map(|row| {
            let pick = |array: &Array| {
                array
                    .take_axis_device(Array::from_slice(&[row], &[1]), 0, &stream)
                    .unwrap()
            };
            let alone = fast::scaled_dot_product_attention_device(
                pick(&query),
                pick(&key),
                pick(&value),
                scale,
                causal(),
                Option::<&Array>::None,
                &stream,
            )
            .unwrap();
            let difference = pick(&batched)
                .subtract(&alone)
                .unwrap()
                .abs()
                .unwrap()
                .max(None)
                .unwrap();
            difference.eval().unwrap();
            difference.item_exact::<f32>()
        })
        .fold(0.0, f32::max)
}

#[test]
fn mlx_short_causal_batches_match_single_sequences() {
    // Qwen3-Embedding-0.6B attention: 16 query heads, 8 key/value heads, width
    // 128. Lengths 2 through 8 are the ones MLX 0.25 got wrong.
    for (batch, length, heads, kv_heads, width) in [
        (2, 2, 16, 8, 128),
        (2, 5, 16, 8, 128),
        (2, 8, 16, 8, 128),
        (3, 4, 16, 8, 64),
        (1, 8, 16, 8, 128),
        (2, 9, 16, 8, 128),
        (2, 1, 16, 8, 128),
        (2, 8, 2, 1, 4),
    ] {
        let difference = batched_minus_alone(batch, length, heads, kv_heads, width);
        assert!(
            difference == 0.0,
            "batch {batch} length {length} width {width}: {difference:e}"
        );
    }
}
