//! A long prefill or extension built as consecutive pieces matches the same
//! tokens built as one graph. Pieces that end before, on and after a K/V
//! storage tier (128, 512) exercise a cache that grows between pieces. Uses
//! the two-layer, mask-sensitive weights of `prefix_extend`, where an
//! unmasked or misaligned piece moves logits well past the tolerance.

use super::{
    super::{PREFILL_CHUNK_OVERRIDE, PREFILL_CHUNK_TOKENS},
    prefix_extend::{sensitive_weights, two_layer_long_config},
    *,
};

/// Runs `body` with this thread's prefill piece size set to `tokens`.
fn with_piece<T>(tokens: usize, body: impl FnOnce() -> T) -> T {
    PREFILL_CHUNK_OVERRIDE.with(|piece| piece.set(Some(tokens)));
    let result = body();
    PREFILL_CHUNK_OVERRIDE.with(|piece| piece.set(None));
    result
}

#[test]
fn chunked_prefill_matches_a_single_graph() {
    let _gpu = GPU_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let config = two_layer_long_config();
    let weights = sensitive_weights();
    let plan = config
        .resident_chat_plan(1_000, u64::MAX, F32)
        .expect("tiny resident plan");
    let prompt: Vec<i32> = (0..900).map(|index| (index * 5 + index / 7) % 8).collect();
    let whole = with_piece(usize::MAX, || {
        let mut executor = Qwen3ForwardExecutor::new_for_resident_chat(&config, &weights, plan);
        let logits = executor.prefill_last_logits(&prompt).expect("one graph");
        (
            logits,
            executor
                .decode_last_logits(3)
                .expect("decode after one graph"),
        )
    });
    for piece in [1, 7, 64, 127, 128, 129, 511, 899] {
        let (logits, next) = with_piece(piece, || {
            let mut executor = Qwen3ForwardExecutor::new_for_resident_chat(&config, &weights, plan);
            let logits = executor.prefill_last_logits(&prompt).expect("pieces");
            assert_eq!(executor.cached_tokens(), prompt.len(), "piece {piece}");
            (
                logits,
                executor.decode_last_logits(3).expect("decode after pieces"),
            )
        });
        assert_logits_match(whole.0.clone(), logits);
        assert_logits_match(whole.1.clone(), next);
    }
    // An extension after a prefix is chunked the same way.
    let extended = with_piece(100, || {
        let mut executor = Qwen3ForwardExecutor::new_for_resident_chat(&config, &weights, plan);
        executor.prefill_last_logits(&prompt[..50]).expect("prefix");
        executor
            .extend_last_logits(&prompt[50..])
            .expect("chunked extension")
    });
    assert_logits_match(whole.0.clone(), extended);
    const { assert!(PREFILL_CHUNK_TOKENS >= 512, "production pieces stay large") };
}
