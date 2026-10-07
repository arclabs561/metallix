//! Embedded prompts against the ids path.
//!
//! The oracle is the ids path itself: replacing positions with their own token
//! embeddings must change nothing, bit for bit, and the K/V a decode step then
//! reads must be the same too. Replacing them with other rows must change the
//! logits, so the first test cannot pass by ignoring the spans.

use media::{EmbeddedPrompt, EmbeddedSpan, SpanKind};

use super::prefix_extend::{sensitive_weights, two_layer_long_config};
use super::*;
use crate::forward::Qwen3ResidentChatPlan;

const PROMPT: [i32; 9] = [1, 5, 2, 7, 7, 7, 3, 6, 0];

fn own_embeddings(weights: &HashMap<String, Array>, ids: &[i32]) -> Array {
    let rows = weights["model.embed_tokens.weight"]
        .take_axis(
            Array::from_slice(ids, &[i32::try_from(ids.len()).unwrap()]),
            0,
        )
        .expect("token rows");
    rows.eval().expect("rows materialize");
    rows
}

fn audio(start: usize, rows: Array) -> EmbeddedSpan<Array> {
    let len = usize::try_from(rows.shape()[0]).unwrap();
    EmbeddedSpan::new(start, len, SpanKind::Audio, [0; 32], rows)
}

fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|value| value.to_bits()).collect()
}

fn plan(config: &Qwen3ForwardConfig) -> Qwen3ResidentChatPlan {
    config
        .resident_chat_plan(64, u64::MAX, crate::forward::Qwen3FloatPrecision::Float32)
        .expect("tiny resident plan")
}

#[test]
fn spans_of_their_own_token_embeddings_match_the_ids_path_bit_for_bit() {
    let _gpu = GPU_TEST_LOCK.lock().expect("GPU test lock");
    let config = two_layer_long_config();
    let weights = sensitive_weights();
    let plan = plan(&config);

    let mut by_ids = Qwen3ForwardExecutor::new_for_resident_chat(&config, &weights, plan);
    let expected = by_ids.prefill_last_logits(&PROMPT).expect("ids prefill");
    let expected_next = by_ids.decode_last_logits(4).expect("ids decode");

    let cases: [Vec<EmbeddedSpan<Array>>; 3] = [
        Vec::new(),
        vec![audio(3, own_embeddings(&weights, &PROMPT[3..6]))],
        vec![
            audio(0, own_embeddings(&weights, &PROMPT[0..1])),
            audio(6, own_embeddings(&weights, &PROMPT[6..9])),
        ],
    ];
    for spans in cases {
        let span_count = spans.len();
        let prompt = EmbeddedPrompt::new(&PROMPT, spans).expect("valid spans");
        let mut embedded = Qwen3ForwardExecutor::new_for_resident_chat(&config, &weights, plan);
        let logits = embedded
            .prefill_embedded_last_logits(&prompt)
            .expect("embedded prefill");
        assert_eq!(bits(&logits), bits(&expected), "{span_count} spans");
        assert_eq!(embedded.cached_tokens(), PROMPT.len());
        let next = embedded.decode_last_logits(4).expect("embedded decode");
        assert_eq!(bits(&next), bits(&expected_next), "{span_count} spans");
    }
}

#[test]
fn chunks_straddling_a_span_match_one_prefill() {
    let _gpu = GPU_TEST_LOCK.lock().expect("GPU test lock");
    let config = two_layer_long_config();
    let weights = sensitive_weights();
    let plan = plan(&config);
    // Rows of another token, so a misplaced row would change the logits.
    let prompt = EmbeddedPrompt::new(
        &PROMPT,
        vec![audio(2, own_embeddings(&weights, &[4, 4, 4, 4, 4]))],
    )
    .expect("valid span");
    let mut whole = Qwen3ForwardExecutor::new_for_resident_chat(&config, &weights, plan);
    let expected = whole
        .prefill_embedded_last_logits(&prompt)
        .expect("one prefill");
    for cut in [1, 3, 5, 7] {
        let mut chunked = Qwen3ForwardExecutor::new_for_resident_chat(&config, &weights, plan);
        chunked
            .extend_embedded_last_logits(&prompt.chunk(0..cut))
            .expect("first chunk");
        let logits = chunked
            .extend_embedded_last_logits(&prompt.chunk(cut..PROMPT.len()))
            .expect("second chunk");
        assert_logits_match(expected.clone(), logits);
    }
    // A chunk that does not start where the cache ends is refused.
    let mut skipped = Qwen3ForwardExecutor::new_for_resident_chat(&config, &weights, plan);
    assert!(matches!(
        skipped.extend_embedded_last_logits(&prompt.chunk(2..5)),
        Err(Qwen3ForwardError::EmbeddedSpan { position: 2, .. })
    ));
}

#[test]
fn other_rows_change_the_logits_and_cast_to_the_embedding_dtype() {
    let _gpu = GPU_TEST_LOCK.lock().expect("GPU test lock");
    let config = two_layer_long_config();
    let weights = sensitive_weights();
    let plan = plan(&config);
    let mut by_ids = Qwen3ForwardExecutor::new_for_resident_chat(&config, &weights, plan);
    let expected = by_ids.prefill_last_logits(&PROMPT).expect("ids prefill");

    // Rows of a different token, supplied in bf16: they must reach the
    // decoder (logits change) and must not promote it (the K/V stays f32).
    let foreign = own_embeddings(&weights, &[2, 2, 2])
        .as_dtype(mlx_rs::Dtype::Bfloat16)
        .expect("bf16 rows");
    let prompt = EmbeddedPrompt::new(&PROMPT, vec![audio(3, foreign)]).expect("valid span");
    let mut embedded = Qwen3ForwardExecutor::new_for_resident_chat(&config, &weights, plan);
    let logits = embedded
        .prefill_embedded_last_logits(&prompt)
        .expect("embedded prefill");
    assert_ne!(bits(&logits), bits(&expected));
    let cached = embedded.cache[0].as_ref().expect("layer 0 K/V");
    assert_eq!(cached.keys.dtype(), mlx_rs::Dtype::Float32);
}

#[test]
fn refuses_image_spans_and_misshapen_rows() {
    let _gpu = GPU_TEST_LOCK.lock().expect("GPU test lock");
    let config = two_layer_long_config();
    let weights = sensitive_weights();
    let plan = plan(&config);
    let image = EmbeddedSpan::new(
        0,
        4,
        SpanKind::Image {
            grid_h: 2,
            grid_w: 2,
        },
        [0; 32],
        own_embeddings(&weights, &PROMPT[0..4]),
    );
    let narrow = audio(0, Array::zeros::<f32>(&[2, 3]).expect("narrow rows"));
    for span in [image, narrow] {
        let prompt = EmbeddedPrompt::new(&PROMPT, vec![span]).expect("valid placement");
        let mut executor = Qwen3ForwardExecutor::new_for_resident_chat(&config, &weights, plan);
        assert!(matches!(
            executor.prefill_embedded_last_logits(&prompt),
            Err(Qwen3ForwardError::EmbeddedSpan { position: 0, .. })
        ));
        assert_eq!(executor.cached_tokens(), 0);
    }
}
