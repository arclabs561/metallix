//! Native layer-four attention against source-captured attention boundaries.

#[path = "support/attention_capture.rs"]
mod attention_capture;

use attention_capture::{
    SOURCE_LAYER, assert_diagnostic, call_frequencies, fixture, forward_case,
    forward_with_publication, frequencies, layout, native_outputs_from_inputs, nonzero,
    prefetched_state, weights,
};
use deepseek::attention::layer::{LayerAttentionError, LayerAttentionLayout, LayerAttentionState};

#[test]
fn native_attention_matches_captured_prefill_and_decode_boundaries() {
    let fixture = fixture();
    let frequencies = frequencies(&fixture);
    let weights = weights(&fixture.encoded_parameters);
    let mut state = LayerAttentionState::new(layout(&fixture.model));
    for (call_id, case) in fixture.cases.iter().enumerate() {
        let diagnostic = forward_case(
            &mut state,
            case,
            0,
            u64::try_from(call_id).expect("bounded calls"),
            SOURCE_LAYER,
            &frequencies,
            weights.borrowed(),
        )
        .expect("source-shaped attention execution");
        assert_diagnostic(case, &diagnostic);
        assert_eq!(
            case.window_indices.dtype, "torch.int32",
            "source window index storage"
        );
        assert_eq!(
            case.wo_b_input.dtype, "torch.bfloat16",
            "source WO-B input storage"
        );
    }
}

#[test]
fn stale_publications_wrong_source_and_discontinuous_position_are_atomic() {
    let fixture = fixture();
    let frequencies = frequencies(&fixture);
    let weights = weights(&fixture.encoded_parameters);
    let mut state = LayerAttentionState::new(layout(&fixture.model));
    forward_case(
        &mut state,
        &fixture.cases[0],
        0,
        0,
        SOURCE_LAYER,
        &frequencies,
        weights.borrowed(),
    )
    .expect("first source prefill");
    assert!(matches!(
        forward_case(
            &mut state,
            &fixture.cases[1],
            0,
            1,
            SOURCE_LAYER + 1,
            &frequencies,
            weights.borrowed()
        ),
        Err(LayerAttentionError::WrongSourceLayer { .. })
    ));
    assert!(matches!(
        forward_case(
            &mut state,
            &fixture.cases[1],
            1,
            1,
            SOURCE_LAYER,
            &frequencies,
            weights.borrowed()
        ),
        Err(LayerAttentionError::WrongEpoch { .. })
    ));
    assert!(matches!(
        forward_case(
            &mut state,
            &fixture.cases[2],
            0,
            2,
            SOURCE_LAYER,
            &frequencies,
            weights.borrowed()
        ),
        Err(LayerAttentionError::DiscontinuousPosition { .. })
    ));
    let diagnostic = forward_case(
        &mut state,
        &fixture.cases[1],
        0,
        1,
        SOURCE_LAYER,
        &frequencies,
        weights.borrowed(),
    )
    .expect("failed calls do not advance state");
    assert_diagnostic(&fixture.cases[1], &diagnostic);
    state.reset().expect("bounded epoch increment");
    assert!(matches!(
        forward_case(
            &mut state,
            &fixture.cases[0],
            0,
            0,
            SOURCE_LAYER,
            &frequencies,
            weights.borrowed()
        ),
        Err(LayerAttentionError::WrongEpoch { .. })
    ));
    let diagnostic = forward_case(
        &mut state,
        &fixture.cases[0],
        1,
        0,
        SOURCE_LAYER,
        &frequencies,
        weights.borrowed(),
    )
    .expect("stale publication failure does not consume reset prefill");
    assert_diagnostic(&fixture.cases[0], &diagnostic);
}

#[test]
fn wrong_call_and_missing_weight_fail_without_consuming_the_prefill() {
    let fixture = fixture();
    let frequencies = frequencies(&fixture);
    let weights = weights(&fixture.encoded_parameters);
    let mut state = LayerAttentionState::new(layout(&fixture.model));
    assert!(matches!(
        forward_case(
            &mut state,
            &fixture.cases[0],
            0,
            1,
            SOURCE_LAYER,
            &frequencies,
            weights.borrowed()
        ),
        Err(LayerAttentionError::WrongCallId { .. })
    ));
    let mut missing = weights.borrowed();
    missing.wq_a.codes = &[];
    assert!(matches!(
        forward_case(
            &mut state,
            &fixture.cases[0],
            0,
            0,
            SOURCE_LAYER,
            &frequencies,
            missing
        ),
        Err(LayerAttentionError::Fp8Linear(_))
    ));
    let diagnostic = forward_case(
        &mut state,
        &fixture.cases[0],
        0,
        0,
        SOURCE_LAYER,
        &frequencies,
        weights.borrowed(),
    )
    .expect("failed prefill remains atomic");
    assert_diagnostic(&fixture.cases[0], &diagnostic);
}

#[test]
fn compressed_prefix_length_is_exact_and_failures_are_atomic() {
    let fixture = fixture();
    let frequencies = frequencies(&fixture);
    let weights = weights(&fixture.encoded_parameters);
    let case = &fixture.cases[1];
    let input = case.input.bf16();
    let compressed = case.compressed_kv.bf16();
    let legal_indices = vec![i32::try_from(fixture.model.window_size).expect("small window")];
    let short = &compressed[..5 * fixture.model.head_dim];
    let mut after_prefill = prefetched_state(&fixture, &frequencies, weights.borrowed());
    assert!(matches!(
        forward_with_publication(
            &mut after_prefill,
            &input,
            case.start_pos,
            0,
            1,
            SOURCE_LAYER,
            short,
            &legal_indices,
            call_frequencies(&frequencies, case),
            weights.borrowed()
        ),
        Err(LayerAttentionError::CompressedKeyCount {
            actual: 5,
            expected: 6
        })
    ));
    let diagnostic = forward_case(
        &mut after_prefill,
        case,
        0,
        1,
        SOURCE_LAYER,
        &frequencies,
        weights.borrowed(),
    )
    .expect("short prefix failure does not consume decode");
    assert_diagnostic(case, &diagnostic);

    let mut too_long = compressed;
    too_long.extend(std::iter::repeat_n(0, fixture.model.head_dim));
    let mut after_prefill = prefetched_state(&fixture, &frequencies, weights.borrowed());
    assert!(matches!(
        forward_with_publication(
            &mut after_prefill,
            &input,
            case.start_pos,
            0,
            1,
            SOURCE_LAYER,
            &too_long,
            &legal_indices,
            call_frequencies(&frequencies, case),
            weights.borrowed()
        ),
        Err(LayerAttentionError::CompressedKeyCount {
            actual: 7,
            expected: 6
        })
    ));
    let diagnostic = forward_case(
        &mut after_prefill,
        case,
        0,
        1,
        SOURCE_LAYER,
        &frequencies,
        weights.borrowed(),
    )
    .expect("long prefix failure does not consume decode");
    assert_diagnostic(case, &diagnostic);
}

#[test]
fn zero_compressed_prefix_is_a_window_only_attention_call() {
    let fixture = fixture();
    let frequencies = frequencies(&fixture);
    let weights = weights(&fixture.encoded_parameters);
    let layout = LayerAttentionLayout::new(
        nonzero(1),
        nonzero(fixture.model.dim),
        nonzero(fixture.model.n_heads),
        nonzero(fixture.model.head_dim),
        nonzero(fixture.model.rope_head_dim / 2),
        nonzero(fixture.model.q_lora_rank),
        nonzero(fixture.model.window_size),
        nonzero(fixture.model.o_groups),
        nonzero(fixture.model.o_lora_rank),
        SOURCE_LAYER,
        nonzero(8),
        fixture.model.norm_eps,
        0.125,
    )
    .expect("window-only ratio layout");
    let mut state = LayerAttentionState::new(layout);
    let case = &fixture.cases[0];
    let call_frequencies = call_frequencies(&frequencies, case);
    let empty_slots = vec![-1; case.input.shape[1]];
    assert!(matches!(
        forward_with_publication(
            &mut state,
            &case.input.bf16(),
            case.start_pos,
            0,
            0,
            SOURCE_LAYER,
            &[],
            &empty_slots,
            call_frequencies,
            weights.borrowed()
        ),
        Err(LayerAttentionError::CompressedSlotsWithoutKeys { slots: 1 })
    ));
    forward_with_publication(
        &mut state,
        &case.input.bf16(),
        case.start_pos,
        0,
        0,
        SOURCE_LAYER,
        &[],
        &[],
        call_frequencies,
        weights.borrowed(),
    )
    .expect("all-negative empty publication failure does not consume window-only prefill");
}

#[test]
#[should_panic(expected = "attention complete capture identity")]
fn attention_harness_rejects_a_different_capture_identity() {
    native_outputs_from_inputs(&[], "not-a-capture-sha");
}

#[test]
#[should_panic(expected = "captured attention start at call 1")]
fn attention_harness_rejects_a_reordered_call_schedule() {
    let fixture = fixture();
    let inputs = vec![
        (fixture.cases[0].start_pos, fixture.cases[0].input.bf16()),
        (fixture.cases[2].start_pos, fixture.cases[2].input.bf16()),
        (fixture.cases[1].start_pos, fixture.cases[1].input.bf16()),
    ];
    native_outputs_from_inputs(
        &inputs,
        "5b4ec6ef18ddee9ca1a91cfd656f66c177aeebff2e8f0dc47b8bc05e070e021c",
    );
}
