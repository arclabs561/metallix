//! Feed layer-three HC/RMSNorm output into the historical candidate oracle.

#[path = "support/candidate_capture.rs"]
mod candidate_capture;
#[path = "support/candidate_hc_capture.rs"]
mod candidate_hc_capture;

#[test]
fn layer_three_hc_premix_and_norm_feed_existing_candidate_oracle() {
    for (start, input) in candidate_hc_capture::derived_inputs() {
        assert_eq!(input, candidate_capture::captured_attention_input(start));
        let keys = candidate_capture::captured_keys(start);
        let candidates = candidate_capture::generated_candidates_from_attention_input(
            start,
            &keys,
            candidate_capture::source_call(start),
            &input,
        );
        assert!(!candidates.mask().is_empty());
    }
}

#[test]
fn layer_three_hc_input_mutations_do_not_cross_the_source_boundary() {
    candidate_hc_capture::assert_mutations_rejected();
}
