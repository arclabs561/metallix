//! Source-captured layer-three candidate-mask qualification.

#[path = "support/candidate_capture.rs"]
mod candidate_capture;

#[test]
fn captured_candidate_masks_match_starts_zero_five_and_six() {
    for start in [0, 5, 6] {
        let keys = candidate_capture::captured_keys(start);
        let candidates = candidate_capture::generated_candidates(start, &keys);
        assert!(!candidates.is_empty(), "start {start} candidates");
    }
}

#[test]
fn candidate_mask_refuses_unmasked_future_scores() {
    candidate_capture::rejects_unmasked_future_candidate();
}

#[test]
fn captured_query_weights_are_observable_before_candidate_masking() {
    candidate_capture::query_weight_perturbation_is_observable();
}
