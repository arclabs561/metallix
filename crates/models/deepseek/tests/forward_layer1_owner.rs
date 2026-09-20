//! Native layer-one ratio-two owner publication against the source fixture.

#[path = "support/layer1_owner_capture.rs"]
mod layer1_owner_capture;

#[test]
fn layer_one_ratio_two_owner_matches_source_publications() {
    let outputs = layer1_owner_capture::native_publications();
    assert_eq!(outputs.len(), 3);
    assert_eq!(outputs[0].start_pos, 0);
    assert_eq!(outputs[1].start_pos, 5);
    assert_eq!(outputs[2].start_pos, 6);
    assert!(
        outputs[2].latent.is_none(),
        "partial group stays unpublished"
    );
    assert_eq!(outputs[1].key_prefix, outputs[2].key_prefix);
    assert_eq!(outputs[1].kv_prefix, outputs[2].kv_prefix);
    assert_eq!(outputs[0].selected_indices, [-1, 5, 5, 5, 5]);
    assert_eq!(outputs[1].selected_indices, [8]);
    assert_eq!(outputs[2].selected_indices, [6]);
}

#[test]
fn partial_owner_prefix_cannot_replace_the_source_score_key_boundary() {
    assert!(layer1_owner_capture::partial_owner_keys_fail_score_gate());
}

#[test]
fn request_local_layer_three_score_state_is_reset_between_requests() {
    assert!(layer1_owner_capture::request_local_score_state_rejects_cross_request_reuse());
}

#[test]
fn partial_owner_publication_is_not_relabelled_as_source_score_operand() {
    let outputs = layer1_owner_capture::native_publications();
    assert_ne!(outputs[2].key_prefix, outputs[2].source_score_key_prefix);
    assert_eq!(
        outputs[2].source_score_key_prefix.len(),
        outputs[2].key_prefix.len()
    );
}

#[test]
fn gate_weight_corruption_changes_completed_ratio_two_latent() {
    assert!(layer1_owner_capture::first_gate_mutation_changes_latent());
}

#[test]
fn source_projection_corruption_exceeds_declared_fp32_envelope() {
    assert!(layer1_owner_capture::projection_envelope_rejects_corruption());
}

#[test]
fn source_projection_envelope_rejects_nonfinite_values() {
    assert!(layer1_owner_capture::projection_envelope_rejects_nonfinite());
}
