//! Native layer-one attention using its native ratio-two owner publication.

#[path = "support/layer1_attention_capture.rs"]
mod layer1_attention_capture;

#[test]
fn layer_one_attention_consumes_native_owner_publications() {
    let outputs = layer1_attention_capture::native_outputs();
    assert_eq!(outputs.len(), 3);
    assert_eq!(
        outputs
            .iter()
            .map(|output| output.start_pos)
            .collect::<Vec<_>>(),
        [0, 5, 6]
    );
    assert!(outputs.iter().all(|output| !output.output.is_empty()));
}

#[test]
fn layer_one_rejects_a_non_owner_publication() {
    assert!(layer1_attention_capture::wrong_owner_publication_is_rejected());
}

#[test]
fn legal_wrong_layer_one_index_changes_layer_one_output() {
    assert!(layer1_attention_capture::legal_wrong_index_changes_output());
}
