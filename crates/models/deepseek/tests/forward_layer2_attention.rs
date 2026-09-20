//! Native layer-two attention using the source-captured layer-one publication.

#[allow(
    dead_code,
    reason = "standalone owner controls have a separate test binary"
)]
#[path = "support/layer1_owner_capture.rs"]
mod layer1_owner_capture;
#[path = "support/layer2_attention_capture.rs"]
mod layer2_attention_capture;

#[test]
fn layer_two_attention_consumes_native_layer_one_publications() {
    let outputs = layer2_attention_capture::native_outputs();
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
fn layer_two_rejects_a_non_owner_publication() {
    assert!(layer2_attention_capture::wrong_owner_publication_is_rejected());
}

#[test]
fn legal_wrong_layer_one_index_changes_layer_two_output() {
    assert!(layer2_attention_capture::legal_wrong_index_changes_output());
}
