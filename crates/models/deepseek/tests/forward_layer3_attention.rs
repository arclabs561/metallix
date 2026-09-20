//! Native layer-three attention from the HC-derived owner input.

#[path = "support/attention_capture.rs"]
mod attention_capture;
#[path = "support/candidate_capture.rs"]
mod candidate_capture;
#[path = "support/owner_attention_capture.rs"]
mod owner_attention_capture;

use attention_capture::{
    SOURCE_LAYER, assert_diagnostic, call_frequencies, forward_case, forward_with_publication,
    frequencies, layer_three_fixture, layout, weights_for_layer,
};
use deepseek::attention::layer::{LayerAttentionError, LayerAttentionState};
use proptest::prelude::*;

#[test]
fn layer_three_hc_owner_and_producer_drive_native_attention_output() {
    let outputs = owner_attention_capture::native_layer_three_outputs_from_ownered_inputs();
    assert_eq!(outputs.len(), 3);
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(24))]

    #[test]
    fn wrong_owner_identity_is_atomic_for_layer_three(
        wrong_source in any::<u16>().prop_filter("different owner", |source| *source != SOURCE_LAYER),
    ) {
        let fixture = layer_three_fixture();
        let frequencies = frequencies(&fixture);
        let weights = weights_for_layer(&fixture.encoded_parameters, 3);
        let mut state = LayerAttentionState::new(layout(&fixture.model));
        let prefill = forward_case(&mut state, &fixture.cases[0], 0, 0, SOURCE_LAYER, &frequencies, weights.borrowed()).unwrap();
        assert_diagnostic(&fixture.cases[0], &prefill);
        let rejected = forward_case(&mut state, &fixture.cases[1], 0, 1, wrong_source, &frequencies, weights.borrowed());
        prop_assert!(matches!(rejected, Err(LayerAttentionError::WrongSourceLayer { .. })), "wrong owner must be rejected");
        let retry = forward_case(&mut state, &fixture.cases[1], 0, 1, SOURCE_LAYER, &frequencies, weights.borrowed()).unwrap();
        assert_diagnostic(&fixture.cases[1], &retry);
    }

    #[test]
    fn legal_wrong_producer_selection_changes_layer_three_output(other_key in 0_i32..4) {
        let fixture = layer_three_fixture();
        let frequencies = frequencies(&fixture);
        let weights = weights_for_layer(&fixture.encoded_parameters, 3);
        let case = &fixture.cases[0];
        let mut indices = case.compressed_indices.i32();
        prop_assert_eq!(indices[4], 9, "captured final prefill row selects key four at offset five");
        indices[4] = 5 + other_key;
        let mut state = LayerAttentionState::new(layout(&fixture.model));
        let changed = forward_with_publication(
            &mut state, &case.input.bf16(), 0, 0, 0, SOURCE_LAYER,
            &case.compressed_kv.bf16(), &indices, call_frequencies(&frequencies, case), weights.borrowed(),
        ).unwrap();
        prop_assert_ne!(changed.final_output, case.output.bf16());
    }
}
