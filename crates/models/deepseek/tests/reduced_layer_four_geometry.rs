//! L4 rejects producer metadata that is incompatible with its fixed source geometry.

use std::num::NonZeroUsize;

use deepseek::{
    RotaryFrequency,
    attention::layer::{Fp8Projection, LayerAttentionLayout, LayerAttentionWeights},
    indexer::{
        cache::IndexKeyPublicationId,
        query::{CandidateQueryLayout, CandidateQueryWeights, IndexQueryLayout, IndexQueryWeights},
        selection::{CandidateSelection, SelectionCall, SelectionGeometry, produce_candidates},
    },
    reduced::{
        LayerFourCall, LayerFourConfig, LayerFourSession, LayerFourSessionError,
        LayerThreePublication,
    },
};

fn nz(value: usize) -> NonZeroUsize {
    NonZeroUsize::new(value).expect("test geometry is nonzero")
}

fn attention_layout(ratio: usize) -> LayerAttentionLayout {
    LayerAttentionLayout::new(
        nz(1),
        nz(32),
        nz(1),
        nz(32),
        nz(1),
        nz(32),
        nz(6),
        nz(1),
        nz(32),
        3,
        nz(ratio),
        1.0e-6,
        0.125,
    )
    .expect("bounded attention layout")
}

fn query_layout() -> CandidateQueryLayout {
    CandidateQueryLayout::new(
        IndexQueryLayout::new(nz(1), nz(32), nz(32), nz(1), nz(32), nz(1))
            .expect("bounded query layout"),
        1.0e-6,
    )
    .expect("bounded candidate query layout")
}

fn candidates(
    batch: usize,
    positions: usize,
    keys: usize,
    ratio: usize,
    offset: usize,
) -> CandidateSelection {
    let scores = vec![0; positions * keys];
    produce_candidates(
        &scores,
        SelectionCall::new(
            IndexKeyPublicationId::new(3, 0, 0),
            batch,
            SelectionGeometry::new(0, nz(positions), nz(keys), nz(ratio), offset)
                .expect("candidate geometry"),
        ),
        1,
        nz(1),
    )
    .expect("candidate mask")
}

#[test]
fn rejects_non_ratio_one_attention_layout_before_allocating_l4_state() {
    assert!(matches!(
        LayerFourConfig::new(query_layout(), attention_layout(2), nz(1)),
        Err(LayerFourSessionError::AttentionCompressionRatio)
    ));
}

#[test]
fn rejects_candidate_metadata_outside_fixed_l4_source_geometry() {
    let codes = vec![0; 32 * 32];
    let scales = [127];
    let norm = [0x3f80; 32];
    let weights = vec![0_u16; 32 * 32];
    let fp8 = Fp8Projection {
        codes: &codes,
        scales: &scales,
    };
    let query = CandidateQueryWeights {
        wq_a: fp8,
        q_norm: &norm,
        index: IndexQueryWeights {
            wq_b_codes: &codes,
            wq_b_scales: &scales,
            weights_proj: &weights[..32],
        },
    };
    let attention = LayerAttentionWeights {
        wq_a: fp8,
        q_norm: &norm,
        wq_b: fp8,
        wkv: fp8,
        kv_norm: &norm,
        attn_sink: &[0.0],
        wo_a: &weights,
        wo_b: fp8,
    };
    let input = [0; 32];
    let frequencies = [RotaryFrequency::new(1.0, 0.0).expect("finite rotary frequency")];
    let invoke = |candidate: &CandidateSelection, keys: &[u16]| {
        LayerFourSession::new(
            LayerFourConfig::new(query_layout(), attention_layout(1), nz(1))
                .expect("native L4 layout"),
        )
        .step(LayerFourCall::new(
            &input,
            &frequencies,
            query,
            attention,
            LayerThreePublication::new(
                IndexKeyPublicationId::new(3, 0, 0),
                keys,
                &input,
                candidate,
            ),
        ))
    };

    let wrong_batch = candidates(1, 1, 1, 1, 1);
    assert!(matches!(
        invoke(&wrong_batch, &input),
        Err(LayerFourSessionError::CandidateBatch { actual: 1 })
    ));

    let wrong_ratio = candidates(0, 2, 1, 2, 2);
    assert!(matches!(
        invoke(&wrong_ratio, &input),
        Err(LayerFourSessionError::CandidateCompressionRatio { actual: 2 })
    ));

    let wrong_offset = candidates(0, 1, 1, 1, 0);
    assert!(matches!(
        invoke(&wrong_offset, &input),
        Err(LayerFourSessionError::CandidateOffset {
            actual: 0,
            expected: 1
        })
    ));

    let two_keys = [0; 64];
    let wrong_key_count = candidates(0, 1, 1, 1, 1);
    assert!(matches!(
        invoke(&wrong_key_count, &two_keys),
        Err(LayerFourSessionError::CandidateKeyCount {
            actual: 1,
            expected: 2
        })
    ));
}
