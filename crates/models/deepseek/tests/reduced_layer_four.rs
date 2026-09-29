//! L4 consumes committed candidate identity and requires reset after failure.
use deepseek::{
    RotaryFrequency,
    attention::layer::{Fp8Projection, LayerAttentionLayout, LayerAttentionWeights},
    indexer::{
        cache::IndexKeyPublicationId,
        query::{CandidateQueryLayout, CandidateQueryWeights, IndexQueryLayout, IndexQueryWeights},
        selection::{SelectionCall, SelectionGeometry, produce_candidates},
    },
    reduced::{
        LayerFourCall, LayerFourConfig, LayerFourSession, LayerFourSessionError,
        LayerThreePublication,
    },
};
use std::num::NonZeroUsize;
fn nz(value: usize) -> NonZeroUsize {
    NonZeroUsize::new(value).unwrap()
}

#[test]
#[allow(
    clippy::too_many_lines,
    reason = "keep ordinary operands and lifecycle assertions together"
)]
fn failed_attention_poisons_and_reset_rejects_old_publication() {
    let codes = vec![0; 32 * 32];
    let scales = [127];
    let norm = [0x3f80; 32];
    let zeros = vec![0; 32 * 32];
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
            weights_proj: &zeros[..32],
        },
    };
    let attention = LayerAttentionWeights {
        wq_a: fp8,
        q_norm: &norm,
        wq_b: fp8,
        wkv: fp8,
        kv_norm: &norm,
        attn_sink: &[0.0],
        wo_a: &zeros,
        wo_b: fp8,
    };
    let layout = LayerAttentionLayout::new(
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
        nz(1),
        1.0e-6,
        0.125,
    )
    .unwrap();
    let query_layout = CandidateQueryLayout::new(
        IndexQueryLayout::new(nz(1), nz(32), nz(32), nz(1), nz(32), nz(1)).unwrap(),
        1.0e-6,
    )
    .unwrap();
    let mut session =
        LayerFourSession::new(LayerFourConfig::new(query_layout, layout, nz(1)).unwrap());
    let candidates = |epoch| {
        produce_candidates(
            &[0],
            SelectionCall::new(
                IndexKeyPublicationId::new(3, epoch, 0),
                0,
                SelectionGeometry::new(0, nz(1), nz(1), nz(1), 1).unwrap(),
            ),
            1,
            nz(1),
        )
        .unwrap()
    };
    let first = candidates(0);
    let input = &zeros[..32];
    let frequencies = [RotaryFrequency::new(1.0, 0.0).unwrap()];
    let publication =
        LayerThreePublication::new(IndexKeyPublicationId::new(3, 0, 0), input, input, &first);
    let mut bad = attention;
    bad.q_norm = &[];
    assert!(matches!(
        session.step(LayerFourCall::new(
            input,
            &frequencies,
            query,
            bad,
            publication
        )),
        Err(LayerFourSessionError::Attention(_))
    ));
    assert!(session.is_poisoned());
    assert_eq!(session.next_start(), 0);
    assert!(matches!(
        session.step(LayerFourCall::new(
            input,
            &frequencies,
            query,
            attention,
            publication
        )),
        Err(LayerFourSessionError::Poisoned)
    ));
    session.reset().unwrap();
    assert!(
        session
            .step(LayerFourCall::new(
                input,
                &frequencies,
                query,
                attention,
                publication
            ))
            .is_err()
    );
    assert!(session.is_poisoned());
    session.reset().unwrap();
    let fresh = candidates(2);
    let publication =
        LayerThreePublication::new(IndexKeyPublicationId::new(3, 2, 0), input, input, &fresh);
    let result = session
        .step(LayerFourCall::new(
            input,
            &frequencies,
            query,
            attention,
            publication,
        ))
        .unwrap();
    assert_eq!(result.selection().indices, &[1]);
    assert_eq!(result.attention().final_output, vec![0; 32]);
    assert_eq!(session.next_start(), 1);
    assert!(!session.is_poisoned());
}
