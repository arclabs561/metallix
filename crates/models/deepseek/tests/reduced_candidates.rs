//! Fixture-independent runtime candidate-projection contract.

use std::num::NonZeroUsize;

use deepseek::{
    RotaryFrequency,
    attention::layer::Fp8Projection,
    indexer::{
        cache::IndexKeyPublicationId,
        query::{CandidateQueryLayout, CandidateQueryWeights, IndexQueryLayout, IndexQueryWeights},
        selection::{SelectionCall, SelectionGeometry},
    },
    reduced::{CandidateProjector, CandidateProjectorError},
};

fn nonzero(value: usize) -> NonZeroUsize {
    NonZeroUsize::new(value).expect("test dimension is nonzero")
}

fn layout() -> CandidateQueryLayout {
    CandidateQueryLayout::new(
        IndexQueryLayout::new(
            nonzero(1),
            nonzero(32),
            nonzero(32),
            nonzero(2),
            nonzero(32),
            nonzero(1),
        )
        .expect("bounded test index layout"),
        1.0e-20,
    )
    .expect("candidate layout")
}

fn projector<'a>(
    low_rank_codes: &'a [u8],
    low_rank_scales: &'a [u8],
    q_norm: &'a [u16],
    index_codes: &'a [u8],
    index_scales: &'a [u8],
    weights_proj: &'a [u16],
) -> CandidateProjector<'a> {
    CandidateProjector::new(
        CandidateQueryWeights {
            wq_a: Fp8Projection {
                codes: low_rank_codes,
                scales: low_rank_scales,
            },
            q_norm,
            index: IndexQueryWeights {
                wq_b_codes: index_codes,
                wq_b_scales: index_scales,
                weights_proj,
            },
        },
        layout(),
        nonzero(32),
        nonzero(1),
        nonzero(1),
    )
}

fn call(start: usize, positions: usize, key_count: usize) -> SelectionCall {
    SelectionCall::new(
        IndexKeyPublicationId::new(3, 0, 0),
        0,
        SelectionGeometry::new(start, nonzero(positions), nonzero(key_count), nonzero(1), 0)
            .expect("call geometry"),
    )
}

#[test]
fn zero_scores_preserve_single_candidate_selection() {
    let low_rank_codes = vec![0; 32 * 32];
    let low_rank_scales = vec![127];
    let q_norm = vec![0x3f80; 32];
    let index_codes = vec![0; 64 * 32];
    let index_scales = vec![127; 2];
    let weights_proj = vec![0; 64];
    let projector = projector(
        &low_rank_codes,
        &low_rank_scales,
        &q_norm,
        &index_codes,
        &index_scales,
        &weights_proj,
    );
    let frequencies = [RotaryFrequency::new(1.0, 0.0).unwrap()];
    let output = projector
        .project(&[0; 32], &frequencies, &[0; 32], call(0, 1, 1))
        .expect("runtime candidate projection");

    assert_eq!(output.scored().scores, vec![0]);
    assert_eq!(output.candidates().mask(), &[true]);
    assert_eq!(output.select(1).unwrap().indices, vec![0]);

    let (scored, candidates) = output.into_parts();
    assert_eq!(scored.scores, vec![0]);
    assert_eq!(candidates.mask(), &[true]);
}

#[test]
fn rejects_call_geometry_even_when_flat_score_element_count_matches() {
    let low_rank_codes = vec![0; 32 * 32];
    let low_rank_scales = vec![127];
    let q_norm = vec![0x3f80; 32];
    let index_codes = vec![0; 64 * 32];
    let index_scales = vec![127; 2];
    let weights_proj = vec![0; 64];
    let projector = projector(
        &low_rank_codes,
        &low_rank_scales,
        &q_norm,
        &index_codes,
        &index_scales,
        &weights_proj,
    );

    // The call expects 1x2 scores. The supplied rows form 2x1 scores, so a
    // flat-length check alone would accept two elements.
    assert!(matches!(
        projector.project(&[0; 64], &[], &[0; 32], call(1, 1, 2)),
        Err(CandidateProjectorError::CallGeometry {
            input_positions: 2,
            key_count: 1,
            expected_positions: 1,
            expected_keys: 2,
        })
    ));
}

#[test]
fn malformed_runtime_buffers_are_rejected_without_a_fixture_fallback() {
    let low_rank_codes = vec![0; 32 * 32];
    let low_rank_scales = vec![127];
    let q_norm = vec![0x3f80; 32];
    let index_codes = vec![0; 64 * 32];
    let index_scales = vec![127; 2];
    let weights_proj = vec![0; 64];
    let projector = projector(
        &low_rank_codes,
        &low_rank_scales,
        &q_norm,
        &index_codes,
        &index_scales,
        &weights_proj,
    );
    let frequencies = [RotaryFrequency::new(1.0, 0.0).unwrap()];

    assert!(matches!(
        projector.project(&[0; 31], &frequencies, &[0; 32], call(0, 1, 1)),
        Err(CandidateProjectorError::Scored(_))
    ));
    assert!(matches!(
        projector.project(&[0; 32], &frequencies, &[0; 31], call(0, 1, 1)),
        Err(CandidateProjectorError::Scored(_))
    ));
    assert!(
        projector
            .project(&[0; 32], &frequencies, &[0; 32], call(0, 1, 1))
            .is_ok(),
        "rejected caller buffers do not alter borrowed runtime operands"
    );
}
