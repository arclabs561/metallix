//! Runtime layer-three candidate-mask projection.
//!
//! This component derives candidate scores and masks from caller-owned BF16
//! attention input, reconstructed index keys, rotary frequencies, and typed
//! query weights. It has no fixture, source-capture, or expected-output
//! dependency.

use std::num::NonZeroUsize;

use thiserror::Error;

use crate::{
    RotaryFrequency,
    indexer::{
        query::{
            CandidateQueryLayout, CandidateQueryWeights, IndexKeyView, ScoredQueryDiagnostic,
            ScoredQueryError, prepare_scored_query,
        },
        selection::{
            CandidateSelection, SelectionAdapterError, SelectionCall, SelectionDiagnostic,
            produce_candidates, select_from_candidates,
        },
    },
};

/// Immutable numerical configuration for one layer-three candidate producer.
#[derive(Clone, Copy, Debug)]
pub struct CandidateProjector<'a> {
    weights: CandidateQueryWeights<'a>,
    layout: CandidateQueryLayout,
    key_head_dimension: NonZeroUsize,
    topk_blocks: NonZeroUsize,
    block_size: NonZeroUsize,
}

impl<'a> CandidateProjector<'a> {
    /// Groups borrowed query operands with the caller-selected mask policy.
    #[must_use]
    pub const fn new(
        weights: CandidateQueryWeights<'a>,
        layout: CandidateQueryLayout,
        key_head_dimension: NonZeroUsize,
        topk_blocks: NonZeroUsize,
        block_size: NonZeroUsize,
    ) -> Self {
        Self {
            weights,
            layout,
            key_head_dimension,
            topk_blocks,
            block_size,
        }
    }

    /// Derives scores and a causal candidate mask for one caller-provided publication.
    ///
    /// The selection call remains explicit because cache/publication provenance is
    /// owned by the surrounding request. Its score matrix dimensions must match
    /// the supplied input rows and reconstructed key rows exactly.
    pub fn project(
        &self,
        input: &[u16],
        frequencies: &[RotaryFrequency],
        keys: &[u16],
        call: SelectionCall,
    ) -> Result<CandidateProjection, CandidateProjectorError> {
        let (batches, input_width) = self.layout.input_geometry();
        if batches != 1 {
            return Err(CandidateProjectorError::BatchCount { actual: batches });
        }

        // Leave malformed buffers to the leaf so it reports the precise
        // input/key error rather than treating a partial row as a geometry
        // disagreement.
        if input.len().is_multiple_of(input_width)
            && keys.len().is_multiple_of(self.key_head_dimension.get())
        {
            let input_positions = input.len() / input_width;
            let key_count = keys.len() / self.key_head_dimension.get();
            let (expected_positions, expected_keys) = call.score_shape();
            if input_positions != expected_positions || key_count != expected_keys {
                return Err(CandidateProjectorError::CallGeometry {
                    input_positions,
                    key_count,
                    expected_positions,
                    expected_keys,
                });
            }
        }

        let keys = IndexKeyView::new(keys, self.key_head_dimension)?;
        let scored = prepare_scored_query(input, frequencies, self.weights, self.layout, keys)?;
        let candidates = produce_candidates(
            &scored.scores,
            call,
            self.topk_blocks.get(),
            self.block_size,
        )?;
        Ok(CandidateProjection { scored, candidates })
    }
}

/// Fresh scores and their candidate mask for one runtime call.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CandidateProjection {
    scored: ScoredQueryDiagnostic,
    candidates: CandidateSelection,
}

impl CandidateProjection {
    /// Returns all numerical query and score boundaries computed from runtime operands.
    #[must_use]
    pub const fn scored(&self) -> &ScoredQueryDiagnostic {
        &self.scored
    }

    /// Returns the opaque candidate mask tied to this call's publication identity.
    #[must_use]
    pub const fn candidates(&self) -> &CandidateSelection {
        &self.candidates
    }

    /// Applies this candidate mask to this producer's freshly computed scores.
    pub fn select(
        &self,
        index_topk: usize,
    ) -> Result<SelectionDiagnostic, CandidateProjectorError> {
        Ok(select_from_candidates(
            &self.scored.scores,
            self.candidates.call(),
            &self.candidates,
            index_topk,
        )?)
    }

    /// Consumes the diagnostic for adapters that retain both score and mask outputs.
    #[must_use]
    pub fn into_parts(self) -> (ScoredQueryDiagnostic, CandidateSelection) {
        (self.scored, self.candidates)
    }
}

/// Rejected candidate-projector configuration or runtime input.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum CandidateProjectorError {
    /// This narrow candidate scorer accepts exactly one explicit batch.
    #[error("candidate projector requires exactly one batch, got {actual}")]
    BatchCount { actual: usize },
    /// Input and key rows disagree with the caller's selection-call matrix.
    #[error(
        "candidate projector input/key geometry is [{input_positions}, {key_count}], expected [{expected_positions}, {expected_keys}]"
    )]
    CallGeometry {
        input_positions: usize,
        key_count: usize,
        expected_positions: usize,
        expected_keys: usize,
    },
    /// Query scoring rejected caller-owned operands.
    #[error(transparent)]
    Scored(#[from] ScoredQueryError),
    /// Candidate masking or final selection rejected the score matrix.
    #[error(transparent)]
    Selection(#[from] SelectionAdapterError),
}
