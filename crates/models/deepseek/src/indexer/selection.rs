//! Stateless source-shaped CSA2 candidate and final-selection composition.
//!
//! This module deliberately consumes already-head-summed BF16 index scores. It
//! owns neither query/key preparation, score reduction, cache publication, nor
//! a cross-layer candidate registry. A [`SelectionCall`] is caller metadata
//! checked for consistency between the candidate and final-selection phases;
//! it is not an authority token or proof that an owner issued the values.

use std::num::NonZeroUsize;

use thiserror::Error;

use crate::{
    csa2::{CandidateError, candidate_mask},
    precision::{bf16_to_f32, f32_to_bf16_rne},
    select_indices,
    selection::SelectionError,
};

use super::cache::IndexKeyPublicationId;

/// Largest `[position, key]` score matrix accepted by this adapter.
pub const MAX_SELECTION_SCORE_ELEMENTS: usize = 1 << 20;

/// The validated, source-specific geometry shared by one candidate/selection call.
///
/// This initial adapter requires a nonempty key prefix. Callers must bypass it
/// while compression has not completed a first group (`key_count == 0`).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SelectionGeometry {
    token_start: usize,
    positions: NonZeroUsize,
    key_count: NonZeroUsize,
    compression_ratio: NonZeroUsize,
    offset: usize,
}

impl SelectionGeometry {
    /// Validates a score matrix and its source causal prefix.
    pub fn new(
        token_start: usize,
        positions: NonZeroUsize,
        key_count: NonZeroUsize,
        compression_ratio: NonZeroUsize,
        offset: usize,
    ) -> Result<Self, SelectionGeometryError> {
        let score_elements = positions.get().checked_mul(key_count.get()).ok_or(
            SelectionGeometryError::ShapeOverflow {
                field: "score matrix",
            },
        )?;
        if score_elements > MAX_SELECTION_SCORE_ELEMENTS {
            return Err(SelectionGeometryError::ElementLimit {
                elements: score_elements,
                maximum: MAX_SELECTION_SCORE_ELEMENTS,
            });
        }
        let end = token_start
            .checked_add(positions.get())
            .ok_or(SelectionGeometryError::PositionOverflow)?;
        if token_start != 0 && positions.get() != 1 {
            return Err(SelectionGeometryError::DecodeMustHaveOnePosition {
                positions: positions.get(),
            });
        }
        let reachable = end / compression_ratio.get();
        if key_count.get() != reachable {
            return Err(SelectionGeometryError::KeyCountMismatch {
                actual: key_count.get(),
                expected: reachable,
            });
        }
        let highest_reachable = reachable - 1;
        let highest_index = offset.checked_add(highest_reachable).ok_or(
            SelectionGeometryError::OffsetOutOfRange {
                offset,
                position: highest_reachable,
            },
        )?;
        if i32::try_from(highest_index).is_err() {
            return Err(SelectionGeometryError::OffsetOutOfRange {
                offset,
                position: highest_reachable,
            });
        }
        Ok(Self {
            token_start,
            positions,
            key_count,
            compression_ratio,
            offset,
        })
    }

    fn score_elements(self) -> usize {
        self.positions.get() * self.key_count.get()
    }

    fn reachable(self, position: usize) -> usize {
        (self.token_start + position + 1) / self.compression_ratio.get()
    }
}

/// Caller-supplied metadata for a single-batch index-selection call.
///
/// `batch_index` remains explicit because this narrow adapter processes one
/// batch at a time. It is metadata only: the adapter does not assert that a
/// cache owner authenticated it.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SelectionCall {
    publication: IndexKeyPublicationId,
    batch_index: usize,
    geometry: SelectionGeometry,
}

impl SelectionCall {
    /// Groups the key publication identity with a validated selection geometry.
    #[must_use]
    pub const fn new(
        publication: IndexKeyPublicationId,
        batch_index: usize,
        geometry: SelectionGeometry,
    ) -> Self {
        Self {
            publication,
            batch_index,
            geometry,
        }
    }
}

/// Opaque candidate mask tied to exactly one [`SelectionCall`].
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CandidateSelection {
    call: SelectionCall,
    causal_scores: Vec<u16>,
    mask: Vec<bool>,
}

impl CandidateSelection {
    /// Borrows the caller metadata this candidate output was built for.
    #[must_use]
    pub const fn call(&self) -> SelectionCall {
        self.call
    }

    /// Borrows the source causal BF16 stage preceding block selection.
    #[must_use]
    pub fn causal_scores(&self) -> &[u16] {
        &self.causal_scores
    }

    /// Borrows the row-major source candidate mask.
    #[must_use]
    pub fn mask(&self) -> &[bool] {
        &self.mask
    }
}

/// Source-visible outputs from causal masking, candidate filtering, and final selection.
#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub struct SelectionDiagnostic {
    /// BF16 score matrix after the source causal `-∞` mask.
    pub causal_scores: Vec<u16>,
    /// BF16 score matrix after the candidate mask.
    pub masked_scores: Vec<u16>,
    /// Final source-domain indices, row-major by query position.
    pub indices: Vec<i32>,
}

/// Invalid selection geometry before score storage is inspected.
#[derive(Clone, Copy, Debug, Error, Eq, PartialEq)]
#[non_exhaustive]
pub enum SelectionGeometryError {
    /// A derived score-matrix element count overflowed `usize`.
    #[error("selection geometry overflowed for {field}")]
    ShapeOverflow { field: &'static str },
    /// The exclusive token end could not be represented.
    #[error("selection token end overflowed")]
    PositionOverflow,
    /// The bounded score matrix exceeds this CPU adapter's limit.
    #[error("selection score matrix has {elements} elements, maximum is {maximum}")]
    ElementLimit { elements: usize, maximum: usize },
    /// The supplied key prefix is not exactly the source causal prefix.
    #[error("key count {actual} does not equal causal reachable prefix {expected}")]
    KeyCountMismatch { actual: usize, expected: usize },
    /// Decode source calls must contain exactly one token position.
    #[error("selection decode at nonzero token start requires one position, got {positions}")]
    DecodeMustHaveOnePosition { positions: usize },
    /// The offset plus highest reachable compressed key cannot fit the i32 index API.
    #[error("offset {offset} plus reachable position {position} does not fit i32")]
    OffsetOutOfRange { offset: usize, position: usize },
}

/// Rejected stateless candidate or final-selection work.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum SelectionAdapterError {
    /// Input storage did not have exactly one BF16 score per query/key pair.
    #[error("selection score length is {actual}, expected {expected}")]
    ScoreLength { actual: usize, expected: usize },
    /// A source score is non-finite before the adapter applies masking.
    #[error("selection source score at element {element} is non-finite")]
    NonFiniteScore { element: usize },
    /// Candidate output belongs to a different caller-asserted selection call.
    #[error("candidate selection metadata does not match this selection call")]
    CandidateCallMismatch,
    /// Candidate block selection rejected its source-shaped score row.
    #[error(transparent)]
    Candidate(#[from] CandidateError),
    /// Final selection rejected its source-shaped masked score row.
    #[error(transparent)]
    FinalSelection(#[from] SelectionError),
    /// A diagnostic allocation could not reserve its bounded output storage.
    #[error("could not allocate {elements} selection diagnostic elements")]
    AllocationFailed { elements: usize },
}

/// Produces one source candidate mask per query row from finite BF16 score storage.
///
/// The source causal `-∞` mask is applied before [`candidate_mask`]. The
/// returned mask is opaque and may only be consumed with the identical
/// [`SelectionCall`], including its publication identity, batch, and offset.
pub fn produce_candidates(
    scores: &[u16],
    call: SelectionCall,
    topk_blocks: usize,
    block_size: NonZeroUsize,
) -> Result<CandidateSelection, SelectionAdapterError> {
    let causal_scores = causal_scores(scores, call)?;
    let mut mask = reserve_bool(causal_scores.len())?;
    for (position, row) in causal_scores
        .chunks_exact(call.geometry.key_count.get())
        .enumerate()
    {
        let row_mask = candidate_mask(
            &row.iter()
                .map(|&bits| bf16_to_f32(bits))
                .collect::<Vec<_>>(),
            call.geometry.reachable(position),
            topk_blocks,
            block_size,
        )?;
        mask.extend(row_mask);
    }
    Ok(CandidateSelection {
        call,
        causal_scores,
        mask,
    })
}

/// Applies an opaque candidate mask and performs final source index selection.
///
/// These are the consumer's scores, which can differ from the producer scores
/// retained in [`CandidateSelection::causal_scores`]. CSA2 intentionally shares
/// a producer's candidate mask across layers with different query projections;
/// matching call metadata does not imply matching score values.
///
/// This repeats causal masking from the finite source score matrix so the
/// diagnostic makes both source boundaries explicit. It does not mutate score,
/// query, key, or cache storage.
pub fn select_from_candidates(
    scores: &[u16],
    call: SelectionCall,
    candidates: &CandidateSelection,
    index_topk: usize,
) -> Result<SelectionDiagnostic, SelectionAdapterError> {
    if candidates.call != call {
        return Err(SelectionAdapterError::CandidateCallMismatch);
    }
    let causal_scores = causal_scores(scores, call)?;
    let mut masked_scores = reserve_u16(causal_scores.len())?;
    for (&score, &keep) in causal_scores.iter().zip(&candidates.mask) {
        masked_scores.push(if keep { score } else { negative_infinity() });
    }
    let rows = call.geometry.positions.get();
    let per_row = index_topk.min(call.geometry.key_count.get());
    let index_elements =
        rows.checked_mul(per_row)
            .ok_or(SelectionAdapterError::AllocationFailed {
                elements: usize::MAX,
            })?;
    let mut indices = Vec::new();
    indices.try_reserve_exact(index_elements).map_err(|_| {
        SelectionAdapterError::AllocationFailed {
            elements: index_elements,
        }
    })?;
    for (position, row) in masked_scores
        .chunks_exact(call.geometry.key_count.get())
        .enumerate()
    {
        indices.extend(select_indices(
            &row.iter()
                .map(|&bits| bf16_to_f32(bits))
                .collect::<Vec<_>>(),
            call.geometry.reachable(position),
            index_topk,
            call.geometry.offset,
        )?);
    }
    Ok(SelectionDiagnostic {
        causal_scores,
        masked_scores,
        indices,
    })
}

fn causal_scores(scores: &[u16], call: SelectionCall) -> Result<Vec<u16>, SelectionAdapterError> {
    let expected = call.geometry.score_elements();
    if scores.len() != expected {
        return Err(SelectionAdapterError::ScoreLength {
            actual: scores.len(),
            expected,
        });
    }
    if let Some(element) = scores
        .iter()
        .position(|&bits| !bf16_to_f32(bits).is_finite())
    {
        return Err(SelectionAdapterError::NonFiniteScore { element });
    }
    let mut output = reserve_u16(expected)?;
    for (position, row) in scores
        .chunks_exact(call.geometry.key_count.get())
        .enumerate()
    {
        let reachable = call.geometry.reachable(position);
        output.extend(row.iter().enumerate().map(|(key, &score)| {
            if key < reachable {
                score
            } else {
                negative_infinity()
            }
        }));
    }
    Ok(output)
}

fn reserve_u16(elements: usize) -> Result<Vec<u16>, SelectionAdapterError> {
    let mut output = Vec::new();
    output
        .try_reserve_exact(elements)
        .map_err(|_| SelectionAdapterError::AllocationFailed { elements })?;
    Ok(output)
}

fn reserve_bool(elements: usize) -> Result<Vec<bool>, SelectionAdapterError> {
    let mut output = Vec::new();
    output
        .try_reserve_exact(elements)
        .map_err(|_| SelectionAdapterError::AllocationFailed { elements })?;
    Ok(output)
}

fn negative_infinity() -> u16 {
    f32_to_bf16_rne(f32::NEG_INFINITY)
}

#[cfg(test)]
mod tests {
    use super::{
        SelectionAdapterError, SelectionCall, SelectionGeometry, SelectionGeometryError,
        produce_candidates, select_from_candidates,
    };
    use crate::indexer::cache::IndexKeyPublicationId;
    use std::num::NonZeroUsize;

    fn nz(value: usize) -> NonZeroUsize {
        NonZeroUsize::new(value).expect("test dimensions are nonzero")
    }

    fn call() -> SelectionCall {
        SelectionCall::new(
            IndexKeyPublicationId::new(3, 0, 0),
            0,
            SelectionGeometry::new(0, nz(2), nz(2), nz(1), 5).expect("small call"),
        )
    }

    #[test]
    fn geometry_rejects_causal_and_score_matrix_overflow() {
        assert!(matches!(
            SelectionGeometry::new(4, nz(1), nz(1), nz(2), 0),
            Err(SelectionGeometryError::KeyCountMismatch {
                actual: 1,
                expected: 2
            })
        ));
        assert!(matches!(
            SelectionGeometry::new(0, nz(1), nz((1 << 20) + 1), nz(1), 0),
            Err(SelectionGeometryError::ElementLimit { .. })
        ));
    }

    #[test]
    fn geometry_rejects_multitoken_decode_and_zero_reachable_prefix() {
        assert!(matches!(
            SelectionGeometry::new(1, nz(2), nz(3), nz(1), 0),
            Err(SelectionGeometryError::DecodeMustHaveOnePosition { positions: 2 })
        ));
        assert!(matches!(
            SelectionGeometry::new(0, nz(1), nz(1), nz(2), 0),
            Err(SelectionGeometryError::KeyCountMismatch {
                actual: 1,
                expected: 0
            })
        ));
    }

    #[test]
    fn candidate_mask_and_final_selection_share_exact_call_metadata() {
        let call = call();
        let scores = [0x3f80, 0x4000, 0x4040, 0x4080];
        let candidates = produce_candidates(&scores, call, 1, nz(1)).expect("candidates");
        let output = select_from_candidates(&scores, call, &candidates, 1).expect("selection");
        assert_eq!(output.causal_scores, [0x3f80, 0xff80, 0x4040, 0x4080]);
        assert_eq!(candidates.call(), call);
        assert_eq!(candidates.causal_scores(), output.causal_scores);
        assert_eq!(candidates.mask(), [true, false, false, true]);
        assert_eq!(output.masked_scores, [0x3f80, 0xff80, 0xff80, 0x4080]);
        assert_eq!(output.indices, [5, 6]);

        let wrong_call = SelectionCall::new(
            IndexKeyPublicationId::new(3, 0, 1),
            0,
            SelectionGeometry::new(0, nz(2), nz(2), nz(1), 5).expect("small call"),
        );
        assert!(matches!(
            select_from_candidates(&scores, wrong_call, &candidates, 1),
            Err(SelectionAdapterError::CandidateCallMismatch)
        ));
    }

    #[test]
    fn source_scores_must_be_finite_before_masking() {
        let error = produce_candidates(&[0x7f80; 4], call(), 1, nz(1))
            .expect_err("source infinity is not a mask");
        assert!(matches!(
            error,
            SelectionAdapterError::NonFiniteScore { element: 0 }
        ));
    }
}
