//! CPU-only qualification of V4.1 CSA2 candidate-block masking.
//!
//! This mirrors `select_candidate_blocks` from the pinned upstream reference
//! ([pinned source](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/dba1be0a40aa45a94ad051997016db3960a90277/inference/model.py#L583),
//! locally retained as `artifacts/v41-reference-model.py`). It is deliberately not an indexer,
//! Metal kernel, or a full V4.1 execution claim.

use std::num::NonZeroUsize;

use thiserror::Error;

/// Errors from the bounded candidate-block qualification helper.
#[derive(Clone, Debug, Error, Eq, PartialEq)]
#[non_exhaustive]
pub enum CandidateError {
    /// The supplied score row has no positions.
    #[error("candidate logits must not be empty")]
    EmptyLogits,
    /// The reachable compressed length exceeded the row width.
    #[error("reachable length {len} exceeds candidate width {width}")]
    LengthExceedsWidth { len: usize, width: usize },
    /// A score was not finite or negative infinity.
    #[error("candidate logit at position {position} is neither finite nor negative infinity")]
    InvalidLogit { position: usize },
    /// A position outside the reachable prefix was not already masked.
    #[error("unreachable candidate position {position} must be negative infinity")]
    UnmaskedFuturePosition { position: usize },
    /// Upstream uses `torch.topk`; its ordering of a cutoff tie is unspecified.
    #[error("equal finite block scores straddle the Top-K cutoff")]
    AmbiguousCutoffTie,
}

/// Produces the V4.1 CSA2 first-level candidate mask for one query.
///
/// The input is one upstream `index_score` row. `len` is its reachable
/// compressed-prefix length; entries at and beyond it must already be
/// `-∞`. The newest reachable block is pinned, exactly as the pinned
/// reference does. This qualification helper rejects finite cutoff ties rather
/// than choosing a Rust ordering for `PyTorch`'s unspecified `topk` tie break.
/// A true bit selects a whole candidate block; it is not a causality mask, so
/// a selected partial final block can include positions beyond `compress_len`.
pub fn candidate_mask(
    logits: &[f32],
    compress_len: usize,
    topk_blocks: usize,
    block_size: NonZeroUsize,
) -> Result<Vec<bool>, CandidateError> {
    if logits.is_empty() {
        return Err(CandidateError::EmptyLogits);
    }
    if compress_len > logits.len() {
        return Err(CandidateError::LengthExceedsWidth {
            len: compress_len,
            width: logits.len(),
        });
    }
    for (position, &logit) in logits.iter().enumerate() {
        if !(logit.is_finite() || logit == f32::NEG_INFINITY) {
            return Err(CandidateError::InvalidLogit { position });
        }
        if position >= compress_len && logit != f32::NEG_INFINITY {
            return Err(CandidateError::UnmaskedFuturePosition { position });
        }
    }

    let block_size = block_size.get();
    let blocks = logits.len().div_ceil(block_size);
    let select = topk_blocks.min(blocks);
    if select == 0 || compress_len == 0 {
        return Ok(vec![false; logits.len()]);
    }

    let mut scores = vec![f32::NEG_INFINITY; blocks];
    for (position, &logit) in logits.iter().enumerate() {
        let block = position / block_size;
        scores[block] = scores[block].max(logit);
    }
    let pinned = (compress_len - 1) / block_size;
    scores[pinned] = f32::INFINITY;

    let mut ranked: Vec<usize> = (0..blocks).collect();
    ranked.sort_by(|&left, &right| scores[right].total_cmp(&scores[left]));
    if select < blocks {
        let cutoff = scores[ranked[select - 1]];
        let next = scores[ranked[select]];
        let equal_finite_scores = cutoff.to_bits() == next.to_bits()
            || (cutoff.abs().to_bits() == 0 && next.abs().to_bits() == 0);
        if cutoff.is_finite() && next.is_finite() && equal_finite_scores {
            return Err(CandidateError::AmbiguousCutoffTie);
        }
    }

    let mut keep = vec![false; blocks];
    for block in ranked.into_iter().take(select) {
        if scores[block] > f32::NEG_INFINITY {
            keep[block] = true;
        }
    }
    Ok((0..logits.len())
        .map(|position| keep[position / block_size])
        .collect())
}

#[cfg(test)]
mod tests {
    use super::{CandidateError, candidate_mask};
    use serde::Deserialize;
    use std::num::NonZeroUsize;

    #[derive(Debug, Deserialize)]
    struct Fixture {
        schema_version: u8,
        source: FixtureSource,
        cases: Vec<FixtureCase>,
    }

    #[derive(Debug, Deserialize)]
    struct FixtureSource {
        revision: String,
        sha256: String,
        symbol: String,
    }

    #[derive(Debug, Deserialize)]
    struct FixtureCase {
        name: String,
        logits: Vec<Option<f32>>,
        compress_len: usize,
        topk_blocks: usize,
        block_size: usize,
        expected_mask: Vec<bool>,
    }

    fn blocks(size: usize) -> NonZeroUsize {
        NonZeroUsize::new(size).unwrap()
    }

    #[test]
    fn keeps_future_bits_within_selected_partial_block() {
        let mask = candidate_mask(&[9.0, 8.0, 1.0, f32::NEG_INFINITY], 3, 1, blocks(2)).unwrap();
        assert_eq!(mask, [false, false, true, true]);
    }

    #[test]
    fn drops_unreachable_negative_infinity_blocks() {
        let mask = candidate_mask(
            &[1.0, f32::NEG_INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY],
            1,
            4,
            blocks(1),
        )
        .unwrap();
        assert_eq!(mask, [true, false, false, false]);
    }

    #[test]
    fn permits_zero_topk() {
        let mask = candidate_mask(&[1.0, f32::NEG_INFINITY], 1, 0, blocks(1)).unwrap();
        assert_eq!(mask, [false, false]);
    }

    #[test]
    fn rejects_a_finite_cutoff_tie_including_signed_zero() {
        let error = candidate_mask(&[2.0, 0.0, -0.0, 1.0], 4, 3, blocks(1)).unwrap_err();
        assert_eq!(error, CandidateError::AmbiguousCutoffTie);
    }

    #[test]
    fn allows_ties_that_do_not_cross_the_cutoff() {
        let mask = candidate_mask(&[4.0, 3.0, 3.0, 1.0], 4, 4, blocks(1)).unwrap();
        assert_eq!(mask, [true, true, true, true]);
    }

    #[test]
    fn rejects_invalid_and_unmasked_inputs() {
        assert_eq!(
            candidate_mask(&[], 0, 1, blocks(1)),
            Err(CandidateError::EmptyLogits)
        );
        assert!(matches!(
            candidate_mask(&[f32::NAN], 1, 1, blocks(1)),
            Err(CandidateError::InvalidLogit { .. })
        ));
        assert!(matches!(
            candidate_mask(&[f32::INFINITY], 1, 1, blocks(1)),
            Err(CandidateError::InvalidLogit { .. })
        ));
        assert!(matches!(
            candidate_mask(&[1.0], 2, 1, blocks(1)),
            Err(CandidateError::LengthExceedsWidth { .. })
        ));
        assert!(matches!(
            candidate_mask(&[1.0, 0.0], 1, 1, blocks(1)),
            Err(CandidateError::UnmaskedFuturePosition { .. })
        ));
    }

    #[test]
    fn accepts_a_maximum_sized_block_without_arithmetic_overflow() {
        let mask = candidate_mask(&[1.0], 1, 1, blocks(usize::MAX)).unwrap();
        assert_eq!(mask, [true]);
    }

    #[test]
    fn matches_pinned_official_cpu_reference_fixture() {
        let fixture: Fixture = serde_json::from_str(include_str!(
            "../../../../fixtures/deepseek-v41/candidate-block-reference.json"
        ))
        .expect("fixture JSON is valid");
        assert_eq!(fixture.schema_version, 1);
        assert_eq!(
            fixture.source.revision,
            "dba1be0a40aa45a94ad051997016db3960a90277"
        );
        assert_eq!(
            fixture.source.sha256,
            "4e9ae23620edc8028ccc5d5fef552ab7fdc7dcd6f79608754fe9f67644056f65"
        );
        assert_eq!(fixture.source.symbol, "select_candidate_blocks");
        assert!(!fixture.cases.is_empty());

        for case in fixture.cases {
            let logits: Vec<f32> = case
                .logits
                .into_iter()
                .map(|value| value.unwrap_or(f32::NEG_INFINITY))
                .collect();
            let actual = candidate_mask(
                &logits,
                case.compress_len,
                case.topk_blocks,
                blocks(case.block_size),
            )
            .unwrap_or_else(|error| panic!("{}: {error}", case.name));
            assert_eq!(actual, case.expected_mask, "{}", case.name);
        }
    }
}
