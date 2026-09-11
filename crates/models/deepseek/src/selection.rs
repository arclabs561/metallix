//! CPU-only qualification of V4.1 final index selection.
//!
//! This mirrors the final `topk`, position sort, causal sentinel, and offset
//! statements in pinned
//! [`Indexer.forward`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/dba1be0a40aa45a94ad051997016db3960a90277/inference/model.py#L578).
//! It is neither candidate-mask computation nor a Metal or full-indexer claim.

use thiserror::Error;

/// Maximum score-row width accepted by this bounded CPU qualification helper.
pub const MAX_SELECTION_WIDTH: usize = 1 << 20;

/// Errors from final V4.1 index selection qualification.
#[derive(Clone, Debug, Error, Eq, PartialEq)]
#[non_exhaustive]
pub enum SelectionError {
    /// The score row exceeds the helper's explicit work bound.
    #[error("selection width {width} exceeds maximum {max_width}")]
    WidthTooLarge { width: usize, max_width: usize },
    /// The reachable compressed length exceeded the supplied score row.
    #[error("reachable length {len} exceeds selection width {width}")]
    LengthExceedsWidth { len: usize, width: usize },
    /// A score was not finite or negative infinity.
    #[error("selection logit at position {position} is neither finite nor negative infinity")]
    InvalidLogit { position: usize },
    /// A position outside the reachable prefix was not already masked.
    #[error("unreachable selection position {position} must be negative infinity")]
    UnmaskedFuturePosition { position: usize },
    /// A score tie across the cutoff would require inventing a `PyTorch` tie order.
    #[error("equal scores straddle the final Top-K cutoff")]
    AmbiguousCutoffTie,
    /// A reachable selected position cannot be offset into the `i32` API result.
    #[error("offset {offset} plus reachable position {position} does not fit i32")]
    OffsetOutOfRange { offset: usize, position: usize },
}

/// Selects one final V4.1 index-score row into position-sorted API indices.
///
/// Inputs are already causally masked scores. The result has `min(index_topk,
/// logits.len())` values, sorted by position after score selection. Reachable
/// `-∞` scores remain valid selected positions; selected future positions map
/// to `-1`. Cutoff ties are rejected unless every position in the tied group
/// is causally unreachable, since their observable result is then identical.
pub fn select_indices(
    logits: &[f32],
    compress_len: usize,
    index_topk: usize,
    offset: usize,
) -> Result<Vec<i32>, SelectionError> {
    if logits.len() > MAX_SELECTION_WIDTH {
        return Err(SelectionError::WidthTooLarge {
            width: logits.len(),
            max_width: MAX_SELECTION_WIDTH,
        });
    }
    if compress_len > logits.len() {
        return Err(SelectionError::LengthExceedsWidth {
            len: compress_len,
            width: logits.len(),
        });
    }
    for (position, &logit) in logits.iter().enumerate() {
        if !(logit.is_finite() || logit == f32::NEG_INFINITY) {
            return Err(SelectionError::InvalidLogit { position });
        }
        if position >= compress_len && logit != f32::NEG_INFINITY {
            return Err(SelectionError::UnmaskedFuturePosition { position });
        }
    }

    let select = index_topk.min(logits.len());
    if select == 0 {
        return Ok(Vec::new());
    }
    if let Some(position) = compress_len.checked_sub(1) {
        let index = offset
            .checked_add(position)
            .ok_or(SelectionError::OffsetOutOfRange { offset, position })?;
        i32::try_from(index).map_err(|_| SelectionError::OffsetOutOfRange { offset, position })?;
    }

    let mut ranked: Vec<usize> = (0..logits.len()).collect();
    if select < logits.len() {
        // Partition at the first excluded score; only the selected positions
        // need sorting later. Neither partition has an internal score order.
        let (selected, next, _) = ranked.select_nth_unstable_by(select, |&left, &right| {
            logits[right].total_cmp(&logits[left])
        });
        let cutoff = selected
            .iter()
            .map(|&position| logits[position])
            .fold(f32::INFINITY, f32::min);
        let next = logits[*next];
        if scores_equal(cutoff, next)
            && logits
                .iter()
                .enumerate()
                .any(|(position, &score)| position < compress_len && scores_equal(score, cutoff))
        {
            return Err(SelectionError::AmbiguousCutoffTie);
        }
    }

    ranked.truncate(select);
    ranked.sort_unstable();
    ranked
        .into_iter()
        .map(|position| {
            if position >= compress_len {
                Ok(-1)
            } else {
                let index = offset
                    .checked_add(position)
                    .ok_or(SelectionError::OffsetOutOfRange { offset, position })?;
                i32::try_from(index)
                    .map_err(|_| SelectionError::OffsetOutOfRange { offset, position })
            }
        })
        .collect()
}

fn scores_equal(left: f32, right: f32) -> bool {
    left.to_bits() == right.to_bits() || (left.abs().to_bits() == 0 && right.abs().to_bits() == 0)
}

#[cfg(test)]
mod tests {
    use super::{MAX_SELECTION_WIDTH, SelectionError, select_indices};
    use serde::Deserialize;

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
        index_topk: usize,
        offset: usize,
        expected_indices: Vec<i32>,
    }

    #[test]
    fn rejects_invalid_inputs_and_validates_offset_boundaries() {
        assert!(matches!(
            select_indices(&[f32::NAN], 1, 1, 0),
            Err(SelectionError::InvalidLogit { position: 0 })
        ));
        assert!(matches!(
            select_indices(&[1.0], 2, 1, 0),
            Err(SelectionError::LengthExceedsWidth { .. })
        ));
        assert!(matches!(
            select_indices(&[1.0, 0.0], 1, 1, 0),
            Err(SelectionError::UnmaskedFuturePosition { position: 1 })
        ));
        assert!(matches!(
            select_indices(&[1.0, 0.0], 2, 1, i32::MAX as usize),
            Err(SelectionError::OffsetOutOfRange { .. })
        ));
        assert_eq!(
            select_indices(&[1.0], 1, 1, i32::MAX as usize).unwrap(),
            [i32::MAX]
        );
        assert!(matches!(
            select_indices(&[1.0, 0.0], 2, 1, usize::MAX),
            Err(SelectionError::OffsetOutOfRange { .. })
        ));
    }

    #[test]
    fn rejects_reachable_cutoff_ties_but_allows_unreachable_ones() {
        assert_eq!(
            select_indices(&[2.0, 1.0, 1.0], 3, 2, 0),
            Err(SelectionError::AmbiguousCutoffTie)
        );
        assert_eq!(
            select_indices(&[1.0, f32::NEG_INFINITY, f32::NEG_INFINITY], 1, 2, 0).unwrap(),
            [0, -1]
        );
        assert_eq!(
            select_indices(&[1.0, f32::NEG_INFINITY, f32::NEG_INFINITY], 2, 2, 0),
            Err(SelectionError::AmbiguousCutoffTie)
        );
        assert_eq!(
            select_indices(&[1.0, 0.0, -0.0], 3, 2, 0),
            Err(SelectionError::AmbiguousCutoffTie)
        );
    }

    #[test]
    fn rejects_rows_larger_than_the_explicit_bound() {
        let logits = vec![f32::NEG_INFINITY; MAX_SELECTION_WIDTH + 1];
        assert!(matches!(
            select_indices(&logits, 0, 0, 0),
            Err(SelectionError::WidthTooLarge { .. })
        ));
    }

    #[test]
    fn matches_pinned_official_cpu_selection_fixture() {
        let fixture: Fixture = serde_json::from_str(include_str!(
            "../../../../fixtures/deepseek-v41/selection-reference.json"
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
        assert_eq!(fixture.source.symbol, "Indexer.forward:selection");
        assert_eq!(fixture.cases.len(), 10);

        for case in fixture.cases {
            let logits: Vec<f32> = case
                .logits
                .into_iter()
                .map(|value| value.unwrap_or(f32::NEG_INFINITY))
                .collect();
            let actual = select_indices(&logits, case.compress_len, case.index_topk, case.offset)
                .unwrap_or_else(|error| panic!("{}: {error}", case.name));
            assert_eq!(actual, case.expected_indices, "{}", case.name);
        }
    }

    #[test]
    fn composes_source_candidates_with_distinct_consumer_scores() {
        let mask = crate::candidate_mask(
            &[9.0, 8.0, 7.0, 6.0, 1.0, f32::NEG_INFINITY],
            5,
            2,
            std::num::NonZeroUsize::new(2).unwrap(),
        )
        .unwrap();
        // Source selects its best block plus the pinned partial newest block.
        assert_eq!(mask, [true, true, false, false, true, true]);
        // Consumer has its own scores. Its highest scores are outside the
        // source candidates, and the last position remains causally masked.
        let scores = [1.0, 9.0, 100.0, 99.0, 3.0, f32::NEG_INFINITY];
        let masked: Vec<f32> = scores
            .iter()
            .zip(mask)
            .map(|(&score, keep)| if keep { score } else { f32::NEG_INFINITY })
            .collect();
        assert_eq!(select_indices(&masked, 5, 2, 128).unwrap(), [129, 132]);
    }

    #[test]
    fn exhaustively_matches_an_independent_small_row_oracle() {
        const SCORES: [f32; 5] = [f32::NEG_INFINITY, -1.0, 0.0, -0.0, 1.0];
        for width in 0..=5 {
            for compress_len in 0..=width {
                let mut logits = vec![f32::NEG_INFINITY; width];
                for_each_reachable_row(&mut logits, compress_len, 0, &SCORES, &mut |row| {
                    for index_topk in 0..=width + 1 {
                        let expected = oracle_select(row, compress_len, index_topk, 17);
                        let actual = select_indices(row, compress_len, index_topk, 17);
                        assert_eq!(
                            actual, expected,
                            "width={width} compress_len={compress_len} topk={index_topk} logits={row:?}"
                        );
                    }
                });
            }
        }
    }

    fn for_each_reachable_row(
        row: &mut [f32],
        reachable: usize,
        position: usize,
        values: &[f32],
        visit: &mut impl FnMut(&[f32]),
    ) {
        if position == reachable {
            visit(row);
            return;
        }
        for &value in values {
            row[position] = value;
            for_each_reachable_row(row, reachable, position + 1, values, visit);
        }
    }

    fn oracle_select(
        logits: &[f32],
        compress_len: usize,
        index_topk: usize,
        offset: usize,
    ) -> Result<Vec<i32>, SelectionError> {
        let select = index_topk.min(logits.len());
        if select == 0 {
            return Ok(Vec::new());
        }
        let mut subsets = Vec::new();
        enumerate_subsets(logits.len(), select, 0, &mut Vec::new(), &mut subsets);
        let mut best_scores: Option<Vec<f32>> = None;
        let mut observed = Vec::new();
        for subset in subsets {
            let mut scores: Vec<f32> = subset.iter().map(|&position| logits[position]).collect();
            scores.sort_by(|left, right| score_desc_cmp(*left, *right));
            match &best_scores {
                None => {
                    best_scores = Some(scores);
                    observed = vec![subset_to_output(&subset, compress_len, offset)];
                }
                Some(best) => match score_sequence_cmp(&scores, best) {
                    std::cmp::Ordering::Less => {
                        best_scores = Some(scores);
                        observed = vec![subset_to_output(&subset, compress_len, offset)];
                    }
                    std::cmp::Ordering::Equal => {
                        let output = subset_to_output(&subset, compress_len, offset);
                        if !observed.contains(&output) {
                            observed.push(output);
                        }
                    }
                    std::cmp::Ordering::Greater => {}
                },
            }
        }
        if observed.len() == 1 {
            Ok(observed.pop().expect("one observed output"))
        } else {
            Err(SelectionError::AmbiguousCutoffTie)
        }
    }

    fn enumerate_subsets(
        width: usize,
        select: usize,
        next: usize,
        current: &mut Vec<usize>,
        output: &mut Vec<Vec<usize>>,
    ) {
        if current.len() == select {
            output.push(current.clone());
            return;
        }
        for position in next..width {
            current.push(position);
            enumerate_subsets(width, select, position + 1, current, output);
            current.pop();
        }
    }

    fn score_sequence_cmp(left: &[f32], right: &[f32]) -> std::cmp::Ordering {
        left.iter()
            .zip(right)
            .map(|(&left, &right)| score_desc_cmp(left, right))
            .find(|&order| order != std::cmp::Ordering::Equal)
            .unwrap_or(std::cmp::Ordering::Equal)
    }

    fn score_desc_cmp(left: f32, right: f32) -> std::cmp::Ordering {
        if left.to_bits() == right.to_bits()
            || (left.abs().to_bits() == 0 && right.abs().to_bits() == 0)
        {
            std::cmp::Ordering::Equal
        } else {
            right.total_cmp(&left)
        }
    }

    fn subset_to_output(subset: &[usize], compress_len: usize, offset: usize) -> Vec<i32> {
        subset
            .iter()
            .map(|&position| {
                if position < compress_len {
                    i32::try_from(offset + position).expect("small oracle offset fits")
                } else {
                    -1
                }
            })
            .collect()
    }
}
