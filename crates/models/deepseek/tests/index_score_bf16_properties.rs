//! Metamorphic coverage for the bounded BF16 index-score diagnostic.
//!
//! This deliberately checks only key-position permutation: reordering heads
//! would change the documented ascending FP32 head reduction.

use std::num::NonZeroUsize;

use deepseek::indexer::bf16::index_scores_bf16_reference;
use proptest::prelude::*;

#[derive(Debug)]
struct ScoreCase {
    heads: usize,
    positions: usize,
    dimension: usize,
    query: Vec<i16>,
    keys: Vec<i16>,
    weights: Vec<i16>,
}

fn score_cases() -> impl Strategy<Value = ScoreCase> {
    (1_usize..=4, 1_usize..=6, 1_usize..=8).prop_flat_map(|(heads, positions, dimension)| {
        (
            Just((heads, positions, dimension)),
            prop::collection::vec(-8_i16..=8, heads * dimension),
            prop::collection::vec(-8_i16..=8, positions * dimension),
            prop::collection::vec(-8_i16..=8, heads),
        )
            .prop_map(
                |((heads, positions, dimension), query, keys, weights)| ScoreCase {
                    heads,
                    positions,
                    dimension,
                    query,
                    keys,
                    weights,
                },
            )
    })
}

fn integer_bf16(value: i16) -> u16 {
    // The generated small integers are exactly representable in BF16.
    u16::try_from(f32::from(value).to_bits() >> 16).expect("upper FP32 bits fit BF16")
}

fn reversed_key_rows(keys: &[u16], dimension: usize) -> Vec<u16> {
    keys.chunks_exact(dimension)
        .rev()
        .flat_map(|row| row.iter().copied())
        .collect()
}

fn reverse_each_head_row(values: &[u16], heads: usize, positions: usize) -> Vec<u16> {
    assert_eq!(values.len(), heads * positions, "diagnostic matrix shape");
    values
        .chunks_exact(positions)
        .flat_map(|row| row.iter().rev().copied())
        .collect()
}

fn reverse_positions(values: &[u16]) -> Vec<u16> {
    values.iter().rev().copied().collect()
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    /// Key rows never interact in the scorer, so reversing their row-major
    /// order must reverse every key-indexed diagnostic stage exactly.
    #[test]
    fn reversing_key_positions_permutates_every_bf16_diagnostic_stage(case in score_cases()) {
        let query: Vec<_> = case.query.into_iter().map(integer_bf16).collect();
        let keys: Vec<_> = case.keys.into_iter().map(integer_bf16).collect();
        let weights: Vec<_> = case.weights.into_iter().map(integer_bf16).collect();
        let dimension = NonZeroUsize::new(case.dimension).expect("generated nonzero dimension");
        let original = index_scores_bf16_reference(&query, &keys, &weights, dimension)
            .expect("small finite generated operands");
        let reversed_keys = reversed_key_rows(&keys, case.dimension);
        let reversed = index_scores_bf16_reference(&query, &reversed_keys, &weights, dimension)
            .expect("key permutation preserves finite generated operands");

        prop_assert_eq!(
            reversed.dot_products,
            reverse_each_head_row(&original.dot_products, case.heads, case.positions),
        );
        prop_assert_eq!(
            reversed.rectified,
            reverse_each_head_row(&original.rectified, case.heads, case.positions),
        );
        prop_assert_eq!(
            reversed.weighted,
            reverse_each_head_row(&original.weighted, case.heads, case.positions),
        );
        prop_assert_eq!(reversed.scores, reverse_positions(&original.scores));
    }
}
