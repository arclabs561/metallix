//! Invariants for the stateless source-shaped index selection adapter.

use std::num::NonZeroUsize;

use deepseek::{
    indexer::{
        cache::IndexKeyPublicationId,
        selection::{SelectionCall, SelectionGeometry, produce_candidates, select_from_candidates},
    },
    select_indices,
};
use proptest::{prelude::*, test_runner::RngSeed};

fn nz(value: usize) -> NonZeroUsize {
    NonZeroUsize::new(value).expect("positive test geometry")
}

fn bits_to_f32(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}

fn distinct_scores(count: usize) -> Vec<u16> {
    (0..count)
        .map(|position| {
            // Distinct normal BF16 values, descending by storage position.
            0x3f00 + u16::try_from((count - position) * 16).expect("small score width")
        })
        .collect()
}

fn geometry(
    start: usize,
    positions: usize,
    keys: usize,
    ratio: usize,
    offset: usize,
) -> SelectionGeometry {
    SelectionGeometry::new(start, nz(positions), nz(keys), nz(ratio), offset)
        .expect("bounded source geometry")
}

fn call(
    publication: IndexKeyPublicationId,
    batch: usize,
    geometry: SelectionGeometry,
) -> SelectionCall {
    SelectionCall::new(publication, batch, geometry)
}

proptest! {
    #![proptest_config(ProptestConfig {
        cases: 24,
        rng_seed: RngSeed::Fixed(0x5e1e_c710),
        .. ProptestConfig::default()
    })]

    #[test]
    fn decode_causal_selection_matches_direct_selection_with_monotonic_offsets(
        ratio in 1_usize..5,
        keys in 1_usize..7,
        index_topk in 1_usize..8,
        offset in 0_usize..100,
        translation in 1_usize..100,
    ) {
        // `start + 1 == keys * ratio`, so a one-token decode sees exactly
        // `keys` compressed positions for every generated ratio.
        let start = keys * ratio - 1;
        let scores = distinct_scores(keys);
        let low = call(
            IndexKeyPublicationId::new(3, 0, 0),
            0,
            geometry(start, 1, keys, ratio, offset),
        );
        let high = call(
            IndexKeyPublicationId::new(3, 0, 0),
            0,
            geometry(start, 1, keys, ratio, offset + translation),
        );
        let low_candidates = produce_candidates(&scores, low, keys, nz(1))
            .expect("all singleton candidate blocks");
        let high_candidates = produce_candidates(&scores, high, keys, nz(1))
            .expect("all singleton candidate blocks");
        let low_output = select_from_candidates(&scores, low, &low_candidates, index_topk)
            .expect("finite decode selection");
        let high_output = select_from_candidates(&scores, high, &high_candidates, index_topk)
            .expect("finite translated decode selection");
        let direct_scores: Vec<f32> = scores.iter().copied().map(bits_to_f32).collect();
        let expected = select_indices(&direct_scores, keys, index_topk, offset)
            .expect("direct unique-score selection");
        prop_assert_eq!(&low_output.indices, &expected);
        prop_assert_eq!(
            high_output.indices,
            low_output.indices.iter().map(|&index| {
                i32::try_from(usize::try_from(index).expect("reachable index") + translation)
                    .expect("small translated index")
            }).collect::<Vec<_>>(),
        );
    }

    #[test]
    fn candidate_metadata_mismatches_reject_while_baseline_and_repeat_are_stable(
        source in 4_u16..64,
        epoch in 1_u64..128,
        call_id in 1_u64..128,
        batch in 1_usize..8,
        offset in 0_usize..100,
    ) {
        let scores = [0x4080_u16];
        let baseline_geometry = geometry(0, 1, 1, 1, offset);
        let baseline = call(IndexKeyPublicationId::new(3, 0, 0), 0, baseline_geometry);
        let candidates = produce_candidates(&scores, baseline, 1, nz(1))
            .expect("baseline candidates");
        let first = select_from_candidates(&scores, baseline, &candidates, 1)
            .expect("baseline selection");
        let repeated = select_from_candidates(&scores, baseline, &candidates, 1)
            .expect("identical repeated selection");
        prop_assert_eq!(first, repeated);
        for mismatched in [
            call(IndexKeyPublicationId::new(source, 0, 0), 0, baseline_geometry),
            call(IndexKeyPublicationId::new(3, epoch, 0), 0, baseline_geometry),
            call(IndexKeyPublicationId::new(3, 0, call_id), 0, baseline_geometry),
            call(IndexKeyPublicationId::new(3, 0, 0), batch, baseline_geometry),
        ] {
            prop_assert!(select_from_candidates(&scores, mismatched, &candidates, 1).is_err());
        }
        prop_assert!(select_from_candidates(&scores, baseline, &candidates, 1).is_ok());
    }
}

#[test]
fn selected_partial_blocks_preserve_candidate_bits_but_final_causality_wins() {
    // Four prefill positions have reachability 1, 2, 3, 4. The single pinned
    // candidate block has size two; its partial rows deliberately preserve
    // future candidate bits, which final selection must still mask to -infinity.
    let scores = [
        0x4080, 0x4040, 0x4000, 0x3f80, // row 0
        0x4080, 0x4040, 0x4000, 0x3f80, // row 1
        0x4080, 0x4040, 0x4000, 0x3f80, // row 2
        0x4080, 0x4040, 0x4000, 0x3f80, // row 3
    ];
    let call = call(
        IndexKeyPublicationId::new(3, 0, 0),
        0,
        geometry(0, 4, 4, 1, 7),
    );
    let candidates = produce_candidates(&scores, call, 1, nz(2)).expect("partial blocks");
    assert_eq!(
        candidates.mask(),
        [
            true, true, false, false, true, true, false, false, false, false, true, true, false,
            false, true, true,
        ]
    );
    let output =
        select_from_candidates(&scores, call, &candidates, 4).expect("final causal selection");
    assert_eq!(
        output.causal_scores[1], 0xff80,
        "row-zero future is causal -inf"
    );
    assert_eq!(
        output.masked_scores[1], 0xff80,
        "candidate true does not unmask future"
    );
    assert_eq!(
        output.causal_scores[11], 0xff80,
        "partial final block retains a future bit"
    );
    assert_eq!(
        output.masked_scores[11], 0xff80,
        "final mask preserves causal -inf"
    );
    assert_eq!(
        output.indices,
        [7, -1, -1, -1, 7, 8, -1, -1, 7, 8, 9, -1, 7, 8, 9, 10]
    );
}

#[test]
fn source_candidates_intentionally_filter_distinct_consumer_scores() {
    // CSA2 candidates belong to the producer layer; final scores belong to
    // its consumer. They share call metadata and a mask, not score values.
    let call = call(
        IndexKeyPublicationId::new(3, 0, 0),
        0,
        geometry(2, 1, 3, 1, 0),
    );
    let producer_scores = [0x4110, 0x4000, 0x3f80]; // 9, 2, 1
    let candidates =
        produce_candidates(&producer_scores, call, 2, nz(1)).expect("producer candidate blocks");
    assert_eq!(candidates.mask(), [true, false, true]);
    let consumer_scores = [0x3f80, 0x4110, 0x4040]; // 1, 9, 3
    let output = select_from_candidates(&consumer_scores, call, &candidates, 1)
        .expect("same-call consumer selection");
    assert_ne!(candidates.causal_scores(), output.causal_scores);
    assert_eq!(output.indices, [2], "masked consumer best is position two");
}

#[test]
fn raw_nan_in_a_future_cell_is_rejected_before_causal_masking() {
    let mut scores = [0x3f80_u16; 16];
    scores[3] = 0x7fc0; // row-zero key three is future but still raw source input.
    let call = call(
        IndexKeyPublicationId::new(3, 0, 0),
        0,
        geometry(0, 4, 4, 1, 0),
    );
    assert!(produce_candidates(&scores, call, 1, nz(2)).is_err());
}
