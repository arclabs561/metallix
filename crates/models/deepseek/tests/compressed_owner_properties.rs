//! Multi-batch properties for the coupled ratio-one owner prefixes.

use std::num::NonZeroUsize;

use deepseek::{
    RotaryFrequency,
    indexer::{
        cache::IndexKeyPublicationId,
        key::{IndexKeyLayout, IndexKeyWeights},
        owner::{RatioOneCompressedOwner, RatioOneOwnerCall, RatioOneOwnerWeights},
    },
};
use proptest::prelude::*;

const INPUT_DIMENSION: usize = 4;
const LATENT_DIMENSION: usize = 16;
const KEY_DIMENSION: usize = 32;

fn nz(value: usize) -> NonZeroUsize {
    NonZeroUsize::new(value).expect("positive test geometry")
}

fn identity_frequency() -> RotaryFrequency {
    RotaryFrequency::new(1.0, 0.0).expect("finite identity frequency")
}

fn weights() -> (Vec<u16>, Vec<u16>, Vec<u16>) {
    let mut wkv = vec![0_u16; LATENT_DIMENSION * INPUT_DIMENSION];
    for output in 0..LATENT_DIMENSION {
        wkv[output * INPUT_DIMENSION + output % INPUT_DIMENSION] = 0x3f80;
    }
    let mut wk = vec![0_u16; KEY_DIMENSION * LATENT_DIMENSION];
    for output in 0..KEY_DIMENSION {
        wk[output * LATENT_DIMENSION + output % LATENT_DIMENSION] = 0x3f80;
    }
    (wkv, wk, vec![0x3f80; KEY_DIMENSION])
}

fn owner(batches: usize, capacity: usize) -> RatioOneCompressedOwner {
    RatioOneCompressedOwner::new(
        IndexKeyLayout::new(
            nz(batches),
            nz(LATENT_DIMENSION),
            nz(KEY_DIMENSION),
            nz(1),
            1.0e-5,
        )
        .expect("small source key layout"),
        nz(INPUT_DIMENSION),
        nz(capacity),
        3,
        &[0x3f80; LATENT_DIMENSION],
        1.0e-5,
    )
    .expect("bounded coupled owner")
}

fn call<'a>(
    call_id: u64,
    start: usize,
    positions: usize,
    input: &'a [u16],
    frequencies: &'a [RotaryFrequency],
    weights: RatioOneOwnerWeights<'a>,
) -> RatioOneOwnerCall<'a> {
    RatioOneOwnerCall::new(
        IndexKeyPublicationId::new(3, 0, call_id),
        start,
        nz(positions),
        input,
        frequencies,
        weights,
    )
}

fn one_position_batch_major(
    input: &[u16],
    batches: usize,
    positions: usize,
    position: usize,
) -> Vec<u16> {
    let mut output = Vec::with_capacity(batches * INPUT_DIMENSION);
    for batch in 0..batches {
        let start = (batch * positions + position) * INPUT_DIMENSION;
        output.extend_from_slice(&input[start..start + INPUT_DIMENSION]);
    }
    output
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(24))]
    #[test]
    fn whole_and_token_chunked_calls_publish_identical_batched_prefixes(
        batches in 1_usize..4,
        positions in 1_usize..5,
        payload in prop::collection::vec(0x3c00_u16..0x4000, 192),
    ) {
        let required = batches * positions * INPUT_DIMENSION;
        let input = &payload[..required];
        let (wkv, wk, key_norm) = weights();
        let owner_weights = RatioOneOwnerWeights::new(&wkv, IndexKeyWeights::new(&wk, &key_norm));
        let frequencies = vec![identity_frequency(); positions];
        let mut whole = owner(batches, positions + 1);
        let mut chunked = owner(batches, positions + 1);

        whole.forward(call(0, 0, positions, input, &frequencies, owner_weights))
            .expect("whole prefill");
        for position in 0..positions {
            let token = one_position_batch_major(input, batches, positions, position);
            chunked.forward(call(
                u64::try_from(position).expect("small call ordinal"),
                position,
                1,
                &token,
                &frequencies[position..=position],
                owner_weights,
            )).expect("one-token continuation");
        }
        for batch in 0..batches {
            prop_assert_eq!(
                whole.key_prefix(batch).expect("whole key prefix"),
                chunked.key_prefix(batch).expect("chunked key prefix"),
            );
            prop_assert_eq!(
                whole.kv_prefix(batch).expect("whole KV prefix"),
                chunked.kv_prefix(batch).expect("chunked KV prefix"),
            );
        }
        prop_assert_eq!(whole.next_position(), positions);
        prop_assert_eq!(chunked.next_position(), positions);
        prop_assert_eq!(whole.valid_positions(), positions);
        prop_assert_eq!(chunked.valid_positions(), positions);
        prop_assert_eq!(whole.next_call_id(), 1);
        prop_assert_eq!(chunked.next_call_id(), u64::try_from(positions).expect("small ordinal"));
    }

    #[test]
    fn rejected_key_frequency_preserves_live_prefixes_for_same_id_retry(
        batches in 1_usize..4,
        payload in prop::collection::vec(0x3c00_u16..0x4000, 24),
    ) {
        let (wkv, wk, key_norm) = weights();
        let owner_weights = RatioOneOwnerWeights::new(&wkv, IndexKeyWeights::new(&wk, &key_norm));
        let frequency = [identity_frequency()];
        let first = &payload[..batches * INPUT_DIMENSION];
        let mut candidate = owner(batches, 4);
        candidate.forward(call(0, 0, 1, first, &frequency, owner_weights))
            .expect("initial publication");
        let mut clean = candidate.clone();
        let key_before: Vec<Vec<u16>> = (0..batches)
            .map(|batch| candidate.key_prefix(batch).expect("key prefix").to_vec())
            .collect();
        let kv_before: Vec<Vec<u16>> = (0..batches)
            .map(|batch| candidate.kv_prefix(batch).expect("KV prefix").to_vec())
            .collect();
        let metadata = (
            candidate.epoch(), candidate.next_call_id(), candidate.next_position(), candidate.valid_positions()
        );

        prop_assert!(candidate.forward(call(1, 1, 1, first, &[], owner_weights)).is_err());
        for batch in 0..batches {
            prop_assert_eq!(
                candidate.key_prefix(batch).expect("unchanged key prefix"),
                key_before[batch].as_slice(),
            );
            prop_assert_eq!(
                candidate.kv_prefix(batch).expect("unchanged KV prefix"),
                kv_before[batch].as_slice(),
            );
        }
        prop_assert_eq!(
            (candidate.epoch(), candidate.next_call_id(), candidate.next_position(), candidate.valid_positions()),
            metadata,
        );
        let retried = candidate.forward(call(1, 1, 1, first, &frequency, owner_weights))
            .expect("same ID retry after invalid key frequency");
        let baseline = clean.forward(call(1, 1, 1, first, &frequency, owner_weights))
            .expect("clean continuation");
        prop_assert_eq!(retried, baseline);
        for batch in 0..batches {
            prop_assert_eq!(
                candidate.key_prefix(batch).expect("retried key prefix"),
                clean.key_prefix(batch).expect("clean key prefix"),
            );
            prop_assert_eq!(
                candidate.kv_prefix(batch).expect("retried KV prefix"),
                clean.kv_prefix(batch).expect("clean KV prefix"),
            );
        }
        prop_assert_eq!(candidate.next_call_id(), 2);
        prop_assert_eq!(candidate.next_position(), 2);
        prop_assert_eq!(candidate.valid_positions(), 2);
    }
}
