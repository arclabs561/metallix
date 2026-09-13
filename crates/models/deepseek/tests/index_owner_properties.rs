//! A rejected owner call must be indistinguishable from never attempting it.

use std::num::NonZeroUsize;

use deepseek::{
    RotaryFrequency,
    indexer::{
        cache::{IndexKeyPublicationId, IndexKeyStateError},
        key::{IndexKeyLayout, IndexKeyWeights},
        owner::{
            RatioOneIndexKeyOwner, RatioOneIndexKeyOwnerError, RatioOneOwnerCall,
            RatioOneOwnerWeights,
        },
    },
};
use proptest::prelude::*;

fn nz(value: usize) -> NonZeroUsize {
    NonZeroUsize::new(value).expect("positive test geometry")
}

#[test]
fn capacity_failure_preserves_live_stream_for_same_id_retry() {
    let layout = IndexKeyLayout::new(nz(1), nz(1), nz(32), nz(1), 1e-5).expect("layout");
    let mut owner =
        RatioOneIndexKeyOwner::new(layout, nz(1), nz(2), 3, &[0x3f80], 1e-5).expect("owner");
    let frequencies = [RotaryFrequency::new(1.0, 0.0).expect("identity"); 2];
    let weights = RatioOneOwnerWeights::new(
        &[0x3f80],
        IndexKeyWeights::new(&[0x3f80; 32], &[0x3f80; 32]),
    );
    let input = [0x3f80, 0x4000];
    owner
        .forward(RatioOneOwnerCall::new(
            IndexKeyPublicationId::new(3, 0, 0),
            0,
            nz(1),
            &input[..1],
            &frequencies[..1],
            weights,
        ))
        .expect("initial prefill");
    let prefix = owner.prefix(0).expect("live prefix").to_vec();
    let id = IndexKeyPublicationId::new(3, 0, 1);
    assert!(matches!(
        owner.forward(RatioOneOwnerCall::new(
            id,
            1,
            nz(2),
            &input,
            &frequencies,
            weights
        )),
        Err(RatioOneIndexKeyOwnerError::Cache(
            IndexKeyStateError::CapacityExceeded { .. }
        ))
    ));
    assert_eq!(owner.prefix(0).expect("unchanged prefix"), prefix);
    assert_eq!(owner.next_position(), 1);
    assert_eq!(owner.next_call_id(), 1);
    owner
        .forward(RatioOneOwnerCall::new(
            id,
            1,
            nz(1),
            &input[..1],
            &frequencies[..1],
            weights,
        ))
        .expect("same-ID shorter continuation retries");
    assert_eq!(owner.next_position(), 2);
    assert_eq!(owner.next_call_id(), 2);
    assert_eq!(&owner.prefix(0).expect("extended prefix")[..32], prefix);
}

#[test]
fn projection_work_guard_precedes_linear_execution() {
    let layout = IndexKeyLayout::new(nz(1), nz(257), nz(32), nz(1), 1e-5).expect("layout");
    let mut owner = RatioOneIndexKeyOwner::new(layout, nz(256), nz(257), 3, &[0x3f80; 257], 1e-5)
        .expect("bounded persistent state");
    let input = vec![0x3f80; 257 * 256];
    // Buffers fit the element cap, but projection exceeds 2^24 scalar terms.
    // Empty weights would produce a different error if linear execution began.
    assert!(matches!(
        owner.forward(RatioOneOwnerCall::new(
            IndexKeyPublicationId::new(3, 0, 0),
            0,
            nz(257),
            &input,
            &[],
            RatioOneOwnerWeights::new(&[], IndexKeyWeights::new(&[], &[])),
        )),
        Err(RatioOneIndexKeyOwnerError::ProjectionWorkloadTooLarge {
            terms: 16_908_544,
            maximum: 16_777_216,
        })
    ));
    assert_eq!(owner.next_position(), 0);
    assert_eq!(owner.next_call_id(), 0);
    assert!(owner.prefix(0).expect("empty prefix").is_empty());
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(24))]
    #[test]
    fn late_failure_then_retry_matches_clean_run(
        batches in 1_usize..4, positions in 1_usize..5,
        payload in prop::collection::vec(0x3c00_u16..0x4000, 48),
    ) {
        let layout = IndexKeyLayout::new(nz(batches), nz(4), nz(32), nz(1), 1e-5)
            .expect("small key layout");
        let make_owner = || RatioOneIndexKeyOwner::new(layout, nz(4), nz(8), 3, &[0x3f80; 4], 1e-5)
            .expect("bounded owner");
        let mut candidate = make_owner();
        let mut clean = make_owner();
        let projection = [0x3f80, 0, 0, 0, 0, 0x3f80, 0, 0, 0, 0, 0x3f80, 0, 0, 0, 0, 0x3f80];
        let wk: Vec<u16> = (0..128).map(|i| if i % 5 == 0 {0x3f80} else {0x3e80}).collect();
        let weights = RatioOneOwnerWeights::new(&projection, IndexKeyWeights::new(&wk, &[0x3f80; 32]));
        let frequencies = vec![RotaryFrequency::new(1.0, 0.0).expect("identity"); positions];
        let input = &payload[..batches * positions * 4];
        let id = IndexKeyPublicationId::new(3, 0, 0);
        prop_assert!(candidate.forward(RatioOneOwnerCall::new(id, 0, nz(positions), input, &[], weights)).is_err());
        prop_assert_eq!(candidate.next_position(), 0);
        prop_assert_eq!(candidate.next_call_id(), 0);
        for batch in 0..batches {
            prop_assert!(candidate.prefix(batch).expect("batch").is_empty());
        }
        let call = || RatioOneOwnerCall::new(id, 0, nz(positions), input, &frequencies, weights);
        let retried = candidate.forward(call()).expect("retry");
        let baseline = clean.forward(call()).expect("clean call");
        prop_assert_eq!(retried.projected, baseline.projected);
        prop_assert_eq!(retried.latent, baseline.latent);
        prop_assert_eq!(retried.keys, baseline.keys);
        for batch in 0..batches {
            prop_assert_eq!(candidate.prefix(batch).expect("retry prefix"), clean.prefix(batch).expect("clean prefix"));
        }
        prop_assert_eq!(candidate.next_position(), clean.next_position());
        prop_assert_eq!(candidate.next_call_id(), clean.next_call_id());
        prop_assert_eq!(candidate.epoch(), clean.epoch());
    }
}
