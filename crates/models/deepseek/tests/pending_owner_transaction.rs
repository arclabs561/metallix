//! A staged owner publication stays private until an explicit commit.

use std::num::NonZeroUsize;

use deepseek::{
    RotaryFrequency,
    indexer::{
        cache::IndexKeyPublicationId,
        key::{IndexKeyLayout, IndexKeyWeights},
        owner::{RatioOneCompressedOwner, RatioOneOwnerCall, RatioOneOwnerWeights},
        query::IndexKeyView,
    },
};

const INPUT_DIMENSION: usize = 4;
const LATENT_DIMENSION: usize = 16;
const KEY_DIMENSION: usize = 32;

fn nz(value: usize) -> NonZeroUsize {
    NonZeroUsize::new(value).expect("positive test geometry")
}

fn owner() -> RatioOneCompressedOwner {
    RatioOneCompressedOwner::new(
        IndexKeyLayout::new(
            nz(1),
            nz(LATENT_DIMENSION),
            nz(KEY_DIMENSION),
            nz(1),
            1.0e-5,
        )
        .expect("small source key layout"),
        nz(INPUT_DIMENSION),
        nz(4),
        3,
        &[0x3f80; LATENT_DIMENSION],
        1.0e-5,
    )
    .expect("bounded coupled owner")
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

fn frequency() -> RotaryFrequency {
    RotaryFrequency::new(1.0, 0.0).expect("finite identity frequency")
}

fn call<'a>(
    call_id: u64,
    start: usize,
    input: &'a [u16],
    frequencies: &'a [RotaryFrequency],
    weights: RatioOneOwnerWeights<'a>,
) -> RatioOneOwnerCall<'a> {
    RatioOneOwnerCall::new(
        IndexKeyPublicationId::new(3, 0, call_id),
        start,
        nz(1),
        input,
        frequencies,
        weights,
    )
}

#[test]
fn discarded_pending_publication_leaves_owner_retry_equivalent() {
    let (wkv, wk, key_norm) = weights();
    let owner_weights = RatioOneOwnerWeights::new(&wkv, IndexKeyWeights::new(&wk, &key_norm));
    let frequencies = [frequency()];
    let initial = [0x3f80, 0x4000, 0x4040, 0x4080];
    let continuation = [0x4080, 0x4040, 0x4000, 0x3f80];
    let mut candidate = owner();

    candidate
        .forward(call(0, 0, &initial, &frequencies, owner_weights))
        .expect("initial publication");
    let before_key = candidate.key_prefix(0).expect("live key prefix").to_vec();
    let before_kv = candidate.kv_prefix(0).expect("live KV prefix").to_vec();
    let before_metadata = (
        candidate.epoch(),
        candidate.next_call_id(),
        candidate.next_position(),
        candidate.valid_positions(),
    );
    let mut clean = candidate.clone();

    let pending = candidate
        .prepare(call(1, 1, &continuation, &frequencies, owner_weights))
        .expect("staged continuation");
    let staged_keys = pending.key_prefix(0).expect("complete staged key prefix");
    let _staged_view = IndexKeyView::new(staged_keys, nz(KEY_DIMENSION))
        .expect("staged keys are ready for scoring");
    assert_eq!(staged_keys.len(), before_key.len() + KEY_DIMENSION);
    assert_eq!(&staged_keys[..before_key.len()], before_key);
    let staged_kv = pending.kv_prefix(0).expect("complete staged KV prefix");
    assert_eq!(staged_kv.len(), before_kv.len() + LATENT_DIMENSION);
    assert_eq!(&staged_kv[..before_kv.len()], before_kv);
    assert_eq!(pending.publication(), IndexKeyPublicationId::new(3, 0, 1));
    drop(pending);

    assert_eq!(
        candidate.key_prefix(0).expect("unchanged key prefix"),
        before_key
    );
    assert_eq!(
        candidate.kv_prefix(0).expect("unchanged KV prefix"),
        before_kv
    );
    assert_eq!(
        (
            candidate.epoch(),
            candidate.next_call_id(),
            candidate.next_position(),
            candidate.valid_positions(),
        ),
        before_metadata,
    );

    let retried = candidate
        .forward(call(1, 1, &continuation, &frequencies, owner_weights))
        .expect("same-ID retry after discard");
    let baseline = clean
        .forward(call(1, 1, &continuation, &frequencies, owner_weights))
        .expect("clean continuation");
    assert_eq!(retried, baseline);
    assert_eq!(
        candidate.key_prefix(0).expect("retried key prefix"),
        clean.key_prefix(0).expect("clean key prefix")
    );
    assert_eq!(
        candidate.kv_prefix(0).expect("retried KV prefix"),
        clean.kv_prefix(0).expect("clean KV prefix")
    );
}

#[test]
fn pending_commit_publishes_exactly_the_values_scored_before_commit() {
    let (wkv, wk, key_norm) = weights();
    let owner_weights = RatioOneOwnerWeights::new(&wkv, IndexKeyWeights::new(&wk, &key_norm));
    let frequencies = [frequency()];
    let input = [0x3f80, 0x4000, 0x4040, 0x4080];
    let mut owner = owner();

    let pending = owner
        .prepare(call(0, 0, &input, &frequencies, owner_weights))
        .expect("staged first publication");
    let staged = pending.diagnostic().clone();
    let committed = pending.commit().expect("atomic publication");

    assert_eq!(committed, staged);
    assert_eq!(
        owner.key_prefix(0).expect("published key prefix"),
        staged.owner.keys.post_fp4
    );
    assert_eq!(
        owner.kv_prefix(0).expect("published KV prefix"),
        staged.compressed_kv.post_fp4
    );
    assert_eq!(
        (
            owner.next_call_id(),
            owner.next_position(),
            owner.valid_positions()
        ),
        (1, 1, 1)
    );
}
