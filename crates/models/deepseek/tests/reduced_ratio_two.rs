//! Runtime ratio-two compressed-owner contract without source fixtures.

use std::num::NonZeroUsize;

use deepseek::{
    RotaryFrequency,
    indexer::{cache::IndexKeyPublicationId, key::IndexKeyWeights},
    reduced::{
        RatioTwoCompressedOwner, RatioTwoOwnerCall, RatioTwoOwnerError, RatioTwoOwnerLayout,
        RatioTwoOwnerWeights,
    },
};

const ONE: u16 = 0x3f80;
const ZERO: u16 = 0;

fn nz(value: usize) -> NonZeroUsize {
    NonZeroUsize::new(value).unwrap()
}

fn layout() -> RatioTwoOwnerLayout {
    RatioTwoOwnerLayout::new(nz(1), nz(2), nz(32), nz(32), nz(1), nz(3), 1.0e-6).unwrap()
}

struct Operands {
    wkv: Vec<f32>,
    wgate: Vec<f32>,
    wk: Vec<u16>,
    key_norm: Vec<u16>,
    compressor_norm: Vec<u16>,
}

fn operands() -> Operands {
    let mut wkv = vec![0.0; 32 * 2];
    for row in 0..32 {
        wkv[row * 2] = 1.0;
    }
    let wgate = vec![0.0; 32 * 2];
    let mut wk = vec![ZERO; 32 * 32];
    for feature in 0..32 {
        wk[feature * 32 + feature] = ONE;
    }
    Operands {
        wkv,
        wgate,
        wk,
        key_norm: vec![ONE; 32],
        compressor_norm: vec![ONE; 32],
    }
}

fn frequencies(count: usize) -> Vec<RotaryFrequency> {
    (0..count)
        .map(|_| RotaryFrequency::new(1.0, 0.0).unwrap())
        .collect()
}

fn weights<'a>(
    wkv: &'a [f32],
    wgate: &'a [f32],
    wk: &'a [u16],
    key_norm: &'a [u16],
) -> RatioTwoOwnerWeights<'a> {
    RatioTwoOwnerWeights::new(wkv, wgate, IndexKeyWeights::new(wk, key_norm))
}

#[test]
fn alternate_four_one_one_one_schedule_preserves_paired_prefixes() {
    let Operands {
        wkv,
        wgate,
        wk,
        key_norm,
        compressor_norm,
    } = operands();
    let weights = weights(&wkv, &wgate, &wk, &key_norm);
    let mut owner = RatioTwoCompressedOwner::new(layout(), 1, &compressor_norm).unwrap();
    let first_frequencies = frequencies(2);
    let prefill_input = [ONE, ZERO, ONE, ZERO, ONE, ZERO, ONE, ZERO];
    let prefill = owner
        .forward(RatioTwoOwnerCall::new(
            IndexKeyPublicationId::new(1, 0, 0),
            0,
            nz(4),
            &prefill_input,
            &first_frequencies,
            weights,
        ))
        .unwrap();
    assert_eq!(prefill.projected(), &[1.0; 128]);
    assert_eq!(prefill.gate(), &[0.0; 128]);
    assert!(prefill.latent().is_some());
    assert_eq!(prefill.key_prefixes()[0].len(), 2 * 32);
    assert_eq!(prefill.kv_prefixes()[0].len(), 2 * 32);

    let prefix_after_prefill = prefill.key_prefixes()[0].clone();
    let partial_at_four = owner
        .forward(RatioTwoOwnerCall::new(
            IndexKeyPublicationId::new(1, 0, 1),
            4,
            nz(1),
            &[ONE, ZERO],
            &[],
            weights,
        ))
        .unwrap();
    assert!(partial_at_four.latent().is_none());
    assert_eq!(partial_at_four.key_prefixes()[0], prefix_after_prefill);

    let completion_frequencies = frequencies(1);
    let completion = owner
        .forward(RatioTwoOwnerCall::new(
            IndexKeyPublicationId::new(1, 0, 2),
            5,
            nz(1),
            &[ONE, ZERO],
            &completion_frequencies,
            weights,
        ))
        .unwrap();
    assert!(completion.latent().is_some());
    assert_eq!(completion.key_prefixes()[0].len(), 3 * 32);

    let final_partial = owner
        .forward(RatioTwoOwnerCall::new(
            IndexKeyPublicationId::new(1, 0, 3),
            6,
            nz(1),
            &[ONE, ZERO],
            &[],
            weights,
        ))
        .unwrap();
    assert!(final_partial.latent().is_none());
    assert_eq!(
        final_partial.key_prefixes()[0],
        completion.key_prefixes()[0]
    );
    assert_eq!(final_partial.kv_prefixes()[0], completion.kv_prefixes()[0]);
    assert_eq!(owner.next_call_id(), 4);
    assert_eq!(owner.next_position(), 7);
}

#[test]
fn rejected_late_frequency_validation_leaves_owner_retryable() {
    let Operands {
        wkv,
        wgate,
        wk,
        key_norm,
        compressor_norm,
    } = operands();
    let weights = weights(&wkv, &wgate, &wk, &key_norm);
    let input = [ONE, ZERO, ONE, ZERO, ONE, ZERO, ONE, ZERO];
    let frequencies = frequencies(2);
    let mut fresh = RatioTwoCompressedOwner::new(layout(), 1, &compressor_norm).unwrap();
    let expected = fresh
        .forward(RatioTwoOwnerCall::new(
            IndexKeyPublicationId::new(1, 0, 0),
            0,
            nz(4),
            &input,
            &frequencies,
            weights,
        ))
        .unwrap();

    let mut owner = RatioTwoCompressedOwner::new(layout(), 1, &compressor_norm).unwrap();
    assert!(
        owner
            .forward(RatioTwoOwnerCall::new(
                IndexKeyPublicationId::new(2, 0, 0),
                0,
                nz(4),
                &input,
                &frequencies,
                weights,
            ))
            .is_err()
    );
    assert_eq!(owner.next_call_id(), 0);
    assert_eq!(owner.next_position(), 0);
    let late_error = owner
        .forward(RatioTwoOwnerCall::new(
            IndexKeyPublicationId::new(1, 0, 0),
            0,
            nz(4),
            &input,
            &[],
            weights,
        ))
        .expect_err("completed group requires rotary frequencies");
    assert!(matches!(late_error, RatioTwoOwnerError::Key(_)));
    assert_eq!(owner.next_call_id(), 0);
    assert_eq!(owner.next_position(), 0);
    let retried = owner
        .forward(RatioTwoOwnerCall::new(
            IndexKeyPublicationId::new(1, 0, 0),
            0,
            nz(4),
            &input,
            &frequencies,
            weights,
        ))
        .unwrap();
    assert_eq!(retried, expected);
}

#[test]
fn reset_requires_a_new_epoch_publication() {
    let Operands {
        wkv,
        wgate,
        wk,
        key_norm,
        compressor_norm,
    } = operands();
    let weights = weights(&wkv, &wgate, &wk, &key_norm);
    let frequencies = frequencies(2);
    let input = [ONE, ZERO, ONE, ZERO, ONE, ZERO, ONE, ZERO];
    let mut owner = RatioTwoCompressedOwner::new(layout(), 1, &compressor_norm).unwrap();
    owner
        .forward(RatioTwoOwnerCall::new(
            IndexKeyPublicationId::new(1, 0, 0),
            0,
            nz(4),
            &input,
            &frequencies,
            weights,
        ))
        .unwrap();
    assert!(
        owner
            .forward(RatioTwoOwnerCall::new(
                IndexKeyPublicationId::new(1, 0, 1),
                0,
                nz(4),
                &input,
                &frequencies,
                weights,
            ))
            .is_err()
    );
    owner.reset().unwrap();
    assert_eq!(owner.epoch(), 1);
    assert_eq!(owner.next_call_id(), 0);
    assert_eq!(owner.next_position(), 0);
    assert!(
        owner
            .forward(RatioTwoOwnerCall::new(
                IndexKeyPublicationId::new(1, 1, 0),
                0,
                nz(4),
                &input,
                &frequencies,
                weights,
            ))
            .is_ok()
    );
}
