//! Chunking must not change the batch-major sequence of prepared keys.

use std::num::NonZeroUsize;

use deepseek::indexer::cache::{IndexKeyPublicationId, IndexKeyState};
use proptest::prelude::*;

fn nz(value: usize) -> NonZeroUsize {
    NonZeroUsize::new(value).expect("positive test geometry")
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(48))]
    #[test]
    fn chunked_appends_preserve_each_batch(
        batches in 1_usize..5, width in 1_usize..9,
        first in 1_usize..5, second in 1_usize..5,
        salt in 0_u16..256,
    ) {
        let positions = first + second;
        // Distinct finite BF16 payloads expose swaps across batch/group boundaries.
        let keys: Vec<u16> = (0..batches * positions * width)
            .map(|i| 0x3000 + u16::try_from(i).expect("small geometry") + salt)
            .collect();
        let chunk = |start: usize, count: usize| -> Vec<u16> {
            (0..batches).flat_map(|batch| {
                let begin = (batch * positions + start) * width;
                keys[begin..begin + count * width].iter().copied()
            }).collect()
        };
        let mut state = IndexKeyState::new(nz(batches), nz(width), nz(positions + 2), 3)
            .expect("bounded cache");
        state.append_prepared(IndexKeyPublicationId::new(3, 0, 0), 0, &chunk(0, first)).expect("first chunk");
        for batch in 0..batches {
            prop_assert_eq!(state.prefix(batch).expect("batch"),
                &keys[batch * positions * width..(batch * positions + first) * width]);
        }
        // An incomplete compressor group consumes a call but publishes no keys.
        state.append_prepared(IndexKeyPublicationId::new(3, 0, 1), first, &[]).expect("empty completion");
        state.append_prepared(IndexKeyPublicationId::new(3, 0, 2), first, &chunk(first, second)).expect("second chunk");
        for batch in 0..batches {
            prop_assert_eq!(state.prefix(batch).expect("batch"),
                &keys[batch * positions * width..(batch + 1) * positions * width]);
        }
        // Invalid late-batch data must not partially overwrite earlier batches.
        let mut bad = vec![0x3f80; batches * width];
        *bad.last_mut().expect("nonempty") = 0x7f80;
        prop_assert!(state.append_prepared(IndexKeyPublicationId::new(3, 0, 3), positions, &bad).is_err());
        for batch in 0..batches {
            prop_assert_eq!(state.prefix(batch).expect("batch"),
                &keys[batch * positions * width..(batch + 1) * positions * width]);
        }
        state.append_prepared(IndexKeyPublicationId::new(3, 0, 3), positions, &vec![0x3f80; batches * width])
            .expect("same call retries after failure");
        state.reset().expect("fresh epoch");
        for batch in 0..batches {
            prop_assert!(state.prefix(batch).expect("batch").is_empty());
        }
        prop_assert!(state.append_prepared(IndexKeyPublicationId::new(3, 0, 0), 0, &keys).is_err());
        state.append_prepared(IndexKeyPublicationId::new(3, 1, 0), 0, &keys).expect("new epoch");
        for batch in 0..batches {
            prop_assert_eq!(state.prefix(batch).expect("batch"),
                &keys[batch * positions * width..(batch + 1) * positions * width]);
        }
    }
}
