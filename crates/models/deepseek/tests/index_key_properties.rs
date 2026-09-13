//! Key preparation must preserve batch-major rows and call-local frequencies.

use std::num::NonZeroUsize;

use deepseek::{
    RotaryFrequency,
    indexer::key::{IndexKeyLayout, IndexKeyWeights, prepare_index_keys},
};
use proptest::prelude::*;

#[test]
fn late_rotary_narrowing_overflow_reports_full_key_element() {
    use deepseek::indexer::key::IndexKeyError;

    let latent = [0x3f80_u16; 2];
    let original = latent;
    let result = prepare_index_keys(
        &latent,
        &[
            RotaryFrequency::new(1.0, 0.0).expect("identity"),
            RotaryFrequency::new(1.003_906_3, 0.0).expect("finite scale"),
        ],
        IndexKeyWeights::new(&[0x3f80; 32], &[0x7f7f; 32]),
        IndexKeyLayout::new(nz(1), nz(1), nz(32), nz(1), 1e-20).expect("bounded layout"),
    );
    // The first row stays finite; the second row's first rotary component is
    // flat element 32 + 30. Its FP32 product is finite but BF16 rounding overflows.
    assert!(
        matches!(result, Err(IndexKeyError::NonFiniteRotary { element: 62 })),
        "{result:?}"
    );
    assert_eq!(latent, original);
}

#[test]
fn key_preparation_rejects_bad_boundaries_and_excessive_scalar_work() {
    use deepseek::indexer::key::IndexKeyError;

    let layout = IndexKeyLayout::new(nz(1), nz(16), nz(32), nz(2), 1e-6).expect("layout");
    let wk = [0_u16; 32 * 16];
    let norm = [0x3f80_u16; 32];
    let frequency = RotaryFrequency::new(1.0, 0.0).expect("identity rotation");
    assert!(matches!(
        prepare_index_keys(
            &[0; 16],
            &[frequency; 2],
            IndexKeyWeights::new(&wk[..511], &norm),
            layout
        ),
        Err(IndexKeyError::WeightLength {
            field: "wk",
            actual: 511,
            expected: 512
        })
    ));
    assert!(matches!(
        prepare_index_keys(
            &[0; 16],
            &[frequency],
            IndexKeyWeights::new(&wk, &norm),
            layout
        ),
        Err(IndexKeyError::FrequencyLength {
            actual: 1,
            expected: 2
        })
    ));
    let mut latent = [0_u16; 16];
    latent[15] = 0x7f80;
    assert!(matches!(
        prepare_index_keys(
            &latent,
            &[frequency; 2],
            IndexKeyWeights::new(&wk, &norm),
            layout
        ),
        Err(IndexKeyError::NonFiniteInput {
            field: "latent",
            position: 15
        })
    ));
    let layout =
        IndexKeyLayout::new(nz(1), nz(1024), nz(512), nz(1), 1e-6).expect("bounded buffers");
    // Every buffer fits its element limit; their product still exceeds the
    // scalar dot-work cap. Reject before attempting the expensive projection.
    assert!(matches!(
        prepare_index_keys(
            &vec![0; 33 * 1024],
            &[frequency; 33],
            IndexKeyWeights::new(&vec![0; 512 * 1024], &[0x3f80; 512]),
            layout,
        ),
        Err(IndexKeyError::WorkloadTooLarge { terms: 17_301_504 })
    ));
}

fn nz(value: usize) -> NonZeroUsize {
    NonZeroUsize::new(value).expect("nonzero test geometry")
}

fn small_bf16(value: i16) -> u16 {
    u16::try_from(f32::from(value).to_bits() >> 16).expect("exact small BF16 integer")
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(32))]

    #[test]
    fn batched_keys_match_independent_rows_without_mutating_latent(
        batches in 1_usize..=3,
        groups in 1_usize..=3,
        values in prop::collection::vec(-8_i16..=8, 9 * 16),
    ) {
        let latent: Vec<_> = values[..batches * groups * 16]
            .iter().copied().map(small_bf16).collect();
        let original = latent.clone();
        // Different rows mix different latent columns; this is not an identity
        // projection or a constant-vector normalization fixture.
        let mut wk = vec![0_u16; 32 * 16];
        for (row, weights) in wk.chunks_exact_mut(16).enumerate() {
            weights[row % 16] = small_bf16(1);
            weights[(row + 3) % 16] = small_bf16(-2);
        }
        let norm: Vec<_> = (0..32).map(|i| small_bf16(if i % 2 == 0 { 3 } else { -2 })).collect();
        let frequencies: Vec<_> = (0..groups)
            .flat_map(|position| {
                let (real, imaginary) = match position {
                    0 => (0.6, 0.8),
                    1 => (0.8, -0.6),
                    _ => (-0.6, 0.8),
                };
                [
                    RotaryFrequency::new(real, imaginary).expect("finite rotation"),
                    RotaryFrequency::new(imaginary, real).expect("finite rotation"),
                ]
            }).collect();
        let weights = IndexKeyWeights::new(&wk, &norm);
        let batched = prepare_index_keys(
            &latent, &frequencies, weights,
            IndexKeyLayout::new(nz(batches), nz(16), nz(32), nz(2), 1e-6).expect("layout"),
        ).expect("bounded batched preparation");
        for batch in 0..batches {
            for group in 0..groups {
                let row = batch * groups + group;
                let independent = prepare_index_keys(
                    &latent[row * 16..(row + 1) * 16],
                    &frequencies[group * 2..(group + 1) * 2],
                    weights,
                    IndexKeyLayout::new(nz(1), nz(16), nz(32), nz(2), 1e-6).expect("single row"),
                ).expect("independent key row");
                prop_assert_eq!(&batched.projected[row * 32..(row + 1) * 32], independent.projected);
                prop_assert_eq!(&batched.normalized[row * 32..(row + 1) * 32], independent.normalized);
                prop_assert_eq!(&batched.post_rope[row * 32..(row + 1) * 32], independent.post_rope);
                prop_assert_eq!(&batched.post_fp4[row * 32..(row + 1) * 32], independent.post_fp4);
            }
        }
        prop_assert_eq!(latent, original);
    }
}
