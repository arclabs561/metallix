use deepseek::checkpoint::mlx::{
    collapse_hc_hidden, decode_affine_row, expand_hc_hidden, mix_hc_coefficients,
};
use proptest::prelude::*;

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    #[test]
    fn expansion_preserves_each_copy(
        input in prop::collection::vec(-100.0_f32..100.0, 1..32),
        copies in 1_usize..=8,
    ) {
        let expanded = expand_hc_hidden(&input, copies).expect("valid expansion");
        prop_assert_eq!(expanded.len(), input.len() * copies);
        for chunk in expanded.chunks_exact(input.len()) {
            prop_assert_eq!(chunk, input.as_slice());
        }
    }

    #[test]
    fn coefficient_split_and_collapse_preserve_finite_shape(
        hidden in prop::collection::vec(-10.0_f32..10.0, 1..16),
        copies in 1_usize..=8,
        sinkhorn_iterations in 1_usize..=8,
    ) {
        let rows = (2 + copies) * copies;
        let fn_matrix = vec![0.01_f32; rows * hidden.len() * copies];
        let base = vec![0.0_f32; rows];
        let coefficients = mix_hc_coefficients(
            &fn_matrix,
            &base,
            &[1.0, 1.0, 1.0],
            &hidden,
            copies,
            1e-6,
            sinkhorn_iterations,
        )
        .expect("bounded finite HC coefficients");
        let collapsed = collapse_hc_hidden(&hidden, &coefficients)
            .expect("bounded finite HC collapse");

        prop_assert_eq!(coefficients.copies(), copies);
        prop_assert_eq!(collapsed.len(), hidden.len());
        prop_assert!(coefficients.pre().iter().all(|value| value.is_finite()));
        prop_assert!(coefficients.post().iter().all(|value| value.is_finite()));
        prop_assert!(coefficients.comb().iter().all(|value| value.is_finite()));
        prop_assert!(collapsed.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn valid_eight_bit_rows_decode_to_finite_values(
        packed in prop::array::uniform2(any::<u32>()),
        scale_bits in prop::array::uniform2(0x3e00_u16..0x4100_u16),
        bias_bits in prop::array::uniform2(0x3e00_u16..0x4100_u16),
    ) {
        let decoded = decode_affine_row(
            &packed,
            &scale_bits,
            &bias_bits,
            8,
            8,
            4,
        )
        .expect("valid eight-bit row");
        prop_assert_eq!(decoded.len(), 8);
        prop_assert!(decoded.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn six_bit_rows_match_an_independent_bit_oracle(
        packed in prop::array::uniform24(any::<u32>()),
    ) {
        let decoded = decode_affine_row(
            &packed,
            &[0x3f80, 0x3f80],
            &[0, 0],
            128,
            6,
            64,
        )
        .expect("valid six-bit row");
        let bytes = packed
            .iter()
            .flat_map(|word| word.to_le_bytes())
            .collect::<Vec<_>>();
        for (column, actual) in decoded.iter().enumerate() {
            let bit_offset = column * 6;
            let expected = (0..6).fold(0_u8, |value, bit| {
                let absolute = bit_offset + bit;
                value
                    | (((bytes[absolute / 8] >> (absolute % 8)) & 1) << bit)
            });
            prop_assert_eq!(actual.to_bits(), f32::from(expected).to_bits());
        }
    }

    #[test]
    fn hc_projection_values_change_when_a_mix_row_changes(
        hidden in prop::collection::vec(1.0_f32..10.0, 1..8),
        copies in 1_usize..=4,
    ) {
        let rows = (2 + copies) * copies;
        let width = hidden.len() * copies;
        let base = vec![0.0_f32; rows];
        let zero = vec![0.0_f32; rows * width];
        let mut changed = zero.clone();
        changed[0] = 1.0;
        let baseline = mix_hc_coefficients(
            &zero,
            &base,
            &[1.0, 1.0, 1.0],
            &hidden,
            copies,
            1e-6,
            4,
        )
        .expect("baseline HC coefficients");
        let variant = mix_hc_coefficients(
            &changed,
            &base,
            &[1.0, 1.0, 1.0],
            &hidden,
            copies,
            1e-6,
            4,
        )
        .expect("changed HC coefficients");
        prop_assert_ne!(baseline.pre(), variant.pre());
    }
}
