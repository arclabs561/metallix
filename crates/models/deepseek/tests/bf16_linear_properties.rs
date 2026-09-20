//! Algebraic and transactional properties of the public BF16 linear reference.

use deepseek::precision::{Bf16LinearError, bf16_linear_reference};
use proptest::prelude::*;

fn small_bf16(index: u8) -> u16 {
    [0xc000, 0xbf80, 0x0000, 0x3f80, 0x4000][usize::from(index)]
}

fn matrix_case() -> impl Strategy<Value = (usize, usize, usize, Vec<u16>, Vec<u16>)> {
    (1_usize..5, 1_usize..5, 1_usize..5)
        .prop_flat_map(|(rows, reduction, outputs)| {
            (
                Just(rows),
                Just(reduction),
                Just(outputs),
                prop::collection::vec(0_u8..5, rows * reduction),
                prop::collection::vec(0_u8..5, outputs * reduction),
            )
        })
        .prop_map(|(rows, reduction, outputs, activations, weights)| {
            (
                rows,
                reduction,
                outputs,
                activations.into_iter().map(small_bf16).collect(),
                weights.into_iter().map(small_bf16).collect(),
            )
        })
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(128))]

    #[test]
    fn concatenated_activation_rows_equal_split_calls(
        (rows, reduction, outputs, activations, weights) in matrix_case(),
    ) {
        let mut whole = vec![0; rows * outputs];
        bf16_linear_reference(&activations, &weights, rows, reduction, outputs, &mut whole).unwrap();
        let mut split = Vec::new();
        for row in activations.chunks_exact(reduction) {
            let mut one = vec![0; outputs];
            bf16_linear_reference(row, &weights, 1, reduction, outputs, &mut one).unwrap();
            split.extend(one);
        }
        prop_assert_eq!(whole, split);
    }

    #[test]
    fn permuting_weight_rows_only_permutes_output_columns(
        (rows, reduction, outputs, activations, weights) in matrix_case(),
    ) {
        let mut baseline = vec![0; rows * outputs];
        bf16_linear_reference(&activations, &weights, rows, reduction, outputs, &mut baseline).unwrap();
        let permutation: Vec<_> = (0..outputs).rev().collect();
        let mut permuted_weights = Vec::new();
        for &column in &permutation {
            permuted_weights.extend_from_slice(&weights[column * reduction..(column + 1) * reduction]);
        }
        let mut observed = vec![0; rows * outputs];
        bf16_linear_reference(&activations, &permuted_weights, rows, reduction, outputs, &mut observed).unwrap();
        for row in 0..rows {
            for (new_column, &old_column) in permutation.iter().enumerate() {
                prop_assert_eq!(observed[row * outputs + new_column], baseline[row * outputs + old_column]);
            }
        }
    }

    #[test]
    fn late_product_overflow_is_transactional_and_identifies_its_coordinate(
        rows in 1_usize..5,
        reduction in 1_usize..5,
        outputs in 1_usize..5,
    ) {
        let mut activations = vec![0x3f80; rows * reduction];
        let mut weights = vec![0x3f80; outputs * reduction];
        activations[(rows - 1) * reduction + (reduction - 1)] = 0x7f7f;
        weights[(outputs - 1) * reduction + (reduction - 1)] = 0x7f7f;
        let mut output = vec![0xdead; rows * outputs];
        prop_assert_eq!(
            bf16_linear_reference(&activations, &weights, rows, reduction, outputs, &mut output),
            Err(Bf16LinearError::ValueOverflow { stage: "product", row: rows - 1, output: outputs - 1 }),
        );
        prop_assert_eq!(output, vec![0xdead; rows * outputs]);
    }
}
