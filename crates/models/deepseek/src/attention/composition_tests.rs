//! Tests of the pinned query-RoPE → attention → output-conjugation ordering.
//! KV is already prepared; these tests do not construct or mutate a KV cache.

use std::num::NonZeroUsize;

use super::{SparseAttentionLayout, sparse_attention_metal_f32, sparse_attention_reference};
use crate::{
    GPU_TEST_LOCK, RotaryDirection, RotaryFrequency, RotaryTailLayout, rotate_tail,
    rotate_tail_metal,
};

#[derive(Clone, Copy)]
enum Backend {
    Cpu,
    Metal,
}

struct Case {
    query: Vec<f32>,
    prepared_kv: Vec<f32>,
    sink: Vec<f32>,
    indices: Vec<i32>,
    frequencies: Vec<RotaryFrequency>,
    layout: SparseAttentionLayout,
    tail_pairs: usize,
    scale: f32,
}

struct Stages {
    query_rotated: Vec<f32>,
    attention: Vec<f32>,
    output: Vec<f32>,
}

impl Case {
    fn run(&self, backend: Backend) -> Stages {
        let query_rotated = self.rotate(&self.query, RotaryDirection::Forward, backend);
        let attention = match backend {
            Backend::Cpu => sparse_attention_reference(
                &query_rotated,
                &self.prepared_kv,
                &self.sink,
                &self.indices,
                self.scale,
                self.layout,
            )
            .expect("bounded CPU attention"),
            Backend::Metal => sparse_attention_metal_f32(
                &query_rotated,
                &self.prepared_kv,
                &self.sink,
                &self.indices,
                self.scale,
                self.layout,
                10_000,
            )
            .expect("bounded Metal attention"),
        };
        let output = self.rotate(&attention, RotaryDirection::Inverse, backend);
        Stages {
            query_rotated,
            attention,
            output,
        }
    }

    fn rotate(&self, values: &[f32], direction: RotaryDirection, backend: Backend) -> Vec<f32> {
        let dimensions = self.layout.dimensions.get();
        let tail_width = self.tail_pairs * 2;
        assert!(tail_width <= dimensions);
        assert_eq!(values.len(), self.layout.query_len().unwrap());
        let prefix_width = dimensions - tail_width;
        let tail: Vec<_> = values
            .chunks_exact(dimensions)
            .flat_map(|row| row[prefix_width..].iter().copied())
            .collect();
        let layout = RotaryTailLayout::new(
            self.layout.batches,
            self.layout.query_positions,
            self.layout.heads,
            nz(self.tail_pairs),
        )
        .unwrap();
        let rotated = match backend {
            Backend::Cpu => {
                let mut rotated = tail.clone();
                rotate_tail(&mut rotated, layout, &self.frequencies, direction).unwrap();
                rotated
            }
            Backend::Metal => {
                rotate_tail_metal(&tail, layout, &self.frequencies, direction).unwrap()
            }
        };
        assert_eq!(rotated.len(), tail.len());
        let mut result = values.to_vec();
        for (row, tail) in result
            .chunks_exact_mut(dimensions)
            .zip(rotated.chunks_exact(tail_width))
        {
            row[prefix_width..].copy_from_slice(tail);
        }
        result
    }

    fn assert_prefix_unchanged(&self, before: &[f32], after: &[f32]) {
        let dimensions = self.layout.dimensions.get();
        let prefix = dimensions - self.tail_pairs * 2;
        assert_eq!(before.len(), after.len());
        for (left, right) in before
            .chunks_exact(dimensions)
            .zip(after.chunks_exact(dimensions))
        {
            for (left, right) in left[..prefix].iter().zip(&right[..prefix]) {
                assert_eq!(left.to_bits(), right.to_bits());
            }
        }
    }
}

fn nz(value: usize) -> NonZeroUsize {
    NonZeroUsize::new(value).unwrap()
}

fn assert_close(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        let tolerance = 1e-4 + 1e-5 * expected.abs();
        assert!(actual.is_finite() && expected.is_finite());
        assert!(
            (actual - expected).abs() <= tolerance,
            "scalar {index}: {actual} versus {expected}, tolerance {tolerance}"
        );
    }
}

#[test]
fn composition_matches_independent_two_position_hand_oracle() {
    let _guard = GPU_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let case = Case {
        query: vec![0.0, 1.0, 0.0, 0.0, 1.0, 0.0],
        prepared_kv: vec![3.0, 2.0, 0.0],
        sink: vec![0.0],
        indices: vec![0, 0],
        frequencies: vec![
            RotaryFrequency::new(1.0, 0.0).unwrap(),
            RotaryFrequency::new(0.0, 1.0).unwrap(),
        ],
        layout: SparseAttentionLayout::new(nz(1), nz(2), nz(1), nz(3), nz(1), nz(1)).unwrap(),
        tail_pairs: 1,
        scale: 1.0,
    };
    // Position 0: score=2, sink=0, weight=sigmoid(2), no rotation.
    // Position 1: q*(i)=[0,0,1], score=0, weight=1/2, then tail*(-i).
    // These values are derived independently of both composition backends.
    let weight = 1.0 / (1.0 + (-2.0_f32).exp());
    let expected = [3.0 * weight, 2.0 * weight, 0.0, 1.5, 0.0, -1.0];
    for backend in [Backend::Cpu, Backend::Metal] {
        let stages = case.run(backend);
        assert_close(&stages.output, &expected);
        assert_close(&stages.query_rotated, &[0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
        case.assert_prefix_unchanged(&case.query, &stages.query_rotated);
        case.assert_prefix_unchanged(&stages.attention, &stages.output);
    }
}

#[test]
fn composition_matches_cpu_across_batches_heads_pairs_and_sparse_masks() {
    let _guard = GPU_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let case = Case {
        query: (0_i16..72)
            .map(|value| f32::from(value % 11 - 5) * 0.1)
            .collect(),
        prepared_kv: (0_i16..48)
            .map(|value| f32::from(value % 13 - 6) * 0.125)
            .collect(),
        sink: vec![0.2, -0.7],
        indices: vec![
            0, 2, 2, -1, 3, 1, -1, -1, -1, -1, -1, -1, 2, 1, 0, -1, 0, 0, 3, -1, 1, 3, 2, 0,
        ],
        frequencies: [
            (1.0, 0.0),
            (0.0, 1.0),
            (0.6, 0.8),
            (0.0, -1.0),
            (-1.0, 0.0),
            (0.8, -0.6),
        ]
        .into_iter()
        .map(|(real, imaginary)| RotaryFrequency::new(real, imaginary).unwrap())
        .collect(),
        layout: SparseAttentionLayout::new(nz(2), nz(3), nz(2), nz(6), nz(4), nz(4)).unwrap(),
        tail_pairs: 2,
        scale: 0.5,
    };
    let cpu = case.run(Backend::Cpu);
    let metal = case.run(Backend::Metal);
    assert_close(&metal.query_rotated, &cpu.query_rotated);
    assert_close(&metal.attention, &cpu.attention);
    assert_close(&metal.output, &cpu.output);
    case.assert_prefix_unchanged(&case.query, &metal.query_rotated);
    case.assert_prefix_unchanged(&metal.attention, &metal.output);
    // Batch zero / query two is entirely masked, for both heads and all dimensions.
    assert!(metal.output[24..36].iter().all(|value| *value == 0.0));
}
