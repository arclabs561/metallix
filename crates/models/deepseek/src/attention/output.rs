//! Bounded scalar V4.1 attention-output staging reference.
//!
//! This composes the pinned order after sparse attention: inverse rotary tail,
//! group-local BF16 `wo_a`, then activation quantization and FP8 `wo_b`. It
//! models explicit scalar precision boundaries, not a GPU GEMM, checkpoint
//! reader, cache, or complete attention layer.

use thiserror::Error;

use crate::{
    RotaryDirection, RotaryError, RotaryFrequency, RotaryTailLayout,
    precision::{
        ActivationGroup, ActivationQuantError, Bf16LinearError, Fp8LinearError,
        bf16_linear_reference, fp8_linear_runtime_f32, quantize_bf16_activations_e4m3fn,
    },
    rotate_tail,
};

const MAX_ATTENTION_OUTPUT_ELEMENTS: usize = 1 << 20;
const MAX_ATTENTION_OUTPUT_WORK: usize = 1 << 24;

/// Explicit layout for attention's inverse-RoPE and output projections.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AttentionOutputLayout {
    batches: usize,
    positions: usize,
    heads: usize,
    head_dimension: usize,
    rope_pairs: usize,
    groups: usize,
    output_rank: usize,
    hidden_dimension: usize,
}

impl AttentionOutputLayout {
    /// Validates a bounded source-shaped output-projection layout.
    ///
    /// `wo_a` is group-local over contiguous heads; `wo_b` receives the
    /// concatenated `groups * output_rank` BF16 vector and therefore requires
    /// that reduction width to be divisible by the pinned FP8 group size 32.
    #[allow(
        clippy::too_many_arguments,
        reason = "the public bounded layout keeps source-shaped dimensions explicit"
    )]
    pub fn new(
        batches: usize,
        positions: usize,
        heads: usize,
        head_dimension: usize,
        rope_pairs: usize,
        groups: usize,
        output_rank: usize,
        hidden_dimension: usize,
    ) -> Result<Self, AttentionOutputLayoutError> {
        if [
            batches,
            positions,
            heads,
            head_dimension,
            rope_pairs,
            groups,
            output_rank,
            hidden_dimension,
        ]
        .contains(&0)
        {
            return Err(AttentionOutputLayoutError::EmptyDimension);
        }
        if !heads.is_multiple_of(groups) {
            return Err(AttentionOutputLayoutError::HeadsNotGrouped { heads, groups });
        }
        let rope_width =
            rope_pairs
                .checked_mul(2)
                .ok_or(AttentionOutputLayoutError::ShapeOverflow {
                    field: "rope width",
                })?;
        if rope_width > head_dimension {
            return Err(AttentionOutputLayoutError::RopeExceedsHead {
                rope_width,
                head_dimension,
            });
        }
        let flattened_rank =
            groups
                .checked_mul(output_rank)
                .ok_or(AttentionOutputLayoutError::ShapeOverflow {
                    field: "flattened output rank",
                })?;
        if !flattened_rank.is_multiple_of(32) {
            return Err(AttentionOutputLayoutError::WoBReductionNotGrouped { flattened_rank });
        }
        let layout = Self {
            batches,
            positions,
            heads,
            head_dimension,
            rope_pairs,
            groups,
            output_rank,
            hidden_dimension,
        };
        for (field, elements) in [
            ("attention", layout.attention_elements()?),
            ("wo_a", layout.wo_a_elements()?),
            ("wo_a output", layout.wo_a_output_elements()?),
            ("wo_b", layout.wo_b_elements()?),
            ("wo_b scales", layout.wo_b_scale_elements()?),
            ("output", layout.output_elements()?),
        ] {
            if elements > MAX_ATTENTION_OUTPUT_ELEMENTS {
                return Err(AttentionOutputLayoutError::ElementLimit { field, elements });
            }
        }
        let work = layout.work()?;
        if work > MAX_ATTENTION_OUTPUT_WORK {
            return Err(AttentionOutputLayoutError::WorkLimit { elements: work });
        }
        Ok(layout)
    }

    fn rows(self) -> Result<usize, AttentionOutputLayoutError> {
        product(&[self.batches, self.positions], "rows")
    }

    fn heads_per_group(self) -> usize {
        self.heads / self.groups
    }

    fn group_reduction(self) -> Result<usize, AttentionOutputLayoutError> {
        self.heads_per_group()
            .checked_mul(self.head_dimension)
            .ok_or(AttentionOutputLayoutError::ShapeOverflow {
                field: "wo_a reduction",
            })
    }

    fn flattened_rank(self) -> usize {
        self.groups * self.output_rank
    }

    fn attention_elements(self) -> Result<usize, AttentionOutputLayoutError> {
        product(
            &[
                self.batches,
                self.positions,
                self.heads,
                self.head_dimension,
            ],
            "attention",
        )
    }

    fn tail_elements(self) -> Result<usize, AttentionOutputLayoutError> {
        product(
            &[self.batches, self.positions, self.heads, self.rope_pairs, 2],
            "rotary tail",
        )
    }

    fn frequency_elements(self) -> Result<usize, AttentionOutputLayoutError> {
        product(&[self.positions, self.rope_pairs], "rotary frequencies")
    }

    fn wo_a_elements(self) -> Result<usize, AttentionOutputLayoutError> {
        product(
            &[self.groups, self.output_rank, self.group_reduction()?],
            "wo_a",
        )
    }

    fn wo_b_elements(self) -> Result<usize, AttentionOutputLayoutError> {
        product(&[self.hidden_dimension, self.flattened_rank()], "wo_b")
    }

    fn wo_a_output_elements(self) -> Result<usize, AttentionOutputLayoutError> {
        product(&[self.rows()?, self.flattened_rank()], "wo_a output")
    }

    fn wo_b_scale_elements(self) -> Result<usize, AttentionOutputLayoutError> {
        product(
            &[
                self.hidden_dimension.div_ceil(32),
                self.flattened_rank() / 32,
            ],
            "wo_b scales",
        )
    }

    fn output_elements(self) -> Result<usize, AttentionOutputLayoutError> {
        product(&[self.rows()?, self.hidden_dimension], "output")
    }

    fn work(self) -> Result<usize, AttentionOutputLayoutError> {
        let wo_a = product(
            &[
                self.rows()?,
                self.groups,
                self.output_rank,
                self.group_reduction()?,
            ],
            "wo_a work",
        )?;
        let wo_b = product(
            &[self.rows()?, self.hidden_dimension, self.flattened_rank()],
            "wo_b work",
        )?;
        wo_a.checked_add(wo_b)
            .ok_or(AttentionOutputLayoutError::ShapeOverflow {
                field: "total work",
            })
    }
}

/// Invalid V4.1 attention-output projection layout.
#[derive(Clone, Copy, Debug, Error, Eq, PartialEq)]
#[non_exhaustive]
pub enum AttentionOutputLayoutError {
    /// Every explicit layout dimension must be nonzero.
    #[error("attention-output dimensions must all be nonzero")]
    EmptyDimension,
    /// Contiguous attention heads cannot be split evenly into output groups.
    #[error("attention-output heads {heads} are not divisible by groups {groups}")]
    HeadsNotGrouped { heads: usize, groups: usize },
    /// The requested inverse-RoPE tail is wider than one attention head.
    #[error("attention-output rotary width {rope_width} exceeds head width {head_dimension}")]
    RopeExceedsHead {
        rope_width: usize,
        head_dimension: usize,
    },
    /// The FP8 `wo_b` reduction is not compatible with fixed G32 activation groups.
    #[error("attention-output flattened rank {flattened_rank} is not divisible by 32")]
    WoBReductionNotGrouped { flattened_rank: usize },
    /// A derived shape could not be represented by `usize`.
    #[error("attention-output shape arithmetic overflowed for {field}")]
    ShapeOverflow { field: &'static str },
    /// A bounded scalar staging buffer would exceed the fixed element cap.
    #[error("attention-output {field} has {elements} elements, maximum is 1048576")]
    ElementLimit {
        field: &'static str,
        elements: usize,
    },
    /// The two scalar projection stages exceed the fixed work bound.
    #[error("attention-output work estimate {elements} exceeds maximum 16777216")]
    WorkLimit { elements: usize },
}

/// Errors from the bounded scalar attention-output staging reference.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum AttentionOutputError {
    /// The explicit output-projection layout was invalid.
    #[error(transparent)]
    Layout(#[from] AttentionOutputLayoutError),
    /// A direct-runtime input has an unexpected exact length.
    #[error("attention-output {field} length is {actual}, expected {expected}")]
    Length {
        field: &'static str,
        actual: usize,
        expected: usize,
    },
    /// An attention BF16 value is nonfinite before the inverse rotation.
    #[error("nonfinite attention BF16 value at element {element}")]
    NonFiniteAttention { element: usize },
    /// The inverse rotation or its BF16 narrowing was nonfinite.
    #[error("attention-output inverse rotary value was nonfinite at element {element}")]
    NonFiniteRotaryOutput { element: usize },
    /// The shared rotary-tail contract was invalid.
    #[error(transparent)]
    Rotary(#[from] RotaryError),
    /// Group-local BF16 output projection staging failed.
    #[error(transparent)]
    WoA(#[from] Bf16LinearError),
    /// FP8 activation preparation for `wo_b` failed.
    #[error(transparent)]
    WoBActivation(#[from] ActivationQuantError),
    /// FP8 `wo_b` scalar staging failed.
    #[error(transparent)]
    WoB(#[from] Fp8LinearError),
    /// The FP8 `wo_b` FP32 output could not narrow to finite BF16 storage.
    #[error("attention-output wo_b result was nonfinite at element {element}")]
    NonFiniteWoBOutput { element: usize },
}

/// Composes V4.1 inverse `RoPE`, grouped BF16 `wo_a`, and FP8 `wo_b`.
///
/// `attention` is BF16 `[batch, position, head, head_dimension]`; `wo_a` is
/// BF16 `[groups, output_rank, heads/groups * head_dimension]`; `wo_b_codes`
/// is E4M3FN `[hidden_dimension, groups * output_rank]`; and `wo_b_scales` is
/// E8M0 `[ceil(hidden_dimension / 32), groups * output_rank / 32]`.
/// Frequencies are `[position, rope_pair]`. The result is BF16
/// `[batch, position, hidden_dimension]`.
///
/// The reference follows the source's `apply_rotary_emb(..., inverse=True)`,
/// block-diagonal `einsum("bsgd,grd->bsgr")`, then FP8 `wo_b` order. Its
/// BF16 `wo_a` and final `wo_b` narrows are scalar precision contracts; this
/// does not claim `PyTorch`, CUDA, `TileLang`, or Metal GEMM reduction parity.
pub fn attention_output_reference(
    attention: &[u16],
    frequencies: &[RotaryFrequency],
    wo_a: &[u16],
    wo_b_codes: &[u8],
    wo_b_scales: &[u8],
    layout: AttentionOutputLayout,
) -> Result<Vec<u16>, AttentionOutputError> {
    validate_lengths(
        attention,
        frequencies,
        wo_a,
        wo_b_codes,
        wo_b_scales,
        layout,
    )?;
    for (element, &bits) in attention.iter().enumerate() {
        if !bf16_to_f32(bits).is_finite() {
            return Err(AttentionOutputError::NonFiniteAttention { element });
        }
    }

    let rotated = inverse_rotate_and_narrow(attention, frequencies, layout)?;
    let rows = layout.rows()?;
    let group_reduction = layout.group_reduction()?;
    let mut grouped_output = vec![0_u16; rows * layout.flattened_rank()];
    for group in 0..layout.groups {
        let mut one_group_input = vec![0_u16; rows * group_reduction];
        for row in 0..rows {
            let source =
                (row * layout.heads + group * layout.heads_per_group()) * layout.head_dimension;
            one_group_input[row * group_reduction..(row + 1) * group_reduction]
                .copy_from_slice(&rotated[source..source + group_reduction]);
        }
        let mut one_group_output = vec![0_u16; rows * layout.output_rank];
        let weight_start = group * layout.output_rank * group_reduction;
        bf16_linear_reference(
            &one_group_input,
            &wo_a[weight_start..weight_start + layout.output_rank * group_reduction],
            rows,
            group_reduction,
            layout.output_rank,
            &mut one_group_output,
        )?;
        for row in 0..rows {
            let destination = row * layout.flattened_rank() + group * layout.output_rank;
            grouped_output[destination..destination + layout.output_rank].copy_from_slice(
                &one_group_output[row * layout.output_rank..(row + 1) * layout.output_rank],
            );
        }
    }

    let flattened_rank = layout.flattened_rank();
    let mut activation_codes = vec![0_u8; grouped_output.len()];
    let mut activation_scales = vec![0_u8; rows * (flattened_rank / 32)];
    quantize_bf16_activations_e4m3fn(
        &grouped_output,
        rows,
        flattened_rank,
        ActivationGroup::Elements32,
        &mut activation_codes,
        &mut activation_scales,
    )?;
    let mut fp32_output = vec![0.0_f32; layout.output_elements()?];
    fp8_linear_runtime_f32(
        &activation_codes,
        &activation_scales,
        wo_b_codes,
        wo_b_scales,
        rows,
        flattened_rank,
        layout.hidden_dimension,
        ActivationGroup::Elements32,
        &mut fp32_output,
    )?;
    fp32_output
        .into_iter()
        .enumerate()
        .map(|(element, value)| {
            let bits = f32_to_bf16_rne(value);
            if bf16_to_f32(bits).is_finite() {
                Ok(bits)
            } else {
                Err(AttentionOutputError::NonFiniteWoBOutput { element })
            }
        })
        .collect()
}

fn validate_lengths(
    attention: &[u16],
    frequencies: &[RotaryFrequency],
    wo_a: &[u16],
    wo_b_codes: &[u8],
    wo_b_scales: &[u8],
    layout: AttentionOutputLayout,
) -> Result<(), AttentionOutputError> {
    for (field, actual, expected) in [
        ("attention", attention.len(), layout.attention_elements()?),
        (
            "frequencies",
            frequencies.len(),
            layout.frequency_elements()?,
        ),
        ("wo_a", wo_a.len(), layout.wo_a_elements()?),
        ("wo_b codes", wo_b_codes.len(), layout.wo_b_elements()?),
        (
            "wo_b scales",
            wo_b_scales.len(),
            layout.wo_b_scale_elements()?,
        ),
    ] {
        if actual != expected {
            return Err(AttentionOutputError::Length {
                field,
                actual,
                expected,
            });
        }
    }
    Ok(())
}

fn inverse_rotate_and_narrow(
    attention: &[u16],
    frequencies: &[RotaryFrequency],
    layout: AttentionOutputLayout,
) -> Result<Vec<u16>, AttentionOutputError> {
    let tail_width = layout.rope_pairs * 2;
    let prefix_width = layout.head_dimension - tail_width;
    let mut tail = Vec::with_capacity(layout.tail_elements()?);
    for head in attention.chunks_exact(layout.head_dimension) {
        tail.extend(head[prefix_width..].iter().map(|&bits| bf16_to_f32(bits)));
    }
    let rotary_layout = RotaryTailLayout::new(
        std::num::NonZeroUsize::new(layout.batches).expect("validated nonzero batches"),
        std::num::NonZeroUsize::new(layout.positions).expect("validated nonzero positions"),
        std::num::NonZeroUsize::new(layout.heads).expect("validated nonzero heads"),
        std::num::NonZeroUsize::new(layout.rope_pairs).expect("validated nonzero rope pairs"),
    )?;
    rotate_tail(
        &mut tail,
        rotary_layout,
        frequencies,
        RotaryDirection::Inverse,
    )?;
    let mut rotated = attention.to_vec();
    for (head, rotated_tail) in rotated
        .chunks_exact_mut(layout.head_dimension)
        .zip(tail.chunks_exact(tail_width))
    {
        for (offset, &value) in rotated_tail.iter().enumerate() {
            let bits = f32_to_bf16_rne(value);
            if !bf16_to_f32(bits).is_finite() {
                return Err(AttentionOutputError::NonFiniteRotaryOutput { element: offset });
            }
            head[prefix_width + offset] = bits;
        }
    }
    Ok(rotated)
}

fn product(values: &[usize], field: &'static str) -> Result<usize, AttentionOutputLayoutError> {
    values.iter().try_fold(1_usize, |total, &value| {
        total
            .checked_mul(value)
            .ok_or(AttentionOutputLayoutError::ShapeOverflow { field })
    })
}

fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}

fn f32_to_bf16_rne(value: f32) -> u16 {
    let bits = value.to_bits();
    let rounded = bits.wrapping_add(0x7fff + ((bits >> 16) & 1));
    u16::try_from(rounded >> 16).expect("an FP32 high half always fits BF16 storage")
}

#[cfg(test)]
mod tests {
    use serde::Deserialize;

    use super::{
        AttentionOutputLayout, AttentionOutputLayoutError, attention_output_reference,
        inverse_rotate_and_narrow,
    };
    use crate::{RotaryFrequency, precision::bf16_linear_reference};

    #[derive(Deserialize)]
    struct Fixture {
        schema_version: u8,
        source: FixtureSource,
        reference: FixtureReference,
        cases: Vec<FixtureCase>,
    }

    #[derive(Deserialize)]
    struct FixtureSource {
        revision: String,
        sha256: String,
        symbol: String,
    }

    #[derive(Deserialize)]
    struct FixtureReference {
        torch_version: String,
        device: String,
        input_dtype: String,
        output_dtype: String,
        einsum: String,
    }

    #[derive(Deserialize)]
    struct FixtureCase {
        name: String,
        input_shape: [usize; 4],
        input_bf16_bits: Vec<u16>,
        weight_shape: [usize; 3],
        weight_bf16_bits: Vec<u16>,
        expected_shape: [usize; 4],
        expected_output_bf16_bits: Vec<u16>,
    }

    #[test]
    fn scalar_grouped_wo_a_matches_independent_pinned_torch_fixture() {
        let fixture: Fixture = serde_json::from_str(include_str!(
            "../../../../../fixtures/deepseek-v41/output-projection-reference.json"
        ))
        .expect("checked-in output-projection fixture JSON");
        assert_eq!(fixture.schema_version, 1);
        assert_eq!(
            fixture.source.revision,
            "dba1be0a40aa45a94ad051997016db3960a90277"
        );
        assert_eq!(
            fixture.source.sha256,
            "4e9ae23620edc8028ccc5d5fef552ab7fdc7dcd6f79608754fe9f67644056f65"
        );
        assert_eq!(
            fixture.source.symbol,
            "Attention.forward:wo_a_grouped_einsum"
        );
        assert_eq!(fixture.reference.input_dtype, "bfloat16");
        assert_eq!(fixture.reference.output_dtype, "bfloat16_bits");
        assert_eq!(fixture.reference.einsum, "bsgd,grd->bsgr");
        assert_eq!(fixture.reference.torch_version, "2.13.0");
        assert_eq!(fixture.reference.device, "cpu");
        assert_eq!(fixture.cases.len(), 3);
        assert_eq!(
            fixture
                .cases
                .iter()
                .map(|case| case.name.as_str())
                .collect::<Vec<_>>(),
            vec![
                "nontrivial_batch_sequence_group_rank",
                "group_isolation",
                "output_rounding_after_multi_term_accumulation",
            ]
        );

        for case in fixture.cases {
            let [batches, positions, groups, reduction] = case.input_shape;
            let [weight_groups, rank, weight_reduction] = case.weight_shape;
            assert_eq!(groups, weight_groups, "{} group count", case.name);
            assert_eq!(reduction, weight_reduction, "{} reduction width", case.name);
            assert_eq!(
                case.expected_shape,
                [batches, positions, groups, rank],
                "{} output shape",
                case.name
            );
            let rows = batches * positions;
            assert_eq!(
                case.input_bf16_bits.len(),
                rows * groups * reduction,
                "{} input length",
                case.name
            );
            assert_eq!(
                case.weight_bf16_bits.len(),
                groups * rank * reduction,
                "{} weight length",
                case.name
            );
            assert_eq!(
                case.expected_output_bf16_bits.len(),
                rows * groups * rank,
                "{} expected length",
                case.name
            );

            let mut actual = vec![0_u16; rows * groups * rank];
            for group in 0..groups {
                let mut input = vec![0_u16; rows * reduction];
                for row in 0..rows {
                    let source = (row * groups + group) * reduction;
                    input[row * reduction..(row + 1) * reduction]
                        .copy_from_slice(&case.input_bf16_bits[source..source + reduction]);
                }
                let mut projected = vec![0_u16; rows * rank];
                let weight = &case.weight_bf16_bits
                    [group * rank * reduction..(group + 1) * rank * reduction];
                bf16_linear_reference(&input, weight, rows, reduction, rank, &mut projected)
                    .unwrap_or_else(|error| panic!("{}: {error}", case.name));
                for row in 0..rows {
                    let destination = (row * groups + group) * rank;
                    actual[destination..destination + rank]
                        .copy_from_slice(&projected[row * rank..(row + 1) * rank]);
                }
            }
            assert_eq!(actual, case.expected_output_bf16_bits, "{}", case.name);
        }
    }

    #[test]
    fn inverse_rope_precedes_grouped_projection_and_groups_stay_isolated() {
        let layout = AttentionOutputLayout::new(1, 1, 2, 32, 1, 2, 16, 32).unwrap();
        let mut attention = vec![0_u16; 64];
        // Inverse rotation by i maps [1, 2] to [2, -1], and [3, 4] to [4, -3].
        attention[30..32].copy_from_slice(&[0x3f80, 0x4000]);
        attention[62..64].copy_from_slice(&[0x4040, 0x4080]);
        let mut wo_a = vec![0_u16; 2 * 16 * 32];
        // Each group rank-0 selects its own head's final (rotated) coordinate.
        wo_a[31] = 0x3f80;
        wo_a[16 * 32 + 31] = 0x3f80;
        let mut wo_b = vec![0_u8; 32 * 32];
        wo_b[0] = 0x38; // hidden 0 reads group 0 rank 0.
        wo_b[32 + 16] = 0x38; // hidden 1 reads group 1 rank 0.
        let output = attention_output_reference(
            &attention,
            &[RotaryFrequency::new(0.0, 1.0).unwrap()],
            &wo_a,
            &wo_b,
            &[127],
            layout,
        )
        .unwrap();
        assert_eq!(output[0], 0xbf80); // -1
        assert_eq!(output[1], 0xc040); // -3
        assert!(output[2..].iter().all(|&bits| bits == 0));
    }

    #[test]
    fn sparse_attention_flows_through_inverse_rope_and_both_projections() {
        use std::num::NonZeroUsize;

        use crate::attention::{SparseAttentionLayout, sparse_attention_bf16_reference};

        let nz = |value| NonZeroUsize::new(value).unwrap();
        let attention_layout =
            SparseAttentionLayout::new(nz(1), nz(1), nz(2), nz(32), nz(2), nz(2)).unwrap();
        let output_layout = AttentionOutputLayout::new(1, 1, 2, 32, 1, 2, 16, 32).unwrap();
        let mut kv = vec![0_u16; 64];
        kv[30..32].copy_from_slice(&[0x4000, 0x4080]); // live tail [2, 4]
        kv[32..].fill(0x42c6); // masked stale key: every coordinate is 99
        let mut wo_a = vec![0_u16; 2 * 16 * 32];
        wo_a[31] = 0x3f80; // group 0 selects final coordinate
        wo_a[16 * 32 + 31] = 0x4000; // group 1 doubles it
        let mut wo_b = vec![0_u8; 32 * 32];
        wo_b[0] = 0x38;
        wo_b[32 + 16] = 0x38;

        // One zero-score key and a zero-score sink give exactly half the KV.
        // Raising the sink mass to three gives one quarter, without adding
        // any value vector. Both cases must survive all precision boundaries.
        for (sink, expected_tail, expected_output) in [
            (0.0, [0x3f80, 0x4000], [0xbf80, 0xc000]),
            (3.0_f32.ln(), [0x3f00, 0x3f80], [0xbf00, 0xbf80]),
        ] {
            let attention = sparse_attention_bf16_reference(
                &[0_u16; 64],
                &kv,
                &[sink; 2],
                &[0, -1],
                1.0,
                attention_layout,
            )
            .unwrap();
            for head in attention.chunks_exact(32) {
                assert_eq!(&head[30..], &expected_tail);
                assert!(head[..30].iter().all(|&bits| bits == 0));
            }
            let output = attention_output_reference(
                &attention,
                &[RotaryFrequency::new(0.0, 1.0).unwrap()],
                &wo_a,
                &wo_b,
                &[127],
                output_layout,
            )
            .unwrap();
            assert_eq!(&output[..2], &expected_output);
            assert!(output[2..].iter().all(|&bits| bits == 0));
        }
    }

    #[test]
    fn non_quarter_inverse_rope_rounds_before_wo_a_staging() {
        let layout = AttentionOutputLayout::new(1, 1, 1, 32, 1, 1, 32, 1).unwrap();
        let mut attention = vec![0_u16; 32];
        attention[30..].copy_from_slice(&[0x3f80, 0x4000]); // [1, 2]
        let rotated = inverse_rotate_and_narrow(
            &attention,
            &[RotaryFrequency::new(0.5, 0.866_025_4).unwrap()],
            layout,
        )
        .unwrap();
        // Inverse 60°: [1, 2] -> [2.2320508, 0.1339746], then BF16 storage.
        assert_eq!(&rotated[30..], &[0x400f, 0x3e09]);
    }

    #[test]
    fn rejects_non_g32_flattened_rank() {
        assert!(AttentionOutputLayout::new(1, 1, 1, 2, 1, 1, 1, 1).is_err());
    }

    #[test]
    fn rejects_total_projection_work_before_staging() {
        assert!(matches!(
            AttentionOutputLayout::new(512, 1, 1, 32, 1, 1, 1_024, 32),
            Err(AttentionOutputLayoutError::WorkLimit { .. })
        ));
    }
}
