//! Bounded query and signed-head-weight preparation for V4.1's indexer.
//!
//! This is the source-shaped prefix of `Indexer.forward`: it stops after the
//! query's FP4 reconstruction and the BF16 `weights_proj` scale boundary. It
//! intentionally does not score keys, apply causal masks, select candidates,
//! or own an index-key cache.

use std::num::NonZeroUsize;

use thiserror::Error;

use crate::{
    RotaryDirection, RotaryError, RotaryFrequency, RotaryTailLayout,
    precision::{
        ActivationGroup, ActivationQuantError, Bf16LinearError, Fp4ActivationError,
        Fp4ActivationMode, Fp8LinearError, bf16_linear_reference, f32_to_bf16_rne,
        fp8_linear_runtime_f32, quantize_bf16_activations_e4m3fn, requantize_bf16_activations_e2m1,
    },
    rotate_tail,
};

const MAX_INDEX_QUERY_ELEMENTS: usize = 1 << 20;

/// Explicit source geometry for one index-query preparation call.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct IndexQueryLayout {
    batches: NonZeroUsize,
    hidden_dimension: NonZeroUsize,
    q_rank: NonZeroUsize,
    heads: NonZeroUsize,
    head_dimension: NonZeroUsize,
    rope_pairs: NonZeroUsize,
}

impl IndexQueryLayout {
    /// Creates a bounded source-shaped index-query layout.
    #[allow(
        clippy::too_many_arguments,
        reason = "the source's six relevant dimensions stay explicit"
    )]
    pub fn new(
        batches: NonZeroUsize,
        hidden_dimension: NonZeroUsize,
        q_rank: NonZeroUsize,
        heads: NonZeroUsize,
        head_dimension: NonZeroUsize,
        rope_pairs: NonZeroUsize,
    ) -> Result<Self, IndexQueryLayoutError> {
        let rope_width =
            rope_pairs
                .get()
                .checked_mul(2)
                .ok_or(IndexQueryLayoutError::ShapeOverflow {
                    field: "rope width",
                })?;
        if rope_width > head_dimension.get() {
            return Err(IndexQueryLayoutError::RopeExceedsHead {
                rope_width,
                head_dimension: head_dimension.get(),
            });
        }
        for (field, width) in [
            ("hidden dimension", hidden_dimension.get()),
            ("q rank", q_rank.get()),
            ("index head dimension", head_dimension.get()),
        ] {
            if !width.is_multiple_of(32) {
                return Err(IndexQueryLayoutError::UngroupedWidth { field, width });
            }
        }
        for (field, elements) in [
            (
                "one-position hidden",
                product(
                    &[batches.get(), hidden_dimension.get()],
                    "one-position hidden",
                )?,
            ),
            (
                "one-position query",
                product(
                    &[batches.get(), heads.get(), head_dimension.get()],
                    "one-position query",
                )?,
            ),
            (
                "one-position projected weights",
                product(
                    &[batches.get(), heads.get()],
                    "one-position projected weights",
                )?,
            ),
        ] {
            if elements > MAX_INDEX_QUERY_ELEMENTS {
                return Err(IndexQueryLayoutError::ElementLimit { field, elements });
            }
        }
        Ok(Self {
            batches,
            hidden_dimension,
            q_rank,
            heads,
            head_dimension,
            rope_pairs,
        })
    }
}

/// Borrowed FP8/BF16 weights used by the index-query prefix.
#[derive(Clone, Copy, Debug)]
pub struct IndexQueryWeights<'a> {
    /// Runtime E4M3FN `wq_b` codes `[heads * head_dimension, q_rank]`.
    pub wq_b_codes: &'a [u8],
    /// Runtime E8M0 `wq_b` scales.
    pub wq_b_scales: &'a [u8],
    /// BF16 `weights_proj` values `[heads, hidden_dimension]`.
    pub weights_proj: &'a [u16],
}

/// Source-visible precision boundaries from index-query preparation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct IndexQueryDiagnostic {
    /// BF16 `wq_b(qr)` before rotary, `[batch, position, head, dimension]`.
    pub query_pre_rope: Vec<u16>,
    /// BF16 query immediately after rotary, before FP4 reconstruction.
    pub query_post_rope: Vec<u16>,
    /// BF16 query after G32/E8M0 logical FP4 reconstruction.
    pub query_post_fp4: Vec<u16>,
    /// BF16 `weights_proj(x)` before the source scalar multiplier.
    pub projected_head_weights: Vec<u16>,
    /// BF16 signed head weights after `index_head_dim^-0.5 * n_heads^-0.5`.
    pub scaled_head_weights: Vec<u16>,
}

/// Invalid index-query geometry.
#[derive(Clone, Copy, Debug, Error, Eq, PartialEq)]
#[non_exhaustive]
pub enum IndexQueryLayoutError {
    #[error("index-query shape arithmetic overflowed for {field}")]
    ShapeOverflow { field: &'static str },
    #[error("rope width {rope_width} exceeds index head dimension {head_dimension}")]
    RopeExceedsHead {
        rope_width: usize,
        head_dimension: usize,
    },
    #[error("{field} width {width} is not divisible by group 32")]
    UngroupedWidth { field: &'static str, width: usize },
    #[error("index-query {field} has {elements} elements, maximum is {MAX_INDEX_QUERY_ELEMENTS}")]
    ElementLimit {
        field: &'static str,
        elements: usize,
    },
}

/// Rejected index-query prefix input.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum IndexQueryError {
    #[error(transparent)]
    Layout(#[from] IndexQueryLayoutError),
    #[error(transparent)]
    ActivationQuant(#[from] ActivationQuantError),
    #[error(transparent)]
    Fp8Linear(#[from] Fp8LinearError),
    #[error(transparent)]
    Rotary(#[from] RotaryError),
    #[error(transparent)]
    Fp4(#[from] Fp4ActivationError),
    #[error(transparent)]
    Bf16Linear(#[from] Bf16LinearError),
    #[error("{field} length is {actual}; expected a nonempty multiple of {stride}")]
    InputLength {
        field: &'static str,
        actual: usize,
        stride: usize,
    },
    #[error("qr and x have inconsistent position counts: qr {qr_positions}, x {x_positions}")]
    PositionMismatch {
        qr_positions: usize,
        x_positions: usize,
    },
    #[error("index-query {field} has {elements} elements, maximum is {MAX_INDEX_QUERY_ELEMENTS}")]
    ElementLimit {
        field: &'static str,
        elements: usize,
    },
    #[error("index-query shape arithmetic overflowed for {field}")]
    ShapeOverflow { field: &'static str },
    #[error("FP8 query projection was nonfinite after BF16 narrowing at {element}")]
    NonFiniteProjection { element: usize },
    #[error("rotary query was nonfinite after BF16 narrowing at tail element {element}")]
    NonFiniteRotary { element: usize },
    #[error("source index scalar could not narrow to finite BF16 at head weight {element}")]
    NonFiniteScale { element: usize },
}

/// Prepares the source indexer's FP4 query and scaled signed head weights.
///
/// `qr` is BF16 `[batch, position, q_rank]`; `x` is BF16
/// `[batch, position, hidden_dimension]`; and `frequencies` is the call-local
/// `[position, rope_pair]` slice, already offset by the caller's start position.
/// The result models scalar BF16/FP8/FP4 boundaries, not `PyTorch` GEMM reduction
/// parity or the downstream scoring and selection operations.
pub fn prepare_index_query(
    qr: &[u16],
    x: &[u16],
    frequencies: &[RotaryFrequency],
    weights: IndexQueryWeights<'_>,
    layout: IndexQueryLayout,
) -> Result<IndexQueryDiagnostic, IndexQueryError> {
    let positions = validate_positions(qr, x, layout)?;
    let rows = product(&[layout.batches.get(), positions], "rows")?;
    let query_width = product(
        &[layout.heads.get(), layout.head_dimension.get()],
        "query width",
    )?;
    let query_pre_rope = fp8_project_bf16(qr, rows, layout.q_rank.get(), query_width, weights)?;
    let query_post_rope = rotate_query(&query_pre_rope, layout, positions, frequencies)?;
    let mut query_post_fp4 = vec![0; query_post_rope.len()];
    let fp4_rows = product(&[rows, layout.heads.get()], "FP4 query rows")?;
    requantize_bf16_activations_e2m1(
        &query_post_rope,
        fp4_rows,
        layout.head_dimension.get(),
        Fp4ActivationMode::Index32E8m0,
        &mut query_post_fp4,
    )?;
    let projected_head_weights = project_head_weights(x, rows, layout, weights.weights_proj)?;
    let scaled_head_weights = scale_head_weights(&projected_head_weights, layout)?;
    Ok(IndexQueryDiagnostic {
        query_pre_rope,
        query_post_rope,
        query_post_fp4,
        projected_head_weights,
        scaled_head_weights,
    })
}

fn validate_positions(
    qr: &[u16],
    x: &[u16],
    layout: IndexQueryLayout,
) -> Result<usize, IndexQueryError> {
    let qr_stride = product(&[layout.batches.get(), layout.q_rank.get()], "qr stride")?;
    let x_stride = product(
        &[layout.batches.get(), layout.hidden_dimension.get()],
        "x stride",
    )?;
    let qr_positions = positions("qr", qr.len(), qr_stride)?;
    let x_positions = positions("x", x.len(), x_stride)?;
    if qr_positions != x_positions {
        return Err(IndexQueryError::PositionMismatch {
            qr_positions,
            x_positions,
        });
    }
    for (field, elements) in [("qr", qr.len()), ("x", x.len())] {
        if elements > MAX_INDEX_QUERY_ELEMENTS {
            return Err(IndexQueryError::ElementLimit { field, elements });
        }
    }
    Ok(qr_positions)
}

fn positions(field: &'static str, length: usize, stride: usize) -> Result<usize, IndexQueryError> {
    if length == 0 || !length.is_multiple_of(stride) {
        return Err(IndexQueryError::InputLength {
            field,
            actual: length,
            stride,
        });
    }
    Ok(length / stride)
}

fn fp8_project_bf16(
    qr: &[u16],
    rows: usize,
    reduction: usize,
    outputs: usize,
    weights: IndexQueryWeights<'_>,
) -> Result<Vec<u16>, IndexQueryError> {
    let mut codes = vec![0; qr.len()];
    let mut scales = vec![0; rows * (reduction / 32)];
    quantize_bf16_activations_e4m3fn(
        qr,
        rows,
        reduction,
        ActivationGroup::Elements32,
        &mut codes,
        &mut scales,
    )?;
    let output_elements = product(&[rows, outputs], "FP8 query output")?;
    check_bound("FP8 query output", output_elements)?;
    let mut fp32 = vec![0.0; output_elements];
    fp8_linear_runtime_f32(
        &codes,
        &scales,
        weights.wq_b_codes,
        weights.wq_b_scales,
        rows,
        reduction,
        outputs,
        ActivationGroup::Elements32,
        &mut fp32,
    )?;
    fp32.into_iter()
        .enumerate()
        .map(|(element, value)| {
            let bits = f32_to_bf16_rne(value);
            if bf16_to_f32(bits).is_finite() {
                Ok(bits)
            } else {
                Err(IndexQueryError::NonFiniteProjection { element })
            }
        })
        .collect()
}

fn rotate_query(
    query: &[u16],
    layout: IndexQueryLayout,
    positions: usize,
    frequencies: &[RotaryFrequency],
) -> Result<Vec<u16>, IndexQueryError> {
    let prefix = layout.head_dimension.get() - layout.rope_pairs.get() * 2;
    let mut tail =
        Vec::with_capacity(query.len() / layout.head_dimension.get() * layout.rope_pairs.get() * 2);
    for head in query.chunks_exact(layout.head_dimension.get()) {
        tail.extend(head[prefix..].iter().map(|&bits| bf16_to_f32(bits)));
    }
    let rotary_layout = RotaryTailLayout::new(
        layout.batches,
        NonZeroUsize::new(positions).expect("validated positions"),
        layout.heads,
        layout.rope_pairs,
    )?;
    rotate_tail(
        &mut tail,
        rotary_layout,
        frequencies,
        RotaryDirection::Forward,
    )?;
    let mut output = query.to_vec();
    for (head, rotated_tail) in output
        .chunks_exact_mut(layout.head_dimension.get())
        .zip(tail.chunks_exact(layout.rope_pairs.get() * 2))
    {
        for (offset, &value) in rotated_tail.iter().enumerate() {
            let bits = f32_to_bf16_rne(value);
            if !bf16_to_f32(bits).is_finite() {
                return Err(IndexQueryError::NonFiniteRotary { element: offset });
            }
            head[prefix + offset] = bits;
        }
    }
    Ok(output)
}

fn project_head_weights(
    x: &[u16],
    rows: usize,
    layout: IndexQueryLayout,
    weights: &[u16],
) -> Result<Vec<u16>, IndexQueryError> {
    let outputs = product(&[rows, layout.heads.get()], "projected head weights")?;
    check_bound("projected head weights", outputs)?;
    let mut output = vec![0; outputs];
    bf16_linear_reference(
        x,
        weights,
        rows,
        layout.hidden_dimension.get(),
        layout.heads.get(),
        &mut output,
    )?;
    Ok(output)
}

#[allow(
    clippy::cast_possible_truncation,
    reason = "the source Python double scalar is cast to the tensor's FP32 scalar"
)]
fn source_head_scale(layout: IndexQueryLayout) -> Result<f32, IndexQueryError> {
    let dim = f64::from(u32::try_from(layout.head_dimension.get()).map_err(|_| {
        IndexQueryError::ShapeOverflow {
            field: "index head dimension f64",
        }
    })?);
    let heads = f64::from(u32::try_from(layout.heads.get()).map_err(|_| {
        IndexQueryError::ShapeOverflow {
            field: "head count f64",
        }
    })?);
    let scale = (dim.sqrt().recip() * heads.sqrt().recip()) as f32;
    if scale.is_finite() && scale > 0.0 {
        Ok(scale)
    } else {
        Err(IndexQueryError::NonFiniteScale { element: 0 })
    }
}

fn scale_head_weights(
    projected: &[u16],
    layout: IndexQueryLayout,
) -> Result<Vec<u16>, IndexQueryError> {
    let scale = source_head_scale(layout)?;
    projected
        .iter()
        .enumerate()
        .map(|(element, &bits)| {
            let value = bf16_to_f32(bits) * scale;
            let result = f32_to_bf16_rne(value);
            if bf16_to_f32(result).is_finite() {
                Ok(result)
            } else {
                Err(IndexQueryError::NonFiniteScale { element })
            }
        })
        .collect()
}

fn product(values: &[usize], field: &'static str) -> Result<usize, IndexQueryLayoutError> {
    values.iter().try_fold(1_usize, |total, &value| {
        total
            .checked_mul(value)
            .ok_or(IndexQueryLayoutError::ShapeOverflow { field })
    })
}

fn check_bound(field: &'static str, elements: usize) -> Result<(), IndexQueryError> {
    if elements > MAX_INDEX_QUERY_ELEMENTS {
        Err(IndexQueryError::ElementLimit { field, elements })
    } else {
        Ok(())
    }
}

fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}

#[cfg(test)]
mod tests {
    use super::{
        IndexQueryError, IndexQueryLayout, IndexQueryLayoutError, IndexQueryWeights,
        prepare_index_query,
    };
    use crate::RotaryFrequency;
    use std::num::NonZeroUsize;

    fn nonzero(value: usize) -> NonZeroUsize {
        NonZeroUsize::new(value).expect("test dimensions are nonzero")
    }
    fn bf16(value: f32) -> u16 {
        let bits = value.to_bits();
        u16::try_from(bits.wrapping_add(0x7fff + ((bits >> 16) & 1)) >> 16).expect("high half fits")
    }
    fn layout() -> IndexQueryLayout {
        IndexQueryLayout::new(
            nonzero(1),
            nonzero(32),
            nonzero(32),
            nonzero(2),
            nonzero(32),
            nonzero(1),
        )
        .expect("small index layout")
    }

    #[test]
    fn projection_and_source_double_scale_preserve_all_boundaries() {
        let mut projection = vec![0_u16; 64];
        projection[0] = bf16(-32.0);
        projection[32] = bf16(8.0);
        let diagnostic = prepare_index_query(
            &[0; 32],
            &[
                bf16(1.0),
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
            ],
            &[RotaryFrequency::new(1.0, 0.0).expect("finite identity frequency")],
            IndexQueryWeights {
                wq_b_codes: &[0; 64 * 32],
                wq_b_scales: &[127, 127],
                weights_proj: &projection,
            },
            layout(),
        )
        .expect("finite source-shaped preparation");
        assert_eq!(diagnostic.query_pre_rope, vec![0; 64]);
        assert_eq!(diagnostic.query_post_rope, vec![0; 64]);
        assert_eq!(diagnostic.query_post_fp4, vec![0; 64]);
        assert_eq!(
            diagnostic.projected_head_weights,
            vec![bf16(-32.0), bf16(8.0)]
        );
        assert_eq!(diagnostic.scaled_head_weights, vec![bf16(-4.0), bf16(1.0)]);
    }

    #[test]
    fn rejects_ungrouped_query_rank_before_runtime_buffers() {
        assert!(matches!(
            IndexQueryLayout::new(
                nonzero(1),
                nonzero(32),
                nonzero(31),
                nonzero(1),
                nonzero(32),
                nonzero(1)
            ),
            Err(IndexQueryLayoutError::UngroupedWidth {
                field: "q rank",
                width: 31
            })
        ));
    }

    #[test]
    fn rejects_mismatched_position_prefixes() {
        let error = prepare_index_query(
            &[0; 32],
            &[0; 64],
            &[],
            IndexQueryWeights {
                wq_b_codes: &[],
                wq_b_scales: &[],
                weights_proj: &[],
            },
            layout(),
        )
        .expect_err("position mismatch precedes weight validation");
        assert!(matches!(
            error,
            IndexQueryError::PositionMismatch {
                qr_positions: 1,
                x_positions: 2
            }
        ));
    }
}
