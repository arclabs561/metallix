//! Bounded scalar BF16 sparse-attention reference for the pinned kernel order.
//!
//! This models block-64 FP32 online normalization, BF16 probability narrowing
//! before the value reduction, and BF16 output storage. It is not a `TileLang`,
//! CUDA, GEMM reduction-order, exponential-implementation, or full-model
//! parity claim.

use thiserror::Error;

use super::{SparseAttentionError, SparseAttentionLayout, validate_inputs};

/// Largest one-buffer BF16 sparse-attention input or output accepted here.
pub const MAX_BF16_ATTENTION_ELEMENTS: usize = 1 << 20;

const MAX_BF16_ATTENTION_WORK: usize = 1 << 24;
const BLOCK: usize = 64;
const EMPTY_ROW_MAX: f32 = -1.0e30;

/// Errors from the bounded BF16 sparse-attention reference.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum SparseAttentionBf16Error {
    /// The shared layout, index, scale, or FP32 sink contract was invalid.
    #[error("invalid sparse-attention input: {0}")]
    Input(#[from] SparseAttentionError),
    /// A raw buffer or returned output exceeds this reference's fixed bound.
    #[error("BF16 sparse-attention {field} has {elements} elements, maximum is {maximum}")]
    ElementLimit {
        /// Buffer role.
        field: &'static str,
        /// Requested element count.
        elements: usize,
        /// Fixed reference-wide maximum.
        maximum: usize,
    },
    /// The conservative dot-plus-value-reduction count overflowed `usize`.
    #[error("BF16 sparse-attention work arithmetic overflowed")]
    WorkOverflow,
    /// The conservative dot-plus-value-reduction count exceeded the fixed bound.
    #[error("BF16 sparse-attention work estimate {required} exceeds maximum {maximum}")]
    WorkloadTooLarge {
        /// Requested scalar multiply-accumulate count.
        required: usize,
        /// Fixed reference-wide maximum.
        maximum: usize,
    },
    /// A bounded host-side widening or output allocation could not be reserved.
    #[error("could not allocate {elements} BF16 sparse-attention {field} elements")]
    AllocationFailed {
        /// Allocation role.
        field: &'static str,
        /// Requested element count.
        elements: usize,
    },
    /// BF16 storage widened to a NaN or infinity.
    #[error("nonfinite BF16 {field} at scalar index {index}")]
    NonFiniteBf16 {
        /// Input buffer role.
        field: &'static str,
        /// Flat BF16 scalar index.
        index: usize,
    },
    /// Ordered scalar FP32 arithmetic or its BF16 reconstruction was nonfinite.
    #[error(
        "BF16 sparse-attention arithmetic became nonfinite at {stage}, batch {batch}, query {query}, head {head}, index {index}"
    )]
    NonFiniteArithmetic {
        /// Named arithmetic stage.
        stage: &'static str,
        /// Batch coordinate.
        batch: usize,
        /// Query-position coordinate.
        query: usize,
        /// Attention-head coordinate.
        head: usize,
        /// Dimension or slot coordinate for the named stage.
        index: usize,
    },
}

/// Computes the bounded BF16 sparse-attention staging reference.
///
/// `query` is `[batch, query_position, head, dimension]`, `shared_kv` is
/// `[batch, key_position, dimension]`, and the returned BF16 storage is shaped
/// like `query`. The source's block-64 online order is explicit: the running
/// score maximum and sum stay FP32, each unnormalized valid-key exponential is
/// rounded to BF16 before multiplying its BF16 key/value vector, and final
/// output is round-to-nearest-even BF16. The denominator-only sink is added
/// only after all key blocks. An all-masked row returns BF16 zero directly.
///
/// This deliberately rejects nonfinite scores, exponentials, accumulations,
/// and outputs instead of returning source-kernel NaNs or infinities. It uses
/// ordered scalar FP32 operations, so it does not qualify device GEMM or
/// transcendental parity.
///
/// # Errors
///
/// Returns [`SparseAttentionBf16Error`] before widening or output allocation
/// for an over-bound geometry, and before publishing a result for invalid data
/// or nonfinite scalar arithmetic.
pub fn sparse_attention_bf16_reference(
    query: &[u16],
    shared_kv: &[u16],
    sink: &[f32],
    indices: &[i32],
    scale: f32,
    layout: SparseAttentionLayout,
) -> Result<Vec<u16>, SparseAttentionBf16Error> {
    let query_elements = layout.query_len()?;
    let kv_elements = layout.kv_len()?;
    let index_elements = layout.index_len()?;
    let output_elements = layout.output_len()?;
    enforce_element_limit("query", query_elements)?;
    enforce_element_limit("shared_kv", kv_elements)?;
    enforce_element_limit("sink", layout.heads.get())?;
    enforce_element_limit("indices", index_elements)?;
    enforce_element_limit("output", output_elements)?;
    enforce_element_limit("query input", query.len())?;
    enforce_element_limit("shared_kv input", shared_kv.len())?;
    enforce_element_limit("sink input", sink.len())?;
    enforce_element_limit("indices input", indices.len())?;
    let work = attention_work(layout)?;
    if work > MAX_BF16_ATTENTION_WORK {
        return Err(SparseAttentionBf16Error::WorkloadTooLarge {
            required: work,
            maximum: MAX_BF16_ATTENTION_WORK,
        });
    }

    let query_f32 = widen_bf16("query", query)?;
    let shared_kv_f32 = widen_bf16("shared_kv", shared_kv)?;
    validate_inputs(&query_f32, &shared_kv_f32, sink, indices, scale, layout)?;

    let mut output = Vec::new();
    output.try_reserve_exact(output_elements).map_err(|_| {
        SparseAttentionBf16Error::AllocationFailed {
            field: "output",
            elements: output_elements,
        }
    })?;
    output.resize(output_elements, 0);
    Reference {
        query: &query_f32,
        shared_kv: &shared_kv_f32,
        sink,
        indices,
        scale,
        layout,
    }
    .run(&mut output)?;
    Ok(output)
}

struct Reference<'a> {
    query: &'a [f32],
    shared_kv: &'a [f32],
    sink: &'a [f32],
    indices: &'a [i32],
    scale: f32,
    layout: SparseAttentionLayout,
}

#[derive(Clone, Copy)]
struct Coordinates {
    batch: usize,
    query: usize,
    head: usize,
}

struct OnlineState {
    accumulator: Vec<f32>,
    scores_max: f32,
    sum_exp: f32,
}

impl Reference<'_> {
    fn run(&self, output: &mut [u16]) -> Result<(), SparseAttentionBf16Error> {
        for batch in 0..self.layout.batches.get() {
            for query in 0..self.layout.query_positions.get() {
                self.run_row(batch, query, output)?;
            }
        }
        Ok(())
    }

    fn run_row(
        &self,
        batch: usize,
        query: usize,
        output: &mut [u16],
    ) -> Result<(), SparseAttentionBf16Error> {
        let index_base = (batch * self.layout.query_positions.get() + query) * self.slots();
        if !self.indices[index_base..index_base + self.slots()]
            .iter()
            .any(|&index| index >= 0)
        {
            return Ok(());
        }
        for (head, &sink) in self.sink.iter().enumerate() {
            self.run_head(Coordinates { batch, query, head }, index_base, sink, output)?;
        }
        Ok(())
    }

    fn run_head(
        &self,
        coordinates: Coordinates,
        index_base: usize,
        sink: f32,
        output: &mut [u16],
    ) -> Result<(), SparseAttentionBf16Error> {
        let query_base = self.query_base(coordinates);
        let mut state = OnlineState::new(self.dimensions())?;
        for block_start in (0..self.slots()).step_by(BLOCK) {
            self.run_block(coordinates, query_base, index_base, block_start, &mut state)?;
        }
        Self::finish_head(coordinates, sink, query_base, state, output)
    }

    fn run_block(
        &self,
        coordinates: Coordinates,
        query_base: usize,
        index_base: usize,
        block_start: usize,
        state: &mut OnlineState,
    ) -> Result<(), SparseAttentionBf16Error> {
        let block_len = (self.slots() - block_start).min(BLOCK);
        let mut scores = [f32::NEG_INFINITY; BLOCK];
        let previous_max = state.scores_max;
        for (lane, score) in scores.iter_mut().take(block_len).enumerate() {
            let key_index = self.indices[index_base + block_start + lane];
            if key_index < 0 {
                continue;
            }
            *score = self.score(coordinates, query_base, key_index, block_start + lane)?;
            state.scores_max = state.scores_max.max(*score);
        }
        let rescale = checked_finite(
            (previous_max - state.scores_max).exp(),
            "rescale",
            coordinates,
            block_start,
        )?;
        Self::rescale_accumulator(coordinates, rescale, &mut state.accumulator)?;
        self.accumulate_block(
            coordinates,
            index_base,
            block_start,
            &scores,
            rescale,
            state,
        )
    }

    fn accumulate_block(
        &self,
        coordinates: Coordinates,
        index_base: usize,
        block_start: usize,
        scores: &[f32; BLOCK],
        rescale: f32,
        state: &mut OnlineState,
    ) -> Result<(), SparseAttentionBf16Error> {
        let block_len = (self.slots() - block_start).min(BLOCK);
        let mut scores_sum = 0.0_f32;
        for (lane, score) in scores.iter().take(block_len).enumerate() {
            let key_index = self.indices[index_base + block_start + lane];
            if key_index < 0 {
                continue;
            }
            let index = block_start + lane;
            let probability = checked_finite(
                (*score - state.scores_max).exp(),
                "probability exponential",
                coordinates,
                index,
            )?;
            scores_sum = checked_finite(
                scores_sum + probability,
                "block probability sum",
                coordinates,
                index,
            )?;
            self.accumulate_value(coordinates, key_index, probability, &mut state.accumulator)?;
        }
        state.sum_exp = checked_finite(
            state.sum_exp * rescale + scores_sum,
            "running probability sum",
            coordinates,
            block_start,
        )?;
        Ok(())
    }

    fn rescale_accumulator(
        coordinates: Coordinates,
        rescale: f32,
        accumulator: &mut [f32],
    ) -> Result<(), SparseAttentionBf16Error> {
        for (dimension, value) in accumulator.iter_mut().enumerate() {
            *value = checked_finite(
                *value * rescale,
                "rescaled accumulator",
                coordinates,
                dimension,
            )?;
        }
        Ok(())
    }

    fn accumulate_value(
        &self,
        coordinates: Coordinates,
        key_index: i32,
        probability: f32,
        accumulator: &mut [f32],
    ) -> Result<(), SparseAttentionBf16Error> {
        let probability = bf16_to_f32(f32_to_bf16_rne(probability));
        let key_base = self.key_base(coordinates, key_index)?;
        for (dimension, value) in accumulator.iter_mut().enumerate() {
            *value = checked_finite(
                *value + probability * self.shared_kv[key_base + dimension],
                "value accumulation",
                coordinates,
                dimension,
            )?;
        }
        Ok(())
    }

    fn finish_head(
        coordinates: Coordinates,
        sink: f32,
        output_base: usize,
        state: OnlineState,
        output: &mut [u16],
    ) -> Result<(), SparseAttentionBf16Error> {
        let sink_probability = checked_finite(
            (sink - state.scores_max).exp(),
            "sink exponential",
            coordinates,
            0,
        )?;
        let denominator = checked_finite(
            state.sum_exp + sink_probability,
            "denominator",
            coordinates,
            0,
        )?;
        for (dimension, value) in state.accumulator.into_iter().enumerate() {
            let normalized = checked_finite(
                value / denominator,
                "normalized output",
                coordinates,
                dimension,
            )?;
            let bits = f32_to_bf16_rne(normalized);
            if !bf16_to_f32(bits).is_finite() {
                return Err(SparseAttentionBf16Error::NonFiniteArithmetic {
                    stage: "BF16 output reconstruction",
                    batch: coordinates.batch,
                    query: coordinates.query,
                    head: coordinates.head,
                    index: dimension,
                });
            }
            output[output_base + dimension] = bits;
        }
        Ok(())
    }

    fn score(
        &self,
        coordinates: Coordinates,
        query_base: usize,
        key_index: i32,
        slot: usize,
    ) -> Result<f32, SparseAttentionBf16Error> {
        let key_base = self.key_base(coordinates, key_index)?;
        let mut dot = 0.0_f32;
        for dimension in 0..self.dimensions() {
            let product = checked_finite(
                self.query[query_base + dimension] * self.shared_kv[key_base + dimension],
                "dot product",
                coordinates,
                dimension,
            )?;
            dot = checked_finite(dot + product, "dot sum", coordinates, dimension)?;
        }
        checked_finite(dot * self.scale, "scaled score", coordinates, slot)
    }

    fn key_base(
        &self,
        coordinates: Coordinates,
        key_index: i32,
    ) -> Result<usize, SparseAttentionBf16Error> {
        let key_index = usize::try_from(key_index).map_err(|_| {
            SparseAttentionBf16Error::Input(SparseAttentionError::InvalidIndex {
                slot: 0,
                index: key_index,
                key_positions: self.layout.key_positions.get(),
            })
        })?;
        Ok((coordinates.batch * self.layout.key_positions.get() + key_index) * self.dimensions())
    }

    fn query_base(&self, coordinates: Coordinates) -> usize {
        ((coordinates.batch * self.layout.query_positions.get() + coordinates.query)
            * self.layout.heads.get()
            + coordinates.head)
            * self.dimensions()
    }

    fn dimensions(&self) -> usize {
        self.layout.dimensions.get()
    }

    fn slots(&self) -> usize {
        self.layout.sparse_slots.get()
    }
}

impl OnlineState {
    fn new(dimensions: usize) -> Result<Self, SparseAttentionBf16Error> {
        let mut accumulator = Vec::new();
        accumulator.try_reserve_exact(dimensions).map_err(|_| {
            SparseAttentionBf16Error::AllocationFailed {
                field: "accumulator",
                elements: dimensions,
            }
        })?;
        accumulator.resize(dimensions, 0.0);
        Ok(Self {
            accumulator,
            scores_max: EMPTY_ROW_MAX,
            sum_exp: 0.0,
        })
    }
}

fn attention_work(layout: SparseAttentionLayout) -> Result<usize, SparseAttentionBf16Error> {
    let mut work = 2_usize;
    for value in [
        layout.batches.get(),
        layout.query_positions.get(),
        layout.heads.get(),
        layout.sparse_slots.get(),
        layout.dimensions.get(),
    ] {
        work = work
            .checked_mul(value)
            .ok_or(SparseAttentionBf16Error::WorkOverflow)?;
    }
    Ok(work)
}

fn enforce_element_limit(
    field: &'static str,
    elements: usize,
) -> Result<(), SparseAttentionBf16Error> {
    if elements > MAX_BF16_ATTENTION_ELEMENTS {
        return Err(SparseAttentionBf16Error::ElementLimit {
            field,
            elements,
            maximum: MAX_BF16_ATTENTION_ELEMENTS,
        });
    }
    Ok(())
}

fn widen_bf16(field: &'static str, input: &[u16]) -> Result<Vec<f32>, SparseAttentionBf16Error> {
    let mut output = Vec::new();
    output.try_reserve_exact(input.len()).map_err(|_| {
        SparseAttentionBf16Error::AllocationFailed {
            field,
            elements: input.len(),
        }
    })?;
    for (index, &bits) in input.iter().enumerate() {
        let value = bf16_to_f32(bits);
        if !value.is_finite() {
            return Err(SparseAttentionBf16Error::NonFiniteBf16 { field, index });
        }
        output.push(value);
    }
    Ok(output)
}

fn checked_finite(
    value: f32,
    stage: &'static str,
    coordinates: Coordinates,
    index: usize,
) -> Result<f32, SparseAttentionBf16Error> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err(SparseAttentionBf16Error::NonFiniteArithmetic {
            stage,
            batch: coordinates.batch,
            query: coordinates.query,
            head: coordinates.head,
            index,
        })
    }
}

fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}

fn f32_to_bf16_rne(value: f32) -> u16 {
    let bits = value.to_bits();
    let rounded = bits.wrapping_add(0x7fff + ((bits >> 16) & 1));
    let bytes = rounded.to_be_bytes();
    u16::from_be_bytes([bytes[0], bytes[1]])
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroUsize;

    use super::{
        MAX_BF16_ATTENTION_ELEMENTS, SparseAttentionBf16Error, sparse_attention_bf16_reference,
    };
    use crate::{SparseAttentionError, SparseAttentionLayout};

    fn nz(value: usize) -> NonZeroUsize {
        NonZeroUsize::new(value).expect("test dimensions are nonzero")
    }

    fn layout(keys: usize, slots: usize, dimensions: usize) -> SparseAttentionLayout {
        SparseAttentionLayout::new(nz(1), nz(1), nz(1), nz(dimensions), nz(keys), nz(slots))
            .expect("small layout")
    }

    #[test]
    fn masks_and_duplicates_reach_the_bf16_probability_value_stage() {
        let duplicate = sparse_attention_bf16_reference(
            &[0x3f80],
            &[0x4000, 0x4080],
            &[0.0],
            &[0, 0, -1],
            1.0,
            layout(2, 3, 1),
        )
        .expect("finite duplicate row");
        let one = sparse_attention_bf16_reference(
            &[0x3f80],
            &[0x4000, 0x4080],
            &[0.0],
            &[0, -1, -1],
            1.0,
            layout(2, 3, 1),
        )
        .expect("finite masked row");
        assert_ne!(duplicate, one);
        assert_eq!(
            sparse_attention_bf16_reference(
                &[0x3f80],
                &[0x4000, 0x4080],
                &[0.0],
                &[-1, -1, -1],
                1.0,
                layout(2, 3, 1),
            )
            .expect("all-masked row"),
            vec![0]
        );
    }

    #[test]
    fn rejects_contract_nonfinite_and_bound_before_result_allocation() {
        assert!(matches!(
            sparse_attention_bf16_reference(&[], &[0x3f80], &[0.0], &[0], 1.0, layout(1, 1, 1),),
            Err(SparseAttentionBf16Error::Input(
                SparseAttentionError::LengthMismatch { field: "query", .. }
            ))
        ));
        assert!(matches!(
            sparse_attention_bf16_reference(
                &[0x7f80],
                &[0x3f80],
                &[0.0],
                &[0],
                1.0,
                layout(1, 1, 1),
            ),
            Err(SparseAttentionBf16Error::NonFiniteBf16 {
                field: "query",
                index: 0
            })
        ));
        assert!(matches!(
            sparse_attention_bf16_reference(
                &[0x3f80],
                &[0x3f80],
                &[0.0],
                &[-2],
                1.0,
                layout(1, 1, 1),
            ),
            Err(SparseAttentionBf16Error::Input(
                SparseAttentionError::InvalidIndex { .. }
            ))
        ));
        let oversized = SparseAttentionLayout::new(
            nz(1),
            nz(1),
            nz(1),
            nz(MAX_BF16_ATTENTION_ELEMENTS + 1),
            nz(1),
            nz(1),
        )
        .expect("shape arithmetic is representable");
        assert!(matches!(
            sparse_attention_bf16_reference(&[], &[], &[], &[], 1.0, oversized),
            Err(SparseAttentionBf16Error::ElementLimit { field: "query", .. })
        ));
    }

    #[test]
    fn sixty_fifth_slot_rescales_the_first_online_block() {
        let keys = vec![0x3f80_u16; 64]
            .into_iter()
            .chain([0x4000])
            .collect::<Vec<_>>();
        let indices = (0_i32..65).collect::<Vec<_>>();
        let output = sparse_attention_bf16_reference(
            &[0x3f80],
            &keys,
            &[-100.0],
            &indices,
            1.0,
            layout(65, 65, 1),
        )
        .expect("two online blocks");
        let value = f32::from_bits(u32::from(output[0]) << 16);
        assert!(value > 1.0 && value < 2.0);
    }
}
