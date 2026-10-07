//! Bounded preparation of V4.1 compressed attention KV rows.
//!
//! This owns the numerical path from an already-produced compressor latent
//! through its rotary tail and compressed-KV FP4 reconstruction. It does not
//! project a latent, schedule compressed groups, retain a cache, or publish
//! data to an attention consumer.

use std::num::NonZeroUsize;

use thiserror::Error;

use crate::{
    RotaryDirection, RotaryError, RotaryFrequency, RotaryTailLayout,
    precision::{
        Fp4ActivationError, Fp4ActivationMode, bf16_to_f32, f32_to_bf16_rne,
        requantize_bf16_activations_e2m1,
    },
    rotate_tail,
};

const MAX_COMPRESSED_KV_ELEMENTS: usize = 1 << 20;

/// Explicit source geometry for one compressed-KV preparation call.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CompressedKvLayout {
    batches: NonZeroUsize,
    value_dimension: NonZeroUsize,
    rope_pairs: NonZeroUsize,
}

impl CompressedKvLayout {
    /// Creates a bounded source-shaped compressed-KV layout.
    ///
    /// # Errors
    ///
    /// * [`CompressedKvLayoutError::RopeExceedsValue`] and
    ///   [`CompressedKvLayoutError::UngroupedValueWidth`] for widths the
    ///   source path cannot use.
    /// * [`CompressedKvLayoutError::ShapeOverflow`] and
    ///   [`CompressedKvLayoutError::ElementLimit`] when one position's values
    ///   do not fit the bound.
    pub fn new(
        batches: NonZeroUsize,
        value_dimension: NonZeroUsize,
        rope_pairs: NonZeroUsize,
    ) -> Result<Self, CompressedKvLayoutError> {
        let rope_width =
            rope_pairs
                .get()
                .checked_mul(2)
                .ok_or(CompressedKvLayoutError::ShapeOverflow {
                    field: "rope width",
                })?;
        if rope_width > value_dimension.get() {
            return Err(CompressedKvLayoutError::RopeExceedsValue {
                rope_width,
                value_dimension: value_dimension.get(),
            });
        }
        if !value_dimension.get().is_multiple_of(16) {
            return Err(CompressedKvLayoutError::UngroupedValueWidth {
                width: value_dimension.get(),
            });
        }
        let elements = layout_product(
            &[batches.get(), value_dimension.get()],
            "one-position compressed KV",
        )?;
        if elements > MAX_COMPRESSED_KV_ELEMENTS {
            return Err(CompressedKvLayoutError::ElementLimit { elements });
        }
        Ok(Self {
            batches,
            value_dimension,
            rope_pairs,
        })
    }
}

/// Source-visible precision boundaries from compressed-KV preparation.
#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub struct CompressedKvDiagnostic {
    /// BF16 compressor latent with its rotary tail applied.
    pub post_rope: Vec<u16>,
    /// BF16 compressed KV after G16/E4M3 logical FP4 reconstruction.
    pub post_fp4: Vec<u16>,
}

/// Invalid compressed-KV geometry.
#[derive(Clone, Copy, Debug, Error, Eq, PartialEq)]
#[non_exhaustive]
pub enum CompressedKvLayoutError {
    /// A derived shape does not fit in `usize`.
    #[error("compressed-KV shape arithmetic overflowed for {field}")]
    ShapeOverflow {
        /// The derived count's role.
        field: &'static str,
    },
    /// The rotary tail is wider than one value row.
    #[error("rope width {rope_width} exceeds compressed-KV value dimension {value_dimension}")]
    RopeExceedsValue {
        /// Rotary tail width, `2 * rope_pairs`.
        rope_width: usize,
        /// Value width.
        value_dimension: usize,
    },
    /// The value width is not a multiple of the FP4 group of 16.
    #[error("compressed-KV value width {width} is not divisible by group 16")]
    UngroupedValueWidth {
        /// Value width.
        width: usize,
    },
    /// One position's values would pass the fixed element cap.
    #[error(
        "compressed-KV one-position buffer has {elements} elements, maximum is {MAX_COMPRESSED_KV_ELEMENTS}"
    )]
    ElementLimit {
        /// `batches * value_dimension`.
        elements: usize,
    },
}

/// Rejected compressed-KV preparation input.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum CompressedKvError {
    /// The layout is invalid.
    #[error(transparent)]
    Layout(#[from] CompressedKvLayoutError),
    /// The rotary stage rejected its input.
    #[error(transparent)]
    Rotary(#[from] RotaryError),
    /// FP4 reconstruction rejected its input or overflowed.
    #[error(transparent)]
    Fp4(#[from] Fp4ActivationError),
    /// The latent is empty or not whole `[batch, position]` rows.
    #[error("compressed-KV latent length is {actual}; expected a nonempty multiple of {stride}")]
    LatentLength {
        /// Supplied length.
        actual: usize,
        /// Elements per compressed position across all batches.
        stride: usize,
    },
    /// The frequencies do not match the positions and rotary pairs.
    #[error("compressed-KV frequencies length is {actual}, expected {expected}")]
    FrequencyLength {
        /// Frequencies supplied.
        actual: usize,
        /// `positions * rope_pairs`.
        expected: usize,
    },
    /// A staging buffer would pass the fixed element cap.
    #[error(
        "compressed-KV {field} has {elements} elements, maximum is {MAX_COMPRESSED_KV_ELEMENTS}"
    )]
    ElementLimit {
        /// The buffer's role.
        field: &'static str,
        /// Its element count.
        elements: usize,
    },
    /// A derived shape does not fit in `usize`.
    #[error("compressed-KV shape arithmetic overflowed for {field}")]
    ShapeOverflow {
        /// The derived count's role.
        field: &'static str,
    },
    /// A latent value is NaN or infinite.
    #[error("nonfinite BF16 compressed-KV latent at position {position}")]
    NonFiniteInput {
        /// Flat index into the latent.
        position: usize,
    },
    /// A rotated value narrowed to a non-finite BF16.
    #[error("compressed-KV rotary result could not narrow to finite BF16 at element {element}")]
    NonFiniteRotary {
        /// Flat element in `[batch, compressed_position, value_dimension]` storage.
        element: usize,
    },
    /// A staging buffer could not be reserved.
    #[error("could not allocate {elements} compressed-KV {field} elements")]
    AllocationFailed {
        /// The buffer's role.
        field: &'static str,
        /// Elements that could not be reserved.
        elements: usize,
    },
}

/// Prepares source-shaped compressed KV from immutable BF16 compressor latents.
///
/// `latent` is `[batch, compressed_position, value_dimension]`; `frequencies`
/// is the call-local `[compressed_position, rope_pair]` slice selected by the
/// owner for each completed group. The source rotates the final `2 * rope_pairs`
/// values, narrows them back to BF16, then applies G16/E4M3 FP4 reconstruction.
/// The caller owns source-layer selection, incomplete-group handling, cache
/// publication, attention, and all request state.
///
/// This is a bounded CPU scalar precision-staging reference, not packed cache
/// storage, GPU execution, or a generic tensor operation. All shape, length,
/// finite-input, and allocation checks happen before a diagnostic can escape.
///
/// # Errors
///
/// * [`CompressedKvError::LatentLength`] and
///   [`CompressedKvError::FrequencyLength`] when the inputs do not match the
///   layout.
/// * [`CompressedKvError::ShapeOverflow`] and
///   [`CompressedKvError::ElementLimit`] past the bounds, and
///   [`CompressedKvError::AllocationFailed`] when staging cannot be reserved.
/// * [`CompressedKvError::NonFiniteInput`] for a NaN or infinite latent.
/// * [`CompressedKvError::Rotary`], [`CompressedKvError::NonFiniteRotary`]
///   and [`CompressedKvError::Fp4`] when the rotary or FP4 stage rejects its
///   input or overflows.
pub fn prepare_compressed_kv(
    latent: &[u16],
    frequencies: &[RotaryFrequency],
    layout: CompressedKvLayout,
) -> Result<CompressedKvDiagnostic, CompressedKvError> {
    let shape = Shape::new(latent, frequencies, layout)?;
    validate_finite(latent)?;

    let mut post_rope = reserve_u16(shape.elements, "post-rope")?;
    post_rope.copy_from_slice(latent);
    rotate_tail_bf16(&mut post_rope, frequencies, layout, shape.positions)?;

    let mut post_fp4 = reserve_u16(shape.elements, "post-FP4")?;
    requantize_bf16_activations_e2m1(
        &post_rope,
        shape.rows,
        layout.value_dimension.get(),
        Fp4ActivationMode::CompressedKv16E4m3,
        &mut post_fp4,
    )?;
    Ok(CompressedKvDiagnostic {
        post_rope,
        post_fp4,
    })
}

#[derive(Clone, Copy)]
struct Shape {
    positions: usize,
    rows: usize,
    elements: usize,
}

impl Shape {
    fn new(
        latent: &[u16],
        frequencies: &[RotaryFrequency],
        layout: CompressedKvLayout,
    ) -> Result<Self, CompressedKvError> {
        let stride = product(
            &[layout.batches.get(), layout.value_dimension.get()],
            "latent stride",
        )?;
        if latent.is_empty() || !latent.len().is_multiple_of(stride) {
            return Err(CompressedKvError::LatentLength {
                actual: latent.len(),
                stride,
            });
        }
        let positions = latent.len() / stride;
        let rows = product(&[layout.batches.get(), positions], "compressed-KV rows")?;
        let elements = product(
            &[rows, layout.value_dimension.get()],
            "compressed-KV output",
        )?;
        let expected_frequencies = product(&[positions, layout.rope_pairs.get()], "frequencies")?;
        if frequencies.len() != expected_frequencies {
            return Err(CompressedKvError::FrequencyLength {
                actual: frequencies.len(),
                expected: expected_frequencies,
            });
        }
        if elements > MAX_COMPRESSED_KV_ELEMENTS {
            return Err(CompressedKvError::ElementLimit {
                field: "latent",
                elements,
            });
        }
        Ok(Self {
            positions,
            rows,
            elements,
        })
    }
}

fn rotate_tail_bf16(
    output: &mut [u16],
    frequencies: &[RotaryFrequency],
    layout: CompressedKvLayout,
    positions: usize,
) -> Result<(), CompressedKvError> {
    let rope_width = layout.rope_pairs.get() * 2;
    let prefix = layout.value_dimension.get() - rope_width;
    let tail_elements = product(
        &[layout.batches.get(), positions, rope_width],
        "rotary tail",
    )?;
    let mut tail = Vec::new();
    tail.try_reserve_exact(tail_elements)
        .map_err(|_| CompressedKvError::AllocationFailed {
            field: "rotary tail",
            elements: tail_elements,
        })?;
    for row in output.chunks_exact(layout.value_dimension.get()) {
        tail.extend(row[prefix..].iter().map(|&bits| bf16_to_f32(bits)));
    }
    let rotary_layout = RotaryTailLayout::new(
        layout.batches,
        NonZeroUsize::new(positions).expect("validated nonempty compressed positions"),
        NonZeroUsize::new(1).expect("one compressed-KV head"),
        layout.rope_pairs,
    )?;
    rotate_tail(
        &mut tail,
        rotary_layout,
        frequencies,
        RotaryDirection::Forward,
    )?;
    for (row, (row_output, rotated_tail)) in output
        .chunks_exact_mut(layout.value_dimension.get())
        .zip(tail.chunks_exact(rope_width))
        .enumerate()
    {
        for (offset, &rotated) in rotated_tail.iter().enumerate() {
            let bits = f32_to_bf16_rne(rotated);
            if !bf16_to_f32(bits).is_finite() {
                return Err(CompressedKvError::NonFiniteRotary {
                    element: row * layout.value_dimension.get() + prefix + offset,
                });
            }
            row_output[prefix + offset] = bits;
        }
    }
    Ok(())
}

fn product(values: &[usize], field: &'static str) -> Result<usize, CompressedKvError> {
    values.iter().try_fold(1_usize, |total, &value| {
        total
            .checked_mul(value)
            .ok_or(CompressedKvError::ShapeOverflow { field })
    })
}

fn layout_product(values: &[usize], field: &'static str) -> Result<usize, CompressedKvLayoutError> {
    values.iter().try_fold(1_usize, |total, &value| {
        total
            .checked_mul(value)
            .ok_or(CompressedKvLayoutError::ShapeOverflow { field })
    })
}

fn validate_finite(values: &[u16]) -> Result<(), CompressedKvError> {
    values
        .iter()
        .position(|&bits| !bf16_to_f32(bits).is_finite())
        .map_or(Ok(()), |position| {
            Err(CompressedKvError::NonFiniteInput { position })
        })
}

fn reserve_u16(elements: usize, field: &'static str) -> Result<Vec<u16>, CompressedKvError> {
    let mut output = Vec::new();
    output
        .try_reserve_exact(elements)
        .map_err(|_| CompressedKvError::AllocationFailed { field, elements })?;
    output.resize(elements, 0);
    Ok(output)
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroUsize;

    use super::{
        CompressedKvError, CompressedKvLayout, CompressedKvLayoutError, prepare_compressed_kv,
    };
    use crate::RotaryFrequency;

    fn nz(value: usize) -> NonZeroUsize {
        NonZeroUsize::new(value).expect("nonzero test dimension")
    }

    fn bf16(value: f32) -> u16 {
        let bits = value.to_bits();
        u16::try_from(bits.wrapping_add(0x7fff + ((bits >> 16) & 1)) >> 16).expect("BF16 high half")
    }

    fn layout() -> CompressedKvLayout {
        CompressedKvLayout::new(nz(1), nz(16), nz(1)).expect("test layout")
    }

    #[test]
    fn rotates_tail_then_requantizes_compressed_kv() {
        let mut latent = vec![0_u16; 16];
        latent[14] = bf16(2.0);
        latent[15] = bf16(3.0);
        let diagnostic = prepare_compressed_kv(
            &latent,
            &[RotaryFrequency::new(0.0, 1.0).expect("finite frequency")],
            layout(),
        )
        .expect("finite staged KV");
        assert_eq!(diagnostic.post_rope[14], bf16(-3.0));
        assert_eq!(diagnostic.post_rope[15], bf16(2.0));
        assert_eq!(diagnostic.post_fp4.len(), 16);
        assert_eq!(latent[14], bf16(2.0), "input remains immutable");
    }

    #[test]
    fn rejects_layout_length_frequency_and_nonfinite_boundaries() {
        assert!(matches!(
            CompressedKvLayout::new(nz(1), nz(15), nz(1)),
            Err(CompressedKvLayoutError::UngroupedValueWidth { width: 15 })
        ));
        assert!(matches!(
            CompressedKvLayout::new(nz(1), nz(16), nz(9)),
            Err(CompressedKvLayoutError::RopeExceedsValue {
                rope_width: 18,
                value_dimension: 16,
            })
        ));
        assert!(matches!(
            prepare_compressed_kv(&[0; 15], &[], layout()),
            Err(CompressedKvError::LatentLength {
                actual: 15,
                stride: 16,
            })
        ));
        assert!(matches!(
            prepare_compressed_kv(&[0; 16], &[], layout()),
            Err(CompressedKvError::FrequencyLength {
                actual: 0,
                expected: 1,
            })
        ));
        assert!(matches!(
            prepare_compressed_kv(
                &[0x7fc0; 16],
                &[RotaryFrequency::new(1.0, 0.0).expect("finite frequency")],
                layout(),
            ),
            Err(CompressedKvError::NonFiniteInput { position: 0 })
        ));
    }
}
