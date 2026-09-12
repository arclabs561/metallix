//! Bounded raw-BF16 sliding-window scheduling and ring-copy reference.
//!
//! This mirrors the pinned attention helper's prefill and decode index order,
//! plus its raw window-cache placement. It accepts opaque BF16 storage bits and
//! does not establish cache continuity, quantization, GPU, or attention parity.

use std::num::NonZeroUsize;

use thiserror::Error;

/// The largest raw BF16 ring or schedule allocation accepted by this reference.
pub const MAX_WINDOW_ELEMENTS: usize = 1 << 20;

/// One source-shaped sliding-window update.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum WindowStep {
    /// A zero-origin chunk containing one or more consecutive tokens.
    Prefill {
        /// Number of tokens in the supplied chunk.
        tokens: NonZeroUsize,
    },
    /// One decode token at a strictly positive absolute position.
    ///
    /// Position zero belongs to [`Self::Prefill`], matching the pinned branch.
    Decode {
        /// Absolute token position of the one supplied decode vector.
        position: NonZeroUsize,
    },
}

impl WindowStep {
    fn input_positions(self) -> usize {
        match self {
            Self::Prefill { tokens } => tokens.get(),
            Self::Decode { .. } => 1,
        }
    }
}

/// Errors from the bounded raw-window reference.
#[derive(Clone, Copy, Debug, Error, Eq, PartialEq)]
#[non_exhaustive]
pub enum WindowError {
    /// A derived buffer length cannot be represented by `usize`.
    #[error("raw-window shape arithmetic overflowed for {field}")]
    ShapeOverflow {
        /// Derived buffer role.
        field: &'static str,
    },
    /// A bounded schedule or raw ring would exceed this reference's limit.
    #[error("raw-window {field} has {elements} elements, maximum is {maximum}")]
    TooLarge {
        /// Derived buffer role.
        field: &'static str,
        /// Requested element count.
        elements: usize,
        /// Reference-wide maximum element count.
        maximum: usize,
    },
    /// An input or caller-owned output has an unexpected exact length.
    #[error("raw-window {field} length is {actual}, expected {expected}")]
    Length {
        /// Buffer role.
        field: &'static str,
        /// Expected element count.
        expected: usize,
        /// Received element count.
        actual: usize,
    },
    /// A scheduled index cannot be represented by the pinned `i32` index type.
    #[error("raw-window scheduled position {position} does not fit i32")]
    IndexOutOfRange {
        /// Absolute prefill position.
        position: usize,
    },
    /// The schedule result allocation could not be reserved.
    #[error("could not allocate {elements} raw-window schedule entries")]
    AllocationFailed {
        /// Requested `i32` schedule entries.
        elements: usize,
    },
}

/// Schedules source-shaped window indices reused for every batch.
///
/// The result is contiguous `[batches, query_positions, slots]`. A prefill
/// has `query_positions = tokens` and `slots = min(tokens, window)`; a decode
/// has one query and `slots = window`. Prefill entries index the current chunk,
/// whereas decode entries index ring slots. `-1` marks an unavailable source
/// position: right padding during prefill and a leading unavailable ring slot
/// during early decode.
///
/// # Errors
///
/// Returns [`WindowError`] when the derived schedule is over the fixed bound,
/// cannot be represented, or cannot be allocated.
pub fn window_topk_indices(
    step: WindowStep,
    window: NonZeroUsize,
    batches: NonZeroUsize,
) -> Result<Vec<i32>, WindowError> {
    let shape = ScheduleShape::new(step, window, batches)?;
    let mut indices = Vec::new();
    indices
        .try_reserve_exact(shape.elements)
        .map_err(|_| WindowError::AllocationFailed {
            elements: shape.elements,
        })?;
    indices.resize(shape.elements, -1);

    match step {
        WindowStep::Prefill { tokens } => {
            let tokens = tokens.get();
            for batch in 0..shape.batches {
                for query in 0..tokens {
                    let first = query.saturating_add(1).saturating_sub(shape.window);
                    let destination = (batch * tokens + query) * shape.slots;
                    for slot in 0..shape.slots {
                        let position = first + slot;
                        if position <= query {
                            indices[destination + slot] = i32::try_from(position)
                                .map_err(|_| WindowError::IndexOutOfRange { position })?;
                        }
                    }
                }
            }
        }
        WindowStep::Decode { position } => {
            let position = position.get();
            let oldest =
                (position % shape.window)
                    .checked_add(1)
                    .ok_or(WindowError::ShapeOverflow {
                        field: "decode oldest",
                    })?;
            let suffix = shape.window - oldest;
            for batch in 0..shape.batches {
                let destination = batch * shape.slots;
                for slot in 0..shape.slots {
                    let ring_slot = if slot < suffix {
                        oldest + slot
                    } else {
                        slot - suffix
                    };
                    if ring_slot <= position {
                        indices[destination + slot] =
                            i32::try_from(ring_slot).map_err(|_| WindowError::IndexOutOfRange {
                                position: ring_slot,
                            })?;
                    }
                }
            }
        }
    }
    Ok(indices)
}

/// Copies prepared raw BF16 vectors into a caller-owned window ring.
///
/// `prepared_kv` is `[batches, tokens, width]` for prefill or
/// `[batches, width]` for decode. `ring` is always `[batches, window, width]`.
/// A long prefill copies only its final `window` vectors to slots
/// `absolute_position % window`; a short prefill leaves later ring slots
/// untouched, so its paired schedule masks them with `-1`.
///
/// BF16 values are copied as opaque `u16` storage. The caller owns cache
/// initialization, inter-call continuity, and all preparation/quantization.
///
/// # Errors
///
/// Returns [`WindowError`] before writing when either exact buffer length or a
/// bounded shape is invalid. A rejection leaves `ring` unchanged.
pub fn write_window_kv_bf16(
    step: WindowStep,
    prepared_kv: &[u16],
    batches: NonZeroUsize,
    window: NonZeroUsize,
    width: NonZeroUsize,
    ring: &mut [u16],
) -> Result<(), WindowError> {
    let shape = CopyShape::new(step, batches, window, width)?;
    shape.validate_lengths(prepared_kv, ring)?;

    match step {
        WindowStep::Prefill { tokens } => {
            let tokens = tokens.get();
            let first = tokens.saturating_sub(shape.window);
            for batch in 0..shape.batches {
                for position in first..tokens {
                    let source = ((batch * tokens) + position) * shape.width;
                    let slot = position % shape.window;
                    let destination = ((batch * shape.window) + slot) * shape.width;
                    ring[destination..destination + shape.width]
                        .copy_from_slice(&prepared_kv[source..source + shape.width]);
                }
            }
        }
        WindowStep::Decode { position } => {
            let slot = position.get() % shape.window;
            for batch in 0..shape.batches {
                let source = batch * shape.width;
                let destination = ((batch * shape.window) + slot) * shape.width;
                ring[destination..destination + shape.width]
                    .copy_from_slice(&prepared_kv[source..source + shape.width]);
            }
        }
    }
    Ok(())
}

#[derive(Clone, Copy)]
struct ScheduleShape {
    batches: usize,
    window: usize,
    slots: usize,
    elements: usize,
}

impl ScheduleShape {
    fn new(
        step: WindowStep,
        window: NonZeroUsize,
        batches: NonZeroUsize,
    ) -> Result<Self, WindowError> {
        let window = window.get();
        let slots = match step {
            WindowStep::Prefill { tokens } => tokens.get().min(window),
            WindowStep::Decode { .. } => window,
        };
        let elements = product(&[batches.get(), step.input_positions(), slots], "schedule")?;
        enforce_bound("schedule", elements)?;
        Ok(Self {
            batches: batches.get(),
            window,
            slots,
            elements,
        })
    }
}

#[derive(Clone, Copy)]
struct CopyShape {
    batches: usize,
    window: usize,
    width: usize,
    input_elements: usize,
    ring_elements: usize,
}

impl CopyShape {
    fn new(
        step: WindowStep,
        batches: NonZeroUsize,
        window: NonZeroUsize,
        width: NonZeroUsize,
    ) -> Result<Self, WindowError> {
        let input_positions = step.input_positions();
        let input_elements = product(
            &[batches.get(), input_positions, width.get()],
            "prepared_kv",
        )?;
        let ring_elements = product(&[batches.get(), window.get(), width.get()], "ring")?;
        enforce_bound("prepared_kv", input_elements)?;
        enforce_bound("ring", ring_elements)?;
        Ok(Self {
            batches: batches.get(),
            window: window.get(),
            width: width.get(),
            input_elements,
            ring_elements,
        })
    }

    fn validate_lengths(self, prepared_kv: &[u16], ring: &[u16]) -> Result<(), WindowError> {
        check_length("prepared_kv", self.input_elements, prepared_kv.len())?;
        check_length("ring", self.ring_elements, ring.len())
    }
}

fn product(values: &[usize], field: &'static str) -> Result<usize, WindowError> {
    values.iter().try_fold(1_usize, |product, value| {
        product
            .checked_mul(*value)
            .ok_or(WindowError::ShapeOverflow { field })
    })
}

fn enforce_bound(field: &'static str, elements: usize) -> Result<(), WindowError> {
    if elements > MAX_WINDOW_ELEMENTS {
        return Err(WindowError::TooLarge {
            field,
            elements,
            maximum: MAX_WINDOW_ELEMENTS,
        });
    }
    Ok(())
}

fn check_length(field: &'static str, expected: usize, actual: usize) -> Result<(), WindowError> {
    if actual != expected {
        return Err(WindowError::Length {
            field,
            expected,
            actual,
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroUsize;

    use super::{
        MAX_WINDOW_ELEMENTS, WindowError, WindowStep, window_topk_indices, write_window_kv_bf16,
    };

    fn nz(value: usize) -> NonZeroUsize {
        NonZeroUsize::new(value).expect("test dimensions are nonzero")
    }

    #[test]
    fn schedules_source_right_padded_prefill_and_ring_ordered_decode() {
        assert_eq!(
            window_topk_indices(WindowStep::Prefill { tokens: nz(4) }, nz(4), nz(1))
                .expect("bounded prefill"),
            vec![0, -1, -1, -1, 0, 1, -1, -1, 0, 1, 2, -1, 0, 1, 2, 3]
        );
        assert_eq!(
            window_topk_indices(WindowStep::Decode { position: nz(2) }, nz(4), nz(1))
                .expect("bounded decode"),
            vec![-1, 0, 1, 2]
        );
        assert_eq!(
            window_topk_indices(WindowStep::Decode { position: nz(6) }, nz(4), nz(1))
                .expect("bounded decode"),
            vec![3, 0, 1, 2]
        );
        assert_eq!(
            window_topk_indices(WindowStep::Decode { position: nz(7) }, nz(4), nz(1))
                .expect("bounded decode"),
            vec![0, 1, 2, 3]
        );
    }

    #[test]
    fn prefill_seeds_ring_then_decode_overwrites_absolute_slots() {
        assert_eq!(
            window_topk_indices(WindowStep::Prefill { tokens: nz(6) }, nz(4), nz(1))
                .expect("long prefill indexes the chunk, not the ring"),
            vec![
                0, -1, -1, -1, 0, 1, -1, -1, 0, 1, 2, -1, 0, 1, 2, 3, 1, 2, 3, 4, 2, 3, 4, 5
            ]
        );
        let mut ring = [99_u16; 4];
        write_window_kv_bf16(
            WindowStep::Prefill { tokens: nz(6) },
            &[0, 1, 2, 3, 4, 5],
            nz(1),
            nz(4),
            nz(1),
            &mut ring,
        )
        .expect("bounded long prefill");
        assert_eq!(ring, [4, 5, 2, 3]);
        write_window_kv_bf16(
            WindowStep::Decode { position: nz(6) },
            &[6],
            nz(1),
            nz(4),
            nz(1),
            &mut ring,
        )
        .expect("bounded decode");
        assert_eq!(ring, [4, 5, 6, 3]);
        write_window_kv_bf16(
            WindowStep::Decode { position: nz(7) },
            &[7],
            nz(1),
            nz(4),
            nz(1),
            &mut ring,
        )
        .expect("bounded decode");
        assert_eq!(ring, [4, 5, 6, 7]);
    }

    #[test]
    fn short_prefill_leaves_stale_slots_masked_and_batches_keep_width_rows() {
        let mut ring = [99_u16; 4];
        write_window_kv_bf16(
            WindowStep::Prefill { tokens: nz(2) },
            &[0, 1],
            nz(1),
            nz(4),
            nz(1),
            &mut ring,
        )
        .expect("bounded short prefill");
        assert_eq!(ring, [0, 1, 99, 99]);
        assert_eq!(
            window_topk_indices(WindowStep::Prefill { tokens: nz(2) }, nz(4), nz(1))
                .expect("bounded short prefill schedule"),
            vec![0, -1, 0, 1]
        );

        let mut batched_ring = [99_u16; 8];
        write_window_kv_bf16(
            WindowStep::Prefill { tokens: nz(3) },
            &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
            nz(2),
            nz(2),
            nz(2),
            &mut batched_ring,
        )
        .expect("bounded batched prefill");
        assert_eq!(batched_ring, [4, 5, 2, 3, 10, 11, 8, 9]);
        let decode = WindowStep::Decode { position: nz(3) };
        write_window_kv_bf16(
            decode,
            &[12, 13, 14, 15],
            nz(2),
            nz(2),
            nz(2),
            &mut batched_ring,
        )
        .expect("batched decode preserves row boundaries");
        assert_eq!(batched_ring, [4, 5, 12, 13, 10, 11, 14, 15]);
        assert_eq!(
            window_topk_indices(decode, nz(2), nz(2)).unwrap(),
            vec![0, 1, 0, 1]
        );
    }

    #[test]
    fn rejects_malformed_or_oversized_geometry_without_writing_ring() {
        let mut ring = [7_u16; 4];
        let before = ring;
        assert_eq!(
            write_window_kv_bf16(
                WindowStep::Prefill { tokens: nz(2) },
                &[0, 1, 2],
                nz(1),
                nz(4),
                nz(2),
                &mut ring,
            ),
            Err(WindowError::Length {
                field: "prepared_kv",
                expected: 4,
                actual: 3,
            })
        );
        assert_eq!(ring, before);
        assert_eq!(
            write_window_kv_bf16(
                WindowStep::Decode { position: nz(1) },
                &[],
                nz(usize::MAX),
                nz(2),
                nz(2),
                &mut ring,
            ),
            Err(WindowError::ShapeOverflow {
                field: "prepared_kv"
            })
        );
        assert_eq!(ring, before);
        assert_eq!(
            window_topk_indices(WindowStep::Prefill { tokens: nz(2) }, nz(2), nz(usize::MAX)),
            Err(WindowError::ShapeOverflow { field: "schedule" })
        );
        let mut short_ring = [7_u16; 3];
        let short_before = short_ring;
        assert_eq!(
            write_window_kv_bf16(
                WindowStep::Decode { position: nz(1) },
                &[0, 1],
                nz(1),
                nz(2),
                nz(2),
                &mut short_ring,
            ),
            Err(WindowError::Length {
                field: "ring",
                expected: 4,
                actual: 3,
            })
        );
        assert_eq!(short_ring, short_before);
        assert_eq!(
            window_topk_indices(
                WindowStep::Decode { position: nz(1) },
                nz(MAX_WINDOW_ELEMENTS + 1),
                nz(1),
            ),
            Err(WindowError::TooLarge {
                field: "schedule",
                elements: MAX_WINDOW_ELEMENTS + 1,
                maximum: MAX_WINDOW_ELEMENTS,
            })
        );
    }

    #[test]
    fn one_slot_decode_avoids_absolute_position_addition() {
        assert_eq!(
            window_topk_indices(
                WindowStep::Decode {
                    position: nz(usize::MAX),
                },
                nz(1),
                nz(1),
            )
            .expect("single-slot modulo schedule"),
            vec![0]
        );
    }
}
