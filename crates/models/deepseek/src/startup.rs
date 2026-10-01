//! Bounded source-shaped token embedding and initial Hyper-Connections expansion.
//!
//! The pinned text graph looks up BF16 token rows, repeats each row into every
//! Hyper-Connections copy, and starts the first block with an identity pre-mix.
//! This module qualifies only that startup boundary; it does not model images,
//! distributed vocabulary shards, tokenizer behavior, or transformer execution.

use thiserror::Error;

const MAX_STARTUP_ELEMENTS: usize = 1 << 20;
const MAX_STARTUP_COPIES: usize = 16;

/// Validated dimensions for a bounded unsharded token embedding startup.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct StartupLayout {
    rows: usize,
    width: usize,
    copies: usize,
    table_elements: usize,
}

impl StartupLayout {
    /// Validates the table and Hyper-Connections startup dimensions.
    pub fn new(rows: usize, width: usize, copies: usize) -> Result<Self, StartupError> {
        if rows == 0 || width == 0 || copies == 0 {
            return Err(StartupError::EmptyDimension);
        }
        if copies > MAX_STARTUP_COPIES {
            return Err(StartupError::CopyCountTooLarge { copies });
        }
        let table_elements = bounded_product(rows, width)?;
        Ok(Self {
            rows,
            width,
            copies,
            table_elements,
        })
    }
}

/// Source-shaped initial residual and pre-mix supplied to block zero.
#[derive(Clone, Debug, PartialEq)]
pub struct StartupOutput {
    residual_bf16: Vec<u16>,
    identity_pre: Vec<f32>,
}

impl StartupOutput {
    /// Returns copy-major BF16 residual rows `[token, copy, width]`.
    #[must_use]
    pub fn residual_bf16(&self) -> &[u16] {
        &self.residual_bf16
    }

    /// Returns the source identity pre-mix `[token, copy]`.
    #[must_use]
    pub fn identity_pre(&self) -> &[f32] {
        &self.identity_pre
    }
}

/// Invalid startup geometry, embedding storage, or token selection.
#[derive(Clone, Debug, Error, Eq, PartialEq)]
#[non_exhaustive]
pub enum StartupError {
    /// A table dimension or Hyper-Connections copy count is zero.
    #[error("startup dimensions must be nonzero")]
    EmptyDimension,
    /// The requested copy count exceeds this scalar reference bound.
    #[error("startup copy count {copies} exceeds maximum {MAX_STARTUP_COPIES}")]
    CopyCountTooLarge { copies: usize },
    /// A derived shape overflowed or exceeded the scalar reference cap.
    #[error("startup {field} exceeds the bounded scalar reference")]
    ElementLimit { field: &'static str },
    /// The BF16 table storage does not match the validated layout.
    #[error("startup embedding table length {actual}, expected {expected}")]
    TableLength { actual: usize, expected: usize },
    /// A token ID cannot select a row in the supplied table.
    #[error("startup token ID {id} is outside table rows {rows}")]
    TokenOutOfRange { id: u64, rows: usize },
    /// The original-token-ID map does not match the supplied row count.
    #[error("startup selected token ID count {actual}, expected {expected}")]
    SelectedTokenCount { actual: usize, expected: usize },
    /// The selected original token IDs are not strictly increasing.
    #[error("startup selected token IDs must be strictly increasing at index {index}")]
    SelectedTokenOrder { index: usize },
    /// An original token ID has no row in the selected embedding storage.
    #[error("startup token ID {id} has no selected embedding row")]
    MissingSelectedToken { id: u64 },
    /// A selected BF16 table element is nonfinite.
    #[error("startup embedding row {row}, feature {feature} is nonfinite")]
    NonFiniteEmbedding { row: usize, feature: usize },
    /// A bounded output allocation failed.
    #[error("startup allocation failed for {field} with {elements} elements")]
    AllocationFailed {
        field: &'static str,
        elements: usize,
    },
}

fn bounded_product(left: usize, right: usize) -> Result<usize, StartupError> {
    left.checked_mul(right)
        .filter(|&elements| elements <= MAX_STARTUP_ELEMENTS)
        .ok_or(StartupError::ElementLimit { field: "shape" })
}

fn finite_bf16(bits: u16) -> bool {
    bits & 0x7f80 != 0x7f80
}

/// Looks up BF16 token rows, repeats them into HC copies, and creates identity pre-mix.
///
/// The table is `[rows, width]`; output residual is `[ids.len(), copies, width]`.
/// Selected IDs and BF16 values are validated before allocation. This is the
/// single-rank, text-only source startup equation.
pub fn startup_bf16_reference(
    ids: &[u64],
    embedding_table_bf16: &[u16],
    layout: StartupLayout,
) -> Result<StartupOutput, StartupError> {
    if embedding_table_bf16.len() != layout.table_elements {
        return Err(StartupError::TableLength {
            actual: embedding_table_bf16.len(),
            expected: layout.table_elements,
        });
    }
    let token_width = bounded_product(ids.len(), layout.width)?;
    let residual_elements = bounded_product(token_width, layout.copies)?;
    let pre_elements = bounded_product(ids.len(), layout.copies)?;
    for &id in ids {
        let row = usize::try_from(id).map_err(|_| StartupError::TokenOutOfRange {
            id,
            rows: layout.rows,
        })?;
        if row >= layout.rows {
            return Err(StartupError::TokenOutOfRange {
                id,
                rows: layout.rows,
            });
        }
        for (feature, &bits) in embedding_table_bf16[row * layout.width..(row + 1) * layout.width]
            .iter()
            .enumerate()
        {
            if !finite_bf16(bits) {
                return Err(StartupError::NonFiniteEmbedding { row, feature });
            }
        }
    }
    let mut residual_bf16 = Vec::new();
    residual_bf16
        .try_reserve_exact(residual_elements)
        .map_err(|_| StartupError::AllocationFailed {
            field: "residual",
            elements: residual_elements,
        })?;
    let mut identity_pre = Vec::new();
    identity_pre
        .try_reserve_exact(pre_elements)
        .map_err(|_| StartupError::AllocationFailed {
            field: "identity pre-mix",
            elements: pre_elements,
        })?;
    for &id in ids {
        let row = usize::try_from(id).map_err(|_| StartupError::TokenOutOfRange {
            id,
            rows: layout.rows,
        })?;
        let token = &embedding_table_bf16[row * layout.width..(row + 1) * layout.width];
        for _ in 0..layout.copies {
            residual_bf16.extend_from_slice(token);
        }
        identity_pre.push(1.0);
        identity_pre.extend(std::iter::repeat_n(0.0, layout.copies - 1));
    }
    Ok(StartupOutput {
        residual_bf16,
        identity_pre,
    })
}

/// Looks up original token IDs in selected BF16 rows and performs the dense startup equation.
///
/// `selected_token_ids` must contain exactly `layout.rows` strictly increasing
/// original IDs, corresponding to the rows of `embedding_rows_bf16`. Requested
/// IDs may repeat; a missing original ID is never treated as a storage offset.
/// Shape limits, requested IDs and requested BF16 rows are validated before
/// allocating the bounded slot map. Unrequested nonfinite rows are allowed,
/// matching [`startup_bf16_reference`].
pub fn startup_selected_bf16_reference(
    ids: &[u64],
    selected_token_ids: &[u64],
    embedding_rows_bf16: &[u16],
    layout: StartupLayout,
) -> Result<StartupOutput, StartupError> {
    if selected_token_ids.len() != layout.rows {
        return Err(StartupError::SelectedTokenCount {
            actual: selected_token_ids.len(),
            expected: layout.rows,
        });
    }
    for (index, pair) in selected_token_ids.windows(2).enumerate() {
        if pair[0] >= pair[1] {
            return Err(StartupError::SelectedTokenOrder { index: index + 1 });
        }
    }
    if embedding_rows_bf16.len() != layout.table_elements {
        return Err(StartupError::TableLength {
            actual: embedding_rows_bf16.len(),
            expected: layout.table_elements,
        });
    }
    let token_width = bounded_product(ids.len(), layout.width)?;
    bounded_product(token_width, layout.copies)?;
    bounded_product(ids.len(), layout.copies)?;
    let selected_row = |id| {
        selected_token_ids
            .binary_search(&id)
            .map_err(|_| StartupError::MissingSelectedToken { id })
    };
    for &id in ids {
        let row = selected_row(id)?;
        for (feature, &bits) in embedding_rows_bf16[row * layout.width..(row + 1) * layout.width]
            .iter()
            .enumerate()
        {
            if !finite_bf16(bits) {
                return Err(StartupError::NonFiniteEmbedding { row, feature });
            }
        }
    }
    let mut slots = Vec::new();
    slots
        .try_reserve_exact(ids.len())
        .map_err(|_| StartupError::AllocationFailed {
            field: "selected token slots",
            elements: ids.len(),
        })?;
    for &id in ids {
        let slot = u64::try_from(selected_row(id)?).map_err(|_| StartupError::ElementLimit {
            field: "selected token slots",
        })?;
        slots.push(slot);
    }
    startup_bf16_reference(&slots, embedding_rows_bf16, layout)
}

#[cfg(test)]
mod tests {
    use super::{
        StartupError, StartupLayout, startup_bf16_reference, startup_selected_bf16_reference,
    };

    #[test]
    fn startup_repeats_selected_rows_and_builds_identity_pre_mix() {
        let output = startup_bf16_reference(
            &[1, 0],
            &[0x3f80, 0x4000, 0x4040, 0x4080],
            StartupLayout::new(2, 2, 2).unwrap(),
        )
        .unwrap();
        assert_eq!(
            output.residual_bf16(),
            &[
                0x4040, 0x4080, 0x4040, 0x4080, 0x3f80, 0x4000, 0x3f80, 0x4000
            ]
        );
        assert_eq!(output.identity_pre(), &[1.0, 0.0, 1.0, 0.0]);
    }

    #[test]
    fn startup_rejects_invalid_table_token_and_selected_nonfinite_value() {
        let layout = StartupLayout::new(2, 2, 2).unwrap();
        assert_eq!(
            startup_bf16_reference(&[0], &[0x3f80; 3], layout),
            Err(StartupError::TableLength {
                actual: 3,
                expected: 4
            })
        );
        assert_eq!(
            startup_bf16_reference(&[2], &[0x3f80; 4], layout),
            Err(StartupError::TokenOutOfRange { id: 2, rows: 2 })
        );
        assert_eq!(
            startup_bf16_reference(&[1], &[0x3f80, 0x3f80, 0x7f80, 0x3f80], layout),
            Err(StartupError::NonFiniteEmbedding { row: 1, feature: 0 })
        );
    }

    #[test]
    fn selected_startup_matches_dense_for_repeated_original_ids() {
        let mut dense_rows = vec![0; 24];
        dense_rows[10..12].copy_from_slice(&[0x3f80, 0x4000]);
        dense_rows[22..24].copy_from_slice(&[0xbf80, 0x4080]);
        let ids = [11, 5, 11];
        let dense =
            startup_bf16_reference(&ids, &dense_rows, StartupLayout::new(12, 2, 3).unwrap())
                .unwrap();
        let selected = startup_selected_bf16_reference(
            &ids,
            &[5, 11],
            &[0x3f80, 0x4000, 0xbf80, 0x4080],
            StartupLayout::new(2, 2, 3).unwrap(),
        )
        .unwrap();
        assert_eq!(selected, dense);
        assert_eq!(
            selected.identity_pre(),
            &[1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0]
        );
    }

    #[test]
    fn selected_startup_keeps_original_ids_distinct_from_storage_slots() {
        let layout = StartupLayout::new(2, 1, 2).unwrap();
        for id in [0, 1, 6] {
            assert_eq!(
                startup_selected_bf16_reference(&[id], &[5, u64::MAX], &[0x3f80, 0x4000], layout,),
                Err(StartupError::MissingSelectedToken { id })
            );
        }
        assert_eq!(
            startup_selected_bf16_reference(&[u64::MAX], &[5, u64::MAX], &[0x3f80, 0x4000], layout,),
            startup_bf16_reference(&[1], &[0x3f80, 0x4000], layout)
        );
    }

    #[test]
    fn selected_startup_rejects_invalid_map_storage_and_output_geometry() {
        let layout = StartupLayout::new(2, 1, 2).unwrap();
        assert_eq!(
            startup_selected_bf16_reference(&[5], &[5], &[0x3f80, 0x4000], layout),
            Err(StartupError::SelectedTokenCount {
                actual: 1,
                expected: 2,
            })
        );
        for selected_ids in [[11, 5], [5, 5]] {
            assert_eq!(
                startup_selected_bf16_reference(&[5], &selected_ids, &[0x3f80, 0x4000], layout),
                Err(StartupError::SelectedTokenOrder { index: 1 })
            );
        }
        assert_eq!(
            startup_selected_bf16_reference(&[5], &[5, 11], &[0x3f80], layout),
            Err(StartupError::TableLength {
                actual: 1,
                expected: 2,
            })
        );
        assert_eq!(
            startup_selected_bf16_reference(
                &[5; 65],
                &[5],
                &[0x3f80; 1024],
                StartupLayout::new(1, 1024, 16).unwrap(),
            ),
            Err(StartupError::ElementLimit { field: "shape" })
        );
    }

    #[test]
    fn selected_startup_matches_dense_nonfinite_and_empty_request_policy() {
        let layout = StartupLayout::new(2, 1, 2).unwrap();
        let rows = [0x3f80, 0x7f80];
        assert_eq!(
            startup_selected_bf16_reference(&[5], &[5, 99], &rows, layout),
            startup_bf16_reference(&[0], &rows, layout)
        );
        assert_eq!(
            startup_selected_bf16_reference(&[99], &[5, 99], &rows, layout),
            Err(StartupError::NonFiniteEmbedding { row: 1, feature: 0 })
        );
        assert_eq!(
            startup_selected_bf16_reference(&[], &[5, 99], &rows, layout),
            startup_bf16_reference(&[], &rows, layout)
        );
    }
}
