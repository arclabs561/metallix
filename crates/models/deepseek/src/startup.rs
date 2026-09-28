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

#[cfg(test)]
mod tests {
    use super::{StartupError, StartupLayout, startup_bf16_reference};

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
}
