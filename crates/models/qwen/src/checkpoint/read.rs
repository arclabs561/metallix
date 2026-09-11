//! Adapter-private exact payload reads for already-inspected Qwen checkpoints.
//!
//! The immutable-input contract is intentionally narrow: before and after each
//! read, the loader compares shard length and modification time to inspection.
//! That detects ordinary replacement/truncation but is not a security boundary
//! against a concurrent in-place write that preserves both values.

use std::{
    fs::File,
    io::{Read, Seek, SeekFrom},
    ops::Range,
};

use super::{
    Qwen3CheckpointError, Qwen3CheckpointInspection, Qwen3TensorBytes, TensorLocation,
    ensure_shard_identity,
};

impl Qwen3CheckpointInspection {
    /// Reads one validated tensor without loading its whole safetensors shard.
    ///
    /// `max_bytes` is checked before opening the shard or allocating. The
    /// returned bytes retain the dtype and shape recorded at inspection.
    pub(crate) fn read_tensor(
        &self,
        name: &str,
        max_bytes: u64,
    ) -> Result<Qwen3TensorBytes, Qwen3CheckpointError> {
        let location = self.tensor_location(name)?;
        let bytes = read_exact_range(
            location,
            name,
            location.range.file_offset,
            location.range.byte_length,
            max_bytes,
        )?;
        Ok(Qwen3TensorBytes {
            dtype: location.range.dtype.clone(),
            shape: location.range.shape.clone(),
            bytes,
        })
    }

    /// Reads a nonempty half-open row range from a rank-two BF16 matrix.
    ///
    /// The selected rows are contiguous in safetensors' row-major payload, so
    /// this allocates and reads only `rows.len() * columns * 2` bytes. As with
    /// [`Self::read_tensor`], `max_bytes` is enforced before file I/O and
    /// allocation.
    pub(crate) fn read_bf16_rows(
        &self,
        name: &str,
        rows: Range<usize>,
        max_bytes: u64,
    ) -> Result<Qwen3TensorBytes, Qwen3CheckpointError> {
        let location = self.tensor_location(name)?;
        if location.range.dtype != "BF16" {
            return Err(Qwen3CheckpointError::TensorRowsRequireBf16 {
                tensor: name.to_owned(),
                dtype: location.range.dtype.clone(),
            });
        }
        let [row_count, columns] = location.range.shape.as_slice() else {
            return Err(Qwen3CheckpointError::TensorRowsRequireMatrix {
                tensor: name.to_owned(),
                shape: location.range.shape.clone(),
            });
        };
        let start = u64::try_from(rows.start).map_err(|_| {
            Qwen3CheckpointError::TensorRowLayoutOverflow {
                tensor: name.to_owned(),
            }
        })?;
        let end =
            u64::try_from(rows.end).map_err(|_| Qwen3CheckpointError::TensorRowLayoutOverflow {
                tensor: name.to_owned(),
            })?;
        if start >= end || end > *row_count {
            return Err(Qwen3CheckpointError::InvalidTensorRowRange {
                tensor: name.to_owned(),
                start: rows.start,
                end: rows.end,
                row_count: *row_count,
            });
        }

        let row_bytes = columns.checked_mul(2).ok_or_else(|| {
            Qwen3CheckpointError::TensorRowLayoutOverflow {
                tensor: name.to_owned(),
            }
        })?;
        let selected_rows = end.checked_sub(start).ok_or_else(|| {
            Qwen3CheckpointError::TensorRowLayoutOverflow {
                tensor: name.to_owned(),
            }
        })?;
        let selected_bytes = selected_rows.checked_mul(row_bytes).ok_or_else(|| {
            Qwen3CheckpointError::TensorRowLayoutOverflow {
                tensor: name.to_owned(),
            }
        })?;
        let row_offset = start.checked_mul(row_bytes).ok_or_else(|| {
            Qwen3CheckpointError::TensorRowLayoutOverflow {
                tensor: name.to_owned(),
            }
        })?;
        let selected_end = row_offset.checked_add(selected_bytes).ok_or_else(|| {
            Qwen3CheckpointError::TensorRowLayoutOverflow {
                tensor: name.to_owned(),
            }
        })?;
        if selected_end > location.range.byte_length {
            return Err(Qwen3CheckpointError::TensorRowLayoutOverflow {
                tensor: name.to_owned(),
            });
        }
        let file_offset = location
            .range
            .file_offset
            .checked_add(row_offset)
            .ok_or_else(|| Qwen3CheckpointError::TensorRowLayoutOverflow {
                tensor: name.to_owned(),
            })?;

        let bytes = read_exact_range(location, name, file_offset, selected_bytes, max_bytes)?;
        Ok(Qwen3TensorBytes {
            dtype: location.range.dtype.clone(),
            shape: vec![selected_rows, *columns],
            bytes,
        })
    }

    fn tensor_location(&self, name: &str) -> Result<&TensorLocation, Qwen3CheckpointError> {
        self.tensors
            .get(name)
            .ok_or_else(|| Qwen3CheckpointError::UnknownTensor(name.to_owned()))
    }
}

fn read_exact_range(
    location: &TensorLocation,
    tensor: &str,
    file_offset: u64,
    byte_length: u64,
    max_bytes: u64,
) -> Result<Vec<u8>, Qwen3CheckpointError> {
    if byte_length > max_bytes {
        return Err(Qwen3CheckpointError::TensorExceedsReadBudget {
            tensor: tensor.to_owned(),
            tensor_bytes: byte_length,
            max_bytes,
        });
    }
    let allocation_bytes = usize::try_from(byte_length).map_err(|_| {
        Qwen3CheckpointError::ReadBudgetCannotFitAddressSpace {
            tensor: tensor.to_owned(),
            tensor_bytes: byte_length,
        }
    })?;
    let mut file =
        File::open(&location.shard).map_err(|source| Qwen3CheckpointError::OpenShard {
            path: location.shard.clone(),
            source,
        })?;
    ensure_shard_identity(&file, &location.shard, location.identity)?;
    file.seek(SeekFrom::Start(file_offset))
        .map_err(|source| Qwen3CheckpointError::ReadShard {
            path: location.shard.clone(),
            source,
        })?;
    let mut bytes = vec![0_u8; allocation_bytes];
    if let Err(source) = file.read_exact(&mut bytes) {
        if source.kind() == std::io::ErrorKind::UnexpectedEof {
            return Err(Qwen3CheckpointError::TensorPayloadTruncated {
                path: location.shard.clone(),
                tensor: tensor.to_owned(),
            });
        }
        return Err(Qwen3CheckpointError::ReadShard {
            path: location.shard.clone(),
            source,
        });
    }
    ensure_shard_identity(&file, &location.shard, location.identity)?;
    Ok(bytes)
}
