//! CPU decoding primitives for MLX affine-quantized tensor rows.
//!
//! This is the first native bridge for the MLX checkpoint layout. It decodes
//! bounded rows only; it does not load a model or allocate a full tensor.

use std::{
    fs::File,
    io::{Read, Seek, SeekFrom},
    path::Path,
};

use super::{V41SafetensorsHeader, V41StorageDtype};
use crate::hc::{HcCoefficients, HcError, split_hc_coefficients};
use thiserror::Error;

/// A validated MLX affine row decoder error.
#[derive(Clone, Debug, Error, Eq, PartialEq)]
#[non_exhaustive]
pub enum MlxAffineRowError {
    /// Packed weight storage is not aligned to the requested bit width.
    #[error("packed weight storage does not match the requested bit width")]
    PackedShape,
    /// Scale and bias group counts do not match the logical row.
    #[error("scale/bias groups do not match the logical row")]
    GroupShape,
    /// The requested bit width or group size is unsupported.
    #[error("unsupported MLX affine layout: bits={bits}, group_size={group_size}")]
    UnsupportedLayout { bits: u8, group_size: usize },
    /// A decoded value is non-finite.
    #[error("MLX affine row produced a non-finite value at column {column}")]
    NonFinite { column: usize },
    /// A named tensor is absent from the validated shard header.
    #[error("MLX affine tensor {name:?} is missing from the shard header")]
    MissingTensor { name: String },
    /// A named tensor has an unexpected dtype, rank, or row geometry.
    #[error("MLX affine tensor {name:?} has an invalid shape or dtype")]
    TensorShape { name: String },
    /// The bounded row read failed.
    #[error("could not read MLX affine tensor bytes: {0}")]
    Io(String),
}

/// Reads and decodes one affine row from a validated safetensors shard.
#[allow(clippy::too_many_arguments)]
pub fn read_affine_row_from_shard(
    shard: &Path,
    header: &V41SafetensorsHeader,
    weight_name: &str,
    scale_name: &str,
    bias_name: &str,
    row: usize,
    logical_width: usize,
    bits: u8,
    group_size: usize,
) -> Result<Vec<f32>, MlxAffineRowError> {
    read_affine_rows_from_shard(
        shard,
        header,
        weight_name,
        scale_name,
        bias_name,
        row,
        1,
        logical_width,
        bits,
        group_size,
    )
}

/// Reads and decodes a contiguous bounded range of affine rows from one shard.
///
/// The file is opened once for the range so callers can stream a large matrix
/// in model-shaped chunks without paying one open/close cycle per row.
#[allow(clippy::too_many_arguments)]
pub fn read_affine_rows_from_shard(
    shard: &Path,
    header: &V41SafetensorsHeader,
    weight_name: &str,
    scale_name: &str,
    bias_name: &str,
    first_row: usize,
    row_count: usize,
    logical_width: usize,
    bits: u8,
    group_size: usize,
) -> Result<Vec<f32>, MlxAffineRowError> {
    let weight = header
        .tensor(weight_name)
        .ok_or_else(|| MlxAffineRowError::MissingTensor {
            name: weight_name.to_owned(),
        })?;
    let scales = header
        .tensor(scale_name)
        .ok_or_else(|| MlxAffineRowError::MissingTensor {
            name: scale_name.to_owned(),
        })?;
    let biases = header
        .tensor(bias_name)
        .ok_or_else(|| MlxAffineRowError::MissingTensor {
            name: bias_name.to_owned(),
        })?;
    let groups = logical_width.div_ceil(group_size);
    let packed_width = (logical_width * usize::from(bits.max(1))).div_ceil(32);
    // MLX keeps any leading dimensions (for example `wo_a` is
    // `[o_groups, rows, packed_width]`) while packing only the final logical
    // dimension. Treat the product of the leading dimensions as a flat row
    // stream, preserving the on-disk contiguous layout and allowing bounded
    // reads without materializing the whole tensor.
    let leading_shape = weight.shape().get(..weight.shape().len().saturating_sub(1));
    let scale_shape = scales.shape();
    let bias_shape = biases.shape();
    let expected_scale_shape = leading_shape.map(|leading| {
        leading
            .iter()
            .copied()
            .chain(std::iter::once(groups as u64))
            .collect::<Vec<_>>()
    });
    let rows = leading_shape.and_then(|leading| {
        leading.iter().try_fold(1_usize, |product, dimension| {
            usize::try_from(*dimension).ok()?.checked_mul(product)
        })
    });
    let valid = weight.dtype() == V41StorageDtype::U32
        && scales.dtype() == V41StorageDtype::Bf16
        && biases.dtype() == V41StorageDtype::Bf16
        && weight.shape().len() >= 2
        && weight.shape().last() == Some(&(packed_width as u64))
        && expected_scale_shape.as_deref() == Some(scale_shape)
        && scale_shape == bias_shape
        && rows
            .is_some_and(|rows| first_row <= rows && row_count <= rows.saturating_sub(first_row));
    if !valid {
        return Err(MlxAffineRowError::TensorShape {
            name: weight_name.to_owned(),
        });
    }
    let mut file = File::open(shard).map_err(|error| MlxAffineRowError::Io(error.to_string()))?;
    let packed_row_bytes = packed_width * 4;
    let scale_row_bytes = groups * 2;
    let bias_row_bytes = groups * 2;
    let packed_start = row_offset(weight.file_range().start, first_row, packed_row_bytes)?;
    let scale_start = row_offset(scales.file_range().start, first_row, scale_row_bytes)?;
    let bias_start = row_offset(biases.file_range().start, first_row, bias_row_bytes)?;
    // The requested rows are contiguous in each MLX tensor. Read each range in
    // one operation so a full projection pays three seeks instead of three per
    // row. This matters for layer-zero KV/WQ matrices on network-backed shards.
    let mut packed_bytes = vec![0_u8; row_count * packed_row_bytes];
    let mut scale_bytes = vec![0_u8; row_count * scale_row_bytes];
    let mut bias_bytes = vec![0_u8; row_count * bias_row_bytes];
    if row_count != 0 {
        read_range(&mut file, packed_start, &mut packed_bytes)?;
        read_range(&mut file, scale_start, &mut scale_bytes)?;
        read_range(&mut file, bias_start, &mut bias_bytes)?;
    }
    let mut decoded = Vec::with_capacity(row_count * logical_width);
    for row in 0..row_count {
        let packed_row = &packed_bytes[row * packed_row_bytes..(row + 1) * packed_row_bytes];
        let scale_row = &scale_bytes[row * scale_row_bytes..(row + 1) * scale_row_bytes];
        let bias_row = &bias_bytes[row * bias_row_bytes..(row + 1) * bias_row_bytes];
        let packed = packed_row
            .chunks_exact(4)
            .map(|bytes| u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
            .collect::<Vec<_>>();
        let scales = scale_row
            .chunks_exact(2)
            .map(|bytes| u16::from_le_bytes([bytes[0], bytes[1]]))
            .collect::<Vec<_>>();
        let biases = bias_row
            .chunks_exact(2)
            .map(|bytes| u16::from_le_bytes([bytes[0], bytes[1]]))
            .collect::<Vec<_>>();
        decoded.extend(decode_affine_row(
            &packed,
            &scales,
            &biases,
            logical_width,
            bits,
            group_size,
        )?);
    }
    Ok(decoded)
}

/// Reads a bounded row-major F32 tensor from a validated shard.
pub fn read_f32_tensor_from_shard(
    shard: &Path,
    header: &V41SafetensorsHeader,
    name: &str,
    rows: usize,
    width: usize,
) -> Result<Vec<f32>, MlxAffineRowError> {
    let tensor = header
        .tensor(name)
        .ok_or_else(|| MlxAffineRowError::MissingTensor {
            name: name.to_owned(),
        })?;
    let shape_matches = tensor.shape() == [rows as u64, width as u64]
        || (rows == 1 && tensor.shape() == [width as u64]);
    if tensor.dtype() != V41StorageDtype::F32
        || !shape_matches
        || tensor.byte_length() != (rows * width * 4) as u64
    {
        return Err(MlxAffineRowError::TensorShape {
            name: name.to_owned(),
        });
    }
    let mut file = File::open(shard).map_err(|error| MlxAffineRowError::Io(error.to_string()))?;
    let mut bytes = vec![0_u8; rows * width * 4];
    read_range(&mut file, tensor.file_range().start, &mut bytes)?;
    Ok(bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect())
}

/// Reads a bounded BF16 tensor and widens it to FP32.
pub fn read_bf16_tensor_from_shard(
    shard: &Path,
    header: &V41SafetensorsHeader,
    name: &str,
    width: usize,
) -> Result<Vec<f32>, MlxAffineRowError> {
    let tensor = header
        .tensor(name)
        .ok_or_else(|| MlxAffineRowError::MissingTensor {
            name: name.to_owned(),
        })?;
    if tensor.dtype() != V41StorageDtype::Bf16
        || (tensor.shape() != [width as u64] && tensor.shape() != [1, width as u64])
        || tensor.byte_length() != (width * 2) as u64
    {
        return Err(MlxAffineRowError::TensorShape {
            name: name.to_owned(),
        });
    }
    let mut file = File::open(shard).map_err(|error| MlxAffineRowError::Io(error.to_string()))?;
    let mut bytes = vec![0_u8; width * 2];
    read_range(&mut file, tensor.file_range().start, &mut bytes)?;
    Ok(bytes
        .chunks_exact(2)
        .map(|chunk| f32::from_bits(u32::from(u16::from_le_bytes([chunk[0], chunk[1]])) << 16))
        .collect())
}

fn row_offset(base: u64, first_row: usize, row_bytes: usize) -> Result<u64, MlxAffineRowError> {
    let first_row = u64::try_from(first_row)
        .map_err(|_| MlxAffineRowError::Io("row offset exceeds u64".to_owned()))?;
    let row_bytes = u64::try_from(row_bytes)
        .map_err(|_| MlxAffineRowError::Io("row byte size exceeds u64".to_owned()))?;
    base.checked_add(
        first_row.checked_mul(row_bytes).ok_or_else(|| {
            MlxAffineRowError::Io("row offset multiplication overflow".to_owned())
        })?,
    )
    .ok_or_else(|| MlxAffineRowError::Io("row offset addition overflow".to_owned()))
}

fn read_range(file: &mut File, offset: u64, bytes: &mut [u8]) -> Result<(), MlxAffineRowError> {
    file.seek(SeekFrom::Start(offset))
        .and_then(|_| file.read_exact(bytes))
        .map_err(|error| MlxAffineRowError::Io(error.to_string()))
}

/// Decodes one packed MLX affine row into FP32 values.
///
/// MLX stores affine codes as a contiguous little-endian bitstream. The
/// current `DeepSeek` embedding uses 8 bits/group 64; attention projections use
/// 6 bits/group 128. Each BF16 scale and bias pair covers one group.
pub fn decode_affine_row(
    packed: &[u32],
    scales_bf16: &[u16],
    biases_bf16: &[u16],
    logical_width: usize,
    bits: u8,
    group_size: usize,
) -> Result<Vec<f32>, MlxAffineRowError> {
    if !matches!(bits, 6 | 8) || group_size == 0 {
        return Err(MlxAffineRowError::UnsupportedLayout { bits, group_size });
    }
    let packed_bits = packed
        .len()
        .checked_mul(32)
        .ok_or(MlxAffineRowError::PackedShape)?;
    if packed_bits / usize::from(bits) != logical_width || packed_bits % usize::from(bits) != 0 {
        return Err(MlxAffineRowError::PackedShape);
    }
    let groups = logical_width.div_ceil(group_size);
    if scales_bf16.len() != groups || biases_bf16.len() != groups {
        return Err(MlxAffineRowError::GroupShape);
    }
    let mut output = Vec::with_capacity(logical_width);
    let packed_bytes = packed
        .iter()
        .flat_map(|word| word.to_le_bytes())
        .collect::<Vec<_>>();
    for column in 0..logical_width {
        let bit_offset = column * usize::from(bits);
        let byte_offset = bit_offset / 8;
        let intra_byte = bit_offset % 8;
        let code = if bits == 8 {
            packed_bytes[byte_offset]
        } else {
            // MLX stores six-bit codes as a contiguous little-endian bitstream:
            // four codes occupy three bytes and may straddle byte boundaries.
            let window = u32::from(packed_bytes[byte_offset])
                | (u32::from(*packed_bytes.get(byte_offset + 1).unwrap_or(&0)) << 8)
                | (u32::from(*packed_bytes.get(byte_offset + 2).unwrap_or(&0)) << 16);
            ((window >> intra_byte) & 0x3f) as u8
        };
        let group = column / group_size;
        let scale = f32::from_bits(u32::from(scales_bf16[group]) << 16);
        let offset = f32::from_bits(u32::from(biases_bf16[group]) << 16);
        let value = f32::from(code).mul_add(scale, offset);
        if !value.is_finite() {
            return Err(MlxAffineRowError::NonFinite { column });
        }
        output.push(value);
    }
    Ok(output)
}

/// Expands one hidden state into the repeated HC input layout.
pub fn expand_hc_hidden(input: &[f32], copies: usize) -> Result<Vec<f32>, MlxAffineRowError> {
    if input.is_empty() || copies == 0 {
        return Err(MlxAffineRowError::GroupShape);
    }
    let total = input
        .len()
        .checked_mul(copies)
        .ok_or(MlxAffineRowError::GroupShape)?;
    let mut expanded = Vec::with_capacity(total);
    for _ in 0..copies {
        expanded.extend_from_slice(input);
    }
    Ok(expanded)
}

/// Computes one Hyper-Connection coefficient row from decoded MLX parameters.
pub fn mix_hc_coefficients(
    fn_matrix: &[f32],
    base: &[f32],
    scale: &[f32; 3],
    hidden: &[f32],
    copies: usize,
    epsilon: f32,
    sinkhorn_iterations: usize,
) -> Result<HcCoefficients, HcError> {
    let expanded = expand_hc_hidden(hidden, copies).map_err(|_| HcError::ShapeOverflow {
        field: "expanded_hidden",
    })?;
    let width = expanded.len();
    let rows = (2 + copies) * copies;
    if fn_matrix.len() != rows * width {
        return Err(HcError::ShapeOverflow { field: "fn_matrix" });
    }
    let width_u16 = u16::try_from(expanded.len()).map_err(|_| HcError::ShapeOverflow {
        field: "expanded_hidden",
    })?;
    let norm = (expanded.iter().map(|value| value * value).sum::<f32>() / f32::from(width_u16)
        + epsilon)
        .sqrt();
    if !norm.is_finite() || norm <= 0.0 {
        return Err(HcError::NonFiniteInput {
            field: "hidden_norm",
            index: 0,
        });
    }
    let normalized = expanded
        .iter()
        .map(|value| *value / norm)
        .collect::<Vec<_>>();
    let mut mixes = Vec::with_capacity(rows);
    for row in fn_matrix.chunks_exact(width) {
        let value = row
            .iter()
            .zip(&normalized)
            .map(|(weight, input)| weight * input)
            .sum::<f32>();
        if !value.is_finite() {
            return Err(HcError::NonFiniteInput {
                field: "fn_mix",
                index: 0,
            });
        }
        mixes.push(value);
    }
    split_hc_coefficients(&mixes, scale, base, copies, sinkhorn_iterations, epsilon)
}

/// Collapses repeated hidden streams with real HC pre coefficients.
pub fn collapse_hc_hidden(
    hidden: &[f32],
    coefficients: &HcCoefficients,
) -> Result<Vec<f32>, HcError> {
    if hidden.is_empty() || coefficients.copies() == 0 {
        return Err(HcError::InvalidCopies {
            copies: coefficients.copies(),
            max_copies: 16,
        });
    }
    let mut output = vec![0.0; hidden.len()];
    for (index, value) in output.iter_mut().enumerate() {
        *value = coefficients
            .pre()
            .iter()
            .map(|coefficient| coefficient * hidden[index])
            .sum();
        if !value.is_finite() {
            return Err(HcError::NonFiniteInput {
                field: "hc_collapse",
                index,
            });
        }
    }
    Ok(output)
}

/// Converts one decoded row to an MLX array and evaluates it on the device.
#[cfg(feature = "metal")]
pub fn decode_affine_row_mlx(values: &[f32]) -> Result<mlx_rs::Array, mlx_rs::error::Exception> {
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_possible_wrap,
        reason = "embedding rows are bounded to model widths"
    )]
    let array = mlx_rs::Array::from_slice(values, &[values.len() as i32]);
    array.eval()?;
    Ok(array)
}

/// Applies a decoded row-major affine matrix to one hidden-state vector on MLX.
///
/// # Panics
///
/// Panics when the supplied matrix or input dimensions do not match the
/// declared `rows` and `width`.
#[cfg(feature = "metal")]
pub fn apply_affine_matrix_mlx(
    matrix: &[f32],
    rows: usize,
    width: usize,
    input: &[f32],
) -> Result<mlx_rs::Array, mlx_rs::error::Exception> {
    assert_eq!(matrix.len(), rows * width, "matrix shape");
    assert_eq!(input.len(), width, "input shape");
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_possible_wrap,
        reason = "model dimensions are bounded"
    )]
    let matrix = mlx_rs::Array::from_slice(matrix, &[rows as i32, width as i32]);
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_possible_wrap,
        reason = "model dimensions are bounded"
    )]
    let input = mlx_rs::Array::from_slice(input, &[width as i32, 1]);
    let output = mlx_rs::ops::matmul(&matrix, &input)?;
    output.eval()?;
    Ok(output)
}

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use std::{
        fmt::Write as _,
        fs::{self, File},
        io::Write as _,
        path::PathBuf,
        time::{SystemTime, UNIX_EPOCH},
    };

    use proptest::prelude::*;

    use super::{MlxAffineRowError, decode_affine_row, read_affine_rows_from_shard};
    use crate::checkpoint::V41SafetensorsHeader;

    #[test]
    fn reads_flattened_leading_dimensions_for_mlx_grouped_projection() {
        // `wo_a` uses [o_groups, rows, packed_width] in the real checkpoint.
        // A bounded read must flatten only the leading dimensions while keeping
        // each row's scale and bias group aligned with its packed payload.
        let logical_width = 128;
        let packed_width = 24; // 128 values * 6 bits / 32 bits
        let rows = 2 * 3;
        let weight_bytes = (0..rows)
            .flat_map(|row| {
                let mut words = vec![0_u32; packed_width];
                words[0] = u32::try_from(row + 1).expect("bounded row");
                words.into_iter().flat_map(u32::to_le_bytes)
            })
            .collect::<Vec<_>>();
        let scale_bytes = [0x80_u8, 0x3f_u8].repeat(rows);
        let bias_bytes = vec![0_u8; rows * 2];
        let header_json = format!(
            r#"{{"weight":{{"dtype":"U32","shape":[2,3,24],"data_offsets":[0,{weight}] }},"scales":{{"dtype":"BF16","shape":[2,3,1],"data_offsets":[{weight},{scale_end}] }},"biases":{{"dtype":"BF16","shape":[2,3,1],"data_offsets":[{scale_end},{payload_end}] }}}}"#,
            weight = weight_bytes.len(),
            scale_end = weight_bytes.len() + scale_bytes.len(),
            payload_end = weight_bytes.len() + scale_bytes.len() + bias_bytes.len(),
        );
        let header_len = u64::try_from(header_json.len()).expect("small test header");
        let mut file_bytes = Vec::with_capacity(
            8 + header_json.len() + weight_bytes.len() + scale_bytes.len() + bias_bytes.len(),
        );
        file_bytes.extend_from_slice(&header_len.to_le_bytes());
        file_bytes.extend_from_slice(header_json.as_bytes());
        file_bytes.extend_from_slice(&weight_bytes);
        file_bytes.extend_from_slice(&scale_bytes);
        file_bytes.extend_from_slice(&bias_bytes);
        // Header offsets are payload-relative; the parser turns them into
        // complete-file ranges.
        let path = PathBuf::from(format!(
            "/tmp/metallix-mlx-3d-{}.safetensors",
            std::process::id()
        ));
        fs::write(&path, &file_bytes).expect("write bounded fixture");
        let header_bytes = 8 + header_json.len();
        let header = V41SafetensorsHeader::parse_prefixed_header(
            &file_bytes[..header_bytes],
            file_bytes.len() as u64,
        )
        .expect("parse grouped fixture");
        let decoded = read_affine_rows_from_shard(
            &path,
            &header,
            "weight",
            "scales",
            "biases",
            1,
            3,
            logical_width,
            6,
            128,
        )
        .expect("decode flattened rows");
        fs::remove_file(&path).expect("remove bounded fixture");
        assert_eq!(decoded.len(), 3 * logical_width);
        assert_eq!(&decoded[..3], &[2.0, 0.0, 0.0]);
        assert_eq!(&decoded[logical_width..logical_width + 3], &[3.0, 0.0, 0.0]);
        assert_eq!(
            &decoded[2 * logical_width..2 * logical_width + 3],
            &[4.0, 0.0, 0.0]
        );
    }

    fn sample_resident() -> super::LayerZeroQkvResident {
        super::LayerZeroQkvResident {
            wq_a: vec![
                0.0;
                super::LayerZeroQkvResident::WQ_A_ROWS
                    * super::LayerZeroQkvResident::HIDDEN_WIDTH
            ],
            attn_norm: vec![1.0; super::LayerZeroQkvResident::HIDDEN_WIDTH],
            hc_fn: vec![1.0; 24 * 16_384],
            hc_base: vec![1.0; 24],
            hc_scale: vec![1.0; 3],
            q_norm: vec![1.0; super::LayerZeroQkvResident::Q_LORA_RANK],
            wkv: vec![
                0.0;
                super::LayerZeroQkvResident::WKV_ROWS
                    * super::LayerZeroQkvResident::HIDDEN_WIDTH
            ],
            kv_norm: vec![1.0; super::LayerZeroQkvResident::KV_LORA_RANK],
        }
    }

    #[test]
    fn decodes_packed_eight_bit_groups() {
        let packed = [0x0403_0201_u32, 0x0807_0605];
        let scale = 0x3f80_u16; // 1.0 in BF16
        let bias = 0x3f00_u16; // 0.5 in BF16
        let decoded = decode_affine_row(&packed, &[scale, scale], &[bias, bias], 8, 8, 4)
            .expect("affine row");
        assert_eq!(decoded, [1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5]);
    }

    #[test]
    fn decodes_six_bit_contiguous_groups() {
        let mut bytes = [0_u8; 96];
        for column in 0..128 {
            let code = u32::try_from(column % 64).unwrap();
            let bit_offset = column * 6;
            let byte_offset = bit_offset / 8;
            let shift = bit_offset % 8;
            let value = code << shift;
            bytes[byte_offset] |= u8::try_from(value & 0xff).unwrap();
            if shift > 2 {
                bytes[byte_offset + 1] |= u8::try_from((value >> 8) & 0xff).unwrap();
            }
            if shift > 10 {
                bytes[byte_offset + 2] |= u8::try_from((value >> 16) & 0xff).unwrap();
            }
        }
        let packed = bytes
            .chunks_exact(4)
            .map(|chunk| u32::from_le_bytes(chunk.try_into().expect("word")))
            .collect::<Vec<_>>();
        let decoded = super::decode_affine_row(&packed, &[0x3f80, 0x3f80], &[0, 0], 128, 6, 64)
            .expect("six-bit affine row");
        assert_eq!(decoded[0], 0.0);
        assert_eq!(decoded[63], 63.0);
        assert_eq!(decoded[64], 0.0);
        assert_eq!(decoded[127], 63.0);
    }

    #[test]
    fn rejects_malformed_rows_before_decoding() {
        assert_eq!(
            decode_affine_row(&[0], &[0], &[0], 4, 4, 4),
            Err(MlxAffineRowError::UnsupportedLayout {
                bits: 4,
                group_size: 4
            })
        );
        assert_eq!(
            decode_affine_row(&[0], &[0], &[0], 8, 8, 4),
            Err(MlxAffineRowError::PackedShape)
        );
        assert_eq!(
            decode_affine_row(&[0], &[0], &[], 4, 8, 4),
            Err(MlxAffineRowError::GroupShape)
        );
    }

    #[test]
    fn expands_hidden_state_for_hyper_connections() {
        assert_eq!(
            super::expand_hc_hidden(&[1.0, 2.0], 4).expect("expanded HC input"),
            [1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0]
        );
        assert_eq!(
            super::expand_hc_hidden(&[], 4),
            Err(MlxAffineRowError::GroupShape)
        );
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(64))]

        #[test]
        fn expansion_preserves_each_hidden_copy(
            input in prop::collection::vec(-100.0_f32..100.0, 1..32),
            copies in 1_usize..=8,
        ) {
            let expanded = super::expand_hc_hidden(&input, copies)
                .expect("positive copies and non-empty input");
            prop_assert_eq!(expanded.len(), input.len() * copies);
            for chunk in expanded.chunks_exact(input.len()) {
                prop_assert_eq!(chunk, input.as_slice());
            }
        }

        #[test]
        fn hc_pre_collapse_preserves_finite_shape(
            hidden in prop::collection::vec(-10.0_f32..10.0, 1..16),
            copies in 1_usize..=8,
            sinkhorn_iterations in 1_usize..=8,
        ) {
            let rows = (2 + copies) * copies;
            let fn_matrix = vec![0.01_f32; rows * hidden.len() * copies];
            let base = vec![0.0_f32; rows];
            let coefficients = super::mix_hc_coefficients(
                &fn_matrix,
                &base,
                &[1.0, 1.0, 1.0],
                &hidden,
                copies,
                1e-6,
                sinkhorn_iterations,
            )
            .expect("bounded finite HC coefficients");
            let collapsed = super::collapse_hc_hidden(&hidden, &coefficients)
                .expect("bounded finite HC collapse");
            prop_assert_eq!(coefficients.copies(), copies);
            prop_assert_eq!(collapsed.len(), hidden.len());
            prop_assert!(coefficients.pre().iter().all(|value| value.is_finite()));
            prop_assert!(coefficients.post().iter().all(|value| value.is_finite()));
            prop_assert!(coefficients.comb().iter().all(|value| value.is_finite()));
            prop_assert!(collapsed.iter().all(|value| value.is_finite()));
        }
    }

    #[test]
    fn mixes_hyperconnection_coefficients_from_decoded_parameters() {
        let coefficients = super::mix_hc_coefficients(
            &vec![0.01; 24 * 8],
            &[0.0; 24],
            &[1.0, 1.0, 1.0],
            &[1.0, 2.0],
            4,
            1e-6,
            2,
        )
        .expect("HC coefficients");
        assert_eq!(coefficients.copies(), 4);
        assert!(coefficients.pre().iter().all(|value| value.is_finite()));
        assert!(super::collapse_hc_hidden(&[], &coefficients).is_err());
        assert_eq!(
            super::collapse_hc_hidden(&[1.0, 2.0], &coefficients)
                .expect("HC collapse")
                .len(),
            2
        );
    }

    #[test]
    fn validates_resident_layer_zero_qkv_geometry_and_finiteness() {
        let resident = sample_resident();
        resident.validate().expect("resident geometry");

        let mut malformed = resident.clone();
        malformed.q_norm.pop();
        assert!(matches!(
            malformed.validate(),
            Err(MlxAffineRowError::TensorShape { name })
                if name == "layer-zero resident Q/KV tensors"
        ));

        let mut non_finite = resident;
        non_finite.kv_norm[0] = f32::NAN;
        assert!(matches!(
            non_finite.validate(),
            Err(MlxAffineRowError::TensorShape { name })
                if name == "layer-zero resident Q/KV tensors"
        ));

        let resident = sample_resident();
        let hidden = vec![0.0; super::LayerZeroQkvResident::HIDDEN_WIDTH];
        let prepared = resident
            .prepare_attention_hidden(&hidden, 1e-6, 1e-20, 4)
            .expect("zero HC activation");
        assert_eq!(prepared.len(), super::LayerZeroQkvResident::HIDDEN_WIDTH);
        assert!(prepared.iter().all(|value| *value == 0.0));
        let (q, kv) = resident
            .project_qkv(&prepared, 1e-20)
            .expect("zero activation");
        assert_eq!(q.len(), super::LayerZeroQkvResident::Q_LORA_RANK);
        assert_eq!(kv.len(), super::LayerZeroQkvResident::KV_LORA_RANK);
        assert!(q.iter().chain(&kv).all(|value| *value == 0.0));
        assert!(matches!(
            resident.project_qkv(&hidden[..hidden.len() - 1], 1e-6),
            Err(MlxAffineRowError::TensorShape { .. })
        ));
        assert!(matches!(
            resident.project_qkv(&hidden, 0.0),
            Err(MlxAffineRowError::TensorShape { .. })
        ));
        assert!(matches!(
            resident.prepare_attention_hidden(&hidden, 0.0, 1e-20, 4),
            Err(MlxAffineRowError::TensorShape { .. })
        ));
        assert!(matches!(
            resident.prepare_attention_hidden(&hidden, 1e-6, 0.0, 4),
            Err(MlxAffineRowError::TensorShape { .. })
        ));

        let mut weighted = resident.clone();
        weighted.wq_a[..super::LayerZeroQkvResident::HIDDEN_WIDTH].fill(1.0);
        weighted.wkv[..super::LayerZeroQkvResident::HIDDEN_WIDTH].fill(1.0);
        let nonzero_hidden = vec![1.0; super::LayerZeroQkvResident::HIDDEN_WIDTH];
        let (q_nonzero, kv_nonzero) = weighted
            .project_qkv(&nonzero_hidden, 1e-20)
            .expect("nonzero activation");
        assert!(q_nonzero.iter().all(|value| value.is_finite()));
        assert!(kv_nonzero.iter().all(|value| value.is_finite()));
        assert!(q_nonzero[0] > 0.0);
        assert!(kv_nonzero[0] > 0.0);
    }

    #[test]
    fn prepares_nonzero_attention_hidden_with_hc_and_norm_epsilons() {
        let mut resident = sample_resident();
        resident.attn_norm[0] = 2.0;
        resident.attn_norm[1] = 3.0;
        let mut hidden = vec![0.0; super::LayerZeroQkvResident::HIDDEN_WIDTH];
        hidden[0] = 3.0;
        hidden[1] = 4.0;

        let prepared = resident
            .prepare_attention_hidden(&hidden, 1e-6, 1e-20, 4)
            .expect("nonzero HC activation");

        assert!(prepared.iter().all(|value| value.is_finite()));
        assert!(prepared.iter().any(|value| *value != 0.0));
        assert!((prepared[0] - 76.8).abs() < 1e-3);
        assert!((prepared[1] - 153.6).abs() < 1e-3);
        assert_ne!(&prepared[..2], &hidden[..2]);
    }

    #[allow(
        clippy::too_many_lines,
        reason = "the synthetic shard keeps every tensor binding in one integration fixture"
    )]
    #[test]
    fn loads_resident_tensor_bindings_from_a_synthetic_shard() {
        let tensors = [
            (
                "model.layers.0.attn.wq_a.weight",
                "U32",
                vec![1024, 768],
                3_145_728,
            ),
            (
                "model.layers.0.attn.wq_a.scales",
                "BF16",
                vec![1024, 32],
                65_536,
            ),
            (
                "model.layers.0.attn.wq_a.biases",
                "BF16",
                vec![1024, 32],
                65_536,
            ),
            ("model.layers.0.attn_norm.weight", "BF16", vec![4096], 8_192),
            (
                "model.layers.0.attn_hc.fn",
                "F32",
                vec![24, 16_384],
                1_572_864,
            ),
            ("model.layers.0.attn_hc.base", "F32", vec![24], 96),
            ("model.layers.0.attn_hc.scale", "F32", vec![3], 12),
            (
                "model.layers.0.attn.q_norm.weight",
                "BF16",
                vec![1024],
                2_048,
            ),
            (
                "model.layers.0.attn.wkv.weight",
                "U32",
                vec![512, 768],
                1_572_864,
            ),
            (
                "model.layers.0.attn.wkv.scales",
                "BF16",
                vec![512, 32],
                32_768,
            ),
            (
                "model.layers.0.attn.wkv.biases",
                "BF16",
                vec![512, 32],
                32_768,
            ),
            (
                "model.layers.0.attn.kv_norm.weight",
                "BF16",
                vec![512],
                1_024,
            ),
        ];
        let mut offset = 0_u64;
        let mut header = String::from("{");
        for (index, (name, dtype, shape, bytes)) in tensors.iter().enumerate() {
            if index != 0 {
                header.push(',');
            }
            let shape = shape
                .iter()
                .map(usize::to_string)
                .collect::<Vec<_>>()
                .join(",");
            let end = offset + *bytes;
            write!(
                header,
                "\"{name}\":{{\"dtype\":\"{dtype}\",\"shape\":[{shape}],\"data_offsets\":[{offset},{end}]}}"
            )
            .expect("header entry");
            offset = end;
        }
        header.push('}');

        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock")
            .as_nanos();
        let path = std::env::temp_dir().join(format!("metallix-resident-{unique}.safetensors"));
        let mut file = File::create(&path).expect("synthetic shard");
        file.write_all(&(header.len() as u64).to_le_bytes())
            .expect("header length");
        file.write_all(header.as_bytes()).expect("header");
        file.write_all(&vec![0_u8; usize::try_from(offset).expect("payload size")])
            .expect("payload");
        drop(file);

        let metadata = [
            (header.len() as u64).to_le_bytes().as_slice(),
            header.as_bytes(),
        ]
        .concat();
        let parsed = super::super::V41SafetensorsHeader::parse_prefixed_header(
            &metadata,
            8 + header.len() as u64 + offset,
        )
        .expect("synthetic header");
        let resident = super::LayerZeroQkvResident::load(&path, &parsed).expect("resident load");
        assert_eq!(resident.attn_norm.len(), 4096);
        assert_eq!(resident.hc_fn.len(), 24 * 16_384);
        assert_eq!(resident.hc_base.len(), 24);
        assert_eq!(resident.hc_scale.len(), 3);
        assert_eq!(resident.wkv.len(), 512 * 4096);
        assert!(resident.validate().is_ok());
        fs::remove_file(path).expect("remove synthetic shard");
    }

    #[cfg(feature = "metal")]
    #[test]
    fn applies_decoded_matrix_to_hidden_state_on_mlx() {
        let output = super::apply_affine_matrix_mlx(&[1.0, 2.0, 3.0, 4.0], 2, 2, &[2.0, 3.0])
            .expect("matrix projection");
        assert_eq!(output.shape(), [2, 1]);
        assert_eq!(output.as_slice::<f32>(), [8.0, 18.0]);
    }
}

/// Resident layer-zero Q/KV tensors decoded from one MLX shard.
///
/// This is deliberately adapter-private in scope: it owns only the tensors
/// needed to prepare the first layer's query and compressed-KV state. It does
/// not imply that attention, logits, or a complete model are resident.
#[derive(Clone, Debug, PartialEq)]
pub struct LayerZeroQkvResident {
    /// Quantized `wq_a`, widened to FP32 row-major storage.
    pub wq_a: Vec<f32>,
    /// Learned layer-zero attention normalization weights, widened from BF16.
    pub attn_norm: Vec<f32>,
    /// Layer-zero hyper-connection coefficient projection, widened from F32.
    pub hc_fn: Vec<f32>,
    /// Layer-zero hyper-connection base and scale controls.
    pub hc_base: Vec<f32>,
    pub hc_scale: Vec<f32>,
    /// Learned Q normalization weights, widened from BF16.
    pub q_norm: Vec<f32>,
    /// Quantized compressed-KV projection, widened to FP32 row-major storage.
    pub wkv: Vec<f32>,
    /// Learned compressed-KV normalization weights, widened from BF16.
    pub kv_norm: Vec<f32>,
}

impl LayerZeroQkvResident {
    /// Hidden-state width consumed by the layer-zero projections.
    pub const HIDDEN_WIDTH: usize = 4096;
    /// Low-rank query width emitted by `wq_a`.
    pub const Q_LORA_RANK: usize = 1024;
    /// Compressed-KV width emitted by `wkv`.
    pub const KV_LORA_RANK: usize = 512;
    /// Number of rows in the full `wq_a` projection.
    pub const WQ_A_ROWS: usize = Self::Q_LORA_RANK;
    /// Number of rows in the compressed-KV projection.
    pub const WKV_ROWS: usize = Self::KV_LORA_RANK;

    /// Loads and validates the resident layer-zero Q/KV tensors.
    ///
    /// The local MLX artifact uses contiguous six-bit affine storage with
    /// group size 128 for these projections. The generic config's stale
    /// quantization fields are intentionally not consulted here; the shard
    /// header and this execution contract are the authority for the layout.
    ///
    /// # Errors
    ///
    /// Returns [`MlxAffineRowError`] when a required tensor is absent, has an
    /// unexpected shape or dtype, or cannot be read and decoded.
    pub fn load(shard: &Path, header: &V41SafetensorsHeader) -> Result<Self, MlxAffineRowError> {
        let wq_a = read_affine_rows_from_shard(
            shard,
            header,
            "model.layers.0.attn.wq_a.weight",
            "model.layers.0.attn.wq_a.scales",
            "model.layers.0.attn.wq_a.biases",
            0,
            Self::WQ_A_ROWS,
            Self::HIDDEN_WIDTH,
            6,
            128,
        )?;
        let attn_norm = read_bf16_tensor_from_shard(
            shard,
            header,
            "model.layers.0.attn_norm.weight",
            Self::HIDDEN_WIDTH,
        )?;
        let hc_fn =
            read_f32_tensor_from_shard(shard, header, "model.layers.0.attn_hc.fn", 24, 16_384)?;
        let hc_base =
            read_f32_tensor_from_shard(shard, header, "model.layers.0.attn_hc.base", 1, 24)?;
        let hc_scale =
            read_f32_tensor_from_shard(shard, header, "model.layers.0.attn_hc.scale", 1, 3)?;
        let q_norm = read_bf16_tensor_from_shard(
            shard,
            header,
            "model.layers.0.attn.q_norm.weight",
            Self::Q_LORA_RANK,
        )?;
        let wkv = read_affine_rows_from_shard(
            shard,
            header,
            "model.layers.0.attn.wkv.weight",
            "model.layers.0.attn.wkv.scales",
            "model.layers.0.attn.wkv.biases",
            0,
            Self::WKV_ROWS,
            Self::HIDDEN_WIDTH,
            6,
            128,
        )?;
        let kv_norm = read_bf16_tensor_from_shard(
            shard,
            header,
            "model.layers.0.attn.kv_norm.weight",
            Self::KV_LORA_RANK,
        )?;
        let resident = Self {
            wq_a,
            attn_norm,
            hc_fn,
            hc_base,
            hc_scale,
            q_norm,
            wkv,
            kv_norm,
        };
        resident.validate()?;
        Ok(resident)
    }

    /// Runs the resident HC-pre and attention-normalization boundary for one
    /// hidden state. The result is the 4,096-wide activation consumed by Q/KV.
    #[allow(clippy::cast_precision_loss, reason = "fixed model hidden width")]
    pub fn prepare_attention_hidden(
        &self,
        hidden: &[f32],
        hc_epsilon: f32,
        norm_epsilon: f32,
        sinkhorn_iterations: usize,
    ) -> Result<Vec<f32>, MlxAffineRowError> {
        self.validate()?;
        if hidden.len() != Self::HIDDEN_WIDTH
            || hidden.iter().any(|value| !value.is_finite())
            || !hc_epsilon.is_finite()
            || hc_epsilon <= 0.0
            || !norm_epsilon.is_finite()
            || norm_epsilon <= 0.0
        {
            return Err(MlxAffineRowError::TensorShape {
                name: "layer-zero hidden activation".to_owned(),
            });
        }
        let scale: [f32; 3] =
            self.hc_scale
                .as_slice()
                .try_into()
                .map_err(|_| MlxAffineRowError::TensorShape {
                    name: "attn_hc.scale".to_owned(),
                })?;
        let coefficients = mix_hc_coefficients(
            &self.hc_fn,
            &self.hc_base,
            &scale,
            hidden,
            4,
            hc_epsilon,
            sinkhorn_iterations,
        )
        .map_err(|_| MlxAffineRowError::TensorShape {
            name: "attn_hc coefficients".to_owned(),
        })?;
        let collapsed = collapse_hc_hidden(hidden, &coefficients).map_err(|_| {
            MlxAffineRowError::TensorShape {
                name: "attn_hc collapse".to_owned(),
            }
        })?;
        let norm = (collapsed.iter().map(|value| value * value).sum::<f32>()
            / Self::HIDDEN_WIDTH as f32
            + norm_epsilon)
            .sqrt();
        if !norm.is_finite() || norm <= 0.0 {
            return Err(MlxAffineRowError::NonFinite { column: 0 });
        }
        let output = collapsed
            .iter()
            .zip(&self.attn_norm)
            .map(|(value, weight)| value / norm * weight)
            .collect::<Vec<_>>();
        if output.iter().any(|value| !value.is_finite()) {
            return Err(MlxAffineRowError::NonFinite { column: 0 });
        }
        Ok(output)
    }

    /// Projects one finite hidden state through resident Q/KV weights and
    /// applies the learned low-rank RMS boundaries. This CPU path is a
    /// deterministic activation contract used before wiring the Metal graph.
    #[allow(
        clippy::items_after_statements,
        clippy::cast_precision_loss,
        reason = "the resident projection keeps its bounded helper local and uses fixed model widths"
    )]
    pub fn project_qkv(
        &self,
        hidden: &[f32],
        norm_epsilon: f32,
    ) -> Result<(Vec<f32>, Vec<f32>), MlxAffineRowError> {
        self.validate()?;
        if hidden.len() != Self::HIDDEN_WIDTH || !norm_epsilon.is_finite() || norm_epsilon <= 0.0 {
            return Err(MlxAffineRowError::TensorShape {
                name: "layer-zero hidden activation".to_owned(),
            });
        }
        if hidden.iter().any(|value| !value.is_finite()) {
            return Err(MlxAffineRowError::NonFinite { column: 0 });
        }
        fn project(rows: &[f32], row_count: usize, hidden: &[f32]) -> Vec<f32> {
            rows.chunks_exact(hidden.len())
                .take(row_count)
                .map(|row| {
                    row.iter()
                        .zip(hidden)
                        .map(|(weight, value)| weight * value)
                        .sum()
                })
                .collect()
        }
        let q_raw = project(&self.wq_a, Self::WQ_A_ROWS, hidden);
        let q_rms = (q_raw.iter().map(|value| value * value).sum::<f32>()
            / Self::Q_LORA_RANK as f32
            + norm_epsilon)
            .sqrt();
        let q: Vec<f32> = q_raw
            .into_iter()
            .zip(&self.q_norm)
            .map(|(value, weight)| value / q_rms * weight)
            .collect();
        let kv_raw = project(&self.wkv, Self::WKV_ROWS, hidden);
        let kv_rms = (kv_raw.iter().map(|value| value * value).sum::<f32>()
            / Self::KV_LORA_RANK as f32
            + norm_epsilon)
            .sqrt();
        let kv: Vec<f32> = kv_raw
            .into_iter()
            .zip(&self.kv_norm)
            .map(|(value, weight)| value / kv_rms * weight)
            .collect();
        if q.iter().chain(&kv).any(|value| !value.is_finite()) {
            return Err(MlxAffineRowError::NonFinite { column: 0 });
        }
        Ok((q, kv))
    }

    /// Validates the shape of the resident arrays after loading or handoff.
    ///
    /// This catches accidental truncation or row-major transposition before a
    /// Metal operation receives the arrays.
    pub fn validate(&self) -> Result<(), MlxAffineRowError> {
        if self.wq_a.len() != Self::WQ_A_ROWS * Self::HIDDEN_WIDTH
            || self.attn_norm.len() != Self::HIDDEN_WIDTH
            || self.hc_fn.len() != 24 * 16_384
            || self.hc_base.len() != 24
            || self.hc_scale.len() != 3
            || self.q_norm.len() != Self::Q_LORA_RANK
            || self.wkv.len() != Self::WKV_ROWS * Self::HIDDEN_WIDTH
            || self.kv_norm.len() != Self::KV_LORA_RANK
            || self
                .wq_a
                .iter()
                .chain(self.attn_norm.iter())
                .chain(self.hc_fn.iter())
                .chain(self.hc_base.iter())
                .chain(self.hc_scale.iter())
                .chain(self.q_norm.iter())
                .chain(self.wkv.iter())
                .chain(self.kv_norm.iter())
                .any(|value| !value.is_finite())
        {
            return Err(MlxAffineRowError::TensorShape {
                name: "layer-zero resident Q/KV tensors".to_owned(),
            });
        }
        Ok(())
    }
}
