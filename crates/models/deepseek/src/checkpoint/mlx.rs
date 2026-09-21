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
    let mut packed_bytes = vec![0_u8; packed_width * 4];
    let mut scale_bytes = vec![0_u8; groups * 2];
    let mut bias_bytes = vec![0_u8; groups * 2];
    let mut decoded = Vec::with_capacity(row_count * logical_width);
    for row in first_row..first_row + row_count {
        read_range(
            &mut file,
            weight.file_range().start + row as u64 * packed_bytes.len() as u64,
            &mut packed_bytes,
        )?;
        read_range(
            &mut file,
            scales.file_range().start + row as u64 * scale_bytes.len() as u64,
            &mut scale_bytes,
        )?;
        read_range(
            &mut file,
            biases.file_range().start + row as u64 * bias_bytes.len() as u64,
            &mut bias_bytes,
        )?;
        let packed = packed_bytes
            .chunks_exact(4)
            .map(|bytes| u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
            .collect::<Vec<_>>();
        let scales = scale_bytes
            .chunks_exact(2)
            .map(|bytes| u16::from_le_bytes([bytes[0], bytes[1]]))
            .collect::<Vec<_>>();
        let biases = bias_bytes
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
    use super::{MlxAffineRowError, decode_affine_row, read_affine_rows_from_shard};
    use crate::checkpoint::V41SafetensorsHeader;
    use std::fs;
    use std::path::PathBuf;

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
        let scale_bytes = vec![0x80_u8, 0x3f_u8].repeat(rows);
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
        let header =
            V41SafetensorsHeader::parse_prefixed_header(&file_bytes, file_bytes.len() as u64)
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
        assert_eq!(
            super::collapse_hc_hidden(&[1.0, 2.0], &coefficients)
                .expect("HC collapse")
                .len(),
            2
        );
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
