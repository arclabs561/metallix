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
    let valid = weight.dtype() == V41StorageDtype::U32
        && scales.dtype() == V41StorageDtype::Bf16
        && biases.dtype() == V41StorageDtype::Bf16
        && weight.shape().len() == 2
        && scales.shape() == [weight.shape()[0], groups as u64]
        && biases.shape() == scales.shape()
        && weight.shape()[1] == packed_width as u64
        && usize::try_from(weight.shape()[0])
            .is_ok_and(|rows| first_row <= rows && row_count <= rows.saturating_sub(first_row));
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
    use super::{MlxAffineRowError, decode_affine_row};

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

    #[test]
    fn validates_resident_layer_zero_qkv_geometry_and_finiteness() {
        let resident = super::LayerZeroQkvResident {
            wq_a: vec![
                0.0;
                super::LayerZeroQkvResident::WQ_A_ROWS
                    * super::LayerZeroQkvResident::HIDDEN_WIDTH
            ],
            attn_norm: vec![1.0; super::LayerZeroQkvResident::HIDDEN_WIDTH],
            q_norm: vec![1.0; super::LayerZeroQkvResident::Q_LORA_RANK],
            wkv: vec![
                0.0;
                super::LayerZeroQkvResident::WKV_ROWS
                    * super::LayerZeroQkvResident::HIDDEN_WIDTH
            ],
            kv_norm: vec![1.0; super::LayerZeroQkvResident::KV_LORA_RANK],
        };
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

        let resident = super::LayerZeroQkvResident {
            wq_a: vec![
                0.0;
                super::LayerZeroQkvResident::WQ_A_ROWS
                    * super::LayerZeroQkvResident::HIDDEN_WIDTH
            ],
            attn_norm: vec![1.0; super::LayerZeroQkvResident::HIDDEN_WIDTH],
            q_norm: vec![1.0; super::LayerZeroQkvResident::Q_LORA_RANK],
            wkv: vec![
                0.0;
                super::LayerZeroQkvResident::WKV_ROWS
                    * super::LayerZeroQkvResident::HIDDEN_WIDTH
            ],
            kv_norm: vec![1.0; super::LayerZeroQkvResident::KV_LORA_RANK],
        };
        let hidden = vec![0.0; super::LayerZeroQkvResident::HIDDEN_WIDTH];
        let (q, kv) = resident
            .project_qkv(&hidden, 1e-6)
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
        Ok(Self {
            wq_a,
            attn_norm,
            q_norm,
            wkv,
            kv_norm,
        })
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
        epsilon: f32,
    ) -> Result<(Vec<f32>, Vec<f32>), MlxAffineRowError> {
        self.validate()?;
        if hidden.len() != Self::HIDDEN_WIDTH || !epsilon.is_finite() || epsilon <= 0.0 {
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
            + epsilon)
            .sqrt();
        let q: Vec<f32> = q_raw
            .into_iter()
            .zip(&self.q_norm)
            .map(|(value, weight)| value / q_rms * weight)
            .collect();
        let kv_raw = project(&self.wkv, Self::WKV_ROWS, hidden);
        let kv_rms = (kv_raw.iter().map(|value| value * value).sum::<f32>()
            / Self::KV_LORA_RANK as f32
            + epsilon)
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
            || self.q_norm.len() != Self::Q_LORA_RANK
            || self.wkv.len() != Self::WKV_ROWS * Self::HIDDEN_WIDTH
            || self.kv_norm.len() != Self::KV_LORA_RANK
            || self
                .wq_a
                .iter()
                .chain(self.attn_norm.iter())
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
