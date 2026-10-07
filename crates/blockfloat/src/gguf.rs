//! GGUF tensor encodings: the closed set of `ggml` type IDs a GGUF file may
//! name, their block geometry, reference decoders, and the repack of the
//! encodings MLX can hold exactly onto MLX affine quantization.
//!
//! Type IDs and block sizes follow `GGMLQuantizationType` and
//! `GGML_QUANT_SIZES` in llama.cpp's `gguf-py/gguf/constants.py` at commit
//! 7fe450e19305b828c199d602c23a8337aaa1f03b. Block layouts follow
//! `ggml/src/ggml-common.h` at the same commit.
//!
//! Decoding is defined for `F32`, `F16`, `BF16` and `Q8_0`. Every other type
//! is known by name and size, so a file can be described, but decoding it is
//! an error rather than a guess.

use thiserror::Error;

/// One `ggml` tensor type. The names are the upstream spellings, which is
/// what GGUF file names and tools print.
#[allow(
    non_camel_case_types,
    reason = "variants keep the upstream ggml type names (Q8_0, IQ4_XS)"
)]
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum GgufEncoding {
    /// The upstream `F32` tensor encoding.
    F32,
    /// The upstream `F16` tensor encoding.
    F16,
    /// The upstream `Q4_0` tensor encoding.
    Q4_0,
    /// The upstream `Q4_1` tensor encoding.
    Q4_1,
    /// The upstream `Q5_0` tensor encoding.
    Q5_0,
    /// The upstream `Q5_1` tensor encoding.
    Q5_1,
    /// The upstream `Q8_0` tensor encoding.
    Q8_0,
    /// The upstream `Q8_1` tensor encoding.
    Q8_1,
    /// The upstream `Q2_K` tensor encoding.
    Q2_K,
    /// The upstream `Q3_K` tensor encoding.
    Q3_K,
    /// The upstream `Q4_K` tensor encoding.
    Q4_K,
    /// The upstream `Q5_K` tensor encoding.
    Q5_K,
    /// The upstream `Q6_K` tensor encoding.
    Q6_K,
    /// The upstream `Q8_K` tensor encoding.
    Q8_K,
    /// The upstream `IQ2_XXS` tensor encoding.
    IQ2_XXS,
    /// The upstream `IQ2_XS` tensor encoding.
    IQ2_XS,
    /// The upstream `IQ3_XXS` tensor encoding.
    IQ3_XXS,
    /// The upstream `IQ1_S` tensor encoding.
    IQ1_S,
    /// The upstream `IQ4_NL` tensor encoding.
    IQ4_NL,
    /// The upstream `IQ3_S` tensor encoding.
    IQ3_S,
    /// The upstream `IQ2_S` tensor encoding.
    IQ2_S,
    /// The upstream `IQ4_XS` tensor encoding.
    IQ4_XS,
    /// The upstream `I8` tensor encoding.
    I8,
    /// The upstream `I16` tensor encoding.
    I16,
    /// The upstream `I32` tensor encoding.
    I32,
    /// The upstream `I64` tensor encoding.
    I64,
    /// The upstream `F64` tensor encoding.
    F64,
    /// The upstream `IQ1_M` tensor encoding.
    IQ1_M,
    /// The upstream `BF16` tensor encoding.
    BF16,
    /// The upstream `TQ1_0` tensor encoding.
    TQ1_0,
    /// The upstream `TQ2_0` tensor encoding.
    TQ2_0,
    /// The upstream `MXFP4` tensor encoding.
    MXFP4,
    /// The upstream `NVFP4` tensor encoding.
    NVFP4,
    /// The upstream `Q1_0` tensor encoding.
    Q1_0,
    /// The upstream `Q2_0` tensor encoding.
    Q2_0,
}

/// `(encoding, type id, elements per block, bytes per block)`.
const TABLE: [(GgufEncoding, u32, u32, u32); 35] = {
    use GgufEncoding as E;
    const K: u32 = 256;
    [
        (E::F32, 0, 1, 4),
        (E::F16, 1, 1, 2),
        (E::Q4_0, 2, 32, 2 + 16),
        (E::Q4_1, 3, 32, 2 + 2 + 16),
        (E::Q5_0, 6, 32, 2 + 4 + 16),
        (E::Q5_1, 7, 32, 2 + 2 + 4 + 16),
        (E::Q8_0, 8, 32, 2 + 32),
        (E::Q8_1, 9, 32, 2 + 2 + 32),
        (E::Q2_K, 10, K, 2 + 2 + K / 16 + K / 4),
        (E::Q3_K, 11, K, 2 + K / 4 + K / 8 + 12),
        (E::Q4_K, 12, K, 2 + 2 + K / 2 + 12),
        (E::Q5_K, 13, K, 2 + 2 + K / 2 + K / 8 + 12),
        (E::Q6_K, 14, K, 2 + K / 2 + K / 4 + K / 16),
        (E::Q8_K, 15, K, 4 + K + K / 8),
        (E::IQ2_XXS, 16, K, 2 + K / 4),
        (E::IQ2_XS, 17, K, 2 + K / 4 + K / 32),
        (E::IQ3_XXS, 18, K, 2 + K / 4 + K / 8),
        (E::IQ1_S, 19, K, 2 + K / 8 + K / 16),
        (E::IQ4_NL, 20, 32, 2 + 16),
        (E::IQ3_S, 21, K, 2 + K / 4 + K / 8 + K / 32 + 4),
        (E::IQ2_S, 22, K, 2 + K / 4 + K / 16),
        (E::IQ4_XS, 23, K, 2 + 2 + K / 2 + K / 64),
        (E::I8, 24, 1, 1),
        (E::I16, 25, 1, 2),
        (E::I32, 26, 1, 4),
        (E::I64, 27, 1, 8),
        (E::F64, 28, 1, 8),
        (E::IQ1_M, 29, K, K / 8 + K / 16 + K / 32),
        (E::BF16, 30, 1, 2),
        (E::TQ1_0, 34, K, 2 + 4 * 13),
        (E::TQ2_0, 35, K, 2 + 64),
        (E::MXFP4, 39, 32, 1 + 16),
        (E::NVFP4, 40, 64, 4 + 32),
        (E::Q1_0, 41, 128, 2 + 16),
        (E::Q2_0, 42, 64, 2 + 16),
    ]
};

impl GgufEncoding {
    /// The encoding a GGUF tensor-info record names, or `None` for an ID this
    /// table does not know (including IDs upstream has retired).
    #[must_use]
    pub fn from_id(id: u32) -> Option<Self> {
        TABLE
            .iter()
            .find(|(_, known, _, _)| *known == id)
            .map(|(encoding, ..)| *encoding)
    }

    fn row(self) -> (u32, u32, u32) {
        let (_, id, elements, bytes) = TABLE
            .iter()
            .find(|(encoding, ..)| *encoding == self)
            .copied()
            .expect("every variant has a table row");
        (id, elements, bytes)
    }

    /// The `ggml` type ID.
    #[must_use]
    pub fn id(self) -> u32 {
        self.row().0
    }

    /// Elements per block; 1 for plain scalar types.
    #[must_use]
    pub fn block_elements(self) -> u32 {
        self.row().1
    }

    /// Bytes per block.
    #[must_use]
    pub fn block_bytes(self) -> u32 {
        self.row().2
    }

    /// Stored bytes of `elements` values, or `None` when `elements` is not a
    /// whole number of blocks or the size overflows.
    #[must_use]
    pub fn byte_len(self, elements: u64) -> Option<u64> {
        let block = u64::from(self.block_elements());
        if !elements.is_multiple_of(block) {
            return None;
        }
        (elements / block).checked_mul(u64::from(self.block_bytes()))
    }

    /// The upstream spelling, such as `Q8_0`.
    #[must_use]
    pub fn name(self) -> String {
        format!("{self:?}")
    }

    /// Whether [`decode`] expands this encoding.
    #[must_use]
    pub const fn decodable(self) -> bool {
        matches!(self, Self::F32 | Self::F16 | Self::BF16 | Self::Q8_0)
    }

    /// The MLX affine bit width and group size that holds this encoding with
    /// values equal to [`decode`]'s, if one does.
    #[must_use]
    pub const fn affine(self) -> Option<(u8, u32)> {
        match self {
            Self::Q8_0 => Some((8, 32)),
            _ => None,
        }
    }
}

/// An MLX affine layout: each group of `group_size` consecutive values in a
/// row shares one scale and one bias, `value = scale * code + bias`, and
/// `bits`-wide codes are packed little-endian into `u32` words.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AffineLayout {
    /// Bits per unsigned quantization code.
    pub bits: u8,
    /// Values sharing each scale and bias.
    pub group_size: u32,
    /// Required parameter precision.
    pub parameters: AffineParameters,
}

/// The narrowest float type that holds every scale and bias of a tensor
/// exactly.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AffineParameters {
    /// IEEE binary16 parameters.
    F16,
    /// IEEE binary32 parameters.
    F32,
}

/// Packed codes plus per-group scales and biases, in row-major order of the
/// source tensor, ready to become MLX arrays.
#[derive(Clone, Debug, PartialEq)]
pub struct AffineRepack {
    /// Packed code and parameter geometry.
    pub layout: AffineLayout,
    /// `elements * bits / 32` words.
    pub codes: Vec<u32>,
    /// One per group, exactly representable in `layout.parameters`.
    pub scales: Vec<f32>,
    /// One per group, exactly representable in `layout.parameters`.
    pub biases: Vec<f32>,
}

/// A GGUF payload that cannot be decoded or repacked.
#[derive(Debug, Error, Eq, PartialEq)]
#[non_exhaustive]
pub enum GgufDecodeError {
    /// The encoding has no implemented reference decoder.
    #[error("{0:?} is not decoded by this crate")]
    Unsupported(GgufEncoding),
    /// The payload ends inside an encoded block.
    #[error("{encoding:?} payload of {bytes} bytes is not a whole number of blocks")]
    Truncated {
        /// Declared encoding.
        encoding: GgufEncoding,
        /// Payload byte count.
        bytes: usize,
    },
    /// A block scale is NaN or infinite.
    #[error("{encoding:?} block {block} has a non-finite scale")]
    NonFiniteScale {
        /// Declared encoding.
        encoding: GgufEncoding,
        /// Zero-based block index.
        block: usize,
    },
    /// The encoding cannot be repacked as MLX affine codes.
    #[error("{0:?} does not map onto MLX affine quantization")]
    NotAffine(GgufEncoding),
}

/// Expands an IEEE binary16 bit pattern to `f32`, exactly (every binary16
/// value, subnormals and signed zero included, is an `f32`).
#[must_use]
pub fn f16_to_f32(bits: u16) -> f32 {
    let sign = u32::from(bits >> 15) << 31;
    let exponent = u32::from((bits >> 10) & 0x1f);
    let mantissa = u32::from(bits & 0x3ff);
    let magnitude = match (exponent, mantissa) {
        (0, 0) => 0,
        // Subnormal: mantissa * 2^-24, renormalized.
        (0, _) => {
            let shift = mantissa.leading_zeros() - 21;
            ((113 - shift) << 23) | ((mantissa << shift) & 0x3ff) << 13
        }
        (0x1f, 0) => 0xff << 23,
        (0x1f, _) => (0xff << 23) | (mantissa << 13),
        _ => ((exponent + 112) << 23) | (mantissa << 13),
    };
    f32::from_bits(sign | magnitude)
}

/// Expands a payload of `encoding` to `f32` values in stored order.
///
/// `Q8_0` is `d * q` per 32-value block, with `d` the block's binary16 scale
/// and `q` its signed bytes, multiplied in `f32` as ggml's reference does
/// (`dequantize_row_q8_0`). Scalar types widen exactly.
///
/// # Panics
///
/// Never: block sizes are small constants.
///
/// # Errors
/// Returns an error for unsupported encoding, incomplete blocks, or non-finite scales.
pub fn decode(encoding: GgufEncoding, bytes: &[u8]) -> Result<Vec<f32>, GgufDecodeError> {
    let block = usize::try_from(encoding.block_bytes()).expect("small");
    if !bytes.len().is_multiple_of(block) {
        return Err(GgufDecodeError::Truncated {
            encoding,
            bytes: bytes.len(),
        });
    }
    match encoding {
        GgufEncoding::F32 => Ok(bytes
            .chunks_exact(4)
            .map(|word| f32::from_le_bytes(word.try_into().expect("four bytes")))
            .collect()),
        GgufEncoding::F16 => Ok(bytes
            .chunks_exact(2)
            .map(|half| f16_to_f32(u16::from_le_bytes([half[0], half[1]])))
            .collect()),
        GgufEncoding::BF16 => Ok(bytes
            .chunks_exact(2)
            .map(|half| crate::bf16_to_f32(u16::from_le_bytes([half[0], half[1]])))
            .collect()),
        GgufEncoding::Q8_0 => {
            let mut values = Vec::with_capacity(bytes.len() / block * 32);
            for (index, chunk) in bytes.chunks_exact(block).enumerate() {
                let scale = q8_0_scale(chunk, index)?;
                values.extend(
                    chunk[2..]
                        .iter()
                        .map(|&code| f32::from(code.cast_signed()) * scale),
                );
            }
            Ok(values)
        }
        other => Err(GgufDecodeError::Unsupported(other)),
    }
}

fn q8_0_scale(block: &[u8], index: usize) -> Result<f32, GgufDecodeError> {
    let scale = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
    if scale.is_finite() {
        Ok(scale)
    } else {
        Err(GgufDecodeError::NonFiniteScale {
            encoding: GgufEncoding::Q8_0,
            block: index,
        })
    }
}

/// Repacks a payload onto MLX affine quantization whose dequantized values
/// equal [`decode`]'s (see [`GgufEncoding::affine`]).
///
/// `Q8_0`: code `q + 128` as an unsigned byte, scale `d`, bias `-128 d`. Both
/// are exact in `f32`; they are also exact in binary16 (`d` is binary16 and
/// `-128 d` a power-of-two multiple) unless some bias exceeds binary16's
/// range, in which case the tensor's parameters are `F32`. Rows must hold
/// whole blocks, so groups never straddle rows; the caller checks the row
/// width.
///
/// # Panics
///
/// Never: block sizes are small constants.
///
/// # Errors
/// Returns an error for unsupported encoding, incomplete blocks, or non-finite scales.
pub fn repack_affine(
    encoding: GgufEncoding,
    bytes: &[u8],
) -> Result<AffineRepack, GgufDecodeError> {
    let (code_bits, group_size) = encoding
        .affine()
        .ok_or(GgufDecodeError::NotAffine(encoding))?;
    let block = usize::try_from(encoding.block_bytes()).expect("small");
    if !bytes.len().is_multiple_of(block) {
        return Err(GgufDecodeError::Truncated {
            encoding,
            bytes: bytes.len(),
        });
    }
    let blocks = bytes.len() / block;
    let mut repack = AffineRepack {
        layout: AffineLayout {
            bits: code_bits,
            group_size,
            parameters: AffineParameters::F16,
        },
        codes: Vec::with_capacity(blocks * 8),
        scales: Vec::with_capacity(blocks),
        biases: Vec::with_capacity(blocks),
    };
    for (index, chunk) in bytes.chunks_exact(block).enumerate() {
        let scale = q8_0_scale(chunk, index)?;
        let bias = -128.0 * scale;
        if bias.abs() > F16_MAX {
            repack.layout.parameters = AffineParameters::F32;
        }
        repack.scales.push(scale);
        repack.biases.push(bias);
        repack.codes.extend(
            chunk[2..]
                .chunks_exact(4)
                .map(|word| u32::from_le_bytes(word.try_into().expect("four bytes")) ^ 0x8080_8080),
        );
    }
    Ok(repack)
}

/// The largest finite binary16 value.
const F16_MAX: f32 = 65504.0;

/// One value of an [`AffineRepack`] as MLX's affine dequantization defines
/// it, `scale * code + bias` in `f32`, for checks against [`decode`].
///
/// # Panics
///
/// If `index` is outside the repacked tensor.
#[must_use]
pub fn affine_value(repack: &AffineRepack, index: usize) -> f32 {
    let AffineLayout {
        bits, group_size, ..
    } = repack.layout;
    let per_word = 32 / usize::from(bits);
    let word = repack.codes[index / per_word];
    let shift = (index % per_word) * usize::from(bits);
    let mask = (1_u32 << bits) - 1;
    #[allow(clippy::cast_precision_loss, reason = "codes are at most 8 bits")]
    let code = ((word >> shift) & mask) as f32;
    let group = index / usize::try_from(group_size).expect("small");
    // Rust does not contract `a * b + c` into a fused multiply-add.
    repack.scales[group] * code + repack.biases[group]
}

#[cfg(test)]
mod tests {
    use super::{
        AffineParameters, GgufDecodeError, GgufEncoding, affine_value, decode, f16_to_f32,
        repack_affine,
    };

    fn q8_0_block(scale: u16, codes: &[i8; 32]) -> Vec<u8> {
        let mut block = scale.to_le_bytes().to_vec();
        block.extend(codes.iter().map(|code| code.cast_unsigned()));
        block
    }

    /// The same value, or zeros of either sign (see `repack_affine`).
    fn same(left: f32, right: f32) -> bool {
        left.to_bits() == right.to_bits() || (left == 0.0 && right == 0.0)
    }

    #[test]
    fn table_ids_and_sizes_round_trip() {
        for id in 0..64 {
            if let Some(encoding) = GgufEncoding::from_id(id) {
                assert_eq!(encoding.id(), id);
            }
        }
        assert_eq!(GgufEncoding::from_id(4), None, "Q4_2 was retired upstream");
        assert_eq!(GgufEncoding::from_id(8), Some(GgufEncoding::Q8_0));
        assert_eq!(GgufEncoding::Q8_0.byte_len(64), Some(68));
        assert_eq!(GgufEncoding::Q8_0.byte_len(48), None);
        assert_eq!(GgufEncoding::Q4_K.byte_len(256), Some(144));
        assert_eq!(GgufEncoding::IQ4_XS.byte_len(256), Some(136));
        assert_eq!(GgufEncoding::Q6_K.byte_len(256), Some(210));
        assert_eq!(GgufEncoding::IQ4_XS.name(), "IQ4_XS");
    }

    #[test]
    fn f16_widening_is_exact_for_every_pattern() {
        for bits in 0..=u16::MAX {
            let wide = f16_to_f32(bits);
            let exponent = (bits >> 10) & 0x1f;
            let mantissa = f64::from(bits & 0x3ff);
            let sign = if bits >> 15 == 1 { -1.0 } else { 1.0 };
            let expected = match exponent {
                0 => sign * mantissa * 2.0_f64.powi(-24),
                0x1f if mantissa == 0.0 => sign * f64::INFINITY,
                0x1f => {
                    assert!(wide.is_nan(), "{bits:#06x}");
                    continue;
                }
                _ => sign * (1.0 + mantissa / 1024.0) * 2.0_f64.powi(i32::from(exponent) - 15),
            };
            assert_eq!(f64::from(wide), expected, "{bits:#06x}");
            assert_eq!(wide.is_sign_negative(), bits >> 15 == 1, "{bits:#06x}");
        }
    }

    #[test]
    fn q8_0_decodes_every_code_with_hand_scales() {
        let codes: Vec<i8> = (-128..=127).collect();
        // 0.5, -2.0, +0, -0, smallest subnormal 2^-24, largest finite 65504.
        for (scale, value) in [
            (0x3800_u16, 0.5_f32),
            (0xc000, -2.0),
            (0x0000, 0.0),
            (0x8000, -0.0),
            (0x0001, 2.0_f32.powi(-24)),
            (0x7bff, 65504.0),
        ] {
            for chunk in codes.chunks_exact(32) {
                let block = q8_0_block(scale, chunk.try_into().expect("32 codes"));
                let decoded = decode(GgufEncoding::Q8_0, &block).expect("decode");
                for (got, code) in decoded.iter().zip(chunk) {
                    let want = f32::from(*code) * value;
                    assert_eq!(
                        got.to_bits(),
                        want.to_bits(),
                        "scale {scale:#06x} code {code}"
                    );
                }
            }
        }
        for scale in [0x7c00_u16, 0xfc00, 0x7e00] {
            let block = q8_0_block(scale, &[1; 32]);
            assert_eq!(
                decode(GgufEncoding::Q8_0, &block),
                Err(GgufDecodeError::NonFiniteScale {
                    encoding: GgufEncoding::Q8_0,
                    block: 0
                })
            );
        }
        assert!(matches!(
            decode(GgufEncoding::Q8_0, &[0; 33]),
            Err(GgufDecodeError::Truncated { .. })
        ));
        assert_eq!(
            decode(GgufEncoding::Q4_K, &[0; 144]),
            Err(GgufDecodeError::Unsupported(GgufEncoding::Q4_K))
        );
    }

    #[test]
    fn q8_0_affine_repack_equals_the_decoder() {
        let codes: Vec<i8> = (-128..=127).collect();
        let mut payload = Vec::new();
        // Every finite binary16 scale, each with a slice of codes, covers
        // subnormals, every exponent and both signs.
        for (index, scale) in (0..=u16::MAX)
            .filter(|bits| f16_to_f32(*bits).is_finite())
            .enumerate()
        {
            let start = (index * 32) % 256;
            let chunk: [i8; 32] = codes[start..start + 32].try_into().expect("32");
            payload.extend(q8_0_block(scale, &chunk));
        }
        let decoded = decode(GgufEncoding::Q8_0, &payload).expect("decode");
        let repack = repack_affine(GgufEncoding::Q8_0, &payload).expect("repack");
        assert_eq!((repack.layout.bits, repack.layout.group_size), (8, 32));
        // Scales above 511.98 give biases beyond binary16.
        assert_eq!(repack.layout.parameters, AffineParameters::F32);
        assert_eq!(repack.codes.len() * 4, decoded.len());
        let mismatches = (0..decoded.len())
            .filter(|&index| !same(affine_value(&repack, index), decoded[index]))
            .count();
        assert_eq!(mismatches, 0);
    }

    #[test]
    fn q8_0_parameters_are_binary16_when_every_bias_fits() {
        let small = q8_0_block(0x5fff, &[3; 32]);
        let repack = repack_affine(GgufEncoding::Q8_0, &small).expect("repack");
        assert_eq!(repack.layout.parameters, AffineParameters::F16);
        // 511.75 * -128 = -65504, binary16's largest magnitude, still fits.
        assert_eq!(repack.biases, [-65504.0]);
        let large = q8_0_block(0x6000, &[3; 32]);
        let repack = repack_affine(GgufEncoding::Q8_0, &large).expect("repack");
        assert_eq!(repack.layout.parameters, AffineParameters::F32);
        assert_eq!(repack.biases, [-65536.0]);
        assert_eq!(
            repack_affine(GgufEncoding::F16, &[0; 2]),
            Err(GgufDecodeError::NotAffine(GgufEncoding::F16))
        );
    }
}
