//! Bounded, header-only DeepSeek-V4.1 safetensors validation.
//!
//! This module consumes an already-bounded safetensors JSON header and a
//! declared shard length. It never opens a file, reads payload bytes, or
//! infers a checkpoint's packed-runtime layout. In particular, packed FP4
//! representations remain unsupported until a separate file-layout contract
//! establishes their storage and scale association.

use std::{collections::BTreeMap, ops::Range};

use serde::Deserialize;
use thiserror::Error;

use crate::manifest::V41SafetensorsIndex;

const SAFETENSORS_PREFIX_BYTES: u64 = 8;
const MAX_HEADER_BYTES: u64 = 100 * 1024 * 1024;

/// A fixed-width storage dtype established by the held safetensors contract.
///
/// The accepted spellings and byte widths mirror the existing local
/// Qwen safetensors inspector. This is a storage fact, not a claim about a
/// `DeepSeek` runtime tensor layout.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum V41StorageDtype {
    /// One-byte Boolean storage.
    Bool,
    /// Unsigned 8-bit integer storage.
    U8,
    /// Signed 8-bit integer storage.
    I8,
    /// Unsigned 16-bit integer storage.
    U16,
    /// Signed 16-bit integer storage.
    I16,
    /// IEEE half-precision storage.
    F16,
    /// Brain floating-point storage.
    Bf16,
    /// Unsigned 32-bit integer storage.
    U32,
    /// Signed 32-bit integer storage.
    I32,
    /// IEEE single-precision storage.
    F32,
    /// Unsigned 64-bit integer storage.
    U64,
    /// Signed 64-bit integer storage.
    I64,
    /// IEEE double-precision storage.
    F64,
    /// One-byte E4M3FN floating-point storage.
    F8E4M3Fn,
    /// One-byte E4M3FNUZ floating-point storage.
    F8E4M3Fnuz,
    /// One-byte E5M2 floating-point storage.
    F8E5M2,
    /// One-byte E5M2FNUZ floating-point storage.
    F8E5M2Fnuz,
    /// One-byte E8M0FNU floating-point storage.
    F8E8M0Fnu,
}

impl V41StorageDtype {
    fn parse(dtype: &str) -> Option<Self> {
        Some(match dtype {
            "BOOL" => Self::Bool,
            "U8" => Self::U8,
            "I8" => Self::I8,
            "U16" => Self::U16,
            "I16" => Self::I16,
            "F16" => Self::F16,
            "BF16" => Self::Bf16,
            "U32" => Self::U32,
            "I32" => Self::I32,
            "F32" => Self::F32,
            "U64" => Self::U64,
            "I64" => Self::I64,
            "F64" => Self::F64,
            "F8_E4M3FN" => Self::F8E4M3Fn,
            "F8_E4M3FNUZ" => Self::F8E4M3Fnuz,
            "F8_E5M2" => Self::F8E5M2,
            "F8_E5M2FNUZ" => Self::F8E5M2Fnuz,
            "F8_E8M0FNU" => Self::F8E8M0Fnu,
            _ => return None,
        })
    }

    /// Returns the declared storage bytes for one logical header element.
    #[must_use]
    pub const fn bytes_per_element(self) -> u64 {
        match self {
            Self::Bool
            | Self::U8
            | Self::I8
            | Self::F8E4M3Fn
            | Self::F8E4M3Fnuz
            | Self::F8E5M2
            | Self::F8E5M2Fnuz
            | Self::F8E8M0Fnu => 1,
            Self::U16 | Self::I16 | Self::F16 | Self::Bf16 => 2,
            Self::U32 | Self::I32 | Self::F32 => 4,
            Self::U64 | Self::I64 | Self::F64 => 8,
        }
    }
}

/// One validated tensor interval in a safetensors shard.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct V41TensorRange {
    dtype: V41StorageDtype,
    shape: Vec<u64>,
    file_range: Range<u64>,
}

impl V41TensorRange {
    /// Returns the validated storage dtype.
    #[must_use]
    pub const fn dtype(&self) -> V41StorageDtype {
        self.dtype
    }

    /// Returns the header's logical tensor shape. An empty slice is a scalar.
    #[must_use]
    pub fn shape(&self) -> &[u64] {
        &self.shape
    }

    /// Returns the checked half-open range in the complete shard file.
    ///
    /// Coordinates are absolute from the start of the complete shard; this
    /// range lies wholly within the payload after the safetensors prefix and
    /// header. It is metadata only; this type cannot read the range.
    #[must_use]
    pub fn file_range(&self) -> Range<u64> {
        self.file_range.clone()
    }

    /// Returns the checked length of this tensor's payload interval.
    #[must_use]
    pub const fn byte_length(&self) -> u64 {
        self.file_range.end - self.file_range.start
    }
}

/// Validated metadata for one complete safetensors shard header.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct V41SafetensorsHeader {
    file_bytes: u64,
    payload_range: Range<u64>,
    tensors: BTreeMap<String, V41TensorRange>,
}

impl V41SafetensorsHeader {
    /// Parses exactly the eight-byte length prefix and its declared JSON header.
    ///
    /// `prefix_and_header` must exclude tensor payload bytes. The little-endian
    /// length is checked against the header cap, declared complete shard length,
    /// and supplied body length before JSON parsing. This method performs no I/O
    /// and does not bound an allocation already made by the caller.
    ///
    /// # Errors
    ///
    /// Returns [`V41SafetensorsHeaderError`] for a short prefix, oversized or
    /// mismatched header length, insufficient shard length, or an invalid header.
    pub fn parse_prefixed_header(
        prefix_and_header: &[u8],
        file_bytes: u64,
    ) -> Result<Self, V41SafetensorsHeaderError> {
        let (prefix, header) = prefix_and_header.split_first_chunk::<8>().ok_or(
            V41SafetensorsHeaderError::PrefixTooShort {
                actual_bytes: prefix_and_header.len(),
            },
        )?;
        let declared_bytes = u64::from_le_bytes(*prefix);
        if declared_bytes > MAX_HEADER_BYTES {
            return Err(V41SafetensorsHeaderError::HeaderTooLarge {
                header_bytes: declared_bytes,
            });
        }
        // The cap above also proves this addition cannot overflow.
        let required_bytes = SAFETENSORS_PREFIX_BYTES + declared_bytes;
        if file_bytes < required_bytes {
            return Err(V41SafetensorsHeaderError::FileLengthTooSmall {
                file_bytes,
                required_bytes,
            });
        }
        if usize::try_from(declared_bytes).ok() != Some(header.len()) {
            return Err(V41SafetensorsHeaderError::HeaderLengthMismatch {
                declared_bytes,
                actual_bytes: header.len(),
            });
        }
        Self::parse(header, file_bytes)
    }

    /// Parses an already-bounded safetensors JSON header.
    ///
    /// `header_bytes` must be the exact bytes after the eight-byte little-endian
    /// safetensors header-length prefix, including any on-disk JSON padding.
    /// `file_bytes` is the declared complete shard length. The parser validates
    /// all ranges before returning and never reads tensor payload bytes.
    ///
    /// Scalars and zero-element tensors are supported. Packed dtypes, including
    /// FP4, are rejected because this metadata parser has no proven file-to-
    /// runtime layout or scale association for them.
    ///
    /// # Errors
    ///
    /// Returns [`V41SafetensorsHeaderError`] when the supplied header is too
    /// large, malformed, ambiguous, or inconsistent with the declared shard
    /// length or its own dtype/shape/range metadata.
    pub fn parse(header_bytes: &[u8], file_bytes: u64) -> Result<Self, V41SafetensorsHeaderError> {
        parse_with_header_limit(header_bytes, file_bytes, MAX_HEADER_BYTES)
    }
}

fn parse_with_header_limit(
    header_bytes: &[u8],
    file_bytes: u64,
    max_header_bytes: u64,
) -> Result<V41SafetensorsHeader, V41SafetensorsHeaderError> {
    let header_length = u64::try_from(header_bytes.len()).map_err(|_| {
        V41SafetensorsHeaderError::HeaderTooLarge {
            header_bytes: u64::MAX,
        }
    })?;
    if header_length > max_header_bytes {
        return Err(V41SafetensorsHeaderError::HeaderTooLarge {
            header_bytes: header_length,
        });
    }
    let payload_start = SAFETENSORS_PREFIX_BYTES.checked_add(header_length).ok_or(
        V41SafetensorsHeaderError::FileLengthTooSmall {
            file_bytes,
            required_bytes: u64::MAX,
        },
    )?;
    if file_bytes < payload_start {
        return Err(V41SafetensorsHeaderError::FileLengthTooSmall {
            file_bytes,
            required_bytes: payload_start,
        });
    }

    let header: UniqueHeader =
        serde_json::from_slice(header_bytes).map_err(V41SafetensorsHeaderError::HeaderJson)?;
    let payload_bytes = file_bytes - payload_start;
    let tensors = validate_tensors(header.0, payload_start, payload_bytes)?;
    if tensors.is_empty() {
        return Err(V41SafetensorsHeaderError::EmptyTensorSet);
    }
    Ok(V41SafetensorsHeader {
        file_bytes,
        payload_range: payload_start..file_bytes,
        tensors,
    })
}

impl V41SafetensorsHeader {
    /// Returns the complete declared shard length.
    #[must_use]
    pub const fn file_bytes(&self) -> u64 {
        self.file_bytes
    }

    /// Returns the checked half-open payload region in absolute file offsets.
    ///
    /// The region starts after the safetensors prefix and header; neither is a
    /// member of this range.
    #[must_use]
    pub fn payload_range(&self) -> Range<u64> {
        self.payload_range.clone()
    }

    /// Returns a tensor interval by exact header name.
    #[must_use]
    pub fn tensor(&self, name: &str) -> Option<&V41TensorRange> {
        self.tensors.get(name)
    }

    /// Iterates validated tensor names and intervals in deterministic order.
    #[must_use]
    pub fn tensors(&self) -> impl ExactSizeIterator<Item = (&str, &V41TensorRange)> + '_ {
        self.tensors
            .iter()
            .map(|(name, range)| (name.as_str(), range))
    }

    /// Confirms exact tensor-name agreement between this header and `shard`'s
    /// entries in a previously validated checkpoint index.
    ///
    /// This needs only this shard's one header and the index: it rejects a
    /// header tensor absent from or assigned differently by the index, and an
    /// index tensor assigned to `shard` but absent from this header.
    ///
    /// # Errors
    ///
    /// Returns [`V41SafetensorsHeaderError`] if the two exact name sets differ
    /// or an indexed tensor is assigned to another shard.
    pub fn validate_index_shard(
        &self,
        index: &V41SafetensorsIndex,
        shard: &str,
    ) -> Result<(), V41SafetensorsHeaderError> {
        for name in self.tensors.keys() {
            let indexed_shard = index.shard_for_tensor(name).ok_or_else(|| {
                V41SafetensorsHeaderError::TensorMissingFromIndex {
                    tensor: name.clone(),
                }
            })?;
            if indexed_shard != shard {
                return Err(V41SafetensorsHeaderError::TensorAssignedToOtherShard {
                    tensor: name.clone(),
                    expected_shard: shard.to_owned(),
                    indexed_shard: indexed_shard.to_owned(),
                });
            }
        }
        for (name, _) in index
            .tensor_shards()
            .filter(|(_, indexed_shard)| *indexed_shard == shard)
        {
            if !self.tensors.contains_key(name) {
                return Err(V41SafetensorsHeaderError::TensorMissingFromHeader {
                    tensor: name.to_owned(),
                });
            }
        }
        Ok(())
    }
}

// Reject repeated keys before a map insertion can discard one spelling. This
// descends into values too, so an ambiguous tensor's dtype or range cannot be
// hidden by repeated nested keys.
struct UniqueHeader(BTreeMap<String, serde_json::Value>);

impl<'de> Deserialize<'de> for UniqueHeader {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct HeaderVisitor;

        impl<'de> serde::de::Visitor<'de> for HeaderVisitor {
            type Value = UniqueHeader;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("a safetensors header with unique keys")
            }

            fn visit_map<M: serde::de::MapAccess<'de>>(
                self,
                mut map: M,
            ) -> Result<Self::Value, M::Error> {
                let mut header = BTreeMap::new();
                while let Some(key) = map.next_key::<String>()? {
                    if header.contains_key(&key) {
                        return Err(serde::de::Error::custom("duplicate safetensors header key"));
                    }
                    header.insert(key, map.next_value::<UniqueJsonValue>()?.0);
                }
                Ok(UniqueHeader(header))
            }
        }

        deserializer.deserialize_map(HeaderVisitor)
    }
}

struct UniqueJsonValue(serde_json::Value);

impl<'de> Deserialize<'de> for UniqueJsonValue {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct ValueVisitor;

        impl<'de> serde::de::Visitor<'de> for ValueVisitor {
            type Value = UniqueJsonValue;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("a JSON value with unique object keys")
            }

            fn visit_bool<E: serde::de::Error>(self, value: bool) -> Result<Self::Value, E> {
                Ok(UniqueJsonValue(value.into()))
            }

            fn visit_i64<E: serde::de::Error>(self, value: i64) -> Result<Self::Value, E> {
                Ok(UniqueJsonValue(value.into()))
            }

            fn visit_u64<E: serde::de::Error>(self, value: u64) -> Result<Self::Value, E> {
                Ok(UniqueJsonValue(value.into()))
            }

            fn visit_f64<E: serde::de::Error>(self, value: f64) -> Result<Self::Value, E> {
                serde_json::Number::from_f64(value)
                    .map(|number| UniqueJsonValue(number.into()))
                    .ok_or_else(|| E::custom("non-finite JSON number"))
            }

            fn visit_str<E: serde::de::Error>(self, value: &str) -> Result<Self::Value, E> {
                self.visit_string(value.to_owned())
            }

            fn visit_string<E: serde::de::Error>(self, value: String) -> Result<Self::Value, E> {
                Ok(UniqueJsonValue(value.into()))
            }

            fn visit_unit<E: serde::de::Error>(self) -> Result<Self::Value, E> {
                Ok(UniqueJsonValue(serde_json::Value::Null))
            }

            fn visit_seq<A: serde::de::SeqAccess<'de>>(
                self,
                mut sequence: A,
            ) -> Result<Self::Value, A::Error> {
                let mut values = Vec::new();
                while let Some(value) = sequence.next_element::<UniqueJsonValue>()? {
                    values.push(value.0);
                }
                Ok(UniqueJsonValue(serde_json::Value::Array(values)))
            }

            fn visit_map<M: serde::de::MapAccess<'de>>(
                self,
                map: M,
            ) -> Result<Self::Value, M::Error> {
                let object =
                    UniqueHeader::deserialize(serde::de::value::MapAccessDeserializer::new(map))?;
                Ok(UniqueJsonValue(serde_json::Value::Object(
                    object.0.into_iter().collect(),
                )))
            }
        }

        deserializer.deserialize_any(ValueVisitor)
    }
}

fn validate_tensors(
    header: BTreeMap<String, serde_json::Value>,
    payload_start: u64,
    payload_bytes: u64,
) -> Result<BTreeMap<String, V41TensorRange>, V41SafetensorsHeaderError> {
    let mut tensors = BTreeMap::new();
    let mut ranges = Vec::new();
    for (name, value) in header {
        if name == "__metadata__" {
            if serde_json::from_value::<BTreeMap<String, String>>(value).is_err() {
                return Err(V41SafetensorsHeaderError::InvalidMetadata);
            }
            continue;
        }
        if name.trim().is_empty() {
            return Err(V41SafetensorsHeaderError::BlankTensorName);
        }
        let tensor: RawTensor = serde_json::from_value(value).map_err(|source| {
            V41SafetensorsHeaderError::InvalidTensorHeader {
                tensor: name.clone(),
                source,
            }
        })?;
        let [start, end] = tensor.data_offsets.as_slice() else {
            return Err(V41SafetensorsHeaderError::InvalidTensorOffsets { tensor: name });
        };
        let start = *start;
        let end = *end;
        if start > end || end > payload_bytes {
            return Err(V41SafetensorsHeaderError::TensorOutsidePayload { tensor: name });
        }
        let dtype = V41StorageDtype::parse(&tensor.dtype).ok_or_else(|| {
            V41SafetensorsHeaderError::UnsupportedTensorDtype {
                tensor: name.clone(),
                dtype: tensor.dtype.clone(),
            }
        })?;
        let element_count = if tensor.shape.contains(&0) {
            0
        } else {
            tensor.shape.iter().try_fold(1_u64, |total, dimension| {
                total.checked_mul(*dimension).ok_or_else(|| {
                    V41SafetensorsHeaderError::TensorShapeOverflow {
                        tensor: name.clone(),
                    }
                })
            })?
        };
        let expected_bytes = element_count
            .checked_mul(dtype.bytes_per_element())
            .ok_or_else(|| V41SafetensorsHeaderError::TensorByteLengthOverflow {
                tensor: name.clone(),
            })?;
        let actual_bytes = end - start;
        if actual_bytes != expected_bytes {
            return Err(V41SafetensorsHeaderError::TensorByteLengthMismatch {
                tensor: name,
                expected_bytes,
                actual_bytes,
            });
        }
        let file_start = payload_start.checked_add(start).ok_or_else(|| {
            V41SafetensorsHeaderError::TensorFileOffsetOverflow {
                tensor: name.clone(),
            }
        })?;
        let file_end = payload_start.checked_add(end).ok_or_else(|| {
            V41SafetensorsHeaderError::TensorFileOffsetOverflow {
                tensor: name.clone(),
            }
        })?;
        ranges.push((start, end));
        tensors.insert(
            name,
            V41TensorRange {
                dtype,
                shape: tensor.shape,
                file_range: file_start..file_end,
            },
        );
    }
    ranges.sort_unstable();
    if ranges.windows(2).any(|ranges| ranges[1].0 < ranges[0].1) {
        return Err(V41SafetensorsHeaderError::OverlappingTensorRanges);
    }
    let mut expected_start = 0_u64;
    for (start, end) in ranges {
        if start != expected_start {
            return Err(V41SafetensorsHeaderError::NonContiguousPayload);
        }
        expected_start = end;
    }
    if expected_start != payload_bytes {
        return Err(V41SafetensorsHeaderError::NonContiguousPayload);
    }
    Ok(tensors)
}

#[derive(Debug, Deserialize)]
struct RawTensor {
    dtype: String,
    data_offsets: Vec<u64>,
    shape: Vec<u64>,
}

/// A supplied safetensors header cannot safely describe a V4.1 tensor range.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum V41SafetensorsHeaderError {
    /// The supplied metadata does not contain the full eight-byte length prefix.
    #[error("safetensors length prefix needs 8 bytes, received {actual_bytes}")]
    PrefixTooShort {
        /// Total supplied metadata bytes.
        actual_bytes: usize,
    },
    /// Supplied header bytes do not exactly match the on-file length prefix.
    #[error("safetensors prefix declares {declared_bytes} header bytes, received {actual_bytes}")]
    HeaderLengthMismatch {
        /// Header byte count encoded in the prefix.
        declared_bytes: u64,
        /// Supplied bytes after the prefix, excluding no trailing bytes implicitly.
        actual_bytes: usize,
    },
    /// The supplied header exceeds the parser's explicit bound.
    #[error("safetensors header is {header_bytes} bytes, above the {MAX_HEADER_BYTES}-byte limit")]
    HeaderTooLarge {
        /// Supplied header byte count.
        header_bytes: u64,
    },
    /// Prefix plus header cannot fit within the declared complete shard length.
    #[error(
        "safetensors shard is {file_bytes} bytes but needs at least {required_bytes} bytes for prefix and header"
    )]
    FileLengthTooSmall {
        /// Declared complete shard length.
        file_bytes: u64,
        /// Minimum prefix-plus-header length.
        required_bytes: u64,
    },
    /// The supplied bytes were not an unambiguous JSON header object.
    #[error("invalid safetensors header JSON: {0}")]
    HeaderJson(serde_json::Error),
    /// The header contained no tensor entries.
    #[error("safetensors header contains no tensor entries")]
    EmptyTensorSet,
    /// Reserved safetensors metadata was not an object of string values.
    #[error("invalid safetensors metadata")]
    InvalidMetadata,
    /// A tensor header was malformed.
    #[error("invalid tensor header for {tensor:?}: {source}")]
    InvalidTensorHeader {
        /// Tensor name.
        tensor: String,
        /// JSON parse failure.
        source: serde_json::Error,
    },
    /// A tensor did not declare exactly one start/end interval.
    #[error("invalid data offsets for tensor {tensor:?}")]
    InvalidTensorOffsets {
        /// Tensor name.
        tensor: String,
    },
    /// A tensor interval was reversed or outside the declared payload.
    #[error("tensor {tensor:?} is outside the safetensors payload")]
    TensorOutsidePayload {
        /// Tensor name.
        tensor: String,
    },
    /// A tensor shape product did not fit in `u64`.
    #[error("tensor {tensor:?} has an overflowing shape")]
    TensorShapeOverflow {
        /// Tensor name.
        tensor: String,
    },
    /// A dtype has no approved fixed-width V4.1 storage mapping.
    #[error("tensor {tensor:?} has unsupported safetensors dtype {dtype:?}")]
    UnsupportedTensorDtype {
        /// Tensor name.
        tensor: String,
        /// Header dtype text.
        dtype: String,
    },
    /// A tensor's implied byte length overflowed `u64`.
    #[error("tensor {tensor:?} byte length overflowed")]
    TensorByteLengthOverflow {
        /// Tensor name.
        tensor: String,
    },
    /// A tensor interval disagrees with its checked dtype and shape.
    #[error("tensor {tensor:?} has {actual_bytes} bytes, expected {expected_bytes}")]
    TensorByteLengthMismatch {
        /// Tensor name.
        tensor: String,
        /// Bytes implied by dtype and shape.
        expected_bytes: u64,
        /// Bytes named by data offsets.
        actual_bytes: u64,
    },
    /// A validated payload-relative offset could not become a file offset.
    #[error("tensor {tensor:?} file offset overflowed")]
    TensorFileOffsetOverflow {
        /// Tensor name.
        tensor: String,
    },
    /// Two tensor intervals overlap.
    #[error("safetensors header has overlapping tensor ranges")]
    OverlappingTensorRanges,
    /// Tensor intervals fail to cover the declared payload exactly once.
    #[error("safetensors header has holes or trailing payload bytes")]
    NonContiguousPayload,
    /// A tensor name was blank after trimming.
    #[error("safetensors header has a blank tensor name")]
    BlankTensorName,
    /// A header tensor was absent from the supplied index.
    #[error("tensor {tensor:?} is absent from the safetensors index")]
    TensorMissingFromIndex {
        /// Tensor name.
        tensor: String,
    },
    /// An index tensor assigned to this shard was absent from its header.
    #[error("tensor {tensor:?} is absent from the safetensors header")]
    TensorMissingFromHeader {
        /// Tensor name.
        tensor: String,
    },
    /// A header tensor was assigned to another shard by the supplied index.
    #[error("tensor {tensor:?} belongs to index shard {indexed_shard:?}, not {expected_shard:?}")]
    TensorAssignedToOtherShard {
        /// Tensor name.
        tensor: String,
        /// Caller-supplied shard for this header.
        expected_shard: String,
        /// Index-declared shard.
        indexed_shard: String,
    },
}

#[cfg(test)]
mod tests {
    use super::{
        V41SafetensorsHeader, V41SafetensorsHeaderError, V41StorageDtype, parse_with_header_limit,
    };
    use crate::manifest::V41SafetensorsIndex;

    fn file_bytes(header: &[u8], payload_bytes: u64) -> u64 {
        8 + u64::try_from(header.len()).expect("small test header") + payload_bytes
    }

    #[test]
    fn prefixed_header_preserves_padded_absolute_ranges_without_payload() {
        let mut header =
            br#"{"weight":{"dtype":"BF16","shape":[2],"data_offsets":[0,4]}}"#.to_vec();
        header.resize(512, b' ');
        let mut metadata = 512_u64.to_le_bytes().to_vec();
        metadata.extend_from_slice(&header);
        let parsed = V41SafetensorsHeader::parse_prefixed_header(&metadata, 524)
            .expect("prefix plus padded header, no payload supplied");
        assert_eq!(parsed.payload_range(), 520..524);
        assert_eq!(
            parsed.tensor("weight").expect("weight").file_range(),
            520..524
        );
        assert_eq!(
            parsed,
            V41SafetensorsHeader::parse(&header, 524).expect("same body contract")
        );
        metadata.push(0);
        assert!(matches!(
            V41SafetensorsHeader::parse_prefixed_header(&metadata, 524),
            Err(V41SafetensorsHeaderError::HeaderLengthMismatch {
                declared_bytes: 512,
                actual_bytes: 513,
            })
        ));
    }

    #[test]
    fn prefixed_header_checks_lengths_before_json_or_allocation() {
        for length in 0..8 {
            assert!(matches!(
                V41SafetensorsHeader::parse_prefixed_header(&[0; 8][..length], 8),
                Err(V41SafetensorsHeaderError::PrefixTooShort { actual_bytes })
                    if actual_bytes == length
            ));
        }
        for declared in [super::MAX_HEADER_BYTES + 1, u64::MAX] {
            assert!(matches!(
                V41SafetensorsHeader::parse_prefixed_header(&declared.to_le_bytes(), u64::MAX),
                Err(V41SafetensorsHeaderError::HeaderTooLarge { header_bytes })
                    if header_bytes == declared
            ));
        }
        assert!(matches!(
            V41SafetensorsHeader::parse_prefixed_header(&64_u64.to_le_bytes(), 71),
            Err(V41SafetensorsHeaderError::FileLengthTooSmall {
                file_bytes: 71,
                required_bytes: 72,
            })
        ));
        assert!(matches!(
            V41SafetensorsHeader::parse_prefixed_header(&64_u64.to_le_bytes(), 72),
            Err(V41SafetensorsHeaderError::HeaderLengthMismatch {
                declared_bytes: 64,
                actual_bytes: 0,
            })
        ));
        assert!(matches!(
            V41SafetensorsHeader::parse_prefixed_header(&0_u64.to_le_bytes(), 7),
            Err(V41SafetensorsHeaderError::FileLengthTooSmall {
                file_bytes: 7,
                required_bytes: 8,
            })
        ));
    }

    #[test]
    fn prefixed_header_still_rejects_invalid_json_and_packed_fp4() {
        let mut invalid = 1_u64.to_le_bytes().to_vec();
        invalid.push(b'!');
        assert!(matches!(
            V41SafetensorsHeader::parse_prefixed_header(&invalid, 9),
            Err(V41SafetensorsHeaderError::HeaderJson(_))
        ));
        let header = br#"{"weight":{"dtype":"F4_E2M1FN_X2","shape":[32],"data_offsets":[0,16]}}"#;
        let mut metadata = u64::try_from(header.len())
            .expect("small header")
            .to_le_bytes()
            .to_vec();
        metadata.extend_from_slice(header);
        assert!(matches!(
            V41SafetensorsHeader::parse_prefixed_header(&metadata, file_bytes(header, 16)),
            Err(V41SafetensorsHeaderError::UnsupportedTensorDtype { .. })
        ));
    }

    #[test]
    fn validates_absolute_ranges_scalars_and_empty_tensors() {
        let header = br#"{
          "__metadata__":{"format":"pt"},
          "scalar":{"dtype":"F32","shape":[],"data_offsets":[0,4]},
          "empty":{"dtype":"U8","shape":[0,3],"data_offsets":[4,4]},
          "weight":{"dtype":"BF16","shape":[2],"data_offsets":[4,8]}
        }"#;
        let parsed = V41SafetensorsHeader::parse(header, file_bytes(header, 8))
            .expect("valid bounded header");

        assert_eq!(parsed.file_bytes(), file_bytes(header, 8));
        assert_eq!(
            parsed.payload_range(),
            (8 + header.len() as u64)..file_bytes(header, 8)
        );
        let scalar = parsed.tensor("scalar").expect("scalar range");
        assert_eq!(scalar.dtype(), V41StorageDtype::F32);
        assert!(scalar.shape().is_empty());
        assert_eq!(
            scalar.file_range(),
            (8 + header.len() as u64)..(12 + header.len() as u64)
        );
        assert_eq!(
            parsed.tensor("empty").expect("empty range").byte_length(),
            0
        );
        assert_eq!(
            parsed.tensor("weight").expect("weight range").file_range(),
            (12 + header.len() as u64)..(16 + header.len() as u64)
        );
    }

    #[test]
    fn rejects_unsupported_packed_dtype_without_inferring_its_layout() {
        let header = br#"{"weight":{"dtype":"F4_E2M1FN_X2","shape":[32],"data_offsets":[0,16]}}"#;
        let error = V41SafetensorsHeader::parse(header, file_bytes(header, 16))
            .expect_err("raw FP4 storage is intentionally not established");
        assert!(matches!(
            error,
            V41SafetensorsHeaderError::UnsupportedTensorDtype { dtype, .. }
                if dtype == "F4_E2M1FN_X2"
        ));
    }

    #[test]
    fn rejects_invalid_ranges_and_ambiguous_keys() {
        let out_of_file = br#"{"weight":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}"#;
        assert!(matches!(
            V41SafetensorsHeader::parse(out_of_file, file_bytes(out_of_file, 4)),
            Err(V41SafetensorsHeaderError::TensorOutsidePayload { .. })
        ));

        let overlap = br#"{
          "first":{"dtype":"U8","shape":[2],"data_offsets":[0,2]},
          "second":{"dtype":"U8","shape":[2],"data_offsets":[1,3]}
        }"#;
        assert!(matches!(
            V41SafetensorsHeader::parse(overlap, file_bytes(overlap, 3)),
            Err(V41SafetensorsHeaderError::OverlappingTensorRanges)
        ));

        let hole = br#"{"weight":{"dtype":"U8","shape":[1],"data_offsets":[1,2]}}"#;
        assert!(matches!(
            V41SafetensorsHeader::parse(hole, file_bytes(hole, 2)),
            Err(V41SafetensorsHeaderError::NonContiguousPayload)
        ));

        let trailing = br#"{"weight":{"dtype":"U8","shape":[1],"data_offsets":[0,1]}}"#;
        assert!(matches!(
            V41SafetensorsHeader::parse(trailing, file_bytes(trailing, 2)),
            Err(V41SafetensorsHeaderError::NonContiguousPayload)
        ));

        let duplicate = br#"{"weight":{"dtype":"U8","shape":[1],"data_offsets":[0,1]},"\u0077eight":{"dtype":"U8","shape":[1],"data_offsets":[0,1]}}"#;
        assert!(matches!(
            V41SafetensorsHeader::parse(duplicate, file_bytes(duplicate, 1)),
            Err(V41SafetensorsHeaderError::HeaderJson(_))
        ));
    }

    #[test]
    fn rejects_oversized_header_before_json_parse() {
        let header = br"{}";
        assert!(matches!(
            parse_with_header_limit(header, file_bytes(header, 0), 1),
            Err(V41SafetensorsHeaderError::HeaderTooLarge { header_bytes: 2 })
        ));
    }

    #[test]
    fn supports_zero_element_shapes_independent_of_dimension_order() {
        for header in [
            &br#"{"empty":{"dtype":"F32","shape":[18446744073709551615,2,0],"data_offsets":[0,0]}}"#[..],
            &br#"{"empty":{"dtype":"F32","shape":[0,18446744073709551615,2],"data_offsets":[0,0]}}"#[..],
        ] {
            V41SafetensorsHeader::parse(header, file_bytes(header, 0))
                .expect("a zero dimension makes this a valid empty tensor");
        }

        let overflowing =
            br#"{"weight":{"dtype":"F32","shape":[18446744073709551615,2],"data_offsets":[0,0]}}"#;
        assert!(matches!(
            V41SafetensorsHeader::parse(overflowing, file_bytes(overflowing, 0)),
            Err(V41SafetensorsHeaderError::TensorShapeOverflow { .. })
        ));
    }

    #[test]
    fn checks_exact_header_and_index_shard_name_sets() {
        let header = br#"{"weight":{"dtype":"U8","shape":[1],"data_offsets":[0,1]}}"#;
        let parsed =
            V41SafetensorsHeader::parse(header, file_bytes(header, 1)).expect("valid header");
        let index = V41SafetensorsIndex::parse(
            r#"{"metadata":{"total_size":1},"weight_map":{"weight":"model-00001.safetensors","other":"model-00002.safetensors"}}"#,
        )
        .expect("valid index");
        parsed
            .validate_index_shard(&index, "model-00001.safetensors")
            .expect("matching shard tensor set");
        assert!(matches!(
            parsed.validate_index_shard(&index, "model-00002.safetensors"),
            Err(V41SafetensorsHeaderError::TensorAssignedToOtherShard { .. })
        ));

        let missing = V41SafetensorsIndex::parse(
            r#"{"metadata":{"total_size":1},"weight_map":{"other":"model-00001.safetensors"}}"#,
        )
        .expect("valid index with a different tensor");
        assert!(matches!(
            parsed.validate_index_shard(&missing, "model-00001.safetensors"),
            Err(V41SafetensorsHeaderError::TensorMissingFromIndex { .. })
        ));

        let missing_from_header = V41SafetensorsIndex::parse(
            r#"{"metadata":{"total_size":2},"weight_map":{"weight":"model-00001.safetensors","other":"model-00001.safetensors"}}"#,
        )
        .expect("valid index with an additional shard tensor");
        assert!(matches!(
            parsed.validate_index_shard(&missing_from_header, "model-00001.safetensors"),
            Err(V41SafetensorsHeaderError::TensorMissingFromHeader { tensor }) if tensor == "other"
        ));
    }
}
