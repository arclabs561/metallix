//! Selected source-I8 expert weight/scale metadata and bounded payload reads.
//!
//! The pinned converter treats source `I8` expert weights as packed pairs: a
//! `[N, P]` source tensor describes a logical `[N, 2P]` matrix with one E8M0
//! scale per 32 logical reduction elements. This descriptor establishes only
//! that selected header and index metadata has that shape. The pair descriptor
//! does not read a file, identify a revision, decode payload bytes, establish
//! a packing convention, or support other checkpoint layouts. The separate
//! three-pair reader verifies the local header again and returns only its six
//! exact raw ranges; execution must bind those bytes to a separately qualified
//! source-packing contract.

use std::{
    fs::File,
    io::{Read, Seek, SeekFrom},
    path::Path,
};

use thiserror::Error;

use super::{
    MAX_HEADER_BYTES, V41SafetensorsHeader, V41SafetensorsHeaderError, V41StorageDtype,
    V41TensorRange,
};
use crate::manifest::V41SafetensorsIndex;

const WEIGHT_GROUP: u64 = 32;

/// The largest payload a one-expert reader may materialize.
///
/// Callers supply their own stricter limit. This ceiling prevents a malformed
/// microartifact request from becoming a whole-checkpoint allocation.
pub(crate) const MAX_EXPERT_PAYLOAD_BYTES: u64 = 32 * 1024 * 1024;

/// The canonical routed-expert projection named by a source checkpoint tensor.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum V41ExpertProjection {
    /// The expert's first gate projection.
    W1,
    /// The expert's down projection.
    W2,
    /// The expert's up projection.
    W3,
}

/// A validated selected source-I8 routed-expert weight and its E8M0 scale.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct V41ExpertI8ScalePair {
    layer: u64,
    expert: u64,
    projection: V41ExpertProjection,
    shard: String,
    weight_name: String,
    scale_name: String,
    weight_range: V41TensorRange,
    scale_range: V41TensorRange,
    logical_shape: [u64; 2],
}

impl V41ExpertI8ScalePair {
    /// Parses one exact canonical expert weight name and its paired metadata.
    ///
    /// Only `layers.<digits>.ffn.experts.<digits>.w1|w2|w3.weight` names are
    /// accepted. Both the exact weight and its exact `.scale` sibling must be
    /// assigned to `shard` by `index` and present in `header`. This deliberately
    /// validates selected entries only, not full header/index agreement.
    /// Layer and expert identifiers are syntactic only. The caller establishes
    /// model-config bounds and binds the supplied metadata to an actual file and
    /// revision.
    ///
    /// # Errors
    ///
    /// Returns [`V41ExpertI8ScalePairError`] for a noncanonical name, missing
    /// or differently assigned selected entry, or incompatible selected dtype
    /// or shape metadata.
    pub fn parse(
        header: &V41SafetensorsHeader,
        index: &V41SafetensorsIndex,
        shard: &str,
        weight_name: &str,
    ) -> Result<Self, V41ExpertI8ScalePairError> {
        let (layer, expert, projection) = parse_weight_name(weight_name)?;
        let base = weight_name
            .strip_suffix(".weight")
            .ok_or_else(|| invalid_name(weight_name))?;
        let scale_name = format!("{base}.scale");
        validate_index(index, shard, weight_name)?;
        validate_index(index, shard, &scale_name)?;
        let weight_range = header.tensor(weight_name).ok_or_else(|| {
            V41ExpertI8ScalePairError::MissingHeaderTensor {
                tensor: weight_name.to_owned(),
            }
        })?;
        let scale_range = header.tensor(&scale_name).ok_or_else(|| {
            V41ExpertI8ScalePairError::MissingHeaderTensor {
                tensor: scale_name.clone(),
            }
        })?;
        let logical_shape = validate_shapes(weight_range, scale_range)?;
        Ok(Self {
            layer,
            expert,
            projection,
            shard: shard.to_owned(),
            weight_name: weight_name.to_owned(),
            scale_name,
            weight_range: weight_range.clone(),
            scale_range: scale_range.clone(),
            logical_shape,
        })
    }

    /// Returns the parsed canonical layer number.
    #[must_use]
    pub const fn layer(&self) -> u64 {
        self.layer
    }

    /// Returns the parsed canonical routed-expert number.
    #[must_use]
    pub const fn expert(&self) -> u64 {
        self.expert
    }

    /// Returns the selected expert projection.
    #[must_use]
    pub const fn projection(&self) -> V41ExpertProjection {
        self.projection
    }

    /// Returns the caller-declared shard name.
    #[must_use]
    pub fn shard(&self) -> &str {
        &self.shard
    }

    /// Returns the exact selected source-I8 tensor name.
    #[must_use]
    pub fn weight_name(&self) -> &str {
        &self.weight_name
    }

    /// Returns the exact selected E8M0 scale tensor name.
    #[must_use]
    pub fn scale_name(&self) -> &str {
        &self.scale_name
    }

    /// Returns the cloned, validated source-I8 tensor interval metadata.
    #[must_use]
    pub fn weight_range(&self) -> &V41TensorRange {
        &self.weight_range
    }

    /// Returns the cloned, validated E8M0 scale interval metadata.
    #[must_use]
    pub fn scale_range(&self) -> &V41TensorRange {
        &self.scale_range
    }

    /// Returns the logical unpacked source matrix shape `[N, K]`.
    #[must_use]
    pub const fn logical_shape(&self) -> [u64; 2] {
        self.logical_shape
    }
}

/// The three validated packed projections for one exact routed expert.
///
/// This binds selected `w1`, `w2`, and `w3` pairs to one layer, expert, index
/// shard, and safetensors header. It reads a bounded payload but does not
/// establish source packing or execute the expert.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct V41ExpertI8ScalePairs {
    w1: V41ExpertI8ScalePair,
    w2: V41ExpertI8ScalePair,
    w3: V41ExpertI8ScalePair,
}

impl V41ExpertI8ScalePairs {
    /// Validates all three canonical projection pairs for one routed expert.
    pub fn parse(
        header: &V41SafetensorsHeader,
        index: &V41SafetensorsIndex,
        shard: &str,
        layer: u64,
        expert: u64,
    ) -> Result<Self, V41ExpertPayloadError> {
        header
            .validate_index_shard(index, shard)
            .map_err(V41ExpertPayloadError::HeaderIndex)?;
        let prefix = format!("layers.{layer}.ffn.experts.{expert}");
        let w1 = V41ExpertI8ScalePair::parse(header, index, shard, &format!("{prefix}.w1.weight"))?;
        let w2 = V41ExpertI8ScalePair::parse(header, index, shard, &format!("{prefix}.w2.weight"))?;
        let w3 = V41ExpertI8ScalePair::parse(header, index, shard, &format!("{prefix}.w3.weight"))?;
        validate_expert_geometry(&w1, &w2, &w3)?;
        Ok(Self { w1, w2, w3 })
    }

    /// Returns the exact first gate projection metadata.
    #[must_use]
    pub const fn w1(&self) -> &V41ExpertI8ScalePair {
        &self.w1
    }

    /// Returns the exact down projection metadata.
    #[must_use]
    pub const fn w2(&self) -> &V41ExpertI8ScalePair {
        &self.w2
    }

    /// Returns the exact up projection metadata.
    #[must_use]
    pub const fn w3(&self) -> &V41ExpertI8ScalePair {
        &self.w3
    }

    /// Returns the group-aligned hidden width shared by `w1` and `w3`.
    #[must_use]
    pub const fn hidden_width(&self) -> u64 {
        self.w1.logical_shape()[1]
    }

    /// Returns the group-aligned routed-expert intermediate width.
    #[must_use]
    pub const fn intermediate_width(&self) -> u64 {
        self.w1.logical_shape()[0]
    }

    /// Reads exactly the six selected payload intervals from a local shard.
    ///
    /// The file's bounded prefix/header is reparsed and compared with `header`
    /// before payload allocation or reading. Its regular-file length must equal
    /// the header's declared complete shard length. The selected intervals are
    /// budgeted before allocating any returned buffer.
    pub fn read_local_shard(
        &self,
        shard: &Path,
        header: &V41SafetensorsHeader,
        max_bytes: u64,
    ) -> Result<V41ExpertI8ScalePayload, V41ExpertPayloadError> {
        let total_bytes = self.total_payload_bytes()?;
        if total_bytes > max_bytes || total_bytes > MAX_EXPERT_PAYLOAD_BYTES {
            return Err(V41ExpertPayloadError::PayloadBudget {
                requested_bytes: total_bytes,
                max_bytes: max_bytes.min(MAX_EXPERT_PAYLOAD_BYTES),
            });
        }
        let mut file = File::open(shard).map_err(V41ExpertPayloadError::Io)?;
        let metadata = file.metadata().map_err(V41ExpertPayloadError::Io)?;
        if !metadata.is_file() {
            return Err(V41ExpertPayloadError::NotRegularFile);
        }
        if metadata.len() != header.file_bytes() {
            return Err(V41ExpertPayloadError::ShardLength {
                actual_bytes: metadata.len(),
                expected_bytes: header.file_bytes(),
            });
        }
        let actual_header = read_and_parse_header(&mut file, metadata.len())?;
        if actual_header != *header {
            return Err(V41ExpertPayloadError::HeaderMismatch);
        }
        self.bind_header(&actual_header)?;
        let w1 = read_pair(&mut file, &self.w1)?;
        let w2 = read_pair(&mut file, &self.w2)?;
        let w3 = read_pair(&mut file, &self.w3)?;
        Ok(V41ExpertI8ScalePayload { w1, w2, w3 })
    }

    fn total_payload_bytes(&self) -> Result<u64, V41ExpertPayloadError> {
        [&self.w1, &self.w2, &self.w3]
            .into_iter()
            .try_fold(0_u64, |total, pair| {
                total
                    .checked_add(pair.weight_range().byte_length())
                    .and_then(|value| value.checked_add(pair.scale_range().byte_length()))
                    .ok_or(V41ExpertPayloadError::PayloadLengthOverflow)
            })
    }

    fn bind_header(&self, header: &V41SafetensorsHeader) -> Result<(), V41ExpertPayloadError> {
        for pair in [&self.w1, &self.w2, &self.w3] {
            let weight = header.tensor(pair.weight_name()).ok_or_else(|| {
                V41ExpertPayloadError::PairHeaderMismatch {
                    tensor: pair.weight_name().to_owned(),
                }
            })?;
            let scale = header.tensor(pair.scale_name()).ok_or_else(|| {
                V41ExpertPayloadError::PairHeaderMismatch {
                    tensor: pair.scale_name().to_owned(),
                }
            })?;
            if weight != pair.weight_range() {
                return Err(V41ExpertPayloadError::PairHeaderMismatch {
                    tensor: pair.weight_name().to_owned(),
                });
            }
            if scale != pair.scale_range() {
                return Err(V41ExpertPayloadError::PairHeaderMismatch {
                    tensor: pair.scale_name().to_owned(),
                });
            }
        }
        Ok(())
    }
}

/// Exact packed bytes for one projection of a selected routed expert.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct V41ExpertProjectionPayload {
    weight: Vec<u8>,
    scales: Vec<u8>,
}

impl V41ExpertProjectionPayload {
    /// Returns exact packed source-I8 bytes in header range order.
    #[must_use]
    pub fn weight(&self) -> &[u8] {
        &self.weight
    }

    /// Returns exact E8M0 bytes in header range order.
    #[must_use]
    pub fn scales(&self) -> &[u8] {
        &self.scales
    }
}

/// Exact bounded payload bytes for `w1`, `w2`, and `w3` of one expert.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct V41ExpertI8ScalePayload {
    w1: V41ExpertProjectionPayload,
    w2: V41ExpertProjectionPayload,
    w3: V41ExpertProjectionPayload,
}

impl V41ExpertI8ScalePayload {
    /// Returns first gate projection bytes.
    #[must_use]
    pub const fn w1(&self) -> &V41ExpertProjectionPayload {
        &self.w1
    }

    /// Returns down projection bytes.
    #[must_use]
    pub const fn w2(&self) -> &V41ExpertProjectionPayload {
        &self.w2
    }

    /// Returns up projection bytes.
    #[must_use]
    pub const fn w3(&self) -> &V41ExpertProjectionPayload {
        &self.w3
    }
}

/// A local selected-expert payload did not meet the bounded source contract.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum V41ExpertPayloadError {
    /// The selected header and index do not agree for the named shard.
    #[error("selected expert header/index identity failed: {0}")]
    HeaderIndex(V41SafetensorsHeaderError),
    /// One selected pair is malformed or assigned differently by the index.
    #[error("selected expert pair failed: {0}")]
    Pair(#[from] V41ExpertI8ScalePairError),
    /// Selected projections do not form the expected one-expert geometry.
    #[error("selected expert projections do not have w1/w3 equal and w2 transposed geometry")]
    ProjectionGeometry,
    /// Selected ranges exceed the caller's or reader's allocation cap.
    #[error(
        "selected expert payload needs {requested_bytes} bytes, above the {max_bytes}-byte limit"
    )]
    PayloadBudget {
        requested_bytes: u64,
        max_bytes: u64,
    },
    /// Summing selected ranges overflowed before allocation.
    #[error("selected expert payload length overflowed")]
    PayloadLengthOverflow,
    /// The supplied shard path is not a regular file.
    #[error("selected expert shard is not a regular file")]
    NotRegularFile,
    /// Local shard length differs from supplied header's declared length.
    #[error("selected expert shard length is {actual_bytes}, expected {expected_bytes}")]
    ShardLength {
        actual_bytes: u64,
        expected_bytes: u64,
    },
    /// Local bounded header differs from the header that selected the ranges.
    #[error("selected expert shard header differs from supplied header")]
    HeaderMismatch,
    /// A cached selected range no longer matches the revalidated shard header.
    #[error("selected expert cached range differs from revalidated header tensor {tensor}")]
    PairHeaderMismatch {
        /// Exact selected tensor whose cached range is stale or absent.
        tensor: String,
    },
    /// The local bounded header is malformed.
    #[error("could not parse selected expert shard header: {0}")]
    Header(#[from] V41SafetensorsHeaderError),
    /// An exact local-range read failed.
    #[error("could not read selected expert payload: {0}")]
    Io(#[source] std::io::Error),
    /// Reserving one already-budgeted selected range failed.
    #[error("could not allocate selected expert payload range")]
    Allocation,
    /// An E8M0 scale code denotes NaN.
    #[error("selected expert {projection:?} scale {index} is nonfinite")]
    NonFiniteScale {
        projection: V41ExpertProjection,
        index: usize,
    },
}

fn validate_expert_geometry(
    w1: &V41ExpertI8ScalePair,
    w2: &V41ExpertI8ScalePair,
    w3: &V41ExpertI8ScalePair,
) -> Result<(), V41ExpertPayloadError> {
    let same_identity = [w1, w2, w3].into_iter().all(|pair| {
        pair.layer() == w1.layer() && pair.expert() == w1.expert() && pair.shard() == w1.shard()
    });
    let [intermediate, hidden] = w1.logical_shape();
    if !same_identity
        || w1.projection() != V41ExpertProjection::W1
        || w2.projection() != V41ExpertProjection::W2
        || w3.projection() != V41ExpertProjection::W3
        || w3.logical_shape() != [intermediate, hidden]
        || w2.logical_shape() != [hidden, intermediate]
    {
        return Err(V41ExpertPayloadError::ProjectionGeometry);
    }
    Ok(())
}

fn read_and_parse_header(
    file: &mut File,
    file_bytes: u64,
) -> Result<V41SafetensorsHeader, V41ExpertPayloadError> {
    file.seek(SeekFrom::Start(0))
        .map_err(V41ExpertPayloadError::Io)?;
    let mut prefix = [0_u8; 8];
    file.read_exact(&mut prefix)
        .map_err(V41ExpertPayloadError::Io)?;
    let header_bytes = u64::from_le_bytes(prefix);
    if header_bytes > MAX_HEADER_BYTES {
        return Err(V41SafetensorsHeaderError::HeaderTooLarge { header_bytes }.into());
    }
    let header_len = usize::try_from(header_bytes)
        .map_err(|_| V41SafetensorsHeaderError::HeaderTooLarge { header_bytes })?;
    let mut prefixed_header = Vec::with_capacity(8 + header_len);
    prefixed_header.extend_from_slice(&prefix);
    prefixed_header.resize(8 + header_len, 0);
    file.read_exact(&mut prefixed_header[8..])
        .map_err(V41ExpertPayloadError::Io)?;
    V41SafetensorsHeader::parse_prefixed_header(&prefixed_header, file_bytes)
        .map_err(V41ExpertPayloadError::Header)
}

fn read_pair(
    file: &mut File,
    pair: &V41ExpertI8ScalePair,
) -> Result<V41ExpertProjectionPayload, V41ExpertPayloadError> {
    let weight = read_range(file, pair.weight_range())?;
    let scales = read_range(file, pair.scale_range())?;
    if let Some((index, _)) = scales
        .iter()
        .enumerate()
        .find(|(_, code)| **code == u8::MAX)
    {
        return Err(V41ExpertPayloadError::NonFiniteScale {
            projection: pair.projection(),
            index,
        });
    }
    Ok(V41ExpertProjectionPayload { weight, scales })
}

fn read_range(file: &mut File, range: &V41TensorRange) -> Result<Vec<u8>, V41ExpertPayloadError> {
    let length = usize::try_from(range.byte_length())
        .map_err(|_| V41ExpertPayloadError::PayloadLengthOverflow)?;
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(length)
        .map_err(|_| V41ExpertPayloadError::Allocation)?;
    bytes.resize(length, 0);
    file.seek(SeekFrom::Start(range.file_range().start))
        .and_then(|_| file.read_exact(&mut bytes))
        .map_err(V41ExpertPayloadError::Io)?;
    Ok(bytes)
}

/// A selected source-I8 expert pair does not meet the bounded metadata contract.
#[derive(Clone, Debug, Error, Eq, PartialEq)]
#[non_exhaustive]
pub enum V41ExpertI8ScalePairError {
    /// The supplied name is not exactly one supported canonical routed-expert weight.
    #[error("unsupported canonical source expert weight name {weight_name}")]
    InvalidWeightName {
        /// Caller-supplied source tensor name.
        weight_name: String,
    },
    /// A selected tensor is absent from the already-validated index.
    #[error("selected tensor {tensor} is absent from the safetensors index")]
    MissingIndexTensor {
        /// Exact selected tensor name.
        tensor: String,
    },
    /// A selected tensor is assigned to a shard other than the caller-declared shard.
    #[error("selected tensor {tensor} is assigned to {indexed_shard}, not {expected_shard}")]
    WrongIndexShard {
        /// Exact selected tensor name.
        tensor: String,
        /// Caller-declared shard.
        expected_shard: String,
        /// Index-declared shard.
        indexed_shard: String,
    },
    /// A selected tensor is absent from the already-validated header.
    #[error("selected tensor {tensor} is absent from the safetensors header")]
    MissingHeaderTensor {
        /// Exact selected tensor name.
        tensor: String,
    },
    /// The source weight dtype is not I8.
    #[error("source expert weight dtype is {actual:?}, expected I8")]
    WeightDtype {
        /// Header-declared dtype.
        actual: V41StorageDtype,
    },
    /// The source scale dtype is not `F8E8M0Fnu`.
    #[error("source expert scale dtype is {actual:?}, expected F8E8M0Fnu")]
    ScaleDtype {
        /// Header-declared dtype.
        actual: V41StorageDtype,
    },
    /// The source-I8 weight is not a nonzero rank-two `[N, P]` tensor.
    #[error("source expert weight shape must be nonzero rank-two [N, P]")]
    WeightShape,
    /// Doubling packed source columns does not fit u64.
    #[error("source expert logical reduction width overflows u64")]
    LogicalReductionOverflow,
    /// The logical reduction width is not divisible by 32.
    #[error("source expert logical reduction width {reduction} is not divisible by 32")]
    LogicalReductionNotGrouped {
        /// Doubled packed-column count.
        reduction: u64,
    },
    /// The E8M0 scale is not exactly rank-two `[N, K / 32]`.
    #[error("source expert scale shape must be exactly [N, K / 32]")]
    ScaleShape,
}

fn parse_weight_name(
    weight_name: &str,
) -> Result<(u64, u64, V41ExpertProjection), V41ExpertI8ScalePairError> {
    let mut parts = weight_name.split('.');
    let parsed = match (
        parts.next(),
        parts.next(),
        parts.next(),
        parts.next(),
        parts.next(),
        parts.next(),
        parts.next(),
        parts.next(),
    ) {
        (
            Some("layers"),
            Some(layer),
            Some("ffn"),
            Some("experts"),
            Some(expert),
            Some(projection),
            Some("weight"),
            None,
        ) => {
            let projection = match projection {
                "w1" => V41ExpertProjection::W1,
                "w2" => V41ExpertProjection::W2,
                "w3" => V41ExpertProjection::W3,
                _ => return Err(invalid_name(weight_name)),
            };
            match (canonical_u64(layer), canonical_u64(expert)) {
                (Some(layer), Some(expert)) => (layer, expert, projection),
                _ => return Err(invalid_name(weight_name)),
            }
        }
        _ => return Err(invalid_name(weight_name)),
    };
    Ok(parsed)
}

fn canonical_u64(text: &str) -> Option<u64> {
    if text.is_empty()
        || (text.len() > 1 && text.starts_with('0'))
        || !text.bytes().all(|b| b.is_ascii_digit())
    {
        return None;
    }
    text.parse().ok()
}

fn invalid_name(weight_name: &str) -> V41ExpertI8ScalePairError {
    V41ExpertI8ScalePairError::InvalidWeightName {
        weight_name: weight_name.to_owned(),
    }
}

fn validate_index(
    index: &V41SafetensorsIndex,
    shard: &str,
    tensor: &str,
) -> Result<(), V41ExpertI8ScalePairError> {
    let indexed_shard = index.shard_for_tensor(tensor).ok_or_else(|| {
        V41ExpertI8ScalePairError::MissingIndexTensor {
            tensor: tensor.to_owned(),
        }
    })?;
    if indexed_shard != shard {
        return Err(V41ExpertI8ScalePairError::WrongIndexShard {
            tensor: tensor.to_owned(),
            expected_shard: shard.to_owned(),
            indexed_shard: indexed_shard.to_owned(),
        });
    }
    Ok(())
}

fn validate_shapes(
    weight: &V41TensorRange,
    scale: &V41TensorRange,
) -> Result<[u64; 2], V41ExpertI8ScalePairError> {
    if weight.dtype() != V41StorageDtype::I8 {
        return Err(V41ExpertI8ScalePairError::WeightDtype {
            actual: weight.dtype(),
        });
    }
    let &[rows, packed_columns] = weight.shape() else {
        return Err(V41ExpertI8ScalePairError::WeightShape);
    };
    if rows == 0 || packed_columns == 0 {
        return Err(V41ExpertI8ScalePairError::WeightShape);
    }
    let reduction = packed_columns
        .checked_mul(2)
        .ok_or(V41ExpertI8ScalePairError::LogicalReductionOverflow)?;
    if !reduction.is_multiple_of(WEIGHT_GROUP) {
        return Err(V41ExpertI8ScalePairError::LogicalReductionNotGrouped { reduction });
    }
    if scale.dtype() != V41StorageDtype::F8E8M0Fnu {
        return Err(V41ExpertI8ScalePairError::ScaleDtype {
            actual: scale.dtype(),
        });
    }
    if scale.shape() != [rows, reduction / WEIGHT_GROUP] {
        return Err(V41ExpertI8ScalePairError::ScaleShape);
    }
    Ok([rows, reduction])
}

#[cfg(test)]
mod tests {
    use std::{
        fs::{self, File, OpenOptions},
        io::{Seek, SeekFrom, Write},
        path::PathBuf,
        sync::atomic::{AtomicU64, Ordering},
    };

    use super::{
        V41ExpertI8ScalePair, V41ExpertI8ScalePairError, V41ExpertI8ScalePairs,
        V41ExpertPayloadError, V41ExpertProjection,
    };
    use crate::{checkpoint::V41SafetensorsHeader, manifest::V41SafetensorsIndex};

    const SHARD: &str = "model-00009-of-00048.safetensors";
    static TEST_SHARD_ID: AtomicU64 = AtomicU64::new(0);

    fn header(
        weight_name: &str,
        weight_dtype: &str,
        weight_shape: &[u64],
        scale_dtype: &str,
        scale_shape: &[u64],
    ) -> V41SafetensorsHeader {
        let scale_name = weight_name
            .strip_suffix(".weight")
            .expect("weight name")
            .to_owned()
            + ".scale";
        let weight_bytes = elements(weight_shape) * bytes_per_element(weight_dtype);
        let scale_bytes = elements(scale_shape) * bytes_per_element(scale_dtype);
        let json = format!(
            r#"{{"{weight_name}":{{"dtype":"{weight_dtype}","shape":{weight_shape:?},"data_offsets":[0,{weight_bytes}]}},"{scale_name}":{{"dtype":"{scale_dtype}","shape":{scale_shape:?},"data_offsets":[{weight_bytes},{}]}}}}"#,
            weight_bytes + scale_bytes
        );
        V41SafetensorsHeader::parse(
            json.as_bytes(),
            8 + u64::try_from(json.len()).expect("small test header") + weight_bytes + scale_bytes,
        )
        .expect("bounded test header")
    }

    fn elements(shape: &[u64]) -> u64 {
        shape.iter().copied().product()
    }

    fn bytes_per_element(dtype: &str) -> u64 {
        match dtype {
            "F32" => 4,
            _ => 1,
        }
    }

    fn index(
        weight_name: &str,
        scale: bool,
        shard: &str,
        unrelated: Option<&str>,
    ) -> V41SafetensorsIndex {
        use std::fmt::Write as _;

        let scale_name = weight_name
            .strip_suffix(".weight")
            .expect("weight name")
            .to_owned()
            + ".scale";
        let mut mappings = format!(r#""{weight_name}":"{shard}""#);
        if scale {
            write!(mappings, r#", "{scale_name}":"{shard}""#).expect("write test mapping");
        }
        if let Some(unrelated_shard) = unrelated {
            write!(mappings, r#", "unrelated":"{unrelated_shard}""#).expect("write test mapping");
        }
        V41SafetensorsIndex::parse(&format!(
            r#"{{"metadata":{{"total_size":1}},"weight_map":{{{mappings}}}}}"#
        ))
        .expect("bounded test index")
    }

    #[test]
    fn parses_each_projection_without_requiring_unrelated_agreement() {
        for (projection_name, projection) in [
            ("w1", V41ExpertProjection::W1),
            ("w2", V41ExpertProjection::W2),
            ("w3", V41ExpertProjection::W3),
        ] {
            let name = format!("layers.12.ffn.experts.34.{projection_name}.weight");
            let header = header(&name, "I8", &[2, 16], "F8_E8M0FNU", &[2, 1]);
            let index = index(&name, true, SHARD, Some("another-shard.safetensors"));
            let pair = V41ExpertI8ScalePair::parse(&header, &index, SHARD, &name)
                .expect("selected pair ignores unrelated disagreement");
            assert_eq!(pair.layer(), 12);
            assert_eq!(pair.expert(), 34);
            assert_eq!(pair.projection(), projection);
            assert_eq!(pair.shard(), SHARD);
            assert_eq!(pair.weight_name(), name);
            assert_eq!(pair.scale_name(), name.replace(".weight", ".scale"));
            assert_eq!(pair.logical_shape(), [2, 32]);
            assert_eq!(pair.weight_range().shape(), [2, 16]);
            assert_eq!(pair.scale_range().shape(), [2, 1]);
        }
    }

    #[test]
    fn rejects_noncanonical_names_before_metadata_lookup() {
        let header = header(
            "layers.1.ffn.experts.2.w1.weight",
            "I8",
            &[2, 16],
            "F8_E8M0FNU",
            &[2, 1],
        );
        let index = index("layers.1.ffn.experts.2.w1.weight", true, SHARD, None);
        for name in [
            "model.layers.1.ffn.experts.2.w1.weight",
            "mtp.layers.1.ffn.experts.2.w1.weight",
            "layers.01.ffn.experts.2.w1.weight",
            "layers.1.ffn.experts.02.w1.weight",
            "layers.1.ffn.shared_experts.2.w1.weight",
            "layers.1.ffn.experts.2.w4.weight",
            "layers.1.ffn.experts.2.w1.scale",
            "vision.layers.1.ffn.experts.2.w1.weight",
            "layers.18446744073709551616.ffn.experts.2.w1.weight",
        ] {
            assert!(matches!(
                V41ExpertI8ScalePair::parse(&header, &index, SHARD, name),
                Err(V41ExpertI8ScalePairError::InvalidWeightName { .. })
            ));
        }
    }

    #[test]
    fn rejects_selected_index_and_header_absence_or_wrong_shard() {
        let name = "layers.1.ffn.experts.2.w1.weight";
        let header = header(name, "I8", &[2, 16], "F8_E8M0FNU", &[2, 1]);
        let missing_scale = index(name, false, SHARD, None);
        assert!(matches!(
            V41ExpertI8ScalePair::parse(&header, &missing_scale, SHARD, name),
            Err(V41ExpertI8ScalePairError::MissingIndexTensor { .. })
        ));
        let wrong_shard = index(name, true, "other.safetensors", None);
        assert!(matches!(
            V41ExpertI8ScalePair::parse(&header, &wrong_shard, SHARD, name),
            Err(V41ExpertI8ScalePairError::WrongIndexShard { .. })
        ));
        let split_index = V41SafetensorsIndex::parse(&format!(
            r#"{{"metadata":{{"total_size":34}},"weight_map":{{"{name}":"{SHARD}","layers.1.ffn.experts.2.w1.scale":"other.safetensors"}}}}"#
        ))
        .expect("selected scale alone is on another shard");
        assert!(matches!(
            V41ExpertI8ScalePair::parse(&header, &split_index, SHARD, name),
            Err(V41ExpertI8ScalePairError::WrongIndexShard { tensor, .. })
                if tensor == "layers.1.ffn.experts.2.w1.scale"
        ));

        let weight_bytes = 32_u64;
        let json = format!(
            r#"{{"{name}":{{"dtype":"I8","shape":[2,16],"data_offsets":[0,{weight_bytes}]}}}}"#
        );
        let only_weight = V41SafetensorsHeader::parse(
            json.as_bytes(),
            8 + u64::try_from(json.len()).expect("small test header") + weight_bytes,
        )
        .expect("bounded one-tensor header");
        let complete_index = index(name, true, SHARD, None);
        assert!(matches!(
            V41ExpertI8ScalePair::parse(&only_weight, &complete_index, SHARD, name),
            Err(V41ExpertI8ScalePairError::MissingHeaderTensor { .. })
        ));
    }

    #[test]
    fn rejects_dtypes_ranks_shapes_and_ungrouped_reduction() {
        let name = "layers.1.ffn.experts.2.w1.weight";
        for (weight_dtype, weight_shape, scale_dtype, scale_shape, expected) in [
            (
                "U8",
                &[2, 16][..],
                "F8_E8M0FNU",
                &[2, 1][..],
                "weight dtype",
            ),
            (
                "I8",
                &[2, 16, 1][..],
                "F8_E8M0FNU",
                &[2, 1][..],
                "weight rank",
            ),
            ("I8", &[0, 16][..], "F8_E8M0FNU", &[0, 1][..], "weight zero"),
            ("I8", &[2, 0][..], "F8_E8M0FNU", &[2, 0][..], "weight zero"),
            ("I8", &[2, 15][..], "F8_E8M0FNU", &[2, 1][..], "group"),
            ("I8", &[2, 16][..], "U8", &[2, 1][..], "scale dtype"),
            (
                "I8",
                &[2, 16][..],
                "F8_E8M0FNU",
                &[2, 1, 1][..],
                "scale rank",
            ),
            (
                "I8",
                &[2, 16][..],
                "F8_E8M0FNU",
                &[3, 1][..],
                "scale dimensions",
            ),
            (
                "I8",
                &[2, 16][..],
                "F8_E8M0FNU",
                &[2, 2][..],
                "scale dimensions",
            ),
        ] {
            let header = header(name, weight_dtype, weight_shape, scale_dtype, scale_shape);
            let index = index(name, true, SHARD, None);
            let error =
                V41ExpertI8ScalePair::parse(&header, &index, SHARD, name).expect_err(expected);
            match expected {
                "weight dtype" => assert!(matches!(
                    error,
                    V41ExpertI8ScalePairError::WeightDtype { .. }
                )),
                "weight rank" | "weight zero" => {
                    assert!(matches!(error, V41ExpertI8ScalePairError::WeightShape));
                }
                "group" => assert!(matches!(
                    error,
                    V41ExpertI8ScalePairError::LogicalReductionNotGrouped { .. }
                )),
                "scale dtype" => assert!(matches!(
                    error,
                    V41ExpertI8ScalePairError::ScaleDtype { .. }
                )),
                "scale rank" | "scale dimensions" => {
                    assert!(matches!(error, V41ExpertI8ScalePairError::ScaleShape));
                }
                _ => unreachable!("fixed test labels"),
            }
        }
    }

    #[test]
    fn rejects_logical_reduction_overflow_without_allocating_payload() {
        let name = "layers.1.ffn.experts.2.w1.weight";
        let packed_columns = u64::MAX / 2 + 1;
        let header = header(name, "I8", &[1, packed_columns], "F8_E8M0FNU", &[1, 1]);
        let index = index(name, true, SHARD, None);
        assert!(matches!(
            V41ExpertI8ScalePair::parse(&header, &index, SHARD, name),
            Err(V41ExpertI8ScalePairError::LogicalReductionOverflow)
        ));
    }

    struct TestShard {
        path: PathBuf,
        header: V41SafetensorsHeader,
        index: V41SafetensorsIndex,
        header_json: String,
    }

    impl Drop for TestShard {
        fn drop(&mut self) {
            let _ = fs::remove_file(&self.path);
        }
    }

    fn expert_shard(layer: u64, w2_shape: [u64; 2], swap_outer_names: bool) -> TestShard {
        let tensors = [
            ("w1", [32, 32], [32, 2], 0x11_u8, vec![127; 64]),
            (
                "w2",
                w2_shape,
                [w2_shape[0], w2_shape[1] * 2 / 32],
                0x22,
                vec![
                    127;
                    usize::try_from(w2_shape[0] * w2_shape[1] * 2 / 32).expect("small scales")
                ],
            ),
            ("w3", [32, 32], [32, 2], 0x33, vec![127; 64]),
        ];
        let names = if swap_outer_names {
            ["w3", "w2", "w1"]
        } else {
            ["w1", "w2", "w3"]
        };
        let mut offset = 0_u64;
        let mut entries = Vec::new();
        let mut payload = Vec::new();
        let mut mappings = Vec::new();
        for ((_, shape, scale_shape, byte, scales), projection) in tensors.into_iter().zip(names) {
            let name = format!("layers.{layer}.ffn.experts.11.{projection}");
            let weight_name = format!("{name}.weight");
            let scale_name = format!("{name}.scale");
            let weight_bytes = shape[0] * shape[1];
            let scale_bytes = scale_shape[0] * scale_shape[1];
            assert_eq!(
                u64::try_from(scales.len()).expect("small scales"),
                scale_bytes
            );
            entries.push(format!(
                r#""{weight_name}":{{"dtype":"I8","shape":[{},{}],"data_offsets":[{offset},{}]}}"#,
                shape[0],
                shape[1],
                offset + weight_bytes
            ));
            offset += weight_bytes;
            entries.push(format!(
                r#""{scale_name}":{{"dtype":"F8_E8M0FNU","shape":[{},{}],"data_offsets":[{offset},{}]}}"#,
                scale_shape[0], scale_shape[1], offset + scale_bytes
            ));
            offset += scale_bytes;
            mappings.push(format!(r#""{weight_name}":"{SHARD}""#));
            mappings.push(format!(r#""{scale_name}":"{SHARD}""#));
            payload.extend(vec![
                byte;
                usize::try_from(weight_bytes).expect("small weight")
            ]);
            payload.extend(scales);
        }
        let header_json = format!("{{{}}}", entries.join(","));
        let file_bytes = 8 + u64::try_from(header_json.len()).expect("small header") + offset;
        let header = V41SafetensorsHeader::parse(header_json.as_bytes(), file_bytes)
            .expect("valid synthetic expert header");
        let index = V41SafetensorsIndex::parse(&format!(
            r#"{{"metadata":{{"total_size":{file_bytes}}},"weight_map":{{{}}}}}"#,
            mappings.join(",")
        ))
        .expect("valid synthetic expert index");
        let path = std::env::temp_dir().join(format!(
            "metallix-v41-expert-payload-{}-{}.safetensors",
            std::process::id(),
            TEST_SHARD_ID.fetch_add(1, Ordering::Relaxed)
        ));
        let mut file = File::create(&path).expect("create synthetic shard");
        file.write_all(&(header_json.len() as u64).to_le_bytes())
            .expect("write prefix");
        file.write_all(header_json.as_bytes())
            .expect("write header");
        file.write_all(&payload).expect("write payload");
        TestShard {
            path,
            header,
            index,
            header_json,
        }
    }

    fn pairs(shard: &TestShard) -> V41ExpertI8ScalePairs {
        V41ExpertI8ScalePairs::parse(&shard.header, &shard.index, SHARD, 7, 11)
            .expect("valid selected expert")
    }

    #[test]
    fn reads_exact_six_ranges_only_after_local_header_identity_check() {
        let shard = expert_shard(7, [64, 16], false);
        let pairs = pairs(&shard);
        let payload = pairs
            .read_local_shard(&shard.path, &shard.header, 10_000)
            .expect("bounded six-range read");
        assert_eq!(payload.w1().weight(), vec![0x11; 1_024]);
        assert_eq!(payload.w1().scales(), vec![127; 64]);
        assert_eq!(payload.w2().weight(), vec![0x22; 1_024]);
        assert_eq!(payload.w2().scales(), vec![127; 64]);
        assert_eq!(payload.w3().weight(), vec![0x33; 1_024]);
        assert_eq!(payload.w3().scales(), vec![127; 64]);
    }

    #[test]
    fn rejects_budget_before_local_payload_allocation() {
        let shard = expert_shard(7, [64, 16], false);
        let pairs = pairs(&shard);
        let error = pairs
            .read_local_shard(&shard.path, &shard.header, 1)
            .expect_err("six ranges exceed one byte");
        assert!(matches!(error, V41ExpertPayloadError::PayloadBudget { .. }));
    }

    #[test]
    fn rejects_a_valid_but_stale_supplied_header_before_payload_read() {
        let shard = expert_shard(7, [64, 16], false);
        let pairs = pairs(&shard);
        let stale_json = shard.header_json.replacen("layers.7", "layers.8", 1);
        assert_eq!(stale_json.len(), shard.header_json.len());
        let stale_header =
            V41SafetensorsHeader::parse(stale_json.as_bytes(), shard.header.file_bytes())
                .expect("same-sized stale header remains structurally valid");
        assert!(matches!(
            pairs.read_local_shard(&shard.path, &stale_header, 10_000),
            Err(V41ExpertPayloadError::HeaderMismatch)
        ));
    }

    #[test]
    fn rejects_cached_pairs_from_a_different_compatible_shard_header() {
        let first = expert_shard(7, [64, 16], false);
        let replacement = expert_shard(8, [64, 16], false);
        assert_eq!(first.header.file_bytes(), replacement.header.file_bytes());
        let pairs = pairs(&first);
        assert!(matches!(
            pairs.read_local_shard(&replacement.path, &replacement.header, 10_000),
            Err(V41ExpertPayloadError::PairHeaderMismatch { tensor })
                if tensor == "layers.7.ffn.experts.11.w1.weight"
        ));
    }

    #[test]
    fn rejects_cached_ranges_when_same_names_and_length_have_permuted_offsets() {
        let first = expert_shard(7, [64, 16], false);
        let replacement = expert_shard(7, [64, 16], true);
        assert_eq!(first.header.file_bytes(), replacement.header.file_bytes());
        assert_eq!(first.index.tensor_count(), replacement.index.tensor_count());
        let pairs = pairs(&first);
        assert!(matches!(
            pairs.read_local_shard(&replacement.path, &replacement.header, 10_000),
            Err(V41ExpertPayloadError::PairHeaderMismatch { tensor })
                if tensor == "layers.7.ffn.experts.11.w1.weight"
        ));
    }

    #[test]
    fn rejects_truncated_shard_before_payload_read() {
        let shard = expert_shard(7, [64, 16], false);
        let pairs = pairs(&shard);
        OpenOptions::new()
            .write(true)
            .open(&shard.path)
            .expect("open synthetic shard")
            .set_len(shard.header.file_bytes() - 1)
            .expect("truncate synthetic shard");
        assert!(matches!(
            pairs.read_local_shard(&shard.path, &shard.header, 10_000),
            Err(V41ExpertPayloadError::ShardLength { .. })
        ));
    }

    #[test]
    fn rejects_nonfinite_selected_scale_without_returning_payload() {
        let shard = expert_shard(7, [64, 16], false);
        let pairs = pairs(&shard);
        let offset = pairs.w3().scale_range().file_range().start;
        let mut file = OpenOptions::new()
            .write(true)
            .open(&shard.path)
            .expect("open synthetic shard");
        file.seek(SeekFrom::Start(offset)).expect("seek scale");
        file.write_all(&[u8::MAX]).expect("write NaN scale");
        assert!(matches!(
            pairs.read_local_shard(&shard.path, &shard.header, 10_000),
            Err(V41ExpertPayloadError::NonFiniteScale {
                projection: V41ExpertProjection::W3,
                index: 0
            })
        ));
    }

    #[test]
    fn rejects_w2_that_is_not_the_w1_w3_transpose() {
        let shard = expert_shard(7, [32, 32], false);
        assert!(matches!(
            V41ExpertI8ScalePairs::parse(&shard.header, &shard.index, SHARD, 7, 11),
            Err(V41ExpertPayloadError::ProjectionGeometry)
        ));
    }
}
