//! Shared source-FP8 expert metadata and bounded payload reads.
//!
//! This reader accepts only the canonical `layers.<L>.ffn.shared_experts`
//! projections. It validates E4M3FN weights and E8M0 `[N / 32, K / 32]`
//! scales, then rereads the local shard header before reading the six exact
//! ranges. It returns raw storage bytes; conversion and expert execution are
//! separate contracts.

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

const BLOCK: u64 = 32;
/// Largest aggregate shared-expert payload this reader will materialize.
pub const MAX_SHARED_EXPERT_PAYLOAD_BYTES: u64 = 64 * 1024 * 1024;

/// One canonical shared-expert projection.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum V41SharedExpertProjection {
    /// `w1`, the gate projection `[intermediate, hidden]`.
    W1,
    /// `w2`, the down projection `[hidden, intermediate]`.
    W2,
    /// `w3`, the up projection `[intermediate, hidden]`.
    W3,
}

/// Validated metadata for one source-FP8 shared-expert weight and scale pair.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct V41SharedExpertFp8ScalePair {
    layer: u64,
    projection: V41SharedExpertProjection,
    shard: String,
    weight_name: String,
    scale_name: String,
    weight_range: V41TensorRange,
    scale_range: V41TensorRange,
    shape: [u64; 2],
}

impl V41SharedExpertFp8ScalePair {
    /// Parses exactly one canonical shared-expert weight and its `.scale` sibling.
    ///
    /// Layer IDs are syntactic. The caller must bind them to model configuration
    /// and authenticate the checkpoint revision; headers alone do not do either.
    ///
    /// # Errors
    ///
    /// * [`V41SharedExpertFp8ScalePairError::InvalidWeightName`] for any other
    ///   name.
    /// * [`V41SharedExpertFp8ScalePairError::MissingIndexTensor`],
    ///   [`V41SharedExpertFp8ScalePairError::WrongIndexShard`] and
    ///   [`V41SharedExpertFp8ScalePairError::MissingHeaderTensor`] when the
    ///   index or header does not place both tensors in `shard`.
    /// * [`V41SharedExpertFp8ScalePairError::WeightDtype`],
    ///   [`V41SharedExpertFp8ScalePairError::ScaleDtype`],
    ///   [`V41SharedExpertFp8ScalePairError::WeightShape`] and
    ///   [`V41SharedExpertFp8ScalePairError::ScaleShape`] when the pair's
    ///   storage is not the source layout.
    pub fn parse(
        header: &V41SafetensorsHeader,
        index: &V41SafetensorsIndex,
        shard: &str,
        weight_name: &str,
    ) -> Result<Self, V41SharedExpertFp8ScalePairError> {
        let (layer, projection) = parse_weight_name(weight_name)?;
        let base = weight_name
            .strip_suffix(".weight")
            .ok_or_else(|| invalid_name(weight_name))?;
        let scale_name = format!("{base}.scale");
        validate_index(index, shard, weight_name)?;
        validate_index(index, shard, &scale_name)?;
        let weight_range = header.tensor(weight_name).ok_or_else(|| {
            V41SharedExpertFp8ScalePairError::MissingHeaderTensor {
                tensor: weight_name.to_owned(),
            }
        })?;
        let scale_range = header.tensor(&scale_name).ok_or_else(|| {
            V41SharedExpertFp8ScalePairError::MissingHeaderTensor {
                tensor: scale_name.clone(),
            }
        })?;
        let shape = validate_shape(weight_range, scale_range)?;
        Ok(Self {
            layer,
            projection,
            shard: shard.to_owned(),
            weight_name: weight_name.to_owned(),
            scale_name,
            weight_range: weight_range.clone(),
            scale_range: scale_range.clone(),
            shape,
        })
    }
    /// The layer index from the weight name.
    #[must_use]
    pub const fn layer(&self) -> u64 {
        self.layer
    }
    /// Which projection this pair is.
    #[must_use]
    pub const fn projection(&self) -> V41SharedExpertProjection {
        self.projection
    }
    /// The shard file name that holds both tensors.
    #[must_use]
    pub fn shard(&self) -> &str {
        &self.shard
    }
    /// The weight tensor's name.
    #[must_use]
    pub fn weight_name(&self) -> &str {
        &self.weight_name
    }
    /// The scale tensor's name, the weight name with `.scale` for `.weight`.
    #[must_use]
    pub fn scale_name(&self) -> &str {
        &self.scale_name
    }
    /// The weight's byte range in the shard.
    #[must_use]
    pub fn weight_range(&self) -> &V41TensorRange {
        &self.weight_range
    }
    /// The scale's byte range in the shard.
    #[must_use]
    pub fn scale_range(&self) -> &V41TensorRange {
        &self.scale_range
    }
    /// The weight shape `[N, K]`.
    #[must_use]
    pub const fn shape(&self) -> [u64; 2] {
        self.shape
    }
}

/// Three canonical shared-expert source-FP8 projections for one layer.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct V41SharedExpertFp8ScalePairs {
    w1: V41SharedExpertFp8ScalePair,
    w2: V41SharedExpertFp8ScalePair,
    w3: V41SharedExpertFp8ScalePair,
}

impl V41SharedExpertFp8ScalePairs {
    /// Validates all shared-expert projections assigned to one exact shard.
    ///
    /// # Errors
    ///
    /// * [`V41SharedExpertPayloadError::HeaderIndex`] when the header and the
    ///   index disagree on `shard`.
    /// * [`V41SharedExpertPayloadError::Pair`] when a projection's pair is
    ///   invalid, as [`V41SharedExpertFp8ScalePair::parse`] describes.
    /// * [`V41SharedExpertPayloadError::ProjectionGeometry`] when the three
    ///   shapes do not fit together.
    pub fn parse(
        header: &V41SafetensorsHeader,
        index: &V41SafetensorsIndex,
        shard: &str,
        layer: u64,
    ) -> Result<Self, V41SharedExpertPayloadError> {
        header
            .validate_index_shard(index, shard)
            .map_err(V41SharedExpertPayloadError::HeaderIndex)?;
        let prefix = format!("layers.{layer}.ffn.shared_experts");
        let w1 = V41SharedExpertFp8ScalePair::parse(
            header,
            index,
            shard,
            &format!("{prefix}.w1.weight"),
        )?;
        let w2 = V41SharedExpertFp8ScalePair::parse(
            header,
            index,
            shard,
            &format!("{prefix}.w2.weight"),
        )?;
        let w3 = V41SharedExpertFp8ScalePair::parse(
            header,
            index,
            shard,
            &format!("{prefix}.w3.weight"),
        )?;
        validate_geometry(&w1, &w2, &w3)?;
        Ok(Self { w1, w2, w3 })
    }
    /// The `w1` pair.
    #[must_use]
    pub const fn w1(&self) -> &V41SharedExpertFp8ScalePair {
        &self.w1
    }
    /// The `w2` pair.
    #[must_use]
    pub const fn w2(&self) -> &V41SharedExpertFp8ScalePair {
        &self.w2
    }
    /// The `w3` pair.
    #[must_use]
    pub const fn w3(&self) -> &V41SharedExpertFp8ScalePair {
        &self.w3
    }
    /// The model's hidden width, `w1`'s `K`.
    #[must_use]
    pub const fn hidden_width(&self) -> u64 {
        self.w1.shape()[1]
    }
    /// The shared expert's intermediate width, `w1`'s `N`.
    #[must_use]
    pub const fn intermediate_width(&self) -> u64 {
        self.w1.shape()[0]
    }

    /// Reads only the six declared payload intervals after rebinding their header.
    ///
    /// `max_bytes` bounds their aggregate raw size, not process memory. This
    /// revalidates layout and finite storage codes, not a payload digest.
    ///
    /// # Errors
    ///
    /// * [`V41SharedExpertPayloadError::PayloadBudget`] and
    ///   [`V41SharedExpertPayloadError::PayloadLengthOverflow`] past the byte
    ///   limit.
    /// * [`V41SharedExpertPayloadError::NotRegularFile`],
    ///   [`V41SharedExpertPayloadError::ShardLength`],
    ///   [`V41SharedExpertPayloadError::Header`],
    ///   [`V41SharedExpertPayloadError::HeaderMismatch`] and
    ///   [`V41SharedExpertPayloadError::PairHeaderMismatch`] when the file is
    ///   not the shard these pairs were parsed from.
    /// * [`V41SharedExpertPayloadError::Io`] and
    ///   [`V41SharedExpertPayloadError::Allocation`] when reading fails.
    /// * [`V41SharedExpertPayloadError::NonFiniteCode`] and
    ///   [`V41SharedExpertPayloadError::NonFiniteScale`] for a NaN storage code.
    pub fn read_local_shard(
        &self,
        shard: &Path,
        header: &V41SafetensorsHeader,
        max_bytes: u64,
    ) -> Result<V41SharedExpertFp8ScalePayload, V41SharedExpertPayloadError> {
        let total = self.total_payload_bytes()?;
        if total > max_bytes || total > MAX_SHARED_EXPERT_PAYLOAD_BYTES {
            return Err(V41SharedExpertPayloadError::PayloadBudget {
                requested_bytes: total,
                max_bytes: max_bytes.min(MAX_SHARED_EXPERT_PAYLOAD_BYTES),
            });
        }
        let mut file = File::open(shard).map_err(V41SharedExpertPayloadError::Io)?;
        let metadata = file.metadata().map_err(V41SharedExpertPayloadError::Io)?;
        if !metadata.is_file() {
            return Err(V41SharedExpertPayloadError::NotRegularFile);
        }
        if metadata.len() != header.file_bytes() {
            return Err(V41SharedExpertPayloadError::ShardLength {
                actual_bytes: metadata.len(),
                expected_bytes: header.file_bytes(),
            });
        }
        let actual = read_and_parse_header(&mut file, metadata.len())?;
        if actual != *header {
            return Err(V41SharedExpertPayloadError::HeaderMismatch);
        }
        self.bind_header(&actual)?;
        Ok(V41SharedExpertFp8ScalePayload {
            w1: read_pair(&mut file, &self.w1)?,
            w2: read_pair(&mut file, &self.w2)?,
            w3: read_pair(&mut file, &self.w3)?,
        })
    }
    fn total_payload_bytes(&self) -> Result<u64, V41SharedExpertPayloadError> {
        [&self.w1, &self.w2, &self.w3]
            .into_iter()
            .try_fold(0_u64, |total, pair| {
                total
                    .checked_add(pair.weight_range().byte_length())
                    .and_then(|v| v.checked_add(pair.scale_range().byte_length()))
                    .ok_or(V41SharedExpertPayloadError::PayloadLengthOverflow)
            })
    }
    fn bind_header(
        &self,
        header: &V41SafetensorsHeader,
    ) -> Result<(), V41SharedExpertPayloadError> {
        for pair in [&self.w1, &self.w2, &self.w3] {
            for (name, cached) in [
                (pair.weight_name(), pair.weight_range()),
                (pair.scale_name(), pair.scale_range()),
            ] {
                if header.tensor(name) != Some(cached) {
                    return Err(V41SharedExpertPayloadError::PairHeaderMismatch {
                        tensor: name.to_owned(),
                    });
                }
            }
        }
        Ok(())
    }
}

/// Raw source bytes for one FP8 shared-expert projection.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct V41SharedExpertProjectionPayload {
    codes: Vec<u8>,
    scales: Vec<u8>,
}
impl V41SharedExpertProjectionPayload {
    /// The E4M3FN weight bytes.
    #[must_use]
    pub fn codes(&self) -> &[u8] {
        &self.codes
    }
    /// The E8M0 scale bytes.
    #[must_use]
    pub fn scales(&self) -> &[u8] {
        &self.scales
    }
}
/// Raw bounded bytes for all three source-FP8 shared-expert projections.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct V41SharedExpertFp8ScalePayload {
    w1: V41SharedExpertProjectionPayload,
    w2: V41SharedExpertProjectionPayload,
    w3: V41SharedExpertProjectionPayload,
}
impl V41SharedExpertFp8ScalePayload {
    /// The `w1` bytes.
    #[must_use]
    pub const fn w1(&self) -> &V41SharedExpertProjectionPayload {
        &self.w1
    }
    /// The `w2` bytes.
    #[must_use]
    pub const fn w2(&self) -> &V41SharedExpertProjectionPayload {
        &self.w2
    }
    /// The `w3` bytes.
    #[must_use]
    pub const fn w3(&self) -> &V41SharedExpertProjectionPayload {
        &self.w3
    }
}

/// A bounded shared-expert payload did not meet the source storage contract.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum V41SharedExpertPayloadError {
    /// The header and the safetensors index disagree on the shard.
    #[error("shared expert header/index identity failed: {0}")]
    HeaderIndex(V41SafetensorsHeaderError),
    /// One projection's weight and scale pair is invalid.
    #[error("shared expert pair failed: {0}")]
    Pair(#[from] V41SharedExpertFp8ScalePairError),
    /// `w1` and `w3` differ in shape, or `w2` is not their transpose.
    #[error("shared expert projections do not have w1/w3 equal and w2 transposed geometry")]
    ProjectionGeometry,
    /// The six ranges together pass the byte limit.
    #[error(
        "shared expert payload needs {requested_bytes} bytes, above the {max_bytes}-byte limit"
    )]
    PayloadBudget {
        /// Bytes the six ranges need.
        requested_bytes: u64,
        /// The smaller of the caller's limit and the fixed cap.
        max_bytes: u64,
    },
    /// The total payload length does not fit in `u64`.
    #[error("shared expert payload length overflowed")]
    PayloadLengthOverflow,
    /// The shard path is not a regular file.
    #[error("shared expert shard is not a regular file")]
    NotRegularFile,
    /// The shard's size differs from the header's.
    #[error("shared expert shard length is {actual_bytes}, expected {expected_bytes}")]
    ShardLength {
        /// The file's size.
        actual_bytes: u64,
        /// The size the header declares.
        expected_bytes: u64,
    },
    /// The shard's header differs from the one supplied.
    #[error("shared expert shard header differs from supplied header")]
    HeaderMismatch,
    /// A cached tensor range differs from the reread header.
    #[error("shared expert cached range differs from revalidated header tensor {tensor}")]
    PairHeaderMismatch {
        /// The tensor whose range changed.
        tensor: String,
    },
    /// The shard's header does not parse.
    #[error("could not parse shared expert shard header: {0}")]
    Header(#[from] V41SafetensorsHeaderError),
    /// Reading the shard failed.
    #[error("could not read shared expert payload: {0}")]
    Io(#[source] std::io::Error),
    /// A payload buffer could not be reserved.
    #[error("could not allocate shared expert payload range")]
    Allocation,
    /// A weight byte is an E4M3FN NaN code.
    #[error("shared expert {projection:?} E4M3 code {index} is nonfinite")]
    NonFiniteCode {
        /// The projection.
        projection: V41SharedExpertProjection,
        /// Index into its weight bytes.
        index: usize,
    },
    /// A scale byte is the E8M0 NaN code.
    #[error("shared expert {projection:?} E8M0 scale {index} is nonfinite")]
    NonFiniteScale {
        /// The projection.
        projection: V41SharedExpertProjection,
        /// Index into its scale bytes.
        index: usize,
    },
}

/// A shared-expert weight and scale pair that does not match the source layout.
#[derive(Clone, Debug, Error, Eq, PartialEq)]
#[non_exhaustive]
pub enum V41SharedExpertFp8ScalePairError {
    /// The name is not `layers.<L>.ffn.shared_experts.w{1,2,3}.weight`.
    #[error("unsupported canonical shared expert weight name {weight_name}")]
    InvalidWeightName {
        /// The rejected name.
        weight_name: String,
    },
    /// The safetensors index does not list the tensor.
    #[error("selected tensor {tensor} is absent from the safetensors index")]
    MissingIndexTensor {
        /// The tensor name.
        tensor: String,
    },
    /// The index assigns the tensor to another shard.
    #[error("selected tensor {tensor} is assigned to {indexed_shard}, not {expected_shard}")]
    WrongIndexShard {
        /// The tensor name.
        tensor: String,
        /// The shard being parsed.
        expected_shard: String,
        /// The shard the index names.
        indexed_shard: String,
    },
    /// The shard header does not list the tensor.
    #[error("selected tensor {tensor} is absent from the safetensors header")]
    MissingHeaderTensor {
        /// The tensor name.
        tensor: String,
    },
    /// The weight is not stored as E4M3FN.
    #[error("shared expert weight dtype is {actual:?}, expected F8E4M3Fn")]
    WeightDtype {
        /// The stored dtype.
        actual: V41StorageDtype,
    },
    /// The scale is not stored as E8M0.
    #[error("shared expert scale dtype is {actual:?}, expected F8E8M0Fnu")]
    ScaleDtype {
        /// The stored dtype.
        actual: V41StorageDtype,
    },
    /// The weight is not a nonzero `[N, K]` with both multiples of 32.
    #[error(
        "shared expert weight shape must be nonzero rank-two [N, K] with dimensions divisible by 32"
    )]
    WeightShape,
    /// The scale is not `[N / 32, K / 32]`.
    #[error("shared expert scale shape must be exactly [N / 32, K / 32]")]
    ScaleShape,
}

fn parse_weight_name(
    name: &str,
) -> Result<(u64, V41SharedExpertProjection), V41SharedExpertFp8ScalePairError> {
    let mut parts = name.split('.');
    let result = match (
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
            Some("shared_experts"),
            Some(projection),
            Some("weight"),
            None,
        ) => {
            let projection = match projection {
                "w1" => V41SharedExpertProjection::W1,
                "w2" => V41SharedExpertProjection::W2,
                "w3" => V41SharedExpertProjection::W3,
                _ => return Err(invalid_name(name)),
            };
            let valid = !(layer.is_empty() || layer.len() > 1 && layer.starts_with('0'))
                && layer.bytes().all(|b| b.is_ascii_digit());
            if !valid {
                return Err(invalid_name(name));
            }
            (layer.parse().map_err(|_| invalid_name(name))?, projection)
        }
        _ => return Err(invalid_name(name)),
    };
    Ok(result)
}
fn invalid_name(name: &str) -> V41SharedExpertFp8ScalePairError {
    V41SharedExpertFp8ScalePairError::InvalidWeightName {
        weight_name: name.to_owned(),
    }
}
fn validate_index(
    index: &V41SafetensorsIndex,
    shard: &str,
    tensor: &str,
) -> Result<(), V41SharedExpertFp8ScalePairError> {
    let indexed_shard = index.shard_for_tensor(tensor).ok_or_else(|| {
        V41SharedExpertFp8ScalePairError::MissingIndexTensor {
            tensor: tensor.to_owned(),
        }
    })?;
    if indexed_shard != shard {
        return Err(V41SharedExpertFp8ScalePairError::WrongIndexShard {
            tensor: tensor.to_owned(),
            expected_shard: shard.to_owned(),
            indexed_shard: indexed_shard.to_owned(),
        });
    }
    Ok(())
}
fn validate_shape(
    weight: &V41TensorRange,
    scale: &V41TensorRange,
) -> Result<[u64; 2], V41SharedExpertFp8ScalePairError> {
    if weight.dtype() != V41StorageDtype::F8E4M3Fn {
        return Err(V41SharedExpertFp8ScalePairError::WeightDtype {
            actual: weight.dtype(),
        });
    }
    let &[rows, columns] = weight.shape() else {
        return Err(V41SharedExpertFp8ScalePairError::WeightShape);
    };
    if rows == 0 || columns == 0 || !rows.is_multiple_of(BLOCK) || !columns.is_multiple_of(BLOCK) {
        return Err(V41SharedExpertFp8ScalePairError::WeightShape);
    }
    if scale.dtype() != V41StorageDtype::F8E8M0Fnu {
        return Err(V41SharedExpertFp8ScalePairError::ScaleDtype {
            actual: scale.dtype(),
        });
    }
    if scale.shape() != [rows / BLOCK, columns / BLOCK] {
        return Err(V41SharedExpertFp8ScalePairError::ScaleShape);
    }
    Ok([rows, columns])
}
fn validate_geometry(
    w1: &V41SharedExpertFp8ScalePair,
    w2: &V41SharedExpertFp8ScalePair,
    w3: &V41SharedExpertFp8ScalePair,
) -> Result<(), V41SharedExpertPayloadError> {
    let [intermediate, hidden] = w1.shape();
    if w1.projection() != V41SharedExpertProjection::W1
        || w2.projection() != V41SharedExpertProjection::W2
        || w3.projection() != V41SharedExpertProjection::W3
        || w2.layer() != w1.layer()
        || w3.layer() != w1.layer()
        || w2.shard() != w1.shard()
        || w3.shard() != w1.shard()
        || w3.shape() != [intermediate, hidden]
        || w2.shape() != [hidden, intermediate]
    {
        return Err(V41SharedExpertPayloadError::ProjectionGeometry);
    }
    Ok(())
}
fn read_and_parse_header(
    file: &mut File,
    file_bytes: u64,
) -> Result<V41SafetensorsHeader, V41SharedExpertPayloadError> {
    file.seek(SeekFrom::Start(0))
        .map_err(V41SharedExpertPayloadError::Io)?;
    let mut prefix = [0; 8];
    file.read_exact(&mut prefix)
        .map_err(V41SharedExpertPayloadError::Io)?;
    let header_bytes = u64::from_le_bytes(prefix);
    if header_bytes > MAX_HEADER_BYTES {
        return Err(V41SafetensorsHeaderError::HeaderTooLarge { header_bytes }.into());
    }
    let length = usize::try_from(header_bytes)
        .map_err(|_| V41SafetensorsHeaderError::HeaderTooLarge { header_bytes })?;
    let mut bytes = Vec::with_capacity(8 + length);
    bytes.extend_from_slice(&prefix);
    bytes.resize(8 + length, 0);
    file.read_exact(&mut bytes[8..])
        .map_err(V41SharedExpertPayloadError::Io)?;
    V41SafetensorsHeader::parse_prefixed_header(&bytes, file_bytes)
        .map_err(V41SharedExpertPayloadError::Header)
}
fn read_pair(
    file: &mut File,
    pair: &V41SharedExpertFp8ScalePair,
) -> Result<V41SharedExpertProjectionPayload, V41SharedExpertPayloadError> {
    let codes = read_range(file, pair.weight_range())?;
    let scales = read_range(file, pair.scale_range())?;
    if let Some((index, _)) = codes
        .iter()
        .enumerate()
        .find(|(_, code)| (**code & 0x7f) == 0x7f)
    {
        return Err(V41SharedExpertPayloadError::NonFiniteCode {
            projection: pair.projection(),
            index,
        });
    }
    if let Some((index, _)) = scales
        .iter()
        .enumerate()
        .find(|(_, code)| **code == u8::MAX)
    {
        return Err(V41SharedExpertPayloadError::NonFiniteScale {
            projection: pair.projection(),
            index,
        });
    }
    Ok(V41SharedExpertProjectionPayload { codes, scales })
}
fn read_range(
    file: &mut File,
    range: &V41TensorRange,
) -> Result<Vec<u8>, V41SharedExpertPayloadError> {
    let length = usize::try_from(range.byte_length())
        .map_err(|_| V41SharedExpertPayloadError::PayloadLengthOverflow)?;
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(length)
        .map_err(|_| V41SharedExpertPayloadError::Allocation)?;
    bytes.resize(length, 0);
    file.seek(SeekFrom::Start(range.file_range().start))
        .and_then(|_| file.read_exact(&mut bytes))
        .map_err(V41SharedExpertPayloadError::Io)?;
    Ok(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{checkpoint::V41SafetensorsHeader, manifest::V41SafetensorsIndex};
    use std::{
        fs::{self, File},
        io::Write,
        path::PathBuf,
        sync::atomic::{AtomicU64, Ordering},
    };
    const SHARD: &str = "model-00003-of-00048.safetensors";
    static ID: AtomicU64 = AtomicU64::new(0);
    struct TestShard {
        path: PathBuf,
        header: V41SafetensorsHeader,
        index: V41SafetensorsIndex,
    }
    impl Drop for TestShard {
        fn drop(&mut self) {
            let _ = fs::remove_file(&self.path);
        }
    }
    fn names() -> [String; 6] {
        let p = "layers.0.ffn.shared_experts";
        [
            format!("{p}.w1.weight"),
            format!("{p}.w1.scale"),
            format!("{p}.w2.weight"),
            format!("{p}.w2.scale"),
            format!("{p}.w3.weight"),
            format!("{p}.w3.scale"),
        ]
    }
    fn make_shard(permuted: bool) -> TestShard {
        let names = names();
        let mut entries = Vec::new();
        let mut payload = Vec::new();
        let shapes = [[32, 64], [1, 2], [64, 32], [2, 1], [32, 64], [1, 2]];
        for (i, (name, shape)) in names.iter().zip(shapes).enumerate() {
            let offset = if permuted {
                match i {
                    0 => 4100,
                    4 => 0,
                    _ => payload.len() as u64,
                }
            } else {
                payload.len() as u64
            };
            let len = shape[0] * shape[1];
            entries.push(format!(
                r#""{name}":{{"dtype":"{}","shape":[{},{}],"data_offsets":[{offset},{}]}}"#,
                if i % 2 == 0 { "F8_E4M3" } else { "F8_E8M0FNU" },
                shape[0],
                shape[1],
                offset + len
            ));
            let marker = u8::try_from(i).expect("six fixture tensors");
            payload.extend(std::iter::repeat_n(
                if i % 2 == 0 {
                    0x30 + marker
                } else {
                    120 + marker
                },
                usize::try_from(len).expect("small fixture payload"),
            ));
        }
        let json = format!("{{{}}}", entries.join(","));
        let bytes = 8 + json.len() as u64 + payload.len() as u64;
        let header = V41SafetensorsHeader::parse_prefixed_header(
            &[&(json.len() as u64).to_le_bytes(), json.as_bytes()].concat(),
            bytes,
        )
        .expect("header");
        let index_json = format!(
            r#"{{"metadata":{{"total_size":1}},"weight_map":{{{}}}}}"#,
            names
                .iter()
                .map(|name| format!(r#""{name}":"{SHARD}""#))
                .collect::<Vec<_>>()
                .join(",")
        );
        let index = V41SafetensorsIndex::parse(&index_json).expect("index");
        let path = std::env::temp_dir().join(format!(
            "metallix-shared-fp8-{}-{}",
            std::process::id(),
            ID.fetch_add(1, Ordering::Relaxed)
        ));
        let mut file = File::options()
            .write(true)
            .create_new(true)
            .open(&path)
            .expect("create");
        file.write_all(&(json.len() as u64).to_le_bytes())
            .expect("prefix");
        file.write_all(json.as_bytes()).expect("header");
        file.write_all(&payload).expect("payload");
        TestShard {
            path,
            header,
            index,
        }
    }
    fn pairs(s: &TestShard) -> V41SharedExpertFp8ScalePairs {
        V41SharedExpertFp8ScalePairs::parse(&s.header, &s.index, SHARD, 0).expect("pairs")
    }
    #[test]
    fn parses_geometry_and_reads_six_exact_ranges() {
        let s = make_shard(false);
        let p = pairs(&s);
        assert_eq!((p.hidden_width(), p.intermediate_width()), (64, 32));
        let data = p
            .read_local_shard(&s.path, &s.header, 10_000)
            .expect("read");
        assert_eq!(data.w1().codes(), vec![0x30; 2048]);
        assert_eq!(data.w1().scales(), &[121, 121]);
        assert_eq!(data.w2().codes(), vec![0x32; 2048]);
        assert_eq!(data.w2().scales(), &[123, 123]);
        assert_eq!(data.w3().codes(), vec![0x34; 2048]);
        assert_eq!(data.w3().scales(), &[125, 125]);
    }
    #[test]
    fn rejects_budget_before_read() {
        let s = make_shard(false);
        assert!(matches!(
            pairs(&s).read_local_shard(&s.path.with_extension("absent"), &s.header, 1),
            Err(V41SharedExpertPayloadError::PayloadBudget { .. })
        ));
    }
    #[test]
    fn rejects_stale_or_rebound_ranges() {
        let s = make_shard(false);
        let p = pairs(&s);
        let stale = make_shard(true);
        assert!(matches!(
            p.read_local_shard(&s.path, &stale.header, 10_000),
            Err(V41SharedExpertPayloadError::HeaderMismatch)
        ));
        let replacement = make_shard(true);
        assert!(matches!(
            p.read_local_shard(&replacement.path, &replacement.header, 10_000),
            Err(V41SharedExpertPayloadError::PairHeaderMismatch { .. })
        ));
    }
    #[test]
    fn rejects_nonfinite_codes_and_scales() {
        let s = make_shard(false);
        let mut bytes = fs::read(&s.path).expect("read");
        let code = usize::try_from(
            s.header
                .tensor("layers.0.ffn.shared_experts.w2.weight")
                .unwrap()
                .file_range()
                .start,
        )
        .expect("fixture offset");
        bytes[code] = 0x7f;
        fs::write(&s.path, bytes).expect("write");
        assert!(matches!(
            pairs(&s).read_local_shard(&s.path, &s.header, 10_000),
            Err(V41SharedExpertPayloadError::NonFiniteCode {
                projection: V41SharedExpertProjection::W2,
                ..
            })
        ));

        let s = make_shard(false);
        let mut bytes = fs::read(&s.path).expect("read");
        let scale = usize::try_from(
            s.header
                .tensor("layers.0.ffn.shared_experts.w3.scale")
                .unwrap()
                .file_range()
                .start,
        )
        .expect("fixture offset");
        bytes[scale] = u8::MAX;
        fs::write(&s.path, bytes).expect("write");
        assert!(matches!(
            pairs(&s).read_local_shard(&s.path, &s.header, 10_000),
            Err(V41SharedExpertPayloadError::NonFiniteScale {
                projection: V41SharedExpertProjection::W3,
                ..
            })
        ));
    }
    #[test]
    fn rejects_wrong_block_geometry() {
        let s = make_shard(false);
        let name = "layers.0.ffn.shared_experts.w1.weight";
        let json = format!(
            r#"{{"{name}":{{"dtype":"F8_E4M3","shape":[31,64],"data_offsets":[0,1984]}},"layers.0.ffn.shared_experts.w1.scale":{{"dtype":"F8_E8M0FNU","shape":[1,2],"data_offsets":[1984,1986]}}}}"#
        );
        let h = V41SafetensorsHeader::parse_prefixed_header(
            &[&(json.len() as u64).to_le_bytes(), json.as_bytes()].concat(),
            8 + json.len() as u64 + 1986,
        )
        .expect("header");
        assert!(matches!(
            V41SharedExpertFp8ScalePair::parse(&h, &s.index, SHARD, name),
            Err(V41SharedExpertFp8ScalePairError::WeightShape)
        ));
    }

    #[test]
    fn rejects_noncanonical_names_before_tensor_lookup() {
        let s = make_shard(false);
        for name in [
            "layers.00.ffn.shared_experts.w1.weight",
            "layers.-1.ffn.shared_experts.w1.weight",
            "layers.0.ffn.experts.0.w1.weight",
            "layers.0.ffn.shared_experts.w4.weight",
            "layers.0.ffn.shared_experts.w1.weight.extra",
        ] {
            assert!(matches!(
                V41SharedExpertFp8ScalePair::parse(&s.header, &s.index, SHARD, name),
                Err(V41SharedExpertFp8ScalePairError::InvalidWeightName { .. })
            ));
        }
    }

    #[test]
    fn rejects_storage_dtype_and_scale_geometry_mismatches() {
        let s = make_shard(false);
        let name = "layers.0.ffn.shared_experts.w1.weight";
        for (weight_dtype, scale_dtype, scale_shape, expected) in [
            ("U8", "F8_E8M0FNU", "1,2", "weight"),
            ("F8_E4M3", "U8", "1,2", "scale"),
            ("F8_E4M3", "F8_E8M0FNU", "2,1", "shape"),
        ] {
            let json = format!(
                r#"{{"{name}":{{"dtype":"{weight_dtype}","shape":[32,64],"data_offsets":[0,2048]}},"layers.0.ffn.shared_experts.w1.scale":{{"dtype":"{scale_dtype}","shape":[{scale_shape}],"data_offsets":[2048,2050]}}}}"#
            );
            let header = V41SafetensorsHeader::parse_prefixed_header(
                &[&(json.len() as u64).to_le_bytes(), json.as_bytes()].concat(),
                8 + json.len() as u64 + 2050,
            )
            .expect("valid byte layout");
            let error = V41SharedExpertFp8ScalePair::parse(&header, &s.index, SHARD, name)
                .expect_err("incompatible storage contract");
            match expected {
                "weight" => assert!(matches!(
                    error,
                    V41SharedExpertFp8ScalePairError::WeightDtype { .. }
                )),
                "scale" => assert!(matches!(
                    error,
                    V41SharedExpertFp8ScalePairError::ScaleDtype { .. }
                )),
                "shape" => assert!(matches!(
                    error,
                    V41SharedExpertFp8ScalePairError::ScaleShape
                )),
                _ => unreachable!("fixed cases"),
            }
        }
    }
}
