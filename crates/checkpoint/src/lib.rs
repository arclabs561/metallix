//! Checkpoint containers: the tensor table and metadata of a weights file,
//! read without loading tensor payloads.
//!
//! A [`TensorSource`] answers, for a tensor name, its stored encoding, shape
//! and byte range, and reads one tensor's bytes on request. Model adapters
//! map names and decode encodings; this crate knows neither models nor MLX.
//! GGUF is the first container ([`gguf::GgufFile`]). Encodings are
//! container-neutral ([`TensorEncoding`]), so one table can describe GGUF
//! blocks, plain safetensors dtypes and MLX affine codes.

#![deny(missing_docs)]
#![warn(clippy::missing_errors_doc)]

pub mod expert;
pub mod gguf;
mod nocache;

use std::{
    fs::File,
    ops::Range,
    path::{Path, PathBuf},
};

pub use blockfloat::gguf::GgufEncoding;
use thiserror::Error;

/// An element type stored one value per element, as safetensors names them.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DType {
    /// 64-bit IEEE float.
    F64,
    /// 32-bit IEEE float.
    F32,
    /// 16-bit IEEE float.
    F16,
    /// bfloat16.
    BF16,
    /// 64-bit signed integer.
    I64,
    /// 32-bit signed integer.
    I32,
    /// 16-bit signed integer.
    I16,
    /// 8-bit signed integer.
    I8,
    /// 64-bit unsigned integer.
    U64,
    /// 32-bit unsigned integer.
    U32,
    /// 16-bit unsigned integer.
    U16,
    /// 8-bit unsigned integer.
    U8,
    /// One byte per boolean.
    Bool,
}

impl DType {
    /// Bytes per element.
    #[must_use]
    pub const fn size(self) -> u64 {
        match self {
            Self::F64 | Self::I64 | Self::U64 => 8,
            Self::F32 | Self::I32 | Self::U32 => 4,
            Self::F16 | Self::BF16 | Self::I16 | Self::U16 => 2,
            Self::I8 | Self::U8 | Self::Bool => 1,
        }
    }
}

/// How a tensor's elements are stored.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TensorEncoding {
    /// One value per element.
    Plain(DType),
    /// A GGUF type, possibly block-quantized.
    Gguf(GgufEncoding),
    /// MLX affine codes: `bits`-wide codes packed in `u32` words, with one
    /// scale and one bias per `group_size` columns stored as the sibling
    /// tensors `<stem>.scales` and `<stem>.biases`.
    Affine {
        /// Bits per code.
        bits: u8,
        /// Columns sharing one scale and bias.
        group_size: u16,
    },
}

impl TensorEncoding {
    /// The GGUF type, if this is one.
    #[must_use]
    pub const fn gguf(self) -> Option<GgufEncoding> {
        match self {
            Self::Gguf(encoding) => Some(encoding),
            Self::Plain(_) | Self::Affine { .. } => None,
        }
    }
}

impl From<GgufEncoding> for TensorEncoding {
    fn from(encoding: GgufEncoding) -> Self {
        Self::Gguf(encoding)
    }
}

/// One stored tensor.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TensorInfo {
    encoding: TensorEncoding,
    shape: Vec<u64>,
    range: Range<u64>,
}

impl TensorInfo {
    /// A record as a container describes it; `shape` is outermost first.
    #[must_use]
    pub const fn new(encoding: TensorEncoding, shape: Vec<u64>, range: Range<u64>) -> Self {
        Self {
            encoding,
            shape,
            range,
        }
    }

    /// The stored encoding.
    #[must_use]
    pub const fn encoding(&self) -> TensorEncoding {
        self.encoding
    }

    /// Dimensions in row-major order, outermost first (`[rows, columns]` for
    /// a matrix), whatever order the container writes them in.
    #[must_use]
    pub fn shape(&self) -> &[u64] {
        &self.shape
    }

    /// Element count.
    #[must_use]
    pub fn elements(&self) -> u64 {
        self.shape.iter().product()
    }

    /// Absolute byte range of the payload in the file.
    #[must_use]
    pub fn range(&self) -> Range<u64> {
        self.range.clone()
    }

    /// Absolute byte range of entry `index` along the outermost dimension
    /// (one expert of a `[experts, rows, columns]` stack), or `None` when
    /// `index` is out of bounds or the payload does not split into equal
    /// whole-byte entries.
    #[must_use]
    pub fn slice_dim0(&self, index: u64) -> Option<Range<u64>> {
        let &outer = self.shape.first()?;
        let bytes = self.range.end - self.range.start;
        if index >= outer || !bytes.is_multiple_of(outer) {
            return None;
        }
        let entry = bytes / outer;
        let start = self.range.start + index * entry;
        Some(start..start + entry)
    }
}

/// A container whose tensors can be listed and read one at a time.
pub trait TensorSource {
    /// Tensor names in file order.
    fn names(&self) -> Vec<&str>;

    /// The tensor called `name`, if stored.
    fn tensor(&self, name: &str) -> Option<&TensorInfo>;

    /// Reads one tensor's payload. `max_bytes` is checked before the file is
    /// opened or memory allocated.
    ///
    /// # Errors
    /// Returns an error for a missing tensor, exceeded byte budget, changed file, or failed read.
    fn read(&self, name: &str, max_bytes: u64) -> Result<Vec<u8>, CheckpointError>;

    /// Reads the absolute byte `range`, which must lie inside `name`'s
    /// payload, into `dst`, which must be exactly that long: the caller's
    /// buffer is the byte budget, checked before any read. Sources holding
    /// their file open answer with one positional read and no allocation; the
    /// default reads the whole tensor and copies the range.
    ///
    /// # Errors
    /// Returns an error for a missing tensor, a range outside its payload or
    /// unequal to `dst`, or a failed read.
    fn read_range(
        &self,
        name: &str,
        range: Range<u64>,
        dst: &mut [u8],
    ) -> Result<(), CheckpointError> {
        let info = self
            .tensor(name)
            .ok_or_else(|| CheckpointError::MissingTensor(name.to_owned()))?;
        let whole = info.range();
        check_range(name, info, &range, dst.len())?;
        let bytes = self.read(name, whole.end - whole.start)?;
        let offset = usize::try_from(range.start - whole.start).map_err(|_| {
            CheckpointError::OutsideTensor {
                tensor: name.to_owned(),
                range: range.clone(),
                len: dst.len(),
            }
        })?;
        dst.copy_from_slice(&bytes[offset..offset + dst.len()]);
        Ok(())
    }
}

/// `range` lies inside `info`'s payload and is exactly `len` bytes long.
pub(crate) fn check_range(
    name: &str,
    info: &TensorInfo,
    range: &Range<u64>,
    len: usize,
) -> Result<(), CheckpointError> {
    let whole = info.range();
    let inside = whole.start <= range.start && range.start <= range.end && range.end <= whole.end;
    if !inside || u64::try_from(len).ok() != Some(range.end - range.start) {
        return Err(CheckpointError::OutsideTensor {
            tensor: name.to_owned(),
            range: range.clone(),
            len,
        });
    }
    Ok(())
}

/// A container that cannot be read or does not describe valid tensors.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum CheckpointError {
    /// A file read failed.
    #[error("could not read {path}: {source}")]
    Io {
        /// The file whose read failed.
        path: PathBuf,
        /// The underlying I/O error.
        source: std::io::Error,
    },
    /// The file does not have the GGUF magic header.
    #[error("{0} is not a GGUF file")]
    NotGguf(PathBuf),
    /// The container version is unsupported.
    #[error("GGUF version {0} is not supported (expected 3)")]
    Version(u32),
    /// The header is truncated.
    #[error("GGUF header ends early in {0}")]
    Truncated(PathBuf),
    /// A count or length exceeds its allocation bound.
    #[error("GGUF {what} of {value} exceeds the limit of {limit}")]
    Limit {
        /// The bounded field.
        what: &'static str,
        /// The observed count or length.
        value: u64,
        /// The maximum allowed count or length.
        limit: u64,
    },
    /// A required UTF-8 string is malformed.
    #[error("GGUF {0} is not valid UTF-8")]
    Utf8(String),
    /// A metadata value uses an unknown type identifier.
    #[error("GGUF metadata {key:?} has unknown value type {id}")]
    ValueType {
        /// Metadata key.
        key: String,
        /// Unsupported value type.
        id: u32,
    },
    /// Nested metadata arrays are unsupported.
    #[error("GGUF metadata {0:?} is a nested array")]
    NestedArray(String),
    /// A metadata key is repeated.
    #[error("GGUF metadata key {0:?} appears twice")]
    DuplicateKey(String),
    /// A tensor name is repeated.
    #[error("GGUF tensor {0:?} appears twice")]
    DuplicateTensor(String),
    /// A tensor uses an unknown encoding identifier.
    #[error("GGUF tensor {tensor:?} has unknown type id {id}")]
    Encoding {
        /// Stored tensor name.
        tensor: String,
        /// Unsupported encoding identifier.
        id: u32,
    },
    /// Tensor dimensions are invalid for the declared encoding.
    #[error("GGUF tensor {tensor:?} has an invalid shape {shape:?} for {encoding:?}")]
    Shape {
        /// The stored tensor name.
        tensor: String,
        /// The invalid stored dimensions.
        shape: Vec<u64>,
        /// The declared tensor encoding.
        encoding: GgufEncoding,
    },
    /// The declared alignment is not a nonzero power of two.
    #[error("GGUF general.alignment {0} is not a nonzero power of two")]
    Alignment(u64),
    /// A tensor range is misaligned or outside the file.
    #[error("GGUF tensor {0:?} is misaligned or extends past the end of the file")]
    Range(String),
    /// Two tensor payload ranges overlap.
    #[error("GGUF tensors {0:?} and {1:?} overlap")]
    Overlap(String, String),
    /// The file changed after header inspection.
    #[error("{0} changed after it was inspected")]
    Changed(PathBuf),
    /// A tensor exceeds the caller-provided read budget.
    #[error("tensor {tensor:?} is {bytes} bytes, more than the {max_bytes} allowed")]
    TooLarge {
        /// The stored tensor name.
        tensor: String,
        /// The requested payload size.
        bytes: u64,
        /// The caller-provided byte budget.
        max_bytes: u64,
    },
    /// The requested tensor is absent.
    #[error("checkpoint has no tensor {0:?}")]
    MissingTensor(String),
    /// Required metadata is absent or has the wrong type.
    #[error("GGUF metadata {key:?} is missing or not {expected}")]
    Metadata {
        /// Missing or mistyped key.
        key: String,
        /// Required metadata type.
        expected: &'static str,
    },
    /// A range read lies outside its tensor's payload, or its buffer is not
    /// the range's length.
    #[error("range {range:?} into a {len}-byte buffer does not fit tensor {tensor:?}")]
    OutsideTensor {
        /// The stored tensor name.
        tensor: String,
        /// The requested absolute range.
        range: Range<u64>,
        /// The destination buffer length.
        len: usize,
    },
}

/// Whether reads go through the operating system's file cache.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum ReadCache {
    /// Cached reads (the default).
    #[default]
    Use,
    /// Ask the OS not to keep read pages (`F_NOCACHE` on macOS; ignored
    /// elsewhere), so a byte budget measured above the source is the real
    /// one and not hidden by the page cache.
    Bypass,
}

impl ReadCache {
    /// Applies this setting to an open `file` (named `path` in errors).
    ///
    /// # Errors
    /// The operating system refuses the setting.
    pub fn apply(self, file: &File, path: &Path) -> Result<(), CheckpointError> {
        match self {
            Self::Use => Ok(()),
            Self::Bypass => nocache::set(file).map_err(|source| CheckpointError::Io {
                path: path.to_owned(),
                source,
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::{CheckpointError, DType, TensorEncoding, TensorInfo, TensorSource};

    /// Tensors held in memory at made-up offsets, to exercise the provided
    /// `read_range`.
    struct Memory(HashMap<String, (TensorInfo, Vec<u8>)>);

    impl TensorSource for Memory {
        fn names(&self) -> Vec<&str> {
            self.0.keys().map(String::as_str).collect()
        }

        fn tensor(&self, name: &str) -> Option<&TensorInfo> {
            self.0.get(name).map(|(info, _)| info)
        }

        fn read(&self, name: &str, max_bytes: u64) -> Result<Vec<u8>, CheckpointError> {
            let (_, bytes) = self
                .0
                .get(name)
                .ok_or_else(|| CheckpointError::MissingTensor(name.to_owned()))?;
            assert!(bytes.len() as u64 <= max_bytes);
            Ok(bytes.clone())
        }
    }

    #[test]
    fn the_provided_range_read_copies_out_of_a_whole_read() {
        let info = TensorInfo::new(TensorEncoding::Plain(DType::U8), vec![4, 2], 100..108);
        let source = Memory(HashMap::from([(
            "t".to_owned(),
            (info.clone(), (0..8).collect()),
        )]));
        let mut dst = [0; 2];
        source
            .read_range("t", info.slice_dim0(2).expect("entry 2"), &mut dst)
            .expect("read");
        assert_eq!(dst, [4, 5]);
        assert!(matches!(
            source.read_range("t", 99..101, &mut dst),
            Err(CheckpointError::OutsideTensor { .. })
        ));
    }

    #[test]
    fn outer_slices_need_equal_whole_byte_entries() {
        let info = TensorInfo::new(TensorEncoding::Plain(DType::U8), vec![3, 1], 10..14);
        assert_eq!(info.slice_dim0(0), None, "4 bytes do not split 3 ways");
        let scalar = TensorInfo::new(TensorEncoding::Plain(DType::F32), vec![], 0..4);
        assert_eq!(scalar.slice_dim0(0), None);
    }

    #[test]
    fn bypassing_the_file_cache_still_reads_the_bytes() {
        use std::{fs, os::unix::fs::FileExt};

        use super::ReadCache;

        let path = std::env::temp_dir().join(format!("checkpoint-nocache-{}", std::process::id()));
        fs::write(&path, [1, 2, 3, 4]).expect("write");
        let file = fs::File::open(&path).expect("open");
        ReadCache::Bypass.apply(&file, &path).expect("F_NOCACHE");
        ReadCache::Use.apply(&file, &path).expect("no-op");
        let mut bytes = [0; 2];
        file.read_exact_at(&mut bytes, 1).expect("read");
        assert_eq!(bytes, [2, 3]);
        fs::remove_file(&path).expect("cleanup");
    }
}
