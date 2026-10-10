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

pub mod gguf;

use std::{ops::Range, path::PathBuf};

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
}
