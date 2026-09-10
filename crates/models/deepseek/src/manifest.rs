//! Validated checkpoint-artifact manifests for DeepSeek-V4.1.
//!
//! Safetensors indexes supply total size and tensor placement; repository file
//! metadata supplies the individual artifact sizes needed for a load manifest.

use std::{
    collections::{BTreeMap, BTreeSet},
    num::NonZeroU64,
    path::{Component, Path},
};

use serde::Deserialize;
use thiserror::Error;

/// A validated V4.1 checkpoint revision and its required artifacts.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct V41CheckpointManifest {
    revision: String,
    files: Vec<CheckpointFile>,
    total_bytes: u64,
}

impl V41CheckpointManifest {
    /// Validates a checkpoint revision and its complete artifact list.
    ///
    /// # Errors
    ///
    /// Returns [`CheckpointManifestError`] if the revision is blank, an
    /// artifact path is duplicated, or the byte total overflows `u64`.
    pub fn new(
        revision: impl Into<String>,
        files: impl IntoIterator<Item = CheckpointFile>,
    ) -> Result<Self, CheckpointManifestError> {
        let revision = revision.into();
        if revision.trim().is_empty() {
            return Err(CheckpointManifestError::BlankRevision);
        }

        let mut paths: BTreeSet<String> = BTreeSet::new();
        let mut total_bytes = 0_u64;
        let mut validated_files: Vec<CheckpointFile> = Vec::new();
        for file in files {
            let path = file.path.clone();
            if !paths.insert(path.clone()) {
                return Err(CheckpointManifestError::DuplicatePath(path));
            }
            total_bytes = total_bytes
                .checked_add(file.byte_length.get())
                .ok_or(CheckpointManifestError::TotalSizeOverflow)?;
            validated_files.push(file);
        }
        if validated_files.is_empty() {
            return Err(CheckpointManifestError::EmptyManifest);
        }

        Ok(Self {
            revision,
            files: validated_files,
            total_bytes,
        })
    }

    /// Returns the immutable revision that identifies this checkpoint.
    #[must_use]
    pub fn revision(&self) -> &str {
        &self.revision
    }

    /// Returns the validated checkpoint artifacts in declared order.
    #[must_use]
    pub fn files(&self) -> &[CheckpointFile] {
        &self.files
    }

    /// Returns the total checkpoint artifact size in bytes.
    #[must_use]
    pub const fn total_bytes(&self) -> u64 {
        self.total_bytes
    }
}

/// A validated safetensors index before per-shard byte metadata is available.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct V41SafetensorsIndex {
    total_bytes: NonZeroU64,
    tensor_count: usize,
    shard_paths: Vec<String>,
}

impl V41SafetensorsIndex {
    /// Parses a safetensors index document.
    ///
    /// # Errors
    ///
    /// Returns [`CheckpointManifestError`] for invalid size metadata or an
    /// empty or invalid tensor-to-shard mapping.
    pub fn parse(json: &str) -> Result<Self, CheckpointManifestError> {
        let index: RawSafetensorsIndex =
            serde_json::from_str(json).map_err(CheckpointManifestError::IndexJson)?;
        let total_bytes = NonZeroU64::new(index.metadata.total_size)
            .ok_or(CheckpointManifestError::ZeroTotalSize)?;
        if index.weight_map.is_empty() {
            return Err(CheckpointManifestError::EmptyWeightMap);
        }

        let mut shard_paths = BTreeSet::new();
        for (tensor, shard_path) in &index.weight_map {
            if tensor.trim().is_empty() {
                return Err(CheckpointManifestError::BlankTensorName);
            }
            if shard_path.trim().is_empty() {
                return Err(CheckpointManifestError::BlankPath);
            }
            if !is_safe_artifact_path(shard_path) {
                return Err(CheckpointManifestError::UnsafePath(shard_path.clone()));
            }
            shard_paths.insert(shard_path.clone());
        }
        Ok(Self {
            total_bytes,
            tensor_count: index.weight_map.len(),
            shard_paths: shard_paths.into_iter().collect(),
        })
    }

    /// Returns the checkpoint total declared by the index.
    #[must_use]
    pub const fn total_bytes(&self) -> NonZeroU64 {
        self.total_bytes
    }

    /// Returns the number of tensors assigned to shards.
    #[must_use]
    pub const fn tensor_count(&self) -> usize {
        self.tensor_count
    }

    /// Returns deduplicated shard paths in deterministic order.
    #[must_use]
    pub fn shard_paths(&self) -> &[String] {
        &self.shard_paths
    }
}

/// One required checkpoint artifact.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CheckpointFile {
    path: String,
    byte_length: NonZeroU64,
}

impl CheckpointFile {
    /// Creates a non-empty checkpoint artifact entry.
    ///
    /// # Errors
    ///
    /// Returns [`CheckpointManifestError`] when the path is blank or the
    /// declared byte length is zero.
    pub fn new(path: impl Into<String>, byte_length: u64) -> Result<Self, CheckpointManifestError> {
        let path = path.into();
        if path.trim().is_empty() {
            return Err(CheckpointManifestError::BlankPath);
        }
        if !is_safe_artifact_path(&path) {
            return Err(CheckpointManifestError::UnsafePath(path));
        }
        let byte_length =
            NonZeroU64::new(byte_length).ok_or(CheckpointManifestError::ZeroByteFile)?;
        Ok(Self { path, byte_length })
    }

    /// Returns the checkpoint-relative artifact path.
    #[must_use]
    pub fn path(&self) -> &str {
        &self.path
    }

    /// Returns the declared artifact length in bytes.
    #[must_use]
    pub const fn byte_length(&self) -> NonZeroU64 {
        self.byte_length
    }
}

fn is_safe_artifact_path(path: &str) -> bool {
    let path = Path::new(path);
    !path.is_absolute()
        && path
            .components()
            .all(|component| matches!(component, Component::Normal(_)))
}

#[derive(Debug, Deserialize)]
struct RawSafetensorsIndex {
    metadata: RawSafetensorsMetadata,
    weight_map: BTreeMap<String, String>,
}

#[derive(Debug, Deserialize)]
struct RawSafetensorsMetadata {
    total_size: u64,
}

/// An invalid V4.1 checkpoint artifact manifest.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum CheckpointManifestError {
    /// The supplied safetensors index was not JSON.
    #[error("invalid safetensors index JSON: {0}")]
    IndexJson(serde_json::Error),
    /// The checkpoint revision was empty or only whitespace.
    #[error("checkpoint revision must not be blank")]
    BlankRevision,
    /// The checkpoint did not declare any required artifacts.
    #[error("checkpoint manifest must contain at least one artifact")]
    EmptyManifest,
    /// An artifact path was empty or only whitespace.
    #[error("checkpoint artifact path must not be blank")]
    BlankPath,
    /// An artifact path was absolute or escaped the checkpoint root.
    #[error("checkpoint artifact path must be a relative normal path: {0:?}")]
    UnsafePath(String),
    /// An artifact declared no bytes.
    #[error("checkpoint artifact byte length must be nonzero")]
    ZeroByteFile,
    /// The safetensors index did not declare a positive total checkpoint size.
    #[error("safetensors index total size must be nonzero")]
    ZeroTotalSize,
    /// The safetensors index did not assign any tensors to shards.
    #[error("safetensors index weight map must not be empty")]
    EmptyWeightMap,
    /// The safetensors index assigned a blank tensor name.
    #[error("safetensors index tensor name must not be blank")]
    BlankTensorName,
    /// The same artifact path occurred more than once.
    #[error("checkpoint manifest repeats artifact path {0:?}")]
    DuplicatePath(String),
    /// The combined artifact byte length exceeded `u64`.
    #[error("checkpoint manifest artifact size overflows u64")]
    TotalSizeOverflow,
}

#[cfg(test)]
mod tests {
    use super::{
        CheckpointFile, CheckpointManifestError, V41CheckpointManifest, V41SafetensorsIndex,
    };

    #[test]
    fn validates_and_totals_checkpoint_artifacts() {
        let manifest = V41CheckpointManifest::new(
            "main",
            [
                CheckpointFile::new("model-00001.safetensors", 11).expect("valid artifact"),
                CheckpointFile::new("model-00002.safetensors", 29).expect("valid artifact"),
            ],
        )
        .expect("valid manifest");

        assert_eq!(manifest.revision(), "main");
        assert_eq!(manifest.files().len(), 2);
        assert_eq!(manifest.files()[1].path(), "model-00002.safetensors");
        assert_eq!(manifest.files()[1].byte_length().get(), 29);
        assert_eq!(manifest.total_bytes(), 40);
    }

    #[test]
    fn rejects_blank_revision_and_invalid_artifact_entries() {
        let file = CheckpointFile::new("weights.safetensors", 1).expect("valid artifact");
        assert!(matches!(
            V41CheckpointManifest::new(" \t", [file]),
            Err(CheckpointManifestError::BlankRevision)
        ));
        assert!(matches!(
            V41CheckpointManifest::new("main", std::iter::empty::<CheckpointFile>()),
            Err(CheckpointManifestError::EmptyManifest)
        ));
        assert!(matches!(
            CheckpointFile::new("", 1),
            Err(CheckpointManifestError::BlankPath)
        ));
        assert!(matches!(
            CheckpointFile::new("../weights.safetensors", 1),
            Err(CheckpointManifestError::UnsafePath(_))
        ));
        assert!(matches!(
            CheckpointFile::new("weights.safetensors", 0),
            Err(CheckpointManifestError::ZeroByteFile)
        ));
    }

    #[test]
    fn rejects_duplicate_paths_and_total_size_overflow() {
        let duplicate = V41CheckpointManifest::new(
            "commit",
            [
                CheckpointFile::new("weights.safetensors", 1).expect("valid artifact"),
                CheckpointFile::new("weights.safetensors", 2).expect("valid artifact"),
            ],
        );
        assert!(matches!(
            duplicate,
            Err(CheckpointManifestError::DuplicatePath(path)) if path == "weights.safetensors"
        ));

        let overflow = V41CheckpointManifest::new(
            "commit",
            [
                CheckpointFile::new("one", u64::MAX).expect("valid artifact"),
                CheckpointFile::new("two", 1).expect("valid artifact"),
            ],
        );
        assert!(matches!(
            overflow,
            Err(CheckpointManifestError::TotalSizeOverflow)
        ));
    }

    #[test]
    fn parses_total_size_and_deduplicated_shards() {
        let index = V41SafetensorsIndex::parse(
            r#"{
                "metadata": { "total_size": 96 },
                "weight_map": {
                    "layer.1": "model-00002.safetensors",
                    "layer.0": "model-00001.safetensors",
                    "layer.2": "model-00002.safetensors"
                }
            }"#,
        )
        .expect("valid index");
        assert_eq!(index.total_bytes().get(), 96);
        assert_eq!(index.tensor_count(), 3);
        assert_eq!(
            index.shard_paths(),
            ["model-00001.safetensors", "model-00002.safetensors"]
        );
    }

    #[test]
    fn rejects_empty_index_metadata_or_mapping() {
        assert!(matches!(
            V41SafetensorsIndex::parse(r#"{"metadata":{"total_size":0},"weight_map":{}}"#),
            Err(CheckpointManifestError::ZeroTotalSize)
        ));
        assert!(matches!(
            V41SafetensorsIndex::parse(r#"{"metadata":{"total_size":1},"weight_map":{}}"#),
            Err(CheckpointManifestError::EmptyWeightMap)
        ));
    }
}
