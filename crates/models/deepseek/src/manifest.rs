//! Validated checkpoint-artifact manifests for DeepSeek-V4.1.
//!
//! This module deliberately does not parse Hugging Face or safetensors index
//! documents. A future parser supplies the discovered revision and artifacts;
//! this boundary ensures the resulting load plan is internally coherent.

use std::{collections::BTreeSet, num::NonZeroU64};

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

/// An invalid V4.1 checkpoint artifact manifest.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum CheckpointManifestError {
    /// The checkpoint revision was empty or only whitespace.
    #[error("checkpoint revision must not be blank")]
    BlankRevision,
    /// An artifact path was empty or only whitespace.
    #[error("checkpoint artifact path must not be blank")]
    BlankPath,
    /// An artifact declared no bytes.
    #[error("checkpoint artifact byte length must be nonzero")]
    ZeroByteFile,
    /// The same artifact path occurred more than once.
    #[error("checkpoint manifest repeats artifact path {0:?}")]
    DuplicatePath(String),
    /// The combined artifact byte length exceeded `u64`.
    #[error("checkpoint manifest artifact size overflows u64")]
    TotalSizeOverflow,
}

#[cfg(test)]
mod tests {
    use super::{CheckpointFile, CheckpointManifestError, V41CheckpointManifest};

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
            CheckpointFile::new("", 1),
            Err(CheckpointManifestError::BlankPath)
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
}
