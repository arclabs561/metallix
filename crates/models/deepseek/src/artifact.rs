//! Bounded local inspection of a complete DeepSeek-V4.1 artifact directory.
//!
//! The inspector checks required metadata and safetensors headers without
//! reading tensor payloads or following shard paths outside the supplied root.

use std::{
    fs::{self, File},
    io::Read,
    path::Path,
};

use serde::Deserialize;
use thiserror::Error;

use crate::{
    V41ConfigError, V41SafetensorsHeader, V41SafetensorsHeaderError, V41TextContract,
    manifest::{CheckpointManifestError, MlxSafetensorsIndex, V41SafetensorsIndex},
};

const CONFIG_FILE: &str = "config.json";
const TOKENIZER_FILE: &str = "tokenizer.json";
const TOKENIZER_CONFIG_FILE: &str = "tokenizer_config.json";
const CHAT_TEMPLATE_FILE: &str = "chat_template.jinja";
const INDEX_FILE: &str = "model.safetensors.index.json";
const MAX_METADATA_BYTES: u64 = 16 * 1024 * 1024;
const MAX_HEADER_BYTES: u64 = 100 * 1024 * 1024;

/// The supported index format found in an inspected artifact.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum V41ArtifactIndexKind {
    /// A V4.1 safetensors index with a positive declared total size.
    V41,
    /// An MLX safetensors weight-map index without V4.1 size metadata.
    Mlx,
}

/// A bounded, metadata-only inspection of a local V4.1 artifact directory.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct V41ArtifactInspection {
    text_contract: V41TextContract,
    index_kind: V41ArtifactIndexKind,
    tensor_count: usize,
    shard_count: usize,
}

impl V41ArtifactInspection {
    /// Inspects one local artifact root without reading any tensor payload bytes.
    ///
    /// # Errors
    ///
    /// Returns [`V41ArtifactInspectionError`] when required metadata is absent,
    /// malformed, oversized, or not a regular file; when the artifact index or
    /// a referenced shard is invalid; or when a shard resolves outside `root`.
    pub fn inspect(root: &Path) -> Result<Self, V41ArtifactInspectionError> {
        let root = canonical_root(root)?;
        let config = read_regular_utf8(&root, CONFIG_FILE)?;
        let text_contract = V41TextContract::parse(&config)?;
        require_regular_file(&root, TOKENIZER_FILE)?;

        let tokenizer_config = read_regular_utf8(&root, TOKENIZER_CONFIG_FILE)?;
        let tokenizer_config: TokenizerConfig = serde_json::from_str(&tokenizer_config)
            .map_err(V41ArtifactInspectionError::TokenizerConfigJson)?;
        let embedded_template = tokenizer_config
            .chat_template
            .as_deref()
            .is_some_and(|template| !template.trim().is_empty());
        let external_template = if embedded_template {
            false
        } else {
            read_regular_utf8(&root, CHAT_TEMPLATE_FILE)
                .is_ok_and(|template| !template.trim().is_empty())
        };
        if !embedded_template && !external_template {
            return Err(V41ArtifactInspectionError::MissingChatTemplate);
        }

        let index_json = read_regular_utf8(&root, INDEX_FILE)?;
        let index = ParsedIndex::parse(&index_json)?;
        for shard in index.shard_paths() {
            let shard_path = checked_shard_path(&root, shard)?;
            let header = read_shard_header(&shard_path)?;
            if let ParsedIndex::V41(index) = &index {
                header
                    .validate_index_shard(index, shard)
                    .map_err(V41ArtifactInspectionError::InvalidShardHeader)?;
            }
        }

        Ok(Self {
            text_contract,
            index_kind: index.kind(),
            tensor_count: index.tensor_count(),
            shard_count: index.shard_paths().len(),
        })
    }

    /// Returns the parsed V4.1 text contract.
    #[must_use]
    pub const fn text_contract(&self) -> &V41TextContract {
        &self.text_contract
    }

    /// Returns which supported safetensors index format was inspected.
    #[must_use]
    pub const fn index_kind(&self) -> V41ArtifactIndexKind {
        self.index_kind
    }

    /// Returns the number of index tensor assignments checked.
    #[must_use]
    pub const fn tensor_count(&self) -> usize {
        self.tensor_count
    }

    /// Returns the number of distinct shard headers checked.
    #[must_use]
    pub const fn shard_count(&self) -> usize {
        self.shard_count
    }
}

/// An artifact inspection failure.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum V41ArtifactInspectionError {
    /// The root could not be resolved or is not a directory.
    #[error("artifact root is not a readable directory")]
    InvalidRoot,
    /// A required artifact file is absent, unreadable, or not a regular file.
    #[error("required artifact file {artifact} is not a readable regular file")]
    RequiredFile { artifact: &'static str },
    /// A required metadata file exceeds the local inspection budget.
    #[error("artifact file {artifact} exceeds the {MAX_METADATA_BYTES}-byte inspection limit")]
    MetadataTooLarge { artifact: &'static str },
    /// A required artifact file was not UTF-8 text.
    #[error("artifact file {artifact} is not UTF-8 text")]
    InvalidUtf8 { artifact: &'static str },
    /// The V4.1 configuration failed the existing text-contract validation.
    #[error(transparent)]
    Config(#[from] V41ConfigError),
    /// The tokenizer configuration was not JSON.
    #[error("invalid tokenizer configuration JSON: {0}")]
    TokenizerConfigJson(serde_json::Error),
    /// The tokenizer configuration has no usable chat template.
    #[error("tokenizer configuration must contain a nonempty chat_template string")]
    MissingChatTemplate,
    /// Neither supported index parser accepted the artifact index.
    #[error("unsupported safetensors index: v4.1={v41}; mlx={mlx}")]
    UnsupportedIndex {
        /// The standard V4.1 parser's rejection.
        v41: CheckpointManifestError,
        /// The MLX weight-map parser's rejection.
        mlx: CheckpointManifestError,
    },
    /// An indexed shard path was not a relative normal path.
    #[error("safetensors index contains an unsafe shard path")]
    UnsafeShardPath,
    /// An indexed shard resolves outside the artifact root.
    #[error("indexed shard resolves outside the artifact root")]
    ShardOutsideRoot,
    /// An indexed shard is absent, unreadable, or not a regular file.
    #[error("indexed shard is not a readable regular file")]
    InvalidShard,
    /// A shard header exceeded the bounded inspection budget.
    #[error("safetensors shard header exceeds the {MAX_HEADER_BYTES}-byte inspection limit")]
    ShardHeaderTooLarge,
    /// The existing safetensors header parser rejected a shard header.
    #[error("invalid safetensors shard header: {0}")]
    InvalidShardHeader(V41SafetensorsHeaderError),
}

#[derive(Deserialize)]
struct TokenizerConfig {
    chat_template: Option<String>,
}

enum ParsedIndex {
    V41(V41SafetensorsIndex),
    Mlx(MlxSafetensorsIndex),
}

impl ParsedIndex {
    fn parse(json: &str) -> Result<Self, V41ArtifactInspectionError> {
        match V41SafetensorsIndex::parse(json) {
            Ok(index) => Ok(Self::V41(index)),
            Err(v41) => match MlxSafetensorsIndex::parse(json) {
                Ok(index) => Ok(Self::Mlx(index)),
                Err(mlx)
                    if matches!(&v41, CheckpointManifestError::UnsafePath(_))
                        || matches!(&mlx, CheckpointManifestError::UnsafePath(_)) =>
                {
                    Err(V41ArtifactInspectionError::UnsafeShardPath)
                }
                Err(mlx) => Err(V41ArtifactInspectionError::UnsupportedIndex { v41, mlx }),
            },
        }
    }

    const fn kind(&self) -> V41ArtifactIndexKind {
        match self {
            Self::V41(_) => V41ArtifactIndexKind::V41,
            Self::Mlx(_) => V41ArtifactIndexKind::Mlx,
        }
    }

    fn tensor_count(&self) -> usize {
        match self {
            Self::V41(index) => index.tensor_count(),
            Self::Mlx(index) => index.tensor_count(),
        }
    }

    fn shard_paths(&self) -> &[String] {
        match self {
            Self::V41(index) => index.shard_paths(),
            Self::Mlx(index) => index.shard_paths(),
        }
    }
}

fn canonical_root(root: &Path) -> Result<std::path::PathBuf, V41ArtifactInspectionError> {
    let root = fs::canonicalize(root).map_err(|_| V41ArtifactInspectionError::InvalidRoot)?;
    if fs::metadata(&root)
        .map_err(|_| V41ArtifactInspectionError::InvalidRoot)?
        .is_dir()
    {
        Ok(root)
    } else {
        Err(V41ArtifactInspectionError::InvalidRoot)
    }
}

fn require_regular_file(
    root: &Path,
    artifact: &'static str,
) -> Result<(), V41ArtifactInspectionError> {
    let metadata = fs::metadata(root.join(artifact))
        .map_err(|_| V41ArtifactInspectionError::RequiredFile { artifact })?;
    if !metadata.is_file() {
        return Err(V41ArtifactInspectionError::RequiredFile { artifact });
    }
    if metadata.len() > MAX_METADATA_BYTES {
        return Err(V41ArtifactInspectionError::MetadataTooLarge { artifact });
    }
    Ok(())
}

fn read_regular_utf8(
    root: &Path,
    artifact: &'static str,
) -> Result<String, V41ArtifactInspectionError> {
    let path = root.join(artifact);
    let metadata =
        fs::metadata(&path).map_err(|_| V41ArtifactInspectionError::RequiredFile { artifact })?;
    if !metadata.is_file() {
        return Err(V41ArtifactInspectionError::RequiredFile { artifact });
    }
    if metadata.len() > MAX_METADATA_BYTES {
        return Err(V41ArtifactInspectionError::MetadataTooLarge { artifact });
    }

    let mut bytes = Vec::with_capacity(usize::try_from(metadata.len()).unwrap_or(0));
    File::open(path)
        .map_err(|_| V41ArtifactInspectionError::RequiredFile { artifact })?
        .take(MAX_METADATA_BYTES + 1)
        .read_to_end(&mut bytes)
        .map_err(|_| V41ArtifactInspectionError::RequiredFile { artifact })?;
    if bytes.len() > usize::try_from(MAX_METADATA_BYTES).expect("metadata cap fits usize") {
        return Err(V41ArtifactInspectionError::MetadataTooLarge { artifact });
    }
    String::from_utf8(bytes).map_err(|_| V41ArtifactInspectionError::InvalidUtf8 { artifact })
}

fn checked_shard_path(
    root: &Path,
    shard: &str,
) -> Result<std::path::PathBuf, V41ArtifactInspectionError> {
    let relative = Path::new(shard);
    if relative.is_absolute()
        || !relative
            .components()
            .all(|component| matches!(component, std::path::Component::Normal(_)))
    {
        return Err(V41ArtifactInspectionError::UnsafeShardPath);
    }
    let path = root.join(relative);
    let metadata = fs::metadata(&path).map_err(|_| V41ArtifactInspectionError::InvalidShard)?;
    if !metadata.is_file() {
        return Err(V41ArtifactInspectionError::InvalidShard);
    }
    let canonical = fs::canonicalize(path).map_err(|_| V41ArtifactInspectionError::InvalidShard)?;
    if canonical.starts_with(root) {
        Ok(canonical)
    } else {
        Err(V41ArtifactInspectionError::ShardOutsideRoot)
    }
}

fn read_shard_header(path: &Path) -> Result<V41SafetensorsHeader, V41ArtifactInspectionError> {
    let file_bytes = fs::metadata(path)
        .map_err(|_| V41ArtifactInspectionError::InvalidShard)?
        .len();
    let mut file = File::open(path).map_err(|_| V41ArtifactInspectionError::InvalidShard)?;
    let mut prefix = [0_u8; 8];
    file.read_exact(&mut prefix)
        .map_err(|_| V41ArtifactInspectionError::InvalidShard)?;
    let header_bytes = u64::from_le_bytes(prefix);
    if header_bytes > MAX_HEADER_BYTES {
        return Err(V41ArtifactInspectionError::ShardHeaderTooLarge);
    }
    let header_len = usize::try_from(header_bytes)
        .map_err(|_| V41ArtifactInspectionError::ShardHeaderTooLarge)?;
    let mut prefix_and_header = Vec::with_capacity(8 + header_len);
    prefix_and_header.extend_from_slice(&prefix);
    prefix_and_header.resize(8 + header_len, 0);
    file.read_exact(&mut prefix_and_header[8..])
        .map_err(|_| V41ArtifactInspectionError::InvalidShard)?;
    V41SafetensorsHeader::parse_prefixed_header(&prefix_and_header, file_bytes)
        .map_err(V41ArtifactInspectionError::InvalidShardHeader)
}

#[cfg(test)]
mod tests {
    use std::{
        fs::{self, File},
        io::Write,
        path::{Path, PathBuf},
        sync::atomic::{AtomicUsize, Ordering},
    };

    use super::{V41ArtifactIndexKind, V41ArtifactInspection, V41ArtifactInspectionError};

    static NEXT_TEMP_DIR: AtomicUsize = AtomicUsize::new(0);

    const CONFIG: &str = r#"{
      "model_type":"deepseek_v41",
      "text_config":{"model_type":"deepseek_v41_text","num_hidden_layers":40,"n_routed_experts":384,"num_experts_per_tok":6,"engram_max_ngram_size":4},
      "quantization_config":{"quant_method":"fp8","activation_scheme":"dynamic","weight_block_size":[32,32],"scale_fmt":"ue8m0","expert_dtype":"fp4"}
    }"#;

    struct TempArtifact(PathBuf);

    impl TempArtifact {
        fn new() -> Self {
            let id = NEXT_TEMP_DIR.fetch_add(1, Ordering::Relaxed);
            let path = std::env::temp_dir()
                .join(format!("metallix-v41-artifact-{}-{id}", std::process::id()));
            fs::create_dir(&path).expect("create temporary artifact root");
            Self(path)
        }

        fn path(&self) -> &Path {
            &self.0
        }
    }

    impl Drop for TempArtifact {
        fn drop(&mut self) {
            fs::remove_dir_all(&self.0).expect("remove temporary artifact root");
        }
    }

    fn write(root: &Path, name: &str, contents: &[u8]) {
        fs::write(root.join(name), contents).expect("write synthetic artifact file");
    }

    fn write_valid_shard(root: &Path) {
        let header = br#"{"weight":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#;
        let mut file = File::create(root.join("model-00001.safetensors")).expect("create shard");
        file.write_all(&(header.len() as u64).to_le_bytes())
            .expect("write header length");
        file.write_all(header).expect("write header");
        file.write_all(&[0; 4]).expect("write synthetic payload");
    }

    fn write_required_files(root: &Path, tokenizer_config: &str, index: &str) {
        write(root, "config.json", CONFIG.as_bytes());
        write(root, "tokenizer.json", b"{}");
        write(root, "tokenizer_config.json", tokenizer_config.as_bytes());
        write(root, "model.safetensors.index.json", index.as_bytes());
    }

    #[test]
    fn rejects_a_missing_chat_template() {
        let artifact = TempArtifact::new();
        write_required_files(
            artifact.path(),
            "{}",
            r#"{"metadata":{"total_size":4},"weight_map":{"weight":"model-00001.safetensors"}}"#,
        );

        assert!(matches!(
            V41ArtifactInspection::inspect(artifact.path()),
            Err(V41ArtifactInspectionError::MissingChatTemplate)
        ));
    }

    #[test]
    fn rejects_an_unsafe_indexed_shard_path() {
        let artifact = TempArtifact::new();
        write_required_files(
            artifact.path(),
            r#"{"chat_template":"{{ messages }}"}"#,
            r#"{"metadata":{"total_size":4},"weight_map":{"weight":"../outside.safetensors"}}"#,
        );

        assert!(matches!(
            V41ArtifactInspection::inspect(artifact.path()),
            Err(V41ArtifactInspectionError::UnsafeShardPath)
        ));
    }

    #[test]
    fn rejects_an_oversized_tokenizer_before_artifact_loading() {
        let artifact = TempArtifact::new();
        write_required_files(
            artifact.path(),
            r#"{"chat_template":"{{ messages }}"}"#,
            r#"{"metadata":{"total_size":4},"weight_map":{"weight":"model-00001.safetensors"}}"#,
        );
        let file = File::create(artifact.path().join("tokenizer.json"))
            .expect("create oversized tokenizer");
        file.set_len(super::MAX_METADATA_BYTES + 1)
            .expect("extend oversized tokenizer");

        assert!(matches!(
            V41ArtifactInspection::inspect(artifact.path()),
            Err(V41ArtifactInspectionError::MetadataTooLarge {
                artifact: "tokenizer.json"
            })
        ));
    }

    #[test]
    fn inspects_a_minimal_valid_artifact_without_loading_payloads() {
        let artifact = TempArtifact::new();
        write_valid_shard(artifact.path());
        write_required_files(
            artifact.path(),
            r#"{"chat_template":"{{ messages }}"}"#,
            r#"{"metadata":{"total_size":4},"weight_map":{"weight":"model-00001.safetensors"}}"#,
        );

        let inspection = V41ArtifactInspection::inspect(artifact.path()).expect("valid artifact");
        assert_eq!(inspection.index_kind(), V41ArtifactIndexKind::V41);
        assert_eq!(inspection.tensor_count(), 1);
        assert_eq!(inspection.shard_count(), 1);
        assert_eq!(inspection.text_contract().total_layers(), 40);
    }

    #[test]
    fn accepts_the_external_chat_template_layout() {
        let artifact = TempArtifact::new();
        write_valid_shard(artifact.path());
        write_required_files(
            artifact.path(),
            "{}",
            r#"{"metadata":{"total_size":4},"weight_map":{"weight":"model-00001.safetensors"}}"#,
        );
        write(artifact.path(), "chat_template.jinja", b"{{ messages }}");

        let inspection = V41ArtifactInspection::inspect(artifact.path())
            .expect("external chat template artifact");
        assert_eq!(inspection.shard_count(), 1);
    }
}
