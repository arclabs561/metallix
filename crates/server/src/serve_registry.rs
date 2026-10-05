//! Models declared to `mx serve`, from a local manifest and the `--model`
//! shorthand. Every entry loads at startup; on-demand loading is a later step.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::{
    chat_generation::{ChatBackend, ChatSession, ResidentChatLimits},
    julia_decisions::JuliaDecider,
    qwen_decisions,
};

const MAX_MANIFEST_BYTES: u64 = 1024 * 1024;

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub(crate) enum ModelKind {
    /// Qwen3 checkpoint: `/v1/responses` generation and `/v1/decisions`.
    Qwen,
    /// Julia-1 checkpoint on the native CPU path: `/v1/decisions` only.
    Julia,
}

impl ModelKind {
    pub(crate) fn generates(self) -> bool {
        self == Self::Qwen
    }

    pub(crate) fn capabilities(self) -> &'static [&'static str] {
        match self {
            Self::Qwen => &["generate", "decide"],
            Self::Julia => &["decide"],
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(crate) struct ServedEntry {
    pub(crate) id: String,
    pub(crate) kind: ModelKind,
    pub(crate) path: PathBuf,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Manifest {
    models: Vec<ServedEntry>,
}

/// Manifest entries followed by the `--model` shorthand, with unique nonempty IDs.
pub(crate) fn entries(
    manifest: Option<&Path>,
    model: Option<&Path>,
    model_id: &str,
) -> Result<Vec<ServedEntry>, String> {
    let mut entries = match manifest {
        Some(path) => parse_manifest(&read_manifest(path)?)?,
        None => Vec::new(),
    };
    if let Some(model) = model {
        entries.push(ServedEntry {
            id: model_id.to_owned(),
            kind: ModelKind::Qwen,
            path: model.to_owned(),
        });
    }
    if entries.is_empty() {
        return Err("serve needs --model or a --registry with at least one model".into());
    }
    for (index, entry) in entries.iter().enumerate() {
        if entry.id.is_empty() || entries[..index].iter().any(|other| other.id == entry.id) {
            return Err(format!(
                "served model IDs must be nonempty and unique: {:?}",
                entry.id
            ));
        }
    }
    Ok(entries)
}

fn read_manifest(path: &Path) -> Result<Vec<u8>, String> {
    let bytes = std::fs::read(path)
        .map_err(|error| format!("registry {} could not be read: {error}", path.display()))?;
    if bytes.len() as u64 > MAX_MANIFEST_BYTES {
        return Err("registry exceeds 1 MiB".into());
    }
    Ok(bytes)
}

fn parse_manifest(bytes: &[u8]) -> Result<Vec<ServedEntry>, String> {
    serde_json::from_slice::<Manifest>(bytes)
        .map(|manifest| manifest.models)
        .map_err(|error| format!("registry is invalid: {error}"))
}

/// Capabilities one loaded model serves. `ChatBackend` stays generation-only;
/// a capability the model lacks returns `None`.
pub(crate) trait ModelWorker {
    fn chat(&mut self) -> Option<&mut dyn ChatBackend>;
    fn decide(&mut self, body: &[u8], model: &str) -> Option<Result<Value, String>>;
}

impl ModelWorker for ChatSession {
    fn chat(&mut self) -> Option<&mut dyn ChatBackend> {
        Some(self)
    }

    fn decide(&mut self, body: &[u8], model: &str) -> Option<Result<Value, String>> {
        Some(qwen_decisions::decide_with_session(self, body, model))
    }
}

impl ModelWorker for JuliaDecider {
    fn chat(&mut self) -> Option<&mut dyn ChatBackend> {
        None
    }

    fn decide(&mut self, body: &[u8], model: &str) -> Option<Result<Value, String>> {
        Some(JuliaDecider::decide(self, body, model))
    }
}

pub(crate) fn load(
    entry: &ServedEntry,
    limits: ResidentChatLimits,
) -> Result<Box<dyn ModelWorker>, String> {
    Ok(match entry.kind {
        ModelKind::Qwen => Box::new(ChatSession::load(&entry.path, limits)?),
        ModelKind::Julia => Box::new(JuliaDecider::load(&entry.path)?),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn manifest_and_shorthand_declare_unique_typed_models() {
        let manifest = br#"{"models": [{"id": "julia-1", "kind": "julia", "path": "/j"}]}"#;
        let mut parsed = parse_manifest(manifest).unwrap();
        assert_eq!(
            parsed,
            [ServedEntry {
                id: "julia-1".into(),
                kind: ModelKind::Julia,
                path: "/j".into()
            }]
        );
        assert!(!parsed[0].kind.generates());
        assert_eq!(ModelKind::Qwen.capabilities(), ["generate", "decide"]);

        let shorthand = entries(None, Some(Path::new("/q")), "qwen").unwrap();
        assert_eq!(shorthand[0].kind, ModelKind::Qwen);
        assert!(entries(None, None, "qwen").is_err());

        // Duplicate or empty IDs are rejected wherever they come from.
        parsed.push(parsed[0].clone());
        let duplicate = serde_json::json!({"models": parsed.iter().map(|e| serde_json::json!({"id": e.id, "kind": "julia", "path": e.path})).collect::<Vec<_>>()});
        let path = std::env::temp_dir().join(format!("serve-registry-{}.json", std::process::id()));
        std::fs::write(&path, duplicate.to_string()).unwrap();
        let outcome = entries(Some(&path), None, "qwen");
        std::fs::write(
            &path,
            br#"{"models": [{"id": "", "kind": "julia", "path": "/j"}]}"#,
        )
        .unwrap();
        let empty = entries(Some(&path), None, "qwen");
        std::fs::remove_file(&path).unwrap();
        assert!(outcome.unwrap_err().contains("unique"));
        assert!(empty.unwrap_err().contains("unique"));

        for invalid in [
            br#"{"models": [{"id": "x", "kind": "llama", "path": "/x"}]}"#.as_slice(),
            br#"{"models": [{"id": "x", "kind": "julia", "path": "/x", "extra": 1}]}"#,
            br#"{"models": []", "other": 1}"#,
        ] {
            assert!(parse_manifest(invalid).is_err());
        }
    }
}
