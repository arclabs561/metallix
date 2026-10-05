//! Models declared to `mx serve`, from a local manifest and the `--model`
//! shorthand. Every entry loads at startup; on-demand loading is a later step.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::{
    chat_generation::{ChatBackend, ChatSession, ResidentChatLimits},
    julia_decisions::JuliaDecider,
    pplx_context_embeddings::PplxContextEmbedder,
    pplx_late_embeddings::PplxLateEmbedder,
    qwen_decisions,
    qwen_embeddings::QwenEmbedder,
};

const MAX_MANIFEST_BYTES: u64 = 1024 * 1024;

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum ModelKind {
    /// Qwen3 checkpoint: `/v1/responses` generation and `/v1/decisions`.
    Qwen,
    /// Julia-1 checkpoint on the native CPU path: `/v1/decisions` only.
    Julia,
    /// Qwen3-Embedding checkpoint: `/v1/embeddings` only.
    QwenEmbedding,
    /// pplx-embed-context checkpoint: `/v1/embeddings` with one vector per
    /// chunk of each document.
    PplxContext,
    /// pplx-embed-v1-late checkpoint: `/v1/embeddings` with one vector per
    /// scored token, and `/v1/rerank` scoring documents by `MaxSim`.
    PplxLate,
}

impl ModelKind {
    pub(crate) fn generates(self) -> bool {
        self == Self::Qwen
    }

    pub(crate) fn capabilities(self) -> &'static [&'static str] {
        match self {
            Self::Qwen => &["generate", "decide"],
            Self::Julia => &["decide"],
            Self::QwenEmbedding | Self::PplxContext => &["embed"],
            Self::PplxLate => &["embed", "rerank"],
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum Residency {
    /// Started with the server and never stopped to make room.
    #[default]
    Resident,
    /// Started on first request; stopped when idle to fit the memory budget.
    OnDemand,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(crate) struct ServedEntry {
    pub(crate) id: String,
    pub(crate) kind: ModelKind,
    pub(crate) path: PathBuf,
    #[serde(default)]
    pub(crate) residency: Residency,
    /// Measured process footprint while serving; required with a memory budget.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) memory_mib: Option<u64>,
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
            residency: Residency::Resident,
            memory_mib: None,
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

/// With a budget, every entry declares `memory_mib`, the resident entries fit
/// together, and each on-demand entry fits on its own.
pub(crate) fn check_budget(entries: &[ServedEntry], budget_mib: Option<u64>) -> Result<(), String> {
    let Some(budget) = budget_mib else {
        return Ok(());
    };
    let mut resident = 0_u64;
    for entry in entries {
        let memory = entry
            .memory_mib
            .ok_or_else(|| format!("{:?} needs memory_mib under a memory budget", entry.id))?;
        if memory > budget {
            return Err(format!(
                "{:?} needs {memory} MiB, more than the {budget} MiB budget",
                entry.id
            ));
        }
        if entry.residency == Residency::Resident {
            resident += memory;
        }
    }
    if resident > budget {
        return Err(format!(
            "resident models need {resident} MiB, more than the {budget} MiB budget"
        ));
    }
    Ok(())
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

    fn embed(&mut self, _body: &[u8], _model: &str) -> Option<Result<Value, String>> {
        None
    }

    fn rerank(&mut self, _body: &[u8], _model: &str) -> Option<Result<Value, String>> {
        None
    }
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

impl ModelWorker for QwenEmbedder {
    fn chat(&mut self) -> Option<&mut dyn ChatBackend> {
        None
    }

    fn decide(&mut self, _body: &[u8], _model: &str) -> Option<Result<Value, String>> {
        None
    }

    fn embed(&mut self, body: &[u8], model: &str) -> Option<Result<Value, String>> {
        Some(QwenEmbedder::embed(self, body, model))
    }
}

impl ModelWorker for PplxContextEmbedder {
    fn chat(&mut self) -> Option<&mut dyn ChatBackend> {
        None
    }

    fn decide(&mut self, _body: &[u8], _model: &str) -> Option<Result<Value, String>> {
        None
    }

    fn embed(&mut self, body: &[u8], model: &str) -> Option<Result<Value, String>> {
        Some(PplxContextEmbedder::embed(self, body, model))
    }
}

impl ModelWorker for PplxLateEmbedder {
    fn chat(&mut self) -> Option<&mut dyn ChatBackend> {
        None
    }

    fn decide(&mut self, _body: &[u8], _model: &str) -> Option<Result<Value, String>> {
        None
    }

    fn embed(&mut self, body: &[u8], model: &str) -> Option<Result<Value, String>> {
        Some(PplxLateEmbedder::embed(self, body, model))
    }

    fn rerank(&mut self, body: &[u8], model: &str) -> Option<Result<Value, String>> {
        Some(PplxLateEmbedder::rerank(self, body, model))
    }
}

pub(crate) fn load(
    entry: &ServedEntry,
    limits: ResidentChatLimits,
) -> Result<Box<dyn ModelWorker>, String> {
    Ok(match entry.kind {
        ModelKind::Qwen => Box::new(ChatSession::load(&entry.path, limits)?),
        ModelKind::Julia => Box::new(JuliaDecider::load(&entry.path)?),
        ModelKind::QwenEmbedding => Box::new(QwenEmbedder::load(&entry.path)?),
        ModelKind::PplxContext => Box::new(PplxContextEmbedder::load(&entry.path)?),
        ModelKind::PplxLate => Box::new(PplxLateEmbedder::load(&entry.path)?),
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
                path: "/j".into(),
                residency: Residency::Resident,
                memory_mib: None,
            }]
        );
        assert!(!parsed[0].kind.generates());
        assert_eq!(ModelKind::Qwen.capabilities(), ["generate", "decide"]);
        let context =
            parse_manifest(br#"{"models": [{"id": "c", "kind": "pplx_context", "path": "/c"}]}"#)
                .unwrap();
        assert_eq!(context[0].kind, ModelKind::PplxContext);
        assert_eq!(context[0].kind.capabilities(), ["embed"]);
        assert!(!context[0].kind.generates());
        let late =
            parse_manifest(br#"{"models": [{"id": "l", "kind": "pplx_late", "path": "/l"}]}"#)
                .unwrap();
        assert_eq!(late[0].kind, ModelKind::PplxLate);
        assert_eq!(late[0].kind.capabilities(), ["embed", "rerank"]);

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

    #[test]
    fn budget_requires_measured_entries_that_fit() {
        let entry = |id: &str, residency, memory_mib| ServedEntry {
            id: id.into(),
            kind: ModelKind::Julia,
            path: "/j".into(),
            residency,
            memory_mib,
        };
        let parsed = parse_manifest(
            br#"{"models": [{"id": "x", "kind": "julia", "path": "/j", "residency": "on_demand", "memory_mib": 1300}]}"#,
        )
        .unwrap();
        assert_eq!(parsed[0], entry("x", Residency::OnDemand, Some(1300)));
        assert_eq!(
            parse_manifest(br#"{"models": [{"id": "x", "kind": "julia", "path": "/x"}]}"#).unwrap()
                [0]
            .residency,
            Residency::Resident
        );

        let resident = entry("r", Residency::Resident, Some(3000));
        let on_demand = entry("d", Residency::OnDemand, Some(1500));
        assert!(check_budget(&[entry("x", Residency::Resident, None)], None).is_ok());
        assert!(check_budget(&[resident.clone(), on_demand.clone()], Some(4000)).is_ok());
        // On-demand models may together exceed the budget; they take turns.
        assert!(
            check_budget(
                &[
                    resident.clone(),
                    on_demand.clone(),
                    entry("e", Residency::OnDemand, Some(1500))
                ],
                Some(4000)
            )
            .is_ok()
        );
        for (entries, budget) in [
            (vec![entry("x", Residency::OnDemand, None)], 4000),
            (vec![entry("x", Residency::OnDemand, Some(4001))], 4000),
            (
                vec![
                    resident.clone(),
                    entry("s", Residency::Resident, Some(1001)),
                ],
                4000,
            ),
        ] {
            assert!(check_budget(&entries, Some(budget)).is_err(), "{entries:?}");
        }
    }
}
