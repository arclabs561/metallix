//! Minimal, privacy-preserving receipts for bounded agent execution.

use std::path::{Component, Path};

use serde::Serialize;
use serde_json::Value;
use sha2::{Digest, Sha256};

use crate::chat_generation::{ChatFinishReason, ChatGeneration, ChatGenerationMetrics};

#[derive(Serialize)]
pub(crate) struct AgentReceipt {
    schema_version: u8,
    status: AgentReceiptStatus,
    #[serde(skip_serializing_if = "Option::is_none")]
    final_text: Option<String>,
    turns: Vec<AgentTurnReceipt>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<AgentReceiptError>,
}

#[derive(Serialize)]
#[serde(rename_all = "snake_case")]
enum AgentReceiptStatus {
    Completed,
    Failed,
}

#[derive(Serialize)]
#[serde(rename_all = "snake_case")]
enum AgentReceiptError {
    ExecutionFailed,
}

#[derive(Serialize)]
pub(crate) struct AgentTurnReceipt {
    turn_index: u32,
    finish_reason: ChatFinishReason,
    metrics: ChatGenerationMetrics,
    generated_text_sha256: String,
    generated_text_utf8_bytes: usize,
    calls: Vec<AgentCallReceipt>,
}

#[derive(Serialize)]
pub(crate) struct AgentCallReceipt {
    name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    relative_path: Option<String>,
    arguments_sha256: String,
    outcome: AgentCallOutcome,
}

#[derive(Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum AgentCallOutcome {
    Ok,
    Error,
}

impl AgentReceipt {
    pub(crate) const fn new() -> Self {
        Self {
            schema_version: 1,
            status: AgentReceiptStatus::Failed,
            final_text: None,
            turns: Vec::new(),
            error: None,
        }
    }

    pub(crate) fn push_turn(&mut self, turn: AgentTurnReceipt) {
        self.turns.push(turn);
    }

    pub(crate) fn complete(&mut self, final_text: String) {
        self.status = AgentReceiptStatus::Completed;
        self.final_text = Some(final_text);
        self.error = None;
    }

    pub(crate) fn fail(&mut self) {
        self.status = AgentReceiptStatus::Failed;
        self.final_text = None;
        self.error = Some(AgentReceiptError::ExecutionFailed);
    }
}

impl AgentTurnReceipt {
    pub(crate) fn from_generation(turn_index: u32, generation: &ChatGeneration) -> Self {
        Self {
            turn_index,
            finish_reason: generation.finish_reason,
            metrics: generation.metrics.clone(),
            generated_text_sha256: sha256(generation.text.as_bytes()),
            generated_text_utf8_bytes: generation.text.len(),
            calls: Vec::new(),
        }
    }

    pub(crate) fn record_call(
        &mut self,
        name: String,
        arguments: &Value,
        outcome: AgentCallOutcome,
    ) {
        self.calls.push(AgentCallReceipt {
            relative_path: relative_path(arguments),
            arguments_sha256: hash_arguments(arguments),
            name,
            outcome,
        });
    }
}

/// Returns the requested relative path without ever placing an absolute or
/// escaping pathname in a receipt.
fn relative_path(arguments: &Value) -> Option<String> {
    let path = arguments.get("path")?.as_str()?;
    if Path::new(path)
        .components()
        .all(|component| matches!(component, Component::Normal(_) | Component::CurDir))
    {
        Some(path.to_owned())
    } else {
        None
    }
}

/// SHA-256 of compact, recursively key-sorted JSON. This binds the exact
/// arguments even when a transitive dependency enables `serde_json`'s
/// insertion-order map feature.
pub(crate) fn hash_arguments(arguments: &Value) -> String {
    let mut canonical = arguments.clone();
    canonical.sort_all_objects();
    sha256(&serde_json::to_vec(&canonical).expect("JSON value serializes"))
}

fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::{hash_arguments, relative_path};

    #[test]
    fn argument_hash_is_independent_of_object_key_order() {
        assert_eq!(
            hash_arguments(&json!({"query":"private", "path":"note.txt"})),
            hash_arguments(&json!({"path":"note.txt", "query":"private"})),
        );
    }

    #[test]
    fn receipt_path_cannot_disclose_an_escape_or_absolute_path() {
        assert_eq!(
            relative_path(&json!({"path":"nested/note.txt"})),
            Some("nested/note.txt".into())
        );
        assert_eq!(relative_path(&json!({"path":"../secret"})), None);
        assert_eq!(relative_path(&json!({"path":"/private/secret"})), None);
    }
}
