//! Conversation messages as chat templates read them.

use serde::{Deserialize, Serialize, ser::SerializeStruct};
use serde_json::Value;

/// A checkpoint-template role with the spellings expected by Qwen's Jinja.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum ChatRole {
    System,
    User,
    Assistant,
    Tool,
}

/// One completed assistant tool call retained in conversation history.
#[derive(Clone, Debug, Deserialize, PartialEq)]
pub struct ChatToolCall {
    pub name: String,
    pub arguments: Value,
}

impl Serialize for ChatToolCall {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let mut state = serializer.serialize_struct("ChatToolCall", 3)?;
        state.serialize_field("name", &self.name)?;
        state.serialize_field("arguments", &self.arguments)?;
        state.serialize_field(
            "function",
            &serde_json::json!({"name": self.name, "arguments": self.arguments}),
        )?;
        state.end()
    }
}

/// One tool result which can be converted into a template-ready tool message.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ChatToolResult {
    pub tool_call_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    pub content: String,
}

impl ChatToolResult {
    /// Returns the tool-role message consumed by the checkpoint chat template.
    #[must_use]
    pub fn into_message(self) -> ChatMessage {
        ChatMessage {
            role: ChatRole::Tool,
            content: self.content,
            reasoning_content: None,
            tool_calls: Vec::new(),
            tool_call_id: Some(self.tool_call_id),
            name: self.name,
        }
    }
}

/// One concrete message supplied to the checkpoint's chat template.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ChatMessage {
    pub role: ChatRole,
    pub content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<ChatToolCall>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

impl ChatMessage {
    /// Makes a text-only system, user, or assistant message.
    #[must_use]
    pub fn text(role: ChatRole, content: impl Into<String>) -> Self {
        Self {
            role,
            content: content.into(),
            reasoning_content: None,
            tool_calls: Vec::new(),
            tool_call_id: None,
            name: None,
        }
    }
}

/// The parts of one chat turn a template renders.
#[derive(Clone, Copy, Debug)]
pub struct Conversation<'a> {
    pub messages: &'a [ChatMessage],
    /// Template-shaped tool definitions.
    pub tools: &'a [Value],
    pub enable_thinking: bool,
    /// Optional model-specific reasoning effort passed through to templates.
    pub reasoning_effort: Option<&'a str>,
}

impl<'a> Conversation<'a> {
    /// A non-thinking conversation with no tools.
    #[must_use]
    pub const fn new(messages: &'a [ChatMessage]) -> Self {
        Self {
            messages,
            tools: &[],
            enable_thinking: false,
            reasoning_effort: None,
        }
    }
}
