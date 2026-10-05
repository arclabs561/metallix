//! One finished assistant turn, parsed in the dialect its checkpoint's
//! template teaches: reasoning, visible text and tool calls.
//!
//! [`parse_turn`] is the only parser entry. The dialect is chosen once, from
//! the template source, and an unrecognized or ambiguous template selects
//! plain text, so a model's call syntax is never misread as another family's.

use serde::Serialize;
use serde_json::Value;

use crate::{
    messages::ChatToolCall,
    tools::{self, ParsedTurn, json_in_tags},
};

/// How a template tells the model to write tool calls.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolDialect {
    /// `<tool_call>{"name": ..., "arguments": {...}}</tool_call>` (Qwen3).
    JsonInTags,
    /// No calls are parsed; the whole answer is text.
    PlainText,
}

/// How a template marks the model's reasoning.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ReasoningDialect {
    /// `<think>...</think>` (Qwen3).
    ThinkTags,
    /// No reasoning is split from the answer.
    None,
}

/// The tool and reasoning dialects of one checkpoint.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub struct TurnFormat {
    pub tools: ToolDialect,
    pub reasoning: ReasoningDialect,
}

impl TurnFormat {
    /// No tool calls and no reasoning: what an unrecognized template gets.
    pub const PLAIN: Self = Self {
        tools: ToolDialect::PlainText,
        reasoning: ReasoningDialect::None,
    };

    /// Selects each dialect whose markers the template source spells. A
    /// template matching several tool dialects selects none.
    #[must_use]
    pub fn from_template(source: &str) -> Self {
        let spells = |markers: &[&str]| markers.iter().all(|marker| source.contains(marker));
        let tools: Vec<ToolDialect> = [(ToolDialect::JsonInTags, &["<tool_call>"][..])]
            .into_iter()
            .filter(|(_, markers)| spells(markers))
            .map(|(dialect, _)| dialect)
            .collect();
        let reasoning = if spells(&["<think>", "</think>"]) {
            ReasoningDialect::ThinkTags
        } else {
            ReasoningDialect::None
        };
        Self {
            tools: match tools.as_slice() {
                [dialect] => *dialect,
                _ => ToolDialect::PlainText,
            },
            reasoning,
        }
    }
}

impl ToolDialect {
    fn parse(self, answer: &str) -> Result<ParsedTurn, String> {
        match self {
            Self::JsonInTags => json_in_tags::parse(answer),
            Self::PlainText => Ok(ParsedTurn {
                text: answer.to_owned(),
                calls: Vec::new(),
            }),
        }
    }
}

impl ReasoningDialect {
    /// Splits a thinking turn into its reasoning and its answer. The model
    /// may open the block itself, or the template may pre-fill the opening
    /// marker and leave only the close.
    fn split(self, text: &str) -> (&str, &str) {
        let (open, close) = match self {
            Self::ThinkTags => ("<think>", "</think>"),
            Self::None => return ("", text),
        };
        let (opened, body) = match text.trim_start().strip_prefix(open) {
            Some(body) => (true, body),
            None => (false, text),
        };
        match body.split_once(close) {
            Some((reasoning, answer)) => (reasoning.trim(), answer.trim_start()),
            None if opened => (body.trim(), ""),
            None => ("", text),
        }
    }
}

/// A finished assistant turn split into reasoning, visible text and tool calls
/// checked against their declared schemas. Each protocol only reshapes it.
#[derive(Debug, PartialEq)]
pub struct AssistantTurn {
    pub reasoning: String,
    pub text: String,
    pub calls: Vec<ChatToolCall>,
    /// The model ended its turn; otherwise it hit the output limit.
    pub complete: bool,
}

/// Parses one generation. `tools` are template-shaped definitions
/// (`{"type": "function", "function": {...}}`). A truncated tool turn is an
/// error, never a partial call.
pub fn parse_turn(
    format: TurnFormat,
    text: &str,
    tools: &[Value],
    enable_thinking: bool,
    complete: bool,
) -> Result<AssistantTurn, String> {
    let (reasoning, answer) = if enable_thinking {
        format.reasoning.split(text)
    } else {
        ("", text)
    };
    let turn = format.tools.parse(answer)?;
    if !turn.calls.is_empty() && !complete {
        return Err("truncated tool turn; no function calls returned".into());
    }
    let mut calls = Vec::new();
    for call in turn.calls {
        let definition = tools
            .iter()
            .find(|tool| tool["function"]["name"] == call.name)
            .ok_or("model requested an undeclared tool")?;
        if !tools::validator(&definition["function"]["parameters"])?.is_valid(&call.arguments) {
            return Err("model tool arguments do not match the declared schema".into());
        }
        calls.push(ChatToolCall {
            name: call.name,
            arguments: call.arguments,
        });
    }
    Ok(AssistantTurn {
        reasoning: reasoning.to_owned(),
        text: turn.text,
        calls,
        complete,
    })
}

#[cfg(test)]
mod tests {
    use serde_json::{Value, json};

    use super::{ReasoningDialect, ToolDialect, TurnFormat, parse_turn};
    use crate::ChatToolCall;

    const QWEN3_TEMPLATE: &str = include_str!("../../../fixtures/qwen3-0.6b/chat-template.jinja");

    const QWEN3: TurnFormat = TurnFormat {
        tools: ToolDialect::JsonInTags,
        reasoning: ReasoningDialect::ThinkTags,
    };

    fn read_file() -> Vec<Value> {
        vec![
            json!({"type":"function","function":{"name":"read_file","parameters":{
            "type":"object","properties":{"path":{"type":"string"}},"required":["path"]}}}),
        ]
    }

    #[test]
    fn dialects_follow_the_template_and_fail_closed() {
        assert_eq!(TurnFormat::from_template(QWEN3_TEMPLATE), QWEN3);
        assert_eq!(
            TurnFormat::from_template("{{ messages }}"),
            TurnFormat::PLAIN
        );
    }

    #[test]
    fn think_tags_split_reasoning_from_the_answer() {
        let split = |text| ReasoningDialect::ThinkTags.split(text);
        assert_eq!(split("<think>\nadd them\n</think>\n\n4"), ("add them", "4"));
        assert_eq!(split("pre-filled\n</think>\n\n4"), ("pre-filled", "4"));
        assert_eq!(split("<think>\nunfinished"), ("unfinished", ""));
        assert_eq!(split("plain answer"), ("", "plain answer"));
        assert_eq!(
            ReasoningDialect::None.split("<think>x</think>y"),
            ("", "<think>x</think>y")
        );
    }

    #[test]
    fn turns_validate_calls_against_declared_tools() {
        let call = "<think>\nread it\n</think>\n\n<tool_call>\n{\"name\": \"read_file\", \"arguments\": {\"path\": \"a.md\"}}\n</tool_call>";
        let turn = parse_turn(QWEN3, call, &read_file(), true, true).unwrap();
        assert_eq!(turn.reasoning, "read it");
        assert_eq!(turn.text, "");
        assert_eq!(
            turn.calls,
            [ChatToolCall {
                name: "read_file".into(),
                arguments: json!({"path":"a.md"}),
            }]
        );
        assert!(parse_turn(QWEN3, call, &read_file(), true, false).is_err());
        assert!(parse_turn(QWEN3, call, &[], true, true).is_err());
        let wrong =
            "<tool_call>{\"name\": \"read_file\", \"arguments\": {\"path\": 5}}</tool_call>";
        assert!(parse_turn(QWEN3, wrong, &read_file(), false, true).is_err());
        // Plain text keeps any markup the model writes as text.
        let plain = parse_turn(TurnFormat::PLAIN, call, &read_file(), true, true).unwrap();
        assert_eq!((plain.reasoning.as_str(), plain.text.as_str()), ("", call));
        assert!(plain.calls.is_empty());
    }
}
