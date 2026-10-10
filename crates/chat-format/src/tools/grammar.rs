//! Lark grammars (llguidance's dialect) that constrain a turn's tool calls,
//! one per tool dialect beside its parser, so a strict tool's arguments, a
//! required call and a named call hold by construction rather than being
//! checked after generation.

use serde_json::{Value, json};

use crate::turn::{ReasoningDialect, ToolDialect, TurnFormat};

/// What a request demands of the calls in one turn.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub enum ToolRequirement {
    /// Text, calls or both, the model's choice (`tool_choice: auto`).
    #[default]
    Auto,
    /// At least one call (`OpenAI` `required`, Anthropic `any`).
    Required,
    /// At least one call, every one to this tool.
    Named(String),
}

/// How a marker the grammar must emit is written: as text the tokenizer
/// may split, or as a special token the grammar names by id. A special
/// token can only appear where the grammar names it; text never spells one.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Marker {
    Text,
    Special(u32),
}

/// The Lark grammar for one turn, or an error the request should get as a
/// 400. `tools` are template-shaped (`{"type":"function","function":{..}}`);
/// calls to a tool named in `strict` must satisfy its parameter schema,
/// calls to the others only need an arguments object. `prompt` is the
/// rendered generation prompt: its tail says whether a reasoning block is
/// already closed, open, or still to come. `marker` reports how the
/// tokenizer holds a marker string.
///
/// With `final_schema` (a response format), the turn either calls tools or
/// ends in one message that matches the schema; the schema binds only that
/// final message, not text before calls.
pub fn tool_grammar(
    format: TurnFormat,
    tools: &[Value],
    strict: &[String],
    requirement: &ToolRequirement,
    final_schema: Option<&Value>,
    prompt: &str,
    marker: impl Fn(&str) -> Marker,
) -> Result<String, String> {
    let offered: Vec<&Value> = match requirement {
        ToolRequirement::Named(name) => {
            let tool = tools
                .iter()
                .find(|tool| tool["function"]["name"] == name.as_str())
                .ok_or_else(|| {
                    format!("tool_choice names {name:?}, which is not an offered tool")
                })?;
            vec![tool]
        }
        ToolRequirement::Auto | ToolRequirement::Required => tools.iter().collect(),
    };
    if offered.is_empty() {
        return Err("a required tool call needs at least one offered tool".into());
    }
    let calls: Vec<Value> = offered
        .iter()
        .map(|tool| {
            let name = &tool["function"]["name"];
            let arguments = if strict.iter().any(|strict| name == strict.as_str()) {
                tool["function"]["parameters"].clone()
            } else {
                json!({"type": "object"})
            };
            json!({
                "type": "object",
                "properties": {"name": {"const": name}, "arguments": arguments},
                "required": ["name", "arguments"],
                "additionalProperties": false,
            })
        })
        .collect();
    let call_schema = if calls.len() == 1 {
        calls.into_iter().next().expect("one call schema")
    } else {
        json!({"anyOf": calls})
    };
    let calls_required = *requirement != ToolRequirement::Auto;
    let reasoning = reasoning_rule(format.reasoning, prompt);
    match format.tools {
        ToolDialect::JsonInTags => match final_schema {
            None => json_in_tags(
                &call_schema,
                if calls_required { "+" } else { "*" },
                reasoning.as_ref(),
                &marker,
            ),
            Some(final_schema) => json_in_tags_with_final(
                &call_schema,
                final_schema,
                calls_required,
                reasoning.as_ref(),
                &marker,
            ),
        },
        ToolDialect::XmlFunctionParams | ToolDialect::MiniCpmXml | ToolDialect::GemmaCall => Err(
            "strict tools and tool_choice required or named are not supported yet for this \
             model's tool-call format"
                .into(),
        ),
        ToolDialect::PlainText => Err("this model has no tool-call format".into()),
    }
}

/// `(rule, definition)` for a reasoning block the model writes before its
/// answer: none when the prompt already closed it, the rest of an open
/// block when the prompt opened one, otherwise an optional whole block.
fn reasoning_rule(dialect: ReasoningDialect, prompt: &str) -> Option<(&'static str, String)> {
    match dialect {
        ReasoningDialect::ThinkTags => {
            let tail = prompt.trim_end_matches([' ', '\n']);
            if tail.ends_with("</think>") {
                None
            } else if tail.ends_with("<think>") {
                Some(("think", r"think[lazy]: /(?s:.)*<\/think>/".into()))
            } else {
                Some(("think?", r"think[lazy]: /<think>(?s:.)*<\/think>/".into()))
            }
        }
        ReasoningDialect::GemmaChannel | ReasoningDialect::None => None,
    }
}

/// Qwen3: free text, then calls `<tool_call>\n{"name": .., "arguments": ..}\n</tool_call>`
/// with text allowed between and after them, as the parser accepts.
fn json_in_tags(
    call_schema: &Value,
    repeat: &str,
    reasoning: Option<&(&'static str, String)>,
    marker: &impl Fn(&str) -> Marker,
) -> Result<String, String> {
    const OPEN: &str = "<tool_call>";
    const CLOSE: &str = "</tool_call>";
    if marker(OPEN) != Marker::Text || marker(CLOSE) != Marker::Text {
        return Err(
            "this model's tool-call markers are special tokens, which the JSON-in-tags \
                    grammar does not support yet"
                .into(),
        );
    }
    let schema = serde_json::to_string(call_schema).map_err(|error| error.to_string())?;
    // With a reasoning rule, answer text may not open with `<think>`, so a
    // leading block always lexes as reasoning even when it mentions a call.
    let (start, think, not_reasoning) = match reasoning {
        Some((rule, definition)) => (
            format!("{rule} "),
            format!("{definition}\n"),
            " & ~/<think>(?s:.)*/",
        ),
        None => (String::new(), String::new(), ""),
    };
    // Text before a call is one lazy lexeme ending at the opening marker, so
    // the lexer never has to choose where free text stops; text after the
    // last call cannot contain the marker, so no call escapes the schema.
    Ok(format!(
        "start: {start}(pre_call call){repeat} FREE\n\
         {think}\
         pre_call[lazy]: /(?s:.)*{open}/{not_reasoning}\n\
         call: \"\\n\" %json{schema} \"\\n{CLOSE}\"\n\
         FREE: /(?s:.)*/ & ~/(?s:.)*{open}(?s:.)*/{not_reasoning}\n",
        open = regex_escape(OPEN),
    ))
}

/// Qwen3 with a response format: a turn of calls, each optionally preceded
/// by commentary, or (unless a call is required) one message matching
/// `final_schema`. Commentary cannot open (after whitespace) with `{` or
/// `<`, so the first visible character decides the branch: the lexer never
/// weighs prose against the schema's JSON, and a leading reasoning block or
/// marker cannot lex as commentary. Nothing but whitespace may follow the last call or the
/// final message.
fn json_in_tags_with_final(
    call_schema: &Value,
    final_schema: &Value,
    calls_required: bool,
    reasoning: Option<&(&'static str, String)>,
    marker: &impl Fn(&str) -> Marker,
) -> Result<String, String> {
    const OPEN: &str = "<tool_call>";
    const CLOSE: &str = "</tool_call>";
    if marker(OPEN) != Marker::Text || marker(CLOSE) != Marker::Text {
        return Err(
            "this model's tool-call markers are special tokens, which the JSON-in-tags \
                    grammar does not support yet"
                .into(),
        );
    }
    let calls = serde_json::to_string(call_schema).map_err(|error| error.to_string())?;
    let answer = serde_json::to_string(final_schema).map_err(|error| error.to_string())?;
    let (start, think) = match reasoning {
        Some((rule, definition)) => (format!("{rule} "), format!("{definition}\n")),
        None => (String::new(), String::new()),
    };
    let body = if calls_required {
        "calls"
    } else {
        "(calls | final)"
    };
    Ok(format!(
        "start: {start}{body}\n\
         {think}\
         calls: (pre_call call)+ WS\n\
         final: WS %json{answer} WS\n\
         pre_call[lazy]: /\\s*([^{{<\\s](?s:.)*)?{open}/ & ~/(?s:.)*<think>(?s:.)*/\n\
         call: \"\\n\" %json{calls} \"\\n{CLOSE}\"\n\
         WS: /\\s*/\n",
        open = regex_escape(OPEN),
    ))
}

/// Escapes regex metacharacters (and the `/` delimiter) in a literal.
fn regex_escape(literal: &str) -> String {
    let mut escaped = String::with_capacity(literal.len());
    for character in literal.chars() {
        if "\\^$.|?*+()[]{}/".contains(character) {
            escaped.push('\\');
        }
        escaped.push(character);
    }
    escaped
}

#[cfg(test)]
mod tests {
    use serde_json::{Value, json};

    use super::{Marker, ToolRequirement, tool_grammar};
    use crate::turn::{ReasoningDialect, ToolDialect, TurnFormat};

    const QWEN3: TurnFormat = TurnFormat {
        tools: ToolDialect::JsonInTags,
        reasoning: ReasoningDialect::ThinkTags,
    };

    fn weather() -> Vec<Value> {
        vec![
            json!({"type":"function","function":{"name":"get_weather","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}),
        ]
    }

    #[test]
    fn strict_tools_carry_their_schema_and_others_any_object() {
        let closed = "<|im_start|>assistant\n<think>\n\n</think>\n\n";
        let strict = tool_grammar(
            QWEN3,
            &weather(),
            &["get_weather".into()],
            &ToolRequirement::Auto,
            None,
            closed,
            |_| Marker::Text,
        )
        .unwrap();
        assert!(strict.contains(r#""city":{"type":"string"}"#), "{strict}");
        assert!(
            strict.starts_with("start: (pre_call call)* FREE"),
            "{strict}"
        );
        let loose = tool_grammar(
            QWEN3,
            &weather(),
            &[],
            &ToolRequirement::Required,
            None,
            closed,
            |_| Marker::Text,
        )
        .unwrap();
        assert!(!loose.contains("city"), "{loose}");
        assert!(loose.starts_with("start: (pre_call call)+ FREE"), "{loose}");
    }

    #[test]
    fn the_prompt_tail_decides_the_reasoning_rule() {
        let rule = |prompt: &str| {
            tool_grammar(
                QWEN3,
                &weather(),
                &[],
                &ToolRequirement::Auto,
                None,
                prompt,
                |_| Marker::Text,
            )
            .unwrap()
            .lines()
            .next()
            .unwrap()
            .to_owned()
        };
        assert_eq!(
            rule("assistant\n<think>\n\n</think>\n\n"),
            "start: (pre_call call)* FREE"
        );
        assert_eq!(
            rule("assistant\n<think>\n"),
            "start: think (pre_call call)* FREE"
        );
        assert_eq!(rule("assistant\n"), "start: think? (pre_call call)* FREE");
    }

    #[test]
    fn named_choices_and_unsupported_formats_are_refused_up_front() {
        let prompt = "assistant\n";
        let named = ToolRequirement::Named("shell".into());
        assert!(
            tool_grammar(QWEN3, &weather(), &[], &named, None, prompt, |_| {
                Marker::Text
            })
            .is_err()
        );
        assert!(
            tool_grammar(
                QWEN3,
                &[],
                &[],
                &ToolRequirement::Required,
                None,
                prompt,
                |_| { Marker::Text }
            )
            .is_err()
        );
        let special = tool_grammar(
            QWEN3,
            &weather(),
            &[],
            &ToolRequirement::Auto,
            None,
            prompt,
            |_| Marker::Special(1),
        );
        assert!(special.is_err());
        let gemma = TurnFormat {
            tools: ToolDialect::GemmaCall,
            reasoning: ReasoningDialect::GemmaChannel,
        };
        assert!(
            tool_grammar(
                gemma,
                &weather(),
                &[],
                &ToolRequirement::Auto,
                None,
                prompt,
                |_| Marker::Text
            )
            .is_err()
        );
    }

    /// With a response format, a turn is calls (each optionally after
    /// commentary) or one final message matching the schema; a required
    /// call drops the final branch.
    #[test]
    fn a_final_schema_binds_only_the_turn_final_message() {
        let schema = json!({"type":"object","properties":{"answer":{"type":"string"}},"required":["answer"]});
        let grammar = |requirement: &ToolRequirement| {
            tool_grammar(
                QWEN3,
                &weather(),
                &[],
                requirement,
                Some(&schema),
                "assistant\n",
                |_| Marker::Text,
            )
            .unwrap()
        };
        let auto = grammar(&ToolRequirement::Auto);
        assert!(
            auto.starts_with("start: think? (calls | final)\n"),
            "{auto}"
        );
        let final_rule = auto
            .lines()
            .find(|line| line.starts_with("final: WS %json{"))
            .unwrap_or_else(|| panic!("no final rule in {auto}"));
        assert!(final_rule.contains(r#""answer""#), "{final_rule}");
        // Commentary may not open with `{` or `<` after whitespace.
        assert!(
            auto.contains(r"pre_call[lazy]: /\s*([^{<\s](?s:.)*)?<tool_call>/"),
            "{auto}"
        );
        assert!(!auto.contains("FREE"), "{auto}");
        let required = grammar(&ToolRequirement::Required);
        assert!(required.starts_with("start: think? calls\n"), "{required}");
    }
}
