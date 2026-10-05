//! Qwen3.5 and Qwen3-Coder calls:
//! `<tool_call>\n<function=NAME>\n<parameter=K>\nV\n</parameter>\n</function>\n</tool_call>`.

use serde_json::Value;

use super::{MAX_CALLS, ParsedTurn, ToolCall, typed_parameter};

const OPEN: &str = "<tool_call>";
const CLOSE: &str = "</tool_call>";

/// Parses complete calls while preserving surrounding text. The template
/// writes each value between newlines, which are not part of it; values are
/// typed by the tool schema as in [`typed_parameter`].
pub(crate) fn parse(input: &str, tools: &[Value]) -> Result<ParsedTurn, String> {
    let mut remaining = input;
    let mut text = String::new();
    let mut calls = Vec::new();
    while let Some(start) = remaining.find(OPEN) {
        let before = &remaining[..start];
        if before.contains(CLOSE) {
            return Err("unmatched tool-call closing marker".into());
        }
        text.push_str(before);
        let (call, rest) = call(&remaining[start + OPEN.len()..], tools)?;
        calls.push(call);
        if calls.len() > MAX_CALLS {
            return Err("at most eight calls per turn".into());
        }
        remaining = rest;
    }
    if remaining.contains(CLOSE) {
        return Err("unmatched tool-call closing marker".into());
    }
    text.push_str(remaining);
    Ok(ParsedTurn { text, calls })
}

/// Parses one call after `<tool_call>`; returns it and the rest.
fn call<'a>(input: &'a str, tools: &[Value]) -> Result<(ToolCall, &'a str), String> {
    let (name, mut remaining) = input
        .trim_start()
        .strip_prefix("<function=")
        .and_then(|rest| rest.split_once('>'))
        .ok_or("incomplete tool call")?;
    if name.is_empty() || name.contains(['<', '\n']) {
        return Err("tool call requires a name".into());
    }
    let mut arguments = serde_json::Map::new();
    loop {
        remaining = remaining.trim_start();
        if let Some(rest) = remaining.strip_prefix("</function>") {
            let rest = rest
                .trim_start()
                .strip_prefix(CLOSE)
                .ok_or("incomplete tool call")?;
            return Ok((
                ToolCall {
                    name: name.to_owned(),
                    arguments: Value::Object(arguments),
                },
                rest,
            ));
        }
        let (key, rest) = remaining
            .strip_prefix("<parameter=")
            .and_then(|rest| rest.split_once('>'))
            .ok_or("incomplete tool call")?;
        let (raw, rest) = rest
            .split_once("</parameter>")
            .ok_or("incomplete tool call")?;
        let raw = raw.strip_prefix('\n').unwrap_or(raw);
        let raw = raw.strip_suffix('\n').unwrap_or(raw);
        if key.is_empty() || key.contains(['<', '\n']) || arguments.contains_key(key) {
            return Err("tool parameters need distinct nonempty names".into());
        }
        arguments.insert(
            key.to_owned(),
            typed_parameter(tools, name, key, raw.to_owned()),
        );
        remaining = rest;
    }
}

#[cfg(test)]
mod tests {
    use serde_json::{Value, json};

    use super::parse;
    use crate::tools::ToolCall;

    fn weather() -> Value {
        json!({"type":"function","function":{"name":"get_current_weather","parameters":{
            "type":"object","properties":{"city":{"type":"string"},"days":{"type":"integer"},
            "units":{"type":["string","null"]},"options":{"type":"object"}}}}})
    }

    /// The shape vLLM's `qwen3_coder` parser tests and the Qwen3.5 template
    /// write: each value on its own lines.
    #[test]
    fn parses_template_shaped_calls_with_schema_types() {
        let input = concat!(
            "Checking.\n<tool_call>\n<function=get_current_weather>\n",
            "<parameter=city>\nDallas\n</parameter>\n",
            "<parameter=days>\n3\n</parameter>\n",
            "<parameter=units>\n12\n</parameter>\n",
            "<parameter=options>\n{\"metric\": true}\n</parameter>\n",
            "</function>\n</tool_call>",
        );
        let turn = parse(input, &[weather()]).unwrap();
        assert_eq!(turn.text, "Checking.\n");
        assert_eq!(
            turn.calls,
            [ToolCall {
                name: "get_current_weather".into(),
                arguments: json!({"city":"Dallas","days":3,"units":"12","options":{"metric":true}}),
            }]
        );
    }

    #[test]
    fn multiline_values_keep_inner_newlines_and_two_calls_parse() {
        let input = concat!(
            "<tool_call>\n<function=write>\n<parameter=body>\nline one\n\nline two\n</parameter>\n</function>\n</tool_call>\n",
            "<tool_call>\n<function=write>\n</function>\n</tool_call>",
        );
        let turn = parse(input, &[]).unwrap();
        assert_eq!(
            turn.calls[0].arguments,
            json!({"body":"line one\n\nline two"})
        );
        assert_eq!(turn.calls[1].arguments, json!({}));
        assert_eq!(turn.text, "\n");
    }

    #[test]
    fn malformed_calls_are_rejected() {
        for input in [
            "<tool_call>\n<function=f>\n<parameter=a>\n1\n</parameter>\n</function>",
            "<tool_call>\n<function=f>\n<parameter=a>\n1\n</function>\n</tool_call>",
            "<tool_call>\n{\"name\": \"f\", \"arguments\": {}}\n</tool_call>",
            "<tool_call>\n<function=>\n</function>\n</tool_call>",
            "<tool_call>\n<function=f>\nstray\n</function>\n</tool_call>",
            "<tool_call>\n<function=f>\n<parameter=a>\n1\n</parameter>\n<parameter=a>\n2\n</parameter>\n</function>\n</tool_call>",
            "done</tool_call>",
        ] {
            assert!(parse(input, &[]).is_err(), "{input}");
        }
        let nine = "<tool_call>\n<function=f>\n</function>\n</tool_call>".repeat(9);
        assert!(parse(&nine, &[]).is_err());
        let prose = parse("use <function=f> outside a call", &[]).unwrap();
        assert!(prose.calls.is_empty());
    }
}
