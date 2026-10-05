//! Gemma 4 calls: `<|tool_call>call:NAME{key:<|"|>text<|"|>,n:3}<tool_call|>`.
//!
//! Arguments are not JSON. Keys are bare, strings sit between `<|"|>`
//! delimiters with no escaping, and other values are JSON numbers, booleans
//! or null, or nested `{...}` and `[...]` of the same grammar (llama.cpp's
//! `common/parsers/gemma4.cpp`). Values carry their own types, so the tool
//! schema is not consulted.

use serde_json::{Map, Value};

use super::{MAX_CALLS, ParsedTurn, ToolCall};

const OPEN: &str = "<|tool_call>";
const CLOSE: &str = "<tool_call|>";
const QUOTE: &str = "<|\"|>";

/// Parses complete calls while preserving surrounding text.
pub(crate) fn parse(input: &str) -> Result<ParsedTurn, String> {
    let mut remaining = input;
    let mut text = String::new();
    let mut calls = Vec::new();
    while let Some(start) = remaining.find(OPEN) {
        let before = &remaining[..start];
        if before.contains(CLOSE) {
            return Err("unmatched tool-call closing marker".into());
        }
        text.push_str(before);
        let (call, rest) = call(&remaining[start + OPEN.len()..])?;
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

/// Parses one call after `<|tool_call>`; returns it and the rest.
fn call(input: &str) -> Result<(ToolCall, &str), String> {
    let body = input.strip_prefix("call:").ok_or("incomplete tool call")?;
    let brace = body.find('{').ok_or("incomplete tool call")?;
    let name = &body[..brace];
    if name.is_empty() || name.contains(|c: char| c.is_whitespace() || c == '<') {
        return Err("tool call requires a name".into());
    }
    let (arguments, rest) = value(&body[brace..])?;
    let rest = rest.strip_prefix(CLOSE).ok_or("incomplete tool call")?;
    Ok((
        ToolCall {
            name: name.to_owned(),
            arguments,
        },
        rest,
    ))
}

/// Parses one value at the start of `input`; returns it and the rest.
fn value(input: &str) -> Result<(Value, &str), String> {
    if let Some(rest) = input.strip_prefix(QUOTE) {
        let (string, rest) = rest.split_once(QUOTE).ok_or("incomplete tool call")?;
        return Ok((Value::String(string.to_owned()), rest));
    }
    if let Some(rest) = input.strip_prefix('{') {
        return object(rest);
    }
    if let Some(rest) = input.strip_prefix('[') {
        return array(rest);
    }
    let end = input.find([',', '}', ']']).ok_or("incomplete tool call")?;
    let scalar: Value = serde_json::from_str(input[..end].trim())
        .map_err(|_| "tool argument is not a string, number, boolean or null")?;
    if scalar.is_object() || scalar.is_array() || scalar.is_string() {
        return Err("tool argument is not a string, number, boolean or null".into());
    }
    Ok((scalar, &input[end..]))
}

/// Parses `key:value, ...}` after an opening brace.
fn object(input: &str) -> Result<(Value, &str), String> {
    let mut members = Map::new();
    let mut remaining = input.trim_start();
    if let Some(rest) = remaining.strip_prefix('}') {
        return Ok((Value::Object(members), rest));
    }
    loop {
        let (key, rest) = remaining.split_once(':').ok_or("incomplete tool call")?;
        let key = key.trim();
        let key = key
            .strip_prefix(QUOTE)
            .and_then(|key| key.strip_suffix(QUOTE))
            .unwrap_or(key);
        if key.is_empty() || key.contains(['}', '{', ',']) || members.contains_key(key) {
            return Err("tool arguments need distinct nonempty names".into());
        }
        let (member, rest) = value(rest.trim_start())?;
        members.insert(key.to_owned(), member);
        let rest = rest.trim_start();
        if let Some(rest) = rest.strip_prefix('}') {
            return Ok((Value::Object(members), rest));
        }
        remaining = rest
            .strip_prefix(',')
            .ok_or("incomplete tool call")?
            .trim_start();
    }
}

/// Parses `value, ...]` after an opening bracket.
fn array(input: &str) -> Result<(Value, &str), String> {
    let mut items = Vec::new();
    let mut remaining = input.trim_start();
    if let Some(rest) = remaining.strip_prefix(']') {
        return Ok((Value::Array(items), rest));
    }
    loop {
        let (item, rest) = value(remaining)?;
        items.push(item);
        let rest = rest.trim_start();
        if let Some(rest) = rest.strip_prefix(']') {
            return Ok((Value::Array(items), rest));
        }
        remaining = rest
            .strip_prefix(',')
            .ok_or("incomplete tool call")?
            .trim_start();
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::parse;

    fn arguments(call: &str) -> serde_json::Value {
        let turn = parse(&format!("<|tool_call>call:f{call}<tool_call|>")).unwrap();
        assert_eq!(turn.text, "");
        turn.calls.into_iter().next().unwrap().arguments
    }

    /// The argument shapes in vLLM's `vllm/parser/gemma4.py` docstring.
    #[test]
    fn parses_the_upstream_argument_shapes() {
        assert_eq!(
            arguments(r#"{location:<|"|>San Francisco<|"|>,unit:<|"|>celsius<|"|>}"#),
            json!({"location":"San Francisco","unit":"celsius"})
        );
        assert_eq!(
            arguments("{count:42,flag:true}"),
            json!({"count":42,"flag":true})
        );
        assert_eq!(
            arguments(r#"{nested:{inner_key:<|"|>val<|"|>}}"#),
            json!({"nested":{"inner_key":"val"}})
        );
        assert_eq!(
            arguments(r#"{items:[<|"|>a<|"|>,<|"|>b<|"|>]}"#),
            json!({"items":["a","b"]})
        );
    }

    #[test]
    fn strings_keep_quotes_braces_and_newlines_and_scalars_keep_types() {
        assert_eq!(
            arguments(
                r#"{text:<|"|>say "hi", {x}: [1]
done<|"|>, ratio: -1.5e2, none:null, empty:{}, list:[1, false, [ ]]}"#
            ),
            json!({"text":"say \"hi\", {x}: [1]\ndone","ratio":-150.0,"none":null,"empty":{},"list":[1,false,[]]})
        );
        assert_eq!(arguments("{}"), json!({}));
    }

    #[test]
    fn surrounding_text_and_several_calls_survive() {
        let input = r#"Reading.<|tool_call>call:read_file{path:<|"|>a.md<|"|>}<tool_call|><|tool_call>call:list_files{path:<|"|>.<|"|>}<tool_call|>"#;
        let turn = parse(input).unwrap();
        assert_eq!(turn.text, "Reading.");
        assert_eq!(turn.calls[0].name, "read_file");
        assert_eq!(turn.calls[1].arguments, json!({"path":"."}));
    }

    #[test]
    fn malformed_calls_are_rejected() {
        for input in [
            r#"<|tool_call>call:f{path:<|"|>a<|"|>}"#,
            r#"<|tool_call>call:f{path:<|"|>a}<tool_call|>"#,
            "<|tool_call>call:f{path:bare words}<tool_call|>",
            "<|tool_call>call:f{path:\"json string\"}<tool_call|>",
            "<|tool_call>call:f{a:1,a:2}<tool_call|>",
            "<|tool_call>call:{a:1}<tool_call|>",
            "<|tool_call>f{a:1}<tool_call|>",
            "<|tool_call>call:f{a:1 b:2}<tool_call|>",
            "done<tool_call|>",
        ] {
            assert!(parse(input).is_err(), "{input}");
        }
        let nine = "<|tool_call>call:f{}<tool_call|>".repeat(9);
        assert!(parse(&nine).is_err());
        assert!(parse("call:f{a:1} is prose").unwrap().calls.is_empty());
    }
}
