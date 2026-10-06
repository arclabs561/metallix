//! `MiniCPM5` calls: `<function name="NAME"><param name="K">V</param></function>`,
//! with CDATA around values holding `<`, `&` or a newline.

use serde_json::Value;

use super::{MAX_CALLS, ParsedTurn, ToolCall, typed_parameter};

pub(crate) const OPEN: &str = "<function name=";
const CLOSE: &str = "</function>";

/// Parses complete calls while preserving surrounding text.
///
/// This inverts the checkpoint template. A CDATA value is taken verbatim;
/// other values decode the five XML entities and numeric references and keep
/// anything else literally. Values are not trimmed, because the template
/// writes them unpadded. A value is a JSON string when the tool schema types
/// the parameter `string`, and is otherwise decoded as JSON when it parses,
/// as `SGLang`'s `minicpm5` detector does; schema validation follows. The
/// template prints non-string history values with Python's `str`, so the
/// bare literals `True`, `False` and `None` are accepted too.
pub(crate) fn parse(input: &str, tools: &[Value]) -> Result<ParsedTurn, String> {
    let mut remaining = input;
    let mut text = String::new();
    let mut calls = Vec::new();
    while let Some(start) = remaining.find(OPEN) {
        let before = &remaining[..start];
        if before.contains(CLOSE) {
            return Err("unmatched function closing tag".into());
        }
        text.push_str(before);
        let (call, rest) = xml_call(&remaining[start + OPEN.len()..], tools)?;
        calls.push(call);
        if calls.len() > MAX_CALLS {
            return Err("at most eight calls per turn".into());
        }
        remaining = rest;
    }
    if remaining.contains(CLOSE) {
        return Err("unmatched function closing tag".into());
    }
    text.push_str(remaining);
    Ok(ParsedTurn { text, calls })
}

/// Parses one call after `<function name=`; returns it and the rest.
fn xml_call<'a>(input: &'a str, tools: &[Value]) -> Result<(ToolCall, &'a str), String> {
    let (name, mut remaining) = quoted_attribute(input).ok_or("incomplete tool call")?;
    if name.is_empty() {
        return Err("tool call requires a name".into());
    }
    let mut arguments = serde_json::Map::new();
    loop {
        remaining = remaining.trim_start();
        if let Some(rest) = remaining.strip_prefix(CLOSE) {
            return Ok((
                ToolCall {
                    name: name.to_owned(),
                    arguments: Value::Object(arguments),
                },
                rest,
            ));
        }
        let (key, rest) = remaining
            .strip_prefix("<param name=")
            .and_then(quoted_attribute)
            .ok_or("incomplete tool call")?;
        let (raw, rest) = if let Some(cdata) = rest.strip_prefix("<![CDATA[") {
            let end = cdata.find("]]>").ok_or("incomplete tool call")?;
            let rest = cdata[end + 3..]
                .strip_prefix("</param>")
                .ok_or("CDATA must be the whole parameter value")?;
            (cdata[..end].to_owned(), rest)
        } else {
            let end = rest.find("</param>").ok_or("incomplete tool call")?;
            (
                decode_xml_text(&rest[..end]),
                &rest[end + "</param>".len()..],
            )
        };
        if key.is_empty() || arguments.contains_key(key) {
            return Err("tool parameters need distinct nonempty names".into());
        }
        arguments.insert(key.to_owned(), typed_parameter(tools, name, key, raw));
        remaining = rest;
    }
}

/// Reads `"value">` or `'value'>` and returns the value and what follows.
fn quoted_attribute(input: &str) -> Option<(&str, &str)> {
    let quote = input.chars().next().filter(|c| matches!(c, '"' | '\''))?;
    let input = &input[1..];
    let end = input.find(quote)?;
    let value = &input[..end];
    if value.contains('<') {
        return None;
    }
    Some((value, input[end + 1..].strip_prefix('>')?))
}

fn decode_xml_text(raw: &str) -> String {
    let mut decoded = String::with_capacity(raw.len());
    let mut remaining = raw;
    while let Some(start) = remaining.find('&') {
        decoded.push_str(&remaining[..start]);
        remaining = &remaining[start..];
        let entity = remaining
            .find(';')
            .filter(|&end| end <= 10)
            .and_then(|end| Some((xml_entity(&remaining[1..end])?, end)));
        if let Some((character, end)) = entity {
            decoded.push(character);
            remaining = &remaining[end + 1..];
        } else {
            decoded.push('&');
            remaining = &remaining[1..];
        }
    }
    decoded.push_str(remaining);
    decoded
}

fn xml_entity(name: &str) -> Option<char> {
    match name {
        "lt" => Some('<'),
        "gt" => Some('>'),
        "amp" => Some('&'),
        "quot" => Some('"'),
        "apos" => Some('\''),
        _ => {
            let number = name.strip_prefix('#')?;
            let code = match number.strip_prefix(['x', 'X']) {
                Some(hex) => u32::from_str_radix(hex, 16).ok()?,
                None => number.parse().ok()?,
            };
            char::from_u32(code)
        }
    }
}

#[cfg(test)]
mod tests {
    use std::fmt::Write as _;

    use proptest::prelude::*;
    use serde_json::{Value, json};

    use super::parse;
    use crate::tools::{ToolCall, test_definitions as definitions};

    fn typed_tool() -> Value {
        json!({"type":"function","function":{"name":"edit","parameters":{"type":"object","properties":{
            "path":{"type":"string"},"line":{"type":"integer"},"dry":{"type":"boolean"},
            "options":{"type":"object"},"note":{"type":["string","null"]}}}}})
    }

    #[test]
    fn xml_calls_parse_the_reference_greedy_output() {
        // MiniCPM5-2B's greedy turn for fixtures/minicpm5-2b chat_tool_call,
        // before its <|im_end|>.
        let turn = parse(
            r#"<function name="read_file"><param name="path">README.md</param></function>"#,
            &definitions(),
        )
        .unwrap();
        assert_eq!(turn.text, "");
        assert_eq!(
            turn.calls,
            [ToolCall {
                name: "read_file".into(),
                arguments: json!({"path":"README.md"}),
            }]
        );
    }

    #[test]
    fn xml_values_decode_cdata_entities_and_schema_types() {
        let input = concat!(
            "Editing.\n<function name=\"edit\">",
            "<param name=\"path\">a &amp; b&#x2F;c &lt;d&gt; &bogus; 5</param>",
            "<param name=\"line\">12</param><param name=\"dry\">True</param>",
            "<param name=\"options\">{\"k\": [1, 2]}</param>",
            "<param name=\"note\"><![CDATA[ x < y\n&amp; ]]></param>",
            "</function>\n<function name='edit'><param name='path'>7</param></function> done",
        );
        let turn = parse(input, &[typed_tool()]).unwrap();
        assert_eq!(turn.text, "Editing.\n\n done");
        assert_eq!(
            turn.calls[0].arguments,
            json!({"path":"a & b/c <d> &bogus; 5","line":12,"dry":true,
                   "options":{"k":[1,2]},"note":" x < y\n&amp; "})
        );
        // A string-typed parameter stays a string even when it reads as JSON.
        assert_eq!(turn.calls[1].arguments, json!({"path":"7"}));
        // Without a declared type, JSON-looking text is decoded; prose is not.
        let undeclared = parse(
            r#"<function name="other"><param name="n">3</param><param name="s">hi there</param></function>"#,
            &[],
        )
        .unwrap();
        assert_eq!(undeclared.calls[0].arguments, json!({"n":3,"s":"hi there"}));
    }

    #[test]
    fn xml_calls_reject_malformed_structure() {
        let tools = definitions();
        for input in [
            r#"<function name="read_file"><param name="path">x</param>"#,
            r#"<function name="read_file"><param name="path">x</function>"#,
            r#"<function name="read_file"><param name="path"><![CDATA[x</param></function>"#,
            r#"<function name="read_file"><param name="path"><![CDATA[x]]> </param></function>"#,
            r#"<function name="read_file"><param name="p">1</param><param name="p">2</param></function>"#,
            r#"<function name=""></function>"#,
            r"<function name=read_file></function>",
            r#"<function name="read_file">stray</function>"#,
            "done</function>",
        ] {
            assert!(parse(input, &tools).is_err(), "{input}");
        }
        let nine =
            r#"<function name="list_files"><param name="path">.</param></function>"#.repeat(9);
        assert!(parse(&nine, &tools).is_err());
        // Text that only mentions the tag name is not a call.
        let prose = parse("use <functions> or function name=x", &tools).unwrap();
        assert!(prose.calls.is_empty());
    }

    proptest! {
        /// Arguments encoded exactly as the MiniCPM5 template writes them
        /// (CDATA when a value holds `<`, `&` or a newline) parse back unchanged.
        #[test]
        fn xml_calls_round_trip_template_encoded_strings(
            arguments in prop::collection::btree_map(
                "[a-z_]{1,8}",
                "[ -~\n\u{e9}\u{4e2d}]{0,24}".prop_filter("CDATA cannot hold ]]>", |v| !v.contains("]]>")),
                0..5,
            ),
            prefix in "[a-zA-Z .]{0,12}",
        ) {
            let properties: serde_json::Map<String, Value> = arguments
                .keys()
                .map(|key| (key.clone(), json!({"type":"string"})))
                .collect();
            let tool = json!({"type":"function","function":{"name":"f","parameters":{"type":"object","properties":properties}}});
            let mut input = format!("{prefix}<function name=\"f\">");
            for (key, value) in &arguments {
                write!(input, "<param name=\"{key}\">").unwrap();
                if value.contains(['<', '&', '\n']) {
                    write!(input, "<![CDATA[{value}]]>").unwrap();
                } else {
                    input.push_str(value);
                }
                input.push_str("</param>");
            }
            input.push_str("</function>");
            let turn = parse(&input, &[tool]).map_err(TestCaseError::fail)?;
            prop_assert_eq!(turn.text, prefix);
            prop_assert_eq!(turn.calls.len(), 1);
            let expected: serde_json::Map<String, Value> = arguments
                .into_iter()
                .map(|(key, value)| (key, Value::String(value)))
                .collect();
            prop_assert_eq!(&turn.calls[0].arguments, &Value::Object(expected));
        }
    }
}
