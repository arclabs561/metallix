//! Qwen3 calls: `<tool_call>{"name": ..., "arguments": {...}}</tool_call>`.

use super::{MAX_CALLS, ParsedTurn, ToolCall};

/// Parses complete calls while preserving surrounding text.
pub(crate) fn parse(input: &str) -> Result<ParsedTurn, String> {
    let mut remaining = input;
    let mut text = String::new();
    let mut calls = Vec::new();
    while let Some(start) = remaining.find("<tool_call>") {
        let before = &remaining[..start];
        if before.contains("</tool_call>") {
            return Err("unmatched tool-call closing marker".into());
        }
        text.push_str(before);
        remaining = &remaining[start + "<tool_call>".len()..];
        // The marker may occur inside a JSON string argument. Let JSON consume
        // its complete value before checking the outer envelope delimiter.
        let mut values = serde_json::Deserializer::from_str(remaining).into_iter::<ToolCall>();
        let call = values
            .next()
            .ok_or("incomplete tool call")?
            .map_err(|error| format!("invalid tool call: {error}"))?;
        remaining = remaining[values.byte_offset()..]
            .trim_start()
            .strip_prefix("</tool_call>")
            .ok_or("incomplete tool call or trailing JSON data")?;
        if call.name.is_empty() || !call.arguments.is_object() {
            return Err("tool call requires a name and an arguments object".into());
        }
        calls.push(call);
        if calls.len() > MAX_CALLS {
            return Err("at most eight calls per turn".into());
        }
    }
    if remaining.contains("</tool_call>") {
        return Err("unmatched tool-call closing marker".into());
    }
    text.push_str(remaining);
    Ok(ParsedTurn { text, calls })
}

#[cfg(test)]
mod tests {
    use std::fmt::Write as _;

    use proptest::prelude::*;
    use serde_json::json;

    use super::parse;
    use crate::tools::ToolCall;

    #[test]
    fn complete_calls_only_and_typed_arguments() {
        let turn = parse("before <tool_call>\n{\"name\":\"read_file\",\"arguments\":{\"path\":\"README.md\"}}\n</tool_call> after").unwrap();
        assert_eq!(turn.text, "before  after");
        assert_eq!(turn.calls.len(), 1);
        assert_eq!(turn.calls[0].arguments["path"], "README.md");
        assert!(parse("<tool_call>{\"name\":\"x\"").is_err());
        assert!(parse("<tool_call>{\"name\":\"x\",\"arguments\":\"{}\"}</tool_call>").is_err());
        assert!(parse("</tool_call>").is_err());
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(128))]

        #[test]
        fn json_arguments_and_unicode_text_survive_envelopes(
            prefix in "[a-zA-Z0-9 \n🦀]{0,48}",
            suffix in "[a-zA-Z0-9 \n🦀]{0,48}",
            payloads in prop::collection::vec(prop_oneof![
                any::<String>(),
                Just("</tool_call>".to_owned()),
                Just("<tool_call>".to_owned()),
                Just("quotes \\\" and \\n and 🦀".to_owned()),
            ], 1..=8),
        ) {
            let mut input = prefix.clone();
            let expected: Vec<ToolCall> = payloads.into_iter().map(|payload| ToolCall {
                name: "search_file".into(),
                arguments: json!({"path":"README.md", "query":payload}),
            }).collect();
            for call in &expected {
                let body = json!({"name":call.name,"arguments":call.arguments});
                write!(input, "<tool_call>\n{body}\n</tool_call>").unwrap();
            }
            input.push_str(&suffix);
            let parsed = parse(&input).map_err(TestCaseError::fail)?;
            prop_assert_eq!(parsed.calls, expected);
            prop_assert_eq!(parsed.text, prefix + &suffix);
        }

        #[test]
        fn incomplete_batch_never_exposes_completed_prefix_calls(payload in any::<String>()) {
            let body = json!({"name":"search_file","arguments":{"query":payload}});
            let input = format!("<tool_call>{body}</tool_call><tool_call>{{");
            prop_assert!(parse(&input).is_err());
        }

        #[test]
        fn call_limit_is_enforced_for_every_oversized_batch(count in 9_usize..32) {
            let input = r#"<tool_call>{"name":"list_files","arguments":{"path":"."}}</tool_call>"#.repeat(count);
            prop_assert!(parse(&input).is_err());
        }

        #[test]
        fn every_truncated_envelope_is_rejected(
            payload in any::<String>(),
            cut_seed in any::<usize>(),
        ) {
            let body = json!({"name":"read_file","arguments":{"path":payload}});
            let input = format!("<tool_call>{body}</tool_call>");
            let boundaries: Vec<_> = input.char_indices()
                .map(|(offset, _)| offset)
                .filter(|&offset| offset >= "<tool_call>".len())
                .collect();
            let cut = boundaries[cut_seed % boundaries.len()];
            prop_assert!(parse(&input[..cut]).is_err());
        }

        #[test]
        fn extra_json_value_cannot_be_smuggled_inside_one_envelope(
            payload in any::<String>(),
            extra in prop_oneof![Just(json!(null)), Just(json!(false)), any::<i64>().prop_map(|x| json!(x)), any::<String>().prop_map(|x| json!(x))],
        ) {
            let body = json!({"name":"read_file","arguments":{"path":payload}});
            let input = format!("<tool_call>{body} {extra}</tool_call>");
            prop_assert!(parse(&input).is_err());
        }

        #[test]
        fn nested_argument_values_survive_json_and_envelope_roundtrip(
            strings in prop::collection::vec(any::<String>(), 0..12),
            number in any::<i64>(),
            flag in any::<bool>(),
        ) {
            let arguments = json!({"nested":[{"strings":strings,"number":number,"flag":flag,"nothing":null}],"marker":"</tool_call>"});
            let body = json!({"name":"custom","arguments":arguments});
            let parsed = parse(&format!("<tool_call>{body}</tool_call>")).map_err(TestCaseError::fail)?;
            prop_assert_eq!(parsed.calls.len(), 1);
            prop_assert_eq!(&parsed.calls[0].arguments, &arguments);
            prop_assert!(parsed.text.is_empty());
        }
    }
}
