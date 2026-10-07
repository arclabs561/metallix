//! Complete-call parsers, one module per tool dialect, and tool schema
//! validation. Reached only through [`crate::parse_turn`].

pub(crate) mod gemma_call;
pub(crate) mod json_in_tags;
pub(crate) mod minicpm_xml;
pub(crate) mod xml_function_params;

use serde::Deserialize;
use serde_json::Value;

/// More calls than this in one turn is an error.
const MAX_CALLS: usize = 8;

/// One complete call: a tool name and its JSON arguments object.
#[derive(Debug, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub(crate) struct ToolCall {
    pub(crate) name: String,
    pub(crate) arguments: Value,
}

/// One model turn with complete tool calls removed from its visible text.
#[derive(Debug, PartialEq)]
pub(crate) struct ParsedTurn {
    pub(crate) text: String,
    pub(crate) calls: Vec<ToolCall>,
}

/// Types a parameter written as text, for dialects that write every value as
/// text. A value is a JSON string when the tool schema types the parameter
/// `string`, and is otherwise decoded as JSON when it parses, as `SGLang`'s
/// `minicpm5` and vLLM's `qwen3_coder` parsers do; schema validation
/// follows. Templates print non-string history values with Python's `str`,
/// so the bare literals `True`, `False` and `None` are accepted too.
fn typed_parameter(tools: &[Value], function: &str, key: &str, raw: String) -> Value {
    let declared = tools
        .iter()
        .find(|tool| tool["function"]["name"] == function)
        .map(|tool| &tool["function"]["parameters"]["properties"][key]["type"]);
    let is_string = match declared {
        Some(Value::String(kind)) => kind == "string",
        Some(Value::Array(kinds)) => kinds.iter().any(|kind| kind == "string"),
        _ => false,
    };
    if is_string {
        return Value::String(raw);
    }
    if let Ok(value) = serde_json::from_str(&raw) {
        return value;
    }
    match raw.as_str() {
        "True" => Value::Bool(true),
        "False" => Value::Bool(false),
        "None" => Value::Null,
        _ => Value::String(raw),
    }
}

/// Compile only bounded local schemas; tool schemas must not cause retrieval.
///
/// # Errors
///
/// Returns a message when the serialized schema passes 32 KiB, has a `$ref`
/// or `$dynamicRef` outside the document, or does not compile.
pub fn validator(schema: &Value) -> Result<jsonschema::Validator, String> {
    fn check_refs(value: &Value) -> Result<(), String> {
        match value {
            Value::Object(map) => {
                for (key, value) in map {
                    if matches!(key.as_str(), "$ref" | "$dynamicRef")
                        && value.as_str().is_none_or(|s| !s.starts_with('#'))
                    {
                        return Err("only document-local schema references are supported".into());
                    }
                    check_refs(value)?;
                }
            }
            Value::Array(values) => {
                for value in values {
                    check_refs(value)?;
                }
            }
            _ => {}
        }
        Ok(())
    }
    if schema.to_string().len() > 32 * 1024 {
        return Err("tool schema exceeds 32 KiB".into());
    }
    check_refs(schema)?;
    jsonschema::validator_for(schema).map_err(|e| format!("invalid tool schema: {e}"))
}

#[cfg(test)]
pub(crate) fn test_definitions() -> Vec<Value> {
    vec![
        serde_json::json!({"type":"function","function":{"name":"read_file","parameters":{"type":"object","properties":{"path":{"type":"string"}},"required":["path"],"additionalProperties":false}}}),
        serde_json::json!({"type":"function","function":{"name":"list_files","parameters":{"type":"object","properties":{"path":{"type":"string"}},"required":["path"],"additionalProperties":false}}}),
    ]
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::validator;

    #[test]
    fn schema_validation_stays_local_and_checks_required_arguments() {
        assert!(validator(&json!({"$ref":"https://example.invalid/schema"})).is_err());
        assert!(validator(&json!({"$dynamicRef":"file:///tmp/schema"})).is_err());
        assert!(validator(&json!({"type":"not-a-json-schema-type"})).is_err());
        let local = validator(&json!({
            "$defs":{"path":{"type":"string"}},
            "type":"object", "properties":{"path":{"$ref":"#/$defs/path"}},
            "required":["path"], "additionalProperties":false
        }))
        .unwrap();
        assert!(local.is_valid(&json!({"path":"README.md"})));
        assert!(!local.is_valid(&json!({"path":5})));
        assert!(!local.is_valid(&json!({})));
    }
}
