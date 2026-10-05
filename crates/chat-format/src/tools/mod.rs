//! Complete-call parsers, one module per tool dialect, and tool schema
//! validation. Reached only through [`crate::parse_turn`].

pub(crate) mod json_in_tags;

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

/// Compile only bounded local schemas; tool schemas must not cause retrieval.
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
