//! Complete-call parsers, one module per tool dialect, and tool schema
//! validation. Reached only through [`crate::parse_turn`].

pub(crate) mod gemma_call;
pub(crate) mod grammar;
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
/// text. A value is a JSON string when the tool schema lets the parameter be
/// a `string` (its `type`, an `anyOf` or `oneOf` branch, or a local `$ref`
/// target, as Pydantic writes `Optional[str]`), and is otherwise decoded as
/// JSON when it parses, as `SGLang`'s `minicpm5` parser does; schema
/// validation follows. Templates print non-string history values with
/// Python's `str`, so the bare literals `True`, `False` and `None` are
/// accepted too. Qwen calls use [`typed_qwen_parameter`] instead.
fn typed_parameter(tools: &[Value], function: &str, key: &str, raw: String) -> Value {
    if parameter_schema(tools, function, key)
        .is_some_and(|(root, schema)| allows_string(root, schema, 0))
    {
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

/// Types a Qwen3-Coder parameter as `SGLang`'s `qwen3_coder` parser
/// (`convert_param_value`) does: `null` in any case is null; a parameter the
/// tool does not declare, or declares without a type, stays text; otherwise
/// the value is typed as in [`typed_parameter`].
fn typed_qwen_parameter(tools: &[Value], function: &str, key: &str, raw: String) -> Value {
    if raw.eq_ignore_ascii_case("null") {
        return Value::Null;
    }
    let typed = parameter_schema(tools, function, key).is_some_and(|(_, schema)| {
        ["type", "anyOf", "oneOf", "$ref"]
            .iter()
            .any(|keyword| schema.get(*keyword).is_some())
    });
    if !typed {
        return Value::String(raw);
    }
    typed_parameter(tools, function, key, raw)
}

/// The tool's parameters schema and the schema of `key` in it, when `key` is
/// declared.
fn parameter_schema<'a>(
    tools: &'a [Value],
    function: &str,
    key: &str,
) -> Option<(&'a Value, &'a Value)> {
    let tool = tools
        .iter()
        .find(|tool| tool["function"]["name"] == function)?;
    let parameters = &tool["function"]["parameters"];
    let schema = parameters["properties"].get(key)?;
    Some((parameters, schema))
}

/// Whether `schema` admits a JSON string through its `type`, an `anyOf` or
/// `oneOf` branch, or a `$ref` into `root` (the parameters schema). Nesting
/// past a small depth, as a cyclic `$ref` would, counts as not a string.
fn allows_string(root: &Value, schema: &Value, depth: usize) -> bool {
    if depth > 16 {
        return false;
    }
    let typed = match &schema["type"] {
        Value::String(kind) => kind == "string",
        Value::Array(kinds) => kinds.iter().any(|kind| kind == "string"),
        _ => false,
    };
    let branch = ["anyOf", "oneOf"].iter().any(|keyword| {
        schema[*keyword].as_array().is_some_and(|branches| {
            branches
                .iter()
                .any(|branch| allows_string(root, branch, depth + 1))
        })
    });
    let referenced = schema["$ref"]
        .as_str()
        .and_then(|reference| reference.strip_prefix('#'))
        .and_then(|pointer| root.pointer(pointer))
        .is_some_and(|target| allows_string(root, target, depth + 1));
    typed || branch || referenced
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

    use super::{typed_parameter, validator};

    /// Pydantic writes `Optional[str]` as `anyOf` with a string branch, with
    /// no top-level `type`; its digits stay text. A parameter declared
    /// without a type also stays text, as `SGLang`'s `qwen3_coder` parser
    /// keeps it.
    #[test]
    fn qwen_xml_optional_and_untyped_parameters_stay_text() {
        use crate::{ReasoningDialect, ToolDialect, TurnFormat, parse_turn};
        let tools = [json!({"type": "function", "function": {
            "name": "lookup",
            "parameters": {"type": "object", "properties": {
                "zip": {"anyOf": [{"type": "string"}, {"type": "null"}], "default": null},
                "note": {"description": "free text, no type"}
            }}
        }})];
        let format = TurnFormat {
            tools: ToolDialect::XmlFunctionParams,
            reasoning: ReasoningDialect::None,
        };
        let text = "<tool_call>\n<function=lookup>\n<parameter=zip>\n21701\n</parameter>\n\
                    <parameter=note>\ntrue\n</parameter>\n</function>\n</tool_call>";
        let turn = parse_turn(format, text, &tools, false, true);
        let arguments = turn.as_ref().map(|turn| turn.calls[0].arguments.clone());
        assert_eq!(
            arguments,
            Ok(json!({"zip": "21701", "note": "true"})),
            "observed: {turn:?}"
        );
    }

    #[test]
    fn qwen_parameters_follow_sglang_qwen3_coder_typing() {
        use super::typed_qwen_parameter;
        let tools = [
            json!({"function": {"name": "f", "parameters": {"properties": {
                "count": {"type": "integer"},
                "free": {"description": "no type"}
            }}}}),
        ];
        let typed = |key: &str, raw: &str| typed_qwen_parameter(&tools, "f", key, raw.into());
        assert_eq!(typed("count", "7"), json!(7));
        assert_eq!(typed("free", "7"), json!("7"));
        assert_eq!(typed("undeclared", "[1, 2]"), json!("[1, 2]"));
        assert_eq!(typed("count", "NULL"), json!(null));
        assert_eq!(typed("free", "null"), json!(null));
        // Other dialects keep decoding untyped values as JSON.
        assert_eq!(typed_parameter(&tools, "f", "free", "7".into()), json!(7));
    }

    #[test]
    fn string_parameters_are_found_through_branches_and_local_refs() {
        let tools = [json!({"function": {"name": "f", "parameters": {
            "$defs": {"text": {"type": "string"}, "loop": {"$ref": "#/$defs/loop"}},
            "properties": {
                "one_of": {"oneOf": [{"type": "integer"}, {"type": ["string", "null"]}]},
                "referenced": {"$ref": "#/$defs/text"},
                "nested": {"anyOf": [{"$ref": "#/$defs/text"}, {"type": "null"}]},
                "cyclic": {"$ref": "#/$defs/loop"},
                "number": {"anyOf": [{"type": "integer"}, {"type": "null"}]}
            }
        }}})];
        let typed = |key: &str| typed_parameter(&tools, "f", key, String::from("7"));
        assert_eq!(typed("one_of"), json!("7"));
        assert_eq!(typed("referenced"), json!("7"));
        assert_eq!(typed("nested"), json!("7"));
        assert_eq!(typed("cyclic"), json!(7));
        assert_eq!(typed("number"), json!(7));
        assert_eq!(typed("absent"), json!(7));
    }

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
