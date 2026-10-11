//! Extra chat-template variables a caller sets (`chat_template_kwargs`, as
//! in vLLM and `SGLang`), such as Qwen3.6's `preserve_thinking`.

use std::collections::BTreeMap;

use serde_json::Value;

/// Most variables one request (or the server default) may set.
pub const MAX_TEMPLATE_KWARGS: usize = 16;
/// Longest string value, in bytes.
pub const MAX_TEMPLATE_KWARG_STRING_BYTES: usize = 1024;
/// Longest variable name, in bytes.
const MAX_NAME_BYTES: usize = 64;

/// Variables the renderer sets itself, or that would change what is
/// rendered rather than how. A caller setting one gets an error naming it,
/// so a caller variable can never shadow the conversation. The thinking
/// switches have their own request fields, which protocols map before
/// parsing.
const RESERVED: &[&str] = &[
    "messages",
    "tools",
    "add_generation_prompt",
    "enable_thinking",
    "thinking_mode",
    "reasoning_effort",
    "drop_thinking",
    "bos_token",
    "eos_token",
    "chat_template",
    "tokenize",
];

/// Caller-set template variables: at most [`MAX_TEMPLATE_KWARGS`] names,
/// each a template identifier no renderer field uses, each holding a JSON
/// scalar. Built only by [`TemplateKwargs::parse`]; a template reads only
/// the names it declares (see [`crate::ChatTemplate::reads`]).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct TemplateKwargs(BTreeMap<String, Value>);

/// The empty set, which [`crate::Conversation::new`] borrows.
pub static NO_TEMPLATE_KWARGS: TemplateKwargs = TemplateKwargs(BTreeMap::new());

impl TemplateKwargs {
    /// Parses a `chat_template_kwargs` object.
    ///
    /// # Errors
    ///
    /// Returns a message naming the field when `value` is not an object, has
    /// too many names, uses a reserved or non-identifier name, or holds a
    /// value that is not a boolean, a finite number, `null` or a short
    /// string.
    pub fn parse(value: &Value) -> Result<Self, String> {
        let Value::Object(object) = value else {
            return Err("chat_template_kwargs must be an object".into());
        };
        if object.len() > MAX_TEMPLATE_KWARGS {
            return Err(format!(
                "chat_template_kwargs has more than {MAX_TEMPLATE_KWARGS} entries"
            ));
        }
        let mut kwargs = BTreeMap::new();
        for (name, value) in object {
            if !is_identifier(name) {
                return Err(format!(
                    "chat_template_kwargs name {name:?} is not a template identifier"
                ));
            }
            if RESERVED.contains(&name.as_str()) {
                return Err(format!(
                    "chat_template_kwargs cannot set {name:?}, which the server sets"
                ));
            }
            match value {
                // serde_json numbers are always finite.
                Value::Null | Value::Bool(_) | Value::Number(_) => {}
                Value::String(text) if text.len() <= MAX_TEMPLATE_KWARG_STRING_BYTES => {}
                Value::String(_) => {
                    return Err(format!(
                        "chat_template_kwargs.{name} is longer than \
                         {MAX_TEMPLATE_KWARG_STRING_BYTES} bytes"
                    ));
                }
                Value::Array(_) | Value::Object(_) => {
                    return Err(format!(
                        "chat_template_kwargs.{name} must be a boolean, number, string or null"
                    ));
                }
            }
            kwargs.insert(name.clone(), value.clone());
        }
        Ok(Self(kwargs))
    }

    /// Whether no variable is set.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// The variables, sorted by name.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &Value)> {
        self.0.iter().map(|(name, value)| (name.as_str(), value))
    }
}

fn is_identifier(name: &str) -> bool {
    let mut chars = name.chars();
    name.len() <= MAX_NAME_BYTES
        && chars
            .next()
            .is_some_and(|first| first.is_ascii_alphabetic() || first == '_')
        && chars.all(|rest| rest.is_ascii_alphanumeric() || rest == '_')
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::{MAX_TEMPLATE_KWARGS, TemplateKwargs};

    #[test]
    fn scalars_under_template_names_parse() {
        let kwargs = TemplateKwargs::parse(
            &json!({"preserve_thinking": true, "budget": 512, "style": "terse", "x": null}),
        )
        .unwrap();
        let names: Vec<&str> = kwargs.iter().map(|(name, _)| name).collect();
        assert_eq!(names, ["budget", "preserve_thinking", "style", "x"]);
    }

    #[test]
    fn reserved_names_shapes_and_sizes_are_refused() {
        for (value, needle) in [
            (json!([]), "must be an object"),
            (json!({"messages": []}), "\"messages\""),
            (json!({"enable_thinking": false}), "\"enable_thinking\""),
            (json!({"chat_template": "{{ 1 }}"}), "\"chat_template\""),
            (json!({"bos_token": ""}), "\"bos_token\""),
            (json!({"has space": true}), "not a template identifier"),
            (json!({"1st": true}), "not a template identifier"),
            (
                json!({"nested": {"a": 1}}),
                "boolean, number, string or null",
            ),
            (json!({"list": [1]}), "boolean, number, string or null"),
            (json!({"long": "x".repeat(1025)}), "longer than 1024 bytes"),
        ] {
            let error = TemplateKwargs::parse(&value).unwrap_err();
            assert!(error.contains(needle), "{value}: {error}");
        }
        let many: serde_json::Map<String, serde_json::Value> = (0..=MAX_TEMPLATE_KWARGS)
            .map(|index| (format!("k{index}"), json!(true)))
            .collect();
        assert!(TemplateKwargs::parse(&many.into()).is_err());
    }
}
