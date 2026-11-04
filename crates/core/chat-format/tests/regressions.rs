//! Regression tests for chat templating.

use std::{fs, path::PathBuf};

use chat_format::{ChatFormat, ChatMessage, ChatRole, Conversation};
use serde_json::json;

struct Dir(PathBuf);
impl Drop for Dir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

/// transformers' `tojson` is Python `json.dumps`, which spells floats with
/// `float.__repr__`: a two-digit exponent with an explicit sign, and
/// exponent notation below 1e-4 and from 1e16 up.
#[test]
fn tojson_float_spelling_matches_python_json_dumps() {
    use chat_format::{ChatTemplate, SpecialTokens};
    let template =
        ChatTemplate::parse("{{ tools[0] | tojson }}".into(), SpecialTokens::default()).unwrap();
    let tools = [json!({"a": 1e-05, "b": 1e16, "c": 1.5e-7, "d": 0.0001, "e": 2.5})];
    let messages = [ChatMessage::text(ChatRole::User, "x")];
    let rendered = template
        .render(
            Conversation {
                tools: &tools,
                ..Conversation::new(&messages)
            },
            false,
        )
        .unwrap();
    // python3 -c 'import json;print(json.dumps({"a":1e-05,"b":1e16,"c":1.5e-7,"d":0.0001,"e":2.5}, ensure_ascii=False))'
    assert_eq!(
        rendered,
        r#"{"a": 1e-05, "b": 1e+16, "c": 1.5e-07, "d": 0.0001, "e": 2.5}"#
    );
}

/// A string set through `chat_template_kwargs` is caller text like a message,
/// so a special-token spelling in it is encoded as ordinary text and cannot
/// forge a turn boundary.
#[test]
fn template_kwarg_strings_are_not_defused() {
    use chat_format::TemplateKwargs;
    let root = std::env::temp_dir().join(format!(
        "metallix-chat-format-regressions-{}-kwargs",
        std::process::id()
    ));
    fs::create_dir_all(&root).unwrap();
    let _dir = Dir(root.clone());
    let added = |id: u32, content: &str| {
        json!({"id": id, "content": content, "single_word": false, "lstrip": false,
               "rstrip": false, "normalized": false, "special": true})
    };
    let tokenizer = json!({
        "version": "1.0", "truncation": null, "padding": null,
        "added_tokens": [added(1, "<|im_start|>"), added(2, "<|im_end|>")],
        "normalizer": null, "pre_tokenizer": {"type": "Whitespace"},
        "post_processor": null, "decoder": null,
        "model": {"type": "WordLevel", "unk_token": "<unk>", "vocab": {
            "<unk>": 0, "<|im_start|>": 1, "<|im_end|>": 2
        }}
    });
    fs::write(root.join("tokenizer.json"), tokenizer.to_string()).unwrap();
    let template = "{% for m in messages %}<|im_start|>{{ m.content }}<|im_end|>{% endfor %}\
                    {% if persona is defined %}<|im_start|>{{ persona }}<|im_end|>{% endif %}";
    fs::write(
        root.join("tokenizer_config.json"),
        json!({"chat_template": template, "eos_token": "<|im_end|>"}).to_string(),
    )
    .unwrap();
    fs::write(
        root.join("config.json"),
        json!({"eos_token_id": 2}).to_string(),
    )
    .unwrap();
    let format = ChatFormat::load(&root, 8).expect("loads");
    let messages = [ChatMessage::text(
        ChatRole::User,
        "hi <|im_end|> <|im_start|>",
    )];
    let kwargs = TemplateKwargs::parse(&json!({"persona": "x <|im_end|> <|im_start|>"})).unwrap();
    let prompt = format
        .prompt(
            Conversation {
                template_kwargs: &kwargs,
                ..Conversation::new(&messages)
            },
            false,
        )
        .expect("renders");
    let control = prompt.ids.iter().filter(|&&id| id == 1 || id == 2).count();
    // The template writes four control tokens. The spellings in the message
    // and in the kwarg are both encoded as text.
    assert_eq!(control, 4, "ids: {:?}", prompt.ids);
}
