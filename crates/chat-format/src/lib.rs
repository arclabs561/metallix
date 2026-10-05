//! A checkpoint's chat format, parsed once at session load: its tokenizer,
//! chat template, special tokens and stop tokens.
//!
//! The types here carry the facts generation has gotten wrong per model
//! family. A template renders only through [`ChatTemplate::render`], whose
//! context always carries the tokenizer's `bos_token` and `eos_token`, and
//! printing a variable the context lacks is an error rather than an empty
//! string. Stop tokens are a nonempty set read from `config.json` and
//! `generation_config.json`, each as one ID or a list, and the decode loop
//! classifies every token against that set.
//!
//! The crate has no model or MLX dependency, so its tests run anywhere.

mod messages;
mod template_json;
mod tokenizer;
mod tools;
mod turn;

use std::path::Path;

use minijinja::{Environment, UndefinedBehavior};
use minijinja_contrib::pycompat::unknown_method_callback;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};

pub use crate::{
    messages::{ChatMessage, ChatRole, ChatToolCall, ChatToolResult, Conversation},
    tokenizer::{QwenIncrementalDecode, QwenTokenizer, read_regular_file},
    tools::validator,
    turn::{AssistantTurn, ReasoningDialect, ToolDialect, TurnFormat, parse_turn},
};

pub const MAX_CHAT_TEMPLATE_BYTES: usize = 1024 * 1024;
pub const MAX_GENERATION_CONFIG_BYTES: usize = 64 * 1024;
const MAX_MODEL_CONFIG_BYTES: usize = 1024 * 1024;
const MAX_CHAT_RENDERED_BYTES: usize = 1024 * 1024;
const TEMPLATE_FUEL: u64 = 100_000;

/// One tokenizer vocabulary index. Model and sampler code still pass `i32`;
/// [`TokenId::from_model`] is the checked crossing.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct TokenId(u32);

impl TokenId {
    #[must_use]
    pub const fn new(id: u32) -> Self {
        Self(id)
    }

    pub fn from_model(id: i32) -> Result<Self, String> {
        u32::try_from(id)
            .map(Self)
            .map_err(|_| String::from("token ID is negative"))
    }

    #[must_use]
    pub const fn get(self) -> u32 {
        self.0
    }
}

/// A list with at least one element.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NonEmpty<T> {
    first: T,
    rest: Vec<T>,
}

impl<T> NonEmpty<T> {
    #[must_use]
    pub fn from_vec(mut items: Vec<T>) -> Option<Self> {
        if items.is_empty() {
            return None;
        }
        let first = items.remove(0);
        Some(Self { first, rest: items })
    }

    pub fn iter(&self) -> impl Iterator<Item = &T> {
        std::iter::once(&self.first).chain(&self.rest)
    }
}

/// What one sampled token means to the decode loop.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TokenClass {
    /// Part of the visible output.
    Normal,
    /// Ends the assistant turn.
    EndTurn,
    /// Ends a turn that issued tool calls and now waits for their results
    /// (Gemma 4's `<|tool_response>`).
    ToolEnd,
}

/// Every token that stops generation. Checkpoints list several: Qwen3's
/// `generation_config.json` names `<|im_end|>` and `<|endoftext|>`,
/// `MiniCPM5`'s config `</s>` and `<|im_end|>`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StopTokens {
    end_turn: NonEmpty<TokenId>,
    tool_end: Vec<TokenId>,
}

/// `eos_token_id` as Hugging Face configs write it: one ID or a list.
#[derive(Deserialize)]
#[serde(untagged)]
enum TokenIds {
    One(u32),
    Many(Vec<u32>),
}

impl StopTokens {
    /// The union, in order, of `eos_token_id` from `config.json` and from
    /// `generation_config.json` (which `generate` in transformers uses).
    fn from_configs(config: &Value, generation_config: Option<&Value>) -> Result<Self, String> {
        let mut ids = Vec::new();
        // Multimodal checkpoints (Qwen3.5) keep the text model's IDs in
        // `text_config`.
        for (file, source) in [
            ("config.json", Some(config)),
            ("config.json", config.get("text_config")),
            ("generation_config.json", generation_config),
        ] {
            let Some(value) = source.map(|source| &source["eos_token_id"]) else {
                continue;
            };
            let listed = match Option::<TokenIds>::deserialize(value).map_err(|_| {
                format!("local {file} eos_token_id must be a token ID or a list of them")
            })? {
                None => Vec::new(),
                Some(TokenIds::One(id)) => vec![id],
                Some(TokenIds::Many(ids)) => ids,
            };
            for id in listed.into_iter().map(TokenId::new) {
                if !ids.contains(&id) {
                    ids.push(id);
                }
            }
        }
        let end_turn = NonEmpty::from_vec(ids).ok_or_else(|| {
            String::from("local config.json and generation_config.json list no eos_token_id")
        })?;
        Ok(Self {
            end_turn,
            tool_end: Vec::new(),
        })
    }

    #[must_use]
    pub fn classify(&self, token: TokenId) -> TokenClass {
        if self.end_turn.iter().any(|&id| id == token) {
            TokenClass::EndTurn
        } else if self.tool_end.contains(&token) {
            TokenClass::ToolEnd
        } else {
            TokenClass::Normal
        }
    }

    pub fn iter(&self) -> impl Iterator<Item = TokenId> + '_ {
        self.end_turn.iter().chain(&self.tool_end).copied()
    }
}

/// A special token's spelling and the ID the tokenizer gives it.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SpecialToken {
    text: String,
    id: TokenId,
}

/// The `bos_token` and `eos_token` from `tokenizer_config.json`, which
/// transformers passes to every chat template.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct SpecialTokens {
    bos: Option<SpecialToken>,
    eos: Option<SpecialToken>,
}

impl SpecialTokens {
    /// Reads each token as a string or an `{"content": ...}` object; a null
    /// or absent entry (Qwen3's `bos_token`) has no token. A spelling must be
    /// one tokenizer token, so encoding the rendered text reproduces its ID.
    fn from_config(tokenizer_config: &Value, tokenizer: &QwenTokenizer) -> Result<Self, String> {
        let token = |name: &str| -> Result<Option<SpecialToken>, String> {
            let entry = &tokenizer_config[name];
            let Some(text) = entry.as_str().or_else(|| entry["content"].as_str()) else {
                return Ok(None);
            };
            let id = tokenizer.token_id(text).ok_or_else(|| {
                format!("tokenizer_config.json {name} {text:?} is not a tokenizer token")
            })?;
            Ok(Some(SpecialToken {
                text: text.to_owned(),
                id: TokenId::new(id),
            }))
        };
        Ok(Self {
            bos: token("bos_token")?,
            eos: token("eos_token")?,
        })
    }
}

/// Everything a chat template sees. Built only by [`ChatTemplate::render`],
/// from the template's own special tokens; an absent token stays undefined,
/// so a template that prints it fails instead of rendering nothing.
#[derive(Serialize)]
struct TemplateContext<'a> {
    messages: Value,
    tools: Value,
    add_generation_prompt: bool,
    enable_thinking: bool,
    thinking_mode: &'static str,
    reasoning_effort: &'a str,
    drop_thinking: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    bos_token: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    eos_token: Option<&'a str>,
}

/// A parsed checkpoint chat template and the special tokens it renders with.
pub struct ChatTemplate {
    environment: Environment<'static>,
    sha256: String,
    specials: SpecialTokens,
}

impl ChatTemplate {
    /// Undefined values may be tested (`{% if message.tool_calls %}`), as Hugging
    /// Face templates do with optional message fields, but printing,
    /// iterating or indexing one fails.
    pub fn parse(source: String, specials: SpecialTokens) -> Result<Self, String> {
        let sha256 = format!("{:x}", Sha256::digest(source.as_bytes()));
        let mut environment = Environment::new();
        environment.set_unknown_method_callback(unknown_method_callback);
        environment.add_filter("tojson", template_json::tojson);
        environment.set_undefined_behavior(UndefinedBehavior::SemiStrict);
        environment.set_fuel(Some(TEMPLATE_FUEL));
        environment
            .add_template_owned("chat", source)
            .map_err(|error| format!("local chat template could not be parsed: {error}"))?;
        Ok(Self {
            environment,
            sha256,
            specials,
        })
    }

    /// SHA-256 of the exact template source.
    #[must_use]
    pub fn sha256(&self) -> &str {
        &self.sha256
    }

    /// Renders `conversation`, optionally without the trailing assistant
    /// generation prompt; the prefix cache renders conversation prefixes that
    /// way.
    pub fn render(
        &self,
        conversation: Conversation<'_>,
        add_generation_prompt: bool,
    ) -> Result<String, String> {
        let context = TemplateContext {
            messages: serde_json::to_value(conversation.messages)
                .map_err(|_| String::from("chat messages could not be serialized"))?,
            tools: serde_json::to_value(conversation.tools)
                .map_err(|_| String::from("chat tools could not be serialized"))?,
            add_generation_prompt,
            enable_thinking: conversation.enable_thinking,
            thinking_mode: if conversation.enable_thinking {
                "thinking"
            } else {
                "non-thinking"
            },
            reasoning_effort: conversation.reasoning_effort.unwrap_or("low"),
            drop_thinking: !conversation.enable_thinking,
            bos_token: self.specials.bos.as_ref().map(|token| token.text.as_str()),
            eos_token: self.specials.eos.as_ref().map(|token| token.text.as_str()),
        };
        let rendered = self
            .environment
            .get_template("chat")
            .map_err(|error| format!("local chat template is unavailable: {error}"))?
            .render(context)
            .map_err(|error| format!("local chat template could not be rendered: {error}"))?;
        if rendered.len() > MAX_CHAT_RENDERED_BYTES {
            return Err(format!(
                "rendered chat prompt exceeds the {MAX_CHAT_RENDERED_BYTES}-byte limit"
            ));
        }
        Ok(rendered)
    }
}

/// A checkpoint's tokenizer, template and stop tokens.
pub struct ChatFormat {
    tokenizer: QwenTokenizer,
    template: ChatTemplate,
    turn: TurnFormat,
    stops: StopTokens,
    vocabulary_size: usize,
}

impl ChatFormat {
    /// Loads the format from a checkpoint directory. `vocabulary_size` is the
    /// model's logit width, which every stop and prompt ID must fit.
    pub fn load(model: &Path, vocabulary_size: usize) -> Result<Self, String> {
        let config = read_json(model, "config.json", MAX_MODEL_CONFIG_BYTES)?;
        let generation_config = if model.join("generation_config.json").exists() {
            Some(read_json(
                model,
                "generation_config.json",
                MAX_GENERATION_CONFIG_BYTES,
            )?)
        } else {
            None
        };
        let stops = StopTokens::from_configs(&config, generation_config.as_ref())?;
        let tokenizer = QwenTokenizer::load(model)?;
        for id in stops.iter() {
            let id = i32::try_from(id.get())
                .map_err(|_| String::from("model EOS token ID does not fit server token IDs"))?;
            tokenizer.check_model_vocabulary(vocabulary_size, id)?;
        }
        let tokenizer_config = read_json(model, "tokenizer_config.json", MAX_CHAT_TEMPLATE_BYTES)?;
        let specials = SpecialTokens::from_config(&tokenizer_config, &tokenizer)?;
        let source = load_template(model, &tokenizer_config)?;
        let turn = TurnFormat::from_template(&source);
        let template = ChatTemplate::parse(source, specials)?;
        let format = Self {
            tokenizer,
            template,
            turn,
            stops,
            vocabulary_size,
        };
        format.check_bos()?;
        Ok(format)
    }

    #[must_use]
    pub fn tokenizer(&self) -> &QwenTokenizer {
        &self.tokenizer
    }

    #[must_use]
    pub fn template(&self) -> &ChatTemplate {
        &self.template
    }

    /// The tool and reasoning dialects the template teaches, which
    /// [`parse_turn`] reads generated text in.
    #[must_use]
    pub const fn turn_format(&self) -> TurnFormat {
        self.turn
    }

    #[must_use]
    pub fn stops(&self) -> &StopTokens {
        &self.stops
    }

    /// Encodes a rendered prompt exactly as written, as transformers'
    /// `apply_chat_template` does: no special tokens are added, so a BOS
    /// must come from the template.
    pub fn encode(&self, prompt: &str) -> Result<Vec<i32>, String> {
        let ids = self.tokenizer.encode_prompt(prompt)?;
        if ids.iter().any(|&id| {
            usize::try_from(id)
                .ok()
                .is_none_or(|id| id >= self.vocabulary_size)
        }) {
            return Err(String::from(
                "chat template token IDs are outside model vocabulary",
            ));
        }
        Ok(ids)
    }

    /// When the template begins a prompt with the BOS spelling, the encoded
    /// prompt must begin with exactly one BOS ID: a spelling that splits into
    /// ordinary pieces, or a second BOS, changes what the model sees.
    /// Rendering here also surfaces, at load, a template that prints a
    /// variable this server does not supply.
    fn check_bos(&self) -> Result<(), String> {
        let messages = [ChatMessage::text(ChatRole::User, "Hello")];
        let rendered = self.template.render(Conversation::new(&messages), true)?;
        let Some(bos) = &self.template.specials.bos else {
            return Ok(());
        };
        if !rendered.starts_with(&bos.text) {
            return Ok(());
        }
        let ids = self
            .encode(&rendered)?
            .into_iter()
            .map(TokenId::from_model)
            .collect::<Result<Vec<_>, _>>()?;
        match ids.as_slice() {
            [first, second, ..] if *first == bos.id && *second == bos.id => Err(format!(
                "chat template renders {:?} twice at the start of a prompt",
                bos.text
            )),
            [first, ..] if *first == bos.id => Ok(()),
            _ => Err(format!(
                "chat template starts with {:?}, but the tokenizer does not encode it as BOS ID {}",
                bos.text,
                bos.id.get()
            )),
        }
    }
}

fn read_json(model: &Path, file: &str, maximum_bytes: usize) -> Result<Value, String> {
    let text = String::from_utf8(read_regular_file(&model.join(file), maximum_bytes, file)?)
        .map_err(|_| format!("local {file} could not be read"))?;
    serde_json::from_str(&text).map_err(|_| format!("local {file} could not be parsed"))
}

/// The template in `tokenizer_config.json`, else `chat_template.jinja`.
pub fn load_template(model: &Path, tokenizer_config: &Value) -> Result<String, String> {
    if let Some(template) = tokenizer_config
        .get("chat_template")
        .and_then(Value::as_str)
        .filter(|template| !template.is_empty())
    {
        return Ok(template.to_owned());
    }

    let external = model.join("chat_template.jinja");
    if !external.exists() {
        return Err(String::from(
            "local tokenizer_config.json has no chat_template and chat_template.jinja is absent",
        ));
    }
    let template = String::from_utf8(read_regular_file(
        &external,
        MAX_CHAT_TEMPLATE_BYTES,
        "chat_template.jinja",
    )?)
    .map_err(|_| String::from("local chat_template.jinja could not be read"))?;
    if template.trim().is_empty() {
        return Err(String::from("local chat_template.jinja must not be empty"));
    }
    Ok(template)
}

/// A checkpoint directory holding only the chat-format files, with a small
/// word-level tokenizer whose special tokens are `<s>` (1), `</s>` (2) and
/// `<|im_end|>` (3).
#[cfg(any(test, feature = "test-model"))]
pub mod test_model {
    use std::{
        fs,
        path::{Path, PathBuf},
        sync::atomic::{AtomicUsize, Ordering},
    };

    use serde_json::{Value, json};

    pub const VOCABULARY_SIZE: usize = 8;

    pub struct ModelDir(PathBuf);

    impl ModelDir {
        /// `tokenizer_config` and `config` are written as given;
        /// `generation_config` only when present.
        ///
        /// # Panics
        ///
        /// When the temporary directory cannot be written.
        #[must_use]
        pub fn new(
            tokenizer_config: &Value,
            config: &Value,
            generation_config: Option<&Value>,
        ) -> Self {
            static NEXT: AtomicUsize = AtomicUsize::new(0);
            let root = std::env::temp_dir().join(format!(
                "metallix-chat-format-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            fs::create_dir_all(&root).expect("model directory");
            fs::write(root.join("tokenizer.json"), tokenizer().to_string()).expect("tokenizer");
            fs::write(
                root.join("tokenizer_config.json"),
                tokenizer_config.to_string(),
            )
            .expect("tokenizer config");
            fs::write(root.join("config.json"), config.to_string()).expect("config");
            if let Some(generation_config) = generation_config {
                fs::write(
                    root.join("generation_config.json"),
                    generation_config.to_string(),
                )
                .expect("generation config");
            }
            Self(root)
        }

        #[must_use]
        pub fn path(&self) -> &Path {
            &self.0
        }
    }

    impl Drop for ModelDir {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn tokenizer() -> Value {
        let added = |id: u32, content: &str| {
            json!({"id": id, "content": content, "single_word": false, "lstrip": false,
                   "rstrip": false, "normalized": false, "special": true})
        };
        json!({
            "version": "1.0",
            "truncation": null,
            "padding": null,
            "added_tokens": [added(1, "<s>"), added(2, "</s>"), added(3, "<|im_end|>")],
            "normalizer": null,
            "pre_tokenizer": {"type": "Whitespace"},
            "post_processor": null,
            "decoder": null,
            "model": {"type": "WordLevel", "unk_token": "<unk>", "vocab": {
                "<unk>": 0, "<s>": 1, "</s>": 2, "<|im_end|>": 3, "Hello": 4, "hi": 5, "<bos>": 6
            }}
        })
    }
}

#[cfg(test)]
mod tests {
    use std::fs;

    use serde_json::{Value, json};

    use super::{
        ChatFormat, ChatTemplate, MAX_CHAT_TEMPLATE_BYTES, NonEmpty, SpecialTokens, StopTokens,
        TokenClass, TokenId, load_template, read_json,
        test_model::{ModelDir, VOCABULARY_SIZE},
    };
    use crate::{ChatMessage, ChatRole, Conversation};

    fn render(format: &ChatFormat, content: &str) -> String {
        let messages = [ChatMessage::text(ChatRole::User, content)];
        format
            .template()
            .render(Conversation::new(&messages), true)
            .expect("renders")
    }

    fn load(template: &str, bos: &Value) -> Result<ChatFormat, String> {
        let model = ModelDir::new(
            &json!({"chat_template": template, "bos_token": bos, "eos_token": "</s>"}),
            &json!({"eos_token_id": 2, "vocab_size": VOCABULARY_SIZE}),
            None,
        );
        ChatFormat::load(model.path(), VOCABULARY_SIZE)
    }

    /// `MiniCPM5` and Gemma 4 templates begin with `{{ bos_token }}` and the
    /// prompt is encoded without added special tokens, so the BOS exists only
    /// if the template context carries it.
    #[test]
    fn bos_token_reaches_the_template_and_encodes_as_one_bos() {
        let format = load("{{ bos_token }}{{ messages[0].content }}", &json!("<s>"))
            .expect("template that prints bos_token loads");
        assert_eq!(render(&format, "hi"), "<s>hi");
        assert_eq!(format.encode("<s>hi"), Ok(vec![1, 5]));
        // The `{"content": ...}` spelling transformers also writes.
        let object = load(
            "{{ bos_token }}{{ messages[0].content }}",
            &json!({"content": "<s>"}),
        )
        .expect("object-spelled bos_token loads");
        assert_eq!(render(&object, "hi"), "<s>hi");
    }

    #[test]
    fn bos_probe_rejects_missing_split_and_doubled_bos() {
        // Qwen3 declares no BOS; a template printing one fails at load
        // instead of rendering an empty string.
        let missing = load("{{ bos_token }}{{ messages[0].content }}", &Value::Null);
        assert!(
            missing
                .err()
                .is_some_and(|error| error.contains("could not be rendered"))
        );
        // Without a BOS, the same template minus the token loads.
        assert!(load("{{ messages[0].content }}", &Value::Null).is_ok());
        assert!(
            load(
                "{{ bos_token }}{{ bos_token }}{{ messages[0].content }}",
                &json!("<s>")
            )
            .err()
            .is_some_and(|error| error.contains("twice"))
        );
        // `<bos>` is an ordinary vocabulary word here, which the whitespace
        // pre-tokenizer splits, so the rendered spelling never becomes ID 6.
        assert!(
            load("{{ bos_token }}{{ messages[0].content }}", &json!("<bos>"))
                .err()
                .is_some_and(|error| error.contains("does not encode it as BOS"))
        );
        assert!(
            load("{{ messages[0].content }}", &json!("<absent>"))
                .err()
                .is_some_and(|error| error.contains("not a tokenizer token"))
        );
    }

    #[test]
    fn templates_fail_on_printing_undefined_values_but_may_test_them() {
        let template = |source: &str| {
            ChatTemplate::parse(source.to_owned(), SpecialTokens::default())
                .expect("parses")
                .render(
                    Conversation::new(&[ChatMessage::text(ChatRole::User, "x")]),
                    true,
                )
        };
        assert!(template("{{ undefined_variable }}").is_err());
        assert!(template("{{ messages[0].tool_calls }}").is_err());
        assert_eq!(
            template(
                "{% if messages[0].tool_calls %}calls{% endif %}{% if bos_token is defined %}bos{% endif %}ok"
            ),
            Ok(String::from("ok"))
        );
    }

    /// `MiniCPM5` lists `eos_token_id: [1, 130073]` and Gemma 4 `[1, 106, 50]`;
    /// a scalar field rejected both checkpoints at load.
    #[test]
    fn stop_tokens_are_every_listed_eos_from_both_configs() {
        let stops = |config: Value, generation: Option<Value>| {
            StopTokens::from_configs(&config, generation.as_ref())
        };
        let minicpm = stops(json!({"eos_token_id": [1, 130_073]}), None).expect("list");
        for id in [1, 130_073] {
            assert_eq!(minicpm.classify(TokenId::new(id)), TokenClass::EndTurn);
        }
        assert_eq!(minicpm.classify(TokenId::new(2)), TokenClass::Normal);
        // Qwen3: one ID in config.json, two in generation_config.json.
        let qwen = stops(
            json!({"eos_token_id": 151_645}),
            Some(json!({"eos_token_id": [151_645, 151_643]})),
        )
        .expect("one plus list");
        assert_eq!(
            qwen.iter().map(TokenId::get).collect::<Vec<_>>(),
            [151_645, 151_643]
        );
        // Qwen3.5-0.8B lists its ID only under `text_config`.
        let nested = stops(json!({"text_config": {"eos_token_id": 248_044}}), None)
            .expect("nested text_config");
        assert_eq!(
            nested.iter().map(TokenId::get).collect::<Vec<_>>(),
            [248_044]
        );
        assert!(stops(json!({"eos_token_id": []}), None).is_err());
        assert!(stops(json!({}), Some(json!({"eos_token_id": null}))).is_err());
        assert!(stops(json!({"eos_token_id": -1}), None).is_err());
        assert!(stops(json!({"eos_token_id": "2"}), None).is_err());

        let tool_turns = StopTokens {
            end_turn: NonEmpty::from_vec(vec![TokenId::new(1)]).expect("nonempty"),
            tool_end: vec![TokenId::new(50)],
        };
        assert_eq!(tool_turns.classify(TokenId::new(50)), TokenClass::ToolEnd);
        assert_eq!(tool_turns.classify(TokenId::new(1)), TokenClass::EndTurn);
    }

    #[test]
    fn listed_eos_ids_load_and_must_be_tokenizer_tokens() {
        let format = |config: Value, generation: Option<Value>| {
            let model = ModelDir::new(
                &json!({"chat_template": "{{ messages[0].content }}"}),
                &config,
                generation.as_ref(),
            );
            ChatFormat::load(model.path(), VOCABULARY_SIZE)
        };
        let loaded = format(
            json!({"eos_token_id": [2, 3]}),
            Some(json!({"eos_token_id": 1})),
        )
        .expect("listed stops load");
        assert_eq!(
            loaded.stops().iter().map(TokenId::get).collect::<Vec<_>>(),
            [2, 3, 1]
        );
        // Outside the logit width, or a row with no tokenizer spelling.
        assert!(format(json!({"eos_token_id": [2, 8]}), None).is_err());
        assert!(format(json!({"eos_token_id": [2, 7]}), None).is_err());
    }

    #[test]
    fn external_chat_template_is_accepted_when_config_has_none() {
        let model = ModelDir::new(&json!({}), &json!({}), None);
        fs::write(model.path().join("chat_template.jinja"), b"{{ messages }}")
            .expect("external template");
        assert_eq!(
            load_template(model.path(), &json!({})).expect("external template"),
            "{{ messages }}"
        );
    }

    #[test]
    fn template_sources_reject_oversized_invalid_utf8_and_nonregular_files() {
        let model = ModelDir::new(&json!({}), &json!({}), None);
        let root = model.path();
        let config = root.join("tokenizer_config.json");
        let external = root.join("chat_template.jinja");
        let oversized = vec![b'x'; MAX_CHAT_TEMPLATE_BYTES + 1];
        let read_config = || read_json(root, "tokenizer_config.json", MAX_CHAT_TEMPLATE_BYTES);

        fs::write(&config, &oversized).expect("oversized tokenizer config");
        assert!(
            read_config()
                .expect_err("oversized tokenizer config must fail")
                .contains("byte limit")
        );
        fs::write(&config, [0xff]).expect("invalid UTF-8 tokenizer config");
        assert!(
            read_config()
                .expect_err("invalid UTF-8 tokenizer config must fail")
                .contains("could not be read")
        );
        fs::remove_file(&config).expect("remove tokenizer config file");
        fs::create_dir(&config).expect("nonregular tokenizer config");
        assert!(
            read_config()
                .expect_err("nonregular tokenizer config must fail")
                .contains("readable regular file")
        );
        fs::remove_dir(&config).expect("remove tokenizer config directory");

        fs::write(&external, &oversized).expect("oversized external template");
        assert!(
            load_template(root, &json!({}))
                .expect_err("oversized external template must fail")
                .contains("byte limit")
        );
        fs::write(&external, [0xff]).expect("invalid UTF-8 external template");
        assert!(
            load_template(root, &json!({}))
                .expect_err("invalid UTF-8 external template must fail")
                .contains("could not be read")
        );
        fs::remove_file(&external).expect("remove external template file");
        fs::create_dir(&external).expect("nonregular external template");
        assert!(
            load_template(root, &json!({}))
                .expect_err("nonregular external template must fail")
                .contains("readable regular file")
        );
    }
}
