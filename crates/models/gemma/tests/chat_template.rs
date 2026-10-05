//! Chat-template and tokenizer parity against `fixtures/gemma-4-12b/`,
//! written by `scripts/gemma4-reference.py`.
//!
//! Renders the checkpoint's `chat_template.jinja` the way the server renders
//! chat templates (minijinja with Python-method compatibility) and compares
//! the text with the source's `apply_chat_template`. The tokenizer checks need
//! the checkpoint: set `METALLIX_GEMMA4_MODEL` and run with `--ignored`.

use std::{env, fs, path::PathBuf};

use gemma::Gemma4GenerationConfig;
use minijinja::{Environment, context};
use serde::Deserialize;
use serde_json::Value;
use tokenizers::Tokenizer;

const TEMPLATE: &str = include_str!("../../../../fixtures/gemma-4-12b/chat-template.jinja");
const REFERENCE: &str = include_str!("../../../../fixtures/gemma-4-12b/reference.json");

#[derive(Deserialize)]
struct Manifest {
    templates: Vec<TemplateCase>,
}

#[derive(Deserialize)]
struct TemplateCase {
    name: String,
    messages: Value,
    tools: Value,
    enable_thinking: bool,
    rendered: String,
    input_ids: Vec<u32>,
}

fn manifest() -> Manifest {
    serde_json::from_str(REFERENCE).expect("reference JSON")
}

fn model() -> Option<PathBuf> {
    let model = env::var_os("METALLIX_GEMMA4_MODEL").map(PathBuf::from);
    if model.is_none() {
        eprintln!("skipping: METALLIX_GEMMA4_MODEL is not set");
    }
    model
}

/// The server's template context keys, plus `bos_token`, which Gemma's
/// template prints and the server does not pass today.
fn render(case: &TemplateCase, bos_token: Option<&str>) -> String {
    let mut environment = Environment::new();
    environment.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
    environment
        .add_template("chat", TEMPLATE)
        .expect("template parses");
    let server_context = context! {
        messages => &case.messages,
        tools => &case.tools,
        add_generation_prompt => true,
        enable_thinking => case.enable_thinking,
    };
    // Absent, as in the server's context today, rather than a `none` value.
    let context = match bos_token {
        Some(bos_token) => context! { bos_token, ..server_context },
        None => server_context,
    };
    environment
        .get_template("chat")
        .expect("template")
        .render(context)
        .expect("template renders")
}

#[test]
fn template_renders_like_the_source_when_bos_token_is_in_the_context() {
    for case in &manifest().templates {
        assert_eq!(render(case, Some("<bos>")), case.rendered, "{}", case.name);
    }
}

/// The template's first output is `{{ bos_token }}`; without it in the
/// context the prompt silently starts at `<|turn>`.
#[test]
fn rendering_without_bos_token_drops_the_bos() {
    for case in &manifest().templates {
        assert_eq!(
            Some(render(case, None).as_str()),
            case.rendered.strip_prefix("<bos>"),
            "{}",
            case.name
        );
    }
}

#[test]
#[ignore = "requires METALLIX_GEMMA4_MODEL (google/gemma-4-12B-it@707f0a3)"]
fn tokenizer_matches_the_source_and_adds_no_bos_itself() {
    let Some(model) = model() else { return };
    let tokenizer = Tokenizer::from_file(model.join("tokenizer.json")).expect("tokenizer");
    for case in &manifest().templates {
        // The rendered text carries <bos> itself, so no special tokens are added.
        let encoding = tokenizer
            .encode(case.rendered.as_str(), false)
            .expect("encode");
        assert_eq!(
            encoding.get_ids(),
            case.input_ids,
            "{}: token IDs",
            case.name
        );
        assert_eq!(encoding.get_ids().first(), Some(&2), "{}: <bos>", case.name);
        // The post-processor adds nothing, so special tokens cannot restore a
        // <bos> the template context left out.
        let without = render(case, None);
        let encoding = tokenizer.encode(without.as_str(), true).expect("encode");
        assert_eq!(
            encoding.get_ids().first(),
            Some(&105),
            "{}: <|turn>",
            case.name
        );
    }
}

#[test]
#[ignore = "requires METALLIX_GEMMA4_MODEL (google/gemma-4-12B-it@707f0a3)"]
fn every_stop_token_is_a_turn_or_tool_boundary() {
    let Some(model) = model() else { return };
    let tokenizer = Tokenizer::from_file(model.join("tokenizer.json")).expect("tokenizer");
    let generation = Gemma4GenerationConfig::parse(
        &fs::read_to_string(model.join("generation_config.json")).expect("generation config"),
    )
    .expect("generation config");
    let spell = |ids: &[i32]| -> Vec<String> {
        ids.iter()
            .map(|&id| {
                tokenizer
                    .id_to_token(u32::try_from(id).expect("nonnegative"))
                    .expect("token in vocabulary")
            })
            .collect()
    };
    // <|tool_response> ends a model turn that issued tool calls.
    assert_eq!(
        spell(&generation.eos_token_ids),
        ["<eos>", "<turn|>", "<|tool_response>"]
    );
    assert_eq!(
        spell(&generation.suppress_token_ids),
        ["<audio|>", "<image|>"]
    );
}
