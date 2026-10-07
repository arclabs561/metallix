//! Prompt ids and decoding against the qwen-asr processor. The fixture comes
//! from `scripts/qwen3-asr-reference.py prompt-fixture`; the tokenizer files
//! come from a local checkpoint named by `METALLIX_QWEN3_ASR_MODEL`.

use std::path::PathBuf;

use qwen3_asr::{AsrTokenizer, Qwen3AsrConfig, lengths};
use serde::Deserialize;

#[derive(Deserialize)]
struct Fixture {
    prompts: Vec<Prompt>,
    decodes: Vec<Decode>,
}

#[derive(Deserialize)]
struct Prompt {
    context: String,
    language: Option<String>,
    samples: usize,
    mel_frames: usize,
    ids: Vec<i32>,
}

#[derive(Deserialize)]
struct Decode {
    ids: Vec<i32>,
    text: String,
}

#[test]
#[ignore = "needs a Qwen3-ASR checkpoint in METALLIX_QWEN3_ASR_MODEL"]
fn prompts_and_decoding_match_the_reference_processor() {
    let model = PathBuf::from(
        std::env::var_os("METALLIX_QWEN3_ASR_MODEL")
            .expect("set METALLIX_QWEN3_ASR_MODEL to a Qwen3-ASR checkpoint"),
    );
    let config =
        Qwen3AsrConfig::parse(&std::fs::read_to_string(model.join("config.json")).unwrap())
            .unwrap();
    let tokenizer = AsrTokenizer::load(&model, config.tokens()).unwrap();
    let fixture: Fixture =
        serde_json::from_str(include_str!("fixtures/qwen3_asr_prompts.json")).unwrap();
    assert_eq!(fixture.prompts.len(), 4);
    for case in fixture.prompts {
        assert_eq!(
            audio::LogMelSpec::QWEN3_ASR.frames_for(case.samples),
            case.mel_frames
        );
        let frames = lengths::encoder_frames(case.mel_frames);
        let prompt = tokenizer
            .prompt(&case.context, case.language.as_deref(), frames)
            .unwrap();
        assert_eq!(prompt.ids(), case.ids, "context {:?}", case.context);
        let pads = case
            .ids
            .iter()
            .filter(|&&id| id == config.tokens().pad)
            .count();
        assert_eq!(pads, frames);
        assert_eq!(case.ids[prompt.audio_start() - 1], config.tokens().start);
        assert_eq!(case.ids[prompt.audio_start() + frames], config.tokens().end);
    }
    for case in fixture.decodes {
        assert_eq!(tokenizer.decode(&case.ids).unwrap(), case.text);
    }
}
