#![cfg(feature = "structured-output")]

use std::{
    env, fs,
    path::{Path, PathBuf},
    sync::Arc,
};

use llguidance::{
    Matcher, ParserFactory,
    api::TopLevelGrammar,
    token_bytes_from_tokenizer_json,
    toktrie::{ApproximateTokEnv, TokEnv, TokRxInfo, TokTrie},
};
use serde_json::{Value, json};

const QWEN_TOKENIZER_ENV: &str = "METALLIX_QWEN_TOKENIZER";
const VALID_JSON: &str = r#"{"ok":true}"#;
const INVALID_JSON: &str = r#"{"ok":false}"#;

fn schema() -> TopLevelGrammar {
    TopLevelGrammar::from_json_schema(json!({
        "type": "object",
        "properties": {"ok": {"const": true}},
        "required": ["ok"],
        "additionalProperties": false,
    }))
}

fn matcher_for(env: &TokEnv) -> Result<Matcher, Box<dyn std::error::Error>> {
    let factory = ParserFactory::new_simple(env)?;
    Ok(Matcher::new(factory.create_parser(schema())))
}

fn replay_valid(matcher: &mut Matcher, tokens: &[u32]) -> Result<(), Box<dyn std::error::Error>> {
    for (index, &token) in tokens.iter().enumerate() {
        assert!(
            !matcher.is_stopped(),
            "grammar stopped before valid token {index}"
        );
        assert!(
            matcher.compute_mask()?.is_allowed(token),
            "grammar rejected valid token {token} at index {index}"
        );
        matcher.consume_token(token)?;
    }
    assert!(matcher.is_stopped(), "grammar did not complete valid JSON");
    Ok(())
}

fn assert_invalid_continuation_rejected(
    matcher: &mut Matcher,
    tokens: &[u32],
) -> Result<(), Box<dyn std::error::Error>> {
    for &token in tokens {
        if matcher.is_stopped() || !matcher.compute_mask()?.is_allowed(token) {
            return Ok(());
        }
        matcher.consume_token(token)?;
    }
    Err("grammar accepted the complete invalid continuation".into())
}

fn assert_reconstructed_json(
    env: &TokEnv,
    tokens: &[u32],
) -> Result<(), Box<dyn std::error::Error>> {
    let bytes = env.tok_trie().decode(tokens);
    assert_eq!(
        bytes,
        VALID_JSON.as_bytes(),
        "token IDs reconstructed different bytes"
    );
    let value: Value = serde_json::from_slice(&bytes)?;
    assert_eq!(value["ok"], Value::Bool(true));
    Ok(())
}

fn qwen_trie(tokenizer_path: &Path) -> Result<(TokEnv, u32), Box<dyn std::error::Error>> {
    let tokenizer: Value = serde_json::from_slice(&fs::read(tokenizer_path)?)?;
    let token_bytes = token_bytes_from_tokenizer_json(&tokenizer)?;
    let config_path = tokenizer_path
        .parent()
        .ok_or("tokenizer path has no parent directory")?
        .join("config.json");
    let config: Value = serde_json::from_slice(&fs::read(config_path)?)?;
    let eos = config["eos_token_id"]
        .as_u64()
        .ok_or("Qwen config must contain a numeric eos_token_id")?;
    let eos = u32::try_from(eos)?;
    let vocab_size = u32::try_from(token_bytes.len())?;
    assert!(eos < vocab_size, "EOS ID is outside tokenizer vocabulary");
    let info = TokRxInfo {
        vocab_size,
        tok_eos: eos,
        tok_bos: None,
        tok_pad: None,
        tok_unk: None,
        tok_end_of_turn: None,
    };
    let env: TokEnv = Arc::new(ApproximateTokEnv::new(TokTrie::from(&info, &token_bytes)));
    Ok((env, eos))
}

#[test]
fn single_byte_schema_rejects_invalid_continuation_and_allows_eos_when_complete()
-> Result<(), Box<dyn std::error::Error>> {
    let env = ApproximateTokEnv::single_byte_env();
    let valid = VALID_JSON.bytes().map(u32::from).collect::<Vec<_>>();
    let invalid = INVALID_JSON.bytes().map(u32::from).collect::<Vec<_>>();
    let mut matcher = matcher_for(&env)?;
    assert!(!matcher.compute_mask()?.is_allowed(env.eos_token()));
    replay_valid(&mut matcher, &valid)?;
    assert_reconstructed_json(&env, &valid)?;
    assert_invalid_continuation_rejected(&mut matcher_for(&env)?, &invalid)?;

    let mut matcher = matcher_for(&env)?;
    replay_valid(&mut matcher, &valid)?;
    assert!(matcher.compute_mask_or_eos()?.is_allowed(env.eos_token()));
    Ok(())
}

#[test]
#[ignore = "requires METALLIX_QWEN_TOKENIZER=/path/to/Qwen3-0.6B/tokenizer.json"]
fn qwen_tokenizer_trie_masks_json_without_claiming_canonical_encoding()
-> Result<(), Box<dyn std::error::Error>> {
    let tokenizer_path = PathBuf::from(env::var(QWEN_TOKENIZER_ENV)?);
    let (env, eos) = qwen_trie(&tokenizer_path)?;
    let valid = env.tokenize(VALID_JSON);
    let invalid = env.tokenize(INVALID_JSON);

    let mut matcher = matcher_for(&env)?;
    assert!(!matcher.compute_mask()?.is_allowed(eos));
    replay_valid(&mut matcher, &valid)?;
    assert_reconstructed_json(&env, &valid)?;
    assert_invalid_continuation_rejected(&mut matcher_for(&env)?, &invalid)?;
    let mut matcher = matcher_for(&env)?;
    replay_valid(&mut matcher, &valid)?;
    assert!(matcher.compute_mask_or_eos()?.is_allowed(eos));

    // ApproximateTokEnv greedily tokenizes trie bytes. It qualifies that the
    // local vocabulary can mask this schema, not Hugging Face canonical BPE
    // encoding or fast-forward-token correctness.
    Ok(())
}
