//! The Qwen3-ASR tokenizer and prompt.
//!
//! The released checkpoints ship `vocab.json`, `merges.txt` and
//! `tokenizer_config.json` but no `tokenizer.json`. This rebuilds the fast
//! tokenizer transformers derives from them (`Qwen2Converter`: byte-level BPE
//! behind an NFC normalizer and the Qwen2 split regex), with the added tokens
//! at the ids `tokenizer_config.json` assigns.

use std::fs;
use std::path::Path;

use serde_json::{Map, Value, json};
use tokenizers::Tokenizer;

use crate::config::AudioTokens;

/// Largest vocabulary, merges or tokenizer-config file read.
const MAX_TOKENIZER_FILE_BYTES: u64 = 64 << 20;

/// transformers' `Qwen2Converter` pre-tokenizer pattern.
const QWEN2_SPLIT: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

/// End-of-turn tokens that stop generation (`generation_config.json`).
pub const STOP_TOKENS: [i32; 2] = [151_643, 151_645];

/// A tokenizer bound to one checkpoint's audio tokens.
pub struct AsrTokenizer {
    tokenizer: Tokenizer,
    audio: AudioTokens,
}

impl std::fmt::Debug for AsrTokenizer {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("AsrTokenizer")
            .field("audio", &self.audio)
            .finish_non_exhaustive()
    }
}

impl AsrTokenizer {
    /// Builds the tokenizer from a checkpoint directory.
    ///
    /// # Errors
    ///
    /// Returns [`AsrTokenizerError`] when a file is missing or malformed, or
    /// an added token or audio token lands on a different id than the
    /// checkpoint declares.
    pub fn load(model_dir: &Path, audio: AudioTokens) -> Result<Self, AsrTokenizerError> {
        let vocab: Value = serde_json::from_str(&read(&model_dir.join("vocab.json"))?)?;
        let merges: Vec<Value> = read(&model_dir.join("merges.txt"))?
            .lines()
            .filter(|line| !line.starts_with("#version") && !line.is_empty())
            .map(|line| {
                line.split_once(' ')
                    .map(|(left, right)| json!([left, right]))
                    .ok_or_else(|| AsrTokenizerError::Merge(line.to_owned()))
            })
            .collect::<Result<_, _>>()?;
        let config: Value = serde_json::from_str(&read(&model_dir.join("tokenizer_config.json"))?)?;
        let declared = config
            .get("added_tokens_decoder")
            .and_then(Value::as_object)
            .ok_or(AsrTokenizerError::AddedTokens)?;
        let added = added_tokens(declared)?;

        let document = json!({
            "version": "1.0",
            "truncation": null,
            "padding": null,
            "added_tokens": added,
            "normalizer": {"type": "NFC"},
            "pre_tokenizer": {"type": "Sequence", "pretokenizers": [
                {"type": "Split", "pattern": {"Regex": QWEN2_SPLIT},
                 "behavior": "Isolated", "invert": false},
                {"type": "ByteLevel", "add_prefix_space": false,
                 "trim_offsets": false, "use_regex": false}
            ]},
            "post_processor": {"type": "ByteLevel", "add_prefix_space": false,
                "trim_offsets": false, "use_regex": false},
            "decoder": {"type": "ByteLevel", "add_prefix_space": false,
                "trim_offsets": false, "use_regex": false},
            "model": {"type": "BPE", "dropout": null, "unk_token": null,
                "continuing_subword_prefix": "", "end_of_word_suffix": "",
                "fuse_unk": false, "byte_fallback": false, "ignore_merges": false,
                "vocab": vocab, "merges": merges}
        });
        let tokenizer = Tokenizer::from_bytes(serde_json::to_vec(&document)?)
            .map_err(|error| AsrTokenizerError::Build(error.to_string()))?;
        for token in &added {
            let content = token["content"].as_str().unwrap_or_default();
            let expected = token["id"].as_u64().and_then(|id| u32::try_from(id).ok());
            if tokenizer.token_to_id(content) != expected {
                return Err(AsrTokenizerError::TokenId(content.to_owned()));
            }
        }
        for (name, id) in [
            ("<|audio_start|>", audio.start),
            ("<|audio_end|>", audio.end),
            ("<|audio_pad|>", audio.pad),
        ] {
            if tokenizer
                .token_to_id(name)
                .and_then(|id| i32::try_from(id).ok())
                != Some(id)
            {
                return Err(AsrTokenizerError::TokenId(name.to_owned()));
            }
        }
        Ok(Self { tokenizer, audio })
    }

    /// Prompt ids for one clip: the checkpoint's chat template with
    /// `context` as the system message and `audio_frames` audio placeholders,
    /// then `language X<asr_text>` when a language is forced.
    ///
    /// # Errors
    ///
    /// Returns [`AsrTokenizerError::Encode`] if tokenization fails.
    pub fn prompt(
        &self,
        context: &str,
        forced_language: Option<&str>,
        audio_frames: usize,
    ) -> Result<AsrPrompt, AsrTokenizerError> {
        let mut text = format!(
            "<|im_start|>system\n{context}<|im_end|>\n<|im_start|>user\n\
             <|audio_start|><|audio_pad|><|audio_end|><|im_end|>\n<|im_start|>assistant\n"
        );
        if let Some(language) = forced_language {
            text.push_str("language ");
            text.push_str(language);
            text.push_str("<asr_text>");
        }
        let encoded = self
            .tokenizer
            .encode(text, true)
            .map_err(|error| AsrTokenizerError::Encode(error.to_string()))?;
        let ids: Vec<i32> = encoded
            .get_ids()
            .iter()
            .map(|&id| i32::try_from(id).map_err(|_| AsrTokenizerError::Encode(id.to_string())))
            .collect::<Result<_, _>>()?;
        // A context containing the placeholder text must not add audio slots.
        let pads: Vec<usize> = ids
            .iter()
            .enumerate()
            .filter(|&(_, &id)| id == self.audio.pad)
            .map(|(index, _)| index)
            .collect();
        let [audio_start] = pads[..] else {
            return Err(AsrTokenizerError::Placeholders(pads.len()));
        };
        let mut expanded = Vec::with_capacity(ids.len() + audio_frames);
        expanded.extend_from_slice(&ids[..audio_start]);
        expanded.extend(std::iter::repeat_n(self.audio.pad, audio_frames));
        expanded.extend_from_slice(&ids[audio_start + 1..]);
        Ok(AsrPrompt {
            ids: expanded,
            audio_start,
            audio_frames,
        })
    }

    /// Decodes generated ids, dropping special tokens (`<asr_text>` is not
    /// special, so it survives for [`crate::output::parse_output`]).
    ///
    /// # Errors
    ///
    /// Returns [`AsrTokenizerError::Decode`] for an id outside the vocabulary.
    pub fn decode(&self, ids: &[i32]) -> Result<String, AsrTokenizerError> {
        let ids: Vec<u32> = ids
            .iter()
            .map(|&id| u32::try_from(id).map_err(|_| AsrTokenizerError::Decode(id.to_string())))
            .collect::<Result<_, _>>()?;
        self.tokenizer
            .decode(&ids, true)
            .map_err(|error| AsrTokenizerError::Decode(error.to_string()))
    }
}

/// Token ids of one prompt and where its audio placeholders sit.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AsrPrompt {
    ids: Vec<i32>,
    audio_start: usize,
    audio_frames: usize,
}

impl AsrPrompt {
    /// Every prompt id, placeholders included.
    #[must_use]
    pub fn ids(&self) -> &[i32] {
        &self.ids
    }

    /// Index of the first placeholder.
    #[must_use]
    pub const fn audio_start(&self) -> usize {
        self.audio_start
    }

    /// Number of placeholders, one per encoder frame.
    #[must_use]
    pub const fn audio_frames(&self) -> usize {
        self.audio_frames
    }
}

fn added_tokens(declared: &Map<String, Value>) -> Result<Vec<Value>, AsrTokenizerError> {
    let mut tokens = declared
        .iter()
        .map(|(id, token)| {
            let id: u64 = id.parse().map_err(|_| AsrTokenizerError::AddedTokens)?;
            let field = |name: &str| token.get(name).and_then(Value::as_bool).unwrap_or(false);
            let content = token
                .get("content")
                .and_then(Value::as_str)
                .ok_or(AsrTokenizerError::AddedTokens)?;
            Ok(json!({
                "id": id,
                "content": content,
                "single_word": field("single_word"),
                "lstrip": field("lstrip"),
                "rstrip": field("rstrip"),
                "normalized": field("normalized"),
                "special": field("special"),
            }))
        })
        .collect::<Result<Vec<_>, AsrTokenizerError>>()?;
    tokens.sort_by_key(|token| token["id"].as_u64());
    Ok(tokens)
}

fn read(path: &Path) -> Result<String, AsrTokenizerError> {
    let length = fs::metadata(path)
        .map_err(|error| AsrTokenizerError::Read(format!("{}: {error}", path.display())))?
        .len();
    if length > MAX_TOKENIZER_FILE_BYTES {
        return Err(AsrTokenizerError::Read(format!(
            "{} is {length} bytes",
            path.display()
        )));
    }
    fs::read_to_string(path)
        .map_err(|error| AsrTokenizerError::Read(format!("{}: {error}", path.display())))
}

/// Errors while building or using the tokenizer.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum AsrTokenizerError {
    /// A tokenizer file could not be read or is too large.
    #[error("tokenizer file: {0}")]
    Read(String),
    /// A tokenizer file is not valid JSON.
    #[error("tokenizer JSON: {0}")]
    Json(#[from] serde_json::Error),
    /// A `merges.txt` line is not two space-separated symbols.
    #[error("malformed merge {0:?}")]
    Merge(String),
    /// `added_tokens_decoder` is missing or malformed.
    #[error("tokenizer_config.json has no valid added_tokens_decoder")]
    AddedTokens,
    /// The tokenizers library refused the assembled tokenizer.
    #[error("tokenizer build failed: {0}")]
    Build(String),
    /// A declared token did not resolve to its declared id.
    #[error("token {0:?} does not have its declared id")]
    TokenId(String),
    /// Encoding failed.
    #[error("tokenizer encode failed: {0}")]
    Encode(String),
    /// Decoding failed.
    #[error("tokenizer decode failed: {0}")]
    Decode(String),
    /// The rendered prompt holds other than exactly one audio placeholder.
    #[error("prompt has {0} audio placeholders; expected 1")]
    Placeholders(usize),
}
