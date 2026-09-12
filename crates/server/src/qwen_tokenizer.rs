//! Bounded local tokenizer access for Qwen prompt and result text.

use std::{fs::File, io::Read, path::Path};

use tokenizers::Tokenizer;

const MAX_TOKENIZER_BYTES: usize = 64 * 1024 * 1024;
const MAX_PROMPT_BYTES: usize = 1024 * 1024;

/// A locally loaded tokenizer whose checkpoint-configured padding and
/// truncation policies have been explicitly disabled for CLI generation.
pub(crate) struct QwenTokenizer {
    tokenizer: Tokenizer,
}

impl QwenTokenizer {
    pub(crate) fn load(model: &Path) -> Result<Self, String> {
        let bytes = read_regular_file(&model.join("tokenizer.json"), MAX_TOKENIZER_BYTES)?;
        Self::from_bytes(bytes)
    }

    fn from_bytes(bytes: Vec<u8>) -> Result<Self, String> {
        let mut tokenizer = Tokenizer::from_bytes(bytes)
            .map_err(|_| String::from("local tokenizer.json could not be parsed"))?;
        tokenizer
            .with_truncation(None)
            .map_err(|_| String::from("local tokenizer could not disable truncation"))?;
        tokenizer.with_padding(None);
        Ok(Self { tokenizer })
    }

    /// Checks representable tokenizer IDs and EOS without requiring padded
    /// model-logit rows to have token spellings. This does not prove that a
    /// tokenizer belongs to a particular checkpoint.
    pub(crate) fn check_model_vocabulary(&self, width: usize, eos: i32) -> Result<(), String> {
        let vocabulary = self.tokenizer.get_vocab(true);
        if vocabulary.is_empty()
            || vocabulary
                .values()
                .any(|&id| usize::try_from(id).ok().is_none_or(|id| id >= width))
        {
            return Err(String::from("local tokenizer IDs exceed model vocabulary"));
        }
        let eos = u32::try_from(eos).map_err(|_| String::from("model EOS token ID is negative"))?;
        if self.tokenizer.id_to_token(eos).is_none() {
            return Err(String::from(
                "model EOS token is absent from local tokenizer",
            ));
        }
        Ok(())
    }

    /// Encodes prompt bytes exactly as supplied: no chat template or special
    /// tokens are added by this CLI layer.
    pub(crate) fn encode_prompt(&self, prompt: &str) -> Result<Vec<i32>, String> {
        if prompt.is_empty() {
            return Err(String::from("prompt must not be empty"));
        }
        if prompt.len() > MAX_PROMPT_BYTES {
            return Err(format!("prompt exceeds the {MAX_PROMPT_BYTES}-byte limit"));
        }
        let encoding = self
            .tokenizer
            .encode(prompt, false)
            .map_err(|_| String::from("prompt could not be encoded by local tokenizer"))?;
        if encoding.get_ids().is_empty() {
            return Err(String::from("prompt produced no token IDs"));
        }
        encoding
            .get_ids()
            .iter()
            .copied()
            .map(|token_id| {
                i32::try_from(token_id)
                    .map_err(|_| String::from("tokenizer token ID does not fit server token IDs"))
            })
            .collect()
    }

    /// Decodes every generated ID, rejecting absent or unsigned-incompatible
    /// IDs rather than quietly omitting output from the JSON receipt.
    pub(crate) fn decode_generated(&self, token_ids: &[i32]) -> Result<String, String> {
        let token_ids = token_ids
            .iter()
            .copied()
            .map(|token_id| {
                let token_id = u32::try_from(token_id)
                    .map_err(|_| String::from("generated token ID is negative"))?;
                self.tokenizer.id_to_token(token_id).ok_or_else(|| {
                    String::from("generated token ID is absent from local tokenizer")
                })?;
                Ok(token_id)
            })
            .collect::<Result<Vec<_>, String>>()?;
        self.tokenizer.decode(&token_ids, false).map_err(|_| {
            String::from("generated token IDs could not be decoded by local tokenizer")
        })
    }
}

/// Reads one local regular tokenizer file with a post-open bound check, keeping
/// a replacement from turning the metadata size into an unbounded allocation.
fn read_regular_file(path: &Path, maximum_bytes: usize) -> Result<Vec<u8>, String> {
    let metadata = path
        .metadata()
        .map_err(|_| String::from("local tokenizer.json must be a readable regular file"))?;
    if !metadata.is_file() {
        return Err(String::from(
            "local tokenizer.json must be a readable regular file",
        ));
    }
    let maximum_bytes_u64 = u64::try_from(maximum_bytes)
        .map_err(|_| String::from("tokenizer byte limit is unsupported on this platform"))?;
    if metadata.len() > maximum_bytes_u64 {
        return Err(format!(
            "local tokenizer.json exceeds the {maximum_bytes}-byte limit"
        ));
    }

    let file = File::open(path)
        .map_err(|_| String::from("local tokenizer.json must be a readable regular file"))?;
    if !file
        .metadata()
        .map_err(|_| String::from("local tokenizer.json must be a readable regular file"))?
        .is_file()
    {
        return Err(String::from(
            "local tokenizer.json must be a readable regular file",
        ));
    }
    let capacity = usize::try_from(metadata.len())
        .map_err(|_| String::from("local tokenizer.json size does not fit this platform"))?;
    let mut bytes = Vec::with_capacity(capacity);
    file.take(maximum_bytes_u64.saturating_add(1))
        .read_to_end(&mut bytes)
        .map_err(|_| String::from("local tokenizer.json could not be read"))?;
    if bytes.len() > maximum_bytes {
        return Err(format!(
            "local tokenizer.json exceeds the {maximum_bytes}-byte limit"
        ));
    }
    Ok(bytes)
}

#[cfg(test)]
mod tests {
    use super::QwenTokenizer;

    const TOKENIZER_JSON: &[u8] = br###"{
      "version": "1.0",
      "truncation": {"max_length": 1, "stride": 0, "strategy": "LongestFirst", "direction": "Right"},
      "padding": {"strategy": {"Fixed": 4}, "direction": "Right", "pad_to_multiple_of": null, "pad_id": 0, "pad_type_id": 0, "pad_token": "[PAD]"},
      "added_tokens": [],
      "normalizer": null,
      "pre_tokenizer": {"type": "Whitespace"},
      "post_processor": null,
      "decoder": {"type": "WordPiece", "prefix": "##", "cleanup": true},
      "model": {"type": "WordPiece", "unk_token": "[UNK]", "continuing_subword_prefix": "##", "max_input_chars_per_word": 100, "vocab": {"[PAD]": 0, "[UNK]": 1, "hello": 2, "world": 3}}
    }"###;

    fn tokenizer() -> QwenTokenizer {
        QwenTokenizer::from_bytes(TOKENIZER_JSON.to_vec()).expect("small test tokenizer")
    }

    #[test]
    fn local_tokenizer_rejects_malformed_nonregular_and_oversized_inputs() {
        assert!(QwenTokenizer::from_bytes(b"not tokenizer JSON".to_vec()).is_err());
        let directory = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
        assert!(super::read_regular_file(directory, 1024).is_err());
        assert!(
            super::read_regular_file(&directory.join("Cargo.toml"), 1)
                .unwrap_err()
                .contains("byte limit")
        );
    }

    #[test]
    fn prompt_encoding_clears_checkpoint_padding_and_truncation() {
        assert_eq!(tokenizer().encode_prompt("hello world"), Ok(vec![2, 3]));
    }

    #[test]
    fn decoded_output_rejects_invalid_ids() {
        let tokenizer = tokenizer();
        assert_eq!(
            tokenizer.decode_generated(&[2, 3]),
            Ok(String::from("hello world"))
        );
        assert!(tokenizer.decode_generated(&[-1]).is_err());
        assert!(tokenizer.decode_generated(&[9]).is_err());
    }

    #[test]
    fn prompt_has_an_explicit_byte_limit_before_tokenization() {
        let prompt = "x".repeat(super::MAX_PROMPT_BYTES + 1);
        assert_eq!(
            tokenizer().encode_prompt(&prompt),
            Err(format!(
                "prompt exceeds the {}-byte limit",
                super::MAX_PROMPT_BYTES
            ))
        );
    }

    #[test]
    fn model_vocabulary_allows_padding_but_rejects_missing_eos_and_out_of_range_ids() {
        let tokenizer = tokenizer();
        assert!(tokenizer.check_model_vocabulary(8, 1).is_ok());
        assert!(tokenizer.check_model_vocabulary(3, 1).is_err());
        assert!(tokenizer.check_model_vocabulary(8, 7).is_err());
        assert!(tokenizer.check_model_vocabulary(8, -1).is_err());
        assert_eq!(
            tokenizer.encode_prompt(""),
            Err(String::from("prompt must not be empty"))
        );
    }
}
