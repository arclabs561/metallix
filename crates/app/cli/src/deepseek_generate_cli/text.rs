//! Local plaintext preparation; the existing tokenizer applies no chat template.

use std::path::Path;

use chat_format::{QwenTokenizer, StopTokens};

pub(super) struct TextInput {
    pub(super) ids: Vec<u32>,
    pub(super) stops: Vec<u32>,
    tokenizer: QwenTokenizer,
}

impl TextInput {
    pub(super) fn load(directory: &Path, prompt: &str) -> Result<Self, String> {
        // Reject user input before opening tokenizer or checkpoint metadata.
        if prompt.is_empty() || prompt.len() > 1024 * 1024 {
            return Err("prompt must contain 1..=1048576 UTF-8 bytes".to_owned());
        }
        let tokenizer = QwenTokenizer::load(directory)?;
        let ids = tokenizer
            .encode_prompt(prompt)?
            .into_iter()
            .map(|id| u32::try_from(id).map_err(|_| "negative tokenizer ID".to_owned()))
            .collect::<Result<Vec<_>, _>>()?;
        let stops = StopTokens::load(directory)?
            .iter()
            .map(chat_format::TokenId::get)
            .collect();
        Ok(Self {
            ids,
            stops,
            tokenizer,
        })
    }

    pub(super) fn validate_vocabulary(&self, width: usize, stops: &[u32]) -> Result<(), String> {
        for &id in stops {
            let id =
                i32::try_from(id).map_err(|_| "EOS token ID exceeds tokenizer representation")?;
            self.tokenizer.check_model_vocabulary(width, id)?;
        }
        Ok(())
    }

    pub(super) fn decode(&self, ids: &[u32]) -> Result<String, String> {
        let ids = ids
            .iter()
            .map(|&id| {
                i32::try_from(id)
                    .map_err(|_| "generated ID exceeds tokenizer representation".to_owned())
            })
            .collect::<Result<Vec<_>, _>>()?;
        self.tokenizer.decode_generated(&ids)
    }

    pub(super) fn sha256(&self) -> &str {
        self.tokenizer.source_sha256()
    }
}

#[cfg(test)]
mod tests {
    use super::TextInput;
    use chat_format::test_model::{ModelDir, VOCABULARY_SIZE};
    use serde_json::json;

    #[test]
    fn plaintext_does_not_add_bos_or_render_template_and_uses_checkpoint_eos() {
        let model = ModelDir::new(
            &json!({"chat_template":"ignored", "bos_token":"<s>", "eos_token":"</s>"}),
            &json!({"eos_token_id":2}),
            None,
        );
        let input = TextInput::load(model.path(), "Hello hi").expect("local plaintext");
        assert_eq!(input.ids, [4, 5]);
        assert_eq!(input.stops, [2]);
        assert_eq!(input.decode(&[5, 4]).expect("decode"), "hi Hello");
        input
            .validate_vocabulary(VOCABULARY_SIZE, &input.stops)
            .expect("vocabulary");
        assert!(input.validate_vocabulary(4, &input.stops).is_err());
        assert!(input.validate_vocabulary(VOCABULARY_SIZE, &[7]).is_err());
        assert!(input.decode(&[7]).is_err());
        assert_eq!(input.sha256().len(), 64);
    }

    #[test]
    fn invalid_text_is_refused_before_missing_assets() {
        assert!(
            TextInput::load(std::path::Path::new("missing"), "")
                .err()
                .expect("empty")
                .contains("UTF-8 bytes")
        );
        assert!(
            TextInput::load(
                std::path::Path::new("missing"),
                &"x".repeat(1024 * 1024 + 1)
            )
            .err()
            .expect("oversized")
            .contains("UTF-8 bytes")
        );
    }

    #[test]
    #[ignore = "requires METALLIX_DEEPSEEK_TOKENIZER_ASSETS with pinned assets and plaintext-reference.json"]
    fn pinned_plaintext_assets_match_reference_ids_and_decode() {
        let Ok(directory) = std::env::var("METALLIX_DEEPSEEK_TOKENIZER_ASSETS") else {
            eprintln!("skipping: METALLIX_DEEPSEEK_TOKENIZER_ASSETS is not set");
            return;
        };
        let directory = std::path::Path::new(&directory);
        let reference: serde_json::Value = serde_json::from_slice(
            &std::fs::read(directory.join("plaintext-reference.json")).expect("reference receipt"),
        )
        .expect("reference JSON");
        let model = ModelDir::new(&json!({}), &json!({}), None);
        for name in ["tokenizer.json", "tokenizer_config.json"] {
            let bytes =
                chat_format::read_regular_file(&directory.join(name), 32 * 1024 * 1024, name)
                    .expect("bounded pinned asset");
            std::fs::write(model.path().join(name), bytes).expect("fixture asset copy");
        }
        assert_eq!(
            reference["revision"],
            "dba1be0a40aa45a94ad051997016db3960a90277"
        );
        let cases = reference["cases"].as_array().expect("reference cases");
        assert_eq!(cases.len(), 5);
        for case in cases {
            let text = case["text"].as_str().expect("reference text");
            let input = TextInput::load(model.path(), text).expect("pinned plaintext");
            assert_eq!(
                input.sha256(),
                "c90dfa01249db1be4245780a052ede752e1361c612ac6d08e2bdada7d599476b"
            );
            assert_eq!(
                serde_json::json!(input.ids),
                case["ids"],
                "{}",
                case["name"]
            );
            assert_eq!(
                input.decode(&input.ids).expect("decoded text"),
                case["decoded"]
            );
            assert_eq!(input.stops.len(), 1);
        }
    }
}
