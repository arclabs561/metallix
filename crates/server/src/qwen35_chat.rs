//! `qwen3_5` and `qwen3_5_moe` hybrid checkpoints (Qwen3.5, Qwen3.6, Qwen3.8)
//! as a [`FullRowDecoder`] for the shared chat session.
//!
//! Most layers carry `GatedDeltaNet` recurrent state, a fold over every token
//! so far that cannot be cut back to a shared prefix the way K/V can, so this
//! decoder has no prefix cache: every turn prefills its whole prompt into a
//! fresh sequence. Reuse needs state snapshots taken at prompt boundaries
//! during prefill, a later change.

use std::path::Path;

use qwen35::{
    Qwen35Config,
    forward::{MAX_CONTEXT_TOKENS, Qwen35Executor, Qwen35Precision, Qwen35Weights},
};

use super::{
    ResidentChatLimits,
    decoder::{DecoderError, DecoderPlan, FullRowDecoder, FullRowSequence},
};

const MAX_CONFIG_BYTES: usize = 1024 * 1024;

/// Resident `qwen3_5` weights at the checkpoint's stored precision (BF16 for
/// published weights); recurrent state and decay math stay f32.
pub(crate) struct Qwen35Decoder {
    weights: Qwen35Weights,
}

impl FullRowDecoder for Qwen35Decoder {
    type Sequence<'a> = Qwen35Sequence<'a>;

    /// Admits the context when the K/V of every full-attention layer at that
    /// length plus every linear layer's fixed recurrent and convolution state
    /// fit the K/V budget. Sizes are logical f32; BF16 K/V needs half of its
    /// part, so this over-admits nothing.
    fn plan(model: &Path, limits: ResidentChatLimits) -> Result<DecoderPlan, DecoderError> {
        let config = read_config(model)?;
        let context = limits.context_tokens();
        let ceiling = MAX_CONTEXT_TOKENS.min(config.max_position_embeddings());
        if context == 0 || context > ceiling {
            return Err(DecoderError::ContextOutOfRange {
                requested: context,
                maximum: ceiling,
            });
        }
        let planned_bytes = config
            .kv_bytes(context)
            .and_then(|kv| Ok(kv.saturating_add(config.fixed_state_bytes()?)))
            .map_err(|error| DecoderError::UnsupportedConfig(error.to_string()))?;
        if planned_bytes > limits.kv_budget_bytes() {
            return Err(DecoderError::StateOverBudget {
                context,
                planned_bytes,
                budget_bytes: limits.kv_budget_bytes(),
            });
        }
        let planned_state_bytes = planned_bytes;
        Ok(DecoderPlan {
            vocabulary_size: config.vocab_size(),
            planned_state_bytes,
        })
    }

    /// Loads and runs on the calling thread: at checkpoint precision most
    /// tensors stay lazy, and MLX 0.32 evaluates a lazy graph only on the
    /// thread whose stream built it.
    fn load(model: &Path, _plan: &DecoderPlan) -> Result<Self, DecoderError> {
        let weights = Qwen35Weights::load(model, Qwen35Precision::Checkpoint)
            .map_err(|error| DecoderError::Load(error.to_string()))?;
        Ok(Self { weights })
    }

    fn sequence(&self, _context_limit: usize) -> Result<Self::Sequence<'_>, String> {
        // The executor enforces its own ceiling, which `plan` already checked.
        Ok(Qwen35Sequence(self.weights.executor()))
    }
}

/// One turn's recurrent, convolution and K/V state; dropped with the turn.
pub(crate) struct Qwen35Sequence<'a>(Qwen35Executor<'a>);

impl FullRowSequence for Qwen35Sequence<'_> {
    fn prefill(&mut self, tokens: &[i32]) -> Result<Vec<f32>, String> {
        self.0
            .prefill_last_logits(tokens)
            .map_err(|error| error.to_string())
    }

    fn step(&mut self, token: i32) -> Result<Vec<f32>, String> {
        self.0
            .decode_last_logits(token)
            .map_err(|error| error.to_string())
    }
}

fn read_config(model: &Path) -> Result<Qwen35Config, DecoderError> {
    let raw =
        chat_format::read_regular_file(&model.join("config.json"), MAX_CONFIG_BYTES, "config.json")
            .map_err(DecoderError::UnsupportedConfig)?;
    let raw = String::from_utf8(raw).map_err(|_| {
        DecoderError::UnsupportedConfig(String::from("local model config.json is not UTF-8"))
    })?;
    Qwen35Config::parse(&raw).map_err(|error| DecoderError::UnsupportedConfig(error.to_string()))
}

#[cfg(test)]
mod tests {
    use std::{env, fs, path::PathBuf};

    use serde_json::{Value, json};

    use super::{DecoderError, FullRowDecoder, Qwen35Decoder, ResidentChatLimits};
    use crate::{
        chat_generation::{
            ChatDecoderSession,
            decoder::test_support::{GpuCap, chat_completion},
        },
        sse::test_support::{events, json_body},
    };

    /// The text fields of Qwen/Qwen3.5-0.8B@2fc06364: 24 layers, every fourth
    /// full attention (2 K/V heads of 256), the rest `GatedDeltaNet` with 16
    /// key and value heads of 128.
    fn qwen35_08b_dir() -> PathBuf {
        let layer_types: Vec<&str> = (0..24)
            .map(|layer| {
                if layer % 4 == 3 {
                    "full_attention"
                } else {
                    "linear_attention"
                }
            })
            .collect();
        let config = json!({
            "model_type": "qwen3_5", "tie_word_embeddings": true,
            "text_config": {
                "model_type": "qwen3_5_text", "attention_bias": false, "attn_output_gate": true,
                "head_dim": 256, "hidden_act": "silu", "hidden_size": 1024,
                "intermediate_size": 3584, "layer_types": layer_types,
                "linear_conv_kernel_dim": 4, "linear_key_head_dim": 128,
                "linear_num_key_heads": 16, "linear_num_value_heads": 16,
                "linear_value_head_dim": 128, "mamba_ssm_dtype": "float32",
                "max_position_embeddings": 262_144, "num_attention_heads": 8,
                "num_hidden_layers": 24, "num_key_value_heads": 2, "rms_norm_eps": 1e-6,
                "rope_parameters": {"mrope_interleaved": true, "mrope_section": [11, 11, 10],
                    "partial_rotary_factor": 0.25, "rope_theta": 10_000_000, "rope_type": "default"},
                "tie_word_embeddings": true, "vocab_size": 248_320
            }
        });
        let dir = env::temp_dir().join(format!("qwen35-plan-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join("config.json"), config.to_string()).unwrap();
        dir
    }

    #[test]
    fn plan_counts_kv_and_the_fixed_recurrent_state() {
        let dir = qwen35_08b_dir();
        let plan = |context, kv_mib| {
            Qwen35Decoder::plan(&dir, ResidentChatLimits::from_mib(context, kv_mib))
        };
        // 6 full layers x 2 (K, V) x 2 heads x 256 x 2048 tokens, plus 18
        // linear layers x (16 x 128 x 128 recurrent + 6144 x 3 conv), f32:
        // 48 MiB + 20,201,472 bytes = 70,533,120 bytes (67.3 MiB).
        let admitted = plan(2_048, 68).unwrap();
        assert_eq!(admitted.vocabulary_size, 248_320);
        assert_eq!(
            admitted.planned_state_bytes,
            6 * 2 * 2 * 256 * 2_048 * 4 + 18 * (16 * 128 * 128 + 6_144 * 3) * 4
        );
        let (over_budget, empty, too_long) = (plan(2_048, 67), plan(0, 512), plan(16_385, 8_192));
        fs::remove_dir_all(&dir).unwrap();
        assert!(matches!(
            over_budget,
            Err(DecoderError::StateOverBudget {
                context: 2_048,
                planned_bytes: 70_533_120,
                ..
            })
        ));
        for refusal in [empty, too_long] {
            assert!(matches!(
                refusal,
                Err(DecoderError::ContextOutOfRange {
                    maximum: 16_384,
                    ..
                })
            ));
        }
    }

    /// Opt-in against Qwen/Qwen3.5-0.8B named by `METALLIX_QWEN35_CHAT_MODEL`
    /// (1.5 GB of BF16 weights).
    #[test]
    #[ignore = "requires METALLIX_QWEN35_CHAT_MODEL (a qwen3_5 checkpoint) on Apple-Silicon Metal"]
    fn real_qwen35_answers_streams_and_stops_at_end_of_turn() {
        let _cap = GpuCap::start(6);
        let model =
            env::var_os("METALLIX_QWEN35_CHAT_MODEL").expect("set METALLIX_QWEN35_CHAT_MODEL");
        let mut session = ChatDecoderSession::<Qwen35Decoder>::load(
            std::path::Path::new(&model),
            ResidentChatLimits::from_mib(2_048, 512),
        )
        .expect("load local qwen3_5 checkpoint");

        let body = json!({"model":"m","messages":[{"role":"user","content":"What is the capital of France? Answer in one word."}],"temperature":0,"max_completion_tokens":32});
        let (status, plain) = json_body(&chat_completion(&body, &mut session));
        assert_eq!(status, "HTTP/1.1 200 OK", "{plain}");
        assert_eq!(plain["choices"][0]["finish_reason"], "stop", "{plain}");
        let text = plain["choices"][0]["message"]["content"]
            .as_str()
            .unwrap()
            .to_owned();
        assert!(text.contains("Paris"), "{text:?}");
        // `<|im_end|>` ends the turn; it must stop decoding, never reach the text.
        assert!(!text.contains("<|im_end|>"), "{text:?}");

        // Each turn starts from fresh recurrent state, so greedy streaming
        // reproduces the same text.
        let mut streamed = body.clone();
        streamed["stream"] = json!(true);
        let joined: String = events(&chat_completion(&streamed, &mut session))
            .iter()
            .filter_map(|(_, data)| serde_json::from_str::<Value>(data).ok())
            .filter_map(|chunk| {
                chunk["choices"][0]["delta"]["content"]
                    .as_str()
                    .map(str::to_owned)
            })
            .collect();
        assert_eq!(joined, text);
        eprintln!(
            "answer {text:?}; MLX held {:?} bytes",
            crate::gpu::held_bytes()
        );
    }
}
