//! Gemma 4 dense text as a [`FullRowDecoder`] for chat sessions.

use std::path::Path;

use gemma::{
    Gemma4TextConfig,
    forward::Gemma4Executor,
    metal::{Gemma4MlxWeights, Gemma4Precision},
};

use super::{
    ResidentChatLimits,
    decoder::{DecoderError, DecoderPlan, FullRowDecoder, FullRowSequence},
};

/// Weights and K/V in bf16, as for Qwen: decode reads every weight once per
/// token, and the 12B's f32 copy (about 44 GiB) buys no serving accuracy.
const SERVING_PRECISION: Gemma4Precision = Gemma4Precision::BFloat16;
const KV_BYTES_PER_ELEMENT: u64 = 2;

/// Loaded Gemma 4 text weights (12B or 31B).
pub(crate) struct GemmaDecoder {
    weights: Gemma4MlxWeights,
}

/// One turn's Gemma 4 sequence.
pub(crate) struct GemmaSequence<'a>(Gemma4Executor<'a>);

impl FullRowDecoder for GemmaDecoder {
    type Sequence<'a> = GemmaSequence<'a>;

    fn plan(model: &Path, limits: ResidentChatLimits) -> Result<DecoderPlan, DecoderError> {
        let raw = std::fs::read_to_string(model.join("config.json")).map_err(|_| {
            DecoderError::UnsupportedConfig("local model config.json could not be read".into())
        })?;
        let config = Gemma4TextConfig::parse(&raw)
            .map_err(|error| DecoderError::UnsupportedConfig(error.to_string()))?;
        Ok(DecoderPlan {
            vocabulary_size: config.vocab_size(),
            planned_state_bytes: plan_kv_bytes(&config, limits)?,
        })
    }

    fn load(model: &Path, _plan: &DecoderPlan) -> Result<Self, DecoderError> {
        Ok(Self {
            weights: Gemma4MlxWeights::load(model, SERVING_PRECISION)
                .map_err(|error| DecoderError::Load(error.to_string()))?,
        })
    }

    fn sequence(&self, context_limit: usize) -> Result<GemmaSequence<'_>, String> {
        self.weights
            .executor(context_limit)
            .map(GemmaSequence)
            .map_err(|error| error.to_string())
    }
}

impl FullRowSequence for GemmaSequence<'_> {
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

/// The bf16 K/V a sequence at the context limit retains (sliding layers keep
/// only their window), refused when it exceeds the K/V budget or the
/// checkpoint's positions.
fn plan_kv_bytes(
    config: &Gemma4TextConfig,
    limits: ResidentChatLimits,
) -> Result<u64, DecoderError> {
    let context = limits.context_tokens();
    let maximum = config.max_position_embeddings();
    if context == 0 || context > maximum {
        return Err(DecoderError::ContextOutOfRange {
            requested: context,
            maximum,
        });
    }
    let planned_bytes = u64::try_from(config.retained_kv_elements(context))
        .ok()
        .and_then(|elements| elements.checked_mul(KV_BYTES_PER_ELEMENT))
        .ok_or_else(|| DecoderError::UnsupportedConfig("K/V plan overflows".into()))?;
    if planned_bytes > limits.kv_budget_bytes() {
        return Err(DecoderError::StateOverBudget {
            context,
            planned_bytes,
            budget_bytes: limits.kv_budget_bytes(),
        });
    }
    Ok(planned_bytes)
}

#[cfg(test)]
mod tests {
    use serde_json::{Value, json};

    use super::{DecoderError, GemmaDecoder, ResidentChatLimits, plan_kv_bytes};
    use crate::{
        chat_generation::{
            ChatDecoderSession,
            decoder::test_support::{GpuCap, chat_completion},
        },
        sse::test_support::{events, json_body},
    };

    /// google/gemma-4-12B-it@707f0a3's text layout: 48 layers, five sliding
    /// (8 K/V heads of 256) then one full (1 K/V head of 512).
    fn gemma4_12b() -> gemma::Gemma4TextConfig {
        let layer_types: Vec<&str> = (0..48)
            .map(|layer| {
                if layer % 6 == 5 {
                    "full_attention"
                } else {
                    "sliding_attention"
                }
            })
            .collect();
        let config = json!({
            "model_type": "gemma4_text", "attention_k_eq_v": true,
            "final_logit_softcapping": 30.0, "global_head_dim": 512, "head_dim": 256,
            "hidden_activation": "gelu_pytorch_tanh", "hidden_size": 3840,
            "intermediate_size": 15360, "layer_types": layer_types,
            "max_position_embeddings": 262_144, "num_attention_heads": 16,
            "num_global_key_value_heads": 1, "num_hidden_layers": 48,
            "num_key_value_heads": 8, "rms_norm_eps": 1e-6,
            "rope_parameters": {
                "full_attention": {"partial_rotary_factor": 0.25, "rope_theta": 1_000_000.0, "rope_type": "proportional"},
                "sliding_attention": {"rope_theta": 10_000.0, "rope_type": "default"}
            },
            "sliding_window": 1024, "tie_word_embeddings": true, "vocab_size": 262_144
        });
        gemma::Gemma4TextConfig::parse(&config.to_string()).unwrap()
    }

    #[test]
    fn kv_plan_counts_only_the_sliding_window_and_respects_the_budget() {
        let config = gemma4_12b();
        // 40 sliding layers keep 1023 positions of 8x256 K and V; 8 full
        // layers keep all 16384 of 1x512, in bf16: 320 MiB + 256 MiB.
        let planned = plan_kv_bytes(&config, ResidentChatLimits::from_mib(16_384, 1_024)).unwrap();
        assert_eq!(
            planned,
            40 * 2 * 8 * 256 * 2 * 1023 + 8 * 2 * 512 * 2 * 16_384
        );
        assert_eq!(
            plan_kv_bytes(&config, ResidentChatLimits::from_mib(16_384, 575)),
            Err(DecoderError::StateOverBudget {
                context: 16_384,
                planned_bytes: planned,
                budget_bytes: 575 << 20,
            })
        );
        for requested in [0, 262_145] {
            assert_eq!(
                plan_kv_bytes(&config, ResidentChatLimits::from_mib(requested, 65_536)),
                Err(DecoderError::ContextOutOfRange {
                    requested,
                    maximum: 262_144
                })
            );
        }
    }

    /// Opt-in against google/gemma-4-12B-it named by `METALLIX_GEMMA4_MODEL`
    /// (about 22 GiB of bf16 weights; take the heavy lease).
    #[test]
    #[ignore = "requires METALLIX_GEMMA4_MODEL (google/gemma-4-12B-it) on Apple-Silicon Metal"]
    fn real_gemma4_answers_streams_and_calls_a_tool() {
        let _cap = GpuCap::start(40);
        let model = std::env::var_os("METALLIX_GEMMA4_MODEL").expect("set METALLIX_GEMMA4_MODEL");
        let mut session = ChatDecoderSession::<GemmaDecoder>::load(
            std::path::Path::new(&model),
            ResidentChatLimits::from_mib(16_384, 1_024),
        )
        .expect("load local Gemma 4 checkpoint");

        let body = json!({"model":"m","messages":[{"role":"user","content":"Name three primary colors."}],"temperature":0,"max_completion_tokens":32});
        let (status, plain) = json_body(&chat_completion(&body, &mut session));
        assert_eq!(status, "HTTP/1.1 200 OK", "{plain}");
        let text = plain["choices"][0]["message"]["content"]
            .as_str()
            .unwrap()
            .to_owned();
        assert!(!text.is_empty(), "{plain}");
        // The template's empty thought channel never reaches the answer.
        assert!(
            !text.contains("<|channel>") && !text.contains("<channel|>"),
            "{text:?}"
        );

        // Greedy streaming reproduces the same text.
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

        // The tool and prompt of the `tool_declaration` case in
        // fixtures/gemma-4-12b/reference.json.
        let tools = json!([{"type":"function","function":{"name":"get_weather","description":"Current weather for a city.","parameters":{"type":"object","properties":{"city":{"type":"string","description":"City name."},"unit":{"type":"string","enum":["celsius","fahrenheit"]}},"required":["city"]}}}]);
        let body = json!({"model":"m","messages":[{"role":"user","content":"Weather in Paris?"}],"tools":tools,"temperature":0,"max_completion_tokens":64});
        let (_, called) = json_body(&chat_completion(&body, &mut session));
        assert_eq!(
            called["choices"][0]["finish_reason"], "tool_calls",
            "{called}"
        );
        let call = &called["choices"][0]["message"]["tool_calls"][0];
        assert_eq!(call["function"]["name"], "get_weather", "{called}");
        let arguments: Value =
            serde_json::from_str(call["function"]["arguments"].as_str().unwrap()).unwrap();
        // transformers' float32 greedy call for this prompt and tool.
        assert_eq!(arguments, json!({"city":"Paris"}), "{called}");
        eprintln!(
            "answer {text:?}; call {arguments}; MLX held {:?} bytes",
            crate::gpu::held_bytes()
        );
    }
}
