//! Gemma 4 dense text as a [`FullRowDecoder`] for chat sessions.

use std::path::Path;

use gemma::{
    Gemma4TextConfig,
    forward::Gemma4Executor,
    metal::{Gemma4MlxWeights, Gemma4Precision},
};

use super::{
    ResidentChatLimits,
    decoder::{DecoderError, DecoderPlan, FullRowDecoder, FullRowSequence, NoSnapshot},
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
    type Snapshot = NoSnapshot;

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

    fn sequence_from(&self, snapshot: &NoSnapshot, _: usize) -> Result<GemmaSequence<'_>, String> {
        match *snapshot {}
    }

    fn snapshot_bytes(snapshot: &NoSnapshot) -> usize {
        match *snapshot {}
    }
}

impl FullRowSequence for GemmaSequence<'_> {
    type Snapshot = NoSnapshot;

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

    fn snapshot(&self) -> Result<NoSnapshot, String> {
        Err(String::from("Gemma 4 sequences do not save state"))
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
            decoder::test_support::{GpuCap, chat_completion, messages, responses},
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

    /// The real-checkpoint tests load about 22 GiB of bf16 weights each, so
    /// they run one at a time under the GPU cap.
    static REAL_CHECKPOINT: std::sync::Mutex<()> = std::sync::Mutex::new(());

    /// google/gemma-4-12B-it named by `METALLIX_GEMMA4_MODEL`; take the heavy
    /// lease.
    fn real_session() -> ChatDecoderSession<GemmaDecoder> {
        let model = std::env::var_os("METALLIX_GEMMA4_MODEL").expect("set METALLIX_GEMMA4_MODEL");
        ChatDecoderSession::<GemmaDecoder>::load(
            std::path::Path::new(&model),
            ResidentChatLimits::from_mib(16_384, 1_024),
        )
        .expect("load local Gemma 4 checkpoint")
    }

    /// The tool and prompt of the `tool_declaration` case in
    /// fixtures/gemma-4-12b/reference.json, as an `OpenAI` function schema.
    fn weather_parameters() -> Value {
        json!({"type":"object","properties":{"city":{"type":"string","description":"City name."},"unit":{"type":"string","enum":["celsius","fahrenheit"]}},"required":["city"]})
    }

    fn assert_no_channel_markers(text: &str) {
        assert!(
            !text.contains("<|channel>") && !text.contains("<channel|>"),
            "{text:?}"
        );
    }

    #[test]
    #[ignore = "requires METALLIX_GEMMA4_MODEL (google/gemma-4-12B-it) on Apple-Silicon Metal"]
    fn real_gemma4_answers_streams_and_calls_a_tool() {
        let _serial = REAL_CHECKPOINT
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let _cap = GpuCap::start(40);
        let mut session = real_session();

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

        let tools = json!([{"type":"function","function":{"name":"get_weather","description":"Current weather for a city.","parameters":weather_parameters()}}]);
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

    /// Over `/v1/messages`: an answer, the same text streamed, a tool call,
    /// and an answer from its result.
    #[test]
    #[ignore = "requires METALLIX_GEMMA4_MODEL (google/gemma-4-12B-it) on Apple-Silicon Metal"]
    fn real_gemma4_messages_answer_stream_and_round_trip_a_tool() {
        let _serial = REAL_CHECKPOINT
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let _cap = GpuCap::start(40);
        let mut session = real_session();

        // Messages: answer, the same text streamed, a tool call and its result.
        let body = json!({"model":"m","max_tokens":32,"temperature":0,"messages":[{"role":"user","content":"Name three primary colors."}]});
        let (status, plain) = json_body(&messages(&body, &mut session));
        assert_eq!(status, "HTTP/1.1 200 OK", "{plain}");
        let text = plain["content"][0]["text"].as_str().unwrap().to_owned();
        assert!(!text.is_empty(), "{plain}");
        assert_no_channel_markers(&text);
        let mut streamed = body.clone();
        streamed["stream"] = json!(true);
        let joined: String = events(&messages(&streamed, &mut session))
            .iter()
            .filter_map(|(_, data)| serde_json::from_str::<Value>(data).ok())
            .filter(|event| event["type"] == "content_block_delta")
            .filter_map(|event| event["delta"]["text"].as_str().map(str::to_owned))
            .collect();
        assert_eq!(joined, text);

        let tools = json!([{"name":"get_weather","description":"Current weather for a city.","input_schema":weather_parameters()}]);
        let body = json!({"model":"m","max_tokens":64,"temperature":0,"tools":tools,"messages":[{"role":"user","content":"Weather in Paris?"}]});
        let (_, called) = json_body(&messages(&body, &mut session));
        assert_eq!(called["stop_reason"], "tool_use", "{called}");
        let call = called["content"]
            .as_array()
            .unwrap()
            .iter()
            .find(|block| block["type"] == "tool_use")
            .unwrap()
            .clone();
        assert_eq!(call["name"], "get_weather", "{called}");
        assert_eq!(call["input"], json!({"city":"Paris"}), "{called}");
        // The result renders as Gemma's tool response; the model answers from it.
        let follow = json!({"model":"m","max_tokens":64,"temperature":0,"tools":tools,"messages":[
            {"role":"user","content":"Weather in Paris?"},
            {"role":"assistant","content":called["content"]},
            {"role":"user","content":[{"type":"tool_result","tool_use_id":call["id"],"content":"{\"temperature\": 18, \"sky\": \"clear\"}"}]}
        ]});
        let (status, answered) = json_body(&messages(&follow, &mut session));
        assert_eq!(status, "HTTP/1.1 200 OK", "{answered}");
        assert_eq!(answered["stop_reason"], "end_turn", "{answered}");
        let answer = answered["content"][0]["text"].as_str().unwrap();
        assert!(answer.contains("18"), "{answered}");
        assert_no_channel_markers(answer);
    }

    /// Over `/v1/responses`: the same answer as chat completions, and a
    /// function call.
    #[test]
    #[ignore = "requires METALLIX_GEMMA4_MODEL (google/gemma-4-12B-it) on Apple-Silicon Metal"]
    fn real_gemma4_responses_answer_and_call_a_function() {
        let _serial = REAL_CHECKPOINT
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let _cap = GpuCap::start(40);
        let mut session = real_session();

        // The chat-completions answer to the same prompt.
        let body = json!({"model":"m","messages":[{"role":"user","content":"Name three primary colors."}],"temperature":0,"max_completion_tokens":32});
        let (_, plain) = json_body(&chat_completion(&body, &mut session));
        let text = plain["choices"][0]["message"]["content"]
            .as_str()
            .unwrap()
            .to_owned();

        // Responses: an answer and a function call.
        let body = json!({"model":"m","input":"Name three primary colors.","temperature":0,"max_output_tokens":32});
        let (status, response) = json_body(&responses(&body, &mut session));
        assert_eq!(status, "HTTP/1.1 200 OK", "{response}");
        let message = response["output"]
            .as_array()
            .unwrap()
            .iter()
            .find(|item| item["type"] == "message")
            .unwrap();
        assert_eq!(
            message["content"][0]["text"].as_str().unwrap(),
            text,
            "{response}"
        );
        let tools = json!([{"type":"function","name":"get_weather","description":"Current weather for a city.","parameters":weather_parameters()}]);
        let body = json!({"model":"m","input":"Weather in Paris?","tools":tools,"temperature":0,"max_output_tokens":64});
        let (_, response) = json_body(&responses(&body, &mut session));
        let call = response["output"]
            .as_array()
            .unwrap()
            .iter()
            .find(|item| item["type"] == "function_call")
            .unwrap_or_else(|| panic!("{response}"));
        assert_eq!(call["name"], "get_weather", "{response}");
        let arguments: Value = serde_json::from_str(call["arguments"].as_str().unwrap()).unwrap();
        assert_eq!(arguments, json!({"city":"Paris"}), "{response}");
    }

    /// Thinking on through `reasoning_effort`.
    #[test]
    #[ignore = "requires METALLIX_GEMMA4_MODEL (google/gemma-4-12B-it) on Apple-Silicon Metal"]
    fn real_gemma4_thinking_streams_reasoning_apart_from_the_answer() {
        let _serial = REAL_CHECKPOINT
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let _cap = GpuCap::start(40);
        let mut session = real_session();

        // Thinking: <|think|> in the system turn; the trace streams as
        // reasoning, never as answer text. 91 = 7 x 13.
        let body = json!({"model":"m","messages":[{"role":"user","content":"Is 91 prime?"}],"reasoning_effort":"low","temperature":0,"max_completion_tokens":768});
        let (_, thought) = json_body(&chat_completion(&body, &mut session));
        assert_eq!(thought["choices"][0]["finish_reason"], "stop", "{thought}");
        let reasoning = thought["choices"][0]["message"]["reasoning_content"]
            .as_str()
            .unwrap_or_else(|| panic!("{thought}"))
            .to_owned();
        let content = thought["choices"][0]["message"]["content"]
            .as_str()
            .unwrap()
            .to_owned();
        assert!(!reasoning.is_empty(), "{thought}");
        assert!(content.contains("13"), "{thought}");
        assert_no_channel_markers(&reasoning);
        assert_no_channel_markers(&content);
        let mut streamed = body.clone();
        streamed["stream"] = json!(true);
        let chunks: Vec<Value> = events(&chat_completion(&streamed, &mut session))
            .iter()
            .filter_map(|(_, data)| serde_json::from_str::<Value>(data).ok())
            .collect();
        let delta = |field: &str| -> String {
            chunks
                .iter()
                .filter_map(|chunk| chunk["choices"][0]["delta"][field].as_str())
                .collect()
        };
        assert_eq!(delta("reasoning_content"), reasoning);
        assert_eq!(delta("content"), content);
        eprintln!(
            "thinking: {} reasoning chars, answer {content:?}; MLX held {:?} bytes",
            reasoning.len(),
            crate::gpu::held_bytes()
        );
    }
}
