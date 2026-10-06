//! Resident Gemma 4 chat generation.
//!
//! The turn rules (render, budget, token picks, stop tokens, streamed text)
//! are the ones the Qwen session uses, from [`super::turn`]; only the logits
//! come from the `gemma` executor. Every step reads the full logit row: the
//! GPU pick path is Qwen's, and gemma-4-12B-it suppresses tokens, which that
//! path cannot. There is no prompt-prefix cache yet, so each turn prefills its
//! whole prompt.

use std::{
    path::{Path, PathBuf},
    time::{Duration, Instant},
};

use chat_format::{ChatFormat, TurnDelta};
use gemma::{
    Gemma4TextConfig,
    metal::{Gemma4MlxWeights, Gemma4Precision},
};
use tracing::field::Empty;

use super::{
    ChatBackend, ChatFinishReason, ChatGeneration, ChatGenerationError, ChatGenerationMetrics,
    ChatRequest, GenerationDeadline, ResidentChatLimits, SamplingDefaults, elapsed_ms, timed,
    turn::{TurnModel, TurnStart, TurnStep},
};

/// Weights and K/V in bf16, as for Qwen: decode reads every weight once per
/// token, and the 12B's f32 copy (about 44 GiB) buys no serving accuracy.
const SERVING_PRECISION: Gemma4Precision = Gemma4Precision::BFloat16;
const KV_BYTES_PER_ELEMENT: u64 = 2;

/// A resident Gemma 4 checkpoint for serial chat turns.
pub(crate) struct GemmaChatSession {
    weights: Gemma4MlxWeights,
    format: ChatFormat,
    vocabulary_size: usize,
    context_limit: usize,
    planned_kv_bytes: u64,
    load_ms: f64,
    /// Checkpoint directory, read again only to compile a JSON-schema grammar.
    model: PathBuf,
    sampling_defaults: SamplingDefaults,
}

impl GemmaChatSession {
    /// Validates the configuration and the K/V plan before any weights load,
    /// then loads the text weights, tokenizer and template.
    pub(crate) fn load(model: &Path, limits: ResidentChatLimits) -> Result<Self, String> {
        let started = Instant::now();
        let raw = std::fs::read_to_string(model.join("config.json"))
            .map_err(|_| String::from("local model config.json could not be read"))?;
        let config = Gemma4TextConfig::parse(&raw).map_err(|error| error.to_string())?;
        let planned_kv_bytes = plan_kv_bytes(&config, limits)?;
        let format = ChatFormat::load(model, config.vocab_size())?;
        let weights =
            Gemma4MlxWeights::load(model, SERVING_PRECISION).map_err(|error| error.to_string())?;
        Ok(Self {
            weights,
            format,
            vocabulary_size: config.vocab_size(),
            context_limit: limits.context_tokens(),
            planned_kv_bytes,
            load_ms: elapsed_ms(started.elapsed()),
            model: model.to_path_buf(),
            sampling_defaults: SamplingDefaults::load(model)?,
        })
    }

    fn turn_model(&self) -> TurnModel<'_> {
        TurnModel {
            format: &self.format,
            sampling_defaults: self.sampling_defaults,
            model: &self.model,
            vocabulary_size: self.vocabulary_size,
            context_limit: self.context_limit,
        }
    }

    fn generate_with_deadline(
        &mut self,
        request: ChatRequest<'_>,
        deadline: GenerationDeadline,
        on_token: &mut dyn FnMut(TurnDelta) -> Result<(), String>,
    ) -> Result<ChatGeneration, ChatGenerationError> {
        let start = TurnStart::prepare(self.turn_model(), request, deadline)?;
        let (render_ms, max_tokens) = (start.render_ms, start.max_tokens);
        let (input_ids, mut turn, mut text) = start.into_parts();
        let mut executor = self
            .weights
            .executor(self.context_limit)
            .map_err(|error| ChatGenerationError::message(error.to_string()))?;

        deadline.check()?;
        // Ends with the full-vocabulary logit readback, so it times the GPU work.
        let prefill = tracing::info_span!(
            "chat.prefill",
            prompt_tokens = input_ids.len(),
            cached_tokens = 0,
            prefill_ms = Empty
        );
        let (mut logits, prefill_ms) = timed(&prefill, "prefill_ms", || {
            executor
                .prefill_last_logits(&input_ids)
                .map_err(|error| ChatGenerationError::message(error.to_string()))
        })?;
        let mut decode_ms = Vec::new();
        let mut finish_reason = ChatFinishReason::Length;
        for step in 0..max_tokens {
            deadline.check()?;
            let accepted = turn.pick(&self.format, &mut logits)?;
            if accepted.visible {
                text.push(&self.format, accepted.token, on_token)?;
            }
            if let TurnStep::Stop(reason) = accepted.step {
                finish_reason = reason;
                break;
            }
            deadline.check()?;
            // One span per generated token, ending with its logit readback.
            let span = tracing::info_span!("chat.decode_step", step, decode_ms = Empty);
            let (next, step_ms) = timed(&span, "decode_ms", || {
                executor
                    .decode_last_logits(accepted.token)
                    .map_err(|error| ChatGenerationError::message(error.to_string()))
            })?;
            logits = next;
            decode_ms.push(step_ms);
        }

        deadline.check()?;
        let end = text.finish(
            &self.format,
            turn.generated(),
            request.tools,
            finish_reason == ChatFinishReason::Eos,
            on_token,
        )?;
        let output = turn.finish()?;
        let decode_total_ms = decode_ms.iter().sum();
        let generated_tokens = output.generated.len();
        Ok(ChatGeneration {
            text: end.text,
            turn: end.turn,
            generated_token_ids: output.generated,
            finish_reason,
            logprobs: output.logprobs,
            sampling: Some(output.sampling),
            metrics: ChatGenerationMetrics {
                context_tokens: self.context_limit,
                planned_kv_bytes: self.planned_kv_bytes,
                session_load_ms: self.load_ms,
                render_ms,
                prefill_ms,
                time_to_first_token_ms: end.time_to_first_token_ms,
                decode_ms,
                decode_total_ms,
                prompt_tokens: input_ids.len(),
                cached_prompt_tokens: 0,
                generated_tokens,
                // Prompt-lookup speculation runs on the Qwen executor only;
                // it never changes output, so a request asking for it is
                // decoded one token per step here.
                speculation: None,
            },
        })
    }
}

impl ChatBackend for GemmaChatSession {
    fn load_ms(&self) -> f64 {
        self.load_ms
    }

    fn generate_with_timeout(
        &mut self,
        request: ChatRequest<'_>,
        timeout: Duration,
        on_token: &mut dyn FnMut(TurnDelta) -> Result<(), String>,
    ) -> Result<ChatGeneration, ChatGenerationError> {
        // Start before validation and rendering so the caller budgets the full turn.
        self.generate_with_deadline(request, GenerationDeadline::after(timeout), on_token)
    }
}

/// The bf16 K/V a sequence at the context limit retains (sliding layers keep
/// only their window), refused when it exceeds the K/V budget or the
/// checkpoint's positions.
fn plan_kv_bytes(config: &Gemma4TextConfig, limits: ResidentChatLimits) -> Result<u64, String> {
    let context = limits.context_tokens();
    if context == 0 || context > config.max_position_embeddings() {
        return Err(format!(
            "Gemma 4 context must be 1 through {} tokens; requested {context}",
            config.max_position_embeddings()
        ));
    }
    let planned = u64::try_from(config.retained_kv_elements(context))
        .ok()
        .and_then(|elements| elements.checked_mul(KV_BYTES_PER_ELEMENT))
        .ok_or_else(|| String::from("Gemma 4 K/V plan overflows"))?;
    if planned > limits.kv_budget_bytes() {
        return Err(format!(
            "Gemma 4 K/V for {context} tokens needs {planned} bytes, more than the {}-byte K/V budget",
            limits.kv_budget_bytes()
        ));
    }
    Ok(planned)
}

#[cfg(test)]
mod tests {
    use std::{
        sync::{
            Arc,
            atomic::{AtomicBool, Ordering},
        },
        thread,
        time::Duration,
    };

    use serde_json::{Value, json};

    use super::{GemmaChatSession, ResidentChatLimits, plan_kv_bytes};
    use crate::sse::test_support::{events, exchange, json_body};

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
        assert!(plan_kv_bytes(&config, ResidentChatLimits::from_mib(16_384, 575)).is_err());
        assert!(plan_kv_bytes(&config, ResidentChatLimits::from_mib(0, 1_024)).is_err());
        assert!(plan_kv_bytes(&config, ResidentChatLimits::from_mib(262_145, 65_536)).is_err());
    }

    /// Aborts the test process when MLX's active plus cached memory passes
    /// `cap_gib` (default 40, `METALLIX_GPU_CAP_GIB`), checked every 200 ms;
    /// stops checking when dropped. MLX Metal buffers can grow far past the
    /// process's resident size, so RSS alone does not show a leak.
    struct GpuCap(Arc<AtomicBool>);

    impl GpuCap {
        fn start() -> Self {
            let cap_gib: u64 = std::env::var("METALLIX_GPU_CAP_GIB")
                .ok()
                .and_then(|value| value.parse().ok())
                .unwrap_or(40);
            let running = Arc::new(AtomicBool::new(true));
            let watching = Arc::clone(&running);
            thread::spawn(move || {
                while watching.load(Ordering::Acquire) {
                    if let Some(used) = crate::gpu::held_bytes() {
                        if used > cap_gib << 30 {
                            eprintln!(
                                "MLX memory {used} bytes passed the {cap_gib} GiB cap; aborting"
                            );
                            std::process::abort();
                        }
                    }
                    thread::sleep(Duration::from_millis(200));
                }
            });
            Self(running)
        }
    }

    impl Drop for GpuCap {
        fn drop(&mut self) {
            self.0.store(false, Ordering::Release);
        }
    }

    fn run(body: &Value, session: &mut GemmaChatSession) -> String {
        exchange(
            "/v1/chat/completions",
            &body.to_string(),
            |connection, body| {
                let (_, prepared) = crate::chat_completions::prepare(body).unwrap();
                crate::chat_completions::respond(
                    connection,
                    &prepared,
                    session,
                    "g",
                    Duration::from_secs(600),
                )
                .unwrap();
            },
        )
    }

    /// Opt-in against google/gemma-4-12B-it named by `METALLIX_GEMMA4_MODEL`
    /// (about 22 GiB of bf16 weights; take the heavy lease).
    #[test]
    #[ignore = "requires METALLIX_GEMMA4_MODEL (google/gemma-4-12B-it) on Apple-Silicon Metal"]
    fn real_gemma4_answers_streams_and_calls_a_tool() {
        let _cap = GpuCap::start();
        let model = std::env::var_os("METALLIX_GEMMA4_MODEL").expect("set METALLIX_GEMMA4_MODEL");
        let mut session = GemmaChatSession::load(
            std::path::Path::new(&model),
            ResidentChatLimits::from_mib(16_384, 1_024),
        )
        .expect("load local Gemma 4 checkpoint");

        let body = json!({"model":"g","messages":[{"role":"user","content":"Name three primary colors."}],"temperature":0,"max_completion_tokens":32});
        let (status, plain) = json_body(&run(&body, &mut session));
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
        let joined: String = events(&run(&streamed, &mut session))
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
        let body = json!({"model":"g","messages":[{"role":"user","content":"Weather in Paris?"}],"tools":tools,"temperature":0,"max_completion_tokens":64});
        let (_, called) = json_body(&run(&body, &mut session));
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
