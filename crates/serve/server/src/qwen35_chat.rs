//! `qwen3_5` and `qwen3_5_moe` hybrid checkpoints (Qwen3.5, Qwen3.6, Qwen3.8)
//! as a [`FullRowDecoder`] for the shared chat session.
//!
//! Most layers carry `GatedDeltaNet` recurrent state, a fold over every token
//! so far that cannot be cut back to a shared prefix the way K/V can. The
//! decoder therefore saves state ([`Qwen35Snapshot`]) only where the session
//! asks, at prompt boundaries while prefilling, and a later turn resumes from
//! the longest saved boundary its prompt starts with.

use std::path::Path;

use qwen35::{
    Qwen35Config,
    forward::{Qwen35Executor, Qwen35Precision, Qwen35Snapshot, Qwen35Weights},
};

use super::{
    ResidentChatLimits,
    decoder::{DecoderError, DecoderPlan, FullRowDecoder, FullRowSequence},
};

const MAX_CONFIG_BYTES: usize = 1024 * 1024;

fn state_bytes(config: &Qwen35Config, context: usize) -> Result<u64, qwen35::Qwen35ConfigError> {
    config
        .kv_bytes(config.kv_storage_tokens(context))?
        .checked_add(config.fixed_state_bytes()?)
        .ok_or(qwen35::Qwen35ConfigError::ShapeOverflow)
}

fn state_capacity(
    config: &Qwen35Config,
    budget: u64,
) -> Result<Option<std::num::NonZeroUsize>, qwen35::Qwen35ConfigError> {
    let mut fitting = 0;
    let mut upper = config.context_limit().min(2_147_483_647);
    while fitting < upper {
        let middle = fitting + (upper - fitting).div_ceil(2);
        if state_bytes(config, middle)? <= budget {
            fitting = middle;
        } else {
            upper = middle - 1;
        }
    }
    Ok(std::num::NonZeroUsize::new(fitting))
}

/// Resident `qwen3_5` weights at the checkpoint's stored precision (BF16 for
/// published weights); recurrent state and decay math stay f32.
pub(crate) struct Qwen35Decoder {
    weights: Qwen35Weights,
}

impl FullRowDecoder for Qwen35Decoder {
    type Sequence<'a> = Qwen35Sequence<'a>;
    type Snapshot = Qwen35Snapshot;
    const SAVES_STATE: bool = true;

    /// Admits the context when the K/V rows the executor allocates for that
    /// length (its storage tier, not just the context) in every
    /// full-attention layer, plus every linear layer's fixed recurrent and
    /// convolution state, fit the K/V budget. Sizes are logical f32; BF16
    /// K/V needs half of its part, so this over-admits nothing.
    fn plan(model: &Path, limits: ResidentChatLimits) -> Result<DecoderPlan, DecoderError> {
        let config = read_config(model)?;
        let context = limits.context_tokens();
        let ceiling = config.context_limit();
        if context == 0 || context > ceiling {
            return Err(DecoderError::ContextOutOfRange {
                requested: context,
                maximum: ceiling,
            });
        }
        let planned_bytes = state_bytes(&config, context)
            .map_err(|error| DecoderError::UnsupportedConfig(error.to_string()))?;
        if planned_bytes > limits.kv_budget_bytes() {
            return Err(DecoderError::StateBudgetCapacity {
                context,
                planned_bytes,
                budget_bytes: limits.kv_budget_bytes(),
                largest_fitting: state_capacity(&config, limits.kv_budget_bytes())
                    .map_err(|error| DecoderError::UnsupportedConfig(error.to_string()))?,
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

    fn sequence_from(
        &self,
        snapshot: &Qwen35Snapshot,
        _context_limit: usize,
    ) -> Result<Qwen35Sequence<'_>, String> {
        self.weights
            .executor_from(snapshot)
            .map(Qwen35Sequence)
            .map_err(|error| error.to_string())
    }

    fn snapshot_bytes(snapshot: &Qwen35Snapshot) -> usize {
        snapshot.state_bytes()
    }
}

/// One turn's recurrent, convolution and K/V state; dropped with the turn.
pub(crate) struct Qwen35Sequence<'a>(Qwen35Executor<'a>);

impl FullRowSequence for Qwen35Sequence<'_> {
    type Snapshot = Qwen35Snapshot;

    fn prefill(&mut self, tokens: &[i32]) -> Result<Vec<f32>, String> {
        self.0
            .extend_last_logits(tokens)
            .map_err(|error| error.to_string())
    }

    fn step(&mut self, token: i32) -> Result<Vec<f32>, String> {
        self.0
            .decode_last_logits(token)
            .map_err(|error| error.to_string())
    }

    fn snapshot(&self) -> Result<Qwen35Snapshot, String> {
        self.0.snapshot().map_err(|error| error.to_string())
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
    use std::{env, fs, path::PathBuf, time::Duration};

    use serde_json::{Value, json};

    use super::{DecoderError, FullRowDecoder, Qwen35Decoder, ResidentChatLimits};
    use crate::{
        chat_generation::{
            ChatBackend, ChatDecoderSession, ChatMessage, ChatRequest, ChatRole,
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

    fn assert_state_capacity(dir: &std::path::Path) {
        let config = super::read_config(dir).unwrap();
        let exact_plan = |context, budget| {
            Qwen35Decoder::plan(
                dir,
                ResidentChatLimits {
                    kv_budget_bytes: budget,
                    ..ResidentChatLimits::from_mib(context, 0)
                },
            )
        };
        for requested in [1, 128, 129, 512, 513, 1024, 1025, 2048, 2049, 262_144] {
            let cost = exact_plan(requested, u64::MAX).unwrap().planned_state_bytes;
            for budget in [cost, cost - 1] {
                let fit = super::state_capacity(&config, budget).unwrap();
                if let Some(tokens) = fit {
                    let tokens = tokens.get();
                    assert!(exact_plan(tokens, budget).is_ok());
                    if tokens < config.context_limit() {
                        assert!(matches!(
                            exact_plan(tokens + 1, budget),
                            Err(DecoderError::StateBudgetCapacity { .. })
                        ));
                    }
                } else {
                    assert!(matches!(
                        exact_plan(1, budget),
                        Err(DecoderError::StateBudgetCapacity { .. })
                    ));
                }
            }
        }
        assert_eq!(
            super::state_capacity(&config, 70_533_120)
                .unwrap()
                .unwrap()
                .get(),
            2048
        );
        assert_eq!(
            super::state_capacity(&config, 70_533_119)
                .unwrap()
                .unwrap()
                .get(),
            1024
        );
        assert!(super::state_capacity(&config, 4096).unwrap().is_none());
        assert_eq!(
            super::state_capacity(&config, u64::MAX)
                .unwrap()
                .unwrap()
                .get(),
            config.context_limit()
        );
        let almost_max = usize::try_from(u64::MAX / config.kv_bytes(1).unwrap()).unwrap();
        assert!(config.kv_bytes(almost_max).is_ok());
        assert!(matches!(
            super::state_bytes(&config, almost_max),
            Err(qwen35::Qwen35ConfigError::ShapeOverflow)
        ));
        let early =
            ChatDecoderSession::<Qwen35Decoder>::load(dir, ResidentChatLimits::from_mib(2049, 68))
                .err()
                .expect("state refusal before missing tokenizer or weights");
        assert!(
            early.contains("largest fitting logical context is 2048 tokens"),
            "{early}"
        );
        let no_fit =
            ChatDecoderSession::<Qwen35Decoder>::load(dir, ResidentChatLimits::from_mib(1, 1))
                .err()
                .expect("no state fits before loading");
        assert!(
            no_fit.contains("no positive logical context fits"),
            "{no_fit}"
        );
        // A non-power-of-two model cap truncates the last storage tier.
        let raw = fs::read_to_string(dir.join("config.json")).unwrap();
        let capped = raw.replace("262144", "1500");
        let capped = qwen35::Qwen35Config::parse(&capped).unwrap();
        assert_eq!(
            super::state_capacity(&capped, u64::MAX)
                .unwrap()
                .unwrap()
                .get(),
            1500
        );
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
        // 16,385 tokens is past the old 16,384 cap and inside the declared
        // 262,144; it is planned at its 32,768-row storage tier.
        let long = plan(16_385, 8_192).unwrap();
        assert_eq!(
            long.planned_state_bytes,
            6 * 2 * 2 * 256 * 32_768 * 4 + 18 * (16 * 128 * 128 + 6_144 * 3) * 4
        );
        let (over_budget, empty, too_long) = (plan(2_048, 67), plan(0, 512), plan(262_145, 8_192));
        // Use the serving CLI and its real limits conversion. The adapter,
        // not the argument parser, owns model-specific context admission.
        for (requested, expected) in [("16385", true), ("262145", false)] {
            use clap::Parser as _;
            let parsed = crate::Cli::try_parse_from([
                "mx",
                "serve",
                "--model",
                dir.to_str().unwrap(),
                "--context-tokens",
                requested,
                "--kv-budget-mib",
                "8192",
            ])
            .unwrap();
            let crate::Command::Serve {
                context_tokens,
                kv_budget_mib,
                ..
            } = parsed.command
            else {
                panic!("wrong command")
            };
            let limits = crate::resident_chat_limits(context_tokens, kv_budget_mib.unwrap());
            let result = Qwen35Decoder::plan(&dir, limits);
            if expected {
                assert_eq!(
                    result.unwrap().planned_state_bytes,
                    long.planned_state_bytes
                );
            } else {
                assert!(matches!(
                    result,
                    Err(DecoderError::ContextOutOfRange {
                        requested: 262_145,
                        maximum: 262_144,
                    })
                ));
            }
        }
        assert_state_capacity(&dir);
        fs::remove_dir_all(&dir).unwrap();
        assert!(matches!(
            over_budget,
            Err(DecoderError::StateBudgetCapacity {
                context: 2_048,
                planned_bytes: 70_533_120,
                ..
            })
        ));
        for refusal in [empty, too_long] {
            assert!(matches!(
                refusal,
                Err(DecoderError::ContextOutOfRange {
                    maximum: 262_144,
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

    /// Opt-in: a three-turn conversation on the checkpoint named by
    /// `METALLIX_QWEN35_CHAT_MODEL`, once with the prefix cache and once
    /// without. Greedy output must be identical; with the cache each later
    /// turn resumes from the previous turn's history boundary. Prints each
    /// turn's messages and counts for an offline check against the template.
    #[test]
    #[ignore = "requires METALLIX_QWEN35_CHAT_MODEL (a qwen3_5 checkpoint) on Apple-Silicon Metal"]
    fn real_qwen35_prefix_cache_resumes_turns_without_changing_output() {
        let _cap = GpuCap::start(8);
        let model =
            env::var_os("METALLIX_QWEN35_CHAT_MODEL").expect("set METALLIX_QWEN35_CHAT_MODEL");
        let load = |prefix_cache_mib| {
            ChatDecoderSession::<Qwen35Decoder>::load(
                std::path::Path::new(&model),
                ResidentChatLimits::from_mib(4_096, 512).with_prefix_cache_mib(prefix_cache_mib),
            )
            .expect("load local qwen3_5 checkpoint")
        };
        let (mut cached, mut uncached) = (load(512), load(0));
        let mut messages = vec![
            ChatMessage::text(
                ChatRole::System,
                "You are a terse assistant. Answer every question in at most five words. "
                    .repeat(20),
            ),
            ChatMessage::text(ChatRole::User, "Name a primary color."),
        ];
        let follow_ups = ["Name another one.", "And a third?"];
        let mut resumed = Vec::new();
        for (turn, follow_up) in follow_ups
            .iter()
            .copied()
            .map(Some)
            .chain(std::iter::once(None))
            .enumerate()
        {
            let request = ChatRequest::new(&messages, 16);
            let run = |session: &mut ChatDecoderSession<Qwen35Decoder>| {
                session
                    .generate_with_timeout(request, Duration::from_secs(120), &mut |_| Ok(()))
                    .expect("turn")
            };
            let (with, without) = (run(&mut cached), run(&mut uncached));
            assert_eq!(
                with.generated_token_ids, without.generated_token_ids,
                "turn {turn}: the cache changed greedy output"
            );
            assert_eq!(without.metrics.cached_prompt_tokens, 0);
            eprintln!(
                "GATE3 turn {turn}: prompt {} resumed {} text {:?} messages {}",
                with.metrics.prompt_tokens,
                with.metrics.cached_prompt_tokens,
                with.text,
                serde_json::to_string(
                    &messages
                        .iter()
                        .map(|m| json!({"role": format!("{:?}", m.role).to_lowercase(), "content": m.content}))
                        .collect::<Vec<_>>()
                )
                .unwrap()
            );
            resumed.push(with.metrics.cached_prompt_tokens);
            if let Some(follow_up) = follow_up {
                messages.push(ChatMessage::text(ChatRole::Assistant, with.text.clone()));
                messages.push(ChatMessage::text(ChatRole::User, follow_up));
            }
        }
        assert_eq!(resumed[0], 0);
        assert!(resumed[1] > 0 && resumed[2] > resumed[1], "{resumed:?}");
    }
}
