//! Opt-in checkpoint checks for the GPU-picked decode path.
//!
//! Run with `METALLIX_QWEN_MODEL=/path/to/Qwen3-0.6B cargo test --release -p
//! server --features metal decode_checkpoint -- --ignored --nocapture
//! --test-threads=1`.

use std::{env, fmt::Write as _, path::PathBuf, time::Instant};

use super::{
    ChatMessage, ChatRequest, ChatRole, ChatSession, GenerationDeadline, ResidentChatLimits,
    SamplingRequest, TurnStart, full_row,
};

const MAX_TOKENS: u32 = 128;
const ROWS: usize = 3;
/// Logprob receipts differ from the host's only by the f32 softmax sum.
const LOGPROB_TOLERANCE: f64 = 1e-4;

fn load_session() -> ChatSession {
    let model = env::var_os("METALLIX_QWEN_MODEL")
        .map(PathBuf::from)
        .expect("METALLIX_QWEN_MODEL is required for these ignored checkpoint tests");
    ChatSession::load(&model, ResidentChatLimits::from_mib(4_096, 2_048)).expect("session load")
}

fn preamble() -> String {
    let mut text = String::from("You are a careful technical writer.\n");
    for rule in 0..70 {
        writeln!(
            text,
            "Rule {rule}: keep paragraphs short, name every unit, and prefer concrete \
             examples over general claims when you explain rule {rule}."
        )
        .expect("writing to a String cannot fail");
    }
    text
}

fn messages() -> [ChatMessage; 2] {
    [
        ChatMessage::text(ChatRole::System, preamble()),
        ChatMessage::text(
            ChatRole::User,
            "Write a long, detailed guide to maintaining a bicycle chain.",
        ),
    ]
}

fn seeded(temperature: Option<f64>, top_p: Option<f64>, top_k: Option<u32>) -> SamplingRequest {
    SamplingRequest {
        temperature,
        top_p,
        top_k,
        seed: Some(7),
    }
}

/// Every GPU-picked step settles the token and receipts that the full-row
/// host path computes from the same logits.
#[test]
#[ignore = "requires METALLIX_QWEN_MODEL pointing to Qwen3-0.6B on Apple-Silicon Metal"]
fn gpu_settled_steps_match_the_full_row_host_path() {
    let session = load_session();
    let messages = messages();
    let cases = [
        ("greedy_top5", SamplingRequest::GREEDY, Some(5)),
        ("defaults", seeded(None, None, None), None),
        ("defaults_top3", seeded(None, None, None), Some(3)),
        ("hot_narrow", seeded(Some(1.2), Some(0.9), Some(5)), Some(0)),
    ];
    for (name, sampling, top_logprobs) in cases {
        let mut request = ChatRequest::new(&messages, 64);
        request.sampling = sampling;
        request.top_logprobs = top_logprobs;
        let prepare = || {
            TurnStart::prepare(
                session.turn_model(),
                request,
                GenerationDeadline::unlimited(),
            )
            .expect("prepare")
            .into_parts()
        };
        let (input_ids, mut host, _) = prepare();
        let (_, mut gpu, _) = prepare();
        let format = &session.format;
        let mut executor = session
            .weights
            .resident_chat_executor(session.context_limit, session.kv_budget_bytes)
            .expect("executor");
        let logits = executor.prefill_last_logits(&input_ids).expect("prefill");
        let mut token = host
            .pick(format, &mut logits.clone())
            .expect("host pick")
            .token;
        assert_eq!(
            gpu.pick(format, &mut logits.clone())
                .expect("gpu-side pick")
                .token,
            token
        );
        let mut gpu_draws_rejected = 0;
        for step in 1..64 {
            let rule = gpu
                .gpu_rule(format, 0)
                .expect("this case takes the GPU path");
            let picks = executor.decode_picks(token, &rule).expect("decode");
            let mut row = full_row(&picks).expect("row");
            let expected = host.pick(format, &mut row).expect("host pick").token;
            let (settled, gpu_token) = gpu.settle(format, &picks).expect("settle");
            assert_eq!(settled.token, expected, "{name} step {step}");
            if gpu_token != settled.token {
                gpu_draws_rejected += 1;
            }
            token = expected;
        }
        // Receipts were recorded by both loops for the same tokens.
        let (host, gpu) = (host.finish().expect("host"), gpu.finish().expect("gpu"));
        assert_eq!(gpu.logprobs.len(), host.logprobs.len(), "{name}");
        for (step, (receipt, expected)) in gpu.logprobs.iter().zip(&host.logprobs).enumerate() {
            assert_eq!(receipt.token, expected.token, "{name} receipt {step}");
            assert!((receipt.logprob - expected.logprob).abs() < LOGPROB_TOLERANCE);
            assert_eq!(receipt.top_logprobs.len(), expected.top_logprobs.len());
            for (actual, expected) in receipt.top_logprobs.iter().zip(&expected.top_logprobs) {
                assert_eq!(actual.bytes, expected.bytes, "{name} receipt {step}");
                assert!((actual.logprob - expected.logprob).abs() < LOGPROB_TOLERANCE);
            }
        }
        println!("decode_checkpoint case={name} steps=63 gpu_draws_rejected={gpu_draws_rejected}");
    }
}

/// Prints wall time per generated token after the first for each decode
/// mode. Indicative only: host wall time on whatever else the machine runs.
#[test]
#[ignore = "requires METALLIX_QWEN_MODEL pointing to Qwen3-0.6B on Apple-Silicon Metal"]
fn per_token_cost_by_decode_mode() {
    let mut session = load_session();
    let messages = messages();
    let modes: [(&str, SamplingRequest, Option<u8>); 5] = [
        ("greedy", SamplingRequest::GREEDY, None),
        ("greedy_top_logprobs_5", SamplingRequest::GREEDY, Some(5)),
        ("sampled_defaults_seeded", seeded(None, None, None), None),
        (
            "sampled_defaults_top_logprobs_5",
            seeded(None, None, None),
            Some(5),
        ),
        // A top_k past the vocabulary is no truncation, so the host path
        // reads whole rows.
        (
            "sampled_no_top_k",
            seeded(Some(0.8), Some(0.9), Some(1_000_000)),
            None,
        ),
    ];
    for (name, sampling, top_logprobs) in modes {
        for row in 0..=ROWS {
            let mut request = ChatRequest::new(&messages, MAX_TOKENS);
            request.sampling = sampling;
            request.top_logprobs = top_logprobs;
            let mut first_token_at = None;
            let generation = session
                .generate(request, &mut |_| {
                    first_token_at.get_or_insert_with(Instant::now);
                    Ok(())
                })
                .expect("generation");
            let finished = Instant::now();
            let tokens = generation.generated_token_ids.len();
            let first = first_token_at.expect("at least one visible token");
            // Row 0 warms the prefix cache and kernels.
            if row > 0 && tokens > 1 {
                let per_token = (finished - first).as_secs_f64() * 1_000.0
                    / f64::from(u32::try_from(tokens - 1).expect("bounded tokens"));
                println!(
                    "decode_probe mode={name} row={row} generated={tokens} \
                     prompt_tokens={} ms_per_token={per_token:.3}",
                    generation.metrics.prompt_tokens
                );
            }
        }
    }
}
