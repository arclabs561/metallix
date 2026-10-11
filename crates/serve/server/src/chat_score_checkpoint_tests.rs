//! Opt-in checkpoint comparison of teacher-forced scores with the receipts
//! greedy decode recorded for the same tokens.
//!
//! Run with `METALLIX_QWEN_MODEL=/path/to/Qwen3-0.6B cargo test --release -p
//! server --all-features score_checkpoint -- --ignored --nocapture
//! --test-threads=1`. Set `METALLIX_SCORE_OUT` to also write every token's
//! values as JSON lines.

use std::{env, fmt::Write as _, fs::OpenOptions, io::Write as _, path::PathBuf};

use qwen::forward::Qwen3Scores;
use serde_json::json;

use super::{
    ChatBackend, ChatMessage, ChatRequest, ChatRole, ChatSession, ResidentChatLimits,
    SamplingRequest, ScoreRequest, ScoreText,
};

const TURNS: usize = 32;
const MAX_TOKENS: u32 = 64;

const TOPICS: [&str; TURNS] = [
    "how a bicycle derailleur shifts",
    "why bread dough rises",
    "what a hash table is",
    "how tides work",
    "the rules of chess castling",
    "how vaccines train the immune system",
    "what a mortgage amortization schedule is",
    "how a refrigerator cools",
    "why the sky is blue",
    "how to tie a bowline knot",
    "what TCP congestion control does",
    "how photosynthesis stores energy",
    "the causes of inflation",
    "how a jet engine produces thrust",
    "what a binary search does",
    "how bees communicate",
    "why ice floats",
    "how compound interest grows",
    "what a compiler does",
    "how earthquakes are measured",
    "the water cycle",
    "how noise-cancelling headphones work",
    "what DNS resolution is",
    "how coffee is decaffeinated",
    "why leaves change color",
    "how a lithium battery charges",
    "what an index fund is",
    "how sourdough starter works",
    "what garbage collection is",
    "how glaciers move",
    "how a piano makes sound",
    "what a Fourier transform does",
];

fn load_session() -> ChatSession {
    let model = env::var_os("METALLIX_QWEN_MODEL")
        .map(PathBuf::from)
        .expect("METALLIX_QWEN_MODEL is required for these ignored checkpoint tests");
    ChatSession::load(&model, ResidentChatLimits::from_mib(4_096, 2_048)).expect("session load")
}

fn quantile(sorted: &[f64], q: f64) -> f64 {
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        clippy::cast_precision_loss,
        reason = "an index into a short sample"
    )]
    let index = ((sorted.len() - 1) as f64 * q).round() as usize;
    sorted[index]
}

/// Scores of the generated tokens, by the route's path and by the two
/// batch-shape controls the floor is measured on.
struct Controls {
    route: Vec<f32>,
    repeat: Vec<f32>,
    chunk_64: Vec<f32>,
    split: Vec<f32>,
    /// The route's top two `(id, logprob)` per token.
    top: Vec<Vec<(i32, f32)>>,
}

fn logprobs(scores: &Qwen3Scores) -> Vec<f32> {
    scores.tokens.iter().map(|token| token.logprob).collect()
}

fn measure_controls(
    session: &ChatSession,
    request: ChatRequest<'_>,
    continuation: &[i32],
) -> Controls {
    let cache_before = session.prefix_cache_stats();
    let scored = |session: &ChatSession| {
        session
            .score(ScoreRequest {
                prompt: ScoreText::Chat(request.conversation()),
                continuation: Some(ScoreText::Ids(continuation)),
                score_prompt: false,
                top_logprobs: 2,
            })
            .expect("score")
    };
    let first = scored(session);
    let repeat = scored(session);
    assert_eq!(
        session.prefix_cache_stats(),
        cache_before,
        "scoring must not query or mutate generation cache"
    );
    let mut executor = session
        .weights
        .resident_chat_executor(session.context_limit, session.kv_budget_bytes)
        .expect("executor");
    let chunk_64 = executor
        .score(&first.ids, first.from, 0, 64)
        .expect("chunked at 64");
    let split = executor
        .score_split(&first.ids, first.from, first.from, 0, 512)
        .expect("two appends");
    Controls {
        route: logprobs(&first.scores),
        repeat: logprobs(&repeat.scores),
        chunk_64: logprobs(&chunk_64),
        split: logprobs(&split),
        top: first.scores.tokens.iter().map(|t| t.top.clone()).collect(),
    }
}

fn qualify(d: &[f64], floor: f64, tokens: usize, rank_flips: &[(usize, usize, f64)], lines: &str) {
    if floor == 0.0 {
        for line in lines.lines() {
            let token: serde_json::Value = serde_json::from_str(line).expect("token receipt");
            if (token["receipt"].as_f64().unwrap() - token["score"].as_f64().unwrap()).abs() > 1e-3
            {
                println!("zero-floor exceedance: {line}");
            }
        }
    }
    let mut sorted = d.to_vec();
    sorted.sort_by(f64::total_cmp);
    #[allow(clippy::cast_precision_loss, reason = "token counts are small")]
    let agreement = 1.0 - rank_flips.len() as f64 / tokens as f64;
    println!(
        "{}",
        json!({
            "turns": TURNS,
            "tokens": tokens,
            "floor_F": floor,
            "d_median": quantile(&sorted, 0.5),
            "d_p99": quantile(&sorted, 0.99),
            "d_max": sorted.last(),
            "rank_agreement": agreement,
            "rank_flips": rank_flips,
        })
    );
    assert!(agreement >= 0.99, "rank agreement below 99%");
    assert!(
        rank_flips.iter().all(|(_, _, gap)| *gap <= floor),
        "rank flip outside measured noise floor"
    );
    if floor == 0.0 {
        assert!(
            quantile(&sorted, 0.99) <= 1e-3,
            "zero-floor fallback p99 exceeded"
        );
    } else {
        assert!(
            quantile(&sorted, 0.5) <= floor,
            "median exceeds noise floor"
        );
        assert!(
            quantile(&sorted, 0.99) <= 4.0 * floor,
            "p99 exceeds four noise floors"
        );
    }
}

/// The procedure and bars are declared before the first run; this test
/// enforces the declared noise-floor and ranking bars.
#[test]
#[ignore = "requires METALLIX_QWEN_MODEL pointing to Qwen3-0.6B on Apple-Silicon Metal"]
fn scored_generated_tokens_match_their_decode_receipts() {
    let mut session = load_session();
    let out = env::var_os("METALLIX_SCORE_OUT").map(PathBuf::from);
    let mut lines = String::new();
    let (mut d, mut floor) = (Vec::new(), 0.0_f64);
    let (mut tokens, mut rank_flips) = (0_usize, Vec::new());
    for (turn, topic) in TOPICS.iter().enumerate() {
        let messages = [ChatMessage::text(
            ChatRole::User,
            format!("In three sentences, explain {topic}."),
        )];
        let mut request = ChatRequest::new(&messages, MAX_TOKENS);
        request.sampling = SamplingRequest::GREEDY;
        request.top_logprobs = Some(0);
        let generated = session.generate(request, &mut |_| Ok(())).expect("turn");
        let receipts: Vec<f64> = generated.logprobs.iter().map(|r| r.logprob).collect();
        let continuation = &generated.generated_token_ids[..receipts.len()];
        let controls = measure_controls(&session, request, continuation);
        assert!(!receipts.is_empty(), "turn {turn} needs generated receipts");
        for values in [
            &controls.route,
            &controls.repeat,
            &controls.chunk_64,
            &controls.split,
        ] {
            assert_eq!(values.len(), receipts.len(), "turn {turn}");
            assert!(values.iter().all(|value| value.is_finite()));
        }
        assert!(receipts.iter().all(|value| value.is_finite()));
        for (index, &receipt) in receipts.iter().enumerate() {
            let route = f64::from(controls.route[index]);
            for control in [&controls.repeat, &controls.chunk_64, &controls.split] {
                floor = floor.max((route - f64::from(control[index])).abs());
            }
            let distance = (route - receipt).abs();
            d.push(distance);
            tokens += 1;
            let top = &controls.top[index];
            assert!(top.len() >= 2, "top-two ranking is required");
            let gap = f64::from(top[0].1 - top[1].1);
            assert!(gap.is_finite() && gap >= 0.0);
            if top[0].0 != continuation[index] {
                rank_flips.push((turn, index, gap));
            }
            writeln!(
                lines,
                "{}",
                json!({"turn":turn,"index":index,"id":continuation[index],"receipt":receipt,"score":route,"repeat":controls.repeat[index],"chunk_64":controls.chunk_64[index],"split":controls.split[index],"top2_gap":gap})
            )
            .expect("writing to a String cannot fail");
        }
    }
    if let Some(out) = out {
        OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&out)
            .expect("create fresh per-token receipt")
            .write_all(lines.as_bytes())
            .expect("write per-token values");
    }
    qualify(&d, floor, tokens, &rank_flips, &lines);
}
