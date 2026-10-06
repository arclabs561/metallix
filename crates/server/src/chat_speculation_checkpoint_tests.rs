//! Opt-in checkpoint checks for speculative decoding in a whole chat turn.
//!
//! Run with `METALLIX_QWEN_MODEL=/path/to/Qwen3-0.6B cargo test --release -p
//! server --features metal speculation_checkpoint -- --ignored --nocapture
//! --test-threads=1`.

use std::{env, path::PathBuf};

use engine::speculative::SpeculationRequest;

use super::{
    ChatFinishReason, ChatMessage, ChatRequest, ChatRole, ChatSession, ResidentChatLimits,
    SamplingRequest,
};

const MAX_TOKENS: u32 = 160;

const EDIT: &str = r#"Rename the function `parse_header` to `read_header` everywhere in this file and return the complete file, unchanged otherwise.

```python
import struct


def parse_header(data: bytes) -> dict:
    magic, version, count = struct.unpack_from("<4sHH", data, 0)
    if magic != b"MTLX":
        raise ValueError("bad magic")
    return {"version": version, "count": count}


def parse_records(data: bytes) -> list:
    header = parse_header(data)
    records = []
    offset = 8
    for _ in range(header["count"]):
        key, value = struct.unpack_from("<II", data, offset)
        records.append((key, value))
        offset += 8
    return records
```"#;

fn load_session() -> ChatSession {
    let model = env::var_os("METALLIX_QWEN_MODEL")
        .map(PathBuf::from)
        .expect("METALLIX_QWEN_MODEL is required for these ignored checkpoint tests");
    ChatSession::load(&model, ResidentChatLimits::from_mib(4_096, 2_048)).expect("session load")
}

/// Every decode mode the turn loop has (GPU-greedy, host rows for logprobs,
/// GPU-candidate sampling, full-row sampling) runs with speculation off and
/// automatic. A verbatim edit must accept drafts and record them; outputs
/// are compared and the first difference is printed, since a BF16 verify
/// may break a near-tie the other way.
#[test]
#[ignore = "requires METALLIX_QWEN_MODEL pointing to Qwen3-0.6B on Apple-Silicon Metal"]
fn speculation_accepts_edit_drafts_in_every_decode_mode() {
    let mut session = load_session();
    let messages = [ChatMessage::text(ChatRole::User, EDIT)];
    let seeded = |top_k| SamplingRequest {
        temperature: Some(0.7),
        top_p: Some(0.95),
        top_k,
        seed: Some(11),
    };
    let cases = [
        ("greedy", SamplingRequest::GREEDY, None),
        ("greedy_top3", SamplingRequest::GREEDY, Some(3)),
        ("sampled_top_k", seeded(Some(20)), None),
        ("sampled_full_row", seeded(None), None),
    ];
    for (name, sampling, top_logprobs) in cases {
        let mut runs = Vec::new();
        for speculation in [SpeculationRequest::Disabled, SpeculationRequest::Automatic] {
            let mut request = ChatRequest::new(&messages, MAX_TOKENS);
            request.sampling = sampling;
            request.top_logprobs = top_logprobs;
            request.speculation = speculation;
            // Warm the graph caches, then measure.
            let _ = session.generate(request, &mut |_| Ok(())).expect("warmup");
            let turn = session.generate(request, &mut |_| Ok(())).expect("turn");
            let tokens = turn.generated_token_ids.len();
            #[allow(clippy::cast_precision_loss, reason = "small token counts")]
            let rate = (tokens.saturating_sub(1)) as f64 / turn.metrics.decode_total_ms * 1e3;
            println!(
                "speculation_turn case={name} speculation={speculation:?} tokens={tokens} \
                 decode_total_ms={:.1} tok_s={rate:.1} receipt={:?} logprobs={}",
                turn.metrics.decode_total_ms,
                turn.metrics.speculation,
                turn.logprobs.len(),
            );
            runs.push(turn);
        }
        let (off, auto) = (&runs[0], &runs[1]);
        assert!(
            off.metrics.speculation.is_none(),
            "{name}: off must not verify"
        );
        let receipt = auto
            .metrics
            .speculation
            .unwrap_or_else(|| panic!("{name}: automatic speculation never verified"));
        assert!(receipt.accepted_tokens > 0, "{name}: {receipt:?}");
        if top_logprobs.is_some() {
            // One receipt per visible token, verified or decoded.
            let stop = usize::from(auto.finish_reason == ChatFinishReason::Eos);
            assert_eq!(
                auto.logprobs.len() + stop,
                auto.generated_token_ids.len(),
                "{name}"
            );
        }
        let divergence = off
            .generated_token_ids
            .iter()
            .zip(&auto.generated_token_ids)
            .position(|(left, right)| left != right);
        println!("speculation_turn case={name} first_divergence={divergence:?}");
    }
}

/// Automatic speculation on a turn with little to copy: drafts are rarer,
/// and a verify that discards a queued step must not make it slower than
/// pipelined greedy decode. The two arms alternate so machine load drifts
/// over both alike. Prints both rates; asserts nothing about speed.
#[test]
#[ignore = "requires METALLIX_QWEN_MODEL pointing to Qwen3-0.6B on Apple-Silicon Metal"]
fn speculation_on_original_writing_reports_its_cost() {
    let mut session = load_session();
    let messages = [ChatMessage::text(
        ChatRole::User,
        "Write a short original story about a lighthouse keeper who collects lost keys.",
    )];
    let arms = [SpeculationRequest::Disabled, SpeculationRequest::Automatic];
    let mut rates = [Vec::new(), Vec::new()];
    let mut receipt = None;
    for repeat in 0..6 {
        for (arm, &speculation) in arms.iter().enumerate() {
            let mut request = ChatRequest::new(&messages, MAX_TOKENS);
            request.speculation = speculation;
            let turn = session.generate(request, &mut |_| Ok(())).expect("turn");
            // The first round warms the graph caches.
            if repeat > 0 {
                #[allow(clippy::cast_precision_loss, reason = "small token counts")]
                let rate = turn.generated_token_ids.len().saturating_sub(1) as f64
                    / turn.metrics.decode_total_ms
                    * 1e3;
                rates[arm].push(rate);
            }
            if arm == 1 {
                receipt = turn.metrics.speculation;
            }
        }
    }
    for (arm, speculation) in arms.iter().enumerate() {
        rates[arm].sort_by(f64::total_cmp);
        println!(
            "speculation_turn case=original speculation={speculation:?} median_tok_s={:.1} rates={:?}",
            rates[arm][2], rates[arm]
        );
    }
    println!("speculation_turn case=original receipt={receipt:?}");
}

/// Under `ignore_eos` a verified end-of-turn token is ordinary output, so a
/// speculative turn runs to its limit like a decoded one.
#[test]
#[ignore = "requires METALLIX_QWEN_MODEL pointing to Qwen3-0.6B on Apple-Silicon Metal"]
fn speculation_runs_to_the_limit_under_ignore_eos() {
    let mut session = load_session();
    let messages = [ChatMessage::text(ChatRole::User, EDIT)];
    let mut runs = Vec::new();
    for speculation in [SpeculationRequest::Disabled, SpeculationRequest::Automatic] {
        let mut request = ChatRequest::new(&messages, 400);
        request.ignore_eos = true;
        request.speculation = speculation;
        let turn = session.generate(request, &mut |_| Ok(())).expect("turn");
        assert_eq!(turn.generated_token_ids.len(), 400, "{speculation:?}");
        assert_eq!(
            turn.finish_reason,
            ChatFinishReason::Length,
            "{speculation:?}"
        );
        runs.push(turn);
    }
    assert!(runs[1].metrics.speculation.is_some());
    let divergence = runs[0]
        .generated_token_ids
        .iter()
        .zip(&runs[1].generated_token_ids)
        .position(|(left, right)| left != right);
    println!("speculation_turn case=ignore_eos first_divergence={divergence:?}");
}
