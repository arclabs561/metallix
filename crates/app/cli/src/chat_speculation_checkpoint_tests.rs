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
/// The decode checkpoint's bound for receipts settled from GPU candidates.
const LOGPROB_TOLERANCE: f64 = 1e-4;
/// MLX active plus cached memory allowed after any turn: Qwen3-0.6B BF16
/// weights (~1.2 GiB), K/V at a 4,096-token context and the 2,048 MiB prefix
/// cache fit well inside it.
const MLX_BOUND_BYTES: u64 = 8 * 1024 * 1024 * 1024;

/// Fails the test when MLX holds more than [`MLX_BOUND_BYTES`], so an opt-in
/// run stops itself instead of filling the GPU.
fn assert_mlx_within_bound() {
    let held = crate::gpu::held_bytes().expect("MLX reports its memory");
    assert!(
        held <= MLX_BOUND_BYTES,
        "MLX holds {held} bytes after a speculation checkpoint turn, above the {MLX_BOUND_BYTES}-byte bound"
    );
}

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

fn load_session() -> Option<ChatSession> {
    let model = env::var_os("METALLIX_QWEN_MODEL").map(PathBuf::from)?;
    Some(
        ChatSession::load(&model, ResidentChatLimits::from_mib(4_096, 2_048))
            .expect("session load"),
    )
}

/// Every decode mode the turn loop has (GPU-greedy, host rows for logprobs,
/// GPU-candidate sampling, full-row sampling) runs with speculation off and
/// automatic. A verbatim edit must accept drafts and record them; outputs
/// are compared and the first difference is printed, since a BF16 verify
/// may break a near-tie the other way.
#[test]
#[ignore = "requires METALLIX_QWEN_MODEL pointing to Qwen3-0.6B on Apple-Silicon Metal"]
fn speculation_accepts_edit_drafts_in_every_decode_mode() {
    let Some(mut session) = load_session() else {
        eprintln!("skipping: METALLIX_QWEN_MODEL is not set");
        return;
    };
    let messages = [ChatMessage::text(ChatRole::User, EDIT)];
    let seeded = |top_k| SamplingRequest {
        temperature: Some(0.7),
        top_p: Some(0.95),
        top_k,
        seed: Some(11),
        sampler: None,
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
            assert_mlx_within_bound();
            let turn = session.generate(request, &mut |_| Ok(())).expect("turn");
            assert_mlx_within_bound();
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
    let Some(mut session) = load_session() else {
        eprintln!("skipping: METALLIX_QWEN_MODEL is not set");
        return;
    };
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
            assert_mlx_within_bound();
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
    let Some(mut session) = load_session() else {
        eprintln!("skipping: METALLIX_QWEN_MODEL is not set");
        return;
    };
    let messages = [ChatMessage::text(ChatRole::User, EDIT)];
    let mut runs = Vec::new();
    for speculation in [SpeculationRequest::Disabled, SpeculationRequest::Automatic] {
        let mut request = ChatRequest::new(&messages, 400);
        request.ignore_eos = true;
        request.speculation = speculation;
        let turn = session.generate(request, &mut |_| Ok(())).expect("turn");
        assert_mlx_within_bound();
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

/// A seeded sampled `top_k` turn verifies with GPU picks and settles each
/// kept row from its candidates, and must emit exactly what the host-row
/// verify emits from the same rows: same tokens and same logprob receipts.
/// Both paths score the same forward, so unlike speculation on vs off there
/// is no BF16 near-tie excuse. The edit prompt makes drafts that are partly
/// rejected, so verifies end mid-draft and the per-row variates must stay
/// aligned with the stream across those rejections.
#[test]
#[ignore = "requires METALLIX_QWEN_MODEL pointing to Qwen3-0.6B on Apple-Silicon Metal"]
fn sampled_gpu_verify_matches_host_row_verify() {
    use std::sync::atomic::Ordering;

    use super::speculation::test_hooks::{GPU_VERIFIES, HOST_ROW_VERIFY};

    let Some(mut session) = load_session() else {
        eprintln!("skipping: METALLIX_QWEN_MODEL is not set");
        return;
    };
    let messages = [ChatMessage::text(ChatRole::User, EDIT)];
    for (seed, top_logprobs) in [(3, None), (17, Some(2)), (29, None)] {
        let mut runs = Vec::new();
        for host_rows in [true, false] {
            HOST_ROW_VERIFY.store(host_rows, Ordering::Relaxed);
            let before = GPU_VERIFIES.load(Ordering::Relaxed);
            let mut request = ChatRequest::new(&messages, MAX_TOKENS);
            request.sampling = SamplingRequest {
                temperature: Some(0.8),
                top_p: Some(0.95),
                top_k: Some(20),
                seed: Some(seed),
                sampler: None,
            };
            request.top_logprobs = top_logprobs;
            request.speculation = SpeculationRequest::Automatic;
            let turn = session.generate(request, &mut |_| Ok(())).expect("turn");
            assert_mlx_within_bound();
            let gpu_verifies = GPU_VERIFIES.load(Ordering::Relaxed) - before;
            let receipt = turn.metrics.speculation.expect("the edit turn verifies");
            assert_eq!(
                gpu_verifies == 0,
                host_rows,
                "seed {seed}: GPU path taken {gpu_verifies} times with host_rows={host_rows}"
            );
            assert!(
                receipt.accepted_tokens > 0 && receipt.accepted_tokens < receipt.drafted_tokens,
                "seed {seed}: needs accepted and rejected drafts: {receipt:?}"
            );
            println!(
                "sampled_verify seed={seed} host_rows={host_rows} tokens={} receipt={receipt:?}",
                turn.generated_token_ids.len()
            );
            runs.push(turn);
        }
        HOST_ROW_VERIFY.store(false, Ordering::Relaxed);
        let (host, gpu) = (&runs[0], &runs[1]);
        assert_eq!(
            host.generated_token_ids, gpu.generated_token_ids,
            "seed {seed}"
        );
        assert_eq!(
            host.metrics.speculation, gpu.metrics.speculation,
            "seed {seed}"
        );
        // Receipts settled from candidates use the GPU's f32 sum of shifted
        // exponentials, as decode's do, so they match the host's f64 pass
        // within the decode checkpoint's tolerance rather than bit for bit.
        assert_eq!(host.logprobs.len(), gpu.logprobs.len(), "seed {seed}");
        for (step, (gpu, host)) in gpu.logprobs.iter().zip(&host.logprobs).enumerate() {
            assert_eq!(gpu.bytes, host.bytes, "seed {seed} receipt {step}");
            assert!(
                (gpu.logprob - host.logprob).abs() < LOGPROB_TOLERANCE,
                "seed {seed} receipt {step}"
            );
            let tops = gpu.top_logprobs.iter().zip(&host.top_logprobs);
            assert_eq!(gpu.top_logprobs.len(), host.top_logprobs.len());
            for (gpu, host) in tops {
                assert_eq!(gpu.bytes, host.bytes, "seed {seed} receipt {step}");
                assert!((gpu.logprob - host.logprob).abs() < LOGPROB_TOLERANCE);
            }
        }
    }
}

/// Whether speculation widens the sampled-vs-greedy per-token gap. Arms
/// alternate round by round so machine load drifts over all of them alike.
/// Prints per-token decode cost and draft counts.
#[test]
#[ignore = "requires METALLIX_QWEN_MODEL pointing to Qwen3-0.6B on Apple-Silicon Metal"]
fn per_token_cost_by_pick_mode_and_speculation() {
    let Some(mut session) = load_session() else {
        eprintln!("skipping: METALLIX_QWEN_MODEL is not set");
        return;
    };
    let story = [ChatMessage::text(
        ChatRole::User,
        "Write a short original story about a lighthouse keeper who collects lost keys.",
    )];
    let edit = [ChatMessage::text(ChatRole::User, EDIT)];
    let sampled = SamplingRequest {
        temperature: Some(0.6),
        top_p: Some(0.95),
        top_k: Some(20),
        seed: Some(5),
        sampler: None,
    };
    let arms = [
        (
            "greedy",
            SamplingRequest::GREEDY,
            SpeculationRequest::Disabled,
        ),
        (
            "greedy",
            SamplingRequest::GREEDY,
            SpeculationRequest::Automatic,
        ),
        ("sampled", sampled, SpeculationRequest::Disabled),
        ("sampled", sampled, SpeculationRequest::Automatic),
    ];
    for (prompt, messages) in [("story", &story[..]), ("edit", &edit[..])] {
        let mut per_token = vec![Vec::new(); arms.len()];
        let mut receipts = vec![None; arms.len()];
        for round in 0..6 {
            for (arm, &(_, sampling, speculation)) in arms.iter().enumerate() {
                let mut request = ChatRequest::new(messages, MAX_TOKENS);
                request.sampling = sampling;
                request.speculation = speculation;
                let turn = session.generate(request, &mut |_| Ok(())).expect("turn");
                assert_mlx_within_bound();
                // The first round warms the graph caches.
                if round > 0 {
                    #[allow(clippy::cast_precision_loss, reason = "small token counts")]
                    let steps = turn.generated_token_ids.len().saturating_sub(1).max(1) as f64;
                    per_token[arm].push(turn.metrics.decode_total_ms / steps);
                    receipts[arm] = turn.metrics.speculation;
                }
            }
        }
        for (arm, &(pick, _, speculation)) in arms.iter().enumerate() {
            per_token[arm].sort_by(f64::total_cmp);
            println!(
                "pick_cost prompt={prompt} pick={pick} speculation={speculation:?} median_ms_per_token={:.3} rates={:?} receipt={:?}",
                per_token[arm][2], per_token[arm], receipts[arm]
            );
        }
    }
}

/// The verify rule carries one variate per scored row, `i` draws ahead,
/// and a grammar-constrained turn gets no GPU verify rule at all: its rows
/// need the grammar mask, which the GPU pick does not apply, so a verify
/// (if schema turns ever speculate) falls back to whole host rows.
#[cfg(feature = "structured-output")]
#[test]
#[ignore = "requires METALLIX_QWEN_MODEL pointing to Qwen3-0.6B on Apple-Silicon Metal"]
fn verify_rule_has_one_variate_per_row_and_refuses_a_grammar() {
    use qwen::forward::Qwen3Selection;
    use serde_json::json;

    use super::{GenerationDeadline, TurnStart};

    let Some(session) = load_session() else {
        eprintln!("skipping: METALLIX_QWEN_MODEL is not set");
        return;
    };
    let messages = [ChatMessage::text(ChatRole::User, EDIT)];
    let schema = json!({"type": "object", "properties": {"name": {"type": "string"}}});
    for constrained in [false, true] {
        let mut request = ChatRequest::new(&messages, 32);
        request.sampling = SamplingRequest {
            temperature: Some(0.8),
            top_p: Some(0.95),
            top_k: Some(20),
            seed: Some(3),
            sampler: None,
        };
        if constrained {
            request.json_schema = Some(&schema);
        }
        let (_, turn, _) = TurnStart::prepare(
            session.turn_model(),
            request,
            GenerationDeadline::unlimited(),
        )
        .expect("prepare")
        .into_parts();
        let format = &session.format;
        let rule = turn.gpu_rule_rows(format, 0, 4);
        if constrained {
            assert!(rule.is_none(), "a grammar turn must verify from host rows");
            continue;
        }
        let Some(Qwen3Selection::TopKNucleus { uniforms, .. }) = rule.map(|rule| rule.selection)
        else {
            panic!("a top_k turn verifies with a sampling rule");
        };
        let ahead: Vec<f64> = (0..4)
            .map(
                |row| match turn.gpu_rule(format, row).map(|rule| rule.selection) {
                    Some(Qwen3Selection::TopKNucleus { uniforms, .. }) => uniforms[0],
                    _ => panic!("decode rule"),
                },
            )
            .collect();
        assert_eq!(uniforms, ahead);
    }
}
