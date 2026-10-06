//! Opt-in checkpoint checks for cross-request prefix reuse.
//!
//! The oracle for every hit is the same greedy request on an empty cache.
//! Run with `METALLIX_QWEN_MODEL=/path/to/Qwen3-0.6B cargo test --release -p
//! server --features metal prefix_cache::checkpoint -- --ignored --nocapture
//! --test-threads=1` to also print TTFT.

use std::{env, fmt::Write as _, path::PathBuf};

use crate::chat_generation::{
    ChatGeneration, ChatMessage, ChatRequest, ChatRole, ChatSession, ResidentChatLimits,
};

const MAX_TOKENS: u32 = 24;

fn load_session() -> Option<ChatSession> {
    let Some(model) = env::var_os("METALLIX_QWEN_MODEL").map(PathBuf::from) else {
        eprintln!("skipping: METALLIX_QWEN_MODEL is not set");
        return None;
    };
    // About 1 GiB per conversation here (preamble plus history entries): room
    // for several, so only the eviction test sees eviction.
    let limits = ResidentChatLimits::from_mib(8_192, 4_096).with_prefix_cache_mib(8_192);
    Some(ChatSession::load(&model, limits).expect("load"))
}

/// A stable operator preamble of roughly 3,000 tokens.
fn preamble(variant: &str) -> String {
    let mut text = String::from("You are the support assistant for a hardware store.\n");
    for rule in 0..60 {
        writeln!(
            text,
            "Rule {rule}: when a customer asks about aisle {rule}, answer briefly, name the \
             aisle, mention that returns within 30 days need a receipt, and never invent stock."
        )
        .expect("writing to a String");
    }
    // The variant sits at the end, so a changed preamble shares every earlier token.
    writeln!(text, "Sign every reply as {variant}.").expect("writing to a String");
    text
}

fn generate(session: &mut ChatSession, messages: &[ChatMessage]) -> ChatGeneration {
    generate_salted(session, messages, None)
}

fn generate_salted(
    session: &mut ChatSession,
    messages: &[ChatMessage],
    cache_salt: Option<&str>,
) -> ChatGeneration {
    let request = ChatRequest {
        cache_salt,
        ..ChatRequest::new(messages, MAX_TOKENS)
    };
    session
        .generate(request, &mut |_| Ok(()))
        .expect("generation")
}

fn conversation(system: &str, user: &str) -> Vec<ChatMessage> {
    vec![
        ChatMessage::text(ChatRole::System, system),
        ChatMessage::text(ChatRole::User, user),
    ]
}

#[test]
#[ignore = "requires METALLIX_QWEN_MODEL and a local Apple-Silicon Metal checkpoint"]
fn checkpoint_cache_write_tokens_count_only_accepted_prefixes() {
    let model = env::var_os("METALLIX_QWEN_MODEL")
        .map(PathBuf::from)
        .expect("explicit checkpoint test requires METALLIX_QWEN_MODEL");
    let limits = ResidentChatLimits::from_mib(128, 64).with_prefix_cache_mib(64);
    let mut session = ChatSession::load(&model, limits).expect("load");
    let messages = conversation("Answer briefly.", "Say hello.");
    let request = ChatRequest::new(&messages, 1);
    let accepted = session
        .generate(request, &mut |_| Ok(()))
        .expect("generation");
    assert!(accepted.metrics.cache_write_tokens > 0);
    assert!(accepted.metrics.cache_write_tokens < accepted.metrics.prompt_tokens);
    assert_eq!(session.prefix_cache_stats().entries, 2);
    let repeated = session
        .generate(request, &mut |_| Ok(()))
        .expect("generation");
    assert_eq!(repeated.metrics.cache_write_tokens, 0, "already cached");
    assert_eq!(repeated.generated_token_ids, accepted.generated_token_ids);
    for budget in [0, 1] {
        session.reset_prefix_cache(budget);
        let refused = session
            .generate(request, &mut |_| Ok(()))
            .expect("generation");
        assert_eq!(refused.metrics.cache_write_tokens, 0, "budget {budget}");
        assert_eq!(session.prefix_cache_stats().entries, 0);
        assert_eq!(refused.generated_token_ids, accepted.generated_token_ids);
    }
}

#[test]
#[ignore = "requires METALLIX_QWEN_MODEL and a local Apple-Silicon Metal checkpoint"]
fn checkpoint_cache_creation_counts_only_new_prompt_positions_after_a_hit() {
    let model = env::var_os("METALLIX_QWEN_MODEL")
        .map(PathBuf::from)
        .expect("explicit checkpoint test requires METALLIX_QWEN_MODEL");
    let limits = ResidentChatLimits::from_mib(256, 64).with_prefix_cache_mib(64);
    let mut session = ChatSession::load(&model, limits).expect("load");
    let first = conversation("Answer briefly and accurately.", "Say hello.");
    let mut request = ChatRequest::new(&first, 1);
    request.sampling = crate::chat_generation::SamplingRequest::GREEDY;
    session
        .generate(request, &mut |_| Ok(()))
        .expect("seed cache");
    let extended = conversation(
        "Answer briefly and accurately.",
        "Name three common colors, then say hello politely.",
    );
    let mut request = ChatRequest::new(&extended, 1);
    request.sampling = crate::chat_generation::SamplingRequest::GREEDY;
    let full = session.render(request).expect("render").ids;
    let accepted_prefix = session
        .format
        .prompt(request.conversation(), false)
        .expect("prefix")
        .ids;
    assert!(full.starts_with(&accepted_prefix));
    let hit = session
        .generate(request, &mut |_| Ok(()))
        .expect("hit and extend");
    let metrics = &hit.metrics;
    assert!(metrics.cached_prompt_tokens > 0);
    assert!(metrics.cache_write_tokens > 0);
    assert_eq!(
        metrics.cache_write_tokens,
        accepted_prefix.len() - metrics.cached_prompt_tokens
    );
    assert!(metrics.cached_prompt_tokens + metrics.cache_write_tokens <= metrics.prompt_tokens);
    session.reset_prefix_cache(0);
    let fresh = session
        .generate(request, &mut |_| Ok(()))
        .expect("uncached oracle");
    assert_eq!(fresh.metrics.cache_write_tokens, 0);
    assert_eq!(fresh.generated_token_ids, hit.generated_token_ids);
}

#[test]
#[ignore = "requires METALLIX_QWEN_MODEL and a local Apple-Silicon Metal checkpoint"]
fn checkpoint_prefix_hits_are_greedy_identical_and_changed_preambles_miss() {
    let Some(mut session) = load_session() else {
        return;
    };
    let system = preamble("Ada");
    let first = generate(
        &mut session,
        &conversation(&system, "Where are the hammers?"),
    );
    assert_eq!(first.metrics.cached_prompt_tokens, 0, "cold cache misses");

    let second_messages = conversation(&system, "Do you sell ladders?");
    let hit = generate(&mut session, &second_messages);
    let preamble_tokens = hit.metrics.cached_prompt_tokens;
    assert!(
        preamble_tokens > 2_000 && preamble_tokens < hit.metrics.prompt_tokens,
        "the shared system preamble is reused: {preamble_tokens} of {}",
        hit.metrics.prompt_tokens
    );

    // The next turn of the same conversation extends the cached history.
    let mut next_turn = second_messages.clone();
    next_turn.push(ChatMessage::text(ChatRole::Assistant, hit.text.clone()));
    next_turn.push(ChatMessage::text(ChatRole::User, "What about paint?"));
    let follow_up = generate(&mut session, &next_turn);
    assert!(
        follow_up.metrics.cached_prompt_tokens > preamble_tokens,
        "conversation history is reused beyond the preamble"
    );

    // A preamble changed only in its last line never hits.
    let changed = generate(
        &mut session,
        &conversation(&preamble("Grace"), "Do you sell ladders?"),
    );
    assert_eq!(changed.metrics.cached_prompt_tokens, 0);

    // Greedy oracle: the same requests on an empty, disabled cache.
    session.reset_prefix_cache(0);
    for (messages, cached) in [(&second_messages, &hit), (&next_turn, &follow_up)] {
        let fresh = generate(&mut session, messages);
        assert_eq!(fresh.metrics.cached_prompt_tokens, 0);
        assert_eq!(
            fresh.generated_token_ids, cached.generated_token_ids,
            "a cache hit must not change greedy output"
        );
    }
}

#[test]
#[ignore = "requires METALLIX_QWEN_MODEL and a local Apple-Silicon Metal checkpoint"]
fn checkpoint_prefix_budget_evicts_least_recently_used_preamble() {
    let Some(mut session) = load_session() else {
        return;
    };
    let first_system = preamble("Ada");
    generate(&mut session, &conversation(&first_system, "Hammers?"));
    let stats = session.prefix_cache_stats();
    let one_conversation = stats.bytes;
    assert_eq!(stats.entries, 2, "preamble and history entries");
    // Room for one conversation's entries plus a little slack, not two.
    session.reset_prefix_cache(one_conversation + one_conversation / 4);
    generate(&mut session, &conversation(&first_system, "Hammers?"));
    let second_system = preamble("Grace");
    generate(&mut session, &conversation(&second_system, "Hammers?"));
    let stats = session.prefix_cache_stats();
    assert!(stats.bytes <= one_conversation + one_conversation / 4);
    assert!(stats.evictions >= 2, "both Ada entries were evicted");
    // Check the kept entry first: the evicted request's miss inserts again.
    let kept = generate(&mut session, &conversation(&second_system, "Ladders?"));
    assert!(kept.metrics.cached_prompt_tokens > 0, "Grace was kept");
    let evicted = generate(&mut session, &conversation(&first_system, "Ladders?"));
    assert_eq!(evicted.metrics.cached_prompt_tokens, 0, "Ada was evicted");
}

#[test]
#[ignore = "requires METALLIX_QWEN_MODEL and a local Apple-Silicon Metal checkpoint"]
fn checkpoint_prefix_hit_time_to_first_token() {
    let Some(mut session) = load_session() else {
        return;
    };
    let system = preamble("Ada");
    // Warm kernels for both the full-prefill and the chunk path.
    generate(&mut session, &conversation(&system, "Warm up."));
    generate(&mut session, &conversation(&system, "Warm up again."));
    for round in 0..3 {
        session.reset_prefix_cache(0);
        let miss = generate(&mut session, &conversation(&system, "Do you sell ladders?"));
        session.reset_prefix_cache(1 << 32);
        generate(
            &mut session,
            &conversation(&system, "Where are the hammers?"),
        );
        let hit = generate(&mut session, &conversation(&system, "Do you sell ladders?"));
        assert_eq!(miss.generated_token_ids, hit.generated_token_ids);
        eprintln!(
            "round={round} prompt_tokens={} cached_tokens={} miss_prefill_ms={:.1} \
             hit_prefill_ms={:.1} miss_ttft_ms={:.1} hit_ttft_ms={:.1}",
            hit.metrics.prompt_tokens,
            hit.metrics.cached_prompt_tokens,
            miss.metrics.prefill_ms,
            hit.metrics.prefill_ms,
            miss.metrics.time_to_first_token_ms.unwrap_or(f64::NAN),
            hit.metrics.time_to_first_token_ms.unwrap_or(f64::NAN),
        );
    }
}

#[test]
#[ignore = "requires METALLIX_QWEN_MODEL and a local Apple-Silicon Metal checkpoint"]
fn checkpoint_prefix_salts_isolate_identical_prompts() {
    let Some(mut session) = load_session() else {
        return;
    };
    let system = preamble("Ada");
    let first = conversation(&system, "Where are the hammers?");
    let second = conversation(&system, "Do you sell ladders?");
    generate_salted(&mut session, &first, Some("tenant-a"));
    // Same prompt, other tenant or no salt: never tenant A's entries.
    for salt in [Some("tenant-b"), None] {
        let other = generate_salted(&mut session, &second, salt);
        assert_eq!(other.metrics.cached_prompt_tokens, 0, "{salt:?} missed");
    }
    // Tenant A still hits its own entry, and B now hits the one B created.
    for salt in [Some("tenant-a"), Some("tenant-b")] {
        let own = generate_salted(&mut session, &second, salt);
        assert!(own.metrics.cached_prompt_tokens > 0, "{salt:?} hit");
    }
    let stats = session.prefix_cache_stats();
    assert_eq!((stats.queries, stats.hits), (5, 2));
}

#[test]
#[ignore = "requires METALLIX_QWEN_MODEL and a local Apple-Silicon Metal checkpoint"]
fn checkpoint_prefix_snapshot_copy_time() {
    let Some(mut session) = load_session() else {
        return;
    };
    let system = preamble("Ada");
    let messages = conversation(&system, "Do you sell ladders?");
    // Warm the prefill and copy kernels once.
    generate(&mut session, &messages);
    for round in 0..3 {
        session.reset_prefix_cache(1 << 32);
        let request = ChatRequest::new(&messages, MAX_TOKENS);
        let input_ids = session.render(request).expect("render").ids;
        let (executor, _, _) = super::prefill(
            &session.weights,
            &mut session.prefix_cache,
            session.context_limit,
            session.kv_budget_bytes,
            request,
            &input_ids,
        )
        .expect("prefill");
        let started = std::time::Instant::now();
        super::remember(
            &session.weights,
            &mut session.prefix_cache,
            &session.format,
            &executor,
            request,
            &input_ids,
        )
        .expect("store");
        let store_ms = started.elapsed().as_secs_f64() * 1e3;
        let stats = session.prefix_cache.stats();
        assert_eq!(stats.entries, 2);
        eprintln!(
            "round={round} prompt_tokens={} entries={} stored_mib={} store_ms={store_ms:.1}",
            input_ids.len(),
            stats.entries,
            stats.bytes >> 20,
        );
    }
}
