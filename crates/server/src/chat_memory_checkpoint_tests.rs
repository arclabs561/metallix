//! Opt-in check that the serial chat path's MLX memory stays bounded.
//!
//! Run with `METALLIX_QWEN_MODEL=/path/to/Qwen3-0.6B cargo test --release -p
//! server --features metal memory_checkpoint -- --ignored --nocapture
//! --test-threads=1`. Add `METALLIX_SERIAL_TEST_UNCAPPED=1` to run without
//! the allocator cache cap and see the check fail.

use std::{env, path::PathBuf};

use super::{ChatMessage, ChatRequest, ChatRole, ChatSession, ResidentChatLimits};
use crate::gpu;

/// Bytes MLX may hold above the post-warmup level beyond the prefix cache's
/// own budget once a turn has finished.
const ACTIVE_GROWTH_BOUND: u64 = 256 << 20;

/// MLX trims its cache to the limit as buffers return to it, so the cache
/// can sit slightly above the limit between trims.
const CACHE_OVERSHOOT_BOUND: u64 = 64 << 20;

/// Turns one after another whose prompt and output lengths keep changing, as
/// a serving child sees them. After each round MLX's active bytes must stay
/// within the prefix cache's budget of the post-warmup level, and its
/// allocator cache under the cap.
#[test]
#[ignore = "requires METALLIX_QWEN_MODEL pointing to Qwen3-0.6B on Apple-Silicon Metal"]
fn serial_memory_stays_bounded_across_changing_shapes() {
    let model = env::var_os("METALLIX_QWEN_MODEL")
        .map(PathBuf::from)
        .expect("METALLIX_QWEN_MODEL is required for these ignored checkpoint tests");
    let uncapped = env::var_os("METALLIX_SERIAL_TEST_UNCAPPED").is_some();
    let cap = if uncapped {
        None
    } else {
        Some(gpu::cap_cache().expect("cache cap") as u64)
    };
    let mut limits = ResidentChatLimits::from_mib(4_096, 2_048);
    if let Some(mib) = env::var_os("METALLIX_SERIAL_TEST_PREFIX_MIB") {
        let mib = mib.to_str().and_then(|mib| mib.parse().ok());
        limits = limits.with_prefix_cache_mib(mib.expect("prefix cache MiB"));
    }
    let mut session = ChatSession::load(&model, limits).expect("session load");
    let words = [
        "river", "stone", "lamp", "orbit", "cedar", "quartz", "harbor",
    ];
    let round = |session: &mut ChatSession, round: usize| {
        // 3 to 9 turns, 20 to ~900 words, 8 to 64 output tokens.
        for turn in 0..3 + (round * 5) % 7 {
            let length = 20 + (round * 131 + turn * 197) % 880;
            let max_tokens = 8 + u32::try_from((round * 7 + turn * 11) % 57).expect("small");
            let prompt = (0..length)
                .map(|word| words[(word + turn + round) % words.len()])
                .collect::<Vec<_>>()
                .join(" ");
            let messages = [ChatMessage::text(ChatRole::User, prompt)];
            session
                .generate(ChatRequest::new(&messages, max_tokens), &mut |_| Ok(()))
                .expect("generation");
        }
    };
    round(&mut session, 0);
    let base = gpu::Memory::read().expect("MLX memory");
    let mut worst = (0_u64, 0_u64);
    for index in 1..=16 {
        round(&mut session, index);
        let memory = gpu::Memory::read().expect("MLX memory");
        let prefix = session.prefix_cache_stats();
        println!(
            "serial_memory round={index} uncapped={uncapped} active_bytes={} cache_bytes={} \
             peak_bytes={} base_active={} base_cache={} prefix_entries={} prefix_bytes={}",
            memory.active,
            memory.cache,
            memory.peak,
            base.active,
            base.cache,
            prefix.entries,
            prefix.bytes
        );
        worst = (worst.0.max(memory.active), worst.1.max(memory.cache));
    }
    assert!(
        worst.0 <= base.active + limits.prefix_cache_bytes() + ACTIVE_GROWTH_BOUND,
        "active bytes grew from {} to {}",
        base.active,
        worst.0
    );
    let limit = cap.unwrap_or(gpu::cache_limit_bytes() as u64);
    assert!(
        worst.1 <= limit + CACHE_OVERSHOOT_BOUND,
        "allocator cache reached {} bytes, above the {limit}-byte cap plus overshoot",
        worst.1
    );
}
