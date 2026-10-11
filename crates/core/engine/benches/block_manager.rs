//! KV block manager operations at serving sizes.
//!
//! Pool: 256 slabs of 32 blocks of 16 tokens (131,072 tokens), larger than
//! any default `--kv-budget-mib` pool, so the free queue and prefix map are
//! not trivially small. Prompts are 2,048 tokens (128 blocks).
//! - `hash_block`: one SHA-256 block key.
//! - `lookup_hit_2k`: a fully cached 2K prompt (127 block keys and map hits).
//! - `admit_commit_free_2k`: admit a 2K prompt with no hit, commit (hashing
//!   128 blocks when caching is on), then free.
//! - `decode_step`: one decode step for `seqs` running sequences: allocate one
//!   token and commit, crossing a block boundary every 16th step.
//! - `fork_free`: fork a 2K sequence and free the child.
//!
//! Setup and drops are outside the timing.

use std::hint::black_box;

use engine::blocks::{BlockManager, BlockTokens, HashKeys, PoolConfig, SequenceId, hash_block};

const SLABS: u32 = 256;
const PROMPT: u32 = 2048;

fn pool(prefix_caching: bool) -> BlockManager {
    let config = PoolConfig::new(BlockTokens::DEFAULT, SLABS)
        .expect("valid pool")
        .with_prefix_caching(prefix_caching);
    BlockManager::new(config)
}

fn prompt(seed: u32) -> Vec<u32> {
    (0..PROMPT).map(|token| seed * PROMPT + token).collect()
}

fn admit(manager: &mut BlockManager, seq: u64, tokens: &[u32]) {
    let hit = manager.lookup_prefix(tokens, HashKeys::new().with_salt("bench"));
    let rest = &tokens[hit.cached_tokens()..];
    manager.admit(SequenceId(seq), hit, rest).expect("room");
    manager.commit(SequenceId(seq)).expect("live");
}

fn main() {
    divan::main();
}

#[divan::bench]
fn hash_block_16(bencher: divan::Bencher) {
    let tokens: Vec<u32> = (0..16).collect();
    let keys = HashKeys::new().with_salt("bench");
    let parent = hash_block(None, &tokens, &keys);
    bencher.bench(|| hash_block(Some(black_box(&parent)), black_box(&tokens), &keys));
}

#[divan::bench]
fn lookup_hit_2k(bencher: divan::Bencher) {
    let mut manager = pool(true);
    let tokens = prompt(0);
    admit(&mut manager, 0, &tokens);
    manager.free(SequenceId(0)).expect("live");
    bencher.bench(|| {
        let hit = manager.lookup_prefix(black_box(&tokens), HashKeys::new().with_salt("bench"));
        assert_eq!(hit.cached_tokens(), PROMPT as usize - 16);
        hit
    });
}

#[divan::bench(args = [false, true])]
fn admit_commit_free_2k(bencher: divan::Bencher, prefix_caching: bool) {
    let tokens = prompt(1);
    bencher
        .with_inputs(|| pool(prefix_caching))
        .bench_local_values(|mut manager| {
            let hit = manager.lookup_prefix(&tokens, HashKeys::new().with_salt("other"));
            manager.admit(SequenceId(0), hit, &tokens).expect("room");
            manager.commit(SequenceId(0)).expect("live");
            manager.free(SequenceId(0)).expect("live");
            manager
        });
}

#[divan::bench(args = [1, 16, 64])]
fn decode_step(bencher: divan::Bencher, seqs: u64) {
    let mut base = pool(true);
    for seq in 0..seqs {
        // Offset each sequence so block boundaries do not align across rows.
        let length = 1024 + usize::try_from(seq % 16).expect("below 16");
        let seed = u32::try_from(seq).expect("few sequences") + 2;
        let tokens: Vec<u32> = prompt(seed)[..length].to_vec();
        admit(&mut base, seq, &tokens);
    }
    bencher
        .with_inputs(|| base.clone())
        .bench_local_values(|mut manager| {
            for step in 0..16 {
                for seq in 0..seqs {
                    manager
                        .allocate(SequenceId(seq), &[black_box(step)])
                        .expect("room");
                    manager.commit(SequenceId(seq)).expect("live");
                }
            }
            manager
        });
}

#[divan::bench]
fn fork_free(bencher: divan::Bencher) {
    let mut manager = pool(true);
    admit(&mut manager, 0, &prompt(3));
    bencher.bench_local(|| {
        manager
            .fork(SequenceId(0), SequenceId(1))
            .expect("committed");
        manager.free(SequenceId(1)).expect("live");
    });
}
