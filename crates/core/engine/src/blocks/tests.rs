// Test sizes are a few hundred tokens and blocks; no cast here truncates.
#![allow(clippy::cast_possible_truncation)]

use std::collections::{BTreeMap, HashSet};

use proptest::prelude::*;

use super::{
    Allocation, BlockConfigError, BlockError, BlockHash, BlockId, BlockManager, BlockTokens,
    HashKeys, PoolConfig, SLAB_BLOCKS, SequenceId, TokenPosition, TokenSpan, block_hashes,
    hash_block,
};

const BLOCK: usize = 4;

fn manager(slabs: u32) -> BlockManager {
    let tokens = BlockTokens::new(BLOCK as u32).expect("power of two");
    BlockManager::new(PoolConfig::new(tokens, slabs).expect("valid pool"))
}

fn admit_all(manager: &mut BlockManager, seq: u64, prompt: &[u32], keys: HashKeys) -> Allocation {
    let hit = manager.lookup_prefix(prompt, keys);
    let rest = &prompt[hit.cached_tokens()..];
    let allocation = manager
        .admit(SequenceId(seq), hit, rest)
        .expect("pool has room");
    manager.commit(SequenceId(seq)).expect("live sequence");
    allocation
}

/// A deterministic view of all manager state, to show a failed call changed
/// nothing.
fn fingerprint(manager: &BlockManager, seqs: &BTreeMap<u64, Shadow>) -> String {
    let blocks: Vec<_> = (0..manager.config().num_blocks())
        .map(|index| {
            let block = BlockId(index);
            (manager.ref_count(block), manager.cached_hash(block))
        })
        .collect();
    let tables: Vec<_> = seqs
        .keys()
        .map(|&seq| {
            let id = SequenceId(seq);
            (
                manager.block_table(id).map(<[BlockId]>::to_vec),
                manager.num_tokens(id),
                manager.num_computed(id),
            )
        })
        .collect();
    format!(
        "{blocks:?}{tables:?}{:?}{:?}",
        manager.free_order(),
        manager.counters()
    )
}

/// The test's view of one sequence: its logical tokens and the prompt tokens
/// not yet scheduled.
#[derive(Clone, Debug)]
struct Shadow {
    stream: Vec<u32>,
    todo: Vec<u32>,
}

/// Simulated device memory: the token each slot holds.
struct Device(Vec<Vec<Option<u32>>>);

impl Device {
    fn new(blocks: u32) -> Self {
        Self(vec![vec![None; BLOCK]; blocks as usize])
    }

    /// Applies an allocation's copy, then writes `tokens`, checking that no
    /// write touches a shared or cached block.
    fn write(
        &mut self,
        manager: &BlockManager,
        seq: u64,
        allocation: &Allocation,
        tokens: &[u32],
    ) -> Result<(), TestCaseError> {
        if let Some(copy) = allocation.copy {
            let (src, dst) = (copy.src.index() as usize, copy.dst.index() as usize);
            for offset in 0..copy.tokens {
                self.0[dst][offset] = self.0[src][offset];
            }
        }
        let slots: Vec<_> = manager
            .slots(SequenceId(seq), allocation.positions)
            .map_err(|error| TestCaseError::fail(error.to_string()))?
            .collect();
        prop_assert_eq!(slots.len(), tokens.len());
        for (slot, &token) in slots.iter().zip(tokens) {
            prop_assert_eq!(
                manager.ref_count(slot.block),
                Some(1),
                "wrote a shared block"
            );
            prop_assert_eq!(
                manager.cached_hash(slot.block),
                None,
                "wrote a cached block"
            );
            self.0[slot.block.index() as usize][slot.offset as usize] = Some(token);
        }
        Ok(())
    }

    /// Checks that every scheduled position of every sequence reads back its
    /// own token.
    fn check(
        &self,
        manager: &BlockManager,
        seqs: &BTreeMap<u64, Shadow>,
    ) -> Result<(), TestCaseError> {
        for (&seq, shadow) in seqs {
            let slots = manager
                .slots(SequenceId(seq), TokenSpan::prefix(shadow.stream.len()))
                .map_err(|error| TestCaseError::fail(error.to_string()))?;
            for (position, slot) in slots.enumerate() {
                let held = self.0[slot.block.index() as usize][slot.offset as usize];
                prop_assert_eq!(
                    held,
                    Some(shadow.stream[position]),
                    "seq {} pos {}",
                    seq,
                    position
                );
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
enum Op {
    Admit {
        seq: u64,
        prefix: usize,
        suffix: Vec<u32>,
        salt: bool,
        chunk: usize,
    },
    Allocate {
        seq: u64,
        tokens: Vec<u32>,
    },
    Commit {
        seq: u64,
    },
    Fork {
        parent: u64,
        child: u64,
    },
    Free {
        seq: u64,
    },
}

/// Shared prompt prefixes, so admissions hit each other's cached blocks.
fn prefixes() -> [Vec<u32>; 3] {
    [
        (0..13).collect(),
        (0..9).chain(100..104).collect(),
        (50..58).collect(),
    ]
}

fn op() -> impl Strategy<Value = Op> {
    let seq = 0_u64..6;
    prop_oneof![
        3 => (seq.clone(), 0_usize..3, prop::collection::vec(0_u32..4, 0..10), any::<bool>(), 0_usize..24)
            .prop_map(|(seq, prefix, suffix, salt, chunk)| Op::Admit { seq, prefix, suffix, salt, chunk }),
        4 => (seq.clone(), prop::collection::vec(0_u32..4, 1..7))
            .prop_map(|(seq, tokens)| Op::Allocate { seq, tokens }),
        3 => seq.clone().prop_map(|seq| Op::Commit { seq }),
        2 => (seq.clone(), seq.clone()).prop_map(|(parent, child)| Op::Fork { parent, child }),
        2 => seq.prop_map(|seq| Op::Free { seq }),
    ]
}

/// Admits `prompt` with a first chunk of up to `chunk` tokens after its
/// cached prefix, checking that hit blocks hold the prompt's tokens and that
/// `admit_cost` predicts the blocks consumed.
fn admit_in_model(
    manager: &mut BlockManager,
    device: &mut Device,
    seqs: &mut BTreeMap<u64, Shadow>,
    seq: u64,
    prompt: &[u32],
    keys: HashKeys,
    chunk: usize,
) -> Result<Result<(), BlockError>, TestCaseError> {
    let hit = manager.lookup_prefix(prompt, keys);
    prop_assert!(hit.cached_tokens() < prompt.len().max(1));
    for (index, block) in hit.blocks().iter().enumerate() {
        let held: Vec<_> = device.0[block.index() as usize].clone();
        let want: Vec<_> = prompt[index * BLOCK..(index + 1) * BLOCK]
            .iter()
            .copied()
            .map(Some)
            .collect();
        prop_assert_eq!(held, want, "prefix hit block {} holds other tokens", index);
    }
    let cached = hit.cached_tokens();
    let end = (cached + chunk).min(prompt.len());
    let cost = manager
        .admit_cost(&hit, end - cached)
        .expect("fresh same-manager hit");
    let free = manager.free_blocks();
    let result = manager.admit(SequenceId(seq), hit, &prompt[cached..end]);
    match &result {
        Ok(allocation) => {
            prop_assert_eq!(free - manager.free_blocks(), cost);
            device.write(manager, seq, allocation, &prompt[cached..end])?;
            seqs.insert(
                seq,
                Shadow {
                    stream: prompt[..end].to_vec(),
                    todo: prompt[end..].to_vec(),
                },
            );
        }
        Err(BlockError::OutOfBlocks { needed, .. }) => prop_assert_eq!(*needed, cost),
        Err(_) => {}
    }
    Ok(result.map(|_| ()))
}

/// Runs random operations against a one-slab pool and checks, after every
/// step: reference counts equal table occurrences; a block is queued free
/// exactly when unreferenced; the prefix map and block keys agree; no write
/// lands in a shared or cached block; every sequence reads back its own
/// tokens (so fork plus append never disturbed a sibling, and prefix hits
/// returned the right content); failed calls change nothing.
fn run_model(ops: &[Op]) -> Result<(), TestCaseError> {
    let mut manager = manager(1);
    let mut device = Device::new(manager.config().num_blocks());
    let mut seqs: BTreeMap<u64, Shadow> = BTreeMap::new();
    let prefixes = prefixes();
    for op in ops {
        let before = fingerprint(&manager, &seqs);
        let result = match op {
            Op::Admit {
                seq,
                prefix,
                suffix,
                salt,
                chunk,
            } => {
                let prompt: Vec<u32> = prefixes[*prefix].iter().chain(suffix).copied().collect();
                let keys = if *salt {
                    HashKeys::new().with_salt("tenant")
                } else {
                    HashKeys::new()
                };
                admit_in_model(
                    &mut manager,
                    &mut device,
                    &mut seqs,
                    *seq,
                    &prompt,
                    keys,
                    *chunk,
                )?
            }
            Op::Allocate { seq, tokens } => {
                let tokens = match seqs.get(seq) {
                    Some(shadow) if !shadow.todo.is_empty() => {
                        shadow.todo[..tokens.len().min(shadow.todo.len())].to_vec()
                    }
                    _ => tokens.clone(),
                };
                let cost = manager.append_cost(SequenceId(*seq), tokens.len());
                let free = manager.free_blocks();
                let result = manager.allocate(SequenceId(*seq), &tokens);
                if let Err(BlockError::OutOfBlocks { needed, .. }) = &result {
                    prop_assert_eq!(Ok(*needed), cost.clone());
                }
                if let Ok(allocation) = &result {
                    prop_assert_eq!(Ok(free - manager.free_blocks()), cost);
                    device.write(&manager, *seq, allocation, &tokens)?;
                    let shadow = seqs.get_mut(seq).expect("allocated a live sequence");
                    shadow.stream.extend_from_slice(&tokens);
                    let scheduled = tokens.len().min(shadow.todo.len());
                    shadow.todo.drain(..scheduled);
                }
                result.map(|_| ())
            }
            Op::Commit { seq } => manager.commit(SequenceId(*seq)),
            Op::Fork { parent, child } => {
                let result = manager.fork(SequenceId(*parent), SequenceId(*child));
                if result.is_ok() {
                    let shadow = seqs[parent].clone();
                    seqs.insert(*child, shadow);
                }
                result
            }
            Op::Free { seq } => {
                let result = manager.free(SequenceId(*seq));
                if result.is_ok() {
                    seqs.remove(seq);
                }
                result
            }
        };
        if result.is_err() {
            prop_assert_eq!(
                &before,
                &fingerprint(&manager, &seqs),
                "failed {:?} changed state",
                op
            );
        }
        manager
            .check_invariants()
            .map_err(|error| TestCaseError::fail(format!("after {op:?}: {error}")))?;
        device.check(&manager, &seqs)?;
    }
    for seq in seqs.keys() {
        manager.free(SequenceId(*seq)).expect("live sequence");
    }
    prop_assert_eq!(manager.free_blocks(), manager.total_blocks());
    manager.check_invariants().map_err(TestCaseError::fail)?;
    Ok(())
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(512))]

    #[test]
    fn random_operations_preserve_invariants(ops in prop::collection::vec(op(), 1..80)) {
        run_model(&ops)?;
    }

    /// Different salts never share a block, not even the first; the same salt
    /// hits every full block except the one holding the last token.
    #[test]
    fn different_salts_never_share_blocks(
        prompt in prop::collection::vec(0_u32..8, BLOCK + 1..64),
        salt_a in prop::collection::vec(any::<u8>(), 0..8),
        salt_b in prop::collection::vec(any::<u8>(), 0..8),
        unsalted_b in any::<bool>(),
    ) {
        let keys_b = if unsalted_b { HashKeys::new() } else { HashKeys::new().with_salt(&salt_b) };
        prop_assume!(keys_b.salt() != Some(salt_a.as_slice()));
        let mut manager = manager(4);
        admit_all(&mut manager, 1, &prompt, HashKeys::new().with_salt(&salt_a));
        let other = manager.lookup_prefix(&prompt, keys_b);
        prop_assert_eq!(other.cached_tokens(), 0);
        let same = manager.lookup_prefix(&prompt, HashKeys::new().with_salt(&salt_a));
        prop_assert_eq!(same.cached_tokens(), (prompt.len() - 1) / BLOCK * BLOCK);
    }

    /// Steering or adapter keys enter every block's key: a different extra
    /// key misses at block one and its later keys differ even from the same
    /// parent.
    #[test]
    fn different_extra_keys_never_share_blocks(
        prompt in prop::collection::vec(0_u32..8, 2 * BLOCK + 1..48),
        extra_a in prop::collection::vec(any::<u8>(), 1..8),
        extra_b in prop::collection::vec(any::<u8>(), 1..8),
    ) {
        prop_assume!(extra_a != extra_b);
        let keys_a = HashKeys::new().with_salt("s").with_extra(&extra_a);
        let keys_b = HashKeys::new().with_salt("s").with_extra(&extra_b);
        let mut manager = manager(4);
        admit_all(&mut manager, 1, &prompt, keys_a.clone());
        prop_assert_eq!(manager.lookup_prefix(&prompt, keys_b.clone()).cached_tokens(), 0);
        let parent = hash_block(None, &prompt[..BLOCK], &keys_a);
        let block = &prompt[BLOCK..2 * BLOCK];
        prop_assert_ne!(hash_block(Some(&parent), block, &keys_a), hash_block(Some(&parent), block, &keys_b));
    }

    /// An identical block after a different prefix has a different key, so a
    /// lookup never matches it.
    #[test]
    fn identical_block_after_different_prefix_does_not_match(
        first_a in prop::collection::vec(0_u32..8, BLOCK),
        first_b in prop::collection::vec(0_u32..8, BLOCK),
        shared in prop::collection::vec(0_u32..8, BLOCK),
    ) {
        prop_assume!(first_a != first_b);
        let keys = HashKeys::new();
        let prompt_a: Vec<u32> = first_a.iter().chain(&shared).chain([&9]).copied().collect();
        let prompt_b: Vec<u32> = first_b.iter().chain(&shared).chain([&9]).copied().collect();
        let hashes_a = block_hashes(&prompt_a, BLOCK, &keys);
        let hashes_b = block_hashes(&prompt_b, BLOCK, &keys);
        prop_assert_ne!(hashes_a[1], hashes_b[1]);
        let mut manager = manager(1);
        admit_all(&mut manager, 1, &prompt_a, keys.clone());
        prop_assert_eq!(manager.lookup_prefix(&prompt_b, keys).cached_tokens(), 0);
    }

    /// After a sequence is freed and nothing reallocates its blocks, the same
    /// prompt hits the same blocks and admitting it reuses them.
    #[test]
    fn freed_prefix_returns_same_blocks_while_cached(
        prompt in prop::collection::vec(0_u32..8, 1..100),
    ) {
        let mut manager = manager(4);
        let keys = HashKeys::new().with_salt("t");
        admit_all(&mut manager, 1, &prompt, keys.clone());
        let table = manager.block_table(SequenceId(1)).expect("live").to_vec();
        manager.free(SequenceId(1)).expect("live");
        let hit = manager.lookup_prefix(&prompt, keys);
        let full = (prompt.len() - 1) / BLOCK;
        prop_assert_eq!(hit.blocks(), &table[..full]);
        let rest = prompt[hit.cached_tokens()..].to_vec();
        manager.admit(SequenceId(2), hit, &rest).expect("room");
        let again = manager.block_table(SequenceId(2)).expect("live");
        prop_assert_eq!(&again[..full], &table[..full]);
        prop_assert_eq!(manager.counters().prefix_hits, (full * BLOCK) as u64);
        prop_assert_eq!(manager.counters().prefix_queries, 2 * prompt.len() as u64);
    }

    /// Reallocation evicts only unreferenced blocks: unkeyed ones first, then
    /// cached ones least recently freed first, each freed chain from its
    /// tail.
    #[test]
    fn eviction_takes_unreferenced_blocks_in_lru_order(
        order in Just((0_u64..5).collect::<Vec<_>>()).prop_shuffle(),
        lengths in prop::collection::vec(1_usize..5, 5),
    ) {
        let mut manager = manager(1);
        // A live sequence whose blocks must never be taken.
        admit_all(&mut manager, 99, &[7; 3 * BLOCK], HashKeys::new());
        let pinned: HashSet<_> = manager.block_table(SequenceId(99)).expect("live").iter().copied().collect();
        for (seq, &blocks) in lengths.iter().enumerate() {
            let prompt: Vec<u32> = (0..blocks * BLOCK).map(|t| (seq * 1000 + t) as u32).collect();
            admit_all(&mut manager, seq as u64, &prompt, HashKeys::new());
        }
        let tables: Vec<Vec<BlockId>> = (0..5)
            .map(|seq| manager.block_table(SequenceId(seq)).expect("live").to_vec())
            .collect();
        for &seq in &order {
            manager.free(SequenceId(seq)).expect("live");
        }
        let expected_cached: Vec<BlockId> = order
            .iter()
            .flat_map(|&seq| tables[seq as usize].iter().rev().copied())
            .collect();
        let queue = manager.free_order();
        let unkeyed = queue.len() - expected_cached.len();
        prop_assert!(queue[..unkeyed].iter().all(|block| manager.cached_hash(*block).is_none()));
        prop_assert_eq!(&queue[unkeyed..], &expected_cached[..]);

        let free = manager.free_blocks();
        let filler: Vec<u32> = (0..free * BLOCK).map(|t| 500_000 + t as u32).collect();
        let evictions = manager.counters().evictions;
        manager.admit(SequenceId(100), manager.lookup_prefix(&filler, HashKeys::new()), &filler).expect("exactly fits");
        let taken = manager.block_table(SequenceId(100)).expect("live");
        prop_assert_eq!(taken, &queue[..]);
        prop_assert!(taken.iter().all(|block| !pinned.contains(block)));
        prop_assert_eq!(manager.counters().evictions - evictions, expected_cached.len() as u64);
        prop_assert_eq!(manager.cached_blocks(), 3);
        let error = manager.allocate(SequenceId(100), &[1]).expect_err("pool is full");
        prop_assert_eq!(error, BlockError::OutOfBlocks { needed: 1, free: 0 });
    }
}

#[test]
fn full_hit_recomputes_the_last_token() {
    let mut manager = manager(1);
    let prompt: Vec<u32> = (0..2 * BLOCK as u32).collect();
    admit_all(&mut manager, 1, &prompt, HashKeys::new());
    assert_eq!(manager.cached_blocks(), 2);
    let hit = manager.lookup_prefix(&prompt, HashKeys::new());
    assert_eq!(hit.cached_tokens(), BLOCK);
    let longer: Vec<u32> = (0..=2 * BLOCK as u32).collect();
    assert_eq!(
        manager
            .lookup_prefix(&longer, HashKeys::new())
            .cached_tokens(),
        2 * BLOCK
    );
    assert_eq!(
        manager.lookup_prefix(&[], HashKeys::new()).cached_tokens(),
        0
    );
}

#[test]
fn fork_copies_a_shared_partial_tail_on_write() {
    let mut manager = manager(1);
    let prompt: Vec<u32> = (0..BLOCK as u32 + 2).collect();
    admit_all(&mut manager, 1, &prompt, HashKeys::new());
    let parent = manager.block_table(SequenceId(1)).expect("live").to_vec();
    manager
        .fork(SequenceId(1), SequenceId(2))
        .expect("committed");
    assert_eq!(manager.ref_count(parent[1]), Some(2));

    let child = manager.allocate(SequenceId(2), &[42]).expect("room");
    let copy = child.copy.expect("shared partial tail");
    assert_eq!((copy.src, copy.tokens), (parent[1], 2));
    assert_eq!(manager.ref_count(parent[1]), Some(1));
    assert_eq!(manager.ref_count(parent[0]), Some(2));
    let table = manager.block_table(SequenceId(2)).expect("live");
    assert_eq!((table[0], table[1]), (parent[0], copy.dst));

    let own = manager.allocate(SequenceId(1), &[43]).expect("room");
    assert_eq!(own.copy, None, "the parent's tail is private again");
    manager.check_invariants().expect("consistent");
}

#[test]
fn fork_rejects_uncommitted_and_duplicate_sequences() {
    let mut manager = manager(1);
    let hit = manager.lookup_prefix(&[1, 2, 3], HashKeys::new());
    manager.admit(SequenceId(1), hit, &[1, 2, 3]).expect("room");
    assert_eq!(
        manager.fork(SequenceId(1), SequenceId(2)),
        Err(BlockError::Uncommitted(SequenceId(1)))
    );
    manager.commit(SequenceId(1)).expect("live");
    manager
        .fork(SequenceId(1), SequenceId(2))
        .expect("committed");
    assert_eq!(
        manager.fork(SequenceId(1), SequenceId(2)),
        Err(BlockError::SequenceExists(SequenceId(2)))
    );
    assert_eq!(
        manager.fork(SequenceId(7), SequenceId(8)),
        Err(BlockError::UnknownSequence(SequenceId(7)))
    );
}

#[test]
fn stale_hit_is_rejected() {
    let mut manager = manager(1);
    let prompt: Vec<u32> = (0..=BLOCK as u32).collect();
    admit_all(&mut manager, 1, &prompt, HashKeys::new());
    manager.free(SequenceId(1)).expect("live");
    let hit = manager.lookup_prefix(&prompt, HashKeys::new());
    assert_eq!(hit.cached_tokens(), BLOCK);
    // Fill the pool so the cached block is evicted and reused.
    let filler: Vec<u32> = (1000..1000 + (SLAB_BLOCKS as usize * BLOCK) as u32).collect();
    let fill = manager.lookup_prefix(&filler, HashKeys::new());
    manager.admit(SequenceId(2), fill, &filler).expect("fits");
    manager.free(SequenceId(2)).expect("live");
    assert_eq!(
        manager.admit(SequenceId(3), hit, &prompt[BLOCK..]),
        Err(BlockError::StalePrefixHit)
    );
}

#[test]
fn publication_receipts_count_only_newly_published_blocks() {
    let mut manager = manager(1);
    let prompt: Vec<u32> = (0..=(2 * BLOCK as u32)).collect();
    // Admit concurrent misses before either has published the same prefix.
    for seq in [SequenceId(1), SequenceId(2)] {
        let hit = manager.lookup_prefix(&prompt, HashKeys::new());
        manager.admit(seq, hit, &prompt).unwrap();
    }
    assert_eq!(manager.published_tokens(), 0);
    manager.commit(SequenceId(1)).unwrap();
    assert_eq!(manager.published_tokens(), 2 * BLOCK);
    manager.commit(SequenceId(1)).unwrap();
    assert_eq!(manager.published_tokens(), 2 * BLOCK);
    manager.commit(SequenceId(2)).unwrap();
    assert_eq!(manager.published_tokens(), 2 * BLOCK);
    assert!(manager.commit(SequenceId(3)).is_err());
    assert_eq!(manager.published_tokens(), 2 * BLOCK);
    assert!(
        manager
            .commit_through(SequenceId(1), prompt.len() + 1)
            .is_err()
    );
    assert_eq!(manager.published_tokens(), 2 * BLOCK);
    let hit = manager.lookup_prefix(&prompt, HashKeys::new().with_salt("other"));
    manager.admit(SequenceId(3), hit, &prompt).unwrap();
    manager.commit(SequenceId(3)).unwrap();
    assert_eq!(manager.published_tokens(), 4 * BLOCK);
}

#[test]
fn disabled_prefix_caching_never_publishes() {
    let tokens = BlockTokens::new(BLOCK as u32).expect("power of two");
    let config = PoolConfig::new(tokens, 1)
        .expect("valid")
        .with_prefix_caching(false);
    let mut manager = BlockManager::new(config);
    let prompt: Vec<u32> = (0..3 * BLOCK as u32).collect();
    admit_all(&mut manager, 1, &prompt, HashKeys::new());
    assert_eq!(manager.cached_blocks(), 0);
    assert_eq!(manager.published_tokens(), 0);
    assert_eq!(
        manager
            .lookup_prefix(&prompt, HashKeys::new())
            .cached_tokens(),
        0
    );
}

#[test]
fn usage_counts_referenced_blocks() {
    let mut manager = manager(1);
    assert!(manager.usage().abs() < f64::EPSILON);
    let prompt: Vec<u32> = (0..8 * BLOCK as u32).collect();
    admit_all(&mut manager, 1, &prompt, HashKeys::new());
    assert!((manager.usage() - 0.25).abs() < f64::EPSILON);
    manager.free(SequenceId(1)).expect("live");
    assert!(manager.usage().abs() < f64::EPSILON);
    assert_eq!(manager.cached_blocks(), 8);
}

#[test]
fn slots_map_positions_into_blocks() {
    let mut manager = manager(1);
    let prompt: Vec<u32> = (0..=BLOCK as u32).collect();
    admit_all(&mut manager, 1, &prompt, HashKeys::new());
    let table = manager.block_table(SequenceId(1)).expect("live").to_vec();
    let slots: Vec<_> = manager
        .slots(
            SequenceId(1),
            TokenSpan::new(TokenPosition::new(BLOCK - 1), 2),
        )
        .expect("in range")
        .collect();
    assert_eq!(
        (slots[0].block, slots[0].offset),
        (table[0], BLOCK as u32 - 1)
    );
    assert_eq!((slots[1].block, slots[1].offset), (table[1], 0));
    assert_eq!(
        slots[1].flat(BLOCK as u32),
        u64::from(table[1].index()) * BLOCK as u64
    );
    assert!(matches!(
        manager.slots(SequenceId(1), TokenSpan::prefix(BLOCK + 2)),
        Err(BlockError::PositionOutOfRange { .. })
    ));
}

#[test]
fn hash_encoding_is_unambiguous() {
    let keys = HashKeys::new();
    // The salt is length-prefixed, so it never bleeds into the token bytes.
    assert_ne!(
        hash_block(None, &[1], &HashKeys::new().with_salt([0, 0, 0, 0])),
        hash_block(None, &[0, 1], &HashKeys::new().with_salt([]))
    );
    assert_ne!(
        hash_block(None, &[1, 2], &keys),
        hash_block(None, &[1, 2], &HashKeys::new().with_salt([]))
    );
    let parent: BlockHash = hash_block(None, &[1], &keys);
    assert_ne!(
        hash_block(Some(&parent), &[2], &keys),
        hash_block(None, &[2], &keys)
    );
    // A salt affects only the first block; later keys inherit it by chaining.
    let salted = HashKeys::new().with_salt("x");
    assert_eq!(
        hash_block(Some(&parent), &[2], &keys),
        hash_block(Some(&parent), &[2], &salted)
    );
}

#[test]
fn config_validates_and_sizes_from_budget() {
    assert_eq!(BlockTokens::new(0), Err(BlockConfigError::BlockTokens(0)));
    assert_eq!(BlockTokens::new(12), Err(BlockConfigError::BlockTokens(12)));
    assert_eq!(BlockTokens::default().get(), 16);
    let tokens = BlockTokens::DEFAULT;
    assert_eq!(PoolConfig::new(tokens, 0), Err(BlockConfigError::ZeroSlabs));
    assert!(matches!(
        PoolConfig::new(tokens, u32::MAX / SLAB_BLOCKS),
        Err(BlockConfigError::TooManySlabs(_))
    ));
    // Qwen3-0.6B in F32: 28 layers x 2 x 8 KV heads x 128 x 4 bytes = 224 KiB
    // per token. 512 MiB holds 2,340 tokens, 146 blocks, so 4 whole slabs.
    let per_token = 28 * 2 * 8 * 128 * 4;
    let config = PoolConfig::from_budget(512 << 20, per_token, tokens).expect("fits");
    assert_eq!((config.slabs(), config.num_blocks()), (4, 128));
    assert_eq!(config.token_capacity(), 2048);
    assert!(matches!(
        PoolConfig::from_budget(per_token * 16 * 31, per_token, tokens),
        Err(BlockConfigError::BudgetBelowOneSlab { .. })
    ));
    assert_eq!(
        PoolConfig::from_budget(1, 0, tokens),
        Err(BlockConfigError::ZeroBytesPerToken)
    );
    let block = BlockId(70);
    assert_eq!((block.slab(), block.index_in_slab()), (2, 6));
}

/// A pipelined decode reserves each step's slot before its input token is
/// known, resolves it when the previous step is read back, and commits only
/// through the finished step. It must publish exactly the blocks, under
/// exactly the keys, that plain allocate-and-commit publishes.
#[test]
fn pipelined_steps_publish_the_same_prefix_as_plain_steps() {
    let prompt: Vec<u32> = (0..6).collect();
    let generated: Vec<u32> = (100..111).collect();
    let keys = HashKeys::new().with_salt("tenant");

    let mut plain = manager(1);
    admit_all(&mut plain, 1, &prompt, keys.clone());
    for &token in &generated {
        plain.allocate(SequenceId(1), &[token]).expect("room");
        plain.commit(SequenceId(1)).expect("live");
    }

    let mut piped = manager(1);
    admit_all(&mut piped, 1, &prompt, keys.clone());
    let seq = SequenceId(1);
    // The first generated token comes from the prefill's host pick.
    piped.allocate(seq, &generated[..1]).expect("room");
    for (step, &token) in generated.iter().enumerate().skip(1) {
        // Step `step` is queued before step `step - 1` finishes.
        piped.allocate_unresolved(seq, 1).expect("room");
        assert!(matches!(
            piped.allocate(seq, &[token]),
            Err(BlockError::Unresolved(_))
        ));
        piped
            .check_invariants()
            .expect("invariants with a queued step");
        // Step `step - 1` finishes: its input is computed, and its pick is
        // the queued step's input.
        piped
            .commit_through(seq, prompt.len() + step)
            .expect("live");
        piped.resolve(seq, &[token]).expect("one unresolved");
        piped
            .check_invariants()
            .expect("invariants after resolving");
    }
    piped
        .commit_through(seq, prompt.len() + generated.len())
        .expect("live");

    let table = |manager: &BlockManager| {
        manager
            .block_table(seq)
            .expect("live")
            .iter()
            .map(|block| manager.cached_hash(*block))
            .collect::<Vec<_>>()
    };
    assert_eq!(table(&piped), table(&plain));
    assert_eq!(piped.cached_blocks(), plain.cached_blocks());
    assert!(piped.cached_blocks() >= 3, "full blocks were published");
    let mut lookup = prompt.clone();
    lookup.extend(&generated);
    assert_eq!(
        piped.lookup_prefix(&lookup, keys.clone()).cached_tokens(),
        plain.lookup_prefix(&lookup, keys).cached_tokens()
    );
}

/// A block holding a queued step's slot is never published before that
/// step's K/V exists, even when its token value is already resolved.
#[test]
fn unfinished_positions_are_not_published() {
    let mut blocks = manager(1);
    let seq = SequenceId(1);
    admit_all(&mut blocks, 1, &[1, 2, 3], HashKeys::new());
    blocks.allocate_unresolved(seq, 1).expect("room");
    blocks.resolve(seq, &[4]).expect("one unresolved");
    // The fourth token completes block 0, but its step has not finished.
    blocks.commit_through(seq, 3).expect("live");
    assert_eq!(blocks.cached_blocks(), 0);
    assert!(matches!(
        blocks.fork(seq, SequenceId(2)),
        Err(BlockError::Uncommitted(_))
    ));
    blocks.commit_through(seq, 4).expect("live");
    assert_eq!(blocks.cached_blocks(), 1);
    assert!(matches!(
        blocks.resolve(seq, &[5]),
        Err(BlockError::Unresolved(_))
    ));
    blocks.check_invariants().expect("invariants");
}

#[test]
fn budget_rejects_mathematical_slab_overflow() {
    let tokens = BlockTokens::DEFAULT;
    let positions_per_slab = u64::from(tokens.get()) * u64::from(SLAB_BLOCKS);
    // Overflow in the first product, then only in the second product.
    for per_token in [u64::MAX, u64::MAX / u64::from(tokens.get())] {
        let mathematical_bytes = u128::from(per_token) * u128::from(positions_per_slab);
        assert!(mathematical_bytes > u128::from(u64::MAX));
        assert_eq!(
            PoolConfig::from_budget(u64::MAX, per_token, tokens),
            Err(BlockConfigError::SlabBytesOverflow {
                bytes_per_token: per_token,
                block_tokens: tokens,
            }),
            "an unrepresentable slab cannot fit even the largest byte budget"
        );
    }
    // The adjacent representable case remains valid; no device allocation.
    let per_token = u64::MAX / positions_per_slab;
    let slab_bytes = per_token * positions_per_slab;
    let config = PoolConfig::from_budget(slab_bytes, per_token, tokens).unwrap();
    assert_eq!(config.slabs(), 1);
    assert!(PoolConfig::from_budget(slab_bytes - 1, per_token, tokens).is_err());
}

#[test]
fn foreign_same_shape_prefix_hit_rejects_without_mutation() {
    let mut owner = manager(1);
    let mut recipient = manager(1);
    let prompt = [11, 12, 13, 14, 15];
    let keys = HashKeys::new();
    admit_all(&mut owner, 1, &prompt, keys.clone());
    admit_all(&mut recipient, 1, &prompt, keys.clone());
    owner.free(SequenceId(1)).unwrap();
    recipient.free(SequenceId(1)).unwrap();
    let foreign = owner.lookup_prefix(&prompt, keys.clone());
    let local = recipient.lookup_prefix(&prompt, keys);
    assert_eq!(foreign.cached_tokens(), BLOCK);
    assert_eq!(
        foreign.blocks(),
        local.blocks(),
        "matching IDs/hashes do not prove pool identity"
    );
    let before = fingerprint(&recipient, &BTreeMap::new());
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        recipient.admit(SequenceId(2), foreign, &prompt[BLOCK..])
    }));
    assert!(
        result.is_ok(),
        "foreign prefix hit must be an error, not a panic"
    );
    assert!(
        result.unwrap().is_err(),
        "foreign owner must be rejected even with identical cached content"
    );
    assert_eq!(recipient.sequences(), 0);
    assert_eq!(fingerprint(&recipient, &BTreeMap::new()), before);
}

#[test]
fn foreign_larger_pool_prefix_hit_rejects_without_panic_or_mutation() {
    let mut owner = manager(2);
    let mut recipient = manager(1);
    // Keep the owner's first slab occupied, forcing the tested hit outside
    // the recipient's block-ID range using only public manager operations.
    let filler: Vec<u32> = (1000..1000 + SLAB_BLOCKS * BLOCK as u32).collect();
    admit_all(&mut owner, 1, &filler, HashKeys::new());
    let prompt = [21, 22, 23, 24, 25];
    let keys = HashKeys::new();
    admit_all(&mut owner, 2, &prompt, keys.clone());
    let foreign = owner.lookup_prefix(&prompt, keys);
    assert_eq!(foreign.cached_tokens(), BLOCK);
    assert!(
        foreign
            .blocks()
            .iter()
            .any(|id| id.index() >= recipient.config().num_blocks())
    );
    let before = fingerprint(&recipient, &BTreeMap::new());
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        recipient.admit(SequenceId(3), foreign, &prompt[BLOCK..])
    }));
    assert!(
        result.is_ok(),
        "foreign prefix hit must reject before indexing recipient storage"
    );
    assert!(result.unwrap().is_err(), "foreign owner must be rejected");
    assert_eq!(recipient.sequences(), 0);
    assert_eq!(fingerprint(&recipient, &BTreeMap::new()), before);
}

#[test]
fn prefill_chunks_end_on_block_boundaries_unless_they_finish_the_prompt() {
    let manager = BlockManager::new(PoolConfig::new(BlockTokens::DEFAULT, 1).expect("pool"));
    // 16-token blocks.
    assert_eq!(manager.chunk_end(0, 100, 64), 64);
    assert_eq!(manager.chunk_end(64, 100, 64), 100, "the rest fits");
    assert_eq!(manager.chunk_end(0, 100, 50), 48, "rounded down to a block");
    assert_eq!(
        manager.chunk_end(48, 100, 10),
        48,
        "short of the next block"
    );
    assert_eq!(
        manager.chunk_end(48, 52, 10),
        52,
        "but the prompt's end is fine"
    );
    assert_eq!(manager.chunk_end(0, 100, usize::MAX), 100);
}
