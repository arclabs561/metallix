//! Least-recently-used reuse of prefilled K/V for exact prompt-token prefixes.
//!
//! An entry is keyed by SHA-256 over the session identity (model directory,
//! config, tokenizer and template hashes, which cover the `RoPE` settings, and
//! the adapter identity) and its token IDs. A lookup reuses an entry only when
//! the whole entry is a prefix of the new prompt and at least one prompt token
//! remains to produce logits; stored tokens are compared as well, so a hash
//! collision cannot select foreign K/V. Entries never match partially: a
//! changed system prompt misses even when it shares leading tokens.
//!
//! A router in front of the server may set a per-tenant cache salt (see
//! [`super::ChatRequest::cache_salt`]). It is mixed into every key and checked
//! on every lookup, so tenants with different salts never reuse each other's
//! K/V and cannot time each other's prompts through hits. Requests without a
//! salt share one namespace, which suits a single-tenant deployment.

use std::collections::HashMap;

use qwen::{
    forward::{Qwen3ForwardExecutor, Qwen3KvSnapshot},
    metal::Qwen3MlxWeights,
};
use sha2::{Digest, Sha256};

use super::ChatRequest;
use chat_format::ChatFormat;

type PrefixKey = [u8; 32];
type Executor<'w> = Qwen3ForwardExecutor<'w, std::collections::hash_map::RandomState>;

/// Prefills `input_ids` on a new resident executor, first restoring the
/// longest prefix cached under the request's salt. Returns the executor, the
/// final prompt logits and the number of prompt tokens taken from the cache.
pub(super) fn prefill<'w>(
    weights: &'w Qwen3MlxWeights,
    cache: &mut PrefixCache<Qwen3KvSnapshot>,
    context_limit: usize,
    kv_budget_bytes: u64,
    request: ChatRequest<'_>,
    input_ids: &[i32],
) -> Result<(Executor<'w>, Vec<f32>, usize), String> {
    if let Some(hit) = cache.lookup(request.cache_salt, input_ids) {
        // Entries come only from this session's weights and limits, so a
        // refused snapshot is a defect and fails loudly.
        let mut executor = weights
            .resident_chat_executor_from(hit.value, context_limit, kv_budget_bytes)
            .map_err(|error| error.to_string())?;
        let logits = executor
            .extend_last_logits(&input_ids[hit.tokens..])
            .map_err(|error| error.to_string())?;
        return Ok((executor, logits, hit.tokens));
    }
    let mut executor = weights
        .resident_chat_executor(context_limit, kv_budget_bytes)
        .map_err(|error| error.to_string())?;
    let logits = executor
        .prefill_last_logits(input_ids)
        .map_err(|error| error.to_string())?;
    Ok((executor, logits, 0))
}

/// Extent of the longest accepted snapshot, including any restored prefix.
/// Convert explicitly before reporting newly created prompt tokens.
pub(super) struct AcceptedPrefixTokens(usize);

impl AcceptedPrefixTokens {
    pub(super) fn from_accepted_extent(tokens: usize) -> Self {
        Self(tokens)
    }

    pub(super) fn new_tokens_after(self, cached: usize) -> usize {
        self.0.saturating_sub(cached)
    }
}

/// The prompt prefixes worth caching for `request`: the leading system and
/// tool preamble, which a later request with the same preamble shares, and
/// the whole conversation before the generation prompt, which the next turn
/// of the same conversation extends. A boundary is returned only when its own
/// rendering tokenizes to a nonempty, exact prefix of `input_ids`; one that
/// does not render or tokenize is skipped. Shortest first.
pub(super) fn boundaries(
    format: &ChatFormat,
    request: ChatRequest<'_>,
    input_ids: &[i32],
) -> Vec<Vec<i32>> {
    let leading_system = request
        .messages
        .iter()
        .take_while(|message| message.role == super::ChatRole::System)
        .count();
    let mut prefixes = Vec::with_capacity(2);
    if leading_system > 0 || !request.tools.is_empty() {
        prefixes.push(&request.messages[..leading_system]);
    }
    prefixes.push(request.messages);
    let mut boundaries: Vec<Vec<i32>> = Vec::with_capacity(prefixes.len());
    for messages in prefixes {
        let prefix = ChatRequest {
            messages,
            ..request
        };
        let Ok(ids) = format
            .prompt(prefix.conversation(), false)
            .map(|prompt| prompt.ids)
        else {
            continue;
        };
        if !ids.is_empty() && input_ids.starts_with(&ids) && !boundaries.contains(&ids) {
            boundaries.push(ids);
        }
    }
    boundaries.sort_by_key(Vec::len);
    boundaries
}

/// Caches the prompt's reusable prefixes from `executor`: the leading system
/// and tool preamble, which a later request with the same preamble shares,
/// and the whole conversation before the generation prompt, which the next
/// turn of the same conversation extends. A boundary is cached only when its
/// own rendering tokenizes to an exact prefix of `input_ids`; one that does
/// not render or tokenize is skipped. A snapshot error is a defect, since
/// every boundary lies within the prefilled prompt.
pub(super) fn remember(
    weights: &Qwen3MlxWeights,
    cache: &mut PrefixCache<Qwen3KvSnapshot>,
    format: &ChatFormat,
    executor: &Executor<'_>,
    request: ChatRequest<'_>,
    input_ids: &[i32],
) -> Result<AcceptedPrefixTokens, String> {
    let mut written = 0;
    for ids in boundaries(format, request, input_ids) {
        if cache.contains(request.cache_salt, &ids) {
            continue;
        }
        let snapshot = weights
            .snapshot_resident_prefix(executor, ids.len())
            .map_err(|error| error.to_string())?;
        let bytes = snapshot.kv_bytes();
        let tokens = ids.len();
        if cache.insert(request.cache_salt, ids, snapshot, bytes) {
            written = written.max(tokens);
        }
    }
    Ok(AcceptedPrefixTokens(written))
}

struct Entry<V> {
    tokens: Vec<i32>,
    salt: Option<String>,
    value: V,
    bytes: usize,
    last_used: u64,
}

/// Counters a metrics endpoint can expose. `queries` counts lookups, so the
/// miss count is `queries - hits`.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct PrefixCacheStats {
    pub(crate) queries: u64,
    pub(crate) hits: u64,
    pub(crate) hit_tokens: u64,
    pub(crate) evictions: u64,
    pub(crate) entries: usize,
    pub(crate) bytes: usize,
    pub(crate) budget_bytes: usize,
}

/// Per-session cache of K/V values bounded by a byte budget.
pub(crate) struct PrefixCache<V> {
    identity: Vec<u8>,
    budget_bytes: usize,
    used_bytes: usize,
    clock: u64,
    entries: HashMap<PrefixKey, Entry<V>>,
    queries: u64,
    hits: u64,
    hit_tokens: u64,
    evictions: u64,
}

/// A reusable prefix found for one prompt.
pub(crate) struct PrefixHit<'a, V> {
    pub(crate) tokens: usize,
    pub(crate) value: &'a V,
}

impl<V> PrefixCache<V> {
    /// An empty cache. A zero budget disables insertion.
    pub(crate) fn new(identity: impl Into<Vec<u8>>, budget_bytes: usize) -> Self {
        Self {
            identity: identity.into(),
            budget_bytes,
            used_bytes: 0,
            clock: 0,
            entries: HashMap::new(),
            queries: 0,
            hits: 0,
            hit_tokens: 0,
            evictions: 0,
        }
    }

    /// The key mixes the tenant salt in ahead of the tokens, so entries under
    /// different salts, or under a salt and none, never share a key. An empty
    /// salt still differs from no salt.
    fn key(&self, salt: Option<&str>, tokens: &[i32]) -> PrefixKey {
        let mut hasher = Sha256::new();
        hasher.update((self.identity.len() as u64).to_le_bytes());
        hasher.update(&self.identity);
        match salt {
            None => hasher.update([0_u8]),
            Some(salt) => {
                hasher.update([1_u8]);
                hasher.update((salt.len() as u64).to_le_bytes());
                hasher.update(salt.as_bytes());
            }
        }
        hasher.update((tokens.len() as u64).to_le_bytes());
        for token in tokens {
            hasher.update(token.to_le_bytes());
        }
        hasher.finalize().into()
    }

    fn matches(entry: &Entry<V>, salt: Option<&str>, tokens: &[i32]) -> bool {
        entry.salt.as_deref() == salt && entry.tokens == tokens
    }

    /// The longest prefix of `prompt` cached under `salt` that leaves at
    /// least one prompt token. Every call counts as one query.
    pub(crate) fn lookup(
        &mut self,
        salt: Option<&str>,
        prompt: &[i32],
    ) -> Option<PrefixHit<'_, V>> {
        self.queries += 1;
        let mut lengths = self
            .entries
            .values()
            .filter(|entry| entry.salt.as_deref() == salt)
            .map(|entry| entry.tokens.len())
            .filter(|&length| length < prompt.len())
            .collect::<Vec<_>>();
        lengths.sort_unstable_by(|left, right| right.cmp(left));
        lengths.dedup();
        let key = lengths.into_iter().find_map(|length| {
            let key = self.key(salt, &prompt[..length]);
            self.entries
                .get(&key)
                .is_some_and(|entry| Self::matches(entry, salt, &prompt[..length]))
                .then_some(key)
        })?;
        self.clock += 1;
        let entry = self.entries.get_mut(&key)?;
        entry.last_used = self.clock;
        self.hits += 1;
        self.hit_tokens += entry.tokens.len() as u64;
        Some(PrefixHit {
            tokens: entry.tokens.len(),
            value: &entry.value,
        })
    }

    /// Whether exactly `tokens` is cached under `salt`.
    pub(crate) fn contains(&self, salt: Option<&str>, tokens: &[i32]) -> bool {
        self.entries
            .get(&self.key(salt, tokens))
            .is_some_and(|entry| Self::matches(entry, salt, tokens))
    }

    /// Caches `value` for `tokens` under `salt`, evicting least-recently-used
    /// entries of any salt until it fits. Returns false, caching nothing, when
    /// the entry alone exceeds the budget or is already present.
    pub(crate) fn insert(
        &mut self,
        salt: Option<&str>,
        tokens: Vec<i32>,
        value: V,
        value_bytes: usize,
    ) -> bool {
        let bytes = value_bytes.saturating_add(tokens.len().saturating_mul(size_of::<i32>()));
        if tokens.is_empty() || bytes > self.budget_bytes || self.contains(salt, &tokens) {
            return false;
        }
        while self.used_bytes + bytes > self.budget_bytes {
            let Some(oldest) = self
                .entries
                .iter()
                .min_by_key(|(_, entry)| entry.last_used)
                .map(|(key, _)| *key)
            else {
                break;
            };
            if let Some(evicted) = self.entries.remove(&oldest) {
                self.used_bytes -= evicted.bytes;
                self.evictions += 1;
            }
        }
        self.clock += 1;
        let key = self.key(salt, &tokens);
        self.used_bytes += bytes;
        self.entries.insert(
            key,
            Entry {
                tokens,
                salt: salt.map(str::to_owned),
                value,
                bytes,
                last_used: self.clock,
            },
        );
        true
    }

    pub(crate) fn stats(&self) -> PrefixCacheStats {
        PrefixCacheStats {
            queries: self.queries,
            hits: self.hits,
            hit_tokens: self.hit_tokens,
            evictions: self.evictions,
            entries: self.entries.len(),
            bytes: self.used_bytes,
            budget_bytes: self.budget_bytes,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::PrefixCache;

    const TOKEN: usize = size_of::<i32>();

    #[test]
    fn whole_entry_prefixes_hit_and_leave_one_prompt_token() {
        let mut cache = PrefixCache::new("model", 1_000);
        assert!(cache.insert(None, vec![1, 2, 3], "abc", 10));
        assert!(cache.insert(None, vec![1, 2], "ab", 10));
        // The longest whole entry that is a strict prefix wins.
        let hit = cache.lookup(None, &[1, 2, 3, 4]).expect("hit");
        assert_eq!((hit.tokens, *hit.value), (3, "abc"));
        // An exact-length prompt needs its own last token for logits.
        let hit = cache.lookup(None, &[1, 2, 3]).expect("shorter hit");
        assert_eq!((hit.tokens, *hit.value), (2, "ab"));
        assert!(cache.lookup(None, &[1, 2]).is_none());
        let stats = cache.stats();
        assert_eq!((stats.queries, stats.hits, stats.hit_tokens), (3, 2, 5));
    }

    #[test]
    fn a_changed_prefix_never_hits_even_when_leading_tokens_match() {
        let mut cache = PrefixCache::new("model", 1_000);
        assert!(cache.insert(None, vec![10, 11, 12, 13], (), 0));
        // Shares three leading tokens but not the whole entry.
        assert!(cache.lookup(None, &[10, 11, 12, 99, 5, 6]).is_none());
        assert!(cache.lookup(None, &[10, 11, 12, 13, 5]).is_some());
    }

    #[test]
    fn identical_prompts_under_different_salts_never_share_entries() {
        let mut cache = PrefixCache::new("model", 1_000);
        let prompt = [1, 2, 3, 4];
        assert!(cache.insert(Some("tenant-a"), vec![1, 2, 3], 'a', 0));
        assert!(cache.lookup(Some("tenant-b"), &prompt).is_none());
        assert!(cache.lookup(None, &prompt).is_none());
        assert!(cache.lookup(Some(""), &prompt).is_none());
        // The same tokens can be cached separately per salt.
        assert!(cache.insert(Some("tenant-b"), vec![1, 2, 3], 'b', 0));
        assert!(cache.insert(None, vec![1, 2, 3], 'n', 0));
        assert_eq!(cache.stats().entries, 3);
        for (salt, value) in [
            (Some("tenant-a"), 'a'),
            (Some("tenant-b"), 'b'),
            (None, 'n'),
        ] {
            assert_eq!(
                *cache.lookup(salt, &prompt).expect("own entry").value,
                value
            );
        }
        assert_ne!(cache.key(None, &[1]), cache.key(Some(""), &[1]));
    }

    #[test]
    fn identity_separates_otherwise_equal_token_prefixes() {
        let left = PrefixCache::<()>::new("model-a", 100);
        let right = PrefixCache::<()>::new("model-b", 100);
        assert_ne!(left.key(None, &[1, 2]), right.key(None, &[1, 2]));
        assert_ne!(left.key(None, &[1, 2]), left.key(None, &[1, 2, 0]));
    }

    #[test]
    fn budget_evicts_least_recently_used_and_refuses_oversized_entries() {
        let entry = 100 + 2 * TOKEN;
        let mut cache = PrefixCache::new("model", 2 * entry);
        assert!(cache.insert(None, vec![1, 1], 'a', 100));
        assert!(cache.insert(None, vec![2, 2], 'b', 100));
        assert_eq!(cache.stats().bytes, 2 * entry);
        // Touch `a`, so `b` is least recently used.
        assert!(cache.lookup(None, &[1, 1, 9]).is_some());
        assert!(cache.insert(None, vec![3, 3], 'c', 100));
        assert_eq!((cache.stats().entries, cache.stats().evictions), (2, 1));
        assert!(cache.contains(None, &[1, 1]));
        assert!(!cache.contains(None, &[2, 2]));
        assert!(cache.contains(None, &[3, 3]));
        assert_eq!(cache.stats().bytes, 2 * entry);
        // Larger than the whole budget: nothing is evicted for it.
        assert!(!cache.insert(None, vec![4, 4], 'd', 2 * entry));
        assert_eq!(cache.stats().entries, 2);
        // Duplicates are not stored twice.
        assert!(!cache.insert(None, vec![3, 3], 'e', 100));
        assert_eq!(cache.stats().bytes, 2 * entry);
    }

    #[test]
    fn zero_budget_caches_nothing() {
        let mut cache = PrefixCache::new("model", 0);
        assert!(!cache.insert(None, vec![1], (), 0));
        assert!(cache.lookup(None, &[1, 2]).is_none());
    }
}

#[cfg(test)]
impl super::ChatSession {
    /// Drops every cached prefix and sets a new byte budget.
    pub(crate) fn reset_prefix_cache(&mut self, budget_bytes: usize) {
        self.prefix_cache = PrefixCache::new(self.prefix_cache.identity.clone(), budget_bytes);
    }
}

impl super::ChatSession {
    /// Prefix-cache counters for this session.
    #[cfg_attr(
        not(test),
        expect(dead_code, reason = "read by the planned /metrics endpoint")
    )]
    pub(crate) fn prefix_cache_stats(&self) -> PrefixCacheStats {
        self.prefix_cache.stats()
    }
}

#[cfg(test)]
#[path = "qwen_prefix_cache_checkpoint_tests.rs"]
mod checkpoint_tests;
