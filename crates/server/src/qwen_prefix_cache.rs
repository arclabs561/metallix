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

use std::{collections::HashMap, rc::Rc};

use qwen::{
    forward::{Qwen3ForwardExecutor, Qwen3KvSnapshot},
    metal::Qwen3MlxWeights,
};
use sha2::{Digest, Sha256};

use super::ChatRequest;
use chat_format::{ChatFormat, PromptPrefix};

type PrefixKey = [u8; 32];
type Executor<'w> = Qwen3ForwardExecutor<'w, std::collections::hash_map::RandomState>;

/// Prefills `input_ids` on a new resident executor, first restoring the
/// longest prefix cached under the request's salt. Returns the executor, the
/// final prompt logits and the number of prompt tokens taken from the cache.
pub(super) fn prefill<'w>(
    weights: &'w Qwen3MlxWeights,
    cache: &mut PrefixCache<Rc<Qwen3KvSnapshot>>,
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
/// does not render or tokenize is skipped. Shortest first; when both render
/// to the same tokens, the boundary counts as the preamble.
///
/// `reused` is the encoded prefix the request's own prompt reused, if any;
/// each boundary then encodes only its text past that prefix, or all of it
/// when the boundary does not extend it.
pub(super) fn boundaries(
    format: &ChatFormat,
    request: ChatRequest<'_>,
    input_ids: &[i32],
    reused: Option<&PromptPrefix>,
) -> Vec<Boundary> {
    let leading_system = request
        .messages
        .iter()
        .take_while(|message| message.role == super::ChatRole::System)
        .count();
    let mut prefixes = Vec::with_capacity(2);
    if leading_system > 0 || !request.tools.is_empty() {
        prefixes.push((&request.messages[..leading_system], PrefixRole::Preamble));
    }
    prefixes.push((request.messages, PrefixRole::Conversation));
    let mut boundaries: Vec<Boundary> = Vec::with_capacity(prefixes.len());
    for (messages, role) in prefixes {
        let prefix = ChatRequest {
            messages,
            ..request
        };
        let encoded = match reused {
            Some(reused) => format.prompt_reusing(prefix.conversation(), false, reused),
            None => format.prompt(prefix.conversation(), false),
        };
        let Ok(ids) = encoded.map(|prompt| prompt.ids) else {
            continue;
        };
        if !ids.is_empty()
            && input_ids.starts_with(&ids)
            && !boundaries.iter().any(|boundary| boundary.ids == ids)
        {
            boundaries.push(Boundary { ids, role });
        }
    }
    boundaries.sort_by_key(|boundary| boundary.ids.len());
    boundaries
}

/// A prompt prefix worth caching, and why.
pub(super) struct Boundary {
    pub(super) ids: Vec<i32>,
    pub(super) role: PrefixRole,
}

/// Why a prefix was cached, which decides when a newer entry makes it
/// redundant.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum PrefixRole {
    /// The system and tool preamble, shared by every conversation that
    /// starts with it.
    Preamble,
    /// A whole conversation before its generation prompt. Chat is
    /// append-only, so its expected reuse is the next turn of the same
    /// conversation; once that turn stores its own longer entry, this one is
    /// superseded.
    Conversation,
}

/// Caches the prompt's reusable prefixes from `executor`: the leading system
/// and tool preamble, which a later request with the same preamble shares,
/// and the whole conversation before the generation prompt, which the next
/// turn of the same conversation extends. A boundary is cached only when its
/// own rendering tokenizes to an exact prefix of `input_ids`; one that does
/// not render or tokenize is skipped. A snapshot error is a defect, since
/// every boundary lies within the prefilled prompt.
///
/// The conversation entry shares the cached preamble's K/V instead of copying
/// it, but only when `executor` provably holds that preamble's exact values:
/// it was restored from the preamble entry or from an entry extending it, or
/// the preamble entry is taken from `executor` here. Otherwise (restored from
/// an entry stored before its preamble was cached) it stores a full copy, so
/// a restore never joins K/V from two computations. `input_ids` must be the
/// prompt [`prefill`] ran, with no cache insert in between.
pub(super) fn remember(
    weights: &Qwen3MlxWeights,
    cache: &mut PrefixCache<Rc<Qwen3KvSnapshot>>,
    format: &ChatFormat,
    executor: &Executor<'_>,
    request: ChatRequest<'_>,
    input_ids: &[i32],
    reused: Option<&PromptPrefix>,
) -> Result<AcceptedPrefixTokens, String> {
    // The entry `prefill` restored from: the cache has not changed since.
    let restored = cache.peek(request.cache_salt, input_ids).map(Rc::clone);
    let holds = |base: &Rc<Qwen3KvSnapshot>| {
        restored.as_ref().is_some_and(|restored| {
            Rc::ptr_eq(restored, base) || restored.base().is_some_and(|of| Rc::ptr_eq(of, base))
        })
    };
    // A cached preamble whose exact values `executor` holds.
    let mut shared: Option<(Vec<i32>, Rc<Qwen3KvSnapshot>)> = None;
    let mut written = 0;
    for Boundary { ids, role } in boundaries(format, request, input_ids, reused) {
        if let Some(cached) = cache.get(request.cache_salt, &ids) {
            if role == PrefixRole::Preamble && holds(cached) {
                shared = Some((ids, Rc::clone(cached)));
            }
            continue;
        }
        // Sized before copying: a boundary the budget cannot hold (a long
        // prompt's whole conversation) is skipped without materializing it.
        // An extending snapshot is sized as its whole prefix: `insert` refuses
        // a child whose parent and own bytes together exceed the budget.
        let estimate = weights
            .resident_prefix_kv_bytes(ids.len())
            .map_err(|error| error.to_string())?;
        if !cache.admits(
            request.cache_salt,
            &ids,
            usize::try_from(estimate).unwrap_or(usize::MAX),
        ) {
            continue;
        }
        let (snapshot, parent) = match &shared {
            Some((base_ids, base)) if ids.len() > base_ids.len() => (
                weights.snapshot_resident_extending(executor, Rc::clone(base), ids.len()),
                Some(base_ids.as_slice()),
            ),
            _ => (weights.snapshot_resident_prefix(executor, ids.len()), None),
        };
        let snapshot = Rc::new(snapshot.map_err(|error| error.to_string())?);
        let bytes = snapshot.kv_bytes();
        let tokens = ids.len();
        let inserted = cache.insert(
            request.cache_salt,
            ids.clone(),
            Rc::clone(&snapshot),
            bytes,
            role,
            parent,
        );
        if inserted {
            written = written.max(tokens);
            if role == PrefixRole::Preamble {
                shared = Some((ids, snapshot));
            }
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
    role: PrefixRole,
    /// The cached entry whose value this one's value shares.
    parent: Option<PrefixKey>,
    /// Entries sharing this one's value; it is not evicted while any remain.
    children: usize,
}

/// Counters a metrics endpoint can expose. `queries` counts lookups, so the
/// miss count is `queries - hits`.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct PrefixCacheStats {
    pub(crate) queries: u64,
    pub(crate) hits: u64,
    pub(crate) hit_tokens: u64,
    pub(crate) evictions: u64,
    /// Conversation entries dropped because a longer turn of the same
    /// conversation was stored; not counted in `evictions`.
    pub(crate) superseded: u64,
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
    superseded: u64,
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
            superseded: 0,
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
        let key = self.longest(salt, prompt)?;
        self.clock += 1;
        let entry = self.entries.get_mut(&key)?;
        entry.last_used = self.clock;
        let (tokens, parent) = (entry.tokens.len(), entry.parent);
        // A shared parent stays at least as recent as its children.
        if let Some(parent) = parent.and_then(|parent| self.entries.get_mut(&parent)) {
            parent.last_used = self.clock;
        }
        self.hits += 1;
        self.hit_tokens += tokens as u64;
        let entry = self.entries.get(&key)?;
        Some(PrefixHit {
            tokens,
            value: &entry.value,
        })
    }

    /// The value [`Self::lookup`] would return for `prompt`, without counting
    /// a query or refreshing recency.
    pub(crate) fn peek(&self, salt: Option<&str>, prompt: &[i32]) -> Option<&V> {
        let key = self.longest(salt, prompt)?;
        self.entries.get(&key).map(|entry| &entry.value)
    }

    /// The value cached for exactly `tokens` under `salt`.
    pub(crate) fn get(&self, salt: Option<&str>, tokens: &[i32]) -> Option<&V> {
        self.entries
            .get(&self.key(salt, tokens))
            .filter(|entry| Self::matches(entry, salt, tokens))
            .map(|entry| &entry.value)
    }

    fn longest(&self, salt: Option<&str>, prompt: &[i32]) -> Option<PrefixKey> {
        let mut lengths = self
            .entries
            .values()
            .filter(|entry| entry.salt.as_deref() == salt)
            .map(|entry| entry.tokens.len())
            .filter(|&length| length < prompt.len())
            .collect::<Vec<_>>();
        lengths.sort_unstable_by(|left, right| right.cmp(left));
        lengths.dedup();
        lengths.into_iter().find_map(|length| {
            let key = self.key(salt, &prompt[..length]);
            self.entries
                .get(&key)
                .is_some_and(|entry| Self::matches(entry, salt, &prompt[..length]))
                .then_some(key)
        })
    }

    /// Whether exactly `tokens` is cached under `salt`.
    pub(crate) fn contains(&self, salt: Option<&str>, tokens: &[i32]) -> bool {
        self.entries
            .get(&self.key(salt, tokens))
            .is_some_and(|entry| Self::matches(entry, salt, tokens))
    }

    /// Whether [`Self::insert`] would cache an entry of `value_bytes` for
    /// `tokens`, so a caller can skip building one that cannot be kept.
    pub(crate) fn admits(&self, salt: Option<&str>, tokens: &[i32], value_bytes: usize) -> bool {
        let bytes = value_bytes.saturating_add(tokens.len().saturating_mul(size_of::<i32>()));
        !tokens.is_empty() && bytes <= self.budget_bytes && !self.contains(salt, tokens)
    }

    /// Caches `value` for `tokens` under `salt`, evicting least-recently-used
    /// entries of any salt until it fits. Returns false, caching nothing, when
    /// the entry alone exceeds the budget or is already present.
    ///
    /// A [`PrefixRole::Conversation`] entry first drops the conversation
    /// entries under the same salt whose tokens are a strict prefix of its
    /// own: earlier turns of this conversation, whose only expected reuse has
    /// just happened. A client that retries an older turn then misses and
    /// re-prefills; preambles are never dropped this way.
    ///
    /// `parent` names a cached entry (a strict prefix of `tokens`, same salt)
    /// whose value `value` shares, so `value_bytes` counts only what `value`
    /// adds. The parent is then kept while this entry is cached, and the
    /// insert is refused when the parent is not cached or the two do not fit
    /// the budget together.
    pub(crate) fn insert(
        &mut self,
        salt: Option<&str>,
        tokens: Vec<i32>,
        value: V,
        value_bytes: usize,
        role: PrefixRole,
        parent: Option<&[i32]>,
    ) -> bool {
        if !self.admits(salt, &tokens, value_bytes) {
            return false;
        }
        let bytes = value_bytes.saturating_add(tokens.len().saturating_mul(size_of::<i32>()));
        let parent = match parent {
            None => None,
            Some(parent) => {
                let key = self.key(salt, parent);
                let Some(entry) = self
                    .entries
                    .get(&key)
                    .filter(|entry| Self::matches(entry, salt, parent))
                else {
                    return false;
                };
                if parent.len() >= tokens.len()
                    || !tokens.starts_with(parent)
                    || entry.bytes.saturating_add(bytes) > self.budget_bytes
                {
                    return false;
                }
                Some(key)
            }
        };
        if role == PrefixRole::Conversation {
            let superseded = self
                .entries
                .iter()
                .filter(|(_, entry)| {
                    entry.role == PrefixRole::Conversation
                        && entry.salt.as_deref() == salt
                        && entry.tokens.len() < tokens.len()
                        && tokens.starts_with(&entry.tokens)
                })
                .map(|(key, _)| *key)
                .collect::<Vec<_>>();
            for key in superseded {
                if self.remove(&key) {
                    self.superseded += 1;
                }
            }
        }
        while self.used_bytes + bytes > self.budget_bytes {
            // Only entries nothing shares, and never the new entry's parent.
            let Some(oldest) = self
                .entries
                .iter()
                .filter(|(key, entry)| entry.children == 0 && Some(**key) != parent)
                .min_by_key(|(_, entry)| entry.last_used)
                .map(|(key, _)| *key)
            else {
                return false;
            };
            if self.remove(&oldest) {
                self.evictions += 1;
            }
        }
        if let Some(entry) = parent.and_then(|parent| self.entries.get_mut(&parent)) {
            entry.children += 1;
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
                role,
                parent,
                children: 0,
            },
        );
        true
    }

    /// Removes one entry and releases its hold on its parent.
    fn remove(&mut self, key: &PrefixKey) -> bool {
        let Some(removed) = self.entries.remove(key) else {
            return false;
        };
        self.used_bytes -= removed.bytes;
        if let Some(parent) = removed
            .parent
            .and_then(|parent| self.entries.get_mut(&parent))
        {
            parent.children -= 1;
        }
        true
    }

    pub(crate) fn stats(&self) -> PrefixCacheStats {
        PrefixCacheStats {
            queries: self.queries,
            hits: self.hits,
            hit_tokens: self.hit_tokens,
            evictions: self.evictions,
            superseded: self.superseded,
            entries: self.entries.len(),
            bytes: self.used_bytes,
            budget_bytes: self.budget_bytes,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{PrefixCache, PrefixRole};

    const TOKEN: usize = size_of::<i32>();

    #[test]
    fn whole_entry_prefixes_hit_and_leave_one_prompt_token() {
        let mut cache = PrefixCache::new("model", 1_000);
        assert!(cache.insert(None, vec![1, 2, 3], "abc", 10, PrefixRole::Preamble, None));
        assert!(cache.insert(None, vec![1, 2], "ab", 10, PrefixRole::Preamble, None));
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
        assert!(cache.insert(
            None,
            vec![10, 11, 12, 13],
            (),
            0,
            PrefixRole::Preamble,
            None
        ));
        // Shares three leading tokens but not the whole entry.
        assert!(cache.lookup(None, &[10, 11, 12, 99, 5, 6]).is_none());
        assert!(cache.lookup(None, &[10, 11, 12, 13, 5]).is_some());
    }

    #[test]
    fn identical_prompts_under_different_salts_never_share_entries() {
        let mut cache = PrefixCache::new("model", 1_000);
        let prompt = [1, 2, 3, 4];
        assert!(cache.insert(
            Some("tenant-a"),
            vec![1, 2, 3],
            'a',
            0,
            PrefixRole::Preamble,
            None
        ));
        assert!(cache.lookup(Some("tenant-b"), &prompt).is_none());
        assert!(cache.lookup(None, &prompt).is_none());
        assert!(cache.lookup(Some(""), &prompt).is_none());
        // The same tokens can be cached separately per salt.
        assert!(cache.insert(
            Some("tenant-b"),
            vec![1, 2, 3],
            'b',
            0,
            PrefixRole::Preamble,
            None
        ));
        assert!(cache.insert(None, vec![1, 2, 3], 'n', 0, PrefixRole::Preamble, None));
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
        assert!(cache.insert(None, vec![1, 1], 'a', 100, PrefixRole::Preamble, None));
        assert!(cache.insert(None, vec![2, 2], 'b', 100, PrefixRole::Preamble, None));
        assert_eq!(cache.stats().bytes, 2 * entry);
        // Touch `a`, so `b` is least recently used.
        assert!(cache.lookup(None, &[1, 1, 9]).is_some());
        assert!(cache.insert(None, vec![3, 3], 'c', 100, PrefixRole::Preamble, None));
        assert_eq!((cache.stats().entries, cache.stats().evictions), (2, 1));
        assert!(cache.contains(None, &[1, 1]));
        assert!(!cache.contains(None, &[2, 2]));
        assert!(cache.contains(None, &[3, 3]));
        assert_eq!(cache.stats().bytes, 2 * entry);
        // Larger than the whole budget: refused before building it, and
        // nothing is evicted for it.
        assert!(!cache.admits(None, &[4, 4], 2 * entry));
        assert!(cache.admits(None, &[4, 4], 100));
        assert!(!cache.admits(None, &[3, 3], 100), "already cached");
        assert!(!cache.insert(None, vec![4, 4], 'd', 2 * entry, PrefixRole::Preamble, None));
        assert_eq!(cache.stats().entries, 2);
        // Duplicates are not stored twice.
        assert!(!cache.insert(None, vec![3, 3], 'e', 100, PrefixRole::Preamble, None));
        assert_eq!(cache.stats().bytes, 2 * entry);
    }

    #[test]
    fn zero_budget_caches_nothing() {
        let mut cache = PrefixCache::new("model", 0);
        assert!(!cache.insert(None, vec![1], (), 0, PrefixRole::Preamble, None));
        assert!(cache.lookup(None, &[1, 2]).is_none());
    }

    #[test]
    fn a_later_turn_supersedes_earlier_turns_of_its_conversation_only() {
        let mut cache = PrefixCache::new("model", 10_000);
        let conversation = PrefixRole::Conversation;
        assert!(cache.insert(None, vec![1, 2], 'p', 100, PrefixRole::Preamble, None));
        assert!(cache.insert(None, vec![1, 2, 3], 'a', 100, conversation, None));
        // Another conversation on the same preamble, and the same tokens
        // under another salt.
        assert!(cache.insert(None, vec![1, 2, 7], 'b', 100, conversation, None));
        assert!(cache.insert(Some("t"), vec![1, 2, 3], 's', 100, conversation, None));
        assert!(cache.insert(None, vec![1, 2, 3, 4, 5], 'c', 100, conversation, None));
        assert!(!cache.contains(None, &[1, 2, 3]), "turn 1 was superseded");
        for (salt, tokens) in [
            (None, &[1, 2][..]),
            (None, &[1, 2, 7]),
            (Some("t"), &[1, 2, 3]),
            (None, &[1, 2, 3, 4, 5]),
        ] {
            assert!(cache.contains(salt, tokens), "{salt:?} {tokens:?} kept");
        }
        let stats = cache.stats();
        assert_eq!(
            (stats.superseded, stats.evictions, stats.entries),
            (1, 0, 4)
        );
        assert_eq!(stats.bytes, 4 * 100 + (2 + 3 + 3 + 5) * TOKEN);
        // A preamble insert never supersedes, even when it extends an entry.
        assert!(cache.insert(None, vec![1, 2, 7, 8], 'q', 100, PrefixRole::Preamble, None));
        assert!(cache.contains(None, &[1, 2, 7]));
    }

    /// Deterministic `SplitMix64` for the trace below.
    struct Rng(u64);

    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }

        fn below(&mut self, bound: u64) -> usize {
            usize::try_from(self.next() % bound).expect("small bound")
        }
    }

    /// Replays an agent trace through the cache the way `remember` and
    /// `prefill` use it: look up the prompt, then store the preamble and the
    /// whole conversation. Six concurrent agents run 60 conversations of
    /// 3-12 turns over three preambles (300, 200 and 450 tokens; 60/30/10
    /// percent of conversations); each turn adds 30-150 user or tool tokens,
    /// and the reply adds 5-40 tokens to the next prompt. Every token costs
    /// `KV` bytes. With `share`, a conversation entry shares its cached
    /// preamble and is charged only for the tokens after it, as `remember`
    /// stores it. Returns the fraction of prompt tokens served from cache.
    #[allow(
        clippy::cast_precision_loss,
        reason = "token counts here stay far below 2^52"
    )]
    fn replay_agent_trace(
        seed: u64,
        budget_tokens: usize,
        conversation: PrefixRole,
        share: bool,
    ) -> f64 {
        const KV: usize = 1_024;
        struct Agent {
            preamble: usize,
            tokens: Vec<i32>,
            turns_left: usize,
        }
        let preambles: [Vec<i32>; 3] = [300, 200, 450]
            .map(|length: i32| (0..length).map(|index| length * 1_000 + index).collect());
        let mut rng = Rng(seed);
        let mut cache = PrefixCache::new("model", budget_tokens * KV);
        let mut next_token = 10_000_000;
        let (mut started, mut active) = (0, Vec::<Agent>::new());
        let (mut prompt_tokens, mut cached_tokens) = (0, 0);
        while started < 60 || !active.is_empty() {
            while active.len() < 6 && started < 60 {
                let draw = rng.below(10);
                let preamble = usize::from(draw >= 6) + usize::from(draw >= 9);
                active.push(Agent {
                    preamble,
                    tokens: preambles[preamble].clone(),
                    turns_left: 3 + rng.below(10),
                });
                started += 1;
            }
            let index = rng.below(active.len() as u64);
            let agent = &mut active[index];
            for _ in 0..30 + rng.below(121) {
                agent.tokens.push(next_token);
                next_token += 1;
            }
            // The prompt also carries a generation prompt after the history.
            let prompt = [agent.tokens.as_slice(), &[-1]].concat();
            prompt_tokens += prompt.len();
            cached_tokens += cache.lookup(None, &prompt).map_or(0, |hit| hit.tokens);
            let preamble = &preambles[agent.preamble];
            cache.insert(
                None,
                preamble.clone(),
                (),
                preamble.len() * KV,
                PrefixRole::Preamble,
                None,
            );
            let parent = (share && cache.contains(None, preamble)).then_some(preamble.as_slice());
            let charged = agent.tokens.len() - parent.map_or(0, <[i32]>::len);
            cache.insert(
                None,
                agent.tokens.clone(),
                (),
                charged * KV,
                conversation,
                parent,
            );
            for _ in 0..5 + rng.below(36) {
                agent.tokens.push(next_token);
                next_token += 1;
            }
            agent.turns_left -= 1;
            if agent.turns_left == 0 {
                active.swap_remove(index);
            }
        }
        cached_tokens as f64 / prompt_tokens as f64
    }

    #[test]
    fn superseding_old_turns_raises_hits_on_an_agent_trace() {
        // Inserting conversations as preambles is the behaviour before
        // supersession: plain LRU over every stored boundary.
        for (budget_tokens, minimum_gain) in [(1_870, 0.0), (7_490, 0.08)] {
            let (mut before, mut after) = (0.0, 0.0);
            for seed in 0..5 {
                before +=
                    replay_agent_trace(seed, budget_tokens, PrefixRole::Preamble, false) / 5.0;
                after +=
                    replay_agent_trace(seed, budget_tokens, PrefixRole::Conversation, false) / 5.0;
            }
            eprintln!("budget {budget_tokens} tokens: LRU {before:.3}, superseding {after:.3}");
            assert!(
                after >= before + minimum_gain,
                "budget {budget_tokens}: {after:.3} vs {before:.3}"
            );
        }
    }

    #[test]
    fn sharing_the_preamble_raises_hits_when_the_budget_is_tight() {
        let budget_tokens = 1_870;
        let (mut copied, mut shared) = (0.0, 0.0);
        for seed in 0..5 {
            copied +=
                replay_agent_trace(seed, budget_tokens, PrefixRole::Conversation, false) / 5.0;
            shared += replay_agent_trace(seed, budget_tokens, PrefixRole::Conversation, true) / 5.0;
        }
        eprintln!("budget {budget_tokens} tokens: copied {copied:.3}, shared {shared:.3}");
        assert!(shared >= copied + 0.05, "{shared:.3} vs {copied:.3}");
    }

    #[test]
    fn a_shared_parent_is_kept_while_children_remain_and_charged_once() {
        let conversation = PrefixRole::Conversation;
        let preamble = PrefixRole::Preamble;
        // Sizes with token IDs: p 108, c 62, x 116, y 66, z 104 bytes.
        let mut cache = PrefixCache::new("model", 300);
        // Refused: no cached parent, a parent that is not a prefix, or a
        // pair that cannot fit together.
        assert!(!cache.insert(None, vec![1, 2, 3], 'c', 50, conversation, Some(&[1, 2])));
        assert!(cache.insert(None, vec![1, 2], 'p', 100, preamble, None));
        assert!(!cache.insert(None, vec![9, 2, 3], 'c', 50, conversation, Some(&[1, 2])));
        assert!(!cache.insert(None, vec![1, 2, 3], 'c', 250, conversation, Some(&[1, 2])));
        assert!(cache.insert(None, vec![1, 2, 3], 'c', 50, conversation, Some(&[1, 2])));
        assert_eq!(
            cache.stats().bytes,
            108 + 62,
            "the child is charged its own bytes"
        );
        assert!(cache.insert(None, vec![5, 5, 5, 5], 'x', 100, preamble, None));
        // One eviction is needed: the parent is the oldest entry but pinned,
        // so its child goes instead.
        assert!(cache.insert(None, vec![6, 6, 6, 6], 'y', 50, preamble, None));
        assert!(cache.contains(None, &[1, 2]) && !cache.contains(None, &[1, 2, 3]));
        // Released, the parent is evicted next as the oldest.
        assert!(cache.insert(None, vec![7], 'z', 100, preamble, None));
        assert!(!cache.contains(None, &[1, 2]) && cache.contains(None, &[5, 5, 5, 5]));
        assert_eq!(cache.stats().evictions, 2);

        // A hit on a child refreshes its parent too.
        let mut cache = PrefixCache::new("model", 300);
        assert!(cache.insert(None, vec![1, 2], 'p', 100, preamble, None));
        assert!(cache.insert(None, vec![1, 2, 3], 'c', 50, conversation, Some(&[1, 2])));
        assert!(cache.insert(None, vec![5, 5, 5, 5], 'x', 100, preamble, None));
        assert_eq!(
            cache.lookup(None, &[1, 2, 3, 9]).map(|hit| *hit.value),
            Some('c')
        );
        assert!(cache.insert(None, vec![6, 6, 6, 6], 'y', 50, preamble, None));
        assert!(cache.contains(None, &[1, 2]) && cache.contains(None, &[1, 2, 3]));
        assert!(!cache.contains(None, &[5, 5, 5, 5]));
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
