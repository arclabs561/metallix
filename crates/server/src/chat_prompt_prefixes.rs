//! Encoded prompt prefixes kept across turns, so a conversation that grows by
//! a turn re-encodes only the new turn (see [`ChatFormat::prompt_reusing`]).
//!
//! An agent loop re-sends its whole history every request. The prefix a
//! request leaves behind is keyed by a digest of the request that produced
//! it: the cache salt, the tools and template switches, and its messages. The
//! next request in the same conversation starts with those messages, so
//! digests of its message prefixes, longest first, find it. The key only
//! picks a candidate: `prompt_reusing` checks that the new rendering really
//! starts with the prefix's text at the same split, and encodes everything
//! when it does not, so a stale or colliding key costs an encode, never a
//! wrong ID.

use std::collections::VecDeque;

use chat_format::{ChatFormat, Conversation, Prompt, PromptPrefix};
use sha2::{Digest, Sha256};

type Key = [u8; 32];

/// The most recent encoded prefixes, at most `capacity` of them. A 16K-token
/// prefix holds about 64 KB of text and 64 KB of IDs.
pub(crate) struct PromptPrefixes {
    capacity: usize,
    /// Most recently used first.
    entries: VecDeque<(Key, PromptPrefix)>,
}

/// Encoded prefixes kept per session; about 128 KB each at 16K tokens.
const DEFAULT_ENTRIES: usize = 64;

impl Default for PromptPrefixes {
    fn default() -> Self {
        Self::new(DEFAULT_ENTRIES)
    }
}

impl PromptPrefixes {
    pub(crate) const fn new(capacity: usize) -> Self {
        Self {
            capacity,
            entries: VecDeque::new(),
        }
    }

    /// Renders and encodes `conversation`, reusing the cached prefix of the
    /// longest earlier request it extends, then caches this prompt's own
    /// prefix for the next turn. Also returns the prefix it reused, which
    /// encodes this conversation's prefix boundaries just as cheaply.
    pub(crate) fn prompt(
        &mut self,
        format: &ChatFormat,
        salt: Option<&str>,
        conversation: Conversation<'_>,
    ) -> Result<(Prompt, Option<PromptPrefix>), String> {
        let keys = message_keys(salt, conversation);
        let reused = self.take_longest(&keys);
        let prompt = match &reused {
            Some(prefix) => format.prompt_reusing(conversation, true, prefix)?,
            None => format.prompt(conversation, true)?,
        };
        if let (Some(prefix), Some(&key)) = (prompt.reusable_prefix(), keys.last()) {
            self.insert(key, prefix);
        }
        Ok((prompt, reused))
    }

    /// Removes and returns the entry for the longest key present; the caller
    /// re-inserts what replaces it.
    fn take_longest(&mut self, keys: &[Key]) -> Option<PromptPrefix> {
        let index = keys
            .iter()
            .rev()
            .find_map(|key| self.entries.iter().position(|(cached, _)| cached == key))?;
        self.entries.remove(index).map(|(_, prefix)| prefix)
    }

    fn insert(&mut self, key: Key, prefix: PromptPrefix) {
        self.entries.retain(|(cached, _)| *cached != key);
        self.entries.push_front((key, prefix));
        self.entries.truncate(self.capacity);
    }
}

/// One digest per message prefix of `conversation`: entry `m` covers the
/// salt, the template switches, the tools and the first `m + 1` messages.
/// Empty when anything fails to serialize, which disables reuse.
fn message_keys(salt: Option<&str>, conversation: Conversation<'_>) -> Vec<Key> {
    let mut hasher = Sha256::new();
    let mut field = |bytes: &[u8]| {
        hasher.update((bytes.len() as u64).to_le_bytes());
        hasher.update(bytes);
    };
    match salt {
        None => field(&[0]),
        Some(salt) => {
            field(&[1]);
            field(salt.as_bytes());
        }
    }
    field(&[u8::from(conversation.enable_thinking)]);
    field(conversation.reasoning_effort.unwrap_or_default().as_bytes());
    let Ok(tools) = serde_json::to_vec(conversation.tools) else {
        return Vec::new();
    };
    field(&tools);
    let mut keys = Vec::with_capacity(conversation.messages.len());
    for message in conversation.messages {
        let Ok(bytes) = serde_json::to_vec(message) else {
            return Vec::new();
        };
        hasher.update((bytes.len() as u64).to_le_bytes());
        hasher.update(&bytes);
        keys.push(hasher.clone().finalize().into());
    }
    keys
}

#[cfg(test)]
mod tests {
    use chat_format::{
        ChatFormat, ChatMessage, ChatRole, Conversation, PromptReuse,
        test_model::{ModelDir, VOCABULARY_SIZE},
    };
    use serde_json::json;

    use super::{PromptPrefixes, message_keys};

    #[test]
    fn keys_extend_with_the_conversation_and_separate_salts_and_switches() {
        let first = [ChatMessage::text(ChatRole::User, "hi")];
        let grown = [
            ChatMessage::text(ChatRole::User, "hi"),
            ChatMessage::text(ChatRole::Assistant, "hello"),
            ChatMessage::text(ChatRole::User, "more"),
        ];
        let short = message_keys(None, Conversation::new(&first));
        let long = message_keys(None, Conversation::new(&grown));
        assert_eq!(long.len(), 3);
        assert_eq!(
            short[0], long[0],
            "a grown conversation finds its earlier key"
        );
        assert_ne!(long[0], long[2]);
        assert_ne!(
            short,
            message_keys(Some("tenant"), Conversation::new(&first))
        );
        assert_ne!(short, message_keys(Some(""), Conversation::new(&first)));
        let mut thinking = Conversation::new(&first);
        thinking.enable_thinking = true;
        assert_ne!(short, message_keys(None, thinking));
    }

    /// A conversation growing by a turn reuses the previous request's
    /// prefix and gets the same IDs as a full encoding; another salt does not
    /// reuse it.
    #[test]
    fn a_growing_conversation_reuses_its_previous_prompt() {
        let model = ModelDir::new(
            &json!({"chat_template": "{% for m in messages %}{{ m.content }}<|im_end|>{% endfor %}{% if add_generation_prompt %}go{% endif %}"}),
            &json!({"eos_token_id": 2, "vocab_size": VOCABULARY_SIZE}),
            None,
        );
        let format = ChatFormat::load(model.path(), VOCABULARY_SIZE).expect("format");
        let mut prefixes = PromptPrefixes::new(4);
        let mut messages = vec![
            ChatMessage::text(ChatRole::System, "You are terse."),
            ChatMessage::text(ChatRole::User, "List three primes."),
        ];
        let (first, reused) = prefixes
            .prompt(&format, None, Conversation::new(&messages))
            .expect("first");
        assert!(reused.is_none());
        assert_eq!(first.reuse, PromptReuse::Full);
        messages.push(ChatMessage::text(ChatRole::Assistant, "2, 3, 5."));
        messages.push(ChatMessage::text(ChatRole::User, "Three more?"));
        let full = format
            .prompt(Conversation::new(&messages), true)
            .expect("full");
        let (other, _) = prefixes
            .prompt(&format, Some("tenant"), Conversation::new(&messages))
            .expect("other salt");
        assert_eq!(other.reuse, PromptReuse::Full);
        let (second, reused) = prefixes
            .prompt(&format, None, Conversation::new(&messages))
            .expect("second");
        assert!(reused.is_some());
        assert!(matches!(second.reuse, PromptReuse::Prefix { ids } if ids > 0));
        assert_eq!(second.ids, full.ids);
        assert_eq!(second.text, full.text);
        // The capacity bounds the entries.
        for index in 0..10 {
            let one = [ChatMessage::text(ChatRole::User, format!("q{index}"))];
            prefixes
                .prompt(&format, None, Conversation::new(&one))
                .expect("filler");
        }
        assert_eq!(prefixes.entries.len(), 4);
    }
}
