//! SHA-256 chain keys for full KV blocks.

use std::fmt;

use sha2::{Digest, Sha256};

/// Separates block keys from any other SHA-256 input with the same layout.
const DOMAIN: &[u8] = b"metallix.kv-block.v1";

/// The content key of one full block: its tokens, every block before it, and
/// the inputs in [`HashKeys`].
#[derive(Clone, Copy, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct BlockHash([u8; 32]);

impl BlockHash {
    /// Returns the SHA-256 digest.
    #[must_use]
    pub const fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

impl fmt::Debug for BlockHash {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "BlockHash(")?;
        for byte in &self.0[..8] {
            write!(f, "{byte:02x}")?;
        }
        write!(f, "..)")
    }
}

/// Inputs other than token IDs that change the K/V a block holds, or that
/// must keep two requests from sharing blocks.
///
/// The cache salt enters only the first block's key; every later key chains
/// from it, so two salts never share any block. Extra keys (a residual
/// steering artifact, an adapter, multimodal inputs) enter every block's key.
/// The checkpoint, `RoPE` configuration and KV dtype are fixed for one pool and
/// are not keys.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct HashKeys {
    salt: Option<Box<[u8]>>,
    extras: Vec<Box<[u8]>>,
}

impl HashKeys {
    /// Keys with no salt and no extra inputs.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the router-supplied cache salt.
    #[must_use]
    pub fn with_salt(mut self, salt: impl AsRef<[u8]>) -> Self {
        self.salt = Some(salt.as_ref().into());
        self
    }

    /// Adds an input that changes K/V for every block, such as a digest of
    /// the installed steering artifact. Order is significant.
    #[must_use]
    pub fn with_extra(mut self, key: impl AsRef<[u8]>) -> Self {
        self.extras.push(key.as_ref().into());
        self
    }

    /// Returns the cache salt, if any.
    #[must_use]
    pub fn salt(&self) -> Option<&[u8]> {
        self.salt.as_deref()
    }
}

/// Returns the key of one full block.
///
/// `parent` is the key of the preceding block, or `None` for the block that
/// starts at token 0; only that first block includes the salt. Every
/// variable-length field is length-prefixed, so distinct inputs never encode
/// to the same bytes.
#[must_use]
pub fn hash_block(parent: Option<&BlockHash>, tokens: &[u32], keys: &HashKeys) -> BlockHash {
    let mut hasher = Sha256::new();
    hasher.update(DOMAIN);
    match parent {
        None => {
            hasher.update([0]);
            match &keys.salt {
                None => hasher.update([0]),
                Some(salt) => {
                    hasher.update([1]);
                    update_bytes(&mut hasher, salt);
                }
            }
        }
        Some(parent) => {
            hasher.update([1]);
            hasher.update(parent.0);
        }
    }
    hasher.update((tokens.len() as u64).to_le_bytes());
    let mut buffer = [0_u8; 64];
    for chunk in tokens.chunks(16) {
        for (bytes, token) in buffer.chunks_exact_mut(4).zip(chunk) {
            bytes.copy_from_slice(&token.to_le_bytes());
        }
        hasher.update(&buffer[..chunk.len() * 4]);
    }
    hasher.update((keys.extras.len() as u64).to_le_bytes());
    for extra in &keys.extras {
        update_bytes(&mut hasher, extra);
    }
    BlockHash(hasher.finalize().into())
}

/// Returns the chained keys of every full block in `tokens`.
#[must_use]
pub fn block_hashes(tokens: &[u32], block_tokens: usize, keys: &HashKeys) -> Vec<BlockHash> {
    let mut hashes: Vec<BlockHash> = Vec::with_capacity(tokens.len() / block_tokens.max(1));
    if block_tokens == 0 {
        return hashes;
    }
    for block in tokens.chunks_exact(block_tokens) {
        let hash = hash_block(hashes.last(), block, keys);
        hashes.push(hash);
    }
    hashes
}

fn update_bytes(hasher: &mut Sha256, bytes: &[u8]) {
    hasher.update((bytes.len() as u64).to_le_bytes());
    hasher.update(bytes);
}
