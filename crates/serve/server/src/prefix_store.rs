//! Prompt-prefix K/V kept on disk, so a restarted server reuses prefixes it
//! prefilled before a restart instead of prefilling them again.
//!
//! Entries live in one directory per model state, named by the SHA-256 of an
//! identity record that covers everything that decides how tokens become
//! K/V (weights, config, tokenizer, chat template, precision, format
//! version). A changed identity selects another directory, so stale K/V is
//! never read; other directories are left in place.
//!
//! Each entry is one file holding its own identity hash, salt, role, prompt
//! tokens and the opaque snapshot bytes. A lookup checks all of them against
//! the request, so a hash collision or a misplaced file cannot hand back
//! foreign K/V. Files are written to a temporary name, synced, then renamed,
//! so a crash leaves either the whole entry or none; leftover temporary
//! files are removed when a store opens. Writes run on a background thread
//! and are dropped, not waited for, when it falls behind.

use std::{
    collections::HashMap,
    fs,
    io::{self, Read, Write as _},
    path::{Path, PathBuf},
    sync::{
        Arc, Mutex, PoisonError,
        mpsc::{Receiver, SyncSender, TrySendError, sync_channel},
    },
    thread::{self, JoinHandle},
};

use sha2::{Digest, Sha256};

/// Leading bytes of an entry file.
const MAGIC: &[u8; 8] = b"MXPFXKV\0";
/// Changed whenever the entry layout changes, so older files are refused.
const VERSION: u32 = 1;
const EXTENSION: &str = "kvs";
const TEMPORARY: &str = ".partial-";
/// Writes waiting for the background thread before new ones are dropped.
const QUEUE: usize = 4;

type Key = [u8; 32];

/// Where and how much prompt-prefix K/V a served model keeps on disk.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct KvPersist {
    pub(crate) dir: PathBuf,
    pub(crate) budget_mib: u32,
}

impl KvPersist {
    pub(crate) fn budget_bytes(&self) -> u64 {
        u64::from(self.budget_mib) << 20
    }
}

/// A stored prefix read back for a prompt.
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct StoredPrefix {
    pub(crate) tokens: Vec<i32>,
    pub(crate) role: u8,
    pub(crate) snapshot: Vec<u8>,
}

struct IndexEntry {
    path: PathBuf,
    tokens: Vec<i32>,
    salt: Option<String>,
    bytes: u64,
    last_used: u64,
}

struct Index {
    entries: HashMap<Key, IndexEntry>,
    used_bytes: u64,
    budget_bytes: u64,
    clock: u64,
}

impl Index {
    /// Deletes least-recently-used entries until `incoming` more bytes fit.
    fn make_room(&mut self, incoming: u64) {
        while self.used_bytes + incoming > self.budget_bytes {
            let Some(oldest) = self
                .entries
                .iter()
                .min_by_key(|(_, entry)| entry.last_used)
                .map(|(key, _)| *key)
            else {
                return;
            };
            if let Some(evicted) = self.entries.remove(&oldest) {
                self.used_bytes -= evicted.bytes;
                let _ = fs::remove_file(&evicted.path);
            }
        }
    }

    fn forget(&mut self, key: &Key) {
        if let Some(entry) = self.entries.remove(key) {
            self.used_bytes -= entry.bytes;
            let _ = fs::remove_file(&entry.path);
        }
    }
}

enum Job {
    Write {
        key: Key,
        salt: Option<String>,
        tokens: Vec<i32>,
        role: u8,
        snapshot: Vec<u8>,
    },
    #[cfg_attr(not(test), expect(dead_code, reason = "only tests wait on the writer"))]
    Flush(SyncSender<()>),
}

/// The on-disk prefix entries of one model state.
pub(crate) struct PrefixStore {
    dir: PathBuf,
    identity: Key,
    index: Arc<Mutex<Index>>,
    jobs: Option<SyncSender<Job>>,
    writer: Option<JoinHandle<()>>,
}

impl PrefixStore {
    /// Opens the directory for `identity` under `root`, creating it, and
    /// indexes its entries from their headers only. Unreadable entries and
    /// leftover temporary files are removed. Entries beyond `budget_bytes`
    /// are evicted oldest first, by modification time.
    pub(crate) fn open(root: &Path, identity: &[u8], budget_bytes: u64) -> io::Result<Self> {
        let identity: Key = Sha256::digest(identity).into();
        let dir = root.join(hex(&identity));
        fs::create_dir_all(&dir)?;
        let mut found = Vec::new();
        for item in fs::read_dir(&dir)? {
            let path = item?.path();
            let name = path
                .file_name()
                .and_then(|name| name.to_str())
                .unwrap_or("");
            if name.starts_with(TEMPORARY) {
                let _ = fs::remove_file(&path);
                continue;
            }
            if path.extension().and_then(|extension| extension.to_str()) != Some(EXTENSION) {
                continue;
            }
            match read_entry(&path, &identity, false) {
                Ok((salt, tokens, _, _)) => {
                    let bytes = fs::metadata(&path)?.len();
                    let modified = fs::metadata(&path)?.modified()?;
                    found.push((modified, path, salt, tokens, bytes));
                }
                Err(_) => {
                    let _ = fs::remove_file(&path);
                }
            }
        }
        found.sort_by_key(|(modified, ..)| *modified);
        let mut index = Index {
            entries: HashMap::new(),
            used_bytes: 0,
            budget_bytes,
            clock: 0,
        };
        for (_, path, salt, tokens, bytes) in found {
            index.make_room(bytes);
            if bytes > budget_bytes {
                let _ = fs::remove_file(&path);
                continue;
            }
            index.clock += 1;
            index.used_bytes += bytes;
            let key = entry_key(&identity, salt.as_deref(), &tokens);
            index.entries.insert(
                key,
                IndexEntry {
                    path,
                    tokens,
                    salt,
                    bytes,
                    last_used: index.clock,
                },
            );
        }
        let index = Arc::new(Mutex::new(index));
        let (jobs, received) = sync_channel(QUEUE);
        let writer = {
            let dir = dir.clone();
            let index = Arc::clone(&index);
            thread::Builder::new()
                .name("prefix-store".into())
                .spawn(move || write_entries(&dir, &identity, &index, &received))?
        };
        Ok(Self {
            dir,
            identity,
            index,
            jobs: Some(jobs),
            writer: Some(writer),
        })
    }

    /// The directory this store's entries live in.
    pub(crate) fn dir(&self) -> &Path {
        &self.dir
    }

    /// Tokens of the longest stored prefix of `prompt` under `salt` that
    /// leaves at least one prompt token, from the index alone.
    pub(crate) fn longest(&self, salt: Option<&str>, prompt: &[i32]) -> Option<usize> {
        self.index
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .entries
            .values()
            .filter(|entry| {
                entry.salt.as_deref() == salt
                    && entry.tokens.len() < prompt.len()
                    && prompt.starts_with(&entry.tokens)
            })
            .map(|entry| entry.tokens.len())
            .max()
    }

    /// The longest stored prefix of `prompt` under `salt` that leaves at least
    /// one prompt token. An entry whose file is missing, truncated or does
    /// not match is removed and skipped.
    pub(crate) fn lookup(&self, salt: Option<&str>, prompt: &[i32]) -> Option<StoredPrefix> {
        let mut index = self.index.lock().unwrap_or_else(PoisonError::into_inner);
        let mut candidates = index
            .entries
            .iter()
            .filter(|(_, entry)| {
                entry.salt.as_deref() == salt
                    && entry.tokens.len() < prompt.len()
                    && prompt.starts_with(&entry.tokens)
            })
            .map(|(key, entry)| (entry.tokens.len(), *key))
            .collect::<Vec<_>>();
        candidates.sort_unstable_by(|left, right| right.cmp(left));
        for (_, key) in candidates {
            let path = index.entries[&key].path.clone();
            match read_entry(&path, &self.identity, true) {
                Ok((stored_salt, tokens, role, Some(snapshot)))
                    if stored_salt.as_deref() == salt && prompt.starts_with(&tokens) =>
                {
                    index.clock += 1;
                    let clock = index.clock;
                    if let Some(entry) = index.entries.get_mut(&key) {
                        entry.last_used = clock;
                    }
                    return Some(StoredPrefix {
                        tokens,
                        role,
                        snapshot,
                    });
                }
                _ => index.forget(&key),
            }
        }
        None
    }

    /// Whether exactly `tokens` under `salt` is stored, so a caller can skip
    /// encoding a snapshot the store already holds.
    pub(crate) fn contains(&self, salt: Option<&str>, tokens: &[i32]) -> bool {
        self.index
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .entries
            .contains_key(&entry_key(&self.identity, salt, tokens))
    }

    /// Queues `snapshot` to be stored for `tokens` under `salt`. Returns
    /// false, storing nothing, when the entry is already stored or the writer
    /// is behind.
    pub(crate) fn save(
        &self,
        salt: Option<&str>,
        tokens: Vec<i32>,
        role: u8,
        snapshot: Vec<u8>,
    ) -> bool {
        if self.contains(salt, &tokens) {
            return false;
        }
        let key = entry_key(&self.identity, salt, &tokens);
        let Some(jobs) = &self.jobs else {
            return false;
        };
        match jobs.try_send(Job::Write {
            key,
            salt: salt.map(str::to_owned),
            tokens,
            role,
            snapshot,
        }) {
            Ok(()) => true,
            Err(TrySendError::Full(_) | TrySendError::Disconnected(_)) => false,
        }
    }

    /// Waits until every write queued so far is on disk or dropped.
    #[cfg_attr(not(test), expect(dead_code, reason = "only tests wait on the writer"))]
    pub(crate) fn flush(&self) {
        let Some(jobs) = &self.jobs else {
            return;
        };
        let (done, wait) = sync_channel(1);
        if jobs.send(Job::Flush(done)).is_ok() {
            let _ = wait.recv();
        }
    }
}

impl Drop for PrefixStore {
    fn drop(&mut self) {
        drop(self.jobs.take());
        if let Some(writer) = self.writer.take() {
            let _ = writer.join();
        }
    }
}

fn write_entries(dir: &Path, identity: &Key, index: &Mutex<Index>, jobs: &Receiver<Job>) {
    let mut sequence = 0_u64;
    for job in jobs {
        let Job::Write {
            key,
            salt,
            tokens,
            role,
            snapshot,
        } = job
        else {
            if let Job::Flush(done) = job {
                let _ = done.send(());
            }
            continue;
        };
        {
            let index = index.lock().unwrap_or_else(PoisonError::into_inner);
            // Two saves of one prefix can be queued before the first lands.
            if index.entries.contains_key(&key) {
                continue;
            }
        }
        let bytes = encode_entry(identity, salt.as_deref(), &tokens, role, &snapshot);
        let size = bytes.len() as u64;
        if size
            > index
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .budget_bytes
        {
            continue;
        }
        sequence += 1;
        let path = dir.join(format!("{}.{EXTENSION}", hex(&key)));
        let temporary = dir.join(format!("{TEMPORARY}{}-{sequence}", std::process::id()));
        if write_atomically(dir, &temporary, &path, &bytes).is_err() {
            let _ = fs::remove_file(&temporary);
            continue;
        }
        let mut index = index.lock().unwrap_or_else(PoisonError::into_inner);
        index.make_room(size);
        index.clock += 1;
        index.used_bytes += size;
        let last_used = index.clock;
        index.entries.insert(
            key,
            IndexEntry {
                path,
                tokens,
                salt,
                bytes: size,
                last_used,
            },
        );
    }
}

/// Writes `bytes` to `temporary`, syncs it, renames it to `path` and syncs
/// the directory, so the entry appears whole or not at all.
fn write_atomically(dir: &Path, temporary: &Path, path: &Path, bytes: &[u8]) -> io::Result<()> {
    let mut file = fs::File::create(temporary)?;
    file.write_all(bytes)?;
    file.sync_all()?;
    drop(file);
    fs::rename(temporary, path)?;
    fs::File::open(dir)?.sync_all()
}

fn entry_key(identity: &Key, salt: Option<&str>, tokens: &[i32]) -> Key {
    let mut hasher = Sha256::new();
    hasher.update(identity);
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

fn encode_entry(
    identity: &Key,
    salt: Option<&str>,
    tokens: &[i32],
    role: u8,
    snapshot: &[u8],
) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(64 + tokens.len() * 4 + snapshot.len());
    bytes.extend_from_slice(MAGIC);
    bytes.extend_from_slice(&VERSION.to_le_bytes());
    bytes.extend_from_slice(identity);
    bytes.push(role);
    match salt {
        None => bytes.push(0),
        Some(salt) => {
            bytes.push(1);
            bytes.extend_from_slice(&(salt.len() as u64).to_le_bytes());
            bytes.extend_from_slice(salt.as_bytes());
        }
    }
    bytes.extend_from_slice(&(tokens.len() as u64).to_le_bytes());
    for token in tokens {
        bytes.extend_from_slice(&token.to_le_bytes());
    }
    bytes.extend_from_slice(&(snapshot.len() as u64).to_le_bytes());
    bytes.extend_from_slice(snapshot);
    bytes
}

type Entry = (Option<String>, Vec<i32>, u8, Option<Vec<u8>>);

/// Sequential reads from an entry file that cannot run past its length.
struct EntryReader {
    file: fs::File,
    remaining: u64,
}

impl EntryReader {
    fn bytes(&mut self, count: u64) -> io::Result<Vec<u8>> {
        if count > self.remaining {
            return Err(invalid("shorter than its header says"));
        }
        self.remaining -= count;
        let mut buffer = vec![0; usize::try_from(count).map_err(|_| invalid("length"))?];
        self.file.read_exact(&mut buffer)?;
        Ok(buffer)
    }

    fn u64(&mut self) -> io::Result<u64> {
        Ok(u64::from_le_bytes(
            self.bytes(8)?.try_into().expect("8 bytes"),
        ))
    }
}

fn invalid(why: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, why.to_owned())
}

/// Reads an entry file's header, and its snapshot when `with_snapshot`.
/// Refuses another magic, version or identity, and a file whose length does
/// not match its header.
fn read_entry(path: &Path, identity: &Key, with_snapshot: bool) -> io::Result<Entry> {
    let file = fs::File::open(path)?;
    let remaining = file.metadata()?.len();
    let mut reader = EntryReader { file, remaining };
    let fixed = reader.bytes(8 + 4 + 32 + 1 + 1)?;
    if &fixed[..8] != MAGIC
        || fixed[8..12] != VERSION.to_le_bytes()
        || fixed[12..44] != identity[..]
    {
        return Err(invalid("another format or identity"));
    }
    let role = fixed[44];
    let salt = match fixed[45] {
        0 => None,
        1 => {
            let len = reader.u64()?;
            Some(String::from_utf8(reader.bytes(len)?).map_err(|_| invalid("salt is not UTF-8"))?)
        }
        _ => return Err(invalid("salt flag")),
    };
    let count = reader.u64()?;
    let tokens = reader
        .bytes(count.saturating_mul(4))?
        .chunks_exact(4)
        .map(|token| i32::from_le_bytes(token.try_into().expect("4 bytes")))
        .collect::<Vec<_>>();
    let snapshot_len = reader.u64()?;
    if snapshot_len != reader.remaining {
        return Err(invalid("length does not match the header"));
    }
    let snapshot = if with_snapshot {
        Some(reader.bytes(snapshot_len)?)
    } else {
        None
    };
    Ok((salt, tokens, role, snapshot))
}

/// A fingerprint of the weights in checkpoint directory `model`: for each
/// `.safetensors` file in name order, its name, size and the SHA-256 of its
/// header (tensor names, dtypes, shapes and offsets), plus the hub snapshot
/// revision when `model` is a Hugging Face cache snapshot. Hashing every
/// weight byte at startup would cost seconds per gigabyte, so a checkpoint
/// rewritten in place with identical shapes and sizes outside a hub
/// snapshot is not told apart.
pub(crate) fn weights_fingerprint(model: &Path) -> io::Result<String> {
    let mut files = fs::read_dir(model)?
        .map(|item| item.map(|item| item.path()))
        .collect::<io::Result<Vec<_>>>()?;
    files.retain(|path| {
        path.extension()
            .is_some_and(|extension| extension == "safetensors")
    });
    files.sort();
    if files.is_empty() {
        return Err(invalid("no safetensors files"));
    }
    let mut hasher = Sha256::new();
    for path in &files {
        let mut file = fs::File::open(path)?;
        let size = file.metadata()?.len();
        let mut length = [0_u8; 8];
        file.read_exact(&mut length)?;
        let header_len = u64::from_le_bytes(length);
        if header_len > size.saturating_sub(8) {
            return Err(invalid("safetensors header past the end"));
        }
        let mut header = vec![0; usize::try_from(header_len).map_err(|_| invalid("header"))?];
        file.read_exact(&mut header)?;
        let name = path
            .file_name()
            .map(|name| name.to_string_lossy())
            .unwrap_or_default();
        hasher.update((name.len() as u64).to_le_bytes());
        hasher.update(name.as_bytes());
        hasher.update(size.to_le_bytes());
        hasher.update(Sha256::digest(&header));
    }
    let canonical = fs::canonicalize(model)?;
    let mut parts = canonical.iter().map(|part| part.to_string_lossy());
    let revision = parts
        .by_ref()
        .find(|part| part == "snapshots")
        .and_then(|_| parts.next())
        .map(std::borrow::Cow::into_owned);
    Ok(format!(
        "{}{}",
        hex(&hasher.finalize()),
        revision.map_or(String::new(), |revision| format!("@{revision}"))
    ))
}

fn hex(bytes: &[u8]) -> String {
    use std::fmt::Write as _;
    bytes
        .iter()
        .fold(String::with_capacity(bytes.len() * 2), |mut text, byte| {
            let _ = write!(text, "{byte:02x}");
            text
        })
}

#[cfg(test)]
#[path = "prefix_store_tests.rs"]
mod tests;
