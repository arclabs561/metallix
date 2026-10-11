//! Append-only packs of table rows fetched from part of a tensor.
//!
//! Each tensor has `<tensor>.pack`, the row bytes in fetch order, and
//! `<tensor>.index`, one JSON line per stored row giving its pack offset,
//! length, SHA-256, shard range and revision. A write appends the data,
//! syncs it, then appends and syncs the index lines, so a crash leaves at
//! worst unindexed pack bytes or a torn final index line. Opening replays
//! every index (later lines for a row win) and drops a torn tail. A row whose
//! bytes are past the end of its pack (the pack was truncated or deleted) is
//! marked lost: its refetch must reproduce the recorded digest. When replay
//! changed anything, the index is rewritten atomically, one line per row, and
//! the pack is truncated to the end of its last present row.

use std::{
    collections::HashMap,
    fs,
    io::{self, Read as _, Seek as _, SeekFrom, Write as _},
    path::{Path, PathBuf},
    sync::{Mutex, MutexGuard, PoisonError},
};

use deepseek::checkpoint::range_cache::V41RangeCacheError;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// One index line: where a row's bytes are and what they must hash to.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub(super) struct RowRecord {
    pub(super) row: u64,
    pub(super) offset: u64,
    pub(super) length: u64,
    pub(super) sha256: String,
    pub(super) shard: String,
    /// Absolute shard byte range of the row.
    pub(super) range: [u64; 2],
    pub(super) revision: String,
    /// The bytes are gone; only the digest is kept.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub(super) lost: bool,
}

/// A row to append: the record's fields other than its offset, plus bytes.
pub(super) struct NewRow {
    pub(super) row: u64,
    pub(super) shard: String,
    pub(super) range: [u64; 2],
    pub(super) revision: String,
    pub(super) bytes: Vec<u8>,
}

/// What the packs hold for one row.
pub(super) enum StoredRow {
    /// Verified bytes.
    Present(Vec<u8>),
    /// A record whose bytes were lost; a refetch must hash to this digest.
    Lost(String),
    Absent,
}

#[derive(Default)]
struct Pack {
    rows: HashMap<u64, RowRecord>,
    /// Pack file length: the end of the last indexed row.
    data_len: u64,
}

/// Every tensor's pack in one directory.
pub(super) struct RowPacks {
    dir: PathBuf,
    packs: Mutex<HashMap<String, Pack>>,
}

impl std::fmt::Debug for RowPacks {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("RowPacks")
            .field("dir", &self.dir)
            .finish_non_exhaustive()
    }
}

impl RowPacks {
    /// Opens (and recovers) every pack in `dir`. Returns the packs and their
    /// total data bytes.
    pub(super) fn open(dir: PathBuf) -> io::Result<(Self, u64)> {
        let mut packs = HashMap::new();
        let mut total = 0;
        let entries = match fs::read_dir(&dir) {
            Ok(entries) => entries.collect::<Result<Vec<_>, _>>()?,
            Err(error) if error.kind() == io::ErrorKind::NotFound => Vec::new(),
            Err(error) => return Err(error),
        };
        for entry in entries {
            let name = entry.file_name().to_string_lossy().into_owned();
            let Some(tensor) = name.strip_suffix(".index") else {
                continue;
            };
            let pack = recover(&dir, tensor)?;
            total += pack.data_len;
            packs.insert(tensor.to_owned(), pack);
        }
        Ok((
            Self {
                dir,
                packs: Mutex::new(packs),
            },
            total,
        ))
    }

    fn lock(&self) -> MutexGuard<'_, HashMap<String, Pack>> {
        self.packs.lock().unwrap_or_else(PoisonError::into_inner)
    }

    fn pack_path(&self, tensor: &str) -> PathBuf {
        self.dir.join(format!("{tensor}.pack"))
    }

    /// Reads `row` of `tensor`, checked against its record and against the
    /// `shard`, `range` and `revision` the caller expects.
    pub(super) fn get(
        &self,
        tensor: &str,
        row: u64,
        expected: (&str, [u64; 2], &str),
    ) -> Result<StoredRow, V41RangeCacheError> {
        let record = match self.lock().get(tensor).and_then(|pack| {
            pack.rows
                .get(&row)
                .map(|record| (record.clone(), pack.data_len))
        }) {
            None => return Ok(StoredRow::Absent),
            Some((record, _)) if record.lost => return Ok(StoredRow::Lost(record.sha256)),
            Some((record, _)) => record,
        };
        let (shard, range, revision) = expected;
        if record.shard != shard || record.range != range || record.revision != revision {
            return Err(io_error(format!(
                "{tensor} row {row} is stored for {} {:?} at revision {}, not {shard} {range:?} at {revision}",
                record.shard, record.range, record.revision
            )));
        }
        let mut file = fs::File::open(self.pack_path(tensor)).map_err(V41RangeCacheError::Io)?;
        file.seek(SeekFrom::Start(record.offset))
            .map_err(V41RangeCacheError::Io)?;
        let mut bytes = vec![
            0;
            usize::try_from(record.length)
                .map_err(|_| io_error("row too large".to_owned()))?
        ];
        file.read_exact(&mut bytes)
            .map_err(V41RangeCacheError::Io)?;
        if format!("{:x}", Sha256::digest(&bytes)) != record.sha256 {
            return Err(V41RangeCacheError::HashMismatch(format!(
                "{tensor} row {row}"
            )));
        }
        Ok(StoredRow::Present(bytes))
    }

    /// Appends `rows` of `tensor`: data first, synced, then index lines,
    /// synced. Returns the pack bytes added.
    pub(super) fn append(
        &self,
        tensor: &str,
        rows: Vec<NewRow>,
    ) -> Result<u64, V41RangeCacheError> {
        if rows.is_empty() {
            return Ok(0);
        }
        fs::create_dir_all(&self.dir).map_err(V41RangeCacheError::Io)?;
        let mut packs = self.lock();
        let pack = packs.entry(tensor.to_owned()).or_default();
        let mut data = fs::OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(false)
            .open(self.pack_path(tensor))
            .map_err(V41RangeCacheError::Io)?;
        // Write at the end of the last indexed row, over any orphaned tail.
        data.seek(SeekFrom::Start(pack.data_len))
            .map_err(V41RangeCacheError::Io)?;
        let mut offset = pack.data_len;
        let mut records = Vec::with_capacity(rows.len());
        for new in rows {
            data.write_all(&new.bytes).map_err(V41RangeCacheError::Io)?;
            let length = new.bytes.len() as u64;
            records.push(RowRecord {
                row: new.row,
                offset,
                length,
                sha256: format!("{:x}", Sha256::digest(&new.bytes)),
                shard: new.shard,
                range: new.range,
                revision: new.revision,
                lost: false,
            });
            offset += length;
        }
        data.set_len(offset).map_err(V41RangeCacheError::Io)?;
        data.sync_data().map_err(V41RangeCacheError::Io)?;
        let mut lines = Vec::new();
        for record in &records {
            serde_json::to_writer(&mut lines, record).map_err(V41RangeCacheError::ReceiptJson)?;
            lines.push(b'\n');
        }
        let mut index = fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(self.dir.join(format!("{tensor}.index")))
            .map_err(V41RangeCacheError::Io)?;
        index.write_all(&lines).map_err(V41RangeCacheError::Io)?;
        index.sync_data().map_err(V41RangeCacheError::Io)?;
        let added = offset - pack.data_len;
        pack.data_len = offset;
        for record in records {
            pack.rows.insert(record.row, record);
        }
        Ok(added)
    }
}

/// A row receipt of the earlier one-file-per-row store,
/// `.rows/<tensor>/<row>.receipt.json` beside `<row>.bin`.
#[derive(Deserialize)]
struct LegacyReceipt {
    tensor: String,
    shard: String,
    row: u64,
    range: [u64; 2],
    sha256: String,
    revision: String,
}

impl RowPacks {
    /// Moves the earlier one-file-per-row store under `legacy` into packs.
    ///
    /// A row whose bytes match its receipt is appended; a receipt without
    /// bytes becomes a lost record, keeping the digest a refetch must match;
    /// a row already packed with the same digest is dropped as redundant. A
    /// row whose bytes fail their receipt, or whose packed digest differs,
    /// stays where it is and is logged. Each tensor's migrated files are
    /// removed only after its pack and index are synced. Returns the pack
    /// bytes added.
    pub(super) fn migrate_legacy(&self, legacy: &Path) -> Result<u64, V41RangeCacheError> {
        let tensors = match fs::read_dir(legacy) {
            Ok(tensors) => tensors
                .collect::<Result<Vec<_>, _>>()
                .map_err(V41RangeCacheError::Io)?,
            Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(0),
            Err(error) => return Err(V41RangeCacheError::Io(error)),
        };
        let mut added = 0;
        for tensor_dir in tensors {
            let tensor = tensor_dir.file_name().to_string_lossy().into_owned();
            let dir = tensor_dir.path();
            let mut present = Vec::new();
            let mut lost = Vec::new();
            let mut migrated = Vec::new();
            for entry in fs::read_dir(&dir).map_err(V41RangeCacheError::Io)? {
                let receipt_path = entry.map_err(V41RangeCacheError::Io)?.path();
                let Some(row) = receipt_path
                    .file_name()
                    .and_then(|name| name.to_str()?.strip_suffix(".receipt.json"))
                    .and_then(|row| row.parse::<u64>().ok())
                else {
                    continue;
                };
                let receipt: LegacyReceipt = serde_json::from_slice(
                    &fs::read(&receipt_path).map_err(V41RangeCacheError::Io)?,
                )
                .map_err(V41RangeCacheError::ReceiptJson)?;
                if receipt.tensor != tensor || receipt.row != row {
                    tracing::warn!(path = %receipt_path.display(), "legacy row receipt names another row; left in place");
                    continue;
                }
                let bin_path = dir.join(format!("{row}.bin"));
                let packed = self
                    .lock()
                    .get(&tensor)
                    .and_then(|pack| pack.rows.get(&row).map(|record| record.sha256.clone()));
                if let Some(sha256) = packed {
                    if sha256 == receipt.sha256 {
                        // Already packed with the same digest: the files are redundant.
                        migrated.extend([bin_path, receipt_path]);
                    } else {
                        tracing::warn!(path = %receipt_path.display(), "legacy row disagrees with its packed digest; left in place");
                    }
                    continue;
                }
                match fs::read(&bin_path) {
                    Ok(bytes) if format!("{:x}", Sha256::digest(&bytes)) == receipt.sha256 => {
                        present.push(NewRow {
                            row,
                            shard: receipt.shard,
                            range: receipt.range,
                            revision: receipt.revision,
                            bytes,
                        });
                        migrated.push(bin_path);
                    }
                    Ok(_) => {
                        tracing::warn!(path = %bin_path.display(), "legacy row fails its receipt; left in place");
                        continue;
                    }
                    Err(error) if error.kind() == io::ErrorKind::NotFound => {
                        lost.push(RowRecord {
                            row,
                            offset: 0,
                            length: receipt.range[1] - receipt.range[0],
                            sha256: receipt.sha256,
                            shard: receipt.shard,
                            range: receipt.range,
                            revision: receipt.revision,
                            lost: true,
                        });
                    }
                    Err(error) => return Err(V41RangeCacheError::Io(error)),
                }
                migrated.push(receipt_path);
            }
            present.sort_unstable_by_key(|row| row.row);
            added += self.append(&tensor, present)?;
            self.append_lost(&tensor, lost)?;
            for path in migrated {
                match fs::remove_file(path) {
                    Err(error) if error.kind() != io::ErrorKind::NotFound => {
                        return Err(V41RangeCacheError::Io(error));
                    }
                    _ => {}
                }
            }
            // Only an emptied directory goes; anything left stays visible.
            let _ = fs::remove_dir(&dir);
        }
        let _ = fs::remove_dir(legacy);
        Ok(added)
    }

    /// Appends lost records (digest only) for rows not yet in `tensor`'s pack.
    fn append_lost(&self, tensor: &str, records: Vec<RowRecord>) -> Result<(), V41RangeCacheError> {
        if records.is_empty() {
            return Ok(());
        }
        fs::create_dir_all(&self.dir).map_err(V41RangeCacheError::Io)?;
        let mut lines = Vec::new();
        for record in &records {
            serde_json::to_writer(&mut lines, record).map_err(V41RangeCacheError::ReceiptJson)?;
            lines.push(b'\n');
        }
        let mut index = fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(self.dir.join(format!("{tensor}.index")))
            .map_err(V41RangeCacheError::Io)?;
        index.write_all(&lines).map_err(V41RangeCacheError::Io)?;
        index.sync_data().map_err(V41RangeCacheError::Io)?;
        let mut packs = self.lock();
        let pack = packs.entry(tensor.to_owned()).or_default();
        for record in records {
            pack.rows.insert(record.row, record);
        }
        Ok(())
    }
}

/// Replays `tensor`'s index; see the module docs.
fn recover(dir: &Path, tensor: &str) -> io::Result<Pack> {
    let index_path = dir.join(format!("{tensor}.index"));
    let text = fs::read(&index_path)?;
    let mut rows = HashMap::new();
    let mut lines = 0;
    let mut valid = 0;
    for line in text.split_inclusive(|&byte| byte == b'\n') {
        if line.last() != Some(&b'\n') {
            break;
        }
        let Ok(record) = serde_json::from_slice::<RowRecord>(&line[..line.len() - 1]) else {
            break;
        };
        valid += line.len();
        lines += 1;
        rows.insert(record.row, record);
    }
    let pack_path = dir.join(format!("{tensor}.pack"));
    let file_len = match fs::metadata(&pack_path) {
        Ok(metadata) => metadata.len(),
        Err(error) if error.kind() == io::ErrorKind::NotFound => 0,
        Err(error) => return Err(error),
    };
    let mut changed = valid < text.len() || lines != rows.len();
    for record in rows.values_mut() {
        if !record.lost && record.offset + record.length > file_len {
            record.lost = true;
            changed = true;
        }
    }
    let data_len = rows
        .values()
        .filter(|record| !record.lost)
        .map(|record| record.offset + record.length)
        .max()
        .unwrap_or(0);
    if changed {
        let mut sorted: Vec<_> = rows.values().collect();
        sorted.sort_unstable_by_key(|record| record.row);
        let mut compacted = Vec::new();
        for record in sorted {
            serde_json::to_writer(&mut compacted, record).map_err(io::Error::other)?;
            compacted.push(b'\n');
        }
        super::write_atomically(&index_path, &compacted)
            .map_err(|error| io::Error::other(error.to_string()))?;
    }
    if data_len < file_len {
        fs::OpenOptions::new()
            .write(true)
            .open(&pack_path)?
            .set_len(data_len)?;
    }
    Ok(Pack { rows, data_len })
}

fn io_error(message: String) -> V41RangeCacheError {
    V41RangeCacheError::Io(io::Error::other(message))
}
