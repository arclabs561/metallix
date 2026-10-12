//! Read-only access to verified whole tensors and CLI `.row-packs` stores.
//!
//! Row indexes are replayed without recovery: later records win, but malformed
//! or torn lines fail. No files are created, repaired, migrated or fetched.

use std::{
    collections::BTreeMap,
    fs,
    io::{self, BufRead as _, BufReader, Read as _, Seek as _, SeekFrom},
    ops::Range,
    path::PathBuf,
};

use serde::Deserialize;
use sha2::{Digest, Sha256};

use super::range_cache::{
    V41LocalWeightsSource, V41RangeCacheError, V41RangeRequest, V41RangeSource,
};

const MAX_INDEX_BYTES: u64 = 64 * 1024 * 1024;
const MAX_LINE_BYTES: u64 = 64 * 1024;
const MAX_ROW_READ_BYTES: u64 = 64 * 1024 * 1024;

/// One index line: where a row's bytes are and what they must hash to.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
struct RowRecord {
    row: u64,
    offset: u64,
    length: u64,
    sha256: String,
    shard: String,
    /// Absolute shard byte range of the row.
    range: [u64; 2],
    revision: String,
    /// The bytes are gone; only the digest is kept.
    #[serde(default)]
    lost: bool,
}

/// Reads acquired whole tensors first, then verified packed rows for subranges.
///
/// Whole tensors use [`V41LocalWeightsSource`]'s receipt and digest checks. Only
/// its `NotLocal` outcome permits row fallback; other failures are preserved.
/// Packed reads require complete rows along the tensor's first dimension and
/// are limited to 64 MiB per request. Each requested tensor index is replayed
/// lazily, under a 64 MiB total and 64 KiB per-line cap. Every returned row is
/// checked against its revision, shard, absolute range, length and SHA-256.
/// Missing, lost, corrupt or truncated rows fail without acquisition or repair.
#[derive(Clone, Debug)]
pub struct V41LocalStoreSource {
    local: V41LocalWeightsSource,
    dir: PathBuf,
    revision: String,
}

impl V41LocalStoreSource {
    /// Selects the weights directory and pinned revision without performing I/O.
    #[must_use]
    pub fn new(weights: PathBuf, revision: String) -> Self {
        let dir = weights.join(".row-packs");
        Self {
            local: V41LocalWeightsSource::new(weights, &revision),
            dir,
            revision,
        }
    }

    fn read_rows(&self, request: &V41RangeRequest<'_>) -> Result<Vec<u8>, V41RangeCacheError> {
        let (row_bytes, rows) = whole_rows(request)
            .ok_or_else(|| invalid("packed reads require complete nonempty rows"))?;
        let length = request.range.end - request.range.start;
        if length > MAX_ROW_READ_BYTES {
            return Err(V41RangeCacheError::BudgetTooSmall {
                item_bytes: length,
                budget_bytes: MAX_ROW_READ_BYTES,
            });
        }
        let records = self.read_index(request.tensor, &rows)?;
        let mut file = fs::File::open(self.dir.join(format!("{}.pack", request.tensor)))
            .map_err(V41RangeCacheError::Io)?;
        let file_len = file.metadata().map_err(V41RangeCacheError::Io)?.len();
        let mut output = Vec::new();
        let length = usize::try_from(length).map_err(|_| V41RangeCacheError::Allocation)?;
        output
            .try_reserve_exact(length)
            .map_err(|_| V41RangeCacheError::Allocation)?;
        for row in rows {
            let record = records
                .get(&row)
                .ok_or_else(|| invalid("requested packed row is missing"))?;
            let start = request.tensor_range.file_range().start + row * row_bytes;
            let mismatch = |field| V41RangeCacheError::ReceiptMismatch {
                tensor: request.tensor.to_owned(),
                field,
            };
            if record.revision != self.revision {
                return Err(mismatch("revision"));
            }
            if record.shard != request.shard {
                return Err(mismatch("shard"));
            }
            if record.range != [start, start + row_bytes] {
                return Err(mismatch("range"));
            }
            if record.length != row_bytes {
                return Err(mismatch("length"));
            }
            if record.lost {
                return Err(invalid("requested packed row is marked lost"));
            }
            if record
                .offset
                .checked_add(record.length)
                .is_none_or(|end| end > file_len)
            {
                return Err(V41RangeCacheError::Io(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    "packed row extends beyond its data file",
                )));
            }
            file.seek(SeekFrom::Start(record.offset))
                .map_err(V41RangeCacheError::Io)?;
            let begin = output.len();
            output.resize(
                begin + usize::try_from(row_bytes).map_err(|_| V41RangeCacheError::Allocation)?,
                0,
            );
            file.read_exact(&mut output[begin..])
                .map_err(V41RangeCacheError::Io)?;
            if format!("{:x}", Sha256::digest(&output[begin..])) != record.sha256 {
                return Err(V41RangeCacheError::HashMismatch(format!(
                    "{} row {row}",
                    request.tensor
                )));
            }
        }
        Ok(output)
    }

    fn read_index(
        &self,
        tensor: &str,
        rows: &Range<u64>,
    ) -> Result<BTreeMap<u64, RowRecord>, V41RangeCacheError> {
        let file = fs::File::open(self.dir.join(format!("{tensor}.index")))
            .map_err(V41RangeCacheError::Io)?;
        if file.metadata().map_err(V41RangeCacheError::Io)?.len() > MAX_INDEX_BYTES {
            return Err(invalid("packed row index exceeds 64 MiB"));
        }
        let mut reader = BufReader::new(file);
        let mut total = 0_u64;
        let mut records = BTreeMap::new();
        loop {
            let mut line = Vec::new();
            let count = reader
                .by_ref()
                .take(MAX_LINE_BYTES + 1)
                .read_until(b'\n', &mut line)
                .map_err(V41RangeCacheError::Io)?;
            if count == 0 {
                break;
            }
            total += count as u64;
            if count as u64 > MAX_LINE_BYTES || total > MAX_INDEX_BYTES {
                return Err(invalid("packed row index exceeds its read bounds"));
            }
            if line.last() != Some(&b'\n') {
                return Err(invalid("torn packed row index line"));
            }
            let record: RowRecord =
                serde_json::from_slice(&line).map_err(V41RangeCacheError::ReceiptJson)?;
            if rows.contains(&record.row) {
                records.insert(record.row, record);
            }
        }
        Ok(records)
    }
}

impl V41RangeSource for V41LocalStoreSource {
    fn read_range(&self, request: &V41RangeRequest<'_>) -> Result<Vec<u8>, V41RangeCacheError> {
        match self.local.read_range(request) {
            Err(V41RangeCacheError::NotLocal(_))
                if request.range != request.tensor_range.file_range() =>
            {
                self.read_rows(request)
            }
            result => result,
        }
    }
}

fn whole_rows(request: &V41RangeRequest<'_>) -> Option<(u64, Range<u64>)> {
    let tensor = request.tensor_range.file_range();
    let rows = *request.tensor_range.shape().first()?;
    let tensor_bytes = tensor.end - tensor.start;
    if rows == 0 || !tensor_bytes.is_multiple_of(rows) {
        return None;
    }
    let row_bytes = tensor_bytes / rows;
    let offset = request.range.start.checked_sub(tensor.start)?;
    let length = request.range.end.checked_sub(request.range.start)?;
    if row_bytes == 0
        || length == 0
        || request.range.end > tensor.end
        || !offset.is_multiple_of(row_bytes)
        || !length.is_multiple_of(row_bytes)
    {
        return None;
    }
    Some((row_bytes, offset / row_bytes..(offset + length) / row_bytes))
}

fn invalid(message: &str) -> V41RangeCacheError {
    V41RangeCacheError::Io(io::Error::new(io::ErrorKind::InvalidData, message))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::checkpoint::V41SafetensorsHeader;
    use std::sync::atomic::{AtomicUsize, Ordering};

    static NEXT: AtomicUsize = AtomicUsize::new(0);

    struct Fixture {
        dir: PathBuf,
        header: V41SafetensorsHeader,
        record: serde_json::Value,
    }

    impl Fixture {
        fn new() -> Self {
            let dir = std::env::temp_dir().join(format!(
                "metallix-local-store-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            fs::create_dir_all(dir.join(".row-packs")).expect("fixture directory");
            let json = br#"{"table":{"dtype":"U8","shape":[3,2],"data_offsets":[0,6]}}"#;
            let header =
                V41SafetensorsHeader::parse(json, 8 + json.len() as u64 + 6).expect("header");
            let start = header.tensor("table").expect("table").file_range().start + 2;
            // Fields mirror the CLI RowRecord writer, with a nonzero row and pack offset.
            let record = serde_json::json!({"row":1,"offset":2,"length":2,
                "sha256":format!("{:x}", Sha256::digest([3,4])),
                "shard":"model.safetensors","range":[start,start+2],"revision":"pinned"});
            let fixture = Self {
                dir,
                header,
                record,
            };
            fs::write(fixture.pack(), [9, 9, 3, 4]).expect("pack");
            fixture.write_record(&fixture.record);
            fixture
        }
        fn pack(&self) -> PathBuf {
            self.dir.join(".row-packs/table.pack")
        }
        fn index(&self) -> PathBuf {
            self.dir.join(".row-packs/table.index")
        }
        fn write_record(&self, record: &serde_json::Value) {
            fs::write(self.index(), format!("{record}\n")).expect("index");
        }
        fn read(&self) -> Result<Vec<u8>, V41RangeCacheError> {
            let tensor_range = self.header.tensor("table").expect("table");
            let start = tensor_range.file_range().start + 2;
            V41LocalStoreSource::new(self.dir.clone(), "pinned".to_owned()).read_range(
                &V41RangeRequest {
                    tensor: "table",
                    shard: "model.safetensors",
                    tensor_range,
                    range: start..start + 2,
                },
            )
        }
    }
    impl Drop for Fixture {
        fn drop(&mut self) {
            fs::remove_dir_all(&self.dir).expect("remove owned fixture");
        }
    }

    #[test]
    fn intact_row_is_verified_without_changing_store() {
        let fixture = Fixture::new();
        let before = fs::read(fixture.index()).expect("index");
        assert_eq!(fixture.read().expect("row"), [3, 4]);
        assert_eq!(fs::read(fixture.index()).expect("index"), before);
        assert_eq!(fs::read(fixture.pack()).expect("pack"), [9, 9, 3, 4]);
    }

    #[test]
    fn missing_row_and_missing_pack_fail() {
        let fixture = Fixture::new();
        fs::write(fixture.index(), []).expect("empty index");
        assert!(
            matches!(fixture.read(), Err(V41RangeCacheError::Io(error)) if error.kind() == io::ErrorKind::InvalidData)
        );
        fixture.write_record(&fixture.record);
        fs::remove_file(fixture.pack()).expect("remove fixture pack");
        assert!(
            matches!(fixture.read(), Err(V41RangeCacheError::Io(error)) if error.kind() == io::ErrorKind::NotFound)
        );
    }

    #[test]
    fn corrupt_and_truncated_payloads_fail() {
        let fixture = Fixture::new();
        fs::write(fixture.pack(), [9, 9, 3, 5]).expect("corrupt pack");
        assert!(matches!(
            fixture.read(),
            Err(V41RangeCacheError::HashMismatch(_))
        ));
        fs::write(fixture.pack(), [9, 9, 3]).expect("truncate pack");
        assert!(
            matches!(fixture.read(), Err(V41RangeCacheError::Io(error)) if error.kind() == io::ErrorKind::UnexpectedEof)
        );
    }

    #[test]
    fn torn_index_and_lost_records_are_not_recovered() {
        let fixture = Fixture::new();
        let torn = format!("{}\n{{", fixture.record);
        fs::write(fixture.index(), &torn).expect("torn index");
        assert!(
            matches!(fixture.read(), Err(V41RangeCacheError::Io(error)) if error.kind() == io::ErrorKind::InvalidData)
        );
        assert_eq!(
            fs::read(fixture.index()).expect("unchanged index"),
            torn.as_bytes()
        );
        let mut lost = fixture.record.clone();
        lost["lost"] = true.into();
        fixture.write_record(&lost);
        assert!(
            matches!(fixture.read(), Err(V41RangeCacheError::Io(error)) if error.kind() == io::ErrorKind::InvalidData)
        );
    }

    #[test]
    fn row_identity_and_length_must_match_request() {
        let fixture = Fixture::new();
        for (field, value) in [
            ("revision", serde_json::json!("wrong")),
            ("shard", serde_json::json!("wrong.safetensors")),
            ("range", serde_json::json!([0, 2])),
            ("length", serde_json::json!(3)),
        ] {
            let mut record = fixture.record.clone();
            record[field] = value;
            fixture.write_record(&record);
            assert!(
                matches!(fixture.read(), Err(V41RangeCacheError::ReceiptMismatch { field: actual, .. }) if actual == field)
            );
        }
    }

    #[test]
    fn oversized_index_is_rejected_before_reading() {
        let fixture = Fixture::new();
        fs::OpenOptions::new()
            .write(true)
            .open(fixture.index())
            .expect("index")
            .set_len(MAX_INDEX_BYTES + 1)
            .expect("sparse index");
        assert!(
            matches!(fixture.read(), Err(V41RangeCacheError::Io(error)) if error.kind() == io::ErrorKind::InvalidData)
        );
    }

    #[test]
    fn newest_record_wins_in_append_order() {
        let fixture = Fixture::new();
        let mut lost = fixture.record.clone();
        lost["lost"] = true.into();
        fs::write(fixture.index(), format!("{}\n{lost}\n", fixture.record)).expect("index");
        assert!(
            matches!(fixture.read(), Err(V41RangeCacheError::Io(error)) if error.kind() == io::ErrorKind::InvalidData)
        );
        fs::write(fixture.index(), format!("{lost}\n{}\n", fixture.record)).expect("index");
        assert_eq!(fixture.read().expect("restored row"), [3, 4]);
    }

    #[test]
    fn whole_tensor_takes_precedence_over_invalid_index() {
        let fixture = Fixture::new();
        let bytes = [1, 2, 7, 8, 5, 6];
        let range = fixture.header.tensor("table").expect("table").file_range();
        let receipt = serde_json::json!({
            "tensor":"table","shard":"model.safetensors","revision":"pinned",
            "range":[range.start,range.end],"bytes":6,
            "metadata":{"dtype":"U8","shape":[3,2]},
            "sha256":format!("{:x}", Sha256::digest(bytes)),
        });
        fs::write(fixture.dir.join("table.bin"), bytes).expect("whole payload");
        fs::write(fixture.dir.join("table.receipt.json"), receipt.to_string())
            .expect("whole receipt");
        fs::write(fixture.index(), b"{").expect("invalid row index");
        assert_eq!(fixture.read().expect("whole tensor row"), [7, 8]);
    }

    #[test]
    fn whole_tensor_corruption_does_not_fall_back_to_valid_pack() {
        let fixture = Fixture::new();
        fs::write(fixture.dir.join("table.receipt.json"), b"{").expect("invalid whole receipt");
        assert!(matches!(
            fixture.read(),
            Err(V41RangeCacheError::ReceiptJson(_))
        ));
    }
}
