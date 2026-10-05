//! Token-embedding rows served from the checkpoint range cache.
//!
//! The real V4.1 `embed.weight` is BF16 `[129280, 5120]` (1.3 GB); a startup
//! step reads only the rows of its distinct token IDs.

use std::sync::Mutex;

use super::range_cache::{V41RangeCache, V41RangeSource};
use crate::reduced::{EmbeddingRowSource, StartupSessionError};

/// Lends `embed.weight` rows from a shared [`V41RangeCache`].
///
/// Each maximal run of consecutive requested rows is one range request.
#[derive(Debug)]
pub struct V41CachedEmbeddingRows<'a, S> {
    cache: &'a Mutex<V41RangeCache<S>>,
    width: usize,
}

impl<'a, S> V41CachedEmbeddingRows<'a, S> {
    /// Serves `embed.weight` rows of `width` BF16 values (5120 for V4.1 Flash).
    pub const fn new(cache: &'a Mutex<V41RangeCache<S>>, width: usize) -> Self {
        Self { cache, width }
    }
}

impl<S: V41RangeSource> EmbeddingRowSource for V41CachedEmbeddingRows<'_, S> {
    fn read_rows(&self, rows: &[usize], output: &mut [u16]) -> Result<(), StartupSessionError> {
        let unavailable = |reason: String| StartupSessionError::RowsUnavailable { reason };
        if output.len() != rows.len() * self.width {
            return Err(unavailable(format!(
                "output holds {} values, expected {}",
                output.len(),
                rows.len() * self.width
            )));
        }
        let mut cache = self
            .cache
            .lock()
            .map_err(|_| unavailable("range cache lock poisoned".to_owned()))?;
        let mut first = 0;
        while first < rows.len() {
            let mut end = first + 1;
            while end < rows.len() && rows[end] == rows[end - 1] + 1 {
                end += 1;
            }
            let range = rows[first] as u64..rows[end - 1] as u64 + 1;
            let bytes = cache
                .get_rows("embed.weight", range.clone())
                .map_err(|error| unavailable(error.to_string()))?;
            let span = &mut output[first * self.width..end * self.width];
            if bytes.len() != span.len() * 2 {
                return Err(unavailable(format!(
                    "embed.weight rows {range:?} hold {} bytes, expected {}",
                    bytes.len(),
                    span.len() * 2
                )));
            }
            for (value, pair) in span.iter_mut().zip(bytes.chunks_exact(2)) {
                *value = u16::from_le_bytes([pair[0], pair[1]]);
            }
            first = end;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::{cell::RefCell, sync::Mutex};

    use super::V41CachedEmbeddingRows;
    use crate::{
        checkpoint::{
            V41SafetensorsHeader,
            range_cache::{V41RangeCache, V41RangeCacheError, V41RangeRequest, V41RangeSource},
        },
        manifest::V41SafetensorsIndex,
        reduced::{EmbeddingRowSource, StartupSessionError},
    };

    const SHARD: &str = "model-00001-of-00001.safetensors";
    const ROWS: usize = 6;
    const WIDTH: usize = 32;

    /// Row `r`, feature `f` holds BF16 bits `0x3f00 + 64 r + f`.
    fn value(row: usize, feature: usize) -> u16 {
        u16::try_from(0x3f00 + 64 * row + feature).expect("small")
    }

    struct Shard {
        payload_start: u64,
        bytes: Vec<u8>,
        reads: RefCell<Vec<(u64, u64)>>,
    }

    impl V41RangeSource for &Shard {
        fn read_range(&self, request: &V41RangeRequest<'_>) -> Result<Vec<u8>, V41RangeCacheError> {
            self.reads
                .borrow_mut()
                .push((request.range.start, request.range.end));
            let start = usize::try_from(request.range.start - self.payload_start).expect("small");
            let end = usize::try_from(request.range.end - self.payload_start).expect("small");
            Ok(self.bytes[start..end].to_vec())
        }
    }

    #[test]
    fn cached_rows_are_the_table_rows_in_request_order() {
        let header_json = format!(
            r#"{{"embed.weight":{{"dtype":"BF16","shape":[{ROWS},{WIDTH}],"data_offsets":[0,{}]}}}}"#,
            ROWS * WIDTH * 2
        );
        let payload_start = 8 + header_json.len() as u64;
        let header = V41SafetensorsHeader::parse(
            header_json.as_bytes(),
            payload_start + (ROWS * WIDTH * 2) as u64,
        )
        .expect("synthetic header");
        let index = V41SafetensorsIndex::parse(&format!(
            r#"{{"metadata":{{"total_size":1}},"weight_map":{{"embed.weight":"{SHARD}"}}}}"#
        ))
        .expect("synthetic index");
        let shard = Shard {
            payload_start,
            bytes: (0..ROWS)
                .flat_map(|row| (0..WIDTH).map(move |feature| value(row, feature)))
                .flat_map(u16::to_le_bytes)
                .collect(),
            reads: RefCell::new(Vec::new()),
        };
        let cache = Mutex::new(
            V41RangeCache::new(&shard, index, [(SHARD.to_owned(), header)], 1 << 16)
                .expect("cache"),
        );
        let rows = V41CachedEmbeddingRows::new(&cache, WIDTH);
        let mut output = vec![0; 3 * WIDTH];
        rows.read_rows(&[1, 2, 4], &mut output).expect("rows");
        let expected: Vec<u16> = [1, 2, 4]
            .into_iter()
            .flat_map(|row| (0..WIDTH).map(move |feature| value(row, feature)))
            .collect();
        assert_eq!(output, expected);
        // Rows 1..3 and 4 are two range requests.
        let row = |r: u64| payload_start + r * (WIDTH as u64) * 2;
        assert_eq!(*shard.reads.borrow(), [(row(1), row(3)), (row(4), row(5))]);
        assert!(matches!(
            rows.read_rows(&[6], &mut [0; WIDTH]),
            Err(StartupSessionError::RowsUnavailable { .. })
        ));
    }
}
