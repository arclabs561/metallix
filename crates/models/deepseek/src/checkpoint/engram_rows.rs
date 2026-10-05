//! Engram embedding rows served from the checkpoint range cache.
//!
//! A real V4.1 Engram table (`layers.{1,14}.engram.embed.weight`, FP8 E4M3,
//! `[rows, 256]`, with `.scale`, E8M0, `[rows, 8]`) is about 100 GB, so a
//! lookup fetches only the rows its hash IDs select.

use std::sync::Mutex;

use super::range_cache::{V41RangeCache, V41RangeSource};
use crate::engram::embedding::{EngramEmbeddingError, EngramRowSource};

/// Lends one layer's Engram table rows from a shared [`V41RangeCache`].
///
/// Each maximal run of consecutive requested rows is one weight and one
/// scale range request. Rows are copied out under the lock, so the lock is
/// never held during lookup arithmetic.
#[derive(Debug)]
pub struct V41CachedEngramRows<'a, S> {
    cache: &'a Mutex<V41RangeCache<S>>,
    layer: usize,
    width: usize,
}

impl<'a, S> V41CachedEngramRows<'a, S> {
    /// Serves `layer`'s Engram table with `width` features per row (256 for
    /// V4.1 Flash).
    pub const fn new(cache: &'a Mutex<V41RangeCache<S>>, layer: usize, width: usize) -> Self {
        Self {
            cache,
            layer,
            width,
        }
    }
}

impl<S: V41RangeSource> EngramRowSource for V41CachedEngramRows<'_, S> {
    fn read_rows(
        &self,
        rows: &[usize],
        codes: &mut [u8],
        scales: &mut [u8],
    ) -> Result<(), EngramEmbeddingError> {
        let scale_width = self.width / 32;
        for (field, actual, expected) in [
            ("row codes", codes.len(), rows.len() * self.width),
            ("row scales", scales.len(), rows.len() * scale_width),
        ] {
            if actual != expected {
                return Err(EngramEmbeddingError::Length {
                    field,
                    actual,
                    expected,
                });
            }
        }
        let unavailable = |reason: String| EngramEmbeddingError::RowsUnavailable { reason };
        let weight = format!("layers.{}.engram.embed.weight", self.layer);
        let scale = format!("layers.{}.engram.embed.scale", self.layer);
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
            for (table, destination, row_width) in [
                (&weight, &mut *codes, self.width),
                (&scale, &mut *scales, scale_width),
            ] {
                let bytes = cache
                    .get_rows(table, range.clone())
                    .map_err(|error| unavailable(error.to_string()))?;
                let span = first * row_width..end * row_width;
                if bytes.len() != span.len() {
                    return Err(unavailable(format!(
                        "{table} rows {range:?} hold {} bytes, expected {}",
                        bytes.len(),
                        span.len()
                    )));
                }
                destination[span].copy_from_slice(&bytes);
            }
            first = end;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::{cell::RefCell, sync::Mutex};

    use super::V41CachedEngramRows;
    use crate::{
        checkpoint::{
            V41SafetensorsHeader,
            range_cache::{V41RangeCache, V41RangeCacheError, V41RangeRequest, V41RangeSource},
        },
        engram::embedding::{
            EngramEmbeddingError, EngramEmbeddingLayout, EngramRowSource,
            engram_embedding_bf16_from_source, engram_embedding_bf16_reference,
        },
        manifest::V41SafetensorsIndex,
    };

    const SHARD: &str = "model-00001-of-00001.safetensors";
    const ROWS: usize = 6;
    const WIDTH: usize = 64;

    /// The full synthetic layer-1 table: varied finite codes and scales.
    fn table() -> (Vec<u8>, Vec<u8>) {
        let codes = (0..ROWS * WIDTH)
            .map(|i| u8::try_from((i * 37 + 11) % 0x7e).expect("code"))
            .collect();
        let scales = (0..ROWS * WIDTH / 32)
            .map(|i| u8::try_from(120 + i % 9).expect("scale"))
            .collect();
        (codes, scales)
    }

    /// Serves the synthetic table's bytes at their absolute shard offsets.
    struct Shard {
        payload_start: u64,
        bytes: Vec<u8>,
        reads: RefCell<Vec<(String, u64, u64)>>,
    }

    impl V41RangeSource for &Shard {
        fn read_range(&self, request: &V41RangeRequest<'_>) -> Result<Vec<u8>, V41RangeCacheError> {
            self.reads.borrow_mut().push((
                request.tensor.to_owned(),
                request.range.start,
                request.range.end,
            ));
            let start = usize::try_from(request.range.start - self.payload_start).expect("small");
            let end = usize::try_from(request.range.end - self.payload_start).expect("small");
            Ok(self.bytes[start..end].to_vec())
        }
    }

    /// A one-shard checkpoint holding only layer 1's Engram weight and scale.
    fn shard() -> (Shard, V41SafetensorsHeader, V41SafetensorsIndex) {
        let (codes, scales) = table();
        let weight_end = codes.len();
        let scale_end = weight_end + scales.len();
        let header_json = format!(
            r#"{{"layers.1.engram.embed.weight":{{"dtype":"F8_E4M3","shape":[{ROWS},{WIDTH}],"data_offsets":[0,{weight_end}]}},"layers.1.engram.embed.scale":{{"dtype":"F8_E8M0","shape":[{ROWS},{}],"data_offsets":[{weight_end},{scale_end}]}}}}"#,
            WIDTH / 32
        );
        let payload_start = 8 + header_json.len() as u64;
        let header =
            V41SafetensorsHeader::parse(header_json.as_bytes(), payload_start + scale_end as u64)
                .expect("synthetic Engram header");
        let index = V41SafetensorsIndex::parse(&format!(
            r#"{{"metadata":{{"total_size":1}},"weight_map":{{"layers.1.engram.embed.weight":"{SHARD}","layers.1.engram.embed.scale":"{SHARD}"}}}}"#
        ))
        .expect("synthetic index");
        let shard = Shard {
            payload_start,
            bytes: [codes, scales].concat(),
            reads: RefCell::new(Vec::new()),
        };
        (shard, header, index)
    }

    #[test]
    fn cached_rows_reproduce_the_owned_table_lookup() {
        let (shard, header, index) = shard();
        let cache = Mutex::new(
            V41RangeCache::new(&shard, index, [(SHARD.to_owned(), header)], 1 << 16)
                .expect("cache"),
        );
        let rows = V41CachedEngramRows::new(&cache, 1, WIDTH);
        let ids = [4, -1, 1, 2, 4, 9, 0];
        let (codes, scales) = table();
        let mut owned = vec![0; ids.len() * WIDTH];
        engram_embedding_bf16_reference(
            &ids,
            &codes,
            &scales,
            EngramEmbeddingLayout::new(ROWS, WIDTH, 32).expect("layout"),
            &mut owned,
        )
        .expect("owned lookup");
        let mut cached = vec![0; ids.len() * WIDTH];
        engram_embedding_bf16_from_source(&ids, ROWS, WIDTH, &rows, &mut cached)
            .expect("cached lookup");
        assert_eq!(cached, owned);
        assert!(owned.iter().any(|&bits| bits != 0));

        // Rows 0..3 and 4 are two runs; rows 3 and 5 are never fetched.
        let row = |r: u64| shard.payload_start + r * WIDTH as u64;
        let scale = |r: u64| shard.payload_start + (ROWS * WIDTH) as u64 + r * 2;
        let weight = "layers.1.engram.embed.weight".to_owned();
        let scale_name = "layers.1.engram.embed.scale".to_owned();
        assert_eq!(
            *shard.reads.borrow(),
            [
                (weight.clone(), row(0), row(3)),
                (scale_name.clone(), scale(0), scale(3)),
                (weight, row(4), row(5)),
                (scale_name, scale(4), scale(5)),
            ]
        );
    }

    #[test]
    fn missing_tables_rows_and_geometry_fail_closed() {
        let (shard, header, index) = shard();
        let cache = Mutex::new(
            V41RangeCache::new(&shard, index, [(SHARD.to_owned(), header)], 1 << 16)
                .expect("cache"),
        );
        let unavailable = |result: Result<(), EngramEmbeddingError>, needle: &str| matches!(result, Err(EngramEmbeddingError::RowsUnavailable { ref reason }) if reason.contains(needle));
        // Layer 14 has no table in this checkpoint.
        let absent = V41CachedEngramRows::new(&cache, 14, WIDTH);
        assert!(unavailable(
            absent.read_rows(&[0], &mut [0; WIDTH], &mut [0; 2]),
            "layers.14.engram.embed.weight"
        ));
        // A row past the table is refused, not zero-filled.
        let rows = V41CachedEngramRows::new(&cache, 1, WIDTH);
        assert!(unavailable(
            rows.read_rows(&[ROWS], &mut [0; WIDTH], &mut [0; 2]),
            "not a valid range"
        ));
        // A width that disagrees with the table's row bytes is refused.
        let narrow = V41CachedEngramRows::new(&cache, 1, 32);
        assert!(unavailable(
            narrow.read_rows(&[0], &mut [0; 32], &mut [0; 1]),
            "expected 32"
        ));
        assert!(matches!(
            rows.read_rows(&[0], &mut [0; WIDTH - 1], &mut [0; 2]),
            Err(EngramEmbeddingError::Length { .. })
        ));
        // A lookup over the failing source leaves its output untouched.
        let mut output = vec![7; WIDTH];
        assert!(
            engram_embedding_bf16_from_source(&[0], ROWS, WIDTH, &absent, &mut output).is_err()
        );
        assert_eq!(output, vec![7; WIDTH]);
    }
}
