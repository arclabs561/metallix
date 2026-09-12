//! Bounded scalar V4.1 Engram n-gram hash-address reference.
//!
//! The caller supplies already-compressed token IDs and explicit, captured
//! per-layer hash tensors. This deliberately does not normalize tokenizer text,
//! generate primes or RNG multipliers, read embedding tables, or run a GPU.

pub mod gate;

use thiserror::Error;

/// Largest number of cached compressed-token slots accepted by this reference.
pub const MAX_ENGRAM_HISTORY: usize = 1 << 20;
const MAX_ENGRAM_LAYOUT_ELEMENTS: usize = 1 << 20;
const MAX_ENGRAM_OUTPUTS: usize = 1 << 20;
const MAX_ENGRAM_WORK: usize = 1 << 24;

/// One already-compressed token participating in Engram history.
///
/// `Dead` models an image/masked span. It is deliberately distinct from a
/// live compressed ID, including the configured compressed pad ID.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CompressedToken {
    /// A non-negative compressed token ID.
    Live(i64),
    /// A token that must break every n-gram spanning it.
    Dead,
}

/// Explicit, already-derived hash tensors for all configured Engram layers.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EngramHashLayout {
    max_ngram_size: usize,
    heads: usize,
    layers: usize,
    compressed_pad_id: i64,
    primes: Vec<i64>,
    offsets: Vec<i64>,
    multipliers: Vec<i64>,
}

impl EngramHashLayout {
    /// Validates explicit per-layer multipliers, bucket divisors, and offsets.
    ///
    /// `primes` is flattened `[layer, ngram(2..=max), head]`; `offsets` has
    /// the same flattening; and `multipliers` is `[layer, lookback(0..max)]`.
    /// This accepts a positive divisor rather than proving primality: upstream
    /// prime generation is intentionally outside this pure reference.
    ///
    /// # Errors
    ///
    /// Returns [`EngramHashError`] for incompatible dimensions, unsafe scalar
    /// inputs, or unbounded derived work.
    pub fn new(
        max_ngram_size: usize,
        heads: usize,
        layers: usize,
        compressed_pad_id: i64,
        primes: Vec<i64>,
        offsets: Vec<i64>,
        multipliers: Vec<i64>,
    ) -> Result<Self, EngramHashError> {
        if max_ngram_size < 2 || heads == 0 || layers == 0 {
            return Err(EngramHashError::InvalidLayout);
        }
        if compressed_pad_id < 0 {
            return Err(EngramHashError::NegativePadId {
                id: compressed_pad_id,
            });
        }
        let columns = columns(max_ngram_size, heads)?;
        let expected_hashes = checked_product(&[layers, columns], "hash tensors")?;
        let expected_multipliers = checked_product(&[layers, max_ngram_size], "multipliers")?;
        for (field, elements) in [
            ("primes", primes.len()),
            ("offsets", offsets.len()),
            ("multipliers", multipliers.len()),
            ("hash tensors", expected_hashes),
            ("multiplier tensor", expected_multipliers),
        ] {
            if elements > MAX_ENGRAM_LAYOUT_ELEMENTS {
                return Err(EngramHashError::LayoutLimit { field, elements });
            }
        }
        check_length("primes", primes.len(), expected_hashes)?;
        check_length("offsets", offsets.len(), expected_hashes)?;
        check_length("multipliers", multipliers.len(), expected_multipliers)?;
        for (index, &divisor) in primes.iter().enumerate() {
            if divisor <= 0 {
                return Err(EngramHashError::NonPositiveDivisor { index, divisor });
            }
        }
        for (index, &offset) in offsets.iter().enumerate() {
            if offset < 0 {
                return Err(EngramHashError::NegativeOffset { index, offset });
            }
        }
        for (index, &multiplier) in multipliers.iter().enumerate() {
            if multiplier < 0 {
                return Err(EngramHashError::NegativeMultiplier { index, multiplier });
            }
        }
        Ok(Self {
            max_ngram_size,
            heads,
            layers,
            compressed_pad_id,
            primes,
            offsets,
            multipliers,
        })
    }

    /// Returns the configured maximum n-gram size.
    #[must_use]
    pub const fn max_ngram_size(&self) -> usize {
        self.max_ngram_size
    }

    /// Returns the configured number of hash heads per n-gram.
    #[must_use]
    pub const fn heads(&self) -> usize {
        self.heads
    }

    /// Returns the configured Engram-layer count.
    #[must_use]
    pub const fn layers(&self) -> usize {
        self.layers
    }

    /// Returns the compressed ID used for blocked history positions.
    #[must_use]
    pub const fn compressed_pad_id(&self) -> i64 {
        self.compressed_pad_id
    }

    fn columns(&self) -> usize {
        // Constructor has already checked this product.
        (self.max_ngram_size - 1) * self.heads
    }

    fn hash_index(&self, layer: usize, ngram_index: usize, head: usize) -> usize {
        (layer * (self.max_ngram_size - 1) + ngram_index) * self.heads + head
    }

    fn multiplier(&self, layer: usize, lookback: usize) -> i64 {
        self.multipliers[layer * self.max_ngram_size + lookback]
    }
}

/// Stateful, bounded compressed-token history for one fixed batch shape.
#[derive(Clone, Debug)]
pub struct EngramHashState {
    layout: EngramHashLayout,
    batches: usize,
    capacity: usize,
    history: Vec<Option<CompressedToken>>,
}

impl EngramHashState {
    /// Allocates empty history for a fixed number of batch rows and positions.
    ///
    /// An attempt to hash a nonzero `start_position` before its required prior
    /// history has been written returns an error rather than reading undefined
    /// cache state. A `start_position` of zero clears the candidate history
    /// before writing, which is stricter than the upstream caller-owned cache
    /// lifecycle and prevents an abbreviated reset from exposing a prior tail.
    ///
    /// # Errors
    ///
    /// Returns [`EngramHashError`] when the history shape is invalid or cannot
    /// be allocated within the fixed scalar bound.
    pub fn new(
        layout: EngramHashLayout,
        batches: usize,
        capacity: usize,
    ) -> Result<Self, EngramHashError> {
        if batches == 0 || capacity == 0 {
            return Err(EngramHashError::EmptyHistoryDimension);
        }
        let history_elements = checked_product(&[batches, capacity], "history")?;
        if history_elements > MAX_ENGRAM_HISTORY {
            return Err(EngramHashError::HistoryLimit {
                elements: history_elements,
            });
        }
        let mut history = Vec::new();
        history.try_reserve_exact(history_elements).map_err(|_| {
            EngramHashError::AllocationFailed {
                elements: history_elements,
            }
        })?;
        history.resize(history_elements, None);
        Ok(Self {
            layout,
            batches,
            capacity,
            history,
        })
    }

    /// Returns the number of independent batch histories.
    #[must_use]
    pub const fn batches(&self) -> usize {
        self.batches
    }

    /// Returns the absolute position capacity per batch history.
    #[must_use]
    pub const fn capacity(&self) -> usize {
        self.capacity
    }

    /// Writes a compressed-token chunk and returns its row addresses.
    ///
    /// `tokens` is batch-major `[batch, positions]`. Returned addresses are
    /// flattened `[batch, positions, layer, ngram(2..=max), head]`, with head
    /// varying fastest. All validation and calculation complete against a copy
    /// of the history before this state is updated.
    ///
    /// # Errors
    ///
    /// Returns [`EngramHashError`] for shape, history, signed-range, output,
    /// work, or allocation failures; rejected calls leave history unchanged.
    pub fn write_and_hash(
        &mut self,
        tokens: &[CompressedToken],
        positions: usize,
        start_position: usize,
    ) -> Result<Vec<i64>, EngramHashError> {
        if positions == 0 {
            return Err(EngramHashError::EmptyChunk);
        }
        let expected_tokens = checked_product(&[self.batches, positions], "input")?;
        check_length("input", tokens.len(), expected_tokens)?;
        let end_position = start_position
            .checked_add(positions)
            .ok_or(EngramHashError::ShapeOverflow { field: "chunk end" })?;
        if end_position > self.capacity {
            return Err(EngramHashError::ChunkExceedsHistory {
                end_position,
                capacity: self.capacity,
            });
        }
        for (index, token) in tokens.iter().enumerate() {
            if let CompressedToken::Live(id) = token {
                if *id < 0 {
                    return Err(EngramHashError::NegativeLiveId { index, id: *id });
                }
            }
        }
        let output_elements = checked_product(
            &[
                self.batches,
                positions,
                self.layout.layers,
                self.layout.columns(),
            ],
            "output",
        )?;
        if output_elements > MAX_ENGRAM_OUTPUTS {
            return Err(EngramHashError::OutputLimit {
                elements: output_elements,
            });
        }
        let work = checked_product(
            &[
                self.batches,
                positions,
                self.layout.layers,
                self.layout.max_ngram_size,
                self.layout.heads,
            ],
            "work",
        )?;
        if work > MAX_ENGRAM_WORK {
            return Err(EngramHashError::WorkLimit { elements: work });
        }

        let mut next_history = Vec::new();
        next_history
            .try_reserve_exact(self.history.len())
            .map_err(|_| EngramHashError::AllocationFailed {
                elements: self.history.len(),
            })?;
        next_history.extend_from_slice(&self.history);
        if start_position == 0 {
            next_history.fill(None);
        }
        for batch in 0..self.batches {
            let source = &tokens[batch * positions..(batch + 1) * positions];
            let destination = batch * self.capacity + start_position;
            next_history[destination..destination + positions]
                .iter_mut()
                .zip(source)
                .for_each(|(slot, &token)| *slot = Some(token));
        }

        let output = self.hash_chunk(&next_history, positions, start_position, output_elements)?;
        self.history = next_history;
        Ok(output)
    }

    fn hash_chunk(
        &self,
        history: &[Option<CompressedToken>],
        positions: usize,
        start_position: usize,
        output_elements: usize,
    ) -> Result<Vec<i64>, EngramHashError> {
        let mut output = Vec::new();
        output.try_reserve_exact(output_elements).map_err(|_| {
            EngramHashError::AllocationFailed {
                elements: output_elements,
            }
        })?;
        for batch in 0..self.batches {
            for relative_position in 0..positions {
                let absolute_position = start_position + relative_position;
                for layer in 0..self.layout.layers {
                    // This follows the source's single cumulative `blocked`
                    // state: one history read and product per lookback, not a
                    // repeated scan of the complete lookback prefix.
                    let mut blocked = false;
                    let mut rolling = 0_i64;
                    for lookback in 0..self.layout.max_ngram_size {
                        let id = if blocked || absolute_position < lookback {
                            blocked = true;
                            self.layout.compressed_pad_id
                        } else {
                            let source_position = absolute_position - lookback;
                            let token = history[batch * self.capacity + source_position].ok_or(
                                EngramHashError::UninitializedHistory {
                                    batch,
                                    position: source_position,
                                },
                            )?;
                            match token {
                                CompressedToken::Dead => {
                                    blocked = true;
                                    self.layout.compressed_pad_id
                                }
                                CompressedToken::Live(id) => id,
                            }
                        };
                        let multiplier = self.layout.multiplier(layer, lookback);
                        let product =
                            id.checked_mul(multiplier)
                                .ok_or(EngramHashError::ProductOverflow {
                                    layer,
                                    lookback,
                                    id,
                                    multiplier,
                                })?;
                        if lookback == 0 {
                            rolling = product;
                            continue;
                        }
                        rolling ^= product;
                        let ngram_index = lookback - 1;
                        for head in 0..self.layout.heads {
                            let index = self.layout.hash_index(layer, ngram_index, head);
                            let bucket = self.layout.primes[index];
                            let remainder = rolling % bucket;
                            let address = remainder.checked_add(self.layout.offsets[index]).ok_or(
                                EngramHashError::AddressOverflow {
                                    layer,
                                    ngram_size: lookback + 1,
                                    head,
                                },
                            )?;
                            output.push(address);
                        }
                    }
                }
            }
        }
        Ok(output)
    }
}

/// Invalid input or bounded arithmetic in an Engram hash reference call.
#[derive(Clone, Debug, Error, Eq, PartialEq)]
#[non_exhaustive]
pub enum EngramHashError {
    /// N-gram size must permit at least a 2-gram, with nonempty layers and heads.
    #[error("Engram layout needs max n-gram size at least 2 and nonzero layers and heads")]
    InvalidLayout,
    /// A derived shape could not fit in `usize`.
    #[error("Engram {field} shape arithmetic overflowed")]
    ShapeOverflow { field: &'static str },
    /// An explicit tensor did not have its exact source-shaped length.
    #[error("Engram {field} length is {actual}, expected {expected}")]
    Length {
        /// Tensor role.
        field: &'static str,
        /// Actual length.
        actual: usize,
        /// Expected length.
        expected: usize,
    },
    /// The compressed pad ID must be non-negative.
    #[error("Engram compressed pad ID {id} is negative")]
    NegativePadId { id: i64 },
    /// A hash bucket divisor must be positive.
    #[error("Engram divisor at index {index} is {divisor}, expected positive")]
    NonPositiveDivisor { index: usize, divisor: i64 },
    /// Offsets are row-address bases and must be non-negative.
    #[error("Engram offset at index {index} is negative: {offset}")]
    NegativeOffset { index: usize, offset: i64 },
    /// Captured multipliers must be non-negative.
    #[error("Engram multiplier at index {index} is negative: {multiplier}")]
    NegativeMultiplier { index: usize, multiplier: i64 },
    /// History must have positive batch and absolute-position dimensions.
    #[error("Engram history requires nonzero batches and capacity")]
    EmptyHistoryDimension,
    /// The bounded scalar history would exceed its fixed cap.
    #[error("Engram history has {elements} slots, maximum is 1048576")]
    HistoryLimit { elements: usize },
    /// An explicit layout tensor would exceed the fixed scalar cap.
    #[error("Engram {field} has {elements} entries, maximum is 1048576")]
    LayoutLimit {
        /// Tensor role.
        field: &'static str,
        /// Entry count.
        elements: usize,
    },
    /// A request cannot hash an empty token span.
    #[error("Engram chunk positions must be nonzero")]
    EmptyChunk,
    /// The input has a negative live compressed ID.
    #[error("Engram live compressed ID at input index {index} is negative: {id}")]
    NegativeLiveId { index: usize, id: i64 },
    /// The requested absolute chunk lies outside state capacity.
    #[error("Engram chunk ends at {end_position}, beyond capacity {capacity}")]
    ChunkExceedsHistory {
        /// Exclusive chunk end.
        end_position: usize,
        /// Per-batch history capacity.
        capacity: usize,
    },
    /// A required earlier position was never written.
    #[error("Engram history for batch {batch}, position {position} was not initialized")]
    UninitializedHistory { batch: usize, position: usize },
    /// The bounded output vector would exceed its fixed cap.
    #[error("Engram output has {elements} addresses, maximum is 1048576")]
    OutputLimit { elements: usize },
    /// Scalar hashing would exceed its fixed operation cap.
    #[error("Engram work estimate {elements} exceeds maximum 16777216")]
    WorkLimit { elements: usize },
    /// A bounded intermediate/output vector could not be reserved.
    #[error("could not allocate {elements} Engram elements")]
    AllocationFailed { elements: usize },
    /// A live-ID/multiplier product cannot remain in signed 64-bit range.
    #[error("Engram product overflowed at layer {layer}, lookback {lookback}: {id} * {multiplier}")]
    ProductOverflow {
        layer: usize,
        lookback: usize,
        id: i64,
        multiplier: i64,
    },
    /// A bucket remainder plus its address offset overflowed signed 64-bit range.
    #[error("Engram address overflowed at layer {layer}, n-gram {ngram_size}, head {head}")]
    AddressOverflow {
        layer: usize,
        ngram_size: usize,
        head: usize,
    },
}

fn columns(max_ngram_size: usize, heads: usize) -> Result<usize, EngramHashError> {
    (max_ngram_size - 1)
        .checked_mul(heads)
        .ok_or(EngramHashError::ShapeOverflow { field: "columns" })
}

fn checked_product(values: &[usize], field: &'static str) -> Result<usize, EngramHashError> {
    values.iter().try_fold(1_usize, |total, &value| {
        total
            .checked_mul(value)
            .ok_or(EngramHashError::ShapeOverflow { field })
    })
}

fn check_length(
    field: &'static str,
    actual: usize,
    expected: usize,
) -> Result<(), EngramHashError> {
    if actual == expected {
        Ok(())
    } else {
        Err(EngramHashError::Length {
            field,
            actual,
            expected,
        })
    }
}

#[cfg(test)]
mod tests {
    use serde_json::Value;

    use super::{CompressedToken, EngramHashError, EngramHashLayout, EngramHashState};

    fn fixture() -> Value {
        serde_json::from_str(include_str!(
            "../../../../fixtures/deepseek-v41/engram-hash-reference.json"
        ))
        .expect("checked-in Engram fixture JSON")
    }

    fn numbers(value: &Value) -> Vec<i64> {
        match value {
            Value::Number(number) => vec![number.as_i64().expect("fixture integer")],
            Value::Array(values) => values.iter().flat_map(numbers).collect(),
            _ => panic!("expected nested fixture integers"),
        }
    }

    fn bools(value: &Value) -> Vec<bool> {
        match value {
            Value::Bool(value) => vec![*value],
            Value::Array(values) => values.iter().flat_map(bools).collect(),
            _ => panic!("expected nested fixture booleans"),
        }
    }

    fn tensor<'a>(fixture: &'a Value, name: &str) -> &'a Value {
        fixture["tensors"]
            .as_array()
            .expect("fixture tensors")
            .iter()
            .find(|candidate| candidate["name"] == name)
            .unwrap_or_else(|| panic!("fixture tensor {name}"))
    }

    fn layout(fixture: &Value) -> EngramHashLayout {
        let primes = numbers(&tensor(fixture, "primes")["values"]);
        let offsets = numbers(&tensor(fixture, "offsets")["values"]);
        let multipliers = numbers(&tensor(fixture, "multipliers")["values"]);
        EngramHashLayout::new(4, 2, 2, 42, primes, offsets, multipliers).expect("fixture layout")
    }

    fn fixture_tokens(fixture: &Value, name: &str) -> Vec<CompressedToken> {
        let raw = numbers(&fixture["input"][name]);
        let mask = bools(&fixture["input"]["token_mask"]);
        let token_map = numbers(&tensor(fixture, "token_map")["values"]);
        raw.into_iter()
            .zip(mask)
            .map(|(raw_id, enabled)| {
                if enabled {
                    CompressedToken::Live(token_map[usize::try_from(raw_id).expect("raw ID")])
                } else {
                    CompressedToken::Dead
                }
            })
            .collect()
    }

    fn run_chunks(
        layout: EngramHashLayout,
        tokens: &[CompressedToken],
        chunks: &[usize],
    ) -> Vec<i64> {
        let mut state = EngramHashState::new(layout, 2, 6).expect("fixture state");
        let mut start = 0;
        let mut per_batch = vec![Vec::new(), Vec::new()];
        for &width in chunks {
            let mut chunk = Vec::new();
            for batch in 0..2 {
                chunk.extend_from_slice(&tokens[batch * 6 + start..batch * 6 + start + width]);
            }
            let addresses = state
                .write_and_hash(&chunk, width, start)
                .expect("fixture chunk");
            let per_batch_len = width * 2 * 6;
            per_batch[0].extend_from_slice(&addresses[..per_batch_len]);
            per_batch[1].extend_from_slice(&addresses[per_batch_len..]);
            start += width;
        }
        per_batch.into_iter().flatten().collect()
    }

    #[test]
    fn explicit_hash_state_matches_independent_pinned_fixture() {
        let fixture = fixture();
        assert_eq!(fixture["schema_version"], 1);
        assert_eq!(
            fixture["source"]["revision"],
            "dba1be0a40aa45a94ad051997016db3960a90277"
        );
        assert_eq!(
            fixture["source"]["sha256"],
            "11f35ecbead8150c35aa002b3d180ef290b05a25afe883a11884f94d476d3897"
        );
        assert_eq!(fixture["source"]["symbol"], "NgramHashState.forward");
        assert_eq!(fixture["receipt"]["device"], "cpu");
        assert_eq!(fixture["receipt"]["torch_version"], "2.13.0");
        assert_eq!(fixture["explicit_state"]["raw_pad_id"], 5);
        assert_eq!(fixture["explicit_state"]["compressed_pad_id"], 42);
        let tokens = fixture_tokens(&fixture, "raw_token_ids");
        let expected = numbers(&tensor(&fixture, "one_shot_hashes")["values"]);
        assert_eq!(run_chunks(layout(&fixture), &tokens, &[6]), expected);
        assert_eq!(run_chunks(layout(&fixture), &tokens, &[3, 3]), expected);
        assert_eq!(
            run_chunks(layout(&fixture), &tokens, &[1, 1, 1, 1, 1, 1]),
            expected
        );

        let per_batch_output = 6 * 2 * 6;
        for batch in 0..2 {
            let mut state = EngramHashState::new(layout(&fixture), 1, 6).expect("single-row state");
            let actual = state
                .write_and_hash(&tokens[batch * 6..(batch + 1) * 6], 6, 0)
                .expect("single-row fixture prefill");
            assert_eq!(
                actual.as_slice(),
                &expected[batch * per_batch_output..(batch + 1) * per_batch_output]
            );
            assert!(state.history.iter().all(Option::is_some));
        }
    }

    #[test]
    fn dead_boundary_and_start_zero_overwrite_match_fixture_contract() {
        let fixture = fixture();
        let tokens = fixture_tokens(&fixture, "raw_token_ids");
        let mut state = EngramHashState::new(layout(&fixture), 2, 6).expect("fixture state");
        let dirty = tokens
            .iter()
            .map(|token| match token {
                CompressedToken::Live(id) => CompressedToken::Live(id + 1),
                CompressedToken::Dead => CompressedToken::Dead,
            })
            .collect::<Vec<_>>();
        state.write_and_hash(&dirty, 6, 0).expect("dirty prefill");
        let reset = state.write_and_hash(&tokens, 6, 0).expect("reset prefill");
        assert_eq!(
            reset,
            numbers(&tensor(&fixture, "one_shot_hashes")["values"])
        );

        let mut changed = fixture_tokens(&fixture, "raw_token_ids");
        // The capture changes raw IDs `[[0, 1], [6, 5]]` to `[[4, 6], [0, 7]]`.
        // These are their explicit compressed IDs, not raw-token values.
        changed[0] = CompressedToken::Live(9);
        changed[1] = CompressedToken::Live(6);
        changed[6] = CompressedToken::Live(7);
        // Row 1 position 1 remains masked even when its raw ID changes.
        assert_eq!(changed[7], CompressedToken::Dead);
        let dead = run_chunks(layout(&fixture), &changed, &[6]);
        let expected = numbers(&tensor(&fixture, "dead_boundary_hashes")["values"]);
        assert_eq!(dead, expected);
    }

    #[test]
    fn rejects_uninitialized_history_and_signed_overflow_without_mutation() {
        let layout = EngramHashLayout::new(2, 1, 1, 0, vec![3], vec![0], vec![i64::MAX, 1])
            .expect("explicit layout");
        let mut state = EngramHashState::new(layout, 1, 2).expect("state");
        assert!(matches!(
            state.write_and_hash(&[CompressedToken::Live(1)], 1, 1),
            Err(EngramHashError::UninitializedHistory { .. })
        ));
        assert!(matches!(
            state.write_and_hash(&[CompressedToken::Live(2)], 1, 0),
            Err(EngramHashError::ProductOverflow { .. })
        ));
        assert!(matches!(
            state.write_and_hash(&[CompressedToken::Live(1)], 1, 1),
            Err(EngramHashError::UninitializedHistory { .. })
        ));
    }

    #[test]
    fn short_start_zero_reset_invalidates_a_prior_tail_before_a_gap() {
        let layout = EngramHashLayout::new(2, 1, 1, 0, vec![3], vec![0], vec![1, 1])
            .expect("explicit layout");
        let mut state = EngramHashState::new(layout, 1, 4).expect("state");
        state
            .write_and_hash(
                &[
                    CompressedToken::Live(1),
                    CompressedToken::Live(2),
                    CompressedToken::Live(3),
                    CompressedToken::Live(4),
                ],
                4,
                0,
            )
            .expect("initial full prefill");
        state
            .write_and_hash(&[CompressedToken::Live(9)], 1, 0)
            .expect("short reset");
        assert!(matches!(
            state.write_and_hash(&[CompressedToken::Live(8)], 1, 2),
            Err(EngramHashError::UninitializedHistory {
                batch: 0,
                position: 1,
            })
        ));
        assert!(state.history[1..].iter().all(Option::is_none));
    }
}
