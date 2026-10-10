//! Routed experts served from a checkpoint under a byte budget.
//!
//! A mixture-of-experts layer runs a few of its experts per token. When the
//! experts do not all fit in memory, [`ExpertCache`] keeps the most recently
//! used ones, and [`ExpertPager`] reads the others' bytes from the checkpoint
//! with a fixed pool of reader threads. An [`ExpertLocator`] (one per model
//! family, in the model crate) says which byte ranges hold an expert. The
//! cache is generic over the payload, so this crate stays free of MLX: the
//! model crate decides what a resident expert is (device arrays, say).
//!
//! The cache changes only where expert bytes come from. Which experts run,
//! their values, and the order of any reduction are the model's.

use std::{
    collections::{BTreeMap, HashMap, VecDeque},
    ops::Range,
    sync::{Arc, Condvar, Mutex, PoisonError},
    thread::{self, JoinHandle},
};

use crate::{CheckpointError, TensorSource};

/// One routed expert of one layer.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ExpertKey {
    /// Decoder layer index.
    pub layer: u16,
    /// Expert index within the layer.
    pub expert: u16,
}

/// One stored byte range that is part of an expert.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExpertRange {
    /// The stored tensor holding the range.
    pub tensor: String,
    /// Absolute byte range inside that tensor's payload.
    pub range: Range<u64>,
}

/// Where a model family stores each routed expert.
pub trait ExpertLocator {
    /// The ranges that make up `key`, in a fixed order the model's
    /// materialization relies on (MLX 4-bit: codes, scales and biases of
    /// each projection, so nine).
    ///
    /// # Errors
    /// The expert is out of range or its tensors are missing or misshapen.
    fn ranges(&self, key: ExpertKey) -> Result<Vec<ExpertRange>, CheckpointError>;
}

/// Total bytes of `ranges`.
#[must_use]
pub fn range_bytes(ranges: &[ExpertRange]) -> u64 {
    ranges
        .iter()
        .map(|part| part.range.end - part.range.start)
        .sum()
}

/// A cache refusal.
#[derive(Debug, thiserror::Error, Eq, PartialEq)]
#[non_exhaustive]
pub enum ExpertCacheError {
    /// Making room would evict an expert the current call still needs.
    #[error("{bytes} more bytes do not fit {budget} once the {pinned} pinned experts are kept")]
    Full {
        /// Bytes the insert needs.
        bytes: u64,
        /// The cache budget.
        budget: u64,
        /// Experts the caller pinned.
        pinned: usize,
    },
}

struct Entry<V> {
    value: V,
    bytes: u64,
    tick: u64,
}

/// Least-recently-used experts under a byte budget, across all layers.
///
/// One cache serves every layer: recency is global, so a layer whose
/// experts repeat keeps them while a layer that churns gives room back.
pub struct ExpertCache<V> {
    budget: u64,
    used: u64,
    tick: u64,
    entries: HashMap<ExpertKey, Entry<V>>,
    /// Tick to key, oldest first.
    order: BTreeMap<u64, ExpertKey>,
}

impl<V> ExpertCache<V> {
    /// An empty cache that never holds more than `budget` bytes.
    #[must_use]
    pub fn new(budget: u64) -> Self {
        Self {
            budget,
            used: 0,
            tick: 0,
            entries: HashMap::new(),
            order: BTreeMap::new(),
        }
    }

    /// The byte budget.
    #[must_use]
    pub const fn budget(&self) -> u64 {
        self.budget
    }

    /// Bytes held now; never more than the budget.
    #[must_use]
    pub const fn used_bytes(&self) -> u64 {
        self.used
    }

    /// Experts held now.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the cache holds nothing.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Whether `key` is held, without counting as a use.
    #[must_use]
    pub fn contains(&self, key: ExpertKey) -> bool {
        self.entries.contains_key(&key)
    }

    /// The held payload of `key`, marked most recently used.
    pub fn get(&mut self, key: ExpertKey) -> Option<&V> {
        self.tick += 1;
        let tick = self.tick;
        let entry = self.entries.get_mut(&key)?;
        self.order.remove(&entry.tick);
        entry.tick = tick;
        self.order.insert(tick, key);
        Some(&entry.value)
    }

    /// Holds `value` for `key` as most recently used, first evicting the
    /// least recently used experts not in `pinned` until `bytes` fit.
    /// Returns what was evicted, oldest first; an existing entry for `key` is
    /// replaced.
    ///
    /// # Errors
    /// [`ExpertCacheError::Full`] when `bytes` cannot fit without evicting
    /// a pinned expert. The cache is then unchanged except for evictions
    /// already made.
    pub fn insert(
        &mut self,
        key: ExpertKey,
        value: V,
        bytes: u64,
        pinned: &[ExpertKey],
    ) -> Result<Vec<(ExpertKey, V)>, ExpertCacheError> {
        let mut evicted = Vec::new();
        if let Some(old) = self.remove(key) {
            evicted.push((key, old));
        }
        let budget = self.budget;
        let full = || ExpertCacheError::Full {
            bytes,
            budget,
            pinned: pinned.len(),
        };
        if bytes > self.budget {
            return Err(full());
        }
        while self.used + bytes > self.budget {
            let victim = self
                .order
                .values()
                .find(|candidate| !pinned.contains(candidate))
                .copied();
            let Some(victim) = victim else {
                return Err(full());
            };
            if let Some(value) = self.remove(victim) {
                evicted.push((victim, value));
            }
        }
        self.tick += 1;
        self.order.insert(self.tick, key);
        self.entries.insert(
            key,
            Entry {
                value,
                bytes,
                tick: self.tick,
            },
        );
        self.used += bytes;
        debug_assert!(self.used <= self.budget);
        Ok(evicted)
    }

    fn remove(&mut self, key: ExpertKey) -> Option<V> {
        let entry = self.entries.remove(&key)?;
        self.order.remove(&entry.tick);
        self.used -= entry.bytes;
        Some(entry.value)
    }
}

/// One expert's bytes as read, one buffer per [`ExpertRange`] in the
/// locator's order.
#[derive(Debug)]
pub struct ExpertBytes {
    /// The expert.
    pub key: ExpertKey,
    /// The ranges' bytes, in order.
    pub parts: Vec<Vec<u8>>,
}

type Job = Box<dyn FnOnce() + Send>;

struct Queue {
    jobs: Mutex<(VecDeque<Job>, bool)>,
    ready: Condvar,
}

/// A fixed pool of reader threads over one [`TensorSource`].
///
/// Each range is one `read_range` call on a pool thread, so the ranges of
/// one expert, and of all experts a call misses, are in flight together:
/// up to `threads` reads queued at the device at once.
pub struct ExpertPager<S> {
    source: Arc<S>,
    queue: Arc<Queue>,
    workers: Vec<JoinHandle<()>>,
}

impl<S: TensorSource + Send + Sync + 'static> ExpertPager<S> {
    /// A pager reading from `source` with `threads` reader threads (at
    /// least one).
    #[must_use]
    pub fn new(source: Arc<S>, threads: usize) -> Self {
        let queue = Arc::new(Queue {
            jobs: Mutex::new((VecDeque::new(), false)),
            ready: Condvar::new(),
        });
        let workers = (0..threads.max(1))
            .map(|_| {
                let queue = Arc::clone(&queue);
                thread::spawn(move || {
                    loop {
                        let job = {
                            let mut jobs =
                                queue.jobs.lock().unwrap_or_else(PoisonError::into_inner);
                            loop {
                                if let Some(job) = jobs.0.pop_front() {
                                    break job;
                                }
                                if jobs.1 {
                                    return;
                                }
                                jobs = queue
                                    .ready
                                    .wait(jobs)
                                    .unwrap_or_else(PoisonError::into_inner);
                            }
                        };
                        job();
                    }
                })
            })
            .collect();
        Self {
            source,
            queue,
            workers,
        }
    }

    /// The source the pager reads.
    #[must_use]
    pub fn source(&self) -> &Arc<S> {
        &self.source
    }

    /// Reads every range of every expert in `wanted` (each `(key, ranges)`)
    /// on the pool and returns their bytes in the same order. Blocks until
    /// all reads finish.
    ///
    /// # Errors
    /// The first failed read; the other reads still complete first.
    pub fn fetch(
        &self,
        wanted: Vec<(ExpertKey, Vec<ExpertRange>)>,
    ) -> Result<Vec<ExpertBytes>, CheckpointError> {
        type Slot = Option<Result<Vec<u8>, CheckpointError>>;
        let total: usize = wanted.iter().map(|(_, ranges)| ranges.len()).sum();
        let done = Arc::new((Mutex::new((Vec::<Slot>::new(), 0_usize)), Condvar::new()));
        done.0
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .0
            .resize_with(total, || None);
        let mut shapes = Vec::with_capacity(wanted.len());
        {
            let mut jobs = self
                .queue
                .jobs
                .lock()
                .unwrap_or_else(PoisonError::into_inner);
            let mut slot = 0;
            for (key, ranges) in wanted {
                shapes.push((key, ranges.len()));
                for part in ranges {
                    let source = Arc::clone(&self.source);
                    let done = Arc::clone(&done);
                    let index = slot;
                    slot += 1;
                    jobs.0.push_back(Box::new(move || {
                        let result = read_part(&*source, &part);
                        let (lock, signal) = &*done;
                        let mut state = lock.lock().unwrap_or_else(PoisonError::into_inner);
                        state.0[index] = Some(result);
                        state.1 += 1;
                        signal.notify_all();
                    }));
                }
            }
        }
        self.queue.ready.notify_all();
        let (lock, signal) = &*done;
        let mut state = lock.lock().unwrap_or_else(PoisonError::into_inner);
        while state.1 < total {
            state = signal.wait(state).unwrap_or_else(PoisonError::into_inner);
        }
        let mut results = std::mem::take(&mut state.0).into_iter();
        drop(state);
        let mut experts = Vec::with_capacity(shapes.len());
        for (key, count) in shapes {
            // Every slot is filled: the wait above ends only when all
            // `total` reads have reported.
            let parts = results
                .by_ref()
                .take(count)
                .flatten()
                .collect::<Result<Vec<_>, _>>()?;
            experts.push(ExpertBytes { key, parts });
        }
        Ok(experts)
    }
}

fn read_part<S: TensorSource + ?Sized>(
    source: &S,
    part: &ExpertRange,
) -> Result<Vec<u8>, CheckpointError> {
    let len = usize::try_from(part.range.end - part.range.start).map_err(|_| {
        CheckpointError::OutsideTensor {
            tensor: part.tensor.clone(),
            range: part.range.clone(),
            len: usize::MAX,
        }
    })?;
    let mut bytes = vec![0; len];
    source.read_range(&part.tensor, part.range.clone(), &mut bytes)?;
    Ok(bytes)
}

impl<S> Drop for ExpertPager<S> {
    fn drop(&mut self) {
        self.queue
            .jobs
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .1 = true;
        self.queue.ready.notify_all();
        for worker in self.workers.drain(..) {
            let _ = worker.join();
        }
    }
}

#[cfg(test)]
mod tests;
