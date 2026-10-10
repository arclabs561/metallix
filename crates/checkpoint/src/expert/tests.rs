use std::{
    collections::HashMap,
    ops::Range,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
};

use super::{ExpertCache, ExpertCacheError, ExpertKey, ExpertPager, ExpertRange, range_bytes};
use crate::{CheckpointError, DType, TensorEncoding, TensorInfo, TensorSource};

const fn key(layer: u16, expert: u16) -> ExpertKey {
    ExpertKey { layer, expert }
}

#[test]
fn evicts_the_least_recently_used_and_keeps_the_budget() {
    let mut cache = ExpertCache::new(30);
    for expert in 0..3 {
        assert!(
            cache
                .insert(key(0, expert), expert, 10, &[])
                .unwrap()
                .is_empty()
        );
    }
    assert_eq!(cache.used_bytes(), 30);
    // Using expert 0 makes expert 1 the oldest.
    assert_eq!(cache.get(key(0, 0)), Some(&0));
    let evicted = cache.insert(key(1, 7), 7, 10, &[]).unwrap();
    assert_eq!(evicted, [(key(0, 1), 1)]);
    assert!(!cache.contains(key(0, 1)));
    // A 20-byte entry evicts the two oldest, 2 then 0.
    let evicted = cache.insert(key(1, 8), 8, 20, &[]).unwrap();
    assert_eq!(evicted, [(key(0, 2), 2), (key(0, 0), 0)]);
    assert_eq!((cache.used_bytes(), cache.len()), (30, 2));
    assert!(cache.used_bytes() <= cache.budget());
}

#[test]
fn pinned_experts_survive_and_a_full_cache_refuses() {
    let mut cache = ExpertCache::new(20);
    cache.insert(key(0, 0), (), 10, &[]).unwrap();
    cache.insert(key(0, 1), (), 10, &[]).unwrap();
    // Expert 0 is oldest but pinned, so expert 1 goes.
    let evicted = cache.insert(key(0, 2), (), 10, &[key(0, 0)]).unwrap();
    assert_eq!(evicted, [(key(0, 1), ())]);
    assert_eq!(
        cache.insert(key(0, 3), (), 10, &[key(0, 0), key(0, 2)]),
        Err(ExpertCacheError::Full {
            bytes: 10,
            budget: 20,
            pinned: 2
        })
    );
    assert!(cache.contains(key(0, 0)) && cache.contains(key(0, 2)));
    assert!(matches!(
        cache.insert(key(0, 4), (), 21, &[]),
        Err(ExpertCacheError::Full { bytes: 21, .. })
    ));
    // Replacing an entry returns the old payload and recounts its bytes.
    let mut cache = ExpertCache::new(20);
    cache.insert(key(0, 0), 'a', 15, &[]).unwrap();
    let evicted = cache.insert(key(0, 0), 'b', 5, &[]).unwrap();
    assert_eq!(evicted, [(key(0, 0), 'a')]);
    assert_eq!(cache.used_bytes(), 5);
}

/// In-memory tensors that count `read_range` calls; `fail` names a tensor
/// whose reads fail.
struct Memory {
    tensors: HashMap<String, (TensorInfo, Vec<u8>)>,
    reads: AtomicUsize,
    fail: Option<&'static str>,
}

impl TensorSource for Memory {
    fn names(&self) -> Vec<&str> {
        self.tensors.keys().map(String::as_str).collect()
    }

    fn tensor(&self, name: &str) -> Option<&TensorInfo> {
        self.tensors.get(name).map(|(info, _)| info)
    }

    fn read(&self, name: &str, _max_bytes: u64) -> Result<Vec<u8>, CheckpointError> {
        unreachable!("the pager reads ranges only, not {name}")
    }

    fn read_range(
        &self,
        name: &str,
        range: Range<u64>,
        dst: &mut [u8],
    ) -> Result<(), CheckpointError> {
        self.reads.fetch_add(1, Ordering::Relaxed);
        if self.fail == Some(name) {
            return Err(CheckpointError::MissingTensor(name.to_owned()));
        }
        let (info, bytes) = &self.tensors[name];
        let start = usize::try_from(range.start - info.range().start).unwrap();
        dst.copy_from_slice(&bytes[start..start + dst.len()]);
        Ok(())
    }
}

/// Two `[4 experts, 3 bytes]` tensors, `up` and `down`, whose byte `i` is
/// `i` (`up`) and `100 + i` (`down`).
fn memory(fail: Option<&'static str>) -> Memory {
    let tensor = |offset: u64, base: u8| {
        (
            TensorInfo::new(
                TensorEncoding::Plain(DType::U8),
                vec![4, 3],
                offset..offset + 12,
            ),
            (base..base + 12).collect(),
        )
    };
    Memory {
        tensors: HashMap::from([
            ("up".to_owned(), tensor(0, 0)),
            ("down".to_owned(), tensor(12, 100)),
        ]),
        reads: AtomicUsize::new(0),
        fail,
    }
}

fn ranges(source: &Memory, expert: u16) -> Vec<ExpertRange> {
    ["up", "down"]
        .into_iter()
        .map(|tensor| ExpertRange {
            tensor: tensor.to_owned(),
            range: source.tensors[tensor]
                .0
                .slice_dim0(u64::from(expert))
                .unwrap(),
        })
        .collect()
}

#[test]
fn the_pager_reads_every_range_in_locator_order() {
    let source = Arc::new(memory(None));
    let wanted: Vec<_> = [3, 0, 2]
        .into_iter()
        .map(|expert| (key(5, expert), ranges(&source, expert)))
        .collect();
    assert_eq!(range_bytes(&wanted[0].1), 6);
    let pager = ExpertPager::new(Arc::clone(&source), 4);
    let experts = pager.fetch(wanted).unwrap();
    assert_eq!(
        experts.iter().map(|bytes| bytes.key).collect::<Vec<_>>(),
        [key(5, 3), key(5, 0), key(5, 2)]
    );
    assert_eq!(experts[0].parts, [vec![9, 10, 11], vec![109, 110, 111]]);
    assert_eq!(experts[1].parts, [vec![0, 1, 2], vec![100, 101, 102]]);
    assert_eq!(
        source.reads.load(Ordering::Relaxed),
        6,
        "one read per range"
    );
    // Reusable across calls, and an empty fetch returns at once.
    assert!(pager.fetch(Vec::new()).unwrap().is_empty());
    assert_eq!(
        pager.fetch(vec![(key(0, 1), ranges(&source, 1))]).unwrap()[0].parts[1],
        [103, 104, 105]
    );
}

#[test]
fn a_failed_read_fails_the_fetch_after_the_others_finish() {
    let source = Arc::new(memory(Some("down")));
    let pager = ExpertPager::new(Arc::clone(&source), 2);
    let wanted = vec![
        (key(0, 0), ranges(&source, 0)),
        (key(0, 1), ranges(&source, 1)),
    ];
    assert!(matches!(
        pager.fetch(wanted),
        Err(CheckpointError::MissingTensor(name)) if name == "down"
    ));
    assert_eq!(source.reads.load(Ordering::Relaxed), 4);
}
