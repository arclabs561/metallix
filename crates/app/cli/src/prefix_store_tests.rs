use std::{fs, path::PathBuf};

use super::{EXTENSION, PrefixStore, StoredPrefix, TEMPORARY};

/// A fresh directory for one test.
fn root(name: &str) -> PathBuf {
    let root = std::env::temp_dir().join(format!(
        "metallix-prefix-store-{}-{name}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&root);
    root
}

fn entries(store: &PrefixStore) -> Vec<PathBuf> {
    let mut paths = fs::read_dir(store.dir())
        .expect("store directory")
        .map(|item| item.expect("entry").path())
        .collect::<Vec<_>>();
    paths.sort();
    paths
}

#[test]
fn a_reopened_store_finds_the_longest_stored_prefix() {
    let root = root("reopen");
    {
        let store = PrefixStore::open(&root, b"model-a", 1 << 20).expect("open");
        assert!(store.save(None, vec![1, 2], 0, vec![7; 10]));
        assert!(store.save(None, vec![1, 2, 3, 4], 1, vec![8; 20]));
        store.flush();
        assert!(!store.save(None, vec![1, 2], 0, vec![9; 5]), "stored twice");
        assert!(store.contains(None, &[1, 2]));
        assert!(!store.contains(None, &[1]));
        assert!(!store.contains(Some(""), &[1, 2]));
    }
    let store = PrefixStore::open(&root, b"model-a", 1 << 20).expect("reopen");
    assert_eq!(
        store.lookup(None, &[1, 2, 3, 4, 5]),
        Some(StoredPrefix {
            tokens: vec![1, 2, 3, 4],
            role: 1,
            snapshot: vec![8; 20],
        })
    );
    // The whole prompt is never a hit: one token must remain to prefill.
    assert_eq!(
        store.lookup(None, &[1, 2, 3, 4]).map(|hit| hit.tokens),
        Some(vec![1, 2])
    );
    assert_eq!(store.lookup(None, &[9, 1, 2, 3]), None);
    drop(store);
    let _ = fs::remove_dir_all(&root);
}

#[test]
fn salts_and_identities_never_share_entries() {
    let root = root("isolation");
    let store = PrefixStore::open(&root, b"model-a", 1 << 20).expect("open");
    store.save(Some("tenant-a"), vec![1, 2], 0, vec![1; 4]);
    store.save(None, vec![5, 6], 0, vec![2; 4]);
    store.flush();
    drop(store);
    let store = PrefixStore::open(&root, b"model-a", 1 << 20).expect("reopen");
    assert!(store.lookup(Some("tenant-a"), &[1, 2, 3]).is_some());
    assert_eq!(store.lookup(Some("tenant-b"), &[1, 2, 3]), None);
    assert_eq!(
        store.lookup(None, &[1, 2, 3]),
        None,
        "salted against unsalted"
    );
    assert_eq!(
        store.lookup(Some(""), &[5, 6, 7]),
        None,
        "empty salt against none"
    );
    assert!(store.lookup(None, &[5, 6, 7]).is_some());
    // Another model state reads another directory.
    let other = PrefixStore::open(&root, b"model-b", 1 << 20).expect("other identity");
    assert_ne!(other.dir(), store.dir());
    assert_eq!(other.lookup(None, &[5, 6, 7]), None);
    drop((store, other));
    let _ = fs::remove_dir_all(&root);
}

#[test]
fn truncated_foreign_and_partial_files_are_removed_not_read() {
    let root = root("damage");
    let store = PrefixStore::open(&root, b"model-a", 1 << 20).expect("open");
    store.save(None, vec![1, 2], 0, vec![3; 64]);
    store.save(None, vec![4, 5], 0, vec![6; 64]);
    store.flush();
    let paths = entries(&store);
    assert_eq!(paths.len(), 2);
    let dir = store.dir().to_path_buf();
    drop(store);
    // Truncate one entry, overwrite the other's identity, and leave a
    // partial write behind as a crash before the rename would.
    let truncated = fs::read(&paths[0]).expect("read");
    fs::write(&paths[0], &truncated[..truncated.len() - 1]).expect("truncate");
    let mut foreign = fs::read(&paths[1]).expect("read");
    foreign[12] ^= 0xff;
    fs::write(&paths[1], &foreign).expect("corrupt identity");
    let partial = dir.join(format!("{TEMPORARY}1-1"));
    fs::write(&partial, b"half an entry").expect("partial");
    let store = PrefixStore::open(&root, b"model-a", 1 << 20).expect("reopen");
    assert!(entries(&store).is_empty(), "{:?}", entries(&store));
    assert_eq!(store.lookup(None, &[1, 2, 3]), None);
    assert_eq!(store.lookup(None, &[4, 5, 6]), None);
    drop(store);
    let _ = fs::remove_dir_all(&root);
}

#[test]
fn an_entry_damaged_after_opening_is_dropped_on_lookup() {
    let root = root("late-damage");
    let store = PrefixStore::open(&root, b"model-a", 1 << 20).expect("open");
    store.save(None, vec![1, 2], 0, vec![3; 64]);
    store.flush();
    let path = entries(&store).remove(0);
    let bytes = fs::read(&path).expect("read");
    fs::write(&path, &bytes[..bytes.len() / 2]).expect("truncate");
    assert_eq!(store.lookup(None, &[1, 2, 3]), None);
    assert!(!path.exists(), "the damaged entry was removed");
    drop(store);
    let _ = fs::remove_dir_all(&root);
}

#[test]
fn the_budget_evicts_the_least_recently_used_entry() {
    let root = root("budget");
    // Each entry is its 1000-byte snapshot plus a header under 100 bytes.
    let store = PrefixStore::open(&root, b"model-a", 2_300).expect("open");
    store.save(None, vec![1], 0, vec![0; 1000]);
    store.save(None, vec![2], 0, vec![0; 1000]);
    store.flush();
    assert!(
        store.lookup(None, &[1, 9]).is_some(),
        "touch the first entry"
    );
    store.save(None, vec![3], 0, vec![0; 1000]);
    store.flush();
    assert!(store.lookup(None, &[1, 9]).is_some());
    assert_eq!(
        store.lookup(None, &[2, 9]),
        None,
        "least recently used went"
    );
    assert!(store.lookup(None, &[3, 9]).is_some());
    assert_eq!(
        entries(&store)
            .iter()
            .filter(|path| path
                .extension()
                .is_some_and(|extension| extension == EXTENSION))
            .count(),
        2
    );
    // An entry larger than the whole budget is not stored.
    store.save(None, vec![4], 0, vec![0; 5000]);
    store.flush();
    assert_eq!(store.lookup(None, &[4, 9]), None);
    drop(store);
    let _ = fs::remove_dir_all(&root);
}

fn write_safetensors(path: &std::path::Path, header: &str, data: usize) {
    let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
    bytes.extend_from_slice(header.as_bytes());
    bytes.extend(std::iter::repeat_n(0_u8, data));
    fs::write(path, bytes).expect("write safetensors");
}

#[test]
fn the_weights_fingerprint_follows_headers_sizes_and_the_hub_revision() {
    let root = root("fingerprint");
    let model = root
        .join("models--org--name")
        .join("snapshots")
        .join("abc123");
    fs::create_dir_all(&model).expect("model dir");
    let header = r#"{"w":{"dtype":"BF16","shape":[2],"data_offsets":[0,4]}}"#;
    write_safetensors(&model.join("model.safetensors"), header, 4);
    fs::write(model.join("config.json"), "{}").expect("config");
    let first = super::weights_fingerprint(&model).expect("fingerprint");
    assert_eq!(super::weights_fingerprint(&model).expect("again"), first);
    assert!(first.ends_with("@abc123"), "{first}");
    // Another header, or the same header over more data, is another model.
    write_safetensors(
        &model.join("model.safetensors"),
        &header.replace("[2]", "[1]"),
        4,
    );
    let reshaped = super::weights_fingerprint(&model).expect("reshaped");
    assert_ne!(reshaped, first);
    write_safetensors(&model.join("model.safetensors"), header, 8);
    assert_ne!(super::weights_fingerprint(&model).expect("resized"), first);
    // Outside a hub snapshot there is no revision suffix.
    let plain = root.join("plain");
    fs::create_dir_all(&plain).expect("plain dir");
    write_safetensors(&plain.join("model.safetensors"), header, 4);
    assert!(
        !super::weights_fingerprint(&plain)
            .expect("plain")
            .contains('@')
    );
    let empty = root.join("empty");
    fs::create_dir_all(&empty).expect("empty dir");
    assert!(super::weights_fingerprint(&empty).is_err());
    let _ = fs::remove_dir_all(&root);
}
