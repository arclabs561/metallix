use std::{
    fs,
    path::PathBuf,
    sync::atomic::{AtomicUsize, Ordering},
};

use super::{GgufArray, GgufFile, GgufValue};
use crate::{CheckpointError, GgufEncoding, TensorSource};

/// A GGUF v3 file built field by field, so each test can break one field.
#[derive(Clone)]
struct Builder {
    version: u32,
    keys: Vec<(String, u32, Vec<u8>)>,
    tensors: Vec<(String, Vec<u64>, u32, u64)>,
    alignment: u64,
    payload: Vec<u8>,
}

fn string(text: &str) -> Vec<u8> {
    let mut bytes = (text.len() as u64).to_le_bytes().to_vec();
    bytes.extend_from_slice(text.as_bytes());
    bytes
}

impl Builder {
    fn new() -> Self {
        Self {
            version: 3,
            keys: Vec::new(),
            tensors: Vec::new(),
            alignment: 32,
            payload: Vec::new(),
        }
    }

    fn key(mut self, name: &str, kind: u32, value: Vec<u8>) -> Self {
        self.keys.push((name.to_owned(), kind, value));
        self
    }

    /// `dims` innermost first, as the file stores them.
    fn tensor(mut self, name: &str, dims: &[u64], kind: u32, offset: u64) -> Self {
        self.tensors
            .push((name.to_owned(), dims.to_vec(), kind, offset));
        self
    }

    fn bytes(&self) -> Vec<u8> {
        let mut out = b"GGUF".to_vec();
        out.extend(self.version.to_le_bytes());
        out.extend((self.tensors.len() as u64).to_le_bytes());
        out.extend((self.keys.len() as u64).to_le_bytes());
        for (name, kind, value) in &self.keys {
            out.extend(string(name));
            out.extend(kind.to_le_bytes());
            out.extend(value);
        }
        for (name, dims, kind, offset) in &self.tensors {
            out.extend(string(name));
            out.extend(u32::try_from(dims.len()).expect("few dims").to_le_bytes());
            for dim in dims {
                out.extend(dim.to_le_bytes());
            }
            out.extend(kind.to_le_bytes());
            out.extend(offset.to_le_bytes());
        }
        while !(out.len() as u64).is_multiple_of(self.alignment) {
            out.push(0);
        }
        out.extend(&self.payload);
        out
    }

    fn write(&self) -> TempFile {
        TempFile::new(&self.bytes())
    }
}

struct TempFile(PathBuf);

impl TempFile {
    fn new(bytes: &[u8]) -> Self {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let path = std::env::temp_dir().join(format!(
            "checkpoint-gguf-{}-{}.gguf",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::write(&path, bytes).expect("write temp GGUF");
        Self(path)
    }
}

impl Drop for TempFile {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.0);
    }
}

fn string_array(items: &[&str]) -> Vec<u8> {
    let mut bytes = 8_u32.to_le_bytes().to_vec();
    bytes.extend((items.len() as u64).to_le_bytes());
    for item in items {
        bytes.extend(string(item));
    }
    bytes
}

/// Two tensors: an F32 `[2, 3]` matrix and a `Q8_0` `[1, 32]` row, with the
/// tokenizer keys a byte-level BPE file carries.
fn valid() -> Builder {
    let mut payload: Vec<u8> = (0..6_u8).flat_map(|v| f32::from(v).to_le_bytes()).collect();
    payload.resize(32, 0);
    payload.extend(0x3c00_u16.to_le_bytes());
    payload.extend((0..32).map(|v: u8| v));
    Builder::new()
        .key("general.architecture", 8, string("qwen35"))
        .key("qwen35.block_count", 4, 24_u32.to_le_bytes().to_vec())
        .key("qwen35.rope.freq_base", 6, 1.0e7_f32.to_le_bytes().to_vec())
        .key("general.base_model.count", 4, 1_u32.to_le_bytes().to_vec())
        .key(
            "general.base_model.0.repo_url",
            8,
            string("https://huggingface.co/Qwen/Qwen3.5-0.8B"),
        )
        .key("tokenizer.ggml.model", 8, string("gpt2"))
        .key("tokenizer.ggml.pre", 8, string("qwen35"))
        .key("tokenizer.ggml.tokens", 9, string_array(&["a", "b", "ab"]))
        .key("tokenizer.ggml.merges", 9, string_array(&["a b"]))
        .key(
            "tokenizer.ggml.eos_token_id",
            4,
            2_u32.to_le_bytes().to_vec(),
        )
        .key("tokenizer.ggml.add_bos_token", 7, vec![0])
        .key("tokenizer.chat_template", 8, string("{{ messages }}"))
        .tensor("matrix", &[3, 2], 0, 0)
        .tensor("row", &[32, 1], 8, 32)
        .with_payload(payload)
}

impl Builder {
    fn with_payload(mut self, payload: Vec<u8>) -> Self {
        self.payload = payload;
        self
    }
}

fn open(builder: &Builder) -> Result<GgufFile, CheckpointError> {
    let file = builder.write();
    let opened = GgufFile::open(&file.0);
    // Keep the file alive only for header checks; reads use `opened_file`.
    drop(file);
    opened
}

#[test]
fn reads_metadata_shapes_and_payloads() {
    let file = valid().write();
    let gguf = GgufFile::open(&file.0).expect("valid GGUF");
    let metadata = gguf.metadata();
    assert_eq!(metadata.architecture().expect("architecture"), "qwen35");
    assert_eq!(metadata.unsigned("qwen35.block_count"), Some(24));
    assert_eq!(metadata.float("qwen35.rope.freq_base"), Some(1.0e7));
    assert_eq!(
        metadata.base_models()[0].repo_url.as_deref(),
        Some("https://huggingface.co/Qwen/Qwen3.5-0.8B")
    );
    let tokenizer = metadata.tokenizer().expect("tokenizer");
    assert_eq!(tokenizer.model, "gpt2");
    assert_eq!(tokenizer.pre.as_deref(), Some("qwen35"));
    assert_eq!(tokenizer.tokens, ["a", "b", "ab"]);
    assert_eq!(tokenizer.merges, ["a b"]);
    assert_eq!((tokenizer.bos, tokenizer.eos), (None, Some(2)));
    assert_eq!(tokenizer.add_bos, Some(false));
    assert_eq!(tokenizer.chat_template.as_deref(), Some("{{ messages }}"));

    assert_eq!(gguf.names(), ["matrix", "row"]);
    let matrix = gguf.tensor("matrix").expect("matrix");
    assert_eq!(matrix.shape(), [2, 3], "shape is outermost first");
    assert_eq!(matrix.encoding(), GgufEncoding::F32);
    let bytes = gguf.read("matrix", 1024).expect("read");
    let values: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|word| f32::from_le_bytes(word.try_into().expect("4")))
        .collect();
    assert_eq!(values, [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
    let row = gguf.tensor("row").expect("row");
    assert_eq!(row.encoding(), GgufEncoding::Q8_0);
    assert_eq!(row.range().end - row.range().start, 34);
    assert!(matches!(
        gguf.read("row", 33),
        Err(CheckpointError::TooLarge { bytes: 34, .. })
    ));
    assert!(matches!(
        gguf.read("absent", 1024),
        Err(CheckpointError::MissingTensor(_))
    ));
}

#[test]
fn detects_a_file_replaced_after_inspection() {
    let file = valid().write();
    let gguf = GgufFile::open(&file.0).expect("valid GGUF");
    let mut longer = valid().bytes();
    longer.push(0);
    fs::write(&file.0, longer).expect("rewrite");
    assert!(matches!(
        gguf.read("matrix", 1024),
        Err(CheckpointError::Changed(_))
    ));
}

#[test]
fn honours_a_custom_alignment() {
    let mut builder = valid().key("general.alignment", 4, 64_u32.to_le_bytes().to_vec());
    builder.alignment = 64;
    builder.tensors[1].3 = 64;
    let mut payload = builder.payload.clone();
    payload.splice(32..32, vec![0; 32]);
    builder.payload = payload;
    let file = builder.write();
    let gguf = GgufFile::open(&file.0).expect("aligned GGUF");
    assert_eq!(gguf.read("row", 64).expect("row")[2], 0);

    let mut misaligned = builder.clone();
    misaligned.tensors[1].3 = 32;
    assert!(matches!(open(&misaligned), Err(CheckpointError::Range(_))));
    let odd = valid().key("general.alignment", 4, 24_u32.to_le_bytes().to_vec());
    assert!(matches!(open(&odd), Err(CheckpointError::Alignment(24))));
}

#[test]
fn rejects_malformed_headers() {
    let mut bytes = valid().bytes();
    bytes[0] = b'X';
    let file = TempFile::new(&bytes);
    assert!(matches!(
        GgufFile::open(&file.0),
        Err(CheckpointError::NotGguf(_))
    ));

    let mut old = valid();
    old.version = 2;
    assert!(matches!(open(&old), Err(CheckpointError::Version(2))));

    let truncated = TempFile::new(&valid().bytes()[..60]);
    assert!(matches!(
        GgufFile::open(&truncated.0),
        Err(CheckpointError::Truncated(_))
    ));

    let huge_string = valid().key("bad", 8, u64::MAX.to_le_bytes().to_vec());
    assert!(matches!(
        open(&huge_string),
        Err(CheckpointError::Limit {
            what: "string length",
            ..
        })
    ));
    let mut nested = 9_u32.to_le_bytes().to_vec();
    nested.extend(1_u64.to_le_bytes());
    assert!(matches!(
        open(&valid().key("nested", 9, nested)),
        Err(CheckpointError::NestedArray(_))
    ));
    assert!(matches!(
        open(&valid().key("bad", 13, Vec::new())),
        Err(CheckpointError::ValueType { id: 13, .. })
    ));
    assert!(matches!(
        open(&valid().key("general.architecture", 8, string("again"))),
        Err(CheckpointError::DuplicateKey(_))
    ));
}

#[test]
fn rejects_invalid_tensor_records() {
    let reject = |edit: &dyn Fn(&mut Builder)| {
        let mut builder = valid();
        edit(&mut builder);
        open(&builder).expect_err("invalid tensor record")
    };
    assert!(matches!(
        reject(&|b| b.tensors[0].2 = 4),
        CheckpointError::Encoding { id: 4, .. }
    ));
    assert!(matches!(
        reject(&|b| b.tensors[1].1 = vec![16, 2]),
        CheckpointError::Shape { .. }
    ));
    assert!(matches!(
        reject(&|b| b.tensors[0].1 = vec![3, 0]),
        CheckpointError::Shape { .. }
    ));
    assert!(matches!(
        reject(&|b| b.tensors[0].1 = vec![1, 1, 1, 1, 1]),
        CheckpointError::Limit { .. }
    ));
    assert!(matches!(
        reject(&|b| b.tensors[1].3 = 64),
        CheckpointError::Range(_)
    ));
    assert!(matches!(
        reject(&|b| b.tensors[1].3 = 0),
        CheckpointError::Overlap(..)
    ));
    assert!(matches!(
        reject(&|b| b.tensors[1].0 = "matrix".into()),
        CheckpointError::DuplicateTensor(_) | CheckpointError::Overlap(..)
    ));
}

#[test]
fn array_integers_widen_every_integer_type() {
    assert_eq!(GgufArray::I32(vec![-1, 2]).integers(), Some(vec![-1, 2]));
    assert_eq!(GgufArray::U64(vec![u64::MAX]).integers(), None);
    assert_eq!(GgufArray::String(Vec::new()).integers(), None);
    assert!(matches!(GgufValue::Bool(true), GgufValue::Bool(true)));
}
