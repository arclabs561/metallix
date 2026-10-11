use std::{
    fs,
    path::{Path, PathBuf},
    sync::atomic::{AtomicUsize, Ordering},
};

use serde_json::{Value, json};

use super::{SafetensorsCheckpoint, SafetensorsFile};
use crate::{CheckpointError, DType, ReadCache, TensorEncoding, TensorSource};

/// A temporary directory removed on drop.
struct TempDir(PathBuf);

impl TempDir {
    fn new() -> Self {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let path = std::env::temp_dir().join(format!(
            "checkpoint-safetensors-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir_all(&path).expect("temp dir");
        Self(path)
    }

    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

/// `(name, dtype, shape, payload)` laid out back to back after the header.
fn file_bytes(tensors: &[(&str, &str, &[u64], Vec<u8>)]) -> Vec<u8> {
    let mut header = serde_json::Map::new();
    header.insert("__metadata__".into(), json!({"format": "mlx"}));
    let mut offset = 0;
    let mut payload = Vec::new();
    for (name, dtype, shape, bytes) in tensors {
        header.insert(
            (*name).to_owned(),
            json!({"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + bytes.len()]}),
        );
        offset += bytes.len();
        payload.extend_from_slice(bytes);
    }
    header_then(&Value::Object(header), &payload)
}

fn header_then(header: &Value, payload: &[u8]) -> Vec<u8> {
    let json = serde_json::to_vec(header).expect("json");
    let mut out = (json.len() as u64).to_le_bytes().to_vec();
    out.extend(json);
    out.extend_from_slice(payload);
    out
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

#[test]
fn reads_the_table_and_payloads_of_one_file() {
    let dir = TempDir::new();
    let path = dir.path().join("model.safetensors");
    fs::write(
        &path,
        file_bytes(&[
            ("b", "U32", &[2], vec![1, 0, 0, 0, 2, 0, 0, 0]),
            (
                "a",
                "F32",
                &[2, 3],
                f32_bytes(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]),
            ),
        ]),
    )
    .expect("write");
    for cache in [ReadCache::Use, ReadCache::Bypass] {
        let file = SafetensorsFile::open(&path, cache).expect("open");
        assert_eq!(file.names(), ["b", "a"], "names in payload order");
        let a = file.tensor("a").expect("a");
        assert_eq!(a.encoding(), TensorEncoding::Plain(DType::F32));
        assert_eq!(a.shape(), [2, 3]);
        assert_eq!(a.range().end - a.range().start, 24);
        assert_eq!(
            file.read("a", 24).expect("read"),
            f32_bytes(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
        );
        assert!(matches!(
            file.read("a", 23),
            Err(CheckpointError::TooLarge { bytes: 24, .. })
        ));
        assert!(matches!(
            file.read("c", 1024),
            Err(CheckpointError::MissingTensor(_))
        ));
    }
}

#[test]
fn refuses_malformed_files() {
    let dir = TempDir::new();
    let open = |bytes: &[u8]| {
        let path = dir.path().join("x.safetensors");
        fs::write(&path, bytes).expect("write");
        SafetensorsFile::open(&path, ReadCache::Use)
    };
    // Header longer than the file.
    let mut short = 1000_u64.to_le_bytes().to_vec();
    short.extend(b"{}");
    assert!(matches!(
        open(&short),
        Err(CheckpointError::NotSafetensors(_))
    ));
    // Not a JSON object.
    assert!(matches!(
        open(&header_then(&json!([1, 2]), &[])),
        Err(CheckpointError::NotSafetensors(_))
    ));
    // Unknown dtype.
    assert!(matches!(
        open(&file_bytes(&[("x", "F8_E5M2", &[1], vec![0])])),
        Err(CheckpointError::UnknownDType { tensor, dtype })
            if tensor == "x" && dtype == "F8_E5M2"
    ));
    // Offsets that disagree with the shape.
    assert!(matches!(
        open(&header_then(
            &json!({"x": {"dtype": "F32", "shape": [2], "data_offsets": [0, 4]}}),
            &[0; 8]
        )),
        Err(CheckpointError::Range(_))
    ));
    // Past the end of the file.
    assert!(matches!(
        open(&header_then(
            &json!({"x": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]}}),
            &[0; 4]
        )),
        Err(CheckpointError::Range(_))
    ));
    // Two tensors sharing bytes.
    assert!(matches!(
        open(&header_then(
            &json!({
                "x": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]},
                "y": {"dtype": "F32", "shape": [1], "data_offsets": [4, 8]},
            }),
            &[0; 8]
        )),
        Err(CheckpointError::Overlap(..))
    ));
}

/// A two-shard checkpoint holding one 4-bit module `m` (`[4, 64]` logical,
/// group 64) and a plain `norm`, with `config` as its `config.json`.
fn sharded(dir: &Path, config: &Value, scale_columns: u64) {
    let scales = vec![0; 4 * usize::try_from(scale_columns).expect("columns") * 2];
    fs::write(
        dir.join("model-00001-of-00002.safetensors"),
        file_bytes(&[
            ("m.weight", "U32", &[4, 8], vec![7; 4 * 8 * 4]),
            ("m.scales", "BF16", &[4, scale_columns], scales.clone()),
        ]),
    )
    .expect("shard 1");
    fs::write(
        dir.join("model-00002-of-00002.safetensors"),
        file_bytes(&[
            ("m.biases", "BF16", &[4, scale_columns], scales),
            ("norm.weight", "F32", &[2], f32_bytes(&[1.0, 2.0])),
        ]),
    )
    .expect("shard 2");
    fs::write(
        dir.join("model.safetensors.index.json"),
        serde_json::to_vec(&json!({"weight_map": {
            "m.weight": "model-00001-of-00002.safetensors",
            "m.scales": "model-00001-of-00002.safetensors",
            "m.biases": "model-00002-of-00002.safetensors",
            "norm.weight": "model-00002-of-00002.safetensors",
        }}))
        .expect("index"),
    )
    .expect("index");
    fs::write(
        dir.join("config.json"),
        serde_json::to_vec(config).expect("config"),
    )
    .expect("config");
}

#[test]
fn a_sharded_checkpoint_marks_affine_codes_from_the_configuration() {
    let dir = TempDir::new();
    sharded(
        dir.path(),
        &json!({"quantization": {"group_size": 64, "bits": 4, "mode": "affine"}}),
        1,
    );
    let checkpoint = SafetensorsCheckpoint::open(dir.path(), ReadCache::Use).expect("open");
    assert_eq!(
        checkpoint.names(),
        ["m.weight", "m.scales", "m.biases", "norm.weight"]
    );
    let codes = checkpoint.tensor("m.weight").expect("codes");
    assert_eq!(
        codes.encoding(),
        TensorEncoding::Affine {
            bits: 4,
            group_size: 64
        }
    );
    assert_eq!(codes.shape(), [4, 8], "the stored packed shape");
    assert_eq!(
        checkpoint.tensor("m.scales").expect("scales").encoding(),
        TensorEncoding::Plain(DType::BF16)
    );
    assert_eq!(
        checkpoint.read("norm.weight", 8).expect("read"),
        f32_bytes(&[1.0, 2.0])
    );
    assert_eq!(
        checkpoint.read("m.weight", 1 << 10).expect("read"),
        vec![7; 128]
    );
}

#[test]
fn quantized_modules_must_match_the_configuration() {
    let affine = json!({"quantization": {"group_size": 64, "bits": 4}});
    // Scales stored, configuration silent or leaving the module unquantized.
    for config in [
        json!({}),
        json!({"quantization": {"group_size": 64, "bits": 4, "m": false}}),
    ] {
        let dir = TempDir::new();
        sharded(dir.path(), &config, 1);
        assert!(matches!(
            SafetensorsCheckpoint::open(dir.path(), ReadCache::Use),
            Err(CheckpointError::UndeclaredQuantization(stem)) if stem == "m"
        ));
    }
    // Integer scales are not affine parameters.
    let dir = TempDir::new();
    sharded(dir.path(), &affine, 1);
    let shard = dir.path().join("model-00001-of-00002.safetensors");
    fs::write(
        &shard,
        file_bytes(&[
            ("m.weight", "U32", &[4, 8], vec![7; 4 * 8 * 4]),
            ("m.scales", "I16", &[4, 1], vec![0; 8]),
        ]),
    )
    .expect("shard 1");
    assert!(matches!(
        SafetensorsCheckpoint::open(dir.path(), ReadCache::Use),
        Err(CheckpointError::AffineParameterDtype { tensor, .. }) if tensor == "m.scales"
    ));
    // Two groups of scales for a 64-column row at group size 64.
    let dir = TempDir::new();
    sharded(dir.path(), &affine, 2);
    assert!(matches!(
        SafetensorsCheckpoint::open(dir.path(), ReadCache::Use),
        Err(CheckpointError::AffineShape { module, expected, codes })
            if module == "m" && expected == [4, 16] && codes == [4, 8]
    ));
    // The same codes read as 8-bit would be 32 columns, half a group.
    let dir = TempDir::new();
    sharded(
        dir.path(),
        &json!({"quantization": {"group_size": 64, "bits": 8}}),
        1,
    );
    assert!(matches!(
        SafetensorsCheckpoint::open(dir.path(), ReadCache::Use),
        Err(CheckpointError::AffineShape { expected, .. }) if expected == [4, 16]
    ));
    // Another format's quantization is refused before any shard is read.
    let dir = TempDir::new();
    sharded(
        dir.path(),
        &json!({"quantization_config": {"quant_method": "gptq", "bits": 4}}),
        1,
    );
    assert!(matches!(
        SafetensorsCheckpoint::open(dir.path(), ReadCache::Use),
        Err(CheckpointError::UnsupportedQuantization(_))
    ));
}

#[test]
fn the_index_must_agree_with_its_shards() {
    let affine = json!({"quantization": {"group_size": 64, "bits": 4}});
    let dir = TempDir::new();
    sharded(dir.path(), &affine, 1);
    let index = dir.path().join("model.safetensors.index.json");
    let mut map: Value = serde_json::from_slice(&fs::read(&index).expect("read")).expect("json");
    map["weight_map"]["norm.weight"] = json!("model-00001-of-00002.safetensors");
    fs::write(&index, serde_json::to_vec(&map).expect("json")).expect("write");
    assert!(matches!(
        SafetensorsCheckpoint::open(dir.path(), ReadCache::Use),
        Err(CheckpointError::MissingTensor(name)) if name == "norm.weight"
    ));
    map["weight_map"]["norm.weight"] = json!("../model-00002-of-00002.safetensors");
    fs::write(&index, serde_json::to_vec(&map).expect("json")).expect("write");
    assert!(matches!(
        SafetensorsCheckpoint::open(dir.path(), ReadCache::Use),
        Err(CheckpointError::UnsafeShard(_))
    ));
}

#[test]
fn reads_ranges_and_outer_slices() {
    let dir = TempDir::new();
    let path = dir.path().join("model.safetensors");
    let values: Vec<f32> = (0..6_u8).map(f32::from).collect();
    fs::write(
        &path,
        file_bytes(&[("a", "F32", &[3, 2], f32_bytes(&values))]),
    )
    .expect("write");
    let file = SafetensorsFile::open(&path, ReadCache::Bypass).expect("open");
    let a = file.tensor("a").expect("a").clone();
    let row = a.slice_dim0(1).expect("row 1");
    assert_eq!(row.end - row.start, 8);
    let mut dst = [0; 8];
    file.read_range("a", row.clone(), &mut dst)
        .expect("read row");
    assert_eq!(dst.to_vec(), f32_bytes(&[2.0, 3.0]));
    assert_eq!(a.slice_dim0(3), None, "past the outer dimension");
    // A buffer of the wrong length, and a range past the payload.
    assert!(matches!(
        file.read_range("a", row.clone(), &mut [0; 7]),
        Err(CheckpointError::OutsideTensor { len: 7, .. })
    ));
    let past = a.range().end - 4..a.range().end + 4;
    assert!(matches!(
        file.read_range("a", past, &mut dst),
        Err(CheckpointError::OutsideTensor { .. })
    ));
    assert!(matches!(
        file.read_range("b", row, &mut dst),
        Err(CheckpointError::MissingTensor(_))
    ));
}

#[test]
fn a_sharded_checkpoint_reads_ranges_from_the_right_shard() {
    let dir = TempDir::new();
    sharded(
        dir.path(),
        &json!({"quantization": {"group_size": 64, "bits": 4}}),
        1,
    );
    let checkpoint = SafetensorsCheckpoint::open(dir.path(), ReadCache::Bypass).expect("open");
    let norm = checkpoint.tensor("norm.weight").expect("norm").clone();
    let mut dst = [0; 4];
    checkpoint
        .read_range("norm.weight", norm.slice_dim0(1).expect("second"), &mut dst)
        .expect("read");
    assert_eq!(dst, 2.0_f32.to_le_bytes());
    let codes = checkpoint.tensor("m.weight").expect("codes").clone();
    let mut row = [0; 32];
    checkpoint
        .read_range("m.weight", codes.slice_dim0(3).expect("row 3"), &mut row)
        .expect("read");
    assert_eq!(row, [7; 32]);
}

#[test]
fn a_file_truncated_after_inspection_fails_the_read() {
    let dir = TempDir::new();
    let path = dir.path().join("model.safetensors");
    let values: Vec<f32> = (0..64_u8).map(f32::from).collect();
    fs::write(
        &path,
        file_bytes(&[("a", "F32", &[8, 8], f32_bytes(&values))]),
    )
    .expect("write");
    let file = SafetensorsFile::open(&path, ReadCache::Bypass).expect("open");
    let a = file.tensor("a").expect("a").clone();
    // Cut the file inside the last row: that row's read comes up short.
    fs::OpenOptions::new()
        .write(true)
        .open(&path)
        .expect("reopen")
        .set_len(a.range().end - 4)
        .expect("truncate");
    let mut row = [0; 32];
    file.read_range("a", a.slice_dim0(0).expect("row 0"), &mut row)
        .expect("an intact row still reads");
    assert!(matches!(
        file.read_range("a", a.slice_dim0(7).expect("row 7"), &mut row),
        Err(CheckpointError::Io { source, .. }) if source.kind() == std::io::ErrorKind::UnexpectedEof
    ));
}

#[test]
fn eight_bit_float_dtypes_are_described_and_other_ones_refused() {
    let dir = TempDir::new();
    let path = dir.path().join("model.safetensors");
    fs::write(
        &path,
        file_bytes(&[
            ("w", "F8_E4M3", &[2, 2], vec![0x38, 0x40, 0xb8, 0xc0]),
            ("s", "F8_E8M0", &[2], vec![127, 128]),
        ]),
    )
    .expect("write");
    let file = SafetensorsFile::open(&path, ReadCache::Use).expect("open");
    let w = file.tensor("w").expect("w");
    assert_eq!(w.encoding(), TensorEncoding::Plain(DType::F8E4M3));
    assert_eq!(w.range().end - w.range().start, 4);
    let s = file.tensor("s").expect("s");
    assert_eq!(s.encoding(), TensorEncoding::Plain(DType::F8E8M0));
    assert_eq!(file.read("s", 2).expect("read"), [127, 128]);

    fs::write(&path, file_bytes(&[("w", "F8_E5M2", &[1], vec![0])])).expect("write");
    assert!(matches!(
        SafetensorsFile::open(&path, ReadCache::Use),
        Err(CheckpointError::UnknownDType { dtype, .. }) if dtype == "F8_E5M2"
    ));
}
