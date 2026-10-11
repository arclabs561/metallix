//! Black-box refusals before checkpoint payload reads; no model fixture required.

use serde_json::json;
use sha2::{Digest, Sha256};
use std::{
    fs,
    path::PathBuf,
    process::{Command, Output},
    sync::atomic::{AtomicU64, Ordering},
};

const REVISION: &str = "dba1be0a40aa45a94ad051997016db3960a90277";
static SEQUENCE: AtomicU64 = AtomicU64::new(0);

struct Fixture {
    dir: PathBuf,
    capture_sha: String,
}
impl Fixture {
    fn new() -> Self {
        let mut selected = None;
        for _ in 0..64 {
            let dir = std::env::temp_dir().join(format!(
                "metallix-selected-cli-{}-{}",
                std::process::id(),
                SEQUENCE.fetch_add(1, Ordering::Relaxed)
            ));
            match fs::create_dir(&dir) {
                Ok(()) => {
                    selected = Some(dir);
                    break;
                }
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {}
                Err(error) => panic!("create fixture: {error}"),
            }
        }
        let dir = selected.expect("unique temporary directory");
        let bytes = vec![0_u8; 5120 * 2];
        fs::write(dir.join("input.bin"), &bytes).expect("write input");
        fs::write(dir.join("expected.bin"), &bytes).expect("write expected");
        fs::write(dir.join("routes.json"), serde_json::to_vec(&json!({"revision":REVISION,"runs":[{"routes":[{"layer":0,"ids":[[0,1,2,3,4,5]]}]}]})).expect("serialize")).expect("write routes");
        Self {
            dir,
            capture_sha: format!("{:x}", Sha256::digest(&bytes)),
        }
    }
    fn command(&self) -> Command {
        let mut command = Command::new(env!("CARGO_BIN_EXE_mx"));
        command
            .arg("run-deepseek-selected")
            .arg("--index")
            .arg(self.dir.join("missing-index.json"))
            .arg("--headers-dir")
            .arg(&self.dir)
            .arg("--weights-dir")
            .arg(self.dir.join("missing-weights"))
            .args([
                "--revision",
                REVISION,
                "--tokens",
                "1",
                "--input-sha256",
                &self.capture_sha,
                "--expected-sha256",
                &self.capture_sha,
            ])
            .arg("--input-bf16")
            .arg(self.dir.join("input.bin"))
            .arg("--expected-bf16")
            .arg(self.dir.join("expected.bin"))
            .arg("--routes")
            .arg(self.dir.join("routes.json"));
        command
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.dir);
    }
}
fn assert_refused(output: &Output, reason: &str) {
    assert_eq!(output.status.code(), Some(1));
    assert!(
        output.stdout.is_empty(),
        "refusal must not manufacture a comparison receipt"
    );
    assert!(
        String::from_utf8_lossy(&output.stderr).contains(reason),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn mutated_expected_capture_is_rejected_before_missing_checkpoint_metadata() {
    let fixture = Fixture::new();
    let mut changed = vec![0_u8; 5120 * 2];
    changed[0] = 1;
    fs::write(fixture.dir.join("expected.bin"), changed).expect("mutate oracle");
    let output = fixture.command().output().expect("mx launches");
    assert_refused(&output, "input/expected SHA256 mismatch");
}

#[test]
fn captures_over_the_row_bound_are_refused_before_reading_routes() {
    let fixture = Fixture::new();
    fs::File::create(fixture.dir.join("input.bin"))
        .expect("input")
        .set_len(5120 * 2 + 1)
        .expect("sparse size");
    let output = fixture.command().output().expect("mx launches");
    assert_refused(&output, "byte bound");
}

#[test]
fn duplicate_or_absent_source_routes_cannot_be_a_successful_comparison() {
    let fixture = Fixture::new();
    for routes in [json!([]), json!([{"layer":0,"ids":[[0,0,2,3,4,5]]}])] {
        fs::write(
            fixture.dir.join("routes.json"),
            serde_json::to_vec(&json!({"revision":REVISION,"runs":[{"routes":routes}]}))
                .expect("serialize"),
        )
        .expect("routes");
        let output = fixture.command().output().expect("mx launches");
        assert_refused(&output, "route");
    }
}

#[test]
fn aggregate_header_budget_is_refused_before_index_or_payload_io() {
    let fixture = Fixture::new();
    fs::write(fixture.dir.join("headers.json"), serde_json::to_vec(&json!({"revision":REVISION,"index_sha256":"unused","shards":{"model.safetensors":{"header_bytes":33*1024*1024,"header_sha256":"unused","file_bytes":33*1024*1024}}})).expect("serialize")).expect("manifest");
    let output = fixture.command().output().expect("mx launches");
    assert_refused(&output, "aggregate headers exceed 32 MiB");
}

#[test]
fn explicit_zero_budget_is_rejected_before_any_checkpoint_read() {
    let fixture = Fixture::new();
    let output = fixture
        .command()
        .args(["--payload-budget-mib", "0"])
        .output()
        .expect("mx launches");
    assert_refused(&output, "budgets must be in 1..=512 MiB");
}

#[test]
fn missing_expected_output_cannot_be_reported_as_zero_error() {
    let fixture = Fixture::new();
    fs::rename(
        fixture.dir.join("expected.bin"),
        fixture.dir.join("unused.bin"),
    )
    .expect("hide expected output");
    let output = fixture.command().output().expect("mx launches");
    assert_refused(&output, "deepseek selected refused");
}

#[test]
fn norm_boundary_requires_independent_intermediate_before_metadata() {
    let fixture = Fixture::new();
    let output = fixture
        .command()
        .args(["--boundary", "ffn-norm-moe"])
        .output()
        .expect("mx launches");
    assert_refused(&output, "requires intermediate expected path/hash");
}

#[test]
fn norm_boundary_refuses_wrong_published_metadata_before_moe_reads() {
    let fixture = Fixture::new();
    fs::create_dir(fixture.dir.join("headers")).expect("headers");
    let name = "layers.0.ffn_norm.weight";
    let shard = "model.safetensors";
    for (dtype, shape, size) in [
        ("F32", json!([5120]), 20480_u64),
        ("BF16", json!([2560, 2]), 10240),
    ] {
        let index =
            serde_json::to_vec(&json!({"metadata":{"total_size":size}, "weight_map":{name:shard}}))
                .unwrap();
        fs::write(fixture.dir.join("missing-index.json"), &index).unwrap();
        let mut header = serde_json::to_vec(
            &json!({name:{"dtype":dtype,"shape":shape,"data_offsets":[0,size]}}),
        )
        .unwrap();
        while !header.len().is_multiple_of(8) {
            header.push(b' ');
        }
        let mut prefix = (header.len() as u64).to_le_bytes().to_vec();
        prefix.extend_from_slice(&header);
        fs::write(
            fixture.dir.join("headers/model.safetensors.header.bin"),
            &prefix,
        )
        .unwrap();
        fs::write(fixture.dir.join("headers.json"), serde_json::to_vec(&json!({
            "revision":REVISION, "index_sha256":format!("{:x}", Sha256::digest(&index)),
            "shards":{shard:{"header_bytes":prefix.len(), "header_sha256":format!("{:x}", Sha256::digest(&prefix)),"file_bytes":prefix.len() as u64+size}}
        })).unwrap()).unwrap();
        let output = fixture
            .command()
            .args([
                "--boundary",
                "ffn-norm-moe",
                "--expected-normalized-sha256",
                &fixture.capture_sha,
            ])
            .arg("--expected-normalized-bf16")
            .arg(fixture.dir.join("expected.bin"))
            .output()
            .unwrap();
        assert_refused(&output, "norm tensor geometry mismatch");
    }
}

#[test]
fn changed_normalized_bits_with_valid_digest_refuse_before_absent_moe_files() {
    let fixture = Fixture::new();
    let name = "layers.0.ffn_norm.weight";
    let shard = "model.safetensors";
    let weights = fixture.dir.join("missing-weights");
    fs::create_dir(&weights).unwrap();
    fs::create_dir(fixture.dir.join("headers")).unwrap();
    let input: Vec<u8> = vec![0x3f80_u16; 5120]
        .into_iter()
        .flat_map(u16::to_le_bytes)
        .collect();
    let weight: Vec<u8> = vec![0x4000_u16; 5120]
        .into_iter()
        .flat_map(u16::to_le_bytes)
        .collect();
    fs::write(fixture.dir.join("input.bin"), &input).unwrap();
    fs::write(weights.join(format!("{name}.bin")), &weight).unwrap();
    let index =
        serde_json::to_vec(&json!({"metadata":{"total_size":10240},"weight_map":{name:shard}}))
            .unwrap();
    fs::write(fixture.dir.join("missing-index.json"), &index).unwrap();
    let mut header =
        serde_json::to_vec(&json!({name:{"dtype":"BF16","shape":[5120],"data_offsets":[0,10240]}}))
            .unwrap();
    while !header.len().is_multiple_of(8) {
        header.push(b' ');
    }
    let mut prefix = (header.len() as u64).to_le_bytes().to_vec();
    prefix.extend_from_slice(&header);
    fs::write(
        fixture.dir.join("headers/model.safetensors.header.bin"),
        &prefix,
    )
    .unwrap();
    fs::write(fixture.dir.join("headers.json"), serde_json::to_vec(&json!({"revision":REVISION,"index_sha256":format!("{:x}",Sha256::digest(&index)),"shards":{shard:{"header_bytes":prefix.len(),"header_sha256":format!("{:x}",Sha256::digest(&prefix)),"file_bytes":prefix.len()+10240}}})).unwrap()).unwrap();
    fs::write(weights.join(format!("{name}.receipt.json")),serde_json::to_vec(&json!({"tensor":name,"shard":shard,"revision":REVISION,"bytes":10240,"range":[prefix.len(),prefix.len()+10240],"metadata":{"dtype":"BF16","shape":[5120]},"sha256":format!("{:x}",Sha256::digest(&weight))})).unwrap()).unwrap();
    let mut changed = weight.clone();
    changed[0] ^= 1;
    fs::write(fixture.dir.join("normalized.bin"), &changed).unwrap();
    // Replace the base command's input hash instead of passing the flag twice.
    let command = fixture.command();
    let args = command
        .get_args()
        .map(std::ffi::OsStr::to_os_string)
        .collect::<Vec<_>>();
    let mut command = Command::new(env!("CARGO_BIN_EXE_mx"));
    let mut args = args;
    let position = args.iter().position(|arg| arg == "--input-sha256").unwrap();
    args[position + 1] = format!("{:x}", Sha256::digest(&input)).into();
    let output = command
        .args(args)
        .args([
            "--boundary",
            "ffn-norm-moe",
            "--expected-normalized-sha256",
            &format!("{:x}", Sha256::digest(&changed)),
        ])
        .arg("--expected-normalized-bf16")
        .arg(fixture.dir.join("normalized.bin"))
        .output()
        .unwrap();
    assert_refused(
        &output,
        "normalized BF16 comparison failed before MoE payload reads",
    );
}

// Fixed source replay contract, with analytic all-zero captures for protocol
// refusals only. These fixtures do not qualify checkpoint arithmetic.
fn tail_fixture(fixture: &Fixture) -> Command {
    let mut outputs = serde_json::Map::new();
    let mut residual_sha = String::new();
    for (role, dtype, shape) in [
        ("after_attention", "bfloat16", vec![1, 1, 4, 5120]),
        ("attention_pre", "float32", vec![1, 1, 4]),
        ("ffn_norm_in", "bfloat16", vec![1, 1, 5120]),
        ("ffn_in", "bfloat16", vec![1, 1, 5120]),
        ("ffn_out", "bfloat16", vec![1, 1, 5120]),
        ("out", "bfloat16", vec![1, 1, 4, 5120]),
        ("ffn_pre", "float32", vec![1, 1, 4]),
        ("ffn_post", "float32", vec![1, 1, 4]),
        ("ffn_comb", "float32", vec![1, 1, 4, 4]),
    ] {
        let bytes =
            vec![0_u8; shape.iter().product::<usize>() * if dtype == "float32" { 4 } else { 2 }];
        let sha = format!("{:x}", Sha256::digest(&bytes));
        let name = format!("layer00.{role}.torch.{dtype}.bin");
        fs::write(fixture.dir.join(&name), &bytes).unwrap();
        if role == "after_attention" {
            fs::write(fixture.dir.join("input.bin"), &bytes).unwrap();
            fs::write(fixture.dir.join("expected.bin"), &bytes).unwrap();
            residual_sha.clone_from(&sha);
        }
        outputs.insert(name, json!({"dtype":format!("torch.{dtype}"),"shape":shape,"bytes":bytes.len(),"sha256":sha}));
    }
    fs::write(
        fixture.dir.join("receipt.json"),
        serde_json::to_vec(&json!({
            "schema_version":1,"revision":REVISION,"tokens":1,"passed_source_joins":true,
            "norm_eps":1e-20,"hc_eps":1e-6,"hc_sinkhorn_iters":20,"outputs":outputs
        }))
        .unwrap(),
    )
    .unwrap();
    let base = fixture.command();
    let mut args = base
        .get_args()
        .map(std::ffi::OsStr::to_os_string)
        .collect::<Vec<_>>();
    for flag in ["--input-sha256", "--expected-sha256"] {
        let position = args.iter().position(|arg| arg == flag).unwrap();
        args[position + 1] = residual_sha.clone().into();
    }
    let mut command = Command::new(env!("CARGO_BIN_EXE_mx"));
    command
        .args(args)
        .args(["--boundary", "ffn-tail", "--tail-capture-dir"])
        .arg(&fixture.dir);
    command
}

fn change_tail_receipt(fixture: &Fixture, change: impl FnOnce(&mut serde_json::Value)) {
    let path = fixture.dir.join("receipt.json");
    let mut value: serde_json::Value = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
    change(&mut value);
    fs::write(path, serde_json::to_vec(&value).unwrap()).unwrap();
}

#[test]
fn full_tail_requires_explicit_capture_contract() {
    let fixture = Fixture::new();
    let output = fixture
        .command()
        .args(["--boundary", "ffn-tail"])
        .output()
        .unwrap();
    assert_refused(&output, "requires --tail-capture-dir");
    let output = fixture
        .command()
        .arg("--tail-capture-dir")
        .arg(&fixture.dir)
        .output()
        .unwrap();
    assert_refused(&output, "other boundaries forbid it");
}

#[test]
fn full_tail_refuses_wrong_copies_and_incoming_pre_width_before_metadata() {
    let fixture = Fixture::new();
    for (name, shape) in [
        (
            "layer00.after_attention.torch.bfloat16.bin",
            json!([1, 1, 2, 5120]),
        ),
        ("layer00.attention_pre.torch.float32.bin", json!([1, 1, 3])),
    ] {
        let mut command = tail_fixture(&fixture);
        change_tail_receipt(&fixture, |receipt| {
            receipt["outputs"][name]["shape"] = shape;
        });
        assert_refused(
            &command.output().unwrap(),
            "capture geometry/hash syntax mismatch",
        );
    }
    let mut command = tail_fixture(&fixture);
    fs::write(fixture.dir.join("input.bin"), vec![0_u8; 5120 * 2]).unwrap();
    assert_refused(
        &command.output().unwrap(),
        "capture length/SHA256 mismatch: after_attention",
    );
}

#[test]
fn full_tail_refuses_nonfinite_incoming_pre_even_with_updated_digest() {
    let fixture = Fixture::new();
    let mut command = tail_fixture(&fixture);
    let name = "layer00.attention_pre.torch.float32.bin";
    let mut bytes = vec![0_u8; 16];
    bytes[..4].copy_from_slice(&f32::NAN.to_le_bytes());
    fs::write(fixture.dir.join(name), &bytes).unwrap();
    change_tail_receipt(&fixture, |receipt| {
        receipt["outputs"][name]["sha256"] = json!(format!("{:x}", Sha256::digest(&bytes)));
    });
    assert_refused(&command.output().unwrap(), "F32 capture contains nonfinite");
}

#[test]
fn full_tail_refuses_source_control_changes_and_unknown_capture_names() {
    let fixture = Fixture::new();
    for field in ["norm_eps", "hc_eps", "hc_sinkhorn_iters", "revision"] {
        let mut command = tail_fixture(&fixture);
        change_tail_receipt(&fixture, |receipt| {
            receipt[field] = if field == "revision" {
                json!("other")
            } else {
                json!(0)
            }
        });
        assert_refused(
            &command.output().unwrap(),
            "receipt controls/revision/count mismatch",
        );
    }
    let mut command = tail_fixture(&fixture);
    change_tail_receipt(&fixture, |receipt| {
        let record = receipt["outputs"]
            .as_object_mut()
            .unwrap()
            .remove("layer00.attention_pre.torch.float32.bin")
            .unwrap();
        receipt["outputs"]["../outside.bin"] = record;
    });
    assert_refused(&command.output().unwrap(), "lacks fixed capture name");
}

#[test]
fn full_tail_changed_final_hash_refuses_before_checkpoint_metadata() {
    let fixture = Fixture::new();
    let mut command = tail_fixture(&fixture);
    let mut bytes = fs::read(fixture.dir.join("expected.bin")).unwrap();
    bytes[0] ^= 1;
    fs::write(fixture.dir.join("expected.bin"), bytes).unwrap();
    assert_refused(
        &command.output().unwrap(),
        "capture length/SHA256 mismatch: out",
    );
}

type TailTensorSpec = (String, &'static str, Vec<usize>, u64);

fn tail_tensor_specs(projection_shape: &[usize]) -> Vec<TailTensorSpec> {
    // Published tensor geometry. Only norm/HC payloads exist: a successful
    // source-bound refusal must happen before the absent MoE files are touched.
    let mut specs = vec![
        (
            "layers.0.ffn_norm.weight".to_owned(),
            "BF16",
            vec![5120],
            10240_u64,
        ),
        (
            "layers.0.hc_ffn_fn".to_owned(),
            "F32",
            projection_shape.to_vec(),
            1_966_080,
        ),
        ("layers.0.hc_ffn_scale".to_owned(), "F32", vec![3], 12),
        ("layers.0.hc_ffn_base".to_owned(), "F32", vec![24], 96),
        (
            "layers.0.ffn.gate.weight".to_owned(),
            "BF16",
            vec![384, 5120],
            3_932_160,
        ),
        ("layers.0.ffn.gate.bias".to_owned(), "F32", vec![384], 1536),
    ];
    for projection in ["w1", "w2", "w3"] {
        let shared_shape = if projection == "w2" {
            vec![5120, 2304]
        } else {
            vec![2304, 5120]
        };
        let shared_scale = if projection == "w2" {
            vec![160, 72]
        } else {
            vec![72, 160]
        };
        specs.push((
            format!("layers.0.ffn.shared_experts.{projection}.weight"),
            "F8_E4M3",
            shared_shape,
            11_796_480,
        ));
        specs.push((
            format!("layers.0.ffn.shared_experts.{projection}.scale"),
            "F8_E8M0",
            shared_scale,
            11_520,
        ));
        for id in 0..6 {
            let shape = if projection == "w2" {
                vec![5120, 1152]
            } else {
                vec![2304, 2560]
            };
            let scale = if projection == "w2" {
                vec![5120, 72]
            } else {
                vec![2304, 160]
            };
            specs.push((
                format!("layers.0.ffn.experts.{id}.{projection}.weight"),
                "I8",
                shape,
                5_898_240,
            ));
            specs.push((
                format!("layers.0.ffn.experts.{id}.{projection}.scale"),
                "F8_E8M0",
                scale,
                368_640,
            ));
        }
    }
    specs
}

fn install_tail_metadata(fixture: &Fixture, projection_shape: &[usize]) {
    let specs = tail_tensor_specs(projection_shape);
    let shard = "model.safetensors";
    let mut header = serde_json::Map::new();
    let mut weight_map = serde_json::Map::new();
    let mut offset = 0_u64;
    for (name, dtype, shape, bytes) in &specs {
        header.insert(
            name.clone(),
            json!({"dtype":dtype,"shape":shape,"data_offsets":[offset,offset+bytes]}),
        );
        weight_map.insert(name.clone(), json!(shard));
        offset += bytes;
    }
    let index =
        serde_json::to_vec(&json!({"metadata":{"total_size":offset},"weight_map":weight_map}))
            .unwrap();
    fs::write(fixture.dir.join("missing-index.json"), &index).unwrap();
    let mut header_bytes = serde_json::to_vec(&header).unwrap();
    while !header_bytes.len().is_multiple_of(8) {
        header_bytes.push(b' ');
    }
    let mut prefix = (header_bytes.len() as u64).to_le_bytes().to_vec();
    prefix.extend_from_slice(&header_bytes);
    fs::create_dir_all(fixture.dir.join("headers")).unwrap();
    fs::write(
        fixture.dir.join("headers/model.safetensors.header.bin"),
        &prefix,
    )
    .unwrap();
    fs::write(fixture.dir.join("headers.json"),serde_json::to_vec(&json!({
        "revision":REVISION,"index_sha256":format!("{:x}",Sha256::digest(&index)),
        "shards":{shard:{"header_bytes":prefix.len(),"header_sha256":format!("{:x}",Sha256::digest(&prefix)),"file_bytes":prefix.len() as u64 + offset}}
    })).unwrap()).unwrap();
    let weights = fixture.dir.join("missing-weights");
    fs::create_dir_all(&weights).unwrap();
    offset = prefix.len() as u64;
    for (name, dtype, shape, size) in specs.iter().take(4) {
        let bytes = match name.as_str() {
            "layers.0.ffn_norm.weight" => vec![0x3f80_u16; 5120]
                .into_iter()
                .flat_map(u16::to_le_bytes)
                .collect::<Vec<_>>(),
            "layers.0.hc_ffn_scale" => vec![1.0_f32; 3]
                .into_iter()
                .flat_map(f32::to_le_bytes)
                .collect(),
            _ => vec![0; usize::try_from(*size).unwrap()],
        };
        fs::write(weights.join(format!("{name}.bin")), &bytes).unwrap();
        fs::write(weights.join(format!("{name}.receipt.json")),serde_json::to_vec(&json!({
            "tensor":name,"shard":shard,"revision":REVISION,"bytes":size,"range":[offset,offset+size],
            "metadata":{"dtype":dtype,"shape":shape},"sha256":format!("{:x}",Sha256::digest(&bytes))
        })).unwrap()).unwrap();
        offset += size;
    }
}

#[test]
fn full_tail_hc_geometry_refuses_before_missing_moe_files() {
    let fixture = Fixture::new();
    let mut command = tail_fixture(&fixture);
    install_tail_metadata(&fixture, &[20480, 24]);
    assert_refused(
        &command.output().unwrap(),
        "tensor geometry mismatch: layers.0.hc_ffn_fn",
    );
}

#[test]
fn full_tail_changed_source_intermediates_with_updated_hash_refuse_before_moe() {
    let fixture = Fixture::new();
    for role in ["ffn_norm_in", "ffn_in"] {
        let mut command = tail_fixture(&fixture);
        install_tail_metadata(&fixture, &[24, 20480]);
        // Independent analytic zero residual and unit norm: both native
        // collapsed and normalized rows must be exactly zero.
        let name = format!("layer00.{role}.torch.bfloat16.bin");
        let mut bytes = vec![0_u8; 5120 * 2];
        bytes[..2].copy_from_slice(&0x3f80_u16.to_le_bytes());
        fs::write(fixture.dir.join(&name), &bytes).unwrap();
        change_tail_receipt(&fixture, |receipt| {
            receipt["outputs"][&name]["sha256"] = json!(format!("{:x}", Sha256::digest(&bytes)));
        });
        assert_refused(
            &command.output().unwrap(),
            if role == "ffn_norm_in" {
                "collapsed BF16 comparison failed before MoE payload reads"
            } else {
                "normalized BF16 comparison failed before MoE payload reads"
            },
        );
    }
}

#[test]
fn full_tail_updated_final_digest_is_comparison_data_and_does_not_change_preflight() {
    let fixture = Fixture::new();
    let command = tail_fixture(&fixture);
    install_tail_metadata(&fixture, &[24, 20480]);
    let mut bytes = fs::read(fixture.dir.join("expected.bin")).unwrap();
    bytes[0] ^= 1;
    fs::write(fixture.dir.join("expected.bin"), &bytes).unwrap();
    let mut args = command
        .get_args()
        .map(std::ffi::OsStr::to_os_string)
        .collect::<Vec<_>>();
    let position = args
        .iter()
        .position(|arg| arg == "--expected-sha256")
        .unwrap();
    args[position + 1] = format!("{:x}", Sha256::digest(&bytes)).into();
    let output = Command::new(env!("CARGO_BIN_EXE_mx"))
        .args(args)
        .output()
        .unwrap();
    // A valid final mutant gets through capture and analytic intermediate
    // checks to the deliberately absent MoE files. Parent checkpoint replay
    // must additionally observe the completed final mismatch.
    assert_refused(&output, "No such file or directory");
}
