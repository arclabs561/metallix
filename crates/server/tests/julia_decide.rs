//! `mx decide-julia` and `mx serve` decisions on the published Julia-1 checkpoint.

#![cfg(feature = "metal")]

use std::{
    io::{Read, Write},
    net::{SocketAddr, TcpListener, TcpStream},
    path::Path,
    process::{Child, Command},
    thread,
    time::{Duration, Instant},
};

use serde_json::{Value, json};
use sha2::Digest;

/// The reference generator's requests, in the typed `decide` shape, as JSON text.
fn typed_requests() -> Vec<(&'static str, String)> {
    let choice = |state: Value, question: &str, options: &[&str]| {
        let criteria: serde_json::Map<_, _> = options
            .iter()
            .enumerate()
            .map(|(i, &o)| (format!("o{i}"), json!(o)))
            .collect();
        json!({"state": state, "questions": {"q": {"type": "choice", "instructions": question, "criteria": criteria}}})
            .to_string()
    };
    vec![
        ("calib-a", choice(json!("hp 3"), "next?", &["heal", "attack"])),
        (
            "calib-b",
            json!({"state": "low", "questions": {"q": {"type": "noul", "instructions": "go?", "criteria": {"false": "yes", "true": "no"}}}})
                .to_string(),
        ),
        ("held-a", choice(json!("rain"), "take?", &["umbrella", "hat"])),
        (
            "held-b",
            json!({"state": "x=2", "questions": {"q": {"type": "score", "instructions": "rate", "criteria": ["good", "bad"]}}})
                .to_string(),
        ),
        ("held-c", choice(json!(""), "pick", &["a", "b"])),
        (
            "held-long",
            // Raw text keeps the source's state member order; `Value` would sort it.
            String::from(
                r#"{"state": {"inventory": ["sword", "potion", "map"], "enemies": 3, "hp": 12, "weather": "storm"},
                    "questions": {"q": {"type": "choice", "instructions": "What should the party do next?",
                    "criteria": {"o0": "retreat to camp", "o1": "fight", "o2": "drink potion", "o3": "read map"}}}}"#,
            ),
        ),
    ]
}

/// Opt-in: `JULIA_CHECKPOINT_DIR` is the pinned checkpoint directory and
/// `JULIA_REAL_REFERENCE` the JSON from `scripts/julia_real_f64_reference.py`.
/// CLI token IDs must equal the source `sequence` IDs, and CLI scores must
/// stay within the published-checkpoint gate: `gate_ratio` times the FP32
/// source's own error against float64.
#[test]
#[allow(clippy::cast_possible_truncation, reason = "printed scores are F32")]
#[ignore = "needs the real Julia-1 checkpoint via JULIA_CHECKPOINT_DIR and JULIA_REAL_REFERENCE"]
fn decide_julia_matches_source_reference_on_the_published_checkpoint() {
    let checkpoint = std::env::var("JULIA_CHECKPOINT_DIR").unwrap();
    let reference: Value = serde_json::from_str(
        &std::fs::read_to_string(std::env::var("JULIA_REAL_REFERENCE").unwrap()).unwrap(),
    )
    .unwrap();
    let gate: Value = serde_json::from_str(include_str!(
        "../../../fixtures/julia-1/f64-gate-reference.json"
    ))
    .unwrap();
    let ratio = gate["gate_ratio"].as_f64().unwrap();
    let scratch = std::env::temp_dir().join(format!("julia-decide-{}", std::process::id()));
    std::fs::create_dir_all(&scratch).unwrap();
    for (name, request) in typed_requests() {
        let case = reference["cases"]
            .as_array()
            .unwrap()
            .iter()
            .find(|case| case["name"] == name)
            .unwrap();
        let path = scratch.join(format!("{name}.json"));
        std::fs::write(&path, request).unwrap();
        let output = Command::new(env!("CARGO_BIN_EXE_mx"))
            .args(["decide-julia", "--model", &checkpoint, "--request"])
            .arg(&path)
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{name}: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        let receipt: Value = serde_json::from_slice(&output.stdout).unwrap();
        let answer = &receipt["answers"]["q"];
        assert_eq!(answer["input_ids"], case["input_ids"], "{name} token IDs");
        assert_eq!(answer["markers"], case["markers"], "{name} markers");
        let error = |scores: &Value| {
            scores
                .as_array()
                .unwrap()
                .iter()
                .zip(case["f64_scores"].as_array().unwrap())
                // Scores are printed F32 values; recover them exactly before widening.
                .map(|(a, b)| (f64::from(a.as_f64().unwrap() as f32) - b.as_f64().unwrap()).abs())
                .fold(0.0, f64::max)
        };
        let (native, source) = (error(&answer["scores"]), error(&case["source_f32_scores"]));
        println!("{name}: native score error {native:e}, source FP32 {source:e}");
        assert!(
            native <= ratio * source,
            "{name}: native score error {native:e} exceeds {ratio} x source {source:e}"
        );
    }
    let _ = std::fs::remove_dir_all(&scratch);
}

/// Removes wall-clock fields, the model label (directory path for the CLI,
/// registered ID for the server) and the request hash (the HTTP body adds
/// `model`), which differ by design.
fn without_timing(mut receipt: Value) -> Value {
    let object = receipt.as_object_mut().unwrap();
    for field in ["model", "checkpoint_load_ms", "session_load_ms"] {
        object.remove(field);
    }
    object["provenance"]
        .as_object_mut()
        .unwrap()
        .remove("request_sha256");
    for answer in object["answers"].as_object_mut().unwrap().values_mut() {
        let answer = answer.as_object_mut().unwrap();
        for field in ["forward_ms", "render_ms", "prefill_ms"] {
            answer.remove(field);
        }
    }
    receipt
}

fn cli_receipt(args: &[&str], request: &Path) -> Value {
    let output = Command::new(env!("CARGO_BIN_EXE_mx"))
        .args(args)
        .arg("--request")
        .arg(request)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    serde_json::from_slice(&output.stdout).unwrap()
}

fn post(address: SocketAddr, path: &str, body: &str) -> (u16, Value) {
    let mut stream = TcpStream::connect(address).unwrap();
    write!(
        stream,
        "POST {path} HTTP/1.1\r\nHost: localhost\r\nContent-Length: {}\r\n\r\n{body}",
        body.len()
    )
    .unwrap();
    stream.shutdown(std::net::Shutdown::Write).unwrap();
    let mut wire = Vec::new();
    stream.read_to_end(&mut wire).unwrap();
    let split = wire.windows(4).position(|w| w == b"\r\n\r\n").unwrap();
    let status = std::str::from_utf8(&wire[9..12]).unwrap().parse().unwrap();
    (status, serde_json::from_slice(&wire[split + 4..]).unwrap())
}

struct Server(Child);

impl Drop for Server {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

/// Opt-in: `JULIA_CHECKPOINT_DIR` as above; `METALLIX_QWEN_MODEL` optionally adds
/// a Qwen checkpoint. One `mx serve` registry loads every model at startup, and
/// each `/v1/decisions` receipt must equal the matching CLI receipt apart from
/// timing and the model label.
#[test]
#[ignore = "needs the real Julia-1 checkpoint via JULIA_CHECKPOINT_DIR"]
fn serve_decisions_equal_cli_receipts() {
    let julia = std::env::var("JULIA_CHECKPOINT_DIR").unwrap();
    let qwen = std::env::var("METALLIX_QWEN_MODEL").ok();
    let scratch = std::env::temp_dir().join(format!("julia-serve-{}", std::process::id()));
    std::fs::create_dir_all(&scratch).unwrap();
    let mut entries = vec![json!({"id": "julia-1", "kind": "julia", "path": julia})];
    if let Some(qwen) = &qwen {
        entries.push(json!({"id": "qwen", "kind": "qwen", "path": qwen}));
    }
    let registry = scratch.join("registry.json");
    std::fs::write(&registry, json!({"models": entries}).to_string()).unwrap();
    let address = TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap();
    let _server = Server(
        Command::new(env!("CARGO_BIN_EXE_mx"))
            .args(["serve", "--registry"])
            .arg(&registry)
            .args(["--listen", &address.to_string()])
            .spawn()
            .unwrap(),
    );
    let deadline = Instant::now() + Duration::from_secs(300);
    while TcpStream::connect(address).is_err() {
        assert!(Instant::now() < deadline, "server did not start");
        thread::sleep(Duration::from_millis(200));
    }

    let mut cases = typed_requests()
        .into_iter()
        .map(|(name, request)| {
            (
                "julia-1",
                vec!["decide-julia", "--model", julia.as_str()],
                name,
                request,
            )
        })
        .collect::<Vec<_>>();
    if let Some(qwen) = &qwen {
        let (name, request) = typed_requests().swap_remove(0);
        cases.push((
            "qwen",
            vec!["decide", "--model", qwen.as_str()],
            name,
            request,
        ));
    }
    for (model, args, name, request) in cases {
        let path = scratch.join(format!("{model}-{name}.json"));
        std::fs::write(&path, &request).unwrap();
        let cli = cli_receipt(&args, &path);
        // Prepend the model to the raw text so the state keeps its member order.
        let body = format!("{{\"model\":\"{model}\",{}", &request[1..]);
        let (status, served) = post(address, "/v1/decisions", &body);
        assert_eq!(status, 200, "{model} {name}: {served}");
        assert_eq!(served["model"], model);
        assert_eq!(
            served["provenance"]["request_sha256"],
            format!("{:x}", sha2::Sha256::digest(body.as_bytes()))
        );
        assert_eq!(
            without_timing(served),
            without_timing(cli),
            "{model} {name}"
        );
    }
    let _ = std::fs::remove_dir_all(&scratch);
}
