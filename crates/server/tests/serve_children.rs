//! `mx serve` runs each registered model in its own child process.

#![cfg(feature = "metal")]

use std::{
    io::{Read, Write},
    net::{Shutdown, SocketAddr, TcpListener, TcpStream},
    process::{Child, Command, Stdio},
    thread,
    time::{Duration, Instant},
};

use serde_json::{Value, json};

struct Server(Child);

impl Drop for Server {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

fn exchange(address: SocketAddr, request: &str, body: &str) -> (u16, Value) {
    let mut stream = TcpStream::connect(address).unwrap();
    write!(
        stream,
        "{request} HTTP/1.1\r\nHost: localhost\r\nContent-Length: {}\r\n\r\n{body}",
        body.len()
    )
    .unwrap();
    stream.shutdown(Shutdown::Write).unwrap();
    let mut wire = Vec::new();
    stream.read_to_end(&mut wire).unwrap();
    let split = wire.windows(4).position(|w| w == b"\r\n\r\n").unwrap() + 4;
    let status = std::str::from_utf8(&wire[9..12]).unwrap().parse().unwrap();
    (status, serde_json::from_slice(&wire[split..]).unwrap())
}

/// A model whose child cannot start is reported unavailable while the front
/// process keeps serving; no checkpoint is needed.
#[test]
fn a_model_that_fails_to_start_is_unavailable_and_the_server_stays_up() {
    let scratch = std::env::temp_dir().join(format!("serve-children-{}", std::process::id()));
    std::fs::create_dir_all(&scratch).unwrap();
    let registry = scratch.join("registry.json");
    let missing = scratch.join("no-such-checkpoint");
    std::fs::write(
        &registry,
        json!({"models": [{"id": "julia-1", "kind": "julia", "path": missing}]}).to_string(),
    )
    .unwrap();
    let address = TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap();
    let _server = Server(
        Command::new(env!("CARGO_BIN_EXE_mx"))
            .args(["serve", "--registry"])
            .arg(&registry)
            .args(["--listen", &address.to_string()])
            .stderr(Stdio::null())
            .spawn()
            .unwrap(),
    );
    let deadline = Instant::now() + Duration::from_secs(60);
    while TcpStream::connect(address).is_err() {
        assert!(Instant::now() < deadline, "server did not start");
        thread::sleep(Duration::from_millis(100));
    }

    let (status, listed) = exchange(address, "GET /v1/models", "");
    assert_eq!(status, 200);
    assert_eq!(listed["data"][0]["id"], "julia-1");
    assert_eq!(listed["data"][0]["loaded"], false);
    let (status, error) = exchange(
        address,
        "POST /v1/decisions",
        r#"{"model":"julia-1","state":"s","questions":{}}"#,
    );
    assert_eq!(status, 503);
    assert_eq!(error["error"]["code"], "model_worker_unavailable");
    let message = error["error"]["message"].as_str().unwrap();
    assert!(message.contains("exited during startup"), "{message}");
    assert_eq!(exchange(address, "GET /healthz", "").0, 200);
    let _ = std::fs::remove_dir_all(&scratch);
}

fn child_pid(id: &str) -> Option<u32> {
    let output = Command::new("pgrep")
        .args(["-f", &format!(r#"worker-entry.*"id":"{id}""#)])
        .output()
        .unwrap();
    String::from_utf8(output.stdout)
        .unwrap()
        .lines()
        .next()
        .map(|pid| pid.parse().unwrap())
}

/// `phys_footprint` in MiB from macOS `footprint`, which counts Metal memory.
fn footprint_mib(pid: u32) -> f64 {
    let output = Command::new("footprint")
        .args(["-p", &pid.to_string()])
        .output()
        .unwrap();
    let text = String::from_utf8(output.stdout).unwrap();
    let line = text
        .lines()
        .find(|line| line.trim_start().starts_with("phys_footprint:"))
        .unwrap_or_else(|| panic!("no footprint for {pid}: {text}"));
    let mut fields = line.split_whitespace().skip(1);
    let value: f64 = fields.next().unwrap().parse().unwrap();
    match fields.next().unwrap() {
        "KB" => value / 1024.0,
        "MB" => value,
        "GB" => value * 1024.0,
        unit => panic!("unexpected unit {unit}"),
    }
}

fn without_timing(mut receipt: Value) -> Value {
    let object = receipt.as_object_mut().unwrap();
    object.remove("checkpoint_load_ms");
    object.remove("session_load_ms");
    for answer in object["answers"].as_object_mut().unwrap().values_mut() {
        for field in ["forward_ms", "render_ms", "prefill_ms"] {
            answer.as_object_mut().unwrap().remove(field);
        }
    }
    receipt
}

/// Opt-in: `JULIA_CHECKPOINT_DIR` and `METALLIX_QWEN_MODEL`. Two on-demand
/// models under a budget that holds only one: each request stops the other
/// model's idle child, whose process and memory are then gone, and the front
/// process holds no model memory. A model started again returns the same
/// receipt as before it was stopped.
#[test]
#[ignore = "needs JULIA_CHECKPOINT_DIR and METALLIX_QWEN_MODEL"]
fn on_demand_models_take_turns_within_the_memory_budget() {
    let julia = std::env::var("JULIA_CHECKPOINT_DIR").unwrap();
    let qwen = std::env::var("METALLIX_QWEN_MODEL").unwrap();
    let scratch = std::env::temp_dir().join(format!("serve-budget-{}", std::process::id()));
    std::fs::create_dir_all(&scratch).unwrap();
    let registry = scratch.join("registry.json");
    // Declared sizes come from measured footprints: about 1.1-1.3 GB for
    // Julia and 3.3-3.7 GB for Qwen3-0.6B after its first request.
    std::fs::write(
        &registry,
        json!({"models": [
            {"id": "julia-1", "kind": "julia", "path": julia, "residency": "on_demand", "memory_mib": 1400},
            {"id": "qwen", "kind": "qwen", "path": qwen, "residency": "on_demand", "memory_mib": 4000},
        ]})
        .to_string(),
    )
    .unwrap();
    let address = TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap();
    let server = Server(
        Command::new(env!("CARGO_BIN_EXE_mx"))
            .args(["serve", "--registry"])
            .arg(&registry)
            .args([
                "--listen",
                &address.to_string(),
                "--memory-budget-mib",
                "4500",
            ])
            .spawn()
            .unwrap(),
    );
    let front = server.0.id();
    let deadline = Instant::now() + Duration::from_secs(60);
    while TcpStream::connect(address).is_err() {
        assert!(Instant::now() < deadline, "server did not start");
        thread::sleep(Duration::from_millis(100));
    }
    let loaded = || -> Vec<bool> {
        exchange(address, "GET /v1/models", "").1["data"]
            .as_array()
            .unwrap()
            .iter()
            .map(|model| model["loaded"].as_bool().unwrap())
            .collect()
    };
    assert_eq!(loaded(), [false, false], "on-demand models start stopped");
    assert_eq!(child_pid("julia-1"), None);

    let decision = |model: &str| {
        let body = format!(
            r#"{{"model":"{model}","state":"hp 3","questions":{{"a":{{"type":"choice","instructions":"next?","criteria":{{"heal":"drink a potion","attack":"swing the sword"}}}}}}}}"#
        );
        let (status, receipt) = exchange(address, "POST /v1/decisions", &body);
        assert_eq!(status, 200, "{model}: {receipt}");
        without_timing(receipt)
    };

    let first = decision("julia-1");
    let julia_pid = child_pid("julia-1").expect("julia child runs");
    let julia_mib = footprint_mib(julia_pid);
    let front_mib = footprint_mib(front);
    assert_eq!(loaded(), [true, false]);

    decision("qwen");
    let qwen_pid = child_pid("qwen").expect("qwen child runs");
    let qwen_mib = footprint_mib(qwen_pid);
    let front_after_mib = footprint_mib(front);
    assert_eq!(loaded(), [false, true], "julia was stopped to fit qwen");
    assert_eq!(child_pid("julia-1"), None, "julia's process is gone");
    println!(
        "front {front_mib:.0} -> {front_after_mib:.0} MiB; julia child {julia_mib:.0} MiB released; qwen child {qwen_mib:.0} MiB"
    );
    // The front process holds no model memory, so stopping a child frees all of it.
    assert!(
        front_after_mib < 100.0,
        "front process holds {front_after_mib} MiB"
    );
    assert!(julia_mib > 500.0, "julia child measured {julia_mib} MiB");

    let again = decision("julia-1");
    assert_eq!(loaded(), [true, false], "qwen was stopped to fit julia");
    assert_eq!(child_pid("qwen"), None, "qwen's process is gone");
    assert_eq!(again, first, "a restarted model returns the same receipt");
    let _ = std::fs::remove_dir_all(&scratch);
}
