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
