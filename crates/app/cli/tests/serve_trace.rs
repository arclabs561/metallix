//! `mx serve --trace-out` keeps the last request when the server is stopped
//! by SIGTERM.

#![cfg(all(feature = "metal", feature = "timeline"))]

use std::{
    io::{Read, Write},
    net::{Shutdown, TcpListener, TcpStream},
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

/// One request to a model whose child cannot start (no checkpoint needed),
/// then SIGTERM: the front's timeline must hold that request's
/// `http.request` span, begun and ended.
#[test]
fn a_sigterm_keeps_the_last_request_in_the_timeline() {
    let scratch = std::env::temp_dir().join(format!("serve-trace-{}", std::process::id()));
    std::fs::create_dir_all(&scratch).unwrap();
    let registry = scratch.join("registry.json");
    let trace = scratch.join("trace.json");
    std::fs::write(
        &registry,
        json!({"models": [{"id": "julia-1", "kind": "julia", "path": scratch.join("missing")}]})
            .to_string(),
    )
    .unwrap();
    let address = TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap();
    let mut server = Server(
        Command::new(env!("CARGO_BIN_EXE_mx"))
            .arg("--trace-out")
            .arg(&trace)
            .args(["serve", "--registry"])
            .arg(&registry)
            .args(["--listen", &address.to_string()])
            .env("METALLIX_LOG", "info")
            .stderr(Stdio::null())
            .spawn()
            .unwrap(),
    );
    let deadline = Instant::now() + Duration::from_secs(60);
    while TcpStream::connect(address).is_err() {
        assert!(Instant::now() < deadline, "server did not start");
        thread::sleep(Duration::from_millis(100));
    }

    let body = r#"{"model":"julia-1","state":"s","questions":{}}"#;
    let mut stream = TcpStream::connect(address).unwrap();
    write!(
        stream,
        "POST /v1/decisions HTTP/1.1\r\nHost: localhost\r\nContent-Length: {}\r\n\r\n{body}",
        body.len()
    )
    .unwrap();
    stream.shutdown(Shutdown::Write).unwrap();
    let mut wire = Vec::new();
    stream.read_to_end(&mut wire).unwrap();
    assert!(
        wire.starts_with(b"HTTP/1.1 503"),
        "{}",
        String::from_utf8_lossy(&wire)
    );
    // The forwarding thread ends its span just after the reply; a stop that
    // lands inside that window cannot record the span's end.
    thread::sleep(Duration::from_millis(300));

    let pid = server.0.id().to_string();
    assert!(
        Command::new("kill")
            .args(["-TERM", &pid])
            .status()
            .unwrap()
            .success()
    );
    let deadline = Instant::now() + Duration::from_secs(10);
    let status = loop {
        if let Some(status) = server.0.try_wait().unwrap() {
            break status;
        }
        assert!(Instant::now() < deadline, "server did not exit on SIGTERM");
        thread::sleep(Duration::from_millis(50));
    };

    // Chrome's format lets a cut-off array omit its closing bracket, so a
    // timeline written without finishing still parses with one appended.
    let text = std::fs::read_to_string(&trace).unwrap();
    let complete = serde_json::from_str::<Vec<Value>>(&text).is_ok();
    let events: Vec<Value> = serde_json::from_str(&text)
        .or_else(|_| serde_json::from_str(&format!("{text}\n]")))
        .unwrap_or_else(|error| panic!("unreadable timeline ({error}): {text}"));
    // The front handled only this request.
    let phases: Vec<&str> = events
        .iter()
        .filter(|event| event["name"] == "http.request")
        .filter_map(|event| event["ph"].as_str())
        .collect();
    assert!(
        phases.contains(&"B") && phases.contains(&"E"),
        "the request's span is not in the timeline: {phases:?}"
    );
    assert!(complete, "the timeline was not finished");
    assert!(status.success(), "{status}");
    let _ = std::fs::remove_dir_all(&scratch);
}
