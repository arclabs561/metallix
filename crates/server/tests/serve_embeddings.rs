//! `mx serve` `/v1/embeddings` on Qwen3-Embedding-0.6B against the source oracle.

#![cfg(feature = "metal")]

use std::{
    io::{Read, Write},
    net::{Shutdown, SocketAddr, TcpListener, TcpStream},
    path::Path,
    process::{Child, Command},
    thread,
    time::{Duration, Instant},
};

use serde_json::{Value, json};

/// The fixture's declared float32 policy: cosine(native, oracle) >= 1 - 1e-5.
const COSINE_MIN: f64 = 1.0 - 1e-5;

struct Server(Child);

impl Drop for Server {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

fn post(address: SocketAddr, body: &Value) -> (u16, Value) {
    let body = body.to_string();
    let mut stream = TcpStream::connect(address).unwrap();
    write!(
        stream,
        "POST /v1/embeddings HTTP/1.1\r\nHost: localhost\r\nContent-Length: {}\r\n\r\n{body}",
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

/// The oracle vector, stored either as a float list or as little-endian f32 hex.
fn oracle(input: &Value) -> Vec<f64> {
    if let Some(values) = input["embedding"].as_array() {
        return values.iter().map(|v| v.as_f64().unwrap()).collect();
    }
    let hex = input["embedding_f32le"].as_str().unwrap().as_bytes();
    hex.chunks_exact(8)
        .map(|word| {
            let bytes: Vec<u8> = word
                .chunks_exact(2)
                .map(|pair| u8::from_str_radix(std::str::from_utf8(pair).unwrap(), 16).unwrap())
                .collect();
            f64::from(f32::from_le_bytes(bytes.try_into().unwrap()))
        })
        .collect()
}

fn floats(value: &Value) -> Vec<f64> {
    value
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect()
}

fn cosine(left: &[f64], right: &[f64]) -> f64 {
    let dot = |a: &[f64], b: &[f64]| a.iter().zip(b).map(|(x, y)| x * y).sum::<f64>();
    dot(left, right) / (dot(left, left) * dot(right, right)).sqrt()
}

/// The request that reproduces a fixture text: a card-style query is sent as
/// `input_type: "query"` with its instruction, anything else as a document.
fn request_for(text: &str) -> Value {
    if let Some((instruction, query)) = text
        .strip_prefix("Instruct: ")
        .and_then(|rest| rest.split_once("\nQuery:"))
    {
        return json!({"model": "embed", "input": query, "input_type": "query", "instruction": instruction});
    }
    json!({"model": "embed", "input": text})
}

/// Opt-in: `METALLIX_QWEN_EMBEDDING_MODEL` names Qwen3-Embedding-0.6B at the
/// fixture's revision. Every fixture input goes through `mx serve` and must
/// tokenize to the oracle's length and match its vector within the fixture's
/// float32 cosine policy, at full width and truncated to 256. A batched request
/// returns the same vectors, in order, as single requests.
#[test]
#[ignore = "needs METALLIX_QWEN_EMBEDDING_MODEL"]
fn served_embeddings_match_the_source_oracle() {
    let model = std::env::var("METALLIX_QWEN_EMBEDDING_MODEL").unwrap();
    let fixture: Value = serde_json::from_str(
        &std::fs::read_to_string(
            Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../fixtures/qwen3-embedding-0.6b/embedding-reference.json"),
        )
        .unwrap(),
    )
    .unwrap();
    let scratch = std::env::temp_dir().join(format!("serve-embeddings-{}", std::process::id()));
    std::fs::create_dir_all(&scratch).unwrap();
    let registry = scratch.join("registry.json");
    std::fs::write(
        &registry,
        json!({"models": [{"id": "embed", "kind": "qwen_embedding", "path": model}]}).to_string(),
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
            .spawn()
            .unwrap(),
    );
    let deadline = Instant::now() + Duration::from_secs(120);
    while TcpStream::connect(address).is_err() {
        assert!(Instant::now() < deadline, "server did not start");
        thread::sleep(Duration::from_millis(200));
    }
    let mut stream = TcpStream::connect(address).unwrap();
    stream
        .write_all(b"GET /v1/models HTTP/1.1\r\nHost: localhost\r\n\r\n")
        .unwrap();
    let mut wire = Vec::new();
    stream.read_to_end(&mut wire).unwrap();
    let split = wire.windows(4).position(|w| w == b"\r\n\r\n").unwrap() + 4;
    let listed: Value = serde_json::from_slice(&wire[split..]).unwrap();
    assert_eq!(listed["data"][0]["capabilities"], json!(["embed"]));

    let mut worst: f64 = 1.0;
    for input in fixture["inputs"].as_array().unwrap() {
        let name = input["name"].as_str().unwrap();
        let expected = oracle(input);
        for dimensions in [None, Some(256)] {
            let mut request = request_for(input["text"].as_str().unwrap());
            if let Some(dimensions) = dimensions {
                request["dimensions"] = json!(dimensions);
            }
            let (status, response) = post(address, &request);
            assert_eq!(status, 200, "{name}: {response}");
            assert_eq!(
                response["usage"]["prompt_tokens"],
                input["input_ids"].as_array().unwrap().len(),
                "{name}: token count"
            );
            let served = floats(&response["data"][0]["embedding"]);
            assert_eq!(served.len(), dimensions.unwrap_or(expected.len()), "{name}");
            let norm = served.iter().map(|x| x * x).sum::<f64>().sqrt();
            assert!((norm - 1.0).abs() < 1e-5, "{name}: norm {norm}");
            let similarity = cosine(&served, &expected[..served.len()]);
            worst = worst.min(similarity);
            assert!(
                similarity >= COSINE_MIN,
                "{name} {dimensions:?}: cosine {similarity}"
            );
        }
    }
    println!("worst cosine against the oracle: {worst}");

    let documents: Vec<&str> = fixture["inputs"]
        .as_array()
        .unwrap()
        .iter()
        .map(|input| input["text"].as_str().unwrap())
        .filter(|text| !text.starts_with("Instruct: "))
        .take(3)
        .collect();
    let (status, batch) = post(address, &json!({"model": "embed", "input": documents}));
    assert_eq!(status, 200, "{batch}");
    for (index, text) in documents.iter().enumerate() {
        assert_eq!(batch["data"][index]["index"], index);
        let (_, single) = post(address, &json!({"model": "embed", "input": text}));
        assert_eq!(
            batch["data"][index]["embedding"],
            single["data"][0]["embedding"]
        );
    }
    let _ = std::fs::remove_dir_all(&scratch);
}
