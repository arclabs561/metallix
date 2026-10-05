//! `mx serve` `/v1/embeddings` on pplx-embed-context-v1-0.6b against the
//! source oracle in `fixtures/pplx-embed-context-v1-0.6b/context-reference.json`.

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

/// The fixture's declared float32 policy: per chunk, cosine(served, oracle)
/// at least 1 - 1e-5; int8 codes at most 1 apart; binary signs equal except
/// where the oracle value's magnitude is at most 1e-4.
const COSINE_MIN: f64 = 1.0 - 1e-5;
const INT8_MAX_CODE_DIFF: i64 = 1;
const BINARY_NEAR_ZERO: f64 = 1e-4;

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

fn f32le_hex(value: &Value) -> Vec<f64> {
    let hex = value.as_str().unwrap().as_bytes();
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

fn integers(value: &Value) -> Vec<i64> {
    value
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_i64().unwrap())
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

/// Starts `mx serve` on a one-entry `pplx_context` registry and waits for it.
fn start(model: &str, scratch: &Path) -> (Server, SocketAddr) {
    std::fs::create_dir_all(scratch).unwrap();
    let registry = scratch.join("registry.json");
    std::fs::write(
        &registry,
        json!({"models": [{"id": "ctx", "kind": "pplx_context", "path": model}]}).to_string(),
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
            .args(["--listen", &address.to_string()])
            .spawn()
            .unwrap(),
    );
    let deadline = Instant::now() + Duration::from_secs(120);
    while TcpStream::connect(address).is_err() {
        assert!(Instant::now() < deadline, "server did not start");
        thread::sleep(Duration::from_millis(200));
    }
    (server, address)
}

/// Running totals across chunks: worst `1 - cosine` and int8 exact matches.
#[derive(Default)]
struct Agreement {
    worst_gap: f64,
    int8_equal: usize,
    int8_total: usize,
}

/// Checks one served chunk against the oracle's chunk under `encoding`.
fn check_chunk(
    encoding: &str,
    document: &Value,
    chunk: usize,
    served: &Value,
    agreement: &mut Agreement,
) {
    let name = document["name"].as_str().unwrap();
    let oracle = f32le_hex(&document["pooled_f32le"][chunk]);
    match encoding {
        "float" => {
            let served = floats(served);
            if oracle.iter().all(|&value| value == 0.0) {
                assert!(served.iter().all(|&value| value == 0.0), "{name}[{chunk}]");
            } else {
                let gap = 1.0 - cosine(&served, &oracle);
                agreement.worst_gap = agreement.worst_gap.max(gap);
                assert!(gap <= 1.0 - COSINE_MIN, "{name}[{chunk}]: {gap}");
            }
        }
        "int8" => {
            let expected = integers(&document["int8"][chunk]);
            for (served, expected) in integers(served).iter().zip(&expected) {
                let diff = (served - expected).abs();
                assert!(diff <= INT8_MAX_CODE_DIFF, "{name}[{chunk}]: {diff}");
                agreement.int8_equal += usize::from(diff == 0);
                agreement.int8_total += 1;
            }
        }
        _ => {
            let expected = integers(&document["binary"][chunk]);
            for ((served, expected), value) in integers(served).iter().zip(&expected).zip(&oracle) {
                assert!(
                    served == expected || value.abs() <= BINARY_NEAR_ZERO,
                    "{name}[{chunk}]: sign at {value}"
                );
            }
        }
    }
}

/// Prints HTTP round-trip and server-reported embedding time for one document.
fn print_latency(address: SocketAddr, document: &Value) {
    let request = json!({"model": "ctx", "input": [&document["chunks"]]});
    let timed = || {
        let started = Instant::now();
        let (status, response) = post(address, &request);
        assert_eq!(status, 200);
        (
            started.elapsed(),
            response["metallix"]["embed_ms"].as_u64().unwrap(),
        )
    };
    let (first, first_embed_ms) = timed();
    let mut warm: Vec<_> = (0..10).map(|_| timed()).collect();
    warm.sort();
    let ms = |duration: Duration| duration.as_secs_f64() * 1e3;
    println!(
        "{}-chunk document ({} tokens) over HTTP: first {:.1} ms (embed {first_embed_ms} ms), warm median {:.1} ms (embed {} ms), min {:.1} ms",
        document["chunks"].as_array().unwrap().len(),
        document["input_ids"].as_array().unwrap().len(),
        ms(first),
        ms(warm[5].0),
        warm[5].1,
        ms(warm[0].0)
    );
}

/// Opt-in: `METALLIX_PPLX_CONTEXT_MODEL` names pplx-embed-context-v1-0.6b at the
/// fixture's revision. All fixture documents go through `mx serve` in one
/// request per encoding; each chunk must match the oracle within the fixture's
/// policy, in document then chunk order. Unknown fields and flat inputs are
/// refused, and a 4-chunk document's HTTP latency is printed.
#[test]
#[ignore = "needs METALLIX_PPLX_CONTEXT_MODEL"]
fn served_context_chunks_match_the_source_oracle() {
    let model = std::env::var("METALLIX_PPLX_CONTEXT_MODEL").unwrap();
    let fixture: Value = serde_json::from_str(
        &std::fs::read_to_string(
            Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../fixtures/pplx-embed-context-v1-0.6b/context-reference.json"),
        )
        .unwrap(),
    )
    .unwrap();
    let scratch = std::env::temp_dir().join(format!("serve-pplx-context-{}", std::process::id()));
    let (_server, address) = start(&model, &scratch);

    let documents = fixture["documents"].as_array().unwrap();
    let input: Vec<&Value> = documents
        .iter()
        .map(|document| &document["chunks"])
        .collect();
    let tokens: usize = documents
        .iter()
        .map(|document| document["input_ids"].as_array().unwrap().len())
        .sum();
    let mut agreement = Agreement::default();
    for encoding in ["float", "int8", "binary"] {
        let (status, response) = post(
            address,
            &json!({"model": "ctx", "input": input, "encoding_format": encoding}),
        );
        assert_eq!(status, 200, "{encoding}: {response}");
        assert_eq!(response["usage"]["prompt_tokens"], tokens, "{encoding}");
        assert_eq!(response["metallix"]["encoding"], encoding);
        let mut data = response["data"].as_array().unwrap().iter();
        for (document_index, document) in documents.iter().enumerate() {
            for chunk in 0..document["chunks"].as_array().unwrap().len() {
                let entry = data.next().expect("an entry per chunk");
                assert_eq!(
                    (&entry["document"], &entry["chunk"]),
                    (&json!(document_index), &json!(chunk)),
                );
                check_chunk(
                    encoding,
                    document,
                    chunk,
                    &entry["embedding"],
                    &mut agreement,
                );
            }
        }
        assert!(data.next().is_none(), "{encoding}: extra entries");
    }
    println!(
        "worst 1 - cosine {:.3e}; int8 exact {}/{}",
        agreement.worst_gap, agreement.int8_equal, agreement.int8_total
    );

    for refused in [
        json!({"model": "ctx", "input": [["x"]], "dimensions": 256}),
        json!({"model": "ctx", "input": "one flat text"}),
        json!({"model": "ctx", "input": [["x"]], "encoding_format": "base64"}),
    ] {
        let (status, response) = post(address, &refused);
        assert_eq!(status, 400, "{refused}: {response}");
    }

    let four_chunks = documents
        .iter()
        .find(|document| document["chunks"].as_array().unwrap().len() == 4)
        .unwrap();
    print_latency(address, four_chunks);
    let _ = std::fs::remove_dir_all(&scratch);
}
