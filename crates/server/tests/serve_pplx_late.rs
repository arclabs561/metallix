//! `mx serve` `/v1/embeddings` on pplx-embed-v1-late-0.6b against the source
//! oracle in `fixtures/pplx-embed-v1-late-0.6b/late-reference.json`.

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

/// The fixture's declared float32 policy: per scored token, cosine(served,
/// oracle) at least 1 - 1e-5; `MaxSim` within 1e-3 of the oracle and 1e-2 of
/// the card's printed scores.
const TOKEN_COSINE_MIN: f64 = 1.0 - 1e-5;
const MAXSIM_VS_ORACLE: f64 = 1e-3;
const MAXSIM_VS_CARD: f64 = 1e-2;

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

/// Starts `mx serve` on a one-entry `pplx_late` registry and waits for it.
fn start(model: &str, scratch: &Path) -> (Server, SocketAddr) {
    std::fs::create_dir_all(scratch).unwrap();
    let registry = scratch.join("registry.json");
    std::fs::write(
        &registry,
        json!({"models": [{"id": "late", "kind": "pplx_late", "path": model}]}).to_string(),
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

/// Per-token vectors from one served entry.
fn vectors(entry: &Value) -> Vec<Vec<f64>> {
    entry["embedding"]
        .as_array()
        .unwrap()
        .iter()
        .map(floats)
        .collect()
}

fn maxsim(query: &[Vec<f64>], document: &[Vec<f64>]) -> f64 {
    query
        .iter()
        .map(|q| {
            document
                .iter()
                .map(|d| q.iter().zip(d).map(|(a, b)| a * b).sum::<f64>())
                .fold(f64::NEG_INFINITY, f64::max)
        })
        .sum()
}

/// Checks one served entry against its oracle input: token count, scored
/// positions, offsets shape, truncation flag and per-token cosine. Returns the
/// worst `1 - cosine`.
fn check_entry(input: &Value, entry: &Value) -> f64 {
    let name = input["name"].as_str().unwrap();
    assert_eq!(
        entry["positions"], input["scored_positions"],
        "{name}: positions"
    );
    assert_eq!(
        entry["offsets"].as_array().unwrap().len(),
        input["scored_positions"].as_array().unwrap().len(),
        "{name}: one offset per vector"
    );
    // Offsets are character spans of the input text, in order and within it;
    // the prefix token and query-expansion tokens have none.
    let text_chars = input["text"].as_str().unwrap().chars().count();
    let spans: Vec<(u64, u64)> = entry["offsets"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|offset| Some((offset[0].as_u64()?, offset[1].as_u64()?)))
        .collect();
    assert_eq!(
        spans.first().map(|span| span.0),
        Some(0),
        "{name}: first span"
    );
    assert!(
        spans.windows(2).all(|pair| pair[0].0 <= pair[1].0),
        "{name}: order"
    );
    assert!(
        spans
            .iter()
            .all(|&(start, end)| start < end && end <= text_chars as u64),
        "{name}"
    );
    let ids = input["input_ids"].as_array().unwrap();
    let expansion = ids.iter().filter(|&id| id == &json!(151_642)).count();
    let nulls = entry["offsets"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|o| o.is_null())
        .count();
    assert_eq!(
        nulls,
        1 + expansion,
        "{name}: prefix and expansion have no offsets"
    );
    // Only the two long fixture inputs exceed the query or document length.
    assert_eq!(
        entry["truncated"],
        json!(name.starts_with("long_")),
        "{name}"
    );
    let oracle: Vec<Vec<f64>> = f32le_hex(&input["embeddings_f32le"])
        .chunks_exact(128)
        .map(<[f64]>::to_vec)
        .collect();
    let served = vectors(entry);
    assert_eq!(served.len(), oracle.len(), "{name}: vectors");
    let mut worst = 0.0_f64;
    for (token, (left, right)) in served.iter().zip(&oracle).enumerate() {
        let gap = 1.0 - cosine(left, right);
        worst = worst.max(gap);
        assert!(gap <= 1.0 - TOKEN_COSINE_MIN, "{name}[{token}]: {gap}");
    }
    worst
}

/// Prints HTTP round-trip and server-reported embedding time for `request`.
fn print_latency(address: SocketAddr, label: &str, request: &Value) {
    let timed = || {
        let started = Instant::now();
        let (status, response) = post(address, request);
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
        "{label} over HTTP: first {:.1} ms (embed {first_embed_ms} ms), warm median {:.1} ms (embed {} ms), min {:.1} ms",
        ms(first),
        ms(warm[5].0),
        warm[5].1,
        ms(warm[0].0)
    );
}

/// Opt-in: `METALLIX_PPLX_LATE_MODEL` names pplx-embed-v1-late-0.6b at the
/// fixture's revision. Every fixture input goes through `mx serve` as a query or
/// document; positions, token counts and each token vector must match the
/// oracle within the fixture's policy, a batch must equal single requests,
/// `MaxSim` over served vectors must match the oracle's and the card's scores,
/// unsupported fields are refused, and HTTP latency is printed.
#[test]
#[ignore = "needs METALLIX_PPLX_LATE_MODEL"]
fn served_late_vectors_match_the_source_oracle() {
    let model = std::env::var("METALLIX_PPLX_LATE_MODEL").unwrap();
    let fixture: Value = serde_json::from_str(
        &std::fs::read_to_string(
            Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../fixtures/pplx-embed-v1-late-0.6b/late-reference.json"),
        )
        .unwrap(),
    )
    .unwrap();
    let scratch = std::env::temp_dir().join(format!("serve-pplx-late-{}", std::process::id()));
    let (_server, address) = start(&model, &scratch);

    let inputs = fixture["inputs"].as_array().unwrap();
    let mut served = std::collections::HashMap::new();
    let mut worst = 0.0_f64;
    for input in inputs {
        let request = json!({"model": "late", "input": input["text"], "input_type": input["task"]});
        let (status, response) = post(address, &request);
        assert_eq!(status, 200, "{response}");
        assert_eq!(
            response["usage"]["prompt_tokens"],
            input["input_ids"].as_array().unwrap().len(),
            "{}",
            input["name"]
        );
        let entry = &response["data"][0];
        worst = worst.max(check_entry(input, entry));
        served.insert(input["name"].as_str().unwrap().to_owned(), vectors(entry));
    }

    // A document batch returns the same vectors, in order, as single requests.
    let documents: Vec<&Value> = inputs
        .iter()
        .filter(|input| input["task"] == "document")
        .collect();
    let texts: Vec<&Value> = documents.iter().map(|input| &input["text"]).collect();
    let (status, batch) = post(address, &json!({"model": "late", "input": texts}));
    assert_eq!(status, 200, "{batch}");
    for (index, input) in documents.iter().enumerate() {
        assert_eq!(
            vectors(&batch["data"][index]),
            served[input["name"].as_str().unwrap()]
        );
    }

    let mut worst_score = 0.0_f64;
    for pair in fixture["pair_scores"].as_array().unwrap() {
        let score = maxsim(
            &served[pair["query"].as_str().unwrap()],
            &served[pair["document"].as_str().unwrap()],
        );
        let delta = (score - pair["score"].as_f64().unwrap()).abs();
        worst_score = worst_score.max(delta);
        assert!(delta <= MAXSIM_VS_ORACLE, "{pair}: {score}");
    }
    for (index, card) in fixture["card_scores"]
        .as_array()
        .unwrap()
        .iter()
        .enumerate()
    {
        let score = maxsim(
            &served["card_query"],
            &served[&format!("card_document_{index}")],
        );
        assert!(
            (score - card.as_f64().unwrap()).abs() <= MAXSIM_VS_CARD,
            "card {index}"
        );
    }
    println!("worst token 1 - cosine {worst:.3e}; worst MaxSim delta {worst_score:.3e}");

    for refused in [
        json!({"model": "late", "input": "x", "dimensions": 64}),
        json!({"model": "late", "input": "x", "encoding_format": "int8"}),
        json!({"model": "late", "input": [["chunked"]]}),
        json!({"model": "late", "input": "x", "instruction": "Find code"}),
    ] {
        let (status, response) = post(address, &refused);
        assert_eq!(status, 400, "{refused}: {response}");
    }

    let text =
        |name: &str| inputs.iter().find(|input| input["name"] == name).unwrap()["text"].clone();
    print_latency(
        address,
        "query (32 ids)",
        &json!({"model": "late", "input": text("card_query"), "input_type": "query"}),
    );
    print_latency(
        address,
        "document (9 ids)",
        &json!({"model": "late", "input": text("card_document_0")}),
    );
    print_latency(
        address,
        "document (512 ids)",
        &json!({"model": "late", "input": text("long_document")}),
    );
    let _ = std::fs::remove_dir_all(&scratch);
}
