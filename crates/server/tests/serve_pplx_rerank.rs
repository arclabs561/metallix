//! `mx serve` `/v1/rerank` on pplx-embed-v1-late-0.6b against the source
//! oracle in `fixtures/pplx-embed-v1-late-0.6b/late-reference.json` and
//! against `MaxSim` over the same server's `/v1/embeddings` vectors.

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

/// The fixture's declared policy: `MaxSim` within 1e-3 of the oracle.
const MAXSIM_VS_ORACLE: f64 = 1e-3;
/// Rerank and client-side `MaxSim` over served vectors use the same float32
/// vectors and summation order, so they agree to rounding.
const MAXSIM_VS_CLIENT: f64 = 1e-9;

struct Server(Child);

impl Drop for Server {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

fn post(address: SocketAddr, path: &str, body: &Value) -> (u16, Value) {
    let body = body.to_string();
    let mut stream = TcpStream::connect(address).unwrap();
    write!(
        stream,
        "POST {path} HTTP/1.1\r\nHost: localhost\r\nContent-Length: {}\r\n\r\n{body}",
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

fn floats(value: &Value) -> Vec<f64> {
    value
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect()
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

/// The fixture text of input `name`.
fn text(fixture: &Value, name: &str) -> String {
    fixture["inputs"]
        .as_array()
        .unwrap()
        .iter()
        .find(|input| input["name"] == name)
        .unwrap()["text"]
        .as_str()
        .unwrap()
        .to_owned()
}

/// Sixteen documents of mixed length: the fixture's seven, then nine
/// character prefixes of its long document.
fn sixteen_documents(fixture: &Value) -> Vec<String> {
    let mut documents: Vec<String> = fixture["inputs"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|input| input["task"] == "document")
        .map(|input| input["text"].as_str().unwrap().to_owned())
        .collect();
    let long = text(fixture, "long_document");
    for characters in [40, 80, 160, 240, 400, 640, 960, 1600, 2400] {
        documents.push(long.chars().take(characters).collect());
    }
    documents
}

/// Opt-in: `METALLIX_PPLX_LATE_MODEL` names pplx-embed-v1-late-0.6b at the
/// fixture's revision. `/v1/rerank` scores must match the oracle's `MaxSim`
/// for every fixture pair; for one query and 16 mixed-length documents its
/// scores and ranking must equal `MaxSim` over the same server's
/// `/v1/embeddings` vectors, `top_n` must cut that ranking, unknown fields are
/// refused, and HTTP latency is printed.
#[test]
#[ignore = "needs METALLIX_PPLX_LATE_MODEL"]
fn served_rerank_matches_the_oracle_and_client_side_maxsim() {
    let model = std::env::var("METALLIX_PPLX_LATE_MODEL").unwrap();
    let fixture: Value = serde_json::from_str(
        &std::fs::read_to_string(
            Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../fixtures/pplx-embed-v1-late-0.6b/late-reference.json"),
        )
        .unwrap(),
    )
    .unwrap();
    let scratch = std::env::temp_dir().join(format!("serve-pplx-rerank-{}", std::process::id()));
    let (_server, address) = start(&model, &scratch);

    let mut worst = 0.0_f64;
    for pair in fixture["pair_scores"].as_array().unwrap() {
        let request = json!({
            "model": "late",
            "query": text(&fixture, pair["query"].as_str().unwrap()),
            "documents": [text(&fixture, pair["document"].as_str().unwrap())],
        });
        let (status, response) = post(address, "/v1/rerank", &request);
        assert_eq!(status, 200, "{response}");
        let score = response["results"][0]["score"].as_f64().unwrap();
        let delta = (score - pair["score"].as_f64().unwrap()).abs();
        worst = worst.max(delta);
        assert!(delta <= MAXSIM_VS_ORACLE, "{pair}: {score}");
    }
    println!("worst rerank delta from the oracle's MaxSim: {worst:.3e}");

    let query = text(&fixture, "card_query");
    let documents = sixteen_documents(&fixture);
    let (status, embedded_query) = post(
        address,
        "/v1/embeddings",
        &json!({"model": "late", "input": query, "input_type": "query"}),
    );
    assert_eq!(status, 200, "{embedded_query}");
    let (status, embedded_documents) = post(
        address,
        "/v1/embeddings",
        &json!({"model": "late", "input": documents}),
    );
    assert_eq!(status, 200, "{embedded_documents}");
    let query_vectors = vectors(&embedded_query["data"][0]);
    let mut client: Vec<(usize, f64)> = (0..documents.len())
        .map(|index| {
            let document = vectors(&embedded_documents["data"][index]);
            (index, maxsim(&query_vectors, &document))
        })
        .collect();
    client.sort_by(|left, right| right.1.total_cmp(&left.1).then(left.0.cmp(&right.0)));

    let request = json!({"model": "late", "query": query, "documents": documents});
    let (status, reranked) = post(address, "/v1/rerank", &request);
    assert_eq!(status, 200, "{reranked}");
    let results = reranked["results"].as_array().unwrap();
    assert_eq!(results.len(), documents.len());
    for ((index, score), result) in client.iter().zip(results) {
        assert_eq!(result["index"], json!(index), "ranking");
        assert!(
            (result["score"].as_f64().unwrap() - score).abs() <= MAXSIM_VS_CLIENT,
            "document {index}"
        );
    }
    let mut top = request.clone();
    top["top_n"] = json!(3);
    let (status, top) = post(address, "/v1/rerank", &top);
    assert_eq!(status, 200, "{top}");
    assert_eq!(top["results"].as_array().unwrap()[..], results[..3]);

    for refused in [
        json!({"model": "late", "query": "q", "documents": ["d"], "return_documents": true}),
        json!({"model": "late", "query": "q", "documents": ["d"], "top_n": 2}),
        json!({"model": "late", "query": "q", "documents": []}),
        json!({"model": "late", "input": "q"}),
    ] {
        let (status, response) = post(address, "/v1/rerank", &refused);
        assert_eq!(status, 400, "{refused}: {response}");
    }

    let timed = || {
        let started = Instant::now();
        let (status, response) = post(address, "/v1/rerank", &request);
        assert_eq!(status, 200);
        (
            started.elapsed(),
            response["metallix"]["rerank_ms"].as_u64().unwrap(),
            response["usage"]["prompt_tokens"].as_u64().unwrap(),
        )
    };
    let (first, first_rerank_ms, tokens) = timed();
    let mut warm: Vec<_> = (0..10).map(|_| timed()).collect();
    warm.sort();
    let ms = |duration: Duration| duration.as_secs_f64() * 1e3;
    println!(
        "rerank 1 query x 16 documents ({tokens} tokens) over HTTP: first {:.1} ms (rerank {first_rerank_ms} ms), warm median {:.1} ms (rerank {} ms), min {:.1} ms",
        ms(first),
        ms(warm[5].0),
        warm[5].1,
        ms(warm[0].0)
    );
    let _ = std::fs::remove_dir_all(&scratch);
}
