//! Exercises the real HTTP acceptor, worker dispatch and admission release.
use super::*;
use crate::transcriptions::{self, Error, Request, Response};
use std::{
    io::{Read, Write},
    net::{Shutdown, TcpStream},
};

struct FixtureAsr;
impl ModelWorker for FixtureAsr {
    fn chat(&mut self) -> Option<&mut dyn ChatBackend> {
        None
    }
    fn decide(&mut self, _: &[u8], _: &str) -> Option<Result<Value, String>> {
        None
    }
    fn transcribe(
        &mut self,
        request: &Request,
        control: &mut transcriptions::Control<'_>,
    ) -> Option<Result<Response, Error>> {
        if let Err(stop) = control.check() {
            return Some(Err(stop.into()));
        }
        let audio = audio::decode_wav(request.file()).unwrap();
        assert_eq!(audio.samples().len(), 6);
        assert_eq!(request.model, "asr");
        Some(Ok(transcriptions::result(
            request.format,
            "heard".into(),
            "English",
            audio.duration_seconds(),
        )))
    }
}

#[test]
fn multipart_socket_reaches_transcription_worker_and_releases_admission() {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    let (sender, receiver) = sync_channel(0);
    let worker = thread::spawn(move || model_worker_loop(&mut FixtureAsr, receiver));
    let occupied = Arc::new(AtomicBool::new(false));
    let check_occupied = Arc::clone(&occupied);
    let server = thread::spawn(move || {
        let models = [ServedModel {
            id: "asr".into(),
            generates: false,
            capabilities: &["transcribe"],
            jobs: sender,
            occupied,
            alive: Arc::new(AtomicBool::new(true)),
            engine: None,
        }];
        serve_models(
            &listener,
            &models,
            Duration::from_secs(2),
            TransportLimits::default(),
            Some(3),
        )
        .unwrap();
    });
    let wav = include_bytes!("../../../audio/tests/fixtures/pcm16_stereo.wav");
    for format in ["json", "text", "verbose_json"] {
        let body = transcriptions::tests::form(&[
            ("model", b"asr"),
            ("file", wav),
            ("response_format", format.as_bytes()),
        ]);
        let mut client = TcpStream::connect(address).unwrap();
        client
            .set_read_timeout(Some(Duration::from_secs(10)))
            .unwrap();
        write!(client, "POST /v1/audio/transcriptions HTTP/1.1\r\nHost: localhost\r\nContent-Type: multipart/form-data; boundary=boundary\r\nContent-Length: {}\r\n\r\n", body.len()).unwrap();
        client.write_all(&body).unwrap();
        client.shutdown(Shutdown::Write).unwrap();
        let mut response = String::new();
        client.read_to_string(&mut response).unwrap();
        // Half-closed clients may receive the transport's one interim probe.
        let response = response
            .strip_prefix("HTTP/1.1 100 Continue\r\n\r\n")
            .unwrap_or(&response);
        assert!(response.starts_with("HTTP/1.1 200"), "{response}");
        assert!(response.contains("heard"));
        assert!(!check_occupied.load(Ordering::Acquire));
        if format == "text" {
            assert!(response.contains("Content-Type: text/plain"));
        }
        if format == "verbose_json" {
            assert!(response.contains("\"duration\":0.000375"));
        }
    }
    server.join().unwrap();
    worker.join().unwrap();
}

#[test]
fn child_rejects_upload_and_json_lengths_before_waiting_for_a_body() {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    let server = thread::spawn(move || {
        serve_models(
            &listener,
            &[],
            Duration::from_secs(2),
            TransportLimits::default(),
            Some(2),
        )
        .unwrap();
    });
    for (path, length, limit) in [
        (
            transcriptions::PATH,
            transcriptions::BODY_BYTES + 1,
            transcriptions::BODY_BYTES,
        ),
        ("/v1/responses", 1024 * 1024 + 1, 1024 * 1024),
    ] {
        let mut client = TcpStream::connect(address).unwrap();
        client
            .set_read_timeout(Some(Duration::from_secs(2)))
            .unwrap();
        write!(
            client,
            "POST {path} HTTP/1.1\r\nHost: localhost\r\nContent-Length: {length}\r\n\r\n"
        )
        .unwrap();
        let mut response = String::new();
        client.read_to_string(&mut response).unwrap();
        assert!(response.starts_with("HTTP/1.1 413"), "{response}");
        assert!(response.contains(&format!("of {limit} bytes")));
    }
    server.join().unwrap();
}

struct BlockedAsr {
    entered: std::sync::mpsc::Sender<()>,
    native_release: std::sync::mpsc::Receiver<()>,
    cleanup_entered: std::sync::mpsc::Sender<()>,
    cleanup_release: std::sync::mpsc::Receiver<()>,
    cleaned: Arc<AtomicBool>,
    first: bool,
}

impl ModelWorker for BlockedAsr {
    fn chat(&mut self) -> Option<&mut dyn ChatBackend> {
        None
    }
    fn decide(&mut self, _: &[u8], _: &str) -> Option<Result<Value, String>> {
        None
    }
    fn transcribe(
        &mut self,
        request: &Request,
        control: &mut transcriptions::Control<'_>,
    ) -> Option<Result<Response, Error>> {
        if !self.first {
            return FixtureAsr.transcribe(request, control);
        }
        self.first = false;
        self.entered.send(()).unwrap();
        // Stand in for an uninterruptible native operation, then its cleanup.
        self.native_release
            .recv_timeout(Duration::from_secs(10))
            .unwrap();
        let stopped = control.check().expect_err("forwarder EOF must cancel");
        self.cleanup_entered.send(()).unwrap();
        self.cleanup_release
            .recv_timeout(Duration::from_secs(10))
            .unwrap();
        self.cleaned.store(true, Ordering::Release);
        Some(Err(stopped.into()))
    }
}

fn transcription_client(address: std::net::SocketAddr) -> TcpStream {
    let wav = include_bytes!("../../../audio/tests/fixtures/pcm16_stereo.wav");
    let body = transcriptions::tests::form(&[("model", b"asr"), ("file", wav)]);
    let mut client = TcpStream::connect(address).unwrap();
    client
        .set_read_timeout(Some(Duration::from_secs(10)))
        .unwrap();
    write!(client, "POST /v1/audio/transcriptions HTTP/1.1\r\nHost: localhost\r\nx-metallix-cancel-on-eof: 1\r\nContent-Type: multipart/form-data; boundary=boundary\r\nContent-Length: {}\r\n\r\n", body.len()).unwrap();
    client.write_all(&body).unwrap();
    client
}

fn assert_socket_retained(client: &mut TcpStream, occupied: &AtomicBool) {
    assert!(occupied.load(Ordering::Acquire));
    client
        .set_read_timeout(Some(Duration::from_millis(50)))
        .unwrap();
    let error = client
        .read(&mut [0])
        .expect_err("socket must stay open through cleanup");
    assert!(matches!(
        error.kind(),
        std::io::ErrorKind::WouldBlock | std::io::ErrorKind::TimedOut
    ));
    client
        .set_read_timeout(Some(Duration::from_secs(10)))
        .unwrap();
}

fn transcription_server(
    worker: impl ModelWorker + Send + 'static,
    timeout: Duration,
    requests: usize,
) -> (
    std::net::SocketAddr,
    Arc<AtomicBool>,
    thread::JoinHandle<()>,
) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    let occupied = Arc::new(AtomicBool::new(false));
    let check = Arc::clone(&occupied);
    let server = thread::spawn(move || {
        let (sender, receiver) = sync_channel(0);
        let worker = thread::spawn(move || model_worker_loop(&mut { worker }, receiver));
        {
            let models = [ServedModel {
                id: "asr".into(),
                generates: false,
                capabilities: &["transcribe"],
                jobs: sender,
                occupied,
                alive: Arc::new(AtomicBool::new(true)),
                engine: None,
            }];
            serve_models(
                &listener,
                &models,
                timeout,
                TransportLimits::default(),
                Some(requests),
            )
            .unwrap();
        }
        worker.join().unwrap();
    });
    (address, check, server)
}

#[test]
fn transcription_disconnect_retains_admission_until_native_cleanup_then_reuses_worker() {
    use std::sync::mpsc::channel;
    let (entered, started) = channel();
    let (release, native_release) = channel();
    let (cleanup_entered, cleanup) = channel();
    let (finish, cleanup_release) = channel();
    let cleaned = Arc::new(AtomicBool::new(false));
    let worker = BlockedAsr {
        entered,
        native_release,
        cleanup_entered,
        cleanup_release,
        cleaned: Arc::clone(&cleaned),
        first: true,
    };
    let (address, occupied, server) = transcription_server(worker, Duration::from_secs(30), 3);
    let mut client = transcription_client(address);
    started.recv_timeout(Duration::from_secs(10)).unwrap();
    client.shutdown(Shutdown::Write).unwrap();
    assert_socket_retained(&mut client, &occupied);
    let mut busy = String::new();
    transcription_client(address)
        .read_to_string(&mut busy)
        .unwrap();
    assert!(busy.starts_with("HTTP/1.1 503"), "{busy}");
    release.send(()).unwrap();
    cleanup.recv_timeout(Duration::from_secs(10)).unwrap();
    assert_socket_retained(&mut client, &occupied);
    assert!(!cleaned.load(Ordering::Acquire));
    finish.send(()).unwrap();
    let mut response = String::new();
    client.read_to_string(&mut response).unwrap();
    assert!(
        response.is_empty(),
        "cancelled child closes without a response"
    );
    assert!(cleaned.load(Ordering::Acquire));
    assert!(!occupied.load(Ordering::Acquire));
    transcription_client(address)
        .read_to_string(&mut response)
        .unwrap();
    assert!(response.starts_with("HTTP/1.1 200"), "{response}");
    assert!(response.contains("heard"));
    server.join().unwrap();
}

#[test]
fn transcription_deadline_uses_generation_timeout_response() {
    let (address, occupied, server) = transcription_server(FixtureAsr, Duration::ZERO, 1);
    let mut response = String::new();
    transcription_client(address)
        .read_to_string(&mut response)
        .unwrap();
    assert!(response.starts_with("HTTP/1.1 408"), "{response}");
    assert!(response.contains("generation_timeout"));
    assert!(!occupied.load(Ordering::Acquire));
    server.join().unwrap();
}

#[test]
fn transcription_deadline_precedes_disconnect_observation() {
    let mut observed = false;
    let mut cancelled = || {
        observed = true;
        true
    };
    let mut control = transcriptions::Control::new(Duration::ZERO, &mut cancelled);
    assert_eq!(control.check(), Err(transcriptions::Stop::DeadlineExceeded));
    assert!(!observed);
}
