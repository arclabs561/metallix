# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Model-free tests for the agent CLI end-to-end harness.

A fake upstream stands in for mx serve so the recording proxy, stream checks,
disconnect propagation, classification and environment isolation can be
checked without a model or the CLIs.
"""

from __future__ import annotations

import argparse
import http.client
import json
import os
import pathlib
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest import mock

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import e2e_agents

# Anthropic Messages stream shape (event order from the Messages streaming
# docs): a tool_use block, then message_stop.
MESSAGES_STREAM = [
    {"type": "message_start", "message": {"id": "msg_1", "role": "assistant"}},
    {
        "type": "content_block_start",
        "index": 0,
        "content_block": {
            "type": "tool_use",
            "id": "toolu_1",
            "name": "Read",
            "input": {},
        },
    },
    {"type": "content_block_stop", "index": 0},
    {"type": "message_delta", "delta": {"stop_reason": "tool_use"}},
    {"type": "message_stop"},
]


class Upstream:
    """A fake mx serve: canned streams, an error route, and an endless stream
    that records when its client goes away."""

    def __init__(self):
        self.disconnected = threading.Event()
        upstream = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args):
                pass

            def do_POST(self):
                self.rfile.read(int(self.headers["Content-Length"]))
                if self.path == "/v1/responses":
                    body = b'{"error":{"message":"bad"}}'
                    self.send_response(400)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body)))
                    self.send_header("X-Request-Id", "rid-400")
                    self.end_headers()
                    self.wfile.write(body)
                    return
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("X-Request-Id", "rid-" + self.path.rsplit("/", 1)[-1])
                self.send_header("Connection", "close")
                self.end_headers()
                if self.path == "/v1/messages":
                    for event in MESSAGES_STREAM:
                        self.wfile.write(f"data: {json.dumps(event)}\n\n".encode())
                elif self.path == "/v1/chat/completions":
                    self.wfile.write(b"data: {not json\n\n")  # No [DONE] either.
                else:  # /v1/endless
                    try:
                        while True:
                            self.wfile.write(
                                b'data: {"choices":[{"delta":{"content":"x"}}]}\n\n'
                            )
                            self.wfile.flush()
                            time.sleep(0.02)
                    except (BrokenPipeError, ConnectionResetError):
                        upstream.disconnected.set()

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.address = f"127.0.0.1:{self.server.server_address[1]}"

    def close(self):
        self.server.shutdown()
        self.server.server_close()


def post(url: str, path: str) -> tuple[int, bytes]:
    host, port = url.removeprefix("http://").split(":")
    connection = http.client.HTTPConnection(host, int(port), timeout=10)
    connection.request(
        "POST", path, b'{"max_tokens": 64}', {"Content-Type": "application/json"}
    )
    response = connection.getresponse()
    return response.status, response.read()


class Proxy(unittest.TestCase):
    def setUp(self):
        self.upstream = Upstream()
        self.tap = e2e_agents.Tap(self.upstream.address)
        self.addCleanup(self.upstream.close)
        self.addCleanup(self.tap.close)

    def test_a_stream_is_relayed_unchanged_and_recorded(self):
        status, body = post(self.tap.url, "/v1/messages")
        self.assertEqual(status, 200)
        expected = "".join(f"data: {json.dumps(e)}\n\n" for e in MESSAGES_STREAM)
        self.assertEqual(body.decode(), expected)
        (exchange,) = self.tap.since(0)
        self.assertEqual(exchange.request_id, "rid-messages")
        self.assertEqual((exchange.events, exchange.terminal), (5, True))
        self.assertEqual(exchange.tool_calls, ["Read"])
        self.assertEqual(e2e_agents.protocol_errors([exchange]), [])

    def test_http_errors_and_broken_streams_are_protocol_errors(self):
        post(self.tap.url, "/v1/responses")
        post(self.tap.url, "/v1/chat/completions")
        errors = e2e_agents.protocol_errors(self.tap.since(0))
        self.assertEqual(len(errors), 3, errors)
        self.assertTrue(errors[0].startswith("HTTP 400 on POST /v1/responses"))
        self.assertIn("undecodable event", errors[1])
        self.assertIn("stream ended without its final event", errors[2])

    def test_a_client_that_goes_away_closes_the_upstream_request(self):
        host, port = self.tap.url.removeprefix("http://").split(":")
        client = socket.create_connection((host, int(port)), timeout=10)
        client.sendall(
            b"POST /v1/endless HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\n"
            b"Content-Length: 2\r\n\r\n{}"
        )
        client.recv(256)
        exchange = self.tap.wait_for(lambda e: e.events >= 3, 5)
        self.assertIsNotNone(exchange)
        client.close()  # The CLI was killed.
        self.assertTrue(self.upstream.disconnected.wait(5), "upstream kept streaming")
        self.tap.wait_for(lambda e: e.ended is not None, 5)
        self.assertTrue(exchange.client_disconnected)


class ServerLog(unittest.TestCase):
    # A line logged by mx serve (built at 717f8b1) with METALLIX_LOG=info.
    LINE = (
        "[q] 2026-10-06T00:52:32.737463Z  INFO model-q ThreadId(04) http.request{request_id="
        "4f15fad44d968edbc16d01a234c0e616 trace_id=4f15fad44d968edbc16d01a234c0e616 "
        "route=/v1/chat/completions http.request.method=POST gen_ai.provider.name="
        '"metallix" gen_ai.operation.name="chat" gen_ai.request.model="q" '
        "gen_ai.usage.input_tokens=15 gen_ai.usage.output_tokens=8 mlx.active_bytes="
        "1206779904 mlx.peak_bytes=1214382208}: server::serving: request finished "
        "elapsed_ms=514.73225"
    )

    # A request whose client went away (mx at 0a4a873): no usage, an error type.
    FAILED = (
        "[qwen3-4b] 2026-10-06T14:24:14.163735Z  INFO model-qwen3-4b ThreadId(03) "
        "http.request{request_id=d750e63e960be3063192f4fd166844f0 trace_id="
        "d750e63e960be3063192f4fd166844f0 route=/v1/responses http.request.method=POST "
        'gen_ai.provider.name="metallix" gen_ai.operation.name="chat" '
        'gen_ai.request.model="qwen3-4b" error.type="response_failed" '
        "mlx.active_bytes=10737817332 mlx.peak_bytes=13750962196}: server::serving: "
        "request finished elapsed_ms=52850.556958"
    )

    def test_request_finished_lines_by_request_id(self):
        noise = "mx listening on http://127.0.0.1:8475; models=1"
        parsed = e2e_agents.server_requests(f"{noise}\n{self.LINE}\n{self.FAILED}")
        ok = parsed["4f15fad44d968edbc16d01a234c0e616"]
        self.assertEqual(
            (ok["route"], ok["input_tokens"], ok["output_tokens"], ok["error"]),
            ("/v1/chat/completions", 15, 8, None),
        )
        self.assertEqual(ok["finished"], 1791247952.737463)
        failed = parsed["d750e63e960be3063192f4fd166844f0"]
        self.assertEqual(
            (failed["output_tokens"], failed["error"]), (None, "response_failed")
        )


def exchange(target_events=20, disconnected=True, request_id="r1"):
    e = e2e_agents.Exchange(0, "POST", "/v1/messages", 0.0)
    e.request_body = json.dumps({"max_tokens": 1024})
    e.content_type = "text/event-stream"
    e.events, e.first_event, e.last_event = target_events, 1.0, 2.0
    e.client_disconnected = disconnected
    e.request_id = request_id
    e.ended = 100.0  # When the proxy saw the client go away.
    return e


class Cancel(unittest.TestCase):
    args = argparse.Namespace(cancel_grace=20)

    def classify(self, target, server):
        run = e2e_agents.Run("claude", "cancel", 0)
        e2e_agents.classify_cancel(run, target, 100.0, server, self.args)
        return run

    @staticmethod
    def finished(after, tokens=None):
        return {
            "r1": {"finished": 100.0 + after, "output_tokens": tokens, "error": None}
        }

    def test_generation_that_stops_soon_after_the_disconnect_passes(self):
        run = self.classify(exchange(), self.finished(1.1, tokens=40))
        self.assertEqual(run.outcome, "pass")
        self.assertIn("finished the request 1.1 s after the disconnect", run.detail)
        self.assertIn("generated 40 of 1024 tokens", run.detail)
        self.assertEqual(
            self.classify(exchange(), self.finished(1.1)).outcome, "protocol-error"
        )

    def test_late_finish_full_output_or_unfinished_is_a_protocol_error(self):
        late = self.classify(exchange(), self.finished(25.0))
        self.assertEqual(late.outcome, "protocol-error")
        full = self.classify(exchange(), self.finished(1.0, tokens=1024))
        self.assertEqual(full.outcome, "protocol-error")
        run = self.classify(exchange(), {})
        self.assertEqual(run.outcome, "protocol-error")
        self.assertIn("still unfinished", run.detail)

    def test_finish_before_cancel_or_normal_terminal_cannot_pass(self):
        self.assertEqual(
            self.classify(exchange(), self.finished(-1, 40)).outcome, "protocol-error"
        )
        target = exchange()
        target.terminal = True
        self.assertEqual(
            self.classify(target, self.finished(1, 40)).outcome, "protocol-error"
        )

    def test_no_stream_long_enough_is_a_model_miss(self):
        self.assertEqual(self.classify(None, {}).outcome, "model-miss")


class Tasks(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.repo = e2e_agents.make_fixture(pathlib.Path(directory.name))

    def run_check(self, task, answer, bodies=("{}", "{}")):
        run = e2e_agents.Run("pi", task, 0, exit_code=0, answer=answer)
        exchanges = []
        for seq, body in enumerate(bodies):
            e = e2e_agents.Exchange(seq, "POST", "/v1/chat/completions", 0.0)
            e.request_body = body
            exchanges.append(e)
        e2e_agents.check_task(task, run, self.repo, exchanges)
        return run

    def feedback(self, argument, result, result_id="call1"):
        return (
            "{}",
            json.dumps(
                {
                    "messages": [
                        {
                            "role": "assistant",
                            "tool_calls": [
                                {
                                    "id": "call1",
                                    "type": "function",
                                    "function": {
                                        "name": "bash",
                                        "arguments": json.dumps({"command": argument}),
                                    },
                                }
                            ],
                        },
                        {"role": "tool", "tool_call_id": result_id, "content": result},
                    ]
                }
            ),
        )

    def test_the_edit_passes_only_when_the_test_does_and_only_calc_changed(self):
        self.assertEqual(self.run_check("edit", "done").outcome, "model-miss")
        (self.repo / "calc.py").write_text("def add(a, b):\n    return a + b\n")
        self.assertEqual(self.run_check("edit", "done").outcome, "model-miss")
        marker = (
            e2e_agents.fixture_text(self.repo, "test_calc.py")
            .split("print('")[1]
            .split("')")[0]
        )
        bodies = self.feedback("python3 test_calc.py", marker)
        self.assertEqual(self.run_check("edit", "done", bodies).outcome, "pass")
        (self.repo / "notes.txt").write_text("changed\n")
        self.assertEqual(self.run_check("edit", "done").outcome, "model-miss")

    def test_the_tool_task_needs_the_file_sent_back_to_the_model(self):
        code = (
            e2e_agents.fixture_text(self.repo, "notes.txt")
            .split("launch code is ")[1]
            .strip()
            .rstrip(".")
        )
        # The model guessed the code without a tool result in a later request.
        self.assertEqual(self.run_check("tool", code).outcome, "model-miss")
        self.assertEqual(
            self.run_check("tool", code, ("{}", json.dumps({"r": code}))).outcome,
            "model-miss",
        )
        self.assertEqual(
            self.run_check(
                "tool", code, self.feedback("cat notes.txt", code, "wrong")
            ).outcome,
            "model-miss",
        )
        run = self.run_check(
            "tool", f"It is {code}.", self.feedback("cat notes.txt", code)
        )
        self.assertEqual(run.outcome, "pass")

    def test_all_protocols_require_a_matching_successful_result(self):
        for call, result, key in (
            (
                {
                    "type": "tool_use",
                    "id": "r",
                    "name": "Read",
                    "input": {"file_path": "notes.txt"},
                },
                {"type": "tool_result", "tool_use_id": "r", "content": "secret"},
                "messages",
            ),
            (
                {
                    "type": "function_call",
                    "call_id": "r",
                    "name": "exec_command",
                    "arguments": '{"cmd":"cat notes.txt"}',
                },
                {"type": "function_call_output", "call_id": "r", "output": "secret"},
                "input",
            ),
        ):
            items = (
                [
                    {"role": "assistant", "content": [call]},
                    {"role": "user", "content": [result]},
                ]
                if key == "messages"
                else [call, result]
            )
            e = e2e_agents.Exchange(
                0, "POST", "/v1/responses", 0, request_body=json.dumps({key: items})
            )
            self.assertTrue(e2e_agents.tool_feedback([e], "notes.txt", "secret"))
            result["is_error"] = True
            e.request_body = json.dumps({key: items})
            self.assertFalse(e2e_agents.tool_feedback([e], "notes.txt", "secret"))

    def test_nonzero_cli_exit_does_not_pass_completed_tool_task(self):
        code = (
            e2e_agents.fixture_text(self.repo, "notes.txt")
            .split("launch code is ")[1]
            .strip()
            .rstrip(".")
        )
        run = e2e_agents.Run("pi", "tool", 0, exit_code=1, answer=code)
        e = e2e_agents.Exchange(
            0,
            "POST",
            "/v1/chat/completions",
            0,
            request_body=self.feedback("cat notes.txt", code)[1],
        )
        e2e_agents.check_task("tool", run, self.repo, [e])
        self.assertEqual(run.outcome, "model-miss")

    def test_no_model_request_at_all_is_a_cli_error(self):
        run = self.run_check("plain", "PONG", bodies=())
        self.assertEqual(run.outcome, "cli-error")


@unittest.skipUnless(sys.platform == "darwin", "requires macOS sandbox-exec")
class Confinement(unittest.TestCase):
    def test_real_sandbox_allows_only_tap_and_denies_protected_reads_in_child(self):
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            listener.listen()
            e2e_agents.validate_sandbox(f"http://127.0.0.1:{listener.getsockname()[1]}")

    def test_bad_endpoint_and_unavailable_guard_fail_closed(self):
        for endpoint in (
            "https://127.0.0.1:1",
            "http://example.com:1",
            "http://127.0.0.1:1/v1",
            "http://user@127.0.0.1:1",
        ):
            with self.assertRaises(ValueError):
                e2e_agents.sandbox_command(["true"], endpoint)
        with (
            mock.patch.object(pathlib.Path, "is_file", return_value=False),
            self.assertRaises(RuntimeError),
        ):
            e2e_agents.sandbox_command(["true"], "http://127.0.0.1:1")

    def test_deadline_and_sigterm_reap_client_and_server(self):
        # Fake processes replace both native CLI and MLX, retaining production
        # process groups, confinement, main-loop signal handlers and finally.
        program = r"""
import argparse, pathlib, subprocess, sys, time, os, signal
sys.path.insert(0, sys.argv[1])
import e2e_agents as e
root = pathlib.Path(sys.argv[2]); mode = sys.argv[3]
class Session:
    def __init__(self, argv, address, args):
        self.aborted = None; self.logs = []; self.traces = []; self.baselines = []; self.memory = []; self.launches = []; self.shutdowns = []; self.server = None
        self.log_path = root / 'server.log'; self.log_path.write_text('')
    def ensure_started(self):
        spec = e.bench_load.ServerSpec('fake', 'chat', [sys.executable, '-c', 'import time; time.sleep(60)'], {}, {})
        self.server = e.bench_load.ManagedServer(spec, '127.0.0.1:1', root / 'managed.log')
        (root / 'server.pid').write_text(str(self.server.process.pid))
    def stop(self):
        if self.server:
            self.server.stop()
e.Session = Session
e.FIXED_BINARIES['pi'] = sys.executable
# Hold the empty PID-file publication window open to reproduce the original race.
# Readiness is published only after that write has closed.
def command(*args):
    return [sys.executable, '-c', "import os,time,pathlib; pidpath=pathlib.Path(" + repr(str(root/'client.pid')) + "); handle=pidpath.open('w'); time.sleep(.1); handle.write(str(os.getpid())); handle.close(); (pidpath.parent/'client.ready').touch(); time.sleep(60)"], {'PATH': e.MINIMAL_PATH, 'HOME': str(root)}
e.cli_command = command
e.tool_versions = lambda args: {}
e.bench_system.system_info = lambda: {}
sys.argv = ['e2e', '--model', str(root), '--clis', 'pi', '--tasks', 'plain', '--pi', sys.executable, '--log-dir', str(root/'logs'), '--session-timeout', '1' if mode == 'deadline' else '30', '--run']
raise SystemExit(e.main())
"""
        for mode in ("deadline", "sigterm"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as root:
                process = subprocess.Popen(
                    [
                        sys.executable,
                        "-c",
                        program,
                        str(pathlib.Path(e2e_agents.__file__).parent),
                        root,
                        mode,
                    ],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                try:
                    deadline = time.monotonic() + 10
                    while (
                        not (pathlib.Path(root) / "client.ready").exists()
                        and time.monotonic() < deadline
                        and process.poll() is None
                    ):
                        time.sleep(0.02)
                    self.assertTrue((pathlib.Path(root) / "client.ready").exists())
                    if mode == "sigterm":
                        process.terminate()
                    stdout, stderr = process.communicate(timeout=15)
                    self.assertEqual(process.returncode, 1, stderr + stdout)
                    for name in ("client", "server"):
                        pid = int((pathlib.Path(root) / f"{name}.pid").read_text())
                        with self.assertRaises(ProcessLookupError):
                            os.kill(pid, 0)
                finally:
                    if process.poll() is None:
                        process.kill()
                    process.wait(timeout=5)


@unittest.skipUnless(sys.platform == "darwin", "requires macOS sandbox-exec")
class SpawnSignals(unittest.TestCase):
    def test_child_term_finalizes_after_parent_registration(self):
        child = r"""
import json, pathlib, signal, sys, time
root = pathlib.Path(sys.argv[1])
def stop(*_):
    (root / 'finished').write_text('graceful')
    raise SystemExit(0)
signal.signal(signal.SIGTERM, stop)
(root / 'ready').write_text(json.dumps(sorted(signal.pthread_sigmask(signal.SIG_BLOCK, set()))))
while True:
    time.sleep(.05)
"""
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            delivered = []
            registered = False
            previous_handler = signal.signal(
                signal.SIGTERM, lambda *_: delivered.append(registered)
            )
            previous_mask = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGUSR1})
            process = None
            try:
                command = [sys.executable, "-I", "-S", "-c", child, str(root)]
                command = e2e_agents.unmasked_command(command)
                command = e2e_agents.sandbox_command(command, "http://127.0.0.1:1")
                with e2e_agents.defer_stop_signals():
                    process = subprocess.Popen(command, start_new_session=True)
                    os.kill(os.getpid(), signal.SIGTERM)
                    self.assertEqual(delivered, [])
                    registered = True
                self.assertEqual(delivered, [True])
                deadline = time.monotonic() + 5
                while not (root / "ready").exists() and time.monotonic() < deadline:
                    time.sleep(0.01)
                self.assertTrue((root / "ready").exists())
                blocked = json.loads((root / "ready").read_text())
                self.assertFalse(
                    set(blocked) & {signal.SIGTERM, signal.SIGINT, signal.SIGALRM}
                )
                self.assertIn(signal.SIGUSR1, blocked)
                process.terminate()
                self.assertEqual(process.wait(timeout=3), 0)
                self.assertEqual((root / "finished").read_text(), "graceful")
                with self.assertRaises(ProcessLookupError):
                    os.kill(process.pid, 0)
            finally:
                if process is not None:
                    if process.poll() is None:
                        process.kill()
                    process.wait(timeout=5)
                signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)
                signal.signal(signal.SIGTERM, previous_handler)

    def test_session_term_closes_timelines_without_escalation(self):
        server = r"""
import json, pathlib, signal, sys, time
root = pathlib.Path(sys.argv[1])
trace = pathlib.Path(sys.argv[sys.argv.index('--trace-out') + 1])
child = trace.with_name(trace.stem + '.test' + trace.suffix)
for path in (trace, child):
    path.write_text('[')
def stop(*_):
    for path in (trace, child):
        with path.open('a') as output:
            output.write(json.dumps({'ph': 'i', 'name': 'finished'}) + ']')
    (root / 'finished').write_text('graceful')
    raise SystemExit(0)
signal.signal(signal.SIGTERM, stop)
(root / 'ready').write_text(json.dumps(sorted(signal.pthread_sigmask(signal.SIG_BLOCK, set()))))
while True:
    time.sleep(.05)
"""
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            args = argparse.Namespace(log_dir=root, abort_gpu_gib=4)
            original = [sys.executable, "-I", "-S", "-c", server, str(root)]
            session = e2e_agents.Session(original, "127.0.0.1:1", args)

            def ready(_server, _timeout):
                deadline = time.monotonic() + 5
                while not (root / "ready").exists() and time.monotonic() < deadline:
                    time.sleep(0.01)
                self.assertTrue((root / "ready").exists())

            with (
                mock.patch.object(e2e_agents.bench_system, "probe", return_value={}),
                mock.patch.object(e2e_agents.bench_system, "Sampler"),
                mock.patch.object(
                    e2e_agents.bench_load.ManagedServer, "wait_ready", ready
                ),
            ):
                try:
                    session.ensure_started()
                    self.assertEqual(session.spec.argv, original)
                    blocked = json.loads((root / "ready").read_text())
                    self.assertFalse(
                        set(blocked) & {signal.SIGTERM, signal.SIGINT, signal.SIGALRM}
                    )
                    process = session.server.process
                    session.stop()
                    self.assertEqual((root / "finished").read_text(), "graceful")
                    self.assertEqual(len(session.shutdowns), 1)
                    shutdown = session.shutdowns[0]
                    self.assertTrue(shutdown["term_sent"])
                    self.assertFalse(shutdown["kill_sent"])
                    self.assertFalse(shutdown["grace_expired"])
                    self.assertEqual(shutdown["returncode"], 0)
                    self.assertTrue(
                        e2e_agents.validate_timelines(session.traces, "test")["valid"]
                    )
                    with self.assertRaises(ProcessLookupError):
                        os.kill(process.pid, 0)
                finally:
                    if session.server is not None:
                        if session.server.process.poll() is None:
                            session.server.process.kill()
                        session.stop()


class Timelines(unittest.TestCase):
    def test_missing_or_unfinished_child_is_invalid_without_repair(self):
        with tempfile.TemporaryDirectory() as directory:
            front = pathlib.Path(directory) / "trace.json"
            child = e2e_agents.mx_spans.child_path(front, "test")
            event = {"ph": "i", "name": "finished"}
            front.write_text(json.dumps([event]))
            self.assertFalse(
                e2e_agents.validate_timelines([str(front)], "test")["valid"]
            )
            truncated = "[" + json.dumps(event)
            child.write_text(truncated)
            report = e2e_agents.validate_timelines([str(front)], "test")
            self.assertFalse(report["valid"])
            self.assertTrue(report["files"][0]["valid"])
            self.assertFalse(report["files"][1]["valid"])
            self.assertEqual(child.read_text(), truncated)
            child.write_text(truncated + "]")
            self.assertTrue(
                e2e_agents.validate_timelines([str(front)], "test")["valid"]
            )
            self.assertFalse(e2e_agents.validate_timelines([], "test")["valid"])


class Isolation(unittest.TestCase):
    def test_default_paths_use_account_home_with_relocated_launch_home(self):
        program = r"""
import json, pathlib, pwd, sys
from types import SimpleNamespace
from unittest import mock
sys.path.insert(0, sys.argv[1])
with mock.patch.object(pwd, 'getpwuid', return_value=SimpleNamespace(pw_dir=sys.argv[2])):
    import e2e_agents
with mock.patch.object(pathlib.Path, 'is_file', return_value=True):
    command = e2e_agents.sandbox_command(['true'], 'http://127.0.0.1:1')
print(json.dumps({'pi': e2e_agents.FIXED_BINARIES['pi'], 'profile': command[2]}))
"""
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            account_home, launch_home = root / "account", root / "launch"
            account_home.mkdir()
            launch_home.mkdir()
            result = subprocess.run(
                [
                    sys.executable,
                    "-I",
                    "-S",
                    "-c",
                    program,
                    str(pathlib.Path(__file__).resolve().parent),
                    str(account_home),
                ],
                env=os.environ | {"HOME": str(launch_home)},
                capture_output=True,
                text=True,
                timeout=5,
                check=True,
            )
            defaults = json.loads(result.stdout)
            self.assertEqual(defaults["pi"], str(account_home / ".node_modules/bin/pi"))
            self.assertIn(str(account_home / "Private"), defaults["profile"])
            self.assertNotIn(str(launch_home / "Private"), defaults["profile"])

    def test_runs_see_no_inherited_credentials_or_config(self):
        home = pathlib.Path(tempfile.mkdtemp())
        args = argparse.Namespace(
            claude="claude",
            codex="codex",
            pi="pi",
            context_tokens=16384,
            max_output_tokens=512,
        )
        secrets = {
            "ANTHROPIC_API_KEY": "sk-real",
            "OPENAI_API_KEY": "sk-real",
            "CODEX_HOME": "/x",
        }
        saved = {k: os.environ.get(k) for k in secrets}
        os.environ.update(secrets)
        try:
            for cli in e2e_agents.CLIS:
                _, env = e2e_agents.cli_command(
                    cli, "tool", "http://127.0.0.1:1", "m", home, home, args
                )
                self.assertNotIn("sk-real", json.dumps(env), cli)
                self.assertEqual(env["HOME"], str(home), cli)
                self.assertTrue(all(not v.startswith("/x") for v in env.values()), cli)
        finally:
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
        self.assertEqual(
            json.loads((home / ".pi/agent/models.json").read_text())["providers"][
                "metallix"
            ]["apiKey"],
            e2e_agents.DUMMY_KEY,
        )

    def test_shell_wrappers_on_path_are_skipped(self):
        directory = pathlib.Path(tempfile.mkdtemp())
        wrapper, real = directory / "wrap", directory / "real"
        wrapper.mkdir()
        real.mkdir()
        (wrapper / "codex").write_text(
            '#!/usr/bin/env bash\nexec agent-env codex "$@"\n'
        )
        (real / "codex").write_bytes(b"\xcf\xfa\xed\xfe binary")
        for path in (wrapper / "codex", real / "codex"):
            path.chmod(0o755)
        found = e2e_agents.real_binary("codex", f"{wrapper}{os.pathsep}{real}")
        self.assertEqual(found, str(real / "codex"))


if __name__ == "__main__":
    unittest.main()
