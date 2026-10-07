#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Opt-in end-to-end runs of real agent CLIs (Claude Code, Codex, pi) on mx serve.

Starts one `mx serve` (GPU-capped like the benchmarks) and puts a recording
HTTP proxy in front of it. Each CLI is pointed at the proxy through its own
documented custom-endpoint settings, with a throwaway HOME and config
directory per run and a placeholder API key. These settings reduce config
discovery; they are not an OS network or filesystem security boundary. Each CLI starts from an empty
environment (PATH, a temp HOME and TMPDIR, LANG, TERM, then its endpoint
settings), and the harness runs the tool's own binary, skipping any shell
wrapper earlier on PATH: a wrapper may inject credentials or user config.
Every run works in a fresh git repository fixture:

- plain: a plain answer, no tools;
- tool: read a file and answer a question about it;
- edit: fix one line and run the test command;
- cancel: stream a long answer, kill the CLI mid-stream, and check from the
  server's own log that generation stopped.

Each run is classified as `pass`, `model-miss` (the protocol worked but the
model did not do the task, which a small model may not) or `protocol-error`
(an HTTP 4xx/5xx on a /v1 route, a stream that does not parse or does not
end, or generation that outlives a cancel). `cli-error` means the CLI exited
before sending any model request, a setup problem rather than a server one.
Outcomes are checked from files, exit codes and the recorded wire traffic,
never from exact model text.

    uv run scripts/e2e_agents.py --mx target/release/mx --model DIR \\
        --model-id qwen3-4b --run --json e2e.json
"""

from __future__ import annotations

import argparse
import codecs
import http.client
import json
import math
import os
import pwd
import re
import secrets
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

import bench_load
import bench_serve
import bench_system
import mx_spans

MODEL_ROUTES = ("/v1/messages", "/v1/responses", "/v1/chat/completions")
CLIS = ("claude", "codex", "pi")
TASKS = ("plain", "tool", "edit", "cancel")
DUMMY_KEY = "local-placeholder-not-a-credential"
# The cancel task waits for this many streamed events before killing the CLI.
CANCEL_AFTER_EVENTS = 12


# ---------------------------------------------------------------------------
# Wire capture


@dataclass
class Exchange:
    """One request through the proxy and what came back."""

    seq: int
    method: str
    path: str
    started: float
    request_body: str = ""
    status: int | None = None
    request_id: str | None = None
    content_type: str = ""
    response_body: str = ""
    events: int = 0
    first_event: float | None = None
    last_event: float | None = None
    stream_errors: list[str] = field(default_factory=list)
    terminal: bool = False
    tool_calls: list[str] = field(default_factory=list)
    client_disconnected: bool = False
    ended: float | None = None

    @property
    def streamed(self) -> bool:
        return self.content_type.startswith("text/event-stream")


def sse_events(text: str) -> tuple[list[str], str]:
    """Complete SSE events' data payloads, and the unfinished tail."""
    payloads = []
    *events, tail = re.split(r"\r?\n\r?\n", text)
    for event in events:
        data = [
            line[5:].lstrip() for line in event.splitlines() if line.startswith("data:")
        ]
        if data:
            payloads.append("\n".join(data))
    return payloads, tail


def check_event(path: str, payload: str, exchange: Exchange) -> None:
    """Record one streamed event: parse failures, errors, tool calls, the end."""
    if payload == "[DONE]":
        if path == "/v1/chat/completions":
            exchange.terminal = True
        else:
            exchange.stream_errors.append(f"unexpected [DONE] on {path}")
        return
    try:
        event = json.loads(payload)
    except json.JSONDecodeError:
        exchange.stream_errors.append(f"undecodable event: {payload[:80]}")
        return
    if not isinstance(event, dict):
        exchange.stream_errors.append(f"event is not an object: {payload[:80]}")
        return
    kind = event.get("type")
    if path == "/v1/messages":
        if kind == "message_stop":
            exchange.terminal = True
        elif kind == "error":
            exchange.stream_errors.append(f"error event: {payload[:200]}")
        elif kind == "content_block_start":
            block = event.get("content_block") or {}
            if block.get("type") == "tool_use":
                exchange.tool_calls.append(str(block.get("name")))
    elif path == "/v1/responses":
        if kind in ("response.completed", "response.incomplete"):
            exchange.terminal = True
        elif kind in ("response.failed", "error"):
            exchange.stream_errors.append(f"{kind}: {payload[:200]}")
        elif kind == "response.output_item.added":
            item = event.get("item") or {}
            if item.get("type") == "function_call":
                exchange.tool_calls.append(str(item.get("name")))
    elif path == "/v1/chat/completions":
        if "error" in event:
            exchange.stream_errors.append(f"error: {payload[:200]}")
        for choice in event.get("choices") or []:
            for call in (choice.get("delta") or {}).get("tool_calls") or []:
                name = (call.get("function") or {}).get("name")
                if name:
                    exchange.tool_calls.append(name)


class Tap:
    """A recording reverse proxy in front of mx serve.

    It streams responses through as they arrive, so a CLI sees the server's
    own streaming behavior, and when the CLI disconnects it closes the
    upstream connection, as a direct client would.
    """

    def __init__(self, upstream: str):
        self.upstream = upstream
        self.exchanges: list[Exchange] = []
        self.lock = threading.Lock()
        self.changed = threading.Condition(self.lock)
        tap = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.0"  # Close-delimited bodies.

            def log_message(self, *args) -> None:
                pass

            def do_GET(self) -> None:
                tap.forward(self, "GET")

            def do_POST(self) -> None:
                tap.forward(self, "POST")

            def do_HEAD(self) -> None:
                tap.forward(self, "HEAD")

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server.server_address[1]}"

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()

    def mark(self) -> int:
        with self.lock:
            return len(self.exchanges)

    def since(self, mark: int) -> list[Exchange]:
        with self.lock:
            return list(self.exchanges[mark:])

    def wait_for(
        self, predicate, timeout: float, give_up=lambda: False
    ) -> Exchange | None:
        """The first exchange matching `predicate`, or None after `timeout`
        seconds or as soon as `give_up()` is true."""
        deadline = time.monotonic() + timeout
        with self.changed:
            while True:
                for exchange in self.exchanges:
                    if predicate(exchange):
                        return exchange
                remaining = deadline - time.monotonic()
                if remaining <= 0 or give_up():
                    return None
                self.changed.wait(min(remaining, 0.5))

    def notify(self) -> None:
        with self.changed:
            self.changed.notify_all()

    def forward(self, handler: BaseHTTPRequestHandler, method: str) -> None:
        body = b""
        if "Content-Length" in handler.headers:
            body = handler.rfile.read(int(handler.headers["Content-Length"]))
        with self.lock:
            exchange = Exchange(len(self.exchanges), method, handler.path, time.time())
            exchange.request_body = body.decode(errors="replace")
            self.exchanges.append(exchange)
        host, port = self.upstream.rsplit(":", 1)
        upstream = http.client.HTTPConnection(host, int(port), timeout=900)
        headers = {
            k: v
            for k, v in handler.headers.items()
            if k.lower()
            not in ("host", "connection", "accept-encoding", "content-length")
        }
        headers["Accept-Encoding"] = "identity"
        try:
            upstream.request(method, handler.path, body or None, headers)
            response = upstream.getresponse()
        except (OSError, http.client.HTTPException) as error:
            exchange.status = 599
            exchange.stream_errors.append(f"upstream: {error}")
            exchange.ended = time.time()
            upstream.close()
            try:
                handler.send_error(502)
            except OSError:
                pass
            self.notify()
            return
        exchange.status = response.status
        exchange.request_id = response.getheader("X-Request-Id")
        exchange.content_type = response.getheader("Content-Type") or ""
        try:
            handler.send_response(response.status)
            for name, value in response.getheaders():
                if name.lower() not in (
                    "transfer-encoding",
                    "content-length",
                    "connection",
                ):
                    handler.send_header(name, value)
            handler.send_header("Connection", "close")
            handler.end_headers()
            buffer = ""
            decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
            while True:
                chunk = response.read1(65536) if method != "HEAD" else b""
                if not chunk:
                    break
                text = decoder.decode(chunk)
                if exchange.streamed:
                    payloads, buffer = sse_events(buffer + text)
                    now = time.time()
                    for payload in payloads:
                        exchange.events += 1
                        exchange.first_event = exchange.first_event or now
                        exchange.last_event = now
                        check_event(handler.path.split("?")[0], payload, exchange)
                else:
                    exchange.response_body += text
                handler.wfile.write(chunk)
                handler.wfile.flush()
                self.notify()
        except (BrokenPipeError, ConnectionResetError):
            exchange.client_disconnected = True
        finally:
            # Closing the upstream socket is how a vanished client looks to mx.
            upstream.close()
            exchange.ended = time.time()
            self.notify()


# ---------------------------------------------------------------------------
# Server log


REQUEST_FINISHED = re.compile(r"request finished")


def server_requests(log_text: str) -> dict[str, dict]:
    """Per request id, what mx logged when the model child finished it."""
    out: dict[str, dict] = {}
    for line in log_text.splitlines():
        # The model child's line carries the operation; a failed response (a
        # client that went away) has no usage, only an error type.
        if not REQUEST_FINISHED.search(line) or "gen_ai.operation.name" not in line:
            continue
        fields = dict(re.findall(r"([\w.]+)=(\"[^\"]*\"|[^\s}]+)", line))
        rid = fields.get("request_id")
        if not rid:
            continue
        stamp = re.search(r"(\d{4}-\d\d-\d\dT[\d:.]+)Z", line)
        usage_in = fields.get("gen_ai.usage.input_tokens")
        usage_out = fields.get("gen_ai.usage.output_tokens")
        out[rid] = {
            "route": fields.get("route"),
            "input_tokens": int(usage_in) if usage_in else None,
            "output_tokens": int(usage_out) if usage_out else None,
            "error": (fields.get("error.type") or "").strip('"') or None,
            "elapsed_ms": float(fields.get("elapsed_ms", 0)),
            "finished_utc": stamp.group(1) if stamp else None,
            "finished": (
                datetime.fromisoformat(stamp.group(1)).replace(tzinfo=UTC).timestamp()
                if stamp
                else None
            ),
        }
    return out


# ---------------------------------------------------------------------------
# Fixture and CLI adapters


def make_fixture(root: Path) -> Path:
    repo = root / "repo"
    repo.mkdir()
    (repo / "notes.txt").write_text(
        "Project notes.\nThe launch code is " + secrets.token_hex(16) + ".\n"
    )
    (repo / "calc.py").write_text("def add(a, b):\n    return a - b\n")
    (repo / ".gitignore").write_text("__pycache__/\n")
    (repo / "test_calc.py").write_text(
        "from calc import add\n\nassert add(2, 3) == 5, add(2, 3)\n"
        + f"print('ok-{secrets.token_hex(16)}')\n"
    )
    git = ["git", "-C", str(repo)]
    subprocess.run([*git, "init", "-q"], check=True)
    subprocess.run(
        [*git, "-c", "user.name=e2e", "-c", "user.email=e2e@localhost", "add", "."],
        check=True,
    )
    subprocess.run(
        [
            *git, "-c", "user.name=e2e", "-c", "user.email=e2e@localhost",
            "commit", "-q", "-m", "fixture",
        ],
        check=True,
    )  # fmt: skip
    (root / "fixture.json").write_text(
        json.dumps(
            {name: (repo / name).read_text() for name in ("notes.txt", "test_calc.py")}
        )
    )
    return repo


PROMPTS = {
    "plain": "Reply with the single word PONG and nothing else.",
    "tool": "Read the file notes.txt in the current directory and tell me the launch code it contains.",
    "edit": (
        "calc.py has a bug: add should return a + b. Fix that one line in calc.py, "
        "then run `python3 test_calc.py` to check it prints ok."
    ),
    "cancel": "Write a long story about a lighthouse keeper, at least 1500 words.",
}

# Tools each task needs, by CLI; fewer tools keep the prompt inside a small context.
TOOLS = {
    "claude": {"plain": "", "tool": "Read", "edit": "Read,Edit,Bash", "cancel": ""},
    "pi": {"plain": None, "tool": "read", "edit": "read,edit,bash", "cancel": None},
}


def real_binary(name: str, path: str | None = None) -> str | None:
    """The tool's own executable on PATH, skipping shell-script wrappers.

    A wrapper earlier on PATH may set up credentials or user configuration
    for the tool, which these runs must not use.
    """
    for directory in (path or os.environ.get("PATH", "")).split(os.pathsep):
        candidate = Path(directory) / name
        if not (candidate.is_file() and os.access(candidate, os.X_OK)):
            continue
        with candidate.open("rb") as handle:
            first = handle.readline(200)
        if first.startswith(b"#!") and re.search(rb"\b(ba|z)?sh\b", first):
            continue
        return str(candidate)
    return None


# Keep native binaries and protected reads tied to the account, even with a temporary HOME.
ACCOUNT_HOME = Path(pwd.getpwuid(os.getuid()).pw_dir)
FIXED_BINARIES = {
    "claude": "/opt/homebrew/bin/claude",
    "codex": "/opt/homebrew/bin/codex",
    "pi": str(ACCOUNT_HOME / ".node_modules/bin/pi"),
}
MINIMAL_PATH = "/opt/homebrew/bin:/usr/bin:/bin"
PRIVATE_ROOT = ACCOUNT_HOME / "Private"


def sandbox_command(
    argv: list[str], endpoint: str, protected: Path = PRIVATE_ROOT
) -> list[str]:
    """Constrain the executable and descendants to one numeric loopback endpoint."""
    url = urlsplit(endpoint)
    if (
        url.scheme != "http"
        or url.hostname != "127.0.0.1"
        or not url.port
        or url.username
        or url.password
        or url.path
        or url.query
        or url.fragment
    ):
        raise ValueError("client endpoint must be http://127.0.0.1:<port>")
    if not Path("/usr/bin/sandbox-exec").is_file():
        raise RuntimeError("sandbox-exec unavailable; refusing unconfined client")
    roots = {str(PRIVATE_ROOT), str(PRIVATE_ROOT.resolve()), str(protected.resolve())}
    profile = "(version 1) (allow default) (deny network*) "
    profile += f'(allow network-outbound (remote tcp "localhost:{url.port}")) '
    profile += " ".join(
        f"(deny file-read* (subpath {json.dumps(root)}))" for root in roots
    )
    return ["/usr/bin/sandbox-exec", "-p", profile, *argv]


def validate_sandbox(endpoint: str) -> None:
    """Require inherited allowed/denied socket and protected-read checks before MLX."""
    port = urlsplit(endpoint).port
    # A second live local listener distinguishes policy denial from refusal.
    with (
        socket.socket() as blocked,
        tempfile.TemporaryDirectory(prefix="e2e-guard-") as root,
    ):
        blocked.bind(("127.0.0.1", 0))
        blocked.listen()
        protected = Path(root) / "protected"
        protected.mkdir()
        (protected / "sentinel").write_text("synthetic guard fixture")
        probe = r"""
import errno, socket, sys
port, denied = map(int, sys.argv[1:3])
with socket.create_connection(('127.0.0.1', port), timeout=2): pass
for address in [('127.0.0.1', denied), ('203.0.113.1', 443)]:
    try:
        connection = socket.create_connection(address, timeout=2)
    except OSError as error:
        assert error.errno in (errno.EPERM, errno.EACCES), error
    else:
        connection.close()
        raise AssertionError('unexpected network permission')
try:
    open(sys.argv[3], 'rb')
except OSError as error:
    assert error.errno in (errno.EPERM, errno.EACCES), error
else:
    raise AssertionError('unexpected protected read permission')
print('GUARD_OK')
"""
        # The sandboxed parent spawns a child: the assertions verify inheritance.
        child = [
            sys.executable,
            "-c",
            probe,
            str(port),
            str(blocked.getsockname()[1]),
            str(protected / "sentinel"),
        ]
        launcher = "import subprocess,sys; r=subprocess.run(sys.argv[1:],check=True,capture_output=True,text=True,timeout=10); print(r.stdout,end='')"
        command = sandbox_command(
            [sys.executable, "-c", launcher, *child], endpoint, protected
        )
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=15,
            check=True,
            env={"PATH": MINIMAL_PATH, "HOME": root},
        )
        if result.stdout.strip() != "GUARD_OK":
            raise RuntimeError("sandbox validation did not complete")


UNMASK_EXEC = (
    "import os,signal,sys; "
    "signal.pthread_sigmask(signal.SIG_UNBLOCK, "
    "{signal.SIGTERM,signal.SIGINT,signal.SIGALRM}); "
    "os.execvpe(sys.argv[1],sys.argv[1:],os.environ)"
)


def unmasked_command(argv: list[str]) -> list[str]:
    """Restore child stop delivery after exec, without threaded preexec_fn."""
    return [sys.executable, "-I", "-S", "-c", UNMASK_EXEC, *argv]


@contextmanager
def defer_stop_signals():
    """Register a newly spawned process before delivering shutdown signals."""
    previous = signal.pthread_sigmask(
        signal.SIG_BLOCK, {signal.SIGTERM, signal.SIGINT, signal.SIGALRM}
    )
    try:
        yield
    finally:
        signal.pthread_sigmask(signal.SIG_SETMASK, previous)


class RunStopped(Exception):
    """The bounded session deadline or a termination signal was reached."""


def stop_signal(signum, _frame) -> None:
    signal.setitimer(signal.ITIMER_REAL, 0)
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    raise RunStopped(f"session interrupted by signal {signum}")


def base_env(home: Path) -> dict[str, str]:
    """A minimal environment: nothing inherited that could carry a credential,
    a proxy setting or a user config path."""
    return {
        "PATH": MINIMAL_PATH,
        "HOME": str(home),
        "TMPDIR": str(home / "tmp"),
        "LANG": "en_US.UTF-8",
        "TERM": "dumb",
        "NO_COLOR": "1",
        # Test runs inside a task must not leave stale bytecode for later ones.
        "PYTHONDONTWRITEBYTECODE": "1",
    }


def toml_string(value: str) -> str:
    return json.dumps(value)


def cli_command(
    cli: str, task: str, base: str, model_id: str, home: Path, repo: Path, args
) -> tuple[list[str], dict[str, str]]:
    env = base_env(home)
    prompt = PROMPTS[task]
    if cli == "claude":
        # Claude Code's documented gateway settings: ANTHROPIC_BASE_URL with an
        # API key; --bare reads no keychain, OAuth, hooks or CLAUDE.md.
        env |= {
            "ANTHROPIC_BASE_URL": base,
            "ANTHROPIC_API_KEY": DUMMY_KEY,
            "CLAUDE_CONFIG_DIR": str(home / ".claude"),
            "ANTHROPIC_DEFAULT_HAIKU_MODEL": model_id,
            "ANTHROPIC_SMALL_FAST_MODEL": model_id,
            "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
            "DISABLE_AUTOUPDATER": "1",
            "DISABLE_TELEMETRY": "1",
            "DISABLE_ERROR_REPORTING": "1",
            "CLAUDE_CODE_MAX_OUTPUT_TOKENS": str(args.max_output_tokens),
        }
        argv = [
            args.claude, "-p", "--bare", "--model", model_id,
            "--output-format", "stream-json", "--verbose",
            "--tools", TOOLS["claude"][task],
            "--dangerously-skip-permissions", "--max-turns", "8",
            "--disable-slash-commands", "--strict-mcp-config",
            "--no-session-persistence", prompt,
        ]  # fmt: skip
        return argv, env
    if cli == "codex":
        env["CODEX_HOME"] = str(home / ".codex")
        (home / ".codex").mkdir(parents=True, exist_ok=True)  # Codex requires it.
        provider = "metallix_e2e"
        configs = [
            f"model_provider={toml_string(provider)}",
            f"model={toml_string(model_id)}",
            f"model_context_window={args.context_tokens}",
            f"model_max_output_tokens={args.max_output_tokens}",
            'web_search="disabled"',
            "analytics.enabled=false",
            'history.persistence="none"',
            "check_for_update_on_startup=false",
            # Off: features that reach other services or add non-function tool
            # types (multi_agent sends a "namespace" tool). Single-agent coding
            # with the shell, apply_patch and view_image tools stays on.
            *(
                f"features.{name}=false"
                for name in (
                    "multi_agent",
                    "goals",
                    "apps",
                    "plugins",
                    "remote_plugin",
                    "hooks",
                    "shell_snapshot",
                    "image_generation",
                    "browser_use",
                    "computer_use",
                    "daemon_auto_start",
                )
            ),
            (
                f"model_providers.{provider}={{name={toml_string('metallix e2e')},"
                f'base_url={toml_string(base + "/v1")},wire_api="responses",'
                "requires_openai_auth=false,supports_websockets=false,"
                "request_max_retries=0,stream_max_retries=0}"
            ),
        ]
        sandbox = "workspace-write" if task == "edit" else "read-only"
        argv = [
            args.codex, "exec", "--ignore-user-config", "--ephemeral",
            "--skip-git-repo-check", "--sandbox", sandbox, "--json", "--cd", str(repo),
        ]  # fmt: skip
        for config in configs:
            argv += ["-c", config]
        return argv + [prompt], env
    if cli == "pi":
        agent_dir = home / ".pi" / "agent"
        agent_dir.mkdir(parents=True, exist_ok=True)
        # pi's documented custom provider (docs/models.md, minimal example).
        (agent_dir / "models.json").write_text(
            json.dumps(
                {
                    "providers": {
                        "metallix": {
                            "baseUrl": base + "/v1",
                            "api": "openai-completions",
                            "apiKey": DUMMY_KEY,
                            "models": [
                                {
                                    "id": model_id,
                                    "contextWindow": args.context_tokens,
                                    "maxTokens": args.max_output_tokens,
                                }
                            ],
                        }
                    }
                },
                indent=1,
            )
        )
        env |= {
            "PI_CODING_AGENT_DIR": str(agent_dir),
            "PI_OFFLINE": "1",
            "PI_TELEMETRY": "0",
            "PI_SKIP_VERSION_CHECK": "1",
        }
        tools = TOOLS["pi"][task]
        argv = [
            args.pi, "-p", "--mode", "json", "--no-session",
            "--provider", "metallix", "--model", model_id, "--thinking", "off",
            "--no-extensions", "--no-skills", "--no-prompt-templates",
            "--no-themes", "--no-context-files",
        ]  # fmt: skip
        argv += ["--tools", tools] if tools else ["--no-tools"]
        return argv + [prompt], env
    raise ValueError(cli)


def final_text(cli: str, stdout: str) -> str | None:
    """The CLI's last assistant answer, from its JSON event stream."""
    events = []
    for line in stdout.splitlines():
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    text = None
    for event in events:
        if not isinstance(event, dict):
            continue
        if cli == "claude" and event.get("type") == "result":
            text = event.get("result")
        elif cli == "codex" and event.get("type") == "item.completed":
            item = event.get("item") or {}
            if item.get("type") == "agent_message":
                text = item.get("text")
        elif cli == "pi" and event.get("type") == "agent_end":
            for message in reversed(event.get("messages") or []):
                if message.get("role") == "assistant":
                    parts = message.get("content") or []
                    if isinstance(parts, str):
                        text = parts
                    else:
                        text = "".join(
                            p.get("text", "") for p in parts if p.get("type") == "text"
                        )
                    break
    return text


# ---------------------------------------------------------------------------
# Running and classifying


@dataclass
class Run:
    cli: str
    task: str
    repeat: int
    outcome: str = "not-run"
    detail: str = ""
    exit_code: int | None = None
    duration_s: float = 0.0
    answer: str | None = None
    cancel_request_id: str | None = None
    cancel_server_tokens: int | None = None
    exchanges: list[dict] = field(default_factory=list)
    server: dict = field(default_factory=dict)
    native_argv: list[str] = field(default_factory=list)


def protocol_errors(exchanges: list[Exchange]) -> list[str]:
    """Hard failures on the wire: HTTP errors on /v1 routes, broken streams."""
    errors = []
    for e in exchanges:
        path = e.path.split("?")[0]
        if path.startswith("/v1/") and (e.status or 0) >= 400:
            errors.append(
                f"HTTP {e.status} on {e.method} {path}: {e.response_body[:300]}"
            )
        if e.streamed and path in MODEL_ROUTES:
            errors += [f"{path}: {msg}" for msg in e.stream_errors]
            if not e.terminal and not e.client_disconnected:
                errors.append(f"{path}: stream ended without its final event")
    return errors


def summarize_exchange(e: Exchange, server: dict[str, dict]) -> dict:
    out = {
        "seq": e.seq,
        "method": e.method,
        "path": e.path,
        "status": e.status,
        "request_id": e.request_id,
        "streamed": e.streamed,
        "events": e.events,
        "tool_calls": e.tool_calls,
        "client_disconnected": e.client_disconnected,
    }
    if e.request_id in server:
        out["server"] = server[e.request_id]
    return out


def fixture_text(repo: Path, name: str) -> str:
    """Read the harness snapshot outside the agent workspace."""
    return json.loads((repo.parent / "fixture.json").read_text())[name]


def tool_feedback(exchanges: list[Exchange], argument: str, expected: str) -> bool:
    """Require a successful structured tool result correlated to its invocation.

    This verifies replayed protocol evidence, not a sandbox or syscall trace.
    Unrelated text and unmatched tool IDs cannot satisfy the task.
    """

    def text(value):
        if isinstance(value, str):
            return value
        if isinstance(value, list):
            return "".join(
                p.get("text", "")
                for p in value
                if isinstance(p, dict) and p.get("type") == "text"
            )
        return ""

    for exchange in exchanges:
        try:
            body = json.loads(exchange.request_body)
        except (ValueError, TypeError):
            continue
        if not isinstance(body, dict):
            continue
        calls = {}
        items = body.get("messages", body.get("input", []))
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, dict):
                continue
            blocks = item.get("content", [])
            if item.get("type") in ("function_call", "function_call_output"):
                blocks = [item]
            for call in item.get("tool_calls", []):
                if isinstance(call, dict):
                    calls[call.get("id")] = json.dumps(
                        call.get("function", {}).get("arguments", "")
                    )
            if (
                item.get("role") == "tool"
                and item.get("tool_call_id")
                and argument in calls.get(item.get("tool_call_id"), "")
                and expected in text(blocks)
            ):
                return True
            if not isinstance(blocks, list):
                continue
            for block in blocks:
                if not isinstance(block, dict):
                    continue
                kind = block.get("type")
                if kind in ("tool_use", "function_call"):
                    key = (
                        block.get("id") if kind == "tool_use" else block.get("call_id")
                    )
                    calls[key] = json.dumps(
                        block.get("input", block.get("arguments", ""))
                    )
                elif kind in ("tool_result", "function_call_output"):
                    key = block.get("tool_use_id", block.get("call_id"))
                    output = text(block.get("content", block.get("output", "")))
                    if (
                        key
                        and not block.get("is_error")
                        and argument in calls.get(key, "")
                        and expected in output
                    ):
                        return True
    return False


def check_task(task: str, run: Run, repo: Path, exchanges: list[Exchange]) -> None:
    """Outcome from files, exit status and the wire, not exact model text."""
    model_calls = [e for e in exchanges if e.path.split("?")[0] in MODEL_ROUTES]
    if not model_calls:
        # The CLI failed before reaching the server: its setup, not the protocol.
        run.outcome, run.detail = "cli-error", "the CLI made no model request"
        return
    answer = run.answer or ""
    if task == "plain":
        ok = answer.strip() == "PONG"
        run.outcome = "pass" if ok and run.exit_code == 0 else "model-miss"
        run.detail = "" if ok else f"answer: {answer[:120]!r}"
    elif task == "tool":
        code = (
            fixture_text(repo, "notes.txt")
            .split("launch code is ")[1]
            .strip()
            .rstrip(".")
        )
        fed_back = tool_feedback(model_calls, "notes.txt", code)
        ok = fed_back and code in answer and run.exit_code == 0
        run.outcome = "pass" if ok else "model-miss"
        run.detail = f"correlated file result: {fed_back}; CLI exit {run.exit_code}"
    elif task == "edit":
        source = (repo / "calc.py").read_text()
        marker = fixture_text(repo, "test_calc.py").split("print('")[1].split("')")[0]
        ran_test = tool_feedback(model_calls, "python3 test_calc.py", marker)
        changed = subprocess.run(
            ["git", "-C", str(repo), "status", "--porcelain", "-z"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        # Exact source and unchanged test enforce the fixture's invariant without
        # executing potentially rewritten test code in the harness process.
        ok = (
            source == "def add(a, b):\n    return a + b\n"
            and changed == " M calc.py\0"
            and (repo / "test_calc.py").read_text()
            == fixture_text(repo, "test_calc.py")
            and (repo / "notes.txt").read_text() == fixture_text(repo, "notes.txt")
            and ran_test
            and run.exit_code == 0
        )
        run.outcome = "pass" if ok else "model-miss"
        run.detail = f"correlated test result: {ran_test}; status {changed!r}; CLI exit {run.exit_code}"


def run_one(cli: str, task: str, repeat: int, tap: Tap, log_path: Path, args) -> Run:
    run = Run(cli, task, repeat)
    root = Path(tempfile.mkdtemp(prefix=f"e2e-{cli}-{task}-"))
    home = root / "home"
    (home / "tmp").mkdir(parents=True)
    repo = make_fixture(root)
    argv, env = cli_command(cli, task, tap.url, args.model_id, home, repo, args)
    run.native_argv = list(argv)
    argv = sandbox_command(unmasked_command(argv), tap.url)
    mark = tap.mark()
    started = time.monotonic()
    out_path, err_path = root / "stdout.jsonl", root / "stderr.txt"
    process = None
    try:
        with out_path.open("w") as out, err_path.open("w") as err, defer_stop_signals():
            process = subprocess.Popen(
                argv,
                cwd=repo,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=out,
                stderr=err,
                start_new_session=True,
            )
        return finish_run(
            run, cli, task, process, tap, mark, started, root, repo, log_path, args
        )
    finally:
        if process is not None:
            kill_group(process)
            process.wait(timeout=10)


def finish_run(
    run, cli, task, process, tap, mark, started, root, repo, log_path, args
) -> Run:
    out_path = root / "stdout.jsonl"
    cancelled_at = None
    target = None
    if task == "cancel":
        target = tap.wait_for(
            lambda e: (
                e.seq >= mark
                and e.path.split("?")[0] in MODEL_ROUTES
                and e.events >= CANCEL_AFTER_EVENTS
                and not e.terminal
                and e.ended is None
            ),
            args.task_timeout,
            give_up=lambda: process.poll() is not None,  # The CLI already failed.
        )
        cancelled_at = time.time()
        kill_group(process)
    deadline = started + args.task_timeout
    while process.poll() is None:
        if paused(args):
            kill_group(process)
            process.wait(timeout=10)
            raise Paused(f"{args.pause_file} appeared during {cli}/{task}")
        if time.monotonic() > deadline:
            kill_group(process)
            run.detail = f"timed out after {args.task_timeout:g} s"
        time.sleep(0.5)
    run.exit_code = process.returncode
    run.duration_s = time.monotonic() - started
    stdout = out_path.read_text(errors="replace")
    run.answer = final_text(cli, stdout)
    if task == "cancel" and target is not None:
        # Let the server finish its bookkeeping for the cancelled request.
        deadline = time.monotonic() + args.cancel_grace
        while time.monotonic() < deadline:
            if target.request_id in server_requests(
                log_path.read_text(errors="replace")
            ):
                break
            time.sleep(0.5)
    exchanges = tap.since(mark)
    if args.keep:
        # Full request and response bodies, for repros.
        (root / "exchanges.json").write_text(
            json.dumps([asdict(e) for e in exchanges], indent=1)
        )
    server = server_requests(log_path.read_text(errors="replace"))
    run.exchanges = [summarize_exchange(e, server) for e in exchanges]
    run.server = {"workdir": str(root)}
    errors = protocol_errors(exchanges)
    if errors:
        run.outcome, run.detail = "protocol-error", "; ".join(errors)[:1000]
    elif task == "cancel":
        classify_cancel(run, target, cancelled_at, server, args)
    elif run.detail.startswith("timed out"):
        run.outcome = "model-miss"
    else:
        check_task(task, run, repo, exchanges)
    if not args.keep:
        shutil.rmtree(root, ignore_errors=True)
    return run


class Paused(Exception):
    """The pause file appeared: stop the CLI and the server until it clears."""


def paused(args) -> bool:
    return bool(args.pause_file) and args.pause_file.exists()


def kill_group(process: subprocess.Popen) -> None:
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass  # Gone already; macOS reports a group of one zombie as EPERM.


def classify_cancel(run: Run, target, cancelled_at, server: dict, args) -> None:
    if target is None:
        run.outcome = "model-miss"
        run.detail = (
            f"no stream reached {CANCEL_AFTER_EVENTS} events before the timeout"
        )
        return
    finished = server.get(target.request_id)
    span = (target.last_event or 0) - (target.first_event or 0)
    if not target.streamed or span <= 0:
        run.outcome, run.detail = "protocol-error", "the response was not streamed"
        return
    if finished is None:
        run.outcome = "protocol-error"
        run.detail = (
            f"request {target.request_id} still unfinished {args.cancel_grace:g} s "
            "after the client disconnected"
        )
        return
    try:
        body = json.loads(target.request_body)
    except json.JSONDecodeError:
        body = {}
    limit = body.get("max_tokens") or body.get("max_output_tokens")
    # Generation stopped if the server finished the request soon after the
    # disconnect, short of its output limit. (Token counts come from the
    # usage when logged, else from the span timeline after the session.)
    after = (
        finished["finished"] - cancelled_at
        if finished["finished"] and cancelled_at is not None
        else None
    )
    tokens = finished["output_tokens"]
    stopped = (after is not None and 0 <= after <= args.cancel_grace) and (
        (tokens is not None and limit is not None and tokens < limit)
        or finished.get("error") is not None
    )
    run.outcome = (
        "pass"
        if stopped and target.client_disconnected and not target.terminal
        else "protocol-error"
    )
    run.cancel_request_id = target.request_id
    run.detail = (
        f"{target.events} events over {span:.2f} s before the cancel; server finished "
        f"the request {after:.1f} s after the disconnect"
        if after is not None
        else "no finish time"
    ) + (
        f" ({finished['error']}); generated {tokens if tokens is not None else '?'} "
        f"of {limit} tokens; client disconnect seen: {target.client_disconnected}"
    )


def render(runs: list[Run], clis: list[str], tasks: list[str]) -> str:
    cells: dict[tuple[str, str], list[Run]] = {}
    for run in runs:
        cells.setdefault((run.cli, run.task), []).append(run)
    width = 26
    lines = ["cli     " + "".join(f"{task:<{width}}" for task in tasks)]
    for cli in clis:
        row = f"{cli:<8}"
        for task in tasks:
            group = cells.get((cli, task), [])
            passed = sum(r.outcome == "pass" for r in group)
            worst = next(
                (
                    o
                    for o in (
                        "protocol-error",
                        "cli-error",
                        "aborted",
                        "not-run",
                        "model-miss",
                    )
                    if any(r.outcome == o for r in group)
                ),
                "pass" if group else "-",
            )
            cell = f"{passed}/{len(group)} " + (worst if worst != "pass" else "pass")
            row += f"{cell:<{width}}"
        lines.append(row)
    lines.append("")
    for run in runs:
        ids = ",".join(
            (e["request_id"] or "-")[:8]
            for e in run.exchanges
            if e["path"].split("?")[0] in MODEL_ROUTES
        )
        lines.append(
            f"{run.cli}/{run.task}#{run.repeat}: {run.outcome} exit={run.exit_code} "
            f"{run.duration_s:.0f}s ids=[{ids}] {run.detail[:160]}"
        )
    return "\n".join(lines)


class Session:
    """One mx serve at a fixed address, GPU-capped, stopped while paused and
    started again afterwards (each start logs to its own file)."""

    def __init__(self, argv: list[str], address: str, args):
        self.spec = bench_load.ServerSpec(
            "metallix", "chat", argv, {}, {}, environment={"METALLIX_LOG": "info"}
        )
        self.address = address
        self.args = args
        self.server = None
        self.sampler = None
        self.logs: list[str] = []
        self.memory: list[dict] = []
        self.traces: list[str] = []
        self.baselines: list[int] = []
        self.launches: list[list[str]] = []
        self.shutdowns: list[dict] = []
        self.aborted: str | None = None

    @property
    def log_path(self) -> Path:
        return Path(self.logs[-1])

    def ensure_started(self) -> None:
        if self.server is not None:
            return
        log = self.args.log_dir / f"mx-serve-{len(self.logs) + 1}.log"
        self.logs.append(str(log))
        trace = log.with_suffix(".trace.json")
        self.traces.append(str(trace))
        argv = unmasked_command(self.spec.argv + ["--trace-out", str(trace)])
        self.launches.append(argv)
        spec = replace(self.spec, argv=argv)
        # The GPU reading is machine-wide; cap what this server adds to what
        # other jobs already hold.
        baseline = bench_system.probe(None).get("gpu_in_use_bytes") or 0
        self.baselines.append(baseline)
        with defer_stop_signals():
            self.server = bench_load.ManagedServer(spec, self.address, log)
        self.sampler = bench_system.Sampler(
            pgid=self.server.process.pid,
            abort_gpu_bytes=baseline + int(self.args.abort_gpu_gib * 2**30),
            on_abort=self.abort,
        )
        self.sampler.__enter__()
        self.server.wait_ready(600)

    def abort(self, reason: str) -> None:
        self.aborted = reason
        if self.server:
            self.server.stop()

    def stop(self) -> None:
        try:
            if self.sampler is not None:
                self.sampler.__exit__(None, None, None)
                self.memory.append(self.sampler.summary())
                self.sampler = None
        finally:
            if self.server is not None:
                self.server.stop()
                self.shutdowns.append(dict(self.server.shutdown))
                self.server = None


def validate_timelines(traces: list[str], model_id: str) -> dict:
    """Strictly validate closed front/child timelines without repairing files."""
    files = []
    for trace in traces:
        for path in (Path(trace), mx_spans.child_path(Path(trace), model_id)):
            row = {"path": str(path), "valid": False}
            try:
                events = json.loads(path.read_text())
                if (
                    not isinstance(events, list)
                    or not events
                    or not all(
                        isinstance(event, dict) and "ph" in event for event in events
                    )
                ):
                    raise ValueError("expected a nonempty Chrome event array")
                row.update(valid=True, events=len(events))
            except (OSError, ValueError) as error:
                row["error"] = str(error)
            files.append(row)
    return {"valid": bool(files) and all(row["valid"] for row in files), "files": files}


def add_span_tokens(runs: list[Run], traces: list[str], model_id: str) -> None:
    """Decode steps the server ran for each cancelled request, from the span
    timelines (complete only once the server has exited)."""
    counts: dict[str, int] = {}
    for trace in traces:
        try:
            proxy = mx_spans.load_trace(Path(trace)) if Path(trace).exists() else []
            child = mx_spans.load_trace(mx_spans.child_path(Path(trace), model_id))
        except (OSError, ValueError):
            continue
        for row in mx_spans.request_timings(proxy, child):
            counts[row["request_id"]] = row["decode_steps"]
    for run in runs:
        if run.cancel_request_id in counts:
            # One token comes from prefill; each decode step adds one.
            run.cancel_server_tokens = counts[run.cancel_request_id] + 1
            run.detail += f"; spans: {run.cancel_server_tokens} tokens generated"


def tool_versions(args) -> dict:
    """Inspect versions with the same confinement and fresh empty auth home."""
    out = {}
    with tempfile.TemporaryDirectory(prefix="e2e-version-") as root:
        home = Path(root)
        (home / "tmp").mkdir()
        for name in args.clis.split(","):
            argv = sandbox_command([getattr(args, name), "--version"], args.tap_url)
            result = subprocess.run(
                argv,
                env=base_env(home),
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            out[name] = {
                "exit_code": result.returncode,
                "version": result.stdout.strip().splitlines()[-1:],
            }
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--run", action="store_true", help="execute the plan (default: dry-run)"
    )
    parser.add_argument("--mx", type=Path, default=Path("target/release/mx"))
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-id", default="qwen3-4b")
    parser.add_argument("--clis", default=",".join(CLIS))
    parser.add_argument("--tasks", default=",".join(TASKS))
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--context-tokens", type=int, default=16384)
    parser.add_argument("--max-output-tokens", type=int, default=1024)
    parser.add_argument("--kv-budget-mib", type=int, default=4096)
    parser.add_argument("--abort-gpu-gib", type=float, default=32)
    parser.add_argument("--task-timeout", type=float, default=180)
    parser.add_argument("--session-timeout", type=float, default=1200)
    parser.add_argument("--cancel-grace", type=float, default=20)
    parser.add_argument("--claude", default=FIXED_BINARIES["claude"])
    parser.add_argument("--codex", default=FIXED_BINARIES["codex"])
    parser.add_argument("--pi", default=FIXED_BINARIES["pi"])
    parser.add_argument(
        "--log-dir", type=Path, default=Path(tempfile.gettempdir()) / "e2e-agents"
    )
    parser.add_argument("--keep", action="store_true", help="keep each run's workdir")
    parser.add_argument(
        "--pause-file",
        type=Path,
        help="while this file exists, stop the CLI and the server, then resume",
    )
    parser.add_argument("--json", type=Path, help="write the receipt here")
    args = parser.parse_args()
    clis = [c for c in args.clis.split(",") if c]
    tasks = [t for t in args.tasks.split(",") if t]
    for name in clis:
        if name not in CLIS:
            parser.error(f"unknown CLI {name!r}")
    for name in tasks:
        if name not in TASKS:
            parser.error(f"unknown task {name!r}")

    if not clis or not tasks:
        parser.error("select at least one CLI and task")
    for name in (
        "repeats",
        "context_tokens",
        "max_output_tokens",
        "kv_budget_mib",
        "abort_gpu_gib",
        "task_timeout",
        "session_timeout",
        "cancel_grace",
    ):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive and finite")
    if not args.run:
        print(
            json.dumps(
                {
                    "dry_run": True,
                    "clis": clis,
                    "tasks": tasks,
                    "repeats": args.repeats,
                    "model": str(args.model),
                    "mx": str(args.mx),
                    "model_id": args.model_id,
                },
                indent=2,
            )
        )
        return 0
    for cli in clis:
        if getattr(args, cli) != FIXED_BINARIES[cli]:
            parser.error(f"{cli} must use the fixed native executable")
    args.log_dir.mkdir(parents=True, exist_ok=True)
    args.log_dir = Path(tempfile.mkdtemp(prefix="run-", dir=args.log_dir))
    if args.json is None:
        args.json = args.log_dir / "receipt.json"
    address = bench_serve.free_address()
    argv = [
        str(args.mx.resolve()), "serve", "--model", str(args.model), "--model-id",
        args.model_id, "--listen", address, "--context-tokens", str(args.context_tokens),
        "--kv-budget-mib", str(args.kv_budget_mib), "--queue-depth", "16",
        # The longest budget mx allows: agent prompts are long, and prefill
        # on a busy machine is slow.
        "--generation-timeout-ms", "120000",
    ]  # fmt: skip
    session = Session(argv, address, args)
    tap = Tap(address)
    args.tap_url = tap.url
    runs: list[Run] = []
    pending = [
        (cli, task, repeat)
        for cli in clis
        for task in tasks
        for repeat in range(args.repeats)
    ]
    previous = {
        sig: signal.getsignal(sig)
        for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGALRM)
    }
    for sig in previous:
        signal.signal(sig, stop_signal)
    signal.setitimer(signal.ITIMER_REAL, args.session_timeout)
    try:
        validate_sandbox(tap.url)
        while pending:
            cli, task, repeat = pending[0]
            if session.aborted:
                runs.append(Run(cli, task, repeat, "not-run", session.aborted))
                pending.pop(0)
                continue
            if paused(args):
                session.stop()
                print(f"paused: {args.pause_file} exists", flush=True)
                while paused(args):
                    time.sleep(15)
            session.ensure_started()
            print(f"{cli}/{task}#{repeat} ...", flush=True)
            try:
                run = run_one(cli, task, repeat, tap, session.log_path, args)
            except Paused as pause:
                print(f"  {pause}; retrying after it clears", flush=True)
                session.stop()
                continue
            if session.aborted:
                # The server was stopped under this run: nothing it saw counts.
                run.outcome, run.detail = "aborted", session.aborted
            print(f"  {run.outcome}: {run.detail[:200]}", flush=True)
            runs.append(run)
            pending.pop(0)
            if run.outcome == "protocol-error":
                # A failed request can leave the model busy (or the server in
                # a bad state); start the next run on a fresh server so one
                # failure is not counted again in every run after it.
                session.stop()
    except RunStopped as error:
        runs.extend(
            Run(cli, task, repeat, "aborted", str(error))
            for cli, task, repeat in pending
        )
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        try:
            session.stop()
        finally:
            tap.close()
            for sig, handler in previous.items():
                signal.signal(sig, handler)
    timeline = validate_timelines(session.traces, args.model_id)
    add_span_tokens(runs, session.traces, args.model_id)
    print()
    print(render(runs, clis, tasks))
    if args.json:
        receipt = {
            "argv": sys.argv,
            "server_argv": argv,
            "server_logs": session.logs,
            "server_traces": session.traces,
            "server_launches": session.launches,
            "server_shutdown": session.shutdowns,
            "exec_launcher": {
                "interpreter": sys.executable,
                "flags": ["-I", "-S"],
                "program": UNMASK_EXEC,
            },
            "timeline": timeline,
            "native_tasks_passed": bool(runs)
            and all(run.outcome == "pass" for run in runs),
            "gpu_baselines": session.baselines,
            "system": bench_system.system_info(),
            "tools": tool_versions(args),
            "memory": session.memory,
            "runs": [asdict(run) for run in runs],
        }
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(receipt, indent=1) + "\n")
    hard = [run for run in runs if run.outcome != "pass"]
    if not timeline["valid"]:
        print(
            "timeline validation failed; native task outcomes are reported separately"
        )
    return 1 if hard or not timeline["valid"] else 0


if __name__ == "__main__":
    sys.exit(main())
