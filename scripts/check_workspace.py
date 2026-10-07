#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Experimental POSIX source isolation for the canonical check, opt-in only.

A cooperating command inherits the lane lease across supervisor death. Detached
processes that close inherited descriptors are outside this lifetime guarantee.
Use check.py --isolated with an absolute METALLIX_CHECK_STATE_DIR outside the
checkout, or opt in locally through that variable alone. CI ignores the variable
unless --isolated is explicit; --direct always selects the ordinary local gate.

Generated files belong outside src: Python bytecode is disabled and Ruff's cache
is redirected. Other unexpected files stop the next sync rather than being erased.
"""

import contextlib
import hashlib
import json
import os
import signal
import stat
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def selected(args, environment):
    """Explicit switches win; CI never inherits local automatic selection."""
    return args.isolated or (
        not args.direct
        and not environment.get("CI")
        and bool(environment.get("METALLIX_CHECK_STATE_DIR"))
    )


def manifest(root):
    """Read working bytes, including nonignored untracked files and deletions."""
    command = ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"]
    paths = subprocess.check_output(command, cwd=root).split(b"\0")
    result = {}
    for raw in sorted(set(paths) - {b""}):
        name = os.fsdecode(raw)
        path = root / name
        if Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError(f"unsafe source path: {name}")
        if not path.exists() and not path.is_symlink():
            continue
        mode = path.lstat().st_mode
        if not stat.S_ISREG(mode) or any(
            parent.is_symlink() for parent in path.parents if parent != root.parent
        ):
            raise ValueError(f"unsupported symlink, submodule or non-file: {name}")
        result[name] = [
            stat.S_IMODE(mode),
            hashlib.sha256(path.read_bytes()).hexdigest(),
        ]
    local_config = root / ".cargo/config.toml"
    if local_config.exists() and ".cargo/config.toml" not in result:
        raise ValueError("ignored .cargo/config.toml needs an explicit input policy")
    return result


def digest(entries):
    return hashlib.sha256(json.dumps(entries, sort_keys=True).encode()).hexdigest()


def capture(root, snapshots):
    """Preserve a receipt even on failure; never claim atomic filesystem capture."""
    before = manifest(root)
    destination = Path(tempfile.mkdtemp(prefix="request-", dir=snapshots)) / "source"
    destination.mkdir()
    for name, (mode, expected) in before.items():
        data = (root / name).read_bytes()
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError("source changed during capture")
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        target.chmod(mode)
    if before != manifest(root):
        raise ValueError("source changed during capture")
    return destination, before


def private_directory(path):
    """Reject symlinked or foreign-owned state before touching its contents."""
    for ancestor in [path, *path.parents]:
        if ancestor.is_symlink():
            raise ValueError(f"symlinked state path: {ancestor}")
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    if path.stat().st_uid != os.getuid() or path.stat().st_mode & 0o022:
        raise ValueError(
            f"state directory must be owned and not group/world writable: {path}"
        )


def previous_manifest(path):
    """Validate persisted ownership before it can authorize any source mutation."""
    if not path.exists():
        return {}
    if not path.is_file() or path.is_symlink():
        raise ValueError("invalid lane manifest file")
    entries = json.loads(path.read_text())
    if not isinstance(entries, dict):
        raise TypeError("invalid lane manifest schema")
    for name, value in entries.items():
        parts = Path(name).parts
        if (
            not name
            or Path(name).is_absolute()
            or ".." in parts
            or name != Path(name).as_posix()
            or name == "."
            or not isinstance(value, list)
            or len(value) != 2
            or type(value[0]) is not int
            or not 0 <= value[0] <= 0o777
            or not isinstance(value[1], str)
            or len(value[1]) != 64
            or any(char not in "0123456789abcdef" for char in value[1])
        ):
            raise ValueError("invalid lane manifest entry")
    return entries


def sync_source(snapshot, entries, lane):
    """Only prior manifest-owned paths can be removed; unknown files fail closed."""
    source = lane / "src"
    previous_path = lane / "manifest.json"
    if previous_path.is_symlink() or (lane / "incomplete").is_symlink():
        raise ValueError("symlinked lane metadata")
    previous = previous_manifest(previous_path)
    owned_directories = {
        parent.as_posix()
        for name in previous
        for parent in Path(name).parents
        if parent != Path(".")
    }
    private_directory(source)
    for path in source.rglob("*"):
        name = path.relative_to(source).as_posix()
        if (
            path.is_symlink()
            or (path.is_dir() and name not in owned_directories)
            or (not path.is_dir() and name not in previous)
        ):
            raise ValueError(f"unexpected lane source path, preserved: {name}")
    (lane / "incomplete").write_text("source sync or check not yet complete\n")
    for name in previous.keys() - entries.keys():
        path = source / name
        if path.exists():
            path.unlink()
    # Remove empty owned source directories, allowing file/directory transitions.
    for path in sorted(source.rglob("*"), key=lambda p: len(p.parts), reverse=True):
        if path.is_dir():
            with contextlib.suppress(OSError):
                path.rmdir()
    for name, (mode, expected) in entries.items():
        target = source / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if (
            not target.exists()
            or hashlib.sha256(target.read_bytes()).hexdigest() != expected
        ):
            target.write_bytes((snapshot / name).read_bytes())
        target.chmod(mode)
    for name, (mode, expected) in entries.items():
        path = source / name
        if (
            stat.S_IMODE(path.stat().st_mode) != mode
            or hashlib.sha256(path.read_bytes()).hexdigest() != expected
        ):
            raise ValueError(f"projected source verification failed: {name}")
    previous_path.write_text(json.dumps(entries, sort_keys=True))
    return source


def supervise(command, *, cwd, environment, lease_fd, timeout=None):
    """Forward cancellation, reap the process group leader, and retain its lease."""
    received = []

    def interrupt(signum, _frame):
        received.append(signum)

    signals = (signal.SIGTERM, signal.SIGINT, signal.SIGHUP)
    previous = {sig: signal.signal(sig, interrupt) for sig in signals}
    process = None
    started = time.monotonic()
    try:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=environment,
            start_new_session=True,
            pass_fds=(lease_fd,),
        )
        while process.poll() is None:
            if received or (
                timeout is not None and time.monotonic() - started >= timeout
            ):
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(process.pid, received[0] if received else signal.SIGTERM)
                try:
                    process.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    pass
                # Also stop descendants when the leader has already exited.
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGKILL)
                process.wait()
                return 128 + received[0] if received else 124
            time.sleep(0.05)
        if process.returncode < 0:
            raise RuntimeError(
                "command died unexpectedly; cleanup state requires review"
            )
        return process.returncode
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def run_isolated(root, args):
    """Capture once, serialize at a stable path, then run the captured gate."""
    if os.name != "posix":
        raise ValueError("experimental isolated checks require POSIX")
    if os.environ.get("SLOT_DIR"):
        raise ValueError("invoke check directly, outside the legacy slot wrapper")
    configured = os.environ.get("METALLIX_CHECK_STATE_DIR")
    if not configured or not Path(configured).is_absolute():
        raise ValueError("--isolated requires absolute METALLIX_CHECK_STATE_DIR")
    state = Path(configured)
    root = root.resolve()
    if state.resolve().is_relative_to(root) or root.is_relative_to(state.resolve()):
        raise ValueError("state must be outside the source tree and its ancestors")
    private_directory(state)
    lane = state / "lane"
    snapshots = state / "requests"
    for directory in (lane, snapshots, lane / "target", lane / "cache"):
        private_directory(directory)
    target = str(lane / "target")
    if os.environ.get("CARGO_TARGET_DIR", target) != target:
        raise ValueError(
            "CARGO_TARGET_DIR conflicts with the persistent isolated target"
        )
    started = time.monotonic()
    snapshot, entries = capture(root, snapshots)
    captured = time.monotonic()
    receipt = {"source_digest": digest(entries), "capture_seconds": captured - started}
    print(
        f"experimental isolated check: {receipt['source_digest']} lane={lane}",
        flush=True,
    )
    import fcntl

    flags = os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW
    with os.fdopen(os.open(lane / "check.lock", flags, 0o600), "w") as lease:
        last_report = 0.0
        while True:
            try:
                fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                waited = time.monotonic() - captured
                if waited >= args.queue_timeout_seconds:
                    raise ValueError("isolated check queue timeout (gate not started)")
                if waited - last_report >= 1:
                    print(f"waiting for check lane: {waited:.1f}s", flush=True)
                    last_report = waited
                time.sleep(0.05)
        receipt["queue_seconds"] = time.monotonic() - captured
        if (lane / "incomplete").exists():
            raise ValueError(
                "incomplete prior isolated run; inspect retained state before explicit recovery"
            )
        sync_started = time.monotonic()
        source = sync_source(snapshot, entries, lane)
        receipt["sync_seconds"] = time.monotonic() - sync_started
        environment = os.environ | {
            "CARGO_TARGET_DIR": target,
            "PYTHONDONTWRITEBYTECODE": "1",
            "RUFF_CACHE_DIR": str(lane / "cache" / "ruff"),
        }
        # Load the captured implementation, whose ROOT points at stable source.
        completed_path = snapshot.parent / "gate-completed.json"
        bootstrap = (
            "import argparse,json,sys; from pathlib import Path; "
            "sys.path.insert(0, 'scripts'); import check; "
            "result=check.run_checks(argparse.Namespace(metal=sys.argv[1]=='1', "
            "timeout_seconds=int(sys.argv[2])), lease_fd=int(sys.argv[3])); "
            "Path(sys.argv[4]).write_text(json.dumps({'exit_code':result})); sys.exit(result)"
        )
        gate_started = time.monotonic()
        result = supervise(
            [
                sys.executable,
                "-c",
                bootstrap,
                str(int(args.metal)),
                str(args.timeout_seconds),
                str(lease.fileno()),
                str(completed_path),
            ],
            cwd=source,
            environment=environment,
            lease_fd=lease.fileno(),
        )
        if not completed_path.is_file():
            raise ValueError(
                "gate supervisor did not record completed cleanup; state retained"
            )
        completed = json.loads(completed_path.read_text())
        # Cancellation can interrupt after a command's completed receipt, but
        # absent receipts never authorize reuse after abrupt worker death.
        if completed.get("exit_code") != result:
            raise ValueError("gate completion status mismatch; state retained")
        receipt.update(gate_seconds=time.monotonic() - gate_started, exit_code=result)
        receipt["source_changed_since_capture"] = manifest(root) != entries
        (snapshot.parent / "receipt.json").write_text(
            json.dumps(receipt, indent=2) + "\n"
        )
        # A failed but fully supervised command is complete too. An abrupt
        # controller death leaves the marker and requires explicit recovery.
        (lane / "incomplete").unlink()
        print(f"check receipt: {snapshot.parent / 'receipt.json'}", flush=True)
        return result
