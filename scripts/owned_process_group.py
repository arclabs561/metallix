"""Unreaped-child anchored cleanup for trusted children that stay in their session."""

import ctypes
import os
import signal
import subprocess
import sys
import time


def members(pgid):
    """Bounded metadata-only process-group inventory, including zombies."""
    result = subprocess.run(
        ["/bin/ps", "-axo", "pid=,pgid=,stat="],
        capture_output=True,
        text=True,
        check=True,
        timeout=0.25,
    )
    found = {}
    for line in result.stdout.splitlines():
        pid, group, state = line.split()
        if int(group) == pgid:
            found[int(pid)] = state
    return found


def exited(process):
    # WNOWAIT keeps the owned child's PID reserved until all group signals end.
    # Do not use Popen.poll()/wait() before cleanup: those reap the anchor.
    return os.waitid(os.P_PID, process.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)


def confirm_terminal_after_disappearance(process, timeout=0.25):
    """Wait boundedly for a still-owned child to become waitable, without reaping.

    macOS may stop resolving process metadata before WNOWAIT exposes terminal
    status. ECHILD propagates: a reaped/recycled PID is never an ownership proof.
    A live child or observation failure remains fatal after the short bound.
    """
    deadline = time.monotonic() + timeout
    while True:
        terminal = exited(process)
        if terminal is not None:
            return terminal
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return None
        time.sleep(min(0.005, remaining))


def cleanup(process, event, grace=0.25):
    """Signal only the anchored group; telemetry cannot interrupt cleanup."""
    telemetry_errors = []

    def note(kind, **fields):
        try:
            event(kind, **fields)
        except (OSError, ValueError, TypeError) as exc:
            telemetry_errors.append(str(exc))

    terminal = exited(process)
    # An unreaped zombie has no getpgid result on macOS. WNOWAIT still
    # proves our child owns the reserved PID, until we reap after signaling.
    if terminal is None:
        try:
            group = os.getpgid(process.pid)
        except ProcessLookupError:
            # The child can exit between WNOWAIT and getpgid. Only a second
            # WNOWAIT confirming our still-unreaped child preserves the anchor.
            # ECHILD (already reaped), or a still-live child, must fail closed.
            if confirm_terminal_after_disappearance(process) is None:
                raise
        else:
            if group != process.pid:
                raise RuntimeError(
                    "owned process-group identity lost; refusing group signal"
                )
    if process.pid == os.getpgrp():
        raise RuntimeError("refusing to signal watchdog process group")
    for sig, wait in ((signal.SIGTERM, grace), (signal.SIGKILL, 2.0)):
        try:
            os.killpg(process.pid, sig)
            note("signal", signal=sig.name, pgid=process.pid)
        except (ProcessLookupError, PermissionError) as exc:
            # A zombie-only group may report EPERM. Defer its disposition
            # to fresh confirmation; never let it skip reaping.
            note("signal_result", signal=sig.name, detail=str(exc))
        deadline = time.monotonic() + wait
        empty = False
        while time.monotonic() < deadline:
            try:
                empty = not any(
                    "Z" not in state for state in members(process.pid).values()
                )
            except (OSError, ValueError, subprocess.SubprocessError) as exc:
                note("cleanup_observation_failed", detail=str(exc))
                break  # escalate even when observation is unavailable
            if empty:
                break
            time.sleep(0.02)
        if empty:
            break
    process.wait(timeout=2.0)
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        remaining = members(process.pid)
        if not remaining:
            note("cleanup", group_empty=True)
            return telemetry_errors
        time.sleep(0.02)
    raise RuntimeError(f"group cleanup unconfirmed: {remaining}")


def require_wait_capabilities():
    """Reject interpreters lacking unreaped-child observation before any launch."""
    required = ("waitid", "P_PID", "WNOWAIT", "WEXITED", "WNOHANG")
    missing = [name for name in required if not hasattr(os, name)]
    if not missing and not callable(os.waitid):
        missing.append("waitid")
    if missing:
        raise RuntimeError(
            "unsupported Python: owned-group cleanup requires os."
            + ", os.".join(missing)
        )


def footprint(process, sampler, identities):
    """Sample live group members; None means the unreaped leader exited."""
    try:
        group = members(process.pid)
        if process.pid not in group:
            if confirm_terminal_after_disappearance(process) is not None:
                return None
            raise RuntimeError("owned leader absent from group inventory")
        total = 0
        for pid, state in group.items():
            if "Z" in state:
                continue
            try:
                if os.getpgid(pid) != process.pid:
                    raise RuntimeError("sampled member changed process group")
                size, identity = sampler(pid)
            except ProcessLookupError:
                if pid == process.pid:
                    raise
                current = members(process.pid).get(pid)
                if current is not None and "Z" not in current:
                    raise
                continue
            if not isinstance(size, int) or size < 0 or not identity:
                raise RuntimeError("invalid footprint measurement")
            if identities.setdefault(pid, identity) != identity:
                raise RuntimeError("process identity changed")
            total += size
        return total
    except ProcessLookupError:
        if confirm_terminal_after_disappearance(process) is None:
            raise
        return None


class Usage(ctypes.Structure):
    _fields_ = [("ri_uuid", ctypes.c_uint8 * 16)] + [
        (name, ctypes.c_uint64)
        for name in (
            "user_time",
            "system_time",
            "pkg_idle_wkups",
            "interrupt_wkups",
            "pageins",
            "wired_size",
            "resident_size",
            "phys_footprint",
            "proc_start_abstime",
            "proc_exit_abstime",
            "child_user_time",
            "child_system_time",
            "child_pkg_idle_wkups",
            "child_interrupt_wkups",
            "child_pageins",
            "child_elapsed_abstime",
            "diskio_bytesread",
            "diskio_byteswritten",
        )
    ]


class NativeSampler:
    def __init__(self):
        if sys.platform != "darwin":
            raise RuntimeError("requires macOS physical footprint API")
        self.lib = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
        self.lib.proc_pid_rusage.argtypes = [
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_void_p,
        ]
        self.lib.proc_pid_rusage.restype = ctypes.c_int
        if ctypes.sizeof(Usage) != 160 or Usage.phys_footprint.offset != 72:
            raise RuntimeError("rusage_info_v2 ABI mismatch")

    def __call__(self, pid):
        usage = Usage()
        if self.lib.proc_pid_rusage(pid, 2, ctypes.byref(usage)) != 0:
            raise OSError(ctypes.get_errno(), "proc_pid_rusage failed", pid)
        if usage.proc_start_abstime == 0:
            raise RuntimeError("missing process start identity")
        return usage.phys_footprint, usage.proc_start_abstime
