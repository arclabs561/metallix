# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Machine probes for the serving benchmarks: system metadata and a 1 Hz sampler.

The parsers take command output as text so tests can feed them recorded
output; the probes run macOS tools (`sw_vers`, `vm_stat`, `ioreg`, `ps`).
"""

from __future__ import annotations

import glob
import os
import platform
import re
import subprocess
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Self

ROOT = Path(__file__).resolve().parent.parent


def command_output(argv: list[str], timeout: float = 10) -> str:
    """Stdout of a probe command, or "" when it is missing or fails."""
    try:
        return subprocess.run(
            argv, capture_output=True, text=True, check=False, timeout=timeout
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return ""


# ---------------------------------------------------------------------------
# Metadata


def parse_sw_vers(text: str) -> dict:
    """`sw_vers` lines (`ProductVersion:\t26.6.2`) as a dict."""
    out = {}
    for line in text.splitlines():
        key, _, value = line.partition(":")
        if value.strip():
            out[key.strip()] = value.strip()
    return out


def lock_versions(lock_text: str, names: tuple[str, ...]) -> dict[str, str]:
    """Versions of the named packages in a Cargo.lock."""
    out = {}
    for block in lock_text.split("[[package]]"):
        name = re.search(r'^name = "([^"]+)"', block, re.MULTILINE)
        version = re.search(r'^version = "([^"]+)"', block, re.MULTILINE)
        if name and version and name.group(1) in names:
            out[name.group(1)] = version.group(1)
    return out


def bundled_mlx_version(mlx_sys_version: str) -> str | None:
    """The MLX release mlx-sys fetches at build time, from its CMake file in the
    Cargo registry (mlx-sys pins it with a FetchContent GIT_TAG)."""
    home = os.environ.get("CARGO_HOME", str(Path.home() / ".cargo"))
    pattern = (
        f"{home}/registry/src/*/mlx-sys-{mlx_sys_version}/src/mlx-c/CMakeLists.txt"
    )
    for path in glob.glob(pattern):
        match = re.search(
            r"GIT_REPOSITORY\s+\"[^\"]*ml-explore/mlx\.git\"\s+GIT_TAG\s+v?([\w.]+)",
            Path(path).read_text(),
        )
        if match:
            return match.group(1)
    return None


def metallix_versions(root: Path = ROOT) -> dict:
    try:
        lock = (root / "Cargo.lock").read_text()
    except OSError:
        return {}
    out = lock_versions(lock, ("mlx-rs", "mlx-sys"))
    if "mlx-sys" in out:
        out["mlx"] = bundled_mlx_version(out["mlx-sys"]) or "unknown"
    return out


def system_info() -> dict:
    """What a result needs to be read later: OS build, chip, memory, MLX."""
    sw = parse_sw_vers(command_output(["sw_vers"]))
    return {
        "macos": sw.get("ProductVersion"),
        "macos_build": sw.get("BuildVersion"),
        "chip": command_output(["sysctl", "-n", "machdep.cpu.brand_string"]).strip(),
        "memory_bytes": int(command_output(["sysctl", "-n", "hw.memsize"]) or 0),
        "machine": platform.machine(),
        "metallix_mlx": metallix_versions(),
    }


# ---------------------------------------------------------------------------
# Memory and GPU


def parse_vm_stat(text: str) -> int | None:
    """Memory in use, in bytes, the way Activity Monitor adds it up: app memory
    (anonymous minus purgeable pages), wired pages, and the compressor's pages."""
    page = re.search(r"page size of (\d+) bytes", text)
    if not page:
        return None
    pages = {
        key.strip().strip('"'): int(value)
        for key, value in re.findall(r"^([^:\n]+):\s+(\d+)\.", text, re.MULTILINE)
    }
    used = (
        pages.get("Anonymous pages", 0)
        - pages.get("Pages purgeable", 0)
        + pages.get("Pages wired down", 0)
        + pages.get("Pages occupied by compressor", 0)
    )
    return used * int(page.group(1))


def parse_gpu_stats(text: str) -> dict:
    """Device utilization (%) and GPU-resident system memory (bytes) from
    `ioreg -r -c AGXAccelerator -d 1`; None for a field it does not report."""

    def field(name: str) -> int | None:
        match = re.search(rf'"{re.escape(name)}"=(\d+)', text)
        return int(match.group(1)) if match else None

    return {
        "utilization_pct": field("Device Utilization %"),
        "in_use_bytes": field("In use system memory"),
    }


def group_rss_bytes(ps_text: str, pgid: int) -> int:
    """Resident memory of every process in one process group, from
    `ps -axo pid=,pgid=,rss=` (rss in KiB)."""
    total = 0
    for line in ps_text.splitlines():
        parts = line.split()
        if len(parts) >= 3 and parts[1] == str(pgid) and parts[2].isdigit():
            total += int(parts[2]) * 1024
    return total


# Darwin's OSThermalPressureLevel, as `notifyutil` reports it. Apple Silicon
# reports throttling only here: `pmset -g therm` prints its CPU limits on
# Intel Macs alone.
THERMAL_PRESSURE = ("nominal", "moderate", "heavy", "trapping", "sleeping")


def parse_thermal_pressure(text: str) -> int | None:
    """The level from `notifyutil -g com.apple.system.thermalpressurelevel`."""
    match = re.search(r"thermalpressurelevel\s+(\d+)", text)
    return int(match.group(1)) if match else None


def thermal_pressure() -> int | None:
    return parse_thermal_pressure(
        command_output(["notifyutil", "-g", "com.apple.system.thermalpressurelevel"])
    )


def probe(pgid: int | None) -> dict:
    """One sample of the quantities the sampler tracks."""
    gpu = parse_gpu_stats(
        command_output(["ioreg", "-r", "-c", "AGXAccelerator", "-d", "1"])
    )
    return {
        "load_1m": os.getloadavg()[0],
        "system_used_bytes": parse_vm_stat(command_output(["vm_stat"])),
        "gpu_in_use_bytes": gpu["in_use_bytes"],
        "gpu_utilization_pct": gpu["utilization_pct"],
        "thermal_level": thermal_pressure(),
        "server_rss_bytes": (
            group_rss_bytes(command_output(["ps", "-axo", "pid=,pgid=,rss="]), pgid)
            if pgid is not None
            else None
        ),
    }


PEAKS = (
    "load_1m",
    "system_used_bytes",
    "gpu_in_use_bytes",
    "server_rss_bytes",
    "thermal_level",
)
# Kept per sample, so a report shows when the GPU went busy or hot, not only
# how far it went.
SERIES = ("gpu_utilization_pct", "thermal_level", "gpu_in_use_bytes")


class Sampler:
    """Samples memory, GPU memory and load every `interval` seconds on a thread.

    Keeps the peak of each quantity, and a series (seconds since the first
    sample, plus SERIES) of every sample. When the 1-minute load average rises above
    `abort_load`, or GPU-resident memory above `abort_gpu_bytes`, records the
    reason and calls `on_abort` once. GPU memory is the guard that matters for
    a leak: Metal buffers count there but not in the server's RSS.
    """

    def __init__(
        self,
        pgid: int | None = None,
        interval: float = 1.0,
        abort_load: float | None = None,
        abort_gpu_bytes: int | None = None,
        on_abort: Callable[[str], None] | None = None,
        probe: Callable[[int | None], dict] = probe,
    ):
        self.pgid = pgid
        self.interval = interval
        self.abort_load = abort_load
        self.abort_gpu_bytes = abort_gpu_bytes
        self.on_abort = on_abort
        self.probe = probe
        self.samples = 0
        self.peaks: dict[str, float | None] = dict.fromkeys(PEAKS)
        self.series: list[dict] = []
        self._started: float | None = None
        self.aborted: str | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def take(self) -> None:
        now = time.monotonic()
        if self._started is None:
            self._started = now
        sample = self.probe(self.pgid)
        self.samples += 1
        self.series.append(
            {"t_s": round(now - self._started, 2)}
            | {key: sample.get(key) for key in SERIES}
        )
        for key in PEAKS:
            value = sample.get(key)
            if value is not None and (
                self.peaks[key] is None or value > self.peaks[key]
            ):
                self.peaks[key] = value
        if self.aborted is not None:
            return
        load = sample.get("load_1m")
        gpu = sample.get("gpu_in_use_bytes")
        if self.abort_load is not None and load is not None and load > self.abort_load:
            self.aborted = f"1-min load {load:.2f} rose above {self.abort_load:g}"
        elif (
            self.abort_gpu_bytes is not None
            and gpu is not None
            and gpu > self.abort_gpu_bytes
        ):
            self.aborted = (
                f"GPU memory {gpu / 2**30:.1f} GiB rose above "
                f"{self.abort_gpu_bytes / 2**30:g} GiB"
            )
        if self.aborted and self.on_abort:
            self.on_abort(self.aborted)

    def _run(self) -> None:
        while True:
            started = time.monotonic()
            self.take()
            if self._stop.wait(max(0.0, self.interval - (time.monotonic() - started))):
                return

    def __enter__(self) -> Self:
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        self._thread.join()

    def summary(self) -> dict:
        return {
            "interval_s": self.interval,
            "samples": self.samples,
            **{f"peak_{key}": value for key, value in self.peaks.items()},
            "series": self.series,
            "aborted": self.aborted,
        }


# ---------------------------------------------------------------------------
# Idle gate

# Compilers from other jobs on the machine; one running skews every arm.
BUILD_PROCESSES = ("cargo", "rustc")


def parse_power_source(text: str) -> str | None:
    """The source `pmset -g batt` reports: 'AC Power' or 'Battery Power'."""
    match = re.search(r"Now drawing from '([^']+)'", text)
    return match.group(1) if match else None


def parse_thermal_warnings(text: str) -> list[str]:
    """Lines of `pmset -g therm` that report throttling.

    A cool machine prints only "Note: No ... has been recorded" lines. A
    recorded thermal or performance warning level, or a CPU speed or scheduler
    limit under 100, counts as a warning.
    """
    warnings = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("Note: No "):
            continue
        limit = re.match(r"CPU_(Speed|Scheduler)_Limit\s*=\s*(\d+)", line)
        if limit:
            if int(limit.group(2)) < 100:
                warnings.append(line)
        elif "warning level" in line.lower():
            warnings.append(line)
    return warnings


def build_processes(ps_text: str) -> list[str]:
    """Running compilers, from `ps -axo comm=` (full executable paths)."""
    return sorted(
        {
            Path(line.strip()).name
            for line in ps_text.splitlines()
            if Path(line.strip()).name in BUILD_PROCESSES
        }
    )


def idle_readings(gpu_samples: int = 5, interval: float = 1.0) -> dict:
    """Everything the idle gate looks at, recorded with the arm it gates."""
    utilization = []
    for index in range(gpu_samples):
        if index:
            time.sleep(interval)
        utilization.append(
            parse_gpu_stats(
                command_output(["ioreg", "-r", "-c", "AGXAccelerator", "-d", "1"])
            )["utilization_pct"]
        )
    return {
        "load_1m": round(os.getloadavg()[0], 2),
        "build_processes": build_processes(command_output(["ps", "-axo", "comm="])),
        "power_source": parse_power_source(command_output(["pmset", "-g", "batt"])),
        "thermal_warnings": parse_thermal_warnings(
            command_output(["pmset", "-g", "therm"])
        ),
        "thermal_pressure_level": thermal_pressure(),
        "gpu_utilization_pct": utilization,
    }


def idle_failures(
    readings: dict, max_load: float = 2.0, max_gpu_pct: int = 5
) -> list[str]:
    """Why the machine is not idle enough to measure; empty when it is."""
    failures = []
    if readings["load_1m"] >= max_load:
        failures.append(f"1-min load {readings['load_1m']} >= {max_load:g}")
    if readings["build_processes"]:
        failures.append(f"running: {', '.join(readings['build_processes'])}")
    if readings["power_source"] != "AC Power":
        failures.append(f"power source {readings['power_source']!r}, not AC")
    if readings["thermal_warnings"]:
        failures.append(f"thermal: {'; '.join(readings['thermal_warnings'])}")
    level = readings.get("thermal_pressure_level")
    if level is None:
        failures.append("thermal pressure unavailable")
    elif level > 0:
        name = THERMAL_PRESSURE[level] if level < len(THERMAL_PRESSURE) else "unknown"
        failures.append(f"thermal pressure {name} ({level})")
    gpu = readings["gpu_utilization_pct"]
    if not gpu or any(value is None for value in gpu):
        failures.append("GPU utilization unavailable")
    elif max(gpu) > max_gpu_pct:
        failures.append(f"GPU utilization up to {max(gpu)}% > {max_gpu_pct}%")
    return failures
