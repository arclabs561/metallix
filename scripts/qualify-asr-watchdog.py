#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Run a prebuilt transcribe_clips example on 1200 s of silence with sampled caps.

This qualifies completion of one synthetic long input, not speech accuracy or
an idle-machine performance result. GPU memory is a machine-wide ioreg total.
No model is downloaded and the binary is never built by this runner.
"""

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
import wave
from pathlib import Path

from bench_system import probe
from owned_process_group import (
    NativeSampler,
    cleanup,
    exited,
    footprint,
    members,
    require_wait_capabilities,
)


def main():
    require_wait_capabilities()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--abort-gpu-gib", type=float, required=True)
    parser.add_argument("--abort-footprint-gib", type=float, required=True)
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    args = parser.parse_args()
    if (
        not 0 < args.abort_gpu_gib < float("inf")
        or not 0 < args.abort_footprint_gib < float("inf")
        or args.timeout_seconds <= 0
    ):
        parser.error("memory and timeout caps must be finite and positive")
    sampler = NativeSampler()
    footprint_cap = int(args.abort_footprint_gib * 2**30)
    footprint_peak, footprint_samples = 0, 0
    identities = {}
    binary, model = args.binary.resolve(strict=True), args.model.resolve(strict=True)
    initial = probe(None).get("gpu_in_use_bytes")
    cap = int(args.abort_gpu_gib * 2**30)
    if initial is None or initial >= cap:
        parser.error("GPU memory probe unavailable or already at the cap")
    args.output.mkdir(parents=True, exist_ok=False)
    clips = args.output / "clips"
    clips.mkdir()
    with wave.open(str(clips / "long.wav"), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        for _ in range(1200):
            wav.writeframesraw(bytes(32000))
    (clips / "manifest.json").write_text(
        json.dumps({"clips": [{"id": "silence-1200", "wav": "long.wav"}]}) + "\n"
    )
    output = args.output / "transcripts.jsonl"
    command = [
        str(binary),
        str(model),
        str(clips.resolve()),
        str(output.resolve()),
        "--language",
        "English",
    ]
    reason, peak, returncode = None, initial, None
    started = time.monotonic()
    with (
        (args.output / "stdout.log").open("w") as stdout,
        (args.output / "stderr.log").open("w") as stderr,
    ):
        stop_signals = {signal.SIGTERM, signal.SIGINT, signal.SIGHUP}
        previous_mask = signal.pthread_sigmask(signal.SIG_BLOCK, stop_signals)
        process = None
        try:
            launch = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--exec-child",
                json.dumps([int(sig) for sig in previous_mask]),
                *command,
            ]
            process = subprocess.Popen(
                launch, stdout=stdout, stderr=stderr, start_new_session=True
            )
            signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)
            while True:
                cycle = time.monotonic()
                terminal = exited(process)
                if terminal is not None:
                    if any("Z" not in state for state in members(process.pid).values()):
                        reason = "orphaned descendants"
                    break
                physical = footprint(process, sampler, identities)
                if physical is None:
                    continue
                footprint_samples += 1
                footprint_peak = max(footprint_peak, physical)
                if physical >= footprint_cap:
                    reason = "physical footprint cap reached"
                    break
                reading = probe(None).get("gpu_in_use_bytes")
                if reading is None:
                    reason = "GPU memory probe unavailable"
                else:
                    peak = max(peak, reading)
                    if reading >= cap:
                        reason = "GPU memory cap reached"
                if time.monotonic() - started >= args.timeout_seconds:
                    reason = "wall timeout reached"
                if time.monotonic() - cycle > 1:
                    reason = "memory sampling exceeded one second"
                if reason:
                    break
                time.sleep(
                    max(
                        0, min(0.2, args.timeout_seconds - (time.monotonic() - started))
                    )
                )
        except (
            OSError,
            ValueError,
            RuntimeError,
            subprocess.SubprocessError,
            KeyboardInterrupt,
        ) as error:
            reason = f"guard failure: {type(error).__name__}"
        finally:
            # WNOWAIT retains the leader identity through every group signal,
            # including normal exit when a child has outlived that leader.
            signal.pthread_sigmask(signal.SIG_BLOCK, stop_signals)
            try:
                if process is not None:
                    cleanup(process, lambda *_args, **_kwargs: None)
                    returncode = process.returncode
            finally:
                signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)
    records = []
    if output.exists():
        try:
            records = [json.loads(line) for line in output.read_text().splitlines()]
        except json.JSONDecodeError:
            reason = "incomplete transcript receipt"
    passed = (
        reason is None
        and returncode == 0
        and footprint_samples > 0
        and len(records) == 1
        and records[0].get("id") == "silence-1200"
        and records[0].get("audio_seconds") == 1200
        and records[0].get("stopped") is True
    )
    with binary.open("rb") as handle:
        binary_sha256 = hashlib.file_digest(handle, "sha256").hexdigest()
    receipt = {
        "passed": passed,
        "reason": reason,
        "exit": returncode,
        "seconds": time.monotonic() - started,
        "gpu_peak_bytes": peak,
        "gpu_cap_bytes": cap,
        "footprint_peak_bytes": footprint_peak,
        "footprint_cap_bytes": footprint_cap,
        "footprint_samples": footprint_samples,
        "footprint_instrument": "proc_pid_rusage.RUSAGE_INFO_V2.ri_phys_footprint",
        "footprint_scope": "sum of owned process-group live members; sampled",
        "binary_sha256": binary_sha256,
    }
    (args.output / "summary.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt))
    return 0 if passed else 1


if __name__ == "__main__":
    if sys.argv[1:2] == ["--exec-child"]:
        signal.pthread_sigmask(signal.SIG_SETMASK, set(json.loads(sys.argv[2])))
        os.execvp(sys.argv[3], sys.argv[3:])

    def interrupted(signum, _frame):
        raise InterruptedError(f"watchdog received signal {signum}")

    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, interrupted)
    raise SystemExit(main())
