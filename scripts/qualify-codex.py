#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Dry-run-first qualification of Codex's native command tool against local mx."""

from __future__ import annotations

import argparse
import hashlib
import ipaddress
import json
import os
import signal
import subprocess
import tempfile
import time
from pathlib import Path
from urllib.parse import urlparse


def positive(value: str) -> int:
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def loopback_http_url(value: str) -> str:
    """Accept only a credential-free HTTP URL with a numeric loopback host."""
    parsed = urlparse(value)
    if (
        parsed.scheme != "http"
        or not parsed.hostname
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
    ):
        raise argparse.ArgumentTypeError(
            "URL must be a credential-free loopback http URL"
        )
    try:
        is_loopback = ipaddress.ip_address(parsed.hostname).is_loopback
    except ValueError:
        is_loopback = False
    if not is_loopback:
        raise argparse.ArgumentTypeError("URL host must be a numeric loopback address")
    try:
        port = parsed.port
    except ValueError as error:
        raise argparse.ArgumentTypeError("URL port is invalid") from error
    if port is None:
        raise argparse.ArgumentTypeError("URL must include an explicit port")
    return value.rstrip("/")


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def workspace_snapshot(workspace: Path) -> tuple[dict[str, str] | None, str | None]:
    """Hash only regular workspace files; report unexpected entries without aborting."""
    snapshot: dict[str, str] = {}
    try:
        for path in workspace.iterdir():
            if path.is_symlink() or not path.is_file():
                return None, f"workspace contains non-file entry: {path.name}"
            snapshot[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as error:
        return None, str(error)
    return snapshot, None


def toml_string(value: str) -> str:
    # JSON strings are valid TOML basic strings and avoid shell interpolation.
    return json.dumps(value)


def qualification_prompt() -> str:
    """Ask for the fixture's value without placing it in the request itself."""
    return (
        "Read facts.txt with the native command tool. Find its qualification_value, "
        "then reply exactly QUALIFIED:<the qualification_value you read> and no other text."
    )


def codex_command(
    binary: str,
    url: str,
    model_id: str,
    workspace: Path,
    instructions: Path,
    prompt: str,
) -> list[str]:
    provider = "metallix_qualification"
    provider_config = (
        f"model_providers.{provider}={{name={toml_string('Metallix qualification')},"
        f'base_url={toml_string(url)},wire_api="responses",'
        "requires_openai_auth=false,supports_websockets=false}"
    )
    configs = [
        f"model_provider={toml_string(provider)}",
        f"model={toml_string(model_id)}",
        "model_context_window=16384",
        "project_doc_max_bytes=0",
        f"model_instructions_file={toml_string(str(instructions))}",
        "features.hooks=false",
        "features.apps=false",
        "features.multi_agent=false",
        "features.remote_plugin=false",
        "features.shell_snapshot=false",
        'web_search="disabled"',
        provider_config,
    ]
    command = [
        binary,
        "exec",
        "--ignore-user-config",
        "--ephemeral",
        "--skip-git-repo-check",
        "--sandbox",
        "read-only",
        "--json",
        "--cd",
        str(workspace),
    ]
    for config in configs:
        command.extend(("-c", config))
    command.append(prompt)
    return command


def execute(
    command: list[str], timeout: int
) -> tuple[int | None, str, str, bool, str | None]:
    """Run an owned process group and always reap it after a timeout."""
    try:
        process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
    except OSError as error:
        return None, "", "", False, str(error)
    try:
        stdout, stderr = process.communicate(timeout=timeout)
        return process.returncode, stdout, stderr, False, None
    except (subprocess.TimeoutExpired, KeyboardInterrupt) as error:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        stdout, stderr = process.communicate()
        if isinstance(error, KeyboardInterrupt):
            raise
        return None, stdout, stderr, True, None


def parse_events(stdout: str) -> tuple[list[dict], str | None]:
    events: list[dict] = []
    for line in stdout.splitlines():
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            return [], "malformed JSONL event"
        if not isinstance(event, dict) or not isinstance(event.get("type"), str):
            return [], "unsupported event shape"
        events.append(event)
    if not events:
        return [], "no JSONL events"
    return events, None


def assess(
    exit_code: int | None, timed_out: bool, stdout: str, value: str, marker: str
) -> dict:
    """Require a completed native command execution and the final exact response."""
    events, parse_error = parse_events(stdout)
    command_outputs: list[str] = []
    assistant_messages: list[str] = []
    command_before_final_marker = False
    turn_completed = False
    terminal_event_is_last = False
    unsupported_shape = parse_error
    if parse_error is None:
        for event in events:
            if event["type"] not in {
                "thread.started",
                "turn.started",
                "item.started",
                "item.updated",
                "item.completed",
                "turn.completed",
                "turn.failed",
                "error",
            }:
                unsupported_shape = "unsupported event type"
                break
            if event["type"] in {"turn.failed", "error"}:
                unsupported_shape = "unsuccessful turn event"
                break
            if event["type"] == "turn.completed":
                if event is not events[-1]:
                    unsupported_shape = "turn.completed is not terminal"
                    break
                turn_completed = True
                terminal_event_is_last = True
                continue
            if event["type"] != "item.completed":
                continue
            item = event.get("item")
            if not isinstance(item, dict) or not isinstance(item.get("type"), str):
                unsupported_shape = "unsupported item.completed shape"
                break
            if item["type"] == "command_execution":
                output = item.get("aggregated_output")
                if (
                    not isinstance(output, str)
                    or type(item.get("exit_code")) is not int
                    or item["exit_code"] != 0
                ):
                    unsupported_shape = "unsupported command_execution shape"
                    break
                command_outputs.append(output)
            elif item["type"] == "agent_message":
                text = item.get("text")
                if not isinstance(text, str):
                    unsupported_shape = "unsupported agent_message shape"
                    break
                assistant_messages.append(text)
                if text == marker and command_outputs:
                    command_before_final_marker = True
            elif item["type"] not in {"reasoning", "error", "todo_list"}:
                unsupported_shape = "unsupported completed item type"
                break
    checks = {
        "process_success": exit_code == 0 and not timed_out,
        "jsonl_valid": parse_error is None,
        "supported_event_shape": unsupported_shape is None,
        "successful_command_execution": any(
            value in output for output in command_outputs
        ),
        "command_precedes_final_marker": command_before_final_marker,
        "final_assistant_marker": bool(assistant_messages)
        and assistant_messages[-1] == marker,
        "turn_completed": turn_completed and terminal_event_is_last,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "event_count": len(events),
        "command_execution_count": len(command_outputs),
        "assistant_message_count": len(assistant_messages),
        "parse_error": unsupported_shape,
    }


def ensure_empty_output(output: Path, parser: argparse.ArgumentParser) -> None:
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        parser.error("--output must name a new or empty directory")
    output.mkdir(parents=True, exist_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run", action="store_true", help="run Codex; default prints plan"
    )
    parser.add_argument("--url", type=loopback_http_url, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--codex-binary", default="codex")
    parser.add_argument("--timeout-seconds", type=positive, default=120)
    parser.add_argument("--repeats", type=positive, default=3)
    args = parser.parse_args()
    plan = {
        "schema_version": 1,
        "status": "dry_run",
        "scope": "Native Codex command-execution qualification against a loopback Responses endpoint",
        "url": args.url,
        "model_id": args.model_id,
        "repeats": args.repeats,
        "timeout_seconds": args.timeout_seconds,
        "fixture": "unique facts.txt in a synthetic output-owned workspace",
    }
    if not args.run:
        print(json.dumps(plan, indent=2))
        return 0

    ensure_empty_output(args.output, parser)
    args.output = args.output.resolve()
    receipt: dict = {**plan, "status": "running", "trials": []}
    receipt_path = args.output / "receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    try:
        (
            version_status,
            version_stdout,
            version_stderr,
            version_timeout,
            version_error,
        ) = execute([args.codex_binary, "--version"], args.timeout_seconds)
        (args.output / "codex-version.stdout.log").write_text(version_stdout)
        (args.output / "codex-version.stderr.log").write_text(version_stderr)
        if version_status != 0 or version_timeout or version_error:
            raise RuntimeError("could not obtain Codex CLI version")
        receipt["codex_version_sha256"] = sha256_text(version_stdout)
        for repeat in range(1, args.repeats + 1):
            workspace = Path(
                tempfile.mkdtemp(prefix=f"trial-{repeat}-", dir=args.output)
            )
            value = f"FACT-{os.urandom(16).hex()}"
            marker = f"QUALIFIED:{value}"
            facts = workspace / "facts.txt"
            facts.write_text(f"qualification_value={value}\n")
            instructions = workspace / "qualification-instructions.md"
            instructions.write_text(
                "Use the native command tool when the request asks you to read a file.\n"
            )
            before, before_error = workspace_snapshot(workspace)
            if before is None:
                raise RuntimeError(before_error or "could not snapshot workspace")
            input_sha256 = hashlib.sha256(facts.read_bytes()).hexdigest()
            prompt = qualification_prompt()
            command = codex_command(
                args.codex_binary,
                args.url,
                args.model_id,
                workspace,
                instructions,
                prompt,
            )
            started = time.monotonic()
            exit_code, stdout, stderr, timed_out, launch_error = execute(
                command, args.timeout_seconds
            )
            label = f"trial-{repeat}"
            (args.output / f"{label}.stdout.log").write_text(stdout)
            (args.output / f"{label}.stderr.log").write_text(stderr)
            after, workspace_error = workspace_snapshot(workspace)
            row = {
                "repeat": repeat,
                "workspace": workspace.name,
                "exit_code": exit_code,
                "timed_out": timed_out,
                "launch_error": launch_error,
                "wall_ms": (time.monotonic() - started) * 1000,
                "input_sha256": input_sha256,
                "argv_sha256": sha256_text(json.dumps(command, separators=(",", ":"))),
                "cli_version_argv_sha256": sha256_text(
                    version_stdout + "\0" + json.dumps(command, separators=(",", ":"))
                ),
                "workspace_unchanged": before == after and workspace_error is None,
                "workspace_error": workspace_error,
                "stdout_log": f"{label}.stdout.log",
                "stderr_log": f"{label}.stderr.log",
                **assess(exit_code, timed_out, stdout, value, marker),
            }
            receipt["trials"].append(row)
            receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
            print(f"trial {repeat}: {'pass' if row['passed'] else 'FAIL'}", flush=True)
        receipt["status"] = (
            "passed"
            if all(
                row["passed"] and row["workspace_unchanged"]
                for row in receipt["trials"]
            )
            else "failed"
        )
        receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
        return 0 if receipt["status"] == "passed" else 1
    except (OSError, RuntimeError, KeyboardInterrupt) as error:
        receipt["status"] = (
            "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        )
        receipt["error"] = str(error) or "interrupted by operator"
        receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
        return 130 if isinstance(error, KeyboardInterrupt) else 1


if __name__ == "__main__":
    raise SystemExit(main())
