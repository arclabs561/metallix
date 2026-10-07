# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Real filesystem and subprocess requirements for opt-in check isolation."""

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

import check_workspace as workspace

SCRIPTS = Path(__file__).resolve().parent


@unittest.skipUnless(os.name == "posix", "POSIX lease prototype")
class WorkspaceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name).resolve()
        self.root = self.base / "repo"
        self.root.mkdir()
        self.state = self.base / "state"
        subprocess.run(["git", "init", "-q", str(self.root)], check=True)
        (self.root / "scripts").mkdir()
        shutil.copyfile(
            SCRIPTS / "check_workspace.py", self.root / "scripts/check_workspace.py"
        )
        shutil.copyfile(SCRIPTS / "check.py", self.root / "scripts/check_runner.py")
        (self.root / ".gitignore").write_text("ignored\n")
        (self.root / "helper.py").write_text("VALUE = 7\n")
        (self.root / "scripts/check.py").write_text(
            "import sys\nfrom check_runner import run\n"
            "def run_checks(args, *, lease_fd=None):\n"
            "    return run([sys.executable, 'job.py'], args.timeout_seconds, lease_fd=lease_fd)\n"
        )
        self.job(
            "import helper, os\nfrom pathlib import Path\n"
            "assert helper.VALUE == 7\n"
            "assert os.environ.get('RUSTC_WRAPPER') == 'preserved-wrapper'\n"
            "cache = Path(os.environ['RUFF_CACHE_DIR']); cache.mkdir(parents=True, exist_ok=True)\n"
            "(cache / 'ruff-like-output').write_text('cache')\n"
            "(Path(os.environ['CARGO_TARGET_DIR']) / 'built').write_text('kept')\n"
        )
        subprocess.run(["git", "add", "."], cwd=self.root, check=True)
        self.environment = os.environ | {
            "METALLIX_CHECK_STATE_DIR": str(self.state),
            "RUSTC_WRAPPER": "preserved-wrapper",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
        for key in ["SLOT_DIR", "CARGO_TARGET_DIR"]:
            self.environment.pop(key, None)

    def job(self, text):
        (self.root / "job.py").write_text(text)

    def launch(self, *, timeout=5, queue_timeout=5):
        bootstrap = (
            "import argparse,sys; from pathlib import Path; "
            f"sys.path.insert(0, {str(SCRIPTS)!r}); import check_workspace as w; "
            "args=argparse.Namespace(metal=False, timeout_seconds=int(sys.argv[2]), "
            "queue_timeout_seconds=int(sys.argv[3])); "
            "sys.exit(w.run_isolated(Path(sys.argv[1]), args))"
        )
        process = subprocess.Popen(
            [
                sys.executable,
                "-c",
                bootstrap,
                str(self.root),
                str(timeout),
                str(queue_timeout),
            ],
            env=self.environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        self.addCleanup(self.stop, process)
        return process

    @staticmethod
    def stop(process):
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=4)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        if process.stdout:
            process.stdout.close()
        if process.stderr:
            process.stderr.close()

    def finish(self, process, expected=0):
        output, errors = process.communicate(timeout=12)
        self.assertEqual(process.returncode, expected, output + errors)
        return output + errors

    def wait_for(self, path):
        deadline = time.monotonic() + 5
        while not path.exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        self.assertTrue(path.exists(), f"missing child marker: {path}")

    def test_two_real_import_and_cache_runs_preserve_target_and_source_mtime(self):
        self.finish(self.launch())
        source = self.state / "lane/src/helper.py"
        modified = source.stat().st_mtime_ns
        self.finish(self.launch())
        self.assertEqual(source.stat().st_mtime_ns, modified)
        self.assertFalse(list((self.state / "lane/src").rglob("__pycache__")))
        self.assertTrue((self.state / "lane/cache/ruff/ruff-like-output").exists())
        self.assertEqual((self.state / "lane/target/built").read_text(), "kept")
        sentinel = self.state / "lane/src/unowned"
        sentinel.write_text("keep")
        self.finish(self.launch(), 1)
        self.assertEqual(sentinel.read_text(), "keep")

    def test_current_bytes_modes_deletions_and_ignored_input(self):
        (self.root / "helper.py").write_text("VALUE = 8\n")
        (self.root / "new.py").write_text("untracked\n")
        (self.root / "new.py").chmod(0o755)
        (self.root / "ignored").write_text("private")
        (self.root / "job.py").unlink()
        snapshots = self.base / "snapshots"
        snapshots.mkdir()
        capture, entries = workspace.capture(self.root, snapshots)
        self.assertEqual((capture / "helper.py").read_text(), "VALUE = 8\n")
        self.assertEqual(entries["new.py"][0], 0o755)
        self.assertNotIn("ignored", entries)
        self.assertNotIn("job.py", entries)
        (self.root / "escape").symlink_to(self.base)
        with self.assertRaisesRegex(ValueError, "symlink"):
            workspace.manifest(self.root)

    def test_successful_resync_removes_only_prior_owned_deleted_source(self):
        self.finish(self.launch())
        (self.root / "helper.py").unlink()
        self.job("pass\n")
        self.finish(self.launch())
        self.assertFalse((self.state / "lane/src/helper.py").exists())
        self.assertTrue((self.state / "lane/target/built").exists())

    def test_failure_and_changed_caller_have_honest_receipt(self):
        self.job("import sys; sys.exit(7)\n")
        self.finish(self.launch(), 7)
        receipts = list((self.state / "requests").glob("*/receipt.json"))
        first_receipt = receipts[0]
        retained = first_receipt.read_bytes()
        first = json.loads(retained)
        self.assertEqual(first["exit_code"], 7)
        self.assertEqual(
            first["source_digest"], workspace.digest(workspace.manifest(self.root))
        )
        self.assertFalse((self.state / "lane/incomplete").exists())
        self.job("pass\n")
        self.finish(self.launch())
        receipts = list((self.state / "requests").glob("*/receipt.json"))
        self.assertEqual(len(receipts), 2)
        self.assertEqual(first_receipt.read_bytes(), retained)
        second_receipt = next(path for path in receipts if path != first_receipt)
        second = json.loads(second_receipt.read_text())
        self.assertEqual(second["exit_code"], 0)
        self.assertEqual(
            second["source_digest"], workspace.digest(workspace.manifest(self.root))
        )
        self.assertNotEqual(first["source_digest"], second["source_digest"])
        self.assertNotEqual(first_receipt.parent, second_receipt.parent)
        self.assertTrue((first_receipt.parent / "source/job.py").exists())
        self.assertTrue((second_receipt.parent / "source/job.py").exists())

    def sleeping_job(self):
        self.job(
            "import os,time\nfrom pathlib import Path\n"
            "target=Path(os.environ['CARGO_TARGET_DIR'])\n"
            "(target/'started').write_text(str(os.getpid()))\n"
            "while not (target/'release').exists(): time.sleep(.02)\n"
        )
        return self.state / "lane/target/started"

    def test_waiter_cannot_overlap_and_queue_timeout_does_not_cancel_holder(self):
        marker = self.sleeping_job()
        first = self.launch()
        self.wait_for(marker)
        second = self.launch(queue_timeout=1)
        self.assertIn("queue timeout", self.finish(second, 1))
        self.assertIsNone(first.poll())
        (marker.parent / "release").touch()
        self.finish(first)

    def test_lease_survives_controller_sigkill_until_command_finishes(self):
        marker = self.sleeping_job()
        first = self.launch()
        self.wait_for(marker)
        first.kill()
        first.wait(timeout=2)
        second = self.launch(queue_timeout=1)
        self.assertIn("queue timeout", self.finish(second, 1))
        (marker.parent / "release").touch()
        # Even after the orphan closes its lease, incomplete state needs review.
        self.assertIn("incomplete prior", self.finish(self.launch(), 1))
        self.assertTrue((self.state / "lane/incomplete").exists())

    def test_term_and_command_timeout_stop_running_command_before_lane_reuse(self):
        marker = self.sleeping_job()
        first = self.launch(timeout=1)
        self.wait_for(marker)
        pid = int(marker.read_text())
        self.finish(first, 124)
        with self.assertRaises(ProcessLookupError):
            os.kill(pid, 0)
        marker.unlink()
        second = self.launch()
        self.wait_for(marker)
        second.send_signal(signal.SIGTERM)
        self.finish(second, 143)
        (marker.parent / "release").touch()
        self.finish(self.launch())

    def test_conflicting_target_nested_slot_and_ignored_config_are_rejected(self):
        self.environment["CARGO_TARGET_DIR"] = str(self.base / "other-target")
        self.assertIn("conflicts", self.finish(self.launch(), 1))
        self.environment.pop("CARGO_TARGET_DIR")
        self.environment["SLOT_DIR"] = str(self.base / "legacy")
        self.assertIn("legacy slot", self.finish(self.launch(), 1))
        self.environment.pop("SLOT_DIR")
        (self.root / ".cargo").mkdir()
        (self.root / ".cargo/config.toml").write_text("[build]\n")
        (self.root / ".gitignore").write_text(".cargo/\n")
        self.assertIn("explicit input policy", self.finish(self.launch(), 1))

    def test_unknown_empty_directory_is_preserved_and_rejected(self):
        self.finish(self.launch())
        sentinel = self.state / "lane/src/unowned-directory"
        sentinel.mkdir()
        self.finish(self.launch(), 1)
        self.assertTrue(sentinel.is_dir())

    def test_cancelled_waiter_does_not_cancel_active_gate(self):
        marker = self.sleeping_job()
        first = self.launch()
        self.wait_for(marker)
        second = self.launch()
        second.terminate()
        second.communicate(timeout=3)
        self.assertIsNone(first.poll())
        (marker.parent / "release").touch()
        self.finish(first)

    def test_caller_edit_after_capture_marks_receipt_without_changing_tested_source(
        self,
    ):
        marker = self.sleeping_job()
        process = self.launch()
        self.wait_for(marker)
        (self.root / "helper.py").write_text("VALUE = 999\n")
        (marker.parent / "release").touch()
        self.finish(process)
        receipt = next((self.state / "requests").glob("*/receipt.json"))
        self.assertTrue(json.loads(receipt.read_text())["source_changed_since_capture"])
        self.assertEqual((self.state / "lane/src/helper.py").read_text(), "VALUE = 7\n")

    def test_corrupt_prior_manifest_never_deletes_unrelated_sentinel(self):
        self.finish(self.launch())
        sentinel = self.base / "outside-sentinel"
        sentinel.write_text("preserve")
        manifest_path = self.state / "lane/manifest.json"
        valid = manifest_path.read_text()
        malformed = [
            {str(sentinel): [0o644, "0" * 64]},
            {"../../../outside-sentinel": [0o644, "0" * 64]},
            {"helper.py": ["644", "0" * 64]},
            {"helper.py": [0o644, "not-a-digest"]},
            [],
        ]
        for entries in malformed:
            with self.subTest(entries=entries):
                manifest_path.write_text(json.dumps(entries))
                self.assertIn("invalid lane manifest", self.finish(self.launch(), 1))
                self.assertEqual(sentinel.read_text(), "preserve")
        manifest_path.write_text(valid)
        self.finish(self.launch())

    def test_term_reaps_real_child_group_before_lane_reuse(self):
        self.job(
            "import os,signal,subprocess,sys,time\nfrom pathlib import Path\n"
            "target=Path(os.environ['CARGO_TARGET_DIR'])\n"
            "child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(20)'])\n"
            "def stop(sig, frame):\n"
            "    child.wait(timeout=2); sys.exit(128+sig)\n"
            "signal.signal(signal.SIGTERM,stop)\n"
            "(target/'child').write_text(str(child.pid))\n"
            "while True: time.sleep(.02)\n"
        )
        first = self.launch()
        marker = self.state / "lane/target/child"
        self.wait_for(marker)
        child_pid = int(marker.read_text())
        first.terminate()
        self.finish(first, 143)
        with self.assertRaises(ProcessLookupError):
            os.kill(child_pid, 0)
        self.job("pass\n")
        self.finish(self.launch())

    def test_abrupt_command_death_leaves_lane_blocked_even_after_lease_closes(self):
        self.job("import os,signal; os.kill(os.getpid(), signal.SIGKILL)\n")
        self.assertIn("completed cleanup", self.finish(self.launch(), 1))
        self.job("pass\n")
        self.assertIn("incomplete prior", self.finish(self.launch(), 1))
        self.assertTrue((self.state / "lane/incomplete").exists())

    def test_mode_precedence(self):
        args = argparse.Namespace(isolated=False, direct=False)
        self.assertFalse(workspace.selected(args, {}))
        configured = {"METALLIX_CHECK_STATE_DIR": "/cache"}
        self.assertTrue(workspace.selected(args, configured))
        self.assertFalse(workspace.selected(args, configured | {"CI": "true"}))
        args.direct = True
        self.assertFalse(workspace.selected(args, configured))
        args.direct, args.isolated = False, True
        self.assertTrue(workspace.selected(args, {"CI": "true"}))


if __name__ == "__main__":
    unittest.main()
