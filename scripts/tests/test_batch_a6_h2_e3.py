# SPDX-License-Identifier: GPL-2.0-or-later

from __future__ import annotations

import importlib.util
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1]
LOCK_PY = SCRIPTS / "lib" / "with_build_lock.py"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


LOCK = _load("with_build_lock", LOCK_PY)


class BuildLockTests(unittest.TestCase):
    def test_lock_path_lives_under_target(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            (repo / "Cargo.toml").write_text("[workspace]\n", encoding="utf-8")
            path = LOCK.lock_path_for(repo)
            self.assertEqual(path, repo / "target" / ".remu-build.lock")
            self.assertTrue((repo / "target").is_dir())

    def test_exclusive_holder_blocks_shared_until_release(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            (repo / "Cargo.toml").write_text("[workspace]\n", encoding="utf-8")
            lock_path = LOCK.lock_path_for(repo)
            exclusive = open(lock_path, "a+", encoding="utf-8")
            LOCK.acquire(exclusive, exclusive=True)
            shared_started = threading.Event()
            shared_acquired = threading.Event()

            def take_shared() -> None:
                shared_started.set()
                with open(lock_path, "a+", encoding="utf-8") as shared:
                    LOCK.acquire(shared, exclusive=False)
                    shared_acquired.set()

            thread = threading.Thread(target=take_shared)
            thread.start()
            self.assertTrue(shared_started.wait(2.0))
            time.sleep(0.2)
            self.assertFalse(
                shared_acquired.is_set(),
                "shared lock must wait while exclusive is held",
            )
            exclusive.close()
            self.assertTrue(shared_acquired.wait(2.0))
            thread.join(timeout=2.0)

    def test_cli_runs_command_under_lock(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            (repo / "Cargo.toml").write_text("[workspace]\n", encoding="utf-8")
            marker = Path(tmp) / "ran"
            result = subprocess.run(
                [
                    sys.executable,
                    str(LOCK_PY),
                    "--exclusive",
                    str(repo),
                    "--",
                    "sh",
                    "-c",
                    f"echo ok > '{marker}'",
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(marker.read_text(encoding="utf-8").strip(), "ok")
            self.assertIn("build lock", result.stderr)


class RemuReplayStubTests(unittest.TestCase):
    def test_replay_stub_refuses_loudly_with_ex_config(self) -> None:
        script = SCRIPTS / "remu-replay.py"
        result = subprocess.run(
            [sys.executable, str(script), "/tmp/does-not-need-to-exist", "--check"],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 78)
        self.assertIn("not implemented", result.stderr)
        self.assertIn("Command::Replay", result.stderr)
        self.assertIn("submissions.bin", result.stderr)

    def test_shell_wrapper_delegates(self) -> None:
        script = SCRIPTS / "remu-replay.sh"
        result = subprocess.run(
            ["sh", str(script), "/tmp/x"],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 78)
        self.assertIn("E3-STUB", result.stderr)


class SubmissionRecordStubTests(unittest.TestCase):
    def test_round_trip_empty_header(self) -> None:
        stub = _load(
            "record_stub",
            SCRIPTS / "scaffolding" / "submission-record" / "record_stub.py",
        )
        raw = stub.encode_header(count=0, flags=0)
        header = stub.decode_header(raw)
        self.assertEqual(header.version, 1)
        self.assertEqual(header.count, 0)
        self.assertTrue(raw.startswith(stub.MAGIC))

    def test_rejects_bad_magic(self) -> None:
        stub = _load(
            "record_stub",
            SCRIPTS / "scaffolding" / "submission-record" / "record_stub.py",
        )
        with self.assertRaises(stub.SubmissionFormatError):
            stub.decode_header(b"NOPESub1" + b"\x00" * 16)

    def test_init_empty_cli(self) -> None:
        script = SCRIPTS / "scaffolding" / "submission-record" / "record_stub.py"
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "submissions.bin"
            result = subprocess.run(
                [sys.executable, str(script), "init-empty", str(path)],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue(path.is_file())
            show = subprocess.run(
                [sys.executable, str(script), "show-header", str(path)],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(show.returncode, 0, show.stderr)
            self.assertIn("count=0", show.stdout)


class CriterionScaffoldingTests(unittest.TestCase):
    def test_bench_stubs_and_seams_doc_exist(self) -> None:
        root = SCRIPTS / "scaffolding" / "criterion-benches"
        for name in (
            "pm4_decode_per_mib.rs",
            "msl_translate_per_shader.rs",
            "write_watch_arm_fault_per_page.rs",
            "tiled_texture_decode_per_mib.rs",
            "SEAMS.md",
        ):
            self.assertTrue((root / name).is_file(), name)
        seams = (root / "SEAMS.md").read_text(encoding="utf-8")
        self.assertIn("Cargo.toml", seams)
        self.assertIn("deferred", seams.lower())
        for rs in root.glob("*.rs"):
            text = rs.read_text(encoding="utf-8")
            self.assertIn("compile_error!", text)


class PrepareRunnerScriptTests(unittest.TestCase):
    def test_prepare_script_is_executable_syntax(self) -> None:
        script = SCRIPTS / "lib" / "prepare-rosetta-runner.sh"
        result = subprocess.run(
            ["sh", "-n", str(script)],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_run_game_and_probes_parse(self) -> None:
        for name in ("run-game.sh", "probe-game.sh", "probe-taskgraph.sh", "remu-replay.sh"):
            result = subprocess.run(
                ["sh", "-n", str(SCRIPTS / name)],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, f"{name}: {result.stderr}")


if __name__ == "__main__":
    unittest.main()
