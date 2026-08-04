# SPDX-License-Identifier: GPL-2.0-or-later

from __future__ import annotations

import subprocess
import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
SCRIPTS = REPO / "scripts"


class FfmpegBuildIntegrationTests(unittest.TestCase):
    def test_ffmpeg_builder_is_valid_posix_shell(self) -> None:
        script = SCRIPTS / "lib" / "ensure-ffmpeg.sh"
        result = subprocess.run(
            ["sh", "-n", str(script)],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_cargo_build_never_compiles_the_libav_stub(self) -> None:
        build_rs = (REPO / "crates" / "remu-runtime" / "build.rs").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("REMU_LIBAV_STUB", build_rs)
        self.assertIn("ensure-ffmpeg.sh", build_rs)

    def test_cold_ffmpeg_build_reserves_stdout_for_the_install_path(self) -> None:
        script = (SCRIPTS / "lib" / "ensure-ffmpeg.sh").read_text(encoding="utf-8")
        self.assertIn(") >&2\n\n", script)
        self.assertIn('make -C "$object_dir" -j"$build_jobs" >&2', script)
        self.assertIn('make -C "$object_dir" install >&2', script)

    def test_rosetta_packager_resolves_the_repo_level_bundle_script(self) -> None:
        prepare = (SCRIPTS / "lib" / "prepare-rosetta-runner.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn('$repo_dir/scripts/build-macos-app.sh', prepare)
        self.assertNotIn('$script_dir/build-macos-app.sh', prepare)


if __name__ == "__main__":
    unittest.main()
