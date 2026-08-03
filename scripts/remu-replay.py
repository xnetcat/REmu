#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later
"""E3 / B23 harness stub: ``remu replay`` is not wired into remu-cli yet.

Plan (testing-observability L4 / MASTER-PLAN E3):
  submission recording → ``remu replay <dir>`` → frame goldens + ``--check`` / ``--bless``.

This script is the scripts/-side entry point so CI and humans have a loud, specific refusal
instead of a silent missing subcommand. When remu-cli gains ``Command::Replay``, this wrapper
should exec that binary; until then it exits 78 (EX_CONFIG) with the deferred seams listed.

Usage:
  remu-replay.py <run-dir> [--frames N] [--dump-frames OUT] [--check] [--bless]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

STUB_EXIT = 78  # EX_CONFIG — configuration incomplete, not a transient failure

DEFERRED_SEAMS = """
deferred seams (do not pretend these exist yet):
  1. remu-cli Command::Replay  — headless decode+translate+encode against synthetic guest RAM
  2. DeterminismMode::Recording in agc_driver_submit_* → $run_dir/submissions.bin
  3. Frame goldens under goldens/<hash>.png + FNV / perceptual --check / --bless
  4. CI Tier B step: remu replay --check over an out-of-repo corpus (plan H1, after E2)
see: scripts/scaffolding/e3-replay/README.md
     scripts/scaffolding/submission-record/FORMAT.md
""".strip()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="remu-replay",
        description="E3 stub for remu replay (not implemented — refuses loudly).",
    )
    parser.add_argument(
        "run_dir",
        type=Path,
        help="run directory containing submissions.bin (when recording exists)",
    )
    parser.add_argument("--frames", type=int, default=None, help="limit replayed frames")
    parser.add_argument("--dump-frames", type=Path, default=None, help="write frames to OUT/")
    parser.add_argument(
        "--check",
        action="store_true",
        help="compare against goldens/<hash>.png (not implemented)",
    )
    parser.add_argument(
        "--bless",
        action="store_true",
        help="regenerate goldens (not implemented)",
    )
    args = parser.parse_args(argv)

    print(
        "[REMu][E3-STUB] remu replay is not implemented yet — refusing rather than fabricating a pass",
        file=sys.stderr,
    )
    print(f"[REMu][E3-STUB] requested run_dir={args.run_dir}", file=sys.stderr)
    if args.frames is not None:
        print(f"[REMu][E3-STUB] --frames={args.frames}", file=sys.stderr)
    if args.dump_frames is not None:
        print(f"[REMu][E3-STUB] --dump-frames={args.dump_frames}", file=sys.stderr)
    if args.check:
        print("[REMu][E3-STUB] --check requested (golden compare not wired)", file=sys.stderr)
    if args.bless:
        print("[REMu][E3-STUB] --bless requested (golden write not wired)", file=sys.stderr)
    print(DEFERRED_SEAMS, file=sys.stderr)
    return STUB_EXIT


if __name__ == "__main__":
    raise SystemExit(main())
