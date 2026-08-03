#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later
"""Serialize REmu release rebuilds across parallel run-game / probe sessions (plan H2).

macOS does not ship a ``flock(1)`` utility, so this helper wraps ``fcntl.flock`` around a
lock file under ``<repo>/target/.remu-build.lock`` and then execs the remaining command.

Modes:
  --exclusive  writers (cargo build / shared app-bundle install)
  --shared     readers (REMU_SKIP_BUILD=1 launches): wait for any exclusive holder, then
               proceed concurrently with other readers

Usage:
  with_build_lock.py --exclusive|--shared <repo_dir> -- <command> [args...]
"""

from __future__ import annotations

import argparse
import fcntl
import os
import sys
from pathlib import Path


def lock_path_for(repo_dir: Path) -> Path:
    target = repo_dir / "target"
    target.mkdir(parents=True, exist_ok=True)
    return target / ".remu-build.lock"


def acquire(lock_file, exclusive: bool) -> None:
    flag = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
    fcntl.flock(lock_file.fileno(), flag)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--exclusive",
        action="store_true",
        help="exclusive lock for cargo build / shared bundle install",
    )
    mode.add_argument(
        "--shared",
        action="store_true",
        help="shared lock so SKIP_BUILD launches wait for in-progress builds",
    )
    parser.add_argument("repo_dir", type=Path)
    parser.add_argument(
        "command",
        nargs=argparse.REMAINDER,
        help="command to run under the lock; pass after --",
    )
    args = parser.parse_args(argv)
    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        parser.error("missing command after --")

    repo_dir = args.repo_dir.resolve()
    if not (repo_dir / "Cargo.toml").is_file():
        print(f"error: not a REmu repo root: {repo_dir}", file=sys.stderr)
        return 66

    path = lock_path_for(repo_dir)
    # Keep the lock file open for the lifetime of this process; flock is released on close/exit.
    lock_file = open(path, "a+", encoding="utf-8")
    try:
        mode_name = "exclusive" if args.exclusive else "shared"
        print(f"[REMu] acquiring {mode_name} build lock: {path}", file=sys.stderr)
        acquire(lock_file, exclusive=args.exclusive)
        print(f"[REMu] held {mode_name} build lock", file=sys.stderr)
        os.execvp(command[0], command)
    finally:
        # Only reached if execvp fails.
        lock_file.close()
    return 127


if __name__ == "__main__":
    raise SystemExit(main())
