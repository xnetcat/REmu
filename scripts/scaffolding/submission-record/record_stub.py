#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later
"""Encode/decode scaffolding for submissions.bin (E3) — format only, no runtime wiring.

Loud about what it cannot do: it does not talk to agc_driver, does not capture guest memory,
and does not feed remu replay.
"""

from __future__ import annotations

import argparse
import struct
import sys
from dataclasses import dataclass
from pathlib import Path

MAGIC = b"REMUSub1"
VERSION = 1
HEADER_STRUCT = struct.Struct("<8sIIQ")  # magic, version, flags, count


@dataclass(frozen=True)
class SubmissionFileHeader:
    version: int
    flags: int
    count: int


class SubmissionFormatError(ValueError):
    pass


def encode_header(*, count: int = 0, flags: int = 0) -> bytes:
    if count < 0:
        raise SubmissionFormatError("count must be non-negative")
    return HEADER_STRUCT.pack(MAGIC, VERSION, flags, count)


def decode_header(data: bytes) -> SubmissionFileHeader:
    if len(data) < HEADER_STRUCT.size:
        raise SubmissionFormatError(
            f"header too short: {len(data)} < {HEADER_STRUCT.size}"
        )
    magic, version, flags, count = HEADER_STRUCT.unpack_from(data)
    if magic != MAGIC:
        raise SubmissionFormatError(f"bad magic {magic!r}; expected {MAGIC!r}")
    if version != VERSION:
        raise SubmissionFormatError(
            f"unsupported submissions.bin version {version}; stub only knows {VERSION}"
        )
    return SubmissionFileHeader(version=version, flags=flags, count=count)


def write_empty_corpus(path: Path) -> None:
    """Write a valid empty corpus so tools can detect the format without inventing submits."""
    path.write_bytes(encode_header(count=0, flags=0))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    init = sub.add_parser("init-empty", help="write an empty REMUSub1 file")
    init.add_argument("path", type=Path)
    show = sub.add_parser("show-header", help="decode and print the file header")
    show.add_argument("path", type=Path)
    args = parser.parse_args(argv)

    if args.cmd == "init-empty":
        write_empty_corpus(args.path)
        print(f"[REMu][E3-STUB] wrote empty submissions corpus: {args.path}")
        print(
            "[REMu][E3-STUB] recording is NOT wired in remu-runtime; this file is format scaffolding only",
            file=sys.stderr,
        )
        return 0

    header = decode_header(args.path.read_bytes())
    print(f"magic=REMUSub1 version={header.version} flags={header.flags} count={header.count}")
    if header.count == 0:
        print(
            "[REMu][E3-STUB] empty corpus — remu replay would still refuse until Command::Replay lands",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
