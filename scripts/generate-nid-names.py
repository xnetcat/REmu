#!/usr/bin/env python3
"""Generate the NID-to-symbol-name table used to label imports in diagnostics.

A PS5 import is identified only by an 11-character NID: the first eight bytes of
`SHA-1(symbol || suffix)`, little-endian, base64 encoded with `/` rewritten to `-`.  The
mapping is one-way, so a name can only be recovered by hashing a candidate symbol and
comparing.  This script hashes every symbol exported by a set of ELF modules and emits the
matches as a sorted Rust table.

The table carries names only.  It records no signature, layout, or behaviour, and it never
makes a NID dispatch anywhere -- an unimplemented import stays unimplemented, it just gets
reported by name instead of as an opaque NID.

The output is a binary table rather than Rust source because the recovered set runs to tens of
thousands of entries, and that many string literals costs far more build time than it is worth
for data the compiler never reasons about.

Usage:
    scripts/generate-nid-names.py <directory-of-elf-modules>... crates/remu-runtime/src/nid_names.bin

Any directory of PS5 ELF modules works: link stubs from an SDK you are licensed to use, or
the `sce_module` directory shipped inside a game.  Run it only when you have new modules to
scan; the generated table is committed so a normal build needs neither this script nor its
inputs.

Table layout, all integers little-endian:

    magic "REMUNID1"      8 bytes
    count                 u32     number of index records
    names_offset          u32     byte offset of the name blob
    index                 count * 16 bytes, sorted by NID for binary search:
                                  11 bytes NID, 1 byte name length, u32 offset into the blob
    names                 concatenated UTF-8 names, not separated
"""

import base64
import hashlib
import pathlib
import re
import struct
import sys

NID_SUFFIX = bytes.fromhex("518d64a635ded8c1e6b039b1c3e55230")
REPO = pathlib.Path(__file__).resolve().parent.parent
RUNTIME_SOURCE = REPO / "crates/remu-runtime/src/lib.rs"


def nid_of(name: str) -> str:
    """Returns the 11-character NID the loader will see for this symbol name."""
    digest = hashlib.sha1(name.encode() + NID_SUFFIX).digest()
    return base64.b64encode(struct.pack(">Q", struct.unpack("<Q", digest[:8])[0])).decode()[:11].replace("/", "-")


def sections(data: bytes):
    e_shoff, = struct.unpack_from("<Q", data, 0x28)
    e_shentsize, e_shnum = struct.unpack_from("<HH", data, 0x3A)
    if not e_shoff or not e_shnum:
        return []
    out = []
    for index in range(e_shnum):
        fields = struct.unpack_from("<IIQQQQIIQQ", data, e_shoff + index * e_shentsize)
        out.append({"type": fields[1], "offset": fields[4], "size": fields[5], "link": fields[6], "entsize": fields[9]})
    return out


def exported_symbols(path: pathlib.Path):
    """Yields every symbol name in an ELF module's symbol tables."""
    try:
        data = path.read_bytes()
    except OSError:
        return
    if data[:4] != b"\x7fELF" or data[4] != 2:
        return
    table = sections(data)
    for section in table:
        if section["type"] not in (2, 11) or not section["entsize"] or section["link"] >= len(table):
            continue
        strings = data[table[section["link"]]["offset"]:][:table[section["link"]]["size"]]
        for index in range(section["size"] // section["entsize"]):
            offset, = struct.unpack_from("<I", data, section["offset"] + index * section["entsize"])
            if not offset or offset >= len(strings):
                continue
            name = strings[offset:strings.find(b"\0", offset)].decode("utf-8", "replace")
            if name and "#" not in name and not name.startswith("$"):
                yield name


def already_named() -> set:
    """NIDs the hand-maintained table in the runtime already covers."""
    source = RUNTIME_SOURCE.read_text()
    start = source.index("pub fn known_nid_name(")
    return set(re.findall(r'"([A-Za-z0-9+\-]{11})" =>', source[start:source.index("\n}\n", start)]))


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__, file=sys.stderr)
        return 2
    *roots, destination = sys.argv[1:]

    names = set()
    for root in roots:
        for path in sorted(pathlib.Path(root).rglob("*")):
            if path.is_file():
                names.update(exported_symbols(path))

    known = already_named()
    table = [(nid, name) for nid, name in sorted({nid_of(name): name for name in names}.items())
             if nid not in known and len(name) < 256]

    index, blob = bytearray(), bytearray()
    for nid, name in table:
        encoded = name.encode()
        index += nid.encode() + bytes([len(encoded)]) + struct.pack("<I", len(blob))
        blob += encoded

    header = b"REMUNID1" + struct.pack("<II", len(table), 16 + len(index))
    pathlib.Path(destination).write_bytes(header + index + blob)
    print(f"{len(table)} names -> {destination} ({(len(header) + len(index) + len(blob)) / 1024:.0f} KiB)",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
