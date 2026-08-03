#!/usr/bin/env python3
"""Generate the NID-to-symbol-name table used to label imports in diagnostics.

A PS5 import is identified only by an 11-character NID.  For almost every export that NID is
the first eight bytes of `SHA-1(symbol || suffix)`, little-endian, base64 encoded with `/`
rewritten to `-`, and since the mapping is one-way a name is recovered by hashing a candidate
symbol and comparing.  This script hashes every symbol exported by a set of ELF modules and
emits the matches as a sorted Rust table.

CAUTION — the hash relation is not universal, and this script assumes it is.  A small number of
exports carry hand-assigned ("pinned") NIDs that are not the hash of their own symbol name: 16
of them in SDK 10.00, 15 in libSceAgc alone, including `sceAgcInit`, `sceAgcDcbSetIndexSize`,
`sceAgcDcbContextStateOp`, `sceAgcGetDataPacketPayloadAddress`, `sceAgcFuseShaderHalves` and
`sceAgcGetFusedShaderSize`.  For those, `nid_of(name)` returns an identifier no binary ever
uses, so this script emits a phantom entry and silently omits the real one.  Hashing therefore
cannot recover them and must not be used to "verify" them either: applied naively to the
hand-maintained table it flags correct pinned entries as wrong.

The authoritative pairing is in the stub library itself — a `.scenid` section holding one
8-byte NID per `.dynsym` entry — which is the only source that can name a pinned export.  This
script reads that section (`scenid_pairs`): stub-named pairs override hash-derived ones, and a
name the stubs pair is never hashed, so no phantom hash NID is emitted beside a pinned export.
Verified against SDK 10.00: 50,206 pairs across all 128 stubs, 16 pinned, in full agreement
with the independently derived `abi-inventory/sdk10-nid-map.tsv`.

The table carries names only.  It records no signature, layout, or behaviour, and it never
makes a NID dispatch anywhere -- an unimplemented import stays unimplemented, it just gets
reported by name instead of as an opaque NID.

The output is a binary table rather than Rust source because the recovered set runs to tens of
thousands of entries, and that many string literals costs far more build time than it is worth
for data the compiler never reasons about.

Usage:
    scripts/generate-nid-names.py <directory-of-elf-modules>... crates/remu-runtime/src/nid_names.bin
    scripts/generate-nid-names.py --force <directory>... crates/remu-runtime/src/nid_names.bin

Inputs must expose plaintext C/C++ export names in ELF symbol tables (typically SDK link stubs
you are licensed to use). Game `sce_module` SELF containers usually ship NID-encoded dynamic
names (`nid#lib#ver`) and zero section headers, so scanning them yields nothing useful — use
decrypted/raw ELF dumps with real symbol names instead.  Run this only when you have new modules
to scan; the generated table is committed so a normal build needs neither this script nor its
inputs.

By default the script refuses to overwrite an existing non-empty table with zero recovered names
(so a game `sce_module` scan cannot wipe the committed blob). Pass `--force` only when you
intentionally want that.

Table layout, all integers little-endian:

    magic "REMUNID1"      8 bytes
    count                 u32     number of index records
    names_offset          u32     byte offset of the name blob
    index                 count * 16 bytes, sorted by NID for binary search:
                                  11 bytes NID, 1 byte name length, u32 offset into the blob
    names                 concatenated UTF-8 names, not separated
"""

from __future__ import annotations

import base64
import hashlib
import pathlib
import re
import struct
import sys

NID_SUFFIX = bytes.fromhex("518d64a635ded8c1e6b039b1c3e55230")
REPO = pathlib.Path(__file__).resolve().parent.parent
NID_TABLE_SOURCE = REPO / "crates/remu-runtime/src/nid_table.rs"
RUNTIME_LIB_SOURCE = REPO / "crates/remu-runtime/src/lib.rs"
PS5_CONTAINER_MAGIC = bytes.fromhex("4f153d1d")
PS5_RETAIL_CONTAINER_MAGIC = bytes.fromhex("5414f5ee")


def nid_of(name: str) -> str:
    """Returns the 11-character NID the loader will see for this symbol name."""
    digest = hashlib.sha1(name.encode() + NID_SUFFIX).digest()
    return base64.b64encode(struct.pack(">Q", struct.unpack("<Q", digest[:8])[0])).decode()[:11].replace("/", "-")


def elf_image(data: bytes) -> bytes | None:
    """Return a view of the ELF64 image, unwrapping a PS5 SELF container when needed."""

    if data[:4] == b"\x7fELF":
        return data if len(data) > 4 and data[4] == 2 else None
    if data[:4] not in (PS5_CONTAINER_MAGIC, PS5_RETAIL_CONTAINER_MAGIC) or len(data) < 32:
        return None
    segment_count = struct.unpack_from("<H", data, 0x18)[0]
    elf_offset = 32 + segment_count * 32
    if elf_offset + 5 > len(data) or data[elf_offset : elf_offset + 4] != b"\x7fELF":
        return None
    if data[elf_offset + 4] != 2:
        return None
    return data[elf_offset:]


def sections(data: bytes):
    e_shoff, = struct.unpack_from("<Q", data, 0x28)
    e_shentsize, e_shnum = struct.unpack_from("<HH", data, 0x3A)
    if not e_shoff or not e_shnum:
        return []
    out = []
    for index in range(e_shnum):
        fields = struct.unpack_from("<IIQQQQIIQQ", data, e_shoff + index * e_shentsize)
        out.append({"name_offset": fields[0], "type": fields[1], "offset": fields[4],
                    "size": fields[5], "link": fields[6], "entsize": fields[9]})
    return out


def exported_symbols(path: pathlib.Path):
    """Yields every symbol name in an ELF module's symbol tables."""
    try:
        data = path.read_bytes()
    except OSError:
        return
    image = elf_image(data)
    if image is None:
        return
    table = sections(image)
    for section in table:
        if section["type"] not in (2, 11) or not section["entsize"] or section["link"] >= len(table):
            continue
        strings = image[table[section["link"]]["offset"]:][:table[section["link"]]["size"]]
        for index in range(section["size"] // section["entsize"]):
            offset, = struct.unpack_from("<I", image, section["offset"] + index * section["entsize"])
            if not offset or offset >= len(strings):
                continue
            name = strings[offset:strings.find(b"\0", offset)].decode("utf-8", "replace")
            if name and "#" not in name and not name.startswith("$"):
                yield name


def scenid_pairs(path: pathlib.Path):
    """Yields authoritative (nid, name) pairs from a link stub's `.scenid` section.

    A stub carries one 8-byte NID per `.dynsym` entry, parallel arrays at shift zero.  This is
    the only oracle that can name a pinned export — a NID that is not the hash of its own
    symbol name — so pairs from here override hash-derived pairs.
    """
    try:
        data = path.read_bytes()
    except OSError:
        return
    image = elf_image(data)
    if image is None:
        return
    table = sections(image)
    if not table:
        return
    shstrndx, = struct.unpack_from("<H", image, 0x3E)
    if shstrndx >= len(table):
        return
    shstr = image[table[shstrndx]["offset"]:][: table[shstrndx]["size"]]

    def section_name(section) -> str:
        offset = section["name_offset"]
        if offset >= len(shstr):
            return ""
        return shstr[offset : shstr.find(b"\0", offset)].decode("utf-8", "replace")

    scenid = next((section for section in table if section_name(section) == ".scenid"), None)
    dynsym = next((section for section in table if section["type"] == 11), None)
    if (
        scenid is None
        or dynsym is None
        or not dynsym["entsize"]
        or dynsym["link"] >= len(table)
    ):
        return
    strings = image[table[dynsym["link"]]["offset"]:][: table[dynsym["link"]]["size"]]
    count = min(scenid["size"] // 8, dynsym["size"] // dynsym["entsize"])
    for index in range(count):
        offset, = struct.unpack_from("<I", image, dynsym["offset"] + index * dynsym["entsize"])
        if not offset or offset >= len(strings):
            continue
        name = strings[offset : strings.find(b"\0", offset)].decode("utf-8", "replace")
        if not name or "#" in name or name.startswith("$"):
            continue
        value, = struct.unpack_from("<Q", image, scenid["offset"] + index * 8)
        nid = base64.b64encode(struct.pack(">Q", value)).decode()[:11].replace("/", "-")
        yield nid, name


def function_slice(source: str, name: str) -> str:
    declaration = re.search(
        rf"(?m)^(?:pub(?:\([^)]*\))?\s+)?(?:const\s+)?(?:unsafe\s+)?"
        rf'(?:extern\s+"C"\s+)?fn\s+{re.escape(name)}\s*\(',
        source,
    )
    if declaration is None:
        raise ValueError(f"Rust function not found: {name}")
    following = re.search(
        r"(?m)^((?:pub(?:\([^)]*\))?\s+)?(?:const\s+)?(?:unsafe\s+)?"
        r'(?:extern\s+"C"\s+)?fn\s+[A-Za-z0-9_]+\s*\()',
        source[declaration.end() :],
    )
    end = len(source) if following is None else declaration.end() + following.start()
    return source[declaration.start() : end]


def sibling_function_slice(runtime_path: pathlib.Path, name: str) -> str:
    """Find a function by following modules declared from a Rust crate root."""

    pending = [runtime_path]
    visited: set[pathlib.Path] = set()
    root_error: ValueError | None = None
    while pending:
        source_path = pending.pop()
        if source_path in visited:
            continue
        visited.add(source_path)
        source = source_path.read_text(encoding="utf-8")
        try:
            return function_slice(source, name)
        except ValueError as error:
            if root_error is None:
                root_error = error

        module_base = (
            source_path.parent
            if source_path.stem in {"lib", "main", "mod"}
            else source_path.parent / source_path.stem
        )
        modules = re.findall(
            r"(?m)^(?:pub(?:\([^)]*\))?\s+)?mod\s+([A-Za-z0-9_]+)\s*;",
            source,
        )
        for module in modules:
            candidates = (
                module_base / f"{module}.rs",
                module_base / module / "mod.rs",
            )
            pending.extend(candidate for candidate in candidates if candidate.is_file())

    assert root_error is not None
    raise root_error


def already_named() -> set:
    """NIDs the hand-maintained table in the runtime already covers."""
    # Prefer the module that owns the table, then fall back to walking from the crate root so a
    # future move does not revive the old lib.rs-only crash.
    for path in (NID_TABLE_SOURCE, RUNTIME_LIB_SOURCE):
        if not path.is_file():
            continue
        try:
            body = (
                function_slice(path.read_text(encoding="utf-8"), "known_nid_name")
                if path == NID_TABLE_SOURCE
                else sibling_function_slice(path, "known_nid_name")
            )
        except ValueError:
            continue
        return set(re.findall(r'"([A-Za-z0-9+\-]{11})"\s*=>', body))
    raise FileNotFoundError(
        f"could not find known_nid_name in {NID_TABLE_SOURCE} or modules from {RUNTIME_LIB_SOURCE}"
    )


def existing_table_count(path: pathlib.Path) -> int | None:
    """Return the entry count of an existing REMUNID1 blob, or None if absent/unreadable."""

    try:
        data = path.read_bytes()
    except OSError:
        return None
    if len(data) < 16 or data[:8] != b"REMUNID1":
        return None
    count, = struct.unpack_from("<I", data, 8)
    return count


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    force = False
    if args and args[0] == "--force":
        force = True
        args = args[1:]
    if len(args) < 2:
        print(__doc__, file=sys.stderr)
        return 2
    *roots, destination = args
    destination_path = pathlib.Path(destination)

    names = set()
    authoritative: dict[str, str] = {}
    files_seen = 0
    files_with_symbols = 0
    for root in roots:
        for path in sorted(pathlib.Path(root).rglob("*")):
            if not path.is_file():
                continue
            files_seen += 1
            found = set(exported_symbols(path))
            if found:
                files_with_symbols += 1
                names.update(found)
            authoritative.update(scenid_pairs(path))

    if files_seen and not files_with_symbols:
        print(
            "warning: scanned files but recovered no ELF export names; "
            "SELF/game modules usually lack plaintext symbol tables — use SDK ELF stubs instead",
            file=sys.stderr,
        )

    known = already_named()
    # `.scenid` pairs are authoritative and include pinned exports; hashing a pinned export's
    # name would emit a phantom NID no binary uses, so names the stubs already pair are never
    # hashed, and on any residual collision the authoritative pair wins.
    scenid_named = set(authoritative.values())
    merged = {nid_of(name): name for name in names if name not in scenid_named}
    merged.update(authoritative)
    table = [(nid, name) for nid, name in sorted(merged.items())
             if nid not in known and len(name) < 256]

    previous = existing_table_count(destination_path)
    if not force and not table and previous:
        print(
            f"error: recovered 0 names but `{destination_path}` already has {previous} entries; "
            "refusing to overwrite (pass --force to replace anyway). "
            "If you already wiped it: git restore -- crates/remu-runtime/src/nid_names.bin",
            file=sys.stderr,
        )
        return 1

    index, blob = bytearray(), bytearray()
    for nid, name in table:
        encoded = name.encode()
        index += nid.encode() + bytes([len(encoded)]) + struct.pack("<I", len(blob))
        blob += encoded

    header = b"REMUNID1" + struct.pack("<II", len(table), 16 + len(index))
    destination_path.write_bytes(header + index + blob)
    print(f"{len(table)} names -> {destination} ({(len(header) + len(index) + len(blob)) / 1024:.0f} KiB)",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
