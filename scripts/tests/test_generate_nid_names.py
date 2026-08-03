# SPDX-License-Identifier: GPL-2.0-or-later

from __future__ import annotations

import importlib.util
import struct
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "generate-nid-names.py"
SPEC = importlib.util.spec_from_file_location("generate_nid_names", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
GEN = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = GEN
SPEC.loader.exec_module(GEN)


def _minimal_elf_with_symbol(name: bytes) -> bytes:
    """Build a tiny ELF64 with a single .dynsym / .dynstr pair the scanner accepts."""

    # Layout: ELF header (64) | section headers | .dynstr | .dynsym
    # Section 0: NULL, 1: dynstr (type 3), 2: dynsym (type 11, link=1, entsize=24)
    e_shoff = 64
    shentsize = 64
    shnum = 3
    dynstr_off = e_shoff + shnum * shentsize
    dynstr = b"\0" + name + b"\0"
    dynsym_off = dynstr_off + len(dynstr)
    # Elf64_Sym: st_name u32, st_info u8, st_other u8, st_shndx u16, st_value u64, st_size u64
    dynsym = struct.pack("<IBBHQQ", 0, 0, 0, 0, 0, 0) + struct.pack(
        "<IBBHQQ", 1, 0x12, 0, 1, 0x1000, 0
    )

    data = bytearray(dynsym_off + len(dynsym))
    data[0:4] = b"\x7fELF"
    data[4] = 2  # ELFCLASS64
    data[5] = 1  # ELFDATA2LSB
    struct.pack_into("<H", data, 16, 3)  # ET_DYN
    struct.pack_into("<H", data, 18, 0x3E)  # EM_X86_64
    struct.pack_into("<I", data, 20, 1)  # EV_CURRENT
    struct.pack_into("<Q", data, 0x28, e_shoff)
    struct.pack_into("<H", data, 0x3A, shentsize)
    struct.pack_into("<H", data, 0x3C, shnum)

    def write_shdr(index: int, sh_type: int, offset: int, size: int, link: int, entsize: int) -> None:
        base = e_shoff + index * shentsize
        struct.pack_into(
            "<IIQQQQIIQQ",
            data,
            base,
            0,
            sh_type,
            0,
            0,
            offset,
            size,
            link,
            0,
            0,
            entsize,
        )

    write_shdr(0, 0, 0, 0, 0, 0)
    write_shdr(1, 3, dynstr_off, len(dynstr), 0, 0)
    write_shdr(2, 11, dynsym_off, len(dynsym), 1, 24)
    data[dynstr_off : dynstr_off + len(dynstr)] = dynstr
    data[dynsym_off : dynsym_off + len(dynsym)] = dynsym
    return bytes(data)


class GenerateNidNamesTests(unittest.TestCase):
    def test_already_named_reads_nid_table_module(self) -> None:
        known = GEN.already_named()
        self.assertIn("T72hz6ffq08", known)
        self.assertIn("vSMAm3cxYTY", known)
        self.assertGreater(len(known), 100)

    def test_already_named_falls_back_through_crate_root(self) -> None:
        original = GEN.NID_TABLE_SOURCE
        try:
            GEN.NID_TABLE_SOURCE = Path("/definitely/missing/nid_table.rs")
            known = GEN.already_named()
        finally:
            GEN.NID_TABLE_SOURCE = original
        self.assertIn("T72hz6ffq08", known)

    def test_exported_symbols_reads_raw_elf(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "stub.so"
            path.write_bytes(_minimal_elf_with_symbol(b"scePthreadYield"))
            self.assertEqual(list(GEN.exported_symbols(path)), ["scePthreadYield"])

    def test_exported_symbols_unwraps_self_container(self) -> None:
        elf = _minimal_elf_with_symbol(b"sceKernelMprotect")
        segment_count = 1
        elf_offset = 32 + segment_count * 32
        container = bytearray(elf_offset + len(elf))
        container[0:4] = GEN.PS5_CONTAINER_MAGIC
        struct.pack_into("<H", container, 0x18, segment_count)
        container[elf_offset:] = elf
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "module.prx"
            path.write_bytes(container)
            self.assertEqual(list(GEN.exported_symbols(path)), ["sceKernelMprotect"])

    def test_nid_of_matches_known_ps5_exports(self) -> None:
        self.assertEqual(GEN.nid_of("scePthreadYield"), "T72hz6ffq08")
        self.assertEqual(GEN.nid_of("sceKernelMprotect"), "vSMAm3cxYTY")

    def test_refuses_to_overwrite_existing_table_with_zero_names(self) -> None:
        repo_table = Path(__file__).resolve().parents[2] / "crates/remu-runtime/src/nid_names.bin"
        previous = GEN.existing_table_count(repo_table)
        self.assertIsNotNone(previous)
        self.assertGreater(previous or 0, 0)
        with tempfile.TemporaryDirectory() as tmp:
            empty_root = Path(tmp) / "empty"
            empty_root.mkdir()
            destination = Path(tmp) / "nid_names.bin"
            destination.write_bytes(repo_table.read_bytes())
            rc = GEN.main([str(empty_root), str(destination)])
            self.assertEqual(rc, 1)
            self.assertEqual(GEN.existing_table_count(destination), previous)

    def test_force_allows_empty_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            empty_root = Path(tmp) / "empty"
            empty_root.mkdir()
            destination = Path(tmp) / "nid_names.bin"
            destination.write_bytes(b"REMUNID1" + struct.pack("<II", 3, 16) + b"\0" * 48)
            rc = GEN.main(["--force", str(empty_root), str(destination)])
            self.assertEqual(rc, 0)
            self.assertEqual(GEN.existing_table_count(destination), 0)

    def test_scenid_recovers_a_pinned_export_hashing_cannot(self) -> None:
        # sceAgcInit's NID is hand-assigned in SDK 10.00; hashing its name yields something else.
        pinned = "23LRUSvYu1M"
        self.assertNotEqual(GEN.nid_of("sceAgcInit"), pinned)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "libSceAgc_stub_weak.a"
            path.write_bytes(
                _elf_with_scenid([(b"sceAgcInit", _scenid_bytes_for(pinned))])
            )
            self.assertEqual(list(GEN.scenid_pairs(path)), [(pinned, "sceAgcInit")])

    def test_scenid_encoding_agrees_with_the_hash_for_unpinned_exports(self) -> None:
        # For a hashed export the .scenid bytes ARE the hash prefix; both oracles must agree.
        import hashlib

        name = "sceExampleHashedExport"
        digest8 = hashlib.sha1(name.encode() + GEN.NID_SUFFIX).digest()[:8]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "stub.a"
            path.write_bytes(_elf_with_scenid([(name.encode(), digest8)]))
            self.assertEqual(list(GEN.scenid_pairs(path)), [(GEN.nid_of(name), name)])

    def test_main_prefers_scenid_pairs_over_hashed_names(self) -> None:
        # A pinned entry must land in the emitted table under its authoritative NID.
        import base64

        raw_le = bytes.fromhex("8877665544332211")
        pinned = (
            base64.b64encode(struct.pack(">Q", struct.unpack("<Q", raw_le)[0]))
            .decode()[:11]
            .replace("/", "-")
        )
        name = b"sceFakePinnedExport"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "stubs"
            root.mkdir()
            (root / "stub.a").write_bytes(_elf_with_scenid([(name, raw_le)]))
            destination = Path(tmp) / "nid_names.bin"
            rc = GEN.main([str(root), str(destination)])
            self.assertEqual(rc, 0)
            data = destination.read_bytes()
            count, names_offset = struct.unpack_from("<II", data, 8)
            entries = {}
            for index in range(count):
                record = data[16 + index * 16 : 16 + (index + 1) * 16]
                nid = record[:11].decode()
                length = record[11]
                offset, = struct.unpack_from("<I", record, 12)
                entries[nid] = data[names_offset + offset : names_offset + offset + length].decode()
            self.assertEqual(entries.get(pinned), "sceFakePinnedExport")
            # The phantom hash-derived NID for the same name must not appear alongside it.
            self.assertNotIn(GEN.nid_of("sceFakePinnedExport"), entries)


def _scenid_bytes_for(nid: str) -> bytes:
    """The little-endian 8 bytes a stub's .scenid holds for this display NID."""
    import base64

    return base64.b64decode(nid.replace("-", "/") + "=")[::-1]


def _elf_with_scenid(entries: list[tuple[bytes, bytes]]) -> bytes:
    """Build an ELF64 stub with .dynstr/.dynsym/.scenid/.shstrtab like the SDK link stubs."""

    e_shoff = 64
    shentsize = 64
    shnum = 5
    dynstr = b"\0"
    name_offsets = []
    for name, _ in entries:
        name_offsets.append(len(dynstr))
        dynstr += name + b"\0"
    dynstr_off = e_shoff + shnum * shentsize
    dynsym_off = dynstr_off + len(dynstr)
    dynsym = struct.pack("<IBBHQQ", 0, 0, 0, 0, 0, 0)
    for offset in name_offsets:
        dynsym += struct.pack("<IBBHQQ", offset, 0x12, 0, 1, 0x1000, 0)
    scenid_off = dynsym_off + len(dynsym)
    scenid = b"\0" * 8  # entry for the null symbol
    for _, raw in entries:
        assert len(raw) == 8
        scenid += raw
    shstrtab_off = scenid_off + len(scenid)
    shstrtab = b"\0.dynstr\0.dynsym\0.scenid\0.shstrtab\0"
    scenid_name_offset = shstrtab.index(b".scenid")

    data = bytearray(shstrtab_off + len(shstrtab))
    data[0:4] = b"\x7fELF"
    data[4] = 2
    data[5] = 1
    struct.pack_into("<H", data, 16, 1)  # ET_REL, like the SDK stubs
    struct.pack_into("<H", data, 18, 0x3E)
    struct.pack_into("<I", data, 20, 1)
    struct.pack_into("<Q", data, 0x28, e_shoff)
    struct.pack_into("<H", data, 0x3A, shentsize)
    struct.pack_into("<H", data, 0x3C, shnum)
    struct.pack_into("<H", data, 0x3E, 4)  # e_shstrndx -> .shstrtab

    def write_shdr(index, sh_name, sh_type, offset, size, link, entsize):
        struct.pack_into(
            "<IIQQQQIIQQ",
            data,
            e_shoff + index * shentsize,
            sh_name,
            sh_type,
            0,
            0,
            offset,
            size,
            link,
            0,
            0,
            entsize,
        )

    write_shdr(0, 0, 0, 0, 0, 0, 0)
    write_shdr(1, shstrtab.index(b".dynstr"), 3, dynstr_off, len(dynstr), 0, 0)
    write_shdr(2, shstrtab.index(b".dynsym"), 11, dynsym_off, len(dynsym), 1, 24)
    write_shdr(3, scenid_name_offset, 1, scenid_off, len(scenid), 0, 8)
    write_shdr(4, shstrtab.index(b".shstrtab"), 3, shstrtab_off, len(shstrtab), 0, 0)
    data[dynstr_off : dynstr_off + len(dynstr)] = dynstr
    data[dynsym_off : dynsym_off + len(dynsym)] = dynsym
    data[scenid_off : scenid_off + len(scenid)] = scenid
    data[shstrtab_off : shstrtab_off + len(shstrtab)] = shstrtab
    return bytes(data)


if __name__ == "__main__":
    unittest.main()
