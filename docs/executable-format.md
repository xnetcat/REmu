# Executable format notes

This document records the minimum input model REmu needs. The ELF portions follow the standard
64-bit little-endian ELF layout. Container facts were cross-checked against the permissively licensed
Obliteration project and then validated independently against the authorized local test inputs.

## Input forms

REmu currently recognizes three executable forms:

- A direct ELF64 image beginning with `7f 45 4c 46`.
- A signed executable container beginning with `4f 15 3d 1d`.
- A retail executable container beginning with `54 14 f5 ee`.

Both observed container forms expose a plaintext ELF header and program-header table. Segment payload
properties determine whether an individual stored segment still requires decryption or expansion.
REmu will reject those operations until an authorized input path supplies already usable payloads.

## Container header

The fixed header is 32 bytes, little-endian:

| Offset | Size | Field |
| ---: | ---: | --- |
| `0x00` | 4 | magic |
| `0x04` | 1 | version |
| `0x05` | 1 | mode |
| `0x06` | 1 | endian marker |
| `0x07` | 1 | attributes |
| `0x08` | 4 | key type |
| `0x0c` | 2 | header size |
| `0x0e` | 2 | metadata size |
| `0x10` | 8 | declared file size |
| `0x18` | 2 | segment count |
| `0x1a` | 2 | flags |
| `0x1c` | 4 | reserved |

The fixed header is followed by `segment_count` 32-byte descriptors. The embedded ELF header starts
at `0x20 + segment_count * 0x20`. All current Dreaming Sarah and Superliminal samples contain 12
descriptors and place the ELF header at `0x1a0`.

Each descriptor contains four little-endian 64-bit values: properties, stored file offset, stored
size, and expanded size. Bits 20 through 31 of properties select the associated ELF program-header
index. Known low property bits include encrypted (`0x2`), signed (`0x4`), and block table (`0x800`).

## Loader invariants

- Every arithmetic operation involving input offsets is checked for overflow.
- Header and descriptor counts are bounded before allocation.
- Every stored segment range must fit inside the physical input file.
- ELF program headers are parsed from the physical header table. Their payload offsets remain logical
  ELF offsets and require container-to-program mapping before data is read.
- The declared container size is retained for diagnostics but is not assumed to equal the host file
  length because legal dump tools may preserve aligned padding or diagnostic trailers.
- Tests use synthetic bytes; local games are opt-in runtime fixtures only.
