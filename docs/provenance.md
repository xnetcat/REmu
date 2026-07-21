# Implementation provenance

This file records upstream implementation material used to accelerate REmu.

## `remu-loader`

- Standard ELF64 structures and dynamic tags follow the ELF specification.
- SELF container segment layout and property meanings were cross-checked against Obliteration
  snapshot `16e981ecd7ac47208697e0e09fb8749c5f13d574`, particularly
  `legacy/src/imgact/orbis/mod.rs`. Obliteration is MIT OR Apache-2.0.
- PS5 loader tag coverage and physical-segment fallback behavior were adapted from SharpEmu snapshot
  `ada67a192495829b5d5408c08453cdfdc8553bb0`, particularly
  `src/SharpEmu.Core/Loader/SelfLoader.cs`. SharpEmu is GPL-2.0-or-later.

The Rust parser adds independent bounded-input validation, checked arithmetic, streaming reads, and
synthetic tests. No game bytes or proprietary SDK material are included.

## `remu-memory`

- Fixed-address macOS allocation, safe hint validation, Mach fixed-allocation fallback, staged
  writes, and merged page protections were adapted from SharpEmu snapshot `ada67a19`, particularly
  `SharpEmu.HLE/Host/Posix/PosixHostMemory.cs` and
  `SharpEmu.Core/Memory/PhysicalVirtualMemory.cs`.
- REmu reserves one contiguous main-image range before copying segments, avoiding partial mapping
  state and handling shared host pages deterministically.

## `remu-runtime`

- The relocation set, unresolved-import trap-stub model, NID extraction, and symbol resolution
  semantics were adapted from SharpEmu `SelfLoader.cs` at snapshot `ada67a19`.
- REmu applies the four relocation types present in the first two authorized test titles before
  finalizing memory protections: `R_X86_64_64`, `GLOB_DAT`, `JUMP_SLOT`, and `RELATIVE`.
