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
- The FreeBSD/AMD64 Variant-II startup TLS layout and `%fs:[0]` load-rewriting strategy were adapted
  from SharpEmu `GuestTlsTemplate.cs` and `DirectExecutionBackend.cs` at snapshot `ada67a19`.
  REmu emits small register-preserving helpers in guest auxiliary memory and keeps the macOS host
  thread pointer untouched.
- Per-worker TLS cloning/rebasing, native-thread scheduling, direct-memory allocation, libc/C++
  startup behavior, and the first mutex/condition/semaphore services follow the corresponding
  SharpEmu CPU and HLE implementations at snapshot `ada67a19`. The Rust implementation uses host
  threads and condition variables while retaining guest-visible handles and ownership state.
- AGC shader-header pointer relocation, program-register patching, and the VideoOut buffer-attribute
  and registration layouts were adapted from SharpEmu's GPL-2.0-or-later `AgcExports.cs` and
  `VideoOutExports.cs` at local snapshot `ada67a19`.
- Read-only app filesystem path resolution and libc stdio forwarding follow SharpEmu's
  GPL-2.0-or-later `LibcStdioExports.cs` design at local snapshot `ada67a19`; REmu uses native
  `FILE` objects directly because the Rosetta guest and host share the x86-64 libc ABI.
- The AudioOut port ABI, format validation, buffer pacing, volume model, and console clock-id
  behavior were cross-checked against SharpEmu's GPL-2.0-or-later `AudioOutExports.cs` and
  `KernelMemoryCompatExports.cs` in the local 2026-07-21 checkout. REmu implements its own Rust
  mixer, CPAL/CoreAudio output path, stereo downmixing, resampling, and bounded queues.
- Universal Data System, Game Intent, and Trophy2 export names and initial argument positions were
  cross-checked against SharpEmu's GPL-2.0-or-later `NpUniversalDataSystemExports.cs`,
  `NpGameIntentExports.cs`, and `NpTrophy2Exports.cs`. Exact Gen5 call shapes were then recovered
  from the authorized Dreaming Sarah executable's import call sites. REmu's Rust implementation
  adds independent context/request ownership, transactional typed event graphs, committed local
  posts, intent-buffer/property behavior, trophy lifecycle validation, and callback retention.
- REmu applies the four relocation types present in the first two authorized test titles before
  finalizing memory protections: `R_X86_64_64`, `GLOB_DAT`, `JUMP_SLOT`, and `RELATIVE`.

## `remu-shader`

- The backend-neutral Gen5 instruction/control model, shader resource metadata layout, GFX10
  unified-format table, and inline-constant table were adapted from SharpEmu's
  GPL-2.0-or-later `SharpEmu.ShaderCompiler` sources in the local 2026-07-21 checkout. REmu uses
  Rust-owned immutable programs, typed control variants, bounded metadata reads, exact sparse
  format decoding, and a backend-independent compute-system-value model.
- Gen5 instruction framing, opcode tables, literal/NSA extension rules, and typed operand/control
  extraction were cross-checked against SharpEmu's GPL-2.0-or-later
  `Gen5ShaderTranslator.cs` in that checkout. REmu independently uses slice-bounded decoding,
  structured reserved-encoding errors, typed raw-family preservation, and explicit load/store/
  atomic data-flow operands so later backends cannot silently treat writes as reads.
- Scalar-state evaluation, hardware user-SGPR count decoding, buffer descriptor layout, image/
  sampler descriptor discovery, scalar/global/buffer memory binding capture, and vertex-fetch
  classification were cross-checked against SharpEmu's GPL-2.0-or-later
  `Gen5ShaderScalarEvaluator.cs` and `Gen5ShaderTranslator.cs`. REmu's implementation uses typed
  failures, exact conditional-branch evaluation, bounded alternate-CFG discovery, strict non-null
  memory reads, and a backend-neutral guest-memory trait; it does not inherit SharpEmu's
  title-specific tracing or permissive zero-buffer fallbacks.
