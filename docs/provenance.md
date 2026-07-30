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
- Commit-on-protect semantics (a guest `mprotect` over a reserved virtual range commits real,
  zero-filled, GPU-referencable memory; flexible mappings are the same committed class) were
  established by black-box observation of an authorized Unity title's VM trace: its allocator
  reserves ranges, commits them with `sceKernelMprotect` carrying CPU, GPU, and AMPR protection
  bits, decommits with `sceKernelMunmap`, and then binds texture descriptors directly into that
  memory — while the title renders correctly on hardware. The in-tree protection translation
  already accepted the GPU bits; no new ABI constants were introduced. The committed-backing
  design (one sparse per-space object whose byte offsets equal virtual offsets) is REmu's own:
  reserve-then-commit memory never exposes a physical address to the guest, so its only stable
  physical identity is its virtual placement, mirroring GPU page-table addressing on the console.
  Decommit zeroes backing pages before releasing them so recommitted ranges read as fresh zeros.
  No upstream emulator code was copied for this path.
- The commit-on-protect design was cross-checked after implementation against shadPS4 at its
  2026-07-28 `main` (GPL-2.0-or-later, SPDX headers verified): its `MemoryManager::PoolCommit`
  (`src/core/memory.cpp`) likewise turns a reservation into real shared host backing
  (shm/memfd-backed) and registers the range with the GPU layer, and its GPU-visibility test
  (`IsValidGpuMapping`, `src/core/memory.h`) is an address-range check, not a memory-class
  check — corroborating that committed non-direct memory is GPU-referencable. fpPS4
  (LGPL-2.1, not GPL — any future borrowing is a relicensing event) corroborates that *pooled*
  memory is physically-block-based ("section of direct memory, 64 KiB blocks",
  `kernel/ps4_map_mm.pas`), which is why REmu's virtually-keyed committed backing is scoped to
  the no-physical-address classes only and pool NIDs remain unimplemented. Kyty (MIT) implements
  neither pools nor reserve-then-commit and was uninformative here. Both projects were read as
  behavioral research only; no source was copied.
- Whether `sceKernelMprotect` alone commits reserved memory is a recorded PS4/PS5 divergence.
  OpenOrbis-based tests running on real PS4 hardware (ps4emulation/integration_tests,
  `tests/code/memory_test/code/test_100.cpp`) show PS4 accepting the call while clamping the
  reservation's protection to zero, and rejecting AMPR protection bits outright. The authorized
  Unity title's complete VM trace under REmu contains zero flexible mappings and no fixed
  direct mapping covering its texture blocks, so reserve-then-`mprotect(0xf2)` is the only
  operation that can commit them — and the title ships and renders on retail PS5 hardware. The
  `0xf2` protection decode (CPU read/write, GPU read/write, AMPR read/write) matches the
  constants in drakmor/ampr_emu, a public PS5 userland `sce::Ampr` reimplementation, whose AMPR
  event surface exists only in PS5 libkernel exports — confirming PS5's protection model
  extends PS4's. All read as behavioral research; no code copied.

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
- Pthread condition-variable wake cardinality, atomic mutex release/reacquisition, recursive-depth
  restoration, dynamic mutex binding, timeout arbitration, and destroy-after-selection behavior
  were cross-checked against POSIX Issue 8 and three compatible emulator implementations: SharpEmu
  snapshot `ada67a192495829b5d5408c08453cdfdc8553bb0`
  (`KernelPthreadCompatExports.cs` and `PthreadCondSemanticsTests.cs`), KytyPS5 snapshot
  `8a4600c5e99ab3b3873b9ea57b70044f0fa34807` (`src/kernel/pthread.h`,
  `src/kernel/pthread.cpp`, and `src/libs/libKernel.cpp`), and shadPS4 snapshot
  `4c72de0732ce7443a1631bf190e2606294f58f65`
  (`src/core/libraries/kernel/threads/pthread.h` and `condvar.cpp`). KytyPS5 was used only as
  behavioral research. REmu independently uses guest tickets outside condition-owned pending state,
  so one signal cannot become a broadcast and a selected waiter can reacquire its mutex after legal
  condition destruction. No upstream source was copied.
- Direct-memory `MAP_NO_COALESCE` (`0x400000`), containment-first/find-next `VirtualQuery`, cross-VMA
  unmap, and release-time alias invalidation were cross-checked against KytyPS5 snapshot
  `8a4600c5e99ab3b3873b9ea57b70044f0fa34807` (`src/kernel/memory.cpp`) and shadPS4 snapshot
  `4c72de0732ce7443a1631bf190e2606294f58f65` (`src/core/memory.h` and
  `src/core/memory.cpp`). SharpEmu snapshot `d765552e69c49abb79b6d9a24d29fe4ae0c63688`
  (`KernelMemoryCompatExports.cs`) independently preserves one query region per map call but has
  incomplete partial-unmap/release alias lifetime, so it was used only as corroborating research.
  REmu independently assigns private identities only to guest no-coalesce mappings, preserves those
  identities through VMA surgery, and removes every overlapping virtual alias before released
  physical pages can be reused. No upstream source was copied.
- The direct- and flexible-memory availability queries (`sceKernelAvailableDirectMemorySize`
  `C0f7TJcbfac`, `sceKernelAvailableFlexibleMemorySize` `aNz11fnnzi4`) had their argument order and
  result shape confirmed against three independent implementations before being written: shadPS4
  snapshot `4c72de0732ce7443a1631bf190e2606294f58f65`
  (`src/core/libraries/kernel/memory.cpp` and `MemoryManager::DirectQueryAvailable` in
  `src/core/memory.cpp`), SharpEmu snapshot `ada67a19` (`KernelMemoryCompatExports.cs`), and fpPS4
  (`kernel/ps4_map_mm.pas`). All three agree that the direct query reports the largest aligned free
  span within the requested window rather than a running total. REmu's implementations are its own:
  the free-span scan reuses the existing direct allocator's records and normalizes its arguments
  through the same path as `sceKernelAllocateDirectMemory`, so a reported span is allocatable by
  construction; flexible usage is derived from live flexible mappings rather than a running counter,
  so partial unmaps and rollbacks stay accounted for. The 448 MiB flexible pool pairs with the
  16 GiB direct total already in tree and matches SharpEmu, which uses the same pair for its PS5
  kernel surface. A title may override its own budget through `mem_param->flexible_memory_size` in
  its process parameters (shadPS4 `src/core/linker.cpp`); that override is not read yet, and the
  titles exercised here leave the pointer null. No upstream source was copied.
- The fiber ABI (`libSceFiber`) had its object layout, error family (`0x8059000x`), state machine,
  and switch semantics cross-checked against shadPS4 snapshot
  `4c72de0732ce7443a1631bf190e2606294f58f65` (`src/core/libraries/fiber/fiber.cpp`,
  `fiber_context.cpp`, `fiber.h`, and `fiber_error.h`), with SharpEmu snapshot `ada67a19`
  (`SharpEmu.Libs/Fiber/FiberExports.cs`) as corroborating research for the exported set. The
  register-save layout here is REmu's own and deliberately differs: both floating-point control
  words sit at naturally aligned offsets rather than the packed offsets the console library uses,
  since the structure never reaches the guest. The switch primitives follow the naked-assembly
  boundary this crate already used for guest thread exit
  (`invoke_guest_entry_with_exit_boundary` / `escape_guest_entry`) rather than being copied. The
  per-thread current-fiber record is a Rust thread local instead of a slot in a synthesized TCB.
  No upstream source was copied.
- AMD64 DTV slot layout and `__tls_get_addr` (`vNe1w4diLCs`) were cross-checked against FreeBSD src
  snapshot `afa048d159f64e1c609475314e19fb030ffd595f` (`sys/x86/include/tls.h` and
  `libexec/rtld-elf/rtld.c`), shadPS4 snapshot
  `4c72de0732ce7443a1631bf190e2606294f58f65` (`src/core/linker.cpp` and
  `src/core/libraries/kernel/threads/tcb.cpp`), KytyPS5 snapshot
  `8a4600c5e99ab3b3873b9ea57b70044f0fa34807` (`src/loader/runtimeLinker.cpp` and
  `src/libs/libKernel.cpp`), and SharpEmu snapshot
  `ada67a192495829b5d5408c08453cdfdc8553bb0` (`GuestTlsTemplate.cs` and
  `KernelMemoryCompatExports.cs`). REmu independently uses its existing per-thread TLS allocation,
  resolves the guest's two-word module/offset descriptor through compact pointer slots, and rebases
  every populated slot when cloning a worker. No upstream source was copied.
- AGC shader-header pointer relocation, program-register patching, and the VideoOut buffer-attribute
  and registration layouts were adapted from SharpEmu's GPL-2.0-or-later `AgcExports.cs` and
  `VideoOutExports.cs` at local snapshot `ada67a19`.
- Submission-bounded targetless scanout association was cross-checked against SharpEmu's retained
  draw-to-VideoOut association in GPL-2.0-or-later snapshot `2ee32b8`, particularly
  `SharpEmu.Libs/Agc/AgcExports.cs`. Partial-flush `EVENT_WRITE` encodings and their ordering role
  were independently checked against KytyPS5 snapshot
  `8a4600c5e99ab3b3873b9ea57b70044f0fa34807` (`src/libs/agc.cpp`,
  `pm4Handlers.cpp`, and `graphicsRun.cpp`), Mesa snapshot
  `d3d45a7fb41433149924d0bbbf9d9bd1ddb9c33a` (`sid.h`, `gfx10.json`, and `radv_cs.c`), and Linux
  snapshot `248951ddc14de84de3910f9b13f51491a8cd91df` (`soc15d.h`). KytyPS5 is
  GPL-2.0-only and was used only as behavioral research; no source was copied. REmu independently
  validates a complete submission, associates only unambiguous straight-line draws, preserves
  packet order, and transports partial flushes as GPU stage barriers rather than CPU notifications.
- Depth-target validity was cross-checked against SharpEmu Metal snapshot
  `2ee32b897504b06e13404749163908346045d60b` (`AgcExports.cs`, `DecodeDepthTarget`). REmu
  independently requires enabled depth control plus a nonzero format, programmed extent, and at
  least one non-null read/write base before depth suppresses targetless scanout association. Stale
  control registers without an actual surface are not treated as a bound attachment.
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
- AvPlayer Ready/Pause/IsActive state semantics (a newly added source is active before Start, and
  a paused player remains active while freezing `CurrentTime` and releasing neither video nor audio
  frames until Resume) were cross-checked against shadPS4's GPL-2.0-or-later AvPlayer state machine
  (`src/core/libraries/avplayer/`). The `sceAvPlayerChangeStream`/`sceAvPlayerSetTrickSpeed`
  export names and the percent-based trick-speed rate follow the public PS4/PS5 AvPlayer API
  surface as documented in open-source homebrew headers. The Gen5 one-based stream discriminators
  and 104-byte extended stream-info fields were recovered from the authorized local Unity and
  Unreal executable call sites. REmu's ffmpeg pipe decoders, input-side `-ss` seek, and
  host-elapsed media clock are independent implementations. No SDK material was copied.

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

## `remu-shader-metal`

- The register-file execution model, scalar/vector operation semantics, export mapping, pixel
  interpolation, image sampling operand order, and descriptor-driven GFX10 buffer-format conversion
  were adapted from SharpEmu's GPL-2.0-or-later `Gen5MslTranslator.cs`,
  `Gen5MslTranslator.Alu.cs`, `Gen5MslTranslator.Pixel.cs`, and Metal templates in the local
  2026-07-21 checkout.
- REmu's Rust lowering operates exclusively on the backend-neutral typed IR, produces structured
  PC/opcode failures instead of substitute shaders, uses draw-evaluated resource bindings, and is
  validated against complete live shader pairs from the authorized local dumps. No game shader
  bytes, game fingerprints, or proprietary SDK material are included in the source tree.
- The GCN cube-map addressing contract — `V_CUBEID/V_CUBESC/V_CUBETC/V_CUBEMA` face selection,
  sign conventions, and the face-atlas coordinate form `(sc/(2|ma|)+1.5, tc/(2|ma|)+1.5, faceId)`
  consumed by `IMAGE_SAMPLE*` on cube resources — follows AMD's public "GCN3 ISA" and
  "RDNA Instruction Set Architecture" reference guides, and was cross-checked behaviorally
  against shadPS4's GPL-2.0-or-later shader recompiler's cube handling. Binding cube resources
  as Metal 2D arrays and indexing the face layer directly from the atlas triple is REmu's own
  lowering choice; no code was copied.
- GFX10 secondary position-export packing and `PA_CL_VS_OUT_CNTL` semantics follow AMD's public
  register definitions and were cross-checked against shadPS4's GPL-2.0-or-later
  `MapOutputs`/`ExportPosition` behavior. REmu independently lowers enabled clip distances to
  Metal's documented `[[clip_distance]]` vertex output. Metal has no native cull-distance output,
  so enabled cull components use the same conservative clipping path rather than being ignored.

## `remu-gpu-metal`

- GFX10 `SW_4KB_S` address bits, thin-resource block dimensions, mip-tail entry
  conditions and coordinates, reverse non-tail mip placement, and per-slice chain
  stride follow AMD's MIT-licensed GPUOpen AddrLib implementation in PAL,
  particularly `addrlib2.cpp`, `gfx10addrlib.cpp`, and
  `gfx10SwizzlePattern.h`. The surrounding texture-cache policy and Metal upload
  implementation are REmu's own. shadPS4's GPL-2.0-or-later texture cache was
  used to cross-check that complete mip/layer ranges remain one image resource;
  its PS4-specific tiling equations were not reused for this PS5 layout.
- GFX10 `CB_COLOR_CONTROL` operation-mode decoding follows AMD's public PAL register
  definitions and was cross-checked against shadPS4's GPL-2.0-or-later renderer handling.
  In particular, `Resolve`, `EliminateFastClear`, and `FmaskDecompress` draw packets are
  command-processor operations rather than ordinary shader rasterization. REmu keeps color
  residents decompressed and ordinary MSAA passes maintain a current single-sample MRT0 resolve;
  for the explicit Resolve operation it then copies MRT0 into the separately programmed MRT1
  destination. `EliminateFastClear` materializes MRT0's programmed clear color into Metal storage
  instead of retaining pixels from an older frame. These surface roles and the CMASK clear
  materialization point were cross-checked against shadPS4's `Rasterizer::Resolve`,
  `Image::Resolve`, and `Rasterizer::EliminateFastClear`; REmu's Metal implementation is
  independent.
- Texture readback storage, synchronization, resolve, and row-layout rules follow Apple's public
  Metal documentation for `MTLStorageMode`, `MTLBlitCommandEncoder`, `MTLCommandBuffer`,
  `MTLTexture.getBytes`, load/store actions, and the published Metal feature limits. They were also
  checked against the corresponding public framework headers in the installed macOS SDK.
- Direct-scanout attachment lifetime follows Apple's public `MTLLoadAction` and render-pass
  attachment documentation: `Clear` writes every attachment pixel at pass start, while `Load`
  preserves the existing contents. REmu therefore clears a newly allocated host-only scanout
  texture exactly once, then loads it for each later translated draw so separate Metal passes do
  not erase earlier guest draws. No upstream emulator implementation was copied for this policy.
- Window telemetry uses public Metal command-buffer status/completion behavior and the installed
  macOS SDK's `libproc.h`/`sys/proc_info.h` declarations for `proc_pidinfo(PROC_PIDTASKINFO)` live
  resident memory. Audio-underrun percentage comes from REmu's own CoreAudio callback frame and
  missing-sample counts; the title never relabels the Metal HUD's overlay cadence as guest FPS.
- Immutable sampler argument-buffer reuse follows the installed macOS SDK's public
  `MTLArgumentEncoder` contract: an encoder reports the exact storage length, selects a destination
  buffer, and writes sampler-state bindings into that buffer. REmu independently retains that
  encoded buffer and its sampler objects under an exact compiled-function/buffer-index/descriptor
  key, with a fixed 1,024-entry bound. No upstream emulator implementation was copied for this
  cache.
- REmu independently implements a checked readback ticket: a backed single-sample texture is blitted
  into an aligned shared buffer on the producing command queue, the source and staging resources are
  retained through command-buffer completion, and padding is removed into the exact native pixel
  layout before any guest tiling conversion. Diagnostic conversion of sRGB storage bytes uses the
  standard IEC 61966-2-1 transfer function before writing the linear reference-frame corpus. No
  upstream emulator code was copied for this path.
