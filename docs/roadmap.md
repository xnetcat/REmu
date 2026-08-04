# Implementation roadmap

Every milestone ends in a bounded CLI test before more subsystems are added. Local game tests remain
opt-in and never become repository fixtures.

## M0: host and input reconnaissance

Status: complete.

- GPL-2.0-or-later Rust workspace and source-provenance rules.
- Native and Rosetta host detection.
- Direct Metal device probe.
- Standard ELF64 and observed executable-container identification.

## M1: read-only executable model

- Parse ELF program headers, dynamic entries, symbols, relocations, and PS5-specific metadata.
- Parse executable-container metadata without copying protected payloads into the repository.
- Produce deterministic inspection output and precise validation errors.
- Inventory required libraries/imports for the first authorized test titles.

This milestone does not map or execute guest code.

## M2: guest virtual memory and dynamic linker

- Reserve and manage a guest address space with explicit page permissions.
- Map loadable segments and zero-fill memory without writable/executable overlap.
- Resolve supported relocations, module dependencies, TLS, and import trampolines.
- Add crash-safe diagnostics for address, module, thread, and guest register state.

## M3: native x86-64 execution and minimum HLE

- Enter guest code only in the Rosetta x86-64 build.
- Implement process/thread startup, synchronization, timing, filesystem, and libc calls demanded by
  the first title.
- Recover guest faults without masking host/runtime faults.
- Keep every HLE function attributable to public documentation or recorded black-box behavior.

## M4: graphics-to-Metal vertical slice

- Define AGC command, resource, format, synchronization, and shader-domain types in backend-neutral
  crates.
- Decode only commands observed in the first title, with unknown-command capture and bounded logs.
- Decode RDNA2 shaders into a typed IR and emit MSL independently.
- Create Metal buffers, textures, pipelines, command buffers, and an offscreen frame target before
  adding a window.

## M5: first game loop

- Bring the first small test title through module initialization, frame submission, input, and audio.
- Add deterministic boot checkpoints and opt-in local smoke-test configuration.
- Fix correctness blockers by subsystem instead of adding title-specific binary patches.

## M6: compatibility growth

- Use progressively larger titles as broader regression targets.
- Add codecs, savedata, richer audio, controller behavior, and synchronization as demanded.
- Build a compatibility database from structured, non-copyrightable observations.

## M7: other host backends

Status: executable vertical slice. Native x86-64 guest execution, sparse memory, WASAPI, XInput,
HLSL/DXIL translation, D3D12 event execution, and DXGI presentation are connected and launch the
first local test title. Transient normalized/integer color and depth/stencil MSAA plus ordered
indirect multi-draw are physically covered; compatibility growth, NGG fallback, GPU residency/batching,
clean-machine packaging, and multi-title validation remain in progress.

- Complete and optimize D3D12 plus HLSL/DXIL compatibility on Windows.
- Add Vulkan plus SPIR-V on Linux.
- Share guest decoding and shader IR, not host-API abstractions that erase important semantics.

## M8: native Apple Silicon CPU backend

- Add an x86-64 decoder, typed CPU IR, code cache, dispatcher, and AArch64 emitter.
- Validate it against the native/Rosetta behavior oracle using synthetic instruction and ABI tests.
- Keep Rosetta as a diagnostic backend while the translator matures.
