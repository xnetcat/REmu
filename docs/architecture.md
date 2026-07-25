# Architecture

## Execution model

REmu is a userspace compatibility layer, not a virtualized PlayStation kernel. On x86-64 hosts it
maps guest executable segments into a managed guest address space, applies supported relocations,
and enters guest code using the System V AMD64 ABI. Calls into operating-system libraries are routed
through generated import trampolines to Rust HLE implementations.

Apple Silicon initially runs the x86-64 build through Rosetta 2. Native AArch64 support will require
a dedicated x86-64 decoder, intermediate representation, optimizer, and AArch64 code generator; it
must remain a separate CPU backend rather than leaking translation details into the loader or HLE.

## Data flow

```text
SELF / ELF / PRX
    -> loader and dynamic linker
    -> guest virtual address space
    -> native x86-64 CPU backend
    -> import trampolines and HLE services
    -> AGC / PM4 command processor
    -> RDNA2 shader decoder and typed shader IR
       -> MSL  -> Metal on macOS
       -> HLSL -> DXIL / D3D12 on Windows
       -> SPIR-V -> Vulkan on Linux
```

## Crate boundaries

- `remu-core`: guest-domain value types shared across subsystems.
- `remu-platform`: host OS and execution-environment discovery.
- `remu-loader`: SELF/ELF executable parsing and module dependency graphs.
- `remu-memory`: guest address-space reservation, mapping, and CPU write watching.
- `remu-runtime`: dynamic linking, HLE kernel and library services, audio, and guest execution.
- `remu-gpu`: backend-neutral host GPU contracts expressed in guest-domain terms.
- `remu-gpu-metal`: direct macOS Metal presenter and resource backend.
- `remu-shader`: RDNA2 shader decoding, typed IR, and static evaluation.
- `remu-shader-metal`: MSL lowering for the typed shader IR.
- `remu-cli`: inspection, diagnostics, and game launch orchestration.

Further crates (a debugger, other GPU backends, a native CPU translator) are introduced only when
a vertical slice requires them. This avoids cementing placeholder APIs before their invariants are
understood.

## Graphics rules

The AGC command processor and shader front-end cannot expose Metal, D3D12, Vulkan, MSL, DXIL, or
SPIR-V types. Backends receive guest-domain resources, formats, synchronization intent, and opaque
compiled-shader handles. Each backend owns API-specific barriers, heaps, descriptors, pipelines,
presentation, and shader emission.

The shader compiler will use a custom typed SSA-like IR designed around RDNA2 semantics. General
shader translation libraries may be useful for validation or bootstrapping, but they are not the
canonical guest IR.

## Compatibility strategy

The fastest route to games is hybrid:

1. HLE kernel, memory, filesystem, libc, threading, synchronization, timing, input, VideoOut, and
   AGC-facing APIs.
2. Load game-bundled PRX modules natively when they are available and resolvable.
3. Permit optional, user-dumped system modules for functionality where HLE would be disproportionately
   expensive.
4. Use host libraries for codecs only when their licenses and redistribution terms are compatible.

