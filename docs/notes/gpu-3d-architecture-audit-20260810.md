# GPU 3D architecture audit — 2026-08-10

This was an offline code, SDK, archived-log, and upstream-source audit. The emulator and game were
not launched. Mesa was cloned read-only at `research/upstreams/mesa`, commit `aebd9a0`, from the
official freedesktop GitLab repository. Silent Hill remains a test case, not a title-specific code
path, and the content-warning hallway remains a 2D scene rather than the 3D checkpoint.

## What the archived frame already proved

The spatially decomposed but recognizable 3D frame was restored by clean commit `3f9e559`. A
same-address, empty R32Float sampling shadow had taken ownership from the native Depth32Float
attachment. Preserving native depth authority raised `depth_native` from 0 to 469 and made the
post-BasePass scene-color writer nonzero. That is the proven geometry restoration; later dirty
experiments must not be credited for it.

The later final black frame has a different proven cause. The last compositor samples a native
32³ RGB10A2 LUT whose 32 captured planes are all `[0,0,0,1]`. The preceding linked NGG writer had
allocated 2,176 bytes for a 2,176-dword (8,704-byte) GS LDS request, so accesses through byte
`0x1d20` were guarded out and the mesh positions stayed zero. The D3 target, RGB10A2 conversion,
layered render-pass setup, and full-volume capture path were re-audited and are correct for this
base-zero/full-depth LUT. The black native planes are therefore credible contents rather than a
readback artifact.

## General fixes made in this audit

| Area | Fault | Correction |
| --- | --- | --- |
| NGG input ABI | Passthrough mode put PrimitiveID in `v1` and zeroed `v2`. | GFX10-11 passthrough now keeps only the packed primitive in `v0`; PrimitiveID remains in fixed ABI `v2`. Referenced undefined `v1` is deterministic zero. |
| NGG sizing fallback | `UcGeCntl.VERT_GRP_SIZE` was reused as `PRIM_AMP_FACTOR` and `VGT_GS_MAX_VERT_OUT`. A common 64/64 no-GS group could be inflated to 4,096 output primitives and rejected. | No-GS amplification is exactly one. GS amplification requires the independent nonzero `VGT_GS_MAX_VERT_OUT`; missing/invalid GS state refuses loudly. |
| Linked ES/GS stride | Packed GS offsets always multiplied local indices by four dwords. | `VGT_ESGS_RING_ITEMSIZE` (Cx `0x2ab`) is captured in `NggRegisters`, included in compile identity, and applied dynamically. Missing linked-stage stride refuses instead of inventing four. |
| Mesh threadgroup memory | Validation counted guest LDS but missed compiler-generated layer and split-wave arrays; one helper used a hardcoded 64-entry estimate. | Exact emitted capacities now contribute to the 32-KiB limit. A full guest allocation plus helper state refuses before Metal pipeline creation. |
| Scalar decode | The GFX10 SOPK tail and `S_BITREPLICATE_B64_B32` used GFX11 opcode numbers. `S_SETREG_IMM32_B32` could be framed as one dword and desynchronize the remainder of a shader. | Corrected GFX10 codes, including the two-dword opcode `0x15`, CALL `0x16`, waits `0x17..0x1a`, and bit-replicate `0x3b`. |
| Scalar condition | `S_ADDK_I32` did not define SCC in either evaluation, liveness, or Metal output. | SCC now reports signed 32-bit addition overflow without invoking signed overflow in MSL. |
| Scissor | Screen scissor was decoded as unsigned 15-bit, so `-1` became 32767; one window-offset-disable bit controlled three independent scissors. | Screen coordinates retain signed 16-bit semantics; window/generic/viewport offset-disable flags are applied independently. |
| Stencil | Disabled stale operations could reject a depth draw; AMD's independent `STENCILOPVAL` was silently replaced with `STENCILTESTVAL`. | Disabled stencil is inert. `REPLACE_OP` is accepted only when Metal's single reference can represent both values; otherwise it refuses loudly. |

The hardware comparisons use AMD's [RDNA2 ISA](https://www.amd.com/content/dam/amd/en/documents/radeon-tech-docs/instruction-set-architectures/rdna2-shader-instruction-set-architecture.pdf),
[CIK register reference](https://www.amd.com/content/dam/amd/en/documents/radeon-tech-docs/programmer-references/CIK_3D_registers_v2.pdf),
Mesa's fixed-revision [GFX10 register database](https://gitlab.freedesktop.org/mesa/mesa/-/blob/aebd9a0a2327aaaabb9d10be6cca8e5d1ed1fc85/src/amd/registers/gfx10.json),
[merged-GS argument layout](https://gitlab.freedesktop.org/mesa/mesa/-/blob/aebd9a0a2327aaaabb9d10be6cca8e5d1ed1fc85/src/amd/common/ac_shader_args.h#L126),
and [NGG sizing derivation](https://gitlab.freedesktop.org/mesa/mesa/-/blob/aebd9a0a2327aaaabb9d10be6cca8e5d1ed1fc85/src/amd/vulkan/radv_shader_info.c#L1285).

## Remaining high-priority 3D gaps

These are proven architectural gaps, but not all are proven participants in the captured Silent
Hill LUT draw.

1. **Split wave64 control flow.** Structured branches exchange EXEC/VCC between the two Metal
   SIMD32 halves, but forward and irreducible-dispatcher branches can still inspect half-private
   masks. SCC produced by B64 mask operations is also half-local, including before
   `S_CBRANCH_SCC*`. One guest wave can therefore choose two different edges.
2. **Split wave64 synchronization.** `V_READLANE`, `V_READFIRSTLANE`, and some LDS paths synthesize
   whole-threadgroup barriers at sites that are only guest-wave-uniform. Apple requires all threads
   in a threadgroup to reach a threadgroup barrier. Compute DS reads also lack general cross-half
   ordering after one AMD wave64 is split into two Metal SIMD32 groups, while `S_WAITCNT` remains a
   no-op. Until redesigned, unsafe mesh/compute patterns need focused rejection rather than
   optimistic synchronization. See Apple's
   [threadgroup guidance](https://developer.apple.com/documentation/apple-silicon/porting-your-metal-code-to-apple-silicon)
   and LLVM's [AMDGPU memory model](https://llvm.org/docs/AMDGPUUsage.html).
3. **Generic wave64 lane reads.** Non-threadgroup `V_READLANE` masks the selector with 31, so an
   architectural selector in 32..63 reads the wrong half instead of the selected guest lane.
4. **Real GS instancing.** `VGT_GS_INSTANCE_CNT` is decoded, but meshlet preparation does not
   execute primitive × invocation work and `v3` remains zero. Instanced geometry-shader draws can
   lose output.
5. **VOPC e64 coverage.** GFX10 VOP3/e64 comparison opcodes remain undecoded. They currently fail
   loudly, but this is meaningful missing shader coverage for future 3D titles.
6. **Array/cube descriptor base slices.** T# `BASE_ARRAY` is decoded and discarded from cache
   identity, tiler source offsets, and host layer mapping. Distinct array subviews can therefore
   alias and sample layer zero. The Prospero SDK defines absolute base/last slices, and Mesa
   preserves both in [descriptor construction](https://gitlab.freedesktop.org/mesa/mesa/-/blob/aebd9a0a2327aaaabb9d10be6cca8e5d1ed1fc85/src/amd/common/ac_descriptors.c#L533).
7. **Layered target views.** Color target arrays and nonzero-base D3 views are rejected. Supporting
   them requires separating allocation depth from active `(base_layer, layer_count)` and applying
   the base at the attachment/view, consistent with Metal's
   [renderTargetArrayLength](https://developer.apple.com/documentation/metal/mtlrenderpassdescriptor/1437975-rendertargetarraylength?language=objc).
8. **Writable D3/array authority.** Only storage type 9 receives non-evictable GPU ownership.
   ImageStore into D3/array textures can be lost on eviction or later sampled through stale guest
   RAM. This needs coordinated residency, alias retirement, publication, and tests—not a local
   descriptor patch.
9. **Texture overlap ranges.** Resource spans cover one face/base level, not all selected layers
   and mip tails, so overlapping writes may fail to retire stale sampled aliases.
10. **Color view compatibility.** The current classifier is narrower than Apple's bit-length view
   rule and misses compatible 32-bit aliases such as RGBA8/RG16/R32 and packed RGB10A2/RG11B10.
   Apple documents the rule in
   [`makeTextureView`](https://developer.apple.com/documentation/metal/mtltexture/maketextureview%28pixelformat%3A%29?changes=__1).
11. **Depth/R32 authority is still approximate.** The carve-out that preserved native depth and
    fixed the decomposed frame always prevents an R16/R32 storage binding from demoting depth. A
    real store through that alias must instead supersede or rematerialize the depth image. The rule
    ultimately needs actual write execution, not only the bound format.
12. **Authored mip coverage.** Outside exact mode-5/mode-27 paths, multi-mip textures upload one
    guest level and synthesize the rest with Metal mip generation. Authored guest mip contents are
    therefore lost; unsupported layouts should refuse until their complete AddrLib layout exists.
13. **D3 descriptor semantics.** Mip validation uses width/height but not depth, and word-5
    `MAX_MIP`/quilt-pitch behavior is not represented. A valid narrow/deep volume can be rejected,
    while a quilted descriptor can be treated as an ordinary array.
14. **MSAA resolve identity.** A same-layout texture in the address gap before the next 2-MiB
    boundary can be guessed to be the resolve target. This adjacency heuristic can select an
    unrelated allocation and should be replaced by recorded resolve relationships.
15. **BGR10A2 diagnostic capture.** Scanout decode can select BGR10A2, but native capture conversion
    does not implement it. This can break evidence capture for that scanout format; it cannot make
    the rendered frame itself black.

## Verification and limits

- `cargo test -p remu-shader`: 158 passed.
- `cargo test -p remu-gpu-metal -- --test-threads=1`: 502 passed, 5 ignored.
- Focused shader tests cover SOPK framing, `S_ADDK_I32` SCC, passthrough VGPRs, linked dynamic ESGS
  stride, LDS byte sizing, and compiler-generated threadgroup-memory accounting.
- Focused GPU-Metal tests cover NGG fallback, screen/window scissors, and stencil behavior.
- Full `remu-shader-metal`: 177 of 179 pass. The same two irreducible-CFG dispatcher expectation
  tests remain red; they predate this audit and are not a green package gate.
- No emulator/game process was run, so none of the new changes is live compatibility evidence and
  the title-background 3D gate remains open.
