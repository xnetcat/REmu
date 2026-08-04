# Direct3D 12 backend integration

## Current status and support baseline

As of 2026-08-03, `remu-gpu-dx12` is an executable backend rather than a device probe. It selects a
hardware DXGI adapter, creates the D3D12 device/queue/fence, translates the shared Gen5 shader IR to
HLSL and DXIL, executes ordered compute and graphics events, copies results back to guest memory,
and presents VideoOut buffers through a winit-owned Win32 window and a DXGI flip-model swap chain.
The Windows `remu run` command is enabled and Dreaming Sarah reaches its first playable forest
layout, including scene scrolling under guest pad input, through this path.

Implemented and physically validated on the development host:

- DXGI high-performance adapter selection and exact `CheckFeatureSupport` capability queries.
- Live DXGI local/non-local process budgets, current usage, available reservation, and current
  reservation through `IDXGIAdapter3::QueryVideoMemoryInfo`; these remain distinct from the
  adapter's static dedicated-memory figure.
- Shader Model 6.7+ HLSL generation and DXC compilation for translated vertex, pixel, and compute
  shaders, with bounded shader and pipeline-state caches.
- Buffer SRV/UAV/CBV resources, sampled and writable 2D textures, single-mip standard-swizzled 3D
  textures, samplers, render-target upload and writeback, multiple render targets, blending,
  raster state, viewports, scissors, depth, and stencil.
- Native two-plane D32/S8 upload/readback footprints, including independent front/back stencil
  references and masks through the D3D12 pipeline-state stream and command-list 8 interfaces.
- 2x/4x/8x color and depth sample-count decoding with adapter quality queries; normalized/float
  color resolves, explicit sample-zero signed/unsigned integer resolves, D16/D32 MIN/MAX depth
  resolves, and plane-aware D32/S8 depth/stencil resolves. Single-sample guest contents are
  broadcast into MSAA attachments before rendering so load semantics are preserved.
- GFX10 tiled texture conversion for the observed mode-24 and mode-27 surfaces, complete mode-5
  and mode-27 2D mip chains including packed mip tails, and thick mode-5/mode-9 3D blocks. Mode-27
  parity tables are cached and the four-byte scanout path uses bounded unaligned pixel copies
  instead of rebuilding offsets per 64 KiB block.
- Persistent single-sample render targets keyed by guest address and descriptor, with Windows VEH
  write watches for CPU/host invalidation. Coherent targets skip redundant 4K uploads and retain
  their already-linear output for DXGI presentation while still writing guest-visible bytes back.
- Ordered compute, graphics, barriers, flip events, display registration, XInput polling, and DXGI
  presentation, including retained-frame restore across minimize, occlusion, suspension, and
  zero-sized resize transitions plus borderless F11/Alt+Enter fullscreen.
- Native D3D12 occlusion queries around guest draws, resolved through a bounded query/readback
  ring and published through the shared guest counter contract. This is baseline D3D12 behavior,
  not an adapter-vendor extension.
- Structural shader and pipeline variants cache translated state independently of live guest
  addresses. Shared shader-visible descriptor/sampler arenas, a persistently mapped 64 MiB upload
  arena, and fence-retired command allocator/list reuse remove per-draw heap and command-object
  construction from the steady-state path.
- CPU-side queue waits use `ID3D12Fence::SetEventOnCompletion` with a Win32 event and a bounded
  wait, shared by self-tests, guest execution, and presentation. A signaled removal fence is
  checked with `GetDeviceRemovedReason` instead of being mistaken for successful completion.
- Opt-in DRED 1.2 configuration occurs before device creation. Device-removal errors collect
  bounded auto-breadcrumb lists, breadcrumb contexts, last/next GPU operations, device state,
  page-fault address/flags, and matching live or recently-freed allocations. Guest queues,
  command lists, fences, render/depth targets, resolves, and textures carry stable diagnostic
  names containing submission or guest-address identity.
- Pixel-shader interpolation metadata drives an exact vertex-output signature. Sparse semantic
  sets such as `TEXCOORD0` plus `TEXCOORD2` no longer retain an unused vertex `TEXCOORD1` that
  causes DXIL stage-link register mismatches on D3D12.
- Headless copy/readback, translated compute/graphics, tiled-resource, depth/stencil, mip-chain,
  primitive-expansion, lifecycle/input-neutralization, DRED-format, and physical-GPU tests. The
  current D3D12 crate has 50 passing tests on the GTX 1050.

This is a playable vertical slice for one tested title, not yet a broad playability claim. The major remaining work is listed below: guest FMASK and full DCC
metadata coherence plus persistent multisample/depth-resource identity; NGG/mesh fallback for devices without mesh
shaders; patch/rectangle/triangle-with-flags edge semantics; mipped arrays/3D and writable-mip UAVs;
cross-draw submission batching and resource retirement; DRED/debug-layer capture; and
broader title validation.

The initial REmu baseline is:

- Direct3D feature level 12_0 or newer.
- Shader Model 6.7 or newer, compiled as DXIL with DXC.
- HLSL wave operations.
- Resource binding tier 3.

Feature level, wave capability, shader model, and resource-binding tier are queried independently.
Optional features used to select exact or fallback paths include enhanced barriers, writable MSAA
textures, independent front/back stencil reference and masks, triangle fan, dynamic strip-cut, and
dynamic depth bias. Mesh-shader tier is also queried independently: tier 1-capable adapters may use
a native mesh path once implemented, while tier-0 adapters require the classic/compute NGG
fallback. This follows Microsoft's guidance to use
`ID3D12Device::CheckFeatureSupport` rather than infer behavior from adapter names or DirectX
marketing versions.

No execution path is selected from adapter name, vendor ID, or device ID. Those values are retained
only for diagnostics, cache identity, and an eventual explicit adapter selector. The GTX 1050
results below are one physical validation point, not the supported-device definition; newer AMD,
Intel, and NVIDIA adapters can use optional capabilities unavailable on that card without changing
the baseline path for older hardware.

DXGI adapters are enumerated with `DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE`, software adapters are
skipped, and the first adapter satisfying the baseline is selected. Adapter selection must later
accept an explicit LUID override for multi-GPU debugging and user configuration.

## Development-host result

The probe was run on 2026-08-03 through `cargo run -p remu-cli -- gpu-info`:

```text
api=Direct3D 12
adapter=NVIDIA GeForce GTX 1050
vendor_id=0x10de
device_id=0x1c8d
dedicated_video_memory_mib=4007
local_memory_budget_mib=3405
local_memory_current_usage_mib=6
local_memory_available_for_reservation_mib=1803
local_memory_current_reservation_mib=0
non_local_memory_budget_mib=7458
non_local_memory_current_usage_mib=8
feature_level=12_1
shader_model=6.8
resource_binding_tier=3
resource_heap_tier=1
wave_ops=true
wave_lane_range=32..=32
int64_shader_ops=true
mesh_shader_tier=0
uma=false
cache_coherent_uma=false
enhanced_barriers=true
writable_msaa_textures=true
independent_front_back_stencil_ref_mask=true
triangle_fan=true
dynamic_index_buffer_strip_cut=true
dynamic_depth_bias=true
supported=true
```

The GTX 1050 is supported by the implemented baseline. The reported local budget is a live sample
and can change as other processes or focus state change; it is already lower than the static
4,007 MiB dedicated-memory label. Its native wave size is 32, so the shader
backend must not assume that a host wave matches a guest RDNA2 wave64. The Intel UHD 630 also
reports DirectX feature level 12_1 through `dxdiag`, but the discrete NVIDIA adapter correctly wins
high-performance selection.

The physical test suite exercises copy/readback, graphics, compute, translated guest shaders,
triangle-fan support, independent stencil state, normalized/integer color MSAA, D32 MSAA, and
combined D32/S8 MSAA on this adapter. A traced Dreaming Sarah run processed 10,725 physical draws
and 840 frames; its steady-state mean reached 535 microseconds per draw and median 413 microseconds.
An untraced 20-second release run presented 998 frames and settled near 53--54 FPS after startup.
New Game then entered `M0T0`, completed its intended white/title fade, rendered the forest and
Sarah, and scrolled to the right under a live guest D-pad event. The audio callback remained active
with zero underrun frames and no D3D12 execution or pipeline error was reported. The 4 GB physical
VRAM figure is not a usable allocation budget; the live probe measured a 3,405 MiB local budget.

A later DRED-enabled 24-second release regression reached `M0T0` and completed submission 1002.
Its 48 kHz stereo WASAPI callback remained active with zero underrun frames, and the captured logs
contained no panic, access violation, backend error, shader-evaluation failure, or device removal.

A subsequent live lifecycle run toggled fullscreen on and off through both F11 and Alt+Enter,
minimized and restored the active game window, and continued to submission 584 before the bounded
timeout. The process stayed alive during both transitions and the audio callback remained active
with zero underrun frames.

Poppy Playtime exposed a distinct single-mip 3D-layout defect. Its 16x16x16 R16 mode-5 UAV is two
thick 4 KiB blocks (8 KiB total); treating depth slices as sixteen independently padded 2D images
incorrectly produced a 64 KiB footprint and overwrote adjacent MallocBinned3 free-list nodes. The
corrected GFX10 S3 address equations survived a 75-second DRED-enabled run, advanced from the old
deterministic access violation at submission 16 to submission 595, kept the GPU worker active, and
left WASAPI active. Cumulative VideoOut telemetry initially appeared to show nine alternating-buffer
flips in a follow-up 60-second run, but one-shot device-removal reporting proved only the first three
were native completions; the backend had continued retiring failed presentations after the device
was lost. This remains a crash and presentation-progress result, not yet a Poppy playability claim:
current per-draw/dispatch fencing and guest-memory publication still make the workload much too slow.

The removal came from a live indirect-argument buffer containing an X dimension of `0x18003fff`
(402,669,567 groups), far above Direct3D 12's 65,535-groups-per-dimension limit. `Dispatch` has no
HRESULT return, so forwarding the value removed the device and only the next allocator call revealed
the failure. The executor now validates all direct and live-resolved indirect dimensions before
shader/resource work. A subsequent 60-second DRED run rejected each oversized dispatch, completed
six real flips, and reached submission 137 without device removal. One-shot removal reports now
include DRED, page-fault, and available SDK debug-queue state; `REMU_TRACE_D3D12_BINDINGS=1` adds
per-dispatch resource evidence for diagnosis. For buffers of at most 64 bytes, that trace includes
the captured input, GPU readback, guarded publication result, and immediately observed guest words.
This is intentionally opt-in because it snapshots live guest memory and produces high-volume logs.

Translated-shader diagnostics are also opt-in. Set `REMU_DUMP_HLSL_DIR` to a directory to retain
vertex/pixel HLSL and the corresponding typed guest IR. Compute programs are numerous, so they are
only retained when `REMU_DUMP_HLSL_CS=all` or when that variable is the hexadecimal guest shader
address printed in the AGC log (with or without `0x`). These files contain translated diagnostic
material and must not be committed when they originate from a proprietary game.

The Poppy counter pipeline has since been isolated further: a one-group reset clears a five-dword
counter buffer, a 23x13 classifier accumulates into it, and a five-lane expander produces indirect
triplets. Packet-time buffer snapshots can predate those GPU writes even when the events themselves
remain ordered. The D3D12 executor now retains a bounded queue of GPU-produced buffer versions,
keys leased entries by physical mapping identity, and forwards a producer only when a consumer's
packet bytes still match either the producer baseline or its current output. A genuinely newer CPU
packet wins. Ordered acquire, DMA, `WRITE_DATA`, and `RELEASE_MEM` destinations invalidate
overlapping virtual or aliased physical entries. The cache is limited to 128 entries, 8 MiB per
entry, and 64 MiB total.

That forwarding produces the intended counter `0x2428` and indirect triplet `[0x2428, 1, 1]`
instead of the recycled `0x18003fff` sentinel. A physical GTX 1050 regression recreates the exact
five-thread formatted-buffer expansion into a 65,532-byte UAV and verifies all five triplets.
Poppy then reaches compute 196 with the correct 9,256-group X dimension. It is not playable yet:
the next blocking graphics feature is an NGG export using LDS/barriers on a device with no mesh
shader tier, and current per-draw/dispatch fencing still makes this workload much too slow for
interactive use. Per-format
`D3D12_FEATURE_DATA_FORMAT_SUPPORT` checks confirm that this adapter advertises the required typed
UAV store support.

The apparent resource-construction removal at compute 196 was isolated to the third void
`ID3D12Device::CreateSampler` call. AMD's descriptor exposes independent U4.8 minimum/maximum LOD
bounds and an S5.8 bias, while `D3D12_SAMPLER_DESC` requires `MaxLOD >= MinLOD` and limits bias to
[-16, 15.99]. Poppy supplied both an inverted interval and a bias above that D3D12 range. The
backend now clamps the bias and collapses an inverted interval at its minimum, the closest legal
D3D12 representation that preserves the guest lower bound. A regression covers the exact invalid
descriptor. With this translation, the GTX 1050 submitted compute 196 as `[9256, 1, 1]`, executed
compute 197, remained available, and continued processing until the bounded 50-second test ended.
Trace-mode descriptor creation also checks removal immediately after every texture and sampler view
so future void-API failures are attributed to the exact descriptor.

The following graphics capture proved that Poppy's formerly rejected targets are single-sample DCC
(`fast_clear=0`, `fmask=0`, `dcc=1`), not FMASK. The DX12 path now follows the same conservative
policy as Metal: compressed first-use bytes are never reinterpreted as pixels; a deterministic clear
materializes uncompressed native storage, which is then retained across draws. CPU ownership changes
force rematerialization. This is enough for the observed DCC-only RGBA16F target, while actual FMASK
and fast-clear metadata remain explicit work rather than silent corruption. The same capture added
Gen5 ALT mappings for BGRA8 sRGB and RGBA16F render targets. It also exposed an HLSL-only bug in the
pixel private-lane tables: DXC rejects a one-element initializer for a 32-element local array. The
generator now explicitly initializes every lane and ADDTID spill record, with a physical DXIL
regression for Poppy's `V_WRITELANE`/`V_READLANE` pattern.

Device removal is terminal. The presenter emits one bounded DRED report and exits immediately;
it no longer consumes queued work, reports cascaded failures, or retires fake submissions/flips.
Fence completion is followed by a removal-state query to reduce asynchronous misattribution.
`REMU_TRACE_D3D12_BINDINGS=1` now includes the four live buffer-descriptor SGPRs, which proved that
Poppy's expander uses the same 32-bit unsigned format covered by the physical regression. The SDK
debug layer remains unavailable on this machine until the optional Windows Graphics Tools feature
is installed, so DRED and fail-closed native checks remain the current diagnostic boundary.

## Crate and data-flow boundaries

The existing split remains the right one:

```text
AGC / PM4 decode (`remu-runtime`)
    -> guest-domain events and resources (`remu-gpu`)
    -> typed Gen5 shader IR (`remu-shader`)
       -> HLSL generation (`remu-shader-metal` shared compiler) + DXIL/DXC (`remu-gpu-dx12`)
    -> D3D12 heaps, resources, pipelines, queues, fences (`remu-gpu-dx12`)
    -> DXGI flip-model swap chain hosted by winit
```

No D3D12, DXGI, HLSL, or DXIL types belong in `remu-gpu`, `remu-shader`, or the AGC decoder.
Conversely, `remu-gpu-dx12` must not reinterpret PM4 packets: it consumes the already ordered,
validated guest-domain events. This lets Metal and D3D12 share console semantics without forcing
both backends through a lowest-common-denominator rendering abstraction.

## Device, queues, and presentation

The executable presenter creates:

1. The selected adapter, `ID3D12Device`, direct queue, and fence.
2. Fence-retired command allocators/lists and shared descriptor, sampler, and upload arenas, with
   dedicated-resource fallback for oversized allocations.
3. A winit window and `IDXGISwapChain3` using flip-model presentation.
4. Back-buffer RTVs and explicit `GetCurrentBackBufferIndex` rotation.
5. An ordered event receiver that executes guest work before acknowledging VideoOut flips.

The event receiver is bounded. Pipeline compilation or a slow GPU applies backpressure
instead of dropping ordered guest work or growing memory without limit. Window resize is deferred
until all back buffers are idle; occlusion and minimize suspend presentation without losing guest
queue progress. Tearing is used only when DXGI reports support and the user has not requested
VSync.

The current executor persists coherent single-sample color targets, hands their linear output
directly to presentation, and uses CPU write watches to invalidate them. Descriptor and upload
arenas plus command objects are recycled after their fence completes. The executor still waits for
a fence and performs guest-memory writeback after each draw or dispatch. The next performance
architecture extends persistence to depth/MSAA/shader resources and batches command lists without
weakening guest-visible acquire/release ordering.

## Guest memory and D3D12 resources

The console and Apple unified-memory path can expose guest bytes more directly than a discrete
Windows GPU. On the GTX 1050, guest CPU memory cannot be treated as a default-heap render resource.
The Windows design therefore needs explicit ownership:

- CPU-visible guest mappings remain owned by `remu-memory`.
- Immutable/transient buffer data moves through persistently mapped upload pages.
- Textures and render/depth targets live in default heaps and are keyed by guest mapping identity,
  descriptor state, format, extent, tiling mode, and generation.
- CPU writes invalidate or incrementally upload affected guest ranges using the write-watch
  contract. GPU writes stay in the default resource until a guest-visible acquire, CPU read, alias,
  or release requires readback.
- `GuestRangeLease` identity checks must guard delayed uploads/readbacks so a recycled virtual
  address never publishes into a new mapping.
- Writable textures retain the exact raw upload snapshot. Completion retiles into that baseline
  and publishes only baseline-matching chunks; newer guest bytes win, retired mappings discard the
  completion, and one-megabyte coalescing avoids turning random image changes into millions of
  backing writes.
- Direct and flexible memory aliases must resolve to one physical identity even when several guest
  virtual ranges name it.

GFX10 standard-swizzled 3D resources are thick, not arrays of padded 2D slices. For mode 5
(`ADDR_SW_4KB_S`), the 4 KiB block dimensions for 1/2/4/8/16-byte elements are respectively
16x16x16, 8x16x16, 8x16x8, 8x8x8, and 4x8x8. Mode 9 uses the corresponding 64 KiB S3 dimensions
and the same low address equation extended through bits 12--15. The implementation derives the
in-block X/Y/Z bit order from AMD AddrLib's GFX10 S3 pattern tables and calculates the outer block
grid from pitch, height, and depth. Thin 3D modes continue to use their documented slice/XOR path.

Committed resources are acceptable for the vertical slice. Placed-resource heap suballocation and
aliasing barriers follow after correctness. Residency must use live `QueryVideoMemoryInfo` budgets;
Microsoft explicitly warns that budgets fluctuate and that dedicated-memory size is not the
process budget.

## Shader translation

The shared shader compiler lowers the existing typed IR to generated HLSL; the D3D12 backend invokes
Microsoft's DirectX Shader Compiler and creates Shader Model 6 DXIL pipelines. Cache keys include:

- Canonical guest program/IR identity and supplemental entry points.
- Stage, exports, resource bindings, render-target formats, and fixed pipeline state.
- Shader-model target, capability bits, compiler version, arguments, and backend schema version.

The persistent cache stores validated DXIL, never proprietary game bytes. Its key covers generated
HLSL, entry point, shader-model profile, compiler arguments, cache schema, and DXC file metadata.
`REMU_DXIL_CACHE_DIR` overrides its location and `REMU_DISABLE_DXIL_CACHE` disables disk access.
Pipeline-library persistence remains separate because it additionally needs adapter and driver
identity in its compatibility key.

Wave behavior retains an explicit lowering policy. This host exposes only wave32. Operations whose
guest semantics span 64 lanes require two-wave coordination through group-shared memory and
barriers, or another proven lowering; native HLSL wave intrinsics alone cover only one host wave.
The translator must also preserve the existing Metal backend's ordering, scalar-state, descriptor,
and memory-visibility rules rather than inventing a second guest interpretation.

DXC should be loaded through `dxcompiler.dll` and shipped with the matching validator/runtime files
when redistribution is introduced. Development may use the Windows SDK compiler. The inbox D3D12
runtime is sufficient for the initial backend; adopting the Agility SDK is a separate packaging
decision and cannot be implicit.

## Synchronization and diagnostics

Every resource has tracked access/state. `GpuPipelineBarrier` and acquire/release events map to
transition, UAV, or aliasing barriers plus queue fences according to the guest intent. Enhanced
barriers may be used when supported, but the legacy barrier path remains the compatibility
baseline until both paths have equivalent tests.

CPU waits no longer poll `GetCompletedValue` in a yield loop. The executor, presenter, copy test,
and pipeline test all arm an auto-reset Win32 event through `SetEventOnCompletion`, block with a
five-second timeout, then distinguish ordinary completion, timeout, unexpected wait status, and
device removal. The physical regression test covers asynchronous CPU fence signaling and the
timeout path.

The debug layer, GPU-based validation, and DRED are opt-in because they cost performance. The CLI
flags are `--d3d12-debug`, `--d3d12-gpu-validation`, and `--d3d12-dred`; matching `REMU_*`
environment variables support development launchers. DRED configuration occurs before every
device enumeration/creation pass and removal detected by fence wait or DXGI presentation appends a
bounded report to the originating error. All persistent and submitted D3D12 objects involved in
the guest/presentation paths receive stable names derived from guest addresses, shader addresses,
or submission sequences. Device removal remains fatal; recovery can be designed only after
deterministic teardown exists. The remaining validation task is an intentional removal capture on
disposable hardware, because deliberately triggering a TDR on a working desktop is disruptive.

## Remaining implementation sequence

### Parity audit against post-divergence main

The 2026-08-04 rebase audited the GPU/runtime work added to main after this branch diverged and maps
it to DX12 by semantic capability rather than by copying Metal APIs:

| Main work | Shared or DX12 equivalent | Current state |
| --- | --- | --- |
| Structured shader CFG (`6f32225`) | Shared IR plus HLSL structured/direct lowering | Integrated and covered by the shared compiler tests |
| Ordered internal-resident writes (`ccc7640`) | Guest-range leases and ordered global versions | Integrated |
| Attachment occlusion queries (`d683bb5`) | D3D12 occlusion query heap and readback publication | Implemented |
| GPU tiler and equation work (`6ed02f8`) | Shared equations plus a future D3D12 compute tiler | CPU conversion exists; GPU path remains |
| Persistent shader caches (`dd62f2b`) | Compiler-keyed DXIL plus adapter/driver-keyed pipeline-library cache | Persistent DXIL implemented; pipeline-library disk cache remains |
| Display-link timebase/pacing (`4f6537e`) | DXGI frame-latency waitable object and shared display timebase | Waitable-object pacing integrated; shared display-time telemetry remains |
| NGG capacities and proof (`ca6c3a0`, `655d3d8`) | Capability-routed native mesh path plus classic/compute fallback | Shared register derivation is available; DX12 execution paths remain |
| Sampled-resident lookup (`30bd087`) | Physical-identity-indexed DX12 texture lookup | Descriptor cache exists; identity index remains |

This table is a porting checklist, not a claim that Metal implementation details should become the
cross-platform contract. Guest semantics and capability decisions belong in shared code; D3D12
resource transitions, queues, queries, DXIL, and DXGI pacing remain backend-specific.

1. Extend the completed ordered single/multi indirect draw/dispatch and primitive normalization
   paths to patch, triangle-with-flags, and rectangle-specific edge semantics.
2. Move the completed transient color/depth/stencil MSAA paths into persistent resources and extend
   conservative DCC materialization to full guest FMASK/DCC metadata coherence plus explicit guest
   resolve-packet semantics.
3. Add a classic/compute NGG fallback for every `mesh_shader_tier=0` adapter, validate it on the
   GTX 1050 and against the shared compiler corpus, then add a separately selected native mesh path
   for adapters reporting mesh-shader tier 1 or newer.
4. Extend the persistent single-sample color cache to depth/MSAA/shader resources and replace
   synchronous per-draw fences with upload/readback rings, fence retirement, physical-alias
   invalidation, residency budgets, and submission batching.
5. Complete mipped array and mipped-3D UAV support beyond the implemented single-mip mode-5/mode-9
   volume path, layered depth targets, DCC/FMASK metadata, and remaining stencil logical/op-value
   emulation.
6. Validate the implemented debug-layer/DRED path under intentional device removal, add GNU/MSVC
   CI and packaging, run long window/audio/input lifecycle tests, and expand game probes.

Every step has a synthetic test or CLI probe. Game testing supplements these checks after local
dumps are available; it does not replace them.

## Primary references

- [Direct3D 12 programming guide](https://learn.microsoft.com/en-us/windows/win32/direct3d12/directx-12-programming-guide)
- [D3D12 hardware feature levels](https://learn.microsoft.com/en-us/windows/win32/direct3d12/hardware-feature-levels)
- [`ID3D12Device::CheckFeatureSupport`](https://learn.microsoft.com/en-us/windows/win32/api/d3d12/nf-d3d12-id3d12device-checkfeaturesupport)
- [`D3D12_FEATURE_DATA_FORMAT_SUPPORT`](https://learn.microsoft.com/en-us/windows/win32/api/d3d12/ns-d3d12-d3d12_feature_data_format_support)
- [`D3D12_FORMAT_SUPPORT2`](https://learn.microsoft.com/en-us/windows/win32/api/d3d12/ne-d3d12-d3d12_format_support2)
- [`D3D12_SAMPLER_DESC`](https://learn.microsoft.com/en-us/windows/win32/api/d3d12/ns-d3d12-d3d12_sampler_desc)
- [Direct3D 12 constants, including the legal mip LOD bias range](https://learn.microsoft.com/en-us/windows/win32/direct3d12/constants)
- [`IDXGIFactory6::EnumAdapterByGpuPreference`](https://learn.microsoft.com/en-us/windows/win32/api/dxgi1_6/nf-dxgi1_6-idxgifactory6-enumadapterbygpupreference)
- [`IDXGIAdapter3::QueryVideoMemoryInfo`](https://learn.microsoft.com/en-us/windows/win32/api/dxgi1_4/nf-dxgi1_4-idxgiadapter3-queryvideomemoryinfo)
- [D3D12 memory-management strategies](https://learn.microsoft.com/en-us/windows/win32/direct3d12/memory-management-strategies)
- [Executing and synchronizing command lists](https://learn.microsoft.com/en-us/windows/win32/direct3d12/executing-and-synchronizing-command-lists)
- [`ID3D12Fence::SetEventOnCompletion`](https://learn.microsoft.com/en-us/windows/win32/api/d3d12/nf-d3d12-id3d12fence-seteventoncompletion)
- [`ID3D12Device::GetDeviceRemovedReason`](https://learn.microsoft.com/en-us/windows/win32/api/d3d12/nf-d3d12-id3d12device-getdeviceremovedreason)
- [Using DRED to diagnose GPU faults](https://learn.microsoft.com/en-us/windows/win32/direct3d12/use-dred)
- [D3D12 index-buffer strip-cut values](https://learn.microsoft.com/en-us/windows/win32/api/d3d12/ne-d3d12-d3d12_index_buffer_strip_cut_value)
- [DirectX triangle-fan and dynamic strip-cut specification](https://microsoft.github.io/DirectX-Specs/d3d/VulkanOn12.html)
- [Mesa GFX10.3 register descriptions](https://chromium.googlesource.com/chromiumos/third_party/mesa/+/a11d1bd247df18057ecb9945e01bf39ff85d8d9d/src/amd/registers/gfx103.json)
- [Mesa GFX10 resource-descriptor field definitions](https://chromium.googlesource.com/chromiumos/third_party/mesa/+/refs/heads/stabilize-13982.70.B-chromeos-amd/src/amd/registers/gfx10-rsrc.json)
- [AMD AddrLib GFX10 swizzle pattern tables](https://github.com/GPUOpen-Drivers/pal/blob/dev/src/core/imported/addrlib/src/gfx10/gfx10SwizzlePattern.h)
- [AMD AddrLib GFX10 resource and thick-block rules](https://github.com/GPUOpen-Drivers/pal/blob/dev/src/core/imported/addrlib/src/gfx10/gfx10addrlib.h)
- [`ID3D12Device::GetCopyableFootprints`](https://learn.microsoft.com/en-us/windows/win32/api/d3d12/nf-d3d12-id3d12device-getcopyablefootprints)
- [`ID3D12GraphicsCommandList::CopyTextureRegion`](https://learn.microsoft.com/en-us/windows/win32/api/d3d12/nf-d3d12-id3d12graphicscommandlist-copytextureregion)
- [`ID3D12GraphicsCommandList::ResolveSubresource`](https://learn.microsoft.com/en-us/windows/win32/api/d3d12/nf-d3d12-id3d12graphicscommandlist-resolvesubresource)
- [`ID3D12GraphicsCommandList1::ResolveSubresourceRegion`](https://learn.microsoft.com/en-us/windows/win32/api/d3d12/nf-d3d12-id3d12graphicscommandlist1-resolvesubresourceregion)
- [D3D12 programmable sample positions and depth/stencil resolve semantics](https://microsoft.github.io/DirectX-Specs/d3d/ProgrammableSamplePositions.html)
- [Mesa GFX10 indirect-multi packet emission](https://chromium.googlesource.com/external/gitlab.freedesktop.org/mesa/mesa/+/e38c01886c76399699d5859431878f710e7e712a/src/amd/vulkan/radv_cmd_buffer.c)
- [Planar depth/stencil resource specification](https://microsoft.github.io/DirectX-Specs/d3d/PlanarDepthStencilDDISpec.html)
- [Pipeline-state streams](https://learn.microsoft.com/en-us/windows/win32/direct3d12/managing-graphics-pipeline-state-in-direct3d-12)
- [DirectX-Headers pipeline-state stream helper](https://github.com/microsoft/DirectX-Headers/blob/main/include/directx/d3dx12_pipeline_state_stream.h)
- [DRED device-removal diagnostics](https://learn.microsoft.com/en-us/windows/win32/direct3d12/use-dred)
- [DirectX Shader Compiler](https://github.com/microsoft/DirectXShaderCompiler)
- [DirectX 12 Agility SDK introduction](https://devblogs.microsoft.com/directx/gettingstarted-dx12agility/)
- [winit Windows API](https://docs.rs/winit/latest/winit/platform/windows/)
