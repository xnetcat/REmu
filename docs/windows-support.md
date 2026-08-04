# Windows host support

## Current state

Windows is now a game-execution target. The workspace and serial synthetic test suite pass locally
on `x86_64-pc-windows-gnu`; the CLI executes x86-64 guest code, manages the sparse guest address
space, translates and submits guest GPU work to D3D12, presents through DXGI, polls XInput, and opens
the default WASAPI output through the production mixer path.

On 2026-08-03, a release build launched Dreaming Sarah from its local `eboot.bin`, selected the
NVIDIA GeForce GTX 1050, registered the title's 3840x2160 display, entered the first `M0T0` forest
layout, rendered Sarah and the level, and accepted a live D-pad Right event through the guest
`scePadReadState` path. The camera advanced into the next part of the level. A separate untraced
20-second performance probe presented 998 frames, reaching about 53--54 FPS after startup, while
the WASAPI callback reported zero underrun frames. This is a playable single-title vertical slice,
not a broad compatibility claim.

On 2026-08-04 the currently installed Dreaming Sarah copy was probed again with both `eboot.bin`
and its read-only `.esbak` backup. Both enter guest code, initialize savedata, AGC, 3840x2160
VideoOut, input, and the 48 kHz stereo WASAPI path, but stop before the first guest GPU submission
while opening vertex-shader assets that are absent from the host `shaders` directory. This is a
local title-package completeness blocker; the VFS reports the exact resolved host paths and no
game files were modified. A complete copy is required for the next game-backed DX12 regression.

Poppy Playtime Chapter 1 now advances through sustained VideoOut, AVPlayer, WASAPI, and D3D12
execution. A 65-second release probe completed 244 guest flips, delivered 226 video frames while
the decoder produced 230 frames of the 10.010-second opening movie, and submitted 2,938,176 audio
output frames with zero underrun frames and no D3D12 device loss. The title has not reached its menu
yet: guest media progress is still about 6.9 times slower than real time, so this is an end-to-end
boot/media result rather than a playability claim.

The following table deliberately distinguishes foundations from runnable subsystems:

| Subsystem | Implemented and validated | Remaining compatibility/performance work |
| --- | --- | --- |
| Build/runtime | Warning-free GNU release build; GNU/MSVC-aware CRT linkage and dependency discovery; Windows FFmpeg bootstrap discovers Git Bash/pkgconf, accepts explicit overrides, and treats cached Windows paths case-insensitively | GNU/MSVC CI, redistributable packaging, crash bundles |
| CPU | Native x86-64 guest entry, compiler-supported System V callbacks/imports, thread and fiber switching, TLS, guarded guest reads, and release game execution | Larger ABI/register stress corpus, structured crash recovery, unwind metadata audit |
| Memory | Sparse placeholder aperture, reserve/commit/decommit, section-backed physical aliases, protection, mapping identity, leases, and VEH write watch with batched contiguous protection/query runs | Long-running alias/remap stress, budget telemetry, more hostile fault-order tests |
| Graphics | Capability-routed adapter probe with live local/non-local DXGI budgets, D3D12/DXC executor, compute/graphics, capability-neutral general-LDS NGG compute-to-raster fallback, persistent coherent single-sample targets, physical-alias-aware range publication, typed SRV component mapping, cached tiled images/mips, MRT/depth/stencil, normalized/integer color and depth/stencil MSAA, ordered indirect multi-draw, shared descriptor/upload arenas, recycled command lists, event-based bounded fence waits, bounded DRED capture, DXGI window, and native BGRA8/RGBA8/R10G10B10A2 presentation | Budget-aware cache trimming/residency, FMASK/DCC and remaining topology/format semantics, layered/extended NGG validation, depth/MSAA persistence, cross-draw submission batching, intentional device-removal capture validation |
| Audio | CPAL/WASAPI production mixer; 48 kHz stereo endpoint, invalidated-stream reopening, guest-port rebinding/resampler retargeting, reroute reporting, and live game callback validated | Physical unplug/default-route-change stress, latency/drift telemetry, long-running A/V synchronization |
| Window/input | winit Win32 event loop, DXGI surface, resize/close/focus flow, four-slot XInput hot-plug polling, dead zones/full-range axes/triggers, two-motor vibration, live keyboard-to-guest-pad press/release, neutral focus/suspend state, F11/Alt+Enter borderless fullscreen, and minimize/occlusion/suspend-safe presentation with retained-frame restore | Per-monitor display-change/sleep-resume and physical controller disconnect/reconnect stress; motion/touchpad and advanced DualSense output |
| Filesystem/libc | Game VFS loading, cursor-preserving positional I/O, GNU/MSVC CRT selection, save-data commits, and CNG random bytes | Case/permission/path edge cases and directory-metadata durability |

`remu run` is enabled on Windows. Unsupported operations remain fail-closed with a diagnostic rather
than silently fabricating guest behavior.

## Toolchains

`x86_64-pc-windows-msvc` and `x86_64-pc-windows-gnu` are the intended build configurations. Local
validation currently covers GNU; the CI matrix should treat MSVC as the normal packaged build and
GNU as a portability check.

Recommended MSVC prerequisites:

- Rust 1.93 or newer for `x86_64-pc-windows-msvc`.
- Visual Studio Build Tools with Desktop C++ and a current Windows SDK.
- Windows Graphics Tools for the D3D12 debug layer; PIX is optional.
- A supported DXC development kit or Windows SDK compiler when shader work begins.

Diagnostics are explicit because they have runtime cost. `--d3d12-debug` enables the SDK debug
layer, `--d3d12-gpu-validation` additionally enables slow GPU-based validation, and
`--d3d12-dred` enables DRED automatic breadcrumbs, breadcrumb contexts, and GPU page-fault
tracking before the first device is created. The equivalent environment variables are
`REMU_D3D12_DEBUG`, `REMU_D3D12_GPU_VALIDATION`, and `REMU_D3D12_DRED`; accepted true values are
`1`, `true`, `yes`, and `on`. If Windows Graphics Tools is absent, the SDK layer reports that fact
without disabling DRED or normal execution.

GNU builds need an actual UCRT64 MinGW toolchain on `PATH`, including `as`, `dlltool`, and the
linker/import libraries. A Rust sysroot may contain `dlltool` without the assembler that raw-DLL
imports require; `cargo check` can then pass while `cargo build` fails. Use the MSYS2 UCRT64
toolchain rather than mixing an MSVCRT MinGW toolchain with UCRT libraries.

The project does not hard-code `ucrt` for all Windows targets. Windows extern blocks select
`ucrt` under `target_env = "msvc"` and `msvcrt` under `target_env = "gnu"` where the same CRT owns
the `FILE` object. Passing a `FILE*` between two CRTs is invalid.

## CPU and exception boundary

The guest ABI is System V AMD64/FreeBSD-like; native Windows Rust ordinarily uses the Microsoft x64
ABI. Guest-facing entries and callbacks use Rust's Windows `extern "sysv64"` support, while explicit
assembly boundaries retain guest register and escape state. The implemented boundary:

- Enter guest code with System V register/stack rules and a controlled return/escape target.
- Route import stubs through a register-preserving System V-to-Windows call adapter.
- Preserve nonvolatile registers required by both ABIs and maintain Windows stack alignment/shadow
  space on every host call.
- Keep guest TLS emulation independent of the Windows TEB and validate every `%fs` rewrite path.
- Use vectored exception handling for guest access violations and illegal instructions, but recover
  only when the instruction pointer and fault address prove guest ownership. Host/runtime faults
  must continue to Windows error handling.
- Do not unwind Rust or Windows exceptions through arbitrary guest frames. Guest escape and thread
  exit use explicit assembly boundaries.
- Saves/restores fiber state on Windows and calls guest fiber entries with the console convention.

Synthetic ABI tests cover guest callbacks, import stubs, TLS, thread exit, fibers, nested HLE calls,
and guarded reads. Coverage must continue expanding to all integer registers, XMM arguments/returns,
stack arguments,
callee-saved state, nested HLE calls, faults, and multi-threaded entry as compatibility expands.

Potential guest pointers are validated through `ReadProcessMemory(GetCurrentProcess(), ...)`. The
API validates the complete requested span before copying, giving Windows the same fail-closed
behavior as the macOS Mach read and Linux `process_vm_readv` paths.

Primary reference: [`ReadProcessMemory`](https://learn.microsoft.com/en-us/windows/win32/api/memoryapi/nf-memoryapi-readprocessmemory).

## Virtual memory and write watching

The fixed image mapper aligns the owning reservation to Windows' allocation granularity while
retaining exact guest segment addresses and applying merged page protections. The full console
memory manager now uses sparse placeholders, partial commit/decommit, section-backed physical
aliases, and stable mapping identity.

The implemented Windows design is:

1. Reserve the guest aperture with `VirtualAlloc2` placeholders.
2. Represent physical/direct backing with section objects and map views into replaced placeholder
   ranges so aliases share bytes.
3. Preserve 16 KiB guest-page semantics over 4 KiB Windows pages in metadata and protection
   operations.
4. Decommit/release deterministically and increment mapping identity before addresses can be reused.
5. Use a vectored write-fault handler that temporarily opens tracked read-only pages, journals the
   dirty guest range, and later re-protects it after GPU synchronization.

`MEM_WRITE_WATCH`/`GetWriteWatch` does not solve section-backed aliases or fault-time ordering and is
therefore not the canonical coherency mechanism. REmu uses protected pages plus a vectored handler
and mapping-generation checks.

Primary references:

- [`VirtualAlloc2` and placeholders](https://learn.microsoft.com/en-us/windows/win32/api/memoryapi/nf-memoryapi-virtualalloc2)
- [`GetWriteWatch`](https://learn.microsoft.com/en-us/windows/win32/api/memoryapi/nf-memoryapi-getwritewatch)

## Audio

The existing mixer, stereo downmix, resampler, prebuffer, and bounded queues are host-independent.
CPAL is now enabled on Windows, where its default backend is WASAPI. `remu audio-info` opens,
starts, and drops the same default output stream that guest AudioOut uses; the development host
negotiated 48,000 Hz and two channels.

Initial shared-mode WASAPI is appropriate. Exclusive mode is not a default because it steals the
endpoint and complicates format/device recovery. Later work should measure callback period,
underruns, queue depth, and A/V clock drift, and decide whether CPAL's realtime feature improves
scheduling without hiding producer stalls.

The CPAL error callback classifies route-change, real-time-priority, and xrun notifications as
nonfatal. Device loss, host loss, stream invalidation, and backend failures atomically request a
rebuild. The next guest open/output call drops the invalid stream, reopens the current default
endpoint with a bounded initialization timeout, re-registers every live guest port in a fresh
mixer, retargets streaming resamplers to the new endpoint rate, and increments the guest-visible
reroute counter. Failed opens use the timed sink and retry once per second, preserving guest pacing
instead of converting a host-device outage into a guest ABI failure.

Primary references:

- [CPAL supported platforms and WASAPI backend](https://github.com/RustAudio/cpal)
- [Windows Core Audio APIs](https://learn.microsoft.com/en-us/windows/win32/coreaudio/about-the-windows-core-audio-apis)
- [Windows low-latency audio guidance](https://learn.microsoft.com/en-us/windows-hardware/drivers/audio/low-latency-audio)

## Windowing, input, and lifecycle

winit 0.30 remains the cross-platform event-loop/window owner. The D3D12 backend consumes its Win32
window handle and owns DXGI. Platform-independent code receives logical focus, resize, close, and
pad events; it must not receive `HWND`, messages, or swap-chain objects.

F11 and Alt+Enter toggle a borderless fullscreen window on the current monitor. Minimize,
occlusion, application suspension, and zero-sized resize events pause swap-chain presentation
without pausing ordered guest GPU execution. The transition into a paused state retains the latest
guest frame; later hidden flips retire without repeated full-resolution readback, and restore or a
nonzero resize refreshes the newest submitted frame once before normal presentation resumes.
Suspension also publishes a neutral pad state and clears held keyboard/modifier state so focus
transitions cannot leave guest buttons stuck. Per-monitor scale-factor changes request a redraw
through the same retained-frame path.

XInput slots 0 through 3 are polled continuously. The first connected slot feeds standard face,
D-pad, shoulder, stick-click, Options, trigger, and analog-stick state; slot connection,
disconnection, and failover are logged. Loss of focus or application suspension forces both
keyboard and controller data to a neutral guest state, suppresses dialog button edges, and clears
held keyboard/modifier state. Controller state remains sampled while inactive, so restoration
does not retain stale axes or buttons.

Guest `scePadSetVibration` large/small motor bytes publish atomically across the platform seam and
map to XInput's full 16-bit low/high-frequency motor ranges. Focus loss, suspension, pad close,
slot failover, disconnect, and presenter shutdown all issue or converge on a zero-motor state.
`remu input-info` reports every connected slot and `remu input-info --rumble-test` performs a
guarded 250 ms low-strength pulse that always sends a stop command as it exits.

Remaining Windows-specific stress coverage includes moving an active window across monitors with
different DPI, resize while GPU work is in flight, device/display changes, sleep/resume, keyboard
layout, XInput/GameInput controller hot-plug, and clean shutdown with audio/GPU workers active.

## Filesystem and randomness

Windows positional reads and writes save and restore the descriptor cursor because Rust's Windows
`FileExt::seek_read`/`seek_write` operations intentionally move it. Save-data commits flush regular
files through handles opened with write access, while directory-handle durability remains future
work. Guest cryptographic random requests use `BCryptGenRandom` with the system-preferred provider.

Primary references:

- [Rust Windows `FileExt`](https://doc.rust-lang.org/std/os/windows/fs/trait.FileExt.html)
- [`BCryptGenRandom`](https://learn.microsoft.com/en-us/windows/win32/api/bcrypt/nf-bcrypt-bcryptgenrandom)

## Validation state and remaining gates

Completed locally on the development Windows host:

1. Warning-free release CLI build and workspace check pass on the development Windows host.
2. `host-info`, `gpu-info`, `gpu-self-test`, `gpu-present-test`, `audio-info`, and `input-info`
   paths exist; the
   D3D12 and audio probes have run against physical hardware.
3. Fixed-image and sparse/alias/write-watch memory suites pass.
4. Runtime CPU, import, thread, TLS, fiber, filesystem, and libc suites pass.
5. The D3D12 crate's 61 tests pass, including event-based asynchronous fence completion and
   timeout, physical copy/readback, translated compute/graphics,
   ordered single/multi indirect arguments, tiled texture, mip-chain, primitive normalization,
   render/depth/stencil, normalized/integer color MSAA, D32 and combined D32/S8 MSAA, and
   independent-stencil coverage. A physical test executes a GFX10 general-LDS NGG shader through
   the compute fallback and verifies raster output on the mesh-shader-tier-0 GTX 1050. The serial
   runtime suite has 662 passing tests.
6. A bounded Dreaming Sarah release launch enters New Game, completes the opening fade, renders the
   first forest layout, moves right under live input, and keeps audio active without a D3D12
   failure, panic, or underrun.
7. A live Dreaming Sarah lifecycle probe enters and leaves borderless fullscreen through both F11
   and Alt+Enter, remains alive while minimized, restores and continues rendering, and reaches the
   timeout with an active WASAPI callback and zero underrun frames.
8. `remu --d3d12-dred gpu-self-test` enables automatic breadcrumbs, contexts, and page-fault
   tracking before device creation and passes copy, generated HLSL/DXIL, graphics, compute, and
   guest-shader validation on the GTX 1050. Synthetic removal-output tests cover bounded list
   traversal and command/resource naming. The optional SDK debug layer is not installed on this
   host and correctly degrades to a diagnostic instead of aborting.
9. `remu input-info --rumble-test` detects XInput slot 0 and physically validates both the query
   and guarded vibration-output path.
10. A final DRED-enabled 24-second release regression on 2026-08-03 entered `M0T0`, reached
    completed submission 1002, and kept the 48 kHz stereo WASAPI callback active with zero
    underrun frames. The bounded run reported no panic, access violation, backend error, shader
    evaluation failure, or D3D12 device removal.
11. A 65-second Poppy Playtime Chapter 1 release probe completed 244 flips, delivered 226 of 230
    decoded opening-movie frames, kept the 48 kHz stereo output active for 2,938,176 frames with
    zero underruns, and reported no D3D12 device loss. The menu/input path remains unvalidated
    because current guest execution is still substantially slower than real time.

Still required before a broad Windows compatibility claim:

1. GNU and MSVC CI plus clean-machine packaging validation.
2. Install Windows Graphics Tools for a local SDK-layer pass and validate the completed DRED
   report path against an intentional device-removal diagnostic on disposable test hardware.
3. Resize-in-flight/per-monitor-DPI/display-change/sleep-resume and controller hot-plug stress.
4. Additional NGG modes/title captures, FMASK/DCC, remaining primitive/texture semantics,
   persistent depth/MSAA resources, and asynchronous GPU batching.
5. Long-running A/V, memory remap, save-data, clean-shutdown, and multi-title interactive tests.

The detailed graphics sequence lives in [gpu-d3d12.md](gpu-d3d12.md).
