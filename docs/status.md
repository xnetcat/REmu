# Subsystem status

This records what each subsystem actually implements and its next concrete gap. A splash screen,
a timeout, or the absence of `unimplemented` log lines is never treated as a compatibility claim;
completion means stable interactive game progress with correct output and measured performance.

## Silent Hill present (2026-08-06) — past splash + hallway; title 3D black

**Compositor tiler:** mode-1/5/9 uncompressed + BC element grids (splash UI assessable).

**Mesh LDS (`f28cee1`):** NGG mesh LDS from SDK `ShShaderResource2Gs::getLdsSize`.

**Mode-5 BC mips (`8e4ed20` on `main`):** GPU tiler owns AddrLib mode-5 compressed mip chains.
Live compressed-mip refuse count **0**. Mode-1/9 BC mips and mode-27 BC stay loud-refused.

**Latched stencil (`0a22e15`):** `STENCIL_ENABLE` with Z ops off no longer aborts color MRTs
(depth decode `Ok(None)`; true stencil-only still refused). Live stencil-only refuses **0**.

**Sample-path (`3c025bd`):** TYPE-0 refuse + BGRA↔RGBA authority; `ALIAS-VIEW` binds on title
intermediate.

**Attribution (`a86c225` clean):** `800e73` intermediate Metal texels are **RGB black**
(`rgb_nonzero=0`); HDR `RG11B10` input black too. Failure is upstream of `800da3`/alias, not a
composite discard.

**`VTX_W0_FMT` (`e75d19c`):** mesh position W=0 was clipping the HDR cover; live now shows
full-screen `pixel_outputs` on `800bf4`. **Still** `nonzero=0` / PS `±0.0` — coverage ≠ content.

**Zero PS exports (proven):** AbgrFp16→RG11B10 is not the discard path. Tonemap tex0 is
float-`±0` (byte-nz from sign bits only); PS exports match.

**`800c7e` ±0 (`32b445d` clean `runs/silent-hill-title-bg-cs800c7e/`):** CS `ImageLoad`s
full-res zero `803d6e` (not SRD-missing). Upstream `800a69` full coverage + zero exports;
GBuffer sample `8023b9` Metal+guest all-zero.

**No BasePass emit (`736d540` clean `runs/silent-hill-title-bg-gbuffer/`):**
`total_mask_mrt3plus=0`; multi-MRT is only 2-slot shadows; GBuffer COLOR writers absent;
depth `801ac2` `gpufalse/v0`. Not Metal-dropped BasePass / not Malformed CB.

**Live with `REMU_CAPTURE_FRAMES=1`:** splash → hallway 3D → title UI; title BG still void.

**Remaining title-3D blockers:** why Unreal never emits deferred BasePass for
`8023b9`/friends → empty lighting → ±0 scene color; null IMAGE-FALLBACK; no-attachment if
on path; mode-1/9 compressed mips; NGG `0x7` if critical. See
`docs/notes/sh-present-20260806-handoff/`. Anti-claim: hallway/title UI / mode-5 / stencil /
type-0+alias / black-intermediate / VTE coverage-without-content / byte-nz-of-±0 /
“CS only binds 1×1 dummies” / “Metal dropped BasePass” ≠ title-background 3D closure.

## Phase 1F (2026-08-06) — execution substrate — CLOSED (on `main` @ `d7414fc`)

Guest exception delivery with Orbis `ucontext_t` (mcontext at `+0x40`); interruptive
`sceKernelRaiseException` via host SIGUSR1/`pthread_kill`; unrecovered SIGSEGV/SIGBUS offered to
installed guest handlers; game default affinity `0x7f` (SYSTEM `0x1fff` still placeable);
`usleep(0)` yields; nanosleep/sleep/sched_yield wired; host QoS from Setprio/setschedparam (or loud
refusal); `_sigprocmask` writes oldset. See `docs/notes/phase1f-20260806-handoff/`.

**Acceptance:** architectural data flow + unit tests + both-arch `cargo check` green. Silent Hill /
Oregon / Poppy watchdog-timeout clean; `pthread_setschedparam` no longer fabricated success. Unity
GC STW **not** live-proven — Dreaming Sarah still null-SIGSEGV before `InstallExceptionHandler`.
Demon's Souls still ACM-blocked.

## Phase 1H (2026-08-06) — GPU sync completeness — CLOSED (on `main` @ `6f503df`)

SDK `sync.cpp` RELEASE_MEM / WAIT_REG_MEM / CB·DB writeback / `gpuSyncDma`; W712 cross-queue
fence DMA on the GPU timeline (`CrossQueue` 1ms host reject deleted); consumer-aware ACQUIRE
demotion. See `docs/notes/phase1h-20260806-handoff/`.

**Acceptance:** package tests + both-arch `cargo check` green; live matrix at `f8bdad5`
(SH/Oregon/Poppy watchdog-timeout; DS/Void null-SIGSEGV; DeS ACM). **Anti-claim:**
`captured_frames=0`; SH DMA-FILL rate meter `attempts=0` — no fabricated Q W712 counter win;
no FPS (SH launch loadavg 5.17). Phase 1 A–H invariants complete after merge.

## Phase 1G (2026-08-06) — memory model — CLOSED (on `main` @ `b4a82f5`)

Guest-backed GPU-tiler upload/publish for tiler-owned textures; anonymous→committed backing;
CPU sampled snapshot/`replaceRegion` path deleted on that surface; fingerprints and dead tiler
seams deleted. Silent Hill 1G-introduced SIGSEGV (`addr=0x8000000080b0`) fixed: alias/sampled
publish refused by default; bookkeeping keeps Metal debt. See
`docs/notes/phase1g-20260806-handoff/`.

**Acceptance:** package tests + both-arch `cargo check` green; SH/Oregon/Poppy watchdog-timeout
clean; SH fault closed vs main control. **Anti-claim:** `captured_frames=0` on tip and main —
PPM UI-garbage unassessable; not a 3D/UI closure. Dreaming Sarah/Void null-SIGSEGV and DeS ACM
unchanged.

## Phase 1E (2026-08-06) — shader core — CLOSED (on `main` @ `5bb169b`)

Exact image compare lattice; VALU sign/saturate/SCC; precise `V_DIV_SCALE/FMAS_F32`; sub-dword
LDS; graphics scratch + GS wave width; no raw VOP3/DS/MUBUF/MIMG operand fabrication; `0x351` =
`V_MIN3_F16`; graphics-Dispatcher single-lane ballots; `S_BARRIER` allowed on structured CF and
refused under Dispatcher; compacted-vertex EXP staging under proven `NggRegisters`;
**`INTERFACE-FILL` deleted**. Follow-up (`5853fa7`): reconcile `SPI_PS_INPUT_CNTL` with EXP PARAM
producers — unmatched slots without general LDS take SDK `UseDefault0` (not RemuVsOut zero-fill /
fabricated UVs); general LDS + missing EXP stays loud-refuse. See
`docs/notes/phase1e-20260806-handoff/`.

**Acceptance:** package tests + both-arch workspace check green (2 pre-existing irreducible CF
unit fails unchanged from `0a8212a`). Silent Hill at `5853fa7`: missing-PARAM Gen5 pair fails
**31→0**; `INTERFACE-FILL=0` — **still not** a 3D-scene closure (“success is not the hallway”;
follow-up run `captured_frames=0` under overloaded `vm.loadavg`). Dreaming Sarah / Void Terrarium
host null SIGSEGV also reproduces on Phase 1D tip `0a8212a` (not a 1E introduction); `bf50225`
DS control same hour was ~30 FPS (not 60) so no FPS claims.

## Phase 1D (2026-08-05) — register model — CLOSED

AGC shadow Cx/Sh/Uc files seed from `agc_register_defaults` on init and selective `resetQueue`
(`ResetQueueOp` 12-bit mask; ops `0` = `kNone`). Decoder fallbacks that invented CW winding,
identity `SPI_PS_INPUT_CNTL`, full-target viewports, or non-zero `SPI_SHADER_COL_FORMAT`
writemasks are deleted. Stencil, depth bias, `CB_COLOR_INFO` blend bypass, `UcIndexOffset`
(`0x24a`), `CxVgtShaderStagesEn`, and `UcGeCntl` NGG sizing are decoded. Viewport-Y keeps the
`6a082b1` absolute-height / no winding-flip contract. See
`docs/notes/phase1d-20260805-handoff/`.

**Acceptance (`1273aab`):** Silent Hill watchdog-times out clean with seeded defaults and CCW NGG
mesh winding in logs. Not a 3D-scene closure. Phase 1E closed separately above.

## Phase 1C (2026-08-05) — ordered publication — CLOSED

Aligned 32/64-bit guest labels and HLE sync words publish as single release stores (not tearable
memcpy). Payload→label batches take an explicit release fence before the label. Word-sized
baseline-guarded writebacks use compare-exchange. Shared accessors live in `remu-memory`;
`docs/review-unsafe.md` requires an ordering clause. See
`docs/notes/phase1c-20260805-handoff/`.

**Acceptance (`1aa4d5b`):** Silent Hill watchdog-times out clean; Demon's Souls still reaches ACM
(`sceAcmBatchStartBuffers` unimplemented) after the Scream mutex path — no new publication blocker.

## Phase 1B (2026-08-05) — TLS / `fs:` data flow — CLOSED

Guest `fs:` accesses resolve against the guest thread pointer under Rosetta: load helpers preserve
red-zone + vector state (helper-local skip + plain `ret`; store sites keep `lea`+`ret 128`);
prefix-aware seeded decode covers displacements / widths / stores / non-mov; SIGSEGV backstop
emulates or loudly refuses misses before null-address death. Byte-scan FS substitute and
boundary-mismatch rewrite gate deleted. See `docs/notes/phase1b-20260805-handoff/`.

**Acceptance (`4e0357f`):** Silent Hill links and watchdog-times out with
`decode-summary aot_loads=9 aot_stores=164`. Demon's Souls reaches ACM again
(`sceAcmBatchStartBuffers` still unimplemented). Dreaming Sarah / Void Terrarium still null-fault
early after TLS links — not claimed as TLS-closed.

## Phase 1A (2026-08-05) — sync-object identity — CLOSED

Mutex/condvar/rwlock identity follows the SDK pointer typedefs (`sys/_pthreadtypes.h`): init writes
a real control-block handle into the guest slot; lock/unlock/wait resolve by that handle (including
stack copies of the same handle). Init always overwrites the slot (Scream stack CreateMutex reuse);
destroy blanks publishing storage only when it still holds this object's handle. Kernel
`SceKernelSema` create writes a full pointer-sized slot. Create-on-first-use mutex fabrication and
its Phase 0 counter are deleted. Static `NULL` / adaptive-mutex `1` initializers still materialize.
See `docs/notes/phase1a-20260805-handoff/`.

**Acceptance:** Demon's Souls Scream mutex path passes on `74aa06d` (no `snd_osfuncs` assert, no
rejected unlock; AudioOut2 + `ncaPumpThread` proceed). Silent Hill still has zero lazily fabricated
mutexes. **Next DeS blocker (not 1A/1B):** `sceAcmBatchStartBuffers` unimplemented, then
SIGSEGV at `addr=0x303030303030303`.

## Phase 0 (2026-08-05) — instrumentation landed

Hygiene/instrumentation only — see `docs/notes/phase0-20260805-handoff/`. Fabricated-success
counters print in the ABI gap summary; SIGSEGV/SIGBUS reporter + alt stacks are unconditional;
`/devlog/app` is writable; evidence runs refuse dirty trees and record `vm.loadavg` / display mode
in `run.json`. Shader opcode frequency backlog (before any shader-core behavior change):
`docs/notes/phase0-20260805-handoff/SHADER-REFUSAL-BACKLOG.md`. The Phase 0 lazily-fabricated-mutex
counter was removed by Phase 1A once identity-by-handle made that safeguard obsolete.

**Live refutation:** `docs/status.md` previously said Void Terrarium sustains AGC submissions for a
full diagnostic timeout. On revision `663bc58` (2026-08-05) it SIGSEGV'd at ~1s (`addr=0x0`). Treat
sustained-run claims for that title as stale until re-proven at one revision with n≥1 clean evidence
run (crash claims still need n≥5 before calling a crash "stable").

Current overall state: the first small test title boots through resource loading, submits AGC
draw streams, compiles observed RDNA2 shaders to Metal, and flips its display buffers. Early
splash frames render correctly; full scene rendering and playability are still open. A second,
Unity/IL2CPP title (which commits nearly all of its memory by reserving virtual ranges and
`sceKernelMprotect`-committing them rather than mapping direct memory) now gets resident,
write-watched textures; its environment backdrop, lighting, and menu UI now render with the
correct orientation. Lighting and post-processing are improved but not yet visually correct.

A third Unity title (Void Terrarium) previously aborted in ~2s with a host stack overflow on
`GraSubmitThread` during the first AGC submit. Guest pthread stacks of 16 KiB are now floored to
a 2 MiB host stack without changing guest-visible attributes. The title now sustains AGC
submissions and VideoOut flips for a full diagnostic timeout. Save-data and message dialogs are
presented as non-blocking AppKit `NSPanel` overlays that stay `RUNNING` until the user clicks a
button or presses Cross/Enter (accept) or Circle/Escape (cancel); trophy unlocks and notifications
publish floating toast panels that auto-dismiss. Remaining visual gap: many draws are still dropped
after pipeline compile failure, so captured frame dumps are still zero even while flips proceed.
Indexed NGG triangle strips (`primitive=0x6`) and point lists (`primitive=0x1`) are now
expanded into the meshlet triangle ABI, and wave32 `SAndSaveexecB32` lowers for mesh shaders.
Void Terrarium’s content quads and amplified point draws compile and encode. The post/composite
pixel path that needed `ImageGetResinfo` now lowers to Metal (`get_width`/`get_height`/
`get_num_mip_levels`, with optional `unorm` float bitcasts), so that previously failing
`es:0x70e5db0400`/`ps:0x70e5da8700` pair compiles. NGG mesh lowering now uses wave32 lane IDs and packs `s3[27:24]` (wave index) plus `s3[31:28]` (wave count) so amplified
point-list particle subgroups (e.g. 96 threads) keep unique LDS/export IDs and a working
per-wave export reduce; unexported mesh verts default to `w=0`. Wave32 `V_CMP` lane masks
no longer zero the adjacent SGPR, which was wiping those packed `s3` fields before the
export reduce. Host present now keeps `RGB10A2` VideoOut on a `BGRA8Unorm` drawable (no
second sRGB encode); float scanouts still use `BGRA8Unorm_sRGB`. Guest common-dialog /
toast UI is a present-path PS5-like overlay (no AppKit panels); soft cursor/glow and
depth residency (`depth_native`) remain open.

## Import coverage

A PS5 import identifies its function only by an 11-character NID, so measuring coverage means
first recovering names. `scripts/generate-nid-names.py` hashes every symbol exported by a
directory of PS5 modules and keeps the matches; `scripts/audit-import-coverage.py` then reports,
per title, how each imported NID is served.

For the larger of the two test titles, of 2028 callable imports: 1140 resolve inside modules the
game itself ships, 436 are implemented here, and 443 remain unserved. All but 12 have a recovered
name, which is what makes the remaining gap a work list rather than a list of hashes. The unserved
majority is concentrated in the graphics and asynchronous-file command-buffer builders.

Naming a NID is not implementing it. The audit deliberately separates the two, and a handler that
claimed a NID while returning a fabricated success would be worse than no handler at all: it turns
a loud gap into a silent one. Unimplemented imports are expected to stay visible in the log.

## Open defects

These are known failures with enough recorded state to be worked on. A defect is listed here
rather than worked around; a title-specific patch that hides one is not an acceptable substitute.

### Guest object storage goes stale mid-run

A guest object's storage is observed zeroed while the guest still holds a live reference to it.
Five distinct consumer sites were recorded faulting on the result, all in one Unreal-based title:

- A graphics pipeline state object is stored and referenced, then its mandatory `+0x18` child
  reads back as null before submission. The producing constructor guarantees that child is
  non-null, so this is not an object that merely needs more construction time.
- A queued render task reads back with a zero vtable pointer.
- A container slot that the guest's own iteration treats as populated reads back null.
- A per-field handler pointer reads back null at a virtual call site.
- A parent object is destroyed by a nested update, after which the caller reloads its vtable
  pointer and finds it zeroed.

All five share one shape: memory the guest wrote and still owns does not read back with the value
it wrote. That points at a REmu defect in memory lifetime, aliasing, write visibility, or an HLE
call returning a value that makes the guest free storage early. It is not a guest defect, and the
same five sites are the evidence for whichever of those turns out to be the cause.

`scripts/lldb_allocator_map_watch.py` and `scripts/lldb_allocator_pop_watch.py` observe the
allocator during this failure and are the current starting point.

This was previously masked by a compatibility guard that patched the five consumer sites in the
guest executable so they tested for the bad value and skipped their work. The guard was removed
because it was locked to hardcoded addresses and byte signatures from one build of one title, and
because suppressing the symptom removed the pressure to find the cause.

After commit-on-protect landed (2026-07-28), the startup crash no longer reproduces: two
consecutive 45-second runs render real scenes (22–26 FPS, ~40 resident textures, heavy
GPU-to-guest writeback traffic) and end only on the diagnostic timer, with no faults logged.
That is consistent with this defect living in the same layer the fix repaired — guest objects
in reserve-then-`mprotect`-committed memory were invisible to every shared-backing path, so
GPU-to-guest writebacks into such storage could not land — but it is not proof. The five
recorded consumer sites remain the checklist: this entry closes when sustained interactive
progress shows none of them faulting, not because the crash stopped appearing in short runs.

### The Bluepoint title idles at startup without its fiber ABI

A third title (Bluepoint's BPE engine) previously aborted about three seconds into startup, inside
its own `PS5PlatformPageAllocator`. Three unserved imports caused it, each one a call whose result
the engine consumes directly:

- `sceKernelAvailableDirectMemorySize` (`C0f7TJcbfac`) left both output parameters untouched, so
  the engine read an available size of zero, computed `0 - trackingOverhead` as its pre-allocation
  size, and failed its own allocation verify at `PS5PlatformPageAllocator.cpp:383`.
- `sceKernelAvailableFlexibleMemorySize` (`aNz11fnnzi4`) failed the same way at line 450.
- `scePthreadRwlockRdlock`/`Wrlock` (`Ox9i0c7L5w0`, `mqdNorrB+gI`) were the only members of the
  console-named reader/writer lock family not routed to the implementation the POSIX-named
  spellings already used, so Dantelion2's `DLReadWriteLock` took a not-implemented return as a
  fatal error rather than as contention.

With those served, the title reaches its own subsystem initialization, opens the presenter, and
runs to the diagnostic timer without faulting. Its job system then stalled because it is built on
the fiber ABI, which was unserved; `sceFiberRun` returning not-implemented meant the job driving
startup never ran and the title blocked in `sceKernelWaitEventFlag` for the rest of the run.

The fiber entry points are now implemented (`fiber.rs`), and a live run exercises the whole
lifecycle without error: `sceFiberOptParamInitialize`, `_sceFiberInitializeImpl`, `sceFiberGetInfo`,
`sceFiberRename`, `sceFiberRun`, `sceFiberReturnToThread`, and later `sceFiberSwitch`. The job
system now runs and reaches asynchronous asset loading, resolving and reading
`/app0/playgo-chunkdefs.xml` through the AMPR command-buffer path.

Two further defects were then found and fixed, both by tracing rather than inspection:

- `sceAmprCommandBufferWriteAddressOnCompletion` was unserved. It is the polled half of the APR
  completion contract — a guest that registers no event queue watches a completion word instead —
  so the buffer was submitted and its transfer performed while the word the guest spun on was
  never written. It is now recorded like the queue-signal command and published on submission,
  before any event is announced, so a guest woken by the queue already observes the word.
- The fiber implementation resolved its per-thread current-fiber slot in a way the compiler could
  hoist across a context switch. That is sound for ordinary code, where a function cannot change
  threads midway, but a fiber can: this title parks a fiber on one worker and lets another resume
  it. The resumed frame then read the *suspending* thread's slot, which by then was cleared, and
  reported `SCE_FIBER_ERROR_PERMISSION` back into the job system. Forcing the resolution to happen
  after the resume fixed it, and a test that resumes a fiber on a second thread covers it.

With those, fiber traffic runs clean — over two thousand transitions in a 45-second run with no
refused call — and the title reaches its Bink video threads and its `Core.Res` decompressor,
file-loader, and task-manager pipeline.

The title now runs to roughly ninety thousand imports. It configures its display — 1920x1080
through `sceVideoOutSetBufferAttribute2` and `sceVideoOutRegisterBuffers2` — and brings up audio.

`sceAcmContextCreate`/`Destroy` are served: an ACM context is an opaque handle the guest hands back
to the batch entry points, so issuing a distinct live id is that call's whole contract. The
signal-processing batch entry points stay unimplemented. This removed a spurious failure but was
not a blocker: the engine creates a context, immediately destroys it, unloads the module, and
finalizes Ajm either way, which reads as a capability probe rather than a failed path.

One APR defect was a REmu performance bug rather than a missing ABI. The successful branch of
path resolution logged unconditionally while every other trace in that module is gated. A title
that indexes its whole asset database resolves hundreds of thousands of paths at startup, so the
log dominated the work: one run wrote 80 MiB across 252,127 lines, of which 251,741 were that
single call. Gating it left 196 KiB. Failed resolutions stay unconditional, since they are rare and
are what a bring-up needs. With it gated the title reports its own timings and finishes indexing:
`RecordResourceDependencies() took 4.7787s`, `GatherResourceFileInfo() took 28.2511s` over 222,670
distinct files, including its startup script.

It is still not playable and no compatibility is claimed. After indexing it reaches a steady state
and stays there for the rest of a five-minute run: all thirteen `BPE JobWorkerThread`s spin in a
tight guest loop around `0x80086ee30`, and every other guest thread — the four `BinkAsy` threads,
the three `Core.Res` threads, `NexusRevolution Event` — is parked on a condition or semaphore.
Nothing produces work. No draw is submitted and no flip requested, though the display is fully
configured.

Every import this title reaches is now served: a run reports zero unimplemented NIDs. Nine were
added, each one unblocking the next: `sceKernelSetPrtAperture` (which records an aperture over the
partially-resident-texture window and is real bookkeeping, not a stub),
`sceUserServiceGetPlatformPrivacySetting`, `sceVideoOutColorSettingsSetGamma_` (whose value arrives
in the first floating-point register), `sceSystemGestureOpen`,
`sceSystemGestureInitializePrimitiveTouchRecognizer`, `sceSystemGestureCreateTouchRecognizer`,
`sceNetCtlInit`, `sceNetCtlGetState`, and `sceNpRegisterStateCallbackA`. The last five model
hardware this host does not have; each reports the state that is actually true here — no touch
occurs, no network link exists — rather than a placeholder, and that is recorded on each call.

The title still submits no draws. Differencing against SharpEmu, which reaches menus and gameplay
on this title, located the divergence. SharpEmu was built from its `main` at `5f1bd5a` and run on
the same dump with `--trace-imports`.

The gross measure first: SharpEmu issues 17.5 million guest import calls in 150 seconds where REmu
issues about 110 thousand in 120. REmu's guest is barely executing by comparison, so the plateau is
not slowness — the engine is not doing the work at all.

The concrete divergence is the audio DSP path. In SharpEmu the title creates one ACM context and
then runs it hard: 7,424 `sceAcmBatchStartBuffers`, 7,424 `sceAcmBatchWait`, and 2,472
`sceAcm_ConvReverb_SharedInput`. In REmu it creates a context, destroys it, unloads the module,
and finalizes Ajm, having issued zero batch calls. Every one of those batch and DSP entry points is
unserved here.

This corrects an earlier reading recorded in this file: the create-then-destroy sequence was
described as a capability probe. It is not. It is a failure path.

`REMU_TRACE_IMPORT_ERRORS=1` found the value the engine was reacting to. Six imports failed in a
whole run, and one was `sceAudioOut2PortCreate` returning `PortFull`. AudioOut2 ports were being
opened through the `sceAudioOutOpen` path and charged against that API's eight-port budget — which
is correct for AudioOut1, whose ports are hardware outputs, and wrong for AudioOut2, which is a
software mixer whose ports are voices. The SharpEmu trace shows this title holding more than forty
at once. The ninth was refused, its audio engine could not build, and it unwound the whole stack.

AudioOut2 now opens against a separate voice budget. The teardown is gone: a run no longer calls
`sceAjmFinalize` at all, and the title proceeds to submit audio-job batches. `sceAjmBatchJobGetStatistics`
is served — the append into the caller's batch arena is real, while the counters it reports are
zero, because no decode work runs on a coprocessor here and invented figures would be worse than
none.

The Ajm batch lifecycle (`sceAjmBatchStart`, `Wait`, `Cancel`) and `sceAjmBatchJobGetStatistics`
are now routed, and a run reports zero unimplemented NIDs. This did not help, and the way it
failed is the useful part.

No decode runs behind those calls: this host has no audio-job coprocessor and no general Ajm
decoder, so a batch is accepted and retired without producing samples. The title's audio engine is
Sony's SCREAM, and SCREAM checks its pipeline. It previously asserted in
`wws_scream/scream/src/ps4/snd_osfuncs.cpp:397` (later `:410`) on a broken mutex path; that
Scream mutex failure is closed as of Phase 1A acceptance on `74aa06d`. The active Demon's Souls
audio gap after that path is unimplemented `sceAcmBatchStartBuffers`.

That is this file's own warning about fabricated success arriving in practice. Accepting a batch
and reporting completion is a claim that audio was decoded, and an engine that verifies its own
pipeline catches the claim. Serving the batch ABI as a lifetime shell is therefore not the route:
Ajm needs a real decoder — the title's audio is ATRAC9 and AAC — or the batch entry points should
go back to being loudly unimplemented so the failure stays a stall rather than an abort.

That decision was made and implemented (2026-08-03): the shell is deleted and an `AjmEngine`
serves the batch ABI from parsed job structures against the SDK's layout, with a worker pool,
blocking waits, real statistics, and one real decoder (LPCM passthrough) proving the decode
shape end to end. A batch referencing a codec without a host decoder fails whole, loudly, with
the SDK's malformed-batch error and the failing codec type in the caller's error block —
SCREAM stalling on a truthful failure is acceptable; SCREAM being lied to was not. Per-codec
first-hit telemetry ranks which decoder to implement next; ATRAC9 and AAC (plan P1/P2) remain
the known blockers for this title's audio.

Import failures are now reported unconditionally rather than behind `REMU_TRACE_IMPORT_ERRORS=1`.
A failed import is the most useful line in a bring-up log, and gating it meant the runs that needed
it were the ones without it. Volume is bounded per call site — first failure, then powers of two —
so a call that legitimately fails often cannot bury the rest; the environment variable now removes
the bound instead of enabling the report. A whole run of this title reports four failures, none of
which is its blocker.

The title still submits no draws.

The auditor under-report is fixed (2026-08-03): `scripts/audit-import-coverage.py` walks every
dispatch chain (including `dispatch_stateless_import`, which serves the fiber family and platform
services), resolves module directories, and no longer truncates a module scan at a bare
`#[cfg(test)]` helper. `generate-nid-names.py` additionally reads the SDK stubs' `.scenid`
sections, the only oracle for pinned NIDs. A live audit of Poppy Playtime now reports 1750
callable imports with 203 unserved — a true work list, not an upper bound.

The fiber implementation is x86-64 only, because it saves and restores x86-64 register state. On
any other host the NIDs fall through to the unimplemented path rather than to a stub, so its tests
run under `--target x86_64-apple-darwin`. The attach-and-run/switch variants and the
context-size-check pair stay unserved until a title needs them.

### A run intermittently renders nothing

Roughly every other launch of the working test title produces `FPS 0.0`, `Draw 0/s` and
`Tex 1` for the whole run while audio continues at its normal rate, then exits on the
diagnostic timer with no error. The next launch of the same binary renders normally.

This is not caused by any recent change: it reproduces at `e57a6f1`, before the per-game
fixes were removed and before any file was split. Audio running normally while no draw is
ever submitted points at display or scanout setup losing a race during startup, not at the
guest failing.

Frame rate is also not stable across successful runs of one binary: 59.9, 61.9, 87.9,
91.9, 106.0 and 119.9 FPS have all been observed at the same revision, depending on window
and display state. Neither frame rate nor draw rate can therefore be used to detect a
performance regression. There is no CPU-throughput benchmark in the tree, which is the gap
to close before any performance claim can be made.

The same class has a half-rate sibling (2026-08-03, 19-run census): a launch sticks for its
whole run at either 60.0 FPS/899 draws/s or ~30 FPS/450 draws/s with present latency ~30 ms,
decided at startup. It reproduces at `bf50225` before any of that day's changes, and late in
the session the machine entered a state where every revision — including that baseline —
ran at 30 (AC power, no thermal pressure, no Low Power Mode, load under 5). In the slow mode
the guest paces on equeue flip completions at exactly half rate from the first window and all
counters halve together, pointing at the flip-completion/presentation cadence rather than any
compute path. Two candidate cures were tried and refuted while the machine state drifted
(once-per-vblank present pacing from a CVDisplayLink; completing translated flips at snapshot
retirement); both are parked unverified until a run on a display state that lets the bf50225
control reach 60 again. Plan E1 owns this class.

### Write-watch arming assumes plain read/write data pages

`CpuWriteWatch` arming and disarming reconstruct protection as read-only and read/write
respectively (`write_watch/page.rs`), never consulting the mapping's real protection. Watching a
read-only range therefore grants writes after the first fault instead of restoring read-only,
and arming a watch whose page-aligned span touches executable pages strips execute permission
permanently. This predates the committed-memory work — read-only direct-memory views were
already registrable — but commit-on-protect widens the reachable surface to any committed
protection. Neither regression title exercises it: every range the texture cache watches today
is plain read/write. The fix shape is to carry the mapping's original protection into
arm/disarm (write-watch arming becomes "remove write, keep the rest"), or to refuse watch
registration for non-read/write ranges explicitly.

### The Unity title's menu runs at 14–22 FPS

With FMOD's AudioOut2 port accepted, the Unity title reaches its interactive menu. Three
successive bottlenecks were profiled on its ~600-draw frames; two are fixed and one remains:

- Fixed: `scan_valid_command_stream_end` proved readability of every candidate command prefix by
  allocating and copying the whole span (up to 4 MiB per candidate per submission). It now uses
  the non-copying mapped-range check; this alone took the title from 4 to ~14 FPS.
- Fixed: per-face shadow-depth writebacks (see the Metal resources row) — `alias_writeback`
  dropped from 36/s (72 MiB/s) to zero.
- Remaining: a mach-exception storm keeps the Rosetta exception server saturated. Disabling CPU
  write watches lifts the title to ~20 FPS, so watch faults are roughly 40% of the remaining
  frame time — Unity co-locates hot mutable data on the same 16 KiB pages as watched texture
  bytes, so unrelated guest stores fault continuously. The fix direction is fault-rate reduction:
  registry/watch coalescing, and demoting chronically dirty textures from fault-based watching to
  acquire-time re-validation. A further CPU-side layer remains beneath that (~20 FPS with watches
  off, GPU under 30%).

The scene-orientation and white-overlay defects are fixed:

- GFX10 `SW_4KB_S` textures now use AddrLib-compatible packed mip chains: the mip-tail occupies
  the first 4 KiB, non-tail levels follow in reverse level order, and array/cube slices use the
  full chain stride. The previous linear mip offset treated tail bytes as mip zero, producing
  mirrored/upside-down scene regions and duplicate light fixtures.
- `CB_COLOR_CONTROL` special modes are decoded. Resolve, eliminate-fast-clear, and FMASK
  decompression packets end the current Metal render pass without rasterizing their attached
  dummy shader. Rasterizing Superliminal's resolve trigger as an ordinary constant-white draw
  caused the washed-out frame.
- GCN cube-resource face-atlas coordinates are lowered to Metal 2D-array sampling rather than
  being passed to direction-based cube samplers.

A live run without diagnostic bypasses now presents the warehouse with the correct orientation,
one physical light fixture, and readable menu text. Lighting, shadows, and post-processing still
do not fully match the intended scene. Performance also remains defective, particularly CPU
write-watch fault volume.

### The Unity title's intro video remains paused after its first frame

The retro-TV intro (60 fps 1440x1080 H.264 inside `sharedassets2.resource`) decodes through the
ffmpeg pipe backend. AvPlayer now follows shadPS4's source-backed pause contract: Pause preserves
an active session but freezes `CurrentTime`, and both data exports return no frame until Resume.
The earlier bounded paused-preroll experiment was removed after live tracing showed that it
advanced stream positions while Unity was paused and prevented the expected state transition.
Start and Resume now use separate state transitions; JumpToTime restarts the decoders with an
input-side `-ss` seek, and `sceAvPlayerChangeStream`/`sceAvPlayerSetTrickSpeed` cover the observed
single-representation and forward-rate cases. `REMU_TRACE_AVPLAYER=1` counts every zero-return by
reason. A live automated pass now reaches Ready, Start, and the first decoded video frame, after
which Unity calls Pause and issues no later Resume in that input path. No intro frame is presented
to the user, so the event/state transition between preparation, pause, and visible playback remains
an open defect.

### Gameplay after NEW GAME remains black

The loading screen completes, but the following in-game view can remain black while UI overlays
still draw. Two host-side causes were confirmed on the default `run-game.sh` path (bundled offline
`REMU_METAL_COMPILER`):

1. **Poisoned persistent metallib cache** — failed or raced offline links left ~92-byte header-only
   `.metallib` files in `~/Library/Caches/REMu/metal/`. Cache hits loaded them successfully but
   exposed no `remu_main`, permanently failing every graphics/compute pair (`failed=25+`,
   `targets=2`, ~28k dropped draws). Fixed by rejecting sub-256-byte cache entries, refusing to
   persist empty libraries, and falling back to in-process Metal compile when an offline library
   lacks the entry point.
2. **Transient offline spawn failures** — `ENOENT` from concurrent `metal` worker spawns were
   recorded as terminal pipeline failures. Fixed by deferring transient tool errors instead of
   inserting into `failed_shader_pairs`.

After both fixes, a 120s foreground verify with the bundled offline compiler reaches
`failed=0/0`, ~30 compiled Gen5 shader pairs, and ~44 color targets (~636 MiB resident) with no
`PERF-DROPPED` lines. A separate in-process-only verify matches verify2-style pipeline health.
Visual confirmation past NEW GAME/character select without pad injection remains open; if the
display is still black with healthy pipelines, the next suspect is post-process composite sampling
(scanout vs resident color RT timing; NGG attr path is `UseDefault0` / loud LDS refuse after
Phase 1E — not INTERFACE-FILL).

Scanout presentation re-snapshots at present only when resident `content_version`
advances past the last private copy (not on every in-flight batch overlap), skips writing
fully black PPM evidence frames until real pixels land, and avoids blocking the AppKit tick
on snapshot blit completion. Consecutive compute dispatches within one guest batch now share
a single Metal compute encoder instead of opening and closing one per dispatch; active pass
encoders are closed before scanout blit (fixes host SIGSEGV at present). A 300s isolated
windowed verify16 at `dfc1f40` completes on the in-process watchdog (`outcome=timeout`,
302s elapsed, 127 captured PPMs, 0 fully-black frames, no host SIGSEGV). Steady-state Draw
~320–360/s and RP-end key ~189–199/s remain ~4–15% below verify11 (~374/s); verify14/15
ended at 143s/154s from orchestration SIGTERM/SIGKILL, not guest faults — `run-game.sh`
now labels exit 137/143 as `host-sigkill`/`host-sigterm`. Side-by-side steady-state
PERF-CAUSES at Pipe 75/0 shows RP-end key ~189/s in both verify11 and verify16 (not the
regression lever); verify16 instead carries ~26–29 sampled `acquire_invalidate`/s where
verify11 carries 0, plus slightly lower RP encode (~274 vs ~285/s). `ae9b141` keeps cached
sampled textures across permission-only ACQUIRE when a GPU-written resident still serves
the SRD (overlap, pitch-mismatched sample storage, depth/color view eligibility). A 180s
windowed verify17 at `ae9b141` completes cleanly (63 PPMs, 0 black, `failed=0/0`); verify18
(300s) measures whether Draw recovers toward verify11 without reintroducing black frames or
Q W 457 queue depth changes. RP-end target-address churn (~180 addr/s) remains inherent to
UE's render-target rotation and cannot be collapsed without incorrect Metal attachment reuse.
Soft cursor/glow and post-composite sampling past title remain open graphics gaps.

### Host writebacks corrupt recycled guest heap, and the corruption widens with queue depth

The UE title's `MallocBinned3 Corruption Canary was 0x40404040` fatal and its two secondary crash
faces (poison-pointer SIGSEGV, null-deref in the fatal handler) reproduced in four of six
120-second runs on 2026-08-03. The guest-write journal (`REMU_GUEST_WRITE_JOURNAL=1`) attributed
two crashes with write-level certainty: `site=global-writeback-identity` published scattered
uniform-0x40 runs — DCC fast-clear key bytes, the GPU's own output — into guest ranges the
allocator had recycled, immediately before the guest dereferenced the pattern.

Two windows exist, and only one is closable at publish time:

- **Guest wrote after capture, before publish.** Closed on 2026-08-03: every writable-global
  writeback run now publishes only while the destination bytes still equal the capture-time
  baseline, compared and written under the registry lock. A run the guest has since written is
  skipped whole and reported (`kept guest bytes over a writable global writeback run`). The
  glyph-motivated producer-identity publish still lands for untouched pages.
- **Guest recycles after publish.** Still open, and structural: the journal shows baseline-clean
  publishes whose target the allocator hands out moments later. A userspace `free()` touches no
  mapping, so no admission check can see it. The publishes are correctly ordered GPU effects
  arriving seconds late — the guest frees pooled render targets on a frame-lag heuristic that
  assumes GPU work retires within frames, and a 350–700-item queue backlog stretches one frame
  to seconds. The corruption rate is therefore a function of queue depth: the deeper the stall,
  the wider every free-versus-publish race. The structural fixes are the `GuestRangeLease`
  design (audit B21) or collapsing the backlog itself; a content-keyed or title-keyed gate is
  not acceptable.

The texture-writeback family has the same two windows. Its capture-to-publish window is still
guarded only by generations sampled after the multi-millisecond readback, and the admission
fingerprint check runs before it; the partial-chunk staleness of `record_content_snapshot`
makes a publish-time full-baseline check produce false positives, so closing that window needs
the chunk bookkeeping reworked first.

The load/movie phase this corruption fires in ran at 2–4 FPS with GPU 16–20% on 2026-08-03
(0.7–0.9 FPS at the 2026-08-02 audit). Its dominant cost is no longer diagnostics: `CB wait`
was 92–112/s of ~100 command buffers/s — nearly every Metal command buffer is committed by a
blocked same-queue `WAIT_REG_MEM`, executing the pipeline synchronously one wait at a time.
Dispatch-lock wait time was ~17% of wall in the same windows. A transient variant of the same
phase blocks on a cold pipeline compile (`active=3` for two windows) and resolves; the
permanent-looking "wedge" recorded at `Q W683` was this phase with the guest dying of the
corruption above before escaping it.

### The runtime test suite is order-dependent

`cargo test -p remu-runtime --lib` fails intermittently, at roughly a one-in-three rate, in
`agc_driver::tests::a_graphics_submission_parses_its_stream_and_requests_a_completion_event`. It
passes reliably with `--test-threads=1`.

`remu_gpu::install_event_sink` writes a process-global sink. Tests that assert on emitted GPU
events take `GPU_EVENT_SINK_TEST` to serialize, but 121 tests reach the import dispatcher without
holding it, so an event emitted by one test can be delivered to another test's receiver.

Until this is fixed, verification runs must pass `--test-threads=1`, which makes the suite
deterministic. The fix is either to make every test that can emit take the guard, or to add a
thread-local event sink to `remu-gpu` and route test events through it. `remu-gpu` has no
thread-local capture today, so the second option is new code, not a rewiring of what exists.

## Subsystem table

| Subsystem | Implemented | Next concrete gap |
| --- | --- | --- |
| CPU, loader, linker | x86-64 guest execution behind a native CPU backend under Rosetta; SELF/ELF mapping, observed relocations, AMD64 DTV slots, per-thread `__tls_get_addr`, TLS clone rebasing, and import trampolines with synthetic coverage. | Continue source-backed HLE import closure. The AArch64 translator stays deferred until a milestone requires it. |
| AGC / PM4 ordering | Native Gen5 command routing, type-2 packets, clear state, packet predication, targetless scanout association, compute submission, ordered memory copy/write/release/wait events, submission boundaries, and a bounded backpressured event channel with an ordered shader-compile gate. | Execution-side `SET_PREDICATION` semantics, then source-backed release/wait, cache visibility, fence writes, and graphics-equeue completion. |
| Metal resources / readback | Persistent guest textures backed by either direct-memory views or committed (reserve-then-commit / flexible) guest memory, direct private scanout targets, command-batch completion, shared staging-buffer readback, exact packed rows, AddrLib-compatible GFX10 `SW_4KB_S` packed mip/array/cube upload, MSAA resolve targets, supported depth readback, and GPU-side assembly of layered depth-sample views (shadow cubemaps/cascades rendered as per-face 2D depth targets and bound through one layered SRD are blit-assembled into a cached depth-format array inside the guest batch instead of round-tripping every face through guest memory each frame). Covered by GPU tests. | Bounded asynchronous capture tickets and a reusable staging pool; explicit block-compressed and combined depth/stencil layouts when guest formats exercise them. |
| Guest virtual/direct memory | Fixed/hinted mappings, physical allocation splits, coherent aliases, strict range unmap/protect, `MAP_NO_COALESCE` identity preservation, release-time alias invalidation, shared-mapping identity tracking, CPU write-watch retirement across remaps, and commit-on-protect reserve+commit semantics: `sceKernelMprotect` over a reserved range and flexible mappings commit zero-filled memory into a virtually-keyed shared backing, so GPU residency and write-watching resolve them exactly like direct-memory views, decommit zeroes pages before release, and guest-visible `VirtualQuery` categories are unchanged. The backing is keyed by virtual placement — a deliberate scope decision: this memory class never exposes a physical address to the guest, so physical aliasing of it is unrepresentable by construction, unlike direct memory, which keeps its own physically-keyed backing object. All with synthetic coverage. | Coalesce adjacent contiguous shared-registry records at registration: commit-heavy titles insert one record per `mprotect`, and resolution currently scans records linearly per watch creation and arms one watch segment per record. Mapping lifecycle counters and exact query error/flag behavior for all exercised cases. Memory-pool NIDs (`sceKernelMemoryPoolReserve`/`Commit`/`Expand`/`Move`/…) stay loudly unimplemented until designed: pools deal in guest-physical block addresses (`Expand` returns one, `Move` re-points committed blocks without copying), so they need physically-keyed backing and must not silently reuse the virtually-keyed commit path. |
| Render state / pipelines | GFX10 render-target decoding, MRT attachment layout, write masks, viewport/scissor, formats, depth-target validity requirements, `CB_COLOR_CONTROL` resolve/eliminate-fast-clear/FMASK-decompression command handling, seeded AGC register defaults (Phase 1D), stencil/depth-bias/`CB_COLOR_INFO` blend-bypass/`UcGeCntl` NGG decode, and translated shader pipelines with cached construction. | Combined depth/stencil surface formats; remaining multisampling behavior; full structural pipeline-cache identity. |
| Shader translation | Typed backend-neutral Gen5 IR, direct single-path MSL lowering, scalar/vector state, exports, sampled/storage images, buffers, zero-copy global bindings, wave64 ballot handling on SIMD32, and observed compute geometry, with offline Metal compilation tests. | Complete the remaining RDNA2 instruction/resource/control-flow surface and eliminate residual per-draw Metal buffer churn without weakening guest-memory visibility. |
| Presentation / frame capture | Guest display size independent of host drawable size; opt-in headless Metal; interactive runs as a regular app bundle; direct scanout without routine GPU-to-CPU round trips; asynchronous shader compilation with strict event ordering; on-disk Metal library and pipeline caches; live title-bar performance statistics. | Resolve early post-boot output into the first real scene and use completed-presentation and guest-flip rates to measure pacing. |
| Runtime / ABI | Whole-process import auditing across HLE, guest-libc, and host routes; FIFO-ticket condition variables with exact signal/broadcast cardinality; guest mutexes/condvars/rwlocks/semaphores keyed by SDK control-block handles (Phase 1A closed — Scream mutex path passes); guest `fs:` TLS via seeded decode + helpers + SIGSEGV backstop (Phase 1B closed); ordered guest label/HLE word publication with release stores + payload→label fence (Phase 1C closed); video-out attributes; save-data basics; NP service objects. | Implement `sceAcmBatchStartBuffers` / ACM DSP for Demon's Souls; cancellation, clock attributes, and absolute timed waits; continue source-backed AGC event queues, save dialogs, and remaining researched NIDs. |
| Audio | Host output, guest port/config validation, bounded queues, resampling, stereo downmixing, and the complete defined AudioOut2 format lattice (float/i16 × mono/stereo/7.1 × standard order), which is what FMOD titles open. Twelve-channel layouts stay explicitly rejected until the mixer models them. | Crackling from 15–17% audio underruns under heavy import-thread contention on command-heavy titles; audit timing, backpressure, device change, pause, teardown, and A/V synchronization with deterministic offline mixer evidence. |
| Video / Bink | Initial AvPlayer MP4/H.264 and audio decode through ffmpeg pipes, stream enumeration, play/pause/resume/seek/loop/trick-speed state, media-clock frame gating, NV12 buffers, and deferred event callbacks, covered by generated fixtures. | Present the Unity title's intro frames and resolve its preparation/pause transition; Bink has no compatibility claim yet. |
| Storage / save data | Read-only app filesystem access and initial save transactions; dumps are never modified. | Complete namespace, path, permission, iteration, atomicity, durability, quota, slot, dialog, and concurrency semantics with crash-consistency tests. |
| Input | Keyboard-backed pad state and opt-in deterministic input sequences. | Complete handle lifetime, disconnect, timestamps, dead zones, motion/touchpad, controller mapping, and focus semantics as titles exercise them. |
| Configuration / telemetry | Environment-controlled presentation, capture, Metal compiler, and diagnostics; bounded logarithmic log checkpoints; opt-in VM tracing; live window-title performance statistics; per-run summaries. | A versioned, documented configuration file with strict precedence and validation, plus structured persistence of the live counters. |
