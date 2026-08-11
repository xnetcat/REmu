# Complete ACM Engine Design

**Outcome (2026-08-11): implemented.** The engine landed through commits `dacdcd5e`, `79e68531`,
`1dcb828d`, `9e2b49bd`, `b57d3d51`, `edcb3ec3`, `7445ea96`, `b20c762a`, and `5acfa6d6`.
The measured bounded run submitted 2,050 batches, retired 2,048, failed none, and progressed into
AGC work. Current compatibility evidence lives in `docs/status.md`; the evidence below is the
pre-implementation problem statement, not the current title state.

## Goal

Implement the PlayStation 5 Audio Compute Manager public contract completely enough that titles
can build, submit, execute, and retire real audio DSP batches without fabricated success. The first
live acceptance target is Demon’s Souls progressing beyond its current pre-VideoOut SCREAM audio
stall. The implementation must remain title-independent and numerically testable.

This is the first subproject in the broader “get the game working” program. Later guest-service,
graphics-correctness, and performance blockers get separate evidence, specifications, and plans
after ACM is complete.

## Pre-implementation evidence

- The 2026-08-11 bounded Demon’s Souls run reached no VideoOut, AGC, draw, flip, or present work.
  It repeatedly failed `sceAcmBatchStartBuffers` and `sceAcm_ConvReverb_SharedInput` while its audio
  workers saturated the CPU.
- REmu currently models only ACM context handles in `remu-runtime/src/lib.rs`; it has no batch
  parser, asynchronous lifecycle, or DSP implementation.
- SDK 10.00 defines ACM as an asynchronous ACV service with contexts, batches, waits,
  notifications, statistics, FFT/IFFT, panning, and convolution reverb.
- SharpEmu commit `5f1bd5a` and KytyPS5 both progress by advancing batch offsets by guessed
  constants and immediately reporting completion. That is useful differential evidence but is not
  an acceptable implementation: it neither executes DSP nor validates the SDK contract.
- REmu’s AJM engine already demonstrates the correct architectural pattern for an independently
  locked audio-job service whose blocking waits never hold the process-wide dispatch lock.

## Success criteria

1. Every callable `sceAcm*` export in the SDK 10 stub archive is either implemented or is proven to
   be an inline/header-only API that requires no import.
2. Context and batch identifiers have checked lifetimes, deterministic reset behavior, and exact
   documented error results.
3. Both single-buffer and multi-buffer submissions validate and execute complete command streams.
4. Polling, finite waits, infinite waits, single-waiter behavior, completion notification, and
   statistics follow the SDK contract without holding `PROBE_DISPATCH`.
5. FFT, IFFT, Panner, ConvReverb SharedInput/SharedIr, and their versioned wrappers produce real
   output for every SDK-documented format, flag, channel layout, and block size.
6. Invalid pointers, alignment, buffer capacity, command order, context, batch, and DSP shape fail
   with the matching `SCE_ACM_ERROR_*` value; invalid work never returns success.
7. Numerical fixtures compare against independent scalar references, not against the production
   implementation itself.
8. Demon’s Souls reaches the next observable subsystem boundary with zero unimplemented ACM NIDs;
   reaching a new blocker is success for this subproject, not permission to hide that blocker.

## Non-goals

- No title-specific NID, address, command-size, or audio-data workaround.
- No immediate-success or forced-silence compatibility mode.
- No Metal compute implementation in the first correct version. GPU acceleration can be considered
  only after CPU output is fixture-verified and profiling identifies ACM DSP as a material cost.
- No claim that completing ACM alone makes Demon’s Souls playable.
- No unrelated AJM, AudioOut2, renderer, or guest-thread refactor in this subproject.

## Sources of truth

The implementation order of authority is:

1. SDK 10 public headers under `REMu-research/sdk-1000/sdk/target/include/acm.h` and `acm/`.
2. SDK stub symbols and `.scenid` records for exact exported names and NIDs.
3. Existing REmu guest-memory, error, import-dispatch, and AJM lifecycle conventions.
4. Independent mathematical definitions for FFT, IFFT, biquad filtering, gain ramps, overlap-add,
   and partitioned convolution.
5. SharpEmu/KytyPS5 only as behavioral differentials; their guessed batch sizes and immediate
   completion are explicitly non-authoritative.

## Architecture

### Module boundary

Create `crates/remu-runtime/src/acm_engine/` beside `ajm_engine/`:

- `mod.rs`: export dispatch, engine ownership, reset, contexts, batches, queueing, and waits.
- `abi.rs`: guest structure decoding/encoding, pointer and range checks, alignment, error values,
  batch-command words, and safe guest writes.
- `batch.rs`: immutable decoded command graph, validation, command ordering, and execution plan.
- `builders.rs`: HLE implementations of exported ACM job-building functions. Builders append
  self-contained records to the caller’s batch buffer and advance `SceAcmBatchInfo::szOffset` by
  the exact SDK-defined command count.
- `dsp/fft.rs`: fp16/fp32 FFT/IFFT formats, packing, splitting, zero/drop-half flags, and plan cache.
- `dsp/panner.rs`: gain matrices, per-sample ramps, biquad cascades, state, and output accumulation.
- `dsp/convolution.rs`: input history, IR partitions, high-frequency block sizes, channel mapping,
  overlap-add, gains, v2 null-tail optimization, and output crossfades.
- `tests.rs` plus focused DSP fixture modules: ABI, lifecycle, malformed streams, concurrency, and
  independent numerical goldens.

ACM leaves `ProbeDispatchState`. `dispatch_stateless_import` consults `acm_engine::dispatch` before
acquiring the process-wide lock, exactly as it does for AJM. No ACM wait or DSP work is allowed to
run under `PROBE_DISPATCH`.

### State ownership

`AcmEngine` is process-global but reset for every new guest process. It owns:

- monotonically allocated nonzero context IDs;
- one FIFO of batches per context;
- a bounded worker pool that may execute different contexts concurrently but preserves FIFO within
  each context;
- a batch table keyed by nonzero batch ID;
- one mutex and condition variable for lifecycle state;
- immutable decoded batch plans separated from mutable completion/waiter state;
- bounded statistics history expressed using `SCE_ACM_JOB_STATISTICS_SCALE`.

Context destruction synchronously drains already accepted batches for that context, wakes waiters,
then removes the context. A nonexistent context returns `SCE_ACM_ERROR_INVALID_CONTEXT`. Process
reset cancels queued work, wakes every waiter, joins workers, clears identifiers and statistics,
and cannot publish into the next title’s guest memory.

SDK 10 documents no `BUSY` result for context destruction, so destruction cannot reject a valid
context merely because it owns work. Draining is the deterministic policy for normal destruction;
process reset is the separate cancellation path used when the guest is already being torn down.

### Batch lifecycle

The lifecycle is `Pending -> Running -> Retired` or `Pending/Running -> Failed`. A failed batch
retains its final documented ACM error code for wait-time error selection and host diagnostics.
SDK 10 declares `SceAcmBatchError` as opaque and explicitly says batch start currently supports
only a null `pBatchError`; a non-null pointer therefore returns `SCE_ACM_ERROR_INVALID_PARAMETER`
and is never dereferenced or populated.

`sceAcmBatchStartBuffer` and `sceAcmBatchStartBuffers`:

1. validate context, the required null `pBatchError`, pointer arrays, `SceAcmBatchInfo` fields,
   offset/capacity arithmetic, mapped command spans, and output batch pointer;
2. copy and decode all command bytes into an immutable host plan before returning;
3. reject malformed ordering, unsupported function/group combinations, unterminated chains,
   invalid notification placement, and pointer/alignment failures;
4. allocate and publish the batch ID only after validation succeeds;
5. queue the batch asynchronously and signal a worker.

The `iBatchInfoCount == 0` convention represents the single `SceAcmBatchInfo*` wrapper used by
`sceAcmBatchStart`. A nonzero count represents an array of batch-info pointers. Every byte-count and
address addition is checked before guest memory is accessed.

Workers execute commands in guest order, retain output addresses but snapshot input and parameter
data at execution start, publish output/state/statistics, issue a release fence, write the optional
notification last, mark the batch retired, and wake waiters.

`sceAcmBatchWait` implements zero-time polling, finite millisecond deadlines, and
`SCE_ACM_WAIT_INFINITE`. Only one waiter may own a batch. Timeout or a second waiter returns
`SCE_ACM_ERROR_BUSY`; successful wait consumes the retired batch’s internal resources. It never
holds the process-wide import lock.

### Command representation

Public generic command identifiers 0 through 17 from `acm/commands.h` are decoded, including chain
activation, barriers, input/output and split transfers, run, statistics, inline/constant data,
notification, queue, and delay.

Exported DSP builders serialize self-contained job records into the guest batch buffer rather than
placing host-only descriptors in a side table. This preserves behavior when a title copies or
reuses batch buffers. Records use the SDK’s eight-byte `SceAcmBatchCommand` units, the exact public
`*_SIZE` command count, checked capacity advancement, and all guest addresses needed for later
execution. The parser accepts these records alongside generic commands emitted directly by inline
SDK helpers.

Unknown command, function, group, flag, or malformed size is a loud malformed-batch failure. It is
never skipped and never treated as a no-op.

## DSP behavior

### FFT and IFFT

The first implementation uses a deterministic, cross-platform CPU backend. A small plan cache is
keyed by transform length and direction; cached plans contain immutable twiddles/permutations.
Supported fp16/fp32 formats, split input/output, zero-second-half FFT input, drop-first-half IFFT
output, channel counts, alignment, and SDK sizes are validated before a job is queued.

Production transforms use `rustfft` 6.4.x with its default AArch64 NEON backend; fp16 conversion
uses `half` 2.7.x. Both are pure Rust and MIT-or-Apache-2.0 licensed. Exact patch versions are
locked by `Cargo.lock`, while the manifest uses compatible minor requirements. Tests use a separate
O(n²) scalar DFT/IDFT, including impulses, DC, Nyquist, tones, seeded random vectors, round trips,
fp16 error bounds, and every flag combination. Production code and reference tests do not share
transform logic.

### Panner

Panner execution reads every documented parameter/state layout, validates channel and biquad
counts, applies key-on/reset semantics, performs per-sample gain interpolation, executes each
biquad in documented order while updating persistent state, optionally accumulates initial output,
and writes every output channel. Tests cover mono/stereo/multichannel mapping, zero gain, unity gain,
ramps, cascaded biquads, persistent state across batches, initial-output accumulation, and invalid
shape/alignment cases.

### Convolution reverb

ConvReverb implements the SDK’s 256/512/1024-sample block sizes and fp16/fp32 frequency-domain IR
formats. Each call:

1. transforms the new PCM block and advances the input history ring;
2. applies per-block IR sizes and gains;
3. multiplies and accumulates compatible input/IR channel partitions;
4. performs the inverse transform;
5. overlap-adds persistent temp data;
6. applies per-IR or per-input gain;
7. publishes the requested output channel layout.

SharedInput supports 1:N, N:1, and N:N mappings for multiple IRs. SharedIr supports the same
mappings for multiple inputs. Version 2 additionally tracks consecutive null input history,
rejects live PCM after tail mode begins as documented, ramps `fGainPrev -> fGain`, and optionally
mixes `pPcmInit` before writing output. All input-history counters and output gain state are updated
only after successful execution.

Independent direct time-domain convolution and overlap-add references verify impulses, delayed
impulses, identity IRs, stereo mapping, multi-partition tails, fp16 IR tolerance, gains, history
wraparound, null tails, crossfades, and initial-output mixing.

## Guest memory and errors

- All guest reads and writes use existing mapped-range validation utilities.
- Required ACM data alignment is 128 bytes; pointer arrays and structures receive their natural SDK
  alignment checks in addition to pointed-to audio buffers.
- Output writes call the existing guest-memory write notification path so CPU write watches and GPU
  residency cannot retain stale bytes.
- Partial output is forbidden: validation and input snapshots finish before the first guest output
  byte changes. A runtime page fault fails the whole batch with `SCE_ACM_ERROR_PAGEFAULT`.
- Builder failures leave `szOffset` and all destination/state data unchanged.
- Batch-start failures do not allocate or publish an ID.
- Every returned error is the documented 32-bit ACM code sign-extended through the emulator ABI.
- No pointer fault is translated into generic kernel `EFAULT` when the ACM API documents
  `SCE_ACM_ERROR_INVALID_PARAMETER` or `SCE_ACM_ERROR_PAGEFAULT`.

## Performance and observability

- Batch parsing, input snapshots, and DSP execution are outside the process-wide dispatch lock.
- Plan/twiddle caches are bounded and immutable after insertion.
- Context FIFO ordering avoids races in guest-owned history/state; different contexts may execute
  concurrently.
- Small batches execute without per-command heap allocation after plan decoding.
- Telemetry reports submitted/retired/failed batches, queue age, execution time, function counts,
  processed samples, and numerical backend. First failure plus power-of-two repeats reports command
  and DSP validation errors without flooding logs.
- No deep tracing, frame capture, or Metal HUD is enabled during FPS checks.

## Testing and verification

Every behavior change follows red/green TDD. Verification layers are:

1. ABI layout and error-code tests derived from SDK 10 headers.
2. Builder capacity/alignment tests proving exact offset advancement and no mutation on failure.
3. Parser tests for all command IDs, valid chains, malformed ordering, overflows, and copied batch
   buffers.
4. Context/batch lifecycle tests for FIFO, independent contexts, polling, finite/infinite waits,
   concurrent waiter rejection, destruction, and process reset.
5. Independent numerical FFT/IFFT, Panner, and ConvReverb fixtures.
6. Full `remu-runtime` tests single-threaded, plus workspace check, formatting, Clippy, and existing
   shader/GPU suites where the shared dirty tree permits.
7. Sparse live gates only after offline suites pass:
   - one bounded boot run proving zero unimplemented ACM imports and retired ACM batches;
   - one bounded progression run proving VideoOut/AGC activity or identifying the next loud blocker;
   - one longer menu/gameplay run only if the second gate advances.

Live runs are tagged non-evidence while unrelated dirty changes remain. A run that advances but
produces incorrect audio is not acceptance; captured PCM hashes/statistics and audible output must
agree with the offline DSP contracts.

## Rollout

Implementation is split into independently reviewed milestones:

1. ABI/export inventory and context/batch lifecycle.
2. Generic command parser, notification, statistics, and buffer-transfer commands.
3. FFT/IFFT builders and execution.
4. Panner builders and execution.
5. ConvReverb SharedInput/SharedIr v1 and v2.
6. Full offline verification and sparse Demon’s Souls gates.

No milestone may replace a missing operation with success. Each milestone either executes the
documented behavior or returns the documented error until the next milestone completes it. The
final live gate begins only after all six milestones are green.
