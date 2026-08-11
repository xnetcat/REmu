# Complete ACM Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement all 13 SDK 10 `libSceAcm` exports as a real asynchronous CPU audio-DSP engine so Demon’s Souls can leave its current pre-VideoOut ACM stall without fabricated success.

**Architecture:** Add a process-global `AcmEngine` beside `AjmEngine`, with private lifecycle synchronization, per-context FIFO execution, typed immutable batch plans, and host-side builders that serialize self-contained records into the guest’s ABI-sized command arena. FFT/IFFT, Panner, and partitioned convolution run on bounded CPU workers using `rustfft`; all guest-memory validation and output publication remain explicit and testable.

**Tech Stack:** Rust 2024, standard-library `Mutex`/`Condvar`, `rustfft` 6.4.x with default NEON support, `half` 2.7.x, Cargo unit tests, SDK 10 ACM headers.

## Global Constraints

- The 13 exported symbols and NIDs are the SDK 10 stub inventory; inline-only wrappers are not additional imports.
- Preserve unrelated dirty workspace changes and stage only exact ACM task files in each commit.
- Never return success for an unknown command, function, group, flag, or malformed record.
- Every production behavior starts with a focused test that is observed failing for the intended reason.
- Required audio data alignment is exactly 128 bytes; structure and pointer-array alignment also follow their natural LP64 alignment.
- `SceAcmBatchError` is opaque in SDK 10; batch start accepts only a null error pointer.
- A valid context destroy drains accepted work; process reset cancels queued work and wakes waiters.
- ACM waits and DSP execution never hold `PROBE_DISPATCH`.
- Production FFT logic and O(n²) reference DFT logic must not share transform code.
- No game launch occurs until all offline ACM tests pass; the first live gate is bounded to 90 seconds.

---

### Task 1: Establish the SDK ABI, export inventory, and context lifecycle

**Files:**
- Create: `crates/remu-runtime/src/acm_engine/mod.rs`
- Create: `crates/remu-runtime/src/acm_engine/abi.rs`
- Create: `crates/remu-runtime/src/acm_engine/tests.rs`
- Modify: `crates/remu-runtime/src/lib.rs`
- Modify: `crates/remu-runtime/src/nid_table.rs`
- Test: `crates/remu-runtime/src/platform_independent_tests/services.rs`

**Interfaces:**
- Produces: `acm_engine::dispatch(nid: &str, arguments: [u64; 6], stack_arguments: [u64; 6]) -> Option<u64>`.
- Produces: `acm_engine::reset_for_new_process()`.
- Produces: `AcmEngine::{new, context_create, context_destroy}`.
- Produces: `SDK10_ACM_NIDS: [&str; 13]` and `owns_nid(nid: &str) -> bool`.
- Produces: `abi::{SceAcmBatchInfo, SceAcmBatchCommand, AcmFormat, AcmError, return_error, read_pod, write_pod, read_array, checked_span}`.
- Consumes: `crate::{process_guest_range_has_read_access, read_guest_bytes, write_guest_bytes, write_guest_u32, sce_kernel_error}`.

- [ ] **Step 1: Write ABI and export-inventory tests before creating the module**

```rust
#[test]
fn sdk10_abi_sizes_and_error_words_are_exact() {
    assert_eq!(size_of::<SceAcmBatchInfo>(), 24);
    assert_eq!(size_of::<SceAcmBatchCommand>(), 8);
    assert_eq!(AcmError::InvalidContext as u32, 0x8194_0002);
    assert_eq!(AcmError::InvalidParameter as u32, 0x8194_0006);
    assert_eq!(AcmError::PageFault as u32, 0x8194_000b);
}

#[test]
fn dispatch_claims_exactly_the_thirteen_sdk10_exports() {
    for nid in [
        "ZIXln2K3XMk", "jBgBjAj02R8", "tW9W+CAG4FE", "8fe55ktlNVo",
        "RLN3gRlXJLE", "r7z5YQFZo+U", "KovqaFbmtsM", "DR-ZCmvVR9Q",
        "LA4RCNKnFjg", "u70oWo92SYQ", "9nLbWmRDpa8", "wM41Cc+ieog",
        "PVx9oQUaDdk",
    ] {
        assert!(super::owns_nid(nid), "missing {nid}");
    }
    assert!(!super::owns_nid("not-an-acm-nid"));
}
```

- [ ] **Step 2: Run the focused tests and verify RED**

Run: `cargo test -p remu-runtime acm_engine::tests::sdk10_abi_sizes_and_error_words_are_exact acm_engine::tests::dispatch_claims_exactly_the_thirteen_sdk10_exports -- --test-threads=1`

Expected: compilation fails because `acm_engine`, its ABI types, and `owns_nid` do not exist.

- [ ] **Step 3: Add exact ABI types, errors, and the NID ownership table**

```rust
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct SceAcmBatchInfo {
    pub(crate) buffer: u64,
    pub(crate) offset: u64,
    pub(crate) size: u64,
}

#[repr(transparent)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct SceAcmBatchCommand(pub(crate) u64);

#[repr(u32)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum AcmFormat {
    Fp32 = 0,
    Fp16 = 1,
}

#[repr(u32)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum AcmError {
    InvalidContext = 0x8194_0002,
    InvalidBatch = 0x8194_0003,
    OutOfMemory = 0x8194_0004,
    OutOfResources = 0x8194_0005,
    InvalidParameter = 0x8194_0006,
    MalformedBatch = 0x8194_0007,
    Busy = 0x8194_0008,
    BatchBufferFull = 0x8194_0009,
    Alignment = 0x8194_000a,
    PageFault = 0x8194_000b,
}

pub(crate) const SDK10_ACM_NIDS: [&str; 13] = [
    "ZIXln2K3XMk", "jBgBjAj02R8", "tW9W+CAG4FE", "8fe55ktlNVo",
    "RLN3gRlXJLE", "r7z5YQFZo+U", "KovqaFbmtsM", "DR-ZCmvVR9Q",
    "LA4RCNKnFjg", "u70oWo92SYQ", "9nLbWmRDpa8", "wM41Cc+ieog",
    "PVx9oQUaDdk",
];

pub(crate) fn owns_nid(nid: &str) -> bool {
    SDK10_ACM_NIDS.contains(&nid)
}
```

- [ ] **Step 4: Write failing context validation, destruction, and reset tests**

```rust
#[test]
fn context_ids_are_checked_and_reset_is_process_local() {
    let engine = AcmEngine::new();
    let mut first = 0_u32;
    assert_eq!(engine.context_create(args(&[address_mut(&mut first)])), 0);
    assert_ne!(first, 0);
    assert_eq!(engine.context_create(args(&[0])), error(AcmError::InvalidParameter));
    assert_eq!(engine.context_destroy(args(&[u64::from(first)])), 0);
    assert_eq!(engine.context_destroy(args(&[u64::from(first)])), error(AcmError::InvalidContext));
    engine.reset();
    let mut after_reset = 0_u32;
    assert_eq!(engine.context_create(args(&[address_mut(&mut after_reset)])), 0);
    assert_eq!(after_reset, 1);
}
```

- [ ] **Step 5: Run the context test and verify RED**

Run: `cargo test -p remu-runtime acm_engine::tests::context_ids_are_checked_and_reset_is_process_local -- --test-threads=1`

Expected: compilation fails because `AcmEngine` and its lifecycle methods are absent.

- [ ] **Step 6: Implement the minimal checked context engine and pre-lock route**

Use `Mutex<EngineState>` with `next_context: u32` and `BTreeMap<u32, ContextState>`. Move context create/destroy out of `ProbeDispatchState`, call `acm_engine::dispatch` immediately after `ajm_engine::dispatch`, call `acm_engine::reset_for_new_process()` from `configure_probe_dispatch`, remove `next_acm_context` and `acm_contexts`, and delete the locked ACM match arms.

- [ ] **Step 7: Add all 13 names to the hand NID table and verify GREEN**

Run: `cargo test -p remu-runtime acm_engine:: platform_independent_tests::services:: -- --test-threads=1`

Expected: all ACM ABI, lifecycle, route, and name tests pass.

- [ ] **Step 8: Commit only Task 1 files**

```bash
git add crates/remu-runtime/src/acm_engine/mod.rs crates/remu-runtime/src/acm_engine/abi.rs crates/remu-runtime/src/acm_engine/tests.rs crates/remu-runtime/src/lib.rs crates/remu-runtime/src/nid_table.rs crates/remu-runtime/src/platform_independent_tests/services.rs
git commit -m "feat: add checked ACM context engine"
```

---

### Task 2: Serialize and parse exact-capacity batch records

**Files:**
- Create: `crates/remu-runtime/src/acm_engine/builders.rs`
- Create: `crates/remu-runtime/src/acm_engine/batch.rs`
- Modify: `crates/remu-runtime/src/acm_engine/mod.rs`
- Modify: `crates/remu-runtime/src/acm_engine/abi.rs`
- Modify: `crates/remu-runtime/src/acm_engine/tests.rs`

**Interfaces:**
- Produces: `builders::append_record(info: u64, words: usize, payload: RecordPayload) -> Result<(), AcmError>`.
- Produces: `batch::decode_buffers(infos: &[SceAcmBatchInfo]) -> Result<BatchPlan, AcmError>`.
- Produces: `BatchPlan { commands: Vec<Command>, notification: Option<Notification> }`.
- Produces: `Command::{Fft, Ifft, Panner, Convolution, Statistics, Transfer, Barrier, Queue, Delay}`.
- Consumes: `abi::SceAcmBatchInfo` and SDK command word size of 8 bytes.

- [ ] **Step 1: Write failing atomic cursor/capacity tests**

```rust
#[repr(align(128))]
struct AlignedBytes([u8; 256]);

#[test]
fn builder_reserves_exact_words_and_does_not_mutate_on_full_or_overflow() {
    let mut arena = AlignedBytes([0; 256]);
    let address = arena.0.as_mut_ptr() as u64;
    let mut info = SceAcmBatchInfo { buffer: address, offset: 8, size: 32 };
    let before = info;
    assert_eq!(append_test_record(address_mut(&mut info), 3), Ok(()));
    assert_eq!(info.offset, 32);
    let full = info;
    assert_eq!(append_test_record(address_mut(&mut info), 1), Err(AcmError::BatchBufferFull));
    assert_eq!(info, full);
    info.offset = u64::MAX - 3;
    assert_eq!(append_test_record(address_mut(&mut info), 1), Err(AcmError::InvalidParameter));
    assert_eq!(info.offset, u64::MAX - 3);
    assert_ne!(before.offset, full.offset);
}
```

- [ ] **Step 2: Run the builder test and verify RED**

Run: `cargo test -p remu-runtime acm_engine::tests::builder_reserves_exact_words_and_does_not_mutate_on_full_or_overflow -- --test-threads=1`

Expected: compilation fails because the builder and aligned test arena do not exist.

- [ ] **Step 3: Implement an atomic record reservation helper**

Use a 64-bit record prefix containing magic `0x524d_4143` (`RMAC`), record kind, version, and total word count. Validate the entire old `SceAcmBatchInfo`, checked `offset + words * 8`, and the complete destination span before writing; write payload and zero padding first, then update `info.offset` last. The arena encoding is private, while its occupied word count exactly matches each public `*_SIZE` macro.

- [ ] **Step 4: Write failing copied-arena and malformed-record parser tests**

```rust
#[test]
fn decoded_plan_survives_arena_copy_and_unknown_records_fail_loudly() {
    let original = fixture_with_notification_and_statistics();
    let copied = original.copy_arena_and_rebase_info();
    let plan = decode_buffers(&[copied.info]).expect("valid copied command stream");
    assert!(matches!(plan.commands.as_slice(), [Command::Statistics(_)]));
    assert_eq!(plan.notification.expect("last notification").value, 7);

    let mut corrupt = copied;
    corrupt.write_u8(4, 0xff);
    assert_eq!(decode_buffers(&[corrupt.info]), Err(AcmError::MalformedBatch));
}
```

- [ ] **Step 5: Run the parser test and verify RED**

Run: `cargo test -p remu-runtime acm_engine::tests::decoded_plan_survives_arena_copy_and_unknown_records_fail_loudly -- --test-threads=1`

Expected: compilation fails because `BatchPlan`, `Command`, and `decode_buffers` are absent.

- [ ] **Step 6: Implement immutable decoding for private records and inline SDK records**

Decode private records by value with no host pointer side table. Decode SDK command 11 as a two-word statistics request and command 14 as a two-word completion notification. Enforce that statistics has no following statistics job and notification is the final command of the final buffer. Reject nonzero reserved bits, truncated records, zero-length records, and an inline command whose required continuation word crosses a buffer boundary.

- [ ] **Step 7: Run Task 2 tests and verify GREEN**

Run: `cargo test -p remu-runtime acm_engine::tests::builder_ acm_engine::tests::decoded_plan_ -- --test-threads=1`

Expected: exact cursor, copy, statistics, notification, and malformed-stream tests pass.

- [ ] **Step 8: Commit only Task 2 files**

```bash
git add crates/remu-runtime/src/acm_engine/mod.rs crates/remu-runtime/src/acm_engine/abi.rs crates/remu-runtime/src/acm_engine/builders.rs crates/remu-runtime/src/acm_engine/batch.rs crates/remu-runtime/src/acm_engine/tests.rs
git commit -m "feat: add ACM batch record parser"
```

---

### Task 3: Execute asynchronous batches with per-context FIFO waits

**Files:**
- Modify: `crates/remu-runtime/src/acm_engine/mod.rs`
- Modify: `crates/remu-runtime/src/acm_engine/batch.rs`
- Modify: `crates/remu-runtime/src/acm_engine/tests.rs`

**Interfaces:**
- Produces: `AcmEngine::{batch_start_buffer, batch_start_buffers, batch_wait, execute_next_for_test}`.
- Produces: `BatchStatus::{Pending, Running, Retired, Failed, Cancelled}`.
- Produces: a bounded two-thread worker pool in `AcmEngine::global()`.
- Consumes: immutable `BatchPlan` from Task 2.

- [ ] **Step 1: Write failing start/wait/error-lifetime tests**

```rust
#[test]
fn batch_start_is_async_wait_consumes_retirement_and_poll_reports_busy() {
    let engine = AcmEngine::new();
    let context = create_context(&engine);
    let mut fixture = empty_valid_batch();
    let batch = fixture.start(&engine, context).expect("valid batch starts");
    assert_eq!(engine.batch_wait(args(&[context.into(), batch.into(), 0])), error(AcmError::Busy));
    assert!(engine.execute_next_for_test());
    assert_eq!(engine.batch_wait(args(&[context.into(), batch.into(), 0])), 0);
    assert_eq!(engine.batch_wait(args(&[context.into(), batch.into(), 0])), error(AcmError::InvalidBatch));
    assert_eq!(fixture.start_with_non_null_error(&engine, context), Err(AcmError::InvalidParameter));
}

#[test]
fn finite_and_infinite_waits_share_one_waiter_slot() {
    let engine = AcmEngine::new();
    let context = create_context(&engine);
    let batch = empty_valid_batch().start(&engine, context).unwrap();
    assert_eq!(engine.batch_wait(args(&[context.into(), batch.into(), 1])), error(AcmError::Busy));
    engine.set_waiter_active_for_test(batch, true);
    assert_eq!(engine.batch_wait(args(&[context.into(), batch.into(), u64::from(u32::MAX)])), error(AcmError::Busy));
    engine.set_waiter_active_for_test(batch, false);
    std::thread::scope(|scope| {
        scope.spawn(|| assert!(engine.execute_next_for_test()));
        assert_eq!(engine.batch_wait(args(&[context.into(), batch.into(), u64::from(u32::MAX)])), 0);
    });
}
```

- [ ] **Step 2: Run the lifecycle test and verify RED**

Run: `cargo test -p remu-runtime acm_engine::tests::batch_start_is_async_wait_consumes_retirement_and_poll_reports_busy acm_engine::tests::finite_and_infinite_waits_share_one_waiter_slot -- --test-threads=1`

Expected: compilation fails because batch lifecycle methods and statuses are absent.

- [ ] **Step 3: Implement checked starts and terminal waits**

For `sceAcmBatchStartBuffers`, interpret count zero as one indirect `SceAcmBatchInfo*`; otherwise read `count` pointers. For `sceAcmBatchStartBuffer`, wrap the raw command address/byte length as one `SceAcmBatchInfo`. Validate and decode before allocating a nonzero batch ID. Implement timeout zero, finite milliseconds using one absolute `Instant`, and `u32::MAX` infinite wait. A timeout or second waiter returns `Busy`; a successful or failed terminal wait removes the batch.

- [ ] **Step 4: Write failing FIFO, cross-context concurrency, destroy, and reset tests**

```rust
#[test]
fn each_context_is_fifo_while_distinct_contexts_can_run_independently() {
    let engine = AcmEngine::new();
    let a = create_context(&engine);
    let b = create_context(&engine);
    let a1 = start_tagged(&engine, a, 11);
    let a2 = start_tagged(&engine, a, 12);
    let b1 = start_tagged(&engine, b, 21);
    assert_eq!(engine.runnable_for_test(), vec![a1, b1]);
    engine.execute_batch_for_test(b1);
    engine.execute_batch_for_test(a1);
    assert_eq!(engine.runnable_for_test(), vec![a2]);
    assert_eq!(engine.execution_tags_for_test(), vec![21, 11]);
}
```

- [ ] **Step 5: Run concurrency tests and verify RED**

Run: `cargo test -p remu-runtime acm_engine::tests::each_context_is_fifo_while_distinct_contexts_can_run_independently acm_engine::tests::destroy_drains_and_reset_cancels -- --test-threads=1`

Expected: assertions fail because queue selection and drain/cancel behavior are not implemented.

- [ ] **Step 6: Add per-context runnable scheduling and process reset cancellation**

Keep one `VecDeque<u32>` per context and an `active_batch: Option<u32>`. Workers claim only a context with no active batch, execute the plan outside the lifecycle mutex, then reacquire it to publish status and expose that context’s next batch. Context destroy waits on the engine condition variable until its queue and active slot are empty. Process reset marks the pool stopping, removes queued work, marks retained batches cancelled, wakes workers and waiters, takes and joins both worker handles outside the state mutex, clears state and identifiers, then starts a fresh two-worker pool. A generation token prevents a worker that was already publishing when reset began from writing into the next process.

- [ ] **Step 7: Verify Task 3 GREEN and the pre-lock wait invariant**

Run: `cargo test -p remu-runtime acm_engine::tests:: platform_independent_tests::synchronization:: -- --test-threads=1`

Expected: all lifecycle and synchronization tests pass without dispatch-lock wait assertions.

- [ ] **Step 8: Commit only Task 3 files**

```bash
git add crates/remu-runtime/src/acm_engine/mod.rs crates/remu-runtime/src/acm_engine/batch.rs crates/remu-runtime/src/acm_engine/tests.rs
git commit -m "feat: execute asynchronous ACM batches"
```

---

### Task 4: Implement generic commands, notification, and statistics

**Files:**
- Modify: `crates/remu-runtime/src/acm_engine/batch.rs`
- Modify: `crates/remu-runtime/src/acm_engine/builders.rs`
- Modify: `crates/remu-runtime/src/acm_engine/mod.rs`
- Modify: `crates/remu-runtime/src/acm_engine/tests.rs`

**Interfaces:**
- Produces: typed decoding for public command IDs 0 through 17.
- Produces: `execute_control_command(command: &Command, execution: &mut ExecutionState) -> Result<(), AcmError>`.
- Produces: `AcmStatistics { flags, usage_batch, usage_intervals, pad }`.
- Produces: `ExecutionMemory` for owned input snapshots and `PendingWrite { address: u64, bytes: Vec<u8> }` for atomic output publication.
- Consumes: task generation, guest output publication, and batch timing from Task 3.

- [ ] **Step 1: Write one failing table test covering all 18 public command IDs**

```rust
#[test]
fn every_public_command_id_has_a_typed_decoder_and_reserved_bits_are_rejected() {
    for id in 0_u8..=17 {
        let stream = minimal_sdk_stream_for(id);
        assert!(decode_buffers(&[stream.info]).is_ok(), "command {id}");
    }
    let invalid = command_word(18, 0, 0, 0, 0, 0);
    assert_eq!(decode_raw_words(&[invalid]), Err(AcmError::MalformedBatch));
    let reserved = command_word(1, 0xff, 0, 0, 0, 0);
    assert_eq!(decode_raw_words(&[reserved]), Err(AcmError::MalformedBatch));
}
```

- [ ] **Step 2: Run the decoder table test and verify RED**

Run: `cargo test -p remu-runtime acm_engine::tests::every_public_command_id_has_a_typed_decoder_and_reserved_bits_are_rejected -- --test-threads=1`

Expected: command IDs other than statistics and notification are rejected.

- [ ] **Step 3: Decode every command into explicit execution semantics**

Map activate/barrier/queue/delay to ordering nodes; input/output/split/inline/constant records to checked byte transfers; run to the selected DSP operation; statistics to a terminal statistics node; notification to a final release-published byte. Validate chain activation before chain commands, buffer indices, function/group matches, `iCount` continuation lengths, `iSize1` byte spans, and queue/delay bounds. If an SDK field lacks a public semantic for the requested function/group combination, return `MalformedBatch` rather than treating it as a no-op. For notification memory, accept a valid aligned CPU range but publish the byte only when `remu_memory::guest_shared_mapping_identity(address, 128)` proves GPU-visible backing, matching the SDK’s success-without-notification rule for non-GPU-mapped storage.

- [ ] **Step 4: Write failing output-atomicity, notification-order, and statistics tests**

```rust
#[test]
fn notification_is_written_last_and_failed_batches_publish_no_partial_output() {
    let fixture = transfer_then_fault_then_notify_fixture();
    let before = fixture.output.clone();
    let result = execute_plan(fixture.plan);
    assert_eq!(result, Err(AcmError::PageFault));
    assert_eq!(fixture.output, before);
    assert_eq!(fixture.notification.value, 0);
}

#[test]
fn statistics_scale_real_elapsed_time_and_mark_unavailable_history() {
    let stats = run_statistics_fixture(Duration::from_millis(2), &[0, 1 << 20]);
    assert!(stats.usage_batch > 0);
    assert_eq!(stats.usage_intervals[0], u32::MAX);
    assert_ne!(stats.usage_intervals[1], u32::MAX);
}
```

- [ ] **Step 5: Run publication tests and verify RED**

Run: `cargo test -p remu-runtime acm_engine::tests::notification_is_written_last_and_failed_batches_publish_no_partial_output acm_engine::tests::statistics_scale_real_elapsed_time_and_mark_unavailable_history -- --test-threads=1`

Expected: output or notification changes before the planned error, and statistics fields remain unset.

- [ ] **Step 6: Execute into host staging buffers and publish atomically**

Snapshot all inputs before the first command executes. Accumulate output writes as `(guest_address, Vec<u8>)`, reject overlapping writes with incompatible sizes, validate every final destination again, then publish output/state/statistics through `write_guest_bytes`. Issue `std::sync::atomic::fence(Ordering::Release)` before writing the optional one-byte notification. Retain a bounded 256-entry execution history for interval statistics.

- [ ] **Step 7: Verify Task 4 GREEN**

Run: `cargo test -p remu-runtime acm_engine::tests:: -- --test-threads=1`

Expected: all command, atomicity, notification, and statistics tests pass.

- [ ] **Step 8: Commit only Task 4 files**

```bash
git add crates/remu-runtime/src/acm_engine/batch.rs crates/remu-runtime/src/acm_engine/builders.rs crates/remu-runtime/src/acm_engine/mod.rs crates/remu-runtime/src/acm_engine/tests.rs
git commit -m "feat: execute ACM control commands"
```

---

### Task 5: Implement numerically verified FFT and IFFT

**Files:**
- Modify: `Cargo.toml`
- Modify: `Cargo.lock`
- Modify: `crates/remu-runtime/Cargo.toml`
- Create: `crates/remu-runtime/src/acm_engine/dsp/mod.rs`
- Create: `crates/remu-runtime/src/acm_engine/dsp/fft.rs`
- Create: `crates/remu-runtime/src/acm_engine/dsp/fft_tests.rs`
- Modify: `crates/remu-runtime/src/acm_engine/builders.rs`
- Modify: `crates/remu-runtime/src/acm_engine/batch.rs`
- Modify: `crates/remu-runtime/src/acm_engine/mod.rs`

**Interfaces:**
- Produces: `FftJob { direction, size, count, input_format, output_format, inputs, outputs, flags }`.
- Produces: `fft::execute(job: &FftJob, memory: &ExecutionMemory) -> Result<Vec<PendingWrite>, AcmError>`.
- Produces: packed real-spectrum helpers with DC in element 0 real and Nyquist in element 0 imaginary.
- Consumes: `rustfft::{FftPlanner, num_complex::Complex32}` and `half::f16`.

- [ ] **Step 1: Add dependencies and write independent DFT fixture tests**

```rust
#[test]
fn fft_impulse_dc_nyquist_and_round_trip_match_scalar_reference() {
    for size in [512_usize, 1024, 2048] {
        for signal in reference_signals(size) {
            let expected = scalar_real_dft(&signal);
            let packed = execute_fft_f32(&signal, size, 0).expect("valid FFT");
            assert_complex_close(&unpack_real_spectrum(&packed), &expected, 2.0e-4);
            let restored = execute_ifft_f32(&packed, size, 0).expect("valid IFFT");
            assert_samples_close(&restored, &signal, 3.0e-4);
        }
    }
}
```

- [ ] **Step 2: Run FFT fixtures and verify RED**

Run: `cargo test -p remu-runtime acm_engine::dsp::fft_tests::fft_impulse_dc_nyquist_and_round_trip_match_scalar_reference -- --test-threads=1`

Expected: compilation fails because the DSP FFT module and execution helpers are absent.

- [ ] **Step 3: Implement cached transforms and Sony real-spectrum packing**

Use one mutex-protected `FftPlanner<f32>` only to construct plans and store plans as `Arc<dyn Fft<f32>>` keyed by `(size, direction)`. Reconstruct the conjugate half before inverse transforms, apply RustFFT’s required `1 / size` normalization on IFFT, and convert fp16 only at guest-memory boundaries.

- [ ] **Step 4: Write failing builder-size, split-buffer, and flag validation tests**

```rust
#[test]
fn fft_builders_use_sdk_word_counts_and_honor_all_split_flags() {
    for flags in 0_u32..=7 {
        let count = 3_usize;
        let fft_words = 2 + count * (7 + usize::from(flags & 2 != 0) + usize::from(flags & 4 != 0));
        assert_eq!(build_fft_fixture(count, flags).used_words(), fft_words);
    }
    assert_eq!(build_fft_fixture(1, 8).result, Err(AcmError::InvalidParameter));
}
```

- [ ] **Step 5: Run builder/flag tests and verify RED**

Run: `cargo test -p remu-runtime acm_engine::dsp::fft_tests::fft_builders_use_sdk_word_counts_and_honor_all_split_flags acm_engine::dsp::fft_tests::fp16_and_half_frame_flags_match_reference -- --test-threads=1`

Expected: builder dispatch is missing or uses an incorrect word count/flag layout.

- [ ] **Step 6: Implement both imported builders and all documented formats/flags**

Handle sizes 512/1024/2048, positive count, formats fp32/fp16, split input/output, FFT zero-second-half, and IFFT drop-first-half. Validate every pointer-array span and each 128-byte audio pointer before mutating the batch cursor. Serialize all addresses and formats into the private record and execute them from the decoded copy.

- [ ] **Step 7: Verify Task 5 GREEN**

Run: `cargo test -p remu-runtime acm_engine::dsp::fft_tests:: -- --test-threads=1`

Expected: scalar references, fp16 tolerances, all flags, alignment failures, and exact builder sizes pass.

- [ ] **Step 8: Commit only Task 5 files**

```bash
git add Cargo.toml Cargo.lock crates/remu-runtime/Cargo.toml crates/remu-runtime/src/acm_engine/dsp/mod.rs crates/remu-runtime/src/acm_engine/dsp/fft.rs crates/remu-runtime/src/acm_engine/dsp/fft_tests.rs crates/remu-runtime/src/acm_engine/builders.rs crates/remu-runtime/src/acm_engine/batch.rs crates/remu-runtime/src/acm_engine/mod.rs
git commit -m "feat: execute ACM FFT jobs"
```

---

### Task 6: Implement Panner filtering, gain ramps, and persistent state

**Files:**
- Create: `crates/remu-runtime/src/acm_engine/dsp/panner.rs`
- Create: `crates/remu-runtime/src/acm_engine/dsp/panner_tests.rs`
- Modify: `crates/remu-runtime/src/acm_engine/dsp/mod.rs`
- Modify: `crates/remu-runtime/src/acm_engine/builders.rs`
- Modify: `crates/remu-runtime/src/acm_engine/batch.rs`
- Modify: `crates/remu-runtime/src/acm_engine/mod.rs`

**Interfaces:**
- Produces: `PannerJob { input_count, biquad_count, update_log2, output_count, inputs, parameters, states, initial_outputs, outputs }`.
- Produces: `panner::execute(job: &PannerJob, memory: &ExecutionMemory) -> Result<Vec<PendingWrite>, AcmError>`.
- Consumes: fixed ACM Panner frame size of 256 fp32 samples per voice and key-on flag bit 0.

- [ ] **Step 1: Write failing identity, ramp, and initial-mix tests**

```rust
#[test]
fn panner_applies_per_sample_gain_ramp_and_optional_initial_output() {
    let input = vec![1.0_f32; 256];
    let initial = vec![0.25_f32; 256];
    let mut fixture = PannerFixture::mono_to_stereo(input, [0.5, 1.0]);
    fixture.previous_gains = [0.0, 0.5];
    fixture.initial_outputs = [Some(initial.clone()), None];
    let output = fixture.execute().expect("valid panner job");
    assert!((output[0][0] - 0.25).abs() < 1.0e-6);
    assert!((output[0][255] - 0.75).abs() < 1.0e-5);
    assert!((output[1][0] - 0.5).abs() < 1.0e-6);
    assert!((output[1][255] - 1.0).abs() < 1.0e-5);
}
```

- [ ] **Step 2: Run Panner output tests and verify RED**

Run: `cargo test -p remu-runtime acm_engine::dsp::panner_tests::panner_applies_per_sample_gain_ramp_and_optional_initial_output -- --test-threads=1`

Expected: compilation fails because Panner DSP and fixtures do not exist.

- [ ] **Step 3: Implement deterministic gain interpolation and accumulation**

For sample `n` in 0..256 use `previous + (target - previous) * (n / 255.0)`. Filter each mono voice before distributing it to outputs, sum every voice contribution, optionally begin from `pOutInit[channel]`, and write the target gains back to each voice state only after the complete job succeeds.

- [ ] **Step 4: Write failing biquad/key-on/persistent-state tests**

```rust
#[test]
fn panner_biquads_update_at_half_frames_and_key_on_resets_state() {
    let coefficients = [identity_biquad(), low_pass_biquad()];
    let mut fixture = PannerFixture::one_voice_one_output_with_updates(coefficients);
    fixture.state_biquads[0] = [0.75, -0.25];
    let continued = fixture.execute().expect("continued state");
    fixture.flags = SCE_ACM_PANNER_FLAGS_KEYON;
    let reset = fixture.execute().expect("key-on state");
    assert_ne!(continued[0], reset[0]);
    assert_reference_biquad_updates(&reset, &fixture.input, &coefficients, 128);
}
```

- [ ] **Step 5: Run state tests and verify RED**

Run: `cargo test -p remu-runtime acm_engine::dsp::panner_tests::panner_biquads_update_at_half_frames_and_key_on_resets_state acm_engine::dsp::panner_tests::panner_rejects_invalid_shapes_without_mutation -- --test-threads=1`

Expected: state is ignored, updated at the wrong boundary, or mutated on failure.

- [ ] **Step 6: Implement variable parameter/state layouts and the 20-word builder**

Validate input count, biquad count 0..=2, update log2 0..=1, output count 1..=36, all pointer arrays, 128-byte voice/parameter/state/output buffers, parameter bytes `4 + biquads * (1 << updates) * 20 + outputs * 4`, and state bytes `biquads * 8 + outputs * 4` rounded to 128. Apply Direct Form II transposed `(b0,b1,b2,a1,a2)` cascades, switching coefficient sets at sample 128 when update log2 is 1. Serialize exactly 20 command words.

- [ ] **Step 7: Verify Task 6 GREEN**

Run: `cargo test -p remu-runtime acm_engine::dsp::panner_tests:: -- --test-threads=1`

Expected: mapping, ramps, two-biquad cascades, coefficient updates, key-on, persistence, initial mixing, builder size, and invalid-shape tests pass.

- [ ] **Step 8: Commit only Task 6 files**

```bash
git add crates/remu-runtime/src/acm_engine/dsp/mod.rs crates/remu-runtime/src/acm_engine/dsp/panner.rs crates/remu-runtime/src/acm_engine/dsp/panner_tests.rs crates/remu-runtime/src/acm_engine/builders.rs crates/remu-runtime/src/acm_engine/batch.rs crates/remu-runtime/src/acm_engine/mod.rs
git commit -m "feat: execute ACM Panner jobs"
```

---

### Task 7: Decode and build all convolution reverb variants

**Files:**
- Create: `crates/remu-runtime/src/acm_engine/dsp/convolution.rs`
- Create: `crates/remu-runtime/src/acm_engine/dsp/convolution_tests.rs`
- Modify: `crates/remu-runtime/src/acm_engine/dsp/mod.rs`
- Modify: `crates/remu-runtime/src/acm_engine/abi.rs`
- Modify: `crates/remu-runtime/src/acm_engine/builders.rs`
- Modify: `crates/remu-runtime/src/acm_engine/batch.rs`
- Modify: `crates/remu-runtime/src/acm_engine/mod.rs`

**Interfaces:**
- Produces: exact LP64 structs `ConvIn`, `ConvInV2`, `ConvIr`, `ConvOut`, `ConvOutV2`, `ConvChannel`, and `ConvBlockSize`.
- Produces: `ConvolutionJob { sharing, version, block_count, inputs, irs, gains, outputs }`.
- Produces: imported builders for SharedInput/SharedIr v1/v2.
- Consumes: FFT packing and transform plans from Task 5.

- [ ] **Step 1: Write failing ABI layout and graph-validation tests**

```rust
#[test]
fn convolution_struct_layouts_match_sdk10_lp64() {
    assert_eq!(size_of::<ConvChannel>(), 16);
    assert_eq!(size_of::<ConvBlockSize>(), 4);
    assert_eq!(size_of::<ConvIn>(), 40);
    assert_eq!(size_of::<ConvInV2>(), 72);
    assert_eq!(size_of::<ConvIr>(), 32);
    assert_eq!(size_of::<ConvOut>(), 24);
    assert_eq!(size_of::<ConvOutV2>(), 56);
}

#[test]
fn convolution_accepts_only_1_to_n_n_to_1_or_n_to_n_channel_graphs() {
    for (input, ir, output) in [(1, 4, 4), (4, 1, 4), (4, 4, 4)] {
        assert!(validate_channel_graph(input, ir, output).is_ok());
    }
    assert_eq!(validate_channel_graph(2, 3, 3), Err(AcmError::InvalidParameter));
}
```

- [ ] **Step 2: Run layout/graph tests and verify RED**

Run: `cargo test -p remu-runtime acm_engine::dsp::convolution_tests::convolution_struct_layouts_match_sdk10_lp64 acm_engine::dsp::convolution_tests::convolution_accepts_only_1_to_n_n_to_1_or_n_to_n_channel_graphs -- --test-threads=1`

Expected: compilation fails because convolution ABI types and validation are absent.

- [ ] **Step 3: Implement recursive pointer-graph validation without mutation**

Validate block count argument equals 1; block sizes are 256/512/1024 and compatible across each tuple; formats are fp16/fp32; input history capacity/count/position are bounded; channel counts are nonzero; `pSizes` is null for input channels; IR runtime byte size never exceeds memory byte size and respects format alignment; every pointer array and pointed buffer is mapped; all audio/history/temp/PCM blocks are 128-byte aligned. Snapshot the complete graph into owned descriptors before reserving batch words.

- [ ] **Step 4: Write failing exact-size and v2-tail-builder tests**

```rust
#[test]
fn convolution_builders_reserve_the_public_macro_size_and_preserve_v2_fields() {
    let input_channels = 2_usize;
    let ir_channels = 4_usize;
    let expected = fft_words(input_channels, FFT_ZERO_SECOND_HALF)
        + fft_words(0, FFT_ZERO_SECOND_HALF)
        + 1 + input_channels * pad_count(ir_channels, 8) * 85;
    let built = build_shared_input_v2_fixture(input_channels, ir_channels, 0.25, 0.75);
    assert_eq!(built.used_words(), expected);
    assert_eq!(built.decoded.outputs[0].previous_gain, 0.25);
    assert_eq!(built.decoded.outputs[0].gain, 0.75);
}
```

- [ ] **Step 5: Run builder tests and verify RED**

Run: `cargo test -p remu-runtime acm_engine::dsp::convolution_tests::convolution_builders_reserve_the_public_macro_size_and_preserve_v2_fields -- --test-threads=1`

Expected: imported builders are absent or do not use the SDK macro’s dynamic word count.

- [ ] **Step 6: Implement all four imported builders**

Decode SysV stack arguments exactly: v1 output pointer array from `stack_arguments[0]`; v2 output pointer array from `stack_arguments[0]` and reserved value from `stack_arguments[1]`. SharedInput snapshots one input plus `count` IR/output tuples; SharedIr snapshots one IR plus `count` input/output tuples. Reject nonzero v2 reserved values. Store the entire owned graph in the private record and zero-fill the remaining ABI-sized command words.

- [ ] **Step 7: Verify Task 7 GREEN**

Run: `cargo test -p remu-runtime acm_engine::dsp::convolution_tests::convolution_struct_ acm_engine::dsp::convolution_tests::convolution_accepts_ acm_engine::dsp::convolution_tests::convolution_builders_ -- --test-threads=1`

Expected: ABI, shape, pointer, alignment, exact-size, SharedInput/SharedIr, and v2 builder tests pass.

- [ ] **Step 8: Commit only Task 7 files**

```bash
git add crates/remu-runtime/src/acm_engine/dsp/mod.rs crates/remu-runtime/src/acm_engine/dsp/convolution.rs crates/remu-runtime/src/acm_engine/dsp/convolution_tests.rs crates/remu-runtime/src/acm_engine/abi.rs crates/remu-runtime/src/acm_engine/builders.rs crates/remu-runtime/src/acm_engine/batch.rs crates/remu-runtime/src/acm_engine/mod.rs
git commit -m "feat: build ACM convolution jobs"
```

---

### Task 8: Execute partitioned convolution, history, tails, and crossfades

**Files:**
- Modify: `crates/remu-runtime/src/acm_engine/dsp/convolution.rs`
- Modify: `crates/remu-runtime/src/acm_engine/dsp/convolution_tests.rs`
- Modify: `crates/remu-runtime/src/acm_engine/batch.rs`

**Interfaces:**
- Produces: `convolution::execute(job: &ConvolutionJob, memory: &ExecutionMemory) -> Result<Vec<PendingWrite>, AcmError>`.
- Produces: history-position/count/null-count and previous-gain state updates as staged writes.
- Consumes: packed frequency-domain input/IR blocks and inverse transforms from Task 5.

- [ ] **Step 1: Write failing direct-reference convolution tests**

```rust
#[test]
fn partitioned_convolution_matches_direct_time_domain_reference() {
    for block_size in [256_usize, 512, 1024] {
        let input = seeded_signal(block_size * 3, 0x1357_2468);
        let impulse = seeded_signal(block_size * 2 + 17, 0x2468_1357);
        let expected = direct_convolution(&input, &impulse);
        let actual = run_partitioned_stream(block_size, &input, &impulse, AcmFormat::Fp32);
        assert_samples_close(&actual[..expected.len()], &expected, 8.0e-4);
    }
}
```

- [ ] **Step 2: Run numerical convolution tests and verify RED**

Run: `cargo test -p remu-runtime acm_engine::dsp::convolution_tests::partitioned_convolution_matches_direct_time_domain_reference -- --test-threads=1`

Expected: execution is absent and produces no output.

- [ ] **Step 3: Implement frequency-domain history and overlap-add**

FFT each new zero-extended PCM block, write its packed spectrum into the next input-history slot, multiply every available history partition by the matching IR partition and optional block/tuple gain, sum in `Complex32`, inverse transform, add the saved temp half, publish the first block, and save the second block as the next temp. Respect per-IR `iSize` by zeroing compressed high-frequency bins beyond the runtime byte count. Advance history position modulo capacity and saturate history count at capacity only after all outputs succeed.

- [ ] **Step 4: Write failing mapping, fp16, wraparound, and v2 tail tests**

```rust
#[test]
fn v2_tail_crossfade_and_initial_mix_follow_reference_state() {
    let mut fixture = ConvFixture::stereo_v2(256);
    fixture.output.previous_gain = 0.25;
    fixture.output.gain = 0.75;
    fixture.output.initial_pcm = Some(vec![0.5; 256]);
    let live = fixture.execute_live().expect("live block");
    let tail = fixture.execute_null().expect("tail block");
    assert_ramped_mix(&live, 0.25, 0.75, 0.5);
    assert_eq!(fixture.input.history_null_count, 1);
    assert!(tail.iter().any(|sample| sample.abs() > 1.0e-6));
    assert_eq!(fixture.execute_live(), Err(AcmError::InvalidParameter));
}
```

- [ ] **Step 5: Run state/mapping tests and verify RED**

Run: `cargo test -p remu-runtime acm_engine::dsp::convolution_tests::v2_tail_crossfade_and_initial_mix_follow_reference_state acm_engine::dsp::convolution_tests::channel_mappings_and_fp16_ir_match_reference acm_engine::dsp::convolution_tests::history_wraparound_preserves_partition_order -- --test-threads=1`

Expected: tail counters, gains, mapping, fp16 conversion, or history order differ from the independent reference.

- [ ] **Step 6: Complete SharedInput/SharedIr mappings and v2 state rules**

Map one input across N IR channels, N input channels across one IR, or equal N:N channels. SharedInput applies each tuple gain to its IR result; SharedIr applies it to its input result. Version 2 counts consecutive null PCM blocks, forbids a non-null PCM after null-tail mode begins, linearly ramps `fGainPrev` to `fGain` across the output block, optionally adds `pPcmInit`, and stages `fGainPrev = fGain` for the next call.

- [ ] **Step 7: Verify Task 8 GREEN**

Run: `cargo test -p remu-runtime acm_engine::dsp::convolution_tests:: -- --test-threads=1`

Expected: direct numerical references, three block sizes, fp16/fp32 IRs, all mappings, gains, compression sizes, overlap, history wrapping, null tails, crossfades, and initial mixing pass.

- [ ] **Step 8: Commit only Task 8 files**

```bash
git add crates/remu-runtime/src/acm_engine/dsp/convolution.rs crates/remu-runtime/src/acm_engine/dsp/convolution_tests.rs crates/remu-runtime/src/acm_engine/batch.rs
git commit -m "feat: execute ACM convolution reverb"
```

---

### Task 9: Verify the complete engine and run one sparse Demon’s Souls gate

**Files:**
- Modify: `crates/remu-runtime/src/acm_engine/mod.rs`
- Modify: `crates/remu-runtime/src/acm_engine/tests.rs`
- Modify: `crates/remu-runtime/src/platform_independent_tests/services.rs`
- Modify: `docs/status.md`

**Interfaces:**
- Produces: bounded ACM telemetry for submitted/retired/failed batches, queue age, execution time, function counts, and sample counts.
- Produces: documented next blocker from one bounded title run.
- Consumes: all prior task interfaces.

- [ ] **Step 1: Write the final failing export and end-to-end batch tests**

```rust
#[test]
fn every_sdk10_export_executes_a_real_checked_path() {
    let inert_count = |key: &str| {
        remu_gpu::fabricated_success_counts()
            .into_iter()
            .find_map(|(candidate, count)| (candidate == key).then_some(count))
            .unwrap_or(0)
    };
    for nid in SDK10_ACM_NIDS {
        let before = inert_count(nid);
        assert_ne!(dispatch(nid, invalid_arguments_for(nid), [0; 6]), None);
        assert_eq!(inert_count(nid), before);
    }
}

#[test]
fn mixed_fft_panner_convolution_batch_retires_with_real_outputs() {
    let result = MixedBatchFixture::new().start_execute_wait();
    assert_eq!(result.wait_result, 0);
    assert!(result.fft_output.iter().any(|value| value.abs() > 1.0e-6));
    assert!(result.panner_output.iter().any(|value| value.abs() > 1.0e-6));
    assert!(result.reverb_output.iter().any(|value| value.abs() > 1.0e-6));
}
```

- [ ] **Step 2: Run the final focused tests and verify RED**

Run: `cargo test -p remu-runtime acm_engine::tests::every_sdk10_export_executes_a_real_checked_path acm_engine::tests::mixed_fft_panner_convolution_batch_retires_with_real_outputs -- --test-threads=1`

Expected: any missing route, inert telemetry, or mixed-job integration gap fails explicitly.

- [ ] **Step 3: Close integration gaps and add rate-limited telemetry**

Increment counters only on accepted work and completed DSP. Log the first validation/execution failure per `(command, error)` and power-of-two repeats. Ensure none of the 13 ACM NIDs reaches `note_inert`, the unimplemented import fallback, or the locked dispatcher.

- [ ] **Step 4: Run formatting and complete offline verification**

Run: `cargo fmt --all -- --check`

Run: `cargo test -p remu-runtime acm_engine:: -- --test-threads=1`

Run: `cargo test -p remu-runtime --lib -- --test-threads=1`

Run: `cargo check --workspace --all-targets`

Run: `cargo clippy -p remu-runtime --all-targets -- -D warnings`

Expected: every command exits 0. If an unrelated dirty-tree failure remains, record its exact test and stack without weakening ACM assertions.

- [ ] **Step 5: Build release and run one bounded 90-second title gate**

```bash
cargo build --release --target x86_64-apple-darwin
REMU_SKIP_BUILD=1 REMU_AUTO_CROSS=8 REMU_AUTO_CROSS_REPEAT_MS=3000 REMU_AUTO_CROSS_HOLD_MS=250 scripts/run-game.sh /Users/xnetcat/Games/sharpemu/PPSA01342/PPSA01342-app0/eboot.bin 90
```

Expected: the run summary contains zero unimplemented ACM imports and nonzero retired ACM batches. Record whether VideoOut/AGC activity begins; if not, name the next first failing import or guest assertion without launching again.

- [ ] **Step 6: Update status documentation with measured evidence**

Document the exact commit, artifact path, ACM call/retirement/failure counts, VideoOut/AGC boundary reached, and any next blocker. Do not claim playability or FPS from a pre-render run.

- [ ] **Step 7: Re-run the offline ACM suite after documentation-adjacent cleanup**

Run: `cargo test -p remu-runtime acm_engine:: -- --test-threads=1`

Expected: all ACM tests pass with zero failures.

- [ ] **Step 8: Commit only Task 9 files**

```bash
git add crates/remu-runtime/src/acm_engine/mod.rs crates/remu-runtime/src/acm_engine/tests.rs crates/remu-runtime/src/platform_independent_tests/services.rs docs/status.md
git commit -m "feat: complete CPU ACM engine"
```
