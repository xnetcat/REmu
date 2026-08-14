# CPU↔GPU sync / enqueue audit (branch main, commit 1058fdd5)

## Verified context

- `upload_global_buffers` is called **twice per draw** — export evaluation and pixel evaluation — at `crates/remu-gpu-metal/src/presenter/execute.rs:963-964`. So the ~0.35 ms/call figure is ~0.7 ms of the ~2.8 ms per-draw budget. The instrumentation producing that number is `crates/remu-gpu-metal/src/presenter/cache.rs:372-385`, reported at `crates/remu-gpu-metal/src/presenter/present/window.rs:523-524`.
- The "~600 pending waits" live in the ordered receiver's per-queue `submissions` deques, **not** in `GuestMemoryWaitTracker::active`.

## 1. The entire GPU translation pipeline is one thread, and that thread is AppKit's main thread

`receive_events` (`crates/remu-gpu-metal/src/presenter/submission.rs:90`) is driven from `about_to_wait` (`crates/remu-gpu-metal/src/presenter/present/window.rs:1716-1717`) via `service_interactive_presenter_before_wait` (`window.rs:1618-1646`). Event decode dispatch, Metal encoding, texture and global-buffer uploads, batch retirement, guest-memory publication, `render()`, `update_window_title()` and `poll_host_ui()` all run on that one thread. Only shader compilation is offloaded (`crates/remu-gpu-metal/src/presenter/shader.rs:130`).

**Cost**: the per-tick budget is 1024 events, 2048 when backlogged (`crates/remu-gpu-metal/src/presenter.rs:293-295`). At ~2.8 ms/draw a single tick can hold the AppKit main thread for seconds. This is also why the 2026-08-12 RenderThread-starve wedge was invisible from the window.

**Better design**: a dedicated translation/encode thread owning `MetalPresenter`'s guest-batch state, with the AppKit thread reduced to drawable acquisition and compositing.

## 2. Backpressure-full state busy-spins the presenter thread at 100% CPU

`receive_events` returns immediately when `guest_submission_backpressure_full()` (`submission.rs:103`). That predicate (`crates/remu-gpu-metal/src/presenter/submission/batch.rs:1120-1125`) is true only when committed batches ≥ 4 **and** the open batch is at its 64-work-item capacity, so nothing can drain. But `about_to_wait` then calls `has_runnable_events()` (`crates/remu-gpu-metal/src/presenter/submission/events.rs:737-755`), which returns true purely because the queues hold un-blocked events, and sets `ControlFlow::Poll` (`window.rs:1731-1732`). The loop re-enters, does no GPU work, calls `render()` and `poll_host_ui()` again, and repeats without ever sleeping until Metal retires a batch.

**Better design**: the completion waker already exists (`batch.rs:1667-1676`, attached at `batch.rs:1237` and `batch.rs:1313`). `has_runnable_events` should answer "runnable *given current backpressure*" and return false in this state, so the loop parks on `WaitUntil`/the waker instead of spinning.

## 3. Guest completion → `sceKernelWaitEqueue` wakeup takes a process-wide write lock per waiter, per wakeup

`retire_completed_guest_batches` (`batch.rs:879`) → `complete_terminal_guest_batch` → `publish_guest_batch_completion` (`batch.rs:2041`) → `publish_guest_submission_completion` (`crates/remu-gpu-metal/src/presenter/present/scanout.rs:383`) → `remu_gpu::record_gpu_submission_completed` (`crates/remu-gpu/src/counters.rs:205-215`, global `GPU_COMPLETED_SUBMISSIONS` mutex, then `notify_gpu_completion_consumer`) → `notify_kernel_equeue_waiters` (`crates/remu-runtime/src/lib.rs:6190-6199`, installed at `lib.rs:2428`): global `KERNEL_EQUEUE_WAKE` mutex, generation bump, `notify_all`.

Every blocked waiter in `kernel_wait_equeue_blocking` (`lib.rs:6574`) wakes, and **each acquires the `PROBE_DISPATCH` write lock** (`lib.rs:6607-6610`) to run `kernel_poll_equeue`. Inside, `kernel_poll_equeue` (`lib.rs:6464`) takes the EventState read lock, calls `refresh_completed_agc_submissions` (`lib.rs:6244`) — O(pending arms) retains plus an O(all registrations) filter/collect **per completed sequence** (`lib.rs:6333-6341`, `lib.rs:6446-6457`) — then takes the EventState write lock and does a linear `position()` scan over `pending_events` per delivered event (`lib.rs:6478-6484`).

**Better design**: per-equeue wait lists (condvar keyed by equeue handle); take `PROBE_DISPATCH` out of the poll path entirely.

## 4. `MAX_COMMITTED_GUEST_BATCHES = 4` is a *global* cap across all hardware queues

`presenter.rs:311`, with `GUEST_BATCH_WORK_CAPACITY = 64` at `presenter.rs:304`. `committed_guest_batch_count` sums across queues (`batch.rs:750-759`). A 2000-draw frame needs ≥32 commits, ≥8 CPU↔GPU round trips as a scheduling artefact; hitting the cap on `FlushPolicy::Required` (`batch.rs:1263-1281`) CPU-blocks the presenter in `settle_next_global_guest_batch`.

**Better design**: per-queue depth budget.

## 5. `wait_until_settled_within` is a yield/sleep poll loop, not a completion wait

`crates/remu-gpu-metal/src/presenter/completion.rs:54-74` spin-yields 2 ms then sleeps 250 µs up to 30 s. Reached from every capped `Required` flush, from `wait_for_guest_command_batches*` (`batch.rs:1509-1664`), and from `QueuedGlobalReadSource::Synchronize` inside `upload_global_buffers_untimed` (`cache.rs:407`) — potentially inside a draw.

**Better design**: `addCompletedHandler` signalling a `Condvar`.

## 6. `queued_global_read_source` is O(all queued writes across all live batches) per shader binding, per draw

`cache.rs:937-969` scans the open batch then **every** committed batch. Each `queued_global_read_source_in_batch` (`cache.rs:1361-1417`) linearly scans that batch's `memory_writes`, `global_buffer_versions`, `global_writebacks`, `guest_write_ranges`. `upload_global_buffers_untimed` runs it per global binding (`cache.rs:399-402`), and `execute_compiled_draw` calls that twice. Dominant term inside the measured ~0.35 ms/call.

**Better design**: an incrementally maintained address-interval index per batch or presenter-wide page-keyed index.

## 7. `unsignaled_host_dependencies` runs for every guest-queue event and allocates three collections

`submission.rs:235` before classification. `batch.rs:495-531` builds a HashMap of every queue's `host_retired_fence_value`, two retains, a Vec collect — almost always empty. 1024–2048 events/tick.

**Better design**: cached per-queue unsignaled-dependency count, invalidated on fence retire / dependency add.

## 8. Producer publication costs one global mutex plus a full active-wait scan per dword

`crates/remu-gpu-metal/src/presenter/submission/memory.rs:946-958`: a `WRITE_DATA` of N dwords calls `record_guest_memory_producer` N times. Each (`memory.rs:880-889`) does a linear scan over `GuestMemoryWaitTracker::active` (`waits.rs:362-371`), then `remu_gpu::record_gpu_wait_producer` takes the process-global `GPU_WAITS` mutex (`crates/remu-gpu/src/waits.rs:214-219`) and clones the Vec of wait IDs (`waits.rs:105`). `Immediate64`/`Timestamp` releases pay three each (`memory.rs:912-923`).

**Better design**: index `active` by `(address, is_64_bit)`; batch a whole write list into one registry lock acquisition.

## 9. Pipeline prefetch rebuilds a fat pipeline key for every queued draw, on every call

`shader.rs:55-98` → `prefetch_pipeline_work` (`events.rs:645-715`): `ingest(512)` then all queued events per source through an `include` closure constructing `graphics_pipeline_key(draw)` (`decode/pipeline.rs:934`) — `[u32; 32]`, `Vec<RenderTargetPipelineState>`, nested structs — hashed into three sets plus `seen`. Called per tick (`submission.rs:112`) and **again on every Blocked poll** (`submission.rs:206`).

**Better design**: memoize the key on the queued event at ingest; per-queue first-uncompiled cursor.

## 10. Publication fairness uses one global mutex plus `notify_all` per 32-event chunk

`crates/remu-gpu/src/channel.rs:240-297`. 8 chunks (`channel.rs:28`) × 32 events (`channel.rs:31`) = 256 in flight. A 3200-event submission = 100 lock/notify_all cycles waking every other submitting guest thread.

**Better design**: per-queue condvars, or one `Arc<[GpuEvent]>` send per submission.

## 11. Dependency ingress is unbounded by design; `pending_event_count()` recomputed inside its loop

`events.rs:157-176`: `pending_capacity` becomes `usize::MAX` under dependency ingress, bounded only by budget 1024/2048 (`events.rs:18-24`) — one call can move up to 65,536 decoded events into deques. `pending_event_count()` (`events.rs:717-725`) is O(queues × submissions), evaluated per iteration (`events.rs:172`).

**Better design**: finite elevated cap; running counter.

## 12. `retire_completed_guest_batches` allocates and sorts on every call — including per draw

`batch.rs:881-882` collects + sorts queue keys, then one ObjC `status` message per queue front. Called per tick (`submission.rs:92`), per queue-local flush (`batch.rs:1260`), **per draw** (`execute.rs:341`).

**Better design**: ordered map / fixed array; atomic completion flag so the no-op case is one relaxed load.

## 13. Correctness cliff: carried completions are silently evicted

`refresh_completed_agc_submissions` re-inserts unmatched sequences into `unclaimed_agc_completions`, evicting the oldest past `MAXIMUM_CARRIED_AGC_COMPLETIONS` (`lib.rs:6307-6318`). An arm whose registration never arrives is dropped; the guest waits forever. (The check-then-wait race is handled correctly via the generation snapshot `lib.rs:6607` + re-check `lib.rs:6642-6645`.) Also `queue_kernel_event_into` (`event_state.rs:153-172`) coalesces N completions into one pending entry; guests that count events under-count (`event_state.rs:107-111`).

## 14. Smaller redundancies on the per-event path

- `receive_shader_compilations()` before the loop (`submission.rs:91`) and again per iteration (`submission.rs:121`).
- Two+ `Instant::now()` per event (`submission.rs:221`, `submission.rs:840`) plus one per pop (`events.rs:523`).
- Four wake points per logical publication (`submission.rs:535/557/572`, `memory.rs:86`), each a retain over `blocked_queues`.
- Headless sleeps a fixed 1 ms/iteration (`window.rs:1775`, `presenter.rs:279`).

## If only three things get fixed

1. Move translation off the AppKit thread (1).
2. Stop the backpressure busy-spin; completion-handler condvar for settles (2, 5).
3. Per-equeue wakeups; PROBE_DISPATCH off the poll path (3).
