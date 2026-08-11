// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

//! PlayStation 5 Audio Compute Manager service.

use std::collections::{BTreeMap, VecDeque};
use std::sync::atomic::{Ordering, fence};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, OnceLock};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use crate::{guest_output_pointer, low_u32, write_guest_bytes, write_guest_u32};

use self::abi::{AcmError, SceAcmBatchInfo, checked_span, read_array, read_pod, return_error};
use self::batch::{AcmStatistics, BatchPlan, Command, PendingWrite, decode_buffers};

mod abi;
mod batch;
mod builders;
mod dsp;

pub(crate) const SDK10_ACM_NIDS: [&str; 13] = [
    "ZIXln2K3XMk",
    "jBgBjAj02R8",
    "tW9W+CAG4FE",
    "8fe55ktlNVo",
    "RLN3gRlXJLE",
    "r7z5YQFZo+U",
    "KovqaFbmtsM",
    "DR-ZCmvVR9Q",
    "LA4RCNKnFjg",
    "u70oWo92SYQ",
    "9nLbWmRDpa8",
    "wM41Cc+ieog",
    "PVx9oQUaDdk",
];

pub(crate) fn owns_nid(nid: &str) -> bool {
    SDK10_ACM_NIDS.contains(&nid)
}

#[derive(Default)]
struct ContextState {
    queue: VecDeque<u32>,
    active_batch: Option<u32>,
    destroying: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum BatchStatus {
    Pending,
    Running,
    Retired,
    Failed(AcmError),
    Cancelled,
}

struct BatchState {
    context: u32,
    plan: BatchPlan,
    status: BatchStatus,
    waiter_active: bool,
}

#[derive(Clone, Copy)]
struct ExecutionSample {
    ended: Instant,
    duration: Duration,
}

#[derive(Default)]
struct EngineState {
    next_context: u32,
    next_batch: u32,
    generation: u64,
    stopping: bool,
    contexts: BTreeMap<u32, ContextState>,
    batches: BTreeMap<u32, BatchState>,
    history: VecDeque<ExecutionSample>,
    #[cfg(test)]
    execution_tags: Vec<u64>,
}

struct EngineCore {
    state: Mutex<EngineState>,
    changed: Condvar,
}

pub(crate) struct AcmEngine {
    core: Arc<EngineCore>,
    workers: Mutex<Vec<JoinHandle<()>>>,
    worker_count: usize,
}

static ENGINE: OnceLock<AcmEngine> = OnceLock::new();

impl AcmEngine {
    pub(crate) fn new() -> Self {
        Self {
            core: Arc::new(EngineCore {
                state: Mutex::new(EngineState::default()),
                changed: Condvar::new(),
            }),
            workers: Mutex::new(Vec::new()),
            worker_count: 0,
        }
    }

    fn with_workers(worker_count: usize) -> Self {
        let engine = Self {
            worker_count,
            ..Self::new()
        };
        engine.start_workers();
        engine
    }

    fn global() -> &'static Self {
        ENGINE.get_or_init(|| Self::with_workers(2))
    }

    fn lock(&self) -> MutexGuard<'_, EngineState> {
        self.core
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    fn context_create(&self, arguments: [u64; 6]) -> u64 {
        let output = arguments[0];
        if guest_output_pointer(output, std::mem::size_of::<u32>()).is_none() {
            return return_error(AcmError::InvalidParameter);
        }

        let mut state = self.lock();
        let context = state.next_context.max(1);
        if context == u32::MAX {
            return return_error(AcmError::OutOfResources);
        }
        state.next_context = context + 1;
        state.contexts.insert(context, ContextState::default());
        drop(state);
        write_guest_u32(output, context);
        0
    }

    fn context_destroy(&self, arguments: [u64; 6]) -> u64 {
        let context = low_u32(arguments[0]);
        let mut state = self.lock();
        let Some(context_state) = state.contexts.get_mut(&context) else {
            return return_error(AcmError::InvalidContext);
        };
        if context_state.destroying {
            return return_error(AcmError::InvalidContext);
        }
        context_state.destroying = true;
        while state
            .contexts
            .get(&context)
            .is_some_and(|context| !context.queue.is_empty() || context.active_batch.is_some())
        {
            state = self
                .core
                .changed
                .wait(state)
                .unwrap_or_else(std::sync::PoisonError::into_inner);
        }
        state.contexts.remove(&context);
        state.batches.retain(|_, batch| batch.context != context);
        0
    }

    fn batch_start_buffer(&self, arguments: [u64; 6]) -> u64 {
        let context = low_u32(arguments[0]);
        let buffer = arguments[1];
        let Ok(size) = usize::try_from(arguments[2]) else {
            return return_error(AcmError::InvalidParameter);
        };
        if buffer == 0 || size == 0 {
            return return_error(AcmError::InvalidParameter);
        }
        let info = SceAcmBatchInfo {
            buffer,
            offset: size as u64,
            size: size as u64,
        };
        self.start_plan(context, &[info], arguments[3], arguments[4])
    }

    fn batch_start_buffers(&self, arguments: [u64; 6]) -> u64 {
        const MAXIMUM_BATCH_BUFFERS: usize = 4096;

        let context = low_u32(arguments[0]);
        let count = low_u32(arguments[1]) as usize;
        if count > MAXIMUM_BATCH_BUFFERS {
            return return_error(AcmError::InvalidParameter);
        }
        let pointer_count = count.max(1);
        let info_pointers = match read_array::<u64>(arguments[2], pointer_count) {
            Ok(pointers) => pointers,
            Err(_) => return return_error(AcmError::InvalidParameter),
        };
        let mut infos = Vec::with_capacity(pointer_count);
        for address in info_pointers {
            let info = match read_pod::<SceAcmBatchInfo>(address) {
                Ok(info) => info,
                Err(_) => return return_error(AcmError::InvalidParameter),
            };
            infos.push(info);
        }
        self.start_plan(context, &infos, arguments[3], arguments[4])
    }

    fn start_plan(&self, context: u32, infos: &[SceAcmBatchInfo], error: u64, output: u64) -> u64 {
        if error != 0 || guest_output_pointer(output, std::mem::size_of::<u32>()).is_none() {
            return return_error(AcmError::InvalidParameter);
        }
        if !self
            .lock()
            .contexts
            .get(&context)
            .is_some_and(|context| !context.destroying)
        {
            return return_error(AcmError::InvalidContext);
        }
        let plan = match decode_buffers(infos) {
            Ok(plan) => plan,
            Err(error) => return return_error(error),
        };

        let mut state = self.lock();
        if !state
            .contexts
            .get(&context)
            .is_some_and(|context| !context.destroying)
        {
            return return_error(AcmError::InvalidContext);
        }
        let batch = state.next_batch.max(1);
        if batch == u32::MAX {
            return return_error(AcmError::OutOfResources);
        }
        state.next_batch = batch + 1;
        state.batches.insert(
            batch,
            BatchState {
                context,
                plan,
                status: BatchStatus::Pending,
                waiter_active: false,
            },
        );
        state
            .contexts
            .get_mut(&context)
            .expect("validated ACM context remains live under the state lock")
            .queue
            .push_back(batch);
        drop(state);
        write_guest_u32(output, batch);
        self.core.changed.notify_all();
        0
    }

    fn batch_wait(&self, arguments: [u64; 6]) -> u64 {
        let context = low_u32(arguments[0]);
        let batch = low_u32(arguments[1]);
        let timeout = low_u32(arguments[2]);
        let deadline = (timeout != u32::MAX)
            .then(|| Instant::now().checked_add(Duration::from_millis(u64::from(timeout))))
            .flatten();

        let mut state = self.lock();
        if !state.contexts.contains_key(&context) {
            return return_error(AcmError::InvalidContext);
        }
        let Some(batch_state) = state.batches.get(&batch) else {
            return return_error(AcmError::InvalidBatch);
        };
        if batch_state.context != context {
            return return_error(AcmError::InvalidBatch);
        }
        if let Some(result) = consume_terminal(&mut state, batch) {
            return result;
        }
        if timeout == 0
            || state
                .batches
                .get(&batch)
                .is_some_and(|batch| batch.waiter_active)
        {
            return return_error(AcmError::Busy);
        }
        state
            .batches
            .get_mut(&batch)
            .expect("validated batch remains live")
            .waiter_active = true;

        loop {
            let timed_out;
            if timeout == u32::MAX {
                state = self
                    .core
                    .changed
                    .wait(state)
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                timed_out = false;
            } else {
                let remaining = deadline
                    .and_then(|deadline| deadline.checked_duration_since(Instant::now()))
                    .unwrap_or_default();
                let (next, timed) = self
                    .core
                    .changed
                    .wait_timeout(state, remaining)
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                state = next;
                timed_out = timed.timed_out();
            }

            if let Some(result) = consume_terminal(&mut state, batch) {
                return result;
            }
            if timed_out {
                if let Some(batch) = state.batches.get_mut(&batch) {
                    batch.waiter_active = false;
                }
                return return_error(AcmError::Busy);
            }
            let Some(batch_state) = state.batches.get(&batch) else {
                return return_error(AcmError::InvalidBatch);
            };
            if batch_state.context != context {
                return return_error(AcmError::InvalidBatch);
            }
        }
    }

    fn execute_next(&self) -> bool {
        execute_selected(&self.core, None)
    }

    #[cfg(test)]
    fn execute_next_for_test(&self) -> bool {
        self.execute_next()
    }

    #[cfg(test)]
    fn set_waiter_active_for_test(&self, batch: u32, active: bool) {
        self.lock()
            .batches
            .get_mut(&batch)
            .expect("test batch is live")
            .waiter_active = active;
    }

    #[cfg(test)]
    fn runnable_for_test(&self) -> Vec<u32> {
        self.lock()
            .contexts
            .values()
            .filter(|context| context.active_batch.is_none())
            .filter_map(|context| context.queue.front().copied())
            .collect()
    }

    #[cfg(test)]
    fn execute_batch_for_test(&self, batch: u32) -> bool {
        execute_selected(&self.core, Some(batch))
    }

    #[cfg(test)]
    fn execution_tags_for_test(&self) -> Vec<u64> {
        self.lock().execution_tags.clone()
    }

    #[cfg(test)]
    fn context_is_destroying_for_test(&self, context: u32) -> bool {
        self.lock()
            .contexts
            .get(&context)
            .is_some_and(|context| context.destroying)
    }

    #[cfg(test)]
    fn waiter_active_for_test(&self, batch: u32) -> bool {
        self.lock()
            .batches
            .get(&batch)
            .is_some_and(|batch| batch.waiter_active)
    }

    #[cfg(test)]
    fn history_len_for_test(&self) -> usize {
        self.lock().history.len()
    }

    fn start_workers(&self) {
        let mut workers = self
            .workers
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        debug_assert!(workers.is_empty());
        for index in 0..self.worker_count {
            let core = Arc::clone(&self.core);
            workers.push(
                std::thread::Builder::new()
                    .name(format!("remu-acm-{index}"))
                    .spawn(move || worker_loop(core))
                    .expect("ACM worker thread creation must succeed"),
            );
        }
    }

    fn reset(&self) {
        let generation;
        let mut state = self.lock();
        state.stopping = true;
        generation = state.generation.wrapping_add(1);
        state.generation = generation;
        for batch in state.batches.values_mut() {
            batch.status = BatchStatus::Cancelled;
        }
        for context in state.contexts.values_mut() {
            context.queue.clear();
            context.active_batch = None;
        }
        drop(state);
        self.core.changed.notify_all();

        let workers = std::mem::take(
            &mut *self
                .workers
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner),
        );
        for worker in workers {
            let _ = worker.join();
        }

        let mut state = self.lock();
        *state = EngineState {
            generation,
            ..EngineState::default()
        };
        drop(state);
        self.core.changed.notify_all();
        self.start_workers();
    }
}

struct ExecutionClaim {
    context: u32,
    batch: u32,
    generation: u64,
    plan: BatchPlan,
    tag: Option<u64>,
}

fn worker_loop(core: Arc<EngineCore>) {
    loop {
        let claim = {
            let mut state = core
                .state
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            loop {
                if state.stopping {
                    return;
                }
                if let Some(claim) = claim_batch(&mut state, None) {
                    break claim;
                }
                state = core
                    .changed
                    .wait(state)
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
            }
        };
        complete_claim(&core, claim);
    }
}

fn execute_selected(core: &EngineCore, requested: Option<u32>) -> bool {
    let claim = {
        let mut state = core
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let Some(claim) = claim_batch(&mut state, requested) else {
            return false;
        };
        claim
    };
    complete_claim(core, claim);
    true
}

fn claim_batch(state: &mut EngineState, requested: Option<u32>) -> Option<ExecutionClaim> {
    let context = if let Some(batch) = requested {
        let context = state.batches.get(&batch)?.context;
        let state = state.contexts.get(&context)?;
        (state.active_batch.is_none() && state.queue.front() == Some(&batch)).then_some(context)?
    } else {
        state.contexts.iter().find_map(|(&context, state)| {
            (state.active_batch.is_none() && !state.queue.is_empty()).then_some(context)
        })?
    };
    let batch = state.contexts.get_mut(&context)?.queue.pop_front()?;
    state.contexts.get_mut(&context)?.active_batch = Some(batch);
    let batch_state = state.batches.get_mut(&batch)?;
    batch_state.status = BatchStatus::Running;
    let plan = batch_state.plan.clone();
    let tag = plan.commands.iter().find_map(|command| match command {
        Command::Barrier { test_tag } => *test_tag,
        _ => None,
    });
    Some(ExecutionClaim {
        context,
        batch,
        generation: state.generation,
        plan,
        tag,
    })
}

fn complete_claim(core: &EngineCore, claim: ExecutionClaim) {
    let history = core
        .state
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .history
        .iter()
        .copied()
        .collect::<Vec<_>>();
    let started = Instant::now();
    let result = execute_plan_with_elapsed(&claim.plan, None, started, &history);
    let elapsed = started.elapsed();
    let mut state = core
        .state
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if state.generation == claim.generation {
        if let Some(batch) = state.batches.get_mut(&claim.batch) {
            batch.status = result.map_or_else(BatchStatus::Failed, |()| BatchStatus::Retired);
        }
        if let Some(context) = state.contexts.get_mut(&claim.context)
            && context.active_batch == Some(claim.batch)
        {
            context.active_batch = None;
        }
        if result.is_ok() {
            state.history.push_back(ExecutionSample {
                ended: Instant::now(),
                duration: elapsed,
            });
            while state.history.len() > 256 {
                state.history.pop_front();
            }
        }
        #[cfg(test)]
        if let Some(tag) = claim.tag {
            state.execution_tags.push(tag);
        }
    }
    drop(state);
    core.changed.notify_all();
}

fn consume_terminal(state: &mut EngineState, batch: u32) -> Option<u64> {
    let status = state.batches.get(&batch)?.status;
    let result = match status {
        BatchStatus::Retired => 0,
        BatchStatus::Failed(error) => return_error(error),
        BatchStatus::Cancelled => return_error(AcmError::InvalidBatch),
        BatchStatus::Pending | BatchStatus::Running => return None,
    };
    state.batches.remove(&batch);
    Some(result)
}

fn execute_plan_with_elapsed(
    plan: &BatchPlan,
    elapsed_override: Option<Duration>,
    started: Instant,
    history: &[ExecutionSample],
) -> Result<(), AcmError> {
    let mut pending = Vec::<PendingWrite>::new();
    for command in &plan.commands {
        match command {
            Command::Activate
            | Command::Barrier { .. }
            | Command::Transfer(_)
            | Command::Run
            | Command::Inline { .. }
            | Command::Queue
            | Command::Delay => {}
            Command::Statistics(statistics) => {
                let mut value = read_pod::<AcmStatistics>(statistics.address)?;
                let elapsed = elapsed_override.unwrap_or_else(|| started.elapsed());
                value.usage_batch = scaled_seconds(elapsed);
                for interval in &mut value.usage_intervals {
                    if *interval == 0 || *interval == u32::MAX {
                        *interval = if *interval == 0 {
                            interval_usage(elapsed, *interval, history, started + elapsed)
                        } else {
                            u32::MAX
                        };
                    } else {
                        *interval = interval_usage(elapsed, *interval, history, started + elapsed);
                    }
                }
                pending.push(PendingWrite {
                    address: statistics.address,
                    bytes: statistics_bytes(value).to_vec(),
                });
            }
            Command::Fft(job) => pending.extend(dsp::execute_fft(job)?),
            Command::Panner(job) => pending.extend(dsp::execute_panner(job)?),
            #[cfg(test)]
            Command::TestWrite(write) => pending.push(write.clone()),
            #[cfg(test)]
            Command::TestFault => return Err(AcmError::PageFault),
        }
    }

    validate_pending_writes(&pending)?;
    for write in &pending {
        write_guest_bytes(write.address, &write.bytes);
    }
    fence(Ordering::Release);
    if let Some(notification) = plan.notification
        && remu_memory::guest_shared_mapping_identity(notification.address, 128).is_some()
    {
        write_guest_bytes(notification.address, &[notification.value]);
    }
    Ok(())
}

fn validate_pending_writes(writes: &[PendingWrite]) -> Result<(), AcmError> {
    for (index, write) in writes.iter().enumerate() {
        checked_span(write.address, write.bytes.len(), 1)?;
        let end = write
            .address
            .checked_add(write.bytes.len() as u64)
            .ok_or(AcmError::InvalidParameter)?;
        for previous in &writes[..index] {
            let previous_end = previous
                .address
                .checked_add(previous.bytes.len() as u64)
                .ok_or(AcmError::InvalidParameter)?;
            let overlaps = write.address < previous_end && previous.address < end;
            if overlaps
                && (write.address != previous.address
                    || write.bytes.len() != previous.bytes.len()
                    || write.bytes != previous.bytes)
            {
                return Err(AcmError::InvalidParameter);
            }
        }
    }
    Ok(())
}

fn scaled_seconds(duration: Duration) -> u32 {
    const SCALE: u128 = 1_u128 << 31;
    let scaled = duration.as_nanos().saturating_mul(SCALE) / 1_000_000_000;
    u32::try_from(scaled.max(u128::from(!duration.is_zero()))).unwrap_or(u32::MAX)
}

fn interval_usage(
    current: Duration,
    interval: u32,
    history: &[ExecutionSample],
    ended: Instant,
) -> u32 {
    const SCALE: u128 = 1_u128 << 31;
    if interval == 0 {
        let Some(first) = history.first() else {
            return u32::MAX;
        };
        let start = first
            .ended
            .checked_sub(first.duration)
            .unwrap_or(first.ended);
        let window = ended.saturating_duration_since(start);
        return scaled_usage(current, history, start, ended, window);
    }
    let window_nanos = u128::from(interval).saturating_mul(1_000_000_000) / SCALE;
    let window = Duration::from_nanos(u64::try_from(window_nanos.max(1)).unwrap_or(u64::MAX));
    let start = ended.checked_sub(window).unwrap_or(ended);
    scaled_usage(current, history, start, ended, window)
}

fn scaled_usage(
    current: Duration,
    history: &[ExecutionSample],
    start: Instant,
    ended: Instant,
    window: Duration,
) -> u32 {
    const SCALE: u128 = 1_u128 << 31;
    let mut used = current.min(window).as_nanos();
    for sample in history {
        let sample_start = sample
            .ended
            .checked_sub(sample.duration)
            .unwrap_or(sample.ended);
        let overlap_start = sample_start.max(start);
        let overlap_end = sample.ended.min(ended);
        if overlap_end > overlap_start {
            used = used.saturating_add(overlap_end.duration_since(overlap_start).as_nanos());
        }
    }
    let usage = used.saturating_mul(SCALE) / window.as_nanos().max(1);
    u32::try_from(usage.min(SCALE)).unwrap_or(u32::MAX)
}

fn statistics_bytes(statistics: AcmStatistics) -> [u8; 24] {
    let mut bytes = [0_u8; 24];
    for (index, value) in [
        statistics.flags,
        statistics.usage_batch,
        statistics.usage_intervals[0],
        statistics.usage_intervals[1],
        statistics.usage_intervals[2],
        statistics.pad,
    ]
    .into_iter()
    .enumerate()
    {
        bytes[index * 4..index * 4 + 4].copy_from_slice(&value.to_le_bytes());
    }
    bytes
}

#[cfg(test)]
fn execute_plan_with_elapsed_for_test(plan: &BatchPlan, elapsed: Duration) -> Result<(), AcmError> {
    execute_plan_with_elapsed(plan, Some(elapsed), Instant::now(), &[])
}

pub(crate) fn dispatch(nid: &str, arguments: [u64; 6], stack_arguments: [u64; 6]) -> Option<u64> {
    Some(match nid {
        "ZIXln2K3XMk" => AcmEngine::global().context_create(arguments),
        "jBgBjAj02R8" => AcmEngine::global().context_destroy(arguments),
        "tW9W+CAG4FE" => AcmEngine::global().batch_start_buffer(arguments),
        "8fe55ktlNVo" => AcmEngine::global().batch_start_buffers(arguments),
        "RLN3gRlXJLE" => AcmEngine::global().batch_wait(arguments),
        "r7z5YQFZo+U" => builders::batch_job_notification(arguments),
        "KovqaFbmtsM" => builders::fft(arguments, stack_arguments, false),
        "DR-ZCmvVR9Q" => builders::fft(arguments, stack_arguments, true),
        "LA4RCNKnFjg" => builders::panner(arguments, stack_arguments),
        "u70oWo92SYQ" | "9nLbWmRDpa8" | "wM41Cc+ieog" | "PVx9oQUaDdk" => {
            return Some(return_error(AcmError::InvalidParameter));
        }
        _ => return None,
    })
}

#[cfg(any(test, target_arch = "x86_64"))]
pub(crate) fn reset_for_new_process() {
    if let Some(engine) = ENGINE.get() {
        engine.reset();
    }
}

#[cfg(test)]
mod tests;
