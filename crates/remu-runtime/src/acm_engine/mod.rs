// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

//! PlayStation 5 Audio Compute Manager service.

use std::collections::{BTreeMap, VecDeque};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, OnceLock};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use crate::{guest_output_pointer, low_u32, write_guest_u32};

use self::abi::{AcmError, SceAcmBatchInfo, read_array, read_pod, return_error};
use self::batch::{BatchPlan, Command, decode_buffers};

mod abi;
mod batch;
mod builders;

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

#[derive(Default)]
struct EngineState {
    next_context: u32,
    next_batch: u32,
    generation: u64,
    stopping: bool,
    contexts: BTreeMap<u32, ContextState>,
    batches: BTreeMap<u32, BatchState>,
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
        Command::Barrier(tag) => Some(*tag),
        Command::Statistics(_) => None,
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
    let result = execute_plan(&claim.plan);
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

fn execute_plan(plan: &BatchPlan) -> Result<(), AcmError> {
    for command in &plan.commands {
        match command {
            Command::Barrier(_) => {}
            Command::Statistics(statistics) => {
                let _ = statistics.address;
            }
        }
    }
    Ok(())
}

pub(crate) fn dispatch(nid: &str, arguments: [u64; 6], _stack_arguments: [u64; 6]) -> Option<u64> {
    Some(match nid {
        "ZIXln2K3XMk" => AcmEngine::global().context_create(arguments),
        "jBgBjAj02R8" => AcmEngine::global().context_destroy(arguments),
        "tW9W+CAG4FE" => AcmEngine::global().batch_start_buffer(arguments),
        "8fe55ktlNVo" => AcmEngine::global().batch_start_buffers(arguments),
        "RLN3gRlXJLE" => AcmEngine::global().batch_wait(arguments),
        "r7z5YQFZo+U" | "KovqaFbmtsM" | "DR-ZCmvVR9Q" | "LA4RCNKnFjg" | "u70oWo92SYQ"
        | "9nLbWmRDpa8" | "wM41Cc+ieog" | "PVx9oQUaDdk" => {
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
