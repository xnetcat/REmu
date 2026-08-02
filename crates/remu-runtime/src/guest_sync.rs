use crate::{
    ImportCallFrame, PthreadMutexAttrState, current_guest_thread_handle, current_guest_thread_name,
    deliver_pending_guest_exception_at_import, diagnostic_timeout_active, guest_import_trace_nanos,
    kernel_process_time, low_u32, posix_pthread_result, relative_microseconds_until,
    trace_guest_sync, try_read_process_guest_u32, try_read_process_guest_u64,
};
use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

mod condition;
mod event_flag;
mod mutex;
mod rwlock;
mod semaphore;
mod state;

#[cfg(test)]
pub(crate) use self::condition::guest_condition_wait;
pub(crate) use self::condition::{
    guest_condition_destroy, guest_condition_init, guest_condition_notify,
    guest_condition_wait_with_interrupt,
};
#[cfg(test)]
pub(crate) use self::event_flag::EVENT_FLAG_WAIT_OR;
pub(crate) use self::event_flag::guest_event_flag_wait;
pub(crate) use self::mutex::{
    guest_mutex_destroy, guest_mutex_entry, guest_mutex_init, guest_mutex_lock, guest_mutex_unlock,
    guest_pthread_once,
};
pub(crate) use self::rwlock::{
    guest_rwlock_destroy, guest_rwlock_init, guest_rwlock_lock, guest_rwlock_unlock,
};
pub(crate) use self::semaphore::guest_semaphore_wait;
#[cfg(test)]
pub(crate) use self::state::GuestConditionState;
#[cfg(any(test, target_arch = "x86_64"))]
pub(crate) use self::state::{GUEST_SYNC, GuestSyncState};
pub(crate) use self::state::{GuestEventFlagState, GuestSemaphoreState, guest_sync};

use self::event_flag::{
    guest_event_flag_cancel, guest_event_flag_clear, guest_event_flag_create,
    guest_event_flag_delete, guest_event_flag_poll, guest_event_flag_set,
};
use self::semaphore::{
    guest_semaphore_create, guest_semaphore_delete, guest_semaphore_poll, guest_semaphore_signal,
};
use self::state::{
    EVENT_FLAG_SET_SEQUENCE, EVENT_FLAG_WAIT_SEQUENCE, GuestMutexEntry, GuestMutexState,
    GuestRwlockState, MUTEX_CONTENTION_SEQUENCE, guest_mutex_shard, guest_mutex_shards,
};

static MAIN_TIMED_CONDITION_TRACE_SEQUENCE: AtomicU64 = AtomicU64::new(0);
static MAIN_CONDITION_NOTIFY_TRACE_SEQUENCE: AtomicU64 = AtomicU64::new(0);
static MAIN_TIMED_WAIT_CONDITION: AtomicU64 = AtomicU64::new(0);
static MAIN_TIMED_WAIT_RETURN: AtomicU64 = AtomicU64::new(0);
static MAIN_TIMED_WAIT_TIMEOUT_USEC: AtomicU64 = AtomicU64::new(0);
static MAIN_TIMED_WAIT_TIMEOUT_STREAK: AtomicU64 = AtomicU64::new(0);
static MAIN_TIMED_WAIT_STREAK_STARTED_NANOS: AtomicU64 = AtomicU64::new(0);

const CONDITION_LIFECYCLE_TRACE_CAPACITY: usize = 256;
const TASKGRAPH_WAIT_EVENT_CALLER_RETURN: u64 = 0x801_be2_28e;
const TASKGRAPH_WAIT_PARENT_RETURN: u64 = 0x801_be0_511;
const TASKGRAPH_WAIT_GRANDPARENT_RETURN: u64 = 0x801_01a_f94;
const TASKGRAPH_SINGLETON_POINTER: u64 = 0x807_ba3_5e0;
const TASKGRAPH_NODE_SLAB_POINTERS: u64 = 0x807_ba8_cd0;
const TASKGRAPH_RENDER_THREAD_INDEX: u64 = 3;
const TASKGRAPH_QUEUE_OBJECT_OFFSET: u64 = TASKGRAPH_RENDER_THREAD_INDEX * 0x18 + 0x8;
const TASKGRAPH_QUEUE_INDEX_STRIDE: u64 = 0x200;
const TASKGRAPH_QUEUE_OFFSET: u64 = 0x38;
const TASKGRAPH_QUEUE_INDEX_COUNT: u64 = 2;
const TASKGRAPH_QUEUE_LANES: [(u64, u64, u64); 2] = [(0, 0x48, 0x90), (1, 0x118, 0x160)];
const TASKGRAPH_QUEUE_WAIT_STATE_OFFSET: u64 = 0x1a8;
const TASKGRAPH_QUEUE_EVENT_WRAPPER_OFFSET: u64 = 0x1f8;
const TASKGRAPH_QUEUE_WAITER_BITS: u32 = 26;
const TASKGRAPH_QUEUE_WAITER_MASK: u64 = (1 << TASKGRAPH_QUEUE_WAITER_BITS) - 1;
const TASKGRAPH_EVENT_UNDERLYING_OFFSET: u64 = 0x10;
const TASKGRAPH_EVENT_WORD_OFFSET: u64 = 0x10;
const TASKGRAPH_EVENT_MUTEX_OFFSET: u64 = 0x20;
const TASKGRAPH_EVENT_CONDITION_OFFSET: u64 = 0x28;
const TASKGRAPH_NODE_ID_BITS: u32 = 26;
const TASKGRAPH_NODE_ID_MASK: u64 = (1 << TASKGRAPH_NODE_ID_BITS) - 1;
const TASKGRAPH_NODE_PAGE_BITS: u32 = 14;
const TASKGRAPH_NODE_PAGE_MASK: u64 = (1 << TASKGRAPH_NODE_PAGE_BITS) - 1;
const TASKGRAPH_NODE_STRIDE: u64 = 24;
const TASKGRAPH_CHAIN_LIMIT: usize = 64;
const TASKGRAPH_ALLOCATOR_PAGE_BYTES: u64 = 1024 * 1024;
const TASKGRAPH_TASK_ALIGNMENT: u64 = 0x100;
const TASKGRAPH_MAX_ALLOCATOR_PAGES: usize = 8;
const TASKGRAPH_MAX_ALLOCATOR_MATCHES: usize = 64;
const TASKGRAPH_FENCE_TASK_VPTR: u64 = 0x807_4b5_c50;
const MINIMUM_GUEST_ADDRESS: u64 = 0x8_0000_0000;
const MAXIMUM_GUEST_ADDRESS: u64 = 0x100_0000_0000;

#[derive(Clone, Debug, Eq, PartialEq)]
struct ConditionWaitCallTrace {
    call_sequence: u64,
    started_nanos: u64,
    thread: u64,
    thread_name: String,
    return_address: u64,
    condition: u64,
    mutex: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct ConditionWaitReturnTrace {
    call: ConditionWaitCallTrace,
    return_sequence: u64,
    finished_nanos: u64,
    result: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct ConditionNotifyCallTrace {
    call_sequence: u64,
    started_nanos: u64,
    thread: u64,
    thread_name: String,
    return_address: u64,
    condition: u64,
    broadcast: bool,
    waiters_before: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct ConditionNotifyReturnTrace {
    call: ConditionNotifyCallTrace,
    return_sequence: u64,
    finished_nanos: u64,
    waiters_after: usize,
    result: u64,
}

#[derive(Debug)]
struct ConditionLifecycleTrace {
    capacity: usize,
    next_sequence: u64,
    active_waits: BTreeMap<u64, ConditionWaitCallTrace>,
    active_notifies: BTreeMap<u64, ConditionNotifyCallTrace>,
    recent_wait_returns: VecDeque<ConditionWaitReturnTrace>,
    recent_notifies: VecDeque<ConditionNotifyReturnTrace>,
    dropped_active_waits: u64,
    dropped_active_notifies: u64,
}

impl ConditionLifecycleTrace {
    fn new(capacity: usize) -> Self {
        Self {
            capacity,
            next_sequence: 0,
            active_waits: BTreeMap::new(),
            active_notifies: BTreeMap::new(),
            recent_wait_returns: VecDeque::new(),
            recent_notifies: VecDeque::new(),
            dropped_active_waits: 0,
            dropped_active_notifies: 0,
        }
    }

    fn next_sequence(&mut self) -> u64 {
        self.next_sequence = self.next_sequence.wrapping_add(1).max(1);
        self.next_sequence
    }

    fn begin_wait(
        &mut self,
        started_nanos: u64,
        thread: u64,
        thread_name: String,
        return_address: u64,
        condition: u64,
        mutex: u64,
    ) -> Option<ConditionWaitCallTrace> {
        if self.capacity == 0 || self.active_waits.len() >= self.capacity {
            self.dropped_active_waits = self.dropped_active_waits.saturating_add(1);
            return None;
        }
        let call = ConditionWaitCallTrace {
            call_sequence: self.next_sequence(),
            started_nanos,
            thread,
            thread_name,
            return_address,
            condition,
            mutex,
        };
        self.active_waits.insert(call.call_sequence, call.clone());
        Some(call)
    }

    fn finish_wait(&mut self, call: ConditionWaitCallTrace, finished_nanos: u64, result: u64) {
        if self.active_waits.remove(&call.call_sequence).is_none() {
            return;
        }
        let return_sequence = self.next_sequence();
        self.recent_wait_returns.retain(|previous| {
            previous.call.thread != call.thread || previous.call.condition != call.condition
        });
        self.recent_wait_returns
            .push_back(ConditionWaitReturnTrace {
                call,
                return_sequence,
                finished_nanos,
                result,
            });
        while self.recent_wait_returns.len() > self.capacity {
            self.recent_wait_returns.pop_front();
        }
    }

    fn has_active_wait(&self, condition: u64) -> bool {
        self.active_waits
            .values()
            .any(|wait| wait.condition == condition)
    }

    fn begin_notify(
        &mut self,
        started_nanos: u64,
        thread: u64,
        thread_name: String,
        return_address: u64,
        condition: u64,
        broadcast: bool,
        waiters_before: usize,
    ) -> Option<ConditionNotifyCallTrace> {
        if self.capacity == 0 || self.active_notifies.len() >= self.capacity {
            self.dropped_active_notifies = self.dropped_active_notifies.saturating_add(1);
            return None;
        }
        let call = ConditionNotifyCallTrace {
            call_sequence: self.next_sequence(),
            started_nanos,
            thread,
            thread_name,
            return_address,
            condition,
            broadcast,
            waiters_before,
        };
        self.active_notifies
            .insert(call.call_sequence, call.clone());
        Some(call)
    }

    fn finish_notify(
        &mut self,
        call: ConditionNotifyCallTrace,
        finished_nanos: u64,
        waiters_after: usize,
        result: u64,
    ) {
        if self.active_notifies.remove(&call.call_sequence).is_none() {
            return;
        }
        let return_sequence = self.next_sequence();
        self.recent_notifies
            .retain(|previous| previous.call.condition != call.condition);
        self.recent_notifies.push_back(ConditionNotifyReturnTrace {
            call,
            return_sequence,
            finished_nanos,
            waiters_after,
            result,
        });
        while self.recent_notifies.len() > self.capacity {
            self.recent_notifies.pop_front();
        }
    }

    fn last_wait_return(&self, thread: u64, condition: u64) -> Option<&ConditionWaitReturnTrace> {
        self.recent_wait_returns
            .iter()
            .rev()
            .find(|wait| wait.call.thread == thread && wait.call.condition == condition)
    }

    fn last_notify(&self, condition: u64) -> Option<&ConditionNotifyReturnTrace> {
        self.recent_notifies
            .iter()
            .rev()
            .find(|notify| notify.call.condition == condition)
    }
}

fn condition_lifecycle_trace() -> &'static Mutex<ConditionLifecycleTrace> {
    static TRACE: OnceLock<Mutex<ConditionLifecycleTrace>> = OnceLock::new();
    TRACE.get_or_init(|| {
        Mutex::new(ConditionLifecycleTrace::new(
            CONDITION_LIFECYCLE_TRACE_CAPACITY,
        ))
    })
}

fn begin_condition_wait_observation(
    frame: Option<&ImportCallFrame>,
    condition: u64,
    mutex: u64,
    timeout_usec: Option<u64>,
) -> Option<ConditionWaitCallTrace> {
    if timeout_usec.is_some() || !diagnostic_timeout_active() {
        return None;
    }
    condition_lifecycle_trace()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .begin_wait(
            guest_import_trace_nanos(),
            current_guest_thread_handle(),
            current_guest_thread_name(),
            frame.map_or(0, |frame| frame.guest_return_address),
            condition,
            mutex,
        )
}

fn finish_condition_wait_observation(observation: Option<ConditionWaitCallTrace>, result: u64) {
    let Some(observation) = observation else {
        return;
    };
    condition_lifecycle_trace()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .finish_wait(observation, guest_import_trace_nanos(), result);
}

fn begin_condition_notify_observation(
    frame: Option<&ImportCallFrame>,
    condition: u64,
    broadcast: bool,
) -> Option<ConditionNotifyCallTrace> {
    if !diagnostic_timeout_active()
        || !condition_lifecycle_trace()
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .has_active_wait(condition)
    {
        return None;
    }
    let waiters_before = condition_trace_stats(condition).0;
    condition_lifecycle_trace()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .begin_notify(
            guest_import_trace_nanos(),
            current_guest_thread_handle(),
            current_guest_thread_name(),
            frame.map_or(0, |frame| frame.guest_return_address),
            condition,
            broadcast,
            waiters_before,
        )
}

fn finish_condition_notify_observation(observation: Option<ConditionNotifyCallTrace>, result: u64) {
    let Some(observation) = observation else {
        return;
    };
    let waiters_after = condition_trace_stats(observation.condition).0;
    condition_lifecycle_trace()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .finish_notify(
            observation,
            guest_import_trace_nanos(),
            waiters_after,
            result,
        );
}

#[cfg_attr(not(target_arch = "x86_64"), allow(dead_code))]
pub(crate) fn report_condition_stall_attribution() {
    let now = guest_import_trace_nanos();
    let trace = condition_lifecycle_trace()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    eprintln!(
        "[REMu][STALL-COND] active_waits={} active_notifies={} recent_returns={} recent_notifies={} dropped_waits={} dropped_notifies={}",
        trace.active_waits.len(),
        trace.active_notifies.len(),
        trace.recent_wait_returns.len(),
        trace.recent_notifies.len(),
        trace.dropped_active_waits,
        trace.dropped_active_notifies,
    );
    for wait in trace.active_waits.values() {
        let age_millis = now.saturating_sub(wait.started_nanos) / 1_000_000;
        let last_return = trace.last_wait_return(wait.thread, wait.condition);
        let last_notify = trace.last_notify(wait.condition);
        eprintln!(
            "[REMu][STALL-COND] active_wait call={} age_ms={age_millis} thread={:?}/{:#x} return={:#x} condition={:#x} mutex={:#x} last_return={:?} last_notify={:?}",
            wait.call_sequence,
            wait.thread_name,
            wait.thread,
            wait.return_address,
            wait.condition,
            wait.mutex,
            last_return.map(|last| (
                last.call.call_sequence,
                last.return_sequence,
                now.saturating_sub(last.finished_nanos) / 1_000_000,
                last.result,
            )),
            last_notify.map(|last| (
                last.call.call_sequence,
                last.return_sequence,
                now.saturating_sub(last.finished_nanos) / 1_000_000,
                last.call.thread_name.as_str(),
                last.call.thread,
                last.call.return_address,
                last.call.broadcast,
                last.call.waiters_before,
                last.waiters_after,
                last.result,
            )),
        );
    }
    for notify in trace.active_notifies.values() {
        eprintln!(
            "[REMu][STALL-COND] active_notify call={} age_ms={} thread={:?}/{:#x} return={:#x} condition={:#x} broadcast={} waiters_before={}",
            notify.call_sequence,
            now.saturating_sub(notify.started_nanos) / 1_000_000,
            notify.thread_name,
            notify.thread,
            notify.return_address,
            notify.condition,
            notify.broadcast,
            notify.waiters_before,
        );
    }
}

#[cfg_attr(not(target_arch = "x86_64"), allow(dead_code))]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct MainTimedWaitSnapshot {
    pub(crate) condition: u64,
    pub(crate) return_address: u64,
    pub(crate) timeout_usec: u64,
    pub(crate) consecutive_timeouts: u64,
    pub(crate) age_millis: u64,
}

#[derive(Clone, Copy)]
struct MainTimedWaitObservation {
    condition: u64,
    return_address: u64,
    timeout_usec: u64,
}

fn begin_main_timed_wait_observation(
    frame: Option<&ImportCallFrame>,
    condition: u64,
    timeout_usec: u64,
) -> Option<MainTimedWaitObservation> {
    if !diagnostic_timeout_active() {
        return None;
    }
    let thread_name = current_guest_thread_name();
    if thread_name != "main" && !thread_name.contains("GameThread") {
        return None;
    }
    Some(MainTimedWaitObservation {
        condition,
        return_address: frame.map_or(0, |frame| frame.guest_return_address),
        timeout_usec,
    })
}

fn finish_main_timed_wait_observation(observation: Option<MainTimedWaitObservation>, result: u64) {
    let Some(observation) = observation else {
        return;
    };
    if low_u32(result) != libc::ETIMEDOUT.cast_unsigned() {
        MAIN_TIMED_WAIT_TIMEOUT_STREAK.store(0, Ordering::Release);
        MAIN_TIMED_WAIT_STREAK_STARTED_NANOS.store(0, Ordering::Release);
        return;
    }
    let previous_condition =
        MAIN_TIMED_WAIT_CONDITION.swap(observation.condition, Ordering::AcqRel);
    let previous_return = MAIN_TIMED_WAIT_RETURN.swap(observation.return_address, Ordering::AcqRel);
    MAIN_TIMED_WAIT_TIMEOUT_USEC.store(observation.timeout_usec, Ordering::Release);
    if previous_condition != observation.condition
        || previous_return != observation.return_address
        || MAIN_TIMED_WAIT_TIMEOUT_STREAK.load(Ordering::Acquire) == 0
    {
        MAIN_TIMED_WAIT_STREAK_STARTED_NANOS.store(guest_import_trace_nanos(), Ordering::Release);
        MAIN_TIMED_WAIT_TIMEOUT_STREAK.store(1, Ordering::Release);
    } else {
        MAIN_TIMED_WAIT_TIMEOUT_STREAK.fetch_add(1, Ordering::AcqRel);
    }
}

#[cfg_attr(not(target_arch = "x86_64"), allow(dead_code))]
pub(crate) fn main_timed_wait_snapshot() -> Option<MainTimedWaitSnapshot> {
    let consecutive_timeouts = MAIN_TIMED_WAIT_TIMEOUT_STREAK.load(Ordering::Acquire);
    if consecutive_timeouts == 0 {
        return None;
    }
    let started = MAIN_TIMED_WAIT_STREAK_STARTED_NANOS.load(Ordering::Acquire);
    Some(MainTimedWaitSnapshot {
        condition: MAIN_TIMED_WAIT_CONDITION.load(Ordering::Acquire),
        return_address: MAIN_TIMED_WAIT_RETURN.load(Ordering::Acquire),
        timeout_usec: MAIN_TIMED_WAIT_TIMEOUT_USEC.load(Ordering::Acquire),
        consecutive_timeouts,
        age_millis: guest_import_trace_nanos().saturating_sub(started) / 1_000_000,
    })
}

fn trace_read_guest_u32(address: u64) -> u32 {
    try_read_process_guest_u32(address).unwrap_or(0)
}

fn trace_read_guest_u64(address: u64) -> u64 {
    try_read_process_guest_u64(address).unwrap_or(0)
}

fn trace_taskgraph_wait_snapshot() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var_os("REMU_TRACE_TASKGRAPH_WAIT").is_some_and(|value| value == "1")
    })
}

fn plausible_guest_address(address: u64) -> bool {
    (MINIMUM_GUEST_ADDRESS..MAXIMUM_GUEST_ADDRESS).contains(&address)
}

fn plausible_guest_pointer(address: u64) -> bool {
    plausible_guest_address(address) && address.is_multiple_of(8)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct TaskGraphQueueNode {
    index: usize,
    handle: u64,
    address: u64,
    next_handle: Option<u64>,
    task: Option<u64>,
    event_match: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct TaskGraphLaneSnapshot {
    queue_index: u64,
    lane: u64,
    head: Option<u64>,
    tail: Option<u64>,
    head_task: Option<u64>,
    tail_task: Option<u64>,
    termination: &'static str,
    classification: &'static str,
    nodes: Vec<TaskGraphQueueNode>,
}

impl TaskGraphLaneSnapshot {
    fn event_match_indexes(&self) -> Vec<usize> {
        self.nodes
            .iter()
            .filter_map(|node| node.event_match.then_some(node.index))
            .collect()
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct TaskGraphAllocatorMatch {
    task: u64,
    words: [Option<u64>; 8],
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct TaskGraphQueueSyncSnapshot {
    wait_state: Option<u64>,
    waiters: Option<u64>,
    generation: Option<u64>,
    event_wrapper: Option<u64>,
    event_underlying: Option<u64>,
    event_word: Option<u64>,
    event_mutex: Option<u64>,
    event_condition: Option<u64>,
}

fn taskgraph_queue_sync_snapshot(
    queue: u64,
    read_u64: &mut impl FnMut(u64) -> Option<u64>,
) -> TaskGraphQueueSyncSnapshot {
    let wait_state = queue
        .checked_add(TASKGRAPH_QUEUE_WAIT_STATE_OFFSET)
        .and_then(&mut *read_u64);
    let event_wrapper = queue
        .checked_add(TASKGRAPH_QUEUE_EVENT_WRAPPER_OFFSET)
        .and_then(&mut *read_u64)
        .filter(|pointer| plausible_guest_pointer(*pointer));
    let event_underlying = event_wrapper
        .and_then(|wrapper| wrapper.checked_add(TASKGRAPH_EVENT_UNDERLYING_OFFSET))
        .and_then(&mut *read_u64)
        .filter(|pointer| plausible_guest_pointer(*pointer));
    let event_word = event_underlying
        .and_then(|underlying| underlying.checked_add(TASKGRAPH_EVENT_WORD_OFFSET))
        .and_then(&mut *read_u64);
    let event_mutex = event_underlying
        .and_then(|underlying| underlying.checked_add(TASKGRAPH_EVENT_MUTEX_OFFSET))
        .filter(|pointer| plausible_guest_pointer(*pointer));
    let event_condition = event_underlying
        .and_then(|underlying| underlying.checked_add(TASKGRAPH_EVENT_CONDITION_OFFSET))
        .filter(|pointer| plausible_guest_pointer(*pointer));
    TaskGraphQueueSyncSnapshot {
        wait_state,
        waiters: wait_state.map(|state| state & TASKGRAPH_QUEUE_WAITER_MASK),
        generation: wait_state.map(|state| state >> TASKGRAPH_QUEUE_WAITER_BITS),
        event_wrapper,
        event_underlying,
        event_word,
        event_mutex,
        event_condition,
    }
}

fn resolve_taskgraph_node(
    handle: u64,
    read_u64: &mut impl FnMut(u64) -> Option<u64>,
) -> Option<u64> {
    if handle == 0 {
        return None;
    }
    let node_id = handle & TASKGRAPH_NODE_ID_MASK;
    let page = node_id >> TASKGRAPH_NODE_PAGE_BITS;
    let slot = node_id & TASKGRAPH_NODE_PAGE_MASK;
    let page_pointer_address = TASKGRAPH_NODE_SLAB_POINTERS.checked_add(page.checked_mul(8)?)?;
    let page_base = read_u64(page_pointer_address)?;
    if !plausible_guest_pointer(page_base) {
        return None;
    }
    let node = page_base.checked_add(slot.checked_mul(TASKGRAPH_NODE_STRIDE)?)?;
    plausible_guest_pointer(node).then_some(node)
}

fn read_taskgraph_node_task(
    handle: Option<u64>,
    read_u64: &mut impl FnMut(u64) -> Option<u64>,
) -> Option<u64> {
    let node = resolve_taskgraph_node(handle?, read_u64)?;
    let task = read_u64(node.checked_add(8)?)?;
    plausible_guest_pointer(task).then_some(task)
}

fn walk_taskgraph_lane(
    queue_index: u64,
    lane: u64,
    head: Option<u64>,
    tail: Option<u64>,
    event: u64,
    read_u64: &mut impl FnMut(u64) -> Option<u64>,
) -> TaskGraphLaneSnapshot {
    let head_task = read_taskgraph_node_task(head, read_u64);
    let tail_task = read_taskgraph_node_task(tail, read_u64);
    let (Some(head), Some(tail)) = (head, tail) else {
        return TaskGraphLaneSnapshot {
            queue_index,
            lane,
            head,
            tail,
            head_task,
            tail_task,
            termination: "unreadable-cursor",
            classification: "unknown",
            nodes: Vec::new(),
        };
    };
    if head == 0 {
        return TaskGraphLaneSnapshot {
            queue_index,
            lane,
            head: Some(head),
            tail: Some(tail),
            head_task,
            tail_task,
            termination: "null-head",
            classification: "absent",
            nodes: Vec::new(),
        };
    }

    let mut nodes = Vec::new();
    let mut seen_node_ids = BTreeSet::new();
    let mut handle = head;
    let mut termination = "limit";
    for index in 0..TASKGRAPH_CHAIN_LIMIT {
        let node_id = handle & TASKGRAPH_NODE_ID_MASK;
        if !seen_node_ids.insert(node_id) {
            termination = "cycle";
            break;
        }
        let Some(address) = resolve_taskgraph_node(handle, read_u64) else {
            termination = "unreadable-node";
            break;
        };
        let next_handle = read_u64(address);
        let task =
            read_u64(address.saturating_add(8)).filter(|task| plausible_guest_pointer(*task));
        let event_match =
            task.is_some_and(|task| task.checked_add(0x18).and_then(&mut *read_u64) == Some(event));
        nodes.push(TaskGraphQueueNode {
            index,
            handle,
            address,
            next_handle,
            task,
            event_match,
        });
        if handle == tail {
            termination = "tail";
            break;
        }
        let Some(next_handle) = next_handle else {
            termination = "unreadable-next";
            break;
        };
        if next_handle == 0 {
            termination = "null-before-tail";
            break;
        }
        handle = next_handle;
    }

    let has_queued_match = nodes.iter().any(|node| node.index > 0 && node.event_match);
    let has_head_match = nodes.first().is_some_and(|node| node.event_match);
    let classification = if has_queued_match {
        "queued"
    } else if has_head_match {
        "popped-head"
    } else {
        "absent"
    };
    TaskGraphLaneSnapshot {
        queue_index,
        lane,
        head: Some(head),
        tail: Some(tail),
        head_task,
        tail_task,
        termination,
        classification,
        nodes,
    }
}

fn taskgraph_allocator_pages(task_anchors: impl IntoIterator<Item = u64>) -> Vec<u64> {
    let page_mask = TASKGRAPH_ALLOCATOR_PAGE_BYTES - 1;
    let mut pages = BTreeSet::new();
    for task in task_anchors {
        if pages.len() >= TASKGRAPH_MAX_ALLOCATOR_PAGES {
            break;
        }
        if plausible_guest_pointer(task) && task.is_multiple_of(TASKGRAPH_TASK_ALIGNMENT) {
            let page = task & !page_mask;
            if page >= MINIMUM_GUEST_ADDRESS
                && page
                    .checked_add(TASKGRAPH_ALLOCATOR_PAGE_BYTES)
                    .is_some_and(|end| end <= MAXIMUM_GUEST_ADDRESS)
            {
                pages.insert(page);
            }
        }
    }
    pages.into_iter().collect()
}

fn scan_taskgraph_allocator_pages(
    pages: &[u64],
    event: u64,
    read_u64: &mut impl FnMut(u64) -> Option<u64>,
) -> (Vec<TaskGraphAllocatorMatch>, bool) {
    let mut matches = Vec::new();
    for page in pages.iter().copied().take(TASKGRAPH_MAX_ALLOCATOR_PAGES) {
        let Some(page_end) = page.checked_add(TASKGRAPH_ALLOCATOR_PAGE_BYTES) else {
            continue;
        };
        if page < MINIMUM_GUEST_ADDRESS
            || page_end > MAXIMUM_GUEST_ADDRESS
            || !page.is_multiple_of(TASKGRAPH_ALLOCATOR_PAGE_BYTES)
        {
            continue;
        }
        for offset in (0..TASKGRAPH_ALLOCATOR_PAGE_BYTES)
            .step_by(usize::try_from(TASKGRAPH_TASK_ALIGNMENT).expect("task alignment fits usize"))
        {
            let task = page + offset;
            if read_u64(task + 0x18) != Some(event) {
                continue;
            }
            let words = std::array::from_fn(|index| {
                let offset = u64::try_from(index).ok()?.checked_mul(8)?;
                read_u64(task.checked_add(offset)?)
            });
            matches.push(TaskGraphAllocatorMatch { task, words });
            if matches.len() == TASKGRAPH_MAX_ALLOCATOR_MATCHES {
                return (matches, true);
            }
        }
    }
    (matches, false)
}

fn taskgraph_wait_chain_matches(
    event_caller_return: u64,
    parent_return: u64,
    grandparent_return: u64,
) -> bool {
    event_caller_return == TASKGRAPH_WAIT_EVENT_CALLER_RETURN
        && parent_return == TASKGRAPH_WAIT_PARENT_RETURN
        && grandparent_return == TASKGRAPH_WAIT_GRANDPARENT_RETURN
}

fn optional_hex(value: Option<u64>) -> String {
    value.map_or_else(|| "unreadable".to_owned(), |value| format!("{value:#x}"))
}

fn emit_taskgraph_wait_snapshot(sequence: u64, event: u64) {
    let mut read_u64 = try_read_process_guest_u64;
    let singleton = read_u64(TASKGRAPH_SINGLETON_POINTER);
    let queue_object = singleton
        .filter(|singleton| plausible_guest_pointer(*singleton))
        .and_then(|singleton| singleton.checked_add(TASKGRAPH_QUEUE_OBJECT_OFFSET))
        .and_then(&mut read_u64)
        .filter(|queue_object| plausible_guest_pointer(*queue_object));
    eprintln!(
        "[REMu][TASKGRAPH][WAIT] sequence={sequence} event={event:#x} singleton={} render_thread={} queue_object={}",
        optional_hex(singleton),
        TASKGRAPH_RENDER_THREAD_INDEX,
        optional_hex(queue_object),
    );
    let Some(queue_object) = queue_object else {
        eprintln!(
            "[REMu][TASKGRAPH][WAIT] sequence={sequence} done reason=unreadable-render-queue"
        );
        return;
    };

    let mut lanes = Vec::new();
    for queue_index in 0..TASKGRAPH_QUEUE_INDEX_COUNT {
        let Some(queue) = queue_object
            .checked_add(queue_index * TASKGRAPH_QUEUE_INDEX_STRIDE)
            .and_then(|queue| queue.checked_add(TASKGRAPH_QUEUE_OFFSET))
        else {
            continue;
        };
        let sync = taskgraph_queue_sync_snapshot(queue, &mut read_u64);
        eprintln!(
            "[REMu][TASKGRAPH][WAIT] sequence={sequence} queue_index={queue_index} queue={queue:#x} wait_state={} waiters={} generation={} event_wrapper={} event_underlying={} event_word={} event_mutex={} event_condition={}",
            optional_hex(sync.wait_state),
            optional_hex(sync.waiters),
            optional_hex(sync.generation),
            optional_hex(sync.event_wrapper),
            optional_hex(sync.event_underlying),
            optional_hex(sync.event_word),
            optional_hex(sync.event_mutex),
            optional_hex(sync.event_condition),
        );
        for (lane, head_offset, tail_offset) in TASKGRAPH_QUEUE_LANES {
            let head = queue.checked_add(head_offset).and_then(&mut read_u64);
            let tail = queue.checked_add(tail_offset).and_then(&mut read_u64);
            lanes.push(walk_taskgraph_lane(
                queue_index,
                lane,
                head,
                tail,
                event,
                &mut read_u64,
            ));
        }
    }

    for snapshot in &lanes {
        eprintln!(
            "[REMu][TASKGRAPH][WAIT] sequence={sequence} queue_index={} lane={} head={} tail={} head_task={} tail_task={} termination={} classification={} nodes={} matches={:?}",
            snapshot.queue_index,
            snapshot.lane,
            optional_hex(snapshot.head),
            optional_hex(snapshot.tail),
            optional_hex(snapshot.head_task),
            optional_hex(snapshot.tail_task),
            snapshot.termination,
            snapshot.classification,
            snapshot.nodes.len(),
            snapshot.event_match_indexes(),
        );
        for node in &snapshot.nodes {
            eprintln!(
                "[REMu][TASKGRAPH][WAIT] sequence={sequence} node queue_index={} lane={} index={} handle={:#x} address={:#x} next={} task={} event_match={}",
                snapshot.queue_index,
                snapshot.lane,
                node.index,
                node.handle,
                node.address,
                optional_hex(node.next_handle),
                optional_hex(node.task),
                node.event_match,
            );
        }
    }

    let allocator_pages = taskgraph_allocator_pages(lanes.iter().flat_map(|snapshot| {
        [snapshot.head_task, snapshot.tail_task]
            .into_iter()
            .flatten()
    }));
    let (allocator_matches, allocator_matches_truncated) =
        scan_taskgraph_allocator_pages(&allocator_pages, event, &mut read_u64);
    eprintln!(
        "[REMu][TASKGRAPH][WAIT] sequence={sequence} allocator_pages={allocator_pages:#x?} allocator_matches={} truncated={allocator_matches_truncated}",
        allocator_matches.len(),
    );
    for allocator_match in allocator_matches {
        eprintln!(
            "[REMu][TASKGRAPH][WAIT] sequence={sequence} allocator_match task={:#x} expected_vptr={} words={:#x?}",
            allocator_match.task,
            allocator_match.words[0] == Some(TASKGRAPH_FENCE_TASK_VPTR),
            allocator_match.words,
        );
    }
    eprintln!("[REMu][TASKGRAPH][WAIT] sequence={sequence} done reason=complete");
}

fn traced_main_conditions() -> &'static Mutex<BTreeSet<u64>> {
    static CONDITIONS: OnceLock<Mutex<BTreeSet<u64>>> = OnceLock::new();
    CONDITIONS.get_or_init(|| Mutex::new(BTreeSet::new()))
}

fn trace_main_condition_notify(frame: Option<&ImportCallFrame>, condition: u64, broadcast: bool) {
    if !trace_guest_sync()
        || !traced_main_conditions()
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .contains(&condition)
    {
        return;
    }
    let sequence = MAIN_CONDITION_NOTIFY_TRACE_SEQUENCE.fetch_add(1, Ordering::Relaxed) + 1;
    let (waiters, notifies, timeouts) = condition_trace_stats(condition);
    let guest_rsp = frame.map_or(0, |frame| frame.guest_rsp);
    let stack = if guest_rsp == 0 {
        [0; 6]
    } else {
        std::array::from_fn(|index| {
            trace_read_guest_u64(
                guest_rsp.saturating_add(u64::try_from(index).unwrap_or(0).saturating_mul(8)),
            )
        })
    };
    eprintln!(
        "[REMu][COND][MAIN-NOTIFY] sequence={sequence} thread={:?}/{:#x} return={:#x} rsp={guest_rsp:#x} stack={:#x},{:#x},{:#x},{:#x},{:#x},{:#x} condition={condition:#x} broadcast={broadcast} waiters={waiters} notifies_before={notifies} timeouts={timeouts}",
        current_guest_thread_name(),
        current_guest_thread_handle(),
        frame.map_or(0, |frame| frame.guest_return_address),
        stack[0],
        stack[1],
        stack[2],
        stack[3],
        stack[4],
        stack[5],
    );
}

#[derive(Clone, Copy)]
struct TimedConditionTrace {
    sequence: u64,
    started: Instant,
    condition: u64,
    waiters_before: usize,
    notifies_before: u64,
    timeouts_before: u64,
}

fn condition_trace_stats(condition: u64) -> (usize, u64, u64) {
    let sync = guest_sync();
    let state = sync
        .0
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    state
        .conditions
        .get(&condition)
        .map_or((0, 0, 0), |condition| {
            (
                condition.waiters.len(),
                condition.notifies,
                condition.timeouts,
            )
        })
}

fn begin_main_timed_condition_trace(
    frame: Option<&ImportCallFrame>,
    condition: u64,
    mutex: u64,
    deadline: u64,
    timeout_usec: u64,
) -> Option<TimedConditionTrace> {
    if !trace_guest_sync() {
        return None;
    }
    let thread_name = current_guest_thread_name();
    if thread_name != "main" && !thread_name.contains("GameThread") {
        return None;
    }
    traced_main_conditions()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .insert(condition);
    let sequence = MAIN_TIMED_CONDITION_TRACE_SEQUENCE.fetch_add(1, Ordering::Relaxed) + 1;
    if sequence > 32 && !sequence.is_power_of_two() && timeout_usec < 10_000 {
        return None;
    }
    let (deadline_seconds, deadline_nanoseconds) = if deadline == 0 {
        (0, 0)
    } else {
        (
            i64::from_ne_bytes(trace_read_guest_u64(deadline).to_ne_bytes()),
            i64::from_ne_bytes(trace_read_guest_u64(deadline + 8).to_ne_bytes()),
        )
    };
    let guest_rsp = frame.map_or(0, |frame| frame.guest_rsp);
    // This trace targets the title's FEvent::Wait implementation at return 0x801210878. Its
    // fixed prologue keeps the caller's frame pointer and return address at +0x88/+0x90 from the
    // import stack. Following the preserved frame pointer identifies the game-loop caller above
    // the generic event wrapper without changing guest synchronization behavior.
    let (event_caller_rbp, event_caller_return) = if guest_rsp == 0 {
        (0, 0)
    } else {
        (
            trace_read_guest_u64(guest_rsp.saturating_add(0x88)),
            trace_read_guest_u64(guest_rsp.saturating_add(0x90)),
        )
    };
    let (parent_rbp, parent_return) = if event_caller_rbp == 0 {
        (0, 0)
    } else {
        (
            trace_read_guest_u64(event_caller_rbp),
            trace_read_guest_u64(event_caller_rbp.saturating_add(8)),
        )
    };
    let (grandparent_rbp, grandparent_return) = if parent_rbp == 0 {
        (0, 0)
    } else {
        (
            trace_read_guest_u64(parent_rbp),
            trace_read_guest_u64(parent_rbp.saturating_add(8)),
        )
    };
    let (great_grandparent_rbp, great_grandparent_return) = if grandparent_rbp == 0 {
        (0, 0)
    } else {
        (
            trace_read_guest_u64(grandparent_rbp),
            trace_read_guest_u64(grandparent_rbp.saturating_add(8)),
        )
    };
    let (clock_start_bits, clock_target_bits) = if event_caller_rbp < 0x78 {
        (0, 0)
    } else {
        (
            trace_read_guest_u64(event_caller_rbp - 0x78),
            trace_read_guest_u64(event_caller_rbp - 0x70),
        )
    };
    let limiter_wait_millis = trace_read_guest_u32(0x807_b26_f74);
    let limiter_target_millis = trace_read_guest_u32(0x807_b26_f78);
    let process_counter = kernel_process_time(true);
    let named_thread = 0x807_b39_d68;
    let named_queue_index = i64::from(i32::from_ne_bytes(
        trace_read_guest_u32(named_thread + 0x20).to_ne_bytes(),
    ));
    let named_queue = named_thread.wrapping_add_signed(named_queue_index.saturating_mul(16));
    let named_queue_state = trace_read_guest_u64(named_queue);
    let named_queue_flags = if named_queue_state == 0 {
        0
    } else {
        trace_read_guest_u64(named_queue_state.saturating_add(0x10))
    };
    // FTaskGraphInterface::WaitUntilTaskCompletes passes an FGraphEventRef wrapper to
    // GameThreadWaitForTask.  In this title, the caller at 0x801be0030 keeps that reference at
    // rbp-0x40 while GameThreadWaitForTask reuses its own rbp-0x60 local for timer state.  Reading
    // the latter produced packed timer bits that merely resembled a low pointer.  The event's
    // +0x10 word contains the completion bit (0x0400_0000); keeping the surrounding words lets a
    // bounded run distinguish a task that was never dequeued from one that executed but failed to
    // dispatch its subsequents.
    let target_graph_event = if parent_rbp < 0x40 {
        0
    } else {
        trace_read_guest_u64(parent_rbp - 0x40)
    };
    if trace_taskgraph_wait_snapshot()
        && sequence >= 4096
        && sequence.is_power_of_two()
        && plausible_guest_pointer(target_graph_event)
        && taskgraph_wait_chain_matches(event_caller_return, parent_return, grandparent_return)
    {
        emit_taskgraph_wait_snapshot(sequence, target_graph_event);
    }
    let target_graph_event_words = if target_graph_event == 0 {
        [0; 12]
    } else {
        std::array::from_fn(|index| {
            trace_read_guest_u64(
                target_graph_event
                    .saturating_add(u64::try_from(index).unwrap_or(0).saturating_mul(8)),
            )
        })
    };
    let render_named_thread = trace_read_guest_u32(0x807_b20_9d0);
    let parent_locals = if parent_rbp < 0x80 {
        [0; 16]
    } else {
        std::array::from_fn(|index| {
            trace_read_guest_u64(
                parent_rbp
                    .saturating_sub(0x80)
                    .saturating_add(u64::try_from(index).unwrap_or(0).saturating_mul(8)),
            )
        })
    };
    let (waiters_before, notifies_before, timeouts_before) = condition_trace_stats(condition);
    eprintln!(
        "[REMu][COND][TIMED] enter sequence={sequence} thread={thread_name:?}/{:#x} return={:#x} rsp={guest_rsp:#x} event_caller={event_caller_return:#x}/rbp:{event_caller_rbp:#x} ancestors={parent_return:#x}/rbp:{parent_rbp:#x},{grandparent_return:#x}/rbp:{grandparent_rbp:#x},{great_grandparent_return:#x}/rbp:{great_grandparent_rbp:#x} graph_event={target_graph_event:#x}/words:{target_graph_event_words:#x?} render_named_thread={render_named_thread:#x} parent_locals={parent_locals:#x?} named_queue={named_queue:#x}/index:{named_queue_index}/state:{named_queue_state:#x}/flags:{named_queue_flags:#x} limiter_start={:#018x}/{:.9} limiter_target={:#018x}/{:.9} limiter_globals={limiter_wait_millis}/{limiter_target_millis} process_counter={process_counter} condition={condition:#x} mutex={mutex:#x} deadline={deadline:#x} timespec={deadline_seconds}s/{deadline_nanoseconds}ns valid_nsec={} timeout_usec={timeout_usec} waiters={waiters_before} notifies={notifies_before} timeouts={timeouts_before}",
        current_guest_thread_handle(),
        frame.map_or(0, |frame| frame.guest_return_address),
        clock_start_bits,
        f64::from_bits(clock_start_bits),
        clock_target_bits,
        f64::from_bits(clock_target_bits),
        (0..1_000_000_000).contains(&deadline_nanoseconds),
    );
    Some(TimedConditionTrace {
        sequence,
        started: Instant::now(),
        condition,
        waiters_before,
        notifies_before,
        timeouts_before,
    })
}

fn finish_main_timed_condition_trace(trace: Option<TimedConditionTrace>, result: u64) {
    let Some(trace) = trace else {
        return;
    };
    let (waiters_after, notifies_after, timeouts_after) = condition_trace_stats(trace.condition);
    eprintln!(
        "[REMu][COND][TIMED] exit sequence={} result={result:#x} elapsed_us={} condition={:#x} waiters={}->{} notifies={}->{} timeouts={}->{}",
        trace.sequence,
        trace.started.elapsed().as_micros(),
        trace.condition,
        trace.waiters_before,
        waiters_after,
        trace.notifies_before,
        notifies_after,
        trace.timeouts_before,
        timeouts_after,
    );
}

/// Reports whether a NID is served before the process-wide dispatch lock is taken.
///
/// Membership is what makes a handler allowed to park. Every entry point that can wait for another
/// guest thread must be listed here, because the thread that would release it needs the dispatch
/// lock to reach its own release call.
pub(crate) fn is_guest_sync_import(nid: &str) -> bool {
    matches!(
        nid,
        "cmo1RIYva9o"
            | "2Of0f+3mhhE"
            | "9UK1vLZQft4"
            | "upoVrzMHFeE"
            | "tn3VlD0hG60"
            | "ttHNfU+qDBU"
            | "ltCfaGr2JGE"
            | "7H0iTOciTLo"
            | "K-jXhbt2gn4"
            | "2Z+PpY6CaJg"
            | "ytQULN-nhL4"
            | "1471ajPzxh0"
            | "iGjsr1WAtI0"
            | "sIlRvQqsN2Y"
            | "SFxTMOfuCkE"
            | "XhWHn6P5R7U"
            | "EgmLo6EWgso"
            | "2Tb92quprl0"
            | "0TyVk4MSLt0"
            | "g+PZd2hiacg"
            | "RXXqi4CtF8w"
            | "WKAXJ4XBPQ4"
            | "BmMjYxmew1w"
            | "27bAgiJmOh0"
            | "kDh-NfxgMtE"
            | "JGgj7Uvrl+A"
            | "mkx2fVhNMsg"
            | "2MOy+rUfuhQ"
            | "Op8TBGY5KHg"
            | "14bOACANTBo"
            | "Z4QosVuAsA0"
            | "188x57JYp0g"
            | "R1Jvn8bSCW8"
            | "Zxa0VhQVTsk"
            | "4czppHBiriw"
            | "BpFoboUJoZU"
            | "8mql9OcQnd4"
            | "IOnSvHzqu6A"
            | "7uhBFWRAS60"
            | "9lvj5DjHZiA"
            | "JTvBflhYazQ"
            | "PZku4ZrXJqg"
            | "6ULAa0fq4jA"
            | "BB+kb08Tl9A"
            | "Ox9i0c7L5w0"
            | "mqdNorrB+gI"
            | "XD3mDeybCnk"
            | "bIHoZCTomsI"
            | "+L98PIbGttk"
            | "12wOHk8ywb0"
    )
}

#[cfg(test)]
pub(crate) fn dispatch_guest_sync_import(
    nid: &str,
    arguments: [u64; 6],
    mutex_attributes: Option<PthreadMutexAttrState>,
) -> u64 {
    dispatch_guest_sync_import_at_import(nid, arguments, mutex_attributes, None)
}

pub(crate) fn dispatch_guest_sync_import_at_import(
    nid: &str,
    arguments: [u64; 6],
    mutex_attributes: Option<PthreadMutexAttrState>,
    frame: Option<&ImportCallFrame>,
) -> u64 {
    match nid {
        "cmo1RIYva9o" | "ttHNfU+qDBU" => {
            guest_mutex_init(arguments[0], mutex_attributes.unwrap_or_default())
        }
        "2Of0f+3mhhE" | "ltCfaGr2JGE" => guest_mutex_destroy(arguments[0]),
        "9UK1vLZQft4" | "7H0iTOciTLo" => guest_mutex_lock(arguments[0], false),
        "upoVrzMHFeE" | "K-jXhbt2gn4" => guest_mutex_lock(arguments[0], true),
        "tn3VlD0hG60" | "2Z+PpY6CaJg" => guest_mutex_unlock(arguments[0]),
        "ytQULN-nhL4" => guest_rwlock_init(arguments[0]),
        "1471ajPzxh0" => guest_rwlock_destroy(arguments[0]),
        "iGjsr1WAtI0" => guest_rwlock_lock(arguments[0], false, false),
        "sIlRvQqsN2Y" => guest_rwlock_lock(arguments[0], true, false),
        "SFxTMOfuCkE" => guest_rwlock_lock(arguments[0], false, true),
        "XhWHn6P5R7U" => guest_rwlock_lock(arguments[0], true, true),
        "EgmLo6EWgso" => guest_rwlock_unlock(arguments[0]),
        // The console reader/writer lock is the POSIX object under another name, so both spellings
        // share one implementation and one convention: the pthread errno as the reported value, no
        // wrapper. The blocking acquires belong on this side of the dispatch lock, not in
        // `kernel_api`, whose entry points run with that lock held.
        "6ULAa0fq4jA" => guest_rwlock_init(arguments[0]),
        "BB+kb08Tl9A" => guest_rwlock_destroy(arguments[0]),
        "Ox9i0c7L5w0" => guest_rwlock_lock(arguments[0], false, false),
        "mqdNorrB+gI" => guest_rwlock_lock(arguments[0], true, false),
        "XD3mDeybCnk" => guest_rwlock_lock(arguments[0], false, true),
        "bIHoZCTomsI" => guest_rwlock_lock(arguments[0], true, true),
        "+L98PIbGttk" => guest_rwlock_unlock(arguments[0]),
        "12wOHk8ywb0" => guest_semaphore_poll(arguments),
        "2Tb92quprl0" => guest_condition_init(arguments[0]),
        "0TyVk4MSLt0" => posix_pthread_result(guest_condition_init(arguments[0])),
        "g+PZd2hiacg" => guest_condition_destroy(arguments[0]),
        "RXXqi4CtF8w" => posix_pthread_result(guest_condition_destroy(arguments[0])),
        "WKAXJ4XBPQ4" => {
            let observation =
                begin_condition_wait_observation(frame, arguments[0], arguments[1], None);
            let result =
                guest_condition_wait_with_interrupt(arguments[0], arguments[1], None, || {
                    if let Some(frame) = frame {
                        deliver_pending_guest_exception_at_import(frame);
                    }
                });
            finish_condition_wait_observation(observation, result);
            result
        }
        "Op8TBGY5KHg" => {
            let observation =
                begin_condition_wait_observation(frame, arguments[0], arguments[1], None);
            let result =
                guest_condition_wait_with_interrupt(arguments[0], arguments[1], None, || {
                    if let Some(frame) = frame {
                        deliver_pending_guest_exception_at_import(frame);
                    }
                });
            finish_condition_wait_observation(observation, result);
            posix_pthread_result(result)
        }
        "BmMjYxmew1w" => {
            let timeout_usec = Some(u64::from(low_u32(arguments[2])));
            let observation =
                begin_condition_wait_observation(frame, arguments[0], arguments[1], timeout_usec);
            let result = guest_condition_wait_with_interrupt(
                arguments[0],
                arguments[1],
                timeout_usec,
                || {
                    if let Some(frame) = frame {
                        deliver_pending_guest_exception_at_import(frame);
                    }
                },
            );
            finish_condition_wait_observation(observation, result);
            result
        }
        // The POSIX form takes an absolute deadline as a timespec; the console form takes a
        // relative microsecond count. Convert against the same realtime clock the deadline names.
        "27bAgiJmOh0" => {
            let timeout_usec = relative_microseconds_until(arguments[2]);
            let observation = begin_main_timed_wait_observation(frame, arguments[0], timeout_usec);
            let trace = begin_main_timed_condition_trace(
                frame,
                arguments[0],
                arguments[1],
                arguments[2],
                timeout_usec,
            );
            let result = posix_pthread_result(guest_condition_wait_with_interrupt(
                arguments[0],
                arguments[1],
                Some(timeout_usec),
                || {
                    if let Some(frame) = frame {
                        deliver_pending_guest_exception_at_import(frame);
                    }
                },
            ));
            finish_main_timed_wait_observation(observation, result);
            finish_main_timed_condition_trace(trace, result);
            result
        }
        "kDh-NfxgMtE" => {
            let observation = begin_condition_notify_observation(frame, arguments[0], false);
            trace_main_condition_notify(frame, arguments[0], false);
            let result = guest_condition_notify(arguments[0], false);
            finish_condition_notify_observation(observation, result);
            result
        }
        "2MOy+rUfuhQ" => {
            let observation = begin_condition_notify_observation(frame, arguments[0], false);
            trace_main_condition_notify(frame, arguments[0], false);
            let result = guest_condition_notify(arguments[0], false);
            finish_condition_notify_observation(observation, result);
            posix_pthread_result(result)
        }
        "JGgj7Uvrl+A" => {
            let observation = begin_condition_notify_observation(frame, arguments[0], true);
            trace_main_condition_notify(frame, arguments[0], true);
            let result = guest_condition_notify(arguments[0], true);
            finish_condition_notify_observation(observation, result);
            result
        }
        "mkx2fVhNMsg" => {
            let observation = begin_condition_notify_observation(frame, arguments[0], true);
            trace_main_condition_notify(frame, arguments[0], true);
            let result = guest_condition_notify(arguments[0], true);
            finish_condition_notify_observation(observation, result);
            posix_pthread_result(result)
        }
        "14bOACANTBo" => guest_pthread_once(arguments[0], arguments[1]),
        "Z4QosVuAsA0" => posix_pthread_result(guest_pthread_once(arguments[0], arguments[1])),
        "188x57JYp0g" => guest_semaphore_create(arguments),
        "R1Jvn8bSCW8" => guest_semaphore_delete(arguments[0]),
        "Zxa0VhQVTsk" => guest_semaphore_wait(arguments),
        "4czppHBiriw" => guest_semaphore_signal(arguments),
        "BpFoboUJoZU" => guest_event_flag_create(arguments),
        "8mql9OcQnd4" => guest_event_flag_delete(arguments[0]),
        "IOnSvHzqu6A" => guest_event_flag_set(arguments[0], arguments[1]),
        "7uhBFWRAS60" => guest_event_flag_clear(arguments[0], arguments[1]),
        "9lvj5DjHZiA" => guest_event_flag_poll(arguments),
        "JTvBflhYazQ" => guest_event_flag_wait(arguments),
        "PZku4ZrXJqg" => guest_event_flag_cancel(arguments),
        _ => 0,
    }
}

#[cfg(test)]
mod taskgraph_wait_snapshot_tests {
    use super::{
        MAXIMUM_GUEST_ADDRESS, MINIMUM_GUEST_ADDRESS, TASKGRAPH_ALLOCATOR_PAGE_BYTES,
        TASKGRAPH_FENCE_TASK_VPTR, TASKGRAPH_NODE_PAGE_BITS, TASKGRAPH_NODE_PAGE_MASK,
        TASKGRAPH_NODE_SLAB_POINTERS, TASKGRAPH_NODE_STRIDE, TASKGRAPH_QUEUE_EVENT_WRAPPER_OFFSET,
        TASKGRAPH_QUEUE_WAIT_STATE_OFFSET, TASKGRAPH_TASK_ALIGNMENT,
        TASKGRAPH_WAIT_EVENT_CALLER_RETURN, TASKGRAPH_WAIT_GRANDPARENT_RETURN,
        TASKGRAPH_WAIT_PARENT_RETURN, scan_taskgraph_allocator_pages, taskgraph_allocator_pages,
        taskgraph_queue_sync_snapshot, taskgraph_wait_chain_matches, walk_taskgraph_lane,
    };
    use std::collections::BTreeMap;

    fn install_node(
        memory: &mut BTreeMap<u64, u64>,
        handle: u64,
        page_base: u64,
        next_handle: u64,
        task: u64,
        event: u64,
    ) {
        let node_id = handle & ((1 << 26) - 1);
        let page = node_id >> TASKGRAPH_NODE_PAGE_BITS;
        let slot = node_id & TASKGRAPH_NODE_PAGE_MASK;
        let node = page_base + slot * TASKGRAPH_NODE_STRIDE;
        memory.insert(TASKGRAPH_NODE_SLAB_POINTERS + page * 8, page_base);
        memory.insert(node, next_handle);
        memory.insert(node + 8, task);
        memory.insert(task, TASKGRAPH_FENCE_TASK_VPTR);
        memory.insert(task + 0x18, event);
    }

    #[test]
    fn exact_silent_hill_wait_chain_is_required() {
        assert!(taskgraph_wait_chain_matches(
            TASKGRAPH_WAIT_EVENT_CALLER_RETURN,
            TASKGRAPH_WAIT_PARENT_RETURN,
            TASKGRAPH_WAIT_GRANDPARENT_RETURN,
        ));
        assert!(!taskgraph_wait_chain_matches(
            TASKGRAPH_WAIT_EVENT_CALLER_RETURN,
            TASKGRAPH_WAIT_PARENT_RETURN + 1,
            TASKGRAPH_WAIT_GRANDPARENT_RETURN,
        ));
    }

    #[test]
    fn queue_sync_snapshot_names_exact_wait_state_and_event_condition() {
        let queue = 0x72_e0df_4dc8;
        let event_wrapper = 0x70_20e1_32c0;
        let event_underlying = 0x70_40e5_50a0;
        let wait_state = (0xb13e << 26) | 1;
        let mut memory = BTreeMap::from([
            (queue + TASKGRAPH_QUEUE_WAIT_STATE_OFFSET, wait_state),
            (queue + TASKGRAPH_QUEUE_EVENT_WRAPPER_OFFSET, event_wrapper),
            (event_wrapper + 0x10, event_underlying),
            (event_underlying + 0x10, 120_127_489),
        ]);
        let mut read_u64 = |address| memory.remove(&address);

        let snapshot = taskgraph_queue_sync_snapshot(queue, &mut read_u64);

        assert_eq!(snapshot.wait_state, Some(wait_state));
        assert_eq!(snapshot.waiters, Some(1));
        assert_eq!(snapshot.generation, Some(0xb13e));
        assert_eq!(snapshot.event_wrapper, Some(event_wrapper));
        assert_eq!(snapshot.event_underlying, Some(event_underlying));
        assert_eq!(snapshot.event_word, Some(120_127_489));
        assert_eq!(snapshot.event_mutex, Some(event_underlying + 0x20));
        assert_eq!(snapshot.event_condition, Some(event_underlying + 0x28));
    }

    #[test]
    fn paged_queue_walk_finds_same_event_task_after_dummy_head() {
        let event = 0x75_00e1_2300;
        let page_base = 0x75_00f0_0000;
        let head = 0x4001;
        let tail = 0x4002;
        let dummy_task = 0x75_00e1_0000;
        let fence_task = 0x75_00e2_0000;
        let mut memory = BTreeMap::new();
        install_node(&mut memory, head, page_base, tail, dummy_task, event + 8);
        install_node(&mut memory, tail, page_base, 0, fence_task, event);
        let mut read_u64 = |address| memory.get(&address).copied();

        let snapshot = walk_taskgraph_lane(1, 0, Some(head), Some(tail), event, &mut read_u64);

        assert_eq!(snapshot.queue_index, 1);
        assert_eq!(snapshot.termination, "tail");
        assert_eq!(snapshot.classification, "queued");
        assert_eq!(snapshot.event_match_indexes(), vec![1]);
        assert_eq!(snapshot.head_task, Some(dummy_task));
        assert_eq!(snapshot.tail_task, Some(fence_task));
    }

    #[test]
    fn queue_walk_stops_on_recycled_node_cycle() {
        let page_base = 0x75_00f0_0000;
        let head = 0x4001;
        let recycled_head = head | (1 << 26);
        let tail = 0x4002;
        let mut memory = BTreeMap::new();
        install_node(
            &mut memory,
            head,
            page_base,
            recycled_head,
            0x75_00e1_0000,
            0x75_00e1_2308,
        );
        let mut read_u64 = |address| memory.get(&address).copied();

        let snapshot =
            walk_taskgraph_lane(0, 1, Some(head), Some(tail), 0x75_00e1_2300, &mut read_u64);

        assert_eq!(snapshot.termination, "cycle");
        assert_eq!(snapshot.nodes.len(), 1);
        assert_eq!(snapshot.classification, "absent");
    }

    #[test]
    fn allocator_pages_are_only_those_containing_cursor_task_anchors() {
        let first_page = 0x75_00e0_0000;
        let second_page = first_page + 3 * TASKGRAPH_ALLOCATOR_PAGE_BYTES;

        let pages = taskgraph_allocator_pages([
            0,
            MINIMUM_GUEST_ADDRESS - TASKGRAPH_TASK_ALIGNMENT,
            first_page + 0x100,
            first_page + 0x800,
            second_page + 0x200,
            MAXIMUM_GUEST_ADDRESS,
        ]);

        assert_eq!(pages, vec![first_page, second_page]);
    }

    #[test]
    fn allocator_scan_is_bounded_to_anchor_page_and_matches_task_event() {
        let page = 0x75_00e0_0000;
        let task = page + 0x200;
        let event = 0x75_00d0_0100;
        let mut memory = BTreeMap::from([(task, TASKGRAPH_FENCE_TASK_VPTR), (task + 0x18, event)]);
        for index in 1..8_u64 {
            memory.entry(task + index * 8).or_insert(0xa0 + index);
        }
        let mut reads = Vec::new();
        let mut read_u64 = |address| {
            reads.push(address);
            memory.get(&address).copied()
        };

        let (matches, truncated) = scan_taskgraph_allocator_pages(&[page], event, &mut read_u64);

        assert!(!truncated);
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].task, task);
        assert_eq!(matches[0].words[0], Some(TASKGRAPH_FENCE_TASK_VPTR));
        assert!(
            reads
                .iter()
                .all(|address| (page..page + TASKGRAPH_ALLOCATOR_PAGE_BYTES).contains(address))
        );
    }
}

#[cfg(test)]
mod condition_lifecycle_trace_tests {
    use super::ConditionLifecycleTrace;

    fn begin_wait(
        trace: &mut ConditionLifecycleTrace,
        started_nanos: u64,
        thread: u64,
        condition: u64,
    ) -> super::ConditionWaitCallTrace {
        trace
            .begin_wait(
                started_nanos,
                thread,
                format!("thread-{thread}"),
                0x8000_0000 + thread,
                condition,
                condition + 8,
            )
            .expect("trace has room for a wait")
    }

    fn begin_notify(
        trace: &mut ConditionLifecycleTrace,
        started_nanos: u64,
        thread: u64,
        condition: u64,
        waiters_before: usize,
    ) -> super::ConditionNotifyCallTrace {
        trace
            .begin_notify(
                started_nanos,
                thread,
                format!("thread-{thread}"),
                0x9000_0000 + thread,
                condition,
                false,
                waiters_before,
            )
            .expect("trace has room for a notification")
    }

    #[test]
    fn condition_lifecycle_orders_notify_wake_and_repark() {
        let mut trace = ConditionLifecycleTrace::new(4);
        let first_wait = begin_wait(&mut trace, 10, 3, 0x1000);
        assert!(trace.has_active_wait(0x1000));

        let notify = begin_notify(&mut trace, 20, 1, 0x1000, 1);
        trace.finish_notify(notify, 30, 0, 0);
        trace.finish_wait(first_wait, 40, 0);
        let second_wait = begin_wait(&mut trace, 50, 3, 0x1000);

        let last_notify = trace.last_notify(0x1000).expect("notify is retained");
        let last_return = trace
            .last_wait_return(3, 0x1000)
            .expect("wake return is retained");
        assert!(last_notify.return_sequence < last_return.return_sequence);
        assert!(last_return.return_sequence < second_wait.call_sequence);
        assert_eq!(last_notify.call.thread, 1);
        assert_eq!(last_notify.call.waiters_before, 1);
        assert_eq!(last_notify.waiters_after, 0);
        assert_eq!(last_return.result, 0);
    }

    #[test]
    fn condition_lifecycle_retains_latest_producer_per_condition() {
        let mut trace = ConditionLifecycleTrace::new(4);
        let _wait = begin_wait(&mut trace, 10, 3, 0x1000);
        for (nanos, thread) in [(20, 1), (30, 2), (40, 4)] {
            let notify = begin_notify(&mut trace, nanos, thread, 0x1000, 1);
            trace.finish_notify(notify, nanos + 1, 0, 0);
        }

        assert_eq!(trace.recent_notifies.len(), 1);
        assert_eq!(
            trace
                .last_notify(0x1000)
                .expect("latest notify is retained")
                .call
                .thread,
            4
        );
    }

    #[test]
    fn condition_lifecycle_has_fixed_active_and_history_capacity() {
        let mut trace = ConditionLifecycleTrace::new(2);
        let first = begin_wait(&mut trace, 10, 1, 0x1000);
        let second = begin_wait(&mut trace, 20, 2, 0x2000);
        assert!(
            trace
                .begin_wait(30, 3, "thread-3".to_owned(), 0x8000_0003, 0x3000, 0x3008)
                .is_none()
        );
        assert_eq!(trace.dropped_active_waits, 1);

        trace.finish_wait(first, 40, 0);
        trace.finish_wait(second, 50, 0);
        for (nanos, thread, condition) in [(60, 4, 0x4000), (70, 5, 0x5000), (80, 6, 0x6000)] {
            let wait = begin_wait(&mut trace, nanos, thread, condition);
            trace.finish_wait(wait, nanos + 1, 0);
        }

        assert_eq!(trace.active_waits.len(), 0);
        assert_eq!(trace.recent_wait_returns.len(), 2);
        assert_eq!(
            trace
                .recent_wait_returns
                .front()
                .expect("bounded tail has a first record")
                .call
                .thread,
            5
        );
        assert_eq!(
            trace
                .recent_wait_returns
                .back()
                .expect("bounded tail has a final record")
                .call
                .thread,
            6
        );
    }
}
