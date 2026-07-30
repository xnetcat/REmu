use crate::{
    ImportCallFrame, PthreadMutexAttrState, current_guest_thread_handle, current_guest_thread_name,
    deliver_pending_guest_exception_at_import, kernel_process_time, low_u32, posix_pthread_result,
    relative_microseconds_until, trace_guest_sync, try_read_process_guest_u32,
    try_read_process_guest_u64,
};
use std::collections::BTreeSet;
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
use self::semaphore::{guest_semaphore_create, guest_semaphore_delete, guest_semaphore_signal};
use self::state::{
    EVENT_FLAG_SET_SEQUENCE, EVENT_FLAG_WAIT_SEQUENCE, GuestMutexEntry, GuestMutexState,
    GuestRwlockReaderState, GuestRwlockState, MUTEX_CONTENTION_SEQUENCE, guest_mutex_shard,
    guest_mutex_shards,
};

static MAIN_TIMED_CONDITION_TRACE_SEQUENCE: AtomicU64 = AtomicU64::new(0);
static MAIN_CONDITION_NOTIFY_TRACE_SEQUENCE: AtomicU64 = AtomicU64::new(0);
static NAMED_THREAD_CONDITION_TRACE_SEQUENCE: AtomicU64 = AtomicU64::new(0);
static NAMED_CONDITION_LIFECYCLE_TRACE_SEQUENCE: AtomicU64 = AtomicU64::new(0);
static LAST_PENDING_RHI_CHAIN_TRACE: AtomicU64 = AtomicU64::new(0);
// Enough for a bounded launch trace while preventing diagnostics from growing without limit.
const NAMED_CONDITION_LIFECYCLE_TRACE_LIMIT: u64 = 8_192;

fn trace_read_guest_u32(address: u64) -> u32 {
    try_read_process_guest_u32(address).unwrap_or(0)
}

fn trace_read_guest_u64(address: u64) -> u64 {
    try_read_process_guest_u64(address).unwrap_or(0)
}

fn trace_guest_words<const N: usize>(address: u64) -> [u64; N] {
    if address == 0 {
        return [0; N];
    }
    std::array::from_fn(|index| {
        trace_read_guest_u64(
            address.saturating_add(u64::try_from(index).unwrap_or(0).saturating_mul(8)),
        )
    })
}

#[allow(dead_code)] // Read through the derived Debug implementation in trace-only output.
#[derive(Debug)]
struct GraphEventTrace {
    event: u64,
    words: [u64; 8],
}

impl GraphEventTrace {
    fn is_pending(&self) -> bool {
        const GRAPH_EVENT_COMPLETE: u64 = 0x0400_0000;

        self.event != 0 && self.words[2] & GRAPH_EVENT_COMPLETE == 0
    }
}

fn trace_graph_event_pointer(pointer_address: u64) -> GraphEventTrace {
    let event = trace_read_guest_u64(pointer_address);
    GraphEventTrace {
        event,
        words: trace_guest_words(event),
    }
}

#[allow(dead_code)] // Read through the derived Debug implementation in trace-only output.
#[derive(Debug)]
struct RhiTaskGraphTrace {
    task_graph: u64,
    task_graph_words: [u64; 24],
    render_named_thread: u32,
    rhi_named_thread: u32,
    command_list_words: [u64; 20],
    dispatch_event: GraphEventTrace,
    completion_event: GraphEventTrace,
    submit_event: GraphEventTrace,
    prerequisite_count: u32,
    prerequisite_control: u32,
    prerequisite_storage: u64,
    prerequisites: Vec<GraphEventTrace>,
}

impl RhiTaskGraphTrace {
    fn pending_chain_identity(&self) -> Option<u64> {
        let events = [
            &self.completion_event,
            &self.submit_event,
            &self.dispatch_event,
        ];
        let mut identity = 0x9e37_79b9_7f4a_7c15_u64;
        let mut pending = false;
        for (index, event) in events.into_iter().enumerate() {
            if event.is_pending() {
                pending = true;
                identity ^= event
                    .event
                    .rotate_left(u32::try_from(index * 21).unwrap_or(0));
            }
        }
        pending.then_some(identity.max(1))
    }
}

fn trace_rhi_task_graph() -> RhiTaskGraphTrace {
    const TASK_GRAPH_POINTER: u64 = 0x807_ba3_5e0;
    const RENDER_NAMED_THREAD: u64 = 0x807_b20_9d0;
    const RHI_NAMED_THREAD: u64 = 0x807_b20_9d4;
    const RHI_COMMAND_LIST: u64 = 0x807_c90_838;
    const PREREQUISITE_INLINE: u64 = 0x807_c92_cc8;
    const PREREQUISITE_HEAP_POINTER: u64 = 0x807_c92_ce8;
    const PREREQUISITE_COUNT: u64 = 0x807_c92_cf0;
    const PREREQUISITE_CONTROL: u64 = 0x807_c92_cf4;
    const COMPLETION_EVENT_POINTER: u64 = 0x807_c92_cf8;
    const SUBMIT_EVENT_POINTER: u64 = 0x807_c92_d00;
    const DISPATCH_EVENT_POINTER: u64 = 0x807_c92_d08;

    let task_graph = trace_read_guest_u64(TASK_GRAPH_POINTER);
    let render_named_thread = trace_read_guest_u32(RENDER_NAMED_THREAD);
    let rhi_named_thread = trace_read_guest_u32(RHI_NAMED_THREAD);
    let prerequisite_count = trace_read_guest_u32(PREREQUISITE_COUNT);
    let prerequisite_control = trace_read_guest_u32(PREREQUISITE_CONTROL);
    let prerequisite_heap = trace_read_guest_u64(PREREQUISITE_HEAP_POINTER);
    let prerequisite_storage = if prerequisite_heap == 0 {
        PREREQUISITE_INLINE
    } else {
        prerequisite_heap
    };
    let prerequisites = (0..prerequisite_count.min(16))
        .map(|index| {
            let event = trace_read_guest_u64(
                prerequisite_storage.saturating_add(u64::from(index).saturating_mul(8)),
            );
            GraphEventTrace {
                event,
                words: trace_guest_words(event),
            }
        })
        .collect();

    RhiTaskGraphTrace {
        task_graph,
        task_graph_words: trace_guest_words(task_graph),
        render_named_thread,
        rhi_named_thread,
        command_list_words: trace_guest_words(RHI_COMMAND_LIST),
        dispatch_event: trace_graph_event_pointer(DISPATCH_EVENT_POINTER),
        completion_event: trace_graph_event_pointer(COMPLETION_EVENT_POINTER),
        submit_event: trace_graph_event_pointer(SUBMIT_EVENT_POINTER),
        prerequisite_count,
        prerequisite_control,
        prerequisite_storage,
        prerequisites,
    }
}

fn traced_main_conditions() -> &'static Mutex<BTreeSet<u64>> {
    static CONDITIONS: OnceLock<Mutex<BTreeSet<u64>>> = OnceLock::new();
    CONDITIONS.get_or_init(|| Mutex::new(BTreeSet::new()))
}

fn traced_named_conditions() -> &'static Mutex<BTreeSet<u64>> {
    static CONDITIONS: OnceLock<Mutex<BTreeSet<u64>>> = OnceLock::new();
    CONDITIONS.get_or_init(|| Mutex::new(BTreeSet::new()))
}

fn named_condition_lifecycle_trace_is_bounded(sequence: u64) -> bool {
    sequence <= NAMED_CONDITION_LIFECYCLE_TRACE_LIMIT
}

fn next_named_condition_lifecycle_trace_sequence(condition: u64) -> Option<u64> {
    if !trace_guest_sync()
        || !traced_named_conditions()
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .contains(&condition)
    {
        return None;
    }
    let sequence = NAMED_CONDITION_LIFECYCLE_TRACE_SEQUENCE.fetch_add(1, Ordering::Relaxed) + 1;
    named_condition_lifecycle_trace_is_bounded(sequence).then_some(sequence)
}

fn trace_named_condition_wait_call(frame: Option<&ImportCallFrame>, condition: u64, mutex: u64) {
    let Some(sequence) = next_named_condition_lifecycle_trace_sequence(condition) else {
        return;
    };
    let guest_rsp = frame.map_or(0, |frame| frame.guest_rsp);
    let stack = trace_guest_words::<16>(guest_rsp);
    let (waiters, notifies, timeouts) = condition_trace_stats(condition);
    eprintln!(
        "[REMu][COND][NAMED-LIFECYCLE] sequence={sequence} event=wait-call thread={:?}/{:#x} return={:#x} rsp={guest_rsp:#x} stack={stack:x?} condition={condition:#x} mutex={mutex:#x} waiters={waiters} notifies={notifies} timeouts={timeouts}",
        current_guest_thread_name(),
        current_guest_thread_handle(),
        frame.map_or(0, |frame| frame.guest_return_address),
    );
}

pub(super) fn trace_named_condition_wait_registered(condition: u64, mutex: u64, ticket: u64) {
    let Some(sequence) = next_named_condition_lifecycle_trace_sequence(condition) else {
        return;
    };
    eprintln!(
        "[REMu][COND][NAMED-LIFECYCLE] sequence={sequence} event=wait-registered thread={:?}/{:#x} condition={condition:#x} mutex={mutex:#x} ticket={ticket}",
        current_guest_thread_name(),
        current_guest_thread_handle(),
    );
}

pub(super) fn trace_named_condition_wait_finished(
    condition: u64,
    mutex: u64,
    ticket: u64,
    timed_out: bool,
    interrupted: bool,
    released_spuriously: bool,
) {
    let Some(sequence) = next_named_condition_lifecycle_trace_sequence(condition) else {
        return;
    };
    eprintln!(
        "[REMu][COND][NAMED-LIFECYCLE] sequence={sequence} event=wait-finished thread={:?}/{:#x} condition={condition:#x} mutex={mutex:#x} ticket={ticket} timed_out={timed_out} interrupted={interrupted} released_spuriously={released_spuriously}",
        current_guest_thread_name(),
        current_guest_thread_handle(),
    );
}

pub(super) fn trace_named_condition_notify_selected(
    condition: u64,
    broadcast: bool,
    selected: &std::collections::VecDeque<u64>,
) {
    let Some(sequence) = next_named_condition_lifecycle_trace_sequence(condition) else {
        return;
    };
    let (selected_head, selected_tail) = selected.as_slices();
    eprintln!(
        "[REMu][COND][NAMED-LIFECYCLE] sequence={sequence} event=notify-selected thread={:?}/{:#x} condition={condition:#x} broadcast={broadcast} selected={selected_head:?}/{selected_tail:?}",
        current_guest_thread_name(),
        current_guest_thread_handle(),
    );
}

struct NamedConditionNotifyTrace {
    condition: u64,
    broadcast: bool,
    waiters_before: usize,
    notifies_before: u64,
    timeouts_before: u64,
}

fn begin_named_condition_notify_trace(
    frame: Option<&ImportCallFrame>,
    condition: u64,
    broadcast: bool,
) -> Option<NamedConditionNotifyTrace> {
    let sequence = next_named_condition_lifecycle_trace_sequence(condition)?;
    let guest_rsp = frame.map_or(0, |frame| frame.guest_rsp);
    let stack = trace_guest_words::<16>(guest_rsp);
    let (waiters_before, notifies_before, timeouts_before) = condition_trace_stats(condition);
    eprintln!(
        "[REMu][COND][NAMED-LIFECYCLE] sequence={sequence} event=notify-call thread={:?}/{:#x} return={:#x} rsp={guest_rsp:#x} stack={stack:x?} condition={condition:#x} broadcast={broadcast} waiters={waiters_before} notifies={notifies_before} timeouts={timeouts_before}",
        current_guest_thread_name(),
        current_guest_thread_handle(),
        frame.map_or(0, |frame| frame.guest_return_address),
    );
    Some(NamedConditionNotifyTrace {
        condition,
        broadcast,
        waiters_before,
        notifies_before,
        timeouts_before,
    })
}

fn finish_named_condition_notify_trace(trace: Option<NamedConditionNotifyTrace>, result: u64) {
    let Some(trace) = trace else {
        return;
    };
    let Some(sequence) = next_named_condition_lifecycle_trace_sequence(trace.condition) else {
        return;
    };
    let (waiters_after, notifies_after, timeouts_after) = condition_trace_stats(trace.condition);
    eprintln!(
        "[REMu][COND][NAMED-LIFECYCLE] sequence={sequence} event=notify-finished thread={:?}/{:#x} condition={:#x} broadcast={} result={result:#x} waiters={}->{} notifies={}->{} timeouts={}->{}",
        current_guest_thread_name(),
        current_guest_thread_handle(),
        trace.condition,
        trace.broadcast,
        trace.waiters_before,
        waiters_after,
        trace.notifies_before,
        notifies_after,
        trace.timeouts_before,
        timeouts_after,
    );
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

struct NamedThreadConditionTrace {
    sequence: u64,
    started: Instant,
    thread_name: String,
    condition: u64,
}

fn begin_named_thread_condition_trace(
    frame: Option<&ImportCallFrame>,
    condition: u64,
    mutex: u64,
) -> Option<NamedThreadConditionTrace> {
    if !trace_guest_sync() {
        return None;
    }
    let thread_name = current_guest_thread_name();
    if !thread_name.contains("RenderThread") && !thread_name.contains("RHIThread") {
        return None;
    }
    traced_named_conditions()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .insert(condition);
    trace_named_condition_wait_call(frame, condition, mutex);
    let sequence = NAMED_THREAD_CONDITION_TRACE_SEQUENCE.fetch_add(1, Ordering::Relaxed) + 1;
    let rhi_task_graph = trace_rhi_task_graph();
    let new_pending_chain = rhi_task_graph
        .pending_chain_identity()
        .is_some_and(|identity| {
            LAST_PENDING_RHI_CHAIN_TRACE.swap(identity, Ordering::Relaxed) != identity
        });
    if sequence > 32 && !sequence.is_power_of_two() && !new_pending_chain {
        return None;
    }
    let guest_rsp = frame.map_or(0, |frame| frame.guest_rsp);
    let stack = trace_guest_words::<24>(guest_rsp);
    let (waiters, notifies, timeouts) = condition_trace_stats(condition);
    eprintln!(
        "[REMu][COND][NAMED-WAIT] enter sequence={sequence} new_pending_rhi_chain={new_pending_chain} thread={thread_name:?}/{:#x} return={:#x} rsp={guest_rsp:#x} stack={stack:x?} condition={condition:#x} mutex={mutex:#x} waiters={waiters} notifies={notifies} timeouts={timeouts} rhi_task_graph={rhi_task_graph:x?}",
        current_guest_thread_handle(),
        frame.map_or(0, |frame| frame.guest_return_address),
    );
    Some(NamedThreadConditionTrace {
        sequence,
        started: Instant::now(),
        thread_name,
        condition,
    })
}

fn finish_named_thread_condition_trace(trace: Option<NamedThreadConditionTrace>, result: u64) {
    let Some(trace) = trace else {
        return;
    };
    let (waiters, notifies, timeouts) = condition_trace_stats(trace.condition);
    eprintln!(
        "[REMu][COND][NAMED-WAIT] exit sequence={} thread={:?} result={result:#x} elapsed_us={} condition={:#x} waiters={waiters} notifies={notifies} timeouts={timeouts}",
        trace.sequence,
        trace.thread_name,
        trace.started.elapsed().as_micros(),
        trace.condition,
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
    let rhi_task_graph = trace_rhi_task_graph();
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
        "[REMu][COND][TIMED] enter sequence={sequence} thread={thread_name:?}/{:#x} return={:#x} rsp={guest_rsp:#x} event_caller={event_caller_return:#x}/rbp:{event_caller_rbp:#x} ancestors={parent_return:#x}/rbp:{parent_rbp:#x},{grandparent_return:#x}/rbp:{grandparent_rbp:#x},{great_grandparent_return:#x}/rbp:{great_grandparent_rbp:#x} graph_event={target_graph_event:#x}/words:{target_graph_event_words:#x?} render_named_thread={render_named_thread:#x} rhi_task_graph={rhi_task_graph:x?} parent_locals={parent_locals:#x?} named_queue={named_queue:#x}/index:{named_queue_index}/state:{named_queue_state:#x}/flags:{named_queue_flags:#x} limiter_start={:#018x}/{:.9} limiter_target={:#018x}/{:.9} limiter_globals={limiter_wait_millis}/{limiter_target_millis} process_counter={process_counter} condition={condition:#x} mutex={mutex:#x} deadline={deadline:#x} timespec={deadline_seconds}s/{deadline_nanoseconds}ns valid_nsec={} timeout_usec={timeout_usec} waiters={waiters_before} notifies={notifies_before} timeouts={timeouts_before}",
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
        "2Tb92quprl0" => guest_condition_init(arguments[0]),
        "0TyVk4MSLt0" => posix_pthread_result(guest_condition_init(arguments[0])),
        "g+PZd2hiacg" => guest_condition_destroy(arguments[0]),
        "RXXqi4CtF8w" => posix_pthread_result(guest_condition_destroy(arguments[0])),
        "WKAXJ4XBPQ4" => {
            let trace = begin_named_thread_condition_trace(frame, arguments[0], arguments[1]);
            let result =
                guest_condition_wait_with_interrupt(arguments[0], arguments[1], None, || {
                    if let Some(frame) = frame {
                        deliver_pending_guest_exception_at_import(frame);
                    }
                });
            finish_named_thread_condition_trace(trace, result);
            result
        }
        "Op8TBGY5KHg" => {
            let trace = begin_named_thread_condition_trace(frame, arguments[0], arguments[1]);
            let result = posix_pthread_result(guest_condition_wait_with_interrupt(
                arguments[0],
                arguments[1],
                None,
                || {
                    if let Some(frame) = frame {
                        deliver_pending_guest_exception_at_import(frame);
                    }
                },
            ));
            finish_named_thread_condition_trace(trace, result);
            result
        }
        "BmMjYxmew1w" => guest_condition_wait_with_interrupt(
            arguments[0],
            arguments[1],
            Some(u64::from(low_u32(arguments[2]))),
            || {
                if let Some(frame) = frame {
                    deliver_pending_guest_exception_at_import(frame);
                }
            },
        ),
        // The POSIX form takes an absolute deadline as a timespec; the console form takes a
        // relative microsecond count. Convert against the same realtime clock the deadline names.
        "27bAgiJmOh0" => {
            let timeout_usec = relative_microseconds_until(arguments[2]);
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
            finish_main_timed_condition_trace(trace, result);
            result
        }
        "kDh-NfxgMtE" => {
            let named_trace = begin_named_condition_notify_trace(frame, arguments[0], false);
            trace_main_condition_notify(frame, arguments[0], false);
            let result = guest_condition_notify(arguments[0], false);
            finish_named_condition_notify_trace(named_trace, result);
            result
        }
        "2MOy+rUfuhQ" => {
            let named_trace = begin_named_condition_notify_trace(frame, arguments[0], false);
            trace_main_condition_notify(frame, arguments[0], false);
            let result = guest_condition_notify(arguments[0], false);
            finish_named_condition_notify_trace(named_trace, result);
            posix_pthread_result(result)
        }
        "JGgj7Uvrl+A" => {
            let named_trace = begin_named_condition_notify_trace(frame, arguments[0], true);
            trace_main_condition_notify(frame, arguments[0], true);
            let result = guest_condition_notify(arguments[0], true);
            finish_named_condition_notify_trace(named_trace, result);
            result
        }
        "mkx2fVhNMsg" => {
            let named_trace = begin_named_condition_notify_trace(frame, arguments[0], true);
            trace_main_condition_notify(frame, arguments[0], true);
            let result = guest_condition_notify(arguments[0], true);
            finish_named_condition_notify_trace(named_trace, result);
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
mod trace_tests {
    use super::{
        GraphEventTrace, NAMED_CONDITION_LIFECYCLE_TRACE_LIMIT, RhiTaskGraphTrace,
        named_condition_lifecycle_trace_is_bounded,
    };

    fn graph_event(event: u64, complete: bool) -> GraphEventTrace {
        let mut words = [0; 8];
        if complete {
            words[2] = 0x0400_0000;
        }
        GraphEventTrace { event, words }
    }

    fn rhi_trace(
        completion_event: GraphEventTrace,
        submit_event: GraphEventTrace,
        dispatch_event: GraphEventTrace,
    ) -> RhiTaskGraphTrace {
        RhiTaskGraphTrace {
            task_graph: 0,
            task_graph_words: [0; 24],
            render_named_thread: 0,
            rhi_named_thread: 0,
            command_list_words: [0; 20],
            dispatch_event,
            completion_event,
            submit_event,
            prerequisite_count: 0,
            prerequisite_control: 0,
            prerequisite_storage: 0,
            prerequisites: Vec::new(),
        }
    }

    #[test]
    fn graph_event_pending_matches_guest_completion_bit() {
        assert!(!graph_event(0, false).is_pending());
        assert!(graph_event(0x1000, false).is_pending());
        assert!(!graph_event(0x1000, true).is_pending());
    }

    #[test]
    fn pending_rhi_chain_identity_includes_the_unfinished_predecessor() {
        let current_only = rhi_trace(
            graph_event(0x1000, false),
            graph_event(0x2000, true),
            graph_event(0, false),
        );
        let current_and_predecessor = rhi_trace(
            graph_event(0x1000, false),
            graph_event(0x2000, false),
            graph_event(0, false),
        );
        let all_complete = rhi_trace(
            graph_event(0x1000, true),
            graph_event(0x2000, true),
            graph_event(0, false),
        );

        assert_ne!(
            current_only.pending_chain_identity(),
            current_and_predecessor.pending_chain_identity()
        );
        assert_eq!(all_complete.pending_chain_identity(), None);
    }

    #[test]
    fn named_condition_lifecycle_trace_has_a_hard_event_limit() {
        assert!(named_condition_lifecycle_trace_is_bounded(1));
        assert!(named_condition_lifecycle_trace_is_bounded(
            NAMED_CONDITION_LIFECYCLE_TRACE_LIMIT
        ));
        assert!(!named_condition_lifecycle_trace_is_bounded(
            NAMED_CONDITION_LIFECYCLE_TRACE_LIMIT + 1
        ));
    }
}
