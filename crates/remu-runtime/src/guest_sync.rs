use crate::{
    ImportCallFrame, PthreadMutexAttrState, current_guest_thread_handle, current_guest_thread_name,
    deliver_pending_guest_exception_at_import, kernel_process_time, low_u32, posix_pthread_result,
    read_guest_u32, read_guest_u64, relative_microseconds_until, trace_guest_sync,
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
    GuestRwlockState, MUTEX_CONTENTION_SEQUENCE, guest_mutex_shard, guest_mutex_shards,
};

static MAIN_TIMED_CONDITION_TRACE_SEQUENCE: AtomicU64 = AtomicU64::new(0);
static MAIN_CONDITION_NOTIFY_TRACE_SEQUENCE: AtomicU64 = AtomicU64::new(0);

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
            read_guest_u64(
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
            i64::from_ne_bytes(read_guest_u64(deadline).to_ne_bytes()),
            i64::from_ne_bytes(read_guest_u64(deadline + 8).to_ne_bytes()),
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
            read_guest_u64(guest_rsp.saturating_add(0x88)),
            read_guest_u64(guest_rsp.saturating_add(0x90)),
        )
    };
    let (parent_rbp, parent_return) = if event_caller_rbp == 0 {
        (0, 0)
    } else {
        (
            read_guest_u64(event_caller_rbp),
            read_guest_u64(event_caller_rbp.saturating_add(8)),
        )
    };
    let (clock_start_bits, clock_target_bits) = if event_caller_rbp < 0x78 {
        (0, 0)
    } else {
        (
            read_guest_u64(event_caller_rbp - 0x78),
            read_guest_u64(event_caller_rbp - 0x70),
        )
    };
    let limiter_wait_millis = read_guest_u32(0x807_b26_f74);
    let limiter_target_millis = read_guest_u32(0x807_b26_f78);
    let process_counter = kernel_process_time(true);
    let (waiters_before, notifies_before, timeouts_before) = condition_trace_stats(condition);
    eprintln!(
        "[REMu][COND][TIMED] enter sequence={sequence} thread={thread_name:?}/{:#x} return={:#x} rsp={guest_rsp:#x} event_caller={event_caller_return:#x}/rbp:{event_caller_rbp:#x} parent={parent_return:#x}/rbp:{parent_rbp:#x} limiter_start={:#018x}/{:.9} limiter_target={:#018x}/{:.9} limiter_globals={limiter_wait_millis}/{limiter_target_millis} process_counter={process_counter} condition={condition:#x} mutex={mutex:#x} deadline={deadline:#x} timespec={deadline_seconds}s/{deadline_nanoseconds}ns valid_nsec={} timeout_usec={timeout_usec} waiters={waiters_before} notifies={notifies_before} timeouts={timeouts_before}",
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
            guest_condition_wait_with_interrupt(arguments[0], arguments[1], None, || {
                if let Some(frame) = frame {
                    deliver_pending_guest_exception_at_import(frame);
                }
            })
        }
        "Op8TBGY5KHg" => posix_pthread_result(guest_condition_wait_with_interrupt(
            arguments[0],
            arguments[1],
            None,
            || {
                if let Some(frame) = frame {
                    deliver_pending_guest_exception_at_import(frame);
                }
            },
        )),
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
            trace_main_condition_notify(frame, arguments[0], false);
            guest_condition_notify(arguments[0], false)
        }
        "2MOy+rUfuhQ" => {
            trace_main_condition_notify(frame, arguments[0], false);
            posix_pthread_result(guest_condition_notify(arguments[0], false))
        }
        "JGgj7Uvrl+A" => {
            trace_main_condition_notify(frame, arguments[0], true);
            guest_condition_notify(arguments[0], true)
        }
        "mkx2fVhNMsg" => {
            trace_main_condition_notify(frame, arguments[0], true);
            posix_pthread_result(guest_condition_notify(arguments[0], true))
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
