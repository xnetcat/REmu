use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, OnceLock};
use std::thread::ThreadId;
use std::time::{Duration, Instant};

use remu_shader::GuestMemoryReader;

use crate::{
    ImportCallFrame, ProcessGuestMemory, PthreadMutexAttrState, SCE_KERNEL_ERROR_BUSY,
    SCE_KERNEL_ERROR_INTERRUPTED, SCE_KERNEL_ERROR_INVALID_ARGUMENT, SCE_KERNEL_ERROR_MEMORY_FAULT,
    SCE_KERNEL_ERROR_NOT_FOUND, SCE_KERNEL_ERROR_OPERATION_NOT_PERMITTED,
    SCE_KERNEL_ERROR_TIMED_OUT, SCE_KERNEL_ERROR_TRY_AGAIN, STALL_SNAPSHOT_EMITTED,
    current_guest_thread_handle, current_guest_thread_name,
    deliver_pending_guest_exception_at_import, guest_calloc, guest_exception_is_pending,
    guest_free, low_i32, low_u32, posix_pthread_result, read_guest_cstring, read_guest_u32,
    relative_microseconds_until, report_kernel_event_registrations, sce_kernel_error,
    stall_break_enabled, stall_report_interval, trace_guest_sync, trace_logarithmic_sequence,
    write_guest_u32, write_guest_u64,
};

#[derive(Debug, Default)]
pub(crate) struct GuestSyncState {
    pub(crate) rwlocks: HashMap<u64, GuestRwlockState>,
    pub(crate) conditions: HashMap<u64, GuestConditionState>,
    pub(crate) condition_waiters: HashMap<u64, GuestConditionWaiterState>,
    pub(crate) next_condition_waiter: u64,
    pub(crate) condition_notify_count: u64,
    pub(crate) semaphores: HashMap<u32, GuestSemaphoreState>,
    pub(crate) next_semaphore_handle: u32,
    pub(crate) semaphore_wait_count: u64,
    pub(crate) semaphore_signal_count: u64,
    pub(crate) event_flags: HashMap<u64, GuestEventFlagState>,
    pub(crate) next_event_flag_handle: u64,
}

#[derive(Debug, Default)]
pub(crate) struct GuestConditionState {
    pub(crate) waiters: VecDeque<u64>,
    pub(crate) bound_mutex: Option<u64>,
    /// Signals delivered to this condition. A stalled waiter on a condition that has never been
    /// signaled is a missing producer; one on a signaled condition is a lost wakeup.
    pub(crate) notifies: u64,
    /// Timed waits that expired without being signaled. A large count with no signals is a poll
    /// whose predicate never becomes true.
    pub(crate) timeouts: u64,
}

impl GuestConditionState {
    pub(crate) fn register_waiter(&mut self, waiter: u64, mutex: u64) -> bool {
        if self.bound_mutex.is_some_and(|bound| bound != mutex) {
            return false;
        }
        self.bound_mutex = Some(mutex);
        self.waiters.push_back(waiter);
        true
    }

    pub(crate) fn release_one(&mut self) -> Option<u64> {
        let waiter = self.waiters.pop_front();
        self.clear_binding_if_idle();
        waiter
    }

    pub(crate) fn release_all(&mut self) -> VecDeque<u64> {
        self.bound_mutex = None;
        std::mem::take(&mut self.waiters)
    }

    pub(crate) fn cancel_wait(&mut self, waiter: u64) {
        self.waiters.retain(|candidate| *candidate != waiter);
        self.clear_binding_if_idle();
    }

    pub(crate) fn is_idle(&self) -> bool {
        self.waiters.is_empty()
    }

    pub(crate) fn clear_binding_if_idle(&mut self) {
        if self.waiters.is_empty() {
            self.bound_mutex = None;
        }
    }
}

#[derive(Debug)]
pub(crate) struct GuestConditionWaiterState {
    pub(crate) released: bool,
}

impl GuestSyncState {
    pub(crate) fn register_condition_waiter(&mut self, condition: u64, mutex: u64) -> Option<u64> {
        if self
            .conditions
            .get(&condition)
            .and_then(|condition| condition.bound_mutex)
            .is_some_and(|bound| bound != mutex)
        {
            return None;
        }
        let waiter = loop {
            self.next_condition_waiter = self.next_condition_waiter.wrapping_add(1).max(1);
            if !self
                .condition_waiters
                .contains_key(&self.next_condition_waiter)
            {
                break self.next_condition_waiter;
            }
        };
        assert!(
            self.conditions
                .entry(condition)
                .or_default()
                .register_waiter(waiter, mutex),
            "condition mutex binding was validated under the synchronization lock"
        );
        self.condition_waiters
            .insert(waiter, GuestConditionWaiterState { released: false });
        Some(waiter)
    }

    pub(crate) fn condition_waiter_released(&self, waiter: u64) -> bool {
        self.condition_waiters
            .get(&waiter)
            .is_some_and(|waiter| waiter.released)
    }

    pub(crate) fn finish_condition_wait(&mut self, condition: u64, waiter: u64, timed_out: bool) {
        if timed_out {
            self.conditions
                .get_mut(&condition)
                .expect("a pending condition waiter prevents destruction")
                .cancel_wait(waiter);
        }
        let waiter_state = self
            .condition_waiters
            .remove(&waiter)
            .expect("registered condition waiter survives condition destruction");
        assert_eq!(
            waiter_state.released, !timed_out,
            "timeout and notification have exactly one winner"
        );
    }
}

#[derive(Debug)]
pub(crate) struct GuestMutexState {
    pub(crate) owner: Option<ThreadId>,
    pub(crate) owner_guest_thread: u64,
    pub(crate) recursion: u32,
    pub(crate) mutex_type: i32,
    pub(crate) waiters: u32,
}

#[derive(Debug)]
pub(crate) struct GuestMutexEntry {
    pub(crate) state: Mutex<GuestMutexState>,
    pub(crate) wake: Condvar,
    pub(crate) storage_address: u64,
    /// Opaque object installed in the guest's `ScePthreadMutex` storage by the init ABI.
    ///
    /// The bundled libc writes bookkeeping fields in this object immediately after
    /// `scePthreadMutexInit` succeeds, even though lock ownership itself is modeled by REmu.
    /// Synthetic unit-test addresses are deliberately allowed to leave this zero.
    pub(crate) guest_object: u64,
}

#[derive(Debug, Default)]
pub(crate) struct GuestRwlockState {
    pub(crate) readers: HashMap<ThreadId, u32>,
    pub(crate) reader_total: u32,
    pub(crate) writer: Option<ThreadId>,
    pub(crate) waiting_writers: u32,
}

impl Default for GuestMutexState {
    fn default() -> Self {
        let attributes = PthreadMutexAttrState::default();
        Self {
            owner: None,
            owner_guest_thread: 0,
            recursion: 0,
            mutex_type: attributes.mutex_type,
            waiters: 0,
        }
    }
}

#[derive(Debug)]
pub(crate) struct GuestSemaphoreState {
    pub(crate) count: i32,
    pub(crate) maximum: i32,
}

#[derive(Debug)]
pub(crate) struct GuestEventFlagState {
    pub(crate) name: String,
    pub(crate) attributes: u32,
    pub(crate) bits: u64,
    pub(crate) waiters: u32,
    pub(crate) cancellation_generation: u64,
}

static GUEST_SYNC: OnceLock<(Mutex<GuestSyncState>, Condvar)> = OnceLock::new();

static EVENT_FLAG_WAIT_SEQUENCE: AtomicU64 = AtomicU64::new(0);

static EVENT_FLAG_SET_SEQUENCE: AtomicU64 = AtomicU64::new(0);

static MUTEX_CONTENTION_SEQUENCE: AtomicU64 = AtomicU64::new(0);

const GUEST_MUTEX_SHARD_COUNT: usize = 64;

static GUEST_MUTEX_SHARDS: OnceLock<
    [Mutex<HashMap<u64, Arc<GuestMutexEntry>>>; GUEST_MUTEX_SHARD_COUNT],
> = OnceLock::new();

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
        "27bAgiJmOh0" => posix_pthread_result(guest_condition_wait_with_interrupt(
            arguments[0],
            arguments[1],
            Some(relative_microseconds_until(arguments[2])),
            || {
                if let Some(frame) = frame {
                    deliver_pending_guest_exception_at_import(frame);
                }
            },
        )),
        "kDh-NfxgMtE" => guest_condition_notify(arguments[0], false),
        "2MOy+rUfuhQ" => posix_pthread_result(guest_condition_notify(arguments[0], false)),
        "JGgj7Uvrl+A" => guest_condition_notify(arguments[0], true),
        "mkx2fVhNMsg" => posix_pthread_result(guest_condition_notify(arguments[0], true)),
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

pub(crate) fn guest_sync() -> &'static (Mutex<GuestSyncState>, Condvar) {
    GUEST_SYNC.get_or_init(|| (Mutex::new(GuestSyncState::default()), Condvar::new()))
}

fn guest_mutex_shards()
-> &'static [Mutex<HashMap<u64, Arc<GuestMutexEntry>>>; GUEST_MUTEX_SHARD_COUNT] {
    GUEST_MUTEX_SHARDS.get_or_init(|| std::array::from_fn(|_| Mutex::new(HashMap::new())))
}

fn guest_mutex_shard(address: u64) -> usize {
    usize::try_from(address >> 4).unwrap_or_default() & (GUEST_MUTEX_SHARD_COUNT - 1)
}

pub(crate) fn guest_mutex_entry(address: u64, create: bool) -> Option<Arc<GuestMutexEntry>> {
    let shard = guest_mutex_shards()[guest_mutex_shard(address)]
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if let Some(entry) = shard.get(&address) {
        return Some(entry.clone());
    }
    drop(shard);
    for candidate in guest_mutex_shards() {
        let candidate = candidate
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Some(entry) = candidate
            .values()
            .find(|entry| entry.guest_object != 0 && entry.guest_object == address)
        {
            return Some(entry.clone());
        }
    }
    if !create {
        return None;
    }
    let mut shard = guest_mutex_shards()[guest_mutex_shard(address)]
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if let Some(entry) = shard.get(&address) {
        return Some(entry.clone());
    }
    create.then(|| {
        let entry = Arc::new(GuestMutexEntry {
            state: Mutex::new(GuestMutexState::default()),
            wake: Condvar::new(),
            storage_address: address,
            guest_object: 0,
        });
        shard.insert(address, entry.clone());
        entry
    })
}

pub(crate) fn guest_pthread_once(control: u64, routine: u64) -> u64 {
    const NEVER_DONE: u32 = 0;
    const DONE: u32 = 1;
    const IN_PROGRESS: u32 = 2;
    const WAIT: u32 = 3;
    if control == 0 || routine == 0 {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    }
    let sync = guest_sync();
    let mut state = sync
        .0
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    loop {
        match read_guest_u32(control) {
            DONE => return 0,
            NEVER_DONE => {
                write_guest_u32(control, IN_PROGRESS);
                break;
            }
            IN_PROGRESS | WAIT => {
                write_guest_u32(control, WAIT);
                state = sync
                    .1
                    .wait(state)
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
            }
            _ => return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT),
        }
    }
    drop(state);

    type OnceRoutine = unsafe extern "C" fn();
    // SAFETY: scePthreadOnce's ABI requires a live no-argument guest function pointer. No REmu
    // state lock is held while it runs, so initialization may call back through other imports.
    let routine: OnceRoutine = unsafe { std::mem::transmute(routine) };
    unsafe { routine() };

    let _state = sync
        .0
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    write_guest_u32(control, DONE);
    sync.1.notify_all();
    0
}

pub(crate) fn guest_mutex_init(address: u64, attributes: PthreadMutexAttrState) -> u64 {
    if address == 0 {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    }
    let guest_object = match allocate_guest_mutex_object(address, attributes) {
        Ok(object) => object,
        Err(error) => return error,
    };
    let mut shard = guest_mutex_shards()[guest_mutex_shard(address)]
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if shard.contains_key(&address) {
        if guest_object != 0 {
            guest_free(guest_object);
        }
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    }
    shard.insert(
        address,
        Arc::new(GuestMutexEntry {
            state: Mutex::new(GuestMutexState {
                mutex_type: attributes.mutex_type,
                ..GuestMutexState::default()
            }),
            wake: Condvar::new(),
            storage_address: address,
            guest_object,
        }),
    );
    if guest_object != 0 {
        write_guest_u64(address, guest_object);
    }
    0
}

fn allocate_guest_mutex_object(
    storage: u64,
    attributes: PthreadMutexAttrState,
) -> Result<u64, u64> {
    let mut existing = [0_u8; 8];
    if !ProcessGuestMemory.read(storage, &mut existing) {
        // Synchronization unit tests also exercise deliberately synthetic, non-dereferenceable
        // addresses. They only require the host-side ownership model; a real mapped guest mutex
        // must always take the object-backed path above.
        return Ok(0);
    }
    const GUEST_MUTEX_OBJECT_SIZE: u64 = 0x100;
    const GUEST_MUTEX_TYPE_OFFSET: u64 = 0x20;
    const GUEST_MUTEX_PROTOCOL_OFFSET: u64 = 0x3c;
    let object = guest_calloc(1, GUEST_MUTEX_OBJECT_SIZE);
    if object == 0 {
        return Err(sce_kernel_error(SCE_KERNEL_ERROR_TRY_AGAIN));
    }
    write_guest_u32(
        object + GUEST_MUTEX_TYPE_OFFSET,
        attributes.mutex_type.cast_unsigned(),
    );
    write_guest_u32(
        object + GUEST_MUTEX_PROTOCOL_OFFSET,
        attributes.protocol.cast_unsigned(),
    );
    Ok(object)
}

pub(crate) fn guest_mutex_destroy(address: u64) -> u64 {
    if address == 0 {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    }
    let Some(entry) = guest_mutex_entry(address, false) else {
        return sce_kernel_error(SCE_KERNEL_ERROR_NOT_FOUND);
    };
    let storage_address = entry.storage_address;
    let mut shard = guest_mutex_shards()[guest_mutex_shard(storage_address)]
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let Some(stored_entry) = shard.get(&storage_address).cloned() else {
        return sce_kernel_error(SCE_KERNEL_ERROR_NOT_FOUND);
    };
    if !Arc::ptr_eq(&entry, &stored_entry) {
        return sce_kernel_error(SCE_KERNEL_ERROR_NOT_FOUND);
    }
    let mutex = stored_entry
        .state
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if mutex.owner.is_some() || mutex.waiters != 0 {
        return u64::from(libc::EBUSY.cast_unsigned());
    }
    drop(mutex);
    let entry = shard
        .remove(&storage_address)
        .expect("the mutex entry checked above remains in its shard");
    let guest_object = entry.guest_object;
    drop(shard);
    if guest_object != 0 {
        write_guest_u64(storage_address, 0);
        guest_free(guest_object);
    }
    0
}

pub(crate) fn guest_mutex_lock(address: u64, try_only: bool) -> u64 {
    if address == 0 {
        return u64::MAX;
    }
    let Some(entry) = guest_mutex_entry(address, true) else {
        return u64::MAX;
    };
    let current = std::thread::current().id();
    let mut mutex = entry
        .state
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let guest_thread = current_guest_thread_handle();
    let mut reported_stall = false;
    loop {
        match mutex.owner.as_ref() {
            None => {
                mutex.owner = Some(current);
                mutex.owner_guest_thread = guest_thread;
                mutex.recursion = 1;
                return 0;
            }
            Some(owner) if *owner == current && mutex.mutex_type == 2 => {
                mutex.recursion = mutex.recursion.saturating_add(1);
                return 0;
            }
            Some(owner) if *owner == current && try_only => {
                return u64::from(libc::EBUSY.cast_unsigned());
            }
            // Error-checking and adaptive mutexes diagnose a same-thread relock. A normal mutex
            // deliberately falls through to the blocking path below and deadlocks, matching the
            // Gen5 pthread contract.
            Some(owner) if *owner == current && matches!(mutex.mutex_type, 1 | 4) => {
                return u64::from(libc::EDEADLK.cast_unsigned());
            }
            Some(_) if try_only => return u64::from(libc::EBUSY as u32),
            Some(_) => {
                mutex.waiters = mutex.waiters.saturating_add(1);
                let contention_sequence =
                    MUTEX_CONTENTION_SEQUENCE.fetch_add(1, Ordering::Relaxed) + 1;
                if trace_guest_sync() && trace_logarithmic_sequence(contention_sequence, 8) {
                    eprintln!(
                        "[REMu][MUTEX] contention sequence={contention_sequence} thread={:?} \
                         handle={guest_thread:#x} mutex={address:#x} type={} owner={:#x} \
                         recursion={} waiters={}",
                        current_guest_thread_name(),
                        mutex.mutex_type,
                        mutex.owner_guest_thread,
                        mutex.recursion,
                        mutex.waiters,
                    );
                }
                if let Some(report_after) = stall_report_interval() {
                    let (next_mutex, wait_result) = entry
                        .wake
                        .wait_timeout(mutex, report_after)
                        .unwrap_or_else(std::sync::PoisonError::into_inner);
                    mutex = next_mutex;
                    if wait_result.timed_out() && !reported_stall {
                        reported_stall = true;
                        eprintln!(
                            "[REMu][STALL] mutex lock exceeded {}s thread={:?} \
                             handle={guest_thread:#x} mutex={address:#x} \
                             owner={:#x} recursion={} waiters={}",
                            report_after.as_secs(),
                            current_guest_thread_name(),
                            mutex.owner_guest_thread,
                            mutex.recursion,
                            mutex.waiters,
                        );
                    }
                } else {
                    mutex = entry
                        .wake
                        .wait(mutex)
                        .unwrap_or_else(std::sync::PoisonError::into_inner);
                }
                mutex.waiters = mutex.waiters.saturating_sub(1);
            }
        }
    }
}

pub(crate) fn guest_mutex_unlock(address: u64) -> u64 {
    let Some(entry) = guest_mutex_entry(address, false) else {
        eprintln!(
            "[REMu][MUTEX] rejected unlock of unknown mutex thread={:?} handle={:#x} mutex={address:#x}",
            current_guest_thread_name(),
            current_guest_thread_handle(),
        );
        return sce_kernel_error(SCE_KERNEL_ERROR_OPERATION_NOT_PERMITTED);
    };
    let current = std::thread::current().id();
    let mut mutex = entry
        .state
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if !mutex.owner.as_ref().is_some_and(|owner| *owner == current) {
        eprintln!(
            "[REMu][MUTEX] rejected non-owner unlock thread={:?} handle={:#x} mutex={address:#x} owner={:#x}",
            current_guest_thread_name(),
            current_guest_thread_handle(),
            mutex.owner_guest_thread,
        );
        return sce_kernel_error(SCE_KERNEL_ERROR_OPERATION_NOT_PERMITTED);
    }
    if mutex.recursion > 1 {
        mutex.recursion -= 1;
    } else {
        mutex.owner = None;
        mutex.owner_guest_thread = 0;
        mutex.recursion = 0;
        entry.wake.notify_one();
    }
    0
}

pub(crate) fn guest_rwlock_init(address: u64) -> u64 {
    if address == 0 {
        return u64::from(libc::EINVAL.cast_unsigned());
    }
    let sync = guest_sync();
    let mut state = sync
        .0
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if state.rwlocks.contains_key(&address) {
        return u64::from(libc::EBUSY.cast_unsigned());
    }
    state.rwlocks.insert(address, GuestRwlockState::default());
    0
}

pub(crate) fn guest_rwlock_destroy(address: u64) -> u64 {
    if address == 0 {
        return u64::from(libc::EINVAL.cast_unsigned());
    }
    let sync = guest_sync();
    let mut state = sync
        .0
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    match state.rwlocks.get(&address) {
        Some(lock)
            if lock.writer.is_some() || lock.reader_total != 0 || lock.waiting_writers != 0 =>
        {
            u64::from(libc::EBUSY.cast_unsigned())
        }
        Some(_) => {
            state.rwlocks.remove(&address);
            0
        }
        None => u64::from(libc::EINVAL.cast_unsigned()),
    }
}

pub(crate) fn guest_rwlock_lock(address: u64, write: bool, try_only: bool) -> u64 {
    if address == 0 {
        return u64::from(libc::EINVAL.cast_unsigned());
    }
    let sync = guest_sync();
    let current = std::thread::current().id();
    let mut state = sync
        .0
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    state.rwlocks.entry(address).or_default();

    if write {
        if state
            .rwlocks
            .get(&address)
            .is_some_and(|lock| lock.writer.as_ref() == Some(&current))
        {
            return u64::from(libc::EDEADLK.cast_unsigned());
        }
        let acquired = state
            .rwlocks
            .get(&address)
            .is_some_and(|lock| lock.writer.is_none() && lock.reader_total == 0);
        if acquired {
            state
                .rwlocks
                .get_mut(&address)
                .expect("rwlock was initialized")
                .writer = Some(current);
            return 0;
        }
        if try_only {
            return u64::from(libc::EBUSY.cast_unsigned());
        }
        state
            .rwlocks
            .get_mut(&address)
            .expect("rwlock was initialized")
            .waiting_writers += 1;
        loop {
            let acquired = state
                .rwlocks
                .get(&address)
                .is_some_and(|lock| lock.writer.is_none() && lock.reader_total == 0);
            if acquired {
                let lock = state
                    .rwlocks
                    .get_mut(&address)
                    .expect("a waiting writer prevents rwlock destruction");
                lock.waiting_writers -= 1;
                lock.writer = Some(current);
                return 0;
            }
            state = sync
                .1
                .wait(state)
                .unwrap_or_else(std::sync::PoisonError::into_inner);
        }
    }

    loop {
        let acquired = state
            .rwlocks
            .get(&address)
            .is_some_and(|lock| lock.writer.is_none() && lock.waiting_writers == 0);
        if acquired {
            let lock = state
                .rwlocks
                .get_mut(&address)
                .expect("rwlock was initialized");
            *lock.readers.entry(current).or_default() += 1;
            lock.reader_total += 1;
            return 0;
        }
        if try_only {
            return u64::from(libc::EBUSY.cast_unsigned());
        }
        state = sync
            .1
            .wait(state)
            .unwrap_or_else(std::sync::PoisonError::into_inner);
    }
}

pub(crate) fn guest_rwlock_unlock(address: u64) -> u64 {
    let sync = guest_sync();
    let current = std::thread::current().id();
    let mut state = sync
        .0
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let Some(lock) = state.rwlocks.get_mut(&address) else {
        return u64::from(libc::EINVAL.cast_unsigned());
    };
    if lock.writer.as_ref() == Some(&current) {
        lock.writer = None;
        sync.1.notify_all();
        return 0;
    }
    let Some(recursion) = lock.readers.get_mut(&current) else {
        return u64::from(libc::EPERM.cast_unsigned());
    };
    *recursion -= 1;
    lock.reader_total -= 1;
    if *recursion == 0 {
        lock.readers.remove(&current);
    }
    if lock.reader_total == 0 {
        sync.1.notify_all();
    }
    0
}

pub(crate) fn guest_condition_init(address: u64) -> u64 {
    if address == 0 {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    }
    let sync = guest_sync();
    sync.0
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .conditions
        .entry(address)
        .or_default();
    0
}

pub(crate) fn guest_condition_destroy(address: u64) -> u64 {
    if address == 0 {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    }
    let sync = guest_sync();
    let mut state = sync
        .0
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    match state.conditions.get(&address) {
        Some(condition) if !condition.is_idle() => sce_kernel_error(SCE_KERNEL_ERROR_BUSY),
        Some(_) => {
            state.conditions.remove(&address);
            0
        }
        None => sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT),
    }
}

#[cfg(test)]
pub(crate) fn guest_condition_wait(
    condition: u64,
    mutex_address: u64,
    timeout_usec: Option<u64>,
) -> u64 {
    guest_condition_wait_with_interrupt(condition, mutex_address, timeout_usec, || {})
}

pub(crate) fn guest_condition_wait_with_interrupt<F>(
    condition: u64,
    mutex_address: u64,
    timeout_usec: Option<u64>,
    mut on_interrupt: F,
) -> u64
where
    F: FnMut(),
{
    if condition == 0 || mutex_address == 0 {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    }
    let sync = guest_sync();
    let current = std::thread::current().id();
    let Some(mutex_entry) = guest_mutex_entry(mutex_address, false) else {
        return sce_kernel_error(SCE_KERNEL_ERROR_OPERATION_NOT_PERMITTED);
    };
    let mut mutex = mutex_entry
        .state
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let released_recursion = if mutex.owner.as_ref().is_some_and(|owner| *owner == current) {
        mutex.recursion.max(1)
    } else {
        return sce_kernel_error(SCE_KERNEL_ERROR_OPERATION_NOT_PERMITTED);
    };
    let mut state = sync
        .0
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let Some(waiter) = state.register_condition_waiter(condition, mutex_address) else {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    };
    if trace_guest_sync() && trace_logarithmic_sequence(waiter, 8) {
        eprintln!(
            "[REMu][COND] wait ticket={waiter} thread={} condition={condition:#x} mutex={mutex_address:#x} recursion={released_recursion}",
            current_guest_thread_handle()
        );
    }
    mutex.owner = None;
    mutex.owner_guest_thread = 0;
    mutex.recursion = 0;
    mutex_entry.wake.notify_one();
    drop(mutex);
    let guest_thread = current_guest_thread_handle();

    // Set when a diagnostic spurious wake retires an otherwise unbounded wait.
    let mut released_spuriously = false;
    let (timed_out, interrupted) = if let Some(timeout) = timeout_usec {
        let duration = Duration::from_micros(timeout);
        let (next_state, wait_result) = sync
            .1
            .wait_timeout_while(state, duration, |state| {
                !state.condition_waiter_released(waiter)
                    && !guest_exception_is_pending(guest_thread)
            })
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        state = next_state;
        let interrupted =
            !state.condition_waiter_released(waiter) && guest_exception_is_pending(guest_thread);
        let expired =
            wait_result.timed_out() && !state.condition_waiter_released(waiter) && !interrupted;
        if expired {
            let guest_condition = state.conditions.entry(condition).or_default();
            guest_condition.timeouts = guest_condition.timeouts.saturating_add(1);
        }
        (expired, interrupted)
    } else if let Some(report_after) = stall_report_interval() {
        // An unbounded wait that never completes is the hardest stall class to attribute, because
        // the thread is parked with no record of what it wanted. Wake periodically to name the
        // wait once; the predicate is unchanged, so this cannot alter wait semantics.
        let mut reported = false;
        loop {
            let (next_state, wait_result) = sync
                .1
                .wait_timeout_while(state, report_after, |state| {
                    !state.condition_waiter_released(waiter)
                        && !guest_exception_is_pending(guest_thread)
                })
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            state = next_state;
            if !wait_result.timed_out() {
                break;
            }
            if stall_break_enabled() {
                // POSIX permits spurious wakeups, so releasing a stalled waiter cannot corrupt a
                // correct guest: it re-tests its predicate and waits again. Whether it proceeds
                // distinguishes a lost wakeup from a predicate that was never satisfied.
                eprintln!(
                    "[REMu][STALL] releasing stalled wait thread={:?} condition={condition:#x} ticket={waiter}",
                    current_guest_thread_name()
                );
                released_spuriously = true;
                break;
            }
            if !reported {
                reported = true;
                let notifies = state
                    .conditions
                    .get(&condition)
                    .map_or(0, |guest_condition| guest_condition.notifies);
                eprintln!(
                    "[REMu][STALL] condition wait exceeded {}s thread={:?} handle={guest_thread:#x} ticket={waiter} condition={condition:#x} mutex={mutex_address:#x} notifies={notifies}",
                    report_after.as_secs(),
                    current_guest_thread_name()
                );
                if !STALL_SNAPSHOT_EMITTED.swap(true, Ordering::Relaxed) {
                    // One process-wide view of every condition: which are being signaled, which
                    // are only ever polled, and which have waiters nobody serves.
                    let mut summary: Vec<_> = state
                        .conditions
                        .iter()
                        .map(|(address, guest_condition)| {
                            (
                                *address,
                                guest_condition.waiters.len(),
                                guest_condition.notifies,
                                guest_condition.timeouts,
                            )
                        })
                        .collect();
                    summary.sort_by_key(|(_, waiters, notifies, timeouts)| {
                        (
                            std::cmp::Reverse(*timeouts),
                            std::cmp::Reverse(*waiters),
                            *notifies,
                        )
                    });
                    for (address, waiters, notifies, timeouts) in summary.iter().take(24) {
                        eprintln!(
                            "[REMu][STALL] condition={address:#x} waiters={waiters} notifies={notifies} timeouts={timeouts}"
                        );
                    }
                    report_kernel_event_registrations();
                }
            }
        }
        let interrupted =
            !state.condition_waiter_released(waiter) && guest_exception_is_pending(guest_thread);
        (false, interrupted)
    } else {
        state = sync
            .1
            .wait_while(state, |state| {
                !state.condition_waiter_released(waiter)
                    && !guest_exception_is_pending(guest_thread)
            })
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let interrupted =
            !state.condition_waiter_released(waiter) && guest_exception_is_pending(guest_thread);
        (false, interrupted)
    };
    state.finish_condition_wait(
        condition,
        waiter,
        timed_out || interrupted || released_spuriously,
    );
    drop(state);

    if interrupted {
        // A cooperative guest exception is analogous to a signal interrupting pthread_cond_wait.
        // Run the handler before trying to reacquire the guest mutex: a stop-the-world handler can
        // suspend the mutex owner, so reacquiring first creates a circular suspension deadlock.
        // Once the handler returns (the guest has resumed), normal cond-wait semantics still
        // require this thread to reacquire the mutex before returning to guest code.
        on_interrupt();
    }

    let mut mutex = mutex_entry
        .state
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    loop {
        if mutex.owner.is_none() {
            mutex.owner = Some(current);
            mutex.owner_guest_thread = guest_thread;
            mutex.recursion = released_recursion;
            return if timed_out {
                sce_kernel_error(SCE_KERNEL_ERROR_TIMED_OUT)
            } else {
                0
            };
        }
        mutex.waiters = mutex.waiters.saturating_add(1);
        mutex = mutex_entry
            .wake
            .wait(mutex)
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        mutex.waiters = mutex.waiters.saturating_sub(1);
    }
}

pub(crate) fn guest_condition_notify(condition: u64, broadcast: bool) -> u64 {
    if condition == 0 {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    }
    let sync = guest_sync();
    let mut state = sync
        .0
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    state.condition_notify_count = state.condition_notify_count.saturating_add(1);
    let notify_sequence = state.condition_notify_count;
    let guest_condition = state.conditions.entry(condition).or_default();
    guest_condition.notifies = guest_condition.notifies.saturating_add(1);
    let pending_before = guest_condition.waiters.len();
    let selected = if broadcast {
        guest_condition.release_all()
    } else {
        guest_condition.release_one().into_iter().collect()
    };
    for waiter in &selected {
        state
            .condition_waiters
            .get_mut(waiter)
            .expect("pending condition waiter exists")
            .released = true;
    }
    if trace_guest_sync() && trace_logarithmic_sequence(notify_sequence, 8) {
        eprintln!(
            "[REMu][COND] notify sequence={notify_sequence} condition={condition:#x} broadcast={broadcast} pending={pending_before} selected={selected:?}"
        );
    }
    if !selected.is_empty() {
        // One host Condvar coordinates all guest synchronization objects. Wake all host waiters so
        // the selected condition ticket cannot be consumed by an unrelated host waiter; only the
        // released guest ticket is allowed to proceed.
        sync.1.notify_all();
    }
    0
}

fn guest_semaphore_create(arguments: [u64; 6]) -> u64 {
    let output = arguments[0];
    let initial = low_i32(arguments[3]);
    let maximum = low_i32(arguments[4]);
    if output == 0 || initial < 0 || maximum <= 0 || initial > maximum {
        return u64::MAX;
    }
    let sync = guest_sync();
    let mut state = sync
        .0
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    state.next_semaphore_handle = state.next_semaphore_handle.wrapping_add(1).max(1);
    let handle = state.next_semaphore_handle;
    state.semaphores.insert(
        handle,
        GuestSemaphoreState {
            count: initial,
            maximum,
        },
    );
    // SAFETY: sceKernelCreateSema supplies a writable 32-bit handle output.
    unsafe {
        std::ptr::write_unaligned(output as *mut u32, handle);
    }
    if trace_guest_sync() {
        eprintln!(
            "[REMu][SEMA] create handle={handle} name={:?} initial={initial} maximum={maximum}",
            read_guest_cstring(arguments[1]).unwrap_or_default()
        );
    }
    0
}

pub(crate) fn guest_semaphore_wait(arguments: [u64; 6]) -> u64 {
    let handle = low_u32(arguments[0]);
    let needed = low_i32(arguments[1]);
    if needed <= 0 {
        return u64::MAX;
    }
    let sync = guest_sync();
    let guest_thread = current_guest_thread_handle();
    let mut state = sync
        .0
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    state.semaphore_wait_count = state.semaphore_wait_count.saturating_add(1);
    let wait_sequence = state.semaphore_wait_count;
    let trace_call = trace_guest_sync() && trace_logarithmic_sequence(wait_sequence, 8);
    let mut logged_block = false;
    loop {
        let Some(semaphore) = state.semaphores.get_mut(&handle) else {
            return u64::MAX;
        };
        if semaphore.count >= needed {
            semaphore.count -= needed;
            if trace_call {
                eprintln!(
                    "[REMu][SEMA] acquire sequence={wait_sequence} handle={handle} needed={needed} remaining={}",
                    semaphore.count
                );
            }
            return 0;
        }
        if guest_exception_is_pending(guest_thread) {
            return sce_kernel_error(SCE_KERNEL_ERROR_INTERRUPTED);
        }
        if trace_call && !logged_block {
            eprintln!(
                "[REMu][SEMA] wait sequence={wait_sequence} handle={handle} needed={needed} available={}",
                semaphore.count
            );
            logged_block = true;
        }
        state = sync
            .1
            .wait(state)
            .unwrap_or_else(std::sync::PoisonError::into_inner);
    }
}

fn guest_semaphore_delete(handle: u64) -> u64 {
    let sync = guest_sync();
    let mut state = sync
        .0
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if state.semaphores.remove(&low_u32(handle)).is_none() {
        return sce_kernel_error(SCE_KERNEL_ERROR_NOT_FOUND);
    }
    sync.1.notify_all();
    0
}

fn guest_semaphore_signal(arguments: [u64; 6]) -> u64 {
    let handle = low_u32(arguments[0]);
    let amount = low_i32(arguments[1]);
    let sync = guest_sync();
    let mut state = sync
        .0
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    state.semaphore_signal_count = state.semaphore_signal_count.saturating_add(1);
    let signal_sequence = state.semaphore_signal_count;
    let Some(semaphore) = state.semaphores.get_mut(&handle) else {
        return u64::MAX;
    };
    semaphore.count = semaphore
        .count
        .saturating_add(amount)
        .min(semaphore.maximum);
    if trace_guest_sync() && trace_logarithmic_sequence(signal_sequence, 8) {
        eprintln!(
            "[REMu][SEMA] signal sequence={signal_sequence} handle={handle} amount={amount} available={}",
            semaphore.count
        );
    }
    sync.1.notify_all();
    0
}

const EVENT_FLAG_WAIT_AND: u32 = 0x01;

pub(crate) const EVENT_FLAG_WAIT_OR: u32 = 0x02;

const EVENT_FLAG_CLEAR_ALL: u32 = 0x10;

const EVENT_FLAG_CLEAR_PATTERN: u32 = 0x20;

fn event_flag_attributes_valid(attributes: u32) -> bool {
    matches!(attributes & 0x0f, 0 | 0x01 | 0x02)
        && matches!(attributes & 0xf0, 0 | 0x10 | 0x20)
        && attributes & !0x33 == 0
}

fn event_flag_wait_mode_valid(wait_mode: u32) -> bool {
    matches!(wait_mode & 0x0f, EVENT_FLAG_WAIT_AND | EVENT_FLAG_WAIT_OR)
        && matches!(
            wait_mode & 0xf0,
            0 | EVENT_FLAG_CLEAR_ALL | EVENT_FLAG_CLEAR_PATTERN
        )
        && wait_mode & !0x33 == 0
}

fn event_flag_satisfied(bits: u64, pattern: u64, wait_mode: u32) -> bool {
    if wait_mode & 0x0f == EVENT_FLAG_WAIT_AND {
        bits & pattern == pattern
    } else {
        bits & pattern != 0
    }
}

fn event_flag_apply_clear(bits: &mut u64, pattern: u64, wait_mode: u32) {
    match wait_mode & 0xf0 {
        EVENT_FLAG_CLEAR_ALL => *bits = 0,
        EVENT_FLAG_CLEAR_PATTERN => *bits &= !pattern,
        _ => {}
    }
}

fn guest_event_flag_create(arguments: [u64; 6]) -> u64 {
    let output = arguments[0];
    let Some(name) = read_guest_cstring(arguments[1]) else {
        return sce_kernel_error(SCE_KERNEL_ERROR_MEMORY_FAULT);
    };
    let attributes = low_u32(arguments[2]);
    if output == 0
        || arguments[1] == 0
        || arguments[4] != 0
        || name.len() > 31
        || !event_flag_attributes_valid(attributes)
    {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    }
    let sync = guest_sync();
    let mut state = sync
        .0
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let handle = loop {
        state.next_event_flag_handle = state.next_event_flag_handle.wrapping_add(1).max(1);
        if !state
            .event_flags
            .contains_key(&state.next_event_flag_handle)
        {
            break state.next_event_flag_handle;
        }
    };
    state.event_flags.insert(
        handle,
        GuestEventFlagState {
            name: name.clone(),
            attributes,
            bits: arguments[3],
            waiters: 0,
            cancellation_generation: 0,
        },
    );
    write_guest_u64(output, handle);
    if trace_guest_sync() {
        eprintln!(
            "[REMu][EVENT_FLAG] create handle={handle} name={name:?} attr={attributes:#x} bits={:#x} thread={:?}/{:#x}",
            arguments[3],
            current_guest_thread_name(),
            current_guest_thread_handle(),
        );
    }
    0
}

fn guest_event_flag_delete(handle: u64) -> u64 {
    let sync = guest_sync();
    let mut state = sync
        .0
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let Some(flag) = state.event_flags.remove(&handle) else {
        return sce_kernel_error(SCE_KERNEL_ERROR_NOT_FOUND);
    };
    if trace_guest_sync() {
        eprintln!(
            "[REMu][EVENT_FLAG] delete handle={handle} name={:?} attr={:#x} waiters={} thread={:?}/{:#x}",
            flag.name,
            flag.attributes,
            flag.waiters,
            current_guest_thread_name(),
            current_guest_thread_handle(),
        );
    }
    sync.1.notify_all();
    0
}

fn guest_event_flag_set(handle: u64, pattern: u64) -> u64 {
    let sync = guest_sync();
    let mut state = sync
        .0
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let Some(flag) = state.event_flags.get_mut(&handle) else {
        return sce_kernel_error(SCE_KERNEL_ERROR_NOT_FOUND);
    };
    flag.bits |= pattern;
    let bits = flag.bits;
    let name = flag.name.clone();
    let waiters = flag.waiters;
    if trace_guest_sync() {
        let sequence = EVENT_FLAG_SET_SEQUENCE
            .fetch_add(1, Ordering::Relaxed)
            .saturating_add(1);
        if sequence <= 32 || sequence.is_power_of_two() {
            eprintln!(
                "[REMu][EVENT_FLAG] set sequence={sequence} handle={handle} name={name:?} pattern={pattern:#x} bits={bits:#x} waiters={waiters} thread={:?}/{:#x}",
                current_guest_thread_name(),
                current_guest_thread_handle(),
            );
        }
    }
    sync.1.notify_all();
    0
}

fn guest_event_flag_clear(handle: u64, mask: u64) -> u64 {
    let sync = guest_sync();
    let mut state = sync
        .0
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let Some(flag) = state.event_flags.get_mut(&handle) else {
        return sce_kernel_error(SCE_KERNEL_ERROR_NOT_FOUND);
    };
    flag.bits &= mask;
    0
}

fn guest_event_flag_poll(arguments: [u64; 6]) -> u64 {
    let pattern = arguments[1];
    let wait_mode = low_u32(arguments[2]);
    if pattern == 0 || !event_flag_wait_mode_valid(wait_mode) {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    }
    let sync = guest_sync();
    let mut state = sync
        .0
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let Some(flag) = state.event_flags.get_mut(&arguments[0]) else {
        return sce_kernel_error(SCE_KERNEL_ERROR_NOT_FOUND);
    };
    if arguments[3] != 0 {
        write_guest_u64(arguments[3], flag.bits);
    }
    if !event_flag_satisfied(flag.bits, pattern, wait_mode) {
        return sce_kernel_error(SCE_KERNEL_ERROR_BUSY);
    }
    event_flag_apply_clear(&mut flag.bits, pattern, wait_mode);
    0
}

pub(crate) fn guest_event_flag_wait(arguments: [u64; 6]) -> u64 {
    let handle = arguments[0];
    let pattern = arguments[1];
    let wait_mode = low_u32(arguments[2]);
    if pattern == 0 || !event_flag_wait_mode_valid(wait_mode) {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    }
    let timeout = (arguments[4] != 0).then(|| u64::from(read_guest_u32(arguments[4])));
    let deadline = timeout
        .and_then(|microseconds| Instant::now().checked_add(Duration::from_micros(microseconds)));
    let sync = guest_sync();
    let guest_thread = current_guest_thread_handle();
    let mut state = sync
        .0
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let Some(flag) = state.event_flags.get_mut(&handle) else {
        return sce_kernel_error(SCE_KERNEL_ERROR_NOT_FOUND);
    };
    let wait_sequence = EVENT_FLAG_WAIT_SEQUENCE
        .fetch_add(1, Ordering::Relaxed)
        .saturating_add(1);
    let trace_wait = trace_guest_sync() && (wait_sequence <= 32 || wait_sequence.is_power_of_two());
    if trace_wait {
        eprintln!(
            "[REMu][EVENT_FLAG] wait sequence={wait_sequence} handle={handle} name={:?} pattern={pattern:#x} mode={wait_mode:#x} bits={:#x} timeout_us={timeout:?} thread={:?}/{guest_thread:#x}",
            flag.name,
            flag.bits,
            current_guest_thread_name(),
        );
    }
    let cancellation_generation = flag.cancellation_generation;
    if event_flag_satisfied(flag.bits, pattern, wait_mode) {
        if arguments[3] != 0 {
            write_guest_u64(arguments[3], flag.bits);
        }
        event_flag_apply_clear(&mut flag.bits, pattern, wait_mode);
        if trace_wait {
            eprintln!(
                "[REMu][EVENT_FLAG] wait satisfied sequence={wait_sequence} handle={handle} immediately=true remaining={:#x}",
                flag.bits
            );
        }
        return 0;
    }
    flag.waiters = flag.waiters.saturating_add(1);
    loop {
        let Some(flag) = state.event_flags.get_mut(&handle) else {
            return sce_kernel_error(SCE_KERNEL_ERROR_NOT_FOUND);
        };
        if guest_exception_is_pending(guest_thread) {
            if arguments[3] != 0 {
                write_guest_u64(arguments[3], flag.bits);
            }
            flag.waiters = flag.waiters.saturating_sub(1);
            return sce_kernel_error(SCE_KERNEL_ERROR_INTERRUPTED);
        }
        if flag.cancellation_generation != cancellation_generation {
            flag.waiters = flag.waiters.saturating_sub(1);
            return sce_kernel_error(SCE_KERNEL_ERROR_INTERRUPTED);
        }
        if event_flag_satisfied(flag.bits, pattern, wait_mode) {
            if arguments[3] != 0 {
                write_guest_u64(arguments[3], flag.bits);
            }
            event_flag_apply_clear(&mut flag.bits, pattern, wait_mode);
            flag.waiters = flag.waiters.saturating_sub(1);
            if trace_wait {
                eprintln!(
                    "[REMu][EVENT_FLAG] wait satisfied sequence={wait_sequence} handle={handle} immediately=false remaining={:#x}",
                    flag.bits
                );
            }
            return 0;
        }
        if let Some(deadline) = deadline {
            let Some(remaining) = deadline.checked_duration_since(Instant::now()) else {
                if arguments[3] != 0 {
                    write_guest_u64(arguments[3], flag.bits);
                }
                if arguments[4] != 0 {
                    write_guest_u32(arguments[4], 0);
                }
                flag.waiters = flag.waiters.saturating_sub(1);
                return sce_kernel_error(SCE_KERNEL_ERROR_TIMED_OUT);
            };
            let (next_state, wait_result) = sync
                .1
                .wait_timeout(state, remaining)
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            state = next_state;
            if wait_result.timed_out() {
                continue;
            }
        } else {
            state = sync
                .1
                .wait(state)
                .unwrap_or_else(std::sync::PoisonError::into_inner);
        }
    }
}

fn guest_event_flag_cancel(arguments: [u64; 6]) -> u64 {
    let sync = guest_sync();
    let mut state = sync
        .0
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let Some(flag) = state.event_flags.get_mut(&arguments[0]) else {
        return sce_kernel_error(SCE_KERNEL_ERROR_NOT_FOUND);
    };
    if arguments[2] != 0 {
        write_guest_u32(arguments[2], flag.waiters);
    }
    flag.bits = arguments[1];
    flag.cancellation_generation = flag.cancellation_generation.wrapping_add(1);
    sync.1.notify_all();
    0
}
