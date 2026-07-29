use std::collections::BTreeMap;
use std::sync::atomic::Ordering;

use crate::{
    GUEST_PTHREAD_SPECIFIC, MACOS_GUEST_PAGE_ALIGNMENT, NEXT_PTHREAD_KEY_GENERATION,
    PROBE_DISPATCH, ProbeDispatchState, PthreadAttrObject, PthreadAttrState, PthreadKeyState,
    PthreadMutexAttrObject, PthreadMutexAttrState, SCE_KERNEL_ERROR_INVALID_ARGUMENT,
    SCE_KERNEL_ERROR_NOT_FOUND, SCE_KERNEL_ERROR_TRY_AGAIN, current_guest_thread_handle,
    guest_calloc, guest_free, low_i32, low_u32, pthread_key_generations, read_guest_cstring,
    read_guest_u32, sce_kernel_error, write_guest_u32, write_guest_u64,
};

pub(crate) fn create_pthread_attr(
    address: u64,
    attributes: PthreadAttrState,
    state: &mut ProbeDispatchState,
) -> u64 {
    if address == 0 {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    }
    if let Some(previous) = state.pthread_attributes.remove(&address) {
        guest_free(previous.handle);
    }
    let handle = guest_calloc(1, 0x40);
    if handle == 0 {
        return sce_kernel_error(SCE_KERNEL_ERROR_TRY_AGAIN);
    }
    write_pthread_attr_object(handle, attributes);
    write_guest_u64(address, handle);
    state
        .pthread_attributes
        .insert(address, PthreadAttrObject { handle, attributes });
    0
}

fn resolve_pthread_attr_key(address: u64, state: &ProbeDispatchState) -> Option<u64> {
    if address == 0 {
        return None;
    }
    if state.pthread_attributes.contains_key(&address) {
        return Some(address);
    }
    state
        .pthread_attributes
        .iter()
        .find_map(|(key, object)| (object.handle == address).then_some(*key))
}

pub(crate) fn pthread_attr_state(
    address: u64,
    state: &ProbeDispatchState,
) -> Option<PthreadAttrState> {
    let key = resolve_pthread_attr_key(address, state)?;
    state
        .pthread_attributes
        .get(&key)
        .map(|object| object.attributes)
}

fn write_pthread_attr_object(address: u64, attributes: PthreadAttrState) {
    write_guest_u64(address, attributes.affinity_mask);
    write_guest_u32(address + 0x08, attributes.detach_state.cast_unsigned());
    write_guest_u64(address + 0x10, attributes.stack_address);
    write_guest_u64(address + 0x18, attributes.stack_size);
    write_guest_u64(address + 0x20, attributes.guard_size);
    write_guest_u32(address + 0x28, attributes.inherit_schedule.cast_unsigned());
    write_guest_u32(address + 0x2c, attributes.schedule_policy.cast_unsigned());
    write_guest_u32(address + 0x30, attributes.schedule_priority.cast_unsigned());
}

pub(crate) fn update_pthread_attr(
    address: u64,
    state: &mut ProbeDispatchState,
    update: impl FnOnce(&mut PthreadAttrState),
) -> u64 {
    let Some(key) = resolve_pthread_attr_key(address, state) else {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    };
    let object = state
        .pthread_attributes
        .get_mut(&key)
        .expect("the pthread attribute key was just resolved");
    update(&mut object.attributes);
    write_pthread_attr_object(object.handle, object.attributes);
    0
}

pub(crate) fn destroy_pthread_attr(address: u64, state: &mut ProbeDispatchState) -> u64 {
    let Some(key) = resolve_pthread_attr_key(address, state) else {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    };
    let object = state
        .pthread_attributes
        .remove(&key)
        .expect("the pthread attribute key was just resolved");
    write_guest_u64(key, 0);
    guest_free(object.handle);
    0
}

pub(crate) fn pthread_attr_get(arguments: [u64; 6], state: &mut ProbeDispatchState) -> u64 {
    let thread = arguments[0];
    let output = arguments[1];
    if thread == 0 || output == 0 {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    }
    let Some(mut attributes) = state.thread_attributes.get(&thread).copied() else {
        return sce_kernel_error(SCE_KERNEL_ERROR_NOT_FOUND);
    };
    if thread == current_guest_thread_handle()
        && let Some((stack_address, stack_size)) = current_native_stack_range()
    {
        attributes.stack_address = stack_address;
        attributes.stack_size = stack_size;
        state.thread_attributes.insert(thread, attributes);
    }
    create_pthread_attr(output, attributes, state)
}

#[cfg(target_os = "macos")]
fn current_native_stack_range() -> Option<(u64, u64)> {
    // SAFETY: pthread_self returns the current live pthread; the two query functions only inspect
    // its immutable stack bounds.
    let thread = unsafe { libc::pthread_self() };
    // SAFETY: `thread` is the current live pthread.
    let top = unsafe { libc::pthread_get_stackaddr_np(thread) } as usize;
    // SAFETY: `thread` is the current live pthread.
    let size = unsafe { libc::pthread_get_stacksize_np(thread) };
    let base = top.checked_sub(size)?;
    Some((u64::try_from(base).ok()?, u64::try_from(size).ok()?))
}

#[cfg(not(target_os = "macos"))]
const fn current_native_stack_range() -> Option<(u64, u64)> {
    None
}

pub(crate) fn pthread_attr_get_stack(arguments: [u64; 6], state: &ProbeDispatchState) -> u64 {
    if arguments[1] == 0 || arguments[2] == 0 {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    }
    let Some(attributes) = pthread_attr_state(arguments[0], state) else {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    };
    write_guest_u64(arguments[1], attributes.stack_address);
    write_guest_u64(arguments[2], attributes.stack_size);
    0
}

pub(crate) fn pthread_attr_get_stack_address(
    arguments: [u64; 6],
    state: &ProbeDispatchState,
) -> u64 {
    if arguments[1] == 0 {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    }
    let Some(attributes) = pthread_attr_state(arguments[0], state) else {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    };
    write_guest_u64(arguments[1], attributes.stack_address);
    attributes.stack_address
}

pub(crate) fn pthread_attr_get_stack_size(arguments: [u64; 6], state: &ProbeDispatchState) -> u64 {
    if arguments[1] == 0 {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    }
    let Some(attributes) = pthread_attr_state(arguments[0], state) else {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    };
    write_guest_u64(arguments[1], attributes.stack_size);
    attributes.stack_size
}

pub(crate) fn pthread_attr_get_affinity(arguments: [u64; 6], state: &ProbeDispatchState) -> u64 {
    if arguments[1] == 0 {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    }
    let Some(attributes) = pthread_attr_state(arguments[0], state) else {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    };
    write_guest_u64(arguments[1], attributes.affinity_mask);
    // The Gen5 kernel ABI returns the observable mask in RAX as well as through the output pointer.
    attributes.affinity_mask
}

pub(crate) fn pthread_attr_set_affinity(
    arguments: [u64; 6],
    state: &mut ProbeDispatchState,
) -> u64 {
    // Sony's scePthread form takes the 64-bit CPU mask directly as its second argument. This is
    // distinct from POSIX pthread_attr_setaffinity_np(attr, cpusetsize, cpuset), and interpreting
    // the mask as a byte count rejects the single-core masks used by FMOD's mixer threads.
    update_pthread_attr(arguments[0], state, |attributes| {
        attributes.affinity_mask = arguments[1];
    })
}

pub(crate) fn pthread_attr_set_detach_state(
    arguments: [u64; 6],
    state: &mut ProbeDispatchState,
) -> u64 {
    let detach_state = low_i32(arguments[1]);
    if !matches!(detach_state, 0 | 1) {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    }
    update_pthread_attr(arguments[0], state, |attributes| {
        attributes.detach_state = detach_state;
    })
}

pub(crate) fn pthread_attr_set_stack_size(
    arguments: [u64; 6],
    state: &mut ProbeDispatchState,
) -> u64 {
    let stack_size = arguments[1];
    if stack_size < MACOS_GUEST_PAGE_ALIGNMENT || usize::try_from(stack_size).is_err() {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    }
    update_pthread_attr(arguments[0], state, |attributes| {
        attributes.stack_size = stack_size;
        attributes.stack_address = 0;
    })
}

pub(crate) fn pthread_attr_set_schedule_param(
    arguments: [u64; 6],
    state: &mut ProbeDispatchState,
) -> u64 {
    if arguments[1] == 0 {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    }
    let priority = read_guest_u32(arguments[1]).cast_signed();
    update_pthread_attr(arguments[0], state, |attributes| {
        attributes.schedule_priority = priority;
    })
}

pub(crate) fn pthread_attr_get_schedule_param(
    arguments: [u64; 6],
    state: &ProbeDispatchState,
) -> u64 {
    if arguments[1] == 0 {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    }
    let Some(attributes) = pthread_attr_state(arguments[0], state) else {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    };
    write_guest_u32(arguments[1], attributes.schedule_priority.cast_unsigned());
    0
}

pub(crate) fn pthread_attr_set_inherit_schedule(
    arguments: [u64; 6],
    state: &mut ProbeDispatchState,
) -> u64 {
    let inherit_schedule = low_i32(arguments[1]);
    update_pthread_attr(arguments[0], state, |attributes| {
        attributes.inherit_schedule = inherit_schedule;
    })
}

pub(crate) fn pthread_attr_set_schedule_policy(
    arguments: [u64; 6],
    state: &mut ProbeDispatchState,
) -> u64 {
    let schedule_policy = low_i32(arguments[1]);
    update_pthread_attr(arguments[0], state, |attributes| {
        attributes.schedule_policy = schedule_policy;
    })
}

pub(crate) fn pthread_get_schedule_param(
    arguments: [u64; 6],
    state: &mut ProbeDispatchState,
) -> u64 {
    if arguments[0] == 0 || arguments[1] == 0 || arguments[2] == 0 {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    }
    let attributes = *state.thread_attributes.entry(arguments[0]).or_default();
    write_guest_u32(arguments[1], attributes.schedule_policy.cast_unsigned());
    write_guest_u32(arguments[2], attributes.schedule_priority.cast_unsigned());
    0
}

pub(crate) fn pthread_set_schedule_param(
    arguments: [u64; 6],
    state: &mut ProbeDispatchState,
) -> u64 {
    if arguments[0] == 0 || arguments[2] == 0 {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    }
    let policy = low_i32(arguments[1]);
    let priority = read_guest_u32(arguments[2]).cast_signed();
    let attributes = state.thread_attributes.entry(arguments[0]).or_default();
    attributes.schedule_policy = policy;
    attributes.schedule_priority = priority;
    0
}

pub(crate) fn pthread_set_affinity(arguments: [u64; 6], state: &mut ProbeDispatchState) -> u64 {
    if arguments[0] == 0 {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    }
    state
        .thread_attributes
        .entry(arguments[0])
        .or_default()
        .affinity_mask = arguments[1];
    0
}

pub(crate) fn pthread_set_priority(arguments: [u64; 6], state: &mut ProbeDispatchState) -> u64 {
    if arguments[0] == 0 {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    }
    state
        .thread_attributes
        .entry(arguments[0])
        .or_default()
        .schedule_priority = low_i32(arguments[1]);
    0
}

pub(crate) fn pthread_rename(arguments: [u64; 6], state: &mut ProbeDispatchState) -> u64 {
    if arguments[0] == 0 || arguments[1] == 0 {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    }
    if let Some(thread) = state.pthread_threads.get_mut(&arguments[0])
        && let Some(name) = read_guest_cstring(arguments[1])
    {
        thread.name = name;
    }
    0
}

pub(crate) fn create_pthread_mutex_attr(address: u64, state: &mut ProbeDispatchState) -> u64 {
    if address == 0 {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    }
    if let Some(previous) = state.pthread_mutex_attributes.remove(&address) {
        guest_free(previous.handle);
    }
    let handle = guest_calloc(1, 0x40);
    if handle == 0 {
        return sce_kernel_error(SCE_KERNEL_ERROR_TRY_AGAIN);
    }
    let attributes = PthreadMutexAttrState::default();
    write_pthread_mutex_attr_object(handle, attributes);
    write_guest_u64(address, handle);
    state
        .pthread_mutex_attributes
        .insert(address, PthreadMutexAttrObject { handle, attributes });
    0
}

fn resolve_pthread_mutex_attr_key(address: u64, state: &ProbeDispatchState) -> Option<u64> {
    if address == 0 {
        return None;
    }
    if state.pthread_mutex_attributes.contains_key(&address) {
        return Some(address);
    }
    state
        .pthread_mutex_attributes
        .iter()
        .find_map(|(key, object)| (object.handle == address).then_some(*key))
}

pub(crate) fn pthread_mutex_attr_state(
    address: u64,
    state: &ProbeDispatchState,
) -> Option<PthreadMutexAttrState> {
    let key = resolve_pthread_mutex_attr_key(address, state)?;
    state
        .pthread_mutex_attributes
        .get(&key)
        .map(|object| object.attributes)
}

fn write_pthread_mutex_attr_object(address: u64, attributes: PthreadMutexAttrState) {
    write_guest_u32(address, attributes.mutex_type.cast_unsigned());
    write_guest_u32(address + 4, attributes.protocol.cast_unsigned());
}

pub(crate) fn destroy_pthread_mutex_attr(address: u64, state: &mut ProbeDispatchState) -> u64 {
    let Some(key) = resolve_pthread_mutex_attr_key(address, state) else {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    };
    let object = state
        .pthread_mutex_attributes
        .remove(&key)
        .expect("the pthread mutex attribute key was just resolved");
    write_guest_u64(key, 0);
    guest_free(object.handle);
    0
}

pub(crate) fn pthread_mutex_attr_set_type(
    arguments: [u64; 6],
    state: &mut ProbeDispatchState,
) -> u64 {
    let mutex_type = match low_i32(arguments[1]) {
        0 | 1 => 1,
        value @ 2..=4 => value,
        _ => return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT),
    };
    let Some(key) = resolve_pthread_mutex_attr_key(arguments[0], state) else {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    };
    let object = state
        .pthread_mutex_attributes
        .get_mut(&key)
        .expect("the pthread mutex attribute key was just resolved");
    object.attributes.mutex_type = mutex_type;
    write_pthread_mutex_attr_object(object.handle, object.attributes);
    0
}

pub(crate) fn pthread_mutex_attr_set_protocol(
    arguments: [u64; 6],
    state: &mut ProbeDispatchState,
) -> u64 {
    // Gen5 accepts NONE, INHERIT, and PROTECT. The host synchronization implementation does not
    // expose priority inheritance, but preserving the attribute is observable and must succeed.
    let protocol = match low_i32(arguments[1]) {
        value @ 0..=2 => value,
        _ => return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT),
    };
    let Some(key) = resolve_pthread_mutex_attr_key(arguments[0], state) else {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    };
    let object = state
        .pthread_mutex_attributes
        .get_mut(&key)
        .expect("the pthread mutex attribute key was just resolved");
    object.attributes.protocol = protocol;
    write_pthread_mutex_attr_object(object.handle, object.attributes);
    0
}

pub(crate) fn pthread_key_create(arguments: [u64; 6], state: &mut ProbeDispatchState) -> u64 {
    if arguments[0] == 0 {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    }
    const MAX_KEYS: u32 = 256;
    let Some(key) = (0..MAX_KEYS).find(|key| !state.pthread_keys.contains_key(key)) else {
        return sce_kernel_error(SCE_KERNEL_ERROR_TRY_AGAIN);
    };
    let generation = NEXT_PTHREAD_KEY_GENERATION
        .fetch_add(1, Ordering::Relaxed)
        .max(1);
    state.pthread_keys.insert(
        key,
        PthreadKeyState {
            destructor: arguments[1],
            generation,
        },
    );
    pthread_key_generations()
        .write()
        .unwrap_or_else(std::sync::PoisonError::into_inner)[key as usize] = generation;
    write_guest_u32(arguments[0], key);
    0
}

pub(crate) fn pthread_key_delete(key: u32, state: &mut ProbeDispatchState) -> u64 {
    if state.pthread_keys.remove(&key).is_none() {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    }
    pthread_key_generations()
        .write()
        .unwrap_or_else(std::sync::PoisonError::into_inner)[key as usize] = 0;
    GUEST_PTHREAD_SPECIFIC.with_borrow_mut(|values| {
        values.remove(&key);
    });
    0
}

pub(crate) fn pthread_set_specific(arguments: [u64; 6], state: &ProbeDispatchState) -> u64 {
    let key = low_u32(arguments[0]);
    let Some(key_state) = state.pthread_keys.get(&key) else {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    };
    GUEST_PTHREAD_SPECIFIC.with_borrow_mut(|values| {
        values.insert(key, (key_state.generation, arguments[1]));
    });
    0
}

pub(crate) fn pthread_get_specific(key: u32) -> u64 {
    let Some(&generation) = pthread_key_generations()
        .read()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .get(key as usize)
    else {
        return 0;
    };
    if generation == 0 {
        return 0;
    }
    GUEST_PTHREAD_SPECIFIC.with_borrow(|values| {
        values
            .get(&key)
            .filter(|(stored_generation, _)| *stored_generation == generation)
            .map_or(0, |(_, value)| *value)
    })
}

pub(crate) fn take_current_thread_specific_destructors(
    state: &ProbeDispatchState,
) -> Vec<(u64, u64)> {
    GUEST_PTHREAD_SPECIFIC.with_borrow_mut(|values| {
        state
            .pthread_keys
            .iter()
            .filter_map(|(&key, &key_state)| {
                let (generation, value) = values.get_mut(&key)?;
                if *generation != key_state.generation || key_state.destructor == 0 || *value == 0 {
                    return None;
                }
                let argument = *value;
                *value = 0;
                Some((key_state.destructor, argument))
            })
            .collect()
    })
}

pub(crate) fn run_guest_thread_destructors() {
    let Some(dispatch) = PROBE_DISPATCH.get() else {
        return;
    };
    let thread_dtors = dispatch
        .read()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .thread_dtors_callback;
    if thread_dtors != 0 {
        type Callback = unsafe extern "C" fn();
        // SAFETY: `_sceKernelSetThreadDtors` registers a live no-argument guest callback. The
        // process dispatch lock is not held, so the callback may freely re-enter guest imports.
        let callback: Callback = unsafe { std::mem::transmute(thread_dtors) };
        // SAFETY: The registration defines the callback ABI above.
        unsafe { callback() };
    }

    const DESTRUCTOR_ITERATIONS: usize = 4;
    for _ in 0..DESTRUCTOR_ITERATIONS {
        let destructors = {
            let state = dispatch
                .read()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            take_current_thread_specific_destructors(&state)
        };
        if destructors.is_empty() {
            break;
        }
        for (function, argument) in destructors {
            type Destructor = unsafe extern "C" fn(u64);
            // SAFETY: pthread-key creation registers a live guest SysV destructor and the value is
            // the opaque pointer installed by this same thread. It was cleared before this call so
            // a destructor may install a replacement for the next iteration.
            let destructor: Destructor = unsafe { std::mem::transmute(function) };
            // SAFETY: The key registration defines the destructor ABI above.
            unsafe { destructor(argument) };
        }
    }
    GUEST_PTHREAD_SPECIFIC.with_borrow_mut(BTreeMap::clear);
}

pub(crate) fn pthread_condition_attr_init(address: u64, state: &mut ProbeDispatchState) -> u64 {
    if address == 0 {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    }
    if !state.pthread_condition_attributes.insert(address) {
        return u64::from(libc::EBUSY.cast_unsigned());
    }
    write_guest_u64(address, 0);
    0
}

pub(crate) fn pthread_condition_attr_destroy(address: u64, state: &mut ProbeDispatchState) -> u64 {
    if address == 0 || !state.pthread_condition_attributes.remove(&address) {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    }
    write_guest_u64(address, 0);
    0
}
