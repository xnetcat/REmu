// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

use std::sync::atomic::Ordering;

use super::{
    DELIVERED_EXCEPTION, DELIVERED_EXCEPTION_RSP, GUEST_EXCEPTION_TEST, GuestExitCallback,
    ProbeDispatchState, PthreadAttrState, SCE_KERNEL_ERROR_NOT_IMPLEMENTED,
    allocate_pthread_thread_object, capture_guest_exception, clone_guest_tls, create_pthread_attr,
    destroy_pthread_attr, dispatch_probe_import, guest_free, pthread_attr_get_affinity,
    pthread_attr_get_stack, pthread_attr_set_affinity, pthread_attr_set_detach_state,
    pthread_attr_set_schedule_param, pthread_attr_set_stack_size, read_guest_u32, read_guest_u64,
    sce_kernel_error, set_current_guest_thread_pointer, take_exit_callbacks, tls_dtv_entry_offset,
    write_guest_u32, write_guest_u64, write_import_stub,
};

#[cfg(target_arch = "x86_64")]
#[test]
fn pthread_exit_returns_to_the_host_thread_boundary() {
    assert_eq!(
        crate::invoke_guest_entry_with_exit(
            super::pthread_exit_boundary_test_entry as *const () as usize as u64,
            20,
            22,
        ),
        42
    );
}

#[test]
fn tls_get_addr_uses_compact_per_thread_module_slots() {
    fn write_template_word(template: &mut [u8], offset: usize, value: u64) {
        template[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
    }

    let template_size = usize::try_from(crate::GUEST_TLS_PREFIX_SIZE + crate::GUEST_TLS_TCB_SIZE)
        .expect("TLS template size fits usize");
    let thread_pointer_offset =
        usize::try_from(crate::GUEST_TLS_PREFIX_SIZE).expect("TLS prefix fits usize");
    let dtv_offset = thread_pointer_offset + 0x100;
    let original_thread_pointer = 0x7000_1000_0000_u64;
    let module1_block = original_thread_pointer - 0x40;
    let module2_block = original_thread_pointer - 0x600;
    let mut template = vec![0_u8; template_size];
    write_template_word(
        &mut template,
        thread_pointer_offset,
        original_thread_pointer,
    );
    write_template_word(&mut template, dtv_offset, 1);
    write_template_word(&mut template, dtv_offset + 8, 2);
    assert_eq!(tls_dtv_entry_offset(1), Some(16));
    assert_eq!(tls_dtv_entry_offset(2), Some(24));
    write_template_word(&mut template, dtv_offset + 16, module1_block);
    write_template_word(&mut template, dtv_offset + 24, module2_block);

    let first = clone_guest_tls(&template);
    let second = clone_guest_tls(&template);
    let first_module2 = first.thread_pointer - 0x600;
    let second_module2 = second.thread_pointer - 0x600;
    assert_ne!(first_module2, second_module2);
    assert_eq!(
        read_guest_u64(first.thread_pointer + 0x100 + 24),
        first_module2
    );
    assert_eq!(
        read_guest_u64(second.thread_pointer + 0x100 + 24),
        second_module2
    );
    assert_eq!(read_guest_u64(first.thread_pointer + 0x100 + 32), 0);

    let mut descriptor = [2_u64, 0x280];
    let descriptor_address = descriptor.as_mut_ptr().addr() as u64;
    let mut state = ProbeDispatchState::default();
    set_current_guest_thread_pointer(first.thread_pointer);
    assert_eq!(
        dispatch_probe_import(
            "vNe1w4diLCs",
            [descriptor_address, 0, 0, 0, 0, 0],
            [0; 6],
            &mut state,
        ),
        first_module2 + 0x280
    );
    assert!(state.unimplemented_nids.is_empty());
    assert_eq!(crate::known_nid_name("vNe1w4diLCs"), Some("__tls_get_addr"));

    write_guest_u32(first_module2 + 0x280, 0x1122_3344);
    assert_eq!(read_guest_u32(first_module2 + 0x280), 0x1122_3344);
    assert_eq!(read_guest_u32(second_module2 + 0x280), 0);
    set_current_guest_thread_pointer(second.thread_pointer);
    assert_eq!(
        dispatch_probe_import(
            "vNe1w4diLCs",
            [descriptor_address, 0, 0, 0, 0, 0],
            [0; 6],
            &mut state,
        ),
        second_module2 + 0x280
    );

    write_guest_u64(second.thread_pointer + 0x100 + 24, 0);
    assert_eq!(crate::guest_tls_get_addr(descriptor_address), 0);
    write_guest_u64(second.thread_pointer + 0x100 + 24, second_module2);
    write_guest_u64(second.thread_pointer + 8, 0);
    assert_eq!(crate::guest_tls_get_addr(descriptor_address), 0);
    write_guest_u64(second.thread_pointer + 8, second.thread_pointer + 0x100);

    descriptor[0] = 0;
    assert_eq!(crate::guest_tls_get_addr(descriptor_address), 0);
    descriptor[0] = 3;
    assert_eq!(crate::guest_tls_get_addr(descriptor_address), 0);
    descriptor[0] = 2;
    descriptor[1] = u64::MAX;
    std::hint::black_box(descriptor[1]);
    assert_eq!(crate::guest_tls_get_addr(descriptor_address), 0);
    assert_eq!(crate::guest_tls_get_addr(0), 0);
    set_current_guest_thread_pointer(0);
    assert_eq!(crate::guest_tls_get_addr(descriptor_address), 0);
}

#[test]
fn address_sanitizer_query_reports_the_uninstrumented_guest_runtime() {
    let mut state = ProbeDispatchState::default();
    assert_eq!(
        dispatch_probe_import("jh+8XiK4LeE", [0; 6], [0; 6], &mut state),
        0
    );
    assert_eq!(
        crate::known_nid_name("jh+8XiK4LeE"),
        Some("sceKernelIsAddressSanitizerEnabled")
    );
    assert!(state.unimplemented_nids.is_empty());
}

#[test]
fn unknown_imports_return_the_gen5_not_implemented_error() {
    let mut state = ProbeDispatchState::default();
    assert_eq!(
        dispatch_probe_import("not-a-real-nid", [0; 6], [0; 6], &mut state),
        sce_kernel_error(SCE_KERNEL_ERROR_NOT_IMPLEMENTED)
    );
    assert_eq!(state.unimplemented_nids.len(), 1);
}

#[test]
fn import_stubs_preserve_guest_rax_for_frame_capture() {
    let mut slot = [0_u8; crate::IMPORT_STUB_SIZE_USIZE];
    write_import_stub(&mut slot, 0x1234, 0x1122_3344_5566_7788).unwrap();

    assert_eq!(slot[0], 0x68);
    assert_eq!(i32::from_le_bytes(slot[1..5].try_into().unwrap()), 0x1234);
    assert_eq!(&slot[5..7], &[0x49, 0xbb]);
    assert_eq!(
        u64::from_le_bytes(slot[7..15].try_into().unwrap()),
        0x1122_3344_5566_7788
    );
    assert_eq!(&slot[15..18], &[0x41, 0xff, 0xe3]);
    assert!(slot[18..].iter().all(|byte| *byte == 0));
}

#[test]
fn import_stack_arguments_are_read_from_the_captured_guest_frame() {
    let stack = [
        0xfeed_face_cafe_beef_u64,
        0x11,
        0x22,
        0x33,
        0x44,
        0x55,
        0x66,
    ];
    let frame = crate::ImportCallFrame {
        guest_rsp: stack.as_ptr().addr() as u64,
        ..crate::ImportCallFrame::default()
    };

    assert_eq!(
        crate::read_import_stack_arguments(&frame),
        [0x11, 0x22, 0x33, 0x44, 0x55, 0x66]
    );
}

#[cfg(all(unix, target_arch = "x86_64"))]
#[test]
fn import_trampoline_captures_the_complete_sysv_register_frame() {
    const PAGE_SIZE: usize = 0x4000;
    let mut slot = [0_u8; crate::IMPORT_STUB_SIZE_USIZE];
    write_import_stub(
        &mut slot,
        usize::try_from(crate::TEST_CAPTURE_DISPATCH_INDEX).unwrap(),
        crate::import_dispatch_trampoline as *const () as usize as u64,
    )
    .unwrap();
    // SAFETY: This test owns one anonymous page and changes it from writable to executable
    // before calling the complete import stub copied into it.
    let mapping = unsafe {
        libc::mmap(
            std::ptr::null_mut(),
            PAGE_SIZE,
            libc::PROT_READ | libc::PROT_WRITE,
            libc::MAP_PRIVATE | libc::MAP_ANON,
            -1,
            0,
        )
    };
    assert_ne!(mapping, libc::MAP_FAILED);
    // SAFETY: mapping owns PAGE_SIZE writable bytes and slot fits at its start.
    unsafe { std::ptr::copy_nonoverlapping(slot.as_ptr(), mapping.cast::<u8>(), slot.len()) };
    // SAFETY: mapping and its exact size came from mmap above.
    assert_eq!(
        unsafe { libc::mprotect(mapping, PAGE_SIZE, libc::PROT_READ | libc::PROT_EXEC) },
        0
    );
    *crate::TEST_CAPTURED_IMPORT_FRAME
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner) = None;

    // SAFETY: the page now contains a complete executable import stub targeting the real
    // trampoline, and invoke_test_import_stub establishes the guest SysV register state.
    let result = unsafe { crate::invoke_test_import_stub(mapping.addr() as u64) };
    let frame = crate::TEST_CAPTURED_IMPORT_FRAME
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .take()
        .expect("test dispatch captured its frame");
    // SAFETY: mapping and its exact size came from mmap and are no longer in use.
    assert_eq!(unsafe { libc::munmap(mapping, PAGE_SIZE) }, 0);

    assert_eq!(
        frame.integer_arguments,
        [0x11, 0x22, 0x33, 0x44, 0x55, 0x66]
    );
    assert_eq!(frame.rax, 0x0102_0304_0506_0708);
    assert_eq!(frame.dispatch_index, crate::TEST_CAPTURE_DISPATCH_INDEX);
    assert_eq!(frame.guest_rbx, 0x7172_7374_7576_7778);
    assert_ne!(frame.guest_return_address, 0);
    assert_eq!(frame.guest_rsp & 0xf, 8);
    for (index, xmm) in frame.xmm_arguments.iter().enumerate() {
        let bytes = xmm.to_le_bytes();
        let low = u64::from_le_bytes(bytes[..8].try_into().unwrap());
        let high = u64::from_le_bytes(bytes[8..].try_into().unwrap());
        assert_eq!(low, u64::try_from(index + 1).unwrap() * 0x101);
        assert_eq!(high, 0);
    }
    assert_eq!(result.integer, frame.rax);
    assert_eq!(result.float.to_bits(), 0x808);
}

#[test]
fn pending_guest_exception_runs_on_target_import_with_orbis_stack_context() {
    let _guard = GUEST_EXCEPTION_TEST
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    const THREAD: u64 = 0x1234;
    const GUEST_RSP: u64 = 0x7000_1234_5678;
    DELIVERED_EXCEPTION.store(0, Ordering::SeqCst);
    DELIVERED_EXCEPTION_RSP.store(0, Ordering::SeqCst);
    crate::set_current_guest_thread_handle(THREAD);
    let handler = capture_guest_exception as *const () as usize as u64;
    let mut delivery = crate::guest_exception_delivery()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    *delivery = crate::GuestExceptionDeliveryState::default();
    delivery.pending.insert(
        THREAD,
        crate::PendingGuestException {
            handler,
            exception: 30,
        },
    );
    drop(delivery);

    crate::deliver_pending_guest_exception_at_import(&crate::ImportCallFrame {
        guest_rsp: GUEST_RSP,
        guest_return_address: 0x8000_1234,
        ..crate::ImportCallFrame::default()
    });

    assert_eq!(DELIVERED_EXCEPTION.load(Ordering::SeqCst), 30);
    assert_eq!(
        DELIVERED_EXCEPTION_RSP.load(Ordering::SeqCst),
        usize::try_from(GUEST_RSP).unwrap()
    );
    let delivery = crate::guest_exception_delivery()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    assert!(delivery.pending.is_empty());
    assert!(delivery.active.is_empty());
    drop(delivery);
    crate::set_current_guest_thread_handle(0);
}

#[test]
fn pending_guest_exception_interrupts_event_flag_and_semaphore_waits() {
    let _guard = GUEST_EXCEPTION_TEST
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    const THREAD: u64 = 0x5678;
    const EVENT_FLAG: u64 = 0x7fff_0000_4000;
    const SEMAPHORE: u32 = 0x7fff_4000;
    crate::set_current_guest_thread_handle(THREAD);
    crate::guest_sync()
        .0
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .event_flags
        .insert(
            EVENT_FLAG,
            crate::guest_event_flag_entry("exception-test", 0, 0),
        );
    crate::guest_sync()
        .0
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .semaphores
        .insert(SEMAPHORE, crate::guest_semaphore_entry(0, 1));
    crate::guest_exception_delivery()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .pending
        .insert(
            THREAD,
            crate::PendingGuestException {
                handler: 1,
                exception: 30,
            },
        );

    assert_eq!(
        crate::guest_event_flag_wait([EVENT_FLAG, 1, crate::EVENT_FLAG_WAIT_OR as u64, 0, 0, 0]),
        sce_kernel_error(crate::SCE_KERNEL_ERROR_INTERRUPTED)
    );
    assert_eq!(
        crate::guest_semaphore_wait([u64::from(SEMAPHORE), 1, 0, 0, 0, 0]),
        sce_kernel_error(crate::SCE_KERNEL_ERROR_INTERRUPTED)
    );

    let sync = crate::guest_sync();
    let mut state = sync
        .0
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    assert_eq!(
        state.event_flags[&EVENT_FLAG]
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .waiters,
        0
    );
    state.event_flags.remove(&EVENT_FLAG);
    state.semaphores.remove(&SEMAPHORE);
    drop(state);
    crate::guest_exception_delivery()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .pending
        .remove(&THREAD);
    crate::set_current_guest_thread_handle(0);
}

#[test]
fn pthread_attributes_preserve_observable_thread_creation_values() {
    let mut state = ProbeDispatchState::default();
    let mut opaque_attribute = 0_u64;
    let attribute_address = (&raw mut opaque_attribute).addr() as u64;
    assert_eq!(
        create_pthread_attr(attribute_address, PthreadAttrState::default(), &mut state),
        0
    );
    assert_ne!(opaque_attribute, 0);
    assert_eq!(
        pthread_attr_set_detach_state([attribute_address, 1, 0, 0, 0, 0], &mut state),
        0
    );
    assert_eq!(
        pthread_attr_set_stack_size([attribute_address, 0x20_0000, 0, 0, 0, 0], &mut state),
        0
    );
    assert_eq!(
        pthread_attr_set_affinity([attribute_address, 0x7f, 0, 0, 0, 0], &mut state),
        0
    );
    let priority = 512_i32;
    assert_eq!(
        pthread_attr_set_schedule_param(
            [
                attribute_address,
                (&raw const priority).addr() as u64,
                0,
                0,
                0,
                0,
            ],
            &mut state,
        ),
        0
    );
    let mut stack_address = u64::MAX;
    let mut stack_size = 0_u64;
    let mut affinity_mask = 0_u64;
    assert_eq!(
        pthread_attr_get_stack(
            [
                attribute_address,
                (&raw mut stack_address).addr() as u64,
                (&raw mut stack_size).addr() as u64,
                0,
                0,
                0,
            ],
            &state,
        ),
        0
    );
    let attributes = state.pthread_attributes[&attribute_address].attributes;
    assert_eq!(attributes.detach_state, 1);
    assert_eq!(attributes.stack_size, 0x20_0000);
    assert_eq!(attributes.schedule_priority, priority);
    assert_eq!(stack_address, 0);
    assert_eq!(stack_size, 0x20_0000);
    assert_eq!(
        pthread_attr_get_affinity(
            [
                attribute_address,
                (&raw mut affinity_mask).addr() as u64,
                0,
                0,
                0,
                0,
            ],
            &state,
        ),
        0x7f
    );
    assert_eq!(affinity_mask, 0x7f);
    assert_eq!(destroy_pthread_attr(attribute_address, &mut state), 0);
    assert_eq!(opaque_attribute, 0);
}

#[test]
fn pthread_threads_are_distinct_guest_pointer_objects() {
    let mut state = ProbeDispatchState {
        next_thread_unique_id: 1,
        ..ProbeDispatchState::default()
    };
    let first = allocate_pthread_thread_object(&mut state, "worker").unwrap();
    let second = allocate_pthread_thread_object(&mut state, "").unwrap();

    assert_ne!(first, 0);
    assert_ne!(first, second);
    assert_eq!(read_guest_u64(first), 0);
    assert_eq!(read_guest_u64(second + 0xff8), 0);
    assert_eq!(state.pthread_threads[&first].unique_id, 1);
    assert_eq!(state.pthread_threads[&first].name, "worker");
    assert_eq!(state.pthread_threads[&second].unique_id, 2);
    assert_eq!(state.pthread_threads[&second].name, "Thread-2");

    crate::set_current_guest_thread_handle(first);
    crate::set_current_guest_thread_unique_id(1);
    assert_eq!(crate::current_guest_thread_handle(), first);
    assert_eq!(crate::current_guest_thread_unique_id(), 1);
    crate::set_current_guest_thread_handle(0);
    crate::set_current_guest_thread_unique_id(0);

    for object in state.pthread_threads.into_values() {
        guest_free(object.handle);
    }

    let mut exhausted = ProbeDispatchState {
        next_thread_unique_id: u64::from(i32::MAX.cast_unsigned()) + 1,
        ..ProbeDispatchState::default()
    };
    assert!(allocate_pthread_thread_object(&mut exhausted, "overflow").is_none());
}

#[test]
fn guest_integer_thread_id_tracks_each_thread_without_stale_values() {
    crate::set_current_guest_thread_unique_id(41);
    let worker = std::thread::spawn(|| {
        let initial = crate::current_guest_thread_unique_id();
        crate::set_current_guest_thread_unique_id(73);
        let assigned = crate::current_guest_thread_unique_id();
        crate::set_current_guest_thread_unique_id(0);
        (initial, assigned, crate::current_guest_thread_unique_id())
    });

    assert_eq!(worker.join().unwrap(), (0, 73, 0));
    assert_eq!(crate::current_guest_thread_unique_id(), 41);
    crate::set_current_guest_thread_unique_id(99);
    assert_eq!(crate::current_guest_thread_unique_id(), 99);
    crate::set_current_guest_thread_unique_id(0);
}

#[test]
fn pthread_keys_reuse_deleted_slots_and_clear_values_before_destructors() {
    let mut state = ProbeDispatchState::default();
    let mut first_key = u32::MAX;
    let first_key_address = (&raw mut first_key).addr() as u64;
    assert_eq!(
        crate::pthread_key_create(
            [first_key_address, 0x1234_5678_9abc_def0, 0, 0, 0, 0],
            &mut state,
        ),
        0
    );
    assert_eq!(first_key, 0);
    assert_eq!(
        crate::pthread_set_specific([u64::from(first_key), 0xcafe_babe, 0, 0, 0, 0]),
        0
    );

    assert_eq!(
        crate::take_current_thread_specific_destructors(&state),
        [(0x1234_5678_9abc_def0, 0xcafe_babe)]
    );
    assert_eq!(crate::pthread_get_specific(first_key), 0);
    assert_eq!(crate::pthread_key_delete(first_key, &mut state), 0);

    let mut replacement_key = u32::MAX;
    assert_eq!(
        crate::pthread_key_create(
            [(&raw mut replacement_key).addr() as u64, 0, 0, 0, 0, 0],
            &mut state,
        ),
        0
    );
    assert_eq!(replacement_key, first_key);
    let replacement_generation = state.pthread_keys[&replacement_key].generation;
    crate::GUEST_PTHREAD_SPECIFIC.with_borrow_mut(|values| {
        values[replacement_key as usize] = (replacement_generation.saturating_sub(1), 0xdead_beef);
    });
    assert_eq!(
        crate::pthread_get_specific(replacement_key),
        0,
        "a value left on another thread under the deleted generation must stay invisible"
    );
    assert!(
        crate::take_current_thread_specific_destructors(&state).is_empty(),
        "a reused key must not inherit the deleted key's thread-specific value"
    );
    crate::GUEST_PTHREAD_SPECIFIC.with_borrow_mut(|values| values.fill((0, 0)));
}

#[test]
fn pthread_set_specific_is_a_stateless_leaf_import() {
    let mut state = ProbeDispatchState::default();
    let mut key = u32::MAX;
    assert_eq!(
        crate::pthread_key_create([(&raw mut key).addr() as u64, 0, 0, 0, 0, 0], &mut state,),
        0
    );
    let arguments = [u64::from(key), 0xdead_beef, 0, 0, 0, 0];
    assert_eq!(
        crate::dispatch_stateless_import("WrOLvHU0yQM", arguments, [0; 6]),
        Some(0)
    );
    assert_eq!(crate::pthread_get_specific(key), 0xdead_beef);
    assert_eq!(crate::pthread_key_delete(key, &mut state), 0);
    assert_eq!(
        crate::dispatch_stateless_import("WrOLvHU0yQM", arguments, [0; 6]),
        Some(u64::from(libc::EINVAL.cast_unsigned()))
    );
    assert_eq!(crate::pthread_get_specific(256), 0);
    assert_eq!(
        crate::pthread_set_specific([256, 0xdead_beef, 0, 0, 0, 0]),
        crate::sce_kernel_error(crate::SCE_KERNEL_ERROR_INVALID_ARGUMENT)
    );
    crate::GUEST_PTHREAD_SPECIFIC.with_borrow_mut(|values| values.fill((0, 0)));
}

#[test]
fn finalization_selects_callbacks_in_reverse_registration_order() {
    let mut state = ProbeDispatchState {
        exit_callbacks: vec![
            GuestExitCallback {
                function: 1,
                argument: None,
                module: 0,
            },
            GuestExitCallback {
                function: 2,
                argument: Some(20),
                module: 7,
            },
            GuestExitCallback {
                function: 3,
                argument: Some(30),
                module: 8,
            },
            GuestExitCallback {
                function: 4,
                argument: Some(40),
                module: 7,
            },
        ],
        ..ProbeDispatchState::default()
    };
    let callbacks = take_exit_callbacks(&mut state, Some(7));
    assert_eq!(
        callbacks
            .iter()
            .map(|callback| callback.function)
            .collect::<Vec<_>>(),
        [4, 2]
    );
    assert_eq!(
        state
            .exit_callbacks
            .iter()
            .map(|callback| callback.function)
            .collect::<Vec<_>>(),
        [1, 3]
    );
}
