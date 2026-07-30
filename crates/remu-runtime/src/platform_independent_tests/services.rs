// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

use std::sync::atomic::Ordering;

use super::{
    ONCE_CALLBACKS, ProbeDispatchState, SCE_AJM_ERROR_INVALID_PARAMETER, SceAjmBatchInfo,
    ajm_batch_initialize, ajm_finalize, ajm_initialize, ajm_instance_create, ajm_instance_destroy,
    ajm_module_register, ajm_module_unregister, dispatch_probe_import, guest_pthread_once,
    once_test_callback, sce_kernel_error, system_service_param_get_int, video_out_open,
};

#[test]
fn module_unwind_query_writes_the_gen5_metadata_layout() {
    let mut state = ProbeDispatchState::default();
    state.unwind_modules.push(crate::GuestUnwindModule {
        name: "eboot.bin".to_owned(),
        base: 0x1000,
        end: 0x9000,
        eh_frame_header: 0x7000,
        eh_frame: 0x5000,
        eh_frame_size: 0x2000,
        segments: Vec::new(),
    });
    let mut output = vec![0xcc_u8; 0x130];
    let output_address = output.as_mut_ptr().addr() as u64;
    crate::write_guest_u64(output_address, 0x130);
    assert_eq!(
        crate::kernel_module_info_for_unwind([0x1234, 1, output_address, 0, 0, 0], &state),
        0
    );
    assert_eq!(crate::read_guest_u64(output_address), 0x130);
    assert_eq!(&output[8..17], b"eboot.bin");
    assert_eq!(crate::read_guest_u64(output_address + 0x108), 0x7000);
    assert_eq!(crate::read_guest_u64(output_address + 0x110), 0x5000);
    assert_eq!(crate::read_guest_u64(output_address + 0x118), 0x2000);
    assert_eq!(crate::read_guest_u64(output_address + 0x120), 0x1000);
    assert_eq!(crate::read_guest_u64(output_address + 0x128), 0x8000);
}

#[test]
fn module_info_from_addr_writes_the_extended_layout_with_segments() {
    let mut state = ProbeDispatchState::default();
    state.unwind_modules.push(crate::GuestUnwindModule {
        name: "eboot.bin".to_owned(),
        base: 0x1000,
        end: 0x9000,
        eh_frame_header: 0x7000,
        eh_frame: 0x5000,
        eh_frame_size: 0x2000,
        segments: vec![crate::GuestModuleSegment {
            address: 0x1000,
            size: 0x8000,
            protection: 5,
        }],
    });
    let mut output = vec![0xcc_u8; 0x1a8];
    let output_address = output.as_mut_ptr().addr() as u64;
    crate::write_guest_u64(output_address, 0x1a8);
    assert_eq!(
        crate::kernel_module_info_from_addr([0x1234, 2, output_address, 0, 0, 0], &state),
        0
    );
    assert_eq!(crate::read_guest_u64(output_address), 0x1a8);
    assert_eq!(&output[8..17], b"eboot.bin");
    assert_eq!(crate::read_guest_u64(output_address + 0x148), 0x7000);
    assert_eq!(crate::read_guest_u64(output_address + 0x150), 0x5000);
    assert_eq!(crate::read_guest_u64(output_address + 0x160), 0x1000);
    assert_eq!(crate::read_guest_u32(output_address + 0x168), 0x8000);
    assert_eq!(crate::read_guest_u32(output_address + 0x16c), 5);
    assert_eq!(crate::read_guest_u32(output_address + 0x1a0), 1);
}

#[test]
fn video_out_open_owns_the_single_main_display_port() {
    let mut state = ProbeDispatchState::default();
    assert_eq!(video_out_open([0xff, 0, 0, 0, 0, 0], &mut state), 1);
    assert_eq!(state.video_out.user_id, 0xff);
    assert_eq!(state.video_out.bus_type, 0);
    assert_eq!(state.video_out.index, 0);
    assert_eq!(
        video_out_open([0xff, 0, 0, 0, 0, 0], &mut state),
        sce_kernel_error(crate::VIDEO_OUT_ERROR_RESOURCE_BUSY)
    );

    let mut other = ProbeDispatchState::default();
    assert_eq!(
        video_out_open([0xff, 1, 0, 0, 0, 0], &mut other),
        sce_kernel_error(crate::VIDEO_OUT_ERROR_INVALID_VALUE)
    );
}

#[test]
fn pthread_once_runs_guest_initializer_exactly_once() {
    ONCE_CALLBACKS.store(0, Ordering::Relaxed);
    let mut control = 0_u32;
    let control_address = (&raw mut control).addr() as u64;
    let routine_address = once_test_callback as *const () as usize as u64;
    std::thread::scope(|scope| {
        for _ in 0..8 {
            scope.spawn(move || {
                assert_eq!(guest_pthread_once(control_address, routine_address), 0);
            });
        }
    });
    assert_eq!(control, 1);
    assert_eq!(ONCE_CALLBACKS.load(Ordering::Relaxed), 1);
}

#[test]
fn returns_stable_integer_system_parameters() {
    let mut value = 0_i32;
    let output = (&raw mut value).addr() as u64;

    assert_eq!(system_service_param_get_int([1, output, 0, 0, 0, 0]), 0);
    assert_eq!(value, 1);
    assert_eq!(system_service_param_get_int([4, output, 0, 0, 0, 0]), 0);
    assert_eq!(value, 180);
    assert_eq!(
        system_service_param_get_int([1, 0, 0, 0, 0, 0]),
        sce_kernel_error(0x80a1_0003)
    );
}

#[test]
fn fresh_boot_does_not_skip_the_platform_notice_screen() {
    let mut skip = 0xff_u8;
    let mut state = ProbeDispatchState::default();

    assert_eq!(
        dispatch_probe_import(
            "3RQ5aQfnstU",
            [(&raw mut skip).addr() as u64, 0, 0, 0, 0, 0],
            [0; 6],
            &mut state,
        ),
        0
    );
    assert_eq!(skip, 0);
    assert!(state.unimplemented_nids.is_empty());
}

#[test]
fn pad_get_handle_observes_the_primary_controller_lifecycle() {
    let mut state = ProbeDispatchState::default();
    let primary = [0x1000_0000, 0, 0, 0, 0, 0];

    assert_eq!(
        dispatch_probe_import("u1GRHp+oWoY", primary, [0; 6], &mut state),
        sce_kernel_error(crate::SCE_PAD_ERROR_NOT_INITIALIZED)
    );
    assert_eq!(
        dispatch_probe_import("hv1luiJrqQM", [0; 6], [0; 6], &mut state),
        0
    );
    assert_eq!(
        dispatch_probe_import("u1GRHp+oWoY", primary, [0; 6], &mut state),
        sce_kernel_error(crate::SCE_PAD_ERROR_DEVICE_NO_HANDLE)
    );
    assert_eq!(
        dispatch_probe_import("xk0AcarP3V4", primary, [0; 6], &mut state),
        1
    );
    assert_eq!(
        dispatch_probe_import("u1GRHp+oWoY", primary, [0; 6], &mut state),
        1
    );
    assert_eq!(
        dispatch_probe_import("xk0AcarP3V4", primary, [0; 6], &mut state),
        sce_kernel_error(crate::SCE_PAD_ERROR_ALREADY_OPENED)
    );
    assert_eq!(
        dispatch_probe_import("6ncge5+l5Qs", [1, 0, 0, 0, 0, 0], [0; 6], &mut state),
        0
    );
    assert_eq!(
        dispatch_probe_import("u1GRHp+oWoY", primary, [0; 6], &mut state),
        sce_kernel_error(crate::SCE_PAD_ERROR_DEVICE_NO_HANDLE)
    );
    assert_eq!(
        dispatch_probe_import("xk0AcarP3V4", primary, [0; 6], &mut state),
        2
    );
    assert_eq!(
        dispatch_probe_import("u1GRHp+oWoY", primary, [0; 6], &mut state),
        2
    );
    assert_eq!(
        dispatch_probe_import("6ncge5+l5Qs", [1, 0, 0, 0, 0, 0], [0; 6], &mut state),
        sce_kernel_error(crate::SCE_PAD_ERROR_INVALID_HANDLE)
    );
    assert_eq!(
        dispatch_probe_import("u1GRHp+oWoY", primary, [0; 6], &mut state),
        2,
        "rejecting a stale handle must not close the new controller"
    );
    assert_eq!(
        crate::known_nid_name("u1GRHp+oWoY"),
        Some("scePadGetHandle")
    );
    assert!(state.unimplemented_nids.is_empty());
}

#[test]
fn save_data_dialog_initialization_requires_and_owns_common_dialog_state() {
    let mut state = ProbeDispatchState::default();
    assert_eq!(
        dispatch_probe_import("s9e3+YpRnzw", [0; 6], [0; 6], &mut state),
        sce_kernel_error(crate::COMMON_DIALOG_ERROR_NOT_SYSTEM_INITIALIZED)
    );
    assert_eq!(
        dispatch_probe_import("uoUpLGNkygk", [0; 6], [0; 6], &mut state),
        0
    );
    assert_eq!(
        dispatch_probe_import("s9e3+YpRnzw", [0; 6], [0; 6], &mut state),
        0
    );
    assert_eq!(
        dispatch_probe_import("ERKzksauAJA", [0; 6], [0; 6], &mut state),
        u64::from(crate::COMMON_DIALOG_STATUS_INITIALIZED)
    );
    assert_eq!(
        dispatch_probe_import("s9e3+YpRnzw", [0; 6], [0; 6], &mut state),
        sce_kernel_error(crate::COMMON_DIALOG_ERROR_DIALOG_ALREADY_INITIALIZED)
    );
    assert_eq!(
        dispatch_probe_import("lDqxaY1UbEo", [0; 6], [0; 6], &mut state),
        sce_kernel_error(crate::COMMON_DIALOG_ERROR_BUSY),
        "save-data dialog must own the shared common-dialog service"
    );
    assert_eq!(
        dispatch_probe_import("YuH2FA7azqQ", [0; 6], [0; 6], &mut state),
        0
    );
    assert_eq!(
        dispatch_probe_import("ERKzksauAJA", [0; 6], [0; 6], &mut state),
        0,
        "terminating the save-data dialog resets its lifecycle"
    );
    assert_eq!(
        dispatch_probe_import("lDqxaY1UbEo", [0; 6], [0; 6], &mut state),
        0,
        "terminating the save-data dialog must release common-dialog ownership"
    );
    assert_eq!(
        dispatch_probe_import("YuH2FA7azqQ", [0; 6], [0; 6], &mut state),
        sce_kernel_error(0x8002_0001),
        "terminating an inactive save-data dialog is rejected"
    );
    assert_eq!(
        crate::known_nid_name("s9e3+YpRnzw"),
        Some("sceSaveDataDialogInitialize")
    );
    assert_eq!(
        crate::known_nid_name("YuH2FA7azqQ"),
        Some("sceSaveDataDialogTerminate")
    );
    assert!(state.unimplemented_nids.is_empty());

    let mut busy = ProbeDispatchState::default();
    assert_eq!(
        dispatch_probe_import("uoUpLGNkygk", [0; 6], [0; 6], &mut busy),
        0
    );
    assert_eq!(
        dispatch_probe_import("lDqxaY1UbEo", [0; 6], [0; 6], &mut busy),
        0
    );
    assert_eq!(
        dispatch_probe_import("s9e3+YpRnzw", [0; 6], [0; 6], &mut busy),
        sce_kernel_error(crate::COMMON_DIALOG_ERROR_BUSY)
    );
    assert_eq!(
        dispatch_probe_import("ePw-kqZmelo", [0; 6], [0; 6], &mut busy),
        0
    );
    assert_eq!(
        dispatch_probe_import("s9e3+YpRnzw", [0; 6], [0; 6], &mut busy),
        0,
        "terminating the message dialog must release common-dialog ownership"
    );
}

#[test]
fn share_and_np_title_services_validate_and_retain_caller_data() {
    let mut state = ProbeDispatchState::default();
    assert_eq!(
        dispatch_probe_import("7QZtURYnXG4", [0; 6], [0; 6], &mut state),
        sce_kernel_error(crate::SCE_KERNEL_ERROR_INVALID_ARGUMENT)
    );
    assert_eq!(
        dispatch_probe_import("Ec63y59l9tw", [0; 6], [0; 6], &mut state),
        sce_kernel_error(crate::SCE_NP_ERROR_INVALID_ARGUMENT)
    );

    let content = std::ffi::CString::new("silent-hill-capture").expect("valid C string");
    assert_eq!(
        dispatch_probe_import(
            "7QZtURYnXG4",
            [content.as_ptr() as u64, 0, 0, 0, 0, 0],
            [0; 6],
            &mut state
        ),
        0
    );
    assert_eq!(
        state.services.share_content_parameter.as_deref(),
        Some("silent-hill-capture")
    );
    assert_eq!(
        dispatch_probe_import("YBiIdcDPrxs", [0; 6], [0; 6], &mut state),
        0
    );
    assert!(state.services.share_feature_permitted);

    let mut title_id = *b"PPSA10112_00\0\0\0\0";
    let title_secret = [0x5a_u8; 128];
    assert_eq!(
        dispatch_probe_import("7QZtURYnXG4", [1, 0, 0, 0, 0, 0], [0; 6], &mut state),
        sce_kernel_error(crate::SCE_KERNEL_ERROR_INVALID_ARGUMENT)
    );
    assert_eq!(
        dispatch_probe_import(
            "Ec63y59l9tw",
            [1, title_secret.as_ptr() as u64, 0, 0, 0, 0],
            [0; 6],
            &mut state
        ),
        sce_kernel_error(crate::SCE_NP_ERROR_INVALID_ARGUMENT)
    );
    assert_eq!(
        dispatch_probe_import(
            "Ec63y59l9tw",
            [
                title_id.as_mut_ptr() as u64,
                title_secret.as_ptr() as u64,
                0,
                0,
                0,
                0
            ],
            [0; 6],
            &mut state
        ),
        0
    );
    assert_eq!(state.services.np_title_id, Some(title_id));
    assert_eq!(
        crate::known_nid_name("YBiIdcDPrxs"),
        Some("sceShareFeaturePermit")
    );
    assert_eq!(
        crate::known_nid_name("7QZtURYnXG4"),
        Some("sceShareSetContentParam")
    );
    assert_eq!(
        crate::known_nid_name("Ec63y59l9tw"),
        Some("sceNpSetNpTitleId")
    );
    assert!(state.unimplemented_nids.is_empty());
}

#[test]
fn ajm_batch_initialize_writes_only_the_five_qword_info_object() {
    let batch = [0x5a_u8; 0x708];
    let original_batch = batch;
    let mut info_storage = [0xa5_u8; 0x48];
    let info_address = info_storage[0x10..0x38].as_mut_ptr() as u64;

    assert_eq!(
        ajm_batch_initialize([
            batch.as_ptr() as u64,
            batch.len() as u64,
            info_address,
            0,
            0,
            0,
        ]),
        0
    );

    assert_eq!(
        batch, original_batch,
        "the caller-owned batch arena is preserved"
    );
    assert_eq!(&info_storage[..0x10], &[0xa5; 0x10]);
    assert_eq!(&info_storage[0x38..], &[0xa5; 0x10]);
    // SAFETY: info_address points at the initialized 0x28-byte subrange above.
    let info = unsafe { std::ptr::read_unaligned(info_address as *const SceAjmBatchInfo) };
    assert_eq!(
        info,
        SceAjmBatchInfo {
            buffer: batch.as_ptr() as u64,
            offset: 0,
            size: batch.len() as u64,
            last_good_job: 0,
            last_good_job_return_address: 0,
        }
    );
}

#[test]
fn ajm_batch_initialize_requires_the_ps5_info_output() {
    let batch = [0x5a_u8; 0x708];
    assert_eq!(
        ajm_batch_initialize([batch.as_ptr() as u64, batch.len() as u64, 0, 0, 0, 0,]),
        sce_kernel_error(SCE_AJM_ERROR_INVALID_PARAMETER)
    );
    assert!(batch.iter().all(|byte| *byte == 0x5a));
}

/// Gen5 tags the reserved argument with a library revision; only the low dword must be zero.
#[test]
fn ajm_initialize_accepts_a_revision_tagged_reserved_argument() {
    let mut state = ProbeDispatchState::default();
    let mut context = 0_u32;
    assert_eq!(
        ajm_initialize(
            [0x3_0000_0000, (&raw mut context).addr() as u64, 0, 0, 0, 0],
            &mut state,
        ),
        0
    );
    assert_ne!(context, 0);
    // A non-zero low dword is still a malformed reserved value.
    assert_ne!(
        ajm_initialize(
            [0x3_0000_0001, (&raw mut context).addr() as u64, 0, 0, 0, 0],
            &mut state,
        ),
        0
    );
}

#[test]
fn ajm_initialize_allocates_stable_nonzero_context_ids() {
    let mut state = ProbeDispatchState::default();
    let mut first = 0_u32;
    let mut second = 0_u32;
    assert_eq!(
        dispatch_probe_import(
            "dl+4eHSzUu4",
            [0, (&raw mut first).addr() as u64, 0, 0, 0, 0],
            [0; 6],
            &mut state,
        ),
        0
    );
    assert_eq!(
        dispatch_probe_import(
            "dl+4eHSzUu4",
            [0, (&raw mut second).addr() as u64, 0, 0, 0, 0],
            [0; 6],
            &mut state,
        ),
        0
    );
    assert_eq!((first, second), (1, 2));
    assert!(state.ajm_contexts.contains_key(&first));
    assert!(state.ajm_contexts.contains_key(&second));
    assert_eq!(
        crate::known_nid_name("dl+4eHSzUu4"),
        Some("sceAjmInitialize")
    );
    assert!(state.unimplemented_nids.is_empty());
}

#[test]
fn np_web_api2_initialize_allocates_bounded_library_contexts() {
    let first = crate::dispatch_stateless_import("+o9816YQhqQ", [1, 0x8_0000, 0, 0, 0, 0], [0; 6])
        .expect("NP Web API2 initialize is implemented");
    let second = crate::dispatch_stateless_import("+o9816YQhqQ", [1, 0x8_0000, 0, 0, 0, 0], [0; 6])
        .expect("second NP Web API2 context");

    assert!((1..0x8000).contains(&first));
    assert_eq!(second, first + 1);
    assert_eq!(
        crate::known_nid_name("+o9816YQhqQ"),
        Some("sceNpWebApi2Initialize")
    );
}

#[test]
fn ajm_lifecycle_registers_codec_and_instances() {
    let mut state = ProbeDispatchState::default();
    let mut context = 0_u32;
    let mut instance = 0_u32;
    let context_output = (&raw mut context).addr() as u64;
    let instance_output = (&raw mut instance).addr() as u64;

    assert_eq!(
        ajm_initialize([0, context_output, 0, 0, 0, 0], &mut state),
        0
    );
    assert_eq!(
        ajm_module_register([u64::from(context), 1, 0, 0, 0, 0], &mut state),
        0
    );
    assert_eq!(
        ajm_instance_create(
            [u64::from(context), 1, 1, instance_output, 0, 0],
            &mut state,
        ),
        0
    );
    assert_eq!(instance, 0x4001);
    assert_eq!(
        ajm_instance_destroy(
            [u64::from(context), u64::from(instance), 0, 0, 0, 0],
            &mut state,
        ),
        0
    );
    assert_eq!(
        ajm_module_unregister([u64::from(context), 1, 0, 0, 0, 0], &mut state),
        0
    );
    assert_eq!(
        ajm_finalize([u64::from(context), 0, 0, 0, 0, 0], &mut state),
        0
    );
    assert!(state.ajm_contexts.is_empty());
}

#[test]
fn json_initializer_tracks_allocator_and_file_buffer_configuration() {
    let mut state = ProbeDispatchState::default();
    let mut initializer = [0_u8; 8];
    let mut parameter = [0_u8; 0x28];
    let mut allocator = [0_u8; 8];
    let initializer_address = initializer.as_mut_ptr().addr() as u64;
    let parameter_address = parameter.as_mut_ptr().addr() as u64;
    let allocator_address = allocator.as_mut_ptr().addr() as u64;

    assert_eq!(
        crate::json_init_parameter_set_allocator(
            [parameter_address, allocator_address, 0x1234, 0, 0, 0,],
            &mut state,
        ),
        0
    );
    assert_eq!(
        crate::json_init_parameter_set_file_buffer_size(
            [parameter_address, 0x200, 0, 0, 0, 0],
            &mut state,
        ),
        0
    );
    assert_eq!(
        crate::json_initializer_initialize(
            [initializer_address, parameter_address, 0, 0, 0, 0],
            &mut state,
        ),
        0
    );
    assert_eq!(
        state.json_runtime.allocators.get(&parameter_address),
        Some(&(allocator_address, 0x1234))
    );
    assert_eq!(
        state.json_runtime.file_buffer_sizes.get(&parameter_address),
        Some(&0x200)
    );
    assert_eq!(
        state
            .json_runtime
            .initialized_objects
            .get(&initializer_address),
        Some(&parameter_address)
    );
    assert_eq!(
        crate::known_nid_name("IXW-z8pggfg"),
        Some("sce::Json::Initializer::initialize(InitParameter2)")
    );
    assert_eq!(
        crate::json_initializer_terminate([initializer_address, 0, 0, 0, 0, 0], &mut state,),
        0
    );
}

#[test]
fn offline_np_service_initializers_retain_context_and_property_ownership() {
    let mut state = ProbeDispatchState::default();
    assert_ne!(crate::sysmodule_is_loaded(0xcc, &state), 0);
    assert_eq!(crate::sysmodule_load(0xcc, &mut state), 0);
    assert_eq!(crate::sysmodule_is_loaded(0xcc, &state), 0);
    assert_eq!(crate::game_update_initialize(&mut state), 0);

    let entitlement_parameters = [0_u8; 0x20];
    let mut entitlement_context = [0_u8; 0x20];
    let entitlement_context_address = entitlement_context.as_mut_ptr().addr() as u64;
    let entitlement_parameter_address = entitlement_parameters.as_ptr().addr() as u64;
    assert_eq!(
        crate::np_entitlement_access_initialize(
            [
                entitlement_context_address,
                entitlement_parameter_address,
                0,
                0,
                0,
                0,
            ],
            &mut state,
        ),
        0
    );
    let signaling_parameters = [0x5a_u8; 0x28];
    assert_eq!(
        crate::np_session_signaling_initialize(
            [signaling_parameters.as_ptr().addr() as u64, 0, 0, 0, 0, 0],
            &mut state,
        ),
        0
    );
    let user_context =
        crate::np_web_api2_create_user_context([1, 0x1000_0000, 0, 0, 0, 0], &mut state);
    assert_eq!(user_context, 1);
    assert_eq!(crate::np_check_callback(&mut state), 0);
    assert_eq!(state.services.np_callback_polls, 1);

    let mut universal_parameters = [0_u8; 16];
    universal_parameters[..8].copy_from_slice(&16_u64.to_ne_bytes());
    assert_eq!(
        crate::np_universal_data_initialize(
            universal_parameters.as_ptr().addr() as u64,
            &mut state,
        ),
        0
    );
    let mut object = 0_u64;
    assert_eq!(
        crate::np_universal_data_create_property_object(
            (&raw mut object).addr() as u64,
            &mut state,
        ),
        0
    );
    assert_ne!(object, 0);
    assert_eq!(
        crate::known_nid_name("s6W4Zl4Slgk"),
        Some("sceNpUniversalDataSystemCreateEventPropertyObject")
    );
}

/// An ACM context is an opaque handle the guest hands back to the batch entry points. Creating one
/// must yield a distinct live id and reject a missing output, and destroying an id that was never
/// issued must not silently succeed.
#[cfg(target_os = "macos")]
#[test]
fn acm_contexts_are_distinct_handles_with_checked_lifetimes() {
    let mut state = ProbeDispatchState::default();
    let mut first = 0_u32;
    let mut second = 0_u32;
    assert_eq!(
        crate::acm_context_create([(&raw mut first).addr() as u64, 0, 0, 0, 0, 0], &mut state),
        0
    );
    assert_eq!(
        crate::acm_context_create([(&raw mut second).addr() as u64, 0, 0, 0, 0, 0], &mut state),
        0
    );
    assert_ne!(first, 0);
    assert_ne!(first, second, "each context must be its own handle");
    assert_eq!(
        crate::acm_context_create([0, 0, 0, 0, 0, 0], &mut state),
        sce_kernel_error(crate::SCE_KERNEL_ERROR_INVALID_ARGUMENT)
    );
    assert_eq!(
        crate::acm_context_destroy([u64::from(first), 0, 0, 0, 0, 0], &mut state),
        0
    );
    assert_eq!(
        crate::acm_context_destroy([u64::from(first), 0, 0, 0, 0, 0], &mut state),
        sce_kernel_error(crate::SCE_KERNEL_ERROR_INVALID_ARGUMENT),
        "destroying a released context must not report success"
    );
    assert_eq!(
        crate::known_nid_name("ZIXln2K3XMk"),
        Some("sceAcmContextCreate")
    );
}
