// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

//! Common dialog and save-data dialog HLE backed by a real host UI surface.
//!
//! Opening a dialog publishes [`remu_gpu::GpuEvent::ShowHostDialog`] to the Metal presenter, which
//! shows a native AppKit alert. Status stays `RUNNING` until the presenter records a result through
//! [`remu_gpu::complete_host_dialog`]. Headless runs and unit tests without a presenter auto-complete
//! with cancel so titles that poll never hang.

use crate::{
    CommonDialogOwner, InitializationState, ProbeDispatchState, guest_output_pointer,
    publish_gpu_event, read_guest_cstring, sce_kernel_error, write_guest_u32,
};
use remu_gpu::{
    HostDialogButtons, HostDialogKind, HostDialogRequest, HostDialogResultCode, HostToastRequest,
    begin_host_dialog, cancel_host_dialog, complete_host_dialog, host_dialog_result,
};

const COMMON_DIALOG_STATUS_NONE: u32 = 0;
pub(crate) const COMMON_DIALOG_STATUS_INITIALIZED: u32 = 1;
const COMMON_DIALOG_STATUS_RUNNING: u32 = 2;
const COMMON_DIALOG_STATUS_FINISHED: u32 = 3;
pub(crate) const COMMON_DIALOG_ERROR_NOT_SYSTEM_INITIALIZED: u32 = 0x80b8_0001;
pub(crate) const COMMON_DIALOG_ERROR_DIALOG_ALREADY_INITIALIZED: u32 = 0x80b8_0004;
const COMMON_DIALOG_ERROR_NOT_INITIALIZED: u32 = 0x8002_0001;
const COMMON_DIALOG_ERROR_ARG_NULL: u32 = 0x8002_000e;
pub(crate) const COMMON_DIALOG_ERROR_BUSY: u32 = 0x80b8_0008;
/// The dialog closed without the user choosing a button.
const MESSAGE_DIALOG_BUTTON_INVALID: u32 = 0;
const MESSAGE_DIALOG_BUTTON_OK: u32 = 1;
const MESSAGE_DIALOG_BUTTON_NO: u32 = 2;
const MESSAGE_DIALOG_RESULT_OK: u32 = 0;
const MESSAGE_DIALOG_RESULT_USER_CANCELED: u32 = 1;
/// `SceSaveDataDialogResult` layout: mode, result, buttonId, reserved...
const SAVE_DATA_DIALOG_RESULT_OK: u32 = 0;
const SAVE_DATA_DIALOG_RESULT_USER_CANCELED: u32 = 1;

pub(crate) fn message_dialog_initialize(state: &mut ProbeDispatchState) -> u64 {
    if state.services.common_dialog != InitializationState::Initialized {
        return sce_kernel_error(COMMON_DIALOG_ERROR_NOT_SYSTEM_INITIALIZED);
    }
    if state.services.message_dialog_status != COMMON_DIALOG_STATUS_NONE {
        return sce_kernel_error(COMMON_DIALOG_ERROR_DIALOG_ALREADY_INITIALIZED);
    }
    if state.services.common_dialog_owner.is_some() {
        return sce_kernel_error(COMMON_DIALOG_ERROR_BUSY);
    }
    state.services.message_dialog_status = COMMON_DIALOG_STATUS_INITIALIZED;
    state.services.common_dialog_owner = Some(CommonDialogOwner::Message);
    state.services.message_dialog_sequence = 0;
    state.services.message_dialog_result_code = MESSAGE_DIALOG_RESULT_USER_CANCELED;
    state.services.message_dialog_button = MESSAGE_DIALOG_BUTTON_INVALID;
    0
}

pub(crate) fn message_dialog_terminate(state: &mut ProbeDispatchState) -> u64 {
    if state.services.message_dialog_status == COMMON_DIALOG_STATUS_NONE {
        return sce_kernel_error(COMMON_DIALOG_ERROR_NOT_INITIALIZED);
    }
    dismiss_message_dialog(state, HostDialogResultCode::Cancel);
    state.services.message_dialog_status = COMMON_DIALOG_STATUS_NONE;
    if state.services.common_dialog_owner == Some(CommonDialogOwner::Message) {
        state.services.common_dialog_owner = None;
    }
    0
}

/// Opens a system message dialog on the host presenter and leaves it `RUNNING` until dismissed.
pub(crate) fn message_dialog_open(parameters: u64, state: &mut ProbeDispatchState) -> u64 {
    if state.services.message_dialog_status == COMMON_DIALOG_STATUS_NONE {
        return sce_kernel_error(COMMON_DIALOG_ERROR_NOT_INITIALIZED);
    }
    if parameters == 0 {
        return sce_kernel_error(COMMON_DIALOG_ERROR_ARG_NULL);
    }
    let message = read_guest_cstring(parameters.saturating_add(0x10))
        .filter(|text| !text.is_empty())
        .unwrap_or_else(|| "Message".to_owned());
    let sequence = begin_host_dialog();
    state.services.message_dialog_sequence = sequence;
    state.services.message_dialog_result_code = MESSAGE_DIALOG_RESULT_USER_CANCELED;
    state.services.message_dialog_button = MESSAGE_DIALOG_BUTTON_INVALID;
    state.services.message_dialog_status = COMMON_DIALOG_STATUS_RUNNING;
    publish_or_auto_dismiss(
        HostDialogRequest {
            kind: HostDialogKind::Message,
            title: "Message".to_owned(),
            message,
            buttons: HostDialogButtons::OkCancel,
            sequence,
        },
        sequence,
    );
    0
}

pub(crate) fn message_dialog_update_status(state: &mut ProbeDispatchState) -> u64 {
    poll_message_dialog(state);
    u64::from(state.services.message_dialog_status)
}

pub(crate) fn message_dialog_get_status(state: &ProbeDispatchState) -> u64 {
    u64::from(state.services.message_dialog_status)
}

pub(crate) fn message_dialog_close(state: &mut ProbeDispatchState) -> u64 {
    if state.services.message_dialog_status == COMMON_DIALOG_STATUS_NONE {
        return sce_kernel_error(COMMON_DIALOG_ERROR_NOT_INITIALIZED);
    }
    dismiss_message_dialog(state, HostDialogResultCode::Cancel);
    state.services.message_dialog_status = COMMON_DIALOG_STATUS_FINISHED;
    0
}

/// Fills `SceMsgDialogResult`: mode, result code, selected button, then reserved bytes.
pub(crate) fn message_dialog_get_result(output: u64, state: &ProbeDispatchState) -> u64 {
    if state.services.message_dialog_status == COMMON_DIALOG_STATUS_NONE {
        return sce_kernel_error(COMMON_DIALOG_ERROR_NOT_INITIALIZED);
    }
    let Some(output) = guest_output_pointer(output, 12) else {
        return sce_kernel_error(COMMON_DIALOG_ERROR_ARG_NULL);
    };
    write_guest_u32(output, 0);
    write_guest_u32(output + 4, state.services.message_dialog_result_code);
    write_guest_u32(output + 8, state.services.message_dialog_button);
    0
}

pub(crate) fn common_dialog_initialize(state: &mut ProbeDispatchState) -> u64 {
    state.services.common_dialog = InitializationState::Initialized;
    0
}

pub(crate) fn save_data_dialog_initialize(state: &mut ProbeDispatchState) -> u64 {
    if state.services.common_dialog != InitializationState::Initialized {
        return sce_kernel_error(COMMON_DIALOG_ERROR_NOT_SYSTEM_INITIALIZED);
    }
    if state.services.save_data_dialog_status != COMMON_DIALOG_STATUS_NONE.cast_signed() {
        return sce_kernel_error(COMMON_DIALOG_ERROR_DIALOG_ALREADY_INITIALIZED);
    }
    if state.services.common_dialog_owner.is_some() {
        return sce_kernel_error(COMMON_DIALOG_ERROR_BUSY);
    }
    state.services.save_data_dialog_status = COMMON_DIALOG_STATUS_INITIALIZED.cast_signed();
    state.services.common_dialog_owner = Some(CommonDialogOwner::SaveData);
    state.services.save_data_dialog_sequence = 0;
    state.services.save_data_dialog_result_code = SAVE_DATA_DIALOG_RESULT_USER_CANCELED;
    0
}

pub(crate) fn save_data_dialog_terminate(state: &mut ProbeDispatchState) -> u64 {
    if state.services.save_data_dialog_status == COMMON_DIALOG_STATUS_NONE.cast_signed() {
        return sce_kernel_error(COMMON_DIALOG_ERROR_NOT_INITIALIZED);
    }
    dismiss_save_data_dialog(state, HostDialogResultCode::Cancel);
    state.services.save_data_dialog_status = COMMON_DIALOG_STATUS_NONE.cast_signed();
    if state.services.common_dialog_owner == Some(CommonDialogOwner::SaveData) {
        state.services.common_dialog_owner = None;
    }
    0
}

pub(crate) fn save_data_dialog_open(parameters: u64, state: &mut ProbeDispatchState) -> u64 {
    if state.services.save_data_dialog_status == COMMON_DIALOG_STATUS_NONE.cast_signed() {
        return sce_kernel_error(COMMON_DIALOG_ERROR_NOT_INITIALIZED);
    }
    if parameters == 0 {
        return sce_kernel_error(COMMON_DIALOG_ERROR_ARG_NULL);
    }
    let sequence = begin_host_dialog();
    state.services.save_data_dialog_sequence = sequence;
    state.services.save_data_dialog_result_code = SAVE_DATA_DIALOG_RESULT_USER_CANCELED;
    state.services.save_data_dialog_status = COMMON_DIALOG_STATUS_RUNNING.cast_signed();
    publish_or_auto_dismiss(
        HostDialogRequest {
            kind: HostDialogKind::SaveData,
            title: "Saved Data".to_owned(),
            message: "Continue without loading saved data?".to_owned(),
            buttons: HostDialogButtons::OkCancel,
            sequence,
        },
        sequence,
    );
    0
}

pub(crate) fn save_data_dialog_update_status(state: &mut ProbeDispatchState) -> u64 {
    poll_save_data_dialog(state);
    state.services.save_data_dialog_status.cast_unsigned().into()
}

pub(crate) fn save_data_dialog_get_status(state: &ProbeDispatchState) -> u64 {
    state.services.save_data_dialog_status.cast_unsigned().into()
}

pub(crate) fn save_data_dialog_close(state: &mut ProbeDispatchState) -> u64 {
    if state.services.save_data_dialog_status == COMMON_DIALOG_STATUS_NONE.cast_signed() {
        return sce_kernel_error(COMMON_DIALOG_ERROR_NOT_INITIALIZED);
    }
    dismiss_save_data_dialog(state, HostDialogResultCode::Cancel);
    state.services.save_data_dialog_status = COMMON_DIALOG_STATUS_FINISHED.cast_signed();
    0
}

pub(crate) fn save_data_dialog_get_result(output: u64, state: &ProbeDispatchState) -> u64 {
    if state.services.save_data_dialog_status == COMMON_DIALOG_STATUS_NONE.cast_signed() {
        return sce_kernel_error(COMMON_DIALOG_ERROR_NOT_INITIALIZED);
    }
    let Some(output) = guest_output_pointer(output, 16) else {
        return sce_kernel_error(COMMON_DIALOG_ERROR_ARG_NULL);
    };
    write_guest_u32(output, 0);
    write_guest_u32(output + 4, state.services.save_data_dialog_result_code);
    write_guest_u32(output + 8, 0);
    write_guest_u32(output + 12, 0);
    0
}

pub(crate) fn save_data_dialog_is_ready_to_display(state: &ProbeDispatchState) -> u64 {
    u64::from(
        state.services.save_data_dialog_status != COMMON_DIALOG_STATUS_NONE.cast_signed(),
    )
}

/// Shows a non-modal trophy / notification toast on the host presenter.
pub(crate) fn publish_host_toast(kind: HostDialogKind, title: &str, message: &str) {
    let _ = publish_gpu_event(remu_gpu::GpuEvent::ShowHostToast(HostToastRequest {
        kind,
        title: title.to_owned(),
        message: message.to_owned(),
        duration_ms: 4_000,
    }));
}

fn publish_or_auto_dismiss(request: HostDialogRequest, sequence: u64) {
    let published = publish_gpu_event(remu_gpu::GpuEvent::ShowHostDialog(request));
    if !published || headless_dialogs_auto_dismiss() {
        // No live presenter (unit tests) or an explicit headless opt-in: complete immediately so
        // titles that poll UpdateStatus still make progress.
        complete_host_dialog(sequence, HostDialogResultCode::Cancel);
    }
}

fn headless_dialogs_auto_dismiss() -> bool {
    env_flag_enabled("REMU_HEADLESS_METAL") || env_flag_enabled("REMU_AUTO_DISMISS_DIALOGS")
}

fn env_flag_enabled(name: &str) -> bool {
    matches!(
        std::env::var_os(name).as_deref(),
        Some(value)
            if value.eq_ignore_ascii_case("1")
                || value.eq_ignore_ascii_case("true")
                || value.eq_ignore_ascii_case("yes")
    )
}

fn poll_message_dialog(state: &mut ProbeDispatchState) {
    if state.services.message_dialog_status != COMMON_DIALOG_STATUS_RUNNING {
        return;
    }
    let Some(result) = host_dialog_result(state.services.message_dialog_sequence) else {
        return;
    };
    apply_message_dialog_result(state, result.code);
    state.services.message_dialog_status = COMMON_DIALOG_STATUS_FINISHED;
}

fn poll_save_data_dialog(state: &mut ProbeDispatchState) {
    if state.services.save_data_dialog_status != COMMON_DIALOG_STATUS_RUNNING.cast_signed() {
        return;
    }
    let Some(result) = host_dialog_result(state.services.save_data_dialog_sequence) else {
        return;
    };
    apply_save_data_dialog_result(state, result.code);
    state.services.save_data_dialog_status = COMMON_DIALOG_STATUS_FINISHED.cast_signed();
}

fn dismiss_message_dialog(state: &mut ProbeDispatchState, code: HostDialogResultCode) {
    let sequence = state.services.message_dialog_sequence;
    if sequence != 0 {
        let _ = publish_gpu_event(remu_gpu::GpuEvent::HideHostDialog { sequence });
        cancel_host_dialog(sequence);
    }
    apply_message_dialog_result(state, code);
}

fn dismiss_save_data_dialog(state: &mut ProbeDispatchState, code: HostDialogResultCode) {
    let sequence = state.services.save_data_dialog_sequence;
    if sequence != 0 {
        let _ = publish_gpu_event(remu_gpu::GpuEvent::HideHostDialog { sequence });
        cancel_host_dialog(sequence);
    }
    apply_save_data_dialog_result(state, code);
}

fn apply_message_dialog_result(state: &mut ProbeDispatchState, code: HostDialogResultCode) {
    match code {
        HostDialogResultCode::Ok | HostDialogResultCode::Yes => {
            state.services.message_dialog_result_code = MESSAGE_DIALOG_RESULT_OK;
            state.services.message_dialog_button = MESSAGE_DIALOG_BUTTON_OK;
        }
        HostDialogResultCode::No => {
            state.services.message_dialog_result_code = MESSAGE_DIALOG_RESULT_OK;
            state.services.message_dialog_button = MESSAGE_DIALOG_BUTTON_NO;
        }
        HostDialogResultCode::Cancel | HostDialogResultCode::None => {
            state.services.message_dialog_result_code = MESSAGE_DIALOG_RESULT_USER_CANCELED;
            state.services.message_dialog_button = MESSAGE_DIALOG_BUTTON_INVALID;
        }
    }
}

fn apply_save_data_dialog_result(state: &mut ProbeDispatchState, code: HostDialogResultCode) {
    state.services.save_data_dialog_result_code = match code {
        HostDialogResultCode::Ok | HostDialogResultCode::Yes => SAVE_DATA_DIALOG_RESULT_OK,
        HostDialogResultCode::Cancel
        | HostDialogResultCode::No
        | HostDialogResultCode::None => SAVE_DATA_DIALOG_RESULT_USER_CANCELED,
    };
}
