// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

//! Common dialog and save-data dialog HLE backed by the present-path system UI overlay.
//!
//! Opening a dialog publishes [`remu_gpu::GpuEvent::ShowHostDialog`] to the presenter, which
//! composites a PS5-like panel over the guest scanout. Status stays `RUNNING` until the presenter
//! records a result through [`remu_gpu::complete_host_dialog`]. Headless runs and
//! `REMU_AUTO_DISMISS_DIALOGS` acknowledge one-button notices and conservatively decline actual
//! choices so titles that poll never hang without changing a notice into a cancellation.

use crate::{
    CommonDialogOwner, InitializationState, ProbeDispatchState, guest_output_pointer,
    publish_gpu_event, read_guest_cstring, read_guest_u32, read_guest_u64, sce_kernel_error,
    write_guest_u32,
};
use remu_gpu::{
    HostDialogButtons, HostDialogKind, HostDialogRequest, HostDialogResultCode, HostToastRequest,
    begin_host_dialog, cancel_host_dialog, complete_host_dialog, host_dialog_result,
    unattended_dialog_result,
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
const SAVE_DATA_DIALOG_MODE_INVALID: u32 = 0;
const SAVE_DATA_DIALOG_MODE_LIST: u32 = 1;
const SAVE_DATA_DIALOG_MODE_USER_MSG: u32 = 2;
const SAVE_DATA_DIALOG_MODE_SYSTEM_MSG: u32 = 3;
const SAVE_DATA_DIALOG_MODE_ERROR_CODE: u32 = 4;
const SAVE_DATA_DIALOG_MODE_PROGRESS_BAR: u32 = 5;
const SAVE_DATA_DIALOG_MODE_WIZARD_LIST: u32 = 6;
const SAVE_DATA_DIALOG_MODE_WIZARD_CONFIRM: u32 = 7;
const SAVE_DATA_DIALOG_BUTTON_INVALID: u32 = 0;
const SAVE_DATA_DIALOG_BUTTON_OK: u32 = 1;
const SAVE_DATA_DIALOG_BUTTON_NO: u32 = 2;
const SAVE_DATA_DIALOG_PARAM_MODE_OFFSET: u64 = 0x34;
const SAVE_DATA_DIALOG_PARAM_DISPLAY_TYPE_OFFSET: u64 = 0x38;
const SAVE_DATA_DIALOG_PARAM_ITEMS_OFFSET: u64 = 0x48;
const SAVE_DATA_DIALOG_PARAM_USER_MESSAGE_OFFSET: u64 = 0x50;
const SAVE_DATA_DIALOG_PARAM_SYSTEM_MESSAGE_OFFSET: u64 = 0x58;
const SAVE_DATA_DIALOG_PARAM_ERROR_CODE_OFFSET: u64 = 0x60;
const SAVE_DATA_DIALOG_PARAM_USER_DATA_OFFSET: u64 = 0x70;
const SAVE_DATA_DIALOG_RESULT_SIZE: usize = 0x48;

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

/// Orbis `OrbisCommonDialogBaseParam` is 48 bytes; `OrbisMsgDialogParam` then has `size`, `mode`,
/// padding, and `userMsgParam*` at +0x40. `OrbisMsgDialogUserMessageParam::msg` is at +0x8.
const MSG_DIALOG_PARAM_MODE_OFFSET: u64 = 0x38;
const MSG_DIALOG_PARAM_USER_MSG_OFFSET: u64 = 0x40;
const MSG_DIALOG_PARAM_PROG_BAR_OFFSET: u64 = 0x48;
const MSG_DIALOG_USER_BUTTON_TYPE_OFFSET: u64 = 0x0;
const MSG_DIALOG_USER_MSG_PTR_OFFSET: u64 = 0x8;
const MSG_DIALOG_MODE_USER_MSG: u32 = 1;
const MSG_DIALOG_MODE_PROGRESS_BAR: u32 = 2;
const MSG_DIALOG_MODE_SYSTEM_MSG: u32 = 3;
const MSG_DIALOG_BUTTON_OK: u32 = 0;
const MSG_DIALOG_BUTTON_YESNO: u32 = 1;
const MSG_DIALOG_BUTTON_OK_CANCEL: u32 = 3;
const MSG_DIALOG_BUTTON_YESNO_FOCUS_NO: u32 = 7;
const MSG_DIALOG_BUTTON_OK_CANCEL_FOCUS_CANCEL: u32 = 8;

/// Opens a system message dialog on the host presenter and leaves it `RUNNING` until dismissed.
pub(crate) fn message_dialog_open(parameters: u64, state: &mut ProbeDispatchState) -> u64 {
    if state.services.message_dialog_status == COMMON_DIALOG_STATUS_NONE {
        return sce_kernel_error(COMMON_DIALOG_ERROR_NOT_INITIALIZED);
    }
    if parameters == 0 {
        return sce_kernel_error(COMMON_DIALOG_ERROR_ARG_NULL);
    }
    let (kind, title, message, buttons) = decode_message_dialog_request(parameters);
    let sequence = begin_host_dialog();
    state.services.message_dialog_sequence = sequence;
    state.services.message_dialog_result_code = MESSAGE_DIALOG_RESULT_USER_CANCELED;
    state.services.message_dialog_button = MESSAGE_DIALOG_BUTTON_INVALID;
    state.services.message_dialog_status = COMMON_DIALOG_STATUS_RUNNING;
    publish_or_auto_dismiss(
        HostDialogRequest {
            kind,
            title,
            message,
            buttons,
            sequence,
        },
        sequence,
    );
    0
}

fn decode_message_dialog_request(
    parameters: u64,
) -> (HostDialogKind, String, String, HostDialogButtons) {
    let mode = read_guest_u32(parameters.saturating_add(MSG_DIALOG_PARAM_MODE_OFFSET));
    let user = read_guest_u64(parameters.saturating_add(MSG_DIALOG_PARAM_USER_MSG_OFFSET));
    let prog = read_guest_u64(parameters.saturating_add(MSG_DIALOG_PARAM_PROG_BAR_OFFSET));
    match mode {
        MSG_DIALOG_MODE_PROGRESS_BAR => {
            let message = read_user_message_text(prog)
                .or_else(|| read_user_message_text(user))
                .unwrap_or_else(|| "Please wait...".to_owned());
            (
                HostDialogKind::Progress,
                "Please Wait".to_owned(),
                message,
                HostDialogButtons::Ok,
            )
        }
        MSG_DIALOG_MODE_SYSTEM_MSG => (
            HostDialogKind::System,
            "System".to_owned(),
            "A system message is displayed.".to_owned(),
            HostDialogButtons::Ok,
        ),
        MSG_DIALOG_MODE_USER_MSG | _ => {
            let button_type = if user != 0 {
                read_guest_u32(user.saturating_add(MSG_DIALOG_USER_BUTTON_TYPE_OFFSET))
            } else {
                MSG_DIALOG_BUTTON_OK_CANCEL
            };
            let message = read_user_message_text(user).unwrap_or_else(|| {
                eprintln!(
                    "[REMu][UI] MsgDialog userMsgParam decode failed mode={mode} user={user:#x} prog={prog:#x}"
                );
                "Message".to_owned()
            });
            (
                HostDialogKind::Message,
                "Message".to_owned(),
                message,
                buttons_from_msg_dialog(button_type),
            )
        }
    }
}

fn read_user_message_text(user_msg_param: u64) -> Option<String> {
    if user_msg_param == 0 {
        return None;
    }
    let msg_ptr = read_guest_u64(user_msg_param.saturating_add(MSG_DIALOG_USER_MSG_PTR_OFFSET));
    read_guest_cstring(msg_ptr).filter(|text| !text.is_empty())
}

fn buttons_from_msg_dialog(button_type: u32) -> HostDialogButtons {
    match button_type {
        MSG_DIALOG_BUTTON_OK => HostDialogButtons::Ok,
        MSG_DIALOG_BUTTON_YESNO | MSG_DIALOG_BUTTON_YESNO_FOCUS_NO => HostDialogButtons::YesNo,
        MSG_DIALOG_BUTTON_OK_CANCEL | MSG_DIALOG_BUTTON_OK_CANCEL_FOCUS_CANCEL => {
            HostDialogButtons::OkCancel
        }
        _ => HostDialogButtons::OkCancel,
    }
}

pub(crate) fn message_dialog_update_status(state: &mut ProbeDispatchState) -> u64 {
    poll_message_dialog(state);
    u64::from(state.services.message_dialog_status)
}

/// A title that only ever calls `GetStatus` still has to observe the dialog finishing, so this
/// polls the host exactly as `UpdateStatus` does instead of returning a status nothing advances.
pub(crate) fn message_dialog_get_status(state: &mut ProbeDispatchState) -> u64 {
    poll_message_dialog(state);
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
    state.services.save_data_dialog_mode = SAVE_DATA_DIALOG_MODE_INVALID;
    state.services.save_data_dialog_result_code = SAVE_DATA_DIALOG_RESULT_USER_CANCELED;
    state.services.save_data_dialog_button = SAVE_DATA_DIALOG_BUTTON_INVALID;
    state.services.save_data_dialog_dir_name = 0;
    state.services.save_data_dialog_user_data = 0;
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
    let request = decode_save_data_dialog_request(parameters);
    let sequence = begin_host_dialog();
    state.services.save_data_dialog_sequence = sequence;
    state.services.save_data_dialog_mode = request.mode;
    state.services.save_data_dialog_result_code = SAVE_DATA_DIALOG_RESULT_USER_CANCELED;
    state.services.save_data_dialog_button = SAVE_DATA_DIALOG_BUTTON_INVALID;
    state.services.save_data_dialog_dir_name = request.dir_name;
    state.services.save_data_dialog_user_data = request.user_data;
    state.services.save_data_dialog_status = COMMON_DIALOG_STATUS_RUNNING.cast_signed();
    publish_or_auto_dismiss(
        HostDialogRequest {
            kind: HostDialogKind::SaveData,
            title: "Saved Data".to_owned(),
            message: request.message,
            buttons: request.buttons,
            sequence,
        },
        sequence,
    );
    0
}

pub(crate) fn save_data_dialog_update_status(state: &mut ProbeDispatchState) -> u64 {
    poll_save_data_dialog(state);
    state
        .services
        .save_data_dialog_status
        .cast_unsigned()
        .into()
}

/// See [`message_dialog_get_status`]: a `GetStatus`-only poll loop must still make progress.
pub(crate) fn save_data_dialog_get_status(state: &mut ProbeDispatchState) -> u64 {
    poll_save_data_dialog(state);
    state
        .services
        .save_data_dialog_status
        .cast_unsigned()
        .into()
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
    let Some(output) = guest_output_pointer(output, SAVE_DATA_DIALOG_RESULT_SIZE) else {
        return sce_kernel_error(COMMON_DIALOG_ERROR_ARG_NULL);
    };
    // SAFETY: `guest_output_pointer` validated the complete SDK `SceSaveDataDialogResult` span.
    unsafe { std::ptr::write_bytes(output as *mut u8, 0, SAVE_DATA_DIALOG_RESULT_SIZE) };
    write_guest_u32(output, state.services.save_data_dialog_mode);
    write_guest_u32(output + 4, state.services.save_data_dialog_result_code);
    write_guest_u32(output + 8, state.services.save_data_dialog_button);
    crate::write_guest_u64(output + 0x10, state.services.save_data_dialog_dir_name);
    crate::write_guest_u64(output + 0x20, state.services.save_data_dialog_user_data);
    0
}

struct DecodedSaveDataDialogRequest {
    mode: u32,
    message: String,
    buttons: HostDialogButtons,
    dir_name: u64,
    user_data: u64,
}

fn decode_save_data_dialog_request(parameters: u64) -> DecodedSaveDataDialogRequest {
    let mode = read_guest_u32(parameters.saturating_add(SAVE_DATA_DIALOG_PARAM_MODE_OFFSET));
    let display_type =
        read_guest_u32(parameters.saturating_add(SAVE_DATA_DIALOG_PARAM_DISPLAY_TYPE_OFFSET));
    let user_data =
        read_guest_u64(parameters.saturating_add(SAVE_DATA_DIALOG_PARAM_USER_DATA_OFFSET));
    let mut dir_name = 0;
    let (message, buttons) = match mode {
        SAVE_DATA_DIALOG_MODE_LIST => {
            let items =
                read_guest_u64(parameters.saturating_add(SAVE_DATA_DIALOG_PARAM_ITEMS_OFFSET));
            if items != 0 && read_guest_u32(items + 0x18) != 0 {
                dir_name = read_guest_u64(items + 0x10);
            }
            (
                format!("Select saved data to {}.", save_data_action(display_type)),
                HostDialogButtons::OkCancel,
            )
        }
        SAVE_DATA_DIALOG_MODE_USER_MSG => {
            let user = read_guest_u64(
                parameters.saturating_add(SAVE_DATA_DIALOG_PARAM_USER_MESSAGE_OFFSET),
            );
            if user == 0 {
                ("Saved data message".to_owned(), HostDialogButtons::Ok)
            } else {
                let button_type = read_guest_u32(user);
                let message = read_guest_cstring(read_guest_u64(user + 0x08))
                    .filter(|value| !value.is_empty())
                    .unwrap_or_else(|| "Saved data message".to_owned());
                (message, save_data_buttons(button_type))
            }
        }
        SAVE_DATA_DIALOG_MODE_SYSTEM_MSG => {
            let system = read_guest_u64(
                parameters.saturating_add(SAVE_DATA_DIALOG_PARAM_SYSTEM_MESSAGE_OFFSET),
            );
            let message_type = if system != 0 {
                read_guest_u32(system)
            } else {
                0
            };
            save_data_system_message(message_type, display_type)
        }
        SAVE_DATA_DIALOG_MODE_ERROR_CODE => {
            let error =
                read_guest_u64(parameters.saturating_add(SAVE_DATA_DIALOG_PARAM_ERROR_CODE_OFFSET));
            let code = if error != 0 { read_guest_u32(error) } else { 0 };
            (
                format!("Saved data error {code:#010x}"),
                HostDialogButtons::Ok,
            )
        }
        SAVE_DATA_DIALOG_MODE_PROGRESS_BAR => {
            ("Processing saved data...".to_owned(), HostDialogButtons::Ok)
        }
        SAVE_DATA_DIALOG_MODE_WIZARD_LIST | SAVE_DATA_DIALOG_MODE_WIZARD_CONFIRM => (
            "Confirm the saved data operation.".to_owned(),
            HostDialogButtons::YesNo,
        ),
        _ => ("Saved data message".to_owned(), HostDialogButtons::Ok),
    };
    DecodedSaveDataDialogRequest {
        mode,
        message,
        buttons,
        dir_name,
        user_data,
    }
}

fn save_data_buttons(button_type: u32) -> HostDialogButtons {
    match button_type {
        1 => HostDialogButtons::YesNo,
        3 => HostDialogButtons::OkCancel,
        _ => HostDialogButtons::Ok,
    }
}

fn save_data_action(display_type: u32) -> &'static str {
    match display_type {
        1 => "save",
        2 => "load",
        3 => "delete",
        _ => "use",
    }
}

fn save_data_system_message(message_type: u32, display_type: u32) -> (String, HostDialogButtons) {
    match message_type {
        1 => ("There is no saved data.".to_owned(), HostDialogButtons::Ok),
        2 => (
            format!(
                "Do you want to {} this saved data?",
                save_data_action(display_type)
            ),
            HostDialogButtons::YesNo,
        ),
        3 => (
            "Do you want to overwrite this saved data?".to_owned(),
            HostDialogButtons::YesNo,
        ),
        4 => (
            "There is not enough free space.".to_owned(),
            HostDialogButtons::Ok,
        ),
        5 => ("Processing saved data...".to_owned(), HostDialogButtons::Ok),
        6 => (
            "The saved data is corrupted.".to_owned(),
            HostDialogButtons::Ok,
        ),
        7 => (
            "The saved data operation finished.".to_owned(),
            HostDialogButtons::Ok,
        ),
        8 => (
            "There is not enough free space. You can continue without saving.".to_owned(),
            HostDialogButtons::Ok,
        ),
        14 => (
            "The total saved data size is too large.".to_owned(),
            HostDialogButtons::Ok,
        ),
        _ => ("Saved data message".to_owned(), HostDialogButtons::Ok),
    }
}

pub(crate) fn save_data_dialog_is_ready_to_display(state: &ProbeDispatchState) -> u64 {
    u64::from(state.services.save_data_dialog_status != COMMON_DIALOG_STATUS_NONE.cast_signed())
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

/// Opens a host error dialog and records it as the active common-dialog owner.
pub(crate) fn error_dialog_initialize(state: &mut ProbeDispatchState) -> u64 {
    if state.services.common_dialog != InitializationState::Initialized {
        return sce_kernel_error(COMMON_DIALOG_ERROR_NOT_SYSTEM_INITIALIZED);
    }
    if state.services.error_dialog_status != COMMON_DIALOG_STATUS_NONE {
        return sce_kernel_error(COMMON_DIALOG_ERROR_DIALOG_ALREADY_INITIALIZED);
    }
    if state.services.common_dialog_owner.is_some() {
        return sce_kernel_error(COMMON_DIALOG_ERROR_BUSY);
    }
    state.services.error_dialog_status = COMMON_DIALOG_STATUS_INITIALIZED;
    state.services.common_dialog_owner = Some(CommonDialogOwner::Error);
    state.services.error_dialog_sequence = 0;
    state.services.error_dialog_result_code = MESSAGE_DIALOG_RESULT_USER_CANCELED;
    0
}

pub(crate) fn error_dialog_terminate(state: &mut ProbeDispatchState) -> u64 {
    if state.services.error_dialog_status == COMMON_DIALOG_STATUS_NONE {
        return sce_kernel_error(COMMON_DIALOG_ERROR_NOT_INITIALIZED);
    }
    dismiss_error_dialog(state);
    state.services.error_dialog_status = COMMON_DIALOG_STATUS_NONE;
    if state.services.common_dialog_owner == Some(CommonDialogOwner::Error) {
        state.services.common_dialog_owner = None;
    }
    0
}

/// `sceErrorDialogOpen`. Titles pass a parameter block whose first fields are the common-dialog
/// base and an SCE error code; the host dialog surfaces that code so the offline path still
/// completes when `REMU_AUTO_DISMISS_DIALOGS` is set.
pub(crate) fn error_dialog_open(parameters: u64, state: &mut ProbeDispatchState) -> u64 {
    if state.services.error_dialog_status == COMMON_DIALOG_STATUS_NONE {
        return sce_kernel_error(COMMON_DIALOG_ERROR_NOT_INITIALIZED);
    }
    if parameters == 0 {
        return sce_kernel_error(COMMON_DIALOG_ERROR_ARG_NULL);
    }
    // After the 0x30-byte common-dialog base, the first dword is the SCE error code.
    let error_code = read_guest_u32(parameters.saturating_add(0x30));
    let message = format!("Error {error_code:#010x}");
    let sequence = begin_host_dialog();
    state.services.error_dialog_sequence = sequence;
    state.services.error_dialog_result_code = MESSAGE_DIALOG_RESULT_USER_CANCELED;
    state.services.error_dialog_status = COMMON_DIALOG_STATUS_RUNNING;
    publish_or_auto_dismiss(
        HostDialogRequest {
            kind: HostDialogKind::Error,
            title: "Error".to_owned(),
            message,
            buttons: HostDialogButtons::Ok,
            sequence,
        },
        sequence,
    );
    0
}

pub(crate) fn error_dialog_update_status(state: &mut ProbeDispatchState) -> u64 {
    poll_error_dialog(state);
    u64::from(state.services.error_dialog_status)
}

/// See [`message_dialog_get_status`]: a `GetStatus`-only poll loop must still make progress.
pub(crate) fn error_dialog_get_status(state: &mut ProbeDispatchState) -> u64 {
    poll_error_dialog(state);
    u64::from(state.services.error_dialog_status)
}

pub(crate) fn error_dialog_close(state: &mut ProbeDispatchState) -> u64 {
    if state.services.error_dialog_status == COMMON_DIALOG_STATUS_NONE {
        return sce_kernel_error(COMMON_DIALOG_ERROR_NOT_INITIALIZED);
    }
    dismiss_error_dialog(state);
    state.services.error_dialog_status = COMMON_DIALOG_STATUS_FINISHED;
    0
}

pub(crate) fn error_dialog_get_result(output: u64, state: &ProbeDispatchState) -> u64 {
    if state.services.error_dialog_status == COMMON_DIALOG_STATUS_NONE {
        return sce_kernel_error(COMMON_DIALOG_ERROR_NOT_INITIALIZED);
    }
    let Some(output) = guest_output_pointer(output, 8) else {
        return sce_kernel_error(COMMON_DIALOG_ERROR_ARG_NULL);
    };
    write_guest_u32(output, state.services.error_dialog_result_code);
    write_guest_u32(output + 4, 0);
    0
}

/// Opens a host error dialog (used when titles hit stubbed error-dialog entry points).
#[allow(dead_code)] // retained for one-shot error toast paths outside the dialog lifecycle
pub(crate) fn open_error_dialog(message: &str) -> u64 {
    let sequence = begin_host_dialog();
    publish_or_auto_dismiss(
        HostDialogRequest {
            kind: HostDialogKind::Error,
            title: "Error".to_owned(),
            message: message.to_owned(),
            buttons: HostDialogButtons::Ok,
            sequence,
        },
        sequence,
    );
    0
}

/// Opens a host progress / please-wait dialog.
#[allow(dead_code)] // wired when progress-dialog NIDs are hooked up
pub(crate) fn open_progress_dialog(message: &str) -> u64 {
    let sequence = begin_host_dialog();
    publish_or_auto_dismiss(
        HostDialogRequest {
            kind: HostDialogKind::Progress,
            title: "Please Wait".to_owned(),
            message: message.to_owned(),
            buttons: HostDialogButtons::Ok,
            sequence,
        },
        sequence,
    );
    0
}

/// Opens a host IME / on-screen keyboard stand-in.
pub(crate) fn ime_keyboard_open() -> u64 {
    let sequence = begin_host_dialog();
    publish_or_auto_dismiss(
        HostDialogRequest {
            kind: HostDialogKind::Ime,
            title: "Keyboard".to_owned(),
            message: "On-screen keyboard is unavailable. Use the host keyboard, then press OK."
                .to_owned(),
            buttons: HostDialogButtons::Ok,
            sequence,
        },
        sequence,
    );
    0
}

fn publish_or_auto_dismiss(request: HostDialogRequest, sequence: u64) {
    let unattended_result = unattended_dialog_result(request.buttons);
    let published = publish_gpu_event(remu_gpu::GpuEvent::ShowHostDialog(request));
    if !published || headless_dialogs_auto_dismiss() {
        // No live presenter (unit tests) or an explicit headless opt-in: complete immediately so
        // titles that poll UpdateStatus still make progress.
        complete_host_dialog(sequence, unattended_result);
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

fn poll_error_dialog(state: &mut ProbeDispatchState) {
    if state.services.error_dialog_status != COMMON_DIALOG_STATUS_RUNNING {
        return;
    }
    let Some(result) = host_dialog_result(state.services.error_dialog_sequence) else {
        return;
    };
    apply_error_dialog_result(state, result.code);
    state.services.error_dialog_status = COMMON_DIALOG_STATUS_FINISHED;
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

fn dismiss_error_dialog(state: &mut ProbeDispatchState) {
    let sequence = state.services.error_dialog_sequence;
    if sequence != 0 {
        let _ = publish_gpu_event(remu_gpu::GpuEvent::HideHostDialog { sequence });
        cancel_host_dialog(sequence);
    }
    apply_error_dialog_result(state, HostDialogResultCode::Cancel);
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
    let (result, button) = match code {
        HostDialogResultCode::Ok | HostDialogResultCode::Yes => {
            (SAVE_DATA_DIALOG_RESULT_OK, SAVE_DATA_DIALOG_BUTTON_OK)
        }
        HostDialogResultCode::No => (SAVE_DATA_DIALOG_RESULT_OK, SAVE_DATA_DIALOG_BUTTON_NO),
        HostDialogResultCode::Cancel | HostDialogResultCode::None => (
            SAVE_DATA_DIALOG_RESULT_USER_CANCELED,
            SAVE_DATA_DIALOG_BUTTON_INVALID,
        ),
    };
    state.services.save_data_dialog_result_code = result;
    state.services.save_data_dialog_button = button;
}

fn apply_error_dialog_result(state: &mut ProbeDispatchState, code: HostDialogResultCode) {
    state.services.error_dialog_result_code = match code {
        HostDialogResultCode::Ok | HostDialogResultCode::Yes => MESSAGE_DIALOG_RESULT_OK,
        HostDialogResultCode::Cancel | HostDialogResultCode::No | HostDialogResultCode::None => {
            MESSAGE_DIALOG_RESULT_USER_CANCELED
        }
    };
}
