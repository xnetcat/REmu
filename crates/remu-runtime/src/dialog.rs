use crate::{
    CommonDialogOwner, InitializationState, ProbeDispatchState, guest_output_pointer,
    sce_kernel_error, write_guest_u32,
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
const MESSAGE_DIALOG_RESULT_USER_CANCELED: u32 = 1;

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
    0
}

pub(crate) fn message_dialog_terminate(state: &mut ProbeDispatchState) -> u64 {
    if state.services.message_dialog_status == COMMON_DIALOG_STATUS_NONE {
        return sce_kernel_error(COMMON_DIALOG_ERROR_NOT_INITIALIZED);
    }
    state.services.message_dialog_status = COMMON_DIALOG_STATUS_NONE;
    if state.services.common_dialog_owner == Some(CommonDialogOwner::Message) {
        state.services.common_dialog_owner = None;
    }
    0
}

/// Opens a system message dialog.
///
/// REmu presents no system dialog surface, so the dialog is immediately dismissed rather than
/// left running: a title that polls `sceMsgDialogUpdateStatus` until the dialog finishes would
/// otherwise never make progress. The recorded result reports that the user canceled without
/// choosing a button, which is the outcome a dismissed dialog produces on the console.
pub(crate) fn message_dialog_open(parameters: u64, state: &mut ProbeDispatchState) -> u64 {
    if state.services.message_dialog_status == COMMON_DIALOG_STATUS_NONE {
        return sce_kernel_error(COMMON_DIALOG_ERROR_NOT_INITIALIZED);
    }
    if parameters == 0 {
        return sce_kernel_error(COMMON_DIALOG_ERROR_ARG_NULL);
    }
    state.services.message_dialog_status = COMMON_DIALOG_STATUS_RUNNING;
    0
}

pub(crate) fn message_dialog_update_status(state: &mut ProbeDispatchState) -> u64 {
    if state.services.message_dialog_status == COMMON_DIALOG_STATUS_RUNNING {
        state.services.message_dialog_status = COMMON_DIALOG_STATUS_FINISHED;
    }
    u64::from(state.services.message_dialog_status)
}

pub(crate) fn message_dialog_get_status(state: &ProbeDispatchState) -> u64 {
    u64::from(state.services.message_dialog_status)
}

pub(crate) fn message_dialog_close(state: &mut ProbeDispatchState) -> u64 {
    if state.services.message_dialog_status == COMMON_DIALOG_STATUS_NONE {
        return sce_kernel_error(COMMON_DIALOG_ERROR_NOT_INITIALIZED);
    }
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
    write_guest_u32(output + 4, MESSAGE_DIALOG_RESULT_USER_CANCELED);
    write_guest_u32(output + 8, MESSAGE_DIALOG_BUTTON_INVALID);
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
    0
}
