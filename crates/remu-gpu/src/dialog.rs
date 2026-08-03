// SPDX-License-Identifier: GPL-2.0-or-later

//! Host-owned dialog and toast surfaces presented to the user.
//!
//! The guest common-dialog ABIs (`sceMsgDialog*`, `sceSaveDataDialog*`, trophy unlock toasts) do
//! not own a UI thread. They publish a request through [`GpuEvent`] and poll
//! [`host_dialog_result`] until the presenter records a dismissal. Headless and unit-test runs
//! without a live presenter auto-complete with a cancel result so titles never hang on the poll.

use std::sync::atomic::{AtomicI32, AtomicU32, AtomicU64, Ordering};

/// System UI surfaces the presenter can composite over the guest scanout.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u32)]
pub enum HostDialogKind {
    Message = 1,
    SaveData = 2,
    Trophy = 3,
    Notification = 4,
    Error = 5,
    Progress = 6,
    Ime = 7,
    System = 8,
}

impl HostDialogKind {
    #[must_use]
    pub const fn from_raw(value: u32) -> Option<Self> {
        match value {
            1 => Some(Self::Message),
            2 => Some(Self::SaveData),
            3 => Some(Self::Trophy),
            4 => Some(Self::Notification),
            5 => Some(Self::Error),
            6 => Some(Self::Progress),
            7 => Some(Self::Ime),
            8 => Some(Self::System),
            _ => None,
        }
    }
}

/// Buttons a modal dialog may offer.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u32)]
pub enum HostDialogButtons {
    Ok = 1,
    OkCancel = 2,
    YesNo = 3,
}

/// How the user (or headless auto-dismiss) closed a dialog.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(i32)]
pub enum HostDialogResultCode {
    None = 0,
    Ok = 1,
    Cancel = 2,
    Yes = 3,
    No = 4,
}

impl HostDialogResultCode {
    #[must_use]
    pub const fn from_raw(value: i32) -> Option<Self> {
        match value {
            0 => Some(Self::None),
            1 => Some(Self::Ok),
            2 => Some(Self::Cancel),
            3 => Some(Self::Yes),
            4 => Some(Self::No),
            _ => None,
        }
    }
}

/// A modal dialog request published to the presenter.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct HostDialogRequest {
    pub kind: HostDialogKind,
    pub title: String,
    pub message: String,
    pub buttons: HostDialogButtons,
    /// Sequence stamped when the request is armed so a stale dismiss cannot complete a later open.
    pub sequence: u64,
}

/// A non-modal toast (trophy unlock, system notice) that does not block dialog polling.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct HostToastRequest {
    pub kind: HostDialogKind,
    pub title: String,
    pub message: String,
    /// Milliseconds the toast remains visible before the presenter clears it.
    pub duration_ms: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct HostDialogResult {
    pub sequence: u64,
    pub code: HostDialogResultCode,
}

const DIALOG_IDLE: u32 = 0;
const DIALOG_PENDING: u32 = 1;
const DIALOG_COMPLETED: u32 = 2;

static DIALOG_STATE: AtomicU32 = AtomicU32::new(DIALOG_IDLE);
static DIALOG_SEQUENCE: AtomicU64 = AtomicU64::new(0);
static DIALOG_RESULT_CODE: AtomicI32 = AtomicI32::new(HostDialogResultCode::None as i32);
static DIALOG_RESULT_SEQUENCE: AtomicU64 = AtomicU64::new(0);

/// Arms a new dialog poll slot and returns the sequence the presenter must complete against.
#[must_use]
pub fn begin_host_dialog() -> u64 {
    let sequence = DIALOG_SEQUENCE
        .fetch_add(1, Ordering::AcqRel)
        .wrapping_add(1);
    DIALOG_RESULT_CODE.store(HostDialogResultCode::None as i32, Ordering::Release);
    DIALOG_RESULT_SEQUENCE.store(0, Ordering::Release);
    DIALOG_STATE.store(DIALOG_PENDING, Ordering::Release);
    sequence
}

/// Records the user's choice for the armed dialog sequence.
pub fn complete_host_dialog(sequence: u64, code: HostDialogResultCode) {
    if sequence == 0 {
        return;
    }
    DIALOG_RESULT_CODE.store(code as i32, Ordering::Release);
    DIALOG_RESULT_SEQUENCE.store(sequence, Ordering::Release);
    DIALOG_STATE.store(DIALOG_COMPLETED, Ordering::Release);
}

/// Cancels any pending dialog without a user gesture (close API, terminate, headless).
pub fn cancel_host_dialog(sequence: u64) {
    complete_host_dialog(sequence, HostDialogResultCode::Cancel);
}

/// Returns the completed result when it matches `sequence`.
#[must_use]
pub fn host_dialog_result(sequence: u64) -> Option<HostDialogResult> {
    if DIALOG_STATE.load(Ordering::Acquire) != DIALOG_COMPLETED {
        return None;
    }
    let completed = DIALOG_RESULT_SEQUENCE.load(Ordering::Acquire);
    if completed != sequence {
        return None;
    }
    let code = HostDialogResultCode::from_raw(DIALOG_RESULT_CODE.load(Ordering::Acquire))?;
    Some(HostDialogResult { sequence, code })
}

/// True while a dialog is waiting for the presenter or the user.
#[must_use]
pub fn host_dialog_is_pending() -> bool {
    DIALOG_STATE.load(Ordering::Acquire) == DIALOG_PENDING
}

/// Clears dialog state between guest process resets.
pub fn reset_host_dialog_state() {
    DIALOG_STATE.store(DIALOG_IDLE, Ordering::Release);
    DIALOG_RESULT_CODE.store(HostDialogResultCode::None as i32, Ordering::Release);
    DIALOG_RESULT_SEQUENCE.store(0, Ordering::Release);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dialog_completion_matches_the_armed_sequence() {
        reset_host_dialog_state();
        let sequence = begin_host_dialog();
        assert!(host_dialog_is_pending());
        assert!(host_dialog_result(sequence).is_none());
        complete_host_dialog(sequence, HostDialogResultCode::Ok);
        assert_eq!(
            host_dialog_result(sequence),
            Some(HostDialogResult {
                sequence,
                code: HostDialogResultCode::Ok,
            })
        );
        assert!(host_dialog_result(sequence.wrapping_add(1)).is_none());
        reset_host_dialog_state();
    }
}
