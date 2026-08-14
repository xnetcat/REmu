// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

use block::ConcreteBlock;
use metal::{CommandBufferRef, MTLCommandBufferStatus};
use objc::runtime::Object;
use objc::{class, msg_send, sel, sel_impl};
use std::ffi::{CStr, c_char};
use std::fmt::Write as _;
use std::sync::{Arc, Condvar, Mutex, OnceLock};
use std::time::Duration;

/// Longest a host thread blocks on one Metal command buffer before reporting it as stalled.
///
/// macOS kills `WindowServer` once it misses a 40-second watchdog checkin and panics the kernel at
/// 120 seconds. The presenter waits on guest batches from the `AppKit` main thread, so a submission
/// that never reaches a terminal state used to stall the host compositor instead of only the
/// emulator: `waitUntilCompleted` has no upper bound, and a batch the driver never schedules never
/// releases it. Silent Hill legitimately submits transition batches that remain scheduled for more
/// than five seconds, so keep a generous margin while still returning before the host watchdog.
///
/// Timing out does not make the command buffer terminal. Callers must retain its resources and
/// ordered guest completion effects until Metal later reports `Completed` or `Error`.
pub(crate) const COMMAND_BUFFER_SETTLE_TIMEOUT: Duration = Duration::from_secs(30);

const METAL_COMMAND_ERROR_TRACE_ENVIRONMENT_VARIABLE: &str = "REMU_TRACE_METAL_COMMAND_ERRORS";
const METAL_COMMAND_STALL_TRACE_ENVIRONMENT_VARIABLE: &str = "REMU_TRACE_METAL_STALLS";

/// Completion notification registered before a Metal command buffer is committed.
///
/// Metal forbids adding completion handlers after `commit`, while most settlement waits happen
/// later, when a committed guest batch reaches the front of a bounded queue. Keep this signal next
/// to the command buffer from creation through retirement so those waits can park on a condvar
/// instead of reading `status` every 250 microseconds.
#[derive(Clone)]
pub(crate) struct CommandBufferCompletion {
    state: Arc<(Mutex<bool>, Condvar)>,
}

impl CommandBufferCompletion {
    fn pending() -> Self {
        Self {
            state: Arc::new((Mutex::new(false), Condvar::new())),
        }
    }

    /// Attaches the one handler that makes later bounded waits event-driven. Call before `commit`.
    pub(crate) fn attach(command_buffer: &CommandBufferRef) -> Self {
        let completion = Self::pending();
        let notified = completion.clone();
        let handler = ConcreteBlock::new(move |_| notified.notify()).copy();
        command_buffer.add_completed_handler(&handler);
        completion
    }

    fn notify(&self) {
        let (completed, changed) = &*self.state;
        *completed
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = true;
        changed.notify_all();
    }

    fn wait_within(&self, timeout: Duration) -> bool {
        let (completed, changed) = &*self.state;
        let completed = completed
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let (completed, _) = changed
            .wait_timeout_while(completed, timeout, |completed| !*completed)
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        *completed
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct MetalCommandErrorDetail {
    domain: String,
    code: isize,
    localized_description: String,
    localized_failure_reason: Option<String>,
    localized_recovery_suggestion: Option<String>,
    encoder_info: Option<String>,
    user_info: Option<String>,
}

/// Blocks until `command_buffer` reaches a terminal state, giving up after
/// [`COMMAND_BUFFER_SETTLE_TIMEOUT`]. Returns whether it settled.
pub(crate) fn wait_until_settled(
    command_buffer: &CommandBufferRef,
    completion: &CommandBufferCompletion,
) -> bool {
    wait_until_settled_within(command_buffer, completion, COMMAND_BUFFER_SETTLE_TIMEOUT)
}

/// [`wait_until_settled`] with an explicit budget, so tests can exercise the timeout directly.
pub(crate) fn wait_until_settled_within(
    command_buffer: &CommandBufferRef,
    completion: &CommandBufferCompletion,
    timeout: Duration,
) -> bool {
    if command_buffer_is_settled(command_buffer) {
        return true;
    }
    let _ = completion.wait_within(timeout);
    // The handler records terminal notification, while Metal's status remains the authority for
    // whether this exact command buffer actually settled. Re-read after both wake and timeout.
    command_buffer_is_settled(command_buffer)
}

/// Commits `command_buffer` and reports whether it completed, naming `operation` on failure.
pub(crate) fn commit_and_settle(
    command_buffer: &CommandBufferRef,
    operation: &str,
) -> Result<(), String> {
    let completion = CommandBufferCompletion::attach(command_buffer);
    command_buffer.commit();
    settle(command_buffer, &completion, operation)
}

/// Waits for an already-committed `command_buffer` and reports whether it completed.
pub(crate) fn settle(
    command_buffer: &CommandBufferRef,
    completion: &CommandBufferCompletion,
    operation: &str,
) -> Result<(), String> {
    if !wait_until_settled(command_buffer, completion) {
        return Err(settle_timeout_message(operation, command_buffer));
    }
    let status = command_buffer.status();
    if status != MTLCommandBufferStatus::Completed {
        return Err(command_buffer_failure_message(operation, command_buffer));
    }
    Ok(())
}

/// Describes a terminal Metal command-buffer failure without changing its retirement policy.
///
/// Native `NSError` inspection is opt-in because encoder/user-info descriptions can be verbose.
/// The ordinary failure wording is retained when diagnostics are disabled.
pub(crate) fn command_buffer_failure_message(
    operation: &str,
    command_buffer: &CommandBufferRef,
) -> String {
    let status = command_buffer.status();
    if !trace_metal_command_errors() {
        return format_command_buffer_failure(operation, status, None);
    }
    let detail = capture_metal_command_error_detail(command_buffer);
    format_command_buffer_failure(operation, status, detail.as_ref())
}

/// Shared wording so a timed-out batch is never mistaken for a driver-reported failure.
pub(crate) fn settle_timeout_message(operation: &str, command_buffer: &CommandBufferRef) -> String {
    format!(
        "{operation} did not complete within {COMMAND_BUFFER_SETTLE_TIMEOUT:?} (status {:?})",
        command_buffer.status()
    )
}

fn command_buffer_is_settled(command_buffer: &CommandBufferRef) -> bool {
    matches!(
        command_buffer.status(),
        MTLCommandBufferStatus::Completed | MTLCommandBufferStatus::Error
    )
}

fn trace_metal_command_errors() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var_os(METAL_COMMAND_ERROR_TRACE_ENVIRONMENT_VARIABLE).is_some_and(|value| {
            let value = value.to_string_lossy();
            !value.is_empty() && value != "0" && !value.eq_ignore_ascii_case("false")
        })
    })
}

pub(crate) fn trace_metal_command_stalls() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var_os(METAL_COMMAND_STALL_TRACE_ENVIRONMENT_VARIABLE).is_some_and(|value| {
            let value = value.to_string_lossy();
            !value.is_empty() && value != "0" && !value.eq_ignore_ascii_case("false")
        })
    })
}

/// Summarizes native scheduling state without waiting for or mutating the command buffer.
///
/// A buffer that remains `Scheduled` can either be waiting behind older queue work or executing a
/// dispatch that never terminates. Metal's native GPU/kernel timestamps distinguish those cases:
/// a nonzero start with a zero end proves the driver began executing the buffer. The raw values are
/// intentionally retained because Metal reports them in its host-time domain.
pub(crate) fn command_buffer_native_state_summary(command_buffer: &CommandBufferRef) -> String {
    let (retained_references, error_options, kernel_start, kernel_end, gpu_start, gpu_end): (
        bool,
        usize,
        f64,
        f64,
        f64,
        f64,
    ) = unsafe {
        (
            msg_send![command_buffer, retainedReferences],
            msg_send![command_buffer, errorOptions],
            msg_send![command_buffer, kernelStartTime],
            msg_send![command_buffer, kernelEndTime],
            msg_send![command_buffer, GPUStartTime],
            msg_send![command_buffer, GPUEndTime],
        )
    };
    let error_present = unsafe {
        let error: *mut Object = msg_send![command_buffer, error];
        !error.is_null()
    };
    format_native_state_summary(
        retained_references,
        error_options,
        kernel_start,
        kernel_end,
        gpu_start,
        gpu_end,
        error_present,
    )
}

fn format_native_state_summary(
    retained_references: bool,
    error_options: usize,
    kernel_start: f64,
    kernel_end: f64,
    gpu_start: f64,
    gpu_end: f64,
    error_present: bool,
) -> String {
    format!(
        "retained={retained_references} error_options={error_options:#x} \
         kernel={kernel_start:.6}/{kernel_end:.6} gpu={gpu_start:.6}/{gpu_end:.6} \
         native_error={error_present}"
    )
}

fn format_command_buffer_failure(
    operation: &str,
    status: MTLCommandBufferStatus,
    detail: Option<&MetalCommandErrorDetail>,
) -> String {
    let mut message = format!("{operation} ended in {status:?}");
    let Some(detail) = detail else {
        return message;
    };
    let _ = write!(
        message,
        " [native_error domain={:?} code={}({}) description={:?}",
        detail.domain,
        detail.code,
        metal_command_error_code_name(detail.code),
        detail.localized_description,
    );
    if let Some(reason) = &detail.localized_failure_reason {
        let _ = write!(message, " failure_reason={reason:?}");
    }
    if let Some(suggestion) = &detail.localized_recovery_suggestion {
        let _ = write!(message, " recovery_suggestion={suggestion:?}");
    }
    if let Some(encoder_info) = &detail.encoder_info {
        let _ = write!(message, " encoder_info={encoder_info:?}");
    }
    if let Some(user_info) = &detail.user_info {
        let _ = write!(message, " user_info={user_info:?}");
    }
    message.push(']');
    message
}

fn metal_command_error_code_name(code: isize) -> &'static str {
    match code {
        0 => "None",
        1 => "Internal",
        2 => "Timeout",
        3 => "PageFault",
        4 => "Blacklisted",
        7 => "NotPermitted",
        8 => "OutOfMemory",
        9 => "InvalidResource",
        10 => "Memoryless",
        11 => "DeviceRemoved",
        12 => "StackOverflow",
        _ => "Unknown",
    }
}

fn capture_metal_command_error_detail(
    command_buffer: &CommandBufferRef,
) -> Option<MetalCommandErrorDetail> {
    unsafe {
        let error: *mut Object = msg_send![command_buffer, error];
        if error.is_null() {
            return None;
        }
        let domain: *mut Object = msg_send![error, domain];
        let code: isize = msg_send![error, code];
        let localized_description: *mut Object = msg_send![error, localizedDescription];
        let localized_failure_reason: *mut Object = msg_send![error, localizedFailureReason];
        let localized_recovery_suggestion: *mut Object =
            msg_send![error, localizedRecoverySuggestion];
        let user_info: *mut Object = msg_send![error, userInfo];
        Some(MetalCommandErrorDetail {
            domain: objc_string(domain).unwrap_or_else(|| "<unavailable>".to_owned()),
            code,
            localized_description: objc_string(localized_description)
                .unwrap_or_else(|| "<unavailable>".to_owned()),
            localized_failure_reason: objc_string(localized_failure_reason),
            localized_recovery_suggestion: objc_string(localized_recovery_suggestion),
            encoder_info: metal_command_buffer_encoder_info(user_info),
            user_info: objc_debug_description(user_info),
        })
    }
}

unsafe fn metal_command_buffer_encoder_info(user_info: *mut Object) -> Option<String> {
    if user_info.is_null() {
        return None;
    }
    let key: *mut Object = unsafe {
        msg_send![
            class!(NSString),
            stringWithUTF8String: c"MTLCommandBufferEncoderInfoErrorKey".as_ptr()
        ]
    };
    if key.is_null() {
        return None;
    }
    let encoder_info: *mut Object = unsafe { msg_send![user_info, objectForKey: key] };
    unsafe { objc_debug_description(encoder_info) }
}

unsafe fn objc_debug_description(object: *mut Object) -> Option<String> {
    if object.is_null() {
        return None;
    }
    let description: *mut Object = unsafe { msg_send![object, debugDescription] };
    unsafe { objc_string(description) }
}

unsafe fn objc_string(string: *mut Object) -> Option<String> {
    if string.is_null() {
        return None;
    }
    let bytes: *const c_char = unsafe { msg_send![string, UTF8String] };
    if bytes.is_null() {
        return None;
    }
    Some(
        unsafe { CStr::from_ptr(bytes) }
            .to_string_lossy()
            .into_owned(),
    )
}

#[cfg(test)]
mod tests {
    use super::{
        CommandBufferCompletion, MetalCommandErrorDetail, command_buffer_native_state_summary,
        format_command_buffer_failure, format_native_state_summary, metal_command_error_code_name,
    };
    use metal::{Device, MTLCommandBufferStatus};
    use std::time::Duration;

    #[test]
    fn completion_signal_wakes_a_parked_waiter() {
        let completion = CommandBufferCompletion::pending();
        assert!(
            !completion.wait_within(Duration::ZERO),
            "a pending command buffer must not manufacture a completion"
        );

        let notifier = completion.clone();
        let thread = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(20));
            notifier.notify();
        });
        assert!(
            completion.wait_within(Duration::from_secs(2)),
            "the completion notification must wake the condvar before its deadline"
        );
        thread.join().expect("completion notifier");
    }

    #[test]
    fn native_command_error_formatter_classifies_and_escapes_detail() {
        let detail = MetalCommandErrorDetail {
            domain: "MTLCommandBufferErrorDomain".to_owned(),
            code: 3,
            localized_description: "GPU page fault\nat encoder".to_owned(),
            localized_failure_reason: Some("invalid address".to_owned()),
            localized_recovery_suggestion: None,
            encoder_info: Some("(label = draw, errorState = faulted)".to_owned()),
            user_info: Some("{ MTLCommandBufferEncoderInfoErrorKey = (...); }".to_owned()),
        };

        let message = format_command_buffer_failure(
            "Metal guest command batch [draw(seq=7)]",
            MTLCommandBufferStatus::Error,
            Some(&detail),
        );

        assert!(message.contains("ended in Error"));
        assert!(message.contains("code=3(PageFault)"));
        assert!(message.contains("description=\"GPU page fault\\nat encoder\""));
        assert!(message.contains("failure_reason=\"invalid address\""));
        assert!(message.contains("encoder_info="));
        assert!(message.contains("user_info="));
        assert!(!message.contains('\n'));
    }

    #[test]
    fn native_command_error_codes_cover_actionable_driver_failures() {
        assert_eq!(metal_command_error_code_name(2), "Timeout");
        assert_eq!(metal_command_error_code_name(8), "OutOfMemory");
        assert_eq!(metal_command_error_code_name(9), "InvalidResource");
        assert_eq!(metal_command_error_code_name(99), "Unknown");
    }

    #[test]
    fn disabled_native_detail_preserves_existing_failure_wording() {
        assert_eq!(
            format_command_buffer_failure(
                "presentation command buffer",
                MTLCommandBufferStatus::Error,
                None,
            ),
            "presentation command buffer ended in Error"
        );
    }

    #[test]
    fn native_state_formatter_keeps_start_and_end_evidence_distinct() {
        let summary = format_native_state_summary(true, 1, 41.25, 0.0, 41.5, 0.0, false);

        assert!(summary.contains("retained=true"));
        assert!(summary.contains("error_options=0x1"));
        assert!(summary.contains("kernel=41.250000/0.000000"));
        assert!(summary.contains("gpu=41.500000/0.000000"));
        assert!(summary.contains("native_error=false"));
    }

    #[test]
    fn native_state_capture_accepts_a_completed_metal_command_buffer() {
        let device = Device::system_default().expect("Metal device");
        let queue = device.new_command_queue();
        let command_buffer = queue.new_command_buffer();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        let summary = command_buffer_native_state_summary(command_buffer);

        assert!(summary.contains("retained=true"));
        assert!(summary.contains("native_error=false"));
        assert!(summary.contains("gpu="));
    }
}
