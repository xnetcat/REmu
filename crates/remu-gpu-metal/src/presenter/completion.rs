// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

use metal::{CommandBufferRef, MTLCommandBufferStatus};
use objc::runtime::Object;
use objc::{class, msg_send, sel, sel_impl};
use std::ffi::{CStr, c_char};
use std::fmt::Write as _;
use std::sync::OnceLock;
use std::thread;
use std::time::{Duration, Instant};

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

/// Yield rather than sleep for this long first. Batches almost always finish inside it, so the
/// common path keeps `waitUntilCompleted`'s latency instead of paying a sleep quantum per wait.
const SETTLE_YIELD_PERIOD: Duration = Duration::from_millis(2);

/// Poll granularity once a batch has outlived [`SETTLE_YIELD_PERIOD`] and is worth sleeping on.
const SETTLE_POLL_INTERVAL: Duration = Duration::from_micros(250);

const METAL_COMMAND_ERROR_TRACE_ENVIRONMENT_VARIABLE: &str = "REMU_TRACE_METAL_COMMAND_ERRORS";

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
pub(crate) fn wait_until_settled(command_buffer: &CommandBufferRef) -> bool {
    wait_until_settled_within(command_buffer, COMMAND_BUFFER_SETTLE_TIMEOUT)
}

/// [`wait_until_settled`] with an explicit budget, so tests can exercise the timeout directly.
pub(crate) fn wait_until_settled_within(
    command_buffer: &CommandBufferRef,
    timeout: Duration,
) -> bool {
    let started = Instant::now();
    loop {
        if command_buffer_is_settled(command_buffer) {
            return true;
        }
        let waited = started.elapsed();
        let Some(remaining) = timeout.checked_sub(waited) else {
            // Re-read once more: the batch may have settled inside the final interval.
            return command_buffer_is_settled(command_buffer);
        };
        if waited < SETTLE_YIELD_PERIOD {
            thread::yield_now();
        } else {
            thread::sleep(SETTLE_POLL_INTERVAL.min(remaining));
        }
    }
}

/// Commits `command_buffer` and reports whether it completed, naming `operation` on failure.
pub(crate) fn commit_and_settle(
    command_buffer: &CommandBufferRef,
    operation: &str,
) -> Result<(), String> {
    command_buffer.commit();
    settle(command_buffer, operation)
}

/// Waits for an already-committed `command_buffer` and reports whether it completed.
pub(crate) fn settle(command_buffer: &CommandBufferRef, operation: &str) -> Result<(), String> {
    if !wait_until_settled(command_buffer) {
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
        MetalCommandErrorDetail, format_command_buffer_failure, metal_command_error_code_name,
    };
    use metal::MTLCommandBufferStatus;

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
}
