// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

//! Native AppKit surfaces for guest common-dialog and toast requests.
//!
//! Modal dialogs are floating `NSPanel` overlays parented to the REmu window. Status stays
//! `RUNNING` until the user clicks a button or presses Cross/Enter (accept) or Circle/Escape
//! (cancel). The presenter keeps draining GPU work while the panel is up — `NSAlert::runModal`
//! would freeze submits and pad input.
//!
//! Toasts (trophies, notifications) use a second floating panel that auto-dismisses after the
//! requested duration.

use core_graphics_types::geometry::{CGPoint, CGRect, CGSize};
use objc::declare::ClassDecl;
use objc::runtime::{Class, Object, Sel, NO, YES};
use objc::{class, msg_send, sel, sel_impl};
use remu_gpu::{
    HostDialogButtons, HostDialogKind, HostDialogRequest, HostDialogResultCode, HostToastRequest,
    PAD_BUTTON_CIRCLE, PAD_BUTTON_CROSS, complete_host_dialog, host_dialog_is_pending,
    host_pad_buttons_held,
};
use std::ffi::CString;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

use super::present::headless_metal_requested;

fn auto_dismiss_dialogs_requested() -> bool {
    matches!(
        std::env::var_os("REMU_AUTO_DISMISS_DIALOGS").as_deref(),
        Some(value)
            if value.eq_ignore_ascii_case("1")
                || value.eq_ignore_ascii_case("true")
                || value.eq_ignore_ascii_case("yes")
    )
}

const PANEL_WIDTH: f64 = 420.0;
const DIALOG_HEIGHT: f64 = 200.0;
const TOAST_HEIGHT: f64 = 88.0;
const BUTTON_ACCEPT_TAG: isize = 1;
const BUTTON_CANCEL_TAG: isize = 2;
const NS_WINDOW_ABOVE: isize = 1;
const NS_BACKING_STORE_BUFFERED: usize = 2;
const NS_FLOATING_WINDOW_LEVEL: isize = 3;
/// `NSWindowStyleMaskTitled | NSWindowStyleMaskNonactivatingPanel`
const PANEL_STYLE_MASK: usize = 1 | 128;

struct ActiveDialog {
    sequence: u64,
    buttons: HostDialogButtons,
    panel: *mut Object,
}

// SAFETY: AppKit UI objects are only touched on the presenter (main) thread.
unsafe impl Send for ActiveDialog {}

struct ActiveToast {
    panel: *mut Object,
    expires_at: Instant,
}

// SAFETY: AppKit UI objects are only touched on the presenter (main) thread.
unsafe impl Send for ActiveToast {}

static ACTIVE_DIALOG: Mutex<Option<ActiveDialog>> = Mutex::new(None);
static ACTIVE_TOAST: Mutex<Option<ActiveToast>> = Mutex::new(None);
static LAST_PAD_BUTTONS: AtomicU32 = AtomicU32::new(0);
static DIALOG_TARGET_CLASS: OnceLock<&'static Class> = OnceLock::new();
static DIALOG_TARGET_INSTANCE: OnceLock<DialogTarget> = OnceLock::new();

struct DialogTarget(*mut Object);

// SAFETY: the singleton is only messaged from the AppKit main thread.
unsafe impl Send for DialogTarget {}
unsafe impl Sync for DialogTarget {}

/// Presents a non-blocking AppKit dialog panel and leaves it pending until dismissed.
pub(crate) fn present_host_dialog(request: &HostDialogRequest) {
    eprintln!(
        "[REMu][UI] dialog kind={:?} title={:?} message={:?} buttons={:?}",
        request.kind, request.title, request.message, request.buttons
    );
    if !host_dialog_is_pending() {
        return;
    }
    if headless_metal_requested() || auto_dismiss_dialogs_requested() {
        complete_host_dialog(request.sequence, HostDialogResultCode::Cancel);
        return;
    }
    close_active_dialog_panel();
    let Some(panel) = build_dialog_panel(request) else {
        complete_host_dialog(request.sequence, HostDialogResultCode::Cancel);
        return;
    };
    show_panel(panel, request.kind);
    *ACTIVE_DIALOG.lock().unwrap_or_else(|poisoned| poisoned.into_inner()) = Some(ActiveDialog {
        sequence: request.sequence,
        buttons: request.buttons,
        panel,
    });
    LAST_PAD_BUTTONS.store(host_pad_buttons_held(), Ordering::Release);
}

/// Dismisses a still-pending dialog when the guest calls Close/Terminate.
pub(crate) fn hide_host_dialog(sequence: u64) {
    let mut guard = ACTIVE_DIALOG
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    if let Some(active) = guard.as_ref()
        && active.sequence == sequence
    {
        close_panel(active.panel);
        *guard = None;
    }
    if host_dialog_is_pending() {
        complete_host_dialog(sequence, HostDialogResultCode::Cancel);
    }
}

/// Shows a non-modal host toast (trophy unlock, notification).
pub(crate) fn present_host_toast(request: &HostToastRequest) {
    eprintln!(
        "[REMu][UI] toast kind={:?} title={:?} message={:?} duration_ms={}",
        request.kind, request.title, request.message, request.duration_ms
    );
    if headless_metal_requested() {
        return;
    }
    close_active_toast_panel();
    let Some(panel) = build_toast_panel(request) else {
        return;
    };
    show_panel(panel, request.kind);
    let duration = Duration::from_millis(u64::from(request.duration_ms.max(1_500)));
    *ACTIVE_TOAST.lock().unwrap_or_else(|poisoned| poisoned.into_inner()) = Some(ActiveToast {
        panel,
        expires_at: Instant::now() + duration,
    });
}

/// Called from the presenter event loop so pad/keyboard can dismiss dialogs and toasts expire.
pub(crate) fn poll_host_ui() {
    poll_dialog_input();
    poll_toast_expiry();
}

fn poll_dialog_input() {
    let mut guard = ACTIVE_DIALOG
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let Some(active) = guard.as_ref() else {
        return;
    };
    if !host_dialog_is_pending() {
        close_panel(active.panel);
        *guard = None;
        return;
    }
    let held = host_pad_buttons_held();
    let previous = LAST_PAD_BUTTONS.swap(held, Ordering::AcqRel);
    let pressed = held & !previous;
    let code = if pressed & PAD_BUTTON_CROSS != 0 {
        Some(accept_code(active.buttons))
    } else if pressed & PAD_BUTTON_CIRCLE != 0 {
        Some(cancel_code(active.buttons))
    } else {
        None
    };
    if let Some(code) = code {
        let sequence = active.sequence;
        close_panel(active.panel);
        *guard = None;
        complete_host_dialog(sequence, code);
    }
}

fn poll_toast_expiry() {
    let mut guard = ACTIVE_TOAST
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let Some(active) = guard.as_ref() else {
        return;
    };
    if Instant::now() < active.expires_at {
        return;
    }
    close_panel(active.panel);
    *guard = None;
}

fn accept_code(buttons: HostDialogButtons) -> HostDialogResultCode {
    match buttons {
        HostDialogButtons::YesNo => HostDialogResultCode::Yes,
        HostDialogButtons::Ok | HostDialogButtons::OkCancel => HostDialogResultCode::Ok,
    }
}

fn cancel_code(buttons: HostDialogButtons) -> HostDialogResultCode {
    match buttons {
        HostDialogButtons::YesNo => HostDialogResultCode::No,
        HostDialogButtons::Ok => HostDialogResultCode::Ok,
        HostDialogButtons::OkCancel => HostDialogResultCode::Cancel,
    }
}

fn close_active_dialog_panel() {
    let mut guard = ACTIVE_DIALOG
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    if let Some(active) = guard.take() {
        close_panel(active.panel);
    }
}

fn close_active_toast_panel() {
    let mut guard = ACTIVE_TOAST
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    if let Some(active) = guard.take() {
        close_panel(active.panel);
    }
}

fn show_panel(panel: *mut Object, kind: HostDialogKind) {
    // SAFETY: `panel` is a freshly created NSPanel on the AppKit main thread.
    unsafe {
        let _: () = msg_send![panel, center];
        if kind == HostDialogKind::Trophy || kind == HostDialogKind::Notification {
            // Pin toasts near the top of the parent window when available.
            if let Some(parent) = key_window() {
                let parent_frame: CGRect = msg_send![parent, frame];
                let toast_frame = CGRect::new(
                    &CGPoint::new(
                        parent_frame.origin.x
                            + (parent_frame.size.width - PANEL_WIDTH).max(0.0) * 0.5,
                        parent_frame.origin.y + parent_frame.size.height - TOAST_HEIGHT - 24.0,
                    ),
                    &CGSize::new(PANEL_WIDTH, TOAST_HEIGHT),
                );
                let _: () = msg_send![panel, setFrame: toast_frame display: YES];
                let _: () = msg_send![parent, addChildWindow: panel ordered: NS_WINDOW_ABOVE];
            }
        } else if let Some(parent) = key_window() {
            let _: () = msg_send![parent, addChildWindow: panel ordered: NS_WINDOW_ABOVE];
        }
        let _: () = msg_send![panel, makeKeyAndOrderFront: std::ptr::null_mut::<Object>()];
    }
}

fn close_panel(panel: *mut Object) {
    if panel.is_null() {
        return;
    }
    // SAFETY: panel was retained when created and is only released here on the main thread.
    unsafe {
        let parent: *mut Object = msg_send![panel, parentWindow];
        if !parent.is_null() {
            let _: () = msg_send![parent, removeChildWindow: panel];
        }
        let _: () = msg_send![panel, orderOut: std::ptr::null_mut::<Object>()];
        let _: () = msg_send![panel, close];
        let _: () = msg_send![panel, release];
    }
}

fn build_dialog_panel(request: &HostDialogRequest) -> Option<*mut Object> {
    // SAFETY: AppKit UI construction on the presenter main thread.
    unsafe {
        let frame = CGRect::new(
            &CGPoint::new(0.0, 0.0),
            &CGSize::new(PANEL_WIDTH, DIALOG_HEIGHT),
        );
        let panel: *mut Object = msg_send![class!(NSPanel), alloc];
        let panel: *mut Object = msg_send![
            panel,
            initWithContentRect: frame
            styleMask: PANEL_STYLE_MASK
            backing: NS_BACKING_STORE_BUFFERED
            defer: NO
        ];
        if panel.is_null() {
            return None;
        }
        let _: () = msg_send![panel, setTitle: ns_string(&dialog_window_title(request))?];
        let _: () = msg_send![panel, setFloatingPanel: YES];
        let _: () = msg_send![panel, setLevel: NS_FLOATING_WINDOW_LEVEL];
        let _: () = msg_send![panel, setReleasedWhenClosed: NO];
        let _: () = msg_send![panel, setHidesOnDeactivate: NO];

        let content: *mut Object = msg_send![panel, contentView];
        if content.is_null() {
            let _: () = msg_send![panel, release];
            return None;
        }

        let title = make_label(
            &request.title,
            CGRect::new(&CGPoint::new(16.0, 150.0), &CGSize::new(PANEL_WIDTH - 32.0, 24.0)),
            true,
        )?;
        let message = make_label(
            &request.message,
            CGRect::new(&CGPoint::new(16.0, 70.0), &CGSize::new(PANEL_WIDTH - 32.0, 72.0)),
            false,
        )?;
        let hint = make_label(
            &button_hint(request.buttons),
            CGRect::new(&CGPoint::new(16.0, 44.0), &CGSize::new(PANEL_WIDTH - 32.0, 18.0)),
            false,
        )?;
        let _: () = msg_send![content, addSubview: title];
        let _: () = msg_send![content, addSubview: message];
        let _: () = msg_send![content, addSubview: hint];

        let target = dialog_target_instance()?;
        match request.buttons {
            HostDialogButtons::Ok => {
                add_action_button(content, target, "OK", BUTTON_ACCEPT_TAG, 160.0)?;
            }
            HostDialogButtons::OkCancel => {
                add_action_button(content, target, "Cancel", BUTTON_CANCEL_TAG, 100.0)?;
                add_action_button(content, target, "OK", BUTTON_ACCEPT_TAG, 220.0)?;
            }
            HostDialogButtons::YesNo => {
                add_action_button(content, target, "No", BUTTON_CANCEL_TAG, 100.0)?;
                add_action_button(content, target, "Yes", BUTTON_ACCEPT_TAG, 220.0)?;
            }
        }
        Some(panel)
    }
}

fn build_toast_panel(request: &HostToastRequest) -> Option<*mut Object> {
    // SAFETY: AppKit UI construction on the presenter main thread.
    unsafe {
        let frame = CGRect::new(
            &CGPoint::new(0.0, 0.0),
            &CGSize::new(PANEL_WIDTH, TOAST_HEIGHT),
        );
        let panel: *mut Object = msg_send![class!(NSPanel), alloc];
        let panel: *mut Object = msg_send![
            panel,
            initWithContentRect: frame
            styleMask: PANEL_STYLE_MASK
            backing: NS_BACKING_STORE_BUFFERED
            defer: NO
        ];
        if panel.is_null() {
            return None;
        }
        let _: () = msg_send![panel, setTitle: ns_string(&toast_window_title(request.kind))?];
        let _: () = msg_send![panel, setFloatingPanel: YES];
        let _: () = msg_send![panel, setLevel: NS_FLOATING_WINDOW_LEVEL];
        let _: () = msg_send![panel, setReleasedWhenClosed: NO];
        let _: () = msg_send![panel, setHidesOnDeactivate: NO];

        let content: *mut Object = msg_send![panel, contentView];
        if content.is_null() {
            let _: () = msg_send![panel, release];
            return None;
        }
        let title = make_label(
            &request.title,
            CGRect::new(&CGPoint::new(16.0, 48.0), &CGSize::new(PANEL_WIDTH - 32.0, 22.0)),
            true,
        )?;
        let message = make_label(
            &request.message,
            CGRect::new(&CGPoint::new(16.0, 16.0), &CGSize::new(PANEL_WIDTH - 32.0, 28.0)),
            false,
        )?;
        let _: () = msg_send![content, addSubview: title];
        let _: () = msg_send![content, addSubview: message];
        Some(panel)
    }
}

fn dialog_window_title(request: &HostDialogRequest) -> String {
    match request.kind {
        HostDialogKind::Message => "Message".to_owned(),
        HostDialogKind::SaveData => "Saved Data".to_owned(),
        HostDialogKind::Trophy => "Trophy".to_owned(),
        HostDialogKind::Notification => "Notification".to_owned(),
    }
}

fn toast_window_title(kind: HostDialogKind) -> String {
    match kind {
        HostDialogKind::Trophy => "Trophy".to_owned(),
        HostDialogKind::Notification => "Notification".to_owned(),
        HostDialogKind::Message => "Message".to_owned(),
        HostDialogKind::SaveData => "Saved Data".to_owned(),
    }
}

fn button_hint(buttons: HostDialogButtons) -> String {
    match buttons {
        HostDialogButtons::Ok => "Enter / Cross: OK".to_owned(),
        HostDialogButtons::OkCancel => "Enter / Cross: OK · Esc / Circle: Cancel".to_owned(),
        HostDialogButtons::YesNo => "Enter / Cross: Yes · Esc / Circle: No".to_owned(),
    }
}

fn make_label(text: &str, frame: CGRect, bold: bool) -> Option<*mut Object> {
    // SAFETY: NSTextField construction on the AppKit main thread.
    unsafe {
        let label: *mut Object = msg_send![class!(NSTextField), alloc];
        let label: *mut Object = msg_send![label, initWithFrame: frame];
        if label.is_null() {
            return None;
        }
        let _: () = msg_send![label, setStringValue: ns_string(text)?];
        let _: () = msg_send![label, setBezeled: NO];
        let _: () = msg_send![label, setDrawsBackground: NO];
        let _: () = msg_send![label, setEditable: NO];
        let _: () = msg_send![label, setSelectable: NO];
        let _: () = msg_send![label, setUsesSingleLineMode: NO];
        let font: *mut Object = if bold {
            msg_send![class!(NSFont), boldSystemFontOfSize: 14.0]
        } else {
            msg_send![class!(NSFont), systemFontOfSize: 12.0]
        };
        if !font.is_null() {
            let _: () = msg_send![label, setFont: font];
        }
        Some(label)
    }
}

fn add_action_button(
    content: *mut Object,
    target: *mut Object,
    title: &str,
    tag: isize,
    x: f64,
) -> Option<()> {
    // SAFETY: NSButton construction on the AppKit main thread; target is a retained singleton.
    unsafe {
        let button: *mut Object = msg_send![class!(NSButton), alloc];
        let button: *mut Object = msg_send![
            button,
            initWithFrame: CGRect::new(&CGPoint::new(x, 12.0), &CGSize::new(100.0, 28.0))
        ];
        if button.is_null() {
            return None;
        }
        let _: () = msg_send![button, setTitle: ns_string(title)?];
        let _: () = msg_send![button, setBezelStyle: 1_usize]; // NSBezelStyleRounded
        let _: () = msg_send![button, setTag: tag];
        let _: () = msg_send![button, setTarget: target];
        let _: () = msg_send![button, setAction: sel!(buttonClicked:)];
        let _: () = msg_send![content, addSubview: button];
        Some(())
    }
}

fn dialog_target_instance() -> Option<*mut Object> {
    let instance = DIALOG_TARGET_INSTANCE.get_or_init(|| {
        let class = DIALOG_TARGET_CLASS.get_or_init(|| {
            // SAFETY: class registration happens once on the AppKit main thread before first use.
            unsafe {
                if let Some(existing) = Class::get("REMuHostDialogTarget") {
                    existing
                } else {
                    let mut decl = ClassDecl::new("REMuHostDialogTarget", class!(NSObject))
                        .expect("REMuHostDialogTarget must be registerable once");
                    decl.add_method(
                        sel!(buttonClicked:),
                        dialog_button_clicked as extern "C" fn(&Object, Sel, *mut Object),
                    );
                    decl.register()
                }
            }
        });
        // SAFETY: allocated singleton target lives for the process.
        unsafe {
            let instance: *mut Object = msg_send![*class, new];
            DialogTarget(instance)
        }
    });
    if instance.0.is_null() {
        None
    } else {
        Some(instance.0)
    }
}

extern "C" fn dialog_button_clicked(_this: &Object, _cmd: Sel, sender: *mut Object) {
    // SAFETY: invoked by AppKit on the main thread for a live NSButton.
    let tag: isize = unsafe { msg_send![sender, tag] };
    let mut guard = ACTIVE_DIALOG
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let Some(active) = guard.as_ref() else {
        return;
    };
    let code = if tag == BUTTON_ACCEPT_TAG {
        accept_code(active.buttons)
    } else {
        cancel_code(active.buttons)
    };
    let sequence = active.sequence;
    close_panel(active.panel);
    *guard = None;
    complete_host_dialog(sequence, code);
}

fn key_window() -> Option<*mut Object> {
    // SAFETY: NSApplication is queried on the AppKit main thread.
    unsafe {
        let app: *mut Object = msg_send![class!(NSApplication), sharedApplication];
        if app.is_null() {
            return None;
        }
        let window: *mut Object = msg_send![app, keyWindow];
        if window.is_null() {
            let window: *mut Object = msg_send![app, mainWindow];
            if window.is_null() {
                None
            } else {
                Some(window)
            }
        } else {
            Some(window)
        }
    }
}

unsafe fn ns_string(text: &str) -> Option<*mut Object> {
    let c_string = CString::new(text).ok()?;
    // SAFETY: `c_string` lives for the duration of the msg_send; NSString copies the bytes.
    unsafe {
        let string: *mut Object =
            msg_send![class!(NSString), stringWithUTF8String: c_string.as_ptr()];
        if string.is_null() {
            None
        } else {
            Some(string)
        }
    }
}
