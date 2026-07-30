// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

//! Present-path system UI for guest common-dialog and toast requests.
//!
//! Dialogs stay `RUNNING` until Cross/Enter (accept) or Circle/Escape (cancel). The presenter
//! composites a PS5-like overlay rasterized by [`remu_gpu::rasterize_system_ui`] after the guest
//! scanout — no AppKit / Swift panels. Headless and `REMU_AUTO_DISMISS_DIALOGS` skip drawing and
//! auto-cancel so CI never blocks.

use remu_gpu::{
    HostDialogButtons, HostDialogRequest, HostDialogResultCode, HostToastRequest,
    SystemUiFramebuffer, complete_host_dialog, host_dialog_is_pending, host_pad_buttons_held,
    rasterize_system_ui, PAD_BUTTON_CIRCLE, PAD_BUTTON_CROSS,
};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Mutex;
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

struct ActiveDialog {
    sequence: u64,
    buttons: HostDialogButtons,
    request: HostDialogRequest,
}

struct ActiveToast {
    request: HostToastRequest,
    expires_at: Instant,
}

struct OverlayState {
    dialog: Option<ActiveDialog>,
    toast: Option<ActiveToast>,
    framebuffer: Option<SystemUiFramebuffer>,
    dirty: bool,
}

impl OverlayState {
    const fn new() -> Self {
        Self {
            dialog: None,
            toast: None,
            framebuffer: None,
            dirty: false,
        }
    }

    fn rebuild(&mut self) {
        let dialog = self.dialog.as_ref().map(|active| &active.request);
        let toast = self.toast.as_ref().map(|active| &active.request);
        if dialog.is_none() && toast.is_none() {
            self.framebuffer = None;
        } else {
            self.framebuffer = Some(rasterize_system_ui(dialog, toast));
        }
        self.dirty = true;
    }
}

static OVERLAY: Mutex<OverlayState> = Mutex::new(OverlayState::new());
static LAST_PAD_BUTTONS: AtomicU32 = AtomicU32::new(0);

/// Arms a modal overlay (or auto-cancels when headless / auto-dismiss).
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
    let mut state = OVERLAY.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    state.dialog = Some(ActiveDialog {
        sequence: request.sequence,
        buttons: request.buttons,
        request: request.clone(),
    });
    state.rebuild();
    LAST_PAD_BUTTONS.store(host_pad_buttons_held(), Ordering::Release);
}

/// Dismisses a still-pending dialog when the guest calls Close/Terminate.
pub(crate) fn hide_host_dialog(sequence: u64) {
    let mut state = OVERLAY.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    if state
        .dialog
        .as_ref()
        .is_some_and(|active| active.sequence == sequence)
    {
        state.dialog = None;
        state.rebuild();
    }
    if host_dialog_is_pending() {
        complete_host_dialog(sequence, HostDialogResultCode::Cancel);
    }
}

/// Shows a non-modal toast (trophy unlock, notification).
pub(crate) fn present_host_toast(request: &HostToastRequest) {
    eprintln!(
        "[REMu][UI] toast kind={:?} title={:?} message={:?} duration_ms={}",
        request.kind, request.title, request.message, request.duration_ms
    );
    if headless_metal_requested() || auto_dismiss_dialogs_requested() {
        return;
    }
    let mut state = OVERLAY.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    let duration = Duration::from_millis(u64::from(request.duration_ms.max(1_500)));
    state.toast = Some(ActiveToast {
        request: request.clone(),
        expires_at: Instant::now() + duration,
    });
    state.rebuild();
}

/// Called from the presenter event loop so pad/keyboard can dismiss dialogs and toasts expire.
pub(crate) fn poll_host_ui() {
    poll_dialog_input();
    poll_toast_expiry();
}

/// True when a dialog or toast overlay should be composited this frame.
#[must_use]
pub(crate) fn system_ui_overlay_visible() -> bool {
    let state = OVERLAY.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    state.framebuffer.is_some()
}

/// Takes a dirty overlay upload, if the rasterizer produced new pixels since the last take.
pub(crate) fn take_system_ui_overlay_upload() -> Option<SystemUiFramebuffer> {
    let mut state = OVERLAY.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    if !state.dirty {
        return None;
    }
    state.dirty = false;
    state.framebuffer.clone()
}

fn poll_dialog_input() {
    let mut state = OVERLAY.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    let Some(active) = state.dialog.as_ref() else {
        return;
    };
    if !host_dialog_is_pending() {
        state.dialog = None;
        state.rebuild();
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
        state.dialog = None;
        state.rebuild();
        drop(state);
        complete_host_dialog(sequence, code);
    }
}

fn poll_toast_expiry() {
    let mut state = OVERLAY.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    let Some(active) = state.toast.as_ref() else {
        return;
    };
    if Instant::now() < active.expires_at {
        return;
    }
    state.toast = None;
    state.rebuild();
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
