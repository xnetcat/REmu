// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

//! Shared HID session/connection state used by libpad.
//!
//! SDK oracle: `sdk/target/include/hid/hid_common_error.h` (SCE_HID_ERROR_ALREADY_LOGGED_OUT)
//! and `pad.h` connection fields (`connected` / `connectedCount` on `ScePadData` /
//! `ScePadControllerInformation`).

use std::sync::{Mutex, OnceLock};

/// `SCE_HID_ERROR_ALREADY_LOGGED_OUT` — hid_common_error.h:19 (`0x803B0101`).
pub(crate) const SCE_HID_ERROR_ALREADY_LOGGED_OUT: u32 = 0x803B_0101;

#[derive(Debug)]
struct HidRuntimeState {
    /// Primary user session; pad open/read refuse with ALREADY_LOGGED_OUT when false.
    user_logged_in: bool,
    /// Physical controller presence for the standard local pad.
    device_connected: bool,
    /// Times the device has become connected against the live pad handle.
    /// Increments on each connect edge; unchanged across disconnects (pad.h:120).
    connected_count: u8,
}

impl Default for HidRuntimeState {
    fn default() -> Self {
        Self {
            user_logged_in: true,
            device_connected: true,
            connected_count: 0,
        }
    }
}

static HID_RUNTIME: OnceLock<Mutex<HidRuntimeState>> = OnceLock::new();

fn hid_runtime() -> &'static Mutex<HidRuntimeState> {
    HID_RUNTIME.get_or_init(|| Mutex::new(HidRuntimeState::default()))
}

#[cfg(any(test, target_arch = "x86_64"))]
pub(crate) fn reset_hid_runtime() {
    *hid_runtime()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner) = HidRuntimeState::default();
}

pub(crate) fn user_logged_in() -> bool {
    hid_runtime()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .user_logged_in
}

/// Host/test seam: mark the primary user as logged out or back in.
#[allow(dead_code)] // host/UI callers land later; pad tests exercise the path today
pub(crate) fn set_user_logged_in(logged_in: bool) {
    let mut state = hid_runtime()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if state.user_logged_in == logged_in {
        return;
    }
    state.user_logged_in = logged_in;
    if !logged_in {
        eprintln!("[REMu][HID] user logged out");
    } else {
        eprintln!("[REMu][HID] user logged in");
    }
}

pub(crate) fn device_connected() -> bool {
    hid_runtime()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .device_connected
}

pub(crate) fn connected_count() -> u8 {
    hid_runtime()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .connected_count
}

/// Host/test seam: connect or disconnect the standard controller against an open pad handle.
///
/// A connect edge bumps `connected_count` (pad.h:120). Disconnect leaves the count alone and
/// only clears the live `connected` flag titles read from `ScePadData`.
#[allow(dead_code)] // host/UI callers land later; pad tests exercise the path today
pub(crate) fn set_device_connected(connected: bool) {
    let mut state = hid_runtime()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if state.device_connected == connected {
        return;
    }
    state.device_connected = connected;
    if connected {
        state.connected_count = state.connected_count.saturating_add(1);
        eprintln!(
            "[REMu][HID] device connected (count={})",
            state.connected_count
        );
    } else {
        eprintln!("[REMu][HID] device disconnected");
    }
}

/// Record that a pad handle just opened while a device was already present.
///
/// Open itself is the first connect edge when the physical device is already attached.
pub(crate) fn note_pad_open_with_device() {
    let mut state = hid_runtime()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if state.device_connected && state.connected_count == 0 {
        state.connected_count = 1;
    }
}

/// Reset connection-count bookkeeping when the pad handle is closed.
pub(crate) fn note_pad_handle_closed() {
    let mut state = hid_runtime()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    state.connected_count = 0;
    // Closing the handle does not invent a reconnect; keep physical presence as-is so a
    // subsequent Open can form a fresh connect edge.
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reset() {
        reset_hid_runtime();
    }

    #[test]
    fn connect_edge_increments_count_disconnect_does_not() {
        reset();
        assert!(device_connected());
        assert_eq!(connected_count(), 0);

        note_pad_open_with_device();
        assert_eq!(connected_count(), 1);

        set_device_connected(false);
        assert!(!device_connected());
        assert_eq!(connected_count(), 1);

        set_device_connected(true);
        assert!(device_connected());
        assert_eq!(connected_count(), 2);

        set_device_connected(true);
        assert_eq!(connected_count(), 2);
    }

    #[test]
    fn closing_the_pad_handle_resets_count_for_the_next_open() {
        reset();
        note_pad_open_with_device();
        set_device_connected(false);
        set_device_connected(true);
        assert_eq!(connected_count(), 2);
        note_pad_handle_closed();
        assert_eq!(connected_count(), 0);
        note_pad_open_with_device();
        assert_eq!(connected_count(), 1);
    }
}
