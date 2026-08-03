// SPDX-License-Identifier: GPL-2.0-or-later

//! Host controller state published to the guest pad service.

use std::sync::atomic::{AtomicU32, Ordering};

pub const PAD_BUTTON_L3: u32 = 0x0002;

pub const PAD_BUTTON_R3: u32 = 0x0004;

pub const PAD_BUTTON_OPTIONS: u32 = 0x0008;

pub const PAD_BUTTON_UP: u32 = 0x0010;

pub const PAD_BUTTON_RIGHT: u32 = 0x0020;

pub const PAD_BUTTON_DOWN: u32 = 0x0040;

pub const PAD_BUTTON_LEFT: u32 = 0x0080;

pub const PAD_BUTTON_L2: u32 = 0x0100;

pub const PAD_BUTTON_R2: u32 = 0x0200;

pub const PAD_BUTTON_L1: u32 = 0x0400;

pub const PAD_BUTTON_R1: u32 = 0x0800;

pub const PAD_BUTTON_TRIANGLE: u32 = 0x1000;

pub const PAD_BUTTON_CIRCLE: u32 = 0x2000;

pub const PAD_BUTTON_CROSS: u32 = 0x4000;

pub const PAD_BUTTON_SQUARE: u32 = 0x8000;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct HostPadState {
    pub buttons: u32,
    pub left_x: u8,
    pub left_y: u8,
    pub right_x: u8,
    pub right_y: u8,
    pub l2: u8,
    pub r2: u8,
}

impl Default for HostPadState {
    fn default() -> Self {
        Self {
            buttons: 0,
            left_x: 128,
            left_y: 128,
            right_x: 128,
            right_y: 128,
            l2: 0,
            r2: 0,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct HostPadVibration {
    /// Low-frequency/large-motor strength in the guest's 0..=255 range.
    pub large_motor: u8,
    /// High-frequency/small-motor strength in the guest's 0..=255 range.
    pub small_motor: u8,
}

static HOST_PAD_BUTTONS: AtomicU32 = AtomicU32::new(0);

static HOST_PAD_PRESSED_EDGES: AtomicU32 = AtomicU32::new(0);

static HOST_PAD_AXES: AtomicU32 = AtomicU32::new(0x8080_8080);

static HOST_PAD_TRIGGERS: AtomicU32 = AtomicU32::new(0);

static HOST_PAD_VIBRATION: AtomicU32 = AtomicU32::new(0);

pub fn set_host_pad_state(state: HostPadState) {
    let axes = u32::from_le_bytes([state.left_x, state.left_y, state.right_x, state.right_y]);
    let triggers = u32::from_le_bytes([state.l2, state.r2, 0, 0]);
    let previous_buttons = HOST_PAD_BUTTONS.load(Ordering::Acquire);
    HOST_PAD_PRESSED_EDGES.fetch_or(state.buttons & !previous_buttons, Ordering::AcqRel);
    HOST_PAD_AXES.store(axes, Ordering::Release);
    HOST_PAD_TRIGGERS.store(triggers, Ordering::Release);
    HOST_PAD_BUTTONS.store(state.buttons, Ordering::Release);
}

/// Current held buttons without consuming rising edges reserved for guest pad reads.
#[must_use]
pub fn host_pad_buttons_held() -> u32 {
    HOST_PAD_BUTTONS.load(Ordering::Acquire)
}

#[must_use]
pub fn host_pad_state() -> HostPadState {
    // The AppKit event loop can publish key-down and key-up between two guest pad polls,
    // especially while a title is compiling shaders or rendering below the host refresh rate.
    // Preserve each rising edge for one read so a valid short press cannot disappear entirely.
    let buttons =
        HOST_PAD_BUTTONS.load(Ordering::Acquire) | HOST_PAD_PRESSED_EDGES.swap(0, Ordering::AcqRel);
    let [left_x, left_y, right_x, right_y] = HOST_PAD_AXES.load(Ordering::Acquire).to_le_bytes();
    let [l2, r2, _, _] = HOST_PAD_TRIGGERS.load(Ordering::Acquire).to_le_bytes();
    HostPadState {
        buttons,
        left_x,
        left_y,
        right_x,
        right_y,
        l2,
        r2,
    }
}

pub fn set_host_pad_vibration(vibration: HostPadVibration) {
    HOST_PAD_VIBRATION.store(
        u32::from_le_bytes([vibration.large_motor, vibration.small_motor, 0, 0]),
        Ordering::Release,
    );
}

#[must_use]
pub fn host_pad_vibration() -> HostPadVibration {
    let [large_motor, small_motor, _, _] = HOST_PAD_VIBRATION.load(Ordering::Acquire).to_le_bytes();
    HostPadVibration {
        large_motor,
        small_motor,
    }
}

#[cfg(test)]
mod tests {
    use super::{HostPadVibration, host_pad_vibration, set_host_pad_vibration};

    #[test]
    fn vibration_snapshot_publishes_both_motors_atomically() {
        let expected = HostPadVibration {
            large_motor: 0x25,
            small_motor: 0xd4,
        };
        set_host_pad_vibration(expected);
        assert_eq!(host_pad_vibration(), expected);
        set_host_pad_vibration(HostPadVibration::default());
    }
}
