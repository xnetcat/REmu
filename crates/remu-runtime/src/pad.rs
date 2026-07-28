use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

use crate::{
    InitializationState, ProbeDispatchState, SCE_KERNEL_ERROR_INVALID_ARGUMENT, guest_memset,
    low_i32, low_u32, read_guest_u8, sce_kernel_error, write_guest_u8, write_guest_u16,
    write_guest_u32, write_guest_u64,
};

pub(crate) const SCE_PAD_ERROR_INVALID_HANDLE: u32 = 0x8092_0003;
pub(crate) const SCE_PAD_ERROR_ALREADY_OPENED: u32 = 0x8092_0004;
pub(crate) const SCE_PAD_ERROR_NOT_INITIALIZED: u32 = 0x8092_0005;
const SCE_PAD_ERROR_DEVICE_NOT_CONNECTED: u32 = 0x8092_0007;
pub(crate) const SCE_PAD_ERROR_DEVICE_NO_HANDLE: u32 = 0x8092_0008;

pub(crate) fn pad_initialize(state: &mut ProbeDispatchState) -> u64 {
    state.services.pad = InitializationState::Initialized;
    0
}

pub(crate) fn pad_open(arguments: [u64; 6], state: &mut ProbeDispatchState) -> u64 {
    if state.services.pad != InitializationState::Initialized {
        return sce_kernel_error(SCE_PAD_ERROR_NOT_INITIALIZED);
    }
    let user = low_i32(arguments[0]);
    let port_type = low_i32(arguments[1]);
    let index = low_i32(arguments[2]);
    if user != 0x1000_0000 || port_type != 0 || index != 0 {
        return sce_kernel_error(SCE_PAD_ERROR_DEVICE_NOT_CONNECTED);
    }
    if state.services.pad_handle.is_some() {
        return sce_kernel_error(SCE_PAD_ERROR_ALREADY_OPENED);
    }
    let handle = state.services.next_pad_handle.max(1);
    state.services.next_pad_handle = handle.saturating_add(1);
    state.services.pad_handle = Some(handle);
    u64::from(handle.cast_unsigned())
}

pub(crate) fn pad_get_handle(arguments: [u64; 6], state: &ProbeDispatchState) -> u64 {
    if state.services.pad != InitializationState::Initialized {
        return sce_kernel_error(SCE_PAD_ERROR_NOT_INITIALIZED);
    }
    let user = low_i32(arguments[0]);
    let port_type = low_i32(arguments[1]);
    let index = low_i32(arguments[2]);
    if user != 0x1000_0000 || port_type != 0 || index != 0 {
        return sce_kernel_error(SCE_PAD_ERROR_DEVICE_NO_HANDLE);
    }
    state.services.pad_handle.map_or_else(
        || sce_kernel_error(SCE_PAD_ERROR_DEVICE_NO_HANDLE),
        |handle| u64::from(handle.cast_unsigned()),
    )
}

fn validate_pad_handle(handle: u64, state: &ProbeDispatchState) -> Result<(), u64> {
    if state.services.pad != InitializationState::Initialized {
        return Err(sce_kernel_error(SCE_PAD_ERROR_NOT_INITIALIZED));
    }
    if state.services.pad_handle != Some(low_i32(handle)) {
        return Err(sce_kernel_error(SCE_PAD_ERROR_INVALID_HANDLE));
    }
    Ok(())
}

pub(crate) fn pad_close(arguments: [u64; 6], state: &mut ProbeDispatchState) -> u64 {
    if let Err(error) = validate_pad_handle(arguments[0], state) {
        return error;
    }
    state.services.pad_handle = None;
    0
}

pub(crate) fn pad_get_extended_information(arguments: [u64; 6], state: &ProbeDispatchState) -> u64 {
    if let Err(error) = validate_pad_handle(arguments[0], state) {
        return error;
    }
    if arguments[1] == 0 {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    }
    // A standard DualSense has no special device class payload. The ABI structure is 20 bytes:
    // class, reserved word, and a 12-byte class-specific union.
    guest_memset(arguments[1], 0, 20);
    0
}

pub(crate) fn pad_set_motion_sensor_state(arguments: [u64; 6], state: &ProbeDispatchState) -> u64 {
    if let Err(error) = validate_pad_handle(arguments[0], state) {
        return error;
    }
    if !matches!(low_i32(arguments[1]), 0 | 1) {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    }
    0
}

pub(crate) fn pad_set_vibration_mode(arguments: [u64; 6], state: &ProbeDispatchState) -> u64 {
    if let Err(error) = validate_pad_handle(arguments[0], state) {
        return error;
    }
    if low_u32(arguments[1]) > 2 {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    }
    0
}

pub(crate) fn pad_handle_operation(arguments: [u64; 6], state: &ProbeDispatchState) -> u64 {
    validate_pad_handle(arguments[0], state).map_or_else(|error| error, |()| 0)
}

pub(crate) fn pad_boolean_operation(arguments: [u64; 6], state: &ProbeDispatchState) -> u64 {
    if let Err(error) = validate_pad_handle(arguments[0], state) {
        return error;
    }
    if !matches!(low_u32(arguments[1]), 0 | 1) {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    }
    0
}

pub(crate) fn pad_pointer_operation(arguments: [u64; 6], state: &ProbeDispatchState) -> u64 {
    if let Err(error) = validate_pad_handle(arguments[0], state) {
        return error;
    }
    if arguments[1] == 0 {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    }
    0
}

pub(crate) fn pad_get_trigger_effect_state(arguments: [u64; 6], state: &ProbeDispatchState) -> u64 {
    if let Err(error) = validate_pad_handle(arguments[0], state) {
        return error;
    }
    if arguments[1] == 0 {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    }
    guest_memset(arguments[1], 0, 8);
    0
}

pub(crate) fn pad_device_class_parse_data(arguments: [u64; 6], state: &ProbeDispatchState) -> u64 {
    if let Err(error) = validate_pad_handle(arguments[0], state) {
        return error;
    }
    if arguments[1] == 0 || arguments[2] == 0 {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    }
    guest_memset(arguments[2], 0, 24);
    write_guest_u8(arguments[2] + 4, read_guest_u8(arguments[1] + 0x4c));
    0
}

pub(crate) fn pad_get_controller_information(
    arguments: [u64; 6],
    state: &ProbeDispatchState,
) -> u64 {
    if let Err(error) = validate_pad_handle(arguments[0], state) {
        return error;
    }
    let output = arguments[1];
    if output == 0 {
        return u64::from(libc::EINVAL.cast_unsigned());
    }
    guest_memset(output, 0, 0x1c);
    write_guest_u32(output, 44.86_f32.to_bits());
    write_guest_u16(output + 0x04, 1920);
    write_guest_u16(output + 0x06, 943);
    write_guest_u8(output + 0x08, 30);
    write_guest_u8(output + 0x09, 30);
    write_guest_u8(output + 0x0b, 1);
    write_guest_u8(output + 0x0c, 1);
    0
}

pub(crate) fn pad_read_state(arguments: [u64; 6], state: &ProbeDispatchState) -> u64 {
    static LAST_LOGGED_BUTTONS: AtomicU32 = AtomicU32::new(u32::MAX);
    if let Err(error) = validate_pad_handle(arguments[0], state) {
        return error;
    }
    let output = arguments[1];
    if output == 0 {
        return u64::from(libc::EINVAL.cast_unsigned());
    }
    guest_memset(output, 0, 0x78);
    let host = remu_gpu::host_pad_state();
    let buttons = if auto_cross_active() {
        host.buttons | remu_gpu::PAD_BUTTON_CROSS
    } else {
        host.buttons
    };
    let previous_buttons = LAST_LOGGED_BUTTONS.swap(buttons, Ordering::Relaxed);
    if previous_buttons != buttons {
        eprintln!("[REMu][PAD] buttons={buttons:#010x}");
    }
    write_guest_u32(output, buttons);
    write_guest_u8(output + 0x04, host.left_x);
    write_guest_u8(output + 0x05, host.left_y);
    write_guest_u8(output + 0x06, host.right_x);
    write_guest_u8(output + 0x07, host.right_y);
    write_guest_u8(output + 0x08, host.l2);
    write_guest_u8(output + 0x09, host.r2);
    write_guest_u32(output + 0x18, 1.0_f32.to_bits());
    write_guest_u8(output + 0x4c, 1);
    let mut timestamp = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    // SAFETY: timestamp is a valid writable host timespec.
    if unsafe { libc::clock_gettime(libc::CLOCK_MONOTONIC, &raw mut timestamp) } == 0 {
        let seconds = u64::try_from(timestamp.tv_sec).unwrap_or(0);
        let nanoseconds = u64::try_from(timestamp.tv_nsec).unwrap_or(0);
        write_guest_u64(
            output + 0x50,
            seconds
                .saturating_mul(1_000_000)
                .saturating_add(nanoseconds / 1_000),
        );
    }
    write_guest_u8(output + 0x68, 1);
    0
}

pub(crate) fn pad_read(arguments: [u64; 6], state: &ProbeDispatchState) -> u64 {
    if arguments[2] == 0 || arguments[2] > 64 {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    }
    let result = pad_read_state(arguments, state);
    if result == 0 { 1 } else { result }
}

fn auto_cross_active() -> bool {
    static START: OnceLock<Instant> = OnceLock::new();
    static TIMES: OnceLock<Vec<Duration>> = OnceLock::new();
    static HOLD: OnceLock<Duration> = OnceLock::new();
    static PULSE: OnceLock<Mutex<(usize, Option<Instant>)>> = OnceLock::new();
    let times = TIMES.get_or_init(|| {
        std::env::var("REMU_AUTO_CROSS")
            .unwrap_or_default()
            .split(',')
            .filter_map(|value| value.trim().parse::<f64>().ok())
            .filter(|seconds| seconds.is_finite() && *seconds >= 0.0)
            .map(Duration::from_secs_f64)
            .collect()
    });
    if times.is_empty() {
        return false;
    }
    let hold = *HOLD.get_or_init(|| {
        std::env::var("REMU_AUTO_CROSS_HOLD_MS")
            .ok()
            .and_then(|value| value.parse::<u64>().ok())
            .map_or(Duration::from_millis(400), Duration::from_millis)
    });
    let start = *START.get_or_init(Instant::now);
    let now = Instant::now();
    let elapsed = now.saturating_duration_since(start);
    let mut pulse = PULSE
        .get_or_init(|| Mutex::new((0, None)))
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if pulse.1.is_some_and(|active_until| now < active_until) {
        return true;
    }
    pulse.1 = None;
    if times
        .get(pulse.0)
        .is_some_and(|scheduled| elapsed >= *scheduled)
    {
        pulse.0 += 1;
        pulse.1 = Some(now + hold);
        return true;
    }
    false
}
