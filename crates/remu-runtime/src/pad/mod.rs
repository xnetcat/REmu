use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

use crate::hid::{
    self, SCE_HID_ERROR_ALREADY_LOGGED_OUT, connected_count, device_connected, user_logged_in,
};
use crate::{
    InitializationState, SCE_KERNEL_ERROR_INVALID_ARGUMENT, guest_memset, low_i32, low_u32,
    process_clock_nanos, process_guest_range_is_readable, read_guest_u8, sce_kernel_error,
    write_guest_u8, write_guest_u16, write_guest_u32, write_guest_u64,
};

pub(crate) const SCE_PAD_ERROR_INVALID_HANDLE: u32 = 0x8092_0003;
pub(crate) const SCE_PAD_ERROR_ALREADY_OPENED: u32 = 0x8092_0004;
pub(crate) const SCE_PAD_ERROR_NOT_INITIALIZED: u32 = 0x8092_0005;
const SCE_PAD_ERROR_DEVICE_NOT_CONNECTED: u32 = 0x8092_0007;
pub(crate) const SCE_PAD_ERROR_DEVICE_NO_HANDLE: u32 = 0x8092_0008;

/// `sizeof(ScePadData)` — pad.h:106-124.
pub(crate) const SCE_PAD_DATA_SIZE: usize = 0x78;
/// DualSense stick deadzone reported via `ScePadStickInformation` (pad.h:443-446).
const SCE_PAD_DEFAULT_DEADZONE: u8 = 30;
/// Neutral stick axis value.
const SCE_PAD_STICK_CENTER: u8 = 128;

#[derive(Debug)]
struct PadRuntimeState {
    initialization: InitializationState,
    handle: Option<i32>,
    next_handle: i32,
    /// `scePadSetMotionSensorState` — pad.h:250-260; default enabled.
    motion_sensor_enabled: bool,
    /// `scePadSetAngularVelocityDeadbandState` — pad.h:278-288; default disabled.
    angular_velocity_deadband: bool,
    /// `scePadSetTiltCorrectionState` — pad.h:264-274; default disabled.
    tilt_correction: bool,
    /// `scePadSetAngularVelocityBiasCorrectionState` — pad.h:291-303; default enabled.
    angular_velocity_bias_correction: bool,
    /// Orientation quaternion `<x,y,z,w>` (pad.h:111-113).
    orientation: [f32; 4],
    /// Latest sample timestamp published by `scePadReadState` (microseconds).
    last_sample_timestamp: u64,
    /// Last timestamp consumed by `scePadRead` (pad.h:241-242: return 0 when unchanged).
    last_history_timestamp: u64,
}

impl Default for PadRuntimeState {
    fn default() -> Self {
        Self {
            initialization: InitializationState::default(),
            handle: None,
            next_handle: 0,
            motion_sensor_enabled: true,
            angular_velocity_deadband: false,
            tilt_correction: false,
            angular_velocity_bias_correction: true,
            orientation: [0.0, 0.0, 0.0, 1.0],
            last_sample_timestamp: 0,
            last_history_timestamp: 0,
        }
    }
}

static PAD_RUNTIME: OnceLock<Mutex<PadRuntimeState>> = OnceLock::new();

fn pad_runtime() -> &'static Mutex<PadRuntimeState> {
    PAD_RUNTIME.get_or_init(|| Mutex::new(PadRuntimeState::default()))
}

#[cfg(any(test, target_arch = "x86_64"))]
pub(crate) fn reset_pad_runtime() {
    *pad_runtime()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner) = PadRuntimeState::default();
}

pub(crate) fn dispatch_pad_import(nid: &str, arguments: [u64; 6]) -> Option<u64> {
    // NIDs from abi-inventory/sdk10-nid-map.tsv (libScePad).
    Some(match nid {
        "xk0AcarP3V4" => pad_open(arguments),                         // scePadOpen
        "u1GRHp+oWoY" => pad_get_handle(arguments),                   // scePadGetHandle
        "clVvL4ZDntw" => pad_set_motion_sensor_state(arguments),      // scePadSetMotionSensorState
        "gjP9-KQzoUk" => pad_get_controller_information(arguments),   // scePadGetControllerInformation
        "6ncge5+l5Qs" => pad_close(arguments),                        // scePadClose
        "AcslpN1jHR8" => pad_get_extended_information(arguments),     // scePadDeviceClassGetExtendedInformation
        "YndgXqQVV7c" => pad_read_state(arguments),                   // scePadReadState
        "q1cHNfGycLI" => pad_read(arguments),                         // scePadRead
        "W2G-yoyMF5U" => pad_set_vibration_mode(arguments),           // scePadSetVibrationMode
        "rIZnR6eSpvk" => pad_reset_orientation(arguments),            // scePadResetOrientation
        "DscD1i9HX1w" => pad_handle_operation(arguments),             // scePadResetLightBar
        "r44mAxdSG+U" => pad_set_angular_velocity_deadband(arguments), // scePadSetAngularVelocityDeadbandState
        "vDLMoJLde8I" => pad_set_tilt_correction(arguments),          // scePadSetTiltCorrectionState
        "KLmYx9ij2h0" => pad_set_angular_velocity_bias(arguments),    // scePadSetAngularVelocityBiasCorrectionState
        "yFVnOdGxvZY" | "2JgFB2n9oUM" | "RR4novUEENY" => pad_pointer_operation(arguments),
        "znaWI0gpuo8" => pad_get_trigger_effect_state(arguments),     // scePadGetTriggerEffectState
        "IHPqcbc0zCA" => pad_device_class_parse_data(arguments),       // scePadDeviceClassParseData
        "hv1luiJrqQM" => pad_initialize(),                            // scePadInit
        _ => return None,
    })
}

fn require_user_logged_in() -> Result<(), u64> {
    if user_logged_in() {
        Ok(())
    } else {
        eprintln!("[REMu][PAD] refused: user already logged out (SCE_HID_ERROR_ALREADY_LOGGED_OUT)");
        Err(sce_kernel_error(SCE_HID_ERROR_ALREADY_LOGGED_OUT))
    }
}

fn validate_output_pointer(address: u64, length: usize, what: &str) -> Result<(), u64> {
    if address == 0 || !process_guest_range_is_readable(address, length) {
        eprintln!("[REMu][PAD] invalid {what} pointer {address:#x} len={length}");
        return Err(sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT));
    }
    Ok(())
}

/// Snap an analogue stick axis to centre when inside the reported deadzone (pad.h:444-445).
pub(crate) fn apply_stick_deadzone(value: u8, deadzone: u8) -> u8 {
    let delta = i16::from(value) - i16::from(SCE_PAD_STICK_CENTER);
    if delta.unsigned_abs() <= u16::from(deadzone) {
        SCE_PAD_STICK_CENTER
    } else {
        value
    }
}

fn pad_system_timestamp_micros() -> u64 {
    process_clock_nanos() / 1_000
}

pub(crate) fn pad_initialize() -> u64 {
    pad_runtime()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .initialization = InitializationState::Initialized;
    0
}

pub(crate) fn pad_open(arguments: [u64; 6]) -> u64 {
    if let Err(error) = require_user_logged_in() {
        return error;
    }
    let mut state = pad_runtime()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if state.initialization != InitializationState::Initialized {
        return sce_kernel_error(SCE_PAD_ERROR_NOT_INITIALIZED);
    }
    let user = low_i32(arguments[0]);
    let port_type = low_i32(arguments[1]);
    let index = low_i32(arguments[2]);
    if user != 0x1000_0000 || port_type != 0 || index != 0 {
        eprintln!(
            "[REMu][PAD] open refused: user={user:#x} type={port_type} index={index} (no device)"
        );
        return sce_kernel_error(SCE_PAD_ERROR_DEVICE_NOT_CONNECTED);
    }
    if state.handle.is_some() {
        return sce_kernel_error(SCE_PAD_ERROR_ALREADY_OPENED);
    }
    let handle = state.next_handle.max(1);
    state.next_handle = handle.saturating_add(1);
    state.handle = Some(handle);
    state.last_sample_timestamp = 0;
    state.last_history_timestamp = 0;
    state.orientation = [0.0, 0.0, 0.0, 1.0];
    drop(state);
    hid::note_pad_open_with_device();
    u64::from(handle.cast_unsigned())
}

pub(crate) fn pad_get_handle(arguments: [u64; 6]) -> u64 {
    if let Err(error) = require_user_logged_in() {
        return error;
    }
    let state = pad_runtime()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if state.initialization != InitializationState::Initialized {
        return sce_kernel_error(SCE_PAD_ERROR_NOT_INITIALIZED);
    }
    let user = low_i32(arguments[0]);
    let port_type = low_i32(arguments[1]);
    let index = low_i32(arguments[2]);
    if user != 0x1000_0000 || port_type != 0 || index != 0 {
        return sce_kernel_error(SCE_PAD_ERROR_DEVICE_NO_HANDLE);
    }
    state.handle.map_or_else(
        || sce_kernel_error(SCE_PAD_ERROR_DEVICE_NO_HANDLE),
        |handle| u64::from(handle.cast_unsigned()),
    )
}

fn validate_pad_handle_state(handle: u64, state: &PadRuntimeState) -> Result<(), u64> {
    if state.initialization != InitializationState::Initialized {
        return Err(sce_kernel_error(SCE_PAD_ERROR_NOT_INITIALIZED));
    }
    if state.handle != Some(low_i32(handle)) {
        return Err(sce_kernel_error(SCE_PAD_ERROR_INVALID_HANDLE));
    }
    Ok(())
}

fn validate_pad_handle(handle: u64) -> Result<(), u64> {
    require_user_logged_in()?;
    let state = pad_runtime()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    validate_pad_handle_state(handle, &state)
}

pub(crate) fn pad_close(arguments: [u64; 6]) -> u64 {
    if let Err(error) = require_user_logged_in() {
        return error;
    }
    let mut state = pad_runtime()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if let Err(error) = validate_pad_handle_state(arguments[0], &state) {
        return error;
    }
    state.handle = None;
    state.last_sample_timestamp = 0;
    state.last_history_timestamp = 0;
    drop(state);
    hid::note_pad_handle_closed();
    0
}

pub(crate) fn pad_get_extended_information(arguments: [u64; 6]) -> u64 {
    if let Err(error) = validate_pad_handle(arguments[0]) {
        return error;
    }
    if let Err(error) = validate_output_pointer(arguments[1], 20, "extended information") {
        return error;
    }
    // A standard DualSense has no special device class payload. The ABI structure is 20 bytes:
    // class, reserved word, and a 12-byte class-specific union.
    guest_memset(arguments[1], 0, 20);
    0
}

pub(crate) fn pad_set_motion_sensor_state(arguments: [u64; 6]) -> u64 {
    if let Err(error) = validate_pad_handle(arguments[0]) {
        return error;
    }
    let enable = low_i32(arguments[1]);
    if !matches!(enable, 0 | 1) {
        eprintln!("[REMu][PAD] SetMotionSensorState: invalid enable={enable}");
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    }
    let mut state = pad_runtime()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    state.motion_sensor_enabled = enable != 0;
    0
}

pub(crate) fn pad_reset_orientation(arguments: [u64; 6]) -> u64 {
    if let Err(error) = validate_pad_handle(arguments[0]) {
        return error;
    }
    let mut state = pad_runtime()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    // pad.h:307-315 — reset accumulated orientation to identity.
    state.orientation = [0.0, 0.0, 0.0, 1.0];
    0
}

fn pad_set_angular_velocity_deadband(arguments: [u64; 6]) -> u64 {
    if let Err(error) = validate_pad_handle(arguments[0]) {
        return error;
    }
    let enable = low_u32(arguments[1]);
    if !matches!(enable, 0 | 1) {
        eprintln!("[REMu][PAD] SetAngularVelocityDeadbandState: invalid enable={enable}");
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    }
    pad_runtime()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .angular_velocity_deadband = enable != 0;
    0
}

fn pad_set_tilt_correction(arguments: [u64; 6]) -> u64 {
    if let Err(error) = validate_pad_handle(arguments[0]) {
        return error;
    }
    let enable = low_u32(arguments[1]);
    if !matches!(enable, 0 | 1) {
        eprintln!("[REMu][PAD] SetTiltCorrectionState: invalid enable={enable}");
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    }
    pad_runtime()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .tilt_correction = enable != 0;
    0
}

fn pad_set_angular_velocity_bias(arguments: [u64; 6]) -> u64 {
    if let Err(error) = validate_pad_handle(arguments[0]) {
        return error;
    }
    let enable = low_u32(arguments[1]);
    if !matches!(enable, 0 | 1) {
        eprintln!("[REMu][PAD] SetAngularVelocityBiasCorrectionState: invalid enable={enable}");
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    }
    pad_runtime()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .angular_velocity_bias_correction = enable != 0;
    0
}

pub(crate) fn pad_set_vibration_mode(arguments: [u64; 6]) -> u64 {
    if let Err(error) = validate_pad_handle(arguments[0]) {
        return error;
    }
    // ScePadVibrationMode values 1..=5 (pad.h:338-344).
    let mode = low_u32(arguments[1]);
    if !(1..=5).contains(&mode) {
        eprintln!("[REMu][PAD] SetVibrationMode: invalid mode={mode}");
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    }
    0
}

pub(crate) fn pad_handle_operation(arguments: [u64; 6]) -> u64 {
    validate_pad_handle(arguments[0]).map_or_else(|error| error, |()| 0)
}

pub(crate) fn pad_pointer_operation(arguments: [u64; 6]) -> u64 {
    if let Err(error) = validate_pad_handle(arguments[0]) {
        return error;
    }
    if arguments[1] == 0 {
        eprintln!("[REMu][PAD] pointer operation: null parameter");
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    }
    0
}

pub(crate) fn pad_get_trigger_effect_state(arguments: [u64; 6]) -> u64 {
    if let Err(error) = validate_pad_handle(arguments[0]) {
        return error;
    }
    if let Err(error) = validate_output_pointer(arguments[1], 8, "trigger effect state") {
        return error;
    }
    guest_memset(arguments[1], 0, 8);
    0
}

pub(crate) fn pad_device_class_parse_data(arguments: [u64; 6]) -> u64 {
    if let Err(error) = validate_pad_handle(arguments[0]) {
        return error;
    }
    if arguments[1] == 0 || arguments[2] == 0 {
        eprintln!("[REMu][PAD] DeviceClassParseData: null data pointer");
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    }
    if !process_guest_range_is_readable(arguments[1], SCE_PAD_DATA_SIZE)
        || !process_guest_range_is_readable(arguments[2], 24)
    {
        eprintln!("[REMu][PAD] DeviceClassParseData: unreadable guest range");
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    }
    guest_memset(arguments[2], 0, 24);
    write_guest_u8(arguments[2] + 4, read_guest_u8(arguments[1] + 0x4c));
    0
}

pub(crate) fn pad_get_controller_information(arguments: [u64; 6]) -> u64 {
    if let Err(error) = validate_pad_handle(arguments[0]) {
        return error;
    }
    let output = arguments[1];
    if let Err(error) = validate_output_pointer(output, 0x1c, "controller information") {
        return error;
    }
    // Layout: ScePadControllerInformation (pad.h:454-462).
    guest_memset(output, 0, 0x1c);
    write_guest_u32(output, 44.86_f32.to_bits()); // touchPadInfo.pixelDensity
    write_guest_u16(output + 0x04, 1920); // resolution.x
    write_guest_u16(output + 0x06, 943); // resolution.y
    write_guest_u8(output + 0x08, SCE_PAD_DEFAULT_DEADZONE); // stickInfo.deadZoneLeft
    write_guest_u8(output + 0x09, SCE_PAD_DEFAULT_DEADZONE); // stickInfo.deadZoneRight
    write_guest_u8(output + 0x0a, 0); // SCE_PAD_CONNECTION_TYPE_LOCAL
    write_guest_u8(output + 0x0b, connected_count());
    write_guest_u8(output + 0x0c, u8::from(device_connected()));
    // deviceClass STANDARD = 0 already via memset
    0
}

pub(crate) fn pad_read_state(arguments: [u64; 6]) -> u64 {
    static LAST_LOGGED_BUTTONS: AtomicU32 = AtomicU32::new(u32::MAX);
    if let Err(error) = validate_pad_handle(arguments[0]) {
        return error;
    }
    let output = arguments[1];
    if let Err(error) = validate_output_pointer(output, SCE_PAD_DATA_SIZE, "pad data") {
        return error;
    }

    let connected = device_connected();
    let count = connected_count();
    let mut state = pad_runtime()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);

    guest_memset(output, 0, SCE_PAD_DATA_SIZE as u64);

    let (buttons, left_x, left_y, right_x, right_y, l2, r2) = if connected {
        let host = remu_gpu::host_pad_state();
        let buttons = host.buttons | auto_cross_buttons() | auto_options_buttons();
        (
            buttons,
            apply_stick_deadzone(host.left_x, SCE_PAD_DEFAULT_DEADZONE),
            apply_stick_deadzone(host.left_y, SCE_PAD_DEFAULT_DEADZONE),
            apply_stick_deadzone(host.right_x, SCE_PAD_DEFAULT_DEADZONE),
            apply_stick_deadzone(host.right_y, SCE_PAD_DEFAULT_DEADZONE),
            host.l2,
            host.r2,
        )
    } else {
        (
            0,
            SCE_PAD_STICK_CENTER,
            SCE_PAD_STICK_CENTER,
            SCE_PAD_STICK_CENTER,
            SCE_PAD_STICK_CENTER,
            0,
            0,
        )
    };

    let previous_buttons = LAST_LOGGED_BUTTONS.swap(buttons, Ordering::Relaxed);
    if previous_buttons != buttons {
        eprintln!("[REMu][PAD] buttons={buttons:#010x} connected={connected}");
    }

    write_guest_u32(output, buttons);
    write_guest_u8(output + 0x04, left_x);
    write_guest_u8(output + 0x05, left_y);
    write_guest_u8(output + 0x06, right_x);
    write_guest_u8(output + 0x07, right_y);
    write_guest_u8(output + 0x08, l2);
    write_guest_u8(output + 0x09, r2);

    // Orientation quaternion <x,y,z,w> at 0x0c (pad.h:111-113).
    let orientation = state.orientation;
    write_guest_u32(output + 0x0c, orientation[0].to_bits());
    write_guest_u32(output + 0x10, orientation[1].to_bits());
    write_guest_u32(output + 0x14, orientation[2].to_bits());
    write_guest_u32(output + 0x18, orientation[3].to_bits());

    if state.motion_sensor_enabled && connected {
        // Resting controller: 1 G on +Y (pad.h:114). No host IMU yet — stationary sample.
        write_guest_u32(output + 0x1c, 0.0_f32.to_bits());
        write_guest_u32(output + 0x20, 1.0_f32.to_bits());
        write_guest_u32(output + 0x24, 0.0_f32.to_bits());
        // Angular velocity stays zero while stationary; deadband would also zero small values.
        let _ = state.angular_velocity_deadband;
        let _ = state.tilt_correction;
        let _ = state.angular_velocity_bias_correction;
    }

    write_guest_u8(output + 0x4c, u8::from(connected));
    let mut timestamp = pad_system_timestamp_micros();
    if timestamp <= state.last_sample_timestamp {
        timestamp = state.last_sample_timestamp.saturating_add(1);
    }
    state.last_sample_timestamp = timestamp;
    write_guest_u64(output + 0x50, timestamp);
    write_guest_u8(output + 0x68, count);
    0
}

pub(crate) fn pad_read(arguments: [u64; 6]) -> u64 {
    if let Err(error) = validate_pad_handle(arguments[0]) {
        return error;
    }
    let num = low_i32(arguments[2]);
    if !(1..=64).contains(&num) {
        eprintln!("[REMu][PAD] Read: invalid history count {num}");
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    }
    let output = arguments[1];
    let bytes = SCE_PAD_DATA_SIZE.saturating_mul(num as usize);
    if let Err(error) = validate_output_pointer(output, bytes, "pad history") {
        return error;
    }

    // pad.h:241-242 — return 0 when controller data is not updated since the previous read.
    {
        let state = pad_runtime()
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if state.last_sample_timestamp != 0
            && state.last_sample_timestamp == state.last_history_timestamp
        {
            return 0;
        }
    }

    let publish = pad_read_state([arguments[0], output, 0, 0, 0, 0]);
    if publish != 0 {
        return publish;
    }
    let mut state = pad_runtime()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    state.last_history_timestamp = state.last_sample_timestamp;
    1
}

#[derive(Default)]
struct AutoCrossPulseState {
    next_scheduled_pulse: usize,
    active_until: Option<Instant>,
    active_buttons: u32,
    next_repeated_pulse: Option<Instant>,
    next_button_is_square: bool,
}

impl AutoCrossPulseState {
    fn begin_pulse(&mut self, now: Instant, hold: Duration, allow_square: bool) -> u32 {
        self.active_buttons = if allow_square && self.next_button_is_square {
            remu_gpu::PAD_BUTTON_SQUARE
        } else {
            remu_gpu::PAD_BUTTON_CROSS
        };
        if allow_square {
            self.next_button_is_square = !self.next_button_is_square;
        }
        self.active_until = Some(now + hold);
        self.active_buttons
    }

    fn buttons_at(
        &mut self,
        now: Instant,
        start: Instant,
        times: &[Duration],
        hold: Duration,
        repeat: Option<Duration>,
    ) -> u32 {
        if self
            .active_until
            .is_some_and(|active_until| now < active_until)
        {
            return self.active_buttons;
        }
        self.active_until = None;
        self.active_buttons = 0;

        let elapsed = now.saturating_duration_since(start);
        if times
            .get(self.next_scheduled_pulse)
            .is_some_and(|scheduled| elapsed >= *scheduled)
        {
            self.next_scheduled_pulse += 1;
            if self.next_scheduled_pulse == times.len() {
                self.next_repeated_pulse = repeat.map(|period| now + period);
                // First post-schedule pulse should be Square so Confirm and Restore Default
                // keep alternating on separate edges.
                self.next_button_is_square = true;
            }
            // Explicit schedule pulses stay on Cross. Square is reserved for the post-schedule
            // repeat phase so a Confirm path cannot be undone by Restore Default mid-menu.
            return self.begin_pulse(now, hold, false);
        }

        if self.next_scheduled_pulse == times.len()
            && self
                .next_repeated_pulse
                .is_some_and(|scheduled| now >= scheduled)
        {
            self.next_repeated_pulse = repeat.map(|period| now + period);
            return self.begin_pulse(now, hold, true);
        }
        0
    }
}

fn auto_cross_buttons() -> u32 {
    static START: OnceLock<Instant> = OnceLock::new();
    static TIMES: OnceLock<Vec<Duration>> = OnceLock::new();
    static HOLD: OnceLock<Duration> = OnceLock::new();
    static REPEAT: OnceLock<Option<Duration>> = OnceLock::new();
    static PULSE: OnceLock<Mutex<AutoCrossPulseState>> = OnceLock::new();
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
        return 0;
    }
    let hold = *HOLD.get_or_init(|| {
        std::env::var("REMU_AUTO_CROSS_HOLD_MS")
            .ok()
            .and_then(|value| value.parse::<u64>().ok())
            .map_or(Duration::from_millis(400), Duration::from_millis)
    });
    let repeat = *REPEAT.get_or_init(|| {
        let requested = std::env::var("REMU_AUTO_CROSS_REPEAT_MS")
            .ok()
            .and_then(|value| value.parse::<u64>().ok())
            .map_or(Duration::from_millis(1_000), Duration::from_millis);
        if requested.is_zero() {
            None
        } else {
            // Always leave a release edge between pulses, even if a caller asks for a repeat
            // period shorter than the configured hold.
            Some(requested.max(hold + Duration::from_millis(50)))
        }
    });
    let start = *START.get_or_init(Instant::now);
    let now = Instant::now();
    let mut pulse = PULSE
        .get_or_init(|| Mutex::new(AutoCrossPulseState::default()))
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    pulse.buttons_at(now, start, times, hold, repeat)
}

#[derive(Default)]
struct AutoScheduledButtonPulseState {
    next_scheduled_pulse: usize,
    active_until: Option<Instant>,
}

impl AutoScheduledButtonPulseState {
    fn buttons_at(
        &mut self,
        now: Instant,
        start: Instant,
        times: &[Duration],
        hold: Duration,
        button: u32,
    ) -> u32 {
        if self
            .active_until
            .is_some_and(|active_until| now < active_until)
        {
            return button;
        }
        self.active_until = None;

        let elapsed = now.saturating_duration_since(start);
        if times
            .get(self.next_scheduled_pulse)
            .is_some_and(|scheduled| elapsed >= *scheduled)
        {
            self.next_scheduled_pulse += 1;
            self.active_until = Some(now + hold);
            return button;
        }
        0
    }
}

fn auto_options_buttons() -> u32 {
    static START: OnceLock<Instant> = OnceLock::new();
    static TIMES: OnceLock<Vec<Duration>> = OnceLock::new();
    static HOLD: OnceLock<Duration> = OnceLock::new();
    static PULSE: OnceLock<Mutex<AutoScheduledButtonPulseState>> = OnceLock::new();
    let times = TIMES.get_or_init(|| {
        std::env::var("REMU_AUTO_OPTIONS")
            .unwrap_or_default()
            .split(',')
            .filter_map(|value| value.trim().parse::<f64>().ok())
            .filter(|seconds| seconds.is_finite() && *seconds >= 0.0)
            .map(Duration::from_secs_f64)
            .collect()
    });
    if times.is_empty() {
        return 0;
    }
    let hold = *HOLD.get_or_init(|| {
        std::env::var("REMU_AUTO_OPTIONS_HOLD_MS")
            .ok()
            .and_then(|value| value.parse::<u64>().ok())
            .map_or(Duration::from_millis(400), Duration::from_millis)
    });
    let start = *START.get_or_init(Instant::now);
    let now = Instant::now();
    let mut pulse = PULSE
        .get_or_init(|| Mutex::new(AutoScheduledButtonPulseState::default()))
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    pulse.buttons_at(now, start, times, hold, remu_gpu::PAD_BUTTON_OPTIONS)
}

#[cfg(test)]
mod tests {
    use super::{
        AutoCrossPulseState, AutoScheduledButtonPulseState, SCE_PAD_DATA_SIZE,
        SCE_PAD_ERROR_INVALID_HANDLE, apply_stick_deadzone, pad_close,
        pad_get_controller_information, pad_initialize, pad_open, pad_read, pad_read_state,
        pad_reset_orientation, pad_set_motion_sensor_state, reset_pad_runtime,
    };
    use crate::hid::{
        SCE_HID_ERROR_ALREADY_LOGGED_OUT, connected_count, reset_hid_runtime, set_device_connected,
        set_user_logged_in,
    };
    use crate::sce_kernel_error;
    use remu_gpu::{HostPadState, set_host_pad_state};
    use std::time::{Duration, Instant};

    const USER: u64 = 0x1000_0000;

    fn reset_input() {
        reset_pad_runtime();
        reset_hid_runtime();
        set_host_pad_state(HostPadState::default());
    }

    fn open_standard_pad() -> i32 {
        assert_eq!(pad_initialize(), 0);
        let handle = pad_open([USER, 0, 0, 0, 0, 0]);
        assert!(handle < 0x8000_0000, "open failed: {handle:#x}");
        handle as i32
    }

    fn args(handle: i32, output: u64) -> [u64; 6] {
        [u64::from(handle.cast_unsigned()), output, 0, 0, 0, 0]
    }

    fn read_u8(data: &[u8], offset: usize) -> u8 {
        data[offset]
    }

    fn read_u32(data: &[u8], offset: usize) -> u32 {
        u32::from_le_bytes(data[offset..offset + 4].try_into().unwrap())
    }

    fn read_u64(data: &[u8], offset: usize) -> u64 {
        u64::from_le_bytes(data[offset..offset + 8].try_into().unwrap())
    }

    #[test]
    fn disconnect_keeps_handle_alive_and_reports_connected_false() {
        reset_input();
        let handle = open_standard_pad();
        assert_eq!(connected_count(), 1);

        let mut data = [0_u8; SCE_PAD_DATA_SIZE];
        let output = data.as_mut_ptr() as u64;
        assert_eq!(pad_read_state(args(handle, output)), 0);
        assert_eq!(read_u8(&data, 0x4c), 1);
        assert_eq!(read_u8(&data, 0x68), 1);

        set_device_connected(false);
        assert_eq!(pad_read_state(args(handle, output)), 0);
        assert_eq!(read_u8(&data, 0x4c), 0);
        assert_eq!(read_u8(&data, 0x68), 1);

        let mut info = [0_u8; 0x1c];
        let info_out = info.as_mut_ptr() as u64;
        assert_eq!(pad_get_controller_information(args(handle, info_out)), 0);
        assert_eq!(read_u8(&info, 0x0b), 1);
        assert_eq!(read_u8(&info, 0x0c), 0);

        set_device_connected(true);
        assert_eq!(connected_count(), 2);
        assert_eq!(pad_read_state(args(handle, output)), 0);
        assert_eq!(read_u8(&data, 0x4c), 1);
        assert_eq!(read_u8(&data, 0x68), 2);

        assert_eq!(
            pad_close([u64::from(handle.cast_unsigned()), 0, 0, 0, 0, 0]),
            0
        );
        assert_eq!(
            pad_read_state(args(handle, output)),
            sce_kernel_error(SCE_PAD_ERROR_INVALID_HANDLE)
        );
    }

    #[test]
    fn logged_out_user_is_a_loud_hid_error_not_a_silent_pad_success() {
        reset_input();
        let handle = open_standard_pad();
        set_user_logged_in(false);
        let mut data = [0_u8; SCE_PAD_DATA_SIZE];
        assert_eq!(
            pad_read_state(args(handle, data.as_mut_ptr() as u64)),
            sce_kernel_error(SCE_HID_ERROR_ALREADY_LOGGED_OUT)
        );
    }

    #[test]
    fn read_state_timestamps_are_monotonic_system_micros() {
        reset_input();
        let handle = open_standard_pad();
        let mut data = [0_u8; SCE_PAD_DATA_SIZE];
        let output = data.as_mut_ptr() as u64;
        assert_eq!(pad_read_state(args(handle, output)), 0);
        let first = read_u64(&data, 0x50);
        std::thread::sleep(Duration::from_millis(2));
        assert_eq!(pad_read_state(args(handle, output)), 0);
        let second = read_u64(&data, 0x50);
        assert!(
            second > first,
            "timestamp did not advance: {first} -> {second}"
        );
    }

    #[test]
    fn pad_read_returns_zero_when_no_new_sample_is_available() {
        reset_input();
        let handle = open_standard_pad();
        let mut data = [0_u8; SCE_PAD_DATA_SIZE];
        let output = data.as_mut_ptr() as u64;
        assert_eq!(
            pad_read([u64::from(handle.cast_unsigned()), output, 1, 0, 0, 0]),
            1
        );
        assert_eq!(
            pad_read([u64::from(handle.cast_unsigned()), output, 1, 0, 0, 0]),
            0
        );
        std::thread::sleep(Duration::from_millis(2));
        assert_eq!(pad_read_state(args(handle, output)), 0);
        assert_eq!(
            pad_read([u64::from(handle.cast_unsigned()), output, 1, 0, 0, 0]),
            1
        );
    }

    #[test]
    fn stick_deadzone_from_controller_information_is_applied_on_read() {
        reset_input();
        let handle = open_standard_pad();
        let mut info = [0_u8; 0x1c];
        let info_out = info.as_mut_ptr() as u64;
        assert_eq!(pad_get_controller_information(args(handle, info_out)), 0);
        let deadzone = read_u8(&info, 0x08);
        assert_eq!(deadzone, 30);

        set_host_pad_state(HostPadState {
            left_x: 128 + 15,
            left_y: 128 - 10,
            right_x: 128 + 40,
            right_y: 200,
            ..HostPadState::default()
        });
        let mut data = [0_u8; SCE_PAD_DATA_SIZE];
        let output = data.as_mut_ptr() as u64;
        assert_eq!(pad_read_state(args(handle, output)), 0);
        assert_eq!(read_u8(&data, 0x04), 128);
        assert_eq!(read_u8(&data, 0x05), 128);
        assert_eq!(read_u8(&data, 0x06), 128 + 40);
        assert_eq!(read_u8(&data, 0x07), 200);
    }

    #[test]
    fn apply_stick_deadzone_snaps_inside_radius_to_center() {
        assert_eq!(apply_stick_deadzone(128, 30), 128);
        assert_eq!(apply_stick_deadzone(128 + 30, 30), 128);
        assert_eq!(apply_stick_deadzone(128 + 31, 30), 128 + 31);
        assert_eq!(apply_stick_deadzone(128 - 31, 30), 128 - 31);
    }

    #[test]
    fn motion_sensor_disable_zeros_imu_and_reset_orientation_restores_identity() {
        reset_input();
        let handle = open_standard_pad();
        let mut data = [0_u8; SCE_PAD_DATA_SIZE];
        let output = data.as_mut_ptr() as u64;

        assert_eq!(
            pad_set_motion_sensor_state([u64::from(handle.cast_unsigned()), 1, 0, 0, 0, 0]),
            0
        );
        assert_eq!(pad_read_state(args(handle, output)), 0);
        // Resting DualSense gravity on +Y when motion is enabled (pad.h:114).
        assert_eq!(read_u32(&data, 0x1c), 0.0_f32.to_bits());
        assert_eq!(read_u32(&data, 0x20), 1.0_f32.to_bits());
        assert_eq!(read_u32(&data, 0x24), 0.0_f32.to_bits());
        assert_eq!(read_u32(&data, 0x18), 1.0_f32.to_bits());

        assert_eq!(
            pad_set_motion_sensor_state([u64::from(handle.cast_unsigned()), 0, 0, 0, 0, 0]),
            0
        );
        assert_eq!(pad_read_state(args(handle, output)), 0);
        assert_eq!(read_u32(&data, 0x1c), 0);
        assert_eq!(read_u32(&data, 0x20), 0);
        assert_eq!(read_u32(&data, 0x24), 0);
        assert_eq!(read_u32(&data, 0x28), 0);
        assert_eq!(read_u32(&data, 0x18), 1.0_f32.to_bits());

        assert_eq!(
            pad_reset_orientation([u64::from(handle.cast_unsigned()), 0, 0, 0, 0, 0]),
            0
        );
        assert_eq!(pad_read_state(args(handle, output)), 0);
        assert_eq!(read_u32(&data, 0x0c), 0);
        assert_eq!(read_u32(&data, 0x10), 0);
        assert_eq!(read_u32(&data, 0x14), 0);
        assert_eq!(read_u32(&data, 0x18), 1.0_f32.to_bits());
    }

    #[test]
    fn null_pad_data_pointer_is_a_loud_invalid_argument() {
        reset_input();
        let handle = open_standard_pad();
        assert_eq!(
            pad_read_state(args(handle, 0)),
            sce_kernel_error(crate::SCE_KERNEL_ERROR_INVALID_ARGUMENT)
        );
    }

    #[test]
    fn auto_cross_alternates_cross_and_square_with_release_edges() {
        let start = Instant::now();
        let hold = Duration::from_millis(100);
        let repeat = Duration::from_millis(500);
        let times = [Duration::from_secs(1)];
        let mut state = AutoCrossPulseState::default();

        assert_eq!(
            state.buttons_at(
                start + Duration::from_millis(999),
                start,
                &times,
                hold,
                Some(repeat),
            ),
            0
        );
        assert_eq!(
            state.buttons_at(
                start + Duration::from_secs(1),
                start,
                &times,
                hold,
                Some(repeat),
            ),
            remu_gpu::PAD_BUTTON_CROSS
        );
        assert_eq!(
            state.buttons_at(
                start + Duration::from_millis(1_099),
                start,
                &times,
                hold,
                Some(repeat),
            ),
            remu_gpu::PAD_BUTTON_CROSS
        );
        assert_eq!(
            state.buttons_at(
                start + Duration::from_millis(1_100),
                start,
                &times,
                hold,
                Some(repeat),
            ),
            0
        );
        assert_eq!(
            state.buttons_at(
                start + Duration::from_millis(1_500),
                start,
                &times,
                hold,
                Some(repeat),
            ),
            remu_gpu::PAD_BUTTON_SQUARE
        );
        assert_eq!(
            state.buttons_at(
                start + Duration::from_millis(1_600),
                start,
                &times,
                hold,
                Some(repeat),
            ),
            0
        );
    }

    #[test]
    fn zero_repeat_keeps_the_explicit_schedule_one_shot() {
        let start = Instant::now();
        let times = [Duration::ZERO];
        let mut state = AutoCrossPulseState::default();

        assert_ne!(
            state.buttons_at(start, start, &times, Duration::from_millis(100), None,),
            0
        );
        assert_eq!(
            state.buttons_at(
                start + Duration::from_secs(1),
                start,
                &times,
                Duration::from_millis(100),
                None,
            ),
            0
        );
    }

    #[test]
    fn scheduled_pulses_stay_on_cross_so_confirm_is_not_undone() {
        let start = Instant::now();
        let hold = Duration::from_millis(50);
        let times = [Duration::from_millis(10), Duration::from_millis(200)];
        let mut state = AutoCrossPulseState::default();

        assert_eq!(
            state.buttons_at(start + Duration::from_millis(10), start, &times, hold, None),
            remu_gpu::PAD_BUTTON_CROSS
        );
        assert_eq!(
            state.buttons_at(
                start + Duration::from_millis(200),
                start,
                &times,
                hold,
                None
            ),
            remu_gpu::PAD_BUTTON_CROSS
        );
    }

    #[test]
    fn scheduled_button_pulses_have_release_edges() {
        let start = Instant::now();
        let hold = Duration::from_millis(100);
        let times = [Duration::from_secs(1), Duration::from_secs(2)];
        let mut state = AutoScheduledButtonPulseState::default();

        assert_eq!(
            state.buttons_at(
                start + Duration::from_millis(999),
                start,
                &times,
                hold,
                remu_gpu::PAD_BUTTON_OPTIONS,
            ),
            0
        );
        assert_eq!(
            state.buttons_at(
                start + Duration::from_secs(1),
                start,
                &times,
                hold,
                remu_gpu::PAD_BUTTON_OPTIONS,
            ),
            remu_gpu::PAD_BUTTON_OPTIONS
        );
        assert_eq!(
            state.buttons_at(
                start + Duration::from_millis(1_100),
                start,
                &times,
                hold,
                remu_gpu::PAD_BUTTON_OPTIONS,
            ),
            0
        );
        assert_eq!(
            state.buttons_at(
                start + Duration::from_secs(2),
                start,
                &times,
                hold,
                remu_gpu::PAD_BUTTON_OPTIONS,
            ),
            remu_gpu::PAD_BUTTON_OPTIONS
        );
        assert_eq!(
            state.buttons_at(
                start + Duration::from_millis(2_100),
                start,
                &times,
                hold,
                remu_gpu::PAD_BUTTON_OPTIONS,
            ),
            0
        );
    }
}
