use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

use crate::{
    InitializationState, SCE_KERNEL_ERROR_INVALID_ARGUMENT, guest_memset, low_i32, low_u32,
    read_guest_u8, sce_kernel_error, write_guest_u8, write_guest_u16, write_guest_u32,
    write_guest_u64,
};

pub(crate) const SCE_PAD_ERROR_INVALID_HANDLE: u32 = 0x8092_0003;
pub(crate) const SCE_PAD_ERROR_ALREADY_OPENED: u32 = 0x8092_0004;
pub(crate) const SCE_PAD_ERROR_NOT_INITIALIZED: u32 = 0x8092_0005;
const SCE_PAD_ERROR_DEVICE_NOT_CONNECTED: u32 = 0x8092_0007;
pub(crate) const SCE_PAD_ERROR_DEVICE_NO_HANDLE: u32 = 0x8092_0008;

#[derive(Debug, Default)]
struct PadRuntimeState {
    initialization: InitializationState,
    handle: Option<i32>,
    next_handle: i32,
}

static PAD_RUNTIME: OnceLock<Mutex<PadRuntimeState>> = OnceLock::new();

fn pad_runtime() -> &'static Mutex<PadRuntimeState> {
    PAD_RUNTIME.get_or_init(|| Mutex::new(PadRuntimeState::default()))
}

#[cfg(target_arch = "x86_64")]
pub(crate) fn reset_pad_runtime() {
    *pad_runtime()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner) = PadRuntimeState::default();
}

pub(crate) fn dispatch_pad_import(nid: &str, arguments: [u64; 6]) -> Option<u64> {
    Some(match nid {
        "xk0AcarP3V4" => pad_open(arguments),
        "u1GRHp+oWoY" => pad_get_handle(arguments),
        "clVvL4ZDntw" => pad_set_motion_sensor_state(arguments),
        "gjP9-KQzoUk" => pad_get_controller_information(arguments),
        "6ncge5+l5Qs" => pad_close(arguments),
        "AcslpN1jHR8" => pad_get_extended_information(arguments),
        "YndgXqQVV7c" => pad_read_state(arguments),
        "q1cHNfGycLI" => pad_read(arguments),
        "W2G-yoyMF5U" => pad_set_vibration_mode(arguments),
        "rIZnR6eSpvk" | "DscD1i9HX1w" => pad_handle_operation(arguments),
        "r44mAxdSG+U" | "vDLMoJLde8I" => pad_boolean_operation(arguments),
        "yFVnOdGxvZY" | "2JgFB2n9oUM" | "RR4novUEENY" => pad_pointer_operation(arguments),
        "znaWI0gpuo8" => pad_get_trigger_effect_state(arguments),
        "IHPqcbc0zCA" => pad_device_class_parse_data(arguments),
        "hv1luiJrqQM" => pad_initialize(),
        _ => return None,
    })
}

pub(crate) fn pad_initialize() -> u64 {
    pad_runtime()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .initialization = InitializationState::Initialized;
    0
}

pub(crate) fn pad_open(arguments: [u64; 6]) -> u64 {
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
        return sce_kernel_error(SCE_PAD_ERROR_DEVICE_NOT_CONNECTED);
    }
    if state.handle.is_some() {
        return sce_kernel_error(SCE_PAD_ERROR_ALREADY_OPENED);
    }
    let handle = state.next_handle.max(1);
    state.next_handle = handle.saturating_add(1);
    state.handle = Some(handle);
    u64::from(handle.cast_unsigned())
}

pub(crate) fn pad_get_handle(arguments: [u64; 6]) -> u64 {
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
    let state = pad_runtime()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    validate_pad_handle_state(handle, &state)
}

pub(crate) fn pad_close(arguments: [u64; 6]) -> u64 {
    let mut state = pad_runtime()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if let Err(error) = validate_pad_handle_state(arguments[0], &state) {
        return error;
    }
    state.handle = None;
    0
}

pub(crate) fn pad_get_extended_information(arguments: [u64; 6]) -> u64 {
    if let Err(error) = validate_pad_handle(arguments[0]) {
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

pub(crate) fn pad_set_motion_sensor_state(arguments: [u64; 6]) -> u64 {
    if let Err(error) = validate_pad_handle(arguments[0]) {
        return error;
    }
    if !matches!(low_i32(arguments[1]), 0 | 1) {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    }
    0
}

pub(crate) fn pad_set_vibration_mode(arguments: [u64; 6]) -> u64 {
    if let Err(error) = validate_pad_handle(arguments[0]) {
        return error;
    }
    if low_u32(arguments[1]) > 2 {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    }
    0
}

pub(crate) fn pad_handle_operation(arguments: [u64; 6]) -> u64 {
    validate_pad_handle(arguments[0]).map_or_else(|error| error, |()| 0)
}

pub(crate) fn pad_boolean_operation(arguments: [u64; 6]) -> u64 {
    if let Err(error) = validate_pad_handle(arguments[0]) {
        return error;
    }
    if !matches!(low_u32(arguments[1]), 0 | 1) {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    }
    0
}

pub(crate) fn pad_pointer_operation(arguments: [u64; 6]) -> u64 {
    if let Err(error) = validate_pad_handle(arguments[0]) {
        return error;
    }
    if arguments[1] == 0 {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    }
    0
}

pub(crate) fn pad_get_trigger_effect_state(arguments: [u64; 6]) -> u64 {
    if let Err(error) = validate_pad_handle(arguments[0]) {
        return error;
    }
    if arguments[1] == 0 {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    }
    guest_memset(arguments[1], 0, 8);
    0
}

pub(crate) fn pad_device_class_parse_data(arguments: [u64; 6]) -> u64 {
    if let Err(error) = validate_pad_handle(arguments[0]) {
        return error;
    }
    if arguments[1] == 0 || arguments[2] == 0 {
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

pub(crate) fn pad_read_state(arguments: [u64; 6]) -> u64 {
    static LAST_LOGGED_BUTTONS: AtomicU32 = AtomicU32::new(u32::MAX);
    if let Err(error) = validate_pad_handle(arguments[0]) {
        return error;
    }
    let output = arguments[1];
    if output == 0 {
        return u64::from(libc::EINVAL.cast_unsigned());
    }
    guest_memset(output, 0, 0x78);
    let host = remu_gpu::host_pad_state();
    let buttons = host.buttons | auto_cross_buttons();
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
    let monotonic = std::time::Instant::now();
    // Host Instant has no public epoch; approximate a rising microsecond counter from a process
    // start so pad consumers that only need monotonicity keep working off-macOS.
    static PAD_MONOTONIC_ORIGIN: std::sync::OnceLock<std::time::Instant> =
        std::sync::OnceLock::new();
    let origin = *PAD_MONOTONIC_ORIGIN.get_or_init(|| monotonic);
    let elapsed = monotonic.saturating_duration_since(origin);
    write_guest_u64(
        output + 0x50,
        u64::try_from(elapsed.as_micros()).unwrap_or(u64::MAX),
    );
    write_guest_u8(output + 0x68, 1);
    0
}

pub(crate) fn pad_read(arguments: [u64; 6]) -> u64 {
    if arguments[2] == 0 || arguments[2] > 64 {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    }
    let result = pad_read_state(arguments);
    if result == 0 { 1 } else { result }
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

#[cfg(test)]
mod tests {
    use super::AutoCrossPulseState;
    use std::time::{Duration, Instant};

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
}
