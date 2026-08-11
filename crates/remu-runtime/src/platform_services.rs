// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

//! Platform service entry points a title reaches during startup.
//!
//! These are grouped because they share one property: each is reached by the Bluepoint title
//! before it renders, and each is complete in itself rather than the front of an unimplemented
//! subsystem. Where a call only makes sense against hardware this host does not model — a
//! touchpad, a network interface — that is stated on the call rather than hidden, because the
//! guest is entitled to know the difference between "no touches happened" and "gesture support is
//! missing", and only the former is being claimed.

use std::sync::Mutex;

use crate::{
    SCE_KERNEL_ERROR_INVALID_ARGUMENT, low_i32, process_guest_range_is_readable, sce_kernel_error,
    write_guest_bytes, write_guest_u32,
};

const SCE_USER_SERVICE_ERROR_INVALID_ARGUMENT: u32 = 0x8096_0005;
const SCE_USER_SERVICE_ERROR_INVALID_PARAMETER: u32 = 0x8096_0009;
const SCE_VIDEO_OUT_ERROR_INVALID_VALUE: u32 = 0x8029_0001;
const SCE_VIDEO_OUT_ERROR_INVALID_ADDRESS: u32 = 0x8029_0002;
const SCE_NET_CTL_ERROR_INVALID_ADDRESS: u32 = 0x8041_2101;
const SCE_SYSTEM_GESTURE_ERROR_INVALID_ARGUMENT: u32 = 0x8089_0001;
const SCE_SYSTEM_GESTURE_ERROR_INVALID_HANDLE: u32 = 0x8089_0006;
/// The guest's "no network connection" state value.
const NET_CTL_STATE_DISCONNECTED: u32 = 0;

const SYSTEM_GESTURE_HANDLE: i32 = 0;
const SYSTEM_GESTURE_PRIMITIVE_PARAMETER_SIZE: usize = 64;
const SYSTEM_GESTURE_RECOGNIZER_SIZE: usize = 361 * std::mem::size_of::<u64>();
const SYSTEM_GESTURE_RECTANGLE_SIZE: usize = 24;
const SYSTEM_GESTURE_PARAMETER_SIZE: usize = 64;
const SYSTEM_GESTURE_TOUCH_PAD_DATA_SIZE: usize = 24;

/// Base of the guest's partially-resident-texture address window.
const PRT_AREA_START: u64 = 0x0000_0010_0000_0000;
/// Extent of that window.
const PRT_AREA_SIZE: u64 = 0x0000_00EC_0000_0000;
/// Apertures the hardware exposes over it.
const PRT_APERTURE_COUNT: usize = 3;

/// The recorded partially-resident-texture apertures, as `(base, size)` per aperture.
///
/// A sparse-texturing title sets these once during startup and the mapping is a property of the
/// process, not of any one dispatch state, so it lives here rather than in the per-process
/// dispatch record that resets with a run.
static PRT_APERTURES: Mutex<[(u64, u64); PRT_APERTURE_COUNT]> =
    Mutex::new([(0, 0); PRT_APERTURE_COUNT]);

/// The only privacy parameter a title may query, and the one this host answers.
const USER_SERVICE_PRIVACY_PARAMETER: i32 = 1000;

/// `sceKernelSetPrtAperture(id, base, size)`.
///
/// Records one aperture over the partially-resident-texture window. The call is bookkeeping: it
/// tells the kernel which guest range an aperture covers, and the guest reads nothing back, so
/// validating the window and remembering the placement is the whole contract.
fn kernel_set_prt_aperture(arguments: [u64; 6]) -> u64 {
    let Ok(aperture) = usize::try_from(low_i32(arguments[0])) else {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    };
    let base = arguments[1];
    let size = arguments[2];
    if aperture >= PRT_APERTURE_COUNT
        || base & 0xfff != 0
        || base < PRT_AREA_START
        || size > PRT_AREA_SIZE
        || base - PRT_AREA_START > PRT_AREA_SIZE - size
    {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    }
    PRT_APERTURES
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)[aperture] = (base, size);
    0
}

/// Returns a recorded aperture, for tests and for a future resident-texture path to consult.
#[cfg(test)]
pub(crate) fn prt_aperture(aperture: usize) -> Option<(u64, u64)> {
    PRT_APERTURES
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .get(aperture)
        .copied()
}

/// `sceUserServiceGetPlatformPrivacySetting(parameter, valueOut)`.
///
/// One parameter is defined. The reported value is the unrestricted setting, which is what a
/// single-user host without parental controls or an account service actually is.
fn user_service_platform_privacy_setting(arguments: [u64; 6]) -> u64 {
    let parameter = low_i32(arguments[0]);
    let output = arguments[1];
    if parameter != USER_SERVICE_PRIVACY_PARAMETER {
        return sce_kernel_error(SCE_USER_SERVICE_ERROR_INVALID_PARAMETER);
    }
    if output == 0 {
        return sce_kernel_error(SCE_USER_SERVICE_ERROR_INVALID_ARGUMENT);
    }
    write_guest_u32(output, 0);
    0
}

/// `sceVideoOutColorSettingsSetGamma(settings, gamma)`.
///
/// The gamma travels in the first floating-point register rather than an integer one. The value is
/// range-checked and stored into the caller's settings object, which is where the guest reads it
/// back from before submitting the settings.
pub(crate) fn video_out_color_settings_set_gamma(settings: u64, gamma: f32) -> u64 {
    if settings == 0 {
        return sce_kernel_error(SCE_VIDEO_OUT_ERROR_INVALID_ADDRESS);
    }
    if !gamma.is_finite() || !(0.1..=2.0).contains(&gamma) {
        return sce_kernel_error(SCE_VIDEO_OUT_ERROR_INVALID_VALUE);
    }
    write_guest_u32(settings, gamma.to_bits());
    0
}

/// `sceNetCtlGetState(stateOut)`.
///
/// Reports the disconnected state. This host has no console network configuration, and saying so
/// is the accurate answer rather than a placeholder: a title that branches on connectivity should
/// take its offline path.
fn net_ctl_get_state(output: u64) -> u64 {
    if output == 0 {
        return sce_kernel_error(SCE_NET_CTL_ERROR_INVALID_ADDRESS);
    }
    write_guest_u32(output, NET_CTL_STATE_DISCONNECTED);
    0
}

fn system_gesture_error(error: u32) -> u64 {
    sce_kernel_error(error)
}

fn system_gesture_valid_handle(handle: u64) -> bool {
    low_i32(handle) == SYSTEM_GESTURE_HANDLE
}

/// Initializes the primitive recognizer library. The parameter is an optional reserved record.
fn system_gesture_initialize(parameter: u64) -> u64 {
    if parameter != 0
        && !process_guest_range_is_readable(parameter, SYSTEM_GESTURE_PRIMITIVE_PARAMETER_SIZE)
    {
        return system_gesture_error(SCE_SYSTEM_GESTURE_ERROR_INVALID_ARGUMENT);
    }
    0
}

/// Opens the one supported input source: the controller touchpad.
fn system_gesture_open(arguments: [u64; 6]) -> u64 {
    if low_i32(arguments[0]) != 0 || arguments[1] != 0 {
        return system_gesture_error(SCE_SYSTEM_GESTURE_ERROR_INVALID_ARGUMENT);
    }
    SYSTEM_GESTURE_HANDLE as u64
}

/// Constructs a valid opaque SDK recognizer whose host input stream contains no touches.
fn system_gesture_create_touch_recognizer(arguments: [u64; 6]) -> u64 {
    if !system_gesture_valid_handle(arguments[0]) {
        return system_gesture_error(SCE_SYSTEM_GESTURE_ERROR_INVALID_HANDLE);
    }
    let recognizer = arguments[1];
    let gesture_type = arguments[2] as u32;
    let rectangle = arguments[3];
    let parameter = arguments[4];
    if recognizer == 0
        || !process_guest_range_is_readable(recognizer, SYSTEM_GESTURE_RECOGNIZER_SIZE)
        || !matches!(gesture_type, 1 | 2 | 4 | 8 | 16 | 32)
        || (rectangle != 0
            && !process_guest_range_is_readable(rectangle, SYSTEM_GESTURE_RECTANGLE_SIZE))
        || (parameter != 0
            && !process_guest_range_is_readable(parameter, SYSTEM_GESTURE_PARAMETER_SIZE))
    {
        return system_gesture_error(SCE_SYSTEM_GESTURE_ERROR_INVALID_ARGUMENT);
    }
    write_guest_bytes(recognizer, &[0; SYSTEM_GESTURE_RECOGNIZER_SIZE]);
    0
}

fn system_gesture_update_primitive(arguments: [u64; 6]) -> u64 {
    if !system_gesture_valid_handle(arguments[0]) {
        return system_gesture_error(SCE_SYSTEM_GESTURE_ERROR_INVALID_HANDLE);
    }
    if arguments[1] == 0
        || !process_guest_range_is_readable(arguments[1], SYSTEM_GESTURE_TOUCH_PAD_DATA_SIZE)
    {
        return system_gesture_error(SCE_SYSTEM_GESTURE_ERROR_INVALID_ARGUMENT);
    }
    0
}

fn system_gesture_update_recognizer(arguments: [u64; 6]) -> u64 {
    if !system_gesture_valid_handle(arguments[0]) {
        return system_gesture_error(SCE_SYSTEM_GESTURE_ERROR_INVALID_HANDLE);
    }
    if arguments[1] == 0
        || !process_guest_range_is_readable(arguments[1], SYSTEM_GESTURE_RECOGNIZER_SIZE)
    {
        return system_gesture_error(SCE_SYSTEM_GESTURE_ERROR_INVALID_ARGUMENT);
    }
    0
}

fn system_gesture_touch_event_count(arguments: [u64; 6]) -> u64 {
    let result = system_gesture_update_recognizer(arguments);
    if result == 0 { 0 } else { result }
}

/// Routes the platform service entry points that need only integer arguments.
pub(crate) fn dispatch(nid: &str, arguments: [u64; 6]) -> Option<u64> {
    Some(match nid {
        "BohYr-F7-is" => kernel_set_prt_aperture(arguments),
        "D-CzAxQL0XI" => user_service_platform_privacy_setting(arguments),
        // This host models the controller touchpad as present but untouched. SDK recognizer
        // records are still fully initialized and every update/query observes an empty stream.
        "3pcAvmwKCvM" => system_gesture_initialize(arguments[0]),
        "qpo-mEOwje0" => system_gesture_open(arguments),
        "FWF8zkhr854" => system_gesture_create_touch_recognizer(arguments),
        "GgFMb22sbbI" => system_gesture_update_primitive(arguments),
        "j4h82CQWENo" => system_gesture_update_recognizer(arguments),
        "h8uongcBNVs" => system_gesture_touch_event_count(arguments),
        // Likewise no network interface: initialization succeeds and the state query reports
        // disconnected, which is what this host is. Nothing here fabricates a link.
        "gky0+oaNM4k" => {
            remu_gpu::note_inert(nid);
            0
        }
        "uBPlr0lbuiI" => net_ctl_get_state(arguments[0]),
        // No signed-in network account exists to change state, so a registered state callback is
        // one that never fires. The queries that would report an account stay unserved.
        "qQJfO8HAiaY" => {
            remu_gpu::note_inert(nid);
            0
        }
        // VoiceQoS is peer-to-peer chat over the console network. This host has no link, so
        // initialization only validates the caller-owned workspace; no QoS sessions are created.
        "U8IfNl6-Css" => voice_qos_init(arguments),
        _ => return None,
    })
}

/// `sceVoiceQoSInit(memBlock, memSize, cpuAffinityMask)`.
///
/// Accepts a non-empty caller-owned workspace. Session create and connect stay unserved so titles
/// that probe for chat disable the feature rather than waiting on peers that never appear.
fn voice_qos_init(arguments: [u64; 6]) -> u64 {
    remu_gpu::note_inert("U8IfNl6-Css");
    const VOICE_QOS_ERROR_INVALID_ARGUMENT: u32 = 0x8041_0401;
    let (mem_block, mem_size) = (arguments[0], arguments[1]);
    if mem_block == 0 || mem_size == 0 {
        return sce_kernel_error(VOICE_QOS_ERROR_INVALID_ARGUMENT);
    }
    let _cpu_affinity = arguments[2];
    eprintln!("[REMu][VOICEQOS] initialized workspace={mem_block:#x} size={mem_size:#x} (offline)");
    0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prt_apertures_accept_the_window_and_reject_anything_outside_it() {
        assert_eq!(
            kernel_set_prt_aperture([0, PRT_AREA_START, PRT_AREA_SIZE, 0, 0, 0]),
            0
        );
        assert_eq!(prt_aperture(0), Some((PRT_AREA_START, PRT_AREA_SIZE)));
        // A second aperture is independent of the first.
        assert_eq!(
            kernel_set_prt_aperture([1, PRT_AREA_START + 0x1000, 0x1000, 0, 0, 0]),
            0
        );
        assert_eq!(prt_aperture(1), Some((PRT_AREA_START + 0x1000, 0x1000)));
        assert_eq!(prt_aperture(0), Some((PRT_AREA_START, PRT_AREA_SIZE)));

        for rejected in [
            // Aperture beyond the ones the hardware exposes.
            [PRT_APERTURE_COUNT as u64, PRT_AREA_START, 0x1000, 0, 0, 0],
            // A base that is not page aligned.
            [0, PRT_AREA_START + 1, 0x1000, 0, 0, 0],
            // A base below the window.
            [0, PRT_AREA_START - 0x1000, 0x1000, 0, 0, 0],
            // A size larger than the window.
            [0, PRT_AREA_START, PRT_AREA_SIZE + 0x1000, 0, 0, 0],
            // A placement that would run off the end of the window.
            [0, PRT_AREA_START + 0x1000, PRT_AREA_SIZE, 0, 0, 0],
        ] {
            assert_eq!(
                kernel_set_prt_aperture(rejected),
                sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT),
                "{rejected:?} describes no usable aperture"
            );
        }
    }

    #[test]
    fn platform_privacy_reports_the_unrestricted_setting_for_its_one_parameter() {
        let mut value = 0xffff_ffff_u32;
        let address = (&raw mut value).addr() as u64;
        assert_eq!(
            user_service_platform_privacy_setting([1000, address, 0, 0, 0, 0]),
            0
        );
        assert_eq!(value, 0);
        assert_eq!(
            user_service_platform_privacy_setting([1001, address, 0, 0, 0, 0]),
            sce_kernel_error(SCE_USER_SERVICE_ERROR_INVALID_PARAMETER),
            "only the defined parameter is answerable"
        );
        assert_eq!(
            user_service_platform_privacy_setting([1000, 0, 0, 0, 0, 0]),
            sce_kernel_error(SCE_USER_SERVICE_ERROR_INVALID_ARGUMENT)
        );
    }

    #[test]
    fn gamma_is_range_checked_before_it_reaches_the_settings_object() {
        let mut settings = 0_u32;
        let address = (&raw mut settings).addr() as u64;
        assert_eq!(video_out_color_settings_set_gamma(address, 1.5), 0);
        assert_eq!(f32::from_bits(settings), 1.5);
        assert_eq!(
            video_out_color_settings_set_gamma(0, 1.0),
            sce_kernel_error(SCE_VIDEO_OUT_ERROR_INVALID_ADDRESS)
        );
        for rejected in [0.0_f32, 0.05, 2.5, f32::NAN, f32::INFINITY] {
            assert_eq!(
                video_out_color_settings_set_gamma(address, rejected),
                sce_kernel_error(SCE_VIDEO_OUT_ERROR_INVALID_VALUE),
                "{rejected} is outside the defined gamma range"
            );
        }
        // A rejected value must leave the previous setting intact.
        assert_eq!(f32::from_bits(settings), 1.5);
    }

    #[test]
    fn the_inert_services_report_success_and_are_named() {
        assert_eq!(dispatch("qpo-mEOwje0", [0; 6]), Some(0));
        assert_eq!(dispatch("3pcAvmwKCvM", [0; 6]), Some(0));
        assert_eq!(dispatch("gky0+oaNM4k", [0; 6]), Some(0));
        assert_eq!(
            dispatch("FWF8zkhr854", [0; 6]),
            Some(sce_kernel_error(SCE_SYSTEM_GESTURE_ERROR_INVALID_ARGUMENT))
        );
        assert_eq!(dispatch("qQJfO8HAiaY", [0; 6]), Some(0));
        let mut state = 0xffff_ffff_u32;
        assert_eq!(net_ctl_get_state((&raw mut state).addr() as u64), 0);
        assert_eq!(state, NET_CTL_STATE_DISCONNECTED, "this host is offline");
        assert_eq!(
            net_ctl_get_state(0),
            sce_kernel_error(SCE_NET_CTL_ERROR_INVALID_ADDRESS)
        );
        assert_eq!(dispatch("not-a-nid-11", [0; 6]), None);
        // This NID is the hash of `sceUserServiceGetPlatformPrivacyWs1`. It was previously named
        // `sceUserServiceGetPlatformPrivacySetting`, which is a real symbol but hashes to
        // `Yd+E-89ZPl4` — a different import wearing this one's name, pinned here by this
        // assertion. Recovering a name by hashing the candidate is the check that settles it.
        assert_eq!(
            crate::known_nid_name("D-CzAxQL0XI"),
            Some("sceUserServiceGetPlatformPrivacyWs1")
        );
    }

    #[test]
    fn a_no_touch_gesture_recognizer_has_initialized_state_and_zero_events() {
        let mut recognizer = [0xff_u8; 361 * 8];
        let input = [0_u8; 24];
        let recognizer_address = recognizer.as_mut_ptr() as u64;
        let input_address = input.as_ptr() as u64;

        assert_eq!(dispatch("3pcAvmwKCvM", [0; 6]), Some(0));
        assert_eq!(dispatch("qpo-mEOwje0", [0, 0, 0, 0, 0, 0]), Some(0));
        assert_eq!(
            dispatch("FWF8zkhr854", [0, recognizer_address, 0x20, 0, 0, 0]),
            Some(0)
        );
        assert_eq!(
            recognizer,
            [0; 361 * 8],
            "the SDK opaque recognizer must not retain caller garbage"
        );
        assert_eq!(
            dispatch("GgFMb22sbbI", [0, input_address, 0, 0, 0, 0]),
            Some(0)
        );
        assert_eq!(
            dispatch("j4h82CQWENo", [0, recognizer_address, 0, 0, 0, 0]),
            Some(0)
        );
        assert_eq!(
            dispatch("h8uongcBNVs", [0, recognizer_address, 0, 0, 0, 0]),
            Some(0),
            "an untouched host touchpad has no gesture events"
        );

        let invalid_handle = sce_kernel_error(SCE_SYSTEM_GESTURE_ERROR_INVALID_HANDLE);
        assert_eq!(
            dispatch("j4h82CQWENo", [1, recognizer_address, 0, 0, 0, 0]),
            Some(invalid_handle)
        );
        let invalid_argument = sce_kernel_error(SCE_SYSTEM_GESTURE_ERROR_INVALID_ARGUMENT);
        assert_eq!(
            dispatch("FWF8zkhr854", [0, 0, 0x20, 0, 0, 0]),
            Some(invalid_argument)
        );
    }
}
