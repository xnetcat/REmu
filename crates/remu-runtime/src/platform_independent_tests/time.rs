// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

use super::{
    civil_from_unix_seconds, guest_clock_gettime, guest_kernel_convert_localtime_to_utc,
    guest_kernel_convert_utc_to_localtime, guest_kernel_gettimeofday,
    rtc_get_current_clock_local_time, rtc_get_current_tick, rtc_get_tick, rtc_set_tick,
    unix_days_from_civil,
};

#[test]
fn translates_console_clock_ids_to_realtime() {
    let before = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs();
    let mut output = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    assert_eq!(guest_clock_gettime(4, (&raw mut output).addr() as u64), 0);
    let seconds = u64::try_from(output.tv_sec).unwrap();
    assert!(seconds >= before);
    assert!(seconds <= before + 2);
    assert!((0..1_000_000_000).contains(&output.tv_nsec));
}

#[test]
fn fills_gen5_calendar_time_and_round_trips_timezone_conversions() {
    let before = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs();
    let mut timeval = [0_u64; 2];
    assert_eq!(
        guest_kernel_gettimeofday((&raw mut timeval).addr() as u64),
        0
    );
    assert!((before..=before + 2).contains(&timeval[0]));
    assert!(timeval[1] < 1_000_000);

    let utc = 1_700_000_000_i64;
    let mut local = 0_i64;
    let mut timesec = [0_u8; 16];
    let mut dst = 0_u64;
    assert_eq!(
        guest_kernel_convert_utc_to_localtime([
            u64::from_ne_bytes(utc.to_ne_bytes()),
            (&raw mut local).addr() as u64,
            (&raw mut timesec).addr() as u64,
            (&raw mut dst).addr() as u64,
            0,
            0,
        ]),
        0
    );
    assert_eq!(i64::from_le_bytes(timesec[..8].try_into().unwrap()), utc);
    assert_eq!(
        u64::from(u32::from_le_bytes(timesec[12..16].try_into().unwrap())),
        dst
    );

    let mut round_trip = 0_i64;
    let mut timezone = [0_i32; 2];
    let mut dst_seconds = 0_i32;
    assert_eq!(
        guest_kernel_convert_localtime_to_utc([
            u64::from_ne_bytes(local.to_ne_bytes()),
            0,
            (&raw mut round_trip).addr() as u64,
            (&raw mut timezone).addr() as u64,
            (&raw mut dst_seconds).addr() as u64,
            0,
        ]),
        0
    );
    assert_eq!(round_trip, utc);
    assert_eq!(dst_seconds, i32::try_from(dst).unwrap());
    assert_eq!(timezone[1], dst_seconds / 60);
}

/// Engines store timestamps as ticks and decompose them again for display. A tick that does
/// not survive the round trip produces an implausible date and fails their validation.
#[test]
fn rtc_tick_round_trips_through_a_date() {
    let mut date = [0_u16; 8];
    date[0] = 2026;
    date[1] = 7;
    date[2] = 25;
    date[3] = 18;
    date[4] = 43;
    date[5] = 7;
    let mut tick = 0_u64;
    assert_eq!(
        rtc_get_tick([
            date.as_mut_ptr().addr() as u64,
            (&raw mut tick).addr() as u64,
            0,
            0,
            0,
            0,
        ]),
        0
    );
    assert_ne!(tick, 0);

    let mut recovered = [0_u16; 8];
    assert_eq!(
        rtc_set_tick([
            recovered.as_mut_ptr().addr() as u64,
            (&raw const tick).addr() as u64,
            0,
            0,
            0,
            0,
        ]),
        0
    );
    assert_eq!(recovered[..6], date[..6]);

    // The current clock must also decompose to the same instant it reports as a tick.
    let mut now_tick = 0_u64;
    assert_eq!(rtc_get_current_tick((&raw mut now_tick).addr() as u64), 0);
    let mut now_date = [0_u16; 8];
    assert_eq!(
        rtc_set_tick([
            now_date.as_mut_ptr().addr() as u64,
            (&raw const now_tick).addr() as u64,
            0,
            0,
            0,
            0,
        ]),
        0
    );
    assert!(now_date[0] >= 2020, "year was {}", now_date[0]);
    assert!((1..=12).contains(&now_date[1]));
}

#[test]
fn civil_days_round_trip_through_civil_decomposition() {
    for seconds in [
        0_i64,
        1_709_164_800,
        951_782_400,
        -2_203_891_201,
        1_800_000_000,
    ] {
        let (year, month, day, hour, minute, second) = civil_from_unix_seconds(seconds);
        let days = unix_days_from_civil(i64::from(year), i64::from(month), i64::from(day));
        let rebuilt =
            days * 86_400 + i64::from(hour) * 3_600 + i64::from(minute) * 60 + i64::from(second);
        assert_eq!(rebuilt, seconds, "round trip failed for {seconds}");
    }
}

#[test]
fn civil_decomposition_matches_known_timestamps() {
    // Epoch, a leap day, the end of a non-leap century year, and a pre-epoch date.
    assert_eq!(civil_from_unix_seconds(0), (1970, 1, 1, 0, 0, 0));
    assert_eq!(
        civil_from_unix_seconds(1_709_164_800),
        (2024, 2, 29, 0, 0, 0)
    );
    // Year 2000 is a leap year despite being a century, so the day after February 28 exists.
    assert_eq!(
        civil_from_unix_seconds(951_782_399),
        (2000, 2, 28, 23, 59, 59)
    );
    assert_eq!(civil_from_unix_seconds(951_782_400), (2000, 2, 29, 0, 0, 0));
    // Year 1900 is not a leap year, so February ends after the 28th.
    assert_eq!(
        civil_from_unix_seconds(-2_203_891_201),
        (1900, 2, 28, 23, 59, 59)
    );
    assert_eq!(civil_from_unix_seconds(-1), (1969, 12, 31, 23, 59, 59));
}

#[test]
fn current_local_time_is_a_valid_date_and_rejects_a_null_output() {
    let mut time = [0_u16; 8];
    assert_eq!(
        rtc_get_current_clock_local_time(time.as_mut_ptr().addr() as u64),
        0
    );
    // Engines abort on an all-zero date, so every field must be in its calendar range.
    assert!(time[0] >= 2020, "year was {}", time[0]);
    assert!((1..=12).contains(&time[1]), "month was {}", time[1]);
    assert!((1..=31).contains(&time[2]), "day was {}", time[2]);
    assert!(time[3] < 24 && time[4] < 60 && time[5] < 60);
    assert_ne!(rtc_get_current_clock_local_time(0), 0);
}
