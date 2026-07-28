// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

use super::*;

#[test]
fn an_address_is_formatted_and_parsed_back_to_the_bytes_it_started_from() {
    let mut state = ProbeDispatchState::default();
    let source = [192_u8, 168, 1, 200];
    let mut text = [0_u8; 46];
    let text_address = text.as_mut_ptr().addr() as u64;

    assert_eq!(
        dispatch(
            "5jRCs2axtr4",
            args(&[
                u64::from(AF_INET.cast_unsigned()),
                address(&source),
                text_address,
                46
            ]),
            &mut state
        ),
        Some(text_address)
    );
    assert_eq!(&text[..16], b"192.168.1.200\0\0\0");

    let mut parsed = [0_u8; 4];
    assert_eq!(
        dispatch(
            "4n51s0zEf0c",
            args(&[
                u64::from(AF_INET.cast_unsigned()),
                text_address,
                parsed.as_mut_ptr().addr() as u64
            ]),
            &mut state
        ),
        Some(1)
    );
    assert_eq!(parsed, source);
}

#[test]
fn a_sixteen_byte_address_round_trips_through_its_compressed_text_form() {
    let mut state = ProbeDispatchState::default();
    let source = [
        0x20_u8, 0x01, 0x0d, 0xb8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0x01,
    ];
    let mut text = [0_u8; 46];
    let text_address = text.as_mut_ptr().addr() as u64;

    assert_eq!(
        dispatch(
            "5jRCs2axtr4",
            args(&[
                u64::from(AF_INET6.cast_unsigned()),
                address(&source),
                text_address,
                46
            ]),
            &mut state
        ),
        Some(text_address)
    );
    assert_eq!(&text[..11], b"2001:db8::1");
    assert_eq!(text[11], 0);

    let mut parsed = [0_u8; 16];
    assert_eq!(
        dispatch(
            "4n51s0zEf0c",
            args(&[
                u64::from(AF_INET6.cast_unsigned()),
                text_address,
                parsed.as_mut_ptr().addr() as u64
            ]),
            &mut state
        ),
        Some(1)
    );
    assert_eq!(parsed, source);
}

#[test]
fn address_conversion_separates_an_unknown_family_from_text_that_is_not_an_address() {
    let mut state = ProbeDispatchState::default();
    let source = [1_u8, 2, 3, 4];
    let mut text = [0_u8; 8];
    let malformed = guest_string("192.168.1");

    assert_eq!(
        dispatch(
            "5jRCs2axtr4",
            args(&[
                99,
                address(&source),
                text.as_mut_ptr().addr() as u64,
                text.len() as u64
            ]),
            &mut state
        ),
        Some(0)
    );
    // A dotted quad needs more room than this buffer offers.
    let source = [192_u8, 168, 1, 200];
    assert_eq!(
        dispatch(
            "5jRCs2axtr4",
            args(&[
                u64::from(AF_INET.cast_unsigned()),
                address(&source),
                text.as_mut_ptr().addr() as u64,
                8
            ]),
            &mut state
        ),
        Some(0)
    );
    let mut parsed = [0_u8; 4];
    assert_eq!(
        dispatch(
            "4n51s0zEf0c",
            args(&[
                u64::from(AF_INET.cast_unsigned()),
                address(&malformed),
                parsed.as_mut_ptr().addr() as u64
            ]),
            &mut state
        ),
        Some(0)
    );
    assert_eq!(
        dispatch(
            "4n51s0zEf0c",
            args(&[99, address(&malformed), parsed.as_mut_ptr().addr() as u64]),
            &mut state
        ),
        Some(POSIX_FAILURE)
    );
}
