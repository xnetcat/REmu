// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

use super::{
    ProbeDispatchState, compare_i32, dispatch_guest_numeric_parser, guest_free,
    guest_global_locale, guest_qsort, guest_tolower_table, guest_toupper_table, guest_wmemchr,
    guest_wmemcmp, guest_wmemcpy, guest_wmemmove, low_f32, read_guest_u16, read_guest_u64,
    sce_kernel_error,
};

#[test]
fn copies_gen5_wide_characters_as_two_byte_elements() {
    let mut words = [
        0x41_u16, 0x42, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0xaaaa, 0xbbbb, 0xcccc, 0xdddd, 0xeeee,
    ];
    let base = u64::try_from(words.as_mut_ptr().addr()).unwrap();
    let destination = base + 4 * 2;

    assert_eq!(guest_wmemcpy(destination, base, 8), destination);
    assert_eq!(
        words,
        [
            0x41, 0x42, 0x43, 0x44, 0x41, 0x42, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0xeeee,
        ]
    );

    assert_eq!(guest_wmemmove(base, destination, 4), base);
    assert_eq!(&words[..4], &[0x41, 0x42, 0x43, 0x44]);
}

#[test]
fn searches_and_compares_gen5_wide_characters() {
    let left = [0x0061_u16, 0x0062, 0x20ac, 0x0064];
    let right = [0x0061_u16, 0x0062, 0x20ad, 0x0064];
    let left_address = u64::try_from(left.as_ptr().addr()).unwrap();
    let right_address = u64::try_from(right.as_ptr().addr()).unwrap();

    assert_eq!(guest_wmemchr(left_address, 0x20ac, 4), left_address + 4);
    assert_eq!(guest_wmemchr(left_address, 0x9999, 4), 0);
    assert_eq!(guest_wmemcmp(left_address, left_address, 4), 0);
    assert_eq!(guest_wmemcmp(left_address, right_address, 4), u64::MAX);
    assert_eq!(guest_wmemcmp(right_address, left_address, 4), 1);
}

#[test]
fn exposes_a_stable_dinkumware_global_locale() {
    let mut state = ProbeDispatchState::default();
    let locale = guest_global_locale(&mut state);
    let vtable = read_guest_u64(locale);
    let retain = read_guest_u64(vtable + 0x10);
    // SAFETY: guest_global_locale installs this exact process-local C ABI function pointer.
    let retain: extern "C" fn(u64) -> u64 = unsafe {
        std::mem::transmute(usize::try_from(retain).expect("function pointer fits usize"))
    };

    assert_ne!(locale, 0);
    assert_ne!(vtable, 0);
    assert_eq!(retain(locale), locale);
    assert_eq!(guest_global_locale(&mut state), locale);

    guest_free(state.locale_object);
    guest_free(state.locale_vtable);
}

#[test]
fn sign_extends_gen5_kernel_errors() {
    assert_eq!(sce_kernel_error(0x8002_0002), 0xffff_ffff_8002_0002);
}

#[test]
fn builds_distinct_dinkumware_case_mapping_tables() {
    let mut state = ProbeDispatchState::default();
    let lower = guest_tolower_table(&mut state);
    let upper = guest_toupper_table(&mut state);

    assert_ne!(lower, 0);
    assert_ne!(upper, 0);
    assert_ne!(lower, upper);
    assert_eq!(read_guest_u16(lower + u64::from(b'A') * 2), u16::from(b'a'));
    assert_eq!(read_guest_u16(lower + u64::from(b'a') * 2), u16::from(b'a'));
    assert_eq!(read_guest_u16(upper + u64::from(b'a') * 2), u16::from(b'A'));
    assert_eq!(read_guest_u16(upper + u64::from(b'A') * 2), u16::from(b'A'));
    assert_eq!(read_guest_u16(lower - 2), u16::MAX);
    assert_eq!(read_guest_u16(upper - 2), u16::MAX);

    guest_free(lower - 128 * 2);
    guest_free(upper - 128 * 2);
}

#[test]
fn decodes_float_argument_from_low_xmm_bits() {
    let expected = -1.25_f32;
    assert_eq!(
        low_f32(f64::from_bits(u64::from(expected.to_bits()))).to_bits(),
        expected.to_bits()
    );
}

#[test]
fn parses_dinkumware_unsigned_longs_with_end_pointer() {
    let text = c"0x2a tail";
    let mut end = std::ptr::null_mut::<libc::c_char>();
    let result = dispatch_guest_numeric_parser(
        "zlfEH8FmyUA",
        [
            text.as_ptr().addr() as u64,
            (&raw mut end).addr() as u64,
            0,
            0,
            0,
            0,
        ],
    );

    assert_eq!(result.integer, 42);
    assert_eq!(end.addr() - text.as_ptr().addr(), 4);
}

#[test]
fn sorts_guest_elements_with_the_guest_comparator() {
    let mut values = [7_i32, -2, 11, 0, 4];
    guest_qsort([
        u64::try_from(values.as_mut_ptr().addr()).unwrap(),
        u64::try_from(values.len()).unwrap(),
        u64::try_from(std::mem::size_of::<i32>()).unwrap(),
        u64::try_from(compare_i32 as *const () as usize).unwrap(),
        0,
        0,
    ]);
    assert_eq!(values, [-2, 0, 4, 7, 11]);
}
