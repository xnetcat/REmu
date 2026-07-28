// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

use super::*;

#[test]
fn a_submitted_read_batch_fills_every_buffer_and_result_before_it_reports_an_identifier() {
    let directory = TemporaryDirectory::new("aio-read");
    let path = directory.path.join("data.bin");
    std::fs::write(&path, b"0123456789").expect("the test file can be written");
    let file = std::fs::File::open(&path).expect("the test file can be opened");
    let fd = register_descriptor(file);
    let mut state = ProbeDispatchState::default();
    let mut first = [0_u8; 4];
    let mut second = [0_u8; 3];
    let mut results = [0_u64; 4];
    let results_address = results.as_mut_ptr().addr() as u64;
    let mut requests = [0_u64; 10];
    requests[0] = 0;
    requests[1] = 4;
    requests[2] = first.as_mut_ptr().addr() as u64;
    requests[3] = results_address;
    requests[4] = u64::from(fd.cast_unsigned());
    requests[5] = 6;
    requests[6] = 3;
    requests[7] = second.as_mut_ptr().addr() as u64;
    requests[8] = results_address + 0x10;
    requests[9] = u64::from(fd.cast_unsigned());
    let mut identifier = 0_u32;

    let submitted = dispatch(
        "HgX7+AORI58",
        args(&[
            requests.as_ptr().addr() as u64,
            2,
            2,
            std::ptr::addr_of_mut!(identifier).addr() as u64,
        ]),
        &mut state,
    );

    assert_eq!(submitted, Some(SUCCESS));
    assert_eq!(&first, b"0123");
    assert_eq!(&second, b"678");
    // Each result carries the transferred count and then the terminal state.
    assert_eq!(results[0], 4);
    assert_eq!(
        results[1] & u64::from(u32::MAX),
        u64::from(AIO_STATE_COMPLETED)
    );
    assert_eq!(results[2], 3);
    assert_eq!(
        results[3] & u64::from(u32::MAX),
        u64::from(AIO_STATE_COMPLETED)
    );
    assert!(identifier > 0);

    let mut reported = 0_u32;
    let identifier_argument = u64::from(identifier);
    assert_eq!(
        dispatch(
            "2pOuoWoCxdk",
            args(&[
                identifier_argument,
                std::ptr::addr_of_mut!(reported).addr() as u64
            ]),
            &mut state
        ),
        Some(SUCCESS)
    );
    assert_eq!(reported, AIO_STATE_COMPLETED);
    assert_eq!(
        dispatch(
            "KOF-oJbQVvc",
            args(&[
                identifier_argument,
                std::ptr::addr_of_mut!(reported).addr() as u64
            ]),
            &mut state
        ),
        Some(SUCCESS)
    );
    let mut outcome = 0xffff_ffff_u32;
    assert_eq!(
        dispatch(
            "5TgME6AYty4",
            args(&[
                identifier_argument,
                std::ptr::addr_of_mut!(outcome).addr() as u64
            ]),
            &mut state
        ),
        Some(SUCCESS)
    );
    assert_eq!(outcome, 0);
    assert_eq!(
        dispatch(
            "5TgME6AYty4",
            args(&[
                identifier_argument,
                std::ptr::addr_of_mut!(outcome).addr() as u64
            ]),
            &mut state
        ),
        Some(KERNEL_NOT_FOUND)
    );
    release_descriptor(fd);
}

#[test]
fn a_submitted_write_batch_reaches_the_host_file_at_the_offset_it_named() {
    let directory = TemporaryDirectory::new("aio-write");
    let path = directory.path.join("data.bin");
    std::fs::write(&path, b"0123456789").expect("the test file can be written");
    let file = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .open(&path)
        .expect("the test file can be opened");
    let fd = register_descriptor(file);
    let mut state = ProbeDispatchState::default();
    let payload = *b"XY";
    let mut requests = [0_u64; 5];
    requests[0] = 4;
    requests[1] = 2;
    requests[2] = address(&payload);
    requests[3] = 0;
    requests[4] = u64::from(fd.cast_unsigned());
    let mut identifier = 0_u32;

    assert_eq!(
        dispatch(
            "XQ8C8y+de+E",
            args(&[
                requests.as_ptr().addr() as u64,
                1,
                3,
                std::ptr::addr_of_mut!(identifier).addr() as u64
            ]),
            &mut state
        ),
        Some(SUCCESS)
    );
    assert_eq!(
        std::fs::read(&path).expect("the file is still there"),
        b"0123XY6789"
    );
    release_descriptor(fd);
}

#[test]
fn a_submission_naming_a_bad_descriptor_or_count_is_refused_whole() {
    let mut state = ProbeDispatchState::default();
    let mut requests = [0_u64; 5];
    requests[1] = 4;
    requests[2] = 0x1000;
    requests[4] = 9_999;
    let mut identifier = 0_u32;
    let identifier_address = std::ptr::addr_of_mut!(identifier).addr() as u64;
    let requests_address = requests.as_ptr().addr() as u64;

    assert_eq!(
        dispatch(
            "HgX7+AORI58",
            args(&[requests_address, 1, 2, identifier_address]),
            &mut state
        ),
        Some(KERNEL_BAD_FILE_DESCRIPTOR)
    );
    assert_eq!(
        dispatch(
            "HgX7+AORI58",
            args(&[requests_address, 0, 2, identifier_address]),
            &mut state
        ),
        Some(KERNEL_INVALID_ARGUMENT)
    );
    assert_eq!(
        dispatch(
            "HgX7+AORI58",
            args(&[requests_address, 129, 2, identifier_address]),
            &mut state
        ),
        Some(KERNEL_INVALID_ARGUMENT)
    );
    assert_eq!(
        dispatch(
            "HgX7+AORI58",
            args(&[requests_address, 1, 9, identifier_address]),
            &mut state
        ),
        Some(KERNEL_INVALID_ARGUMENT)
    );
    assert_eq!(
        dispatch(
            "HgX7+AORI58",
            args(&[0, 1, 2, identifier_address]),
            &mut state
        ),
        Some(KERNEL_MEMORY_FAULT)
    );
    assert_eq!(
        dispatch(
            "HgX7+AORI58",
            args(&[requests_address, 1, 2, 0]),
            &mut state
        ),
        Some(KERNEL_MEMORY_FAULT)
    );
    assert_eq!(identifier, 0);
}

#[test]
fn polling_several_identifiers_reports_nothing_unless_the_runtime_knows_them_all() {
    let mut state = ProbeDispatchState::default();
    let identifiers = [0x7fff_0001_u32, 0x7fff_0002];
    let mut states = [0xffff_ffff_u32; 2];

    assert_eq!(
        dispatch(
            "o7O4z3jwKzo",
            args(&[
                identifiers.as_ptr().addr() as u64,
                2,
                states.as_mut_ptr().addr() as u64
            ]),
            &mut state
        ),
        Some(KERNEL_NOT_FOUND)
    );
    assert_eq!(states, [0xffff_ffff; 2]);
    assert_eq!(
        dispatch("o7O4z3jwKzo", args(&[0, 2, 0x1000]), &mut state),
        Some(KERNEL_MEMORY_FAULT)
    );
    assert_eq!(
        dispatch(
            "o7O4z3jwKzo",
            args(&[
                identifiers.as_ptr().addr() as u64,
                0,
                states.as_mut_ptr().addr() as u64
            ]),
            &mut state
        ),
        Some(KERNEL_INVALID_ARGUMENT)
    );
}

#[test]
fn a_nid_this_module_does_not_own_is_left_for_the_next_one() {
    let mut state = ProbeDispatchState::default();

    assert_eq!(dispatch("AAAAAAAAAAA", [0; 6], &mut state), None);
    assert_eq!(dispatch("", [0; 6], &mut state), None);
}
