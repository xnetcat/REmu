// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

use super::*;

#[test]
fn unlinking_removes_a_file_and_refuses_a_directory_or_a_missing_name() {
    let mut fixture = MountFixture::new("unlink");
    fixture.write("save.bin", b"payload");
    std::fs::create_dir(fixture.directory.path.join("folder")).expect("the directory is made");
    let file = guest_string("/savedata0/save.bin");
    let folder = guest_string("/savedata0/folder");
    let missing = guest_string("/savedata0/absent.bin");

    assert_eq!(
        fixture.call("AUXVxWeJU-A", args(&[address(&file)])),
        SUCCESS
    );
    assert!(!fixture.directory.path.join("save.bin").exists());
    assert_eq!(
        fixture.call("AUXVxWeJU-A", args(&[address(&folder)])),
        KERNEL_IS_DIRECTORY
    );
    assert_eq!(
        fixture.call("AUXVxWeJU-A", args(&[address(&missing)])),
        KERNEL_NOT_FOUND
    );
    assert_eq!(fixture.call("AUXVxWeJU-A", [0; 6]), KERNEL_MEMORY_FAULT);
}

#[test]
fn the_posix_spelling_of_a_path_call_reports_minus_one_rather_than_a_kernel_code() {
    let mut fixture = MountFixture::new("posix-path");
    fixture.write("save.bin", b"payload");
    let file = guest_string("/savedata0/save.bin");
    let missing = guest_string("/savedata0/absent.bin");

    assert_eq!(
        fixture.call("VAzswvTOCzI", args(&[address(&missing)])),
        POSIX_FAILURE
    );
    assert_eq!(
        fixture.call("VAzswvTOCzI", args(&[address(&file)])),
        SUCCESS
    );
    assert!(!fixture.directory.path.join("save.bin").exists());
}

#[test]
fn a_read_only_mount_refuses_every_call_that_would_change_it() {
    let mut fixture = MountFixture::with_read_only("read-only", true);
    fixture.write("save.bin", b"payload");
    let file = guest_string("/savedata0/save.bin");
    let renamed = guest_string("/savedata0/other.bin");
    let times = [0_u64; 4];

    for nid in ["AUXVxWeJU-A", "fgIsQ10xYVA", "WlyEA-sLDf0"] {
        assert_eq!(
            fixture.call(nid, args(&[address(&file), 0])),
            KERNEL_PERMISSION_DENIED,
            "{nid} accepted a read-only mount"
        );
    }
    assert_eq!(
        fixture.call("52NcYU9+lEo", args(&[address(&file), address(&renamed)])),
        KERNEL_PERMISSION_DENIED
    );
    assert_eq!(
        fixture.call(
            "0Cq8ipKr9n0",
            args(&[address(&file), times.as_ptr().addr() as u64])
        ),
        KERNEL_PERMISSION_DENIED
    );
    // Reading the mount is still allowed.
    assert_eq!(
        fixture.call("uWyW3v98sU4", args(&[address(&file)])),
        SUCCESS
    );
    assert!(fixture.directory.path.join("save.bin").exists());
}

#[test]
fn renaming_moves_a_file_and_reports_a_missing_source() {
    let mut fixture = MountFixture::new("rename");
    fixture.write("from.bin", b"payload");
    let from = guest_string("/savedata0/from.bin");
    let to = guest_string("/savedata0/to.bin");
    let missing = guest_string("/savedata0/absent.bin");

    assert_eq!(
        fixture.call("52NcYU9+lEo", args(&[address(&from), address(&to)])),
        SUCCESS
    );
    assert_eq!(
        std::fs::read(fixture.directory.path.join("to.bin")).expect("the moved file is there"),
        b"payload"
    );
    assert_eq!(
        fixture.call("52NcYU9+lEo", args(&[address(&missing), address(&to)])),
        KERNEL_NOT_FOUND
    );
    assert_eq!(
        fixture.call("NN01qLRhiqU", args(&[address(&missing), address(&to)])),
        POSIX_FAILURE
    );
}

#[test]
fn removing_a_directory_through_the_posix_name_reports_minus_one_when_it_is_absent() {
    let mut fixture = MountFixture::new("rmdir");
    std::fs::create_dir(fixture.directory.path.join("folder")).expect("the directory is made");
    let folder = guest_string("/savedata0/folder");
    let missing = guest_string("/savedata0/absent");

    assert_eq!(
        fixture.call("c7ZnT7V1B98", args(&[address(&missing)])),
        POSIX_FAILURE
    );
    assert_eq!(
        fixture.call("c7ZnT7V1B98", args(&[address(&folder)])),
        SUCCESS
    );
    assert!(!fixture.directory.path.join("folder").exists());
}

#[test]
fn changing_permissions_keeps_only_the_mode_bits_the_guest_may_set() {
    let mut fixture = MountFixture::new("chmod");
    let path = fixture.write("save.bin", b"payload");
    let file = guest_string("/savedata0/save.bin");

    // The high bits above the permission field belong to the file type and must not survive.
    let mode = 0xffff_0000 | u64::from(0o640_u32);
    assert_eq!(
        fixture.call("fgIsQ10xYVA", args(&[address(&file), mode])),
        SUCCESS
    );
    let mode = std::fs::metadata(&path)
        .expect("the file is still there")
        .permissions()
        .mode()
        & KERNEL_MODE_BITS;
    assert_eq!(mode, 0o640);
}

#[test]
fn truncating_a_path_sets_its_length_and_rejects_a_negative_one() {
    let mut fixture = MountFixture::new("truncate");
    let path = fixture.write("save.bin", b"0123456789");
    let file = guest_string("/savedata0/save.bin");

    assert_eq!(
        fixture.call("WlyEA-sLDf0", args(&[address(&file), 4])),
        SUCCESS
    );
    assert_eq!(
        std::fs::read(&path).expect("the file is still there"),
        b"0123"
    );
    assert_eq!(
        fixture.call("WlyEA-sLDf0", args(&[address(&file), u64::MAX])),
        KERNEL_INVALID_ARGUMENT
    );
}

#[test]
fn setting_times_decodes_both_eight_byte_fields_of_each_entry() {
    let mut fixture = MountFixture::new("utimes");
    let path = fixture.write("save.bin", b"payload");
    let file = guest_string("/savedata0/save.bin");
    // Access seconds and microseconds, then modification seconds and microseconds.
    let times: [u64; 4] = [1_000_000, 250_000, 2_000_000, 500_000];

    assert_eq!(
        fixture.call(
            "0Cq8ipKr9n0",
            args(&[address(&file), times.as_ptr().addr() as u64])
        ),
        SUCCESS
    );
    let modified = std::fs::metadata(&path)
        .expect("the file is still there")
        .modified()
        .expect("the host reports a modification time")
        .duration_since(std::time::UNIX_EPOCH)
        .expect("the modification time is after the epoch");
    assert_eq!(modified.as_secs(), 2_000_000);
    assert_eq!(modified.subsec_micros(), 500_000);
}

#[test]
fn a_fractional_second_outside_its_range_is_rejected_before_the_host_is_asked() {
    let mut fixture = MountFixture::new("utimes-range");
    fixture.write("save.bin", b"payload");
    let file = guest_string("/savedata0/save.bin");
    let times: [u64; 4] = [1, 1_000_000, 2, 0];

    assert_eq!(
        fixture.call(
            "0Cq8ipKr9n0",
            args(&[address(&file), times.as_ptr().addr() as u64])
        ),
        KERNEL_INVALID_ARGUMENT
    );
}

#[test]
fn reachability_separates_a_present_path_from_an_absent_one() {
    let mut fixture = MountFixture::new("reachability");
    fixture.write("save.bin", b"payload");
    let file = guest_string("/savedata0/save.bin");
    let missing = guest_string("/savedata0/absent.bin");

    assert_eq!(
        fixture.call("uWyW3v98sU4", args(&[address(&file)])),
        SUCCESS
    );
    assert_eq!(
        fixture.call("uWyW3v98sU4", args(&[address(&missing)])),
        KERNEL_NOT_FOUND
    );
    assert_eq!(fixture.call("uWyW3v98sU4", [0; 6]), KERNEL_MEMORY_FAULT);
}

#[test]
fn positional_transfers_leave_the_descriptor_cursor_where_they_found_it() {
    let directory = TemporaryDirectory::new("positional");
    let path = directory.path.join("data.bin");
    std::fs::write(&path, b"0123456789").expect("the test file can be written");
    let file = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .open(&path)
        .expect("the test file can be opened");
    let fd = register_descriptor(file);
    let mut state = ProbeDispatchState::default();
    let mut buffer = [0_u8; 4];

    let fd_argument = u64::from(fd.cast_unsigned());
    let read = dispatch(
        "+r3rMFwItV4",
        args(&[fd_argument, buffer.as_mut_ptr().addr() as u64, 4, 6]),
        &mut state,
    );
    assert_eq!(read, Some(4));
    assert_eq!(&buffer, b"6789");

    let written = guest_string("AB");
    let wrote = dispatch(
        "nKWi-N2HBV4",
        args(&[fd_argument, address(&written), 2, 0]),
        &mut state,
    );
    assert_eq!(wrote, Some(2));
    assert_eq!(
        std::fs::read(&path).expect("the file is still there"),
        b"AB23456789"
    );

    // A positional transfer never moves the shared cursor, so the descriptor still reads from
    // the beginning of the file.
    let position = kernel_files()
        .files
        .get_mut(&fd)
        .expect("the descriptor is open")
        .stream_position()
        .expect("the host reports the descriptor cursor");
    assert_eq!(position, 0);
    release_descriptor(fd);
}

#[test]
#[cfg(any(target_os = "linux", target_os = "macos"))]
fn positional_reads_copy_through_cpu_write_watches() {
    let directory = TemporaryDirectory::new("positional-write-watch");
    let path = directory.path.join("data.bin");
    std::fs::write(&path, b"watched").expect("the test file can be written");
    let file = std::fs::File::open(&path).expect("the test file can be opened");
    let fd = register_descriptor(file);
    let page_size =
        usize::try_from(unsafe { libc::sysconf(libc::_SC_PAGESIZE) }).expect("host page size");
    // SAFETY: this test owns the anonymous mapping until after the watch is retired.
    let mapping = unsafe {
        libc::mmap(
            std::ptr::null_mut(),
            page_size,
            libc::PROT_READ | libc::PROT_WRITE,
            libc::MAP_PRIVATE | libc::MAP_ANON,
            -1,
            0,
        )
    };
    assert_ne!(mapping, libc::MAP_FAILED);
    let address = mapping.addr() as u64;
    // SAFETY: the complete watched page is a live readable/writable anonymous mapping.
    let watch = unsafe { remu_memory::CpuWriteWatch::new(address, page_size as u64).unwrap() };
    let mut state = ProbeDispatchState::default();

    let read = dispatch(
        "+r3rMFwItV4",
        args(&[u64::from(fd.cast_unsigned()), address + 7, 7, 0]),
        &mut state,
    );

    assert_eq!(read, Some(7));
    assert!(watch.is_dirty());
    // SAFETY: the mapping remains live and the staged read made its first page readable.
    let bytes = unsafe { std::slice::from_raw_parts((address + 7) as *const u8, 7) };
    assert_eq!(bytes, b"watched");
    drop(watch);
    remu_memory::CpuWriteWatch::retire_overlapping(address, page_size as u64);
    // SAFETY: no watch retains this mapping and the test owns the exact mmap span.
    assert_eq!(unsafe { libc::munmap(mapping, page_size) }, 0);
    release_descriptor(fd);
}

#[test]
fn a_positional_transfer_rejects_an_unopened_descriptor_and_a_null_buffer() {
    let mut state = ProbeDispatchState::default();

    assert_eq!(
        dispatch("+r3rMFwItV4", args(&[9_999, 0x1000, 4, 0]), &mut state),
        Some(KERNEL_BAD_FILE_DESCRIPTOR)
    );
    assert_eq!(
        dispatch("nKWi-N2HBV4", args(&[9_999, 0, 4, 0]), &mut state),
        Some(KERNEL_MEMORY_FAULT)
    );
    assert_eq!(
        dispatch(
            "+r3rMFwItV4",
            args(&[9_999, 0x1000, 4, u64::MAX]),
            &mut state
        ),
        Some(KERNEL_INVALID_ARGUMENT)
    );
    assert_eq!(
        dispatch("fTx66l5iWIA", args(&[9_999]), &mut state),
        Some(KERNEL_BAD_FILE_DESCRIPTOR)
    );
}

#[test]
fn flushing_an_open_descriptor_commits_its_contents() {
    let directory = TemporaryDirectory::new("fsync");
    let path = directory.path.join("data.bin");
    let file = std::fs::File::create(&path).expect("the test file can be created");
    let fd = register_descriptor(file);
    let mut state = ProbeDispatchState::default();

    assert_eq!(
        dispatch(
            "fTx66l5iWIA",
            args(&[u64::from(fd.cast_unsigned())]),
            &mut state
        ),
        Some(SUCCESS)
    );
    release_descriptor(fd);
}
