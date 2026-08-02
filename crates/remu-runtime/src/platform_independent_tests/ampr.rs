// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

use std::sync::{Arc, Mutex, RwLock, mpsc};
use std::time::Duration;

use super::{
    AmprCommandBufferState, ProbeDispatchState, SCE_KERNEL_ERROR_NOT_FOUND,
    ampr_command_buffer_write_address_on_completion, ampr_command_buffer_write_kernel_event_queue,
    ampr_submit_command_buffer, create_ampr_test_file, dispatch_probe_import,
    kernel_add_ampr_event, read_guest_u32, sce_kernel_error, set_current_guest_thread_pointer,
    write_guest_u32,
};

#[cfg(target_os = "macos")]
#[test]
fn apr_resolver_imports_report_posix_failures_to_the_sdk_wrappers() {
    let mut tls = [0_u64; 16];
    let thread_pointer = tls.as_mut_ptr().addr() as u64;
    set_current_guest_thread_pointer(thread_pointer);

    let prefix = b"/app0\0";
    let absolute = b"/app0/missing-apr-file.bin\0";
    let relative = b"missing-apr-file.bin\0";
    let absolute_list = [absolute.as_ptr().addr() as u64];
    let relative_list = [relative.as_ptr().addr() as u64];
    let mut state = ProbeDispatchState::default();

    for (nid, prefixed, per_entry) in [
        ("gEpBkcwxUjw", false, false),
        ("WT-5NKy42fw", false, false),
        ("eYAh2vlCY-U", false, true),
        ("QzB4O+bJQyA", false, true),
        ("i3HWvW35jao", true, false),
        ("w5fcCG+t31g", true, false),
        ("VB-BtuIW8Xc", true, true),
        ("C+Khtbbx2g8", true, true),
    ] {
        let mut id = 0_u32;
        let mut size = u64::MAX;
        let mut detail = u32::MAX;
        write_guest_u32(thread_pointer + 0x40, 77);
        let arguments = if prefixed {
            [
                prefix.as_ptr().addr() as u64,
                relative_list.as_ptr().addr() as u64,
                1,
                (&raw mut id).addr() as u64,
                if matches!(nid, "w5fcCG+t31g" | "C+Khtbbx2g8") {
                    (&raw mut size).addr() as u64
                } else {
                    (&raw mut detail).addr() as u64
                },
                (&raw mut detail).addr() as u64,
            ]
        } else {
            [
                absolute_list.as_ptr().addr() as u64,
                1,
                (&raw mut id).addr() as u64,
                if matches!(nid, "gEpBkcwxUjw" | "QzB4O+bJQyA") {
                    (&raw mut size).addr() as u64
                } else {
                    (&raw mut detail).addr() as u64
                },
                (&raw mut detail).addr() as u64,
                0,
            ]
        };
        let result = dispatch_probe_import(nid, arguments, [0; 6], &mut state);

        assert_eq!(id, crate::AMPR_INVALID_FILE_ID, "{nid}");
        if matches!(
            nid,
            "gEpBkcwxUjw" | "QzB4O+bJQyA" | "w5fcCG+t31g" | "C+Khtbbx2g8"
        ) {
            assert_eq!(size, 0, "{nid}");
        }
        if per_entry {
            assert_eq!(result, 0, "{nid}");
            assert_eq!(detail, SCE_KERNEL_ERROR_NOT_FOUND, "{nid} per-entry result");
            assert_eq!(read_guest_u32(thread_pointer + 0x40), 77, "{nid}");
        } else {
            assert_eq!(result, u64::MAX, "{nid}");
            assert_eq!(
                read_guest_u32(thread_pointer + 0x40),
                libc::ENOENT.cast_unsigned(),
                "{nid}"
            );
            assert_eq!(detail, 0, "{nid} error index");
        }
    }

    set_current_guest_thread_pointer(0);
}

#[cfg(target_os = "macos")]
#[test]
fn apr_resolvers_reject_unreadable_guest_ranges_without_dereferencing_them() {
    let mut tls = [0_u64; 16];
    let thread_pointer = tls.as_mut_ptr().addr() as u64;
    set_current_guest_thread_pointer(thread_pointer);

    let path = b"/app0/missing-apr-file.bin\0";
    let path_list = [path.as_ptr().addr() as u64];
    let mut id = 0x1234_5678_u32;
    let mut size = u64::MAX;
    let mut state = ProbeDispatchState::default();

    let result = dispatch_probe_import(
        "gEpBkcwxUjw",
        [
            path_list.as_ptr().addr() as u64,
            1,
            (&raw mut id).addr() as u64,
            1,
            0,
            0,
        ],
        [0; 6],
        &mut state,
    );
    assert_eq!(result, u64::MAX);
    assert_eq!(read_guest_u32(thread_pointer + 0x40), libc::EFAULT.cast_unsigned());
    assert_eq!(id, 0x1234_5678, "validation must precede output writes");

    write_guest_u32(thread_pointer + 0x40, 0);
    let result = dispatch_probe_import(
        "w5fcCG+t31g",
        [
            1,
            path_list.as_ptr().addr() as u64,
            1,
            (&raw mut id).addr() as u64,
            (&raw mut size).addr() as u64,
            0,
        ],
        [0; 6],
        &mut state,
    );
    assert_eq!(result, u64::MAX);
    assert_eq!(read_guest_u32(thread_pointer + 0x40), libc::EFAULT.cast_unsigned());

    set_current_guest_thread_pointer(0);
}

#[test]
fn ampr_read_records_a_deferred_transfer_without_touching_the_destination() {
    let path = create_ampr_test_file(b"host payload");
    let file_id = 0x1234_5678;
    let command_buffer = 0x5000;
    let mut command_records = [0_u8; 0x60];
    let mut destination = *b"live";
    let mut state = ProbeDispatchState::default();
    state.ampr_files.insert(file_id, path.clone());
    state.ampr_command_buffers.insert(
        command_buffer,
        AmprCommandBufferState {
            buffer: command_records.as_mut_ptr().addr() as u64,
            size: command_records.len() as u64,
            ..AmprCommandBufferState::default()
        },
    );
    let destination_address = destination.as_mut_ptr().addr() as u64;

    assert_eq!(
        crate::ampr_command_buffer_record_read(
            [command_buffer, 0, 0, file_id.into(), destination_address, 4],
            [7, 0, 0, 0, 0, 0],
            &mut state,
        ),
        0
    );

    assert_eq!(&destination, b"live");
    let command = state.ampr_command_buffers.get(&command_buffer).unwrap();
    assert_eq!(command.write_offset, crate::AMPR_COMMAND_RECORD_SIZE);
    assert_eq!(command.command_count, 1);
    assert_eq!(command.pending_reads.len(), 1);
    assert!(state.ampr_file_handles.is_empty());
    assert_eq!(
        u32::from_le_bytes(command_records[0..4].try_into().unwrap()),
        1
    );
    assert_eq!(
        u32::from_le_bytes(command_records[4..8].try_into().unwrap()),
        file_id
    );
    assert_eq!(
        u64::from_le_bytes(command_records[0x18..0x20].try_into().unwrap()),
        7
    );
    assert_eq!(
        u64::from_le_bytes(command_records[0x20..0x28].try_into().unwrap()),
        0
    );
    std::fs::remove_file(path).ok();
}

#[test]
fn ampr_host_read_honors_offset_and_short_end_of_file() {
    let path = create_ampr_test_file(b"012345");
    let file = std::fs::File::open(&path).unwrap();
    let mut destination = [0xcc_u8; 8];
    assert_eq!(
        crate::ampr_read_file_to_guest(
            &file,
            0x1234,
            destination.as_mut_ptr().addr() as u64,
            destination.len() as u64,
            2,
            destination.len(),
        ),
        Ok(4)
    );
    assert_eq!(&destination[..4], b"2345");
    assert_eq!(&destination[4..], &[0xcc; 4]);
    std::fs::remove_file(path).ok();
}

#[test]
fn missing_ampr_file_is_rejected_before_recording() {
    let file_id = 0x2468_1357_u32;
    let command_buffer = 0x5500;
    let mut command_records = [0xa5_u8; 0x60];
    let mut destination = [0_u8; 4];
    let mut state = ProbeDispatchState::default();
    state.ampr_command_buffers.insert(
        command_buffer,
        AmprCommandBufferState {
            buffer: command_records.as_mut_ptr().addr() as u64,
            size: command_records.len() as u64,
            ..AmprCommandBufferState::default()
        },
    );

    assert_eq!(
        crate::ampr_command_buffer_record_read(
            [
                command_buffer,
                0,
                0,
                file_id.into(),
                destination.as_mut_ptr().addr() as u64,
                destination.len() as u64,
            ],
            [0; 6],
            &mut state,
        ),
        sce_kernel_error(SCE_KERNEL_ERROR_NOT_FOUND)
    );

    let command = state.ampr_command_buffers.get(&command_buffer).unwrap();
    assert_eq!(command.write_offset, 0);
    assert_eq!(command.command_count, 0);
    assert_eq!(command_records, [0xa5; 0x60]);
}

#[test]
fn ampr_submit_runs_reads_in_order_outside_dispatch_and_reuses_the_program() {
    let path = create_ampr_test_file(b"01234567");
    let file_id = 0x8765_4321;
    let command_buffer = 0x6000;
    let mut command_records = [0_u8; 0x90];
    let mut first = [0_u8; 4];
    let mut second = [0_u8; 4];
    let mut state = ProbeDispatchState::default();
    state.ampr_files.insert(file_id, path.clone());
    state.ampr_command_buffers.insert(
        command_buffer,
        AmprCommandBufferState {
            buffer: command_records.as_mut_ptr().addr() as u64,
            size: command_records.len() as u64,
            ..AmprCommandBufferState::default()
        },
    );
    assert_eq!(
        crate::ampr_command_buffer_record_read(
            [
                command_buffer,
                0,
                0,
                file_id.into(),
                first.as_mut_ptr().addr() as u64,
                4,
            ],
            [0; 6],
            &mut state,
        ),
        0
    );
    assert_eq!(
        crate::ampr_command_buffer_record_read(
            [
                command_buffer,
                0,
                0,
                file_id.into(),
                second.as_mut_ptr().addr() as u64,
                4,
            ],
            [4, 0, 0, 0, 0, 0],
            &mut state,
        ),
        0
    );
    let dispatch = RwLock::new(state);
    let order = Mutex::new(());
    let mut observed_offsets = Vec::new();
    assert_eq!(
        crate::ampr_submit_command_buffer_two_phase_with(
            &order,
            &dispatch,
            [command_buffer, 0, 0, 0, 0, 0],
            |file, id, destination, size, offset, capacity| {
                let unrelated = dispatch
                    .try_write()
                    .expect("submit I/O retained the process dispatch lock");
                drop(unrelated);
                observed_offsets.push(offset);
                crate::ampr_read_file_to_guest(file, id, destination, size, offset, capacity)
            },
        ),
        0
    );
    assert_eq!(observed_offsets, [0, 4]);
    assert_eq!(&first, b"0123");
    assert_eq!(&second, b"4567");
    assert_eq!(
        u64::from_le_bytes(command_records[0x20..0x28].try_into().unwrap()),
        4
    );
    assert_eq!(
        u64::from_le_bytes(command_records[0x50..0x58].try_into().unwrap()),
        4
    );

    std::fs::write(&path, b"abcdefgh").unwrap();
    first.fill(0);
    second.fill(0);
    assert_eq!(
        crate::ampr_submit_command_buffer_two_phase_with(
            &order,
            &dispatch,
            [command_buffer, 0, 0, 0, 0, 0],
            crate::ampr_read_file_to_guest,
        ),
        0
    );
    assert_eq!(&first, b"abcd");
    assert_eq!(&second, b"efgh");
    let state = dispatch.read().unwrap();
    assert_eq!(state.next_apr_submission_id, 2);
    assert_eq!(
        state
            .ampr_command_buffers
            .get(&command_buffer)
            .unwrap()
            .write_offset,
        crate::AMPR_COMMAND_RECORD_SIZE * 2
    );
    drop(state);
    std::fs::remove_file(path).ok();
}

#[test]
fn ampr_ordering_classifier_covers_every_stateful_operation() {
    for nid in [
        "gEpBkcwxUjw",
        "WT-5NKy42fw",
        "eYAh2vlCY-U",
        "QzB4O+bJQyA",
        "i3HWvW35jao",
        "w5fcCG+t31g",
        "VB-BtuIW8Xc",
        "C+Khtbbx2g8",
        "WvEu7yl3Ivg",
        "ApkYaHb8Sek",
        "eE4Szl8sil8",
        "qvMUCyyaCSI",
        "ASoW5WE-UPo",
        "rqwFKI4PAiM",
        "bBfz7kMF2Ho",
        "8aI7R7WaOlc",
        "a8uLzYY--tM",
        "N-FSPA4S3nI",
        "baQO9ez2gL4",
        "ULvXMDz56po",
        "tZDDEo2tE5k",
        "GnxKOHEawhk",
        "H896Pt-yB4I",
        "GuchCTefuZw",
        "Qs1xtplKo0U",
        "mQ16-QdKv7k",
        "YPxkUDhgoNI",
        "tNn5WBkta60",
        "pFQ9UHpO52s",
        "GmOguNIsuKk",
        "4UkZbYKVF7c",
        "4quckD2y7Pg",
        "sWbST0oQKsc",
        "dXPaz65HNmk",
        "f12ObAMEi9A",
        "mv0O8Zg0woU",
        "o67gODLFpls",
        // Recording a completion word mutates the same command-buffer state, so it must not race
        // a submission of that buffer.
        "sJXyWHjP-F8",
    ] {
        assert!(
            crate::is_serialized_ampr_import(nid),
            "{nid} bypassed AMPR ordering"
        );
    }
    for nid in ["eE4Szl8sil8", "qvMUCyyaCSI", "ASoW5WE-UPo"] {
        assert!(
            crate::is_ordered_ampr_submit_import(nid),
            "{nid} bypassed deferred APR execution"
        );
    }
}

#[test]
fn ampr_submit_preserves_result_and_submission_id_outputs() {
    let command_buffer = 0x6800;
    let mut command_records = [0_u8; 0x30];
    let mut result = u64::MAX;
    let mut submission_id = u32::MAX;
    let mut state = ProbeDispatchState::default();
    state.ampr_command_buffers.insert(
        command_buffer,
        AmprCommandBufferState {
            buffer: command_records.as_mut_ptr().addr() as u64,
            size: command_records.len() as u64,
            ..AmprCommandBufferState::default()
        },
    );
    let dispatch = RwLock::new(state);
    let order = Mutex::new(());

    assert_eq!(
        crate::ampr_submit_command_buffer_two_phase_with(
            &order,
            &dispatch,
            [
                command_buffer,
                6,
                (&mut result as *mut u64).addr() as u64,
                (&mut submission_id as *mut u32).addr() as u64,
                0,
                0,
            ],
            |_, _, _, _, _, _| unreachable!("empty command buffer performed I/O"),
        ),
        0
    );
    assert_eq!(result, 0);
    assert_eq!(submission_id, 1);
}

#[test]
fn ampr_reset_and_destroy_wait_for_the_same_buffer_submission_order() {
    let command_buffer = 0x6a00;
    let mut command_records = [0_u8; 0x30];
    let state = Arc::new(RwLock::new(ProbeDispatchState::default()));
    state.write().unwrap().ampr_command_buffers.insert(
        command_buffer,
        AmprCommandBufferState {
            buffer: command_records.as_mut_ptr().addr() as u64,
            size: command_records.len() as u64,
            command_count: 1,
            write_offset: crate::AMPR_COMMAND_RECORD_SIZE,
            ..AmprCommandBufferState::default()
        },
    );

    for destroy in [false, true] {
        if !state
            .read()
            .unwrap()
            .ampr_command_buffers
            .contains_key(&command_buffer)
        {
            state.write().unwrap().ampr_command_buffers.insert(
                command_buffer,
                AmprCommandBufferState {
                    buffer: command_records.as_mut_ptr().addr() as u64,
                    size: command_records.len() as u64,
                    command_count: 1,
                    write_offset: crate::AMPR_COMMAND_RECORD_SIZE,
                    ..AmprCommandBufferState::default()
                },
            );
        }
        let order = crate::ampr_command_order(command_buffer);
        let guard = order
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let worker_state = Arc::clone(&state);
        let (started_tx, started_rx) = mpsc::channel();
        let (finished_tx, finished_rx) = mpsc::channel();
        let worker = std::thread::spawn(move || {
            started_tx.send(()).unwrap();
            let _ordered = crate::ampr_command_order(command_buffer)
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            let result = if destroy {
                crate::ampr_command_buffer_destructor(
                    [command_buffer, 0, 0, 0, 0, 0],
                    &mut worker_state.write().unwrap(),
                )
            } else {
                crate::ampr_command_buffer_reset(
                    [command_buffer, 0, 0, 0, 0, 0],
                    &mut worker_state.write().unwrap(),
                )
            };
            finished_tx.send(result).unwrap();
        });
        started_rx.recv().unwrap();
        assert!(
            finished_rx.recv_timeout(Duration::from_millis(20)).is_err(),
            "buffer mutation raced the active submission order"
        );
        drop(guard);
        assert_eq!(finished_rx.recv_timeout(Duration::from_secs(1)).unwrap(), 0);
        worker.join().unwrap();

        let state = state.read().unwrap();
        if destroy {
            assert!(!state.ampr_command_buffers.contains_key(&command_buffer));
        } else {
            let command = state.ampr_command_buffers.get(&command_buffer).unwrap();
            assert_eq!(command.write_offset, 0);
            assert_eq!(command.command_count, 0);
            assert!(command.pending_reads.is_empty());
            assert!(command.pending_completions.is_empty());
            assert!(command.pending_address_writes.is_empty());
        }
    }
}

/// Registering interest must not announce a completion: the guest's listener resolves the
/// event against state the submission creates, so an early wake dereferences a context that
/// does not exist.
#[test]
fn ampr_completions_are_announced_by_submission_not_registration() {
    let mut state = ProbeDispatchState::default();
    let equeue = 0x4000;
    state.kernel_equeues.insert(equeue);
    assert_eq!(
        kernel_add_ampr_event([equeue, 0x7502, 0xabc, 0, 0, 0], &mut state),
        0
    );
    assert!(
        state.pending_events.is_empty(),
        "registration announced a completion"
    );

    let mut buffer = [0_u8; 0x100];
    let command_buffer = 0x5000_u64;
    state.ampr_command_buffers.insert(
        command_buffer,
        AmprCommandBufferState {
            buffer: buffer.as_mut_ptr().addr() as u64,
            size: buffer.len() as u64,
            ..AmprCommandBufferState::default()
        },
    );
    assert_eq!(
        ampr_command_buffer_write_kernel_event_queue(
            [command_buffer, equeue, 0x7502, 0xabc, 0, 0],
            &mut state,
        ),
        0
    );
    assert!(
        state.pending_events.is_empty(),
        "recording announced a completion before submission"
    );

    assert_eq!(
        ampr_submit_command_buffer([command_buffer, 0, 0, 0, 0, 0], &mut state),
        0
    );
    assert_eq!(state.pending_events.len(), 1);
    let event = state.pending_events.front().unwrap();
    assert_eq!(event.equeue, equeue);
    assert_eq!(event.ident, 0x7502);
    assert_eq!(event.user_data, 0xabc);
    state.pending_events.pop_front();

    assert_eq!(
        ampr_submit_command_buffer([command_buffer, 0, 0, 0, 0, 0], &mut state),
        0
    );
    assert_eq!(
        state.pending_events.len(),
        1,
        "reusing the command buffer lost its completion command"
    );
}

/// The completion word a guest polls must be written by the submission that does the work, not
/// when the command is recorded, and must be repeated on every reuse of the program.
#[test]
fn ampr_address_completions_are_written_by_submission_and_repeat_on_reuse() {
    let mut state = ProbeDispatchState::default();
    let mut buffer = [0_u8; 0x100];
    let command_buffer = 0x5100_u64;
    state.ampr_command_buffers.insert(
        command_buffer,
        AmprCommandBufferState {
            buffer: buffer.as_mut_ptr().addr() as u64,
            size: buffer.len() as u64,
            ..AmprCommandBufferState::default()
        },
    );

    let mut flag = 0_u64;
    let flag_address = (&raw mut flag).addr() as u64;
    assert_eq!(
        ampr_command_buffer_write_address_on_completion(
            [command_buffer, flag_address, 0xa5, 0, 0, 0],
            &mut state,
        ),
        0
    );
    assert_eq!(
        flag, 0,
        "recording the command must not write the completion word"
    );

    assert_eq!(
        ampr_submit_command_buffer([command_buffer, 0, 0, 0, 0, 0], &mut state),
        0
    );
    assert_eq!(flag, 0xa5, "submission must write the recorded value");

    // A command buffer is a reusable program, so a second submission repeats the write.
    flag = 0;
    assert_eq!(
        ampr_submit_command_buffer([command_buffer, 0, 0, 0, 0, 0], &mut state),
        0
    );
    assert_eq!(flag, 0xa5, "reusing the program lost its completion write");

    // Resetting the program clears the recorded write along with everything else.
    assert_eq!(
        crate::ampr_command_buffer_reset([command_buffer, 0, 0, 0, 0, 0], &mut state),
        0
    );
    flag = 0;
    assert_eq!(
        ampr_submit_command_buffer([command_buffer, 0, 0, 0, 0, 0], &mut state),
        0
    );
    assert_eq!(flag, 0, "a reset program must not still write its old word");
}

#[test]
fn ampr_address_completion_rejects_a_missing_buffer_or_address() {
    let mut state = ProbeDispatchState::default();
    let mut flag = 0_u64;
    let flag_address = (&raw mut flag).addr() as u64;
    assert_eq!(
        ampr_command_buffer_write_address_on_completion([0, flag_address, 1, 0, 0, 0], &mut state),
        sce_kernel_error(crate::SCE_KERNEL_ERROR_INVALID_ARGUMENT)
    );
    assert_eq!(
        ampr_command_buffer_write_address_on_completion(
            [0x5200, flag_address, 1, 0, 0, 0],
            &mut state
        ),
        sce_kernel_error(crate::SCE_KERNEL_ERROR_INVALID_ARGUMENT),
        "an unknown command buffer is not recordable"
    );
    let mut buffer = [0_u8; 0x100];
    state.ampr_command_buffers.insert(
        0x5200,
        AmprCommandBufferState {
            buffer: buffer.as_mut_ptr().addr() as u64,
            size: buffer.len() as u64,
            ..AmprCommandBufferState::default()
        },
    );
    // A null destination would make the completion write to address zero on every submission.
    assert_eq!(
        ampr_command_buffer_write_address_on_completion([0x5200, 0, 1, 0, 0, 0], &mut state),
        sce_kernel_error(crate::SCE_KERNEL_ERROR_INVALID_ARGUMENT)
    );
}

fn ampr_record_result(records: &[u8], record_offset: usize) -> u64 {
    let result = record_offset + 0x20;
    u64::from_le_bytes(records[result..result + 8].try_into().unwrap())
}

const AMPR_TEST_COMPLETION_VALUE: u64 = 0xa5;
const AMPR_TEST_EVENT_IDENT: u64 = 0x7601;
const AMPR_TEST_EVENT_DATA: u64 = 0xdef;

/// Records one 4-byte read per `(file id, destination)`, each continuing where the previous one
/// left off in the file, followed by the completion word and, given an equeue, the completion
/// event.
fn ampr_record_read_program(
    command_buffer: u64,
    reads: &[(u32, u64)],
    completion_word: u64,
    equeue: Option<u64>,
    records: &mut [u8],
    state: &mut ProbeDispatchState,
) {
    state.ampr_command_buffers.insert(
        command_buffer,
        AmprCommandBufferState {
            buffer: records.as_mut_ptr().addr() as u64,
            size: records.len() as u64,
            ..AmprCommandBufferState::default()
        },
    );
    for (index, (file_id, destination)) in reads.iter().enumerate() {
        assert_eq!(
            crate::ampr_command_buffer_record_read(
                [command_buffer, 0, 0, u64::from(*file_id), *destination, 4],
                [index as u64 * 4, 0, 0, 0, 0, 0],
                state,
            ),
            0
        );
    }
    assert_eq!(
        ampr_command_buffer_write_address_on_completion(
            [
                command_buffer,
                completion_word,
                AMPR_TEST_COMPLETION_VALUE,
                0,
                0,
                0,
            ],
            state,
        ),
        0
    );
    if let Some(equeue) = equeue {
        assert_eq!(
            ampr_command_buffer_write_kernel_event_queue(
                [
                    command_buffer,
                    equeue,
                    AMPR_TEST_EVENT_IDENT,
                    AMPR_TEST_EVENT_DATA,
                    0,
                    0,
                ],
                state,
            ),
            0
        );
    }
}

/// A failed transfer is contained to its own record: the records after it still run, its result
/// word reports zero bytes rather than the previous submission's count, and the completion the
/// guest polls or waits on is still published — carrying a failure, because a submission that
/// publishes nothing leaves the guest waiting on an error it can never observe.
#[test]
fn ampr_submit_contains_a_failed_read_and_still_publishes_its_completion() {
    let path = create_ampr_test_file(b"0123456789ab");
    let file_id = 0x1357_2468;
    let command_buffer = 0x6c00;
    let equeue = 0x4100;
    let mut command_records = [0_u8; 0x180];
    let mut first = [0_u8; 4];
    let mut second = [0_u8; 4];
    let mut third = [0_u8; 4];
    let mut flag = 0_u64;
    let mut state = ProbeDispatchState::default();
    state.kernel_equeues.insert(equeue);
    state.ampr_files.insert(file_id, path.clone());
    ampr_record_read_program(
        command_buffer,
        &[
            (file_id, first.as_mut_ptr().addr() as u64),
            (file_id, second.as_mut_ptr().addr() as u64),
            (file_id, third.as_mut_ptr().addr() as u64),
        ],
        (&raw mut flag).addr() as u64,
        Some(equeue),
        &mut command_records,
        &mut state,
    );

    let dispatch = RwLock::new(state);
    let order = Mutex::new(());
    assert_eq!(
        crate::ampr_submit_command_buffer_two_phase_with(
            &order,
            &dispatch,
            [command_buffer, 0, 0, 0, 0, 0],
            crate::ampr_read_file_to_guest,
        ),
        0
    );
    assert_eq!(ampr_record_result(&command_records, 0x30), 4);
    assert_eq!(flag, AMPR_TEST_COMPLETION_VALUE);

    // The same program over new contents, with the middle transfer failing. A reusable program's
    // records and destinations still hold what the previous submission left there.
    std::fs::write(&path, b"ABCDEFGHIJKL").unwrap();
    flag = 0;
    dispatch.write().unwrap().pending_events.clear();
    let mut submission_result = u64::MAX;
    let mut submission_id = 0_u32;
    let mut observed_offsets = Vec::new();
    let read_error = sce_kernel_error(crate::SCE_KERNEL_ERROR_IO);
    assert_eq!(
        crate::ampr_submit_command_buffer_two_phase_with(
            &order,
            &dispatch,
            [
                command_buffer,
                0,
                (&raw mut submission_result).addr() as u64,
                (&raw mut submission_id).addr() as u64,
                0,
                0,
            ],
            |file, id, destination, size, offset, capacity| {
                observed_offsets.push(offset);
                if offset == 4 {
                    return Err(read_error);
                }
                crate::ampr_read_file_to_guest(file, id, destination, size, offset, capacity)
            },
        ),
        read_error
    );

    assert_eq!(
        observed_offsets,
        [0, 4, 8],
        "a failed read cancelled the rest of the program"
    );
    assert_eq!(&first, b"ABCD");
    assert_eq!(&second, b"4567", "the failed read wrote its destination");
    assert_eq!(&third, b"IJKL", "a read after a failure did not run");
    assert_eq!(ampr_record_result(&command_records, 0), 4);
    assert_eq!(
        ampr_record_result(&command_records, 0x30),
        0,
        "the failed record kept the previous submission's byte count"
    );
    assert_eq!(ampr_record_result(&command_records, 0x60), 4);
    assert_eq!(flag, u64::MAX, "the completion word did not report failure");
    assert_eq!(submission_result, read_error);
    assert_eq!(submission_id, 2);
    let state = dispatch.read().unwrap();
    let event = state
        .pending_events
        .front()
        .expect("the failed submission announced no completion event");
    assert_eq!(event.equeue, equeue);
    assert_eq!(
        event.data, AMPR_TEST_EVENT_DATA,
        "the event lost its request identity"
    );
    assert_eq!(event.flags, 0x4000, "the event did not report the failure");
    drop(state);
    std::fs::remove_file(path).ok();
}

/// The exhausted-descriptor shape: one file of a program cannot be opened. The records that read
/// other files still run, and the program still completes.
#[test]
fn ampr_submit_reports_an_unopenable_read_without_cancelling_the_program() {
    let readable = create_ampr_test_file(b"abcdefghijkl");
    let unopenable = create_ampr_test_file(b"unreadable");
    let readable_id = 0x2222_1111;
    let unopenable_id = 0x3333_4444;
    let command_buffer = 0x6e00;
    let mut command_records = [0_u8; 0x120];
    let mut first = [0_u8; 4];
    let mut second = [0_u8; 4];
    let mut third = [0_u8; 4];
    let mut flag = 0_u64;
    let mut state = ProbeDispatchState::default();
    state.ampr_files.insert(readable_id, readable.clone());
    state.ampr_files.insert(unopenable_id, unopenable.clone());
    ampr_record_read_program(
        command_buffer,
        &[
            (readable_id, first.as_mut_ptr().addr() as u64),
            (unopenable_id, second.as_mut_ptr().addr() as u64),
            (readable_id, third.as_mut_ptr().addr() as u64),
        ],
        (&raw mut flag).addr() as u64,
        None,
        &mut command_records,
        &mut state,
    );
    std::fs::remove_file(&unopenable).unwrap();

    let dispatch = RwLock::new(state);
    let order = Mutex::new(());
    assert_eq!(
        crate::ampr_submit_command_buffer_two_phase_with(
            &order,
            &dispatch,
            [command_buffer, 0, 0, 0, 0, 0],
            crate::ampr_read_file_to_guest,
        ),
        sce_kernel_error(SCE_KERNEL_ERROR_NOT_FOUND)
    );

    assert_eq!(&first, b"abcd");
    assert_eq!(second, [0; 4], "the unopenable read wrote its destination");
    assert_eq!(&third, b"ijkl", "a read after an unopenable file failed");
    assert_eq!(ampr_record_result(&command_records, 0), 4);
    assert_eq!(ampr_record_result(&command_records, 0x30), 0);
    assert_eq!(ampr_record_result(&command_records, 0x60), 4);
    assert_eq!(flag, u64::MAX, "the completion word did not report failure");
    assert!(
        !dispatch
            .read()
            .unwrap()
            .ampr_file_handles
            .contains_key(&unopenable_id)
    );
    std::fs::remove_file(readable).ok();
}
