// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

use std::sync::{Arc, Mutex, RwLock, mpsc};
use std::time::Duration;

use super::{
    AmprCommandBufferState, ProbeDispatchState, SCE_KERNEL_ERROR_NOT_FOUND,
    ampr_command_buffer_write_kernel_event_queue, ampr_submit_command_buffer,
    create_ampr_test_file, kernel_add_ampr_event, sce_kernel_error,
};

#[test]
fn ampr_read_releases_process_dispatch_lock_while_copying_and_records_result() {
    let path = create_ampr_test_file(b"host payload");
    let file_id = 0x1234_5678;
    let command_buffer = 0x5000;
    let mut command_records = [0_u8; 0x60];
    let mut destination = [0_u8; 4];
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
    let dispatch = RwLock::new(state);
    let order = Mutex::new(());
    let destination_address = destination.as_mut_ptr().addr() as u64;

    assert_eq!(
        crate::ampr_command_buffer_read_file_two_phase_with(
            &order,
            &dispatch,
            [command_buffer, 0, 0, file_id.into(), destination_address, 4],
            [7, 0, 0, 0, 0, 0],
            |_, observed_file_id, observed_destination, size, offset, capacity| {
                let unrelated_state = dispatch
                    .try_write()
                    .expect("process dispatch lock must be released during AMPR I/O");
                drop(unrelated_state);
                assert_eq!(observed_file_id, file_id);
                assert_eq!(observed_destination, destination_address);
                assert_eq!((size, offset, capacity), (4, 7, 4));
                // SAFETY: `destination` remains live for the complete synchronous call.
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        b"data".as_ptr(),
                        observed_destination as *mut u8,
                        4,
                    );
                }
                Ok(4)
            },
        ),
        0
    );

    assert_eq!(&destination, b"data");
    let state = dispatch.read().unwrap();
    let command = state.ampr_command_buffers.get(&command_buffer).unwrap();
    assert_eq!(command.write_offset, crate::AMPR_COMMAND_RECORD_SIZE);
    assert_eq!(command.command_count, 1);
    assert!(state.ampr_file_handles.contains_key(&file_id));
    drop(state);
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
        4
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
fn failed_ampr_read_keeps_command_buffer_unmodified() {
    let path = create_ampr_test_file(b"host payload");
    let file_id = 0x2468_1357;
    let command_buffer = 0x5500;
    let mut command_records = [0xa5_u8; 0x60];
    let mut destination = [0_u8; 4];
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
    let dispatch = RwLock::new(state);
    let error = sce_kernel_error(SCE_KERNEL_ERROR_NOT_FOUND);

    assert_eq!(
        crate::ampr_command_buffer_read_file_two_phase_with(
            &Mutex::new(()),
            &dispatch,
            [
                command_buffer,
                0,
                0,
                file_id.into(),
                destination.as_mut_ptr().addr() as u64,
                destination.len() as u64,
            ],
            [0; 6],
            |_, _, _, _, _, _| Err(error),
        ),
        error
    );

    let state = dispatch.read().unwrap();
    let command = state.ampr_command_buffers.get(&command_buffer).unwrap();
    assert_eq!(command.write_offset, 0);
    assert_eq!(command.command_count, 0);
    assert!(state.ampr_file_handles.contains_key(&file_id));
    assert_eq!(command_records, [0xa5; 0x60]);
    drop(state);
    std::fs::remove_file(path).ok();
}

#[test]
fn ampr_submit_cannot_overtake_an_in_flight_read() {
    let path = create_ampr_test_file(b"host payload");
    let file_id = 0x8765_4321;
    let command_buffer = 0x6000;
    let mut command_records = [0_u8; 0x60];
    let mut destination = [0_u8; 4];
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
    let dispatch = Arc::new(RwLock::new(state));
    let order = Arc::new(Mutex::new(()));
    let destination_address = destination.as_mut_ptr().addr() as u64;
    let (read_entered_tx, read_entered_rx) = mpsc::channel();
    let (release_read_tx, release_read_rx) = mpsc::channel();
    let read_dispatch = Arc::clone(&dispatch);
    let read_order = Arc::clone(&order);
    let read = std::thread::spawn(move || {
        crate::ampr_command_buffer_read_file_two_phase_with(
            &read_order,
            &read_dispatch,
            [command_buffer, 0, 0, file_id.into(), destination_address, 4],
            [0; 6],
            move |_, _, observed_destination, _, _, _| {
                read_entered_tx.send(()).unwrap();
                release_read_rx.recv().unwrap();
                // SAFETY: the owning test thread retains this destination until both workers
                // join, and the AMPR ordering lock prevents submission from observing it early.
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        b"done".as_ptr(),
                        observed_destination as *mut u8,
                        4,
                    );
                }
                Ok(4)
            },
        )
    });
    read_entered_rx
        .recv_timeout(Duration::from_secs(1))
        .expect("read reached unlocked I/O phase");

    let submit_dispatch = Arc::clone(&dispatch);
    let submit_order = Arc::clone(&order);
    let (submit_started_tx, submit_started_rx) = mpsc::channel();
    let (submit_done_tx, submit_done_rx) = mpsc::channel();
    let submit = std::thread::spawn(move || {
        submit_started_tx.send(()).unwrap();
        let _order = submit_order.lock().unwrap();
        let mut state = submit_dispatch.write().unwrap();
        let result = ampr_submit_command_buffer([command_buffer, 0, 0, 0, 0, 0], &mut state);
        submit_done_tx.send(result).unwrap();
    });
    submit_started_rx.recv().unwrap();
    assert!(
        submit_done_rx
            .recv_timeout(Duration::from_millis(50))
            .is_err(),
        "submission overtook the blocked file read"
    );

    release_read_tx.send(()).unwrap();
    assert_eq!(read.join().unwrap(), 0);
    assert_eq!(
        submit_done_rx.recv_timeout(Duration::from_secs(1)).unwrap(),
        0
    );
    submit.join().unwrap();
    assert_eq!(&destination, b"done");
    let state = dispatch.read().unwrap();
    assert_eq!(state.next_apr_submission_id, 1);
    assert_eq!(
        state
            .ampr_command_buffers
            .get(&command_buffer)
            .unwrap()
            .write_offset,
        crate::AMPR_COMMAND_RECORD_SIZE
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
    ] {
        assert!(
            crate::is_serialized_ampr_import(nid),
            "{nid} bypassed AMPR ordering"
        );
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
}
