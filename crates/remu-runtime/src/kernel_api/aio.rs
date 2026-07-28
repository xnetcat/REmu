// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

use crate::{
    SCE_KERNEL_ERROR_BAD_FILE_DESCRIPTOR, SCE_KERNEL_ERROR_INVALID_ARGUMENT,
    SCE_KERNEL_ERROR_MEMORY_FAULT, SCE_KERNEL_ERROR_NOT_FOUND, low_i32, low_u32, read_guest_u32,
    read_guest_u64, sce_kernel_error, write_guest_u32, write_guest_u64,
};
use std::collections::BTreeMap;
use std::sync::{Mutex, MutexGuard, OnceLock};

use super::{guest_range_is_readable, kernel_files, kernel_positional_transfer};

/// Bytes one asynchronous read or write request occupies.
pub(super) const AIO_REQUEST_SIZE: u64 = 0x28;

pub(super) const AIO_REQUEST_LENGTH_OFFSET: u64 = 0x08;

pub(super) const AIO_REQUEST_BUFFER_OFFSET: u64 = 0x10;

pub(super) const AIO_REQUEST_RESULT_OFFSET: u64 = 0x18;

pub(super) const AIO_REQUEST_FD_OFFSET: u64 = 0x20;

/// Byte offset of a request result's state word; the transferred count precedes it.
pub(super) const AIO_RESULT_STATE_OFFSET: u64 = 0x08;

/// Terminal state of a request whose transfer has finished.
pub(super) const AIO_STATE_COMPLETED: u32 = 3;

/// Requests one submission may carry.
pub(super) const AIO_REQUEST_LIMIT: i32 = 128;

/// Lowest and highest scheduling priorities a submission may name.
pub(super) const AIO_PRIORITY_LOW: i32 = 1;

pub(super) const AIO_PRIORITY_HIGH: i32 = 3;

/// Outcome of one completed asynchronous submission.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct AioSubmission {
    state: u32,
    result: i32,
}

/// Submissions the guest has not deleted yet, keyed by the identifier it was given.
#[derive(Debug)]
pub(super) struct AioState {
    next_id: i32,
    submissions: BTreeMap<i32, AioSubmission>,
}

impl Default for AioState {
    fn default() -> Self {
        Self {
            next_id: 1,
            submissions: BTreeMap::new(),
        }
    }
}

pub(super) static AIO_SUBMISSIONS: OnceLock<Mutex<AioState>> = OnceLock::new();

/// Performs one batch of positional transfers and records its terminal state.
///
/// The transfers run before the call returns, so a submission is already complete by the time the
/// guest first asks about it. That is a legal schedule for the ABI: every later query reports the
/// terminal state, and the guest's buffers hold their data as soon as it can observe them.
pub(super) fn kernel_aio_submit(arguments: [u64; 6], write: bool) -> u64 {
    let (requests, identifier) = (arguments[0], arguments[3]);
    let count = low_i32(arguments[1]);
    let priority = low_i32(arguments[2]);
    if requests == 0 || identifier == 0 {
        return sce_kernel_error(SCE_KERNEL_ERROR_MEMORY_FAULT);
    }
    if count <= 0 || count > AIO_REQUEST_LIMIT {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    }
    if !(AIO_PRIORITY_LOW..=AIO_PRIORITY_HIGH).contains(&priority) {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    }
    let count = u64::from(count.cast_unsigned());
    let Ok(span) = usize::try_from(count * AIO_REQUEST_SIZE) else {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    };
    if !guest_range_is_readable(requests, span) {
        return sce_kernel_error(SCE_KERNEL_ERROR_MEMORY_FAULT);
    }
    // A batch naming an unusable descriptor is rejected whole, so no request in it is half applied.
    let files = kernel_files();
    for index in 0..count {
        let request = requests + index * AIO_REQUEST_SIZE;
        if !files
            .files
            .contains_key(&read_guest_u32(request + AIO_REQUEST_FD_OFFSET).cast_signed())
        {
            return sce_kernel_error(SCE_KERNEL_ERROR_BAD_FILE_DESCRIPTOR);
        }
    }
    drop(files);
    let mut outcome = 0;
    for index in 0..count {
        let request = requests + index * AIO_REQUEST_SIZE;
        let transfer = [
            u64::from(read_guest_u32(request + AIO_REQUEST_FD_OFFSET)),
            read_guest_u64(request + AIO_REQUEST_BUFFER_OFFSET),
            read_guest_u64(request + AIO_REQUEST_LENGTH_OFFSET),
            read_guest_u64(request),
            0,
            0,
        ];
        let transferred = kernel_positional_transfer(transfer, write);
        let result = read_guest_u64(request + AIO_REQUEST_RESULT_OFFSET);
        if result != 0 {
            write_guest_u64(result, transferred);
            write_guest_u32(result + AIO_RESULT_STATE_OFFSET, AIO_STATE_COMPLETED);
        }
        if low_u32(transferred) & 0x8000_0000 != 0 {
            outcome = low_i32(transferred);
        }
    }
    let mut aio = aio_submissions();
    let id = aio.next_id;
    aio.next_id = aio.next_id.checked_add(1).unwrap_or(1);
    aio.submissions.insert(
        id,
        AioSubmission {
            state: AIO_STATE_COMPLETED,
            result: outcome,
        },
    );
    drop(aio);
    write_guest_u32(identifier, id.cast_unsigned());
    0
}

/// Borrows the table of submissions the guest has not deleted yet.
pub(super) fn aio_submissions() -> MutexGuard<'static, AioState> {
    AIO_SUBMISSIONS
        .get_or_init(|| Mutex::new(AioState::default()))
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// Reports the state of one submission without waiting.
pub(super) fn kernel_aio_poll_request(arguments: [u64; 6]) -> u64 {
    let output = arguments[1];
    if output == 0 {
        return sce_kernel_error(SCE_KERNEL_ERROR_MEMORY_FAULT);
    }
    let aio = aio_submissions();
    let Some(submission) = aio.submissions.get(&low_i32(arguments[0])) else {
        return sce_kernel_error(SCE_KERNEL_ERROR_NOT_FOUND);
    };
    let state = submission.state;
    drop(aio);
    write_guest_u32(output, state);
    0
}

/// Reports the state of several submissions without waiting.
pub(super) fn kernel_aio_poll_requests(arguments: [u64; 6]) -> u64 {
    let (identifiers, states) = (arguments[0], arguments[2]);
    let count = low_i32(arguments[1]);
    if identifiers == 0 || states == 0 {
        return sce_kernel_error(SCE_KERNEL_ERROR_MEMORY_FAULT);
    }
    if count <= 0 || count > AIO_REQUEST_LIMIT {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    }
    let count = u64::from(count.cast_unsigned());
    let Ok(span) = usize::try_from(count * 4) else {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    };
    if !guest_range_is_readable(identifiers, span) {
        return sce_kernel_error(SCE_KERNEL_ERROR_MEMORY_FAULT);
    }
    let aio = aio_submissions();
    let mut resolved = Vec::with_capacity(span / 4);
    for index in 0..count {
        let id = read_guest_u32(identifiers + index * 4).cast_signed();
        let Some(submission) = aio.submissions.get(&id) else {
            return sce_kernel_error(SCE_KERNEL_ERROR_NOT_FOUND);
        };
        resolved.push(submission.state);
    }
    drop(aio);
    for (index, state) in resolved.into_iter().enumerate() {
        let offset = u64::try_from(index * 4).expect("the request index fits u64");
        write_guest_u32(states + offset, state);
    }
    0
}

/// Reports the terminal state of one submission.
///
/// Every submission has already finished when it is created, so no time is consumed and the
/// caller's remaining-timeout cell is left as it was.
pub(super) fn kernel_aio_wait_request(arguments: [u64; 6]) -> u64 {
    kernel_aio_poll_request(arguments)
}

/// Discards the bookkeeping for one submission and reports its outcome.
pub(super) fn kernel_aio_delete_request(arguments: [u64; 6]) -> u64 {
    let mut aio = aio_submissions();
    let Some(submission) = aio.submissions.remove(&low_i32(arguments[0])) else {
        return sce_kernel_error(SCE_KERNEL_ERROR_NOT_FOUND);
    };
    drop(aio);
    if arguments[1] != 0 {
        write_guest_u32(arguments[1], submission.result.cast_unsigned());
    }
    0
}
