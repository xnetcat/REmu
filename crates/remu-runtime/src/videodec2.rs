// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

//! `libSceVideodec2` compute-queue bring-up entry points.
//!
//! Titles size a shared CPU/GPU workspace, allocate a compute queue from it, and later attach that
//! queue to a decoder. REmu does not yet decode through Videodec2; AvPlayer remains the media path.
//! The compute-queue calls still have to honour the ABI so a title that only needs the queue handle
//! (or that probes memory requirements before falling back) does not proceed with uninitialized
//! memory descriptors.
//!
//! Struct layouts and error codes were triangulated against the public Orbis Videodec2 surface and
//! cross-checked with shadPS4's GPL-2.0-or-later `videodec2` headers (snapshot referenced in
//! `docs/provenance.md`). The offline behaviour is REmu's: query reports a fixed workspace size,
//! allocate returns the caller-supplied workspace pointer as the queue opaque, and release accepts
//! any previously allocated queue.

use crate::{
    ProbeDispatchState, process_guest_range_is_readable, read_guest_u64, sce_kernel_error,
    write_guest_u64,
};

fn low_u8(value: u64) -> u8 {
    value as u8
}

fn low_u16(value: u64) -> u16 {
    value as u16
}

const VIDEODEC2_ERROR_STRUCT_SIZE: u32 = 0x811d_0101;
const VIDEODEC2_ERROR_ARGUMENT_POINTER: u32 = 0x811d_0102;
const VIDEODEC2_ERROR_MEMORY_POINTER: u32 = 0x811d_0105;
const VIDEODEC2_ERROR_COMPUTE_QUEUE: u32 = 0x811d_0110;
const VIDEODEC2_ERROR_CONFIG_INFO: u32 = 0x811d_0200;
const VIDEODEC2_ERROR_COMPUTE_PIPE_ID: u32 = 0x811d_0201;
const VIDEODEC2_ERROR_COMPUTE_QUEUE_ID: u32 = 0x811d_0202;

/// `sizeof(OrbisVideodec2ComputeMemoryInfo)`.
const COMPUTE_MEMORY_INFO_SIZE: u64 = 0x18;
/// `sizeof(OrbisVideodec2ComputeConfigInfo)`.
const COMPUTE_CONFIG_INFO_SIZE: u64 = 0x10;
/// Workspace size reported when a title has not yet attached decoder configuration.
///
/// Real consoles size this from the compute pipe; offline bring-up only needs a non-zero, aligned
/// footprint the guest can allocate before `AllocateComputeQueue`.
const MINIMUM_COMPUTE_MEMORY_SIZE: u64 = 16 * 1024 * 1024;

/// Handles a Videodec2 NID, or returns `None` so the caller can keep looking.
pub(crate) fn dispatch(
    nid: &str,
    arguments: [u64; 6],
    state: &mut ProbeDispatchState,
) -> Option<u64> {
    Some(match nid {
        "RnDibcGCPKw" => query_compute_memory_info(arguments[0]),
        "eD+X2SmxUt4" => allocate_compute_queue(arguments, state),
        "UvtA3FAiF4Y" => release_compute_queue(arguments[0], state),
        _ => return None,
    })
}

/// `sceVideodec2QueryComputeMemoryInfo(OrbisVideodec2ComputeMemoryInfo*)`.
fn query_compute_memory_info(info: u64) -> u64 {
    if info == 0 || !process_guest_range_is_readable(info, COMPUTE_MEMORY_INFO_SIZE as usize) {
        return sce_kernel_error(VIDEODEC2_ERROR_ARGUMENT_POINTER);
    }
    let this_size = read_guest_u64(info);
    if this_size != COMPUTE_MEMORY_INFO_SIZE {
        return sce_kernel_error(VIDEODEC2_ERROR_STRUCT_SIZE);
    }
    write_guest_u64(info + 0x08, MINIMUM_COMPUTE_MEMORY_SIZE);
    write_guest_u64(info + 0x10, 0);
    0
}

/// `sceVideodec2AllocateComputeQueue(config*, memInfo*, queueOut*)`.
fn allocate_compute_queue(arguments: [u64; 6], state: &mut ProbeDispatchState) -> u64 {
    let (config, mem_info, queue_out) = (arguments[0], arguments[1], arguments[2]);
    if config == 0
        || mem_info == 0
        || queue_out == 0
        || !process_guest_range_is_readable(config, COMPUTE_CONFIG_INFO_SIZE as usize)
        || !process_guest_range_is_readable(mem_info, COMPUTE_MEMORY_INFO_SIZE as usize)
        || !process_guest_range_is_readable(queue_out, 8)
    {
        return sce_kernel_error(VIDEODEC2_ERROR_ARGUMENT_POINTER);
    }
    if read_guest_u64(config) != COMPUTE_CONFIG_INFO_SIZE
        || read_guest_u64(mem_info) != COMPUTE_MEMORY_INFO_SIZE
    {
        return sce_kernel_error(VIDEODEC2_ERROR_STRUCT_SIZE);
    }
    // Layout: thisSize, computePipeId:u16, computeQueueId:u16, checkMemoryType:u8, reserved0:u8,
    // reserved1:u16. Non-zero reserved fields are rejected by the console ABI.
    let reserved0 = low_u8(read_guest_u64(config + 0x08) >> 40);
    let reserved1 = low_u16(read_guest_u64(config + 0x08) >> 48);
    if reserved0 != 0 || reserved1 != 0 {
        return sce_kernel_error(VIDEODEC2_ERROR_CONFIG_INFO);
    }
    let pipe_id = low_u16(read_guest_u64(config + 0x08));
    let queue_id = low_u16(read_guest_u64(config + 0x08) >> 16);
    if pipe_id > 4 {
        return sce_kernel_error(VIDEODEC2_ERROR_COMPUTE_PIPE_ID);
    }
    if queue_id > 7 {
        return sce_kernel_error(VIDEODEC2_ERROR_COMPUTE_QUEUE_ID);
    }
    let memory = read_guest_u64(mem_info + 0x10);
    let memory_size = read_guest_u64(mem_info + 0x08);
    if memory == 0
        || memory_size < MINIMUM_COMPUTE_MEMORY_SIZE
        || !process_guest_range_is_readable(memory, memory_size as usize)
    {
        return sce_kernel_error(VIDEODEC2_ERROR_MEMORY_POINTER);
    }
    // The console returns an opaque pointing into the caller-owned workspace.
    state.videodec2_compute_queues.insert(memory);
    write_guest_u64(queue_out, memory);
    eprintln!(
        "[REMu][VIDEODEC2] allocate compute queue pipe={pipe_id} queue={queue_id} memory={memory:#x} size={memory_size:#x}"
    );
    0
}

/// `sceVideodec2ReleaseComputeQueue(queue)`.
fn release_compute_queue(queue: u64, state: &mut ProbeDispatchState) -> u64 {
    if queue == 0 || !state.videodec2_compute_queues.remove(&queue) {
        return sce_kernel_error(VIDEODEC2_ERROR_COMPUTE_QUEUE);
    }
    eprintln!("[REMu][VIDEODEC2] release compute queue={queue:#x}");
    0
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ProbeDispatchState;

    #[test]
    fn query_reports_a_fixed_workspace_and_clears_the_pointer() {
        let mut info = [0_u64; 3];
        info[0] = COMPUTE_MEMORY_INFO_SIZE;
        info[1] = 0xdead;
        info[2] = 0xbeef;
        assert_eq!(
            query_compute_memory_info(info.as_mut_ptr().addr() as u64),
            0
        );
        assert_eq!(info[1], MINIMUM_COMPUTE_MEMORY_SIZE);
        assert_eq!(info[2], 0);
    }

    #[test]
    fn allocate_returns_the_workspace_pointer_as_the_queue() {
        let mut config = [0_u8; COMPUTE_CONFIG_INFO_SIZE as usize];
        config[..8].copy_from_slice(&COMPUTE_CONFIG_INFO_SIZE.to_le_bytes());
        // pipe=1, queue=2, checkMemoryType=0, reserved=0
        config[8] = 1;
        config[9] = 0;
        config[10] = 2;
        config[11] = 0;
        let mut workspace = vec![0_u8; MINIMUM_COMPUTE_MEMORY_SIZE as usize];
        let mut mem_info = [0_u64; 3];
        mem_info[0] = COMPUTE_MEMORY_INFO_SIZE;
        mem_info[1] = MINIMUM_COMPUTE_MEMORY_SIZE;
        mem_info[2] = workspace.as_mut_ptr().addr() as u64;
        let mut queue = 0_u64;
        let mut state = ProbeDispatchState::default();
        assert_eq!(
            allocate_compute_queue(
                [
                    config.as_ptr().addr() as u64,
                    mem_info.as_ptr().addr() as u64,
                    (&raw mut queue).addr() as u64,
                    0,
                    0,
                    0,
                ],
                &mut state,
            ),
            0
        );
        assert_eq!(queue, mem_info[2]);
        assert!(state.videodec2_compute_queues.contains(&queue));
        assert_eq!(release_compute_queue(queue, &mut state), 0);
        assert!(state.videodec2_compute_queues.is_empty());
    }

    #[test]
    fn query_rejects_a_mismatched_struct_size() {
        let mut info = [0_u64; 3];
        info[0] = 0x10;
        assert_eq!(
            query_compute_memory_info(info.as_mut_ptr().addr() as u64),
            sce_kernel_error(VIDEODEC2_ERROR_STRUCT_SIZE)
        );
    }

    #[test]
    fn dispatch_leaves_unrelated_nids_alone() {
        let mut state = ProbeDispatchState::default();
        assert_eq!(dispatch("not-a-videodec2-nid", [0; 6], &mut state), None);
    }
}
