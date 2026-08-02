use crate::{
    AGC_IT_DRAW_INDEX_OFFSET, AGC_IT_INDEX_BASE, AGC_IT_INDEX_BUFFER_SIZE, AGC_IT_INDEX_TYPE,
    AGC_IT_NOP, AGC_IT_NUM_INSTANCES, AGC_R_DRAW_INDEX_AUTO, AGC_R_FLIP, CompletedVideoOutFlip,
    KERNEL_EVENT_FILTER_GRAPHICS, KernelQueuedEvent, PendingVideoOutFlip, ProbeDispatchState,
    QUEUED_FLIPS, VIDEO_OUT_ERROR_INVALID_ADDRESS, VIDEO_OUT_ERROR_INVALID_HANDLE,
    VIDEO_OUT_ERROR_INVALID_INDEX, VideoOutBufferAttributes, VideoOutState,
    agc_allocate_command_dwords, agc_pm4, emit_agc_gpu_event, kernel_process_time, low_i32,
    low_u32, notify_kernel_equeue_waiters, publish_gpu_event, queue_kernel_event, read_guest_u32,
    read_guest_u64, sce_kernel_error, trace_logarithmic_sequence, write_guest_u32,
};

pub(crate) fn video_out_register_buffers2(
    arguments: [u64; 6],
    state: &mut ProbeDispatchState,
) -> u64 {
    let set_index = low_u32(arguments[1]);
    let buffer_index_start = usize::try_from(low_u32(arguments[2])).unwrap_or(usize::MAX);
    let entries = arguments[3];
    let buffer_count = usize::try_from(low_u32(arguments[4])).unwrap_or(usize::MAX);
    let attribute = arguments[5];
    if low_i32(arguments[0]) != 1 || !state.video_out.is_open {
        return sce_kernel_error(VIDEO_OUT_ERROR_INVALID_HANDLE);
    }
    if entries == 0 || attribute == 0 {
        return sce_kernel_error(VIDEO_OUT_ERROR_INVALID_ADDRESS);
    }
    if buffer_index_start > 15
        || buffer_count == 0
        || buffer_count > 16
        || buffer_index_start.saturating_add(buffer_count) > 16
    {
        return sce_kernel_error(VIDEO_OUT_ERROR_INVALID_INDEX);
    }

    state.video_out.pixel_format = read_guest_u64(attribute + 0x20);
    state.video_out.tiling_mode = read_guest_u32(attribute + 0x04);
    state.video_out.width = read_guest_u32(attribute + 0x0c);
    state.video_out.height = read_guest_u32(attribute + 0x10);
    state
        .video_out
        .display_buffers
        .resize(buffer_index_start + buffer_count, 0);
    state
        .video_out
        .display_buffer_attributes
        .resize(buffer_index_start + buffer_count, None);
    let attributes = VideoOutBufferAttributes {
        width: state.video_out.width,
        height: state.video_out.height,
        pitch: state.video_out.width,
        pixel_format: state.video_out.pixel_format,
        tiling_mode: state.video_out.tiling_mode,
    };
    for index in 0..buffer_count {
        let entry = entries + u64::try_from(index).unwrap_or(u64::MAX) * 0x20;
        state.video_out.display_buffers[buffer_index_start + index] = read_guest_u64(entry);
        state.video_out.display_buffer_attributes[buffer_index_start + index] = Some(attributes);
    }
    eprintln!(
        "[REMu][VIDEO] registered {} buffers at {}x{} format={:#x} tiling={} addresses={:x?}",
        buffer_count,
        state.video_out.width,
        state.video_out.height,
        state.video_out.pixel_format,
        state.video_out.tiling_mode,
        state.video_out.display_buffers
    );
    publish_gpu_event(remu_gpu::GpuEvent::DisplayRegistered {
        width: state.video_out.width,
        height: state.video_out.height,
        pitch: state.video_out.width,
        pixel_format: state.video_out.pixel_format,
        tiling_mode: state.video_out.tiling_mode,
        buffers: state.video_out.display_buffers.clone(),
    });
    u64::from(set_index)
}

pub(crate) fn video_out_add_flip_event(arguments: [u64; 6], state: &mut ProbeDispatchState) -> u64 {
    let equeue = arguments[0];
    let handle = low_i32(arguments[1]);
    if handle != 1 || !state.video_out.is_open {
        return sce_kernel_error(VIDEO_OUT_ERROR_INVALID_HANDLE);
    }
    if equeue == 0 {
        return sce_kernel_error(VIDEO_OUT_ERROR_INVALID_ADDRESS);
    }
    state.video_out.flip_events.insert(equeue, arguments[2]);
    0
}

pub(crate) fn video_out_add_vblank_event(
    arguments: [u64; 6],
    state: &mut ProbeDispatchState,
) -> u64 {
    let equeue = arguments[0];
    let handle = low_i32(arguments[1]);
    if handle != 1 || !state.video_out.is_open {
        return sce_kernel_error(VIDEO_OUT_ERROR_INVALID_HANDLE);
    }
    if equeue == 0 {
        return sce_kernel_error(VIDEO_OUT_ERROR_INVALID_ADDRESS);
    }
    // OrbisVideoOutInternalEventId::Vblank == 1. Flip uses ident 6; keep them distinct so
    // sceVideoOutGetEventId can classify the queue entry.
    state.video_out.vblank_events.insert(equeue, arguments[2]);
    0
}

pub(crate) fn agc_dcb_set_flip(arguments: [u64; 6]) -> u64 {
    let command_buffer = arguments[0];
    if command_buffer == 0 {
        return 0;
    }
    let Some(command) = agc_allocate_command_dwords(command_buffer, 6) else {
        return 0;
    };
    write_guest_u32(command, agc_pm4(6, AGC_IT_NOP, AGC_R_FLIP));
    write_guest_u32(command + 4, low_u32(arguments[1]));
    write_guest_u32(command + 8, low_u32(arguments[2]));
    write_guest_u32(command + 12, low_u32(arguments[3]));
    write_guest_u32(command + 16, low_u32(arguments[4]));
    write_guest_u32(command + 20, low_u32(arguments[4] >> 32));
    command
}

pub(crate) fn agc_dcb_set_index_buffer(arguments: [u64; 6]) -> u64 {
    let Some(command) = agc_allocate_command_dwords(arguments[0], 5) else {
        return 0;
    };
    // Gen5 takes the command buffer, index address, and element capacity. The encoded index
    // address is both the CPU-readable allocation and the GPU virtual address, so no process-wide
    // resource registry is needed to recover it during submission parsing.
    let index_buffer_address = arguments[1];
    let index_count = low_u32(arguments[2]);
    write_guest_u32(command, agc_pm4(3, AGC_IT_INDEX_BASE, 0));
    write_guest_u32(command + 4, low_u32(index_buffer_address));
    write_guest_u32(command + 8, low_u32(index_buffer_address >> 32));
    write_guest_u32(command + 12, agc_pm4(2, AGC_IT_INDEX_BUFFER_SIZE, 0));
    write_guest_u32(command + 16, index_count);
    command
}

pub(crate) fn agc_dcb_set_index_size(arguments: [u64; 6]) -> u64 {
    if low_u32(arguments[2]) != 0 {
        return 0;
    }
    let Some(command) = agc_allocate_command_dwords(arguments[0], 2) else {
        return 0;
    };
    write_guest_u32(command, agc_pm4(2, AGC_IT_INDEX_TYPE, 0));
    write_guest_u32(command + 4, low_u32(arguments[1]) & 0xff);
    command
}

pub(crate) fn agc_dcb_set_num_instances(arguments: [u64; 6]) -> u64 {
    let Some(command) = agc_allocate_command_dwords(arguments[0], 2) else {
        return 0;
    };
    write_guest_u32(command, agc_pm4(2, AGC_IT_NUM_INSTANCES, 0));
    write_guest_u32(command + 4, low_u32(arguments[1]));
    command
}

pub(crate) fn agc_dcb_draw_index_auto(arguments: [u64; 6]) -> u64 {
    if low_u32(arguments[2]) != 0x4000_0000 {
        return 0;
    }
    let Some(command) = agc_allocate_command_dwords(arguments[0], 7) else {
        return 0;
    };
    write_guest_u32(command, agc_pm4(7, AGC_IT_NOP, AGC_R_DRAW_INDEX_AUTO));
    write_guest_u32(command + 4, low_u32(arguments[1]));
    for index in 2..7_u64 {
        write_guest_u32(command + index * 4, 0);
    }
    command
}

pub(crate) fn agc_dcb_draw_index_offset(arguments: [u64; 6]) -> u64 {
    let Some(command) = agc_allocate_command_dwords(arguments[0], 5) else {
        return 0;
    };
    write_guest_u32(command, agc_pm4(5, AGC_IT_DRAW_INDEX_OFFSET, 0));
    write_guest_u32(command + 4, low_u32(arguments[2]));
    write_guest_u32(command + 8, low_u32(arguments[1]));
    write_guest_u32(command + 12, low_u32(arguments[2]));
    write_guest_u32(command + 16, low_u32(arguments[3]) & 0xe000_0001);
    command
}

pub(crate) fn queue_video_out_flip(
    state: &mut ProbeDispatchState,
    handle: u32,
    buffer_index: u32,
    flip_argument: u64,
) {
    let Some(&display_address) = state.video_out.display_buffers.get(buffer_index as usize) else {
        eprintln!("[REMu][VIDEO] ignored flip for unregistered buffer index {buffer_index}");
        return;
    };
    QUEUED_FLIPS.with(|queued| queued.set(queued.get().saturating_add(1)));
    state
        .pending_video_out_flips
        .push_back(PendingVideoOutFlip {
            completion: remu_gpu::GpuFlipCompletion {
                address: display_address,
                buffer_index,
                flip_argument,
            },
            handle,
            submit_process_time_counter: kernel_process_time(true),
        });
}

pub(crate) fn refresh_completed_video_out_flips(state: &mut ProbeDispatchState) {
    let completed = remu_gpu::take_gpu_completed_flips();
    if completed.is_empty() {
        return;
    }
    let mut queued = false;
    for completion in completed {
        let Some(index) = state
            .pending_video_out_flips
            .iter()
            .position(|pending| pending.completion == completion)
        else {
            eprintln!(
                "[REMu][VIDEO] ignored unmatched completed flip index={} address={:#x} argument={:#x}",
                completion.buffer_index, completion.address, completion.flip_argument
            );
            continue;
        };
        let pending = state
            .pending_video_out_flips
            .remove(index)
            .expect("completed flip index was just found");
        state.video_out.completed_flip_count =
            state.video_out.completed_flip_count.saturating_add(1);
        state.video_out.last_completed_flip = Some(CompletedVideoOutFlip {
            process_time: kernel_process_time(false),
            process_time_counter: kernel_process_time(true),
            flip_argument: pending.completion.flip_argument,
            buffer_index: pending.completion.buffer_index,
            submit_process_time_counter: pending.submit_process_time_counter,
        });
        let event_data = 6 | ((pending.completion.flip_argument & 0x0000_ffff_ffff_ffff) << 16);
        let registrations = state
            .video_out
            .flip_events
            .iter()
            .map(|(&equeue, &user_data)| (equeue, user_data))
            .collect::<Vec<_>>();
        for (equeue, user_data) in registrations {
            queue_kernel_event(
                state,
                KernelQueuedEvent {
                    equeue,
                    ident: 6,
                    filter: -13,
                    flags: 0,
                    fflags: 0,
                    data: event_data,
                    user_data,
                    agc_graphics_completions: 0,
                    agc_user_interrupts: 0,
                },
            );
            queued = true;
            if trace_logarithmic_sequence(state.agc_submission_count, 8) {
                eprintln!(
                    "[REMu][VIDEO] completed flip handle={} index={} address={:#x} equeue={equeue:#x}",
                    pending.handle, pending.completion.buffer_index, pending.completion.address,
                );
            }
        }
        // Flip completion coincides with a display vblank on the console. Wake vblank waiters with
        // OrbisVideoOutInternalEventId::Vblank (ident=1).
        let vblank_registrations = state
            .video_out
            .vblank_events
            .iter()
            .map(|(&equeue, &user_data)| (equeue, user_data))
            .collect::<Vec<_>>();
        for (equeue, user_data) in vblank_registrations {
            queue_kernel_event(
                state,
                KernelQueuedEvent {
                    equeue,
                    ident: 1,
                    filter: -13,
                    flags: 0,
                    fflags: 0,
                    data: state.video_out.completed_flip_count,
                    user_data,
                    agc_graphics_completions: 0,
                    agc_user_interrupts: 0,
                },
            );
            queued = true;
        }
    }
    if queued {
        notify_kernel_equeue_waiters();
    }
}

pub(crate) fn queue_agc_event_write(state: &mut ProbeDispatchState, event_type: u32) {
    let registrations = state
        .kernel_event_registrations
        .iter()
        .filter_map(|(&(equeue, filter, ident), &user_data)| {
            (filter == KERNEL_EVENT_FILTER_GRAPHICS).then_some((equeue, ident, filter, user_data))
        })
        .collect::<Vec<_>>();
    for (equeue, ident, filter, user_data) in registrations {
        queue_kernel_event(
            state,
            KernelQueuedEvent {
                equeue,
                ident,
                filter,
                flags: 0,
                fflags: 1,
                data: u64::from(event_type),
                user_data,
                agc_graphics_completions: 0,
                agc_user_interrupts: 0,
            },
        );
    }
    notify_kernel_equeue_waiters();
}

pub(crate) fn request_agc_submission_completion(
    state: &mut ProbeDispatchState,
    queue_id: u64,
    mut events: Vec<remu_gpu::GpuEvent>,
) -> bool {
    let sequence = state.agc_submission_count;
    annotate_submission_flips(&mut events, sequence);
    events.push(remu_gpu::GpuEvent::SubmissionBoundary(sequence));
    if publish_gpu_event(remu_gpu::GpuEvent::QueueSubmission { queue_id, events }) {
        state
            .pending_agc_completions
            .push_back((sequence, queue_id));
        true
    } else {
        false
    }
}

pub(crate) fn request_agc_compute_submission_completion(
    state: &mut ProbeDispatchState,
    owner: u64,
    mut events: Vec<remu_gpu::GpuEvent>,
) -> bool {
    let sequence = state.agc_submission_count;
    annotate_submission_flips(&mut events, sequence);
    events.push(remu_gpu::GpuEvent::SubmissionBoundary(sequence));
    // Backend queue zero is reserved for graphics. Compute completion is delivered exclusively
    // through the driver-wide AGC user interrupt below; also routing it through the owner's
    // graphics-filter event makes the guest retire the same ACB/RHI storage twice.
    let queue_id = owner.saturating_add(1);
    if publish_gpu_event(remu_gpu::GpuEvent::QueueSubmission { queue_id, events }) {
        state.pending_agc_user_interrupts.push_back(sequence);
        true
    } else {
        false
    }
}

fn annotate_submission_flips(events: &mut [remu_gpu::GpuEvent], sequence: u64) {
    for event in events {
        if let remu_gpu::GpuEvent::Flip {
            submission_sequence,
            ..
        } = event
        {
            *submission_sequence = Some(sequence);
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct AgcVideoOutTarget {
    pub(crate) address: u64,
    pub(crate) attributes: VideoOutBufferAttributes,
}

pub(crate) fn agc_video_out_target(
    video_out: &VideoOutState,
    handle: u32,
    buffer_index: u32,
) -> Option<AgcVideoOutTarget> {
    if handle != 1 || !video_out.is_open {
        return None;
    }
    let address = *video_out.display_buffers.get(buffer_index as usize)?;
    let attributes = video_out
        .display_buffer_attributes
        .get(buffer_index as usize)
        .copied()
        .flatten()?;
    Some(AgcVideoOutTarget {
        address,
        attributes,
    })
}

pub(crate) fn agc_video_out_target_for_address(
    video_out: &VideoOutState,
    address: u64,
) -> Option<AgcVideoOutTarget> {
    let index = video_out
        .display_buffers
        .iter()
        .position(|&candidate| candidate == address)?;
    let attributes = video_out
        .display_buffer_attributes
        .get(index)
        .copied()
        .flatten()?;
    Some(AgcVideoOutTarget {
        address,
        attributes,
    })
}

pub(crate) fn agc_submit_ordered_video_out_flip(
    video_out: &VideoOutState,
    handle: u32,
    buffer_index: u32,
    flip_argument: u64,
) -> bool {
    if handle != 1 || !video_out.is_open {
        eprintln!("[REMu][VIDEO] ignored flip for invalid handle {handle}");
        return false;
    }
    if video_out
        .display_buffers
        .get(buffer_index as usize)
        .is_none()
    {
        eprintln!("[REMu][VIDEO] ignored flip for unregistered buffer index {buffer_index}");
        return false;
    }
    let Some(target) = agc_video_out_target(video_out, handle, buffer_index) else {
        eprintln!("[REMu][VIDEO] ignored flip without attributes for buffer index {buffer_index}");
        return false;
    };
    emit_agc_gpu_event(remu_gpu::GpuEvent::Flip {
        address: target.address,
        buffer_index,
        flip_argument,
        submission_sequence: None,
        width: target.attributes.width,
        height: target.attributes.height,
        pitch: target.attributes.pitch,
        pixel_format: target.attributes.pixel_format,
        tiling_mode: target.attributes.tiling_mode,
    });
    true
}
