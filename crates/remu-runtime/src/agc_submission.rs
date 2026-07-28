use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

use remu_shader::GuestMemoryReader;

use crate::{
    AGC_IT_CLEAR_STATE, AGC_IT_COND_EXEC, AGC_IT_DISPATCH_DIRECT, AGC_IT_DISPATCH_INDIRECT,
    AGC_IT_DMA_DATA, AGC_IT_DRAW_INDEX_2, AGC_IT_DRAW_INDEX_AUTO, AGC_IT_DRAW_INDEX_OFFSET,
    AGC_IT_EVENT_WRITE, AGC_IT_INDEX_BASE, AGC_IT_INDEX_BUFFER_SIZE, AGC_IT_INDEX_TYPE,
    AGC_IT_INDIRECT_BUFFER, AGC_IT_NOP, AGC_IT_NUM_INSTANCES, AGC_IT_RELEASE_MEM, AGC_IT_SET_BASE,
    AGC_IT_SET_PREDICATION, AGC_IT_WAIT_REG_MEM, AGC_IT_WRITE_DATA, AGC_R_ACB_RESET,
    AGC_R_ACQUIRE_MEM, AGC_R_DMA_DATA, AGC_R_DRAW_INDEX_AUTO, AGC_R_DRAW_RESET, AGC_R_FLIP,
    AGC_R_RELEASE_MEM, AGC_R_WAIT_MEM_32, AGC_R_WAIT_MEM_64, AGC_R_WRITE_DATA, AgcGraphicsState,
    AgcShaderRegistry, AgcSubmissionError, AgcVideoOutTarget, ProbeDispatchState,
    ProcessGuestMemory, VideoOutState, agc_apply_set_base, agc_apply_submission_registers,
    agc_command_abi, agc_decode_compute_dispatch, agc_emit_draw, agc_emit_indexed_draw,
    agc_image_binding_address, agc_submit_ordered_video_out_flip, agc_video_out_target,
    capture_agc_gpu_events, emit_agc_gpu_event, low_u32, process_guest_range_is_readable,
    queue_agc_event_write, queue_video_out_flip, read_guest_u32, read_guest_u64,
    request_agc_compute_submission_completion, request_agc_submission_completion,
    trace_agc_programs, trace_agc_registers, trace_agc_shaders, trace_logarithmic_sequence,
};

#[derive(Clone, Copy)]
pub(crate) struct AgcSubmissionPacket {
    pub(crate) address: u64,
    pub(crate) length: u32,
    pub(crate) opcode: u32,
    pub(crate) register: u32,
    pub(crate) predicated: bool,
}

fn agc_apply_clear_state(
    packet_address: u64,
    length: u32,
    graphics: &mut AgcGraphicsState,
) -> Result<(), AgcSubmissionError> {
    if length != 2 {
        return Err(AgcSubmissionError::ClearStatePacketLength { actual: length });
    }
    let control = read_guest_u32(packet_address + 4);
    if control & !0x0f != 0 {
        return Err(AgcSubmissionError::UnsupportedClearStateControl(control));
    }
    // Gen5's DcbResetQueue emits CLEAR_STATE with a four-bit state selector. KytyPS5's command
    // processor validates that selector and resets the complete context, shader, user-config,
    // index, instance and indirect-argument state regardless of its admitted value. Resource
    // residency and telemetry are host bookkeeping, not register state.
    if trace_agc_registers() {
        eprintln!(
            "[REMu][AGC][REG] clear state packet={packet_address:#x} control={control:#x} cx={} sh={} uc={}",
            graphics.context_registers.len(),
            graphics.shader_registers.len(),
            graphics.user_config_registers.len(),
        );
    }
    graphics.reset_pipeline();
    Ok(())
}

pub(crate) fn decode_agc_pipeline_barrier(
    packet_address: u64,
    length: u32,
) -> Result<remu_gpu::GpuPipelineBarrier, AgcSubmissionError> {
    if length < 2 {
        return Err(AgcSubmissionError::EventWritePacketLength { actual: length });
    }
    let control = read_guest_u32(packet_address + 4);
    let event_type = control & 0x3f;
    let event_index = (control >> 8) & 0x7;
    let wait_for = match event_type {
        0x07 => remu_gpu::GpuPipelineStage::ComputeShader,
        0x0f => remu_gpu::GpuPipelineStage::VertexShader,
        0x10 => remu_gpu::GpuPipelineStage::PixelShader,
        _ => {
            return Err(AgcSubmissionError::UnsupportedEventWrite {
                event_type,
                event_index,
                length,
            });
        }
    };
    if control & !0x0000_073f != 0 {
        return Err(AgcSubmissionError::UnsupportedEventWriteControl(control));
    }
    if event_index != 4 {
        return Err(AgcSubmissionError::PartialFlushEventIndex {
            event_type,
            actual: event_index,
        });
    }
    Ok(remu_gpu::GpuPipelineBarrier { wait_for })
}

pub(crate) fn decode_agc_event_write(
    packet_address: u64,
    length: u32,
) -> Result<Option<remu_gpu::GpuPipelineBarrier>, AgcSubmissionError> {
    if length < 2 {
        return Err(AgcSubmissionError::EventWritePacketLength { actual: length });
    }
    let control = read_guest_u32(packet_address + 4);
    let event_type = control & 0x3f;
    let event_index = (control >> 8) & 0xf;
    if matches!(event_type, 0x07 | 0x0f | 0x10) {
        return decode_agc_pipeline_barrier(packet_address, length).map(Some);
    }
    // EVENT_TYPE=0x39 is PIXEL_PIPE_STAT_DUMP. AMD assigns it EVENT_INDEX=1 (the
    // ZPASS_DONE/PIXEL_PIPE_STAT group), and the two remaining packet dwords name the qword-aligned
    // occlusion-counter destination. The attached RDNA2 ISA manual covers shader instructions,
    // not PM4, but AMD's packet documentation and Linux's GFX10 `nvd.h` agree on this four-dword
    // form. KytyPS5 deliberately accepts this event without synthesizing counters; do the same
    // until Metal visibility-result buffers are wired into guest query memory. Rejecting it is
    // worse than the documented no-result fallback because Unreal treats a refused DCB as fatal.
    if event_type == 0x39 {
        if event_index != 1 || length != 4 || control & !0x0000_0f3f != 0 {
            return Err(AgcSubmissionError::UnsupportedEventWrite {
                event_type,
                event_index,
                length,
            });
        }
        let destination = read_guest_u64(packet_address + 8);
        if destination == 0 || !destination.is_multiple_of(8) {
            return Err(AgcSubmissionError::InvalidPixelPipeStatAddress(destination));
        }
        return Ok(None);
    }
    // Non-addressed Gen5 events use only EVENT_TYPE. EVENT_INDEX is reserved for the three
    // partial-flush forms above and the addressed pixel-pipe dump handled above; these packets
    // perform GPU cache actions and do not themselves signal a kernel graphics queue. The packet
    // processor consumes the header-declared length: some native command streams pad these events
    // to four dwords, and the unused payload does not change the event semantics.
    if control & !0x3f != 0 {
        return if event_index != 0 {
            Err(AgcSubmissionError::UnsupportedEventWrite {
                event_type,
                event_index,
                length,
            })
        } else {
            Err(AgcSubmissionError::UnsupportedEventWriteControl(control))
        };
    }
    Ok(None)
}

fn decode_agc_memory_write(
    packet_address: u64,
    length: u32,
    standard_packet: bool,
) -> Option<remu_gpu::GpuMemoryWrite> {
    if length < 5 {
        return None;
    }
    let control = read_guest_u32(packet_address + 4);
    let (destination, increment_address) = if standard_packet {
        // GFX10 PKT3_WRITE_DATA: DST_SEL is 11:8 and ADDR_INCR=0 advances the address.
        ((control >> 8) & 0x0f, control & (1 << 16) == 0)
    } else {
        // sceAgc's NOP wrapper stores four byte-sized fields: destination, cache policy,
        // increment suppression and write confirmation.
        (control & 0xff, (control >> 16) & 0xff == 0)
    };
    if !matches!(destination, 1 | 2 | 4 | 5) {
        return None;
    }
    let address = read_guest_u64(packet_address + 8);
    if address == 0 || !address.is_multiple_of(4) {
        return None;
    }
    let values = (0..length - 4)
        .map(|index| read_guest_u32(packet_address + 16 + u64::from(index) * 4))
        .collect::<Vec<_>>();
    static WRITE_DATA_TRACE_COUNT: AtomicUsize = AtomicUsize::new(0);
    let sequence = WRITE_DATA_TRACE_COUNT.fetch_add(1, Ordering::Relaxed) + 1;
    if sequence <= 32 || sequence.is_power_of_two() {
        eprintln!(
            "[REMu][AGC][SYNC] write sequence={sequence} packet={packet_address:#x} standard={standard_packet} destination={destination} address={address:#x} dwords={} increment={increment_address} values={values:?}",
            values.len()
        );
    }
    Some(remu_gpu::GpuMemoryWrite {
        address,
        values,
        increment_address,
    })
}

pub(crate) fn decode_agc_dma_data(
    packet_address: u64,
    length: u32,
    standard_packet: bool,
) -> Option<remu_gpu::GpuMemoryCopy> {
    let (source_address, destination_address, byte_count, source_selection, destination_selection) =
        if standard_packet {
            if length < 7 {
                return None;
            }
            let control = read_guest_u32(packet_address + 4);
            let command = read_guest_u32(packet_address + 24);
            (
                read_guest_u64(packet_address + 8),
                read_guest_u64(packet_address + 16),
                u64::from(command & 0x03ff_ffff),
                ((control >> 29) & 0x3) | ((command >> 24) & 0x4) | ((command >> 25) & 0x8),
                ((control >> 20) & 0x3) | ((command >> 25) & 0x4) | ((command >> 26) & 0x8),
            )
        } else {
            if length < 8 {
                return None;
            }
            let control = read_guest_u32(packet_address + 4);
            (
                read_guest_u64(packet_address + 24),
                read_guest_u64(packet_address + 16),
                u64::from(read_guest_u32(packet_address + 12)),
                (control >> 16) & 0xff,
                control & 0xff,
            )
        };
    let source_is_memory = matches!(source_selection, 0 | 3);
    let source_is_immediate = source_selection == 2;
    if (!source_is_memory && !source_is_immediate)
        || (source_is_memory && source_address < 0x10_000)
        || destination_address < 0x10_000
        || byte_count == 0
        || byte_count > 512 * 1024 * 1024
    {
        return None;
    }
    Some(remu_gpu::GpuMemoryCopy {
        source_address,
        destination_address,
        byte_count,
        source_selection,
        destination_selection,
        cpu_applied: false,
    })
}

fn mirror_agc_cpu_memory_copy(copy: &remu_gpu::GpuMemoryCopy) -> bool {
    const MAX_MIRRORED_COPY_BYTES: u64 = 1024 * 1024;
    if !matches!(copy.source_selection, 0 | 3)
        || !matches!(copy.destination_selection, 0 | 3)
        || copy.byte_count > MAX_MIRRORED_COPY_BYTES
    {
        return false;
    }
    let Ok(byte_count) = usize::try_from(copy.byte_count) else {
        return false;
    };
    if ProcessGuestMemory.readable_prefix_length(copy.source_address, byte_count)
        != Some(byte_count)
        || ProcessGuestMemory.readable_prefix_length(copy.destination_address, byte_count)
            != Some(byte_count)
    {
        return false;
    }
    // SAFETY: both complete ranges were validated against the runtime's live guest mappings.
    // `copy` provides memmove semantics. A destination protected by CpuWriteWatch is made
    // writable by its fault handler and dirtied exactly as a guest CPU write would be.
    unsafe {
        std::ptr::copy(
            copy.source_address as *const u8,
            copy.destination_address as *mut u8,
            byte_count,
        );
    }
    true
}

pub(crate) fn agc_handle_submission_packet(
    packet: AgcSubmissionPacket,
    graphics: &mut AgcGraphicsState,
    shaders: &mut AgcShaderRegistry,
    index_resources: &BTreeMap<u64, u64>,
    video_out: &VideoOutState,
    flips: &mut Vec<(u32, u32, u64)>,
    event_writes: &mut Vec<u32>,
    future_scanout_target: Option<AgcVideoOutTarget>,
) -> Result<(), AgcSubmissionError> {
    let AgcSubmissionPacket {
        address: packet_address,
        length,
        opcode,
        register,
        predicated: _,
    } = packet;
    if opcode == AGC_IT_CLEAR_STATE {
        agc_apply_clear_state(packet_address, length, graphics)?;
    }
    if opcode == AGC_IT_NOP && matches!(register, AGC_R_DRAW_RESET | AGC_R_ACB_RESET) && length >= 2
    {
        if trace_agc_registers() {
            eprintln!(
                "[REMu][AGC][REG] queue reset packet={packet_address:#x} register={register:#x} cx={} sh={} uc={}",
                graphics.context_registers.len(),
                graphics.shader_registers.len(),
                graphics.user_config_registers.len(),
            );
        }
        graphics.reset_pipeline();
    }
    agc_apply_submission_registers(packet_address, length, opcode, register, graphics)?;
    if opcode == AGC_IT_SET_BASE {
        agc_apply_set_base(packet_address, length, graphics)?;
    }
    if opcode == AGC_IT_EVENT_WRITE {
        if let Some(barrier) = decode_agc_event_write(packet_address, length)? {
            emit_agc_gpu_event(remu_gpu::GpuEvent::PipelineBarrier(barrier));
        }
        // EVENT_WRITE performs a cache or pipeline action. It is not an end-of-pipe interrupt;
        // completion notifications come from RELEASE_MEM interrupt selectors or the driver's
        // submit-on-done fallback.
        let _ = event_writes;
    }
    if ((opcode == AGC_IT_WRITE_DATA) || (opcode == AGC_IT_NOP && register == AGC_R_WRITE_DATA))
        && let Some(write) =
            decode_agc_memory_write(packet_address, length, opcode == AGC_IT_WRITE_DATA)
    {
        emit_agc_gpu_event(remu_gpu::GpuEvent::MemoryWrite(write));
    }
    if (opcode == AGC_IT_DMA_DATA || (opcode == AGC_IT_NOP && register == AGC_R_DMA_DATA))
        && let Some(mut copy) =
            decode_agc_dma_data(packet_address, length, opcode == AGC_IT_DMA_DATA)
    {
        // Shader evaluation happens while the command buffer is parsed, before the asynchronous
        // Metal consumer reaches this event. Mirror bounded CPU-sourced uploads now so later
        // draws observe the descriptors and vertex data that DMA_DATA established. The ordered
        // event remains authoritative for copies whose source is produced by prior GPU work.
        copy.cpu_applied = mirror_agc_cpu_memory_copy(&copy);
        static DMA_TRACE_COUNT: AtomicUsize = AtomicUsize::new(0);
        let sequence = DMA_TRACE_COUNT.fetch_add(1, Ordering::Relaxed) + 1;
        if trace_agc_programs()
            || sequence <= 32
            || sequence.is_power_of_two()
            || graphics
                .sampled_texture_addresses
                .contains(&copy.destination_address)
        {
            eprintln!(
                "[REMu][AGC][DMA] sequence={sequence} packet={packet_address:#x} standard={} src_sel={} dst_sel={} source={:#x} destination={:#x} bytes={:#x}",
                opcode == AGC_IT_DMA_DATA,
                copy.source_selection,
                copy.destination_selection,
                copy.source_address,
                copy.destination_address,
                copy.byte_count,
            );
        }
        emit_agc_gpu_event(remu_gpu::GpuEvent::MemoryCopy(copy));
    }
    if opcode == AGC_IT_NOP && register == AGC_R_ACQUIRE_MEM && length >= 8 {
        emit_agc_gpu_event(remu_gpu::GpuEvent::MemoryAcquire(
            decode_agc_memory_acquire(packet_address),
        ));
    }
    if opcode == AGC_IT_NOP
        && register == AGC_R_RELEASE_MEM
        && length >= 7
        && let Some(release) = decode_agc_memory_release(packet_address, false)
    {
        emit_agc_gpu_event(remu_gpu::GpuEvent::MemoryRelease(release));
    }
    if opcode == AGC_IT_RELEASE_MEM
        && length >= 8
        && let Some(release) = decode_agc_memory_release(packet_address, true)
    {
        emit_agc_gpu_event(remu_gpu::GpuEvent::MemoryRelease(release));
    }
    if opcode == AGC_IT_NOP
        && matches!(register, AGC_R_WAIT_MEM_32 | AGC_R_WAIT_MEM_64)
        && length >= if register == AGC_R_WAIT_MEM_64 { 9 } else { 6 }
        && let Some(wait) = decode_agc_memory_wait(packet_address, register, false)
    {
        emit_agc_gpu_event(remu_gpu::GpuEvent::MemoryWait(wait));
    }
    if opcode == AGC_IT_WAIT_REG_MEM
        && length >= 7
        && let Some(wait) = decode_agc_memory_wait(packet_address, register, true)
    {
        emit_agc_gpu_event(remu_gpu::GpuEvent::MemoryWait(wait));
    }
    if opcode == AGC_IT_INDEX_BASE && length >= 3 {
        graphics.index_buffer_address = u64::from(read_guest_u32(packet_address + 4))
            | (u64::from(read_guest_u32(packet_address + 8)) << 32);
        graphics.index_resource_address = index_resources
            .get(&graphics.index_buffer_address)
            .copied()
            .unwrap_or_default();
    }
    if opcode == AGC_IT_INDEX_BUFFER_SIZE && length >= 2 {
        graphics.index_buffer_count = read_guest_u32(packet_address + 4);
    }
    if opcode == AGC_IT_INDEX_TYPE && length >= 2 {
        graphics.index_size = read_guest_u32(packet_address + 4) & 0x3;
    }
    if opcode == AGC_IT_NUM_INSTANCES && length >= 2 {
        graphics.instance_count = read_guest_u32(packet_address + 4).max(1);
    }
    if opcode == AGC_IT_DRAW_INDEX_2 && length >= 6 {
        graphics.index_buffer_count = read_guest_u32(packet_address + 4);
        graphics.index_buffer_address = read_guest_u64(packet_address + 8);
        graphics.index_resource_address = index_resources
            .get(&graphics.index_buffer_address)
            .copied()
            .unwrap_or_default();
        agc_emit_draw(
            packet_address,
            graphics,
            shaders,
            video_out,
            true,
            0,
            read_guest_u32(packet_address + 16),
            read_guest_u32(packet_address + 20),
            future_scanout_target,
        );
    }
    if opcode == AGC_IT_DRAW_INDEX_OFFSET && length >= 5 {
        agc_emit_indexed_draw(
            packet_address,
            graphics,
            shaders,
            video_out,
            future_scanout_target,
        );
    }
    if opcode == AGC_IT_DRAW_INDEX_AUTO && length >= 3 {
        agc_emit_draw(
            packet_address,
            graphics,
            shaders,
            video_out,
            false,
            0,
            read_guest_u32(packet_address + 4),
            read_guest_u32(packet_address + 8),
            future_scanout_target,
        );
    }
    if opcode == AGC_IT_NOP && register == AGC_R_DRAW_INDEX_AUTO && length >= 2 {
        agc_emit_draw(
            packet_address,
            graphics,
            shaders,
            video_out,
            false,
            0,
            read_guest_u32(packet_address + 4),
            0,
            future_scanout_target,
        );
    }
    if matches!(opcode, AGC_IT_DISPATCH_DIRECT | AGC_IT_DISPATCH_INDIRECT) {
        let dispatch =
            agc_decode_compute_dispatch(packet_address, length, opcode, graphics, shaders)
                .map_err(|source| AgcSubmissionError::Compute {
                    packet_address,
                    source,
                })?;
        if trace_logarithmic_sequence(dispatch.sequence, 8) {
            eprintln!(
                "[REMu][AGC] compute={} source={:?} grid={:?} base={:?} local={:?} wave={} cs={:#x}",
                dispatch.sequence,
                dispatch.source,
                dispatch.launch.grid,
                dispatch.launch.base_group,
                dispatch.launch.threads_per_group,
                dispatch.launch.wave_lane_count,
                dispatch.shader_address,
            );
        }
        if trace_agc_shaders()
            && dispatch.sequence <= 4
            && let Some(shader) = dispatch.shader.as_ref()
        {
            let opcodes = shader
                .program
                .instructions
                .iter()
                .map(|instruction| instruction.opcode.to_string())
                .collect::<Vec<_>>()
                .join(",");
            eprintln!(
                "[REMu][AGC][SHADER][COMPUTE] cs={:#x} wave={} ir={} ops=[{opcodes}] instructions={:#?}",
                dispatch.shader_address,
                dispatch.launch.wave_lane_count,
                shader.program.instructions.len(),
                shader.program.instructions,
            );
        }
        if trace_agc_shaders()
            && let Some(evaluation) = dispatch.shader_evaluation.as_deref()
            && evaluation.image_bindings.iter().any(|binding| {
                agc_image_binding_address(binding)
                    .is_some_and(|address| graphics.sampled_texture_addresses.contains(&address))
            })
            && graphics
                .traced_sampled_compute_shaders
                .insert(dispatch.shader_address)
        {
            let images = evaluation
                .image_bindings
                .iter()
                .map(|binding| {
                    format!(
                        "{:#x}:{}@{:#x}:r={:x?}",
                        binding.pc,
                        binding.opcode,
                        agc_image_binding_address(binding).unwrap_or_default(),
                        binding.resource_descriptor,
                    )
                })
                .collect::<Vec<_>>()
                .join(",");
            eprintln!(
                "[REMu][AGC][SHADER][COMPUTE-SAMPLED] dispatch={} cs={:#x} images=[{images}]",
                dispatch.sequence, dispatch.shader_address,
            );
        }
        if dispatch.launch.is_empty() {
            eprintln!(
                "[REMu][AGC] skipped empty compute dispatch={} source={:?} grid={:?} base={:?} cs={:#x}",
                dispatch.sequence,
                dispatch.source,
                dispatch.launch.grid,
                dispatch.launch.base_group,
                dispatch.shader_address,
            );
        } else {
            emit_agc_gpu_event(remu_gpu::GpuEvent::ComputeDispatch(Box::new(dispatch)));
        }
    }
    if opcode == AGC_IT_NOP && register == AGC_R_FLIP && length >= 6 {
        let handle = read_guest_u32(packet_address + 4);
        let buffer_index = read_guest_u32(packet_address + 8);
        let flip_argument = u64::from(read_guest_u32(packet_address + 16))
            | (u64::from(read_guest_u32(packet_address + 20)) << 32);
        if agc_submit_ordered_video_out_flip(video_out, handle, buffer_index, flip_argument) {
            flips.push((handle, buffer_index, flip_argument));
        }
    }
    Ok(())
}

pub(crate) fn decode_agc_memory_acquire(packet_address: u64) -> remu_gpu::GpuMemoryAcquire {
    let coher_control = read_guest_u32(packet_address + 4);
    let size_units = u64::from(read_guest_u32(packet_address + 8))
        | (u64::from(read_guest_u32(packet_address + 12) & 0xff) << 32);
    let base_units = u64::from(read_guest_u32(packet_address + 16))
        | (u64::from(read_guest_u32(packet_address + 20) & 0x00ff_ffff) << 32);
    remu_gpu::GpuMemoryAcquire {
        engine: coher_control >> 31,
        cb_db_control: coher_control & 0x7fff_ffff,
        base_address: base_units << 8,
        size_bytes: size_units << 8,
        poll_interval: read_guest_u32(packet_address + 24) & 0xffff,
        gcr_control: read_guest_u32(packet_address + 28) & 0x7_ffff,
    }
}

fn decode_agc_memory_release(
    packet_address: u64,
    standard_packet: bool,
) -> Option<remu_gpu::GpuMemoryRelease> {
    static TRACE_COUNT: AtomicUsize = AtomicUsize::new(0);
    let control = read_guest_u32(packet_address + 8);
    let address = read_guest_u64(packet_address + 12);
    let data = read_guest_u64(packet_address + 20);
    let data_selection = if standard_packet {
        (control >> 29) & 0x7
    } else {
        (control >> 16) & 0xff
    };
    let value = match data_selection {
        1 if address.is_multiple_of(4) => {
            Some(remu_gpu::GpuMemoryReleaseValue::Immediate32(low_u32(data)))
        }
        2 if address.is_multiple_of(8) => Some(remu_gpu::GpuMemoryReleaseValue::Immediate64(data)),
        3 | 4 if address.is_multiple_of(8) => Some(remu_gpu::GpuMemoryReleaseValue::Timestamp),
        _ => None,
    };
    let trace = TRACE_COUNT.fetch_add(1, Ordering::Relaxed) + 1;
    if trace <= 96 || trace.is_power_of_two() {
        eprintln!(
            "[REMu][AGC][SYNC] release packet={packet_address:#x} standard={standard_packet} control={control:#010x} selection={data_selection} address={address:#x} data={data:#x} emitted={}",
            value.is_some() && address != 0
        );
    }
    Some(remu_gpu::GpuMemoryRelease {
        address,
        value: value?,
    })
    .filter(|_| address != 0)
}

pub(crate) fn decode_agc_memory_wait(
    packet_address: u64,
    register: u32,
    standard_packet: bool,
) -> Option<remu_gpu::GpuMemoryWait> {
    static TRACE_COUNT: AtomicUsize = AtomicUsize::new(0);
    let is_64_bit = !standard_packet && register == AGC_R_WAIT_MEM_64;
    let (address, reference, mask, control) = if standard_packet {
        (
            read_guest_u64(packet_address + 8),
            u64::from(read_guest_u32(packet_address + 16)),
            u64::from(read_guest_u32(packet_address + 20)),
            read_guest_u32(packet_address + 4),
        )
    } else if is_64_bit {
        (
            read_guest_u64(packet_address + 4),
            read_guest_u64(packet_address + 20),
            read_guest_u64(packet_address + 12),
            read_guest_u32(packet_address + 28),
        )
    } else {
        (
            read_guest_u64(packet_address + 4),
            u64::from(read_guest_u32(packet_address + 16)),
            u64::from(read_guest_u32(packet_address + 12)),
            read_guest_u32(packet_address + 20),
        )
    };
    let compare_function = control & 0x7;
    let alignment = if is_64_bit { 8 } else { 4 };
    let decoded = if address == 0 || !address.is_multiple_of(alignment) || compare_function == 7 {
        None
    } else {
        Some(remu_gpu::GpuMemoryWait {
            address,
            reference,
            mask,
            compare_function,
            is_64_bit,
        })
    };
    let trace = TRACE_COUNT.fetch_add(1, Ordering::Relaxed) + 1;
    if trace <= 64 || trace.is_power_of_two() {
        let current = decoded.map(|wait| {
            if wait.is_64_bit {
                read_guest_u64(wait.address)
            } else {
                u64::from(read_guest_u32(wait.address))
            }
        });
        eprintln!(
            "[REMu][AGC][SYNC] wait packet={packet_address:#x} standard={standard_packet} bits={} control={control:#010x} compare={compare_function} address={address:#x} mask={mask:#x} reference={reference:#x} current={current:?} satisfied={:?}",
            if is_64_bit { 64 } else { 32 },
            current
                .zip(decoded)
                .map(|(value, wait)| wait.is_satisfied_by(value))
        );
    }
    decoded
}

fn agc_read_mapped_u32(address: u64) -> Option<u32> {
    let mut bytes = [0_u8; 4];
    ProcessGuestMemory
        .read(address, &mut bytes)
        .then(|| u32::from_le_bytes(bytes))
}

pub(crate) fn agc_read_mapped_u64(address: u64) -> Option<u64> {
    let mut bytes = [0_u8; 8];
    ProcessGuestMemory
        .read(address, &mut bytes)
        .then(|| u64::from_le_bytes(bytes))
}

fn agc_apply_predication(
    packet: AgcSubmissionPacket,
    graphics: &mut AgcGraphicsState,
) -> Result<(), AgcSubmissionError> {
    if packet.length < 3 {
        return Err(AgcSubmissionError::PredicationPacketLength {
            actual: packet.length,
        });
    }
    let payload_dwords = packet.length - 1;
    let first = read_guest_u32(packet.address + 4);
    let second = read_guest_u32(packet.address + 8);
    let third = (payload_dwords >= 3).then(|| read_guest_u32(packet.address + 12));
    const PREDICATION_FLAGS_MASK: u32 = 0x0007_1100;
    let (flags, value_address) =
        if third.is_some_and(|third| first & !PREDICATION_FLAGS_MASK == 0 && third <= 0xffff) {
            (
                first,
                u64::from(second & 0xffff_fff0) | (u64::from(third.unwrap_or_default()) << 32),
            )
        } else {
            (
                second,
                u64::from(first & 0xffff_fff0) | (u64::from(second & 0xff) << 32),
            )
        };
    let condition = (flags >> 8) & 1;
    let operation = (flags >> 16) & 7;
    match operation {
        0 => graphics.predicate_skip = false,
        3 => {
            let value = agc_read_mapped_u64(value_address).ok_or(
                AgcSubmissionError::UnreadablePredicationValue(value_address),
            )?;
            graphics.predicate_skip = match condition {
                0 => value != 0,
                _ => value == 0,
            };
            if trace_agc_registers() || trace_agc_programs() {
                eprintln!(
                    "[REMu][AGC][PRED] address={value_address:#x} value={value:#x} condition={condition} skip={}",
                    graphics.predicate_skip
                );
            }
        }
        _ => {
            return Err(AgcSubmissionError::UnsupportedPredicationOperation(
                operation,
            ));
        }
    }
    Ok(())
}

fn agc_cond_exec_skip_end(
    packet: AgcSubmissionPacket,
    command_end: u64,
) -> Result<Option<u64>, AgcSubmissionError> {
    if packet.length != 5 {
        return Err(AgcSubmissionError::CondExecPacketLength {
            actual: packet.length,
        });
    }
    let low = read_guest_u32(packet.address + 4);
    let high = read_guest_u32(packet.address + 8);
    let reserved = read_guest_u32(packet.address + 12);
    let control = low & 3;
    if control != 0 || reserved != 0 {
        return Err(AgcSubmissionError::UnsupportedCondExecControl(
            control | reserved,
        ));
    }
    let value_address = u64::from(low & !3) | (u64::from(high) << 32);
    let value = agc_read_mapped_u32(value_address)
        .ok_or(AgcSubmissionError::UnreadableCondExecValue(value_address))?;
    if value != 0 {
        return Ok(None);
    }
    let skip_dwords = read_guest_u32(packet.address + 16) & 0x3fff;
    let packet_end = packet.address + u64::from(packet.length) * 4;
    let Some(skip_end) = packet_end.checked_add(u64::from(skip_dwords) * 4) else {
        return Err(AgcSubmissionError::InvalidCondExecRange {
            packet_address: packet.address,
            skip_dwords,
        });
    };
    if skip_end > command_end {
        return Err(AgcSubmissionError::InvalidCondExecRange {
            packet_address: packet.address,
            skip_dwords,
        });
    }
    if trace_agc_registers() || trace_agc_programs() {
        eprintln!(
            "[REMu][AGC][COND] address={value_address:#x} value=0 skip_dwords={skip_dwords} end={skip_end:#x}"
        );
    }
    Ok(Some(skip_end))
}

fn agc_branch_is_satisfied(value: u64, reference: u64, mask: u64, comparison: u32) -> bool {
    match comparison {
        0 => true,
        1 => value & mask < reference,
        2 => value & mask <= reference,
        3 => value & mask == reference,
        4 => value & mask != reference,
        5 => value & mask >= reference,
        6 => value & mask > reference,
        _ => false,
    }
}

fn agc_branch_target(
    packet: AgcSubmissionPacket,
) -> Result<Option<(u64, u32)>, AgcSubmissionError> {
    let control = read_guest_u32(packet.address + 4);
    let mode = control & 3;
    let comparison = (control >> 8) & 7;
    if !matches!(mode, 1 | 2) {
        return Err(AgcSubmissionError::UnsupportedBranchMode(mode));
    }
    if comparison > 6 {
        return Err(AgcSubmissionError::UnsupportedBranchComparison(comparison));
    }
    let compare_address = u64::from(read_guest_u32(packet.address + 8) & !7)
        | (u64::from(read_guest_u32(packet.address + 12)) << 32);
    // Comparison zero is the native unconditional form. The address still acts as a queue label
    // for command-buffer stitching, but the command processor does not need its contents to pick
    // a target.
    let value = if comparison == 0 {
        0
    } else {
        agc_read_mapped_u64(compare_address)
            .ok_or(AgcSubmissionError::UnreadableBranchValue(compare_address))?
    };
    let mask = read_guest_u64(packet.address + 16);
    let reference = read_guest_u64(packet.address + 24);
    let take_then = agc_branch_is_satisfied(value, reference, mask, comparison);
    let (address_offset, count_offset) = if take_then {
        (32, 40)
    } else if mode == 2 {
        (44, 52)
    } else {
        return Ok(None);
    };
    let address = u64::from(read_guest_u32(packet.address + address_offset) & !3)
        | (u64::from(read_guest_u32(packet.address + address_offset + 4)) << 32);
    let dword_count = read_guest_u32(packet.address + count_offset) & 0x000f_ffff;
    if trace_agc_registers() || trace_agc_programs() {
        eprintln!(
            "[REMu][AGC][BRANCH] compare={compare_address:#x} value={value:#x} reference={reference:#x} mask={mask:#x} function={comparison} then={take_then} target={address:#x}/{dword_count}"
        );
    }
    (address != 0 && dword_count != 0)
        .then_some((address, dword_count))
        .map_or(Ok(None), |(address, dword_count)| {
            let byte_count = usize::try_from(u64::from(dword_count) * 4)
                .expect("the 20-bit branch dword count fits usize");
            if process_guest_range_is_readable(address, byte_count) {
                Ok(Some((address, dword_count)))
            } else {
                Err(AgcSubmissionError::InvalidIndirectBuffer {
                    address,
                    dword_count,
                })
            }
        })
}

pub(crate) struct AgcSubmissionContext<'a> {
    pub(crate) graphics: &'a mut AgcGraphicsState,
    pub(crate) shaders: &'a mut AgcShaderRegistry,
    pub(crate) index_resources: &'a BTreeMap<u64, u64>,
    pub(crate) video_out: &'a VideoOutState,
    pub(crate) submission_count: &'a mut u64,
    pub(crate) flips: &'a mut Vec<(u32, u32, u64)>,
    pub(crate) event_writes: &'a mut Vec<u32>,
}

fn agc_packet_video_out_target(
    packet: AgcSubmissionPacket,
    video_out: &VideoOutState,
) -> Option<AgcVideoOutTarget> {
    if packet.opcode != AGC_IT_NOP || packet.register != AGC_R_FLIP || packet.length < 6 {
        return None;
    }
    agc_video_out_target(
        video_out,
        read_guest_u32(packet.address + 4),
        read_guest_u32(packet.address + 8),
    )
}

fn agc_scanout_inference_boundary(packet: AgcSubmissionPacket) -> bool {
    packet.predicated
        || matches!(
            packet.opcode,
            AGC_IT_SET_PREDICATION | AGC_IT_COND_EXEC | AGC_IT_INDIRECT_BUFFER | AGC_IT_CLEAR_STATE
        )
        || (packet.opcode == AGC_IT_NOP
            && matches!(packet.register, AGC_R_DRAW_RESET | AGC_R_ACB_RESET))
}

pub(crate) fn agc_future_scanout_targets(
    packets: &[AgcSubmissionPacket],
    video_out: &VideoOutState,
) -> Vec<Option<AgcVideoOutTarget>> {
    let mut future = vec![None; packets.len()];
    let mut next = None;
    for (index, packet) in packets.iter().copied().enumerate().rev() {
        if agc_scanout_inference_boundary(packet) {
            next = None;
            continue;
        }
        future[index] = next;
        if let Some(target) = agc_packet_video_out_target(packet, video_out) {
            next = Some(target);
        }
    }
    future
}

pub(crate) fn agc_parse_submission(
    command_address: u64,
    dword_count: u32,
    context: &mut AgcSubmissionContext<'_>,
) -> Result<(), AgcSubmissionError> {
    let (offset, packet_counts) =
        agc_parse_command_stream(command_address, dword_count, context, 0)?;

    *context.submission_count = context.submission_count.saturating_add(1);
    if trace_agc_programs() || trace_logarithmic_sequence(*context.submission_count, 8) {
        let packet_summary = packet_counts
            .iter()
            .map(|(&(opcode, register), &count)| {
                format!("op={opcode:#04x}/reg={register:#04x}:{count}")
            })
            .collect::<Vec<_>>()
            .join(",");
        eprintln!(
            "[REMu][AGC] submit={} address={command_address:#x} dwords={dword_count} parsed={offset} packets=[{packet_summary}] flips={}",
            context.submission_count,
            context.flips.len()
        );
    }
    Ok(())
}

pub(crate) fn agc_parse_graphics_submission(
    command_address: u64,
    dword_count: u32,
    context: &mut AgcSubmissionContext<'_>,
) -> Result<(), AgcSubmissionError> {
    let continuation_packets = agc_command_abi::take_graphics_continuations_before(command_address);
    let has_explicit_continuation = !continuation_packets.is_empty();
    for packet_address in continuation_packets {
        let packet = AgcSubmissionPacket {
            address: packet_address,
            length: 14,
            opcode: AGC_IT_INDIRECT_BUFFER,
            register: 0,
            predicated: false,
        };
        if let Some((target_address, target_dword_count)) = agc_branch_target(packet)? {
            eprintln!(
                "[REMu][AGC] applying graphics continuation packet={packet_address:#x} target={target_address:#x}/{target_dword_count} before={command_address:#x}"
            );
            agc_parse_command_stream(target_address, target_dword_count, context, 1)?;
            agc_command_abi::consume_recorded_graphics_range(target_address, target_dword_count);
        }
    }
    if !has_explicit_continuation {
        if let Some((prefix_address, prefix_dword_count)) =
            agc_command_abi::take_recorded_graphics_prefix(command_address)
        {
            eprintln!(
                "[REMu][AGC] applying recorded graphics prefix address={prefix_address:#x} dwords={prefix_dword_count} before={command_address:#x}"
            );
            agc_parse_command_stream(prefix_address, prefix_dword_count, context, 1)?;
        }
    }
    agc_parse_submission(command_address, dword_count, context)
}

fn agc_parse_command_stream(
    command_address: u64,
    dword_count: u32,
    context: &mut AgcSubmissionContext<'_>,
    nesting: u32,
) -> Result<(u32, BTreeMap<(u32, u32), u32>), AgcSubmissionError> {
    const MAX_INDIRECT_BUFFER_NESTING: u32 = 32;
    if nesting > MAX_INDIRECT_BUFFER_NESTING {
        return Err(AgcSubmissionError::IndirectBufferNesting {
            maximum: MAX_INDIRECT_BUFFER_NESTING,
        });
    }
    let mut offset = 0_u32;
    let mut packet_counts = BTreeMap::<(u32, u32), u32>::new();
    let mut packets = Vec::new();
    while offset < dword_count {
        let packet_address = command_address + u64::from(offset) * 4;
        let header = read_guest_u32(packet_address);
        let packet_type = header >> 30;
        if packet_type == 2 {
            offset += 1;
            continue;
        }
        if packet_type != 3 {
            return Err(AgcSubmissionError::UnsupportedPacketType {
                dword: offset,
                packet_type,
                header,
            });
        }
        let length = ((header >> 16) & 0x3fff) + 2;
        let remaining = dword_count - offset;
        if length > remaining {
            return Err(AgcSubmissionError::InvalidPacketLength {
                dword: offset,
                length,
                remaining,
            });
        }
        let opcode = (header >> 8) & 0xff;
        let register = (header >> 2) & 0x3f;
        *packet_counts.entry((opcode, register)).or_default() += 1;
        packets.push(AgcSubmissionPacket {
            address: packet_address,
            length,
            opcode,
            register,
            predicated: header & 1 != 0,
        });
        offset += length;
    }

    // Validate newly decoded order-visible packet forms before emitting any GPU side effect from
    // the submission.
    for packet in &packets {
        if packet.opcode == AGC_IT_EVENT_WRITE {
            decode_agc_event_write(packet.address, packet.length)?;
        }
        if packet.opcode == AGC_IT_INDIRECT_BUFFER && !matches!(packet.length, 4 | 14) {
            return Err(AgcSubmissionError::IndirectBufferPacketLength {
                actual: packet.length,
            });
        }
    }

    let future_scanout_targets = agc_future_scanout_targets(&packets, context.video_out);
    let command_end = command_address + u64::from(dword_count) * 4;
    let mut conditional_skip_end = None;
    for (packet, future_scanout_target) in packets
        .iter()
        .copied()
        .zip(future_scanout_targets.into_iter())
    {
        if conditional_skip_end.is_some_and(|skip_end| packet.address < skip_end) {
            let packet_end = packet.address + u64::from(packet.length) * 4;
            if packet_end > conditional_skip_end.unwrap_or_default() {
                return Err(AgcSubmissionError::InvalidCondExecRange {
                    packet_address: packet.address,
                    skip_dwords: packet.length,
                });
            }
            continue;
        }
        conditional_skip_end = None;
        if packet.predicated && context.graphics.predicate_skip {
            if trace_agc_registers() {
                eprintln!(
                    "[REMu][AGC][PRED] skipped packet={:#x} opcode={:#x} register={:#x} dwords={}",
                    packet.address, packet.opcode, packet.register, packet.length
                );
            }
            continue;
        }
        if packet.opcode == AGC_IT_SET_PREDICATION {
            agc_apply_predication(packet, context.graphics)?;
            continue;
        }
        if packet.opcode == AGC_IT_COND_EXEC {
            conditional_skip_end = agc_cond_exec_skip_end(packet, command_end)?;
            continue;
        }
        if packet.opcode == AGC_IT_INDIRECT_BUFFER {
            if packet.length == 14 {
                if agc_command_abi::graphics_continuation_was_consumed(packet.address) {
                    if trace_agc_registers() || trace_agc_programs() {
                        eprintln!(
                            "[REMu][AGC][BRANCH] skipped already-applied continuation packet={:#x}",
                            packet.address
                        );
                    }
                    continue;
                }
                if let Some((branch_address, branch_dword_count)) = agc_branch_target(packet)? {
                    agc_parse_command_stream(
                        branch_address,
                        branch_dword_count,
                        context,
                        nesting + 1,
                    )?;
                    // This target has now executed as part of the native branch. Its constructor
                    // records exist only so a target separated from its submitted tail can be
                    // recovered when the branch packet itself is unavailable. Leaving them live
                    // lets the adjacent tail reconstruct and execute this prefix a second time,
                    // after completion labels in it may already have been returned to the guest
                    // allocator. That stale replay creates producerless WAIT_REG_MEM deadlocks.
                    agc_command_abi::consume_recorded_graphics_range(
                        branch_address,
                        branch_dword_count,
                    );
                }
                agc_command_abi::retire_pending_graphics_continuation(packet.address);
                continue;
            }
            let indirect_address = u64::from(read_guest_u32(packet.address + 4) & !3)
                | (u64::from(read_guest_u32(packet.address + 8)) << 32);
            let indirect_dword_count = read_guest_u32(packet.address + 12) & 0x000f_ffff;
            if indirect_dword_count == 0 {
                continue;
            }
            let byte_count = usize::try_from(u64::from(indirect_dword_count) * 4)
                .expect("the 20-bit indirect dword count fits usize");
            if indirect_address == 0
                || !process_guest_range_is_readable(indirect_address, byte_count)
            {
                return Err(AgcSubmissionError::InvalidIndirectBuffer {
                    address: indirect_address,
                    dword_count: indirect_dword_count,
                });
            }
            static INDIRECT_TRACE_COUNT: AtomicU64 = AtomicU64::new(0);
            let sequence = INDIRECT_TRACE_COUNT.fetch_add(1, Ordering::Relaxed) + 1;
            if trace_agc_programs() || (nesting < 8 && trace_logarithmic_sequence(sequence, 32)) {
                eprintln!(
                    "[REMu][AGC] indirect sequence={sequence} depth={} packet={:#x} address={indirect_address:#x} dwords={indirect_dword_count}",
                    nesting + 1,
                    packet.address,
                );
            }
            agc_parse_command_stream(indirect_address, indirect_dword_count, context, nesting + 1)?;
            continue;
        }
        agc_handle_submission_packet(
            packet,
            context.graphics,
            context.shaders,
            context.index_resources,
            context.video_out,
            context.flips,
            context.event_writes,
            future_scanout_target,
        )?;
    }

    Ok((offset, packet_counts))
}

pub(crate) fn agc_driver_submit_dcb(arguments: [u64; 6], state: &mut ProbeDispatchState) -> u64 {
    let packet = arguments[0];
    if packet == 0 {
        eprintln!("[REMu][AGC] rejected DCB submission: null packet");
        return u64::MAX;
    }
    let command_address = read_guest_u64(packet);
    let dword_count = read_guest_u32(packet + 8);
    if command_address == 0 || dword_count == 0 || dword_count > 1_000_000 {
        // The guest treats a refused submission as fatal, so name the packet it rejected.
        eprintln!(
            "[REMu][AGC] rejected DCB submission: packet={packet:#x} commands={command_address:#x} dwords={dword_count} args={:#x},{:#x},{:#x}",
            arguments[1], arguments[2], arguments[3]
        );
        return u64::MAX;
    }
    let next_submission = state.agc_submission_count.saturating_add(1);
    if trace_agc_programs() || trace_logarithmic_sequence(next_submission, 8) {
        eprintln!(
            "[REMu][AGC] queue=graphics submit={next_submission} address={command_address:#x} dwords={dword_count}"
        );
    }
    let mut flips = Vec::new();
    let mut event_writes = Vec::new();
    let (result, gpu_events) = capture_agc_gpu_events(|| {
        agc_parse_graphics_submission(
            command_address,
            dword_count,
            &mut AgcSubmissionContext {
                graphics: &mut state.agc_graphics,
                shaders: &mut state.agc_shader_registry,
                index_resources: &state.agc_index_resources,
                video_out: &state.video_out,
                submission_count: &mut state.agc_submission_count,
                flips: &mut flips,
                event_writes: &mut event_writes,
            },
        )
    });
    match result {
        Ok(()) => {
            agc_command_abi::track_graphics_submission(command_address, dword_count);
            if !request_agc_submission_completion(state, 0, gpu_events) {
                eprintln!("[REMu][AGC] rejected DCB submission: presenter is unavailable");
                return u64::MAX;
            }
            for (handle, buffer_index, flip_argument) in flips {
                queue_video_out_flip(state, handle, buffer_index, flip_argument);
            }
            for event_type in event_writes {
                queue_agc_event_write(state, event_type);
            }
            0
        }
        Err(error) => {
            eprintln!("[REMu][AGC] rejected DCB submission: {error}");
            u64::MAX
        }
    }
}

pub(crate) fn agc_driver_submit_multi_dcbs(
    arguments: [u64; 6],
    state: &mut ProbeDispatchState,
) -> u64 {
    let addresses = arguments[0];
    let sizes = arguments[1];
    let count = low_u32(arguments[2]);
    if addresses == 0 || sizes == 0 || count == 0 || count > 4096 {
        return u64::MAX;
    }
    for index in 0..count {
        let command_address = read_guest_u64(addresses + u64::from(index) * 8);
        let dword_count = read_guest_u32(sizes + u64::from(index) * 4);
        if command_address != 0 && dword_count != 0 && dword_count <= 1_000_000 {
            let mut flips = Vec::new();
            let mut event_writes = Vec::new();
            let (result, gpu_events) = capture_agc_gpu_events(|| {
                agc_parse_graphics_submission(
                    command_address,
                    dword_count,
                    &mut AgcSubmissionContext {
                        graphics: &mut state.agc_graphics,
                        shaders: &mut state.agc_shader_registry,
                        index_resources: &state.agc_index_resources,
                        video_out: &state.video_out,
                        submission_count: &mut state.agc_submission_count,
                        flips: &mut flips,
                        event_writes: &mut event_writes,
                    },
                )
            });
            match result {
                Ok(()) => {
                    agc_command_abi::track_graphics_submission(command_address, dword_count);
                    if !request_agc_submission_completion(state, 0, gpu_events) {
                        eprintln!(
                            "[REMu][AGC] rejected DCB {index} in multi-submission: presenter is unavailable"
                        );
                        return u64::MAX;
                    }
                    for (handle, buffer_index, flip_argument) in flips {
                        queue_video_out_flip(state, handle, buffer_index, flip_argument);
                    }
                    for event_type in event_writes {
                        queue_agc_event_write(state, event_type);
                    }
                }
                Err(error) => {
                    eprintln!("[REMu][AGC] rejected DCB {index} in multi-submission: {error}");
                    return u64::MAX;
                }
            }
        }
    }
    0
}

fn agc_resolve_acb_descriptor(
    command_address: u64,
    dword_count: u32,
) -> Result<(u64, u32), AgcSubmissionError> {
    const ACB_DESCRIPTOR_MAGIC: u32 = 0x5533_ccaa;
    if dword_count < 5
        || read_guest_u32(command_address + 16) != ACB_DESCRIPTOR_MAGIC
        || read_guest_u32(command_address + 12) != 0
    {
        return Ok((command_address, dword_count));
    }
    let descriptor_address = read_guest_u64(command_address);
    let descriptor_dword_count = read_guest_u32(command_address + 8);
    if descriptor_address == 0 || descriptor_dword_count == 0 || descriptor_dword_count > 1_000_000
    {
        return Err(AgcSubmissionError::InvalidAcbDescriptor {
            address: descriptor_address,
            dword_count: descriptor_dword_count,
        });
    }
    Ok((descriptor_address, descriptor_dword_count))
}

fn agc_submit_compute_commands(
    owner: u32,
    command_address: u64,
    dword_count: u32,
    state: &mut ProbeDispatchState,
    flips: &mut Vec<(u32, u32, u64)>,
    event_writes: &mut Vec<u32>,
) -> Result<(), AgcSubmissionError> {
    let (command_address, dword_count) = agc_resolve_acb_descriptor(command_address, dword_count)?;
    let graphics = state.agc_compute_queues.entry(owner).or_default();
    agc_parse_submission(
        command_address,
        dword_count,
        &mut AgcSubmissionContext {
            graphics,
            shaders: &mut state.agc_shader_registry,
            index_resources: &state.agc_index_resources,
            video_out: &state.video_out,
            submission_count: &mut state.agc_submission_count,
            flips,
            event_writes,
        },
    )
}

pub(crate) fn agc_driver_submit_acb(arguments: [u64; 6], state: &mut ProbeDispatchState) -> u64 {
    let owner = low_u32(arguments[0]);
    let packet = arguments[1];
    if packet == 0 {
        eprintln!("[REMu][AGC] rejected ACB submission for owner {owner}: null packet");
        return u64::MAX;
    }
    let command_address = read_guest_u64(packet);
    let dword_count = read_guest_u32(packet + 8);
    if command_address == 0 || dword_count == 0 || dword_count > 1_000_000 {
        // A submission the driver refuses is fatal to the guest, so describe the packet it
        // rejected rather than returning a bare error the title can only assert on.
        eprintln!(
            "[REMu][AGC] rejected ACB submission for owner {owner}: packet={packet:#x} commands={command_address:#x} dwords={dword_count} args={:#x},{:#x},{:#x},{:#x}",
            arguments[2], arguments[3], arguments[4], arguments[5]
        );
        return u64::MAX;
    }
    let next_submission = state.agc_submission_count.saturating_add(1);
    if trace_agc_programs() || trace_logarithmic_sequence(next_submission, 8) {
        eprintln!(
            "[REMu][AGC] queue=async-compute owner={owner} submit={next_submission} address={command_address:#x} dwords={dword_count}"
        );
    }
    let mut flips = Vec::new();
    let mut event_writes = Vec::new();
    let (result, gpu_events) = capture_agc_gpu_events(|| {
        agc_submit_compute_commands(
            owner,
            command_address,
            dword_count,
            state,
            &mut flips,
            &mut event_writes,
        )
    });
    match result {
        Ok(()) => {
            // The native driver asks the compute backend to trigger an interrupt when the
            // command buffer does not already contain one. SubmissionBoundary is our backend
            // flush/completion point, so every accepted ACB needs one just like a DCB does.
            // Without it, the last compute batch remains open and the guest waits forever for
            // the queue completion before it can submit the next frame.
            if !request_agc_compute_submission_completion(state, u64::from(owner), gpu_events) {
                eprintln!(
                    "[REMu][AGC] rejected ACB submission for owner {owner}: presenter is unavailable"
                );
                return u64::MAX;
            }
            for (handle, buffer_index, flip_argument) in flips {
                queue_video_out_flip(state, handle, buffer_index, flip_argument);
            }
            for event_type in event_writes {
                queue_agc_event_write(state, event_type);
            }
            0
        }
        Err(error) => {
            eprintln!("[REMu][AGC] rejected ACB submission for owner {owner}: {error}");
            u64::MAX
        }
    }
}

pub(crate) fn agc_driver_submit_multi_acbs(
    arguments: [u64; 6],
    state: &mut ProbeDispatchState,
) -> u64 {
    let owner = low_u32(arguments[0]);
    let addresses = arguments[1];
    let sizes = arguments[2];
    let count = low_u32(arguments[3]);
    if addresses == 0 || sizes == 0 || count == 0 || count > 4096 {
        return u64::MAX;
    }
    for index in 0..count {
        let command_address = read_guest_u64(addresses + u64::from(index) * 8);
        let dword_count = read_guest_u32(sizes + u64::from(index) * 4);
        if command_address == 0 || dword_count == 0 || dword_count > 1_000_000 {
            return u64::MAX;
        }
        let mut flips = Vec::new();
        let mut event_writes = Vec::new();
        let (result, gpu_events) = capture_agc_gpu_events(|| {
            agc_submit_compute_commands(
                owner,
                command_address,
                dword_count,
                state,
                &mut flips,
                &mut event_writes,
            )
        });
        match result {
            Ok(()) => {
                if !request_agc_compute_submission_completion(state, u64::from(owner), gpu_events) {
                    eprintln!(
                        "[REMu][AGC] rejected ACB {index} for owner {owner}: presenter is unavailable"
                    );
                    return u64::MAX;
                }
                for (handle, buffer_index, flip_argument) in flips {
                    queue_video_out_flip(state, handle, buffer_index, flip_argument);
                }
                for event_type in event_writes {
                    queue_agc_event_write(state, event_type);
                }
            }
            Err(error) => {
                eprintln!(
                    "[REMu][AGC] rejected ACB {index} for owner {owner} in multi-submission: {error}"
                );
                return u64::MAX;
            }
        }
    }
    0
}
