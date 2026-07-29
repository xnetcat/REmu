use std::sync::atomic::{AtomicUsize, Ordering};

use crate::{
    AGC_IT_CLEAR_STATE, AGC_IT_DISPATCH_DIRECT, AGC_IT_DISPATCH_INDIRECT, AGC_IT_DMA_DATA,
    AGC_IT_DRAW_INDEX_2, AGC_IT_DRAW_INDEX_AUTO, AGC_IT_DRAW_INDEX_OFFSET, AGC_IT_EVENT_WRITE,
    AGC_IT_INDEX_BASE, AGC_IT_INDEX_BUFFER_SIZE, AGC_IT_INDEX_TYPE, AGC_IT_NOP,
    AGC_IT_NUM_INSTANCES, AGC_IT_RELEASE_MEM, AGC_IT_SET_BASE, AGC_IT_WAIT_REG_MEM,
    AGC_IT_WRITE_DATA, AGC_R_ACB_RESET, AGC_R_ACQUIRE_MEM, AGC_R_DMA_DATA, AGC_R_DRAW_INDEX_AUTO,
    AGC_R_DRAW_RESET, AGC_R_FLIP, AGC_R_RELEASE_MEM, AGC_R_WAIT_MEM_32, AGC_R_WAIT_MEM_64,
    AGC_R_WRITE_DATA, AgcGraphicsState, AgcShaderRegistry, AgcSubmissionError, AgcVideoOutTarget,
    VideoOutState, agc_apply_set_base, agc_apply_submission_registers, agc_decode_compute_dispatch,
    agc_emit_draw, agc_emit_indexed_draw, agc_image_binding_address,
    agc_submit_ordered_video_out_flip, emit_agc_gpu_event, read_guest_u32, read_guest_u64,
    trace_agc_programs, trace_agc_registers, trace_agc_shaders, trace_guest_sync,
    trace_logarithmic_sequence,
};
mod decode;
mod driver;
mod flow;
mod parse;
mod shadow;

#[cfg(test)]
pub(crate) use self::decode::decode_agc_pipeline_barrier;
pub(crate) use self::decode::{
    decode_agc_dma_data, decode_agc_event_write, decode_agc_memory_acquire, decode_agc_memory_wait,
};
pub(crate) use self::driver::{
    agc_driver_submit_acb, agc_driver_submit_dcb, agc_driver_submit_multi_acbs,
    agc_driver_submit_multi_dcbs,
};
pub(crate) use self::flow::agc_read_mapped_u64;
#[cfg(test)]
pub(crate) use self::parse::agc_future_scanout_targets;
pub(crate) use self::parse::{
    AgcSubmissionContext, agc_parse_graphics_submission, agc_parse_submission,
};

use self::decode::{decode_agc_memory_release, decode_agc_memory_write};
use self::flow::{agc_apply_predication, agc_branch_target, agc_cond_exec_skip_end};
use self::shadow::stage_agc_memory_copy;
pub(crate) use self::shadow::{
    agc_memory_shadow_overlaps, apply_agc_memory_shadow, begin_agc_memory_shadow,
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

pub(crate) fn agc_handle_submission_packet(
    packet: AgcSubmissionPacket,
    graphics: &mut AgcGraphicsState,
    shaders: &mut AgcShaderRegistry,
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
        && let Some(copy) = decode_agc_dma_data(packet_address, length, opcode == AGC_IT_DMA_DATA)
    {
        // Descriptor and indirect-argument decoding happens synchronously, so following packets
        // need to observe this copy before the asynchronous backend reaches it. Stage that view
        // in the submission shadow without changing guest memory; the MemoryCopy event below is
        // the sole authoritative mutation and executes at this exact queue position.
        let shadowed = stage_agc_memory_copy(&copy);
        static DMA_TRACE_COUNT: AtomicUsize = AtomicUsize::new(0);
        let sequence = DMA_TRACE_COUNT.fetch_add(1, Ordering::Relaxed) + 1;
        if trace_guest_sync()
            || trace_agc_programs()
            || sequence <= 32
            || sequence.is_power_of_two()
            || graphics
                .sampled_texture_addresses
                .contains(&copy.destination_address)
        {
            eprintln!(
                "[REMu][AGC][DMA] sequence={sequence} packet={packet_address:#x} standard={} src_sel={} dst_sel={} source={:#x} destination={:#x} bytes={:#x} shadowed={shadowed}",
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
        graphics.index_resource_address = graphics.index_buffer_address;
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
        graphics.index_resource_address = graphics.index_buffer_address;
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
        if trace_agc_programs() || trace_logarithmic_sequence(dispatch.sequence, 8) {
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
