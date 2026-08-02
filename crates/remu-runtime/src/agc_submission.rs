use std::sync::atomic::{AtomicUsize, Ordering};

use remu_shader::GuestMemoryReader;

use crate::{
    AGC_IT_CLEAR_STATE, AGC_IT_DISPATCH_DIRECT, AGC_IT_DISPATCH_INDIRECT, AGC_IT_DMA_DATA,
    AGC_IT_DRAW_INDEX_2, AGC_IT_DRAW_INDEX_AUTO, AGC_IT_DRAW_INDEX_INDIRECT,
    AGC_IT_DRAW_INDEX_INDIRECT_MULTI, AGC_IT_DRAW_INDEX_OFFSET, AGC_IT_DRAW_INDIRECT,
    AGC_IT_EVENT_WRITE, AGC_IT_INDEX_BASE, AGC_IT_INDEX_BUFFER_SIZE, AGC_IT_INDEX_TYPE, AGC_IT_NOP,
    AGC_IT_NUM_INSTANCES, AGC_IT_PFP_SYNC_ME, AGC_IT_RELEASE_MEM, AGC_IT_SET_BASE,
    AGC_IT_WAIT_REG_MEM, AGC_IT_WRITE_DATA, AGC_R_ACB_RESET, AGC_R_ACQUIRE_MEM, AGC_R_DMA_DATA,
    AGC_R_DRAW_INDEX_AUTO, AGC_R_DRAW_RESET, AGC_R_FLIP, AGC_R_RELEASE_MEM, AGC_R_WAIT_MEM_32,
    AGC_R_WAIT_MEM_64, AGC_R_WRITE_DATA, AgcComputeDispatchError, AgcGraphicsState,
    AgcShaderRegistry, AgcSubmissionError, AgcVideoOutTarget, ProcessGuestMemory, VideoOutState,
    agc_apply_set_base, agc_apply_submission_registers, agc_decode_compute_dispatch, agc_emit_draw,
    agc_emit_draw_with_indirect_arguments, agc_emit_indexed_draw, agc_image_binding_address,
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
    AgcEventWrite, decode_agc_dma_data, decode_agc_event_write, decode_agc_memory_acquire,
    decode_agc_memory_wait,
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

const INDIRECT_DRAW_REGISTER_UNMAPPED: u32 = 0x280;
const PIXEL_PIPE_STAT_VALID: u64 = 1 << 63;
const PIXEL_PIPE_STAT_STEP: u64 = 0x02ff_ffff;
/// Gen5 result blocks contain sixteen 16-byte render-backend counter pairs.
///
/// Native PS5 command streams clear the full 0x100-byte payload before issuing
/// `PIXEL_PIPE_STAT_DUMP`; every even qword is a counter and every odd qword is padding.
const PIXEL_PIPE_STAT_COUNTER_PAIRS: u64 = 16;

fn emit_agc_memory_release(packet_address: u64, release: remu_gpu::GpuMemoryRelease) {
    if trace_guest_sync()
        && let Some(trace) = remu_gpu::record_pixel_pipe_query_completion_release_trace(release)
    {
        eprintln!(
            "[REMu][AGC][PIXEL-PIPE][COMPLETE][EMIT] first_sequence={} second_sequence={} packet={packet_address:#x} payload={:#x} address={:#x} value=0x0000000000000001",
            trace.first_dump_sequence,
            trace.second_dump_sequence,
            trace.payload_address,
            trace.completion_address,
        );
    }
    emit_agc_gpu_event(remu_gpu::GpuEvent::MemoryRelease(release));
}

#[derive(Clone, Copy)]
struct AgcIndirectDrawArguments {
    element_count: u32,
    instance_count: u32,
    element_offset: u32,
    base_vertex: i32,
    first_instance: u32,
}

fn agc_read_indirect_draw_arguments(
    address: u64,
    indexed: bool,
) -> Result<AgcIndirectDrawArguments, AgcSubmissionError> {
    let byte_count = if indexed { 20 } else { 16 };
    let mut bytes = [0_u8; 20];
    if !ProcessGuestMemory.read(address, &mut bytes[..byte_count]) {
        return Err(AgcSubmissionError::UnreadableIndirectDrawArguments(address));
    }
    let word = |index: usize| {
        let offset = index * 4;
        u32::from_le_bytes(
            bytes[offset..offset + 4]
                .try_into()
                .expect("indirect draw argument word is four bytes"),
        )
    };
    Ok(AgcIndirectDrawArguments {
        element_count: word(0),
        instance_count: word(1),
        element_offset: word(2),
        base_vertex: if indexed { word(3) as i32 } else { 0 },
        first_instance: word(if indexed { 4 } else { 3 }),
    })
}

fn agc_emit_indirect_draw(
    packet_address: u64,
    graphics: &mut AgcGraphicsState,
    shaders: &mut AgcShaderRegistry,
    video_out: &VideoOutState,
    indexed: bool,
    arguments: AgcIndirectDrawArguments,
    base_vertex_register: u32,
    first_element_register: Option<u32>,
    first_instance_register: u32,
    draw_flags: u32,
    arguments_address: u64,
    future_scanout_target: Option<AgcVideoOutTarget>,
) {
    static INDIRECT_DRAW_COUNT: AtomicUsize = AtomicUsize::new(0);
    let sequence = INDIRECT_DRAW_COUNT.fetch_add(1, Ordering::Relaxed) + 1;
    if trace_agc_programs()
        || sequence <= 8
        || sequence.is_power_of_two()
        || arguments.element_count > 1_048_576
        || arguments.instance_count > 65_536
    {
        let first_element_register = first_element_register.map_or_else(
            || "unmapped".to_owned(),
            |register| format!("{register:#x}"),
        );
        eprintln!(
            "[REMu][AGC][INDIRECT-DRAW] sequence={sequence} packet={packet_address:#x} indexed={indexed} arguments={arguments_address:#x} observed=[count={},instances={},first={},base_vertex={},base_instance={}] registers=[base={base_vertex_register:#x},first={first_element_register},instance={first_instance_register:#x}] flags={draw_flags:#x}",
            arguments.element_count,
            arguments.instance_count,
            arguments.element_offset,
            arguments.base_vertex,
            arguments.first_instance,
        );
    }
    // The packet-time CPU observation can precede a GPU producer of these dwords. Preserve the
    // event even when it presently looks empty and let the backend consume the argument buffer at
    // its ordered execution point.
    //
    // Native indirect drawing applies first vertex/base vertex and first instance to the shader
    // built-ins. Keep the corresponding translated scalar inputs neutral instead of baking this
    // potentially stale CPU observation into the shader metadata as well. These are per-draw
    // command-processor patches, so restore the persistent register file afterwards.
    let mut neutralized_registers = Vec::with_capacity(3);
    for register in std::iter::once(base_vertex_register)
        .chain(first_element_register)
        .chain(std::iter::once(first_instance_register))
    {
        if register != INDIRECT_DRAW_REGISTER_UNMAPPED
            && !neutralized_registers
                .iter()
                .any(|&(saved, _)| saved == register)
        {
            neutralized_registers.push((register, graphics.shader_registers.insert(register, 0)));
        }
    }

    let previous_instance_count =
        std::mem::replace(&mut graphics.instance_count, arguments.instance_count);
    let previous_base_vertex = graphics
        .context_registers
        .insert(0x102, arguments.base_vertex as u32);
    agc_emit_draw_with_indirect_arguments(
        packet_address,
        graphics,
        shaders,
        video_out,
        indexed,
        arguments.element_offset,
        arguments.element_count,
        draw_flags,
        future_scanout_target,
        Some(arguments_address),
    );
    graphics.instance_count = previous_instance_count;
    if let Some(value) = previous_base_vertex {
        graphics.context_registers.insert(0x102, value);
    } else {
        graphics.context_registers.remove(&0x102);
    }
    for (register, value) in neutralized_registers {
        if let Some(value) = value {
            graphics.shader_registers.insert(register, value);
        } else {
            graphics.shader_registers.remove(&register);
        }
    }
}

fn agc_indirect_draw_argument_address(
    graphics: &AgcGraphicsState,
    data_offset: u32,
) -> Result<u64, AgcSubmissionError> {
    if graphics.indirect_args_address == 0 {
        return Err(AgcSubmissionError::MissingGraphicsIndirectBase);
    }
    graphics
        .indirect_args_address
        .checked_add(u64::from(data_offset))
        .ok_or(AgcSubmissionError::IndirectDrawAddressOverflow)
}

fn agc_handle_indirect_draw(
    packet: AgcSubmissionPacket,
    graphics: &mut AgcGraphicsState,
    shaders: &mut AgcShaderRegistry,
    video_out: &VideoOutState,
    future_scanout_target: Option<AgcVideoOutTarget>,
) -> Result<(), AgcSubmissionError> {
    if packet.opcode == AGC_IT_DRAW_INDEX_INDIRECT_MULTI {
        return Err(AgcSubmissionError::UnsupportedIndirectDrawPacket(
            packet.opcode,
        ));
    }
    let indexed = packet.opcode != AGC_IT_DRAW_INDIRECT;
    if packet.length != 5 {
        return Err(AgcSubmissionError::IndirectDrawPacketLength {
            opcode: packet.opcode,
            actual: packet.length,
            expected: 5,
        });
    }
    let data_offset = read_guest_u32(packet.address + 4);
    let register_word = read_guest_u32(packet.address + 8);
    let base_vertex_register = register_word & 0xffff;
    let first_element_register = indexed.then_some(register_word >> 16);
    let first_instance_register = read_guest_u32(packet.address + 12) & 0xffff;
    let draw_flags = read_guest_u32(packet.address + 16);
    let address = agc_indirect_draw_argument_address(graphics, data_offset)?;
    let arguments = agc_read_indirect_draw_arguments(address, indexed)?;
    agc_emit_indirect_draw(
        packet.address,
        graphics,
        shaders,
        video_out,
        indexed,
        arguments,
        base_vertex_register,
        first_element_register,
        first_instance_register,
        draw_flags,
        address,
        future_scanout_target,
    );
    Ok(())
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
    if opcode == AGC_IT_PFP_SYNC_ME {
        if length != 2 {
            return Err(AgcSubmissionError::PfpSyncMePacketLength { actual: length });
        }
        // AMD's parser front end can fetch indirect arguments before earlier micro-engine work
        // has produced them unless this packet stalls it. Preserve that dependency in the ordered
        // backend stream; it is a GPU queue barrier, not a CPU-visible completion.
        emit_agc_gpu_event(remu_gpu::GpuEvent::CommandProcessorBarrier);
    }
    if opcode == AGC_IT_EVENT_WRITE {
        match decode_agc_event_write(packet_address, length)? {
            Some(AgcEventWrite::PipelineBarrier(barrier)) => {
                emit_agc_gpu_event(remu_gpu::GpuEvent::PipelineBarrier(barrier));
            }
            Some(AgcEventWrite::PixelPipeStatDump { destination }) => {
                let value = graphics.pixel_pipe_stat_counter | PIXEL_PIPE_STAT_VALID;
                static PIXEL_PIPE_STAT_TRACE_SEQUENCE: AtomicUsize = AtomicUsize::new(0);
                let trace_sequence = trace_guest_sync()
                    .then(|| PIXEL_PIPE_STAT_TRACE_SEQUENCE.fetch_add(1, Ordering::Relaxed) + 1);
                let trace_query =
                    trace_sequence.filter(|sequence| *sequence <= 32 || sequence.is_power_of_two());
                if let Some(trace_sequence) = trace_sequence {
                    remu_gpu::record_pixel_pipe_stat_dump_trace(
                        u64::try_from(trace_sequence).unwrap_or(u64::MAX),
                        destination,
                        trace_query.is_some(),
                    );
                }
                if let Some(trace_sequence) = trace_query {
                    eprintln!(
                        "[REMu][AGC][PIXEL-PIPE][EMIT] sequence={trace_sequence} packet={packet_address:#x} destination={destination:#x} value={value:#018x} pairs={PIXEL_PIPE_STAT_COUNTER_PAIRS}"
                    );
                }
                for pair in 0..PIXEL_PIPE_STAT_COUNTER_PAIRS {
                    let release = remu_gpu::GpuMemoryRelease {
                        address: destination + pair * 16,
                        value: remu_gpu::GpuMemoryReleaseValue::Immediate64(value),
                    };
                    if let Some(trace_sequence) = trace_query {
                        remu_gpu::record_pixel_pipe_stat_release_trace(
                            release,
                            remu_gpu::PixelPipeStatReleaseTrace {
                                dump_sequence: u64::try_from(trace_sequence).unwrap_or(u64::MAX),
                                pair_index: u8::try_from(pair).expect(
                                    "pixel-pipe counter-pair count fits the diagnostic index",
                                ),
                            },
                        );
                    }
                    emit_agc_gpu_event(remu_gpu::GpuEvent::MemoryRelease(release));
                }
                graphics.pixel_pipe_stat_counter = graphics
                    .pixel_pipe_stat_counter
                    .wrapping_add(PIXEL_PIPE_STAT_STEP)
                    & !PIXEL_PIPE_STAT_VALID;
            }
            None => {}
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
        let mut copy = copy;
        let staged = stage_agc_memory_copy(&copy).map_err(|error| {
            AgcSubmissionError::MemorySnapshotBudget {
                requested_bytes: error.requested_bytes,
                limit_bytes: error.limit_bytes,
            }
        })?;
        copy.source_snapshot = staged.source_snapshot;
        static DMA_TRACE_COUNT: AtomicUsize = AtomicUsize::new(0);
        static DMA_SNAPSHOT_COUNT: AtomicUsize = AtomicUsize::new(0);
        static DMA_SNAPSHOT_BYTES: AtomicUsize = AtomicUsize::new(0);
        let sequence = DMA_TRACE_COUNT.fetch_add(1, Ordering::Relaxed) + 1;
        let snapshot_bytes = copy
            .source_snapshot
            .as_ref()
            .map_or(0, |snapshot| snapshot.len());
        if snapshot_bytes != 0 {
            DMA_SNAPSHOT_COUNT.fetch_add(1, Ordering::Relaxed);
            DMA_SNAPSHOT_BYTES.fetch_add(snapshot_bytes, Ordering::Relaxed);
        }
        let fence_label_dma = (0x7e00_0000_0000..0x8000_0000_0000)
            .contains(&copy.destination_address);
        if fence_label_dma
            || trace_guest_sync()
            || trace_agc_programs()
            || sequence <= 32
            || sequence.is_power_of_two()
            || graphics
                .sampled_texture_addresses
                .contains(&copy.destination_address)
        {
            eprintln!(
                "[REMu][AGC][DMA] sequence={sequence} packet={packet_address:#x} standard={} src_sel={} dst_sel={} source={:#x} destination={:#x} bytes={:#x} shadowed={} snapshot_bytes={} snapshot_count={} snapshot_cumulative_bytes={} snapshot_retained={:?}",
                opcode == AGC_IT_DMA_DATA,
                copy.source_selection,
                copy.destination_selection,
                copy.source_address,
                copy.destination_address,
                copy.byte_count,
                staged.shadowed,
                snapshot_bytes,
                DMA_SNAPSHOT_COUNT.load(Ordering::Relaxed),
                DMA_SNAPSHOT_BYTES.load(Ordering::Relaxed),
                remu_gpu::gpu_memory_copy_snapshot_statistics(),
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
        emit_agc_memory_release(packet_address, release);
    }
    if opcode == AGC_IT_RELEASE_MEM
        && length >= 8
        && let Some(release) = decode_agc_memory_release(packet_address, true)
    {
        emit_agc_memory_release(packet_address, release);
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
    if matches!(
        opcode,
        AGC_IT_DRAW_INDIRECT | AGC_IT_DRAW_INDEX_INDIRECT | AGC_IT_DRAW_INDEX_INDIRECT_MULTI
    ) {
        agc_handle_indirect_draw(packet, graphics, shaders, video_out, future_scanout_target)?;
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
        agc_handle_compute_dispatch_packet(packet_address, length, opcode, graphics, shaders)?;
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

/// Report a dispatch REmu executed nothing for. Rate-limited per shader address so a dispatch
/// repeated every frame cannot flood the log while every distinct shader is still named once.
fn report_skipped_compute_dispatch(packet_address: u64, source: &AgcComputeDispatchError) {
    static REPORTED: std::sync::Mutex<Option<std::collections::HashMap<u64, u64>>> =
        std::sync::Mutex::new(None);
    let shader_address = source.shader_address().unwrap_or_default();
    let Ok(mut reported) = REPORTED.lock() else {
        return;
    };
    let occurrences = reported
        .get_or_insert_with(std::collections::HashMap::new)
        .entry(shader_address)
        .or_default();
    *occurrences = occurrences.saturating_add(1);
    let occurrences = *occurrences;
    drop(reported);
    if !trace_logarithmic_sequence(occurrences, 1) {
        return;
    }
    eprintln!(
        "[REMu][AGC] skipped compute dispatch at {packet_address:#018x} (occurrence \
         {occurrences}): {source}. The rest of the submission still executes."
    );
}

fn agc_handle_compute_dispatch_packet(
    packet_address: u64,
    length: u32,
    opcode: u32,
    graphics: &mut AgcGraphicsState,
    shaders: &mut AgcShaderRegistry,
) -> Result<(), AgcSubmissionError> {
    let dispatch =
        match agc_decode_compute_dispatch(packet_address, length, opcode, graphics, shaders) {
            Ok(dispatch) => dispatch,
            // One shader REmu cannot decode or evaluate is a capability gap in this emulator, not
            // a malformed command stream: the packets around it are intact and executable. Unreal
            // turns a failed graphics submit into LowLevelFatalError, so rejecting the whole DCB
            // over one dispatch kills the title over work the guest never depended on being
            // rejected. Skip only this dispatch; every other error class still rejects.
            Err(source) if source.shader_is_unsupported() => {
                report_skipped_compute_dispatch(packet_address, &source);
                return Ok(());
            }
            Err(source) => {
                return Err(AgcSubmissionError::Compute {
                    packet_address,
                    source,
                });
            }
        };
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
    // Direct dimensions are immutable packet data, so a zero extent is genuinely empty.
    // Indirect dwords can be produced by earlier GPU work and the packet-time CPU snapshot
    // can still be zero here. Preserve that event so a backend with native indirect dispatch
    // reads the live dimensions at its ordered execution point.
    if dispatch.launch.is_empty()
        && matches!(dispatch.source, remu_gpu::ComputeDispatchSource::Direct)
    {
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
    Ok(())
}
