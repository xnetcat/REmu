use std::collections::HashSet;
use std::sync::{Mutex, OnceLock};

use remu_shader::GuestMemoryReader;

use crate::{
    AGC_CB_SHADER_MASK, AGC_CB_TARGET_MASK, AGC_COLOR_TARGET_COUNT, AGC_COMPUTE_USER_DATA_0,
    AGC_IT_DISPATCH_DIRECT, AGC_SPI_SHADER_PGM_HI_ES, AGC_SPI_SHADER_PGM_HI_PS,
    AGC_SPI_SHADER_PGM_LO_ES, AGC_SPI_SHADER_PGM_LO_PS, AGC_VGT_PRIMITIVE_TYPE,
    AgcColorTargetState, AgcComputeDispatchError, AgcGraphicsState, AgcShaderRegistry,
    AgcSubmissionError, AgcVideoOutTarget, ProcessGuestMemory, ShaderEvaluationWorkers,
    VideoOutState, agc_color_target_state, agc_has_active_depth_state, agc_program_address,
    agc_register_snapshot, agc_shader_prefix_snapshot, agc_shader_snapshot,
    agc_video_out_target_for_address, draw_required_vertex_record_count, emit_agc_gpu_event,
    evaluate_draw_shader_requests_parallel, evaluate_shader_register_snapshot_cached,
    export_user_data_register, read_guest_u32, shader_control_flow_summary,
    shader_evaluation_worker_request, snapshot_draw_indices,
    snapshot_small_readonly_global_bindings, trace_agc_draws, trace_agc_programs,
    trace_agc_shader_draw, trace_agc_shaders, trace_agc_target_address, trace_logarithmic_sequence,
};

fn agc_packet_words(packet_address: u64, length: u32) -> Vec<u32> {
    (0..length)
        .map(|index| read_guest_u32(packet_address + u64::from(index) * 4))
        .collect()
}

fn agc_read_compute_dimensions(address: u64) -> Result<[u32; 3], AgcComputeDispatchError> {
    let mut bytes = [0_u8; 12];
    address
        .checked_add(u64::try_from(bytes.len()).expect("dimension byte count fits u64"))
        .ok_or(AgcComputeDispatchError::IndirectAddressOverflow)?;
    if !ProcessGuestMemory.read(address, &mut bytes) {
        return Err(AgcComputeDispatchError::UnreadableIndirectDimensions(
            address,
        ));
    }
    Ok(std::array::from_fn(|axis| {
        let start = axis * 4;
        u32::from_le_bytes(
            bytes[start..start + 4]
                .try_into()
                .expect("dimension chunk is four bytes"),
        )
    }))
}

pub(crate) fn agc_apply_set_base(
    packet_address: u64,
    length: u32,
    graphics: &mut AgcGraphicsState,
) -> Result<(), AgcSubmissionError> {
    if length != 4 {
        return Err(AgcSubmissionError::SetBasePacketLength { actual: length });
    }
    let base_index = read_guest_u32(packet_address + 4) & 0xf;
    if base_index != 1 {
        return Err(AgcSubmissionError::UnsupportedBaseIndex(base_index));
    }
    let header = read_guest_u32(packet_address);
    let shader_type = (header >> 1) & 0x3;
    if shader_type > 1 {
        return Err(AgcSubmissionError::UnsupportedSetBaseShaderType(
            shader_type,
        ));
    }
    if shader_type == 1 {
        let low = read_guest_u32(packet_address + 8) & !0x7;
        let high = read_guest_u32(packet_address + 12) & 0xffff;
        graphics.compute_indirect_args_address = u64::from(low) | (u64::from(high) << 32);
    }
    Ok(())
}

pub(crate) fn agc_decode_compute_dispatch(
    packet_address: u64,
    length: u32,
    opcode: u32,
    graphics: &mut AgcGraphicsState,
    shaders: &mut AgcShaderRegistry,
) -> Result<remu_gpu::ComputeDispatch, AgcComputeDispatchError> {
    let packet = agc_packet_words(packet_address, length);
    let shader_registers = agc_register_snapshot(&graphics.shader_registers);
    let (source, launch) = if opcode == AGC_IT_DISPATCH_DIRECT {
        (
            remu_gpu::ComputeDispatchSource::Direct,
            remu_gpu::decode_compute_dispatch_direct(&packet, &shader_registers)?,
        )
    } else {
        let indirect = remu_gpu::decode_compute_dispatch_indirect(&packet)?;
        let dimensions_address = match indirect.dimensions {
            remu_gpu::ComputeIndirectDimensions::BaseOffset(offset) => {
                if graphics.compute_indirect_args_address == 0 {
                    return Err(AgcComputeDispatchError::MissingIndirectBase);
                }
                graphics
                    .compute_indirect_args_address
                    .checked_add(u64::from(offset))
                    .ok_or(AgcComputeDispatchError::IndirectAddressOverflow)?
            }
            remu_gpu::ComputeIndirectDimensions::Absolute(address) => address,
        };
        let dimensions = agc_read_compute_dimensions(dimensions_address)?;
        (
            remu_gpu::ComputeDispatchSource::Indirect { dimensions_address },
            remu_gpu::decode_compute_launch(dimensions, indirect.initiator, &shader_registers)?,
        )
    };

    let shader_address = remu_gpu::compute_program_address(&shader_registers)
        .ok_or(AgcComputeDispatchError::MissingProgramAddress)?;
    if shader_address == 0 {
        return Err(AgcComputeDispatchError::NullProgramAddress);
    }
    let shader = agc_shader_snapshot(shaders, shader_address)
        .ok_or(AgcComputeDispatchError::ShaderDecode(shader_address))?;
    let compute_system_registers = remu_gpu::decode_compute_system_registers(&shader_registers);
    let shader_evaluation = evaluate_shader_register_snapshot_cached(
        &shader_registers,
        &shader,
        AGC_COMPUTE_USER_DATA_0,
        0,
        Some(compute_system_registers),
        remu_shader::EvaluationOptions {
            capture_global_binding_data: false,
            ..remu_shader::EvaluationOptions::default()
        },
        &mut graphics.shader_evaluation_cache,
    )
    .and_then(snapshot_small_readonly_global_bindings)
    .map_err(|reason| AgcComputeDispatchError::ShaderEvaluation {
        address: shader_address,
        reason: format!("{reason}; {}", shader_control_flow_summary(&shader.program)),
    })?;
    graphics.emitted_compute_dispatches = graphics.emitted_compute_dispatches.saturating_add(1);
    Ok(remu_gpu::ComputeDispatch {
        sequence: graphics.emitted_compute_dispatches,
        source,
        launch,
        shader_address,
        shader_prefix: agc_shader_prefix_snapshot(shader_address),
        shader: Some(shader),
        shader_evaluation: Some(shader_evaluation),
        shader_registers,
    })
}

pub(crate) fn agc_prepare_draw_color_targets(
    graphics: &mut AgcGraphicsState,
) -> (Vec<remu_gpu::GpuColorTarget>, bool) {
    let (targets, absent) = match agc_color_target_state(&graphics.context_registers) {
        AgcColorTargetState::Absent => (Vec::new(), true),
        AgcColorTargetState::Disabled => (Vec::new(), false),
        AgcColorTargetState::Programmed(targets) => (targets, false),
        AgcColorTargetState::Malformed { slot } => {
            if trace_logarithmic_sequence(graphics.emitted_draws, 8) {
                eprintln!(
                    "[REMu][AGC] draw={} has incomplete enabled CB_COLOR{slot} state",
                    graphics.emitted_draws
                );
            }
            (Vec::new(), false)
        }
    };
    for target in &targets {
        graphics.known_color_targets.insert(target.address, *target);
    }
    (targets, absent)
}

fn agc_draw_writes_image(draw: &remu_gpu::IndexedDraw) -> bool {
    [
        draw.export_shader_evaluation.as_deref(),
        draw.pixel_shader_evaluation.as_deref(),
    ]
    .into_iter()
    .flatten()
    .flat_map(|evaluation| &evaluation.image_bindings)
    .any(|binding| {
        binding.opcode.starts_with("ImageStore") || binding.opcode.starts_with("ImageAtomic")
    })
}

fn agc_draw_reads_image(draw: &remu_gpu::IndexedDraw) -> bool {
    [
        draw.export_shader_evaluation.as_deref(),
        draw.pixel_shader_evaluation.as_deref(),
    ]
    .into_iter()
    .flatten()
    .flat_map(|evaluation| &evaluation.image_bindings)
    .any(|binding| {
        !binding.opcode.starts_with("ImageStore") && !binding.opcode.starts_with("ImageAtomic")
    })
}

fn agc_shader_has_memory_side_effects(shader: Option<&remu_gpu::ShaderSnapshot>) -> bool {
    shader.is_some_and(|shader| {
        shader.program.instructions.iter().any(|instruction| {
            instruction.opcode.contains("Store") || instruction.opcode.contains("Atomic")
        })
    })
}

fn agc_draw_resources_evaluated(draw: &remu_gpu::IndexedDraw) -> bool {
    (draw.export_shader.is_none() || draw.export_shader_evaluation.is_some())
        && (draw.pixel_shader.is_none() || draw.pixel_shader_evaluation.is_some())
}

pub(crate) fn agc_image_binding_address(binding: &remu_shader::ImageBinding) -> Option<u64> {
    let low = u64::from(*binding.resource_descriptor.first()?);
    let high = u64::from(*binding.resource_descriptor.get(1)? & 0xff);
    Some(((high << 32) | low) << 8)
}

fn agc_is_large_linear_video_binding(binding: &remu_shader::ImageBinding) -> bool {
    let words = binding.resource_descriptor.as_slice();
    if words.len() < 4 {
        return false;
    }
    let width = ((words[1] >> 30) | ((words[2] & 0x3fff) << 2)).saturating_add(1);
    let height = ((words[2] >> 14) & 0xffff).saturating_add(1);
    let tile_mode = (words[3] >> 20) & 0x1f;
    let unified_format = (words[1] >> 20) & 0x1ff;
    let data_format = remu_shader::decode_unified_format(unified_format)
        .map_or(u32::MAX, |(data_format, _)| data_format);
    tile_mode == 0
        && width >= 700
        && height >= 500
        && matches!(data_format, 1 | 3)
        && (binding.opcode.starts_with("ImageSample") || binding.opcode.starts_with("ImageGather"))
}

pub(crate) fn agc_resolve_targetless_scanout(
    graphics: &mut AgcGraphicsState,
    draw: &mut remu_gpu::IndexedDraw,
    color_target_absent: bool,
    future_target: Option<AgcVideoOutTarget>,
) {
    if !color_target_absent
        || agc_has_active_depth_state(&graphics.context_registers)
        || !agc_draw_resources_evaluated(draw)
        || !agc_draw_reads_image(draw)
        || agc_draw_writes_image(draw)
    {
        return;
    }
    let Some(target) = future_target else {
        return;
    };
    let export_masks = draw
        .pixel_shader
        .as_ref()
        .map_or(0, |shader| shader.program.pixel_color_export_masks())
        & draw
            .context_registers
            .iter()
            .find(|register| register.offset == AGC_CB_SHADER_MASK)
            .map_or(u32::MAX, |register| register.value);
    let export_slots = (0..AGC_COLOR_TARGET_COUNT)
        .filter(|slot| (export_masks >> (slot * 4)) & 0xf != 0)
        .collect::<Vec<_>>();
    let [export_slot] = export_slots.as_slice() else {
        if trace_logarithmic_sequence(draw.sequence, 8) {
            eprintln!(
                "[REMu][AGC] did not associate targetless draw={}: future flip target {:#x} has ambiguous pixel exports {export_slots:?}",
                draw.sequence, target.address,
            );
        }
        return;
    };
    let color_surface = graphics
        .known_color_targets
        .get(&target.address)
        .copied()
        .filter(|surface| {
            ((surface.attrib2 >> 14) & 0x3fff) + 1 == target.attributes.width
                && (surface.attrib2 & 0x3fff) + 1 == target.attributes.height
        });
    draw.scanout_target = Some(remu_gpu::GpuScanoutTarget {
        address: target.address,
        width: target.attributes.width,
        height: target.attributes.height,
        pitch: target.attributes.pitch,
        pixel_format: target.attributes.pixel_format,
        tiling_mode: target.attributes.tiling_mode,
        export_slot: *export_slot,
        write_mask: (export_masks >> (*export_slot * 4)) & 0xf,
        color_surface,
    });
    if trace_logarithmic_sequence(draw.sequence, 8) {
        eprintln!(
            "[REMu][AGC] resolved targetless draw={} against future scanout target={:#x} programmed_surface={}",
            draw.sequence,
            target.address,
            color_surface.is_some(),
        );
    }
}

fn agc_trace_draw_summary(draw: &remu_gpu::IndexedDraw) {
    if trace_agc_draws() || trace_logarithmic_sequence(draw.sequence, 8) {
        let export_user_data_base = export_user_data_register(draw);
        let export_user_data = std::array::from_fn::<_, 8, _>(|index| {
            draw.shader_registers
                .iter()
                .find(|register| {
                    register.offset
                        == export_user_data_base
                            + u32::try_from(index).expect("export scalar index fits")
                })
                .map(|register| register.value)
        });
        let pixel_user_data = std::array::from_fn::<_, 8, _>(|index| {
            draw.shader_registers
                .iter()
                .find(|register| {
                    register.offset
                        == 0x00c + u32::try_from(index).expect("pixel scalar index fits")
                })
                .map(|register| register.value)
        });
        let viewport = [0x10f, 0x110, 0x111, 0x112].map(|offset| {
            draw.context_registers
                .iter()
                .find(|register| register.offset == offset)
                .map(|register| register.value)
        });
        eprintln!(
            "[REMu][AGC] draw={} indexed={} count={} instances={} offset={} index_buffer={:#x}/{} resource={:#x} index_size={} primitive={:#x} es={:#x} ps={:#x} target={:#x} scanout={:#x} target_mask={:#x} viewport={viewport:x?} es_user_base={export_user_data_base:#x} es_user={export_user_data:x?} ps_user={pixel_user_data:x?}",
            draw.sequence,
            draw.indexed,
            draw.index_count,
            draw.instance_count,
            draw.index_offset,
            draw.index_buffer_address,
            draw.index_buffer_count,
            draw.index_resource_address,
            draw.index_size,
            draw.primitive_type,
            draw.export_shader_address,
            draw.pixel_shader_address,
            draw.color_target_address,
            draw.scanout_target.map_or(0, |target| target.address),
            draw.context_registers
                .iter()
                .find(|register| register.offset == AGC_CB_TARGET_MASK)
                .map_or(0, |register| register.value),
        );
    }
}

pub(crate) fn agc_emit_draw(
    packet_address: u64,
    graphics: &mut AgcGraphicsState,
    shaders: &mut AgcShaderRegistry,
    video_out: &VideoOutState,
    indexed: bool,
    element_offset: u32,
    element_count: u32,
    draw_flags: u32,
    future_scanout_target: Option<AgcVideoOutTarget>,
) {
    if element_count == 0 {
        return;
    }
    graphics.draw_index_offset = element_offset;
    graphics.emitted_draws = graphics.emitted_draws.saturating_add(1);

    let export_shader_address = agc_program_address(
        &graphics.shader_registers,
        AGC_SPI_SHADER_PGM_LO_ES,
        AGC_SPI_SHADER_PGM_HI_ES,
    );
    let pixel_shader_address = agc_program_address(
        &graphics.shader_registers,
        AGC_SPI_SHADER_PGM_LO_PS,
        AGC_SPI_SHADER_PGM_HI_PS,
    );
    if trace_agc_programs() {
        let pixel_user_data = std::array::from_fn::<_, 8, _>(|index| {
            graphics
                .shader_registers
                .get(&(0x00c + u32::try_from(index).expect("pixel scalar index fits")))
                .copied()
                .unwrap_or_default()
        });
        eprintln!(
            "[REMu][AGC][PROGRAM] draw={} packet={packet_address:#x} indexed={indexed} count={element_count} es={export_shader_address:#x} ps={pixel_shader_address:#x} ps_user={pixel_user_data:x?}",
            graphics.emitted_draws,
        );
    }
    let (color_targets, color_target_absent) = agc_prepare_draw_color_targets(graphics);
    let color_target_address = color_targets.first().map_or(0, |target| target.address);
    let primitive_type = graphics
        .user_config_registers
        .get(&AGC_VGT_PRIMITIVE_TYPE)
        .copied()
        .unwrap_or_default();
    let export_shader = agc_shader_snapshot(shaders, export_shader_address);
    let pixel_shader = agc_shader_snapshot(shaders, pixel_shader_address);
    // A zero CB_TARGET_MASK explicitly disables every color write. With no active depth surface
    // or shader memory side effect, the hardware draw is observably a no-op. Unity emits many of
    // these while preparing command buffers; evaluating both shaders and sending them through the
    // Metal presenter consumed substantial CPU despite every pipeline being rejected there.
    if graphics
        .context_registers
        .get(&AGC_CB_TARGET_MASK)
        .is_some_and(|&mask| mask == 0)
        && !agc_has_active_depth_state(&graphics.context_registers)
        && !agc_shader_has_memory_side_effects(export_shader.as_ref())
        && !agc_shader_has_memory_side_effects(pixel_shader.as_ref())
    {
        return;
    }
    let mut draw = remu_gpu::IndexedDraw {
        sequence: graphics.emitted_draws,
        indexed,
        instance_count: graphics.instance_count.max(1),
        index_buffer_address: graphics.index_buffer_address,
        index_resource_address: graphics.index_resource_address,
        index_buffer_count: graphics.index_buffer_count,
        index_size: graphics.index_size,
        index_offset: element_offset,
        index_count: element_count,
        index_snapshot: None,
        draw_flags,
        primitive_type,
        export_shader_address,
        pixel_shader_address,
        export_shader_prefix: agc_shader_prefix_snapshot(export_shader_address),
        pixel_shader_prefix: agc_shader_prefix_snapshot(pixel_shader_address),
        export_shader,
        pixel_shader,
        export_shader_evaluation: None,
        pixel_shader_evaluation: None,
        color_target_address,
        scanout_target: None,
        context_registers: agc_register_snapshot(&graphics.context_registers),
        shader_registers: agc_register_snapshot(&graphics.shader_registers),
        user_config_registers: agc_register_snapshot(&graphics.user_config_registers),
    };
    draw.index_snapshot = snapshot_draw_indices(&draw);
    let export_user_data_base = export_user_data_register(&draw);
    let required_vertex_record_count = draw_required_vertex_record_count(&draw);
    let export_request = draw.export_shader.as_ref().map(|snapshot| {
        shader_evaluation_worker_request(
            &graphics.shader_registers,
            snapshot,
            export_user_data_base,
            8,
            required_vertex_record_count,
        )
    });
    let pixel_request = draw.pixel_shader.as_ref().map(|snapshot| {
        shader_evaluation_worker_request(&graphics.shader_registers, snapshot, 0x00c, 0, None)
    });
    if graphics.shader_evaluation_workers.is_none() {
        match ShaderEvaluationWorkers::new() {
            Ok(workers) => graphics.shader_evaluation_workers = Some(workers),
            Err(error) => {
                eprintln!("[REMu][AGC] parallel shader evaluation unavailable: {error}");
            }
        }
    }
    let (export_evaluation, pixel_evaluation) =
        if let Some(workers) = graphics.shader_evaluation_workers.as_mut() {
            evaluate_draw_shader_requests_parallel(
                workers,
                &mut graphics.shader_evaluation_cache,
                export_request,
                pixel_request,
            )
        } else {
            let export = export_request.map(|request| {
                request.and_then(|request| {
                    graphics
                        .shader_evaluation_cache
                        .evaluate(&request.state, request.options)
                })
            });
            let pixel = pixel_request.map(|request| {
                request.and_then(|request| {
                    graphics
                        .shader_evaluation_cache
                        .evaluate(&request.state, request.options)
                })
            });
            (export, pixel)
        };
    if let Some(evaluation) = export_evaluation {
        draw.export_shader_evaluation = evaluation
            .and_then(snapshot_small_readonly_global_bindings)
            .map_err(|error| {
                if graphics
                    .failed_shader_evaluations
                    .insert((0, draw.export_shader_address))
                {
                    let resource2_register = export_user_data_base.saturating_sub(1);
                    let resource2 = graphics.shader_registers.get(&resource2_register).copied();
                    let user_sgpr_count =
                        resource2.map(|value| ((value >> 1) & 0x1f) | (((value >> 27) & 1) << 5));
                    let required_inputs = draw
                        .export_shader
                        .as_ref()
                        .map(|snapshot| {
                            snapshot
                                .program
                                .input_scalar_registers()
                                .iter()
                                .copied()
                                .collect::<Vec<_>>()
                        })
                        .unwrap_or_default();
                    eprintln!(
                        "[REMu][AGC][SHADER] export evaluation failed at {:#x}: {error}; \
                         user_base={export_user_data_base:#x} resource2={resource2:#x?} \
                         user_sgprs={user_sgpr_count:?} required_inputs={required_inputs:x?}",
                        draw.export_shader_address,
                    );
                }
            })
            .ok();
    }
    if let Some(evaluation) = pixel_evaluation {
        draw.pixel_shader_evaluation = evaluation
            .and_then(snapshot_small_readonly_global_bindings)
            .map_err(|error| {
                if graphics
                    .failed_shader_evaluations
                    .insert((1, draw.pixel_shader_address))
                {
                    let control_flow = draw.pixel_shader.as_ref().map_or_else(
                        || "missing shader".to_owned(),
                        |snapshot| shader_control_flow_summary(&snapshot.program),
                    );
                    eprintln!(
                        "[REMu][AGC][SHADER] pixel evaluation failed at {:#x}: {error}; {}",
                        draw.pixel_shader_address, control_flow,
                    );
                }
            })
            .ok();
    }
    if trace_logarithmic_sequence(draw.sequence, 8) {
        let mut cache_stats = graphics.shader_evaluation_cache.stats();
        if let Some(workers) = graphics.shader_evaluation_workers.as_ref() {
            cache_stats = cache_stats.merge(workers.stats());
        }
        eprintln!(
            "[REMu][AGC] shader evaluation cache draw={} hits={} misses={} failures={} invalidations={} entries={} evictions={} pending_admissions={}",
            draw.sequence,
            cache_stats.hits,
            cache_stats.misses,
            cache_stats.failures,
            cache_stats.invalidations,
            cache_stats.entries,
            cache_stats.evictions,
            cache_stats.pending_admissions,
        );
    }
    agc_resolve_targetless_scanout(
        graphics,
        &mut draw,
        color_target_absent,
        future_scanout_target,
    );
    let samples_large_linear_video = draw
        .pixel_shader_evaluation
        .as_deref()
        .into_iter()
        .flat_map(|evaluation| &evaluation.image_bindings)
        .any(agc_is_large_linear_video_binding);
    if samples_large_linear_video {
        static LOGGED_VIDEO_DRAWS: OnceLock<Mutex<HashSet<u64>>> = OnceLock::new();
        let mut logged = LOGGED_VIDEO_DRAWS
            .get_or_init(|| Mutex::new(HashSet::new()))
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if logged.len() < 16 && logged.insert(draw.pixel_shader_address) {
            let image_bindings = draw
                .pixel_shader_evaluation
                .as_deref()
                .into_iter()
                .flat_map(|evaluation| &evaluation.image_bindings)
                .map(|binding| {
                    (
                        binding.opcode,
                        agc_image_binding_address(binding),
                        binding.resource_descriptor.clone(),
                    )
                })
                .collect::<Vec<_>>();
            eprintln!(
                "[REMu][AGC][VIDEO] draw={} ps={:#x} color_absent={} target={:#x} target_mask={:?} shader_mask={:?} future={:#x} scanout={:#x} depth={} writes_image={} exports={:#x} bindings={image_bindings:x?}",
                draw.sequence,
                draw.pixel_shader_address,
                color_target_absent,
                draw.color_target_address,
                graphics.context_registers.get(&AGC_CB_TARGET_MASK),
                graphics.context_registers.get(&AGC_CB_SHADER_MASK),
                future_scanout_target.map_or(0, |target| target.address),
                draw.scanout_target.map_or(0, |target| target.address),
                agc_has_active_depth_state(&graphics.context_registers),
                agc_draw_writes_image(&draw),
                draw.pixel_shader
                    .as_ref()
                    .map_or(0, |shader| shader.program.pixel_color_export_masks()),
            );
        }
    }
    if trace_agc_shaders() {
        for address in [
            draw.export_shader_evaluation.as_deref(),
            draw.pixel_shader_evaluation.as_deref(),
        ]
        .into_iter()
        .flatten()
        .flat_map(|evaluation| &evaluation.image_bindings)
        .filter(|binding| {
            binding.opcode.starts_with("ImageSample") || binding.opcode.starts_with("ImageGather")
        })
        .filter_map(agc_image_binding_address)
        {
            graphics.sampled_texture_addresses.insert(address);
        }
        let pair = (draw.export_shader_address, draw.pixel_shader_address);
        let scanout_address = draw
            .scanout_target
            .map(|target| target.address)
            .or_else(|| {
                agc_video_out_target_for_address(video_out, draw.color_target_address)
                    .map(|target| target.address)
            })
            .or_else(|| {
                future_scanout_target
                    .filter(|target| target.address == draw.color_target_address)
                    .map(|target| target.address)
            });
        let explicit_target = trace_agc_target_address();
        let trace_pair = if let Some(address) = explicit_target {
            if address == draw.color_target_address
                && graphics.traced_target_shader_pairs.insert(pair)
            {
                eprintln!(
                    "[REMu][AGC][SHADER][TARGET] draw={} target={:#x} es={:#x} ps={:#x}",
                    draw.sequence,
                    draw.color_target_address,
                    draw.export_shader_address,
                    draw.pixel_shader_address,
                );
                true
            } else {
                false
            }
        } else if graphics
            .sampled_texture_addresses
            .contains(&draw.color_target_address)
        {
            if graphics.traced_sampled_target_shader_pairs.insert(pair) {
                eprintln!(
                    "[REMu][AGC][SHADER][SAMPLED-TARGET] draw={} target={:#x} es={:#x} ps={:#x}",
                    draw.sequence,
                    draw.color_target_address,
                    draw.export_shader_address,
                    draw.pixel_shader_address,
                );
                true
            } else {
                false
            }
        } else if let Some(address) = scanout_address {
            if graphics.traced_scanout_shader_pairs.insert(pair) {
                eprintln!(
                    "[REMu][AGC][SHADER][SCANOUT] draw={} target={address:#x} es={:#x} ps={:#x}",
                    draw.sequence, draw.export_shader_address, draw.pixel_shader_address,
                );
                true
            } else {
                false
            }
        } else if graphics.traced_shader_pairs.len() < 16 {
            graphics.traced_shader_pairs.insert(pair)
        } else {
            false
        };
        if trace_pair {
            trace_agc_shader_draw(&draw);
        }
    }
    agc_trace_draw_summary(&draw);
    emit_agc_gpu_event(remu_gpu::GpuEvent::IndexedDraw(Box::new(draw)));
}

pub(crate) fn agc_emit_indexed_draw(
    packet_address: u64,
    graphics: &mut AgcGraphicsState,
    shaders: &mut AgcShaderRegistry,
    video_out: &VideoOutState,
    future_scanout_target: Option<AgcVideoOutTarget>,
) {
    agc_emit_draw(
        packet_address,
        graphics,
        shaders,
        video_out,
        true,
        read_guest_u32(packet_address + 8),
        read_guest_u32(packet_address + 12),
        read_guest_u32(packet_address + 16),
        future_scanout_target,
    );
}
