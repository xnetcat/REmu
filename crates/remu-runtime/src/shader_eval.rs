use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Arc;

use remu_shader::GuestMemoryReader;

use crate::{
    AGC_SPI_PS_INPUT_CNTL_0, ProcessGuestMemory, ShaderEvaluationCache,
    ShaderEvaluationWorkerRequest, ShaderEvaluationWorkers,
};

pub(crate) fn shader_control_flow_summary(program: &remu_shader::Program) -> String {
    const MAXIMUM_REPORTED_BRANCHES: usize = 32;
    let all_branches = program
        .instructions
        .iter()
        .filter(|instruction| instruction.encoding == remu_shader::ShaderEncoding::Sopp)
        .filter(|instruction| {
            instruction.opcode.starts_with("SBranch") || instruction.opcode.starts_with("SCbranch")
        })
        .map(|instruction| {
            let encoded_offset = instruction.words.first().copied().unwrap_or_default() as u16;
            let offset = i16::from_ne_bytes(encoded_offset.to_ne_bytes());
            let next = i64::from(instruction.pc)
                + i64::try_from(instruction.words.len().saturating_mul(4)).unwrap_or(i64::MAX);
            let target = next.saturating_add(i64::from(offset) * 4);
            format!("{:#x}:{}->{target:#x}", instruction.pc, instruction.opcode)
        })
        .collect::<Vec<_>>();
    let omitted = all_branches.len().saturating_sub(MAXIMUM_REPORTED_BRANCHES);
    let branches = all_branches
        .iter()
        .take(MAXIMUM_REPORTED_BRANCHES)
        .cloned()
        .collect::<Vec<_>>();
    format!(
        "instructions={} branches=[{}] omitted={omitted}",
        program.instructions.len(),
        branches.join(",")
    )
}

pub(crate) fn trace_agc_shader_draw(draw: &remu_gpu::IndexedDraw) {
    let instruction_counts = [&draw.export_shader, &draw.pixel_shader].map(|shader| {
        shader
            .as_ref()
            .map_or(0, |shader| shader.program.instructions.len())
    });
    let raw_counts = [&draw.export_shader, &draw.pixel_shader].map(|shader| {
        shader.as_ref().map_or(0, |shader| {
            shader
                .program
                .instructions
                .iter()
                .filter(|instruction| instruction.opcode.known_name().is_none())
                .count()
        })
    });
    let opcode_lists = [&draw.export_shader, &draw.pixel_shader].map(|shader| {
        shader.as_ref().map_or_else(String::new, |shader| {
            shader
                .program
                .instructions
                .iter()
                .map(|instruction| instruction.opcode.to_string())
                .collect::<Vec<_>>()
                .join(",")
        })
    });
    let export_lists = [&draw.export_shader, &draw.pixel_shader].map(|shader| {
        shader.as_ref().map_or_else(String::new, |shader| {
            shader
                .program
                .instructions
                .iter()
                .filter_map(|instruction| {
                    let remu_shader::InstructionControl::Export(control) =
                        instruction.control.as_ref()?
                    else {
                        return None;
                    };
                    Some(format!(
                        "{:#x}:target={}/mask={:#x}/compressed={}/done={}/valid={}",
                        instruction.pc,
                        control.target,
                        control.enable_mask,
                        control.compressed,
                        control.done,
                        control.valid_mask,
                    ))
                })
                .collect::<Vec<_>>()
                .join(",")
        })
    });
    let interpolation_lists = [&draw.export_shader, &draw.pixel_shader].map(|shader| {
        shader.as_ref().map_or_else(String::new, |shader| {
            shader
                .program
                .instructions
                .iter()
                .filter_map(|instruction| {
                    let remu_shader::InstructionControl::Interpolation(control) =
                        instruction.control.as_ref()?
                    else {
                        return None;
                    };
                    Some(format!(
                        "{:#x}:attr={}/channel={}",
                        instruction.pc, control.attribute, control.channel
                    ))
                })
                .collect::<Vec<_>>()
                .join(",")
        })
    });
    let export_user_data_base = export_user_data_register(draw);
    let evaluation_summaries = [
        trace_shader_evaluation(
            draw,
            draw.export_shader.as_ref(),
            export_user_data_base,
            8,
            draw_required_vertex_record_count(draw),
        ),
        trace_shader_evaluation(draw, draw.pixel_shader.as_ref(), 0x00c, 0, None),
    ];
    let msl_summary = trace_msl_pair(draw, export_user_data_base);
    eprintln!(
        "[REMu][AGC][SHADER] es={:#x} ps={:#x} es_user_base={export_user_data_base:#x} indexed={} count={} instances={} target={:#x} es_ir={}/{} ps_ir={}/{} es_eval={} ps_eval={} msl={} es_ops=[{}] ps_ops=[{}] es_exports=[{}] ps_exports=[{}] es_interpolations=[{}] ps_interpolations=[{}] es_prefix={:x?} ps_prefix={:x?} cx={:x?} sh={:x?} uc={:x?}",
        draw.export_shader_address,
        draw.pixel_shader_address,
        draw.indexed,
        draw.index_count,
        draw.instance_count,
        draw.color_target_address,
        instruction_counts[0],
        raw_counts[0],
        instruction_counts[1],
        raw_counts[1],
        evaluation_summaries[0],
        evaluation_summaries[1],
        msl_summary,
        opcode_lists[0],
        opcode_lists[1],
        export_lists[0],
        export_lists[1],
        interpolation_lists[0],
        interpolation_lists[1],
        draw.export_shader_prefix,
        draw.pixel_shader_prefix,
        draw.context_registers,
        draw.shader_registers,
        draw.user_config_registers,
    );
}

pub(crate) fn export_user_data_register(draw: &remu_gpu::IndexedDraw) -> u32 {
    [0x08c, 0x0cc, 0x04c]
        .into_iter()
        .find(|base| {
            draw.shader_registers
                .iter()
                .any(|register| register.offset == base - 1)
        })
        .unwrap_or(0x0cc)
}

pub(crate) fn draw_required_vertex_record_count(draw: &remu_gpu::IndexedDraw) -> Option<u32> {
    let per_instance = if !draw.indexed {
        let count = if draw.primitive_type == 0x11 {
            4
        } else {
            draw.index_count
        };
        let first = i64::from(draw.index_offset) + i64::from(draw.base_vertex);
        u32::try_from(first).ok()?.checked_add(count)
    } else {
        let restart = draw_primitive_restart_index(draw);
        draw.index_snapshot
            .as_deref()
            .unwrap_or_default()
            .iter()
            .copied()
            .filter(|index| restart != Some(*index))
            .try_fold(None, |maximum, index| {
                let adjusted = i64::from(index) + i64::from(draw.base_vertex);
                let adjusted = u32::try_from(adjusted).ok()?;
                Some(Some(
                    maximum.map_or(adjusted, |current: u32| current.max(adjusted)),
                ))
            })?
            .and_then(|maximum| maximum.checked_add(1))
    }?;
    // Instanced NGG UI packs per-instance attributes after the base vertex span. Clamping the
    // binding to max(index)+1 truncates later instances to OOB zeroes (solid white glyph boxes).
    let instances = draw.instance_count.max(1);
    per_instance.checked_mul(instances)
}

pub(crate) fn draw_primitive_restart_index(draw: &remu_gpu::IndexedDraw) -> Option<u32> {
    const VGT_MULTI_PRIM_IB_RESET_EN: u32 = 0x2a5;
    const VGT_MULTI_PRIM_IB_RESET_INDX: u32 = 0x103;

    let enabled = draw
        .context_registers
        .iter()
        .find(|register| register.offset == VGT_MULTI_PRIM_IB_RESET_EN)
        .is_some_and(|register| register.value & 1 != 0);
    if !enabled {
        return None;
    }
    let programmed = draw
        .context_registers
        .iter()
        .find(|register| register.offset == VGT_MULTI_PRIM_IB_RESET_INDX)
        .map_or(u32::MAX, |register| register.value);
    match draw.index_size {
        0 => Some(programmed & u32::from(u16::MAX)),
        1 => Some(programmed),
        2 => Some(programmed & u32::from(u8::MAX)),
        _ => None,
    }
}

pub(crate) fn snapshot_draw_indices(draw: &remu_gpu::IndexedDraw) -> Option<Arc<[u32]>> {
    if !draw.indexed || draw.index_buffer_address < 0x10_000 || draw.index_count > 1_048_576 {
        return None;
    }
    let element_size = match draw.index_size {
        0 => 2_usize,
        1 => 4,
        2 => 1,
        _ => return None,
    };
    let indices = snapshot_draw_indices_with_element_size(draw, element_size)?;
    // When VGT_INDEX_TYPE is stuck at the 16-bit CLEAR default but the buffer is 32-bit, small
    // indices decode as `(lo, 0)` pairs. Recover by re-reading as 32-bit before the snapshot is
    // frozen onto the draw (Metal prefers `index_snapshot` over a live re-read).
    if element_size == 2
        && indices_look_like_u32_read_as_u16(indices.as_ref())
        && let Some(recovered) = snapshot_draw_indices_with_element_size(draw, 4)
    {
        return Some(recovered);
    }
    Some(indices)
}

fn snapshot_draw_indices_with_element_size(
    draw: &remu_gpu::IndexedDraw,
    element_size: usize,
) -> Option<Arc<[u32]>> {
    let first = usize::try_from(draw.index_offset)
        .ok()?
        .checked_mul(element_size)?;
    let byte_count = usize::try_from(draw.index_count)
        .ok()?
        .checked_mul(element_size)?;
    let address = draw
        .index_buffer_address
        .checked_add(u64::try_from(first).ok()?)?;
    let mut bytes = vec![0_u8; byte_count];
    if !ProcessGuestMemory.read(address, &mut bytes) {
        return None;
    }
    Some(
        bytes
            .chunks_exact(element_size)
            .map(|value| match element_size {
                1 => u32::from(value[0]),
                2 => u32::from(u16::from_le_bytes([value[0], value[1]])),
                _ => u32::from_le_bytes([value[0], value[1], value[2], value[3]]),
            })
            .collect::<Vec<_>>()
            .into(),
    )
}

fn indices_look_like_u32_read_as_u16(indices: &[u32]) -> bool {
    indices.len() >= 4
        && indices.len().is_multiple_of(2)
        && indices
            .chunks_exact(2)
            .all(|pair| pair[1] == 0 && pair[0] < 0x1_0000)
        && indices.iter().step_by(2).any(|value| *value != 0)
}

pub(crate) fn snapshot_small_readonly_global_bindings(
    evaluation: Arc<remu_shader::ShaderEvaluation>,
) -> Result<Arc<remu_shader::ShaderEvaluation>, String> {
    // Unity's transient vertex rings in Superliminal expose five overlapping attribute views
    // whose live ranges are about 76 KiB for an ordinary scene draw. Keeping the old 64 KiB
    // cutoff made those views the only draw inputs read lazily by the presenter, after the guest
    // had already recycled the ring allocation. The stale vertices produced intermittent
    // full-height triangles/vertical smears. Retain packet-time contents for those normal-sized
    // inputs while still avoiding accidental snapshots of genuinely large persistent resources.
    const SNAPSHOT_LIMIT: usize = 128 * 1024;

    let requires_snapshot = evaluation.global_memory_bindings.iter().any(|binding| {
        !binding.writable
            && binding.data.is_empty()
            && binding.byte_count <= SNAPSHOT_LIMIT
            && binding.base_address >= 0x10_000
    });
    if !requires_snapshot {
        return Ok(evaluation);
    }

    let mut snapshot = (*evaluation).clone();
    for binding in &mut snapshot.global_memory_bindings {
        if binding.writable
            || !binding.data.is_empty()
            || binding.byte_count > SNAPSHOT_LIMIT
            || binding.base_address < 0x10_000
        {
            continue;
        }
        let mut data = vec![0_u8; binding.byte_count];
        if !ProcessGuestMemory.read(binding.base_address, &mut data) {
            return Err(format!(
                "could not snapshot read-only global {:#x}/{} at packet submission",
                binding.base_address, binding.byte_count
            ));
        }
        binding.data = data;
    }
    Ok(Arc::new(snapshot))
}

fn evaluate_shader_snapshot(
    draw: &remu_gpu::IndexedDraw,
    snapshot: &remu_gpu::ShaderSnapshot,
    user_data_register: u32,
    scalar_register_base: u32,
    required_vertex_record_count: Option<u32>,
) -> Result<remu_shader::ShaderEvaluation, String> {
    evaluate_shader_register_snapshot_with_options(
        &draw.shader_registers,
        snapshot,
        user_data_register,
        scalar_register_base,
        None,
        remu_shader::EvaluationOptions {
            required_vertex_record_count,
            capture_global_binding_data: false,
            ..remu_shader::EvaluationOptions::default()
        },
    )
}

pub(crate) fn shader_evaluation_worker_request(
    shader_registers: &BTreeMap<u32, u32>,
    snapshot: &remu_gpu::ShaderSnapshot,
    user_data_register: u32,
    scalar_register_base: u32,
    required_vertex_record_count: Option<u32>,
) -> Result<ShaderEvaluationWorkerRequest, String> {
    Ok(ShaderEvaluationWorkerRequest {
        state: remu_shader::build_shader_state(
            Arc::clone(&snapshot.program),
            user_data_register,
            scalar_register_base,
            None,
            |offset| shader_registers.get(&offset).copied(),
        )
        .map_err(|error| format!("state-error({error})"))?,
        options: remu_shader::EvaluationOptions {
            required_vertex_record_count,
            capture_global_binding_data: false,
            ..remu_shader::EvaluationOptions::default()
        },
    })
}

pub(crate) fn evaluate_draw_shader_requests_parallel(
    workers: &mut ShaderEvaluationWorkers,
    local_cache: &mut ShaderEvaluationCache,
    export: Option<Result<ShaderEvaluationWorkerRequest, String>>,
    pixel: Option<Result<ShaderEvaluationWorkerRequest, String>>,
) -> (
    Option<Result<Arc<remu_shader::ShaderEvaluation>, String>>,
    Option<Result<Arc<remu_shader::ShaderEvaluation>, String>>,
) {
    let evaluate_local = |request: ShaderEvaluationWorkerRequest,
                          cache: &mut ShaderEvaluationCache| {
        cache.evaluate(&request.state, request.options)
    };
    match (export, pixel) {
        (Some(Ok(export)), Some(Ok(pixel))) => {
            // Keep the graphics task productive while the shorter shader runs on the helper. The
            // decoded instruction count is a stable cost proxy on cache misses. Running the larger
            // stage locally makes the helper likely to finish before this thread reaches the join,
            // while retaining two-core resource discovery and ordered draw emission.
            if export.state.program.instructions.len() >= pixel.state.program.instructions.len() {
                let pixel_submit = workers.parallel.submit(pixel);
                let export = evaluate_local(export, local_cache);
                let pixel = pixel_submit.and_then(|()| workers.parallel.receive());
                (Some(export), Some(pixel))
            } else {
                let export_submit = workers.parallel.submit(export);
                let pixel = evaluate_local(pixel, local_cache);
                let export = export_submit.and_then(|()| workers.parallel.receive());
                (Some(export), Some(pixel))
            }
        }
        (Some(Ok(export)), pixel) => (
            Some(evaluate_local(export, local_cache)),
            pixel.map(|result| result.map(|_| unreachable!("pixel request was not successful"))),
        ),
        (export, Some(Ok(pixel))) => (
            export.map(|result| result.map(|_| unreachable!("export request was not successful"))),
            Some(evaluate_local(pixel, local_cache)),
        ),
        (export, pixel) => (
            export.map(|result| result.map(|_| unreachable!("export request was not successful"))),
            pixel.map(|result| result.map(|_| unreachable!("pixel request was not successful"))),
        ),
    }
}

pub(crate) fn evaluate_shader_register_snapshot_cached(
    shader_registers: &[remu_gpu::GpuRegister],
    snapshot: &remu_gpu::ShaderSnapshot,
    user_data_register: u32,
    scalar_register_base: u32,
    compute_system_registers: Option<remu_shader::ComputeSystemRegisters>,
    options: remu_shader::EvaluationOptions,
    cache: &mut ShaderEvaluationCache,
) -> Result<Arc<remu_shader::ShaderEvaluation>, String> {
    let state = build_shader_register_state(
        shader_registers,
        snapshot,
        user_data_register,
        scalar_register_base,
        compute_system_registers,
    )?;
    cache.evaluate(&state, options)
}

fn evaluate_shader_register_snapshot_with_options(
    shader_registers: &[remu_gpu::GpuRegister],
    snapshot: &remu_gpu::ShaderSnapshot,
    user_data_register: u32,
    scalar_register_base: u32,
    compute_system_registers: Option<remu_shader::ComputeSystemRegisters>,
    options: remu_shader::EvaluationOptions,
) -> Result<remu_shader::ShaderEvaluation, String> {
    let state = build_shader_register_state(
        shader_registers,
        snapshot,
        user_data_register,
        scalar_register_base,
        compute_system_registers,
    )?;
    remu_shader::evaluate_shader(&ProcessGuestMemory, &state, options)
        .map_err(|error| format!("evaluation-error({error})"))
}

fn build_shader_register_state(
    shader_registers: &[remu_gpu::GpuRegister],
    snapshot: &remu_gpu::ShaderSnapshot,
    user_data_register: u32,
    scalar_register_base: u32,
    compute_system_registers: Option<remu_shader::ComputeSystemRegisters>,
) -> Result<remu_shader::ShaderState, String> {
    remu_shader::build_shader_state(
        Arc::clone(&snapshot.program),
        user_data_register,
        scalar_register_base,
        compute_system_registers,
        |offset| {
            shader_registers
                .iter()
                .find(|register| register.offset == offset)
                .map(|register| register.value)
        },
    )
    .map_err(|error| format!("state-error({error})"))
}

fn trace_msl_pair(draw: &remu_gpu::IndexedDraw, export_user_data_base: u32) -> String {
    let Some(export_snapshot) = draw.export_shader.as_ref() else {
        return "missing-es".to_owned();
    };
    let Some(pixel_snapshot) = draw.pixel_shader.as_ref() else {
        return "missing-ps".to_owned();
    };
    // Preserve the decoded IR even when resource discovery fails before either stage can reach
    // Metal translation. These failures are precisely where the instruction operands and raw
    // descriptor controls are needed to distinguish an invalid resource from a decoder bug.
    dump_shader_trace(draw, export_snapshot, pixel_snapshot, None, None);
    let export_evaluation = match evaluate_shader_snapshot(
        draw,
        export_snapshot,
        export_user_data_base,
        8,
        draw_required_vertex_record_count(draw),
    ) {
        Ok(evaluation) => evaluation,
        Err(error) => return format!("es-{error}"),
    };
    let pixel_evaluation = match evaluate_shader_snapshot(draw, pixel_snapshot, 0x00c, 0, None) {
        Ok(evaluation) => evaluation,
        Err(error) => return format!("ps-{error}"),
    };
    let mut host_location = 0_u32;
    let shader_color_formats = draw
        .context_registers
        .iter()
        .find(|register| register.offset == 0x1c5)
        .map_or(0, |register| register.value);
    let mut pixel_outputs = Vec::new();
    for guest_slot in 0..8 {
        let mask = (pixel_snapshot.program.pixel_color_export_masks() >> (guest_slot * 4)) & 0xf;
        if mask == 0 {
            continue;
        }
        let format_nibble = (shader_color_formats >> (guest_slot * 4)) & 0xf;
        let Some(export_format) =
            remu_shader::PixelExportFormat::from_register_nibble(format_nibble)
        else {
            return format!(
                "ps-invalid-col-format(slot={guest_slot},nibble={format_nibble:#x},register={shader_color_formats:#010x})"
            );
        };
        pixel_outputs.push(remu_shader::PixelOutputBinding {
            guest_slot,
            host_location,
            kind: remu_shader::PixelOutputKind::Float,
            export_format,
        });
        host_location += 1;
    }
    let mut pixel_options = remu_shader_metal::MetalCompileOptions::pixel(
        pixel_outputs,
        draw.context_registers
            .iter()
            .find(|register| register.offset == 0x1b3)
            .map_or(0, |register| register.value),
        draw.context_registers
            .iter()
            .find(|register| register.offset == 0x1b4)
            .map_or(0, |register| register.value),
    );
    pixel_options.pixel_input_controls = std::array::from_fn(|index| {
        let offset = AGC_SPI_PS_INPUT_CNTL_0
            + u32::try_from(index).expect("PS input-control index fits u32");
        let value = draw
            .context_registers
            .iter()
            .find(|register| register.offset == offset)
            .map_or(offset - AGC_SPI_PS_INPUT_CNTL_0, |register| register.value);
        remu_shader_metal::PixelInputControl::from_register(value)
    });
    let pixel = match remu_shader_metal::compile_msl(
        &pixel_snapshot.program,
        &pixel_evaluation,
        &pixel_options,
    ) {
        Ok(shader) => shader,
        Err(error) => {
            dump_shader_trace(draw, export_snapshot, pixel_snapshot, None, None);
            return format!("ps-error({error})");
        }
    };
    let vertex_options =
        remu_shader_metal::MetalCompileOptions::vertex(pixel.interpolated_attribute_count);
    let vertex = match remu_shader_metal::compile_msl(
        &export_snapshot.program,
        &export_evaluation,
        &vertex_options,
    ) {
        Ok(shader) => shader,
        Err(error) => {
            dump_shader_trace(draw, export_snapshot, pixel_snapshot, None, Some(&pixel));
            return format!("es-error({error})");
        }
    };
    dump_shader_trace(
        draw,
        export_snapshot,
        pixel_snapshot,
        Some(&vertex),
        Some(&pixel),
    );
    format!(
        "ok(es={}B/{}b/{}t,ps={}B/{}b/{}t)",
        vertex.source.len(),
        vertex.bindings.global_buffer_count,
        vertex.bindings.texture_count,
        pixel.source.len(),
        pixel.bindings.global_buffer_count,
        pixel.bindings.texture_count,
    )
}

fn dump_shader_trace(
    draw: &remu_gpu::IndexedDraw,
    export_snapshot: &remu_gpu::ShaderSnapshot,
    pixel_snapshot: &remu_gpu::ShaderSnapshot,
    vertex: Option<&remu_shader_metal::MetalShader>,
    pixel: Option<&remu_shader_metal::MetalShader>,
) {
    let Some(directory) = std::env::var_os("REMU_DUMP_SCANOUT_MSL_DIR")
        .or_else(|| std::env::var_os("REMU_DUMP_MSL_DIR"))
        .map(PathBuf::from)
    else {
        return;
    };
    if let Err(error) = std::fs::create_dir_all(&directory) {
        eprintln!("[REMu][AGC][SHADER] could not create MSL dump directory: {error}");
        return;
    }
    let stem = format!(
        "{:016x}-{:016x}",
        draw.export_shader_address, draw.pixel_shader_address
    );
    let ir = format!(
        "export={:#?}\n\nexport_supplemental={:#?}\n\npixel={:#?}\n\npixel_supplemental={:#?}\n",
        export_snapshot.program.instructions,
        export_snapshot.supplemental_programs,
        pixel_snapshot.program.instructions,
        pixel_snapshot.supplemental_programs,
    );
    for (suffix, contents) in [
        ("ir.txt", Some(ir.as_str())),
        ("vertex.metal", vertex.map(|shader| shader.source.as_str())),
        ("pixel.metal", pixel.map(|shader| shader.source.as_str())),
    ] {
        let Some(contents) = contents else {
            continue;
        };
        if let Err(error) = std::fs::write(directory.join(format!("{stem}-{suffix}")), contents) {
            eprintln!("[REMu][AGC][SHADER] could not write MSL trace: {error}");
        }
    }
}

pub(crate) fn dump_draw_shader_ir(draw: &remu_gpu::IndexedDraw) {
    let (Some(export), Some(pixel)) = (&draw.export_shader, &draw.pixel_shader) else {
        return;
    };
    dump_shader_trace(draw, export, pixel, None, None);
}

fn trace_shader_evaluation(
    draw: &remu_gpu::IndexedDraw,
    snapshot: Option<&remu_gpu::ShaderSnapshot>,
    user_data_register: u32,
    scalar_register_base: u32,
    required_vertex_record_count: Option<u32>,
) -> String {
    let Some(snapshot) = snapshot else {
        return "missing".to_owned();
    };
    match evaluate_shader_snapshot(
        draw,
        snapshot,
        user_data_register,
        scalar_register_base,
        required_vertex_record_count,
    ) {
        Ok(evaluation) => {
            let images = evaluation
                .image_bindings
                .iter()
                .map(|binding| {
                    format!(
                        "{:#x}:{}:r={:x?}:s={:x?}:m={:?}",
                        binding.pc,
                        binding.opcode,
                        binding.resource_descriptor,
                        binding.sampler_descriptor,
                        binding.mip_level,
                    )
                })
                .collect::<Vec<_>>()
                .join(",");
            let globals = evaluation
                .global_memory_bindings
                .iter()
                .map(|binding| {
                    let prefix = binding
                        .data
                        .chunks_exact(4)
                        .take(4)
                        .map(|word| {
                            u32::from_le_bytes(word.try_into().expect("four-byte binding word"))
                        })
                        .collect::<Vec<_>>();
                    format!(
                        "s{}@{:#x}/{}:{prefix:x?}",
                        binding.scalar_address, binding.base_address, binding.byte_count
                    )
                })
                .collect::<Vec<_>>()
                .join(",");
            format!(
                "ok(i={},g={},v={},images=[{images}],globals=[{globals}])",
                evaluation.image_bindings.len(),
                evaluation.global_memory_bindings.len(),
                evaluation.vertex_inputs.len(),
            )
        }
        Err(error) => format!("error({error})"),
    }
}

#[cfg(all(test, target_os = "macos"))]
mod tests {
    use std::sync::Arc;

    use remu_shader::{GlobalMemoryBinding, ShaderEvaluation};

    use super::snapshot_small_readonly_global_bindings;

    #[test]
    fn snapshots_transient_vertex_bindings_larger_than_sixty_four_kibibytes() {
        let source = (0..96 * 1024)
            .map(|offset| u8::try_from(offset & 0xff).unwrap())
            .collect::<Vec<_>>();
        let evaluation = ShaderEvaluation {
            initial_scalar_registers: Default::default(),
            scalar_registers: Default::default(),
            scalar_memory_reads: Vec::new(),
            image_bindings: Vec::new(),
            global_memory_bindings: vec![GlobalMemoryBinding {
                scalar_address: 0,
                base_address: source.as_ptr() as u64,
                instruction_pcs: Default::default(),
                byte_count: source.len(),
                data: Vec::new(),
                writable: false,
                write_back_to_guest: false,
            }],
            compute_system_registers: None,
            vertex_inputs: Vec::new(),
        };

        let snapshot =
            snapshot_small_readonly_global_bindings(Arc::new(evaluation)).expect("snapshot");
        assert_eq!(snapshot.global_memory_bindings[0].data, source);
    }
}
