// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

use super::{
    ComputePipelineKey, ProgramIdentity, image_component_class, pixel_input_controls,
    register_value,
};
use remu_gpu::{ComputeDispatch, GpuRegister, IndexedDraw};
use remu_shader::{
    InstructionControl, OperandKind, Program, ShaderEvaluation, image_is_comparison,
};
use std::collections::BTreeSet;
use std::sync::{Arc, OnceLock};

pub(crate) fn graphics_allocated_vector_register_count(
    shader_registers: &[GpuRegister],
    user_data_register: u32,
) -> u32 {
    register_value(shader_registers, user_data_register.saturating_sub(2)).map_or(
        256,
        |resource1| {
            // GFX10 graphics PGM_RSRC1.VGPRS is a zero-based count of four-register blocks.
            ((resource1 & 0x3f) + 1) * 4
        },
    )
}

pub(crate) fn compute_allocated_vector_register_count(shader_registers: &[GpuRegister]) -> u32 {
    register_value(shader_registers, 0x212).map_or(256, |resource1| {
        // The Gen5 compute state consumed by shadPS4 stores the block count directly. Preserve at
        // least one four-register block so relative-source overflow can still return VGPR0.
        (resource1 & 0x3f).max(1) * 4
    })
}

pub(crate) fn compute_pipeline_key(
    dispatch: &ComputeDispatch,
    evaluation: &ShaderEvaluation,
    program: &std::sync::Arc<remu_shader::Program>,
) -> ComputePipelineKey {
    let image_slots = evaluation.image_binding_slots();
    ComputePipelineKey {
        // Retaining the Arc makes pointer identity stable for the lifetime of both successful and
        // negative cache entries; an allocator cannot recycle a dropped Program into this key.
        program_identity: ProgramIdentity(Arc::clone(program)),
        lds_size_bytes: dispatch.launch.lds_size_bytes,
        wave_lane_count: dispatch.launch.wave_lane_count,
        allocated_vector_register_count: compute_allocated_vector_register_count(
            &dispatch.shader_registers,
        ),
        compute_system_registers: evaluation.compute_system_registers,
        initial_scalar_count: evaluation.initial_scalar_registers.len(),
        global_bindings: evaluation
            .global_memory_bindings
            .iter()
            .map(|binding| (binding.instruction_pcs.to_vec(), 0, 0, binding.writable))
            .collect(),
        image_bindings: evaluation
            .image_bindings
            .iter()
            .enumerate()
            .map(|(index, binding)| {
                let storage = is_storage_image_opcode(binding.opcode);
                let comparison = image_is_comparison(binding.opcode);
                let texture_type = binding
                    .resource_descriptor
                    .get(3)
                    .map_or(0, |word| (word >> 28) & 0xf);
                (
                    image_component_class(binding),
                    storage,
                    comparison,
                    binding.control.dimension,
                    texture_type,
                    image_slots.texture_slots[index],
                    image_slots.sampler_slots[index],
                )
            })
            .collect(),
    }
}

pub(crate) fn program_embeds_global_base(program: &remu_shader::Program) -> bool {
    program.instructions.iter().any(|instruction| {
        matches!(
            instruction.control.as_ref(),
            Some(remu_shader::InstructionControl::GlobalMemory(_))
        )
    })
}

pub(crate) fn is_writable_image_opcode(opcode: remu_shader::Opcode) -> bool {
    opcode.starts_with("ImageStore") || opcode.starts_with("ImageAtomic")
}

pub(crate) fn is_storage_image_opcode(opcode: remu_shader::Opcode) -> bool {
    is_writable_image_opcode(opcode) || opcode.starts_with("ImageLoad")
}

pub(crate) fn image_resource_descriptor_is_null(words: &[u32]) -> bool {
    !words.is_empty() && words.iter().all(|word| *word == 0)
}

pub(crate) fn validate_compute_launch_for_metal(
    launch: remu_gpu::ComputeLaunch,
    execution_width: u32,
    wave64_split_safe: bool,
    maximum_threads_per_group: u64,
    maximum_threadgroup_memory_bytes: u64,
    static_threadgroup_memory_bytes: u64,
    dynamic_threadgroup_memory_bytes: u32,
) -> Result<(), String> {
    if launch.base_group != [0; 3] {
        return Err(format!(
            "Metal compute does not yet support nonzero base groups {:?}",
            launch.base_group
        ));
    }
    if execution_width != launch.wave_lane_count
        && !(launch.wave_lane_count == 64 && execution_width == 32 && wave64_split_safe)
    {
        return Err(format!(
            "guest wave{} cannot execute on Metal SIMD{} without changing wave semantics",
            launch.wave_lane_count, execution_width
        ));
    }
    let threadgroup_threads = launch.threads_per_group_total();
    if threadgroup_threads == 0 || threadgroup_threads > maximum_threads_per_group {
        return Err(format!(
            "guest compute threadgroup {:?} has {threadgroup_threads} threads, Metal permits {maximum_threads_per_group}",
            launch.threads_per_group
        ));
    }
    if dynamic_threadgroup_memory_bytes != 0
        && dynamic_threadgroup_memory_bytes != launch.lds_size_bytes
    {
        return Err(format!(
            "Metal shader requests {dynamic_threadgroup_memory_bytes} bytes of LDS, dispatch declares {}",
            launch.lds_size_bytes
        ));
    }
    let total_threadgroup_memory = static_threadgroup_memory_bytes
        .checked_add(u64::from(dynamic_threadgroup_memory_bytes))
        .ok_or("Metal threadgroup memory size overflows")?;
    if total_threadgroup_memory > maximum_threadgroup_memory_bytes {
        return Err(format!(
            "compute pipeline needs {total_threadgroup_memory} bytes of threadgroup memory, this Metal device permits {maximum_threadgroup_memory_bytes}"
        ));
    }
    Ok(())
}

pub(crate) fn compute_program_can_split_wave64(program: &Program) -> bool {
    wave64_split_blockers(program).is_empty()
}

pub(crate) fn wave64_split_blockers(program: &Program) -> Vec<String> {
    let mut blockers = Vec::new();
    for instruction in &program.instructions {
        let name = instruction.opcode.known_name();
        let wave_sensitive = name.is_none_or(|name| {
            ["Swizzle", "Ballot", "Saveexec"]
                .iter()
                .any(|wave_operation| name.contains(wave_operation))
                // Structured wave64 predication is represented half-locally across the two
                // Metal SIMD32 groups. These exercised operations consume only the current
                // architectural half of the mask, so each SIMD32 subgroup computes it exactly.
                && !matches!(name, "SAndSaveexecB64" | "SOrn2SaveexecB64")
        });
        if wave_sensitive {
            blockers.push(format!("{:#x}:{}", instruction.pc, instruction.opcode));
        }
    }
    blockers
}

pub(crate) fn shader_scalar_metadata(evaluation: &ShaderEvaluation) -> Result<Vec<u32>, String> {
    let mut words = evaluation.initial_scalar_registers.to_vec();
    words.reserve(evaluation.global_memory_bindings.len().saturating_mul(4));
    for binding in &evaluation.global_memory_bindings {
        words.extend([
            binding.base_address as u32,
            (binding.base_address >> 32) as u32,
            u32::try_from(binding.byte_count)
                .map_err(|_| "shader global binding exceeds the 32-bit Metal byte range")?,
            u32::from(binding.writable),
        ]);
    }
    Ok(words)
}

pub(crate) fn shader_scalar_metadata_for_registers(
    evaluation: &ShaderEvaluation,
    shader_registers: &[GpuRegister],
    user_data_register: u32,
    scalar_register_base: usize,
) -> Result<Vec<u32>, String> {
    debug_assert!(
        shader_registers
            .windows(2)
            .all(|registers| registers[0].offset <= registers[1].offset)
    );
    let mut words = shader_scalar_metadata(evaluation)?;
    let scalar_end = scalar_register_base
        .saturating_add(64)
        .min(evaluation.initial_scalar_registers.len());
    for scalar_register in scalar_register_base..scalar_end {
        let offset = u32::try_from(scalar_register - scalar_register_base)
            .map_err(|_| "shader scalar register offset does not fit u32")?;
        words[scalar_register] =
            register_value(shader_registers, user_data_register + offset).unwrap_or_default();
    }
    Ok(words)
}

pub(crate) fn program_uses_lds(program: &remu_shader::Program) -> bool {
    program.instructions.iter().any(|instruction| {
        instruction.encoding == remu_shader::ShaderEncoding::Ds
            && !instruction.opcode.is("DsSwizzleB32")
    })
}

pub(crate) fn vertex_parameter_export_mask(program: &Program) -> u32 {
    program
        .instructions
        .iter()
        .filter_map(|instruction| {
            let InstructionControl::Export(export) = instruction.control.as_ref()? else {
                return None;
            };
            ((32..64).contains(&export.target) && export.enable_mask != 0)
                .then(|| 1_u32 << (export.target - 32))
        })
        .fold(0, |mask, export| mask | export)
}

pub(crate) fn pixel_interpolation_attribute_mask(program: &Program) -> u32 {
    program
        .instructions
        .iter()
        .filter_map(|instruction| {
            let InstructionControl::Interpolation(interpolation) = instruction.control.as_ref()?
            else {
                return None;
            };
            (interpolation.attribute < 32).then_some(1_u32 << interpolation.attribute)
        })
        .fold(0, |mask, attribute| mask | attribute)
}

/// Determines which missing physical rectangle parameters supply texture coordinates.
///
/// The AGC procedural rectangle stage does not contain conventional PARAM exports. Its pixel
/// consumer still makes the parameter roles unambiguous: trace the first texture sample's address
/// VGPRs backward through their vector data dependencies and collect the interpolation operations
/// which produce them. `V_INTERP_P1` only creates an intermediate; `V_INTERP_P2` or
/// `V_INTERP_MOV` materializes the attribute value. Return `None` when no analyzable texture sample
/// exists so callers can retain a conservative legacy fallback for unrelated procedural shaders.
pub(crate) fn rectangle_coordinate_parameter_mask(
    program: &Program,
    controls: &[u32; 32],
) -> Option<u32> {
    let (sample_index, image) =
        program
            .instructions
            .iter()
            .enumerate()
            .find_map(|(index, instruction)| {
                (instruction.opcode.starts_with("ImageSample")
                    || instruction.opcode.starts_with("ImageGather"))
                .then(|| match instruction.control.as_ref() {
                    Some(InstructionControl::Image(image)) => Some((index, image)),
                    _ => None,
                })
                .flatten()
            })?;

    let mut live_vectors = image
        .address_registers
        .iter()
        .copied()
        .collect::<BTreeSet<_>>();
    let mut mask = 0_u32;
    for instruction in program.instructions[..sample_index].iter().rev() {
        let writes_live_vector = instruction.destinations.iter().any(|destination| {
            destination.kind == OperandKind::VectorRegister
                && live_vectors.contains(&destination.value)
        });
        if !writes_live_vector {
            continue;
        }
        for destination in &instruction.destinations {
            if destination.kind == OperandKind::VectorRegister {
                live_vectors.remove(&destination.value);
            }
        }

        if let Some(InstructionControl::Interpolation(interpolation)) = instruction.control.as_ref()
            && !instruction.opcode.starts_with("VInterpP1")
            && interpolation.attribute < 32
        {
            let control = remu_shader_metal::PixelInputControl::from_register(
                controls[interpolation.attribute as usize],
            );
            if !control.uses_default() {
                mask |= 1_u32 << control.input_offset;
            }
            continue;
        }
        for source in &instruction.sources {
            if source.kind == OperandKind::VectorRegister {
                live_vectors.insert(source.value);
            }
        }
    }
    // A texture operation with no recoverable interpolation dependency is not evidence that the
    // rectangle has no coordinate parameter. Returning `Some(0)` makes the synthesized stage
    // fill every input with unit modulation, so the consumer samples one constant texel and a
    // fullscreen composite can become uniformly black. Let the caller use its conservative
    // parameter convention when the decoded address provenance is incomplete.
    (mask != 0).then_some(mask)
}

pub(crate) fn missing_vertex_parameter_exports(
    exported_parameters: u32,
    interpolated_attributes: u32,
    controls: &[u32; 32],
) -> Vec<(u32, u32, u32)> {
    (0..32)
        .filter(|logical| interpolated_attributes & (1 << logical) != 0)
        .filter_map(|logical| {
            let raw = controls[logical as usize];
            let control = remu_shader_metal::PixelInputControl::from_register(raw);
            if control.uses_default() {
                return None;
            }
            let physical = u32::from(control.input_offset);
            (exported_parameters & (1 << physical) == 0).then_some((logical, physical, raw))
        })
        .collect()
}

pub(crate) fn skip_incomplete_rectangle_draw(draw: &IndexedDraw) -> bool {
    // A non-default pixel interpolation has no defined source when the vertex/primitive stage
    // exports no matching PARAM. Do not treat recognition of the position-only procedural stage
    // as proof that its consumer interface has been reconstructed: until every required PARAM is
    // supplied, executing a synthesized fullscreen triangle samples arbitrary texels and erases
    // the valid composite immediately before scanout. Reject every incomplete interface; complete
    // procedural rectangles still take the fixed-function lowering path.
    static ALLOW_INVALID: OnceLock<bool> = OnceLock::new();
    !*ALLOW_INVALID.get_or_init(|| {
        std::env::var_os("REMU_ALLOW_INCOMPLETE_RECTANGLES").is_some_and(|value| value == "1")
    }) && rectangle_draw_has_incomplete_interface(draw)
}

/// Recognizes a fused GFX10 NGG export/geometry program that requires a Metal mesh stage.
///
/// These programs allocate primitive output, exchange records through general LDS, then publish
/// both primitive connectivity and position. Lane-private ADDTID compiler spills alone do not
/// qualify: ordinary vertex/pixel lowering already preserves those without a mesh pipeline.
pub(crate) fn ngg_mesh_export_stage(program: &Program) -> bool {
    let mut allocation_message = false;
    let mut primitive_export = false;
    let mut position_export = false;
    let mut general_lds = false;
    for instruction in &program.instructions {
        if instruction.opcode.is("SSendmsg")
            && instruction
                .sources
                .first()
                .is_some_and(|source| source.value == 9)
        {
            allocation_message = true;
        }
        if let Some(InstructionControl::Export(export)) = instruction.control.as_ref() {
            primitive_export |= export.target == 20 && export.enable_mask != 0;
            position_export |= export.target == 12 && export.enable_mask != 0;
        }
        if matches!(instruction.control, Some(InstructionControl::DataShare(_)))
            && !matches!(
                instruction.opcode.known_name(),
                Some("DsSwizzleB32" | "DsWriteAddtidB32" | "DsReadAddtidB32")
            )
        {
            general_lds = true;
        }
    }
    allocation_message && primitive_export && position_export && general_lds
}

/// Recognizes the position-only AGC primitive stage used for procedural fullscreen rectangles.
///
/// This deliberately describes architectural behavior instead of matching a game address or an
/// exact byte string: the stage sends GS allocation message 9, exports primitive data at target
/// 20, derives X/Y from the vertex ID, and exports position at target 12. It must have no memory,
/// LDS, image, interpolation, or PARAM-export work. More complex merged ES/GS programs are not
/// accepted here and continue through the ordinary translator/rejection path.
pub(crate) fn procedural_rectangle_export_stage(program: &Program) -> bool {
    let mut primitive_export = false;
    let mut position_export = false;
    let mut allocation_message = false;
    let mut vertex_id_x = false;
    let mut vertex_id_y = false;
    let mut coordinate_conversions = 0_u32;
    let mut coordinate_mads = 0_u32;

    for instruction in &program.instructions {
        match instruction.control.as_ref() {
            Some(InstructionControl::Export(export)) => match export.target {
                20 if export.enable_mask != 0 && export.done => primitive_export = true,
                12 if export.enable_mask == 0x0f && export.done => position_export = true,
                _ => return false,
            },
            Some(
                InstructionControl::Image(_)
                | InstructionControl::GlobalMemory(_)
                | InstructionControl::BufferMemory(_)
                | InstructionControl::Interpolation(_)
                | InstructionControl::DataShare(_),
            ) => return false,
            _ => {}
        }

        match instruction.opcode.known_name() {
            Some("SSendmsg")
                if instruction
                    .sources
                    .first()
                    .is_some_and(|source| source.value == 9) =>
            {
                allocation_message = true;
            }
            Some("VAndB32") => vertex_id_x = true,
            Some("VLshrrevB32") => vertex_id_y = true,
            Some("VCvtF32U32" | "VCvtF32I32") => {
                coordinate_conversions = coordinate_conversions.saturating_add(1);
            }
            Some("VMadF32" | "VFmaF32" | "VLshlAddU32") => {
                coordinate_mads = coordinate_mads.saturating_add(1);
            }
            Some(name)
                if name.starts_with("Ds")
                    || name.starts_with("Image")
                    || name.starts_with("Buffer")
                    || name.starts_with("Global")
                    || name.starts_with("Flat") =>
            {
                return false;
            }
            None => return false,
            _ => {}
        }
    }

    primitive_export
        && position_export
        && allocation_message
        && vertex_id_x
        && vertex_id_y
        && coordinate_conversions >= 2
        && coordinate_mads >= 2
}

pub(crate) fn rectangle_draw_has_incomplete_interface(draw: &IndexedDraw) -> bool {
    if !matches!(draw.primitive_type, 7 | 0x11) || draw.indexed || draw.index_count != 3 {
        return false;
    }
    let Some(export) = draw.export_shader.as_ref() else {
        return false;
    };
    let Some(pixel) = draw.pixel_shader.as_ref() else {
        return false;
    };
    let exported_parameters = vertex_parameter_export_mask(&export.program);
    let interpolated_attributes = pixel_interpolation_attribute_mask(&pixel.program);
    !missing_vertex_parameter_exports(
        exported_parameters,
        interpolated_attributes,
        &pixel_input_controls(&draw.context_registers),
    )
    .is_empty()
}

pub(crate) fn pixel_scalars(draw: &IndexedDraw) -> [u32; 4] {
    std::array::from_fn(|index| {
        register_value(
            &draw.shader_registers,
            0x00c + u32::try_from(index).expect("scalar index fits"),
        )
        .unwrap_or_default()
    })
}
