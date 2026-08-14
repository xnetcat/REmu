// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

use super::{ComputePipelineKey, ProgramIdentity, image_component_class, register_value};
#[cfg(test)]
use crate::presenter::decode::pixel_input_controls;
use remu_gpu::{ComputeDispatch, GpuRegister, IndexedDraw, ShaderEvaluationSnapshot};
use remu_shader::{
    InstructionControl, OperandKind, Program, ShaderEvaluation, image_is_comparison,
};
use std::collections::{BTreeSet, HashMap};
use std::sync::{Arc, OnceLock};

/// Canonical empty pixel program used by depth/stencil-only draws.
///
/// GFX10 permits a graphics pipeline with no pixel program when no color attachment is written.
/// Keeping one process-wide IR allocation gives those fragmentless pipelines a stable cache
/// identity without pretending that guest address zero names shader code.
pub(crate) fn fragmentless_pixel_program() -> &'static Arc<Program> {
    static PROGRAM: OnceLock<Arc<Program>> = OnceLock::new();
    PROGRAM.get_or_init(|| {
        Arc::new(
            remu_shader::decode_program(0, &[0xbf81_0000])
                .expect("the canonical S_ENDPGM instruction must decode"),
        )
    })
}

/// Canonical resource state paired with [`fragmentless_pixel_program`].
///
/// The generated Metal `fragment void` entry point still carries the backend's scalar-buffer
/// parameter. Supply one inert dword so normal encoder binding remains valid while the stage has
/// no guest resources or side effects.
pub(crate) fn fragmentless_pixel_evaluation() -> &'static ShaderEvaluationSnapshot {
    static EVALUATION: OnceLock<ShaderEvaluationSnapshot> = OnceLock::new();
    EVALUATION.get_or_init(|| {
        Arc::new(ShaderEvaluation {
            initial_scalar_registers: std::iter::once(0).collect(),
            scalar_registers: std::iter::once(0).collect(),
            scalar_memory_reads: Vec::new(),
            image_bindings: Vec::new(),
            global_memory_bindings: Vec::new(),
            compute_system_registers: None,
            vertex_inputs: Vec::new(),
        })
        .into()
    })
}

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
        compute_threadgroup_size: dispatch
            .launch
            .threads_per_group
            .into_iter()
            .fold(1_u32, u32::saturating_mul),
        scratch_size_bytes: dispatch.launch.scratch_size_bytes,
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
                    binding.descriptor_index_is_runtime,
                    // Constant `false` today: `presenter/shader.rs` sets runtime_heap_tables on
                    // the pixel options only, so a compute binding always lowers from the
                    // snapshot. The field is here so that giving compute tables is a one-line
                    // change at that assignment rather than a silent key collision.
                    super::image_binding_fetches_descriptor_at_runtime(binding, false),
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

/// One-time diagnostic census of split-wave64 hazard classes in a guest program.
///
/// The 2026-08-12 split-wave64 audit (REmu-research/silent-hill) showed the substring-based
/// `wave64_split_blockers` passes nearly every construct whose semantics are half-local under
/// the two-SIMD32-halves model. Until the lane-doubled lowering lands, this census names which
/// hazard classes each wave64 program actually uses, so a live run's wrong output can be
/// attributed to a concrete class without an instrumented rebuild. Logging only; no behavior
/// change.
pub(crate) fn log_wave64_hazard_census(program: &Program, stage: &str) {
    use std::collections::HashSet;
    use std::sync::Mutex;
    static LOGGED: Mutex<Option<HashSet<u64>>> = Mutex::new(None);
    {
        let mut logged = LOGGED.lock().unwrap();
        if !logged
            .get_or_insert_with(HashSet::new)
            .insert(program.address)
        {
            return;
        }
    }
    let mut readlane = 0u32;
    let mut readfirstlane = 0u32;
    let mut mbcnt_mask = 0u32;
    let mut bcnt1_b64 = 0u32;
    let mut scc_branch = 0u32;
    let mut exec_branch = 0u32;
    let mut mask64_ops = 0u32;
    let mut barriers = 0u32;
    for instruction in &program.instructions {
        let Some(name) = instruction.opcode.known_name() else {
            continue;
        };
        match name {
            "VReadlaneB32" => readlane += 1,
            "VReadfirstlaneB32" => readfirstlane += 1,
            "SBcnt1I32B64" => bcnt1_b64 += 1,
            "SBarrier" => barriers += 1,
            "SCbranchScc0" | "SCbranchScc1" => scc_branch += 1,
            "SCbranchExecz" | "SCbranchExecnz" | "SCbranchVccz" | "SCbranchVccnz" => {
                exec_branch += 1;
            }
            _ => {
                if name.starts_with("VMbcnt")
                    && instruction
                        .sources
                        .iter()
                        .any(|source| source.kind == OperandKind::ScalarRegister)
                {
                    mbcnt_mask += 1;
                }
                if name.ends_with("B64")
                    && ["SAnd", "SOr", "SXor", "SNand", "SNor", "SXnor"]
                        .iter()
                        .any(|prefix| name.starts_with(prefix))
                {
                    mask64_ops += 1;
                }
            }
        }
    }
    if readlane + readfirstlane + mbcnt_mask + bcnt1_b64 + scc_branch + exec_branch + mask64_ops
        == 0
    {
        return;
    }
    eprintln!(
        "[REMu][WAVE64-CENSUS] stage={stage} program={:#x} readlane={readlane} readfirstlane={readfirstlane} mbcnt_mask={mbcnt_mask} bcnt1_b64={bcnt1_b64} scc_branch={scc_branch} exec_branch={exec_branch} mask64_ops={mask64_ops} barriers={barriers}",
        program.address
    );
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
                && !matches!(
                    name,
                    "SAndSaveexecB64"
                        | "SOrn2SaveexecB64"
                        | "SAndSaveexecB32"
                        | "SOrSaveexecB32"
                        | "SXorSaveexecB32"
                        | "SAndn2SaveexecB32"
                        | "SOrn2SaveexecB32"
                        | "SNandSaveexecB32"
                        | "SNorSaveexecB32"
                        | "SXnorSaveexecB32"
                        | "SAndn1SaveexecB32"
                        | "SOrn1SaveexecB32"
                )
        });
        if wave_sensitive {
            blockers.push(format!("{:#x}:{}", instruction.pc, instruction.opcode));
        }
    }
    blockers
}

/// When evaluation left a four-byte null stub beside a live binding for the same SGPR slot,
/// prefer the live span for Metal length/base metadata (Poppy glyph PS b2→b0 tofu).
pub(crate) fn effective_global_memory_binding<'a>(
    evaluation: &'a ShaderEvaluation,
    binding: &'a remu_shader::GlobalMemoryBinding,
) -> &'a remu_shader::GlobalMemoryBinding {
    if binding.base_address != 0 || binding.byte_count > 4 {
        return binding;
    }
    evaluation
        .global_memory_bindings
        .iter()
        .find(|candidate| {
            candidate.scalar_address == binding.scalar_address && candidate.base_address != 0
        })
        .unwrap_or(binding)
}

/// Resolves every packet binding to its effective cached-evaluation index in one linear pass.
///
/// Null stubs can alias a later live binding on the same scalar slot. Returning indices lets
/// packet sidecars follow that alias without a pointer scan for each upload.
pub(crate) fn effective_global_memory_binding_indices(
    evaluation: &ShaderEvaluation,
) -> Vec<usize> {
    let mut live_by_scalar = HashMap::new();
    for (index, binding) in evaluation.global_memory_bindings.iter().enumerate() {
        if binding.base_address != 0 {
            live_by_scalar.entry(binding.scalar_address).or_insert(index);
        }
    }
    evaluation
        .global_memory_bindings
        .iter()
        .enumerate()
        .map(|(index, binding)| {
            if binding.base_address != 0 || binding.byte_count > 4 {
                index
            } else {
                live_by_scalar
                    .get(&binding.scalar_address)
                    .copied()
                    .unwrap_or(index)
            }
        })
        .collect()
}

pub(crate) fn shader_scalar_metadata(evaluation: &ShaderEvaluation) -> Result<Vec<u32>, String> {
    let mut words = evaluation.initial_scalar_registers.to_vec();
    words.reserve(evaluation.global_memory_bindings.len().saturating_mul(4));
    for binding in &evaluation.global_memory_bindings {
        let effective = effective_global_memory_binding(evaluation, binding);
        if !std::ptr::eq(effective, binding) {
            static ALIASED: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
            let sequence = ALIASED.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
            if sequence <= 16 || sequence.is_power_of_two() {
                eprintln!(
                    "[REMu][METAL] alias null global metadata sequence={sequence} scalar={} placeholder_bytes={} -> live base={:#x} bytes={}",
                    binding.scalar_address,
                    binding.byte_count,
                    effective.base_address,
                    effective.byte_count
                );
            }
        }
        words.extend([
            effective.base_address as u32,
            (effective.base_address >> 32) as u32,
            u32::try_from(effective.byte_count)
                .map_err(|_| "shader global binding exceeds the 32-bit Metal byte range")?,
            u32::from(effective.writable),
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
    for (offset, word) in words[scalar_register_base..scalar_end]
        .iter_mut()
        .enumerate()
    {
        let offset =
            u32::try_from(offset).map_err(|_| "shader scalar register offset does not fit u32")?;
        *word = register_value(shader_registers, user_data_register + offset).unwrap_or_default();
    }
    // Live SH/user-data reloads restore guest NUM_RECORDS after evaluation expanded the Metal
    // buffer for instanced vertex fetches. Without lifting NUM_RECORDS to the bound span,
    // OOB_SELECT=0/1 still zeroes every instance past the guest record count (Poppy UI tofu).
    lift_expanded_vertex_buffer_num_records(&mut words, evaluation);
    Ok(words)
}

/// When evaluation bound more bytes than `NUM_RECORDS * stride`, rewrite `NUM_RECORDS` so Metal
/// buffer OOB checks accept the same instance-indexed records the host buffer now covers.
pub(crate) fn lift_expanded_vertex_buffer_num_records(
    scalar_metadata: &mut [u32],
    evaluation: &ShaderEvaluation,
) {
    for binding in &evaluation.global_memory_bindings {
        let scalar = usize::try_from(binding.scalar_address).unwrap_or(usize::MAX);
        if scalar.saturating_add(3) >= scalar_metadata.len() || binding.byte_count <= 4 {
            continue;
        }
        let stride = (scalar_metadata[scalar + 1] >> 16) & 0x3fff;
        if stride == 0 {
            continue;
        }
        let oob_select = (scalar_metadata[scalar + 3] >> 28) & 3;
        // Raw (OOB_SELECT=3) stores a byte extent in word2; do not reinterpret it as records.
        if oob_select == 3 {
            continue;
        }
        let num_records = scalar_metadata[scalar + 2];
        let descriptor_bytes = u64::from(num_records).saturating_mul(u64::from(stride));
        let bound_bytes = u64::try_from(binding.byte_count).unwrap_or(0);
        if bound_bytes <= descriptor_bytes {
            continue;
        }
        let expanded_records =
            u32::try_from(bound_bytes / u64::from(stride)).unwrap_or(num_records);
        if expanded_records > num_records {
            scalar_metadata[scalar + 2] = expanded_records;
        }
    }
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
                .then_some(match instruction.control.as_ref() {
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

/// SPI register value for an unmatched PS input when the export stage has no EXP PARAM producer.
///
/// Matches `agc_create_interpolant_mapping` / SDK `CxPsShaderUsage` when the geometry shader has
/// no matching output semantic: `UseDefault0` with the programmed `Default0` selector, no flat
/// shade, and a cleared PARAM offset. Compacted-record staging only relocates EXP payloads, so a
/// missing EXP PARAM is exactly this unmatched-semantic case — not a RemuVsOut zero-fill.
pub(crate) fn unmatched_param_default_control(raw: u32) -> u32 {
    (raw & 0x0000_0300) | 0x20
}

/// Reconcile `SPI_PS_INPUT_CNTL` with EXP PARAM producers in the export program.
///
/// - Physical PARAM slots that EXP wrote keep their programmed controls (compacted staging
///   publishes those payloads under proven `NggRegisters`).
/// - Slots without an EXP producer and without general LDS traffic become architectural defaults
///   (`UseDefault0`), matching SDK interpolant mapping when GS output semantics do not match.
/// - Slots without an EXP producer **with** general LDS remain missing: LDS attribute rings are a
///   real producer class REmu has not lowered, so callers must refuse loudly rather than invent
///   UVs or zero-fill `RemuVsOut`.
pub(crate) fn reconcile_pixel_input_controls_with_export_producers(
    controls: &[u32; 32],
    exported_parameters: u32,
    interpolated_attributes: u32,
    export_uses_general_lds: bool,
) -> ([u32; 32], Vec<(u32, u32, u32)>) {
    let mut reconciled = *controls;
    let mut lds_missing = Vec::new();
    for (logical, physical, raw) in
        missing_vertex_parameter_exports(exported_parameters, interpolated_attributes, controls)
    {
        if export_uses_general_lds {
            lds_missing.push((logical, physical, raw));
            continue;
        }
        reconciled[logical as usize] = unmatched_param_default_control(raw);
    }
    (reconciled, lds_missing)
}

/// Recognizes a fused GFX10 NGG export/geometry program that requires a Metal mesh stage.
///
/// The allocation message and primitive export are the architectural contract that distinguishes
/// an NGG geometry stage from an ordinary vertex stage. General LDS traffic is optional: compact
/// pass-through stages can publish connectivity and position directly from their merged-wave
/// inputs without staging records through LDS. Lowering those stages as ordinary Metal vertices
/// loses the merged-wave system values and can collapse every output position to zero.
pub(crate) fn ngg_mesh_export_stage(program: &Program) -> bool {
    let mut allocation_message = false;
    let mut primitive_export = false;
    let mut position_export = false;
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
    }
    allocation_message && primitive_export && position_export
}

/// Returns whether an export program manipulates the architectural 64-bit EXEC pair.
///
/// Prospero NGG exports ship in both wave sizes. Wave32 programs touch EXEC only through the
/// `_B32` scalar forms, while wave64 programs build full 64-bit masks (`S_BFE_U64`/`S_MOV_B64`
/// into EXEC, or the 64-bit SAVEEXEC forms) and recover their subgroup thread index as
/// `s3[27:24] * 64 + v_mbcnt`. The mesh lowering must synthesize the merged wave info with the
/// matching granularity, so this classification selects its architectural lane count.
pub(crate) fn ngg_export_uses_wave64_exec(program: &Program) -> bool {
    const EXEC_LO: u32 = 126;
    program.instructions.iter().any(|instruction| {
        let Some(name) = instruction.opcode.known_name() else {
            return false;
        };
        if !(name.ends_with("B64") || name.ends_with("U64") || name.ends_with("I64")) {
            return false;
        }
        name.contains("Saveexec")
            || instruction.destinations.first().is_some_and(|destination| {
                destination.kind == OperandKind::ScalarRegister && destination.value == EXEC_LO
            })
    })
}

/// True when an NGG export stages geometry records through general LDS (not ADDTID spills).
///
/// This distinguishes pass-through NGG from stages that exchange ES/GS records; it does not
/// select the host stage. Even LDS-free NGG still consumes merged-wave system values and emits
/// primitive connectivity, so non-procedural programs require the mesh ABI. Also the producer
/// gate for PARAM-less SPI reconciliation: general LDS may carry an attribute ring that EXP does
/// not name, so those holes stay loud-refused instead of UseDefault0.
pub(crate) fn ngg_export_uses_general_lds(program: &Program) -> bool {
    program.instructions.iter().any(|instruction| {
        matches!(instruction.control, Some(InstructionControl::DataShare(_)))
            && !matches!(
                instruction.opcode.known_name(),
                Some("DsSwizzleB32" | "DsWriteAddtidB32" | "DsReadAddtidB32")
            )
    })
}

/// True when an NGG export only computes cover geometry (no memory/image/LDS traffic).
///
/// Poppy-style UE fullscreen covers arrive as NGG-shaped programs that derive position from the
/// vertex ID alone. Dreaming Sarah UI sprites also use NGG markers with indexed two-triangle
/// quads, but they fetch per-vertex attributes from buffers — those must keep the mesh path so
/// sprite bounds survive. Topology alone cannot distinguish the two.
pub(crate) fn ngg_memoryless_cover_export_stage(program: &Program) -> bool {
    if !ngg_mesh_export_stage(program) {
        return false;
    }
    for instruction in &program.instructions {
        if matches!(
            instruction.control.as_ref(),
            Some(
                InstructionControl::Image(_)
                    | InstructionControl::GlobalMemory(_)
                    | InstructionControl::BufferMemory(_)
                    | InstructionControl::DataShare(_),
            )
        ) {
            return false;
        }
        if let Some(name) = instruction.opcode.known_name()
            && (name.starts_with("Ds")
                || name.starts_with("Image")
                || name.starts_with("Buffer")
                || name.starts_with("SBuffer")
                || name.starts_with("Global")
                // Flat scalar/constant traffic is common in UE fullscreen covers that still derive
                // position from the vertex ID. Buffer/Image/Global fetches remain disqualifying so
                // sprite quads that pull per-vertex attributes stay on the ordinary VS path.
                || name.starts_with("Scratch"))
        {
            return false;
        }
    }
    true
}

/// Returns whether a direct native draw can be represented by the CPU-prepared NGG meshlet ABI.
pub(crate) const fn ngg_mesh_draw_supported(draw: &IndexedDraw) -> bool {
    match (draw.indexed, draw.primitive_type) {
        // Triangle lists/strips expand into the meshlet triangle ABI.
        (true, 4 | 6) => true,
        // Point lists feed one vertex per input primitive; GS/NGG amplification emits triangles.
        (true, 1) | (false, 1) => true,
        // Non-indexed strips synthesize consecutive vertex IDs, then expand the same way.
        (false, 6) => true,
        _ => false,
    }
}

/// Selects NGG mesh lowering only when the presenter can prepare exact ordered source geometry.
pub(crate) fn ngg_mesh_stage_for_draw(
    program: &Program,
    draw: &IndexedDraw,
) -> Result<bool, String> {
    if !ngg_mesh_export_stage(program) {
        return Ok(false);
    }
    if ngg_mesh_draw_supported(draw) {
        return Ok(true);
    }
    // Three-vertex AMD rectangle lists expand to a four-corner Metal triangle strip. The guest
    // export still looks like an NGG stage (GS_ALLOC_REQ + prim EXP), but the host runs it as an
    // ordinary vertex shader so per-draw rectangle bounds from the export survive.
    if !draw.indexed
        && draw.indirect_arguments_address.is_none()
        && draw.index_count == 3
        && matches!(draw.primitive_type, 7 | 0x11)
    {
        return Ok(false);
    }
    // Non-indexed triangle lists still prefer VS when the meshlet ABI cannot prepare them.
    if !draw.indexed && draw.indirect_arguments_address.is_none() && draw.primitive_type == 4 {
        return Ok(false);
    }
    if procedural_rectangle_stage_for_draw(program, draw) {
        return Ok(false);
    }
    Err(format!(
        "draw {} uses unsupported topology {:#x} (indexed={}) with an NGG mesh stage",
        draw.sequence, draw.primitive_type, draw.indexed,
    ))
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

/// Poppy/UE memoryless NGG rectangle covers: GS_ALLOC + prim EXP, no memory/PARAM traffic.
///
/// Real binaries are often a short live prefix (sometimes without a position EXP) followed by
/// padded/undecoded tails. Rejecting on `Opcode::known_name() == None` missed every Poppy cover
/// and previously left INTERFACE-FILL zeroes on scanout (mid-band noise). Phase 1E deletes that
/// zero-fill; these covers must take the synthesized rectangle vertex path instead.
pub(crate) fn truncated_memoryless_ngg_rectangle_cover(program: &Program) -> bool {
    let mut allocation_message = false;
    let mut primitive_export = false;
    for instruction in &program.instructions {
        match instruction.control.as_ref() {
            Some(InstructionControl::Export(export)) => match export.target {
                20 if export.enable_mask != 0 => primitive_export = true,
                // Position may appear in longer UE covers; PARAMs mean a real attribute path.
                12 => {}
                32..=63 if export.enable_mask != 0 => return false,
                _ => {}
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
            Some(name)
                if name.starts_with("Ds")
                    || name.starts_with("Image")
                    || name.starts_with("Buffer")
                    || name.starts_with("SBuffer")
                    || name.starts_with("Global")
                    || name.starts_with("Scratch") =>
            {
                return false;
            }
            // Padding / undecoded tails after the live cover prefix.
            None => {}
            _ => {}
        }
    }
    allocation_message && primitive_export
}

/// Returns whether this exact draw uses the synthesized procedural-rectangle vertex stage.
pub(crate) fn procedural_rectangle_stage_for_draw(program: &Program, draw: &IndexedDraw) -> bool {
    !draw.indexed
        && draw.indirect_arguments_address.is_none()
        && draw.index_count == 3
        && matches!(draw.primitive_type, 7 | 0x11)
        && (procedural_rectangle_export_stage(program)
            || truncated_memoryless_ngg_rectangle_cover(program))
}

#[cfg(test)]
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
