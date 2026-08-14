// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

//! Compiler construction and the per-instruction lowering loop: interface
//! validation, initial register state and the opcode class dispatch.

use std::collections::{BTreeMap, BTreeSet};

use remu_shader::{
    BasicBlock, Instruction, InstructionControl, OperandKind, Program, ShaderEvaluation,
    StructuredCfg, image_is_comparison,
};

use crate::compiler::control_flow::{
    ControlFlowLowering, LoweringFailure, analyze_control_flow, branch_target,
    combined_lowering_error, dispatcher_basic_blocks, scalar_branch_taken_condition,
};
use crate::compiler::export::{mesh_export_vertex_vgpr, mesh_ngg_packed_export_sgpr};
use crate::compiler::image::{
    ImageKind, heap_texture_table_type, image_component_type, is_storage_image, is_writable_image,
};
use crate::compiler::interface::{PixelInterpolation, interpolation_for_barycentric_vgpr};
use crate::compiler::memory::is_lds_memory_instruction;
use crate::compiler::{
    HeapTextureRoute, METAL_THREADGROUP_MEMORY_LIMIT_BYTES, SPLIT_WAVE64_PAIR_STATE_BYTES,
    is_full_exec_restore, scalar_memory_image_descriptor_offset,
};
use crate::{
    Compiler, HEAP_TEXTURE_DESCRIPTOR_STRIDE_BYTES, HeapTextureTableBinding,
    METAL_DIRECT_TEXTURE_LIMIT, MetalBindingLayout, MetalCompileError, MetalCompileOptions,
    MetalRegisterFootprint, MetalShader, MetalShaderStage, PIXEL_INPUT_CONTROL_COUNT,
    SCALAR_REGISTER_COUNT, VECTOR_REGISTER_COUNT,
};

impl<'a> Compiler<'a> {
    fn emit_diagnostic_compute_state_trace(&mut self, pc: u32) {
        let Some(checkpoint) = self.diagnostic_compute_state_trace_checkpoint(pc) else {
            return;
        };
        let Some([group_x, group_y, group_z]) = self.diagnostic_compute_state_trace_group() else {
            return;
        };
        self.line(
            3,
            &format!(
                "if (remu_group_id.x == {group_x}u && remu_group_id.y == {group_y}u && remu_group_id.z == {group_z}u) {{"
            ),
        );
        self.line(
            4,
            &format!("uint remu_trace_base = (({checkpoint}u * 64u + remu_thread_index) * 128u);"),
        );
        for (offset, value) in [
            format!("{pc}u"),
            "exec ? 1u : 0u".to_owned(),
            "vcc ? 1u : 0u".to_owned(),
            "scc ? 1u : 0u".to_owned(),
            "remu_thread_index".to_owned(),
            "remu_wave_half".to_owned(),
            "remu_lane".to_owned(),
            "0u".to_owned(),
        ]
        .into_iter()
        .enumerate()
        {
            self.line(
                4,
                &format!("remu_debug_compute[remu_trace_base + {offset}u] = {value};"),
            );
        }
        for register in 0..48u32 {
            self.line(
                4,
                &format!(
                    "remu_debug_compute[remu_trace_base + {}u] = v[{register}];",
                    8 + register
                ),
            );
        }
        for register in 0..64u32 {
            self.line(
                4,
                &format!(
                    "remu_debug_compute[remu_trace_base + {}u] = s[{register}];",
                    56 + register
                ),
            );
        }
        for (offset, register) in [106u32, 107, 126, 127].into_iter().enumerate() {
            self.line(
                4,
                &format!(
                    "remu_debug_compute[remu_trace_base + {}u] = s[{register}];",
                    120 + offset
                ),
            );
        }
        self.line(3, "}");
    }

    pub(crate) fn new(
        program: &'a Program,
        evaluation: &'a ShaderEvaluation,
        options: &'a MetalCompileOptions,
    ) -> Result<Self, MetalCompileError> {
        if program.instructions.is_empty() {
            return Err(MetalCompileError::EmptyProgram);
        }
        validate_interface(options)?;
        if let Some(system_registers) = evaluation.compute_system_registers {
            if options.stage != MetalShaderStage::Compute {
                return Err(MetalCompileError::InvalidInterface(
                    "compute system SGPRs require a compute shader stage".into(),
                ));
            }
            for register in [
                system_registers.work_group_x,
                system_registers.work_group_y,
                system_registers.work_group_z,
                system_registers.thread_group_size,
            ]
            .into_iter()
            .flatten()
            {
                if register >= SCALAR_REGISTER_COUNT {
                    return Err(MetalCompileError::InvalidInterface(format!(
                        "compute system SGPR s{register} exceeds the scalar register file"
                    )));
                }
            }
        }

        let mut vertex_outputs = BTreeSet::new();
        let mut pixel_attributes = BTreeSet::new();
        let mut pixel_interpolations = BTreeMap::new();
        for instruction in &program.instructions {
            match instruction.control.as_ref() {
                Some(InstructionControl::Export(export))
                    if matches!(
                        options.stage,
                        MetalShaderStage::Vertex | MetalShaderStage::Mesh
                    ) && (32..64).contains(&export.target) =>
                {
                    vertex_outputs.insert(export.target - 32);
                }
                Some(InstructionControl::Interpolation(interpolation))
                    if options.stage == MetalShaderStage::Pixel =>
                {
                    let attribute_index =
                        usize::try_from(interpolation.attribute).map_err(|_| {
                            MetalCompileError::InvalidInterface(format!(
                                "pixel input attribute {} does not fit the host index",
                                interpolation.attribute
                            ))
                        })?;
                    let Some(input) = options.pixel_input_controls.get(attribute_index).copied()
                    else {
                        return Err(MetalCompileError::InvalidInterface(format!(
                            "pixel input attribute {} exceeds the {} SPI_PS_INPUT_CNTL registers",
                            interpolation.attribute, PIXEL_INPUT_CONTROL_COUNT
                        )));
                    };
                    if input.input_offset >= 32 {
                        return Err(MetalCompileError::InvalidInterface(format!(
                            "pixel input attribute {} maps to invalid parameter {}",
                            interpolation.attribute, input.input_offset
                        )));
                    }
                    if input.default_value > 3 {
                        return Err(MetalCompileError::InvalidInterface(format!(
                            "pixel input attribute {} uses invalid default selector {}",
                            interpolation.attribute, input.default_value
                        )));
                    }
                    pixel_attributes.insert(interpolation.attribute);
                    if input.flat_shade {
                        pixel_interpolations
                            .insert(interpolation.attribute, PixelInterpolation::Flat);
                    } else if !input.uses_default() && instruction.opcode.starts_with("VInterpP2") {
                        let source = instruction.sources.first().copied().ok_or_else(|| {
                            MetalCompileError::InvalidInterface(format!(
                                "pc 0x{:X} {} has no P2 barycentric source",
                                instruction.pc, instruction.opcode
                            ))
                        })?;
                        if source.kind != OperandKind::VectorRegister {
                            return Err(MetalCompileError::InvalidInterface(format!(
                                "pc 0x{:X} {} P2 barycentric source is not a VGPR",
                                instruction.pc, instruction.opcode
                            )));
                        }
                        let interpolation_mode = interpolation_for_barycentric_vgpr(
                            options.pixel_input_address,
                            source.value,
                        )
                        .map_err(MetalCompileError::InvalidInterface)?;
                        if let Some(previous) =
                            pixel_interpolations.insert(interpolation.attribute, interpolation_mode)
                            && previous != interpolation_mode
                        {
                            return Err(MetalCompileError::InvalidInterface(format!(
                                "pixel input attribute {} mixes {:?} and {:?} interpolation",
                                interpolation.attribute, previous, interpolation_mode
                            )));
                        }
                    }
                }
                _ => {}
            }
        }
        if matches!(
            options.stage,
            MetalShaderStage::Vertex | MetalShaderStage::Mesh
        ) {
            // Guest EXP PARAM targets already populated `vertex_outputs` above. Only a host-
            // synthesized interface (procedural rectangle, layered cover) passes a required count
            // without those exports — never pad missing NGG attributes with zero-initialized
            // RemuVsOut fields.
            if vertex_outputs.is_empty() && options.required_vertex_output_count > 0 {
                vertex_outputs.extend(0..options.required_vertex_output_count);
            }
        }

        let image_slots = evaluation.image_binding_slots();
        if image_slots.texture_count > METAL_DIRECT_TEXTURE_LIMIT {
            return Err(MetalCompileError::InvalidInterface(format!(
                "{} distinct image descriptors/types/access modes exceed Metal's {} direct \
                 texture slots",
                image_slots.texture_count, METAL_DIRECT_TEXTURE_LIMIT
            )));
        }
        let mut image_kinds = Vec::with_capacity(
            usize::try_from(image_slots.texture_count)
                .map_err(|_| MetalCompileError::InvalidInterface("too many textures".into()))?,
        );
        for (binding_index, binding) in evaluation.image_bindings.iter().enumerate() {
            let texture_slot = usize::try_from(image_slots.texture_slots[binding_index])
                .map_err(|_| MetalCompileError::InvalidInterface("texture slot overflow".into()))?;
            if texture_slot == image_kinds.len() {
                let storage = is_storage_image(binding.opcode);
                let writable = is_writable_image(binding.opcode);
                let comparison = image_is_comparison(binding.opcode);
                image_kinds.push(ImageKind {
                    component: image_component_type(binding),
                    storage,
                    writable,
                    comparison,
                    dimension: binding.control.dimension,
                    texture_type: binding
                        .resource_descriptor
                        .get(3)
                        .map_or(0, |word| (word >> 28) & 0xf),
                });
            } else if texture_slot > image_kinds.len() {
                return Err(MetalCompileError::InvalidInterface(
                    "non-dense canonical texture slots".into(),
                ));
            }
        }
        let control_flow = analyze_control_flow(program, options.stage)?;
        let split_wave64 = matches!(
            options.stage,
            MetalShaderStage::Compute | MetalShaderStage::Mesh
        ) && options.wave_lane_count == 64;
        let needs_split_wave64_pair_rendezvous = split_wave64
            && ((options.stage == MetalShaderStage::Compute
                && program.instructions.iter().any(|instruction| {
                    matches!(
                        instruction.opcode.known_name(),
                        Some("SCbranchVccz" | "SCbranchVccnz" | "SCbranchExecz" | "SCbranchExecnz")
                    )
                }))
                || (options.stage == MetalShaderStage::Compute
                    && !(1..=64).contains(&options.compute_threadgroup_size)
                    && program.instructions.iter().any(is_lds_memory_instruction)
                    && program.instructions.iter().any(is_full_exec_restore)));
        if needs_split_wave64_pair_rendezvous
            && program.instructions.iter().any(is_lds_memory_instruction)
            && usize::try_from(options.lds_size_bytes)
                .ok()
                .and_then(|bytes| bytes.checked_add(SPLIT_WAVE64_PAIR_STATE_BYTES))
                .is_none_or(|bytes| bytes > METAL_THREADGROUP_MEMORY_LIMIT_BYTES)
        {
            return Err(MetalCompileError::InvalidInterface(format!(
                "wave64 paired-half rendezvous requires {} bytes of state in addition to {} bytes of guest LDS, exceeding Metal's {}-byte threadgroup-memory limit",
                SPLIT_WAVE64_PAIR_STATE_BYTES,
                options.lds_size_bytes,
                METAL_THREADGROUP_MEMORY_LIMIT_BYTES
            )));
        }
        // The GFX10 empty-workaround count rewrite stays deleted: detecting that pattern
        // without proven registers is a loud refuse; with proof the guest GS_ALLOC_REQ / M0
        // sequence lowers as-is.
        if options.stage == MetalShaderStage::Mesh
            && !options.ngg_registers_proven
            && !options.ngg_fixed_allocation_fallback
            && mesh_ngg_packed_export_sgpr(program).is_some()
        {
            return Err(MetalCompileError::InvalidInterface(
                "NGG GFX10 empty-workaround / packed export-count sequence requires proven NggRegisters (Q5 mesh_from_ngg_registers); refusing the title-specific snapshot/restore peephole"
                    .into(),
            ));
        }
        // PRIM exports address the compacted vertex domain when the hardware allocation ABI is
        // proven. The fixed no-GS compatibility path intentionally retains the original one
        // vertex record per invocation contract: treating an ordinary V_CMPX source as a
        // compacted output id changes the pre-subgroup-allocation ABI used by older titles.
        let mesh_export_vertex_vgpr = (options.stage == MetalShaderStage::Mesh
            && options.ngg_registers_proven
            && !options.ngg_fixed_allocation_fallback)
            .then(|| mesh_export_vertex_vgpr(program))
            .flatten();

        // Runtime heap tables (docs/design/2026-08-13-runtime-image-descriptor-fetch.md):
        // Map every canonical texture slot to exactly one lowering domain. A slot cannot mix a
        // fixed snapshot use with a runtime descriptor use, nor can two runtime uses disagree on
        // their descriptor SGPR/heaps. Canonicalization is intentionally broader than runtime
        // routing, so disagreement is a loud interface refusal rather than a silent fallback.
        //
        // The map is keyed by CANONICAL TEXTURE SLOT — the domain `texture_expression` looks up,
        // and the one `image_kinds` and the `tex{N}` parameters already use. An evaluation index
        // is a different domain: `image_binding_slots` merges bindings that share a descriptor,
        // so the two agree only when that map is the identity, which it exists not to be.
        let mut heap_tables = std::collections::BTreeMap::new();
        if options.runtime_heap_tables {
            // `None` is the snapshot lowering; `Some` is one proven runtime route.
            let mut routes: std::collections::BTreeMap<usize, Option<HeapTextureRoute>> =
                std::collections::BTreeMap::new();
            for (evaluation_index, binding) in evaluation.image_bindings.iter().enumerate() {
                let texture_slot = usize::try_from(image_slots.texture_slots[evaluation_index])
                    .map_err(|_| {
                        MetalCompileError::InvalidInterface("texture slot overflow".into())
                    })?;
                let route = if binding.descriptor_index_is_runtime {
                    let base = binding.control.scalar_resource;
                    let mut heaps = std::collections::BTreeSet::new();
                    for instruction in &program.instructions {
                        if !matches!(
                            instruction.control.as_ref(),
                            Some(remu_shader::InstructionControl::ScalarMemory(_))
                        ) {
                            continue;
                        }
                        let Some(descriptor_tuple_offset) =
                            scalar_memory_image_descriptor_offset(instruction, base)
                                .map_err(MetalCompileError::InvalidInterface)?
                        else {
                            continue;
                        };
                        if let Some(heap) = evaluation
                            .global_memory_bindings
                            .iter()
                            .position(|global| global.instruction_pcs.contains(&instruction.pc))
                        {
                            let Some(remu_shader::InstructionControl::ScalarMemory(control)) =
                                instruction.control.as_ref()
                            else {
                                unreachable!("descriptor discovery already selected an SMEM load");
                            };
                            let dynamic_offset = if let Some(register) =
                                control.dynamic_offset_register
                            {
                                evaluation
                                    .scalar_registers
                                    .get(register as usize)
                                    .copied()
                                    .ok_or_else(|| {
                                        MetalCompileError::InvalidInterface(format!(
                                            "runtime image at pc {:#x} uses missing dynamic offset s{register}",
                                            binding.pc
                                        ))
                                    })?
                            } else {
                                0
                            };
                            let observed_descriptor_offset = dynamic_offset
                                .wrapping_add(control.immediate_offset_bytes.cast_unsigned())
                                .wrapping_add(descriptor_tuple_offset);
                            if !observed_descriptor_offset
                                .is_multiple_of(HEAP_TEXTURE_DESCRIPTOR_STRIDE_BYTES)
                            {
                                return Err(MetalCompileError::InvalidInterface(format!(
                                    "runtime image at pc {:#x} has descriptor phase {} at SMEM pc {:#x}, but its heap binding publishes {}-byte records from phase zero",
                                    binding.pc,
                                    observed_descriptor_offset
                                        % HEAP_TEXTURE_DESCRIPTOR_STRIDE_BYTES,
                                    instruction.pc,
                                    HEAP_TEXTURE_DESCRIPTOR_STRIDE_BYTES,
                                )));
                            }
                            heaps.insert(heap);
                        }
                    }
                    if heaps.is_empty() {
                        return Err(MetalCompileError::InvalidInterface(format!(
                            "runtime image at pc {:#x} uses s{base} without a complete SMEM heap descriptor load",
                            binding.pc
                        )));
                    }
                    Some(HeapTextureRoute {
                        scalar_resource_base: base,
                        heap_binding_indices: heaps.into_iter().collect(),
                    })
                } else {
                    None
                };
                match routes.entry(texture_slot) {
                    std::collections::btree_map::Entry::Vacant(entry) => {
                        entry.insert(route);
                    }
                    std::collections::btree_map::Entry::Occupied(entry) => {
                        if entry.get() != &route {
                            return Err(MetalCompileError::InvalidInterface(format!(
                                "canonical texture slot {texture_slot} mixes snapshot/runtime descriptor routes at image pc {:#x}",
                                binding.pc
                            )));
                        }
                    }
                }
            }
            heap_tables.extend(
                routes
                    .into_iter()
                    .filter_map(|(slot, route)| route.map(|route| (slot, route))),
            );
            // A key outside the canonical slot domain would index the wrong `image_kinds` entry
            // or panic in `emit_image`; refuse the lowering instead.
            if let Some(slot) = heap_tables.keys().find(|slot| **slot >= image_kinds.len()) {
                return Err(MetalCompileError::InvalidInterface(format!(
                    "heap texture table keyed by {slot}, outside the {} canonical texture slots",
                    image_kinds.len()
                )));
            }
        }

        let compiler = Self {
            program,
            evaluation,
            options,
            body: String::new(),
            next_temp: 0,
            vertex_outputs,
            pixel_attributes,
            pixel_interpolations,
            image_kinds,
            image_slots,
            heap_tables,
            control_flow,
            mesh_export_vertex_vgpr,
        };
        let total_threadgroup_memory = compiler.total_threadgroup_memory_bytes();
        if total_threadgroup_memory.is_none_or(|bytes| bytes > METAL_THREADGROUP_MEMORY_LIMIT_BYTES)
        {
            return Err(MetalCompileError::InvalidInterface(format!(
                "guest LDS plus compiler-generated state need {} bytes of threadgroup memory, exceeding Metal's {}-byte limit",
                total_threadgroup_memory.unwrap_or(usize::MAX),
                METAL_THREADGROUP_MEMORY_LIMIT_BYTES
            )));
        }
        Ok(compiler)
    }

    pub(crate) fn compile(mut self) -> Result<MetalShader, MetalCompileError> {
        // Lowering continues past a failed instruction so that one attempt names every opcode a
        // shader needs. A body produced after a failure is only ever discarded with the error.
        let mut failures = Vec::new();
        let mut failed_opcodes = BTreeSet::new();
        match self.control_flow.clone() {
            ControlFlowLowering::Direct(indices) => {
                for index in indices {
                    let instruction = &self.program.instructions[index];
                    if let Err(message) = self.emit_instruction(instruction, index, false)
                        && failed_opcodes.insert(instruction.opcode)
                    {
                        failures.push(LoweringFailure {
                            pc: instruction.pc,
                            opcode: instruction.opcode,
                            message,
                        });
                    }
                }
            }
            ControlFlowLowering::Structured { tree, blocks } => {
                if let Err(message) =
                    self.emit_structured_cfg(&tree, &blocks, 2, &mut failures, &mut failed_opcodes)
                {
                    // Structural emission failure (missing condition block, etc.) — surface as a
                    // loud interface error rather than a silently truncated dispatcher body.
                    return Err(MetalCompileError::InvalidInterface(message));
                }
            }
            ControlFlowLowering::Forward | ControlFlowLowering::Dispatcher => {
                // Dispatch whole basic blocks instead of one instruction per switch iteration.
                // The architectural PC only has to be materialized at control-flow boundaries;
                // straight-line instructions inside a block execute in their original order.
                // Large Unity pixel shaders contain hundreds of instructions but only a handful
                // of branches, so this avoids hundreds of divergent switch operations per pixel
                // without changing branch, loop, or EXEC/VCC semantics.
                let forward = matches!(self.control_flow, ControlFlowLowering::Forward);
                for block in dispatcher_basic_blocks(self.program) {
                    let first = &self.program.instructions[block.start];
                    let block_instruction_count = (!forward).then_some(block.len());
                    if forward {
                        self.line(2, &format!("if (active && pc == {}u) {{", first.pc));
                    } else {
                        self.line(2, &format!("case {}u: {{", first.pc));
                    }
                    let last = block.end - 1;
                    for index in block {
                        let instruction = &self.program.instructions[index];
                        if let Err(message) =
                            self.emit_instruction(instruction, index, index == last)
                            && failed_opcodes.insert(instruction.opcode)
                        {
                            failures.push(LoweringFailure {
                                pc: instruction.pc,
                                opcode: instruction.opcode,
                                message,
                            });
                        }
                    }
                    if let Some(block_instruction_count) = block_instruction_count {
                        self.line(3, &format!("steps += {block_instruction_count}u;"));
                        self.line(3, "break;");
                    }
                    self.line(2, "}");
                }
            }
        }
        if let Some(error) = combined_lowering_error(failures) {
            return Err(error);
        }

        let direct_control_flow = matches!(
            self.control_flow,
            ControlFlowLowering::Direct(_) | ControlFlowLowering::Structured { .. }
        );
        let (source, register_references) = self.emit_module()?;
        let global_buffer_count = u32::try_from(self.evaluation.global_memory_bindings.len())
            .map_err(|_| MetalCompileError::InvalidInterface("too many global buffers".into()))?;
        let sampler_count = self.image_slots.sampler_count;
        let global_argument_buffer_index = (global_buffer_count > 30).then_some(0);
        let initial_scalar_buffer_index = if global_argument_buffer_index.is_some() {
            1
        } else {
            global_buffer_count
        };
        let direct_sampler_bindings =
            sampler_count != 0 && initial_scalar_buffer_index.saturating_add(1) > 30;
        let threadgroup_memory_bytes = if self
            .program
            .instructions
            .iter()
            .any(is_lds_memory_instruction)
        {
            self.options.lds_size_bytes
        } else {
            0
        };
        Ok(MetalShader {
            source,
            entry_point: "remu_main".into(),
            stage: self.options.stage,
            bindings: MetalBindingLayout {
                global_buffer_count,
                global_argument_buffer_index,
                initial_scalar_buffer_index,
                sampler_argument_buffer_index: (sampler_count != 0 && !direct_sampler_bindings)
                    .then_some(initial_scalar_buffer_index + 1),
                texture_count: self.image_slots.texture_count,
                sampler_count,
                threadgroup_memory_bytes,
                heap_texture_tables: {
                    let mut tables = Vec::new();
                    for (texture_slot, route) in &self.heap_tables {
                        for heap in &route.heap_binding_indices {
                            let ordinal = tables.len();
                            tables.push(HeapTextureTableBinding {
                                global_binding_index: u32::try_from(*heap)
                                    .expect("validated heap binding fits u32"),
                                buffer_index: u32::try_from(
                                    initial_scalar_buffer_index as usize + 2 + ordinal,
                                )
                                .expect("validated Metal buffer index fits u32"),
                                canonical_texture_slot: u32::try_from(*texture_slot)
                                    .expect("validated canonical texture slot fits u32"),
                                descriptor_stride_bytes: HEAP_TEXTURE_DESCRIPTOR_STRIDE_BYTES,
                                texture_type: heap_texture_table_type(
                                    self.image_kinds[*texture_slot],
                                ),
                            });
                        }
                    }
                    tables
                },
            },
            vertex_inputs: self.evaluation.vertex_inputs.clone(),
            interpolated_attribute_count: self
                .pixel_attributes
                .iter()
                .filter_map(|attribute| {
                    let input = self.options.pixel_input_controls[*attribute as usize];
                    (!input.uses_default()).then_some(u32::from(input.input_offset) + 1)
                })
                .max()
                .unwrap_or(0),
            register_footprint: MetalRegisterFootprint {
                scalar_values: u32::try_from(register_references.scalar.len()).map_err(|_| {
                    MetalCompileError::InvalidInterface("too many live scalar values".into())
                })?,
                vector_values: u32::try_from(register_references.vector.len()).map_err(|_| {
                    MetalCompileError::InvalidInterface("too many live vector values".into())
                })?,
                direct_control_flow,
            },
        })
    }

    fn emit_instruction(
        &mut self,
        instruction: &Instruction,
        index: usize,
        dispatched: bool,
    ) -> Result<(), String> {
        self.emit_diagnostic_compute_state_trace(instruction.pc);
        if self.options.stage == MetalShaderStage::Mesh
            && self
                .program
                .instructions
                .iter()
                .filter_map(|candidate| self.program.linked_transfer_target(candidate.pc))
                .any(|target| target == instruction.pc)
        {
            // A GFX9-11 merged ES/GS binary is two independent programs sharing one code
            // allocation. The ES half is per vertex and freely overwrites its VGPR inputs; when
            // hardware enters the GS half it supplies a different VGPR ABI. Re-seed that ABI at
            // the linked continuation entry, after the ES has populated LDS. Seeding these only
            // in the mesh-function prologue made the values stale by the time the GS decoded its
            // input vertex offsets, which produced valid-looking positions but null/garbled EXP
            // PRIM connectivity in Silent Hill's layered color-LUT draw.
            self.line(3, "// GFX9-11 linked GS continuation VGPR ABI");
            self.line(3, "v[0] = remu_mesh_gs_vtx_offset_0;");
            self.line(3, "v[1] = remu_mesh_gs_vtx_offset_1;");
            self.line(3, "v[2] = remu_mesh_primitive_base + remu_thread_index;");
            self.line(3, "v[3] = 0u;");
            self.line(3, "v[4] = 0u;");
        }
        let next_pc = self.program.instructions.get(index + 1).map(|next| next.pc);
        let name = instruction
            .opcode
            .known_name()
            .ok_or_else(|| "raw opcode has no lowering semantics".to_owned())?;

        match name {
            "SEndpgm" => {
                if dispatched {
                    self.line(3, "active = false;");
                }
                return Ok(());
            }
            "SSetpcB64" => {
                if dispatched {
                    if let Some(target) = self.program.linked_transfer_target(instruction.pc) {
                        self.line(3, &format!("pc = {target}u;"));
                    } else {
                        self.line(3, "active = false;");
                    }
                }
                return Ok(());
            }
            "SBranch" => {
                if dispatched {
                    let target = branch_target(instruction)?;
                    self.line(3, &format!("pc = {target}u;"));
                }
                return Ok(());
            }
            "SCbranchScc0" | "SCbranchScc1" | "SCbranchVccz" | "SCbranchVccnz"
            | "SCbranchExecz" | "SCbranchExecnz" => {
                if !dispatched {
                    // Structured lowering consumes the condition at the If/Loop node; the
                    // condition block emits every preceding instruction but not the branch itself.
                    return Err("conditional branch reached direct control-flow lowering".into());
                }
                let target = branch_target(instruction)?;
                let fallthrough =
                    next_pc.ok_or_else(|| "conditional branch has no fallthrough".to_owned())?;
                let condition = scalar_branch_taken_condition(name)?;
                self.line(
                    3,
                    &format!("pc = ({condition}) ? {target}u : {fallthrough}u;"),
                );
                return Ok(());
            }
            "SSendmsg" => {
                let message = instruction
                    .sources
                    .first()
                    .filter(|source| source.kind == OperandKind::EncodedConstant)
                    .map(|source| source.value)
                    .ok_or("S_SENDMSG requires an encoded message immediate")?;
                if message == 9 {
                    // NGG passthrough rectangle-list stages still emit GS_ALLOC_REQ even when the
                    // host expands the three-vertex AMD rect into a Metal triangle strip and runs
                    // the guest export as an ordinary vertex shader. Ignore the alloc on that
                    // path; only the mesh ABI needs the output primitive count.
                    if self.options.stage != MetalShaderStage::Mesh {
                        return Ok(());
                    }
                    // A no-API-GS passthrough stage has no VGT_GS_MAX_VERT_OUT proof for this
                    // payload. The known-good fixed-wave64 ABI publishes the meshlet's input
                    // primitive count and intentionally treats GS_ALLOC_REQ as a hardware-only
                    // provisioning message.
                    if self.options.ngg_fixed_allocation_fallback {
                        return Ok(());
                    }
                    // GFX10 GS_ALLOC_REQ is issued by subgroup zero. M0 carries the actual
                    // allocation: vertices[10:0], primitives[22:12]. Metal has no vertex-count
                    // setter and only thread zero publishes the primitive count, so this value
                    // can remain private to thread zero. A shared counter would exceed Metal's
                    // memory limit when the guest already uses the full 32 KiB of LDS.
                    self.line(3, "if (remu_thread_index == 0u) {");
                    self.line(
                        4,
                        "remu_mesh_output_primitive_count = (s[124] >> 12u) & 0x7ffu;",
                    );
                    self.line(3, "}");
                }
            }
            // These operations order guest hardware. MSL executes translated memory operations
            // in program order, so they require no separate host instruction.
            "SNop" | "SWaitcnt" | "SWaitcntDepctr" | "SInstPrefetch" | "STtraceData"
            | "SClause" | "VNop" => {}
            // Per-wave hardware state that Metal neither exposes nor needs: the remaining wait
            // counters, the scheduling, cache and performance-counter hints, and the rounding and
            // denormal mode registers, whose behaviour the host GPU fixes for us. S_VERSION is a
            // toolchain declaration, and S_SETREG writes an opaque hardware register from a field
            // the decoder parks in the scalar-destination slot, so it produces no register value
            // either. The evaluator already runs all of these as benign; refusing them here threw
            // away an entire draw over a scheduling hint. S_GETREG_B32 stays a loud refusal,
            // because its result is unmodeled hardware state a later address could depend on.
            "SWaitcntVscnt" | "SWaitcntVmcnt" | "SWaitcntExpcnt" | "SWaitcntLgkmcnt"
            | "SSetprio" | "SIcacheInv" | "SSleep" | "SWakeup" | "SSethalt" | "SIncperflevel"
            | "SDecperflevel" | "STtracedataImm" | "SWaitIdle" | "SRoundMode" | "SDenormMode"
            | "SVersion" | "SSetregB32" | "SSetregImm32B32" => {}
            "SBarrier" => {
                if !self.is_threadgroup_stage() {
                    return Err(
                        "workgroup barriers cannot be represented by a Metal graphics stage"
                            .to_owned(),
                    );
                }
                // SharpEmu refuses cross-lane ops under the divergent PC dispatcher (whole-quad
                // death). Structured if/while is the proven reducible path and still needs
                // architectural workgroup barriers for NGG/LDS; only the step-capped Dispatcher
                // is refused here. Graphics Dispatcher ballots already use the single-lane model.
                if matches!(
                    self.control_flow,
                    crate::compiler::control_flow::ControlFlowLowering::Dispatcher
                ) {
                    return Err(
                        "S_BARRIER under step-capped dispatcher control flow is undefined on \
                         Metal simdgroups; refusing rather than emitting a divergent barrier"
                            .to_owned(),
                    );
                }
                self.line(3, "threadgroup_barrier(mem_flags::mem_threadgroup);");
            }
            "Exp" => self.emit_export(instruction)?,
            _ if matches!(
                instruction.control,
                Some(InstructionControl::Interpolation(_))
            ) =>
            {
                self.emit_interpolation(instruction)?;
            }
            _ if matches!(instruction.control, Some(InstructionControl::Image(_))) => {
                self.emit_image(instruction)?;
            }
            _ if matches!(
                instruction.control,
                Some(InstructionControl::ScalarMemory(_))
            ) =>
            {
                self.emit_scalar_memory(instruction)?;
            }
            _ if matches!(
                instruction.control,
                Some(InstructionControl::BufferMemory(_))
            ) =>
            {
                self.emit_buffer_memory(instruction)?;
            }
            _ if matches!(
                instruction.control,
                Some(InstructionControl::GlobalMemory(_))
            ) =>
            {
                self.emit_global_memory(instruction)?;
            }
            _ if matches!(instruction.control, Some(InstructionControl::DataShare(_))) => {
                self.emit_data_share(instruction)?;
            }
            _ if instruction.opcode.starts_with("V") => self.emit_vector_alu(instruction)?,
            _ if instruction.opcode.starts_with("S") => self.emit_scalar_alu(instruction)?,
            _ => return Err("unsupported instruction family".to_owned()),
        }
        if dispatched {
            if let Some(next_pc) = next_pc {
                self.line(3, &format!("pc = {next_pc}u;"));
            } else {
                self.line(3, "active = false;");
            }
        }
        Ok(())
    }

    /// Emits a proven Dispatcher-free [`StructuredCfg`] as nested MSL `if`/`while`.
    ///
    /// Control transfers are structural (no PC machine, no step-cap). Terminal and unconditional
    /// branch instructions inside blocks are skipped — the tree edges already encode them.
    fn emit_structured_cfg(
        &mut self,
        tree: &StructuredCfg,
        blocks: &[BasicBlock],
        indent: usize,
        failures: &mut Vec<LoweringFailure>,
        failed_opcodes: &mut BTreeSet<remu_shader::Opcode>,
    ) -> Result<(), String> {
        match tree {
            StructuredCfg::Block(block_id) => {
                self.emit_structured_block(
                    *block_id,
                    blocks,
                    indent,
                    failures,
                    failed_opcodes,
                    false,
                )?;
            }
            StructuredCfg::Sequence(nodes) => {
                for node in nodes {
                    self.emit_structured_cfg(node, blocks, indent, failures, failed_opcodes)?;
                }
            }
            StructuredCfg::If {
                condition,
                then_is_taken_edge,
                then_branch,
                else_branch,
            } => {
                let taken = self.emit_structured_condition_block(
                    *condition,
                    blocks,
                    indent,
                    failures,
                    failed_opcodes,
                )?;
                let predicate = if *then_is_taken_edge == 1 {
                    taken
                } else {
                    format!("!({taken})")
                };
                self.line(indent, &format!("if ({predicate}) {{"));
                self.emit_structured_cfg(
                    then_branch,
                    blocks,
                    indent + 1,
                    failures,
                    failed_opcodes,
                )?;
                self.line(indent, "}");
                if let Some(else_branch) = else_branch {
                    self.line(indent, "else {");
                    self.emit_structured_cfg(
                        else_branch,
                        blocks,
                        indent + 1,
                        failures,
                        failed_opcodes,
                    )?;
                    self.line(indent, "}");
                }
            }
            StructuredCfg::Loop(body) => {
                self.line(indent, "while (true) {");
                self.emit_structured_cfg(body, blocks, indent + 1, failures, failed_opcodes)?;
                self.line(indent, "}");
            }
            StructuredCfg::Continue => {
                self.line(indent, "continue;");
            }
            StructuredCfg::Break => {
                self.line(indent, "break;");
            }
            StructuredCfg::Dispatcher { .. } => {
                return Err(
                    "StructuredCfg::Dispatcher reached structured MSL emission — refusing to emit step-capped control flow from an unproven tree"
                        .into(),
                );
            }
        }
        Ok(())
    }

    /// Emits the non-branch prefix of a condition block and returns the MSL taken-edge predicate.
    fn emit_structured_condition_block(
        &mut self,
        block_id: usize,
        blocks: &[BasicBlock],
        indent: usize,
        failures: &mut Vec<LoweringFailure>,
        failed_opcodes: &mut BTreeSet<remu_shader::Opcode>,
    ) -> Result<String, String> {
        let block = blocks
            .get(block_id)
            .ok_or_else(|| format!("StructuredCfg condition block {block_id} is missing"))?;
        if block.instructions.is_empty() {
            return Err(format!("StructuredCfg condition block {block_id} is empty"));
        }
        let branch_index = block.instructions.end - 1;
        self.emit_structured_block(block_id, blocks, indent, failures, failed_opcodes, true)?;
        let branch = &self.program.instructions[branch_index];
        let name = branch
            .opcode
            .known_name()
            .ok_or_else(|| format!("condition block {block_id} does not end in a named branch"))?;
        let mut exchanged_mask_predicate = None;
        if self.needs_split_wave64_branch_mask_exchange() {
            let registers = match name {
                "SCbranchVccz" | "SCbranchVccnz" => Some((106_u32, 107_u32)),
                "SCbranchExecz" | "SCbranchExecnz" => Some((126_u32, 127_u32)),
                _ => None,
            };
            if let Some((low, high)) = registers {
                // One guest wave64 is represented by two Metal SIMD32 groups. A scalar mask
                // branch only needs to know whether either architectural word is nonzero; it
                // does not need the remote word copied into every lane's private SGPR state.
                //
                // A whole-threadgroup barrier is illegal here: independent guest waves can be
                // in different structured branches or loop iterations. Rendezvous only the two
                // SIMD32 halves through a double-buffered atomic record. Packing the epoch and
                // the nonzero bit into one atomic avoids depending on ordering between separate
                // data and ready variables (pre-Metal-4.1 atomics only support relaxed order).
                let exchange = self.next_temp;
                self.next_temp += 1;
                let prefix = format!("remu_branch_mask_{exchange}");
                self.line(3, "remu_wave_pair_epoch += 1u;");
                self.line(
                    3,
                    &format!(
                        "uint {prefix}_local = (remu_wave_half == 0u ? s[{low}] : s[{high}]) != 0u ? 1u : 0u;"
                    ),
                );
                self.line(
                    3,
                    &format!(
                        "uint {prefix}_slot = ((remu_wave_pair_epoch & 1u) << 5u) + (remu_wave_index << 1u) + remu_wave_half;"
                    ),
                );
                self.line(
                    3,
                    &format!(
                        "bool {prefix}_has_partner = ((remu_wave_index << 6u) + ((remu_wave_half ^ 1u) << 5u)) < remu_thread_count;"
                    ),
                );
                self.line(3, &format!("uint {prefix}_remote = 0u;"));
                self.line(3, "if (remu_lane == 0u) {");
                self.line(
                    4,
                    &format!(
                        "atomic_store_explicit(&remu_wave_pair_states[{prefix}_slot], (remu_wave_pair_epoch << 1u) | {prefix}_local, memory_order_relaxed);"
                    ),
                );
                self.line(4, &format!("if ({prefix}_has_partner) {{"));
                self.line(5, "do {");
                self.line(
                    6,
                    &format!(
                        "{prefix}_remote = atomic_load_explicit(&remu_wave_pair_states[{prefix}_slot ^ 1u], memory_order_relaxed);"
                    ),
                );
                self.line(
                    5,
                    &format!("}} while (({prefix}_remote >> 1u) != remu_wave_pair_epoch);"),
                );
                self.line(4, "}");
                self.line(3, "}");
                self.line(
                    3,
                    &format!("{prefix}_remote = simd_broadcast_first({prefix}_remote);"),
                );
                self.line(
                    3,
                    &format!(
                        "bool {prefix}_any = {prefix}_local != 0u || ({prefix}_has_partner && ({prefix}_remote & 1u) != 0u);"
                    ),
                );
                exchanged_mask_predicate = Some(match name {
                    "SCbranchVccz" | "SCbranchExecz" => format!("!{prefix}_any"),
                    "SCbranchVccnz" | "SCbranchExecnz" => format!("{prefix}_any"),
                    _ => unreachable!("register selection only accepts scalar mask branches"),
                });
            }
        }
        exchanged_mask_predicate.map_or_else(
            || scalar_branch_taken_condition(name).map(str::to_owned),
            Ok,
        )
    }

    fn emit_structured_block(
        &mut self,
        block_id: usize,
        blocks: &[BasicBlock],
        indent: usize,
        failures: &mut Vec<LoweringFailure>,
        failed_opcodes: &mut BTreeSet<remu_shader::Opcode>,
        skip_terminal_transfer: bool,
    ) -> Result<(), String> {
        let block = blocks
            .get(block_id)
            .ok_or_else(|| format!("StructuredCfg block {block_id} is missing"))?;
        let last = block.instructions.end.saturating_sub(1);
        for index in block.instructions.clone() {
            let instruction = &self.program.instructions[index];
            let is_last = index == last;
            if skip_terminal_transfer && is_last {
                continue;
            }
            if is_last && is_structured_control_transfer(instruction) {
                // Structural edges already encode S_BRANCH / S_ENDPGM / linked S_SETPC.
                continue;
            }
            // Indent structured bodies by rewriting through a temporary indent bump: emit at the
            // default dispatcher indent (3) then is wrong for nested depth. Use `line(indent, …)`
            // via a small adapter — emit_instruction hard-codes indent 3, so temporarily patch by
            // emitting then… actually emit_instruction uses self.line(3, …). For nested structured
            // CF we need variable indent. Approach: emit_instruction always uses indent 3 relative
            // to body start; structured body is inlined like Direct (indent 2 prelude in module).
            // Nested if/while add their own braces at `indent`; instructions inside stay at indent+1
            // only if emit_instruction cooperates.
            //
            // Practical choice matching Direct: keep instruction emission at indent 3 and accept
            // that nested structured braces use indent 2/3/4 while instructions stay at 3. That is
            // valid MSL (whitespace-insensitive) — braces still nest correctly.
            if let Err(message) = self.emit_instruction(instruction, index, false)
                && failed_opcodes.insert(instruction.opcode)
            {
                failures.push(LoweringFailure {
                    pc: instruction.pc,
                    opcode: instruction.opcode,
                    message,
                });
            }
            let _ = indent; // braces carry nesting; instruction lines stay at the Direct indent.
        }
        Ok(())
    }
}

fn is_structured_control_transfer(instruction: &Instruction) -> bool {
    match instruction.opcode.known_name() {
        Some("SEndpgm" | "SSetpcB64" | "SBranch") => true,
        Some(name) if name.starts_with("SCbranch") => true,
        _ => false,
    }
}

fn validate_interface(options: &MetalCompileOptions) -> Result<(), MetalCompileError> {
    if !(1..=VECTOR_REGISTER_COUNT).contains(&options.allocated_vector_register_count) {
        return Err(MetalCompileError::InvalidInterface(format!(
            "allocated vector-register count {} is outside 1..={VECTOR_REGISTER_COUNT}",
            options.allocated_vector_register_count
        )));
    }
    if options.pixel_outputs.len() > 8 {
        return Err(MetalCompileError::InvalidInterface(
            "Metal supports at most eight color outputs".into(),
        ));
    }
    if options.stage == MetalShaderStage::Mesh
        && (!(1..=256).contains(&options.mesh_max_vertex_count)
            || !(1..=256).contains(&options.mesh_max_primitive_count)
            || !(1..=256).contains(&options.mesh_threadgroup_size))
    {
        return Err(MetalCompileError::InvalidInterface(format!(
            "Metal mesh vertex/primitive/thread capacities {}/{}/{} are outside 1..=256",
            options.mesh_max_vertex_count,
            options.mesh_max_primitive_count,
            options.mesh_threadgroup_size
        )));
    }
    let mut guest_slots = BTreeSet::new();
    let mut host_locations = BTreeSet::new();
    for output in &options.pixel_outputs {
        if output.guest_slot > 7
            || !guest_slots.insert(output.guest_slot)
            || !host_locations.insert(output.host_location)
        {
            return Err(MetalCompileError::InvalidInterface(
                "pixel output guest slots and host locations must be unique and in range".into(),
            ));
        }
    }
    if host_locations
        .iter()
        .copied()
        .ne(0..u32::try_from(host_locations.len()).unwrap_or(0))
    {
        return Err(MetalCompileError::InvalidInterface(
            "pixel output host locations must be dense from zero".into(),
        ));
    }
    Ok(())
}
