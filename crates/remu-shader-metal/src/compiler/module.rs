// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

//! Metal module assembly: the emitted translation unit, its resource bindings
//! and the register-reference scan that scalarizes unused register state.

use std::collections::BTreeSet;
use std::fmt::Write as _;

use remu_shader::{InstructionControl, PixelOutputKind};

use crate::compiler::control_flow::ControlFlowLowering;
use crate::compiler::image::metal_texture_declaration;
use crate::compiler::memory::{
    is_format_buffer_memory, is_lds_memory_instruction, is_writable_memory,
};
use crate::compiler::prelude::{
    emit_data_share_prelude, emit_div_fixup_f32_prelude, emit_div_fmas_f32_prelude,
    emit_div_scale_f32_prelude, emit_format_prelude, emit_global_memory_prelude,
    emit_vop3p_prelude,
};
use crate::{
    Compiler, EXEC_HI, EXEC_LO, MetalCompileError, MetalShaderStage, SCALAR_REGISTER_COUNT, VCC_HI,
    VCC_LO, VECTOR_REGISTER_COUNT,
};

impl Compiler<'_> {
    #[allow(clippy::too_many_lines)]
    pub(crate) fn emit_module(&self) -> Result<(String, RegisterReferences), MetalCompileError> {
        let global_count = self.evaluation.global_memory_bindings.len();
        let global_argument_buffer = global_count > 30;
        let initial_index = if global_argument_buffer {
            1
        } else {
            global_count
        };
        let sampler_count = usize::try_from(self.image_slots.sampler_count)
            .map_err(|_| MetalCompileError::InvalidInterface("too many samplers".into()))?;
        let direct_sampler_bindings = sampler_count != 0 && initial_index.saturating_add(1) > 30;
        if direct_sampler_bindings && sampler_count > 16 {
            return Err(MetalCompileError::InvalidInterface(format!(
                "{sampler_count} direct samplers exceed Metal's per-stage limit of 16"
            )));
        }
        let mesh_reserved_buffer_start = if self.options.diagnostic_capture_vertex_position {
            28
        } else {
            29
        };
        if self.options.stage == MetalShaderStage::Mesh
            && !global_argument_buffer
            && (initial_index >= mesh_reserved_buffer_start
                || (!direct_sampler_bindings
                    && sampler_count != 0
                    && initial_index + 1 >= mesh_reserved_buffer_start))
        {
            return Err(MetalCompileError::InvalidInterface(
                if self.options.diagnostic_capture_vertex_position {
                    "diagnostic mesh lowering reserves Metal buffers 28 through 30"
                } else {
                    "mesh lowering reserves Metal buffers 29 and 30 for indices and draw geometry"
                }
                .into(),
            ));
        }
        let mut source = String::from(
            "#include <metal_stdlib>\nusing namespace metal;\n\n\
             inline bool remu_range_valid(uint offset, uint size, uint length) {\n\
             \x20   return offset <= length && size <= length - offset;\n\
             }\n\n\
             inline uint remu_load_u8(const device uchar* data, uint offset, uint length) {\n\
             \x20   return remu_range_valid(offset, 1u, length) ? (uint)data[offset] : 0u;\n\
             }\n\n\
             inline uint remu_load_u16(const device uchar* data, uint offset, uint length) {\n\
             \x20   if (!remu_range_valid(offset, 2u, length)) return 0u;\n\
             \x20   return (uint)data[offset] | ((uint)data[offset + 1u] << 8u);\n\
             }\n\n\
             inline uint remu_load_u32(const device uchar* data, uint offset, uint length) {\n\
             \x20   if (!remu_range_valid(offset, 4u, length)) return 0u;\n\
             \x20   if ((offset & 3u) == 0u) {\n\
             \x20       return *((const device uint*)(data + offset));\n\
             \x20   }\n\
             \x20   uint value = 0u;\n\
             \x20   for (uint i = 0u; i < 4u; ++i) {\n\
             \x20       value |= (uint)data[offset + i] << (i * 8u);\n\
             \x20   }\n\
             \x20   return value;\n}\n\n",
        );
        if self
            .program
            .instructions
            .iter()
            .any(is_format_buffer_memory)
        {
            emit_format_prelude(&mut source);
        }
        if self.program.instructions.iter().any(is_writable_memory) {
            emit_global_memory_prelude(&mut source);
        }
        if self.program.instructions.iter().any(|instruction| {
            matches!(instruction.control, Some(InstructionControl::Vop3p(_)))
                || instruction.opcode.is("VCvtPkrtzF16F32")
        }) {
            emit_vop3p_prelude(&mut source);
        }
        if self
            .program
            .instructions
            .iter()
            .any(|instruction| instruction.opcode.is("VDivFixupF32"))
        {
            emit_div_fixup_f32_prelude(&mut source);
        }
        if self
            .program
            .instructions
            .iter()
            .any(|instruction| instruction.opcode.is("VDivScaleF32"))
        {
            emit_div_scale_f32_prelude(&mut source);
        }
        if self
            .program
            .instructions
            .iter()
            .any(|instruction| instruction.opcode.is("VDivFmasF32"))
        {
            emit_div_fmas_f32_prelude(&mut source);
        }
        let uses_lds_memory = self
            .program
            .instructions
            .iter()
            .any(is_lds_memory_instruction);
        if uses_lds_memory {
            emit_data_share_prelude(&mut source);
        }
        for texture_slot in self.heap_tables.keys().copied() {
            let kind = self.image_kinds[texture_slot];
            let declaration = metal_texture_declaration(kind);
            let _ = writeln!(
                source,
                "struct RemuHeapTable_{texture_slot} {{\n    metal::array<{declaration}, 16> tex [[id(0)]];\n}};\n"
            );
        }
        if sampler_count != 0 && !direct_sampler_bindings {
            source.push_str("struct RemuSamplers {\n");
            for slot in 0..sampler_count {
                let _ = writeln!(source, "    sampler s{slot} [[id({slot})]];");
            }
            source.push_str("};\n\n");
        }
        if global_argument_buffer {
            source.push_str("struct RemuGlobalBuffers {\n");
            for (index, binding) in self.evaluation.global_memory_bindings.iter().enumerate() {
                let qualifier = if binding.writable {
                    "device"
                } else {
                    "const device"
                };
                let _ = writeln!(source, "    {qualifier} uchar* b{index} [[id({index})]];");
            }
            source.push_str("};\n\n");
        }

        match self.options.stage {
            MetalShaderStage::Vertex | MetalShaderStage::Mesh => {
                self.emit_vertex_interface(&mut source);
            }
            MetalShaderStage::Pixel => self.emit_pixel_interface(&mut source),
            MetalShaderStage::Compute => {}
        }

        let mut parameters = Vec::new();
        match self.options.stage {
            MetalShaderStage::Vertex => {
                if !self.evaluation.vertex_inputs.is_empty() {
                    parameters.push("RemuVsIn remu_in [[stage_in]]".to_owned());
                }
            }
            MetalShaderStage::Pixel => parameters.push("RemuPsIn remu_in [[stage_in]]".to_owned()),
            MetalShaderStage::Compute | MetalShaderStage::Mesh => {}
        }
        if global_argument_buffer {
            parameters.push("constant RemuGlobalBuffers& remu_globals [[buffer(0)]]".to_owned());
        } else {
            for (index, binding) in self.evaluation.global_memory_bindings.iter().enumerate() {
                let qualifier = if binding.writable {
                    "device"
                } else {
                    "const device"
                };
                parameters.push(format!("{qualifier} uchar* b{index} [[buffer({index})]]"));
            }
        }
        parameters.push(format!(
            "constant uint* remu_initial_s [[buffer({initial_index})]]"
        ));
        for (index, kind) in self.image_kinds.iter().enumerate() {
            let declaration = metal_texture_declaration(*kind);
            parameters.push(format!("{declaration} tex{index} [[texture({index})]]"));
        }
        {
            let mut ordinal = 0_usize;
            for (texture_slot, route) in &self.heap_tables {
                for heap in &route.heap_binding_indices {
                    let buffer_index = initial_index + 2 + ordinal;
                    if buffer_index > 30 {
                        return Err(MetalCompileError::InvalidInterface(format!(
                            "runtime heap table for texture slot {texture_slot} needs Metal buffer {buffer_index}, beyond the per-stage limit"
                        )));
                    }
                    // Slots after the sampler argument buffer: initial_s, samplers, then tables.
                    parameters.push(format!(
                        "constant RemuHeapTable_{texture_slot}& remu_heap_{texture_slot}_{heap} [[buffer({buffer_index})]]"
                    ));
                    ordinal += 1;
                }
            }
        }
        if direct_sampler_bindings {
            for slot in 0..sampler_count {
                parameters.push(format!("sampler remu_sampler_{slot} [[sampler({slot})]]"));
            }
        } else if sampler_count != 0 {
            parameters.push(format!(
                "constant RemuSamplers& remu_samplers [[buffer({})]]",
                initial_index + 1
            ));
        }
        if self.options.stage == MetalShaderStage::Vertex
            && self.options.diagnostic_capture_vertex_position
        {
            parameters.push("device uint* remu_debug_position [[buffer(30)]]".to_owned());
        }
        if self.options.stage == MetalShaderStage::Mesh
            && self.options.diagnostic_capture_vertex_position
        {
            parameters.push("device uint* remu_debug_position [[buffer(28)]]".to_owned());
        }
        if self.options.stage == MetalShaderStage::Pixel
            && self.options.diagnostic_capture_pixel_output
        {
            parameters.push("device atomic_uint* remu_debug_pixel [[buffer(30)]]".to_owned());
        }
        if self.diagnostic_compute_state_trace_group().is_some() {
            parameters.push("device uint* remu_debug_compute [[buffer(30)]]".to_owned());
        }
        match self.options.stage {
            MetalShaderStage::Vertex => {
                parameters.push("uint remu_vertex_id [[vertex_id]]".to_owned());
                parameters.push("uint remu_instance_id [[instance_id]]".to_owned());
            }
            MetalShaderStage::Pixel => {}
            MetalShaderStage::Compute => {
                if uses_lds_memory {
                    parameters
                        .push("threadgroup atomic_uint* remu_lds [[threadgroup(0)]]".to_owned());
                }
                parameters
                    .push("uint3 remu_local_id [[thread_position_in_threadgroup]]".to_owned());
                parameters.push("uint3 remu_group_id [[threadgroup_position_in_grid]]".to_owned());
                parameters.push("uint remu_lane [[thread_index_in_simdgroup]]".to_owned());
                if self.split_wave64()
                    || self.program.instructions.iter().any(|instruction| {
                        matches!(
                            instruction.opcode.known_name(),
                            Some("DsWriteAddtidB32" | "DsReadAddtidB32")
                        )
                    })
                    || self
                        .evaluation
                        .compute_system_registers
                        .and_then(|registers| registers.thread_group_size)
                        .is_some_and(|register| {
                            contains_register_reference(&self.body, 's', register)
                        })
                {
                    parameters
                        .push("uint remu_thread_index [[thread_index_in_threadgroup]]".to_owned());
                    parameters.push("uint3 remu_group_size [[threads_per_threadgroup]]".to_owned());
                }
            }
            MetalShaderStage::Mesh => {
                let primitive_type = if self.options.render_target_array_length > 1 {
                    "RemuPrimitiveOut"
                } else {
                    "void"
                };
                parameters.push(format!(
                    "mesh<RemuVsOut, {primitive_type}, {}, {}, topology::triangle> remu_mesh",
                    self.options.mesh_max_vertex_count, self.options.mesh_max_primitive_count
                ));
                parameters.push("const device uint* remu_meshlets [[buffer(29)]]".to_owned());
                parameters.push("constant uint* remu_draw [[buffer(30)]]".to_owned());
                parameters
                    .push("uint3 remu_local_id [[thread_position_in_threadgroup]]".to_owned());
                parameters.push("uint3 remu_group_id [[threadgroup_position_in_grid]]".to_owned());
                parameters.push("uint remu_lane [[thread_index_in_simdgroup]]".to_owned());
                parameters
                    .push("uint remu_thread_index [[thread_index_in_threadgroup]]".to_owned());
                parameters.push("uint3 remu_group_size [[threads_per_threadgroup]]".to_owned());
            }
        }
        let signature = match self.options.stage {
            MetalShaderStage::Vertex => "vertex RemuVsOut remu_main",
            MetalShaderStage::Pixel
                if self.options.pixel_outputs.is_empty() && !self.options.pixel_depth_export =>
            {
                "fragment void remu_main"
            }
            MetalShaderStage::Pixel => "fragment RemuPsOut remu_main",
            MetalShaderStage::Compute => "kernel void remu_main",
            MetalShaderStage::Mesh => "[[mesh]] void remu_main",
        };
        let _ = writeln!(
            source,
            "{signature}(\n    {}\n) {{",
            parameters.join(",\n    ")
        );
        if global_argument_buffer {
            for (index, binding) in self.evaluation.global_memory_bindings.iter().enumerate() {
                let qualifier = if binding.writable {
                    "device"
                } else {
                    "const device"
                };
                let _ = writeln!(
                    source,
                    "    {qualifier} uchar* b{index} = remu_globals.b{index};"
                );
            }
        }
        source.push_str("    /* REMU_REGISTER_STATE */\n");
        {
            let mut bases: Vec<u32> = self
                .heap_tables
                .values()
                .map(|route| route.scalar_resource_base)
                .collect();
            bases.sort_unstable();
            bases.dedup();
            for base in bases {
                let _ = writeln!(source, "    uint remu_srd_slot_{base} = 0xffffffffu;");
                let _ = writeln!(source, "    uint remu_srd_heap_{base} = 0xffffffffu;");
            }
        }
        source.push_str("    bool exec = true;\n    bool vcc = false;\n    bool scc = false;\n");
        if self
            .program
            .instructions
            .iter()
            .any(|instruction| instruction.opcode.starts_with("Scratch"))
        {
            let _ = writeln!(
                source,
                "    thread uchar remu_scratch[{}];",
                self.options.scratch_size_bytes
            );
        }
        if self.options.stage == MetalShaderStage::Mesh && uses_lds_memory {
            let _ = writeln!(
                source,
                "    threadgroup atomic_uint remu_lds_storage[{}];\n    threadgroup atomic_uint* remu_lds = remu_lds_storage;",
                self.options.lds_size_bytes / 4
            );
        }
        if self.options.stage == MetalShaderStage::Mesh
            && self.options.render_target_array_length > 1
        {
            let _ = writeln!(
                source,
                "    threadgroup uint remu_mesh_render_target_indices[{}];",
                self.options.mesh_max_vertex_count
            );
            if !self.options.ngg_fixed_allocation_fallback {
                let _ = writeln!(
                    source,
                    "    threadgroup uint remu_mesh_primitive_source_vertices[{}];",
                    self.options.mesh_max_primitive_count
                );
            }
        }
        if self.options.stage == MetalShaderStage::Mesh
            && !self.options.ngg_fixed_allocation_fallback
        {
            source.push_str("    uint remu_mesh_output_primitive_count = 0u;\n");
        }
        if self.split_wave64() {
            source.push_str(
                "    uint remu_wave_index = remu_thread_index >> 6u;\n\
                 \x20   uint remu_wave_half = (remu_thread_index >> 5u) & 1u;\n\
                 \x20   uint remu_wave_lane = remu_lane + (remu_wave_half << 5u);\n",
            );
            if self.needs_split_wave64_pair_rendezvous() {
                // Two parity banks of one atomic epoch/nonzero record per Metal SIMD32 half.
                // The sole whole-group barrier is in this uniform kernel prologue; dynamic
                // scalar branches rendezvous only the two halves of their guest wave64.
                let _ = writeln!(
                    source,
                    "    threadgroup atomic_uint remu_wave_pair_states[{}];",
                    super::SPLIT_WAVE64_PAIR_STATE_SLOTS
                );
                source.push_str(
                    "    uint remu_wave_pair_epoch = 0u;\n\
                     \x20   uint remu_thread_count = remu_group_size.x * remu_group_size.y * remu_group_size.z;\n\
                     \x20   for (uint remu_wave_pair_init = remu_thread_index; remu_wave_pair_init < 64u; remu_wave_pair_init += remu_thread_count) {\n\
                     \x20       atomic_store_explicit(&remu_wave_pair_states[remu_wave_pair_init], 0u, memory_order_relaxed);\n\
                     \x20   }\n\
                     \x20   threadgroup_barrier(mem_flags::mem_threadgroup);\n",
                );
            }
            if self
                .program
                .instructions
                .iter()
                .any(|instruction| instruction.opcode.is("VReadfirstlaneB32"))
            {
                // A Metal threadgroup can contain at most 1,024 threads, or 32 SIMD32 halves.
                // Acyclic programs use disjoint records per static site so they need no reuse
                // barrier. Cyclic programs and memory-heavy acyclic programs retain 32 shared
                // records and the conservative two-barrier protocol.
                let slot_count = self.disjoint_readfirstlane_slot_count().unwrap_or(32);
                let _ = writeln!(
                    source,
                    "    threadgroup uint remu_readfirstlane_values[{slot_count}];"
                );
                let _ = writeln!(
                    source,
                    "    threadgroup uint remu_readfirstlane_active[{slot_count}];"
                );
            }
            if self
                .program
                .instructions
                .iter()
                .any(|instruction| instruction.opcode.is("VReadlaneB32"))
            {
                // V_READLANE may select either half of a guest wave64. Metal's simd_shuffle
                // cannot cross the two SIMD32 groups, so exchange every architectural lane
                // through threadgroup memory. The maximum Metal threadgroup is 1,024 threads.
                source.push_str("    threadgroup uint remu_readlane_values[1024];\n");
            }
        }
        match self.control_flow {
            ControlFlowLowering::Forward => {
                source.push_str("    uint pc = 0u;\n    bool active = true;\n");
            }
            ControlFlowLowering::Dispatcher => {
                source
                    .push_str("    uint pc = 0u;\n    bool active = true;\n    uint steps = 0u;\n");
            }
            ControlFlowLowering::Direct(_) | ControlFlowLowering::Structured { .. } => {}
        }
        if matches!(
            self.options.stage,
            MetalShaderStage::Vertex | MetalShaderStage::Pixel
        ) {
            // Graphics stages cannot declare thread_index_in_simdgroup, but the exclusive prefix
            // of one is the architectural lane index. Wave operations are emitted before the
            // EXEC-guarded write, keeping every host lane converged at simdgroup intrinsics.
            source.push_str("    uint remu_lane = simd_prefix_exclusive_sum(1u);\n");
            let lane_file_width = if self.options.wave_lane_count == 64 {
                64
            } else {
                32
            };
            for vgpr in self.graphics_lane_file_vgprs() {
                // Private wave-VGPR tables for V_WRITELANE/V_READLANE constant packing. Metal
                // fragment quads rarely occupy architectural lanes past the first few, so a
                // simd_shuffle-based table would leave Poppy Slate SDF constants as zeroes.
                let _ = writeln!(
                    source,
                    "    uint remu_v_{vgpr}_lanes[{lane_file_width}] = {{}};"
                );
            }
        }
        let graphics_addtid_slots = self.graphics_addtid_write_slot_count();
        if graphics_addtid_slots != 0 {
            // DS_*_ADDTID_B32 is the architectural lane-private spill form: the only implicit
            // address term is TID*4, and compiler-generated spill regions are separated by the
            // 256-byte wave footprint. Metal graphics stages have no threadgroup address space,
            // so preserve each invocation's dynamically selected M0+offset records locally.
            // One slot per static write plus a sequence number also preserves branches and loops.
            let _ = writeln!(
                source,
                "    uint remu_addtid_address[{graphics_addtid_slots}] = {{}};"
            );
            let _ = writeln!(
                source,
                "    uint remu_addtid_value[{graphics_addtid_slots}] = {{}};"
            );
            let _ = writeln!(
                source,
                "    uint remu_addtid_sequence[{graphics_addtid_slots}] = {{}};"
            );
            let _ = writeln!(
                source,
                "    bool remu_addtid_valid[{graphics_addtid_slots}] = {{}};"
            );
            source.push_str("    uint remu_addtid_clock = 0u;\n");
        }
        match self.options.stage {
            MetalShaderStage::Vertex => {
                source.push_str("    RemuVsOut remu_out = {};\n");
                if self.options.render_target_array_length > 1 {
                    let _ = writeln!(
                        source,
                        "    remu_out.render_target_array_index = min(remu_instance_id, {}u);",
                        self.options.render_target_array_length - 1
                    );
                }
                if contains_register_reference(&self.body, 'v', 5) {
                    source.push_str("    v[5] = remu_vertex_id;\n");
                }
                if contains_register_reference(&self.body, 'v', 8) {
                    source.push_str("    v[8] = remu_instance_id;\n");
                }
            }
            MetalShaderStage::Pixel => {
                if !self.options.pixel_outputs.is_empty() || self.options.pixel_depth_export {
                    source.push_str("    RemuPsOut remu_out = {};\n");
                    // GFX color exports supply architectural defaults for components that the
                    // shader masks out: RGB are zero and alpha is one. The alpha default is
                    // observable even when CB_TARGET_MASK disables the alpha write because the
                    // blend unit still consumes the source alpha. Leaving Metal's aggregate
                    // initialization at zero made SRC_ALPHA passes contribute no source color,
                    // which rendered opaque Unreal materials ghosted or fully transparent.
                    for output in &self.options.pixel_outputs {
                        let default = match output.kind {
                            PixelOutputKind::Float => "float4(0.0f, 0.0f, 0.0f, 1.0f)",
                            PixelOutputKind::Uint => "uint4(0u, 0u, 0u, 1u)",
                            PixelOutputKind::Sint => "int4(0, 0, 0, 1)",
                        };
                        let _ =
                            writeln!(source, "    remu_out.mrt{} = {default};", output.guest_slot);
                    }
                    if self.options.pixel_depth_export {
                        source.push_str("    remu_out.depth = remu_in.frag_coord.z;\n");
                    }
                }
                self.emit_pixel_input_state(&mut source);
            }
            MetalShaderStage::Compute => {
                for (register, component) in [(0, "x"), (1, "y"), (2, "z")] {
                    if contains_register_reference(&self.body, 'v', register) {
                        let _ = writeln!(source, "    v[{register}] = remu_local_id.{component};");
                    }
                }
                self.emit_compute_system_registers(&mut source);
            }
            MetalShaderStage::Mesh => {
                let has_linked_geometry_stage =
                    self.program.instructions.iter().any(|instruction| {
                        self.program
                            .linked_transfer_target(instruction.pc)
                            .is_some()
                    });
                if has_linked_geometry_stage && self.options.ngg_primgen_passthrough {
                    return Err(MetalCompileError::InvalidInterface(
                        "linked ES/GS cannot use the no-GS NGG passthrough input ABI".into(),
                    ));
                }
                if has_linked_geometry_stage && self.options.ngg_esgs_ring_itemsize == 0 {
                    return Err(MetalCompileError::InvalidInterface(
                        "linked ES/GS is missing VGT_ESGS_RING_ITEMSIZE; refusing an invented vertex stride"
                            .into(),
                    ));
                }
                if self.options.ngg_fixed_allocation_fallback {
                    source.push_str(
                            "    RemuVsOut remu_out = {};\n\
                             \x20   remu_out.position = float4(0.0f, 0.0f, 0.0f, 1.0f);\n\
                             \x20   uint remu_meshlet_base = remu_group_id.x * 130u;\n\
                             \x20   uint remu_mesh_counts = remu_meshlets[remu_meshlet_base + 0u];\n\
                             \x20   uint remu_mesh_primitive_base = remu_meshlets[remu_meshlet_base + 1u];\n\
                             \x20   uint remu_mesh_input_vertex_count = remu_mesh_counts & 0xffu;\n\
                             \x20   uint remu_mesh_input_primitive_count = (remu_mesh_counts >> 8u) & 0xffu;\n\
                             \x20   uint remu_mesh_primitive = remu_meshlets[remu_meshlet_base + 66u + remu_thread_index];\n\
                             \x20   remu_mesh.set_index(remu_thread_index * 3u + 0u, 0u);\n\
                             \x20   remu_mesh.set_index(remu_thread_index * 3u + 1u, 0u);\n\
                             \x20   remu_mesh.set_index(remu_thread_index * 3u + 2u, 0u);\n",
                        );
                } else {
                    // Default w=0 so threads that never EXP are hardware-clipped. The previous
                    // (0,0,0,1) default mapped to NDC center and produced a central additive haze
                    // whenever GS_ALLOC reserved more vertices than lanes that successfully exported.
                    source.push_str(
                            "    RemuVsOut remu_out = {};\n\
                             \x20   remu_out.position = float4(0.0f, 0.0f, 0.0f, 0.0f);\n\
                             \x20   uint remu_meshlet_base = remu_group_id.x * 514u;\n\
                             \x20   uint remu_mesh_counts = remu_meshlets[remu_meshlet_base + 0u];\n\
                             \x20   uint remu_mesh_primitive_base = remu_meshlets[remu_meshlet_base + 1u];\n\
                             \x20   uint remu_mesh_input_vertex_count_encoded = remu_mesh_counts & 0xffu;\n\
                             \x20   uint remu_mesh_input_primitive_count_encoded = (remu_mesh_counts >> 8u) & 0xffu;\n\
                             \x20   uint remu_mesh_input_vertex_count = remu_mesh_input_vertex_count_encoded == 0u ? 256u : remu_mesh_input_vertex_count_encoded;\n\
                             \x20   uint remu_mesh_input_primitive_count = remu_mesh_input_primitive_count_encoded == 0u ? 256u : remu_mesh_input_primitive_count_encoded;\n\
                             \x20   uint remu_mesh_primitive = remu_thread_index < remu_mesh_input_primitive_count ? remu_meshlets[remu_meshlet_base + 258u + remu_thread_index] : 0x80000000u;\n",
                        );
                }
                // GS_ALLOC_REQ reserves output slots, but culled or otherwise null primitives do
                // not necessarily execute EXP PRIM. AMD hardware treats those slots as null.
                // Metal has no null-primitive flag and consumes every index below the published
                // primitive count, so leaving an unexported slot unwritten feeds undefined
                // indices to the rasterizer. Initialize every possible slot to a degenerate
                // triangle; a live EXP PRIM below overwrites its own three indices.
                if !self.options.ngg_fixed_allocation_fallback {
                    let _ = writeln!(
                        source,
                        "    for (uint remu_null_primitive = remu_thread_index; remu_null_primitive < {}u; remu_null_primitive += remu_group_size.x * remu_group_size.y * remu_group_size.z) {{\n\
                         \x20       remu_mesh.set_index(remu_null_primitive * 3u + 0u, 0u);\n\
                         \x20       remu_mesh.set_index(remu_null_primitive * 3u + 1u, 0u);\n\
                         \x20       remu_mesh.set_index(remu_null_primitive * 3u + 2u, 0u);\n\
                         \x20   }}",
                        self.options.mesh_max_primitive_count
                    );
                    if self.options.render_target_array_length > 1 {
                        let _ = writeln!(
                            source,
                            "    for (uint remu_null_vertex = remu_thread_index; remu_null_vertex < {}u; remu_null_vertex += remu_group_size.x * remu_group_size.y * remu_group_size.z) {{\n\
                             \x20       remu_mesh_render_target_indices[remu_null_vertex] = 0u;\n\
                             \x20   }}",
                            self.options.mesh_max_vertex_count
                        );
                        let _ = writeln!(
                            source,
                            "    for (uint remu_null_primitive = remu_thread_index; remu_null_primitive < {}u; remu_null_primitive += remu_group_size.x * remu_group_size.y * remu_group_size.z) {{\n\
                             \x20       remu_mesh_primitive_source_vertices[remu_null_primitive] = 0u;\n\
                             \x20   }}",
                            self.options.mesh_max_primitive_count
                        );
                    }
                }
                if !self.options.ngg_fixed_allocation_fallback
                    && !self.options.ngg_primgen_passthrough
                {
                    // Default GFX10 NGG input ABI. Unlike passthrough mode, SPI supplies six
                    // packed 16-bit ES vertex offsets in v0/v1/v4, the primitive ID in v2 and
                    // the GS invocation ID in v3. No-GS NGG lowering consumes the offsets as
                    // local vertex indices; an actual linked GS consumes dword ESGS offsets.
                    // Hardware scales every packed local index by VGT_ESGS_RING_ITEMSIZE.
                    let itemsize = if has_linked_geometry_stage {
                        self.options.ngg_esgs_ring_itemsize
                    } else {
                        1
                    };
                    let _ = writeln!(
                        source,
                        "    uint remu_mesh_gs_vtx_offset_0 = ((((remu_mesh_primitive >> 0u) & 0x1ffu) * {itemsize}u) & 0xffffu)\n\
                         \x20       | (((((remu_mesh_primitive >> 10u) & 0x1ffu) * {itemsize}u) & 0xffffu) << 16u);\n\
                         \x20   uint remu_mesh_gs_vtx_offset_1 = (((remu_mesh_primitive >> 20u) & 0x1ffu) * {itemsize}u) & 0xffffu;",
                    );
                }
                if self.options.render_target_array_length > 1 {
                    let _ = writeln!(
                        source,
                        "    remu_mesh_render_target_indices[remu_thread_index] = min(remu_group_id.y, {}u);",
                        self.options.render_target_array_length - 1
                    );
                }
                // Seed s[3] for the register-sized mesh ABI: GFX10 merged NGG shaders read
                // - lane ID as `(s3[27:24] << log2(wave)) + v_mbcnt`
                // - wave count as `s3[31:28]`, then `(1 << count) - 1` as the reduce mask
                //   that loads per-wave totals from LDS (Void Terrarium soft/main particles).
                // Bytes 0/1 are the ES/GS invocation counts for the *current guest wave*, not
                // workgroup-wide totals. Repeating a small meshlet's totals in every wave makes
                // the second wave run the first few ES/GS lanes again with out-of-range inputs;
                // Silent Hill's two-wave LUT quad then polluted the NGG primitive flags and
                // allocated 10 vertices / 32 primitives instead of 6 / 2. Byte 2 is the GS wave
                // id supplied by hardware; preserve the meshlet-provided value for now. Clear
                // [31:24] before packing so stale meshlet high bits cannot override the wave
                // fields.
                // The wave fields must use the guest's architectural wave size: wave64 NGG
                // exports (Poppy Playtime UI/LUT quads) recover their meshlet thread index as
                // `s3[27:24] * 64 + v_mbcnt`, so packing SIMD32 halves there sent every
                // thread past the first 32 out of range and left their vertices unexported.
                let (wave_round, wave_shift) = if self.split_wave64() {
                    (63u32, 6u32)
                } else {
                    (31u32, 5u32)
                };
                // The original fixed wave64 ABI predates these synthesized merged-stage SGPRs.
                // Its shaders consume the incoming guest s2 unchanged and seed s3 directly from
                // the 130-word meshlet below. Applying the register-sized prologue as well
                // corrupts live guest scalar data (notably the Superliminal UI vertex selector).
                if !self.options.ngg_fixed_allocation_fallback {
                    let _ = write!(
                        source,
                        "    {{\n\
                     \x20       uint remu_mesh_wave_index = remu_thread_index >> {wave_shift}u;\n\
                     \x20       uint remu_mesh_wave_base = remu_mesh_wave_index << {wave_shift}u;\n\
                     \x20       uint remu_mesh_wave_vertex_count = min(remu_mesh_input_vertex_count > remu_mesh_wave_base ? remu_mesh_input_vertex_count - remu_mesh_wave_base : 0u, {wave_round}u + 1u);\n\
                     \x20       uint remu_mesh_wave_primitive_count = min(remu_mesh_input_primitive_count > remu_mesh_wave_base ? remu_mesh_input_primitive_count - remu_mesh_wave_base : 0u, {wave_round}u + 1u);\n\
                     \x20       uint remu_mesh_wave_count = ((remu_group_size.x * remu_group_size.y * remu_group_size.z) + {wave_round}u) >> {wave_shift}u;\n\
                     \x20       s[3] = (remu_mesh_counts & 0x00ff0000u)\n\
                     \x20           | (remu_mesh_wave_vertex_count & 0xffu)\n\
                     \x20           | ((remu_mesh_wave_primitive_count & 0xffu) << 8u)\n\
                     \x20           | ((remu_mesh_wave_index & 0xfu) << 24u)\n\
                     \x20           | ((remu_mesh_wave_count & 0xfu) << 28u);\n\
                     \x20   }}\n",
                    );
                }
                if !self.options.ngg_fixed_allocation_fallback
                    && contains_register_reference(&self.body, 's', 2)
                {
                    // Merged NGG SGPR s2 is gs_tg_info: [11:0] ordered_id (unused here),
                    // [20:12] workgroup input vertex count, [30:22] workgroup input primitive
                    // count (Mesa ac_nir_lower_intrinsics_to_args.c). Non-culling NGG shaders
                    // build the GS_ALLOC_REQ M0 payload directly from these fields; leaving s2
                    // unseeded publishes a zero primitive count and erases the whole draw.
                    source.push_str(
                        "    s[2] = ((remu_mesh_input_vertex_count & 0x1ffu) << 12u)\n\
                         \x20       | ((remu_mesh_input_primitive_count & 0x1ffu) << 22u);\n",
                    );
                }
                if !self.options.ngg_fixed_allocation_fallback
                    && contains_register_reference(&self.body, 'v', 0)
                {
                    if !self.options.ngg_primgen_passthrough {
                        source.push_str("    v[0] = remu_mesh_gs_vtx_offset_0;\n");
                    } else {
                        source.push_str("    v[0] = remu_mesh_primitive;\n");
                    }
                }
                if !self.options.ngg_fixed_allocation_fallback
                    && contains_register_reference(&self.body, 'v', 1)
                {
                    if !self.options.ngg_primgen_passthrough {
                        source.push_str("    v[1] = remu_mesh_gs_vtx_offset_1;\n");
                    } else {
                        // GFX10-11 passthrough packs all vertex indices and edge flags in v0.
                        // v1 is not the primitive-ID input; keep an explicitly referenced but
                        // architecturally unused component deterministic.
                        source.push_str("    v[1] = 0u;\n");
                    }
                }
                if !self.options.ngg_fixed_allocation_fallback
                    && contains_register_reference(&self.body, 'v', 2)
                {
                    // The fixed GFX9-11 merged-GS ABI keeps PrimitiveID in v2 in both ordinary
                    // and NGG passthrough mode. Only the vertex-offset payload moves into v0.
                    source.push_str("    v[2] = remu_mesh_primitive_base + remu_thread_index;\n");
                }
                if !self.options.ngg_fixed_allocation_fallback
                    && contains_register_reference(&self.body, 'v', 3)
                {
                    source.push_str("    v[3] = 0u;\n");
                }
                if !self.options.ngg_fixed_allocation_fallback
                    && contains_register_reference(&self.body, 'v', 4)
                {
                    source.push_str("    v[4] = 0u;\n");
                }
                if !self.options.ngg_fixed_allocation_fallback
                    && contains_register_reference(&self.body, 'v', 5)
                {
                    source.push_str(
                        "    v[5] = remu_thread_index < remu_mesh_input_vertex_count ? remu_meshlets[remu_meshlet_base + 2u + remu_thread_index] + remu_draw[2] : 0u;\n",
                    );
                }
                if !self.options.ngg_fixed_allocation_fallback
                    && contains_register_reference(&self.body, 'v', 8)
                {
                    source.push_str("    v[8] = remu_group_id.y + remu_draw[3];\n");
                }
                // Preserve the legacy no-GS input-register contract for the fixed wave64
                // fallback. Current linked-GS draws use the decoded merged-GS layout above,
                // but ordinary no-GS shaders from the pre-subgroup-allocation path consume
                // the packed primitive and primitive ID from v0/v1 and expect v2 cleared.
                if self.options.ngg_fixed_allocation_fallback {
                    if contains_register_reference(&self.body, 's', 3) {
                        source.push_str("    s[3] = remu_mesh_counts;\n");
                    }
                    if contains_register_reference(&self.body, 'v', 0) {
                        source.push_str("    v[0] = remu_mesh_primitive;\n");
                    }
                    if contains_register_reference(&self.body, 'v', 1) {
                        source
                            .push_str("    v[1] = remu_mesh_primitive_base + remu_thread_index;\n");
                    }
                    if contains_register_reference(&self.body, 'v', 2) {
                        source.push_str("    v[2] = 0u;\n");
                    }
                    if contains_register_reference(&self.body, 'v', 5) {
                        source.push_str(
                            "    v[5] = remu_thread_index < remu_mesh_input_vertex_count ? remu_meshlets[remu_meshlet_base + 2u + remu_thread_index] + remu_draw[2] : 0u;\n",
                        );
                    }
                    if contains_register_reference(&self.body, 'v', 8) {
                        source.push_str("    v[8] = remu_group_id.y;\n");
                    }
                }
            }
        }
        for register in [VCC_LO, VCC_HI] {
            if contains_register_reference(&self.body, 's', register) {
                let _ = writeln!(source, "    s[{register}] = 0u;");
            }
        }
        if contains_register_reference(&self.body, 's', EXEC_LO)
            || contains_register_reference(&self.body, 's', EXEC_HI)
        {
            // Hardware launches a merged NGG wave with the full architectural EXEC mask and
            // the shader narrows it against merged_wave_info (Mesa aco lanecount_to_mask). A
            // ballot over the Metal threads that happen to exist would zero the tail of a
            // partial final wave and skew lane counts taken before the guest's own narrowing.
            let (low, high) = if self.options.stage == MetalShaderStage::Mesh
                && !self.options.ngg_fixed_allocation_fallback
            {
                let high = if self.split_wave64() {
                    "0xffffffffu"
                } else {
                    "0u"
                };
                ("0xffffffffu".to_owned(), high.to_owned())
            } else {
                self.ballot_words("true")
            };
            if contains_register_reference(&self.body, 's', EXEC_LO) {
                let _ = writeln!(source, "    s[{EXEC_LO}] = {low};");
            }
            if contains_register_reference(&self.body, 's', EXEC_HI) {
                let _ = writeln!(source, "    s[{EXEC_HI}] = {high};");
            }
        }
        match self.control_flow {
            ControlFlowLowering::Direct(_) | ControlFlowLowering::Structured { .. } => {
                source.push_str(&self.body)
            }
            ControlFlowLowering::Forward => source.push_str(&self.body),
            ControlFlowLowering::Dispatcher => {
                let step_limit = self.dispatcher_step_limit();
                source.push_str("    while (active) {\n        switch (pc) {\n");
                source.push_str(&self.body);
                let _ = writeln!(
                    source,
                    "        default: active = false; break;\n        }}\n\
                     \x20       if (steps >= {step_limit}u) active = false;\n    }}"
                );
            }
        }
        match self.options.stage {
            MetalShaderStage::Vertex | MetalShaderStage::Pixel => {
                if self.options.stage == MetalShaderStage::Vertex {
                    append_vertex_position_format_conversion(
                        &mut source,
                        "remu_out.position",
                        self.options.viewport_transform_control,
                        None,
                    );
                }
                if self.options.stage == MetalShaderStage::Pixel {
                    // EXEC is the guest fragment-live mask. Apply the architectural kill before
                    // the diagnostic write so a fragment that executes shader arithmetic but is
                    // discarded at the export boundary cannot be reported as a committed MRT
                    // pixel. Forced-color diagnostics intentionally bypass the guest kill.
                    if !self.options.diagnostic_force_pixel_color {
                        source.push_str("    if (!exec) discard_fragment();\n");
                    }
                    if self.options.diagnostic_capture_pixel_output
                        && let Some(output) = self.options.pixel_outputs.first()
                    {
                        let diagnostic_enable_index = self
                            .evaluation
                            .initial_scalar_registers
                            .len()
                            .saturating_add(
                                self.evaluation
                                    .global_memory_bindings
                                    .len()
                                    .saturating_mul(4),
                            );
                        let field = format!("remu_out.mrt{}", output.guest_slot);
                        let diagnostic_texture_sample = self
                            .image_kinds
                            .first()
                            .filter(|kind| {
                                kind.component == "float"
                                    && !kind.storage
                                    && !kind.comparison
                                    && crate::compiler::image::metal_image_dimension(**kind) == 1
                                    && sampler_count != 0
                            })
                            .map(|_| {
                                let sampler = if direct_sampler_bindings {
                                    "remu_sampler_0"
                                } else {
                                    "remu_samplers.s0"
                                };
                                format!(
                                    "float4 remu_debug_texture = tex0.sample({sampler}, float2(remu_in.attr1[0], remu_in.attr1[1]));\n"
                                )
                            });
                        let _ = writeln!(
                            source,
                            "    uint2 remu_debug_pixel_xy = uint2(remu_in.frag_coord.xy);\n\
                             \x20   if (remu_initial_s[{diagnostic_enable_index}] != 0u && remu_debug_pixel_xy.x < 1920u && remu_debug_pixel_xy.y < 1080u) {{\n\
                             \x20       uint remu_debug_pixel_base = (remu_debug_pixel_xy.y * 1920u + remu_debug_pixel_xy.x) * 11u;\n\
                             \x20       atomic_store_explicit(&remu_debug_pixel[remu_debug_pixel_base + 1u], as_type<uint>(remu_in.attr1[0]), memory_order_relaxed);\n\
                             \x20       atomic_store_explicit(&remu_debug_pixel[remu_debug_pixel_base + 2u], as_type<uint>(remu_in.attr1[1]), memory_order_relaxed);{}",
                            diagnostic_texture_sample.as_deref().unwrap_or_default()
                        );
                        let diagnostic_values = self
                            .options
                            .pixel_outputs
                            .iter()
                            .find(|output| output.guest_slot == 4)
                            .map(|output| {
                                let field = format!("remu_out.mrt{}", output.guest_slot);
                                [
                                    "as_type<uint>(remu_in.attr1[0])".to_owned(),
                                    format!("as_type<uint>({field}[0])"),
                                    format!("as_type<uint>({field}[1])"),
                                    format!("as_type<uint>({field}[2])"),
                                ]
                            });
                        for component in 0..4 {
                            let value = diagnostic_values.as_ref().map_or_else(
                                || match output.kind {
                                    PixelOutputKind::Float => {
                                        format!("as_type<uint>({field}[{component}])")
                                    }
                                    PixelOutputKind::Uint => format!("{field}[{component}]"),
                                    PixelOutputKind::Sint => {
                                        format!("as_type<uint>({field}[{component}])")
                                    }
                                },
                                |values| values[component].clone(),
                            );
                            let _ = writeln!(
                                source,
                                "        atomic_store_explicit(&remu_debug_pixel[remu_debug_pixel_base + {}u], {value}, memory_order_relaxed);",
                                component + 3
                            );
                        }
                        if diagnostic_texture_sample.is_some() {
                            for component in 0..4 {
                                let _ = writeln!(
                                    source,
                                    "        atomic_store_explicit(&remu_debug_pixel[remu_debug_pixel_base + {}u], as_type<uint>(remu_debug_texture[{component}]), memory_order_relaxed);",
                                    component + 7
                                );
                            }
                        }
                        source.push_str(
                            "        atomic_store_explicit(&remu_debug_pixel[remu_debug_pixel_base], 1u, memory_order_relaxed);\n\
                             \x20   }\n",
                        );
                    }
                    if self.options.diagnostic_force_pixel_color {
                        for output in &self.options.pixel_outputs {
                            let value = match output.kind {
                                PixelOutputKind::Float => "float4(1.0f, 0.0f, 1.0f, 1.0f)",
                                PixelOutputKind::Uint => {
                                    "uint4(0xffffffffu, 0u, 0xffffffffu, 0xffffffffu)"
                                }
                                PixelOutputKind::Sint => {
                                    "int4(0x7fffffff, 0, 0x7fffffff, 0x7fffffff)"
                                }
                            };
                            let _ = writeln!(
                                source,
                                "    remu_out.mrt{} = {value};",
                                output.guest_slot
                            );
                        }
                    }
                } else if self.options.diagnostic_force_vertex_pattern {
                    source.push_str(
                        "    uint remu_diagnostic_vertex = remu_vertex_id % 3u;\n\
                         \x20   float2 remu_diagnostic_xy = remu_diagnostic_vertex == 0u ? float2(-1.0f, -1.0f) : (remu_diagnostic_vertex == 1u ? float2(3.0f, -1.0f) : float2(-1.0f, 3.0f));\n\
                         \x20   remu_out.position = float4(remu_diagnostic_xy, 0.5f, 1.0f);\n",
                    );
                } else if self.options.diagnostic_force_visible_clip_depth {
                    // Keep the guest X/Y projection intact while forcing a finite positive W and
                    // an in-range Metal clip-space Z. This is a diagnostic for separating broken
                    // vertex/resource evaluation from Z/W clipping; it is never enabled by
                    // default and deliberately does not alter ordinary translated shaders.
                    source.push_str(
                        "    float remu_diagnostic_w = max(fabs(remu_out.position.w), 1.0e-6f);\n\
                         \x20   remu_out.position.w = remu_diagnostic_w;\n\
                         \x20   remu_out.position.z = 0.5f * remu_diagnostic_w;\n",
                    );
                }
                if self.options.diagnostic_capture_vertex_position {
                    let diagnostic_enable_index = self
                        .evaluation
                        .initial_scalar_registers
                        .len()
                        .saturating_add(
                            self.evaluation
                                .global_memory_bindings
                                .len()
                                .saturating_mul(4),
                        );
                    let _ = writeln!(
                        source,
                        "    if (remu_initial_s[{diagnostic_enable_index}] != 0u && remu_vertex_id < 262144u) {{\n\
                         \x20       uint remu_debug_base = remu_vertex_id * 36u;\n\
                         \x20       remu_debug_position[remu_debug_base + 0u] = as_type<uint>(remu_out.position.x);\n\
                         \x20       remu_debug_position[remu_debug_base + 1u] = as_type<uint>(remu_out.position.y);\n\
                         \x20       remu_debug_position[remu_debug_base + 2u] = as_type<uint>(remu_out.position.z);\n\
                         \x20       remu_debug_position[remu_debug_base + 3u] = as_type<uint>(remu_out.position.w);\n\
                         \x20       remu_debug_position[remu_debug_base + 4u] = v[0];\n\
                         \x20       remu_debug_position[remu_debug_base + 5u] = v[1];\n\
                         \x20       remu_debug_position[remu_debug_base + 6u] = v[2];\n\
                         \x20       remu_debug_position[remu_debug_base + 7u] = v[3];\n\
                         \x20       remu_debug_position[remu_debug_base + 8u] = v[4];\n\
                         \x20       remu_debug_position[remu_debug_base + 9u] = v[5];\n\
                         \x20       remu_debug_position[remu_debug_base + 10u] = v[6];\n\
                         \x20       remu_debug_position[remu_debug_base + 11u] = v[7];\n\
                         \x20       remu_debug_position[remu_debug_base + 12u] = v[8];\n\
                         \x20       remu_debug_position[remu_debug_base + 13u] = v[9];\n\
                         \x20       remu_debug_position[remu_debug_base + 14u] = v[10];\n\
                         \x20       remu_debug_position[remu_debug_base + 15u] = v[11];\n\
                         \x20       remu_debug_position[remu_debug_base + 16u] = v[12];\n\
                         \x20       remu_debug_position[remu_debug_base + 17u] = v[13];\n\
                         \x20       remu_debug_position[remu_debug_base + 18u] = v[14];\n\
                         \x20       remu_debug_position[remu_debug_base + 19u] = v[15];\n\
                         \x20       remu_debug_position[remu_debug_base + 20u] = v[16];\n\
                         \x20       remu_debug_position[remu_debug_base + 21u] = v[17];\n\
                         \x20       remu_debug_position[remu_debug_base + 22u] = v[18];\n\
                         \x20       remu_debug_position[remu_debug_base + 23u] = v[19];\n\
                         \x20   }}"
                    );
                }
                if self.options.stage == MetalShaderStage::Vertex
                    || !self.options.pixel_outputs.is_empty()
                    || self.options.pixel_depth_export
                {
                    source.push_str("    return remu_out;\n");
                }
            }
            MetalShaderStage::Compute => {}
            MetalShaderStage::Mesh => {
                if self.options.ngg_fixed_allocation_fallback {
                    if self.options.render_target_array_length > 1 {
                        source.push_str(
                            "    threadgroup_barrier(mem_flags::mem_threadgroup);\n\
                             \x20   if (remu_thread_index < remu_mesh_input_primitive_count) {\n\
                             \x20       RemuPrimitiveOut remu_primitive_out = {};\n\
                             \x20       uint remu_primitive_source_vertex = remu_mesh_primitive & 0x1ffu;\n\
                             \x20       remu_primitive_out.render_target_array_index = remu_mesh_render_target_indices[remu_primitive_source_vertex];\n\
                             \x20       remu_mesh.set_primitive(remu_thread_index, remu_primitive_out);\n\
                             \x20   }\n",
                        );
                    }
                    if self.options.diagnostic_capture_vertex_position {
                        let diagnostic_enable_index = self
                            .evaluation
                            .initial_scalar_registers
                            .len()
                            .saturating_add(
                                self.evaluation
                                    .global_memory_bindings
                                    .len()
                                    .saturating_mul(4),
                            );
                        let _ = writeln!(
                            source,
                            "    if (remu_initial_s[{diagnostic_enable_index}] != 0u && remu_thread_index < 64u) {{\n\
                             \x20       uint remu_debug_base = remu_thread_index * 36u;\n\
                             \x20       remu_debug_position[remu_debug_base + 0u] = as_type<uint>(remu_out.position.x);\n\
                             \x20       remu_debug_position[remu_debug_base + 1u] = as_type<uint>(remu_out.position.y);\n\
                             \x20       remu_debug_position[remu_debug_base + 2u] = as_type<uint>(remu_out.position.z);\n\
                             \x20       remu_debug_position[remu_debug_base + 3u] = as_type<uint>(remu_out.position.w);"
                        );
                        for register in 0..32 {
                            let _ = writeln!(
                                source,
                                "        remu_debug_position[remu_debug_base + {}u] = v[{register}];",
                                register + 4
                            );
                        }
                        source.push_str("    }\n");
                    }
                    source.push_str(
                        "    if (remu_thread_index < remu_mesh_input_vertex_count) remu_mesh.set_vertex(remu_thread_index, remu_out);\n\
                         \x20   if (remu_thread_index == 0u) remu_mesh.set_primitive_count(remu_mesh_input_primitive_count);\n",
                    );
                    source.push_str("}\n");
                    return scalarize_register_state(&source);
                }
                let _ = writeln!(
                    source,
                    "    uint remu_mesh_output_primitives = min(remu_mesh_output_primitive_count, {}u);",
                    self.options.mesh_max_primitive_count
                );
                if self.options.render_target_array_length > 1 {
                    let _ = writeln!(
                        source,
                        "    threadgroup_barrier(mem_flags::mem_threadgroup);\n\
                         \x20   if (remu_thread_index < {}u) {{\n\
                         \x20       RemuPrimitiveOut remu_primitive_out = {{}};\n\
                         \x20       uint remu_primitive_source_vertex = remu_mesh_primitive_source_vertices[remu_thread_index];\n\
                         \x20       remu_primitive_out.render_target_array_index = remu_mesh_render_target_indices[remu_primitive_source_vertex];\n\
                         \x20       remu_mesh.set_primitive(remu_thread_index, remu_primitive_out);\n\
                         \x20   }}",
                        self.options.mesh_max_primitive_count
                    );
                }
                // Compacted outputs publish directly at EXP because PRIM records name that
                // domain. Mesh programs without a compacted gate retain one vertex per lane and
                // publish here after position conversion.
                if self.mesh_export_vertex_vgpr.is_none() {
                    append_vertex_position_format_conversion(
                        &mut source,
                        "remu_out.position",
                        self.options.viewport_transform_control,
                        None,
                    );
                }
                if self.options.diagnostic_capture_vertex_position {
                    let diagnostic_enable_index = self
                        .evaluation
                        .initial_scalar_registers
                        .len()
                        .saturating_add(
                            self.evaluation
                                .global_memory_bindings
                                .len()
                                .saturating_mul(4),
                        );
                    let position = "remu_out.position";
                    let _ = writeln!(
                        source,
                        "    if (remu_initial_s[{diagnostic_enable_index}] != 0u && remu_thread_index < 64u) {{\n\
                         \x20       uint remu_debug_base = remu_thread_index * 36u;\n\
                         \x20       remu_debug_position[remu_debug_base + 0u] = as_type<uint>({position}.x);\n\
                         \x20       remu_debug_position[remu_debug_base + 1u] = as_type<uint>({position}.y);\n\
                         \x20       remu_debug_position[remu_debug_base + 2u] = as_type<uint>({position}.z);\n\
                         \x20       remu_debug_position[remu_debug_base + 3u] = as_type<uint>({position}.w);\n\
                         \x20       remu_debug_position[remu_debug_base + 4u] = v[0];\n\
                         \x20       remu_debug_position[remu_debug_base + 5u] = v[1];\n\
                         \x20       remu_debug_position[remu_debug_base + 6u] = v[2];\n\
                         \x20       remu_debug_position[remu_debug_base + 7u] = v[3];\n\
                         \x20       remu_debug_position[remu_debug_base + 8u] = v[4];\n\
                         \x20       remu_debug_position[remu_debug_base + 9u] = v[5];\n\
                         \x20       remu_debug_position[remu_debug_base + 10u] = v[6];\n\
                         \x20       remu_debug_position[remu_debug_base + 11u] = v[7];\n\
                         \x20       remu_debug_position[remu_debug_base + 12u] = v[8];\n\
                         \x20       remu_debug_position[remu_debug_base + 13u] = v[9];\n\
                         \x20       remu_debug_position[remu_debug_base + 14u] = v[10];\n\
                         \x20       remu_debug_position[remu_debug_base + 15u] = v[11];\n\
                         \x20       remu_debug_position[remu_debug_base + 16u] = v[12];\n\
                         \x20       remu_debug_position[remu_debug_base + 17u] = v[13];\n\
                         \x20       remu_debug_position[remu_debug_base + 18u] = v[14];\n\
                         \x20       remu_debug_position[remu_debug_base + 19u] = v[15];\n\
                         \x20       remu_debug_position[remu_debug_base + 20u] = s[2];\n\
                         \x20       remu_debug_position[remu_debug_base + 21u] = s[4];\n\
                         \x20       remu_debug_position[remu_debug_base + 22u] = s[107];\n\
                         \x20       remu_debug_position[remu_debug_base + 23u] = s[124];\n\
                         \x20   }}"
                    );
                }
                if self.mesh_export_vertex_vgpr.is_none() {
                    if self.options.ngg_fixed_allocation_fallback {
                        source.push_str(
                            "    if (remu_thread_index < remu_mesh_input_vertex_count) remu_mesh.set_vertex(remu_thread_index, remu_out);\n",
                        );
                    } else {
                        let _ = writeln!(
                            source,
                            "    if (remu_thread_index < {}u) remu_mesh.set_vertex(remu_thread_index, remu_out);",
                            self.options.mesh_max_vertex_count
                        );
                    }
                }
                source.push_str(
                    "    if (remu_thread_index == 0u) {\n\
                     \x20       remu_mesh.set_primitive_count(remu_mesh_output_primitives);\n\
                     \x20   }\n",
                );
            }
        }
        source.push_str("}\n");
        scalarize_register_state(&source)
    }
}

/// Converts the guest VTE position representation into the homogeneous clip coordinates Metal
/// expects. AMD's setup engine can accept either W0 or 1/W0 and optionally pre-divided X/Y/Z;
/// Metal always accepts W0 and performs its own perspective divide.
fn append_vertex_position_format_conversion(
    source: &mut String,
    position: &str,
    viewport_transform_control: u32,
    guard: Option<&str>,
) {
    const VTX_XY_FMT: u32 = 1 << 8;
    const VTX_Z_FMT: u32 = 1 << 9;
    const VTX_W0_FMT: u32 = 1 << 10;

    let xy_predivided = viewport_transform_control & VTX_XY_FMT != 0;
    let z_predivided = viewport_transform_control & VTX_Z_FMT != 0;
    let w_is_w0 = viewport_transform_control & VTX_W0_FMT != 0;
    if !xy_predivided && !z_predivided && w_is_w0 {
        return;
    }

    if let Some(guard) = guard {
        let _ = writeln!(source, "    if ({guard}) {{");
    }
    let indent = if guard.is_some() { "        " } else { "    " };
    let w0 = if w_is_w0 {
        format!("{position}.w")
    } else {
        format!("1.0f / {position}.w")
    };
    let _ = writeln!(source, "{indent}float remu_position_w0 = {w0};");
    if xy_predivided {
        let _ = writeln!(source, "{indent}{position}.xy *= remu_position_w0;");
    }
    if z_predivided {
        let _ = writeln!(source, "{indent}{position}.z *= remu_position_w0;");
    }
    let _ = writeln!(source, "{indent}{position}.w = remu_position_w0;");
    if guard.is_some() {
        source.push_str("    }\n");
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub(crate) struct RegisterReferences {
    pub(crate) scalar: BTreeSet<u32>,
    pub(crate) vector: BTreeSet<u32>,
}

pub(crate) fn contains_register_reference(source: &str, kind: char, register: u32) -> bool {
    register_references(source).is_ok_and(|references| match kind {
        's' => references.scalar.contains(&register),
        'v' => references.vector.contains(&register),
        _ => false,
    })
}

fn scalarize_register_state(
    source: &str,
) -> Result<(String, RegisterReferences), MetalCompileError> {
    const MARKER: &str = "    /* REMU_REGISTER_STATE */\n";
    let references = register_references(source)?;
    let mut rewritten = String::with_capacity(source.len());
    let bytes = source.as_bytes();
    let mut cursor = 0;
    while cursor < bytes.len() {
        if let Some((kind, register, end)) = parse_register_reference(bytes, cursor)? {
            let _ = write!(rewritten, "remu_{kind}_{register}");
            cursor = end;
        } else {
            rewritten.push(char::from(bytes[cursor]));
            cursor += 1;
        }
    }

    let mut declarations = String::new();
    for register in &references.scalar {
        let _ = writeln!(
            declarations,
            "    uint remu_s_{register} = remu_initial_s[{register}];"
        );
    }
    for register in &references.vector {
        let _ = writeln!(declarations, "    uint remu_v_{register} = 0u;");
    }
    if !rewritten.contains(MARKER) {
        return Err(MetalCompileError::InvalidInterface(
            "internal register-state marker was lost".into(),
        ));
    }
    Ok((rewritten.replacen(MARKER, &declarations, 1), references))
}

fn register_references(source: &str) -> Result<RegisterReferences, MetalCompileError> {
    let bytes = source.as_bytes();
    let mut references = RegisterReferences::default();
    for cursor in 0..bytes.len() {
        let Some((kind, register, _)) = parse_register_reference(bytes, cursor)? else {
            continue;
        };
        match kind {
            's' => {
                references.scalar.insert(register);
            }
            'v' => {
                references.vector.insert(register);
            }
            _ => unreachable!("register parser accepts only scalar and vector files"),
        }
    }
    Ok(references)
}

fn parse_register_reference(
    bytes: &[u8],
    start: usize,
) -> Result<Option<(char, u32, usize)>, MetalCompileError> {
    let Some(&kind @ (b's' | b'v')) = bytes.get(start) else {
        return Ok(None);
    };
    if start != 0
        && bytes
            .get(start - 1)
            .is_some_and(|byte| byte.is_ascii_alphanumeric() || *byte == b'_')
    {
        return Ok(None);
    }
    if bytes.get(start + 1) != Some(&b'[') {
        return Ok(None);
    }
    let mut cursor = start + 2;
    let digits_start = cursor;
    while bytes.get(cursor).is_some_and(u8::is_ascii_digit) {
        cursor += 1;
    }
    if cursor == digits_start || bytes.get(cursor) != Some(&b']') {
        return Err(MetalCompileError::InvalidInterface(format!(
            "dynamic {} register indexing survived lowering",
            char::from(kind)
        )));
    }
    let register = std::str::from_utf8(&bytes[digits_start..cursor])
        .ok()
        .and_then(|digits| digits.parse::<u32>().ok())
        .ok_or_else(|| {
            MetalCompileError::InvalidInterface("invalid generated register index".into())
        })?;
    let limit = if kind == b's' {
        SCALAR_REGISTER_COUNT
    } else {
        VECTOR_REGISTER_COUNT
    };
    if register >= limit {
        return Err(MetalCompileError::InvalidInterface(format!(
            "generated {} register {} exceeds architectural limit {}",
            char::from(kind),
            register,
            limit
        )));
    }
    Ok(Some((char::from(kind), register, cursor + 1)))
}
