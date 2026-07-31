// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

//! Metal module assembly: the emitted translation unit, its resource bindings
//! and the register-reference scan that scalarizes unused register state.

use std::collections::BTreeSet;
use std::fmt::Write as _;

use remu_shader::{InstructionControl, PixelOutputKind};

use crate::compiler::control_flow::ControlFlowLowering;
use crate::compiler::memory::{
    is_format_buffer_memory, is_lds_memory_instruction, is_writable_memory,
};
use crate::compiler::prelude::{
    emit_data_share_prelude, emit_div_fixup_f32_prelude, emit_format_prelude,
    emit_global_memory_prelude, emit_vop3p_prelude,
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
        if self.options.stage == MetalShaderStage::Mesh
            && !global_argument_buffer
            && (initial_index >= 29
                || (!direct_sampler_bindings && sampler_count != 0 && initial_index + 1 >= 29))
        {
            return Err(MetalCompileError::InvalidInterface(
                "mesh lowering reserves Metal buffers 29 and 30 for indices and draw geometry"
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
        let uses_lds_memory = self
            .program
            .instructions
            .iter()
            .any(is_lds_memory_instruction);
        if uses_lds_memory {
            emit_data_share_prelude(&mut source);
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
            let access = if kind.writable {
                ", access::read_write"
            } else if kind.storage {
                ", access::read"
            } else {
                ""
            };
            // GCN cube resources are addressed with face-atlas coordinates produced by
            // V_CUBEID/SC/TC/MA, not with a direction vector, so they bind as 2D arrays and
            // sampling indexes the face layer directly (see emit_image_sample).
            let texture_kind = if kind.comparison {
                match kind.dimension {
                    3 => "depth2d_array",
                    _ => "depth2d",
                }
            } else if kind.storage {
                if kind.texture_type == 10 {
                    "texture3d"
                } else {
                    "texture2d"
                }
            } else {
                match kind.dimension {
                    2 => "texture3d",
                    3 => "texture2d_array",
                    _ => "texture2d",
                }
            };
            let component = if kind.comparison {
                "float"
            } else {
                kind.component
            };
            parameters.push(format!(
                "{texture_kind}<{component}{access}> tex{index} [[texture({index})]]"
            ));
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
        if self.options.stage == MetalShaderStage::Pixel
            && self.options.diagnostic_capture_pixel_output
        {
            parameters.push("device atomic_uint* remu_debug_pixel [[buffer(30)]]".to_owned());
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
        source.push_str("    bool exec = true;\n    bool vcc = false;\n    bool scc = false;\n");
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
            let _ = writeln!(
                source,
                "    threadgroup uint remu_mesh_primitive_source_vertices[{}];",
                self.options.mesh_max_primitive_count
            );
        }
        if self.options.stage == MetalShaderStage::Mesh {
            source.push_str(
                "    threadgroup atomic_uint remu_mesh_output_vertex_count;\n\
                 \x20   threadgroup atomic_uint remu_mesh_output_primitive_count;\n",
            );
        }
        if self.split_wave64() {
            source.push_str(
                "    uint remu_wave_index = remu_thread_index >> 6u;\n\
                 \x20   uint remu_wave_half = (remu_thread_index >> 5u) & 1u;\n\
                 \x20   uint remu_wave_lane = remu_lane + (remu_wave_half << 5u);\n",
            );
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
            ControlFlowLowering::Direct(_) => {}
        }
        if matches!(
            self.options.stage,
            MetalShaderStage::Vertex | MetalShaderStage::Pixel
        ) {
            // Graphics stages cannot declare thread_index_in_simdgroup, but the exclusive prefix
            // of one is the architectural lane index. Wave operations are emitted before the
            // EXEC-guarded write, keeping every host lane converged at simdgroup intrinsics.
            source.push_str("    uint remu_lane = simd_prefix_exclusive_sum(1u);\n");
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
                     \x20   uint remu_mesh_primitive = remu_thread_index < remu_mesh_input_primitive_count ? remu_meshlets[remu_meshlet_base + 258u + remu_thread_index] : 0x80000000u;\n\
                     \x20   if (remu_thread_index == 0u) {\n\
                     \x20       atomic_store_explicit(&remu_mesh_output_vertex_count, 0u, memory_order_relaxed);\n\
                     \x20       atomic_store_explicit(&remu_mesh_output_primitive_count, 0u, memory_order_relaxed);\n\
                     \x20   }\n\
                     \x20   threadgroup_barrier(mem_flags::mem_threadgroup);\n",
                );
                if self.options.render_target_array_length > 1 {
                    let _ = writeln!(
                        source,
                        "    remu_mesh_render_target_indices[remu_thread_index] = min(remu_group_id.y, {}u);",
                        self.options.render_target_array_length - 1
                    );
                }
                // Always seed s[3] for mesh: GFX10 merged NGG shaders read
                // - lane ID as `(s3[27:24] << 5) + v_mbcnt_lo`
                // - wave count as `s3[31:28]`, then `(1 << count) - 1` as the reduce mask
                //   that loads per-wave totals from LDS (Void Terrarium soft/main particles).
                // Bits [15:0] still carry the meshlet vertex/primitive counts. Clear [31:24]
                // before packing so stale meshlet high bits cannot override the wave fields.
                source.push_str(
                    "    {\n\
                     \x20       uint remu_mesh_wave_count = ((remu_group_size.x * remu_group_size.y * remu_group_size.z) + 31u) >> 5u;\n\
                     \x20       s[3] = (remu_mesh_counts & 0x00ffffffu)\n\
                     \x20           | (((remu_thread_index >> 5u) & 0xfu) << 24u)\n\
                     \x20           | ((remu_mesh_wave_count & 0xfu) << 28u);\n\
                     \x20   }\n",
                );
                if contains_register_reference(&self.body, 'v', 0) {
                    source.push_str("    v[0] = remu_mesh_primitive;\n");
                }
                if contains_register_reference(&self.body, 'v', 1) {
                    source.push_str("    v[1] = remu_mesh_primitive_base + remu_thread_index;\n");
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
        for register in [VCC_LO, VCC_HI] {
            if contains_register_reference(&self.body, 's', register) {
                let _ = writeln!(source, "    s[{register}] = 0u;");
            }
        }
        if contains_register_reference(&self.body, 's', EXEC_LO)
            || contains_register_reference(&self.body, 's', EXEC_HI)
        {
            let (low, high) = self.ballot_words("true");
            if contains_register_reference(&self.body, 's', EXEC_LO) {
                let _ = writeln!(source, "    s[{EXEC_LO}] = {low};");
            }
            if contains_register_reference(&self.body, 's', EXEC_HI) {
                let _ = writeln!(source, "    s[{EXEC_HI}] = {high};");
            }
        }
        match self.control_flow {
            ControlFlowLowering::Direct(_) => source.push_str(&self.body),
            ControlFlowLowering::Forward => source.push_str(&self.body),
            ControlFlowLowering::Dispatcher => {
                source.push_str("    while (active) {\n        switch (pc) {\n");
                source.push_str(&self.body);
                source.push_str(
                    "        default: active = false; break;\n        }\n\
                     \x20       if (steps >= 16384u) active = false;\n    }\n",
                );
            }
        }
        match self.options.stage {
            MetalShaderStage::Vertex | MetalShaderStage::Pixel => {
                if self.options.stage == MetalShaderStage::Pixel {
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
                        let _ = writeln!(
                            source,
                            "    uint2 remu_debug_pixel_xy = uint2(remu_in.frag_coord.xy);\n\
                             \x20   if (remu_initial_s[{diagnostic_enable_index}] != 0u && remu_debug_pixel_xy.x < 1920u && remu_debug_pixel_xy.y < 1080u) {{\n\
                             \x20       uint remu_debug_pixel_base = (remu_debug_pixel_xy.y * 1920u + remu_debug_pixel_xy.x) * 5u;"
                        );
                        for component in 0..4 {
                            let value = match output.kind {
                                PixelOutputKind::Float => {
                                    format!("as_type<uint>({field}[{component}])")
                                }
                                PixelOutputKind::Uint => format!("{field}[{component}]"),
                                PixelOutputKind::Sint => {
                                    format!("as_type<uint>({field}[{component}])")
                                }
                            };
                            let _ = writeln!(
                                source,
                                "        atomic_store_explicit(&remu_debug_pixel[remu_debug_pixel_base + {}u], {value}, memory_order_relaxed);",
                                component + 1
                            );
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
                    } else {
                        source.push_str("    if (!exec) discard_fragment();\n");
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
                         \x20       uint remu_debug_base = remu_vertex_id * 24u;\n\
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
                let _ = writeln!(
                    source,
                    "    threadgroup_barrier(mem_flags::mem_threadgroup);\n\
                     \x20   uint remu_mesh_output_vertices = min(atomic_load_explicit(&remu_mesh_output_vertex_count, memory_order_relaxed), {}u);\n\
                     \x20   uint remu_mesh_output_primitives = min(atomic_load_explicit(&remu_mesh_output_primitive_count, memory_order_relaxed), {}u);",
                    self.options.mesh_max_vertex_count, self.options.mesh_max_primitive_count
                );
                if self.options.render_target_array_length > 1 {
                    source.push_str(
                        "    if (remu_thread_index < remu_mesh_output_primitives) {\n\
                         \x20       RemuPrimitiveOut remu_primitive_out = {};\n\
                         \x20       uint remu_primitive_source_vertex = remu_mesh_primitive_source_vertices[remu_thread_index];\n\
                         \x20       remu_primitive_out.render_target_array_index = remu_mesh_render_target_indices[remu_primitive_source_vertex];\n\
                         \x20       remu_mesh.set_primitive(remu_thread_index, remu_primitive_out);\n\
                         \x20   }\n",
                    );
                }
                // Metal mesh has no set_vertex_count; only set_primitive_count exists.
                // Vertex occupancy comes from set_vertex + index references.
                source.push_str(
                    "    if (remu_thread_index < remu_mesh_output_vertices) remu_mesh.set_vertex(remu_thread_index, remu_out);\n\
                     \x20   if (remu_thread_index == 0u) {\n\
                     \x20       remu_mesh.set_primitive_count(remu_mesh_output_primitives);\n\
                     \x20   }\n",
                );
            }
        }
        source.push_str("}\n");
        scalarize_register_state(&source)
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
