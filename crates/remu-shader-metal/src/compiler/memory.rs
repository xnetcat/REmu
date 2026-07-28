// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

//! Memory lowering: scalar loads, MUBUF/MTBUF buffers, FLAT/GLOBAL and LDS data share.

use std::fmt::Write as _;

use remu_shader::{DataShareControl, Instruction, InstructionControl, OperandKind};

use crate::compiler::vector::vector_operand;
use crate::{Compiler, MetalShaderStage, format_u32};

fn raw_load_shape(name: &str) -> Result<(u32, bool, bool, bool), String> {
    if !name.contains("Load") {
        return Err(format!("{name} is not a raw load"));
    }
    let width = if name.contains("Ubyte") || name.contains("Sbyte") {
        1
    } else if name.contains("Ushort") || name.contains("Sshort") || name.contains("ShortD16") {
        2
    } else if name.contains("Dword") {
        4
    } else {
        return Err(format!("unsupported raw load width for {name}"));
    };
    Ok((
        width,
        name.contains("Sbyte") || name.contains("Sshort"),
        name.contains("D16"),
        name.ends_with("D16Hi"),
    ))
}

fn raw_store_shape(name: &str) -> Result<(u32, bool), String> {
    if !name.contains("Store") {
        return Err(format!("{name} is not a raw store"));
    }
    if name.contains("Byte") {
        Ok((1, name.ends_with("D16Hi")))
    } else if name.contains("Short") {
        Ok((2, name.ends_with("D16Hi")))
    } else if name.contains("Dword") {
        Ok((4, false))
    } else {
        Err(format!("unsupported raw store width for {name}"))
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn buffer_oob_expression(
    descriptor_null: &str,
    oob_select: &str,
    index: &str,
    record_count: &str,
    offset: &str,
    stride: &str,
    swizzled: &str,
    buffer_offset: &str,
    reduced_records: &str,
    payload: &str,
    component_offset: u32,
) -> String {
    let offset = if component_offset == 0 {
        offset.to_owned()
    } else {
        format!("{offset} + {component_offset}u")
    };
    let buffer_offset = if component_offset == 0 {
        buffer_offset.to_owned()
    } else {
        format!("{buffer_offset} + {component_offset}u")
    };
    format!(
        "{descriptor_null} || ({oob_select} == 0u ? ({index} >= {record_count} || {offset} >= {stride}) : {oob_select} == 1u ? ({index} >= {record_count}) : {oob_select} == 2u ? ({record_count} == 0u) : ({swizzled} && {stride} != 0u ? ({index} >= {record_count} || !remu_range_valid({offset}, {payload}, {stride})) : !remu_range_valid({buffer_offset}, {payload}, {reduced_records})))"
    )
}

pub(crate) fn format_buffer_element_oob_expression(
    descriptor_oob: &str,
    address: &str,
    element_size: &str,
    binding_length: &str,
) -> String {
    // A structured buffer's stride, not the total formatted element size, positions records.
    // Requiring a float3 payload to begin at a multiple of 12 would reject legal records in a
    // 28-byte vertex layout. The bytewise format loader below already handles every component at
    // its architectural byte offset, so only format validity and the actual bound range matter.
    format!(
        "{descriptor_oob} || {element_size} == 0u || !remu_range_valid({address}, {element_size}, {binding_length})"
    )
}

fn memory_payload_bytes(name: &str, _dword_count: u32) -> Result<u32, String> {
    if name.contains("Atomic") || name.contains("Dword") {
        Ok(4)
    } else if name.contains("Byte") || name.contains("Ubyte") || name.contains("Sbyte") {
        Ok(1)
    } else if name.contains("Short") || name.contains("Ushort") || name.contains("Sshort") {
        Ok(2)
    } else {
        Err(format!("unknown memory payload width for {name}"))
    }
}

fn format_component_count(name: &str) -> u32 {
    if name.ends_with("Xyzw") {
        4
    } else if name.ends_with("Xyz") {
        3
    } else if name.ends_with("Xy") {
        2
    } else {
        1
    }
}

pub(crate) fn is_lds_memory_instruction(instruction: &Instruction) -> bool {
    match instruction.control {
        Some(InstructionControl::DataShare(_)) => !instruction.opcode.is("DsSwizzleB32"),
        Some(InstructionControl::BufferMemory(control)) => control.lds,
        Some(InstructionControl::GlobalMemory(control)) => control.lds,
        _ => false,
    }
}

fn is_lds_atomic_name(name: &str) -> bool {
    matches!(
        name,
        "DsAddU32"
            | "DsSubU32"
            | "DsIncU32"
            | "DsDecU32"
            | "DsMinI32"
            | "DsMaxI32"
            | "DsMinU32"
            | "DsMaxU32"
            | "DsAndB32"
            | "DsOrB32"
            | "DsXorB32"
            | "DsCmpstB32"
            | "DsAddRtnU32"
            | "DsSubRtnU32"
            | "DsIncRtnU32"
            | "DsDecRtnU32"
            | "DsMinRtnI32"
            | "DsMaxRtnI32"
            | "DsMinRtnU32"
            | "DsMaxRtnU32"
            | "DsAndRtnB32"
            | "DsOrRtnB32"
            | "DsXorRtnB32"
            | "DsWrxchgRtnB32"
            | "DsCmpstRtnB32"
    )
}

const fn ds_combined_offset(control: &DataShareControl) -> u32 {
    control.offset0 | (control.offset1 << 8)
}

fn ds_swizzle_lane_expression(offset: u16) -> String {
    if offset >= 0xe000 {
        let mask = u32::from(offset & 0x1f);
        let reverse = if mask.count_ones() == 5 {
            "0u".to_owned()
        } else {
            format!(
                "(reverse_bits(remu_lane & 31u) >> {}u)",
                27 + mask.count_ones()
            )
        };
        format!("({reverse} | (remu_lane & {mask}u))")
    } else if offset >= 0xc000 {
        let rotate = u32::from((offset >> 5) & 0x1f);
        let mask = u32::from(offset & 0x1f);
        let delta = if offset & (1 << 10) == 0 {
            rotate
        } else {
            32 - rotate
        };
        format!(
            "((remu_lane & {mask}u) | (((remu_lane + {delta}u) & 31u) & {}u))",
            31 ^ mask
        )
    } else if offset & 0x8000 != 0 {
        format!("((remu_lane & 0xFFFFFFFCu) | ((0x{offset:04X}u >> ((remu_lane & 3u) * 2u)) & 3u))")
    } else {
        let and_mask = u32::from(offset & 0x1f);
        let or_mask = u32::from((offset >> 5) & 0x1f);
        let xor_mask = u32::from((offset >> 10) & 0x1f);
        format!(
            "((remu_lane & 0xFFFFFFE0u) | ((((remu_lane & 31u) & {and_mask}u) | {or_mask}u) ^ {xor_mask}u))"
        )
    }
}

pub(crate) fn is_format_buffer_memory(instruction: &Instruction) -> bool {
    instruction.opcode.starts_with("BufferLoadFormat")
        || instruction.opcode.starts_with("BufferStoreFormat")
        || instruction.opcode.starts_with("TBufferLoadFormat")
        || instruction.opcode.starts_with("TBufferStoreFormat")
}

pub(crate) fn is_writable_memory(instruction: &Instruction) -> bool {
    instruction.opcode.contains("Store") || instruction.opcode.contains("Atomic")
}

impl Compiler<'_> {
    pub(crate) fn emit_scalar_memory(&mut self, instruction: &Instruction) -> Result<(), String> {
        let Some(InstructionControl::ScalarMemory(control)) = instruction.control.as_ref() else {
            return Err("missing scalar-memory control".to_owned());
        };
        let binding_index = self
            .evaluation
            .global_memory_bindings
            .iter()
            .position(|binding| binding.instruction_pcs.contains(&instruction.pc))
            .ok_or_else(|| "scalar load has no evaluated memory binding".to_owned())?;
        let binding_length = self.global_binding_length(binding_index);
        let dynamic = control
            .dynamic_offset_register
            .map_or_else(|| "0u".to_owned(), |register| format!("s[{register}]"));
        let address = self.temp(
            "uint",
            &format!(
                "({dynamic} + {})",
                format_u32(control.immediate_offset_bytes.cast_unsigned())
            ),
        );
        for (index, destination) in instruction.destinations.iter().enumerate() {
            if destination.kind != OperandKind::ScalarRegister {
                return Err("scalar load has a non-scalar destination".to_owned());
            }
            let offset = u32::try_from(index)
                .map_err(|_| "scalar load destination count overflow".to_owned())?
                * 4;
            let expression =
                format!("remu_load_u32(b{binding_index}, {address} + {offset}u, {binding_length})");
            self.store_scalar(destination.value, &expression);
        }
        Ok(())
    }

    #[allow(clippy::similar_names, clippy::too_many_lines)]
    pub(crate) fn emit_buffer_memory(&mut self, instruction: &Instruction) -> Result<(), String> {
        let Some(InstructionControl::BufferMemory(control)) = instruction.control.as_ref() else {
            return Err("missing buffer-memory control".to_owned());
        };
        let name = instruction
            .opcode
            .known_name()
            .ok_or_else(|| "raw buffer opcode".to_owned())?;
        if control.lds && (!name.starts_with("BufferLoad") || control.tfe) {
            return Err("MUBUF LDS transfer is legal only for non-TFE buffer loads".to_owned());
        }
        if control.lds && control.vector_data != 124 {
            return Err("MUBUF LDS transfer must encode M0 in VDATA".to_owned());
        }
        if self.options.stage == MetalShaderStage::Vertex
            && let Some(input) = self
                .evaluation
                .vertex_inputs
                .iter()
                .find(|input| input.pc == instruction.pc)
        {
            if control.dword_count == 0 || control.dword_count > input.component_count {
                return Err(format!(
                    "vertex fetch requests {} components from a {}-component input",
                    control.dword_count, input.component_count
                ));
            }
            for component in 0..control.dword_count {
                let field = if input.component_count == 1 {
                    format!("remu_in.in{}", input.location)
                } else {
                    format!("remu_in.in{}[{component}]", input.location)
                };
                self.store_vector(
                    control.vector_data + component,
                    &format!("as_type<uint>({field})"),
                );
            }
            return Ok(());
        }
        let binding_index = self
            .evaluation
            .global_memory_bindings
            .iter()
            .position(|binding| binding.instruction_pcs.contains(&instruction.pc))
            .ok_or_else(|| format!("{} has no evaluated buffer binding", instruction.opcode))?;
        let binding_length = self.global_binding_length(binding_index);
        let scalar_offset = self.raw_source(instruction, 2)?;
        let vector_index = if self.options.stage == MetalShaderStage::Vertex
            && self.options.diagnostic_force_vertex_fetch_id
            && name.starts_with("BufferLoadFormat")
            && control.index_enabled
        {
            "remu_vertex_id".to_owned()
        } else if control.index_enabled {
            format!("v[{}]", control.vector_address)
        } else {
            "0u".to_owned()
        };
        let vector_offset = if control.offset_enabled {
            format!(
                "v[{}]",
                control.vector_address + u32::from(control.index_enabled)
            )
        } else {
            "0u".to_owned()
        };
        let resource0 = format!("s[{}]", control.scalar_resource);
        let resource1 = format!("s[{}]", control.scalar_resource + 1);
        let resource2 = format!("s[{}]", control.scalar_resource + 2);
        let resource3 = format!("s[{}]", control.scalar_resource + 3);
        let resource_base = self.temp(
            "ulong",
            &format!("(ulong){resource0} | (((ulong){resource1} & 0xFFFFul) << 32)"),
        );
        let resource_bias = self.temp("uint", &format!("(uint)({resource_base} & 3ul)"));
        let stride = self.temp("uint", &format!("({resource1} >> 16) & 0x3FFFu"));
        let swizzled = self.temp("bool", &format!("({resource1} & 0x80000000u) != 0u"));
        let add_tid = self.temp("bool", &format!("({resource3} & 0x00800000u) != 0u"));
        let index = self.temp(
            "uint",
            &format!(
                "{vector_index} + ({add_tid} ? ({} & 63u) : 0u)",
                self.architectural_lane()
            ),
        );
        let offset = self.temp(
            "uint",
            &format!(
                "{vector_offset} + {}",
                format_u32(control.offset_bytes.cast_unsigned())
            ),
        );
        let index_stride = self.temp("uint", &format!("8u << (({resource3} >> 21) & 3u)"));
        let element_size = "4u";
        let index_msb = self.temp("uint", &format!("{index} / {index_stride}"));
        let index_lsb = self.temp("uint", &format!("{index} % {index_stride}"));
        let offset_msb = self.temp("uint", &format!("{offset} / {element_size}"));
        let offset_lsb = self.temp("uint", &format!("{offset} % {element_size}"));
        let linear_offset = self.temp("uint", &format!("{index} * {stride} + {offset}"));
        let swizzled_offset = self.temp(
            "uint",
            &format!(
                "({index_msb} * {stride} + {offset_msb} * {element_size}) * {index_stride} + {index_lsb} * {element_size} + {offset_lsb}"
            ),
        );
        let buffer_offset = self.temp(
            "uint",
            &format!("{swizzled} ? {swizzled_offset} : {linear_offset}"),
        );
        let address = self.temp(
            "uint",
            &format!("{resource_bias} + {scalar_offset} + {buffer_offset}"),
        );
        let payload = if is_format_buffer_memory(instruction) {
            let unified = control
                .instruction_format
                .map_or_else(|| format!("({resource3} >> 12) & 0x7Fu"), format_u32);
            let table = self.temp("uint", &format!("remu_gfx10_formats[{unified}]"));
            let data_format = self.temp("uint", &format!("{table} & 0xFFu"));
            self.temp("uint", &format!("remu_format_size({data_format})"))
        } else {
            format_u32(memory_payload_bytes(name, control.dword_count)?)
        };
        let record_count = self.temp("uint", &resource2);
        let oob_select = self.temp("uint", &format!("({resource3} >> 28) & 3u"));
        let descriptor_null = self.temp(
            "bool",
            &format!("({resource0} | {resource1} | {resource2} | {resource3}) == 0u"),
        );
        let reduced_records = self.temp(
            "uint",
            &format!("{record_count} > {scalar_offset} ? {record_count} - {scalar_offset} : 0u"),
        );
        let oob = self.temp(
            "bool",
            &buffer_oob_expression(
                &descriptor_null,
                &oob_select,
                &index,
                &record_count,
                &offset,
                &stride,
                &swizzled,
                &buffer_offset,
                &reduced_records,
                &payload,
                0,
            ),
        );
        if is_format_buffer_memory(instruction) {
            return self.emit_format_buffer_memory(
                instruction,
                control,
                binding_index,
                &binding_length,
                &address,
                &oob,
            );
        }
        let component_oobs = (control.dword_count > 1
            && name.contains("Dword")
            && !name.contains("Atomic"))
        .then(|| {
            (0..control.dword_count)
                .map(|component| {
                    let expression = buffer_oob_expression(
                        &descriptor_null,
                        &oob_select,
                        &index,
                        &record_count,
                        &offset,
                        &stride,
                        &swizzled,
                        &buffer_offset,
                        &reduced_records,
                        "4u",
                        component * 4,
                    );
                    self.temp("bool", &expression)
                })
                .collect::<Vec<_>>()
        });
        self.emit_raw_memory_operation(
            name,
            control.vector_data,
            control.vector_data,
            control.dword_count,
            binding_index,
            &binding_length,
            &address,
            &oob,
            component_oobs.as_deref(),
            control.glc,
            control.lds,
            control.tfe,
        )
    }

    pub(crate) fn emit_global_memory(&mut self, instruction: &Instruction) -> Result<(), String> {
        let Some(InstructionControl::GlobalMemory(control)) = instruction.control.as_ref() else {
            return Err("missing global-memory control".to_owned());
        };
        let name = instruction
            .opcode
            .known_name()
            .ok_or_else(|| "raw global opcode".to_owned())?;
        let guest_address = if let Some(scalar_address) = control.scalar_address {
            self.temp(
                "ulong",
                &format!(
                    "((ulong)s[{}] | ((ulong)s[{}] << 32)) + (ulong)v[{}] + (ulong)(long){}",
                    scalar_address,
                    scalar_address + 1,
                    control.vector_address,
                    control.offset_bytes
                ),
            )
        } else {
            self.temp(
                "ulong",
                &format!(
                    "((ulong)v[{}] | ((ulong)v[{}] << 32)) + (ulong)(long){}",
                    control.vector_address,
                    control.vector_address + 1,
                    control.offset_bytes
                ),
            )
        };
        let binding_index = self
            .evaluation
            .global_memory_bindings
            .iter()
            .position(|binding| binding.instruction_pcs.contains(&instruction.pc))
            .or_else(|| {
                (control.scalar_address.is_none()
                    && self.evaluation.global_memory_bindings.len() == 1)
                    .then_some(0)
            })
            .ok_or_else(|| {
                "GLOBAL with a VGPR-only address has no statically resolved guest memory range"
                    .to_owned()
            })?;
        let binding_base = self.temp("ulong", &self.global_binding_base(binding_index));
        let binding_length = self.global_binding_length(binding_index);
        let local_address = self.temp("uint", &format!("(uint)({guest_address} - {binding_base})"));
        let before = self.temp("bool", &format!("{guest_address} < {binding_base}"));
        let oob = self.temp("bool", &before);
        self.emit_raw_memory_operation(
            name,
            control.vector_data,
            control.vector_destination,
            control.dword_count,
            binding_index,
            &binding_length,
            &local_address,
            &oob,
            None,
            control.glc,
            control.lds,
            false,
        )
    }

    #[allow(clippy::too_many_lines)]
    pub(crate) fn emit_data_share(&mut self, instruction: &Instruction) -> Result<(), String> {
        let Some(InstructionControl::DataShare(control)) = instruction.control.as_ref() else {
            return Err("missing data-share control".to_owned());
        };
        if control.gds {
            return Err("GDS operations require a system-wide data-share resource".to_owned());
        }
        let name = instruction
            .opcode
            .known_name()
            .ok_or_else(|| "raw data-share opcode".to_owned())?;
        if name == "DsSwizzleB32" {
            return self.emit_ds_swizzle(instruction, control);
        }
        if !self.is_threadgroup_stage() {
            return match name {
                "DsWriteAddtidB32" => self.emit_graphics_ds_write_addtid(instruction, control),
                "DsReadAddtidB32" => self.emit_graphics_ds_read_addtid(instruction, control),
                _ => Err(
                    "LDS memory is workgroup-shared and cannot be represented by a Metal graphics stage"
                        .to_owned(),
                ),
            };
        }
        if self.options.lds_size_bytes == 0 || self.options.lds_size_bytes & 3 != 0 {
            return Err(
                "LDS memory operations require a non-zero dword-aligned allocation size".to_owned(),
            );
        }
        if self.options.lds_size_bytes > 32 * 1024 {
            return Err(format!(
                "{} bytes of guest LDS exceed Apple Metal's 32768-byte threadgroup-memory limit",
                self.options.lds_size_bytes
            ));
        }
        if self.options.stage == MetalShaderStage::Mesh && name.starts_with("DsRead") {
            // One guest wave64 is executed by two Apple SIMD32 groups. RDNA2 wave-ordering makes
            // an earlier LDS write visible to a later read without an explicit S_BARRIER, but
            // the two host SIMD groups otherwise advance independently. Converge both halves
            // before a read so the mesh function observes the same wave-local LDS state.
            self.line(3, "threadgroup_barrier(mem_flags::mem_threadgroup);");
        }

        match name {
            "DsWriteAddtidB32" => self.emit_ds_write_addtid(instruction, control),
            "DsReadAddtidB32" => self.emit_ds_read_addtid(instruction, control),
            "DsWriteB32" => self.emit_ds_write(instruction, control, 1),
            "DsWriteB64" => self.emit_ds_write(instruction, control, 2),
            "DsWriteB96" => self.emit_ds_write(instruction, control, 3),
            "DsWriteB128" => self.emit_ds_write(instruction, control, 4),
            "DsWrite2B32" => self.emit_ds_write_pair(instruction, control, false),
            "DsWrite2St64B32" => self.emit_ds_write_pair(instruction, control, true),
            "DsReadB32" => self.emit_ds_read(instruction, control, 1),
            "DsReadB96" => self.emit_ds_read(instruction, control, 3),
            "DsReadB128" => self.emit_ds_read(instruction, control, 4),
            "DsRead2B32" => self.emit_ds_read_pair(instruction, control, false),
            "DsRead2St64B32" => self.emit_ds_read_pair(instruction, control, true),
            _ if is_lds_atomic_name(name) => self.emit_ds_atomic(instruction, control, name),
            _ => Err(format!("unsupported LDS opcode {name}")),
        }
    }

    fn emit_ds_write(
        &mut self,
        instruction: &Instruction,
        control: &DataShareControl,
        dword_count: usize,
    ) -> Result<(), String> {
        if instruction.sources.len() < dword_count + 1 {
            return Err(format!("LDS write requires {dword_count} data dwords"));
        }
        let base = self.temp("uint", &self.raw_source(instruction, 0)?);
        let first_offset = ds_combined_offset(control);
        let mut values = Vec::with_capacity(dword_count);
        for component in 0..dword_count {
            values.push(self.temp("uint", &self.raw_source(instruction, component + 1)?));
        }
        for (component, value) in values.iter().enumerate() {
            let component_offset = u32::try_from(component)
                .map_err(|_| "LDS component index overflow".to_owned())?
                * 4;
            let address = self.ds_address(&base, first_offset + component_offset);
            let index = self.temp("uint", &format!("{address} >> 2"));
            self.emit_ds_store_index(&index, value);
        }
        Ok(())
    }

    fn emit_ds_write_addtid(
        &mut self,
        instruction: &Instruction,
        control: &DataShareControl,
    ) -> Result<(), String> {
        if instruction.sources.len() != 1 {
            return Err("DS_WRITE_ADDTID_B32 requires DATA0 only".to_owned());
        }
        let value = self.temp("uint", &self.raw_source(instruction, 0)?);
        let offset = ds_combined_offset(control);
        let address = self.temp(
            "uint",
            &format!(
                "(s[124] & 0xFFFFu) + {} + remu_thread_index * 4u",
                format_u32(offset)
            ),
        );
        let index = self.temp("uint", &format!("{address} >> 2"));
        self.emit_ds_store_index(&index, &value);
        Ok(())
    }

    fn emit_ds_read_addtid(
        &mut self,
        instruction: &Instruction,
        control: &DataShareControl,
    ) -> Result<(), String> {
        if !instruction.sources.is_empty() || instruction.destinations.len() != 1 {
            return Err("DS_READ_ADDTID_B32 requires VDST only".to_owned());
        }
        let offset = ds_combined_offset(control);
        let address = self.temp(
            "uint",
            &format!(
                "(s[124] & 0xFFFFu) + {} + remu_thread_index * 4u",
                format_u32(offset)
            ),
        );
        let index = self.temp("uint", &format!("{address} >> 2"));
        let value = self.emit_ds_load_index(&index);
        self.store_vector(vector_operand(instruction.destinations[0])?, &value);
        Ok(())
    }

    fn emit_graphics_ds_write_addtid(
        &mut self,
        instruction: &Instruction,
        control: &DataShareControl,
    ) -> Result<(), String> {
        if instruction.sources.len() != 1 {
            return Err("DS_WRITE_ADDTID_B32 requires DATA0 only".to_owned());
        }
        let offset = ds_combined_offset(control);
        if !offset.is_multiple_of(256) {
            return Err(format!(
                "graphics DS_WRITE_ADDTID_B32 offset {offset} can overlap another lane"
            ));
        }
        let slot = self
            .graphics_addtid_write_slot(instruction.pc)
            .ok_or_else(|| "graphics ADDTID write has no local spill slot".to_owned())?;
        let value = self.temp("uint", &self.raw_source(instruction, 0)?);
        let address = self.temp(
            "uint",
            &format!(
                "(s[124] & 0xFFFFu) + {}",
                format_u32(ds_combined_offset(control))
            ),
        );
        self.line(3, "if (exec) {");
        self.line(4, "remu_addtid_clock += 1u;");
        self.line(4, &format!("remu_addtid_address[{slot}] = {address};"));
        self.line(4, &format!("remu_addtid_value[{slot}] = {value};"));
        self.line(
            4,
            &format!("remu_addtid_sequence[{slot}] = remu_addtid_clock;"),
        );
        self.line(4, &format!("remu_addtid_valid[{slot}] = true;"));
        self.line(3, "}");
        Ok(())
    }

    fn emit_graphics_ds_read_addtid(
        &mut self,
        instruction: &Instruction,
        control: &DataShareControl,
    ) -> Result<(), String> {
        if !instruction.sources.is_empty() || instruction.destinations.len() != 1 {
            return Err("DS_READ_ADDTID_B32 requires VDST only".to_owned());
        }
        let offset = ds_combined_offset(control);
        if !offset.is_multiple_of(256) {
            return Err(format!(
                "graphics DS_READ_ADDTID_B32 offset {offset} can overlap another lane"
            ));
        }
        let slot_count = self.graphics_addtid_write_slot_count();
        if slot_count == 0 {
            return Err("graphics ADDTID read has no local spill writes".to_owned());
        }
        let address = self.temp(
            "uint",
            &format!(
                "(s[124] & 0xFFFFu) + {}",
                format_u32(ds_combined_offset(control))
            ),
        );
        let value = self.temp("uint", "0u");
        let sequence = self.temp("uint", "0u");
        self.line(3, "if (exec) {");
        let _ = writeln!(
            self.body,
            "        for (uint remu_addtid_i = 0u; remu_addtid_i < {slot_count}u; ++remu_addtid_i) {{"
        );
        self.line(
            5,
            &format!(
                "if (remu_addtid_valid[remu_addtid_i] && remu_addtid_address[remu_addtid_i] == {address} && remu_addtid_sequence[remu_addtid_i] >= {sequence}) {{"
            ),
        );
        self.line(
            6,
            &format!("{sequence} = remu_addtid_sequence[remu_addtid_i];"),
        );
        self.line(6, &format!("{value} = remu_addtid_value[remu_addtid_i];"));
        self.line(5, "}");
        self.line(4, "}");
        self.line(3, "}");
        self.store_vector(vector_operand(instruction.destinations[0])?, &value);
        Ok(())
    }

    fn emit_ds_write_pair(
        &mut self,
        instruction: &Instruction,
        control: &DataShareControl,
        stride64: bool,
    ) -> Result<(), String> {
        if instruction.sources.len() < 3 {
            return Err("LDS write2 requires two data dwords".to_owned());
        }
        let base = self.temp("uint", &self.raw_source(instruction, 0)?);
        let first = self.temp("uint", &self.raw_source(instruction, 1)?);
        let second = self.temp("uint", &self.raw_source(instruction, 2)?);
        let scale = if stride64 { 4 * 64 } else { 4 };
        for (offset, value) in [
            (control.offset0 * scale, first),
            (control.offset1 * scale, second),
        ] {
            let address = self.ds_address(&base, offset);
            let index = self.temp("uint", &format!("{address} >> 2"));
            self.emit_ds_store_index(&index, &value);
        }
        Ok(())
    }

    fn emit_ds_read(
        &mut self,
        instruction: &Instruction,
        control: &DataShareControl,
        dword_count: usize,
    ) -> Result<(), String> {
        if instruction.destinations.len() < dword_count {
            return Err(format!(
                "LDS read requires {dword_count} destination dwords"
            ));
        }
        let base = self.temp("uint", &self.raw_source(instruction, 0)?);
        let first_offset = ds_combined_offset(control);
        let mut values = Vec::with_capacity(dword_count);
        for component in 0..dword_count {
            let component_offset = u32::try_from(component)
                .map_err(|_| "LDS component index overflow".to_owned())?
                * 4;
            let address = self.ds_address(&base, first_offset + component_offset);
            let index = self.temp("uint", &format!("{address} >> 2"));
            values.push(self.emit_ds_load_index(&index));
        }
        for (destination, value) in instruction.destinations.iter().zip(values) {
            self.store_vector(vector_operand(*destination)?, &value);
        }
        Ok(())
    }

    fn emit_ds_read_pair(
        &mut self,
        instruction: &Instruction,
        control: &DataShareControl,
        stride64: bool,
    ) -> Result<(), String> {
        if instruction.destinations.len() < 2 {
            return Err("LDS read2 requires two destination dwords".to_owned());
        }
        let base = self.temp("uint", &self.raw_source(instruction, 0)?);
        let scale = if stride64 { 4 * 64 } else { 4 };
        let mut values = Vec::with_capacity(2);
        for offset in [control.offset0 * scale, control.offset1 * scale] {
            let address = self.ds_address(&base, offset);
            let index = self.temp("uint", &format!("{address} >> 2"));
            values.push(self.emit_ds_load_index(&index));
        }
        for (destination, value) in instruction.destinations.iter().zip(values) {
            self.store_vector(vector_operand(*destination)?, &value);
        }
        Ok(())
    }

    fn emit_ds_atomic(
        &mut self,
        instruction: &Instruction,
        control: &DataShareControl,
        name: &str,
    ) -> Result<(), String> {
        let required_sources = if matches!(name, "DsCmpstB32" | "DsCmpstRtnB32") {
            3
        } else {
            2
        };
        if instruction.sources.len() < required_sources {
            return Err(format!("{name} has missing atomic operands"));
        }
        let base = self.temp("uint", &self.raw_source(instruction, 0)?);
        let address = self.ds_address(&base, ds_combined_offset(control));
        let index = self.temp("uint", &format!("{address} >> 2"));
        let data0 = self.temp("uint", &self.raw_source(instruction, 1)?);
        let data1 = if required_sources == 3 {
            Some(self.temp("uint", &self.raw_source(instruction, 2)?))
        } else {
            None
        };
        let original = self.temp("uint", "0u");
        let pointer = format!("remu_lds + {index}");
        let operation = match name {
            "DsAddU32" | "DsAddRtnU32" => {
                format!("atomic_fetch_add_explicit({pointer}, {data0}, memory_order_relaxed)")
            }
            "DsSubU32" | "DsSubRtnU32" => {
                format!("atomic_fetch_sub_explicit({pointer}, {data0}, memory_order_relaxed)")
            }
            "DsIncU32" | "DsIncRtnU32" => format!("remu_ds_inc({pointer}, {data0})"),
            "DsDecU32" | "DsDecRtnU32" => format!("remu_ds_dec({pointer}, {data0})"),
            "DsMinI32" | "DsMinRtnI32" => format!("remu_ds_smin({pointer}, {data0})"),
            "DsMaxI32" | "DsMaxRtnI32" => format!("remu_ds_smax({pointer}, {data0})"),
            "DsMinU32" | "DsMinRtnU32" => {
                format!("atomic_fetch_min_explicit({pointer}, {data0}, memory_order_relaxed)")
            }
            "DsMaxU32" | "DsMaxRtnU32" => {
                format!("atomic_fetch_max_explicit({pointer}, {data0}, memory_order_relaxed)")
            }
            "DsAndB32" | "DsAndRtnB32" => {
                format!("atomic_fetch_and_explicit({pointer}, {data0}, memory_order_relaxed)")
            }
            "DsOrB32" | "DsOrRtnB32" => {
                format!("atomic_fetch_or_explicit({pointer}, {data0}, memory_order_relaxed)")
            }
            "DsXorB32" | "DsXorRtnB32" => {
                format!("atomic_fetch_xor_explicit({pointer}, {data0}, memory_order_relaxed)")
            }
            "DsWrxchgRtnB32" => {
                format!("atomic_exchange_explicit({pointer}, {data0}, memory_order_relaxed)")
            }
            "DsCmpstB32" | "DsCmpstRtnB32" => format!(
                "remu_ds_cmpst({pointer}, {data0}, {})",
                data1.expect("compare/store requires DATA1")
            ),
            _ => return Err(format!("unsupported LDS atomic opcode {name}")),
        };
        self.line(
            3,
            &format!(
                "if (exec && {index} < {}u) {original} = {operation};",
                self.options.lds_size_bytes / 4
            ),
        );
        if name.contains("Rtn") {
            let destination = instruction
                .destinations
                .first()
                .copied()
                .ok_or_else(|| format!("{name} has no return destination"))?;
            self.store_vector(vector_operand(destination)?, &original);
        }
        Ok(())
    }

    fn emit_ds_swizzle(
        &mut self,
        instruction: &Instruction,
        control: &DataShareControl,
    ) -> Result<(), String> {
        if instruction.sources.len() != 1 || instruction.destinations.len() != 1 {
            return Err("DS_SWIZZLE_B32 requires one source and one destination".to_owned());
        }
        let source = vector_operand(instruction.sources[0])?;
        let destination = vector_operand(instruction.destinations[0])?;
        let offset = u16::try_from(ds_combined_offset(control))
            .expect("two 8-bit data-share offsets fit u16");
        let source_lane = self.temp("uint", &ds_swizzle_lane_expression(offset));
        let shuffled = self.temp(
            "uint",
            &format!("simd_shuffle(v[{source}], (ushort)({source_lane} & 31u))"),
        );
        let source_active = self.temp(
            "uint",
            &format!("simd_shuffle(exec ? 1u : 0u, (ushort)({source_lane} & 31u))"),
        );
        self.store_vector(
            destination,
            &format!("{source_active} != 0u ? {shuffled} : 0u"),
        );
        Ok(())
    }

    fn ds_address(&mut self, base: &str, offset_bytes: u32) -> String {
        if offset_bytes == 0 {
            base.to_owned()
        } else {
            self.temp("uint", &format!("{base} + {}", format_u32(offset_bytes)))
        }
    }

    fn emit_ds_load_index(&mut self, index: &str) -> String {
        let value = self.temp("uint", "0u");
        self.line(
            3,
            &format!(
                "if (exec && {index} < {}u) {value} = atomic_load_explicit(remu_lds + {index}, memory_order_relaxed);",
                self.options.lds_size_bytes / 4
            ),
        );
        value
    }

    fn emit_ds_store_index(&mut self, index: &str, value: &str) {
        self.line(
            3,
            &format!(
                "if (exec && {index} < {}u) atomic_store_explicit(remu_lds + {index}, {value}, memory_order_relaxed);",
                self.options.lds_size_bytes / 4
            ),
        );
    }

    #[allow(clippy::too_many_arguments, clippy::too_many_lines)]
    fn emit_raw_memory_operation(
        &mut self,
        name: &str,
        vector_data: u32,
        vector_destination: u32,
        dword_count: u32,
        binding_index: usize,
        binding_length: &str,
        address: &str,
        descriptor_oob: &str,
        component_descriptor_oobs: Option<&[String]>,
        glc: bool,
        lds: bool,
        tfe: bool,
    ) -> Result<(), String> {
        let load = name.contains("Load");
        let store = name.contains("Store");
        let atomic = name.contains("Atomic");
        if !load && !store && !atomic {
            return Err(format!("unsupported memory opcode {name}"));
        }
        let subdword = name.contains("Byte")
            || name.contains("Ubyte")
            || name.contains("Sbyte")
            || name.contains("Short")
            || name.contains("Ushort")
            || name.contains("Sshort");
        let aligned_address = if subdword {
            address.to_owned()
        } else {
            self.temp("uint", &format!("{address} & ~3u"))
        };

        if load {
            let (width, signed, d16, high) = raw_load_shape(name)?;
            if lds {
                if !self.is_threadgroup_stage() {
                    return Err("memory-to-LDS transfers require a Metal compute stage".to_owned());
                }
                if dword_count != 1 || tfe {
                    return Err(
                        "memory-to-LDS transfer must return exactly one non-TFE dword".to_owned(),
                    );
                }
            }
            for component in 0..dword_count {
                let descriptor_oob = component_descriptor_oobs
                    .and_then(|bounds| bounds.get(component as usize))
                    .map_or(descriptor_oob, String::as_str);
                let component_address = if component == 0 {
                    aligned_address.clone()
                } else {
                    format!("{aligned_address} + {}u", component * 4)
                };
                let loaded = match width {
                    1 => format!(
                        "remu_load_u8(b{binding_index}, {component_address}, {binding_length})"
                    ),
                    2 => format!(
                        "remu_load_u16(b{binding_index}, {component_address}, {binding_length})"
                    ),
                    4 => format!(
                        "remu_load_u32(b{binding_index}, {component_address}, {binding_length})"
                    ),
                    _ => unreachable!("raw load widths are fixed"),
                };
                let in_range = format!(
                    "!{descriptor_oob} && remu_range_valid({component_address}, {width}u, {binding_length})"
                );
                let raw = self.temp("uint", &format!("{in_range} ? {loaded} : 0u"));
                let converted = if signed && width == 1 {
                    format!("({raw} & 0x80u) != 0u ? ({raw} | 0xFFFFFF00u) : {raw}")
                } else if signed && width == 2 {
                    format!("({raw} & 0x8000u) != 0u ? ({raw} | 0xFFFF0000u) : {raw}")
                } else {
                    raw.clone()
                };
                let value = self.temp("uint", &converted);
                if lds {
                    let lds_index = self.temp("uint", "((s[124] & 0xFFFFu) >> 2) + remu_lane");
                    self.line(
                        3,
                        &format!(
                            "if (exec && {lds_index} < {}u) atomic_store_explicit(remu_lds + {lds_index}, {value}, memory_order_relaxed);",
                            self.options.lds_size_bytes / 4
                        ),
                    );
                } else if d16 {
                    let register = vector_destination + component;
                    let merged = if high {
                        format!("(v[{register}] & 0xFFFFu) | (({value} & 0xFFFFu) << 16)")
                    } else {
                        format!("(v[{register}] & 0xFFFF0000u) | ({value} & 0xFFFFu)")
                    };
                    self.store_vector(register, &merged);
                } else {
                    self.store_vector(vector_destination + component, &value);
                }
            }
            if tfe {
                self.store_vector(vector_destination + dword_count, "0u");
            }
            return Ok(());
        }

        if store {
            let (width, high) = raw_store_shape(name)?;
            for component in 0..dword_count {
                let descriptor_oob = component_descriptor_oobs
                    .and_then(|bounds| bounds.get(component as usize))
                    .map_or(descriptor_oob, String::as_str);
                let component_address = if component == 0 {
                    aligned_address.clone()
                } else {
                    format!("{aligned_address} + {}u", component * 4)
                };
                let source = format!("v[{}]", vector_data + component);
                let value = if high {
                    format!("{source} >> 16")
                } else {
                    source
                };
                self.line(
                    3,
                    &format!(
                        "if (exec && !{descriptor_oob} && remu_range_valid({component_address}, {width}u, {binding_length})) remu_store_u{bits}(b{binding_index}, {component_address}, {value});",
                        bits = width * 8
                    ),
                );
            }
            return Ok(());
        }

        let atomic_address = aligned_address;
        let old = self.temp("uint", "0u");
        let source = format!("v[{vector_data}]");
        let operation = match name
            .strip_prefix("BufferAtomic")
            .or_else(|| name.strip_prefix("GlobalAtomic"))
        {
            Some("Swap") => {
                format!("remu_atomic_swap(b{binding_index}, {atomic_address}, {source})")
            }
            Some("Cmpswap") => format!(
                "remu_atomic_cmpswap(b{binding_index}, {atomic_address}, {source}, v[{}])",
                vector_data + 1
            ),
            Some("Add") => format!("remu_atomic_add(b{binding_index}, {atomic_address}, {source})"),
            Some("Sub") => format!("remu_atomic_sub(b{binding_index}, {atomic_address}, {source})"),
            Some("Smin") => {
                format!("remu_atomic_smin(b{binding_index}, {atomic_address}, {source})")
            }
            Some("Umin") => {
                format!("remu_atomic_umin(b{binding_index}, {atomic_address}, {source})")
            }
            Some("Smax") => {
                format!("remu_atomic_smax(b{binding_index}, {atomic_address}, {source})")
            }
            Some("Umax" | "UMax") => {
                format!("remu_atomic_umax(b{binding_index}, {atomic_address}, {source})")
            }
            Some("And") => format!("remu_atomic_and(b{binding_index}, {atomic_address}, {source})"),
            Some("Or") => format!("remu_atomic_or(b{binding_index}, {atomic_address}, {source})"),
            Some("Xor") => format!("remu_atomic_xor(b{binding_index}, {atomic_address}, {source})"),
            Some("Inc") => format!("remu_atomic_inc(b{binding_index}, {atomic_address}, {source})"),
            Some("Dec") => format!("remu_atomic_dec(b{binding_index}, {atomic_address}, {source})"),
            _ => return Err(format!("unsupported atomic opcode {name}")),
        };
        self.line(
            3,
            &format!(
                "if (exec && !{descriptor_oob} && remu_range_valid({atomic_address}, 4u, {binding_length})) {old} = {operation};"
            ),
        );
        if glc {
            self.store_vector(vector_destination, &old);
        }
        Ok(())
    }

    #[allow(clippy::too_many_lines)]
    fn emit_format_buffer_memory(
        &mut self,
        instruction: &Instruction,
        control: &remu_shader::BufferMemoryControl,
        binding_index: usize,
        binding_length: &str,
        address: &str,
        descriptor_oob: &str,
    ) -> Result<(), String> {
        if control.dword_count == 0 || control.dword_count > 4 {
            return Err(format!(
                "format load has invalid component count {}",
                control.dword_count
            ));
        }
        let descriptor_word = self.temp("uint", &format!("s[{}]", control.scalar_resource + 3));
        let unified_format = control
            .instruction_format
            .map_or_else(|| format!("({descriptor_word} >> 12) & 0x7Fu"), format_u32);
        let table_entry = self.temp("uint", &format!("remu_gfx10_formats[{unified_format}]"));
        let data_format = self.temp("uint", &format!("{table_entry} & 0xFFu"));
        let number_format = self.temp("uint", &format!("({table_entry} >> 8) & 0xFFu"));
        let element_size = self.temp("uint", &format!("remu_format_size({data_format})"));
        let oob = self.temp(
            "bool",
            &format_buffer_element_oob_expression(
                descriptor_oob,
                address,
                &element_size,
                binding_length,
            ),
        );
        if instruction.opcode.contains("Store") {
            return self.emit_format_buffer_store(
                instruction,
                control,
                binding_index,
                address,
                &descriptor_word,
                &data_format,
                &number_format,
                &oob,
            );
        }
        let mut canonical = Vec::with_capacity(4);
        for component in 0..4 {
            let byte_offset = self.temp("uint", "0u");
            let bit_offset = self.temp("uint", "0u");
            let bits = self.temp("uint", "0u");
            self.line(
                3,
                &format!(
                    "remu_format_layout({data_format}, {component}u, {byte_offset}, {bit_offset}, {bits});"
                ),
            );
            let raw = self.temp(
                "uint",
                &format!(
                    "{oob} || {bits} == 0u ? 0u : remu_load_format_bits(b{binding_index}, {address}, {byte_offset}, {bit_offset}, {bits})"
                ),
            );
            let missing = if component == 3 {
                format!("remu_format_one({number_format})")
            } else {
                "0u".to_owned()
            };
            canonical.push(self.temp(
                "uint",
                &format!(
                    "{bits} == 0u ? {missing} : remu_format_convert({raw}, {bits}, {number_format}, {data_format})"
                ),
            ));
        }
        let component_count = format_component_count(
            instruction
                .opcode
                .known_name()
                .ok_or_else(|| "raw format opcode".to_owned())?,
        );
        let mut selected = Vec::with_capacity(component_count as usize);
        for destination in 0..component_count {
            let selector = if control.instruction_format.is_some() {
                format_u32(destination + 4)
            } else {
                self.temp(
                    "uint",
                    &format!("({descriptor_word} >> {}u) & 7u", destination * 3),
                )
            };
            selected.push(self.temp(
                "uint",
                &format!(
                    "{selector} == 1u ? remu_format_one({number_format}) : {oob} ? 0u : {selector} == 4u ? {} : {selector} == 5u ? {} : {selector} == 6u ? {} : {selector} == 7u ? {} : 0u",
                    canonical[0], canonical[1], canonical[2], canonical[3]
                ),
            ));
        }
        let d16 = instruction.opcode.contains("FormatD16");
        let d16_hi = instruction.opcode.contains("FormatD16Hi");
        if control.lds {
            if !self.is_threadgroup_stage() || component_count != 1 || d16 || control.tfe {
                return Err(
                    "formatted memory-to-LDS requires a non-D16 FORMAT_X compute load".to_owned(),
                );
            }
            let lds_index = self.temp("uint", "((s[124] & 0xFFFFu) >> 2) + remu_lane");
            self.line(
                3,
                &format!(
                    "if (exec && {lds_index} < {}u) atomic_store_explicit(remu_lds + {lds_index}, {}, memory_order_relaxed);",
                    self.options.lds_size_bytes / 4,
                    selected[0]
                ),
            );
        } else if d16_hi {
            let converted = self.temp(
                "uint",
                &format!("remu_format_to_d16({}, {number_format})", selected[0]),
            );
            self.store_vector(
                control.vector_data,
                &format!(
                    "(v[{}] & 0xFFFFu) | ({converted} << 16)",
                    control.vector_data
                ),
            );
        } else if d16 {
            for register in 0..control.dword_count {
                let low_index = register * 2;
                let low = self.temp(
                    "uint",
                    &format!(
                        "remu_format_to_d16({}, {number_format})",
                        selected[low_index as usize]
                    ),
                );
                let expression = if low_index + 1 < component_count {
                    let high = self.temp(
                        "uint",
                        &format!(
                            "remu_format_to_d16({}, {number_format})",
                            selected[(low_index + 1) as usize]
                        ),
                    );
                    format!("{low} | ({high} << 16)")
                } else {
                    format!(
                        "(v[{}] & 0xFFFF0000u) | {low}",
                        control.vector_data + register
                    )
                };
                self.store_vector(control.vector_data + register, &expression);
            }
        } else {
            for (destination, value) in selected.iter().enumerate() {
                self.store_vector(
                    control.vector_data
                        + u32::try_from(destination)
                            .map_err(|_| "format destination overflow".to_owned())?,
                    value,
                );
            }
        }
        if control.tfe {
            self.store_vector(control.vector_data + control.dword_count, "0u");
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn emit_format_buffer_store(
        &mut self,
        instruction: &Instruction,
        control: &remu_shader::BufferMemoryControl,
        binding_index: usize,
        address: &str,
        descriptor_word: &str,
        data_format: &str,
        number_format: &str,
        oob: &str,
    ) -> Result<(), String> {
        let name = instruction
            .opcode
            .known_name()
            .ok_or_else(|| "raw format store opcode".to_owned())?;
        let component_count = format_component_count(name);
        let d16 = name.contains("FormatD16");
        let d16_hi = name.contains("FormatD16Hi");
        for memory_component in 0..4_u32 {
            let byte_offset = self.temp("uint", "0u");
            let bit_offset = self.temp("uint", "0u");
            let bits = self.temp("uint", "0u");
            self.line(
                3,
                &format!(
                    "remu_format_layout({data_format}, {memory_component}u, {byte_offset}, {bit_offset}, {bits});"
                ),
            );
            let enabled = self.temp("bool", "false");
            let canonical = self.temp("uint", "0u");
            if control.instruction_format.is_some() {
                if memory_component < component_count {
                    let register = control.vector_data + memory_component / if d16 { 2 } else { 1 };
                    let source = if d16_hi {
                        format!("v[{register}] >> 16")
                    } else if d16 {
                        format!("v[{register}] >> {}u", (memory_component & 1) * 16)
                    } else {
                        format!("v[{register}]")
                    };
                    let expanded = if d16 {
                        format!("remu_format_from_d16({source} & 0xFFFFu, {number_format})")
                    } else {
                        source
                    };
                    self.line(3, &format!("{enabled} = true; {canonical} = {expanded};"));
                }
            } else {
                for source_component in 0..component_count {
                    let selector = self.temp(
                        "uint",
                        &format!("({descriptor_word} >> {}u) & 7u", source_component * 3),
                    );
                    let register = control.vector_data + source_component / if d16 { 2 } else { 1 };
                    let source = if d16_hi {
                        format!("v[{register}] >> 16")
                    } else if d16 {
                        format!("v[{register}] >> {}u", (source_component & 1) * 16)
                    } else {
                        format!("v[{register}]")
                    };
                    let expanded = if d16 {
                        format!("remu_format_from_d16({source} & 0xFFFFu, {number_format})")
                    } else {
                        source
                    };
                    self.line(
                        3,
                        &format!(
                            "if ({selector} == {}u) {{ {enabled} = true; {canonical} = {expanded}; }}",
                            memory_component + 4
                        ),
                    );
                }
            }
            let packed = self.temp(
                "uint",
                &format!("remu_format_pack({canonical}, {bits}, {number_format}, {data_format})"),
            );
            self.line(
                3,
                &format!(
                    "if (exec && !{oob} && {enabled} && {bits} != 0u) remu_store_format_bits(b{binding_index}, {address}, {byte_offset}, {bit_offset}, {bits}, {packed});"
                ),
            );
        }
        Ok(())
    }
}
