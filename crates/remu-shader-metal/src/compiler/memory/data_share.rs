// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

//! LDS lowering: data share reads, writes, atomics, `DS_SWIZZLE` and the
//! lane-private `ADDTID` records the graphics stages spill through.

use std::fmt::Write as _;

use remu_shader::{DataShareControl, Instruction, InstructionControl};

use crate::compiler::vector::vector_operand;
use crate::{Compiler, MetalShaderStage, format_u32};

impl Compiler<'_> {
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
            "DsWrite2B32" => self.emit_ds_write_pair(instruction, control, false, 1),
            "DsWrite2St64B32" => self.emit_ds_write_pair(instruction, control, true, 1),
            "DsWrite2B64" => self.emit_ds_write_pair(instruction, control, false, 2),
            "DsWrite2St64B64" => self.emit_ds_write_pair(instruction, control, true, 2),
            "DsReadB32" => self.emit_ds_read(instruction, control, 1),
            "DsReadB64" => self.emit_ds_read(instruction, control, 2),
            "DsReadB96" => self.emit_ds_read(instruction, control, 3),
            "DsReadB128" => self.emit_ds_read(instruction, control, 4),
            "DsRead2B32" => self.emit_ds_read_pair(instruction, control, false, 1),
            "DsRead2St64B32" => self.emit_ds_read_pair(instruction, control, true, 1),
            "DsRead2B64" => self.emit_ds_read_pair(instruction, control, false, 2),
            "DsRead2St64B64" => self.emit_ds_read_pair(instruction, control, true, 2),
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
        dwords_per_element: usize,
    ) -> Result<(), String> {
        let required_sources = 1 + 2 * dwords_per_element;
        if instruction.sources.len() < required_sources {
            return Err(format!(
                "LDS write2 requires {} data dwords",
                2 * dwords_per_element
            ));
        }
        let base = self.temp("uint", &self.raw_source(instruction, 0)?);
        let element_bytes = u32::try_from(dwords_per_element)
            .map_err(|_| "LDS write2 element width overflows".to_owned())?
            * 4;
        let scale = if stride64 {
            element_bytes * 64
        } else {
            element_bytes
        };
        for (element, element_offset) in [control.offset0, control.offset1].into_iter().enumerate()
        {
            for component in 0..dwords_per_element {
                let source = 1 + element * dwords_per_element + component;
                let value = self.temp("uint", &self.raw_source(instruction, source)?);
                let component_offset = u32::try_from(component)
                    .map_err(|_| "LDS write2 component index overflows".to_owned())?
                    * 4;
                let address = self.ds_address(&base, element_offset * scale + component_offset);
                let index = self.temp("uint", &format!("{address} >> 2"));
                self.emit_ds_store_index(&index, &value);
            }
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
        dwords_per_element: usize,
    ) -> Result<(), String> {
        let required_destinations = 2 * dwords_per_element;
        if instruction.destinations.len() < required_destinations {
            return Err(format!(
                "LDS read2 requires {required_destinations} destination dwords"
            ));
        }
        let base = self.temp("uint", &self.raw_source(instruction, 0)?);
        let element_bytes = u32::try_from(dwords_per_element)
            .map_err(|_| "LDS read2 element width overflows".to_owned())?
            * 4;
        let scale = if stride64 {
            element_bytes * 64
        } else {
            element_bytes
        };
        let mut values = Vec::with_capacity(required_destinations);
        for element_offset in [control.offset0, control.offset1] {
            for component in 0..dwords_per_element {
                let component_offset = u32::try_from(component)
                    .map_err(|_| "LDS read2 component index overflows".to_owned())?
                    * 4;
                let address = self.ds_address(&base, element_offset * scale + component_offset);
                let index = self.temp("uint", &format!("{address} >> 2"));
                values.push(self.emit_ds_load_index(&index));
            }
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
