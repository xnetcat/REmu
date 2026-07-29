// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

//! VOP1/VOP2/VOP3 lowering: the vector ALU opcode dispatch and the scalar
//! expression each unpacked, converted or arithmetic opcode produces.

use remu_shader::{Instruction, InstructionControl, OperandKind};

use crate::Compiler;
use crate::compiler::scalar::destination_scalar;
use crate::compiler::vector::{destination_vector, is_supported_dpp_control};

impl Compiler<'_> {
    #[allow(clippy::too_many_lines)]
    pub(crate) fn emit_vector_alu(&mut self, instruction: &Instruction) -> Result<(), String> {
        if let Some(InstructionControl::Sdwa(control)) = instruction.control.as_ref()
            && (control.source0_select == 7
                || control.source1_select == 7
                || control.destination_select == 7
                || control.destination_unused == 3)
        {
            return Err("reserved SDWA selector or destination-unused mode".to_owned());
        }
        if let Some(InstructionControl::Dpp(control)) = instruction.control
            && !is_supported_dpp_control(control.control)
        {
            return Err(format!(
                "unsupported DPP16 control 0x{:03X}",
                control.control
            ));
        }
        let name = instruction
            .opcode
            .known_name()
            .ok_or_else(|| "raw vector opcode".to_owned())?;
        if matches!(instruction.control, Some(InstructionControl::Vop3p(_))) {
            return self.emit_vop3p(instruction, name);
        }
        if name.starts_with("VCmp") {
            return self.emit_vector_compare(instruction, name);
        }
        if matches!(
            name,
            "VAddcU32"
                | "VSubbU32"
                | "VSubbrevU32"
                | "VAddCoU32"
                | "VSubCoU32"
                | "VSubrevCoU32"
                | "VAddCoCiU32"
        ) {
            return self.emit_vector_carry(instruction, name);
        }
        if matches!(name, "VMovreldB32" | "VMovrelsB32" | "VMovrelsdB32") {
            return self.emit_relative_vector_move(instruction, name);
        }
        if name == "VReadlaneB32" {
            let destination = destination_scalar(instruction)?;
            let value = self.raw_source(instruction, 0)?;
            let lane = self.raw_source(instruction, 1)?;
            let selected = if self.split_wave64() {
                self.line(
                    3,
                    &format!("remu_readlane_values[remu_thread_index] = {value};"),
                );
                self.line(3, "threadgroup_barrier(mem_flags::mem_threadgroup);");
                let selected_lane = self.temp("uint", &format!("({lane}) & 63u"));
                let wave_base = self.temp("uint", "remu_wave_index << 6u");
                let selected = self.temp(
                    "uint",
                    &format!("remu_readlane_values[{wave_base} + {selected_lane}]"),
                );
                // No lane may reuse its slot for a later V_READLANE until both SIMD32 halves
                // have consumed the current value.
                self.line(3, "threadgroup_barrier(mem_flags::mem_threadgroup);");
                selected
            } else {
                self.temp(
                    "uint",
                    &format!("simd_shuffle({value}, (ushort)({lane} & 31u))"),
                )
            };
            // V_READLANE is a wave operation and is architecturally independent of EXEC.
            self.store_scalar(destination, &selected);
            return Ok(());
        }
        if name == "VReadfirstlaneB32" {
            let destination = destination_scalar(instruction)?;
            let value = self.raw_source(instruction, 0)?;
            // The lane read is the lowest one EXEC has enabled, which is not necessarily lane
            // zero and is not the same as the SIMD group's own notion of an active thread. A
            // wave with nothing enabled still has to produce a value, and the hardware reads
            // lane zero in that case, which is what the mask of zero falls through to here.
            let active = self.temp("uint", "(uint)(uint64_t)simd_ballot(exec)");
            let lane = self.temp("uint", &format!("{active} == 0u ? 0u : ctz({active})"));
            let mut selected = self.temp("uint", &format!("simd_shuffle({value}, (ushort){lane})"));
            if self.split_wave64() {
                let slot = self.temp("uint", "(remu_wave_index << 1u) | remu_wave_half");
                self.line(
                    3,
                    &format!(
                        "if (remu_lane == 0u) {{ remu_readfirstlane_values[{slot}] = {selected}; remu_readfirstlane_active[{slot}] = {active} != 0u; }}"
                    ),
                );
                self.line(3, "threadgroup_barrier(mem_flags::mem_threadgroup);");
                let low_slot = self.temp("uint", "remu_wave_index << 1u");
                selected = self.temp(
                    "uint",
                    &format!(
                        "remu_readfirstlane_active[{low_slot}] != 0u ? remu_readfirstlane_values[{low_slot}] : (remu_readfirstlane_active[{low_slot} + 1u] != 0u ? remu_readfirstlane_values[{low_slot} + 1u] : remu_readfirstlane_values[{low_slot}])"
                    ),
                );
                self.line(3, "threadgroup_barrier(mem_flags::mem_threadgroup);");
            }
            self.store_scalar(destination, &selected);
            return Ok(());
        }
        if matches!(name, "VPermlane16B32" | "VPermlanex16B32") {
            return self.emit_permlane16(instruction, name == "VPermlanex16B32");
        }
        if name == "VMadU64U32" {
            return self.emit_mad_u64_u32(instruction);
        }
        if matches!(name, "VLshlrevB64" | "VLshrrevB64") {
            return self.emit_vector_shift64(instruction, name == "VLshlrevB64");
        }
        if name == "VWritelaneB32" {
            let destination = destination_vector(instruction)?;
            let value = self.raw_source(instruction, 0)?;
            let lane = self.raw_source(instruction, 1)?;
            // V_WRITELANE also ignores EXEC: exactly the selected physical lane is replaced and
            // all other lanes retain the old VGPR value.
            let architectural_lane = if self.split_wave64() {
                "remu_wave_lane"
            } else {
                "remu_lane"
            };
            let lane_mask = if self.options.wave_lane_count == 64 {
                63
            } else {
                31
            };
            self.line(
                3,
                &format!(
                    "if ({architectural_lane} == (({lane}) & {lane_mask}u)) v[{destination}] = {value};"
                ),
            );
            return Ok(());
        }
        let destination = destination_vector(instruction)?;
        let expression = match name {
            "VCndmaskB32" => {
                let mask = if instruction.sources.len() > 2 {
                    self.mask_source(instruction.sources[2])?
                } else {
                    "vcc".to_owned()
                };
                if Self::has_float_modifiers(instruction) {
                    // V_CNDMASK_B32 is bitwise only without modifiers. RDNA2 explicitly permits
                    // VOP3/SDWA floating-point ABS/NEG (and result OMOD/CLAMP) so shaders can
                    // conditionally change a float's sign without a separate VALU instruction.
                    // Treating those sources as raw uints silently drops the modifier.
                    let false_value = self.float_source(instruction, 0)?;
                    let true_value = self.float_source(instruction, 1)?;
                    self.float_result(
                        instruction,
                        &format!("({mask}) ? ({true_value}) : ({false_value})"),
                    )
                } else {
                    let false_value = self.raw_source(instruction, 0)?;
                    let true_value = self.raw_source(instruction, 1)?;
                    format!("({mask}) ? ({true_value}) : ({false_value})")
                }
            }
            "VMovB32" => {
                // RDNA2 defines V_MOV_B32 as an untyped move only when no floating-point
                // modifiers are present. ABS/NEG (and SDWA CLAMP/OMOD) explicitly reinterpret
                // S0 as f32. Applying SDWA NEG to the raw uint would compute two's-complement
                // negation instead of toggling the floating-point sign, e.g. 1.0 became -4.0.
                if Self::has_float_modifiers(instruction) {
                    let source = self.float_source(instruction, 0)?;
                    self.float_result(instruction, &source)
                } else {
                    self.raw_source(instruction, 0)?
                }
            }
            "VBfrevB32" => format!("reverse_bits({})", self.raw_source(instruction, 0)?),
            "VBcntU32B32" => format!(
                "popcount({}) + {}",
                self.raw_source(instruction, 0)?,
                self.raw_source(instruction, 1)?
            ),
            "VMbcntLoU32B32" | "VMbcntHiU32B32" => {
                let source = self.raw_source(instruction, 0)?;
                let addend = self.raw_source(instruction, 1)?;
                let below = self.temp("uint", "remu_lane == 0u ? 0u : ((1u << remu_lane) - 1u)");
                let mask = match (name, self.split_wave64()) {
                    ("VMbcntLoU32B32", true) => {
                        format!("remu_wave_half == 0u ? {below} : 0xFFFFFFFFu")
                    }
                    ("VMbcntHiU32B32", true) => {
                        format!("remu_wave_half == 0u ? 0u : {below}")
                    }
                    ("VMbcntLoU32B32", false) => below,
                    ("VMbcntHiU32B32", false) => "0u".to_owned(),
                    _ => unreachable!("matched masked bit-count operation"),
                };
                format!("popcount(({source}) & ({mask})) + {addend}")
            }
            "VXor3B32" => format!(
                "{} ^ {} ^ {}",
                self.raw_source(instruction, 0)?,
                self.raw_source(instruction, 1)?,
                self.raw_source(instruction, 2)?
            ),
            "VAddF32" => self.float_result(
                instruction,
                &format!(
                    "{} + {}",
                    self.float_source(instruction, 0)?,
                    self.float_source(instruction, 1)?
                ),
            ),
            "VSubF32" => self.float_result(
                instruction,
                &format!(
                    "{} - {}",
                    self.float_source(instruction, 0)?,
                    self.float_source(instruction, 1)?
                ),
            ),
            "VSubrevF32" => self.float_result(
                instruction,
                &format!(
                    "{} - {}",
                    self.float_source(instruction, 1)?,
                    self.float_source(instruction, 0)?
                ),
            ),
            "VMulF32" => self.float_result(
                instruction,
                &format!(
                    "{} * {}",
                    self.float_source(instruction, 0)?,
                    self.float_source(instruction, 1)?
                ),
            ),
            "VAddF16" | "VSubF16" | "VSubrevF16" | "VMulF16" | "VMaxF16" | "VMinF16" => {
                let first = self.half_float_source(instruction, 0)?;
                let second = self.half_float_source(instruction, 1)?;
                let expression = match name {
                    "VAddF16" => format!("{first} + {second}"),
                    "VSubF16" => format!("{first} - {second}"),
                    "VSubrevF16" => format!("{second} - {first}"),
                    "VMulF16" => format!("{first} * {second}"),
                    "VMaxF16" => format!("fmax({first}, {second})"),
                    "VMinF16" => format!("fmin({first}, {second})"),
                    _ => unreachable!("matched observed half-float arithmetic"),
                };
                self.f16_result(instruction, destination, &expression)
            }
            "VFmacF16" => {
                let first = self.half_float_source(instruction, 0)?;
                let second = self.half_float_source(instruction, 1)?;
                let previous_raw = self.temp("uint", &format!("v[{destination}] & 0xFFFFu"));
                let previous = self.temp(
                    "float",
                    &format!("as_type<float>(remu_f16_to_f32_bits({previous_raw}))"),
                );
                self.f16_result(
                    instruction,
                    destination,
                    &format!("fma({first}, {second}, {previous})"),
                )
            }
            "VFmaF16" | "VFmaMkF16" | "VFmaAkF16" => {
                let first = self.half_float_source(instruction, 0)?;
                let second = self.half_float_source(instruction, 1)?;
                let third = self.half_float_source(instruction, 2)?;
                self.f16_result(
                    instruction,
                    destination,
                    &format!("fma({first}, {second}, {third})"),
                )
            }
            "VLdexpF16" => {
                let first = self.half_float_source(instruction, 0)?;
                let second_raw = self.temp("uint", &self.raw_source(instruction, 1)?);
                self.f16_result(
                    instruction,
                    destination,
                    &format!("ldexp({first}, (int)(short)({second_raw} & 0xFFFFu))"),
                )
            }
            "VMinF32" => self.float_result(
                instruction,
                &format!(
                    "fmin({}, {})",
                    self.float_source(instruction, 0)?,
                    self.float_source(instruction, 1)?
                ),
            ),
            "VMaxF32" => self.float_result(
                instruction,
                &format!(
                    "fmax({}, {})",
                    self.float_source(instruction, 0)?,
                    self.float_source(instruction, 1)?
                ),
            ),
            "VMin3F32" => self.float_result(
                instruction,
                &format!(
                    "fmin(fmin({}, {}), {})",
                    self.float_source(instruction, 0)?,
                    self.float_source(instruction, 1)?,
                    self.float_source(instruction, 2)?
                ),
            ),
            "VMax3F32" => self.float_result(
                instruction,
                &format!(
                    "fmax(fmax({}, {}), {})",
                    self.float_source(instruction, 0)?,
                    self.float_source(instruction, 1)?,
                    self.float_source(instruction, 2)?
                ),
            ),
            "VMed3F32" => {
                let first = self.float_source(instruction, 0)?;
                let second = self.float_source(instruction, 1)?;
                let third = self.float_source(instruction, 2)?;
                self.float_result(
                    instruction,
                    &format!(
                        "fmax(fmin({first}, {second}), fmin(fmax({first}, {second}), {third}))"
                    ),
                )
            }
            "VMin3I32" => format!(
                "as_type<uint>(min(min(as_type<int>({}), as_type<int>({})), as_type<int>({})))",
                self.raw_source(instruction, 0)?,
                self.raw_source(instruction, 1)?,
                self.raw_source(instruction, 2)?
            ),
            "VMax3I32" => format!(
                "as_type<uint>(max(max(as_type<int>({}), as_type<int>({})), as_type<int>({})))",
                self.raw_source(instruction, 0)?,
                self.raw_source(instruction, 1)?,
                self.raw_source(instruction, 2)?
            ),
            "VMed3I32" => {
                let first = format!("as_type<int>({})", self.raw_source(instruction, 0)?);
                let second = format!("as_type<int>({})", self.raw_source(instruction, 1)?);
                let third = format!("as_type<int>({})", self.raw_source(instruction, 2)?);
                format!(
                    "as_type<uint>(max(min({first}, {second}), min(max({first}, {second}), {third})))"
                )
            }
            "VMin3U32" => format!(
                "min(min({}, {}), {})",
                self.raw_source(instruction, 0)?,
                self.raw_source(instruction, 1)?,
                self.raw_source(instruction, 2)?
            ),
            "VMax3U32" => format!(
                "max(max({}, {}), {})",
                self.raw_source(instruction, 0)?,
                self.raw_source(instruction, 1)?,
                self.raw_source(instruction, 2)?
            ),
            "VMed3U32" => {
                let first = self.raw_source(instruction, 0)?;
                let second = self.raw_source(instruction, 1)?;
                let third = self.raw_source(instruction, 2)?;
                format!("max(min({first}, {second}), min(max({first}, {second}), {third}))")
            }
            "VSadU32" => {
                let first = self.raw_source(instruction, 0)?;
                let second = self.raw_source(instruction, 1)?;
                let addend = self.raw_source(instruction, 2)?;
                format!("(max({first}, {second}) - min({first}, {second}) + {addend})")
            }
            "VFmacF32" | "VMacF32" => self.float_result(
                instruction,
                &format!(
                    "fma({}, {}, as_type<float>(v[{destination}]))",
                    self.float_source(instruction, 0)?,
                    self.float_source(instruction, 1)?
                ),
            ),
            "VFloorF32" => self.float_result(
                instruction,
                &format!("floor({})", self.float_source(instruction, 0)?),
            ),
            "VCeilF32" => self.float_result(
                instruction,
                &format!("ceil({})", self.float_source(instruction, 0)?),
            ),
            "VTruncF32" => self.float_result(
                instruction,
                &format!("trunc({})", self.float_source(instruction, 0)?),
            ),
            "VRndneF32" => self.float_result(
                instruction,
                &format!("rint({})", self.float_source(instruction, 0)?),
            ),
            "VFractF32" => self.float_result(
                instruction,
                &format!("fract({})", self.float_source(instruction, 0)?),
            ),
            "VSqrtF32" => self.float_result(
                instruction,
                &format!("sqrt({})", self.float_source(instruction, 0)?),
            ),
            "VRsqF32" => self.float_result(
                instruction,
                &format!("rsqrt({})", self.float_source(instruction, 0)?),
            ),
            "VRsqF16" => {
                let result = self.temp(
                    "uint",
                    &format!(
                        "as_type<uint>(rsqrt({}))",
                        self.half_float_source(instruction, 0)?
                    ),
                );
                format!("remu_f32_to_f16_bits({result})")
            }
            "VRcpF16" => {
                let result = self.temp(
                    "uint",
                    &format!(
                        "as_type<uint>(1.0f / {})",
                        self.half_float_source(instruction, 0)?
                    ),
                );
                format!("remu_f32_to_f16_bits({result})")
            }
            "VSqrtF16" => {
                let result = self.temp(
                    "uint",
                    &format!(
                        "as_type<uint>(sqrt({}))",
                        self.half_float_source(instruction, 0)?
                    ),
                );
                format!("remu_f32_to_f16_bits({result})")
            }
            "VLogF16" => {
                let result = self.temp(
                    "uint",
                    &format!(
                        "as_type<uint>(log2({}))",
                        self.half_float_source(instruction, 0)?
                    ),
                );
                format!("remu_f32_to_f16_bits({result})")
            }
            "VExpF16" => {
                let result = self.temp(
                    "uint",
                    &format!(
                        "as_type<uint>(exp2({}))",
                        self.half_float_source(instruction, 0)?
                    ),
                );
                format!("remu_f32_to_f16_bits({result})")
            }
            "VRcpF32" | "VRcpIflagF32" => self.float_result(
                instruction,
                &format!("1.0f / {}", self.float_source(instruction, 0)?),
            ),
            "VLogF32" => self.float_result(
                instruction,
                &format!("log2({})", self.float_source(instruction, 0)?),
            ),
            "VExpF32" => self.float_result(
                instruction,
                &format!("exp2({})", self.float_source(instruction, 0)?),
            ),
            "VLdexpF32" => self.float_result(
                instruction,
                &format!(
                    "ldexp({}, as_type<int>({}))",
                    self.float_source(instruction, 0)?,
                    self.raw_source(instruction, 1)?
                ),
            ),
            "VDivFixupF32" => self.float_result(
                instruction,
                &format!(
                    "as_type<float>(remu_div_fixup_f32_bits(as_type<uint>({}), as_type<uint>({}), as_type<uint>({})))",
                    self.float_source(instruction, 0)?,
                    self.float_source(instruction, 1)?,
                    self.float_source(instruction, 2)?
                ),
            ),
            "VSinF32" => self.float_result(
                instruction,
                &format!(
                    "sin({} * 6.28318530717958647692f)",
                    self.float_source(instruction, 0)?
                ),
            ),
            "VCosF32" => self.float_result(
                instruction,
                &format!(
                    "cos({} * 6.28318530717958647692f)",
                    self.float_source(instruction, 0)?
                ),
            ),
            "VAndB32" => format!(
                "({} & {})",
                self.raw_source(instruction, 0)?,
                self.raw_source(instruction, 1)?
            ),
            "VOrB32" => format!(
                "({} | {})",
                self.raw_source(instruction, 0)?,
                self.raw_source(instruction, 1)?
            ),
            "VXorB32" => format!(
                "({} ^ {})",
                self.raw_source(instruction, 0)?,
                self.raw_source(instruction, 1)?
            ),
            "VXnorB32" => format!(
                "~({} ^ {})",
                self.raw_source(instruction, 0)?,
                self.raw_source(instruction, 1)?
            ),
            "VNotB32" => format!("~{}", self.raw_source(instruction, 0)?),
            "VAddI32" => format!(
                "({} + {})",
                self.raw_source(instruction, 0)?,
                self.raw_source(instruction, 1)?
            ),
            "VSubI32" => format!(
                "({} - {})",
                self.raw_source(instruction, 0)?,
                self.raw_source(instruction, 1)?
            ),
            "VSubrevI32" => format!(
                "({} - {})",
                self.raw_source(instruction, 1)?,
                self.raw_source(instruction, 0)?
            ),
            "VMinI32" => format!(
                "as_type<uint>(min(as_type<int>({}), as_type<int>({})))",
                self.raw_source(instruction, 0)?,
                self.raw_source(instruction, 1)?
            ),
            "VMaxI32" => format!(
                "as_type<uint>(max(as_type<int>({}), as_type<int>({})))",
                self.raw_source(instruction, 0)?,
                self.raw_source(instruction, 1)?
            ),
            "VLshrB32" => format!(
                "({} >> ({} & 31u))",
                self.raw_source(instruction, 0)?,
                self.raw_source(instruction, 1)?
            ),
            "VLshrrevB32" => format!(
                "({} >> ({} & 31u))",
                self.raw_source(instruction, 1)?,
                self.raw_source(instruction, 0)?
            ),
            "VAshrI32" => format!(
                "as_type<uint>(as_type<int>({}) >> ({} & 31u))",
                self.raw_source(instruction, 0)?,
                self.raw_source(instruction, 1)?
            ),
            "VAshrrevI32" => format!(
                "as_type<uint>(as_type<int>({}) >> ({} & 31u))",
                self.raw_source(instruction, 1)?,
                self.raw_source(instruction, 0)?
            ),
            "VLshlB32" => format!(
                "({} << ({} & 31u))",
                self.raw_source(instruction, 0)?,
                self.raw_source(instruction, 1)?
            ),
            "VLshlrevB32" => format!(
                "({} << ({} & 31u))",
                self.raw_source(instruction, 1)?,
                self.raw_source(instruction, 0)?
            ),
            "VLshlAddU32" => format!(
                "(({} << ({} & 31u)) + {})",
                self.raw_source(instruction, 0)?,
                self.raw_source(instruction, 1)?,
                self.raw_source(instruction, 2)?
            ),
            "VAddLshlU32" => format!(
                "(({} + {}) << ({} & 31u))",
                self.raw_source(instruction, 0)?,
                self.raw_source(instruction, 1)?,
                self.raw_source(instruction, 2)?
            ),
            "VBfeU32" => {
                let value = self.temp("uint", &self.raw_source(instruction, 0)?);
                let offset = self.temp(
                    "uint",
                    &format!("{} & 31u", self.raw_source(instruction, 1)?),
                );
                let width = self.temp(
                    "uint",
                    &format!(
                        "min({} & 31u, 32u - {offset})",
                        self.raw_source(instruction, 2)?
                    ),
                );
                format!("{width} == 0u ? 0u : extract_bits({value}, {offset}, {width})")
            }
            "VBfeI32" => {
                let value = self.temp(
                    "int",
                    &format!("as_type<int>({})", self.raw_source(instruction, 0)?),
                );
                let offset = self.temp(
                    "uint",
                    &format!("{} & 31u", self.raw_source(instruction, 1)?),
                );
                let width = self.temp(
                    "uint",
                    &format!("{} & 31u", self.raw_source(instruction, 2)?),
                );
                let mask = self.temp(
                    "uint",
                    &format!("{width} == 0u ? 0u : ((1u << {width}) - 1u)"),
                );
                format!("(as_type<uint>({value} >> {offset}) & {mask})")
            }
            "VBfiB32" => {
                let mask = self.temp("uint", &self.raw_source(instruction, 0)?);
                let insert = self.temp("uint", &self.raw_source(instruction, 1)?);
                let base = self.temp("uint", &self.raw_source(instruction, 2)?);
                format!("(({mask} & {insert}) | (~{mask} & {base}))")
            }
            // Three-operand integer forms fold a second ALU step into one instruction. The shift
            // amount is masked because the hardware uses only the low five bits.
            "VAdd3U32" => format!(
                "({} + {} + {})",
                self.raw_source(instruction, 0)?,
                self.raw_source(instruction, 1)?,
                self.raw_source(instruction, 2)?
            ),
            "VXadU32" => format!(
                "(({} ^ {}) + {})",
                self.raw_source(instruction, 0)?,
                self.raw_source(instruction, 1)?,
                self.raw_source(instruction, 2)?
            ),
            "VOr3U32" => format!(
                "({} | {} | {})",
                self.raw_source(instruction, 0)?,
                self.raw_source(instruction, 1)?,
                self.raw_source(instruction, 2)?
            ),
            "VAndOrB32" => format!(
                "(({} & {}) | {})",
                self.raw_source(instruction, 0)?,
                self.raw_source(instruction, 1)?,
                self.raw_source(instruction, 2)?
            ),
            "VLshlOrU32" => format!(
                "(({} << ({} & 31u)) | {})",
                self.raw_source(instruction, 0)?,
                self.raw_source(instruction, 1)?,
                self.raw_source(instruction, 2)?
            ),
            "VCvtF32I32" => self.float_result(
                instruction,
                &format!("(float)as_type<int>({})", self.raw_source(instruction, 0)?),
            ),
            "VCvtF32U32" => self.float_result(
                instruction,
                &format!("(float){}", self.raw_source(instruction, 0)?),
            ),
            "VCvtF16F32" => {
                let source = self.float_source(instruction, 0)?;
                self.f16_result(instruction, destination, &source)
            }
            "VCvtF32F16" => {
                let source = self.half_float_source(instruction, 0)?;
                self.float_result(instruction, &source)
            }
            "VCvtU16F16" => {
                let source = self.temp("float", &self.half_float_source(instruction, 0)?);
                format!(
                    "(isnan({source}) || {source} <= 0.0f) ? 0u : ({source} >= 65535.0f ? 0xFFFFu : (uint)trunc({source}))"
                )
            }
            "VMinU32" => format!(
                "min({}, {})",
                self.raw_source(instruction, 0)?,
                self.raw_source(instruction, 1)?
            ),
            "VMaxU32" => format!(
                "max({}, {})",
                self.raw_source(instruction, 0)?,
                self.raw_source(instruction, 1)?
            ),
            "VMulLoU32" => format!(
                "({} * {})",
                self.raw_source(instruction, 0)?,
                self.raw_source(instruction, 1)?
            ),
            "VMulHiU32" => format!(
                "(uint)(((ulong){} * (ulong){}) >> 32)",
                self.raw_source(instruction, 0)?,
                self.raw_source(instruction, 1)?
            ),
            "VMulHiI32" => format!(
                "(uint)(int)(((long)(int){} * (long)(int){}) >> 32)",
                self.raw_source(instruction, 0)?,
                self.raw_source(instruction, 1)?
            ),
            "VMulU32U24" => format!(
                "(({} & 0xFFFFFFu) * ({} & 0xFFFFFFu))",
                self.raw_source(instruction, 0)?,
                self.raw_source(instruction, 1)?
            ),
            "VMadU32U24" => format!(
                "(({} & 0xFFFFFFu) * ({} & 0xFFFFFFu) + {})",
                self.raw_source(instruction, 0)?,
                self.raw_source(instruction, 1)?,
                self.raw_source(instruction, 2)?
            ),
            "VMulHiU32U24" => format!(
                "(uint)(((ulong)({} & 0xFFFFFFu) * (ulong)({} & 0xFFFFFFu)) >> 32)",
                self.raw_source(instruction, 0)?,
                self.raw_source(instruction, 1)?
            ),
            "VCvtU32F32" => {
                let source = self.temp("float", &self.float_source(instruction, 0)?);
                format!(
                    "(isnan({source}) || {source} <= 0.0f) ? 0u : ({source} >= 4294967296.0f ? 0xFFFFFFFFu : (uint)trunc({source}))"
                )
            }
            "VCvtI32F32" => {
                let source = self.temp("float", &self.float_source(instruction, 0)?);
                format!(
                    "isnan({source}) ? 0u : ({source} <= -2147483648.0f ? 0x80000000u : ({source} >= 2147483648.0f ? 0x7FFFFFFFu : as_type<uint>((int)trunc({source}))))"
                )
            }
            "VCvtFlrI32F32" => {
                let source = self.temp("float", &self.float_source(instruction, 0)?);
                // The conversion is floor followed by the architectural saturating F32->I32
                // conversion. Guard both endpoints before the MSL cast: converting NaN,
                // infinity, or an out-of-range float to int is otherwise undefined.
                format!(
                    "isnan({source}) ? 0u : ({source} <= -2147483648.0f ? 0x80000000u : ({source} >= 2147483648.0f ? 0x7FFFFFFFu : as_type<uint>((int)floor({source}))))"
                )
            }
            "VCvtOffF32I4" => {
                let source = self.temp("uint", &self.raw_source(instruction, 0)?);
                self.float_result(
                    instruction,
                    &format!("(float)((int)({source} << 28) >> 28) * 0.0625f"),
                )
            }
            "VCubeidF32" | "VCubescF32" | "VCubetcF32" | "VCubemaF32" => {
                self.cube_result(instruction, name)?
            }
            "VCvtPkrtzF16F32" => {
                let first = self.temp(
                    "uint",
                    &format!("as_type<uint>({})", self.float_source(instruction, 0)?),
                );
                let second = self.temp(
                    "uint",
                    &format!("as_type<uint>({})", self.float_source(instruction, 1)?),
                );
                format!(
                    "(remu_f32_to_f16_rtz_bits({first}) | (remu_f32_to_f16_rtz_bits({second}) << 16))"
                )
            }
            "VCvtPknormU16F32" => {
                let first = self.temp("float", &self.float_source(instruction, 0)?);
                let second = self.temp("float", &self.float_source(instruction, 1)?);
                // RDNA2's UNORM conversion is round-to-nearest-even after saturation. Keep the
                // cast on the finite, in-range branch so Metal never observes an undefined
                // float-to-uint conversion. The usual DX10 NaN mode maps NaN to zero.
                let low = self.temp(
                    "uint",
                    &format!(
                        "(isnan({first}) || {first} <= 0.0f) ? 0u : ({first} >= 1.0f ? 0xFFFFu : (uint)rint({first} * 65535.0f))"
                    ),
                );
                let high = self.temp(
                    "uint",
                    &format!(
                        "(isnan({second}) || {second} <= 0.0f) ? 0u : ({second} >= 1.0f ? 0xFFFFu : (uint)rint({second} * 65535.0f))"
                    ),
                );
                format!("({low} | ({high} << 16))")
            }
            "VCvtPkU16U32" => format!(
                "(({} & 0xFFFFu) | (({} & 0xFFFFu) << 16))",
                self.raw_source(instruction, 0)?,
                self.raw_source(instruction, 1)?
            ),
            "VMadF32" | "VFmaF32" | "VMadAkF32" | "VMadMkF32" | "VFmaAkF32" | "VFmaMkF32" => self
                .float_result(
                    instruction,
                    &format!(
                        "fma({}, {}, {})",
                        self.float_source(instruction, 0)?,
                        self.float_source(instruction, 1)?,
                        self.float_source(instruction, 2)?
                    ),
                ),
            _ => return Err(format!("unsupported vector opcode {name}")),
        };
        let expression =
            if let Some(InstructionControl::Dpp(control)) = instruction.control.as_ref() {
                let result = self.temp("uint", &expression);
                let write_enabled = self.emit_dpp_write_enabled(*control)?;
                format!("({write_enabled}) ? {result} : v[{destination}]")
            } else if matches!(instruction.control, Some(InstructionControl::Dpp8(_))) {
                // Keep the shuffle converged across inactive guest lanes; only the architectural
                // destination write is EXEC-masked below.
                self.temp("uint", &expression)
            } else if let Some(InstructionControl::Sdwa(control)) = instruction.control.as_ref()
                && control.scalar_destination.is_none()
            {
                let result = self.temp("uint", &expression);
                self.apply_sdwa_destination(control, &result, &format!("v[{destination}]"))?
            } else {
                expression
            };
        self.store_vector(destination, &expression);
        Ok(())
    }

    fn emit_relative_vector_move(
        &mut self,
        instruction: &Instruction,
        name: &str,
    ) -> Result<(), String> {
        let destination = destination_vector(instruction)?;
        let source = instruction
            .sources
            .first()
            .copied()
            .ok_or_else(|| "missing relative-move source operand".to_owned())?;
        let relative_source = matches!(name, "VMovrelsB32" | "VMovrelsdB32");
        let relative_destination = matches!(name, "VMovreldB32" | "VMovrelsdB32");
        let value = if relative_source {
            if source.kind != OperandKind::VectorRegister {
                return Err(format!(
                    "{name} requires a vector-register source, found {source}"
                ));
            }
            self.relative_vector_source(source.value)?
        } else {
            self.raw_source(instruction, 0)?
        };
        if relative_destination {
            self.store_relative_vector(destination, &value);
        } else {
            self.store_vector(destination, &value);
        }
        Ok(())
    }

    fn relative_vector_source(&mut self, base: u32) -> Result<String, String> {
        if base >= crate::VECTOR_REGISTER_COUNT {
            return Err(format!(
                "relative vector source v{base} exceeds the vector register file"
            ));
        }
        // GFX10 returns VGPR0 when M0 selects beyond the stage's allocation. Expanding this into
        // fixed cases preserves that behavior and lets module scalarization keep independent
        // register locals instead of materializing a 256-word addressable array per Metal lane.
        let selected = self.temp("uint", "v[0]");
        self.line(3, "switch (s[124]) {");
        for register in base..self.options.allocated_vector_register_count {
            self.line(
                4,
                &format!(
                    "case {}u: {selected} = v[{register}]; break;",
                    register - base
                ),
            );
        }
        self.line(4, "default: break;");
        self.line(3, "}");
        Ok(selected)
    }

    fn store_relative_vector(&mut self, base: u32, value: &str) {
        for register in base..self.options.allocated_vector_register_count {
            self.line(
                3,
                &format!(
                    "if (exec && s[124] == {}u) v[{register}] = {value};",
                    register - base
                ),
            );
        }
    }
}
