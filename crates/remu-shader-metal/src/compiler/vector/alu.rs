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
            let source_vgpr = instruction.sources.first().and_then(|source| {
                (source.kind == OperandKind::VectorRegister).then_some(source.value)
            });
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
            } else if source_vgpr
                .filter(|vgpr| self.graphics_lane_file_vgpr(*vgpr))
                .is_some()
            {
                return Err(
                    "pixel/vertex V_READLANE of a V_WRITELANE wave-constant table has no proven Metal lane occupancy; refusing the private lane-file peephole"
                        .into(),
                );
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
                let site_base = self.disjoint_readfirstlane_site_base(instruction.pc);
                let slot_expression = site_base.map_or_else(
                    || "(remu_wave_index << 1u) | remu_wave_half".to_owned(),
                    |base| format!("{base}u + ((remu_wave_index << 1u) | remu_wave_half)"),
                );
                let slot = self.temp("uint", &slot_expression);
                self.line(
                    3,
                    &format!(
                        "if (remu_lane == 0u) {{ remu_readfirstlane_values[{slot}] = {selected}; remu_readfirstlane_active[{slot}] = {active} != 0u; }}"
                    ),
                );
                self.line(3, "threadgroup_barrier(mem_flags::mem_threadgroup);");
                let low_slot_expression = site_base.map_or_else(
                    || "remu_wave_index << 1u".to_owned(),
                    |base| format!("{base}u + (remu_wave_index << 1u)"),
                );
                let low_slot = self.temp("uint", &low_slot_expression);
                selected = self.temp(
                    "uint",
                    &format!(
                        "remu_readfirstlane_active[{low_slot}] != 0u ? remu_readfirstlane_values[{low_slot}] : (remu_readfirstlane_active[{low_slot} + 1u] != 0u ? remu_readfirstlane_values[{low_slot} + 1u] : remu_readfirstlane_values[{low_slot}])"
                    ),
                );
                if site_base.is_none() {
                    self.line(3, "threadgroup_barrier(mem_flags::mem_threadgroup);");
                }
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
            // J3: the private graphics lane-file peephole (Poppy Slate SDF constants) is deleted.
            // Metal fragment/vertex simdgroups do not prove architectural lane occupancy for
            // V_WRITELANE tables, so this path refuses loudly rather than silently filling tofu
            // via simd_shuffle or inventing a title-specific private table.
            if !self.is_threadgroup_stage() {
                return Err(
                    "pixel/vertex V_WRITELANE wave-constant table has no proven Metal lane occupancy; refusing the private lane-file peephole"
                        .into(),
                );
            }
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
            let lane_index = format!("({lane}) & {lane_mask}u");
            self.line(
                3,
                &format!("if ({architectural_lane} == ({lane_index})) v[{destination}] = {value};"),
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
            "VMulLegacyF32" => {
                // The DX9 multiply: a zero operand forces a zero result, where IEEE would return
                // NaN for zero times infinity. Shaders reach for it exactly when they rely on that
                // difference, so lowering it as a plain multiply would turn a masked-off term into
                // a NaN that then poisons everything it is added to.
                let first_source = self.float_source(instruction, 0)?;
                let second_source = self.float_source(instruction, 1)?;
                let first = self.temp("float", &first_source);
                let second = self.temp("float", &second_source);
                self.float_result(
                    instruction,
                    &format!(
                        "({first} == 0.0f || {second} == 0.0f) ? 0.0f : {first} * {second}"
                    ),
                )
            }
            "VFfbhU32" | "VFfblB32" | "VFfbhI32" => {
                let raw = self.raw_source(instruction, 0)?;
                let source = self.temp("uint", &raw);
                // The signed scan looks for the highest bit that differs from the sign bit, which
                // is the highest set bit of the inverted value when the input is negative. All
                // three forms report 0xFFFFFFFF when there is no such bit, so a caller that
                // divides by the count has a value it can test.
                let scanned = if name == "VFfbhI32" {
                    self.temp(
                        "uint",
                        &format!("as_type<int>({source}) < 0 ? ~{source} : {source}"),
                    )
                } else {
                    source
                };
                let count = if name == "VFfblB32" {
                    format!("ctz({scanned})")
                } else {
                    format!("clz({scanned})")
                };
                format!("{scanned} == 0u ? 0xFFFFFFFFu : {count}")
            }
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
            "VMin3F16" => {
                let first = self.half_float_source(instruction, 0)?;
                let second = self.half_float_source(instruction, 1)?;
                let third = self.half_float_source(instruction, 2)?;
                self.f16_result(
                    instruction,
                    destination,
                    &format!("fmin(fmin({first}, {second}), {third})"),
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
            "VDivFmasF32" => self.float_result(
                instruction,
                &format!(
                    "remu_div_fmas_f32({}, {}, {}, vcc)",
                    self.float_source(instruction, 0)?,
                    self.float_source(instruction, 1)?,
                    self.float_source(instruction, 2)?
                ),
            ),
            // V_DIV_SCALE_F32 is the entry point of the same sequence: it decides whether the
            // numerator or denominator needs an exponent shift and publishes that decision in
            // VCC. Its exponent thresholds and denormal tests are not settled here, and a scale
            // that guesses them would corrupt the quotient rather than fail to build it.
            "VDivScaleF32" => {
                return Err(
                    "V_DIV_SCALE_F32 operand-scaling thresholds are unproven; refusing to guess the precise-division preop"
                        .to_owned(),
                );
            }
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
            // GFX10 encodes these as the carryless V_ADD_NC_U32 / V_SUB_NC_U32 /
            // V_SUBREV_NC_U32, which is why the truncated result matches either signedness and
            // the legacy I32 names survived. Their CLAMP bit is not decorative: it is how the
            // hardware spells an unsigned saturating add or subtract, so the sum has to be
            // formed at 64-bit width before it can be narrowed.
            "VAddI32" => {
                let expression = format!(
                    "(long){} + (long){}",
                    self.raw_source(instruction, 0)?,
                    self.raw_source(instruction, 1)?
                );
                self.integer_result(instruction, &expression, false)
            }
            "VSubI32" => {
                let expression = format!(
                    "(long){} - (long){}",
                    self.raw_source(instruction, 0)?,
                    self.raw_source(instruction, 1)?
                );
                self.integer_result(instruction, &expression, false)
            }
            "VSubrevI32" => {
                let expression = format!(
                    "(long){} - (long){}",
                    self.raw_source(instruction, 1)?,
                    self.raw_source(instruction, 0)?
                );
                self.integer_result(instruction, &expression, false)
            }
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
                // The signed extract replicates the extracted field's top bit through bit 31.
                // Masking the shifted value instead zero-fills those bits, so every negative
                // field (a packed signed offset, a sign-extended index) came out positive.
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
                    &format!(
                        "min({} & 31u, 32u - {offset})",
                        self.raw_source(instruction, 2)?
                    ),
                );
                format!(
                    "{width} == 0u ? 0u : as_type<uint>(extract_bits({value}, {offset}, {width}))"
                )
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
            name @ ("VCvtF32Ubyte0"
            | "VCvtF32Ubyte1"
            | "VCvtF32Ubyte2"
            | "VCvtF32Ubyte3") => {
                // RDNA2 V_CVT_F32_UBYTE{0..3} selects one unsigned byte from S0 and
                // numerically converts it to f32.  Keep S0 untyped until after extraction;
                // interpreting the source as a float first changes the byte payload.
                let shift = match name {
                    "VCvtF32Ubyte0" => 0,
                    "VCvtF32Ubyte1" => 8,
                    "VCvtF32Ubyte2" => 16,
                    "VCvtF32Ubyte3" => 24,
                    _ => unreachable!("matched the complete UBYTE conversion family"),
                };
                let source = self.raw_source(instruction, 0)?;
                self.float_result(
                    instruction,
                    &format!("(float)(({source} >> {shift}u) & 0xFFu)"),
                )
            }
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
            // The low half of the product is the same bit pattern for either signedness, so the
            // signed and unsigned low multiplies share one lowering.
            "VMulLoU32" | "VMulLoI32" => format!(
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
        let relative_index = self.relative_vector_index_expression();
        let value = if relative_source {
            if source.kind != OperandKind::VectorRegister {
                return Err(format!(
                    "{name} requires a vector-register source, found {source}"
                ));
            }
            self.relative_vector_source(source.value, &relative_index)?
        } else {
            self.raw_source(instruction, 0)?
        };
        if relative_destination {
            self.store_relative_vector(destination, &value, &relative_index);
        } else {
            self.store_vector(destination, &value);
        }
        Ok(())
    }

    /// Selector for `V_MOVREL*`: architectural M0 (`s[124]`).
    fn relative_vector_index_expression(&self) -> String {
        "s[124]".to_owned()
    }

    /// Detects `V_READFIRSTLANE vIdx → sTmp; V_CMPX_EQ sTmp, vIdx; S_MOV M0, sTmp; V_MOVRELS`.
    ///
    /// Used only to refuse the pattern loudly (J3); the former EXEC-preserving peephole is gone.
    pub(crate) fn pixel_movrels_serialization_index_vgpr(&self, cmpx: &Instruction) -> Option<u32> {
        if self.options.stage != crate::MetalShaderStage::Pixel {
            return None;
        }
        if !cmpx.opcode.is("VCmpxEqU32") {
            return None;
        }
        let index = self
            .program
            .instructions
            .iter()
            .position(|instruction| instruction.pc == cmpx.pc)?;
        if index == 0 {
            return None;
        }
        let readfirst = &self.program.instructions[index - 1];
        let mov_m0 = self.program.instructions.get(index + 1)?;
        let movrels = self.program.instructions.get(index + 2)?;
        if !readfirst.opcode.is("VReadfirstlaneB32")
            || !mov_m0.opcode.is("SMovB32")
            || !(movrels.opcode.is("VMovrelsB32") || movrels.opcode.is("VMovrelsdB32"))
        {
            return None;
        }
        let index_vgpr = readfirst.sources.first().and_then(|source| {
            (source.kind == OperandKind::VectorRegister).then_some(source.value)
        })?;
        let tmp_sgpr = readfirst.destinations.first().and_then(|destination| {
            (destination.kind == OperandKind::ScalarRegister).then_some(destination.value)
        })?;
        let cmp_scalar = cmpx.sources.first()?;
        let cmp_vector = cmpx.sources.get(1)?;
        if cmp_scalar.kind != OperandKind::ScalarRegister
            || cmp_scalar.value != tmp_sgpr
            || cmp_vector.kind != OperandKind::VectorRegister
            || cmp_vector.value != index_vgpr
        {
            return None;
        }
        let m0_source = mov_m0.sources.first()?;
        let m0_destination = mov_m0.destinations.first()?;
        if m0_source.kind != OperandKind::ScalarRegister
            || m0_source.value != tmp_sgpr
            || m0_destination.kind != OperandKind::ScalarRegister
            || m0_destination.value != 124
        {
            return None;
        }
        Some(index_vgpr)
    }

    fn relative_vector_source(&mut self, base: u32, index: &str) -> Result<String, String> {
        if base >= crate::VECTOR_REGISTER_COUNT {
            return Err(format!(
                "relative vector source v{base} exceeds the vector register file"
            ));
        }
        // GFX10 returns VGPR0 when M0 selects beyond the stage's allocation. Expanding this into
        // fixed cases preserves that behavior and lets module scalarization keep independent
        // register locals instead of materializing a 256-word addressable array per Metal lane.
        let selected = self.temp("uint", "v[0]");
        self.line(3, &format!("switch ({index}) {{"));
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

    fn store_relative_vector(&mut self, base: u32, value: &str, index: &str) {
        for register in base..self.options.allocated_vector_register_count {
            self.line(
                3,
                &format!(
                    "if (exec && {index} == {}u) v[{register}] = {value};",
                    register - base
                ),
            );
        }
    }
}
