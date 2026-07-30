// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

//! SALU lowering: scalar arithmetic, scalar compares and lane-mask bookkeeping.

use remu_shader::{Instruction, Operand, OperandKind, ShaderEncoding};

use crate::{Compiler, EXEC_HI, EXEC_LO, SCALAR_REGISTER_COUNT, VCC_HI, VCC_LO, format_u32};

pub(crate) fn destination_scalar(instruction: &Instruction) -> Result<u32, String> {
    instruction
        .destinations
        .first()
        .copied()
        .filter(|operand| operand.kind == OperandKind::ScalarRegister)
        .map(|operand| operand.value)
        .ok_or_else(|| "missing scalar destination".to_owned())
}

impl Compiler<'_> {
    pub(crate) fn store_lane_mask(&mut self, register: u32, condition: &str) {
        let (low, high) = self.ballot_words(condition);
        match register {
            VCC_LO => {
                self.line(3, &format!("vcc = {condition};"));
                self.line(3, &format!("s[{VCC_LO}] = {low};"));
                self.line(3, &format!("s[{VCC_HI}] = {high};"));
            }
            EXEC_LO => self.store_exec_mask(condition),
            register if register < SCALAR_REGISTER_COUNT => {
                self.line(3, &format!("s[{register}] = {low};"));
                if register + 1 < SCALAR_REGISTER_COUNT {
                    self.line(3, &format!("s[{}] = {high};", register + 1));
                }
            }
            _ => {}
        }
    }

    pub(crate) fn store_exec_mask(&mut self, condition: &str) {
        let (low, high) = self.ballot_words(condition);
        self.line(3, &format!("exec = {condition};"));
        self.line(3, &format!("s[{EXEC_LO}] = {low};"));
        self.line(3, &format!("s[{EXEC_HI}] = {high};"));
    }

    pub(crate) fn ballot(condition: &str) -> String {
        format!("(uint)(uint64_t)simd_ballot({condition})")
    }

    pub(crate) fn mask_source(&self, operand: Operand) -> Result<String, String> {
        match operand.kind {
            OperandKind::ScalarRegister if operand.value + 1 < SCALAR_REGISTER_COUNT => Ok(self
                .lane_mask_predicate(
                    &format!("s[{}]", operand.value),
                    &format!("s[{}]", operand.value + 1),
                )),
            _ => Err(format!("invalid lane-mask operand {operand}")),
        }
    }

    #[allow(clippy::too_many_lines)]
    pub(crate) fn emit_scalar_alu(&mut self, instruction: &Instruction) -> Result<(), String> {
        let name = instruction
            .opcode
            .known_name()
            .ok_or_else(|| "raw scalar opcode".to_owned())?;
        if instruction.encoding == ShaderEncoding::Sopc {
            return self.emit_scalar_compare(instruction, name, "SCmp");
        }
        let destination = destination_scalar(instruction)?;
        if instruction.encoding == ShaderEncoding::Sopk {
            let immediate = u16::try_from(instruction.words[0] & 0xffff)
                .expect("masked immediate fits u16")
                .cast_signed();
            if name.starts_with("SCmpk") {
                return self.emit_scalar_compare_values(
                    name,
                    "SCmpk",
                    &format!("s[{destination}]"),
                    &format_u32(u32::from_ne_bytes(i32::from(immediate).to_ne_bytes())),
                );
            }
            let value = match name {
                "SMovkI32" => format_u32(u32::from_ne_bytes(i32::from(immediate).to_ne_bytes())),
                "SAddkI32" => format!(
                    "(s[{destination}] + {})",
                    format_u32(u32::from_ne_bytes(i32::from(immediate).to_ne_bytes()))
                ),
                "SMulkI32" => format!(
                    "(s[{destination}] * {})",
                    format_u32(u32::from_ne_bytes(i32::from(immediate).to_ne_bytes()))
                ),
                _ => return Err(format!("unsupported scalar immediate opcode {name}")),
            };
            self.store_scalar(destination, &value);
            return Ok(());
        }
        if name == "SGetpcB64" {
            let address = self.program.address
                + u64::from(instruction.pc)
                + u64::try_from(instruction.words.len() * 4)
                    .map_err(|_| "instruction size overflow".to_owned())?;
            let low = u32::try_from(address & u64::from(u32::MAX))
                .expect("masked address low half fits u32");
            let high = u32::try_from(address >> 32).expect("address high half fits u32");
            self.store_scalar(destination, &format_u32(low));
            self.store_scalar(destination + 1, &format_u32(high));
            return Ok(());
        }
        if name.ends_with("SaveexecB64") {
            return self.emit_saveexec64(instruction, name, destination);
        }
        if name == "SBcnt1I32B64" {
            let source = self.temp("ulong", &self.raw_source64(instruction, 0)?);
            let result = self.temp(
                "uint",
                &format!("popcount((uint){source}) + popcount((uint)({source} >> 32))"),
            );
            self.store_scalar(destination, &result);
            self.line(3, &format!("scc = {result} != 0u;"));
            return Ok(());
        }
        if name == "SBfmB64" {
            let width = self.raw_source(instruction, 0)?;
            let offset = self.raw_source(instruction, 1)?;
            let mask = self.temp("ulong", &format!("(1ul << ({width} & 63u)) - 1ul"));
            let result = self.temp("ulong", &format!("{mask} << ({offset} & 63u)"));
            self.store_scalar_pair(destination, &result);
            return Ok(());
        }
        if name.ends_with("B64") || matches!(name, "SBfeU64" | "SBfeI64") {
            return self.emit_scalar64(instruction, name, destination);
        }

        let left = self.raw_source(instruction, 0)?;
        match name {
            "SMovB32" => {
                self.store_scalar(destination, &left);
                return Ok(());
            }
            "SNotB32" => {
                let result = self.temp("uint", &format!("~({left})"));
                self.store_scalar(destination, &result);
                self.line(3, &format!("scc = {result} != 0u;"));
                return Ok(());
            }
            "SBrevB32" => {
                let result = self.temp("uint", &format!("reverse_bits({left})"));
                self.store_scalar(destination, &result);
                self.line(3, &format!("scc = {result} != 0u;"));
                return Ok(());
            }
            "SBcnt1I32B32" => {
                let result = self.temp("uint", &format!("popcount({left})"));
                self.store_scalar(destination, &result);
                self.line(3, &format!("scc = {result} != 0u;"));
                return Ok(());
            }
            _ => {}
        }
        let right = self.raw_source(instruction, 1)?;
        if matches!(name, "SAddcU32" | "SSubbU32") {
            let carry = self.temp("ulong", "scc ? 1ul : 0ul");
            if name == "SAddcU32" {
                let wide = self.temp(
                    "ulong",
                    &format!("(ulong){left} + (ulong){right} + {carry}"),
                );
                self.store_scalar(destination, &format!("(uint){wide}"));
                self.line(3, &format!("scc = ({wide} >> 32) != 0ul;"));
            } else {
                let subtrahend = self.temp("ulong", &format!("(ulong){right} + {carry}"));
                self.store_scalar(destination, &format!("{left} - (uint){subtrahend}"));
                self.line(3, &format!("scc = (ulong){left} < {subtrahend};"));
            }
            return Ok(());
        }
        let (expression, scc) = match name {
            "SAddU32" => (
                format!("({left} + {right})"),
                Some(format!("RESULT < {left}")),
            ),
            "SSubU32" => (
                format!("({left} - {right})"),
                Some(format!("{right} > {left}")),
            ),
            "SAddI32" => (
                format!("({left} + {right})"),
                Some(format!(
                    "((~({left} ^ {right}) & ({left} ^ RESULT)) >> 31) != 0u"
                )),
            ),
            "SSubI32" => (
                format!("({left} - {right})"),
                Some(format!(
                    "((({left} ^ {right}) & ({left} ^ RESULT)) >> 31) != 0u"
                )),
            ),
            "SMulI32" => (format!("({left} * {right})"), None),
            "SMulHiU32" => (format!("mulhi({left}, {right})"), None),
            "SAndB32" => (format!("({left} & {right})"), Some("RESULT != 0u".into())),
            "SOrB32" => (format!("({left} | {right})"), Some("RESULT != 0u".into())),
            "SXorB32" => (format!("({left} ^ {right})"), Some("RESULT != 0u".into())),
            "SAndn2B32" => (format!("({left} & ~{right})"), Some("RESULT != 0u".into())),
            "SOrn2B32" => (format!("({left} | ~{right})"), Some("RESULT != 0u".into())),
            "SLshlB32" => (
                format!("({left} << ({right} & 31u))"),
                Some("RESULT != 0u".into()),
            ),
            "SLshrB32" => (
                format!("({left} >> ({right} & 31u))"),
                Some("RESULT != 0u".into()),
            ),
            "SAshrI32" => (
                format!("as_type<uint>(as_type<int>({left}) >> ({right} & 31u))"),
                Some("RESULT != 0u".into()),
            ),
            "SBfmB32" => (
                format!("(((1u << ({left} & 31u)) - 1u) << ({right} & 31u))"),
                None,
            ),
            "SBfeU32" | "SBfeI32" => {
                let offset = self.temp("uint", &format!("{right} & 31u"));
                let width = self.temp(
                    "uint",
                    &format!("min(({right} >> 16) & 0x7Fu, 32u - {offset})"),
                );
                let value = if name == "SBfeI32" {
                    format!(
                        "{width} == 0u ? 0u : as_type<uint>(extract_bits(as_type<int>({left}), {offset}, {width}))"
                    )
                } else {
                    format!("{width} == 0u ? 0u : extract_bits({left}, {offset}, {width})")
                };
                (value, Some("RESULT != 0u".into()))
            }
            "SCselectB32" => (format!("(scc ? {left} : {right})"), None),
            "SMinU32" => (
                format!("min({left}, {right})"),
                Some(format!("{left} < {right}")),
            ),
            "SMaxU32" => (
                format!("max({left}, {right})"),
                Some(format!("{left} > {right}")),
            ),
            "SMinI32" => (
                format!("as_type<uint>(min(as_type<int>({left}), as_type<int>({right})))"),
                Some(format!("as_type<int>({left}) < as_type<int>({right})")),
            ),
            "SMaxI32" => (
                format!("as_type<uint>(max(as_type<int>({left}), as_type<int>({right})))"),
                Some(format!("as_type<int>({left}) > as_type<int>({right})")),
            ),
            "SAbsdiffI32" => {
                // Keep the arithmetic unsigned so both subtraction and negation
                // retain the ISA's wrapping i32 behavior without invoking signed
                // overflow in Metal.
                let difference = self.temp("uint", &format!("{left} - {right}"));
                (
                    format!("as_type<int>({difference}) < 0 ? 0u - {difference} : {difference}"),
                    Some("RESULT != 0u".into()),
                )
            }
            "SLshl1AddU32" | "SLshl2AddU32" | "SLshl3AddU32" | "SLshl4AddU32" => {
                let shift = name.as_bytes()[5] - b'0';
                (format!("(({left} << {shift}) + {right})"), None)
            }
            "SPackLlB32B16" => (
                format!("(({left} & 0xFFFFu) | (({right} & 0xFFFFu) << 16))"),
                None,
            ),
            _ => return Err(format!("unsupported scalar opcode {name}")),
        };
        let result = self.temp("uint", &expression);
        self.store_scalar(destination, &result);
        if let Some(scc) = scc {
            self.line(3, &format!("scc = {};", scc.replace("RESULT", &result)));
        }
        Ok(())
    }

    fn emit_saveexec64(
        &mut self,
        instruction: &Instruction,
        name: &str,
        destination: u32,
    ) -> Result<(), String> {
        let source = self.raw_source64(instruction, 0)?;
        // Preserve the complete local portion of old EXEC in its architectural 64-bit position.
        // A split guest wave64 runs as two SIMD32 groups: the first group owns bits 0..31 and the
        // second owns bits 32..63. Keeping masks half-local lets structured EXEC predication and
        // later B64 restore operations remain exact without cross-SIMD communication.
        let old_expression = if self.split_wave64() {
            "((ulong)(uint64_t)simd_ballot(exec) << (remu_wave_half * 32u))"
        } else {
            "(ulong)(uint64_t)simd_ballot(exec)"
        };
        let old = self.temp("ulong", old_expression);
        let expression = match name {
            "SAndSaveexecB64" => format!("({old} & {source})"),
            "SOrSaveexecB64" => format!("({old} | {source})"),
            "SXorSaveexecB64" => format!("({old} ^ {source})"),
            "SAndn1SaveexecB64" => format!("(~{source} & {old})"),
            "SAndn2SaveexecB64" => format!("({source} & ~{old})"),
            "SOrn1SaveexecB64" => format!("(~{source} | {old})"),
            "SOrn2SaveexecB64" => format!("({source} | ~{old})"),
            "SNandSaveexecB64" => format!("~({source} & {old})"),
            "SNorSaveexecB64" => format!("~({source} | {old})"),
            "SXnorSaveexecB64" => format!("~({source} ^ {old})"),
            _ => return Err(format!("unsupported 64-bit saveexec opcode {name}")),
        };
        let architectural_mask = if self.split_wave64() {
            "0xFFFFFFFFFFFFFFFFul"
        } else {
            "0xFFFFFFFFul"
        };
        let new = self.temp("ulong", &format!("({expression}) & {architectural_mask}"));
        self.store_scalar_pair(destination, &old);
        self.line(3, &format!("s[{EXEC_LO}] = (uint){new};"));
        self.line(3, &format!("s[{EXEC_HI}] = (uint)({new} >> 32);"));
        let exec =
            self.lane_mask_predicate(&format!("(uint){new}"), &format!("(uint)({new} >> 32)"));
        self.line(3, &format!("exec = {exec};"));
        self.line(3, &format!("scc = {new} != 0ul;"));
        Ok(())
    }

    fn emit_scalar_compare(
        &mut self,
        instruction: &Instruction,
        name: &str,
        prefix: &str,
    ) -> Result<(), String> {
        if matches!(name, "SBitcmp0B32" | "SBitcmp1B32") {
            let left = self.raw_source(instruction, 0)?;
            let right = self.raw_source(instruction, 1)?;
            let set = format!("((({left}) >> (({right}) & 31u)) & 1u) != 0u");
            if name == "SBitcmp1B32" {
                self.line(3, &format!("scc = {set};"));
            } else {
                self.line(3, &format!("scc = !({set});"));
            }
            return Ok(());
        }
        if matches!(name, "SCmpEqU64" | "SCmpLgU64") {
            let left = self.raw_source64(instruction, 0)?;
            let right = self.raw_source64(instruction, 1)?;
            let operator = if name == "SCmpEqU64" { "==" } else { "!=" };
            self.line(3, &format!("scc = ({left}) {operator} ({right});"));
            return Ok(());
        }
        let left = self.raw_source(instruction, 0)?;
        let right = self.raw_source(instruction, 1)?;
        self.emit_scalar_compare_values(name, prefix, &left, &right)
    }

    fn emit_scalar_compare_values(
        &mut self,
        name: &str,
        prefix: &str,
        left: &str,
        right: &str,
    ) -> Result<(), String> {
        let suffix = name
            .strip_prefix(prefix)
            .ok_or_else(|| format!("invalid scalar compare name {name}"))?;
        let signed = suffix.ends_with("I32");
        let relation = suffix
            .get(..suffix.len().saturating_sub(3))
            .ok_or_else(|| format!("invalid scalar compare suffix {suffix}"))?;
        let operator = match relation {
            "Eq" => "==",
            "Lg" => "!=",
            "Gt" => ">",
            "Ge" => ">=",
            "Lt" => "<",
            "Le" => "<=",
            _ => return Err(format!("unsupported scalar comparison {name}")),
        };
        if signed {
            self.line(
                3,
                &format!("scc = as_type<int>({left}) {operator} as_type<int>({right});"),
            );
        } else {
            self.line(3, &format!("scc = ({left}) {operator} ({right});"));
        }
        Ok(())
    }

    fn emit_scalar64(
        &mut self,
        instruction: &Instruction,
        name: &str,
        destination: u32,
    ) -> Result<(), String> {
        let left = self.raw_source64(instruction, 0)?;
        let (value, sets_scc) = match name {
            "SMovB64" => (left, false),
            "SNotB64" => (format!("~({left})"), true),
            "SWqmB64" => {
                let quad = self.temp(
                    "ulong",
                    &format!("({left} | ({left} >> 1) | ({left} >> 2) | ({left} >> 3)) & 0x1111111111111111ul"),
                );
                (format!("({quad} * 0xFul)"), true)
            }
            "SLshlB64" | "SLshrB64" => {
                let shift = self.raw_source(instruction, 1)?;
                let operator = if name == "SLshlB64" { "<<" } else { ">>" };
                (format!("({left} {operator} ({shift} & 63u))"), true)
            }
            "SBfeU64" | "SBfeI64" => {
                let control = self.raw_source(instruction, 1)?;
                let offset = self.temp("uint", &format!("{control} & 63u"));
                let width = self.temp(
                    "uint",
                    &format!("min(({control} >> 16) & 0x7Fu, 64u - {offset})"),
                );
                let mask = self.temp(
                    "ulong",
                    &format!("{width} >= 64u ? 0xFFFFFFFFFFFFFFFFul : ((1ul << {width}) - 1ul)"),
                );
                let extracted = format!("({left} >> {offset}) & {mask}");
                if name == "SBfeI64" {
                    (
                        format!(
                            "({width} == 0u || {width} >= 64u) ? {extracted} : (({extracted} ^ (1ul << ({width} - 1u))) - (1ul << ({width} - 1u)))"
                        ),
                        true,
                    )
                } else {
                    (extracted, true)
                }
            }
            "SCselectB64" => {
                let right = self.raw_source64(instruction, 1)?;
                (format!("(scc ? {left} : {right})"), false)
            }
            "SAndB64" | "SOrB64" | "SXorB64" | "SAndn2B64" | "SOrn2B64" | "SNandB64"
            | "SNorB64" | "SXnorB64" => {
                let right = self.raw_source64(instruction, 1)?;
                let expression = match name {
                    "SAndB64" => format!("({left} & {right})"),
                    "SOrB64" => format!("({left} | {right})"),
                    "SXorB64" => format!("({left} ^ {right})"),
                    "SAndn2B64" => format!("({left} & ~{right})"),
                    "SOrn2B64" => format!("({left} | ~{right})"),
                    "SNandB64" => format!("~({left} & {right})"),
                    "SNorB64" => format!("~({left} | {right})"),
                    "SXnorB64" => format!("~({left} ^ {right})"),
                    _ => unreachable!("64-bit bitwise opcode is matched above"),
                };
                (expression, true)
            }
            _ => return Err(format!("unsupported 64-bit scalar opcode {name}")),
        };
        let result = self.temp("ulong", &value);
        self.store_scalar_pair(destination, &result);
        if self.split_wave64()
            && self.uses_lds_memory()
            && name == "SMovB64"
            && destination == EXEC_LO
            && instruction.sources.first().is_some_and(|source| {
                source.kind == OperandKind::EncodedConstant && source.value == 193
            })
        {
            // One guest wave64 executes in lockstep, but Metal realizes it as two independent
            // SIMD32 groups. Guest shaders can use an EXEC=-1 restore as their wave reconvergence
            // point and immediately consume LDS values produced on either side of the preceding
            // branch without an S_BARRIER. Synchronize the host halves here so the high half
            // cannot read the low half's shared tile (or vice versa) before its stores complete.
            self.line(3, "threadgroup_barrier(mem_flags::mem_threadgroup);");
        }
        if sets_scc {
            self.line(3, &format!("scc = {result} != 0ul;"));
        }
        Ok(())
    }

    pub(crate) fn store_scalar(&mut self, register: u32, expression: &str) {
        if register >= SCALAR_REGISTER_COUNT {
            return;
        }
        match register {
            VCC_LO | VCC_HI => {
                let value = self.temp("uint", expression);
                self.line(3, &format!("s[{register}] = {value};"));
                let predicate =
                    self.lane_mask_predicate(&format!("s[{VCC_LO}]"), &format!("s[{VCC_HI}]"));
                self.line(3, &format!("vcc = {predicate};"));
            }
            EXEC_LO | EXEC_HI => {
                let value = self.temp("uint", expression);
                self.line(3, &format!("s[{register}] = {value};"));
                let predicate =
                    self.lane_mask_predicate(&format!("s[{EXEC_LO}]"), &format!("s[{EXEC_HI}]"));
                self.line(3, &format!("exec = {predicate};"));
            }
            _ => self.line(3, &format!("s[{register}] = {expression};")),
        }
    }

    fn store_scalar_pair(&mut self, register: u32, expression: &str) {
        if register + 1 >= SCALAR_REGISTER_COUNT {
            return;
        }
        self.line(3, &format!("s[{register}] = (uint){expression};"));
        self.line(
            3,
            &format!("s[{}] = (uint)({expression} >> 32);", register + 1),
        );
        if register == VCC_LO {
            let predicate =
                self.lane_mask_predicate(&format!("s[{VCC_LO}]"), &format!("s[{VCC_HI}]"));
            self.line(3, &format!("vcc = {predicate};"));
        } else if register == EXEC_LO {
            let predicate =
                self.lane_mask_predicate(&format!("s[{EXEC_LO}]"), &format!("s[{EXEC_HI}]"));
            self.line(3, &format!("exec = {predicate};"));
        }
    }
}
