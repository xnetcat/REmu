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
        // EXEC has its own bookkeeping path. Select it before materializing ballot words so a
        // split wave64 compare does not emit one unused ballot here and another in
        // `store_exec_mask`.
        if register == EXEC_LO {
            self.store_exec_mask(condition);
            return;
        }
        self.invalidate_heap_descriptor_shadows(
            register,
            if self.uses_wave64_lane_mask_pairs() {
                2
            } else {
                1
            },
        );
        let (low, high) = self.ballot_words(condition);
        match register {
            VCC_LO => {
                self.line(3, &format!("vcc = {condition};"));
                self.line(3, &format!("s[{VCC_LO}] = {low};"));
                // VCC is an architectural pair; keep HI coherent even on wave32.
                self.line(3, &format!("s[{VCC_HI}] = {high};"));
            }
            register if register < SCALAR_REGISTER_COUNT => {
                self.line(3, &format!("s[{register}] = {low};"));
                // Wave32 lane masks occupy a single SGPR. Writing a high half there would
                // clobber the adjacent SGPR — on NGG mesh that is often s[3], which carries
                // packed wave index/count used by the export reduce. Wave64 masks are pairs even
                // when a non-threadgroup Metal stage can execute only their local SIMD32 half;
                // clear that absent high half before subsequent B64 scalar mask operations.
                if self.uses_wave64_lane_mask_pairs() && register + 1 < SCALAR_REGISTER_COUNT {
                    self.line(3, &format!("s[{}] = {high};", register + 1));
                }
            }
            _ => {}
        }
    }

    pub(crate) fn store_exec_mask(&mut self, condition: &str) {
        self.invalidate_heap_descriptor_shadows(EXEC_LO, 2);
        self.line(3, &format!("exec = {condition};"));
        let (low, high) = self.ballot_words(condition);
        self.line(3, &format!("s[{EXEC_LO}] = {low};"));
        self.line(3, &format!("s[{EXEC_HI}] = {high};"));
    }

    /// Wave ballot for packing EXEC/VCC lane masks.
    ///
    /// SharpEmu documents that Metal leaves simdgroup ops undefined inside a divergent
    /// PC-dispatcher: real `simd_ballot` there corrupts EXEC/VCC and kills whole quads.
    /// Fragment functions have the same gap even on Forward/Structured control flow —
    /// `simd_ballot(true)` is observed to return 0, after which `S_WQM` plus an EXEC restore
    /// discards the whole coverage mask. Pixel (and Dispatcher vertex) therefore publish this
    /// thread's predicate in every local SIMD32 bit so `(mask >> remu_lane) & 1` stays exact.
    /// Compute/mesh keep real ballots — the scalar PC keeps the wave in lockstep on that path.
    pub(crate) fn ballot(&self, condition: &str) -> String {
        if self.uses_per_thread_lane_masks() {
            format!("({condition}) ? 0xffffffffu : 0u")
        } else {
            format!("(uint)(uint64_t)simd_ballot({condition})")
        }
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
            let immediate =
                u16::try_from(instruction.words[0] & 0xffff).expect("masked immediate fits u16");
            if name.starts_with("SCmpk") {
                // SOPK's unsigned compares are the exception to the normally signed SIMM16
                // encoding: AMDGPU zero-extends IMM16 for every S_CMPK_*_U32 opcode. This is
                // observable in NGG's `s_cmpk_le_u32 packed_counts, 0xffff` empty-workaround
                // check; sign extension turns the limit into UINT_MAX and collapses non-empty
                // output allocations to the workaround's one-vertex/one-primitive fallback.
                let immediate = if name.ends_with("U32") {
                    u32::from(immediate)
                } else {
                    u32::from_ne_bytes(i32::from(immediate.cast_signed()).to_ne_bytes())
                };
                return self.emit_scalar_compare_values(
                    name,
                    "SCmpk",
                    &format!("s[{destination}]"),
                    &format_u32(immediate),
                );
            }
            let immediate = immediate.cast_signed();
            let immediate_bits = u32::from_ne_bytes(i32::from(immediate).to_ne_bytes());
            if name == "SAddkI32" {
                let immediate = format_u32(immediate_bits);
                let old = self.temp("uint", &format!("s[{destination}]"));
                let result = self.temp("uint", &format!("({old} + {immediate})"));
                self.store_scalar(destination, &result);
                self.line(
                    3,
                    &format!(
                        "scc = (({old} ^ {result}) & ({immediate} ^ {result}) & 0x80000000u) != 0u;"
                    ),
                );
                return Ok(());
            }
            let value = match name {
                "SMovkI32" => format_u32(immediate_bits),
                "SMulkI32" => format!("(s[{destination}] * {})", format_u32(immediate_bits)),
                _ => return Err(format!("unsupported scalar immediate opcode {name}")),
            };
            self.store_scalar(destination, &value);
            return Ok(());
        }
        if name == "SGetpcB64" {
            let address = self
                .program
                .instruction_address(instruction.pc)
                .checked_add(
                    u64::try_from(instruction.words.len() * 4)
                        .map_err(|_| "instruction size overflow".to_owned())?,
                )
                .ok_or_else(|| "instruction address overflow".to_owned())?;
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
        if name.ends_with("SaveexecB32") {
            return self.emit_saveexec32(instruction, name, destination);
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
                // S_BREV_B32 is one of the SOP1 operations that leaves SCC alone (RDNA2 ISA):
                // only the bit counts and the logical operations define it. Writing SCC here
                // destroyed the condition a preceding compare had set for a later branch.
                let result = self.temp("uint", &format!("reverse_bits({left})"));
                self.store_scalar(destination, &result);
                return Ok(());
            }
            "SBcnt1I32B32" => {
                let result = self.temp("uint", &format!("popcount({left})"));
                self.store_scalar(destination, &result);
                self.line(3, &format!("scc = {result} != 0u;"));
                return Ok(());
            }
            "SFF1I32B32" => {
                self.store_scalar(
                    destination,
                    &format!("{left} == 0u ? 0xFFFFFFFFu : ctz({left})"),
                );
                return Ok(());
            }
            "SBitset0B32" => {
                self.store_scalar(
                    destination,
                    &format!("s[{destination}] & ~(1u << ({left} & 31u))"),
                );
                return Ok(());
            }
            "SBitset1B32" => {
                self.store_scalar(
                    destination,
                    &format!("s[{destination}] | (1u << ({left} & 31u))"),
                );
                return Ok(());
            }
            _ => {}
        }
        if instruction.encoding == ShaderEncoding::Sop1 {
            return Err(format!("unsupported scalar unary opcode {name}"));
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
            "SMaxU32" => {
                let expression = format!("max({left}, {right})");
                let scc_expr = format!("{left} > {right}");
                let result = self.temp("uint", &expression);
                self.store_scalar(destination, &result);
                self.line(3, &format!("scc = {scc_expr};"));
                return Ok(());
            }
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
                if self.options.ngg_fixed_allocation_fallback {
                    // The fixed wave64 path is a compatibility ABI. Keep its generated scalar
                    // dataflow identical to the last verified Superliminal renderer: these
                    // shifts build descriptor offsets in the VCC register pair, and injecting a
                    // new SCC write changed the fixed mesh program even though the following
                    // instruction normally replaces it.
                    (format!("(({left} << {shift}) + {right})"), None)
                } else {
                    // SCC is the unsigned carry-out of the shifted addition, so the sum has to be
                    // formed at 64-bit width; the shifted-out high bits participate in that carry.
                    let wide = self.temp(
                        "ulong",
                        &format!("((ulong){left} << {shift}) + (ulong){right}"),
                    );
                    let result = self.temp("uint", &format!("(uint){wide}"));
                    self.store_scalar(destination, &result);
                    self.line(3, &format!("scc = ({wide} >> 32) != 0ul;"));
                    return Ok(());
                }
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
        let old_expression = if self.uses_per_thread_lane_masks() {
            "(exec ? 0xfffffffffffffffful : 0ul)"
        } else if self.split_wave64() {
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

    fn emit_saveexec32(
        &mut self,
        instruction: &Instruction,
        name: &str,
        destination: u32,
    ) -> Result<(), String> {
        let source = self.raw_source(instruction, 0)?;
        // Wave32 saveexec only updates EXEC_LO. Preserve the active SIMD32 ballot as the old mask
        // so later EXEC restores remain exact for GFX10 wave32 programs.
        let old = self.temp("uint", &self.ballot("exec"));
        let expression = match name {
            "SAndSaveexecB32" => format!("({old} & {source})"),
            "SOrSaveexecB32" => format!("({old} | {source})"),
            "SXorSaveexecB32" => format!("({old} ^ {source})"),
            "SAndn1SaveexecB32" => format!("(~{source} & {old})"),
            "SAndn2SaveexecB32" => format!("({source} & ~{old})"),
            "SOrn1SaveexecB32" => format!("(~{source} | {old})"),
            "SOrn2SaveexecB32" => format!("({source} | ~{old})"),
            "SNandSaveexecB32" => format!("~({source} & {old})"),
            "SNorSaveexecB32" => format!("~({source} | {old})"),
            "SXnorSaveexecB32" => format!("~({source} ^ {old})"),
            _ => return Err(format!("unsupported 32-bit saveexec opcode {name}")),
        };
        let new = self.temp("uint", &expression);
        self.store_scalar(destination, &old);
        self.line(3, &format!("s[{EXEC_LO}] = {new};"));
        self.line(3, &format!("s[{EXEC_HI}] = 0u;"));
        let exec = self.lane_mask_predicate(&new, "0u");
        self.line(3, &format!("exec = {exec};"));
        self.line(3, &format!("scc = {new} != 0u;"));
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
        if self.split_wave64_is_whole_compute_threadgroup()
            && self.uses_lds_memory()
            && name == "SMovB64"
            && destination == EXEC_LO
            && instruction.sources.first().is_some_and(|source| {
                source.kind == OperandKind::EncodedConstant && source.value == 193
            })
        {
            // This guest workgroup is exactly one wave64 (or a partial final wave), so every
            // thread reaches the architectural full-EXEC reconvergence together. Unlike a
            // SIMD-local barrier, Metal's whole-threadgroup barrier publishes LDS writes from
            // both host SIMD32 halves before either half performs the following reads.
            self.line(3, "threadgroup_barrier(mem_flags::mem_threadgroup);");
        } else if self.options.stage == crate::MetalShaderStage::Compute
            && self.split_wave64()
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
            // branch without an S_BARRIER. A whole-threadgroup barrier is invalid here because
            // independent guest waves may be in different branches or loop iterations. First
            // finish every lane's local SIMD32 LDS accesses, then rendezvous only the paired half
            // through the same double-buffered epoch records used by scalar mask branches.
            let rendezvous = self.next_temp;
            self.next_temp += 1;
            let prefix = format!("remu_wave_memory_{rendezvous}");
            self.line(3, "simdgroup_barrier(mem_flags::mem_threadgroup);");
            self.line(3, "remu_wave_pair_epoch += 1u;");
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
                    "atomic_store_explicit(&remu_wave_pair_states[{prefix}_slot], remu_wave_pair_epoch << 1u, memory_order_relaxed);"
                ),
            );
            self.line(3, "}");
            self.line(3, &format!("if ({prefix}_has_partner) {{"));
            self.line(4, "do {");
            self.line(
                5,
                &format!(
                    "{prefix}_remote = atomic_load_explicit(&remu_wave_pair_states[{prefix}_slot ^ 1u], memory_order_relaxed);"
                ),
            );
            self.line(
                4,
                &format!("}} while (({prefix}_remote >> 1u) != remu_wave_pair_epoch);"),
            );
            self.line(3, "}");
            self.line(3, "simdgroup_barrier(mem_flags::mem_threadgroup);");
        } else if self.split_wave64()
            && self.uses_lds_memory()
            && name == "SMovB64"
            && destination == EXEC_LO
            && instruction.sources.first().is_some_and(|source| {
                source.kind == OperandKind::EncodedConstant && source.value == 193
            })
        {
            // Mesh kernels retain the converged whole-threadgroup rendezvous used before the
            // compute-only paired-half experiment. A pairwise atomic spin is not scheduling-safe
            // for mesh threadgroups on Apple GPUs and caused command buffers to enter Error.
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
        self.invalidate_heap_descriptor_shadows(register, 1);
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
        self.invalidate_heap_descriptor_shadows(register, 2);
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
