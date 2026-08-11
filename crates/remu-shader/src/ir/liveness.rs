// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

//! Scalar liveness analysis used to size the evaluation slice of a decoded program.

use std::collections::{BTreeMap, BTreeSet, VecDeque};

use super::{Instruction, InstructionControl, Opcode, OperandKind, ShaderEncoding};

pub(crate) fn instruction_requires_evaluation(instruction: &Instruction) -> bool {
    matches!(
        instruction.encoding,
        ShaderEncoding::Sop1
            | ShaderEncoding::Sop2
            | ShaderEncoding::Sopc
            | ShaderEncoding::Sopp
            | ShaderEncoding::Sopk
    ) || matches!(
        instruction.control,
        Some(
            InstructionControl::ScalarMemory(_)
                | InstructionControl::GlobalMemory(_)
                | InstructionControl::BufferMemory(_)
                | InstructionControl::Image(_)
        )
    )
}

const SCALAR_LIVE_WORDS: usize = 4;

type ScalarLiveSet = [u64; SCALAR_LIVE_WORDS];

#[derive(Clone, Copy, Default, Eq, PartialEq)]
struct EvaluationLiveSet {
    registers: ScalarLiveSet,
    scalar_condition: bool,
}

fn scalar_operand_width(instruction: &Instruction) -> u32 {
    if instruction
        .opcode
        .known_name()
        .is_some_and(|name| name.ends_with("B64") || name.ends_with("I64") || name.ends_with("U64"))
    {
        2
    } else {
        1
    }
}

/// Width of one scalar source operand. A `B64` result does not imply that every input is 64-bit:
/// shifts, bit-field extracts and bit tests take a 32-bit control operand.
fn scalar_source_width(instruction: &Instruction, source_index: usize) -> u32 {
    if source_index > 0
        && matches!(
            instruction.opcode.known_name(),
            Some(
                "SLshlB64"
                    | "SLshrB64"
                    | "SAshrI64"
                    | "SBfeU64"
                    | "SBfeI64"
                    | "SBitcmp0B64"
                    | "SBitcmp1B64"
            )
        )
    {
        return 1;
    }
    if matches!(
        instruction.opcode.known_name(),
        Some("SBfmB64" | "SBitset0B64" | "SBitset1B64")
    ) {
        return 1;
    }
    scalar_operand_width(instruction)
}

/// Width of the scalar destination write, which differs from the source width for the 64-bit
/// scans and counts: they read a register pair but write one 32-bit result. Killing the pair
/// would under-approximate liveness for the neighboring register.
fn scalar_destination_width(instruction: &Instruction) -> u32 {
    if instruction.opcode.is("SBitreplicateB64B32") {
        return 2;
    }
    if matches!(
        instruction.opcode.known_name(),
        Some(
            "SBcnt0I32B64"
                | "SBcnt1I32B64"
                | "SFF0I32B64"
                | "SFF1I32B64"
                | "SFlbitI32B64"
                | "SFlbitI32I64"
        )
    ) {
        return 1;
    }
    scalar_operand_width(instruction)
}

/// The M0-relative moves touch a dynamically selected register; every scalar register must stay
/// live around them so path dedup and recurrence detection never merge states they distinguish.
fn scalar_instruction_touches_dynamic_register(instruction: &Instruction) -> bool {
    matches!(
        instruction.opcode.known_name(),
        Some("SMovrelsB32" | "SMovrelsB64" | "SMovreldB32" | "SMovreldB64")
    )
}

fn scalar_instruction_defines_condition(instruction: &Instruction) -> bool {
    if instruction.encoding == ShaderEncoding::Sopc
        || (instruction.encoding == ShaderEncoding::Sopk && instruction.opcode.starts_with("SCmpk"))
    {
        return true;
    }
    matches!(
        instruction.opcode.known_name(),
        Some(
            "SNotB32"
                | "SBcnt0I32B32"
                | "SBcnt1I32B32"
                | "SBcnt0I32B64"
                | "SBcnt1I32B64"
                | "SWqmB32"
                | "SQuadmaskB32"
                | "SQuadmaskB64"
                | "SAbsI32"
                | "SNotB64"
                | "SWqmB64"
                | "SLshlB64"
                | "SLshrB64"
                | "SAshrI64"
                | "SBfeU64"
                | "SBfeI64"
                | "SCselectB64"
                | "SAndB64"
                | "SOrB64"
                | "SXorB64"
                | "SAndn2B64"
                | "SOrn2B64"
                | "SNandB64"
                | "SNorB64"
                | "SXnorB64"
                | "SAddU32"
                | "SSubU32"
                | "SAddI32"
                | "SSubI32"
                | "SAddcU32"
                | "SSubbU32"
                | "SMinI32"
                | "SMinU32"
                | "SMaxI32"
                | "SMaxU32"
                | "SAndB32"
                | "SOrB32"
                | "SXorB32"
                | "SAndn2B32"
                | "SOrn2B32"
                | "SNandB32"
                | "SNorB32"
                | "SXnorB32"
                | "SLshlB32"
                | "SLshrB32"
                | "SAshrI32"
                | "SBfeU32"
                | "SBfeI32"
                | "SAbsdiffI32"
                | "SLshl1AddU32"
                | "SLshl2AddU32"
                | "SLshl3AddU32"
                | "SLshl4AddU32"
                | "SAddkI32"
                | "SAndSaveexecB32"
                | "SOrSaveexecB32"
                | "SXorSaveexecB32"
                | "SAndn1SaveexecB32"
                | "SAndn2SaveexecB32"
                | "SOrn1SaveexecB32"
                | "SOrn2SaveexecB32"
                | "SNandSaveexecB32"
                | "SNorSaveexecB32"
                | "SXnorSaveexecB32"
                | "SAndSaveexecB64"
                | "SOrSaveexecB64"
                | "SXorSaveexecB64"
                | "SAndn1SaveexecB64"
                | "SAndn2SaveexecB64"
                | "SOrn1SaveexecB64"
                | "SOrn2SaveexecB64"
                | "SNandSaveexecB64"
                | "SNorSaveexecB64"
                | "SXnorSaveexecB64"
        )
    )
}

fn scalar_instruction_reads_condition(instruction: &Instruction) -> bool {
    matches!(
        instruction.opcode.known_name(),
        Some(
            "SCselectB32"
                | "SCselectB64"
                | "SAddcU32"
                | "SSubbU32"
                | "SCmovB32"
                | "SCmovB64"
                | "SCmovkI32"
        )
    ) || instruction
        .sources
        .iter()
        .any(|operand| operand.kind == OperandKind::EncodedConstant && operand.value == 253)
}

fn scalar_destination_is_also_source(instruction: &Instruction) -> bool {
    matches!(
        instruction.opcode.known_name(),
        Some(
            "SAddkI32"
                | "SMulkI32"
                | "SBitset0B32"
                | "SBitset1B32"
                | "SBitset0B64"
                | "SBitset1B64"
                | "SCmovkI32"
                | "SCmovB32"
                | "SCmovB64"
        )
    )
}

#[derive(Clone, Copy, Default, Eq, PartialEq)]
struct ScalarMaskTaint {
    registers: ScalarLiveSet,
    scalar_condition: bool,
}

fn scalar_range_is_mask_tainted(taint: &ScalarMaskTaint, first: u32, count: u32) -> bool {
    (first..first.saturating_add(count).min(256))
        .any(|register| scalar_live_contains(&taint.registers, register))
}

fn set_scalar_taint_range(taint: &mut ScalarMaskTaint, first: u32, count: u32, value: bool) {
    scalar_live_remove_range(&mut taint.registers, first, count);
    if value {
        scalar_live_insert_range(&mut taint.registers, first, count);
    }
}

fn scalar_sources_are_mask_tainted(instruction: &Instruction, taint: &ScalarMaskTaint) -> bool {
    let explicit_source_tainted = instruction
        .sources
        .iter()
        .enumerate()
        .any(|(index, source)| match source.kind {
            OperandKind::ScalarRegister => scalar_range_is_mask_tainted(
                taint,
                source.value,
                scalar_source_width(instruction, index),
            ),
            OperandKind::EncodedConstant => match source.value {
                251 => scalar_range_is_mask_tainted(taint, 106, 2),
                252 => scalar_range_is_mask_tainted(taint, 126, 2),
                253 => taint.scalar_condition,
                _ => false,
            },
            OperandKind::VectorRegister => true,
            OperandKind::LiteralConstant => false,
        });
    let destination_source_tainted = (instruction.encoding == ShaderEncoding::Sopk
        && instruction.opcode.starts_with("SCmpk")
        || scalar_destination_is_also_source(instruction))
        && instruction.destinations.iter().any(|destination| {
            destination.kind == OperandKind::ScalarRegister
                && scalar_range_is_mask_tainted(
                    taint,
                    destination.value,
                    scalar_destination_width(instruction),
                )
        });

    explicit_source_tainted
        || destination_source_tainted
        || (scalar_instruction_reads_condition(instruction) && taint.scalar_condition)
}

fn scalar_memory_address_is_mask_tainted(
    instruction: &Instruction,
    taint: &ScalarMaskTaint,
) -> bool {
    instruction
        .sources
        .iter()
        .enumerate()
        .any(|(index, source)| match source.kind {
            OperandKind::ScalarRegister => {
                let width = if index == 0 {
                    if instruction.opcode.starts_with("SBufferLoad") {
                        4
                    } else {
                        2
                    }
                } else {
                    1
                };
                scalar_range_is_mask_tainted(taint, source.value, width)
            }
            OperandKind::EncodedConstant => match source.value {
                251 => scalar_range_is_mask_tainted(taint, 106, 2),
                252 => scalar_range_is_mask_tainted(taint, 126, 2),
                253 => taint.scalar_condition,
                _ => false,
            },
            OperandKind::VectorRegister => true,
            OperandKind::LiteralConstant => false,
        })
}

fn apply_scalar_mask_taint(
    instruction: &Instruction,
    mut taint: ScalarMaskTaint,
) -> ScalarMaskTaint {
    let scalar_encoding = matches!(
        instruction.encoding,
        ShaderEncoding::Sop1 | ShaderEncoding::Sop2 | ShaderEncoding::Sopc | ShaderEncoding::Sopk
    );
    if scalar_encoding {
        let saveexec = instruction.opcode.contains("Saveexec");
        let saveexec_width = if saveexec
            && instruction
                .opcode
                .known_name()
                .is_some_and(|name| name.ends_with("SaveexecB32"))
        {
            1
        } else {
            2
        };
        let old_exec_tainted = scalar_range_is_mask_tainted(&taint, 126, saveexec_width);
        let source_tainted = scalar_sources_are_mask_tainted(instruction, &taint);
        let result_tainted = source_tainted || (saveexec && old_exec_tainted);
        let compare_k =
            instruction.encoding == ShaderEncoding::Sopk && instruction.opcode.starts_with("SCmpk");
        if !compare_k {
            for destination in &instruction.destinations {
                if destination.kind == OperandKind::ScalarRegister {
                    set_scalar_taint_range(
                        &mut taint,
                        destination.value,
                        scalar_destination_width(instruction),
                        if saveexec {
                            old_exec_tainted
                        } else {
                            result_tainted
                        },
                    );
                }
            }
        }
        if scalar_instruction_defines_condition(instruction) {
            taint.scalar_condition = result_tainted;
        }
        if saveexec {
            set_scalar_taint_range(&mut taint, 126, saveexec_width, result_tainted);
            if saveexec_width == 1 {
                set_scalar_taint_range(&mut taint, 127, 1, false);
            }
        }
        return taint;
    }

    if matches!(
        instruction.encoding,
        ShaderEncoding::Smem | ShaderEncoding::Smrd
    ) {
        let result_tainted = scalar_memory_address_is_mask_tainted(instruction, &taint);
        for destination in &instruction.destinations {
            if destination.kind == OperandKind::ScalarRegister {
                set_scalar_taint_range(&mut taint, destination.value, 1, result_tainted);
            }
        }
        return taint;
    }

    let vector_compare = instruction.opcode.starts_with("VCmp");
    let vector_compare_updates_exec = instruction.opcode.starts_with("VCmpx");
    let control_destination = match instruction.control {
        Some(InstructionControl::Vop3(control)) => control.scalar_destination,
        Some(InstructionControl::Sdwa(control)) => control.scalar_destination,
        _ => None,
    };
    // VCMPX updates only EXEC. Ordinary compares update their explicit SDWA destination or the
    // default VCC pair, including VOP3/e64 instructions whose control has no scalar destination.
    // Handle this before generic vector scalar outputs because decoded VCMPX SDWA fields are not
    // architectural destinations.
    if vector_compare_updates_exec {
        set_scalar_taint_range(&mut taint, 126, 2, true);
        return taint;
    }
    if vector_compare {
        set_scalar_taint_range(&mut taint, control_destination.unwrap_or(106), 2, true);
        return taint;
    }

    // Vector lane results written to SGPRs are mask-derived even when VOP3B/SDWA encodes the
    // scalar destination in control metadata rather than `Instruction::destinations`.
    let explicit_destination_width = if instruction.encoding == ShaderEncoding::Vopc {
        2
    } else {
        scalar_destination_width(instruction)
    };
    for destination in &instruction.destinations {
        if destination.kind == OperandKind::ScalarRegister {
            set_scalar_taint_range(
                &mut taint,
                destination.value,
                explicit_destination_width,
                true,
            );
        }
    }
    if let Some(destination) = control_destination {
        set_scalar_taint_range(&mut taint, destination, 2, true);
    }
    taint
}

fn merge_scalar_mask_taint(target: &mut ScalarMaskTaint, source: ScalarMaskTaint) -> bool {
    let previous = *target;
    for word in 0..SCALAR_LIVE_WORDS {
        target.registers[word] |= source.registers[word];
    }
    target.scalar_condition |= source.scalar_condition;
    *target != previous
}

fn scalar_mask_taint_successors(instructions: &[Instruction], index: usize) -> Vec<usize> {
    let instruction = &instructions[index];
    if instruction.opcode == Opcode::END_PROGRAM {
        return Vec::new();
    }
    let name = instruction.opcode.known_name();
    let is_branch = matches!(
        name,
        Some(
            "SBranch"
                | "SCbranchScc0"
                | "SCbranchScc1"
                | "SCbranchVccz"
                | "SCbranchVccnz"
                | "SCbranchExecz"
                | "SCbranchExecnz"
        )
    );
    let mut successors = Vec::with_capacity(2);
    if is_branch
        && let Some(target) = evaluation_branch_target(instruction)
        && let Ok(target_index) =
            instructions.binary_search_by_key(&target, |candidate| candidate.pc)
    {
        successors.push(target_index);
    }
    if !matches!(name, Some("SBranch")) && index + 1 < instructions.len() {
        successors.push(index + 1);
    }
    successors
}

/// Returns whether SCC at `condition_index` can depend on EXEC/VCC along a reachable path in this
/// natural loop. A forward may-taint fixed point follows branch targets and joins, so unreachable
/// definitions cannot replace or contaminate the definition that reaches the SCC exit.
pub(crate) fn scalar_condition_depends_on_vector_mask(
    instructions: &[Instruction],
    condition_index: usize,
) -> bool {
    if condition_index >= instructions.len() || instructions.is_empty() {
        return false;
    }
    // Dynamic SGPR addressing and indirect/subroutine transfers need alias or interprocedural
    // state that this local natural-loop analysis does not have. Decline classification rather
    // than attributing a mask definition to the wrong reaching value.
    if instructions.iter().any(|instruction| {
        scalar_instruction_touches_dynamic_register(instruction)
            || matches!(
                instruction.opcode.known_name(),
                Some("SCallB64" | "SSetpcB64" | "SSwappcB64")
            )
    }) {
        return false;
    }

    let mut entry_taint = vec![None; instructions.len()];
    let mut initial_taint = ScalarMaskTaint::default();
    scalar_live_insert_range(&mut initial_taint.registers, 106, 2);
    scalar_live_insert_range(&mut initial_taint.registers, 126, 2);
    entry_taint[0] = Some(initial_taint);
    let mut pending = VecDeque::from([0]);
    while let Some(index) = pending.pop_front() {
        let Some(input) = entry_taint[index] else {
            continue;
        };
        let output = apply_scalar_mask_taint(&instructions[index], input);
        for successor in scalar_mask_taint_successors(instructions, index) {
            let changed = if let Some(current) = &mut entry_taint[successor] {
                merge_scalar_mask_taint(current, output)
            } else {
                entry_taint[successor] = Some(output);
                true
            };
            if changed {
                pending.push_back(successor);
            }
        }
    }

    entry_taint[condition_index].is_some_and(|taint| taint.scalar_condition)
}

#[allow(clippy::too_many_lines)]
pub(crate) fn evaluation_scalar_liveness(
    instructions: &[Instruction],
    next_evaluation_instruction_indices: &[u32],
    linked_transfer_targets: &BTreeMap<u32, u32>,
) -> (Vec<Vec<u8>>, Vec<bool>) {
    let instruction_count = instructions.len();
    let mut successors = vec![Vec::<usize>::with_capacity(2); instruction_count];

    let next_evaluation = |index: usize| {
        next_evaluation_instruction_indices
            .get(index)
            .copied()
            .filter(|index| *index != u32::MAX)
            .and_then(|index| usize::try_from(index).ok())
    };

    for (index, instruction) in instructions.iter().enumerate() {
        if !instruction_requires_evaluation(instruction) {
            continue;
        }
        let name = instruction.opcode.known_name();
        if instruction.opcode == Opcode::END_PROGRAM || matches!(name, Some("SSwappcB64")) {
            continue;
        }
        if matches!(name, Some("SSetpcB64")) {
            if let Some(target) = linked_transfer_targets.get(&instruction.pc)
                && let Ok(target_index) =
                    instructions.binary_search_by_key(target, |candidate| candidate.pc)
                && let Some(target_evaluation) = next_evaluation(target_index)
            {
                successors[index].push(target_evaluation);
            }
            continue;
        }
        let is_branch = matches!(
            name,
            Some(
                "SBranch"
                    | "SCbranchScc0"
                    | "SCbranchScc1"
                    | "SCbranchVccz"
                    | "SCbranchVccnz"
                    | "SCbranchExecz"
                    | "SCbranchExecnz"
            )
        );
        if (is_branch || matches!(name, Some("SCallB64")))
            && let Some(target) = evaluation_branch_target(instruction)
            && let Ok(target_index) =
                instructions.binary_search_by_key(&target, |candidate| candidate.pc)
            && let Some(target_evaluation) = next_evaluation(target_index)
        {
            successors[index].push(target_evaluation);
        }
        if !matches!(name, Some("SBranch"))
            && let Some(fallthrough) = next_evaluation(index.saturating_add(1))
            && !successors[index].contains(&fallthrough)
        {
            successors[index].push(fallthrough);
        }
    }

    let mut live_in = vec![EvaluationLiveSet::default(); instruction_count];
    loop {
        let mut changed = false;
        for index in (0..instruction_count).rev() {
            if !instruction_requires_evaluation(&instructions[index]) {
                continue;
            }
            let instruction = &instructions[index];
            let mut live_out = EvaluationLiveSet::default();
            for &successor in &successors[index] {
                for word in 0..SCALAR_LIVE_WORDS {
                    live_out.registers[word] |= live_in[successor].registers[word];
                }
                live_out.scalar_condition |= live_in[successor].scalar_condition;
            }

            let compare_k = instruction.encoding == ShaderEncoding::Sopk
                && instruction.opcode.starts_with("SCmpk");
            let destination_live = instruction.destinations.iter().any(|operand| {
                operand.kind == OperandKind::ScalarRegister
                    && scalar_live_contains(&live_out.registers, operand.value)
            });
            let condition_live =
                scalar_instruction_defines_condition(instruction) && live_out.scalar_condition;
            let result_live = destination_live || condition_live;
            let mut updated = live_out;

            // A dynamically targeted transfer or register move invalidates every static liveness
            // assumption at this point: the indirect return path or the M0-relative index can
            // observe any register. Saturating the live set here is the sound over-approximation.
            let unlinked_indirect_transfer =
                matches!(instruction.opcode.known_name(), Some("SSwappcB64"))
                    || (matches!(instruction.opcode.known_name(), Some("SSetpcB64"))
                        && !linked_transfer_targets.contains_key(&instruction.pc));
            if unlinked_indirect_transfer
                || scalar_instruction_touches_dynamic_register(instruction)
            {
                scalar_live_insert_range(&mut updated.registers, 0, 256);
                updated.scalar_condition = true;
                if updated != live_in[index] {
                    live_in[index] = updated;
                    changed = true;
                }
                continue;
            }

            if !compare_k {
                let destination_width = scalar_destination_width(instruction);
                for operand in &instruction.destinations {
                    if operand.kind == OperandKind::ScalarRegister {
                        scalar_live_remove_range(
                            &mut updated.registers,
                            operand.value,
                            destination_width,
                        );
                    }
                }
            }
            if scalar_instruction_defines_condition(instruction) {
                updated.scalar_condition = false;
            }

            match instruction.control.as_ref() {
                Some(InstructionControl::Image(image)) => {
                    scalar_live_insert_range(&mut updated.registers, image.scalar_resource, 8);
                    if instruction.opcode.starts_with("ImageSample")
                        || instruction.opcode.starts_with("ImageGather")
                    {
                        scalar_live_insert_range(&mut updated.registers, image.scalar_sampler, 4);
                    }
                }
                Some(InstructionControl::GlobalMemory(global)) => {
                    if let Some(register) = global.scalar_address {
                        scalar_live_insert_range(&mut updated.registers, register, 2);
                    }
                }
                Some(InstructionControl::BufferMemory(buffer)) => {
                    scalar_live_insert_range(&mut updated.registers, buffer.scalar_resource, 4);
                }
                Some(InstructionControl::ScalarMemory(scalar)) => {
                    if let Some(base) = instruction.sources.first()
                        && base.kind == OperandKind::ScalarRegister
                    {
                        let base_width = if instruction.opcode.starts_with("SBufferLoad") {
                            4
                        } else {
                            2
                        };
                        scalar_live_insert_range(&mut updated.registers, base.value, base_width);
                    }
                    // The offset selects loaded scalar values, but not the Metal buffer identity.
                    // It only participates in recurrence when a loaded destination can affect a
                    // later binding or scalar branch.
                    if destination_live && let Some(register) = scalar.dynamic_offset_register {
                        scalar_live_insert_range(&mut updated.registers, register, 1);
                    }
                }
                _ => {}
            }

            match instruction.opcode.known_name() {
                Some("SCbranchScc0" | "SCbranchScc1") => {
                    updated.scalar_condition = true;
                }
                Some("SCbranchVccz" | "SCbranchVccnz") => {
                    scalar_live_insert_range(&mut updated.registers, 106, 2);
                }
                Some("SCbranchExecz" | "SCbranchExecnz") => {
                    scalar_live_insert_range(&mut updated.registers, 126, 2);
                }
                _ => {}
            }

            let scalar_operation = matches!(
                instruction.encoding,
                ShaderEncoding::Sop1
                    | ShaderEncoding::Sop2
                    | ShaderEncoding::Sopc
                    | ShaderEncoding::Sopk
            );
            if scalar_operation && result_live {
                let source_width = scalar_operand_width(instruction);
                for operand in &instruction.sources {
                    if operand.kind == OperandKind::ScalarRegister {
                        scalar_live_insert_range(
                            &mut updated.registers,
                            operand.value,
                            source_width,
                        );
                    }
                }
                if compare_k || scalar_destination_is_also_source(instruction) {
                    for operand in &instruction.destinations {
                        if operand.kind == OperandKind::ScalarRegister {
                            scalar_live_insert_range(
                                &mut updated.registers,
                                operand.value,
                                source_width,
                            );
                        }
                    }
                }
                if scalar_instruction_reads_condition(instruction) {
                    updated.scalar_condition = true;
                }
            }

            if updated != live_in[index] {
                live_in[index] = updated;
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }

    let registers = live_in
        .iter()
        .map(|set| {
            (0_u32..256)
                .filter(|register| scalar_live_contains(&set.registers, *register))
                .filter_map(|register| u8::try_from(register).ok())
                .collect()
        })
        .collect();
    let scalar_conditions = live_in
        .into_iter()
        .map(|set| set.scalar_condition)
        .collect();
    (registers, scalar_conditions)
}

pub(crate) fn evaluation_branch_target(instruction: &Instruction) -> Option<u32> {
    let encoded = instruction.words.first().copied()?;
    #[allow(clippy::cast_possible_truncation)]
    let offset = (encoded as u16).cast_signed();
    let next_pc = i64::from(instruction.pc)
        .checked_add(i64::try_from(instruction.words.len().checked_mul(4)?).ok()?)?;
    let target = next_pc.checked_add(i64::from(offset).checked_mul(4)?)?;
    u32::try_from(target).ok()
}

pub(crate) fn scalar_live_insert_range(set: &mut ScalarLiveSet, first: u32, count: u32) {
    for register in first..first.saturating_add(count).min(256) {
        let word = usize::try_from(register / 64).expect("scalar register word fits usize");
        let bit = register % 64;
        set[word] |= 1_u64 << bit;
    }
}

pub(crate) fn scalar_live_contains(set: &ScalarLiveSet, register: u32) -> bool {
    let word = usize::try_from(register / 64).expect("scalar register word fits usize");
    let bit = register % 64;
    (set[word] & (1_u64 << bit)) != 0
}

pub(crate) fn scalar_live_remove_range(set: &mut ScalarLiveSet, first: u32, count: u32) {
    for register in first..first.saturating_add(count).min(256) {
        let word = usize::try_from(register / 64).expect("scalar register word fits usize");
        let bit = register % 64;
        set[word] &= !(1_u64 << bit);
    }
}

pub(crate) fn insert_scalar_range(registers: &mut BTreeSet<u32>, first: u32, count: u32) {
    for register in first..first.saturating_add(count) {
        if register < 256 {
            registers.insert(register);
        }
    }
}

pub(crate) fn insert_input_scalar_range(
    inputs: &mut BTreeSet<u32>,
    defined: &BTreeSet<u32>,
    first: u32,
    count: u32,
    linear_scalar_flow: bool,
) {
    for register in first..first.saturating_add(count) {
        if register < 256 && (!linear_scalar_flow || !defined.contains(&register)) {
            inputs.insert(register);
        }
    }
}
