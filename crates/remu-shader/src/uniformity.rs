// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

//! Conservative def/use, side-effect and value-uniformity analysis.
//!
//! This is deliberately backend neutral.  A backend may consume a proof from this module, but an
//! unknown opcode, memory domain or implicit register effect is represented as an optimization
//! barrier rather than guessed from generated source text.

use std::collections::{BTreeMap, BTreeSet, VecDeque};

use crate::{
    Instruction, InstructionControl, OperandKind, Program, ShaderEncoding, build_control_flow_graph,
};

/// Stable identifier of one instruction definition or one CFG-join value.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ValueId(pub u32);

/// Architectural register/state named by an instruction.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum RegisterRef {
    Scalar(u32),
    Vector(u32),
    Scc,
    Vcc,
    Exec,
}

/// Value variation domain.  Ordering is from least to most varying.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum Variation {
    #[default]
    Unreachable,
    KernelUniform,
    WorkgroupUniform,
    GuestWaveUniform,
    Divergent,
}

impl Variation {
    #[must_use]
    pub const fn join(self, other: Self) -> Self {
        if (self as u8) >= (other as u8) {
            self
        } else {
            other
        }
    }

    #[must_use]
    pub const fn is_uniform(self) -> bool {
        matches!(
            self,
            Self::KernelUniform | Self::WorkgroupUniform | Self::GuestWaveUniform
        )
    }
}

/// Memory region named by an effect.  Unknown guest aliases deliberately share `Global`.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum MemoryRegion {
    Global,
    Image,
    Lds,
    Scratch,
}

/// Side-effect summary for one instruction.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum MemoryEffect {
    Pure,
    Collective,
    Read(MemoryRegion),
    Write(MemoryRegion),
    Atomic(MemoryRegion),
    Barrier,
    Unknown,
}

/// Exhaustive explicit and implicit semantics required by optimized lowering.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct InstructionSemantics {
    pub pc: u32,
    pub uses: Vec<RegisterRef>,
    pub definitions: Vec<RegisterRef>,
    pub effect: MemoryEffect,
    /// False means the instruction remains a loud legacy-lowering barrier.
    pub modeled: bool,
}

/// Host-proven variation of shader inputs.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct UniformityInputs {
    /// Initial SGPRs are user data and therefore kernel-uniform unless listed below.
    pub workgroup_uniform_scalars: BTreeSet<u32>,
    /// Explicitly divergent scalar inputs (for unusual host ABIs).
    pub divergent_scalars: BTreeSet<u32>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct ReachingValue {
    id: ValueId,
    variation: Variation,
}

/// Fixed-point analysis result indexed in program instruction order.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct UniformityAnalysis {
    semantics: Vec<InstructionSemantics>,
    definition_values: Vec<Vec<(RegisterRef, ValueId, Variation)>>,
    input_values: Vec<Vec<(RegisterRef, ValueId, Variation)>>,
    block_entries: Vec<BTreeMap<RegisterRef, (ValueId, Variation)>>,
}

impl UniformityAnalysis {
    #[must_use]
    pub fn semantics(&self, instruction_index: usize) -> Option<&InstructionSemantics> {
        self.semantics.get(instruction_index)
    }

    #[must_use]
    pub fn definition(
        &self,
        instruction_index: usize,
        register: RegisterRef,
    ) -> Option<(ValueId, Variation)> {
        self.definition_values
            .get(instruction_index)?
            .iter()
            .find_map(|(candidate, id, variation)| {
                (*candidate == register).then_some((*id, *variation))
            })
    }

    #[must_use]
    pub fn input(
        &self,
        instruction_index: usize,
        register: RegisterRef,
    ) -> Option<(ValueId, Variation)> {
        self.input_values
            .get(instruction_index)?
            .iter()
            .find_map(|(candidate, id, variation)| {
                (*candidate == register).then_some((*id, *variation))
            })
    }

    #[must_use]
    pub fn block_entry(&self, block: usize, register: RegisterRef) -> Option<(ValueId, Variation)> {
        self.block_entries.get(block)?.get(&register).copied()
    }

    #[must_use]
    pub fn analyzed_instruction_count(&self) -> usize {
        self.semantics.len()
    }

    #[must_use]
    pub fn uniform_definition_count(&self) -> usize {
        self.definition_values
            .iter()
            .flatten()
            .filter(|(_, _, variation)| variation.is_uniform())
            .count()
    }
}

/// Build exhaustive instruction records and solve reaching variation to a fixed point.
#[must_use]
pub fn analyze_uniformity(program: &Program, inputs: &UniformityInputs) -> UniformityAnalysis {
    let semantics: Vec<_> = program
        .instructions
        .iter()
        .map(classify_instruction)
        .collect();
    let cfg = build_control_flow_graph(program);
    if cfg.blocks().is_empty() {
        return UniformityAnalysis {
            semantics,
            definition_values: Vec::new(),
            input_values: Vec::new(),
            block_entries: Vec::new(),
        };
    }

    let registers: BTreeSet<_> = semantics
        .iter()
        .flat_map(|record| record.uses.iter().chain(&record.definitions).copied())
        .collect();
    let initial = initial_state(&registers, inputs);
    let definition_ids = stable_definition_ids(&semantics);
    let phi_ids = stable_phi_ids(cfg.blocks().len(), &registers, definition_ids.next);
    let mut entries = vec![BTreeMap::new(); cfg.blocks().len()];
    let mut exits = vec![BTreeMap::new(); cfg.blocks().len()];
    entries[cfg.entry()] = initial.clone();
    let mut queue: VecDeque<_> = (0..cfg.blocks().len()).collect();
    let mut queued = vec![true; cfg.blocks().len()];

    while let Some(block) = queue.pop_front() {
        queued[block] = false;
        let entry = if block == cfg.entry() {
            initial.clone()
        } else {
            join_predecessors(block, &registers, &exits, &cfg, &phi_ids)
        };
        let mut state = entry.clone();
        for index in cfg.blocks()[block].instructions.clone() {
            transfer(
                &semantics[index],
                &definition_ids.by_instruction[index],
                &mut state,
            );
        }
        let changed = entries[block] != entry || exits[block] != state;
        entries[block] = entry;
        exits[block] = state;
        if changed {
            for &successor in &cfg.blocks()[block].successors {
                if !queued[successor] {
                    queue.push_back(successor);
                    queued[successor] = true;
                }
            }
        }
    }

    let mut definition_values = vec![Vec::new(); semantics.len()];
    let mut input_values = vec![Vec::new(); semantics.len()];
    for (block, data) in cfg.blocks().iter().enumerate() {
        let mut state = entries[block].clone();
        for index in data.instructions.clone() {
            for register in &semantics[index].uses {
                if let Some(value) = state.get(register).copied() {
                    input_values[index].push((*register, value.id, value.variation));
                }
            }
            transfer(
                &semantics[index],
                &definition_ids.by_instruction[index],
                &mut state,
            );
            for register in &semantics[index].definitions {
                if let Some(value) = state.get(register).copied() {
                    definition_values[index].push((*register, value.id, value.variation));
                }
            }
        }
    }

    UniformityAnalysis {
        semantics,
        definition_values,
        input_values,
        block_entries: entries
            .into_iter()
            .map(|state| {
                state
                    .into_iter()
                    .map(|(register, value)| (register, (value.id, value.variation)))
                    .collect()
            })
            .collect(),
    }
}

struct DefinitionIds {
    by_instruction: Vec<BTreeMap<RegisterRef, ValueId>>,
    next: u32,
}

fn stable_definition_ids(semantics: &[InstructionSemantics]) -> DefinitionIds {
    let mut next = 0u32;
    let by_instruction = semantics
        .iter()
        .map(|record| {
            record
                .definitions
                .iter()
                .copied()
                .map(|register| {
                    let id = ValueId(next);
                    next = next.saturating_add(1);
                    (register, id)
                })
                .collect()
        })
        .collect();
    DefinitionIds {
        by_instruction,
        next,
    }
}

fn stable_phi_ids(
    block_count: usize,
    registers: &BTreeSet<RegisterRef>,
    mut next: u32,
) -> Vec<BTreeMap<RegisterRef, ValueId>> {
    (0..block_count)
        .map(|_| {
            registers
                .iter()
                .copied()
                .map(|register| {
                    let id = ValueId(next);
                    next = next.saturating_add(1);
                    (register, id)
                })
                .collect()
        })
        .collect()
}

fn initial_state(
    registers: &BTreeSet<RegisterRef>,
    inputs: &UniformityInputs,
) -> BTreeMap<RegisterRef, ReachingValue> {
    let mut next = u32::MAX;
    registers
        .iter()
        .copied()
        .map(|register| {
            let variation = match register {
                RegisterRef::Scalar(index) if inputs.divergent_scalars.contains(&index) => {
                    Variation::Divergent
                }
                RegisterRef::Scalar(index) if inputs.workgroup_uniform_scalars.contains(&index) => {
                    Variation::WorkgroupUniform
                }
                RegisterRef::Scalar(_) => Variation::KernelUniform,
                RegisterRef::Vector(_) | RegisterRef::Vcc | RegisterRef::Exec => {
                    Variation::Divergent
                }
                RegisterRef::Scc => Variation::GuestWaveUniform,
            };
            let id = ValueId(next);
            next = next.saturating_sub(1);
            (register, ReachingValue { id, variation })
        })
        .collect()
}

fn join_predecessors(
    block: usize,
    registers: &BTreeSet<RegisterRef>,
    exits: &[BTreeMap<RegisterRef, ReachingValue>],
    cfg: &crate::ControlFlowGraph,
    phi_ids: &[BTreeMap<RegisterRef, ValueId>],
) -> BTreeMap<RegisterRef, ReachingValue> {
    let predecessors = cfg.predecessors(block);
    if predecessors.is_empty() {
        return BTreeMap::new();
    }
    registers
        .iter()
        .filter_map(|register| {
            let values: Vec<_> = predecessors
                .iter()
                .filter_map(|predecessor| exits[*predecessor].get(register).copied())
                .collect();
            let first = values.first().copied()?;
            let variation = values.iter().fold(Variation::Unreachable, |joined, value| {
                joined.join(value.variation)
            });
            let id = if values.iter().all(|value| value.id == first.id) {
                first.id
            } else {
                phi_ids[block][register]
            };
            Some((*register, ReachingValue { id, variation }))
        })
        .collect()
}

fn transfer(
    semantics: &InstructionSemantics,
    definition_ids: &BTreeMap<RegisterRef, ValueId>,
    state: &mut BTreeMap<RegisterRef, ReachingValue>,
) {
    if !semantics.modeled {
        // With no exhaustive architectural record, a destination-free instruction can still
        // change implicit scalar state or invalidate a reusable memory epoch.  Poison every
        // reaching proof at the barrier; later optimized sites must establish fresh definitions.
        for value in state.values_mut() {
            value.variation = Variation::Divergent;
        }
    }
    let inputs = semantics
        .uses
        .iter()
        .filter_map(|register| state.get(register))
        .fold(Variation::KernelUniform, |joined, value| {
            joined.join(value.variation)
        });
    for register in &semantics.definitions {
        let variation = output_variation(*register, semantics, inputs);
        state.insert(
            *register,
            ReachingValue {
                id: definition_ids[register],
                variation,
            },
        );
    }
}

fn output_variation(
    register: RegisterRef,
    semantics: &InstructionSemantics,
    inputs: Variation,
) -> Variation {
    if !semantics.modeled || semantics.effect == MemoryEffect::Unknown {
        return Variation::Divergent;
    }
    match register {
        RegisterRef::Vector(_) => Variation::Divergent,
        RegisterRef::Vcc | RegisterRef::Exec => Variation::Divergent,
        RegisterRef::Scalar(_) | RegisterRef::Scc => match semantics.effect {
            MemoryEffect::Read(MemoryRegion::Global) | MemoryEffect::Collective => {
                inputs.max(Variation::GuestWaveUniform)
            }
            MemoryEffect::Read(_) => Variation::Divergent,
            _ => inputs,
        },
    }
}

fn classify_instruction(instruction: &Instruction) -> InstructionSemantics {
    let mut uses = instruction
        .sources
        .iter()
        .filter_map(register_for_operand)
        .collect::<Vec<_>>();
    let mut definitions = instruction
        .destinations
        .iter()
        .filter_map(register_for_operand)
        .collect::<Vec<_>>();
    add_control_registers(instruction, &mut uses, &mut definitions);
    add_implicit_registers(instruction, &mut uses, &mut definitions);
    uses.sort_unstable();
    uses.dedup();
    definitions.sort_unstable();
    definitions.dedup();
    let (effect, modeled) = classify_effect(instruction);
    InstructionSemantics {
        pc: instruction.pc,
        uses,
        definitions,
        effect,
        modeled,
    }
}

fn register_for_operand(operand: &crate::Operand) -> Option<RegisterRef> {
    match operand.kind {
        OperandKind::ScalarRegister => Some(match operand.value {
            106 | 107 => RegisterRef::Vcc,
            126 | 127 => RegisterRef::Exec,
            index => RegisterRef::Scalar(index),
        }),
        OperandKind::VectorRegister => Some(RegisterRef::Vector(operand.value)),
        OperandKind::EncodedConstant | OperandKind::LiteralConstant => None,
    }
}

fn add_control_registers(
    instruction: &Instruction,
    uses: &mut Vec<RegisterRef>,
    definitions: &mut Vec<RegisterRef>,
) {
    match instruction.control.as_ref() {
        Some(InstructionControl::ScalarMemory(control)) => {
            if let Some(base) = instruction.sources.first().and_then(register_for_operand) {
                uses.push(base);
            }
            if let Some(offset) = control.dynamic_offset_register {
                uses.push(RegisterRef::Scalar(offset));
            }
        }
        Some(InstructionControl::BufferMemory(control)) => {
            uses.extend((0..4).map(|word| RegisterRef::Scalar(control.scalar_resource + word)));
            if control.index_enabled || control.offset_enabled {
                uses.push(RegisterRef::Vector(control.vector_address));
            }
        }
        Some(InstructionControl::Image(control)) => {
            uses.extend((0..8).map(|word| RegisterRef::Scalar(control.scalar_resource + word)));
            uses.extend((0..4).map(|word| RegisterRef::Scalar(control.scalar_sampler + word)));
            uses.extend(
                control
                    .address_registers
                    .iter()
                    .copied()
                    .map(RegisterRef::Vector),
            );
        }
        Some(InstructionControl::GlobalMemory(control)) => {
            uses.push(RegisterRef::Vector(control.vector_address));
            if let Some(base) = control.scalar_address {
                uses.push(RegisterRef::Scalar(base));
                uses.push(RegisterRef::Scalar(base + 1));
            }
        }
        Some(InstructionControl::DataShare(_)) => {
            uses.push(RegisterRef::Scalar(124));
        }
        Some(
            InstructionControl::Export(_)
            | InstructionControl::Interpolation(_)
            | InstructionControl::Vop3(_)
            | InstructionControl::Sdwa(_)
            | InstructionControl::Vop3p(_)
            | InstructionControl::Dpp(_)
            | InstructionControl::Dpp8(_),
        )
        | None => {}
    }
    if instruction.opcode.starts_with("BufferStore")
        && let Some(InstructionControl::BufferMemory(control)) = instruction.control
    {
        definitions.retain(|register| *register != RegisterRef::Vector(control.vector_data));
    }
}

fn add_implicit_registers(
    instruction: &Instruction,
    uses: &mut Vec<RegisterRef>,
    definitions: &mut Vec<RegisterRef>,
) {
    let Some(name) = instruction.opcode.known_name() else {
        return;
    };
    if matches!(name, "SCbranchScc0" | "SCbranchScc1") {
        uses.push(RegisterRef::Scc);
    }
    if matches!(name, "SCbranchVccz" | "SCbranchVccnz") {
        uses.push(RegisterRef::Vcc);
    }
    if matches!(name, "SCbranchExecz" | "SCbranchExecnz") {
        uses.push(RegisterRef::Exec);
    }
    if name.contains("Saveexec") {
        uses.push(RegisterRef::Exec);
        definitions.push(RegisterRef::Exec);
        definitions.push(RegisterRef::Scc);
    }
    if scalar_opcode_uses_scc(name) {
        uses.push(RegisterRef::Scc);
    }
    if instruction.encoding == ShaderEncoding::Sopc
        || name.starts_with("SCmp")
        || scalar_opcode_defines_scc(name)
    {
        definitions.push(RegisterRef::Scc);
    }
    if instruction.encoding == ShaderEncoding::Vopc || name.starts_with("VCmp") {
        definitions.push(RegisterRef::Vcc);
    }
}

fn scalar_opcode_uses_scc(name: &str) -> bool {
    matches!(
        name,
        "SAddcU32" | "SSubbU32" | "SCselectB32" | "SCselectB64"
    )
}

fn scalar_opcode_defines_scc(name: &str) -> bool {
    name.contains("Saveexec")
        || matches!(
            name,
            "SAbsdiffI32"
                | "SAddI32"
                | "SAddU32"
                | "SAddcU32"
                | "SAddkI32"
                | "SAndB32"
                | "SAndB64"
                | "SAndn2B32"
                | "SAndn2B64"
                | "SAshrI32"
                | "SBcnt1I32B32"
                | "SBcnt1I32B64"
                | "SBfeI32"
                | "SBfeI64"
                | "SBfeU32"
                | "SBfeU64"
                | "SBitcmp0B32"
                | "SBitcmp1B32"
                | "SLshl1AddU32"
                | "SLshl2AddU32"
                | "SLshl3AddU32"
                | "SLshl4AddU32"
                | "SLshlB32"
                | "SLshlB64"
                | "SLshrB32"
                | "SLshrB64"
                | "SMaxI32"
                | "SMaxU32"
                | "SMinI32"
                | "SMinU32"
                | "SNandB64"
                | "SNorB64"
                | "SNotB32"
                | "SNotB64"
                | "SOrB32"
                | "SOrB64"
                | "SOrn2B32"
                | "SOrn2B64"
                | "SSubI32"
                | "SSubU32"
                | "SSubbU32"
                | "SWqmB64"
                | "SXnorB64"
                | "SXorB32"
                | "SXorB64"
        )
}

fn classify_effect(instruction: &Instruction) -> (MemoryEffect, bool) {
    let Some(name) = instruction.opcode.known_name() else {
        return (MemoryEffect::Unknown, false);
    };
    if name.contains("Atomic") {
        let region = if instruction.encoding == ShaderEncoding::Ds {
            MemoryRegion::Lds
        } else {
            MemoryRegion::Global
        };
        return (MemoryEffect::Atomic(region), true);
    }
    if matches!(name, "SBarrier" | "SWaitcnt" | "SWaitcntDepctr") {
        return (MemoryEffect::Barrier, true);
    }
    if instruction.encoding == ShaderEncoding::Ds {
        return if name.contains("Read") {
            (MemoryEffect::Read(MemoryRegion::Lds), true)
        } else if name.contains("Write") {
            (MemoryEffect::Write(MemoryRegion::Lds), true)
        } else {
            (MemoryEffect::Unknown, false)
        };
    }
    if matches!(
        instruction.encoding,
        ShaderEncoding::Smem | ShaderEncoding::Smrd
    ) {
        return (MemoryEffect::Read(MemoryRegion::Global), true);
    }
    if matches!(
        instruction.encoding,
        ShaderEncoding::Mubuf | ShaderEncoding::Mtbuf | ShaderEncoding::Flat
    ) {
        return if name.contains("Load") {
            (MemoryEffect::Read(MemoryRegion::Global), true)
        } else if name.contains("Store") {
            (MemoryEffect::Write(MemoryRegion::Global), true)
        } else {
            (MemoryEffect::Unknown, false)
        };
    }
    if instruction.encoding == ShaderEncoding::Mimg {
        return if name.contains("Store") {
            (MemoryEffect::Write(MemoryRegion::Image), true)
        } else {
            (MemoryEffect::Read(MemoryRegion::Image), true)
        };
    }
    if instruction.encoding == ShaderEncoding::Exp {
        return (MemoryEffect::Write(MemoryRegion::Global), true);
    }
    if matches!(name, "VReadfirstlaneB32" | "VReadlaneB32") {
        return (MemoryEffect::Collective, true);
    }
    if matches!(
        instruction.encoding,
        ShaderEncoding::Vop1
            | ShaderEncoding::Vop2
            | ShaderEncoding::Vopc
            | ShaderEncoding::Vop3
            | ShaderEncoding::Vop3p
            | ShaderEncoding::Vintrp
    ) {
        // These decoded families carry all register operands and VCC destinations explicitly;
        // lane collectives were removed above and therefore cannot be mistaken for pure ALU.
        return (MemoryEffect::Pure, true);
    }
    if scalar_opcode_is_fully_modeled(name) || control_opcode_is_fully_modeled(name) {
        return (MemoryEffect::Pure, true);
    }
    (MemoryEffect::Unknown, false)
}

fn control_opcode_is_fully_modeled(name: &str) -> bool {
    matches!(
        name,
        "SEndpgm"
            | "SBranch"
            | "SCbranchScc0"
            | "SCbranchScc1"
            | "SCbranchVccz"
            | "SCbranchVccnz"
            | "SCbranchExecz"
            | "SCbranchExecnz"
            | "SNop"
            | "SClause"
            | "SInstPrefetch"
            | "VNop"
    )
}

fn scalar_opcode_is_fully_modeled(name: &str) -> bool {
    matches!(
        name,
        "SAbsdiffI32"
            | "SAddI32"
            | "SAddU32"
            | "SAddcU32"
            | "SAddkI32"
            | "SAndB32"
            | "SAndB64"
            | "SAndSaveexecB32"
            | "SAndSaveexecB64"
            | "SAndn1SaveexecB32"
            | "SAndn1SaveexecB64"
            | "SAndn2B32"
            | "SAndn2B64"
            | "SAndn2SaveexecB32"
            | "SAndn2SaveexecB64"
            | "SAshrI32"
            | "SBcnt1I32B32"
            | "SBcnt1I32B64"
            | "SBfeI32"
            | "SBfeI64"
            | "SBfeU32"
            | "SBfeU64"
            | "SBfmB32"
            | "SBfmB64"
            | "SBitcmp0B32"
            | "SBitcmp1B32"
            | "SBrevB32"
            | "SCmpEqU64"
            | "SCmpLgU64"
            | "SCselectB32"
            | "SCselectB64"
            | "SGetpcB64"
            | "SLshl1AddU32"
            | "SLshl2AddU32"
            | "SLshl3AddU32"
            | "SLshl4AddU32"
            | "SLshlB32"
            | "SLshlB64"
            | "SLshrB32"
            | "SLshrB64"
            | "SMaxI32"
            | "SMaxU32"
            | "SMinI32"
            | "SMinU32"
            | "SMovB32"
            | "SMovB64"
            | "SMovkI32"
            | "SMulHiU32"
            | "SMulI32"
            | "SMulkI32"
            | "SNandB64"
            | "SNandSaveexecB32"
            | "SNandSaveexecB64"
            | "SNorB64"
            | "SNorSaveexecB32"
            | "SNorSaveexecB64"
            | "SNotB32"
            | "SNotB64"
            | "SOrB32"
            | "SOrB64"
            | "SOrSaveexecB32"
            | "SOrSaveexecB64"
            | "SOrn1SaveexecB32"
            | "SOrn1SaveexecB64"
            | "SOrn2B32"
            | "SOrn2B64"
            | "SOrn2SaveexecB32"
            | "SOrn2SaveexecB64"
            | "SPackLlB32B16"
            | "SSubI32"
            | "SSubU32"
            | "SSubbU32"
            | "SWqmB64"
            | "SXnorB64"
            | "SXnorSaveexecB32"
            | "SXnorSaveexecB64"
            | "SXorB32"
            | "SXorB64"
            | "SXorSaveexecB32"
            | "SXorSaveexecB64"
    ) || name.starts_with("SCmp")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Opcode, Operand};

    fn instruction(
        pc: u32,
        encoding: ShaderEncoding,
        name: &'static str,
        sources: Vec<crate::Operand>,
        destinations: Vec<crate::Operand>,
    ) -> Instruction {
        Instruction {
            pc,
            encoding,
            opcode: Opcode::from_static(name),
            words: vec![0],
            sources,
            destinations,
            control: None,
        }
    }

    #[test]
    fn variation_lattice_joins_toward_more_varying() {
        assert_eq!(
            Variation::KernelUniform.join(Variation::WorkgroupUniform),
            Variation::WorkgroupUniform
        );
        assert_eq!(
            Variation::GuestWaveUniform.join(Variation::Divergent),
            Variation::Divergent
        );
        assert!(Variation::GuestWaveUniform.is_uniform());
        assert!(!Variation::Unreachable.is_uniform());
    }

    #[test]
    fn salu_chain_is_wave_uniform_but_vector_output_is_divergent() {
        let program = Program::new(
            0x1000,
            vec![
                instruction(
                    0,
                    ShaderEncoding::Sop2,
                    "SAddU32",
                    vec![Operand::scalar(0), Operand::scalar(1)],
                    vec![Operand::scalar(2)],
                ),
                instruction(
                    4,
                    ShaderEncoding::Vop2,
                    "VAddU32",
                    vec![Operand::scalar(2), Operand::vector(0)],
                    vec![Operand::vector(1)],
                ),
                instruction(8, ShaderEncoding::Sopp, "SEndpgm", vec![], vec![]),
            ],
        );
        let analysis = analyze_uniformity(&program, &UniformityInputs::default());
        assert_eq!(
            analysis.definition(0, RegisterRef::Scalar(2)).unwrap().1,
            Variation::KernelUniform
        );
        assert_eq!(
            analysis.definition(1, RegisterRef::Vector(1)).unwrap().1,
            Variation::Divergent
        );
        assert_eq!(
            analysis.definition(0, RegisterRef::Scc).unwrap().1,
            Variation::KernelUniform
        );
    }

    #[test]
    fn cfg_join_makes_one_stable_phi_and_converges() {
        // s_cbranch_scc1 +1: block 0 branches to pc 8 or falls through pc 4. Both reach pc 12.
        let mut branch = instruction(0, ShaderEncoding::Sopp, "SCbranchScc1", vec![], vec![]);
        branch.words = vec![0xbf85_0001];
        let mut jump = instruction(4, ShaderEncoding::Sopp, "SBranch", vec![], vec![]);
        jump.words = vec![0xbf82_0001];
        let program = Program::new(
            0x1000,
            vec![
                branch,
                jump,
                instruction(
                    8,
                    ShaderEncoding::Sop1,
                    "SMovB32",
                    vec![Operand::scalar(0)],
                    vec![Operand::scalar(4)],
                ),
                instruction(
                    12,
                    ShaderEncoding::Sop1,
                    "SMovB32",
                    vec![Operand::scalar(4)],
                    vec![Operand::scalar(5)],
                ),
                instruction(16, ShaderEncoding::Sopp, "SEndpgm", vec![], vec![]),
            ],
        );
        let inputs = UniformityInputs {
            divergent_scalars: BTreeSet::from([0]),
            ..UniformityInputs::default()
        };
        let first = analyze_uniformity(&program, &inputs);
        let second = analyze_uniformity(&program, &inputs);
        let joined = first.input(3, RegisterRef::Scalar(4)).unwrap();
        assert_eq!(joined.1, Variation::Divergent);
        assert_eq!(joined, second.input(3, RegisterRef::Scalar(4)).unwrap());
    }

    #[test]
    fn implicit_branch_and_mask_effects_are_exhaustive() {
        let branch = instruction(0, ShaderEncoding::Sopp, "SCbranchExecz", vec![], vec![]);
        let semantics = classify_instruction(&branch);
        assert_eq!(semantics.uses, vec![RegisterRef::Exec]);

        let saveexec = instruction(
            4,
            ShaderEncoding::Sop1,
            "SAndSaveexecB64",
            vec![Operand::scalar(0)],
            vec![Operand::scalar(2)],
        );
        let semantics = classify_instruction(&saveexec);
        assert!(semantics.uses.contains(&RegisterRef::Exec));
        assert!(semantics.definitions.contains(&RegisterRef::Exec));
        assert!(semantics.definitions.contains(&RegisterRef::Scc));
    }

    #[test]
    fn known_unmodeled_opcode_is_a_barrier_and_poisons_downstream_proofs() {
        let program = Program::new(
            0x1000,
            vec![
                instruction(
                    0,
                    ShaderEncoding::Sop1,
                    "SMovB32",
                    vec![Operand::scalar(0)],
                    vec![Operand::scalar(1)],
                ),
                instruction(4, ShaderEncoding::Sop1, "SGetregB32", vec![], vec![]),
                instruction(
                    8,
                    ShaderEncoding::Sop1,
                    "SMovB32",
                    vec![Operand::scalar(1)],
                    vec![Operand::scalar(2)],
                ),
                instruction(12, ShaderEncoding::Sopp, "SEndpgm", vec![], vec![]),
            ],
        );
        let analysis = analyze_uniformity(&program, &UniformityInputs::default());
        let barrier = analysis.semantics(1).unwrap();
        assert!(!barrier.modeled);
        assert_eq!(barrier.effect, MemoryEffect::Unknown);
        assert_eq!(
            analysis.input(2, RegisterRef::Scalar(1)).unwrap().1,
            Variation::Divergent
        );
        assert_eq!(
            analysis.definition(2, RegisterRef::Scalar(2)).unwrap().1,
            Variation::Divergent
        );
    }
}
