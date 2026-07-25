// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::sync::Arc;

use smallvec::SmallVec;

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum ShaderEncoding {
    Sop1,
    Sop2,
    Sopc,
    Sopp,
    Sopk,
    Smrd,
    Smem,
    Mubuf,
    Mtbuf,
    Vop1,
    Vop2,
    Vopc,
    Vop3,
    Vintrp,
    Ds,
    Flat,
    Vop3p,
    Mimg,
    Exp,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum OperandKind {
    ScalarRegister,
    VectorRegister,
    EncodedConstant,
    LiteralConstant,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum ResourceKind {
    ReadOnlyTexture,
    ReadWriteTexture,
    Sampler,
    ConstantBuffer,
}

impl TryFrom<usize> for ResourceKind {
    type Error = ();

    fn try_from(value: usize) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::ReadOnlyTexture),
            1 => Ok(Self::ReadWriteTexture),
            2 => Ok(Self::Sampler),
            3 => Ok(Self::ConstantBuffer),
            _ => Err(()),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum PixelOutputKind {
    Float,
    Uint,
    Sint,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct PixelOutputBinding {
    pub guest_slot: u32,
    pub host_location: u32,
    pub kind: PixelOutputKind,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct ResourceMapping {
    pub kind: ResourceKind,
    pub slot: u32,
    pub offset_dwords: u32,
    pub size_flag: bool,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ShaderMetadata {
    pub extended_user_data_size_dwords: u32,
    pub shader_resource_table_size_dwords: u32,
    pub direct_resources: BTreeMap<u32, u32>,
    pub resources: Vec<ResourceMapping>,
}

#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub struct ComputeSystemRegisters {
    pub work_group_x: Option<u32>,
    pub work_group_y: Option<u32>,
    pub work_group_z: Option<u32>,
    pub thread_group_size: Option<u32>,
}

impl ComputeSystemRegisters {
    #[must_use]
    pub fn value_for_register(self, scalar_register: u32) -> Option<ComputeSystemValue> {
        if self.work_group_x == Some(scalar_register) {
            Some(ComputeSystemValue::WorkGroupX)
        } else if self.work_group_y == Some(scalar_register) {
            Some(ComputeSystemValue::WorkGroupY)
        } else if self.work_group_z == Some(scalar_register) {
            Some(ComputeSystemValue::WorkGroupZ)
        } else if self.thread_group_size == Some(scalar_register) {
            Some(ComputeSystemValue::ThreadGroupSize)
        } else {
            None
        }
    }

    pub fn clear_static_values(self, scalar_registers: &mut [u32]) {
        for register in [
            self.work_group_x,
            self.work_group_y,
            self.work_group_z,
            self.thread_group_size,
        ]
        .into_iter()
        .flatten()
        {
            if let Some(value) = scalar_registers.get_mut(register as usize) {
                *value = 0;
            }
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum ComputeSystemValue {
    WorkGroupX,
    WorkGroupY,
    WorkGroupZ,
    ThreadGroupSize,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct Operand {
    pub kind: OperandKind,
    pub value: u32,
}

impl Operand {
    #[must_use]
    pub const fn scalar(index: u32) -> Self {
        Self {
            kind: OperandKind::ScalarRegister,
            value: index,
        }
    }

    #[must_use]
    pub const fn vector(index: u32) -> Self {
        Self {
            kind: OperandKind::VectorRegister,
            value: index,
        }
    }

    #[must_use]
    pub const fn source(encoded: u32, literal: Option<u32>) -> Self {
        if encoded >= 256 {
            return Self::vector(encoded - 256);
        }
        if matches!(encoded, 249 | 255)
            && let Some(literal) = literal
        {
            return Self {
                kind: OperandKind::LiteralConstant,
                value: literal,
            };
        }
        if encoded <= 105 || matches!(encoded, 106 | 107 | 124 | 126 | 127) {
            return Self::scalar(encoded);
        }
        Self {
            kind: OperandKind::EncodedConstant,
            value: encoded,
        }
    }
}

impl fmt::Display for Operand {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.kind {
            OperandKind::ScalarRegister => write!(formatter, "s{}", self.value),
            OperandKind::VectorRegister => write!(formatter, "v{}", self.value),
            OperandKind::LiteralConstant => write!(formatter, "0x{:08X}", self.value),
            OperandKind::EncodedConstant => write!(formatter, "src[{}]", self.value),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum RawOpcodeFamily {
    Vop3,
    Vop3b,
    Vop3p,
    Mubuf,
}

/// A decoded opcode. Known operations retain their canonical name while architecturally framed
/// opcodes outside the semantic table retain their exact family and numeric code. Keeping those
/// values typed is important: diagnostics can distinguish an unknown opcode from corrupt framing,
/// and a later ISA-table expansion cannot change instruction boundaries.
#[derive(Clone, Copy, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum Opcode {
    Known(&'static str),
    Raw { family: RawOpcodeFamily, code: u16 },
}

impl Opcode {
    pub const END_PROGRAM: Self = Self::from_static("SEndpgm");
    pub const EXPORT: Self = Self::from_static("Exp");

    pub(crate) const fn from_static(name: &'static str) -> Self {
        Self::Known(name)
    }

    #[must_use]
    pub(crate) const fn raw(family: RawOpcodeFamily, code: u16) -> Self {
        Self::Raw { family, code }
    }

    #[must_use]
    pub const fn known_name(self) -> Option<&'static str> {
        match self {
            Self::Known(name) => Some(name),
            Self::Raw { .. } => None,
        }
    }

    #[must_use]
    pub fn is(self, name: &str) -> bool {
        self.known_name() == Some(name)
    }

    #[must_use]
    pub fn starts_with(self, prefix: &str) -> bool {
        self.known_name()
            .is_some_and(|name| name.starts_with(prefix))
    }

    #[must_use]
    pub fn contains(self, part: &str) -> bool {
        self.known_name().is_some_and(|name| name.contains(part))
    }
}

impl fmt::Debug for Opcode {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, formatter)
    }
}

impl fmt::Display for Opcode {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Known(name) => formatter.write_str(name),
            Self::Raw { family, code } => write!(formatter, "{family:?}[0x{code:X}]"),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
#[allow(clippy::struct_excessive_bools)]
pub struct ImageControl {
    pub dmask: u32,
    pub vector_address: u32,
    pub address_registers: Vec<u32>,
    pub vector_data: u32,
    pub scalar_resource: u32,
    pub scalar_sampler: u32,
    pub dimension: u32,
    pub is_array: bool,
    pub glc: bool,
    pub slc: bool,
    pub a16: bool,
    pub d16: bool,
}

impl ImageControl {
    #[must_use]
    pub fn address_register(&self, component: usize) -> u32 {
        self.address_registers
            .get(component)
            .copied()
            .unwrap_or_else(|| {
                self.vector_address
                    .saturating_add(u32::try_from(component).unwrap_or(u32::MAX))
            })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[allow(clippy::struct_excessive_bools)]
pub struct GlobalMemoryControl {
    pub dword_count: u32,
    pub vector_address: u32,
    pub vector_data: u32,
    pub vector_destination: u32,
    pub scalar_address: Option<u32>,
    pub offset_bytes: i32,
    pub dlc: bool,
    pub lds: bool,
    pub glc: bool,
    pub slc: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[allow(clippy::struct_excessive_bools)]
pub struct BufferMemoryControl {
    pub dword_count: u32,
    pub vector_address: u32,
    pub vector_data: u32,
    pub scalar_resource: u32,
    pub instruction_format: Option<u32>,
    pub offset_bytes: i32,
    pub index_enabled: bool,
    pub offset_enabled: bool,
    pub dlc: bool,
    pub lds: bool,
    pub glc: bool,
    pub slc: bool,
    pub tfe: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ExportControl {
    pub target: u32,
    pub enable_mask: u32,
    pub compressed: bool,
    pub done: bool,
    pub valid_mask: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct InterpolationControl {
    pub attribute: u32,
    pub channel: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Vop3Control {
    pub absolute_mask: u32,
    pub negate_mask: u32,
    pub output_modifier: u32,
    pub clamp: bool,
    pub operand_select: u32,
    pub scalar_destination: Option<u32>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SdwaControl {
    pub destination_select: u32,
    pub destination_unused: u32,
    pub source0_select: u32,
    pub source1_select: u32,
    pub source0_sign_extend: bool,
    pub source1_sign_extend: bool,
    pub absolute_mask: u32,
    pub negate_mask: u32,
    pub output_modifier: u32,
    pub clamp: bool,
    pub scalar_destination: Option<u32>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Vop3pControl {
    pub operand_select_mask: u32,
    pub operand_select_high_mask: u32,
    pub negate_low_mask: u32,
    pub negate_high_mask: u32,
    pub clamp: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DppControl {
    pub control: u32,
    pub fetch_inactive: bool,
    pub bound_control: bool,
    pub absolute_mask: u32,
    pub negate_mask: u32,
    pub bank_mask: u32,
    pub row_mask: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Dpp8Control {
    pub lane_selectors: u32,
    pub fetch_inactive: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ScalarMemoryControl {
    pub destination_count: u32,
    pub immediate_offset_bytes: i32,
    pub dynamic_offset_register: Option<u32>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DataShareControl {
    pub offset0: u32,
    pub offset1: u32,
    pub gds: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum InstructionControl {
    Image(ImageControl),
    GlobalMemory(GlobalMemoryControl),
    BufferMemory(BufferMemoryControl),
    Export(ExportControl),
    Interpolation(InterpolationControl),
    Vop3(Vop3Control),
    Sdwa(SdwaControl),
    Vop3p(Vop3pControl),
    Dpp(DppControl),
    Dpp8(Dpp8Control),
    ScalarMemory(ScalarMemoryControl),
    DataShare(DataShareControl),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ImageBinding {
    pub pc: u32,
    pub opcode: Opcode,
    pub control: ImageControl,
    pub resource_descriptor: SmallVec<[u32; 8]>,
    pub sampler_descriptor: SmallVec<[u32; 4]>,
    pub mip_level: Option<u32>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GlobalMemoryBinding {
    pub scalar_address: u32,
    pub base_address: u64,
    pub instruction_pcs: SmallVec<[u32; 8]>,
    /// Valid byte extent of the binding. This remains populated when `data` is intentionally
    /// empty because the backend consumes identity-mapped guest memory directly.
    pub byte_count: usize,
    pub data: Vec<u8>,
    pub writable: bool,
    pub write_back_to_guest: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct VertexInputBinding {
    pub pc: u32,
    pub location: u32,
    pub component_count: u32,
    pub data_format: u32,
    pub number_format: u32,
    pub base_address: u64,
    pub stride: u32,
    pub offset_bytes: u32,
    pub data: Vec<u8>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ShaderEvaluation {
    pub initial_scalar_registers: SmallVec<[u32; 128]>,
    pub scalar_registers: SmallVec<[u32; 128]>,
    /// Guest words consumed by scalar-memory instructions while discovering resources and
    /// control flow. Runtime caches must revalidate these dependencies because descriptor-table
    /// pointers can remain constant while Unity rewrites the descriptors behind them.
    pub scalar_memory_reads: Vec<(u64, u32)>,
    pub image_bindings: Vec<ImageBinding>,
    pub global_memory_bindings: Vec<GlobalMemoryBinding>,
    pub compute_system_registers: Option<ComputeSystemRegisters>,
    pub vertex_inputs: Vec<VertexInputBinding>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Instruction {
    pub pc: u32,
    pub encoding: ShaderEncoding,
    pub opcode: Opcode,
    pub words: Vec<u32>,
    pub sources: Vec<Operand>,
    pub destinations: Vec<Operand>,
    pub control: Option<InstructionControl>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Program {
    pub address: u64,
    pub instructions: Vec<Instruction>,
    pixel_color_export_masks: u32,
    next_evaluation_instruction_indices: Vec<u32>,
    scalar_memory_read_capacity_hint: usize,
    image_binding_capacity_hint: usize,
    global_binding_capacity_hint: usize,
    runtime_scalar_registers: BTreeSet<u32>,
    runtime_scalar_register_indices: Vec<u8>,
    input_scalar_registers: BTreeSet<u32>,
}

impl Program {
    #[must_use]
    pub fn new(address: u64, instructions: Vec<Instruction>) -> Self {
        let pixel_color_export_masks = instructions.iter().fold(0, |masks, instruction| {
            let Some(InstructionControl::Export(export)) = instruction.control else {
                return masks;
            };
            if export.target < 8 {
                masks | ((export.enable_mask & 0x0f) << (export.target * 4))
            } else {
                masks
            }
        });
        let mut runtime_scalar_registers = BTreeSet::from([106, 107, 124, 125, 126, 127]);
        let mut input_scalar_registers = BTreeSet::new();
        let mut defined_scalar_registers = BTreeSet::new();
        // A linear shader can safely exclude temporaries that are assigned before their first
        // use from the per-draw input identity. For shaders with scalar branches, retain the
        // conservative all-sources rule because a branch may bypass an apparent definition.
        let linear_scalar_flow = !instructions.iter().any(|instruction| {
            matches!(
                instruction.opcode.known_name(),
                Some(
                    "SBranch"
                        | "SCbranchScc0"
                        | "SCbranchScc1"
                        | "SCbranchVccz"
                        | "SCbranchVccnz"
                        | "SCbranchExecz"
                        | "SCbranchExecnz"
                        | "SSetpcB64"
                        | "SSwappcB64"
                )
            )
        });
        for instruction in &instructions {
            let evaluation_relevant = instruction_requires_evaluation(instruction);
            for operand in &instruction.sources {
                if operand.kind == OperandKind::ScalarRegister {
                    insert_scalar_range(&mut runtime_scalar_registers, operand.value, 2);
                    if evaluation_relevant {
                        insert_input_scalar_range(
                            &mut input_scalar_registers,
                            &defined_scalar_registers,
                            operand.value,
                            2,
                            linear_scalar_flow,
                        );
                    }
                }
            }
            match instruction.control.as_ref() {
                Some(InstructionControl::Image(image)) => {
                    insert_scalar_range(&mut runtime_scalar_registers, image.scalar_resource, 8);
                    insert_scalar_range(&mut runtime_scalar_registers, image.scalar_sampler, 4);
                    insert_input_scalar_range(
                        &mut input_scalar_registers,
                        &defined_scalar_registers,
                        image.scalar_resource,
                        8,
                        linear_scalar_flow,
                    );
                    insert_input_scalar_range(
                        &mut input_scalar_registers,
                        &defined_scalar_registers,
                        image.scalar_sampler,
                        4,
                        linear_scalar_flow,
                    );
                }
                Some(InstructionControl::GlobalMemory(global)) => {
                    if let Some(scalar_address) = global.scalar_address {
                        insert_scalar_range(&mut runtime_scalar_registers, scalar_address, 2);
                        insert_input_scalar_range(
                            &mut input_scalar_registers,
                            &defined_scalar_registers,
                            scalar_address,
                            2,
                            linear_scalar_flow,
                        );
                    }
                }
                Some(InstructionControl::BufferMemory(buffer)) => {
                    insert_scalar_range(&mut runtime_scalar_registers, buffer.scalar_resource, 4);
                    insert_input_scalar_range(
                        &mut input_scalar_registers,
                        &defined_scalar_registers,
                        buffer.scalar_resource,
                        4,
                        linear_scalar_flow,
                    );
                }
                Some(InstructionControl::ScalarMemory(scalar)) => {
                    if matches!(
                        instruction.encoding,
                        ShaderEncoding::Smem | ShaderEncoding::Smrd
                    ) && let Some(base) = instruction.sources.first()
                        && base.kind == OperandKind::ScalarRegister
                    {
                        insert_scalar_range(&mut runtime_scalar_registers, base.value, 4);
                        insert_input_scalar_range(
                            &mut input_scalar_registers,
                            &defined_scalar_registers,
                            base.value,
                            4,
                            linear_scalar_flow,
                        );
                    }
                    if let Some(register) = scalar.dynamic_offset_register {
                        insert_scalar_range(&mut runtime_scalar_registers, register, 2);
                        insert_input_scalar_range(
                            &mut input_scalar_registers,
                            &defined_scalar_registers,
                            register,
                            2,
                            linear_scalar_flow,
                        );
                    }
                }
                _ => {}
            }
            for operand in &instruction.destinations {
                if operand.kind == OperandKind::ScalarRegister {
                    insert_scalar_range(&mut runtime_scalar_registers, operand.value, 1);
                    if evaluation_relevant {
                        insert_scalar_range(&mut defined_scalar_registers, operand.value, 1);
                    }
                }
            }
        }
        let runtime_scalar_register_indices = runtime_scalar_registers
            .iter()
            .filter_map(|register| u8::try_from(*register).ok())
            .collect();
        let scalar_memory_read_capacity_hint = instructions
            .iter()
            .filter(|instruction| {
                matches!(
                    instruction.control,
                    Some(InstructionControl::ScalarMemory(_))
                )
            })
            .map(|instruction| instruction.destinations.len())
            .fold(0_usize, usize::saturating_add)
            .min(4_096);
        let image_binding_capacity_hint = instructions
            .iter()
            .filter(|instruction| matches!(instruction.control, Some(InstructionControl::Image(_))))
            .count()
            .min(256);
        let global_binding_capacity_hint = instructions
            .iter()
            .filter(|instruction| {
                matches!(
                    instruction.control,
                    Some(
                        InstructionControl::GlobalMemory(_)
                            | InstructionControl::BufferMemory(_)
                            | InstructionControl::ScalarMemory(_)
                    )
                )
            })
            .count()
            .min(256);
        // Resource discovery executes scalar control flow and descriptor-producing memory
        // operations, but it intentionally does not simulate vector ALU. Most production shaders
        // contain hundreds of vector-only instructions between the handful relevant here. Keep an
        // O(1) jump table so every dynamic evaluation skips each such run in one step.
        let mut next_evaluation_instruction_indices = vec![u32::MAX; instructions.len()];
        let mut next = u32::MAX;
        for (index, instruction) in instructions.iter().enumerate().rev() {
            if instruction_requires_evaluation(instruction) {
                next = u32::try_from(index).unwrap_or(u32::MAX);
            }
            next_evaluation_instruction_indices[index] = next;
        }
        Self {
            address,
            instructions,
            pixel_color_export_masks,
            next_evaluation_instruction_indices,
            scalar_memory_read_capacity_hint,
            image_binding_capacity_hint,
            global_binding_capacity_hint,
            runtime_scalar_registers,
            runtime_scalar_register_indices,
            input_scalar_registers,
        }
    }

    #[must_use]
    pub const fn pixel_color_export_masks(&self) -> u32 {
        self.pixel_color_export_masks
    }

    #[must_use]
    pub fn next_evaluation_instruction_index(&self, index: usize) -> Option<usize> {
        self.next_evaluation_instruction_indices
            .get(index)
            .copied()
            .filter(|index| *index != u32::MAX)
            .and_then(|index| usize::try_from(index).ok())
    }

    #[must_use]
    pub const fn scalar_memory_read_capacity_hint(&self) -> usize {
        self.scalar_memory_read_capacity_hint
    }

    #[must_use]
    pub const fn image_binding_capacity_hint(&self) -> usize {
        self.image_binding_capacity_hint
    }

    #[must_use]
    pub const fn global_binding_capacity_hint(&self) -> usize {
        self.global_binding_capacity_hint
    }

    pub fn image_resources(&self) -> impl Iterator<Item = &ImageControl> {
        self.instructions.iter().filter_map(|instruction| {
            let InstructionControl::Image(control) = instruction.control.as_ref()? else {
                return None;
            };
            Some(control)
        })
    }

    #[must_use]
    pub const fn runtime_scalar_registers(&self) -> &BTreeSet<u32> {
        &self.runtime_scalar_registers
    }

    #[must_use]
    pub fn runtime_scalar_register_indices(&self) -> &[u8] {
        &self.runtime_scalar_register_indices
    }

    /// Scalar registers whose initial values can affect resource discovery or scalar control
    /// flow. Vector-only arithmetic receives current scalar values through the runtime Metal
    /// buffer and therefore must not create false shader-evaluation cache identities.
    #[must_use]
    pub const fn input_scalar_registers(&self) -> &BTreeSet<u32> {
        &self.input_scalar_registers
    }
}

fn instruction_requires_evaluation(instruction: &Instruction) -> bool {
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

fn insert_scalar_range(registers: &mut BTreeSet<u32>, first: u32, count: u32) {
    for register in first..first.saturating_add(count) {
        if register < 256 {
            registers.insert(register);
        }
    }
}

fn insert_input_scalar_range(
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

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ShaderState {
    pub program: Arc<Program>,
    pub user_data: SmallVec<[u32; 64]>,
    pub compute_system_registers: Option<ComputeSystemRegisters>,
    pub user_data_scalar_register_base: u32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classifies_operands_without_conflating_literals_and_inline_constants() {
        assert_eq!(Operand::source(12, None), Operand::scalar(12));
        assert_eq!(Operand::source(256, None), Operand::vector(0));
        assert_eq!(
            Operand::source(249, Some(0xdead_beef)),
            Operand {
                kind: OperandKind::LiteralConstant,
                value: 0xdead_beef,
            }
        );
        assert_eq!(
            Operand::source(242, None),
            Operand {
                kind: OperandKind::EncodedConstant,
                value: 242,
            }
        );
    }

    #[test]
    fn computes_pixel_masks_and_runtime_scalar_registers_once() {
        let program = Program::new(
            0x1000,
            vec![
                Instruction {
                    pc: 0,
                    encoding: ShaderEncoding::Sop2,
                    opcode: Opcode::from_static("SAddU32"),
                    words: vec![0],
                    sources: vec![Operand::scalar(4), Operand::vector(1)],
                    destinations: vec![Operand::scalar(6)],
                    control: None,
                },
                Instruction {
                    pc: 4,
                    encoding: ShaderEncoding::Exp,
                    opcode: Opcode::EXPORT,
                    words: vec![0, 0],
                    sources: Vec::new(),
                    destinations: Vec::new(),
                    control: Some(InstructionControl::Export(ExportControl {
                        target: 2,
                        enable_mask: 0b1011,
                        compressed: false,
                        done: true,
                        valid_mask: false,
                    })),
                },
            ],
        );
        assert_eq!(program.pixel_color_export_masks(), 0x0b00);
        assert!(
            program
                .runtime_scalar_registers()
                .is_superset(&BTreeSet::from([4, 5, 6, 106, 107, 124, 125, 126, 127,]))
        );
        assert!(program.input_scalar_registers().contains(&4));
        assert!(!program.input_scalar_registers().contains(&6));
    }

    #[test]
    fn straight_line_temporaries_do_not_become_shader_inputs() {
        let program = Program::new(
            0x1000,
            vec![
                Instruction {
                    pc: 0,
                    encoding: ShaderEncoding::Sop1,
                    opcode: Opcode::from_static("SMovB32"),
                    words: vec![0],
                    sources: vec![Operand::scalar(4)],
                    destinations: vec![Operand::scalar(6)],
                    control: None,
                },
                Instruction {
                    pc: 4,
                    encoding: ShaderEncoding::Sop2,
                    opcode: Opcode::from_static("SAddU32"),
                    words: vec![0],
                    sources: vec![Operand::scalar(6), Operand::scalar(8)],
                    destinations: vec![Operand::scalar(10)],
                    control: None,
                },
            ],
        );
        assert!(program.input_scalar_registers().contains(&4));
        assert!(program.input_scalar_registers().contains(&8));
        assert!(!program.input_scalar_registers().contains(&6));
        assert!(!program.input_scalar_registers().contains(&10));
    }

    #[test]
    fn vector_only_scalar_sources_do_not_become_evaluation_inputs() {
        let program = Program::new(
            0x1000,
            vec![Instruction {
                pc: 0,
                encoding: ShaderEncoding::Vop2,
                opcode: Opcode::from_static("VAddF32"),
                words: vec![0],
                sources: vec![Operand::scalar(4), Operand::vector(1)],
                destinations: vec![Operand::vector(0)],
                control: None,
            }],
        );
        assert!(program.runtime_scalar_registers().contains(&4));
        assert!(!program.input_scalar_registers().contains(&4));
    }

    #[test]
    fn branched_programs_keep_conservative_shader_inputs() {
        let program = Program::new(
            0x1000,
            vec![
                Instruction {
                    pc: 0,
                    encoding: ShaderEncoding::Sop1,
                    opcode: Opcode::from_static("SMovB32"),
                    words: vec![0],
                    sources: vec![Operand::scalar(4)],
                    destinations: vec![Operand::scalar(6)],
                    control: None,
                },
                Instruction {
                    pc: 4,
                    encoding: ShaderEncoding::Sopp,
                    opcode: Opcode::from_static("SCbranchScc0"),
                    words: vec![0],
                    sources: Vec::new(),
                    destinations: Vec::new(),
                    control: None,
                },
                Instruction {
                    pc: 8,
                    encoding: ShaderEncoding::Sop2,
                    opcode: Opcode::from_static("SAddU32"),
                    words: vec![0],
                    sources: vec![Operand::scalar(6), Operand::scalar(8)],
                    destinations: vec![Operand::scalar(10)],
                    control: None,
                },
            ],
        );
        assert!(program.input_scalar_registers().contains(&6));
    }
}
