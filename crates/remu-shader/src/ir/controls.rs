// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

//! Per-encoding instruction control payloads carried alongside a decoded instruction.

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
