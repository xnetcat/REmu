// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

//! Resource bindings discovered by shader evaluation and the evaluation result itself.

use smallvec::SmallVec;

use super::{ComputeSystemRegisters, ImageControl, Opcode};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ImageBinding {
    pub pc: u32,
    pub opcode: Opcode,
    pub control: ImageControl,
    pub resource_descriptor: SmallVec<[u32; 8]>,
    pub sampler_descriptor: SmallVec<[u32; 4]>,
    pub mip_level: Option<u32>,
}

impl ImageBinding {
    /// Returns whether two guest image instructions can use one native texture binding.
    ///
    /// Descriptor identity alone is insufficient: Metal encodes the component/dimension type and
    /// access mode in the function signature. Comparison sampling also requires a depth texture
    /// declaration, so it cannot alias an otherwise identical ordinary sampled view.
    #[must_use]
    pub fn shares_texture_slot_with(&self, other: &Self) -> bool {
        self.resource_descriptor == other.resource_descriptor
            && self.control.dimension == other.control.dimension
            && self.control.is_array == other.control.is_array
            && image_access_class(self.opcode) == image_access_class(other.opcode)
            && image_is_comparison(self.opcode) == image_is_comparison(other.opcode)
    }

    /// Returns whether two sampled-image instructions can use one native sampler binding.
    #[must_use]
    pub fn shares_sampler_slot_with(&self, other: &Self) -> bool {
        image_uses_sampler(self.opcode)
            && image_uses_sampler(other.opcode)
            && self.sampler_descriptor == other.sampler_descriptor
            && image_is_comparison(self.opcode) == image_is_comparison(other.opcode)
    }
}

/// Returns whether an image instruction performs a depth comparison.
///
/// RDNA2 names comparison samples and comparison gathers differently, so checking only the
/// `ImageSampleC*` family silently misclassifies the known `ImageGather4C*` opcodes.
#[must_use]
pub fn image_is_comparison(opcode: Opcode) -> bool {
    opcode.contains("SampleC") || opcode.contains("Gather4C")
}

fn image_access_class(opcode: Opcode) -> u8 {
    if opcode.starts_with("ImageStore") || opcode.starts_with("ImageAtomic") {
        2
    } else if opcode.starts_with("ImageLoad") {
        1
    } else {
        0
    }
}

fn image_uses_sampler(opcode: Opcode) -> bool {
    opcode.starts_with("ImageSample") || opcode.starts_with("ImageGather")
}

/// Dense native texture/sampler slots for the per-instruction image bindings in one evaluation.
///
/// The vectors retain one entry per image instruction while the counts describe the canonical
/// native resources. This preserves exact PC lookup without consuming a Metal resource slot for
/// every repeated use of the same descriptor.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ImageBindingSlotMap {
    pub texture_slots: Vec<u32>,
    pub sampler_slots: Vec<Option<u32>>,
    pub texture_count: u32,
    pub sampler_count: u32,
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

impl ShaderEvaluation {
    /// Canonicalizes image resources by descriptor, native type, and access semantics.
    #[must_use]
    pub fn image_binding_slots(&self) -> ImageBindingSlotMap {
        let mut texture_representatives = Vec::<usize>::new();
        let mut sampler_representatives = Vec::<usize>::new();
        let mut texture_slots = Vec::with_capacity(self.image_bindings.len());
        let mut sampler_slots = Vec::with_capacity(self.image_bindings.len());

        for (index, binding) in self.image_bindings.iter().enumerate() {
            let texture_slot = texture_representatives
                .iter()
                .position(|representative| {
                    binding.shares_texture_slot_with(&self.image_bindings[*representative])
                })
                .unwrap_or_else(|| {
                    let slot = texture_representatives.len();
                    texture_representatives.push(index);
                    slot
                });
            texture_slots
                .push(u32::try_from(texture_slot).expect("image texture slot count fits in u32"));

            let sampler_slot = image_uses_sampler(binding.opcode).then(|| {
                sampler_representatives
                    .iter()
                    .position(|representative| {
                        binding.shares_sampler_slot_with(&self.image_bindings[*representative])
                    })
                    .unwrap_or_else(|| {
                        let slot = sampler_representatives.len();
                        sampler_representatives.push(index);
                        slot
                    })
            });
            sampler_slots
                .push(sampler_slot.map(|slot| {
                    u32::try_from(slot).expect("image sampler slot count fits in u32")
                }));
        }

        ImageBindingSlotMap {
            texture_slots,
            sampler_slots,
            texture_count: u32::try_from(texture_representatives.len())
                .expect("image texture slot count fits in u32"),
            sampler_count: u32::try_from(sampler_representatives.len())
                .expect("image sampler slot count fits in u32"),
        }
    }
}
