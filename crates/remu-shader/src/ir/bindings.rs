// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

//! Resource bindings discovered by shader evaluation and the evaluation result itself.

use smallvec::SmallVec;

use super::{ComputeSystemRegisters, ImageControl, Opcode};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ImageBinding {
    pub pc: u32,
    /// The descriptor's index chain passes through registers only unevaluated VALU
    /// instructions write (readlane family, carry-outs): the snapshot below points at a
    /// stale heap slot and the binding must resolve through a runtime heap texture table.
    pub descriptor_index_is_runtime: bool,
    pub opcode: Opcode,
    pub control: ImageControl,
    pub resource_descriptor: SmallVec<[u32; 8]>,
    pub sampler_descriptor: SmallVec<[u32; 4]>,
    pub mip_level: Option<u32>,
}

/// Returns whether a GFX10 image SRD names no resource.
///
/// The image base is a 40-bit field split across DWORD0 and DWORD1[7:0]. Descriptor builders may
/// leave format, dimension, and destination-select metadata populated when that base is zero.
#[must_use]
pub fn image_resource_descriptor_is_null(words: &[u32]) -> bool {
    words.len() >= 2 && words[0] == 0 && (words[1] & 0xff) == 0
}

/// Native texture type carried in descriptor DWORD3 bits 31:28.
///
/// The field and values are the SDK's `Agc::Core::Texture`: `Type m_type : 4` closes DWORD3
/// (agc/core/texture.h:249) and the values are k1d=8 through k2dArrayMsaa=15
/// (agc/core/texture.h:54-63). Values below 8 do not name a valid texture descriptor.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum TextureDescriptorType {
    OneD,
    TwoD,
    ThreeD,
    Cube,
    OneDArray,
    TwoDArray,
    TwoDMsaa,
    TwoDMsaaArray,
}

/// The native signature shape an image instruction addresses, derived from the MIMG `DIM` field.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum ImageSignatureClass {
    OneD,
    TwoD,
    ThreeD,
    Cube,
    OneDArray,
    TwoDArray,
    TwoDMsaa,
    TwoDMsaaArray,
}

/// How the bound descriptor's native type must be presented to satisfy the instruction's
/// coordinate model. The instruction DIM decides the signature; the descriptor TYPE decides the
/// stored layout; a backend resolves any disagreement with a texture view — or refuses loudly.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum ImageViewReinterpretation {
    /// Descriptor TYPE and instruction DIM agree (or no live descriptor constrains the shape).
    Direct,
    /// A cubemap addressed with 2D-array coordinates: its six-faces-per-slice array layout
    /// (the SDK's array view on `kCubemap`, agc/core/texture.h:321).
    CubeFacesAsArray,
    /// A 2D-array descriptor sampled with cube coordinates: re-view six slices as a cube.
    ArrayAsCube,
    /// A non-array access over an array descriptor: the base slice.
    ArrayBaseSlice,
    /// An array access over a single-layer descriptor: a one-element array view.
    SingleLayerAsArray,
    /// No coherent view exists; the backend must refuse this draw loudly, never guess.
    Incompatible,
}

impl ImageBinding {
    /// Returns whether this binding's image SRD names no resource.
    #[must_use]
    pub fn resource_descriptor_is_null(&self) -> bool {
        image_resource_descriptor_is_null(&self.resource_descriptor)
    }

    /// The descriptor's native texture type, when a live descriptor with a valid TYPE nibble is
    /// bound (DWORD3 bits 31:28, agc/core/texture.h:249; values texture.h:54-63).
    #[must_use]
    pub fn descriptor_type(&self) -> Option<TextureDescriptorType> {
        if self.resource_descriptor_is_null() {
            return None;
        }
        match self.resource_descriptor.get(3)? >> 28 {
            8 => Some(TextureDescriptorType::OneD),
            9 => Some(TextureDescriptorType::TwoD),
            10 => Some(TextureDescriptorType::ThreeD),
            11 => Some(TextureDescriptorType::Cube),
            12 => Some(TextureDescriptorType::OneDArray),
            13 => Some(TextureDescriptorType::TwoDArray),
            14 => Some(TextureDescriptorType::TwoDMsaa),
            15 => Some(TextureDescriptorType::TwoDMsaaArray),
            _ => None,
        }
    }

    /// The signature shape the instruction's MIMG `DIM` field addresses.
    #[must_use]
    pub fn signature_class(&self) -> ImageSignatureClass {
        match self.control.dimension {
            0 => ImageSignatureClass::OneD,
            2 => ImageSignatureClass::ThreeD,
            3 => ImageSignatureClass::Cube,
            4 => ImageSignatureClass::OneDArray,
            5 => ImageSignatureClass::TwoDArray,
            6 => ImageSignatureClass::TwoDMsaa,
            7 => ImageSignatureClass::TwoDMsaaArray,
            _ => ImageSignatureClass::TwoD,
        }
    }

    /// Reconciles the descriptor's native TYPE against the instruction's DIM signature.
    #[must_use]
    pub fn view_reinterpretation(&self) -> ImageViewReinterpretation {
        let Some(descriptor_type) = self.descriptor_type() else {
            return ImageViewReinterpretation::Direct;
        };
        let native_class = match descriptor_type {
            TextureDescriptorType::OneD => ImageSignatureClass::OneD,
            TextureDescriptorType::TwoD => ImageSignatureClass::TwoD,
            TextureDescriptorType::ThreeD => ImageSignatureClass::ThreeD,
            TextureDescriptorType::Cube => ImageSignatureClass::Cube,
            TextureDescriptorType::OneDArray => ImageSignatureClass::OneDArray,
            TextureDescriptorType::TwoDArray => ImageSignatureClass::TwoDArray,
            TextureDescriptorType::TwoDMsaa => ImageSignatureClass::TwoDMsaa,
            TextureDescriptorType::TwoDMsaaArray => ImageSignatureClass::TwoDMsaaArray,
        };
        let signature = self.signature_class();
        if native_class == signature {
            return ImageViewReinterpretation::Direct;
        }
        match (descriptor_type, signature) {
            // A cubemap is stored as six slices per element; both the array view of a cube and a
            // cube view of an array are exact layout reinterpretations.
            (TextureDescriptorType::Cube, ImageSignatureClass::TwoDArray) => {
                ImageViewReinterpretation::CubeFacesAsArray
            }
            (TextureDescriptorType::TwoDArray, ImageSignatureClass::Cube) => {
                ImageViewReinterpretation::ArrayAsCube
            }
            // Dimension-preserving array/non-array disagreement selects the base slice or wraps
            // the single layer.
            (TextureDescriptorType::OneDArray, ImageSignatureClass::OneD)
            | (
                TextureDescriptorType::TwoDArray | TextureDescriptorType::Cube,
                ImageSignatureClass::TwoD,
            )
            | (TextureDescriptorType::TwoDMsaaArray, ImageSignatureClass::TwoDMsaa) => {
                ImageViewReinterpretation::ArrayBaseSlice
            }
            (TextureDescriptorType::OneD, ImageSignatureClass::OneDArray)
            | (TextureDescriptorType::TwoD, ImageSignatureClass::TwoDArray)
            | (TextureDescriptorType::TwoDMsaa, ImageSignatureClass::TwoDMsaaArray) => {
                ImageViewReinterpretation::SingleLayerAsArray
            }
            _ => ImageViewReinterpretation::Incompatible,
        }
    }

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

/// The address-operand modifiers a MIMG sample or gather opcode names.
///
/// GFX10 spells the modifier lattice into the mnemonic, and the spelling is case-significant in a
/// way that defeats substring matching: `Cl` is a LOD clamp, `CL` is a comparison plus an explicit
/// LOD, `Cd` is a coarse derivative and `CD` is a comparison plus fine derivatives. Only an exact
/// name decides which operands an instruction carries.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub struct ImageSampleModifiers {
    /// `O`: a packed signed texel offset dword ahead of every other address operand.
    pub offset: bool,
    /// `B`: a LOD bias float.
    pub bias: bool,
    /// `C`: a depth comparison reference float.
    pub compare: bool,
    /// `D` (fine) or `Cd` (coarse) derivatives: a pair of gradient vectors.
    pub gradients: bool,
    /// `Cl`: the trailing minimum-LOD clamp float, never a comparison.
    pub lod_clamp: bool,
    /// `L`: the trailing explicit LOD float.
    pub lod: bool,
    /// `Lz`: LOD zero, which carries no address operand at all.
    pub zero_lod: bool,
}

/// Returns the address-operand modifiers of a sampled-image opcode, or `None` for any opcode
/// that is not a MIMG sample or gather.
///
/// The table is exact because the GFX10 mnemonics alias under prefix or substring tests:
/// `ImageSampleCl` (LOD clamp), `ImageSampleCL` (compare plus LOD), `ImageSampleCd` (coarse
/// derivatives) and `ImageSampleCD` (compare plus derivatives) all contain `SampleC`, yet only
/// two of them compare. Opcode numbers are the public RDNA2 MIMG lattice decoded in
/// `decoder::memory`; the operand set of each is the LLVM `AMDGPUDimProfile` address list
/// (offset, bias, zcompare, gradients, coordinates, then one trailing LOD-or-clamp dword).
#[must_use]
pub fn image_sample_modifiers(opcode: Opcode) -> Option<ImageSampleModifiers> {
    const OFFSET: u8 = 1 << 0;
    const BIAS: u8 = 1 << 1;
    const COMPARE: u8 = 1 << 2;
    const GRADIENTS: u8 = 1 << 3;
    const LOD_CLAMP: u8 = 1 << 4;
    const LOD: u8 = 1 << 5;
    const ZERO_LOD: u8 = 1 << 6;

    let flags = match opcode.known_name()? {
        // IMAGE_SAMPLE, OP 0x20-0x2f.
        "ImageSample" => 0,
        "ImageSampleCl" => LOD_CLAMP,
        "ImageSampleD" => GRADIENTS,
        "ImageSampleDCl" => GRADIENTS | LOD_CLAMP,
        "ImageSampleL" => LOD,
        "ImageSampleB" => BIAS,
        "ImageSampleBCl" => BIAS | LOD_CLAMP,
        "ImageSampleLz" => ZERO_LOD,
        "ImageSampleC" => COMPARE,
        "ImageSampleCCl" => COMPARE | LOD_CLAMP,
        "ImageSampleCD" => COMPARE | GRADIENTS,
        "ImageSampleCDCl" => COMPARE | GRADIENTS | LOD_CLAMP,
        "ImageSampleCL" => COMPARE | LOD,
        "ImageSampleCB" => COMPARE | BIAS,
        "ImageSampleCBCl" => COMPARE | BIAS | LOD_CLAMP,
        "ImageSampleCLz" => COMPARE | ZERO_LOD,
        // The same lattice with a packed texel offset, OP 0x30-0x3f.
        "ImageSampleO" => OFFSET,
        "ImageSampleClO" => OFFSET | LOD_CLAMP,
        "ImageSampleDO" => OFFSET | GRADIENTS,
        "ImageSampleDClO" => OFFSET | GRADIENTS | LOD_CLAMP,
        "ImageSampleLO" => OFFSET | LOD,
        "ImageSampleBO" => OFFSET | BIAS,
        "ImageSampleBClO" => OFFSET | BIAS | LOD_CLAMP,
        "ImageSampleLzO" => OFFSET | ZERO_LOD,
        "ImageSampleCO" => OFFSET | COMPARE,
        "ImageSampleCClO" => OFFSET | COMPARE | LOD_CLAMP,
        "ImageSampleCDO" => OFFSET | COMPARE | GRADIENTS,
        "ImageSampleCDClO" => OFFSET | COMPARE | GRADIENTS | LOD_CLAMP,
        "ImageSampleCLO" => OFFSET | COMPARE | LOD,
        "ImageSampleCBO" => OFFSET | COMPARE | BIAS,
        "ImageSampleCBClO" => OFFSET | COMPARE | BIAS | LOD_CLAMP,
        "ImageSampleCLzO" => OFFSET | COMPARE | ZERO_LOD,
        // Coarse-derivative samples, OP 0x68-0x6f. `Cd` is a gradient modifier: only the `CCd`
        // forms compare.
        "ImageSampleCd" => GRADIENTS,
        "ImageSampleCdCl" => GRADIENTS | LOD_CLAMP,
        "ImageSampleCCd" => COMPARE | GRADIENTS,
        "ImageSampleCCdCl" => COMPARE | GRADIENTS | LOD_CLAMP,
        "ImageSampleCdO" => OFFSET | GRADIENTS,
        "ImageSampleCdClO" => OFFSET | GRADIENTS | LOD_CLAMP,
        "ImageSampleCCdO" => OFFSET | COMPARE | GRADIENTS,
        "ImageSampleCCdClO" => OFFSET | COMPARE | GRADIENTS | LOD_CLAMP,
        // IMAGE_GATHER4, OP 0x40-0x5f. Gathers have no derivative forms.
        "ImageGather4" => 0,
        "ImageGather4Cl" => LOD_CLAMP,
        "ImageGather4L" => LOD,
        "ImageGather4B" => BIAS,
        "ImageGather4BCl" => BIAS | LOD_CLAMP,
        "ImageGather4Lz" => ZERO_LOD,
        "ImageGather4C" => COMPARE,
        "ImageGather4CCl" => COMPARE | LOD_CLAMP,
        "ImageGather4CL" => COMPARE | LOD,
        "ImageGather4CB" => COMPARE | BIAS,
        "ImageGather4CBCl" => COMPARE | BIAS | LOD_CLAMP,
        "ImageGather4CLz" => COMPARE | ZERO_LOD,
        "ImageGather4O" => OFFSET,
        "ImageGather4ClO" => OFFSET | LOD_CLAMP,
        "ImageGather4LzO" => OFFSET | ZERO_LOD,
        "ImageGather4CO" => OFFSET | COMPARE,
        "ImageGather4CLzO" => OFFSET | COMPARE | ZERO_LOD,
        _ => return None,
    };
    Some(ImageSampleModifiers {
        offset: flags & OFFSET != 0,
        bias: flags & BIAS != 0,
        compare: flags & COMPARE != 0,
        gradients: flags & GRADIENTS != 0,
        lod_clamp: flags & LOD_CLAMP != 0,
        lod: flags & LOD != 0,
        zero_lod: flags & ZERO_LOD != 0,
    })
}

/// Returns whether an image instruction performs a depth comparison.
///
/// A depth comparison decides the native texture declaration and the address layout, so it is
/// answered from the exact opcode lattice rather than a `SampleC`/`Gather4C` substring: the
/// LOD-clamp (`ImageSampleCl`, `ImageGather4Cl`) and coarse-derivative (`ImageSampleCd`) families
/// contain that substring while comparing nothing.
#[must_use]
pub fn image_is_comparison(opcode: Opcode) -> bool {
    image_sample_modifiers(opcode).is_some_and(|modifiers| modifiers.compare)
}

fn image_access_class(opcode: Opcode) -> u8 {
    if opcode.starts_with("ImageStore") || opcode.starts_with("ImageAtomic") {
        2
    } else {
        u8::from(opcode.starts_with("ImageLoad"))
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
    ///
    /// # Panics
    ///
    /// Panics only if one evaluation contains more than `u32::MAX` distinct native image or
    /// sampler bindings, which cannot be represented by the public slot-map type.
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
