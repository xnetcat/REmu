// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

//! MIMG lowering: image loads, stores and sample operations.

use remu_shader::{
    ImageBinding, ImageControl, Instruction, InstructionControl, Opcode, decode_unified_format,
    image_is_comparison,
};

use crate::Compiler;

#[derive(Clone, Copy)]
pub(crate) struct ImageKind {
    pub(crate) component: &'static str,
    pub(crate) storage: bool,
    pub(crate) writable: bool,
    pub(crate) comparison: bool,
    pub(crate) dimension: u32,
    pub(crate) texture_type: u32,
}

/// Metal texture dimensionality after Kyty-style SRD rematerialization.
///
/// MIMG `DIM` encodes the guest address layout, while the image SRD `TYPE` field selects the
/// actual resource. Unreal Slate font atlases commonly issue DIM=3D samples (`dimension == 2`)
/// against TYPE=2D descriptors; binding those as `texture3d` yields solid tofu glyphs. Prefer
/// the live descriptor type when it is a known Prospero 2D/3D/array encoding.
pub(crate) fn metal_image_dimension(kind: ImageKind) -> u32 {
    // Prospero SQ_IMG TYPE (dwords[3]>>28), matching Kyty DescriptorDimension:
    // 1D/2D/2D_MSAA → Dim2D; 3D → Dim3D; cube/arrays → Dim2DArray.
    match kind.texture_type {
        8 | 9 | 14 => 1,
        10 => 2,
        11 | 12 | 13 | 15 => 3,
        _ => kind.dimension,
    }
}

const fn image_address_register(image: &ImageControl, component: usize) -> usize {
    if image.a16 { component / 2 } else { component }
}

const fn image_full_address_slots(image: &ImageControl) -> usize {
    if image.a16 { 2 } else { 1 }
}

const fn align_full_image_address(image: &ImageControl, component: usize) -> usize {
    if image.a16 {
        (component + 1) & !1
    } else {
        component
    }
}

pub(crate) fn is_storage_image(opcode: Opcode) -> bool {
    opcode.starts_with("ImageLoad") || is_writable_image(opcode)
}

pub(crate) fn is_writable_image(opcode: Opcode) -> bool {
    opcode.starts_with("ImageStore") || opcode.starts_with("ImageAtomic")
}

pub(crate) fn image_component_type(binding: &ImageBinding) -> &'static str {
    let Some(word) = binding.resource_descriptor.get(1) else {
        return "float";
    };
    let unified_format = (word >> 20) & 0x1ff;
    let Some((_, number_format)) = decode_unified_format(unified_format) else {
        return "float";
    };
    match number_format {
        4 => "uint",
        5 => "int",
        _ => "float",
    }
}

impl Compiler<'_> {
    pub(crate) fn emit_image(&mut self, instruction: &Instruction) -> Result<(), String> {
        let Some(InstructionControl::Image(image)) = instruction.control.as_ref() else {
            return Err("missing image control".to_owned());
        };
        let evaluation_index = self
            .evaluation
            .image_bindings
            .iter()
            .position(|binding| binding.pc == instruction.pc)
            .ok_or_else(|| "image instruction has no evaluated image binding".to_owned())?;
        let binding_index = usize::try_from(self.image_slots.texture_slots[evaluation_index])
            .map_err(|_| "canonical image texture slot does not fit usize".to_owned())?;
        let kind = self.image_kinds[binding_index];
        let name = instruction
            .opcode
            .known_name()
            .ok_or_else(|| "raw image opcode".to_owned())?;
        if name.starts_with("ImageLoad") {
            return self.emit_image_load(image, name, binding_index);
        }
        if name.starts_with("ImageStore") {
            return self.emit_image_store(image, name, binding_index);
        }
        if name == "ImageGetResinfo" {
            return self.emit_image_get_resinfo(image, binding_index);
        }
        let gather = name.starts_with("ImageGather4");
        if !name.starts_with("ImageSample") && !gather {
            return Err(format!("unsupported image opcode {name}"));
        }
        if kind.storage {
            return Err("sample operation resolved to a storage image".to_owned());
        }
        let native_2d_array = image.dimension == 5 && kind.texture_type == 13;
        if (image.is_array && !native_2d_array)
            || (!native_2d_array && !matches!(image.dimension, 0 | 1 | 2 | 3))
        {
            return Err(format!(
                "sampled image dimension {} array={} requires its matching Metal texture type",
                image.dimension, image.is_array
            ));
        }
        let sampler_slot = self.image_slots.sampler_slots[evaluation_index]
            .ok_or_else(|| "sampled image has no sampler slot".to_owned())?;
        let sampled =
            self.emit_image_sample(image, instruction.opcode, name, binding_index, sampler_slot)?;
        if gather {
            for component in 0..4 {
                let raw = if kind.component == "uint" {
                    format!("{sampled}[{component}]")
                } else {
                    format!("as_type<uint>({sampled}[{component}])")
                };
                self.store_vector(image.vector_data + component, &raw);
            }
            return Ok(());
        }
        let mut output = 0_u32;
        for component in 0..4 {
            if image.dmask & (1 << component) == 0 {
                continue;
            }
            let raw = if kind.component == "uint" {
                format!("{sampled}[{component}]")
            } else {
                format!("as_type<uint>({sampled}[{component}])")
            };
            self.store_vector(image.vector_data + output, &raw);
            output += 1;
        }
        Ok(())
    }

    fn emit_image_get_resinfo(
        &mut self,
        image: &ImageControl,
        binding_index: usize,
    ) -> Result<(), String> {
        let mip = self.temp("uint", &self.image_uint_address(image, 0));
        let kind = self.image_kinds[binding_index];
        let width = self.temp("uint", &format!("tex{binding_index}.get_width({mip})"));
        let height = self.temp("uint", &format!("tex{binding_index}.get_height({mip})"));
        // AMD Z is depth for 3D, otherwise array/cube layer count (or 1 for plain 2D).
        let metal_dim = metal_image_dimension(kind);
        let depth = if metal_dim == 2 {
            self.temp("uint", &format!("tex{binding_index}.get_depth({mip})"))
        } else if metal_dim == 3 || image.is_array {
            self.temp("uint", &format!("tex{binding_index}.get_array_size()"))
        } else {
            self.temp("uint", "1u")
        };
        let levels = self.temp(
            "uint",
            &format!("tex{binding_index}.get_num_mip_levels()"),
        );
        let components = [width, height, depth, levels];
        let mut values = Vec::with_capacity(4);
        for (component, value) in components.iter().enumerate() {
            if image.dmask & (1 << component) == 0 {
                continue;
            }
            if image.unorm {
                values.push(format!("as_type<uint>((float){value})"));
            } else {
                values.push(value.clone());
            }
        }
        if image.d16 {
            for (index, pair) in values.chunks(2).enumerate() {
                let low = &pair[0];
                let high = pair.get(1).map_or("0u", String::as_str);
                let packed = self.pack_image_d16("uint", low, high);
                self.store_vector(
                    image.vector_data
                        + u32::try_from(index).map_err(|_| "D16 resinfo output overflow")?,
                    &packed,
                );
            }
        } else {
            for (index, raw) in values.iter().enumerate() {
                self.store_vector(
                    image.vector_data
                        + u32::try_from(index).map_err(|_| "resinfo output overflow")?,
                    raw,
                );
            }
        }
        Ok(())
    }

    fn image_destination_select(&mut self, image: &ImageControl) -> String {
        let descriptor_word = format!("s[{}] & 0xFFFu", image.scalar_resource + 3);
        // A zeroed descriptor is used by a few partially initialized Unity paths; retain the
        // architectural XYZW identity fallback while keeping real component selection dynamic.
        // The latter prevents render-material swizzles from creating new MSL functions.
        self.temp(
            "uint",
            &format!("({descriptor_word}) == 0u ? 0xFACu : ({descriptor_word})"),
        )
    }

    fn emit_image_store(
        &mut self,
        image: &ImageControl,
        _name: &str,
        binding_index: usize,
    ) -> Result<(), String> {
        // GFX10 DIM describes the address operand layout, while the descriptor selects the actual
        // Metal resource view. Unreal can execute DIM=2/5 instructions through a single-layer 2D
        // descriptor; consume the extra address only when that descriptor really is a 3D view.
        let image_kind = self.image_kinds[binding_index];
        let is_3d = image_kind.texture_type == 10;
        if !matches!(image.dimension, 0 | 1 | 2 | 4 | 5) {
            return Err(format!(
                "storage image dimension {} requires a non-2D Metal texture type",
                image.dimension
            ));
        }
        let raw_x = self.image_uint_address(image, 0);
        let raw_y = if matches!(image.dimension, 0 | 4) {
            "0u".to_owned()
        } else {
            self.image_uint_address(image, 1)
        };
        let x = self.temp("int", &format!("as_type<int>({raw_x})"));
        let y = self.temp("int", &format!("as_type<int>({raw_y})"));
        let z = is_3d.then(|| {
            let raw_z = self.image_uint_address(image, 2);
            self.temp("int", &format!("as_type<int>({raw_z})"))
        });
        let kind = image_kind.component;
        let mut logical_components = Vec::with_capacity(4);
        let mut input = 0_u32;
        for component in 0..4 {
            if image.dmask & (1 << component) == 0 {
                logical_components.push(if kind == "float" { "0.0f" } else { "0" }.to_owned());
                continue;
            }
            let raw = self.image_store_raw_component(image, kind, input);
            logical_components.push(match kind {
                "uint" => raw,
                "int" => format!("as_type<int>({raw})"),
                "float" => format!("as_type<float>({raw})"),
                _ => unreachable!("image component type is selected from a closed set"),
            });
            input += 1;
        }
        let z_bounds = z.as_ref().map_or_else(String::new, |z| {
            format!(" && {z} >= 0 && {z} < (int)tex{binding_index}.get_depth()")
        });
        let coordinates = z.as_ref().map_or_else(
            || format!("uint2((uint){x}, (uint){y})"),
            |z| format!("uint3((uint){x}, (uint){y}, (uint){z})"),
        );
        self.line(3, &format!("if (exec && {x} >= 0 && {y} >= 0 && {x} < (int)tex{binding_index}.get_width() && {y} < (int)tex{binding_index}.get_height(){z_bounds}) {{"));
        self.line(
            4,
            &format!(
                "tex{binding_index}.write(vec<{kind}, 4>({}, {}, {}, {}), {coordinates});",
                logical_components[0],
                logical_components[1],
                logical_components[2],
                logical_components[3]
            ),
        );
        self.line(3, "}");
        Ok(())
    }

    fn emit_image_load(
        &mut self,
        image: &ImageControl,
        _name: &str,
        binding_index: usize,
    ) -> Result<(), String> {
        let image_kind = self.image_kinds[binding_index];
        let is_3d = image_kind.texture_type == 10;
        if !matches!(image.dimension, 0 | 1 | 2 | 4 | 5) {
            return Err(format!(
                "storage image dimension {} requires a non-2D Metal texture type",
                image.dimension
            ));
        }
        let raw_x = self.image_uint_address(image, 0);
        let raw_y = if matches!(image.dimension, 0 | 4) {
            "0u".to_owned()
        } else {
            self.image_uint_address(image, 1)
        };
        let x = self.temp("int", &format!("as_type<int>({raw_x})"));
        let y = self.temp("int", &format!("as_type<int>({raw_y})"));
        let z = is_3d.then(|| {
            let raw_z = self.image_uint_address(image, 2);
            self.temp("int", &format!("as_type<int>({raw_z})"))
        });
        let z_bounds = z.as_ref().map_or_else(String::new, |z| {
            format!(" && {z} >= 0 && {z} < (int)tex{binding_index}.get_depth()")
        });
        let coordinates = z.as_ref().map_or_else(
            || format!("uint2((uint){x}, (uint){y})"),
            |z| format!("uint3((uint){x}, (uint){y}, (uint){z})"),
        );
        let kind = self.image_kinds[binding_index].component;
        let zero = if kind == "float" { "0.0f" } else { "0" };
        let loaded = self.temp(
            &format!("vec<{kind}, 4>"),
            &format!("vec<{kind}, 4>({zero})"),
        );
        self.line(
            3,
            &format!(
                "if (exec && {x} >= 0 && {y} >= 0 && {x} < (int)tex{binding_index}.get_width() && {y} < (int)tex{binding_index}.get_height(){z_bounds}) {loaded} = tex{binding_index}.read({coordinates});"
            ),
        );
        let destination_select = self.image_destination_select(image);
        let mut values = Vec::with_capacity(4);
        for component in 0..4 {
            if image.dmask & (1 << component) == 0 {
                continue;
            }
            let kind = self.image_kinds[binding_index];
            let selector = format!("(({destination_select} >> {}u) & 7u)", component * 3);
            let one = if kind.component == "float" {
                "0x3F800000u"
            } else {
                "1u"
            };
            let physical = match kind.component {
                "uint" => format!("{loaded}[{selector} - 4u]"),
                "int" | "float" => {
                    format!("as_type<uint>({loaded}[{selector} - 4u])")
                }
                _ => unreachable!("image component type is selected from a closed set"),
            };
            let raw = format!(
                "({selector} == 1u ? {one} : ({selector} >= 4u && {selector} <= 7u ? {physical} : 0u))"
            );
            values.push(raw);
        }
        if image.d16 {
            for (index, pair) in values.chunks(2).enumerate() {
                let low = &pair[0];
                let high = pair.get(1).map_or("0u", String::as_str);
                let packed =
                    self.pack_image_d16(self.image_kinds[binding_index].component, low, high);
                self.store_vector(
                    image.vector_data
                        + u32::try_from(index).map_err(|_| "D16 image output overflow")?,
                    &packed,
                );
            }
        } else {
            for (index, raw) in values.iter().enumerate() {
                self.store_vector(
                    image.vector_data
                        + u32::try_from(index).map_err(|_| "image output overflow")?,
                    raw,
                );
            }
        }
        Ok(())
    }

    #[allow(clippy::unused_self)]
    fn image_store_raw_component(
        &self,
        image: &ImageControl,
        kind: &str,
        component: u32,
    ) -> String {
        if !image.d16 {
            return format!("v[{}]", image.vector_data + component);
        }
        let packed = format!("v[{}]", image.vector_data + component / 2);
        if kind == "float" {
            return format!(
                "as_type<uint>((float)as_type<half2>({packed})[{}])",
                component & 1
            );
        }
        let low = format!("(({packed} >> {}u) & 0xFFFFu)", (component & 1) * 16);
        if kind == "int" {
            format!("(uint)extract_bits(as_type<int>({low}), 0u, 16u)")
        } else {
            low
        }
    }

    #[allow(clippy::unused_self)]
    fn pack_image_d16(&self, kind: &str, low: &str, high: &str) -> String {
        if kind == "float" {
            format!(
                "((uint)as_type<ushort>(half(as_type<float>({low}))) | ((uint)as_type<ushort>(half(as_type<float>({high}))) << 16))"
            )
        } else {
            format!("(({low} & 0xFFFFu) | (({high} & 0xFFFFu) << 16))")
        }
    }

    fn emit_image_sample(
        &mut self,
        image: &ImageControl,
        opcode: Opcode,
        name: &str,
        binding_index: usize,
        sampler_slot: u32,
    ) -> Result<String, String> {
        let kind = self.image_kinds[binding_index];
        let metal_dim = metal_image_dimension(kind);
        let is_gather = name.starts_with("ImageGather4");
        let gathers_from_2d_array = metal_dim == 3 && matches!(image.dimension, 2 | 5);
        if is_gather && metal_dim != 1 && !gathers_from_2d_array {
            return Err(format!(
                "gather image dimension {} requires a 2D or 2D-array Metal texture",
                image.dimension
            ));
        }
        if is_gather && image.dmask.count_ones() != 1 {
            return Err(format!(
                "gather image dmask {:#x} must select exactly one component",
                image.dmask
            ));
        }
        let has_offset = name.ends_with('O');
        let has_compare = image_is_comparison(opcode);
        let has_gradients = name.contains("SampleD");
        let has_zero_lod = name.contains("Lz");
        let has_lod = !has_zero_lod && name.contains("SampleL");
        let has_bias = name.contains("SampleB");
        let mut cursor = 0_usize;
        let mut offset_x = "0".to_owned();
        let mut offset_y = "0".to_owned();
        let mut offset_z = "0".to_owned();
        if has_offset {
            cursor = align_full_image_address(image, cursor);
            let packed = self.temp(
                "int",
                &format!(
                    "as_type<int>(v[{}])",
                    image.address_register(image_address_register(image, cursor))
                ),
            );
            offset_x = self.temp("int", &format!("extract_bits({packed}, 0u, 6u)"));
            offset_y = self.temp("int", &format!("extract_bits({packed}, 8u, 6u)"));
            if image.dimension == 2 {
                offset_z = self.temp("int", &format!("extract_bits({packed}, 16u, 6u)"));
            }
            cursor += image_full_address_slots(image);
        }
        let bias = if has_bias {
            let value = self.image_float_address(image, cursor);
            cursor += 1;
            self.temp("float", &value)
        } else {
            "0.0f".to_owned()
        };
        let compare = if has_compare {
            let value = self.image_float_address(image, cursor);
            cursor += 1;
            self.temp("float", &value)
        } else {
            String::new()
        };
        let mut gradient_x = String::new();
        let mut gradient_y = String::new();
        if has_gradients {
            let gradient_components = if matches!(image.dimension, 2 | 3) {
                3
            } else {
                2
            };
            let gradient_kind = if gradient_components == 3 {
                "float3"
            } else {
                "float2"
            };
            gradient_x = self.temp(
                gradient_kind,
                &if gradient_components == 3 {
                    format!(
                        "float3({}, {}, {})",
                        self.image_float_address(image, cursor),
                        self.image_float_address(image, cursor + 1),
                        self.image_float_address(image, cursor + 2)
                    )
                } else {
                    format!(
                        "float2({}, {})",
                        self.image_float_address(image, cursor),
                        self.image_float_address(image, cursor + 1)
                    )
                },
            );
            gradient_y = self.temp(
                gradient_kind,
                &if gradient_components == 3 {
                    format!(
                        "float3({}, {}, {})",
                        self.image_float_address(image, cursor + 3),
                        self.image_float_address(image, cursor + 4),
                        self.image_float_address(image, cursor + 5)
                    )
                } else {
                    format!(
                        "float2({}, {})",
                        self.image_float_address(image, cursor + 2),
                        self.image_float_address(image, cursor + 3)
                    )
                },
            );
            cursor += gradient_components * 2;
        }
        let coordinate_kind = if matches!(image.dimension, 2 | 3 | 5) {
            "float3"
        } else {
            "float2"
        };
        let mut coordinates = self.temp(
            coordinate_kind,
            &if matches!(image.dimension, 2 | 3 | 5) {
                format!(
                    "float3({}, {}, {})",
                    self.image_float_address(image, cursor),
                    self.image_float_address(image, cursor + 1),
                    self.image_float_address(image, cursor + 2)
                )
            } else {
                format!(
                    "float2({}, {})",
                    self.image_float_address(image, cursor),
                    self.image_float_address(image, cursor + 1)
                )
            },
        );
        let lod = if has_zero_lod {
            "0.0f".to_owned()
        } else if has_lod {
            self.temp(
                "float",
                &self.image_float_address(
                    image,
                    cursor
                        + if matches!(image.dimension, 2 | 3 | 5) {
                            3
                        } else {
                            2
                        },
                ),
            )
        } else {
            bias.clone()
        };
        if has_offset && metal_dim == 2 {
            if image.dimension == 3 {
                return Err("cube sampling does not support integer texel offsets".to_owned());
            }
            let offset_lod = if (has_zero_lod || has_lod) && !has_gradients {
                lod.clone()
            } else {
                "0.0f".to_owned()
            };
            let mip = self.temp("uint", &format!("(uint)max((int)({offset_lod}), 0)"));
            coordinates = if image.dimension == 2 {
                self.temp(
                    "float3",
                    &format!(
                        "{coordinates} + float3((float){offset_x} / (float)tex{binding_index}.get_width({mip}), (float){offset_y} / (float)tex{binding_index}.get_height({mip}), (float){offset_z} / (float)tex{binding_index}.get_depth({mip}))"
                    ),
                )
            } else {
                self.temp(
                    "float2",
                    &format!(
                        "{coordinates} + float2((float){offset_x} / (float)tex{binding_index}.get_width({mip}), (float){offset_y} / (float)tex{binding_index}.get_height({mip}))"
                    ),
                )
            };
        } else if has_offset && image.dimension == 3 {
            return Err("cube sampling does not support integer texel offsets".to_owned());
        }
        // GCN cube sampling receives face-atlas coordinates per the ISA contract established by
        // V_CUBEID/V_CUBESC/V_CUBETC/V_CUBEMA: x = sc/(2|ma|) + 1.5, y = tc/(2|ma|) + 1.5 and
        // z = faceId (plus 6*slice for cube arrays). The in-face texel position is therefore
        // (x - 1, y - 1) with layer round(z). The cube resource is bound as a Metal 2D array,
        // so this addresses exactly the texel the guest computed; passing the raw atlas triple
        // to a direction-vector cube sampler reads arbitrary texels instead.
        //
        // When the SRD TYPE rematerializes a DIM=3D address layout onto a 2D texture (UE Slate
        // font atlases), keep the guest float3 parse for VGPR layout but sample with .xy.
        let mut sample_coordinates = coordinates.clone();
        if metal_dim == 1 && matches!(image.dimension, 2 | 3 | 5) {
            sample_coordinates = self.temp("float2", &format!("{coordinates}.xy"));
        }
        let coordinate_arguments = if metal_dim == 3 {
            let (array_uv, layer) = if image.dimension == 3 {
                // Cube coordinates are the face-atlas triple described above.
                (
                    self.temp(
                        "float2",
                        &format!(
                            "float2({coordinates}.x - 1.0f, {coordinates}.y - 1.0f)"
                        ),
                    ),
                    self.temp(
                        "uint",
                        &format!(
                            "min(uint(rint(max({coordinates}.z, 0.0f))), tex{binding_index}.get_array_size() - 1u)"
                        ),
                    ),
                )
            } else if matches!(image.dimension, 2 | 5) {
                // A 3D MIMG address rematerialized by an array SRD retains Z as the discrete
                // layer. A native 2D-array MIMG address has the same UV-plus-layer shape. Metal
                // requires the coordinate and layer as separate arguments in both cases.
                (
                    self.temp("float2", &format!("{coordinates}.xy")),
                    self.temp(
                        "uint",
                        &format!(
                            "min(uint(rint(max({coordinates}.z, 0.0f))), tex{binding_index}.get_array_size() - 1u)"
                        ),
                    ),
                )
            } else {
                // A 1D/2D MIMG address has no layer operand. Descriptor rematerialization can
                // still bind an array resource, in which case the only represented layer is 0.
                (sample_coordinates.clone(), "0u".to_owned())
            };
            format!("{array_uv}, {layer}")
        } else {
            sample_coordinates
        };
        let extra = if is_gather {
            String::new()
        } else if has_gradients {
            if metal_dim == 3 {
                // Array textures always take 2D gradients. A guest 3D/cube address carries a
                // third coordinate for depth/layer selection, never a spatial array derivative.
                if matches!(image.dimension, 2 | 3) {
                    format!(", gradient2d({gradient_x}.xy, {gradient_y}.xy)")
                } else {
                    format!(", gradient2d({gradient_x}, {gradient_y})")
                }
            } else if metal_dim == 2 {
                let gradient = "gradient3d";
                format!(", {gradient}({gradient_x}, {gradient_y})")
            } else if matches!(image.dimension, 2 | 3) && metal_dim == 1 {
                // Rematerialized 2D: discard the unused Z derivative lane.
                format!(", gradient2d({gradient_x}.xy, {gradient_y}.xy)")
            } else {
                let gradient = "gradient2d";
                format!(", {gradient}({gradient_x}, {gradient_y})")
            }
        } else if has_zero_lod || has_lod {
            format!(", level({lod})")
        } else if has_bias {
            format!(", bias({bias})")
        } else {
            String::new()
        };
        let kind = self.image_kinds[binding_index];
        let vector_kind = format!("vec<{}, 4>", kind.component);
        let global_count = self.evaluation.global_memory_bindings.len();
        let initial_index = if global_count > 30 { 1 } else { global_count };
        let sampler = if initial_index.saturating_add(1) > 30 {
            format!("remu_sampler_{sampler_slot}")
        } else {
            format!("remu_samplers.s{sampler_slot}")
        };
        let native_offset = if has_offset && metal_dim != 2 {
            format!("int2({offset_x}, {offset_y})")
        } else {
            "int2(0)".to_owned()
        };
        let sample_offset = if has_offset && metal_dim != 2 {
            format!(", {native_offset}")
        } else {
            String::new()
        };
        let physical = if is_gather && has_compare {
            self.temp(
                &vector_kind,
                &format!(
                    "tex{binding_index}.gather_compare({sampler}, {coordinate_arguments}, {compare}, {native_offset})"
                ),
            )
        } else if is_gather {
            let destination_select = self.image_destination_select(image);
            let logical_component = image.dmask.trailing_zeros();
            let selector = self.temp(
                "uint",
                &format!("({destination_select} >> {}u) & 7u", logical_component * 3),
            );
            let gathers = ["x", "y", "z", "w"].map(|component| {
                self.temp(
                    &vector_kind,
                    &format!(
                        "tex{binding_index}.gather({sampler}, {coordinate_arguments}, {native_offset}, component::{component})"
                    ),
                )
            });
            let zero = if kind.component == "float" {
                format!("{vector_kind}(0.0f)")
            } else if kind.component == "uint" {
                format!("{vector_kind}(0u)")
            } else {
                format!("{vector_kind}(0)")
            };
            let one = if kind.component == "float" {
                format!("{vector_kind}(1.0f)")
            } else if kind.component == "uint" {
                format!("{vector_kind}(1u)")
            } else {
                format!("{vector_kind}(1)")
            };
            self.temp(
                &vector_kind,
                &format!(
                    "{selector} == 1u ? {one} : ({selector} == 4u ? {} : ({selector} == 5u ? {} : ({selector} == 6u ? {} : ({selector} == 7u ? {} : {zero}))))",
                    gathers[0], gathers[1], gathers[2], gathers[3]
                ),
            )
        } else if has_compare {
            let compared = self.temp(
                "float",
                &format!(
                    "tex{binding_index}.sample_compare({sampler}, {coordinate_arguments}, {compare}{extra}{sample_offset})"
                ),
            );
            self.temp(
                &vector_kind,
                &format!("{vector_kind}({compared}, {compared}, {compared}, {compared})"),
            )
        } else {
            self.temp(
                &vector_kind,
                &format!(
                    "tex{binding_index}.sample({sampler}, {coordinate_arguments}{extra}{sample_offset})"
                ),
            )
        };
        if is_gather {
            return Ok(physical);
        }
        let zero = if kind.component == "float" {
            "0.0f"
        } else if kind.component == "uint" {
            "0u"
        } else {
            "0"
        };
        let one = if kind.component == "float" {
            "1.0f"
        } else if kind.component == "uint" {
            "1u"
        } else {
            "1"
        };
        let destination_select = self.image_destination_select(image);
        let mut logical = Vec::with_capacity(4);
        for component in 0..4 {
            let selector = format!("(({destination_select} >> {}u) & 7u)", component * 3);
            logical.push(format!(
                "({selector} == 1u ? {one} : ({selector} >= 4u && {selector} <= 7u ? {physical}[{selector} - 4u] : {zero}))"
            ));
        }
        Ok(self.temp(
            &vector_kind,
            &format!(
                "{vector_kind}({}, {}, {}, {})",
                logical[0], logical[1], logical[2], logical[3]
            ),
        ))
    }

    #[allow(clippy::unused_self)]
    fn image_float_address(&self, image: &ImageControl, component: usize) -> String {
        let register = image.address_register(image_address_register(image, component));
        if image.a16 {
            format!("(float)as_type<half2>(v[{register}])[{}]", component & 1)
        } else {
            format!("as_type<float>(v[{register}])")
        }
    }

    #[allow(clippy::unused_self)]
    fn image_uint_address(&self, image: &ImageControl, component: usize) -> String {
        let register = image.address_register(image_address_register(image, component));
        if image.a16 {
            format!("((v[{register}] >> {}u) & 0xFFFFu)", (component & 1) * 16)
        } else {
            format!("v[{register}]")
        }
    }
}
