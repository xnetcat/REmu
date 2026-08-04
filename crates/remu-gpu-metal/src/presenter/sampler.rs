// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

use super::UploadedGlobalBuffer;
use metal::{
    Buffer, Device, Function, MTLCompareFunction, MTLResourceOptions, MTLSamplerAddressMode,
    MTLSamplerMinMagFilter, MTLSamplerMipFilter, SamplerDescriptor, SamplerDescriptorRef,
    SamplerState,
};
use objc::runtime::{BOOL, YES};
use objc::{msg_send, sel, sel_impl};
use remu_shader::{ImageBinding, ShaderEvaluation, image_is_comparison};
use std::ffi::c_void;
use std::sync::OnceLock;

pub(super) const SAMPLER_ARGUMENT_CACHE_CAPACITY: usize = 1_024;

pub(super) fn metal_buffer_with_bytes(device: &Device, bytes: &[u8]) -> Result<Buffer, String> {
    let empty = [0_u8];
    let bytes = if bytes.is_empty() {
        empty.as_slice()
    } else {
        bytes
    };
    let length =
        u64::try_from(bytes.len()).map_err(|_| "global buffer byte count does not fit Metal")?;
    Ok(device.new_buffer_with_data(
        bytes.as_ptr().cast::<c_void>(),
        length,
        MTLResourceOptions::StorageModeShared,
    ))
}

pub(super) fn sampler_address_mode(value: u32) -> MTLSamplerAddressMode {
    match value {
        0 => MTLSamplerAddressMode::Repeat,
        1 => MTLSamplerAddressMode::MirrorRepeat,
        2 => MTLSamplerAddressMode::ClampToEdge,
        3 | 5 | 7 => MTLSamplerAddressMode::MirrorClampToEdge,
        4 | 6 => MTLSamplerAddressMode::ClampToBorderColor,
        _ => unreachable!("three-bit sampler address mode"),
    }
}

pub(super) fn sampler_min_mag_filter(value: u32) -> MTLSamplerMinMagFilter {
    if matches!(value, 1 | 3) {
        MTLSamplerMinMagFilter::Linear
    } else {
        MTLSamplerMinMagFilter::Nearest
    }
}

pub(super) fn sampler_mip_filter(value: u32) -> MTLSamplerMipFilter {
    match value {
        0 => MTLSamplerMipFilter::NotMipmapped,
        1 => MTLSamplerMipFilter::Nearest,
        2 | 3 => MTLSamplerMipFilter::Linear,
        _ => unreachable!("two-bit sampler mip filter"),
    }
}

pub(super) fn sampler_compare_function(value: u32) -> MTLCompareFunction {
    match value & 0x7 {
        0 => MTLCompareFunction::Never,
        1 => MTLCompareFunction::Less,
        2 => MTLCompareFunction::Equal,
        3 => MTLCompareFunction::LessEqual,
        4 => MTLCompareFunction::Greater,
        5 => MTLCompareFunction::NotEqual,
        6 => MTLCompareFunction::GreaterEqual,
        7 => MTLCompareFunction::Always,
        _ => unreachable!("three-bit sampler comparison function"),
    }
}

fn diagnostic_sampler_compare_function() -> Option<MTLCompareFunction> {
    static FUNCTION: OnceLock<Option<MTLCompareFunction>> = OnceLock::new();
    *FUNCTION.get_or_init(|| {
        let value = std::env::var("REMU_DIAGNOSTIC_COMPARE_FUNCTION").ok()?;
        match value.to_ascii_lowercase().as_str() {
            "never" => Some(MTLCompareFunction::Never),
            "less" => Some(MTLCompareFunction::Less),
            "less-equal" => Some(MTLCompareFunction::LessEqual),
            "greater" => Some(MTLCompareFunction::Greater),
            "greater-equal" => Some(MTLCompareFunction::GreaterEqual),
            "always" => Some(MTLCompareFunction::Always),
            _ => {
                eprintln!(
                    "[REMu][METAL] ignoring unsupported REMU_DIAGNOSTIC_COMPARE_FUNCTION={value:?}"
                );
                None
            }
        }
    })
}

pub(super) fn sampled_image_bindings(
    evaluation: &ShaderEvaluation,
) -> impl Iterator<Item = &ImageBinding> {
    evaluation.image_bindings.iter().filter(|binding| {
        binding.opcode.starts_with("ImageSample") || binding.opcode.starts_with("ImageGather")
    })
}

pub(super) fn canonical_texture_bindings(evaluation: &ShaderEvaluation) -> Vec<&ImageBinding> {
    let slots = evaluation.image_binding_slots();
    let mut representatives = vec![
        None;
        usize::try_from(slots.texture_count)
            .expect("canonical texture count fits the host index")
    ];
    for (binding, slot) in evaluation
        .image_bindings
        .iter()
        .zip(slots.texture_slots.iter().copied())
    {
        let slot = usize::try_from(slot).expect("canonical texture slot fits the host index");
        representatives[slot].get_or_insert(binding);
    }
    representatives
        .into_iter()
        .map(|binding| binding.expect("canonical texture slots are dense"))
        .collect()
}

pub(super) fn canonical_sampler_bindings(evaluation: &ShaderEvaluation) -> Vec<&ImageBinding> {
    let slots = evaluation.image_binding_slots();
    let mut representatives = vec![
        None;
        usize::try_from(slots.sampler_count)
            .expect("canonical sampler count fits the host index")
    ];
    for (binding, slot) in evaluation
        .image_bindings
        .iter()
        .zip(slots.sampler_slots.iter().copied())
    {
        let Some(slot) = slot else {
            continue;
        };
        let slot = usize::try_from(slot).expect("canonical sampler slot fits the host index");
        representatives[slot].get_or_insert(binding);
    }
    representatives
        .into_iter()
        .map(|binding| binding.expect("canonical sampler slots are dense"))
        .collect()
}

pub(super) fn sampler_uses_normalized_coordinates(words: &[u32]) -> bool {
    words.first().is_none_or(|word| word & (1 << 15) == 0)
}

pub(super) fn sampler_lod_bias(words: &[u32]) -> f32 {
    let raw = words.get(2).copied().unwrap_or_default() & 0x3fff;
    let signed = ((raw.cast_signed() ^ 0x2000) - 0x2000) as f32;
    signed / 256.0
}

fn set_sampler_lod_bias_if_supported(descriptor: &SamplerDescriptorRef, bias: f32) {
    // `lodBias` became a public MTLSamplerDescriptor property after the macOS 14 deployment
    // target still used by CI and supported REmu hosts. Sending the setter unconditionally makes
    // those systems raise an Objective-C `unrecognized selector` exception across Rust frames.
    // Runtime selector discovery keeps the public property on newer systems and safely leaves the
    // default zero bias on older Metal runtimes.
    let supported: BOOL = unsafe { msg_send![descriptor, respondsToSelector: sel!(setLodBias:)] };
    if supported == YES {
        let _: () = unsafe { msg_send![descriptor, setLodBias: bias] };
    }
}

pub(super) fn create_global_argument_buffer(
    device: &Device,
    function: &Function,
    buffer_index: Option<u32>,
    buffers: &[UploadedGlobalBuffer],
) -> Result<Option<Buffer>, String> {
    let Some(buffer_index) = buffer_index else {
        return Ok(None);
    };
    if buffers.is_empty() {
        return Err("Metal shader declares an empty global argument buffer".to_owned());
    }
    let encoder = function.new_argument_encoder(u64::from(buffer_index));
    let argument_buffer = device.new_buffer(
        encoder.encoded_length(),
        MTLResourceOptions::StorageModeShared,
    );
    argument_buffer.set_label("REMu guest global argument buffer");
    encoder.set_argument_buffer(&argument_buffer, 0);
    for (slot, buffer) in buffers.iter().enumerate() {
        encoder.set_buffer(
            u64::try_from(slot).map_err(|_| "global argument slot does not fit Metal")?,
            &buffer.buffer,
            buffer.offset,
        );
    }
    Ok(Some(argument_buffer))
}

pub(super) fn create_sampler_argument_buffer(
    device: &Device,
    function: &Function,
    buffer_index: Option<u32>,
    evaluation: &ShaderEvaluation,
) -> Result<(Option<Buffer>, Vec<SamplerState>), String> {
    let sampled_bindings = canonical_sampler_bindings(evaluation);
    if sampled_bindings.is_empty() {
        return if buffer_index.is_some() {
            Err("Metal shader declares a sampler argument buffer without samplers".to_owned())
        } else {
            Ok((None, Vec::new()))
        };
    }
    let argument_encoder =
        buffer_index.map(|index| function.new_argument_encoder(u64::from(index)));
    let argument_buffer = argument_encoder.as_ref().map(|encoder| {
        let buffer = device.new_buffer(
            encoder.encoded_length(),
            MTLResourceOptions::StorageModeShared,
        );
        encoder.set_argument_buffer(&buffer, 0);
        buffer
    });
    let mut samplers = Vec::with_capacity(sampled_bindings.len());
    for (slot, binding) in sampled_bindings.iter().enumerate() {
        if binding.sampler_descriptor.len() < 4 {
            return Err(format!(
                "sampler at PC {:#x} has {} dwords, expected four",
                binding.pc,
                binding.sampler_descriptor.len()
            ));
        }
        let words = &binding.sampler_descriptor;
        let descriptor = SamplerDescriptor::new();
        descriptor.set_address_mode_s(sampler_address_mode(words[0] & 0x7));
        descriptor.set_address_mode_t(sampler_address_mode((words[0] >> 3) & 0x7));
        descriptor.set_address_mode_r(sampler_address_mode((words[0] >> 6) & 0x7));
        descriptor.set_max_anisotropy(u64::from(1_u32 << ((words[0] >> 9) & 0x7).min(4)));
        descriptor.set_normalized_coordinates(sampler_uses_normalized_coordinates(words));
        // GFX10 stores a signed 14-bit fixed-point (8 fractional bits) sampler bias. Modern Metal
        // exposes the equivalent descriptor property; older supported runtimes keep zero bias.
        set_sampler_lod_bias_if_supported(&descriptor, sampler_lod_bias(words));
        descriptor.set_mag_filter(sampler_min_mag_filter((words[2] >> 20) & 0x3));
        descriptor.set_min_filter(sampler_min_mag_filter((words[2] >> 22) & 0x3));
        descriptor.set_mip_filter(sampler_mip_filter((words[2] >> 26) & 0x3));
        if image_is_comparison(binding.opcode) {
            let guest_function = sampler_compare_function((words[0] >> 12) & 0x7);
            let diagnostic_function = diagnostic_sampler_compare_function();
            if let Some(function) = diagnostic_function {
                static LOGGED: OnceLock<()> = OnceLock::new();
                LOGGED.get_or_init(|| {
                    eprintln!(
                        "[REMu][METAL][DIAGNOSTIC] overriding comparison samplers from {guest_function:?} to {function:?}"
                    );
                });
            }
            descriptor.set_compare_function(diagnostic_function.unwrap_or(guest_function));
        }
        descriptor.set_lod_min_clamp(
            f32::from(u16::try_from(words[1] & 0xfff).expect("12-bit LOD fits u16")) / 256.0,
        );
        descriptor.set_lod_max_clamp(
            f32::from(u16::try_from((words[1] >> 12) & 0xfff).expect("12-bit LOD fits u16"))
                / 256.0,
        );
        descriptor.set_support_argument_buffers(argument_encoder.is_some());
        let sampler = device.new_sampler(&descriptor);
        if let Some(encoder) = argument_encoder.as_ref() {
            encoder.set_sampler_state(
                u64::try_from(slot).map_err(|_| "sampler slot does not fit Metal")?,
                &sampler,
            );
        }
        samplers.push(sampler);
    }
    Ok((argument_buffer, samplers))
}
