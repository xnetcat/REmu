// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

use super::{
    CB_BLEND0_CONTROL, CB_TARGET_MASK, DecodedImageDescriptor, DiagnosticScanoutWriter,
    GuestTextureDescriptor, decode_depth_target, decode_image_descriptor,
    decode_metal_blend_attachment, decode_raster_state, decode_render_targets, decode_scissor,
    decode_viewport, guest_ranges_overlap, guest_texture_byte_count,
    guest_texture_descriptors_overlap, guest_texture_descriptors_share_storage,
    is_storage_image_opcode, is_writable_image_opcode, register_value, sampled_image_bindings,
};
use metal::MTLPixelFormat;
use remu_gpu::{ComputeDispatch, GpuMemoryCopy, IndexedDraw};
use remu_shader::{GlobalMemoryBinding, ImageBinding, ShaderEvaluation, image_is_comparison};
use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};

pub(super) const DIAGNOSTIC_VERTEX_POSITION_VERTEX_CAPACITY: usize = 262_144;

pub(super) const DIAGNOSTIC_VERTEX_POSITION_BUFFER_INDEX: u64 = 30;

pub(super) const DIAGNOSTIC_PIXEL_OUTPUT_WIDTH: usize = 1_920;

pub(super) const DIAGNOSTIC_PIXEL_OUTPUT_HEIGHT: usize = 1_080;

pub(super) const DIAGNOSTIC_PIXEL_OUTPUT_BUFFER_INDEX: u64 = 30;

pub(super) fn trace_image_descriptors() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("REMU_TRACE_IMAGES").is_some_and(|value| value == "1"))
}

pub(super) fn trace_target_sampling() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var_os("REMU_TRACE_TARGET_SAMPLING").is_some_and(|value| value == "1")
    })
}

pub(super) fn trace_video_target_consumers() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var_os("REMU_TRACE_VIDEO_TARGETS").is_some_and(|value| value == "1")
    })
}

pub(super) fn diagnostic_dump_video_target_consumer() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var_os("REMU_DUMP_VIDEO_TARGET_CONSUMER").is_some_and(|value| value == "1")
    })
}

pub(super) fn diagnostic_dump_video_chain_consumer() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var_os("REMU_DUMP_VIDEO_CHAIN_CONSUMER").is_some_and(|value| value == "1")
    })
}

pub(super) fn diagnostic_dump_shadow_consumer() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var_os("REMU_DUMP_SHADOW_CONSUMER").is_some_and(|value| value == "1")
    })
}

pub(super) fn evaluation_samples_layered_comparison(evaluation: &ShaderEvaluation) -> bool {
    evaluation.image_bindings.iter().any(|binding| {
        image_is_comparison(binding.opcode)
            && decode_image_descriptor(&binding.resource_descriptor)
                .is_ok_and(|descriptor| descriptor.layer_count > 1)
    })
}

pub(super) fn diagnostic_video_chain_selected_draw() -> &'static AtomicU64 {
    static DRAW: AtomicU64 = AtomicU64::new(0);
    &DRAW
}

pub(super) fn video_target_binding_summary(evaluation: &ShaderEvaluation) -> Option<String> {
    let bindings = evaluation
        .image_bindings
        .iter()
        .filter_map(|binding| {
            let descriptor = decode_image_descriptor(&binding.resource_descriptor).ok()?;
            let known_small_video = descriptor.guest.width == 378 && descriptor.guest.height == 150;
            let large_linear_video = descriptor.guest.tile_mode == 0
                && descriptor.guest.width >= 700
                && descriptor.guest.height >= 500
                && matches!(
                    descriptor.guest.pixel_format,
                    MTLPixelFormat::R8Unorm | MTLPixelFormat::RG8Unorm
                );
            (known_small_video || large_linear_video).then(|| {
                format!(
                    "pc={:#x} op={:?} addr={:#x} format={:?} tile={} base_mip={} mips={} storage={}",
                    binding.pc,
                    binding.opcode,
                    descriptor.guest.address,
                    descriptor.guest.pixel_format,
                    descriptor.guest.tile_mode,
                    descriptor.base_mip_level,
                    descriptor.mip_level_count,
                    is_storage_image_opcode(binding.opcode),
                )
            })
        })
        .collect::<Vec<_>>();
    (!bindings.is_empty()).then(|| bindings.join("; "))
}

#[derive(Clone, Copy, Debug)]
pub(super) struct DiagnosticVideoChainProducer {
    stage: &'static str,
    sequence: u64,
    shader: u64,
    generation: u32,
}

pub(super) fn diagnostic_video_chain_textures()
-> &'static Mutex<HashMap<GuestTextureDescriptor, DiagnosticVideoChainProducer>> {
    static TEXTURES: OnceLock<
        Mutex<HashMap<GuestTextureDescriptor, DiagnosticVideoChainProducer>>,
    > = OnceLock::new();
    TEXTURES.get_or_init(|| Mutex::new(HashMap::new()))
}

pub(super) fn diagnostic_video_chain_matches(
    evaluation: &ShaderEvaluation,
) -> Vec<(
    &ImageBinding,
    DecodedImageDescriptor,
    GuestTextureDescriptor,
    DiagnosticVideoChainProducer,
)> {
    let tracked = diagnostic_video_chain_textures()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .iter()
        .map(|(descriptor, producer)| (*descriptor, *producer))
        .collect::<Vec<_>>();
    evaluation
        .image_bindings
        .iter()
        .filter(|binding| !is_writable_image_opcode(binding.opcode))
        .filter_map(|binding| {
            let descriptor = decode_image_descriptor(&binding.resource_descriptor).ok()?;
            tracked
                .iter()
                .find(|(candidate, _)| {
                    candidate.address == descriptor.guest.address
                        || guest_texture_descriptors_share_storage(*candidate, descriptor.guest)
                        || guest_texture_descriptors_overlap(*candidate, descriptor.guest)
                })
                .map(|(candidate, producer)| (binding, descriptor, *candidate, *producer))
        })
        .collect()
}

pub(super) fn video_chain_binding_summary(evaluation: &ShaderEvaluation) -> Option<String> {
    let bindings = diagnostic_video_chain_matches(evaluation)
        .into_iter()
        .map(|(binding, descriptor, source, producer)| {
            format!(
                "pc={:#x} op={:?} addr={:#x} {}x{} format={:?} tile={} base_mip={} mips={} source={:#x}/{}x{} producer={}:{}:{:#x} generation={}",
                binding.pc,
                binding.opcode,
                descriptor.guest.address,
                descriptor.guest.width,
                descriptor.guest.height,
                descriptor.guest.pixel_format,
                descriptor.guest.tile_mode,
                descriptor.base_mip_level,
                descriptor.mip_level_count,
                source.address,
                source.width,
                source.height,
                producer.stage,
                producer.sequence,
                producer.shader,
                producer.generation,
            )
        })
        .collect::<Vec<_>>();
    (!bindings.is_empty()).then(|| bindings.join("; "))
}

pub(super) fn record_diagnostic_video_chain_texture(
    descriptor: GuestTextureDescriptor,
    stage: &'static str,
    sequence: u64,
    shader: u64,
    generation: u32,
) {
    diagnostic_video_chain_textures()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .insert(
            descriptor,
            DiagnosticVideoChainProducer {
                stage,
                sequence,
                shader,
                generation,
            },
        );
}

pub(super) fn trace_video_target_draw_consumer(draw: &IndexedDraw) {
    if !trace_video_target_consumers() && !diagnostic_dump_video_chain_consumer() {
        return;
    }
    let export = draw
        .export_shader_evaluation
        .as_deref()
        .and_then(video_target_binding_summary);
    let pixel = draw
        .pixel_shader_evaluation
        .as_deref()
        .and_then(video_target_binding_summary);
    let export_chain = draw
        .export_shader_evaluation
        .as_deref()
        .and_then(video_chain_binding_summary);
    let pixel_chain = draw
        .pixel_shader_evaluation
        .as_deref()
        .and_then(video_chain_binding_summary);
    let root = export.is_some() || pixel.is_some();
    let chained = export_chain.is_some() || pixel_chain.is_some();
    if !root && !chained {
        return;
    }
    if chained
        && diagnostic_dump_video_chain_consumer()
        && draw.sequence >= diagnostic_dump_target_minimum_draw()
        && decode_render_targets(draw).is_ok_and(|targets| !targets.is_empty())
    {
        let _ = diagnostic_video_chain_selected_draw().compare_exchange(
            0,
            draw.sequence,
            Ordering::Relaxed,
            Ordering::Relaxed,
        );
    }
    let target_mask = register_value(&draw.context_registers, CB_TARGET_MASK).unwrap_or_default();
    let depth = decode_depth_target(draw)
        .map(|depth| format!("{depth:?}"))
        .unwrap_or_else(|error| format!("unavailable: {error}"));
    let raster = decode_raster_state(draw)
        .map(|raster| format!("{raster:?}"))
        .unwrap_or_else(|error| format!("unavailable: {error}"));
    let geometry = decode_render_targets(draw)
        .ok()
        .and_then(|targets| {
            targets
                .first()
                .map(|target| (target.texture.width, target.texture.height))
        })
        .or_else(|| {
            decode_depth_target(draw)
                .ok()
                .flatten()
                .map(|depth| (depth.texture.width, depth.texture.height))
        })
        .map(|(width, height)| {
            decode_scissor(draw, width, height)
                .map(|scissor| {
                    let viewport = decode_viewport(draw, width, height, scissor);
                    format!(
                        "scissor={scissor:?} viewport=({:.1},{:.1} {:.1}x{:.1} {:.3}..{:.3})",
                        viewport.originX,
                        viewport.originY,
                        viewport.width,
                        viewport.height,
                        viewport.znear,
                        viewport.zfar,
                    )
                })
                .unwrap_or_else(|error| format!("unavailable: {error}"))
        })
        .unwrap_or_else(|| "unavailable: no attachment extent".to_owned());
    let decoded_targets = decode_render_targets(draw);
    let targets = decoded_targets
        .as_ref()
        .map(|targets| {
            targets
                .iter()
                .map(|target| {
                    let blend_control = register_value(
                        &draw.context_registers,
                        CB_BLEND0_CONTROL + target.guest_slot,
                    )
                    .unwrap_or(0x2001_0001);
                    let blend = decode_metal_blend_attachment(blend_control)
                        .map(|blend| format!("{blend:?}"))
                        .unwrap_or_else(|error| format!("unavailable: {error}"));
                    format!(
                        "slot={} addr={:#x} {}x{} format={:?} mask={:#x} blend_control={:#010x} blend=[{}]",
                        target.guest_slot,
                        target.texture.address,
                        target.texture.width,
                        target.texture.height,
                        target.texture.pixel_format,
                        target.write_mask,
                        blend_control,
                        blend,
                    )
                })
                .collect::<Vec<_>>()
                .join("; ")
        })
        .unwrap_or_else(|error| format!("unavailable: {error}"));
    if let Ok(targets) = &decoded_targets {
        let generation = if root {
            1
        } else {
            diagnostic_video_chain_matches(
                draw.pixel_shader_evaluation
                    .as_deref()
                    .or(draw.export_shader_evaluation.as_deref())
                    .expect("a chained draw has an evaluated stage"),
            )
            .into_iter()
            .map(|(_, _, _, producer)| producer.generation.saturating_add(1))
            .min()
            .unwrap_or(2)
        };
        for target in targets {
            // A direct root draw may intentionally position the movie outside the scanout while
            // another draw writes the actual intermediate. Do not let that known-empty terminal
            // branch contaminate the transitive compositor trace.
            if !root || !target.direct_scanout {
                record_diagnostic_video_chain_texture(
                    target.texture,
                    "draw",
                    draw.sequence,
                    draw.pixel_shader_address,
                    generation,
                );
            }
        }
    }
    eprintln!(
        "[REMu][METAL][VIDEO-CHAIN] kind=draw generation={} root={} sequence={} es={:#x} ps={:#x} ps_insns={} indexed={} count={} instances={} target_mask={:#010x} depth=[{}] raster=[{}] geometry=[{}] export=[{}] pixel=[{}] export_chain=[{}] pixel_chain=[{}] targets=[{}]",
        if root { 1 } else { 2 },
        root,
        draw.sequence,
        draw.export_shader_address,
        draw.pixel_shader_address,
        draw.pixel_shader
            .as_ref()
            .map_or(0, |shader| shader.program.instructions.len()),
        draw.indexed,
        draw.index_count,
        draw.instance_count,
        target_mask,
        depth,
        raster,
        geometry,
        export.as_deref().unwrap_or(""),
        pixel.as_deref().unwrap_or(""),
        export_chain.as_deref().unwrap_or(""),
        pixel_chain.as_deref().unwrap_or(""),
        targets,
    );
}

pub(super) fn trace_video_target_compute_consumer(dispatch: &ComputeDispatch) {
    if !trace_video_target_consumers() && !diagnostic_dump_video_chain_consumer() {
        return;
    }
    let Some(evaluation) = dispatch.shader_evaluation.as_deref() else {
        return;
    };
    let root = video_target_binding_summary(evaluation);
    let chain = video_chain_binding_summary(evaluation);
    if root.is_none() && chain.is_none() {
        return;
    }
    let generation = if root.is_some() {
        1
    } else {
        diagnostic_video_chain_matches(evaluation)
            .into_iter()
            .map(|(_, _, _, producer)| producer.generation.saturating_add(1))
            .min()
            .unwrap_or(2)
    };
    let mut outputs = Vec::new();
    for binding in evaluation
        .image_bindings
        .iter()
        .filter(|binding| is_writable_image_opcode(binding.opcode))
    {
        let Ok(descriptor) = decode_image_descriptor(&binding.resource_descriptor) else {
            continue;
        };
        record_diagnostic_video_chain_texture(
            descriptor.guest,
            "compute",
            dispatch.sequence,
            dispatch.shader_address,
            generation,
        );
        outputs.push(format!(
            "pc={:#x} op={:?} addr={:#x} {}x{} format={:?} tile={}",
            binding.pc,
            binding.opcode,
            descriptor.guest.address,
            descriptor.guest.width,
            descriptor.guest.height,
            descriptor.guest.pixel_format,
            descriptor.guest.tile_mode,
        ));
    }
    eprintln!(
        "[REMu][METAL][VIDEO-CHAIN] kind=compute generation={} root={} sequence={} cs={:#x} grid={:?} groups={:?} threads_per_group={:?} root_bindings=[{}] chain_bindings=[{}] outputs=[{}]",
        generation,
        root.is_some(),
        dispatch.sequence,
        dispatch.shader_address,
        dispatch.launch.grid,
        dispatch.launch.group_count,
        dispatch.launch.threads_per_group,
        root.as_deref().unwrap_or(""),
        chain.as_deref().unwrap_or(""),
        outputs.join("; "),
    );
}

pub(super) fn capture_target_sample_textures() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var_os("REMU_CAPTURE_TARGET_SAMPLES").is_some_and(|value| value == "1")
    })
}

pub(super) fn trace_scanout_writes() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var_os("REMU_TRACE_SCANOUT_WRITES").is_some_and(|value| value == "1")
    })
}

pub(super) fn diagnostic_dump_target_before_pixel_shader() -> Option<u64> {
    static ADDRESS: OnceLock<Option<u64>> = OnceLock::new();
    *ADDRESS.get_or_init(|| {
        std::env::var("REMU_DUMP_TARGET_BEFORE_PS")
            .ok()
            .and_then(|value| parse_diagnostic_hex_address(&value))
    })
}

pub(super) const fn should_record_diagnostic_target_writer(
    dump_target_before_pixel_shader: bool,
    trace_scanout_writes: bool,
    is_known_scanout: bool,
) -> bool {
    dump_target_before_pixel_shader || (trace_scanout_writes && is_known_scanout)
}

pub(super) fn parse_diagnostic_hex_address(value: &str) -> Option<u64> {
    let value = value
        .strip_prefix("0x")
        .or_else(|| value.strip_prefix("0X"))
        .unwrap_or(value);
    (!value.is_empty())
        .then(|| u64::from_str_radix(value, 16).ok())
        .flatten()
}

pub(super) fn diagnostic_scanout_writers() -> &'static Mutex<HashMap<u64, DiagnosticScanoutWriter>>
{
    static WRITERS: OnceLock<Mutex<HashMap<u64, DiagnosticScanoutWriter>>> = OnceLock::new();
    WRITERS.get_or_init(|| Mutex::new(HashMap::new()))
}

pub(super) fn diagnostic_scanout_writer(address: u64) -> Option<DiagnosticScanoutWriter> {
    diagnostic_scanout_writers()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .get(&address)
        .copied()
}

pub(super) fn record_diagnostic_scanout_writer(address: u64, writer: DiagnosticScanoutWriter) {
    diagnostic_scanout_writers()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .insert(address, writer);
}

pub(super) fn trace_msaa_writes() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var_os("REMU_TRACE_MSAA_WRITES").is_some_and(|value| value == "1")
    })
}

pub(super) fn diagnostic_disable_depth_and_culling() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var_os("REMU_DIAGNOSTIC_RASTER").is_some_and(|value| value == "1")
    })
}

pub(super) fn diagnostic_disable_culling() -> bool {
    diagnostic_disable_depth_and_culling() || {
        static ENABLED: OnceLock<bool> = OnceLock::new();
        *ENABLED.get_or_init(|| {
            std::env::var_os("REMU_DIAGNOSTIC_CULL").is_some_and(|value| value == "1")
        })
    }
}

pub(super) fn diagnostic_disable_depth() -> bool {
    diagnostic_disable_depth_and_culling() || {
        static ENABLED: OnceLock<bool> = OnceLock::new();
        *ENABLED.get_or_init(|| {
            std::env::var_os("REMU_DIAGNOSTIC_DEPTH").is_some_and(|value| value == "1")
        })
    }
}

pub(super) fn diagnostic_force_visible_clip_depth() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED
        .get_or_init(|| std::env::var_os("REMU_DIAGNOSTIC_CLIP").is_some_and(|value| value == "1"))
}

pub(super) fn diagnostic_force_msaa_pixel_color() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var_os("REMU_DIAGNOSTIC_MSAA_COLOR").is_some_and(|value| value == "1")
    })
}

pub(super) fn diagnostic_force_msaa_vertex_pattern() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var_os("REMU_DIAGNOSTIC_MSAA_PATTERN").is_some_and(|value| value == "1")
    })
}

pub(super) fn diagnostic_force_vertex_fetch_id() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var_os("REMU_DIAGNOSTIC_VERTEX_FETCH_ID").is_some_and(|value| value == "1")
    })
}

pub(super) fn diagnostic_capture_vertex_position() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var_os("REMU_DIAGNOSTIC_CAPTURE_VERTEX_POSITION")
            .is_some_and(|value| value == "1")
    })
}

pub(super) fn diagnostic_capture_pixel_output() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var_os("REMU_DIAGNOSTIC_CAPTURE_PIXEL_OUTPUT").is_some_and(|value| value == "1")
    })
}

pub(super) fn diagnostic_disable_blending() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED
        .get_or_init(|| std::env::var_os("REMU_DIAGNOSTIC_BLEND").is_some_and(|value| value == "1"))
}

pub(super) fn diagnostic_use_draw_merged_wave_counts() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var_os("REMU_DIAGNOSTIC_MERGED_COUNTS").is_some_and(|value| value == "1")
    })
}

pub(super) fn diagnostic_dump_target_address() -> Option<u64> {
    static ADDRESS: OnceLock<Option<u64>> = OnceLock::new();
    *ADDRESS.get_or_init(|| {
        let value = std::env::var("REMU_DUMP_TARGET_ADDRESS").ok()?;
        if value.eq_ignore_ascii_case("auto") {
            return Some(0);
        }
        let value = value.strip_prefix("0x").unwrap_or(&value);
        u64::from_str_radix(value, 16).ok()
    })
}

pub(super) fn diagnostic_dump_target_pixel_shader() -> Option<u64> {
    static ADDRESS: OnceLock<Option<u64>> = OnceLock::new();
    *ADDRESS.get_or_init(|| {
        let value = std::env::var("REMU_DUMP_TARGET_PS").ok()?;
        let value = value.strip_prefix("0x").unwrap_or(&value);
        u64::from_str_radix(value, 16).ok()
    })
}

pub(super) fn diagnostic_dump_target_pixel_instruction_count() -> Option<usize> {
    static COUNT: OnceLock<Option<usize>> = OnceLock::new();
    *COUNT.get_or_init(|| {
        std::env::var("REMU_DUMP_TARGET_PS_INSNS")
            .ok()
            .and_then(|value| value.parse().ok())
    })
}

pub(super) fn diagnostic_dump_complex_msaa() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var_os("REMU_DUMP_COMPLEX_MSAA").is_some_and(|value| value == "1")
    })
}

pub(super) fn diagnostic_texture_writer_enabled() -> bool {
    trace_scanout_writes()
        || diagnostic_dump_target_address().is_some()
        || diagnostic_dump_target_pixel_shader().is_some()
        || diagnostic_dump_target_pixel_instruction_count().is_some()
        || diagnostic_dump_complex_msaa()
        || diagnostic_dump_shadow_consumer()
        || diagnostic_dump_video_target_consumer()
        || diagnostic_dump_video_chain_consumer()
}

pub(super) fn diagnostic_dump_target_minimum_draw() -> u64 {
    static DRAW: OnceLock<u64> = OnceLock::new();
    *DRAW.get_or_init(|| {
        std::env::var("REMU_DUMP_TARGET_MIN_DRAW")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(131_072)
    })
}

pub(super) fn diagnostic_dump_target_match_index() -> usize {
    static INDEX: OnceLock<usize> = OnceLock::new();
    *INDEX.get_or_init(|| {
        std::env::var("REMU_DUMP_TARGET_MATCH_INDEX")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(0)
    })
}

pub(super) fn diagnostic_trace_target_contents() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("REMU_TRACE_TARGET_CONTENTS")
            .ok()
            .is_some_and(|value| value != "0")
    })
}

pub(super) fn diagnostic_dump_video_draw() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("REMU_DUMP_VIDEO_DRAW")
            .ok()
            .is_some_and(|value| value != "0")
    })
}

pub(super) fn diagnostic_trace_all_target_contents() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("REMU_TRACE_ALL_TARGET_CONTENTS")
            .ok()
            .is_some_and(|value| value != "0")
    })
}

pub(super) fn diagnostic_trace_target_maximum_draw() -> u64 {
    static DRAW: OnceLock<u64> = OnceLock::new();
    *DRAW.get_or_init(|| {
        std::env::var("REMU_TRACE_TARGET_MAX_DRAW")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(u64::MAX)
    })
}

pub(super) fn diagnostic_trace_target_stride() -> u64 {
    static STRIDE: OnceLock<u64> = OnceLock::new();
    *STRIDE.get_or_init(|| {
        std::env::var("REMU_TRACE_TARGET_STRIDE")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(1)
            .max(1)
    })
}

pub(super) fn diagnostic_global_binding_head(binding: &GlobalMemoryBinding) -> Vec<u8> {
    let byte_count = binding.byte_count.min(256);
    if byte_count == 0 {
        return Vec::new();
    }
    if let Some(bytes) = binding.data.get(..byte_count) {
        return bytes.to_vec();
    }
    if binding.base_address < 0x10_000 {
        return Vec::new();
    }
    // SAFETY: Shader evaluation has already validated this exact live guest binding and the
    // presenter consumes it synchronously while the draw snapshot owns the allocation.
    unsafe { std::slice::from_raw_parts(binding.base_address as *const u8, byte_count).to_vec() }
}

pub(super) fn record_diagnostic_texture_writer(
    descriptor: GuestTextureDescriptor,
    stage: &'static str,
    sequence: u64,
    shader: u64,
    sampled: Vec<GuestTextureDescriptor>,
) {
    if !diagnostic_texture_writer_enabled() {
        return;
    }
    diagnostic_texture_writers()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .insert(descriptor, (stage, sequence, shader, sampled));
}

pub(super) type DiagnosticTextureWriter = (&'static str, u64, u64, Vec<GuestTextureDescriptor>);

pub(super) fn diagnostic_texture_writers()
-> &'static Mutex<HashMap<GuestTextureDescriptor, DiagnosticTextureWriter>> {
    static WRITERS: OnceLock<Mutex<HashMap<GuestTextureDescriptor, DiagnosticTextureWriter>>> =
        OnceLock::new();
    WRITERS.get_or_init(|| Mutex::new(HashMap::new()))
}

pub(super) fn diagnostic_texture_writer(
    descriptor: GuestTextureDescriptor,
) -> Option<DiagnosticTextureWriter> {
    diagnostic_texture_writers()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .get(&descriptor)
        .cloned()
}

pub(super) fn diagnostic_overlapping_texture_writers(
    descriptor: GuestTextureDescriptor,
) -> Vec<(GuestTextureDescriptor, DiagnosticTextureWriter)> {
    diagnostic_texture_writers()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .iter()
        .filter(|(candidate, _)| {
            **candidate != descriptor && guest_texture_descriptors_overlap(**candidate, descriptor)
        })
        .map(|(candidate, writer)| (*candidate, writer.clone()))
        .collect()
}

pub(super) fn diagnostic_related_texture_writers(
    descriptor: GuestTextureDescriptor,
) -> Vec<(GuestTextureDescriptor, DiagnosticTextureWriter)> {
    let mut writers = diagnostic_texture_writers()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .iter()
        .filter(|(candidate, _)| {
            candidate.width == descriptor.width
                && candidate.height == descriptor.height
                && candidate.pixel_format == descriptor.pixel_format
        })
        .map(|(candidate, writer)| (*candidate, writer.clone()))
        .collect::<Vec<_>>();
    writers.sort_unstable_by_key(|(_, writer)| writer.1);
    if writers.len() > 64 {
        writers.drain(..writers.len() - 64);
    }
    writers
}

pub(super) fn record_diagnostic_memory_copy(copy: GpuMemoryCopy) {
    if !diagnostic_texture_writer_enabled() {
        return;
    }
    const CAPACITY: usize = 8_192;
    let mut copies = diagnostic_memory_copies()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if copies.len() == CAPACITY {
        copies.pop_front();
    }
    copies.push_back(copy);
}

pub(super) fn diagnostic_memory_copies() -> &'static Mutex<VecDeque<GpuMemoryCopy>> {
    static COPIES: OnceLock<Mutex<VecDeque<GpuMemoryCopy>>> = OnceLock::new();
    COPIES.get_or_init(|| Mutex::new(VecDeque::new()))
}

pub(super) fn diagnostic_overlapping_memory_copies(
    descriptor: GuestTextureDescriptor,
) -> Vec<GpuMemoryCopy> {
    let Some(byte_count) = guest_texture_byte_count(descriptor) else {
        return Vec::new();
    };
    diagnostic_memory_copies()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .iter()
        .rev()
        .filter(|copy| {
            guest_ranges_overlap(
                descriptor.address,
                byte_count,
                copy.destination_address,
                copy.byte_count,
            )
        })
        .take(16)
        .copied()
        .collect()
}

#[derive(Clone, Copy, Debug)]
#[allow(dead_code)] // Retained as structured fields for the diagnostic tuple's Debug output.
pub(super) struct DiagnosticGlobalWriter {
    pub(super) stage: &'static str,
    pub(super) sequence: u64,
    pub(super) shader: u64,
    address: u64,
    byte_count: u64,
}

pub(super) fn record_diagnostic_global_writers(
    stage: &'static str,
    sequence: u64,
    shader: u64,
    evaluation: &ShaderEvaluation,
) {
    let retain_history = diagnostic_texture_writer_enabled();
    let trace_range = diagnostic_guest_write_trace_range();
    if !retain_history && trace_range.is_none() {
        return;
    }
    let writable = evaluation
        .global_memory_bindings
        .iter()
        .filter(|binding| binding.writable && binding.base_address >= 0x10_000)
        .filter_map(|binding| {
            let writer = DiagnosticGlobalWriter {
                stage,
                sequence,
                shader,
                address: binding.base_address,
                byte_count: u64::try_from(binding.byte_count).ok()?,
            };
            if trace_range.is_some_and(|(address, byte_count)| {
                guest_ranges_overlap(address, byte_count, writer.address, writer.byte_count)
            }) {
                eprintln!(
                    "[REMu][METAL][GUEST-WRITE] stage={stage} sequence={sequence} shader={shader:#x} address={:#x} bytes={} snapshot={} writeback={} pcs={:x?}",
                    binding.base_address,
                    binding.byte_count,
                    binding.data.len() == binding.byte_count,
                    binding.write_back_to_guest,
                    binding.instruction_pcs,
                );
            }
            Some(writer)
        })
        .collect::<Vec<_>>();
    if writable.is_empty() || !retain_history {
        return;
    }
    const CAPACITY: usize = 16_384;
    let mut writers = diagnostic_global_writers()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    for writer in writable {
        if writers.len() == CAPACITY {
            writers.pop_front();
        }
        writers.push_back(writer);
    }
}

pub(super) fn diagnostic_guest_write_trace_range() -> Option<(u64, u64)> {
    static RANGE: OnceLock<Option<(u64, u64)>> = OnceLock::new();
    *RANGE.get_or_init(|| {
        let value = std::env::var("REMU_TRACE_GUEST_WRITE_RANGE").ok()?;
        let (start, end) = value.split_once(':')?;
        let start = parse_diagnostic_hex_address(start)?;
        let end = parse_diagnostic_hex_address(end)?;
        (end > start).then_some((start, end - start))
    })
}

pub(super) fn diagnostic_global_writers() -> &'static Mutex<VecDeque<DiagnosticGlobalWriter>> {
    static WRITERS: OnceLock<Mutex<VecDeque<DiagnosticGlobalWriter>>> = OnceLock::new();
    WRITERS.get_or_init(|| Mutex::new(VecDeque::new()))
}

pub(super) fn diagnostic_overlapping_global_writers(
    descriptor: GuestTextureDescriptor,
) -> Vec<DiagnosticGlobalWriter> {
    let Some(byte_count) = guest_texture_byte_count(descriptor) else {
        return Vec::new();
    };
    diagnostic_global_writers()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .iter()
        .rev()
        .filter(|writer| {
            guest_ranges_overlap(
                descriptor.address,
                byte_count,
                writer.address,
                writer.byte_count,
            )
        })
        .take(32)
        .copied()
        .collect()
}

pub(super) fn evaluation_samples_large_linear_video(evaluation: &ShaderEvaluation) -> bool {
    sampled_image_bindings(evaluation).any(|binding| {
        decode_image_descriptor(&binding.resource_descriptor).is_ok_and(|descriptor| {
            descriptor.guest.tile_mode == 0
                && descriptor.guest.width >= 700
                && descriptor.guest.height >= 500
                && matches!(
                    descriptor.guest.pixel_format,
                    MTLPixelFormat::R8Unorm | MTLPixelFormat::RG8Unorm
                )
        })
    })
}

pub(super) fn dump_msl_pixel_filter_matches(pixel_shader_address: u64) -> bool {
    let Some(filter) = std::env::var_os("REMU_DUMP_MSL_PS") else {
        return true;
    };
    let filter = filter.to_string_lossy();
    let filter = filter.strip_prefix("0x").unwrap_or(&filter);
    u64::from_str_radix(filter, 16).is_ok_and(|filter| filter == pixel_shader_address)
}

pub(super) fn dump_msl_compute_filter_matches(compute_shader_address: u64) -> bool {
    let Some(filter) = std::env::var_os("REMU_DUMP_MSL_CS") else {
        return false;
    };
    let filter = filter.to_string_lossy();
    if filter.eq_ignore_ascii_case("all") {
        return true;
    }
    let filter = filter.strip_prefix("0x").unwrap_or(&filter);
    u64::from_str_radix(filter, 16).is_ok_and(|filter| filter == compute_shader_address)
}

pub(super) fn dump_msl_instruction_filter_matches(pixel_instruction_count: usize) -> bool {
    let Some(filter) = std::env::var_os("REMU_DUMP_MSL_PS_INSNS") else {
        return true;
    };
    filter
        .to_string_lossy()
        .parse::<usize>()
        .is_ok_and(|filter| filter == pixel_instruction_count)
}

pub(super) fn diagnostic_procedural_rectangle_fallback() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var_os("REMU_DIAGNOSTIC_PROCEDURAL_RECTANGLE").is_some_and(|value| value == "1")
    })
}
