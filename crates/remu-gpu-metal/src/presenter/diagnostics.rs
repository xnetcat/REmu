// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

use super::{
    CB_BLEND0_CONTROL, CB_TARGET_MASK, DecodedImageDescriptor, DiagnosticScanoutWriter,
    GuestTextureDescriptor, UploadedGlobalBuffer, decode_depth_target, decode_image_descriptor,
    decode_metal_blend_attachment, decode_raster_state, decode_render_targets, decode_scissor,
    decode_viewport, guest_ranges_overlap, guest_texture_byte_count,
    guest_texture_descriptors_overlap, guest_texture_descriptors_share_storage,
    is_storage_image_opcode, is_writable_image_opcode, register_value, sampled_image_bindings,
};
use metal::MTLPixelFormat;
use remu_gpu::{ComputeDispatch, GpuMemoryCopy, GpuRegister, IndexedDraw, ShaderSnapshot};
use remu_memory::{
    GuestWriteSite, guest_shared_backing_identity, guest_shared_backing_ranges_overlap,
    guest_write_journal_enabled, record_guest_write,
};
use remu_shader::{
    GlobalMemoryBinding, ImageBinding, Program, ShaderEvaluation, ShaderMetadata,
    image_is_comparison,
};
use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};

pub(super) const DIAGNOSTIC_VERTEX_POSITION_VERTEX_CAPACITY: usize = 262_144;

pub(super) const DIAGNOSTIC_VERTEX_STATE_WORDS_PER_VERTEX: usize = 24;

pub(super) const DIAGNOSTIC_VERTEX_POSITION_BUFFER_INDEX: u64 = 30;

pub(super) const DIAGNOSTIC_PIXEL_OUTPUT_WIDTH: usize = 1_920;

pub(super) const DIAGNOSTIC_PIXEL_OUTPUT_HEIGHT: usize = 1_080;

pub(super) const DIAGNOSTIC_PIXEL_OUTPUT_BUFFER_INDEX: u64 = 30;

/// Whether draw execution is under an opt-in diagnostic which can alter the normal presenter
/// path, inspect native resources, drain Metal work, or write an artifact.
///
/// Shared-event draw preflight treats the diagnostic namespaces as default-deny. Keeping that
/// policy here means a new `REMU_DUMP_*`, `REMU_TRACE_*`, or `REMU_DIAGNOSTIC_*` switch cannot be
/// accidentally GPU-ordered merely because the scheduler's allow-list was not updated.
pub(super) fn draw_diagnostics_are_host_effectful() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::vars_os().any(|(name, value)| {
            let name = name.to_string_lossy();
            let selected_namespace = name.starts_with("REMU_DUMP_")
                || name.starts_with("REMU_TRACE_")
                || name.starts_with("REMU_DIAGNOSTIC_");
            let selected_capture = matches!(
                name.as_ref(),
                "REMU_CAPTURE_FRAMES" | "REMU_CAPTURE_TARGET_SAMPLES"
            );
            let selected_shader_skip = matches!(name.as_ref(), "REMU_SKIP_ES" | "REMU_SKIP_PS");
            (selected_namespace || selected_capture || selected_shader_skip)
                && !value.is_empty()
                && value != "0"
        })
    })
}

pub(super) fn trace_image_descriptors() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("REMU_TRACE_IMAGES").is_some_and(|value| value == "1"))
}

/// Traces single-channel sampled images (Slate/UMG font atlases) at descriptor decode time:
/// the raw T# words, the decoded destination select, and the uploaded payload occupancy. The
/// glyph-tofu class of defect needs exactly this evidence to separate wrong atlas bytes from a
/// lost channel swizzle.
pub(super) fn trace_glyph_atlas() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var_os("REMU_TRACE_GLYPH_ATLAS").is_some_and(|value| value == "1")
    })
}

pub(super) fn diagnostic_dump_glyph_draw() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED
        .get_or_init(|| std::env::var_os("REMU_DUMP_GLYPH_DRAW").is_some_and(|value| value == "1"))
}

pub(super) fn evaluation_samples_glyph_atlas(evaluation: &ShaderEvaluation) -> bool {
    evaluation.image_bindings.iter().any(|binding| {
        binding.opcode.starts_with("ImageSample")
            && decode_image_descriptor(&binding.resource_descriptor).is_ok_and(|descriptor| {
                descriptor.guest.tile_mode == 9
                    && descriptor.destination_select == 0x800
                    && matches!(
                        descriptor.guest.pixel_format,
                        MTLPixelFormat::R8Unorm | MTLPixelFormat::A8Unorm
                    )
            })
    })
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

fn descriptors_are_nv12_plane_pair(
    left: DecodedImageDescriptor,
    right: DecodedImageDescriptor,
) -> bool {
    let (luma, chroma) = if left.guest.pixel_format == MTLPixelFormat::R8Unorm
        && right.guest.pixel_format == MTLPixelFormat::RG8Unorm
    {
        (left, right)
    } else if right.guest.pixel_format == MTLPixelFormat::R8Unorm
        && left.guest.pixel_format == MTLPixelFormat::RG8Unorm
    {
        (right, left)
    } else {
        return false;
    };
    luma.guest.tile_mode == 0
        && chroma.guest.tile_mode == 0
        && luma.texture_type == 9
        && chroma.texture_type == 9
        && luma.layer_count == 1
        && chroma.layer_count == 1
        && chroma.guest.width.checked_mul(2) == Some(luma.guest.width)
        && chroma.guest.height.checked_mul(2) == Some(luma.guest.height)
}

fn descriptor_is_planar_video_plane(
    candidate: DecodedImageDescriptor,
    descriptors: &[DecodedImageDescriptor],
) -> bool {
    descriptors
        .iter()
        .copied()
        .any(|other| descriptors_are_nv12_plane_pair(candidate, other))
}

fn descriptor_is_large_linear_video_plane(descriptor: DecodedImageDescriptor) -> bool {
    descriptor.guest.tile_mode == 0
        && descriptor.guest.width >= 700
        && descriptor.guest.height >= 500
        && matches!(
            descriptor.guest.pixel_format,
            MTLPixelFormat::R8Unorm | MTLPixelFormat::RG8Unorm
        )
}

pub(super) fn video_target_binding_summary(evaluation: &ShaderEvaluation) -> Option<String> {
    let decoded = evaluation
        .image_bindings
        .iter()
        .filter_map(|binding| {
            let descriptor = decode_image_descriptor(&binding.resource_descriptor).ok()?;
            (!is_writable_image_opcode(binding.opcode)).then_some((binding, descriptor))
        })
        .collect::<Vec<_>>();
    let descriptors = decoded
        .iter()
        .map(|(_, descriptor)| *descriptor)
        .collect::<Vec<_>>();
    let bindings = decoded
        .into_iter()
        .filter(|(_, descriptor)| {
            descriptor_is_large_linear_video_plane(*descriptor)
                || descriptor_is_planar_video_plane(*descriptor, &descriptors)
        })
        .map(|(binding, descriptor)| {
            format!(
                "pc={:#x} op={:?} addr={:#x} {}x{} pitch={} format={:?} tile={} base_mip={} mips={} storage={}",
                binding.pc,
                binding.opcode,
                descriptor.guest.address,
                descriptor.guest.width,
                descriptor.guest.height,
                descriptor.guest.pitch,
                descriptor.guest.pixel_format,
                descriptor.guest.tile_mode,
                descriptor.base_mip_level,
                descriptor.mip_level_count,
                is_storage_image_opcode(binding.opcode),
            )
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

fn diagnostic_video_chain_match(
    tracked: &[(GuestTextureDescriptor, DiagnosticVideoChainProducer)],
    sampled: GuestTextureDescriptor,
) -> Option<(GuestTextureDescriptor, DiagnosticVideoChainProducer)> {
    // HashMap iteration order is intentionally unspecified. Prefer the exact producer before
    // progressively weaker alias relationships so an overlapping transient allocation cannot
    // make the diagnostic chain hide a producer for the descriptor that is actually sampled.
    tracked
        .iter()
        .copied()
        .find(|(candidate, _)| *candidate == sampled)
        .or_else(|| {
            tracked
                .iter()
                .copied()
                .find(|(candidate, _)| guest_texture_descriptors_share_storage(*candidate, sampled))
        })
        .or_else(|| {
            tracked
                .iter()
                .copied()
                .find(|(candidate, _)| guest_texture_descriptors_overlap(*candidate, sampled))
        })
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
            diagnostic_video_chain_match(&tracked, descriptor.guest)
                .map(|(candidate, producer)| (binding, descriptor, candidate, producer))
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

fn record_diagnostic_video_chain_texture_in(
    tracked: &mut HashMap<GuestTextureDescriptor, DiagnosticVideoChainProducer>,
    descriptor: GuestTextureDescriptor,
    stage: &'static str,
    sequence: u64,
    shader: u64,
    generation: u32,
) {
    // A GPU write through one alias makes every overlapping interpretation stale. Mirror the
    // resident-texture lifetime rule here; otherwise an old exact descriptor can hide the newer
    // physical producer, or HashMap order can select either one nondeterministically.
    tracked.retain(|candidate, _| !guest_texture_descriptors_overlap(*candidate, descriptor));
    tracked.insert(
        descriptor,
        DiagnosticVideoChainProducer {
            stage,
            sequence,
            shader,
            generation,
        },
    );
}

pub(super) fn record_diagnostic_video_chain_texture(
    descriptor: GuestTextureDescriptor,
    stage: &'static str,
    sequence: u64,
    shader: u64,
    generation: u32,
) {
    record_diagnostic_video_chain_texture_in(
        &mut diagnostic_video_chain_textures()
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner),
        descriptor,
        stage,
        sequence,
        shader,
        generation,
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
    static LOGGED_DRAWS: AtomicU64 = AtomicU64::new(0);
    let logged = LOGGED_DRAWS.fetch_add(1, Ordering::Relaxed) + 1;
    if logged <= 16 || logged.is_power_of_two() {
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
    static LOGGED_DISPATCHES: AtomicU64 = AtomicU64::new(0);
    let logged = LOGGED_DISPATCHES.fetch_add(1, Ordering::Relaxed) + 1;
    if logged <= 16 || logged.is_power_of_two() {
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

#[derive(Debug, Eq, PartialEq)]
enum DiagnosticScanoutWriterFilter {
    Disabled,
    All,
    Addresses(Vec<u64>),
}

fn parse_diagnostic_scanout_writer_filter(value: &str) -> DiagnosticScanoutWriterFilter {
    if value.trim().eq_ignore_ascii_case("auto") {
        return DiagnosticScanoutWriterFilter::All;
    }
    let addresses = value
        .split(',')
        .filter_map(|part| parse_diagnostic_hex_address(part.trim()))
        .collect::<Vec<_>>();
    if addresses.is_empty() {
        DiagnosticScanoutWriterFilter::Disabled
    } else {
        DiagnosticScanoutWriterFilter::Addresses(addresses)
    }
}

/// Selects scanout writers for an ordered, per-draw source/capture trace.
///
/// This diagnostic is deliberately narrower than `REMU_TRACE_SCANOUT_WRITES`: a title can issue
/// dozens of UI writers per frame, and settling all of them would perturb the workload beyond
/// recognition. The filter is either `auto` for every known scanout writer after the minimum
/// draw, or comma-separated pixel-shader addresses in hexadecimal.
pub(super) fn diagnostic_scanout_writer_selected(pixel_shader_address: u64) -> bool {
    static FILTER: OnceLock<DiagnosticScanoutWriterFilter> = OnceLock::new();
    match FILTER.get_or_init(|| {
            std::env::var("REMU_DIAGNOSTIC_SCANOUT_WRITERS")
                .ok()
                .map(|value| parse_diagnostic_scanout_writer_filter(&value))
                .unwrap_or(DiagnosticScanoutWriterFilter::Disabled)
        }) {
        DiagnosticScanoutWriterFilter::Disabled => false,
        DiagnosticScanoutWriterFilter::All => true,
        DiagnosticScanoutWriterFilter::Addresses(addresses) => {
            addresses.contains(&pixel_shader_address)
        }
    }
}

/// Allocates a bounded trace slot for a selected scanout writer.
pub(super) fn diagnostic_scanout_writer_slot(
    draw_sequence: u64,
    pixel_shader_address: u64,
) -> Option<u64> {
    if draw_sequence < diagnostic_dump_target_minimum_draw()
        || !diagnostic_scanout_writer_selected(pixel_shader_address)
    {
        return None;
    }
    static LIMIT: OnceLock<u64> = OnceLock::new();
    let limit = *LIMIT.get_or_init(|| {
        std::env::var("REMU_DIAGNOSTIC_SCANOUT_WRITER_LIMIT")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(96)
    });
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let slot = NEXT.fetch_add(1, Ordering::Relaxed);
    (slot < limit).then_some(slot)
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

/// Opt-in: skip encoding draws whose pixel shader address is listed in `REMU_SKIP_PS`
/// (comma-separated hex, optional `0x` prefix).
pub(super) fn diagnostic_skip_pixel_shader(pixel_shader_address: u64) -> bool {
    static FILTER: OnceLock<Vec<u64>> = OnceLock::new();
    let filters = FILTER.get_or_init(|| {
        std::env::var("REMU_SKIP_PS")
            .ok()
            .map(|value| {
                value
                    .split(',')
                    .filter_map(|part| {
                        let part = part.trim().strip_prefix("0x").unwrap_or(part.trim());
                        u64::from_str_radix(part, 16).ok()
                    })
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default()
    });
    !filters.is_empty() && filters.contains(&pixel_shader_address)
}

/// Opt-in: skip encoding draws whose export/ES shader address is listed in `REMU_SKIP_ES`.
pub(super) fn diagnostic_skip_export_shader(export_shader_address: u64) -> bool {
    static FILTER: OnceLock<Vec<u64>> = OnceLock::new();
    let filters = FILTER.get_or_init(|| {
        std::env::var("REMU_SKIP_ES")
            .ok()
            .map(|value| {
                value
                    .split(',')
                    .filter_map(|part| {
                        let part = part.trim().strip_prefix("0x").unwrap_or(part.trim());
                        u64::from_str_radix(part, 16).ok()
                    })
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default()
    });
    !filters.is_empty() && filters.contains(&export_shader_address)
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

pub(super) const fn diagnostic_target_address_matches(
    expected: Option<u64>,
    target_address: u64,
    target_width: u32,
    target_height: u32,
) -> bool {
    match expected {
        Some(0) => target_width == 1_920 && target_height == 1_080,
        Some(expected) => target_address == expected,
        None => false,
    }
}

pub(super) fn diagnostic_dump_target_pixel_shader() -> Option<u64> {
    static ADDRESS: OnceLock<Option<u64>> = OnceLock::new();
    *ADDRESS.get_or_init(|| {
        let value = std::env::var("REMU_DUMP_TARGET_PS").ok()?;
        let value = value.strip_prefix("0x").unwrap_or(&value);
        u64::from_str_radix(value, 16).ok()
    })
}

fn diagnostic_dump_target_sample_address() -> Option<u64> {
    static ADDRESS: OnceLock<Option<u64>> = OnceLock::new();
    *ADDRESS.get_or_init(|| {
        let value = std::env::var("REMU_DUMP_TARGET_SAMPLE_ADDRESS").ok()?;
        let value = value.strip_prefix("0x").unwrap_or(&value);
        u64::from_str_radix(value, 16).ok()
    })
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct DiagnosticTargetDrawFilter {
    pixel_shader_address: Option<u64>,
    target_address: Option<u64>,
    sampled_address: Option<u64>,
    minimum_draw: u64,
}

impl DiagnosticTargetDrawFilter {
    #[must_use]
    pub(super) const fn matches(
        self,
        draw_sequence: u64,
        pixel_shader_address: u64,
        target_address: u64,
        target_width: u32,
        target_height: u32,
    ) -> bool {
        if draw_sequence < self.minimum_draw {
            return false;
        }
        if let Some(expected) = self.pixel_shader_address
            && pixel_shader_address != expected
        {
            return false;
        }
        if let Some(expected) = self.target_address {
            if expected == 0 {
                if target_width != 1_920 || target_height != 1_080 {
                    return false;
                }
            } else if target_address != expected {
                return false;
            }
        }
        true
    }

    pub(super) fn matches_sampled_addresses(
        self,
        addresses: impl IntoIterator<Item = u64>,
    ) -> bool {
        self.sampled_address
            .is_none_or(|expected| addresses.into_iter().any(|address| address == expected))
    }
}

pub(super) fn diagnostic_target_draw_filter() -> Option<DiagnosticTargetDrawFilter> {
    let pixel_shader_address = diagnostic_dump_target_pixel_shader();
    let target_address = diagnostic_dump_target_address();
    let sampled_address = diagnostic_dump_target_sample_address();
    if pixel_shader_address.is_none() && target_address.is_none() && sampled_address.is_none() {
        return None;
    }
    Some(DiagnosticTargetDrawFilter {
        pixel_shader_address,
        target_address,
        sampled_address,
        minimum_draw: diagnostic_dump_target_minimum_draw(),
    })
}

const fn is_diagnostic_context_register(offset: u32) -> bool {
    matches!(
        offset,
        0x191..=0x1b0 | 0x1b3 | 0x1b4 | 0x1b6 | 0x1e0..=0x1e7 | 0x291 | 0x29b
    )
}

pub(super) fn diagnostic_context_register_snapshot(registers: &[GpuRegister]) -> Vec<GpuRegister> {
    registers
        .iter()
        .copied()
        .filter(|register| is_diagnostic_context_register(register.offset))
        .collect()
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct DiagnosticProgramSummary {
    address: u64,
    decoded_byte_count: usize,
    instruction_count: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct DiagnosticShaderSummary {
    stage: &'static str,
    bound_address: u64,
    prefix: [u32; 4],
    header_address: u64,
    program: Option<DiagnosticProgramSummary>,
    supplemental_programs: Vec<DiagnosticProgramSummary>,
    metadata: Option<ShaderMetadata>,
}

fn diagnostic_program_summary(program: &Program) -> DiagnosticProgramSummary {
    let decoded_byte_count = program
        .instructions
        .iter()
        .filter_map(|instruction| {
            usize::try_from(instruction.pc)
                .ok()?
                .checked_add(instruction.words.len().checked_mul(4)?)
        })
        .max()
        .unwrap_or_default();
    DiagnosticProgramSummary {
        address: program.address,
        decoded_byte_count,
        instruction_count: program.instructions.len(),
    }
}

pub(super) fn diagnostic_shader_summary(
    stage: &'static str,
    bound_address: u64,
    prefix: [u32; 4],
    snapshot: Option<&ShaderSnapshot>,
) -> DiagnosticShaderSummary {
    DiagnosticShaderSummary {
        stage,
        bound_address,
        prefix,
        header_address: snapshot.map_or(0, |snapshot| snapshot.header_address),
        program: snapshot.map(|snapshot| diagnostic_program_summary(&snapshot.program)),
        supplemental_programs: snapshot
            .map(|snapshot| {
                snapshot
                    .supplemental_programs
                    .iter()
                    .map(diagnostic_program_summary)
                    .collect()
            })
            .unwrap_or_default(),
        metadata: snapshot
            .and_then(|snapshot| snapshot.metadata.as_deref())
            .cloned(),
    }
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
        || diagnostic_guest_write_trace_range().is_some()
        || diagnostic_dump_target_address().is_some()
        || diagnostic_dump_target_pixel_shader().is_some()
        || diagnostic_dump_target_sample_address().is_some()
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

pub(super) fn diagnostic_dump_target_index_count() -> Option<u32> {
    static COUNT: OnceLock<Option<u32>> = OnceLock::new();
    *COUNT.get_or_init(|| {
        std::env::var("REMU_DUMP_TARGET_INDEX_COUNT")
            .ok()
            .and_then(|value| value.parse().ok())
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

#[derive(Debug, Eq, PartialEq)]
pub(super) struct DiagnosticUploadedGlobalBinding {
    pub(super) offset: u64,
    pub(super) buffer_length: u64,
    pub(super) guest_memory: bool,
    pub(super) bytes: Result<Vec<u8>, String>,
}

fn diagnostic_uploaded_global_binding_range(
    buffer_length: u64,
    offset: u64,
    binding_byte_count: usize,
) -> Result<(usize, usize), String> {
    let byte_count = binding_byte_count.min(256);
    let start = usize::try_from(offset)
        .map_err(|_| format!("Metal buffer offset {offset:#x} does not fit usize"))?;
    let end = start.checked_add(byte_count).ok_or_else(|| {
        format!("Metal buffer range offset={offset:#x} bytes={byte_count} overflows")
    })?;
    if u64::try_from(end).map_or(true, |end| end > buffer_length) {
        return Err(format!(
            "Metal buffer range offset={offset:#x} bytes={byte_count} exceeds length={buffer_length}"
        ));
    }
    Ok((start, byte_count))
}

/// Captures the bytes from the exact Metal buffer and offset bound for a shader global.
///
/// Callers must first complete every command buffer that can produce or consume this binding.
/// In particular, a forwarded GPU writeback can contain newer data than the packet-time
/// `GlobalMemoryBinding::data`. Selected target dumps already establish that completion boundary;
/// keeping the wait at the caller avoids adding stalls to ordinary rendering.
pub(super) fn diagnostic_uploaded_global_binding_head(
    binding: &GlobalMemoryBinding,
    uploaded: &UploadedGlobalBuffer,
) -> DiagnosticUploadedGlobalBinding {
    let buffer_length = uploaded.buffer.length();
    let bytes = diagnostic_uploaded_global_binding_range(
        buffer_length,
        uploaded.offset,
        binding.byte_count,
    )
    .and_then(|(start, byte_count)| {
        if byte_count == 0 {
            return Ok(Vec::new());
        }
        let contents = uploaded.buffer.contents().cast::<u8>();
        if contents.is_null() {
            return Err("Metal buffer storage is not CPU-visible".to_owned());
        }
        // SAFETY: the range helper proved `start + byte_count <= buffer.length()`. The caller
        // completed every command buffer that can access this shared/managed resource, and
        // `contents` was checked for CPU visibility before constructing the slice.
        Ok(unsafe { std::slice::from_raw_parts(contents.add(start), byte_count) }.to_vec())
    });
    DiagnosticUploadedGlobalBinding {
        offset: uploaded.offset,
        buffer_length,
        guest_memory: uploaded.guest_memory,
        bytes,
    }
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
    if let Some(byte_count) = guest_texture_byte_count(descriptor)
        && let Some(overlap) = diagnostic_guest_write_overlap(descriptor.address, byte_count)
    {
        eprintln!(
            "[REMu][METAL][GUEST-WRITE] kind=texture-encode overlap={overlap} stage={stage} sequence={sequence} shader={shader:#x} address={:#x} bytes={byte_count:#x} identity={:?} descriptor={descriptor:?}",
            descriptor.address,
            guest_shared_backing_identity(descriptor.address, byte_count),
        );
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

pub(super) fn record_diagnostic_memory_copy(copy: &GpuMemoryCopy) {
    if !diagnostic_texture_writer_enabled() {
        return;
    }
    if let Some(overlap) = diagnostic_guest_write_overlap(copy.destination_address, copy.byte_count)
    {
        eprintln!(
            "[REMu][METAL][GUEST-WRITE] kind=dma overlap={overlap} source={:#x} destination={:#x} bytes={:#x} source-selection={} destination-selection={} destination-identity={:?}",
            copy.source_address,
            copy.destination_address,
            copy.byte_count,
            copy.source_selection,
            copy.destination_selection,
            guest_shared_backing_identity(copy.destination_address, copy.byte_count),
        );
    }
    const CAPACITY: usize = 8_192;
    let mut copies = diagnostic_memory_copies()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if copies.len() == CAPACITY {
        copies.pop_front();
    }
    copies.push_back(GpuMemoryCopy {
        source_address: copy.source_address,
        destination_address: copy.destination_address,
        byte_count: copy.byte_count,
        source_selection: copy.source_selection,
        destination_selection: copy.destination_selection,
        source_snapshot: None,
    });
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
        .cloned()
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
            if let Some(overlap) =
                diagnostic_guest_write_overlap(writer.address, writer.byte_count)
            {
                eprintln!(
                    "[REMu][METAL][GUEST-WRITE] kind=global-encode overlap={overlap} stage={stage} sequence={sequence} shader={shader:#x} address={:#x} bytes={} identity={:?} snapshot={} writeback={} pcs={:x?}",
                    binding.base_address,
                    binding.byte_count,
                    guest_shared_backing_identity(writer.address, writer.byte_count),
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

fn diagnostic_guest_write_overlap(address: u64, byte_count: u64) -> Option<&'static str> {
    let (trace_address, trace_byte_count) = diagnostic_guest_write_trace_range()?;
    if guest_ranges_overlap(trace_address, trace_byte_count, address, byte_count) {
        return Some("virtual");
    }
    guest_shared_backing_ranges_overlap(trace_address, trace_byte_count, address, byte_count)
        .then_some("physical-alias")
}

/// Records one host-authored guest range in the guest-write journal.
///
/// `payload` names the bytes published at `address`; `byte_count` may exceed it when the site only
/// has a prefix, which the journal reports rather than assuming.
pub(super) fn journal_guest_write(
    site: GuestWriteSite,
    address: u64,
    byte_count: u64,
    payload: &[u8],
) {
    record_guest_write(
        site,
        address,
        byte_count,
        payload,
        remu_gpu::gpu_completed_submission_sequence(),
    );
}

/// Records a scattered publication, one entry per contiguous run it actually wrote.
pub(super) fn journal_guest_write_segments(
    site: GuestWriteSite,
    address: u64,
    segments: &[(u64, &[u8])],
) {
    if !guest_write_journal_enabled() {
        return;
    }
    for (offset, bytes) in segments {
        journal_guest_write(
            site,
            address.wrapping_add(*offset),
            bytes.len() as u64,
            bytes,
        );
    }
}

pub(super) fn trace_diagnostic_guest_write(kind: &'static str, address: u64, byte_count: u64) {
    let Some(overlap) = diagnostic_guest_write_overlap(address, byte_count) else {
        return;
    };
    eprintln!(
        "[REMu][METAL][GUEST-WRITE] kind={kind} overlap={overlap} address={address:#x} bytes={byte_count:#x} identity={:?}",
        guest_shared_backing_identity(address, byte_count),
    );
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
    let descriptors = sampled_image_bindings(evaluation)
        .filter_map(|binding| decode_image_descriptor(&binding.resource_descriptor).ok())
        .collect::<Vec<_>>();
    descriptors.iter().copied().any(|descriptor| {
        descriptor_is_large_linear_video_plane(descriptor)
            || descriptor_is_planar_video_plane(descriptor, &descriptors)
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

#[cfg(test)]
mod tests {
    use super::super::ResourceSpan;
    use super::*;
    use metal::{Device, MTLResourceOptions};

    #[test]
    fn glyph_draw_match_requires_sampled_mode9_red_atlas_with_alpha_destination() {
        let binding = ImageBinding {
            pc: 0,
            opcode: remu_shader::Opcode::Known("ImageSample"),
            control: remu_shader::ImageControl {
                dmask: 0xf,
                vector_address: 0,
                address_registers: vec![0, 1],
                vector_data: 4,
                scalar_resource: 0,
                scalar_sampler: 8,
                dimension: 1,
                is_array: false,
                glc: false,
                slc: false,
                a16: false,
                d16: false,
                unorm: false,
            },
            resource_descriptor: vec![
                0x912e_bd00,
                0xc010_0000,
                0x00ff_c0ff,
                0x9090_0800,
                0,
                0x0070_0000,
                0,
                0,
            ]
            .into(),
            sampler_descriptor: vec![0x92, 0xfff000, 0x6500000, 0].into(),
            mip_level: None,
        };
        let evaluation = ShaderEvaluation {
            initial_scalar_registers: Vec::new().into(),
            scalar_registers: Vec::new().into(),
            scalar_memory_reads: Vec::new(),
            image_bindings: vec![binding.clone()],
            global_memory_bindings: Vec::new(),
            compute_system_registers: None,
            vertex_inputs: Vec::new(),
        };

        assert!(evaluation_samples_glyph_atlas(&evaluation));

        let mut wrong_operation = evaluation.clone();
        wrong_operation.image_bindings[0].opcode = remu_shader::Opcode::Known("ImageLoad");
        assert!(!evaluation_samples_glyph_atlas(&wrong_operation));

        let mut wrong_destination = evaluation;
        wrong_destination.image_bindings[0].resource_descriptor[3] = 0x9090_0000;
        assert!(!evaluation_samples_glyph_atlas(&wrong_destination));
    }

    #[test]
    fn video_chain_match_prefers_exact_descriptor_over_earlier_overlap() {
        let sampled = GuestTextureDescriptor {
            address: 0x1000,
            width: 4,
            height: 4,
            pitch: 4,
            tile_mode: 0,
            pixel_format: MTLPixelFormat::RGBA8Unorm,
            bytes_per_pixel: 4,
        };
        let overlapping = GuestTextureDescriptor {
            address: 0x1010,
            width: 1,
            height: 1,
            pitch: 1,
            tile_mode: 0,
            pixel_format: MTLPixelFormat::R8Unorm,
            bytes_per_pixel: 1,
        };
        let overlap_producer = DiagnosticVideoChainProducer {
            stage: "draw",
            sequence: 20,
            shader: 0x2000,
            generation: 1,
        };
        let exact_producer = DiagnosticVideoChainProducer {
            stage: "compute",
            sequence: 10,
            shader: 0x1000,
            generation: 3,
        };

        assert_eq!(
            diagnostic_video_chain_match(
                &[(overlapping, overlap_producer), (sampled, exact_producer)],
                sampled,
            )
            .map(|(descriptor, producer)| (descriptor, producer.sequence)),
            Some((sampled, exact_producer.sequence)),
        );
    }

    #[test]
    fn video_chain_write_retires_stale_overlapping_producer() {
        let old_surface = GuestTextureDescriptor {
            address: 0x1000,
            width: 4,
            height: 4,
            pitch: 4,
            tile_mode: 0,
            pixel_format: MTLPixelFormat::RGBA8Unorm,
            bytes_per_pixel: 4,
        };
        let new_alias = GuestTextureDescriptor {
            address: 0x1010,
            width: 1,
            height: 1,
            pitch: 1,
            tile_mode: 0,
            pixel_format: MTLPixelFormat::R8Unorm,
            bytes_per_pixel: 1,
        };
        let mut tracked = HashMap::new();

        record_diagnostic_video_chain_texture_in(
            &mut tracked,
            old_surface,
            "compute",
            10,
            0x1000,
            3,
        );
        record_diagnostic_video_chain_texture_in(&mut tracked, new_alias, "draw", 20, 0x2000, 1);

        assert!(!tracked.contains_key(&old_surface));
        assert_eq!(
            diagnostic_video_chain_match(
                &tracked
                    .iter()
                    .map(|entry| (*entry.0, *entry.1))
                    .collect::<Vec<_>>(),
                old_surface,
            )
            .map(|(descriptor, producer)| (descriptor, producer.sequence)),
            Some((new_alias, 20)),
        );
    }

    fn video_plane(
        address: u64,
        width: u32,
        height: u32,
        pixel_format: MTLPixelFormat,
    ) -> DecodedImageDescriptor {
        let bytes_per_pixel = if pixel_format == MTLPixelFormat::RG8Unorm {
            2
        } else {
            1
        };
        let guest = GuestTextureDescriptor {
            address,
            width,
            height,
            pitch: width,
            tile_mode: 0,
            pixel_format,
            bytes_per_pixel,
        };
        DecodedImageDescriptor {
            guest,
            span: ResourceSpan::for_texture(guest),
            destination_select: 0,
            base_mip_level: 0,
            mip_level_count: 1,
            texture_type: 9,
            layer_count: 1,
        }
    }

    #[test]
    fn planar_video_detection_accepts_pitched_nv12_plane_geometry() {
        let luma = video_plane(0x80b2_030000, 512, 150, MTLPixelFormat::R8Unorm);
        let chroma = video_plane(0x804e_fc0000, 256, 75, MTLPixelFormat::RG8Unorm);
        let descriptors = [luma, chroma];

        assert!(descriptors_are_nv12_plane_pair(luma, chroma));
        assert!(descriptor_is_planar_video_plane(luma, &descriptors));
        assert!(descriptor_is_planar_video_plane(chroma, &descriptors));
    }

    #[test]
    fn planar_video_detection_rejects_unrelated_linear_images() {
        let luma = video_plane(0x1000_0000, 512, 150, MTLPixelFormat::R8Unorm);
        let wrong_height = video_plane(0x1002_0000, 256, 76, MTLPixelFormat::RG8Unorm);
        let rgba = video_plane(0x1004_0000, 256, 75, MTLPixelFormat::RGBA8Unorm);

        assert!(!descriptors_are_nv12_plane_pair(luma, wrong_height));
        assert!(!descriptors_are_nv12_plane_pair(luma, rgba));
    }

    #[test]
    fn explicit_target_filter_conjoins_draw_shader_and_target() {
        let filter = DiagnosticTargetDrawFilter {
            pixel_shader_address: Some(0x0080_0e73_0000),
            target_address: Some(0x00ef_c200_0000),
            sampled_address: None,
            minimum_draw: 38,
        };

        assert!(filter.matches(38, 0x0080_0e73_0000, 0x00ef_c200_0000, 1_920, 1_080));
        assert!(!filter.matches(37, 0x0080_0e73_0000, 0x00ef_c200_0000, 1_920, 1_080));
        assert!(!filter.matches(38, 0x0080_0802_0000, 0x00ef_c200_0000, 1_920, 1_080));
        assert!(!filter.matches(38, 0x0080_0e73_0000, 0x00ef_c000_0000, 1_920, 1_080));
    }

    #[test]
    fn scanout_writer_shader_filter_accepts_prefixed_and_plain_hex() {
        assert_eq!(
            parse_diagnostic_scanout_writer_filter(
                "0x8059850000, 8051dc0000,invalid,,0X80d3b70000"
            ),
            DiagnosticScanoutWriterFilter::Addresses(vec![
                0x8059_850000,
                0x8051_dc0000,
                0x80d3_b70000
            ])
        );
    }

    #[test]
    fn scanout_writer_auto_filter_selects_runtime_discovered_writers() {
        assert_eq!(
            parse_diagnostic_scanout_writer_filter(" AUTO "),
            DiagnosticScanoutWriterFilter::All
        );
        assert_eq!(
            parse_diagnostic_scanout_writer_filter("invalid,,"),
            DiagnosticScanoutWriterFilter::Disabled
        );
    }

    #[test]
    fn automatic_target_filter_selects_only_display_extent() {
        let filter = DiagnosticTargetDrawFilter {
            pixel_shader_address: None,
            target_address: Some(0),
            sampled_address: None,
            minimum_draw: 0,
        };

        assert!(filter.matches(1, 0x0080_0e73_0000, 0x1234, 1_920, 1_080));
        assert!(!filter.matches(1, 0x0080_0e73_0000, 0x1234, 1_280, 720));
    }

    #[test]
    fn target_address_selection_can_enable_pipeline_instrumentation() {
        assert!(diagnostic_target_address_matches(
            Some(0x00ef_c000_0000),
            0x00ef_c000_0000,
            1_280,
            720
        ));
        assert!(diagnostic_target_address_matches(
            Some(0),
            0x00ef_c000_0000,
            1_920,
            1_080
        ));
        assert!(!diagnostic_target_address_matches(
            Some(0),
            0x00ef_c000_0000,
            1_280,
            720
        ));
    }

    #[test]
    fn sampled_address_filter_selects_only_the_source_surface() {
        let filter = DiagnosticTargetDrawFilter {
            pixel_shader_address: Some(0x0080_0802_0000),
            target_address: Some(0x00ef_c000_0000),
            sampled_address: Some(0x0080_65b3_0000),
            minimum_draw: 0,
        };

        assert!(filter.matches_sampled_addresses([0x0080_65b3_0000]));
        assert!(filter.matches_sampled_addresses([0x1234, 0x0080_65b3_0000]));
        assert!(!filter.matches_sampled_addresses([0x0080_65b4_0000]));
        assert!(!filter.matches_sampled_addresses([]));
    }

    #[test]
    fn uploaded_global_binding_snapshot_uses_submitted_offset_and_extent() {
        let device = Device::system_default().expect("Metal device");
        let buffer = device.new_buffer(24, MTLResourceOptions::StorageModeShared);
        // SAFETY: the shared test buffer exposes exactly 24 writable bytes.
        let contents =
            unsafe { std::slice::from_raw_parts_mut(buffer.contents().cast::<u8>(), 24) };
        contents.copy_from_slice(&(0_u8..24).collect::<Vec<_>>());
        let binding = GlobalMemoryBinding {
            scalar_address: 4,
            base_address: 0x10000,
            instruction_pcs: Default::default(),
            byte_count: 6,
            data: vec![0xff; 6],
            writable: false,
            write_back_to_guest: false,
        };
        let uploaded = UploadedGlobalBuffer {
            buffer,
            offset: 9,
            guest_memory: false,
            physical_identity: None,
        };

        assert_eq!(
            diagnostic_uploaded_global_binding_head(&binding, &uploaded),
            DiagnosticUploadedGlobalBinding {
                offset: 9,
                buffer_length: 24,
                guest_memory: false,
                bytes: Ok(vec![9, 10, 11, 12, 13, 14]),
            }
        );
    }

    #[test]
    fn uploaded_global_binding_snapshot_rejects_out_of_bounds_extent() {
        let device = Device::system_default().expect("Metal device");
        let binding = GlobalMemoryBinding {
            scalar_address: 4,
            base_address: 0x10000,
            instruction_pcs: Default::default(),
            byte_count: 8,
            data: vec![0xff; 8],
            writable: false,
            write_back_to_guest: false,
        };
        let uploaded = UploadedGlobalBuffer {
            buffer: device.new_buffer(12, MTLResourceOptions::StorageModeShared),
            offset: 8,
            guest_memory: true,
            physical_identity: None,
        };
        let snapshot = diagnostic_uploaded_global_binding_head(&binding, &uploaded);

        assert_eq!(snapshot.offset, 8);
        assert_eq!(snapshot.buffer_length, 12);
        assert!(snapshot.bytes.is_err());
    }

    #[test]
    fn context_snapshot_captures_only_requested_raw_registers() {
        let registers = [
            GpuRegister {
                offset: 0x190,
                value: 0xaaaa_aaaa,
            },
            GpuRegister {
                offset: 0x191,
                value: 0x1111_1111,
            },
            GpuRegister {
                offset: 0x1b0,
                value: 0xb0b0_b0b0,
            },
            GpuRegister {
                offset: 0x1b1,
                value: 0xbbbb_bbbb,
            },
            GpuRegister {
                offset: 0x1b3,
                value: 0xb3b3_b3b3,
            },
            GpuRegister {
                offset: 0x1b4,
                value: 0xb4b4_b4b4,
            },
            GpuRegister {
                offset: 0x1b6,
                value: 0xb6b6_b6b6,
            },
            GpuRegister {
                offset: 0x291,
                value: 0x2912_9129,
            },
            GpuRegister {
                offset: 0x29b,
                value: 0x29b2_9b29,
            },
            GpuRegister {
                offset: 0x29c,
                value: 0xcccc_cccc,
            },
        ];

        assert_eq!(
            diagnostic_context_register_snapshot(&registers),
            vec![
                registers[1],
                registers[2],
                registers[4],
                registers[5],
                registers[6],
                registers[7],
                registers[8],
            ]
        );
    }
}
