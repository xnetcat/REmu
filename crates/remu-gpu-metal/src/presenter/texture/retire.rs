// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

use super::{
    CachedSampledTexture, GuestTextureDescriptor, MultisampleResidentTexture,
    MultisampleTextureKey, ResidentGuestTexture, ResidentTextureMap, ResourceSpan,
    SampledTextureKey, guest_ranges_overlap, guest_texture_byte_count,
    quarantine_retired_resident,
};
use std::collections::HashMap;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc::{self};

pub(crate) fn trace_resident_texture_removal(reason: &str, resident: &ResidentGuestTexture) {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    if !*ENABLED.get_or_init(|| std::env::var_os("REMU_TRACE_TEXTURE_RESIDENCY").is_some()) {
        return;
    }
    static COUNT: AtomicUsize = AtomicUsize::new(0);
    if COUNT.fetch_add(1, Ordering::Relaxed) >= 1_024 {
        return;
    }
    let descriptor = resident.descriptor;
    let activity_stamp = |work: Option<u64>, flip: Option<u64>| {
        work.zip(flip).map_or_else(
            || "never".to_owned(),
            |(work, flip)| format!("{work}@f{flip}"),
        )
    };
    let (last_read, last_write) = resident.activity.map_or_else(
        || ("disabled".to_owned(), "disabled".to_owned()),
        |activity| {
            (
                activity_stamp(activity.last_read_work, activity.last_read_flip),
                activity_stamp(activity.last_write_work, activity.last_write_flip),
            )
        },
    );
    eprintln!(
        "[REMu][METAL][RESIDENCY-REMOVE] reason={reason} texture={:#x}/{}x{}/p{}/{:?}/tile{}/type{}/layers{}/mips{}/gpu{}/v{} dirty={} read={} write={}",
        descriptor.address,
        descriptor.width,
        descriptor.height,
        descriptor.pitch,
        descriptor.pixel_format,
        descriptor.tile_mode,
        resident.texture_type,
        resident.layer_count,
        resident.mip_level_count,
        resident.gpu_written(),
        resident.content_version,
        resident.write_watch.is_dirty(),
        last_read,
        last_write,
    );
}

pub(crate) fn retire_sampled_texture(texture: CachedSampledTexture) {
    static RETIRER: OnceLock<mpsc::Sender<CachedSampledTexture>> = OnceLock::new();
    let sender = RETIRER.get_or_init(|| {
        let (sender, receiver) = mpsc::channel::<CachedSampledTexture>();
        if let Err(error) = std::thread::Builder::new()
            .name("REMu texture reclaimer".into())
            .spawn(move || {
                while let Ok(texture) = receiver.recv() {
                    drop(texture);
                }
            })
        {
            eprintln!("[REMu][METAL] could not start texture reclaimer: {error}");
        }
        sender
    });
    if let Err(error) = sender.send(texture) {
        drop(error.0);
    }
}

pub(crate) fn retire_sampled_textures_matching(
    cache: &mut HashMap<SampledTextureKey, CachedSampledTexture>,
    mut should_retire: impl FnMut(SampledTextureKey, &CachedSampledTexture) -> bool,
) {
    // Dropping a write watch republishes the signal-handler's immutable page-owner snapshot.
    // Invalidating a render graph can evict hundreds of sampled aliases at once, so doing those
    // drops inside HashMap::retain stalls the presenter for an entire guest frame. Remove the
    // entries immediately for correctness, then let the existing reclaimer perform their costly
    // destruction away from the event-loop thread.
    let retired = cache
        .iter()
        .filter_map(|(key, cached)| should_retire(*key, cached).then_some(*key))
        .collect::<Vec<_>>();
    for key in retired {
        if let Some(texture) = cache.remove(&key) {
            retire_sampled_texture(texture);
        }
    }
}

/// Retires every sampled interpretation that overlaps `span`, which the caller resolved at decode.
///
/// A `None` span names a descriptor the overlap predicate can never match, so nothing is retired.
pub(crate) fn retire_overlapping_sampled_textures(
    cache: &mut HashMap<SampledTextureKey, CachedSampledTexture>,
    span: Option<ResourceSpan>,
) {
    let Some(span) = span else {
        return;
    };
    retire_sampled_textures_matching(cache, |_, cached| {
        cached.overlap.is_some_and(|cached| span.overlaps(&cached))
    });
}

pub(crate) fn retire_sampled_textures_overlapping_writeback(
    cache: &mut HashMap<SampledTextureKey, CachedSampledTexture>,
    alias_ranges: &[(u64, u64)],
) {
    let retired = cache
        .iter()
        .filter_map(|(key, cached)| {
            cached
                .write_watch
                .overlaps_alias_ranges(alias_ranges)
                .then_some(*key)
        })
        .collect::<Vec<_>>();
    for key in retired {
        if let Some(texture) = cache.remove(&key) {
            retire_sampled_texture(texture);
        }
    }
}

pub(crate) fn retire_resident_textures_overlapping_writeback(
    cache: &mut ResidentTextureMap,
    retired_texture_quarantine: &mut HashMap<GuestTextureDescriptor, ResidentGuestTexture>,
    alias_ranges: &[(u64, u64)],
    source_descriptor: GuestTextureDescriptor,
    source_content_version: u64,
) {
    let retired = cache
        .iter()
        .filter_map(|(descriptor, resident)| {
            (*descriptor != source_descriptor
                && resident.content_version <= source_content_version
                && resident.write_watch.overlaps_alias_ranges(alias_ranges))
            .then_some(*descriptor)
        })
        .collect::<Vec<_>>();
    for descriptor in retired {
        if let Some(mut resident) = cache.remove(&descriptor) {
            trace_resident_texture_removal("overlapping-writeback", &resident);
            // The imminent writeback supersedes this resident's pixels, so it stops being
            // GPU-owned here, and its shape, watch and fingerprints stay adoptable. The
            // writeback notifies every intersecting watch, so a content-consuming adoption
            // revalidates and a stale entry can never serve pixels; a discarded-render-target
            // adoption reuses the allocation without reading them.
            resident.clear_gpu_authored();
            quarantine_retired_resident(retired_texture_quarantine, resident);
        }
    }
}

pub(crate) fn retire_multisample_textures_overlapping_writeback(
    cache: &mut HashMap<MultisampleTextureKey, MultisampleResidentTexture>,
    alias_ranges: &[(u64, u64)],
) {
    cache.retain(|key, _| {
        guest_texture_byte_count(key.descriptor).is_none_or(|byte_count| {
            !alias_ranges.iter().any(|&(address, alias_bytes)| {
                guest_ranges_overlap(address, alias_bytes, key.descriptor.address, byte_count)
            })
        })
    });
}
