// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

use super::{
    CachedSampledTexture, GuestTextureDescriptor, GuestTextureOverlapQuery,
    MultisampleResidentTexture, MultisampleTextureKey, ResidentGuestTexture, SampledTextureKey,
    guest_ranges_overlap, guest_texture_byte_count,
};
use std::collections::HashMap;
use std::sync::OnceLock;
use std::sync::mpsc::{self};

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
    mut should_retire: impl FnMut(SampledTextureKey) -> bool,
) {
    // Dropping a write watch republishes the signal-handler's immutable page-owner snapshot.
    // Invalidating a render graph can evict hundreds of sampled aliases at once, so doing those
    // drops inside HashMap::retain stalls the presenter for an entire guest frame. Remove the
    // entries immediately for correctness, then let the existing reclaimer perform their costly
    // destruction away from the event-loop thread.
    let retired = cache
        .keys()
        .copied()
        .filter(|key| should_retire(*key))
        .collect::<Vec<_>>();
    for key in retired {
        if let Some(texture) = cache.remove(&key) {
            retire_sampled_texture(texture);
        }
    }
}

pub(crate) fn retire_overlapping_sampled_textures(
    cache: &mut HashMap<SampledTextureKey, CachedSampledTexture>,
    descriptor: GuestTextureDescriptor,
) {
    let overlap = GuestTextureOverlapQuery::new(descriptor);
    retire_sampled_textures_matching(cache, |key| overlap.overlaps(key.descriptor));
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
    cache: &mut HashMap<GuestTextureDescriptor, ResidentGuestTexture>,
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
        cache.remove(&descriptor);
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
