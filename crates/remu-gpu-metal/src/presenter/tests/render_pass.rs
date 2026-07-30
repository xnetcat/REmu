// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

use metal::MTLPixelFormat;

use crate::presenter::{
    GuestRenderPassKey, GuestRenderPassKeyField, GuestRenderPassKeyTransition,
    GuestRenderPassKeyTransitionStatistics, GuestTextureDescriptor,
};

fn texture(address: u64) -> GuestTextureDescriptor {
    GuestTextureDescriptor {
        address,
        width: 1280,
        height: 720,
        pitch: 1280,
        tile_mode: 0x1b,
        pixel_format: MTLPixelFormat::RGBA8Unorm,
        bytes_per_pixel: 4,
    }
}

fn key() -> GuestRenderPassKey {
    GuestRenderPassKey {
        targets: vec![(texture(0x1000), 1, false, 2, 1, 0, 0, 1)],
        depth: Some((texture(0x2000), 1, 1, None, None)),
    }
}

fn transition(fields: &[GuestRenderPassKeyField]) -> GuestRenderPassKeyTransition {
    let mut transition = GuestRenderPassKeyTransition::default();
    for &field in fields {
        transition.mark(field, true);
    }
    transition
}

#[test]
fn render_pass_key_transition_classifies_color_fields_independently() {
    let current = key();
    assert_eq!(
        current.transition_to(&current),
        GuestRenderPassKeyTransition::default()
    );

    let mut next = current.clone();
    next.targets
        .push((texture(0x3000), 1, false, 2, 1, 0, 0, 1));
    assert_eq!(
        current.transition_to(&next),
        transition(&[GuestRenderPassKeyField::TargetCount])
    );

    let mut next = current.clone();
    next.targets[0].0.address += 0x1000;
    next.targets[0].0.width += 1;
    next.targets[0].0.height += 1;
    next.targets[0].0.pitch += 1;
    next.targets[0].0.tile_mode += 1;
    next.targets[0].0.pixel_format = MTLPixelFormat::BGRA8Unorm;
    next.targets[0].0.bytes_per_pixel = 8;
    next.targets[0].1 = 4;
    next.targets[0].2 = true;
    next.targets[0].3 += 1;
    next.targets[0].4 += 1;
    next.targets[0].5 += 1;
    next.targets[0].6 += 1;
    next.targets[0].7 += 1;
    assert_eq!(
        current.transition_to(&next),
        transition(&[
            GuestRenderPassKeyField::TargetAddress,
            GuestRenderPassKeyField::TargetWidth,
            GuestRenderPassKeyField::TargetHeight,
            GuestRenderPassKeyField::TargetPitch,
            GuestRenderPassKeyField::TargetTileMode,
            GuestRenderPassKeyField::TargetPixelFormat,
            GuestRenderPassKeyField::TargetBytesPerPixel,
            GuestRenderPassKeyField::TargetSampleCount,
            GuestRenderPassKeyField::TargetDirectScanout,
            GuestRenderPassKeyField::TargetTextureType,
            GuestRenderPassKeyField::TargetLayerCount,
            GuestRenderPassKeyField::TargetBaseLayer,
            GuestRenderPassKeyField::TargetMipLevel,
            GuestRenderPassKeyField::TargetMipLevelCount,
        ])
    );
}

#[test]
fn render_pass_key_transition_classifies_depth_fields_independently() {
    let current = key();

    let mut next = current.clone();
    next.depth = None;
    assert_eq!(
        current.transition_to(&next),
        transition(&[GuestRenderPassKeyField::DepthPresence])
    );

    let mut next = current.clone();
    let depth = next.depth.as_mut().expect("depth target");
    depth.0.address += 0x1000;
    depth.0.width += 1;
    depth.0.height += 1;
    depth.0.pitch += 1;
    depth.0.tile_mode += 1;
    depth.0.pixel_format = MTLPixelFormat::Depth32Float;
    depth.0.bytes_per_pixel = 8;
    depth.1 = 4;
    depth.2 += 1;
    depth.3 = Some(1.0_f32.to_bits());
    depth.4 = Some(7);
    assert_eq!(
        current.transition_to(&next),
        transition(&[
            GuestRenderPassKeyField::DepthAddress,
            GuestRenderPassKeyField::DepthWidth,
            GuestRenderPassKeyField::DepthHeight,
            GuestRenderPassKeyField::DepthPitch,
            GuestRenderPassKeyField::DepthTileMode,
            GuestRenderPassKeyField::DepthPixelFormat,
            GuestRenderPassKeyField::DepthBytesPerPixel,
            GuestRenderPassKeyField::DepthSampleCount,
            GuestRenderPassKeyField::DepthLayerCount,
            GuestRenderPassKeyField::DepthClearValue,
            GuestRenderPassKeyField::DepthClearSequence,
        ])
    );
}

#[test]
fn render_pass_key_transition_counters_count_each_cause_once_per_transition() {
    let current = key();
    let mut next = current.clone();
    next.targets[0].0.address += 0x1000;
    next.depth.as_mut().expect("depth target").3 = Some(0.0_f32.to_bits());
    let transition = current.transition_to(&next);
    let mut statistics = GuestRenderPassKeyTransitionStatistics::default();

    statistics.record(transition);
    statistics.record(transition);

    assert_eq!(statistics.target_address, 2);
    assert_eq!(statistics.depth_clear_value, 2);
    assert_eq!(
        statistics,
        GuestRenderPassKeyTransitionStatistics {
            target_address: 2,
            depth_clear_value: 2,
            ..GuestRenderPassKeyTransitionStatistics::default()
        }
    );
}
