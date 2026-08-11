// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

use super::{
    DecodedImageDescriptor, DecodedRenderTarget, GuestScissor, GuestTextureDescriptor,
    GuestTextureWriteWatch, OWNED_FALLBACK_FULL_RANGE, OWNED_RANGE_SET_COARSENED, RangeSet,
    ResourceSpan, compressed_texture_block_extent, guest_texture_byte_count,
    mode5_mip_chain_layout, mode64k_2d_mip_chain_layout, texture_element_layout, tiled_block_bytes,
    tiled_block_dimensions,
};
use metal::MTLScissorRect;
use std::sync::atomic::Ordering;

/// The region of a resident a single GPU write covers.
///
/// `level_count` and `layer_count` are counts from `first_level`/`first_layer`; `u32::MAX` names
/// every remaining one, so a caller that cannot bound the write does not have to know the
/// resident's shape.
#[derive(Clone, Copy, Debug)]
pub(crate) struct WrittenRegion {
    pub(crate) first_level: u32,
    pub(crate) level_count: u32,
    pub(crate) first_layer: u32,
    pub(crate) layer_count: u32,
    /// `None` means the whole level extent. A rect is in the written level's own texels, which is
    /// the space the scissor is already decoded and clamped in.
    pub(crate) rect: Option<MTLScissorRect>,
}

impl WrittenRegion {
    /// The conservative default: every level, every layer, every texel.
    pub(crate) fn whole_resource() -> Self {
        Self {
            first_level: 0,
            level_count: u32::MAX,
            first_layer: 0,
            layer_count: u32::MAX,
            rect: Self::WHOLE_EXTENT,
        }
    }

    pub(crate) fn render_target(target: DecodedRenderTarget, scissor: GuestScissor) -> Self {
        Self {
            first_level: target.mip_level,
            level_count: 1,
            // A pass may route primitives to any slice through the layer index, so layers are
            // deliberately not refined. Over-claiming a layer only costs a re-materialisation.
            first_layer: 0,
            layer_count: u32::MAX,
            rect: Self::scissor_rect(scissor),
        }
    }

    /// A depth attachment's written region. A clear covers the whole attachment.
    pub(crate) fn depth_target(scissor: GuestScissor, cleared: bool) -> Self {
        Self {
            rect: if cleared {
                Self::WHOLE_EXTENT
            } else {
                Self::scissor_rect(scissor)
            },
            ..Self::whole_resource()
        }
    }

    /// A storage image's written region: the bound mip range, whole extent.
    ///
    /// A dispatch's write extent is not statically known, so no rect is derived.
    pub(crate) fn storage_image(descriptor: DecodedImageDescriptor) -> Self {
        Self {
            first_level: descriptor.base_mip_level,
            level_count: descriptor.mip_level_count,
            first_layer: 0,
            layer_count: u32::MAX,
            rect: Self::WHOLE_EXTENT,
        }
    }

    const WHOLE_EXTENT: Option<MTLScissorRect> = None;

    fn scissor_rect(scissor: GuestScissor) -> Option<MTLScissorRect> {
        match scissor {
            GuestScissor::Rect(rect) => Some(rect),
            // An empty scissor encodes no draw at all, so the mark that follows it will be skipped
            // outright once the subtraction is live. Claiming zero bytes here instead would make
            // the write invisible to `gpu_written()` while the write still happens today.
            GuestScissor::Full | GuestScissor::Empty => Self::WHOLE_EXTENT,
        }
    }
}

/// The guest bytes a resident declares, as the range its write watch already covers.
///
/// The watch is the single source of truth for a resident's extent: it is the range every
/// snapshot, fingerprint and writeback already uses, and it accounts for mip chains and array
/// faces without duplicating their layout arithmetic.
pub(crate) fn resident_declared_span(write_watch: &GuestTextureWriteWatch) -> Option<ResourceSpan> {
    let (address, byte_count) = write_watch.range();
    (byte_count != 0).then(|| ResourceSpan::new(address, byte_count))
}

/// Byte ranges, in `span`'s offset space, that `region` of this resident covers.
///
/// Over-approximation is always safe for authority — claiming a byte a neighbour still holds costs
/// a re-materialisation, while missing one leaves that neighbour serving superseded pixels — so
/// every error path returns the resident's whole span.
pub(crate) fn written_byte_ranges(
    span: ResourceSpan,
    descriptor: GuestTextureDescriptor,
    texture_type: u32,
    layer_count: u32,
    mip_level_count: u32,
    region: WrittenRegion,
) -> RangeSet {
    let (base, end) = (span.range_start(), span.range_end());
    let Some(relative) = resource_relative_runs(
        descriptor,
        texture_type,
        layer_count,
        mip_level_count,
        region,
    ) else {
        return whole_span(base, end);
    };
    let mut rebased = RangeSet::default();
    for &(start, run_end) in relative.runs() {
        rebased.insert_run(
            base.saturating_add(start).min(end),
            base.saturating_add(run_end).min(end),
        );
    }
    // An empty result would make `gpu_written()` false for a write that really happened, so it is
    // treated as a failure of the region computation rather than as "no bytes were written".
    if rebased.is_empty() {
        return whole_span(base, end);
    }
    rebased
}

fn whole_span(base: u64, end: u64) -> RangeSet {
    OWNED_FALLBACK_FULL_RANGE.fetch_add(1, Ordering::Relaxed);
    RangeSet::from_run(base, end)
}

/// One level of one layer, in bytes relative to the resource's own base address.
#[derive(Clone, Copy, Debug)]
struct LevelExtent {
    offset: u64,
    len: u64,
    width: usize,
    height: usize,
    /// Texels between the starts of two consecutive block rows, which is the guest pitch for a
    /// single-level surface and the padded level width inside a mip chain.
    row_texels: usize,
    in_tail: bool,
}

fn resource_relative_runs(
    descriptor: GuestTextureDescriptor,
    texture_type: u32,
    layer_count: u32,
    mip_level_count: u32,
    region: WrittenRegion,
) -> Option<RangeSet> {
    // A mip chain only exists for a plain 2D single-layer surface, which is the same shape
    // `watch_resident_texture` restricts chains to; nothing else has a layout to answer from.
    if layer_count == 0
        || mip_level_count == 0
        || (mip_level_count > 1 && (texture_type != 9 || layer_count != 1))
    {
        return None;
    }
    let (_, _, _, bytes_per_element) = texture_element_layout(descriptor).ok()?;
    let face_bytes = guest_texture_byte_count(descriptor)?;
    let levels = level_extents(descriptor, mip_level_count)?;
    let first_level = region.first_level;
    let level_end = first_level
        .saturating_add(region.level_count)
        .min(mip_level_count);
    let first_layer = region.first_layer;
    let layer_end = first_layer
        .saturating_add(region.layer_count)
        .min(layer_count);
    if first_level >= level_end || first_layer >= layer_end {
        return None;
    }
    // Mode-27 slices are both strided and XOR-permuted inside the block, so a rect cannot be
    // refined across them. The whole written face is claimed instead.
    let volume_faces = descriptor.tile_mode == 27 && layer_count > 1;
    if volume_faces && region.rect.is_some() {
        OWNED_RANGE_SET_COARSENED.fetch_add(1, Ordering::Relaxed);
    }
    let mut runs = RangeSet::default();
    for layer in first_layer..layer_end {
        let layer_offset = face_bytes.checked_mul(u64::from(layer))?;
        if volume_faces {
            runs.insert_run(layer_offset, layer_offset.checked_add(face_bytes)?);
            continue;
        }
        for level in first_level..level_end {
            let extent = levels.get(usize::try_from(level).ok()?)?;
            extend_level_runs(
                *extent,
                region.rect,
                descriptor,
                bytes_per_element,
                layer_offset,
                &mut runs,
            )?;
        }
    }
    Some(runs)
}

fn level_extents(
    descriptor: GuestTextureDescriptor,
    mip_level_count: u32,
) -> Option<Vec<LevelExtent>> {
    if mip_level_count == 1 {
        let (width, height, pitch, _) = texture_element_layout(descriptor).ok()?;
        return Some(vec![LevelExtent {
            offset: 0,
            len: guest_texture_byte_count(descriptor)?,
            width,
            height,
            row_texels: pitch,
            in_tail: false,
        }]);
    }
    let layout = match descriptor.tile_mode {
        9 | 27
            if compressed_texture_block_extent(descriptor.pixel_format).is_none()
                && super::super::metal_uncompressed_bytes_per_pixel(descriptor.pixel_format)
                    == Some(descriptor.bytes_per_pixel) =>
        {
            mode64k_2d_mip_chain_layout(descriptor, mip_level_count).ok()?
        }
        5 => mode5_mip_chain_layout(descriptor, mip_level_count).ok()?,
        _ => return None,
    };
    // Chain levels are stacked with level 0 last, so a level's storage ends where the previous
    // level begins. Deriving the size that way avoids re-deriving the layout's own padding rules.
    let tail_bytes = u64::try_from(tiled_block_bytes(descriptor.tile_mode)?).ok()?;
    let mut extents = Vec::with_capacity(layout.levels.len());
    for (index, level) in layout.levels.iter().enumerate() {
        let (offset, len) = if level.in_tail {
            (0, tail_bytes)
        } else {
            let end = if index == 0 {
                layout.byte_count
            } else {
                layout.levels[index - 1].offset
            };
            (
                u64::try_from(level.offset).ok()?,
                u64::try_from(end.checked_sub(level.offset)?).ok()?,
            )
        };
        extents.push(LevelExtent {
            offset,
            len,
            width: level.width,
            height: level.height,
            row_texels: level.padded_width,
            in_tail: level.in_tail,
        });
    }
    Some(extents)
}

fn extend_level_runs(
    level: LevelExtent,
    rect: Option<MTLScissorRect>,
    descriptor: GuestTextureDescriptor,
    bytes_per_element: usize,
    layer_offset: u64,
    runs: &mut RangeSet,
) -> Option<()> {
    let level_start = layer_offset.checked_add(level.offset)?;
    let level_end = level_start.checked_add(level.len)?;
    let Some(rect) = rect else {
        runs.insert_run(level_start, level_end);
        return Some(());
    };
    if level.in_tail {
        // Every tail level shares one packed block, so any write into one claims all of them.
        // They belong to the same chain, so this is intra-allocation coarsening only.
        OWNED_RANGE_SET_COARSENED.fetch_add(1, Ordering::Relaxed);
        runs.insert_run(level_start, level_end);
        return Some(());
    }
    // `texture_element_layout` reports a compressed level in block units, so the rect converts to
    // the same units before it can index the level.
    let (element_width, element_height) =
        compressed_texture_block_extent(descriptor.pixel_format).unwrap_or((1, 1));
    let x0 = (usize::try_from(rect.x).ok()? / element_width).min(level.width);
    let x1 = usize::try_from(rect.x.checked_add(rect.width)?)
        .ok()?
        .div_ceil(element_width)
        .min(level.width);
    let y0 = (usize::try_from(rect.y).ok()? / element_height).min(level.height);
    let y1 = usize::try_from(rect.y.checked_add(rect.height)?)
        .ok()?
        .div_ceil(element_height)
        .min(level.height);
    if x0 >= x1 || y0 >= y1 {
        return Some(());
    }
    match tiled_block_bytes(descriptor.tile_mode) {
        Some(block_bytes) => {
            let (block_width, block_height) =
                tiled_block_dimensions(block_bytes, bytes_per_element)?;
            // The block-grid stride is the padded row width, matching `copy_tiled_to_linear_at_z`.
            let blocks_per_row = level.row_texels.div_ceil(block_width);
            let (first_column, column_end) = (x0 / block_width, x1.div_ceil(block_width));
            for block_row in (y0 / block_height)..y1.div_ceil(block_height) {
                let row_base = block_row.checked_mul(blocks_per_row)?;
                let start = u64::try_from(
                    row_base
                        .checked_add(first_column)?
                        .checked_mul(block_bytes)?,
                )
                .ok()?;
                let end =
                    u64::try_from(row_base.checked_add(column_end)?.checked_mul(block_bytes)?)
                        .ok()?;
                runs.insert_run(
                    level_start.checked_add(start)?.min(level_end),
                    level_start.checked_add(end)?.min(level_end),
                );
            }
        }
        None => {
            let element_bytes = u64::try_from(bytes_per_element).ok()?;
            for row in y0..y1 {
                let row_base = row.checked_mul(level.row_texels)?;
                let start = u64::try_from(row_base.checked_add(x0)?)
                    .ok()?
                    .checked_mul(element_bytes)?;
                let end = u64::try_from(row_base.checked_add(x1)?)
                    .ok()?
                    .checked_mul(element_bytes)?;
                runs.insert_run(
                    level_start.checked_add(start)?.min(level_end),
                    level_start.checked_add(end)?.min(level_end),
                );
            }
        }
    }
    Some(())
}

#[cfg(test)]
mod tests {
    use super::super::copy_linear_to_tiled_at_z;
    use super::{
        GuestTextureDescriptor, OWNED_FALLBACK_FULL_RANGE, RangeSet, ResourceSpan, WrittenRegion,
        guest_texture_byte_count, mode64k_2d_mip_chain_layout, written_byte_ranges,
    };
    use metal::{MTLPixelFormat, MTLScissorRect};
    use std::sync::atomic::Ordering;

    /// A descriptor whose bytes live in a plain host allocation.
    ///
    /// The range algebra never dereferences the address, and an unmapped one resolves no shared
    /// backing, so the span falls back to virtual-address space — which is exactly the case the
    /// algebra must keep self-consistent.
    fn descriptor(width: u32, height: u32, pitch: u32, tile_mode: u32) -> GuestTextureDescriptor {
        GuestTextureDescriptor {
            address: 0x1_0000_0000,
            width,
            height,
            pitch,
            tile_mode,
            pixel_format: MTLPixelFormat::RGBA8Unorm,
            bytes_per_pixel: 4,
        }
    }

    fn span_of(descriptor: GuestTextureDescriptor) -> ResourceSpan {
        ResourceSpan::new(
            descriptor.address,
            guest_texture_byte_count(descriptor).expect("descriptor byte count"),
        )
    }

    fn ranges(
        descriptor: GuestTextureDescriptor,
        span: ResourceSpan,
        region: WrittenRegion,
    ) -> RangeSet {
        written_byte_ranges(span, descriptor, 9, 1, 1, region)
    }

    #[test]
    fn a_whole_mode27_surface_write_is_one_run_of_the_declared_byte_count() {
        let descriptor = descriptor(256, 128, 256, 27);
        let span = span_of(descriptor);
        let written = ranges(descriptor, span, WrittenRegion::whole_resource());
        assert_eq!(
            written.runs(),
            [(span.range_start(), span.range_end())],
            "a full-surface write must not fragment its own allocation"
        );
        assert_eq!(written.byte_count(), span.len());
        assert_eq!(
            span.len(),
            guest_texture_byte_count(descriptor).expect("byte count")
        );
    }

    #[test]
    fn a_scissored_mode27_write_covers_every_byte_the_tiler_touches() {
        // A padded surface: the block-grid stride follows the pitch, not the visible width.
        let descriptor = descriptor(200, 96, 256, 27);
        let span = span_of(descriptor);
        let rect = MTLScissorRect {
            x: 3,
            y: 40,
            width: 61,
            height: 20,
        };
        let written = ranges(
            descriptor,
            span,
            WrittenRegion {
                rect: Some(rect),
                ..WrittenRegion::whole_resource()
            },
        );
        assert!(!written.is_empty());
        assert!(
            written.byte_count() < span.len(),
            "a small scissor must claim less than the whole surface"
        );

        // Differential check against the real tiler: retile a linear image whose rect texels are
        // the only non-zero ones, and require every byte it moved to be inside the claimed set.
        let (width, height) = (
            usize::try_from(descriptor.width).expect("width"),
            usize::try_from(descriptor.height).expect("height"),
        );
        let mut linear = vec![0_u8; width * height * 4];
        for row in usize::try_from(rect.y).expect("y")
            ..usize::try_from(rect.y + rect.height).expect("y end")
        {
            for column in usize::try_from(rect.x).expect("x")
                ..usize::try_from(rect.x + rect.width).expect("x end")
            {
                linear[(row * width + column) * 4..(row * width + column) * 4 + 4]
                    .copy_from_slice(&[0xa5; 4]);
            }
        }
        let mut tiled =
            vec![
                0_u8;
                usize::try_from(guest_texture_byte_count(descriptor).expect("byte count"))
                    .expect("byte count fits usize")
            ];
        copy_linear_to_tiled_at_z(&linear, descriptor, &mut tiled, 0).expect("retile the rect");
        for (offset, byte) in tiled.iter().enumerate() {
            if *byte == 0 {
                continue;
            }
            let address = span.range_start() + u64::try_from(offset).expect("offset fits u64");
            assert!(
                written.contains_run(address, address + 1),
                "the tiler wrote {offset:#x}, which the claimed ranges {:?} do not cover",
                written.runs()
            );
        }
    }

    #[test]
    fn a_scissored_linear_write_covers_only_the_rows_it_touches() {
        let descriptor = descriptor(64, 64, 64, 0);
        let span = span_of(descriptor);
        let written = ranges(
            descriptor,
            span,
            WrittenRegion {
                rect: Some(MTLScissorRect {
                    x: 0,
                    y: 8,
                    width: 64,
                    height: 4,
                }),
                ..WrittenRegion::whole_resource()
            },
        );
        let row_bytes = u64::from(descriptor.pitch) * u64::from(descriptor.bytes_per_pixel);
        assert_eq!(
            written.runs(),
            [(
                span.range_start() + 8 * row_bytes,
                span.range_start() + 12 * row_bytes
            )]
        );
    }

    #[test]
    fn a_chain_level_lands_inside_its_own_level_and_a_tail_level_claims_the_tail_block() {
        for tile_mode in [27, 9] {
            let descriptor = descriptor(1024, 1024, 1024, tile_mode);
            let levels = 11;
            let layout =
                mode64k_2d_mip_chain_layout(descriptor, levels).expect("64 KiB 2D chain layout");
            let span = ResourceSpan::new(
                descriptor.address,
                u64::try_from(layout.byte_count).expect("chain byte count"),
            );
            for level in 0..levels {
                let written = written_byte_ranges(
                    span,
                    descriptor,
                    9,
                    1,
                    levels,
                    WrittenRegion {
                        first_level: level,
                        level_count: 1,
                        ..WrittenRegion::whole_resource()
                    },
                );
                let layer = layout.levels[usize::try_from(level).expect("level")];
                let (start, end) = written.bounding_run().expect("a level claims bytes");
                if layer.in_tail {
                    assert_eq!(
                        (start - span.range_start(), end - span.range_start()),
                        (0, 65_536),
                        "a tail level claims the packed tail block"
                    );
                    continue;
                }
                let level_start = span.range_start() + u64::try_from(layer.offset).expect("offset");
                assert!(
                    start >= level_start && end <= span.range_end(),
                    "tile {tile_mode} level {level} claimed {start:#x}..{end:#x} outside its own storage"
                );
                assert!(
                    written.byte_count() < span.len() || level == 0,
                    "only level 0 may be the bulk of the chain"
                );
            }
        }
    }

    #[test]
    fn every_failure_path_claims_the_whole_span_and_counts_itself() {
        let span = span_of(descriptor(64, 64, 64, 27));
        let whole = [(span.range_start(), span.range_end())];
        // An unsupported tile mode has no chain layout at all.
        let unsupported = descriptor(64, 64, 64, 8);
        // A zero layer count, a level outside the resident, and a layered mip chain are all
        // shapes the layout arithmetic cannot answer for.
        let cases: [(GuestTextureDescriptor, u32, u32, WrittenRegion); 4] = [
            (unsupported, 1, 4, WrittenRegion::whole_resource()),
            (unsupported, 0, 1, WrittenRegion::whole_resource()),
            (unsupported, 6, 4, WrittenRegion::whole_resource()),
            (
                unsupported,
                1,
                1,
                WrittenRegion {
                    first_level: 7,
                    level_count: 1,
                    ..WrittenRegion::whole_resource()
                },
            ),
        ];
        for (descriptor, layer_count, mip_level_count, region) in cases {
            let before = OWNED_FALLBACK_FULL_RANGE.load(Ordering::Relaxed);
            let written =
                written_byte_ranges(span, descriptor, 9, layer_count, mip_level_count, region);
            assert_eq!(written.runs(), whole);
            assert_eq!(
                OWNED_FALLBACK_FULL_RANGE.load(Ordering::Relaxed),
                before + 1,
                "a fallback must be counted, not silent"
            );
        }
    }

    #[test]
    fn a_scissor_outside_the_level_falls_back_instead_of_claiming_nothing() {
        let descriptor = descriptor(64, 64, 64, 27);
        let span = span_of(descriptor);
        let written = ranges(
            descriptor,
            span,
            WrittenRegion {
                rect: Some(MTLScissorRect {
                    x: 4_096,
                    y: 4_096,
                    width: 16,
                    height: 16,
                }),
                ..WrittenRegion::whole_resource()
            },
        );
        assert_eq!(written.runs(), [(span.range_start(), span.range_end())]);
    }
}
