// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

//! Opt-in correlation for texture aliases which share guest physical backing.
//!
//! Residency churn, cache invalidation, and shader binding normally log at different layers. A
//! virtual address is insufficient to join those records because Prospero may expose the same
//! direct-memory pages through unrelated VAs. This module retains only diagnostic physical spans
//! observed at a churn boundary and uses them to annotate later ACQUIRE and SRD activity.
//! Enable it with `REMU_TRACE_TEXTURE_ALIAS_CORRELATION=1`; tracked state and emitted records are
//! bounded so a long-running guest cannot turn the diagnostic into an independent memory leak.

use super::{
    DecodedImageDescriptor, GuestTextureDescriptor, ResidentTextureActivityStamp,
    WriteWatchRefresh, decode_image_descriptor, guest_texture_byte_count, is_storage_image_opcode,
    is_writable_image_opcode,
};
use remu_gpu::GpuMemoryAcquire;
use remu_memory::{GuestSharedMappingIdentity, guest_shared_backing_identity};
use remu_shader::{ShaderEvaluation, image_is_comparison};
use std::collections::{BTreeMap, HashSet};
use std::fmt::Write;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Mutex, OnceLock};

const MAX_TRACKED_SPANS: usize = 1_024;
const MAX_CHURN_LOGS: usize = 512;
const MAX_ACQUIRE_LOGS: usize = 4_096;
const MAX_BINDING_LOGS: usize = 4_096;
static CHURN_LOG_COUNT: AtomicUsize = AtomicUsize::new(0);
static ACQUIRE_LOG_COUNT: AtomicUsize = AtomicUsize::new(0);
static NEXT_CHURN: AtomicU64 = AtomicU64::new(1);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct PhysicalSpan {
    backing_id: u64,
    start: u64,
    end: u64,
}

impl PhysicalSpan {
    fn new(identity: GuestSharedMappingIdentity, byte_count: u64) -> Option<Self> {
        Some(Self {
            backing_id: identity.backing_id,
            start: identity.backing_offset,
            end: identity.backing_offset.checked_add(byte_count)?,
        })
    }

    fn overlaps(self, other: Self) -> bool {
        self.backing_id == other.backing_id && self.start < other.end && other.start < self.end
    }
}

#[derive(Clone, Copy, Debug)]
struct DescriptorSpan {
    descriptor: GuestTextureDescriptor,
    byte_count: Option<u64>,
    physical: Option<PhysicalSpan>,
}

impl DescriptorSpan {
    fn new(descriptor: GuestTextureDescriptor) -> Self {
        let byte_count = guest_texture_byte_count(descriptor);
        let physical = byte_count.and_then(|byte_count| {
            guest_shared_backing_identity(descriptor.address, byte_count)
                .and_then(|identity| PhysicalSpan::new(identity, byte_count))
        });
        Self {
            descriptor,
            byte_count,
            physical,
        }
    }

    fn guest_overlaps(self, other: Self) -> bool {
        let (Some(self_bytes), Some(other_bytes)) = (self.byte_count, other.byte_count) else {
            return false;
        };
        let Some(self_end) = self.descriptor.address.checked_add(self_bytes) else {
            return false;
        };
        let Some(other_end) = other.descriptor.address.checked_add(other_bytes) else {
            return false;
        };
        self.descriptor.address < other_end && other.descriptor.address < self_end
    }
}

#[derive(Clone, Copy, Debug)]
struct TrackedSpan {
    churn: u64,
    span: DescriptorSpan,
}

#[derive(Clone, Copy, Debug, Default)]
struct MatchBasis {
    guest: bool,
    backing: bool,
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct BindingTraceKey {
    churn: u64,
    stage: &'static str,
    pc: u32,
    opcode: remu_shader::Opcode,
    resource_descriptor: Vec<u32>,
}

#[derive(Debug)]
pub(crate) struct TextureAliasTraceMatch {
    span: DescriptorSpan,
    churns: Vec<(u64, MatchBasis)>,
}

fn enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var_os("REMU_TRACE_TEXTURE_ALIAS_CORRELATION").is_some_and(|value| value != "0")
    })
}

fn tracked_spans() -> &'static Mutex<Vec<TrackedSpan>> {
    static TRACKED: OnceLock<Mutex<Vec<TrackedSpan>>> = OnceLock::new();
    TRACKED.get_or_init(|| Mutex::new(Vec::new()))
}

fn logged_bindings() -> &'static Mutex<HashSet<BindingTraceKey>> {
    static LOGGED: OnceLock<Mutex<HashSet<BindingTraceKey>>> = OnceLock::new();
    LOGGED.get_or_init(|| Mutex::new(HashSet::new()))
}

fn push_tracked_span(tracked: &mut Vec<TrackedSpan>, span: TrackedSpan) {
    if let Some(existing) = tracked.iter_mut().find(|existing| {
        existing.span.descriptor == span.span.descriptor
            && existing.span.physical == span.span.physical
    }) {
        existing.churn = span.churn;
        return;
    }
    if tracked.len() == MAX_TRACKED_SPANS {
        tracked.remove(0);
    }
    tracked.push(span);
}

fn format_descriptor_span(span: DescriptorSpan) -> String {
    let descriptor = span.descriptor;
    let mut output = format!(
        "{:#x}/{}x{}/p{}/{:?}/tile{}/bytes",
        descriptor.address,
        descriptor.width,
        descriptor.height,
        descriptor.pitch,
        descriptor.pixel_format,
        descriptor.tile_mode,
    );
    if let Some(byte_count) = span.byte_count {
        let _ = write!(output, "{byte_count:#x}");
    } else {
        output.push('?');
    }
    if let Some(physical) = span.physical {
        let _ = write!(
            output,
            "/backing{}:{:#x}..{:#x}",
            physical.backing_id, physical.start, physical.end,
        );
    } else {
        output.push_str("/backing-unresolved");
    }
    output
}

fn matching_churns(span: DescriptorSpan) -> Vec<(u64, MatchBasis)> {
    let mut matches = BTreeMap::<u64, MatchBasis>::new();
    for tracked in tracked_spans()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .iter()
    {
        let guest = tracked.span.guest_overlaps(span);
        let backing = tracked
            .span
            .physical
            .zip(span.physical)
            .is_some_and(|(tracked, current)| tracked.overlaps(current));
        if guest || backing {
            let basis = matches.entry(tracked.churn).or_default();
            basis.guest |= guest;
            basis.backing |= backing;
        }
    }
    matches.into_iter().collect()
}

fn format_matches(trace: &TextureAliasTraceMatch) -> String {
    trace
        .churns
        .iter()
        .map(|(churn, basis)| {
            let basis = match (basis.guest, basis.backing) {
                (true, true) => "guest+backing",
                (true, false) => "guest",
                (false, true) => "backing",
                (false, false) => "none",
            };
            format!("{churn}:{basis}")
        })
        .collect::<Vec<_>>()
        .join(",")
}

pub(crate) fn texture_alias_trace_match(
    descriptor: GuestTextureDescriptor,
) -> Option<TextureAliasTraceMatch> {
    if !enabled() {
        return None;
    }
    let span = DescriptorSpan::new(descriptor);
    let churns = matching_churns(span);
    (!churns.is_empty()).then_some(TextureAliasTraceMatch { span, churns })
}

pub(crate) fn trace_residency_churn(
    descriptor: GuestTextureDescriptor,
    upload_guest_contents: bool,
    aliases: impl Iterator<Item = (GuestTextureDescriptor, bool, u64)>,
) {
    static LEGACY_ENABLED: OnceLock<bool> = OnceLock::new();
    let legacy_enabled =
        *LEGACY_ENABLED.get_or_init(|| std::env::var_os("REMU_TRACE_TEXTURE_RESIDENCY").is_some());
    if !legacy_enabled && !enabled() {
        return;
    }
    if CHURN_LOG_COUNT.fetch_add(1, Ordering::Relaxed) >= MAX_CHURN_LOGS {
        return;
    }
    let churn = NEXT_CHURN.fetch_add(1, Ordering::Relaxed);
    let target = DescriptorSpan::new(descriptor);
    let aliases = aliases
        .map(|(alias, gpu_written, version)| (DescriptorSpan::new(alias), gpu_written, version))
        .collect::<Vec<_>>();

    if enabled() {
        let mut tracked = tracked_spans()
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        for span in std::iter::once(target).chain(aliases.iter().map(|(span, _, _)| *span)) {
            push_tracked_span(&mut tracked, TrackedSpan { churn, span });
        }
    }

    let aliases = aliases
        .into_iter()
        .map(|(span, gpu_written, version)| {
            format!(
                "{}/gpu{gpu_written}/v{version}",
                format_descriptor_span(span)
            )
        })
        .collect::<Vec<_>>();
    eprintln!(
        "[REMu][METAL][RESIDENCY-CHURN] correlation={churn} target={} upload={} aliases=[{}]",
        format_descriptor_span(target),
        upload_guest_contents,
        aliases.join(", "),
    );
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn trace_alias_correlated_acquire(
    trace: &TextureAliasTraceMatch,
    acquire: GpuMemoryAcquire,
    before_gpu_owned: bool,
    after_owner: &str,
    refresh: WriteWatchRefresh,
    dirty: bool,
    precise: bool,
    dirty_pages: &[(u64, u64)],
    watch_generations_before: &[u64],
    watch_generations_after: &[u64],
    gpu_write_generations_before: &[u64],
    gpu_write_generations_after: &[u64],
    content_changed: Option<bool>,
    outcome: &str,
) {
    if ACQUIRE_LOG_COUNT.fetch_add(1, Ordering::Relaxed) >= MAX_ACQUIRE_LOGS {
        return;
    }
    let dirty_page_count = dirty_pages.len();
    let dirty_bytes = dirty_pages
        .iter()
        .fold(0_u64, |total, (_, bytes)| total.saturating_add(*bytes));
    let dirty_pages = dirty_pages
        .iter()
        .take(16)
        .map(|(address, bytes)| format!("{address:#x}+{bytes:#x}"))
        .collect::<Vec<_>>();
    eprintln!(
        "[REMu][METAL][ALIAS-ACQUIRE] correlations=[{}] resource={} owner_before={} owner_after={after_owner} refresh={refresh:?} dirty={dirty} precise={precise} dirty_pages={:?}/count{}/bytes{dirty_bytes:#x} watch_gen_before={watch_generations_before:?} watch_gen_after={watch_generations_after:?} gpu_boundary_gen_before={gpu_write_generations_before:?} gpu_boundary_gen_after={gpu_write_generations_after:?} content_changed={content_changed:?} outcome={outcome} acquire=engine{:#x}/cbdb{:#x}/base{:#x}/size{:#x}/all{}/gcr{:#x}",
        format_matches(trace),
        format_descriptor_span(trace.span),
        if before_gpu_owned { "gpu" } else { "cpu" },
        dirty_pages,
        dirty_page_count,
        acquire.engine,
        acquire.cb_db_control,
        acquire.base_address,
        acquire.size_bytes,
        acquire.covers_all_guest_memory(),
        acquire.gcr_control,
    );
}

pub(crate) fn trace_alias_correlated_image_bindings(
    stage: &'static str,
    evaluation: &ShaderEvaluation,
    activity: ResidentTextureActivityStamp,
) {
    if !enabled() {
        return;
    }
    for binding in &evaluation.image_bindings {
        let Ok(descriptor) = decode_image_descriptor(&binding.resource_descriptor) else {
            continue;
        };
        let Some(trace) = texture_alias_trace_match(descriptor.guest) else {
            continue;
        };
        let key = BindingTraceKey {
            churn: trace
                .churns
                .last()
                .map(|(churn, _)| *churn)
                .unwrap_or_default(),
            stage,
            pc: binding.pc,
            opcode: binding.opcode,
            resource_descriptor: binding.resource_descriptor.to_vec(),
        };
        let mut logged = logged_bindings()
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if logged.len() >= MAX_BINDING_LOGS || !logged.insert(key) {
            continue;
        }
        drop(logged);
        trace_correlated_binding(stage, binding, descriptor, &trace, activity);
    }
}

fn trace_correlated_binding(
    stage: &str,
    binding: &remu_shader::ImageBinding,
    descriptor: DecodedImageDescriptor,
    trace: &TextureAliasTraceMatch,
    activity: ResidentTextureActivityStamp,
) {
    eprintln!(
        "[REMu][METAL][ALIAS-SRD] correlations=[{}] stage={stage} work={} flip={} pc={:#x} op={} storage={} writable={} comparison={} select={:#x} base_mip={} mips={} type={} layers={} resource={} words={:08x?}",
        format_matches(trace),
        activity.work,
        activity.flip,
        binding.pc,
        binding.opcode,
        is_storage_image_opcode(binding.opcode),
        is_writable_image_opcode(binding.opcode),
        image_is_comparison(binding.opcode),
        descriptor.destination_select,
        descriptor.base_mip_level,
        descriptor.mip_level_count,
        descriptor.texture_type,
        descriptor.layer_count,
        format_descriptor_span(trace.span),
        binding.resource_descriptor,
    );
}

#[cfg(test)]
mod tests {
    use super::{
        DescriptorSpan, MatchBasis, PhysicalSpan, TextureAliasTraceMatch, format_descriptor_span,
        format_matches,
    };
    use crate::presenter::GuestTextureDescriptor;
    use metal::MTLPixelFormat;

    #[test]
    fn physical_spans_only_overlap_within_one_backing() {
        let first = PhysicalSpan {
            backing_id: 7,
            start: 0x1_000,
            end: 0x1_600,
        };
        assert!(first.overlaps(PhysicalSpan {
            backing_id: 7,
            start: 0x1_500,
            end: 0x1_700,
        }));
        assert!(!first.overlaps(PhysicalSpan {
            backing_id: 8,
            start: 0x1_500,
            end: 0x1_700,
        }));
        assert!(!first.overlaps(PhysicalSpan {
            backing_id: 7,
            start: 0x1_600,
            end: 0x1_700,
        }));
    }

    #[test]
    fn descriptor_span_format_includes_physical_identity_and_extent() {
        let descriptor = GuestTextureDescriptor {
            address: 0x80_4584_0000,
            width: 378,
            height: 150,
            pitch: 378,
            tile_mode: 27,
            pixel_format: MTLPixelFormat::BGRA8Unorm_sRGB,
            bytes_per_pixel: 4,
        };
        let text = format_descriptor_span(DescriptorSpan {
            descriptor,
            byte_count: Some(0x6_0000),
            physical: Some(PhysicalSpan {
                backing_id: 11,
                start: 0x20_0000,
                end: 0x26_0000,
            }),
        });
        assert_eq!(
            text,
            "0x8045840000/378x150/p378/BGRA8Unorm_sRGB/tile27/bytes0x60000/backing11:0x200000..0x260000"
        );
    }

    #[test]
    fn correlation_format_distinguishes_guest_and_backing_overlap() {
        let descriptor = GuestTextureDescriptor {
            address: 0x80_4584_0000,
            width: 378,
            height: 150,
            pitch: 378,
            tile_mode: 27,
            pixel_format: MTLPixelFormat::BGRA8Unorm_sRGB,
            bytes_per_pixel: 4,
        };
        let trace = TextureAliasTraceMatch {
            span: DescriptorSpan {
                descriptor,
                byte_count: Some(0x6_0000),
                physical: None,
            },
            churns: vec![
                (
                    17,
                    MatchBasis {
                        guest: true,
                        backing: false,
                    },
                ),
                (
                    23,
                    MatchBasis {
                        guest: true,
                        backing: true,
                    },
                ),
                (
                    41,
                    MatchBasis {
                        guest: false,
                        backing: true,
                    },
                ),
            ],
        };

        assert_eq!(
            format_matches(&trace),
            "17:guest,23:guest+backing,41:backing"
        );
    }
}
