// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

//! Signal-safe evidence for Metal render-resource binding faults.

use metal::foreign_types::ForeignTypeRef;
use std::ffi::c_void;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};

const MAX_RESOURCE_CALLS: usize = 256;
const LABEL_BYTES: usize = 24;
const PHASE_BEFORE: u64 = 1;
const PHASE_RETURNED: u64 = 2;

#[derive(Clone, Copy)]
pub(super) enum ResourceKind {
    Encoder = 1,
    Pipeline = 2,
    DepthState = 3,
    ColorAttachment = 10,
    ColorResolve = 11,
    DepthAttachment = 12,
    DepthResolve = 13,
    VertexBuffer = 20,
    MeshBuffer = 21,
    FragmentBuffer = 22,
    VertexInline = 23,
    MeshInline = 24,
    FragmentInline = 25,
    VertexTexture = 30,
    MeshTexture = 31,
    FragmentTexture = 32,
    VertexSampler = 40,
    MeshSampler = 41,
    FragmentSampler = 42,
    UseResource = 50,
    Draw = 60,
}

#[derive(Clone, Copy)]
pub(super) enum ResourceOwner {
    Batch = 1,
    PipelineCache = 2,
    DepthCache = 3,
    ResidentTarget = 4,
    ResidentSample = 5,
    DrawGlobal = 6,
    InlineCopy = 7,
    PresenterDiagnostic = 8,
    SamplerCache = 9,
    DrawArgument = 10,
    MeshletCache = 11,
    Encoder = 12,
}

#[derive(Clone, Copy)]
pub(super) struct ResourceCallToken(usize);

impl ResourceCallToken {
    const DISABLED: Self = Self(usize::MAX);
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct ResourceCallRecord {
    phase: u64,
    kind: u64,
    index: u64,
    object: u64,
    allocated_length: u64,
    offset: u64,
    range: u64,
    subresource: u64,
    owner: u64,
    label_length: u64,
    label_words: [u64; LABEL_BYTES / 8],
}

struct AtomicResourceCallRecord {
    phase: AtomicU64,
    kind: AtomicU64,
    index: AtomicU64,
    object: AtomicU64,
    allocated_length: AtomicU64,
    offset: AtomicU64,
    range: AtomicU64,
    subresource: AtomicU64,
    owner: AtomicU64,
    label_length: AtomicU64,
    label_words: [AtomicU64; LABEL_BYTES / 8],
}

impl AtomicResourceCallRecord {
    const fn new() -> Self {
        Self {
            phase: AtomicU64::new(0),
            kind: AtomicU64::new(0),
            index: AtomicU64::new(0),
            object: AtomicU64::new(0),
            allocated_length: AtomicU64::new(0),
            offset: AtomicU64::new(0),
            range: AtomicU64::new(0),
            subresource: AtomicU64::new(0),
            owner: AtomicU64::new(0),
            label_length: AtomicU64::new(0),
            label_words: [const { AtomicU64::new(0) }; LABEL_BYTES / 8],
        }
    }

    fn store(&self, record: ResourceCallRecord) {
        self.phase.store(0, Ordering::Relaxed);
        self.kind.store(record.kind, Ordering::Relaxed);
        self.index.store(record.index, Ordering::Relaxed);
        self.object.store(record.object, Ordering::Relaxed);
        self.allocated_length
            .store(record.allocated_length, Ordering::Relaxed);
        self.offset.store(record.offset, Ordering::Relaxed);
        self.range.store(record.range, Ordering::Relaxed);
        self.subresource
            .store(record.subresource, Ordering::Relaxed);
        self.owner.store(record.owner, Ordering::Relaxed);
        self.label_length
            .store(record.label_length, Ordering::Relaxed);
        for (destination, source) in self.label_words.iter().zip(record.label_words) {
            destination.store(source, Ordering::Relaxed);
        }
        self.phase.store(PHASE_BEFORE, Ordering::Release);
    }

    fn mark_returned(&self) {
        self.phase.store(PHASE_RETURNED, Ordering::Release);
    }

    fn load(&self) -> Option<ResourceCallRecord> {
        let phase = self.phase.load(Ordering::Acquire);
        (phase != 0).then(|| ResourceCallRecord {
            phase,
            kind: self.kind.load(Ordering::Relaxed),
            index: self.index.load(Ordering::Relaxed),
            object: self.object.load(Ordering::Relaxed),
            allocated_length: self.allocated_length.load(Ordering::Relaxed),
            offset: self.offset.load(Ordering::Relaxed),
            range: self.range.load(Ordering::Relaxed),
            subresource: self.subresource.load(Ordering::Relaxed),
            owner: self.owner.load(Ordering::Relaxed),
            label_length: self.label_length.load(Ordering::Relaxed),
            label_words: std::array::from_fn(|index| {
                self.label_words[index].load(Ordering::Relaxed)
            }),
        })
    }
}

struct ResourceCallTrace {
    draw: AtomicU64,
    export_shader: AtomicU64,
    pixel_shader: AtomicU64,
    batch: AtomicU64,
    new_pass: AtomicBool,
    overflowed: AtomicBool,
    count: AtomicUsize,
    calls: [AtomicResourceCallRecord; MAX_RESOURCE_CALLS],
}

impl ResourceCallTrace {
    const fn new() -> Self {
        Self {
            draw: AtomicU64::new(0),
            export_shader: AtomicU64::new(0),
            pixel_shader: AtomicU64::new(0),
            batch: AtomicU64::new(0),
            new_pass: AtomicBool::new(false),
            overflowed: AtomicBool::new(false),
            count: AtomicUsize::new(0),
            calls: [const { AtomicResourceCallRecord::new() }; MAX_RESOURCE_CALLS],
        }
    }

    fn begin(&self, draw: u64, export_shader: u64, pixel_shader: u64, batch: u64, new_pass: bool) {
        self.count.store(0, Ordering::Release);
        self.overflowed.store(false, Ordering::Relaxed);
        self.draw.store(draw, Ordering::Relaxed);
        self.export_shader.store(export_shader, Ordering::Relaxed);
        self.pixel_shader.store(pixel_shader, Ordering::Relaxed);
        self.batch.store(batch, Ordering::Relaxed);
        self.new_pass.store(new_pass, Ordering::Release);
    }

    fn push(&self, record: ResourceCallRecord) -> ResourceCallToken {
        let index = self.count.load(Ordering::Relaxed);
        if index >= self.calls.len() {
            self.overflowed.store(true, Ordering::Relaxed);
            return ResourceCallToken::DISABLED;
        }
        self.calls[index].store(record);
        self.count.store(index + 1, Ordering::Release);
        ResourceCallToken(index)
    }
}

static ENABLED: AtomicBool = AtomicBool::new(false);
static TRACE: ResourceCallTrace = ResourceCallTrace::new();

pub(super) fn install_fragment_scalar_binding_fault_diagnostic() {
    let requested =
        std::env::var_os("REMU_DIAGNOSTIC_METAL_BINDING_FAULT").is_some_and(|value| value == "1");
    if !requested {
        return;
    }
    if remu_memory::register_host_fault_diagnostic(write_resource_call_trace) {
        ENABLED.store(true, Ordering::Release);
        eprintln!("[REMu][METAL-BINDING-FAULT] enabled marker=render-resource-calls-v2");
    } else {
        eprintln!(
            "[REMu][METAL-BINDING-FAULT] unavailable: another host-fault callback is installed"
        );
    }
}

pub(super) fn begin_resource_call_trace(
    draw: u64,
    export_shader: u64,
    pixel_shader: u64,
    command_buffer: &metal::CommandBufferRef,
    new_pass: bool,
) {
    if !ENABLED.load(Ordering::Acquire) {
        return;
    }
    TRACE.begin(
        draw,
        export_shader,
        pixel_shader,
        native_object(command_buffer.as_ptr()),
        new_pass,
    );
}

#[allow(clippy::too_many_arguments)]
fn record_before(
    kind: ResourceKind,
    index: u64,
    object: u64,
    allocated_length: u64,
    offset: u64,
    range: u64,
    subresource: u64,
    owner: ResourceOwner,
    label: &str,
) -> ResourceCallToken {
    if !ENABLED.load(Ordering::Acquire) {
        return ResourceCallToken::DISABLED;
    }
    let (label_words, label_length) = encode_label(label);
    TRACE.push(ResourceCallRecord {
        phase: PHASE_BEFORE,
        kind: kind as u64,
        index,
        object,
        allocated_length,
        offset,
        range,
        subresource,
        owner: owner as u64,
        label_length,
        label_words,
    })
}

pub(super) fn record_buffer_before(
    kind: ResourceKind,
    index: u64,
    buffer: &metal::BufferRef,
    offset: u64,
    range: u64,
    owner: ResourceOwner,
    fallback_label: &str,
) -> ResourceCallToken {
    if !ENABLED.load(Ordering::Acquire) {
        return ResourceCallToken::DISABLED;
    }
    let label = nonempty_label(buffer.label(), fallback_label);
    record_before(
        kind,
        index,
        native_object(buffer.as_ptr()),
        buffer.length(),
        offset,
        range,
        0,
        owner,
        label,
    )
}

pub(super) fn record_texture_before(
    kind: ResourceKind,
    index: u64,
    texture: &metal::TextureRef,
    subresource: u64,
    owner: ResourceOwner,
    fallback_label: &str,
) -> ResourceCallToken {
    if !ENABLED.load(Ordering::Acquire) {
        return ResourceCallToken::DISABLED;
    }
    let label = nonempty_label(texture.label(), fallback_label);
    record_before(
        kind,
        index,
        native_object(texture.as_ptr()),
        texture.allocated_size(),
        0,
        texture.allocated_size(),
        subresource,
        owner,
        label,
    )
}

pub(super) fn record_sampler_before(
    kind: ResourceKind,
    index: u64,
    sampler: &metal::SamplerStateRef,
) -> ResourceCallToken {
    if !ENABLED.load(Ordering::Acquire) {
        return ResourceCallToken::DISABLED;
    }
    record_before(
        kind,
        index,
        native_object(sampler.as_ptr()),
        0,
        0,
        0,
        0,
        ResourceOwner::SamplerCache,
        nonempty_label(sampler.label(), "cached-sampler"),
    )
}

pub(super) fn record_pipeline_before(
    pipeline: &metal::RenderPipelineStateRef,
) -> ResourceCallToken {
    if !ENABLED.load(Ordering::Acquire) {
        return ResourceCallToken::DISABLED;
    }
    record_before(
        ResourceKind::Pipeline,
        0,
        native_object(pipeline.as_ptr()),
        0,
        0,
        0,
        0,
        ResourceOwner::PipelineCache,
        nonempty_label(pipeline.label(), "graphics-pipeline"),
    )
}

pub(super) fn record_depth_state_before(state: &metal::DepthStencilStateRef) -> ResourceCallToken {
    if !ENABLED.load(Ordering::Acquire) {
        return ResourceCallToken::DISABLED;
    }
    record_before(
        ResourceKind::DepthState,
        0,
        native_object(state.as_ptr()),
        0,
        0,
        0,
        0,
        ResourceOwner::DepthCache,
        nonempty_label(state.label(), "depth-state"),
    )
}

pub(super) fn record_inline_before(
    kind: ResourceKind,
    index: u64,
    byte_count: u64,
    label: &str,
) -> ResourceCallToken {
    record_before(
        kind,
        index,
        0,
        byte_count,
        0,
        byte_count,
        0,
        ResourceOwner::InlineCopy,
        label,
    )
}

pub(super) fn record_action_before(kind: ResourceKind, label: &str) -> ResourceCallToken {
    let owner = if matches!(kind, ResourceKind::Encoder) {
        ResourceOwner::Batch
    } else {
        ResourceOwner::Encoder
    };
    record_before(kind, 0, 0, 0, 0, 0, 0, owner, label)
}

pub(super) fn record_call_returned(token: ResourceCallToken) {
    if token.0 < MAX_RESOURCE_CALLS {
        TRACE.calls[token.0].mark_returned();
    }
}

fn nonempty_label<'a>(label: &'a str, fallback: &'a str) -> &'a str {
    if label.is_empty() { fallback } else { label }
}

fn native_object<T>(pointer: *mut T) -> u64 {
    u64::try_from(pointer as usize).expect("native Metal object pointer fits u64")
}

fn encode_label(label: &str) -> ([u64; LABEL_BYTES / 8], u64) {
    let bytes = label.as_bytes();
    let length = bytes.len().min(LABEL_BYTES);
    let mut words = [0_u64; LABEL_BYTES / 8];
    for (index, &byte) in bytes[..length].iter().enumerate() {
        words[index / 8] |= u64::from(byte) << ((index % 8) * 8);
    }
    (
        words,
        u64::try_from(length).expect("bounded label length fits u64"),
    )
}

unsafe extern "C" fn write_resource_call_trace() {
    let mut header = DiagnosticLine::new();
    header.push_bytes(b"[REMu][METAL-BINDING-FAULT] trace draw=");
    header.push_decimal(TRACE.draw.load(Ordering::Relaxed));
    header.push_bytes(b" es=");
    header.push_hexadecimal(TRACE.export_shader.load(Ordering::Relaxed));
    header.push_bytes(b" ps=");
    header.push_hexadecimal(TRACE.pixel_shader.load(Ordering::Relaxed));
    header.push_bytes(b" batch=");
    header.push_hexadecimal(TRACE.batch.load(Ordering::Relaxed));
    header.push_bytes(b" pass=");
    header.push_bytes(if TRACE.new_pass.load(Ordering::Acquire) {
        b"new"
    } else {
        b"reused"
    });
    header.push_bytes(b" calls=");
    let count = TRACE.count.load(Ordering::Acquire).min(MAX_RESOURCE_CALLS);
    header.push_decimal(count as u64);
    header.push_bytes(b" truncated=");
    header.push_bytes(if TRACE.overflowed.load(Ordering::Relaxed) {
        b"yes"
    } else {
        b"no"
    });
    header.push_bytes(b"\n");
    write_line(&header);

    for index in 0..count {
        let Some(record) = TRACE.calls[index].load() else {
            continue;
        };
        let mut line = DiagnosticLine::new();
        format_resource_call(index, record, &mut line);
        write_line(&line);
    }
}

fn write_line(line: &DiagnosticLine) {
    // SAFETY: `write` is async-signal-safe, and `line` remains live for this bounded call.
    unsafe {
        libc::write(
            libc::STDERR_FILENO,
            line.as_slice().as_ptr().cast::<c_void>(),
            line.as_slice().len(),
        );
    }
}

fn format_resource_call(index: usize, record: ResourceCallRecord, line: &mut DiagnosticLine) {
    line.push_bytes(b"[REMu][METAL-BINDING-FAULT] call=");
    line.push_decimal(index as u64);
    line.push_bytes(b" phase=");
    line.push_bytes(if record.phase == PHASE_RETURNED {
        b"returned"
    } else {
        b"before"
    });
    line.push_bytes(b" kind=");
    line.push_bytes(kind_name(record.kind));
    line.push_bytes(b" index=");
    line.push_decimal(record.index);
    line.push_bytes(b" object=");
    line.push_hexadecimal(record.object);
    line.push_bytes(b" allocated=");
    line.push_decimal(record.allocated_length);
    line.push_bytes(b" offset=");
    line.push_decimal(record.offset);
    line.push_bytes(b" range=");
    line.push_decimal(record.range);
    line.push_bytes(b" subresource=");
    line.push_hexadecimal(record.subresource);
    line.push_bytes(b" owner=");
    line.push_bytes(owner_name(record.owner));
    line.push_bytes(b" label=");
    line.push_label(record.label_words, record.label_length);
    line.push_bytes(b"\n");
}

fn kind_name(kind: u64) -> &'static [u8] {
    match kind {
        1 => b"encoder",
        2 => b"pipeline",
        3 => b"depth-state",
        10 => b"color-attachment",
        11 => b"color-resolve",
        12 => b"depth-attachment",
        13 => b"depth-resolve",
        20 => b"vertex-buffer",
        21 => b"mesh-buffer",
        22 => b"fragment-buffer",
        23 => b"vertex-inline",
        24 => b"mesh-inline",
        25 => b"fragment-inline",
        30 => b"vertex-texture",
        31 => b"mesh-texture",
        32 => b"fragment-texture",
        40 => b"vertex-sampler",
        41 => b"mesh-sampler",
        42 => b"fragment-sampler",
        50 => b"use-resource",
        60 => b"draw",
        _ => b"unknown",
    }
}

fn owner_name(owner: u64) -> &'static [u8] {
    match owner {
        1 => b"guest-batch",
        2 => b"pipeline-cache",
        3 => b"depth-cache",
        4 => b"resident-target",
        5 => b"resident-sample",
        6 => b"draw-global+encoder",
        7 => b"metal-inline-copy",
        8 => b"presenter-diagnostic",
        9 => b"sampler-cache",
        10 => b"draw-argument+encoder",
        11 => b"meshlet-cache",
        12 => b"render-encoder",
        _ => b"unknown",
    }
}

struct DiagnosticLine {
    bytes: [u8; 512],
    length: usize,
}

impl DiagnosticLine {
    const fn new() -> Self {
        Self {
            bytes: [0; 512],
            length: 0,
        }
    }

    fn push_bytes(&mut self, value: &[u8]) {
        let copied = value.len().min(self.bytes.len() - self.length);
        self.bytes[self.length..self.length + copied].copy_from_slice(&value[..copied]);
        self.length += copied;
    }

    fn push_decimal(&mut self, value: u64) {
        let mut digits = [0_u8; 20];
        let mut index = digits.len();
        let mut remaining = value;
        loop {
            index -= 1;
            digits[index] = b'0' + u8::try_from(remaining % 10).expect("decimal digit fits u8");
            remaining /= 10;
            if remaining == 0 {
                break;
            }
        }
        self.push_bytes(&digits[index..]);
    }

    fn push_hexadecimal(&mut self, value: u64) {
        self.push_bytes(b"0x");
        let mut digits = [0_u8; 16];
        for (index, digit) in digits.iter_mut().enumerate() {
            let nibble = (value >> (60 - index * 4)) & 0xf;
            *digit = b"0123456789abcdef"
                [usize::try_from(nibble).expect("hexadecimal nibble fits usize")];
        }
        let leading = digits
            .iter()
            .position(|&digit| digit != b'0')
            .unwrap_or(digits.len() - 1);
        self.push_bytes(&digits[leading..]);
    }

    fn push_label(&mut self, words: [u64; LABEL_BYTES / 8], length: u64) {
        let length = usize::try_from(length).unwrap_or(0).min(LABEL_BYTES);
        for index in 0..length {
            let byte = u8::try_from((words[index / 8] >> ((index % 8) * 8)) & 0xff)
                .expect("masked label byte fits u8");
            self.push_bytes(if byte.is_ascii_graphic() || byte == b' ' {
                std::slice::from_ref(&byte)
            } else {
                b"?"
            });
        }
    }

    fn as_slice(&self) -> &[u8] {
        &self.bytes[..self.length]
    }
}

#[cfg(test)]
mod tests {
    use super::{
        DiagnosticLine, PHASE_BEFORE, PHASE_RETURNED, ResourceCallRecord, ResourceCallTrace,
        ResourceKind, ResourceOwner, encode_label, format_resource_call,
    };
    use std::sync::atomic::Ordering;

    fn record(phase: u64) -> ResourceCallRecord {
        let (label_words, label_length) = encode_label("REMu scalar metadata");
        ResourceCallRecord {
            phase,
            kind: ResourceKind::FragmentBuffer as u64,
            index: 0,
            object: 0x7f7c_d001_8000,
            allocated_length: 65_536,
            offset: 384,
            range: 416,
            subresource: 0,
            owner: ResourceOwner::Batch as u64,
            label_length,
            label_words,
        }
    }

    #[test]
    fn publishes_only_complete_fixed_resource_calls_and_marks_returns() {
        let trace = ResourceCallTrace::new();
        trace.begin(42, 0x8055_cf00_00, 0x8054_cc00_00, 0x1234, true);
        let token = trace.push(record(PHASE_BEFORE));
        assert_eq!(trace.count.load(Ordering::Acquire), 1);
        assert_eq!(trace.calls[token.0].load(), Some(record(PHASE_BEFORE)));
        trace.calls[token.0].mark_returned();
        assert_eq!(trace.calls[token.0].load(), Some(record(PHASE_RETURNED)));
    }

    #[test]
    fn formats_resource_identity_range_owner_and_label_without_allocation() {
        let mut line = DiagnosticLine::new();
        format_resource_call(7, record(PHASE_RETURNED), &mut line);
        assert_eq!(
            line.as_slice(),
            b"[REMu][METAL-BINDING-FAULT] call=7 phase=returned kind=fragment-buffer index=0 \
              object=0x7f7cd0018000 allocated=65536 offset=384 range=416 subresource=0x0 \
              owner=guest-batch label=REMu scalar metadata\n"
        );
    }
}
