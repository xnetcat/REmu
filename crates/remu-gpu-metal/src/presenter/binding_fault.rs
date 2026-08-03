// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

//! Signal-safe evidence for Metal scalar-buffer binding faults.

use metal::foreign_types::ForeignTypeRef;
use std::ffi::c_void;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};

const REPRESENTATION_RETAINED_BUFFER: u64 = 2;
const REPRESENTATION_SKIPPED_EMPTY: u64 = 3;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct FragmentScalarBindingRecord {
    event_sequence: u64,
    export_shader: u64,
    pixel_shader: u64,
    binding_index: u64,
    byte_count: u64,
    render_pass_reused: u64,
    representation: u64,
    retained_resource: u64,
    retained_offset: u64,
}

struct AtomicFragmentScalarBindingRecord {
    valid: AtomicBool,
    event_sequence: AtomicU64,
    export_shader: AtomicU64,
    pixel_shader: AtomicU64,
    binding_index: AtomicU64,
    byte_count: AtomicU64,
    render_pass_reused: AtomicU64,
    representation: AtomicU64,
    retained_resource: AtomicU64,
    retained_offset: AtomicU64,
}

impl AtomicFragmentScalarBindingRecord {
    const fn new() -> Self {
        Self {
            valid: AtomicBool::new(false),
            event_sequence: AtomicU64::new(0),
            export_shader: AtomicU64::new(0),
            pixel_shader: AtomicU64::new(0),
            binding_index: AtomicU64::new(0),
            byte_count: AtomicU64::new(0),
            render_pass_reused: AtomicU64::new(0),
            representation: AtomicU64::new(0),
            retained_resource: AtomicU64::new(0),
            retained_offset: AtomicU64::new(0),
        }
    }

    fn store(&self, record: FragmentScalarBindingRecord) {
        self.event_sequence
            .store(record.event_sequence, Ordering::Relaxed);
        self.export_shader
            .store(record.export_shader, Ordering::Relaxed);
        self.pixel_shader
            .store(record.pixel_shader, Ordering::Relaxed);
        self.binding_index
            .store(record.binding_index, Ordering::Relaxed);
        self.byte_count.store(record.byte_count, Ordering::Relaxed);
        self.render_pass_reused
            .store(record.render_pass_reused, Ordering::Relaxed);
        self.representation
            .store(record.representation, Ordering::Relaxed);
        self.retained_resource
            .store(record.retained_resource, Ordering::Relaxed);
        self.retained_offset
            .store(record.retained_offset, Ordering::Relaxed);
        self.valid.store(true, Ordering::Relaxed);
    }

    fn load(&self) -> Option<FragmentScalarBindingRecord> {
        self.valid
            .load(Ordering::Relaxed)
            .then(|| FragmentScalarBindingRecord {
                event_sequence: self.event_sequence.load(Ordering::Relaxed),
                export_shader: self.export_shader.load(Ordering::Relaxed),
                pixel_shader: self.pixel_shader.load(Ordering::Relaxed),
                binding_index: self.binding_index.load(Ordering::Relaxed),
                byte_count: self.byte_count.load(Ordering::Relaxed),
                render_pass_reused: self.render_pass_reused.load(Ordering::Relaxed),
                representation: self.representation.load(Ordering::Relaxed),
                retained_resource: self.retained_resource.load(Ordering::Relaxed),
                retained_offset: self.retained_offset.load(Ordering::Relaxed),
            })
    }
}

struct LastFragmentScalarBinding {
    published: AtomicUsize,
    slots: [AtomicFragmentScalarBindingRecord; 2],
}

impl LastFragmentScalarBinding {
    const fn new() -> Self {
        Self {
            published: AtomicUsize::new(0),
            slots: [
                AtomicFragmentScalarBindingRecord::new(),
                AtomicFragmentScalarBindingRecord::new(),
            ],
        }
    }

    fn publish(&self, record: FragmentScalarBindingRecord) {
        let next = self.published.load(Ordering::Relaxed) ^ 1;
        self.slots[next].store(record);
        self.published.store(next, Ordering::Release);
    }

    fn snapshot(&self) -> Option<FragmentScalarBindingRecord> {
        let published = self.published.load(Ordering::Acquire);
        self.slots[published].load()
    }
}

static ENABLED: AtomicBool = AtomicBool::new(false);
static LAST_FRAGMENT_SCALAR_BINDING: LastFragmentScalarBinding = LastFragmentScalarBinding::new();

pub(super) fn install_fragment_scalar_binding_fault_diagnostic() {
    let requested =
        std::env::var_os("REMU_DIAGNOSTIC_METAL_BINDING_FAULT").is_some_and(|value| value == "1");
    if !requested {
        return;
    }
    if remu_memory::register_host_fault_diagnostic(write_last_fragment_scalar_binding) {
        ENABLED.store(true, Ordering::Release);
        eprintln!("[REMu][METAL-BINDING-FAULT] enabled marker=fragment-scalar-binding-v2");
    } else {
        eprintln!(
            "[REMu][METAL-BINDING-FAULT] unavailable: another host-fault callback is installed"
        );
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn record_fragment_scalar_retained_buffer_binding(
    event_sequence: u64,
    export_shader: u64,
    pixel_shader: u64,
    binding_index: u64,
    byte_count: u64,
    render_pass_reused: bool,
    buffer: &metal::BufferRef,
    offset: u64,
) {
    if !ENABLED.load(Ordering::Acquire) {
        return;
    }
    LAST_FRAGMENT_SCALAR_BINDING.publish(FragmentScalarBindingRecord {
        event_sequence,
        export_shader,
        pixel_shader,
        binding_index,
        byte_count,
        render_pass_reused: u64::from(render_pass_reused),
        representation: REPRESENTATION_RETAINED_BUFFER,
        retained_resource: u64::try_from(buffer.as_ptr() as usize)
            .expect("Metal resource pointer fits u64"),
        retained_offset: offset,
    });
}

#[allow(clippy::too_many_arguments)]
pub(super) fn record_fragment_scalar_empty_binding(
    event_sequence: u64,
    export_shader: u64,
    pixel_shader: u64,
    binding_index: u64,
    render_pass_reused: bool,
) {
    if !ENABLED.load(Ordering::Acquire) {
        return;
    }
    LAST_FRAGMENT_SCALAR_BINDING.publish(FragmentScalarBindingRecord {
        event_sequence,
        export_shader,
        pixel_shader,
        binding_index,
        byte_count: 0,
        render_pass_reused: u64::from(render_pass_reused),
        representation: REPRESENTATION_SKIPPED_EMPTY,
        retained_resource: 0,
        retained_offset: 0,
    });
}

unsafe extern "C" fn write_last_fragment_scalar_binding() {
    let Some(record) = LAST_FRAGMENT_SCALAR_BINDING.snapshot() else {
        return;
    };
    let mut line = DiagnosticLine::new();
    format_fragment_scalar_binding(record, &mut line);
    // SAFETY: `write` is async-signal-safe, and `line` remains live for this bounded call.
    unsafe {
        libc::write(
            libc::STDERR_FILENO,
            line.as_slice().as_ptr().cast::<c_void>(),
            line.as_slice().len(),
        );
    }
}

fn format_fragment_scalar_binding(record: FragmentScalarBindingRecord, line: &mut DiagnosticLine) {
    line.push_bytes(b"[REMu][METAL-BINDING-FAULT] last event=");
    line.push_decimal(record.event_sequence);
    line.push_bytes(b" es=");
    line.push_hexadecimal(record.export_shader);
    line.push_bytes(b" ps=");
    line.push_hexadecimal(record.pixel_shader);
    line.push_bytes(b" stage=fragment index=");
    line.push_decimal(record.binding_index);
    line.push_bytes(b" bytes=");
    line.push_decimal(record.byte_count);
    line.push_bytes(b" render_pass=");
    line.push_bytes(if record.render_pass_reused != 0 {
        b"reused"
    } else {
        b"new"
    });
    line.push_bytes(b" representation=");
    line.push_bytes(match record.representation {
        REPRESENTATION_RETAINED_BUFFER => b"retained-buffer",
        REPRESENTATION_SKIPPED_EMPTY => b"skipped-empty",
        _ => b"unknown",
    });
    line.push_bytes(b" resource=");
    line.push_hexadecimal(record.retained_resource);
    line.push_bytes(b" offset=");
    line.push_hexadecimal(record.retained_offset);
    line.push_bytes(b"\n");
}

struct DiagnosticLine {
    bytes: [u8; 384],
    length: usize,
}

impl DiagnosticLine {
    const fn new() -> Self {
        Self {
            bytes: [0; 384],
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

    fn as_slice(&self) -> &[u8] {
        &self.bytes[..self.length]
    }
}

#[cfg(test)]
mod tests {
    use super::{
        format_fragment_scalar_binding, DiagnosticLine, FragmentScalarBindingRecord,
        LastFragmentScalarBinding, REPRESENTATION_RETAINED_BUFFER, REPRESENTATION_SKIPPED_EMPTY,
    };

    fn record(sequence: u64, representation: u64) -> FragmentScalarBindingRecord {
        FragmentScalarBindingRecord {
            event_sequence: sequence,
            export_shader: 0x8055_cf00_00,
            pixel_shader: 0x8054_cc00_00,
            binding_index: 0,
            byte_count: 416,
            render_pass_reused: 1,
            representation,
            retained_resource: 0x7f7c_d001_8000,
            retained_offset: 0x180,
        }
    }

    #[test]
    fn publishes_only_complete_fixed_slot_records() {
        let state = LastFragmentScalarBinding::new();
        assert_eq!(state.snapshot(), None);
        state.publish(record(41, REPRESENTATION_SKIPPED_EMPTY));
        state.publish(record(42, REPRESENTATION_RETAINED_BUFFER));
        assert_eq!(
            state.snapshot(),
            Some(record(42, REPRESENTATION_RETAINED_BUFFER))
        );
    }

    #[test]
    fn formats_the_complete_binding_identity_without_allocation() {
        let mut line = DiagnosticLine::new();
        format_fragment_scalar_binding(record(42, REPRESENTATION_RETAINED_BUFFER), &mut line);
        assert_eq!(
            line.as_slice(),
            b"[REMu][METAL-BINDING-FAULT] last event=42 es=0x8055cf0000 ps=0x8054cc0000 \
              stage=fragment index=0 bytes=416 render_pass=reused representation=retained-buffer \
              resource=0x7f7cd0018000 offset=0x180\n"
        );
    }

    #[test]
    fn formats_an_empty_binding_as_skipped_without_a_resource() {
        let mut empty = record(43, REPRESENTATION_SKIPPED_EMPTY);
        empty.byte_count = 0;
        empty.retained_resource = 0;
        empty.retained_offset = 0;
        let mut line = DiagnosticLine::new();
        format_fragment_scalar_binding(empty, &mut line);
        assert!(line
            .as_slice()
            .windows(b"representation=skipped-empty".len())
            .any(|window| { window == b"representation=skipped-empty" }));
        assert!(line.as_slice().ends_with(b"resource=0x0 offset=0x0\n"));
    }
}
