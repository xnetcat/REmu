// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

//! Allocation-free line rendering for code that runs in signal context.
//!
//! Signal context forbids allocation and `core::fmt`, so every field is rendered by hand into
//! caller-owned storage.

/// Fixed capacity for one complete report line, including its register list and newline.
pub(crate) const DIAGNOSTIC_CAPACITY: usize = 1280;

pub(crate) struct DiagnosticBuffer {
    bytes: [u8; DIAGNOSTIC_CAPACITY],
    length: usize,
}

impl DiagnosticBuffer {
    pub(crate) const fn new() -> Self {
        Self {
            bytes: [0; DIAGNOSTIC_CAPACITY],
            length: 0,
        }
    }

    /// Appends as much of `value` as still fits. Truncation keeps the report readable instead of
    /// panicking inside a signal handler.
    pub(crate) fn push_bytes(&mut self, value: &[u8]) {
        let available = DIAGNOSTIC_CAPACITY - self.length;
        let copied = value.len().min(available);
        self.bytes[self.length..self.length + copied].copy_from_slice(&value[..copied]);
        self.length += copied;
    }

    pub(crate) fn push_hexadecimal(&mut self, value: u64) {
        self.push_bytes(b"0x");
        let mut digits = [0_u8; 16];
        for (index, digit) in digits.iter_mut().enumerate() {
            let nibble = (value >> (60 - index * 4)) & 0xf;
            *digit = b"0123456789abcdef"[nibble as usize];
        }
        let leading = digits
            .iter()
            .position(|&digit| digit != b'0')
            .unwrap_or(digits.len() - 1);
        self.push_bytes(&digits[leading..]);
    }

    pub(crate) fn push_signed(&mut self, value: i64) {
        let magnitude = if value < 0 {
            self.push_bytes(b"-");
            value.unsigned_abs()
        } else {
            value.unsigned_abs()
        };
        let mut digits = [0_u8; 20];
        let mut index = digits.len();
        let mut remaining = magnitude;
        loop {
            index -= 1;
            digits[index] = b'0' + (remaining % 10) as u8;
            remaining /= 10;
            if remaining == 0 {
                break;
            }
        }
        self.push_bytes(&digits[index..]);
    }

    /// Restarts the buffer so one instance can render an unbounded number of lines.
    pub(crate) const fn clear(&mut self) {
        self.length = 0;
    }

    pub(crate) fn as_slice(&self) -> &[u8] {
        &self.bytes[..self.length]
    }
}

#[cfg(test)]
mod tests {
    use super::{DIAGNOSTIC_CAPACITY, DiagnosticBuffer};

    fn push_hexadecimal_for_test(value: u64) -> String {
        let mut buffer = DiagnosticBuffer::new();
        buffer.push_hexadecimal(value);
        String::from_utf8(buffer.as_slice().to_vec()).expect("hexadecimal digits are ASCII")
    }

    #[test]
    fn renders_boundary_addresses_without_truncation_or_padding() {
        assert_eq!(push_hexadecimal_for_test(0), "0x0");
        assert_eq!(push_hexadecimal_for_test(u64::MAX), "0xffffffffffffffff");
        assert_eq!(push_hexadecimal_for_test(0x10), "0x10");
    }

    #[test]
    fn truncates_instead_of_panicking_when_a_report_exceeds_its_buffer() {
        let mut buffer = DiagnosticBuffer::new();
        for _ in 0..=DIAGNOSTIC_CAPACITY / 10 {
            buffer.push_bytes(b"0123456789");
        }
        assert_eq!(buffer.as_slice().len(), DIAGNOSTIC_CAPACITY);
    }

    #[test]
    fn clearing_restarts_rendering_without_retaining_earlier_bytes() {
        let mut buffer = DiagnosticBuffer::new();
        buffer.push_bytes(b"first");
        buffer.clear();
        buffer.push_bytes(b"second");
        assert_eq!(buffer.as_slice(), b"second");
    }
}
