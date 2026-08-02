// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

//! Allocation-free fault reporting for the write-watch signal handlers.
//!
//! The handler observes each process SIGSEGV/SIGBUS while CPU write watching is active, which makes
//! this the only report emitted before an unrelated fault refaults under the previous action and
//! terminates the process.

pub(super) use crate::signal_text::DiagnosticBuffer;

/// Where a faulting address lives relative to the emulator's guest address-space reservations.
#[derive(Clone, Copy, Eq, PartialEq)]
pub(super) enum FaultSpace {
    /// Inside a live guest reservation: translated guest code or an emulator access to guest bytes.
    Guest,
    /// Outside every guest reservation: ordinary host memory such as shared-backing scratch maps.
    Host,
}

impl FaultSpace {
    const fn label(self) -> &'static [u8] {
        match self {
            Self::Guest => b"guest",
            Self::Host => b"host",
        }
    }
}

/// Why the write-watch layer could not service a fault.
#[derive(Clone, Copy, Eq, PartialEq)]
pub(super) enum FaultReason {
    NullAddress,
    NoSnapshot,
    Unwatched,
    AddressConversion,
    LengthConversion,
    ProtectFailed,
    IllegalInstruction,
}

impl FaultReason {
    const fn label(self) -> &'static [u8] {
        match self {
            Self::IllegalInstruction => b"illegal-instruction",
            Self::NullAddress => b"null-address",
            Self::NoSnapshot => b"no-watch-snapshot",
            Self::Unwatched => b"outside-watched-memory",
            Self::AddressConversion => b"address-conversion-failed",
            Self::LengthConversion => b"length-conversion-failed",
            Self::ProtectFailed => b"mprotect-failed",
        }
    }
}

/// The kernel-supplied description of one delivered protection fault.
#[derive(Clone, Copy)]
pub(super) struct FaultReport {
    pub(super) signal: i32,
    pub(super) code: i32,
    pub(super) address: u64,
    pub(super) program_counter: u64,
    pub(super) space: FaultSpace,
}

/// Renders the full fault line into `buffer`.
///
/// `si_code` is the field that separates the plausible causes: `SEGV_MAPERR` is an access to
/// memory that is not mapped at all, `SEGV_ACCERR` is a permission failure on mapped memory (the
/// signature of lost write-watch bookkeeping), and `BUS_OBJERR` is a shared-backing object that
/// could not supply a page. The program counter separates translated guest code from emulator code
/// touching guest bytes, which the fault address alone cannot do.
pub(super) fn format_fault_report(
    report: FaultReport,
    reason: FaultReason,
    buffer: &mut DiagnosticBuffer,
) {
    buffer.push_bytes(b"[REMu][WATCH] unhandled ");
    buffer.push_bytes(signal_name(report.signal));
    buffer.push_bytes(b" code=");
    buffer.push_signed(i64::from(report.code));
    buffer.push_bytes(b"(");
    buffer.push_bytes(signal_code_name(report.signal, report.code));
    buffer.push_bytes(b") addr=");
    buffer.push_hexadecimal(report.address);
    buffer.push_bytes(b" pc=");
    buffer.push_hexadecimal(report.program_counter);
    buffer.push_bytes(b" space=");
    buffer.push_bytes(report.space.label());
    buffer.push_bytes(b" reason=");
    buffer.push_bytes(reason.label());
    buffer.push_bytes(b"; forwarding\n");
}

/// One interrupted machine register, named as the guest ABI names it.
#[derive(Clone, Copy)]
pub(super) struct FaultRegister {
    pub(super) name: &'static [u8],
    pub(super) value: u64,
}

/// Renders the interrupted integer register file as a second report line.
///
/// A wild pointer is only actionable once its provenance is known. The register that holds the
/// faulting address names the guest expression that produced it, and the neighbouring registers
/// usually still hold the structure base it was loaded from, which identifies the guest allocation
/// whose contents are wrong. Neither is recoverable from a crash report once the process is gone,
/// because the faulting store repeats under the previous action on a thread the reporter has left.
pub(super) fn format_fault_registers(registers: &[FaultRegister], buffer: &mut DiagnosticBuffer) {
    buffer.push_bytes(b"[REMu][WATCH] fault registers");
    for register in registers {
        buffer.push_bytes(b" ");
        buffer.push_bytes(register.name);
        buffer.push_bytes(b"=");
        buffer.push_hexadecimal(register.value);
    }
    buffer.push_bytes(b"\n");
}

/// Renders a bounded snapshot of the interrupted stack without allocating.
///
/// Raw words are intentional: Rosetta does not always produce a Darwin crash report after the
/// previous handler is restored, while return addresses in this snapshot can still be resolved
/// against the exact release binary and shared-cache image list from the run.
pub(super) fn format_fault_stack_words(
    stack_pointer: u64,
    words: &[u64],
    buffer: &mut DiagnosticBuffer,
) {
    buffer.push_bytes(b"[REMu][WATCH] fault stack rsp=");
    buffer.push_hexadecimal(stack_pointer);
    buffer.push_bytes(b" words=");
    for (index, word) in words.iter().enumerate() {
        if index != 0 {
            buffer.push_bytes(b",");
        }
        buffer.push_hexadecimal(*word);
    }
    buffer.push_bytes(b"\n");
}

fn signal_name(signal: i32) -> &'static [u8] {
    if signal == libc::SIGBUS {
        b"SIGBUS"
    } else if signal == libc::SIGSEGV {
        b"SIGSEGV"
    } else {
        b"signal"
    }
}

// Darwin's `<sys/signal.h>` fault codes. The `libc` crate does not expose them for Apple targets.
/// The access had no mapping at all.
const SEGV_MAPERR: i32 = 1;
/// The mapping exists but denied the access, which is the signature of lost protection bookkeeping.
pub(super) const SEGV_ACCERR: i32 = 2;
const BUS_ADRALN: i32 = 1;
const BUS_ADRERR: i32 = 2;
/// The backing object could not supply the page, for example exhausted shared memory.
const BUS_OBJERR: i32 = 3;

fn signal_code_name(signal: i32, code: i32) -> &'static [u8] {
    if signal == libc::SIGSEGV {
        match code {
            SEGV_MAPERR => b"SEGV_MAPERR",
            SEGV_ACCERR => b"SEGV_ACCERR",
            _ => b"unknown",
        }
    } else if signal == libc::SIGBUS {
        match code {
            BUS_ADRALN => b"BUS_ADRALN",
            BUS_ADRERR => b"BUS_ADRERR",
            BUS_OBJERR => b"BUS_OBJERR",
            _ => b"unknown",
        }
    } else {
        b"unknown"
    }
}

#[cfg(test)]
mod tests {
    use super::{
        BUS_OBJERR, DiagnosticBuffer, FaultReason, FaultRegister, FaultReport, FaultSpace,
        SEGV_ACCERR, format_fault_registers, format_fault_report, format_fault_stack_words,
    };

    fn rendered(report: FaultReport, reason: FaultReason) -> String {
        let mut buffer = DiagnosticBuffer::new();
        format_fault_report(report, reason, &mut buffer);
        String::from_utf8(buffer.as_slice().to_vec()).expect("fault reports are ASCII")
    }

    #[test]
    fn reports_permission_faults_inside_guest_space_with_their_program_counter() {
        assert_eq!(
            rendered(
                FaultReport {
                    signal: libc::SIGSEGV,
                    code: SEGV_ACCERR,
                    address: 0x8063_d500_0000,
                    program_counter: 0x8055_ca00_1234,
                    space: FaultSpace::Guest,
                },
                FaultReason::Unwatched,
            ),
            "[REMu][WATCH] unhandled SIGSEGV code=2(SEGV_ACCERR) addr=0x8063d5000000 \
             pc=0x8055ca001234 space=guest reason=outside-watched-memory; forwarding\n"
        );
    }

    #[test]
    fn reports_shared_backing_object_errors_as_host_space_bus_faults() {
        assert_eq!(
            rendered(
                FaultReport {
                    signal: libc::SIGBUS,
                    code: BUS_OBJERR,
                    address: 0x1_0000,
                    program_counter: 0,
                    space: FaultSpace::Host,
                },
                FaultReason::Unwatched,
            ),
            "[REMu][WATCH] unhandled SIGBUS code=3(BUS_OBJERR) addr=0x10000 pc=0x0 space=host \
             reason=outside-watched-memory; forwarding\n"
        );
    }

    #[test]
    fn reports_the_register_that_carried_a_wild_pointer_next_to_its_structure_base() {
        let mut buffer = DiagnosticBuffer::new();
        format_fault_registers(
            &[
                FaultRegister {
                    name: b"rax",
                    value: 0xff00_0000_ff00_0008,
                },
                FaultRegister {
                    name: b"rbx",
                    value: 0x8063_d500_0000,
                },
            ],
            &mut buffer,
        );
        assert_eq!(
            String::from_utf8(buffer.as_slice().to_vec()).expect("register lines are ASCII"),
            "[REMu][WATCH] fault registers rax=0xff000000ff000008 rbx=0x8063d5000000\n"
        );
    }

    #[test]
    fn holds_a_complete_report_and_register_line_for_the_widest_register_file() {
        let mut buffer = DiagnosticBuffer::new();
        format_fault_report(
            FaultReport {
                signal: libc::SIGSEGV,
                code: SEGV_ACCERR,
                address: u64::MAX,
                program_counter: u64::MAX,
                space: FaultSpace::Guest,
            },
            FaultReason::Unwatched,
            &mut buffer,
        );
        let registers = [FaultRegister {
            name: b"x28",
            value: u64::MAX,
        }; 32];
        format_fault_registers(&registers, &mut buffer);
        assert!(buffer.as_slice().ends_with(b"\n"));
    }

    #[test]
    fn reports_raw_interrupted_stack_words_for_offline_symbolication() {
        let mut buffer = DiagnosticBuffer::new();
        format_fault_stack_words(
            0x30a8_2f2b0,
            &[0x1002_bd1a0, 0x7ff9_1107_0960, 0],
            &mut buffer,
        );
        assert_eq!(
            String::from_utf8(buffer.as_slice().to_vec()).expect("stack lines are ASCII"),
            "[REMu][WATCH] fault stack rsp=0x30a82f2b0 \
             words=0x1002bd1a0,0x7ff911070960,0x0\n"
        );
    }
}
