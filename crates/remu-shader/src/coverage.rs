// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

//! Process-wide opcode coverage recording for decoded shader programs.
//!
//! A backend reports the instructions it cannot translate one shader at a time, and only for the
//! shaders a session actually reaches. Recording every decoded opcode instead describes the whole
//! ISA surface a title needs from a single run, so a translation gap can be closed as a batch.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::{Mutex, OnceLock, PoisonError};

use crate::ir::{Opcode, Program};

/// Set to `1` to accumulate decoded opcodes. Recording costs one atomic load per decoded program
/// while disabled, which keeps decoding on the draw path unaffected by default.
const COVERAGE_VARIABLE: &str = "REMU_TRACE_SHADER_OPCODES";

const FNV_OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

static HISTOGRAM: Mutex<OpcodeHistogram> = Mutex::new(OpcodeHistogram::new());

/// How many times one opcode was decoded, and how many distinct shaders contained it.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct OpcodeUsage {
    pub occurrences: u64,
    pub shaders: u64,
}

/// Opcode usage accumulated across decoded shader programs.
///
/// Shaders are identified by their instruction stream rather than their guest address, so a title
/// that re-decodes the same program on every draw, or relocates it, contributes exactly once.
#[derive(Debug, Default)]
pub struct OpcodeHistogram {
    usage: BTreeMap<Opcode, OpcodeUsage>,
    programs: BTreeSet<u64>,
}

impl OpcodeHistogram {
    /// Creates an empty histogram.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            usage: BTreeMap::new(),
            programs: BTreeSet::new(),
        }
    }

    /// Folds one decoded program into the histogram. A program whose instruction stream was
    /// already recorded is ignored.
    pub fn record(&mut self, program: &Program) {
        if !self.programs.insert(program_fingerprint(program)) {
            return;
        }
        let mut counted = BTreeSet::new();
        for instruction in &program.instructions {
            let usage = self.usage.entry(instruction.opcode).or_default();
            usage.occurrences = usage.occurrences.saturating_add(1);
            if counted.insert(instruction.opcode) {
                usage.shaders = usage.shaders.saturating_add(1);
            }
        }
    }

    /// Number of distinct shaders folded into the histogram.
    #[must_use]
    pub fn shader_count(&self) -> usize {
        self.programs.len()
    }

    /// Usage recorded for one opcode.
    #[must_use]
    pub fn usage(&self, opcode: Opcode) -> OpcodeUsage {
        self.usage.get(&opcode).copied().unwrap_or_default()
    }

    /// Renders the histogram as report lines, most used opcode first, without any log prefix.
    ///
    /// Opcodes the decoder resolved are reported by name. Opcodes it could only frame are reported
    /// by family and numeric code and counted separately in the header, because those are gaps in
    /// the decoder rather than in a backend.
    #[must_use]
    pub fn report_lines(&self) -> Vec<String> {
        if self.usage.is_empty() {
            return Vec::new();
        }
        let mut entries: Vec<_> = self.usage.iter().collect();
        entries.sort_by(|(left_opcode, left), (right_opcode, right)| {
            right
                .occurrences
                .cmp(&left.occurrences)
                .then_with(|| left_opcode.cmp(right_opcode))
        });
        let raw_count = entries
            .iter()
            .filter(|(opcode, _)| opcode.known_name().is_none())
            .count();
        let mut lines = Vec::with_capacity(entries.len() + 1);
        lines.push(format!(
            "shaders={} opcodes={} raw={raw_count}",
            self.programs.len(),
            entries.len()
        ));
        lines.extend(entries.into_iter().map(|(opcode, usage)| {
            let description = match opcode {
                Opcode::Known(name) => format!("opcode={name}"),
                Opcode::Raw { family, code } => format!("raw={family:?} code=0x{code:X}"),
            };
            format!(
                "uses={} shaders={} {description}",
                usage.occurrences, usage.shaders
            )
        }));
        lines
    }
}

/// Renders the process-wide opcode histogram as report lines without any log prefix.
///
/// The report is empty unless `REMU_TRACE_SHADER_OPCODES=1` was set for the process, and while no
/// shader has been decoded.
#[must_use]
pub fn opcode_coverage_report() -> Vec<String> {
    if !enabled() {
        return Vec::new();
    }
    HISTOGRAM
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .report_lines()
}

pub(crate) fn record_program(program: &Program) {
    if !enabled() {
        return;
    }
    HISTOGRAM
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .record(program);
}

fn enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os(COVERAGE_VARIABLE).is_some_and(|value| value == "1"))
}

fn program_fingerprint(program: &Program) -> u64 {
    let mut hash = FNV_OFFSET_BASIS;
    for instruction in &program.instructions {
        for word in &instruction.words {
            for byte in word.to_le_bytes() {
                hash = (hash ^ u64::from(byte)).wrapping_mul(FNV_PRIME);
            }
        }
    }
    hash
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{Instruction, Operand, RawOpcodeFamily, ShaderEncoding};

    fn program(address: u64, opcodes: &[(Opcode, u32)]) -> Program {
        let instructions = opcodes
            .iter()
            .enumerate()
            .map(|(index, (opcode, word))| Instruction {
                pc: u32::try_from(index * 4).expect("test program fits the 32-bit program counter"),
                encoding: ShaderEncoding::Vop1,
                opcode: *opcode,
                words: vec![*word],
                sources: vec![Operand::vector(1)],
                destinations: vec![Operand::vector(2)],
                control: None,
            })
            .collect();
        Program::new(address, instructions)
    }

    #[test]
    fn the_histogram_counts_repeated_opcodes_and_the_shaders_that_use_them() {
        let mut histogram = OpcodeHistogram::new();
        histogram.record(&program(
            0x1000,
            &[
                (Opcode::Known("VMovB32"), 1),
                (Opcode::Known("VMovB32"), 2),
                (Opcode::Known("VAddF32"), 3),
            ],
        ));
        histogram.record(&program(0x2000, &[(Opcode::Known("VMovB32"), 4)]));

        assert_eq!(histogram.shader_count(), 2);
        assert_eq!(
            histogram.usage(Opcode::Known("VMovB32")),
            OpcodeUsage {
                occurrences: 3,
                shaders: 2,
            }
        );
        assert_eq!(
            histogram.usage(Opcode::Known("VAddF32")),
            OpcodeUsage {
                occurrences: 1,
                shaders: 1,
            }
        );
        assert_eq!(
            histogram.usage(Opcode::Known("VMulF32")),
            OpcodeUsage::default()
        );
    }

    #[test]
    fn decoding_the_same_instruction_stream_again_does_not_inflate_the_histogram() {
        let mut histogram = OpcodeHistogram::new();
        let opcodes = [(Opcode::Known("VMovB32"), 7)];
        histogram.record(&program(0x1000, &opcodes));
        histogram.record(&program(0x1000, &opcodes));
        histogram.record(&program(0x9000, &opcodes));

        assert_eq!(histogram.shader_count(), 1);
        assert_eq!(
            histogram.usage(Opcode::Known("VMovB32")),
            OpcodeUsage {
                occurrences: 1,
                shaders: 1,
            }
        );
    }

    #[test]
    fn the_report_is_ordered_by_use_and_names_raw_opcodes_by_family_and_code() {
        let raw = Opcode::Raw {
            family: RawOpcodeFamily::Vop3,
            code: 0x2a5,
        };
        let mut histogram = OpcodeHistogram::new();
        histogram.record(&program(
            0x1000,
            &[
                (raw, 1),
                (Opcode::Known("VAddF32"), 2),
                (Opcode::Known("VAddF32"), 3),
                (Opcode::Known("VAddF32"), 4),
                (Opcode::Known("VMovB32"), 5),
                (Opcode::Known("VMovB32"), 6),
            ],
        ));

        assert_eq!(
            histogram.report_lines(),
            vec![
                "shaders=1 opcodes=3 raw=1".to_owned(),
                "uses=3 shaders=1 opcode=VAddF32".to_owned(),
                "uses=2 shaders=1 opcode=VMovB32".to_owned(),
                "uses=1 shaders=1 raw=Vop3 code=0x2A5".to_owned(),
            ]
        );
    }

    #[test]
    fn an_empty_histogram_reports_nothing() {
        assert!(OpcodeHistogram::new().report_lines().is_empty());
    }
}
