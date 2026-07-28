// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

//! Memory lowering: scalar loads, MUBUF/MTBUF buffers, FLAT/GLOBAL and LDS data share.
//!
//! Each submodule carries one address space; this module keeps the classification
//! predicates the compiler shares between them.

use remu_shader::Instruction;

mod buffer;
mod data_share;
mod global;

pub(crate) use buffer::is_format_buffer_memory;
#[cfg(test)]
pub(crate) use buffer::{buffer_oob_expression, format_buffer_element_oob_expression};
pub(crate) use data_share::is_lds_memory_instruction;

pub(crate) fn is_writable_memory(instruction: &Instruction) -> bool {
    instruction.opcode.contains("Store") || instruction.opcode.contains("Atomic")
}
