// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

//! Shader evaluation tests, split by instruction family.
//!
//! This module holds the guest memory stub and program fixtures shared across
//! families; each submodule owns the tests for one family.

use std::sync::Arc;

use super::*;
use crate::{ExportControl, Program};

mod buffer;
mod control_flow;
mod resources;
mod scalar_alu;

#[derive(Default)]
struct TestMemory(BTreeMap<u64, u8>);

impl TestMemory {
    fn write(&mut self, address: u64, bytes: &[u8]) {
        for (offset, byte) in bytes.iter().copied().enumerate() {
            self.0.insert(address + offset as u64, byte);
        }
    }
}

impl GuestMemoryReader for TestMemory {
    fn read(&self, address: u64, output: &mut [u8]) -> bool {
        for (offset, destination) in output.iter_mut().enumerate() {
            let Some(byte) = self.0.get(&(address + offset as u64)) else {
                return false;
            };
            *destination = *byte;
        }
        true
    }
}

fn instruction(
    pc: u32,
    encoding: ShaderEncoding,
    opcode: &'static str,
    sources: Vec<Operand>,
    destinations: Vec<Operand>,
    control: Option<InstructionControl>,
) -> Instruction {
    Instruction {
        pc,
        encoding,
        opcode: Opcode::from_static(opcode),
        words: vec![0],
        sources,
        destinations,
        control,
    }
}

fn end(pc: u32) -> Instruction {
    instruction(
        pc,
        ShaderEncoding::Sopp,
        "SEndpgm",
        Vec::new(),
        Vec::new(),
        None,
    )
}

fn state(instructions: Vec<Instruction>, user_data: Vec<u32>) -> ShaderState {
    ShaderState {
        program: Arc::new(Program::new(0x1000, instructions)),
        user_data: user_data.into(),
        compute_system_registers: None,
        user_data_scalar_register_base: 0,
    }
}
