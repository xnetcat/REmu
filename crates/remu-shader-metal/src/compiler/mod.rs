// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

//! Instruction-class lowering for the Metal shader compiler.
//!
//! `Compiler` remains one type, declared in the crate root; these modules
//! carry the parts of its inherent `impl` that belong to one instruction class,
//! plus the free helpers that only that class uses.

pub(crate) mod control_flow;
pub(crate) mod image;
pub(crate) mod memory;
pub(crate) mod prelude;
pub(crate) mod scalar;
pub(crate) mod vector;
