// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

//! Backend-neutral Gen5/RDNA2 shader decoding, evaluation, and intermediate representation.

mod decoder;
mod evaluator;
mod format;
mod inline_constant;
mod ir;
mod metadata;
mod state;

pub use decoder::{DecodeError, decode_program, read_program};
pub use evaluator::{EvaluationError, EvaluationOptions, evaluate_shader};
pub use format::decode_unified_format;
pub use inline_constant::{decode_inline_constant, decode_v2f16_inline_constant};
pub use ir::{
    BufferMemoryControl, ComputeSystemRegisters, ComputeSystemValue, DataShareControl, Dpp8Control,
    DppControl, ExportControl, GlobalMemoryBinding, GlobalMemoryControl, ImageBinding,
    ImageControl, Instruction, InstructionControl, InterpolationControl, Opcode, Operand,
    OperandKind, PixelOutputBinding, PixelOutputKind, Program, RawOpcodeFamily, ResourceKind,
    ResourceMapping, ScalarMemoryControl, SdwaControl, ShaderEncoding, ShaderEvaluation,
    ShaderMetadata, ShaderState, VertexInputBinding, Vop3Control, Vop3pControl,
};
pub use metadata::{GuestMemoryReader, MetadataError, read_shader_metadata};
pub use state::{ShaderStateError, build_shader_state};
