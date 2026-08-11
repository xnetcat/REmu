// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

//! Backend-neutral Gen5/RDNA2 shader decoding, evaluation, and intermediate representation.

mod cfg;
mod coverage;
mod decoder;
mod evaluator;
mod format;
mod inline_constant;
mod ir;
mod metadata;
mod state;

pub use cfg::{
    BasicBlock, BlockExit, BlockId, ControlFlowGraph, Dominators, NaturalLoop, PostDominators,
    StructuredCfg, build_control_flow_graph,
};
pub use coverage::{OpcodeHistogram, OpcodeUsage, opcode_coverage_report};
pub use decoder::{
    DecodeError, decode_program, decode_program_segments, read_program, read_program_bounded,
    read_program_segments_bounded,
};
pub use evaluator::{
    EvaluationError, EvaluationOptions, evaluate_shader, truncated_evaluation_report,
};
pub use format::decode_unified_format;
pub use inline_constant::{decode_inline_constant, decode_v2f16_inline_constant};
pub use ir::{
    BufferMemoryControl, ComputeSystemRegisters, ComputeSystemValue, DataShareControl, Dpp8Control,
    DppControl, ExportControl, GlobalMemoryBinding, GlobalMemoryControl, ImageBinding,
    ImageBindingSlotMap, ImageControl, ImageSampleModifiers, ImageSignatureClass,
    ImageViewReinterpretation, Instruction, InstructionControl, InterpolationControl, Opcode,
    Operand, OperandKind, PixelExportFormat, PixelOutputBinding, PixelOutputKind, Program,
    ProgramLinkError, RawOpcodeFamily, ResourceKind, ResourceMapping, ScalarMemoryControl,
    SdwaControl, ShaderEncoding, ShaderEvaluation, ShaderMetadata, ShaderState,
    TextureDescriptorType, VertexInputBinding, Vop3Control, Vop3pControl, image_is_comparison,
    image_resource_descriptor_is_null, image_sample_modifiers, link_program_continuation,
};
pub use metadata::{GuestMemoryReader, MetadataError, read_shader_metadata};
pub use state::{ShaderStateError, build_shader_state};
