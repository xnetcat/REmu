// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

//! Lowering for the Metal shader compiler.
//!
//! `Compiler` remains one type; these modules carry the parts of its inherent
//! `impl` that belong to one compilation phase or one instruction class, plus
//! the free helpers that only that part uses. Declared here: `Compiler`
//! and the shared state accessors and emission primitives.

use std::collections::{BTreeMap, BTreeSet};

use remu_shader::{ImageBindingSlotMap, Program, ShaderEvaluation};

use crate::compiler::control_flow::ControlFlowLowering;
use crate::compiler::image::ImageKind;
use crate::compiler::interface::PixelInterpolation;
use crate::compiler::memory::is_lds_memory_instruction;
use crate::{MetalCompileOptions, MetalShaderStage};

pub(crate) mod control_flow;
pub(crate) mod driver;
pub(crate) mod export;
pub(crate) mod image;
pub(crate) mod interface;
pub(crate) mod memory;
pub(crate) mod module;
pub(crate) mod operand;
pub(crate) mod prelude;
pub(crate) mod scalar;
pub(crate) mod vector;

const MAX_METAL_THREADGROUP_THREADS: usize = 1_024;
const METAL_THREADGROUP_MEMORY_LIMIT_BYTES: usize = 32 * 1_024;
const SPLIT_WAVE64_HALF_WAVES_PER_THREADGROUP: usize = MAX_METAL_THREADGROUP_THREADS / 32;

pub(crate) struct Compiler<'a> {
    program: &'a Program,
    evaluation: &'a ShaderEvaluation,
    options: &'a MetalCompileOptions,
    pub(crate) body: String,
    pub(crate) next_temp: u32,
    vertex_outputs: BTreeSet<u32>,
    pixel_attributes: BTreeSet<u32>,
    pixel_interpolations: BTreeMap<u32, PixelInterpolation>,
    image_kinds: Vec<ImageKind>,
    image_slots: ImageBindingSlotMap,
    control_flow: ControlFlowLowering,
    /// VGPR holding the NGG allocated/compacted vertex id that gates EXP (e.g. `v18` in
    /// `V_CMPX_GT_U32 s4, v18` before POS). When set, mesh exports stage `remu_out` into
    /// `remu_mesh_staged_vertices[vN]` so the epilogue can `set_vertex` by compacted index
    /// instead of `remu_thread_index` (Poppy Slate UI tofu).
    mesh_export_vertex_vgpr: Option<u32>,
}

impl Compiler<'_> {
    fn is_threadgroup_stage(&self) -> bool {
        matches!(
            self.options.stage,
            MetalShaderStage::Compute | MetalShaderStage::Mesh
        )
    }

    fn split_wave64(&self) -> bool {
        self.is_threadgroup_stage() && self.options.wave_lane_count == 64
    }

    fn readfirstlane_site_count(&self) -> usize {
        self.program
            .instructions
            .iter()
            .filter(|instruction| instruction.opcode.is("VReadfirstlaneB32"))
            .count()
    }

    /// Returns the number of half-wave exchange records when every static `V_READFIRSTLANE` owns
    /// disjoint storage.
    ///
    /// Direct and forward-only control flow is acyclic, so an invocation can execute each static
    /// instruction at most once. Giving every site its own records removes the second whole-group
    /// barrier which otherwise protects a reused record from the next instruction. Dispatcher
    /// shaders can revisit a site and must retain the reused-record protocol.
    fn disjoint_readfirstlane_slot_count(&self) -> Option<usize> {
        if !self.split_wave64() || matches!(self.control_flow, ControlFlowLowering::Dispatcher) {
            return None;
        }
        let site_count = self.readfirstlane_site_count();
        let slot_count = site_count.checked_mul(SPLIT_WAVE64_HALF_WAVES_PER_THREADGROUP)?;
        let readfirstlane_bytes = slot_count.checked_mul(2 * size_of::<u32>())?;
        let readlane_bytes = if self
            .program
            .instructions
            .iter()
            .any(|instruction| instruction.opcode.is("VReadlaneB32"))
        {
            MAX_METAL_THREADGROUP_THREADS * size_of::<u32>()
        } else {
            0
        };
        let mesh_layer_bytes = if self.options.stage == MetalShaderStage::Mesh
            && self.options.render_target_array_length > 1
        {
            64 * size_of::<u32>()
        } else {
            0
        };
        let total_bytes = usize::try_from(self.options.lds_size_bytes)
            .ok()?
            .checked_add(readfirstlane_bytes)?
            .checked_add(readlane_bytes)?
            .checked_add(mesh_layer_bytes)?;
        (site_count != 0 && total_bytes <= METAL_THREADGROUP_MEMORY_LIMIT_BYTES)
            .then_some(slot_count)
    }

    fn disjoint_readfirstlane_site_base(&self, pc: u32) -> Option<usize> {
        self.disjoint_readfirstlane_slot_count()?;
        self.program
            .instructions
            .iter()
            .filter(|instruction| instruction.opcode.is("VReadfirstlaneB32"))
            .position(|instruction| instruction.pc == pc)
            .and_then(|site| site.checked_mul(SPLIT_WAVE64_HALF_WAVES_PER_THREADGROUP))
    }

    fn uses_lds_memory(&self) -> bool {
        self.program
            .instructions
            .iter()
            .any(is_lds_memory_instruction)
    }

    fn graphics_addtid_write_slot_count(&self) -> usize {
        if self.is_threadgroup_stage() {
            return 0;
        }
        self.program
            .instructions
            .iter()
            .filter(|instruction| instruction.opcode.is("DsWriteAddtidB32"))
            .count()
    }

    fn graphics_addtid_write_slot(&self, pc: u32) -> Option<usize> {
        self.program
            .instructions
            .iter()
            .filter(|instruction| instruction.opcode.is("DsWriteAddtidB32"))
            .position(|instruction| instruction.pc == pc)
    }

    fn architectural_lane(&self) -> &'static str {
        if self.split_wave64() {
            "remu_wave_lane"
        } else {
            "remu_lane"
        }
    }

    fn lane_mask_predicate(&self, low: &str, high: &str) -> String {
        if self.split_wave64() {
            format!(
                "(remu_wave_half == 0u ? ((({low}) >> remu_lane) & 1u) : ((({high}) >> remu_lane) & 1u)) != 0u"
            )
        } else {
            format!("((({low}) >> remu_lane) & 1u) != 0u")
        }
    }

    fn ballot_words(&self, condition: &str) -> (String, String) {
        let ballot = Self::ballot(condition);
        if self.split_wave64() {
            (
                format!("remu_wave_half == 0u ? {ballot} : 0u"),
                format!("remu_wave_half == 0u ? 0u : {ballot}"),
            )
        } else {
            (ballot, "0u".to_owned())
        }
    }

    fn global_metadata_word(&self, binding_index: usize, word: usize) -> String {
        let scalar_words = self.evaluation.initial_scalar_registers.len();
        let index = scalar_words + binding_index * 4 + word;
        format!("remu_initial_s[{index}]")
    }

    fn global_binding_base(&self, binding_index: usize) -> String {
        let low = self.global_metadata_word(binding_index, 0);
        let high = self.global_metadata_word(binding_index, 1);
        format!("((ulong){low} | ((ulong){high} << 32))")
    }

    fn global_binding_length(&self, binding_index: usize) -> String {
        self.global_metadata_word(binding_index, 2)
    }

    fn line(&mut self, indent: usize, text: &str) {
        for _ in 0..indent {
            self.body.push_str("    ");
        }
        self.body.push_str(text);
        self.body.push('\n');
    }

    fn temp(&mut self, ty: &str, expression: &str) -> String {
        let name = format!("t{}", self.next_temp);
        self.next_temp += 1;
        self.line(3, &format!("{ty} {name} = {expression};"));
        name
    }
}
