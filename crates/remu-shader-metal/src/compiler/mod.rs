// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

//! Lowering for the Metal shader compiler.
//!
//! `Compiler` remains one type; these modules carry the parts of its inherent
//! `impl` that belong to one compilation phase or one instruction class, plus
//! the free helpers that only that part uses. Declared here: `Compiler`
//! and the shared state accessors and emission primitives.

use std::collections::{BTreeMap, BTreeSet};

use remu_shader::{
    ImageBindingSlotMap, Instruction, OperandKind, Program, ShaderEvaluation, StructuredCfg,
};

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
const SPLIT_WAVE64_PAIR_STATE_SLOTS: usize = SPLIT_WAVE64_HALF_WAVES_PER_THREADGROUP * 2;
const SPLIT_WAVE64_PAIR_STATE_BYTES: usize = SPLIT_WAVE64_PAIR_STATE_SLOTS * size_of::<u32>();

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
    /// Image binding index -> (srsrc base SGPR, candidate heap global-binding indices) for
    /// bindings whose descriptor index resolves only at runtime; empty unless
    /// options.runtime_heap_tables. Several heaps mean the base is loaded from different
    /// SMEM sources across control-flow paths and the shader selects at runtime.
    pub(crate) heap_tables: std::collections::BTreeMap<usize, (u32, Vec<usize>)>,
    control_flow: ControlFlowLowering,
    /// VGPR holding the NGG compacted output vertex id that gates EXP (the `vM` in
    /// `V_CMPX_* sN, vM` immediately before `EXP POS`). When the mesh ABI has an explicit
    /// capacity, each export publishes directly to that Metal vertex slot so PRIM indices and
    /// vertex/PARAM records share the same domain.
    mesh_export_vertex_vgpr: Option<u32>,
}

impl Compiler<'_> {
    fn diagnostic_compute_state_trace_group(&self) -> Option<[u32; 3]> {
        if self.options.stage != MetalShaderStage::Compute {
            return None;
        }
        let filter = std::env::var("REMU_TRACE_COMPUTE_STATE_CS").ok()?;
        let filter = filter.trim().trim_start_matches("0x");
        let address = u64::from_str_radix(filter, 16).ok()?;
        if address != self.program.address {
            return None;
        }
        let group = std::env::var("REMU_TRACE_COMPUTE_STATE_GROUP")
            .unwrap_or_else(|_| "72,40,0".to_owned());
        let mut components = group.split(',').map(str::trim).map(str::parse::<u32>);
        let parsed = [
            components.next()?.ok()?,
            components.next()?.ok()?,
            components.next()?.ok()?,
        ];
        components.next().is_none().then_some(parsed)
    }

    fn diagnostic_compute_state_trace_checkpoint(&self, pc: u32) -> Option<usize> {
        self.diagnostic_compute_state_trace_group()?;
        // These PCs bracket the texture reads and final color calculation in the selected
        // diagnostic shader. The mechanism remains address-gated, so ordinary compilation is
        // byte-for-byte unchanged.
        [0x20, 0x228, 0x340, 0x3dc, 0x7e4, 0x884, 0xa4c, 0xa88]
            .iter()
            .position(|&checkpoint| checkpoint == pc)
    }

    fn is_threadgroup_stage(&self) -> bool {
        matches!(
            self.options.stage,
            MetalShaderStage::Compute | MetalShaderStage::Mesh
        )
    }

    /// The heap-table route for one image binding, when analysis produced one.
    pub(crate) fn runtime_heap_table(&self, binding_index: usize) -> Option<&(u32, Vec<usize>)> {
        self.heap_tables.get(&binding_index)
    }

    fn split_wave64(&self) -> bool {
        self.is_threadgroup_stage() && self.options.wave_lane_count == 64
    }

    /// Whether the two Metal SIMD32 halves are the complete compute workgroup.
    ///
    /// In this shape a guest-wave reconvergence point is also a legal Metal threadgroup barrier,
    /// providing the cross-SIMD LDS visibility that a pair of relaxed atomics plus
    /// `simdgroup_barrier` cannot establish.
    fn split_wave64_is_whole_compute_threadgroup(&self) -> bool {
        self.options.stage == MetalShaderStage::Compute
            && self.split_wave64()
            && (1..=64).contains(&self.options.compute_threadgroup_size)
    }

    /// Whether a wave64 compute program needs its two host SIMD32 mask words exchanged before a
    /// scalar mask branch.
    ///
    /// Each Metal SIMD32 half owns the correct local ballot word, but `S_CBRANCH_{EXEC,VCC}{Z,NZ}`
    /// tests the complete architectural 64-bit mask. Leaving the opposite word at zero lets the
    /// two halves take different structured-CFG edges. Keep this compute-only: mesh shaders can
    /// already consume the full 32 KiB Metal threadgroup-memory budget, and their control-flow
    /// synchronization needs a storage-free lowering.
    fn needs_split_wave64_branch_mask_exchange(&self) -> bool {
        self.options.stage == MetalShaderStage::Compute
            && self.split_wave64()
            && self.program.instructions.iter().any(|instruction| {
                matches!(
                    instruction.opcode.known_name(),
                    Some("SCbranchVccz" | "SCbranchVccnz" | "SCbranchExecz" | "SCbranchExecnz")
                )
            })
    }

    /// Whether split host SIMD32 groups need pair-scoped storage for guest-wave64 rendezvous.
    fn needs_split_wave64_pair_rendezvous(&self) -> bool {
        self.needs_split_wave64_branch_mask_exchange()
            || (self.options.stage == MetalShaderStage::Compute
                && self.split_wave64()
                && !self.split_wave64_is_whole_compute_threadgroup()
                && self.uses_lds_memory()
                && self.program.instructions.iter().any(is_full_exec_restore))
    }

    /// Whether explicit VALU comparison destinations are architectural SGPR pairs.
    ///
    /// Pixel shaders have no threadgroup in which Metal can execute both halves of a guest
    /// wave64, so they deliberately use the local SIMD32 mask in the low word.  The high word
    /// still has to be cleared, however: wave64 shaders immediately consume these comparison
    /// destinations with B64 scalar operations.  Leaving it untouched aliases unrelated SGPR
    /// state (commonly the high words of image descriptors) into EXEC/SCC control flow.
    fn uses_wave64_lane_mask_pairs(&self) -> bool {
        self.split_wave64()
            || self.program.instructions.iter().any(|instruction| {
                instruction
                    .opcode
                    .known_name()
                    .is_some_and(|name| name.ends_with("SaveexecB64"))
            })
    }

    fn control_flow_may_revisit_instruction_sites(&self) -> bool {
        match &self.control_flow {
            ControlFlowLowering::Dispatcher => true,
            ControlFlowLowering::Structured { tree, .. } => structured_cfg_contains_loop(tree),
            ControlFlowLowering::Direct(_) | ControlFlowLowering::Forward => false,
        }
    }

    /// Upper bound on dispatcher loop iterations before the shader aborts.
    ///
    /// Poppy's Slate glyph PS is ~2.7k guest instructions with dozens of relative-move
    /// serialization loops. A fixed 16k budget is only a few single walks of that program and
    /// can `active = false` mid-SDF, leaving the default opaque clear color (solid tofu quads).
    fn dispatcher_step_limit(&self) -> u32 {
        let instructions = u32::try_from(self.program.instructions.len()).unwrap_or(u32::MAX);
        instructions.saturating_mul(32).max(16_384)
    }

    fn readfirstlane_site_count(&self) -> usize {
        self.program
            .instructions
            .iter()
            .filter(|instruction| instruction.opcode.is("VReadfirstlaneB32"))
            .count()
    }

    /// Threadgroup storage emitted independently of the per-site `V_READFIRSTLANE` records.
    ///
    /// Metal applies one memory limit to guest LDS and every compiler-generated helper array.
    /// Keeping this accounting beside the declarations prevents a valid 32-KiB guest allocation
    /// from being combined with layer or split-wave state into an invalid pipeline.
    fn threadgroup_memory_base_bytes(&self) -> Option<usize> {
        let guest_lds_bytes = if self.uses_lds_memory() {
            usize::try_from(self.options.lds_size_bytes).ok()?
        } else {
            0
        };
        let mesh_layer_bytes = if self.options.stage == MetalShaderStage::Mesh
            && self.options.render_target_array_length > 1
        {
            usize::try_from(self.options.mesh_max_vertex_count)
                .ok()?
                .checked_add(usize::try_from(self.options.mesh_max_primitive_count).ok()?)?
                .checked_mul(size_of::<u32>())?
        } else {
            0
        };
        let split_pair_bytes = if self.needs_split_wave64_pair_rendezvous() {
            SPLIT_WAVE64_PAIR_STATE_BYTES
        } else {
            0
        };
        let readlane_bytes = if self.split_wave64()
            && self
                .program
                .instructions
                .iter()
                .any(|instruction| instruction.opcode.is("VReadlaneB32"))
        {
            MAX_METAL_THREADGROUP_THREADS * size_of::<u32>()
        } else {
            0
        };
        guest_lds_bytes
            .checked_add(mesh_layer_bytes)?
            .checked_add(split_pair_bytes)?
            .checked_add(readlane_bytes)
    }

    /// Returns the number of half-wave exchange records when every static `V_READFIRSTLANE` owns
    /// disjoint storage.
    ///
    /// Direct, forward-only, and structured-If control flow is acyclic, so an invocation can
    /// execute each static instruction at most once. Giving every site its own records removes the
    /// second whole-group barrier which otherwise protects a reused record from the next
    /// instruction. Dispatcher shaders and structured loops can revisit a site and must retain the
    /// reused-record protocol.
    fn disjoint_readfirstlane_slot_count(&self) -> Option<usize> {
        if !self.split_wave64() || self.control_flow_may_revisit_instruction_sites() {
            return None;
        }
        let site_count = self.readfirstlane_site_count();
        let slot_count = site_count.checked_mul(SPLIT_WAVE64_HALF_WAVES_PER_THREADGROUP)?;
        let readfirstlane_bytes = slot_count.checked_mul(2 * size_of::<u32>())?;
        let total_bytes = self
            .threadgroup_memory_base_bytes()?
            .checked_add(readfirstlane_bytes)?;
        (site_count != 0 && total_bytes <= METAL_THREADGROUP_MEMORY_LIMIT_BYTES)
            .then_some(slot_count)
    }

    fn total_threadgroup_memory_bytes(&self) -> Option<usize> {
        let mut total = self.threadgroup_memory_base_bytes()?;
        if self.split_wave64() && self.readfirstlane_site_count() != 0 {
            let slots = self
                .disjoint_readfirstlane_slot_count()
                // The cyclic/memory-limited fallback declaration reserves one record for every
                // possible SIMD32 half in a 1,024-thread Metal group, not merely one guest-wave
                // pair. Keep validation identical to the emitted arrays.
                .unwrap_or(MAX_METAL_THREADGROUP_THREADS / 32);
            total = total.checked_add(slots.checked_mul(2 * size_of::<u32>())?)?;
        }
        Some(total)
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

    /// VGPRs that Poppy's glyph PS (and similar) fill via `V_WRITELANE` then reload with
    /// `V_READLANE` as a wave-wide constant table. Metal fragment/vertex simdgroups do not
    /// guarantee those architectural lanes are occupied, so `simd_shuffle` cannot emulate the
    /// table — each invocation keeps a private copy instead.
    fn graphics_lane_file_vgprs(&self) -> BTreeSet<u32> {
        if self.is_threadgroup_stage() {
            return BTreeSet::new();
        }
        self.program
            .instructions
            .iter()
            .filter(|instruction| instruction.opcode.is("VWritelaneB32"))
            .filter_map(|instruction| {
                instruction.destinations.first().and_then(|destination| {
                    (destination.kind == OperandKind::VectorRegister).then_some(destination.value)
                })
            })
            .collect()
    }

    fn graphics_lane_file_vgpr(&self, vgpr: u32) -> bool {
        self.graphics_lane_file_vgprs().contains(&vgpr)
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
        let ballot = self.ballot(condition);
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

fn is_full_exec_restore(instruction: &Instruction) -> bool {
    instruction.opcode.is("SMovB64")
        && instruction.destinations.first().is_some_and(|destination| {
            destination.kind == OperandKind::ScalarRegister && destination.value == 126
        })
        && instruction.sources.first().is_some_and(|source| {
            source.kind == OperandKind::EncodedConstant && source.value == 193
        })
}

fn structured_cfg_contains_loop(tree: &StructuredCfg) -> bool {
    match tree {
        StructuredCfg::Loop(_) => true,
        StructuredCfg::Sequence(nodes) => nodes.iter().any(structured_cfg_contains_loop),
        StructuredCfg::If {
            then_branch,
            else_branch,
            ..
        } => {
            structured_cfg_contains_loop(then_branch)
                || else_branch
                    .as_ref()
                    .is_some_and(|branch| structured_cfg_contains_loop(branch))
        }
        StructuredCfg::Block(_)
        | StructuredCfg::Continue
        | StructuredCfg::Break
        | StructuredCfg::Dispatcher { .. } => false,
    }
}
