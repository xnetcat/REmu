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
    readfirstlane_site_bases: BTreeMap<u32, usize>,
    disjoint_readfirstlane_slot_count: Option<usize>,
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

    /// Plans half-wave exchange records when every static `V_READFIRSTLANE` can own disjoint
    /// storage.
    ///
    /// Direct and forward-only control flow is acyclic, so an invocation can execute each static
    /// instruction at most once. Giving every site its own records removes the second whole-group
    /// barrier which otherwise protects a reused record from the next instruction. Dispatcher
    /// shaders can revisit a site and must retain the reused-record protocol.
    fn plan_disjoint_readfirstlane_layout(
        program: &Program,
        options: &MetalCompileOptions,
        control_flow: &ControlFlowLowering,
    ) -> (BTreeMap<u32, usize>, Option<usize>) {
        if !matches!(
            options.stage,
            MetalShaderStage::Compute | MetalShaderStage::Mesh
        ) || options.wave_lane_count != 64
            || matches!(control_flow, ControlFlowLowering::Dispatcher)
        {
            return (BTreeMap::new(), None);
        }
        let mut site_pcs = Vec::new();
        let mut uses_readlane = false;
        for instruction in &program.instructions {
            if instruction.opcode.is("VReadfirstlaneB32") {
                site_pcs.push(instruction.pc);
            } else if instruction.opcode.is("VReadlaneB32") {
                uses_readlane = true;
            }
        }
        let site_count = site_pcs.len();
        let Some(slot_count) = site_count.checked_mul(SPLIT_WAVE64_HALF_WAVES_PER_THREADGROUP)
        else {
            return (BTreeMap::new(), None);
        };
        let Some(readfirstlane_bytes) = slot_count.checked_mul(2 * size_of::<u32>()) else {
            return (BTreeMap::new(), None);
        };
        let readlane_bytes = if uses_readlane {
            MAX_METAL_THREADGROUP_THREADS * size_of::<u32>()
        } else {
            0
        };
        let mesh_layer_bytes =
            if options.stage == MetalShaderStage::Mesh && options.render_target_array_length > 1 {
                64 * size_of::<u32>()
            } else {
                0
            };
        let Some(total_bytes) = usize::try_from(options.lds_size_bytes)
            .ok()
            .and_then(|bytes| bytes.checked_add(readfirstlane_bytes))
            .and_then(|bytes| bytes.checked_add(readlane_bytes))
            .and_then(|bytes| bytes.checked_add(mesh_layer_bytes))
        else {
            return (BTreeMap::new(), None);
        };
        if site_count == 0 || total_bytes > METAL_THREADGROUP_MEMORY_LIMIT_BYTES {
            return (BTreeMap::new(), None);
        }
        let site_bases = site_pcs
            .into_iter()
            .enumerate()
            .map(|(site, pc)| {
                (
                    pc,
                    site.checked_mul(SPLIT_WAVE64_HALF_WAVES_PER_THREADGROUP)
                        .expect("the checked total slot count bounds every site base"),
                )
            })
            .collect();
        (site_bases, Some(slot_count))
    }

    fn disjoint_readfirstlane_slot_count(&self) -> Option<usize> {
        self.disjoint_readfirstlane_slot_count
    }

    fn disjoint_readfirstlane_site_base(&self, pc: u32) -> Option<usize> {
        self.disjoint_readfirstlane_slot_count()?;
        self.readfirstlane_site_bases.get(&pc).copied()
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
