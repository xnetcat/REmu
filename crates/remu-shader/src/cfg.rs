// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

//! Structured control-flow reconstruction for a decoded shader program (workstream Q3).
//!
//! A translated shader has to run structured control flow (MSL, like every high-level shading
//! language, has no `goto`). The scalar evaluator explores paths under a global step budget; a
//! genuine backend instead needs the program's control-flow *shape*: which conditionals reconverge
//! into if/else, which back edges form loops, and — when the graph is irreducible — an explicit
//! dispatcher state machine. Producing that shape here, from the backend-neutral IR, is what lets
//! the backend stop truncating at a step cap and lower every reachable block exactly once.
//!
//! The reconstruction is validated by a differential harness (see the test module): the structured
//! tree is executed under an arbitrary branch oracle and its block-visit trace is compared against
//! the same oracle driving a direct walk of the control-flow graph. Reducible graphs reduce to
//! [`StructuredCfg::If`]/[`StructuredCfg::Loop`]; anything the strict structurer cannot prove SESE
//! (single entry, single exit) — including every irreducible graph — lowers to a
//! [`StructuredCfg::Dispatcher`], which is a walk of the raw graph and therefore always faithful.

use std::collections::{BTreeMap, BTreeSet};

use crate::ir::{Opcode, Program, ShaderEncoding};

/// Index of a basic block within a [`ControlFlowGraph`].
pub type BlockId = usize;

/// How a basic block hands control to its successors.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BlockExit {
    /// Falls straight through to the one successor (or ends the program when it has none).
    Fallthrough,
    /// Ends the program: `S_ENDPGM`, or an indirect `S_SETPC_B64` with no resolved continuation.
    Terminal,
    /// An unconditional `S_BRANCH`/linked `S_SETPC_B64` to the single successor.
    Unconditional,
    /// A scalar conditional branch. `successors[0]` is the fall-through (not-taken) block and
    /// `successors[1]` is the taken branch target.
    Conditional,
}

/// A maximal straight-line run of instructions with a single entry and a single exit transfer.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BasicBlock {
    /// Guest program counter of the block's first instruction.
    pub start_pc: u32,
    /// Half-open range of instruction indices `[first, end)` within [`Program::instructions`].
    pub instructions: std::ops::Range<usize>,
    /// The exit classification of the block's last instruction.
    pub exit: BlockExit,
    /// Successor block ids, ordered per [`BlockExit`].
    pub successors: Vec<BlockId>,
}

/// The control-flow graph of a decoded program: basic blocks plus their successor edges.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ControlFlowGraph {
    blocks: Vec<BasicBlock>,
    predecessors: Vec<Vec<BlockId>>,
}

impl ControlFlowGraph {
    /// The basic blocks, in ascending start-pc order. Block `0` is the entry.
    #[must_use]
    pub fn blocks(&self) -> &[BasicBlock] {
        &self.blocks
    }

    /// Predecessor block ids of one block.
    #[must_use]
    pub fn predecessors(&self, block: BlockId) -> &[BlockId] {
        self.predecessors.get(block).map_or(&[], Vec::as_slice)
    }

    /// The entry block id (always `0` for a non-empty program).
    #[must_use]
    pub const fn entry(&self) -> BlockId {
        0
    }
}

/// The pc a SOPP branch (conditional or `S_BRANCH`) transfers to, from its signed 16-bit
/// displacement resolved against the instruction following the branch. `None` for non-branches.
fn branch_target_pc(program: &Program, index: usize) -> Option<u32> {
    let instruction = program.instructions.get(index)?;
    let is_branch = instruction.encoding == ShaderEncoding::Sopp
        && (is_conditional(instruction.opcode) || instruction.opcode.is("SBranch"));
    if !is_branch {
        return None;
    }
    let raw = u16::try_from(instruction.words.first().copied()? & 0xffff).ok()?;
    let offset = i32::from(raw.cast_signed());
    let next_pc = i64::from(instruction.pc)
        + i64::try_from(instruction.words.len().saturating_mul(4)).ok()?;
    let target = next_pc + i64::from(offset) * 4;
    u32::try_from(target).ok()
}

fn is_conditional(opcode: Opcode) -> bool {
    matches!(
        opcode.known_name(),
        Some(
            "SCbranchScc0"
                | "SCbranchScc1"
                | "SCbranchVccz"
                | "SCbranchVccnz"
                | "SCbranchExecz"
                | "SCbranchExecnz"
        )
    )
}

fn is_terminal(opcode: Opcode) -> bool {
    opcode == Opcode::END_PROGRAM
}

/// Builds the control-flow graph of a decoded program.
///
/// Leaders are the entry, every branch target, and the instruction after every branch or terminal
/// transfer. A linked `S_SETPC_B64` continuation (see [`crate::link_program_continuation`]) is
/// followed to its registered target; an unlinked indirect transfer is treated as a program exit,
/// exactly where the evaluator's step-cap truncation used to abandon the shader.
#[must_use]
pub fn build_control_flow_graph(program: &Program) -> ControlFlowGraph {
    let instructions = &program.instructions;
    if instructions.is_empty() {
        return ControlFlowGraph {
            blocks: Vec::new(),
            predecessors: Vec::new(),
        };
    }

    // Map each instruction pc to its index for target resolution.
    let pc_to_index: BTreeMap<u32, usize> = instructions
        .iter()
        .enumerate()
        .map(|(index, instruction)| (instruction.pc, index))
        .collect();

    // Collect leader instruction indices.
    let mut leaders: BTreeSet<usize> = BTreeSet::new();
    leaders.insert(0);
    for (index, instruction) in instructions.iter().enumerate() {
        let is_branch = instruction.encoding == ShaderEncoding::Sopp
            && (is_conditional(instruction.opcode) || instruction.opcode.is("SBranch"));
        let is_linked = instruction.opcode.is("SSetpcB64")
            && program.linked_transfer_target(instruction.pc).is_some();
        let is_exit = is_terminal(instruction.opcode) || instruction.opcode.is("SSetpcB64");
        if is_branch {
            if let Some(target_pc) = branch_target_pc(program, index)
                && let Some(&target_index) = pc_to_index.get(&target_pc) {
                    leaders.insert(target_index);
                }
            if index + 1 < instructions.len() {
                leaders.insert(index + 1);
            }
        } else if is_linked {
            if let Some(target_pc) = program.linked_transfer_target(instruction.pc)
                && let Some(&target_index) = pc_to_index.get(&target_pc) {
                    leaders.insert(target_index);
                }
            if index + 1 < instructions.len() {
                leaders.insert(index + 1);
            }
        } else if is_exit && index + 1 < instructions.len() {
            leaders.insert(index + 1);
        }
    }

    let leader_list: Vec<usize> = leaders.iter().copied().collect();
    let index_to_block: BTreeMap<usize, BlockId> = leader_list
        .iter()
        .enumerate()
        .map(|(block, &first)| (first, block))
        .collect();

    let mut blocks: Vec<BasicBlock> = Vec::with_capacity(leader_list.len());
    for (block, &first) in leader_list.iter().enumerate() {
        let end = leader_list
            .get(block + 1)
            .copied()
            .unwrap_or(instructions.len());
        let last = end - 1;
        let last_instruction = &instructions[last];
        let (exit, successors) = classify_exit(
            program,
            last,
            last_instruction,
            &pc_to_index,
            &index_to_block,
            end,
            instructions.len(),
        );
        blocks.push(BasicBlock {
            start_pc: instructions[first].pc,
            instructions: first..end,
            exit,
            successors,
        });
    }

    let mut predecessors = vec![Vec::new(); blocks.len()];
    for (block, data) in blocks.iter().enumerate() {
        for &successor in &data.successors {
            if !predecessors[successor].contains(&block) {
                predecessors[successor].push(block);
            }
        }
    }

    ControlFlowGraph {
        blocks,
        predecessors,
    }
}

fn classify_exit(
    program: &Program,
    last_index: usize,
    last_instruction: &crate::ir::Instruction,
    pc_to_index: &BTreeMap<u32, usize>,
    index_to_block: &BTreeMap<usize, BlockId>,
    fallthrough_index: usize,
    instruction_count: usize,
) -> (BlockExit, Vec<BlockId>) {
    let block_of = |index: usize| index_to_block.get(&index).copied();
    let fallthrough = (fallthrough_index < instruction_count)
        .then(|| block_of(fallthrough_index))
        .flatten();
    let opcode = last_instruction.opcode;
    if is_terminal(opcode) {
        return (BlockExit::Terminal, Vec::new());
    }
    if opcode.is("SSetpcB64") {
        return match program.linked_transfer_target(last_instruction.pc) {
            Some(target_pc) => match pc_to_index.get(&target_pc).and_then(|&i| block_of(i)) {
                Some(target) => (BlockExit::Unconditional, vec![target]),
                None => (BlockExit::Terminal, Vec::new()),
            },
            // An unlinked indirect transfer statically ends the decodable stream.
            None => (BlockExit::Terminal, Vec::new()),
        };
    }
    if last_instruction.encoding == ShaderEncoding::Sopp && opcode.is("SBranch") {
        return match branch_target_pc(program, last_index)
            .and_then(|pc| pc_to_index.get(&pc).copied())
            .and_then(block_of)
        {
            Some(target) => (BlockExit::Unconditional, vec![target]),
            None => (BlockExit::Terminal, Vec::new()),
        };
    }
    if last_instruction.encoding == ShaderEncoding::Sopp && is_conditional(opcode) {
        let target = branch_target_pc(program, last_index)
            .and_then(|pc| pc_to_index.get(&pc).copied())
            .and_then(block_of);
        return match (fallthrough, target) {
            (Some(fall), Some(target)) => (BlockExit::Conditional, vec![fall, target]),
            // A conditional whose only reachable edge is statically known collapses to that edge.
            (Some(fall), None) => (BlockExit::Unconditional, vec![fall]),
            (None, Some(target)) => (BlockExit::Unconditional, vec![target]),
            (None, None) => (BlockExit::Terminal, Vec::new()),
        };
    }
    // A straight-line block: fall through to the next block, or end the program.
    match fallthrough {
        Some(next) => (BlockExit::Fallthrough, vec![next]),
        None => (BlockExit::Terminal, Vec::new()),
    }
}

// ------------------------------------------------------------------------------------------------
// Dominators, post-dominators, natural loops and reducibility.
// ------------------------------------------------------------------------------------------------

/// Immediate-dominator tree of a control-flow graph (Cooper–Harvey–Kennedy).
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Dominators {
    /// `idom[b]` is the immediate dominator of block `b`; the entry dominates itself.
    idom: Vec<BlockId>,
    /// Reverse-postorder rank of each block; `usize::MAX` for unreachable blocks.
    position: Vec<usize>,
}

impl Dominators {
    /// Whether `dominator` dominates `block` (every path from entry to `block` passes through it).
    #[must_use]
    pub fn dominates(&self, dominator: BlockId, block: BlockId) -> bool {
        let mut current = block;
        loop {
            if current == dominator {
                return true;
            }
            let next = self.idom[current];
            if next == current {
                return dominator == current;
            }
            current = next;
        }
    }

    /// The immediate dominator of `block` (the entry's is itself).
    #[must_use]
    pub fn immediate_dominator(&self, block: BlockId) -> BlockId {
        self.idom[block]
    }
}

fn compute_dominators(successors: &[Vec<BlockId>], predecessors: &[Vec<BlockId>]) -> Dominators {
    let count = successors.len();
    let order = reverse_postorder_from(successors, 0);
    let mut position = vec![usize::MAX; count];
    for (rank, &block) in order.iter().enumerate() {
        position[block] = rank;
    }
    let mut idom = vec![usize::MAX; count];
    idom[0] = 0;
    let intersect = |mut a: BlockId, mut b: BlockId, idom: &[BlockId], position: &[usize]| {
        while a != b {
            while position[a] > position[b] {
                a = idom[a];
            }
            while position[b] > position[a] {
                b = idom[b];
            }
        }
        a
    };
    let mut changed = true;
    while changed {
        changed = false;
        for &block in order.iter().skip(1) {
            let mut new_idom = usize::MAX;
            for &predecessor in &predecessors[block] {
                if idom[predecessor] == usize::MAX {
                    continue;
                }
                new_idom = if new_idom == usize::MAX {
                    predecessor
                } else {
                    intersect(predecessor, new_idom, &idom, &position)
                };
            }
            if new_idom != usize::MAX && idom[block] != new_idom {
                idom[block] = new_idom;
                changed = true;
            }
        }
    }
    // Unreachable blocks dominate only themselves (they never appear in a reachable path).
    for (block, idom_entry) in idom.iter_mut().enumerate().take(count) {
        if *idom_entry == usize::MAX {
            *idom_entry = block;
        }
    }
    Dominators { idom, position }
}

impl ControlFlowGraph {
    /// The immediate-dominator tree.
    #[must_use]
    pub fn dominators(&self) -> Dominators {
        let successors: Vec<Vec<BlockId>> =
            self.blocks.iter().map(|b| b.successors.clone()).collect();
        compute_dominators(&successors, &self.predecessors)
    }

    /// The immediate-post-dominator tree, computed as the dominators of the reversed graph rooted
    /// at a virtual exit that sinks every terminal block. Returns `idom_post[b]`, with unreachable
    /// entries pointing at themselves.
    #[must_use]
    pub fn post_dominators(&self) -> PostDominators {
        let count = self.blocks.len();
        // Reverse edges; add a virtual exit node `count` that all terminal blocks point to (in the
        // reversed graph the virtual exit is the entry).
        let virtual_exit = count;
        let mut reverse_successors = vec![Vec::new(); count + 1];
        let mut reverse_predecessors = vec![Vec::new(); count + 1];
        for (block, data) in self.blocks.iter().enumerate() {
            if data.successors.is_empty() {
                reverse_successors[virtual_exit].push(block);
                reverse_predecessors[block].push(virtual_exit);
            }
            for &successor in &data.successors {
                reverse_successors[successor].push(block);
                reverse_predecessors[block].push(successor);
            }
        }
        // Root the reversed-graph dominators at the virtual exit by swapping it into slot 0's role;
        // compute_dominators roots at index 0, so re-index with the virtual exit as the root.
        let rpo = reverse_postorder_from(&reverse_successors, virtual_exit);
        let mut position = vec![usize::MAX; count + 1];
        for (rank, &block) in rpo.iter().enumerate() {
            position[block] = rank;
        }
        let mut idom = vec![usize::MAX; count + 1];
        idom[virtual_exit] = virtual_exit;
        let intersect = |mut a: BlockId, mut b: BlockId, idom: &[BlockId], position: &[usize]| {
            while a != b {
                while position[a] > position[b] {
                    a = idom[a];
                }
                while position[b] > position[a] {
                    b = idom[b];
                }
            }
            a
        };
        let mut changed = true;
        while changed {
            changed = false;
            for &block in rpo.iter() {
                if block == virtual_exit {
                    continue;
                }
                let mut new_idom = usize::MAX;
                for &predecessor in &reverse_predecessors[block] {
                    if idom[predecessor] == usize::MAX {
                        continue;
                    }
                    new_idom = if new_idom == usize::MAX {
                        predecessor
                    } else {
                        intersect(predecessor, new_idom, &idom, &position)
                    };
                }
                if new_idom != usize::MAX && idom[block] != new_idom {
                    idom[block] = new_idom;
                    changed = true;
                }
            }
        }
        for entry in idom.iter_mut().take(count) {
            if *entry == usize::MAX {
                *entry = virtual_exit;
            }
        }
        PostDominators {
            ipdom: idom,
            virtual_exit,
        }
    }

    /// Every natural loop, keyed by header block. A natural loop exists for each back edge
    /// `latch -> header` where the header dominates the latch; the body is every block that can
    /// reach the latch without leaving the header.
    #[must_use]
    pub fn natural_loops(&self) -> BTreeMap<BlockId, NaturalLoop> {
        let dominators = self.dominators();
        let mut loops: BTreeMap<BlockId, NaturalLoop> = BTreeMap::new();
        for (block, data) in self.blocks.iter().enumerate() {
            for &successor in &data.successors {
                if dominators.dominates(successor, block) {
                    // `block -> successor` is a back edge; `successor` is the header.
                    let entry = loops.entry(successor).or_insert_with(|| NaturalLoop {
                        header: successor,
                        latches: Vec::new(),
                        body: BTreeSet::from([successor]),
                    });
                    entry.latches.push(block);
                    collect_loop_body(self, successor, block, &mut entry.body);
                }
            }
        }
        loops
    }

    /// Whether the graph is reducible: every retreating edge (target precedes source in reverse
    /// postorder) is a back edge whose target dominates its source. Irreducible graphs have a
    /// multi-entry cycle and cannot be expressed with structured `if`/`loop` alone.
    #[must_use]
    pub fn is_reducible(&self) -> bool {
        let dominators = self.dominators();
        for (block, data) in self.blocks.iter().enumerate() {
            if dominators.position[block] == usize::MAX {
                continue; // unreachable
            }
            for &successor in &data.successors {
                if dominators.position[successor] == usize::MAX {
                    continue;
                }
                // A retreating edge points to an equal-or-earlier reverse-postorder rank.
                if dominators.position[successor] <= dominators.position[block]
                    && !dominators.dominates(successor, block)
                {
                    return false;
                }
            }
        }
        true
    }
}

fn reverse_postorder_from(successors: &[Vec<BlockId>], entry: BlockId) -> Vec<BlockId> {
    let mut visited = vec![false; successors.len()];
    let mut postorder = Vec::with_capacity(successors.len());
    let mut stack: Vec<(BlockId, usize)> = vec![(entry, 0)];
    visited[entry] = true;
    while let Some(&(block, next)) = stack.last() {
        if next < successors[block].len() {
            stack.last_mut().unwrap().1 += 1;
            let successor = successors[block][next];
            if !visited[successor] {
                visited[successor] = true;
                stack.push((successor, 0));
            }
        } else {
            postorder.push(block);
            stack.pop();
        }
    }
    postorder.reverse();
    postorder
}

fn collect_loop_body(
    cfg: &ControlFlowGraph,
    header: BlockId,
    latch: BlockId,
    body: &mut BTreeSet<BlockId>,
) {
    let mut stack = vec![latch];
    body.insert(latch);
    while let Some(block) = stack.pop() {
        if block == header {
            continue;
        }
        for &predecessor in cfg.predecessors(block) {
            if body.insert(predecessor) {
                stack.push(predecessor);
            }
        }
    }
}

/// A natural loop: its header, the latch blocks whose back edges close it, and its body blocks.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NaturalLoop {
    pub header: BlockId,
    pub latches: Vec<BlockId>,
    pub body: BTreeSet<BlockId>,
}

/// Immediate-post-dominator tree.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PostDominators {
    ipdom: Vec<BlockId>,
    virtual_exit: BlockId,
}

impl PostDominators {
    /// The immediate post-dominator of `block`, or `None` when it post-dominates to the virtual
    /// program exit (no in-graph reconvergence).
    #[must_use]
    pub fn immediate_post_dominator(&self, block: BlockId) -> Option<BlockId> {
        let ipdom = self.ipdom.get(block).copied()?;
        (ipdom != self.virtual_exit && ipdom != block).then_some(ipdom)
    }
}

// ------------------------------------------------------------------------------------------------
// Structured reconstruction.
// ------------------------------------------------------------------------------------------------

/// A structured control-flow tree. `Dispatcher` is the always-correct fallback: an explicit walk
/// of a region's raw graph, used for irreducible graphs and any region the strict structurer
/// cannot prove single-entry/single-exit.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum StructuredCfg {
    /// A straight-line basic block. Its structural successor is supplied by the enclosing node.
    Block(BlockId),
    /// Run each child in order.
    Sequence(Vec<StructuredCfg>),
    /// An if-diamond. `condition` is executed, then the taken subtree runs. `then_is_taken_edge`
    /// records which successor index (`0` fall-through, `1` target) selects `then_branch`.
    If {
        condition: BlockId,
        then_is_taken_edge: usize,
        then_branch: Box<StructuredCfg>,
        else_branch: Option<Box<StructuredCfg>>,
    },
    /// A natural loop: run `body` until a `Break` fires; `Continue` restarts it.
    Loop(Box<StructuredCfg>),
    /// A back edge to the enclosing loop header.
    Continue,
    /// An exit edge from the enclosing loop.
    Break,
    /// A dispatcher state machine over `blocks`, entered at `entry`. Faithful for any region.
    Dispatcher {
        entry: BlockId,
        blocks: Vec<BlockId>,
    },
}

impl ControlFlowGraph {
    /// Reconstructs a structured control-flow tree.
    ///
    /// Reducible graphs whose conditionals reconverge cleanly reduce to [`StructuredCfg::If`] and
    /// [`StructuredCfg::Loop`]; every irreducible graph, and any region the strict structurer
    /// cannot prove SESE, lowers to a faithful [`StructuredCfg::Dispatcher`]. In all cases every
    /// reachable block is covered exactly once, which is what removes the step-cap truncation.
    ///
    /// Correctness is enforced structurally: the strict structurer emits `If`/`Loop` only for the
    /// canonical reducible shapes, and the result is accepted only if it covers every reachable
    /// block exactly once. Any deviation — nested or multi-exit loops, non-reconverging
    /// conditionals, irreducibility — falls back to the dispatcher, which covers the reachable set
    /// once by construction.
    #[must_use]
    pub fn structure(&self) -> StructuredCfg {
        if self.blocks.is_empty() {
            return StructuredCfg::Sequence(Vec::new());
        }
        let reachable = self.reachable_blocks();
        let loops = self.natural_loops();
        if !self.is_reducible() {
            return self.dispatcher_over(&reachable);
        }
        let mut structurer = Structurer {
            cfg: self,
            post_dominators: self.post_dominators(),
            loops,
            bailed: false,
        };
        let structured = structurer.region(self.entry(), None, None, false);
        if structurer.bailed || !covers_each_reachable_block_once(&structured, &reachable) {
            self.dispatcher_over(&reachable)
        } else {
            structured
        }
    }

    fn reachable_blocks(&self) -> BTreeSet<BlockId> {
        let mut reachable = BTreeSet::new();
        let mut stack = vec![self.entry()];
        reachable.insert(self.entry());
        while let Some(block) = stack.pop() {
            for &successor in &self.blocks[block].successors {
                if reachable.insert(successor) {
                    stack.push(successor);
                }
            }
        }
        reachable
    }

    fn dispatcher_over(&self, reachable: &BTreeSet<BlockId>) -> StructuredCfg {
        StructuredCfg::Dispatcher {
            entry: self.entry(),
            blocks: reachable.iter().copied().collect(),
        }
    }
}

struct Structurer<'a> {
    cfg: &'a ControlFlowGraph,
    post_dominators: PostDominators,
    loops: BTreeMap<BlockId, NaturalLoop>,
    bailed: bool,
}

impl Structurer<'_> {
    /// Structures the region entered at `entry`, stopping when it reaches `stop` (the follow block
    /// owned by the enclosing region). Within a loop, an edge to the header becomes `Continue` and
    /// an edge to the loop exit becomes `Break`.
    ///
    /// `entering_header` is `true` only for the first block of a loop body root: it is what
    /// distinguishes emitting the header block (loop entry) from a back edge to it. Every other
    /// reference to the header — from a sub-region or a later block — is a `Continue`.
    fn region(
        &mut self,
        entry: BlockId,
        stop: Option<BlockId>,
        loop_ctx: Option<(BlockId, Option<BlockId>)>,
        entering_header: bool,
    ) -> StructuredCfg {
        let mut sequence = Vec::new();
        let mut cursor = Some(entry);
        let mut entering = entering_header;
        let mut guard = 0usize;
        while let Some(current) = cursor {
            guard += 1;
            if guard > self.cfg.blocks.len() + 2 {
                self.bailed = true;
                break;
            }
            if Some(current) == stop {
                break;
            }
            if let Some((header, exit)) = loop_ctx {
                if current == header && !entering {
                    sequence.push(StructuredCfg::Continue);
                    return StructuredCfg::Sequence(sequence);
                }
                if Some(current) == exit {
                    sequence.push(StructuredCfg::Break);
                    return StructuredCfg::Sequence(sequence);
                }
            }

            // A loop header other than the one this region roots opens its own Loop node here,
            // which is how cleanly nested natural loops structure recursively. An edge from an
            // inner body to an outer header or exit (a multi-level continue/break) re-walks
            // already-covered blocks instead, and the covers-each-reachable-block-once acceptance
            // check in `structure` demotes that tree to the faithful dispatcher.
            if self.loops.contains_key(&current)
                && loop_ctx.map(|(header, _)| header) != Some(current)
            {
                match self.structure_loop(current) {
                    Some((node, exit)) => {
                        sequence.push(node);
                        cursor = exit;
                        entering = false;
                        continue;
                    }
                    None => {
                        self.bailed = true;
                        break;
                    }
                }
            }

            let block = &self.cfg.blocks[current];
            match block.exit {
                BlockExit::Terminal => {
                    sequence.push(StructuredCfg::Block(current));
                    cursor = None;
                }
                BlockExit::Fallthrough | BlockExit::Unconditional => {
                    sequence.push(StructuredCfg::Block(current));
                    cursor = block.successors.first().copied();
                }
                BlockExit::Conditional => {
                    let follow = self.post_dominators.immediate_post_dominator(current);
                    let [fall, target] = [block.successors[0], block.successors[1]];
                    let then_stop = follow.or(stop);
                    let then_branch = Box::new(self.region(fall, then_stop, loop_ctx, false));
                    let else_branch = if Some(target) == then_stop {
                        None
                    } else {
                        Some(Box::new(self.region(target, then_stop, loop_ctx, false)))
                    };
                    sequence.push(StructuredCfg::If {
                        condition: current,
                        then_is_taken_edge: 0,
                        then_branch,
                        else_branch,
                    });
                    cursor = follow;
                }
            }
            entering = false;
        }
        StructuredCfg::Sequence(sequence)
    }

    /// Structures the natural loop headed at `header`, returning the loop node and its exit block.
    /// Bails (returns `None`) for a loop that is not single-exit, which the caller turns into a
    /// whole-function dispatcher.
    fn structure_loop(&mut self, header: BlockId) -> Option<(StructuredCfg, Option<BlockId>)> {
        let natural = self.loops.get(&header)?.clone();
        // A structurable loop has at most one block outside its body reached from within it.
        let mut exits: BTreeSet<BlockId> = BTreeSet::new();
        for &block in &natural.body {
            for &successor in &self.cfg.blocks[block].successors {
                if !natural.body.contains(&successor) {
                    exits.insert(successor);
                }
            }
        }
        if exits.len() > 1 {
            return None;
        }
        let exit = exits.into_iter().next();
        let body = self.region(header, None, Some((header, exit)), true);
        Some((StructuredCfg::Loop(Box::new(body)), exit))
    }
}

/// Whether the structured tree names every reachable block exactly once. A dispatcher covers its
/// whole block set, so this is trivially true for the fallback; for `If`/`Loop` trees it rejects a
/// structuring that dropped or duplicated a block.
fn covers_each_reachable_block_once(
    structured: &StructuredCfg,
    reachable: &BTreeSet<BlockId>,
) -> bool {
    let mut seen: BTreeMap<BlockId, u32> = BTreeMap::new();
    collect_covered_blocks(structured, &mut seen);
    seen.len() == reachable.len()
        && seen.keys().all(|block| reachable.contains(block))
        && seen.values().all(|&count| count == 1)
}

fn collect_covered_blocks(structured: &StructuredCfg, seen: &mut BTreeMap<BlockId, u32>) {
    match structured {
        StructuredCfg::Block(block) => {
            *seen.entry(*block).or_default() += 1;
        }
        StructuredCfg::Sequence(nodes) => {
            for node in nodes {
                collect_covered_blocks(node, seen);
            }
        }
        StructuredCfg::If {
            condition,
            then_branch,
            else_branch,
            ..
        } => {
            *seen.entry(*condition).or_default() += 1;
            collect_covered_blocks(then_branch, seen);
            if let Some(else_branch) = else_branch {
                collect_covered_blocks(else_branch, seen);
            }
        }
        StructuredCfg::Loop(body) => collect_covered_blocks(body, seen),
        StructuredCfg::Continue | StructuredCfg::Break => {}
        StructuredCfg::Dispatcher { blocks, .. } => {
            for &block in blocks {
                *seen.entry(block).or_default() += 1;
            }
        }
    }
}

#[cfg(test)]
mod tests;
