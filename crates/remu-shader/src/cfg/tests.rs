// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

//! Workstream Q3 tests.
//!
//! The differential harness is the load-bearing check: [`assert_structured_matches_graph`] drives
//! an arbitrary branch oracle through both a direct walk of the control-flow graph and an
//! execution of the reconstructed [`StructuredCfg`], and asserts the two produce the identical
//! block-visit trace. A reconstruction that dropped, duplicated or reordered any block — the
//! step-cap truncation class — fails it. The analysis unit tests (dominators, post-dominators,
//! natural loops, reducibility) pin the primitives the reconstruction is built on.

use std::collections::{BTreeMap, BTreeSet};

use super::*;
use crate::decode_program;

/// Builds a control-flow graph directly from `(exit, successors)` block descriptions, bypassing a
/// decoded program so the graph shapes under test are explicit.
fn graph(blocks: &[(BlockExit, &[BlockId])]) -> ControlFlowGraph {
    let blocks: Vec<BasicBlock> = blocks
        .iter()
        .enumerate()
        .map(|(index, (exit, successors))| BasicBlock {
            start_pc: u32::try_from(index * 4).unwrap(),
            instructions: index..index + 1,
            exit: *exit,
            successors: successors.to_vec(),
        })
        .collect();
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

// ------------------------------------------------------------------------------------------------
// Differential harness.
// ------------------------------------------------------------------------------------------------

const STEP_CAP: usize = 100_000;

/// A branch oracle: given a conditional block and how many times it has already been decided on
/// this run, returns the successor index to take.
type Oracle<'a> = &'a mut dyn FnMut(BlockId, usize) -> usize;

/// Walks the raw graph, consulting the oracle at every multi-successor block. This is the reference
/// semantics the structured reconstruction must reproduce.
fn walk_graph(cfg: &ControlFlowGraph, oracle: Oracle) -> Vec<BlockId> {
    let mut trace = Vec::new();
    let mut occurrences: BTreeMap<BlockId, usize> = BTreeMap::new();
    let mut cursor = Some(cfg.entry());
    while let Some(block) = cursor {
        trace.push(block);
        if trace.len() > STEP_CAP {
            break;
        }
        let successors = &cfg.blocks[block].successors;
        cursor = match successors.len() {
            0 => None,
            1 => Some(successors[0]),
            _ => {
                let occurrence = occurrences.entry(block).or_default();
                let index = oracle(block, *occurrence).min(successors.len() - 1);
                *occurrence += 1;
                Some(successors[index])
            }
        };
    }
    trace
}

enum Flow {
    Fall,
    Continue,
    Break,
}

struct StructuredWalk<'a> {
    cfg: &'a ControlFlowGraph,
    oracle: Oracle<'a>,
    occurrences: BTreeMap<BlockId, usize>,
    trace: Vec<BlockId>,
}

impl StructuredWalk<'_> {
    fn visit(&mut self, block: BlockId) {
        self.trace.push(block);
    }

    fn decide(&mut self, block: BlockId) -> usize {
        let occurrence = self.occurrences.entry(block).or_default();
        let successors = self.cfg.blocks[block].successors.len().max(1);
        let index = (self.oracle)(block, *occurrence).min(successors - 1);
        *occurrence += 1;
        index
    }

    fn exec(&mut self, node: &StructuredCfg) -> Flow {
        if self.trace.len() > STEP_CAP {
            return Flow::Fall;
        }
        match node {
            StructuredCfg::Block(block) => {
                self.visit(*block);
                Flow::Fall
            }
            StructuredCfg::Sequence(nodes) => {
                for node in nodes {
                    match self.exec(node) {
                        Flow::Fall => {}
                        other => return other,
                    }
                }
                Flow::Fall
            }
            StructuredCfg::If {
                condition,
                then_is_taken_edge,
                then_branch,
                else_branch,
            } => {
                self.visit(*condition);
                if self.decide(*condition) == *then_is_taken_edge {
                    self.exec(then_branch)
                } else {
                    else_branch
                        .as_ref()
                        .map_or(Flow::Fall, |branch| self.exec(branch))
                }
            }
            StructuredCfg::Loop(body) => {
                loop {
                    if self.trace.len() > STEP_CAP {
                        break;
                    }
                    match self.exec(body) {
                        Flow::Continue => {}
                        Flow::Break | Flow::Fall => break,
                    }
                }
                Flow::Fall
            }
            StructuredCfg::Continue => Flow::Continue,
            StructuredCfg::Break => Flow::Break,
            StructuredCfg::Dispatcher { entry, blocks } => {
                let region: BTreeSet<BlockId> = blocks.iter().copied().collect();
                let mut cursor = Some(*entry);
                while let Some(block) = cursor {
                    if self.trace.len() > STEP_CAP {
                        break;
                    }
                    self.visit(block);
                    let successors = &self.cfg.blocks[block].successors;
                    cursor = match successors.len() {
                        0 => None,
                        1 => Some(successors[0]).filter(|s| region.contains(s)),
                        _ => {
                            let index = self.decide(block);
                            Some(successors[index]).filter(|s| region.contains(s))
                        }
                    };
                }
                Flow::Fall
            }
        }
    }
}

fn walk_structured(
    cfg: &ControlFlowGraph,
    structured: &StructuredCfg,
    oracle: Oracle,
) -> Vec<BlockId> {
    let mut walk = StructuredWalk {
        cfg,
        oracle,
        occurrences: BTreeMap::new(),
        trace: Vec::new(),
    };
    walk.exec(structured);
    walk.trace
}

/// Asserts the structured reconstruction reproduces the raw graph walk for every supplied oracle.
fn assert_structured_matches_graph(
    cfg: &ControlFlowGraph,
    oracles: &[fn(BlockId, usize) -> usize],
) {
    let structured = cfg.structure();
    for (index, base) in oracles.iter().enumerate() {
        let mut raw_oracle = *base;
        let raw = walk_graph(cfg, &mut raw_oracle);
        let mut structured_oracle = *base;
        let structured_trace = walk_structured(cfg, &structured, &mut structured_oracle);
        assert_eq!(
            raw, structured_trace,
            "oracle {index} diverged\nstructured = {structured:?}"
        );
    }
}

fn take_fallthrough(_block: BlockId, _occurrence: usize) -> usize {
    0
}

fn take_branch(_block: BlockId, _occurrence: usize) -> usize {
    1
}

fn alternate(_block: BlockId, occurrence: usize) -> usize {
    occurrence % 2
}

/// Loop back-edge is successor 1; exit on the third decision. Bounds any single loop.
fn loop_three_times(_block: BlockId, occurrence: usize) -> usize {
    usize::from(occurrence < 3)
}

const ALL_ORACLES: &[fn(BlockId, usize) -> usize] =
    &[take_fallthrough, take_branch, alternate, loop_three_times];

// ------------------------------------------------------------------------------------------------
// Analysis primitives.
// ------------------------------------------------------------------------------------------------

#[test]
fn dominators_of_a_diamond_meet_at_the_branch() {
    // 0 -> {1, 2} -> 3
    let cfg = graph(&[
        (BlockExit::Conditional, &[1, 2]),
        (BlockExit::Unconditional, &[3]),
        (BlockExit::Unconditional, &[3]),
        (BlockExit::Terminal, &[]),
    ]);
    let dominators = cfg.dominators();
    assert!(dominators.dominates(0, 3));
    assert!(!dominators.dominates(1, 3));
    assert!(!dominators.dominates(2, 3));
    assert_eq!(dominators.immediate_dominator(3), 0);
    assert_eq!(dominators.immediate_dominator(1), 0);
}

#[test]
fn post_dominators_find_the_reconvergence_of_a_diamond() {
    let cfg = graph(&[
        (BlockExit::Conditional, &[1, 2]),
        (BlockExit::Unconditional, &[3]),
        (BlockExit::Unconditional, &[3]),
        (BlockExit::Terminal, &[]),
    ]);
    let post = cfg.post_dominators();
    assert_eq!(post.immediate_post_dominator(0), Some(3));
    // A branch whose arms exit to distinct terminals has no in-graph reconvergence.
    let forked = graph(&[
        (BlockExit::Conditional, &[1, 2]),
        (BlockExit::Terminal, &[]),
        (BlockExit::Terminal, &[]),
    ]);
    assert_eq!(forked.post_dominators().immediate_post_dominator(0), None);
}

#[test]
fn a_self_loop_and_a_two_block_loop_are_natural_and_reducible() {
    // 0 -> 1 -> {1 (back), 2} ; 2 terminal
    let cfg = graph(&[
        (BlockExit::Unconditional, &[1]),
        (BlockExit::Conditional, &[1, 2]),
        (BlockExit::Terminal, &[]),
    ]);
    let loops = cfg.natural_loops();
    assert_eq!(loops.len(), 1);
    let natural = &loops[&1];
    assert_eq!(natural.header, 1);
    assert_eq!(natural.latches, vec![1]);
    assert_eq!(natural.body, BTreeSet::from([1]));
    assert!(cfg.is_reducible());
}

#[test]
fn a_multi_entry_cycle_is_irreducible() {
    // 0 -> {1, 2}; 1 <-> 2 form a cycle with two entries.
    let cfg = graph(&[
        (BlockExit::Conditional, &[1, 2]),
        (BlockExit::Unconditional, &[2]),
        (BlockExit::Conditional, &[1, 3]),
        (BlockExit::Terminal, &[]),
    ]);
    assert!(!cfg.is_reducible());
    assert!(matches!(cfg.structure(), StructuredCfg::Dispatcher { .. }));
}

// ------------------------------------------------------------------------------------------------
// Structured reconstruction shapes + differential equivalence.
// ------------------------------------------------------------------------------------------------

#[test]
fn straight_line_reconstructs_as_a_block_sequence() {
    let cfg = graph(&[
        (BlockExit::Fallthrough, &[1]),
        (BlockExit::Fallthrough, &[2]),
        (BlockExit::Terminal, &[]),
    ]);
    assert_eq!(
        cfg.structure(),
        StructuredCfg::Sequence(vec![
            StructuredCfg::Block(0),
            StructuredCfg::Block(1),
            StructuredCfg::Block(2),
        ])
    );
    assert_structured_matches_graph(&cfg, ALL_ORACLES);
}

#[test]
fn an_if_diamond_reconstructs_as_an_if_then_else() {
    let cfg = graph(&[
        (BlockExit::Conditional, &[1, 2]),
        (BlockExit::Unconditional, &[3]),
        (BlockExit::Unconditional, &[3]),
        (BlockExit::Terminal, &[]),
    ]);
    let structured = cfg.structure();
    assert!(
        matches!(
            &structured,
            StructuredCfg::Sequence(nodes)
                if matches!(nodes.first(), Some(StructuredCfg::If { condition: 0, else_branch: Some(_), .. }))
        ),
        "expected an if/else diamond, got {structured:?}"
    );
    assert_structured_matches_graph(&cfg, ALL_ORACLES);
}

#[test]
fn an_if_then_without_else_reconstructs_when_one_arm_is_the_merge() {
    // 0 -> {merge(2), 1}; 1 -> 2 ; 2 terminal. The fall-through arm is the merge.
    let cfg = graph(&[
        (BlockExit::Conditional, &[2, 1]),
        (BlockExit::Unconditional, &[2]),
        (BlockExit::Terminal, &[]),
    ]);
    assert_structured_matches_graph(&cfg, ALL_ORACLES);
}

#[test]
fn a_while_loop_reconstructs_as_a_loop_and_matches_the_graph() {
    // 0 header: cond -> {1 body, 2 exit}; 1 -> 0 back; 2 terminal.
    let cfg = graph(&[
        (BlockExit::Conditional, &[1, 2]),
        (BlockExit::Unconditional, &[0]),
        (BlockExit::Terminal, &[]),
    ]);
    assert!(
        matches!(
            &cfg.structure(),
            StructuredCfg::Sequence(nodes) if nodes.iter().any(|n| matches!(n, StructuredCfg::Loop(_)))
        ),
        "expected a loop, got {:?}",
        cfg.structure()
    );
    assert_structured_matches_graph(&cfg, ALL_ORACLES);
}

#[test]
fn a_do_while_loop_with_an_inner_if_matches_the_graph() {
    // 0 header -> 1; 1 cond -> {2 then, 3 else}; 2 -> 4; 3 -> 4; 4 latch cond -> {0 back, 5 exit}.
    let cfg = graph(&[
        (BlockExit::Fallthrough, &[1]),
        (BlockExit::Conditional, &[2, 3]),
        (BlockExit::Unconditional, &[4]),
        (BlockExit::Unconditional, &[4]),
        (BlockExit::Conditional, &[0, 5]),
        (BlockExit::Terminal, &[]),
    ]);
    assert_structured_matches_graph(&cfg, ALL_ORACLES);
}

#[test]
fn a_nested_loop_falls_back_to_a_faithful_dispatcher() {
    // Outer 0..; inner 1<->2; 0 -> 1; 2 -> {1 inner-back, 3}; 3 -> {0 outer-back, 4}; 4 terminal.
    let cfg = graph(&[
        (BlockExit::Unconditional, &[1]),
        (BlockExit::Unconditional, &[2]),
        (BlockExit::Conditional, &[1, 3]),
        (BlockExit::Conditional, &[0, 4]),
        (BlockExit::Terminal, &[]),
    ]);
    assert!(matches!(cfg.structure(), StructuredCfg::Dispatcher { .. }));
    assert_structured_matches_graph(&cfg, ALL_ORACLES);
}

#[test]
fn a_multi_exit_loop_falls_back_to_a_faithful_dispatcher() {
    // 0 header -> 1; 1 cond -> {2 early-exit, 3}; 3 cond -> {0 back, 4 exit}. Two exits: 2 and 4.
    let cfg = graph(&[
        (BlockExit::Fallthrough, &[1]),
        (BlockExit::Conditional, &[2, 3]),
        (BlockExit::Terminal, &[]),
        (BlockExit::Conditional, &[0, 4]),
        (BlockExit::Terminal, &[]),
    ]);
    assert!(matches!(cfg.structure(), StructuredCfg::Dispatcher { .. }));
    assert_structured_matches_graph(&cfg, ALL_ORACLES);
}

#[test]
fn every_reconstruction_covers_each_reachable_block_exactly_once() {
    let graphs = [
        graph(&[(BlockExit::Terminal, &[])]),
        graph(&[
            (BlockExit::Conditional, &[1, 2]),
            (BlockExit::Unconditional, &[3]),
            (BlockExit::Unconditional, &[3]),
            (BlockExit::Terminal, &[]),
        ]),
        graph(&[
            (BlockExit::Conditional, &[1, 2]),
            (BlockExit::Unconditional, &[2]),
            (BlockExit::Conditional, &[1, 3]),
            (BlockExit::Terminal, &[]),
        ]),
    ];
    for cfg in &graphs {
        let reachable = cfg.reachable_blocks();
        assert!(super::covers_each_reachable_block_once(
            &cfg.structure(),
            &reachable
        ));
    }
}

// ------------------------------------------------------------------------------------------------
// Program -> control-flow-graph construction.
// ------------------------------------------------------------------------------------------------

#[test]
fn builds_a_conditional_diamond_from_a_decoded_program() {
    // s_cbranch_scc0 +2 (to the else block), then two arms, s_endpgm.
    // pc0: s_cbranch_scc0 to pc12   (offset (12-4)/4 = 2)
    // pc4: s_nop  (then arm)        -> falls to pc8
    // pc8: s_branch to pc16         (skip else)  offset (16-12)/4 = 1
    // pc12: s_nop (else arm)        -> falls to pc16
    // pc16: s_endpgm
    let program = decode_program(
        0x1000,
        &[
            0xbf84_0002, // s_cbranch_scc0 +2 -> pc12
            0xbf80_0000, // s_nop
            0xbf82_0001, // s_branch +1 -> pc16
            0xbf80_0000, // s_nop
            0xbf81_0000, // s_endpgm
            0xdead_beef,
        ],
    )
    .expect("diamond program");
    let cfg = build_control_flow_graph(&program);
    // Entry conditional splits to the two arms; both reach the endpgm block.
    let entry = &cfg.blocks()[cfg.entry()];
    assert_eq!(entry.exit, BlockExit::Conditional);
    assert_eq!(entry.successors.len(), 2);
    // The post-dominator of the entry is the shared endpgm block.
    assert!(
        cfg.post_dominators()
            .immediate_post_dominator(cfg.entry())
            .is_some()
    );
    assert!(cfg.is_reducible());
    assert_structured_matches_graph(&cfg, ALL_ORACLES);
}

#[test]
fn builds_a_backward_loop_edge_from_a_decoded_program() {
    // pc0: s_nop ; pc4: s_cbranch_scc1 back to pc0 ; pc8: s_endpgm.
    // Displacement: next_pc(8) + offset*4 == 0  =>  offset == -2 == 0xfffe.
    let program = decode_program(
        0x2000,
        &[
            0xbf80_0000, // s_nop
            0xbf85_fffe, // s_cbranch_scc1 -2 -> pc0
            0xbf81_0000, // s_endpgm
            0xdead_beef,
        ],
    )
    .expect("loop program");
    let cfg = build_control_flow_graph(&program);
    let loops = cfg.natural_loops();
    assert_eq!(loops.len(), 1, "one natural loop from the backward branch");
    assert!(cfg.is_reducible());
    assert_structured_matches_graph(&cfg, ALL_ORACLES);
}
