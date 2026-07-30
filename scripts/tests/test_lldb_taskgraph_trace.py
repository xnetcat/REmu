# SPDX-License-Identifier: GPL-2.0-or-later

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


SCRIPT = Path(__file__).resolve().parents[1] / "lldb_taskgraph_trace.py"
PROBE_SCRIPT = Path(__file__).resolve().parents[1] / "probe-taskgraph.sh"
SPEC = importlib.util.spec_from_file_location("lldb_taskgraph_trace", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
TRACE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = TRACE
SPEC.loader.exec_module(TRACE)


class TaskGraphTraceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.memory = {
            0x1018: 0x2000,
            0x31A8: (0x15 << TRACE.TASK_QUEUE_WAITER_BITS) | 0x12,
            **{0x2000 + index * 8: 0xA0 + index for index in range(12)},
        }
        self.state = TRACE.TaskGraphTraceState(cycle_limit=2, hit_limit=32)

    def read_u64(self, address: int) -> int | None:
        return self.memory.get(address)

    def observe(
        self, label: str, registers: dict[str, int], thread: int = 7
    ) -> TRACE.TraceResult:
        return self.state.observe(label, registers, thread, self.read_u64)

    def test_every_phase_fits_four_hardware_breakpoints(self) -> None:
        self.assertTrue(
            all(
                len(breakpoints) <= TRACE.MAXIMUM_HARDWARE_BREAKPOINTS
                for breakpoints in TRACE.BREAKPOINTS.values()
            )
        )
        self.assertEqual(
            self.state.active_breakpoints(),
            {
                TRACE.PREPARE: TRACE.PREPARE_ADDRESS,
                TRACE.DISPATCH: TRACE.BREAKPOINTS[TRACE.PHASE_PUBLISH][
                    TRACE.DISPATCH
                ],
            },
        )
        self.assertEqual(len(TRACE.BREAKPOINTS[TRACE.PHASE_CONSUME]), 4)
        self.assertLessEqual(
            len(TRACE.QUEUE_HANDOFF_BREAKPOINTS),
            TRACE.MAXIMUM_HARDWARE_BREAKPOINTS,
        )
        self.assertLessEqual(
            len(TRACE.QUEUE_HANDOFF_POST_PUBLISH_BREAKPOINTS),
            TRACE.MAXIMUM_HARDWARE_BREAKPOINTS,
        )

    def test_prepare_captures_a_fence_blocked_before_conditional_dispatch(
        self,
    ) -> None:
        prepared = self.observe(
            TRACE.PREPARE,
            {"rbx": 0x1000, "r12": 3, "rax": 2},
        )

        self.assertTrue(prepared.accepted)
        self.assertEqual(self.state.task, 0x1000)
        self.assertEqual(self.state.completion_event, 0x2000)
        self.assertEqual(self.state.counter_before_release, 3)
        self.assertEqual(self.state.release_threshold, 2)
        self.assertIn("dispatchable=False", prepared.message or "")
        self.assertEqual(
            set(self.state.active_breakpoints()),
            {TRACE.PREPARE, TRACE.DISPATCH, TRACE.PUBLISH},
        )

        dispatched = self.observe(
            TRACE.DISPATCH,
            {"rsi": 0x1000, "rdx": 3, "rcx": 2},
        )
        self.assertTrue(dispatched.accepted)
        self.assertIn("conditional-dispatch", dispatched.message or "")

    def test_prepare_only_keeps_one_breakpoint_and_records_every_fence(
        self,
    ) -> None:
        state = TRACE.TaskGraphTraceState(
            cycle_limit=2,
            hit_limit=32,
            prepare_only=True,
        )

        ready = state.observe(
            TRACE.PREPARE,
            {"rbx": 0x1000, "r12": 2, "rax": 2},
            7,
            self.read_u64,
        )
        self.assertTrue(ready.accepted)
        self.assertEqual(state.task, 0)
        self.assertEqual(state.cycles_started, 0)
        self.assertIn("sample=1 prepare-only", ready.message or "")
        self.assertIn("dispatchable=True", ready.message or "")
        self.assertEqual(
            state.active_breakpoints(),
            {TRACE.PREPARE: TRACE.PREPARE_ADDRESS},
        )

        self.memory.update(
            {
                0x1008: 5 << 32,
                0x5020: 0x6000,
                0x5028: 2,
                0x6000: 0x7000,
                0x6008: 0,
                0x7000: 0xA,
                0x7008: 0xB,
                0x7010: 0xC,
                0x7018: 0xD,
                0x8000: 0x9000,
                0x8008: 0x801BE0507,
                0x9008: 0x80101AF94,
            }
        )
        blocked = state.observe(
            TRACE.PREPARE,
            {
                "rbx": 0x1000,
                "r12": 3,
                "rax": 2,
                "r15": 0x5000,
                "rbp": 0x8000,
            },
            7,
            self.read_u64,
        )
        self.assertTrue(blocked.accepted)
        self.assertEqual(state.task, 0)
        self.assertIn("sample=2 prepare-only", blocked.message or "")
        self.assertIn("dispatchable=False", blocked.message or "")
        self.assertIn("task_counter_after=5", blocked.message or "")
        self.assertIn("caller=0x801be0507", blocked.message or "")
        self.assertIn("caller_parent=0x80101af94", blocked.message or "")
        self.assertIn("prerequisite_count=2", blocked.message or "")
        self.assertIn("0x7000:['0xa', '0xb', '0xc', '0xd']", blocked.message or "")
        self.assertEqual(
            state.active_breakpoints(),
            {TRACE.PREPARE: TRACE.PREPARE_ADDRESS},
        )

    def test_target_follow_filters_caller_and_keeps_fixed_stage_breakpoints(
        self,
    ) -> None:
        state = TRACE.TaskGraphTraceState(
            cycle_limit=2,
            hit_limit=32,
            target_follow=True,
        )
        self.memory.update(
            {
                0x8000: 0x9000,
                0x8008: TRACE.DEFAULT_TARGET_FENCE_CALLER,
                0x9008: TRACE.DEFAULT_TARGET_FENCE_PARENT,
            }
        )

        ignored = state.observe(
            TRACE.PREPARE,
            {"rbx": 0x1000, "rbp": 0x7000},
            7,
            self.read_u64,
        )
        self.assertFalse(ignored.accepted)
        self.assertEqual(state.task, 0)
        self.assertEqual(
            state.active_breakpoints(),
            {TRACE.PREPARE: TRACE.PREPARE_ADDRESS},
        )

        captured = state.observe(
            TRACE.PREPARE,
            {
                "rbx": 0x1000,
                "rbp": 0x8000,
                "r12": 1,
                "rax": 1,
            },
            7,
            self.read_u64,
        )
        self.assertTrue(captured.accepted)
        self.assertIn("target=1 capture", captured.message or "")
        fixed = {
            TRACE.PREPARE: TRACE.PREPARE_ADDRESS,
            TRACE.PUBLISH: TRACE.BREAKPOINTS[TRACE.PHASE_PUBLISH][TRACE.PUBLISH],
            TRACE.EXECUTE: TRACE.BREAKPOINTS[TRACE.PHASE_CONSUME][TRACE.EXECUTE],
            TRACE.COMPLETE: TRACE.BREAKPOINTS[TRACE.PHASE_COMPLETE][TRACE.COMPLETE],
        }
        self.assertEqual(state.active_breakpoints(), fixed)

        self.assertFalse(
            state.observe(
                TRACE.PUBLISH,
                {"rsi": 0x9990},
                8,
                self.read_u64,
            ).accepted
        )
        published = state.observe(
            TRACE.PUBLISH,
            {"rsi": 0x1000, "rdi": 0x3000, "rdx": 1},
            8,
            self.read_u64,
        )
        self.assertTrue(published.accepted)
        executed = state.observe(
            TRACE.EXECUTE,
            {"rdi": 0x1000},
            9,
            self.read_u64,
        )
        self.assertTrue(executed.accepted)
        completed = state.observe(
            TRACE.COMPLETE,
            {"r14": 0x1000},
            9,
            self.read_u64,
        )
        self.assertTrue(completed.accepted)
        self.assertIn("published=True executed=True", completed.message or "")
        self.assertEqual(state.task, 0)
        self.assertEqual(state.active_breakpoints(), fixed)

    def test_queue_handoff_correlates_enqueue_register_and_event_wait(
        self,
    ) -> None:
        state = TRACE.TaskGraphTraceState(
            cycle_limit=2,
            hit_limit=32,
            target_follow=True,
            queue_handoff=True,
        )
        queue = 0x3000
        queue_object = queue - 0x38
        wrapper = 0x4000
        underlying = 0x5000
        singleton = 0x6000
        self.memory.update(
            {
                0x1008: 1,
                TRACE.TASKGRAPH_SINGLETON_POINTER: singleton,
                singleton + 0x18 + 0x8: queue_object,
                queue + 0x1A8: (0x15 << TRACE.TASK_QUEUE_WAITER_BITS) | 1,
                queue + 0x118: 0xA1,
                queue + 0x160: 0xB1,
                queue + 0x1F8: wrapper,
                TRACE.TASKGRAPH_NODE_SLAB_POINTERS: 0xD000,
                0xD000 + 0xA1 * TRACE.TASK_QUEUE_NODE_STRIDE + 8: 0x1110,
                0xD000 + 0xB1 * TRACE.TASK_QUEUE_NODE_STRIDE + 8: 0x2220,
                wrapper + 0x10: underlying,
                underlying + 0x10: 0,
                0x8000: 0x9000,
                0x8008: TRACE.DEFAULT_TARGET_FENCE_CALLER,
                0x9008: TRACE.DEFAULT_TARGET_FENCE_PARENT,
            }
        )

        ignored = state.observe(
            TRACE.PREPARE,
            {"rbx": 0x1000, "rbp": 0x7000},
            7,
            self.read_u64,
        )
        self.assertFalse(ignored.accepted)
        self.assertEqual(
            state.active_breakpoints(),
            TRACE.QUEUE_HANDOFF_DISCOVERY_BREAKPOINTS,
        )

        self.memory[0x1018] = 0x2100
        target = state.observe(
            TRACE.PREPARE,
            {
                "rbx": 0x1000,
                "rbp": 0x8000,
                "r12": 1,
                "rax": 1,
            },
            7,
            self.read_u64,
        )
        self.assertTrue(target.accepted)
        self.assertIn("target=1 capture", target.message or "")
        self.assertIn("taskgraph=0x6000", target.message or "")
        self.assertIn("event_condition=0x5028", target.message or "")
        self.assertEqual(state.queue_object, queue_object)
        self.assertEqual(
            state.active_breakpoints(),
            TRACE.QUEUE_HANDOFF_BREAKPOINTS,
        )
        self.assertNotIn(TRACE.PREPARE, state.active_breakpoints())
        self.assertIn(TRACE.PUBLISH, state.active_breakpoints())

        self.memory[queue + 0x1A8] = 0x16 << TRACE.TASK_QUEUE_WAITER_BITS
        self.memory[queue + 0x118] = 0xA2
        self.memory[queue + 0x160] = 0xB2
        self.memory[
            0xD000 + 0xA2 * TRACE.TASK_QUEUE_NODE_STRIDE + 8
        ] = 0x1000
        self.memory[
            0xD000 + 0xB2 * TRACE.TASK_QUEUE_NODE_STRIDE + 8
        ] = 0x3330
        publish = state.observe(
            TRACE.PUBLISH,
            {
                "rsi": 0x1000,
                "rdi": queue,
                "rdx": 1,
                "rsp": 0x7000,
            },
            7,
            self.read_u64,
        )
        self.assertTrue(publish.accepted)
        self.assertIn("target-publish-entry", publish.message or "")
        self.assertFalse(
            state.observe(
                TRACE.PUBLISH_RETURN,
                {
                    "r14": queue_object,
                    "rbx": 0,
                    "rax": 0,
                    "rsp": 0x7008,
                },
                8,
                self.read_u64,
            ).accepted
        )
        enqueue = state.observe(
            TRACE.PUBLISH_RETURN,
            {
                "r14": queue_object,
                "rbx": 0,
                "rax": 0,
                "rsp": 0x7008,
            },
            7,
            self.read_u64,
        )
        self.assertTrue(enqueue.accepted)
        self.assertIn("target=True", enqueue.message or "")
        self.assertIn("wake_waiter=0", enqueue.message or "")
        self.assertIn("lane1_head=162", enqueue.message or "")
        self.assertIn("lane1_tail=178", enqueue.message or "")
        self.assertIn("/task=0x1000/target=True", enqueue.message or "")
        self.assertIn("/task=0x3330/target=False", enqueue.message or "")
        self.assertIn("chain=linked", enqueue.message or "")
        self.assertIn("target_handle=0xa2", enqueue.message or "")
        self.assertIn("target_slot=", enqueue.message or "")
        self.assertEqual(
            state.active_breakpoints(),
            TRACE.QUEUE_HANDOFF_POST_PUBLISH_BREAKPOINTS,
        )
        self.assertIn(TRACE.EXECUTE, state.active_breakpoints())

        self.memory[0xA000 - 0x48] = 1
        self.memory[queue + 0x1A8] |= 1
        self.memory[queue + 0x118] = 0xB2
        self.memory[
            0xD000 + 0xA2 * TRACE.TASK_QUEUE_NODE_STRIDE + 8
        ] = 0
        self.memory[0xD000 + 0xB2 * TRACE.TASK_QUEUE_NODE_STRIDE] = 0
        self.memory[
            0xD000 + 0xB2 * TRACE.TASK_QUEUE_NODE_STRIDE + 0x10
        ] = 0x4440
        registered = state.observe(
            TRACE.QUEUE_REGISTER,
            {
                "r13": queue,
                "rflags": 1 << 6,
                "rbp": 0xA000,
                "rax": 0x58000000,
                "rbx": 0x58000001,
            },
            8,
            self.read_u64,
        )
        self.assertTrue(registered.accepted)
        self.assertIn("registered=True", registered.message or "")
        self.assertIn("waiter_bit=1", registered.message or "")
        self.assertIn("chain=null", registered.message or "")
        self.assertIn("known_target_mutated=True", registered.message or "")
        self.assertIn("known_target_current=[unreadable,0x0,unreadable]", registered.message or "")

        waiting = state.observe(
            TRACE.EVENT_WAIT_ENTRY,
            {"r12": underlying},
            8,
            self.read_u64,
        )
        self.assertTrue(waiting.accepted)
        self.assertIn("mutex=0x5020", waiting.message or "")
        self.assertIn("condition=0x5028", waiting.message or "")
        self.assertFalse(state.done)
        self.assertEqual(
            state.active_breakpoints(),
            TRACE.QUEUE_HANDOFF_POST_PUBLISH_BREAKPOINTS,
        )
        final_snapshot = state.final_queue_handoff_snapshot(
            self.read_u64
        )
        self.assertIn("unresolved-target task=0x1000", final_snapshot)
        self.assertIn("published=True executed=False", final_snapshot)
        self.assertIn("known_target_mutated=True", final_snapshot)

    def test_queue_handoff_cycles_a_normally_completed_fence(self) -> None:
        state = TRACE.TaskGraphTraceState(
            cycle_limit=2,
            hit_limit=32,
            target_follow=True,
            queue_handoff=True,
        )
        task = 0x1000
        queue = 0x3000
        queue_object = queue - 0x38
        wrapper = 0x4000
        underlying = 0x5000
        singleton = 0x6000
        memory = {
            task + 0x8: 1,
            task + 0x18: 0x2100,
            TRACE.TASKGRAPH_SINGLETON_POINTER: singleton,
            singleton + 0x18 + 0x8: queue_object,
            queue + 0x1F8: wrapper,
            wrapper + 0x10: underlying,
            0x8000: 0x9000,
            0x8008: TRACE.DEFAULT_TARGET_FENCE_CALLER,
            0x9008: TRACE.DEFAULT_TARGET_FENCE_PARENT,
        }

        captured = state.observe(
            TRACE.PREPARE,
            {
                "rbx": task,
                "rbp": 0x8000,
                "r12": 1,
                "rax": 1,
            },
            7,
            memory.get,
        )
        self.assertTrue(captured.accepted)
        self.assertIn(TRACE.COMPLETE, state.active_breakpoints())
        published = state.observe(
            TRACE.PUBLISH,
            {
                "rsi": task,
                "rdi": queue,
                "rdx": 1,
                "rsp": 0x7000,
            },
            7,
            memory.get,
        )
        self.assertTrue(published.accepted)
        enqueue = state.observe(
            TRACE.PUBLISH_RETURN,
            {
                "r14": queue_object,
                "rbx": 0,
                "rax": 0,
                "rsp": 0x7008,
            },
            7,
            memory.get,
        )
        self.assertTrue(enqueue.accepted)
        self.assertIn(TRACE.EXECUTE, state.active_breakpoints())
        executed = state.observe(
            TRACE.EXECUTE,
            {"rdi": task},
            8,
            memory.get,
        )
        self.assertTrue(executed.accepted)
        self.assertTrue(state.target_executed)
        completed = state.observe(
            TRACE.COMPLETE,
            {"r14": task},
            8,
            memory.get,
        )

        self.assertTrue(completed.accepted)
        self.assertIn("normal-complete", completed.message or "")
        self.assertFalse(state.done)
        self.assertEqual(state.cycles_started, 1)
        self.assertEqual(state.cycles_completed, 1)
        self.assertEqual(state.task, 0)
        self.assertEqual(state.queue, 0)
        self.assertEqual(
            state.active_breakpoints(),
            TRACE.QUEUE_HANDOFF_DISCOVERY_BREAKPOINTS,
        )

    def test_queue_handoff_requires_target_follow(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires target follow"):
            TRACE.TaskGraphTraceState(
                cycle_limit=1,
                hit_limit=8,
                queue_handoff=True,
            )

    def test_task_queue_node_decoder_uses_page_and_slot_slab(self) -> None:
        node_id = (2 << TRACE.TASK_QUEUE_NODE_PAGE_BITS) | 7
        value = (0x15 << TRACE.TASK_QUEUE_WAITER_BITS) | node_id
        page_base = 0xD000
        task = 0x123400
        memory = {
            TRACE.TASKGRAPH_NODE_SLAB_POINTERS + 2 * 8: page_base,
            page_base + 7 * TRACE.TASK_QUEUE_NODE_STRIDE + 8: task,
        }

        description = TRACE.describe_task_queue_node(
            value,
            memory.get,
            task,
        )

        self.assertIn(f"{value}/id={node_id:#x}", description)
        self.assertIn("/page=0x2/slot=0x7/base=0xd000", description)
        self.assertIn("/task=0x123400/target=True", description)

    def test_task_queue_chain_walks_raw_handles_to_target(self) -> None:
        first_id = (2 << TRACE.TASK_QUEUE_NODE_PAGE_BITS) | 7
        second_id = (2 << TRACE.TASK_QUEUE_NODE_PAGE_BITS) | 9
        first_handle = (0x15 << TRACE.TASK_QUEUE_WAITER_BITS) | first_id
        second_handle = (0x16 << TRACE.TASK_QUEUE_WAITER_BITS) | second_id
        page_base = 0xD000
        first_slot = page_base + 7 * TRACE.TASK_QUEUE_NODE_STRIDE
        second_slot = page_base + 9 * TRACE.TASK_QUEUE_NODE_STRIDE
        target = 0x123400
        memory = {
            TRACE.TASKGRAPH_NODE_SLAB_POINTERS + 2 * 8: page_base,
            first_slot: second_handle,
            first_slot + 8: 0xAAAA,
            first_slot + 0x10: 0xBBBB,
            second_slot: 0,
            second_slot + 8: target,
            second_slot + 0x10: 0xCCCC,
            target: 0x1111,
            target + 8: 0x2222,
            target + 0x10: 0x3333,
            target + 0x18: 0x4444,
        }

        snapshot = TRACE.inspect_task_queue_chain(
            first_handle,
            target,
            memory.get,
        )
        description = TRACE.describe_task_queue_chain(snapshot)

        self.assertEqual(snapshot.termination, "linked")
        self.assertEqual(snapshot.handles, (first_handle, second_handle))
        self.assertEqual(snapshot.target_handle, second_handle)
        self.assertEqual(snapshot.target_slot, second_slot)
        self.assertEqual(
            snapshot.nodes[-1],
            TRACE.TaskQueueNodeSnapshot(
                second_handle,
                second_slot,
                (0, target, 0xCCCC),
            ),
        )
        self.assertEqual(
            snapshot.target_slot_words,
            (0, target, 0xCCCC),
        )
        self.assertEqual(
            snapshot.target_words,
            (0x1111, 0x2222, 0x3333, 0x4444),
        )
        self.assertIn(
            f"handles=[{first_handle:#x},{second_handle:#x}]",
            description,
        )
        self.assertIn(
            f"{second_handle:#x}@{second_slot:#x}:"
            "[0x0,0x123400,0xcccc]",
            description,
        )
        self.assertIn(
            "target_slot_words=[0x0,0x123400,0xcccc]",
            description,
        )
        self.assertIn(
            "target_words=[0x1111,0x2222,0x3333,0x4444]",
            description,
        )
        self.assertEqual(snapshot.match_reason, "pointer")

        memory[second_slot + 8] = 0
        vanished = TRACE.inspect_task_queue_chain(
            first_handle,
            target,
            memory.get,
        )
        self.assertEqual(vanished.termination, "null")
        self.assertEqual(vanished.target_slot, 0)

        replacement = 0x222200
        completion_event = 0x333300
        memory[second_slot + 8] = replacement
        memory[replacement + 0x18] = completion_event
        event_match = TRACE.inspect_task_queue_chain(
            first_handle,
            target,
            memory.get,
            completion_event,
        )
        self.assertEqual(event_match.termination, "linked")
        self.assertEqual(event_match.matched_task, replacement)
        self.assertEqual(event_match.match_reason, "event")

    def test_persistent_breakpoint_planner_adds_without_replacing(self) -> None:
        prepare = {TRACE.PREPARE: TRACE.PREPARE_ADDRESS}
        fixed = {
            **prepare,
            TRACE.PUBLISH: TRACE.BREAKPOINTS[TRACE.PHASE_PUBLISH][TRACE.PUBLISH],
            TRACE.EXECUTE: TRACE.BREAKPOINTS[TRACE.PHASE_CONSUME][TRACE.EXECUTE],
            TRACE.COMPLETE: TRACE.BREAKPOINTS[TRACE.PHASE_COMPLETE][TRACE.COMPLETE],
        }
        self.assertEqual(
            TRACE.persistent_breakpoint_additions({}, prepare),
            prepare,
        )
        additions = TRACE.persistent_breakpoint_additions(prepare, fixed)
        self.assertNotIn(TRACE.PREPARE, additions)
        self.assertEqual(set(additions), {TRACE.PUBLISH, TRACE.EXECUTE, TRACE.COMPLETE})
        self.assertEqual(
            TRACE.persistent_breakpoint_additions(fixed, fixed),
            {},
        )

    def test_target_addresses_are_validated(self) -> None:
        name = "REMU_TASKGRAPH_TRACE_TARGET_CALLER"
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(
                TRACE.guest_address_setting(name, 0x801BE0507),
                0x801BE0507,
            )
        with mock.patch.dict(os.environ, {name: "0x80101af94"}, clear=True):
            self.assertEqual(
                TRACE.guest_address_setting(name, 0x801BE0507),
                0x80101AF94,
            )
        for value in ("not-an-address", "0x7ffffffff", "0x10000000000"):
            with self.subTest(value=value):
                with mock.patch.dict(os.environ, {name: value}, clear=True):
                    with self.assertRaises(ValueError):
                        TRACE.guest_address_setting(name, 0x801BE0507)

    def test_direct_attach_requires_stopped_process_with_guest_mapping(
        self,
    ) -> None:
        class FakeError:
            def __init__(self) -> None:
                self.failed = False

            def Fail(self) -> bool:
                return self.failed

        class FakeProcess:
            def __init__(
                self,
                *,
                valid: bool = True,
                state: int = 5,
                mapped: bool = True,
            ) -> None:
                self.valid = valid
                self.state = state
                self.mapped = mapped

            def IsValid(self) -> bool:
                return self.valid

            def GetState(self) -> int:
                return self.state

            def ReadMemory(
                self, address: int, size: int, error: FakeError
            ) -> bytes:
                self.testcase.assertEqual(address, TRACE.PREPARE_ADDRESS)
                self.testcase.assertEqual(size, 1)
                error.failed = not self.mapped
                return b"\x90" if self.mapped else b""

        class FakeTarget:
            def __init__(self, process: FakeProcess) -> None:
                self.process = process
                process.testcase = self_test

            def GetProcess(self) -> FakeProcess:
                return self.process

        self_test = self
        fake_lldb = SimpleNamespace(eStateStopped=5, SBError=FakeError)
        with mock.patch.object(TRACE, "lldb", fake_lldb):
            self.assertEqual(
                TRACE._direct_attach_error(
                    FakeTarget(FakeProcess(valid=False))
                ),
                "no valid process is attached",
            )
            self.assertIn(
                "must be stopped",
                TRACE._direct_attach_error(
                    FakeTarget(FakeProcess(state=4))
                )
                or "",
            )
            self.assertIn(
                "has not mapped",
                TRACE._direct_attach_error(
                    FakeTarget(FakeProcess(mapped=False))
                )
                or "",
            )
            self.assertIsNone(
                TRACE._direct_attach_error(FakeTarget(FakeProcess()))
            )

    def test_direct_attach_arms_without_a_host_seed_breakpoint(self) -> None:
        target = mock.Mock()
        debugger = mock.Mock()
        debugger.GetSelectedTarget.return_value = target
        result = mock.Mock()
        with mock.patch.dict(
            os.environ,
            {"REMU_TASKGRAPH_TRACE_ATTACH": "1"},
            clear=True,
        ), mock.patch.object(TRACE, "_arm_direct_attach") as arm:
            TRACE.start_taskgraph_trace(debugger, "", result, {})

        arm.assert_called_once_with(target, result)
        target.BreakpointCreateByRegex.assert_not_called()

    def test_attach_cleanup_detaches_without_terminating_target(self) -> None:
        class FakeError:
            def Fail(self) -> bool:
                return False

        process = mock.Mock()
        process.IsValid.return_value = True
        process.GetState.return_value = 5
        process.Detach.return_value = FakeError()
        target = mock.Mock()
        target.GetProcess.return_value = process
        debugger = mock.Mock()
        debugger.GetSelectedTarget.return_value = target
        result = mock.Mock()
        fake_lldb = SimpleNamespace(
            eStateStopped=5,
            eStateExited=10,
            eStateDetached=9,
        )

        with mock.patch.object(TRACE, "lldb", fake_lldb):
            TRACE.detach_taskgraph_trace(debugger, "", result, {})

        process.Detach.assert_called_once_with(False)
        result.SetError.assert_not_called()
        result.AppendMessage.assert_called_once_with(
            "[REMu][LLDB][TASKGRAPH] detached without terminating target"
        )

    def test_finish_stops_and_detaches_even_if_backtrace_fails(self) -> None:
        class FakeError:
            def Fail(self) -> bool:
                return False

            def GetCString(self) -> str:
                return ""

        state = {"value": 6}
        process = mock.Mock()
        process.IsValid.return_value = True
        process.GetState.side_effect = lambda: state["value"]

        def stop() -> FakeError:
            state["value"] = 5
            return FakeError()

        process.Stop.side_effect = stop
        process.Detach.return_value = FakeError()
        target = mock.Mock()
        target.GetProcess.return_value = process
        debugger = mock.Mock()
        debugger.GetSelectedTarget.return_value = target
        result = mock.Mock()
        fake_lldb = SimpleNamespace(
            eStateRunning=6,
            eStateStopped=5,
            eStateCrashed=8,
            eStateDetached=9,
            eStateExited=10,
            eStateSuspended=11,
        )
        session = mock.Mock()
        session.state.queue_handoff = True
        session.state.final_queue_handoff_snapshot.return_value = (
            "unresolved-target task=0x1000"
        )

        def report(_debugger, command: str, _result) -> None:
            if command.startswith("thread backtrace"):
                raise RuntimeError("synthetic report failure")

        with mock.patch.object(TRACE, "lldb", fake_lldb), \
            mock.patch.object(TRACE, "SESSION", session), \
            mock.patch.object(
                TRACE,
                "_append_report_command",
                side_effect=report,
            ) as report_command:
            TRACE.finish_taskgraph_trace(debugger, "", result, {})

        process.Stop.assert_called_once_with()
        self.assertEqual(report_command.call_count, 2)
        process.Detach.assert_called_once_with(False)
        session.cancel_attach_timeout.assert_called_once_with()
        session.state.final_queue_handoff_snapshot.assert_called_once()
        self.assertTrue(
            any(
                "report exception" in call.args[0]
                for call in result.AppendMessage.call_args_list
            )
        )
        self.assertTrue(
            any(
                "detached without terminating target" in call.args[0]
                for call in result.AppendMessage.call_args_list
            )
        )
        self.assertTrue(
            any(
                "final-snapshot unresolved-target task=0x1000"
                in call.args[0]
                for call in result.AppendMessage.call_args_list
            )
        )

    def test_attach_duration_interrupts_before_external_watchdog(self) -> None:
        class FakeTimer:
            def __init__(self, duration: int, callback) -> None:
                self.duration = duration
                self.callback = callback
                self.daemon = False
                self.started = False
                self.cancelled = False

            def start(self) -> None:
                self.started = True

            def cancel(self) -> None:
                self.cancelled = True

        process = mock.Mock()
        process.IsValid.return_value = True
        process.GetState.return_value = 6
        target = mock.Mock()
        target.GetProcess.return_value = process
        trace = object.__new__(TRACE.LldbTaskGraphTrace)
        trace.target = target
        trace.attach_timer = None
        trace.attach_timeout_requested = False
        fake_lldb = SimpleNamespace(
            eStateRunning=6,
            eStateExited=10,
            eStateDetached=9,
        )

        with mock.patch.dict(
            os.environ,
            {"REMU_TASKGRAPH_TRACE_ATTACH_DURATION_SECONDS": "45"},
            clear=True,
        ), mock.patch.object(TRACE.threading, "Timer", FakeTimer), \
            mock.patch.object(TRACE, "lldb", fake_lldb):
            trace.start_attach_timeout()
            timer = trace.attach_timer
            self.assertIsInstance(timer, FakeTimer)
            assert isinstance(timer, FakeTimer)
            self.assertEqual(timer.duration, 45)
            self.assertTrue(timer.daemon)
            self.assertTrue(timer.started)
            with mock.patch("builtins.print"):
                timer.callback()

        self.assertTrue(trace.attach_timeout_requested)
        process.SendAsyncInterrupt.assert_called_once_with()
        trace.cancel_attach_timeout()
        self.assertTrue(timer.cancelled)

    def test_attach_cli_rejects_non_numeric_pid_before_any_launch(self) -> None:
        completed = subprocess.run(
            [
                "/bin/sh",
                str(PROBE_SCRIPT),
                "--attach",
                "not-a-pid",
                "/not/used/eboot.bin",
            ],
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(completed.returncode, 64)
        self.assertIn("attach PID must be an integer", completed.stderr)

    def test_attach_contract_checks_target_write_watch_environment(
        self,
    ) -> None:
        source = PROBE_SCRIPT.read_text()

        self.assertIn(
            '/bin/ps eww -p "$pid" -o command=',
            source,
        )
        self.assertIn(
            "requires target environment "
            "REMU_DISABLE_CPU_WRITE_WATCH=1",
            source,
        )
        self.assertIn(
            'echo "target_cpu_write_watch_disabled='
            '$target_cpu_write_watch_disabled"',
            source,
        )
        self.assertIn(
            'echo "debugger_cpu_write_watch_disabled='
            '$disable_cpu_write_watch"',
            source,
        )
        self.assertIn(
            'REMU_TASKGRAPH_TRACE_ATTACH_DURATION_SECONDS='
            '"$attach_trace_seconds"',
            source,
        )
        self.assertIn(
            "settings set "
            "interpreter.stop-command-source-on-error false",
            source,
        )
        self.assertIn(
            "-o 'remu-taskgraph-trace-finish'",
            source,
        )

    def test_wake_path_correlates_only_the_exact_task_and_thread(self) -> None:
        dispatch = self.observe(TRACE.DISPATCH, {"rsi": 0x1000, "rdx": 3, "rcx": 2})
        self.assertTrue(dispatch.accepted)
        self.assertEqual(self.state.task, 0x1000)
        self.assertEqual(self.state.completion_event, 0x2000)
        self.assertEqual(
            set(self.state.active_breakpoints()),
            {TRACE.PREPARE, TRACE.DISPATCH, TRACE.PUBLISH},
        )

        other = self.observe(TRACE.PUBLISH, {"rsi": 0x9990, "rdi": 0x3000})
        self.assertFalse(other.accepted)
        self.assertEqual(self.state.queue, 0)

        published = self.observe(
            TRACE.PUBLISH,
            {"rsi": 0x1000, "rdi": 0x3000, "rdx": 0, "rsp": 0x5000},
        )
        self.assertTrue(published.accepted)
        self.assertIn(
            "wait_state_before=0x54000012/waiters=0x12/generation=0x15",
            published.message or "",
        )
        self.assertEqual(self.state.publisher_thread, 7)
        self.assertEqual(
            set(self.state.active_breakpoints()),
            {TRACE.PREPARE, TRACE.DISPATCH, TRACE.PUBLISH_RETURN},
        )

        wrong_thread = self.observe(
            TRACE.PUBLISH_RETURN, {"rax": 0, "rsp": 0x5008}, thread=8
        )
        self.assertFalse(wrong_thread.accepted)
        self.assertEqual(self.state.phase, TRACE.PHASE_PUBLISH)

        wrong_frame = self.observe(
            TRACE.PUBLISH_RETURN, {"rax": 0, "rsp": 0x6008}
        )
        self.assertFalse(wrong_frame.accepted)
        self.assertEqual(self.state.phase, TRACE.PHASE_PUBLISH)

        wake_required = self.observe(
            TRACE.PUBLISH_RETURN, {"rax": 0, "rsp": 0x5008}
        )
        self.assertTrue(wake_required.accepted)
        self.assertIn(
            "wait_state_after=0x54000012/waiters=0x12/generation=0x15",
            wake_required.message or "",
        )
        self.assertEqual(self.state.phase, TRACE.PHASE_PUBLISH)
        self.assertEqual(
            set(self.state.active_breakpoints()),
            {TRACE.PREPARE, TRACE.DISPATCH, TRACE.WAKE},
        )

        wrong_wake_frame = self.observe(
            TRACE.WAKE, {"rdi": 0x4000, "rsp": 0x6008}
        )
        self.assertFalse(wrong_wake_frame.accepted)

        wake = self.observe(TRACE.WAKE, {"rdi": 0x4000, "rsp": 0x5008})
        self.assertTrue(wake.accepted)
        self.assertEqual(self.state.wake_event, 0x4000)
        self.assertEqual(self.state.phase, TRACE.PHASE_CONSUME)
        self.assertEqual(
            set(self.state.active_breakpoints()),
            {TRACE.PREPARE, TRACE.POP_LOCAL, TRACE.POP_WAIT, TRACE.POP_DRAIN},
        )

    def test_negative_publish_result_skips_wake_phase(self) -> None:
        self.observe(TRACE.DISPATCH, {"rsi": 0x1000})
        self.observe(
            TRACE.PUBLISH, {"rsi": 0x1000, "rdi": 0x3000, "rsp": 0x5000}
        )
        result = self.observe(
            TRACE.PUBLISH_RETURN,
            {"rax": (1 << 64) - 1, "rsp": 0x5008},
        )
        self.assertTrue(result.accepted)
        self.assertIn("wake=not-required", result.message or "")
        self.assertEqual(self.state.phase, TRACE.PHASE_CONSUME)

    def test_pop_execute_and_completion_filter_exact_task(self) -> None:
        self.observe(TRACE.DISPATCH, {"rsi": 0x1000})
        self.observe(
            TRACE.PUBLISH,
            {"rsi": 0x1000, "rdi": 0x3000, "rsp": 0x5000},
        )
        self.observe(
            TRACE.PUBLISH_RETURN,
            {"rax": (1 << 64) - 1, "rsp": 0x5008},
        )

        self.assertFalse(
            self.observe(TRACE.POP_WAIT, {"rax": 0x9990}).accepted
        )
        popped = self.observe(TRACE.POP_DRAIN, {"rax": 0x1000})
        self.assertTrue(popped.accepted)
        self.assertIn(
            "wait_state=0x54000012/waiters=0x12/generation=0x15",
            popped.message or "",
        )
        self.assertEqual(self.state.pop_site, TRACE.POP_DRAIN)
        self.assertEqual(
            set(self.state.active_breakpoints()),
            {TRACE.PREPARE, TRACE.EXECUTE},
        )
        self.assertFalse(
            self.observe(TRACE.EXECUTE, {"rdi": 0x9990}).accepted
        )

        execute = self.observe(TRACE.EXECUTE, {"rdi": 0x1000})
        self.assertTrue(execute.accepted)
        self.assertEqual(self.state.phase, TRACE.PHASE_COMPLETE)
        self.assertFalse(
            self.observe(TRACE.COMPLETE, {"r14": 0x9990}).accepted
        )

        completed = self.observe(TRACE.COMPLETE, {"r14": 0x1000})
        self.assertTrue(completed.accepted)
        self.assertIn("'0xa0'", completed.message or "")
        self.assertEqual(self.state.cycles_completed, 1)
        self.assertEqual(self.state.phase, TRACE.PHASE_PUBLISH)
        self.assertEqual(self.state.task, 0)

    def test_new_fence_dispatch_supersedes_a_task_that_was_never_popped(self) -> None:
        self.observe(TRACE.DISPATCH, {"rsi": 0x1000})
        self.observe(
            TRACE.PUBLISH, {"rsi": 0x1000, "rdi": 0x3000, "rsp": 0x5000}
        )
        self.observe(
            TRACE.PUBLISH_RETURN,
            {"rax": (1 << 64) - 1, "rsp": 0x5008},
        )

        self.memory[0x3018] = 0x4000
        replacement = self.observe(
            TRACE.PREPARE, {"rbx": 0x3000, "r12": 1, "rax": 1}
        )

        self.assertTrue(replacement.accepted)
        self.assertIn("superseded task=0x1000", replacement.message or "")
        self.assertIn("phase=consume", replacement.message or "")
        self.assertIn("next=(cycle=2 prepare", replacement.message or "")
        self.assertEqual(self.state.task, 0x3000)
        self.assertEqual(self.state.completion_event, 0x4000)
        self.assertEqual(self.state.cycles_started, 2)
        self.assertEqual(self.state.cycles_completed, 0)

    def test_new_fence_dispatch_supersedes_a_task_after_execute_before_complete(
        self,
    ) -> None:
        self.observe(TRACE.DISPATCH, {"rsi": 0x1000})
        self.observe(
            TRACE.PUBLISH, {"rsi": 0x1000, "rdi": 0x3000, "rsp": 0x5000}
        )
        self.observe(
            TRACE.PUBLISH_RETURN,
            {"rax": (1 << 64) - 1, "rsp": 0x5008},
        )
        self.observe(TRACE.POP_WAIT, {"rax": 0x1000})
        self.observe(TRACE.EXECUTE, {"rdi": 0x1000})

        self.memory[0x3018] = 0x4000
        replacement = self.observe(
            TRACE.PREPARE, {"rbx": 0x3000, "r12": 1, "rax": 1}
        )

        self.assertTrue(replacement.accepted)
        self.assertIn("phase=complete", replacement.message or "")
        self.assertEqual(self.state.task, 0x3000)
        self.assertEqual(self.state.phase, TRACE.PHASE_PUBLISH)

    def test_cycle_limit_finishes_after_bounded_completions(self) -> None:
        state = TRACE.TaskGraphTraceState(cycle_limit=1, hit_limit=16)
        state.observe(TRACE.DISPATCH, {"rsi": 0x1000}, 7, self.read_u64)
        state.observe(
            TRACE.PUBLISH,
            {"rsi": 0x1000, "rsp": 0x5000},
            7,
            self.read_u64,
        )
        state.observe(
            TRACE.PUBLISH_RETURN,
            {"rax": (1 << 64) - 1, "rsp": 0x5008},
            7,
            self.read_u64,
        )
        state.observe(TRACE.POP_WAIT, {"rax": 0x1000}, 7, self.read_u64)
        state.observe(TRACE.EXECUTE, {"rdi": 0x1000}, 7, self.read_u64)
        result = state.observe(
            TRACE.COMPLETE, {"r14": 0x1000}, 7, self.read_u64
        )
        self.assertTrue(result.accepted)
        self.assertTrue(state.done)
        self.assertIn("done reason=cycle-limit", result.message or "")

    def test_hit_limit_disarms_a_hot_filtered_trace(self) -> None:
        state = TRACE.TaskGraphTraceState(cycle_limit=2, hit_limit=2)
        state.observe(TRACE.DISPATCH, {"rsi": 0x1000}, 7, self.read_u64)
        state.observe(TRACE.PUBLISH, {"rsi": 0x9990}, 7, self.read_u64)
        result = state.observe(
            TRACE.PUBLISH, {"rsi": 0x9990}, 7, self.read_u64
        )
        self.assertTrue(result.accepted)
        self.assertTrue(state.done)
        self.assertIn("done reason=hit-limit", result.message or "")

    def test_signed_u32_preserves_eax_wake_result_convention(self) -> None:
        self.assertEqual(TRACE.signed_u32(0), 0)
        self.assertEqual(TRACE.signed_u32((1 << 32) - 1), -1)
        self.assertEqual(TRACE.signed_u32(1 << 31), -(1 << 31))
        self.assertEqual(TRACE.signed_u32((1 << 64) - 1), -1)

    def test_queue_wait_state_decodes_bitmap_and_generation(self) -> None:
        value = (0x123456 << TRACE.TASK_QUEUE_WAITER_BITS) | 0x02000001
        self.assertEqual(
            TRACE.describe_queue_wait_state(value),
            f"{value:#x}/waiters=0x2000001/generation=0x123456",
        )
        self.assertEqual(TRACE.describe_queue_wait_state(None), "unreadable")

    def test_loader_notification_filter_is_exact_to_dyld_trace_symbol(self) -> None:
        self.assertTrue(
            TRACE.is_loader_notification_symbol("dyld`lldb_image_notifier")
        )
        self.assertTrue(
            TRACE.is_loader_notification_symbol("lldb_image_notifier_sim_trap")
        )
        self.assertFalse(TRACE.is_loader_notification_symbol(None))
        self.assertFalse(
            TRACE.is_loader_notification_symbol("libsystem`pthread_wqthread")
        )

    def test_taskgraph_breakpoint_filter_rejects_unrelated_guest_code(self) -> None:
        for phase_breakpoints in TRACE.BREAKPOINTS.values():
            for address in phase_breakpoints.values():
                self.assertTrue(TRACE.is_taskgraph_breakpoint_address(address))
                self.assertTrue(
                    TRACE.is_taskgraph_diagnostic_stop_address(address)
                )
        self.assertTrue(
            TRACE.is_taskgraph_breakpoint_address(TRACE.PREPARE_ADDRESS)
        )
        self.assertFalse(TRACE.is_taskgraph_breakpoint_address(0x801BE1AF9))
        self.assertFalse(
            TRACE.is_taskgraph_diagnostic_stop_address(0x801BE1AF9)
        )
        for address in TRACE.TASKGRAPH_TRACE_FALLTHROUGH_ADDRESSES:
            self.assertFalse(TRACE.is_taskgraph_breakpoint_address(address))
            self.assertTrue(TRACE.is_taskgraph_diagnostic_stop_address(address))
        self.assertFalse(TRACE.is_taskgraph_breakpoint_address(0x200D91D61))
        self.assertFalse(
            TRACE.is_taskgraph_diagnostic_stop_address(0x200D91D61)
        )

    def test_guest_bad_access_filter_passes_only_direct_guest_execution(self) -> None:
        self.assertTrue(
            TRACE.is_guest_bad_access_stop(
                0x804F9CF72,
                "EXC_BAD_ACCESS (code=2, address=0x8002ee0120)",
            )
        )
        self.assertTrue(
            TRACE.is_guest_bad_access_stop(
                TRACE.MAXIMUM_GUEST_EXECUTION_ADDRESS - 1,
                "EXC_BAD_ACCESS",
            )
        )
        self.assertFalse(
            TRACE.is_guest_bad_access_stop(
                TRACE.MINIMUM_GUEST_EXECUTION_ADDRESS - 1,
                "EXC_BAD_ACCESS (code=1, address=0x0)",
            )
        )
        self.assertFalse(
            TRACE.is_guest_bad_access_stop(
                TRACE.MAXIMUM_GUEST_EXECUTION_ADDRESS,
                "EXC_BAD_ACCESS (code=2, address=0x1)",
            )
        )
        self.assertFalse(
            TRACE.is_guest_bad_access_stop(
                0x804F9CF72,
                "EXC_BREAKPOINT (code=1, subcode=0x0)",
            )
        )
        self.assertFalse(TRACE.is_guest_bad_access_stop(0x804F9CF72, None))

    def test_guest_bad_access_filter_stops_all_faults_without_write_watch(
        self,
    ) -> None:
        self.assertFalse(
            TRACE.is_guest_bad_access_stop(
                0x801239079,
                "EXC_BAD_ACCESS (code=2, address=0x8010bd0000)",
                write_watch_enabled=False,
            )
        )


if __name__ == "__main__":
    unittest.main()
