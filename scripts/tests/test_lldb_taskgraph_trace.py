# SPDX-License-Identifier: GPL-2.0-or-later

from __future__ import annotations

import importlib.util
import os
import sys
import unittest
from pathlib import Path
from unittest import mock


SCRIPT = Path(__file__).resolve().parents[1] / "lldb_taskgraph_trace.py"
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
        self.memory.update(
            {
                0x1008: 1,
                queue + 0x1A8: (0x15 << TRACE.TASK_QUEUE_WAITER_BITS) | 1,
                queue + 0x118: 0xA1,
                queue + 0x160: 0xB1,
                queue + 0x1F8: wrapper,
                wrapper + 0x10: underlying,
                underlying + 0x10: 0,
                0x8000: 0x9000,
                0x8008: TRACE.DEFAULT_TARGET_FENCE_CALLER,
                0x9008: TRACE.DEFAULT_TARGET_FENCE_PARENT,
            }
        )

        candidate = state.observe(
            TRACE.PREPARE,
            {"rbx": 0x1000, "rbp": 0x7000},
            7,
            self.read_u64,
        )
        self.assertTrue(candidate.accepted)
        self.assertIn("queue-candidate", candidate.message or "")
        self.assertEqual(
            state.active_breakpoints(),
            TRACE.QUEUE_HANDOFF_DISCOVERY_BREAKPOINTS,
        )

        discovered = state.observe(
            TRACE.PUBLISH,
            {"rsi": 0x1000, "rdi": queue},
            7,
            self.read_u64,
        )
        self.assertTrue(discovered.accepted)
        self.assertIn("event_condition=0x5028", discovered.message or "")
        self.assertEqual(state.queue_object, queue_object)
        self.assertEqual(
            state.active_breakpoints(),
            TRACE.QUEUE_HANDOFF_BREAKPOINTS,
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

        self.memory[queue + 0x1A8] = 0x16 << TRACE.TASK_QUEUE_WAITER_BITS
        self.memory[queue + 0x118] = 0xA2
        self.memory[queue + 0x160] = 0xB2
        enqueue = state.observe(
            TRACE.PUBLISH_RETURN,
            {"r14": queue_object, "rbx": 0, "rax": 0},
            7,
            self.read_u64,
        )
        self.assertTrue(enqueue.accepted)
        self.assertIn("target=True", enqueue.message or "")
        self.assertIn("wake_waiter=0", enqueue.message or "")
        self.assertIn("lane1_head=162", enqueue.message or "")
        self.assertIn("lane1_tail=178", enqueue.message or "")

        self.memory[0xA000 - 0x48] = 1
        self.memory[queue + 0x1A8] |= 1
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

        waiting = state.observe(
            TRACE.EVENT_WAIT_ENTRY,
            {"r12": underlying},
            8,
            self.read_u64,
        )
        self.assertTrue(waiting.accepted)
        self.assertIn("mutex=0x5020", waiting.message or "")
        self.assertIn("condition=0x5028", waiting.message or "")

    def test_queue_handoff_requires_target_follow(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires target follow"):
            TRACE.TaskGraphTraceState(
                cycle_limit=1,
                hit_limit=8,
                queue_handoff=True,
            )

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
