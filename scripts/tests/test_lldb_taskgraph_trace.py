# SPDX-License-Identifier: GPL-2.0-or-later

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


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
        self.assertEqual(self.state.active_breakpoints(), {
            TRACE.DISPATCH: TRACE.BREAKPOINTS[TRACE.PHASE_PUBLISH][TRACE.DISPATCH]
        })
        self.assertEqual(len(TRACE.BREAKPOINTS[TRACE.PHASE_CONSUME]), 4)

    def test_wake_path_correlates_only_the_exact_task_and_thread(self) -> None:
        dispatch = self.observe(TRACE.DISPATCH, {"rsi": 0x1000, "rdx": 3, "rcx": 2})
        self.assertTrue(dispatch.accepted)
        self.assertEqual(self.state.task, 0x1000)
        self.assertEqual(self.state.completion_event, 0x2000)
        self.assertEqual(
            set(self.state.active_breakpoints()), {TRACE.DISPATCH, TRACE.PUBLISH}
        )

        other = self.observe(TRACE.PUBLISH, {"rsi": 0x9990, "rdi": 0x3000})
        self.assertFalse(other.accepted)
        self.assertEqual(self.state.queue, 0)

        published = self.observe(
            TRACE.PUBLISH,
            {"rsi": 0x1000, "rdi": 0x3000, "rdx": 0, "rsp": 0x5000},
        )
        self.assertTrue(published.accepted)
        self.assertEqual(self.state.publisher_thread, 7)
        self.assertEqual(
            set(self.state.active_breakpoints()),
            {TRACE.DISPATCH, TRACE.PUBLISH_RETURN},
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
        self.assertEqual(self.state.phase, TRACE.PHASE_PUBLISH)
        self.assertEqual(
            set(self.state.active_breakpoints()), {TRACE.DISPATCH, TRACE.WAKE}
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
            {TRACE.DISPATCH, TRACE.POP_LOCAL, TRACE.POP_WAIT, TRACE.POP_DRAIN},
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
        self.observe(TRACE.PUBLISH, {"rsi": 0x1000, "rsp": 0x5000})
        self.observe(
            TRACE.PUBLISH_RETURN,
            {"rax": (1 << 64) - 1, "rsp": 0x5008},
        )

        self.assertFalse(
            self.observe(TRACE.POP_WAIT, {"rax": 0x9990}).accepted
        )
        popped = self.observe(TRACE.POP_DRAIN, {"rax": 0x1000})
        self.assertTrue(popped.accepted)
        self.assertEqual(self.state.pop_site, TRACE.POP_DRAIN)
        self.assertEqual(
            set(self.state.active_breakpoints()), {TRACE.DISPATCH, TRACE.EXECUTE}
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
            TRACE.DISPATCH, {"rsi": 0x3000, "rdx": 3, "rcx": 2}
        )

        self.assertTrue(replacement.accepted)
        self.assertIn("superseded task=0x1000", replacement.message or "")
        self.assertIn("phase=consume", replacement.message or "")
        self.assertIn("next=(cycle=2 dispatch", replacement.message or "")
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
        replacement = self.observe(TRACE.DISPATCH, {"rsi": 0x3000})

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


if __name__ == "__main__":
    unittest.main()
