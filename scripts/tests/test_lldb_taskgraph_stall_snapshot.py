# SPDX-License-Identifier: GPL-2.0-or-later

from __future__ import annotations

import importlib.util
import subprocess
import sys
import unittest
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "lldb_taskgraph_stall_snapshot.py"
)
PROBE_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "probe-taskgraph-stall.sh"
)
SPEC = importlib.util.spec_from_file_location(
    "lldb_taskgraph_stall_snapshot", SCRIPT
)
assert SPEC is not None and SPEC.loader is not None
SNAPSHOT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = SNAPSHOT
SPEC.loader.exec_module(SNAPSHOT)


class TaskGraphStallSnapshotTests(unittest.TestCase):
    def setUp(self) -> None:
        self.import_frame = 0x7FF000001000
        self.native_sp = self.import_frame - 0x280
        self.guest_rsp = 0x3000001000
        self.wait_rbp = 0x3000001200
        self.outer_rbp = 0x3000001400
        self.parent_rbp = 0x3000001600
        self.event = 0x7500E12300
        self.memory: dict[int, int] = {
            self.import_frame + SNAPSHOT.IMPORT_FRAME_GUEST_RSP: self.guest_rsp,
            self.import_frame
            + SNAPSHOT.IMPORT_FRAME_GUEST_RETURN: SNAPSHOT.HOST_EVENT_WAIT_RETURN,
            self.guest_rsp
            + SNAPSHOT.GUEST_RSP_WAIT_RBP_OFFSET: self.wait_rbp,
            self.guest_rsp
            + SNAPSHOT.GUEST_RSP_WAIT_RETURN_OFFSET: SNAPSHOT.GAME_THREAD_WAIT_RETURN,
            self.wait_rbp: self.outer_rbp,
            self.wait_rbp + 8: SNAPSHOT.GAME_THREAD_WAIT_RETURN,
            self.outer_rbp: self.parent_rbp,
            self.outer_rbp + 8: SNAPSHOT.BEGIN_FENCE_WAIT_RETURN,
            self.parent_rbp + 8: SNAPSHOT.BEGIN_FENCE_PARENT_RETURN,
            self.outer_rbp
            - SNAPSHOT.OUTER_EVENT_LOCAL_OFFSET: self.event,
        }

    def read_u64(self, address: int) -> int | None:
        return self.memory.get(address)

    def test_decodes_exact_active_wait_chain_and_outer_event(self) -> None:
        candidate = SNAPSHOT.decode_wait_candidate(
            self.import_frame,
            9,
            self.native_sp,
            self.read_u64,
        )

        self.assertIsNotNone(candidate)
        assert candidate is not None
        self.assertEqual(candidate.thread_id, 9)
        self.assertEqual(candidate.guest_rsp, self.guest_rsp)
        self.assertEqual(candidate.outer_rbp, self.outer_rbp)
        self.assertEqual(candidate.event, self.event)
        self.assertEqual(candidate.native_distance, 0x280)

    def test_rejects_a_stale_or_different_guest_caller_chain(self) -> None:
        self.memory[self.outer_rbp + 8] = 0x801BE0999

        self.assertIsNone(
            SNAPSHOT.decode_wait_candidate(
                self.import_frame,
                9,
                self.native_sp,
                self.read_u64,
            )
        )

    def test_decodes_wait_while_main_thread_is_between_import_calls(self) -> None:
        current_rbp = self.guest_rsp + 0x40
        self.memory[current_rbp] = self.wait_rbp
        self.memory[current_rbp + 8] = 0x8012108A1

        candidate = SNAPSHOT.decode_direct_guest_wait(
            9,
            0x80121086F,
            self.guest_rsp,
            current_rbp,
            self.read_u64,
        )

        self.assertIsNotNone(candidate)
        assert candidate is not None
        self.assertEqual(candidate.source, "direct-guest")
        self.assertEqual(candidate.import_frame, 0)
        self.assertEqual(candidate.wait_rbp, self.wait_rbp)
        self.assertEqual(candidate.event, self.event)

    def test_direct_guest_decoder_requires_bounded_monotonic_rbp_chain(
        self,
    ) -> None:
        current_rbp = self.guest_rsp + 0x40
        self.memory[current_rbp] = current_rbp - 0x10
        self.memory[current_rbp + 8] = 0x8012108A1

        self.assertIsNone(
            SNAPSHOT.decode_direct_guest_wait(
                9,
                0x80121086F,
                self.guest_rsp,
                current_rbp,
                self.read_u64,
            )
        )

    def test_decodes_exact_wait_from_an_lldb_unwound_guest_frame(self) -> None:
        candidate = SNAPSHOT.decode_unwound_guest_wait(
            9,
            SNAPSHOT.GAME_THREAD_WAIT_RETURN,
            self.guest_rsp,
            self.wait_rbp,
            self.read_u64,
        )

        self.assertIsNotNone(candidate)
        assert candidate is not None
        self.assertEqual(candidate.source, "unwound-guest")
        self.assertEqual(candidate.wait_rbp, self.wait_rbp)
        self.assertEqual(candidate.event, self.event)
        self.assertEqual(candidate.native_distance, 0)

    def test_unwound_decoder_rejects_a_different_guest_frame(self) -> None:
        self.assertIsNone(
            SNAPSHOT.decode_unwound_guest_wait(
                9,
                SNAPSHOT.GAME_THREAD_WAIT_RETURN + 1,
                self.guest_rsp,
                self.wait_rbp,
                self.read_u64,
            )
        )

    def test_selects_nearest_native_import_frame(self) -> None:
        first = SNAPSHOT.decode_wait_candidate(
            self.import_frame, 9, self.native_sp, self.read_u64
        )
        assert first is not None
        second = SNAPSHOT.WaitCandidate(
            thread_id=9,
            native_sp=self.native_sp,
            import_frame=self.import_frame + 0x1000,
            guest_rsp=first.guest_rsp,
            wait_rbp=first.wait_rbp,
            outer_rbp=first.outer_rbp,
            parent_rbp=first.parent_rbp,
            event=first.event,
        )

        self.assertEqual(
            SNAPSHOT.select_wait_candidate((second, first)),
            first,
        )

    def test_rejects_equal_distance_candidates_on_different_threads(self) -> None:
        first = SNAPSHOT.decode_wait_candidate(
            self.import_frame, 9, self.native_sp, self.read_u64
        )
        assert first is not None
        second = SNAPSHOT.WaitCandidate(
            thread_id=10,
            native_sp=first.native_sp,
            import_frame=first.import_frame,
            guest_rsp=first.guest_rsp,
            wait_rbp=first.wait_rbp,
            outer_rbp=first.outer_rbp,
            parent_rbp=first.parent_rbp,
            event=first.event,
        )

        self.assertIsNone(
            SNAPSHOT.select_wait_candidate((first, second))
        )

    def _install_node(
        self,
        handle: int,
        node_address: int,
        next_handle: int,
        task: int,
        event: int,
    ) -> None:
        node_id = handle & SNAPSHOT.TASK_QUEUE_WAITER_MASK
        page = node_id >> SNAPSHOT.TASK_QUEUE_NODE_PAGE_BITS
        slot = node_id & SNAPSHOT.TASK_QUEUE_NODE_PAGE_MASK
        page_base = node_address - slot * SNAPSHOT.TASK_QUEUE_NODE_STRIDE
        self.memory[
            SNAPSHOT.TASKGRAPH_NODE_SLAB_POINTERS + page * 8
        ] = page_base
        self.memory[node_address] = next_handle
        self.memory[node_address + 8] = task
        for index in range(SNAPSHOT.TASK_WORD_COUNT):
            self.memory[task + index * 8] = 0xA0 + index
        self.memory[task + 0x18] = event

    def test_walks_both_paged_nodes_and_finds_queued_same_event_task(self) -> None:
        head = 0x4001
        tail = 0x4002
        self._install_node(
            head,
            0x7500F00018,
            tail,
            0x7500E10000,
            0x7500E99900,
        )
        self._install_node(
            tail,
            0x7500F00030,
            0,
            0x7500E20000,
            self.event,
        )

        lane = SNAPSHOT.walk_task_queue_lane(
            0, head, tail, self.event, self.read_u64
        )

        self.assertEqual(lane.termination, "tail")
        self.assertEqual(lane.classification, "queued")
        self.assertEqual(lane.match_indexes, (1,))
        self.assertEqual(lane.nodes[1].task_words[3], self.event)

    def test_head_match_is_classified_as_already_popped_dummy(self) -> None:
        head = tail = 0x4001
        self._install_node(
            head,
            0x7500F00018,
            0,
            0x7500E10000,
            self.event,
        )

        lane = SNAPSHOT.walk_task_queue_lane(
            1, head, tail, self.event, self.read_u64
        )

        self.assertEqual(lane.classification, "popped-head")
        self.assertEqual(lane.match_indexes, (0,))

    def test_allocator_scan_is_bounded_by_safe_aligned_anchors(self) -> None:
        start, end, reason = SNAPSHOT.allocator_scan_window(
            (0x7500E10000, 0x7500E80000)
        )

        self.assertEqual(reason, "safe")
        assert start is not None and end is not None
        self.assertLessEqual(
            end - start, SNAPSHOT.MAXIMUM_ALLOCATOR_SCAN_BYTES
        )
        self.assertEqual(start % SNAPSHOT.ALLOCATOR_PAGE_BYTES, 0)
        self.assertEqual(end % SNAPSHOT.ALLOCATOR_PAGE_BYTES, 0)

    def test_allocator_scan_skips_without_safe_anchors(self) -> None:
        self.assertEqual(
            SNAPSHOT.allocator_scan_window((0, 0x1234)),
            (None, None, "no-safe-task-anchors"),
        )

    def test_pure_allocator_decoder_matches_event_at_task_plus_0x18(self) -> None:
        base = 0x7500E00000
        contents = bytearray(0x400)
        contents[0x218:0x220] = self.event.to_bytes(8, "little")
        contents[0x318:0x320] = (self.event + 1).to_bytes(8, "little")

        self.assertEqual(
            SNAPSHOT.scan_aligned_task_candidates(
                base, bytes(contents), self.event
            ),
            (base + 0x200,),
        )

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

    def test_wrapper_is_attach_only_and_never_signals_target(self) -> None:
        source = PROBE_SCRIPT.read_text()

        self.assertNotIn("process launch", source)
        self.assertNotIn("process continue", source)
        self.assertNotIn("kill -", source)
        self.assertIn('process attach --pid $attach_pid', source)
        self.assertIn("REMU_DISABLE_CPU_WRITE_WATCH=1", source)
        self.assertIn('"$process_uid" != "$(id -u)"', source)
        self.assertIn('" run --timeout-seconds "', source)
        self.assertIn("remu-taskgraph-stall-detach", source)


if __name__ == "__main__":
    unittest.main()
