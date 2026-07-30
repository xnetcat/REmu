# SPDX-License-Identifier: GPL-2.0-or-later

"""Bounded LLDB trace of an Unreal RenderThread fence task.

REmu executes mapped x86-64 guest code directly.  This diagnostic therefore
uses LLDB hardware *code* breakpoints: guest bytes are never changed and there
is no software-breakpoint fallback.  Four x86 debug-register slots are rotated
through a small state machine so one ``FNullGraphTask`` can be correlated from
prerequisite release through queue publication, wake, pop, execution, and
completion.

Load this file from LLDB, then run ``remu-taskgraph-trace-start`` before
launching REmu.  ``scripts/probe-taskgraph.sh`` provides the complete bounded
invocation.  Its ``--attach PID`` mode instead starts the command after LLDB
has stopped one explicitly validated, already-running REmu process; guest
mapping is verified before any hardware breakpoint is armed.  Set
``REMU_TASKGRAPH_TRACE_PREPARE_ONLY=1`` to keep one persistent
hardware breakpoint at the prerequisite-release site and avoid dynamic
breakpoint rotation while logging every fence.  Set
``REMU_TASKGRAPH_TRACE_TARGET_FOLLOW=1`` to log only the configured BeginFence
caller chain and add persistent publish, execute, and complete breakpoints once
that task is captured.  Add ``REMU_TASKGRAPH_TRACE_QUEUE_HANDOFF=1`` to derive
the RenderThread queue when the selected fence is captured and trace that
queue's post-enqueue state, waiter registration, and event wait entry.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, Mapping, Optional

try:
    import lldb  # type: ignore[import-not-found]
except ImportError:
    # The state machine is deliberately testable without LLDB or an x86 host.
    lldb = None


MAXIMUM_HARDWARE_BREAKPOINTS = 4
DEFAULT_CYCLE_LIMIT = 16
DEFAULT_HIT_LIMIT = 4096
MAXIMUM_CYCLE_LIMIT = 64
MAXIMUM_HIT_LIMIT = 100_000
# REmu executes PS5 code directly in low canonical host mappings.  The main
# image starts at 0x8_0000_0000, while guest modules, heaps, and direct-memory
# aliases observed by the runtime remain below one tebibyte.  Native macOS
# libraries and Rust allocations are in the high 0x7f/0x7ff ranges instead.
MINIMUM_GUEST_EXECUTION_ADDRESS = 0x8_0000_0000
MAXIMUM_GUEST_EXECUTION_ADDRESS = 0x100_0000_0000
DEFAULT_TARGET_FENCE_CALLER = 0x801BE0507
DEFAULT_TARGET_FENCE_PARENT = 0x80101AF94
TASKGRAPH_SINGLETON_POINTER = 0x807BA35E0
TASKGRAPH_NODE_SLAB_POINTERS = 0x807BA8C48 + 0x88
TASK_QUEUE_WAITER_BITS = 26
TASK_QUEUE_WAITER_MASK = (1 << TASK_QUEUE_WAITER_BITS) - 1
TASK_QUEUE_NODE_PAGE_BITS = 14
TASK_QUEUE_NODE_PAGE_MASK = (1 << TASK_QUEUE_NODE_PAGE_BITS) - 1
TASK_QUEUE_NODE_STRIDE = 24

PHASE_PUBLISH = "publish"
PHASE_CONSUME = "consume"
PHASE_COMPLETE = "complete"
PHASE_DONE = "done"

PREPARE = "prepare-task"
DISPATCH = "dispatch"
PUBLISH = "publish-task"
PUBLISH_RETURN = "publish-return"
WAKE = "wake"
POP_LOCAL = "pop-local-return"
POP_WAIT = "pop-wait-return"
POP_DRAIN = "pop-drain-return"
EXECUTE = "execute"
COMPLETE = "complete"
QUEUE_REGISTER = "queue-register"
EVENT_WAIT_ENTRY = "event-wait-entry"

BREAKPOINTS = {
    PHASE_PUBLISH: {
        DISPATCH: 0x801BE1AF8,
        PUBLISH: 0x8011C6F20,
        PUBLISH_RETURN: 0x8011C6C5F,
        WAKE: 0x8011C6C71,
    },
    PHASE_CONSUME: {
        POP_LOCAL: 0x8011C6B5C,
        POP_WAIT: 0x8011C6E98,
        POP_DRAIN: 0x8011C6EEC,
        EXECUTE: 0x8011CD3F0,
    },
    PHASE_COMPLETE: {
        COMPLETE: 0x8011CD416,
    },
    PHASE_DONE: {},
}
PREPARE_ADDRESS = 0x801BE1ADF
QUEUE_HANDOFF_DISCOVERY_BREAKPOINTS = {
    PREPARE: PREPARE_ADDRESS,
}
QUEUE_HANDOFF_BREAKPOINTS = {
    PREPARE: PREPARE_ADDRESS,
    PUBLISH_RETURN: BREAKPOINTS[PHASE_PUBLISH][PUBLISH_RETURN],
    QUEUE_REGISTER: 0x8011C4C00,
    EVENT_WAIT_ENTRY: 0x801210940,
}
TASKGRAPH_BREAKPOINT_ADDRESSES = frozenset(
    {
        PREPARE_ADDRESS,
        *QUEUE_HANDOFF_BREAKPOINTS.values(),
        *(
            address
            for phase_breakpoints in BREAKPOINTS.values()
            for address in phase_breakpoints.values()
        ),
    }
)
# Rosetta occasionally reports a hardware code breakpoint as a trace stop at the
# instruction immediately after the watched instruction rather than as an LLDB
# breakpoint stop at the watched address. These fallthrough addresses are fixed
# instruction-boundary facts for the PPSA10112 code above. They are accepted only
# by the batch-mode resume filter; correlation callbacks still require the exact
# breakpoint address.
TASKGRAPH_TRACE_FALLTHROUGH_ADDRESSES = frozenset(
    {
        0x801BE1AE2,  # PREPARE dependency-count comparison completed
        0x801BE1AFB,  # DISPATCH call completed
        0x8011C6F21,  # PUBLISH entry push completed
        0x8011C6C62,  # PUBLISH_RETURN mov completed
        0x8011C6C74,  # WAKE call completed
        0x8011C6B5F,  # POP_LOCAL test completed
        0x8011C6E9B,  # POP_WAIT test completed
        0x8011C6EEF,  # POP_DRAIN test completed
        0x8011CD3F1,  # EXECUTE entry push completed
        0x8011CD41D,  # COMPLETE lea completed
        0x8011C4C06,  # waiter-registration cmpxchg/branch completed
        0x801210944,  # event wait loaded its condition address
    }
)

# These are instruction-boundary facts from the materialized PPSA10112 SELF,
# decoded by REmu's non-executing ``disassemble`` command:
#
# * PREPARE follows the atomic prerequisite-count release in
#   ``FRenderCommandFence::BeginFence``.  ``rbx`` is the task, ``r12d`` is the
#   old counter value, and ``eax`` is the release threshold that makes the task
#   immediately dispatchable.  Unlike DISPATCH, this site is reached even if a
#   prerequisite prevents the fence task from ever entering a queue.
# * DISPATCH is the indirect call after ``rsi = rbx`` (task), ``edx =
#   [rbx+8]`` (target), and ``ecx = [rbp-0x68]`` (current thread).
# * PUBLISH is a SysV entry: rdi=queue, rsi=task, edx=lane.
# * PUBLISH_RETURN is the first instruction after ``call PUBLISH`` and still
#   has its signed result in eax.  The state also requires the exact caller
#   thread and the callee-entry rsp + 8, so an unrelated publish return cannot
#   advance the trace.
# * WAKE is an indirect call after its event object is loaded into rdi.
# * POP_* are first instructions after queue-pop calls, with the task in rax.
# * EXECUTE is a SysV entry with the task in rdi.
# * COMPLETE follows ``call DispatchSubsequents``.  EXECUTE saved ``rdi`` in
#   callee-saved r14 before that call, so r14 is still the exact task there.
for _phase_breakpoints in BREAKPOINTS.values():
    assert len(_phase_breakpoints) <= MAXIMUM_HARDWARE_BREAKPOINTS

ReadU64 = Callable[[int], Optional[int]]


def _positive_limit(name: str, default: int, maximum: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        parsed = int(raw, 0)
    except ValueError:
        return default
    return parsed if 1 <= parsed <= maximum else default


def guest_address_setting(name: str, default: int) -> int:
    """Read one validated direct-guest code address from the environment."""

    raw = os.environ.get(name)
    if raw in (None, ""):
        return default
    try:
        address = int(raw, 0)
    except ValueError as error:
        raise ValueError(f"{name} is not an integer address") from error
    if not (
        MINIMUM_GUEST_EXECUTION_ADDRESS
        <= address
        < MAXIMUM_GUEST_EXECUTION_ADDRESS
    ):
        raise ValueError(f"{name} is outside the direct guest execution window")
    return address


def signed_u32(value: int) -> int:
    """Interpret eax as the signed 32-bit queue-publish result."""

    value &= (1 << 32) - 1
    return value - (1 << 32) if value & (1 << 31) else value


def is_loader_notification_symbol(symbol: Optional[str]) -> bool:
    """Return whether LLDB stopped in dyld's private image notification trap."""

    return symbol is not None and "lldb_image_notifier" in symbol


def is_taskgraph_breakpoint_address(address: int) -> bool:
    """Return whether an address belongs to this diagnostic's guest trace."""

    return address in TASKGRAPH_BREAKPOINT_ADDRESSES


def is_taskgraph_diagnostic_stop_address(address: int) -> bool:
    """Accept exact breakpoints and Rosetta's post-instruction trace stops."""

    return (
        address in TASKGRAPH_BREAKPOINT_ADDRESSES
        or address in TASKGRAPH_TRACE_FALLTHROUGH_ADDRESSES
    )


def is_guest_bad_access_stop(
    address: int,
    description: Optional[str],
    write_watch_enabled: bool = True,
) -> bool:
    """Return whether LLDB intercepted a guest fault meant for REmu's handler.

    REmu uses page protection faults for guest-memory dirty/write tracking.
    When running under LLDB, debugserver sees those Mach exceptions before
    REmu's installed signal handler.  Only pass through EXC_BAD_ACCESS stops
    whose instruction pointer is in the direct guest execution window; a host
    fault must remain stopped for diagnosis.  When the probe disables precise
    CPU write watching, no deliberate protection fault can exist, so every bad
    access remains stopped.
    """

    return (
        write_watch_enabled
        and MINIMUM_GUEST_EXECUTION_ADDRESS
        <= address
        < MAXIMUM_GUEST_EXECUTION_ADDRESS
        and description is not None
        and description.startswith("EXC_BAD_ACCESS")
    )


def describe_queue_wait_state(value: Optional[int]) -> str:
    """Decode the queue waiter bitmap and ABA generation used by UE TaskGraph."""

    if value is None:
        return "unreadable"
    return (
        f"{value:#x}/waiters={value & TASK_QUEUE_WAITER_MASK:#x}"
        f"/generation={value >> TASK_QUEUE_WAITER_BITS:#x}"
    )


def describe_task_queue_node(
    value: Optional[int],
    read_u64: ReadU64,
    target_task: int = 0,
) -> str:
    """Resolve a TaskGraph queue cursor through its page/slot node slab."""

    if value is None:
        return "unreadable"
    node_id = value & TASK_QUEUE_WAITER_MASK
    page = node_id >> TASK_QUEUE_NODE_PAGE_BITS
    slot = node_id & TASK_QUEUE_NODE_PAGE_MASK
    page_base = read_u64(TASKGRAPH_NODE_SLAB_POINTERS + page * 8)
    task = (
        read_u64(page_base + slot * TASK_QUEUE_NODE_STRIDE + 8)
        if page_base not in (None, 0)
        else None
    )
    page_base_text = (
        f"{page_base:#x}" if page_base is not None else "unreadable"
    )
    task_text = f"{task:#x}" if task is not None else "unreadable"
    is_target = (
        target_task != 0 and task is not None and task == target_task
    )
    return (
        f"{value}/id={node_id:#x}/page={page:#x}/slot={slot:#x}/"
        f"base={page_base_text}/task={task_text}/target={is_target}"
    )


def _register(registers: Mapping[str, int], name: str) -> int:
    return registers.get(name, 0) & ((1 << 64) - 1)


def persistent_breakpoint_additions(
    installed_labels: Mapping[str, object],
    requested: Mapping[str, int],
) -> dict[str, int]:
    """Return only fixed breakpoints not already installed."""

    return {
        label: address
        for label, address in requested.items()
        if label not in installed_labels
    }


@dataclass(frozen=True)
class TraceResult:
    """One state-machine observation returned to the LLDB adapter."""

    accepted: bool
    message: str | None = None


class TaskGraphTraceState:
    """Pure state machine for one address-correlated fence task at a time."""

    def __init__(
        self,
        cycle_limit: int,
        hit_limit: int,
        prepare_only: bool = False,
        target_follow: bool = False,
        target_caller: int = DEFAULT_TARGET_FENCE_CALLER,
        target_parent: int = DEFAULT_TARGET_FENCE_PARENT,
        queue_handoff: bool = False,
    ) -> None:
        if not 1 <= cycle_limit <= MAXIMUM_CYCLE_LIMIT:
            raise ValueError("cycle_limit is outside the bounded diagnostic range")
        if not 1 <= hit_limit <= MAXIMUM_HIT_LIMIT:
            raise ValueError("hit_limit is outside the bounded diagnostic range")
        self.cycle_limit = cycle_limit
        self.hit_limit = hit_limit
        self.prepare_only = prepare_only or target_follow
        self.target_follow = target_follow
        self.queue_handoff = queue_handoff
        if self.queue_handoff and not self.target_follow:
            raise ValueError("queue handoff tracing requires target follow")
        self.target_caller = target_caller
        self.target_parent = target_parent
        self.target_stages_installed = False
        self.target_capture_count = 0
        self.target_published = False
        self.target_executed = False
        self.queue_object = 0
        self.queue_event_wrapper = 0
        self.queue_event_underlying = 0
        self.target_publisher_thread = 0
        self.queue_transitions = 0
        self.phase = PHASE_PUBLISH
        self.hits = 0
        self.cycles_started = 0
        self.cycles_completed = 0
        self.task = 0
        self.completion_event = 0
        self.queue = 0
        self.publisher_thread = 0
        self.publisher_entry_rsp = 0
        self.publisher_return_rsp = 0
        self.publish_wait_state_before: int | None = None
        self.wake_event = 0
        self.pop_site = ""
        self.counter_before_release = 0
        self.release_threshold = 0
        self.conditional_dispatch_observed = False

    @property
    def done(self) -> bool:
        return self.phase == PHASE_DONE

    def active_breakpoints(self) -> Mapping[str, int]:
        if self.done:
            return BREAKPOINTS[PHASE_DONE]
        if self.queue_handoff:
            if self.queue == 0:
                return QUEUE_HANDOFF_DISCOVERY_BREAKPOINTS
            return QUEUE_HANDOFF_BREAKPOINTS
        if self.target_follow:
            breakpoints = {PREPARE: PREPARE_ADDRESS}
            if self.target_stages_installed:
                breakpoints.update(
                    {
                        PUBLISH: BREAKPOINTS[PHASE_PUBLISH][PUBLISH],
                        EXECUTE: BREAKPOINTS[PHASE_CONSUME][EXECUTE],
                        COMPLETE: BREAKPOINTS[PHASE_COMPLETE][COMPLETE],
                    }
                )
            return breakpoints
        if self.prepare_only:
            # Keep one hardware breakpoint installed for the whole run.  In
            # particular, do not make debugserver remove/re-add translated
            # guest breakpoints after each fence.
            return {PREPARE: PREPARE_ADDRESS}
        if self.phase == PHASE_PUBLISH:
            if self.task == 0:
                return {
                    PREPARE: PREPARE_ADDRESS,
                    DISPATCH: BREAKPOINTS[PHASE_PUBLISH][DISPATCH],
                }
            if self.publisher_entry_rsp == 0:
                return {
                    PREPARE: PREPARE_ADDRESS,
                    DISPATCH: BREAKPOINTS[PHASE_PUBLISH][DISPATCH],
                    PUBLISH: BREAKPOINTS[PHASE_PUBLISH][PUBLISH],
                }
            if self.publisher_return_rsp == 0:
                return {
                    PREPARE: PREPARE_ADDRESS,
                    DISPATCH: BREAKPOINTS[PHASE_PUBLISH][DISPATCH],
                    PUBLISH_RETURN: BREAKPOINTS[PHASE_PUBLISH][PUBLISH_RETURN]
                }
            return {
                PREPARE: PREPARE_ADDRESS,
                DISPATCH: BREAKPOINTS[PHASE_PUBLISH][DISPATCH],
                WAKE: BREAKPOINTS[PHASE_PUBLISH][WAKE],
            }
        if self.phase == PHASE_CONSUME and self.pop_site:
            return {
                PREPARE: PREPARE_ADDRESS,
                EXECUTE: BREAKPOINTS[PHASE_CONSUME][EXECUTE],
            }
        if self.phase == PHASE_CONSUME:
            return {
                PREPARE: PREPARE_ADDRESS,
                POP_LOCAL: BREAKPOINTS[PHASE_CONSUME][POP_LOCAL],
                POP_WAIT: BREAKPOINTS[PHASE_CONSUME][POP_WAIT],
                POP_DRAIN: BREAKPOINTS[PHASE_CONSUME][POP_DRAIN],
            }
        if self.phase == PHASE_COMPLETE:
            return {
                PREPARE: PREPARE_ADDRESS,
                COMPLETE: BREAKPOINTS[PHASE_COMPLETE][COMPLETE],
            }
        return BREAKPOINTS[self.phase]

    def _finish(self, reason: str) -> TraceResult:
        self.phase = PHASE_DONE
        return TraceResult(
            True,
            f"done reason={reason} hits={self.hits} "
            f"cycles={self.cycles_completed}/{self.cycles_started}",
        )

    def _reset_cycle(self) -> None:
        self.task = 0
        self.completion_event = 0
        self.queue = 0
        self.publisher_thread = 0
        self.publisher_entry_rsp = 0
        self.publisher_return_rsp = 0
        self.publish_wait_state_before = None
        self.wake_event = 0
        self.pop_site = ""
        self.counter_before_release = 0
        self.release_threshold = 0
        self.conditional_dispatch_observed = False

    def observe(
        self,
        label: str,
        registers: Mapping[str, int],
        thread_id: int,
        read_u64: ReadU64,
    ) -> TraceResult:
        """Consume one breakpoint hit and update the bounded trace phase."""

        if self.done:
            return TraceResult(False)
        self.hits += 1
        if self.hits > self.hit_limit:
            return self._finish("hit-limit")
        if label not in self.active_breakpoints():
            return TraceResult(False)
        if self.queue_handoff:
            return self._observe_queue_handoff(
                label,
                registers,
                thread_id,
                read_u64,
            )
        if self.target_follow:
            return self._observe_target_follow(
                label,
                registers,
                thread_id,
                read_u64,
            )
        if self.prepare_only:
            return self._observe_prepare_only(registers, read_u64)

        candidate_task = (
            _register(registers, "rbx")
            if label == PREPARE
            else _register(registers, "rsi")
            if label == DISPATCH
            else 0
        )
        superseded = None
        if (
            label in (PREPARE, DISPATCH)
            and self.task != 0
            and candidate_task != 0
            and candidate_task != self.task
        ):
            superseded = (
                f"superseded task={self.task:#x} event={self.completion_event:#x} "
                f"phase={self.phase} published={self.queue != 0} "
                f"release={self.counter_before_release:#x}/"
                f"{self.release_threshold:#x} "
                f"wake={self.wake_event:#x} pop={self.pop_site or 'not-observed'}"
            )
            self.phase = PHASE_PUBLISH
            self._reset_cycle()

        if self.phase == PHASE_PUBLISH:
            result = self._observe_publish(label, registers, thread_id, read_u64)
        elif self.phase == PHASE_CONSUME:
            result = self._observe_consume(label, registers, read_u64)
        elif self.phase == PHASE_COMPLETE:
            result = self._observe_complete(label, registers, read_u64)
        else:
            result = TraceResult(False)
        if superseded is not None and result.accepted:
            message = superseded
            if result.message is not None:
                message += f" next=({result.message})"
            return TraceResult(True, message)
        return result

    @staticmethod
    def _fence_caller_chain(
        registers: Mapping[str, int],
        read_u64: ReadU64,
    ) -> tuple[int, int]:
        frame_pointer = _register(registers, "rbp")
        caller_frame = read_u64(frame_pointer) or 0
        caller = read_u64(frame_pointer + 0x8) or 0
        parent = read_u64(caller_frame + 0x8) or 0
        return caller, parent

    def _queue_event_description(self, read_u64: ReadU64) -> str:
        event_word = (
            read_u64(self.queue_event_underlying + 0x10)
            if self.queue_event_underlying != 0
            else None
        )
        condition = (
            self.queue_event_underlying + 0x28
            if self.queue_event_underlying != 0
            else 0
        )
        return (
            f"object={self.queue_object:#x} "
            f"event_wrapper={self.queue_event_wrapper:#x} "
            f"event_underlying={self.queue_event_underlying:#x} "
            f"event_condition={condition:#x} "
            f"event_word="
            f"{event_word if event_word is not None else 'unreadable'}"
        )

    def _queue_snapshot(self, read_u64: ReadU64) -> str:
        wait_state = read_u64(self.queue + 0x1A8)
        lane_one_head = read_u64(self.queue + 0x118)
        lane_one_tail = read_u64(self.queue + 0x160)
        return (
            f"wait_state={describe_queue_wait_state(wait_state)} "
            f"lane1_head="
            f"{describe_task_queue_node(lane_one_head, read_u64, self.task)} "
            f"lane1_tail="
            f"{describe_task_queue_node(lane_one_tail, read_u64, self.task)}"
        )

    def _observe_queue_handoff(
        self,
        label: str,
        registers: Mapping[str, int],
        thread_id: int,
        read_u64: ReadU64,
    ) -> TraceResult:
        """Follow one RenderThread queue from enqueue through its event wait."""

        if label == PREPARE:
            task = _register(registers, "rbx")
            if task == 0:
                return TraceResult(False)
            task_word = read_u64(task + 0x8)
            desired_word = (
                task_word & 0xFFFFFFFF if task_word is not None else 0xFFFFFFFF
            )
            desired_thread = desired_word & 0xFF
            queue_index = (desired_word >> 8) & 1
            caller, parent = self._fence_caller_chain(registers, read_u64)
            if (
                caller != self.target_caller
                or parent != self.target_parent
                or task == self.task
            ):
                return TraceResult(False)
            singleton = read_u64(TASKGRAPH_SINGLETON_POINTER) or 0
            self.queue_object = (
                read_u64(singleton + desired_thread * 0x18 + 0x8) or 0
            )
            self.queue = (
                self.queue_object + queue_index * 0x200 + 0x38
                if self.queue_object != 0
                else 0
            )
            self.queue_event_wrapper = (
                read_u64(self.queue + 0x1F8) or 0
                if self.queue != 0
                else 0
            )
            self.queue_event_underlying = (
                read_u64(self.queue_event_wrapper + 0x10) or 0
            )
            self.task = task
            self.completion_event = read_u64(task + 0x18) or 0
            self.counter_before_release = (
                _register(registers, "r12") & 0xFFFFFFFF
            )
            self.release_threshold = (
                _register(registers, "rax") & 0xFFFFFFFF
            )
            self.target_capture_count += 1
            self.target_published = False
            self.target_publisher_thread = thread_id
            return TraceResult(
                True,
                f"target={self.target_capture_count} capture task={task:#x} "
                f"event={self.completion_event:#x} "
                f"desired={desired_thread:#x} queue_index={queue_index} "
                f"dispatchable="
                f"{self.counter_before_release == self.release_threshold} "
                f"caller={caller:#x} caller_parent={parent:#x} "
                f"publisher_thread={thread_id:#x} "
                f"taskgraph={singleton:#x} queue={self.queue:#x} "
                f"{self._queue_snapshot(read_u64)} "
                f"{self._queue_event_description(read_u64)}",
            )

        if self.queue == 0:
            return TraceResult(False)
        if label == PUBLISH_RETURN:
            queue = (
                _register(registers, "r14")
                + _register(registers, "rbx")
                + 0x38
            )
            if queue != self.queue:
                return TraceResult(False)
            self.queue_transitions += 1
            result = signed_u32(_register(registers, "rax"))
            target = (
                self.task != 0
                and not self.target_published
                and thread_id == self.target_publisher_thread
            )
            if target:
                self.target_published = True
            return TraceResult(
                True,
                f"post-enqueue transition={self.queue_transitions} "
                f"queue={queue:#x} target={target} "
                f"wake_waiter={result} "
                f"{self._queue_snapshot(read_u64)} "
                f"{self._queue_event_description(read_u64)} "
                f"thread={thread_id:#x}",
            )

        if label == QUEUE_REGISTER:
            queue = _register(registers, "r13")
            if queue != self.queue:
                return TraceResult(False)
            self.queue_transitions += 1
            registered = bool(_register(registers, "rflags") & (1 << 6))
            waiter_bit = read_u64(_register(registers, "rbp") - 0x48)
            return TraceResult(
                True,
                f"queue-register transition={self.queue_transitions} "
                f"queue={queue:#x} registered={registered} "
                f"waiter_bit="
                f"{waiter_bit if waiter_bit is not None else 'unreadable'} "
                f"expected={_register(registers, 'rax'):#x} "
                f"desired={_register(registers, 'rbx'):#x} "
                f"{self._queue_snapshot(read_u64)} "
                f"thread={thread_id:#x}",
            )

        if label != EVENT_WAIT_ENTRY:
            return TraceResult(False)
        underlying = _register(registers, "r12")
        if underlying != self.queue_event_underlying:
            return TraceResult(False)
        self.queue_transitions += 1
        return TraceResult(
            True,
            f"event-wait-entry transition={self.queue_transitions} "
            f"queue={self.queue:#x} mutex={underlying + 0x20:#x} "
            f"condition={underlying + 0x28:#x} "
            f"{self._queue_snapshot(read_u64)} "
            f"{self._queue_event_description(read_u64)} "
            f"thread={thread_id:#x}",
        )

    def _observe_target_follow(
        self,
        label: str,
        registers: Mapping[str, int],
        thread_id: int,
        read_u64: ReadU64,
    ) -> TraceResult:
        """Follow only a fence selected by its BeginFence return chain."""

        if label == PREPARE:
            if self.task != 0:
                return TraceResult(False)
            frame_pointer = _register(registers, "rbp")
            caller_frame = read_u64(frame_pointer) or 0
            caller = read_u64(frame_pointer + 0x8) or 0
            parent = read_u64(caller_frame + 0x8) or 0
            if caller != self.target_caller or parent != self.target_parent:
                return TraceResult(False)
            task = _register(registers, "rbx")
            if task == 0:
                return TraceResult(False)
            self.task = task
            self.completion_event = read_u64(task + 0x18) or 0
            self.counter_before_release = _register(registers, "r12") & 0xFFFFFFFF
            self.release_threshold = _register(registers, "rax") & 0xFFFFFFFF
            task_word = read_u64(task + 0x8)
            task_counter_after = (
                (task_word >> 32) & 0xFFFFFFFF
                if task_word is not None
                else None
            )
            self.target_stages_installed = True
            self.target_capture_count += 1
            self.target_published = False
            self.target_executed = False
            return TraceResult(
                True,
                f"target={self.target_capture_count} capture task={task:#x} "
                f"event={self.completion_event:#x} "
                f"counter_before_release={self.counter_before_release:#x} "
                f"release_threshold={self.release_threshold:#x} "
                f"task_counter_after="
                f"{task_counter_after if task_counter_after is not None else 'unreadable'} "
                f"dispatchable="
                f"{self.counter_before_release == self.release_threshold} "
                f"caller={caller:#x} caller_parent={parent:#x}",
            )

        if self.task == 0:
            return TraceResult(False)
        if label == PUBLISH:
            if _register(registers, "rsi") != self.task:
                return TraceResult(False)
            self.target_published = True
            self.queue = _register(registers, "rdi")
            return TraceResult(
                True,
                f"target={self.target_capture_count} publish "
                f"task={self.task:#x} event={self.completion_event:#x} "
                f"queue={self.queue:#x} "
                f"priority={_register(registers, 'rdx') & 0xffffffff} "
                f"thread={thread_id:#x} "
                f"wait_state={describe_queue_wait_state(read_u64(self.queue + 0x1A8))}",
            )
        if label == EXECUTE:
            if _register(registers, "rdi") != self.task:
                return TraceResult(False)
            self.target_executed = True
            return TraceResult(
                True,
                f"target={self.target_capture_count} execute "
                f"task={self.task:#x} event={self.completion_event:#x} "
                f"thread={thread_id:#x}",
            )
        if label != COMPLETE or _register(registers, "r14") != self.task:
            return TraceResult(False)
        event_words = [
            value
            for index in range(12)
            if (
                value := read_u64(self.completion_event + index * 8)
            )
            is not None
        ]
        task = self.task
        event = self.completion_event
        capture = self.target_capture_count
        published = self.target_published
        executed = self.target_executed
        self._reset_cycle()
        return TraceResult(
            True,
            f"target={capture} complete task={task:#x} event={event:#x} "
            f"published={published} executed={executed} "
            f"event_words={[hex(value) for value in event_words]}",
        )

    def _observe_prepare_only(
        self,
        registers: Mapping[str, int],
        read_u64: ReadU64,
    ) -> TraceResult:
        """Record one fence without changing the persistent breakpoint set."""

        task = _register(registers, "rbx")
        if task == 0:
            return TraceResult(False)
        event = read_u64(task + 0x18) or 0
        counter_before_release = _register(registers, "r12") & 0xFFFFFFFF
        release_threshold = _register(registers, "rax") & 0xFFFFFFFF
        task_word = read_u64(task + 0x8)
        task_counter_after = (
            (task_word >> 32) & 0xFFFFFFFF
            if task_word is not None
            else None
        )
        frame_pointer = _register(registers, "rbp")
        caller_frame = read_u64(frame_pointer) or 0
        caller = read_u64(frame_pointer + 0x8) or 0
        caller_parent = read_u64(caller_frame + 0x8) or 0
        prerequisite_list = _register(registers, "r15")
        count_word = (
            read_u64(prerequisite_list + 0x28)
            if prerequisite_list != 0
            else None
        )
        prerequisite_count = (
            signed_u32(count_word)
            if count_word is not None
            else 0
        )
        storage = (
            read_u64(prerequisite_list + 0x20)
            if prerequisite_list != 0
            else None
        )
        storage_base = (
            storage
            if storage not in (None, 0)
            else prerequisite_list
        )
        prerequisite_events = []
        if counter_before_release != release_threshold and storage_base != 0:
            for index in range(min(max(prerequisite_count, 0), 16)):
                prerequisite = read_u64(storage_base + index * 8) or 0
                if prerequisite == 0:
                    continue
                words = [
                    value
                    for word_index in range(4)
                    if (
                        value := read_u64(
                            prerequisite + word_index * 8
                        )
                    )
                    is not None
                ]
                prerequisite_events.append(
                    f"{prerequisite:#x}:{[hex(value) for value in words]}"
                )
        return TraceResult(
            True,
            f"sample={self.hits} prepare-only task={task:#x} "
            f"event={event:#x} "
            f"counter_before_release={counter_before_release:#x} "
            f"release_threshold={release_threshold:#x} "
            f"task_counter_after="
            f"{task_counter_after if task_counter_after is not None else 'unreadable'} "
            f"dispatchable={counter_before_release == release_threshold} "
            f"caller={caller:#x} caller_parent={caller_parent:#x} "
            f"prerequisite_list={prerequisite_list:#x} "
            f"prerequisite_count={prerequisite_count} "
            f"prerequisite_storage={storage_base:#x} "
            f"prerequisite_events={prerequisite_events}",
        )

    def _observe_publish(
        self,
        label: str,
        registers: Mapping[str, int],
        thread_id: int,
        read_u64: ReadU64,
    ) -> TraceResult:
        if label == PREPARE:
            if self.task != 0:
                return TraceResult(False)
            if self.cycles_started >= self.cycle_limit:
                return self._finish("cycle-limit")
            task = _register(registers, "rbx")
            if task == 0:
                return TraceResult(False)
            counter_before_release = _register(registers, "r12") & 0xFFFFFFFF
            release_threshold = _register(registers, "rax") & 0xFFFFFFFF
            self.task = task
            self.completion_event = read_u64(task + 0x18) or 0
            self.counter_before_release = counter_before_release
            self.release_threshold = release_threshold
            self.cycles_started += 1
            return TraceResult(
                True,
                f"cycle={self.cycles_started} prepare task={task:#x} "
                f"event={self.completion_event:#x} "
                f"counter_before_release={self.counter_before_release:#x} "
                f"release_threshold={self.release_threshold:#x} "
                "dispatchable="
                f"{self.counter_before_release == self.release_threshold}",
            )

        if label == DISPATCH:
            task = _register(registers, "rsi")
            if self.task != 0:
                if task != self.task:
                    return TraceResult(False)
                self.conditional_dispatch_observed = True
                return TraceResult(
                    True,
                    f"cycle={self.cycles_started} conditional-dispatch "
                    f"task={task:#x} event={self.completion_event:#x} "
                    f"target={_register(registers, 'rdx') & 0xffffffff:#x} "
                    f"current={_register(registers, 'rcx') & 0xffffffff:#x}",
                )
            if self.cycles_started >= self.cycle_limit:
                return self._finish("cycle-limit")
            if task == 0:
                return TraceResult(False)
            self.task = task
            self.completion_event = read_u64(task + 0x18) or 0
            self.cycles_started += 1
            return TraceResult(
                True,
                f"cycle={self.cycles_started} dispatch task={task:#x} "
                f"event={self.completion_event:#x} "
                f"target={_register(registers, 'rdx') & 0xffffffff:#x} "
                f"current={_register(registers, 'rcx') & 0xffffffff:#x}",
            )

        if self.task == 0:
            return TraceResult(False)
        if label == PUBLISH:
            if _register(registers, "rsi") != self.task:
                return TraceResult(False)
            self.queue = _register(registers, "rdi")
            self.publisher_thread = thread_id
            self.publisher_entry_rsp = _register(registers, "rsp")
            self.publish_wait_state_before = read_u64(self.queue + 0x1A8)
            return TraceResult(
                True,
                f"cycle={self.cycles_started} publish task={self.task:#x} "
                f"queue={self.queue:#x} "
                f"priority={_register(registers, 'rdx') & 0xffffffff} "
                f"entry_rsp={self.publisher_entry_rsp:#x} "
                "wait_state_before="
                f"{describe_queue_wait_state(self.publish_wait_state_before)}",
            )

        if label == PUBLISH_RETURN:
            return_rsp = _register(registers, "rsp")
            if (
                self.publisher_thread == 0
                or self.publisher_entry_rsp == 0
                or thread_id != self.publisher_thread
                or return_rsp != (self.publisher_entry_rsp + 8) & ((1 << 64) - 1)
            ):
                return TraceResult(False)
            self.publisher_return_rsp = return_rsp
            wake_result = signed_u32(_register(registers, "rax"))
            wait_state_after = read_u64(self.queue + 0x1A8)
            message = (
                f"cycle={self.cycles_started} publish-return task={self.task:#x} "
                f"result={wake_result} "
                f"wait_state_before="
                f"{describe_queue_wait_state(self.publish_wait_state_before)} "
                f"wait_state_after={describe_queue_wait_state(wait_state_after)}"
            )
            if wake_result < 0:
                self.phase = PHASE_CONSUME
                message += " wake=not-required"
            else:
                message += " wake=required"
            return TraceResult(True, message)

        if label == WAKE:
            if (
                self.publisher_thread == 0
                or self.publisher_return_rsp == 0
                or thread_id != self.publisher_thread
                or _register(registers, "rsp") != self.publisher_return_rsp
            ):
                return TraceResult(False)
            self.wake_event = _register(registers, "rdi")
            self.phase = PHASE_CONSUME
            return TraceResult(
                True,
                f"cycle={self.cycles_started} wake task={self.task:#x} "
                f"event={self.wake_event:#x}",
            )
        return TraceResult(False)

    def _observe_consume(
        self,
        label: str,
        registers: Mapping[str, int],
        read_u64: ReadU64,
    ) -> TraceResult:
        if label in (POP_LOCAL, POP_WAIT, POP_DRAIN):
            if _register(registers, "rax") != self.task:
                return TraceResult(False)
            self.pop_site = label
            return TraceResult(
                True,
                f"cycle={self.cycles_started} pop task={self.task:#x} "
                f"site={label} wait_state="
                f"{describe_queue_wait_state(read_u64(self.queue + 0x1A8))}",
            )
        if label == EXECUTE:
            if _register(registers, "rdi") != self.task:
                return TraceResult(False)
            self.phase = PHASE_COMPLETE
            return TraceResult(
                True,
                f"cycle={self.cycles_started} execute task={self.task:#x} "
                f"pop={self.pop_site or 'not-observed'}",
            )
        return TraceResult(False)

    def _observe_complete(
        self,
        label: str,
        registers: Mapping[str, int],
        read_u64: ReadU64,
    ) -> TraceResult:
        if label != COMPLETE or _register(registers, "r14") != self.task:
            return TraceResult(False)
        event_words = [
            value
            for index in range(12)
            if (value := read_u64(self.completion_event + index * 8)) is not None
        ]
        task = self.task
        event = self.completion_event
        cycle = self.cycles_started
        self.cycles_completed += 1
        message = (
            f"cycle={cycle} complete task={task:#x} event={event:#x} "
            f"event_words={[hex(value) for value in event_words]}"
        )
        self._reset_cycle()
        if self.cycles_completed >= self.cycle_limit:
            self.phase = PHASE_DONE
            message += (
                f" done reason=cycle-limit hits={self.hits} "
                f"cycles={self.cycles_completed}/{self.cycles_started}"
            )
        else:
            self.phase = PHASE_PUBLISH
        return TraceResult(True, message)


class LldbTaskGraphTrace:
    """LLDB adapter that owns only enabled hardware guest breakpoints."""

    REGISTER_NAMES = (
        "rax",
        "rbx",
        "rcx",
        "rdx",
        "rsi",
        "rdi",
        "r8",
        "r9",
        "r12",
        "r13",
        "r14",
        "r15",
        "rbp",
        "rsp",
        "rflags",
    )

    def __init__(self, target) -> None:
        self.target = target
        target_follow = (
            os.environ.get("REMU_TASKGRAPH_TRACE_TARGET_FOLLOW") == "1"
        )
        queue_handoff = (
            os.environ.get("REMU_TASKGRAPH_TRACE_QUEUE_HANDOFF") == "1"
        )
        self.state = TaskGraphTraceState(
            _positive_limit(
                "REMU_TASKGRAPH_TRACE_CYCLES",
                DEFAULT_CYCLE_LIMIT,
                MAXIMUM_CYCLE_LIMIT,
            ),
            _positive_limit(
                "REMU_TASKGRAPH_TRACE_HITS",
                DEFAULT_HIT_LIMIT,
                MAXIMUM_HIT_LIMIT,
            ),
            prepare_only=(
                os.environ.get("REMU_TASKGRAPH_TRACE_PREPARE_ONLY") == "1"
            ),
            target_follow=target_follow,
            target_caller=guest_address_setting(
                "REMU_TASKGRAPH_TRACE_TARGET_CALLER",
                DEFAULT_TARGET_FENCE_CALLER,
            ),
            target_parent=guest_address_setting(
                "REMU_TASKGRAPH_TRACE_TARGET_PARENT",
                DEFAULT_TARGET_FENCE_PARENT,
            ),
            queue_handoff=queue_handoff,
        )
        self.enabled_breakpoints = {}

    def _command(self, command: str):
        result = lldb.SBCommandReturnObject()
        self.target.GetDebugger().GetCommandInterpreter().HandleCommand(command, result)
        return result

    def _disable_active(self) -> None:
        for breakpoint in self.enabled_breakpoints.values():
            breakpoint.SetEnabled(False)
        self.enabled_breakpoints = {}

    def _install_breakpoint(self, label: str, address: int):
        result = self._command(
            f"breakpoint set --hardware --address {address:#x}"
        )
        if not result.Succeeded():
            print(
                "[REMu][LLDB][TASKGRAPH] hardware breakpoint install failed "
                f"label={label} address={address:#x}: "
                f"{result.GetError().strip()}",
                flush=True,
            )
            return None
        breakpoint = self.target.GetBreakpointAtIndex(
            self.target.GetNumBreakpoints() - 1
        )
        callback = self._command(
            "breakpoint command add --script-type python "
            "--python-function lldb_taskgraph_trace.trace_breakpoint "
            f"{breakpoint.GetID()}"
        )
        if not callback.Succeeded():
            breakpoint.SetEnabled(False)
            print(
                "[REMu][LLDB][TASKGRAPH] callback install failed "
                f"label={label} address={address:#x}: "
                f"{callback.GetError().strip()}",
                flush=True,
            )
            return None
        return breakpoint

    def arm_phase(self) -> bool:
        requested = self.state.active_breakpoints()
        if len(requested) > MAXIMUM_HARDWARE_BREAKPOINTS:
            if not self.state.target_follow:
                self._disable_active()
            print(
                "[REMu][LLDB][TASKGRAPH] refusing phase with more than four breakpoints",
                flush=True,
            )
            return False
        if self.state.target_follow and not self.state.queue_handoff:
            added = []
            additions = persistent_breakpoint_additions(
                self.enabled_breakpoints,
                requested,
            )
            for label, address in additions.items():
                breakpoint = self._install_breakpoint(label, address)
                if breakpoint is None:
                    self.state.phase = PHASE_DONE
                    return False
                self.enabled_breakpoints[label] = breakpoint
                added.append(label)
            print(
                "[REMu][LLDB][TASKGRAPH] persistent breakpoints="
                f"{','.join(self.enabled_breakpoints) or 'none'} "
                f"added={','.join(added) or 'none'}",
                flush=True,
            )
            return True
        self._disable_active()
        installed = {}
        for label, address in requested.items():
            breakpoint = self._install_breakpoint(label, address)
            if breakpoint is None:
                for installed_breakpoint in installed.values():
                    installed_breakpoint.SetEnabled(False)
                self.state.phase = PHASE_DONE
                self.enabled_breakpoints = {}
                return False
            installed[label] = breakpoint
        self.enabled_breakpoints = installed
        labels = ",".join(requested)
        print(
            f"[REMu][LLDB][TASKGRAPH] armed phase={self.state.phase} "
            f"breakpoints={labels or 'none'}",
            flush=True,
        )
        return True

    @staticmethod
    def _registers(frame) -> dict[str, int]:
        return {
            name: frame.FindRegister(name).GetValueAsUnsigned()
            for name in LldbTaskGraphTrace.REGISTER_NAMES
        }

    @staticmethod
    def _read_u64(process, address: int) -> int | None:
        if address == 0:
            return None
        error = lldb.SBError()
        contents = process.ReadMemory(address, 8, error)
        if error.Fail() or len(contents) != 8:
            return None
        return int.from_bytes(contents, "little")

    def breakpoint_hit(self, frame, breakpoint_location) -> bool:
        address = breakpoint_location.GetAddress().GetLoadAddress(self.target)
        labels = {
            breakpoint_address: label
            for label, breakpoint_address in self.state.active_breakpoints().items()
        }
        label = labels.get(address)
        if label is None:
            return False
        thread = frame.GetThread()
        process = thread.GetProcess()
        registers = self._registers(frame)
        previous_breakpoints = dict(self.state.active_breakpoints())
        result = self.state.observe(
            label,
            registers,
            thread.GetThreadID(),
            lambda guest_address: self._read_u64(process, guest_address),
        )
        if result.accepted and result.message is not None:
            register_text = ",".join(
                f"{name}={value:#x}" for name, value in registers.items()
            )
            print(
                f"[REMu][LLDB][TASKGRAPH] {result.message} "
                f"thread={thread.GetName()!r}/{thread.GetThreadID():#x} "
                f"pc={address:#x} regs={register_text}",
                flush=True,
            )
        if dict(self.state.active_breakpoints()) != previous_breakpoints:
            self.arm_phase()
        return False


SESSION: LldbTaskGraphTrace | None = None


def install_taskgraph_trace(frame, breakpoint_location, _internal_dict):
    """Host-import callback that arms guest hardware breakpoints after mapping."""

    global SESSION
    if SESSION is not None:
        breakpoint_location.GetBreakpoint().SetEnabled(False)
        return False
    target = frame.GetThread().GetProcess().GetTarget()
    SESSION = LldbTaskGraphTrace(target)
    breakpoint_location.GetBreakpoint().SetEnabled(False)
    print(
        "[REMu][LLDB][TASKGRAPH] guest mapping reached; "
        f"cycle_limit={SESSION.state.cycle_limit} "
        f"hit_limit={SESSION.state.hit_limit} "
        f"prepare_only={SESSION.state.prepare_only} "
        f"target_follow={SESSION.state.target_follow} "
        f"queue_handoff={SESSION.state.queue_handoff} "
        f"target_caller={SESSION.state.target_caller:#x} "
        f"target_parent={SESSION.state.target_parent:#x}",
        flush=True,
    )
    SESSION.arm_phase()
    return False


def _direct_attach_error(target) -> Optional[str]:
    """Validate that a stopped attached process already has guest code mapped."""

    process = target.GetProcess()
    if not process.IsValid():
        return "no valid process is attached"
    if process.GetState() != lldb.eStateStopped:
        return "attached process must be stopped before arming guest breakpoints"
    error = lldb.SBError()
    contents = process.ReadMemory(PREPARE_ADDRESS, 1, error)
    if error.Fail() or len(contents) != 1:
        return (
            "attached process has not mapped the TaskGraph guest code at "
            f"{PREPARE_ADDRESS:#x}"
        )
    return None


def _arm_direct_attach(target, result) -> None:
    """Arm guest hardware breakpoints in a validated late-attach target."""

    global SESSION
    if SESSION is not None:
        result.SetError("TaskGraph trace is already active")
        return
    error = _direct_attach_error(target)
    if error is not None:
        result.SetError(error)
        return
    SESSION = LldbTaskGraphTrace(target)
    print(
        "[REMu][LLDB][TASKGRAPH] attached guest mapping verified; "
        f"pid={target.GetProcess().GetProcessID()} "
        f"cycle_limit={SESSION.state.cycle_limit} "
        f"hit_limit={SESSION.state.hit_limit} "
        f"prepare_only={SESSION.state.prepare_only} "
        f"target_follow={SESSION.state.target_follow} "
        f"queue_handoff={SESSION.state.queue_handoff} "
        f"target_caller={SESSION.state.target_caller:#x} "
        f"target_parent={SESSION.state.target_parent:#x}",
        flush=True,
    )
    if not SESSION.arm_phase():
        SESSION = None
        result.SetError("could not arm TaskGraph hardware breakpoints")


def trace_breakpoint(frame, breakpoint_location, _internal_dict):
    """Shared callback for every guest hardware breakpoint."""

    if SESSION is None:
        return False
    return SESSION.breakpoint_hit(frame, breakpoint_location)


def start_taskgraph_trace(debugger, _command, result, _internal_dict):
    """Arm a late attach directly, or seed launch mode until guest mapping."""

    if os.environ.get("REMU_TASKGRAPH_TRACE_TARGET_FOLLOW") == "1":
        try:
            guest_address_setting(
                "REMU_TASKGRAPH_TRACE_TARGET_CALLER",
                DEFAULT_TARGET_FENCE_CALLER,
            )
            guest_address_setting(
                "REMU_TASKGRAPH_TRACE_TARGET_PARENT",
                DEFAULT_TARGET_FENCE_PARENT,
            )
        except ValueError as error:
            result.SetError(str(error))
            return
    if (
        os.environ.get("REMU_TASKGRAPH_TRACE_QUEUE_HANDOFF") == "1"
        and os.environ.get("REMU_TASKGRAPH_TRACE_TARGET_FOLLOW") != "1"
    ):
        result.SetError(
            "REMU_TASKGRAPH_TRACE_QUEUE_HANDOFF requires "
            "REMU_TASKGRAPH_TRACE_TARGET_FOLLOW=1"
        )
        return
    target = debugger.GetSelectedTarget()
    if os.environ.get("REMU_TASKGRAPH_TRACE_ATTACH") == "1":
        _arm_direct_attach(target, result)
        return
    seed = target.BreakpointCreateByRegex("remu_runtime::import_dispatch_entry")
    if not seed.IsValid() or seed.GetNumLocations() == 0:
        result.SetError(
            "could not resolve remu_runtime::import_dispatch_entry in the selected target"
        )
        return
    seed.SetScriptCallbackFunction(
        "lldb_taskgraph_trace.install_taskgraph_trace"
    )
    result.AppendMessage(
        "[REMu][LLDB][TASKGRAPH] seed breakpoint installed on host import dispatch"
    )


def resume_diagnostic_stops(debugger, _command, result, _internal_dict):
    """Continue diagnostic traps and guest faults owned by REmu's handler."""

    target = debugger.GetSelectedTarget()
    process = target.GetProcess()
    write_watch_enabled = os.environ.get("REMU_DISABLE_CPU_WRITE_WATCH") != "1"
    resumed = 0
    while process.GetState() == lldb.eStateStopped:
        diagnostic_stop = False
        stop_kind = ""
        for thread in process:
            frame = thread.GetFrameAtIndex(0)
            reason = thread.GetStopReason()
            if reason == lldb.eStopReasonTrace:
                symbol = frame.GetFunctionName()
                if symbol is None and frame.GetSymbol().IsValid():
                    symbol = frame.GetSymbol().GetName()
                if is_loader_notification_symbol(symbol):
                    diagnostic_stop = True
                    stop_kind = "dyld-loader"
                    break
            if reason == lldb.eStopReasonBreakpoint:
                address = frame.GetPCAddress().GetLoadAddress(target)
                if is_taskgraph_breakpoint_address(address):
                    diagnostic_stop = True
                    stop_kind = f"taskgraph-{address:#x}"
                    break
            if reason == lldb.eStopReasonTrace:
                address = frame.GetPCAddress().GetLoadAddress(target)
                if is_taskgraph_diagnostic_stop_address(address):
                    diagnostic_stop = True
                    stop_kind = f"taskgraph-trace-{address:#x}"
                    break
            if reason == lldb.eStopReasonException:
                address = frame.GetPCAddress().GetLoadAddress(target)
                description = thread.GetStopDescription(256)
                if is_guest_bad_access_stop(
                    address,
                    description,
                    write_watch_enabled,
                ):
                    diagnostic_stop = True
                    stop_kind = f"guest-bad-access-{address:#x}"
                    break
        if not diagnostic_stop:
            break
        resumed += 1
        print(
            "[REMu][LLDB][TASKGRAPH] resuming diagnostic stop "
            f"count={resumed} kind={stop_kind}",
            flush=True,
        )
        error = process.Continue()
        if error.Fail():
            result.SetError(
                "could not resume diagnostic stop: "
                f"{error.GetCString()}"
            )
            return
    if resumed:
        result.AppendMessage(
            "[REMu][LLDB][TASKGRAPH] resumed "
            f"{resumed} diagnostic stop(s)"
        )


def resume_loader_notifications(debugger, command, result, internal_dict):
    """Backward-compatible alias for the diagnostic resume command."""

    resume_diagnostic_stops(debugger, command, result, internal_dict)


def detach_taskgraph_trace(debugger, _command, result, _internal_dict):
    """Detach and resume only a still-live process attached by the diagnostic."""

    target = debugger.GetSelectedTarget()
    process = target.GetProcess()
    if not process.IsValid():
        result.AppendMessage(
            "[REMu][LLDB][TASKGRAPH] detach skipped: no valid process"
        )
        return
    if process.GetState() in (lldb.eStateExited, lldb.eStateDetached):
        result.AppendMessage(
            "[REMu][LLDB][TASKGRAPH] detach skipped: process already ended"
        )
        return
    error = process.Detach(False)
    if error.Fail():
        result.SetError(
            "could not detach TaskGraph diagnostic: "
            f"{error.GetCString()}"
        )
        return
    result.AppendMessage(
        "[REMu][LLDB][TASKGRAPH] detached without terminating target"
    )


def __lldb_init_module(debugger, _internal_dict):
    if lldb is None:
        return
    debugger.HandleCommand(
        "command script add "
        "-f lldb_taskgraph_trace.start_taskgraph_trace "
        "remu-taskgraph-trace-start"
    )
    debugger.HandleCommand(
        "command script add "
        "-f lldb_taskgraph_trace.resume_diagnostic_stops "
        "remu-taskgraph-resume-diagnostic-stops"
    )
    debugger.HandleCommand(
        "command script add "
        "-f lldb_taskgraph_trace.detach_taskgraph_trace "
        "remu-taskgraph-trace-detach"
    )
