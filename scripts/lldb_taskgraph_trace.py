# SPDX-License-Identifier: GPL-2.0-or-later

"""Bounded LLDB trace of an Unreal RenderThread fence task.

REmu executes mapped x86-64 guest code directly.  This diagnostic therefore
uses LLDB hardware *code* breakpoints: guest bytes are never changed and there
is no software-breakpoint fallback.  Four x86 debug-register slots are rotated
through a small state machine so one ``FNullGraphTask`` can be correlated from
dispatch through queue publication, wake, pop, execution, and completion.

Load this file from LLDB, then run ``remu-taskgraph-trace-start`` before
launching REmu.  ``scripts/probe-taskgraph.sh`` provides the complete bounded
invocation.
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

PHASE_PUBLISH = "publish"
PHASE_CONSUME = "consume"
PHASE_COMPLETE = "complete"
PHASE_DONE = "done"

DISPATCH = "dispatch"
PUBLISH = "publish-task"
PUBLISH_RETURN = "publish-return"
WAKE = "wake"
POP_LOCAL = "pop-local-return"
POP_WAIT = "pop-wait-return"
POP_DRAIN = "pop-drain-return"
EXECUTE = "execute"
COMPLETE = "complete"

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

# These are instruction-boundary facts from the materialized PPSA10112 SELF,
# decoded by REmu's non-executing ``disassemble`` command:
#
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


def signed_u32(value: int) -> int:
    """Interpret eax as the signed 32-bit queue-publish result."""

    value &= (1 << 32) - 1
    return value - (1 << 32) if value & (1 << 31) else value


def is_loader_notification_symbol(symbol: Optional[str]) -> bool:
    """Return whether LLDB stopped in dyld's private image notification trap."""

    return symbol is not None and "lldb_image_notifier" in symbol


def _register(registers: Mapping[str, int], name: str) -> int:
    return registers.get(name, 0) & ((1 << 64) - 1)


@dataclass(frozen=True)
class TraceResult:
    """One state-machine observation returned to the LLDB adapter."""

    accepted: bool
    message: str | None = None


class TaskGraphTraceState:
    """Pure state machine for one address-correlated fence task at a time."""

    def __init__(self, cycle_limit: int, hit_limit: int) -> None:
        if not 1 <= cycle_limit <= MAXIMUM_CYCLE_LIMIT:
            raise ValueError("cycle_limit is outside the bounded diagnostic range")
        if not 1 <= hit_limit <= MAXIMUM_HIT_LIMIT:
            raise ValueError("hit_limit is outside the bounded diagnostic range")
        self.cycle_limit = cycle_limit
        self.hit_limit = hit_limit
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
        self.wake_event = 0
        self.pop_site = ""

    @property
    def done(self) -> bool:
        return self.phase == PHASE_DONE

    def active_breakpoints(self) -> Mapping[str, int]:
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
        self.wake_event = 0
        self.pop_site = ""

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

        if self.phase == PHASE_PUBLISH:
            return self._observe_publish(label, registers, thread_id, read_u64)
        if self.phase == PHASE_CONSUME:
            return self._observe_consume(label, registers)
        if self.phase == PHASE_COMPLETE:
            return self._observe_complete(label, registers, read_u64)
        return TraceResult(False)

    def _observe_publish(
        self,
        label: str,
        registers: Mapping[str, int],
        thread_id: int,
        read_u64: ReadU64,
    ) -> TraceResult:
        if label == DISPATCH:
            if self.task != 0:
                return TraceResult(False)
            if self.cycles_started >= self.cycle_limit:
                return self._finish("cycle-limit")
            task = _register(registers, "rsi")
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
            return TraceResult(
                True,
                f"cycle={self.cycles_started} publish task={self.task:#x} "
                f"queue={self.queue:#x} lane={_register(registers, 'rdx') & 0xffffffff} "
                f"entry_rsp={self.publisher_entry_rsp:#x}",
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
            message = (
                f"cycle={self.cycles_started} publish-return task={self.task:#x} "
                f"result={wake_result}"
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
        self, label: str, registers: Mapping[str, int]
    ) -> TraceResult:
        if label in (POP_LOCAL, POP_WAIT, POP_DRAIN):
            if _register(registers, "rax") != self.task:
                return TraceResult(False)
            self.pop_site = label
            return TraceResult(
                True,
                f"cycle={self.cycles_started} pop task={self.task:#x} site={label}",
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
        "r14",
        "r15",
        "rbp",
        "rsp",
    )

    def __init__(self, target) -> None:
        self.target = target
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
        )
        self.enabled_breakpoints = []

    def _command(self, command: str):
        result = lldb.SBCommandReturnObject()
        self.target.GetDebugger().GetCommandInterpreter().HandleCommand(command, result)
        return result

    def _disable_active(self) -> None:
        for breakpoint in self.enabled_breakpoints:
            breakpoint.SetEnabled(False)
        self.enabled_breakpoints = []

    def arm_phase(self) -> bool:
        requested = self.state.active_breakpoints()
        if len(requested) > MAXIMUM_HARDWARE_BREAKPOINTS:
            self._disable_active()
            print(
                "[REMu][LLDB][TASKGRAPH] refusing phase with more than four breakpoints",
                flush=True,
            )
            return False
        self._disable_active()
        installed = []
        for label, address in requested.items():
            result = self._command(
                f"breakpoint set --hardware --address {address:#x}"
            )
            if not result.Succeeded():
                for breakpoint in installed:
                    breakpoint.SetEnabled(False)
                print(
                    "[REMu][LLDB][TASKGRAPH] hardware breakpoint install failed "
                    f"label={label} address={address:#x}: "
                    f"{result.GetError().strip()}",
                    flush=True,
                )
                self.state.phase = PHASE_DONE
                self.enabled_breakpoints = []
                return False
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
                for installed_breakpoint in installed:
                    installed_breakpoint.SetEnabled(False)
                print(
                    "[REMu][LLDB][TASKGRAPH] callback install failed "
                    f"label={label} address={address:#x}: "
                    f"{callback.GetError().strip()}",
                    flush=True,
                )
                self.state.phase = PHASE_DONE
                self.enabled_breakpoints = []
                return False
            installed.append(breakpoint)
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
        previous_phase = self.state.phase
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
        if self.state.phase != previous_phase:
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
        f"cycle_limit={SESSION.state.cycle_limit} hit_limit={SESSION.state.hit_limit}",
        flush=True,
    )
    SESSION.arm_phase()
    return False


def trace_breakpoint(frame, breakpoint_location, _internal_dict):
    """Shared callback for every dynamically armed guest hardware breakpoint."""

    if SESSION is None:
        return False
    return SESSION.breakpoint_hit(frame, breakpoint_location)


def start_taskgraph_trace(debugger, _command, result, _internal_dict):
    """Install one host-code seed breakpoint; guest breakpoints are armed later."""

    target = debugger.GetSelectedTarget()
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


def resume_loader_notifications(debugger, _command, result, _internal_dict):
    """Continue only LLDB's spurious dyld image-notification trace stops."""

    process = debugger.GetSelectedTarget().GetProcess()
    resumed = 0
    while process.GetState() == lldb.eStateStopped:
        loader_notification = False
        for thread in process:
            if thread.GetStopReason() != lldb.eStopReasonTrace:
                continue
            frame = thread.GetFrameAtIndex(0)
            symbol = frame.GetFunctionName()
            if symbol is None and frame.GetSymbol().IsValid():
                symbol = frame.GetSymbol().GetName()
            if is_loader_notification_symbol(symbol):
                loader_notification = True
                break
        if not loader_notification:
            break
        resumed += 1
        print(
            "[REMu][LLDB][TASKGRAPH] resuming dyld loader notification "
            f"count={resumed}",
            flush=True,
        )
        error = process.Continue()
        if error.Fail():
            result.SetError(
                "could not resume dyld loader notification: "
                f"{error.GetCString()}"
            )
            return
    if resumed:
        result.AppendMessage(
            "[REMu][LLDB][TASKGRAPH] resumed "
            f"{resumed} dyld loader notification(s)"
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
        "-f lldb_taskgraph_trace.resume_loader_notifications "
        "remu-taskgraph-resume-loader-notifications"
    )
