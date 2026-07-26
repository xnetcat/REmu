"""Catch an invalid link before Unreal's small-block allocator dereferences it.

The guest executable is mapped as raw x86-64 memory, so LLDB cannot resolve a
breakpoint in it until REmu has started dispatching imports.  The first callback
installs the raw guest breakpoint and disables itself.  The second callback
filters ordinary free-list pops and stops only when the next link is neither
null nor a plausible guest pointer.
"""

import lldb


ALLOCATOR_POP_LINK_COMMITTED = 0x801241009
# REmu maps the PS5 main image at 0x8_0000_0000.  The extra zero in the
# previous cutoff classified every normal Silent Hill heap link as corrupt and
# made the diagnostic stop on its first allocator pop.
MINIMUM_GUEST_POINTER = 0x8_0000_0000


def install_allocator_watch(frame, breakpoint_location, _internal_dict):
    target = frame.GetThread().GetProcess().GetTarget()
    # Software breakpoints in Rosetta-translated guest code leave a follow-up
    # single-step trap after an auto-continued callback.  That trap stops the
    # whole process even when the free-list link is valid.  Requiring a hardware
    # breakpoint avoids modifying translated guest instructions and lets the
    # callback filter the hot path normally.
    interpreter = target.GetDebugger().GetCommandInterpreter()
    result = lldb.SBCommandReturnObject()
    interpreter.HandleCommand(
        f"breakpoint set --hardware --address {ALLOCATOR_POP_LINK_COMMITTED:#x}",
        result,
    )
    if not result.Succeeded():
        print(
            f"[REMu][LLDB] failed to install allocator watch: "
            f"{result.GetError().strip()}",
            flush=True,
        )
        return True
    guest_breakpoint = target.GetBreakpointAtIndex(target.GetNumBreakpoints() - 1)
    result = lldb.SBCommandReturnObject()
    interpreter.HandleCommand(
        "breakpoint command add --script-type python "
        "--python-function lldb_allocator_pop_watch.stop_on_invalid_link "
        f"{guest_breakpoint.GetID()}",
        result,
    )
    if not result.Succeeded():
        print(
            f"[REMu][LLDB] failed to attach allocator callback: "
            f"{result.GetError().strip()}",
            flush=True,
        )
        return True
    breakpoint_location.GetBreakpoint().SetEnabled(False)
    print(
        f"[REMu][LLDB] watching allocator pop at "
        f"0x{ALLOCATOR_POP_LINK_COMMITTED:x}",
        flush=True,
    )
    return False


def _read_hex(process, address, byte_count):
    error = lldb.SBError()
    contents = process.ReadMemory(address, byte_count, error)
    if error.Fail():
        return f"<read failed: {error.GetCString()}>"
    return contents.hex(" ")


class IgnoreRosettaTraceStop:
    """Auto-continue the trace stop Rosetta emits after a filtered breakpoint."""

    def __init__(self, _target, _extra_args, _internal_dict):
        pass

    def handle_stop(self, exe_ctx, _stream):
        reason = exe_ctx.GetThread().GetStopReason()
        return reason not in (
            lldb.eStopReasonInvalid,
            lldb.eStopReasonNone,
            lldb.eStopReasonTrace,
        )


def stop_on_invalid_link(frame, _breakpoint_location, _internal_dict):
    invalid_link = frame.FindRegister("rax").GetValueAsUnsigned()
    if invalid_link == 0 or invalid_link >= MINIMUM_GUEST_POINTER:
        return False

    current_block = frame.FindRegister("rcx").GetValueAsUnsigned()
    cache_slot = frame.FindRegister("rsi").GetValueAsUnsigned()
    bin_offset = frame.FindRegister("rdx").GetValueAsUnsigned()
    thread = frame.GetThread()
    process = thread.GetProcess()

    print(
        "[REMu][LLDB] INVALID ALLOCATOR LINK "
        f"thread={thread.GetName()!r} "
        f"block=0x{current_block:x} link=0x{invalid_link:x} "
        f"cache=0x{cache_slot:x} bin_offset=0x{bin_offset:x}",
        flush=True,
    )
    print(
        f"[REMu][LLDB] block bytes: {_read_hex(process, current_block, 128)}",
        flush=True,
    )
    print(
        f"[REMu][LLDB] cache bytes: {_read_hex(process, cache_slot - 32, 96)}",
        flush=True,
    )
    for index, stack_frame in enumerate(thread):
        function = stack_frame.GetFunctionName() or "<guest>"
        print(
            f"[REMu][LLDB] frame[{index}] pc=0x{stack_frame.GetPC():x} {function}",
            flush=True,
        )
        if index == 31:
            break
    return True
