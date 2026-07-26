"""LLDB breakpoint callbacks for the Silent Hill allocator corruption diagnostic.

This is intentionally small and side-effect free: it filters the very hot guest
mapping breakpoints without invoking LLDB's expression evaluator.  Returning
False auto-continues; returning True leaves the process stopped immediately
after the mapping that contains the allocator cache slot.
"""

WATCH_ADDRESS = 0x801B880020


def _contains(start, size):
    return start <= WATCH_ADDRESS < start + size


def map_anonymous(frame, _breakpoint_location, _internal_dict):
    start = frame.FindRegister("rbx").GetValueAsUnsigned()
    size = frame.FindRegister("r13").GetValueAsUnsigned()
    if not _contains(start, size):
        return False
    print(
        f"[REMu][LLDB] allocator slot mapped anonymously: "
        f"start=0x{start:x} size=0x{size:x}"
    )
    return True


def map_host_shared(frame, _breakpoint_location, _internal_dict):
    start = frame.FindRegister("r14").GetValueAsUnsigned()
    size = frame.FindRegister("r15").GetValueAsUnsigned()
    if not _contains(start, size):
        return False
    print(
        f"[REMu][LLDB] allocator slot mapped from shared memory: "
        f"start=0x{start:x} size=0x{size:x}"
    )
    return True
