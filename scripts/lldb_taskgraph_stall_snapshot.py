# SPDX-License-Identifier: GPL-2.0-or-later

"""Read-only post-stall snapshot of Silent Hill's TaskGraph wait.

The command in this module is intentionally attach-only and does not install
breakpoints, continue the process, or write target memory.  The accompanying
``probe-taskgraph-stall.sh`` validates the exact bounded REmu process before
LLDB attaches and always follows the snapshot with a non-terminating detach.

The pure frame, queue, and allocator decoders are importable without LLDB so
their title-specific address relationships can be unit-tested.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, Iterable, Optional

try:
    import lldb  # type: ignore[import-not-found]
except ImportError:
    lldb = None


ReadU64 = Callable[[int], Optional[int]]

TAG = "[REMu][LLDB][TASKGRAPH-STALL]"

MINIMUM_GUEST_ADDRESS = 0x8_0000_0000
MAXIMUM_GUEST_ADDRESS = 0x100_0000_0000

# ImportCallFrame offsets from crates/remu-runtime/src/lib.rs.
IMPORT_FRAME_GUEST_RSP = 0x38
IMPORT_FRAME_GUEST_RETURN = 0x40

# Exact wait chain decoded from PPSA10112's materialized eboot.
HOST_EVENT_WAIT_RETURN = 0x801210878
GAME_THREAD_WAIT_RETURN = 0x801BE228E
BEGIN_FENCE_WAIT_RETURN = 0x801BE0511
BEGIN_FENCE_PARENT_RETURN = 0x80101AF94
GUEST_RSP_WAIT_RBP_OFFSET = 0x88
GUEST_RSP_WAIT_RETURN_OFFSET = 0x90
OUTER_EVENT_LOCAL_OFFSET = 0x40

TASKGRAPH_SINGLETON_POINTER = 0x807BA35E0
TASKGRAPH_NODE_SLAB_POINTERS = 0x807BA8CD0
TASKGRAPH_RENDER_THREAD_INDEX = 3
TASKGRAPH_QUEUE_OBJECT_OFFSET = TASKGRAPH_RENDER_THREAD_INDEX * 0x18 + 0x8
TASKGRAPH_QUEUE_OFFSET = 0x38
TASK_QUEUE_WAITER_BITS = 26
TASK_QUEUE_WAITER_MASK = (1 << TASK_QUEUE_WAITER_BITS) - 1
TASK_QUEUE_NODE_PAGE_BITS = 14
TASK_QUEUE_NODE_PAGE_MASK = (1 << TASK_QUEUE_NODE_PAGE_BITS) - 1
TASK_QUEUE_NODE_STRIDE = 24
TASK_WORD_COUNT = 8
EXPECTED_FENCE_TASK_VPTR = 0x8074B5C50

LANE_OFFSETS = (
    (0, 0x48, 0x90),
    (1, 0x118, 0x160),
)

DEFAULT_CHAIN_LIMIT = 64
MAXIMUM_CHAIN_LIMIT = 512
MAXIMUM_NATIVE_THREADS = 512
NATIVE_STACK_SCAN_BYTES = 256 * 1024
NATIVE_STACK_READ_CHUNK = 32 * 1024
GUEST_FRAME_CHAIN_LIMIT = 64
MAXIMUM_GUEST_FRAME_SPAN = 16 * 1024 * 1024
ALLOCATOR_PAGE_BYTES = 1024 * 1024
MAXIMUM_ALLOCATOR_SCAN_BYTES = 8 * ALLOCATOR_PAGE_BYTES
MAXIMUM_ALLOCATOR_MATCHES = 64


def _optional_hex(value: Optional[int]) -> str:
    return "unreadable" if value is None else f"{value:#x}"


def _words_text(words: Iterable[Optional[int]]) -> str:
    return "[" + ",".join(_optional_hex(word) for word in words) + "]"


def plausible_guest_address(address: Optional[int]) -> bool:
    """Return whether a value lies in REmu's direct-guest address window."""

    return (
        address is not None
        and MINIMUM_GUEST_ADDRESS <= address < MAXIMUM_GUEST_ADDRESS
    )


def plausible_guest_pointer(address: Optional[int]) -> bool:
    """Return whether a value can be an aligned direct-guest pointer."""

    return plausible_guest_address(address) and address % 8 == 0


@dataclass(frozen=True)
class WaitCandidate:
    """One fully validated active host-import/guest-wait frame chain."""

    thread_id: int
    native_sp: int
    import_frame: int
    guest_rsp: int
    wait_rbp: int
    outer_rbp: int
    parent_rbp: int
    event: int
    source: str = "import-frame"

    @property
    def native_distance(self) -> int:
        if self.source == "direct-guest":
            return 0
        return self.import_frame - self.native_sp


def _decode_wait_rbp(
    wait_rbp: int,
    thread_id: int,
    native_sp: int,
    import_frame: int,
    guest_rsp: int,
    source: str,
    read_u64: ReadU64,
) -> Optional[WaitCandidate]:
    """Validate the fixed frame-pointer chain above GameThreadWaitForTask."""

    if not plausible_guest_pointer(wait_rbp):
        return None
    if read_u64(wait_rbp + 8) != GAME_THREAD_WAIT_RETURN:
        return None
    outer_rbp = read_u64(wait_rbp)
    if not plausible_guest_pointer(outer_rbp):
        return None
    assert outer_rbp is not None
    if read_u64(outer_rbp + 8) != BEGIN_FENCE_WAIT_RETURN:
        return None
    parent_rbp = read_u64(outer_rbp)
    if not plausible_guest_pointer(parent_rbp):
        return None
    assert parent_rbp is not None
    if read_u64(parent_rbp + 8) != BEGIN_FENCE_PARENT_RETURN:
        return None
    event = read_u64(outer_rbp - OUTER_EVENT_LOCAL_OFFSET)
    if not plausible_guest_pointer(event):
        return None
    assert event is not None
    return WaitCandidate(
        thread_id=thread_id,
        native_sp=native_sp,
        import_frame=import_frame,
        guest_rsp=guest_rsp,
        wait_rbp=wait_rbp,
        outer_rbp=outer_rbp,
        parent_rbp=parent_rbp,
        event=event,
        source=source,
    )


def decode_wait_candidate(
    import_frame: int,
    thread_id: int,
    native_sp: int,
    read_u64: ReadU64,
) -> Optional[WaitCandidate]:
    """Validate one possible ImportCallFrame and the exact guest caller chain."""

    if import_frame < native_sp:
        return None
    if read_u64(import_frame + IMPORT_FRAME_GUEST_RETURN) != HOST_EVENT_WAIT_RETURN:
        return None
    guest_rsp = read_u64(import_frame + IMPORT_FRAME_GUEST_RSP)
    if not plausible_guest_pointer(guest_rsp):
        return None
    assert guest_rsp is not None
    if (
        read_u64(guest_rsp + GUEST_RSP_WAIT_RETURN_OFFSET)
        != GAME_THREAD_WAIT_RETURN
    ):
        return None
    wait_rbp = read_u64(guest_rsp + GUEST_RSP_WAIT_RBP_OFFSET)
    if wait_rbp is None:
        return None
    return _decode_wait_rbp(
        wait_rbp,
        thread_id,
        native_sp,
        import_frame,
        guest_rsp,
        "import-frame",
        read_u64,
    )


def decode_direct_guest_wait(
    thread_id: int,
    guest_pc: int,
    guest_rsp: int,
    guest_rbp: int,
    read_u64: ReadU64,
    limit: int = GUEST_FRAME_CHAIN_LIMIT,
) -> Optional[WaitCandidate]:
    """Find the same wait while a sampled thread is executing guest code.

    The main loop leaves the timed-wait import roughly every millisecond.  In
    that interval there is no live ImportCallFrame, but Rosetta exposes the
    direct guest RSP/RBP in the stopped x86 thread.  Walking only the bounded,
    monotonic RBP chain keeps this fallback read-only and title-specific.
    """

    if (
        not plausible_guest_address(guest_pc)
        or not plausible_guest_pointer(guest_rsp)
        or not plausible_guest_pointer(guest_rbp)
    ):
        return None
    frame_rbp = guest_rbp
    for _ in range(limit):
        candidate = _decode_wait_rbp(
            frame_rbp,
            thread_id,
            guest_rsp,
            0,
            guest_rsp,
            "direct-guest",
            read_u64,
        )
        if candidate is not None:
            return candidate
        next_rbp = read_u64(frame_rbp)
        if (
            not plausible_guest_pointer(next_rbp)
            or next_rbp <= frame_rbp
            or next_rbp - frame_rbp > MAXIMUM_GUEST_FRAME_SPAN
        ):
            return None
        frame_rbp = next_rbp
    return None


def select_wait_candidate(
    candidates: Iterable[WaitCandidate],
) -> Optional[WaitCandidate]:
    """Select the active frame nearest its native SP, rejecting an exact tie."""

    ordered = sorted(
        candidates,
        key=lambda candidate: (
            candidate.native_distance,
            candidate.thread_id,
            candidate.import_frame,
        ),
    )
    if not ordered:
        return None
    if (
        len(ordered) > 1
        and ordered[0].native_distance == ordered[1].native_distance
        and ordered[0].thread_id != ordered[1].thread_id
    ):
        return None
    return ordered[0]


def task_queue_node_address(
    handle: int,
    read_u64: ReadU64,
) -> Optional[int]:
    """Resolve a UE TaskGraph queue handle through its paged node slab."""

    node_id = handle & TASK_QUEUE_WAITER_MASK
    page = node_id >> TASK_QUEUE_NODE_PAGE_BITS
    slot = node_id & TASK_QUEUE_NODE_PAGE_MASK
    page_base = read_u64(TASKGRAPH_NODE_SLAB_POINTERS + page * 8)
    if not plausible_guest_pointer(page_base):
        return None
    assert page_base is not None
    return page_base + slot * TASK_QUEUE_NODE_STRIDE


@dataclass(frozen=True)
class QueueNode:
    index: int
    handle: int
    address: int
    next_handle: Optional[int]
    task: Optional[int]
    task_words: tuple[Optional[int], ...]
    event_match: bool


@dataclass(frozen=True)
class LaneSnapshot:
    lane: int
    head: Optional[int]
    tail: Optional[int]
    termination: str
    classification: str
    nodes: tuple[QueueNode, ...]

    @property
    def match_indexes(self) -> tuple[int, ...]:
        return tuple(node.index for node in self.nodes if node.event_match)


def walk_task_queue_lane(
    lane: int,
    head: Optional[int],
    tail: Optional[int],
    event: int,
    read_u64: ReadU64,
    limit: int = DEFAULT_CHAIN_LIMIT,
) -> LaneSnapshot:
    """Walk one bounded queue chain and classify same-event task reachability."""

    if head is None or tail is None:
        return LaneSnapshot(
            lane, head, tail, "unreadable-cursor", "unknown", ()
        )
    if head == 0:
        return LaneSnapshot(lane, head, tail, "null-head", "absent", ())

    nodes: list[QueueNode] = []
    seen_node_ids: set[int] = set()
    handle = head
    termination = "limit"
    for index in range(limit):
        node_id = handle & TASK_QUEUE_WAITER_MASK
        if node_id in seen_node_ids:
            termination = "cycle"
            break
        seen_node_ids.add(node_id)
        address = task_queue_node_address(handle, read_u64)
        if address is None:
            termination = "unreadable-node"
            break
        next_handle = read_u64(address)
        task = read_u64(address + 8)
        task_words = (
            tuple(read_u64(task + word * 8) for word in range(TASK_WORD_COUNT))
            if plausible_guest_pointer(task)
            else ()
        )
        event_match = (
            plausible_guest_pointer(task)
            and len(task_words) > 3
            and task_words[3] == event
        )
        nodes.append(
            QueueNode(
                index=index,
                handle=handle,
                address=address,
                next_handle=next_handle,
                task=task,
                task_words=task_words,
                event_match=event_match,
            )
        )
        if handle == tail:
            termination = "tail"
            break
        if next_handle is None:
            termination = "unreadable-next"
            break
        if next_handle == 0:
            termination = "null-before-tail"
            break
        handle = next_handle

    matches = [node.index for node in nodes if node.event_match]
    if any(index > 0 for index in matches):
        classification = "queued"
    elif matches:
        classification = "popped-head"
    else:
        classification = "absent"
    return LaneSnapshot(
        lane,
        head,
        tail,
        termination,
        classification,
        tuple(nodes),
    )


def allocator_scan_window(
    task_anchors: Iterable[int],
) -> tuple[Optional[int], Optional[int], str]:
    """Return a bounded allocator range only from plausible aligned task anchors."""

    anchors = sorted(
        {
            task
            for task in task_anchors
            if plausible_guest_pointer(task) and task % 0x100 == 0
        }
    )
    if not anchors:
        return None, None, "no-safe-task-anchors"
    page_mask = ALLOCATOR_PAGE_BYTES - 1
    first_page = anchors[0] & ~page_mask
    last_page_end = (anchors[-1] + 0x100 + page_mask) & ~page_mask
    start = max(
        MINIMUM_GUEST_ADDRESS,
        first_page - ALLOCATOR_PAGE_BYTES,
    )
    end = min(
        MAXIMUM_GUEST_ADDRESS,
        last_page_end + ALLOCATOR_PAGE_BYTES,
    )
    if end <= start or end - start > MAXIMUM_ALLOCATOR_SCAN_BYTES:
        return None, None, "unsafe-anchor-span"
    return start, end, "safe"


def scan_aligned_task_candidates(
    base: int,
    contents: bytes,
    event: int,
) -> tuple[int, ...]:
    """Find 0x100-aligned records whose +0x18 qword equals the event."""

    matches = []
    first = (-base) & 0xFF
    for offset in range(first, max(0, len(contents) - 0x20 + 1), 0x100):
        if int.from_bytes(contents[offset + 0x18 : offset + 0x20], "little") == event:
            matches.append(base + offset)
    return tuple(matches)


def _positive_limit(name: str, default: int, maximum: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw, 0)
    except ValueError:
        return default
    return value if 1 <= value <= maximum else default


def _process_reader(process) -> ReadU64:
    def read_u64(address: int) -> Optional[int]:
        if address == 0:
            return None
        error = lldb.SBError()
        contents = process.ReadMemory(address, 8, error)
        if error.Fail() or len(contents) != 8:
            return None
        return int.from_bytes(contents, "little")

    return read_u64


def _read_chunk(process, address: int, size: int) -> Optional[bytes]:
    error = lldb.SBError()
    contents = process.ReadMemory(address, size, error)
    if error.Fail() or len(contents) != size:
        return None
    return bytes(contents)


@dataclass(frozen=True)
class NativeStackScan:
    candidates: tuple[WaitCandidate, ...]
    chunks_read: int
    return_words: int


def _direct_thread_wait_candidate(
    thread,
    read_u64: ReadU64,
) -> Optional[WaitCandidate]:
    frame = thread.GetFrameAtIndex(0)
    if not frame.IsValid():
        return None
    return decode_direct_guest_wait(
        thread.GetThreadID(),
        frame.GetPC(),
        frame.GetSP(),
        frame.FindRegister("rbp").GetValueAsUnsigned(),
        read_u64,
    )


def _thread_wait_candidates(
    process,
    thread,
    read_u64: ReadU64,
) -> NativeStackScan:
    frame = thread.GetFrameAtIndex(0)
    if not frame.IsValid():
        return NativeStackScan((), 0, 0)
    native_sp = frame.GetSP()
    if native_sp in (0, (1 << 64) - 1):
        return NativeStackScan((), 0, 0)
    thread_id = thread.GetThreadID()
    needle = HOST_EVENT_WAIT_RETURN.to_bytes(8, "little")
    candidates = []
    seen: set[int] = set()
    chunks_read = 0
    return_words = 0
    for offset in range(0, NATIVE_STACK_SCAN_BYTES, NATIVE_STACK_READ_CHUNK):
        address = native_sp + offset
        contents = _read_chunk(
            process, address, NATIVE_STACK_READ_CHUNK
        )
        if contents is None:
            break
        chunks_read += 1
        found = contents.find(needle)
        while found != -1:
            return_words += 1
            return_word = address + found
            if return_word % 8 == 0 and return_word >= IMPORT_FRAME_GUEST_RETURN:
                import_frame = return_word - IMPORT_FRAME_GUEST_RETURN
                if import_frame not in seen:
                    seen.add(import_frame)
                    candidate = decode_wait_candidate(
                        import_frame,
                        thread_id,
                        native_sp,
                        read_u64,
                    )
                    if candidate is not None:
                        candidates.append(candidate)
            found = contents.find(needle, found + 1)
    return NativeStackScan(
        tuple(candidates),
        chunks_read,
        return_words,
    )


def _print_lane(snapshot: LaneSnapshot) -> None:
    print(
        f"{TAG} lane={snapshot.lane} "
        f"head={_optional_hex(snapshot.head)} "
        f"tail={_optional_hex(snapshot.tail)} "
        f"termination={snapshot.termination} "
        f"classification={snapshot.classification} "
        f"matches={snapshot.match_indexes}",
        flush=True,
    )
    for node in snapshot.nodes:
        print(
            f"{TAG} node lane={snapshot.lane} index={node.index} "
            f"handle={node.handle:#x} address={node.address:#x} "
            f"next={_optional_hex(node.next_handle)} "
            f"task={_optional_hex(node.task)} "
            f"event_match={node.event_match} "
            f"task_words={_words_text(node.task_words)}",
            flush=True,
        )


def _scan_allocator(
    process,
    event: int,
    task_anchors: Iterable[int],
    read_u64: ReadU64,
) -> None:
    start, end, reason = allocator_scan_window(task_anchors)
    if start is None or end is None:
        print(f"{TAG} allocator-scan skipped reason={reason}", flush=True)
        return
    matches: list[int] = []
    unreadable_chunks = 0
    for address in range(start, end, ALLOCATOR_PAGE_BYTES):
        size = min(ALLOCATOR_PAGE_BYTES, end - address)
        contents = _read_chunk(process, address, size)
        if contents is None:
            unreadable_chunks += 1
            continue
        matches.extend(
            scan_aligned_task_candidates(address, contents, event)
        )
        if len(matches) >= MAXIMUM_ALLOCATOR_MATCHES:
            matches = matches[:MAXIMUM_ALLOCATOR_MATCHES]
            break
    print(
        f"{TAG} allocator-scan range={start:#x}..{end:#x} "
        f"bytes={end - start} unreadable_chunks={unreadable_chunks} "
        f"matches={len(matches)}",
        flush=True,
    )
    for task in matches:
        words = tuple(
            read_u64(task + index * 8) for index in range(TASK_WORD_COUNT)
        )
        print(
            f"{TAG} allocator-match task={task:#x} "
            f"expected_vptr={words[0] == EXPECTED_FENCE_TASK_VPTR} "
            f"task_words={_words_text(words)}",
            flush=True,
        )


def snapshot_taskgraph_stall(debugger, _command, result, _internal_dict):
    """Capture one stopped, read-only TaskGraph snapshot."""

    try:
        target = debugger.GetSelectedTarget()
        process = target.GetProcess()
        if not process.IsValid():
            result.AppendMessage(f"{TAG} done reason=no-valid-process")
            return
        if process.GetState() != lldb.eStateStopped:
            result.AppendMessage(
                f"{TAG} done reason=process-not-stopped "
                f"state={process.GetState()}"
            )
            return

        read_u64 = _process_reader(process)
        threads = list(process)
        scanned_threads = threads[:MAXIMUM_NATIVE_THREADS]
        direct_guest_threads = 0
        candidates = []
        for thread in scanned_threads:
            frame = thread.GetFrameAtIndex(0)
            if frame.IsValid() and plausible_guest_address(frame.GetPC()):
                direct_guest_threads += 1
            candidate = _direct_thread_wait_candidate(thread, read_u64)
            if candidate is not None:
                candidates.append(candidate)

        native_chunks = 0
        native_return_words = 0
        native_threads_scanned = 0
        if not candidates:
            prioritized_threads = sorted(
                scanned_threads,
                key=lambda thread: (
                    0
                    if thread.GetName() in ("main", "GameThread")
                    else 1,
                    thread.GetIndexID(),
                ),
            )
            for thread in prioritized_threads:
                native_threads_scanned += 1
                scan = _thread_wait_candidates(
                    process, thread, read_u64
                )
                candidates.extend(scan.candidates)
                native_chunks += scan.chunks_read
                native_return_words += scan.return_words
                if scan.candidates:
                    break
        print(
            f"{TAG} scan-stats total_threads={len(threads)} "
            f"scanned_threads={len(scanned_threads)} "
            f"direct_guest_threads={direct_guest_threads} "
            f"native_threads_scanned={native_threads_scanned} "
            f"native_chunks={native_chunks} "
            f"native_return_words={native_return_words}",
            flush=True,
        )
        unique_candidates = {
            (
                candidate.source,
                candidate.thread_id,
                candidate.import_frame,
                candidate.wait_rbp,
            ): candidate
            for candidate in candidates
        }
        candidates = list(unique_candidates.values())
        for candidate in sorted(
            candidates,
            key=lambda item: (item.native_distance, item.thread_id),
        ):
            print(
                f"{TAG} wait-candidate source={candidate.source} "
                f"thread={candidate.thread_id:#x} "
                f"native_sp={candidate.native_sp:#x} "
                f"import_frame={candidate.import_frame:#x} "
                f"distance={candidate.native_distance:#x} "
                f"guest_rsp={candidate.guest_rsp:#x} "
                f"wait_rbp={candidate.wait_rbp:#x} "
                f"outer_rbp={candidate.outer_rbp:#x} "
                f"parent_rbp={candidate.parent_rbp:#x} "
                f"event={candidate.event:#x}",
                flush=True,
            )
        selected = select_wait_candidate(candidates)
        if selected is None:
            reason = "no-exact-wait" if not candidates else "ambiguous-wait"
            result.AppendMessage(
                f"{TAG} done reason={reason} candidates={len(candidates)}"
            )
            return

        event_words = tuple(
            read_u64(selected.event + index * 8)
            for index in range(TASK_WORD_COUNT)
        )
        print(
            f"{TAG} selected source={selected.source} "
            f"thread={selected.thread_id:#x} "
            f"outer_rbp={selected.outer_rbp:#x} "
            f"event={selected.event:#x} "
            f"event_words={_words_text(event_words)}",
            flush=True,
        )

        singleton = read_u64(TASKGRAPH_SINGLETON_POINTER)
        queue_object = (
            read_u64(singleton + TASKGRAPH_QUEUE_OBJECT_OFFSET)
            if plausible_guest_pointer(singleton)
            else None
        )
        queue = (
            queue_object + TASKGRAPH_QUEUE_OFFSET
            if plausible_guest_pointer(queue_object)
            else None
        )
        print(
            f"{TAG} taskgraph singleton={_optional_hex(singleton)} "
            f"queue_object={_optional_hex(queue_object)} "
            f"queue={_optional_hex(queue)}",
            flush=True,
        )
        if queue is None:
            result.AppendMessage(
                f"{TAG} done reason=unreadable-render-queue"
            )
            return

        limit = _positive_limit(
            "REMU_TASKGRAPH_STALL_CHAIN_LIMIT",
            DEFAULT_CHAIN_LIMIT,
            MAXIMUM_CHAIN_LIMIT,
        )
        lanes = []
        for lane, head_offset, tail_offset in LANE_OFFSETS:
            lanes.append(
                walk_task_queue_lane(
                    lane,
                    read_u64(queue + head_offset),
                    read_u64(queue + tail_offset),
                    selected.event,
                    read_u64,
                    limit,
                )
            )
        for lane in lanes:
            _print_lane(lane)

        if not any(lane.match_indexes for lane in lanes):
            task_anchors = [
                node.task
                for lane in lanes
                for node in lane.nodes
                if node.task is not None
            ]
            _scan_allocator(
                process,
                selected.event,
                task_anchors,
                read_u64,
            )
        else:
            print(
                f"{TAG} allocator-scan skipped "
                "reason=event-task-present-in-queue",
                flush=True,
            )
        result.AppendMessage(
            f"{TAG} done reason=snapshot-complete "
            f"event={selected.event:#x}"
        )
    except Exception as error:  # Keep the following detach command reachable.
        result.AppendMessage(
            f"{TAG} done reason=snapshot-exception "
            f"type={type(error).__name__} detail={error}"
        )


def detach_taskgraph_stall(debugger, _command, result, _internal_dict):
    """Detach and resume without terminating the explicitly attached target."""

    try:
        process = debugger.GetSelectedTarget().GetProcess()
        if not process.IsValid():
            result.AppendMessage(f"{TAG} detach-skipped no-valid-process")
            return
        if process.GetState() in (lldb.eStateExited, lldb.eStateDetached):
            result.AppendMessage(f"{TAG} detach-skipped process-ended")
            return
        error = process.Detach(False)
        if error.Fail():
            result.AppendMessage(
                f"{TAG} detach-failed error={error.GetCString()}"
            )
            return
        result.AppendMessage(
            f"{TAG} detached without terminating target"
        )
    except Exception as error:
        result.AppendMessage(
            f"{TAG} detach-exception type={type(error).__name__} "
            f"detail={error}"
        )


def __lldb_init_module(debugger, _internal_dict):
    if lldb is None:
        return
    debugger.HandleCommand(
        "command script add "
        "-f lldb_taskgraph_stall_snapshot.snapshot_taskgraph_stall "
        "remu-taskgraph-stall-snapshot"
    )
    debugger.HandleCommand(
        "command script add "
        "-f lldb_taskgraph_stall_snapshot.detach_taskgraph_stall "
        "remu-taskgraph-stall-detach"
    )
