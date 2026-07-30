// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

//! Ampr asynchronous file command buffers.
//!
//! This module implements sceAmpr* entry points that construct and inspect asynchronous file-I/O
//! command buffers.
//!
//! The bytes a command occupies are the runtime's own encoding. Nothing outside this crate reads
//! a recorded command back, so the only property a guest can observe is how much room each
//! command takes: a size query and the builder it names are two halves of one contract, because
//! the guest reserves buffer space with the query and then fills that space with the builder.
//! Every size reported here therefore matches, byte for byte, what the corresponding builder
//! appends.
//!
//! [`dispatch`] is consulted by the runtime's import dispatcher only after the main table
//! misses, and returns `None` for any NID it does not own so the caller can keep looking.

use crate::{
    AMPR_COMMAND_RECORD_SIZE, ProbeDispatchState, SCE_KERNEL_ERROR_BUSY,
    SCE_KERNEL_ERROR_INVALID_ARGUMENT, SCE_KERNEL_ERROR_OPERATION_NOT_PERMITTED,
    ampr_command_buffer_write_address_on_completion, ampr_command_buffer_write_kernel_event_queue,
    ampr_measure_command_size, guest_memset, low_u32, process_guest_range_is_readable,
    read_guest_bytes, read_guest_u32, read_guest_u64, sce_kernel_error, write_guest_bytes,
    write_guest_u32,
};

/// Record tag for the command that invalidates a buffer's gather/scatter chain.
///
/// Tags 1 and 2 already name the file read and the event-queue signal, and the tags between this
/// one and [`AMPR_COMMAND_NOP`] are left free for the file and synchronisation commands that have
/// no builder yet.
const AMPR_COMMAND_RESET_GATHER_SCATTER: u32 = 6;
/// Record tag for a nop command, which reserves a caller-chosen span of DWORDs.
const AMPR_COMMAND_NOP: u32 = 17;
/// Record tag for the command that opens a trace marker.
const AMPR_COMMAND_MARKER: u32 = 18;
/// Record tag for the commands carrying the remainder of a marker's message.
const AMPR_COMMAND_MARKER_TEXT: u32 = 19;

/// Marker discriminant for an annotation that leaves the reader's marker stack alone.
const MARKER_KIND_SET: u32 = 0;
/// Marker discriminant for an annotation that opens a nested region.
const MARKER_KIND_PUSH: u32 = 1;
/// Marker discriminant for an annotation that closes the innermost region.
const MARKER_KIND_POP: u32 = 2;

/// Bytes ahead of the message in a marker's first command: its header and its display hint.
const MARKER_HEADER_BYTES: u64 = 8;
/// Bytes ahead of the message in a marker's continuation commands.
const MARKER_CONTINUATION_HEADER_BYTES: u64 = 4;
/// Message bytes a marker's first command carries, leaving room for the display hint.
const MARKER_FIRST_TEXT_BYTES: usize = 56;
/// Message bytes each continuation command carries.
const MARKER_CONTINUATION_TEXT_BYTES: usize = 60;
/// Longest message recorded for one marker; anything beyond this is dropped.
const MARKER_TEXT_LIMIT: usize = 1024;
/// Bytes examined per step while scanning a message for its terminator.
const MARKER_SCAN_BYTES: usize = 64;

/// DWORDs a single nop command can occupy, payload included.
const NOP_MAX_DWORDS: u32 = 16;
/// DWORDs of payload a nop command can carry, the command's own DWORD excluded.
const NOP_MAX_PAYLOAD_DWORDS: u32 = NOP_MAX_DWORDS - 1;

/// Offset at which a command-buffer object records the base of the memory attached to it.
const COMMAND_BUFFER_BASE_OFFSET: u64 = 8;

/// Handles a NID belonging to this module, or returns `None` if it owns no such NID.
pub(crate) fn dispatch(
    nid: &str,
    arguments: [u64; 6],
    state: &mut ProbeDispatchState,
) -> Option<u64> {
    dispatch_builder(nid, arguments, state).or_else(|| dispatch_command_size(nid, arguments))
}

/// Handles the entry points that append a command to a buffer or report a buffer's state.
fn dispatch_builder(nid: &str, arguments: [u64; 6], state: &mut ProbeDispatchState) -> Option<u64> {
    Some(match nid {
        "YPxkUDhgoNI" => reset_gather_scatter_state(arguments[0], state),
        "tNn5WBkta60" => nop(arguments, state),
        "pFQ9UHpO52s" => nop_with_data(arguments, state),
        "GmOguNIsuKk" => construct_nop(arguments, state),
        "4UkZbYKVF7c" => construct_marker(arguments, state),
        "4quckD2y7Pg" => marker(arguments[0], MARKER_KIND_SET, arguments[1], None, state),
        "sWbST0oQKsc" => marker(
            arguments[0],
            MARKER_KIND_SET,
            arguments[1],
            Some(low_u32(arguments[2])),
            state,
        ),
        "dXPaz65HNmk" => marker(arguments[0], MARKER_KIND_PUSH, arguments[1], None, state),
        "f12ObAMEi9A" => marker(
            arguments[0],
            MARKER_KIND_PUSH,
            arguments[1],
            Some(low_u32(arguments[2])),
            state,
        ),
        "mv0O8Zg0woU" => append_marker(arguments[0], MARKER_KIND_POP, None, None, state),
        // The command that signals an event queue on completion is exported under both its
        // current and its original name, and the runtime already builds it.
        "o67gODLFpls" => ampr_command_buffer_write_kernel_event_queue(arguments, state),
        "sJXyWHjP-F8" => ampr_command_buffer_write_address_on_completion(arguments, state),
        "RPCAhx-aabE" => buffer_base_address(arguments[0], state),
        "gzndltBEzWc" => num_commands(arguments[0], state),
        _ => return None,
    })
}

/// Handles the size queries, which report a command's footprint without touching any buffer.
fn dispatch_command_size(nid: &str, arguments: [u64; 6]) -> Option<u64> {
    Some(match nid {
        "NNIZ-FMyz3M" => measure_nop(arguments[0]),
        "Xp85BP3+BBI" => measure_nop_with_data(arguments[0]),
        // A display hint occupies its slot whether or not the caller supplies one, so the two
        // overloads of each marker query report the same size.
        "VGkEj4d6-Kg" | "tmfr97+ED5I" | "0RdLmAh7WVo" | "3OfeY4pzDV0" => {
            measure_marker(arguments[0])
        }
        "pbnNnahE8vk" => MARKER_HEADER_BYTES,
        // The gather and scatter reads have no builder yet, but their records are the same fixed
        // stride as every other file command, so their footprint is already settled.
        "rddQYXM0CjM" | "qesF88X4DRg" | "7nXGDGMXSqo" | "DXmgc5op8Yw" | "C+IEj+BsAFM" => {
            AMPR_COMMAND_RECORD_SIZE
        }
        "Zi3dBUjgyXI" => ampr_measure_command_size(),
        _ => return None,
    })
}

/// Appends the command that ends a buffer's gather/scatter chain.
///
/// A gather, scatter or gather-scatter read appended after this one has no preceding read to
/// borrow a file, a destination or a file position from, and fails until a plain read
/// re-establishes the chain.
fn reset_gather_scatter_state(command_buffer: u64, state: &mut ProbeDispatchState) -> u64 {
    let record = match reserve_record(state, command_buffer, AMPR_COMMAND_RECORD_SIZE, 1) {
        Ok(record) => record,
        Err(error) => return error,
    };
    guest_memset(record, 0, AMPR_COMMAND_RECORD_SIZE);
    write_guest_u32(record, AMPR_COMMAND_RESET_GATHER_SCATTER);
    0
}

/// Appends a nop command occupying exactly the requested number of DWORDs.
///
/// The span is the point of the command: a guest reserves it here and overwrites it with a real
/// command later, so the recorded size has to be exactly what was asked for.
fn nop(arguments: [u64; 6], state: &mut ProbeDispatchState) -> u64 {
    let dwords = low_u32(arguments[1]);
    if dwords == 0 || dwords > NOP_MAX_DWORDS {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    }
    let bytes = u64::from(dwords) * 4;
    let record = match reserve_record(state, arguments[0], bytes, 1) {
        Ok(record) => record,
        Err(error) => return error,
    };
    guest_memset(record, 0, bytes);
    write_guest_u32(record, record_header(AMPR_COMMAND_NOP, dwords, 0));
    0
}

/// Appends a nop command that carries a copy of the caller's DWORDs as its payload.
fn nop_with_data(arguments: [u64; 6], state: &mut ProbeDispatchState) -> u64 {
    let payload_dwords = low_u32(arguments[1]);
    let data = arguments[2];
    if payload_dwords > NOP_MAX_PAYLOAD_DWORDS {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    }
    let payload_bytes = u64::from(payload_dwords) * 4;
    if payload_dwords != 0 && !guest_range_is_readable(data, payload_bytes) {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    }
    let dwords = payload_dwords + 1;
    let bytes = u64::from(dwords) * 4;
    let record = match reserve_record(state, arguments[0], bytes, 1) {
        Ok(record) => record,
        Err(error) => return error,
    };
    guest_memset(record, 0, bytes);
    write_guest_u32(record, record_header(AMPR_COMMAND_NOP, dwords, 0));
    for index in 0..u64::from(payload_dwords) {
        write_guest_u32(record + 4 + index * 4, read_guest_u32(data + index * 4));
    }
    0
}

/// Appends a nop command from the discriminant, payload and display hint the caller chose.
///
/// The nop and marker entry points are built on this one, and a guest that was compiled against
/// an inlined copy of them calls it directly.
fn construct_nop(arguments: [u64; 6], state: &mut ProbeDispatchState) -> u64 {
    let kind = low_u32(arguments[1]);
    let payload = arguments[2];
    let payload_bytes = u64::from(low_u32(arguments[3]));
    let color = match read_optional_color(arguments[4]) {
        Ok(color) => color,
        Err(error) => return error,
    };
    let hint_bytes = if color.is_some() { 4 } else { 0 };
    let Ok(payload_dwords) = u32::try_from(payload_bytes.div_ceil(4)) else {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    };
    let dwords = payload_dwords + 1 + hint_bytes / 4;
    if dwords > NOP_MAX_DWORDS {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    }
    if payload_bytes != 0 && !guest_range_is_readable(payload, payload_bytes) {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    }
    let bytes = u64::from(dwords) * 4;
    let record = match reserve_record(state, arguments[0], bytes, 1) {
        Ok(record) => record,
        Err(error) => return error,
    };
    guest_memset(record, 0, bytes);
    write_guest_u32(record, record_header(AMPR_COMMAND_NOP, dwords, kind));
    if let Some(color) = color {
        write_guest_u32(record + 4, color);
    }
    copy_guest_range(record + 4 + u64::from(hint_bytes), payload, payload_bytes);
    0
}

/// Appends a trace marker from the discriminant, message and display hint the caller chose.
///
/// The message is optional here: the entry point that closes a marker region supplies none.
fn construct_marker(arguments: [u64; 6], state: &mut ProbeDispatchState) -> u64 {
    let kind = low_u32(arguments[1]);
    let message = arguments[2];
    let color = match read_optional_color(arguments[3]) {
        Ok(color) => color,
        Err(error) => return error,
    };
    if message == 0 {
        return append_marker(arguments[0], kind, None, color, state);
    }
    marker(arguments[0], kind, message, color, state)
}

/// Appends a trace marker carrying the message at `message`.
fn marker(
    command_buffer: u64,
    kind: u32,
    message: u64,
    color: Option<u32>,
    state: &mut ProbeDispatchState,
) -> u64 {
    let Some(text) = read_marker_text(message) else {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    };
    append_marker(command_buffer, kind, Some(&text), color, state)
}

/// Records a marker as the sequence of nop commands its message needs.
///
/// A marker never changes what a buffer transfers; a reader that is not capturing a trace skips
/// it as a plain nop, so only the space it occupies matters here.
fn append_marker(
    command_buffer: u64,
    kind: u32,
    message: Option<&[u8]>,
    color: Option<u32>,
    state: &mut ProbeDispatchState,
) -> u64 {
    let text = message.unwrap_or(&[]);
    // A message is stored terminated, so the terminator has to be paid for even when the message
    // is empty. A marker with no message at all carries no terminator either.
    let payload = text.len() + usize::from(message.is_some());
    let chunks = marker_chunks(payload);
    let bytes = chunks.iter().map(|chunk| chunk.bytes).sum();
    let Ok(commands) = u64::try_from(chunks.len()) else {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    };
    let record = match reserve_record(state, command_buffer, bytes, commands) {
        Ok(record) => record,
        Err(error) => return error,
    };
    guest_memset(record, 0, bytes);
    let mut cursor = record;
    for (index, chunk) in chunks.iter().enumerate() {
        if index == 0 {
            write_guest_u32(
                cursor,
                record_header(AMPR_COMMAND_MARKER, chunk.dwords(), kind),
            );
            write_guest_u32(cursor + 4, color.unwrap_or(0));
        } else {
            write_guest_u32(
                cursor,
                record_header(AMPR_COMMAND_MARKER_TEXT, chunk.dwords(), kind),
            );
        }
        let start = chunk.start.min(text.len());
        let end = chunk.start.saturating_add(chunk.length).min(text.len());
        if end > start {
            write_guest_bytes(cursor + chunk.header_bytes, &text[start..end]);
        }
        cursor += chunk.bytes;
    }
    0
}

/// Reports the base of the memory attached to a command buffer, or zero if it has none.
fn buffer_base_address(command_buffer: u64, state: &ProbeDispatchState) -> u64 {
    if command_buffer == 0 {
        return 0;
    }
    if let Some(command) = state.ampr_command_buffers.get(&command_buffer) {
        return command.buffer;
    }
    let Some(field) = command_buffer.checked_add(COMMAND_BUFFER_BASE_OFFSET) else {
        return 0;
    };
    if !guest_range_is_readable(field, 8) {
        return 0;
    }
    read_guest_u64(field)
}

/// Reports how many commands a buffer holds.
fn num_commands(command_buffer: u64, state: &ProbeDispatchState) -> u64 {
    if command_buffer == 0 {
        return 0;
    }
    state
        .ampr_command_buffers
        .get(&command_buffer)
        .map_or(0, |command| {
            u64::from(u32::try_from(command.command_count).unwrap_or(u32::MAX))
        })
}

/// Reports the space a nop command occupying `dwords` DWORDs needs.
fn measure_nop(dwords: u64) -> u64 {
    let dwords = low_u32(dwords);
    if dwords == 0 || dwords > NOP_MAX_DWORDS {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    }
    u64::from(dwords) * 4
}

/// Reports the space a nop command carrying `dwords` DWORDs of payload needs.
fn measure_nop_with_data(dwords: u64) -> u64 {
    let dwords = low_u32(dwords);
    if dwords > NOP_MAX_PAYLOAD_DWORDS {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    }
    u64::from(dwords + 1) * 4
}

/// Reports the space the marker carrying the message at `message` needs.
fn measure_marker(message: u64) -> u64 {
    let Some(text) = read_marker_text(message) else {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    };
    marker_chunks(text.len() + 1)
        .iter()
        .map(|chunk| chunk.bytes)
        .sum()
}

/// One nop command in the sequence a marker is recorded as.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct MarkerChunk {
    /// Bytes the command occupies, header and DWORD padding included.
    bytes: u64,
    /// Bytes the command spends on its header before the message resumes.
    header_bytes: u64,
    /// First message byte the command carries.
    start: usize,
    /// Message bytes the command carries, the terminator included when it lands here.
    length: usize,
}

impl MarkerChunk {
    /// Reports the DWORD count recorded in the command's header.
    fn dwords(&self) -> u32 {
        u32::try_from(self.bytes / 4).unwrap_or(u32::MAX)
    }
}

/// Splits a marker message of `payload` bytes across the commands that carry it.
///
/// A marker always records at least one command, so an annotation without a message still costs
/// its header.
fn marker_chunks(payload: usize) -> Vec<MarkerChunk> {
    let mut chunks = Vec::new();
    let mut written = 0;
    loop {
        let (header_bytes, capacity) = if chunks.is_empty() {
            (MARKER_HEADER_BYTES, MARKER_FIRST_TEXT_BYTES)
        } else {
            (
                MARKER_CONTINUATION_HEADER_BYTES,
                MARKER_CONTINUATION_TEXT_BYTES,
            )
        };
        let length = (payload - written).min(capacity);
        chunks.push(MarkerChunk {
            bytes: header_bytes + round_up_to_dword(length),
            header_bytes,
            start: written,
            length,
        });
        written += length;
        if written >= payload {
            return chunks;
        }
    }
}

/// Reads the display hint at `address`, or reports the error a bad pointer earns.
///
/// The hint is optional throughout the marker and nop family, so a null pointer is a valid
/// absence rather than a rejected argument.
fn read_optional_color(address: u64) -> Result<Option<u32>, u64> {
    if address == 0 {
        return Ok(None);
    }
    if !guest_range_is_readable(address, 4) {
        return Err(sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT));
    }
    Ok(Some(read_guest_u32(address)))
}

/// Reads a marker's message, reporting `None` when the pointer is null or unreadable.
fn read_marker_text(address: u64) -> Option<Vec<u8>> {
    if address == 0 {
        return None;
    }
    let mut text: Vec<u8> = Vec::new();
    while text.len() < MARKER_TEXT_LIMIT {
        let start = address.checked_add(u64::try_from(text.len()).ok()?)?;
        let mut span = MARKER_SCAN_BYTES.min(MARKER_TEXT_LIMIT - text.len());
        while span > 0 && !guest_range_is_readable(start, u64::try_from(span).ok()?) {
            span /= 2;
        }
        if span == 0 {
            break;
        }
        let mut chunk = vec![0_u8; span];
        read_guest_bytes(start, &mut chunk);
        if let Some(end) = chunk.iter().position(|byte| *byte == 0) {
            text.extend_from_slice(&chunk[..end]);
            return Some(text);
        }
        text.extend_from_slice(&chunk);
    }
    (!text.is_empty()).then_some(text)
}

/// Reserves `bytes` at the head of a command buffer for `commands` commands, and reports where
/// the caller must write them.
///
/// The reservation is what a guest observes through the buffer's offset and command count, so it
/// is taken only once every argument, the remaining capacity and the target range have been
/// accepted.
fn reserve_record(
    state: &mut ProbeDispatchState,
    command_buffer: u64,
    bytes: u64,
    commands: u64,
) -> Result<u64, u64> {
    if command_buffer == 0 || bytes == 0 {
        return Err(sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT));
    }
    let Some(command) = state.ampr_command_buffers.get_mut(&command_buffer) else {
        return Err(sce_kernel_error(SCE_KERNEL_ERROR_OPERATION_NOT_PERMITTED));
    };
    if command.buffer == 0 || command.size == 0 {
        return Err(sce_kernel_error(SCE_KERNEL_ERROR_OPERATION_NOT_PERMITTED));
    }
    if command.write_offset > command.size || bytes > command.size - command.write_offset {
        return Err(sce_kernel_error(SCE_KERNEL_ERROR_BUSY));
    }
    let Some(record) = command.buffer.checked_add(command.write_offset) else {
        return Err(sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT));
    };
    if !guest_range_is_readable(record, bytes) {
        return Err(sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT));
    }
    command.write_offset += bytes;
    command.command_count += commands;
    Ok(record)
}

/// Builds the header word shared by the commands whose size varies with their payload: the record
/// tag, the DWORDs the command occupies, and a command-specific discriminant.
fn record_header(tag: u32, dwords: u32, kind: u32) -> u32 {
    (tag & 0xff) | ((dwords & 0xff) << 8) | ((kind & 0xffff) << 16)
}

/// Rounds a byte count up to whole DWORDs.
fn round_up_to_dword(bytes: usize) -> u64 {
    u64::try_from(bytes).unwrap_or(0).div_ceil(4) * 4
}

/// Copies `length` bytes of guest memory that a caller has already proved readable.
fn copy_guest_range(destination: u64, source: u64, length: u64) {
    let Ok(length) = usize::try_from(length) else {
        return;
    };
    if length == 0 {
        return;
    }
    let mut bytes = vec![0_u8; length];
    read_guest_bytes(source, &mut bytes);
    write_guest_bytes(destination, &bytes);
}

/// Reports whether `length` bytes from `address` are mapped for the guest.
fn guest_range_is_readable(address: u64, length: u64) -> bool {
    if address == 0 || address.checked_add(length).is_none() {
        return false;
    }
    usize::try_from(length).is_ok_and(|length| process_guest_range_is_readable(address, length))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::AmprCommandBufferState;

    const EPERM: u64 = 0xffff_ffff_8002_0001;
    const EBUSY: u64 = 0xffff_ffff_8002_0010;
    const EINVAL: u64 = 0xffff_ffff_8002_0016;

    struct Fixture {
        object: Box<[u64; 8]>,
        memory: Box<[u8; 4096]>,
        state: ProbeDispatchState,
    }

    impl Fixture {
        fn new(size: u64) -> Self {
            let mut fixture = Self {
                object: Box::new([0_u64; 8]),
                memory: Box::new([0_u8; 4096]),
                state: ProbeDispatchState::default(),
            };
            let buffer = fixture.memory.as_ptr().addr() as u64;
            let object = fixture.object.as_ptr().addr() as u64;
            fixture.state.ampr_command_buffers.insert(
                object,
                AmprCommandBufferState {
                    buffer,
                    size,
                    ..AmprCommandBufferState::default()
                },
            );
            fixture
        }

        fn object(&self) -> u64 {
            self.object.as_ptr().addr() as u64
        }

        fn call(&mut self, nid: &str, arguments: [u64; 6]) -> u64 {
            dispatch(nid, arguments, &mut self.state).expect("the test names an owned NID")
        }

        fn word(&self, index: usize) -> u32 {
            let bytes = &self.memory[index * 4..index * 4 + 4];
            u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
        }

        fn text(&self, offset: usize, length: usize) -> Vec<u8> {
            self.memory[offset..offset + length].to_vec()
        }

        fn write_offset(&self) -> u64 {
            self.state.ampr_command_buffers[&self.object()].write_offset
        }

        fn command_count(&self) -> u64 {
            self.state.ampr_command_buffers[&self.object()].command_count
        }
    }

    fn args(values: &[u64]) -> [u64; 6] {
        let mut arguments = [0_u64; 6];
        arguments[..values.len()].copy_from_slice(values);
        arguments
    }

    fn message(text: &str) -> Vec<u8> {
        let mut bytes = text.as_bytes().to_vec();
        bytes.push(0);
        bytes
    }

    fn address(bytes: &[u8]) -> u64 {
        bytes.as_ptr().addr() as u64
    }

    #[test]
    fn the_error_codes_returned_here_are_the_kernel_codes_the_guest_expects() {
        assert_eq!(
            EPERM,
            sce_kernel_error(SCE_KERNEL_ERROR_OPERATION_NOT_PERMITTED)
        );
        assert_eq!(EBUSY, sce_kernel_error(SCE_KERNEL_ERROR_BUSY));
        assert_eq!(EINVAL, sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT));
    }

    #[test]
    fn a_nid_this_module_does_not_own_is_left_for_the_next_handler() {
        let mut state = ProbeDispatchState::default();
        assert_eq!(dispatch("mQ16-QdKv7k", [0; 6], &mut state), None);
        assert_eq!(dispatch("", [0; 6], &mut state), None);
    }

    #[test]
    fn resetting_the_gather_scatter_state_records_one_fixed_size_command() {
        let mut fixture = Fixture::new(0x100);
        let object = fixture.object();

        assert_eq!(fixture.call("YPxkUDhgoNI", args(&[object])), 0);

        assert_eq!(fixture.word(0), 6);
        assert_eq!(fixture.word(1), 0);
        assert_eq!(fixture.word(11), 0);
        assert_eq!(fixture.write_offset(), 0x30);
        assert_eq!(fixture.command_count(), 1);
    }

    #[test]
    fn a_nop_occupies_exactly_the_requested_number_of_dwords() {
        let mut fixture = Fixture::new(0x100);
        let object = fixture.object();

        assert_eq!(fixture.call("tNn5WBkta60", args(&[object, 4])), 0);

        assert_eq!(fixture.word(0), 0x0000_0411);
        assert_eq!(fixture.word(1), 0);
        assert_eq!(fixture.word(2), 0);
        assert_eq!(fixture.word(3), 0);
        assert_eq!(fixture.write_offset(), 16);
        assert_eq!(fixture.command_count(), 1);
    }

    #[test]
    fn a_nop_of_no_dwords_or_of_more_than_sixteen_is_rejected() {
        let mut fixture = Fixture::new(0x100);
        let object = fixture.object();

        assert_eq!(fixture.call("tNn5WBkta60", args(&[object, 0])), EINVAL);
        assert_eq!(fixture.call("tNn5WBkta60", args(&[object, 17])), EINVAL);

        assert_eq!(fixture.write_offset(), 0);
        assert_eq!(fixture.command_count(), 0);
    }

    #[test]
    fn a_nop_with_data_copies_its_payload_after_the_command_word() {
        let mut fixture = Fixture::new(0x100);
        let object = fixture.object();
        let payload = [0x1111_2222_u32, 0x3333_4444, 0x5555_6666];

        assert_eq!(
            fixture.call(
                "pFQ9UHpO52s",
                args(&[object, 3, payload.as_ptr().addr() as u64])
            ),
            0
        );

        assert_eq!(fixture.word(0), 0x0000_0411);
        assert_eq!(fixture.word(1), 0x1111_2222);
        assert_eq!(fixture.word(2), 0x3333_4444);
        assert_eq!(fixture.word(3), 0x5555_6666);
        assert_eq!(fixture.write_offset(), 16);
    }

    #[test]
    fn a_nop_with_data_rejects_an_unreadable_payload_rather_than_faulting() {
        let mut fixture = Fixture::new(0x100);
        let object = fixture.object();

        assert_eq!(fixture.call("pFQ9UHpO52s", args(&[object, 3, 0])), EINVAL);
        assert_eq!(fixture.call("pFQ9UHpO52s", args(&[object, 3, 8])), EINVAL);
        assert_eq!(fixture.call("pFQ9UHpO52s", args(&[object, 16, 0])), EINVAL);

        assert_eq!(fixture.write_offset(), 0);
    }

    #[test]
    fn a_nop_with_no_payload_still_occupies_its_own_dword() {
        let mut fixture = Fixture::new(0x100);
        let object = fixture.object();

        assert_eq!(fixture.call("pFQ9UHpO52s", args(&[object, 0, 0])), 0);

        assert_eq!(fixture.word(0), 0x0000_0111);
        assert_eq!(fixture.write_offset(), 4);
    }

    #[test]
    fn a_constructed_nop_carries_its_discriminant_and_optional_display_hint() {
        let mut fixture = Fixture::new(0x100);
        let object = fixture.object();
        let payload = [0xaabb_ccdd_u32, 0x0011_2233];
        let color = 0x00ff_8040_u32;

        assert_eq!(
            fixture.call(
                "GmOguNIsuKk",
                args(&[
                    object,
                    7,
                    payload.as_ptr().addr() as u64,
                    8,
                    std::ptr::from_ref(&color).addr() as u64,
                ])
            ),
            0
        );

        assert_eq!(fixture.word(0), 0x0007_0411);
        assert_eq!(fixture.word(1), 0x00ff_8040);
        assert_eq!(fixture.word(2), 0xaabb_ccdd);
        assert_eq!(fixture.word(3), 0x0011_2233);
        assert_eq!(fixture.write_offset(), 16);
    }

    #[test]
    fn a_constructed_nop_larger_than_sixteen_dwords_is_rejected() {
        let mut fixture = Fixture::new(0x100);
        let object = fixture.object();
        let payload = [0_u8; 64];

        assert_eq!(
            fixture.call("GmOguNIsuKk", args(&[object, 0, address(&payload), 60])),
            0
        );
        assert_eq!(fixture.write_offset(), 64);

        assert_eq!(
            fixture.call("GmOguNIsuKk", args(&[object, 0, address(&payload), 61])),
            EINVAL
        );
        assert_eq!(fixture.write_offset(), 64);
    }

    #[test]
    fn a_short_marker_records_one_command_holding_its_hint_and_terminated_message() {
        let mut fixture = Fixture::new(0x100);
        let object = fixture.object();
        let text = message("load");

        assert_eq!(
            fixture.call("sWbST0oQKsc", args(&[object, address(&text), 0x1234_5678])),
            0
        );

        assert_eq!(fixture.word(0), 0x0000_0412);
        assert_eq!(fixture.word(1), 0x1234_5678);
        assert_eq!(fixture.text(8, 8), b"load\0\0\0\0".to_vec());
        assert_eq!(fixture.write_offset(), 16);
        assert_eq!(fixture.command_count(), 1);
    }

    #[test]
    fn the_marker_entry_points_differ_only_in_the_discriminant_they_record() {
        let mut fixture = Fixture::new(0x100);
        let object = fixture.object();
        let text = message("frame");

        assert_eq!(
            fixture.call("4quckD2y7Pg", args(&[object, address(&text)])),
            0
        );
        assert_eq!(
            fixture.call("dXPaz65HNmk", args(&[object, address(&text)])),
            0
        );
        assert_eq!(fixture.call("mv0O8Zg0woU", args(&[object])), 0);

        assert_eq!(fixture.word(0), 0x0000_0412);
        assert_eq!(fixture.word(1), 0);
        assert_eq!(fixture.word(4), 0x0001_0412);
        assert_eq!(fixture.word(8), 0x0002_0212);
        assert_eq!(fixture.write_offset(), 40);
        assert_eq!(fixture.command_count(), 3);
    }

    #[test]
    fn a_marker_without_a_message_is_rejected_unless_the_command_allows_one() {
        let mut fixture = Fixture::new(0x100);
        let object = fixture.object();

        assert_eq!(fixture.call("4quckD2y7Pg", args(&[object, 0])), EINVAL);
        assert_eq!(fixture.call("dXPaz65HNmk", args(&[object, 0])), EINVAL);
        assert_eq!(fixture.call("f12ObAMEi9A", args(&[object, 8, 1])), EINVAL);
        assert_eq!(fixture.write_offset(), 0);

        assert_eq!(fixture.call("4UkZbYKVF7c", args(&[object, 9, 0, 0])), 0);
        assert_eq!(fixture.word(0), 0x0009_0212);
        assert_eq!(fixture.write_offset(), 8);
    }

    #[test]
    fn a_message_longer_than_one_command_is_split_across_continuations() {
        let mut fixture = Fixture::new(0x400);
        let object = fixture.object();
        let text = message(&"a".repeat(60));

        assert_eq!(
            fixture.call("4quckD2y7Pg", args(&[object, address(&text)])),
            0
        );

        assert_eq!(fixture.word(0), 0x0000_1012);
        assert_eq!(fixture.text(8, 56), vec![b'a'; 56]);
        assert_eq!(fixture.word(16), 0x0000_0313);
        assert_eq!(fixture.text(68, 4), b"aaaa".to_vec());
        assert_eq!(fixture.text(72, 4), vec![0; 4]);
        assert_eq!(fixture.write_offset(), 76);
        assert_eq!(fixture.command_count(), 2);
    }

    #[test]
    fn a_marker_message_that_is_never_terminated_is_bounded_rather_than_scanned_forever() {
        let mut fixture = Fixture::new(0x1000);
        let object = fixture.object();
        let text = vec![b'z'; MARKER_TEXT_LIMIT * 2];

        assert_eq!(
            fixture.call("4quckD2y7Pg", args(&[object, address(&text)])),
            0
        );

        assert_eq!(
            fixture.write_offset(),
            fixture.call("VGkEj4d6-Kg", args(&[address(&text)]))
        );
    }

    #[test]
    fn every_marker_size_query_matches_what_the_matching_builder_records() {
        let mut fixture = Fixture::new(0x1000);
        let object = fixture.object();
        let empty = message("");
        let short = message("stream");
        let long = message(&"b".repeat(200));

        for (query, builder, text) in [
            ("VGkEj4d6-Kg", "4quckD2y7Pg", &empty),
            ("tmfr97+ED5I", "sWbST0oQKsc", &short),
            ("0RdLmAh7WVo", "dXPaz65HNmk", &long),
            ("3OfeY4pzDV0", "f12ObAMEi9A", &long),
        ] {
            let expected = fixture.call(query, args(&[address(text), 0x40]));
            let before = fixture.write_offset();
            assert_eq!(
                fixture.call(builder, args(&[object, address(text), 0x40])),
                0
            );
            assert_eq!(fixture.write_offset() - before, expected);
        }

        let before = fixture.write_offset();
        assert_eq!(fixture.call("mv0O8Zg0woU", args(&[object])), 0);
        assert_eq!(
            fixture.write_offset() - before,
            fixture.call("pbnNnahE8vk", [0; 6])
        );
    }

    #[test]
    fn a_message_the_size_query_cannot_read_is_rejected() {
        let mut fixture = Fixture::new(0x100);

        assert_eq!(fixture.call("VGkEj4d6-Kg", args(&[0])), EINVAL);
        assert_eq!(fixture.call("0RdLmAh7WVo", args(&[8])), EINVAL);
    }

    #[test]
    fn every_nop_size_query_matches_what_the_matching_builder_records() {
        let mut fixture = Fixture::new(0x1000);
        let object = fixture.object();
        let payload = [0_u32; 15];

        for dwords in 1..=16 {
            let expected = fixture.call("NNIZ-FMyz3M", args(&[dwords]));
            let before = fixture.write_offset();
            assert_eq!(fixture.call("tNn5WBkta60", args(&[object, dwords])), 0);
            assert_eq!(fixture.write_offset() - before, expected);
        }
        for dwords in 0..=15 {
            let expected = fixture.call("Xp85BP3+BBI", args(&[dwords]));
            let before = fixture.write_offset();
            assert_eq!(
                fixture.call(
                    "pFQ9UHpO52s",
                    args(&[object, dwords, payload.as_ptr().addr() as u64])
                ),
                0
            );
            assert_eq!(fixture.write_offset() - before, expected);
        }
    }

    #[test]
    fn the_nop_size_queries_reject_the_counts_their_builders_reject() {
        let mut fixture = Fixture::new(0x100);

        assert_eq!(fixture.call("NNIZ-FMyz3M", args(&[0])), EINVAL);
        assert_eq!(fixture.call("NNIZ-FMyz3M", args(&[17])), EINVAL);
        assert_eq!(fixture.call("Xp85BP3+BBI", args(&[16])), EINVAL);
        assert_eq!(fixture.call("Xp85BP3+BBI", args(&[15])), 64);
    }

    #[test]
    fn the_fixed_size_queries_report_one_command_record() {
        let mut fixture = Fixture::new(0x100);

        assert_eq!(fixture.call("rddQYXM0CjM", [0; 6]), 0x30);
        assert_eq!(fixture.call("qesF88X4DRg", [0; 6]), 0x30);
        assert_eq!(fixture.call("7nXGDGMXSqo", [0; 6]), 0x30);
        assert_eq!(fixture.call("DXmgc5op8Yw", [0; 6]), 0x30);
        assert_eq!(fixture.call("Zi3dBUjgyXI", [0; 6]), 0x30);
        assert_eq!(fixture.call("C+IEj+BsAFM", [0; 6]), 0x30);
    }

    #[test]
    fn a_command_that_does_not_fit_in_the_remaining_space_is_refused() {
        let mut fixture = Fixture::new(0x30);
        let object = fixture.object();
        let text = message("late");

        assert_eq!(fixture.call("YPxkUDhgoNI", args(&[object])), 0);
        assert_eq!(fixture.write_offset(), 0x30);

        assert_eq!(fixture.call("YPxkUDhgoNI", args(&[object])), EBUSY);
        assert_eq!(fixture.call("tNn5WBkta60", args(&[object, 1])), EBUSY);
        assert_eq!(
            fixture.call("4quckD2y7Pg", args(&[object, address(&text)])),
            EBUSY
        );
        assert_eq!(fixture.call("mv0O8Zg0woU", args(&[object])), EBUSY);

        assert_eq!(fixture.write_offset(), 0x30);
        assert_eq!(fixture.command_count(), 1);
    }

    #[test]
    fn a_null_command_buffer_is_rejected_rather_than_faulting() {
        let mut fixture = Fixture::new(0x100);
        let text = message("orphan");

        assert_eq!(fixture.call("YPxkUDhgoNI", [0; 6]), EINVAL);
        assert_eq!(fixture.call("tNn5WBkta60", args(&[0, 4])), EINVAL);
        assert_eq!(fixture.call("pFQ9UHpO52s", args(&[0, 0, 0])), EINVAL);
        assert_eq!(
            fixture.call("4quckD2y7Pg", args(&[0, address(&text)])),
            EINVAL
        );
        assert_eq!(fixture.call("mv0O8Zg0woU", [0; 6]), EINVAL);
        assert_eq!(fixture.call("GmOguNIsuKk", args(&[0, 0, 0, 0, 0])), EINVAL);
    }

    #[test]
    fn a_command_buffer_with_no_memory_attached_refuses_to_record() {
        let mut state = ProbeDispatchState::default();
        let object = 0x4000_u64;
        let text = message("nowhere");

        assert_eq!(
            dispatch("YPxkUDhgoNI", args(&[object]), &mut state),
            Some(EPERM)
        );
        assert_eq!(
            dispatch("4quckD2y7Pg", args(&[object, address(&text)]), &mut state),
            Some(EPERM)
        );

        state
            .ampr_command_buffers
            .insert(object, AmprCommandBufferState::default());
        assert_eq!(
            dispatch("tNn5WBkta60", args(&[object, 1]), &mut state),
            Some(EPERM)
        );
    }

    #[test]
    fn the_accessors_report_the_memory_and_the_command_count_a_buffer_holds() {
        let mut fixture = Fixture::new(0x100);
        let object = fixture.object();
        let buffer = fixture.state.ampr_command_buffers[&object].buffer;

        assert_eq!(fixture.call("RPCAhx-aabE", args(&[object])), buffer);
        assert_eq!(fixture.call("gzndltBEzWc", args(&[object])), 0);

        assert_eq!(fixture.call("YPxkUDhgoNI", args(&[object])), 0);
        assert_eq!(fixture.call("tNn5WBkta60", args(&[object, 2])), 0);

        assert_eq!(fixture.call("gzndltBEzWc", args(&[object])), 2);
        assert_eq!(fixture.call("RPCAhx-aabE", args(&[object])), buffer);
    }

    #[test]
    fn the_accessors_report_nothing_for_a_buffer_the_runtime_never_saw() {
        let mut state = ProbeDispatchState::default();

        assert_eq!(dispatch("RPCAhx-aabE", [0; 6], &mut state), Some(0));
        assert_eq!(dispatch("gzndltBEzWc", [0; 6], &mut state), Some(0));
        assert_eq!(dispatch("RPCAhx-aabE", args(&[8]), &mut state), Some(0));
        assert_eq!(
            dispatch("gzndltBEzWc", args(&[0x4000]), &mut state),
            Some(0)
        );
    }

    #[test]
    fn an_untracked_command_buffer_reports_the_memory_recorded_in_the_guest_object() {
        let mut object = Box::new([0_u64; 8]);
        object[1] = 0x1234_5000;
        let mut state = ProbeDispatchState::default();
        let address = object.as_ptr().addr() as u64;

        assert_eq!(
            dispatch("RPCAhx-aabE", args(&[address]), &mut state),
            Some(0x1234_5000)
        );
    }

    #[test]
    fn the_completion_signal_is_recorded_under_both_of_its_exported_names() {
        let mut fixture = Fixture::new(0x100);
        let object = fixture.object();

        assert_eq!(
            fixture.call("o67gODLFpls", args(&[object, 0x9000, 7, 0xfeed])),
            0
        );

        assert_eq!(fixture.word(0), 2);
        assert_eq!(fixture.write_offset(), 0x30);
        assert_eq!(fixture.command_count(), 1);
    }

    #[test]
    fn the_address_completion_is_recorded_without_publishing_it() {
        let mut fixture = Fixture::new(0x100);
        let object = fixture.object();
        let mut completion = 0_u64;

        assert_eq!(
            fixture.call(
                "sJXyWHjP-F8",
                args(&[object, (&raw mut completion).addr() as u64, 1]),
            ),
            0
        );

        assert_eq!(completion, 0);
        assert_eq!(fixture.word(0), 3);
        assert_eq!(fixture.write_offset(), 0x30);
        assert_eq!(fixture.command_count(), 1);
    }
}
