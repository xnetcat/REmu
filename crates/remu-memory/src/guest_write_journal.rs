// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

//! Env-gated ring buffer naming every host-authored write into guest memory.
//!
//! A guest crash on a pointer read out of recycled heap names the victim address but not the
//! writer. Recording each host->guest publication with its site tag, span and payload shape lets
//! the fault handler answer "which emulator path last wrote these bytes" from the address alone,
//! which no after-the-fact log correlation can do.

use std::sync::atomic::{AtomicU32, AtomicU64, AtomicU8, Ordering};

use crate::signal_text::DiagnosticBuffer;

/// Set to `1` to record and dump the journal.
pub const GUEST_WRITE_JOURNAL_VARIABLE: &str = "REMU_GUEST_WRITE_JOURNAL";

/// Ring capacity in entries; a power of two so slot selection is a mask. Sized so a scattered
/// per-row texture publication cannot evict a whole frame of DMA fills.
const CAPACITY: usize = 64 * 1024;

/// Entries printed after the covering set, newest last.
const RECENT_DUMP_ENTRIES: u64 = 64;

/// Payload byte count retained per entry.
const PAYLOAD_BYTES: usize = 8;

/// Encodes "the run held more than one distinct byte" outside the `u8` value range.
const MIXED_PAYLOAD: u32 = 0x100;

/// Marks a uniformity verdict drawn from a prefix rather than the whole published run.
const SAMPLED_PAYLOAD: u32 = 0x200;

/// The emulator path that authored one guest range.
#[derive(Clone, Copy, Eq, PartialEq)]
pub enum GuestWriteSite {
    TextureWritebackRange,
    TextureWritebackRows,
    TextureWritebackTiled,
    DmaFill,
    DmaCopy,
    WriteData,
    ReleaseMem,
    GlobalWriteback,
    GlobalWritebackIdentity,
    GlobalWritebackDirect,
}

impl GuestWriteSite {
    const fn label(self) -> &'static [u8] {
        match self {
            Self::TextureWritebackRange => b"texture-writeback-range",
            Self::TextureWritebackRows => b"texture-writeback-rows",
            Self::TextureWritebackTiled => b"texture-writeback-tiled",
            Self::DmaFill => b"dma-fill",
            Self::DmaCopy => b"dma-copy",
            Self::WriteData => b"write-data",
            Self::ReleaseMem => b"release-mem",
            Self::GlobalWriteback => b"global-writeback",
            Self::GlobalWritebackIdentity => b"global-writeback-identity",
            Self::GlobalWritebackDirect => b"global-writeback-direct",
        }
    }

    const fn from_index(index: u32) -> Self {
        match index {
            0 => Self::TextureWritebackRange,
            1 => Self::TextureWritebackRows,
            2 => Self::TextureWritebackTiled,
            3 => Self::DmaFill,
            4 => Self::DmaCopy,
            5 => Self::WriteData,
            6 => Self::ReleaseMem,
            7 => Self::GlobalWriteback,
            8 => Self::GlobalWritebackIdentity,
            _ => Self::GlobalWritebackDirect,
        }
    }

    const fn index(self) -> u32 {
        match self {
            Self::TextureWritebackRange => 0,
            Self::TextureWritebackRows => 1,
            Self::TextureWritebackTiled => 2,
            Self::DmaFill => 3,
            Self::DmaCopy => 4,
            Self::WriteData => 5,
            Self::ReleaseMem => 6,
            Self::GlobalWriteback => 7,
            Self::GlobalWritebackIdentity => 8,
            Self::GlobalWritebackDirect => 9,
        }
    }
}

/// One recorded publication. `stamp` is `sequence + 1` once every other field is committed, so a
/// reader interrupted by a writer sees a changed stamp and skips the torn slot.
struct JournalSlot {
    stamp: AtomicU64,
    address: AtomicU64,
    byte_count: AtomicU64,
    payload: AtomicU64,
    submission: AtomicU64,
    site: AtomicU32,
    uniform: AtomicU32,
}

impl JournalSlot {
    const fn new() -> Self {
        Self {
            stamp: AtomicU64::new(0),
            address: AtomicU64::new(0),
            byte_count: AtomicU64::new(0),
            payload: AtomicU64::new(0),
            submission: AtomicU64::new(0),
            site: AtomicU32::new(0),
            uniform: AtomicU32::new(MIXED_PAYLOAD),
        }
    }
}

#[expect(
    clippy::declare_interior_mutable_const,
    reason = "array initialiser for a static of interior-mutable slots"
)]
const EMPTY_SLOT: JournalSlot = JournalSlot::new();

static SLOTS: [JournalSlot; CAPACITY] = [EMPTY_SLOT; CAPACITY];

static NEXT_SEQUENCE: AtomicU64 = AtomicU64::new(0);

const ENABLED_UNKNOWN: u8 = 0;
const ENABLED_YES: u8 = 1;
const ENABLED_NO: u8 = 2;

static ENABLED: AtomicU8 = AtomicU8::new(ENABLED_UNKNOWN);

/// Resolves the enable flag once. Callers outside signal context must reach this before a fault so
/// the handler never touches the environment.
#[must_use]
pub fn guest_write_journal_enabled() -> bool {
    match ENABLED.load(Ordering::Relaxed) {
        ENABLED_YES => true,
        ENABLED_NO => false,
        _ => {
            let enabled = std::env::var_os(GUEST_WRITE_JOURNAL_VARIABLE)
                .is_some_and(|value| value == "1" || value == "true");
            ENABLED.store(
                if enabled { ENABLED_YES } else { ENABLED_NO },
                Ordering::Relaxed,
            );
            enabled
        }
    }
}

/// Records one contiguous host-authored guest range.
///
/// `payload` is the bytes published at `address`; its first eight are retained verbatim, and its
/// uniformity — one repeated byte over the whole run — is what separates a metadata fill from
/// pixels. A payload shorter than `byte_count` marks the verdict as drawn from a prefix.
pub fn record_guest_write(
    site: GuestWriteSite,
    address: u64,
    byte_count: u64,
    payload: &[u8],
    submission: u64,
) {
    if !guest_write_journal_enabled() || byte_count == 0 {
        return;
    }
    let mut uniform = uniform_byte(payload);
    if u64::try_from(payload.len()).unwrap_or(u64::MAX) < byte_count {
        uniform |= SAMPLED_PAYLOAD;
    }
    let sequence = NEXT_SEQUENCE.fetch_add(1, Ordering::Relaxed);
    let slot = &SLOTS[(sequence as usize) & (CAPACITY - 1)];
    slot.stamp.store(0, Ordering::Release);
    slot.address.store(address, Ordering::Relaxed);
    slot.byte_count.store(byte_count, Ordering::Relaxed);
    slot.payload
        .store(payload_prefix(payload), Ordering::Relaxed);
    slot.submission.store(submission, Ordering::Relaxed);
    slot.site.store(site.index(), Ordering::Relaxed);
    slot.uniform.store(uniform, Ordering::Relaxed);
    slot.stamp.store(sequence.wrapping_add(1), Ordering::Release);
}

fn payload_prefix(payload: &[u8]) -> u64 {
    let mut prefix = [0_u8; PAYLOAD_BYTES];
    let copied = payload.len().min(PAYLOAD_BYTES);
    prefix[..copied].copy_from_slice(&payload[..copied]);
    u64::from_le_bytes(prefix)
}

/// Returns the single byte the run repeats, or [`MIXED_PAYLOAD`].
///
/// The shifted self-comparison is one memcmp, so a whole-surface run costs a linear scan the
/// enabled journal can afford and the disabled journal never performs.
fn uniform_byte(payload: &[u8]) -> u32 {
    match payload.split_first() {
        None => MIXED_PAYLOAD,
        Some((first, rest)) if rest == &payload[..rest.len()] => u32::from(*first),
        Some(_) => MIXED_PAYLOAD,
    }
}

/// Reports whether the high bytes of `address` repeat, the signature of a dereferenced fill.
///
/// Five bytes is the shortest prefix that no live guest pointer satisfies: guest space is
/// `0x0000_00xx_xxxx_xxxx`, so a repeated fifth byte means either an all-zero pointer read out of a
/// zero fill or a pointer whose high half came from a byte fill such as `0x40`.
#[must_use]
pub const fn address_looks_poisoned(address: u64) -> bool {
    let high = (address >> 24).to_le_bytes();
    let first = high[0];
    high[1] == first && high[2] == first && high[3] == first && high[4] == first
}

/// Bytes around the faulting address inside which an entry still counts as circumstantial.
const NEAR_FAULT_WINDOW: u64 = 64 * 1024;

/// Bytes past a register value inside which an entry names the structure that register points at.
/// The corrupted-slot loads seen so far are `[reg]`, `[reg+8]`, `[reg+0x118]` and `[reg+0x3e8]`,
/// so one page bounds the plausible field offsets.
const REGISTER_ANCHOR_WINDOW: u64 = 0x1000;

/// Register values below this cannot be guest heap and only produce noise matches.
const ANCHOR_FLOOR: u64 = 0x10000;

/// Writes every journal entry covering `fault_address`, near it, or covering memory any fault-time
/// register points at, then the newest entries, to stderr.
///
/// The faulting address of a poison dereference is the *derived* garbage pointer, which no
/// publication ever wrote; the corrupted slot it was loaded from is still in the interrupted
/// registers. Anchoring on register values is what lets one crash name the writer.
///
/// Signal-safe: it allocates nothing, takes no lock, and renders through a fixed buffer.
pub fn dump_guest_write_journal(fault_address: u64, anchors: &[u64]) {
    if ENABLED.load(Ordering::Relaxed) != ENABLED_YES {
        return;
    }
    let recorded = NEXT_SEQUENCE.load(Ordering::Acquire);
    let mut buffer = DiagnosticBuffer::new();
    buffer.push_bytes(b"[REMu][JOURNAL] fault=");
    buffer.push_hexadecimal(fault_address);
    buffer.push_bytes(b" poisoned=");
    buffer.push_bytes(if address_looks_poisoned(fault_address) {
        b"1"
    } else {
        b"0"
    });
    buffer.push_bytes(b" recorded=");
    buffer.push_signed(recorded.try_into().unwrap_or(i64::MAX));
    buffer.push_bytes(b" capacity=");
    buffer.push_signed(CAPACITY as i64);
    buffer.push_bytes(b"\n");
    write_diagnostic(&buffer);

    let live_from = recorded.saturating_sub(CAPACITY as u64);
    let mut covering = 0_u64;
    let mut matched = 0_u64;
    for sequence in live_from..recorded {
        let Some(entry) = read_slot(sequence) else {
            continue;
        };
        if entry.covers(fault_address) {
            covering += 1;
            matched += 1;
            emit_entry(b"covering", &entry, None, &mut buffer);
        } else if let Some(anchor) = anchors.iter().copied().find(|&anchor| {
            anchor >= ANCHOR_FLOOR
                && entry.intersects(anchor, anchor.saturating_add(REGISTER_ANCHOR_WINDOW))
        }) {
            matched += 1;
            emit_entry(b"register", &entry, Some(anchor), &mut buffer);
        } else if entry.intersects(
            fault_address.saturating_sub(NEAR_FAULT_WINDOW),
            fault_address.saturating_add(NEAR_FAULT_WINDOW),
        ) {
            matched += 1;
            emit_entry(b"near-fault", &entry, None, &mut buffer);
        }
    }
    let recent_from = recorded.saturating_sub(RECENT_DUMP_ENTRIES).max(live_from);
    for sequence in recent_from..recorded {
        if let Some(entry) = read_slot(sequence) {
            emit_entry(b"recent", &entry, None, &mut buffer);
        }
    }
    buffer.clear();
    buffer.push_bytes(b"[REMu][JOURNAL] end covering=");
    buffer.push_signed(covering.try_into().unwrap_or(i64::MAX));
    buffer.push_bytes(b" matched=");
    buffer.push_signed(matched.try_into().unwrap_or(i64::MAX));
    buffer.push_bytes(b"\n");
    write_diagnostic(&buffer);
}

struct JournalEntry {
    sequence: u64,
    site: GuestWriteSite,
    address: u64,
    byte_count: u64,
    payload: u64,
    uniform: u32,
    submission: u64,
}

impl JournalEntry {
    const fn covers(&self, address: u64) -> bool {
        address >= self.address && address - self.address < self.byte_count
    }

    /// Reports whether any published byte falls inside `[start, end)`.
    const fn intersects(&self, start: u64, end: u64) -> bool {
        self.address < end && start < self.address.saturating_add(self.byte_count)
    }
}

/// Reads the slot holding `sequence`, or `None` if it was recycled or is mid-write.
fn read_slot(sequence: u64) -> Option<JournalEntry> {
    let slot = &SLOTS[(sequence as usize) & (CAPACITY - 1)];
    let stamp = slot.stamp.load(Ordering::Acquire);
    if stamp != sequence.wrapping_add(1) {
        return None;
    }
    let entry = JournalEntry {
        sequence,
        site: GuestWriteSite::from_index(slot.site.load(Ordering::Relaxed)),
        address: slot.address.load(Ordering::Relaxed),
        byte_count: slot.byte_count.load(Ordering::Relaxed),
        payload: slot.payload.load(Ordering::Relaxed),
        uniform: slot.uniform.load(Ordering::Relaxed),
        submission: slot.submission.load(Ordering::Relaxed),
    };
    (slot.stamp.load(Ordering::Acquire) == stamp).then_some(entry)
}

fn emit_entry(
    kind: &'static [u8],
    entry: &JournalEntry,
    anchor: Option<u64>,
    buffer: &mut DiagnosticBuffer,
) {
    buffer.clear();
    buffer.push_bytes(b"[REMu][JOURNAL] ");
    buffer.push_bytes(kind);
    if let Some(anchor) = anchor {
        buffer.push_bytes(b" anchor=");
        buffer.push_hexadecimal(anchor);
    }
    buffer.push_bytes(b" seq=");
    buffer.push_signed(entry.sequence.try_into().unwrap_or(i64::MAX));
    buffer.push_bytes(b" site=");
    buffer.push_bytes(entry.site.label());
    buffer.push_bytes(b" addr=");
    buffer.push_hexadecimal(entry.address);
    buffer.push_bytes(b" bytes=");
    buffer.push_hexadecimal(entry.byte_count);
    buffer.push_bytes(b" payload=");
    buffer.push_hexadecimal(entry.payload);
    buffer.push_bytes(b" uniform=");
    if entry.uniform & MIXED_PAYLOAD != 0 {
        buffer.push_bytes(b"mixed");
    } else {
        buffer.push_hexadecimal(u64::from(entry.uniform & 0xff));
    }
    if entry.uniform & SAMPLED_PAYLOAD != 0 {
        buffer.push_bytes(b"(prefix)");
    }
    buffer.push_bytes(b" submission=");
    buffer.push_signed(entry.submission.try_into().unwrap_or(i64::MAX));
    buffer.push_bytes(b"\n");
    write_diagnostic(buffer);
}

#[cfg(unix)]
fn write_diagnostic(buffer: &DiagnosticBuffer) {
    let message = buffer.as_slice();
    // SAFETY: `write` is async-signal-safe and the buffer outlives the call. Short writes are
    // ignored because this is a last-chance crash diagnostic.
    unsafe {
        libc::write(
            libc::STDERR_FILENO,
            message.as_ptr().cast::<std::ffi::c_void>(),
            message.len(),
        );
    }
}

#[cfg(not(unix))]
fn write_diagnostic(_buffer: &DiagnosticBuffer) {}

#[cfg(test)]
mod tests {
    use super::{
        CAPACITY, ENABLED, ENABLED_YES, GuestWriteSite, JournalEntry, MIXED_PAYLOAD, NEXT_SEQUENCE,
        SAMPLED_PAYLOAD, address_looks_poisoned, payload_prefix, read_slot, record_guest_write,
        uniform_byte,
    };
    use std::sync::atomic::Ordering;

    fn enable_journal() {
        ENABLED.store(ENABLED_YES, Ordering::Relaxed);
    }

    #[test]
    fn recognizes_pointers_assembled_from_a_repeated_fill_byte() {
        assert!(address_looks_poisoned(0x4040_4040_4040_4158));
        assert!(address_looks_poisoned(0x4040_4040_4040_40a0));
        assert!(address_looks_poisoned(0x0000_0000_0000_00a0));
        assert!(!address_looks_poisoned(0x0000_0071_c13c_4a80));
        assert!(!address_looks_poisoned(0x0000_0090_0c1c_0320));
        assert!(!address_looks_poisoned(0x0000_7f00_1234_5678));
    }

    #[test]
    fn reports_the_repeated_byte_of_a_metadata_fill_and_mixed_pixels() {
        assert_eq!(uniform_byte(&[0x40; 4096]), 0x40);
        assert_eq!(uniform_byte(&[0x00; 32]), 0x00);
        assert_eq!(uniform_byte(&[0x40]), 0x40);
        assert_eq!(uniform_byte(&[]), MIXED_PAYLOAD);
        let mut pixels = [0x40_u8; 4096];
        pixels[4095] = 0x41;
        assert_eq!(uniform_byte(&pixels), MIXED_PAYLOAD);
    }

    #[test]
    fn retains_the_leading_payload_bytes_of_a_short_run() {
        assert_eq!(payload_prefix(&[0x40; 8]), 0x4040_4040_4040_4040);
        assert_eq!(payload_prefix(&[0x11, 0x22]), 0x0000_0000_0000_2211);
        assert_eq!(payload_prefix(&[]), 0);
    }

    #[test]
    fn covers_only_the_bytes_a_publication_wrote() {
        let entry = JournalEntry {
            sequence: 0,
            site: GuestWriteSite::DmaFill,
            address: 0x900c_1c00_00,
            byte_count: 0x1000,
            payload: 0,
            uniform: 0x40,
            submission: 0,
        };
        assert!(entry.covers(0x900c_1c00_00));
        assert!(entry.covers(0x900c_1c00_00 + 0xfff));
        assert!(!entry.covers(0x900c_1c00_00 + 0x1000));
        assert!(!entry.covers(0x900c_1c00_00 - 1));
    }

    #[test]
    fn recorded_entries_survive_a_full_ring_wrap_and_name_their_site() {
        enable_journal();
        let first = NEXT_SEQUENCE.load(Ordering::Relaxed);
        for index in 0..CAPACITY as u64 + 4 {
            record_guest_write(
                GuestWriteSite::DmaFill,
                0x9000_0000 + index * 0x100,
                0x100,
                &[0x40; 0x100],
                index,
            );
        }
        let last = NEXT_SEQUENCE.load(Ordering::Relaxed) - 1;
        let entry = read_slot(last).expect("the newest entry is live");
        assert_eq!(entry.uniform, 0x40);
        assert_eq!(entry.payload, 0x4040_4040_4040_4040);
        assert_eq!(entry.byte_count, 0x100);
        assert!(entry.site == GuestWriteSite::DmaFill);
        assert!(
            read_slot(first).is_none(),
            "the wrapped slot no longer answers for its original sequence"
        );
    }

    #[test]
    fn marks_a_uniformity_verdict_taken_from_less_than_the_published_run() {
        enable_journal();
        record_guest_write(
            GuestWriteSite::DmaCopy,
            0x9100_0000,
            0x2000,
            &[0x00; 8],
            7,
        );
        let entry = read_slot(NEXT_SEQUENCE.load(Ordering::Relaxed) - 1)
            .expect("the newest entry is live");
        assert_eq!(entry.uniform, 0x00 | SAMPLED_PAYLOAD);
        assert_eq!(entry.byte_count, 0x2000);
    }
}
