// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

//! Env-gated ring buffer naming host-authored writes and watched guest page openings.
//!
//! A guest crash on a pointer read out of recycled heap names the victim address but not the
//! writer. Recording each host->guest publication with its site tag, span and payload shape lets
//! the fault handler answer "which emulator path last wrote these bytes" from the address alone.
//! Recording the first trapped guest store after each watched-page rearm additionally identifies
//! guest code which took CPU ownership of a later victim page.

use std::sync::atomic::{AtomicU8, AtomicU32, AtomicU64, Ordering};

use crate::signal_text::DiagnosticBuffer;

/// Set to `1` to record and dump the journal.
pub const GUEST_WRITE_JOURNAL_VARIABLE: &str = "REMU_GUEST_WRITE_JOURNAL";

/// Ring capacity in entries; a power of two so slot selection is a mask.
///
/// Corruption sits latent: MallocBinned3 only trips over a poisoned free-list entry when it next
/// pops it, which the first soak measured at more than the ~5 seconds a 64k ring held (all 15
/// crashes dumped `matched=0` against ~480k recorded publications). Four million entries hold
/// minutes of publication traffic — the whole run — at ~200 MiB, allocated only when the journal
/// is armed.
#[cfg(not(test))]
const CAPACITY: usize = 1 << 22;
#[cfg(test)]
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
    /// AMPR file read replayed at submit time into a destination recorded earlier.
    AmprReadReplay,
    /// AMPR completion/result word written on every submission of a reusable command buffer.
    AmprCompletion,
    /// AvPlayer video/audio frame staged into guest-allocated buffers off the decoder timeline.
    AvPlayerFrame,
    /// HLE'd guest `memset`.
    GuestMemset,
    /// HLE'd guest `memcpy`/`memmove`.
    GuestMemcpy,
    /// Runtime bulk byte publication (save data and similar guest-named outputs).
    RuntimeBytes,
    /// REmu's mspace suballocator zero-filling a `calloc` block inside a guest region.
    MspaceZero,
    /// REmu's mspace suballocator migrating bytes for `realloc`.
    MspaceCopy,
    /// In-place mutation of guest AGC shader headers (pointer relocation, register patches).
    ShaderPatch,
    /// `MAP_FIXED` anonymous mapping: the whole range becomes zeros without any byte store.
    MapZero,
    /// Commit-on-protect remap: the range's contents are replaced by the committed backing's.
    MapCommit,
    /// The first guest CPU store which opened one protected host page for later writes.
    ///
    /// Write watching traps once per page/rearm, not once per instruction. The journal therefore
    /// records the whole page as potentially changed while retaining the exact fault address and
    /// guest instruction pointer separately. This lets a later poison crash identify the guest
    /// code which first acquired write ownership of the victim page without pretending that one
    /// fault names every subsequent store on it.
    GuestCpuPageOpen,
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
            Self::AmprReadReplay => b"ampr-read-replay",
            Self::AmprCompletion => b"ampr-completion",
            Self::AvPlayerFrame => b"avplayer-frame",
            Self::GuestMemset => b"guest-memset",
            Self::GuestMemcpy => b"guest-memcpy",
            Self::RuntimeBytes => b"runtime-bytes",
            Self::MspaceZero => b"mspace-zero",
            Self::MspaceCopy => b"mspace-copy",
            Self::ShaderPatch => b"shader-patch",
            Self::MapZero => b"map-zero",
            Self::MapCommit => b"map-commit",
            Self::GuestCpuPageOpen => b"guest-cpu-page-open",
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
            9 => Self::GlobalWritebackDirect,
            10 => Self::AmprReadReplay,
            11 => Self::AmprCompletion,
            12 => Self::AvPlayerFrame,
            13 => Self::GuestMemset,
            14 => Self::GuestMemcpy,
            15 => Self::RuntimeBytes,
            16 => Self::MspaceZero,
            17 => Self::MspaceCopy,
            18 => Self::ShaderPatch,
            19 => Self::MapZero,
            20 => Self::MapCommit,
            21 => Self::GuestCpuPageOpen,
            _ => Self::MapCommit,
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
            Self::AmprReadReplay => 10,
            Self::AmprCompletion => 11,
            Self::AvPlayerFrame => 12,
            Self::GuestMemset => 13,
            Self::GuestMemcpy => 14,
            Self::RuntimeBytes => 15,
            Self::MspaceZero => 16,
            Self::MspaceCopy => 17,
            Self::ShaderPatch => 18,
            Self::MapZero => 19,
            Self::MapCommit => 20,
            Self::GuestCpuPageOpen => 21,
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

/// The ring is ~200 MiB at production capacity, so it exists only when the journal is armed.
/// Recording paths allocate it on first use (never in signal context); the fault-time dump only
/// ever observes it through [`std::sync::OnceLock::get`], which is a plain atomic load.
static SLOTS: std::sync::OnceLock<Box<[JournalSlot]>> = std::sync::OnceLock::new();

fn slots_for_recording() -> &'static [JournalSlot] {
    SLOTS.get_or_init(|| (0..CAPACITY).map(|_| JournalSlot::new()).collect())
}

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
            if enabled {
                // Pay the ~200 MiB ring allocation at resolution time (process start) rather
                // than as a hitch inside the first recorded publication.
                let _ = slots_for_recording();
            }
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
    let slot = &slots_for_recording()[(sequence as usize) & (CAPACITY - 1)];
    slot.stamp.store(0, Ordering::Release);
    slot.address.store(address, Ordering::Relaxed);
    slot.byte_count.store(byte_count, Ordering::Relaxed);
    slot.payload
        .store(payload_prefix(payload), Ordering::Relaxed);
    slot.submission.store(submission, Ordering::Relaxed);
    slot.site.store(site.index(), Ordering::Relaxed);
    slot.uniform.store(uniform, Ordering::Relaxed);
    slot.stamp
        .store(sequence.wrapping_add(1), Ordering::Release);
}

/// Records the guest instruction which opened a watched host page for CPU writes.
///
/// This is called from the SIGBUS/SIGSEGV write-watch handler after the page is proven live. It
/// must remain async-signal-safe: the journal allocation and environment lookup happen when the
/// handler is installed, and this path performs only atomic operations on that existing ring.
/// `payload` stores the exact faulting byte address and `submission` stores the interrupted guest
/// instruction pointer; [`emit_entry`] gives those fields their precise names for this site.
pub(crate) fn record_guest_cpu_page_open(
    page_address: u64,
    page_byte_count: u64,
    fault_address: u64,
    program_counter: u64,
) {
    if ENABLED.load(Ordering::Relaxed) != ENABLED_YES || page_byte_count == 0 {
        return;
    }
    let Some(slots) = SLOTS.get() else {
        return;
    };
    let sequence = NEXT_SEQUENCE.fetch_add(1, Ordering::Relaxed);
    let slot = &slots[(sequence as usize) & (CAPACITY - 1)];
    slot.stamp.store(0, Ordering::Release);
    slot.address.store(page_address, Ordering::Relaxed);
    slot.byte_count.store(page_byte_count, Ordering::Relaxed);
    slot.payload.store(fault_address, Ordering::Relaxed);
    slot.submission.store(program_counter, Ordering::Relaxed);
    slot.site
        .store(GuestWriteSite::GuestCpuPageOpen.index(), Ordering::Relaxed);
    slot.uniform.store(MIXED_PAYLOAD, Ordering::Relaxed);
    slot.stamp
        .store(sequence.wrapping_add(1), Ordering::Release);
}

/// Returns the already-resolved journal state without consulting the environment.
///
/// Fault handlers use this to avoid decoding a machine context on every ordinary watched write
/// when diagnostics are disabled. [`guest_write_journal_enabled`] resolves this state before the
/// handlers are installed.
pub(crate) fn guest_cpu_page_journal_armed() -> bool {
    ENABLED.load(Ordering::Relaxed) == ENABLED_YES
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
    // A fault raised by the dump's own probing must not recurse into a second dump.
    static DUMPING: AtomicU8 = AtomicU8::new(0);
    if DUMPING.swap(1, Ordering::AcqRel) != 0 {
        let mut buffer = DiagnosticBuffer::new();
        buffer.push_bytes(b"[REMu][JOURNAL] nested fault during dump; skipping\n");
        write_diagnostic(&buffer);
        return;
    }
    if ENABLED.load(Ordering::Relaxed) == ENABLED_YES {
        dump_guest_write_journal_locked(fault_address, anchors);
    } else {
        // The journal is deliberately expensive and normally disabled, but the bounded memory
        // neighbourhood dump is cheap, allocation-free and often the only way to recover the
        // corrupted object or vtable after Rosetta forwards the fatal signal. Keep that last-
        // chance evidence available on every unhandled fault instead of coupling it to the
        // 200 MiB publication history.
        let mut buffer = DiagnosticBuffer::new();
        dump_anchor_neighbourhoods(fault_address, anchors, &mut buffer);
    }
    DUMPING.store(0, Ordering::Release);
}

fn dump_guest_write_journal_locked(fault_address: u64, anchors: &[u64]) {
    /// Matched-entry print ceiling; the counters in the end line stay exact.
    const REGISTER_PRINT_CAP: u64 = 200;
    const NEAR_PRINT_CAP: u64 = 100;
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
    let mut register_matches = 0_u64;
    let mut near_matches = 0_u64;
    for sequence in live_from..recorded {
        let Some(entry) = read_slot(sequence) else {
            continue;
        };
        if entry.covers(fault_address) {
            covering += 1;
            emit_entry(b"covering", &entry, None, &mut buffer);
        } else if let Some(anchor) = anchors.iter().copied().find(|&anchor| {
            anchor >= ANCHOR_FLOOR
                && entry.intersects(anchor, anchor.saturating_add(REGISTER_ANCHOR_WINDOW))
        }) {
            register_matches += 1;
            if register_matches <= REGISTER_PRINT_CAP {
                emit_entry(b"register", &entry, Some(anchor), &mut buffer);
            }
        } else if entry.intersects(
            fault_address.saturating_sub(NEAR_FAULT_WINDOW),
            fault_address.saturating_add(NEAR_FAULT_WINDOW),
        ) {
            near_matches += 1;
            if near_matches <= NEAR_PRINT_CAP {
                emit_entry(b"near-fault", &entry, None, &mut buffer);
            }
        }
    }
    let matched = covering + register_matches + near_matches;
    let recent_from = recorded.saturating_sub(RECENT_DUMP_ENTRIES).max(live_from);
    for sequence in recent_from..recorded {
        if let Some(entry) = read_slot(sequence) {
            emit_entry(b"recent", &entry, None, &mut buffer);
        }
    }

    // Inventory of every uniform fill the emulator applied, newest first. A poison crash whose
    // victim holds 0x40s while this inventory is empty proves the fill came from guest stores the
    // journal cannot see, which redirects the whole investigation.
    let mut forty_fills = 0_u64;
    let mut bulk_zero_fills = 0_u64;
    for sequence in (live_from..recorded).rev() {
        let Some(entry) = read_slot(sequence) else {
            continue;
        };
        let uniform = entry.uniform & !SAMPLED_PAYLOAD;
        if uniform == 0x40 {
            forty_fills += 1;
            if forty_fills <= 96 {
                emit_entry(b"fill-0x40", &entry, None, &mut buffer);
            }
        } else if uniform == 0x00 && entry.byte_count >= 0x1000 {
            bulk_zero_fills += 1;
            if bulk_zero_fills <= 32 {
                emit_entry(b"fill-zero", &entry, None, &mut buffer);
            }
        }
    }

    buffer.clear();
    buffer.push_bytes(b"[REMu][JOURNAL] end covering=");
    buffer.push_signed(covering.try_into().unwrap_or(i64::MAX));
    buffer.push_bytes(b" matched=");
    buffer.push_signed(matched.try_into().unwrap_or(i64::MAX));
    buffer.push_bytes(b" fill40=");
    buffer.push_signed(forty_fills.try_into().unwrap_or(i64::MAX));
    buffer.push_bytes(b" fillzero=");
    buffer.push_signed(bulk_zero_fills.try_into().unwrap_or(i64::MAX));
    buffer.push_bytes(b"\n");
    write_diagnostic(&buffer);

    // Last because a nested fault while probing guest bytes would truncate everything after it:
    // the raw neighbourhood of each plausible victim pointer. The shape of the damage — an
    // unbroken fill, PM4-packet-like structure, or one isolated dword — separates writers no
    // journal entry can.
    dump_anchor_neighbourhoods(fault_address, anchors, &mut buffer);
}

/// Hex-dumps the bytes around each guest-plausible anchor and the fault address, clamped to each
/// anchor's own 4 KiB page.
///
/// Readability is proven with `write(pipe, window, len)` before any load: a pipe write must copy
/// the window into the pipe buffer, so an unreadable window reports EFAULT instead of faulting
/// the handler. This matters because the interesting anchors are frequently garbage — the poison
/// pointer itself sits in a register, and a BUS_ADRERR fault address is mapped but unreadable.
/// (`/dev/null` cannot serve as the probe sink: Darwin's null driver discards writes without
/// reading the buffer, so the probe passed and the load faulted inside the handler.)
///
/// The probed windows total at most 33 × 128 bytes, far below the pipe's capacity, so the unread
/// pipe never blocks the writer.
#[cfg(unix)]
fn dump_anchor_neighbourhoods(fault_address: u64, anchors: &[u64], buffer: &mut DiagnosticBuffer) {
    const HALF_WINDOW: u64 = 64;
    let mut probe = [0_i32; 2];
    // SAFETY: pipe is a plain syscall filling the two-descriptor array; both ends are closed
    // before returning.
    if unsafe { libc::pipe(probe.as_mut_ptr()) } != 0 {
        return;
    }
    let probe_writer = probe[1];
    let mut printed = [0_u64; 33];
    let mut printed_count = 0;
    for &anchor in anchors.iter().chain(std::iter::once(&fault_address)) {
        if anchor < ANCHOR_FLOOR {
            continue;
        }
        let page = anchor & !0xfff;
        // Deduplicate by window, not by page: the fault registers routinely hold both a table
        // base and the victim slot inside it, and dropping the second window because it shares
        // the first one's page discarded exactly the bytes the dump exists to show.
        let window = anchor & !(HALF_WINDOW - 1);
        if printed[..printed_count].contains(&window) {
            continue;
        }
        if printed_count == printed.len() {
            break;
        }
        printed[printed_count] = window;
        printed_count += 1;
        let start = anchor.saturating_sub(HALF_WINDOW).max(page);
        let end = anchor
            .saturating_add(HALF_WINDOW)
            .min(page.saturating_add(0x1000));
        let length = (end - start) as usize;
        // SAFETY: the pipe write copies the window into the pipe buffer; an unreadable window
        // returns EFAULT rather than raising a nested fault.
        let probed = unsafe { libc::write(probe_writer, start as *const std::ffi::c_void, length) };
        buffer.clear();
        buffer.push_bytes(b"[REMu][JOURNAL] bytes at ");
        buffer.push_hexadecimal(start);
        buffer.push_bytes(b" around ");
        buffer.push_hexadecimal(anchor);
        if probed != length as isize {
            buffer.push_bytes(b": unreadable\n");
            write_diagnostic(buffer);
            continue;
        }
        buffer.push_bytes(b": ");
        for offset in start..end {
            // SAFETY: the kernel just read this exact window through the probe write.
            let byte = unsafe { std::ptr::read_volatile(offset as *const u8) };
            let digits = b"0123456789abcdef";
            buffer.push_bytes(&[digits[(byte >> 4) as usize], digits[(byte & 0xf) as usize]]);
        }
        buffer.push_bytes(b"\n");
        write_diagnostic(buffer);
    }
    // SAFETY: close is async-signal-safe and both descriptors came from the pipe above.
    unsafe {
        libc::close(probe[0]);
        libc::close(probe[1]);
    }
}

#[cfg(not(unix))]
fn dump_anchor_neighbourhoods(
    _fault_address: u64,
    _anchors: &[u64],
    _buffer: &mut DiagnosticBuffer,
) {
}

/// Reports a signal no recovery claimed, from signal context, then dumps the journal.
///
/// The SHA-NI SIGILL recovery re-raises unrecognised illegal instructions with the default
/// disposition, which previously killed the process with no pc, no address and no journal — three
/// of the first fifteen soak runs died that way and yielded nothing. `anchors` should carry the
/// interrupted registers when the caller has them; the pc is always used as one.
pub fn report_unrecovered_signal(name: &[u8], pc: u64, fault_address: u64, anchors: &[u64]) {
    let mut buffer = DiagnosticBuffer::new();
    buffer.push_bytes(b"[REMu][WATCH] unrecovered ");
    buffer.push_bytes(name);
    buffer.push_bytes(b" pc=");
    buffer.push_hexadecimal(pc);
    buffer.push_bytes(b" addr=");
    buffer.push_hexadecimal(fault_address);
    buffer.push_bytes(b"; forwarding\n");
    write_diagnostic(&buffer);
    let mut all = [0_u64; 33];
    let count = anchors.len().min(32);
    all[..count].copy_from_slice(&anchors[..count]);
    all[count] = pc;
    dump_guest_write_journal(fault_address, &all[..=count]);
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
///
/// Runs in signal context: it must not allocate, so an unallocated ring reads as empty.
fn read_slot(sequence: u64) -> Option<JournalEntry> {
    let slot = SLOTS.get()?.get((sequence as usize) & (CAPACITY - 1))?;
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
    if entry.site == GuestWriteSite::GuestCpuPageOpen {
        buffer.push_bytes(b" fault=");
        buffer.push_hexadecimal(entry.payload);
        buffer.push_bytes(b" pc=");
        buffer.push_hexadecimal(entry.submission);
    } else {
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
    }
    buffer.push_bytes(b"\n");
    write_diagnostic(buffer);
}

#[cfg(unix)]
fn write_diagnostic(buffer: &DiagnosticBuffer) {
    let message = buffer.as_slice();
    // SAFETY: `write` is async-signal-safe and the buffer outlives the call. Short writes are
    // ignored because this is a last-chance crash diagnostic. The preserved real stderr is used
    // because bytes queued in the asynchronous stderr pipe die with the process.
    unsafe {
        libc::write(
            crate::crash_output_fd(),
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
        SAMPLED_PAYLOAD, address_looks_poisoned, payload_prefix, read_slot,
        record_guest_cpu_page_open, record_guest_write, slots_for_recording, uniform_byte,
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
        record_guest_write(GuestWriteSite::DmaCopy, 0x9100_0000, 0x2000, &[0x00; 8], 7);
        let entry =
            read_slot(NEXT_SEQUENCE.load(Ordering::Relaxed) - 1).expect("the newest entry is live");
        assert_eq!(entry.uniform, SAMPLED_PAYLOAD);
        assert_eq!(entry.byte_count, 0x2000);
    }

    #[test]
    fn watched_guest_page_open_retains_the_exact_fault_and_guest_pc() {
        enable_journal();
        let _ = slots_for_recording();
        record_guest_cpu_page_open(0x900c_8d00_00, 0x4000, 0x900c_8d05_a0, 0x8012_3456_7);
        let entry = read_slot(NEXT_SEQUENCE.load(Ordering::Relaxed) - 1)
            .expect("the watched guest page opening is live");
        assert!(entry.site == GuestWriteSite::GuestCpuPageOpen);
        assert_eq!(entry.address, 0x900c_8d00_00);
        assert_eq!(entry.byte_count, 0x4000);
        assert_eq!(entry.payload, 0x900c_8d05_a0);
        assert_eq!(entry.submission, 0x8012_3456_7);
        assert!(entry.covers(0x900c_8d05_a0));
    }
}
