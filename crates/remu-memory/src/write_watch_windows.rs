// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

//! Precise Windows CPU-write tracking and mapping-lifetime validation.
//!
//! Writable guest pages are temporarily made read-only. A vectored exception handler recognizes
//! write access violations in watched pages, marks every overlapping watch dirty, restores the
//! page's writable protection, and resumes the guest instruction. This mirrors the Darwin
//! protection-fault backend while retaining Windows page attributes and coherent alias behavior.

use std::collections::{BTreeMap, BTreeSet};
use std::io;
use std::sync::atomic::{AtomicBool, AtomicU8, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use windows::Win32::Foundation::EXCEPTION_ACCESS_VIOLATION;
use windows::Win32::System::Diagnostics::Debug::{
    AddVectoredExceptionHandler, EXCEPTION_CONTINUE_EXECUTION, EXCEPTION_CONTINUE_SEARCH,
    EXCEPTION_POINTERS,
};
use windows::Win32::System::Memory::{
    MEMORY_BASIC_INFORMATION, PAGE_EXECUTE_READ, PAGE_EXECUTE_READWRITE, PAGE_EXECUTE_WRITECOPY,
    PAGE_PROTECTION_FLAGS, PAGE_READONLY, PAGE_READWRITE, PAGE_WRITECOPY, VirtualProtect,
    VirtualQuery,
};

const HOST_PAGE_SIZE: u64 = 4096;
const WATCH_ACTIVE: u8 = 0;
const WATCH_REPROTECTING: u8 = 1;
const WATCH_REPROTECTED: u8 = 2;
const WATCH_REPLACED: u8 = 3;

#[derive(Debug)]
struct WatchedRange {
    id: u64,
    address: u64,
    byte_count: u64,
    dirty: AtomicBool,
    dirty_pages: Mutex<BTreeSet<u64>>,
    generation: AtomicU64,
    retirement: AtomicU8,
}

#[derive(Debug)]
struct WatchedPage {
    watchers: BTreeSet<u64>,
    writable_protection: PAGE_PROTECTION_FLAGS,
    readonly_protection: PAGE_PROTECTION_FLAGS,
    open: bool,
}

#[derive(Default)]
struct WatchState {
    next_id: u64,
    ranges: BTreeMap<u64, Arc<WatchedRange>>,
    pages: BTreeMap<u64, WatchedPage>,
}

fn state() -> &'static Mutex<WatchState> {
    static STATE: OnceLock<Mutex<WatchState>> = OnceLock::new();
    STATE.get_or_init(|| Mutex::new(WatchState::default()))
}

fn align_down(value: u64) -> u64 {
    value & !(HOST_PAGE_SIZE - 1)
}

fn align_up(value: u64) -> Option<u64> {
    value
        .checked_add(HOST_PAGE_SIZE - 1)
        .map(|value| value & !(HOST_PAGE_SIZE - 1))
}

fn overlaps(left: (u64, u64), right: (u64, u64)) -> bool {
    let Some(left_end) = left.0.checked_add(left.1) else {
        return false;
    };
    let Some(right_end) = right.0.checked_add(right.1) else {
        return false;
    };
    left.0 < right_end && right.0 < left_end
}

fn readonly_protection(protection: PAGE_PROTECTION_FLAGS) -> Option<PAGE_PROTECTION_FLAGS> {
    let modifiers = protection.0 & !0xff;
    let base = protection.0 & 0xff;
    let readonly = match PAGE_PROTECTION_FLAGS(base) {
        PAGE_READWRITE | PAGE_WRITECOPY => PAGE_READONLY,
        PAGE_EXECUTE_READWRITE | PAGE_EXECUTE_WRITECOPY => PAGE_EXECUTE_READ,
        _ => return None,
    };
    Some(PAGE_PROTECTION_FLAGS(readonly.0 | modifiers))
}

fn set_range_protection(
    address: u64,
    byte_count: u64,
    protection: PAGE_PROTECTION_FLAGS,
) -> io::Result<()> {
    let byte_count = usize::try_from(byte_count)
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "protection span is too large"))?;
    let mut old = PAGE_PROTECTION_FLAGS::default();
    unsafe {
        VirtualProtect(
            address as *const std::ffi::c_void,
            byte_count,
            protection,
            &raw mut old,
        )
    }
    .map_err(|_| io::Error::last_os_error())
}

fn set_page_protection(address: u64, protection: PAGE_PROTECTION_FLAGS) -> io::Result<()> {
    set_range_protection(address, HOST_PAGE_SIZE, protection)
}

fn query_page_region(address: u64) -> io::Result<(PAGE_PROTECTION_FLAGS, u64)> {
    let mut information = MEMORY_BASIC_INFORMATION::default();
    let queried = unsafe {
        VirtualQuery(
            Some(address as *const std::ffi::c_void),
            &raw mut information,
            size_of::<MEMORY_BASIC_INFORMATION>(),
        )
    };
    if queried != size_of::<MEMORY_BASIC_INFORMATION>() {
        return Err(io::Error::last_os_error());
    }
    let base = information.BaseAddress as usize as u64;
    let end = base
        .checked_add(u64::try_from(information.RegionSize).map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "memory region size is too large",
            )
        })?)
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "memory region overflows"))?;
    if address < base || address >= end {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "VirtualQuery returned a region that does not cover the requested page",
        ));
    }
    Ok((information.Protect, end))
}

fn install_exception_handler() -> io::Result<()> {
    static INSTALLED: OnceLock<Result<(), i32>> = OnceLock::new();
    INSTALLED
        .get_or_init(|| {
            let handler = unsafe { AddVectoredExceptionHandler(1, Some(write_fault_handler)) };
            if handler.is_null() {
                Err(io::Error::last_os_error().raw_os_error().unwrap_or(1))
            } else {
                Ok(())
            }
        })
        .map_err(io::Error::from_raw_os_error)
}

unsafe extern "system" fn write_fault_handler(exception: *mut EXCEPTION_POINTERS) -> i32 {
    let Some(exception) = (unsafe { exception.as_ref() }) else {
        return EXCEPTION_CONTINUE_SEARCH;
    };
    let Some(record) = (unsafe { exception.ExceptionRecord.as_ref() }) else {
        return EXCEPTION_CONTINUE_SEARCH;
    };
    if record.ExceptionCode != EXCEPTION_ACCESS_VIOLATION
        || record.NumberParameters < 2
        || record.ExceptionInformation[0] != 1
    {
        return EXCEPTION_CONTINUE_SEARCH;
    }
    let fault_address = record.ExceptionInformation[1] as u64;
    let page_address = align_down(fault_address);
    if crate::guest_write_journal::guest_cpu_page_journal_armed() {
        let program_counter = unsafe { exception.ContextRecord.as_ref() }.map_or(0, |context| {
            #[cfg(target_arch = "x86_64")]
            {
                context.Rip
            }
            #[cfg(not(target_arch = "x86_64"))]
            {
                let _ = context;
                0
            }
        });
        let _ = crate::guest_write_journal::record_guest_cpu_page_open(
            page_address,
            HOST_PAGE_SIZE,
            fault_address,
            program_counter,
        );
    }
    let mut state = state()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let Some(page) = state.pages.get(&page_address) else {
        return EXCEPTION_CONTINUE_SEARCH;
    };
    let watcher_ids = page.watchers.iter().copied().collect::<Vec<_>>();
    let writable_protection = page.writable_protection;
    for watcher_id in watcher_ids {
        let Some(range) = state.ranges.get(&watcher_id) else {
            continue;
        };
        if range.retirement.load(Ordering::Acquire) != WATCH_ACTIVE {
            continue;
        }
        range.dirty.store(true, Ordering::Release);
        range.generation.fetch_add(1, Ordering::AcqRel);
        range
            .dirty_pages
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .insert(page_address);
    }
    if set_page_protection(page_address, writable_protection).is_err() {
        return EXCEPTION_CONTINUE_SEARCH;
    }
    if let Some(page) = state.pages.get_mut(&page_address) {
        page.open = true;
    }
    EXCEPTION_CONTINUE_EXECUTION
}

fn page_addresses(address: u64, byte_count: u64) -> Option<impl Iterator<Item = u64>> {
    let start = align_down(address);
    let end = align_up(address.checked_add(byte_count)?)?;
    Some((start..end).step_by(HOST_PAGE_SIZE as usize))
}

fn mark_range_dirty(range: &WatchedRange, address: u64, byte_count: u64) {
    range.dirty.store(true, Ordering::Release);
    range.generation.fetch_add(1, Ordering::AcqRel);
    let Some(pages) = page_addresses(address.max(range.address), byte_count) else {
        return;
    };
    let range_end = range.address + range.byte_count;
    let mut dirty_pages = range
        .dirty_pages
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    for page in pages {
        if page < range_end && page + HOST_PAGE_SIZE > range.address {
            dirty_pages.insert(page);
        }
    }
}

fn open_pages(state: &mut WatchState, address: u64, byte_count: u64) -> io::Result<()> {
    let Some(range_end) = address.checked_add(byte_count).and_then(align_up) else {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "page range overflows",
        ));
    };
    let range_start = align_down(address);
    let mut runs = Vec::<(u64, u64, PAGE_PROTECTION_FLAGS)>::new();
    for (&page_address, page) in state.pages.range(range_start..range_end) {
        if page.open {
            continue;
        }
        if let Some((_, end, protection)) = runs.last_mut()
            && *end == page_address
            && *protection == page.writable_protection
        {
            *end += HOST_PAGE_SIZE;
        } else {
            runs.push((
                page_address,
                page_address + HOST_PAGE_SIZE,
                page.writable_protection,
            ));
        }
    }
    for (start, end, protection) in runs {
        set_range_protection(start, end - start, protection)?;
        for (_, page) in state.pages.range_mut(start..end) {
            page.open = true;
        }
    }
    Ok(())
}

fn remove_range_pages(state: &mut WatchState, range: &WatchedRange) {
    let Some(pages) = page_addresses(range.address, range.byte_count) else {
        return;
    };
    for page_address in pages {
        let remove = if let Some(page) = state.pages.get_mut(&page_address) {
            page.watchers.remove(&range.id);
            page.watchers.is_empty()
        } else {
            false
        };
        if remove {
            if let Some(page) = state.pages.remove(&page_address) {
                if !page.open {
                    let _ = set_page_protection(page_address, page.writable_protection);
                }
            }
        }
    }
}

pub(crate) fn publish_guest_reservation(_address: u64, _byte_count: u64) {}

pub(crate) fn forget_guest_reservation(_address: u64, _byte_count: u64) {}

#[must_use]
pub fn notify_guest_memory_written(address: u64, byte_count: u64) -> bool {
    notify_guest_memory_written_excluding(address, byte_count, &[])
}

#[must_use]
pub fn notify_guest_memory_written_excluding(
    address: u64,
    byte_count: u64,
    excluded: &[&CpuWriteWatch],
) -> bool {
    if byte_count == 0 || address.checked_add(byte_count).is_none() {
        return false;
    }
    let aliases = crate::address_space::guest_write_alias_segments(address, byte_count);
    let mut state = state()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let mut notified = false;
    for range in state.ranges.values() {
        if range.retirement.load(Ordering::Acquire) != WATCH_ACTIVE
            || excluded
                .iter()
                .any(|watch| Arc::ptr_eq(&watch.range, range))
        {
            continue;
        }
        for alias in &aliases {
            if !overlaps(
                (range.address, range.byte_count),
                (alias.address, alias.byte_count),
            ) {
                continue;
            }
            let intersection_start = range.address.max(alias.address);
            let intersection_end =
                (range.address + range.byte_count).min(alias.address + alias.byte_count);
            mark_range_dirty(
                range,
                intersection_start,
                intersection_end - intersection_start,
            );
            notified = true;
        }
    }
    for alias in aliases {
        let _ = open_pages(&mut state, alias.address, alias.byte_count);
    }
    notified
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CpuWriteWatchRetirement {
    Reprotecting,
    Reprotected,
    Replaced,
}

#[must_use]
pub struct CpuWriteWatchReprotection {
    ranges: Vec<Arc<WatchedRange>>,
}

impl Drop for CpuWriteWatchReprotection {
    fn drop(&mut self) {
        for range in &self.ranges {
            let _ = range.retirement.compare_exchange(
                WATCH_REPROTECTING,
                WATCH_REPROTECTED,
                Ordering::AcqRel,
                Ordering::Acquire,
            );
        }
    }
}

#[derive(Debug)]
pub struct CpuWriteWatch {
    range: Arc<WatchedRange>,
}

impl CpuWriteWatch {
    /// # Safety
    ///
    /// The range must remain a live readable and writable mapping until the watch is retired.
    pub unsafe fn new(address: u64, byte_count: u64) -> io::Result<Self> {
        if address == 0 || byte_count == 0 || address.checked_add(byte_count).is_none() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "a CPU write watch requires a nonempty, non-overflowing mapped range",
            ));
        }
        install_exception_handler()?;
        let mut state = state()
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        state.next_id = state
            .next_id
            .checked_add(1)
            .ok_or_else(|| io::Error::other("CPU write-watch identifier space was exhausted"))?;
        let range = Arc::new(WatchedRange {
            id: state.next_id,
            address,
            byte_count,
            dirty: AtomicBool::new(false),
            dirty_pages: Mutex::new(BTreeSet::new()),
            generation: AtomicU64::new(0),
            retirement: AtomicU8::new(WATCH_ACTIVE),
        });
        state.ranges.insert(range.id, Arc::clone(&range));
        let setup = (|| {
            let pages = page_addresses(address, byte_count)
                .expect("validated CPU write-watch range has non-overflowing page bounds");
            let mut new_pages = Vec::new();
            let mut queried_region_end = 0;
            let mut queried_protection = PAGE_PROTECTION_FLAGS::default();
            for page_address in pages {
                if let Some(page) = state.pages.get_mut(&page_address) {
                    page.watchers.insert(range.id);
                    continue;
                }
                if page_address >= queried_region_end {
                    (queried_protection, queried_region_end) = query_page_region(page_address)?;
                }
                let writable_protection = queried_protection;
                let readonly_protection =
                    readonly_protection(writable_protection).ok_or_else(|| {
                        io::Error::new(
                            io::ErrorKind::PermissionDenied,
                            "CPU write watch requires writable guest pages",
                        )
                    })?;
                new_pages.push((page_address, writable_protection, readonly_protection));
            }
            let mut index = 0;
            while index < new_pages.len() {
                let (start, _, protection) = new_pages[index];
                let mut end_index = index + 1;
                while end_index < new_pages.len()
                    && new_pages[end_index].0 == new_pages[end_index - 1].0 + HOST_PAGE_SIZE
                    && new_pages[end_index].2 == protection
                {
                    end_index += 1;
                }
                let end = new_pages[end_index - 1].0 + HOST_PAGE_SIZE;
                set_range_protection(start, end - start, protection)?;
                for &(page_address, writable_protection, readonly_protection) in
                    &new_pages[index..end_index]
                {
                    state.pages.insert(
                        page_address,
                        WatchedPage {
                            watchers: BTreeSet::from([range.id]),
                            writable_protection,
                            readonly_protection,
                            open: false,
                        },
                    );
                }
                index = end_index;
            }
            Ok::<(), io::Error>(())
        })();
        if let Err(error) = setup {
            remove_range_pages(&mut state, &range);
            state.ranges.remove(&range.id);
            return Err(error);
        }
        Ok(Self { range })
    }

    #[must_use]
    pub fn consume_dirty(&self) -> bool {
        self.range.dirty.swap(false, Ordering::AcqRel)
    }

    #[must_use]
    pub fn is_dirty(&self) -> bool {
        self.range.dirty.load(Ordering::Acquire)
    }

    #[must_use]
    pub const fn precise_tracking(&self) -> bool {
        true
    }

    #[must_use]
    pub fn generation(&self) -> u64 {
        self.range.generation.load(Ordering::Acquire)
    }

    #[must_use]
    pub fn dirty_page_ranges(&self) -> Vec<(u64, u64)> {
        let range_end = self.range.address + self.range.byte_count;
        self.range
            .dirty_pages
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .iter()
            .map(|&page| {
                let start = page.max(self.range.address);
                let end = (page + HOST_PAGE_SIZE).min(range_end);
                (start, end - start)
            })
            .collect()
    }

    pub fn rearm(&self) -> io::Result<()> {
        let mut state = state()
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if self.is_retired() || !state.ranges.contains_key(&self.range.id) {
            return Err(io::Error::new(
                io::ErrorKind::NotFound,
                "CPU write watch was retired by a guest mapping change",
            ));
        }
        self.range
            .dirty_pages
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clear();
        let range_start = align_down(self.range.address);
        let range_end = align_up(self.range.address + self.range.byte_count)
            .expect("live CPU write-watch page range remains valid");
        let mut runs = Vec::<(u64, u64, PAGE_PROTECTION_FLAGS)>::new();
        for (&page_address, page) in state.pages.range(range_start..range_end) {
            if !page.open {
                continue;
            }
            if let Some((_, end, protection)) = runs.last_mut()
                && *end == page_address
                && *protection == page.readonly_protection
            {
                *end += HOST_PAGE_SIZE;
            } else {
                runs.push((
                    page_address,
                    page_address + HOST_PAGE_SIZE,
                    page.readonly_protection,
                ));
            }
        }
        for (start, end, protection) in runs {
            set_range_protection(start, end - start, protection)?;
            for (_, page) in state.pages.range_mut(start..end) {
                page.open = false;
            }
        }
        Ok(())
    }

    pub fn with_live_mapping<T>(&self, operation: impl FnOnce() -> T) -> io::Result<T> {
        let mut state = state()
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if self.is_retired() || !state.ranges.contains_key(&self.range.id) {
            return Err(io::Error::new(
                io::ErrorKind::NotFound,
                "CPU write-watch mapping is no longer live",
            ));
        }
        open_pages(&mut state, self.range.address, self.range.byte_count)?;
        Ok(operation())
    }

    #[must_use]
    pub fn is_retired(&self) -> bool {
        self.retirement().is_some()
    }

    #[must_use]
    pub fn retirement(&self) -> Option<CpuWriteWatchRetirement> {
        match self.range.retirement.load(Ordering::Acquire) {
            WATCH_ACTIVE => None,
            WATCH_REPROTECTING => Some(CpuWriteWatchRetirement::Reprotecting),
            WATCH_REPROTECTED => Some(CpuWriteWatchRetirement::Reprotected),
            WATCH_REPLACED => Some(CpuWriteWatchRetirement::Replaced),
            _ => unreachable!("CPU write-watch retirement has an invalid state"),
        }
    }

    pub fn retire_overlapping(address: u64, byte_count: u64) {
        let _ = retire_overlapping(address, byte_count, WATCH_REPLACED);
    }

    pub fn begin_reprotect_overlapping(address: u64, byte_count: u64) -> CpuWriteWatchReprotection {
        CpuWriteWatchReprotection {
            ranges: retire_overlapping(address, byte_count, WATCH_REPROTECTING),
        }
    }

    pub fn begin_alias_change_overlapping(
        address: u64,
        byte_count: u64,
    ) -> CpuWriteWatchReprotection {
        let ranges = retire_overlapping(address, byte_count, WATCH_REPROTECTING);
        for range in &ranges {
            range.dirty.store(true, Ordering::Release);
        }
        CpuWriteWatchReprotection { ranges }
    }

    #[must_use]
    pub fn range(&self) -> (u64, u64) {
        (self.range.address, self.range.byte_count)
    }
}

fn retire_overlapping(address: u64, byte_count: u64, reason: u8) -> Vec<Arc<WatchedRange>> {
    if byte_count == 0 || address.checked_add(byte_count).is_none() {
        return Vec::new();
    }
    let mut state = state()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let ranges = state
        .ranges
        .values()
        .filter(|range| overlaps((range.address, range.byte_count), (address, byte_count)))
        .filter_map(|range| {
            range
                .retirement
                .compare_exchange(WATCH_ACTIVE, reason, Ordering::AcqRel, Ordering::Acquire)
                .is_ok()
                .then(|| Arc::clone(range))
        })
        .collect::<Vec<_>>();
    for range in &ranges {
        remove_range_pages(&mut state, range);
    }
    ranges
}

impl Drop for CpuWriteWatch {
    fn drop(&mut self) {
        let mut state = state()
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if self.range.retirement.load(Ordering::Acquire) == WATCH_ACTIVE {
            remove_range_pages(&mut state, &self.range);
        }
        state.ranges.remove(&self.range.id);
        self.range
            .retirement
            .store(WATCH_REPLACED, Ordering::Release);
    }
}
