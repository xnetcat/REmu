#[cfg(target_os = "macos")]
use remu_core::GuestAddress;

use crate::{
    GuestMspace, ProbeDispatchState, ProbeImportReturn, current_guest_thread_pointer, guest_calloc,
    low_i32, read_guest_cstring, write_guest_u16, write_guest_u32, write_guest_u64,
};

pub(crate) fn guest_ctype_table(state: &mut ProbeDispatchState) -> u64 {
    if state.ctype_table == 0 {
        // Sony's libc follows the Dinkumware mask layout and indexes this table over
        // [-128, 255]. Return the c == 0 entry, leaving 128 valid entries before it.
        let storage = guest_calloc(384, 2);
        if storage == 0 {
            return 0;
        }
        state.ctype_table = storage + 128 * 2;
        for character in 0_u16..=0x7f {
            write_guest_u16(
                state.ctype_table + u64::from(character) * 2,
                guest_ctype_flags(character),
            );
        }
    }
    state.ctype_table
}

pub(crate) fn guest_tolower_table(state: &mut ProbeDispatchState) -> u64 {
    if state.tolower_table == 0 {
        // Dinkumware's narrow-character case tables contain signed 16-bit values and cover
        // [-128, 255]. The returned pointer addresses the c == 0 element.
        let storage = guest_calloc(384, 2);
        if storage == 0 {
            return 0;
        }
        state.tolower_table = storage + 128 * 2;
        for index in 0_u16..384 {
            let character = i32::from(index) - 128;
            let mapped = if (i32::from(b'A')..=i32::from(b'Z')).contains(&character) {
                character - i32::from(b'A') + i32::from(b'a')
            } else {
                character
            };
            write_guest_u16(
                storage + u64::from(index) * 2,
                i16::try_from(mapped)
                    .expect("case mapping fits i16")
                    .cast_unsigned(),
            );
        }
    }
    state.tolower_table
}

pub(crate) fn guest_toupper_table(state: &mut ProbeDispatchState) -> u64 {
    if state.toupper_table == 0 {
        // Dinkumware indexes the signed-char range as well as 0..=255. Each entry is a 16-bit
        // character value and the exported pointer addresses the c == 0 element.
        let storage = guest_calloc(384, 2);
        if storage == 0 {
            return 0;
        }
        state.toupper_table = storage + 128 * 2;
        for index in 0_u16..384 {
            let character = i32::from(index) - 128;
            let mapped = if (i32::from(b'a')..=i32::from(b'z')).contains(&character) {
                character - i32::from(b'a') + i32::from(b'A')
            } else {
                character
            };
            write_guest_u16(
                storage + u64::from(index) * 2,
                i16::try_from(mapped)
                    .expect("case mapping fits i16")
                    .cast_unsigned(),
            );
        }
    }
    state.toupper_table
}

fn guest_ctype_flags(character: u16) -> u16 {
    const XDIGIT: u16 = 0x001;
    const UPPER: u16 = 0x002;
    const SPACE: u16 = 0x004;
    const PUNCT: u16 = 0x008;
    const LOWER: u16 = 0x010;
    const DIGIT: u16 = 0x020;
    const CONTROL_SPACE: u16 = 0x040;
    const CONTROL: u16 = 0x080;
    const BLANK: u16 = 0x400;

    let upper = (u16::from(b'A')..=u16::from(b'Z')).contains(&character);
    let lower = (u16::from(b'a')..=u16::from(b'z')).contains(&character);
    let digit = (u16::from(b'0')..=u16::from(b'9')).contains(&character);
    let hexadecimal = digit
        || (u16::from(b'A')..=u16::from(b'F')).contains(&character)
        || (u16::from(b'a')..=u16::from(b'f')).contains(&character);
    let alphanumeric = upper || lower || digit;
    let control_space = (0x09..=0x0d).contains(&character);
    let control = character <= 0x08 || (0x0e..=0x1f).contains(&character) || character == 0x7f;
    let printable = (0x20..=0x7e).contains(&character);

    let mut flags = 0;
    if upper {
        flags |= UPPER;
    }
    if lower {
        flags |= LOWER;
    }
    if digit {
        flags |= DIGIT;
    }
    if hexadecimal {
        flags |= XDIGIT;
    }
    if character == u16::from(b' ') {
        flags |= SPACE | BLANK;
    }
    if character == u16::from(b'\t') {
        flags |= BLANK;
    }
    if control_space {
        flags |= CONTROL_SPACE;
    }
    if printable && !alphanumeric && character != u16::from(b' ') {
        flags |= PUNCT;
    }
    if control {
        flags |= CONTROL;
    }
    flags
}

pub(crate) fn guest_errno_address() -> u64 {
    let thread_pointer = current_guest_thread_pointer();
    if thread_pointer == 0 {
        0
    } else {
        thread_pointer.checked_add(0x40).unwrap_or(0)
    }
}

pub(crate) fn guest_net_errno_address() -> u64 {
    // libSceNet exposes a distinct thread-local errno cell. The guest TCB reservation is private
    // to each emulated pthread; libc errno occupies +0x40, so keep the network value beside it.
    let thread_pointer = current_guest_thread_pointer();
    if thread_pointer == 0 {
        0
    } else {
        thread_pointer.checked_add(0x44).unwrap_or(0)
    }
}

pub(crate) fn dispatch_guest_numeric_parser(nid: &str, arguments: [u64; 6]) -> ProbeImportReturn {
    if arguments[0] == 0 {
        return ProbeImportReturn::integer(0);
    }
    set_host_errno(0);
    if guest_errno_address() != 0 {
        write_guest_u32(guest_errno_address(), 0);
    }
    let end_pointer = arguments[1] as *mut *mut libc::c_char;
    let result = match nid {
        // Dinkumware _Stoul uses the target's unsigned-long parser and the same three-argument
        // signature as strtoul. Gen5 uses a 64-bit unsigned long; Windows LLP64 hosts use
        // `strtoull` so the guest still observes a 64-bit parse width.
        "zlfEH8FmyUA" => {
            // SAFETY: The guest supplies a NUL-terminated string and optional writable end pointer.
            ProbeImportReturn::integer(unsafe {
                #[cfg(windows)]
                {
                    libc::strtoull(
                        arguments[0] as *const libc::c_char,
                        end_pointer,
                        low_i32(arguments[2]),
                    )
                }
                #[cfg(not(windows))]
                {
                    libc::strtoul(
                        arguments[0] as *const libc::c_char,
                        end_pointer,
                        low_i32(arguments[2]),
                    ) as u64
                }
            })
        }
        // libc strtoull
        "5OqszGpy7Mg" => {
            // SAFETY: The guest supplies a NUL-terminated string and optional writable end pointer.
            ProbeImportReturn::integer(unsafe {
                libc::strtoull(
                    arguments[0] as *const libc::c_char,
                    end_pointer,
                    low_i32(arguments[2]),
                )
            })
        }
        // libc strtoll
        "VOBg+iNwB-4" => {
            // SAFETY: The guest supplies a NUL-terminated string and optional writable end pointer.
            ProbeImportReturn::integer(unsafe {
                libc::strtoll(
                    arguments[0] as *const libc::c_char,
                    end_pointer,
                    low_i32(arguments[2]),
                )
                .cast_unsigned()
            })
        }
        // libc strtod. ProbeImportReturn's mixed INTEGER/SSE ABI returns this through XMM0 while
        // preserving RAX for every ordinary import.
        "2vDqwBlpF-o" => {
            // SAFETY: The guest supplies a NUL-terminated string and optional writable end pointer.
            ProbeImportReturn::float(unsafe {
                libc::strtod(arguments[0] as *const libc::c_char, end_pointer)
            })
        }
        _ => ProbeImportReturn::integer(0),
    };
    let guest_errno = guest_errno_address();
    if guest_errno != 0 {
        write_guest_u32(guest_errno, host_errno().cast_unsigned());
    }
    result
}

#[cfg(target_os = "macos")]
fn set_host_errno(value: i32) {
    // SAFETY: __error returns this host thread's writable errno cell.
    unsafe {
        *libc::__error() = value;
    }
}

#[cfg(target_os = "macos")]
fn host_errno() -> i32 {
    // SAFETY: __error returns this host thread's live errno cell.
    unsafe { *libc::__error() }
}

#[cfg(target_os = "linux")]
fn set_host_errno(value: i32) {
    // SAFETY: __errno_location returns this host thread's writable errno cell.
    unsafe {
        *libc::__errno_location() = value;
    }
}

#[cfg(target_os = "linux")]
fn host_errno() -> i32 {
    // SAFETY: __errno_location returns this host thread's live errno cell.
    unsafe { *libc::__errno_location() }
}

#[cfg(not(any(target_os = "linux", target_os = "macos")))]
fn set_host_errno(_value: i32) {}

#[cfg(not(any(target_os = "linux", target_os = "macos")))]
fn host_errno() -> i32 {
    0
}

pub(crate) fn guest_memset(destination: u64, value: u64, length: u64) -> u64 {
    if destination == 0 || length == 0 {
        return destination;
    }
    let Ok(length) = usize::try_from(length) else {
        return 0;
    };
    let fill = value.to_le_bytes()[0];
    // SAFETY: The call originates in guest code and supplies a live destination mapping.
    unsafe {
        std::ptr::write_bytes(destination as *mut u8, fill, length);
    }
    remu_memory::record_guest_write(
        remu_memory::GuestWriteSite::GuestMemset,
        destination,
        length as u64,
        &[fill; 8][..length.min(8)],
        0,
    );
    destination
}

pub(crate) fn guest_memcpy(destination: u64, source: u64, length: u64) -> u64 {
    if length == 0 {
        return destination;
    }
    if destination == 0 || source == 0 {
        return 0;
    }
    let Ok(length) = usize::try_from(length) else {
        return 0;
    };
    // SAFETY: memcpy's guest ABI contract requires two valid non-overlapping ranges.
    unsafe {
        std::ptr::copy_nonoverlapping(source as *const u8, destination as *mut u8, length);
    }
    journal_guest_copy(destination, length);
    destination
}

/// Journals one HLE'd copy into guest memory. Sub-16-byte copies are the emulator's hottest
/// call and cannot carry a bulk fill, so they stay out of the ring; the uniformity verdict is
/// drawn from the leading page.
fn journal_guest_copy(destination: u64, length: usize) {
    if length >= 16 && remu_memory::guest_write_journal_enabled() {
        // SAFETY: the copy that just completed wrote exactly these destination bytes.
        let sample =
            unsafe { std::slice::from_raw_parts(destination as *const u8, length.min(4096)) };
        remu_memory::record_guest_write(
            remu_memory::GuestWriteSite::GuestMemcpy,
            destination,
            length as u64,
            sample,
            0,
        );
    }
}

pub(crate) fn guest_memmove(destination: u64, source: u64, length: u64) -> u64 {
    if length == 0 {
        return destination;
    }
    if destination == 0 || source == 0 {
        return 0;
    }
    let Ok(length) = usize::try_from(length) else {
        return 0;
    };
    // SAFETY: memmove's guest ABI contract supplies two valid ranges; `copy` permits overlap.
    unsafe {
        std::ptr::copy(source as *const u8, destination as *mut u8, length);
    }
    journal_guest_copy(destination, length);
    destination
}

pub(crate) fn guest_mspace_create(arguments: [u64; 6], state: &mut ProbeDispatchState) -> u64 {
    let Some(name) = read_guest_cstring(arguments[0]) else {
        return 0;
    };
    let (base, capacity) = (arguments[1], arguments[2]);
    if base.checked_add(capacity).is_none() {
        return 0;
    }
    if base == 0 || capacity == 0 {
        return 0;
    }
    #[cfg(target_os = "macos")]
    {
        let end = base + capacity;
        let Some(mapping) = state
            .kernel_memory
            .as_ref()
            .and_then(|memory| memory.query(GuestAddress::new(base), false))
        else {
            return 0;
        };
        let mapping_end = mapping.address.get().saturating_add(mapping.size);
        if base < mapping.address.get() || end > mapping_end {
            return 0;
        }
    }
    let Some(mspace) = GuestMspace::new(name, base, capacity) else {
        return 0;
    };
    if state.guest_mspaces.contains_key(&base) {
        return 0;
    }
    state.guest_mspaces.insert(base, mspace);
    if let Some(mspace) = state.guest_mspaces.get(&base) {
        eprintln!(
            "[REMu][MSPACE] created name={:?} base={:#x} end={:#x}",
            mspace.name, mspace.base, mspace.end
        );
    }
    base
}

pub(crate) fn guest_mspace_malloc(arguments: [u64; 6], state: &mut ProbeDispatchState) -> u64 {
    state
        .guest_mspaces
        .get_mut(&arguments[0])
        .and_then(|mspace| mspace.allocate(arguments[1], GuestMspace::ALLOCATION_ALIGNMENT))
        .unwrap_or(0)
}

pub(crate) fn guest_mspace_calloc(arguments: [u64; 6], state: &mut ProbeDispatchState) -> u64 {
    let Some(size) = arguments[1].checked_mul(arguments[2]) else {
        return 0;
    };
    let address = state
        .guest_mspaces
        .get_mut(&arguments[0])
        .and_then(|mspace| mspace.allocate(size, GuestMspace::ALLOCATION_ALIGNMENT))
        .unwrap_or(0);
    if address != 0 {
        let Ok(size) = usize::try_from(size) else {
            return 0;
        };
        // SAFETY: The mspace allocator returned a live block of at least `size` bytes.
        unsafe { std::ptr::write_bytes(address as *mut u8, 0, size) };
        remu_memory::record_guest_write(
            remu_memory::GuestWriteSite::MspaceZero,
            address,
            size as u64,
            &[0; 8][..size.min(8)],
            0,
        );
    }
    address
}

pub(crate) fn guest_mspace_free(arguments: [u64; 6], state: &mut ProbeDispatchState) -> u64 {
    if let Some(mspace) = state.guest_mspaces.get_mut(&arguments[0]) {
        mspace.free(arguments[1]);
    }
    0
}

pub(crate) fn guest_mspace_destroy(arguments: [u64; 6], state: &mut ProbeDispatchState) -> u64 {
    if state.guest_mspaces.remove(&arguments[0]).is_some() {
        0
    } else {
        u64::MAX
    }
}

pub(crate) fn guest_mspace_realloc(arguments: [u64; 6], state: &mut ProbeDispatchState) -> u64 {
    state
        .guest_mspaces
        .get_mut(&arguments[0])
        .and_then(|mspace| mspace.realloc(arguments[1], arguments[2]))
        .unwrap_or(0)
}

pub(crate) fn guest_mspace_memalign(arguments: [u64; 6], state: &mut ProbeDispatchState) -> u64 {
    state
        .guest_mspaces
        .get_mut(&arguments[0])
        .and_then(|mspace| mspace.allocate(arguments[2], arguments[1]))
        .unwrap_or(0)
}

pub(crate) fn guest_mspace_posix_memalign(
    arguments: [u64; 6],
    state: &mut ProbeDispatchState,
) -> u64 {
    let (handle, output, alignment, size) =
        (arguments[0], arguments[1], arguments[2], arguments[3]);
    if output == 0
        || !alignment.is_power_of_two()
        || alignment < u64::try_from(std::mem::size_of::<usize>()).unwrap_or(8)
    {
        return u64::from(libc::EINVAL.cast_unsigned());
    }
    let Some(address) = state
        .guest_mspaces
        .get_mut(&handle)
        .and_then(|mspace| mspace.allocate(size, alignment))
    else {
        return u64::from(libc::ENOMEM.cast_unsigned());
    };
    write_guest_u64(output, address);
    0
}

pub(crate) fn guest_mspace_usable_size(arguments: [u64; 6], state: &ProbeDispatchState) -> u64 {
    state
        .guest_mspaces
        .get(&arguments[0])
        .map_or(0, |mspace| mspace.usable_size(arguments[1]))
}

pub(crate) fn guest_mspace_is_heap_empty(arguments: [u64; 6], state: &ProbeDispatchState) -> u64 {
    state
        .guest_mspaces
        .get(&arguments[0])
        .map_or(0, |mspace| u64::from(mspace.allocations.is_empty()))
}
