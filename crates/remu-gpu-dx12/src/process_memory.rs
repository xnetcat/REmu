// SPDX-License-Identifier: GPL-2.0-or-later

//! Fail-closed access to guest ranges mapped in the current Windows process.

use std::ffi::c_void;

pub(super) fn read_process_memory(address: u64, destination: &mut [u8]) -> Result<(), String> {
    if destination.is_empty() {
        return Ok(());
    }
    let pointer = usize::try_from(address).map_err(|_| "guest address does not fit usize")?;
    address
        .checked_add(u64::try_from(destination.len()).map_err(|_| "read size does not fit u64")?)
        .ok_or("guest read range overflows")?;
    let mut read = 0_usize;
    // SAFETY: ReadProcessMemory validates the complete source range in this process and writes
    // only into the provided Rust slice.
    let result = unsafe {
        ReadProcessMemory(
            GetCurrentProcess(),
            pointer as *const c_void,
            destination.as_mut_ptr().cast::<c_void>(),
            destination.len(),
            &raw mut read,
        )
    };
    if result == 0 || read != destination.len() {
        Err(format!(
            "ReadProcessMemory failed at {address:#x} for {} bytes: {}",
            destination.len(),
            std::io::Error::last_os_error()
        ))
    } else {
        Ok(())
    }
}

pub(super) fn write_process_memory(address: u64, source: &[u8]) -> Result<(), String> {
    if source.is_empty() {
        return Ok(());
    }
    let pointer = usize::try_from(address).map_err(|_| "guest address does not fit usize")?;
    address
        .checked_add(u64::try_from(source.len()).map_err(|_| "write size does not fit u64")?)
        .ok_or("guest write range overflows")?;
    let _ = remu_memory::notify_guest_memory_written(
        address,
        u64::try_from(source.len()).map_err(|_| "write size does not fit u64")?,
    );
    let mut written = 0_usize;
    // SAFETY: WriteProcessMemory validates the complete destination range in this process and
    // reads only from the provided Rust slice.
    let result = unsafe {
        WriteProcessMemory(
            GetCurrentProcess(),
            pointer as *mut c_void,
            source.as_ptr().cast::<c_void>(),
            source.len(),
            &raw mut written,
        )
    };
    if result == 0 || written != source.len() {
        Err(format!(
            "WriteProcessMemory failed at {address:#x} for {} bytes: {}",
            source.len(),
            std::io::Error::last_os_error()
        ))
    } else {
        Ok(())
    }
}

#[link(name = "kernel32")]
unsafe extern "system" {
    fn GetCurrentProcess() -> *mut c_void;
    fn ReadProcessMemory(
        process: *mut c_void,
        base_address: *const c_void,
        buffer: *mut c_void,
        size: usize,
        bytes_read: *mut usize,
    ) -> i32;
    fn WriteProcessMemory(
        process: *mut c_void,
        base_address: *mut c_void,
        buffer: *const c_void,
        size: usize,
        bytes_written: *mut usize,
    ) -> i32;
}

#[cfg(test)]
mod tests {
    use super::{read_process_memory, write_process_memory};

    #[test]
    fn validates_complete_ranges_and_round_trips_current_process_memory() {
        let mut storage = [1_u8, 2, 3, 4];
        let address = storage.as_mut_ptr() as u64;
        let mut readback = [0_u8; 4];
        read_process_memory(address, &mut readback).expect("read mapped process bytes");
        assert_eq!(readback, storage);
        write_process_memory(address + 1, &[9, 8]).expect("write mapped process bytes");
        assert_eq!(storage, [1, 9, 8, 4]);
        assert!(read_process_memory(1, &mut readback).is_err());
    }
}
